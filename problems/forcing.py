from __future__ import annotations

import copy
from typing import Any

import numpy as np

from problems.base import OODAxis, ProblemDims, ProblemSpec, empty_forcing_seq
from src.physics.boundary_forcing import (
    DT_PULSE_FRAC_HI,
    DT_PULSE_FRAC_LO,
    GAUSS_SIGMA_RANGE,
    PATCH_W_RANGE,
    PULSE_AMP_RANGE,
    SIN_FREQ_RANGE,
    SPATIAL_FAMILIES,
    TEMPORAL_FAMILIES,
    TRIANGLE_ELL_RANGE,
    TEMPORAL_SAMPLERS,
    SPATIAL_BUILDERS,
    SPATIAL_SAMPLERS,
    SIN_AMP_RANGE,
    FORCING_BINS,
    default_ramp_seconds,
    integrate_temporal_bins_ramped_signed,
    ramped_temporal,
    sample_spatial_family,
    sample_temporal_family,
    build_qL,
    build_qL_integral,
)
from src.physics.fv_solver_2d import FVSolver2D, Layer2D

# ----- representation constants (canonical home for the forcing benchmark) -----

RC_RANGE = (0.05, 1.0)
T_EPS = 1e-6

SPATIAL_FAMILY_ORDER = ("uniform", "patch", "gaussian", "triangle")

_BASE_DIM = 2
_SPATIAL_ONEHOT_DIM = len(SPATIAL_FAMILY_ORDER)
_SPATIAL_PARAM_DIM = 4
COND_STATIC_DIM = (
    _BASE_DIM + _SPATIAL_ONEHOT_DIM + _SPATIAL_PARAM_DIM
)  # 10 (forcing-agnostic: no temporal one-hot, no forcing summary)

# Spatial-descriptor conditioning ablation slices (single source of truth for the
# mask helper and the tests). cond layout is [t_bar_norm, R_c_norm,
# spatial_onehot(4), spatial_params(4)], so the one-hot family label is [2:6] and
# the full spatial descriptor (family label + continuous params) is [2:10].
FORCING_FAMILY_SLICE = slice(_BASE_DIM, _BASE_DIM + _SPATIAL_ONEHOT_DIM)  # slice(2, 6)
FORCING_SPATIAL_DESCRIPTOR_SLICE = slice(_BASE_DIM, COND_STATIC_DIM)  # slice(2, 10)

# temporal_encoder mode: 128 samples x 2 tokens [r_m, a_m / A_ref].
FORCING_TEMPORAL_SAMPLES = 128
FORCING_TEMPORAL_TOKEN_DIM = 2
A_AMP_REF = 300.0
INTERFACE_X = 0.5
K_LEFT = 2.0
K_RIGHT = 1.0
RHO_LEFT = 1.0
RHO_RIGHT = 1.0
CP_LEFT = 1.0
CP_RIGHT = 1.0
T_RIGHT = 300.0

# Spatial channels per representation. temporal_encoder mode lifts the lean
# [T_tilde, x, y, s_y] base; bins mode appends the FORCING_BINS integral Q-bins.
SPATIAL_CHANNELS_TEMPORAL = 4
SPATIAL_CHANNELS_BINS = SPATIAL_CHANNELS_TEMPORAL + FORCING_BINS  # 20

# s_y is the 4th spatial channel (index 3) in both representations.
S_Y_CHANNEL = 3

_LOG_SIGMA_LO = float(np.log(GAUSS_SIGMA_RANGE[0]))
_LOG_SIGMA_HI = float(np.log(GAUSS_SIGMA_RANGE[1]))


# ----- conditioning / forcing-sequence builders -----

def _normalize_y_position(y: float, y_bounds: tuple[float, float]) -> float:
    y_lo, y_hi = map(float, y_bounds)
    return (float(y) - y_lo) / (y_hi - y_lo)


def _normalize_y_length(length: float, y_bounds: tuple[float, float]) -> float:
    y_lo, y_hi = map(float, y_bounds)
    return float(length) / (y_hi - y_lo)


def build_cond_vector(t_bar_norm: float, R_c: float,
                      spatial_family: str, spatial_params: dict,
                      y_bounds: tuple[float, float] = (0.0, 1.0),
                      *,
                      allow_unknown_spatial_family: bool = False) -> np.ndarray:
    """Assemble the 10-dim forcing-agnostic static conditioning vector.

    Layout: [t_bar_norm, R_c_norm, spatial_onehot(4),
    spatial_params(4)]. Carries no temporal-family identity or forcing
    summary; in temporal_encoder mode the only temporal forcing information
    lives in forcing_seq, in bins mode it lives in the Q-bin spatial channels.

    An unknown spatial_family raises by default so a typo cannot silently
    become an all-zero descriptor. allow_unknown_spatial_family=True is only
    honest when the [2:10] slice is masked out anyway, i.e. under
    spatial_conditioning='spatial_field_only'.
    """
    R_c_norm = (R_c - RC_RANGE[0]) / (RC_RANGE[1] - RC_RANGE[0])
    base = np.array([t_bar_norm, R_c_norm], dtype=np.float32)

    spatial_oh = np.zeros(_SPATIAL_ONEHOT_DIM, dtype=np.float32)
    if spatial_family in SPATIAL_FAMILY_ORDER:
        spatial_oh[SPATIAL_FAMILY_ORDER.index(spatial_family)] = 1.0
    elif not allow_unknown_spatial_family:
        raise ValueError(
            f"unknown spatial_family {spatial_family!r}; expected one of "
            f"{list(SPATIAL_FAMILY_ORDER)}. Pass allow_unknown_spatial_family=True "
            "only when the spatial descriptor slice is masked out anyway "
            "(spatial_conditioning='spatial_field_only')."
        )
    y_c_norm = w_norm = sigma_y_norm = ell_norm = 0.0
    if spatial_family == "patch":
        y_c_norm = _normalize_y_position(spatial_params["y_c"], y_bounds)
        w_frac = _normalize_y_length(spatial_params["w"], y_bounds)
        w_norm = (w_frac - PATCH_W_RANGE[0]) / (PATCH_W_RANGE[1] - PATCH_W_RANGE[0])
    elif spatial_family == "gaussian":
        y_c_norm = _normalize_y_position(spatial_params["y_c"], y_bounds)
        sigma_frac = _normalize_y_length(spatial_params["sigma_y"], y_bounds)
        sigma_y_norm = (np.log(sigma_frac) - _LOG_SIGMA_LO) / (_LOG_SIGMA_HI - _LOG_SIGMA_LO)
    elif spatial_family == "triangle":
        y_c_norm = _normalize_y_position(spatial_params["y_c"], y_bounds)
        ell_frac = _normalize_y_length(spatial_params["ell"], y_bounds)
        ell_norm = (ell_frac - TRIANGLE_ELL_RANGE[0]) / (TRIANGLE_ELL_RANGE[1] - TRIANGLE_ELL_RANGE[0])
    spatial_p = np.array([y_c_norm, w_norm, sigma_y_norm, ell_norm], dtype=np.float32)

    return np.concatenate([base, spatial_oh, spatial_p]).astype(np.float32)


def _sample_a(q, t_s: float, t_j: float, M: int) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a(t) at M uniform samples spanning [t_s, t_j] (both endpoints)."""
    t_bar = float(t_j) - float(t_s)
    r = np.linspace(0.0, 1.0, int(M), dtype=np.float32)
    t_samples = (float(t_s) + r * t_bar).astype(np.float32)

    try:
        a_m = np.asarray(q(t_samples), dtype=np.float32)
    except Exception:
        a_m = np.array([q(float(tk)) for tk in t_samples], dtype=np.float32)
    if a_m.shape == ():
        a_m = np.full_like(t_samples, float(a_m), dtype=np.float32)
    elif a_m.shape != t_samples.shape:
        a_m = np.array([q(float(tk)) for tk in t_samples], dtype=np.float32)
    return t_samples, a_m


def _forcing_seq_2tok_from_samples(
    t_samples: np.ndarray,
    a_m: np.ndarray,
    *,
    A_amp_ref: float,
) -> np.ndarray:
    """temporal_encoder forcing sequence: (M, 2) tokens [r_m, a_m / A_ref].

    r_m is the normalized sample position in [0, 1] over [t_s, t_j]; the second
    token is the amplitude sample scaled by the reference. Shared by every
    benchmark's temporal_encoder representation (forcing, source, interfaces).
    """
    r = np.linspace(0.0, 1.0, t_samples.shape[0], dtype=np.float32)
    tok0 = r
    tok1 = (a_m / np.float32(A_amp_ref)).astype(np.float32)
    return np.stack([tok0, tok1], axis=-1).astype(np.float32)


def _generate_lhs_R_c(num_sims: int, seed: int = 0) -> np.ndarray:
    """LHS over the single material parameter R_c (matches generate_dataset).

    Scaling uses float32 bounds (not Python floats) so the float32 stream is
    bit-identical to generate_dataset.generate_lhs_samples.
    """
    lower_bounds = np.array([RC_RANGE[0]], dtype=np.float32)
    upper_bounds = np.array([RC_RANGE[1]], dtype=np.float32)
    rng = np.random.default_rng(seed)
    edges = np.linspace(0.0, 1.0, num_sims + 1, dtype=np.float64)
    widths = edges[1:] - edges[:-1]
    samples_unit = edges[:-1, None] + widths[:, None] * rng.random((num_sims, 1))
    rng.shuffle(samples_unit[:, 0])
    samples_scaled = lower_bounds + samples_unit * (upper_bounds - lower_bounds)
    samples_scaled = samples_scaled.astype(np.float32)
    return samples_scaled[:, 0]


# ----- OOD CRN helpers (pulse_count exceeds NP_MAX, so it needs a pool) -----

def _draw_pulse_pool(rng: np.random.Generator, dt: float, t_final: float,
                     n_pulses: int) -> dict:
    """Draw an unsorted pool of ``n_pulses`` in-distribution pulses.

    ``pulse_count`` sweeps ``Np`` past ``NP_MAX``, so a single pool is drawn once
    per CRN repeat and the first ``Np`` pulses are taken for each swept value,
    yielding nested paired pulse trains across the sweep. Per-pulse draws reuse
    the ``sample_pulse_train_params`` formulas (amplitude, log-uniform width,
    uniform start).
    """
    dt_lo = DT_PULSE_FRAC_LO * dt
    dt_hi = DT_PULSE_FRAC_HI * t_final
    A_pool, t_pool, dt_pool = [], [], []
    for _ in range(int(n_pulses)):
        A_pool.append(float(rng.uniform(*PULSE_AMP_RANGE)))
        u = rng.uniform(0.0, 1.0)
        dtn = float(dt_lo * (dt_hi / dt_lo) ** u)
        dt_pool.append(dtn)
        t_pool.append(float(rng.uniform(0.0, t_final - dtn)))
    return {"A_pool": A_pool, "t_pool": t_pool, "dt_pool": dt_pool}


def _slice_pulse_pool(pool: dict, Np: int) -> dict:
    """Take the first ``Np`` pulses from a pool and sort them by start time.

    Mirrors ``sample_pulse_train_params``'s output schema so the result passes
    straight into ``build_qL``.
    """
    Np = int(Np)
    A = pool["A_pool"][:Np]
    t = pool["t_pool"][:Np]
    dtn = pool["dt_pool"][:Np]
    order = np.argsort(t)
    return {
        "Np": Np,
        "A_list": [float(A[i]) for i in order],
        "t_list": [float(t[i]) for i in order],
        "dt_list": [float(dtn[i]) for i in order],
    }


class ForcingProblem(ProblemSpec):
    """Separable boundary flux q_L(y, t) = a(t) * s(y) benchmark.

    Two representations over the same trajectories/sim_params:

    - ``temporal_encoder``: 4 spatial channels [T_tilde, x, y, s_y], 10
      forcing-agnostic static dims, a (128, 2) forcing_seq [r_m, a_m / A_ref],
      temporal encoder on with time-augmented spatial injection.
    - ``bins``: 20 spatial channels [T_tilde, x, y, s_y, Q_y_bin_0..15], same
      10 static dims, an empty forcing_seq, temporal encoder off.

    Interface fixed at x = 0.5.
    """

    name = "forcing"

    family_cond_slice = FORCING_FAMILY_SLICE
    spatial_descriptor_cond_slice = FORCING_SPATIAL_DESCRIPTOR_SLICE

    def __init__(self, representation: str = "temporal_encoder"):
        self.representation = representation
        if representation == "temporal_encoder":
            self.dims = ProblemDims(
                in_channels=SPATIAL_CHANNELS_TEMPORAL,
                cond_static_dim=COND_STATIC_DIM,
                has_forcing_seq=True,
                temporal_token_dim=FORCING_TEMPORAL_TOKEN_DIM,
                t_stats_dim=2,
                use_temporal_encoder=True,
                s_y_channel=S_Y_CHANNEL,
                use_forcing_time_aug=True,
                forcing_extender_rc_cond_index=1,
            )
        elif representation == "bins":
            self.dims = ProblemDims(
                in_channels=SPATIAL_CHANNELS_BINS,
                cond_static_dim=COND_STATIC_DIM,
                has_forcing_seq=False,
                temporal_token_dim=FORCING_TEMPORAL_TOKEN_DIM,
                t_stats_dim=2,
                use_temporal_encoder=False,
                s_y_channel=S_Y_CHANNEL,
                use_forcing_time_aug=False,
            )
        else:
            raise ValueError(
                f"Unknown representation {representation!r} for benchmark {self.name!r}."
            )

    # ---- data generation ----

    def sample_sim_params(
        self,
        rng: np.random.Generator,
        rng_profile: np.random.Generator,
        grids: dict[str, np.ndarray],
        time_cfg: dict[str, Any],
    ) -> list[dict]:
        X = grids["X"]
        y_grid = grids.get("y_grid")
        if y_grid is None:
            Y = grids["Y"]
            c, d = float(np.min(Y)), float(np.max(Y))
        else:
            c, d = float(y_grid[0]), float(y_grid[-1])
        num_sims = int(time_cfg["num_sims"])
        dt = float(time_cfg["dt"])
        t_final = float(time_cfg["t_final"])
        lhs_seed = int(time_cfg.get("lhs_seed", 0))
        temporal_window = dict(
            t_on=float(time_cfg.get("t_on", 0.0)),
            t_off=float(time_cfg.get("t_off", 0.2)),
            phase=float(time_cfg.get("phase", 0.0)),
            tukey_alpha=float(time_cfg.get("tukey_alpha", 0.5)),
        )

        R_c_values = _generate_lhs_R_c(num_sims, seed=lhs_seed)

        sim_params = []
        for i in range(num_sims):
            R_c = float(R_c_values[i])

            ic_family = "uniform_2d"
            ic_params = {"T0_offset": 0.0}
            T0 = np.full(X.shape, 300.0, dtype=np.float32)

            temporal_family = sample_temporal_family(rng_profile)
            temporal_params = TEMPORAL_SAMPLERS[temporal_family](
                rng_profile, dt=dt, t_final=t_final, **temporal_window
            )

            spatial_family = sample_spatial_family(rng_profile)
            spatial_params = SPATIAL_SAMPLERS[spatial_family](rng_profile, c=c, d=d)

            sim_params.append({
                "R_c": R_c,
                "interface_x": INTERFACE_X,
                "T0": T0,
                "ic_family": ic_family,
                "ic_params": ic_params,
                "temporal_family": temporal_family,
                "temporal_params": temporal_params,
                "spatial_family": spatial_family,
                "spatial_params": spatial_params,
            })

        return sim_params

    def physics_parameters(
        self, a: float = 0.0, b: float = 1.0,
        c: float = 0.0, d: float = 1.0,
    ) -> dict[str, Any]:
        if not all(np.isclose(float(value), target) for value, target in (
            (a, 0.0), (b, 1.0), (c, 0.0), (d, 1.0),
        )):
            raise ValueError("forcing physics is defined on the fixed domain [0, 1]^2")
        return {
            "a": 0.0,
            "b": 1.0,
            "c": 0.0,
            "d": 1.0,
            "interface_x": INTERFACE_X,
            "k_left": K_LEFT,
            "k_right": K_RIGHT,
            "rho_left": RHO_LEFT,
            "rho_right": RHO_RIGHT,
            "cp_left": CP_LEFT,
            "cp_right": CP_RIGHT,
            "T_right": T_RIGHT,
            "layers": [
                Layer2D(
                    x_left=0.0, x_right=INTERFACE_X,
                    rho=RHO_LEFT, cp=CP_LEFT, k=K_LEFT,
                ),
                Layer2D(
                    x_left=INTERFACE_X, x_right=1.0,
                    rho=RHO_RIGHT, cp=CP_RIGHT, k=K_RIGHT,
                ),
            ],
        }

    def configure_solver(self, params: dict, base_kwargs: dict) -> FVSolver2D:
        physics = self.physics_parameters(
            base_kwargs["a"], base_kwargs["b"],
            base_kwargs["c"], base_kwargs["d"],
        )
        y_grid = base_kwargs["y_grid"]
        ramp = base_kwargs.get("ramp_seconds")
        t_ramp = float(ramp) if ramp is not None else default_ramp_seconds(base_kwargs["dt"])
        q_left_fn, _ = build_qL(
            temporal_family=params["temporal_family"],
            temporal_params=params["temporal_params"],
            spatial_family=params["spatial_family"],
            spatial_params=params["spatial_params"],
            y_grid=y_grid,
            t_ramp=t_ramp,
        )
        q_left_integral_fn, _ = build_qL_integral(
            temporal_family=params["temporal_family"],
            temporal_params=params["temporal_params"],
            spatial_family=params["spatial_family"],
            spatial_params=params["spatial_params"],
            y_grid=y_grid,
            t_ramp=t_ramp,
        )
        return FVSolver2D(
            a=base_kwargs["a"], b=base_kwargs["b"],
            c=base_kwargs["c"], d=base_kwargs["d"],
            Nx=base_kwargs["Nx"], Ny=base_kwargs["Ny"],
            lam_target=base_kwargs["lam_target"],
            layers=physics["layers"],
            t_final=base_kwargs["t_final"],
            flux_f=base_kwargs["flux_f"], flux_A=base_kwargs["flux_A"],
            t_on=base_kwargs["t_on"], t_off=base_kwargs["t_off"],
            phase=base_kwargs["phase"],
            dt=base_kwargs["dt"], tukey_alpha=base_kwargs["tukey_alpha"],
            interface_R=[params["R_c"]],
            q_left_fn=q_left_fn,
            q_left_integral_fn=q_left_integral_fn,
        )

    # ---- dataset item ----

    def setup_dataset(self, ds) -> None:
        if self.representation == "temporal_encoder":
            # Force the 128-sample token grid regardless of the config default.
            ds.temporal_samples = FORCING_TEMPORAL_SAMPLES
            ds._q_callables = {}
        else:
            ds.q_ref = np.float32(SIN_AMP_RANGE[1] * ds.t_final / FORCING_BINS)

        profiles = {}
        for sim_id in ds.sim_ids:
            params = ds.sim_params[int(sim_id)]
            s_vec = SPATIAL_BUILDERS[params["spatial_family"]](
                ds.y_grid, **params["spatial_params"]
            )
            profiles[int(sim_id)] = np.asarray(s_vec, dtype=np.float32)
        ds.s_y_profiles = profiles

    def _build_item_with_resistance(
        self,
        ds,
        sid: int,
        s: int,
        j: int,
        R_c: float,
    ) -> dict[str, np.ndarray]:
        sid = int(sid)
        params = ds.sim_params[sid]
        spatial_family = params["spatial_family"]
        spatial_params = params["spatial_params"]

        T_source = ds.trajectories[sid, s, :, :]
        T_target = ds.trajectories[sid, j, :, :]

        T_source_norm = (T_source - ds.mu_global) / (ds.sigma_global + T_EPS)
        T_target_norm = (T_target - ds.mu_global) / (ds.sigma_global + T_EPS)

        if ds.noise_std > 0:
            T_source_norm = T_source_norm + np.random.randn(*T_source_norm.shape).astype(np.float32) * ds.noise_std

        t_s_val = float(ds.t_grid[s])
        t_j_val = float(ds.t_grid[j])
        t_bar = t_j_val - t_s_val
        # Temporal model-input features scale by the trained normalization horizon
        # (defaults to t_final for non-OOD sets); a lead past the trained horizon
        # therefore yields t_bar_norm > 1.0.
        t_bar_norm = t_bar / ds.time_norm_horizon

        s_y = ds.s_y_profiles[sid]
        S_y = np.broadcast_to(s_y[None, :], (ds.Nx, ds.Ny))
        spatial_base = np.stack(
            [T_source_norm, ds.X_norm, ds.Y_norm, S_y], axis=-1,
        ).astype(np.float32)

        cond_static = build_cond_vector(
            t_bar_norm=t_bar_norm,
            R_c=R_c,
            spatial_family=spatial_family,
            spatial_params=spatial_params,
            y_bounds=(float(ds.y_grid[0]), float(ds.y_grid[-1])),
            allow_unknown_spatial_family=(
                self.spatial_conditioning == "spatial_field_only"
            ),
        )

        Y = T_target_norm[:, :, None].astype(np.float32)
        T_stats = np.array([ds.mu_global, ds.sigma_global], dtype=np.float32)

        if self.representation == "temporal_encoder":
            if sid not in ds._q_callables:
                ds._q_callables[sid] = ramped_temporal(
                    params["temporal_family"],
                    params["temporal_params"],
                    ds.ramp_seconds,
                )
            q = ds._q_callables[sid]
            t_samples, a_m = _sample_a(q, t_s_val, t_j_val, ds.temporal_samples)
            forcing_seq = _forcing_seq_2tok_from_samples(
                t_samples, a_m, A_amp_ref=A_AMP_REF,
            )
            spatial = spatial_base
        else:
            bins = integrate_temporal_bins_ramped_signed(
                params["temporal_family"],
                params["temporal_params"],
                t_s_val,
                t_j_val,
                ds.ramp_seconds,
                K=FORCING_BINS,
            ).astype(np.float32)
            Q_y_bins = (s_y[None, :, None] * bins[None, None, :] / ds.q_ref).astype(np.float32)
            Q_y_bins_2d = np.broadcast_to(Q_y_bins, (ds.Nx, ds.Ny, FORCING_BINS))
            spatial = np.concatenate([spatial_base, Q_y_bins_2d], axis=-1).astype(np.float32)
            forcing_seq = empty_forcing_seq()

        return {
            "spatial": spatial,
            "cond_static": cond_static,
            "forcing_seq": forcing_seq,
            "Y": Y,
            "T_stats": T_stats,
        }

    def build_item(self, ds, sid: int, s: int, j: int) -> dict[str, np.ndarray]:
        params = ds.sim_params[int(sid)]
        item = self._build_item_with_resistance(
            ds, sid, s, j, R_c=float(params["R_c"])
        )
        item["cond_static"] = self._apply_spatial_conditioning_mask(
            item["cond_static"]
        )
        return item

    # ---- val-pair logging ----

    val_pair_fields = ("temporal_family", "spatial_family")

    def val_pair_row(self, ds, sim_id: int, s: int, j: int) -> dict[str, Any]:
        p = ds.sim_params[int(sim_id)]
        return {
            "temporal_family": p.get("temporal_family", ""),
            "spatial_family": p.get("spatial_family", ""),
        }

    # ---- schema / labels ----

    def validate_schema(self, sim_params: np.ndarray, sim_ids: np.ndarray) -> None:
        required = ("R_c", "temporal_family", "temporal_params",
                    "spatial_family", "spatial_params")
        for sid in sim_ids:
            entry = sim_params[int(sid)]
            missing = [k for k in required if k not in entry]
            if missing:
                raise ValueError(
                    f"sim_params[{int(sid)}] missing keys {missing} for benchmark "
                    f"{self.name!r}."
                )

    def plot_label(self, params: dict) -> str:
        spatial = params.get("spatial_family", "?")
        temporal = params.get("temporal_family", "?")
        if temporal == "sin":
            tp = params.get("temporal_params", {})
            A = tp.get("A")
            f = tp.get("f")
            if A is not None and f is not None:
                return f"{spatial}/rectified sin A={A:.0f} f={f:.1f}"
        return f"{spatial}/{temporal}"

    # ---- OOD hooks ----

    def ood_axes(self) -> dict[str, OODAxis]:
        # ID-reference values are the trained sampler-range mids (geometric mid
        # for log-uniform families: f, exp tau, gaussian sigma; arithmetic mid
        # otherwise). exp tau range = (TAU_FRAC_LO*dt, TAU_EXP_FRAC_HI*t_final)
        # = (0.025, 0.15) at the dataset dt=0.005, t_final=0.30; the generator
        # re-validates id_reference/trained_range against the trained config.
        return {
            "rc": OODAxis(
                name="rc", kind="simulation_parameter", field="R_c",
                id_reference=(0.5,), ood_values=(0.0, 0.01, 0.025, 1.25, 1.5, 2.0),
                trained_range=RC_RANGE, resolution_kind="none",
            ),
            "sin_freq": OODAxis(
                name="sin_freq", kind="simulation_parameter", field="f",
                pinned_family="sin", id_reference=(4.472135955,),
                ood_values=(0.5, 25.0, 40.0), trained_range=SIN_FREQ_RANGE,
                resolution_kind="temporal_period",
            ),
            "sin_amp": OODAxis(
                name="sin_amp", kind="simulation_parameter", field="A",
                pinned_family="sin", id_reference=(175.0,),
                ood_values=(25.0, 325.0, 350.0), trained_range=SIN_AMP_RANGE,
                resolution_kind="none",
            ),
            "pulse_count": OODAxis(
                name="pulse_count", kind="simulation_parameter", field="Np",
                pinned_family="pulse_train", id_reference=(3,),
                ood_values=(5, 6), trained_range=(1.0, 4.0),
                resolution_kind="temporal_pulse",
            ),
            "fast_timescale": OODAxis(
                name="fast_timescale", kind="simulation_parameter", field="tau",
                pinned_family="exp", id_reference=(0.061237243,),
                ood_values=(0.02, 0.015, 0.01), trained_range=(0.025, 0.15),
                resolution_kind="temporal_timescale",
            ),
            "slow_decay": OODAxis(
                name="slow_decay", kind="simulation_parameter", field="tau",
                pinned_family="exp", id_reference=(0.061237243,),
                ood_values=(0.20, 0.25, 0.30), trained_range=(0.025, 0.15),
                resolution_kind="none",
            ),
            "patch_width": OODAxis(
                name="patch_width", kind="simulation_parameter", field="w",
                pinned_family="patch", id_reference=(0.35,),
                ood_values=(0.05, 0.75), trained_range=PATCH_W_RANGE,
                resolution_kind="spatial_patch_edge",
            ),
            "gaussian_sigma_y": OODAxis(
                name="gaussian_sigma_y", kind="simulation_parameter",
                field="sigma_y", pinned_family="gaussian",
                id_reference=(0.077459667,), ood_values=(0.015, 0.30),
                trained_range=GAUSS_SIGMA_RANGE,
                resolution_kind="spatial_gaussian_sigma",
            ),
            "triangle_ell": OODAxis(
                name="triangle_ell", kind="simulation_parameter", field="ell",
                pinned_family="triangle", id_reference=(0.20,),
                ood_values=(0.025, 0.50), trained_range=TRIANGLE_ELL_RANGE,
                resolution_kind="spatial_triangle",
            ),
        }

    def draw_latents(
        self,
        rng: np.random.Generator,
        rng_profile: np.random.Generator,
        grids: dict[str, np.ndarray],
        time_cfg: dict[str, Any],
        axis: OODAxis,
    ) -> dict[str, Any]:
        X = grids["X"]
        y_grid = grids.get("y_grid")
        if y_grid is None:
            Y = grids["Y"]
            c, d = float(np.min(Y)), float(np.max(Y))
        else:
            c, d = float(y_grid[0]), float(y_grid[-1])
        dt = float(time_cfg["dt"])
        t_final = float(time_cfg["t_final"])
        window = dict(
            t_on=float(time_cfg.get("t_on", 0.0)),
            t_off=float(time_cfg.get("t_off", 0.2)),
            phase=float(time_cfg.get("phase", 0.0)),
            tukey_alpha=float(time_cfg.get("tukey_alpha", 0.5)),
        )

        # Shared in-distribution background reused verbatim for every non-swept
        # field. For a temporal axis the background spatial draw is the shared
        # non-swept field (and vice versa); the pinned modality below supplies
        # the family the axis sweeps within.
        bg_temporal_family = sample_temporal_family(rng_profile)
        bg_temporal_params = TEMPORAL_SAMPLERS[bg_temporal_family](
            rng_profile, dt=dt, t_final=t_final, **window
        )
        bg_spatial_family = sample_spatial_family(rng_profile)
        bg_spatial_params = SPATIAL_SAMPLERS[bg_spatial_family](rng_profile, c=c, d=d)

        latents: dict[str, Any] = {
            "R_c": float(rng.uniform(*RC_RANGE)),
            "T0": np.full(X.shape, 300.0, dtype=np.float32),
            "ic_family": "uniform_2d",
            "ic_params": {"T0_offset": 0.0},
            "temporal_family": bg_temporal_family,
            "temporal_params": bg_temporal_params,
            "spatial_family": bg_spatial_family,
            "spatial_params": bg_spatial_params,
            "_y_bounds": (c, d),
            "_span": d - c,
        }

        pinned = axis.pinned_family
        if pinned in TEMPORAL_FAMILIES:
            if pinned == "pulse_train":
                n_pool = max(int(v) for v in axis.sweep_values())
                latents["pulse_pool"] = _draw_pulse_pool(
                    rng_profile, dt=dt, t_final=t_final, n_pulses=n_pool
                )
            else:
                latents["pinned_temporal_params"] = TEMPORAL_SAMPLERS[pinned](
                    rng_profile, dt=dt, t_final=t_final, **window
                )
        elif pinned in SPATIAL_FAMILIES:
            latents["pinned_spatial_params"] = SPATIAL_SAMPLERS[pinned](
                rng_profile, c=c, d=d
            )
        return latents

    def apply_ood_value(
        self, latents: dict[str, Any], axis: OODAxis, value: Any,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "R_c": float(latents["R_c"]),
            "T0": latents["T0"],
            "ic_family": latents["ic_family"],
            "ic_params": copy.deepcopy(latents["ic_params"]),
            "temporal_family": latents["temporal_family"],
            "temporal_params": copy.deepcopy(latents["temporal_params"]),
            "spatial_family": latents["spatial_family"],
            "spatial_params": copy.deepcopy(latents["spatial_params"]),
        }

        if axis.name == "rc":
            params["R_c"] = float(value)
            return params

        pinned = axis.pinned_family
        if pinned in TEMPORAL_FAMILIES:
            params["temporal_family"] = pinned
            if pinned == "pulse_train":
                params["temporal_params"] = _slice_pulse_pool(
                    latents["pulse_pool"], int(value)
                )
            else:
                tp = copy.deepcopy(latents["pinned_temporal_params"])
                tp[axis.field] = float(value)
                params["temporal_params"] = tp
            return params

        if pinned in SPATIAL_FAMILIES:
            params["spatial_family"] = pinned
            sp = copy.deepcopy(latents["pinned_spatial_params"])
            span = float(latents["_span"])
            phys = float(value) * span
            sp[axis.field] = phys
            params["spatial_params"] = sp
            if pinned == "patch":
                c, d = latents["_y_bounds"]
                y_c = float(sp["y_c"])
                lo = max(float(c), y_c - 0.5 * phys)
                hi = min(float(d), y_c + 0.5 * phys)
                params["_ood_realized"] = {
                    "patch_w_requested": phys,
                    "patch_w_actual": hi - lo,
                    "patch_y_lo_actual": lo,
                    "patch_y_hi_actual": hi,
                    "patch_centroid_actual": 0.5 * (lo + hi),
                }
            return params

        raise ValueError(
            f"Unhandled OOD axis {axis.name!r} for benchmark {self.name!r}."
        )

    def resolution_scale(self, axis: OODAxis, params: dict) -> float | None:
        kind = axis.resolution_kind
        if kind == "temporal_period":
            return 1.0 / float(params["temporal_params"]["f"])
        if kind == "temporal_timescale":
            return float(params["temporal_params"]["tau"])
        if kind == "temporal_pulse":
            return float(min(params["temporal_params"]["dt_list"]))
        if kind in ("spatial_patch_edge", "spatial_gaussian_sigma", "spatial_triangle"):
            return float(params["spatial_params"][axis.field])
        return None
