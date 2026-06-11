from __future__ import annotations

from typing import Any

import numpy as np

from problems.base import ProblemDims, ProblemSpec, empty_forcing_seq
from src.physics.boundary_forcing import (
    GAUSS_SIGMA_RANGE,
    PATCH_W_RANGE,
    TRIANGLE_ELL_RANGE,
    TEMPORAL_BUILDERS,
    TEMPORAL_SAMPLERS,
    SPATIAL_BUILDERS,
    SPATIAL_SAMPLERS,
    SIN_AMP_RANGE,
    FORCING_BINS,
    integrate_temporal_bins_signed,
    sample_spatial_family,
    sample_temporal_family,
    build_qL,
    build_qL_integral,
)
from src.physics.fv_solver_2d import FVSolver2D

# ----- representation constants (canonical home for the forcing benchmark) -----

RC_RANGE = (0.05, 1.0)
T_EPS = 1e-6

SPATIAL_FAMILY_ORDER = ("uniform", "patch", "gaussian", "triangle")

_BASE_DIM = 3
_SPATIAL_ONEHOT_DIM = len(SPATIAL_FAMILY_ORDER)
_SPATIAL_PARAM_DIM = 4
COND_STATIC_DIM = (
    _BASE_DIM + _SPATIAL_ONEHOT_DIM + _SPATIAL_PARAM_DIM
)  # 11 (forcing-agnostic: no temporal one-hot, no forcing summary)

# temporal_encoder mode: 128 samples x 2 tokens [r_m, a_m / A_ref].
FORCING_TEMPORAL_SAMPLES = 128
FORCING_TEMPORAL_TOKEN_DIM = 2
A_AMP_REF = 300.0

# Spatial channels per representation. temporal_encoder mode lifts the lean
# [T_tilde, x, y, s_y] base; bins mode appends the FORCING_BINS integral Q-bins.
SPATIAL_CHANNELS_TEMPORAL = 4
SPATIAL_CHANNELS_BINS = SPATIAL_CHANNELS_TEMPORAL + FORCING_BINS  # 20

# s_y is the 4th spatial channel (index 3) in both representations.
S_Y_CHANNEL = 3

_LOG_SIGMA_LO = float(np.log(GAUSS_SIGMA_RANGE[0]))
_LOG_SIGMA_HI = float(np.log(GAUSS_SIGMA_RANGE[1]))


# ----- conditioning / forcing-sequence builders -----

def build_cond_vector(t_bar_norm: float, t_s_norm: float, R_c: float,
                      spatial_family: str, spatial_params: dict) -> np.ndarray:
    """Assemble the 11-dim forcing-agnostic static conditioning vector.

    Layout: [t_bar_norm, t_s_norm, R_c_norm, spatial_onehot(4),
    spatial_params(4)]. Carries no temporal-family identity or forcing
    summary; in temporal_encoder mode the only temporal forcing information
    lives in forcing_seq, in bins mode it lives in the Q-bin spatial channels.
    """
    R_c_norm = (R_c - RC_RANGE[0]) / (RC_RANGE[1] - RC_RANGE[0])
    base = np.array([t_bar_norm, t_s_norm, R_c_norm], dtype=np.float32)

    spatial_oh = np.zeros(_SPATIAL_ONEHOT_DIM, dtype=np.float32)
    spatial_oh[SPATIAL_FAMILY_ORDER.index(spatial_family)] = 1.0
    y_c_norm = w_norm = sigma_y_norm = ell_norm = 0.0
    if spatial_family == "patch":
        y_c_norm = float(spatial_params["y_c"])
        w_norm = (spatial_params["w"] - PATCH_W_RANGE[0]) / (PATCH_W_RANGE[1] - PATCH_W_RANGE[0])
    elif spatial_family == "gaussian":
        y_c_norm = float(spatial_params["y_c"])
        sigma_y_norm = (np.log(spatial_params["sigma_y"]) - _LOG_SIGMA_LO) / (_LOG_SIGMA_HI - _LOG_SIGMA_LO)
    elif spatial_family == "triangle":
        y_c_norm = float(spatial_params["y_c"])
        ell_norm = (spatial_params["ell"] - TRIANGLE_ELL_RANGE[0]) / (TRIANGLE_ELL_RANGE[1] - TRIANGLE_ELL_RANGE[0])
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


class ForcingProblem(ProblemSpec):
    """Separable boundary flux q_L(y, t) = a(t) * s(y) benchmark.

    Two representations over the same trajectories/sim_params:

    - ``temporal_encoder``: 4 spatial channels [T_tilde, x, y, s_y], 11
      forcing-agnostic static dims, a (128, 2) forcing_seq [r_m, a_m / A_ref],
      temporal encoder on with time-augmented spatial injection.
    - ``bins``: 20 spatial channels [T_tilde, x, y, s_y, Q_y_bin_0..15], same
      11 static dims, an empty forcing_seq, temporal encoder off.

    Interface fixed at x = 0.5.
    """

    name = "forcing"

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
            spatial_params = SPATIAL_SAMPLERS[spatial_family](rng_profile)

            sim_params.append({
                "R_c": R_c,
                "T0": T0,
                "ic_family": ic_family,
                "ic_params": ic_params,
                "temporal_family": temporal_family,
                "temporal_params": temporal_params,
                "spatial_family": spatial_family,
                "spatial_params": spatial_params,
            })

        return sim_params

    def configure_solver(self, params: dict, base_kwargs: dict) -> FVSolver2D:
        y_grid = base_kwargs["y_grid"]
        q_left_fn, _ = build_qL(
            temporal_family=params["temporal_family"],
            temporal_params=params["temporal_params"],
            spatial_family=params["spatial_family"],
            spatial_params=params["spatial_params"],
            y_grid=y_grid,
        )
        q_left_integral_fn, _ = build_qL_integral(
            temporal_family=params["temporal_family"],
            temporal_params=params["temporal_params"],
            spatial_family=params["spatial_family"],
            spatial_params=params["spatial_params"],
            y_grid=y_grid,
        )
        return FVSolver2D(
            a=base_kwargs["a"], b=base_kwargs["b"],
            c=base_kwargs["c"], d=base_kwargs["d"],
            Nx=base_kwargs["Nx"], Ny=base_kwargs["Ny"],
            lam_target=base_kwargs["lam_target"],
            layers=base_kwargs["layers"],
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
            ds._r = np.linspace(0.0, 1.0, ds.temporal_samples, dtype=np.float32)
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

    def build_item(self, ds, sid: int, s: int, j: int) -> dict[str, np.ndarray]:
        sid = int(sid)
        params = ds.sim_params[sid]
        R_c = float(params["R_c"])
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
        t_bar_norm = t_bar / ds.t_final
        t_s_norm = t_s_val / ds.t_final

        s_y = ds.s_y_profiles[sid]
        S_y = np.broadcast_to(s_y[None, :], (ds.Nx, ds.Ny))
        spatial_base = np.stack(
            [T_source_norm, ds.X_norm, ds.Y_norm, S_y], axis=-1,
        ).astype(np.float32)

        cond_static = build_cond_vector(
            t_bar_norm=t_bar_norm,
            t_s_norm=t_s_norm,
            R_c=R_c,
            spatial_family=spatial_family,
            spatial_params=spatial_params,
        )

        Y = T_target_norm[:, :, None].astype(np.float32)
        T_stats = np.array([ds.mu_global, ds.sigma_global], dtype=np.float32)

        if self.representation == "temporal_encoder":
            if sid not in ds._q_callables:
                ds._q_callables[sid] = TEMPORAL_BUILDERS[params["temporal_family"]](
                    **params["temporal_params"]
                )
            q = ds._q_callables[sid]
            t_samples, a_m = _sample_a(q, t_s_val, t_j_val, ds.temporal_samples)
            forcing_seq = _forcing_seq_2tok_from_samples(
                t_samples, a_m, A_amp_ref=A_AMP_REF,
            )
            spatial = spatial_base
        else:
            bins = integrate_temporal_bins_signed(
                params["temporal_family"],
                params["temporal_params"],
                t_s_val,
                t_j_val,
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
                return f"{spatial}/sin A={A:.0f} f={f:.1f}"
        return f"{spatial}/{temporal}"
