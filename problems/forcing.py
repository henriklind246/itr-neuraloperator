from __future__ import annotations

from typing import Any

import numpy as np

from problems.base import ProblemDims, ProblemSpec
from src.physics.boundary_forcing import (
    GAUSS_SIGMA_RANGE,
    PATCH_W_RANGE,
    TRIANGLE_ELL_RANGE,
    TEMPORAL_FAMILY_ORDER,
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
from src.physics.init_conditions import IC_FAMILIES, IC_SAMPLERS, sample_ic_family, build_ic
from src.physics.fv_solver_2d import FVSolver2D

# ----- representation constants (canonical home for the forcing benchmark) -----

RC_RANGE = (0.05, 1.0)
T_EPS = 1e-6

SPATIAL_FAMILY_ORDER = ("uniform", "patch", "gaussian", "triangle")

_BASE_DIM = 3
_SPATIAL_ONEHOT_DIM = len(SPATIAL_FAMILY_ORDER)
_SPATIAL_PARAM_DIM = 4
_TEMPORAL_ONEHOT_DIM = len(TEMPORAL_FAMILY_ORDER)
_FORCING_SUMMARY_DIM = 8
COND_STATIC_DIM = (
    _BASE_DIM + _SPATIAL_ONEHOT_DIM + _SPATIAL_PARAM_DIM
    + _TEMPORAL_ONEHOT_DIM + _FORCING_SUMMARY_DIM
)  # 23

TEMPORAL_SAMPLES = 64
TEMPORAL_TOKEN_DIM = 5
A_AMP_REF = 300.0
SPATIAL_IN_CHANNELS = 4 + FORCING_BINS  # 20

_LOG_SIGMA_LO = float(np.log(GAUSS_SIGMA_RANGE[0]))
_LOG_SIGMA_HI = float(np.log(GAUSS_SIGMA_RANGE[1]))


# ----- conditioning / forcing-sequence builders -----

def build_cond_vector(t_bar_norm: float, t_s_norm: float, R_c: float,
                      spatial_family: str, spatial_params: dict,
                      temporal_family: str,
                      forcing_summary: np.ndarray) -> np.ndarray:
    """Assemble the 23-dim static conditioning vector (see COND_STATIC_DIM)."""
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

    temporal_oh = np.zeros(_TEMPORAL_ONEHOT_DIM, dtype=np.float32)
    temporal_oh[TEMPORAL_FAMILY_ORDER.index(temporal_family)] = 1.0

    fs = np.asarray(forcing_summary, dtype=np.float32).reshape(-1)
    if fs.shape[0] != _FORCING_SUMMARY_DIM:
        raise ValueError(
            f"forcing_summary must have shape ({_FORCING_SUMMARY_DIM},), got {fs.shape}"
        )

    return np.concatenate([base, spatial_oh, spatial_p, temporal_oh, fs]).astype(np.float32)


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


def _forcing_seq_from_samples(
    t_samples: np.ndarray,
    a_m: np.ndarray,
    t_s: float,
    t_j: float,
    t_final: float,
    A_amp_ref: float,
    A_cum_ref: float,
) -> np.ndarray:
    t_bar = float(t_j) - float(t_s)
    M = t_samples.shape[0]
    r = np.linspace(0.0, 1.0, M, dtype=np.float32)

    A_cum = np.empty_like(a_m)
    A_cum[0] = 0.0
    A_cum[1:] = np.cumsum(0.5 * (a_m[1:] + a_m[:-1]) * np.diff(t_samples))

    tok0 = r
    tok1 = a_m / np.float32(A_amp_ref)
    tok2 = A_cum / np.float32(A_cum_ref)
    tok3 = ((1.0 - r) * t_bar / float(t_final)).astype(np.float32)
    tok4 = (t_samples / float(t_final)).astype(np.float32)
    return np.stack([tok0, tok1, tok2, tok3, tok4], axis=-1).astype(np.float32)


def build_forcing_seq(
    q,
    t_s: float,
    t_j: float,
    t_final: float,
    M: int = TEMPORAL_SAMPLES,
    A_amp_ref: float = A_AMP_REF,
    A_cum_ref: float | None = None,
) -> np.ndarray:
    """Sample a(t) on M points over [t_s, t_j] and build the (M, 5) token array."""
    if A_cum_ref is None:
        A_cum_ref = float(A_amp_ref) * float(t_final)
    t_samples, a_m = _sample_a(q, t_s, t_j, int(M))
    return _forcing_seq_from_samples(t_samples, a_m, t_s, t_j, t_final, A_amp_ref, A_cum_ref)


def build_forcing_summary(
    a_vals: np.ndarray,
    t_vals: np.ndarray,
    t_s: float,
    t_j: float,
    t_final: float,
    A_amp_ref: float = A_AMP_REF,
) -> np.ndarray:
    """Global interval-level descriptors S1..S8 of a(t) over [t_s, t_j]."""
    A_ref = float(A_amp_ref)
    dt_interval = max(float(t_j) - float(t_s), 1e-8)
    denom_impulse = A_ref * float(t_final)

    if hasattr(np, "trapezoid"):
        trapz = np.trapezoid
    else:
        trapz = np.trapz
    I_signed = float(trapz(a_vals, t_vals))
    I_abs = float(trapz(np.abs(a_vals), t_vals))
    I_pos = float(trapz(np.clip(a_vals, 0.0, None), t_vals))
    I_neg = float(trapz(np.clip(-a_vals, 0.0, None), t_vals))
    I_sq = float(trapz(a_vals ** 2, t_vals))

    S1 = I_signed / denom_impulse
    S2 = I_abs / denom_impulse
    S3 = I_pos / denom_impulse
    S4 = I_neg / denom_impulse
    S5 = (I_signed / dt_interval) / A_ref
    S6 = np.sqrt(max(I_sq / dt_interval, 0.0)) / A_ref
    S7 = float(np.max(np.abs(a_vals))) / A_ref
    S8 = float(a_vals[-1]) / A_ref

    return np.array([S1, S2, S3, S4, S5, S6, S7, S8], dtype=np.float32)


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

    20 spatial channels [T_tilde, x, y, s_y, Q_y_bin_0..15], 23 static
    conditioning dims, a (64, 5) forcing_seq, temporal encoder on, interface
    fixed at x = 0.5.
    """

    name = "forcing"
    dims = ProblemDims(
        in_channels=SPATIAL_IN_CHANNELS,
        cond_static_dim=COND_STATIC_DIM,
        has_forcing_seq=True,
        temporal_token_dim=TEMPORAL_TOKEN_DIM,
        t_stats_dim=2,
        use_temporal_encoder=True,
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
        Y = grids["Y"]
        Nx, Ny = X.shape[0], X.shape[1]

        num_sims = int(time_cfg["num_sims"])
        dt = float(time_cfg["dt"])
        t_final = float(time_cfg["t_final"])
        b = float(time_cfg.get("b", 1.0))
        T_right = float(time_cfg.get("T_right", 300.0))
        lhs_seed = int(time_cfg.get("lhs_seed", 0))
        temporal_window = dict(
            t_on=float(time_cfg.get("t_on", 0.0)),
            t_off=float(time_cfg.get("t_off", 0.2)),
            phase=float(time_cfg.get("phase", 0.0)),
            tukey_alpha=float(time_cfg.get("tukey_alpha", 0.5)),
        )

        R_c_values = _generate_lhs_R_c(num_sims, seed=lhs_seed)

        # Optional IC-family restriction (e.g. exclude grf_2d for the spatial
        # resolution-invariance sweep). None => unchanged default (uniform over
        # all families). One rng.choice draw per sim either way, so the IC rng
        # stream stays in lock-step across resolutions.
        ic_families = time_cfg.get("ic_families")
        if ic_families is None:
            ic_probs = None
        else:
            allowed = set(ic_families)
            ic_probs = np.array(
                [1.0 if fam in allowed else 0.0 for fam in IC_FAMILIES],
                dtype=float,
            )

        sim_params = []
        for i in range(num_sims):
            R_c = float(R_c_values[i])

            ic_family = sample_ic_family(rng, probs=ic_probs)
            ic_params = IC_SAMPLERS[ic_family](rng, Nx=Nx, Ny=Ny)
            T0 = build_ic(ic_family, ic_params, X, Y, T_right=T_right, b=b)

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
        ds.A_cum_ref = float(A_AMP_REF * ds.t_final)
        ds.q_ref = np.float32(SIN_AMP_RANGE[1] * ds.t_final / FORCING_BINS)
        ds._r = np.linspace(0.0, 1.0, ds.temporal_samples, dtype=np.float32)
        ds._q_callables = {}

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
        bins = integrate_temporal_bins_signed(
            params["temporal_family"],
            params["temporal_params"],
            t_s_val,
            t_j_val,
            K=FORCING_BINS,
        ).astype(np.float32)
        Q_y_bins = (s_y[None, :, None] * bins[None, None, :] / ds.q_ref).astype(np.float32)
        Q_y_bins_2d = np.broadcast_to(Q_y_bins, (ds.Nx, ds.Ny, FORCING_BINS))
        spatial_base = np.stack([T_source_norm, ds.X_norm, ds.Y_norm, S_y], axis=-1).astype(np.float32)
        spatial = np.concatenate([spatial_base, Q_y_bins_2d], axis=-1).astype(np.float32)

        if sid not in ds._q_callables:
            ds._q_callables[sid] = TEMPORAL_BUILDERS[params["temporal_family"]](
                **params["temporal_params"]
            )
        q = ds._q_callables[sid]

        t_samples, a_m = _sample_a(q, t_s_val, t_j_val, ds.temporal_samples)
        forcing_seq = _forcing_seq_from_samples(
            t_samples, a_m, t_s_val, t_j_val, ds.t_final, A_AMP_REF, ds.A_cum_ref,
        )
        forcing_summary = build_forcing_summary(
            a_m, t_samples, t_s_val, t_j_val, ds.t_final, A_AMP_REF,
        )

        cond_static = build_cond_vector(
            t_bar_norm=t_bar_norm,
            t_s_norm=t_s_norm,
            R_c=R_c,
            spatial_family=spatial_family,
            spatial_params=spatial_params,
            temporal_family=params["temporal_family"],
            forcing_summary=forcing_summary,
        )

        Y = T_target_norm[:, :, None].astype(np.float32)
        T_stats = np.array([ds.mu_global, ds.sigma_global], dtype=np.float32)

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
