from __future__ import annotations

from typing import Any

import numpy as np

from problems.base import ProblemDims, ProblemSpec
from problems.forcing import (
    A_AMP_REF,
    FORCING_TEMPORAL_SAMPLES,
    FORCING_TEMPORAL_TOKEN_DIM,
    T_EPS,
    _forcing_seq_2tok_from_samples,
    _sample_a,
)
from src.physics.boundary_forcing import (
    FORCING_SCHEMA_VERSION,
    RAMP_SCHEMA_VERSION,
    SPATIAL_BUILDERS,
    SPATIAL_SAMPLERS,
    TEMPORAL_SAMPLERS,
    build_qL,
    build_qL_integral,
    build_interface_forcing,
    default_ramp_seconds,
    ramped_temporal,
    sample_spatial_family,
    sample_temporal_family,
)
from src.physics.fv_solver_2d import FVSolver2D, Layer2D

# ----- representation constants (single-slab forcing benchmark) -----------------

# Single homogeneous slab; k == 1 keeps the interior operator alpha == 1 so the
# left-Neumann residual normalization is q_L / (k * sigma) with k = 1.
K_SLAB = 1.0

# Right Dirichlet wall temperature (K); the hard-Dirichlet ansatz pins x = 1.
T_RIGHT = 300.0

# [t_bar_norm, t_s_norm]; no material/spatial conditioning. The forcing is the
# only per-sim signal and the physics-only path conditions on it via the encoder
# image, not through cond_static.
COND_STATIC_DIM = 2

# Lean spatial channels [T_tilde, x_norm, y_norm, s_y]; s_y is the sampled left-
# wall spatial profile so the (unused-in-PINO) supervised item stays coherent.
SPATIAL_CHANNELS_TEMPORAL = 4
S_Y_CHANNEL = 3


class DiffusionForcingProblem(ProblemSpec):
    """Single-slab, forcing-driven diffusion benchmark (fixed uniform IC).

    A single homogeneous slab on ``[a, b] x [c, d]`` (uniform ``k = K_SLAB``, no
    interface, no contact resistance) with a fixed uniform initial condition of
    300 K, driven by an inhomogeneous left-wall Neumann flux
    ``q_L(y, t) = a(t) * s(y)`` from the forcing families in
    ``src.physics.boundary_forcing``. Because the IC is constant, the per-sim
    signal is the forcing; the physics-only (PINO) path conditions the encoder on
    the forcing image rather than the IC.

    This spec exists primarily so the data generator can produce FV validation
    trajectories and the ``ForcingCViT`` PINO trainer can reconstruct
    ``q_L(y, t)``. The supervised ``build_item`` is intentionally minimal (the
    physics-only path does not consume paired snapshots); only
    ``temporal_encoder`` is supported.
    """

    name = "diffusion_forcing"

    def __init__(self, representation: str = "temporal_encoder"):
        self.representation = representation
        if representation != "temporal_encoder":
            raise ValueError(
                f"Unknown representation {representation!r} for benchmark "
                f"{self.name!r}. Only 'temporal_encoder' is supported."
            )
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
        temporal_window = dict(
            t_on=float(time_cfg.get("t_on", 0.0)),
            t_off=float(time_cfg.get("t_off", 0.2)),
            phase=float(time_cfg.get("phase", 0.0)),
            tukey_alpha=float(time_cfg.get("tukey_alpha", 0.5)),
        )

        sim_params = []
        for _ in range(num_sims):
            # Fixed uniform 300 K IC: the forcing is the only per-sim signal.
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
        a = float(base_kwargs["a"])
        b = float(base_kwargs["b"])
        # Single homogeneous slab: no interface, no contact resistance.
        layers = [Layer2D(x_left=a, x_right=b, rho=1.0, cp=1.0, k=K_SLAB)]
        ramp = base_kwargs.get("ramp_seconds")
        t_ramp = float(ramp) if ramp is not None else default_ramp_seconds(base_kwargs["dt"])
        # The inhomogeneous left-Neumann flux is the per-sim signal; q_left_fn
        # overrides the scalar flux path, so flux_A is inert here.
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
            layers=layers,
            t_final=base_kwargs["t_final"],
            flux_f=base_kwargs["flux_f"], flux_A=0.0,
            t_on=base_kwargs["t_on"], t_off=base_kwargs["t_off"],
            phase=base_kwargs["phase"],
            dt=base_kwargs["dt"], tukey_alpha=base_kwargs["tukey_alpha"],
            interface_R=None,
            q_left_fn=q_left_fn,
            q_left_integral_fn=q_left_integral_fn,
            source=None,
        )

    # ---- collocation hooks (shared by forcing descendants) ----

    def collocation_geom_cfg(
        self, ds, phys_cfg: dict, mu_global: float, sigma_global: float, dt: float,
    ) -> dict[str, Any]:
        """Direct homogeneous geometry and solver-matching closure metadata."""
        sigma = float(sigma_global)
        if not np.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("sigma_global must be finite and > 0 for FV residuals")
        return {
            "geometry_kind": "homogeneous",
            "geometry_version": "homogeneous_cn_v1",
            "x_grid": np.asarray(ds.x_grid, dtype=np.float64),
            "y_grid": np.asarray(ds.y_grid, dtype=np.float64),
            "k": K_SLAB,
            "rho": 1.0,
            "cp": 1.0,
            "dt": float(dt),
            "sigma_global": sigma,
            "T_right_tilde": (T_RIGHT - float(mu_global)) / sigma,
            "forcing_quadrature": "exact_interval_integral",
            "closure_identity": "separable_left_flux_exact_v1",
            "forcing_schema_version": FORCING_SCHEMA_VERSION,
            "ramp_schema_version": RAMP_SCHEMA_VERSION,
        }

    def collocation_closure(
        self, ds, sid: int, params: dict, t: float, t_dt: float,
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """Reconstruct the exact left-flux integral used by ``FVSolver2D``."""
        qn, qnp1, qint = build_interface_forcing(
            params["temporal_family"], params["temporal_params"],
            params["spatial_family"], params["spatial_params"],
            np.asarray(ds.y_grid, dtype=np.float64), float(t), float(t_dt),
            float(ds.ramp_seconds),
        )
        return 0.0, qn, qnp1, qint

    # ---- dataset item (minimal; physics-only path does not consume it) ----

    def setup_dataset(self, ds) -> None:
        # Force the 128-sample token grid regardless of the config default.
        ds.temporal_samples = FORCING_TEMPORAL_SAMPLES
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

        T_source = ds.trajectories[sid, s, :, :]
        T_target = ds.trajectories[sid, j, :, :]

        T_source_norm = (T_source - ds.mu_global) / (ds.sigma_global + T_EPS)
        T_target_norm = (T_target - ds.mu_global) / (ds.sigma_global + T_EPS)

        if ds.noise_std > 0:
            T_source_norm = T_source_norm + np.random.randn(
                *T_source_norm.shape
            ).astype(np.float32) * ds.noise_std

        t_s_val = float(ds.t_grid[s])
        t_j_val = float(ds.t_grid[j])
        t_bar_norm = (t_j_val - t_s_val) / ds.time_norm_horizon
        t_s_norm = t_s_val / ds.time_norm_horizon

        s_y = ds.s_y_profiles[sid]
        S_y = np.broadcast_to(s_y[None, :], (ds.Nx, ds.Ny))
        spatial = np.stack(
            [T_source_norm, ds.X_norm, ds.Y_norm, S_y], axis=-1,
        ).astype(np.float32)

        cond_static = np.array([t_bar_norm, t_s_norm], dtype=np.float32)

        if sid not in ds._q_callables:
            ds._q_callables[sid] = ramped_temporal(
                params["temporal_family"],
                params["temporal_params"],
                ds.ramp_seconds,
            )
        q = ds._q_callables[sid]
        t_samples, a_m = _sample_a(q, t_s_val, t_j_val, ds.temporal_samples)
        forcing_seq = _forcing_seq_2tok_from_samples(t_samples, a_m, A_amp_ref=A_AMP_REF)

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
        required = ("T0", "ic_family", "temporal_family", "temporal_params",
                    "spatial_family", "spatial_params")
        forbidden = ("interface_x", "R_c")
        for sid in sim_ids:
            entry = sim_params[int(sid)]
            missing = [k for k in required if k not in entry]
            if missing:
                raise ValueError(
                    f"sim_params[{int(sid)}] missing keys {missing} for benchmark "
                    f"{self.name!r}."
                )
            present_forbidden = [k for k in forbidden if k in entry]
            if present_forbidden:
                raise ValueError(
                    f"sim_params[{int(sid)}] contains keys {present_forbidden} that "
                    f"benchmark {self.name!r} forbids (single slab, no interface/R_c); "
                    f"regenerate the dataset."
                )

    def plot_label(self, params: dict) -> str:
        return str(params.get("temporal_family", self.name))
