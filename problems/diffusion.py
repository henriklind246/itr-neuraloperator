from __future__ import annotations

from typing import Any

import numpy as np

from problems.base import ProblemDims, ProblemSpec
from problems.forcing import (
    FORCING_TEMPORAL_SAMPLES,
    FORCING_TEMPORAL_TOKEN_DIM,
    T_EPS,
    _forcing_seq_2tok_from_samples,
    _sample_a,
)
from src.physics.fv_solver_2d import FVSolver2D, Layer2D
from src.physics.init_conditions import (
    IC_BUILDER_SCHEMA_VERSION,
    IC_SAMPLERS,
    ONLINE_IC_SAMPLER_VERSION,
    build_ic,
    sample_ic_family,
)

# ----- representation constants (canonical home for the diffusion benchmark) ----

# Single homogeneous slab; reused by the collocation geom hook so the data
# trajectories and the physics residual share one conductance.
K_SLAB = 1.0

# Right Dirichlet wall temperature (K). The IC builder pins the right column to
# this and tapers the deviation into that wall; the left/top/bottom zero-Neumann
# conditions are satisfied by the builders themselves.
T_RIGHT = 300.0

# Bump whenever the sampled IC distribution changes; stale datasets then hard
# fail instead of silently mixing distributions.
PROBLEM_VERSION = "diffusion_neumann_ic_v1"

COND_STATIC_DIM = 2  # [t_bar_norm, t_s_norm]; no material/forcing conditioning

# Lean spatial channels [T_tilde, x_norm, y_norm, s_y_const]; the 4th is a
# constant-ones channel the model multiplies the learned temporal latent by.
SPATIAL_CHANNELS_TEMPORAL = 4
S_Y_CHANNEL = 3

# forcing_seq carries a zero-amplitude token stream; with a_m == 0 the reference
# is irrelevant, so use 1.0 (no division by an external amplitude scale).
A_AMP_REF = 1.0


class DiffusionProblem(ProblemSpec):
    """Pure-relaxation diffusion benchmark: only the initial condition varies.

    A single homogeneous slab on ``[a, b] x [c, d]`` with zero left Neumann
    flux, adiabatic top/bottom, and a right Dirichlet wall at 300 K. No
    forcing, no interface, no contact resistance. The dynamics are the decay of
    a per-sim sampled initial field toward the right wall. This isolates the W2
    collocation + full_bc residual code path used by ``forcing`` so the physics
    machinery can be debugged on an easy problem, and it adds an explicit
    initial-condition anchor so physics-only training cannot collapse to the
    constant-300 K field (which satisfies the homogeneous PDE and every BC).

    Only the ``temporal_encoder`` representation is supported; ``bins`` raises.
    """

    name = "diffusion"
    problem_version = PROBLEM_VERSION
    online_sampler_version = ONLINE_IC_SAMPLER_VERSION
    ic_builder_version = IC_BUILDER_SCHEMA_VERSION

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
        Y = grids["Y"]
        b_temp = float(time_cfg.get("b", 1.0))
        T_right = float(time_cfg.get("T_right", T_RIGHT))
        num_sims = int(time_cfg["num_sims"])

        sim_params = []
        for _ in range(num_sims):
            ic_family = sample_ic_family(rng)
            ic_params = IC_SAMPLERS[ic_family](rng)
            T0 = build_ic(
                ic_family, ic_params, X, Y,
                T_right=T_right, b=b_temp,
                taper=True, pin_right_edge=True,
            )
            sim_params.append({
                "ic_family": ic_family,
                "ic_params": ic_params,
                "T0": T0,
            })
        return sim_params

    def configure_solver(self, params: dict, base_kwargs: dict) -> FVSolver2D:
        y_grid = base_kwargs["y_grid"]
        a = float(base_kwargs["a"])
        b = float(base_kwargs["b"])
        # Single homogeneous slab: no interface, no contact resistance.
        layers = [Layer2D(x_left=a, x_right=b, rho=1.0, cp=1.0, k=K_SLAB)]
        # Zero-flux left boundary in vector form so the solver's flux detection
        # uses the vector branch; flux_A=0.0 keeps the scalar-flux path inert.
        q_left_fn = lambda t: np.zeros_like(y_grid)
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
            source=None,
        )

    # ---- dataset item ----

    def setup_dataset(self, ds) -> None:
        # Force the 128-sample token grid regardless of the config default.
        ds.temporal_samples = FORCING_TEMPORAL_SAMPLES

    def build_item(self, ds, sid: int, s: int, j: int) -> dict[str, np.ndarray]:
        sid = int(sid)

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

        ones = np.ones((ds.Nx, ds.Ny), dtype=np.float32)
        spatial = np.stack(
            [T_source_norm, ds.X_norm, ds.Y_norm, ones], axis=-1,
        ).astype(np.float32)

        cond_static = np.array([t_bar_norm, t_s_norm], dtype=np.float32)

        # Zero-amplitude token stream over [t_s, t_j]. _sample_a is NaN-safe at
        # t_s == t_j (r from linspace; t_bar only multiplies).
        t_samples, a_m = _sample_a(lambda t: 0.0, t_s_val, t_j_val, ds.temporal_samples)
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

    # ---- collocation hooks (forcing-free W2 wiring) ----

    def collocation_geom_cfg(
        self, ds, phys_cfg: dict, mu_global: float, sigma_global: float, dt: float,
    ) -> dict[str, Any]:
        """Homogeneous CN geometry for the full_bc residual.

        ``k_left == k_right == K_SLAB`` and ``R_c = 0`` collapse the interface
        face to the ordinary ``k/hx``, so the geometry is uniform; the
        ``interface_x`` value is irrelevant to the conductances but is pinned to
        an actual interior x-grid node so any face-alignment check passes.
        """
        x_grid = np.asarray(ds.x_grid)
        xm = 0.5 * (float(x_grid[0]) + float(x_grid[-1]))
        idx = int(np.argmin(np.abs(x_grid - xm)))
        idx = min(max(idx, 1), len(x_grid) - 2)
        interface_x = float(x_grid[idx])
        return {
            "x_grid": x_grid,
            "y_grid": np.asarray(ds.y_grid),
            "k_left": K_SLAB,
            "k_right": K_SLAB,
            "interface_x": interface_x,
            "dt": float(dt),
            "sigma_global": float(sigma_global),
            "T_right_tilde": (T_RIGHT - float(mu_global)) / float(sigma_global),
        }

    def collocation_closure(
        self, ds, sid: int, params: dict, t: float, t_dt: float,
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """Zero left-flux, zero contact resistance at both collocation times."""
        Ny = int(ds.Ny)
        zeros = np.zeros(Ny, dtype=np.float32)
        return (0.0, zeros, zeros.copy(), zeros.copy())

    def collocation_base_plan(self, ds, dt: float) -> dict[str, Any]:
        """Base is the IC (snapshot 0, t_s = 0); pairs are consecutive grid points."""
        return {"base_snapshot_index": 0, "on_grid_pairs": True}

    # ---- val-pair logging ----

    val_pair_fields = ("ic_family",)

    def val_pair_row(self, ds, sim_id: int, s: int, j: int) -> dict[str, Any]:
        p = ds.sim_params[int(sim_id)]
        return {"ic_family": p.get("ic_family", "")}

    # ---- schema / labels ----

    def validate_schema(self, sim_params: np.ndarray, sim_ids: np.ndarray) -> None:
        required = ("ic_family", "ic_params", "T0")
        forbidden = ("temporal_family", "spatial_family", "interface_x", "R_c")
        for sid in sim_ids:
            entry = sim_params[int(sid)]
            missing = [k for k in required if k not in entry]
            if missing:
                raise ValueError(
                    f"sim_params[{int(sid)}] missing keys {missing} for benchmark "
                    f"{self.name!r}."
                )
            present_legacy = [k for k in forbidden if k in entry]
            if present_legacy:
                raise ValueError(
                    f"sim_params[{int(sid)}] still contains keys {present_legacy} "
                    f"that benchmark {self.name!r} forbids; regenerate the dataset."
                )

    def plot_label(self, params: dict) -> str:
        return str(params.get("ic_family", self.name))
