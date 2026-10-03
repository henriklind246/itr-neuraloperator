from __future__ import annotations

from typing import Any
import numpy as np
import torch
from problems.base import ProblemDims, ProblemSpec
from problems.forcing import (A_AMP_REF, FORCING_TEMPORAL_SAMPLES,
    FORCING_TEMPORAL_TOKEN_DIM, T_EPS, _sample_a, _forcing_seq_3tok_from_samples)
from src.physics.boundary_forcing import (SPATIAL_BUILDERS, SPATIAL_SAMPLERS,
    TEMPORAL_SAMPLERS, build_qL, build_qL_integral, build_interface_forcing,
    default_ramp_seconds, ramped_temporal, integrate_temporal_intervals_ramped_signed,
    FORCING_SCHEMA_VERSION, RAMP_SCHEMA_VERSION)
from src.physics.init_conditions import (IC_BUILDER_SCHEMA_VERSION, IC_FAMILIES,
    IC_SAMPLERS, ONLINE_IC_SAMPLER_VERSION, build_ic, canonical_ic_params,
    sample_ic_family)
from src.physics.fv_solver_2d import FVSolver2D, Layer2D

K_SLAB = 1.0
T_RIGHT = 300.0
COND_STATIC_DIM = 1
SPATIAL_CHANNELS_TEMPORAL = 4
S_Y_CHANNEL = 3
FIXED_TEMPORAL_FAMILY = "sin"
FIXED_SPATIAL_FAMILY = "uniform"
PROBLEM_VERSION = "forcing_single_varying_ic_v2"
IC_MODE = "varying"

class DiffusionForcingSingleProblem(ProblemSpec):
    """Homogeneous sin/uniform slab with prescribed varying initial conditions."""

    name = "diffusion_forcing_single"
    problem_version = PROBLEM_VERSION
    ic_mode = IC_MODE
    online_sampler_version = ONLINE_IC_SAMPLER_VERSION
    ic_builder_version = IC_BUILDER_SCHEMA_VERSION
    val_pair_fields = ("temporal_family", "spatial_family", "ic_family")
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
            temporal_token_dim=FORCING_TEMPORAL_TOKEN_DIM,
            t_stats_dim=2,
            s_y_channel=S_Y_CHANNEL,
            use_forcing_time_aug=True,
        )

    def configure_spatial_input(self, *, itr: bool, lead_time: bool, material_side: bool) -> None:
        if material_side:
            raise ValueError("material_side requires layered geometry; this benchmark is homogeneous")
        super().configure_spatial_input(itr=itr, lead_time=lead_time, material_side=False)

    def configure_solver(self, params: dict, base_kwargs: dict) -> FVSolver2D:
        y_grid = base_kwargs["y_grid"]
        a = float(base_kwargs["a"])
        b = float(base_kwargs["b"])

        layers = [Layer2D(x_left=a, x_right=b, rho=1.0, cp=1.0, k=K_SLAB)]
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

    def setup_dataset(self, ds) -> None:

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

    def build_step_inputs(self, state, coords, average_flux, dt, horizon):
        """Encode the solver's interval-average forcing at the fixed step size."""
        batch, nx, ny = state.shape
        xy = coords.reshape(1, nx, ny, 2).expand(batch, -1, -1, -1)
        spatial = torch.cat((state[..., None], xy, torch.ones_like(state[..., None])), dim=-1)
        r = torch.linspace(0, 1, FORCING_TEMPORAL_SAMPLES,
                           device=state.device, dtype=state.dtype).expand(batch, -1)
        # The benchmark is uniform in y; repeating its exact mean avoids giving
        # the FNO waveform information that the CN step and CViT never consume.
        a = average_flux[:, :1].expand_as(r)
        return dict(spatial=spatial,
                    cond_static=state.new_full((batch, self.dims.cond_static_dim), dt / horizon),
                    forcing_seq=torch.stack((r, a, a), dim=-1))

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
        fields = self.build_inputs(ds, sid, T_source_norm, t_s_val, t_j_val)
        fields.update(
            Y=T_target_norm[:, :, None].astype(np.float32),
            T_stats=np.array([ds.mu_global, ds.sigma_global], dtype=np.float32),
        )
        return fields

    def build_inputs(self, ds, sid: int, T_source_norm: np.ndarray,
                     t_s_val: float, t_j_val: float) -> dict[str, np.ndarray]:
        """Encode prescribed fields and forcing without reading target temperatures."""
        params = ds.sim_params[sid]
        t_bar_norm = (t_j_val - t_s_val) / ds.time_norm_horizon

        s_y = ds.s_y_profiles[sid]
        S_y = np.broadcast_to(s_y[None, :], (ds.Nx, ds.Ny))
        spatial = np.stack(
            [T_source_norm, ds.X_norm, ds.Y_norm, S_y], axis=-1,
        ).astype(np.float32)

        cond_static = np.array([t_bar_norm], dtype=np.float32)

        if sid not in ds._q_callables:
            ds._q_callables[sid] = ramped_temporal(
                params["temporal_family"],
                params["temporal_params"],
                ds.ramp_seconds,
            )
        q = ds._q_callables[sid]
        t_samples, a_m = _sample_a(q, t_s_val, t_j_val, ds.temporal_samples)
        forcing_seq = _forcing_seq_3tok_from_samples(
            t_samples, a_m, A_amp_ref=A_AMP_REF,
            interval_integral_fn=lambda lo, hi: integrate_temporal_intervals_ramped_signed(
                params["temporal_family"], params["temporal_params"], lo, hi, ds.ramp_seconds),
        )

        return {
            "spatial": spatial,
            "cond_static": cond_static,
            "forcing_seq": forcing_seq,
        }

    def val_pair_row(self, ds, sim_id: int, s: int, j: int) -> dict[str, Any]:
        p = ds.sim_params[int(sim_id)]
        return {
            "temporal_family": p.get("temporal_family", ""),
            "spatial_family": p.get("spatial_family", ""),
            "ic_family": p.get("ic_family", ""),
        }

    def sample_sim_params(
        self,
        rng: np.random.Generator,
        rng_profile: np.random.Generator,
        grids: dict[str, np.ndarray],
        time_cfg: dict[str, Any],
    ) -> list[dict]:
        X = grids["X"]
        Y = grids["Y"]
        y_grid = grids.get("y_grid")
        if y_grid is None:
            c, d = float(np.min(Y)), float(np.max(Y))
        else:
            c, d = float(y_grid[0]), float(y_grid[-1])
        num_sims = int(time_cfg["num_sims"])
        dt = float(time_cfg["dt"])
        t_final = float(time_cfg["t_final"])
        b_temp = float(time_cfg.get("b", 1.0))
        T_right = float(time_cfg.get("T_right", 300.0))
        temporal_window = dict(
            t_on=float(time_cfg.get("t_on", 0.0)),
            t_off=float(time_cfg.get("t_off", 0.2)),
            phase=float(time_cfg.get("phase", 0.0)),
            tukey_alpha=float(time_cfg.get("tukey_alpha", 0.5)),
        )




        assignment = time_cfg.get("ic_family_assignment")
        if assignment is not None and len(assignment) != num_sims:
            raise ValueError(
                f"ic_family_assignment length {len(assignment)} != num_sims "
                f"{num_sims} for benchmark {self.name!r}."
            )

        sim_params = []
        for i in range(num_sims):
            if assignment is not None:
                ic_family = str(assignment[i])
            else:
                ic_family = str(rng.choice(list(IC_FAMILIES.keys())))
            if ic_family not in IC_FAMILIES:
                raise ValueError(
                    f"ic_family {ic_family!r} for sim {i} is not a known family "
                    f"{tuple(IC_FAMILIES)}."
                )
            ic_params = IC_SAMPLERS[ic_family](rng)
            T0 = build_ic(ic_family, ic_params, X, Y, T_right=T_right, b=b_temp)

            temporal_family = FIXED_TEMPORAL_FAMILY
            temporal_params = TEMPORAL_SAMPLERS[temporal_family](
                rng_profile, dt=dt, t_final=t_final, **temporal_window
            )
            spatial_family = FIXED_SPATIAL_FAMILY
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

    def sample_online_params(
        self,
        rng: np.random.Generator,
        n: int,
        grids: dict[str, np.ndarray],
        time_cfg: dict[str, Any],
        rng_profile: np.random.Generator | None = None,
        *,
        rng_streams: dict[str, np.random.Generator] | None = None,
        ic_family_assignment: list[str] | tuple[str, ...] | None = None,
    ) -> list[dict]:
        streams = rng_streams or {}
        ic_family_rng = streams.get("ic_family", rng)
        ic_param_rng = streams.get("ic_params", rng)
        forcing_rng = streams.get(
            "forcing_params", rng_profile if rng_profile is not None else rng,
        )
        X = np.asarray(grids["X"], dtype=np.float64)
        Y = np.asarray(grids["Y"], dtype=np.float64)
        y_grid = np.asarray(grids.get("y_grid", Y[0]), dtype=np.float64)
        c, d = float(y_grid[0]), float(y_grid[-1])
        dt = float(time_cfg["dt"])
        t_final = float(time_cfg["t_final"])
        T_right = float(time_cfg.get("T_right", 300.0))
        b_temp = float(time_cfg.get("b", 1.0))
        temporal_window = {
            "t_on": float(time_cfg.get("t_on", 0.0)),
            "t_off": float(time_cfg.get("t_off", 0.2)),
            "phase": float(time_cfg.get("phase", 0.0)),
            "tukey_alpha": float(time_cfg.get("tukey_alpha", 0.5)),
        }
        if ic_family_assignment is not None and len(ic_family_assignment) != int(n):
            raise ValueError(
                "ic_family_assignment length must equal the online batch size"
            )

        records: list[dict] = []
        for i in range(int(n)):
            family = (
                str(ic_family_assignment[i])
                if ic_family_assignment is not None
                else sample_ic_family(ic_family_rng)
            )
            if family not in IC_FAMILIES:
                raise ValueError(f"unknown IC family {family!r}")
            ic_params = canonical_ic_params(
                family, IC_SAMPLERS[family](ic_param_rng),
            )
            temporal_params = TEMPORAL_SAMPLERS[FIXED_TEMPORAL_FAMILY](
                forcing_rng, dt=dt, t_final=t_final, **temporal_window,
            )
            spatial_params = SPATIAL_SAMPLERS[FIXED_SPATIAL_FAMILY](
                forcing_rng, c=c, d=d,
            )
            records.append({
                "T0": build_ic(
                    family, ic_params, X, Y, T_right=T_right, b=b_temp,
                ),
                "ic_family": family,
                "ic_params": ic_params,
                "temporal_family": FIXED_TEMPORAL_FAMILY,
                "temporal_params": temporal_params,
                "spatial_family": FIXED_SPATIAL_FAMILY,
                "spatial_params": spatial_params,
            })
        return records

    def validate_schema(self, sim_params: np.ndarray, sim_ids: np.ndarray) -> None:
        required = ("T0", "ic_family", "ic_params", "temporal_family", "temporal_params",
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
        for sid in sim_ids:
            entry = sim_params[int(sid)]
            if entry["temporal_family"] != FIXED_TEMPORAL_FAMILY:
                raise ValueError(
                    f"sim_params[{int(sid)}] has temporal_family="
                    f"{entry['temporal_family']!r}; benchmark {self.name!r} admits "
                    f"only {FIXED_TEMPORAL_FAMILY!r}."
                )
            if entry["spatial_family"] != FIXED_SPATIAL_FAMILY:
                raise ValueError(
                    f"sim_params[{int(sid)}] has spatial_family="
                    f"{entry['spatial_family']!r}; benchmark {self.name!r} admits "
                    f"only {FIXED_SPATIAL_FAMILY!r}."
                )
            if entry["ic_family"] not in IC_FAMILIES:
                raise ValueError(
                    f"sim_params[{int(sid)}] has ic_family="
                    f"{entry['ic_family']!r}; benchmark {self.name!r} admits only "
                    f"{tuple(IC_FAMILIES)}."
                )
