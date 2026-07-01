from __future__ import annotations

from typing import Any

import numpy as np

from problems.base import OODAxis, ProblemDims, ProblemSpec, empty_forcing_seq
from problems.forcing import (
    FORCING_TEMPORAL_SAMPLES,
    FORCING_TEMPORAL_TOKEN_DIM,
    T_EPS,
    _forcing_seq_2tok_from_samples,
    _sample_a,
)
from src.physics.boundary_forcing import (
    FORCING_BINS,
    SIN_AMP_RANGE,
    SPATIAL_BUILDERS,
    SPATIAL_SAMPLERS,
    TEMPORAL_SAMPLERS,
    build_qL,
    default_ramp_seconds,
    integrate_temporal_bins_ramped_signed,
    ramped_temporal,
)
from src.physics.fv_solver_2d import FVSolver2D, Layer2D
from src.physics.init_conditions import IC_SAMPLERS, build_ic, sample_ic_family

# ----- representation constants (canonical home for the interfaces benchmark) ---

RC_RANGE = (0.05, 1.0)
INTERFACE_X_RANGE = (0.2, 0.8)

COND_STATIC_DIM = 4
A_AMP_REF = float(SIN_AMP_RANGE[1])
K_LEFT = 2.0
K_RIGHT = 1.0

# Spatial channels per representation. temporal_encoder mode lifts the lean
# [T_tilde, x, y, K_norm, D_norm, s_y] base; bins mode appends FORCING_BINS
# integral Q-bins.
SPATIAL_CHANNELS_TEMPORAL = 6
SPATIAL_CHANNELS_BINS = SPATIAL_CHANNELS_TEMPORAL + FORCING_BINS  # 22

# s_y is the 6th spatial channel (index 5) in both representations; the model's
# learned forcing injection must read this channel, not K_norm at index 3.
S_Y_CHANNEL = 5

# 2D LHS column order; must match generate_lhs_samples in vary-interfaces.
LHS_PARAM_RANGES = {"R_c": RC_RANGE, "interface_x": INTERFACE_X_RANGE}
_NODE_TOL_FRAC = 1e-3
_NODE_JITTER_FRAC = 0.25

# In-distribution generation pins sin/uniform forcing. The family_transfer OOD
# axis is the one place that drives the other trained families, so the schema
# guard admits these (and only these) families rather than a single value.
ALLOWED_TEMPORAL_FAMILIES = ("sin", "exp")
ALLOWED_SPATIAL_FAMILIES = ("uniform", "patch")


# ----- conditioning -----

def _physical_interface_range(a: float, b: float) -> tuple[float, float]:
    span = float(b) - float(a)
    return (
        float(a) + INTERFACE_X_RANGE[0] * span,
        float(a) + INTERFACE_X_RANGE[1] * span,
    )


def build_cond_vector(t_bar_norm: float, t_s_norm: float, R_c: float,
                      interface_x: float,
                      interface_x_range: tuple[float, float] = INTERFACE_X_RANGE) -> np.ndarray:
    """Assemble the 4-dim static conditioning vector.

    Layout: [t_bar_norm, t_s_norm, R_c_norm, interface_x_norm].
    """
    R_c_norm = (R_c - RC_RANGE[0]) / (RC_RANGE[1] - RC_RANGE[0])
    x_lo, x_hi = map(float, interface_x_range)
    interface_x_norm = (interface_x - x_lo) / (x_hi - x_lo)
    return np.array(
        [t_bar_norm, t_s_norm, R_c_norm, interface_x_norm], dtype=np.float32
    )


# ----- sampling helpers (bit-parity with vary-interfaces generate_dataset) -----

def _generate_lhs_samples(num_sims: int, seed: int = 0) -> np.ndarray:
    """2D LHS over (R_c, interface_x). Returns (num_sims, 2) float32."""
    param_names = list(LHS_PARAM_RANGES.keys())
    sample_dim = len(param_names)
    lower_bounds = np.array([LHS_PARAM_RANGES[n][0] for n in param_names], dtype=np.float32)
    upper_bounds = np.array([LHS_PARAM_RANGES[n][1] for n in param_names], dtype=np.float32)

    rng = np.random.default_rng(seed)
    edges = np.linspace(0.0, 1.0, num_sims + 1, dtype=np.float64)
    widths = edges[1:] - edges[:-1]
    samples_unit = edges[:-1, None] + widths[:, None] * rng.random((num_sims, sample_dim))
    for j in range(sample_dim):
        rng.shuffle(samples_unit[:, j])

    samples_scaled = lower_bounds + samples_unit * (upper_bounds - lower_bounds)
    return samples_scaled.astype(np.float32)


def _jitter_off_node(interface_x: np.ndarray, a: float, b: float, Nx: int) -> np.ndarray:
    """Nudge any interface_x sitting on a grid node into the adjacent cell."""
    hx = (b - a) / (Nx - 1)
    node_tol = _NODE_TOL_FRAC * hx
    out = interface_x.copy()
    for i, x in enumerate(out):
        k = round((x - a) / hx)
        x_node = a + k * hx
        if abs(x - x_node) < node_tol:
            face_idx = int(np.floor((x - a) / hx))
            if face_idx >= Nx - 1:
                face_idx = Nx - 2
            x_left = a + face_idx * hx
            x_center = x_left + 0.5 * hx
            direction = 1.0 if x_center > x else -1.0
            out[i] = float(x + direction * _NODE_JITTER_FRAC * hx)
    return out


class InterfacesProblem(ProblemSpec):
    """Varying-interface benchmark: fixed sin/uniform forcing, interface_x
    sampled in [0.2, 0.8].

    Two representations over the same trajectories/sim_params:

    - ``temporal_encoder``: 6 spatial channels [T_tilde, x, y, K_norm, D_norm,
      s_y], 4 static conditioning dims, a (128, 2) forcing_seq [r_m, a_m/A_ref],
      temporal encoder on with time-augmented spatial injection.
    - ``bins``: 22 spatial channels [..., s_y, Q_y_bin_0..15], same 4 static
      dims, an empty forcing_seq, temporal encoder off.

    s_y lives at spatial channel 5 in both modes; the 3-dim T_stats carries
    interface_x in slot 2 for dynamic mask building.
    """

    name = "interfaces"

    def __init__(self, representation: str = "temporal_encoder"):
        self.representation = representation
        if representation == "temporal_encoder":
            self.dims = ProblemDims(
                in_channels=SPATIAL_CHANNELS_TEMPORAL,
                cond_static_dim=COND_STATIC_DIM,
                has_forcing_seq=True,
                temporal_token_dim=FORCING_TEMPORAL_TOKEN_DIM,
                t_stats_dim=3,
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
                t_stats_dim=3,
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
        Y = grids["Y"]
        x_grid = grids["x_grid"]
        y_grid = grids["y_grid"]
        Nx, Ny = X.shape[0], X.shape[1]
        a = float(x_grid[0])
        b = float(x_grid[-1])
        c = float(y_grid[0])
        d = float(y_grid[-1])

        num_sims = int(time_cfg["num_sims"])
        dt = float(time_cfg["dt"])
        t_final = float(time_cfg["t_final"])
        b_temp = float(time_cfg.get("b", 1.0))
        T_right = float(time_cfg.get("T_right", 300.0))
        lhs_seed = int(time_cfg.get("lhs_seed", 0))
        temporal_window = dict(
            t_on=float(time_cfg.get("t_on", 0.0)),
            t_off=float(time_cfg.get("t_off", 0.2)),
            phase=float(time_cfg.get("phase", 0.0)),
            tukey_alpha=float(time_cfg.get("tukey_alpha", 0.5)),
        )

        samples_scaled = _generate_lhs_samples(num_sims, seed=lhs_seed)
        R_c_values = samples_scaled[:, 0]
        interface_x_values = a + samples_scaled[:, 1] * (b - a)
        interface_x_values = _jitter_off_node(interface_x_values, a=a, b=b, Nx=Nx)

        sim_params = []
        for i in range(num_sims):
            R_c = float(R_c_values[i])
            interface_x = float(interface_x_values[i])

            ic_family = sample_ic_family(rng)
            ic_params = IC_SAMPLERS[ic_family](rng, Nx=Nx, Ny=Ny)
            T0 = build_ic(ic_family, ic_params, X, Y, T_right=T_right, b=b_temp)

            temporal_family = "sin"
            temporal_params = TEMPORAL_SAMPLERS["sin"](
                rng_profile, dt=dt, t_final=t_final, **temporal_window
            )

            spatial_family = "uniform"
            spatial_params = SPATIAL_SAMPLERS["uniform"](rng_profile, c=c, d=d)

            sim_params.append({
                "R_c": R_c,
                "interface_x": interface_x,
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
        x_I = float(params["interface_x"])
        layers = [
            Layer2D(x_left=a, x_right=x_I, rho=1.0, cp=1.0, k=K_LEFT),
            Layer2D(x_left=x_I, x_right=b, rho=1.0, cp=1.0, k=K_RIGHT),
        ]
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
        return FVSolver2D(
            a=base_kwargs["a"], b=base_kwargs["b"],
            c=base_kwargs["c"], d=base_kwargs["d"],
            Nx=base_kwargs["Nx"], Ny=base_kwargs["Ny"],
            lam_target=base_kwargs["lam_target"],
            layers=layers,
            t_final=base_kwargs["t_final"],
            flux_f=base_kwargs["flux_f"], flux_A=base_kwargs["flux_A"],
            t_on=base_kwargs["t_on"], t_off=base_kwargs["t_off"],
            phase=base_kwargs["phase"],
            dt=base_kwargs["dt"], tukey_alpha=base_kwargs["tukey_alpha"],
            interface_R=[params["R_c"]],
            q_left_fn=q_left_fn,
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

    def _material_channel(self, ds, interface_x: float) -> np.ndarray:
        k_x = np.where(ds.x_grid <= float(interface_x), K_LEFT, K_RIGHT).astype(np.float32)
        k_norm = (k_x - np.float32(0.5 * (K_LEFT + K_RIGHT))) / np.float32(0.5 * (K_LEFT - K_RIGHT))
        return np.broadcast_to(k_norm[:, None], (ds.Nx, ds.Ny)).astype(np.float32)

    def _signed_distance_channel(self, ds, interface_x: float) -> np.ndarray:
        extent = np.float32(ds.x_grid[-1] - ds.x_grid[0])
        d = (ds.x_grid - np.float32(interface_x)) / extent
        return np.broadcast_to(d[:, None], (ds.Nx, ds.Ny)).astype(np.float32)

    def build_item(self, ds, sid: int, s: int, j: int) -> dict[str, np.ndarray]:
        sid = int(sid)
        params = ds.sim_params[sid]
        R_c = float(params["R_c"])
        interface_x = float(params["interface_x"])

        T_source = ds.trajectories[sid, s, :, :]
        T_target = ds.trajectories[sid, j, :, :]

        T_source_norm = (T_source - ds.mu_global) / (ds.sigma_global + T_EPS)
        T_target_norm = (T_target - ds.mu_global) / (ds.sigma_global + T_EPS)

        if ds.noise_std > 0:
            T_source_norm = T_source_norm + np.random.randn(*T_source_norm.shape).astype(np.float32) * ds.noise_std

        t_s_val = float(ds.t_grid[s])
        t_j_val = float(ds.t_grid[j])
        t_bar_norm = (t_j_val - t_s_val) / ds.time_norm_horizon
        t_s_norm = t_s_val / ds.time_norm_horizon

        s_y = ds.s_y_profiles[sid]
        S_y = np.broadcast_to(s_y[None, :], (ds.Nx, ds.Ny))
        K_norm = self._material_channel(ds, interface_x)
        D_norm = self._signed_distance_channel(ds, interface_x)
        spatial_base = np.stack(
            [T_source_norm, ds.X_norm, ds.Y_norm, K_norm, D_norm, S_y], axis=-1,
        ).astype(np.float32)

        cond_static = build_cond_vector(
            t_bar_norm=float(t_bar_norm),
            t_s_norm=float(t_s_norm),
            R_c=R_c,
            interface_x=interface_x,
            interface_x_range=_physical_interface_range(
                float(ds.x_grid[0]), float(ds.x_grid[-1])
            ),
        )

        Y = T_target_norm[:, :, None].astype(np.float32)
        T_stats = np.array([ds.mu_global, ds.sigma_global, interface_x], dtype=np.float32)

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

    # ---- val-pair logging ----

    val_pair_fields = ("interface_x",)

    def val_pair_row(self, ds, sim_id: int, s: int, j: int) -> dict[str, Any]:
        p = ds.sim_params[int(sim_id)]
        return {"interface_x": float(p["interface_x"])}

    # ---- schema / labels ----

    def validate_schema(self, sim_params: np.ndarray, sim_ids: np.ndarray) -> None:
        required = ("R_c", "interface_x", "temporal_family", "temporal_params",
                    "spatial_family", "spatial_params")
        for sid in sim_ids:
            entry = sim_params[int(sid)]
            missing = [k for k in required if k not in entry]
            if missing:
                raise ValueError(
                    f"sim_params[{int(sid)}] missing keys {missing} for benchmark "
                    f"{self.name!r}."
                )
            if entry["temporal_family"] not in ALLOWED_TEMPORAL_FAMILIES:
                raise ValueError(
                    f"sim_params[{int(sid)}] has temporal_family="
                    f"{entry['temporal_family']!r}; benchmark {self.name!r} "
                    f"admits only {ALLOWED_TEMPORAL_FAMILIES}."
                )
            if entry["spatial_family"] not in ALLOWED_SPATIAL_FAMILIES:
                raise ValueError(
                    f"sim_params[{int(sid)}] has spatial_family="
                    f"{entry['spatial_family']!r}; benchmark {self.name!r} "
                    f"admits only {ALLOWED_SPATIAL_FAMILIES}."
                )

    # ---- OOD hooks ----

    def ood_axes(self) -> dict[str, OODAxis]:
        return {
            "rc": OODAxis(
                name="rc", kind="simulation_parameter", field="R_c",
                id_reference=(0.5,), ood_values=(0.0, 0.01, 0.025, 1.25, 1.5, 2.0),
                trained_range=RC_RANGE, resolution_kind="none",
                notes="scalar interface resistance; same sweep as the forcing "
                "benchmark. The interface location and forcing stay fixed at the "
                "trained sin/uniform background; only R_c is swept.",
            ),
            "interface_x": OODAxis(
                name="interface_x",
                kind="simulation_parameter",
                id_reference=(0.5,),
                ood_values=(0.1, 0.9),
                field="interface_x",
                trained_range=(float(INTERFACE_X_RANGE[0]), float(INTERFACE_X_RANGE[1])),
                resolution_kind="none",
                notes=(
                    "Interface fraction in [0,1]; converted to a physical "
                    "position and face-aligned via _jitter_off_node. The "
                    "requested fraction and the snapped physical value are "
                    "recorded under _ood_realized."
                ),
            ),
            "family_transfer": OODAxis(
                name="family_transfer",
                kind="compound",
                id_reference=(("sin", "uniform"),),
                ood_values=(
                    ("exp", "uniform"),
                    ("sin", "patch"),
                    ("exp", "patch"),
                ),
                field="forcing_family",
                pinned_family="sin/uniform",
                compound=True,
                resolution_kind="none",
                notes=(
                    "Full 2x2 temporal x spatial family grid against the "
                    "trained sin/uniform forcing. Non-family background (R_c, "
                    "interface_x, IC) is shared per repeat; CRN pairing is on "
                    "the shared-latent fingerprint, not byte-equal params."
                ),
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
        Y = grids["Y"]
        x_grid = grids["x_grid"]
        y_grid = grids["y_grid"]
        Nx, Ny = X.shape[0], X.shape[1]
        a = float(x_grid[0])
        b = float(x_grid[-1])
        c = float(y_grid[0])
        d = float(y_grid[-1])

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

        # Shared background reused by every swept value of this repeat.
        R_c = float(rng.uniform(*RC_RANGE))
        ic_family = sample_ic_family(rng)
        ic_params = IC_SAMPLERS[ic_family](rng, Nx=Nx, Ny=Ny)
        T0 = build_ic(ic_family, ic_params, X, Y, T_right=T_right, b=b_temp)

        x_lo, x_hi = INTERFACE_X_RANGE
        interface_frac_bg = float(rng.uniform(x_lo, x_hi))
        interface_x_bg = float(
            _jitter_off_node(
                np.array([a + interface_frac_bg * (b - a)], dtype=np.float64),
                a=a, b=b, Nx=Nx,
            )[0]
        )

        # One per-repeat seed; family_transfer re-seeds from it so each family
        # samples from the same background stream (CRN coupling on the seed).
        profile_seed = int(rng_profile.integers(0, 2**31 - 1))
        fixed_temporal = TEMPORAL_SAMPLERS["sin"](
            np.random.default_rng(profile_seed),
            dt=dt, t_final=t_final, **temporal_window,
        )
        fixed_spatial = SPATIAL_SAMPLERS["uniform"](
            np.random.default_rng(profile_seed), c=c, d=d,
        )

        return {
            "domain": (a, b, c, d),
            "Nx": Nx,
            "Ny": Ny,
            "dt": dt,
            "t_final": t_final,
            "temporal_window": temporal_window,
            "R_c": R_c,
            "interface_x_bg": interface_x_bg,
            "T0": T0,
            "ic_family": ic_family,
            "ic_params": ic_params,
            "profile_seed": profile_seed,
            "fixed_temporal": fixed_temporal,
            "fixed_spatial": fixed_spatial,
        }

    def apply_ood_value(
        self, latents: dict[str, Any], axis: OODAxis, value: Any,
    ) -> dict[str, Any]:
        a, b, c, d = latents["domain"]
        Nx = latents["Nx"]
        dt = latents["dt"]
        t_final = latents["t_final"]
        temporal_window = latents["temporal_window"]

        params: dict[str, Any] = {
            "R_c": float(latents["R_c"]),
            "interface_x": float(latents["interface_x_bg"]),
            "T0": latents["T0"],
            "ic_family": latents["ic_family"],
            "ic_params": latents["ic_params"],
            "temporal_family": "sin",
            "temporal_params": latents["fixed_temporal"],
            "spatial_family": "uniform",
            "spatial_params": latents["fixed_spatial"],
        }

        if axis.name == "rc":
            params["R_c"] = float(value)
            params["_ood_realized"] = {"requested": float(value)}
            return params

        if axis.name == "interface_x":
            frac = float(value)
            interface_x_req = a + frac * (b - a)
            interface_x_actual = float(
                _jitter_off_node(
                    np.array([interface_x_req], dtype=np.float64),
                    a=a, b=b, Nx=Nx,
                )[0]
            )
            params["interface_x"] = interface_x_actual
            params["_ood_realized"] = {
                "requested": frac,
                "interface_x_requested": float(interface_x_req),
                "interface_x_actual": interface_x_actual,
            }
            return params

        if axis.name == "family_transfer":
            temporal_family, spatial_family = value
            if temporal_family not in ALLOWED_TEMPORAL_FAMILIES:
                raise ValueError(
                    f"family_transfer temporal_family={temporal_family!r} not in "
                    f"{ALLOWED_TEMPORAL_FAMILIES}."
                )
            if spatial_family not in ALLOWED_SPATIAL_FAMILIES:
                raise ValueError(
                    f"family_transfer spatial_family={spatial_family!r} not in "
                    f"{ALLOWED_SPATIAL_FAMILIES}."
                )
            seed = latents["profile_seed"]
            temporal_params = TEMPORAL_SAMPLERS[temporal_family](
                np.random.default_rng(seed),
                dt=dt, t_final=t_final, **temporal_window,
            )
            spatial_params = SPATIAL_SAMPLERS[spatial_family](
                np.random.default_rng(seed), c=c, d=d,
            )
            params["temporal_family"] = temporal_family
            params["temporal_params"] = temporal_params
            params["spatial_family"] = spatial_family
            params["spatial_params"] = spatial_params
            params["_ood_realized"] = {
                "requested": f"{temporal_family}+{spatial_family}",
                "temporal_family": temporal_family,
                "spatial_family": spatial_family,
            }
            return params

        raise ValueError(
            f"Unknown OOD axis {axis.name!r} for benchmark {self.name!r}."
        )

    def plot_label(self, params: dict) -> str:
        interface_x = params.get("interface_x")
        tp = params.get("temporal_params", {})
        A = tp.get("A")
        f = tp.get("f")
        bits = []
        if interface_x is not None:
            bits.append(f"x_I={interface_x:.2f}")
        if A is not None and f is not None:
            bits.append(f"rectified sin A={A:.0f} f={f:.1f}")
        return " ".join(bits) if bits else self.name
