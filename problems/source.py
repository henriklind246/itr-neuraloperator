from __future__ import annotations

import copy
from typing import Any

import numpy as np

from problems.base import OODAxis, ProblemDims, ProblemSpec
from problems.forcing import (
    FORCING_TEMPORAL_SAMPLES,
    FORCING_TEMPORAL_TOKEN_DIM,
    T_EPS,
    _forcing_seq_3tok_from_samples,
    _sample_a,
)
import src.physics.internal_source as _internal_source
from src.physics.internal_source import (
    PATCH_H,
    PATCH_W,
    PATCH_X_RANGE,
    PATCH_Y_RANGE,
    build_patch_source,
    integrate_sin2_pulse,
    make_patch_indicator,
    make_sin2_pulse,
)
from src.physics.fv_solver_2d import FVSolver2D, Layer2D

# ----- representation constants (canonical home for the source benchmark) ------

RC_RANGE = (0.05, 1.0)
INTERFACE_X = 0.5
T_OFF_FRAC = 0.75

# Millimetres, seconds, kg, and K: representative Ti-6Al-4V / cartridge brass.
# k = 6.97 / 112 W/(m K), rho = 4440 / 8530 kg/m^3, cp = 523 / 377 J/(kg K).
K_LEFT, K_RIGHT = 0.00697, 0.112
RHO_LEFT, RHO_RIGHT = 4.44e-6, 8.53e-6
CP_LEFT, CP_RIGHT = 523.0, 377.0
PROBLEM_VERSION = "source_ti_brass_mm_v1"
MATERIAL_PROPERTIES = {
    "k_left": K_LEFT, "k_right": K_RIGHT,
    "rho_left": RHO_LEFT, "rho_right": RHO_RIGHT,
    "cp_left": CP_LEFT, "cp_right": CP_RIGHT,
    "length_unit": "mm", "time_unit": "s", "mass_unit": "kg",
}

COND_STATIC_DIM = 6  # [lead, R_c] + [x_h, y_h, w_h, h_h]; no source amplitude leak

# Spatial-descriptor conditioning ablation slice (single source of truth for the
# mask helper and the tests). cond layout is [t_bar_norm, R_c_norm, x_h_norm,
# y_h_norm, w_h_norm, h_h_norm]; the patch-geometry descriptor is [2:6].
SOURCE_SPATIAL_DESCRIPTOR_SLICE = slice(2, COND_STATIC_DIM)  # slice(2, 6)

# The lift sees the lean [T_tilde, x, y, S_h] base.
SPATIAL_CHANNELS_TEMPORAL = 4

# S_h is the 4th spatial channel (index 3).
S_Y_CHANNEL = 3

# Stratified-sampling shares for x_h relative to the interface (left/near/right).
STRATIFIED_SHARES = {"left": 0.35, "near": 0.30, "right": 0.35}


def _patch_a_range() -> tuple[float, float]:
    """Resolve PATCH_A_RANGE through the module so calibration-time mutation is
    visible regardless of import order."""
    if _internal_source.PATCH_A_RANGE is None:
        raise RuntimeError(
            "PATCH_A_RANGE is None. Run scripts/calibrate_patch_amplitude.py and "
            "set src.physics.internal_source.PATCH_A_RANGE before constructing "
            "datasets."
        )
    return _internal_source.PATCH_A_RANGE


# ----- conditioning -----

def build_cond_vector(
    t_bar_norm: float,
    R_c: float,
    x_h: float,
    y_h: float,
    w_h: float,
    h_h: float,
    *,
    x_center_range: tuple[float, float] = PATCH_X_RANGE,
    y_center_range: tuple[float, float] = PATCH_Y_RANGE,
    x_length_scale: float = 1.0,
    y_length_scale: float = 1.0,
) -> np.ndarray:
    """Assemble the 6-dim static conditioning vector.

    Layout: [t_bar_norm, R_c_norm, x_h_norm, y_h_norm, w_h_norm,
    h_h_norm]. The source amplitude A is deliberately excluded from
    conditioning; the heating signal is carried by the temporal token stream.
    """
    R_c_norm = (R_c - RC_RANGE[0]) / (RC_RANGE[1] - RC_RANGE[0])
    x_h_norm = (x_h - x_center_range[0]) / (x_center_range[1] - x_center_range[0])
    y_h_norm = (y_h - y_center_range[0]) / (y_center_range[1] - y_center_range[0])
    w_h_norm = w_h / x_length_scale
    h_h_norm = h_h / y_length_scale
    return np.array(
        [t_bar_norm, R_c_norm, x_h_norm, y_h_norm, w_h_norm, h_h_norm],
        dtype=np.float32,
    )


# ----- sampling helpers (bit-parity with source-itr generate_dataset) ----------

def _build_lhs_param_ranges() -> dict:
    A_min, A_max = _patch_a_range()
    return {
        "R_c": (0.05, 1.0),
        "y_h": PATCH_Y_RANGE,
        "log_A": (float(np.log(A_min)), float(np.log(A_max))),
    }


def _classify_regime(x_h: float, x_I: float, w: float) -> str:
    half = 0.5 * w
    if x_h + half < x_I:
        return "left"
    if x_h - half > x_I:
        return "right"
    return "near"


def _lhs_unit(num_sims: int, n_dims: int, rng) -> np.ndarray:
    edges = np.linspace(0.0, 1.0, num_sims + 1, dtype=np.float64)
    widths = edges[1:] - edges[:-1]
    samples = edges[:-1, None] + widths[:, None] * rng.random((num_sims, n_dims))
    for j in range(n_dims):
        rng.shuffle(samples[:, j])
    return samples


def _sample_x_h_stratified(num_sims: int, x_I: float, w: float,
                           x_range: tuple[float, float], rng) -> np.ndarray:
    half = 0.5 * w
    a, b = x_range
    left_hi = x_I - half
    near_lo = x_I - half
    near_hi = x_I + half
    right_lo = x_I + half

    left_lo, left_hi = a, min(left_hi, b)
    near_lo, near_hi = max(near_lo, a), min(near_hi, b)
    right_lo, right_hi = max(right_lo, a), b

    counts = {
        "left": int(round(STRATIFIED_SHARES["left"] * num_sims)),
        "near": int(round(STRATIFIED_SHARES["near"] * num_sims)),
    }
    counts["right"] = num_sims - counts["left"] - counts["near"]

    samples = []
    for regime, (lo, hi) in [
        ("left", (left_lo, left_hi)),
        ("near", (near_lo, near_hi)),
        ("right", (right_lo, right_hi)),
    ]:
        n = counts[regime]
        if n <= 0:
            continue
        u = _lhs_unit(n, 1, rng)[:, 0]
        samples.append(lo + u * (hi - lo))

    out = np.concatenate(samples).astype(np.float32)
    rng.shuffle(out)
    return out


def _generate_lhs_samples(num_sims: int, seed: int = 0,
                          stratified_x_h: bool = True) -> dict:
    ranges = _build_lhs_param_ranges()
    rng = np.random.default_rng(seed)

    other_dims = ["R_c", "y_h", "log_A"]
    if not stratified_x_h:
        other_dims = ["R_c", "x_h", "y_h", "log_A"]
        ranges["x_h"] = PATCH_X_RANGE
    u = _lhs_unit(num_sims, len(other_dims), rng)
    scaled = {}
    for j, name in enumerate(other_dims):
        lo, hi = ranges[name]
        scaled[name] = (lo + u[:, j] * (hi - lo)).astype(np.float32)

    if stratified_x_h:
        scaled["x_h"] = _sample_x_h_stratified(
            num_sims=num_sims, x_I=INTERFACE_X, w=PATCH_W,
            x_range=PATCH_X_RANGE, rng=rng,
        )

    scaled["A"] = np.exp(scaled.pop("log_A")).astype(np.float32)
    return scaled


def _scale_fraction(value: float, lo: float, hi: float) -> float:
    return float(lo) + float(value) * (float(hi) - float(lo))


def _patch_center_ranges(
    a: float,
    b: float,
    c: float,
    d: float,
    w_h: float,
    h_h: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    return (
        (float(a) + 0.5 * float(w_h), float(b) - 0.5 * float(w_h)),
        (float(c) + 0.5 * float(h_h), float(d) - 0.5 * float(h_h)),
    )


def _validate_patch_bounds(
    *,
    x_h: float,
    y_h: float,
    w_h: float,
    h_h: float,
    a: float,
    b: float,
    c: float,
    d: float,
) -> None:
    x_min = float(x_h) - 0.5 * float(w_h)
    x_max = float(x_h) + 0.5 * float(w_h)
    y_min = float(y_h) - 0.5 * float(h_h)
    y_max = float(y_h) + 0.5 * float(h_h)
    if x_min < float(a) or x_max > float(b) or y_min < float(c) or y_max > float(d):
        raise ValueError(
            "Source patch extends outside the domain: "
            f"x=[{x_min}, {x_max}] vs [{a}, {b}], "
            f"y=[{y_min}, {y_max}] vs [{c}, {d}]."
        )


class SourceProblem(ProblemSpec):
    """Internal volumetric chip-heating patch benchmark.

    4 spatial channels [T_tilde, x, y, S_h], 6 static conditioning dims, and a
    (128, 3) forcing_seq carrying point samples and local interval averages of
    the sin^2 heating pulse.

    The source amplitude A never enters conditioning. Interface fixed at
    x = 0.5.
    """

    name = "source"
    problem_version = PROBLEM_VERSION

    spatial_descriptor_cond_slice = SOURCE_SPATIAL_DESCRIPTOR_SLICE

    def __init__(self, representation: str = "temporal_encoder"):
        self.representation = representation
        if representation == "temporal_encoder":
            self.dims = ProblemDims(
                in_channels=SPATIAL_CHANNELS_TEMPORAL,
                cond_static_dim=COND_STATIC_DIM,
                temporal_token_dim=FORCING_TEMPORAL_TOKEN_DIM,
                t_stats_dim=3,
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
        x_grid = grids["x_grid"]
        y_grid = grids["y_grid"]
        a, b = float(x_grid[0]), float(x_grid[-1])
        c, d = float(y_grid[0]), float(y_grid[-1])
        Lx, Ly = b - a, d - c
        interface_x = _scale_fraction(INTERFACE_X, a, b)
        w_h = PATCH_W * Lx
        h_h = PATCH_H * Ly

        num_sims = int(time_cfg["num_sims"])
        t_final = float(time_cfg["t_final"])
        lhs_seed = int(time_cfg.get("lhs_seed", 0))
        stratified_x_h = bool(time_cfg.get("stratified_x_h", True))

        samples = _generate_lhs_samples(
            num_sims=num_sims, seed=lhs_seed, stratified_x_h=stratified_x_h
        )
        t_off = T_OFF_FRAC * t_final

        sim_params = []
        for i in range(num_sims):
            R_c = float(samples["R_c"][i])
            x_h = _scale_fraction(float(samples["x_h"][i]), a, b)
            y_h = _scale_fraction(float(samples["y_h"][i]), c, d)
            A = float(samples["A"][i])
            regime = _classify_regime(x_h, interface_x, w_h)
            _validate_patch_bounds(
                x_h=x_h, y_h=y_h, w_h=w_h, h_h=h_h,
                a=a, b=b, c=c, d=d,
            )

            ic_family = "uniform_2d"
            ic_params = {"T0_offset": 0.0}
            T0 = np.full(X.shape, 300.0, dtype=np.float32)

            sim_params.append({
                "material_properties": dict(MATERIAL_PROPERTIES),
                "R_c": R_c,
                "interface_x": interface_x,
                "x_h": x_h,
                "y_h": y_h,
                "w_h": w_h,
                "h_h": h_h,
                "A": A,
                "t_off": t_off,
                "regime": regime,
                "T0": T0,
                "ic_family": ic_family,
                "ic_params": ic_params,
            })

        return sim_params

    def configure_solver(self, params: dict, base_kwargs: dict) -> FVSolver2D:
        y_grid = base_kwargs["y_grid"]
        x_I = float(params["interface_x"])
        a = float(base_kwargs["a"])
        b = float(base_kwargs["b"])
        layers = [
            Layer2D(x_left=a, x_right=x_I, rho=RHO_LEFT, cp=CP_LEFT, k=K_LEFT),
            Layer2D(x_left=x_I, x_right=b, rho=RHO_RIGHT, cp=CP_RIGHT, k=K_RIGHT),
        ]
        # Zero-flux left boundary in vector form so the solver's flux detection
        # uses the vector branch.
        q_left_fn = lambda t: np.zeros_like(y_grid)
        source = build_patch_source(
            x_h=params["x_h"], y_h=params["y_h"],
            w=params["w_h"], h=params["h_h"],
            A=params["A"], t_off=params["t_off"],
        )
        return FVSolver2D(
            a=base_kwargs["a"], b=base_kwargs["b"],
            c=base_kwargs["c"], d=base_kwargs["d"],
            Nx=base_kwargs["Nx"], Ny=base_kwargs["Ny"],
            lam_target=base_kwargs["lam_target"],
            layers=layers,
            t_final=base_kwargs["t_final"],
            flux_f=base_kwargs["flux_f"], flux_A=base_kwargs["flux_A"],
            t_on=base_kwargs["t_on"], t_off=float(params["t_off"]),
            phase=base_kwargs["phase"],
            dt=base_kwargs["dt"], tukey_alpha=base_kwargs["tukey_alpha"],
            interface_R=[params["R_c"]],
            q_left_fn=q_left_fn,
            source=source,
        )

    # ---- dataset item ----

    def setup_dataset(self, ds) -> None:
        ds._X, ds._Y = np.meshgrid(ds.x_grid, ds.y_grid, indexing="ij")
        ds._patch_mask_cache = {}
        # Force the 128-sample token grid regardless of the config default.
        ds.temporal_samples = FORCING_TEMPORAL_SAMPLES
        ds.a_amp_ref = float(_patch_a_range()[1])
        ds._q_callables = {}

    def _patch_mask(self, ds, sid: int) -> np.ndarray:
        if sid not in ds._patch_mask_cache:
            p = ds.sim_params[sid]
            ds._patch_mask_cache[sid] = make_patch_indicator(
                ds._X, ds._Y,
                float(p["x_h"]), float(p["y_h"]),
                float(p["w_h"]), float(p["h_h"]),
            )
        return ds._patch_mask_cache[sid]

    def build_item(self, ds, sid: int, s: int, j: int) -> dict[str, np.ndarray]:
        sid = int(sid)
        params = ds.sim_params[sid]
        R_c = float(params["R_c"])
        interface_x = float(params.get("interface_x", INTERFACE_X))

        T_source = ds.trajectories[sid, s, :, :]
        T_target = ds.trajectories[sid, j, :, :]

        T_source_norm = (T_source - ds.mu_global) / (ds.sigma_global + T_EPS)
        T_target_norm = (T_target - ds.mu_global) / (ds.sigma_global + T_EPS)

        if ds.noise_std > 0:
            T_source_norm = T_source_norm + np.random.randn(*T_source_norm.shape).astype(np.float32) * ds.noise_std

        t_s_val = float(ds.t_grid[s])
        t_j_val = float(ds.t_grid[j])
        t_bar_norm = (t_j_val - t_s_val) / ds.time_norm_horizon

        S_h = self._patch_mask(ds, sid)
        spatial_channels = [T_source_norm, ds.X_norm, ds.Y_norm, S_h]
        if getattr(self, "spatial_input_material_side", False):
            spatial_channels.append(
                self._material_side_channel(ds, interface_x)
            )
        spatial_base = np.stack(spatial_channels, axis=-1).astype(np.float32)

        cond_static = build_cond_vector(
            t_bar_norm=float(t_bar_norm),
            R_c=R_c,
            x_h=float(params["x_h"]), y_h=float(params["y_h"]),
            w_h=float(params["w_h"]), h_h=float(params["h_h"]),
            x_center_range=_patch_center_ranges(
                float(ds.x_grid[0]), float(ds.x_grid[-1]),
                float(ds.y_grid[0]), float(ds.y_grid[-1]),
                float(params["w_h"]), float(params["h_h"]),
            )[0],
            y_center_range=_patch_center_ranges(
                float(ds.x_grid[0]), float(ds.x_grid[-1]),
                float(ds.y_grid[0]), float(ds.y_grid[-1]),
                float(params["w_h"]), float(params["h_h"]),
            )[1],
            x_length_scale=float(ds.x_grid[-1] - ds.x_grid[0]),
            y_length_scale=float(ds.y_grid[-1] - ds.y_grid[0]),
        )
        cond_static = self._apply_spatial_conditioning_mask(cond_static)

        Y = T_target_norm[:, :, None].astype(np.float32)
        T_stats = np.array([ds.mu_global, ds.sigma_global, interface_x], dtype=np.float32)

        A = float(params["A"])
        t_off = float(params["t_off"])
        if sid not in ds._q_callables:
            ds._q_callables[sid] = make_sin2_pulse(A, t_off)
        q = ds._q_callables[sid]
        t_samples, a_m = _sample_a(q, t_s_val, t_j_val, ds.temporal_samples)
        forcing_seq = _forcing_seq_3tok_from_samples(
            t_samples,
            a_m,
            interval_integral_fn=lambda t_lo, t_hi: integrate_sin2_pulse(
                A, t_off, t_lo, t_hi
            ),
            A_amp_ref=ds.a_amp_ref,
        )

        return {
            "spatial": spatial_base,
            "cond_static": cond_static,
            "forcing_seq": forcing_seq,
            "Y": Y,
            "T_stats": T_stats,
        }

    # ---- val-pair logging ----

    val_pair_fields = ("x_h", "y_h", "A", "regime")

    def val_pair_row(self, ds, sim_id: int, s: int, j: int) -> dict[str, Any]:
        p = ds.sim_params[int(sim_id)]
        return {
            "x_h": float(p["x_h"]),
            "y_h": float(p["y_h"]),
            "A": float(p["A"]),
            "regime": p.get("regime", ""),
        }

    # ---- schema / labels ----

    def validate_schema(self, sim_params: np.ndarray, sim_ids: np.ndarray) -> None:
        required = ("R_c", "interface_x", "x_h", "y_h", "A", "t_off", "w_h", "h_h")
        forbidden = ("temporal_family", "temporal_params",
                     "spatial_family", "spatial_params")
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
                    f"sim_params[{int(sid)}] still contains legacy keys "
                    f"{present_legacy}; regenerate with benchmark {self.name!r}."
                )

    def plot_label(self, params: dict) -> str:
        x_h = params.get("x_h")
        y_h = params.get("y_h")
        A = params.get("A")
        regime = params.get("regime", "?")
        bits = []
        if x_h is not None and y_h is not None:
            bits.append(f"patch=({x_h:.2f},{y_h:.2f})")
        if A is not None:
            bits.append(f"A={A:.0f}")
        bits.append(regime)
        return " ".join(bits)

    # ---- OOD hooks (W3 out-of-distribution path) ----

    def ood_axes(self) -> dict[str, OODAxis]:
        return {
            "rc": OODAxis(
                name="rc", kind="simulation_parameter", field="R_c",
                id_reference=(0.5,), ood_values=(0.0, 0.01, 0.025, 1.25, 1.5, 2.0),
                trained_range=RC_RANGE, resolution_kind="none",
                notes="scalar interface resistance; same sweep as the forcing "
                "benchmark. 0.0 is the limiting (perfect-contact) case; values "
                "above RC_RANGE[1]=1.0 are out-of-distribution.",
            ),
            "patch_size": OODAxis(
                name="patch_size",
                kind="simulation_parameter",
                id_reference=(PATCH_W,),
                ood_values=(0.2,),
                field="w_h",
                trained_range=(PATCH_W, PATCH_W),
                resolution_kind="spatial_patch_edge",
                notes="square patch domain-fraction; overrides the fixed "
                "PATCH_W=PATCH_H=0.1 (w_h=value*Lx, h_h=value*Ly)",
            ),
            "t_off": OODAxis(
                name="t_off",
                kind="simulation_parameter",
                id_reference=(0.225,),
                ood_values=(0.075, 0.15, 0.30),
                field="t_off",
                trained_range=(0.225, 0.225),
                notes="heating cutoff in physical seconds; trained value is "
                "T_OFF_FRAC * t_final = 0.75 * 0.30 = 0.225",
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
        x_grid = grids["x_grid"]
        y_grid = grids["y_grid"]
        X = grids["X"]
        a, b = float(x_grid[0]), float(x_grid[-1])
        c, d = float(y_grid[0]), float(y_grid[-1])
        Lx, Ly = b - a, d - c
        interface_x = _scale_fraction(INTERFACE_X, a, b)
        t_final = float(time_cfg["t_final"])

        w_h_default = PATCH_W * Lx
        h_h_default = PATCH_H * Ly
        t_off_default = T_OFF_FRAC * t_final

        # Bound the shared centroid draw by the LARGEST patch in the sweep so the
        # patch stays in-domain for every value and the centroid is byte-identical
        # across the sweep (only the swept patch extent / regime changes).
        if axis.name == "patch_size":
            max_frac = max(float(v) for v in axis.sweep_values())
            w_h_bound = max_frac * Lx
            h_h_bound = max_frac * Ly
        else:
            w_h_bound = w_h_default
            h_h_bound = h_h_default

        (cx_lo, cx_hi), (cy_lo, cy_hi) = _patch_center_ranges(
            a, b, c, d, w_h_bound, h_h_bound
        )
        u_xh = float(rng.uniform(0.0, 1.0))
        u_yh = float(rng.uniform(0.0, 1.0))
        x_h = float(cx_lo + u_xh * (cx_hi - cx_lo))
        y_h = float(cy_lo + u_yh * (cy_hi - cy_lo))

        R_c = float(rng.uniform(*RC_RANGE))
        A_min, A_max = _patch_a_range()
        A = float(np.exp(rng.uniform(float(np.log(A_min)), float(np.log(A_max)))))

        return {
            "domain": (a, b, c, d),
            "interface_x": interface_x,
            "R_c": R_c,
            "A": A,
            "x_h": x_h,
            "y_h": y_h,
            "w_h_default": w_h_default,
            "h_h_default": h_h_default,
            "t_off_default": t_off_default,
            "T0": np.full(X.shape, 300.0, dtype=np.float32),
            "ic_family": "uniform_2d",
            "ic_params": {"T0_offset": 0.0},
        }

    def apply_ood_value(
        self, latents: dict[str, Any], axis: OODAxis, value: Any,
    ) -> dict[str, Any]:
        a, b, c, d = latents["domain"]
        Lx, Ly = b - a, d - c
        interface_x = float(latents["interface_x"])
        x_h = float(latents["x_h"])
        y_h = float(latents["y_h"])

        R_c = float(latents["R_c"])
        if axis.name == "patch_size":
            w_h = float(value) * Lx
            h_h = float(value) * Ly
            t_off = float(latents["t_off_default"])
        elif axis.name == "t_off":
            w_h = float(latents["w_h_default"])
            h_h = float(latents["h_h_default"])
            t_off = float(value)
        elif axis.name == "rc":
            w_h = float(latents["w_h_default"])
            h_h = float(latents["h_h_default"])
            t_off = float(latents["t_off_default"])
            R_c = float(value)
        else:
            raise ValueError(
                f"Unknown OOD axis {axis.name!r} for benchmark {self.name!r}."
            )

        _validate_patch_bounds(
            x_h=x_h, y_h=y_h, w_h=w_h, h_h=h_h, a=a, b=b, c=c, d=d,
        )
        regime = _classify_regime(x_h, interface_x, w_h)

        params = {
            "material_properties": dict(MATERIAL_PROPERTIES),
            "R_c": R_c,
            "interface_x": interface_x,
            "x_h": x_h,
            "y_h": y_h,
            "w_h": w_h,
            "h_h": h_h,
            "A": float(latents["A"]),
            "t_off": t_off,
            "regime": regime,
            "T0": np.array(latents["T0"], dtype=np.float32),
            "ic_family": latents["ic_family"],
            "ic_params": copy.deepcopy(latents["ic_params"]),
        }
        params["_ood_realized"] = {
            "requested": float(value),
            "w_h": w_h,
            "h_h": h_h,
            "x_h": x_h,
            "y_h": y_h,
            "x_extent": (x_h - 0.5 * w_h, x_h + 0.5 * w_h),
            "y_extent": (y_h - 0.5 * h_h, y_h + 0.5 * h_h),
        }
        return params

    def resolution_scale(self, axis: OODAxis, params: dict) -> float | None:
        if axis.resolution_kind == "spatial_patch_edge":
            return float(params["w_h"])
        return None
