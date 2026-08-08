from __future__ import annotations

import copy
from typing import Any

import numpy as np

from problems.base import OODAxis, ProblemDims
from problems.forcing import FORCING_TEMPORAL_TOKEN_DIM
from problems.source import (
    SourceProblem,
    _classify_regime,
    _lhs_unit,
    _validate_patch_bounds,
)
from problems.source_itr import (
    RC_Y_CHANNEL,  # noqa: F401  (documented channel index; kept for parity)
    S_Y_CHANNEL,
    SPATIAL_CHANNELS_BINS,
    SPATIAL_CHANNELS_TEMPORAL,
    SourceItrProblem,
)
from src.physics.internal_source import (
    RC_MIN,
    RC_SIN_RANGES,
    R_PEAK_MAX,
    make_rc_sin_profile,
)

# ----- representation constants (source_itr_sin tensor contract) ---------------

# cond_static: lead time + 2 sinusoid scalars [R_base, A] + 4 patch scalars
# [x_h, y_h, w_h, h_h]. The four Gaussian-void scalars of `source_itr` collapse
# to the two sinusoid parameters; A (patch amplitude) still never conditions.
COND_STATIC_DIM = 7

# Spatial-descriptor conditioning ablation slice. cond layout is
# [t_bar_norm, R_base_norm, A_norm, x_h_norm, y_h_norm, w_h_norm, h_h_norm]; the
# patch-geometry descriptor is [3:7], while the two sinusoid scalars [1:3] are
# retained. No one-hot family label, so family_cond_slice stays None.
SOURCE_ITR_SIN_SPATIAL_DESCRIPTOR_SLICE = slice(3, COND_STATIC_DIM)  # slice(3, 7)


def build_cond_vector_sin(
    t_bar_norm: float,
    R_base: float,
    A: float,
    x_h: float,
    y_h: float,
    w_h: float,
    h_h: float,
    *,
    x_center_range: tuple[float, float],
    y_center_range: tuple[float, float],
    x_length_scale: float,
    y_length_scale: float,
) -> np.ndarray:
    """Assemble the 7-dim static conditioning vector for source_itr_sin.

    Layout: [t_bar_norm, R_base_norm, A_norm, x_h_norm, y_h_norm, w_h_norm,
    h_h_norm]. R_base is min-max normalized over RC_SIN_RANGES["R_base"]. A uses
    the **global** conditioning normalization A_norm = A / (R_PEAK_MAX - RC_MIN)
    over the constant RC_SIN_RANGES["A"] (NOT the headroom fraction
    A / (R_PEAK_MAX - R_base) used for sampling/OOD/inverse): a constant
    denominator keeps this conditioning channel stationary across sims. The patch
    scalars match `source.build_cond_vector`.
    """
    base_lo, base_hi = RC_SIN_RANGES["R_base"]
    A_lo, A_hi = RC_SIN_RANGES["A"]

    R_base_norm = (R_base - base_lo) / (base_hi - base_lo)
    A_norm = (A - A_lo) / (A_hi - A_lo)

    x_h_norm = (x_h - x_center_range[0]) / (x_center_range[1] - x_center_range[0])
    y_h_norm = (y_h - y_center_range[0]) / (y_center_range[1] - y_center_range[0])
    w_h_norm = w_h / x_length_scale
    h_h_norm = h_h / y_length_scale

    return np.array(
        [
            t_bar_norm,
            R_base_norm, A_norm,
            x_h_norm, y_h_norm, w_h_norm, h_h_norm,
        ],
        dtype=np.float32,
    )


def _sin_severity(A: float) -> float:
    """Analytic integrated excess resistance int_0^1 (R_c - R_base) dy = A*(2/pi)."""
    return float(A) * 2.0 / np.pi


class SourceItrSinProblem(SourceItrProblem):
    """Source benchmark with a *sinusoidal* interface resistance R_c(y).

    Subclasses :class:`SourceItrProblem`, reusing its patch/amplitude/IC sampling
    and inheriting `build_item`, `configure_solver`, and `_rc_channel` unchanged.
    The interface at x = 0.5 carries a single-hump profile
        R_c(y) = R_base + A * sin(pi * y)
    (2 parameters) instead of the 4-parameter Gaussian void. On [0, 1],
    sin(pi y) >= 0, so with A in [0, R_PEAK_MAX - R_base] the profile stays in
    [R_base, R_base + A] subset [RC_MIN, R_PEAK_MAX] -- strictly positive, so the
    inherited rc_log_norm channel works unchanged. Only the two parent hooks
    `_canonical_profile` and `_cond_vector` are overridden; the Gaussian-void
    `source_itr` benchmark is untouched.
    """

    name = "source_itr_sin"

    spatial_descriptor_cond_slice = SOURCE_ITR_SIN_SPATIAL_DESCRIPTOR_SLICE

    def __init__(self, representation: str = "temporal_encoder"):
        self.representation = representation
        self.rc_channel_mode = "broadcast"
        # rc_ell only matters in localized mode; keep the parent default.
        from problems.source_itr import DEFAULT_RC_ELL

        self.rc_ell = DEFAULT_RC_ELL

        if representation == "temporal_encoder":
            self.dims = ProblemDims(
                in_channels=SPATIAL_CHANNELS_TEMPORAL,
                cond_static_dim=COND_STATIC_DIM,
                has_forcing_seq=True,
                temporal_token_dim=FORCING_TEMPORAL_TOKEN_DIM,
                t_stats_dim=3,
                use_temporal_encoder=True,
                s_y_channel=S_Y_CHANNEL,
                use_forcing_time_aug=False,
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
        # Reuse the parent's (Gaussian-void) patch/A/IC sampling so the patch
        # distribution stays identical, then overwrite the interface params with
        # the 2-param sinusoid. The parent writes the void keys R_c_amp/R_c_y0/
        # R_c_sigma, which this benchmark's validate_schema forbids, so pop them.
        sim_params = super().sample_sim_params(rng, rng_profile, grids, time_cfg)
        num_sims = len(sim_params)
        lhs_seed = int(time_cfg.get("lhs_seed", 0))

        # Fresh 2-col LHS stream for [R_base, u_A]; offset from the parent void
        # stream (+991) so the two do not alias.
        sin_rng = np.random.default_rng(lhs_seed + 993)
        u = _lhs_unit(num_sims, 2, sin_rng)

        base_lo, base_hi = RC_SIN_RANGES["R_base"]
        R_base = base_lo + u[:, 0] * (base_hi - base_lo)
        u_A = u[:, 1]
        # Headroom fraction (NOT the global conditioning norm): keeps R_c(y) in
        # [RC_MIN, R_PEAK_MAX] by construction; (R_base, A) jointly triangular.
        A = u_A * (R_PEAK_MAX - R_base)

        for i, params in enumerate(sim_params):
            params.pop("R_c_amp", None)
            params.pop("R_c_y0", None)
            params.pop("R_c_sigma", None)
            R_c_base = float(R_base[i])
            params["R_c_base"] = R_c_base
            params["R_c_A"] = float(A[i])
            # Universal R_c column mirrors R_base (as in source_itr).
            params["R_c"] = R_c_base

        return sim_params

    # ---- interface-resistance profile / conditioning hooks ----

    def _canonical_profile(self, params: dict, y_grid: np.ndarray) -> np.ndarray:
        return make_rc_sin_profile(
            y_grid,
            R_base=float(params["R_c_base"]),
            A=float(params["R_c_A"]),
        )

    def _cond_vector(
        self,
        params: dict,
        *,
        t_bar_norm: float,
        x_center_range: tuple[float, float],
        y_center_range: tuple[float, float],
        x_length_scale: float,
        y_length_scale: float,
    ) -> np.ndarray:
        return build_cond_vector_sin(
            t_bar_norm=float(t_bar_norm),
            R_base=float(params["R_c_base"]), A=float(params["R_c_A"]),
            x_h=float(params["x_h"]), y_h=float(params["y_h"]),
            w_h=float(params["w_h"]), h_h=float(params["h_h"]),
            x_center_range=x_center_range,
            y_center_range=y_center_range,
            x_length_scale=x_length_scale,
            y_length_scale=y_length_scale,
        )

    # ---- val-pair logging ----

    val_pair_fields = ("x_h", "y_h", "A", "regime", "R_c_A")

    def val_pair_row(self, ds, sim_id: int, s: int, j: int) -> dict[str, Any]:
        p = ds.sim_params[int(sim_id)]
        return {
            "x_h": float(p["x_h"]),
            "y_h": float(p["y_h"]),
            "A": float(p["A"]),
            "regime": p.get("regime", ""),
            "R_c_A": float(p["R_c_A"]),
        }

    # ---- schema ----

    def validate_schema(self, sim_params: np.ndarray, sim_ids: np.ndarray) -> None:
        required = (
            "R_c", "interface_x", "x_h", "y_h", "A", "t_off", "w_h", "h_h",
            "R_c_base", "R_c_A",
        )
        forbidden = (
            "R_c_amp", "R_c_y0", "R_c_sigma",
            "temporal_family", "temporal_params",
            "spatial_family", "spatial_params",
        )
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
                    f"sim_params[{int(sid)}] still contains forbidden keys "
                    f"{present_legacy}; regenerate with benchmark {self.name!r}."
                )
            R_base = float(entry["R_c_base"])
            A = float(entry["R_c_A"])
            if not (0.0 <= A <= R_PEAK_MAX - R_base + 1e-9):
                raise ValueError(
                    f"sim_params[{int(sid)}] R_c_A={A} outside the dependent bound "
                    f"[0, R_PEAK_MAX - R_base] = [0, {R_PEAK_MAX - R_base}]."
                )

    # ---- OOD hooks ----

    def ood_axes(self) -> dict[str, OODAxis]:
        base_lo, base_hi = RC_SIN_RANGES["R_base"]
        A_lo, A_hi = RC_SIN_RANGES["A"]
        return {
            "rc_base": OODAxis(
                name="rc_base",
                kind="simulation_parameter",
                id_reference=(0.5,),
                ood_values=(0.1, 1.25, 1.5, 2.0),
                field="R_c_base",
                trained_range=(base_lo, base_hi),
                notes="sinusoid floor R_base; reject < RC_MIN (log-normalized "
                "R_c(y) channel hits log(0)); A held at the shared background "
                "value.",
            ),
            "rc_A": OODAxis(
                name="rc_A",
                kind="simulation_parameter",
                id_reference=(0.5 * (A_lo + A_hi),),
                ood_values=(0.0, 3.5, 4.0),
                field="R_c_A",
                trained_range=(A_lo, A_hi),
                notes="sinusoid depth A; R_base held at background. A=0 is the "
                "flat-profile limiting case.",
            ),
            "rc_severity": OODAxis(
                name="rc_severity",
                kind="compound",
                id_reference=(0.5,),
                ood_values=(0.1, 1.0, 1.5),
                field="R_c_A",
                trained_range=(0.0, 1.0),
                compound=True,
                notes="severity-fraction of the no-clip headroom "
                "(A = value * (R_PEAK_MAX - R_base)); value > 1.0 drives the "
                "profile peak past R_PEAK_MAX (extreme). CRN paired on latents.",
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
        # Reuse the grandparent Source patch / amplitude / IC background. Skip the
        # SourceItrProblem override (which draws the 4 void background scalars we
        # do not use) by calling SourceProblem.draw_latents directly.
        latents = SourceProblem.draw_latents(
            self, rng, rng_profile, grids, time_cfg, axis
        )

        base_lo, base_hi = RC_SIN_RANGES["R_base"]
        u_base = float(rng.uniform(0.0, 1.0))
        u_A = float(rng.uniform(0.0, 1.0))
        R_base_bg = base_lo + u_base * (base_hi - base_lo)
        A_bg = u_A * (R_PEAK_MAX - R_base_bg)

        latents["y_grid"] = np.asarray(grids["y_grid"], dtype=np.float64)
        latents["R_base_bg"] = float(R_base_bg)
        latents["A_bg"] = float(A_bg)
        return latents

    def apply_ood_value(
        self, latents: dict[str, Any], axis: OODAxis, value: Any,
    ) -> dict[str, Any]:
        y_grid = latents["y_grid"]
        R_base_bg = float(latents["R_base_bg"])
        A_bg = float(latents["A_bg"])

        R_base, A = R_base_bg, A_bg
        if axis.name == "rc_base":
            if float(value) < RC_MIN:
                raise ValueError(
                    f"rc_base value {value} < RC_MIN ({RC_MIN}); the log-normalized "
                    f"R_c(y) channel is undefined at zero floor."
                )
            R_base = float(value)
        elif axis.name == "rc_A":
            A = float(value)
        elif axis.name == "rc_severity":
            A = float(value) * (R_PEAK_MAX - R_base_bg)
        else:
            raise ValueError(
                f"Unknown OOD axis {axis.name!r} for benchmark {self.name!r}."
            )

        a, b, c, d = latents["domain"]
        interface_x = float(latents["interface_x"])
        x_h = float(latents["x_h"])
        y_h = float(latents["y_h"])
        w_h = float(latents["w_h_default"])
        h_h = float(latents["h_h_default"])
        t_off = float(latents["t_off_default"])
        _validate_patch_bounds(
            x_h=x_h, y_h=y_h, w_h=w_h, h_h=h_h, a=a, b=b, c=c, d=d,
        )
        regime = _classify_regime(x_h, interface_x, w_h)

        params = {
            "R_c": float(R_base),
            "interface_x": interface_x,
            "x_h": x_h,
            "y_h": y_h,
            "w_h": w_h,
            "h_h": h_h,
            "A": float(latents["A"]),
            "t_off": t_off,
            "regime": regime,
            "R_c_base": float(R_base),
            "R_c_A": float(A),
            "T0": np.array(latents["T0"], dtype=np.float32),
            "ic_family": latents["ic_family"],
            "ic_params": copy.deepcopy(latents["ic_params"]),
        }
        params["_ood_realized"] = {
            "requested": float(value),
            "R_c_base": float(R_base),
            "R_c_A": float(A),
            "realized_severity": _sin_severity(A),
        }
        return params

    def resolution_scale(self, axis: OODAxis, params: dict) -> float | None:
        return None
