from __future__ import annotations

from typing import Any

import numpy as np

from problems.base import OODAxis, ProblemDims
from problems.forcing import (
    COND_STATIC_DIM as FORCING_COND_STATIC_DIM,
    FORCING_TEMPORAL_TOKEN_DIM,
    SPATIAL_PROFILE_BINS,
)
from problems.forcing_itr import (
    DEFAULT_ITR_PROFILE_SEED,
    SPATIAL_CHANNELS_TEMPORAL,
    S_Y_CHANNEL,
    ForcingItrProblem,
    _lhs_unit,
)
from src.physics.internal_source import (
    RC_SIN_RANGES,
    R_PEAK_MAX,
    integrated_excess_resistance,
    make_rc_sin_profile,
)


# cond_static: forcing's [t_bar, R_c, profile_bins(8)] has its scalar R_c slot
# replaced by [R_base, A], giving [t_bar, R_base, A, profile_bins(8)].
COND_STATIC_DIM = FORCING_COND_STATIC_DIM + 1
FORCING_ITR_SIN_SPATIAL_DESCRIPTOR_SLICE = slice(
    COND_STATIC_DIM - SPATIAL_PROFILE_BINS,
    COND_STATIC_DIM,
)

DEFAULT_SIN_PROFILE_SEED = DEFAULT_ITR_PROFILE_SEED + 993


def build_cond_vector_forcing_itr_sin(
    parent_cond: np.ndarray,
    *,
    R_base: float,
    A: float,
) -> np.ndarray:
    """Replace forcing's scalar-resistance slot with the two sinusoid parameters.

    Layout: [t_bar, R_base_norm, A_norm, profile_bins(8)] (11-dim). R_base is
    min-max normalized over RC_SIN_RANGES["R_base"]; A uses the **global**
    conditioning normalization A_norm = A / (R_PEAK_MAX - RC_MIN) over the constant
    RC_SIN_RANGES["A"] (NOT the headroom fraction A / (R_PEAK_MAX - R_base) used for
    sampling/OOD/inverse) so the conditioning channel stays stationary across sims.
    Mirrors :func:`build_cond_vector_forcing_itr` for the 2-param sinusoid.
    """
    parent = np.asarray(parent_cond, dtype=np.float32)
    if parent.shape != (FORCING_COND_STATIC_DIM,):
        raise ValueError(
            f"parent forcing condition must have shape ({FORCING_COND_STATIC_DIM},), "
            f"got {parent.shape}."
        )
    base_lo, base_hi = RC_SIN_RANGES["R_base"]
    A_lo, A_hi = RC_SIN_RANGES["A"]
    itr = np.array(
        [
            (R_base - base_lo) / (base_hi - base_lo),
            (A - A_lo) / (A_hi - A_lo),
        ],
        dtype=np.float32,
    )
    return np.concatenate([parent[:1], itr, parent[2:]]).astype(np.float32)


class ForcingItrSinProblem(ForcingItrProblem):
    """Forcing benchmark with a *sinusoidal* interface resistance R_c(y).

    Subclasses :class:`ForcingItrProblem`, reusing its forcing-family sampling and
    inheriting `build_item`, `configure_solver`, and the R_c(y) spatial channel
    unchanged. The interface at x = 0.5 carries a single-hump profile
        R_c(y) = R_base + A * sin(pi * y)
    (2 parameters) instead of the 4-parameter Gaussian void. Only the two parent
    hooks `_canonical_profile` and `_cond_vector` are overridden; the Gaussian-void
    `forcing_itr` benchmark is untouched.
    """

    name = "forcing_itr_sin"
    spatial_descriptor_cond_slice = FORCING_ITR_SIN_SPATIAL_DESCRIPTOR_SLICE

    def __init__(self, representation: str = "temporal_encoder"):
        self.representation = representation
        self.rc_channel_mode = "broadcast"
        if representation == "temporal_encoder":
            self.dims = ProblemDims(
                in_channels=SPATIAL_CHANNELS_TEMPORAL,
                cond_static_dim=COND_STATIC_DIM,
                temporal_token_dim=FORCING_TEMPORAL_TOKEN_DIM,
                t_stats_dim=2,
                s_y_channel=S_Y_CHANNEL,
                use_forcing_time_aug=True,
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
        # Reuse the parent's (Gaussian-void) forcing-family sampling so the
        # temporal/spatial forcing distribution and R_c_base stay identical, then
        # overwrite the interface params with the 2-param sinusoid. The parent
        # writes the void keys R_c_amp/R_c_y0/R_c_sigma, forbidden here, so pop them.
        params = super().sample_sim_params(rng, rng_profile, grids, time_cfg)
        n = len(params)
        sin_seed = int(time_cfg.get("sin_profile_seed", DEFAULT_SIN_PROFILE_SEED))
        u = _lhs_unit(n, 1, sin_seed)
        for i, entry in enumerate(params):
            entry.pop("R_c_amp", None)
            entry.pop("R_c_y0", None)
            entry.pop("R_c_sigma", None)
            R_base = float(entry["R_c_base"])
            # Headroom fraction (NOT the global conditioning norm): keeps R_c(y)
            # in [RC_MIN, R_PEAK_MAX] by construction; (R_base, A) jointly
            # triangular.
            entry["R_c_A"] = float(u[i, 0] * (R_PEAK_MAX - R_base))
        return params

    # ---- interface-resistance profile / conditioning hooks ----

    @staticmethod
    def _canonical_profile(params: dict, y_grid: np.ndarray) -> np.ndarray:
        return make_rc_sin_profile(
            y_grid,
            R_base=float(params["R_c_base"]),
            A=float(params["R_c_A"]),
        )

    def _cond_vector(self, params: dict, parent_cond: np.ndarray) -> np.ndarray:
        return build_cond_vector_forcing_itr_sin(
            parent_cond,
            R_base=float(params["R_c_base"]),
            A=float(params["R_c_A"]),
        )

    # ---- val-pair logging ----

    val_pair_fields = ("temporal_family", "spatial_family", "R_c_A", "S_R")

    def val_pair_row(self, ds, sim_id: int, s: int, j: int) -> dict[str, Any]:
        p = ds.sim_params[int(sim_id)]
        profile = self._canonical_profile(p, ds.y_grid)
        severity = integrated_excess_resistance(
            ds.y_grid, profile, float(p["R_c_base"]),
            bounds=(0.0, 1.0),
        )
        return {
            "temporal_family": p["temporal_family"],
            "spatial_family": p["spatial_family"],
            "R_c_A": float(p["R_c_A"]),
            "S_R": severity,
        }

    # ---- schema ----

    def validate_schema(self, sim_params: np.ndarray, sim_ids: np.ndarray) -> None:
        required = (
            "R_c_base", "R_c_A",
            "temporal_family", "temporal_params", "spatial_family", "spatial_params",
            "parent_benchmark", "parent_sim_id", "forcing_sample_key",
        )
        forbidden = ("R_c_amp", "R_c_y0", "R_c_sigma")
        base_lo, base_hi = RC_SIN_RANGES["R_base"]
        for sid in sim_ids:
            entry = sim_params[int(sid)]
            missing = [key for key in required if key not in entry]
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
            if not base_lo <= R_base <= base_hi:
                raise ValueError(f"R_c_base={R_base} is outside [{base_lo}, {base_hi}].")
            if not 0.0 <= A <= R_PEAK_MAX - R_base + 1e-9:
                raise ValueError("R_c_A violates its R_c_base-dependent ceiling.")
            if entry["parent_benchmark"] != "forcing":
                raise ValueError("parent_benchmark must be 'forcing'.")
            if int(entry["parent_sim_id"]) != int(sid):
                raise ValueError("parent_sim_id must match the paired simulation ID.")

    def ood_axes(self) -> dict[str, OODAxis]:
        return {}
