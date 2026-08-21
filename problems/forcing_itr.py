from __future__ import annotations

import re
from typing import Any

import numpy as np

from problems.base import OODAxis, ProblemDims
from problems.forcing import (
    COND_STATIC_DIM as FORCING_COND_STATIC_DIM,
    FORCING_TEMPORAL_TOKEN_DIM,
    SPATIAL_CHANNELS_TEMPORAL as FORCING_SPATIAL_CHANNELS_TEMPORAL,
    ForcingProblem,
)
from problems.source_itr import RC_Y_CHANNEL, rc_log_norm
from src.physics.boundary_forcing import (
    build_qL,
    build_qL_integral,
    default_ramp_seconds,
)
from src.physics.fv_solver_2d import FVSolver2D
from src.physics.internal_source import (
    RC_VOID_RANGES,
    R_PEAK_MAX,
    integrated_excess_resistance,
    make_rc_void_profile,
)


# build_cond_vector_forcing_itr rewrites forcing's [t_bar, R_c] into
# [t_bar, itr(4)]: the scalar R_c slot at index 1 becomes the four void params.
COND_STATIC_DIM = FORCING_COND_STATIC_DIM + 3
SPATIAL_CHANNELS_TEMPORAL = FORCING_SPATIAL_CHANNELS_TEMPORAL + 1
S_Y_CHANNEL = 3
REQUIRED_OBSERVATION_TIMES = (0.07, 0.15, 0.30)
FORCING_SAMPLER_VERSION = "forcing-v1"
DEFAULT_FORCING_PROFILE_SEED = 1
DEFAULT_ITR_PROFILE_SEED = 992
_FORCING_KEY_RE = re.compile(r"^forcing-v1:seed=\d+:index=\d{6}$")


def _lhs_unit(n: int, d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    edges = np.linspace(0.0, 1.0, n + 1, dtype=np.float64)
    points = edges[:-1, None] + np.diff(edges)[:, None] * rng.random((n, d))
    for j in range(d):
        rng.shuffle(points[:, j])
    return points


def forcing_sample_key(sample_index: int, *, seed: int) -> str:
    return (
        f"{FORCING_SAMPLER_VERSION}:seed={int(seed)}:"
        f"index={int(sample_index):06d}"
    )


def build_cond_vector_forcing_itr(
    parent_cond: np.ndarray,
    *,
    R_base: float,
    R_amp: float,
    y0: float,
    sigma: float,
) -> np.ndarray:
    """Expand forcing's scalar-resistance slot to the four ITR parameters."""
    parent = np.asarray(parent_cond, dtype=np.float32)
    if parent.shape != (FORCING_COND_STATIC_DIM,):
        raise ValueError(
            f"parent forcing condition must have shape ({FORCING_COND_STATIC_DIM},), "
            f"got {parent.shape}."
        )
    base_lo, base_hi = RC_VOID_RANGES["R_base"]
    amp_lo, amp_hi = RC_VOID_RANGES["R_amp"]
    y0_lo, y0_hi = RC_VOID_RANGES["y0"]
    sig_lo, sig_hi = RC_VOID_RANGES["sigma"]
    itr = np.array(
        [
            (R_base - base_lo) / (base_hi - base_lo),
            (R_amp - amp_lo) / (amp_hi - amp_lo),
            (y0 - y0_lo) / (y0_hi - y0_lo),
            (sigma - sig_lo) / (sig_hi - sig_lo),
        ],
        dtype=np.float32,
    )
    return np.concatenate([parent[:1], itr, parent[2:]]).astype(np.float32)


class ForcingItrProblem(ForcingProblem):
    """Forcing benchmark with a Gaussian spatial interface-resistance profile."""

    name = "forcing_itr"
    required_observation_times = REQUIRED_OBSERVATION_TIMES

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

    def sample_sim_params(
        self,
        rng: np.random.Generator,
        rng_profile: np.random.Generator,
        grids: dict[str, np.ndarray],
        time_cfg: dict[str, Any],
    ) -> list[dict]:
        params = super().sample_sim_params(rng, rng_profile, grids, time_cfg)
        n = len(params)
        itr_seed = int(time_cfg.get("itr_profile_seed", DEFAULT_ITR_PROFILE_SEED))
        forcing_seed = int(
            time_cfg.get("forcing_profile_seed", DEFAULT_FORCING_PROFILE_SEED)
        )
        u = _lhs_unit(n, 3, itr_seed)
        y0_lo, y0_hi = RC_VOID_RANGES["y0"]
        sig_lo, sig_hi = RC_VOID_RANGES["sigma"]
        for i, entry in enumerate(params):
            R_base = float(entry["R_c"])
            entry.update(
                {
                    "R_c_base": R_base,
                    "R_c_amp": float(u[i, 0] * (R_PEAK_MAX - R_base)),
                    "R_c_y0": float(y0_lo + u[i, 1] * (y0_hi - y0_lo)),
                    "R_c_sigma": float(sig_lo + u[i, 2] * (sig_hi - sig_lo)),
                    "parent_benchmark": "forcing",
                    "parent_sim_id": int(i),
                    "forcing_sample_key": forcing_sample_key(i, seed=forcing_seed),
                }
            )
        return params

    @staticmethod
    def _canonical_profile(params: dict, y_grid: np.ndarray) -> np.ndarray:
        return make_rc_void_profile(
            y_grid,
            R_base=float(params["R_c_base"]),
            R_amp=float(params["R_c_amp"]),
            y0=float(params["R_c_y0"]),
            sigma=float(params["R_c_sigma"]),
        )

    def _cond_vector(self, params: dict, parent_cond: np.ndarray) -> np.ndarray:
        """Insert the interface-family conditioning into the parent cond vector.

        Required override point: subclasses whose interface family conditions on
        a different parameter set (e.g. the 2-param sinusoid) override only this
        hook so `build_item` stays inherited and cannot drift from the spatial
        R_c(y) channel built via `_canonical_profile`.
        """
        return build_cond_vector_forcing_itr(
            parent_cond,
            R_base=float(params["R_c_base"]),
            R_amp=float(params["R_c_amp"]),
            y0=float(params["R_c_y0"]),
            sigma=float(params["R_c_sigma"]),
        )

    def configure_solver(self, params: dict, base_kwargs: dict) -> FVSolver2D:
        physics = self.physics_parameters(
            base_kwargs["a"], base_kwargs["b"],
            base_kwargs["c"], base_kwargs["d"],
        )
        y_grid = np.asarray(base_kwargs["y_grid"], dtype=np.float64)
        ramp = base_kwargs.get("ramp_seconds")
        t_ramp = (
            float(ramp)
            if ramp is not None
            else default_ramp_seconds(base_kwargs["dt"])
        )
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
        profile = self._canonical_profile(params, y_grid)
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
            interface_R=[profile],
            q_left_fn=q_left_fn,
            q_left_integral_fn=q_left_integral_fn,
        )

    def setup_dataset(self, ds) -> None:
        if self.rc_channel_mode != "broadcast":
            raise ValueError("forcing_itr currently supports rc_channel_mode='broadcast'.")
        self.validate_schema(ds.sim_params, ds.sim_ids)
        super().setup_dataset(ds)

    def build_item(self, ds, sid: int, s: int, j: int) -> dict[str, np.ndarray]:
        params = ds.sim_params[int(sid)]
        parent = self._build_item_with_resistance(
            ds, sid, s, j, R_c=float(params["R_c_base"])
        )
        profile = self._canonical_profile(params, ds.y_grid)
        rc_channel = np.broadcast_to(
            rc_log_norm(profile)[None, :], (ds.Nx, ds.Ny)
        )[..., None]
        parent["spatial"] = np.concatenate(
            [
                parent["spatial"][..., :RC_Y_CHANNEL],
                rc_channel,
                parent["spatial"][..., RC_Y_CHANNEL:],
            ],
            axis=-1,
        ).astype(np.float32)
        parent["cond_static"] = self._cond_vector(params, parent["cond_static"])
        parent["cond_static"] = self._apply_spatial_conditioning_mask(
            parent["cond_static"]
        )
        return parent

    val_pair_fields = (
        "temporal_family", "spatial_family",
        "R_c_amp", "R_c_y0", "R_c_sigma", "S_R",
    )

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
            "R_c_amp": float(p["R_c_amp"]),
            "R_c_y0": float(p["R_c_y0"]),
            "R_c_sigma": float(p["R_c_sigma"]),
            "S_R": severity,
        }

    def validate_schema(self, sim_params: np.ndarray, sim_ids: np.ndarray) -> None:
        required = (
            "R_c_base", "R_c_amp", "R_c_y0", "R_c_sigma",
            "temporal_family", "temporal_params", "spatial_family", "spatial_params",
            "parent_benchmark", "parent_sim_id", "forcing_sample_key",
        )
        base_lo, base_hi = RC_VOID_RANGES["R_base"]
        y0_lo, y0_hi = RC_VOID_RANGES["y0"]
        sig_lo, sig_hi = RC_VOID_RANGES["sigma"]
        for sid in sim_ids:
            entry = sim_params[int(sid)]
            missing = [key for key in required if key not in entry]
            if missing:
                raise ValueError(
                    f"sim_params[{int(sid)}] missing keys {missing} for benchmark "
                    f"{self.name!r}."
                )
            R_base = float(entry["R_c_base"])
            R_amp = float(entry["R_c_amp"])
            if not base_lo <= R_base <= base_hi:
                raise ValueError(f"R_c_base={R_base} is outside [{base_lo}, {base_hi}].")
            if not 0.0 <= R_amp <= R_PEAK_MAX - R_base:
                raise ValueError("R_c_amp violates its R_c_base-dependent ceiling.")
            if not y0_lo <= float(entry["R_c_y0"]) <= y0_hi:
                raise ValueError("R_c_y0 is outside its sampling range.")
            if not sig_lo <= float(entry["R_c_sigma"]) <= sig_hi:
                raise ValueError("R_c_sigma is outside its sampling range.")
            if entry["parent_benchmark"] != "forcing":
                raise ValueError("parent_benchmark must be 'forcing'.")
            if int(entry["parent_sim_id"]) != int(sid):
                raise ValueError("parent_sim_id must match the paired simulation ID.")
            if not _FORCING_KEY_RE.match(str(entry["forcing_sample_key"])):
                raise ValueError("forcing_sample_key does not match the versioned schema.")

    def ood_axes(self) -> dict[str, OODAxis]:
        return {}
