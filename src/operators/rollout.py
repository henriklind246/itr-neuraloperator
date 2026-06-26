from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from problems import diffusion as diffusion_problem
from problems import forcing as forcing_problem
from problems import interfaces as interfaces_problem
from problems import source as source_problem
from problems.base import ProblemSpec, empty_forcing_seq
from src.physics.boundary_forcing import (
    FORCING_BINS,
    integrate_temporal_bins_ramped_signed,
    ramped_temporal,
)
from src.physics.internal_source import make_sin2_pulse


@dataclass(frozen=True)
class RolloutOptions:
    enabled: bool = False
    num_substeps: int = 1
    partition: str = "homogeneous"


def rollout_options_from_config(
    config: dict[str, Any],
    *,
    enabled: bool | None = None,
    num_substeps: int | None = None,
    partition: str | None = None,
) -> RolloutOptions:
    cfg = (config.get("evaluation") or {}).get("rollout") or {}
    resolved_num_substeps = (
        int(cfg.get("num_substeps", 1)) if num_substeps is None else int(num_substeps)
    )
    if enabled is None and num_substeps is not None and resolved_num_substeps > 1:
        resolved_enabled = True
    else:
        resolved_enabled = bool(cfg.get("enabled", False)) if enabled is None else bool(enabled)
    resolved_partition = str(cfg.get("partition", "homogeneous") if partition is None else partition)
    return RolloutOptions(
        enabled=resolved_enabled,
        num_substeps=resolved_num_substeps,
        partition=resolved_partition,
    )


def rollout_is_active(options: RolloutOptions | None) -> bool:
    if options is None:
        return False
    validate_rollout_options(options)
    return bool(options.enabled and options.num_substeps > 1)


def validate_rollout_options(options: RolloutOptions) -> None:
    if options.num_substeps < 1:
        raise ValueError(f"num_substeps must be >= 1, got {options.num_substeps}")
    if options.partition != "homogeneous":
        raise ValueError(
            f"Unsupported rollout partition {options.partition!r}; only 'homogeneous' is implemented."
        )


def build_homogeneous_rollout_times(
    t_s: float,
    t_j: float,
    num_substeps: int,
) -> list[tuple[float, float]]:
    """Partition [t_s, t_j] into equal-duration autoregressive subintervals."""
    if num_substeps < 1:
        raise ValueError(f"num_substeps must be >= 1, got {num_substeps}")
    t_s = float(t_s)
    t_j = float(t_j)
    if t_j <= t_s:
        raise ValueError(f"rollout requires t_j > t_s, got t_s={t_s}, t_j={t_j}")
    edges = np.linspace(t_s, t_j, int(num_substeps) + 1, dtype=np.float64)
    edges[-1] = t_j
    return [(float(edges[k]), float(edges[k + 1])) for k in range(int(num_substeps))]


def _copy_item(item: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {k: np.array(v, copy=True) for k, v in item.items()}


def _current_field(current_T_norm: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(current_T_norm, torch.Tensor):
        current_T_norm = current_T_norm.detach().cpu().numpy()
    arr = np.asarray(current_T_norm, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"current_T_norm must have shape (Nx, Ny), got {arr.shape}")
    return arr


def _q_callable_for_boundary(ds, sid: int, params: dict):
    if not hasattr(ds, "_q_callables"):
        ds._q_callables = {}
    if sid not in ds._q_callables:
        ds._q_callables[sid] = ramped_temporal(
            params["temporal_family"],
            params["temporal_params"],
            ds.ramp_seconds,
        )
    return ds._q_callables[sid]


def _build_forcing_rollout_item(
    out: dict[str, np.ndarray],
    ds,
    problem,
    sid: int,
    current: np.ndarray,
    t_lo: float,
    t_hi: float,
) -> dict[str, np.ndarray]:
    params = ds.sim_params[sid]
    spatial = out["spatial"]
    spatial[..., 0] = current
    t_bar_norm = (float(t_hi) - float(t_lo)) / float(ds.t_final)
    t_s_norm = float(t_lo) / float(ds.t_final)
    out["cond_static"] = forcing_problem.build_cond_vector(
        t_bar_norm=t_bar_norm,
        t_s_norm=t_s_norm,
        R_c=float(params["R_c"]),
        spatial_family=params["spatial_family"],
        spatial_params=params["spatial_params"],
        y_bounds=(float(ds.y_grid[0]), float(ds.y_grid[-1])),
    )

    if problem.representation == "temporal_encoder":
        q = _q_callable_for_boundary(ds, sid, params)
        t_samples, a_m = forcing_problem._sample_a(q, t_lo, t_hi, ds.temporal_samples)
        out["forcing_seq"] = forcing_problem._forcing_seq_2tok_from_samples(
            t_samples,
            a_m,
            A_amp_ref=forcing_problem.A_AMP_REF,
        )
    else:
        s_y = ds.s_y_profiles[sid]
        bins = integrate_temporal_bins_ramped_signed(
            params["temporal_family"],
            params["temporal_params"],
            float(t_lo),
            float(t_hi),
            ds.ramp_seconds,
            K=FORCING_BINS,
        ).astype(np.float32)
        Q_y_bins = (s_y[None, :, None] * bins[None, None, :] / ds.q_ref).astype(np.float32)
        spatial[..., forcing_problem.SPATIAL_CHANNELS_TEMPORAL:] = np.broadcast_to(
            Q_y_bins,
            (ds.Nx, ds.Ny, FORCING_BINS),
        )
        out["forcing_seq"] = empty_forcing_seq()
    return out


def _build_interfaces_rollout_item(
    out: dict[str, np.ndarray],
    ds,
    problem,
    sid: int,
    current: np.ndarray,
    t_lo: float,
    t_hi: float,
) -> dict[str, np.ndarray]:
    params = ds.sim_params[sid]
    spatial = out["spatial"]
    spatial[..., 0] = current
    t_bar_norm = (float(t_hi) - float(t_lo)) / float(ds.t_final)
    t_s_norm = float(t_lo) / float(ds.t_final)
    out["cond_static"] = interfaces_problem.build_cond_vector(
        t_bar_norm=t_bar_norm,
        t_s_norm=t_s_norm,
        R_c=float(params["R_c"]),
        interface_x=float(params["interface_x"]),
        interface_x_range=interfaces_problem._physical_interface_range(
            float(ds.x_grid[0]), float(ds.x_grid[-1])
        ),
    )

    if problem.representation == "temporal_encoder":
        q = _q_callable_for_boundary(ds, sid, params)
        t_samples, a_m = interfaces_problem._sample_a(q, t_lo, t_hi, ds.temporal_samples)
        out["forcing_seq"] = interfaces_problem._forcing_seq_2tok_from_samples(
            t_samples,
            a_m,
            A_amp_ref=interfaces_problem.A_AMP_REF,
        )
    else:
        s_y = ds.s_y_profiles[sid]
        bins = integrate_temporal_bins_ramped_signed(
            params["temporal_family"],
            params["temporal_params"],
            float(t_lo),
            float(t_hi),
            ds.ramp_seconds,
            K=FORCING_BINS,
        ).astype(np.float32)
        Q_y_bins = (s_y[None, :, None] * bins[None, None, :] / ds.q_ref).astype(np.float32)
        spatial[..., interfaces_problem.SPATIAL_CHANNELS_TEMPORAL:] = np.broadcast_to(
            Q_y_bins,
            (ds.Nx, ds.Ny, FORCING_BINS),
        )
        out["forcing_seq"] = empty_forcing_seq()
    return out


def _build_source_rollout_item(
    out: dict[str, np.ndarray],
    ds,
    problem,
    sid: int,
    current: np.ndarray,
    t_lo: float,
    t_hi: float,
) -> dict[str, np.ndarray]:
    params = ds.sim_params[sid]
    spatial = out["spatial"]
    spatial[..., 0] = current
    t_bar_norm = (float(t_hi) - float(t_lo)) / float(ds.t_final)
    t_s_norm = float(t_lo) / float(ds.t_final)
    out["cond_static"] = source_problem.build_cond_vector(
        t_bar_norm=t_bar_norm,
        t_s_norm=t_s_norm,
        R_c=float(params["R_c"]),
        x_h=float(params["x_h"]),
        y_h=float(params["y_h"]),
        w_h=float(params["w_h"]),
        h_h=float(params["h_h"]),
        x_center_range=source_problem._patch_center_ranges(
            float(ds.x_grid[0]), float(ds.x_grid[-1]),
            float(ds.y_grid[0]), float(ds.y_grid[-1]),
            float(params["w_h"]), float(params["h_h"]),
        )[0],
        y_center_range=source_problem._patch_center_ranges(
            float(ds.x_grid[0]), float(ds.x_grid[-1]),
            float(ds.y_grid[0]), float(ds.y_grid[-1]),
            float(params["w_h"]), float(params["h_h"]),
        )[1],
        x_length_scale=float(ds.x_grid[-1] - ds.x_grid[0]),
        y_length_scale=float(ds.y_grid[-1] - ds.y_grid[0]),
    )

    if problem.representation == "temporal_encoder":
        if not hasattr(ds, "_q_callables"):
            ds._q_callables = {}
        if sid not in ds._q_callables:
            ds._q_callables[sid] = make_sin2_pulse(float(params["A"]), float(params["t_off"]))
        q = ds._q_callables[sid]
        t_samples, a_m = source_problem._sample_a(q, t_lo, t_hi, ds.temporal_samples)
        out["forcing_seq"] = source_problem._forcing_seq_2tok_from_samples(
            t_samples,
            a_m,
            A_amp_ref=ds.a_amp_ref,
        )
    else:
        spatial[..., source_problem.SPATIAL_CHANNELS_TEMPORAL:] = problem._source_bin_channels(
            ds,
            sid,
            float(t_lo),
            float(t_hi),
        )
        out["forcing_seq"] = empty_forcing_seq()
    return out


def _build_diffusion_rollout_item(
    out: dict[str, np.ndarray],
    ds,
    problem,
    sid: int,
    current: np.ndarray,
    t_lo: float,
    t_hi: float,
) -> dict[str, np.ndarray]:
    spatial = out["spatial"]
    spatial[..., 0] = current
    # Channels 1-3 (x_norm, y_norm, s_y_const) are static; leave them intact.
    t_bar_norm = (float(t_hi) - float(t_lo)) / float(ds.t_final)
    t_s_norm = float(t_lo) / float(ds.t_final)
    out["cond_static"] = np.array([t_bar_norm, t_s_norm], dtype=np.float32)
    t_samples, a_m = diffusion_problem._sample_a(
        lambda t: 0.0, t_lo, t_hi, ds.temporal_samples
    )
    out["forcing_seq"] = diffusion_problem._forcing_seq_2tok_from_samples(
        t_samples, a_m, A_amp_ref=diffusion_problem.A_AMP_REF,
    )
    return out


def build_rollout_item_from_base(
    base_item: dict[str, np.ndarray],
    dataset,
    problem: ProblemSpec,
    sim_id: int,
    current_T_norm: np.ndarray | torch.Tensor,
    t_lo: float,
    t_hi: float,
) -> dict[str, np.ndarray]:
    """Build one rollout substep item without changing dataset item semantics.

    The normal dataset item is the source of truth for schema, static spatial
    fields, and T_stats. Rollout only swaps the current normalized temperature
    source and recomputes subinterval time/forcing representations.
    """
    validate_rollout_options(RolloutOptions(enabled=True, num_substeps=1))
    sid = int(sim_id)
    out = _copy_item(base_item)
    current = _current_field(current_T_norm)
    if out["spatial"].shape[:2] != current.shape:
        raise ValueError(
            f"current_T_norm shape {current.shape} does not match spatial shape {out['spatial'].shape[:2]}"
        )

    if problem.name == "forcing":
        return _build_forcing_rollout_item(out, dataset, problem, sid, current, t_lo, t_hi)
    if problem.name == "interfaces":
        return _build_interfaces_rollout_item(out, dataset, problem, sid, current, t_lo, t_hi)
    if problem.name == "source":
        return _build_source_rollout_item(out, dataset, problem, sid, current, t_lo, t_hi)
    if problem.name == "diffusion":
        return _build_diffusion_rollout_item(out, dataset, problem, sid, current, t_lo, t_hi)
    raise ValueError(f"Unsupported benchmark for rollout: {problem.name!r}")


def _forward_numpy_item(model, item: dict[str, np.ndarray], device: torch.device) -> torch.Tensor:
    spatial = torch.from_numpy(item["spatial"]).unsqueeze(0).to(device)
    cond = torch.from_numpy(item["cond_static"]).unsqueeze(0).to(device)
    forcing_seq = item.get("forcing_seq")
    if bool(getattr(model, "use_temporal_encoder", True)) and forcing_seq is not None:
        forcing = torch.from_numpy(forcing_seq).unsqueeze(0).to(device)
        return model(spatial, cond, forcing)
    return model(spatial, cond)


def predict_autoregressive(
    model,
    dataset,
    sim_id: int,
    s: int,
    j: int,
    num_substeps: int,
    device: torch.device,
) -> torch.Tensor:
    """Predict T(t_j) by repeated homogeneous time-conditioned substeps."""
    problem = dataset.problem
    t_s = float(dataset.t_grid[int(s)])
    t_j = float(dataset.t_grid[int(j)])
    intervals = build_homogeneous_rollout_times(t_s, t_j, int(num_substeps))
    base_item = problem.build_item(dataset, int(sim_id), int(s), int(j))
    current = np.asarray(base_item["spatial"][..., 0], dtype=np.float32)

    y_pred = None
    for t_lo, t_hi in intervals:
        item = build_rollout_item_from_base(
            base_item,
            dataset,
            problem,
            int(sim_id),
            current,
            t_lo,
            t_hi,
        )
        y_pred = _forward_numpy_item(model, item, device)
        current = y_pred.squeeze(0).squeeze(-1).detach().cpu().numpy().astype(np.float32)

    if y_pred is None:
        raise RuntimeError("rollout produced no prediction")
    return y_pred
