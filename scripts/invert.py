"""Inverse estimation for scalar and sinusoidal interface resistance.

Supports forcing, forcing_itr_sin and source_itr_sin via InverseAdapter.
Sinusoidal profiles use theta=(R_base, A), with A bounded by R_PEAK_MAX-R_base.
The trained model is frozen; only interface conditioning is optimized.
Sensor noise, surrogate-error calibration, profile likelihood, FV refinement
and sensitivity diagnostics retain their existing metric conventions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import torch
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import chi2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import (  # noqa: E402
    SnapshotPairDataset,
    T_EPS,
    load_sim_data,
    problem_from_config,
)
from scripts.inverse_adapters import (  # noqa: E402
    InverseAdapter,
    SourceItrSinAdapter,
    SIN_PARAM_NAMES,
)
from src.operators.fno2d import FNO2d  # noqa: E402
from src.physics.internal_source import R_PEAK_MAX  # noqa: E402

_SOURCE_ITR_SIN_ADAPTER = SourceItrSinAdapter()
PAPER_OBSERVATION_TIMES = (0.07, 0.15, 0.30)
PAPER_SENSOR_Y = tuple(np.linspace(0.1, 0.9, 8))
# Outward march: one fixed step is 5% of the parameter's physical range, so the
# walk reaches either bound within PROFILE_WALK_MAX_STEPS from anywhere inside.
PROFILE_STEP_FRACTION = 0.05
PROFILE_WALK_MAX_STEPS = 40
PROFILE_ROOT_RTOL = 5e-5
PROFILE_ROOT_MAX_STEPS = 24
# Bounded scalar nuisance fit operates on the sigmoid coordinate s in (0, 1),
# which maps linearly onto the free physical coordinate.
PROFILE_NUISANCE_XATOL = 1e-6
PROFILE_NUISANCE_EPS = 1e-9
# Coarse sweep over the closed box before Brent refines. Brent is local and
# skips the bounds, so without this a boundary or secondary optimum is missed.
PROFILE_NUISANCE_SCAN = 9
PROFILE_OPTIMUM_ATOL = 1e-5
# Re-profile rounds allowed when a walk finds a better optimum than its reference.
PROFILE_OPTIMUM_MAX_ROUNDS = 3

# Inversion always runs on a freshly generated dataset that is disjoint from the
# training corpus. Training draws use ``rng_seed=0``; a non-zero seed here makes
# the inversion draws provably disjoint and records that seed in the dataset
# ``meta`` (see ``data/generate_dataset.py``).
INVERSION_DATASET_SEED = 1
# Calibration estimates the FNO's sensor-level error at true theta. It is cheap
# (one forward per sim, no optimization/FV), so we hold out a fixed floor of
# calibration sims regardless of how many sims are inverted: enough sims to keep
# the pooled sigma_FNO and the per-sim p90 local gate stable.
CALIBRATION_FLOOR = 32


# ---------------------------------------------------------------------------
# Reparameterization: unconstrained u  <->  physical theta  (the heart of the
# operator wiring; checkpoint-independent and unit-tested separately).
# ---------------------------------------------------------------------------

def theta_from_unconstrained(u: torch.Tensor) -> torch.Tensor:
    """Source-ITR sinusoid wrapper for ``SourceItrSinAdapter.theta_from_unconstrained``."""
    return _SOURCE_ITR_SIN_ADAPTER.theta_from_unconstrained(u)


def unconstrained_from_theta(theta: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Source-ITR sinusoid wrapper for ``SourceItrSinAdapter.unconstrained_from_theta``."""
    return _SOURCE_ITR_SIN_ADAPTER.unconstrained_from_theta(theta)


def theta_to_cond_slice(theta: torch.Tensor) -> torch.Tensor:
    """Source-ITR sinusoid wrapper for cond-static injection parity tests."""
    return _SOURCE_ITR_SIN_ADAPTER.cond_slice_from_theta(theta)


def theta_to_rc_channel(
    theta: torch.Tensor, y_grid: torch.Tensor, Nx: int
) -> torch.Tensor:
    """Source-ITR sinusoid wrapper for the normalized R_c(y) spatial channel."""
    return _SOURCE_ITR_SIN_ADAPTER.spatial_channel_from_theta(theta, y_grid, Nx)


# ---------------------------------------------------------------------------
# Sensor operator M (Stage 2): a sparse, interface-proximal pixel mask.
# ---------------------------------------------------------------------------

def resolve_observation_times(
    t_grid: np.ndarray,
    requested_times: list[float] | tuple[float, ...],
    *,
    atol: float = 1e-8,
) -> tuple[list[int], np.ndarray]:
    """Resolve physical observation times without interpolation or substitution."""
    stored_grid = np.asarray(t_grid)
    grid = stored_grid.astype(np.float64, copy=False)
    scale = max(
        1.0,
        float(np.max(np.abs(grid))),
        max((abs(float(value)) for value in requested_times), default=0.0),
    )
    effective_atol = max(
        float(atol), 0.5 * np.finfo(np.float32).eps * scale
    )
    indices: list[int] = []
    resolved: list[float] = []
    for requested in requested_times:
        matches = np.flatnonzero(
            np.isclose(grid, float(requested), rtol=0.0, atol=effective_atol)
        )
        if matches.size != 1:
            nearest = float(grid[int(np.argmin(np.abs(grid - float(requested))))])
            raise ValueError(
                f"requested observation time {float(requested):g} is not uniquely "
                f"present in t_grid within atol={effective_atol:g}; "
                f"nearest is {nearest:g}."
            )
        idx = int(matches[0])
        if idx == 0:
            raise ValueError("observation times must be later than the source IC time.")
        indices.append(idx)
        resolved.append(float(grid[idx]))
    return indices, np.asarray(resolved, dtype=np.float64)


def resolve_sensor_y(
    y_grid: np.ndarray,
    requested_y: list[float] | tuple[float, ...] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve requested physical y coordinates to the nearest grid entries."""
    y = np.asarray(y_grid, dtype=np.float64)
    requested = np.asarray(requested_y, dtype=np.float64)
    indices = np.empty(requested.size, dtype=np.int64)
    resolved = np.empty(requested.size, dtype=np.float64)
    for k, value in enumerate(requested):
        idx = int(np.argmin(np.abs(y - value)))
        if idx == 0:
            local = y[1] - y[0]
        elif idx == y.size - 1:
            local = y[-1] - y[-2]
        else:
            local = min(y[idx] - y[idx - 1], y[idx + 1] - y[idx])
        if abs(float(y[idx]) - float(value)) > 0.5 * float(local) + 1e-12:
            raise ValueError(
                f"requested sensor y={value:g} is farther than half a local grid "
                f"spacing from the nearest location {float(y[idx]):g}."
            )
        indices[k] = idx
        resolved[k] = y[idx]
    if np.unique(indices).size != indices.size:
        raise ValueError("requested y sensors resolve to duplicate grid locations.")
    return indices, resolved


def interface_pair_columns(
    x_grid: np.ndarray, interface_x: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return the grid indices and coordinates immediately bracketing an interface."""
    x = np.asarray(x_grid, dtype=np.float64)
    left = np.flatnonzero(x < float(interface_x))
    right = np.flatnonzero(x > float(interface_x))
    if left.size == 0 or right.size == 0:
        raise ValueError("interface must lie strictly between two x-grid locations.")
    indices = np.array([int(left[-1]), int(right[0])], dtype=np.int64)
    return indices, x[indices]


def build_interface_pair_mask(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    *,
    interface_x: float = 0.5,
    requested_y: list[float] | tuple[float, ...] | np.ndarray = PAPER_SENSOR_Y,
) -> tuple[torch.Tensor, dict[str, np.ndarray]]:
    """Build the paired left/right interface-adjacent physical sensor mask."""
    x_indices, resolved_x = interface_pair_columns(x_grid, interface_x)
    y_indices, resolved_y = resolve_sensor_y(y_grid, requested_y)
    mask = np.zeros((len(x_grid), len(y_grid)), dtype=bool)
    mask[np.ix_(x_indices, y_indices)] = True
    metadata = {
        "requested_y": np.asarray(requested_y, dtype=np.float64),
        "resolved_y": resolved_y,
        "y_indices": y_indices,
        "x_indices": x_indices,
        "resolved_x": resolved_x,
    }
    return torch.from_numpy(mask), metadata

def build_interface_sensor_mask(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    *,
    interface_x: float = 0.5,
    x_halfwidth: float = 0.05,
    n_y: int = 8,
    requested_y: Optional[list[float] | tuple[float, ...] | np.ndarray] = None,
) -> torch.Tensor:
    """Boolean ``(Nx, Ny)`` mask of interface-proximal sensor pixels.

    Sensors sit where ``R_c(y)`` is identifiable: a band of x-columns within
    ``|x - interface_x| <= x_halfwidth`` crossed with ``n_y`` (roughly) evenly
    spaced y rows. If no column falls inside the band, the single nearest column
    is used. ``n_y >= Ny`` selects every row (a full y line of the band).

    Full-field observation is recovered by selecting all pixels, which callers
    express by passing ``mask=None`` (see :func:`apply_sensor_mask`); this helper
    always returns a *sparse* mask.
    """
    x = np.asarray(x_grid, dtype=np.float64)
    y = np.asarray(y_grid, dtype=np.float64)
    Nx, Ny = x.shape[0], y.shape[0]

    col_sel = np.abs(x - interface_x) <= x_halfwidth
    if not col_sel.any():
        col_sel = np.zeros(Nx, dtype=bool)
        col_sel[int(np.argmin(np.abs(x - interface_x)))] = True

    if requested_y is not None:
        row_idx, _ = resolve_sensor_y(y, requested_y)
    elif n_y >= Ny:
        row_idx = np.arange(Ny)
    else:
        row_idx = np.unique(np.linspace(0, Ny - 1, n_y).round().astype(int))

    mask = np.zeros((Nx, Ny), dtype=bool)
    cols = np.where(col_sel)[0]
    mask[np.ix_(cols, row_idx)] = True
    return torch.from_numpy(mask)


def apply_sensor_mask(
    field: torch.Tensor, mask: Optional[torch.Tensor]
) -> torch.Tensor:
    """Gather a ``(N, Nx, Ny, C)`` field at sensor pixels.

    ``mask`` is a boolean ``(Nx, Ny)`` tensor; the result is
    ``(N, n_sensors, C)``. ``mask=None`` means full-field: the field is returned
    flattened over space as ``(N, Nx*Ny, C)`` so the loss reduction matches the
    masked path elementwise (same mean-of-squares convention, no spatial weight
    change).
    """
    N, Nx, Ny, C = field.shape
    if mask is None:
        return field.reshape(N, Nx * Ny, C)
    return field[:, mask]  # boolean index over dims (1, 2) -> (N, n_sensors, C)


# ---------------------------------------------------------------------------
# Checkpoint loading (mirrors src/operators/eval.py construction).
# ---------------------------------------------------------------------------

@dataclass
class LoadedModel:
    model: FNO2d
    config: dict
    dims: object
    mu_global: float
    sigma_global: float


def _adapt_legacy_zero_source_time_state(
    state: dict[str, torch.Tensor], config: dict, dims
) -> dict[str, torch.Tensor]:
    model_cfg = config["model"]["parameters"]
    checkpoint_cond_dim = int(
        model_cfg.get("cond_static_dim", dims.cond_static_dim)
    )
    if checkpoint_cond_dim == dims.cond_static_dim:
        return state

    benchmark_cfg = config.get("benchmark", {})
    benchmark = (
        str(benchmark_cfg.get("name", "forcing"))
        if isinstance(benchmark_cfg, dict)
        else str(benchmark_cfg)
    )
    forcing_benchmarks = {"forcing", "forcing_itr_sin"}
    if (
        benchmark not in forcing_benchmarks
        or checkpoint_cond_dim != dims.cond_static_dim + 1
        or not dims.use_forcing_time_aug
    ):
        return state

    cond_key = "cond_mlp.net.0.weight"
    aug_key = "forcing_aug_mlp.0.weight"
    cond_weight = state[cond_key]
    aug_weight = state[aug_key]
    forcing_embed_dim = int(model_cfg.get("forcing_embed_dim", 64))
    if cond_weight.shape[1] != checkpoint_cond_dim + forcing_embed_dim:
        return state
    if aug_weight.shape[1] != forcing_embed_dim + 2:
        return state

    # Inverse observations always use s=0, so the retired t_s_norm feature is
    # identically zero. Removing its two input columns is therefore exact.
    adapted = dict(state)
    adapted[cond_key] = torch.cat(
        [cond_weight[:, :1], cond_weight[:, 2:]], dim=1
    )
    adapted[aug_key] = aug_weight[:, :-1]
    return adapted


def load_checkpoint(ckpt_path: str, device: str = "cpu") -> LoadedModel:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["conf"]
    dims = problem_from_config(config).dims
    model_cfg = config["model"]["parameters"]
    model = FNO2d(
        modes1=model_cfg["modes1"],
        modes2=model_cfg["modes2"],
        width=model_cfg["width"],
        in_channels=dims.in_channels,
        out_channels=model_cfg.get("out_channels", 1),
        n_layers=model_cfg.get("n_layers", 4),
        cond_static_dim=dims.cond_static_dim,
        cond_hidden=model_cfg.get("cond_hidden", 256),
        temporal_token_dim=dims.temporal_token_dim,
        temporal_hidden=model_cfg.get("temporal_hidden", 128),
        forcing_embed_dim=model_cfg.get("forcing_embed_dim", 64),
        forcing_spatial_dim=model_cfg.get("forcing_spatial_dim", 16),
        use_forcing_time_aug=dims.use_forcing_time_aug,
        forcing_cond_mode=model_cfg.get("forcing_cond_mode", "both"),
        forcing_spatial_mode=model_cfg.get("forcing_spatial_mode", "broadcast"),
        forcing_extender_grid_size=model_cfg.get("forcing_extender_grid_size", 16),
        forcing_extender_heads=model_cfg.get("forcing_extender_heads", 4),
        forcing_extender_depth=model_cfg.get("forcing_extender_depth", 1),
        forcing_extender_condition_on_rc=model_cfg.get(
            "forcing_extender_condition_on_rc", False
        ),
        forcing_extender_rc_cond_index=dims.forcing_extender_rc_cond_index,
        forcing_extender_physics_hidden=model_cfg.get(
            "forcing_extender_physics_hidden", 16
        ),
        forcing_extender_interface_x_norm=model_cfg.get(
            "forcing_extender_interface_x_norm", 0.5
        ),
        s_y_channel=dims.s_y_channel,
        padding_reference_resolution=model_cfg.get("padding_reference_resolution"),
        padding_mode=model_cfg.get("padding_mode", "zeros"),
        cin_exclude_padding=model_cfg.get("cin_exclude_padding", False),
        hard_right_dirichlet=model_cfg.get("hard_right_dirichlet", False),
    )
    state = _adapt_legacy_zero_source_time_state(
        ckpt["model_state"], config, dims
    )
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    mu = float(ckpt.get("mu_global"))
    sigma = float(ckpt.get("sigma_global"))
    return LoadedModel(model=model, config=config, dims=dims, mu_global=mu, sigma_global=sigma)


# ---------------------------------------------------------------------------
# Dataset construction over a local source_itr_sin artifact directory.
# ---------------------------------------------------------------------------

def prepare_inversion_dataset(
    *,
    benchmark: str,
    n_invert: int,
    n_calibration: int,
    out_dir: str,
    nx: int = 100,
    ny: int = 100,
    save_stride: int = 2,
    dataset_seed: int = INVERSION_DATASET_SEED,
    generate_fn=None,
) -> dict[str, object]:
    """Generate one disjoint held-out dataset and return its slice partition.

    The single dataset of ``n_invert + n_calibration`` sims is generated with a
    non-zero ``dataset_seed`` so its draws are disjoint from the training corpus
    (``rng_seed=0``). The first ``n_invert`` sims are the inversion slice; the
    next ``n_calibration`` are the calibration slice. Both entry points (this
    module's ``main`` and ``scripts/run_inverse_sensor_sweep.py``) route through
    here so calibration and inversion always share one dataset -- hence one
    ``_dataset_fingerprint`` -- while remaining disjoint by construction.

    ``generate_fn`` defaults to ``data.generate_dataset.generate_sim_data`` and
    is injectable so tests can substitute a lightweight writer.
    """
    if n_invert <= 0:
        raise ValueError(f"n_invert must be positive, got {n_invert}.")
    if n_calibration <= 0:
        raise ValueError(f"n_calibration must be positive, got {n_calibration}.")
    if generate_fn is None:
        from data.generate_dataset import generate_sim_data

        generate_fn = generate_sim_data
    total = n_invert + n_calibration
    os.makedirs(out_dir, exist_ok=True)
    generate_fn(
        num_sims=total,
        save_dir=out_dir,
        benchmark=benchmark,
        nx=nx,
        ny=ny,
        save_stride=save_stride,
        rng_seed=dataset_seed,
    )
    return {
        "data_dir": str(out_dir),
        "invert_ids": list(range(n_invert)),
        "calibration_ids": list(range(n_invert, total)),
    }


def build_dataset_from_dir(
    data_dir: str,
    config: dict,
    *,
    mu_global: float,
    sigma_global: float,
    n_invert: int | None = None,
    n_calibration: int | None = None,
) -> SnapshotPairDataset:
    """Construct a SnapshotPairDataset over ``data_dir`` with the checkpoint's stats.

    Uses the checkpoint's baked ``(mu_global, sigma_global)`` so normalization
    matches training exactly (never re-normalized per-sample).

    The directory is a freshly generated held-out dataset (disjoint from the
    training corpus by seed). It is partitioned into two disjoint slices: the
    first ``n_invert`` sims are the inversion slice (``test``) and the next
    ``n_calibration`` sims are the calibration slice (``val``); ``train`` is
    empty. Keeping calibration and inversion sims disjoint avoids measuring
    surrogate error on the very sims being inverted. When ``n_calibration`` is
    ``None``/0 the whole corpus (or the first ``n_invert`` sims) is the inversion
    slice and no calibration slice is reserved.
    """
    traj_path = os.path.join(data_dir, "trajectories.npy")
    x_path = os.path.join(data_dir, "x_grid.npy")
    y_path = os.path.join(data_dir, "y_grid.npy")
    t_path = os.path.join(data_dir, "t_grid.npy")
    sim_params_path = os.path.join(data_dir, "sim_params.npy")

    trajectories, x_grid, y_grid, t_grid = load_sim_data(traj_path, x_path, y_path, t_path)
    sim_params = np.load(sim_params_path, allow_pickle=True)
    num_sims = trajectories.shape[0]
    n_calib = int(n_calibration or 0)
    n_inv = int(n_invert) if n_invert is not None else num_sims - n_calib
    if n_inv <= 0:
        raise ValueError(
            f"n_invert resolves to {n_inv} for a {num_sims}-sim dataset."
        )
    if n_inv + n_calib > num_sims:
        raise ValueError(
            f"dataset has {num_sims} sims but the requested partition needs "
            f"{n_inv} inversion + {n_calib} calibration = {n_inv + n_calib}."
        )
    invert_ids = np.arange(n_inv)
    calibration_ids = np.arange(n_inv, n_inv + n_calib)

    dt_path = os.path.join(data_dir, "dt.npy")
    ramp_path = os.path.join(data_dir, "ramp_seconds.npy")
    dt = float(np.load(dt_path)) if os.path.exists(dt_path) else None
    ramp_seconds = float(np.load(ramp_path)) if os.path.exists(ramp_path) else 0.0

    problem = problem_from_config(config)
    ds = SnapshotPairDataset(
        trajectories=trajectories,
        sim_params=sim_params,
        t_grid=t_grid,
        x_grid=x_grid,
        y_grid=y_grid,
        sim_ids=np.arange(num_sims),
        mu_global=mu_global,
        sigma_global=sigma_global,
        problem=problem,
        dt=dt,
        ramp_seconds=ramp_seconds,
    )
    ds._split_ids = {
        "train": np.array([], dtype=int),
        "val": calibration_ids,
        "test": invert_ids,
    }
    required_times = getattr(problem, "required_observation_times", ())
    if required_times:
        resolve_observation_times(ds.t_grid, required_times)
    return ds


# ---------------------------------------------------------------------------
# Direct-from-IC observation operator.
# ---------------------------------------------------------------------------

@dataclass
class ObservationSet:
    """theta-independent scaffolding + full-field targets for one sim.

    Built once from ``problem.build_item(ds, sid, s=0, j=n)`` for each
    observation time index ``n``. ``spatial`` carries the four theta-independent
    channels verbatim; channel ``RC_Y_CHANNEL`` is overwritten per-theta in the
    loss. ``cond`` carries the lead/patch slots; columns 1:3 are overwritten.
    """

    sid: int
    time_indices: list[int]
    spatial: torch.Tensor       # (N, Nx, Ny, C)
    cond: torch.Tensor          # (N, cond_dim)
    forcing_seq: torch.Tensor   # (N, M, token_dim)
    targets: torch.Tensor       # (N, Nx, Ny, 1)  normalized full-field T
    y_grid: torch.Tensor        # (Ny,)
    Nx: int
    theta_true: Optional[torch.Tensor] = None  # (theta_dim,) physical, if known
    mask: Optional[torch.Tensor] = None  # bool (Nx, Ny) sensor pixels; None=full-field
    observation_times: Optional[np.ndarray] = None
    sensor_layout: str = "full_field"
    requested_y: Optional[np.ndarray] = None
    resolved_y: Optional[np.ndarray] = None
    y_indices: Optional[np.ndarray] = None
    resolved_x: Optional[np.ndarray] = None
    interface_x: float = 0.5
    sensor_x_halfwidth: Optional[float] = None
    noise_seed: Optional[int] = None
    noise_std: float = 0.0


def build_observation_set(
    ds: SnapshotPairDataset,
    sid: int,
    time_indices: list[int],
    device: str = "cpu",
    *,
    adapter: InverseAdapter = _SOURCE_ITR_SIN_ADAPTER,
    interface_x: float = 0.5,
    sensor_x_halfwidth: Optional[float] = None,
    sensor_n_y: int = 8,
    sensor_layout: Optional[str] = None,
    sensor_y: Optional[list[float] | tuple[float, ...] | np.ndarray] = None,
    observation_times: Optional[np.ndarray] = None,
) -> ObservationSet:
    """Assemble direct-from-IC observations for ``sid`` at the given time indices.

    For each ``n`` we call the real ``build_item(ds, sid, s=0, j=n)``: with
    ``s=0`` the input snapshot *is* the uniform-300 K IC and ``t_s=0``, so the
    operator is a pure function of theta. The returned spatial/cond/forcing are
    the exact training-pipeline tensors; ``targets`` are the FV trajectory
    snapshots (normalized).

    When ``sensor_x_halfwidth`` is ``None`` the observation is full-field
    (Stage 1). Otherwise a sparse interface-proximal sensor mask M (Stage 2) is
    attached: a band of x-columns within ``sensor_x_halfwidth`` of
    ``interface_x`` crossed with ``sensor_n_y`` y rows. The full ``targets``
    field is still stored; the loss restricts to the masked pixels.
    """
    spatials, conds, fseqs, targets = [], [], [], []
    for n in time_indices:
        item = ds.problem.build_item(ds, int(sid), 0, int(n))
        spatials.append(torch.from_numpy(item["spatial"]))
        conds.append(torch.from_numpy(item["cond_static"]))
        fseqs.append(torch.from_numpy(item["forcing_seq"]))
        targets.append(torch.from_numpy(item["Y"]))

    spatial = torch.stack(spatials).to(device)
    cond = torch.stack(conds).to(device)
    forcing_seq = torch.stack(fseqs).to(device)
    target = torch.stack(targets).to(device)
    y_grid = torch.from_numpy(np.asarray(ds.y_grid, dtype=np.float32)).to(device)

    layout = sensor_layout
    if layout is None:
        layout = "interface_band" if sensor_x_halfwidth is not None else "full_field"
    requested_y = None if sensor_y is None else np.asarray(sensor_y, dtype=np.float64)
    mask = None
    resolved_y = y_indices = resolved_x = None
    if layout == "interface_pair":
        if requested_y is None:
            requested_y = np.asarray(PAPER_SENSOR_Y, dtype=np.float64)
        mask, sensor_meta = build_interface_pair_mask(
            np.asarray(ds.x_grid),
            np.asarray(ds.y_grid),
            interface_x=interface_x,
            requested_y=requested_y,
        )
        resolved_y = sensor_meta["resolved_y"]
        y_indices = sensor_meta["y_indices"]
        resolved_x = sensor_meta["resolved_x"]
        mask = mask.to(device)
    elif layout == "interface_band":
        if requested_y is None:
            requested_y = np.linspace(
                float(ds.y_grid[0]), float(ds.y_grid[-1]), int(sensor_n_y)
            )
        mask = build_interface_sensor_mask(
            np.asarray(ds.x_grid),
            np.asarray(ds.y_grid),
            interface_x=interface_x,
            x_halfwidth=(
                0.05 if sensor_x_halfwidth is None else float(sensor_x_halfwidth)
            ),
            n_y=int(sensor_n_y),
            requested_y=requested_y,
        ).to(device)
        y_indices, resolved_y = resolve_sensor_y(ds.y_grid, requested_y)
        resolved_x = np.asarray(ds.x_grid, dtype=np.float64)[
            np.flatnonzero(mask.detach().cpu().numpy().any(axis=1))
        ]
    elif layout != "full_field":
        raise ValueError(
            f"Unknown sensor layout {layout!r}; expected full_field, "
            "interface_band, or interface_pair."
        )

    params = ds.sim_params[int(sid)]
    theta_true = adapter.theta_from_sim_params(
        params, dtype=torch.float32, device=device
    )
    return ObservationSet(
        sid=int(sid),
        time_indices=list(time_indices),
        spatial=spatial,
        cond=cond,
        forcing_seq=forcing_seq,
        targets=target,
        y_grid=y_grid,
        Nx=spatial.shape[1],
        theta_true=theta_true,
        mask=mask,
        observation_times=(
            np.asarray(observation_times, dtype=np.float64)
            if observation_times is not None
            else np.asarray(ds.t_grid, dtype=np.float64)[list(time_indices)]
        ),
        sensor_layout=layout,
        requested_y=requested_y,
        resolved_y=resolved_y,
        y_indices=y_indices,
        resolved_x=resolved_x,
        interface_x=float(interface_x),
        sensor_x_halfwidth=sensor_x_halfwidth,
    )


def predict_fullfield(
    model: FNO2d,
    obs: ObservationSet,
    theta: torch.Tensor,
    adapter: InverseAdapter = _SOURCE_ITR_SIN_ADAPTER,
) -> torch.Tensor:
    """G_theta over all observation times: returns (N, Nx, Ny, 1) normalized T.

    ``theta`` is a single physical (4,) vector. Both injection points are
    rebuilt from theta and spliced into the (otherwise detached) scaffolding so
    autograd flows only through cond[1:3] and channel RC_Y_CHANNEL.

    The splice is done out-of-place (``torch.where`` / ``torch.cat`` rather than
    indexed assignment) so the function is safe under ``vmap`` — needed for the
    Stage 3 ``torch.autograd.functional.jacobian`` sensitivity diagnostic.
    """
    N = obs.spatial.shape[0]
    Nx, Ny, C = (obs.spatial.shape[1], obs.spatial.shape[2], obs.spatial.shape[3])

    channel_index = adapter.spatial_channel_index()
    if channel_index is None:
        spatial = obs.spatial
    else:
        theta_channel = adapter.spatial_channel_from_theta(theta, obs.y_grid, Nx)
        channel_b = theta_channel.unsqueeze(0).unsqueeze(-1).expand(N, Nx, Ny, C)
        chan_sel = torch.arange(C, device=obs.spatial.device) == channel_index
        spatial = torch.where(chan_sel, channel_b, obs.spatial)

    start, stop = adapter.cond_slice_indices()
    cond_slice = adapter.cond_slice_from_theta(theta)
    cond = torch.cat(
        [
            obs.cond[:, :start],
            cond_slice.unsqueeze(0).expand(N, stop - start),
            obs.cond[:, stop:],
        ],
        dim=1,
    )

    return model(spatial, cond, obs.forcing_seq)


# ---------------------------------------------------------------------------
# Multistart sampling (Stage 3): Latin-hypercube starts over the theta-box.
# ---------------------------------------------------------------------------

def lhs_unit_samples(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Jittered Latin-hypercube sample on the open unit cube ``(0, 1)^d``.

    Each of the ``d`` columns is stratified into ``n`` equal bins with one
    sample jittered inside each bin, then the bins are permuted independently
    per column. Returns ``(n, d)``.
    """
    cut = np.linspace(0.0, 1.0, n + 1)
    lo = cut[:n]
    hi = cut[1:]
    pts = lo[:, None] + rng.uniform(size=(n, d)) * (hi - lo)[:, None]
    for j in range(d):
        rng.shuffle(pts[:, j])
    return pts


def lhs_starts_u(n_starts: int, rng: np.random.Generator, eps: float = 1e-3) -> np.ndarray:
    """LHS starts in unconstrained ``u``-space, stratified over the theta-box.

    LHS is taken on the sigmoid-fraction cube (which *is* the physical box via
    :func:`theta_from_unconstrained`), then mapped to ``u`` with the logit so the
    starts evenly tile each sinusoidal parameter's range. Returns ``(n_starts, 2)``.
    """
    return _SOURCE_ITR_SIN_ADAPTER.lhs_starts_unconstrained(n_starts, rng, eps)


# ---------------------------------------------------------------------------
# MAP optimization.
# ---------------------------------------------------------------------------

@dataclass
class InversionConfig:
    n_starts: int = 8
    adam_steps: int = 400
    adam_lr: float = 0.05
    lbfgs_steps: int = 50
    seed: int = 0
    reg_weight: float = 0.0  # weak Gaussian-in-u optimization regularizer
    device: str = "cpu"
    start_sampling: str = "lhs"  # "lhs" (Stage 3) or "normal" (legacy N(0,1))
    optimizer: str = "adam_lbfgs"
    nm_maxiter: int = 60
    nm_xatol: float = 1e-4
    nm_fatol: float = 1e-14


@dataclass
class InversionResult:
    theta_hat: torch.Tensor
    loss: float
    theta_true: Optional[torch.Tensor]
    per_start_losses: list[float] = field(default_factory=list)
    per_start_theta: list[torch.Tensor] = field(default_factory=list)


def _data_loss(
    model: FNO2d,
    obs: ObservationSet,
    u: torch.Tensor,
    reg_weight: float,
    adapter: InverseAdapter = _SOURCE_ITR_SIN_ADAPTER,
):
    theta = adapter.theta_from_unconstrained(u)
    pred = predict_fullfield(model, obs, theta, adapter)
    pred_obs = apply_sensor_mask(pred, obs.mask)
    target_obs = apply_sensor_mask(obs.targets, obs.mask)
    resid = pred_obs - target_obs
    loss = 0.5 * torch.mean(resid ** 2)
    if reg_weight > 0:
        loss = loss + 0.5 * reg_weight * torch.mean(u ** 2)
    return loss


def invert_sim(
    model: FNO2d,
    obs: ObservationSet,
    cfg: InversionConfig,
    adapter: InverseAdapter = _SOURCE_ITR_SIN_ADAPTER,
) -> InversionResult:
    device = cfg.device
    rng = np.random.default_rng(cfg.seed)
    per_start_losses: list[float] = []
    per_start_theta: list[torch.Tensor] = []
    best_loss = float("inf")
    best_theta = None

    if cfg.start_sampling == "lhs":
        starts_u = adapter.lhs_starts_unconstrained(cfg.n_starts, rng)
    elif cfg.start_sampling == "normal":
        starts_u = rng.normal(0.0, 1.0, size=(cfg.n_starts, adapter.theta_dim)).astype(np.float32)
    else:
        raise ValueError(f"Unknown start_sampling {cfg.start_sampling!r}")

    for start in range(cfg.n_starts):
        u0 = torch.from_numpy(starts_u[start]).to(device)
        if cfg.optimizer == "nelder_mead":
            def objective(u_np):
                candidate = torch.as_tensor(u_np, dtype=u0.dtype, device=device)
                with torch.no_grad():
                    return float(
                        _data_loss(model, obs, candidate, cfg.reg_weight, adapter)
                    )

            optimized = minimize(
                objective,
                starts_u[start].astype(np.float64),
                method="Nelder-Mead",
                options={
                    "maxiter": cfg.nm_maxiter,
                    "xatol": cfg.nm_xatol,
                    "fatol": cfg.nm_fatol,
                },
            )
            u = torch.as_tensor(optimized.x, dtype=u0.dtype, device=device)
        elif cfg.optimizer == "adam_lbfgs":
            u = u0.clone().requires_grad_(True)
            adam = torch.optim.Adam([u], lr=cfg.adam_lr)
            for _ in range(cfg.adam_steps):
                adam.zero_grad()
                loss = _data_loss(model, obs, u, cfg.reg_weight, adapter)
                loss.backward()
                adam.step()

            lbfgs = torch.optim.LBFGS(
                [u], max_iter=cfg.lbfgs_steps, line_search_fn="strong_wolfe"
            )

            def closure():
                lbfgs.zero_grad()
                loss = _data_loss(model, obs, u, cfg.reg_weight, adapter)
                loss.backward()
                return loss

            lbfgs.step(closure)
        else:
            raise ValueError(f"Unknown optimizer {cfg.optimizer!r}")

        with torch.no_grad():
            final_loss = float(_data_loss(model, obs, u, cfg.reg_weight, adapter))
            theta = adapter.theta_from_unconstrained(u).detach().cpu()
        per_start_losses.append(final_loss)
        per_start_theta.append(theta)
        if final_loss < best_loss:
            best_loss = final_loss
            best_theta = theta

    return InversionResult(
        theta_hat=best_theta,
        loss=best_loss,
        theta_true=None if obs.theta_true is None else obs.theta_true.detach().cpu(),
        per_start_losses=per_start_losses,
        per_start_theta=per_start_theta,
    )


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------


def summarize(
    result: InversionResult,
    y_grid: torch.Tensor,
    adapter: InverseAdapter = _SOURCE_ITR_SIN_ADAPTER,
) -> dict:
    return adapter.summarize_result(result, y_grid)


def _default_time_indices(Nt: int) -> list[int]:
    """Early / mid / late observation snapshots (skip t=0, which is the IC)."""
    early = max(1, Nt // 4)
    mid = Nt // 2
    late = Nt - 1
    return sorted(set([early, mid, late]))


# ---------------------------------------------------------------------------
# Reported statistic 2: FV-verified sensor residual at theta_hat.
#
# The FNO is the *exploration* engine; the conservative FV solver is the
# *accuracy* engine. We rebuild the EXACT data-generation operator via
# ``ds.problem.configure_solver`` (k=3/35 layers, q_left=0, the patch source,
# ``interface_R=[Rc(y)]``) and re-evaluate the FNO MAP estimate with it, then
# report the FV sensor residual in Kelvin against the measurement noise floor.
# It never sets the UQ width.
#
# ``fv_refine_report(polish=True)`` reaches the retired Nelder-Mead polish at
# the bottom of this file; main() never passes it.
# ---------------------------------------------------------------------------

def build_fv_base_kwargs(ds: SnapshotPairDataset) -> dict:
    """Source-ITR sinusoid wrapper for exact FV base kwargs."""
    return _SOURCE_ITR_SIN_ADAPTER.fv_base_kwargs(ds)


def _theta_to_numpy(theta) -> np.ndarray:
    if torch.is_tensor(theta):
        return theta.detach().cpu().numpy().astype(np.float64)
    return np.asarray(theta, dtype=np.float64)


def sensor_mask_for_grid(
    obs: ObservationSet,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
) -> Optional[torch.Tensor]:
    """Resolve an observation's physical sensor definition on a numerical grid."""
    if obs.sensor_layout == "full_field":
        return None
    if obs.sensor_layout == "interface_pair":
        mask, _ = build_interface_pair_mask(
            x_grid,
            y_grid,
            interface_x=obs.interface_x,
            requested_y=(
                PAPER_SENSOR_Y if obs.requested_y is None else obs.requested_y
            ),
        )
        return mask
    if obs.sensor_layout == "interface_band":
        requested_x = (
            np.asarray(obs.resolved_x, dtype=np.float64)
            if obs.resolved_x is not None
            else np.asarray([obs.interface_x], dtype=np.float64)
        )
        x_indices, _ = resolve_sensor_y(x_grid, requested_x)
        requested_y = (
            np.asarray(obs.requested_y, dtype=np.float64)
            if obs.requested_y is not None
            else np.asarray(obs.resolved_y, dtype=np.float64)
        )
        y_indices, _ = resolve_sensor_y(y_grid, requested_y)
        mask = np.zeros((len(x_grid), len(y_grid)), dtype=bool)
        mask[np.ix_(x_indices, y_indices)] = True
        return torch.from_numpy(mask)
    raise ValueError(f"Unknown observation sensor layout {obs.sensor_layout!r}.")


def fv_predict_masked(
    ds: SnapshotPairDataset,
    base_kwargs: dict,
    sid: int,
    theta,
    time_indices: list[int],
    *,
    mu_global: float,
    sigma_global: float,
    mask: Optional[torch.Tensor],
    adapter: InverseAdapter = _SOURCE_ITR_SIN_ADAPTER,
    obs: Optional[ObservationSet] = None,
) -> torch.Tensor:
    """FV forward of ``theta`` at the observation sensors/times (normalized).

    Overrides only the two sinusoidal parameters (and the mirrored ``R_c`` slot) in a
    copy of the stored ``sim_params[sid]`` so the patch / IC / interface_x are
    bit-identical to data generation, rebuilds the solver, integrates from the
    stored IC, selects the observation snapshots by nearest solver time, and
    normalizes with the checkpoint's baked global stats. Returns a CPU tensor
    ``(N, n_sensors, 1)`` (or ``(N, Nx*Ny, 1)`` full-field when ``mask`` is None).
    """
    params = dict(ds.sim_params[int(sid)])
    updated = adapter.inject_theta_into_sim_params(params, theta)
    solver = ds.problem.configure_solver(updated, base_kwargs)
    T0 = adapter.fv_initial_condition(ds, updated, solver)
    t_solver, _x, _y, T_hist = solver.solve(T0=T0, store_trajectory=True)

    t_obs = (
        np.asarray(obs.observation_times, dtype=np.float64)
        if obs is not None and obs.observation_times is not None
        else np.asarray(ds.t_grid, dtype=np.float64)[list(time_indices)]
    )
    t_solver = np.asarray(t_solver, dtype=np.float64)
    sel, _ = resolve_observation_times(t_solver, t_obs, atol=1e-8)

    T_sel = np.asarray(T_hist)[sel]  # (N, Nx, Ny) physical Kelvin
    T_norm = (T_sel - mu_global) / (sigma_global + T_EPS)
    field = torch.from_numpy(T_norm.astype(np.float32)).unsqueeze(-1)  # (N,Nx,Ny,1)
    resolved_mask = sensor_mask_for_grid(obs, _x, _y) if obs is not None else mask
    return apply_sensor_mask(field, resolved_mask)


def _masked_half_mse(pred_obs: torch.Tensor, target_obs: torch.Tensor) -> float:
    """0.5 * mean squared residual over the observed pixels.

    Same convention as the MAP data loss (:func:`_data_loss` with reg_weight=0),
    so FV and FNO residuals are directly comparable to the inversion ``loss``.
    """
    resid = pred_obs - target_obs
    return 0.5 * float(torch.mean(resid ** 2))


@dataclass
class FVRefineResult:
    fno_resid: float            # 0.5*mean (FNO(theta_hat) - obs)^2  over sensors
    fv_resid: float             # 0.5*mean (FV(theta_hat)  - obs)^2  over sensors
    fno_vs_fv_resid: float      # C_FNO(theta_hat): 0.5*mean (FNO - FV)^2
    theta_fv_polish: Optional[torch.Tensor] = None
    fv_polish_resid: Optional[float] = None
    fv_polish_evals: Optional[int] = None


def fv_refine_report(
    model: FNO2d,
    ds: SnapshotPairDataset,
    base_kwargs: dict,
    obs: ObservationSet,
    theta_hat: torch.Tensor,
    *,
    mu_global: float,
    sigma_global: float,
    polish: bool = False,
    polish_maxiter: int = 60,
    adapter: InverseAdapter = _SOURCE_ITR_SIN_ADAPTER,
) -> FVRefineResult:
    """FNO-vs-FV residual report at ``theta_hat`` (+ optional FV polish)."""
    mask_cpu = None if obs.mask is None else obs.mask.cpu()
    target_obs = apply_sensor_mask(obs.targets.detach().cpu(), mask_cpu)

    with torch.no_grad():
        fno_pred = predict_fullfield(
            model, obs, theta_hat.to(obs.spatial.device), adapter
        )
    fno_obs = apply_sensor_mask(fno_pred.detach().cpu(), mask_cpu)
    fv_obs = fv_predict_masked(
        ds, base_kwargs, obs.sid, theta_hat, obs.time_indices,
        mu_global=mu_global, sigma_global=sigma_global, mask=mask_cpu,
        adapter=adapter, obs=obs,
    )

    out = FVRefineResult(
        fno_resid=_masked_half_mse(fno_obs, target_obs),
        fv_resid=_masked_half_mse(fv_obs, target_obs),
        fno_vs_fv_resid=_masked_half_mse(fno_obs, fv_obs),
    )
    if polish:
        theta_p, fv_polish_resid, nfev = fv_polish(
            ds, base_kwargs, obs, theta_hat,
            mu_global=mu_global, sigma_global=sigma_global, maxiter=polish_maxiter,
            adapter=adapter,
        )
        out.theta_fv_polish = theta_p
        out.fv_polish_resid = fv_polish_resid
        out.fv_polish_evals = nfev
    return out


def fv_refine_summary(
    res: FVRefineResult,
    obs: ObservationSet,
    y_grid: torch.Tensor,
    adapter: InverseAdapter = _SOURCE_ITR_SIN_ADAPTER,
) -> dict:
    """Flatten a :func:`fv_refine_report` into scalar CSV columns."""
    return adapter.fv_refine_summary(res, obs, y_grid)


def fv_verification_summary(
    res: FVRefineResult,
    *,
    sigma_global: float,
    noise_std_norm: float,
) -> dict:
    """Reported form of statistic 2: the FV sensor residual in Kelvin.

    ``FVRefineResult`` carries half-MSEs in normalized temperature units, which
    are not quotable. The reported number is the physical RMS
    ``sigma_global * sqrt(2 * fv_resid)`` and its ratio to the measurement
    noise std; a ratio at or below 1 is the claim that the recovered theta
    reproduces the observations to within noise *under the real solver*. The
    FNO-vs-FV term is converted the same way so the surrogate's own share of
    the residual is visible next to it.

    ``fv_resid_over_noise`` is NaN when the run is noiseless: the residual has
    no floor to be measured against.
    """

    def _rms_K(half_mse: float) -> float:
        return float(sigma_global) * float(np.sqrt(2.0 * float(half_mse)))

    noise_std_K = float(sigma_global) * float(noise_std_norm)
    fv_rms_K = _rms_K(res.fv_resid)
    return {
        "fv_resid_rms_K": fv_rms_K,
        "fno_vs_fv_rms_K": _rms_K(res.fno_vs_fv_resid),
        "fv_resid_over_noise": (
            fv_rms_K / noise_std_K if noise_std_K > 0.0 else float("nan")
        ),
    }


# ---------------------------------------------------------------------------
# Parameter-specific profile-likelihood intervals.
#
# iid Gaussian sensor noise is added once to the observations, and the
# likelihood uses the frozen validation-calibrated variance
# ``sigma_eff^2 = (sigma_meas^2 + sigma_FNO,cal^2) * D``. Each parameter is
# profiled while the other coordinates are nuisance parameters.
#
# ``D`` is the frozen residual design effect (``_residual_design_effect``): the
# sum-of-squares likelihood counts every sensor/time residual as independent,
# but validation shows the FNO surrogate error is correlated across sensors and
# times, so the raw count overstates the information and the interval would be
# too narrow. ``D = 1^T Sigma 1 / tr(Sigma)`` on the total observation
# covariance deflates the effective count to ``m / D`` and widens the interval
# by ``sqrt(D)``; the iid measurement term keeps ``D -> 1`` when noise dominates.
#
# The surrogate scale and design effect are calibrated once on validation
# sensors and frozen before test inversion; test-local FNO/FV discrepancies
# never set them.
# ---------------------------------------------------------------------------

# Wilks thresholds on (NLL - NLL_min): 0.5 * chi2_inv(level, df=1). A profile
# confidence interval at `level` is where 2*(NLL - NLL_min) <= chi2_inv, i.e.
# (NLL - NLL_min) <= the value below.


def _residual_design_effect(
    residuals: np.ndarray, sigma_meas2: float
) -> dict[str, float]:
    """Correlation-aware effective-sample deflation for the FNO residuals.

    ``residuals`` is ``(n_sim, m)``: one flattened sensor/time residual vector
    per validation simulation, evaluated at the true theta, so it is pure
    surrogate error (measurement noise is added later and is iid). The profile
    likelihood sums the ``m`` residuals as if independent; positively correlated
    surrogate errors make that count too generous and the interval too narrow.
    The design effect on the total observation covariance
    ``Sigma = sigma_meas2 * I + C_FNO`` is

        D = 1^T Sigma 1 / tr(Sigma)
          = (m*sigma_meas2 + 1^T C_FNO 1) / (m*sigma_meas2 + tr(C_FNO)),

    which deflates the effective per-simulation information from ``m`` to
    ``m / D``; scaling ``sigma_eff^2`` by ``D`` widens the interval by ``sqrt(D)``.
    The iid measurement term is uncorrected: as it dominates, ``D -> 1``. ``D``
    is clamped to ``>= 1`` so the correction never claims more information than
    the independent baseline. ``1^T C_FNO 1`` is estimated as the sample
    variance of the per-simulation residual sums and ``tr(C_FNO)`` as the sum of
    per-component sample variances, so nothing is inverted and the estimate is
    stable even when ``m`` exceeds ``n_sim``.
    """
    residuals = np.asarray(residuals, dtype=np.float64)
    m = int(residuals.shape[1]) if residuals.ndim == 2 else 0
    if residuals.ndim != 2 or residuals.shape[0] < 2 or m == 0:
        return {
            "design_effect": 1.0,
            "design_effect_raw": 1.0,
            "fno_var_trace": 0.0,
            "fno_var_sum": 0.0,
            "dim": m,
            "n_eff": float(m),
        }
    sigma_meas2 = max(float(sigma_meas2), 0.0)
    fno_var_trace = float(np.sum(np.var(residuals, axis=0, ddof=1)))
    fno_var_sum = float(np.var(residuals.sum(axis=1), ddof=1))
    total_trace = m * sigma_meas2 + fno_var_trace
    total_sum = m * sigma_meas2 + fno_var_sum
    design_raw = total_sum / total_trace if total_trace > 0.0 else 1.0
    design = max(design_raw, 1.0)
    return {
        "design_effect": float(design),
        "design_effect_raw": float(design_raw),
        "fno_var_trace": fno_var_trace,
        "fno_var_sum": fno_var_sum,
        "dim": m,
        "n_eff": float(m) / float(design),
    }


def effective_sigma2(
    noise_std: float,
    c_fno: float = 0.0,
    *,
    sigma_fno_cal: Optional[float] = None,
    design_effect: float = 1.0,
) -> float:
    """Effective normalized variance from measurement and frozen FNO scales.

    ``sigma_fno_cal`` is the validation-calibrated RMS scale used by production
    inversion. The legacy ``c_fno`` half-MSE argument remains available for
    old callers, but the CLI never uses a test-local discrepancy for UQ.

    ``design_effect`` is the frozen correlation-aware deflation factor from
    ``_residual_design_effect``; scaling the variance by it (clamped to
    ``>= 1``) widens the profile interval to account for correlated surrogate
    error that the sum-of-squares likelihood would otherwise treat as
    independent. It defaults to ``1.0`` so old callers and old calibration
    artifacts keep the uncorrected variance.
    """
    fno_variance = (
        float(sigma_fno_cal) ** 2
        if sigma_fno_cal is not None
        else 2.0 * float(c_fno)
    )
    base = float(noise_std) ** 2 + fno_variance
    return base * max(float(design_effect), 1.0)


def _masked_residual(
    model: FNO2d,
    obs: ObservationSet,
    theta: torch.Tensor,
    adapter: InverseAdapter = _SOURCE_ITR_SIN_ADAPTER,
) -> torch.Tensor:
    """Flat residual vector ``(m,)`` ``= masked(G_theta) - masked(obs)``."""
    pred = predict_fullfield(model, obs, theta, adapter)
    pred_obs = apply_sensor_mask(pred, obs.mask)
    target_obs = apply_sensor_mask(obs.targets, obs.mask)
    return (pred_obs - target_obs).reshape(-1)


def neg_log_likelihood(
    model: FNO2d,
    obs: ObservationSet,
    theta: torch.Tensor,
    sigma_eff2: float,
    adapter: InverseAdapter = _SOURCE_ITR_SIN_ADAPTER,
) -> torch.Tensor:
    """Gaussian negative log-likelihood (sum convention) at ``theta``.

    ``NLL = 0.5 * sum_resid^2 / sigma_eff2`` up to a theta-independent constant.
    This is the proper likelihood used by both the profile-likelihood and the
    MCMC target; it differs from the MAP ``_data_loss`` (a *mean*-of-squares) by
    the count ``m`` and the variance ``sigma_eff2``, so the two are not the same
    scale — UQ needs the sum/variance form to get the chi-square calibration
    right.
    """
    resid = _masked_residual(model, obs, theta, adapter)
    return 0.5 * torch.sum(resid ** 2) / float(sigma_eff2)


def add_measurement_noise(
    obs: ObservationSet, noise_std: float, seed: int = 0
) -> None:
    """Add iid ``N(0, noise_std^2)`` noise to ``obs.targets`` in place.

    Noise is added to the full target field before masking; since the mask is a
    deterministic pixel selection, this is exactly iid noise on the sensors. A
    non-positive ``noise_std`` is a no-op. The generator is seeded for
    reproducibility across a sweep (callers pass ``noise_seed + sid``).
    """
    if noise_std is None or noise_std <= 0:
        obs.noise_seed = int(seed)
        obs.noise_std = float(noise_std or 0.0)
        return
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    noise = torch.randn(obs.targets.shape, generator=gen, dtype=torch.float32)
    obs.targets = obs.targets + (noise * float(noise_std)).to(obs.targets)
    obs.noise_seed = int(seed)
    obs.noise_std = float(noise_std)


# --- Profile likelihood (primary, frequentist) -----------------------------

def _theta_profile(
    u: torch.Tensor, fixed_index: int, fixed_value: float
) -> torch.Tensor:
    """Source-ITR sinusoid wrapper for profile pinning tests."""
    return _SOURCE_ITR_SIN_ADAPTER.theta_profile(u, fixed_index, fixed_value)


def _polish_unrestricted(
    model: FNO2d,
    obs: ObservationSet,
    sigma_eff2: float,
    u_seed: torch.Tensor,
    *,
    adam_steps: int = 150,
    adam_lr: float = 0.05,
    lbfgs_steps: int = 30,
    adapter: InverseAdapter = _SOURCE_ITR_SIN_ADAPTER,
) -> tuple[float, torch.Tensor]:
    """Minimize the Gaussian NLL over all of ``u``, Adam then an L-BFGS polish.

    Used once per inversion to move the MAP estimate onto the likelihood's own
    optimum, which is the reference the Wilks threshold is measured against.
    """
    device = obs.spatial.device
    u = u_seed.clone().detach().to(device).requires_grad_(True)

    def nll():
        return neg_log_likelihood(
            model, obs, adapter.theta_from_unconstrained(u), sigma_eff2, adapter
        )

    adam = torch.optim.Adam([u], lr=adam_lr)
    for _ in range(adam_steps):
        adam.zero_grad()
        loss = nll()
        loss.backward()
        adam.step()

    lbfgs = torch.optim.LBFGS([u], max_iter=lbfgs_steps, line_search_fn="strong_wolfe")

    def closure():
        lbfgs.zero_grad()
        loss = nll()
        loss.backward()
        return loss

    lbfgs.step(closure)

    with torch.no_grad():
        theta = adapter.theta_from_unconstrained(u)
        final = float(neg_log_likelihood(model, obs, theta, sigma_eff2, adapter))
    return final, theta.detach().cpu()


def _profile_refit_scalar(
    model: FNO2d,
    obs: ObservationSet,
    sigma_eff2: float,
    fixed_index: int,
    fixed_value: float,
    *,
    adapter: InverseAdapter = _SOURCE_ITR_SIN_ADAPTER,
    xatol: float = PROFILE_NUISANCE_XATOL,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    """Re-optimize the one free scalar at a pinned constraint, in closed box form.

    Pinning one of two physical coordinates leaves exactly one unknown, and
    ``adapter.theta_profile`` reaches it only through ``sigmoid(u)[free]``. So
    the nuisance problem is a bounded 1-D minimization over ``s in (0, 1)``,
    which maps linearly onto the free physical coordinate and its dependent
    ceiling. A coarse sweep of the closed box picks the bracket and Brent's
    bounded method refines inside it, together a few tens of forward passes and
    no backward passes at all.

    The sweep is not optional. Brent alone is local and never evaluates the
    bounds, and this objective is bimodal on the tiny fixtures -- it converges to
    an interior basin while the optimum sits against ``s = 0``. An overstated
    profiled NLL pushes the Wilks crossings inward, so a missed nuisance optimum
    reports an interval that is too narrow.

    This is a genuine profile: the nuisance is re-optimized at every pin, never
    held at its joint MLE.
    """
    if adapter.theta_dim > 2:
        raise ValueError(
            "scalar profile nuisance needs a 1- or 2-parameter benchmark; "
            f"{adapter.benchmark} recovers {adapter.theta_dim}"
        )
    device = obs.spatial.device
    u = torch.zeros(max(adapter.theta_dim, 1), dtype=torch.float32, device=device)

    def evaluate(u_: torch.Tensor) -> tuple[float, torch.Tensor]:
        with torch.no_grad():
            theta = adapter.theta_profile(u_, fixed_index, fixed_value)
            return float(neg_log_likelihood(model, obs, theta, sigma_eff2, adapter)), theta

    if adapter.theta_dim == 1:
        value, theta = evaluate(u)
        return value, theta.detach().cpu(), u.detach().cpu()

    free = 1 - int(fixed_index)

    def objective(s: float) -> float:
        u[free] = float(np.log(s) - np.log1p(-s))
        return evaluate(u)[0]

    scan = np.linspace(
        PROFILE_NUISANCE_EPS, 1.0 - PROFILE_NUISANCE_EPS, PROFILE_NUISANCE_SCAN
    )
    scanned = [objective(float(s)) for s in scan]
    best = int(np.argmin(scanned))
    lo = float(scan[max(best - 1, 0)])
    hi = float(scan[min(best + 1, PROFILE_NUISANCE_SCAN - 1)])

    opt = minimize_scalar(
        objective, bounds=(lo, hi), method="bounded", options={"xatol": xatol}
    )
    s_best = float(opt.x) if float(opt.fun) < scanned[best] else float(scan[best])
    u[free] = float(np.log(s_best) - np.log1p(-s_best))
    value, theta = evaluate(u)
    return value, theta.detach().cpu(), u.detach().cpu()


@dataclass
class ProfileResult:
    param_index: int
    param_name: str
    level: float
    grid: np.ndarray            # (G,) pinned physical values actually evaluated
    nll: np.ndarray             # (G,) profiled NLL at each pin
    nll_min: float
    ci_low: float               # physical param CI (Wilks)
    ci_high: float
    n_evals: int = 0
    theta_mle: Optional[torch.Tensor] = None
    lower_closed: bool = False
    upper_closed: bool = False


def profile_likelihood(
    model: FNO2d,
    obs: ObservationSet,
    theta_hat: torch.Tensor,
    sigma_eff2: float,
    *,
    param_index: int = 1,
    level: float = 0.95,
    adapter: InverseAdapter = _SOURCE_ITR_SIN_ADAPTER,
) -> ProfileResult:
    """Bracket and bisect the Wilks crossings of one profiled coordinate.

    The reported statistic is the interval width, not a rendered profile curve,
    so the profile is only evaluated where it decides an endpoint: march outward
    from ``theta_hat[param_index]`` in fixed steps until the threshold is
    bracketed or a physical bound is reached, then bisect the bracket. Bisection
    keeps the *outside* end, so a finite root tolerance can only widen the
    interval, never shrink it below its nominal level.

    The nuisance coordinate is re-optimized at every pin. Holding it at the
    joint MLE would be an estimated (plug-in) likelihood, which under the
    R_base/A correlation reports intervals that are too narrow.
    """
    if not (0 <= param_index < adapter.theta_dim):
        raise ValueError("profile parameter index is out of range")
    if not np.isfinite(sigma_eff2) or sigma_eff2 <= 0 or not 0 < level < 1:
        raise ValueError(
            "profile requires a finite positive variance and 0 < level < 1"
        )
    name = adapter.param_names[param_index]
    lo_b, hi_b = adapter.profile_bounds(param_index)
    theta_mle = theta_hat.detach().cpu()
    nll_hat = float(neg_log_likelihood(
        model, obs, theta_mle.to(obs.spatial.device), sigma_eff2, adapter
    ).detach())
    threshold = float(chi2.ppf(level, 1) / 2.0)
    cache: dict[float, tuple[float, torch.Tensor]] = {}

    def profiled(pin: float) -> float:
        if pin not in cache:
            cache[pin] = _profile_refit_scalar(
                model, obs, sigma_eff2, param_index, pin, adapter=adapter
            )[:2]
        return cache[pin][0]

    center = float(np.clip(float(theta_mle[param_index]), lo_b, hi_b))
    # Re-optimizing the nuisance at the centre pin can only lower the NLL, and
    # the reference has to be the lowest value the profile itself can reach or
    # delta-NLL would go negative and the interval would come out too wide.
    nll_min = min(nll_hat, profiled(center))

    def outside(pin: float) -> bool:
        return profiled(pin) - nll_min > threshold

    def bisect(inside_x: float, outside_x: float) -> float:
        tol = PROFILE_ROOT_RTOL * (hi_b - lo_b)
        for _ in range(PROFILE_ROOT_MAX_STEPS):
            if abs(outside_x - inside_x) <= tol:
                break
            midpoint = 0.5 * (inside_x + outside_x)
            if outside(midpoint):
                outside_x = midpoint
            else:
                inside_x = midpoint
        return outside_x

    def endpoint(direction: int) -> tuple[float, bool]:
        bound = lo_b if direction < 0 else hi_b
        step = direction * PROFILE_STEP_FRACTION * (hi_b - lo_b)
        inside_x = center
        for _ in range(PROFILE_WALK_MAX_STEPS):
            if inside_x == bound:
                return bound, False
            nxt = float(np.clip(inside_x + step, lo_b, hi_b))
            if outside(nxt):
                return bisect(inside_x, nxt), True
            inside_x = nxt
        return bound, False

    ci_low, lower_closed = endpoint(-1)
    ci_high, upper_closed = endpoint(+1)
    grid = np.asarray(sorted(cache), dtype=np.float64)
    nll = np.asarray([cache[float(x)][0] for x in grid], dtype=np.float64)
    best_x = min(cache, key=lambda x: cache[x][0])
    # Hand a better fit found while bracketing back to the caller; the driver
    # re-profiles against it rather than reporting an interval whose reference
    # NLL was never the minimum.
    if cache[best_x][0] < nll_min - PROFILE_OPTIMUM_ATOL:
        theta_mle = cache[best_x][1]
    return ProfileResult(
        param_index=param_index, param_name=name, level=level,
        grid=grid, nll=nll, nll_min=nll_min,
        ci_low=float(ci_low), ci_high=float(ci_high),
        n_evals=len(cache), theta_mle=theta_mle,
        lower_closed=lower_closed, upper_closed=upper_closed,
    )


def profile_interval_summary(res: ProfileResult) -> dict:
    stem = f"profile_{res.param_name}"
    return {
        "profile_level": res.level,
        f"{stem}_ci_low": res.ci_low,
        f"{stem}_ci_high": res.ci_high,
        f"{stem}_ci_width": res.ci_high - res.ci_low,
        f"{stem}_lower_closed": res.lower_closed,
        f"{stem}_upper_closed": res.upper_closed,
        f"{stem}_bound_limited": not (res.lower_closed and res.upper_closed),
        f"{stem}_n_evals": res.n_evals,
    }


def profile_parameters(
    model, obs, theta_hat, sigma_eff2, *, adapter, indices=None,
    adam_steps: int = 150, lbfgs_steps: int = 30, **kwargs,
):
    """Polish the joint MLE, then profile every parameter against that one optimum.

    Profiling can walk into a lower NLL than the reference it started from. The
    interval it reported is then measured against a value that was never the
    minimum, so it is too narrow. When that happens the better point is adopted
    and every parameter is profiled again, which also keeps the parameters on a
    shared reference.
    """
    indices = list(range(adapter.theta_dim)) if indices is None else list(indices)
    theta = theta_hat.detach().cpu()

    def nll_at(candidate: torch.Tensor) -> float:
        return float(neg_log_likelihood(
            model, obs, candidate.to(obs.spatial.device), sigma_eff2, adapter
        ).detach())

    # The MAP fit minimizes a regularized mean-of-squares, so its optimum is not
    # exactly the likelihood's. One unrestricted refit puts theta_hat on the NLL
    # the Wilks threshold is measured against.
    baseline = nll_at(theta)
    value, polished = _polish_unrestricted(
        model, obs, sigma_eff2, adapter.unconstrained_from_theta(theta),
        adam_steps=adam_steps, lbfgs_steps=lbfgs_steps, adapter=adapter,
    )
    if value < baseline:
        theta, baseline = polished, value

    for _ in range(PROFILE_OPTIMUM_MAX_ROUNDS):
        profiles = [profile_likelihood(
            model, obs, theta, sigma_eff2, param_index=i, adapter=adapter, **kwargs
        ) for i in indices]
        found = [p.theta_mle for p in profiles if p.theta_mle is not None]
        best = min(found, key=nll_at, default=None)
        if best is None or nll_at(best) >= baseline - PROFILE_OPTIMUM_ATOL:
            return theta, profiles
        theta, baseline = best.detach().cpu(), nll_at(best)
    return theta, profiles


def joint_nll_grid(
    model: FNO2d,
    obs: ObservationSet,
    sigma_eff2: float,
    *,
    adapter: InverseAdapter,
    profiles: Sequence[ProfileResult],
    theta_hat: torch.Tensor,
    n_grid: int,
    level: float = 0.95,
) -> dict:
    """Evaluate the NLL on an ``n_grid x n_grid`` physical grid over both parameters.

    Unlike the per-parameter profiles, which re-fit the nuisance coordinate at
    every pin, this fixes *both* coordinates. The result is therefore the
    measured joint likelihood surface rather than a quadratic approximation of
    it, which is what a joint-region figure has to claim. The span is the union
    of the two profile grids clipped to the adapter bounds, so the joint surface
    covers at least the range the marginal profiles already show.

    Rows index ``param_names[0]``, columns index ``param_names[1]``. Points
    violating the dependent ceiling ``A <= R_PEAK_MAX - R_base`` are outside the
    sampled parameterization and stay ``nan``; the admissible set is triangular.
    """
    if adapter.theta_dim != 2:
        raise ValueError(
            f"joint NLL grid requires a 2-parameter benchmark; {adapter.benchmark} "
            f"recovers {adapter.theta_dim}"
        )
    if n_grid < 3:
        raise ValueError("joint NLL grid requires n_grid >= 3")
    if not np.isfinite(sigma_eff2) or sigma_eff2 <= 0 or not 0 < level < 1:
        raise ValueError("joint NLL grid requires a finite positive variance and 0 < level < 1")
    by_index = {int(p.param_index): p for p in profiles}
    if set(by_index) != {0, 1}:
        raise ValueError("joint NLL grid requires a profile for both parameters")

    axes = []
    for index in (0, 1):
        lo_b, hi_b = adapter.profile_bounds(index)
        grid = np.asarray(by_index[index].grid, dtype=np.float64)
        lo = max(float(lo_b), float(grid.min()))
        hi = min(float(hi_b), float(grid.max()))
        if not hi > lo:
            raise ValueError(
                f"joint NLL grid span for {adapter.param_names[index]} collapsed to a point"
            )
        axes.append(np.linspace(lo, hi, int(n_grid)))

    theta_mle = theta_hat.detach().cpu().reshape(-1)
    device = obs.spatial.device
    nll = np.full((int(n_grid), int(n_grid)), np.nan, dtype=np.float64)
    with torch.no_grad():
        for i, r_base in enumerate(axes[0]):
            for j, amp in enumerate(axes[1]):
                if amp > R_PEAK_MAX - r_base:
                    continue
                theta = torch.tensor([r_base, amp], dtype=theta_mle.dtype)
                nll[i, j] = float(neg_log_likelihood(
                    model, obs, theta.to(device), sigma_eff2, adapter
                ).detach())
        at_mle = float(neg_log_likelihood(
            model, obs, theta_mle.to(device), sigma_eff2, adapter
        ).detach())
    finite = nll[np.isfinite(nll)]
    if finite.size == 0:
        raise ValueError("joint NLL grid contains no admissible points")
    # Reference against the fit the marginal profiles used, so delta-ell is on
    # one scale across all three panels and can never come out negative.
    return {
        f"joint_nll_grid_{adapter.param_names[0]}": axes[0],
        f"joint_nll_grid_{adapter.param_names[1]}": axes[1],
        "joint_nll": nll,
        "joint_nll_min": np.float64(min(at_mle, float(finite.min()))),
        "joint_threshold": np.float64(chi2.ppf(level, 2) / 2.0),
    }


# ---------------------------------------------------------------------------
# Per-sim artifact dump: raw profile curves, MCMC samples, and the observation
# Jacobian only ever exist as locals in main()'s loop; the CSV keeps one summary
# row per sim. Figs 2 & 4 of the inverse-results section need the raw arrays, so
# (opt-in via --artifact-dir) we write one self-describing NPZ per sim.
# ---------------------------------------------------------------------------

# 4 added the optional joint_nll_* block (--joint-nll-grid). Readers that only
# need the version-3 keys should accept >= 3 and treat joint_nll_* as optional.
_ARTIFACT_SCHEMA_VERSION = 4


def _dataset_fingerprint(ds: SnapshotPairDataset) -> str:
    """Stable hash of corpus geometry, split IDs, metadata, and sampled states."""
    h = hashlib.sha256()
    h.update(np.asarray(ds.trajectories.shape, dtype=np.int64).tobytes())
    h.update(str(ds.trajectories.dtype).encode())
    for array in (ds.x_grid, ds.y_grid, ds.t_grid):
        h.update(np.asarray(array).tobytes())
    for split in ("train", "val", "test"):
        h.update(split.encode())
        h.update(np.asarray(ds._split_ids.get(split, []), dtype=np.int64).tobytes())
    for params in ds.sim_params:
        metadata = {
            key: value
            for key, value in dict(params).items()
            if key != "T0"
        }
        h.update(
            json.dumps(
                metadata,
                sort_keys=True,
                separators=(",", ":"),
                default=lambda value: (
                    np.asarray(value).tolist()
                    if isinstance(value, np.ndarray)
                    else float(value)
                    if isinstance(value, np.floating)
                    else int(value)
                    if isinstance(value, np.integer)
                    else str(value)
                ),
            ).encode()
        )
    trajectories = np.asarray(ds.trajectories)
    sample_ids = np.unique(
        np.linspace(0, trajectories.shape[0] - 1, min(7, trajectories.shape[0]))
        .round()
        .astype(int)
    )
    h.update(np.ascontiguousarray(trajectories[sample_ids]).tobytes())
    return h.hexdigest()[:16]


def _checkpoint_fingerprint(checkpoint_path: str) -> str:
    h = hashlib.sha256()
    with open(checkpoint_path, "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()[:16]


def _observation_fingerprint(
    *,
    sensor_layout: str,
    observation_times: np.ndarray,
    requested_y: Optional[np.ndarray],
    interface_x: float,
    sensor_x_halfwidth: Optional[float],
) -> str:
    payload = {
        "sensor_layout": str(sensor_layout),
        "observation_times": [
            format(float(value), ".17g") for value in observation_times
        ],
        "requested_y": (
            []
            if requested_y is None
            else [format(float(value), ".17g") for value in requested_y]
        ),
        "interface_x": format(float(interface_x), ".17g"),
        "sensor_x_halfwidth": (
            None
            if sensor_x_halfwidth is None
            else format(float(sensor_x_halfwidth), ".17g")
        ),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


def _split_name_for(ds: SnapshotPairDataset, sid: int) -> str:
    """Which split ``sid`` falls in (``test`` for held-out inversion targets)."""
    for name in ("test", "val", "train"):
        ids = ds._split_ids.get(name, [])
        if int(sid) in {int(i) for i in ids}:
            return name
    return "unknown"


_CALIBRATION_SCHEMA_VERSION = 2


def build_surrogate_calibration(
    model: FNO2d,
    ds: SnapshotPairDataset,
    adapter: InverseAdapter,
    *,
    time_indices: list[int],
    observation_times: np.ndarray,
    sensor_layout: str,
    sensor_y: Optional[np.ndarray],
    interface_x: float,
    sensor_x_halfwidth: Optional[float],
    sensor_n_y: int,
    mu_global: float,
    sigma_global: float,
    noise_std_norm: float,
    checkpoint_fingerprint: str,
    dataset_fingerprint: str,
    device: str,
) -> dict[str, object]:
    """Estimate one frozen sensor-scale FNO residual on the validation split."""
    if float(ds.noise_std) != 0.0:
        raise ValueError("surrogate calibration requires dataset source-noise disabled.")
    val_ids = np.asarray(ds._split_ids["val"], dtype=np.int64)
    if val_ids.size == 0:
        raise ValueError("validation split is empty; cannot calibrate surrogate error.")

    squared_residuals: list[np.ndarray] = []
    residual_vectors: list[np.ndarray] = []
    sensor_rms_norm: list[float] = []
    jump_rms_k: list[float] = []
    theta_values: list[np.ndarray] = []
    first_obs: Optional[ObservationSet] = None
    for sid in val_ids:
        obs = build_observation_set(
            ds, int(sid), time_indices, device=device, adapter=adapter,
            interface_x=interface_x,
            sensor_x_halfwidth=sensor_x_halfwidth,
            sensor_n_y=sensor_n_y,
            sensor_layout=sensor_layout,
            sensor_y=sensor_y,
            observation_times=observation_times,
        )
        if first_obs is None:
            first_obs = obs
        theta_values.append(obs.theta_true.detach().cpu().numpy().astype(np.float64))
        with torch.no_grad():
            prediction = predict_fullfield(
                model, obs, obs.theta_true, adapter
            )
        pred_obs = apply_sensor_mask(prediction, obs.mask)
        true_obs = apply_sensor_mask(obs.targets, obs.mask)
        residual = (
            pred_obs.detach().cpu().numpy()
            - true_obs.detach().cpu().numpy()
        ).astype(np.float64)
        residual_flat = residual.reshape(-1)
        squared_residuals.append(np.square(residual_flat))
        residual_vectors.append(residual_flat)
        sensor_rms_norm.append(float(np.sqrt(np.mean(np.square(residual)))))
        if sensor_layout == "interface_pair":
            n_y = (
                len(obs.requested_y)
                if obs.requested_y is not None
                else len(PAPER_SENSOR_Y)
            )
            pred_jump = pred_obs[:, n_y:, :] - pred_obs[:, :n_y, :]
            true_jump = true_obs[:, n_y:, :] - true_obs[:, :n_y, :]
            jump_rms_k.append(
                float(
                    torch.sqrt(torch.mean((pred_jump - true_jump) ** 2))
                    * float(sigma_global)
                )
            )

    all_squared = np.concatenate(squared_residuals)
    sigma_fno_norm = float(np.sqrt(np.mean(all_squared)))
    residual_sizes = {int(v.size) for v in residual_vectors}
    if len(residual_vectors) >= 2 and len(residual_sizes) == 1:
        residual_matrix = np.vstack(residual_vectors)
    else:
        residual_matrix = np.empty((0, 0), dtype=np.float64)
    design = _residual_design_effect(
        residual_matrix, float(noise_std_norm) ** 2
    )
    rms_norm = np.asarray(sensor_rms_norm, dtype=np.float64)
    rms_k = rms_norm * float(sigma_global)
    sigma_meas_k = float(noise_std_norm) * float(sigma_global)
    median_k = float(np.median(rms_k))
    p90_k = float(np.percentile(rms_k, 90.0))
    gate_pass = bool(
        noise_std_norm > 0.0
        and median_k <= sigma_meas_k
        and p90_k <= 2.0 * sigma_meas_k
    )
    assert first_obs is not None
    observation_fingerprint = _observation_fingerprint(
        sensor_layout=sensor_layout,
        observation_times=observation_times,
        requested_y=first_obs.requested_y,
        interface_x=interface_x,
        sensor_x_halfwidth=sensor_x_halfwidth,
    )
    return {
        "calibration_schema_version": np.int64(_CALIBRATION_SCHEMA_VERSION),
        "benchmark": np.str_(adapter.benchmark),
        "split": np.str_("val"),
        "sigma_fno_norm": np.float64(sigma_fno_norm),
        "sigma_fno_K": np.float64(sigma_fno_norm * float(sigma_global)),
        "residual_design_effect": np.float64(design["design_effect"]),
        "residual_design_effect_raw": np.float64(design["design_effect_raw"]),
        "residual_fno_var_trace": np.float64(design["fno_var_trace"]),
        "residual_fno_var_sum": np.float64(design["fno_var_sum"]),
        "residual_dim": np.int64(design["dim"]),
        "residual_n_eff": np.float64(design["n_eff"]),
        "sigma_global_K": np.float64(sigma_global),
        "noise_std_norm": np.float64(noise_std_norm),
        "sigma_meas_K": np.float64(sigma_meas_k),
        "checkpoint_fingerprint": np.str_(checkpoint_fingerprint),
        "dataset_fingerprint": np.str_(dataset_fingerprint),
        "observation_fingerprint": np.str_(observation_fingerprint),
        "observation_times": np.asarray(observation_times, dtype=np.float64),
        "sensor_layout": np.str_(sensor_layout),
        "requested_y": (
            np.empty(0, dtype=np.float64)
            if first_obs.requested_y is None
            else np.asarray(first_obs.requested_y, dtype=np.float64)
        ),
        "resolved_y": (
            np.empty(0, dtype=np.float64)
            if first_obs.resolved_y is None
            else np.asarray(first_obs.resolved_y, dtype=np.float64)
        ),
        "y_indices": (
            np.empty(0, dtype=np.int64)
            if first_obs.y_indices is None
            else np.asarray(first_obs.y_indices, dtype=np.int64)
        ),
        "resolved_x": (
            np.empty(0, dtype=np.float64)
            if first_obs.resolved_x is None
            else np.asarray(first_obs.resolved_x, dtype=np.float64)
        ),
        "sample_count": np.int64(all_squared.size),
        "simulation_count": np.int64(val_ids.size),
        "simulation_ids": val_ids,
        "param_names": np.asarray(adapter.param_names, dtype="U32"),
        "theta_true": np.asarray(theta_values, dtype=np.float64),
        "sensor_rms_norm": rms_norm,
        "sensor_rms_K": rms_k,
        "interface_jump_rms_K": np.asarray(jump_rms_k, dtype=np.float64),
        "sensor_rms_median_K": np.float64(median_k),
        "sensor_rms_p90_K": np.float64(p90_k),
        "local_gate_pass": np.bool_(gate_pass),
    }


def write_surrogate_calibration(path: str, calibration: dict[str, object]) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, **calibration)
    return path


def load_surrogate_calibration(
    path: str,
    *,
    benchmark: str,
    checkpoint_fingerprint: str,
    dataset_fingerprint: str,
    observation_fingerprint: str,
    noise_std_norm: float,
) -> dict[str, object]:
    """Load a frozen calibration artifact and reject protocol mismatches."""
    with np.load(path, allow_pickle=False) as stored:
        calibration = {key: stored[key] for key in stored.files}
    checks = {
        "benchmark": benchmark,
        "checkpoint_fingerprint": checkpoint_fingerprint,
        "dataset_fingerprint": dataset_fingerprint,
        "observation_fingerprint": observation_fingerprint,
        "split": "val",
    }
    for key, expected in checks.items():
        actual = str(calibration.get(key, ""))
        if actual != str(expected):
            raise ValueError(
                f"calibration {key} mismatch: artifact={actual!r}, "
                f"current={str(expected)!r}."
            )
    version = int(calibration.get("calibration_schema_version", -1))
    if version != _CALIBRATION_SCHEMA_VERSION:
        raise ValueError(
            f"calibration schema version {version} is unsupported; "
            f"expected {_CALIBRATION_SCHEMA_VERSION}."
        )
    stored_noise = float(calibration["noise_std_norm"])
    if not np.isclose(stored_noise, float(noise_std_norm), rtol=0.0, atol=1e-12):
        raise ValueError(
            f"calibration noise_std_norm={stored_noise:g} does not match "
            f"requested {float(noise_std_norm):g}."
        )
    return calibration


def _write_sim_artifact(
    artifact_dir: str,
    sid: int,
    adapter: InverseAdapter,
    *,
    obs: ObservationSet,
    result,
    report: Optional[dict],
    J: Optional[torch.Tensor],
    fv_res: Optional[FVRefineResult],
    profiles: Optional[list[ProfileResult]],
    joint: Optional[dict] = None,
    sigma_eff2: Optional[float],
    c_fno: Optional[float],
    noise_std: float,
    ci_level: float,
    dataset_path: str,
    dataset_fingerprint: str,
    split_seed: int,
    split_name: str,
    calibration_fingerprint: Optional[str] = None,
    sigma_fno_cal: Optional[float] = None,
) -> str:
    """Write one ``sim_{sid:05d}.npz`` with the raw inverse-problem arrays.

    Only the schema metadata is always present; every stage block is included
    only when its source object exists (the stage was run). The persisted
    Jacobian and SVD diagnostics are the *raw physical* quantities plus
    ``param_scales`` so the plot layer can rebuild the normalized-coordinate SVD
    (``J_scaled = J . diag(param_scales)``) itself rather than trusting a
    pre-baked coordinate choice.
    """
    os.makedirs(artifact_dir, exist_ok=True)
    theta_dim = adapter.theta_dim

    theta_hat = np.asarray(result.theta_hat.detach().cpu().numpy(), dtype=np.float64)
    if obs.theta_true is not None:
        theta_true = np.asarray(obs.theta_true.detach().cpu().numpy(), dtype=np.float64)
    else:
        theta_true = np.full(theta_dim, np.nan, dtype=np.float64)
    theta_bounds = np.asarray(
        [adapter.profile_bounds(i) for i in range(theta_dim)], dtype=np.float64
    )

    payload: dict = {
        "artifact_schema_version": np.int64(_ARTIFACT_SCHEMA_VERSION),
        "benchmark": np.str_(adapter.benchmark),
        "sim_id": np.int64(int(sid)),
        "param_names": np.asarray(adapter.param_names, dtype="U32"),
        "theta_hat": theta_hat,
        "theta_true": theta_true,
        "theta_bounds": theta_bounds,
        "param_scales": np.asarray(adapter.param_scales, dtype=np.float64),
        "noise_std": np.float64(noise_std),
        "ci_level": np.float64(ci_level),
        "profile_threshold": np.float64(
            chi2.ppf(ci_level, 1) / 2.0
        ),
        "dataset_path": np.str_(dataset_path),
        "dataset_fingerprint": np.str_(dataset_fingerprint),
        "split_seed": np.int64(int(split_seed)),
        "split_name": np.str_(split_name),
        "time_indices": np.asarray(obs.time_indices, dtype=np.int64),
        "observation_times": (
            np.empty(0, dtype=np.float64)
            if obs.observation_times is None
            else np.asarray(obs.observation_times, dtype=np.float64)
        ),
        "sensor_layout": np.str_(obs.sensor_layout),
        "requested_y": (
            np.empty(0, dtype=np.float64)
            if obs.requested_y is None
            else np.asarray(obs.requested_y, dtype=np.float64)
        ),
        "resolved_y": (
            np.empty(0, dtype=np.float64)
            if obs.resolved_y is None
            else np.asarray(obs.resolved_y, dtype=np.float64)
        ),
        "y_indices": (
            np.empty(0, dtype=np.int64)
            if obs.y_indices is None
            else np.asarray(obs.y_indices, dtype=np.int64)
        ),
        "resolved_x": (
            np.empty(0, dtype=np.float64)
            if obs.resolved_x is None
            else np.asarray(obs.resolved_x, dtype=np.float64)
        ),
        "noise_seed": np.int64(-1 if obs.noise_seed is None else obs.noise_seed),
    }
    if calibration_fingerprint is not None:
        payload["calibration_fingerprint"] = np.str_(calibration_fingerprint)
    if sigma_fno_cal is not None:
        payload["sigma_fno_cal"] = np.float64(sigma_fno_cal)

    if report is not None:
        payload["singular_values"] = np.asarray(report["singular_values"], dtype=np.float64)
        payload["right_vectors"] = np.asarray(report["right_vectors"], dtype=np.float64)
        payload["param_sensitivity"] = np.asarray(report["param_sensitivity"], dtype=np.float64)
        payload["least_identified_dir"] = np.asarray(report["least_identified_dir"], dtype=np.float64)
        payload["cond_number"] = np.float64(report["cond_number"])
        align = report.get("ramp_sigma_alignment", float("nan"))
        if np.isfinite(align):
            payload["ramp_sigma_alignment"] = np.float64(align)

    if J is not None:
        payload["observation_jacobian"] = np.asarray(J.detach().cpu().numpy(), dtype=np.float64)

    if fv_res is not None:
        payload["fno_resid"] = np.float64(fv_res.fno_resid)
        payload["fv_resid"] = np.float64(fv_res.fv_resid)
        payload["fno_vs_fv_resid"] = np.float64(fv_res.fno_vs_fv_resid)

    for prof in profiles or []:
        stem = f"profile_{prof.param_name}"
        payload[f"{stem}_grid"] = np.asarray(prof.grid, dtype=np.float64)
        payload[f"{stem}_nll"] = np.asarray(prof.nll, dtype=np.float64)
        payload[f"{stem}_nll_min"] = np.float64(prof.nll_min)
        payload.update(profile_interval_summary(prof))

    payload.update(joint or {})

    if sigma_eff2 is not None:
        payload["sigma_eff2"] = np.float64(sigma_eff2)
    if c_fno is not None:
        payload["c_fno"] = np.float64(c_fno)

    path = os.path.join(artifact_dir, f"sim_{int(sid):05d}.npz")
    np.savez_compressed(path, **payload)
    return path


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="Path to fno2d_best.pt")
    ap.add_argument("--data-dir", default=None,
                    help="Existing disjoint held-out dataset dir. If omitted, a "
                         "fresh disjoint dataset is generated into --generate-dir")
    ap.add_argument("--generate-dir", default=None,
                    help="Where to write the generated disjoint dataset when "
                         "--data-dir is omitted (default: <artifact-dir>/inversion_data)")
    ap.add_argument("--n-invert", type=int, default=None,
                    help="Size of the inversion slice (first N sims). When "
                         "generating, the dataset holds n-invert + n-calibration sims")
    ap.add_argument("--n-calibration", type=int, default=CALIBRATION_FLOOR,
                    help="Size of the disjoint calibration slice (fixed floor, "
                         f"default {CALIBRATION_FLOOR})")
    ap.add_argument("--dataset-seed", type=int, default=INVERSION_DATASET_SEED,
                    help="rng_seed for generated inversion data (non-zero keeps "
                         "it disjoint from the seed-0 training corpus)")
    ap.add_argument("--sim-ids", type=int, nargs="*", default=None,
                    help="Sim ids to invert (default: the whole inversion slice)")
    time_group = ap.add_mutually_exclusive_group()
    time_group.add_argument("--time-indices", type=int, nargs="*", default=None,
                            help="Legacy observation snapshot indices")
    time_group.add_argument(
        "--observation-times", type=float, nargs="+", default=None,
        help="Exact physical observation times; no interpolation or nearest substitution",
    )
    ap.add_argument("--n-starts", type=int, default=8)
    ap.add_argument("--optimizer", choices=("adam_lbfgs", "nelder_mead"),
                    default="adam_lbfgs", help="Optimizer for the FNO parameter fit")
    ap.add_argument("--nm-maxiter", type=int, default=60)
    ap.add_argument("--nm-xatol", type=float, default=1e-4)
    ap.add_argument("--nm-fatol", type=float, default=1e-14)
    ap.add_argument("--adam-steps", type=int, default=400)
    ap.add_argument("--adam-lr", type=float, default=0.05)
    ap.add_argument("--lbfgs-steps", type=int, default=50)
    ap.add_argument("--reg-weight", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-csv", default=None)
    # Stage 2 sensor operator. Omit --sensor-x-halfwidth for full-field (Stage 1).
    ap.add_argument("--sensor-x-halfwidth", type=float, default=None,
                    help="Half-width used only by the interface_band layout")
    ap.add_argument("--sensor-n-y", type=int, default=8,
                    help="Number of y sensor rows in the interface band")
    ap.add_argument("--interface-x", type=float, default=0.5,
                    help="Physical interface x location for sensor placement")
    ap.add_argument(
        "--sensor-layout",
        choices=["full_field", "interface_band", "interface_pair"],
        default="interface_pair",
        help="Physical observation layout (default: paired interface-adjacent nodes)",
    )
    ap.add_argument(
        "--sensor-y", type=float, nargs="+", default=None,
        help="Requested physical y sensor coordinates (default: linspace(0.1,0.9,8) for interface_pair)",
    )
    ap.add_argument("--self-consistency", action="store_true",
                    help="Invert against the FNO's own output at theta_true "
                         "(isolates operator wiring from surrogate accuracy)")
    ap.add_argument("--start-sampling", choices=["lhs", "normal"], default="lhs",
                    help="Multistart sampling over the theta-box (default: lhs)")
    # Statistic 2 knob. The FV verification itself is unconditional.
    ap.add_argument(
        "--fv-grid-size", type=int, default=None,
        help="Optional square FV grid size for verification; sensors are re-resolved physically",
    )
    # Measurement noise (feeds statistic 2's noise floor and statistic 3's likelihood).
    ap.add_argument("--noise-std", type=float, default=0.0,
                    help="Std of iid Gaussian sensor noise (normalized temp units); "
                         "0 disables noise")
    ap.add_argument("--noise-seed", type=int, default=0,
                    help="Base seed for measurement noise (per-sim seed = noise-seed + sim_id)")
    calibration_group = ap.add_mutually_exclusive_group()
    calibration_group.add_argument(
        "--calibration-out", default=None,
        help="Estimate validation-only surrogate error and write a frozen NPZ artifact, then exit",
    )
    calibration_group.add_argument(
        "--calibration-artifact", default=None,
        help="Frozen validation calibration NPZ required by likelihood-based test inversion",
    )
    ap.add_argument(
        "--allow-surrogate-limited", action="store_true",
        help="Allow diagnostic inversion when the frozen local sensor gate failed",
    )
    # Statistic 3 knobs. The profile itself runs whenever a frozen calibration
    # artifact is supplied.
    ap.add_argument("--uq-level", type=float, default=0.95,
                    help="Confidence level for the profile-likelihood interval")
    ap.add_argument("--profile-index", type=int, default=None,
                    help="Profile only this parameter index (default: every recovered parameter)")
    ap.add_argument("--joint-nll-grid", type=int, default=0,
                    help="If > 0, also evaluate the NLL on an NxN grid over both "
                         "physical parameters and store it in the artifact (a "
                         "measured joint region, not a quadratic approximation). "
                         "Two-parameter benchmarks only; 25 is a reasonable N")
    ap.add_argument("--artifact-dir", default=None,
                    help="If set, write one self-describing sim_<id>.npz per sim "
                         "(raw profile curves + split provenance) for the "
                         "inverse-results figures")
    args = ap.parse_args(argv)

    if args.calibration_artifact is not None and args.reg_weight != 0.0:
        raise ValueError("Profile-likelihood intervals require --reg-weight=0 (an unregularized fit)")
    loaded = load_checkpoint(args.checkpoint, device=args.device)
    adapter = InverseAdapter.from_config(loaded.config)
    adapter.validate_model(loaded)
    profile_indices = None if args.profile_index is None else [args.profile_index]
    if args.joint_nll_grid:
        if args.joint_nll_grid < 3:
            raise ValueError("--joint-nll-grid must be 0 (off) or at least 3")
        if adapter.theta_dim != 2:
            raise ValueError(
                f"--joint-nll-grid is only defined for 2-parameter benchmarks; "
                f"{adapter.benchmark} recovers {adapter.theta_dim}"
            )
        if profile_indices is not None:
            raise ValueError(
                "--joint-nll-grid needs both parameter profiles; drop --profile-index"
            )
        if args.calibration_artifact is None:
            raise ValueError(
                "--joint-nll-grid needs the calibrated variance from "
                "--calibration-artifact"
            )
        if not args.artifact_dir:
            raise ValueError("--joint-nll-grid has nowhere to write without --artifact-dir")

    if args.data_dir is not None:
        data_dir = args.data_dir
    else:
        if args.n_invert is None:
            raise ValueError(
                "--n-invert is required when generating a dataset (no --data-dir)."
            )
        data_cfg = loaded.config.get("data", {})
        generate_dir = args.generate_dir or os.path.join(
            args.artifact_dir or ".", "inversion_data"
        )
        partition = prepare_inversion_dataset(
            benchmark=adapter.benchmark,
            n_invert=args.n_invert,
            n_calibration=args.n_calibration,
            out_dir=generate_dir,
            nx=int(data_cfg.get("nx", 100)),
            ny=int(data_cfg.get("ny", 100)),
            save_stride=int(data_cfg.get("save_stride", 2)),
            dataset_seed=args.dataset_seed,
        )
        data_dir = partition["data_dir"]
        print(
            f"Generated disjoint inversion dataset at {data_dir} "
            f"({args.n_invert} inversion + {args.n_calibration} calibration "
            f"sims, seed={args.dataset_seed})."
        )

    ds = build_dataset_from_dir(
        data_dir, loaded.config,
        mu_global=loaded.mu_global, sigma_global=loaded.sigma_global,
        n_invert=args.n_invert, n_calibration=args.n_calibration,
    )
    adapter.validate_dataset(ds)

    if args.observation_times is not None:
        requested_times = np.asarray(args.observation_times, dtype=np.float64)
        time_indices, observation_times = resolve_observation_times(
            ds.t_grid, requested_times
        )
    elif args.time_indices is not None:
        time_indices = list(args.time_indices)
        if any(index <= 0 or index >= ds.Nt for index in time_indices):
            raise ValueError(
                f"time indices must lie in [1, {ds.Nt - 1}], got {time_indices}."
            )
        observation_times = np.asarray(ds.t_grid, dtype=np.float64)[time_indices]
    else:
        time_indices, observation_times = resolve_observation_times(
            ds.t_grid, PAPER_OBSERVATION_TIMES
        )

    sensor_y = (
        np.asarray(args.sensor_y, dtype=np.float64)
        if args.sensor_y is not None
        else np.asarray(PAPER_SENSOR_Y, dtype=np.float64)
        if args.sensor_layout == "interface_pair"
        else np.linspace(
            float(ds.y_grid[0]), float(ds.y_grid[-1]), int(args.sensor_n_y)
        )
        if args.sensor_layout == "interface_band"
        else None
    )
    if (
        args.fv_grid_size is not None
        and args.sensor_layout == "full_field"
        and int(args.fv_grid_size) != int(ds.Nx)
    ):
        raise ValueError(
            "cross-grid FV checks require coordinate-defined sparse sensors; "
            "full_field observations have grid-dependent cardinality."
        )
    checkpoint_fingerprint = _checkpoint_fingerprint(args.checkpoint)
    dataset_fingerprint = _dataset_fingerprint(ds)
    observation_fingerprint = _observation_fingerprint(
        sensor_layout=args.sensor_layout,
        observation_times=observation_times,
        requested_y=sensor_y,
        interface_x=args.interface_x,
        sensor_x_halfwidth=args.sensor_x_halfwidth,
    )

    if args.calibration_out:
        calibration = build_surrogate_calibration(
            loaded.model, ds, adapter,
            time_indices=time_indices,
            observation_times=observation_times,
            sensor_layout=args.sensor_layout,
            sensor_y=sensor_y,
            interface_x=args.interface_x,
            sensor_x_halfwidth=args.sensor_x_halfwidth,
            sensor_n_y=args.sensor_n_y,
            mu_global=loaded.mu_global,
            sigma_global=loaded.sigma_global,
            noise_std_norm=args.noise_std,
            checkpoint_fingerprint=checkpoint_fingerprint,
            dataset_fingerprint=dataset_fingerprint,
            device=args.device,
        )
        write_surrogate_calibration(args.calibration_out, calibration)
        print(
            f"Wrote validation calibration to {args.calibration_out}: "
            f"sigma_FNO={float(calibration['sigma_fno_norm']):.6g} normalized "
            f"({float(calibration['sigma_fno_K']):.6g} K), "
            f"median={float(calibration['sensor_rms_median_K']):.6g} K, "
            f"p90={float(calibration['sensor_rms_p90_K']):.6g} K, "
            f"gate_pass={bool(calibration['local_gate_pass'])}"
        )
        return 0

    if args.sim_ids is not None:
        sim_ids = list(args.sim_ids)
    else:
        sim_ids = list(ds._split_ids["test"])
    cfg = InversionConfig(
        n_starts=args.n_starts, adam_steps=args.adam_steps, adam_lr=args.adam_lr,
        lbfgs_steps=args.lbfgs_steps, reg_weight=args.reg_weight, seed=args.seed,
        device=args.device, start_sampling=args.start_sampling,
        optimizer=args.optimizer, nm_maxiter=args.nm_maxiter,
        nm_xatol=args.nm_xatol, nm_fatol=args.nm_fatol,
    )

    # Statistic 2 is unconditional, so the FV operator is always rebuilt.
    fv_base_kwargs = adapter.fv_base_kwargs(ds, grid_size=args.fv_grid_size)

    # Statistic 3 needs a frozen validation scale for its likelihood. Without
    # one it is skipped for every sim rather than run against an invented
    # variance; statistics 1 and 2 are unaffected.
    do_profile = args.calibration_artifact is not None
    if not do_profile:
        print(
            "No --calibration-artifact: reporting recovery error and FV "
            "verification only. The profile-likelihood interval needs a frozen "
            "validation calibration (build one with --calibration-out)."
        )
    calibration = None
    sigma_fno_cal = None
    design_effect_cal = 1.0
    calibration_fingerprint = None
    if do_profile:
        calibration = load_surrogate_calibration(
            args.calibration_artifact,
            benchmark=adapter.benchmark,
            checkpoint_fingerprint=checkpoint_fingerprint,
            dataset_fingerprint=dataset_fingerprint,
            observation_fingerprint=observation_fingerprint,
            noise_std_norm=args.noise_std,
        )
        if (
            not bool(calibration["local_gate_pass"])
            and not args.allow_surrogate_limited
        ):
            raise ValueError(
                "frozen validation calibration failed the median/p90 local "
                "sensor gate; pass --allow-surrogate-limited only for a labeled "
                "diagnostic run."
            )
        sigma_fno_cal = float(calibration["sigma_fno_norm"])
        design_effect_cal = float(
            calibration.get("residual_design_effect", 1.0)
        )
        calibration_fingerprint = _checkpoint_fingerprint(
            args.calibration_artifact
        )

    rows = []
    for sid in sim_ids:
        # None-init the per-sim stage locals so artifact writing never hits an
        # unbound local when a stage was skipped for this sim. `report` and `J`
        # stay None for good now: they feed retired diagnostics, and
        # _write_sim_artifact omits their blocks when they are absent.
        report = J = None
        fv_res = sigma_eff2 = joint = None
        profiles = []
        obs = build_observation_set(
            ds, int(sid), time_indices, device=args.device,
            adapter=adapter,
            interface_x=args.interface_x,
            sensor_x_halfwidth=args.sensor_x_halfwidth,
            sensor_n_y=args.sensor_n_y,
            sensor_layout=args.sensor_layout,
            sensor_y=sensor_y,
            observation_times=observation_times,
        )
        if args.self_consistency:
            with torch.no_grad():
                obs.targets = predict_fullfield(
                    loaded.model, obs, obs.theta_true, adapter
                ).detach()
        add_measurement_noise(
            obs, args.noise_std, seed=args.noise_seed + int(sid)
        )
        result = invert_sim(loaded.model, obs, cfg, adapter)
        if do_profile:
            sigma_eff2 = effective_sigma2(
                args.noise_std, sigma_fno_cal=sigma_fno_cal,
                design_effect=design_effect_cal,
            )
            if sigma_eff2 <= 0:
                raise ValueError("Profile likelihood requires positive calibrated variance")
            result.theta_hat, profiles = profile_parameters(
                loaded.model, obs, result.theta_hat, sigma_eff2, adapter=adapter,
                indices=profile_indices, level=args.uq_level,
            )
            if args.joint_nll_grid:
                joint = joint_nll_grid(
                    loaded.model, obs, sigma_eff2, adapter=adapter,
                    profiles=profiles, theta_hat=result.theta_hat,
                    n_grid=args.joint_nll_grid, level=args.uq_level,
                )
            residual = _masked_residual(
                loaded.model, obs, result.theta_hat.to(args.device), adapter
            )
            result.loss = float((0.5 * torch.mean(residual ** 2)).detach())
        theta_hat = result.theta_hat.to(args.device)
        summary = summarize(result, obs.y_grid, adapter)
        for prof in profiles:
            summary.update(profile_interval_summary(prof))
        summary["benchmark"] = adapter.benchmark
        summary["optimizer"] = cfg.optimizer
        summary["sim_id"] = int(sid)
        summary["time_indices"] = ";".join(str(t) for t in time_indices)
        summary["observation_times"] = ";".join(
            format(float(t), ".8g") for t in observation_times
        )
        summary["sensor_layout"] = obs.sensor_layout
        summary["requested_y"] = (
            ""
            if obs.requested_y is None
            else ";".join(format(float(v), ".8g") for v in obs.requested_y)
        )
        summary["resolved_y"] = (
            ""
            if obs.resolved_y is None
            else ";".join(format(float(v), ".8g") for v in obs.resolved_y)
        )
        summary["resolved_x"] = (
            ""
            if obs.resolved_x is None
            else ";".join(format(float(v), ".8g") for v in obs.resolved_x)
        )
        summary["noise_std_norm"] = float(args.noise_std)
        summary["noise_std_K"] = float(args.noise_std) * loaded.sigma_global
        summary["noise_seed"] = int(args.noise_seed + int(sid))
        n_sensors = (
            int(obs.spatial.shape[1] * obs.spatial.shape[2])
            if obs.mask is None
            else int(obs.mask.sum())
        )
        summary["n_sensors"] = n_sensors
        summary["self_consistency"] = bool(args.self_consistency)

        # --- Statistic 2: FV-verified sensor residual at theta_hat. ---
        fv_res = fv_refine_report(
            loaded.model, ds, fv_base_kwargs, obs, theta_hat,
            mu_global=loaded.mu_global, sigma_global=loaded.sigma_global,
            polish=False,
            adapter=adapter,
        )
        summary.update(
            fv_refine_summary(
                fv_res,
                obs,
                torch.as_tensor(
                    fv_base_kwargs["y_grid"], dtype=obs.y_grid.dtype
                ),
                adapter,
            )
        )
        summary.update(
            fv_verification_summary(
                fv_res,
                sigma_global=loaded.sigma_global,
                noise_std_norm=args.noise_std,
            )
        )

        if args.artifact_dir:
            _write_sim_artifact(
                args.artifact_dir, int(sid), adapter,
                obs=obs, result=result, report=report, J=J,
                fv_res=fv_res, profiles=profiles, joint=joint,
                sigma_eff2=sigma_eff2, c_fno=None,
                noise_std=args.noise_std, ci_level=args.uq_level,
                dataset_path=data_dir,
                dataset_fingerprint=dataset_fingerprint,
                split_seed=args.dataset_seed,
                split_name=_split_name_for(ds, int(sid)),
                calibration_fingerprint=calibration_fingerprint,
                sigma_fno_cal=sigma_fno_cal,
            )
        rows.append(summary)
        print(
            f"[sim {sid}] n_sensors={n_sensors} loss={summary['loss']:.3e} "
            f"FV residual={summary['fv_resid_rms_K']:.6g} K "
            f"({summary['fv_resid_over_noise']:.3g}x noise std)"
        )
        for i, name in enumerate(adapter.param_names):
            hat = float(result.theta_hat[i])
            true = float(result.theta_true[i]) if result.theta_true is not None else float("nan")
            relative = summary.get(f"{name}_rel_error_pct", float("nan"))
            relative_text = f"{relative:.6g}%" if np.isfinite(relative) else "unavailable (near-zero truth)"
            line = (f"  {name}: true={true:.6g}, estimate={hat:.6g}, "
                    f"absolute error={abs(hat - true):.6g}, relative error={relative_text}")
            stem = f"profile_{name}"
            if f"{stem}_ci_low" in summary:
                line += (f"; {100 * args.uq_level:g}% CI=[{summary[f'{stem}_ci_low']:.6g}, "
                         f"{summary[f'{stem}_ci_high']:.6g}], width={summary[f'{stem}_ci_width']:.6g}")
                if summary[f"{stem}_bound_limited"]:
                    line += " (bound-limited)"
            print(line)

    if args.out_csv and rows:
        fieldnames = sorted({k for r in rows for k in r})
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} rows to {args.out_csv}")

    return 0


# ===========================================================================
# RETIRED DIAGNOSTICS - not called by main().
#
# These were part of the earlier "compute everything" inverse report. They are
# kept verbatim, importable, and unit-tested (tests/test_invert.py exercises
# each one directly), but nothing on the live path reaches them: the CLI no
# longer has flags that trigger them and main() never calls them.
#
# Retired here, with the reason each one left the reported set:
#   * observation_jacobian / svd_identifiability / sensitivity_report /
#     sensitivity_summary - the Jacobian-SVD identifiability block. Degenerate
#     for scalar-theta benchmarks (forcing): J is (m, 1), so cond_number == 1,
#     least_dir == 1, and sv_0 == sens_R_c identically. Four columns carrying
#     one number.
#   * laplace_spectrum / laplace_summary - the same Jacobian SVD rescaled by
#     1/sigma_eff2 (eig = S**2 / sigma_eff2). A third copy of one matrix, and
#     never used for an interval even when it was live.
#   * fv_polish - changes the estimator rather than measuring it: either
#     theta_hat is the FNO MAP (and the polish is a second, unbenchmarked
#     method) or it is the polished value (and the method is FNO + Nelder-Mead
#     over the full FV solver, which undercuts the surrogate-speed claim).
#     Reached only via fv_refine_report(polish=True), which main() never sets.
#   * _solve_scalar_fv_masked / equivalent_scalar_fv_summary - the
#     "is R_c(y) distinguishable from a conductance-matched scalar" question.
#     A separate study with its own protocol, and forcing_itr_sin-only.
#
# The statistics-only members of the old block were removed outright rather than
# kept here: the RW-Metropolis MCMC (MCMCResult / _autocovariance /
# geyer_ess_iat / run_mcmc / mcmc_summary), a second interval on the same scalar
# the profile already covers and mistuned when live; the profile_summary /
# mcmc_summary coverage flatteners, whose per-case coverage boolean is not a
# frequentist coverage statement at n=8; and theta_logabsdet_du, the uniform-
# theta prior Jacobian whose only consumer was run_mcmc. The live profile path
# uses profile_interval_summary(), which reports the width without coverage.
# ===========================================================================


# ---------------------------------------------------------------------------
# Sensitivity / identifiability diagnostics (Stage 3): J = dG_theta/dtheta at a
# point, and the SVD of J that exposes ill-conditioned directions (notably the
# no Laplace/Hessian intervals (those are Stage 5).
# ---------------------------------------------------------------------------

def observation_jacobian(
    model: FNO2d,
    obs: ObservationSet,
    theta: torch.Tensor,
    adapter: InverseAdapter = _SOURCE_ITR_SIN_ADAPTER,
) -> torch.Tensor:
    """Jacobian ``J = d(masked G_theta)/d theta`` at ``theta``: ``(m, 4)``.

    ``m`` is the number of observed scalars (sensor pixels times observation
    time points times channels). The model is frozen; only the two sinusoidal
    scalars are differentiated, through both injection points.
    """
    theta0 = theta.detach().to(obs.spatial.device)

    def obs_vec(th: torch.Tensor) -> torch.Tensor:
        pred = predict_fullfield(model, obs, th, adapter)
        return apply_sensor_mask(pred, obs.mask).reshape(-1)

    try:
        J = torch.autograd.functional.jacobian(obs_vec, theta0, vectorize=True)
    except RuntimeError:
        # vmap may not support every op (e.g. some FFT paths); fall back to the
        # slower row-by-row reverse-mode Jacobian, which is always correct.
        J = torch.autograd.functional.jacobian(obs_vec, theta0, vectorize=False)
    return J.detach()


def svd_identifiability(J: np.ndarray) -> dict:
    """Identifiability summary from the observation Jacobian ``J`` (m, n_parameters).

    Pure linear algebra (unit-tested independently of the FNO):
      * ``singular_values`` — descending; small values flag directions the data
        cannot constrain.
      * ``right_vectors`` — (n_parameters, n_parameters), columns are orthonormal directions in
        theta-space ``(R_base, A)``.
      * ``param_sensitivity`` — per-parameter column norms ``||dG/dtheta_k||``.
      * ``least_identified_dir`` — right singular vector of the *smallest*
        singular value (the worst-constrained combination).
      * ``cond_number`` — s_max / s_min.
    """
    Jn = np.asarray(J, dtype=np.float64)
    param_sensitivity = np.linalg.norm(Jn, axis=0)
    _, S, Vt = np.linalg.svd(Jn, full_matrices=False)
    V = Vt.T
    least = V[:, -1]
    least = least / (np.linalg.norm(least) + 1e-300)
    cond_number = float(S[0] / S[-1]) if S[-1] > 0 else float("inf")
    return {
        "singular_values": S,
        "right_vectors": V,
        "param_sensitivity": param_sensitivity,
        "least_identified_dir": least,
        "cond_number": cond_number,
    }


def sensitivity_report(
    model: FNO2d,
    obs: ObservationSet,
    theta: torch.Tensor,
    adapter: InverseAdapter = _SOURCE_ITR_SIN_ADAPTER,
) -> dict:
    """``svd_identifiability`` of the observation Jacobian at ``theta``."""
    J = observation_jacobian(model, obs, theta, adapter)
    return svd_identifiability(J.cpu().numpy())


def sensitivity_summary(
    report: dict, adapter: InverseAdapter = _SOURCE_ITR_SIN_ADAPTER
) -> dict:
    """Flatten a :func:`sensitivity_report` into scalar CSV columns."""
    return adapter.sensitivity_summary(report)


def _solve_scalar_fv_masked(
    ds: SnapshotPairDataset,
    base_kwargs: dict,
    obs: ObservationSet,
    resistance: float,
    *,
    mu_global: float,
    sigma_global: float,
    adapter: InverseAdapter,
) -> torch.Tensor:
    params = dict(ds.sim_params[int(obs.sid)])
    solver = adapter.build_scalar_fv_solver(
        ds, params, float(resistance), base_kwargs
    )
    T0 = adapter.fv_initial_condition(ds, params, solver)
    t_solver, x_grid, y_grid, T_hist = solver.solve(
        T0=T0, store_trajectory=True
    )
    indices, _ = resolve_observation_times(
        np.asarray(t_solver, dtype=np.float64),
        np.asarray(obs.observation_times, dtype=np.float64),
    )
    field = torch.from_numpy(
        (
            (np.asarray(T_hist)[indices] - mu_global)
            / (sigma_global + T_EPS)
        ).astype(np.float32)
    ).unsqueeze(-1)
    return apply_sensor_mask(
        field, sensor_mask_for_grid(obs, x_grid, y_grid)
    )


def equivalent_scalar_fv_summary(
    ds: SnapshotPairDataset,
    base_kwargs: dict,
    obs: ObservationSet,
    theta: torch.Tensor,
    *,
    mu_global: float,
    sigma_global: float,
    adapter: InverseAdapter,
) -> dict:
    """Compare a spatial profile with base- and conductance-matched scalar FV cases."""
    if not adapter.supports_equivalent_scalar:
        raise ValueError(
            f"{type(adapter).__name__} does not support equivalent-scalar FV checks."
        )
    profile_obs = fv_predict_masked(
        ds, base_kwargs, obs.sid, theta, obs.time_indices,
        mu_global=mu_global, sigma_global=sigma_global,
        mask=None if obs.mask is None else obs.mask.cpu(),
        adapter=adapter, obs=obs,
    )
    R_base, R_eq = adapter.equivalent_scalar_values(
        theta,
        torch.as_tensor(base_kwargs["y_grid"], dtype=theta.dtype),
    )
    base_obs = _solve_scalar_fv_masked(
        ds, base_kwargs, obs, R_base,
        mu_global=mu_global, sigma_global=sigma_global, adapter=adapter,
    )
    req_obs = _solve_scalar_fv_masked(
        ds, base_kwargs, obs, R_eq,
        mu_global=mu_global, sigma_global=sigma_global, adapter=adapter,
    )
    return {
        "R_eq": R_eq,
        "fv_base_match_vs_profile_rms": float(
            torch.sqrt(torch.mean((base_obs - profile_obs) ** 2))
        ),
        "fv_req_match_vs_profile_rms": float(
            torch.sqrt(torch.mean((req_obs - profile_obs) ** 2))
        ),
    }


def fv_polish(
    ds: SnapshotPairDataset,
    base_kwargs: dict,
    obs: ObservationSet,
    theta_hat: torch.Tensor,
    *,
    mu_global: float,
    sigma_global: float,
    maxiter: int = 60,
    adapter: InverseAdapter = _SOURCE_ITR_SIN_ADAPTER,
) -> tuple[torch.Tensor, float, int]:
    """Derivative-free FV polish of ``theta_hat`` against the FV sensor residual.

    Optimizes in the unconstrained ``u``-space (so every FV query stays inside
    the physical box and honors the R_base-dependent amplitude ceiling) with
    Nelder-Mead, starting from the FNO MAP. Returns ``(theta_polished,
    final_half_mse, n_func_evals)``.
    """
    from scipy.optimize import minimize

    mask_cpu = None if obs.mask is None else obs.mask.cpu()
    target_obs = apply_sensor_mask(obs.targets.detach().cpu(), mask_cpu)
    u0 = adapter.unconstrained_from_theta(theta_hat.detach().cpu()).numpy().astype(np.float64)

    def objective(u_np: np.ndarray) -> float:
        u_t = torch.from_numpy(u_np.astype(np.float32))
        theta = adapter.theta_from_unconstrained(u_t)
        pred_obs = fv_predict_masked(
            ds, base_kwargs, obs.sid, theta, obs.time_indices,
            mu_global=mu_global, sigma_global=sigma_global, mask=mask_cpu,
            adapter=adapter, obs=obs,
        )
        return _masked_half_mse(pred_obs, target_obs)

    res = minimize(
        objective, u0, method="Nelder-Mead",
        options={"maxiter": int(maxiter), "xatol": 1e-4, "fatol": 1e-14},
    )
    theta_polished = adapter.theta_from_unconstrained(
        torch.from_numpy(np.asarray(res.x, dtype=np.float32))
    )
    return theta_polished, float(res.fun), int(res.nfev)


# --- Laplace degeneracy diagnostic (diagnostic only) -----------------------

def laplace_spectrum(
    model: FNO2d,
    obs: ObservationSet,
    theta_hat: torch.Tensor,
    sigma_eff2: float,
    adapter: InverseAdapter = _SOURCE_ITR_SIN_ADAPTER,
) -> dict:
    """Gauss-Newton eigen-spectrum at ``theta_hat`` (degeneracy diagnostic).

    Builds the observation Jacobian ``J = dG/dtheta`` and the Gauss-Newton
    Hessian of the NLL ``H = J^T J / sigma_eff2``. Its eigenvalues are
    ``S_k^2 / sigma_eff2`` and eigenvectors the right singular vectors of ``J``,
    so this reuses :func:`svd_identifiability`. Small eigenvalues + an
    least-identified direction describe the uncertainty that the
    profile must handle. Reported as a diagnostic; NOT used for intervals
    (the Gaussian-around-MAP is least valid exactly along this degenerate
    direction and at the A=0 boundary).
    """
    J = observation_jacobian(model, obs, theta_hat, adapter).cpu().numpy()
    ident = svd_identifiability(J)
    S = ident["singular_values"]
    eig = (S ** 2) / float(sigma_eff2)
    return {
        "eigenvalues": eig,                       # descending
        "right_vectors": ident["right_vectors"],
        "least_identified_dir": ident["least_identified_dir"],
        "cond_number": float(eig[0] / eig[-1]) if eig[-1] > 0 else float("inf"),
    }


def laplace_summary(
    spec: dict, adapter: InverseAdapter = _SOURCE_ITR_SIN_ADAPTER
) -> dict:
    """Flatten a :func:`laplace_spectrum` into CSV columns."""
    return adapter.laplace_summary(spec)


if __name__ == "__main__":
    raise SystemExit(main())
