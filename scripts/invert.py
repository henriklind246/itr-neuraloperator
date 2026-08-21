"""Benchmark-adapter inverse solver for source_itr, forcing, and forcing_itr.

For a real forcing checkpoint + held-out sims, run:

    PYTHONPATH=$(pwd) .venv/bin/python scripts/invert.py --checkpoint <forcing_run>/seed42/fno2d_best.pt --data-dir <forcing_data_dir> --sim-ids <ids...>

Use the *frozen forward FNO* as a
differentiable surrogate to recover inverse parameters

    source_itr / forcing_itr: theta = (R_base, R_amp, y0, sigma)
    forcing:    theta = (R_c,)

from observed temperatures, via a MAP / least-squares fit. The forward code is
untouched; this script only *imports* it.

The pipeline that produces the estimate is unchanged:
  * direct-from-IC observation operator with every non-theta input held fixed,
  * observations at several times spanning early/mid/late, either full-field or
    through a masked interface-proximal *sensor operator* M (paired
    interface-adjacent nodes or a finite interface band crossed with physical y
    coordinates; full-field is the all-pixels case),
  * LHS multistart Adam -> L-BFGS over an unconstrained reparameterization that
    keeps every iterate inside the physical box and honors the R_base-dependent
    R_amp ceiling (so the FNO is never queried out-of-distribution),
  * iid Gaussian sensor noise added once to the observations.

**Three statistics are reported per simulation, and only these three.** Each
run emits them all; there are no flags to turn individual ones on or off.

  1. *Recovery error of the lead estimand* (``summarize``). ``R_c`` for
     forcing, the integrated excess resistance ``S_R`` for the spatial-ITR
     benchmarks: the estimate, the truth, and the absolute error. This is the
     statistic the inverse claim rests on.

  2. *FV-verified sensor residual at theta_hat* (``fv_refine_report`` +
     ``fv_verification_summary``). The FNO is the fast exploration engine, but
     the conservative Crank-Nicolson finite-volume solver is the accuracy
     engine, so ``theta_hat`` is re-evaluated with the **real** ``FVSolver2D``
     (rebuilt via ``ds.problem.configure_solver``, i.e. the exact
     data-generation operator) and the residual is reported in Kelvin against
     the measurement noise floor. Without it, "the surrogate recovered theta"
     is indistinguishable from "the surrogate fit its own error".

  3. *Profile-likelihood interval on the lead estimand*
     (``profile_likelihood`` + ``profile_interval_summary``). A Wilks interval
     from the Gaussian NLL at the frozen validation-calibrated variance
     ``sigma_eff^2 = (sigma_meas^2 + sigma_FNO,cal^2) * D``, where ``D`` is the
     residual design effect that deflates the effective sensor/time count to
     account for correlated surrogate error the sum-of-squares NLL would treat
     as independent. Deterministic, with no sampler to mistune. Each pinned
     nuisance refit is multistart (``--profile-starts``) with bidirectional
     continuation across the grid, for the same reason the MAP fit is
     multistart: a nuisance fit stuck in a local minimum reports an inflated
     NLL, which biases the Wilks interval *narrow*. ``profile_sweep_gap``
     records how much the reverse sweep improved on the forward one and should
     be ~0. Requires ``--calibration-artifact``; without one it is skipped with
     a printed note rather than silently faked.

The diagnostics the earlier report computed beyond these three (the Jacobian-SVD
identifiability block, the Laplace spectrum, the FV polish, and the
equivalent-scalar comparison) are retained verbatim at the bottom of this file
under "RETIRED DIAGNOSTICS", where the header records why each one left the
reported set. Nothing on the live path calls them. The statistics-only members
of that block - the RW-Metropolis MCMC (a second interval on the same scalar the
profile already covers, and mistuned when live), the ``profile_summary`` /
``mcmc_summary`` coverage flatteners, and their ``theta_logabsdet_du`` prior
Jacobian - have been removed outright.

theta enters the FNO in *two* places, and both must agree with the data
pipeline exactly:
  1. ``cond_static[1:5]`` — linear-normalized over ``RC_VOID_RANGES``
     (mirrors the active spatial-ITR condition builder).
  2. spatial channel ``RC_Y_CHANNEL`` (=4) — ``rc_log_norm(make_rc_void_profile(...))``
     broadcast across x.

Everything Stage 1 needs that is *not* theta (IC snapshot, x/y channels, known
forcing/source fields, forcing sequence, and non-ITR condition slots) is taken verbatim from the
real ``problem.build_item(ds, sid, s=0, j=n)`` so the scaffolding cannot drift
from training. Only channel 4 and cond[1:5] are replaced by torch functions of
theta, so autograd produces dT/dtheta through the frozen model.

The profile interval is conditional on one frozen global calibration scale, not
a test-case oracle covariance. The recoverable quantity under sparse sensing is
the integrated excess resistance S_R, not necessarily the individual
R_amp/sigma shape scalars.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import (  # noqa: E402
    SnapshotPairDataset,
    T_EPS,
    load_sim_data,
    problem_from_config,
)
from scripts.inverse_adapters import (  # noqa: E402
    InverseAdapter,
    SourceItrAdapter,
    VOID_PARAM_NAMES,
)
from src.operators.fno2d import FNO2d  # noqa: E402

_SOURCE_ITR_ADAPTER = SourceItrAdapter()
PAPER_OBSERVATION_TIMES = (0.07, 0.15, 0.30)
PAPER_SENSOR_Y = tuple(np.linspace(0.1, 0.9, 8))

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
    """Legacy source_itr wrapper for ``SourceItrAdapter.theta_from_unconstrained``."""
    return _SOURCE_ITR_ADAPTER.theta_from_unconstrained(u)


def unconstrained_from_theta(theta: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Legacy source_itr wrapper for ``SourceItrAdapter.unconstrained_from_theta``."""
    return _SOURCE_ITR_ADAPTER.unconstrained_from_theta(theta)


def theta_to_cond_slice(theta: torch.Tensor) -> torch.Tensor:
    """Legacy source_itr wrapper for cond-static injection parity tests."""
    return _SOURCE_ITR_ADAPTER.cond_slice_from_theta(theta)


def theta_to_rc_channel(
    theta: torch.Tensor, y_grid: torch.Tensor, Nx: int
) -> torch.Tensor:
    """Legacy source_itr wrapper for the normalized R_c(y) spatial channel."""
    return _SOURCE_ITR_ADAPTER.spatial_channel_from_theta(theta, y_grid, Nx)


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
    forcing_benchmarks = {"forcing", "forcing_itr", "forcing_itr_sin"}
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
# Dataset construction over a local source_itr artifact directory.
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
    loss. ``cond`` carries the lead/patch slots; columns 1:5 are overwritten.
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
    adapter: InverseAdapter = _SOURCE_ITR_ADAPTER,
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
    adapter: InverseAdapter = _SOURCE_ITR_ADAPTER,
) -> torch.Tensor:
    """G_theta over all observation times: returns (N, Nx, Ny, 1) normalized T.

    ``theta`` is a single physical (4,) vector. Both injection points are
    rebuilt from theta and spliced into the (otherwise detached) scaffolding so
    autograd flows only through cond[1:5] and channel RC_Y_CHANNEL.

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
    starts evenly tile every void scalar's range. Returns ``(n_starts, 4)``.
    """
    return _SOURCE_ITR_ADAPTER.lhs_starts_unconstrained(n_starts, rng, eps)


# ---------------------------------------------------------------------------
# MAP optimization: multistart Adam -> L-BFGS.
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
    adapter: InverseAdapter = _SOURCE_ITR_ADAPTER,
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
    adapter: InverseAdapter = _SOURCE_ITR_ADAPTER,
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

def integrated_excess_resistance(theta: torch.Tensor, y_grid: torch.Tensor) -> float:
    """Trapezoidal integral of (R_c(y) - R_base) over y: total void "mass"."""
    return _SOURCE_ITR_ADAPTER.uq_quantity(theta, y_grid)


def summarize(
    result: InversionResult,
    y_grid: torch.Tensor,
    adapter: InverseAdapter = _SOURCE_ITR_ADAPTER,
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
    """Legacy source_itr wrapper for exact FV base kwargs."""
    return _SOURCE_ITR_ADAPTER.fv_base_kwargs(ds)


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
    adapter: InverseAdapter = _SOURCE_ITR_ADAPTER,
    obs: Optional[ObservationSet] = None,
) -> torch.Tensor:
    """FV forward of ``theta`` at the observation sensors/times (normalized).

    Overrides only the four void scalars (and the mirrored ``R_c`` slot) in a
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
    adapter: InverseAdapter = _SOURCE_ITR_ADAPTER,
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
    adapter: InverseAdapter = _SOURCE_ITR_ADAPTER,
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
# Reported statistic 3: profile-likelihood interval on the lead estimand.
#
# iid Gaussian sensor noise is added once to the observations, and the
# likelihood uses the frozen validation-calibrated variance
# ``sigma_eff^2 = (sigma_meas^2 + sigma_FNO,cal^2) * D``. The interval leads with
# the well-conditioned quantity (``S_R`` for the spatial-ITR benchmarks, ``R_c``
# for forcing) rather than with individually weak shape scalars.
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
_CHI2_HALF_THRESH = {0.68: 0.5, 0.90: 1.352772, 0.95: 1.920729, 0.99: 3.317448}


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
    adapter: InverseAdapter = _SOURCE_ITR_ADAPTER,
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
    adapter: InverseAdapter = _SOURCE_ITR_ADAPTER,
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
    """Legacy source_itr wrapper for profile pinning tests."""
    return _SOURCE_ITR_ADAPTER.theta_profile(u, fixed_index, fixed_value)


def _profile_refit(
    model: FNO2d,
    obs: ObservationSet,
    sigma_eff2: float,
    fixed_index: int,
    fixed_value: float,
    u_seeds: Sequence[torch.Tensor],
    *,
    adam_steps: int = 150,
    adam_lr: float = 0.05,
    lbfgs_steps: int = 30,
    adapter: InverseAdapter = _SOURCE_ITR_ADAPTER,
    theta_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    """Re-optimize the free scalars at a pinned constraint, over several starts.

    Minimizes the Gaussian NLL over ``u`` (the pinned coord's ``u`` is inert),
    Adam then an L-BFGS polish, from *each* seed in ``u_seeds``, and returns the
    best ``(nll_min, theta, u)``. By default the pin is
    ``theta[fixed_index] = fixed_value`` via ``adapter.theta_profile``; pass
    ``theta_fn`` to pin a derived quantity instead (e.g. the integrated
    severity), in which case ``fixed_index``/``fixed_value`` are ignored.

    Multistart is not optional decoration: the nuisance objective is the same
    nonconvex surface the MAP fit already needs ``n_starts`` for, and a nuisance
    refit stuck in a local minimum reports an inflated NLL, which pushes the
    profile up and makes the Wilks interval too *narrow*.
    """
    if len(u_seeds) == 0:
        raise ValueError("_profile_refit needs at least one seed")

    def make_theta(u_: torch.Tensor) -> torch.Tensor:
        if theta_fn is not None:
            return theta_fn(u_)
        return adapter.theta_profile(u_, fixed_index, fixed_value)

    device = obs.spatial.device

    if theta_fn is None and adapter.theta_dim == 1:
        # Nothing is free once the single scalar is pinned: every seed gives the
        # same theta, so extra starts would be pure cost.
        u0 = u_seeds[0].detach().to(device)
        theta_device = make_theta(u0)
        final = float(neg_log_likelihood(model, obs, theta_device, sigma_eff2, adapter))
        return final, theta_device.detach().cpu(), u0.detach().cpu()

    best_nll = float("inf")
    best_theta: Optional[torch.Tensor] = None
    best_u: Optional[torch.Tensor] = None

    for u_seed in u_seeds:
        u = u_seed.clone().detach().to(device).requires_grad_(True)

        def nll():
            theta = make_theta(u)
            return neg_log_likelihood(model, obs, theta, sigma_eff2, adapter)

        adam = torch.optim.Adam([u], lr=adam_lr)
        for _ in range(adam_steps):
            adam.zero_grad()
            loss = nll()
            loss.backward()
            adam.step()

        lbfgs = torch.optim.LBFGS(
            [u], max_iter=lbfgs_steps, line_search_fn="strong_wolfe"
        )

        def closure():
            lbfgs.zero_grad()
            loss = nll()
            loss.backward()
            return loss

        lbfgs.step(closure)

        with torch.no_grad():
            theta_device = make_theta(u)
            final = float(
                neg_log_likelihood(model, obs, theta_device, sigma_eff2, adapter)
            )
        if final < best_nll:
            best_nll = final
            best_theta = theta_device.detach().cpu()
            best_u = u.detach().cpu()

    return best_nll, best_theta, best_u


def _profile_sweep(
    model: FNO2d,
    obs: ObservationSet,
    sigma_eff2: float,
    grid: np.ndarray,
    u_hat: torch.Tensor,
    *,
    adapter: InverseAdapter,
    pin_index: int = -1,
    theta_fn_of: Optional[Callable[[float], Callable[[torch.Tensor], torch.Tensor]]] = None,
    n_starts: int = 3,
    adam_steps: int = 150,
    lbfgs_steps: int = 30,
    seed: int = 0,
) -> tuple[np.ndarray, list[torch.Tensor], float]:
    """Trace a profile over ``grid`` with multistart + bidirectional continuation.

    Two sweeps are run. The forward sweep (low to high) fits each point from the
    MAP seed ``u_hat``, ``n_starts - 1`` LHS starts, and the previous point's
    solution; the reverse sweep re-fits each point from its neighbor on the
    other side. Each grid point keeps the lowest NLL seen by either sweep, so a
    minimum found anywhere on the grid propagates to its neighbors instead of
    leaving a spurious bump in the profile.

    Returns ``(nll, thetas, sweep_gap)`` where ``sweep_gap`` is the largest
    amount the reverse sweep improved on the forward one — a residual-nonconvexity
    diagnostic that should be at or near zero on a converged profile.
    """
    rng = np.random.default_rng(seed)
    extra = adapter.lhs_starts_unconstrained(max(0, n_starts - 1), rng)
    extra_seeds = [torch.from_numpy(row) for row in extra]

    def refit(value: float, seeds: list[torch.Tensor]):
        theta_fn = None if theta_fn_of is None else theta_fn_of(value)
        return _profile_refit(
            model, obs, sigma_eff2, pin_index, value, seeds,
            adam_steps=adam_steps, lbfgs_steps=lbfgs_steps, adapter=adapter,
            theta_fn=theta_fn,
        )

    n = len(grid)
    nll = np.empty(n, dtype=np.float64)
    thetas: list[Optional[torch.Tensor]] = [None] * n
    u_best: list[Optional[torch.Tensor]] = [None] * n

    prev_u: Optional[torch.Tensor] = None
    for i in range(n):
        seeds = [u_hat] + extra_seeds
        if prev_u is not None:
            seeds.append(prev_u)
        val, theta, u = refit(float(grid[i]), seeds)
        nll[i], thetas[i], u_best[i] = val, theta, u
        prev_u = u

    forward_nll = nll.copy()
    for i in range(n - 2, -1, -1):
        val, theta, u = refit(float(grid[i]), [u_best[i + 1]])
        if val < nll[i]:
            nll[i], thetas[i], u_best[i] = val, theta, u

    sweep_gap = float(np.max(forward_nll - nll)) if n else 0.0
    return nll, [t for t in thetas], sweep_gap


@dataclass
class ProfileResult:
    param_index: int
    param_name: str
    level: float
    grid: np.ndarray            # (G,) pinned physical values
    nll: np.ndarray             # (G,) profiled NLL at each grid value
    excess_int: np.ndarray      # (G,) integrated severity along the profile
    nll_min: float
    ci_low: float               # physical param CI (Wilks)
    ci_high: float
    excess_ci_low: float        # severity CI (lead deliverable)
    excess_ci_high: float
    sweep_gap: float = 0.0      # max NLL the reverse sweep clawed back (~0 = converged)
    n_starts: int = 1


def _threshold_crossings(
    grid: np.ndarray, delta: np.ndarray, thresh: float
) -> tuple[float, float]:
    """Linear-interpolated ``[lo, hi]`` where ``delta(grid) <= thresh``.

    ``delta`` is ``NLL - NLL_min`` over the grid (assumed roughly U-shaped). The
    interval is the outermost crossings of ``thresh``; if the boundary itself is
    below ``thresh`` the interval is open there (clamped to the grid edge).
    """
    inside = delta <= thresh
    if not inside.any():
        k = int(np.argmin(delta))
        return float(grid[k]), float(grid[k])
    idx = np.where(inside)[0]
    i0, i1 = idx[0], idx[-1]

    def interp(a, b):
        da, db = delta[a], delta[b]
        if db == da:
            return float(grid[a])
        w = (thresh - da) / (db - da)
        return float(grid[a] + w * (grid[b] - grid[a]))

    lo = float(grid[i0]) if i0 == 0 else interp(i0, i0 - 1)
    hi = float(grid[i1]) if i1 == len(grid) - 1 else interp(i1, i1 + 1)
    return lo, hi


def profile_likelihood(
    model: FNO2d,
    obs: ObservationSet,
    theta_hat: torch.Tensor,
    sigma_eff2: float,
    *,
    param_index: int = 1,
    n_grid: int = 11,
    span: float = 0.6,
    level: float = 0.95,
    adam_steps: int = 150,
    lbfgs_steps: int = 30,
    n_starts: int = 3,
    seed: int = 0,
    adapter: InverseAdapter = _SOURCE_ITR_ADAPTER,
) -> ProfileResult:
    """Profile-likelihood interval for one void scalar (default ``R_amp``).

    Pins ``theta[param_index]`` to each of ``n_grid`` physical values spanning
    ``theta_hat[param_index] +/- span`` (clamped to the parameter's physical
    range), re-fits the other three by minimizing the Gaussian NLL, and traces
    the profile. The confidence interval uses Wilks' theorem
    (``2*(NLL - NLL_min) <= chi2_{1,level}``). Crucially we also record the
    integrated severity along the profile and report its min/max inside the CI:
    that severity interval is the well-conditioned lead deliverable, while the
    ``R_amp`` CI itself is expected to be wide / boundary-limited.

    Each nuisance refit is multistart (``n_starts`` seeds: ``theta_hat`` plus
    LHS) with bidirectional continuation across the grid; see ``_profile_sweep``.
    """
    name = adapter.param_names[param_index]
    lo_b, hi_b = adapter.profile_bounds(param_index)

    center = float(theta_hat[param_index])
    grid = np.linspace(
        max(lo_b, center - span), min(hi_b, center + span), int(n_grid)
    )
    u_hat = adapter.unconstrained_from_theta(theta_hat.detach().cpu())

    nll, thetas, sweep_gap = _profile_sweep(
        model, obs, sigma_eff2, grid, u_hat, adapter=adapter,
        pin_index=param_index, n_starts=n_starts,
        adam_steps=adam_steps, lbfgs_steps=lbfgs_steps, seed=seed,
    )
    excess = np.array(
        [adapter.uq_quantity(t, obs.y_grid) for t in thetas], dtype=np.float64
    )

    nll_min = float(nll.min())
    thresh = _CHI2_HALF_THRESH.get(level, 1.920729)
    delta = nll - nll_min
    ci_low, ci_high = _threshold_crossings(grid, delta, thresh)
    inside = delta <= thresh
    exc_in = excess[inside] if inside.any() else excess[np.argmin(delta):np.argmin(delta) + 1]
    return ProfileResult(
        param_index=param_index, param_name=name, level=level,
        grid=grid, nll=nll, excess_int=excess, nll_min=nll_min,
        ci_low=ci_low, ci_high=ci_high,
        excess_ci_low=float(exc_in.min()), excess_ci_high=float(exc_in.max()),
        sweep_gap=sweep_gap, n_starts=int(n_starts),
    )


def severity_profile_likelihood(
    model: FNO2d,
    obs: ObservationSet,
    theta_hat: torch.Tensor,
    sigma_eff2: float,
    *,
    n_grid: int = 11,
    span: float = 0.6,
    level: float = 0.95,
    adam_steps: int = 150,
    lbfgs_steps: int = 30,
    n_starts: int = 3,
    seed: int = 0,
    adapter: InverseAdapter = _SOURCE_ITR_ADAPTER,
) -> ProfileResult:
    """Direct profile-likelihood interval for the integrated severity ``S_R``.

    Unlike ``profile_likelihood`` (which profiles a single scalar such as
    ``R_amp`` and then reads off the severity *along that path*), this pins the
    severity itself onto a grid and re-fits *all* remaining freedom by
    minimizing the Gaussian NLL, so the reported interval is a genuine
    profile-likelihood CI for ``S_R``: ``l_p(s) = min_{theta: S_R(theta)=s}
    NLL(theta)``, thresholded by Wilks. The pin is enforced exactly and
    differentiably by ``adapter.theta_profile_severity`` (solving the amplitude
    that realizes ``s`` for the shape drawn from the free coordinates).

    ``span`` is a *fractional* half-width around ``S_R(theta_hat)`` here (grid
    spans ``s_hat * [1 - span, 1 + span]``), not a physical-unit offset, because
    severity has no fixed scale across benchmarks. The grid is clamped to
    ``(0, max_feasible_severity]``.

    The nuisance fit here is over the void *shape* ``(y0, sigma)`` plus
    ``R_base``, which is genuinely multimodal in ``y0``, so each pin is refit
    from ``n_starts`` seeds with bidirectional continuation (``_profile_sweep``).
    """
    y_grid = obs.y_grid
    s_hat = float(adapter.uq_quantity(theta_hat, y_grid))
    s_max = float(adapter.max_feasible_severity(y_grid))
    atol = 1e-9 + 1e-6 * max(s_max, 1.0)
    lo = max(atol, s_hat * (1.0 - span))
    hi = min(s_max, s_hat * (1.0 + span))
    if hi <= lo:
        hi = min(s_max, lo + atol)
    grid = np.linspace(lo, hi, int(n_grid))
    u_hat = adapter.unconstrained_from_theta(theta_hat.detach().cpu())

    def theta_fn_of(s_val: float) -> Callable[[torch.Tensor], torch.Tensor]:
        def theta_fn(uu: torch.Tensor, _s: float = float(s_val)) -> torch.Tensor:
            return adapter.theta_profile_severity(uu, _s, y_grid)
        return theta_fn

    nll, thetas, sweep_gap = _profile_sweep(
        model, obs, sigma_eff2, grid, u_hat, adapter=adapter,
        theta_fn_of=theta_fn_of, n_starts=n_starts,
        adam_steps=adam_steps, lbfgs_steps=lbfgs_steps, seed=seed,
    )
    excess = np.array(
        [float(adapter.uq_quantity(t, y_grid)) for t in thetas], dtype=np.float64
    )

    nll_min = float(nll.min())
    thresh = _CHI2_HALF_THRESH.get(level, 1.920729)
    ci_low, ci_high = _threshold_crossings(grid, nll - nll_min, thresh)
    return ProfileResult(
        param_index=adapter.default_profile_index, param_name="S_R", level=level,
        grid=grid, nll=nll, excess_int=excess, nll_min=nll_min,
        ci_low=ci_low, ci_high=ci_high,
        excess_ci_low=float(ci_low), excess_ci_high=float(ci_high),
        sweep_gap=sweep_gap, n_starts=int(n_starts),
    )


def profile_interval_summary(
    res: ProfileResult,
    adapter: InverseAdapter = _SOURCE_ITR_ADAPTER,
) -> dict:
    """Reported form of statistic 3: the lead-estimand interval, no coverage.

    Emits ``low``/``high``/``width`` for the *lead* quantity only — the
    severity integral where the benchmark has one, otherwise the profiled
    scalar itself — under the column names the plot layer's
    ``lead_profile_ci_*_col`` fields already expect.

    ``bound_limited`` flags an interval that ran into the edge of the profiled
    parameter's physical range instead of closing on a Wilks crossing. Those
    widths are censored and must not be averaged in as if they were measured;
    the removed ``profile_summary`` reported width without saying so.

    Deliberately no ``*_covered`` boolean: see the coverage note in the retired
    section header.
    """
    severity_profile = res.param_name == "S_R"
    if adapter.reports_spatial_severity:
        low, high = res.excess_ci_low, res.excess_ci_high
        stem = "profile_excess"
    else:
        low, high = res.ci_low, res.ci_high
        stem = f"profile_{res.param_name}"
    if severity_profile:
        # The severity grid is its own domain: a censored width is one that
        # closed on the grid edge rather than a Wilks crossing.
        bound_lo, bound_hi = float(res.grid[0]), float(res.grid[-1])
    else:
        bound_lo, bound_hi = adapter.profile_bounds(res.param_index)
    atol = 1e-9 + 1e-6 * (bound_hi - bound_lo)
    return {
        "profile_param": res.param_name,
        "profile_level": res.level,
        f"{stem}_ci_low": float(low),
        f"{stem}_ci_high": float(high),
        f"{stem}_ci_width": float(high - low),
        "profile_bound_limited": bool(
            low <= bound_lo + atol or high >= bound_hi - atol
        ),
        "profile_n_starts": int(res.n_starts),
        "profile_sweep_gap": float(res.sweep_gap),
    }


# ---------------------------------------------------------------------------
# Per-sim artifact dump: raw profile curves, MCMC samples, and the observation
# Jacobian only ever exist as locals in main()'s loop; the CSV keeps one summary
# row per sim. Figs 2 & 4 of the inverse-results section need the raw arrays, so
# (opt-in via --artifact-dir) we write one self-describing NPZ per sim.
# ---------------------------------------------------------------------------

_ARTIFACT_SCHEMA_VERSION = 2


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
    severity_values: list[float] = []
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
        if adapter.reports_spatial_severity:
            severity_values.append(adapter.uq_quantity(obs.theta_true, obs.y_grid))
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
        "S_R": np.asarray(severity_values, dtype=np.float64),
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
    prof: Optional[ProfileResult],
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
        "param_names": np.asarray(list(adapter.param_names), dtype=object),
        "theta_hat": theta_hat,
        "theta_true": theta_true,
        "theta_bounds": theta_bounds,
        "param_scales": np.asarray(adapter.param_scales, dtype=np.float64),
        "noise_std": np.float64(noise_std),
        "ci_level": np.float64(ci_level),
        "profile_threshold": np.float64(
            _CHI2_HALF_THRESH.get(round(float(ci_level), 2), float("nan"))
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

    if prof is not None:
        payload["profile_grid"] = np.asarray(prof.grid, dtype=np.float64)
        payload["profile_nll"] = np.asarray(prof.nll, dtype=np.float64)
        payload["profile_excess_int"] = np.asarray(prof.excess_int, dtype=np.float64)
        payload["profile_nll_min"] = np.float64(prof.nll_min)
        payload["profile_param_index"] = np.int64(prof.param_index)
        payload["profile_param_name"] = np.str_(prof.param_name)
        payload["profile_excess_ci_low"] = np.float64(prof.excess_ci_low)
        payload["profile_excess_ci_high"] = np.float64(prof.excess_ci_high)
        payload["profile_n_starts"] = np.int64(prof.n_starts)
        payload["profile_sweep_gap"] = np.float64(prof.sweep_gap)

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
                    help="Parameter index to profile (adapter default: R_amp for spatial ITR, R_c for scalar forcing)")
    ap.add_argument("--profile-grid", type=int, default=11,
                    help="Number of pinned grid points for the profile (default 11)")
    ap.add_argument("--profile-starts", type=int, default=3,
                    help="Optimization starts per profile grid point (theta_hat "
                         "plus LHS). Bidirectional continuation across the grid "
                         "is always on. 1 restores the old single-start refit")
    ap.add_argument("--profile-span", type=float, default=0.6,
                    help="Grid half-width around theta_hat: a physical offset for "
                         "the scalar-parameter profile, a fractional half-width "
                         "for the direct severity profile (spatial-ITR benchmarks)")
    ap.add_argument("--artifact-dir", default=None,
                    help="If set, write one self-describing sim_<id>.npz per sim "
                         "(raw profile curves + split provenance) for the "
                         "inverse-results figures")
    args = ap.parse_args(argv)

    loaded = load_checkpoint(args.checkpoint, device=args.device)
    adapter = InverseAdapter.from_config(loaded.config)
    adapter.validate_model(loaded)
    profile_index = (
        args.profile_index
        if args.profile_index is not None
        else adapter.default_profile_index
    )
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
        fv_res = prof = sigma_eff2 = None
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
        theta_hat = result.theta_hat.to(args.device)
        summary = summarize(result, obs.y_grid, adapter)
        summary["benchmark"] = adapter.benchmark
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

        # --- Statistic 3: profile-likelihood interval on the lead estimand. ---
        if do_profile:
            sigma_eff2 = effective_sigma2(
                args.noise_std, sigma_fno_cal=sigma_fno_cal,
                design_effect=design_effect_cal,
            )
            if sigma_eff2 <= 0:
                print(
                    f"[sim {sid}] profile interval skipped: sigma_eff2<=0 "
                    f"(supply --noise-std for a measurement-noise model)"
                )
                sigma_eff2 = None
            elif adapter.reports_spatial_severity:
                prof = severity_profile_likelihood(
                    loaded.model, obs, theta_hat, sigma_eff2,
                    n_grid=args.profile_grid, span=args.profile_span,
                    level=args.uq_level, n_starts=args.profile_starts,
                    seed=args.seed, adapter=adapter,
                )
                summary.update(profile_interval_summary(prof, adapter))
            else:
                prof = profile_likelihood(
                    loaded.model, obs, theta_hat, sigma_eff2,
                    param_index=profile_index, n_grid=args.profile_grid,
                    span=args.profile_span, level=args.uq_level,
                    n_starts=args.profile_starts, seed=args.seed,
                    adapter=adapter,
                )
                summary.update(profile_interval_summary(prof, adapter))
        if args.artifact_dir:
            _write_sim_artifact(
                args.artifact_dir, int(sid), adapter,
                obs=obs, result=result, report=report, J=J,
                fv_res=fv_res, prof=prof,
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
        param_parts = []
        for i, name in enumerate(adapter.param_names):
            hat = float(result.theta_hat[i])
            true = (
                float(result.theta_true[i])
                if result.theta_true is not None
                else float("nan")
            )
            param_parts.append(f"{name} {hat:.3f}/{true:.3f}")
        quantity_part = ""
        if adapter.reports_spatial_severity:
            quantity_part = (
                f"  excess_int {summary['excess_int_hat']:.4f}/"
                f"{summary.get('excess_int_true', float('nan')):.4f}"
            )
        ci_part = ""
        stem = (
            "profile_excess"
            if adapter.reports_spatial_severity
            else f"profile_{adapter.param_names[profile_index]}"
        )
        if f"{stem}_ci_low" in summary:
            ci_part = (
                f"  CI=[{summary[f'{stem}_ci_low']:.4f},"
                f"{summary[f'{stem}_ci_high']:.4f}]"
                + ("*" if summary.get("profile_bound_limited") else "")
            )
        print(
            f"[sim {sid}] loss={summary['loss']:.3e}  n_sensors={n_sensors}  "
            + "  ".join(param_parts)
            + quantity_part
            + (
                f"  fv_resid={summary['fv_resid_rms_K']:.4g} K"
                + (
                    f" ({summary['fv_resid_over_noise']:.2f}x noise)"
                    if np.isfinite(summary["fv_resid_over_noise"])
                    else ""
                )
            )
            + ci_part
        )

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
#     A separate study with its own protocol, and forcing_itr-only.
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
# coupled R_amp/sigma void-"mass" ridge). This is *diagnostic only* — no UQ,
# no Laplace/Hessian intervals (those are Stage 5).
# ---------------------------------------------------------------------------

def observation_jacobian(
    model: FNO2d,
    obs: ObservationSet,
    theta: torch.Tensor,
    adapter: InverseAdapter = _SOURCE_ITR_ADAPTER,
) -> torch.Tensor:
    """Jacobian ``J = d(masked G_theta)/d theta`` at ``theta``: ``(m, 4)``.

    ``m`` is the number of observed scalars (sensor pixels times observation
    times times channels). The model is frozen; only the four physical void
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
    """Identifiability summary from the observation Jacobian ``J`` (m, 4).

    Pure linear algebra (unit-tested independently of the FNO):
      * ``singular_values`` — descending; small values flag directions the data
        cannot constrain.
      * ``right_vectors`` — (4, 4), columns are orthonormal directions in
        theta-space ``(R_base, R_amp, y0, sigma)``.
      * ``param_sensitivity`` — per-parameter column norms ``||dG/dtheta_k||``.
      * ``least_identified_dir`` — right singular vector of the *smallest*
        singular value (the worst-constrained combination).
      * ``ramp_sigma_alignment`` — source_itr-only fraction of that direction's
        norm lying in the ``(R_amp, sigma)`` plane.
      * ``cond_number`` — s_max / s_min.
    """
    Jn = np.asarray(J, dtype=np.float64)
    param_sensitivity = np.linalg.norm(Jn, axis=0)
    _, S, Vt = np.linalg.svd(Jn, full_matrices=False)
    V = Vt.T
    least = V[:, -1]
    least = least / (np.linalg.norm(least) + 1e-300)
    ramp_sigma_alignment = (
        float(np.linalg.norm(least[[1, 3]]) / (np.linalg.norm(least) + 1e-300))
        if least.shape[0] > 3
        else float("nan")
    )
    cond_number = float(S[0] / S[-1]) if S[-1] > 0 else float("inf")
    return {
        "singular_values": S,
        "right_vectors": V,
        "param_sensitivity": param_sensitivity,
        "least_identified_dir": least,
        "ramp_sigma_alignment": ramp_sigma_alignment,
        "cond_number": cond_number,
    }


def sensitivity_report(
    model: FNO2d,
    obs: ObservationSet,
    theta: torch.Tensor,
    adapter: InverseAdapter = _SOURCE_ITR_ADAPTER,
) -> dict:
    """``svd_identifiability`` of the observation Jacobian at ``theta``."""
    J = observation_jacobian(model, obs, theta, adapter)
    return svd_identifiability(J.cpu().numpy())


def sensitivity_summary(
    report: dict, adapter: InverseAdapter = _SOURCE_ITR_ADAPTER
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
    adapter: InverseAdapter = _SOURCE_ITR_ADAPTER,
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
    adapter: InverseAdapter = _SOURCE_ITR_ADAPTER,
) -> dict:
    """Gauss-Newton eigen-spectrum at ``theta_hat`` (degeneracy diagnostic).

    Builds the observation Jacobian ``J = dG/dtheta`` and the Gauss-Newton
    Hessian of the NLL ``H = J^T J / sigma_eff2``. Its eigenvalues are
    ``S_k^2 / sigma_eff2`` and eigenvectors the right singular vectors of ``J``,
    so this reuses :func:`svd_identifiability`. Small eigenvalues + an
    ``(R_amp, sigma)``-aligned least-identified direction confirm the ridge the
    profile must handle. Reported as a diagnostic; NOT used for intervals
    (the Gaussian-around-MAP is least valid exactly along this degenerate
    direction and at the R_amp~0 boundary).
    """
    J = observation_jacobian(model, obs, theta_hat, adapter).cpu().numpy()
    ident = svd_identifiability(J)
    S = ident["singular_values"]
    eig = (S ** 2) / float(sigma_eff2)
    return {
        "eigenvalues": eig,                       # descending
        "right_vectors": ident["right_vectors"],
        "least_identified_dir": ident["least_identified_dir"],
        "ramp_sigma_alignment": ident["ramp_sigma_alignment"],
        "cond_number": float(eig[0] / eig[-1]) if eig[-1] > 0 else float("inf"),
    }


def laplace_summary(
    spec: dict, adapter: InverseAdapter = _SOURCE_ITR_ADAPTER
) -> dict:
    """Flatten a :func:`laplace_spectrum` into CSV columns."""
    return adapter.laplace_summary(spec)


if __name__ == "__main__":
    raise SystemExit(main())
