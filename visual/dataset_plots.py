"""Dataset exploration and model-evaluation plots.

Covers raw trajectory/parameter visualisation (uniformly under registry group
``data``) plus held-out model diagnostics (predictions vs truth, error
breakdowns, interface jump analysis).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from data.dataset import (
    COND_DIM,
    RC_RANGE,
    T_EPS,
    SnapshotPairDataset,
    build_cond_vector,
    compute_global_stats,
    load_sim_data,
    split_sim_ids,
)
from src.physics.boundary_forcing import (
    FORCING_BINS,
    SIN_AMP_RANGE,
    SPATIAL_BUILDERS,
    build_qL,
    integrate_temporal_bins,
)


def _sin_amp_freq(params: dict) -> tuple[float, float]:
    """Pull (amp, freq) from a sim's temporal params; NaN if not the sin family."""
    if params.get("temporal_family") == "sin":
        tp = params["temporal_params"]
        return float(tp["A"]), float(tp["f"])
    return float("nan"), float("nan")


def _forcing_label(params: dict) -> str:
    """Short human-readable label for a sim's temporal forcing."""
    fam = params.get("temporal_family", "?")
    tp = params.get("temporal_params", {})
    if fam == "sin":
        return f"sin A={tp['A']:.0f} f={tp['f']:.2f}"
    if fam == "exp":
        return f"exp A={tp['A']:.0f} t0={tp['t0']:.2f} tau={tp['tau']:.3f}"
    if fam in ("pulse_train", "exp_train"):
        return f"{fam} Np={tp['Np']}"
    return fam
from src.operators.train import load_config
from src.physics.fv_solver_2d import FVSolver2D, Layer2D
from src.operators.fno2d import FNO2d

from visual._common import (
    PLOT_STYLE,
    _add_interface_lines,
    _evaluate_q_left_field,
    _resolve_interface_metadata,
    _save_figure,
)


# ============================================================
# HELPERS — Index sampling and binning
# ============================================================

def _future_target_indices(start_idx: int, n_requested: int, n_total: int) -> np.ndarray:
    """Return unique, increasing future indices after start_idx."""
    if start_idx >= n_total - 1:
        raise ValueError("Source index must leave at least one future target.")

    available = np.arange(start_idx + 1, n_total, dtype=int)
    if n_requested >= len(available):
        return available

    positions = np.linspace(0, len(available) - 1, n_requested)
    indices = np.unique(np.round(positions).astype(int))
    while len(indices) < n_requested:
        indices = np.unique(np.concatenate([indices, np.arange(len(available), dtype=int)]))
        indices = indices[:n_requested]
    return available[indices]


def _build_interface_mask(x_grid: np.ndarray, interface_x: float, interface_half_width: float) -> np.ndarray:
    """Return a boolean mask for the configured interface region."""
    return np.abs(x_grid - interface_x) <= interface_half_width


def _interface_flanking_nodes_from_grid(x_grid: np.ndarray, interface_x: float) -> tuple[int, int]:
    """Return node indices immediately flanking an interface location."""
    iface_idx = int(np.argmin(np.abs(x_grid - interface_x)))
    left_node = iface_idx - 1 if x_grid[iface_idx] >= interface_x else iface_idx
    right_node = left_node + 1
    if left_node < 0 or right_node >= len(x_grid):
        raise ValueError("Interface location is outside the interior of the provided x_grid.")
    return left_node, right_node


def _build_interface_mask_2d(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    interface_x: float,
    interface_half_width: float,
) -> np.ndarray:
    """Return a 2D boolean mask for the configured interface region."""
    mask_1d = _build_interface_mask(x_grid, interface_x, interface_half_width)
    return np.broadcast_to(mask_1d[:, None], (len(x_grid), len(y_grid))).copy()


def _relative_l2_percent(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Return relative L2 error in percent."""
    denom = max(float(np.mean(y_true ** 2)), 1e-12)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2) / denom) * 100.0)


def _interface_jump_profile(field_2d: np.ndarray, left_node: int, right_node: int) -> np.ndarray:
    """Return the discrete interface jump profile ΔT(y) from adjacent cell centers."""
    # Discrete analog of the interface jump, evaluated from the two cell centers
    # immediately adjacent to the interface face.
    return np.asarray(field_2d[right_node, :] - field_2d[left_node, :], dtype=np.float32)


def _interface_jump_map(fields: np.ndarray, left_node: int, right_node: int) -> np.ndarray:
    """Return the discrete interface jump history ΔT(y, t) from a stack of fields."""
    return np.asarray(fields[:, right_node, :] - fields[:, left_node, :], dtype=np.float32)


def _jump_rms(jump_profile: np.ndarray) -> float:
    """Return RMS jump magnitude over y."""
    return float(np.sqrt(np.mean(np.asarray(jump_profile, dtype=np.float64) ** 2)))


def _shared_thumbnail_grid(n_panels: int) -> tuple[int, int]:
    """Return a near-square (nrows, ncols) layout for thumbnail-style figures."""
    if n_panels <= 0:
        return 1, 1
    ncols = int(np.ceil(np.sqrt(n_panels)))
    nrows = int(np.ceil(n_panels / ncols))
    return nrows, ncols


def _plot_field_2d(
    ax,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    field: np.ndarray,
    *,
    cmap: str = "inferno",
    vmin: float | None = None,
    vmax: float | None = None,
    norm=None,
    interface_positions: list[float] | None = None,
):
    """Render one 2D scalar field with the project plotting conventions."""
    pcm = ax.pcolormesh(x_grid, y_grid, np.asarray(field).T, cmap=cmap, shading="auto", vmin=vmin, vmax=vmax, norm=norm)
    if interface_positions:
        _add_interface_lines(ax, interface_positions)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    return pcm


def _compute_binned_quantiles(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bin x and return centers, median, q25, q75, counts for non-empty bins.

    Drops (x, y) pairs where either is NaN — needed because mixed-family
    datasets have NaN amplitude/frequency for non-sin sims.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if x.size == 0:
        empty = np.array([])
        return empty, empty, empty, empty, empty

    x_min = float(x.min())
    x_max = float(x.max())
    if np.isclose(x_min, x_max):
        q25, median, q75 = np.percentile(y, [25, 50, 75])
        return (
            np.array([x_min]),
            np.array([median]),
            np.array([q25]),
            np.array([q75]),
            np.array([len(x)]),
        )

    bins = np.linspace(x_min, x_max, n_bins + 1)
    bin_ids = np.clip(np.digitize(x, bins[1:-1], right=False), 0, n_bins - 1)

    centers = []
    medians = []
    q25s = []
    q75s = []
    counts = []
    for idx in range(n_bins):
        mask = bin_ids == idx
        if not np.any(mask):
            continue
        y_bin = y[mask]
        centers.append(0.5 * (bins[idx] + bins[idx + 1]))
        q25, median, q75 = np.percentile(y_bin, [25, 50, 75])
        q25s.append(float(q25))
        medians.append(float(median))
        q75s.append(float(q75))
        counts.append(int(mask.sum()))

    return (
        np.array(centers),
        np.array(medians),
        np.array(q25s),
        np.array(q75s),
        np.array(counts),
    )


def _parameter_arrays(sim_params: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return amplitude, frequency, and contact resistance arrays.

    Amplitude / frequency are sin-specific; non-sin sims contribute NaN.
    """
    pairs = [_sin_amp_freq(p) for p in sim_params]
    amplitudes = np.array([a for a, _ in pairs], dtype=np.float32)
    frequencies = np.array([f for _, f in pairs], dtype=np.float32)
    contact_resistance = np.array([float(p["R_c"]) for p in sim_params], dtype=np.float32)
    return amplitudes, frequencies, contact_resistance


def _select_representative_sim_id(sim_params: np.ndarray) -> int:
    """Choose a deterministic representative simulation near the median condition point.

    Restricted to sin-family sims because the median is computed over (A, f, R_c).
    Falls back to the first sim if no sin sims exist.
    """
    sin_idx = np.array(
        [i for i, p in enumerate(sim_params) if p.get("temporal_family") == "sin"],
        dtype=np.int64,
    )
    if sin_idx.size == 0:
        return 0
    amplitudes, frequencies, contact_resistance = _parameter_arrays(sim_params)
    A = amplitudes[sin_idx]
    F = frequencies[sin_idx]
    R = contact_resistance[sin_idx]
    stacked = np.column_stack(
        [
            (A - A.min()) / max(A.max() - A.min(), 1e-12),
            (np.log10(F) - np.log10(F).min())
            / max(np.log10(F).max() - np.log10(F).min(), 1e-12),
            (R - R.min()) / max(R.max() - R.min(), 1e-12),
        ]
    )
    target = np.median(stacked, axis=0)
    return int(sin_idx[int(np.argmin(np.sum((stacked - target) ** 2, axis=1)))])


def _spatial_localness(p: dict) -> float:
    """Smaller is more localized along y. Uniform returns +inf so it sorts last."""
    sf = p["spatial_family"]
    sp = p.get("spatial_params", {})
    if sf == "gaussian":
        return float(sp["sigma_y"])
    if sf == "triangle":
        return float(sp["ell"])
    if sf == "patch":
        return 0.5 * float(sp["w"])
    return float("inf")


def _select_localized_sim_id(sim_params: np.ndarray, family: str | None = None) -> int:
    """Return the sim index with the most localized spatial profile.

    If ``family`` is given, restrict the search to that family. Falls back to
    the first sim when no candidate matches (e.g. uniform-only).
    """
    candidates = [
        i for i, p in enumerate(sim_params)
        if (family is None and p["spatial_family"] != "uniform") or p["spatial_family"] == family
    ]
    if not candidates:
        return 0
    return min(candidates, key=lambda i: _spatial_localness(sim_params[i]))


def _format_spatial_family(p: dict) -> str:
    """Compact one-line description of a sim's spatial profile."""
    sf = p["spatial_family"]
    sp = p.get("spatial_params", {})
    if sf == "uniform":
        return "uniform"
    if sf == "patch":
        return f"patch(y_c={sp['y_c']:.2f}, w={sp['w']:.2f})"
    if sf == "gaussian":
        return f"gaussian(y_c={sp['y_c']:.2f}, σ={sp['sigma_y']:.3f})"
    if sf == "triangle":
        return f"triangle(y_c={sp['y_c']:.2f}, ell={sp['ell']:.2f})"
    return sf


def _snap_indices_in_window(t_grid: np.ndarray, t_lo: float, t_hi: float, n: int) -> np.ndarray:
    """Return ``n`` evenly spaced snapshot indices inside [t_lo, t_hi]."""
    in_window = np.where((t_grid >= t_lo) & (t_grid <= t_hi))[0]
    if len(in_window) == 0:
        return np.linspace(0, len(t_grid) - 1, n, dtype=int)
    if len(in_window) <= n:
        return in_window
    pick = np.linspace(0, len(in_window) - 1, n).round().astype(int)
    return in_window[pick]


# ============================================================
# HELPERS — Data loading and dataset splits
# ============================================================

def _load_plot_data(
    data_path: str | Path,
    x_grid_path: str | Path,
    y_grid_path: str | Path,
    t_grid_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load trajectories and saved grids for plot generation."""
    trajectories, x_grid, y_grid, t_grid = load_sim_data(
        sim_traj_path=str(Path(data_path)),
        x_grid_path=str(Path(x_grid_path)),
        y_grid_path=str(Path(y_grid_path)),
        t_grid_path=str(Path(t_grid_path)),
    )
    return trajectories, x_grid, y_grid, t_grid


def _resolve_plot_config(checkpoint_conf: dict | None = None) -> dict:
    """Prefer checkpoint config when available, otherwise load the default project config."""
    return checkpoint_conf if checkpoint_conf is not None else load_config()


def _load_checkpoint_model(checkpoint_path: str | Path) -> tuple[FNO2d, dict]:
    """Load a checkpoint and reconstruct a 2D FNO model on CPU."""
    import torch

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    conf = ckpt["conf"]
    model_cfg = conf.get("model", {}).get("parameters", {})
    if "modes1" not in model_cfg or "modes2" not in model_cfg:
        raise ValueError("Checkpoint is not a 2D FNO checkpoint: missing modes1/modes2")
    if model_cfg.get("in_channels", 8) != 8:
        raise ValueError("Checkpoint uses an incompatible spatial input; train a fresh 8-channel forcing-bin model.")

    model = FNO2d(
        modes1=model_cfg["modes1"],
        modes2=model_cfg["modes2"],
        width=model_cfg.get("width", 64),
        in_channels=model_cfg.get("in_channels", 8),
        out_channels=model_cfg.get("out_channels", 1),
        n_layers=model_cfg.get("n_layers", 4),
        cond_dim=model_cfg.get("cond_dim", COND_DIM),
        cond_hidden=model_cfg.get("cond_hidden", 256),
        dropout=model_cfg.get("dropout", 0.0),
        spectral_dropout=model_cfg.get("spectral_dropout", 0.0),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    model._mu_global = ckpt.get("mu_global")
    model._sigma_global = ckpt.get("sigma_global")
    return model, conf


def _build_split_datasets(
    trajectories: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    t_grid: np.ndarray,
    sim_params: np.ndarray,
    config: dict,
    dt: float | None = None,
) -> dict[str, SnapshotPairDataset]:
    """Build train/val/test snapshot-pair datasets that mirror training-time sampling.

    `dt` is the solver dt (saved alongside trajectories). Required for cond-vec
    consistency when save_stride > 1; falls back to t_grid spacing if None.
    """
    num_sims = trajectories.shape[0]
    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})
    train_ids, val_ids, test_ids = split_sim_ids(
        num_sims=num_sims,
        train_frac=data_cfg.get("train_split", 0.7),
        val_frac=data_cfg.get("val_split", 0.15),
        seed=0,
    )
    n_snapshots = training_cfg.get("n_snapshots", 15)
    n_snapshots_test = training_cfg.get("n_snapshots_test", None)
    test_snapshots = n_snapshots_test if n_snapshots_test is not None else n_snapshots
    mu_global, sigma_global = compute_global_stats(trajectories, train_ids)
    return {
        "train": SnapshotPairDataset(
            trajectories=trajectories,
            t_grid=t_grid,
            x_grid=x_grid,
            y_grid=y_grid,
            sim_ids=train_ids,
            sim_params=sim_params,
            mu_global=mu_global,
            sigma_global=sigma_global,
            n_snapshots=n_snapshots,
            dt=dt,
        ),
        "val": SnapshotPairDataset(
            trajectories=trajectories,
            t_grid=t_grid,
            x_grid=x_grid,
            y_grid=y_grid,
            sim_ids=val_ids,
            sim_params=sim_params,
            mu_global=mu_global,
            sigma_global=sigma_global,
            n_snapshots=n_snapshots,
            dt=dt,
        ),
        "test": SnapshotPairDataset(
            trajectories=trajectories,
            t_grid=t_grid,
            x_grid=x_grid,
            y_grid=y_grid,
            sim_ids=test_ids,
            sim_params=sim_params,
            mu_global=mu_global,
            sigma_global=sigma_global,
            n_snapshots=test_snapshots,
            dt=dt,
        ),
    }


def _prepare_prediction_case(
    model,
    trajectories: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    t_grid: np.ndarray,
    sim_params: np.ndarray,
    sim_id: int,
    s: int,
    target_indices: np.ndarray,
    config: dict | None = None,
    dt: float | None = None,
) -> dict[str, np.ndarray | float]:
    """Prepare truth/prediction arrays for one source snapshot and multiple target times."""
    import torch

    interface_meta = _resolve_interface_metadata(config=_resolve_plot_config(config))
    t_targets = t_grid[target_indices]

    # Use global normalization stats from model (attached during checkpoint loading)
    mu_global = getattr(model, "_mu_global", None)
    sigma_global = getattr(model, "_sigma_global", None)
    if mu_global is None or sigma_global is None:
        # Fallback: compute from training split
        from data.dataset import split_sim_ids as _split
        train_ids, _, _ = _split(trajectories.shape[0], 0.7, 0.15, seed=0)
        mu_global, sigma_global = compute_global_stats(trajectories, train_ids)

    T_source = trajectories[sim_id, s].astype(np.float32)
    T_source_norm = (T_source - mu_global) / (sigma_global + T_EPS)
    x_norm = ((x_grid - x_grid[0]) / (x_grid[-1] - x_grid[0])).astype(np.float32)
    y_norm = ((y_grid - y_grid[0]) / (y_grid[-1] - y_grid[0])).astype(np.float32)
    X_norm = np.broadcast_to(x_norm[:, None], T_source.shape).astype(np.float32)
    Y_norm = np.broadcast_to(y_norm[None, :], T_source.shape).astype(np.float32)

    params = sim_params[sim_id]
    amp, freq = _sin_amp_freq(params)
    R_c = float(params["R_c"])
    spatial_family = params["spatial_family"]
    spatial_params = params["spatial_params"]
    temporal_family = params["temporal_family"]
    temporal_params = params["temporal_params"]

    s_vec = np.asarray(SPATIAL_BUILDERS[spatial_family](y_grid, **spatial_params), dtype=np.float32)
    S_y = np.broadcast_to(np.asarray(s_vec, dtype=np.float32)[None, :], T_source.shape)
    x_spatial_base = np.stack([T_source_norm, X_norm, Y_norm, S_y], axis=-1)

    t_bars = t_grid[target_indices] - t_grid[s]
    t_s_norm = t_grid[s] / t_grid[-1]
    # dt is the solver dt (saved alongside trajectories); fall back to t_grid
    # spacing if not provided. Mismatch corrupts tau / dt_n cond slots.
    dt_grid = float(dt) if dt is not None else float(t_grid[1] - t_grid[0])
    t_final_grid = float(t_grid[-1])
    q_ref = np.float32(SIN_AMP_RANGE[1] * t_final_grid / FORCING_BINS)
    x_spatial_batch = []
    for target_idx in target_indices:
        bins = integrate_temporal_bins(
            temporal_family,
            temporal_params,
            float(t_grid[s]),
            float(t_grid[target_idx]),
            K=FORCING_BINS,
        ).astype(np.float32)
        Q_y_bins = (s_vec[None, :, None] * bins[None, None, :] / q_ref).astype(np.float32)
        Q_y_bins_2d = np.broadcast_to(Q_y_bins, T_source.shape + (FORCING_BINS,))
        x_spatial_batch.append(np.concatenate([x_spatial_base, Q_y_bins_2d], axis=-1))
    x_spatial_batch = np.stack(x_spatial_batch, axis=0).astype(np.float32)
    cond_rows = [
        build_cond_vector(
            t_bar_norm=float(t_bars[k]) / t_final_grid,
            t_s_norm=float(t_s_norm),
            R_c=R_c,
            spatial_family=spatial_family, spatial_params=spatial_params,
            temporal_family=temporal_family, temporal_params=temporal_params,
            dt=dt_grid, t_final=t_final_grid,
        )
        for k in range(len(target_indices))
    ]
    cond_batch = np.stack(cond_rows, axis=0).astype(np.float32)

    device = next(model.parameters()).device
    with torch.no_grad():
        x_tensor = torch.from_numpy(x_spatial_batch).to(device)
        c_tensor = torch.from_numpy(cond_batch).to(device)
        Y_pred_norm = model(x_tensor, c_tensor).cpu().numpy()[..., 0]

    Y_pred = (Y_pred_norm * (sigma_global + T_EPS) + mu_global).astype(np.float32)
    Y_true = trajectories[sim_id, target_indices].astype(np.float32)

    return {
        "interface_x": interface_meta["interface_x"],
        "interface_positions": interface_meta["positions"],
        "interface_half_width": interface_meta["interface_half_width"],
        "t_targets": t_targets,
        "t_bars": t_bars,
        "source_time": float(t_grid[s]),
        "Y_pred": Y_pred,
        "Y_true": Y_true,
        "amp": amp,
        "freq": freq,
        "R_c": R_c,
    }


def _compute_pair_error_records(
    model,
    dataset: SnapshotPairDataset,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    config: dict | None = None,
    max_samples: int = 256,
    batch_size: int = 64,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Evaluate sampled snapshot pairs and return per-pair error records."""
    import torch

    if len(dataset) == 0:
        raise ValueError("Dataset is empty; cannot compute error records.")

    rng = np.random.default_rng(seed)
    sample_size = min(max_samples, len(dataset))
    sample_indices = np.arange(len(dataset))
    if sample_size < len(dataset):
        sample_indices = np.sort(rng.choice(sample_indices, size=sample_size, replace=False))

    x_batch = []
    cond_batch = []
    y_batch = []
    stats_batch = []
    for idx in sample_indices:
        x_spatial, cond, Y, T_stats = dataset[idx]
        x_batch.append(x_spatial)
        cond_batch.append(cond)
        y_batch.append(Y)
        stats_batch.append(T_stats)

    x_tensor = torch.stack(x_batch, dim=0)
    cond_tensor = torch.stack(cond_batch, dim=0)
    y_tensor = torch.stack(y_batch, dim=0)
    stats_tensor = torch.stack(stats_batch, dim=0)

    interface_meta = _resolve_interface_metadata(config=_resolve_plot_config(config))
    interface_metric_mask = torch.from_numpy(
        _build_interface_mask_2d(x_grid, y_grid, interface_meta["interface_x"], interface_meta["interface_half_width"])
    )
    left_node, right_node = _interface_flanking_nodes_from_grid(x_grid, interface_meta["interface_x"])
    device = next(model.parameters()).device
    model.eval()

    preds = []
    for start in range(0, sample_size, batch_size):
        stop = min(start + batch_size, sample_size)
        with torch.no_grad():
            pred = model(
                x_tensor[start:stop].to(device),
                cond_tensor[start:stop].to(device),
            ).cpu()
        preds.append(pred)
    y_pred = torch.cat(preds, dim=0)

    mu_s = stats_tensor[:, 0][:, None, None, None]
    sigma_s = stats_tensor[:, 1][:, None, None, None]
    y_pred_phys = y_pred * (sigma_s + T_EPS) + mu_s
    y_true_phys = y_tensor * (sigma_s + T_EPS) + mu_s

    global_rel = (
        torch.mean((y_pred_phys - y_true_phys) ** 2, dim=(1, 2, 3))
        / torch.clamp(torch.mean(y_true_phys ** 2, dim=(1, 2, 3)), min=1e-12)
    ).sqrt() * 100.0
    y_pred_phys_field = y_pred_phys[..., 0]
    y_true_phys_field = y_true_phys[..., 0]
    iface_rel = (
        torch.mean((y_pred_phys_field[:, interface_metric_mask] - y_true_phys_field[:, interface_metric_mask]) ** 2, dim=1)
        / torch.clamp(torch.mean(y_true_phys_field[:, interface_metric_mask] ** 2, dim=1), min=1e-12)
    ).sqrt() * 100.0
    pred_jump = y_pred_phys_field[:, right_node, :] - y_pred_phys_field[:, left_node, :]
    true_jump = y_true_phys_field[:, right_node, :] - y_true_phys_field[:, left_node, :]
    jump_rel = (
        torch.mean((pred_jump - true_jump) ** 2, dim=1)
        / torch.clamp(torch.mean(true_jump ** 2, dim=1), min=1e-12)
    ).sqrt() * 100.0
    true_jump_rms = torch.sqrt(torch.mean(true_jump ** 2, dim=1))
    pred_jump_rms = torch.sqrt(torch.mean(pred_jump ** 2, dim=1))

    sim_ids = np.array([dataset._pairs[idx][0] for idx in sample_indices], dtype=np.int64)
    params = dataset.sim_params[sim_ids]
    source_indices = np.array([dataset._pairs[idx][1] for idx in sample_indices], dtype=np.int64)
    target_indices = np.array([dataset._pairs[idx][2] for idx in sample_indices], dtype=np.int64)

    return {
        "lead_time": np.array([dataset._lead_times[idx] for idx in sample_indices], dtype=np.float32),
        "global_rel_l2": global_rel.numpy(),
        "iface_rel_l2": iface_rel.numpy(),
        "jump_rel_l2": jump_rel.numpy(),
        "amplitude": np.array([_sin_amp_freq(p)[0] for p in params], dtype=np.float32),
        "frequency": np.array([_sin_amp_freq(p)[1] for p in params], dtype=np.float32),
        "R_c": np.array([float(p["R_c"]) for p in params], dtype=np.float32),
        "true_jump_rms": true_jump_rms.numpy(),
        "pred_jump_rms": pred_jump_rms.numpy(),
        "source_index": source_indices,
        "target_index": target_indices,
    }


def _compute_binned_means(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin x and return (centers, mean_y, counts) for non-empty bins."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.size == 0:
        return np.array([]), np.array([]), np.array([])

    x_min = float(x.min())
    x_max = float(x.max())
    if np.isclose(x_min, x_max):
        return np.array([x_min]), np.array([float(y.mean())]), np.array([len(x)])

    bins = np.linspace(x_min, x_max, n_bins + 1)
    bin_ids = np.clip(np.digitize(x, bins[1:-1], right=False), 0, n_bins - 1)

    centers = []
    means = []
    counts = []
    for idx in range(n_bins):
        mask = bin_ids == idx
        if not np.any(mask):
            continue
        centers.append(0.5 * (bins[idx] + bins[idx + 1]))
        means.append(float(y[mask].mean()))
        counts.append(int(mask.sum()))
    return np.array(centers), np.array(means), np.array(counts)


def _lead_time_coverage_counts(
    trajectories: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    t_grid: np.ndarray,
    sim_params: np.ndarray,
    config: dict,
) -> dict[str, np.ndarray]:
    """Return lead-time arrays and pair counts for split-aware coverage plots/tests."""
    datasets = _build_split_datasets(trajectories, x_grid, y_grid, t_grid, sim_params, config)
    return {
        "train": datasets["train"]._lead_times.copy(),
        "val": datasets["val"]._lead_times.copy(),
        "test": datasets["test"]._lead_times.copy(),
        "train_count": np.array([len(datasets["train"])]),
        "val_count": np.array([len(datasets["val"])]),
        "test_count": np.array([len(datasets["test"])]),
    }


# ============================================================
# PLOTS — Model evaluation
# ============================================================

def plot_prediction_vs_truth(
    model,
    trajectories: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    t_grid: np.ndarray,
    sim_params: np.ndarray,
    sim_id: int,
    s: int,
    n_steps: int = 40,
    config: dict | None = None,
    save_path: str | Path | None = None,
    dt: float | None = None,
):
    """Plot full 2D truth, prediction, residual, and jump profile diagnostics."""
    config = _resolve_plot_config(config)
    candidate_indices = _future_target_indices(s, n_steps, len(t_grid))
    if len(candidate_indices) > 3:
        pick = np.unique(np.round(np.linspace(0, len(candidate_indices) - 1, 3)).astype(int))
        target_indices = candidate_indices[pick]
    else:
        target_indices = candidate_indices
    case = _prepare_prediction_case(
        model,
        trajectories,
        x_grid,
        y_grid,
        t_grid,
        sim_params,
        sim_id,
        s,
        target_indices,
        config=config,
        dt=dt,
    )
    t_targets = case["t_targets"]
    Y_pred = case["Y_pred"]
    Y_true = case["Y_true"]
    vmin = min(Y_true.min(), Y_pred.min())
    vmax = max(Y_true.max(), Y_pred.max())
    interface_metric_mask = _build_interface_mask_2d(x_grid, y_grid, case["interface_x"], case["interface_half_width"])
    left_node, right_node = _interface_flanking_nodes_from_grid(x_grid, case["interface_x"])
    residual = Y_pred - Y_true
    true_jump = _interface_jump_map(Y_true, left_node, right_node)
    pred_jump = _interface_jump_map(Y_pred, left_node, right_node)
    residual_limit = max(float(np.max(np.abs(residual))), 1e-8)
    residual_norm = TwoSlopeNorm(vcenter=0.0, vmin=-residual_limit, vmax=residual_limit)

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(len(target_indices), 4, figsize=(18, 5.0 * len(target_indices)), squeeze=False, constrained_layout=True)
        truth_pred_axes = []
        resid_axes = []
        pcm_truth = None
        pcm_resid = None
        for row, t_target in enumerate(t_targets):
            true_field = Y_true[row]
            pred_field = Y_pred[row]
            resid_field = residual[row]
            true_jump_row = true_jump[row]
            pred_jump_row = pred_jump[row]
            global_rel = _relative_l2_percent(pred_field, true_field)
            iface_rel = _relative_l2_percent(pred_field[interface_metric_mask], true_field[interface_metric_mask])
            jump_rel = _relative_l2_percent(pred_jump_row, true_jump_row)

            ax_truth, ax_pred, ax_resid, ax_jump = axes[row]
            pcm_truth = _plot_field_2d(
                ax_truth,
                x_grid,
                y_grid,
                true_field,
                vmin=vmin,
                vmax=vmax,
                interface_positions=case["interface_positions"],
            )
            pcm_truth = _plot_field_2d(
                ax_pred,
                x_grid,
                y_grid,
                pred_field,
                vmin=vmin,
                vmax=vmax,
                interface_positions=case["interface_positions"],
            )
            pcm_resid = _plot_field_2d(
                ax_resid,
                x_grid,
                y_grid,
                resid_field,
                cmap="coolwarm",
                norm=residual_norm,
                interface_positions=case["interface_positions"],
            )
            ax_truth.set_title(f"Truth — t={t_target:.3f}")
            ax_pred.set_title("Prediction")
            ax_resid.set_title("Signed Residual")
            truth_pred_axes.extend([ax_truth, ax_pred])
            resid_axes.append(ax_resid)

            ax_resid.text(
                0.03,
                0.03,
                f"Global rel. L2: {global_rel:.2f}%\nIface rel. L2: {iface_rel:.2f}%\nJump rel. err.: {jump_rel:.2f}%",
                transform=ax_resid.transAxes,
                va="bottom",
                ha="left",
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.8"},
            )

            ax_jump.plot(y_grid, true_jump_row, color="black", label="Truth")
            ax_jump.plot(y_grid, pred_jump_row, color="C3", linestyle="--", label="Prediction")
            ax_jump.axhline(0.0, color="0.45", linestyle=":", linewidth=1.0)
            ax_jump.set_xlabel("y")
            ax_jump.set_ylabel("ΔT")
            ax_jump.set_title("Interface Jump Profile")
            ax_jump.grid(True)
            if row == 0:
                ax_jump.legend(loc="best")

        fig.colorbar(pcm_truth, ax=truth_pred_axes, label="Temperature", shrink=0.92)
        fig.colorbar(pcm_resid, ax=resid_axes, label="Pred - Truth", shrink=0.92)

        fig.suptitle(
            f"Prediction vs Truth — Sim {sim_id}, t_s={case['source_time']:.3f}, "
            f"A={case['amp']:.0f}, f={case['freq']:.2f}, R_c={case['R_c']:.2f}"
        )
        _save_figure(fig, save_path, "data", "prediction_vs_truth", layout="constrained")


def plot_trajectory_heatmap(
    trajectories: np.ndarray,
    sim_id: int,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    t_grid: np.ndarray,
    sim_params: np.ndarray | None = None,
    t_window: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
):
    """Plot six full-field temperature snapshots for one simulation.

    When ``t_window`` is given, five snapshots are drawn from that window plus
    one trailing post-window frame (useful for visualizing forcing during the
    flux-active window).
    """
    Nt = trajectories.shape[1]
    if t_window is not None:
        snap_main = _snap_indices_in_window(t_grid, t_window[0], t_window[1], 5)
        post = np.array([Nt - 1], dtype=int)
        snap_indices = np.unique(np.concatenate([snap_main, post]))
    else:
        snap_indices = np.linspace(0, Nt - 1, 6, dtype=int)
    snapshots = trajectories[sim_id, snap_indices]
    interface_meta = _resolve_interface_metadata()
    vmin = float(np.min(snapshots))
    vmax = float(np.max(snapshots))

    family_str = ""
    if sim_params is not None:
        family_str = f" — {_format_spatial_family(sim_params[sim_id])}"

    with plt.rc_context(PLOT_STYLE):
        nrows = 2
        ncols = int(np.ceil(len(snap_indices) / nrows))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.4 * nrows), constrained_layout=True, squeeze=False)
        axes_flat = axes.ravel()
        pcm = None
        for ax, t_idx, field in zip(axes_flat, snap_indices, snapshots):
            pcm = _plot_field_2d(
                ax,
                x_grid,
                y_grid,
                field,
                vmin=vmin,
                vmax=vmax,
                interface_positions=interface_meta["positions"],
            )
            ax.set_title(f"t = {t_grid[t_idx]:.3f}")
        for ax in axes_flat[len(snap_indices):]:
            ax.set_axis_off()

        fig.colorbar(pcm, ax=axes_flat.tolist(), label="Temperature", shrink=0.9)
        fig.suptitle(f"Temperature Snapshots — Simulation {sim_id}{family_str}")
        _save_figure(fig, save_path, "data", "trajectory_heatmap", layout="constrained")


def plot_trajectory_deviation_heatmap(
    trajectories: np.ndarray,
    sim_id: int,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    t_grid: np.ndarray,
    sim_params: np.ndarray | None = None,
    t_window: tuple[float, float] | None = None,
    T_right: float = 300.0,
    save_path: str | Path | None = None,
):
    """Plot the *deviation* T(x, y, t) - T_right for one simulation.

    Subtracting the right Dirichlet exposes the flux-induced 2D pattern, which
    is otherwise crushed by the dominant uniform background. Uses a diverging
    colormap so heating (>0) and cooling (<0) regions are visually distinct.
    """
    Nt = trajectories.shape[1]
    if t_window is not None:
        snap_main = _snap_indices_in_window(t_grid, t_window[0], t_window[1], 5)
        post = np.array([Nt - 1], dtype=int)
        snap_indices = np.unique(np.concatenate([snap_main, post]))
    else:
        snap_indices = np.linspace(0, Nt - 1, 6, dtype=int)
    snapshots = trajectories[sim_id, snap_indices].astype(np.float32) - float(T_right)
    interface_meta = _resolve_interface_metadata()

    # Symmetric range so 0 (no deviation) sits at the colormap midpoint.
    vmax = float(np.max(np.abs(snapshots)))
    vmax = max(vmax, 1e-6)
    vmin = -vmax

    family_str = ""
    if sim_params is not None:
        family_str = f" — {_format_spatial_family(sim_params[sim_id])}"

    with plt.rc_context(PLOT_STYLE):
        nrows = 2
        ncols = int(np.ceil(len(snap_indices) / nrows))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.4 * nrows),
                                 constrained_layout=True, squeeze=False)
        axes_flat = axes.ravel()
        pcm = None
        for ax, t_idx, field in zip(axes_flat, snap_indices, snapshots):
            pcm = _plot_field_2d(
                ax,
                x_grid,
                y_grid,
                field,
                cmap="RdBu_r",
                vmin=vmin,
                vmax=vmax,
                interface_positions=interface_meta["positions"],
            )
            ax.set_title(f"t = {t_grid[t_idx]:.3f}")
        for ax in axes_flat[len(snap_indices):]:
            ax.set_axis_off()

        fig.colorbar(pcm, ax=axes_flat.tolist(),
                     label=f"T - T_right  (T_right = {T_right:.0f} K)", shrink=0.9)
        fig.suptitle(f"Deviation from Right Dirichlet — Simulation {sim_id}{family_str}")
        _save_figure(fig, save_path, "data", "trajectory_deviation_heatmap",
                     layout="constrained")


def plot_y_perturbation(
    trajectories: np.ndarray,
    sim_id: int,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    t_grid: np.ndarray,
    sim_params: np.ndarray | None = None,
    t_window: tuple[float, float] = (0.0, 0.25),
    save_path: str | Path | None = None,
):
    """Plot the y-perturbation T(x, y, t) - <T>_y(x, t) for one simulation.

    Subtracting the y-mean exposes the y-structure imprinted by the spatial
    flux profile, regardless of the absolute temperature scale.
    """
    Nt = trajectories.shape[1]
    snap_main = _snap_indices_in_window(t_grid, t_window[0], t_window[1], 5)
    post = np.array([Nt - 1], dtype=int)
    snap_indices = np.unique(np.concatenate([snap_main, post]))

    fields = trajectories[sim_id, snap_indices].astype(np.float32)  # (S, Nx, Ny)
    perturbations = fields - fields.mean(axis=2, keepdims=True)
    abs_max = max(float(np.max(np.abs(perturbations))), 1e-12)
    interface_meta = _resolve_interface_metadata()

    family_str = ""
    if sim_params is not None:
        family_str = f" — {_format_spatial_family(sim_params[sim_id])}"

    with plt.rc_context(PLOT_STYLE):
        nrows = 2
        ncols = int(np.ceil(len(snap_indices) / nrows))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.4 * nrows), constrained_layout=True, squeeze=False)
        axes_flat = axes.ravel()
        pcm = None
        for ax, t_idx, field in zip(axes_flat, snap_indices, perturbations):
            pcm = _plot_field_2d(
                ax,
                x_grid,
                y_grid,
                field,
                cmap="coolwarm",
                vmin=-abs_max,
                vmax=abs_max,
                interface_positions=interface_meta["positions"],
            )
            ax.set_title(f"t = {t_grid[t_idx]:.3f}")
        for ax in axes_flat[len(snap_indices):]:
            ax.set_axis_off()

        fig.colorbar(pcm, ax=axes_flat.tolist(), label=r"$T - \langle T \rangle_y$", shrink=0.9)
        fig.suptitle(f"y-Perturbation — Simulation {sim_id}{family_str}")
        _save_figure(fig, save_path, "data", "y_perturbation", layout="constrained")


def plot_spatial_family_breakdown(
    trajectories: np.ndarray,
    sim_params: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    t_grid: np.ndarray,
    t_window: tuple[float, float] = (0.0, 0.20),
    n_snaps: int = 4,
    save_path: str | Path | None = None,
):
    """Tile y-perturbation snapshots for one representative sim per spatial family.

    Rows: spatial families (uniform, patch, gaussian, triangle).
    Cols: ``n_snaps`` snapshots in the flux-active window.
    Each panel shows T(x, y, t) - <T>_y(x, t) so the y-structure is visible.
    Each row uses its own symmetric colorbar.
    """
    families = ["uniform", "patch", "gaussian", "triangle"]
    selected_ids = []
    for fam in families:
        sid = _select_localized_sim_id(sim_params, family=fam)
        if sim_params[sid]["spatial_family"] == fam:
            selected_ids.append((fam, sid))

    if not selected_ids:
        raise ValueError("No simulations available for any spatial family.")

    snap_indices = _snap_indices_in_window(t_grid, t_window[0], t_window[1], n_snaps)
    interface_meta = _resolve_interface_metadata()
    nrows = len(selected_ids)
    ncols = len(snap_indices)

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(3.8 * ncols + 1.5, 3.6 * nrows),
            constrained_layout=True, squeeze=False,
        )

        for r, (fam, sid) in enumerate(selected_ids):
            fields = trajectories[sid, snap_indices].astype(np.float32)
            perturbations = fields - fields.mean(axis=2, keepdims=True)
            abs_max = max(float(np.max(np.abs(perturbations))), 1e-12)
            row_pcm = None
            for c, (t_idx, field) in enumerate(zip(snap_indices, perturbations)):
                ax = axes[r, c]
                row_pcm = _plot_field_2d(
                    ax,
                    x_grid,
                    y_grid,
                    field,
                    cmap="coolwarm",
                    vmin=-abs_max,
                    vmax=abs_max,
                    interface_positions=interface_meta["positions"],
                )
                if r == 0:
                    ax.set_title(f"t = {t_grid[t_idx]:.3f}")
                if c != 0:
                    ax.set_ylabel("")
            row_label = f"{fam}\nsim {sid}\n{_format_spatial_family(sim_params[sid])}"
            axes[r, 0].annotate(
                row_label,
                xy=(-0.30, 0.5), xycoords="axes fraction",
                ha="right", va="center", fontsize=10,
            )
            fig.colorbar(row_pcm, ax=axes[r, :].tolist(), label=r"$T - \langle T \rangle_y$", shrink=0.85)

        fig.suptitle("Spatial-Family Breakdown — y-perturbation per family", fontsize=14)
        _save_figure(fig, save_path, "data", "spatial_family_breakdown", layout="constrained")


def plot_initial_conditions(
    trajectories: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    n_samples: int = 20,
    seed: int = 42,
    save_path: str | Path | None = None,
):
    """Plot sampled 2D initial-condition fields."""
    rng = np.random.default_rng(seed)
    num_sims = trajectories.shape[0]
    selected = rng.choice(num_sims, size=min(n_samples, num_sims), replace=False)
    nrows, ncols = _shared_thumbnail_grid(len(selected))
    fields = trajectories[selected, 0]
    interface_meta = _resolve_interface_metadata()
    vmin = float(np.min(fields))
    vmax = float(np.max(fields))

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.8 * nrows), squeeze=False, constrained_layout=True)
        axes_flat = axes.ravel()
        pcm = None
        for ax, sim_id, field in zip(axes_flat, sorted(selected), trajectories[sorted(selected), 0]):
            pcm = _plot_field_2d(
                ax,
                x_grid,
                y_grid,
                field,
                vmin=vmin,
                vmax=vmax,
                interface_positions=interface_meta["positions"],
            )
            ax.set_title(f"Sim {sim_id}")

        for ax in axes_flat[len(selected):]:
            ax.set_axis_off()

        fig.colorbar(pcm, ax=axes.ravel().tolist(), label="Temperature", shrink=0.9)
        fig.suptitle(f"Initial Conditions T(x, y, t=0) — {len(selected)} simulations")
        _save_figure(fig, save_path, "data", "initial_conditions", layout="constrained")


def plot_lhs_scatter(
    sim_params: np.ndarray,
    save_path: str | Path | None = None,
):
    """Coverage summary for sampled conditioning parameters.

    Amplitude / frequency are NaN for non-sin sims; nan-aware ops keep the sin
    marginals visible while ignoring NaN entries (matplotlib drops them too).
    """
    amplitudes, frequencies, contact_resistance = _parameter_arrays(sim_params)

    def _normalize(values: np.ndarray, log: bool = False) -> np.ndarray:
        v = np.log10(values) if log else values
        lo = np.nanmin(v)
        hi = np.nanmax(v)
        return (v - lo) / max(hi - lo, 1e-12)

    normalized_marginals = [
        ("A",        _normalize(amplitudes),                "C0"),
        ("log10(f)", _normalize(frequencies, log=True),     "C3"),
        ("R_c",      _normalize(contact_resistance),        "C2"),
    ]

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
        scatter_specs = [
            (axes[0, 0], amplitudes, frequencies, "Flux Amplitude (A)", "Flux Frequency (f)"),
            (axes[0, 1], amplitudes, contact_resistance, "Flux Amplitude (A)", "Contact Resistance (R_c)"),
            (axes[1, 0], frequencies, contact_resistance, "Flux Frequency (f)", "Contact Resistance (R_c)"),
        ]
        for ax, x_vals, y_vals, x_label, y_label in scatter_specs:
            ax.scatter(x_vals, y_vals, s=16, alpha=0.45, color="C0", edgecolors="none")
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.grid(True)
            if "Frequency" in x_label:
                ax.set_xscale("log")
            if "Frequency" in y_label:
                ax.set_yscale("log")

        hist_ax = axes[1, 1]
        bins = np.linspace(0.0, 1.0, 12)
        for label, values, color in normalized_marginals:
            hist_ax.hist(values, bins=bins, histtype="step", linewidth=1.8, label=label, color=color)
        hist_ax.set_xlabel("Normalized parameter value")
        hist_ax.set_ylabel("Count")
        hist_ax.set_title("Marginal Coverage")
        hist_ax.legend(loc="upper center")
        hist_ax.grid(True)

        fig.suptitle(f"Conditioning Parameter Coverage — {len(amplitudes)} simulations")
        _save_figure(fig, save_path, "data", "lhs_scatter", layout="constrained")


def plot_flux_profiles(
    t_on: float = 0.0,
    t_off: float = 0.2,
    t_final: float = 1.0,
    save_path: str | Path | None = None,
):
    """Tile the new q_L(y, t) representation across patch / gaussian / triangle.

    Row 0 (full width): shared temporal forcing a(t).
    Rows 1-3: per family — s(y) | q_left(y, t) heatmap | q_x at t_peak.
    """
    layers = [
        Layer2D(x_left=0.0, x_right=0.5, rho=1.0, cp=1.0, k=2.0),
        Layer2D(x_left=0.5, x_right=1.0, rho=1.0, cp=1.0, k=1.0),
    ]
    families = [
        ("patch",    dict(y_c=0.50, w=0.30)),
        ("gaussian", dict(y_c=0.50, sigma_y=0.08)),
        ("triangle", dict(y_c=0.50, ell=0.20)),
    ]
    Nx, Ny = 60, 60
    a_x, b_x, c_y, d_y = 0.0, 1.0, 0.0, 1.0
    flux_A, flux_f, phase = 175.0, 10.5, 0.0
    y_grid = np.linspace(c_y, d_y, Ny)

    runs = []
    for name, sp in families:
        q_fn, s_vec = build_qL(
            "sin",
            dict(A=flux_A, f=flux_f, t_on=t_on, t_off=t_off, phase=phase),
            name,
            sp,
            y_grid,
        )
        solver = FVSolver2D(
            a=a_x, b=b_x, c=c_y, d=d_y,
            Nx=Nx, Ny=Ny,
            lam_target=0.8,
            layers=layers,
            interface_R=[0.5],
            t_final=t_final,
            flux_f=flux_f,
            flux_A=flux_A,
            t_on=t_on,
            t_off=t_off,
            phase=phase,
            dt=0.005,
            q_left_fn=q_fn,
        )
        _, _, _, T_hist = solver.solve(store_trajectory=True)
        Q = _evaluate_q_left_field(solver)
        t_peak = int(np.argmax(np.max(np.abs(Q), axis=1)))
        dT = T_hist[t_peak, 1:, :] - T_hist[t_peak, :-1, :]
        q_x = -solver.G_x * dT
        runs.append({
            "name": name, "sp": sp, "solver": solver, "T_hist": T_hist,
            "Q": Q, "s_vec": s_vec, "q_x": q_x, "t_peak": t_peak,
        })

    # Reconstruct a(t) from one run: q(y, t) = a(t) * s(y) → a(t) = q(y_peak, t)/s(y_peak).
    run0 = runs[0]
    j0 = int(np.argmax(np.abs(run0["s_vec"])))
    s0 = float(run0["s_vec"][j0])
    a_t_shared = run0["Q"][:, j0] / s0 if abs(s0) > 1e-12 else run0["Q"][:, j0]

    q_x_global = max(float(np.max([np.max(np.abs(r["q_x"])) for r in runs])), 1e-8)
    interface_meta = _resolve_interface_metadata(solver=runs[0]["solver"])

    with plt.rc_context(PLOT_STYLE):
        fig = plt.figure(figsize=(16, 13), constrained_layout=True)
        gs = fig.add_gridspec(4, 3, height_ratios=[0.7, 1.0, 1.0, 1.0])

        ax_at = fig.add_subplot(gs[0, :])
        solver0 = runs[0]["solver"]
        ax_at.plot(solver0.t, a_t_shared, color="C3")
        ax_at.axvline(t_on, color="C2", linestyle=":", alpha=0.8, label=f"t_on={t_on}")
        ax_at.axvline(t_off, color="C1", linestyle=":", alpha=0.8, label=f"t_off={t_off}")
        ax_at.set_xlabel("Time")
        ax_at.set_ylabel(r"$a(t)$")
        ax_at.set_title("Temporal forcing (shared across families)")
        ax_at.legend(loc="upper right")
        ax_at.grid(True)

        qx_axes = []
        pcm_x = None
        for row, run in enumerate(runs, start=1):
            solver = run["solver"]
            ax_s = fig.add_subplot(gs[row, 0])
            ax_q = fig.add_subplot(gs[row, 1])
            ax_qx = fig.add_subplot(gs[row, 2])

            ax_s.plot(solver.grid_y, run["s_vec"], color="C0")
            sp_str = ", ".join(f"{k}={v:.2f}" for k, v in run["sp"].items())
            ax_s.set_xlabel("y")
            ax_s.set_ylabel(r"$s(y)$")
            ax_s.set_title(f"{run['name']}: {sp_str}")
            ax_s.set_ylim(-0.1, 1.15)
            ax_s.grid(True)

            q_abs_row = max(float(np.max(np.abs(run["Q"]))), 1e-12)
            pcm_q = ax_q.pcolormesh(
                solver.t, solver.grid_y, run["Q"].T,
                cmap="coolwarm", vmin=-q_abs_row, vmax=q_abs_row, shading="auto",
            )
            ax_q.axvline(t_on, color="0.2", linestyle=":", alpha=0.6)
            ax_q.axvline(t_off, color="0.2", linestyle=":", alpha=0.6)
            ax_q.set_xlabel("Time")
            ax_q.set_ylabel("y")
            ax_q.set_title(rf"$q_{{left}}(y,t)$ — {run['name']}")
            fig.colorbar(pcm_q, ax=ax_q, shrink=0.9)

            pcm_x = ax_qx.pcolormesh(
                solver.face_positions_x, solver.grid_y, run["q_x"].T,
                cmap="coolwarm", vmin=-q_x_global, vmax=q_x_global, shading="nearest",
            )
            _add_interface_lines(ax_qx, interface_meta["positions"])
            ax_qx.set_xlabel("x-face position")
            ax_qx.set_ylabel("y")
            ax_qx.set_title(rf"$q_x$ at t={solver.t[run['t_peak']]:.3f}")
            qx_axes.append(ax_qx)

        fig.colorbar(pcm_x, ax=qx_axes, label=r"$q_x$", shrink=0.85)
        fig.suptitle("Boundary Flux Demo — Spatial Family Tiling")
        _save_figure(fig, save_path, "data", "flux_profiles", layout="constrained")


def plot_snapshot_pair_samples(
    trajectories: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    t_grid: np.ndarray,
    sim_params: np.ndarray,
    config: dict,
    n_samples: int = 3,
    seed: int = 42,
    save_path: str | Path | None = None,
):
    """Plot sampled 2D source/target snapshot pairs from train/val/test splits."""
    config = _resolve_plot_config(config)
    interface_meta = _resolve_interface_metadata(config=config)
    datasets = _build_split_datasets(trajectories, x_grid, y_grid, t_grid, sim_params, config)
    rng = np.random.default_rng(seed)
    splits = [("Train", datasets["train"]), ("Val", datasets["val"]), ("Test", datasets["test"])]

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(3, 2 * n_samples, figsize=(5.2 * n_samples, 11), squeeze=False, constrained_layout=True)

        for row, (split_name, dataset) in enumerate(splits):
            row_axes = axes[row]
            if len(dataset) == 0:
                for ax in row_axes:
                    ax.set_axis_off()
                continue

            sample_size = min(n_samples, len(dataset))
            chosen_indices = rng.choice(len(dataset), size=sample_size, replace=False)

            for col in range(n_samples):
                if col >= sample_size:
                    row_axes[2 * col].set_axis_off()
                    row_axes[2 * col + 1].set_axis_off()
                    continue
                pair_idx = int(chosen_indices[col])
                sim_id, s, j = dataset._pairs[pair_idx]
                lead_time = dataset._lead_times[pair_idx]
                source = trajectories[sim_id, s]
                target = trajectories[sim_id, j]
                params = sim_params[sim_id]
                R_c = float(params["R_c"])
                forcing_label = _forcing_label(params)
                vmin = float(min(np.min(source), np.min(target)))
                vmax = float(max(np.max(source), np.max(target)))
                ax_source = row_axes[2 * col]
                ax_target = row_axes[2 * col + 1]

                pcm = _plot_field_2d(
                    ax_source,
                    x_grid,
                    y_grid,
                    source,
                    vmin=vmin,
                    vmax=vmax,
                    interface_positions=interface_meta["positions"],
                )
                _plot_field_2d(
                    ax_target,
                    x_grid,
                    y_grid,
                    target,
                    vmin=vmin,
                    vmax=vmax,
                    interface_positions=interface_meta["positions"],
                )

                text = f"Δt={lead_time:.3f}\n{forcing_label}\nR_c={float(R_c):.2f}"
                ax_target.text(
                    0.03,
                    0.97,
                    text,
                    transform=ax_target.transAxes,
                    va="top",
                    ha="left",
                    bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.85"},
                )
                ax_source.set_title(f"{split_name} | Sim {sim_id} | source t={t_grid[s]:.3f}")
                ax_target.set_title(f"target t={t_grid[j]:.3f}")
                fig.colorbar(pcm, ax=[ax_source, ax_target], label="Temperature", shrink=0.82)

        fig.suptitle("Snapshot Pair Samples — All-to-All Conditioning View")
        _save_figure(fig, save_path, "data", "snapshot_pair_samples", layout="constrained")


def plot_interface_error(
    model,
    trajectories: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    t_grid: np.ndarray,
    sim_params: np.ndarray,
    sim_ids: np.ndarray,
    n_samples: int = 4,
    n_targets: int = 20,
    seed: int = 42,
    config: dict | None = None,
    save_path: str | Path | None = None,
    dt: float | None = None,
):
    """Diagnose interface-jump prediction accuracy over y and target time."""
    config = _resolve_plot_config(config)
    interface_meta = _resolve_interface_metadata(config=config)
    interface_x = interface_meta["interface_x"]
    left_node, right_node = _interface_flanking_nodes_from_grid(x_grid, interface_x)
    interface_metric_mask = _build_interface_mask_2d(x_grid, y_grid, interface_x, interface_meta["interface_half_width"])

    rng = np.random.default_rng(seed)
    chosen_sims = rng.choice(sim_ids, size=min(n_samples, len(sim_ids)), replace=False)
    cases = []
    for sim_id in chosen_sims:
        s = int(rng.integers(0, len(t_grid) - 1))
        target_indices = _future_target_indices(s, n_targets, len(t_grid))
        case = _prepare_prediction_case(
            model,
            trajectories,
            x_grid,
            y_grid,
            t_grid,
            sim_params,
            sim_id,
            s,
            target_indices,
            config=config,
            dt=dt,
        )
        cases.append(case)

    truth_limits = []
    residual_limits = []
    for case in cases:
        Y_true = case["Y_true"]
        Y_pred = case["Y_pred"]
        true_jump = _interface_jump_map(Y_true, left_node, right_node)
        pred_jump = _interface_jump_map(Y_pred, left_node, right_node)
        resid_jump = pred_jump - true_jump
        truth_limits.extend([float(np.min(true_jump)), float(np.max(true_jump)), float(np.min(pred_jump)), float(np.max(pred_jump))])
        residual_limits.extend([float(np.min(resid_jump)), float(np.max(resid_jump))])

    jump_abs = max(abs(min(truth_limits)), abs(max(truth_limits)), 1e-8)
    resid_abs = max(abs(min(residual_limits)), abs(max(residual_limits)), 1e-8)

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(len(cases), 3, figsize=(18, 4.1 * len(cases)), squeeze=False)
        for row_idx, case in enumerate(cases):
            ax_left, ax_mid, ax_right = axes[row_idx]
            t_targets = case["t_targets"]
            Y_true = case["Y_true"]
            Y_pred = case["Y_pred"]
            true_jump = _interface_jump_map(Y_true, left_node, right_node)
            pred_jump = _interface_jump_map(Y_pred, left_node, right_node)
            residual_jump = pred_jump - true_jump
            global_rel = _relative_l2_percent(Y_pred, Y_true)
            iface_rel = _relative_l2_percent(Y_pred[:, interface_metric_mask], Y_true[:, interface_metric_mask])
            jump_rel = _relative_l2_percent(pred_jump, true_jump)
            pcm_true = ax_left.pcolormesh(t_targets, y_grid, true_jump.T, cmap="coolwarm", vmin=-jump_abs, vmax=jump_abs, shading="auto")
            pcm_pred = ax_mid.pcolormesh(t_targets, y_grid, pred_jump.T, cmap="coolwarm", vmin=-jump_abs, vmax=jump_abs, shading="auto")
            pcm_resid = ax_right.pcolormesh(
                t_targets,
                y_grid,
                residual_jump.T,
                cmap="coolwarm",
                vmin=-resid_abs,
                vmax=resid_abs,
                shading="auto",
            )
            ax_left.set_xlabel("Time")
            ax_left.set_ylabel("y")
            ax_left.set_title("True Jump Map")
            ax_mid.set_xlabel("Time")
            ax_mid.set_ylabel("y")
            ax_mid.set_title("Predicted Jump Map")
            ax_right.set_xlabel("Time")
            ax_right.set_ylabel("y")
            ax_right.set_title("Jump Residual Map")

            ax_right.text(
                0.03,
                0.05,
                f"Sim {row_idx + 1}: A={case['amp']:.0f}, f={case['freq']:.2f}, R_c={case['R_c']:.2f}\n"
                f"Global rel. L2: {global_rel:.2f}%\nIface rel. L2: {iface_rel:.2f}%\nJump rel. err.: {jump_rel:.2f}%",
                transform=ax_right.transAxes,
                va="bottom",
                ha="left",
                    bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.8"},
            )
            fig.colorbar(pcm_true, ax=[ax_left, ax_mid], label="ΔT", shrink=0.82)
            fig.colorbar(pcm_resid, ax=ax_right, label="Pred - Truth", shrink=0.82)

        fig.suptitle(f"Interface Diagnostic — x = {interface_x:.3f}")
        _save_figure(fig, save_path, "data", "interface_error", layout="constrained")


def plot_lead_time_coverage(
    trajectories: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    t_grid: np.ndarray,
    sim_params: np.ndarray,
    config: dict,
    bins: int = 20,
    save_path: str | Path | None = None,
):
    """Visualize train/test pair coverage across lead times and curriculum cutoffs."""
    coverage = _lead_time_coverage_counts(trajectories, x_grid, y_grid, t_grid, sim_params, config)
    train_leads = coverage["train"]
    val_leads = coverage["val"]
    test_leads = coverage["test"]
    max_lead = max(
        float(train_leads.max()) if len(train_leads) > 0 else 0.0,
        float(val_leads.max()) if len(val_leads) > 0 else 0.0,
        float(test_leads.max()) if len(test_leads) > 0 else 0.0,
        float(t_grid[-1] - t_grid[0]),
    )
    hist_bins = np.linspace(0.0, max_lead, bins + 1)
    bin_centers = 0.5 * (hist_bins[:-1] + hist_bins[1:])
    warmup_epochs = int(config.get("training", {}).get("curriculum_warmup", 0))
    fractions = np.array([0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    cutoff_positions = fractions * max_lead

    with plt.rc_context(PLOT_STYLE):
        fig, (ax_hist, ax_frac) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        split_series = [
            ("Train", train_leads, "C0"),
            ("Val", val_leads, "C2"),
            ("Test", test_leads, "C3"),
        ]
        for label, values, color in split_series:
            counts, _ = np.histogram(values, bins=hist_bins)
            ax_hist.step(bin_centers, counts, where="mid", color=color, linewidth=1.8, label=f"{label} ({len(values)})")
            fractions_hist = counts / max(len(values), 1)
            ax_frac.step(bin_centers, fractions_hist, where="mid", color=color, linewidth=1.8, label=label)

        for frac, cutoff in zip(fractions, cutoff_positions):
            epoch_label = int(round(frac * warmup_epochs)) if warmup_epochs > 0 else 0
            label = "curriculum lead-time cutoff"
            if warmup_epochs > 0:
                label += f" (epoch {epoch_label})"
            ax_hist.axvline(cutoff, color="0.35", linestyle=":", alpha=0.8)
            ax_frac.axvline(cutoff, color="0.35", linestyle=":", alpha=0.8)
            ax_frac.text(cutoff, 1.02 * max(ax_frac.get_ylim()[1], 1.0), label, rotation=90, va="bottom", ha="center", fontsize=8)

        ax_hist.set_ylabel("Pair count")
        ax_hist.set_title("Lead-Time Coverage by Split")
        ax_hist.legend(loc="upper right")
        ax_hist.grid(True)

        ax_frac.set_xlabel("Lead time Δt")
        ax_frac.set_ylabel("Fraction of split")
        ax_frac.set_title("Normalized Coverage")
        ax_frac.legend(loc="upper right")
        ax_frac.grid(True)
        _save_figure(fig, save_path, "data", "lead_time_coverage", layout="constrained")


def plot_lead_time_error(
    model,
    dataset: SnapshotPairDataset,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    n_bins: int = 8,
    max_samples: int = 256,
    seed: int = 42,
    config: dict | None = None,
    save_path: str | Path | None = None,
):
    """Plot held-out error as a function of lead time using raw points, binned medians, and IQR bands."""
    records = _compute_pair_error_records(
        model=model,
        dataset=dataset,
        x_grid=x_grid,
        y_grid=y_grid,
        config=config,
        max_samples=max_samples,
        seed=seed,
    )
    centers, global_med, global_q25, global_q75, counts = _compute_binned_quantiles(
        records["lead_time"], records["global_rel_l2"], n_bins
    )
    _, iface_med, iface_q25, iface_q75, _ = _compute_binned_quantiles(
        records["lead_time"], records["iface_rel_l2"], n_bins
    )

    with plt.rc_context(PLOT_STYLE):
        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
        ax_top.scatter(records["lead_time"], records["global_rel_l2"], s=16, alpha=0.18, color="0.55", label="Raw global")
        ax_top.scatter(records["lead_time"], records["iface_rel_l2"], s=16, alpha=0.12, color="C3", label="Raw interface")
        ax_top.plot(centers, global_med, marker="o", color="C0", label="Binned median global")
        ax_top.fill_between(centers, global_q25, global_q75, color="C0", alpha=0.15, label="Global IQR")
        ax_top.plot(centers, iface_med, marker="s", color="C3", label="Binned median interface")
        ax_top.fill_between(centers, iface_q25, iface_q75, color="C3", alpha=0.12, label="Interface IQR")
        ax_top.set_ylabel("Error (%)")
        ax_top.set_title("Held-Out Error vs Lead Time")
        ax_top.legend(loc="upper left")
        ax_top.grid(True)

        width = 0.8 * (centers[1] - centers[0]) if len(centers) > 1 else 0.02
        ax_bot.bar(centers, counts, width=width, color="0.65")
        ax_bot.set_xlabel("Lead time Δt")
        ax_bot.set_ylabel("Samples")
        ax_bot.grid(True, axis="y")

        _save_figure(fig, save_path, "data", "lead_time_error")


def plot_parameter_error_slices(
    model,
    dataset: SnapshotPairDataset,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    max_samples: int = 256,
    n_bins: int = 6,
    seed: int = 42,
    config: dict | None = None,
    save_path: str | Path | None = None,
):
    """Plot held-out error against amplitude, frequency, and contact resistance using medians and IQR bands."""
    records = _compute_pair_error_records(
        model=model,
        dataset=dataset,
        x_grid=x_grid,
        y_grid=y_grid,
        config=config,
        max_samples=max_samples,
        seed=seed,
    )
    specs = [
        ("amplitude", "Flux Amplitude (A)", False),
        ("frequency", "Flux Frequency (f)", True),
        ("R_c", "Contact Resistance (R_c)", False),
    ]

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
        for ax, (key, x_label, use_log) in zip(axes, specs):
            x_vals = records[key]
            ax.scatter(x_vals, records["global_rel_l2"], s=14, alpha=0.18, color="0.55", label="Raw global")
            ax.scatter(x_vals, records["iface_rel_l2"], s=14, alpha=0.12, color="C3", label="Raw interface")
            centers, global_med, global_q25, global_q75, _ = _compute_binned_quantiles(x_vals, records["global_rel_l2"], n_bins)
            _, iface_med, iface_q25, iface_q75, _ = _compute_binned_quantiles(x_vals, records["iface_rel_l2"], n_bins)
            ax.plot(centers, global_med, color="C0", linewidth=1.8, marker="o", label="Binned median global")
            ax.fill_between(centers, global_q25, global_q75, color="C0", alpha=0.14, label="Global IQR")
            ax.plot(centers, iface_med, color="C3", linewidth=1.8, marker="s", label="Binned median interface")
            ax.fill_between(centers, iface_q25, iface_q75, color="C3", alpha=0.12, label="Interface IQR")
            ax.set_xlabel(x_label)
            ax.set_ylabel("Error (%)")
            ax.grid(True)
            if use_log:
                ax.set_xscale("log")

        axes[0].legend(loc="upper left")
        fig.suptitle("Held-Out Error Across Conditioning Dimensions")
        _save_figure(fig, save_path, "data", "parameter_error_slices")


def plot_dataset_summary(
    trajectories: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    t_grid: np.ndarray,
    sim_params: np.ndarray,
    config: dict,
    representative_sim_id: int | None = None,
    save_path: str | Path | None = None,
):
    """Report-style overview of the 2D dataset."""
    config = _resolve_plot_config(config)
    amplitudes, frequencies, contact_resistance = _parameter_arrays(sim_params)
    sim_id = _select_representative_sim_id(sim_params) if representative_sim_id is None else int(representative_sim_id)
    coverage = _lead_time_coverage_counts(trajectories, x_grid, y_grid, t_grid, sim_params, config)
    interface_meta = _resolve_interface_metadata(config=config)
    left_node, right_node = _interface_flanking_nodes_from_grid(x_grid, interface_meta["interface_x"])
    rc_order = np.argsort(contact_resistance)
    low_idx = int(rc_order[len(rc_order) // 4])
    high_idx = int(rc_order[(3 * len(rc_order)) // 4])
    late_idx = max(1, int(round(0.75 * (len(t_grid) - 1))))
    ic_std = np.std(trajectories[:, 0], axis=0)
    low_jump = _interface_jump_profile(trajectories[low_idx, late_idx], left_node, right_node)
    high_jump = _interface_jump_profile(trajectories[high_idx, late_idx], left_node, right_node)
    max_lead = max(
        float(coverage["train"].max()) if len(coverage["train"]) > 0 else 0.0,
        float(coverage["val"].max()) if len(coverage["val"]) > 0 else 0.0,
        float(coverage["test"].max()) if len(coverage["test"]) > 0 else 0.0,
        float(t_grid[-1] - t_grid[0]),
    )
    hist_bins = np.linspace(0.0, max_lead, 16)

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
        temp_fields = np.stack([trajectories[sim_id, 0], trajectories[sim_id, -1]])
        temp_vmin = float(np.min(temp_fields))
        temp_vmax = float(np.max(temp_fields))
        std_vmax = float(np.max(ic_std))

        pcm = _plot_field_2d(
            axes[0, 0],
            x_grid,
            y_grid,
            trajectories[sim_id, 0],
            vmin=temp_vmin,
            vmax=temp_vmax,
            interface_positions=interface_meta["positions"],
        )
        axes[0, 0].set_title(f"Representative Initial Field — Sim {sim_id}")

        _plot_field_2d(
            axes[0, 1],
            x_grid,
            y_grid,
            trajectories[sim_id, -1],
            vmin=temp_vmin,
            vmax=temp_vmax,
            interface_positions=interface_meta["positions"],
        )
        axes[0, 1].set_title(f"Representative Final Field — t={t_grid[-1]:.3f}")

        pcm_std = _plot_field_2d(
            axes[0, 2],
            x_grid,
            y_grid,
            ic_std,
            cmap="viridis",
            vmin=0.0,
            vmax=std_vmax,
            interface_positions=interface_meta["positions"],
        )
        axes[0, 2].set_title("Initial-Condition Std. Dev.")

        scatter = axes[1, 0].scatter(amplitudes, frequencies, c=contact_resistance, cmap="viridis", s=18, alpha=0.7, edgecolors="none")
        axes[1, 0].set_title("Parameter-Space Coverage")
        axes[1, 0].set_xlabel("Flux Amplitude (A)")
        axes[1, 0].set_ylabel("Flux Frequency (f)")
        axes[1, 0].set_yscale("log")
        axes[1, 0].grid(True)

        for label, values, color in [("Train", coverage["train"], "C0"), ("Val", coverage["val"], "C2"), ("Test", coverage["test"], "C3")]:
            axes[1, 1].hist(values, bins=hist_bins, histtype="step", linewidth=1.8, label=label, color=color)
        axes[1, 1].set_title("Lead-Time Coverage by Split")
        axes[1, 1].set_xlabel("Lead time Δt")
        axes[1, 1].set_ylabel("Pair count")
        axes[1, 1].legend(loc="upper right")
        axes[1, 1].grid(True)

        axes[1, 2].plot(y_grid, low_jump, color="C0", label=f"Low $R_c$={contact_resistance[low_idx]:.2f}")
        axes[1, 2].plot(y_grid, high_jump, color="C3", linestyle="--", label=f"High $R_c$={contact_resistance[high_idx]:.2f}")
        axes[1, 2].axhline(0.0, color="0.45", linestyle=":", linewidth=1.0)
        axes[1, 2].set_xlabel("y")
        axes[1, 2].set_ylabel("ΔT")
        axes[1, 2].set_title(f"Interface Jump Example — t={t_grid[late_idx]:.3f}")
        axes[1, 2].legend(loc="best")
        axes[1, 2].grid(True)

        fig.colorbar(pcm, ax=axes[0, :2].tolist(), label="Temperature", shrink=0.92)
        fig.colorbar(pcm_std, ax=axes[0, 2], label="Std. Dev.", shrink=0.92)
        fig.colorbar(scatter, ax=axes[1, 0], label="R_c", shrink=0.92)

        fig.suptitle("Dataset Summary")
        _save_figure(fig, save_path, "data", "dataset_summary", layout="constrained")


def plot_interface_jump_summary(
    model,
    dataset: SnapshotPairDataset,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    config: dict,
    max_samples: int = 256,
    seed: int = 42,
    save_path: str | Path | None = None,
):
    """Aggregate held-out interface-jump diagnostics over sampled test pairs."""
    config = _resolve_plot_config(config)
    records = _compute_pair_error_records(
        model=model,
        dataset=dataset,
        x_grid=x_grid,
        y_grid=y_grid,
        config=config,
        max_samples=max_samples,
        seed=seed,
    )

    lead_centers, lead_med, lead_q25, lead_q75, _ = _compute_binned_quantiles(records["lead_time"], records["jump_rel_l2"], 8)
    rc_centers, rc_med, rc_q25, rc_q75, _ = _compute_binned_quantiles(records["R_c"], records["jump_rel_l2"], 6)
    jump_rel_median = float(np.median(records["jump_rel_l2"]))
    jump_rel_p90 = float(np.percentile(records["jump_rel_l2"], 90))
    global_rel_median = float(np.median(records["global_rel_l2"]))
    iface_rel_median = float(np.median(records["iface_rel_l2"]))

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

        axes[0, 0].scatter(records["lead_time"], records["jump_rel_l2"], s=16, alpha=0.18, color="0.55")
        axes[0, 0].plot(lead_centers, lead_med, color="C0", marker="o", label="Binned median jump rel. err.")
        axes[0, 0].fill_between(lead_centers, lead_q25, lead_q75, color="C0", alpha=0.15, label="Jump rel. err. IQR")
        axes[0, 0].set_title("Jump Relative Error vs Lead Time")
        axes[0, 0].set_xlabel("Lead time Δt")
        axes[0, 0].set_ylabel("Jump relative error (%)")
        axes[0, 0].legend(loc="upper left")
        axes[0, 0].grid(True)

        axes[0, 1].scatter(records["R_c"], records["jump_rel_l2"], s=16, alpha=0.18, color="0.55")
        axes[0, 1].plot(rc_centers, rc_med, color="C3", marker="o", label="Binned median jump rel. err.")
        axes[0, 1].fill_between(rc_centers, rc_q25, rc_q75, color="C3", alpha=0.15, label="Jump rel. err. IQR")
        axes[0, 1].set_title("Jump Relative Error vs R_c")
        axes[0, 1].set_xlabel("Contact Resistance (R_c)")
        axes[0, 1].set_ylabel("Jump relative error (%)")
        axes[0, 1].legend(loc="upper left")
        axes[0, 1].grid(True)

        min_jump = min(float(np.min(records["true_jump_rms"])), float(np.min(records["pred_jump_rms"])))
        max_jump = max(float(np.max(records["true_jump_rms"])), float(np.max(records["pred_jump_rms"])))
        axes[1, 0].scatter(records["true_jump_rms"], records["pred_jump_rms"], s=18, alpha=0.25, color="C2")
        axes[1, 0].plot([min_jump, max_jump], [min_jump, max_jump], color="black", linestyle="--", label="Identity")
        axes[1, 0].set_title("Predicted vs True Jump RMS")
        axes[1, 0].set_xlabel("True jump RMS")
        axes[1, 0].set_ylabel("Predicted jump RMS")
        axes[1, 0].legend(loc="upper left")
        axes[1, 0].grid(True)

        axes[1, 1].axis("off")
        axes[1, 1].text(
            0.02,
            0.98,
            "Aggregate Metrics\n\n"
            f"Median jump rel. err.: {jump_rel_median:.2f}%\n"
            f"90th pct jump rel. err.: {jump_rel_p90:.2f}%\n"
            f"Median global rel. L2: {global_rel_median:.2f}%\n"
            f"Median interface rel. L2: {iface_rel_median:.2f}%",
            va="top",
            ha="left",
            fontsize=11,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f7f7f7", "edgecolor": "0.85"},
        )

        fig.suptitle("Interface Jump Summary")
        _save_figure(fig, save_path, "data", "interface_jump_summary", layout="constrained")
