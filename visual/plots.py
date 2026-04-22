from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from data.dataset import (
    AMP_RANGE,
    FREQ_RANGE,
    RC_RANGE,
    T_EPS,
    SnapshotPairDataset,
    compute_global_stats,
    load_sim_data,
    split_sim_ids,
)
from src.operators.train import load_config
from src.physics.fv_solver_2d import FVSolver2D, Layer2D
from src.operators.fno2d import FNO2d



# ============================================================
# REGISTRY — Plot names → group mapping + dispatch helpers
# ============================================================

PLOT_REGISTRY: dict[str, str] = {
    # group: physics
    "final_temperature":         "physics",
    "layer_geometry":            "physics",
    "face_conductance":          "physics",
    "multilayer_evolution":      "physics",
    "heat_flux_profile":         "physics",
    # group: mms
    "mms_convergence":           "mms",
    "mms_order_estimation":      "mms",
    # group: training
    "training_curves":           "training",
    "seed_comparison":           "training",
    # group: data
    "trajectory_heatmap":        "data",
    "initial_conditions":        "data",
    "lhs_scatter":               "data",
    "flux_profiles":             "data",
    "snapshot_pair_samples":     "data",
    "prediction_vs_truth":       "data",
    "interface_error":           "data",
    "lead_time_coverage":        "data",
    "lead_time_error":           "data",
    "parameter_error_slices":    "data",
    "dataset_summary":           "data",
    "interface_jump_summary":    "data",
    # group: sweep
    "sweep_ranking":             "sweep",
    "sweep_convergence":         "sweep",
    "sweep_hyperparams":         "sweep",
}

GROUPS = {"physics", "mms", "training", "data", "sweep"}

PLOT_STYLE = {
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "lines.linewidth": 1.6,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
}


def _ensure_parent(path: Path) -> Path:
    """Create parent directory if needed and return path unchanged."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _default_plot_path(group: str, name: str) -> Path:
    """Return the default output path for a named plot."""
    return Path(__file__).resolve().parent / group / f"{name}.png"


def _save_figure(
    fig,
    save_path: str | Path | None,
    group: str,
    name: str,
    layout: str = "tight",
) -> Path:
    """Finalize, save, and close a figure using a consistent output policy."""
    if layout == "tight":
        fig.tight_layout()
    elif layout == "constrained":
        pass
    elif layout == "none":
        pass
    else:
        raise ValueError(f"Unknown layout mode: {layout}")

    resolved = _ensure_parent(Path(save_path) if save_path is not None else _default_plot_path(group, name))
    fig.savefig(resolved, dpi=300)
    print(f"Saved {name} to: {resolved}")
    plt.close(fig)
    return resolved


def _should_run(name: str, groups: list[str], individual: list[str] | None) -> bool:
    """Check whether a named plot should be generated given the CLI flags."""
    if individual is not None:
        return name in individual
    if "all" in groups:
        return True
    return PLOT_REGISTRY.get(name) in groups


def _plots_for_group(group: str) -> list[str]:
    """Return plot names belonging to one registry group."""
    return [name for name, plot_group in PLOT_REGISTRY.items() if plot_group == group]


def _print_skip(name: str, reason: str) -> None:
    """Standardized skip message for CLI-driven plots."""
    print(f"Skipping {name} ({reason})")


def _add_interface_lines(ax, positions: list[float], axis: str = "x") -> None:
    """Draw interface markers on an axis."""
    draw = ax.axvline if axis == "x" else ax.axhline
    for position in positions:
        draw(position, color="0.35", linestyle=":", linewidth=1.2, alpha=0.9)


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


def _resolve_interface_metadata(config: dict | None = None, solver=None) -> dict:
    """Resolve interface metadata from solver geometry first, then config."""
    positions: list[float] = []
    if solver is not None and hasattr(solver, "interface_positions"):
        positions = [float(pos) for pos in solver.interface_positions]

    loss_cfg = {}
    if config is not None:
        loss_cfg = config.get("training", {}).get("loss", {})

    # Defaults match src/operators/{losses,train,eval}.py so older configs/checkpoints
    # that don't serialize these keys still plot instead of raising.
    interface_x = loss_cfg.get("interface_x", 0.5)
    interface_half_width = loss_cfg.get("interface_half_width", 0.05)

    if positions:
        resolved_x = float(interface_x) if interface_x is not None else positions[0]
    else:
        resolved_x = float(interface_x)
        positions = [resolved_x]

    return {
        "positions": positions,
        "interface_x": resolved_x,
        "interface_half_width": float(interface_half_width),
    }


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
    """Bin x and return centers, median, q25, q75, counts for non-empty bins."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

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
    """Return amplitude, frequency, and contact resistance arrays."""
    amplitudes = np.array([float(p[0]) for p in sim_params], dtype=np.float32)
    frequencies = np.array([float(p[1]) for p in sim_params], dtype=np.float32)
    contact_resistance = np.array([float(p[3]) for p in sim_params], dtype=np.float32)
    return amplitudes, frequencies, contact_resistance


def _select_representative_sim_id(sim_params: np.ndarray) -> int:
    """Choose a deterministic representative simulation near the median condition point."""
    amplitudes, frequencies, contact_resistance = _parameter_arrays(sim_params)
    stacked = np.column_stack(
        [
            (amplitudes - amplitudes.min()) / max(amplitudes.max() - amplitudes.min(), 1e-12),
            (np.log10(frequencies) - np.log10(frequencies).min())
            / max(np.log10(frequencies).max() - np.log10(frequencies).min(), 1e-12),
            (contact_resistance - contact_resistance.min())
            / max(contact_resistance.max() - contact_resistance.min(), 1e-12),
        ]
    )
    target = np.median(stacked, axis=0)
    return int(np.argmin(np.sum((stacked - target) ** 2, axis=1)))


# ============================================================
# HELPERS — Demo solver + interface utilities
# ============================================================

def create_demo_multilayer_solver() -> FVSolver2D:
    """Create a 2-layer 2D demo solver for physics visualisation plots.

    Matches the layer configuration in generate_dataset.py:
      Layer 1: [0.0, 0.5], rho=1.0, cp=1.0, k=2.0
      Layer 2: [0.5, 1.0], rho=1.0, cp=1.0, k=1.0
    Nx=100 (even) ensures x=0.5 lies on a cell face (required by the solver).
    Ny=100 matches for isotropic grid on [0,1]x[0,1].
    """
    layers = [
        Layer2D(x_left=0.0, x_right=0.5, rho=1.0, cp=1.0, k=2.0),
        Layer2D(x_left=0.5, x_right=1.0, rho=1.0, cp=1.0, k=1.0),
    ]
    return FVSolver2D(
        a=0.0, b=1.0, c=0.0, d=1.0,
        Nx=100, Ny=100,
        lam_target=0.8, layers=layers, interface_R=[0.5],
        t_final=0.3, flux_f=2.0, flux_A=50.0,
        t_on=0.0, t_off=0.2, phase=0.0, dt=0.005,
    )


def _interface_flanking_nodes(solver: FVSolver2D) -> list[tuple[int, int, float]]:
    """Return (left_node, right_node, interface_x) for each internal interface."""
    return [
        (f, f + 1, solver.face_positions_x[f])
        for f in sorted(solver.interface_face_map.keys())
    ]


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

    model = FNO2d(
        modes1=model_cfg["modes1"],
        modes2=model_cfg["modes2"],
        width=model_cfg.get("width", 64),
        in_channels=model_cfg.get("in_channels", 3),
        out_channels=model_cfg.get("out_channels", 1),
        n_layers=model_cfg.get("n_layers", 4),
        cond_dim=model_cfg.get("cond_dim", 5),
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
) -> dict[str, SnapshotPairDataset]:
    """Build train/val/test snapshot-pair datasets that mirror training-time sampling."""
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

    amp, freq, _T0, R_c = sim_params[sim_id]
    amp = float(amp)
    freq = float(freq)
    R_c = float(R_c)

    x_spatial_single = np.stack([T_source_norm, X_norm, Y_norm], axis=-1)
    x_spatial_batch = np.repeat(x_spatial_single[None, :, :, :], len(target_indices), axis=0)

    t_bars = t_grid[target_indices] - t_grid[s]
    t_s_norm = t_grid[s] / t_grid[-1]
    cond_batch = np.column_stack([
        t_bars / t_grid[-1],
        np.full(len(target_indices), t_s_norm),
        np.full(len(target_indices), (amp - AMP_RANGE[0]) / (AMP_RANGE[1] - AMP_RANGE[0])),
        np.full(len(target_indices), (freq - FREQ_RANGE[0]) / (FREQ_RANGE[1] - FREQ_RANGE[0])),
        np.full(len(target_indices), (R_c - RC_RANGE[0]) / (RC_RANGE[1] - RC_RANGE[0])),
    ]).astype(np.float32)

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
        "amplitude": np.array([float(p[0]) for p in params], dtype=np.float32),
        "frequency": np.array([float(p[1]) for p in params], dtype=np.float32),
        "R_c": np.array([float(p[3]) for p in params], dtype=np.float32),
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
# SECTION: PHYSICS — Multilayer FV Solver Diagnostics
# ============================================================

def plot_final_temperature(
    solver: FVSolver2D,
    save_path: Optional[Path] = None,
):
    """2D temperature field at final time as a pcolormesh heatmap."""
    _, _, _, T_final = solver.solve(store_trajectory=False)
    interface_meta = _resolve_interface_metadata(solver=solver)

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(8, 6))
        pc = ax.pcolormesh(solver.X, solver.Y, T_final, cmap="inferno", shading="auto")
        fig.colorbar(pc, ax=ax, label="Temperature")
        _add_interface_lines(ax, interface_meta["positions"])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Final Temperature at t = {solver.t[-1]:.3f}")
        ax.set_aspect("equal")
        _save_figure(fig, save_path, "physics", "final_temperature")


def plot_layer_geometry(
    solver: FVSolver2D,
    save_path: str | Path | None = None,
):
    """Layer geometry and material properties for a 2D multilayer domain.

    3-panel layout:
        (a) Domain schematic with colored layers, subsampled grid nodes, interface lines
        (b) Material properties k, rho, cp vs x (1D slice at mid-y)
        (c) Thermal diffusivity alpha = k/(rho*cp) vs x
    """
    interface_meta = _resolve_interface_metadata(solver=solver)

    with plt.rc_context(PLOT_STYLE):
        fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(14, 4.5))

        layer_colors = plt.cm.Pastel1(np.linspace(0, 1, max(len(solver.layers), 3)))

        for j, layer in enumerate(solver.layers):
            ax_a.axvspan(
                layer.x_left,
                layer.x_right,
                alpha=0.4,
                color=layer_colors[j],
                label=f"Layer {j + 1} (k={layer.k})",
            )
        step = max(1, solver.Nx // 20)
        ax_a.plot(
            solver.X[::step, ::step].ravel(),
            solver.Y[::step, ::step].ravel(),
            ".",
            color="black",
            markersize=2,
            alpha=0.35,
        )
        _add_interface_lines(ax_a, interface_meta["positions"])
        ax_a.set_xlabel("x")
        ax_a.set_ylabel("y")
        ax_a.set_title("Domain Schematic")
        ax_a.legend(loc="upper right")
        ax_a.set_xlim(solver.a, solver.b)
        ax_a.set_ylim(solver.c, solver.d)
        ax_a.set_aspect("equal")

        ax_b.plot(solver.grid_x, solver.k_nodes[:, 0], drawstyle="steps-mid", label="k")
        ax_b.plot(solver.grid_x, solver.rho_nodes[:, 0], drawstyle="steps-mid", label=r"$\rho$")
        ax_b.plot(solver.grid_x, solver.cp_nodes[:, 0], drawstyle="steps-mid", label=r"$c_p$")
        _add_interface_lines(ax_b, interface_meta["positions"])
        ax_b.set_xlabel("x")
        ax_b.set_ylabel("Property value")
        ax_b.set_title("Material Properties")
        ax_b.legend()
        ax_b.grid(True)

        alpha_nodes = solver.k_nodes[:, 0] / (solver.rho_nodes[:, 0] * solver.cp_nodes[:, 0])
        ax_c.plot(solver.grid_x, alpha_nodes, drawstyle="steps-mid", color="C2")
        _add_interface_lines(ax_c, interface_meta["positions"])
        ax_c.set_xlabel("x")
        ax_c.set_ylabel(r"$\alpha = k / (\rho \, c_p)$")
        ax_c.set_title("Thermal Diffusivity")
        ax_c.grid(True)

        _save_figure(fig, save_path, "physics", "layer_geometry")


def plot_face_conductance(
    solver: FVSolver2D,
    save_path: str | Path | None = None,
):
    """Face conductance and CN coefficients for the 2D solver.

    2x2 layout:
        (a) G_x heatmap (x-direction face conductance)
        (b) G_y heatmap (y-direction face conductance)
        (c) CN coefficients r_w, r_e at mid-y slice (1D view)
        (d) Diagonal dominance margin heatmap
    """
    interface_meta = _resolve_interface_metadata(solver=solver)

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
        ax_a, ax_b = axes[0]
        ax_c, ax_d = axes[1]

        face_x = solver.face_positions_x
        pc_a = ax_a.pcolormesh(face_x, solver.grid_y, solver.G_x.T, cmap="viridis", shading="nearest")
        fig.colorbar(pc_a, ax=ax_a, label=r"$G_x$ [W/(m$^2$K)]", shrink=0.85)
        _add_interface_lines(ax_a, interface_meta["positions"])
        ax_a.set_xlabel("x (face position)")
        ax_a.set_ylabel("y")
        ax_a.set_title("x-Face Conductance")

        face_y = (solver.grid_y[:-1] + solver.grid_y[1:]) / 2.0
        pc_b = ax_b.pcolormesh(solver.grid_x, face_y, solver.G_y.T, cmap="viridis", shading="nearest")
        fig.colorbar(pc_b, ax=ax_b, label=r"$G_y$ [W/(m$^2$K)]", shrink=0.85)
        ax_b.set_xlabel("x")
        ax_b.set_ylabel("y (face position)")
        ax_b.set_title("y-Face Conductance")

        j_mid = solver.Ny // 2
        interior_idx = np.arange(0, solver.Nx - 1)
        ax_c.plot(interior_idx, solver.r_w[:solver.Nx - 1, j_mid], color="C0", label=r"$r_w$")
        ax_c.plot(interior_idx, solver.r_e[:solver.Nx - 1, j_mid], color="C3", label=r"$r_e$")
        for xi in interface_meta["positions"]:
            node_approx = (xi - solver.a) / solver.hx
            ax_c.axvline(node_approx, color="0.35", linestyle=":", alpha=0.8)
        ax_c.set_xlabel("Node index i")
        ax_c.set_ylabel("CN coefficient")
        ax_c.set_title(f"CN Coefficients at j={j_mid}")
        ax_c.legend()
        ax_c.grid(True)

        margin = 1.0 - solver.r_w - solver.r_e - solver.r_s - solver.r_n
        margin_active = margin[:solver.Nx - 1, :]
        vmax = float(np.max(np.abs(margin_active)))
        pc_d = ax_d.pcolormesh(
            solver.grid_x[:solver.Nx - 1],
            solver.grid_y,
            margin_active.T,
            cmap="RdYlGn",
            vmin=-vmax,
            vmax=vmax,
            shading="nearest",
        )
        fig.colorbar(pc_d, ax=ax_d, label=r"$1 - r_w - r_e - r_s - r_n$", shrink=0.85)
        ax_d.set_xlabel("x")
        ax_d.set_ylabel("y")
        ax_d.set_title("Diagonal Dominance Margin")

        _save_figure(fig, save_path, "physics", "face_conductance", layout="constrained")


def plot_multilayer_evolution(
    solver: FVSolver2D,
    T_hist: np.ndarray,
    save_path: str | Path | None = None,
):
    """Temperature evolution as 2D snapshots with one shared temperature scale per figure."""
    Nt = len(solver.t)
    n_snaps = 6
    snap_indices = np.linspace(0, Nt - 1, n_snaps, dtype=int)
    snapshots = T_hist[snap_indices]
    vmin = float(np.min(snapshots))
    vmax = float(np.max(snapshots))
    interface_meta = _resolve_interface_metadata(solver=solver)

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
        pcm = None
        for ax, t_idx, snap in zip(axes.ravel(), snap_indices, snapshots):
            pcm = ax.pcolormesh(
                solver.X,
                solver.Y,
                snap,
                cmap="inferno",
                shading="auto",
                vmin=vmin,
                vmax=vmax,
            )
            _add_interface_lines(ax, interface_meta["positions"])
            ax.set_title(f"t = {solver.t[t_idx]:.3f}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_aspect("equal")

        fig.colorbar(pcm, ax=axes.ravel().tolist(), label="Temperature", shrink=0.9)
        fig.suptitle("Temperature Evolution")
        _save_figure(fig, save_path, "physics", "multilayer_evolution", layout="constrained")


def plot_heat_flux_profile(
    solver: FVSolver2D,
    T_hist: np.ndarray,
    save_path: str | Path | None = None,
):
    """Applied boundary flux and numerical x-direction flux at mid-y.

    2-panel layout:
        (a) Applied left boundary flux q_left(t) over full simulation time
        (b) Numerical x-flux at y=mid at selected time snapshots
    """
    Nt = len(solver.t)
    j_mid = solver.Ny // 2
    interface_meta = _resolve_interface_metadata(solver=solver)

    with plt.rc_context(PLOT_STYLE):
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 5))

        q_left_vals = np.array([solver.q_left(ti) for ti in solver.t])
        ax_a.plot(solver.t, q_left_vals, color="C3")
        ax_a.axvline(solver.t_on, color="C2", linestyle=":", alpha=0.8, label=f"t_on={solver.t_on}")
        ax_a.axvline(solver.t_off, color="C1", linestyle=":", alpha=0.8, label=f"t_off={solver.t_off}")
        ax_a.set_xlabel("Time")
        ax_a.set_ylabel("q_left(t)")
        ax_a.set_title("Applied Boundary Flux")
        ax_a.legend()
        ax_a.grid(True)

        snap_indices = np.linspace(1, Nt - 1, min(5, Nt - 1), dtype=int)
        cmap_snap = plt.cm.viridis
        for i, t_idx in enumerate(snap_indices):
            dT = T_hist[t_idx, 1:, j_mid] - T_hist[t_idx, :-1, j_mid]
            q_face = -solver.G_x[:, j_mid] * dT
            ax_b.plot(
                solver.face_positions_x,
                q_face,
                color=cmap_snap(i / max(len(snap_indices) - 1, 1)),
                label=f"t={solver.t[t_idx]:.3f}",
            )

        _add_interface_lines(ax_b, interface_meta["positions"])
        ax_b.set_xlabel("Face position (x)")
        ax_b.set_ylabel("Heat flux q")
        ax_b.set_title(f"Numerical x-Flux at y = {solver.grid_y[j_mid]:.2f}")
        ax_b.legend(loc="best")
        ax_b.grid(True)

        _save_figure(fig, save_path, "physics", "heat_flux_profile")


# ============================================================
# SECTION: MMS — Method of Manufactured Solutions (2D)
# ============================================================

def plot_mms_convergence(save_path: str | Path | None = None):
    """MMS convergence for all three 2D test cases in log-log space.

    2x3 layout:
        Top row: spatial convergence (varying N, fixed dt) for y-independent, full 2D, interface
        Bottom row: temporal convergence (varying dt, fixed N) for the same 3 cases
    """
    from src.physics.mms_2d import (
        run_mms_y_independent,
        run_mms_2d,
        run_mms_2d_interface,
    )

    # helper for log-log regression + plotting
    def _plot_convergence(ax, refinement, errors, xlabel, title):
        log_r = np.log10(refinement)
        log_e = np.log10(errors)
        coeffs = np.polyfit(log_r, log_e, 1)
        slope = coeffs[0]
        fit_line = 10 ** np.polyval(coeffs, log_r)

        ax.loglog(refinement, errors, "ko", markersize=7, label="MMS data")
        ax.loglog(refinement, fit_line, "C0-", linewidth=1.5, label=f"Fit: slope = {slope:.2f}")
        ref = errors[-1] * (refinement / refinement[-1]) ** 2
        ax.loglog(refinement, ref, "k--", alpha=0.4, linewidth=1, label="$O(h^2)$ reference")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("L2 error")
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(True, which="both", linestyle="--", alpha=0.3)

    cases = [
        ("y-independent", run_mms_y_independent),
        ("Full 2D", run_mms_2d),
        ("Interface", run_mms_2d_interface),
    ]

    # N lists: interface requires even N
    N_lists = {
        "y-independent": [21, 41, 81, 161],
        "Full 2D": [21, 41, 81, 161],
        "Interface": [50, 100, 200],
    }
    fixed_dt = 0.0001
    dt_list = [0.02, 0.01, 0.005]
    fixed_N_map = {
        "y-independent": 201,
        "Full 2D": 201,
        "Interface": 200,
    }

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))

        for col, (name, run_fn) in enumerate(cases):
            h_vals, h_l2 = [], []
            for N in N_lists[name]:
                h, _, _, l2 = run_fn(N, dt=fixed_dt)
                h_vals.append(h)
                h_l2.append(l2)
            _plot_convergence(axes[0, col], np.array(h_vals), np.array(h_l2),
                              r"$\Delta x$", f"{name} — Spatial (dt={fixed_dt})")

            dt_vals, dt_l2 = [], []
            fixed_N = fixed_N_map[name]
            for dt_val in dt_list:
                _, _, _, l2 = run_fn(fixed_N, dt=dt_val)
                dt_vals.append(dt_val)
                dt_l2.append(l2)
            _plot_convergence(axes[1, col], np.array(dt_vals), np.array(dt_l2),
                              r"$\Delta t$", f"{name} — Temporal (N={fixed_N})")

        fig.suptitle("2D MMS Convergence — Crank-Nicolson FV Solver")
        _save_figure(fig, save_path, "mms", "mms_convergence")


def plot_mms_order_estimation(save_path: str | Path | None = None):
    """Pairwise order estimates for all three 2D MMS cases.

    2x3 layout:
        Top row: estimated spatial order p_x vs dx for 3 cases
        Bottom row: estimated temporal order p_t vs dt for 3 cases
    """
    from src.physics.mms_2d import (
        run_mms_y_independent,
        run_mms_2d,
        run_mms_2d_interface,
    )

    cases = [
        ("y-independent", run_mms_y_independent),
        ("Full 2D", run_mms_2d),
        ("Interface", run_mms_2d_interface),
    ]

    N_lists = {
        "y-independent": [21, 41, 81, 161],
        "Full 2D": [21, 41, 81, 161],
        "Interface": [50, 100, 200],
    }
    fixed_dt = 0.0001
    dt_list = [0.02, 0.01, 0.005]
    fixed_N_map = {
        "y-independent": 201,
        "Full 2D": 201,
        "Interface": 200,
    }

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))

        for col, (name, run_fn) in enumerate(cases):
            h_vals, h_l2 = [], []
            for N in N_lists[name]:
                h, _, _, l2 = run_fn(N, dt=fixed_dt)
                h_vals.append(h)
                h_l2.append(l2)

            px_vals, px_mid = [], []
            for i in range(len(h_vals) - 1):
                p = np.log(h_l2[i] / h_l2[i + 1]) / np.log(h_vals[i] / h_vals[i + 1])
                px_vals.append(p)
                px_mid.append(np.sqrt(h_vals[i] * h_vals[i + 1]))

            ax = axes[0, col]
            ax.plot(px_mid, px_vals, "ko-", markersize=7)
            ax.axhline(2.0, color="C3", linestyle="--", alpha=0.6, label="Expected order = 2")
            ax.set_xscale("log")
            ax.set_xlabel(r"$\Delta x$ (geometric midpoint)")
            ax.set_ylabel("Estimated order $p_x$")
            ax.set_title(f"{name} — Spatial (dt={fixed_dt})")
            ax.legend(fontsize=7)
            ax.grid(True)
            ax.set_ylim(0, 4)

            fixed_N = fixed_N_map[name]
            dt_vals, dt_l2 = [], []
            for dt_val in dt_list:
                _, _, _, l2 = run_fn(fixed_N, dt=dt_val)
                dt_vals.append(dt_val)
                dt_l2.append(l2)

            pt_vals, pt_mid = [], []
            for i in range(len(dt_vals) - 1):
                p = np.log(dt_l2[i] / dt_l2[i + 1]) / np.log(dt_vals[i] / dt_vals[i + 1])
                pt_vals.append(p)
                pt_mid.append(np.sqrt(dt_vals[i] * dt_vals[i + 1]))

            ax = axes[1, col]
            ax.plot(pt_mid, pt_vals, "ko-", markersize=7)
            ax.axhline(2.0, color="C3", linestyle="--", alpha=0.6, label="Expected order = 2")
            ax.set_xscale("log")
            ax.set_xlabel(r"$\Delta t$ (geometric midpoint)")
            ax.set_ylabel("Estimated order $p_t$")
            ax.set_title(f"{name} — Temporal (N={fixed_N})")
            ax.legend(fontsize=7)
            ax.grid(True)
            ax.set_ylim(0, 4)

        fig.suptitle("2D MMS Order Estimation — Successive Refinement Pairs")
        _save_figure(fig, save_path, "mms", "mms_order_estimation")


# ============================================================
# SECTION: TRAINING — Neural Operator Training Metrics
# ============================================================

def plot_training_curves(csv_path: str | Path, save_path: str | Path | None = None):
    """Plot training loss, rel L2 metrics, interface metrics, and learning rate.

    Layout (3-row if interface columns present, 2-row otherwise):
        Top:    train_loss (log, left y) and train/val rel_l2 (log, right y), best checkpoint starred
        Middle: train/val interface rel_l2 (log y) — only when CSV has iface columns
        Bottom: learning rate vs epoch (log y)
    """
    import csv as csv_mod

    csv_path = Path(csv_path)
    epochs, train_losses, train_rel_l2s, val_rel_l2s, lrs, is_bests = [], [], [], [], [], []
    train_iface_rel_l2s, val_iface_rel_l2s = [], []

    has_train_rel_l2 = False
    has_iface = False
    with csv_path.open("r") as f:
        reader = csv_mod.DictReader(f)
        fieldnames = reader.fieldnames or []
        has_train_rel_l2 = "train_rel_l2" in fieldnames
        has_iface = "train_iface_rel_l2" in fieldnames
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_losses.append(float(row["train_loss"]))
            if has_train_rel_l2:
                train_rel_l2s.append(float(row["train_rel_l2"]))
            val_rel_l2s.append(float(row["val_rel_l2"]) if row["val_rel_l2"] != "" else None)
            lrs.append(float(row["lr"]))
            is_bests.append(int(row["is_best"]))
            if has_iface:
                train_iface_rel_l2s.append(float(row["train_iface_rel_l2"]))
                val_iface_rel_l2s.append(float(row["val_iface_rel_l2"]) if row["val_iface_rel_l2"] != "" else None)

    epochs = np.array(epochs)
    train_losses = np.array(train_losses)
    if has_train_rel_l2:
        train_rel_l2s = np.array(train_rel_l2s)
    if has_iface:
        train_iface_rel_l2s = np.array(train_iface_rel_l2s)
    lrs = np.array(lrs)

    # filter validation epochs
    val_epochs = np.array([e for e, v in zip(epochs, val_rel_l2s) if v is not None])
    val_values = np.array([v for v in val_rel_l2s if v is not None])

    if has_iface:
        val_iface_epochs = np.array([e for e, v in zip(epochs, val_iface_rel_l2s) if v is not None])
        val_iface_values = np.array([v for v in val_iface_rel_l2s if v is not None])

    # best checkpoint epochs
    best_epochs = np.array([e for e, b in zip(epochs, is_bests) if b == 1])
    best_vals = np.array([v for v, b in zip(val_rel_l2s, is_bests) if b == 1 and v is not None])

    if has_iface:
        best_iface_vals = np.array([v for v, b in zip(val_iface_rel_l2s, is_bests) if b == 1 and v is not None])

    best_epoch = int(best_epochs[-1]) if len(best_epochs) > 0 else None
    best_val = float(best_vals[-1]) if len(best_vals) > 0 else None
    nrows = 4 if has_iface else 3

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(nrows, 1, figsize=(10, 2.8 * nrows), sharex=True)
        axes = np.atleast_1d(axes)

        ax_loss = axes[0]
        ax_rel = axes[1]
        ax_lr = axes[-1]

        ax_loss.semilogy(epochs, train_losses, color="C0", label="Train MSE")
        ax_loss.set_ylabel("Train MSE")
        ax_loss.set_title("Training Loss")
        ax_loss.legend(loc="upper right")
        ax_loss.grid(True)

        if has_train_rel_l2:
            ax_rel.semilogy(epochs, train_rel_l2s, color="C0", alpha=0.4, label="Train rel. L2 (%)")
        ax_rel.semilogy(val_epochs, val_values, color="C3", label="Val rel. L2 (%)")
        if best_epoch is not None and best_val is not None:
            ax_rel.scatter([best_epoch], [best_val], color="gold", edgecolors="black", zorder=5, s=60)
            ax_rel.annotate(
                f"best @ epoch {best_epoch}\n{best_val:.3f}%",
                xy=(best_epoch, best_val),
                xytext=(6, 8),
                textcoords="offset points",
                fontsize=8,
            )
        ax_rel.set_ylabel("Rel. L2 (%)")
        ax_rel.set_title("Global Error")
        ax_rel.legend(loc="upper right")
        ax_rel.grid(True)

        if has_iface:
            ax_iface = axes[2]
            ax_iface.semilogy(epochs, train_iface_rel_l2s, color="C1", alpha=0.4, label="Train iface rel. L2 (%)")
            ax_iface.semilogy(val_iface_epochs, val_iface_values, color="C4", label="Val iface rel. L2 (%)")
            if best_epoch is not None and len(best_iface_vals) > 0:
                ax_iface.scatter([best_epoch], [best_iface_vals[-1]], color="gold", edgecolors="black", zorder=5, s=60)
            ax_iface.set_ylabel("Iface rel. L2 (%)")
            ax_iface.set_title("Interface Region Error")
            ax_iface.legend(loc="upper right")
            ax_iface.grid(True)

        ax_lr.semilogy(epochs, lrs, color="C2", label="Learning rate")
        ax_lr.set_xlabel("Epoch")
        ax_lr.set_ylabel("Learning rate")
        ax_lr.set_title("Learning Rate")
        ax_lr.grid(True)

        _save_figure(fig, save_path if save_path is not None else csv_path.parent / "training_curves.png", "training", "training_curves")


def plot_seed_comparison(report_path: str | Path, save_path: str | Path | None = None):
    """Plot grouped bar chart of val, test, and interface test metrics per seed."""
    import json

    report_path = Path(report_path)
    with report_path.open("r") as f:
        report = json.load(f)

    per_seed = report["per_seed"]
    summary = report["summary"]

    seeds = [str(r["seed"]) for r in per_seed]
    val_losses = [r["best_val"] for r in per_seed]
    test_losses = [r["test_rel_l2"] for r in per_seed]
    test_iface_losses = [r["test_iface_rel_l2"] for r in per_seed]

    x = np.arange(len(seeds), dtype=float)
    series = [
        ("Val rel. L2 (%)", np.array(val_losses, dtype=float), "C0", -0.18, summary["best_val_loss_mean"], summary["best_val_loss_std"]),
        ("Test rel. L2 (%)", np.array(test_losses, dtype=float), "C3", 0.0, summary["test_rel_l2_mean"], summary["test_rel_l2_std"]),
        ("Test iface rel. L2 (%)", np.array(test_iface_losses, dtype=float), "C1", 0.18, summary["test_iface_rel_l2_mean"], summary["test_iface_rel_l2_std"]),
    ]

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(10, 5.5))
        x_band = np.array([-0.5, len(seeds) - 0.5], dtype=float)
        for label, values, color, offset, mean_val, std_val in series:
            ax.fill_between(x_band, mean_val - std_val, mean_val + std_val, color=color, alpha=0.08)
            ax.axhline(mean_val, color=color, linestyle="--", alpha=0.6)
            ax.scatter(x + offset, values, color=color, s=50, label=f"{label} (mean {mean_val:.3f}%)")

        ax.set_xlabel("Seed")
        ax.set_ylabel("Relative L2 Error (%)")
        ax.set_title("Model Performance Across Seeds")
        ax.set_xticks(x)
        ax.set_xticklabels(seeds)
        ax.grid(True, axis="y")
        ax.legend(loc="upper right")

        _save_figure(fig, save_path if save_path is not None else report_path.parent / "seed_comparison.png", "training", "seed_comparison")


# ============================================================
# SECTION: DATA — Dataset & Parameter Space Visualization
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
    save_path: str | Path | None = None,
):
    """Plot six full-field temperature snapshots for one simulation."""
    snap_indices = np.linspace(0, trajectories.shape[1] - 1, 6, dtype=int)
    snapshots = trajectories[sim_id, snap_indices]
    interface_meta = _resolve_interface_metadata()
    vmin = float(np.min(snapshots))
    vmax = float(np.max(snapshots))

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)
        pcm = None
        for ax, t_idx, field in zip(axes.ravel(), snap_indices, snapshots):
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

        fig.colorbar(pcm, ax=axes.ravel().tolist(), label="Temperature", shrink=0.9)
        fig.suptitle(f"Temperature Snapshots — Simulation {sim_id}")
        _save_figure(fig, save_path, "data", "trajectory_heatmap", layout="constrained")


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
    """Coverage summary for sampled conditioning parameters."""
    amplitudes, frequencies, contact_resistance = _parameter_arrays(sim_params)
    normalized_marginals = [
        ("A", (amplitudes - amplitudes.min()) / max(amplitudes.max() - amplitudes.min(), 1e-12), "C0"),
        ("log10(f)", (np.log10(frequencies) - np.log10(frequencies).min()) / max(np.ptp(np.log10(frequencies)), 1e-12), "C3"),
        ("R_c", (contact_resistance - contact_resistance.min()) / max(contact_resistance.max() - contact_resistance.min(), 1e-12), "C2"),
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
    """Plot the applied boundary flux and 2D x-flux maps from a demo 2D solver."""
    layers = [
        Layer2D(x_left=0.0, x_right=0.5, rho=1.0, cp=1.0, k=2.0),
        Layer2D(x_left=0.5, x_right=1.0, rho=1.0, cp=1.0, k=1.0),
    ]
    solver = FVSolver2D(
        a=0.0,
        b=1.0,
        c=0.0,
        d=1.0,
        Nx=60,
        Ny=60,
        lam_target=0.8,
        layers=layers,
        interface_R=[0.5],
        t_final=t_final,
        flux_f=10.5,
        flux_A=175.0,
        t_on=t_on,
        t_off=t_off,
        phase=0.0,
        dt=0.005,
    )
    _, _, _, T_hist = solver.solve(store_trajectory=True)
    snap_indices = np.linspace(1, len(solver.t) - 1, 4, dtype=int)
    q_maps = []
    for t_idx in snap_indices:
        dT = T_hist[t_idx, 1:, :] - T_hist[t_idx, :-1, :]
        q_maps.append(-solver.G_x * dT)
    q_maps = np.asarray(q_maps)
    q_abs = max(float(np.max(np.abs(q_maps))), 1e-8)
    interface_meta = _resolve_interface_metadata(solver=solver)

    with plt.rc_context(PLOT_STYLE):
        fig = plt.figure(figsize=(16, 9), constrained_layout=True)
        gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1.0, 1.0])
        ax_flux = fig.add_subplot(gs[:, 0])
        axes_maps = [
            fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[0, 2]),
            fig.add_subplot(gs[1, 1]),
            fig.add_subplot(gs[1, 2]),
        ]

        q_left_vals = np.array([solver.q_left(ti) for ti in solver.t])
        ax_flux.plot(solver.t, q_left_vals, color="C3")
        ax_flux.axvline(solver.t_on, color="C2", linestyle=":", alpha=0.8, label=f"t_on={solver.t_on}")
        ax_flux.axvline(solver.t_off, color="C1", linestyle=":", alpha=0.8, label=f"t_off={solver.t_off}")
        ax_flux.set_xlabel("Time")
        ax_flux.set_ylabel(r"Applied flux $q_{left}(t)$")
        ax_flux.set_title("Applied Boundary Flux")
        ax_flux.legend(loc="upper right")
        ax_flux.grid(True)

        pcm = None
        for ax, t_idx, q_map in zip(axes_maps, snap_indices, q_maps):
            pcm = ax.pcolormesh(
                solver.face_positions_x,
                solver.grid_y,
                q_map.T,
                cmap="coolwarm",
                vmin=-q_abs,
                vmax=q_abs,
                shading="nearest",
            )
            _add_interface_lines(ax, interface_meta["positions"])
            ax.set_xlabel("x-face position")
            ax.set_ylabel("y")
            ax.set_title(f"$q_x$ at t={solver.t[t_idx]:.3f}")

        fig.colorbar(pcm, ax=axes_maps, label=r"$q_x$", shrink=0.9)
        fig.suptitle("2D x-Flux Diagnostic")
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
                amp, freq, _T0, R_c = sim_params[sim_id]
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

                text = f"Δt={lead_time:.3f}\nA={float(amp):.0f}\nf={float(freq):.2f}\nR_c={float(R_c):.2f}"
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


# ============================================================
# SECTION: SWEEP — Hyperparameter Sweep Visualization
# ============================================================

def _load_sweep_data(experiment_dir: Path) -> list[dict]:
    """Load hyperparameters and per-seed results for all configs in an experiment.

    Reads conf*.json (per-seed val results) and conf*.yaml (hyperparameters)
    from the experiment's generated config directory.

    Returns list of dicts sorted by objective (ascending = best first).
    """
    import json
    import yaml

    records = []
    for json_path in sorted(experiment_dir.glob("conf*.json")):
        config_name = json_path.stem  # e.g. "conf0"
        yaml_path = json_path.with_suffix(".yaml")

        with json_path.open("r") as f:
            jdata = json.load(f)

        if not yaml_path.exists():
            print(f"Warning: {yaml_path.name} not found, skipping {config_name}")
            continue

        with yaml_path.open("r") as f:
            ydata = yaml.safe_load(f)

        per_seed_vals = [s["best_val"] for s in jdata["per_seed"]]
        model_params = ydata.get("model", {}).get("parameters", {})
        training = ydata.get("training", {})

        records.append({
            "config_name": config_name,
            "objective": jdata["objective_mean_best_val"],
            "num_seeds": jdata["num_seeds"],
            "per_seed_vals": per_seed_vals,
            # Hyperparameters — handle both 1D (modes) and 2D (modes1) FNO:
            "modes": model_params.get("modes", model_params.get("modes1")),
            "width": model_params.get("width"),
            "batch_size": training.get("batch_size"),
            "learning_rate": training.get("learning_rate"),
            "weight_decay": training.get("weight_decay"),
        })

    records.sort(key=lambda r: r["objective"])
    return records


def _load_sweep_curves(
    runs_dir: Path,
    config_names: list[str],
    seed: int = 0,
) -> dict[str, dict]:
    """Load train_metrics.csv for one seed from each config's run directory."""
    import csv as csv_mod

    curves = {}
    for cname in config_names:
        csv_path = runs_dir / cname / f"seed{seed}" / "train_metrics.csv"
        if not csv_path.exists():
            continue

        epochs, train_losses, val_rel_l2s = [], [], []
        with csv_path.open("r") as f:
            reader = csv_mod.DictReader(f)
            for row in reader:
                epochs.append(int(row["epoch"]))
                train_losses.append(float(row["train_loss"]))
                val_str = row.get("val_rel_l2", "")
                val_rel_l2s.append(float(val_str) if val_str != "" else None)

        val_epochs = np.array([e for e, v in zip(epochs, val_rel_l2s) if v is not None])
        val_values = np.array([v for v in val_rel_l2s if v is not None])

        curves[cname] = {
            "epochs": np.array(epochs),
            "train_loss": np.array(train_losses),
            "val_epochs": val_epochs,
            "val_rel_l2": val_values,
        }

    return curves


def plot_sweep_ranking(
    experiment_dir: str | Path,
    save_path: str | Path | None = None,
):
    """Horizontal bar chart ranking configs by mean validation loss with seed-level detail."""
    experiment_dir = Path(experiment_dir)
    records = _load_sweep_data(experiment_dir)

    if not records:
        print("No configs found — skipping sweep_ranking.")
        return

    n = len(records)
    # records already sorted best-first; reverse for bottom-to-top bar ordering
    records_plot = list(reversed(records))

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(10, max(3, 0.8 * n + 1.5)))

        y_pos = np.arange(n)
        means = [r["objective"] for r in records_plot]
        stds = [float(np.std(r["per_seed_vals"])) for r in records_plot]
        labels = [r["config_name"] for r in records_plot]

        colors = ["#b0b0b0"] * n
        colors[-1] = "#4CAF50"

        ax.barh(y_pos, means, xerr=stds, height=0.6, color=colors,
                edgecolor="k", linewidth=0.5, capsize=3, ecolor="#555")

        for i, r in enumerate(records_plot):
            seed_vals = r["per_seed_vals"]
            ax.scatter(seed_vals, [i] * len(seed_vals), color="k", s=18, zorder=5, alpha=0.7)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel("Validation Rel. L2 (%)")
        ax.set_title("Hyperparameter Sweep — Config Ranking")

        obj_vals = [r["objective"] for r in records]
        if len(obj_vals) >= 2 and max(obj_vals) / max(min(obj_vals), 1e-12) > 10:
            ax.set_xscale("log")

        ax.grid(True, axis="x")
        _save_figure(fig, save_path if save_path is not None else experiment_dir / "sweep_ranking.png", "sweep", "sweep_ranking")


def plot_sweep_convergence(
    experiment_dir: str | Path,
    runs_dir: str | Path,
    seed: int = 0,
    save_path: str | Path | None = None,
):
    """Overlay validation and training loss curves for all configs from one seed."""
    experiment_dir = Path(experiment_dir)
    runs_dir = Path(runs_dir)
    records = _load_sweep_data(experiment_dir)

    if not records:
        print("No configs found — skipping sweep_convergence.")
        return

    config_names = [r["config_name"] for r in records]
    curves = _load_sweep_curves(runs_dir, config_names, seed=seed)

    if not curves:
        print(f"No train_metrics.csv found for seed {seed} — skipping sweep_convergence.")
        return

    # color map: one color per config, consistent with ranking order
    cmap = plt.cm.tab10
    color_map = {r["config_name"]: cmap(i % 10) for i, r in enumerate(records)}

    with plt.rc_context(PLOT_STYLE):
        fig, (ax_val, ax_train) = plt.subplots(1, 2, figsize=(14, 5))

        for cname in config_names:
            if cname not in curves:
                continue
            c = curves[cname]
            color = color_map[cname]

            if len(c["val_epochs"]) > 0:
                ax_val.semilogy(c["val_epochs"], c["val_rel_l2"], color=color, linewidth=1.2, label=cname)
                best_idx = np.argmin(c["val_rel_l2"])
                ax_val.scatter(c["val_epochs"][best_idx], c["val_rel_l2"][best_idx],
                               marker="*", s=80, color=color, edgecolors="k", linewidths=0.5, zorder=5)

            ax_train.semilogy(c["epochs"], c["train_loss"], color=color, linewidth=0.8, alpha=0.8, label=cname)

        ax_val.set_xlabel("Epoch")
        ax_val.set_ylabel("Val Rel. L2 (%)")
        ax_val.set_title(f"Validation Loss (seed {seed})")
        ax_val.legend(loc="upper right")
        ax_val.grid(True)

        ax_train.set_xlabel("Epoch")
        ax_train.set_ylabel("Train MSE")
        ax_train.set_title(f"Training Loss (seed {seed})")
        ax_train.legend(loc="upper right")
        ax_train.grid(True)

        _save_figure(fig, save_path if save_path is not None else experiment_dir / "sweep_convergence.png", "sweep", "sweep_convergence")


def plot_sweep_hyperparams(
    experiment_dir: str | Path,
    save_path: str | Path | None = None,
):
    """2x3 grid of scatter plots showing each hyperparameter vs performance."""
    experiment_dir = Path(experiment_dir)
    records = _load_sweep_data(experiment_dir)

    if len(records) < 2:
        print("Need at least 2 configs for hyperparameter plots — skipping.")
        return

    n = len(records)
    objectives = np.array([r["objective"] for r in records])

    # rank-based coloring: best=green, worst=red
    ranks = np.argsort(np.argsort(objectives))  # 0=best, n-1=worst
    norm = plt.Normalize(vmin=0, vmax=n - 1)
    cmap = plt.cm.RdYlGn_r  # green (low rank) -> red (high rank)
    colors = [cmap(norm(rank)) for rank in ranks]

    # hyperparameter panels
    hp_specs = [
        ("learning_rate", "Learning Rate", True),   # (key, label, log_x)
        ("weight_decay",  "Weight Decay",  True),
        ("batch_size",    "Batch Size",    False),
        ("modes",         "Modes",         False),
        ("width",         "Width",         False),
    ]

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        axes = axes.flatten()

        for idx, (key, label, log_x) in enumerate(hp_specs):
            ax = axes[idx]
            vals = [r[key] for r in records]

            if all(v is None for v in vals):
                ax.set_visible(False)
                continue

            vals = np.array([v if v is not None else np.nan for v in vals], dtype=float)
            ax.scatter(vals, objectives, c=colors, s=80, edgecolors="k", linewidths=0.5, zorder=3)
            ax.scatter(vals[0], objectives[0], s=200, facecolors="none", edgecolors="#4CAF50", linewidths=2.0, zorder=4)

            if log_x:
                ax.set_xscale("log")
            if objectives.max() / max(objectives.min(), 1e-12) > 10:
                ax.set_yscale("log")

            ax.set_xlabel(label)
            ax.set_ylabel("Val Rel. L2 (%)")
            ax.grid(True)

        ax = axes[5]
        modes_vals = np.array([r["modes"] if r["modes"] is not None else np.nan for r in records], dtype=float)
        width_vals = np.array([r["width"] if r["width"] is not None else np.nan for r in records], dtype=float)
        capacity = modes_vals * width_vals

        if not np.all(np.isnan(capacity)):
            ax.scatter(capacity, objectives, c=colors, s=80, edgecolors="k", linewidths=0.5, zorder=3)
            ax.scatter(capacity[0], objectives[0], s=200, facecolors="none", edgecolors="#4CAF50", linewidths=2.0, zorder=4)
            if objectives.max() / max(objectives.min(), 1e-12) > 10:
                ax.set_yscale("log")
            ax.set_xlabel("Modes × Width")
            ax.set_ylabel("Val Rel. L2 (%)")
            ax.grid(True)
        else:
            ax.set_visible(False)

        fig.suptitle("Hyperparameter Sensitivity")
        _save_figure(fig, save_path if save_path is not None else experiment_dir / "sweep_hyperparams.png", "sweep", "sweep_hyperparams")


# ============================================================
# CLI — Group-based dispatch
# ============================================================

if __name__ == "__main__":

    """
    Usage:
      python -m visual.plots --out visual/                          # all plots (default)
      python -m visual.plots --group physics --out visual/          # physics diagnostics only
      python -m visual.plots --group mms --out visual/              # MMS convergence only
      python -m visual.plots --group training --csv <path> --out visual/
      python -m visual.plots --group data --data <path> --x-grid <path> --y-grid <path> --t-grid <path> --params <path> --out visual/
      python -m visual.plots --group sweep --experiment <path> --runs <path> --out visual/
      python -m visual.plots --plots prediction_vs_truth lead_time_error --checkpoint <path> --out visual/

    Optional data flags:
      --data        Path to trajectories .npy file
      --x-grid      Path to x_grid .npy file
      --y-grid      Path to y_grid .npy file
      --t-grid      Path to t_grid .npy file
      --params      Path to sim_params .npy file
      --csv         Path to train_metrics.csv
      --report      Path to seed_report.json
      --checkpoint  Path to model checkpoint .pt (for model diagnostic plots)
      --experiment  Path to conf/generated/experiment{N}/ directory (sweep plots)
      --runs        Path to runs/experiment{N}/ directory (sweep convergence)
      --sweep-seed  Which seed to show in convergence plot (default: 0)
    """

    import argparse

    parser = argparse.ArgumentParser(description="Generate plots for the ITR project")
    parser.add_argument("--data", type=str, default=None, help="Path to trajectories .npy file")
    parser.add_argument("--x-grid", type=str, default=None, help="Path to x_grid .npy file")
    parser.add_argument("--y-grid", type=str, default=None, help="Path to y_grid .npy file")
    parser.add_argument("--t-grid", type=str, default=None, help="Path to t_grid .npy file")
    parser.add_argument("--params", type=str, default=None, help="Path to sim_params .npy file")
    parser.add_argument("--csv", type=str, default=None, help="Path to train_metrics.csv")
    parser.add_argument("--report", type=str, default=None, help="Path to seed_report.json")
    parser.add_argument("--experiment", type=str, default=None,
                        help="Path to conf/generated/experiment{N}/ directory (sweep plots)")
    parser.add_argument("--runs", type=str, default=None,
                        help="Path to runs/experiment{N}/ directory (sweep convergence)")
    parser.add_argument("--sweep-seed", type=int, default=0,
                        help="Which seed to show in convergence plot (default: 0)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (for interface_error plot)")
    parser.add_argument("--out", type=str, default=None, help="Output directory for plots")
    parser.add_argument("--group", type=str, nargs="+", default=["all"],
                        choices=["all", "physics", "mms", "training", "data", "sweep"],
                        help="Which plot group(s) to generate (default: all)")
    parser.add_argument("--plots", type=str, nargs="+", default=None,
                        help="Individual plot names to generate (overrides --group)")
    args = parser.parse_args()

    out_dir = Path(args.out).resolve() if args.out else Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = args.group
    individual = args.plots

    # ---- PHYSICS GROUP ----
    physics_plots = _plots_for_group("physics")
    need_physics = any(_should_run(p, groups, individual) for p in physics_plots)

    if need_physics:
        print("=== PHYSICS GROUP ===")
        solver = create_demo_multilayer_solver()
        t_sol, gx_sol, gy_sol, T_hist = solver.solve(store_trajectory=True)

        physics_dir = out_dir / "physics"

        if _should_run("final_temperature", groups, individual):
            print("--- final_temperature ---")
            plot_final_temperature(solver, save_path=physics_dir / "final_temperature.png")

        if _should_run("layer_geometry", groups, individual):
            print("--- layer_geometry ---")
            plot_layer_geometry(solver, save_path=physics_dir / "layer_geometry.png")

        if _should_run("face_conductance", groups, individual):
            print("--- face_conductance ---")
            plot_face_conductance(solver, save_path=physics_dir / "face_conductance.png")

        if _should_run("multilayer_evolution", groups, individual):
            print("--- multilayer_evolution ---")
            plot_multilayer_evolution(solver, T_hist, save_path=physics_dir / "multilayer_evolution.png")

        if _should_run("heat_flux_profile", groups, individual):
            print("--- heat_flux_profile ---")
            plot_heat_flux_profile(solver, T_hist, save_path=physics_dir / "heat_flux_profile.png")

    # ---- MMS GROUP ----
    mms_plots = _plots_for_group("mms")
    if any(_should_run(p, groups, individual) for p in mms_plots):
        print("=== MMS GROUP ===")

        mms_dir = out_dir / "mms"

        if _should_run("mms_convergence", groups, individual):
            print("--- mms_convergence ---")
            plot_mms_convergence(save_path=mms_dir / "mms_convergence.png")

        if _should_run("mms_order_estimation", groups, individual):
            print("--- mms_order_estimation ---")
            plot_mms_order_estimation(save_path=mms_dir / "mms_order_estimation.png")

    # ---- TRAINING GROUP ----
    training_plots = _plots_for_group("training")
    if any(_should_run(p, groups, individual) for p in training_plots):
        print("=== TRAINING GROUP ===")

        training_dir = out_dir / "training"

        if _should_run("training_curves", groups, individual):
            if args.csv:
                print("--- training_curves ---")
                plot_training_curves(args.csv, save_path=training_dir / "training_curves.png")
            else:
                _print_skip("training_curves", "need --csv")

        if _should_run("seed_comparison", groups, individual):
            if args.report:
                print("--- seed_comparison ---")
                plot_seed_comparison(args.report, save_path=training_dir / "seed_comparison.png")
            else:
                _print_skip("seed_comparison", "need --report")

    # ---- DATA GROUP ----
    data_plots = _plots_for_group("data")
    need_data = any(_should_run(p, groups, individual) for p in data_plots)

    if need_data:
        print("=== DATA GROUP ===")

        data_dir = out_dir / "data"
        checkpoint_conf = None
        model = None

        # flux_profiles needs no data files
        if _should_run("flux_profiles", groups, individual):
            print("--- flux_profiles ---")
            plot_flux_profiles(save_path=data_dir / "flux_profiles.png")

        sim_params = None
        if args.params:
            sim_params = np.load(args.params, allow_pickle=True)

        grid_data_plots = {
            "trajectory_heatmap",
            "initial_conditions",
            "snapshot_pair_samples",
            "prediction_vs_truth",
            "interface_error",
            "lead_time_coverage",
            "lead_time_error",
            "parameter_error_slices",
            "dataset_summary",
            "interface_jump_summary",
        }
        needs_grid_data = any(_should_run(name, groups, individual) for name in grid_data_plots)
        trajectories = None
        x_grid = None
        y_grid = None
        t_grid = None
        plot_config = None
        split_datasets = None

        if needs_grid_data:
            if args.data and args.x_grid and args.y_grid and args.t_grid:
                print(f"Loading trajectories and grids from {args.data}, {args.x_grid}, {args.y_grid}, {args.t_grid} ...")
                trajectories, x_grid, y_grid, t_grid = _load_plot_data(args.data, args.x_grid, args.y_grid, args.t_grid)
            else:
                for name in grid_data_plots:
                    if _should_run(name, groups, individual):
                        _print_skip(name, "need --data, --x-grid, --y-grid, and --t-grid")

        model_plots = {
            "prediction_vs_truth",
            "interface_error",
            "lead_time_error",
            "parameter_error_slices",
            "interface_jump_summary",
        }
        needs_model = any(_should_run(name, groups, individual) for name in model_plots)
        if needs_model and args.checkpoint:
            try:
                model, checkpoint_conf = _load_checkpoint_model(args.checkpoint)
            except ValueError as exc:
                for name in model_plots:
                    if _should_run(name, groups, individual):
                        _print_skip(name, str(exc))
        elif needs_model:
            for name in model_plots:
                if _should_run(name, groups, individual):
                    _print_skip(name, "need --checkpoint")

        if trajectories is not None and sim_params is not None:
            plot_config = _resolve_plot_config(checkpoint_conf)
            split_datasets = _build_split_datasets(trajectories, x_grid, y_grid, t_grid, sim_params, plot_config)

        if sim_params is not None and _should_run("lhs_scatter", groups, individual):
            print("--- lhs_scatter ---")
            plot_lhs_scatter(sim_params, save_path=data_dir / "lhs_scatter.png")

        if trajectories is not None:
            if _should_run("trajectory_heatmap", groups, individual):
                print("--- trajectory_heatmap ---")
                plot_trajectory_heatmap(trajectories, sim_id=0, x_grid=x_grid, y_grid=y_grid, t_grid=t_grid,
                                        save_path=data_dir / "trajectory_heatmap.png")

            if _should_run("initial_conditions", groups, individual):
                print("--- initial_conditions ---")
                plot_initial_conditions(trajectories, x_grid=x_grid, y_grid=y_grid,
                                        save_path=data_dir / "initial_conditions.png")

            # plots requiring sim_params
            if sim_params is not None:
                if split_datasets is not None and _should_run("dataset_summary", groups, individual):
                    print("--- dataset_summary ---")
                    plot_dataset_summary(
                        trajectories,
                        x_grid,
                        y_grid,
                        t_grid,
                        sim_params,
                        config=plot_config,
                        save_path=data_dir / "dataset_summary.png",
                    )

                if split_datasets is not None and _should_run("snapshot_pair_samples", groups, individual):
                    print("--- snapshot_pair_samples ---")
                    plot_snapshot_pair_samples(
                        trajectories,
                        x_grid,
                        y_grid,
                        t_grid,
                        sim_params,
                        config=plot_config,
                        save_path=data_dir / "snapshot_pair_samples.png",
                    )

                if split_datasets is not None and _should_run("lead_time_coverage", groups, individual):
                    print("--- lead_time_coverage ---")
                    plot_lead_time_coverage(
                        trajectories,
                        x_grid,
                        y_grid,
                        t_grid,
                        sim_params,
                        config=plot_config,
                        save_path=data_dir / "lead_time_coverage.png",
                    )

                if model is not None and split_datasets is not None:
                    test_dataset = split_datasets["test"]
                    if _should_run("prediction_vs_truth", groups, individual):
                        print("--- prediction_vs_truth ---")
                        representative_idx = min(len(test_dataset) // 2, len(test_dataset) - 1)
                        sim_id, s, _ = test_dataset._pairs[representative_idx]
                        n_steps = max(2, min(40, len(t_grid) - s - 1))
                        plot_prediction_vs_truth(
                            model,
                            trajectories,
                            x_grid,
                            y_grid,
                            t_grid,
                            sim_params,
                            sim_id=sim_id,
                            s=s,
                            n_steps=n_steps,
                            config=plot_config,
                            save_path=data_dir / "prediction_vs_truth.png",
                        )

                    if _should_run("interface_error", groups, individual):
                        print("--- interface_error ---")
                        plot_interface_error(
                            model,
                            trajectories,
                            x_grid,
                            y_grid,
                            t_grid,
                            sim_params,
                            test_dataset.sim_ids,
                            config=plot_config,
                            save_path=data_dir / "interface_error.png",
                        )

                    if _should_run("lead_time_error", groups, individual):
                        print("--- lead_time_error ---")
                        plot_lead_time_error(
                            model,
                            test_dataset,
                            x_grid,
                            y_grid,
                            config=plot_config,
                            save_path=data_dir / "lead_time_error.png",
                        )

                    if _should_run("parameter_error_slices", groups, individual):
                        print("--- parameter_error_slices ---")
                        plot_parameter_error_slices(
                            model,
                            test_dataset,
                            x_grid,
                            y_grid,
                            config=plot_config,
                            save_path=data_dir / "parameter_error_slices.png",
                        )

                    if _should_run("interface_jump_summary", groups, individual):
                        print("--- interface_jump_summary ---")
                        plot_interface_jump_summary(
                            model,
                            test_dataset,
                            x_grid,
                            y_grid,
                            config=plot_config,
                            save_path=data_dir / "interface_jump_summary.png",
                        )
            else:
                for name in {
                    "lhs_scatter",
                    "snapshot_pair_samples",
                    "lead_time_coverage",
                    "dataset_summary",
                    "prediction_vs_truth",
                    "interface_error",
                    "lead_time_error",
                    "parameter_error_slices",
                    "interface_jump_summary",
                }:
                    if _should_run(name, groups, individual):
                        _print_skip(name, "need --params")
        else:
            if needs_grid_data:
                for name in grid_data_plots:
                    if _should_run(name, groups, individual):
                        _print_skip(name, "trajectory grids are unavailable")

    # ---- SWEEP GROUP ----
    sweep_plots = _plots_for_group("sweep")
    need_sweep = any(_should_run(p, groups, individual) for p in sweep_plots)

    if need_sweep:
        print("=== SWEEP GROUP ===")

        if args.experiment:
            experiment_dir = Path(args.experiment)
            sweep_dir = out_dir / "sweep"

            if _should_run("sweep_ranking", groups, individual):
                print("--- sweep_ranking ---")
                plot_sweep_ranking(experiment_dir, save_path=sweep_dir / "sweep_ranking.png")

            if _should_run("sweep_convergence", groups, individual):
                if args.runs:
                    print("--- sweep_convergence ---")
                    plot_sweep_convergence(
                        experiment_dir,
                        args.runs,
                        seed=args.sweep_seed,
                        save_path=sweep_dir / "sweep_convergence.png",
                    )
                else:
                    _print_skip("sweep_convergence", "need --runs")

            if _should_run("sweep_hyperparams", groups, individual):
                print("--- sweep_hyperparams ---")
                plot_sweep_hyperparams(experiment_dir, save_path=sweep_dir / "sweep_hyperparams.png")
        else:
            for name in sweep_plots:
                if _should_run(name, groups, individual):
                    _print_skip(name, "need --experiment")

    print(f"\nAll requested plots saved to: {out_dir}")
