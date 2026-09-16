"""Recurring dataset, checkpoint, and physics diagnostics."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

from data.dataset import (
    T_EPS,
    SnapshotPairDataset,
    compute_global_stats,
    load_sim_data,
    problem_from_config,
    split_sim_ids,
)
from src.physics.boundary_forcing import FORCING_TEMPORAL_SAMPLES
from problems.source import INTERFACE_X
from src.physics.internal_source import rc_log_norm
from src.physics.internal_source import (
    PATCH_H,
    PATCH_W,
    interface_control_volume_weights,
    make_rc_sin_profile,
    make_patch_indicator,
    make_sin2_pulse,
)


def _sin_amp_freq(params: dict) -> tuple[float, float]:
    """Pull (amp, freq) from a sim's temporal params; NaN if not the sin family."""
    if params.get("temporal_family") == "sin":
        tp = params["temporal_params"]
        return float(tp["A"]), float(tp["f"])
    return float("nan"), float("nan")


def _safe_param_float(params: dict, key: str) -> float:
    """Read a top-level scalar sim param as float; NaN if missing/non-numeric.

    Lets benchmark-agnostic record builders collect keys that exist only for
    some benchmarks (e.g. source ``A``/``x_h`` vs forcing ``temporal_params``).
    """
    try:
        return float(params.get(key))
    except (TypeError, ValueError):
        return float("nan")


from src.operators.train import load_config
from src.operators.fno2d import FNO2d
from visual._common import (
    PLOT_STYLE,
    _add_interface_lines,
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


def _interface_jump_map(fields: np.ndarray, left_node: int, right_node: int) -> np.ndarray:
    """Return the discrete interface jump history ΔT(y, t) from a stack of fields."""
    return np.asarray(fields[:, right_node, :] - fields[:, left_node, :], dtype=np.float32)


# ------------------------------------------------------------
# Imperfect-interface CONTACT jump (physical Kelvin)
# ------------------------------------------------------------
# The node-to-node jump above mixes three effects: the left half-cell bulk
# conduction drop, the contact discontinuity, and the right half-cell bulk drop.
# The *contact* jump isolates the discontinuity at the imperfect interface:
#
#     dT_contact(t, y) = R_c * q_I = R_c * G * (T_left - T_right),
#     G = 1 / (h_L/k_L + R_c + h_R/k_R),
#
# where G is the per-area interface conductance used by the FV solver
# (src/physics/fv_solver_2d.py:376) and (h_L, h_R) are the node->interface
# distances (h_L + h_R = hx). The FNO predicts nodal temperatures; we postprocess
# them with the known interface law to obtain the implied contact jump.

_DEFAULT_K_LEFT = 2.0
_DEFAULT_K_RIGHT = 1.0


def _interface_conductance_G(x_grid: np.ndarray, interface_x: float, R_c,
                             k_left: float, k_right: float):
    """Per-area interface conductance G = 1/(h_L/k_L + R_c + h_R/k_R)."""
    left_node, right_node = _interface_flanking_nodes_from_grid(x_grid, interface_x)
    h_L = float(interface_x) - float(x_grid[left_node])
    h_R = float(x_grid[right_node]) - float(interface_x)
    Rc = np.asarray(R_c, dtype=np.float64)
    G = 1.0 / (h_L / float(k_left) + Rc + h_R / float(k_right))
    return float(G) if G.ndim == 0 else G


def _interface_contact_jump_map(fields: np.ndarray, x_grid: np.ndarray, interface_x: float,
                                R_c, k_left: float, k_right: float) -> np.ndarray:
    """Return the physical contact-jump history dT_contact(t, y) in Kelvin.

    ``fields`` is a stack of shape (Nt, Nx, Ny). Sign convention: positive means a
    temperature drop left->right (T_L^face - T_R^face = R_c*q_I).
    """
    left_node, right_node = _interface_flanking_nodes_from_grid(x_grid, interface_x)
    G = _interface_conductance_G(x_grid, interface_x, R_c, k_left, k_right)
    Rc = np.asarray(R_c, dtype=np.float64)
    jump = Rc * G * (fields[:, left_node, :] - fields[:, right_node, :])
    return np.asarray(jump, dtype=np.float64)


def _contact_jump_reductions(jump_map: np.ndarray) -> dict[str, np.ndarray]:
    """Per-time y-reductions of a contact-jump map (Nt, Ny) -> dict of (Nt,) arrays.

    ``mean_abs``/``rms`` are robust magnitude headlines; ``signed_mean`` (with
    ``signed_std`` for a band) is secondary and meaningful only where the
    through-interface flux keeps one sign.
    """
    j = np.asarray(jump_map, dtype=np.float64)
    return {
        "mean_abs": np.mean(np.abs(j), axis=1),
        "rms": np.sqrt(np.mean(j ** 2, axis=1)),
        "signed_mean": np.mean(j, axis=1),
        "signed_std": np.std(j, axis=1),
    }


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

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    conf = ckpt["conf"]
    model_cfg = conf.get("model", {}).get("parameters", {})
    if "modes1" not in model_cfg or "modes2" not in model_cfg:
        raise ValueError("Checkpoint is not a 2D FNO checkpoint: missing modes1/modes2")

    dims = problem_from_config(conf).dims
    model = FNO2d(
        modes1=model_cfg["modes1"],
        modes2=model_cfg["modes2"],
        width=model_cfg.get("width", 64),
        in_channels=dims.in_channels,
        out_channels=model_cfg.get("out_channels", 1),
        n_layers=model_cfg.get("n_layers", 4),
        cond_static_dim=dims.cond_static_dim,
        cond_hidden=model_cfg.get("cond_hidden", 256),
        temporal_token_dim=dims.temporal_token_dim,
        temporal_hidden=model_cfg.get("temporal_hidden", 128),
        forcing_embed_dim=model_cfg.get("forcing_embed_dim", 64),
        forcing_spatial_dim=model_cfg.get("forcing_spatial_dim", 16),
        dropout=model_cfg.get("dropout", 0.0),
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
    problem = problem_from_config(config)
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
            problem=problem,
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
            problem=problem,
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
            problem=problem,
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

    config = _resolve_plot_config(config)
    problem = problem_from_config(config)
    interface_meta = _resolve_interface_metadata(
        config=config, sim_params=sim_params, sim_id=sim_id,
    )
    t_targets = t_grid[target_indices]

    # Use global normalization stats from model (attached during checkpoint loading)
    mu_global = getattr(model, "_mu_global", None)
    sigma_global = getattr(model, "_sigma_global", None)
    if mu_global is None or sigma_global is None:
        # Fallback: compute from training split
        from data.dataset import split_sim_ids as _split
        train_ids, _, _ = _split(trajectories.shape[0], 0.7, 0.15, seed=0)
        mu_global, sigma_global = compute_global_stats(trajectories, train_ids)

    plot_dataset = SnapshotPairDataset(
        trajectories=trajectories,
        t_grid=t_grid,
        x_grid=x_grid,
        y_grid=y_grid,
        sim_ids=np.array([sim_id], dtype=np.int64),
        sim_params=sim_params,
        mu_global=float(mu_global),
        sigma_global=float(sigma_global),
        n_snapshots=None,
        noise_std=0.0,
        dt=dt,
        temporal_samples=config.get("model", {}).get("parameters", {}).get(
            "temporal_samples", FORCING_TEMPORAL_SAMPLES,
        ),
        problem=problem,
    )
    items = [
        problem.build_item(plot_dataset, int(sim_id), int(s), int(target_idx))
        for target_idx in target_indices
    ]
    spatial = torch.from_numpy(np.stack([item["spatial"] for item in items], axis=0))
    cond_static = torch.from_numpy(np.stack([item["cond_static"] for item in items], axis=0))
    forcing_seq = torch.from_numpy(np.stack([item["forcing_seq"] for item in items], axis=0))

    device = next(model.parameters()).device
    with torch.no_grad():
        Y_pred_norm = model(
            spatial.to(device),
            cond_static.to(device),
            forcing_seq.to(device),
        ).cpu().numpy()[..., 0]

    Y_pred_rows = []
    Y_true_rows = []
    for row, item in enumerate(items):
        stats = np.asarray(item["T_stats"], dtype=np.float32)
        mu_s = float(stats[0])
        sigma_s = float(stats[1])
        Y_pred_rows.append((Y_pred_norm[row] * (sigma_s + T_EPS) + mu_s).astype(np.float32))
        Y_true_rows.append((item["Y"][..., 0] * (sigma_s + T_EPS) + mu_s).astype(np.float32))
        if stats.shape[0] > 2:
            interface_meta = {
                **interface_meta,
                "interface_x": float(stats[2]),
                "positions": [float(stats[2])],
            }

    params = sim_params[int(sim_id)]
    amp, freq = _sin_amp_freq(params)
    return {
        "interface_x": interface_meta["interface_x"],
        "interface_positions": interface_meta["positions"],
        "interface_half_width": interface_meta["interface_half_width"],
        "t_targets": t_targets,
        "t_bars": t_grid[target_indices] - t_grid[s],
        "source_time": float(t_grid[s]),
        "Y_pred": np.stack(Y_pred_rows, axis=0),
        "Y_true": np.stack(Y_true_rows, axis=0),
        "amp": amp,
        "freq": freq,
        "R_c": float(params["R_c"]),
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
    forcing_batch = []
    y_batch = []
    stats_batch = []
    for idx in sample_indices:
        item = dataset[idx]
        x_batch.append(item["spatial"])
        cond_batch.append(item["cond_static"])
        forcing_batch.append(item["forcing_seq"])
        y_batch.append(item["Y"])
        stats_batch.append(item["T_stats"])

    x_tensor = torch.stack(x_batch, dim=0)
    cond_tensor = torch.stack(cond_batch, dim=0)
    forcing_tensor = torch.stack(forcing_batch, dim=0)
    y_tensor = torch.stack(y_batch, dim=0)
    stats_tensor = torch.stack(stats_batch, dim=0)

    interface_meta = _resolve_interface_metadata(config=_resolve_plot_config(config))
    device = next(model.parameters()).device
    model.eval()

    preds = []
    for start in range(0, sample_size, batch_size):
        stop = min(start + batch_size, sample_size)
        with torch.no_grad():
            pred = model(
                x_tensor[start:stop].to(device),
                cond_tensor[start:stop].to(device),
                forcing_tensor[start:stop].to(device),
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
    if stats_tensor.shape[1] > 2:
        interface_x_values = stats_tensor[:, 2].numpy().astype(np.float32)
        iface_rel_rows = []
        jump_rel_rows = []
        true_jump_rms_rows = []
        pred_jump_rms_rows = []
        for row, interface_x in enumerate(interface_x_values):
            interface_metric_mask = torch.from_numpy(
                _build_interface_mask_2d(
                    x_grid, y_grid, float(interface_x),
                    interface_meta["interface_half_width"],
                )
            )
            left_node, right_node = _interface_flanking_nodes_from_grid(x_grid, float(interface_x))
            pred_iface = y_pred_phys_field[row, interface_metric_mask]
            true_iface = y_true_phys_field[row, interface_metric_mask]
            iface_rel_rows.append(
                (
                    torch.mean((pred_iface - true_iface) ** 2)
                    / torch.clamp(torch.mean(true_iface ** 2), min=1e-12)
                ).sqrt() * 100.0
            )
            pred_jump_row = y_pred_phys_field[row, right_node, :] - y_pred_phys_field[row, left_node, :]
            true_jump_row = y_true_phys_field[row, right_node, :] - y_true_phys_field[row, left_node, :]
            jump_rel_rows.append(
                (
                    torch.mean((pred_jump_row - true_jump_row) ** 2)
                    / torch.clamp(torch.mean(true_jump_row ** 2), min=1e-12)
                ).sqrt() * 100.0
            )
            true_jump_rms_rows.append(torch.sqrt(torch.mean(true_jump_row ** 2)))
            pred_jump_rms_rows.append(torch.sqrt(torch.mean(pred_jump_row ** 2)))

        iface_rel = torch.stack(iface_rel_rows)
        jump_rel = torch.stack(jump_rel_rows)
        true_jump_rms = torch.stack(true_jump_rms_rows)
        pred_jump_rms = torch.stack(pred_jump_rms_rows)
    else:
        interface_x_values = np.full(sample_size, float(interface_meta["interface_x"]), dtype=np.float32)
        interface_metric_mask = torch.from_numpy(
            _build_interface_mask_2d(
                x_grid, y_grid,
                interface_meta["interface_x"],
                interface_meta["interface_half_width"],
            )
        )
        left_node, right_node = _interface_flanking_nodes_from_grid(x_grid, interface_meta["interface_x"])
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
        "sim_id": sim_ids,
        "lead_time": np.array([dataset._lead_times[idx] for idx in sample_indices], dtype=np.float32),
        "global_rel_l2": global_rel.numpy(),
        "iface_rel_l2": iface_rel.numpy(),
        "jump_rel_l2": jump_rel.numpy(),
        "amplitude": np.array([_sin_amp_freq(p)[0] for p in params], dtype=np.float32),
        "frequency": np.array([_sin_amp_freq(p)[1] for p in params], dtype=np.float32),
        "R_c": np.array([float(p["R_c"]) for p in params], dtype=np.float32),
        # source-benchmark fields (NaN/empty for forcing/interface sims)
        "A": np.array([_safe_param_float(p, "A") for p in params], dtype=np.float32),
        "x_h": np.array([_safe_param_float(p, "x_h") for p in params], dtype=np.float32),
        "y_h": np.array([_safe_param_float(p, "y_h") for p in params], dtype=np.float32),
        "interface_x": interface_x_values,
        "regime": np.array([str(p.get("regime", "")) for p in params], dtype=object),
        "true_jump_rms": true_jump_rms.numpy(),
        "pred_jump_rms": pred_jump_rms.numpy(),
        "source_index": source_indices,
        "target_index": target_indices,
    }


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
            jump_rmse = float(np.sqrt(np.mean((pred_jump_row - true_jump_row) ** 2)))

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
            ax_jump.set_ylabel("Node-to-node ΔT")
            ax_jump.set_title(f"Interface Jump Profile — jump rel. err. {jump_rel:.2f}%")
            ax_jump.text(
                0.03,
                0.04,
                f"Field rel. L2: {global_rel:.2f}%\nJump RMSE: {jump_rmse:.3f} K",
                transform=ax_jump.transAxes,
                va="bottom",
                ha="left",
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.8"},
            )
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
        left_node, right_node = _interface_flanking_nodes_from_grid(x_grid, case["interface_x"])
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
            left_node, right_node = _interface_flanking_nodes_from_grid(x_grid, case["interface_x"])
            interface_metric_mask = _build_interface_mask_2d(
                x_grid, y_grid,
                case["interface_x"],
                case["interface_half_width"],
            )
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
            ax_left.set_title(f"True Jump Map — x={case['interface_x']:.3f}")
            ax_mid.set_xlabel("Time")
            ax_mid.set_ylabel("y")
            ax_mid.set_title(f"Predicted Jump Map — x={case['interface_x']:.3f}")
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

        fig.suptitle("Interface Diagnostic")
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


def plot_initial_conditions(
    trajectories: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    n_samples: int = 20,
    seed: int = 42,
    save_path: str | Path | None = None,
):
    """Plot sampled 2D initial-condition fields.

    Shows only the IC draw itself, so no interface geometry is overlaid.
    """
    rng = np.random.default_rng(seed)
    num_sims = trajectories.shape[0]
    selected = sorted(rng.choice(num_sims, size=min(n_samples, num_sims), replace=False))
    nrows, ncols = _shared_thumbnail_grid(len(selected))
    fields = trajectories[selected, 0]
    vmin = float(np.min(fields))
    vmax = float(np.max(fields))

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(4.4 * ncols, 3.8 * nrows),
            squeeze=False, constrained_layout=True,
        )
        axes_flat = axes.ravel()
        pcm = None
        for ax, sim_id, field in zip(axes_flat, selected, fields):
            pcm = _plot_field_2d(ax, x_grid, y_grid, field, vmin=vmin, vmax=vmax)
            ax.set_title(f"Sim {sim_id}")

        for ax in axes_flat[len(selected):]:
            ax.set_axis_off()

        fig.colorbar(pcm, ax=axes.ravel().tolist(), label="Temperature", shrink=0.9)
        fig.suptitle(f"Initial Conditions T(x, y, t=0) — {len(selected)} simulations")
        _save_figure(fig, save_path, "data", "initial_conditions", layout="constrained")


# ============================================================
# SOURCE BENCHMARK — helpers (ported from experiment/source-itr)
# ============================================================

def _patch_param_arrays(sim_params: np.ndarray) -> dict[str, np.ndarray]:
    """Pull per-sim patch parameters into arrays."""
    return {
        "R_c": np.array([float(p["R_c"]) for p in sim_params], dtype=np.float32),
        "A": np.array([float(p["A"]) for p in sim_params], dtype=np.float32),
        "x_h": np.array([float(p["x_h"]) for p in sim_params], dtype=np.float32),
        "y_h": np.array([float(p["y_h"]) for p in sim_params], dtype=np.float32),
        "regime": np.array([str(p.get("regime", "")) for p in sim_params], dtype=object),
    }


def _itr_param_arrays(sim_params: np.ndarray) -> dict[str, np.ndarray]:
    """Pull source-ITR sinusoidal parameters into arrays."""
    R_base = np.array([float(p["R_c_base"]) for p in sim_params], dtype=np.float64)
    R_amp = np.array([float(p["R_c_A"]) for p in sim_params], dtype=np.float64)
    return {
        "R_c_base": R_base,
        "R_c_A": R_amp,
        "R_c_peak": R_base + R_amp,
    }


def _interface_conductance_profile(
    y_grid: np.ndarray,
    R_c_y: np.ndarray,
    x_grid: np.ndarray | None = None,
    interface_x: float = INTERFACE_X,
    k_left: float = _DEFAULT_K_LEFT,
    k_right: float = _DEFAULT_K_RIGHT,
) -> np.ndarray:
    """Return the per-row interface conductance profile with solver-matching distances."""
    y = np.asarray(y_grid, dtype=np.float64)
    Rc = np.asarray(R_c_y, dtype=np.float64)
    if x_grid is not None:
        x = np.asarray(x_grid, dtype=np.float64)
        return np.asarray(
            _interface_conductance_G(x, interface_x, Rc, k_left, k_right),
            dtype=np.float64,
        )

    dy = float(y[1] - y[0]) if len(y) > 1 else 1.0
    return 1.0 / ((0.5 * dy) / float(k_left) + Rc + (0.5 * dy) / float(k_right))


def _itr_amplitude_arrays(
    sim_params: np.ndarray,
    y_grid: np.ndarray,
    x_grid: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Return resistance parameters and a conductance-deficit diagnostic."""
    arrs = _itr_param_arrays(sim_params)
    y = np.asarray(y_grid, dtype=np.float64)
    conductance_deficit = np.zeros(len(sim_params), dtype=np.float64)

    for i, p in enumerate(sim_params):
        Rc_y = make_rc_sin_profile(
            y,
            R_base=float(p["R_c_base"]),
            A=float(p["R_c_A"]),
        )
        base = np.full_like(Rc_y, float(p["R_c_base"]), dtype=np.float64)
        G_base = _interface_conductance_profile(
            y, base, x_grid=x_grid, interface_x=float(p.get("interface_x", INTERFACE_X))
        )
        G_y = _interface_conductance_profile(
            y, Rc_y, x_grid=x_grid, interface_x=float(p.get("interface_x", INTERFACE_X))
        )
        bounds = (0.0, 1.0)
        weights = interface_control_volume_weights(y, bounds)
        conductance_deficit[i] = float(np.sum(weights * (G_base - G_y)))

    arrs["conductance_deficit"] = conductance_deficit
    arrs["is_itr_active"] = arrs["R_c_A"] > 1e-12
    return arrs


def _representative_itr_sim_ids(
    sim_params: np.ndarray,
    y_grid: np.ndarray,
    x_grid: np.ndarray | None = None,
) -> list[int]:
    """Pick deterministic low/median/high amplitude source-ITR samples."""
    severity = _itr_amplitude_arrays(sim_params, y_grid, x_grid)["R_c_A"]
    if len(severity) == 0:
        return []
    order = np.argsort(severity)
    picks = [int(order[0]), int(order[len(order) // 2]), int(order[-1])]
    out: list[int] = []
    for sid in picks:
        if sid not in out:
            out.append(sid)
    return out


def _require_source_itr_sin_params(sim_params: np.ndarray) -> None:
    required = ("R_c_base", "R_c_A")
    for i, p in enumerate(sim_params):
        missing = [key for key in required if key not in p]
        if missing:
            raise ValueError(f"sim_params[{i}] missing source-ITR keys {missing}.")


def _select_representative_source_sim_id(sim_params: np.ndarray) -> int:
    """Choose a deterministic representative source sim near the median patch params."""
    arrs = _patch_param_arrays(sim_params)
    A = arrs["A"]
    if len(A) == 0:
        return 0
    Rc = arrs["R_c"]
    xh = arrs["x_h"]
    yh = arrs["y_h"]

    def _norm(v: np.ndarray) -> np.ndarray:
        v_min, v_max = float(v.min()), float(v.max())
        return (v - v_min) / max(v_max - v_min, 1e-12)

    stacked = np.column_stack([_norm(A), _norm(Rc), _norm(xh), _norm(yh)])
    target = np.median(stacked, axis=0)
    return int(np.argmin(np.sum((stacked - target) ** 2, axis=1)))


def _format_patch_label(p: dict) -> str:
    """Compact label describing a sim's patch parameters."""
    return (
        f"x_h={float(p['x_h']):.2f}, y_h={float(p['y_h']):.2f}, "
        f"A={float(p['A']):.0f}"
    )


def _bilinear_resample(
    field: np.ndarray, x_grid: np.ndarray, y_grid: np.ndarray,
    xs: np.ndarray, ys: np.ndarray,
) -> np.ndarray:
    """Bilinear resample ``field`` of shape (Nx, Ny) at sample points (xs, ys).

    xs, ys arrays of the same shape. Returns the resampled field with that
    shape. Points outside the convex hull return NaN.
    """
    Nx = len(x_grid)
    Ny = len(y_grid)
    x_lo = float(x_grid[0])
    x_hi = float(x_grid[-1])
    y_lo = float(y_grid[0])
    y_hi = float(y_grid[-1])
    valid = (xs >= x_lo) & (xs <= x_hi) & (ys >= y_lo) & (ys <= y_hi)
    fx = (xs - x_lo) / max(x_hi - x_lo, 1e-12) * (Nx - 1)
    fy = (ys - y_lo) / max(y_hi - y_lo, 1e-12) * (Ny - 1)
    i0 = np.clip(np.floor(fx).astype(int), 0, Nx - 2)
    j0 = np.clip(np.floor(fy).astype(int), 0, Ny - 2)
    i1 = i0 + 1
    j1 = j0 + 1
    wx = fx - i0
    wy = fy - j0
    f00 = field[i0, j0]
    f01 = field[i0, j1]
    f10 = field[i1, j0]
    f11 = field[i1, j1]
    out = (
        (1 - wx) * (1 - wy) * f00
        + (1 - wx) * wy * f01
        + wx * (1 - wy) * f10
        + wx * wy * f11
    )
    out = np.where(valid, out, np.nan)
    return out


# ============================================================
# SOURCE BENCHMARK — plots (ported from experiment/source-itr)
# ============================================================

def plot_source_itr_sin_resistance_profiles(
    sim_params: np.ndarray,
    y_grid: np.ndarray,
    x_grid: np.ndarray | None = None,
    save_path: str | Path | None = None,
):
    """Explain sampled sinusoidal interface resistance profiles."""
    _require_source_itr_sin_params(sim_params)
    y = np.asarray(y_grid, dtype=np.float64)
    severity = _itr_amplitude_arrays(sim_params, y, x_grid)
    representative_ids = _representative_itr_sim_ids(sim_params, y, x_grid)

    profile_rows = []
    for sid in representative_ids:
        p = sim_params[int(sid)]
        Rc_y = make_rc_sin_profile(
            y,
            R_base=float(p["R_c_base"]),
            A=float(p["R_c_A"]),
        )
        G_y = _interface_conductance_profile(
            y, Rc_y, x_grid=x_grid, interface_x=float(p.get("interface_x", INTERFACE_X))
        )
        profile_rows.append((int(sid), p, Rc_y, rc_log_norm(Rc_y), G_y))

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, max(len(profile_rows), 1)))
    active = severity["is_itr_active"]

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)

        ax = axes[0, 0]
        for color, (sid, p, Rc_y, _Rc_norm, _G_y) in zip(colors, profile_rows):
            label = (
                f"sim {sid}: peak={float(p['R_c_base']) + float(p['R_c_A']):.2f}, "
                f"amplitude={severity['R_c_A'][sid]:.3f}"
            )
            ax.plot(y, Rc_y, color=color, label=label)
        ax.set_xlabel("y")
        ax.set_ylabel(r"$R_c(y)$")
        ax.set_title("Physical interface resistance")
        ax.legend(loc="upper right")
        ax.grid(True)

        ax = axes[0, 1]
        for color, (sid, _p, _Rc_y, Rc_norm, _G_y) in zip(colors, profile_rows):
            ax.plot(y, Rc_norm, color=color, label=f"sim {sid}")
        ax.axhline(-1.0, color="0.7", linestyle=":", linewidth=1.0)
        ax.axhline(1.0, color="0.7", linestyle=":", linewidth=1.0)
        ax.set_xlabel("y")
        ax.set_ylabel("normalized channel")
        ax.set_title(r"Model input channel: log-normalized $R_c(y)$")
        ax.grid(True)

        ax = axes[1, 0]
        for color, (sid, _p, _Rc_y, _Rc_norm, G_y) in zip(colors, profile_rows):
            ax.plot(y, G_y, color=color, label=f"sim {sid}")
        ax.set_xlabel("y")
        ax.set_ylabel(r"$G(y)$")
        ax.set_title("Discrete interface conductance")
        ax.grid(True)

        ax = axes[1, 1]
        scatter = ax.scatter(severity["R_c_base"], severity["R_c_A"],
                             c=severity["R_c_A"], cmap="magma")
        fig.colorbar(scatter, ax=ax, label=r"Amplitude $A$")
        ax.set_xlabel(r"Base resistance $R_b$")
        ax.set_ylabel(r"Amplitude $A$")
        ax.set_title("Sinusoidal profile coverage")
        ax.grid(True)
        fig.suptitle("Source-ITR Sinusoidal Profiles")
        _save_figure(fig, save_path, "source", "source_itr_sin_resistance_profiles", layout="constrained")


def plot_source_interface_zone_error(
    model,
    dataset: SnapshotPairDataset,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    config: dict,
    max_samples: int = 256,
    seed: int = 42,
    save_path: str | Path | None = None,
):
    """Lightweight interface-zone diagnostic.

    (a) Scatter of pred-vs-truth jump magnitude |T(0.5+) - T(0.5-)| (RMS over y).
    (b) Scatter of rel-L2 inside the band |x - 0.5| < 0.05 vs R_c.
    The jump is computed from the two cell centers adjacent to x=0.5, matching
    the FV face convention.
    """
    config = _resolve_plot_config(config)
    records = _compute_pair_error_records(
        model=model, dataset=dataset,
        x_grid=x_grid, y_grid=y_grid,
        config=config, max_samples=max_samples, seed=seed,
    )

    true_jump = records["true_jump_rms"]
    pred_jump = records["pred_jump_rms"]
    rc = records["R_c"]
    iface_rel = records["iface_rel_l2"]

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)

        ax = axes[0]
        if len(true_jump) > 0:
            lo = float(min(true_jump.min(), pred_jump.min()))
            hi = float(max(true_jump.max(), pred_jump.max()))
        else:
            lo, hi = 0.0, 1.0
        ax.scatter(true_jump, pred_jump, s=18, alpha=0.45, color="C2", edgecolors="none")
        ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1.0, label="Identity")
        ax.set_xlabel(r"True $|T(0.5^+) - T(0.5^-)|$ RMS")
        ax.set_ylabel(r"Predicted $|T(0.5^+) - T(0.5^-)|$ RMS")
        ax.set_title("Jump magnitude — pred vs truth")
        ax.legend(loc="upper left")
        ax.grid(True)

        ax = axes[1]
        ax.scatter(rc, iface_rel, s=18, alpha=0.45, color="C3", edgecolors="none")
        ax.set_xlabel(r"Contact resistance $R_c$")
        ax.set_ylabel("Interface band rel. L2 (%)")
        ax.set_title(r"Band $|x - 0.5| < 0.05$ error vs $R_c$")
        ax.grid(True)

        fig.suptitle("Interface Zone Diagnostic")
        _save_figure(fig, save_path, "source", "source_interface_zone_error", layout="constrained")


def plot_patch_region_error_map(
    model,
    dataset: SnapshotPairDataset,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    sim_params: np.ndarray,
    config: dict | None = None,
    window: float = 0.3,
    max_samples: int = 128,
    seed: int = 42,
    save_path: str | Path | None = None,
):
    """Mean per-pixel MAE recentered to each pair's patch center (x_h, y_h)."""
    import torch

    if len(dataset) == 0:
        raise ValueError("Dataset is empty; cannot compute patch-region error map.")

    rng = np.random.default_rng(seed)
    sample_size = min(max_samples, len(dataset))
    sample_indices = np.arange(len(dataset))
    if sample_size < len(dataset):
        sample_indices = np.sort(rng.choice(sample_indices, size=sample_size, replace=False))

    x_batch, c_batch, y_batch, stats_batch = [], [], [], []
    forcing_batch = []
    for idx in sample_indices:
        item = dataset[idx]
        x_batch.append(item["spatial"])
        c_batch.append(item["cond_static"])
        forcing_batch.append(item["forcing_seq"])
        y_batch.append(item["Y"])
        stats_batch.append(item["T_stats"])
    x_tensor = torch.stack(x_batch, dim=0)
    c_tensor = torch.stack(c_batch, dim=0)
    forcing_tensor = torch.stack(forcing_batch, dim=0)
    y_tensor = torch.stack(y_batch, dim=0)
    stats_tensor = torch.stack(stats_batch, dim=0)

    device = next(model.parameters()).device
    model.eval()
    preds = []
    batch_size = 32
    for start in range(0, sample_size, batch_size):
        stop = min(start + batch_size, sample_size)
        with torch.no_grad():
            p = model(x_tensor[start:stop].to(device),
                     c_tensor[start:stop].to(device),
                     forcing_tensor[start:stop].to(device)).cpu()
        preds.append(p)
    y_pred = torch.cat(preds, dim=0)
    mu_s = stats_tensor[:, 0][:, None, None, None]
    sigma_s = stats_tensor[:, 1][:, None, None, None]
    y_pred_phys = (y_pred * (sigma_s + T_EPS) + mu_s).numpy()[..., 0]
    y_true_phys = (y_tensor * (sigma_s + T_EPS) + mu_s).numpy()[..., 0]
    abs_err = np.abs(y_pred_phys - y_true_phys)

    sim_ids = np.array([dataset._pairs[i][0] for i in sample_indices], dtype=np.int64)

    local_n = 64
    grid_local = np.linspace(-0.5 * window, 0.5 * window, local_n)
    Xl, Yl = np.meshgrid(grid_local, grid_local, indexing="ij")

    accum = np.zeros((local_n, local_n), dtype=np.float64)
    weight = np.zeros((local_n, local_n), dtype=np.float64)
    used = 0
    for k, sid in enumerate(sim_ids):
        params = sim_params[int(sid)]
        x_h = float(params["x_h"])
        y_h = float(params["y_h"])
        if (x_h - 0.5 * window < x_grid[0] or x_h + 0.5 * window > x_grid[-1]
            or y_h - 0.5 * window < y_grid[0] or y_h + 0.5 * window > y_grid[-1]):
            continue
        sampled = _bilinear_resample(abs_err[k], x_grid, y_grid, Xl + x_h, Yl + y_h)
        valid = ~np.isnan(sampled)
        accum[valid] += sampled[valid]
        weight[valid] += 1.0
        used += 1
    mean_err = np.where(weight > 0, accum / np.maximum(weight, 1.0), np.nan)

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
        pcm = ax.pcolormesh(grid_local, grid_local, mean_err.T, cmap="magma",
                           shading="auto")
        fig.colorbar(pcm, ax=ax, label="Mean |pred - truth|")
        rect = Rectangle((-0.5 * PATCH_W, -0.5 * PATCH_H), PATCH_W, PATCH_H,
                        fill=False, edgecolor="cyan", linewidth=1.6)
        ax.add_patch(rect)
        ax.axhline(0.0, color="white", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.axvline(0.0, color="white", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.set_xlabel(r"$x - x_h$")
        ax.set_ylabel(r"$y - y_h$")
        ax.set_aspect("equal")
        ax.set_title(
            f"Mean Absolute Error Centered on Patch ({used} pairs; interface fixed at x=0.5)"
        )
        _save_figure(fig, save_path, "source", "patch_region_error_map", layout="constrained")


def plot_energy_budget(
    trajectories: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    t_grid: np.ndarray,
    sim_params: np.ndarray,
    sim_id: int | None = None,
    save_path: str | Path | None = None,
):
    """Compare injected, stored, and outgoing energy for one representative sim.

    Assumes rho = cp = 1 (from CLAUDE.md) and right-wall material k=1
    matching the source-itr experiment. Right-wall Dirichlet uses one-sided
    cell-center to ghost difference, consistent with the FV solver stencil.
    """
    if sim_id is None:
        sim_id = _select_representative_source_sim_id(sim_params)
    params = sim_params[int(sim_id)]
    x_h = float(params["x_h"])
    y_h = float(params["y_h"])
    w_h = float(params["w_h"])
    h_h = float(params["h_h"])
    A = float(params["A"])
    t_off = float(params["t_off"])

    Nx = len(x_grid)
    Ny = len(y_grid)
    dx = float(x_grid[1] - x_grid[0])
    dy = float(y_grid[1] - y_grid[0])

    cell_dx = np.full(Nx, dx)
    cell_dx[0] = dx / 2.0
    cell_dx[-1] = dx / 2.0
    cell_dy = np.full(Ny, dy)
    cell_dy[0] = dy / 2.0
    cell_dy[-1] = dy / 2.0
    cell_area = cell_dx[:, None] * cell_dy[None, :]

    X, Y = np.meshgrid(x_grid, y_grid, indexing="ij")
    indicator = make_patch_indicator(X, Y, x_h, y_h, w_h, h_h)
    pulse = make_sin2_pulse(A, t_off)
    a_t = np.array([pulse(float(t)) for t in t_grid], dtype=np.float64)
    Q_per_cell_t = a_t[:, None, None] * indicator[None, ...]
    integrand_t = np.sum(Q_per_cell_t * cell_area[None, ...], axis=(1, 2))
    dt = np.diff(t_grid)
    E_in = np.concatenate([[0.0], np.cumsum(0.5 * (integrand_t[1:] + integrand_t[:-1]) * dt)])

    traj = trajectories[int(sim_id)].astype(np.float64)
    T0_field = traj[0]
    delta = traj - T0_field[None, ...]
    E_stored = np.sum(delta * cell_area[None, ...], axis=(1, 2))

    T_right_target = 300.0
    k_right = 1.0
    T_ghost = 2.0 * T_right_target - traj[:, -1, :]
    flux_t = -k_right * (T_ghost - traj[:, -1, :]) / (dx / 2.0)
    out_per_t = np.sum(flux_t * cell_dy[None, :], axis=1)
    E_out_right = np.concatenate([[0.0], np.cumsum(0.5 * (out_per_t[1:] + out_per_t[:-1]) * dt)])

    residual = E_in - E_stored - E_out_right

    arrs = _patch_param_arrays(sim_params)
    max_dev = np.array([
        float(np.max(np.mean(trajectories[i] - trajectories[i, 0], axis=(1, 2))))
        for i in range(len(sim_params))
    ])

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

        ax = axes[0]
        ax.plot(t_grid, E_in, color="C0", label=r"$E_{in}$")
        ax.plot(t_grid, E_stored, color="C2", label=r"$\Delta E_{stored}$")
        ax.plot(t_grid, E_out_right, color="C3", label=r"$E_{out}^{right}$")
        ax.set_xlabel("t")
        ax.set_ylabel("Energy")
        ax.legend(loc="upper left")
        ax.grid(True)
        ax.set_title(f"Energy Budget — sim {sim_id} ({_format_patch_label(params)})")
        ax2 = ax.twinx()
        ax2.plot(t_grid, residual, color="0.4", linestyle="--",
                 label=r"residual $E_{in} - \Delta E_{stored} - E_{out}$")
        ax2.set_ylabel("Residual")
        ax2.legend(loc="lower right")

        ax = axes[1]
        ax.scatter(arrs["A"], max_dev, c="C1", alpha=0.7, edgecolors="none", s=18)
        ax.set_xscale("log")
        ax.set_xlabel("A")
        ax.set_ylabel(r"$\max_t \langle T - T_0 \rangle_\Omega$")
        ax.set_title("Response Scaling vs A")
        ax.grid(True, which="both")

        fig.suptitle("Energy Budget and Response Scaling")
        _save_figure(fig, save_path, "source", "energy_budget", layout="constrained")


def _interface_x_array(sim_params: np.ndarray) -> np.ndarray:
    """Return per-sim interface_x array."""
    return np.array([float(p["interface_x"]) for p in sim_params], dtype=np.float64)


def _pick_interface_quartile_sims(sim_params: np.ndarray) -> list[tuple[str, int]]:
    """Pick one sim per interface_x quartile (Q1, Q2, Q3, Q4).

    Returns a list of (label, sim_id). Order is from low to high interface_x.
    Matches the existing breakdown style (no controlling for other params).
    """
    interface_x = _interface_x_array(sim_params)
    n = len(interface_x)
    if n == 0:
        return []
    order = np.argsort(interface_x)
    picks: list[tuple[str, int]] = []
    for q in range(4):
        lo = q * n // 4
        hi = (q + 1) * n // 4 if q < 3 else n
        if hi <= lo:
            continue
        mid = lo + (hi - lo) // 2
        sid = int(order[mid])
        picks.append((f"Q{q + 1}", sid))
    return picks


# ============================================================
# INTERFACES BENCHMARK — plots
# ============================================================


def plot_interface_x_breakdown(
    trajectories: np.ndarray,
    sim_params: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    t_grid: np.ndarray,
    t_window: tuple[float, float] = (0.0, 0.20),
    n_snaps: int = 4,
    save_path: str | Path | None = None,
):
    """Tile field snapshots across interface_x quartiles.

    Rows: one representative sim per interface_x quartile (Q1..Q4).
    Cols: ``n_snaps`` snapshots in the flux-active window.
    Each row uses its own colorbar and overlays its sim's interface_x.
    """
    picks = _pick_interface_quartile_sims(sim_params)
    if not picks:
        raise ValueError("No sims available for interface_x breakdown.")

    snap_indices = _snap_indices_in_window(t_grid, t_window[0], t_window[1], n_snaps)
    nrows = len(picks)
    ncols = len(snap_indices)

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(3.8 * ncols + 1.5, 3.6 * nrows),
            constrained_layout=True, squeeze=False,
        )
        for r, (qlabel, sid) in enumerate(picks):
            fields = trajectories[sid, snap_indices].astype(np.float32)
            vmin = float(np.min(fields))
            vmax = float(np.max(fields))
            iface = _resolve_interface_metadata(sim_params=sim_params, sim_id=sid)
            row_pcm = None
            for c, (t_idx, field) in enumerate(zip(snap_indices, fields)):
                ax = axes[r, c]
                row_pcm = _plot_field_2d(
                    ax, x_grid, y_grid, field,
                    cmap="inferno", vmin=vmin, vmax=vmax,
                    interface_positions=iface["positions"],
                )
                if r == 0:
                    ax.set_title(f"t = {t_grid[t_idx]:.3f}")
                if c != 0:
                    ax.set_ylabel("")
            row_label = (
                f"{qlabel}\nsim {sid}\n"
                f"$x_I$={iface['interface_x']:.3f}\n"
                f"$R_c$={float(sim_params[sid]['R_c']):.2f}"
            )
            axes[r, 0].annotate(
                row_label,
                xy=(-0.32, 0.5), xycoords="axes fraction",
                ha="right", va="center", fontsize=10,
            )
            fig.colorbar(row_pcm, ax=axes[r, :].tolist(), label="Temperature", shrink=0.85)

        fig.suptitle(r"Interface-Location Breakdown — fields per $x_I$ quartile", fontsize=14)
        _save_figure(fig, save_path, "interfaces", "interface_x_breakdown", layout="constrained")
