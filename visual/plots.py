from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from data.dataset import (
    AMP_RANGE,
    FREQ_RANGE,
    RC_RANGE,
    T_EPS,
    SnapshotPairDataset,
    load_sim_data,
    split_sim_ids,
)
from src.operators.train import load_config
from src.physics.fd_solver_1d import FDSolver1D, Layer1D
from src.operators.fno1d import FNO1d



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
    # group: sweep
    "sweep_ranking":             "sweep",
    "sweep_convergence":         "sweep",
    "sweep_hyperparams":         "sweep",
}

GROUPS = {"physics", "mms", "training", "data", "sweep"}


def _ensure_parent(path: Path) -> Path:
    """Create parent directory if needed and return path unchanged."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _should_run(name: str, groups: list[str], individual: list[str] | None) -> bool:
    """Check whether a named plot should be generated given the CLI flags."""
    if individual is not None:
        return name in individual
    if "all" in groups:
        return True
    return PLOT_REGISTRY.get(name) in groups


# ============================================================
# HELPERS — Demo solver + interface utilities
# ============================================================

def create_demo_multilayer_solver() -> FDSolver1D:
    """Create a 2-layer demo solver for physics visualisation plots.

    Matches the layer configuration in generate_dataset.py:
      Layer 1: [0.0, 0.5], rho=1.0, cp=1.0, k=2.0
      Layer 2: [0.5, 1.0], rho=1.0, cp=1.0, k=1.0
    N=100 ensures x=0.5 lies on a cell face (required by the solver).
    """
    layers = [
        Layer1D(x_left=0.0, x_right=0.5, rho=1.0, cp=1.0, k=2.0),
        Layer1D(x_left=0.5, x_right=1.0, rho=1.0, cp=1.0, k=1.0),
    ]
    return FDSolver1D(
        a=0.0, b=1.0, N=100, layers=layers, lam_target=0.8,
        interface_R=[0.5],
        t_final=1.0, flux_f=2.0, flux_A=50.0,
        t_on=0.0, t_off=0.2, phase=0.0, dt=0.005,
    )


def _interface_flanking_nodes(solver: FDSolver1D) -> list[tuple[int, int, float]]:
    """Return (left_node, right_node, interface_x) for each internal interface."""
    return [
        (f, f + 1, solver.face_positions[f])
        for f in sorted(solver.interface_face_map.keys())
    ]


def _load_plot_data(
    data_path: str | Path,
    x_grid_path: str | Path,
    t_grid_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load trajectories and saved grids for plot generation."""
    trajectories, x_grid, t_grid = load_sim_data(
        sim_traj_path=str(Path(data_path)),
        x_grid_path=str(Path(x_grid_path)),
        t_grid_path=str(Path(t_grid_path)),
    )
    return trajectories, x_grid, t_grid


def _resolve_plot_config(checkpoint_conf: dict | None = None) -> dict:
    """Prefer checkpoint config when available, otherwise load the default project config."""
    return checkpoint_conf if checkpoint_conf is not None else load_config()


def _load_checkpoint_model(checkpoint_path: str | Path) -> tuple[FNO1d, dict]:
    """Load a checkpoint and reconstruct the matching FNO model on CPU."""
    import torch

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    conf = ckpt["conf"]
    model_cfg = conf.get("model", {}).get("parameters", {})
    model = FNO1d(
        modes=model_cfg.get("modes", 16),
        width=model_cfg.get("width", 64),
        in_channels=model_cfg.get("in_channels", 2),
        out_channels=model_cfg.get("out_channels", 1),
        n_layers=model_cfg.get("n_layers", 4),
        cond_dim=model_cfg.get("cond_dim", 5),
        cond_hidden=model_cfg.get("cond_hidden", 256),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, conf


def _build_split_datasets(
    trajectories: np.ndarray,
    x_grid: np.ndarray,
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
    return {
        "train": SnapshotPairDataset(
            trajectories=trajectories,
            t_grid=t_grid,
            x_grid=x_grid,
            sim_ids=train_ids,
            sim_params=sim_params,
            n_snapshots=n_snapshots,
        ),
        "val": SnapshotPairDataset(
            trajectories=trajectories,
            t_grid=t_grid,
            x_grid=x_grid,
            sim_ids=val_ids,
            sim_params=sim_params,
            n_snapshots=n_snapshots,
        ),
        "test": SnapshotPairDataset(
            trajectories=trajectories,
            t_grid=t_grid,
            x_grid=x_grid,
            sim_ids=test_ids,
            sim_params=sim_params,
            n_snapshots=test_snapshots,
        ),
    }


def _prepare_prediction_case(
    model,
    trajectories: np.ndarray,
    x_grid: np.ndarray,
    t_grid: np.ndarray,
    sim_params: np.ndarray,
    sim_id: int,
    s: int,
    target_indices: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Prepare truth/prediction arrays for one source snapshot and multiple target times."""
    import torch

    interface_x = 0.5
    t_targets = t_grid[target_indices]

    T_source = trajectories[sim_id, s, :].astype(np.float32)
    mu_s = T_source.mean()
    sigma_s = T_source.std()
    T_source_norm = (T_source - mu_s) / (sigma_s + T_EPS)
    x_norm = ((x_grid - x_grid[0]) / (x_grid[-1] - x_grid[0])).astype(np.float32)

    amp, freq, _T0, R_c = sim_params[sim_id]
    amp = float(amp)
    freq = float(freq)
    R_c = float(R_c)

    x_spatial_single = np.stack([T_source_norm, x_norm], axis=-1)
    x_spatial_batch = np.tile(x_spatial_single[None, :, :], (len(target_indices), 1, 1))

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
        Y_pred_norm = model(x_tensor, c_tensor).cpu().numpy().squeeze(-1)

    Y_pred = (Y_pred_norm * (sigma_s + T_EPS) + mu_s).T
    Y_true = trajectories[sim_id, target_indices, :].T.astype(np.float32)

    return {
        "interface_x": interface_x,
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

    iface_mask = torch.from_numpy((x_grid >= 0.4) & (x_grid <= 0.6))
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

    mu_s = stats_tensor[:, 0][:, None, None]
    sigma_s = stats_tensor[:, 1][:, None, None]
    y_pred_phys = y_pred * (sigma_s + T_EPS) + mu_s
    y_true_phys = y_tensor * (sigma_s + T_EPS) + mu_s

    global_rel = (
        torch.mean((y_pred_phys - y_true_phys) ** 2, dim=(1, 2))
        / torch.clamp(torch.mean(y_true_phys ** 2, dim=(1, 2)), min=1e-12)
    ).sqrt() * 100.0
    iface_rel = (
        torch.mean((y_pred_phys[:, iface_mask, :] - y_true_phys[:, iface_mask, :]) ** 2, dim=(1, 2))
        / torch.clamp(torch.mean(y_true_phys[:, iface_mask, :] ** 2, dim=(1, 2)), min=1e-12)
    ).sqrt() * 100.0

    sim_ids = np.array([dataset._pairs[idx][0] for idx in sample_indices], dtype=np.int64)
    params = dataset.sim_params[sim_ids]

    return {
        "lead_time": np.array([dataset._lead_times[idx] for idx in sample_indices], dtype=np.float32),
        "global_rel_l2": global_rel.numpy(),
        "iface_rel_l2": iface_rel.numpy(),
        "amplitude": np.array([float(p[0]) for p in params], dtype=np.float32),
        "frequency": np.array([float(p[1]) for p in params], dtype=np.float32),
        "R_c": np.array([float(p[3]) for p in params], dtype=np.float32),
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
    t_grid: np.ndarray,
    sim_params: np.ndarray,
    config: dict,
) -> dict[str, np.ndarray]:
    """Return lead-time arrays and pair counts for split-aware coverage plots/tests."""
    datasets = _build_split_datasets(trajectories, x_grid, t_grid, sim_params, config)
    return {
        "train": datasets["train"]._lead_times.copy(),
        "val": datasets["val"]._lead_times.copy(),
        "test": datasets["test"]._lead_times.copy(),
        "train_count": np.array([len(datasets["train"])]),
        "val_count": np.array([len(datasets["val"])]),
        "test_count": np.array([len(datasets["test"])]),
    }


# ============================================================
# SECTION: PHYSICS — Multilayer FD Solver Diagnostics
# ============================================================

def plot_final_temperature(
    solver: FDSolver1D,
    save_path: Optional[Path] = None,
    title: str = "Final temperature vs spatial nodes",
):
    """Run solver to final time and plot spatial grid (x) vs final temperature (T)."""
    t, x, T_final = solver.solve(store_trajectory=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, T_final, marker="o", linestyle="-", color="C0", markersize=5)

    # mark interfaces
    for xi in solver.interface_positions:
        ax.axvline(xi, color="gray", linestyle="--", alpha=0.5)

    ax.set_xlabel("x (spatial nodes)")
    ax.set_ylabel("Temperature")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.5)

    if save_path is None:
        save_path = Path(__file__).resolve().parent / "physics" / "final_temperature.png"
    save_path = _ensure_parent(Path(save_path))
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    print(f"Saved final temperature plot to: {save_path}")
    plt.close(fig)


def plot_layer_geometry(
    solver: FDSolver1D,
    save_path: str | Path | None = None,
):
    """Layer geometry and material properties for a multilayer domain.

    3-panel layout:
        (a) Domain schematic with colored layer regions, grid nodes, interface markers
        (b) Material properties k, rho, cp vs x as step-style lines
        (c) Thermal diffusivity alpha = k/(rho*cp) vs x
    """
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(14, 4.5))

    # color palette for layers
    layer_colors = plt.cm.Pastel1(np.linspace(0, 1, max(len(solver.layers), 3)))

    # --- Panel (a): Domain schematic ---
    for j, layer in enumerate(solver.layers):
        ax_a.axvspan(layer.x_left, layer.x_right, alpha=0.4, color=layer_colors[j],
                     label=f"Layer {j} (k={layer.k})")
    for xi in solver.interface_positions:
        ax_a.axvline(xi, color="red", linestyle="--", linewidth=1.5, alpha=0.8)
    ax_a.plot(solver.grid, np.zeros(solver.N), "|", color="black", markersize=8, alpha=0.5)
    ax_a.set_xlabel("x")
    ax_a.set_title("(a) Domain Schematic")
    ax_a.set_yticks([])
    ax_a.legend(fontsize=7, loc="upper right")
    ax_a.set_xlim(solver.a, solver.b)

    # --- Panel (b): Material properties ---
    ax_b.plot(solver.grid, solver.k_nodes, drawstyle="steps-mid", label="k", linewidth=1.5)
    ax_b.plot(solver.grid, solver.rho_nodes, drawstyle="steps-mid", label=r"$\rho$", linewidth=1.5)
    ax_b.plot(solver.grid, solver.cp_nodes, drawstyle="steps-mid", label=r"$c_p$", linewidth=1.5)
    for xi in solver.interface_positions:
        ax_b.axvline(xi, color="gray", linestyle=":", alpha=0.5)
    ax_b.set_xlabel("x")
    ax_b.set_ylabel("Property value")
    ax_b.set_title(r"(b) Material Properties")
    ax_b.legend(fontsize=8)
    ax_b.grid(True, linestyle="--", alpha=0.3)

    # --- Panel (c): Thermal diffusivity ---
    alpha_nodes = solver.k_nodes / (solver.rho_nodes * solver.cp_nodes)
    ax_c.plot(solver.grid, alpha_nodes, drawstyle="steps-mid", color="C2", linewidth=1.5)
    for xi in solver.interface_positions:
        ax_c.axvline(xi, color="gray", linestyle=":", alpha=0.5)
    ax_c.set_xlabel("x")
    ax_c.set_ylabel(r"$\alpha = k / (\rho \, c_p)$")
    ax_c.set_title("(c) Thermal Diffusivity")
    ax_c.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    if save_path is None:
        save_path = Path(__file__).resolve().parent / "physics" / "layer_geometry.png"
    save_path = _ensure_parent(Path(save_path))
    fig.savefig(save_path, dpi=200)
    print(f"Saved layer geometry to: {save_path}")
    plt.close(fig)


def plot_face_conductance(
    solver: FDSolver1D,
    save_path: str | Path | None = None,
):
    """Face conductance and Crank-Nicolson coupling coefficients.

    3-panel layout:
        (a) G_face vs face position — interface faces highlighted
        (b) r_minus and r_plus vs node index (interior nodes only)
        (c) Diagonal dominance margin: 1 - r_minus - r_plus
    """
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(14, 4.5))

    interface_faces = set(solver.interface_face_map.keys())
    interior_mask = np.array([i not in interface_faces for i in range(solver.N - 1)])
    iface_mask = ~interior_mask

    # --- Panel (a): G_face ---
    ax_a.plot(solver.face_positions[interior_mask], solver.G_face[interior_mask],
              "o", color="C0", markersize=4, alpha=0.6, label="Interior faces")
    if np.any(iface_mask):
        ax_a.plot(solver.face_positions[iface_mask], solver.G_face[iface_mask],
                  "o", color="red", markersize=7, zorder=5, label="Interface faces")
    ax_a.set_xlabel("Face position")
    ax_a.set_ylabel(r"$G_{\mathrm{face}}$ (conductance / area)")
    ax_a.set_title("(a) Face Conductance")
    ax_a.legend(fontsize=8)
    ax_a.grid(True, linestyle="--", alpha=0.3)

    # --- Panel (b): r_minus, r_plus ---
    interior_idx = np.arange(1, solver.N - 1)
    ax_b.plot(interior_idx, solver.r_minus[1:-1], "-", color="C0", linewidth=1.2, label=r"$r^-$")
    ax_b.plot(interior_idx, solver.r_plus[1:-1], "-", color="C3", linewidth=1.2, label=r"$r^+$")
    for xi in solver.interface_positions:
        # convert x to node index (approximate)
        node_approx = (xi - solver.a) / solver.h
        ax_b.axvline(node_approx, color="gray", linestyle=":", alpha=0.5)
    ax_b.set_xlabel("Node index")
    ax_b.set_ylabel("CN coefficient")
    ax_b.set_title(r"(b) Local CN Coefficients $r^-$, $r^+$")
    ax_b.legend(fontsize=8)
    ax_b.grid(True, linestyle="--", alpha=0.3)

    # --- Panel (c): Diagonal dominance margin ---
    margin = 1.0 - solver.r_minus[1:-1] - solver.r_plus[1:-1]
    ax_c.fill_between(interior_idx, margin, 0, where=(margin >= 0),
                      color="green", alpha=0.3, label="Stable")
    ax_c.fill_between(interior_idx, margin, 0, where=(margin < 0),
                      color="red", alpha=0.3, label="Unstable")
    ax_c.plot(interior_idx, margin, "-", color="black", linewidth=0.8)
    ax_c.axhline(0, color="red", linestyle="--", linewidth=0.8)
    ax_c.set_xlabel("Node index")
    ax_c.set_ylabel(r"$1 - r^- - r^+$")
    ax_c.set_title("(c) Diagonal Dominance Margin")
    ax_c.legend(fontsize=8)
    ax_c.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    if save_path is None:
        save_path = Path(__file__).resolve().parent / "physics" / "face_conductance.png"
    save_path = _ensure_parent(Path(save_path))
    fig.savefig(save_path, dpi=200)
    print(f"Saved face conductance to: {save_path}")
    plt.close(fig)


def plot_multilayer_evolution(
    solver: FDSolver1D,
    T_hist: np.ndarray,
    save_path: str | Path | None = None,
):
    """Temperature evolution through a multilayer domain.

    3-panel layout:
        (a) T(x,t) heatmap with interface dashed lines
        (b) T(x) spatial profiles at selected time snapshots
        (c) T(t) at nodes flanking each interface (continuity check)
    """
    Nt = len(solver.t)
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(15, 5))

    # --- Panel (a): Heatmap ---
    T_mesh, X_mesh = np.meshgrid(solver.t, solver.grid)
    pc = ax_a.pcolormesh(T_mesh, X_mesh, T_hist.T, cmap="inferno", shading="auto")
    fig.colorbar(pc, ax=ax_a, label="Temperature", shrink=0.85)
    for xi in solver.interface_positions:
        ax_a.axhline(xi, color="white", linestyle="--", linewidth=1.0, alpha=0.8)
    ax_a.set_xlabel("Time")
    ax_a.set_ylabel("x")
    ax_a.set_title("(a) T(x, t)")

    # --- Panel (b): Spatial profiles at selected times ---
    n_snaps = 6
    snap_indices = np.linspace(0, Nt - 1, n_snaps, dtype=int)
    cmap_snap = plt.cm.viridis
    for i, t_idx in enumerate(snap_indices):
        color = cmap_snap(i / max(n_snaps - 1, 1))
        ax_b.plot(solver.grid, T_hist[t_idx, :], color=color, linewidth=1.2,
                  label=f"t={solver.t[t_idx]:.3f}")
    for xi in solver.interface_positions:
        ax_b.axvline(xi, color="gray", linestyle="--", alpha=0.5)
    ax_b.set_xlabel("x")
    ax_b.set_ylabel("Temperature")
    ax_b.set_title("(b) Spatial Profiles")
    ax_b.legend(fontsize=7, loc="best")
    ax_b.grid(True, linestyle="--", alpha=0.3)

    # --- Panel (c): Temperature at flanking nodes ---
    flanks = _interface_flanking_nodes(solver)
    for left_i, right_i, xi in flanks:
        ax_c.plot(solver.t, T_hist[:, left_i], linewidth=1.2,
                  label=f"Node {left_i} (left of x={xi:.2f})")
        ax_c.plot(solver.t, T_hist[:, right_i], "--", linewidth=1.2,
                  label=f"Node {right_i} (right of x={xi:.2f})")
    ax_c.set_xlabel("Time")
    ax_c.set_ylabel("Temperature")
    ax_c.set_title("(c) Flanking Node Temperatures")
    ax_c.legend(fontsize=7, loc="best")
    ax_c.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    if save_path is None:
        save_path = Path(__file__).resolve().parent / "physics" / "multilayer_evolution.png"
    save_path = _ensure_parent(Path(save_path))
    fig.savefig(save_path, dpi=200)
    print(f"Saved multilayer evolution to: {save_path}")
    plt.close(fig)


def plot_heat_flux_profile(
    solver: FDSolver1D,
    T_hist: np.ndarray,
    save_path: str | Path | None = None,
):
    """Numerical heat flux profiles and applied boundary forcing.

    2-panel layout:
        (a) Face flux q = -G_face * (T[i+1] - T[i]) at selected time snapshots
        (b) Applied left boundary flux q_left(t) over full simulation time
    """
    Nt = len(solver.t)
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 5))

    # --- Panel (a): Numerical flux at snapshots ---
    n_snaps = 6
    snap_indices = np.linspace(0, Nt - 1, n_snaps, dtype=int)
    # skip t=0 if IC is uniform (flux = 0 everywhere, not interesting)
    if snap_indices[0] == 0 and n_snaps > 1:
        snap_indices = snap_indices[1:]

    cmap_snap = plt.cm.viridis
    for i, t_idx in enumerate(snap_indices):
        dT = T_hist[t_idx, 1:] - T_hist[t_idx, :-1]
        q_face = -solver.G_face * dT
        color = cmap_snap(i / max(len(snap_indices) - 1, 1))
        ax_a.plot(solver.face_positions, q_face, color=color, linewidth=1.2,
                  label=f"t={solver.t[t_idx]:.3f}")

    for xi in solver.interface_positions:
        ax_a.axvline(xi, color="gray", linestyle="--", alpha=0.5)
    ax_a.set_xlabel("Face position")
    ax_a.set_ylabel("Heat flux q")
    ax_a.set_title("(a) Numerical Face Flux")
    ax_a.legend(fontsize=7, loc="best")
    ax_a.grid(True, linestyle="--", alpha=0.3)

    # --- Panel (b): Applied boundary flux ---
    q_left_vals = np.array([solver.q_left(ti) for ti in solver.t])
    ax_b.plot(solver.t, q_left_vals, color="C3", linewidth=1.2)
    ax_b.axvline(solver.t_on, color="green", linestyle=":", alpha=0.7, label=f"t_on={solver.t_on}")
    ax_b.axvline(solver.t_off, color="red", linestyle=":", alpha=0.7, label=f"t_off={solver.t_off}")
    ax_b.set_xlabel("Time")
    ax_b.set_ylabel("q_left(t)")
    ax_b.set_title("(b) Applied Boundary Flux")
    ax_b.legend(fontsize=8)
    ax_b.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    if save_path is None:
        save_path = Path(__file__).resolve().parent / "physics" / "heat_flux_profile.png"
    save_path = _ensure_parent(Path(save_path))
    fig.savefig(save_path, dpi=200)
    print(f"Saved heat flux profile to: {save_path}")
    plt.close(fig)


# ============================================================
# SECTION: MMS — Method of Manufactured Solutions
# ============================================================

def plot_mms_convergence(save_path: str | Path | None = None):
    """Run MMS at multiple refinement levels and plot L2 error convergence in log-log space.

    Left panel: spatial convergence (varying dx, fixed dt)
    Right panel: temporal convergence (varying dt, fixed N)
    Each panel includes data points, linear regression fit, and ideal O(h^2) reference.
    """
    from src.physics.mms_1d import run_mms_once

    # --- spatial convergence (fix dt small enough that temporal error is negligible) ---
    N_list = [26, 51, 101, 201, 401]
    fixed_dt = 0.0005
    h_vals, h_l2_errors = [], []
    for N in N_list:
        h, _, _, l2 = run_mms_once(N, dt=fixed_dt)
        h_vals.append(h)
        h_l2_errors.append(l2)
    h_vals = np.array(h_vals)
    h_l2_errors = np.array(h_l2_errors)

    # --- temporal convergence (fix N fine enough that spatial error is negligible) ---
    dt_list = [0.04, 0.02, 0.01, 0.005, 0.0025]
    fixed_N = 801
    dt_vals, dt_l2_errors = [], []
    for dt in dt_list:
        _, dt_val, _, l2 = run_mms_once(fixed_N, dt=dt)
        dt_vals.append(dt_val)
        dt_l2_errors.append(l2)
    dt_vals = np.array(dt_vals)
    dt_l2_errors = np.array(dt_l2_errors)

    fig, (ax_x, ax_t) = plt.subplots(1, 2, figsize=(12, 5))

    # helper for log-log regression + plotting
    def _plot_convergence(ax, refinement, errors, xlabel, title):
        log_r = np.log10(refinement)
        log_e = np.log10(errors)

        # linear regression in log-log
        coeffs = np.polyfit(log_r, log_e, 1)
        slope = coeffs[0]
        fit_line = 10 ** np.polyval(coeffs, log_r)

        ax.loglog(refinement, errors, "ko", markersize=7, label="MMS data")
        ax.loglog(refinement, fit_line, "C0-", linewidth=1.5, label=f"Fit: slope = {slope:.2f}")

        # ideal O(h^2) reference
        ref = errors[-1] * (refinement / refinement[-1]) ** 2
        ax.loglog(refinement, ref, "k--", alpha=0.4, linewidth=1, label="$O(h^2)$ reference")

        ax.set_xlabel(xlabel)
        ax.set_ylabel("L2 error")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, which="both", linestyle="--", alpha=0.3)

    _plot_convergence(ax_x, h_vals, h_l2_errors, r"$\Delta x$", f"Spatial Convergence (dt={fixed_dt})")
    _plot_convergence(ax_t, dt_vals, dt_l2_errors, r"$\Delta t$", f"Temporal Convergence (N={fixed_N})")

    fig.suptitle("MMS Convergence — Crank-Nicolson FD Solver", fontsize=13)
    fig.tight_layout()
    if save_path is None:
        save_path = Path(__file__).resolve().parent / "mms" / "mms_convergence.png"
    save_path = _ensure_parent(Path(save_path))
    fig.savefig(save_path, dpi=200)
    print(f"Saved MMS convergence to: {save_path}")
    plt.close(fig)


def plot_mms_order_estimation(save_path: str | Path | None = None):
    """Estimate spatial and temporal order from successive refinement pairs across 5+ levels.

    Left panel: estimated spatial order p_x vs dx
    Right panel: estimated temporal order p_t vs dt
    Horizontal reference line at p=2 (expected for Crank-Nicolson).
    """
    from src.physics.mms_1d import run_mms_once

    # --- spatial order estimation (fix dt small enough for spatial error to dominate) ---
    N_list = [26, 51, 101, 201, 401]
    fixed_dt = 0.0005
    h_vals, h_l2_errors = [], []
    for N in N_list:
        h, _, _, l2 = run_mms_once(N, dt=fixed_dt)
        h_vals.append(h)
        h_l2_errors.append(l2)

    # pairwise order from successive refinements: p = log(e1/e2) / log(h1/h2)
    px_vals, px_h_midpoints = [], []
    for i in range(len(h_vals) - 1):
        p = np.log(h_l2_errors[i] / h_l2_errors[i + 1]) / np.log(h_vals[i] / h_vals[i + 1])
        px_vals.append(p)
        px_h_midpoints.append(np.sqrt(h_vals[i] * h_vals[i + 1]))  # geometric midpoint

    # --- temporal order estimation (fix N fine enough for temporal error to dominate) ---
    dt_list = [0.04, 0.02, 0.01, 0.005, 0.0025]
    fixed_N = 801
    dt_vals, dt_l2_errors = [], []
    for dt in dt_list:
        _, dt_val, _, l2 = run_mms_once(fixed_N, dt=dt)
        dt_vals.append(dt_val)
        dt_l2_errors.append(l2)

    pt_vals, pt_dt_midpoints = [], []
    for i in range(len(dt_vals) - 1):
        p = np.log(dt_l2_errors[i] / dt_l2_errors[i + 1]) / np.log(dt_vals[i] / dt_vals[i + 1])
        pt_vals.append(p)
        pt_dt_midpoints.append(np.sqrt(dt_vals[i] * dt_vals[i + 1]))

    fig, (ax_x, ax_t) = plt.subplots(1, 2, figsize=(12, 5))

    # spatial order
    ax_x.plot(px_h_midpoints, px_vals, "ko-", markersize=7)
    ax_x.axhline(2.0, color="C3", linestyle="--", alpha=0.6, label="Expected order = 2")
    ax_x.set_xscale("log")
    ax_x.set_xlabel(r"$\Delta x$ (geometric midpoint)")
    ax_x.set_ylabel("Estimated order $p_x$")
    ax_x.set_title(f"Spatial Order Estimation (dt={fixed_dt})")
    ax_x.legend(fontsize=8)
    ax_x.grid(True, linestyle="--", alpha=0.3)
    ax_x.set_ylim(0, 4)

    # temporal order
    ax_t.plot(pt_dt_midpoints, pt_vals, "ko-", markersize=7)
    ax_t.axhline(2.0, color="C3", linestyle="--", alpha=0.6, label="Expected order = 2")
    ax_t.set_xscale("log")
    ax_t.set_xlabel(r"$\Delta t$ (geometric midpoint)")
    ax_t.set_ylabel("Estimated order $p_t$")
    ax_t.set_title(f"Temporal Order Estimation (N={fixed_N})")
    ax_t.legend(fontsize=8)
    ax_t.grid(True, linestyle="--", alpha=0.3)
    ax_t.set_ylim(0, 4)

    fig.suptitle("MMS Order Estimation — Successive Refinement Pairs", fontsize=13)
    fig.tight_layout()
    if save_path is None:
        save_path = Path(__file__).resolve().parent / "mms" / "mms_order_estimation.png"
    save_path = _ensure_parent(Path(save_path))
    fig.savefig(save_path, dpi=200)
    print(f"Saved MMS order estimation to: {save_path}")
    plt.close(fig)


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

    # --- build figure: 3 panels if iface data present, 2 otherwise ---
    if has_iface:
        fig, (ax_top, ax_mid, ax_bot) = plt.subplots(
            3, 1, figsize=(10, 10), gridspec_kw={"height_ratios": [3, 2, 1]})
    else:
        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [3, 1]})

    # --- top panel: train MSE (left y) + rel L2 metrics (right y) ---
    color_train = "C0"
    color_val = "C3"

    ax_top.semilogy(epochs, train_losses, color=color_train, alpha=0.7, linewidth=0.8, label="Train MSE")
    ax_top.set_xlabel("Epoch")
    ax_top.set_ylabel("Train MSE", color=color_train)
    ax_top.tick_params(axis="y", labelcolor=color_train)

    ax_val = ax_top.twinx()
    if has_train_rel_l2:
        ax_val.semilogy(epochs, train_rel_l2s, color=color_val, alpha=0.3, linewidth=0.6, label="Train rel. L2 (%)")
    ax_val.semilogy(val_epochs, val_values, color=color_val, linewidth=1.2, label="Val rel. L2 (%)")
    if len(best_epochs) > 0 and len(best_vals) > 0:
        ax_val.scatter(best_epochs, best_vals, marker="*", s=60, color="gold", zorder=5, edgecolors="k", linewidths=0.5, label="Best checkpoint")
    ax_val.set_ylabel("Rel. L2 (%)", color=color_val)
    ax_val.tick_params(axis="y", labelcolor=color_val)

    # combine legends
    lines_1, labels_1 = ax_top.get_legend_handles_labels()
    lines_2, labels_2 = ax_val.get_legend_handles_labels()
    ax_val.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right", fontsize=8)
    ax_top.set_title("Training Curves")
    ax_top.grid(True, linestyle="--", alpha=0.3)

    # --- middle panel: interface rel L2 (only if iface data present) ---
    if has_iface:
        color_iface_train = "C4"  # purple
        color_iface_val = "C1"    # orange

        ax_mid.semilogy(epochs, train_iface_rel_l2s, color=color_iface_train, alpha=0.3,
                         linewidth=0.6, label="Train iface rel. L2 (%)")
        ax_mid.semilogy(val_iface_epochs, val_iface_values, color=color_iface_val,
                         linewidth=1.2, label="Val iface rel. L2 (%)")
        if len(best_epochs) > 0 and len(best_iface_vals) > 0:
            ax_mid.scatter(best_epochs, best_iface_vals, marker="*", s=60, color="gold",
                           zorder=5, edgecolors="k", linewidths=0.5, label="Best checkpoint")
        ax_mid.set_xlabel("Epoch")
        ax_mid.set_ylabel("Interface Rel. L2 (%)")
        ax_mid.set_title("Interface Region Error")
        ax_mid.legend(fontsize=8, loc="upper right")
        ax_mid.grid(True, linestyle="--", alpha=0.3)

    # --- bottom panel: learning rate ---
    ax_bot.semilogy(epochs, lrs, color="C2", linewidth=1.0)
    ax_bot.set_xlabel("Epoch")
    ax_bot.set_ylabel("Learning rate")
    ax_bot.set_title("Learning Rate Schedule")
    ax_bot.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    if save_path is None:
        save_path = csv_path.parent / "training_curves.png"
    save_path = _ensure_parent(Path(save_path))
    fig.savefig(save_path, dpi=200)
    print(f"Saved training curves to: {save_path}")
    plt.close(fig)


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

    x = np.arange(len(seeds))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, val_losses, width, label="Val rel. L2 (%)", color="C0", alpha=0.8)
    ax.bar(x, test_losses, width, label="Test rel. L2 (%)", color="C3", alpha=0.8)
    ax.bar(x + width, test_iface_losses, width, label="Test iface rel. L2 (%)", color="C1", alpha=0.8)

    # mean lines
    ax.axhline(summary["best_val_loss_mean"], color="C0", linestyle="--", alpha=0.6, label=f"Val mean: {summary['best_val_loss_mean']:.3f}")
    ax.axhline(summary["test_rel_l2_mean"], color="C3", linestyle="--", alpha=0.6, label=f"Test mean: {summary['test_rel_l2_mean']:.3f}")
    ax.axhline(summary["test_iface_rel_l2_mean"], color="C1", linestyle="--", alpha=0.6, label=f"Iface mean: {summary['test_iface_rel_l2_mean']:.3f}")

    ax.set_xlabel("Seed")
    ax.set_ylabel("Relative L2 Error (%)")
    ax.set_title("Model Performance Across Seeds")
    ax.set_xticks(x)
    ax.set_xticklabels(seeds)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    if save_path is None:
        save_path = report_path.parent / "seed_comparison.png"
    fig.savefig(save_path, dpi=200)
    print(f"Saved seed comparison to: {save_path}")
    plt.close(fig)


# ============================================================
# SECTION: DATA — Dataset & Parameter Space Visualization
# ============================================================

def plot_prediction_vs_truth(
    model,
    trajectories: np.ndarray,
    x_grid: np.ndarray,
    t_grid: np.ndarray,
    sim_params: np.ndarray,
    sim_id: int,
    s: int,
    n_steps: int = 40,
    save_path: str | Path | None = None,
):
    """Plot ground truth, FNO prediction, pointwise error, and interface cross-section.

    Uses the time-conditioned FNO1d: for a given source snapshot at time s,
    predict n_steps future target times and assemble into a space-time field.

    4-panel layout:
        (a) Ground truth heatmap with interface marker
        (b) FNO prediction heatmap with interface marker
        (c) Absolute error heatmap with interface marker
        (d) T(x) cross-section at mid-horizon timestep (truth vs prediction)
    """
    Nt = len(t_grid)

    # evenly-spaced target indices after source time s
    target_indices = np.linspace(s + 1, Nt - 1, n_steps, dtype=int)
    case = _prepare_prediction_case(model, trajectories, x_grid, t_grid, sim_params, sim_id, s, target_indices)
    t_targets = case["t_targets"]
    Y_pred = case["Y_pred"]
    Y_true = case["Y_true"]
    error = np.abs(Y_pred - Y_true)

    # meshgrid for pcolormesh
    T_mesh, X_mesh = np.meshgrid(t_targets, x_grid)

    vmin = min(Y_true.min(), Y_pred.min())
    vmax = max(Y_true.max(), Y_pred.max())

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))

    # ground truth
    axes[0].pcolormesh(T_mesh, X_mesh, Y_true, cmap="inferno", vmin=vmin, vmax=vmax, shading="auto")
    axes[0].axhline(case["interface_x"], color="white", linestyle="--", linewidth=1.0, alpha=0.8)
    axes[0].set_title("Ground Truth")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("x")

    # prediction
    pc1 = axes[1].pcolormesh(T_mesh, X_mesh, Y_pred, cmap="inferno", vmin=vmin, vmax=vmax, shading="auto")
    axes[1].axhline(case["interface_x"], color="white", linestyle="--", linewidth=1.0, alpha=0.8)
    axes[1].set_title("FNO Prediction")
    axes[1].set_xlabel("Time")

    # shared colorbar for truth/pred
    fig.colorbar(pc1, ax=axes[:2].tolist(), label="Temperature", shrink=0.85)

    # error
    pc2 = axes[2].pcolormesh(T_mesh, X_mesh, error, cmap="Reds", shading="auto")
    axes[2].axhline(case["interface_x"], color="black", linestyle="--", linewidth=1.0, alpha=0.8)
    axes[2].set_title("Absolute Error")
    axes[2].set_xlabel("Time")
    fig.colorbar(pc2, ax=axes[2], label="|Error|", shrink=0.85)

    # cross-section at mid-horizon
    mid_h = n_steps // 2
    axes[3].plot(x_grid, Y_true[:, mid_h], "k-", linewidth=1.5, label="Truth")
    axes[3].plot(x_grid, Y_pred[:, mid_h], "r--", linewidth=1.5, label="Prediction")
    axes[3].axvline(case["interface_x"], color="gray", linestyle=":", linewidth=1.5, alpha=0.7, label="Interface")
    axes[3].set_xlabel("x")
    axes[3].set_ylabel("Temperature")
    axes[3].set_title(f"T(x) at t={t_targets[mid_h]:.3f}")
    axes[3].legend(fontsize=8)
    axes[3].grid(True, linestyle="--", alpha=0.3)

    fig.suptitle(
        f"Prediction vs Truth — Sim {sim_id}, t_s={case['source_time']:.3f}, "
        f"A={case['amp']:.0f}, f={case['freq']:.2f}, R_c={case['R_c']:.2f}",
        fontsize=12,
    )
    fig.tight_layout()

    if save_path is None:
        save_path = Path(__file__).resolve().parent / "data" / "prediction_vs_truth.png"
    save_path = _ensure_parent(Path(save_path))
    fig.savefig(save_path, dpi=200)
    print(f"Saved prediction vs truth to: {save_path}")
    plt.close(fig)


def plot_trajectory_heatmap(
    trajectories: np.ndarray,
    sim_id: int,
    x_grid: np.ndarray,
    t_grid: np.ndarray,
    save_path: str | Path | None = None,
):
    """Plot a single simulation's full temperature evolution as a 2D heatmap."""
    T = trajectories[sim_id]  # (Nt, Nx)
    T_mesh, X_mesh = np.meshgrid(t_grid, x_grid)

    fig, ax = plt.subplots(figsize=(10, 5))
    pc = ax.pcolormesh(T_mesh, X_mesh, T.T, cmap="inferno", shading="auto")
    fig.colorbar(pc, ax=ax, label="Temperature")
    ax.set_xlabel("Time")
    ax.set_ylabel("x")
    ax.set_title(f"Temperature Evolution — Simulation {sim_id}")

    fig.tight_layout()
    if save_path is None:
        save_path = Path(__file__).resolve().parent / "data" / "trajectory_heatmap.png"
    save_path = _ensure_parent(Path(save_path))
    fig.savefig(save_path, dpi=200)
    print(f"Saved trajectory heatmap to: {save_path}")
    plt.close(fig)


def plot_initial_conditions(
    trajectories: np.ndarray,
    x_grid: np.ndarray,
    n_samples: int = 20,
    seed: int = 42,
    save_path: str | Path | None = None,
):
    """Plot overlaid initial conditions T(x, t=0) for n randomly selected simulations."""
    rng = np.random.default_rng(seed)
    num_sims = trajectories.shape[0]
    selected = rng.choice(num_sims, size=min(n_samples, num_sims), replace=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.cm.viridis
    for i, sim_id in enumerate(sorted(selected)):
        color = cmap(i / max(len(selected) - 1, 1))
        ax.plot(x_grid, trajectories[sim_id, 0, :], color=color, alpha=0.7, linewidth=0.8)

    ax.set_xlabel("x")
    ax.set_ylabel("Temperature")
    ax.set_title(f"Initial Conditions T(x, t=0) — {len(selected)} simulations")
    ax.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    if save_path is None:
        save_path = Path(__file__).resolve().parent / "data" / "initial_conditions.png"
    save_path = _ensure_parent(Path(save_path))
    fig.savefig(save_path, dpi=200)
    print(f"Saved initial conditions plot to: {save_path}")
    plt.close(fig)


def plot_lhs_scatter(
    sim_params: np.ndarray,
    save_path: str | Path | None = None,
):
    """Pairwise scatter overview of sampled conditioning parameters."""
    amplitudes = np.array([p[0] for p in sim_params], dtype=np.float32)
    frequencies = np.array([p[1] for p in sim_params], dtype=np.float32)
    contact_resistance = np.array([p[3] for p in sim_params], dtype=np.float32)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    scatter_specs = [
        (axes[0], amplitudes, frequencies, contact_resistance, "Flux Amplitude (A)", "Flux Frequency (f)", "R_c"),
        (axes[1], amplitudes, contact_resistance, frequencies, "Flux Amplitude (A)", "Contact Resistance (R_c)", "f"),
        (axes[2], frequencies, contact_resistance, amplitudes, "Flux Frequency (f)", "Contact Resistance (R_c)", "A"),
    ]

    for ax, x_vals, y_vals, colors, x_label, y_label, color_label in scatter_specs:
        sc = ax.scatter(x_vals, y_vals, c=colors, s=14, alpha=0.7, cmap="viridis", edgecolors="none")
        fig.colorbar(sc, ax=ax, label=color_label)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if x_label.endswith("(f)"):
            ax.set_xscale("log")
        ax.grid(True, linestyle="--", alpha=0.3)

    fig.suptitle(f"Conditioning Parameter Coverage — {len(amplitudes)} simulations")

    fig.tight_layout()
    if save_path is None:
        save_path = Path(__file__).resolve().parent / "data" / "lhs_scatter.png"
    save_path = _ensure_parent(Path(save_path))
    fig.savefig(save_path, dpi=200)
    print(f"Saved LHS scatter to: {save_path}")
    plt.close(fig)


def plot_flux_profiles(
    t_on: float = 0.0,
    t_off: float = 0.2,
    t_final: float = 1.0,
    save_path: str | Path | None = None,
):
    """Plot q(t) for representative (A, f) combos at corners and center of the parameter space."""
    from src.physics.fd_solver_1d import windowed_sin_flux

    combos = [
        (50.0, 1.0, "A=50, f=1 (low-low)"),
        (50.0, 20.0, "A=50, f=20 (low-high)"),
        (300.0, 1.0, "A=300, f=1 (high-low)"),
        (300.0, 20.0, "A=300, f=20 (high-high)"),
        (175.0, 10.5, "A=175, f=10.5 (center)"),
    ]

    t = np.linspace(0.0, t_final, 2000)

    fig, ax = plt.subplots(figsize=(10, 5))
    for A, f, label in combos:
        q_fn = windowed_sin_flux(f, A, t_on, t_off, tukey_alpha=0.5)
        q_vals = np.array([q_fn(ti) for ti in t])
        ax.plot(t, q_vals, linewidth=1.2, label=label)

    ax.set_xlabel("Time")
    ax.set_ylabel("Heat Flux q(t)")
    ax.set_title(f"Windowed Sinusoidal Flux Profiles (t_on={t_on}, t_off={t_off})")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    if save_path is None:
        save_path = Path(__file__).resolve().parent / "data" / "flux_profiles.png"
    save_path = _ensure_parent(Path(save_path))
    fig.savefig(save_path, dpi=200)
    print(f"Saved flux profiles to: {save_path}")
    plt.close(fig)


def plot_trajectory_comparison_grid(
    trajectories: np.ndarray,
    sim_params: np.ndarray,
    x_grid: np.ndarray,
    t_grid: np.ndarray,
    save_path: str | Path | None = None,
):
    """2x2 grid of trajectory heatmaps at the parameter space corners."""
    amplitudes = np.array([p[0] for p in sim_params], dtype=np.float32)
    frequencies = np.array([p[1] for p in sim_params], dtype=np.float32)

    # find sims closest to each corner
    corners = [
        ("Low A, Low f", 50.0, 1.0),
        ("Low A, High f", 50.0, 20.0),
        ("High A, Low f", 300.0, 1.0),
        ("High A, High f", 300.0, 20.0),
    ]

    T_mesh, X_mesh = np.meshgrid(t_grid, x_grid)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)

    for ax, (label, target_A, target_f) in zip(axes.flat, corners):
        dist = (amplitudes - target_A) ** 2 + (frequencies - target_f) ** 2
        sim_id = int(np.argmin(dist))
        T = trajectories[sim_id]  # (Nt, Nx)

        pc = ax.pcolormesh(T_mesh, X_mesh, T.T, cmap="inferno", shading="auto")
        fig.colorbar(pc, ax=ax, shrink=0.8)
        ax.set_title(f"{label}\nSim {sim_id}: A={amplitudes[sim_id]:.0f}, f={frequencies[sim_id]:.1f}")
        ax.set_xlabel("Time")
        ax.set_ylabel("x")

    fig.suptitle("Trajectory Comparison — Parameter Space Corners", fontsize=13)
    fig.tight_layout()
    if save_path is None:
        save_path = Path(__file__).resolve().parent / "data" / "trajectory_comparison_grid.png"
    save_path = _ensure_parent(Path(save_path))
    fig.savefig(save_path, dpi=200)
    print(f"Saved trajectory comparison grid to: {save_path}")
    plt.close(fig)


def plot_boundary_temperature(
    trajectories: np.ndarray,
    sim_params: np.ndarray,
    t_grid: np.ndarray,
    n_samples: int = 20,
    color_by: str = "amplitude",
    seed: int = 42,
    save_path: str | Path | None = None,
):
    """Plot T(x=0, t) for multiple simulations, colored by amplitude or frequency."""
    amplitudes = np.array([p[0] for p in sim_params], dtype=np.float32)
    frequencies = np.array([p[1] for p in sim_params], dtype=np.float32)

    rng = np.random.default_rng(seed)
    num_sims = trajectories.shape[0]
    selected = rng.choice(num_sims, size=min(n_samples, num_sims), replace=False)

    color_vals = amplitudes[selected] if color_by == "amplitude" else frequencies[selected]
    color_label = "Amplitude (A)" if color_by == "amplitude" else "Frequency (f)"

    fig, ax = plt.subplots(figsize=(10, 5))
    norm = plt.Normalize(vmin=color_vals.min(), vmax=color_vals.max())
    cmap = plt.cm.viridis

    for i, sim_id in enumerate(selected):
        T_boundary = trajectories[sim_id, :, 0]  # T(x=0, t)
        ax.plot(t_grid, T_boundary, color=cmap(norm(color_vals[i])), alpha=0.7, linewidth=0.8)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, ax=ax, label=color_label)

    ax.set_xlabel("Time")
    ax.set_ylabel("Temperature at x=0")
    ax.set_title(f"Boundary Temperature Response — {len(selected)} simulations (colored by {color_by})")
    ax.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    if save_path is None:
        save_path = Path(__file__).resolve().parent / "data" / "boundary_temperature.png"
    save_path = _ensure_parent(Path(save_path))
    fig.savefig(save_path, dpi=200)
    print(f"Saved boundary temperature to: {save_path}")
    plt.close(fig)


def plot_parameter_response(
    trajectories: np.ndarray,
    sim_params: np.ndarray,
    save_path: str | Path | None = None,
):
    """2D scatter of (amplitude, frequency) colored by peak boundary temperature."""
    amplitudes = np.array([p[0] for p in sim_params], dtype=np.float32)
    frequencies = np.array([p[1] for p in sim_params], dtype=np.float32)

    # response metric: peak temperature at x=0
    peak_T = trajectories[:, :, 0].max(axis=1)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(amplitudes, frequencies, c=peak_T, s=15, cmap="inferno", alpha=0.8, edgecolors="none")
    fig.colorbar(sc, ax=ax, label="Peak T at x=0")

    ax.set_xlabel("Flux Amplitude (A)")
    ax.set_ylabel("Flux Frequency (f)")
    ax.set_title("Parameter–Response: Peak Boundary Temperature")
    ax.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    if save_path is None:
        save_path = Path(__file__).resolve().parent / "data" / "parameter_response.png"
    save_path = _ensure_parent(Path(save_path))
    fig.savefig(save_path, dpi=200)
    print(f"Saved parameter-response to: {save_path}")
    plt.close(fig)


def plot_snapshot_pair_samples(
    trajectories: np.ndarray,
    x_grid: np.ndarray,
    t_grid: np.ndarray,
    sim_params: np.ndarray,
    config: dict,
    n_samples: int = 3,
    interface_x: float = 0.5,
    seed: int = 42,
    save_path: str | Path | None = None,
):
    """Plot sampled source/target snapshot pairs from train/val/test splits.

    Layout: 3 rows (train / val / test) × n_samples columns.
    Each panel shows the source snapshot and one future target snapshot from the
    actual all-to-all pair dataset, with conditioning values annotated in-place.
    """
    datasets = _build_split_datasets(trajectories, x_grid, t_grid, sim_params, config)
    rng = np.random.default_rng(seed)
    splits = [("Train", datasets["train"]), ("Val", datasets["val"]), ("Test", datasets["test"])]

    fig, axes = plt.subplots(3, n_samples, figsize=(5 * n_samples, 11), squeeze=False)

    for row, (split_name, dataset) in enumerate(splits):
        if len(dataset) == 0:
            for col in range(n_samples):
                axes[row, col].set_axis_off()
            continue

        sample_size = min(n_samples, len(dataset))
        chosen_indices = rng.choice(len(dataset), size=sample_size, replace=False)

        for col in range(n_samples):
            ax = axes[row, col]
            if col >= sample_size:
                ax.set_axis_off()
                continue
            pair_idx = int(chosen_indices[col])
            sim_id, s, j = dataset._pairs[pair_idx]
            lead_time = dataset._lead_times[pair_idx]
            source = trajectories[sim_id, s, :]
            target = trajectories[sim_id, j, :]
            amp, freq, _T0, R_c = sim_params[sim_id]

            ax.plot(x_grid, source, color="black", linewidth=1.4, label=f"source t={t_grid[s]:.3f}")
            ax.plot(x_grid, target, color="C3", linewidth=1.4, linestyle="--", label=f"target t={t_grid[j]:.3f}")
            ax.axvline(interface_x, color="gray", linestyle=":", linewidth=1.2, alpha=0.8)

            text = f"Δt={lead_time:.3f}\nA={float(amp):.0f}\nf={float(freq):.2f}\nR_c={float(R_c):.2f}"
            ax.text(
                0.03,
                0.97,
                text,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
            )
            ax.set_title(f"{split_name} | Sim {sim_id}", fontsize=9)
            ax.set_xlabel("x")
            ax.set_ylabel("Temperature")
            ax.legend(fontsize=6, loc="best")
            ax.grid(True, linestyle="--", alpha=0.3)

    fig.suptitle("Snapshot Pair Samples — All-to-All Conditioning View", fontsize=13)
    fig.tight_layout()
    if save_path is None:
        save_path = Path(__file__).resolve().parent / "data" / "snapshot_pair_samples.png"
    save_path = _ensure_parent(Path(save_path))
    fig.savefig(save_path, dpi=200)
    print(f"Saved snapshot pair samples to: {save_path}")
    plt.close(fig)


def plot_interface_error(
    model,
    trajectories: np.ndarray,
    x_grid: np.ndarray,
    t_grid: np.ndarray,
    sim_params: np.ndarray,
    sim_ids: np.ndarray,
    interface_x: float = 0.5,
    n_samples: int = 4,
    n_targets: int = 20,
    seed: int = 42,
    save_path: str | Path | None = None,
):
    """Diagnose whether the model captures the temperature jump at the interface.

    Uses the time-conditioned FNO1d: for a given source snapshot, predict
    multiple future target times and compare with ground truth.

    Layout: n_samples rows × 3 columns.
        Left:   T(x) cross-sections near interface (colored=truth, red dashed=pred).
        Middle: Residual (pred − truth) zoomed to interface — reveals error structure.
        Right:  Jump magnitude ΔT = T[right_node] - T[left_node] over target times.
    """
    Nt = len(t_grid)

    # find flanking nodes for the interface
    iface_idx = int(np.argmin(np.abs(x_grid - interface_x)))
    left_node = iface_idx - 1 if x_grid[iface_idx] >= interface_x else iface_idx
    right_node = left_node + 1

    rng = np.random.default_rng(seed)
    chosen_sims = rng.choice(sim_ids, size=min(n_samples, len(sim_ids)), replace=False)

    fig, axes = plt.subplots(n_samples, 3, figsize=(20, 4 * n_samples), squeeze=False)
    cmap_snap = plt.cm.viridis
    n_snaps = 5  # number of snapshots shown in cross-section panels

    # zoom region for cross-section
    zoom_mask = (x_grid >= 0.3) & (x_grid <= 0.7)
    x_zoom = x_grid[zoom_mask]

    # interface region for interface-specific rel L2
    iface_mask = (x_grid >= 0.4) & (x_grid <= 0.6)

    for row_idx, sim_id in enumerate(chosen_sims):
        # pick a random source time (leave room for targets after it)
        max_s = Nt - 2  # need at least one target after source
        s = int(rng.integers(0, max_s + 1))

        # evenly-spaced target indices after s
        target_indices = np.linspace(s + 1, Nt - 1, n_targets, dtype=int)
        case = _prepare_prediction_case(model, trajectories, x_grid, t_grid, sim_params, sim_id, s, target_indices)
        t_targets = case["t_targets"]
        Y_pred = case["Y_pred"]
        Y_true = case["Y_true"]
        amp = case["amp"]
        freq = case["freq"]
        R_c = case["R_c"]

        # --- Left panel: T(x) overlay (truth vs pred) zoomed to interface ---
        ax_left = axes[row_idx, 0]
        snap_indices = np.linspace(0, n_targets - 1, n_snaps, dtype=int)

        for i, h_idx in enumerate(snap_indices):
            color = cmap_snap(i / max(n_snaps - 1, 1))
            ax_left.plot(x_zoom, Y_true[zoom_mask, h_idx], color=color, linewidth=2.0,
                         label=f"t={t_targets[h_idx]:.3f}")
            ax_left.plot(x_zoom, Y_pred[zoom_mask, h_idx], color="red", linewidth=1.0,
                         linestyle="--", alpha=0.9)

        ax_left.axvline(interface_x, color="red", linestyle=":", linewidth=1.5, alpha=0.7)
        ax_left.set_xlabel("x")
        ax_left.set_ylabel("Temperature")
        ax_left.set_title(f"Sim {sim_id} | A={amp:.0f}, f={freq:.1f}, R_c={R_c:.2f}, s={s}\n"
                          f"Colored=truth, red dashed=pred", fontsize=9)
        ax_left.legend(fontsize=6, loc="best")
        ax_left.grid(True, linestyle="--", alpha=0.3)

        # --- Middle panel: residual (pred − truth) near interface ---
        ax_mid = axes[row_idx, 1]

        for i, h_idx in enumerate(snap_indices):
            color = cmap_snap(i / max(n_snaps - 1, 1))
            residual = Y_pred[zoom_mask, h_idx] - Y_true[zoom_mask, h_idx]
            ax_mid.plot(x_zoom, residual, color=color, linewidth=1.5,
                        label=f"t={t_targets[h_idx]:.3f}")

        ax_mid.axvline(interface_x, color="red", linestyle=":", linewidth=1.5, alpha=0.7)
        ax_mid.axhline(0, color="gray", linestyle=":", alpha=0.5)
        ax_mid.set_xlabel("x")
        ax_mid.set_ylabel("Pred − Truth")
        ax_mid.set_title("Residual at interface region", fontsize=9)
        ax_mid.legend(fontsize=6, loc="best")
        ax_mid.grid(True, linestyle="--", alpha=0.3)

        # --- Right panel: jump magnitude over time ---
        ax_right = axes[row_idx, 2]
        true_jump = Y_true[right_node, :] - Y_true[left_node, :]
        pred_jump = Y_pred[right_node, :] - Y_pred[left_node, :]

        ax_right.plot(t_targets, true_jump, "k-", linewidth=1.5, label="True ΔT")
        ax_right.plot(t_targets, pred_jump, "r--", linewidth=1.5, label="Pred ΔT")
        ax_right.axhline(0, color="gray", linestyle=":", alpha=0.5)
        ax_right.set_xlabel("Time")
        ax_right.set_ylabel("ΔT (right − left)")

        # compute interface-specific metrics
        global_rel_l2 = (np.mean((Y_pred - Y_true) ** 2) / np.mean(Y_true ** 2)) ** 0.5 * 100
        iface_pred = Y_pred[iface_mask, :]
        iface_true = Y_true[iface_mask, :]
        iface_rel_l2 = (np.mean((iface_pred - iface_true) ** 2) /
                        np.mean(iface_true ** 2)) ** 0.5 * 100
        jump_rel_err = (np.mean((pred_jump - true_jump) ** 2) /
                        max(np.mean(true_jump ** 2), 1e-12)) ** 0.5 * 100

        ax_right.set_title(f"Global L2: {global_rel_l2:.3f}% | "
                           f"Iface L2: {iface_rel_l2:.3f}% | "
                           f"Jump err: {jump_rel_err:.1f}%", fontsize=9)
        ax_right.legend(fontsize=8)
        ax_right.grid(True, linestyle="--", alpha=0.3)

    fig.suptitle("Interface Diagnostic — Prediction vs Truth at x=0.5", fontsize=13)
    fig.tight_layout()
    if save_path is None:
        save_path = Path(__file__).resolve().parent / "data" / "interface_error.png"
    save_path = _ensure_parent(Path(save_path))
    fig.savefig(save_path, dpi=200)
    print(f"Saved interface error diagnostics to: {save_path}")
    plt.close(fig)


def plot_lead_time_coverage(
    trajectories: np.ndarray,
    x_grid: np.ndarray,
    t_grid: np.ndarray,
    sim_params: np.ndarray,
    config: dict,
    bins: int = 20,
    save_path: str | Path | None = None,
):
    """Visualize train/test pair coverage across lead times and curriculum cutoffs."""
    coverage = _lead_time_coverage_counts(trajectories, x_grid, t_grid, sim_params, config)
    train_leads = coverage["train"]
    test_leads = coverage["test"]

    fig, (ax_hist, ax_frac) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [3, 1]})
    ax_hist.hist(train_leads, bins=bins, alpha=0.65, color="C0", label=f"Train pairs ({len(train_leads)})")
    ax_hist.hist(test_leads, bins=bins, alpha=0.5, color="C3", label=f"Test pairs ({len(test_leads)})")
    ax_hist.set_xlabel("Lead time Δt")
    ax_hist.set_ylabel("Pair count")
    ax_hist.set_title("Lead-Time Coverage for All-to-All Snapshot Pairs")
    ax_hist.legend()
    ax_hist.grid(True, linestyle="--", alpha=0.3)

    warmup_epochs = int(config.get("training", {}).get("curriculum_warmup", 0))
    max_lead = float(train_leads.max()) if len(train_leads) > 0 else float(t_grid[-1] - t_grid[0])
    fractions = np.array([0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    cutoff_positions = fractions * max_lead
    for frac, cutoff in zip(fractions, cutoff_positions):
        epoch_label = int(round(frac * warmup_epochs)) if warmup_epochs > 0 else 0
        ax_hist.axvline(cutoff, color="k", linestyle=":", alpha=0.35)
        label = f"{int(frac * 100)}% pairs"
        if warmup_epochs > 0:
            label += f" (epoch {epoch_label})"
        ax_frac.scatter(cutoff, frac, color="C2", s=35)
        ax_frac.text(cutoff, frac + 0.03, label, ha="center", va="bottom", fontsize=8)

    ax_frac.plot(cutoff_positions, fractions, color="C2", linewidth=1.2)
    ax_frac.set_xlabel("Lead time Δt")
    ax_frac.set_ylabel("Curriculum fraction")
    ax_frac.set_ylim(0.0, 1.1)
    ax_frac.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    if save_path is None:
        save_path = Path(__file__).resolve().parent / "data" / "lead_time_coverage.png"
    save_path = _ensure_parent(Path(save_path))
    fig.savefig(save_path, dpi=200)
    print(f"Saved lead-time coverage plot to: {save_path}")
    plt.close(fig)


def plot_lead_time_error(
    model,
    dataset: SnapshotPairDataset,
    x_grid: np.ndarray,
    n_bins: int = 8,
    max_samples: int = 256,
    seed: int = 42,
    save_path: str | Path | None = None,
):
    """Plot held-out error as a function of lead time."""
    records = _compute_pair_error_records(
        model=model,
        dataset=dataset,
        x_grid=x_grid,
        max_samples=max_samples,
        seed=seed,
    )
    centers, global_mean, counts = _compute_binned_means(records["lead_time"], records["global_rel_l2"], n_bins)
    _, iface_mean, _ = _compute_binned_means(records["lead_time"], records["iface_rel_l2"], n_bins)

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [3, 1]})
    ax_top.plot(centers, global_mean, marker="o", color="C0", label="Global rel. L2 (%)")
    ax_top.plot(centers, iface_mean, marker="s", color="C3", label="Interface rel. L2 (%)")
    ax_top.set_xlabel("Lead time Δt")
    ax_top.set_ylabel("Mean error (%)")
    ax_top.set_title("Held-Out Error vs Lead Time")
    ax_top.legend()
    ax_top.grid(True, linestyle="--", alpha=0.3)

    width = 0.8 * (centers[1] - centers[0]) if len(centers) > 1 else 0.02
    ax_bot.bar(centers, counts, width=width, color="0.6")
    ax_bot.set_xlabel("Lead time Δt")
    ax_bot.set_ylabel("Samples")
    ax_bot.grid(True, axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    if save_path is None:
        save_path = Path(__file__).resolve().parent / "data" / "lead_time_error.png"
    save_path = _ensure_parent(Path(save_path))
    fig.savefig(save_path, dpi=200)
    print(f"Saved lead-time error plot to: {save_path}")
    plt.close(fig)


def plot_parameter_error_slices(
    model,
    dataset: SnapshotPairDataset,
    x_grid: np.ndarray,
    max_samples: int = 256,
    n_bins: int = 6,
    seed: int = 42,
    save_path: str | Path | None = None,
):
    """Plot held-out error against amplitude, frequency, and contact resistance."""
    records = _compute_pair_error_records(
        model=model,
        dataset=dataset,
        x_grid=x_grid,
        max_samples=max_samples,
        seed=seed,
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    specs = [
        ("amplitude", "Flux Amplitude (A)", False),
        ("frequency", "Flux Frequency (f)", True),
        ("R_c", "Contact Resistance (R_c)", False),
    ]

    for ax, (key, x_label, use_log) in zip(axes, specs):
        x_vals = records[key]
        ax.scatter(x_vals, records["global_rel_l2"], s=14, alpha=0.25, color="0.5", label="Per-pair global")
        centers, global_mean, _ = _compute_binned_means(x_vals, records["global_rel_l2"], n_bins)
        _, iface_mean, _ = _compute_binned_means(x_vals, records["iface_rel_l2"], n_bins)
        ax.plot(centers, global_mean, color="C0", linewidth=1.8, marker="o", label="Binned global")
        ax.plot(centers, iface_mean, color="C3", linewidth=1.8, marker="s", label="Binned interface")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Error (%)")
        ax.grid(True, linestyle="--", alpha=0.3)
        if use_log:
            ax.set_xscale("log")

    axes[0].legend(fontsize=8)
    fig.suptitle("Held-Out Error Across Conditioning Dimensions", fontsize=13)
    fig.tight_layout()
    if save_path is None:
        save_path = Path(__file__).resolve().parent / "data" / "parameter_error_slices.png"
    save_path = _ensure_parent(Path(save_path))
    fig.savefig(save_path, dpi=200)
    print(f"Saved parameter error slices to: {save_path}")
    plt.close(fig)


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

    fig, ax = plt.subplots(figsize=(10, max(3, 0.8 * n + 1.5)))

    y_pos = np.arange(n)
    means = [r["objective"] for r in records_plot]
    stds = [float(np.std(r["per_seed_vals"])) for r in records_plot]
    labels = [r["config_name"] for r in records_plot]

    # bar colors: best (last in reversed list) = green, rest = gray
    colors = ["#b0b0b0"] * n
    colors[-1] = "#4CAF50"  # best config (top bar)

    ax.barh(y_pos, means, xerr=stds, height=0.6, color=colors,
            edgecolor="k", linewidth=0.5, capsize=3, ecolor="#555")

    # overlay individual seed points
    for i, r in enumerate(records_plot):
        seed_vals = r["per_seed_vals"]
        ax.scatter(seed_vals, [i] * len(seed_vals), color="k", s=18, zorder=5, alpha=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Validation Rel. L2 (%)")
    ax.set_title("Hyperparameter Sweep — Config Ranking")

    # log scale if range > 10×
    obj_vals = [r["objective"] for r in records]
    if len(obj_vals) >= 2 and max(obj_vals) / max(min(obj_vals), 1e-12) > 10:
        ax.set_xscale("log")

    ax.grid(True, axis="x", linestyle="--", alpha=0.3)

    fig.tight_layout()
    if save_path is None:
        save_path = experiment_dir / "sweep_ranking.png"
    save_path = _ensure_parent(Path(save_path))
    fig.savefig(save_path, dpi=200)
    print(f"Saved sweep ranking to: {save_path}")
    plt.close(fig)


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

    fig, (ax_val, ax_train) = plt.subplots(1, 2, figsize=(14, 5))

    for cname in config_names:
        if cname not in curves:
            continue
        c = curves[cname]
        color = color_map[cname]

        # validation loss
        if len(c["val_epochs"]) > 0:
            ax_val.semilogy(c["val_epochs"], c["val_rel_l2"], color=color,
                            linewidth=1.2, label=cname)
            # star at best checkpoint
            best_idx = np.argmin(c["val_rel_l2"])
            ax_val.scatter(c["val_epochs"][best_idx], c["val_rel_l2"][best_idx],
                           marker="*", s=80, color=color, edgecolors="k",
                           linewidths=0.5, zorder=5)

        # training loss
        ax_train.semilogy(c["epochs"], c["train_loss"], color=color,
                          linewidth=0.8, alpha=0.8, label=cname)

    ax_val.set_xlabel("Epoch")
    ax_val.set_ylabel("Val Rel. L2 (%)")
    ax_val.set_title(f"Validation Loss (seed {seed})")
    ax_val.legend(fontsize=8, loc="upper right")
    ax_val.grid(True, linestyle="--", alpha=0.3)

    ax_train.set_xlabel("Epoch")
    ax_train.set_ylabel("Train MSE")
    ax_train.set_title(f"Training Loss (seed {seed})")
    ax_train.legend(fontsize=8, loc="upper right")
    ax_train.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    if save_path is None:
        save_path = experiment_dir / "sweep_convergence.png"
    save_path = _ensure_parent(Path(save_path))
    fig.savefig(save_path, dpi=200)
    print(f"Saved sweep convergence to: {save_path}")
    plt.close(fig)


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

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for idx, (key, label, log_x) in enumerate(hp_specs):
        ax = axes[idx]
        vals = [r[key] for r in records]

        # skip panel if all values are None
        if all(v is None for v in vals):
            ax.set_visible(False)
            continue

        vals = np.array([v if v is not None else np.nan for v in vals], dtype=float)
        ax.scatter(vals, objectives, c=colors, s=80, edgecolors="k", linewidths=0.5, zorder=3)

        # highlight best config with a circle
        ax.scatter(vals[0], objectives[0], s=200, facecolors="none",
                   edgecolors="#4CAF50", linewidths=2.0, zorder=4)

        if log_x:
            ax.set_xscale("log")

        # y-axis log if range > 10x
        if objectives.max() / max(objectives.min(), 1e-12) > 10:
            ax.set_yscale("log")

        ax.set_xlabel(label)
        ax.set_ylabel("Val Rel. L2 (%)")
        ax.grid(True, linestyle="--", alpha=0.3)

    # 6th panel: modes × width (capacity proxy)
    ax = axes[5]
    modes_vals = np.array([r["modes"] if r["modes"] is not None else np.nan for r in records], dtype=float)
    width_vals = np.array([r["width"] if r["width"] is not None else np.nan for r in records], dtype=float)
    capacity = modes_vals * width_vals

    if not np.all(np.isnan(capacity)):
        ax.scatter(capacity, objectives, c=colors, s=80, edgecolors="k", linewidths=0.5, zorder=3)
        ax.scatter(capacity[0], objectives[0], s=200, facecolors="none",
                   edgecolors="#4CAF50", linewidths=2.0, zorder=4)
        if objectives.max() / max(objectives.min(), 1e-12) > 10:
            ax.set_yscale("log")
        ax.set_xlabel("Modes × Width")
        ax.set_ylabel("Val Rel. L2 (%)")
        ax.grid(True, linestyle="--", alpha=0.3)
    else:
        ax.set_visible(False)

    fig.suptitle("Hyperparameter Sensitivity", fontsize=13)
    fig.tight_layout()
    if save_path is None:
        save_path = experiment_dir / "sweep_hyperparams.png"
    save_path = _ensure_parent(Path(save_path))
    fig.savefig(save_path, dpi=200)
    print(f"Saved sweep hyperparams to: {save_path}")
    plt.close(fig)


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
      python -m visual.plots --group data --data <path> --x-grid <path> --t-grid <path> --params <path> --out visual/
      python -m visual.plots --group sweep --experiment <path> --runs <path> --out visual/
      python -m visual.plots --plots prediction_vs_truth lead_time_error --checkpoint <path> --out visual/

    Optional data flags:
      --data        Path to trajectories .npy file
      --x-grid      Path to x_grid .npy file
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
    physics_plots = ["final_temperature", "layer_geometry", "face_conductance",
                     "multilayer_evolution", "heat_flux_profile"]
    need_physics = any(_should_run(p, groups, individual) for p in physics_plots)

    if need_physics:
        print("=== PHYSICS GROUP ===")
        solver = create_demo_multilayer_solver()
        t_sol, x_sol, T_hist = solver.solve(store_trajectory=True)

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
    if _should_run("mms_convergence", groups, individual) or \
       _should_run("mms_order_estimation", groups, individual):
        print("=== MMS GROUP ===")

        mms_dir = out_dir / "mms"

        if _should_run("mms_convergence", groups, individual):
            print("--- mms_convergence ---")
            plot_mms_convergence(save_path=mms_dir / "mms_convergence.png")

        if _should_run("mms_order_estimation", groups, individual):
            print("--- mms_order_estimation ---")
            plot_mms_order_estimation(save_path=mms_dir / "mms_order_estimation.png")

    # ---- TRAINING GROUP ----
    if _should_run("training_curves", groups, individual) or \
       _should_run("seed_comparison", groups, individual):
        print("=== TRAINING GROUP ===")

        training_dir = out_dir / "training"

        if _should_run("training_curves", groups, individual):
            if args.csv:
                print("--- training_curves ---")
                plot_training_curves(args.csv, save_path=training_dir / "training_curves.png")
            else:
                print("Skipping training_curves (no --csv provided)")

        if _should_run("seed_comparison", groups, individual):
            if args.report:
                print("--- seed_comparison ---")
                plot_seed_comparison(args.report, save_path=training_dir / "seed_comparison.png")
            else:
                print("Skipping seed_comparison (no --report provided)")

    # ---- DATA GROUP ----
    data_plots = [
        "trajectory_heatmap",
        "initial_conditions",
        "lhs_scatter",
        "flux_profiles",
        "snapshot_pair_samples",
        "prediction_vs_truth",
        "interface_error",
        "lead_time_coverage",
        "lead_time_error",
        "parameter_error_slices",
    ]
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
        }
        needs_grid_data = any(_should_run(name, groups, individual) for name in grid_data_plots)
        trajectories = None
        x_grid = None
        t_grid = None
        plot_config = None
        split_datasets = None

        if needs_grid_data:
            if args.data and args.x_grid and args.t_grid:
                print(f"Loading trajectories and grids from {args.data}, {args.x_grid}, {args.t_grid} ...")
                trajectories, x_grid, t_grid = _load_plot_data(args.data, args.x_grid, args.t_grid)
            else:
                print("Skipping grid-based data plots (need --data, --x-grid, and --t-grid)")

        model_plots = {"prediction_vs_truth", "interface_error", "lead_time_error", "parameter_error_slices"}
        needs_model = any(_should_run(name, groups, individual) for name in model_plots)
        if needs_model and args.checkpoint:
            model, checkpoint_conf = _load_checkpoint_model(args.checkpoint)
        elif needs_model:
            print("Skipping checkpoint-based model plots (need --checkpoint)")

        if trajectories is not None and sim_params is not None:
            plot_config = _resolve_plot_config(checkpoint_conf)
            split_datasets = _build_split_datasets(trajectories, x_grid, t_grid, sim_params, plot_config)

        if sim_params is not None and _should_run("lhs_scatter", groups, individual):
            print("--- lhs_scatter ---")
            plot_lhs_scatter(sim_params, save_path=data_dir / "lhs_scatter.png")

        if trajectories is not None:
            if _should_run("trajectory_heatmap", groups, individual):
                print("--- trajectory_heatmap ---")
                plot_trajectory_heatmap(trajectories, sim_id=0, x_grid=x_grid, t_grid=t_grid,
                                        save_path=data_dir / "trajectory_heatmap.png")

            if _should_run("initial_conditions", groups, individual):
                print("--- initial_conditions ---")
                plot_initial_conditions(trajectories, x_grid=x_grid,
                                        save_path=data_dir / "initial_conditions.png")

            # plots requiring sim_params
            if sim_params is not None:
                if split_datasets is not None and _should_run("snapshot_pair_samples", groups, individual):
                    print("--- snapshot_pair_samples ---")
                    plot_snapshot_pair_samples(
                        trajectories,
                        x_grid,
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
                            t_grid,
                            sim_params,
                            sim_id=sim_id,
                            s=s,
                            n_steps=n_steps,
                            save_path=data_dir / "prediction_vs_truth.png",
                        )

                    if _should_run("interface_error", groups, individual):
                        print("--- interface_error ---")
                        plot_interface_error(
                            model,
                            trajectories,
                            x_grid,
                            t_grid,
                            sim_params,
                            test_dataset.sim_ids,
                            save_path=data_dir / "interface_error.png",
                        )

                    if _should_run("lead_time_error", groups, individual):
                        print("--- lead_time_error ---")
                        plot_lead_time_error(
                            model,
                            test_dataset,
                            x_grid,
                            save_path=data_dir / "lead_time_error.png",
                        )

                    if _should_run("parameter_error_slices", groups, individual):
                        print("--- parameter_error_slices ---")
                        plot_parameter_error_slices(
                            model,
                            test_dataset,
                            x_grid,
                            save_path=data_dir / "parameter_error_slices.png",
                        )
            else:
                print("Skipping param-dependent plots (no --params provided)")
        else:
            if needs_grid_data:
                print("Skipping trajectory-dependent plots (data/grids not available)")

    # ---- SWEEP GROUP ----
    sweep_plots = ["sweep_ranking", "sweep_convergence", "sweep_hyperparams"]
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
                    plot_sweep_convergence(experiment_dir, args.runs,
                                           seed=args.sweep_seed,
                                           save_path=sweep_dir / "sweep_convergence.png")
                else:
                    print("Skipping sweep_convergence (no --runs provided)")

            if _should_run("sweep_hyperparams", groups, individual):
                print("--- sweep_hyperparams ---")
                plot_sweep_hyperparams(experiment_dir, save_path=sweep_dir / "sweep_hyperparams.png")
        else:
            print("Skipping sweep plots (no --experiment provided)")

    print(f"\nAll requested plots saved to: {out_dir}")
