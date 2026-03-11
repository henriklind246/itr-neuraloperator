from pathlib import Path
import importlib.util
import sys
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

# Utility to load the solver module by file path so this script works regardless of package layout.
def load_fd_solver_module() -> object:
    # locate fd_solver_1d relative to this file: ../src/physics/fd_solver_1d.py
    base = Path(__file__).resolve().parent.parent
    solver_path = base / "src" / "physics" / "fd_solver_1d.py"
    if not solver_path.exists():
        raise FileNotFoundError(f"Could not find fd_solver_1d.py at expected location: {solver_path}")

    spec = importlib.util.spec_from_file_location("fd_solver_1d_for_plots", str(solver_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def plot_final_temperature(
    solver: object,
    save_path: Optional[Path] = None,
    title: str = "Final temperature vs spatial nodes",
):
    """Run solver to final time and plot spatial grid (x) vs final temperature (T).

    - solver: an instance of FDSolver1D (must have .solve(store_trajectory=False) -> (t, x, T_final)).
    - save_path: where to save the figure. If None, saves to ./final_temperature.png next to this file.
    """
    # run solver without storing full history to get final T
    t, x, T_final = solver.solve(store_trajectory=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, T_final, marker="o", linestyle="-", color="C0", markersize=5)
    ax.set_xlabel("x (spatial nodes)")
    ax.set_ylabel("Temperature")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.5)

    if save_path is None:
        save_path = Path(__file__).resolve().parent / "final_temperature.png"

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    print(f"Saved final temperature plot to: {save_path}")
    plt.close(fig)


# ---------- 1. TRAINING CURVES ----------

def plot_training_curves(csv_path: str | Path, save_path: str | Path | None = None):
    """Plot training loss, validation rel L2, and learning rate from a train_metrics.csv file.

    Layout: 2-row subplot
        Top: train_loss (log) and val_rel_l2 vs epoch (dual y-axes), best checkpoint starred
        Bottom: learning rate vs epoch (log y)
    """
    import csv as csv_mod

    csv_path = Path(csv_path)
    epochs, train_losses, val_rel_l2s, lrs, is_bests = [], [], [], [], []

    with csv_path.open("r") as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_losses.append(float(row["train_loss"]))
            val_rel_l2s.append(float(row["val_rel_l2"]) if row["val_rel_l2"] != "" else None)
            lrs.append(float(row["lr"]))
            is_bests.append(int(row["is_best"]))

    epochs = np.array(epochs)
    train_losses = np.array(train_losses)
    lrs = np.array(lrs)

    # filter validation epochs
    val_epochs = np.array([e for e, v in zip(epochs, val_rel_l2s) if v is not None])
    val_values = np.array([v for v in val_rel_l2s if v is not None])

    # best checkpoint epochs
    best_epochs = np.array([e for e, b in zip(epochs, is_bests) if b == 1])
    best_vals = np.array([v for v, b in zip(val_rel_l2s, is_bests) if b == 1 and v is not None])

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [3, 1]})

    # --- top panel: train loss + val rel L2 ---
    color_train = "C0"
    color_val = "C3"

    ax_top.semilogy(epochs, train_losses, color=color_train, alpha=0.7, linewidth=0.8, label="Train loss (MSE)")
    ax_top.set_xlabel("Epoch")
    ax_top.set_ylabel("Train loss (MSE)", color=color_train)
    ax_top.tick_params(axis="y", labelcolor=color_train)

    ax_val = ax_top.twinx()
    ax_val.semilogy(val_epochs, val_values, color=color_val, linewidth=1.2, label="Val rel. L2 (%)")
    if len(best_epochs) > 0 and len(best_vals) > 0:
        ax_val.scatter(best_epochs, best_vals, marker="*", s=60, color="gold", zorder=5, edgecolors="k", linewidths=0.5, label="Best checkpoint")
    ax_val.set_ylabel("Val rel. L2 (%)", color=color_val)
    ax_val.tick_params(axis="y", labelcolor=color_val)

    # combine legends
    lines_1, labels_1 = ax_top.get_legend_handles_labels()
    lines_2, labels_2 = ax_val.get_legend_handles_labels()
    ax_val.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right", fontsize=8)
    ax_top.set_title("Training Curves")
    ax_top.grid(True, linestyle="--", alpha=0.3)

    # --- bottom panel: learning rate ---
    ax_bot.semilogy(epochs, lrs, color="C2", linewidth=1.0)
    ax_bot.set_xlabel("Epoch")
    ax_bot.set_ylabel("Learning rate")
    ax_bot.set_title("Learning Rate Schedule")
    ax_bot.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    if save_path is None:
        save_path = csv_path.parent / "training_curves.png"
    fig.savefig(save_path, dpi=200)
    print(f"Saved training curves to: {save_path}")
    plt.close(fig)


# ---------- 2. PREDICTION VS TRUTH HEATMAP ----------

def plot_prediction_vs_truth(
    model,
    trajectories: np.ndarray,
    x_grid: np.ndarray,
    t_grid: np.ndarray,
    sim_id: int,
    s: int,
    k: int = 10,
    H: int = 40,
    save_path: str | Path | None = None,
):
    """Plot ground truth, FNO prediction, and pointwise error as 3-panel heatmaps.

    Args:
        model: trained FNO2d model (on CPU or GPU)
        trajectories: raw data array (num_sims, Nt, Nx)
        x_grid: spatial grid (Nx,)
        t_grid: time grid (Nt,)
        sim_id: which simulation to visualize
        s: starting index for the forecast window
        k: history window length
        H: forecast horizon
    """
    import torch

    Nx = x_grid.shape[0]

    # build a single input sample (same logic as WindowedForecastDataset.__getitem__)
    T_hist = trajectories[sim_id]  # (Nt, Nx)
    history = T_hist[s:s + k, :].T  # (Nx, k)
    target = T_hist[s + k:s + k + H, :].T  # (Nx, H)

    t_future = t_grid[s + k:s + k + H]
    x_norm = (x_grid - x_grid[0]) / (x_grid[-1] - x_grid[0])
    t_norm = (t_future - t_grid[0]) / (t_grid[-1] - t_grid[0])

    history_grid = np.broadcast_to(history[:, None, :], (Nx, H, k))
    x_channel = np.broadcast_to(x_norm[:, None, None], (Nx, H, 1))
    t_channel = np.broadcast_to(t_norm[None, :, None], (Nx, H, 1))

    X = np.concatenate([history_grid, x_channel, t_channel], axis=-1).astype(np.float32)
    X_tensor = torch.from_numpy(X).unsqueeze(0)  # (1, Nx, H, 12)

    device = next(model.parameters()).device
    X_tensor = X_tensor.to(device)

    with torch.no_grad():
        model.eval()
        Y_pred = model(X_tensor).cpu().numpy().squeeze()  # (Nx, H)

    Y_true = target  # (Nx, H)
    error = np.abs(Y_pred - Y_true)

    # meshgrid for pcolormesh
    T_mesh, X_mesh = np.meshgrid(t_future, x_grid)

    vmin = min(Y_true.min(), Y_pred.min())
    vmax = max(Y_true.max(), Y_pred.max())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    # ground truth
    pc0 = axes[0].pcolormesh(T_mesh, X_mesh, Y_true, cmap="inferno", vmin=vmin, vmax=vmax, shading="auto")
    axes[0].set_title("Ground Truth")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("x")

    # prediction
    pc1 = axes[1].pcolormesh(T_mesh, X_mesh, Y_pred, cmap="inferno", vmin=vmin, vmax=vmax, shading="auto")
    axes[1].set_title("FNO Prediction")
    axes[1].set_xlabel("Time")

    # shared colorbar for truth/pred
    fig.colorbar(pc1, ax=axes[:2].tolist(), label="Temperature", shrink=0.85)

    # error
    pc2 = axes[2].pcolormesh(T_mesh, X_mesh, error, cmap="Reds", shading="auto")
    axes[2].set_title("Absolute Error")
    axes[2].set_xlabel("Time")
    fig.colorbar(pc2, ax=axes[2], label="|Error|", shrink=0.85)

    fig.suptitle(f"Sim {sim_id}, window start s={s}", fontsize=12)
    fig.tight_layout()

    if save_path is None:
        save_path = Path(__file__).resolve().parent / "prediction_vs_truth.png"
    fig.savefig(save_path, dpi=200)
    print(f"Saved prediction vs truth to: {save_path}")
    plt.close(fig)


# ---------- 3. SIMULATION TRAJECTORY HEATMAP ----------

def plot_trajectory_heatmap(
    trajectories: np.ndarray,
    sim_id: int,
    x_grid: np.ndarray,
    t_grid: np.ndarray,
    save_path: str | Path | None = None,
):
    """Plot a single simulation's full temperature evolution as a 2D heatmap.

    Args:
        trajectories: (num_sims, Nt, Nx)
        sim_id: which simulation
        x_grid: (Nx,)
        t_grid: (Nt,)
    """
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
        save_path = Path(__file__).resolve().parent / "trajectory_heatmap.png"
    fig.savefig(save_path, dpi=200)
    print(f"Saved trajectory heatmap to: {save_path}")
    plt.close(fig)


# ---------- 4. MMS CONVERGENCE ----------

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
        save_path = Path(__file__).resolve().parent / "mms_convergence.png"
    fig.savefig(save_path, dpi=200)
    print(f"Saved MMS convergence to: {save_path}")
    plt.close(fig)


# ---------- 5. SEED COMPARISON BAR CHART ----------

def plot_seed_comparison(report_path: str | Path, save_path: str | Path | None = None):
    """Plot grouped bar chart of val and test rel L2 per seed from a seed_report.json.

    Args:
        report_path: path to seed_report.json produced by eval.py
    """
    import json

    report_path = Path(report_path)
    with report_path.open("r") as f:
        report = json.load(f)

    per_seed = report["per_seed"]
    summary = report["summary"]

    seeds = [str(r["seed"]) for r in per_seed]
    val_losses = [r["best_val"] for r in per_seed]
    test_losses = [r["test_rel_l2"] for r in per_seed]

    x = np.arange(len(seeds))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars_val = ax.bar(x - width / 2, val_losses, width, label="Val rel. L2 (%)", color="C0", alpha=0.8)
    bars_test = ax.bar(x + width / 2, test_losses, width, label="Test rel. L2 (%)", color="C3", alpha=0.8)

    # mean lines
    ax.axhline(summary["best_val_loss_mean"], color="C0", linestyle="--", alpha=0.6, label=f"Val mean: {summary['best_val_loss_mean']:.3f}")
    ax.axhline(summary["test_rel_l2_mean"], color="C3", linestyle="--", alpha=0.6, label=f"Test mean: {summary['test_rel_l2_mean']:.3f}")

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


# ---------- 6. INITIAL CONDITION DISTRIBUTION ----------

def plot_initial_conditions(
    trajectories: np.ndarray,
    x_grid: np.ndarray,
    n_samples: int = 20,
    seed: int = 42,
    save_path: str | Path | None = None,
):
    """Plot overlaid initial conditions T(x, t=0) for n randomly selected simulations.

    Args:
        trajectories: (num_sims, Nt, Nx)
        x_grid: (Nx,)
        n_samples: number of ICs to overlay
        seed: random seed for selecting simulations
    """
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
        save_path = Path(__file__).resolve().parent / "initial_conditions.png"
    fig.savefig(save_path, dpi=200)
    print(f"Saved initial conditions plot to: {save_path}")
    plt.close(fig)


# ---------- 7. MMS ORDER ESTIMATION (MULTI-POINT) ----------

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
        save_path = Path(__file__).resolve().parent / "mms_order_estimation.png"
    fig.savefig(save_path, dpi=200)
    print(f"Saved MMS order estimation to: {save_path}")
    plt.close(fig)


if __name__ == "__main__":


    """
    How to run (--data, --csv, and --report are optional):
    python -m visual.plots.py \
  --data path/to/trajectories.npy \
  --csv path/to/train_metrics.csv \
  --report path/to/seed_report.json \
  --out path/to/output_dir
    """

    import argparse

    parser = argparse.ArgumentParser(description="Generate all plots")
    parser.add_argument("--data", type=str, default=None, help="Path to trajectories .npy file")
    parser.add_argument("--csv", type=str, default=None, help="Path to train_metrics.csv")
    parser.add_argument("--report", type=str, default=None, help="Path to seed_report.json")
    parser.add_argument("--out", type=str, default=None, help="Output directory for plots")
    args = parser.parse_args()

    out_dir = Path(args.out).resolve() if args.out else Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Training curves
    if args.csv:
        print("--- Plot 1: Training Curves ---")
        plot_training_curves(args.csv, save_path=out_dir / "training_curves.png")
    else:
        print("Skipping training curves (no --csv provided)")

    # 2. MMS Convergence
    print("--- Plot 4: MMS Convergence ---")
    plot_mms_convergence(save_path=out_dir / "mms_convergence.png")

    # 3. MMS Order Estimation
    print("--- Plot 7: MMS Order Estimation ---")
    plot_mms_order_estimation(save_path=out_dir / "mms_order_estimation.png")

    # 4. Seed comparison
    if args.report:
        print("--- Plot 5: Seed Comparison ---")
        plot_seed_comparison(args.report, save_path=out_dir / "seed_comparison.png")
    else:
        print("Skipping seed comparison (no --report provided)")

    # 5. Data-dependent plots (trajectories)
    if args.data:
        data_path = Path(args.data)
        print(f"Loading trajectories from {data_path} ...")
        trajectories = np.load(data_path)

        # infer grids from shape (num_sims, Nt, Nx)
        _, Nt, Nx = trajectories.shape
        x_grid = np.linspace(0.0, 1.0, Nx)
        t_grid = np.linspace(0.0, 1.0, Nt)

        print("--- Plot 3: Trajectory Heatmap (sim 0) ---")
        plot_trajectory_heatmap(trajectories, sim_id=0, x_grid=x_grid, t_grid=t_grid,
                                save_path=out_dir / "trajectory_heatmap.png")

        print("--- Plot 6: Initial Conditions ---")
        plot_initial_conditions(trajectories, x_grid=x_grid,
                                save_path=out_dir / "initial_conditions.png")
    else:
        print("Skipping trajectory/initial-condition plots (no --data provided)")

    # 6. Final temperature (FD solver)
    print("--- Plot: Final Temperature (FD Solver) ---")
    try:
        fd_mod = load_fd_solver_module()
        solver = fd_mod.FDSolver1D()
        plot_final_temperature(solver, save_path=out_dir / "final_temperature.png")
    except Exception as e:
        print(f"Skipping final temperature plot: {e}")

    print(f"\nAll plots saved to: {out_dir}")
