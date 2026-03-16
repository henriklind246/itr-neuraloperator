from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from src.physics.fd_solver_1d import FDSolver1D, Layer1D


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
    "trajectory_comparison_grid":"data",
    "boundary_temperature":      "data",
    "parameter_response":        "data",
}

GROUPS = {"physics", "mms", "training", "data"}


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
      Layer 1: [0.0, 0.5], rho=1.0, cp=1.0, k=1.0
      Layer 2: [0.5, 1.0], rho=2.0, cp=1.5, k=1.5
    N=100 ensures x=0.5 lies on a cell face (required by the solver).
    """
    layers = [
        Layer1D(x_left=0.0, x_right=0.5, rho=1.0, cp=1.0, k=1.0),
        Layer1D(x_left=0.5, x_right=1.0, rho=2.0, cp=1.5, k=1.5),
    ]
    return FDSolver1D(
        a=0.0, b=1.0, N=100, layers=layers, lam_target=0.5,
        t_final=1.0, flux_f=2.0, flux_A=50.0,
        t_on=0.0, t_off=0.2, phase=0.0, dt=0.005,
    )


def _interface_flanking_nodes(solver: FDSolver1D) -> list[tuple[int, int, float]]:
    """Return (left_node, right_node, interface_x) for each internal interface."""
    return [
        (f, f + 1, solver.face_positions[f])
        for f in sorted(solver.interface_face_map.keys())
    ]


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
        save_path = Path(__file__).resolve().parent / "final_temperature.png"
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
        save_path = Path(__file__).resolve().parent / "layer_geometry.png"
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
        save_path = Path(__file__).resolve().parent / "face_conductance.png"
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
        save_path = Path(__file__).resolve().parent / "multilayer_evolution.png"
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
        save_path = Path(__file__).resolve().parent / "heat_flux_profile.png"
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
        save_path = Path(__file__).resolve().parent / "mms_convergence.png"
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
        save_path = Path(__file__).resolve().parent / "mms_order_estimation.png"
    fig.savefig(save_path, dpi=200)
    print(f"Saved MMS order estimation to: {save_path}")
    plt.close(fig)


# ============================================================
# SECTION: TRAINING — Neural Operator Training Metrics
# ============================================================

def plot_training_curves(csv_path: str | Path, save_path: str | Path | None = None):
    """Plot training loss, rel L2 metrics, and learning rate from a train_metrics.csv file.

    Layout: 2-row subplot
        Top: train_loss (log, left y) and train/val rel_l2 (log, right y), best checkpoint starred
        Bottom: learning rate vs epoch (log y)
    """
    import csv as csv_mod

    csv_path = Path(csv_path)
    epochs, train_losses, train_rel_l2s, val_rel_l2s, lrs, is_bests = [], [], [], [], [], []

    has_train_rel_l2 = False
    with csv_path.open("r") as f:
        reader = csv_mod.DictReader(f)
        has_train_rel_l2 = "train_rel_l2" in (reader.fieldnames or [])
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_losses.append(float(row["train_loss"]))
            if has_train_rel_l2:
                train_rel_l2s.append(float(row["train_rel_l2"]))
            val_rel_l2s.append(float(row["val_rel_l2"]) if row["val_rel_l2"] != "" else None)
            lrs.append(float(row["lr"]))
            is_bests.append(int(row["is_best"]))

    epochs = np.array(epochs)
    train_losses = np.array(train_losses)
    if has_train_rel_l2:
        train_rel_l2s = np.array(train_rel_l2s)
    lrs = np.array(lrs)

    # filter validation epochs
    val_epochs = np.array([e for e, v in zip(epochs, val_rel_l2s) if v is not None])
    val_values = np.array([v for v in val_rel_l2s if v is not None])

    # best checkpoint epochs
    best_epochs = np.array([e for e, b in zip(epochs, is_bests) if b == 1])
    best_vals = np.array([v for v, b in zip(val_rel_l2s, is_bests) if b == 1 and v is not None])

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [3, 1]})

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


def plot_seed_comparison(report_path: str | Path, save_path: str | Path | None = None):
    """Plot grouped bar chart of val and test rel L2 per seed from a seed_report.json."""
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
    ax.bar(x - width / 2, val_losses, width, label="Val rel. L2 (%)", color="C0", alpha=0.8)
    ax.bar(x + width / 2, test_losses, width, label="Test rel. L2 (%)", color="C3", alpha=0.8)

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


# ============================================================
# SECTION: DATA — Dataset & Parameter Space Visualization
# ============================================================

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
    """Plot ground truth, FNO prediction, and pointwise error as 3-panel heatmaps."""
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
    axes[0].pcolormesh(T_mesh, X_mesh, Y_true, cmap="inferno", vmin=vmin, vmax=vmax, shading="auto")
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
        save_path = Path(__file__).resolve().parent / "trajectory_heatmap.png"
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
        save_path = Path(__file__).resolve().parent / "initial_conditions.png"
    fig.savefig(save_path, dpi=200)
    print(f"Saved initial conditions plot to: {save_path}")
    plt.close(fig)


def plot_lhs_scatter(
    sim_params: np.ndarray,
    save_path: str | Path | None = None,
):
    """Scatter plot of LHS-sampled (amplitude, frequency) pairs."""
    amplitudes = np.array([p[0] for p in sim_params], dtype=np.float32)
    frequencies = np.array([p[1] for p in sim_params], dtype=np.float32)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(amplitudes, frequencies, s=12, alpha=0.6, edgecolors="none")
    ax.set_xlabel("Flux Amplitude (A)")
    ax.set_ylabel("Flux Frequency (f)")
    ax.set_title(f"LHS Parameter Space — {len(amplitudes)} simulations")
    ax.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    if save_path is None:
        save_path = Path(__file__).resolve().parent / "lhs_scatter.png"
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
        save_path = Path(__file__).resolve().parent / "flux_profiles.png"
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
        save_path = Path(__file__).resolve().parent / "trajectory_comparison_grid.png"
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
        save_path = Path(__file__).resolve().parent / "boundary_temperature.png"
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
        save_path = Path(__file__).resolve().parent / "parameter_response.png"
    fig.savefig(save_path, dpi=200)
    print(f"Saved parameter-response to: {save_path}")
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
      python -m visual.plots --group data --data <path> --params <path> --out visual/
      python -m visual.plots --plots layer_geometry heat_flux_profile --out visual/

    Optional data flags:
      --data    Path to trajectories .npy file,
      --params  Path to sim_params .npy file
      --csv     Path to train_metrics.csv
      --report  Path to seed_report.json
    """

    import argparse

    parser = argparse.ArgumentParser(description="Generate plots for the IHCP project")
    parser.add_argument("--data", type=str, default=None, help="Path to trajectories .npy file")
    parser.add_argument("--params", type=str, default=None, help="Path to sim_params .npy file")
    parser.add_argument("--csv", type=str, default=None, help="Path to train_metrics.csv")
    parser.add_argument("--report", type=str, default=None, help="Path to seed_report.json")
    parser.add_argument("--out", type=str, default=None, help="Output directory for plots")
    parser.add_argument("--group", type=str, nargs="+", default=["all"],
                        choices=["all", "physics", "mms", "training", "data"],
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

        if _should_run("final_temperature", groups, individual):
            print("--- final_temperature ---")
            plot_final_temperature(solver, save_path=out_dir / "final_temperature.png")

        if _should_run("layer_geometry", groups, individual):
            print("--- layer_geometry ---")
            plot_layer_geometry(solver, save_path=out_dir / "layer_geometry.png")

        if _should_run("face_conductance", groups, individual):
            print("--- face_conductance ---")
            plot_face_conductance(solver, save_path=out_dir / "face_conductance.png")

        if _should_run("multilayer_evolution", groups, individual):
            print("--- multilayer_evolution ---")
            plot_multilayer_evolution(solver, T_hist, save_path=out_dir / "multilayer_evolution.png")

        if _should_run("heat_flux_profile", groups, individual):
            print("--- heat_flux_profile ---")
            plot_heat_flux_profile(solver, T_hist, save_path=out_dir / "heat_flux_profile.png")

    # ---- MMS GROUP ----
    if _should_run("mms_convergence", groups, individual) or \
       _should_run("mms_order_estimation", groups, individual):
        print("=== MMS GROUP ===")

        if _should_run("mms_convergence", groups, individual):
            print("--- mms_convergence ---")
            plot_mms_convergence(save_path=out_dir / "mms_convergence.png")

        if _should_run("mms_order_estimation", groups, individual):
            print("--- mms_order_estimation ---")
            plot_mms_order_estimation(save_path=out_dir / "mms_order_estimation.png")

    # ---- TRAINING GROUP ----
    if _should_run("training_curves", groups, individual) or \
       _should_run("seed_comparison", groups, individual):
        print("=== TRAINING GROUP ===")

        if _should_run("training_curves", groups, individual):
            if args.csv:
                print("--- training_curves ---")
                plot_training_curves(args.csv, save_path=out_dir / "training_curves.png")
            else:
                print("Skipping training_curves (no --csv provided)")

        if _should_run("seed_comparison", groups, individual):
            if args.report:
                print("--- seed_comparison ---")
                plot_seed_comparison(args.report, save_path=out_dir / "seed_comparison.png")
            else:
                print("Skipping seed_comparison (no --report provided)")

    # ---- DATA GROUP ----
    data_plots = ["trajectory_heatmap", "initial_conditions", "lhs_scatter",
                  "flux_profiles", "trajectory_comparison_grid",
                  "boundary_temperature", "parameter_response"]
    need_data = any(_should_run(p, groups, individual) for p in data_plots)

    if need_data:
        print("=== DATA GROUP ===")

        # flux_profiles needs no data files
        if _should_run("flux_profiles", groups, individual):
            print("--- flux_profiles ---")
            plot_flux_profiles(save_path=out_dir / "flux_profiles.png")

        sim_params = None
        if args.params:
            sim_params = np.load(args.params, allow_pickle=True)

        if args.data:
            data_path = Path(args.data)
            print(f"Loading trajectories from {data_path} ...")
            trajectories = np.load(data_path)

            # infer grids from shape (num_sims, Nt, Nx)
            _, Nt, Nx = trajectories.shape
            x_grid = np.linspace(0.0, 1.0, Nx)
            t_grid = np.linspace(0.0, 1.0, Nt)

            if _should_run("trajectory_heatmap", groups, individual):
                print("--- trajectory_heatmap ---")
                plot_trajectory_heatmap(trajectories, sim_id=0, x_grid=x_grid, t_grid=t_grid,
                                        save_path=out_dir / "trajectory_heatmap.png")

            if _should_run("initial_conditions", groups, individual):
                print("--- initial_conditions ---")
                plot_initial_conditions(trajectories, x_grid=x_grid,
                                        save_path=out_dir / "initial_conditions.png")

            # plots requiring sim_params
            if sim_params is not None:
                if _should_run("lhs_scatter", groups, individual):
                    print("--- lhs_scatter ---")
                    plot_lhs_scatter(sim_params, save_path=out_dir / "lhs_scatter.png")

                if _should_run("trajectory_comparison_grid", groups, individual):
                    print("--- trajectory_comparison_grid ---")
                    plot_trajectory_comparison_grid(trajectories, sim_params, x_grid, t_grid,
                                                    save_path=out_dir / "trajectory_comparison_grid.png")

                if _should_run("boundary_temperature", groups, individual):
                    print("--- boundary_temperature ---")
                    plot_boundary_temperature(trajectories, sim_params, t_grid,
                                               save_path=out_dir / "boundary_temperature.png")

                if _should_run("parameter_response", groups, individual):
                    print("--- parameter_response ---")
                    plot_parameter_response(trajectories, sim_params,
                                             save_path=out_dir / "parameter_response.png")
            else:
                print("Skipping param-dependent plots (no --params provided)")
        else:
            print("Skipping trajectory/initial-condition plots (no --data provided)")

    print(f"\nAll requested plots saved to: {out_dir}")
