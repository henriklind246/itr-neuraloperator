"""Physics solver diagnostics: temperature fields, geometry, conductance, and flux profiles."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from src.physics.fv_solver_2d import FVSolver2D, Layer2D

from visual._common import (
    PLOT_STYLE,
    _add_interface_lines,
    _evaluate_q_left_field,
    _resolve_interface_metadata,
    _save_figure,
)


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

    2x2 layout:
        (a) Temporal trace a(t) at the y of peak |s(y)|
        (b) Spatial profile s(y) at the time of peak |q|
        (c) Heatmap q_left(y, t)
        (d) Numerical x-flux at y=mid at selected time snapshots
    """
    Nt = len(solver.t)
    j_mid = solver.Ny // 2
    interface_meta = _resolve_interface_metadata(solver=solver)

    Q = _evaluate_q_left_field(solver)  # (Nt, Ny)
    is_vector = bool(getattr(solver, "_q_left_is_vector", False))

    abs_max_per_y = np.max(np.abs(Q), axis=0)
    j_peak = int(np.argmax(abs_max_per_y)) if is_vector else j_mid
    abs_max_per_t = np.max(np.abs(Q), axis=1)
    t_peak = int(np.argmax(abs_max_per_t))
    q_abs = max(float(np.max(np.abs(Q))), 1e-12)

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        ax_a, ax_b = axes[0, 0], axes[0, 1]
        ax_c, ax_d = axes[1, 0], axes[1, 1]

        ax_a.plot(solver.t, Q[:, j_peak], color="C3")
        ax_a.axvline(solver.t_on, color="C2", linestyle=":", alpha=0.8, label=f"t_on={solver.t_on}")
        ax_a.axvline(solver.t_off, color="C1", linestyle=":", alpha=0.8, label=f"t_off={solver.t_off}")
        ax_a.set_xlabel("Time")
        ax_a.set_ylabel("q_left(t)")
        title_a = f"Temporal trace at y = {solver.grid_y[j_peak]:.2f}" if is_vector else "Applied Boundary Flux (uniform)"
        ax_a.set_title(title_a)
        ax_a.legend()
        ax_a.grid(True)

        if is_vector:
            ax_b.plot(solver.grid_y, Q[t_peak, :], color="C0")
            ax_b.set_title(f"Spatial profile at t = {solver.t[t_peak]:.3f}")
        else:
            ax_b.plot(solver.grid_y, np.ones_like(solver.grid_y), color="C0")
            ax_b.set_ylim(-0.1, 1.5)
            ax_b.set_title("Spatial profile (uniform)")
        ax_b.set_xlabel("y")
        ax_b.set_ylabel("q_left(y) at t_peak")
        ax_b.grid(True)

        pcm = ax_c.pcolormesh(
            solver.t,
            solver.grid_y,
            Q.T,
            cmap="coolwarm",
            vmin=-q_abs,
            vmax=q_abs,
            shading="auto",
        )
        ax_c.axvline(solver.t_on, color="0.2", linestyle=":", alpha=0.8)
        ax_c.axvline(solver.t_off, color="0.2", linestyle=":", alpha=0.8)
        ax_c.set_xlabel("Time")
        ax_c.set_ylabel("y")
        ax_c.set_title("q_left(y, t)")
        fig.colorbar(pcm, ax=ax_c, label=r"$q_{left}$", shrink=0.9)

        snap_indices = np.linspace(1, Nt - 1, min(5, Nt - 1), dtype=int)
        cmap_snap = plt.cm.viridis
        for i, t_idx in enumerate(snap_indices):
            dT = T_hist[t_idx, 1:, j_mid] - T_hist[t_idx, :-1, j_mid]
            q_face = -solver.G_x[:, j_mid] * dT
            ax_d.plot(
                solver.face_positions_x,
                q_face,
                color=cmap_snap(i / max(len(snap_indices) - 1, 1)),
                label=f"t={solver.t[t_idx]:.3f}",
            )

        _add_interface_lines(ax_d, interface_meta["positions"])
        ax_d.set_xlabel("Face position (x)")
        ax_d.set_ylabel("Heat flux q")
        ax_d.set_title(f"Numerical x-Flux at y = {solver.grid_y[j_mid]:.2f}")
        ax_d.legend(loc="best")
        ax_d.grid(True)

        _save_figure(fig, save_path, "physics", "heat_flux_profile")
