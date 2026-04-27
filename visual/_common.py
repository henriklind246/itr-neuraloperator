"""Shared registry, style, dispatch, and cross-module helpers for the plot package."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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
    "trajectory_deviation_heatmap": "data",
    "y_perturbation":            "data",
    "spatial_family_breakdown":  "data",
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


def _evaluate_q_left_field(solver) -> np.ndarray:
    """Sample solver.q_left on solver.t × solver.grid_y as shape (Nt, Ny).

    Broadcasts the legacy scalar path so callers always see a 2D field.
    """
    Ny = solver.Ny
    is_vector = bool(getattr(solver, "_q_left_is_vector", False))
    Q = np.empty((len(solver.t), Ny), dtype=float)
    for i, ti in enumerate(solver.t):
        val = np.asarray(solver.q_left(ti), dtype=float)
        if is_vector:
            Q[i, :] = val
        else:
            Q[i, :] = float(val)
    return Q
