"""Shared registry, style, dispatch, and cross-module helpers for the plot package."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# REGISTRY — Plot names → group mapping + dispatch helpers
# ============================================================

PLOT_REGISTRY: dict[str, str] = {
    # group: physics
    "bc_verification":           "physics",
    "itr_temperature_jump_sweep": "physics",
    # group: mms
    "mms_convergence":           "mms",
    "mms_order_estimation":      "mms",
    # group: training
    "training_curves":           "training",
    "seed_comparison":           "training",
    # group: data
    "prediction_vs_truth":       "data",
    "interface_error":           "data",
    "lead_time_coverage":        "data",
    "initial_conditions":        "data",
    # group: forcing
    "forcing_temporal_families":            "forcing",
    "forcing_spatial_profiles":             "forcing",
    "forcing_separable_assembly":           "forcing",
    "forcing_seq_tokens":                   "forcing",
    # group: source
    "energy_budget":             "source",
    "patch_region_error_map":    "source",
    "source_interface_zone_error": "source",
    "source_itr_sin_resistance_profiles":  "source",
    # group: interfaces
    "interface_x_breakdown":     "interfaces",
}

PLOT_RERUN_TRIGGERS: dict[str, str] = {
    "bc_verification": "FV boundary-condition or boundary-stencil change",
    "itr_temperature_jump_sweep": "interface-resistance physics or ProblemSpec change",
    "mms_convergence": "2D numerical-scheme or solver change",
    "mms_order_estimation": "2D numerical-scheme or solver change",
    "training_curves": "completed or actively monitored training run",
    "seed_comparison": "completed replicated-seed experiment",
    "prediction_vs_truth": "new checkpoint or changed field-evaluation path",
    "interface_error": "new checkpoint on an interface-bearing benchmark",
    "lead_time_coverage": "snapshot pairing, split, or curriculum change",
    "initial_conditions": "IC family, sampler, or regenerated dataset",
    "forcing_temporal_families": "temporal forcing builder or sampler change",
    "forcing_spatial_profiles": "spatial forcing builder or sampler change",
    "forcing_separable_assembly": "separable forcing assembly change",
    "forcing_seq_tokens": "forcing representation or token-contract change",
    "energy_budget": "source term, FV balance, or regenerated source dataset",
    "patch_region_error_map": "new source-family checkpoint",
    "source_interface_zone_error": "new source-family checkpoint",
    "source_itr_sin_resistance_profiles": "resistance profile-profile sampler or normalization change",
    "interface_x_breakdown": "regenerated interfaces dataset or geometry-sampling change",
}

GROUPS = {"physics", "mms", "training", "data", "forcing", "source", "interfaces"}

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


def _resolve_interface_metadata(config: dict | None = None, solver=None,
                                sim_params=None, sim_id: int | None = None) -> dict:
    """Resolve interface metadata.

    Priority for ``interface_x``:
        1. ``sim_params[sim_id]["interface_x"]`` if both are provided (per-sim
           value from the vary-interfaces experiment).
        2. config ``interface_x`` if present.
        3. ``solver.interface_positions[0]`` if a solver is supplied.
        4. Legacy fallback ``0.5`` so older configs/checkpoints still plot.
    """
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

    per_sim_x = None
    if sim_params is not None and sim_id is not None:
        try:
            per_sim_x = float(sim_params[int(sim_id)]["interface_x"])
        except (KeyError, IndexError, TypeError):
            per_sim_x = None

    if per_sim_x is not None:
        resolved_x = per_sim_x
        positions = [resolved_x]
    elif positions:
        resolved_x = float(interface_x) if interface_x is not None else positions[0]
    else:
        resolved_x = float(interface_x)
        positions = [resolved_x]

    return {
        "positions": positions,
        "interface_x": resolved_x,
        "interface_half_width": float(interface_half_width),
    }
