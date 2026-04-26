"""Hyperparameter sweep visualization: ranking, convergence overlays, and sensitivity scatter."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from visual._common import PLOT_STYLE, _save_figure


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

    # hyperparameter panels.
    # NOTE (stale): "modes" was the 1D-only key; 2D sweeps emit modes1/modes2.
    # _load_sweep_data still falls back modes -> modes1, so this column shows
    # only x-mode count for 2D runs (modes2 is silently ignored). Capacity
    # panel below uses the same single value × width. Re-work to plot modes1
    # and modes2 separately — capacity = modes1 * modes2 * width — when
    # touching this function next.
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
