"""Training metrics: loss curves, rel-L2, interface metrics, learning rate, and seed comparisons."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from visual._common import PLOT_STYLE, _save_figure


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
