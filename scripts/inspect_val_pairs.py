"""Inspect a val_pairs.csv produced during training.

Run:
    python scripts/inspect_val_pairs.py <path/to/val_pairs.csv> [--out <dir>]

Produces a text report on stdout and five diagnostic PNGs in --out
(defaults to a sibling directory of the CSV).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _quantile(q: float):
    def _f(s: pd.Series) -> float:
        return float(s.quantile(q))
    _f.__name__ = f"p{int(q * 100)}"
    return _f


def report_training_dynamics(df: pd.DataFrame) -> None:
    print("\n=== 1. Per-epoch validation error ===")
    agg = df.groupby("epoch")["rel_l2"].agg(["mean", "median", _quantile(0.9), _quantile(0.99)])
    print(agg.to_string(float_format=lambda x: f"{x:.4f}"))


def report_worst_sims(latest: pd.DataFrame) -> None:
    print("\n=== 2. Hardest sims (top-20 mean rel_l2 at final epoch) ===")
    per_sim = latest.groupby("sim_id")["rel_l2"].agg(["mean", "count"]).sort_values("mean")
    print(per_sim.tail(20).to_string(float_format=lambda x: f"{x:.4f}"))


def report_regime_stratification(latest: pd.DataFrame) -> None:
    print("\n=== 3. Error stratified by physical regime (final epoch) ===")
    for col in ["t_s", "t_bar", "R_c"]:
        bins = pd.qcut(latest[col], 5, duplicates="drop")
        agg = latest.groupby(bins, observed=True)["rel_l2"].agg(["mean", "median", _quantile(0.99)])
        print(f"\n-- by {col} quintile --")
        print(agg.to_string(float_format=lambda x: f"{x:.4f}"))


def report_ic_family(latest: pd.DataFrame) -> None:
    print("\n=== 4. Error by IC family (final epoch) ===")
    agg = latest.groupby(["temporal_family", "spatial_family"])["rel_l2"].agg(
        ["mean", "median", "count"]
    )
    print(agg.sort_values("mean").to_string(float_format=lambda x: f"{x:.4f}"))


def report_between_vs_within(latest: pd.DataFrame) -> None:
    print("\n=== 5. Between-sim vs within-sim variance decomposition ===")
    sim_means = latest.groupby("sim_id")["rel_l2"].mean()
    between = sim_means.var()
    within = latest.groupby("sim_id")["rel_l2"].var().mean()
    total = between + within
    print(f"Between-sim variance: {between:.4f}  ({between / total:.1%} of total)")
    print(f"Within-sim variance:  {within:.4f}  ({within / total:.1%} of total)")
    print(
        "High between-sim share -> coverage / OOD problem.  "
        "High within-sim share -> regime / conditioning problem."
    )


def report_tails(latest: pd.DataFrame) -> None:
    print("\n=== 6. Tail vs mean (final epoch) ===")
    s = latest["rel_l2"]
    print(
        f"mean={s.mean():.4f}  p50={s.median():.4f}  "
        f"p90={s.quantile(0.9):.4f}  p99={s.quantile(0.99):.4f}  max={s.max():.4f}"
    )
    print("\nTop-20 worst pairs:")
    cols = ["sim_id", "temporal_family", "spatial_family", "t_s", "t_bar", "R_c", "rel_l2", "iface_rel_l2"]
    print(latest.nlargest(20, "rel_l2")[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def plot_training_curves(df: pd.DataFrame, out: Path) -> None:
    agg = df.groupby("epoch")["rel_l2"].agg(["mean", "median", _quantile(0.9), _quantile(0.99)])
    fig, ax = plt.subplots(figsize=(8, 5))
    for col in agg.columns:
        ax.plot(agg.index, agg[col], marker="o", label=col)
    ax.set_xlabel("epoch")
    ax.set_ylabel("rel_l2")
    ax.set_yscale("log")
    ax.set_title("Validation rel_l2 across training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "01_training_dynamics.png", dpi=130)
    plt.close(fig)


def plot_regime_stratification(latest: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, col in zip(axes, ["t_s", "t_bar", "R_c"]):
        bins = pd.qcut(latest[col], 8, duplicates="drop")
        grouped = [g["rel_l2"].values for _, g in latest.groupby(bins, observed=True)]
        labels = [f"{i.left:.2f}-{i.right:.2f}" for i in bins.cat.categories]
        ax.boxplot(grouped, tick_labels=labels, showfliers=False)
        ax.set_xlabel(col)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("rel_l2")
    axes[0].set_yscale("log")
    fig.suptitle("Error stratified by conditioning variables (final epoch)")
    fig.tight_layout()
    fig.savefig(out / "02_regime_stratification.png", dpi=130)
    plt.close(fig)


def plot_lead_time(latest: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = pd.qcut(latest["t_bar"], 20, duplicates="drop")
    agg = latest.groupby(bins, observed=True)["rel_l2"].agg(["mean", _quantile(0.9), _quantile(0.99)])
    centers = [i.mid for i in agg.index]
    for col in agg.columns:
        ax.plot(centers, agg[col], marker="o", label=col)
    ax.set_xlabel("lead time t_bar")
    ax.set_ylabel("rel_l2")
    ax.set_yscale("log")
    ax.set_title("Error vs lead time (should grow monotonically with horizon)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "03_error_vs_lead_time.png", dpi=130)
    plt.close(fig)


def plot_interface_ratio(latest: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ratio = latest["iface_rel_l2"] / latest["rel_l2"].clip(lower=1e-12)
    bins = pd.qcut(latest["R_c"], 12, duplicates="drop")
    df = pd.DataFrame({"R_c_bin": bins, "ratio": ratio})
    agg = df.groupby("R_c_bin", observed=True)["ratio"].agg(["mean", "median", _quantile(0.9)])
    centers = [i.mid for i in agg.index]
    for col in agg.columns:
        ax.plot(centers, agg[col], marker="o", label=col)
    ax.axhline(1.0, color="k", linestyle="--", alpha=0.5, label="iface == bulk")
    ax.set_xlabel("R_c")
    ax.set_ylabel("iface_rel_l2 / rel_l2")
    ax.set_title("Interface error relative to bulk, vs contact resistance")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "04_interface_vs_bulk.png", dpi=130)
    plt.close(fig)


def plot_ic_family_heatmap(latest: pd.DataFrame, out: Path) -> None:
    pivot = latest.pivot_table(
        index="temporal_family",
        columns="spatial_family",
        values="rel_l2",
        aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:.3f}", ha="center", va="center", color="w", fontsize=9)
    fig.colorbar(im, ax=ax, label="mean rel_l2")
    ax.set_title("Mean rel_l2 by IC family (final epoch)")
    fig.tight_layout()
    fig.savefig(out / "05_ic_family_heatmap.png", dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Path to val_pairs.csv")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory for plots (default: <repo>/visual/val_pairs/)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    out = args.out or repo_root / "visual" / "val_pairs"
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.csv} ...")
    df = pd.read_csv(args.csv)
    print(f"Rows: {len(df):,}  Epochs logged: {sorted(df.epoch.unique())}  Sims: {df.sim_id.nunique()}")

    latest = df[df.epoch == df.epoch.max()].copy()

    report_training_dynamics(df)
    report_worst_sims(latest)
    report_regime_stratification(latest)
    report_ic_family(latest)
    report_between_vs_within(latest)
    report_tails(latest)

    plot_training_curves(df, out)
    plot_regime_stratification(latest, out)
    plot_lead_time(latest, out)
    plot_interface_ratio(latest, out)
    plot_ic_family_heatmap(latest, out)

    print(f"\nPlots written to {out}/")


if __name__ == "__main__":
    main()
