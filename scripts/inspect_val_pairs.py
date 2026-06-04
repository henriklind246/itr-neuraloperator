"""Inspect a val_pairs.csv produced during training.

Run:
    python scripts/inspect_val_pairs.py <path/to/val_pairs.csv> [--out <dir>]

Produces a text report on stdout and a set of diagnostic PNGs in --out
(defaults to <repo>/visual/val_pairs/).

The CSV carries a ``benchmark`` column (one of ``forcing``, ``interfaces``,
``source``). Universal plots (training dynamics, lead-time error, conditioning
stratification, interface-vs-bulk) are emitted for every benchmark. The final
"structure" plot is benchmark-specific:

* forcing    -> mean error by (temporal_family x spatial_family) IC family
* interfaces -> error vs interface_x
* source     -> error by patch regime + scatter over (x_h, A)
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Conditioning columns to stratify by, per benchmark. The base set applies to
# all benchmarks; benchmark extras are appended when present and varying.
BASE_COND_COLS = ["t_s", "t_bar", "R_c"]
BENCHMARK_COND_COLS = {
    "forcing": [],
    "interfaces": ["interface_x"],
    "source": ["x_h", "y_h", "A"],
}


def _quantile(q: float):
    def _f(s: pd.Series) -> float:
        return float(s.quantile(q))
    _f.__name__ = f"p{int(q * 100)}"
    return _f


def _is_varying_numeric(df: pd.DataFrame, col: str) -> bool:
    return (
        col in df.columns
        and pd.api.types.is_numeric_dtype(df[col])
        and df[col].nunique(dropna=True) > 1
    )


def _detect_benchmark(df: pd.DataFrame, csv_path: Path) -> str:
    """Resolve the benchmark from the CSV column, a sibling config, or columns."""
    if "benchmark" in df.columns:
        vals = [str(b) for b in df["benchmark"].dropna().unique() if str(b) != ""]
        if len(vals) == 1:
            return vals[0]

    cfg = csv_path.parent / "config_used.yaml"
    if cfg.is_file():
        name = _benchmark_from_config(cfg)
        if name is not None:
            return name

    # Legacy CSVs (no benchmark column): infer from populated columns.
    if "x_h" in df.columns and df["x_h"].notna().any():
        return "source"
    if _is_varying_numeric(df, "interface_x"):
        return "interfaces"
    return "forcing"


def _benchmark_from_config(cfg_path: Path) -> str | None:
    """Minimal scan for ``benchmark:\\n  name: <x>`` in an OmegaConf YAML dump."""
    in_benchmark = False
    for line in cfg_path.read_text().splitlines():
        if re.match(r"^benchmark:\s*$", line):
            in_benchmark = True
            continue
        if in_benchmark:
            m = re.match(r"^\s+name:\s*(\S+)\s*$", line)
            if m:
                return m.group(1).strip().strip("'\"")
            if re.match(r"^\S", line):  # dedented out of the benchmark block
                in_benchmark = False
    return None


def _cond_cols(df: pd.DataFrame, benchmark: str) -> list[str]:
    cols = BASE_COND_COLS + BENCHMARK_COND_COLS.get(benchmark, [])
    return [c for c in cols if _is_varying_numeric(df, c)]


# --------------------------------------------------------------------------- #
# text reports
# --------------------------------------------------------------------------- #

def report_training_dynamics(df: pd.DataFrame) -> None:
    print("\n=== 1. Per-epoch validation error ===")
    agg = df.groupby("epoch")["rel_l2"].agg(["mean", "median", _quantile(0.9), _quantile(0.99)])
    print(agg.to_string(float_format=lambda x: f"{x:.4f}"))


def report_worst_sims(latest: pd.DataFrame) -> None:
    print("\n=== 2. Hardest sims (top-20 mean rel_l2 at final epoch) ===")
    per_sim = latest.groupby("sim_id")["rel_l2"].agg(["mean", "count"]).sort_values("mean")
    print(per_sim.tail(20).to_string(float_format=lambda x: f"{x:.4f}"))


def report_regime_stratification(latest: pd.DataFrame, cond_cols: list[str]) -> None:
    print("\n=== 3. Error stratified by conditioning variables (final epoch) ===")
    if not cond_cols:
        print("(no varying conditioning columns found)")
        return
    for col in cond_cols:
        bins = pd.qcut(latest[col], 5, duplicates="drop")
        agg = latest.groupby(bins, observed=True)["rel_l2"].agg(["mean", "median", _quantile(0.99)])
        print(f"\n-- by {col} quintile --")
        print(agg.to_string(float_format=lambda x: f"{x:.4f}"))


def report_structure(latest: pd.DataFrame, benchmark: str) -> None:
    print(f"\n=== 4. Structure breakdown for benchmark={benchmark!r} (final epoch) ===")
    if benchmark == "forcing" and {"temporal_family", "spatial_family"} <= set(latest.columns):
        agg = latest.groupby(["temporal_family", "spatial_family"])["rel_l2"].agg(
            ["mean", "median", "count"]
        )
        print(agg.sort_values("mean").to_string(float_format=lambda x: f"{x:.4f}"))
    elif benchmark == "interfaces" and _is_varying_numeric(latest, "interface_x"):
        bins = pd.qcut(latest["interface_x"], 5, duplicates="drop")
        agg = latest.groupby(bins, observed=True)["rel_l2"].agg(["mean", "median", _quantile(0.99)])
        print(agg.to_string(float_format=lambda x: f"{x:.4f}"))
    elif benchmark == "source" and "regime" in latest.columns:
        agg = latest.groupby("regime")["rel_l2"].agg(["mean", "median", "count"])
        print(agg.sort_values("mean").to_string(float_format=lambda x: f"{x:.4f}"))
    else:
        print("(no structure columns available for this benchmark)")


def report_between_vs_within(latest: pd.DataFrame) -> None:
    print("\n=== 5. Between-sim vs within-sim variance decomposition ===")
    sim_means = latest.groupby("sim_id")["rel_l2"].mean()
    between = sim_means.var()
    within = latest.groupby("sim_id")["rel_l2"].var().mean()
    total = between + within
    if not total or pd.isna(total):
        print("(insufficient data)")
        return
    print(f"Between-sim variance: {between:.4f}  ({between / total:.1%} of total)")
    print(f"Within-sim variance:  {within:.4f}  ({within / total:.1%} of total)")
    print(
        "High between-sim share -> coverage / OOD problem.  "
        "High within-sim share -> regime / conditioning problem."
    )


def report_tails(latest: pd.DataFrame, cond_cols: list[str]) -> None:
    print("\n=== 6. Tail vs mean (final epoch) ===")
    s = latest["rel_l2"]
    print(
        f"mean={s.mean():.4f}  p50={s.median():.4f}  "
        f"p90={s.quantile(0.9):.4f}  p99={s.quantile(0.99):.4f}  max={s.max():.4f}"
    )
    print("\nTop-20 worst pairs:")
    struct_cols = [c for c in ("temporal_family", "spatial_family", "regime") if c in latest.columns]
    cols = ["sim_id", *struct_cols, *cond_cols, "rel_l2"]
    if "iface_rel_l2" in latest.columns:
        cols.append("iface_rel_l2")
    cols = list(dict.fromkeys(cols))  # dedupe, preserve order
    print(latest.nlargest(20, "rel_l2")[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# --------------------------------------------------------------------------- #
# plots
# --------------------------------------------------------------------------- #

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


def plot_regime_stratification(latest: pd.DataFrame, cond_cols: list[str], out: Path) -> None:
    if not cond_cols:
        print("  [skip] 02_regime_stratification: no varying conditioning columns")
        return
    n = len(cond_cols)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True, squeeze=False)
    for ax, col in zip(axes[0], cond_cols):
        bins = pd.qcut(latest[col], 8, duplicates="drop")
        grouped = [g["rel_l2"].values for _, g in latest.groupby(bins, observed=True)]
        labels = [f"{i.left:.2f}-{i.right:.2f}" for i in bins.cat.categories]
        ax.boxplot(grouped, tick_labels=labels, showfliers=False)
        ax.set_xlabel(col)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3)
    axes[0][0].set_ylabel("rel_l2")
    axes[0][0].set_yscale("log")
    fig.suptitle("Error stratified by conditioning variables (final epoch)")
    fig.tight_layout()
    fig.savefig(out / "02_regime_stratification.png", dpi=130)
    plt.close(fig)


def plot_lead_time(latest: pd.DataFrame, out: Path) -> None:
    if not _is_varying_numeric(latest, "t_bar"):
        print("  [skip] 03_error_vs_lead_time: t_bar missing or constant")
        return
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
    if "iface_rel_l2" not in latest.columns or not _is_varying_numeric(latest, "R_c"):
        print("  [skip] 04_interface_vs_bulk: iface_rel_l2 or varying R_c missing")
        return
    if latest["iface_rel_l2"].abs().sum() == 0:
        print("  [skip] 04_interface_vs_bulk: iface_rel_l2 all zero")
        return
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


def plot_structure(latest: pd.DataFrame, benchmark: str, out: Path) -> None:
    """Benchmark-specific final plot (the 5th figure)."""
    if benchmark == "forcing":
        _plot_ic_family_heatmap(latest, out)
    elif benchmark == "interfaces":
        _plot_error_vs_interface_x(latest, out)
    elif benchmark == "source":
        _plot_source_structure(latest, out)
    else:
        print(f"  [skip] 05_structure: no structure plot for benchmark={benchmark!r}")


def _plot_ic_family_heatmap(latest: pd.DataFrame, out: Path) -> None:
    if not {"temporal_family", "spatial_family"} <= set(latest.columns):
        print("  [skip] 05_ic_family_heatmap: family columns missing")
        return
    pivot = latest.pivot_table(
        index="temporal_family", columns="spatial_family", values="rel_l2", aggfunc="mean",
    )
    if pivot.size == 0:
        print("  [skip] 05_ic_family_heatmap: no family data")
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", color="w", fontsize=9)
    fig.colorbar(im, ax=ax, label="mean rel_l2")
    ax.set_title("Mean rel_l2 by IC family (final epoch)")
    fig.tight_layout()
    fig.savefig(out / "05_ic_family_heatmap.png", dpi=130)
    plt.close(fig)


def _plot_error_vs_interface_x(latest: pd.DataFrame, out: Path) -> None:
    if not _is_varying_numeric(latest, "interface_x"):
        print("  [skip] 05_error_vs_interface_x: interface_x missing or constant")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = pd.qcut(latest["interface_x"], 12, duplicates="drop")
    agg = latest.groupby(bins, observed=True)["rel_l2"].agg(["mean", "median", _quantile(0.9)])
    centers = [i.mid for i in agg.index]
    for col in agg.columns:
        ax.plot(centers, agg[col], marker="o", label=col)
    ax.set_xlabel("interface_x")
    ax.set_ylabel("rel_l2")
    ax.set_yscale("log")
    ax.set_title("Error vs interface position (final epoch)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "05_error_vs_interface_x.png", dpi=130)
    plt.close(fig)


def _plot_source_structure(latest: pd.DataFrame, out: Path) -> None:
    has_regime = "regime" in latest.columns and latest["regime"].nunique() > 1
    has_scatter = _is_varying_numeric(latest, "x_h") and _is_varying_numeric(latest, "A")
    if not (has_regime or has_scatter):
        print("  [skip] 05_source_structure: regime/x_h/A columns missing")
        return
    n = int(has_regime) + int(has_scatter)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    col_idx = 0
    if has_regime:
        ax = axes[0][col_idx]
        order = ["left", "near", "right"]
        present = [r for r in order if r in set(latest["regime"])]
        grouped = [latest.loc[latest["regime"] == r, "rel_l2"].values for r in present]
        ax.boxplot(grouped, tick_labels=present, showfliers=False)
        ax.set_xlabel("patch regime (vs interface)")
        ax.set_ylabel("rel_l2")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        col_idx += 1
    if has_scatter:
        ax = axes[0][col_idx]
        sc = ax.scatter(latest["x_h"], latest["A"], c=latest["rel_l2"], cmap="viridis", s=12)
        fig.colorbar(sc, ax=ax, label="rel_l2")
        ax.set_xlabel("x_h (patch x)")
        ax.set_ylabel("A (amplitude)")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Source patch error structure (final epoch)")
    fig.tight_layout()
    fig.savefig(out / "05_source_structure.png", dpi=130)
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
    benchmark = _detect_benchmark(df, Path(args.csv))
    cond_cols = _cond_cols(df, benchmark)
    print(
        f"Benchmark: {benchmark}  Rows: {len(df):,}  "
        f"Epochs logged: {sorted(df.epoch.unique())}  Sims: {df.sim_id.nunique()}"
    )
    print(f"Conditioning columns: {cond_cols}")

    latest = df[df.epoch == df.epoch.max()].copy()

    report_training_dynamics(df)
    report_worst_sims(latest)
    report_regime_stratification(latest, cond_cols)
    report_structure(latest, benchmark)
    report_between_vs_within(latest)
    report_tails(latest, cond_cols)

    plot_training_curves(df, out)
    plot_regime_stratification(latest, cond_cols, out)
    plot_lead_time(latest, out)
    plot_interface_ratio(latest, out)
    plot_structure(latest, benchmark, out)

    print(f"\nPlots written to {out}/")


if __name__ == "__main__":
    main()
