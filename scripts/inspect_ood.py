"""Aggregate OOD ``test_records.csv`` files into a hierarchical per-axis report.

Run:
    python scripts/inspect_ood.py <records.csv> [<records.csv> ...] [--out <dir>]

Each input CSV is a ``test_records.csv`` written by ``write_test_records`` with
the OOD/protocol/horizon columns and pooled sufficient statistics (Step 4). The
model seed is read from the parent directory name (``seed42`` -> ``42``); pass
``--seed-from-path`` off only when every CSV already carries a distinct seed via
``--seed``.

Aggregation is a strict hierarchy and never pools raw pair rows across levels:

1. Pair -> simulation/bucket. All pairs of one
   ``(seed, sim_id, ood_axis, ood_value, protocol, source_time_actual,
   target_time_actual, lead_time_actual)`` bucket reduce to ONE per-simulation
   metric by POOLING squared errors from the stored sufficient statistics:
   ``RMSE_sim = sqrt(sum sse_K2 / sum num_error_cells)`` and
   ``relL2_sim = sqrt(sum sse_K2 / sum target_sse_K2)`` (interface analogues from
   the interface sufficient stats). The arithmetic mean of per-pair RMSEs is a
   different statistic, reported separately as ``mean_pair_rmse_K``.
2. Simulation -> repeat distribution. Per-simulation metrics group by
   ``(seed, ood_axis, ood_value, protocol, source/target/lead)`` across
   ``ood_repeat`` and report mean / median / CI / max, plus the CRN paired delta
   vs the in-distribution reference at the matching repeat.
3. Repeat -> seed. Per-seed value summaries roll across model seeds for the final
   mean +/- spread.

``rmse_K`` (physical Kelvin) is the primary statistic; ``rel_l2_pct`` is
secondary and unreliable in low-signal cells (small denominator). Resolution
columns (``eff_cells_per_scale``/``eff_steps_per_scale``) are carried through so
under-resolved cells can be annotated in the plots.
"""

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Identity of one per-simulation bucket (Step 1). Includes source/target/lead so
# a curve never pools different prediction tasks (e.g. fixed_initial vs
# anchored_from_horizon, or different leads).
BUCKET_KEYS = [
    "seed", "benchmark", "ood_axis", "ood_value", "ood_repeat",
    "distribution_class", "protocol",
    "source_time_actual", "target_time_actual", "lead_time_actual",
    "sim_id",
]

# Per-(axis,value,protocol,source/target/lead) cell within one seed (Step 2).
VALUE_KEYS = [
    "seed", "benchmark", "ood_axis", "ood_value", "distribution_class",
    "protocol", "source_time_actual", "target_time_actual", "lead_time_actual",
]

# Same cell rolled across seeds (Step 3).
CELL_KEYS = [k for k in VALUE_KEYS if k != "seed"]

# Sufficient-statistic sums pooled in Step 1.
_SUM_COLS = [
    "sse_K2", "num_error_cells", "target_sse_K2",
    "interface_sse_K2", "num_interface_cells", "interface_target_sse_K2",
]

# Resolution columns carried through untouched (filled by Step 6).
_RES_COLS = ["eff_cells_per_scale", "eff_steps_per_scale"]

# Numeric metadata averaged through every reduction (resolution + horizons) so
# the plots can read e.g. the trained horizon at the cell level.
_CARRY_NUM = _RES_COLS + ["time_norm_horizon", "dataset_t_final"]

_GROUP_STR_FILL = {
    "benchmark": "", "ood_axis": "", "ood_value": "", "distribution_class": "",
    "protocol": "", "latents_hash": "",
}


def _seed_from_path(csv_path: Path) -> str:
    """Return the seed tag from a ``.../seed42/test_records.csv`` style path."""
    name = csv_path.parent.name
    if name.startswith("seed"):
        return name[len("seed"):]
    return name


def load_records(paths, seed_from_path: bool = True) -> pd.DataFrame:
    """Read and concatenate OOD ``test_records.csv`` files into one DataFrame.

    Adds a ``seed`` column (from the parent dir name unless already present),
    fills absent OOD/protocol columns with in-distribution defaults so legacy
    records load, and coerces a numeric ``ood_value_num`` for plotting (NaN for
    compound string values like ``exp+patch``).
    """
    frames = []
    for p in paths:
        p = Path(p)
        df = pd.read_csv(p)
        if "seed" not in df.columns or df["seed"].isna().all():
            if seed_from_path:
                df["seed"] = _seed_from_path(p)
        df["_source_csv"] = str(p)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    for col, fill in _GROUP_STR_FILL.items():
        if col not in df.columns:
            df[col] = fill
        df[col] = df[col].fillna(fill).astype(str)
    df.loc[df["distribution_class"].isin(["", "nan"]), "distribution_class"] = (
        "in_distribution"
    )

    for col in ["ood_repeat", "source_time_actual", "target_time_actual",
                "lead_time_actual", "source_time_requested",
                "target_time_requested", "dataset_t_final", "time_norm_horizon"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in _SUM_COLS:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in _RES_COLS:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["seed"] = df.get("seed", "").astype(str)
    df["ood_repeat"] = df["ood_repeat"].fillna(-1)
    df["ood_value_num"] = pd.to_numeric(df["ood_value"], errors="coerce")
    return df


# The pooling identities live in visual/pub/stats.py so this script and the
# publication figures cannot drift apart on what "per-simulation RMSE" means.
from visual.pub.stats import pooled_rel_l2_pct as _pooled_rel_l2_pct  # noqa: E402
from visual.pub.stats import pooled_rmse as _pooled_rmse  # noqa: E402


def reduce_pairs_to_sims(df: pd.DataFrame) -> pd.DataFrame:
    """Step 1: pool a bucket's pairs into one per-simulation metric row.

    Pools squared errors from the sufficient statistics (NOT a mean of pair
    RMSEs). ``mean_pair_rmse_K`` is the secondary arithmetic mean reported beside
    the pooled value.
    """
    have_pair_rmse = "rmse_K" in df.columns
    out_rows = []
    for keys, g in df.groupby(BUCKET_KEYS, dropna=False):
        rec = dict(zip(BUCKET_KEYS, keys))
        sums = {c: float(np.nansum(g[c].to_numpy(dtype=float))) for c in _SUM_COLS}
        rec["n_pairs"] = int(len(g))
        rec["rmse_K"] = _pooled_rmse(sums["sse_K2"], sums["num_error_cells"])
        rec["rel_l2_pct"] = _pooled_rel_l2_pct(sums["sse_K2"], sums["target_sse_K2"])
        rec["iface_rmse_K"] = _pooled_rmse(
            sums["interface_sse_K2"], sums["num_interface_cells"]
        )
        rec["iface_rel_l2_pct"] = _pooled_rel_l2_pct(
            sums["interface_sse_K2"], sums["interface_target_sse_K2"]
        )
        if have_pair_rmse:
            rec["mean_pair_rmse_K"] = float(np.nanmean(g["rmse_K"].to_numpy(dtype=float)))
        else:
            rec["mean_pair_rmse_K"] = float("nan")
        for c in _CARRY_NUM:
            vals = g[c].to_numpy(dtype=float)
            rec[c] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
        rec["ood_value_num"] = pd.to_numeric(
            pd.Series([rec["ood_value"]]), errors="coerce"
        ).iloc[0]
        out_rows.append(rec)
    return pd.DataFrame(out_rows)


def _ci95(std: float, n: int) -> float:
    if n <= 1 or not np.isfinite(std):
        return float("nan")
    return 1.96 * std / math.sqrt(n)


def _summary_stats(values: np.ndarray, prefix: str) -> dict:
    v = values[np.isfinite(values)]
    n = int(v.size)
    if n == 0:
        return {f"{prefix}_mean": float("nan"), f"{prefix}_median": float("nan"),
                f"{prefix}_std": float("nan"), f"{prefix}_ci95": float("nan"),
                f"{prefix}_max": float("nan"), f"{prefix}_n": 0}
    mean = float(np.mean(v))
    std = float(np.std(v, ddof=1)) if n > 1 else 0.0
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_median": float(np.median(v)),
        f"{prefix}_std": std,
        f"{prefix}_ci95": _ci95(std, n),
        f"{prefix}_max": float(np.max(v)),
        f"{prefix}_n": n,
    }


def _baseline_value(sim_df: pd.DataFrame, axis: str) -> str | None:
    """Pick the in-distribution reference ``ood_value`` for an axis.

    The CRN sweep leads with the ID reference; pick the in_distribution value
    whose numeric is closest to the median of the in_distribution values, which
    is unambiguous for the common single-reference case.
    """
    sub = sim_df[(sim_df["ood_axis"] == axis)
                 & (sim_df["distribution_class"] == "in_distribution")]
    if sub.empty:
        return None
    vals = sub["ood_value"].unique()
    nums = pd.to_numeric(pd.Series(vals), errors="coerce")
    if nums.notna().all():
        med = float(np.median(nums))
        return str(vals[int(np.argmin(np.abs(nums.to_numpy() - med)))])
    return str(sorted(vals)[0])


def summarize_across_repeats(sim_df: pd.DataFrame) -> pd.DataFrame:
    """Step 2: per-seed cell summary across OOD repeats, with paired deltas.

    Groups per-simulation rows by ``VALUE_KEYS`` and reports rmse_K and
    rel_l2_pct mean/median/std/CI95/max/n across repeats. ``paired_delta_rmse_K``
    is the mean over repeats of ``RMSE_sim(value, r) - RMSE_sim(id_ref, r)`` at
    the matching seed/repeat/protocol/source-context (NaN when no paired
    reference exists).
    """
    rows = []
    for keys, g in sim_df.groupby(VALUE_KEYS, dropna=False):
        rec = dict(zip(VALUE_KEYS, keys))
        rec.update(_summary_stats(g["rmse_K"].to_numpy(dtype=float), "rmse_K"))
        rec.update(_summary_stats(g["rel_l2_pct"].to_numpy(dtype=float), "rel_l2_pct"))
        rec.update(_summary_stats(g["iface_rmse_K"].to_numpy(dtype=float), "iface_rmse_K"))
        rec["mean_pair_rmse_K"] = float(
            np.nanmean(g["mean_pair_rmse_K"].to_numpy(dtype=float))
        )
        rec["n_sims"] = int(len(g))
        rec["n_repeats"] = int(g["ood_repeat"].nunique())
        for c in _CARRY_NUM:
            vals = g[c].to_numpy(dtype=float)
            rec[c] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
        rec["ood_value_num"] = pd.to_numeric(
            pd.Series([rec["ood_value"]]), errors="coerce"
        ).iloc[0]
        rows.append(rec)
    out = pd.DataFrame(rows)
    out = _attach_paired_deltas(sim_df, out)
    return out


def _attach_paired_deltas(sim_df: pd.DataFrame, value_df: pd.DataFrame) -> pd.DataFrame:
    """Add ``paired_delta_rmse_K`` to a per-seed value summary (CRN matching)."""
    deltas = []
    # Map (seed, axis, protocol, source, target, lead, repeat, value) -> RMSE_sim.
    key_cols = ["seed", "ood_axis", "protocol", "source_time_actual",
                "target_time_actual", "lead_time_actual", "ood_repeat", "ood_value"]
    lut = {
        tuple(r[c] for c in key_cols): r["rmse_K"]
        for _, r in sim_df.iterrows()
    }
    baselines = {
        axis: _baseline_value(sim_df, axis)
        for axis in sim_df["ood_axis"].unique()
    }
    for _, row in value_df.iterrows():
        axis = row["ood_axis"]
        base_val = baselines.get(axis)
        if base_val is None or str(row["ood_value"]) == str(base_val):
            deltas.append(float("nan"))
            continue
        sub = sim_df[
            (sim_df["seed"] == row["seed"])
            & (sim_df["ood_axis"] == axis)
            & (sim_df["protocol"] == row["protocol"])
            & (sim_df["ood_value"] == row["ood_value"])
        ]
        diffs = []
        for _, sr in sub.iterrows():
            bkey = (sr["seed"], axis, sr["protocol"], sr["source_time_actual"],
                    sr["target_time_actual"], sr["lead_time_actual"],
                    sr["ood_repeat"], base_val)
            base_rmse = lut.get(bkey)
            if base_rmse is None:
                # Time axis: baseline target differs; match on repeat only.
                cand = sim_df[
                    (sim_df["seed"] == sr["seed"])
                    & (sim_df["ood_axis"] == axis)
                    & (sim_df["protocol"] == sr["protocol"])
                    & (sim_df["ood_repeat"] == sr["ood_repeat"])
                    & (sim_df["ood_value"] == base_val)
                ]
                if cand.empty:
                    continue
                base_rmse = float(cand["rmse_K"].mean())
            if np.isfinite(base_rmse) and np.isfinite(sr["rmse_K"]):
                diffs.append(float(sr["rmse_K"]) - float(base_rmse))
        deltas.append(float(np.mean(diffs)) if diffs else float("nan"))
    value_df = value_df.copy()
    value_df["paired_delta_rmse_K"] = deltas
    return value_df


def rollup_across_seeds(value_df: pd.DataFrame) -> pd.DataFrame:
    """Step 3: roll per-seed cell means across model seeds (mean +/- spread)."""
    rows = []
    for keys, g in value_df.groupby(CELL_KEYS, dropna=False):
        rec = dict(zip(CELL_KEYS, keys))
        rec.update(_summary_stats(g["rmse_K_mean"].to_numpy(dtype=float), "seed_rmse_K"))
        rec.update(_summary_stats(
            g["rel_l2_pct_mean"].to_numpy(dtype=float), "seed_rel_l2_pct"
        ))
        rec.update(_summary_stats(
            g["paired_delta_rmse_K"].to_numpy(dtype=float), "seed_paired_delta_rmse_K"
        ))
        rec["n_seeds"] = int(g["seed"].nunique())
        rec["n_sims_total"] = int(g["n_sims"].sum())
        for c in _CARRY_NUM:
            vals = g[c].to_numpy(dtype=float)
            rec[c] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
        rec["ood_value_num"] = pd.to_numeric(
            pd.Series([rec["ood_value"]]), errors="coerce"
        ).iloc[0]
        rows.append(rec)
    return pd.DataFrame(rows)


def _is_under_resolved(row) -> bool:
    """Annotate a cell whose swept feature is at/under its adequacy threshold.

    The generator records effective discretization per sim
    (``eff_cells_per_scale`` for spatial features, ``eff_steps_per_scale`` for
    temporal) under per-feature-type criteria. These thresholds mirror the "ok"
    boundary used there (spatial: 4 cells per feature scale; temporal: 8 steps
    per timescale), so any cell the generator flags "watch"/"under_resolved" is
    annotated here. Only one of the two columns is finite per axis.
    """
    cells = row.get("eff_cells_per_scale", float("nan"))
    steps = row.get("eff_steps_per_scale", float("nan"))
    if np.isfinite(cells) and cells < 4.0:
        return True
    if np.isfinite(steps) and steps < 8.0:
        return True
    return False


def plot_axis_sweep(cell_df: pd.DataFrame, axis: str, out_dir: Path) -> Path | None:
    """Per-simulation error vs OOD value with the in-distribution baseline band.

    Used for ``simulation_parameter``/``compound`` axes (one OOD value per cell).
    """
    sub = cell_df[(cell_df["ood_axis"] == axis) & (cell_df["protocol"] == "")].copy()
    if sub.empty:
        return None
    sub = sub.sort_values("ood_value_num", na_position="last")
    numeric = sub["ood_value_num"].notna().all()
    x = sub["ood_value_num"].to_numpy() if numeric else np.arange(len(sub))
    y = sub["seed_rmse_K_mean"].to_numpy(dtype=float)
    yerr = sub["seed_rmse_K_ci95"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(x, y, yerr=yerr, marker="o", lw=1.5, capsize=3, label="RMSE_sim mean")

    id_mask = sub["distribution_class"].to_numpy() == "in_distribution"
    if id_mask.any():
        band = y[id_mask]
        lo, hi = float(np.nanmin(band)), float(np.nanmax(band))
        ax.axhspan(lo, hi, color="tab:green", alpha=0.12,
                   label="in-distribution band")

    for xi, (_, r) in zip(x, sub.iterrows()):
        if _is_under_resolved(r):
            ax.axvline(xi, color="0.6", ls=":", lw=1.0)
            ax.annotate("res?", (xi, np.nanmax(y)), fontsize=7, color="0.4",
                        ha="center", va="bottom")

    if not numeric:
        ax.set_xticks(x)
        ax.set_xticklabels(sub["ood_value"].astype(str), rotation=30, ha="right")
    ax.set_xlabel(f"OOD value ({axis})")
    ax.set_ylabel("RMSE_sim (K)")
    ax.set_title(f"OOD sweep: {axis}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = out_dir / f"ood_sweep_{axis}.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_time_protocols(cell_df: pd.DataFrame, out_dir: Path,
                        horizon: float | None = None) -> Path | None:
    """Time axis: error vs target time, one line per protocol, boundary marked."""
    sub = cell_df[cell_df["protocol"] != ""].copy()
    sub = sub[sub["target_time_actual"].notna()]
    if sub.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for protocol, g in sub.groupby("protocol"):
        g = g.sort_values("target_time_actual")
        ax.errorbar(
            g["target_time_actual"].to_numpy(dtype=float),
            g["seed_rmse_K_mean"].to_numpy(dtype=float),
            yerr=g["seed_rmse_K_ci95"].to_numpy(dtype=float),
            marker="o", lw=1.5, capsize=3, label=protocol,
        )
    if horizon is None and sub["time_norm_horizon"].notna().any():
        horizon = float(sub["time_norm_horizon"].dropna().iloc[0])
    if horizon is not None:
        ax.axvline(horizon, color="k", ls="--", lw=1.0,
                   label=f"trained horizon {horizon:g}")
    ax.set_xlabel("target time t_j")
    ax.set_ylabel("RMSE_sim (K)")
    ax.set_title("OOD time extrapolation by protocol")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = out_dir / "ood_time_protocols.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def _axis_frame(cell_df: pd.DataFrame, axis: str):
    """Sorted per-axis sweep rows with an ``id_offset`` column (0 = ID anchor).

    Restricts to the simulation-parameter rows (``protocol == ""``) so the
    consolidated views never mix in the time-extrapolation protocols.
    """
    sub = cell_df[(cell_df["ood_axis"] == axis) & (cell_df["protocol"] == "")].copy()
    if sub.empty:
        return None
    sub = sub.sort_values("ood_value_num", na_position="last").reset_index(drop=True)
    id_mask = sub["distribution_class"].to_numpy() == "in_distribution"
    id_idx = int(np.flatnonzero(id_mask)[0]) if id_mask.any() else len(sub) // 2
    sub["id_offset"] = np.arange(len(sub)) - id_idx
    return sub, id_idx


def _axis_baseline_rmse(sub: pd.DataFrame, id_idx: int) -> float:
    """In-distribution RMSE_K for an axis; falls back to the swept minimum."""
    vals = sub["seed_rmse_K_mean"].to_numpy(dtype=float)
    base = vals[id_idx]
    if not np.isfinite(base) or base <= 0:
        finite = vals[np.isfinite(vals)]
        base = float(np.min(finite)) if finite.size else float("nan")
    return float(base)


def _degradation_table(cell_df: pd.DataFrame) -> pd.DataFrame:
    """Per-axis worst-OOD degradation factor vs the in-distribution baseline."""
    rows = []
    for axis in sorted(a for a in cell_df["ood_axis"].unique() if a):
        res = _axis_frame(cell_df, axis)
        if res is None:
            continue
        sub, id_idx = res
        base = _axis_baseline_rmse(sub, id_idx)
        ood = sub[sub["distribution_class"] != "in_distribution"]
        if ood.empty or not np.isfinite(base) or base <= 0:
            continue
        rmse = ood["seed_rmse_K_mean"].to_numpy(dtype=float)
        if not np.isfinite(rmse).any():
            continue
        worst = float(np.nanmax(rmse))
        rows.append({"axis": axis, "baseline_K": base, "worst_K": worst,
                     "factor": worst / base})
    return pd.DataFrame(rows).sort_values("factor") if rows else pd.DataFrame()


def _draw_rank_degradation(ax, cell_df: pd.DataFrame) -> bool:
    """View 1: horizontal bars ranking axes by worst-OOD / baseline RMSE."""
    tbl = _degradation_table(cell_df)
    if tbl.empty:
        return False
    y = np.arange(len(tbl))
    ax.barh(y, tbl["factor"].to_numpy(dtype=float), color="tab:red", alpha=0.75)
    ax.axvline(1.0, color="0.4", ls="--", lw=1.0, label="in-distribution")
    for yi, (_, r) in zip(y, tbl.iterrows()):
        ax.annotate(f"{r['factor']:.1f}x  ({r['worst_K']:.3f} K)",
                    (r["factor"], yi), va="center", ha="left", fontsize=7,
                    xytext=(3, 0), textcoords="offset points")
    ax.set_yticks(y)
    ax.set_yticklabels(tbl["axis"].astype(str))
    ax.set_xlim(left=0.0)
    ax.margins(x=0.25)
    ax.set_xlabel("worst-OOD RMSE / in-distribution RMSE")
    ax.set_title("OOD degradation ranking")
    ax.legend(fontsize=8, loc="lower right")
    return True


def _draw_overlay_normalized(ax, cell_df: pd.DataFrame) -> bool:
    """View 2: all axes overlaid, RMSE / baseline on a shared log axis."""
    drew = False
    for axis in sorted(a for a in cell_df["ood_axis"].unique() if a):
        res = _axis_frame(cell_df, axis)
        if res is None:
            continue
        sub, id_idx = res
        base = _axis_baseline_rmse(sub, id_idx)
        if not np.isfinite(base) or base <= 0:
            continue
        y = sub["seed_rmse_K_mean"].to_numpy(dtype=float) / base
        x = sub["id_offset"].to_numpy(dtype=float)
        ax.plot(x, y, marker="o", lw=1.3, ms=4, label=axis)
        drew = True
    if not drew:
        return False
    ax.axhline(1.0, color="0.5", ls="--", lw=1.0)
    ax.axvline(0.0, color="tab:green", ls=":", lw=1.0)
    ax.set_yscale("log")
    ax.set_xlabel("sweep step relative to in-distribution (0 = ID)")
    ax.set_ylabel("RMSE / in-distribution RMSE")
    ax.set_title("Degradation vs in-distribution (shared log scale)")
    ax.legend(fontsize=7, ncol=2)
    return True


def _draw_overlay_rel_l2(ax, cell_df: pd.DataFrame) -> bool:
    """View 3: all axes overlaid in scale-free rel-L2 %, ID band shaded."""
    drew = False
    id_band = []
    for axis in sorted(a for a in cell_df["ood_axis"].unique() if a):
        res = _axis_frame(cell_df, axis)
        if res is None:
            continue
        sub, id_idx = res
        y = sub["seed_rel_l2_pct_mean"].to_numpy(dtype=float)
        x = sub["id_offset"].to_numpy(dtype=float)
        ax.plot(x, y, marker="o", lw=1.3, ms=4, label=axis)
        if np.isfinite(y[id_idx]):
            id_band.append(float(y[id_idx]))
        drew = True
    if not drew:
        return False
    if id_band:
        ax.axhspan(min(id_band), max(id_band), color="tab:green", alpha=0.12,
                   label="in-distribution band")
    ax.axvline(0.0, color="tab:green", ls=":", lw=1.0)
    ax.set_xlabel("sweep step relative to in-distribution (0 = ID)")
    ax.set_ylabel("rel-L2 (%)")
    ax.set_title("Relative L2 by axis (scale-free)")
    ax.legend(fontsize=7, ncol=2)
    return True


def _draw_delta_heatmap(ax, cell_df: pd.DataFrame, fig) -> bool:
    """View 4: axes x sweep-offset heatmap of paired RMSE delta (shared scale)."""
    per_axis = {}
    offsets: set[int] = set()
    for axis in sorted(a for a in cell_df["ood_axis"].unique() if a):
        res = _axis_frame(cell_df, axis)
        if res is None:
            continue
        sub, _ = res
        d = {int(o): float(v) for o, v in
             zip(sub["id_offset"], sub["seed_paired_delta_rmse_K_mean"])}
        if not any(np.isfinite(v) for v in d.values()):
            continue
        per_axis[axis] = d
        offsets.update(d.keys())
    if not per_axis:
        return False
    cols = sorted(offsets)
    col_idx = {o: i for i, o in enumerate(cols)}
    axes_names = list(per_axis.keys())
    mat = np.full((len(axes_names), len(cols)), np.nan)
    for r, axis in enumerate(axes_names):
        for o, v in per_axis[axis].items():
            mat[r, col_idx[o]] = v
    finite = mat[np.isfinite(mat)]
    vmax = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
    vmax = vmax if vmax > 0 else 1.0
    im = ax.imshow(mat, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([str(o) for o in cols], fontsize=7)
    ax.set_yticks(range(len(axes_names)))
    ax.set_yticklabels(axes_names, fontsize=8)
    if 0 in col_idx:
        ax.axvline(col_idx[0], color="k", ls=":", lw=1.0)
    ax.set_xlabel("sweep step relative to in-distribution (0 = ID)")
    ax.set_title("Paired RMSE delta vs in-distribution (K)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="delta RMSE (K)")
    return True


def plot_ood_consolidated(cell_df: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Emit the 4 consolidated metric views as standalone PNGs plus a 2x2 panel."""
    paths: list[Path] = []
    standalone = [
        ("ood_rank_degradation.png", _draw_rank_degradation, False),
        ("ood_overlay_normalized.png", _draw_overlay_normalized, False),
        ("ood_overlay_rel_l2.png", _draw_overlay_rel_l2, False),
        ("ood_delta_heatmap.png", _draw_delta_heatmap, True),
    ]
    for name, fn, needs_fig in standalone:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ok = fn(ax, cell_df, fig) if needs_fig else fn(ax, cell_df)
        if ok:
            fig.tight_layout()
            fig.savefig(out_dir / name, dpi=130)
            paths.append(out_dir / name)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    drawn = _draw_rank_degradation(axes[0, 0], cell_df)
    drawn = _draw_overlay_normalized(axes[0, 1], cell_df) or drawn
    drawn = _draw_overlay_rel_l2(axes[1, 0], cell_df) or drawn
    drawn = _draw_delta_heatmap(axes[1, 1], cell_df, fig) or drawn
    if drawn:
        bench = ""
        bvals = cell_df["benchmark"].dropna().unique()
        if bvals.size:
            bench = str(bvals[0])
        fig.suptitle(f"OOD overview: {bench}" if bench else "OOD overview")
        fig.tight_layout()
        fig.savefig(out_dir / "ood_overview.png", dpi=130)
        paths.append(out_dir / "ood_overview.png")
    plt.close(fig)
    return paths


def aggregate(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Run the full Step1->Step3 hierarchy; return the three frames."""
    sim_df = reduce_pairs_to_sims(df)
    value_df = summarize_across_repeats(sim_df)
    cell_df = rollup_across_seeds(value_df)
    return {"sim": sim_df, "value": value_df, "cell": cell_df}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Hierarchically aggregate OOD test records into a per-axis report.",
    )
    parser.add_argument("records", nargs="+",
                        help="One or more test_records.csv paths (one per seed).")
    parser.add_argument("--out", default=None,
                        help="Output directory (defaults to <repo>/visual/ood/).")
    parser.add_argument("--no-seed-from-path", action="store_true",
                        help="Do not infer the model seed from the parent dir name.")
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else (
        Path(__file__).resolve().parents[1] / "visual" / "ood"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_records(args.records, seed_from_path=not args.no_seed_from_path)
    frames = aggregate(df)
    sim_df, value_df, cell_df = frames["sim"], frames["value"], frames["cell"]

    sim_df.to_csv(out_dir / "ood_per_sim.csv", index=False)
    value_df.to_csv(out_dir / "ood_per_seed.csv", index=False)
    cell_df.to_csv(out_dir / "ood_summary.csv", index=False)

    plots = []
    for axis in sorted(a for a in cell_df["ood_axis"].unique() if a):
        p = plot_axis_sweep(cell_df, axis, out_dir)
        if p is not None:
            plots.append(p)
    tp = plot_time_protocols(cell_df, out_dir)
    if tp is not None:
        plots.append(tp)
    plots.extend(plot_ood_consolidated(cell_df, out_dir))

    print(f"Loaded {len(df)} pair rows across {df['seed'].nunique()} seed(s).")
    print(f"Per-sim buckets: {len(sim_df)}; per-seed cells: {len(value_df)}; "
          f"pooled cells: {len(cell_df)}.")
    print(f"Wrote 3 CSVs and {len(plots)} plot(s) -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
