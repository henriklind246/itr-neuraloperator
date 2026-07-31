"""The reduction layer. Every number in a published figure is produced here.

A row of ``test_records.csv`` is one ``(sim_id, s, j)`` snapshot pair. A single
simulation contributes ``S x J`` strongly correlated pairs, so pairs are *not*
independent observations and must never be the unit a statistic is computed
over. The four levels are:

======  ==========================================  ==================================
level   grouping                                    role
======  ==========================================  ==================================
0       ``(sim_id, s, j)``                          raw pair, correlated
1       ``(seed, benchmark, [strata], sim_id)``     **the unit of replication**
2       ``(seed, benchmark, [strata])``             median / IQR / p90 + bootstrap CI
3       ``(benchmark, [strata])``                   across model seeds, paired deltas
======  ==========================================  ==================================

Level 0 -> 1 uses pooled sufficient statistics, not a mean of per-pair metrics:
``rmse_K = sqrt(sum sse_K2 / sum num_error_cells)`` and
``rel_l2_pct = 100 * sqrt(sum sse_K2 / sum target_sse_K2)``. Those identities
are shared with ``scripts/inspect_ood.py``, which imports them from here.

This module must not read files and must not plot.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from visual.pub.manifest import Degradation

# Guard thresholds. Below these the corresponding statistic is emitted as NaN
# with a Degradation instead of being reported as if it were determined.
MIN_SIMS_FOR_CI = 20
MIN_SIMS_FOR_P99 = 100
MIN_SEEDS_FOR_SPREAD = 3

DEFAULT_RNG_SEED = 20260728
DEFAULT_N_BOOT = 10_000

Z_95 = 1.959963984540054


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricSpec:
    """Name, physical space, and pooling rule for one reported quantity."""

    name: str
    space: str          # "kelvin" | "normalized"
    unit: str
    label: str
    how: str            # "pooled" | "ratio" | "mean"
    numerator: str = ""
    denominator: str = ""
    direction: str = "lower_better"


METRICS: dict[str, MetricSpec] = {
    "rmse_K": MetricSpec(
        "rmse_K", "kelvin", "K", "Global RMSE", "pooled",
        numerator="sse_K2", denominator="num_error_cells"),
    "iface_rmse_K": MetricSpec(
        "iface_rmse_K", "kelvin", "K", "Interface-region RMSE", "pooled",
        numerator="interface_sse_K2", denominator="num_interface_cells"),
    "rel_l2_pct": MetricSpec(
        "rel_l2_pct", "normalized", "%", "Relative $L_2$", "ratio",
        numerator="sse_K2", denominator="target_sse_K2"),
    "iface_rel_l2_pct": MetricSpec(
        "iface_rel_l2_pct", "normalized", "%", "Interface relative $L_2$", "ratio",
        numerator="interface_sse_K2", denominator="interface_target_sse_K2"),
    "node_jump_rmse_K": MetricSpec(
        "node_jump_rmse_K", "kelvin", "K", "Node-jump RMSE", "mean"),
    "contact_jump_rmse_K": MetricSpec(
        "contact_jump_rmse_K", "kelvin", "K", "Contact-jump RMSE", "mean"),
    "gnrmse_pct": MetricSpec(
        "gnrmse_pct", "normalized", "%", "Global NRMSE", "mean"),
    "nrmse_pct": MetricSpec(
        "nrmse_pct", "normalized", "%", "NRMSE", "mean"),
}

POOLED_METRICS = tuple(m for m, s in METRICS.items() if s.how == "pooled")
RATIO_METRICS = tuple(m for m, s in METRICS.items() if s.how == "ratio")
MEAN_METRICS = tuple(m for m, s in METRICS.items() if s.how == "mean")

DEFAULT_METRICS = ("rmse_K", "rel_l2_pct", "iface_rmse_K", "node_jump_rmse_K")

_SUM_COLS = (
    "sse_K2", "num_error_cells", "target_sse_K2",
    "interface_sse_K2", "num_interface_cells", "interface_target_sse_K2",
)


def metric_spec(metric: str) -> MetricSpec:
    try:
        return METRICS[metric]
    except KeyError:
        raise KeyError(
            f"unknown metric {metric!r}; known: {sorted(METRICS)}"
        ) from None


def assert_same_space(metrics) -> str:
    """Return the shared metric space, or raise when metrics would be mixed.

    Normalized-space training percentages and physical Kelvin eval metrics are
    different quantities; a single axis may never carry both.
    """
    spaces = {metric_spec(m).space for m in metrics}
    if len(spaces) > 1:
        raise ValueError(
            f"refusing to mix metric spaces {sorted(spaces)} for metrics "
            f"{sorted(metrics)}"
        )
    return spaces.pop() if spaces else "none"


def pooled_rmse(sse_sum, n_sum):
    """``sqrt(sum sse / sum cells)``; NaN where the denominator is unusable."""
    sse = np.asarray(sse_sum, dtype=np.float64)
    n = np.asarray(n_sum, dtype=np.float64)
    out = np.full(np.broadcast(sse, n).shape, np.nan, dtype=np.float64)
    ok = np.isfinite(sse) & np.isfinite(n) & (n > 0) & (sse >= 0)
    np.sqrt(np.divide(sse, n, out=np.zeros_like(out), where=ok),
            out=out, where=ok)
    return out if out.ndim else float(out)


def pooled_rel_l2_pct(sse_sum, target_sum):
    """``100 * sqrt(sum sse / sum target)``; sigma-invariant relative error."""
    sse = np.asarray(sse_sum, dtype=np.float64)
    tgt = np.asarray(target_sum, dtype=np.float64)
    out = np.full(np.broadcast(sse, tgt).shape, np.nan, dtype=np.float64)
    ok = np.isfinite(sse) & np.isfinite(tgt) & (tgt > 0) & (sse >= 0)
    np.sqrt(np.divide(sse, tgt, out=np.zeros_like(out), where=ok),
            out=out, where=ok)
    out = np.where(ok, out * 100.0, np.nan)
    return out if out.ndim else float(out)


# ---------------------------------------------------------------------------
# strata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StratumSpec:
    """How one stratification column is derived from pair-level records."""

    name: str
    source: str
    kind: str                       # "categorical" | "quantile" | "fixed"
    order: tuple[str, ...] = ()
    edges: tuple[float, ...] = ()
    n_bins: int = 0
    label: str = ""


STRATUM_SPECS: dict[str, StratumSpec] = {
    "temporal_family": StratumSpec(
        "temporal_family", "temporal_family", "categorical",
        order=("sin", "exp", "pulse_train", "exp_train"), label="Temporal family"),
    "spatial_family": StratumSpec(
        "spatial_family", "spatial_family", "categorical",
        order=("uniform", "patch", "gaussian", "triangle"), label="Spatial family"),
    "regime": StratumSpec(
        "regime", "regime", "categorical",
        order=("left", "near", "right"), label="Patch regime"),
    "protocol": StratumSpec("protocol", "protocol", "categorical", label="Protocol"),
    "distribution_class": StratumSpec(
        "distribution_class", "distribution_class", "categorical",
        label="Distribution"),
    "lead_bin": StratumSpec(
        "lead_bin", "t_bar", "quantile", n_bins=4, label="Lead time"),
    "interface_x_bin": StratumSpec(
        "interface_x_bin", "x_I", "fixed",
        edges=(0.2, 0.35, 0.5, 0.65, 0.8), label="Interface position $x_I$"),
    "R_c_bin": StratumSpec(
        "R_c_bin", "R_c", "quantile", n_bins=4, label="Contact resistance $R_c$"),
    "void_severity": StratumSpec(
        "void_severity", "R_c_amp", "quantile", n_bins=3, label="Void severity"),
    "patch_x_bin": StratumSpec(
        "patch_x_bin", "x_h", "quantile", n_bins=4, label="Patch position $x_h$"),
    "amplitude_bin": StratumSpec(
        "amplitude_bin", "A", "quantile", n_bins=4, label="Source amplitude $A$"),
}


def stratum_spec(name: str) -> StratumSpec:
    try:
        return STRATUM_SPECS[name]
    except KeyError:
        raise KeyError(
            f"unknown stratum {name!r}; known: {sorted(STRATUM_SPECS)}"
        ) from None


def shared_bin_edges(frames, stratum: str) -> np.ndarray:
    """Bin edges computed on the union of every frame that will be compared.

    Groups compared in one panel must share edges. Computing them per group,
    as ``visual/paper_plots.py:_quantile_bin_strata`` does, makes the bins
    themselves a function of the group and the comparison meaningless.
    """
    spec = stratum_spec(stratum)
    if spec.kind == "categorical":
        raise ValueError(f"{stratum!r} is categorical; it has no bin edges")
    if spec.kind == "fixed":
        return np.asarray(spec.edges, dtype=np.float64)

    if isinstance(frames, pd.DataFrame):
        frames = [frames]
    values = np.concatenate([
        pd.to_numeric(f[spec.source], errors="coerce").to_numpy(dtype=np.float64)
        for f in frames if spec.source in f.columns
    ]) if frames else np.array([])
    values = values[np.isfinite(values)]
    if values.size == 0 or np.unique(values).size < 2:
        raise ValueError(
            f"cannot bin {stratum!r}: fewer than two distinct finite "
            f"{spec.source} values"
        )
    qs = np.linspace(0.0, 1.0, spec.n_bins + 1)
    return np.unique(np.quantile(values, qs))


def _bin_labels(edges: np.ndarray) -> list[str]:
    return [f"[{edges[i]:.3g}, {edges[i + 1]:.3g}]" for i in range(len(edges) - 1)]


def assign_strata(df: pd.DataFrame, strata, *,
                  edges: dict[str, np.ndarray] | None = None) -> pd.DataFrame:
    """Attach stratum columns at **pair** level, before any aggregation.

    Binning a per-simulation aggregate would assign a simulation to a bin using
    a number that already averaged over the axis being binned.
    """
    out = df.copy()
    edges = edges or {}
    for name in strata:
        spec = stratum_spec(name)
        if spec.source not in out.columns:
            raise KeyError(f"stratum {name!r} needs column {spec.source!r}")
        if spec.kind == "categorical":
            out[name] = out[spec.source].fillna("").astype(str)
            continue
        edge = np.asarray(edges.get(name, shared_bin_edges(out, name)),
                          dtype=np.float64)
        values = pd.to_numeric(out[spec.source], errors="coerce")
        idx = np.clip(np.searchsorted(edge, values, side="right") - 1,
                      0, len(edge) - 2)
        labels = _bin_labels(edge)
        assigned = pd.Series([labels[i] for i in idx], index=out.index)
        out[name] = assigned.where(values.notna(), "")
    return out


def stratum_order(df: pd.DataFrame, name: str) -> tuple[str, ...]:
    """Fixed display order for a stratum, so colours stay stable across figures."""
    spec = stratum_spec(name)
    present = [v for v in df[name].dropna().astype(str).unique() if v != ""]
    if spec.order:
        ordered = [v for v in spec.order if v in present]
        return tuple(ordered + sorted(v for v in present if v not in spec.order))
    if spec.kind in ("quantile", "fixed"):
        return tuple(sorted(present, key=lambda s: float(s.strip("[]").split(",")[0])))
    return tuple(sorted(present))


# ---------------------------------------------------------------------------
# level 0 -> 1
# ---------------------------------------------------------------------------


@dataclass
class SimFrame:
    """Per-simulation aggregates: the frame every later level is computed from."""

    df: pd.DataFrame
    metrics: tuple[str, ...]
    strata: tuple[str, ...]
    pooled: bool
    degradations: list[Degradation] = field(default_factory=list)

    @property
    def n_sims(self) -> int:
        return int(len(self.df))

    @property
    def n_pairs(self) -> int:
        return int(self.df["n_pairs"].sum()) if "n_pairs" in self.df.columns else 0

    @property
    def seeds(self) -> tuple[str, ...]:
        if "seed" not in self.df.columns:
            return ()
        return tuple(sorted(self.df["seed"].astype(str).unique()))


def per_sim(records, *, strata=(), metrics=DEFAULT_METRICS,
            edges: dict[str, np.ndarray] | None = None) -> SimFrame:
    """Reduce pair-level records to one row per simulation.

    ``records`` may be a :class:`~visual.pub.records.RecordFrame` or a plain
    frame. Strata are assigned at pair level and become part of the grouping
    key, so a simulation that spans several bins contributes one row per bin.
    """
    degradations: list[Degradation] = []
    df = getattr(records, "df", records)
    degradations.extend(list(getattr(records, "degradations", [])))

    strata = tuple(strata)
    metrics = tuple(metrics)
    if "sim_id" not in df.columns:
        raise KeyError("per_sim requires a sim_id column")

    if strata:
        df = assign_strata(df, strata, edges=edges)

    has_pooled = all(c in df.columns for c in _SUM_COLS)
    if not has_pooled and not any(
            d.code == "SCHEMA_V1_NO_POOLED_STATS" for d in degradations):
        degradations.append(Degradation(
            "SCHEMA_V1_NO_POOLED_STATS",
            "pooled sufficient statistics are absent; per-simulation metrics "
            "fall back to an unpooled mean of per-pair values, which is a "
            "different statistic. Re-run scripts/write_test_records.py.",
        ))

    group_keys = [k for k in ("seed", "benchmark") if k in df.columns]
    group_keys += list(strata)
    group_keys.append("sim_id")

    grouped = df.groupby(group_keys, dropna=False, observed=True)
    out = grouped.size().rename("n_pairs").reset_index()

    if has_pooled:
        sums = grouped[list(_SUM_COLS)].sum(min_count=1).reset_index()
        out = out.merge(sums, on=group_keys, how="left")

    mean_cols = [m for m in metrics
                 if (not has_pooled or METRICS[m].how == "mean") and m in df.columns]
    if mean_cols:
        means = grouped[mean_cols].mean().reset_index()
        out = out.merge(means, on=group_keys, how="left", suffixes=("", "_pairmean"))

    if has_pooled:
        for name in metrics:
            spec = metric_spec(name)
            if spec.how == "pooled":
                out[name] = pooled_rmse(out[spec.numerator], out[spec.denominator])
            elif spec.how == "ratio":
                out[name] = pooled_rel_l2_pct(out[spec.numerator], out[spec.denominator])

    missing = [m for m in metrics if m not in out.columns]
    if missing:
        raise KeyError(
            f"per_sim cannot produce {missing}: the records frame has neither the "
            "pooled sufficient statistics nor per-pair columns for them"
        )

    return SimFrame(df=out, metrics=metrics, strata=strata, pooled=has_pooled,
                    degradations=degradations)


def as_sim_frame(df: pd.DataFrame, *, metrics, strata=()) -> SimFrame:
    """Wrap a table that is **already** one row per simulation.

    ``ood_per_sim.csv`` is written by ``scripts/run_ood_suite.py``, which applies
    the same pair-to-simulation pooling this module performs, so re-running
    :func:`per_sim` over it would group singletons and silently relabel the
    replication unit. This adapter takes the reduction as given and asserts the
    invariant that makes it safe: the grouping key plus ``sim_id`` identifies a
    row uniquely. A duplicate means the caller handed over pair-level rows.
    """
    metrics = tuple(metrics)
    strata = tuple(strata)
    if "sim_id" not in df.columns:
        raise KeyError("as_sim_frame requires a sim_id column")
    missing = [m for m in metrics if m not in df.columns]
    if missing:
        raise KeyError(f"as_sim_frame is missing metric columns {missing}")

    keys = [k for k in ("seed", "benchmark") if k in df.columns]
    keys += [s for s in strata if s in df.columns]
    keys.append("sim_id")
    if df.duplicated(subset=keys).any():
        raise ValueError(
            f"as_sim_frame received rows that are not unique per {keys}; the "
            "table is below the simulation level and needs per_sim instead"
        )
    return SimFrame(df=df, metrics=metrics, strata=strata, pooled=True)


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------


def bootstrap_ci(values, statistic=np.median, *, n_boot: int = DEFAULT_N_BOOT,
                 alpha: float = 0.05, method: str = "bca",
                 rng_seed: int = DEFAULT_RNG_SEED) -> tuple[float, float, str]:
    """Resample the given values (always simulations) for a CI on ``statistic``.

    Returns ``(lo, hi, method_used)``. ``method_used`` is ``"none"`` when the
    sample is too small for an interval to mean anything, so callers can render
    the absence of an interval rather than a misleadingly narrow one.
    """
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = x.size
    if n < MIN_SIMS_FOR_CI:
        return float("nan"), float("nan"), "none"

    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = np.apply_along_axis(statistic, 1, x[idx]).astype(np.float64)
    boot = boot[np.isfinite(boot)]
    if boot.size == 0:
        return float("nan"), float("nan"), "none"

    if method == "percentile":
        lo, hi = np.quantile(boot, [alpha / 2.0, 1.0 - alpha / 2.0])
        return float(lo), float(hi), "percentile"

    theta = float(statistic(x))
    prop = float(np.mean(boot < theta))
    if prop <= 0.0 or prop >= 1.0:
        lo, hi = np.quantile(boot, [alpha / 2.0, 1.0 - alpha / 2.0])
        return float(lo), float(hi), "percentile"

    from scipy.stats import norm

    z0 = norm.ppf(prop)
    jack = np.array([statistic(np.delete(x, i)) for i in range(n)], dtype=np.float64)
    jack_mean = jack.mean()
    num = np.sum((jack_mean - jack) ** 3)
    den = 6.0 * (np.sum((jack_mean - jack) ** 2) ** 1.5)
    accel = 0.0 if den == 0 else num / den

    z_lo, z_hi = norm.ppf(alpha / 2.0), norm.ppf(1.0 - alpha / 2.0)
    a_lo = norm.cdf(z0 + (z0 + z_lo) / (1.0 - accel * (z0 + z_lo)))
    a_hi = norm.cdf(z0 + (z0 + z_hi) / (1.0 - accel * (z0 + z_hi)))
    if not (np.isfinite(a_lo) and np.isfinite(a_hi)):
        lo, hi = np.quantile(boot, [alpha / 2.0, 1.0 - alpha / 2.0])
        return float(lo), float(hi), "percentile"
    lo, hi = np.quantile(boot, [float(a_lo), float(a_hi)])
    return float(lo), float(hi), "bca"


# ---------------------------------------------------------------------------
# level 1 -> 2
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Summary:
    """The distribution of one metric across simulations within one stratum."""

    key: tuple
    metric: str
    space: str
    n_sims: int
    n_pairs: int
    median: float
    q25: float
    q75: float
    p90: float
    p99: float
    mean: float
    max: float
    ci_lo: float
    ci_hi: float
    ci_method: str
    degradations: tuple[Degradation, ...] = ()

    @property
    def iqr(self) -> float:
        return self.q75 - self.q25


def summarize(sims: SimFrame, metric: str, *, strata=(),
              ci: str = "bca", n_boot: int = DEFAULT_N_BOOT,
              rng_seed: int = DEFAULT_RNG_SEED,
              statistic=np.median) -> dict[tuple, Summary]:
    """Median / IQR / p90 / p99 plus a bootstrap CI, across **simulations**.

    Keys are the values of ``("seed", "benchmark") + strata`` that are present
    in the frame. The bootstrap resampling unit is always the simulation.
    """
    spec = metric_spec(metric)
    df = sims.df
    if metric not in df.columns:
        raise KeyError(f"{metric!r} not present in the SimFrame; "
                       f"available: {sorted(sims.metrics)}")

    strata = tuple(strata) or sims.strata
    group_keys = [k for k in ("seed", "benchmark") if k in df.columns]
    group_keys += [s for s in strata if s in df.columns]

    out: dict[tuple, Summary] = {}
    groups = df.groupby(group_keys, dropna=False, observed=True) if group_keys \
        else [((), df)]
    for key, part in (groups if group_keys else groups):
        key_tuple = key if isinstance(key, tuple) else (key,)
        values = pd.to_numeric(part[metric], errors="coerce").to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]
        n_sims = int(values.size)
        n_pairs = int(part["n_pairs"].sum()) if "n_pairs" in part.columns else 0

        degradations = list(sims.degradations)
        if n_sims == 0:
            out[key_tuple] = Summary(
                key_tuple, metric, spec.space, 0, n_pairs,
                *([float("nan")] * 7), float("nan"), float("nan"), "none",
                tuple(degradations + [Degradation(
                    "EMPTY_STRATUM", f"no finite {metric} for key {key_tuple}")]),
            )
            continue

        if n_sims < MIN_SIMS_FOR_CI:
            degradations.append(Degradation(
                "TOO_FEW_SIMS",
                f"{key_tuple}: {n_sims} simulations (< {MIN_SIMS_FOR_CI}); no "
                "confidence interval is reported.",
            ))
        p99 = float(np.percentile(values, 99))
        if n_sims < MIN_SIMS_FOR_P99:
            p99 = float("nan")
            degradations.append(Degradation(
                "P99_UNDERDETERMINED",
                f"{key_tuple}: {n_sims} simulations (< {MIN_SIMS_FOR_P99}); the "
                "99th percentile is not determined and is reported as NaN.",
            ))

        lo, hi, method = bootstrap_ci(values, statistic, n_boot=n_boot,
                                      method=ci, rng_seed=rng_seed)
        out[key_tuple] = Summary(
            key=key_tuple, metric=metric, space=spec.space,
            n_sims=n_sims, n_pairs=n_pairs,
            median=float(np.median(values)),
            q25=float(np.percentile(values, 25)),
            q75=float(np.percentile(values, 75)),
            p90=float(np.percentile(values, 90)),
            p99=p99,
            mean=float(np.mean(values)),
            max=float(np.max(values)),
            ci_lo=lo, ci_hi=hi, ci_method=method,
            degradations=tuple(degradations),
        )
    return out


# ---------------------------------------------------------------------------
# distributions
# ---------------------------------------------------------------------------


def ecdf(values) -> tuple[np.ndarray, np.ndarray]:
    """Empirical CDF of per-simulation values: sorted x and ``(1..n)/n``."""
    x = np.asarray(values, dtype=np.float64)
    x = np.sort(x[np.isfinite(x)])
    if x.size == 0:
        return x, x
    return x, np.arange(1, x.size + 1, dtype=np.float64) / x.size


def ecdf_band(values_by_seed: dict, *, quantiles=(0.05, 0.95), n_grid: int = 256
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Across-seed band of ECDFs evaluated on one shared grid.

    Returns ``(grid, lo, hi)``. The band is over model seeds, so it answers
    "how much does this curve move when we retrain", not "how noisy is one run".
    """
    arrays = [np.asarray(v, dtype=np.float64) for v in values_by_seed.values()]
    arrays = [a[np.isfinite(a)] for a in arrays]
    arrays = [a for a in arrays if a.size]
    if not arrays:
        empty = np.array([])
        return empty, empty, empty
    lo_x = min(a.min() for a in arrays)
    hi_x = max(a.max() for a in arrays)
    grid = np.linspace(lo_x, hi_x, n_grid)
    curves = np.stack([np.searchsorted(np.sort(a), grid, side="right") / a.size
                       for a in arrays])
    lo = np.quantile(curves, quantiles[0], axis=0)
    hi = np.quantile(curves, quantiles[1], axis=0)
    return grid, lo, hi


@dataclass(frozen=True)
class RateCI:
    """A proportion with a Wilson score interval."""

    rate: float
    lo: float
    hi: float
    k: int
    n: int


def wilson_ci(k: float, n: float, z: float = Z_95) -> tuple[float, float]:
    """95% Wilson score interval; ``(nan, nan)`` when ``n == 0``."""
    n = float(n)
    if n <= 0:
        return float("nan"), float("nan")
    k = float(min(max(k, 0.0), n))
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return float(center - half), float(center + half)


def exceedance_rate(values, threshold: float) -> RateCI:
    """Fraction of **simulations** whose metric exceeds ``threshold``."""
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = int(x.size)
    k = int(np.count_nonzero(x > threshold))
    lo, hi = wilson_ci(k, n)
    return RateCI(rate=(k / n if n else float("nan")), lo=lo, hi=hi, k=k, n=n)


def proportion(flags) -> RateCI:
    """Fraction of cases whose flag is true, with a Wilson interval.

    Used for interval coverage, where the quantity of interest is already a
    yes/no per case rather than a metric compared against a threshold. Missing
    flags are dropped from the denominator rather than counted as failures.
    """
    x = np.asarray(flags)
    if x.dtype != bool:
        x = np.asarray(pd.Series(x).map(
            {True: True, False: False, "True": True, "False": False}))
        keep = np.array([v is True or v is False for v in x])
        x = x[keep].astype(bool)
    n = int(x.size)
    k = int(np.count_nonzero(x))
    lo, hi = wilson_ci(k, n)
    return RateCI(rate=(k / n if n else float("nan")), lo=lo, hi=hi, k=k, n=n)


@dataclass(frozen=True)
class RankCorrelation:
    """Spearman rho with a bootstrap interval over the replication unit."""

    rho: float
    ci_lo: float
    ci_hi: float
    ci_method: str
    n: int
    degradations: tuple[Degradation, ...] = ()


def _average_ranks(a: np.ndarray) -> np.ndarray:
    """Average-rank transform; ties share the mean of the ranks they span."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.size, dtype=np.float64)
    ranks[order] = np.arange(1, a.size + 1, dtype=np.float64)
    sa = a[order]
    i = 0
    while i < a.size:
        j = i
        while j + 1 < a.size and sa[j + 1] == sa[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(x, y, *, n_boot: int = DEFAULT_N_BOOT,
             rng_seed: int = DEFAULT_RNG_SEED) -> RankCorrelation:
    """Rank correlation of paired values, with a percentile bootstrap CI.

    Rank rather than Pearson because the inverse quantities this describes are
    heavy-tailed and the claim being tested is monotone association, not a
    linear one. The bootstrap resamples the **pair**, which for the inverse
    figures is the inversion case, i.e. the replication unit.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = int(x.size)

    degradations: list[Degradation] = []
    if n < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        degradations.append(Degradation(
            "RANK_CORRELATION_UNDERDETERMINED",
            f"Spearman needs >= 3 pairs with variation on both axes; got n={n}.",
        ))
        return RankCorrelation(float("nan"), float("nan"), float("nan"),
                               "none", n, tuple(degradations))

    rho = float(np.corrcoef(_average_ranks(x), _average_ranks(y))[0, 1])
    if n < MIN_SIMS_FOR_CI:
        degradations.append(Degradation(
            "TOO_FEW_SIMS",
            f"{n} cases is below the {MIN_SIMS_FOR_CI}-case floor for an "
            "interval; the correlation is reported as a point estimate.",
        ))
        return RankCorrelation(rho, float("nan"), float("nan"), "none", n,
                               tuple(degradations))

    rng = np.random.default_rng(rng_seed)
    draws = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xb, yb = x[idx], y[idx]
        if np.unique(xb).size < 2 or np.unique(yb).size < 2:
            draws[b] = np.nan
            continue
        draws[b] = np.corrcoef(_average_ranks(xb), _average_ranks(yb))[0, 1]
    draws = draws[np.isfinite(draws)]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return RankCorrelation(rho, float(lo), float(hi), "percentile", n,
                           tuple(degradations))


# ---------------------------------------------------------------------------
# level 2 -> 3
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeedRoll:
    """One stratum's statistic across model seeds."""

    key: tuple
    metric: str
    per_seed: dict[str, float]
    mean: float
    std: float
    min: float
    max: float
    degradations: tuple[Degradation, ...] = ()

    @property
    def n_seeds(self) -> int:
        return len(self.per_seed)


def across_seeds(sims: SimFrame, metric: str, *, strata=(),
                 statistic=np.median) -> dict[tuple, SeedRoll]:
    """Collapse each seed to one number, then describe the spread across seeds.

    With fewer than three seeds no spread is reported: two seeds cannot support
    a standard error, and the figure shows the individual seed values instead.
    """
    df = sims.df
    if "seed" not in df.columns:
        raise KeyError("across_seeds requires a seed column")
    strata = tuple(strata) or sims.strata
    key_cols = [k for k in ("benchmark",) if k in df.columns]
    key_cols += [s for s in strata if s in df.columns]

    out: dict[tuple, SeedRoll] = {}
    groups = df.groupby(key_cols, dropna=False, observed=True) if key_cols \
        else [((), df)]
    for key, part in groups:
        key_tuple = key if isinstance(key, tuple) else (key,)
        per_seed: dict[str, float] = {}
        for seed, seed_part in part.groupby("seed", observed=True):
            vals = pd.to_numeric(seed_part[metric], errors="coerce").to_numpy(
                dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                per_seed[str(seed)] = float(statistic(vals))
        values = np.asarray(list(per_seed.values()), dtype=np.float64)
        degradations = []
        std = float(np.std(values, ddof=1)) if values.size >= MIN_SEEDS_FOR_SPREAD \
            else float("nan")
        if values.size < MIN_SEEDS_FOR_SPREAD:
            degradations.append(Degradation(
                "SEED_SPREAD_UNDERDETERMINED",
                f"{key_tuple}: {values.size} seed(s) (< {MIN_SEEDS_FOR_SPREAD}); "
                "no seed spread is reported, individual seeds are shown instead.",
            ))
        out[key_tuple] = SeedRoll(
            key=key_tuple, metric=metric, per_seed=per_seed,
            mean=float(np.mean(values)) if values.size else float("nan"),
            std=std,
            min=float(np.min(values)) if values.size else float("nan"),
            max=float(np.max(values)) if values.size else float("nan"),
            degradations=tuple(degradations),
        )
    return out


@dataclass(frozen=True)
class PairedDelta:
    """Within-simulation difference between two conditions under common seeds."""

    metric: str
    values: np.ndarray
    n_pairs: int
    median: float
    ci_lo: float
    ci_hi: float
    ci_method: str
    degradations: tuple[Degradation, ...] = ()


def paired_seed_delta(sims_a: SimFrame, sims_b: SimFrame, metric: str, *,
                      match_on=("seed", "sim_id"),
                      n_boot: int = DEFAULT_N_BOOT,
                      rng_seed: int = DEFAULT_RNG_SEED) -> PairedDelta:
    """``b - a`` matched on the same simulations and seeds (common random numbers).

    Pairing removes simulation-to-simulation variance, which is the dominant
    term; an unpaired comparison of the same two conditions is far noisier.
    """
    match_on = tuple(match_on)
    left = sims_a.df[[*match_on, metric]].rename(columns={metric: "_a"})
    right = sims_b.df[[*match_on, metric]].rename(columns={metric: "_b"})
    merged = left.merge(right, on=list(match_on), how="inner")
    values = (pd.to_numeric(merged["_b"], errors="coerce")
              - pd.to_numeric(merged["_a"], errors="coerce")).to_numpy(dtype=np.float64)
    values = values[np.isfinite(values)]

    degradations = []
    if values.size == 0:
        degradations.append(Degradation(
            "NO_PAIRED_SIMS",
            f"no simulations match on {list(match_on)} between the two frames; "
            "a paired comparison is not possible.",
        ))
        return PairedDelta(metric, values, 0, float("nan"), float("nan"),
                           float("nan"), "none", tuple(degradations))

    lo, hi, method = bootstrap_ci(values, np.median, n_boot=n_boot,
                                  rng_seed=rng_seed)
    return PairedDelta(
        metric=metric, values=values, n_pairs=int(values.size),
        median=float(np.median(values)), ci_lo=lo, ci_hi=hi, ci_method=method,
        degradations=tuple(degradations),
    )


__all__ = [
    "DEFAULT_METRICS",
    "DEFAULT_N_BOOT",
    "DEFAULT_RNG_SEED",
    "MEAN_METRICS",
    "METRICS",
    "MIN_SEEDS_FOR_SPREAD",
    "MIN_SIMS_FOR_CI",
    "MIN_SIMS_FOR_P99",
    "MetricSpec",
    "POOLED_METRICS",
    "PairedDelta",
    "RATIO_METRICS",
    "RankCorrelation",
    "RateCI",
    "STRATUM_SPECS",
    "SeedRoll",
    "SimFrame",
    "StratumSpec",
    "Summary",
    "across_seeds",
    "as_sim_frame",
    "assert_same_space",
    "assign_strata",
    "bootstrap_ci",
    "ecdf",
    "ecdf_band",
    "exceedance_rate",
    "metric_spec",
    "paired_seed_delta",
    "per_sim",
    "pooled_rel_l2_pct",
    "pooled_rmse",
    "proportion",
    "shared_bin_edges",
    "spearman",
    "stratum_order",
    "stratum_spec",
    "summarize",
    "wilson_ci",
]
