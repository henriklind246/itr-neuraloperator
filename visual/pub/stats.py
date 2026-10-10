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

from visual.pub.manifest import Degradation, ProvenanceError

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
    "itr_amplitude": StratumSpec(
        "itr_amplitude", "R_c_A", "quantile", n_bins=3, label="ITR amplitude"),
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
    as the legacy per-group binning did, makes the bins
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


@dataclass(frozen=True)
class ContactJumpCurve:
    """Simulation-level summaries of a physical contact-jump lead curve."""

    lead_times: np.ndarray
    truth_median: np.ndarray
    pred_median: np.ndarray
    rmse_median: np.ndarray
    rmse_q25: np.ndarray
    rmse_q75: np.ndarray
    relative_median_pct: np.ndarray
    relative_q25_pct: np.ndarray
    relative_q75_pct: np.ndarray
    n_sims: int


@dataclass(frozen=True)
class ParityFit:
    """Agreement between a predicted and a true quantity on a parity plot."""

    slope: float
    rmse: float
    truth_rms: float
    relative_pct: float
    span: tuple[float, float]
    n_points: int
    n_sims: int


@dataclass(frozen=True)
class RolloutDeltaCurve:
    """Paired autoregressive-minus-direct error at exact lead times."""

    substeps: int
    lead_times: np.ndarray
    median_pp: np.ndarray
    q25_pp: np.ndarray
    q75_pp: np.ndarray
    direct_median_pct: np.ndarray
    n_sims: int
    n_snapshot_pairs: int


@dataclass(frozen=True)
class SensorSweepCurve:
    """Paired case values plus summaries at discrete sensor counts."""

    metric: str
    unit: str
    sensor_counts: np.ndarray
    case_keys: tuple[tuple[int, int, int], ...]
    case_values: np.ndarray
    median: np.ndarray
    q25: np.ndarray
    q75: np.ndarray
    censored: np.ndarray | None = None

    @property
    def n_cases(self) -> int:
        return len(self.case_keys)


@dataclass(frozen=True)
class InverseSensorSweepSummary:
    """Paired F18 quantities for one benchmark and recovered parameter."""

    benchmark: str
    estimand: str
    estimand_unit: str
    recovery_error: SensorSweepCurve
    relative_error: SensorSweepCurve
    fv_residual: SensorSweepCurve
    noise_floor: SensorSweepCurve
    profile_width: SensorSweepCurve
    fv_over_noise_median: np.ndarray
    bound_limited_cases: np.ndarray


_INVERSE_SENSOR_METRICS = {
    benchmark: tuple({
        "estimand": name, "unit": "m² K/W",
        "truth": f"{name}_true",
        "estimate": f"{name}_map" if name == "R_c" else f"{name}_hat",
        "absolute_error": f"{name}_abs_error" if name == "R_c" else f"{name}_abserr",
        "relative_error": f"{name}_rel_error_pct",
        "profile_low": f"profile_{name}_ci_low",
        "profile_high": f"profile_{name}_ci_high",
        "limited": f"profile_{name}_bound_limited",
    } for name in names)
    for benchmark, names in {
        "forcing": ("R_c",), "forcing_itr_sin": ("R_base", "A"),
        "source_itr_sin": ("R_base", "A"),
    }.items()
}

# Gram matrix G[i,j] = \int_0^1 b_i(y) b_j(y) dy of the basis each benchmark's
# parameters multiply in R_c(y), so that a parameter-error vector d has profile
# error ||dR_c||_L2 = sqrt(d^T G d) in closed form -- no quadrature, no grid.
#
#   forcing:          R_c(y) = R_c                  basis (1,)
#   *_itr_sin:        R_c(y) = R_base + A sin(pi y) basis (1, sin(pi y))
#
# \int sin(pi y) dy = 2/pi and \int sin^2(pi y) dy = 1/2 on [0, 1]. The
# off-diagonal term is what makes this worth doing: sin(pi y) >= 0 everywhere on
# the interface, so a positive R_base error and a positive A error push the
# profile the same way at every y and compound, while opposite signs partly
# cancel into a genuinely better reconstruction. Ranking on either parameter
# alone cannot see that, and it is precisely the profile that F29 draws.
_INVERSE_PROFILE_GRAM = {
    "forcing": np.asarray([[1.0]]),
    "forcing_itr_sin": np.asarray([[1.0, 2.0 / np.pi], [2.0 / np.pi, 0.5]]),
    "source_itr_sin": np.asarray([[1.0, 2.0 / np.pi], [2.0 / np.pi, 0.5]]),
}


def _sensor_sweep_curve(
    values,
    *,
    metric: str,
    unit: str,
    sensor_counts,
    case_keys,
    censored=None,
    allow_missing=False,
) -> SensorSweepCurve:
    values = np.asarray(values, dtype=np.float64)
    counts = np.asarray(sensor_counts, dtype=np.int64)
    if values.shape != (len(case_keys), counts.size):
        raise ValueError(
            f"{metric} has shape {values.shape}; expected "
            f"({len(case_keys)}, {counts.size})"
        )
    if np.any(np.isinf(values)) or (not allow_missing and np.any(np.isnan(values))) or np.any(values < 0.0):
        raise ValueError(f"{metric} must contain finite non-negative values")
    censored_values = None
    if censored is not None:
        censored_values = np.asarray(censored, dtype=bool)
        if censored_values.shape != values.shape:
            raise ValueError(f"{metric} censoring mask does not match its values")
    quantiles = []
    for j in range(counts.size):
        keep = np.isfinite(values[:, j])
        if censored_values is not None:
            keep &= ~censored_values[:, j]
        quantiles.append(np.percentile(values[keep, j], [25, 50, 75]) if keep.any() else [np.nan] * 3)
    q25, median, q75 = np.asarray(quantiles).T
    return SensorSweepCurve(
        metric=metric,
        unit=unit,
        sensor_counts=counts,
        case_keys=tuple(case_keys),
        case_values=values,
        median=median, q25=q25, q75=q75,
        censored=censored_values,
    )


def inverse_sensor_sweep_summaries(
    table: pd.DataFrame,
    *,
    sensor_counts: tuple[int, ...] = (8, 16, 32),
) -> dict[tuple[str, str], InverseSensorSweepSummary]:
    """Reduce F18 at the paired inversion-case level.

    Lines connect identical ``(sim_id, noise_seed, init_seed)`` cases across the
    three discrete sensor counts. Medians and IQRs are descriptive across those
    cases; with eight cases no inferential confidence interval is estimated.
    """
    from scripts.inverse_adapters import InverseAdapter, relative_error_percent

    pairing = ["sim_id", "noise_seed", "init_seed"]
    output: dict[tuple[str, str], InverseSensorSweepSummary] = {}
    for benchmark in sorted(table["benchmark"].dropna().astype(str).unique()):
        try:
            parameter_columns = _INVERSE_SENSOR_METRICS[benchmark]
        except KeyError:
            raise ValueError(
                f"unsupported inverse sensor sweep benchmark {benchmark!r}"
            ) from None
        adapter = InverseAdapter.from_config({"benchmark": {"name": benchmark}})
        for parameter_index, columns in enumerate(parameter_columns):
            part = table.loc[table["benchmark"].astype(str) == benchmark].copy()
            counts = tuple(sorted(part["n_sensors"].astype(int).unique()))
            if counts != tuple(sensor_counts):
                raise ValueError(
                    f"{benchmark} sensor counts {counts} do not match {sensor_counts}"
                )
            if part.duplicated(["n_sensors", *pairing]).any():
                raise ValueError(f"{benchmark} has duplicate paired sensor-sweep rows")

            case_keys = tuple(sorted({
                tuple(int(value) for value in row)
                for row in part[pairing].itertuples(index=False, name=None)
            }))
            matrices: dict[str, np.ndarray] = {}
            numeric_columns = (
                columns["truth"], columns["estimate"], columns["absolute_error"], columns["relative_error"],
                "fv_resid_rms_K", "fv_resid_over_noise", "noise_std_K",
                columns["profile_low"], columns["profile_high"],
            )
            for column in numeric_columns:
                columns_by_count = []
                for count in counts:
                    arm = part.loc[part["n_sensors"].astype(int) == count].copy()
                    arm["_pair"] = [
                        tuple(int(value) for value in row)
                        for row in arm[pairing].itertuples(index=False, name=None)
                    ]
                    values = arm.set_index("_pair")[column].reindex(case_keys)
                    if values.isna().any() and column != columns["relative_error"]:
                        raise ValueError(
                            f"{benchmark} {column} is not paired across sensor counts"
                        )
                    columns_by_count.append(values.to_numpy(dtype=np.float64))
                matrices[column] = np.column_stack(columns_by_count)

            limited_columns = []
            for count in counts:
                arm = part.loc[part["n_sensors"].astype(int) == count].copy()
                arm["_pair"] = [
                    tuple(int(value) for value in row)
                    for row in arm[pairing].itertuples(index=False, name=None)
                ]
                flags = arm.set_index("_pair")[columns["limited"]].reindex(case_keys)
                if flags.isna().any():
                    raise ValueError(
                        f"{benchmark} profile censoring is not paired across counts"
                    )
                limited_columns.append(flags.astype(bool).to_numpy())
            censored = np.column_stack(limited_columns)

            truth = matrices[columns["truth"]]
            estimate = matrices[columns["estimate"]]
            absolute_error = matrices[columns["absolute_error"]]
            if not np.allclose(truth, truth[:, :1], rtol=1e-9, atol=1e-12):
                raise ValueError(f"{benchmark} changes the true estimand across sensor counts")
            if not np.allclose(
                absolute_error, np.abs(estimate - truth), rtol=1e-6, atol=1e-12
            ):
                raise ValueError(f"{benchmark} absolute recovery error is inconsistent")

            expected_relative = np.asarray([
                relative_error_percent(hat, true, adapter.param_scales[parameter_index])
                for hat, true in zip(estimate.ravel(), truth.ravel())
            ]).reshape(truth.shape)
            if not np.allclose(matrices[columns["relative_error"]], expected_relative,
                               rtol=1e-6, atol=1e-12, equal_nan=True):
                raise ValueError(f"{benchmark} relative recovery error is inconsistent")

            fv = matrices["fv_resid_rms_K"]
            noise = matrices["noise_std_K"]
            over_noise = matrices["fv_resid_over_noise"]
            if np.any(noise <= 0.0):
                raise ValueError(f"{benchmark} noise_std_K must be positive")
            if not np.allclose(fv / noise, over_noise, rtol=1e-5, atol=1e-8):
                raise ValueError(f"{benchmark} FV/noise ratio is inconsistent")

            profile_width = (
                matrices[columns["profile_high"]]
                - matrices[columns["profile_low"]]
            )
            recovery_curve = _sensor_sweep_curve(
                absolute_error, metric="absolute recovery error",
                unit=columns["unit"], sensor_counts=counts, case_keys=case_keys,
            )
            relative_curve = _sensor_sweep_curve(
                matrices[columns["relative_error"]], metric="relative recovery error",
                unit="%", sensor_counts=counts, case_keys=case_keys, allow_missing=True,
            )
            fv_curve = _sensor_sweep_curve(
                fv, metric="FV residual at recovered parameter", unit="K",
                sensor_counts=counts, case_keys=case_keys,
            )
            noise_curve = _sensor_sweep_curve(
                noise, metric="measurement noise standard deviation", unit="K",
                sensor_counts=counts, case_keys=case_keys,
            )
            width_curve = _sensor_sweep_curve(
                profile_width, metric="profile interval width",
                unit=columns["unit"], sensor_counts=counts, case_keys=case_keys,
                censored=censored,
            )
            output[(benchmark, columns["estimand"])] = InverseSensorSweepSummary(
                benchmark=benchmark,
                estimand=columns["estimand"],
                estimand_unit=columns["unit"],
                recovery_error=recovery_curve, relative_error=relative_curve,
                fv_residual=fv_curve,
                noise_floor=noise_curve,
                profile_width=width_curve,
                fv_over_noise_median=np.median(over_noise, axis=0),
                bound_limited_cases=np.count_nonzero(censored, axis=0),
            )
    return output


def inverse_profile_l2_error(table: pd.DataFrame, *, benchmark: str) -> np.ndarray:
    """Per-row L2 error of the recovered ``R_c(y)`` profile, in m² K/W.

    ``sqrt(d^T G d)`` where ``d`` is the signed parameter-error vector and ``G``
    is :data:`_INVERSE_PROFILE_GRAM`. This is the root-mean-square discrepancy
    between the true and recovered resistance profiles over ``y`` in ``[0, 1]``,
    which is the quantity F29 plots and the one a reader judges a reconstruction
    by. For the scalar-``R_c`` benchmark ``G`` is ``[[1]]`` and it reduces
    exactly to the absolute recovery error, so one statistic covers both.
    """
    try:
        columns = _INVERSE_SENSOR_METRICS[benchmark]
        gram = _INVERSE_PROFILE_GRAM[benchmark]
    except KeyError:
        raise ValueError(
            f"unsupported inverse sensor sweep benchmark {benchmark!r}"
        ) from None

    needed = [c[key] for c in columns for key in ("truth", "estimate")]
    missing = [c for c in needed if c not in table.columns]
    if missing:
        raise ValueError(f"inverse sensor sweep is missing columns {missing}")

    delta = np.column_stack([
        table[c["estimate"]].to_numpy(dtype=np.float64)
        - table[c["truth"]].to_numpy(dtype=np.float64)
        for c in columns
    ])
    return np.sqrt(np.maximum(np.einsum("ni,ij,nj->n", delta, gram, delta), 0.0))


def representative_inverse_case(
    table: pd.DataFrame,
    *,
    benchmark: str,
    reference_sensors: int = 32,
) -> int:
    """Pre-registered choice of the single inversion case a figure shows.

    Cases are ranked by :func:`inverse_profile_l2_error` at one sensor arm and
    the lower median is taken. Three choices, each of which is a claim:

    *Median, not best.* A best-case figure is selected in its own favour and
    tells a reader nothing about what to expect. The median case is the one the
    aggregate table's central tendency actually refers to.

    *Profile L2, not one parameter's error.* The figure shows a curve, so it is
    ranked by how far that curve is from the truth. Ranking on ``A`` alone would
    call a case typical on the strength of one coefficient while the profile it
    draws was off by a constant offset the ranking never looked at.

    *One arm, declared by the caller.* Ranking on the best-resolved arm makes
    the selection a statement about the method at its intended operating point.
    Ranking separately per arm would let each panel of F29 show a different
    simulation, which would destroy the controlled comparison the figure is.

    Fixing the rule here rather than inside each figure is what keeps every
    figure that needs "the representative case" showing the same inversion.
    """
    if benchmark not in _INVERSE_SENSOR_METRICS:
        raise ValueError(f"unsupported inverse sensor sweep benchmark {benchmark!r}")

    arm = table.loc[
        (table["benchmark"].astype(str) == benchmark)
        & (table["n_sensors"].astype(int) == int(reference_sensors))
    ]
    if arm.empty:
        raise ValueError(
            f"{benchmark} has no {reference_sensors}-sensor arm to rank cases by"
        )
    ranked = pd.DataFrame({
        "sim_id": arm["sim_id"].astype(int).to_numpy(),
        "profile_l2": inverse_profile_l2_error(arm, benchmark=benchmark),
    })
    if ranked["sim_id"].duplicated().any():
        raise ValueError(
            f"{benchmark} repeats a sim_id in the {reference_sensors}-sensor arm"
        )
    if not np.all(np.isfinite(ranked["profile_l2"].to_numpy())):
        raise ValueError(
            f"{benchmark} profile L2 error is not finite at "
            f"{reference_sensors} sensors"
        )
    # Lower median on an even count: the smaller of the two central ranks, so
    # the rule resolves to one existing case rather than an average of two cases
    # that cannot be averaged. sim_id breaks ties.
    ranked = ranked.sort_values(["profile_l2", "sim_id"], kind="mergesort")
    return int(ranked["sim_id"].to_numpy()[(len(ranked) - 1) // 2])


@dataclass(frozen=True)
class LaplaceJointRegion:
    """Gauss-Newton approximation to a 2-parameter joint likelihood region.

    Field-compatible with :class:`visual.pub.records.JointNLLGrid` so a contour
    panel consumes either without branching, but ``approximate`` is ``True``:
    this surface is quadratic by construction and so cannot show the curvature
    the measured grid exists to reveal.
    """

    param_names: tuple[str, str]
    axes: tuple[np.ndarray, np.ndarray]
    delta_ell: np.ndarray
    threshold: float
    approximate: bool = True
    cond_number: float = float("nan")


def laplace_joint_region(
    jacobian,
    sigma_eff2: float,
    theta_hat,
    *,
    param_names: tuple[str, str],
    level: float = 0.95,
    bounds=None,
    peak_max: float | None = None,
    n_grid: int = 81,
    pad: float = 1.35,
) -> LaplaceJointRegion:
    """Quadratic fallback for the joint region when no measured grid was stored.

    ``H = Jᵀ J / sigma_eff2`` and ``dell(theta) ~= 0.5 (theta-hat)ᵀ H (theta-hat)``,
    contoured at ``chi2.ppf(level, 2) / 2``. The grid spans the bounding box of
    the ``dell = threshold`` ellipse scaled by ``pad``, so it always contains the
    region of interest without the caller having to guess a span.

    ``peak_max`` applies the dependent ceiling ``theta_1 <= peak_max - theta_0``
    as ``nan``; the caller supplies it because the ceiling is a property of the
    benchmark's parameterization, not of the approximation.
    """
    from scipy.stats import chi2

    jacobian = np.asarray(jacobian, dtype=np.float64)
    theta_hat = np.asarray(theta_hat, dtype=np.float64).ravel()
    if theta_hat.size != 2 or len(param_names) != 2:
        raise ValueError("the joint region is only defined for two parameters")
    if jacobian.ndim != 2 or jacobian.shape[1] != 2:
        raise ValueError(f"jacobian has shape {jacobian.shape}; expected (m, 2)")
    if not np.isfinite(sigma_eff2) or sigma_eff2 <= 0.0:
        raise ValueError("sigma_eff2 must be positive and finite")

    hessian = jacobian.T @ jacobian / float(sigma_eff2)
    singular = np.linalg.svd(jacobian, compute_uv=False)
    # numpy's matrix_rank tolerance: below it the inverse is numerical noise, so
    # the "region" would be an artefact of round-off rather than a bound.
    tolerance = singular.max() * max(jacobian.shape) * np.finfo(np.float64).eps
    if singular.min() <= tolerance:
        raise ValueError(
            "the Jacobian is rank deficient: the quadratic region is unbounded "
            "and would understate the uncertainty rather than approximate it"
        )
    threshold = float(chi2.ppf(level, 2) / 2.0)

    # Half-widths of the ellipse bounding box: the largest |d_i| on
    # d^T H d = 2c is sqrt(2c (H^-1)_ii), the marginal extent rather than the
    # conditional one, so the box encloses the whole contour.
    covariance = np.linalg.inv(hessian)
    half = pad * np.sqrt(2.0 * threshold * np.diag(covariance))
    axes = []
    for index in range(2):
        low = theta_hat[index] - half[index]
        high = theta_hat[index] + half[index]
        if bounds is not None:
            limits = np.asarray(bounds, dtype=np.float64)[index]
            low = max(low, float(limits[0]))
            high = min(high, float(limits[1]))
        axes.append(np.linspace(low, high, int(n_grid)))

    offsets = np.stack(
        np.meshgrid(axes[0] - theta_hat[0], axes[1] - theta_hat[1], indexing="ij"),
        axis=-1,
    )
    delta = 0.5 * np.einsum("...i,ij,...j->...", offsets, hessian, offsets)
    if peak_max is not None:
        ceiling = axes[1][None, :] > float(peak_max) - axes[0][:, None]
        delta = np.where(ceiling, np.nan, delta)
    return LaplaceJointRegion(
        param_names=(str(param_names[0]), str(param_names[1])),
        axes=(axes[0], axes[1]),
        delta_ell=delta,
        threshold=threshold,
        cond_number=float(singular.max() / singular.min()),
    )


def contact_jump_curve(lead_times, truth, pred) -> ContactJumpCurve:
    """Reduce ``(simulation, lead, y)`` contact jumps without pseudo-replication.

    Each truth and prediction profile is first reduced to an RMS magnitude over
    ``y``. Error is likewise an RMSE over ``y`` and relative error is its
    profile-L2 ratio. Medians and IQRs are then computed across simulations at
    each lead time.
    """
    lead = np.asarray(lead_times, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if truth.shape != pred.shape or truth.ndim != 3:
        raise ValueError(
            "truth and pred must have the same (n_sims, n_leads, Ny) shape"
        )
    if truth.shape[1] != lead.size:
        raise ValueError(
            f"lead_times has {lead.size} entries for {truth.shape[1]} lead slices"
        )
    if truth.shape[0] == 0 or truth.shape[2] == 0:
        raise ValueError("contact-jump arrays must contain simulations and y nodes")
    if not np.all(np.isfinite(truth)) or not np.all(np.isfinite(pred)):
        raise ValueError("contact-jump arrays must be finite")

    truth_energy = np.sum(truth ** 2, axis=2)
    pred_energy = np.sum(pred ** 2, axis=2)
    error_energy = np.sum((pred - truth) ** 2, axis=2)
    ny = truth.shape[2]
    truth_rms = np.sqrt(truth_energy / ny)
    pred_rms = np.sqrt(pred_energy / ny)
    rmse = np.sqrt(error_energy / ny)
    relative = np.full(error_energy.shape, np.nan, dtype=np.float64)
    usable = truth_energy > 1e-12
    relative[usable] = 100.0 * np.sqrt(error_energy[usable] / truth_energy[usable])

    def _percentile(values, q):
        return np.nanpercentile(values, q, axis=0)

    return ContactJumpCurve(
        lead_times=lead,
        truth_median=np.median(truth_rms, axis=0),
        pred_median=np.median(pred_rms, axis=0),
        rmse_median=np.median(rmse, axis=0),
        rmse_q25=np.percentile(rmse, 25, axis=0),
        rmse_q75=np.percentile(rmse, 75, axis=0),
        relative_median_pct=_percentile(relative, 50),
        relative_q25_pct=_percentile(relative, 25),
        relative_q75_pct=_percentile(relative, 75),
        n_sims=int(truth.shape[0]),
    )


def parity_fit(truth, pred, *, n_sims: int) -> ParityFit:
    """Summarize a truth-versus-prediction cloud for a parity panel.

    The slope is least-squares **through the origin**: a zero jump has to predict
    a zero jump, so an intercept would be fitting a physical impossibility and
    would let a biased model report a slope of 1. Below 1 is systematic
    under-prediction of the jump magnitude, which is the failure mode the panel
    exists to expose.

    Every ``(simulation, lead, y)`` sample is pooled, so ``rmse`` here is a
    point-wise dispersion and not a simulation-level estimate; ``n_sims`` is
    carried through so a caller cannot quote it as one. Use
    :func:`contact_jump_curve` where the claim needs replication units.
    """
    truth = np.asarray(truth, dtype=np.float64).ravel()
    pred = np.asarray(pred, dtype=np.float64).ravel()
    if truth.shape != pred.shape:
        raise ValueError("truth and pred must have the same number of samples")
    if truth.size == 0:
        raise ValueError("parity_fit requires at least one sample")
    if not np.all(np.isfinite(truth)) or not np.all(np.isfinite(pred)):
        raise ValueError("parity arrays must be finite")

    truth_energy = float(np.sum(truth ** 2))
    slope = float(np.sum(truth * pred) / truth_energy) if truth_energy > 1e-12 \
        else float("nan")
    rmse = float(np.sqrt(np.mean((pred - truth) ** 2)))
    truth_rms = float(np.sqrt(truth_energy / truth.size))
    relative = 100.0 * rmse / truth_rms if truth_rms > 1e-12 else float("nan")
    lo = float(min(truth.min(), pred.min()))
    hi = float(max(truth.max(), pred.max()))
    return ParityFit(
        slope=slope, rmse=rmse, truth_rms=truth_rms, relative_pct=relative,
        span=(lo, hi), n_points=int(truth.size), n_sims=int(n_sims),
    )


def _rollout_per_sim(records) -> pd.DataFrame:
    """Pool one rollout arm to ``(seed, benchmark, sim_id, exact lead)``."""
    df = getattr(records, "df", records).copy()
    actual_lead = pd.to_numeric(
        df.get("lead_time_actual", pd.Series(index=df.index, dtype=float)),
        errors="coerce",
    )
    lead_col = "lead_time_actual" if actual_lead.notna().all() else "t_bar"
    required = {
        "seed", "benchmark", "sim_id", "s", "j", lead_col,
        "sse_K2", "target_sse_K2",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"rollout records are missing columns {missing}")
    df["_lead_index"] = (
        pd.to_numeric(df["j"], errors="coerce")
        - pd.to_numeric(df["s"], errors="coerce")
    )
    df["_lead_physical"] = pd.to_numeric(df[lead_col], errors="coerce")
    if df[["_lead_index", "_lead_physical"]].isna().any().any():
        raise ValueError("rollout records contain non-finite lead times")

    keys = ["seed", "benchmark", "sim_id", "_lead_index"]
    grouped = df.groupby(keys, dropna=False, observed=True)
    out = grouped.size().rename("n_pair_rows").reset_index()
    sums = grouped[["sse_K2", "target_sse_K2"]].sum(min_count=1).reset_index()
    out = out.merge(sums, on=keys, how="left")
    lead_lookup = (
        df.groupby("_lead_index", observed=True)["_lead_physical"]
        .median().rename("_lead").reset_index()
    )
    out = out.merge(lead_lookup, on="_lead_index", how="left")
    out["rel_l2_pct"] = pooled_rel_l2_pct(out["sse_K2"], out["target_sse_K2"])
    if not np.all(np.isfinite(out["rel_l2_pct"])):
        raise ValueError("rollout records contain unusable pooled relative-L2 sums")
    return out


def rollout_delta_curves(arms) -> dict[int, RolloutDeltaCurve]:
    """Reduce aligned rollout arms using simulations as the replication unit.

    Every arm is first pooled within simulation and exact lead time. The plotted
    quantity is then computed as a within-simulation paired difference,
    autoregressive minus direct, before taking its median and IQR across
    simulations.
    """
    if set(arms) != {1, 2, 4, 8}:
        raise ValueError(
            f"rollout_delta_curves expects arms [1, 2, 4, 8], got {sorted(arms)}"
        )
    raw = {int(substeps): getattr(arm, "records", arm)
           for substeps, arm in arms.items()}
    pooled = {substeps: _rollout_per_sim(records)
              for substeps, records in raw.items()}
    direct = pooled[1]
    match_on = ["seed", "benchmark", "sim_id", "_lead_index"]
    output: dict[int, RolloutDeltaCurve] = {}
    for substeps in (2, 4, 8):
        merged = direct.merge(
            pooled[substeps], on=match_on, how="outer", indicator=True,
            suffixes=("_direct", "_rollout"),
        )
        if not (merged["_merge"] == "both").all():
            raise ValueError(
                f"K={substeps} is not aligned to direct at the simulation/lead level"
            )
        if not np.allclose(
            merged["target_sse_K2_direct"], merged["target_sse_K2_rollout"],
            rtol=1e-10, atol=1e-10,
        ):
            raise ValueError(f"K={substeps} and direct use different targets")
        if not np.allclose(
            merged["_lead_direct"], merged["_lead_rollout"],
            rtol=1e-6, atol=1e-8,
        ):
            raise ValueError(f"K={substeps} and direct use different lead times")
        merged["_lead"] = merged["_lead_direct"]
        merged["delta_pp"] = (
            merged["rel_l2_pct_rollout"] - merged["rel_l2_pct_direct"]
        )

        leads = np.sort(merged["_lead"].unique().astype(np.float64))
        medians, q25, q75, direct_medians = [], [], [], []
        counts: list[int] = []
        for lead in leads:
            at_lead = merged.loc[merged["_lead"] == lead]
            delta = at_lead["delta_pp"].to_numpy(dtype=np.float64)
            baseline = at_lead["rel_l2_pct_direct"].to_numpy(dtype=np.float64)
            medians.append(float(np.median(delta)))
            q25.append(float(np.percentile(delta, 25)))
            q75.append(float(np.percentile(delta, 75)))
            direct_medians.append(float(np.median(baseline)))
            counts.append(int(delta.size))
        if len(set(counts)) != 1:
            raise ValueError(
                f"K={substeps} has different simulation counts across lead times: "
                f"{counts}"
            )
        output[substeps] = RolloutDeltaCurve(
            substeps=substeps,
            lead_times=leads,
            median_pp=np.asarray(medians),
            q25_pp=np.asarray(q25),
            q75_pp=np.asarray(q75),
            direct_median_pct=np.asarray(direct_medians),
            n_sims=counts[0],
            n_snapshot_pairs=int(len(getattr(raw[substeps], "df", raw[substeps]))),
        )
    return output


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


GLOBAL_FIELD_BENCHMARKS = ("forcing", "source", "source_itr_sin", "interfaces")

# Forcing and interfaces sample a nondimensional R_c in [0.05, 1]; the source
# benchmarks are dimensional (mm^2 K/W, from 17.5 upward). Quantile bins pooled
# over both scales would give each family only the bins its own range happens to
# straddle, so F28 bins each scale separately and draws it on its own x axis.
RESISTANCE_SCALES = {
    "forcing": "nondimensional",
    "interfaces": "nondimensional",
    "source": "source_mm2K_per_W",
    "source_itr_sin": "source_mm2K_per_W",
}


def resistance_scale(benchmark: str) -> str:
    return RESISTANCE_SCALES.get(benchmark, benchmark)


def interface_mean_resistance(frame: pd.DataFrame, benchmark: str,
                              bounds=(0.0, 1.0)) -> np.ndarray:
    """Finite-domain arithmetic mean of the recorded scalar/sinusoidal ITR."""
    base = frame["R_c"].to_numpy(dtype=float)
    if not np.all(np.isfinite(base) & (base >= 0)):
        raise ProvenanceError(f"{benchmark}: invalid scalar/base resistance")
    if benchmark != "source_itr_sin":
        return base
    if "R_c_A" not in frame:
        raise ProvenanceError("source_itr_sin: missing sinusoidal parameter R_c_A")
    amp = frame["R_c_A"].to_numpy(dtype=float)
    if not np.all(np.isfinite(amp) & (amp >= 0)):
        raise ProvenanceError("source_itr_sin: invalid sinusoidal resistance amplitude")
    if "R_c_base" in frame and not np.allclose(frame["R_c_base"], base):
        raise ProvenanceError("source_itr_sin: R_c must agree with R_c_base")
    c, d = bounds
    if not np.isfinite([c, d]).all() or d <= c:
        raise ProvenanceError("invalid interface domain bounds")
    return base + amp * (np.cos(np.pi*c) - np.cos(np.pi*d)) / (np.pi*(d-c))


def _resolved_field_leads(frame, grid=None):
    if grid is None:
        nodes = pd.concat([
            frame[["s", "t_s"]].rename(columns={"s": "index", "t_s": "time"}),
            pd.DataFrame({"index": frame["j"], "time": frame["t_s"] + frame["t_bar"]}),
        ], ignore_index=True)
        grouped = nodes.groupby("index")["time"]
        times = grouped.median()
        if not np.allclose(grouped.min(), grouped.max(), rtol=1e-7, atol=1e-9):
            raise ProvenanceError("records disagree on the time of a snapshot index")
    else:
        times = pd.Series(np.asarray(grid, dtype=float))
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ProvenanceError("snapshot time grid must be finite and increasing")
    start = frame["s"].map(times).to_numpy(dtype=float)
    target = frame["j"].map(times).to_numpy(dtype=float)
    lead = target - start
    if (not np.isfinite(lead).all()
            or not np.allclose(start, frame["t_s"], rtol=1e-7, atol=1e-9)
            or not np.allclose(lead, frame["t_bar"], rtol=1e-7, atol=1e-9)):
        raise ProvenanceError("record times disagree with the snapshot time grid")
    if "lead_time_actual" in frame:
        actual = pd.to_numeric(frame["lead_time_actual"], errors="coerce")
        valid = actual.notna()
        if not np.allclose(actual[valid], lead[valid], rtol=1e-7, atol=1e-9):
            raise ProvenanceError("lead_time_actual disagrees with the snapshot grid")
    snapshot_indices = times.index.to_numpy(dtype=np.int64)
    spacing = (times.iloc[-1] - times.iloc[0]) / (snapshot_indices[-1] - snapshot_indices[0])
    uniform_times = times.iloc[0] + (snapshot_indices - snapshot_indices[0]) * spacing
    # Float32 snapshot roundoff is set by absolute time, not the much smaller
    # lead. Equal index lags must pool together on a verified uniform grid.
    if np.allclose(times, uniform_times, rtol=1e-7, atol=1e-9):
        return (frame["j"] - frame["s"]).to_numpy(dtype=np.int64) * spacing, times
    # Subtraction at different start indices introduces roundoff even on an
    # exact regular grid. Resolve only differences below the grid's precision.
    unique = np.sort(np.unique(lead))
    canonical = []
    for value in unique:
        if not canonical or not np.isclose(value, canonical[-1], rtol=1e-7, atol=1e-9):
            canonical.append(float(value))
    canonical = np.asarray(canonical)
    indices = np.abs(lead[:, None] - canonical).argmin(axis=1)
    return canonical[indices], times


def _prepare_global_field_frames(frames, metadata):
    """Validate F27/F28/F32 records and attach resolved snapshot times.

    Returns ``(prepared, seeds, resistance_info, protocol_info,
    reference_protocol)``. Every downstream reduction reads ``lead_time`` and
    ``source_time`` from here rather than from the raw ``t_bar``/``t_s``
    columns, so the whole family shares one snapshot-grid resolution and one
    cross-benchmark protocol check.
    """
    metadata = metadata or {}
    missing = set(GLOBAL_FIELD_BENCHMARKS) - set(frames)
    if missing:
        raise ProvenanceError(f"global field figures missing benchmarks: {sorted(missing)}")
    prepared, seeds, resistance_info, protocol_info = {}, {}, {}, {}
    reference_protocol = None
    required = {"seed", "benchmark", "sim_id", "s", "j", "t_s", "t_bar",
                "R_c", "sse_K2", "num_error_cells"}
    for benchmark in GLOBAL_FIELD_BENCHMARKS:
        frame = getattr(frames[benchmark], "df", frames[benchmark]).copy()
        if required - set(frame.columns):
            raise ProvenanceError(f"{benchmark}: missing columns {sorted(required - set(frame.columns))}")
        if set(frame["benchmark"]) != {benchmark}:
            raise ProvenanceError(f"{benchmark}: inconsistent record benchmark")
        for name in ("sim_id", "s", "j"):
            values = frame[name].to_numpy(dtype=float)
            if not np.all(np.isfinite(values) & (values >= 0) & (values == np.floor(values))):
                raise ProvenanceError(f"{benchmark}: invalid {name}")
            frame[name] = values.astype(np.int64)
        for name in ("t_s", "t_bar", "sse_K2", "num_error_cells"):
            values = frame[name].to_numpy(dtype=float)
            if not np.all(np.isfinite(values) & (values >= 0)):
                raise ProvenanceError(f"{benchmark}: invalid {name}")
        if (frame["num_error_cells"] <= 0).any():
            raise ProvenanceError(f"{benchmark}: nonpositive error-cell count")
        for name, allowed in (("protocol", {"", "direct_pair"}),
                              ("distribution_class", {"", "in_distribution"}),
                              ("ood_axis", {""})):
            if name in frame and not set(frame[name].fillna("").astype(str)).issubset(allowed):
                raise ProvenanceError(f"{benchmark}: incompatible {name}; requires in-distribution direct pairs")
        frame["seed"] = frame["seed"].astype(str)
        evaluated_seeds = set(frame["seed"])
        if not np.array_equal(frame["j"] > frame["s"], frame["t_bar"] > 0):
            raise ProvenanceError(f"{benchmark}: snapshot ordering disagrees with lead-time sign")
        frame = frame.loc[(frame["j"] > frame["s"]) & (frame["t_bar"] > 0)].copy()
        if frame.empty:
            raise ProvenanceError(f"{benchmark}: no positive-lead prediction pairs")
        if set(frame["seed"]) != evaluated_seeds:
            raise ProvenanceError(f"{benchmark}: an evaluated seed has no eligible positive-lead pairs")
        if frame.duplicated(["seed", "sim_id", "s", "j"]).any():
            raise ProvenanceError(f"{benchmark}: duplicate simulation-pair records")
        seeds[benchmark] = sorted(frame["seed"].unique().tolist())
        seed_parts, first_keys, populations = [], None, []
        for seed in seeds[benchmark]:
            part = frame.loc[frame["seed"] == seed].sort_values(["sim_id", "s", "j"]).copy()
            keys = part[["sim_id", "s", "j"]].to_numpy()
            if first_keys is not None and not np.array_equal(keys, first_keys):
                raise ProvenanceError(f"{benchmark}: mismatched simulation cohorts or eligible pairs across seeds")
            first_keys = keys
            meta = metadata.get(benchmark, {}).get(seed, {})
            population = meta.get("evaluation_population_hash")
            populations.append(population)
            part["lead_time"], times = _resolved_field_leads(part, meta.get("t_grid"))
            part["source_time"] = part["s"].map(times)
            bounds = tuple(meta.get("y_bounds", (0.0, 1.0)))
            part["R_c_mean"] = interface_mean_resistance(part, benchmark, bounds)
            protocol = part[["s", "j", "source_time", "lead_time"]].drop_duplicates().sort_values(["s", "j"])
            protocol = protocol.to_numpy(dtype=float)
            if reference_protocol is not None and (
                    protocol.shape != reference_protocol.shape
                    or not np.allclose(protocol, reference_protocol, rtol=1e-7, atol=1e-9)):
                raise ProvenanceError("incompatible snapshot-pair evaluation protocols across benchmarks/seeds")
            if reference_protocol is None:
                reference_protocol = protocol
            canonical_pairs = {(int(s), int(j)): float(lead)
                               for s, j, _, lead in reference_protocol}
            part["lead_time"] = [canonical_pairs[(s, j)] for s, j in zip(part.s, part.j)]
            protocol_info.setdefault(benchmark, {})[seed] = {
                "time_grid_source": meta.get("time_grid_source", "validated record snapshot times"),
                "snapshot_times": {str(int(i)): float(t) for i, t in times.items()},
                "prediction_mode": meta.get("prediction_mode", "direct_pair (legacy test-record contract)"),
                "cohort_verification": "evaluation population hash" if population else
                                       "legacy simulation/pair keys and recorded parameters",
            }
            resistance_info.setdefault(benchmark, {})[seed] = {
                "kind": "finite-domain sinusoidal mean" if benchmark == "source_itr_sin" else "scalar R_c",
                "y_bounds": list(bounds),
                "bounds_source": meta.get("bounds_source", "benchmark unit-domain convention"),
            }
            seed_parts.append(part)
        if len(set(populations)) > 1:
            raise ProvenanceError(f"{benchmark}: mismatched evaluation populations across seeds")
        combined = pd.concat(seed_parts, ignore_index=True)
        invariant_cols = ["R_c", "R_c_mean"]
        if benchmark == "source_itr_sin":
            invariant_cols += ["R_c_A"]
        grouped = combined.groupby("sim_id")[invariant_cols]
        if not np.allclose(grouped.min(), grouped.max(), rtol=1e-10, atol=1e-12):
            raise ProvenanceError(f"{benchmark}: simulation resistance changes across pairs or seeds")
        identity_fields = [name for name in ("temporal_family", "spatial_family", "regime",
                                             "x_h", "y_h", "A", "freq", "x_I")
                           if name in combined]
        if identity_fields and (combined.groupby("sim_id")[identity_fields].nunique(dropna=False) > 1).to_numpy().any():
            raise ProvenanceError(f"{benchmark}: recorded simulation parameters differ across pairs or seeds")
        prepared[benchmark] = combined
    return prepared, seeds, resistance_info, protocol_info, reference_protocol


def global_field_error_summary(frames, *, metadata=None,
                               n_boot=DEFAULT_N_BOOT, rng_seed=DEFAULT_RNG_SEED):
    """F27/F28 reductions: pooled simulation RMSE, seed mean, cohort median.

    Both strata are reduced together so their exported axes share the same
    range, including when only one figure is requested through the CLI.
    """
    prepared, seeds, resistance_info, protocol_info, _ = _prepare_global_field_frames(
        frames, metadata)

    edges_by_scale = {}
    for scale in dict.fromkeys(resistance_scale(b) for b in prepared):
        values = np.concatenate([
            frame.drop_duplicates("sim_id")["R_c_mean"].to_numpy()
            for b, frame in prepared.items() if resistance_scale(b) == scale])
        scale_edges = np.unique(np.quantile(values, np.linspace(0, 1, 6)))
        if len(scale_edges) == 1:
            scale_edges = np.repeat(scale_edges, 2)
        edges_by_scale[scale] = scale_edges
    lead_rows, itr_rows = [], []
    for benchmark, frame in prepared.items():
        scale = resistance_scale(benchmark)
        edges = edges_by_scale[scale]
        frame["resistance_bin"] = np.clip(
            np.searchsorted(edges, frame["R_c_mean"], side="right") - 1,
            0, len(edges) - 2)
        for stratum, rows, levels in (
                ("lead_time", lead_rows, np.sort(frame["lead_time"].unique())),
                ("resistance_bin", itr_rows, range(len(edges) - 1))):
            pooled = frame.groupby([stratum, "sim_id", "seed"], sort=True).agg(
                sse=("sse_K2", "sum"), cells=("num_error_cells", "sum"),
                n_pairs=("s", "size"), resistance=("R_c_mean", "first"))
            pooled["rmse_K"] = pooled_rmse(pooled["sse"], pooled["cells"])
            simulations = pooled.groupby([stratum, "sim_id"]).agg(
                rmse_K=("rmse_K", "mean"), resistance=("resistance", "first"))
            for level in levels:
                present = level in simulations.index.get_level_values(stratum)
                sample = simulations.xs(level, level=stratum) if present else simulations.iloc[:0]
                values = sample["rmse_K"].to_numpy(dtype=float)
                lo, hi, method = bootstrap_ci(values, n_boot=n_boot, rng_seed=rng_seed)
                part = frame.loc[frame[stratum] == level]
                pair_counts = {seed: int((part["seed"] == seed).sum()) for seed in seeds[benchmark]}
                row = {
                    "benchmark": benchmark, "n_simulations": int(len(values)),
                    "n_seeds": len(seeds[benchmark]), "seed_ids": seeds[benchmark],
                    "eligible_pairs_by_seed": pair_counts,
                    "eligible_pair_rows_total": int(len(part)),
                    "unique_simulation_pairs": int(len(part.drop_duplicates(["sim_id", "s", "j"]))),
                    "median_rmse_K": float(np.median(values)) if len(values) else None,
                    "ci_lower_K": float(lo) if np.isfinite(lo) else None,
                    "ci_upper_K": float(hi) if np.isfinite(hi) else None,
                    "interval_method": method,
                }
                if stratum == "lead_time":
                    row["lead_time"] = float(level)
                else:
                    row.update({"resistance_scale": scale,
                                "bin_index": int(level), "bin_lower": float(edges[level]),
                                "bin_upper": float(edges[level + 1]),
                                "median_resistance": float(sample["resistance"].median()) if len(values) else None})
                rows.append(row)
    values = [row[name] for row in lead_rows + itr_rows
              for name in ("median_rmse_K", "ci_lower_K", "ci_upper_K")
              if row[name] is not None]
    lower, upper = min(values), max(values)
    if lower > 0:
        padding = max((np.log10(upper) - np.log10(lower)) * 0.08, 0.06)
        yscale, ylim = "log", [10 ** (np.log10(lower) - padding), 10 ** (np.log10(upper) + padding)]
    else:
        yscale, ylim = "linear", [0.0, upper * 1.08 if upper > 0 else 1.0]
    counts = {b: len(s) for b, s in seeds.items()}
    return {"lead": lead_rows, "itr": itr_rows, "seeds": seeds,
            "seed_counts": counts, "unequal_seed_counts": len(set(counts.values())) > 1,
            "resistance_bin_edges": {k: v.tolist() for k, v in edges_by_scale.items()},
            "resistance_definitions": resistance_info,
            "protocols": protocol_info, "yscale": yscale, "ylim": ylim,
            "n_boot": n_boot, "rng_seed": rng_seed}


# The source time is chosen as a fraction of the evaluated horizon rather than
# as an absolute time so the choice survives a change of t_final. 0.2 is early
# enough to keep most of the lead range while staying clear of the initial
# condition, which is a uniform 300 K equilibrium in three of the four
# benchmarks and would otherwise turn the figure into a question about
# forecasting from that one special state.
DEFAULT_SOURCE_FRACTION = 0.2
SOURCE_FRACTION_SWEEP = (0.1, 0.2, 0.4, 0.6)


def _select_source_index(protocol, fraction):
    """Snapshot index whose time is nearest ``fraction * horizon``.

    ``protocol`` is the validated ``(s, j, source_time, lead_time)`` array, so
    the horizon is the last evaluated target time and every candidate index is
    guaranteed to carry at least one positive lead. Ties resolve to the earlier
    index, which retains more lead times.
    """
    horizon = float(np.max(protocol[:, 2] + protocol[:, 3]))
    candidates = pd.DataFrame(protocol[:, [0, 2]], columns=["s", "source_time"])
    candidates = candidates.drop_duplicates().sort_values("source_time")
    times = candidates["source_time"].to_numpy(dtype=float)
    pick = int(np.argmin(np.abs(times - fraction * horizon)))
    return int(candidates["s"].to_numpy()[pick]), float(times[pick]), horizon


def _has_target_sse(prepared):
    return all("target_sse_K2" in frame for frame in prepared.values())


def _seed_mean_rmse_by_lead(part, has_target):
    """Simulation RMSE per ``(lead_time, sim_id)``, pooled within a seed first.

    SSE and cell counts are summed before the square root, then the per-seed
    RMSEs are averaged, so a simulation contributes one value per lead however
    many seeds evaluated it.
    """
    pooled = part.groupby(["lead_time", "sim_id", "seed"], sort=True).agg(
        sse=("sse_K2", "sum"), cells=("num_error_cells", "sum"),
        target=("target_sse_K2", "sum") if has_target else ("sse_K2", "size"))
    pooled["rmse_K"] = pooled_rmse(pooled["sse"], pooled["cells"])
    pooled["target_rms_K"] = (pooled_rmse(pooled["target"], pooled["cells"])
                              if has_target else np.nan)
    return pooled.groupby(["lead_time", "sim_id"]).agg(
        rmse_K=("rmse_K", "mean"), target_rms_K=("target_rms_K", "mean"))


def fixed_source_lead_summary(frames, *, metadata=None,
                              source_fraction=DEFAULT_SOURCE_FRACTION,
                              fraction_sweep=SOURCE_FRACTION_SWEEP,
                              n_boot=DEFAULT_N_BOOT, rng_seed=DEFAULT_RNG_SEED):
    """F32 reduction: lead-time curves at a single fixed source snapshot.

    Unlike :func:`global_field_error_summary`, which pools every source time at
    a given lead, this conditions on one source index. Each lead then draws on
    the same simulation cohort, so the curve varies only the prediction horizon
    and the sample size no longer shrinks as the lead grows.

    The primary ``source_fraction`` is the one a figure should plot. Every
    fraction in ``fraction_sweep`` is reduced as well and returned in the same
    row list, flagged by ``is_primary``, so the robustness check over the choice
    of source time is exported without a second render.
    """
    prepared, seeds, _, protocol_info, protocol = _prepare_global_field_frames(
        frames, metadata)

    fractions, seen = [], set()
    for fraction in (source_fraction, *fraction_sweep):
        value = float(fraction)
        if not 0.0 <= value < 1.0:
            raise ProvenanceError(f"source fraction must lie in [0, 1), got {value}")
        if value not in seen:
            seen.add(value)
            fractions.append(value)

    has_target = _has_target_sse(prepared)
    rows, selections = [], {}
    for fraction in fractions:
        index, source_time, horizon = _select_source_index(protocol, fraction)
        primary = fraction == float(source_fraction)
        leads = np.sort(protocol[protocol[:, 0] == index][:, 3])
        selections[f"{fraction:g}"] = {
            "source_index": index, "source_time": source_time,
            "requested_time": fraction * horizon, "horizon": horizon,
            "n_lead_times": int(len(leads)),
            "max_lead_time": float(leads[-1]), "is_primary": primary,
        }
        for benchmark, frame in prepared.items():
            part = frame.loc[frame["s"] == index]
            if part.empty:
                raise ProvenanceError(
                    f"{benchmark}: no pairs at source snapshot {index}")
            simulations = _seed_mean_rmse_by_lead(part, has_target)
            for lead in leads:
                present = lead in simulations.index.get_level_values("lead_time")
                sample = (simulations.xs(lead, level="lead_time") if present
                          else simulations.iloc[:0])
                values = sample["rmse_K"].to_numpy(dtype=float)
                amplitude = sample["target_rms_K"].to_numpy(dtype=float)
                lo, hi, method = bootstrap_ci(values, n_boot=n_boot, rng_seed=rng_seed)
                at_lead = part.loc[np.isclose(part["lead_time"], lead)]
                rows.append({
                    "benchmark": benchmark, "source_fraction": float(fraction),
                    "is_primary": primary, "source_index": index,
                    "source_time": source_time, "lead_time": float(lead),
                    "target_time": source_time + float(lead),
                    "n_simulations": int(len(values)),
                    "n_seeds": len(seeds[benchmark]), "seed_ids": seeds[benchmark],
                    "eligible_pairs_by_seed": {
                        seed: int((at_lead["seed"] == seed).sum())
                        for seed in seeds[benchmark]},
                    "eligible_pair_rows_total": int(len(at_lead)),
                    "unique_simulation_pairs": int(
                        len(at_lead.drop_duplicates(["sim_id", "s", "j"]))),
                    "median_rmse_K": float(np.median(values)) if len(values) else None,
                    "ci_lower_K": float(lo) if np.isfinite(lo) else None,
                    "ci_upper_K": float(hi) if np.isfinite(hi) else None,
                    "interval_method": method,
                    "median_target_rms_K": (float(np.median(amplitude))
                                            if has_target and len(amplitude) else None),
                })

    plotted = [row for row in rows if row["is_primary"]]
    values = [row[name] for row in plotted
              for name in ("median_rmse_K", "ci_lower_K", "ci_upper_K")
              if row[name] is not None]
    if not values:
        raise ProvenanceError("no plottable statistics at the selected source time")
    lower, upper = min(values), max(values)
    if lower > 0:
        padding = max((np.log10(upper) - np.log10(lower)) * 0.08, 0.06)
        yscale = "log"
        ylim = [10 ** (np.log10(lower) - padding), 10 ** (np.log10(upper) + padding)]
    else:
        yscale, ylim = "linear", [0.0, upper * 1.08 if upper > 0 else 1.0]
    counts = {b: len(s) for b, s in seeds.items()}
    return {"rows": rows, "primary": selections[f"{float(source_fraction):g}"],
            "selections": selections, "seeds": seeds, "seed_counts": counts,
            "unequal_seed_counts": len(set(counts.values())) > 1,
            "field_amplitude_available": has_target,
            "protocols": protocol_info, "yscale": yscale, "ylim": ylim,
            "n_boot": n_boot, "rng_seed": rng_seed}


# F34 conditions every panel on the initial condition. That is a fixed time, not
# a fraction of the horizon, so records without an evaluated t_s = 0 snapshot
# are refused rather than plotted from whichever snapshot happens to be nearest.
LEAD_SPREAD_SOURCE_TIME = 0.0
LEAD_SPREAD_QUANTILES = (0.25, 0.5, 0.75)


def lead_error_spread(frames, *, metadata=None,
                      source_time=LEAD_SPREAD_SOURCE_TIME):
    """F34 reduction: per-lead quartiles of simulation RMSE from one source time.

    The band this feeds is the spread of simulation-level error across the test
    cohort, not an interval for the median: it shows how much accuracy varies
    between cases at a lead, and it does not narrow as the cohort grows.
    """
    prepared, seeds, _, protocol_info, protocol = _prepare_global_field_frames(
        frames, metadata)
    at_source = protocol[np.isclose(protocol[:, 2], source_time, rtol=0.0, atol=1e-9)]
    if not len(at_source):
        earliest = ", ".join(f"{t:g}" for t in np.unique(protocol[:, 2])[:5])
        raise ProvenanceError(
            f"no evaluated source snapshot at t_s={source_time:g}; "
            f"earliest evaluated source times: {earliest}")
    index = int(at_source[0, 0])
    leads = np.sort(at_source[:, 3])
    has_target = _has_target_sse(prepared)

    rows, growth = [], {}
    for benchmark in GLOBAL_FIELD_BENCHMARKS:
        simulations = _seed_mean_rmse_by_lead(
            prepared[benchmark].loc[prepared[benchmark]["s"] == index], has_target)
        present = set(simulations.index.get_level_values("lead_time"))
        cohorts = set()
        for lead in leads:
            if lead not in present:
                raise ProvenanceError(
                    f"{benchmark}: no simulations at t_s={source_time:g}, lead {lead:g}")
            sample = simulations.xs(lead, level="lead_time")
            values = sample["rmse_K"].to_numpy(dtype=float)
            q25, q50, q75 = np.quantile(values, LEAD_SPREAD_QUANTILES)
            amplitude = sample["target_rms_K"].to_numpy(dtype=float)
            cohorts.add(tuple(sample.index))
            rows.append({
                "benchmark": benchmark, "source_index": index,
                "source_time": float(source_time), "lead_time": float(lead),
                "n_simulations": int(len(values)),
                "n_seeds": len(seeds[benchmark]), "seed_ids": seeds[benchmark],
                "median_rmse_K": float(q50),
                "q25_rmse_K": float(q25), "q75_rmse_K": float(q75),
                "median_target_rms_K": float(np.median(amplitude)) if has_target else None,
            })
        curve = [row for row in rows if row["benchmark"] == benchmark]
        first, last = curve[0], curve[-1]
        peak = max(curve, key=lambda row: row["median_rmse_K"])
        growth[benchmark] = {
            "first_lead": first["lead_time"], "first_median_rmse_K": first["median_rmse_K"],
            "last_lead": last["lead_time"], "last_median_rmse_K": last["median_rmse_K"],
            "ratio_last_to_first": (last["median_rmse_K"] / first["median_rmse_K"]
                                    if first["median_rmse_K"] > 0 else None),
            "increase_K": last["median_rmse_K"] - first["median_rmse_K"],
            "peak_lead": peak["lead_time"], "peak_median_rmse_K": peak["median_rmse_K"],
            "n_simulations": max(row["n_simulations"] for row in curve),
            "constant_cohort": len(cohorts) == 1,
        }

    # One RMSE range per benchmark: their error levels differ several-fold, and a
    # shared range flattens the curves of the more accurate ones.
    ylims = {}
    for benchmark in GLOBAL_FIELD_BENCHMARKS:
        upper = max(row["q75_rmse_K"] for row in rows if row["benchmark"] == benchmark)
        ylims[benchmark] = [0.0, upper * 1.08 if upper > 0 else 1.0]
    counts = {b: len(s) for b, s in seeds.items()}
    return {"rows": rows, "growth": growth,
            "source_index": index, "source_time": float(source_time),
            "lead_times": leads.tolist(),
            "quantiles": list(LEAD_SPREAD_QUANTILES),
            "seeds": seeds, "seed_counts": counts,
            "unequal_seed_counts": len(set(counts.values())) > 1,
            "field_amplitude_available": has_target,
            "protocols": protocol_info, "ylims": ylims}


# A curve's shape can only be compared against another curve over leads both of
# them reach, and a source index s reaches lag K only when s + K is still on the
# grid. Every window is therefore a trade: a long window compares few curves
# over many leads, a short one compares many curves over few. The defaults are
# fractions of the longest lag rather than absolute lags so the same three
# points on that trade survive a change of snapshot count.
DEFAULT_SHAPE_WINDOW_FRACTIONS = (0.8, 0.35, 0.15)
DEFAULT_SHAPE_N_BOOT = 1_000
MIN_CURVES_FOR_SHAPE = 3
_SHAPE_BOOT_CHUNK = 100
# Ceiling on one gathered resample block, in float64 elements (~160 MB).
_SHAPE_BOOT_MAX_ELEMENTS = 20_000_000


def _lead_by_lag(protocol):
    """Lead time as a function of snapshot lag ``j - s``.

    Comparing curves across source times presumes lag ``K`` means the same
    prediction horizon wherever it is measured. On a non-uniform snapshot grid
    it does not, and the comparison has no meaning, so that case is refused
    rather than silently reduced.
    """
    lag = np.rint(protocol[:, 1] - protocol[:, 0]).astype(np.int64)
    leads = pd.DataFrame({"lag": lag, "lead": protocol[:, 3]}).groupby("lag")["lead"]
    if not np.allclose((leads.max() - leads.min()).to_numpy(), 0.0, atol=1e-9):
        raise ProvenanceError(
            "lead time is not a function of snapshot lag; comparing error-vs-lead "
            "shape across source times is undefined on a non-uniform snapshot grid")
    return leads.min()


def _shape_lags_by_source(protocol):
    lags = {}
    for s, j, _, _ in protocol:
        lags.setdefault(int(s), set()).add(int(round(j - s)))
    return lags


def _complete_shape_cohort(frame, n_pairs):
    """Simulations contributing every evaluated pair under every seed.

    A curve that gains and loses simulations as the lag grows changes cohort
    along its own length, and that shows up as shape. Restricting to the
    complete cohort is what makes the residual below attributable to the model
    rather than to who was measured.
    """
    counts = frame.groupby(["sim_id", "seed"]).size().unstack("seed")
    complete = (counts == n_pairs).all(axis=1).to_numpy()
    ids = counts.index.to_numpy()
    return ids[complete], ids[~complete]


def _additive_interaction(matrix):
    """Residual of the two-way additive fit ``mu + alpha_s + beta_k``.

    The input is ``log10`` error, so additive means multiplicative in kelvin:
    the model asserts that every source time traces one common lead-time shape
    ``beta`` and differs from the others only by a constant factor ``alpha``.
    Whatever the fit cannot reach is exactly the disagreement in shape.

    Axes beyond the first two broadcast, so a stack of bootstrap replicates
    reduces in a single call.
    """
    grand = matrix.mean(axis=(0, 1))
    alpha = matrix.mean(axis=1) - grand
    beta = matrix.mean(axis=0) - grand
    return matrix - (grand + alpha[:, None] + beta[None, :]), alpha, beta


def _shape_decomposition(log_matrix):
    """Interaction size, common-shape fraction, and per-source residual RMS."""
    resid, alpha, beta = _additive_interaction(log_matrix)
    n_cells = resid.shape[0] * resid.shape[1]
    ss_beta = (beta ** 2).sum(axis=0) * resid.shape[0]
    ss_resid = (resid ** 2).sum(axis=(0, 1))
    return {
        "interaction_rms_log10": np.sqrt(ss_resid / n_cells),
        "common_shape_fraction": np.where(ss_beta + ss_resid > 0,
                                          ss_beta / np.where(ss_beta + ss_resid > 0,
                                                             ss_beta + ss_resid, 1.0), 1.0),
        "level_spread_log10": alpha.max(axis=0) - alpha.min(axis=0),
        "per_source_rms": np.sqrt((resid ** 2).mean(axis=1)),
    }


def _shape_bootstrap(log_values, shape, n_boot, rng_seed):
    """Paired simulation resample of an ``(n_cells, n_sims)`` log value table.

    Every source time's curve is measured on the same simulations, so the
    replicate has to redraw the simulations once and recompute all of the
    curves on that draw. Resampling each cell on its own would break the
    pairing that the whole comparison rests on.

    Replicates run in chunks because the gather is ``(n_cells, chunk, n_sims)``
    before the median reduces it. A published cohort is thousands of
    simulations wide, where a fixed chunk would reserve gigabytes, so the chunk
    is sized to the table instead.
    """
    rng = np.random.default_rng(rng_seed)
    n_sims = log_values.shape[1]
    chunk = max(1, min(_SHAPE_BOOT_CHUNK,
                       _SHAPE_BOOT_MAX_ELEMENTS // max(1, log_values.size)))
    draws = []
    for start in range(0, n_boot, chunk):
        size = min(chunk, n_boot - start)
        idx = rng.integers(0, n_sims, size=(size, n_sims))
        block = np.median(log_values[:, idx], axis=2)
        draws.append(block.reshape(*shape, size))
    return np.concatenate(draws, axis=-1)


def source_time_shape_typicality(frames, *, metadata=None,
                                 source_fraction=DEFAULT_SOURCE_FRACTION,
                                 window_fractions=DEFAULT_SHAPE_WINDOW_FRACTIONS,
                                 n_boot=DEFAULT_SHAPE_N_BOOT,
                                 rng_seed=DEFAULT_RNG_SEED):
    """Is the published source time's error-vs-lead curve shaped like the rest?

    F32 plots one source snapshot. This asks whether that snapshot's lead-time
    degradation is representative by fitting ``log10 RMSE(s, k) = mu + alpha_s
    + beta_k`` over every source time that spans a common lead window. If the
    fit is tight, all source times share one shape and differ only in level, so
    no choice of source time could have been unrepresentative. Only when the
    interaction is material does the published curve's standing among the
    others carry information, and it is then reported as a rank with a paired
    bootstrap interval.

    Note this describes a direct time-conditioned operator: each pair is
    predicted independently from its source snapshot, so a rising curve is
    lead-time-dependent degradation, not error accumulated over a rollout.
    """
    prepared, seeds, _, protocol_info, protocol = _prepare_global_field_frames(
        frames, metadata)

    lead_by_lag = _lead_by_lag(protocol)
    lags_by_source = _shape_lags_by_source(protocol)
    max_lag = int(max(lead_by_lag.index))
    published_index, published_time, horizon = _select_source_index(
        protocol, float(source_fraction))

    windows, seen = [], set()
    for fraction in window_fractions:
        window = max(2, int(round(float(fraction) * max_lag)))
        if window <= max_lag and window not in seen:
            seen.add(window)
            windows.append(window)
    if not windows:
        raise ProvenanceError("no usable lead window for a shape comparison")

    n_pairs = len(protocol)
    cohort, degradations = {}, []
    tables = {}
    for benchmark, frame in prepared.items():
        keep, dropped = _complete_shape_cohort(frame, n_pairs)
        if len(keep) < MIN_CURVES_FOR_SHAPE:
            raise ProvenanceError(
                f"{benchmark}: {len(keep)} simulations evaluate the full pair "
                f"protocol; a shape comparison needs at least {MIN_CURVES_FOR_SHAPE}")
        if len(dropped):
            degradations.append(Degradation(
                "incomplete_pair_cohort",
                f"{benchmark}: {len(dropped)} of {len(keep) + len(dropped)} simulations "
                "miss at least one evaluated pair and are excluded so every curve is "
                "measured on one cohort"))
        part = frame.loc[frame["sim_id"].isin(set(keep.tolist()))].copy()
        part["lag"] = part["j"] - part["s"]
        pooled = part.groupby(["s", "lag", "sim_id", "seed"], sort=True).agg(
            sse=("sse_K2", "sum"), cells=("num_error_cells", "sum"))
        pooled["rmse_K"] = pooled_rmse(pooled["sse"], pooled["cells"])
        table = pooled.groupby(["s", "lag", "sim_id"])["rmse_K"].mean().unstack("sim_id")
        if not np.all(np.isfinite(table.to_numpy()) & (table.to_numpy() > 0)):
            raise ProvenanceError(
                f"{benchmark}: nonpositive or missing pooled RMSE; the log-space "
                "decomposition is undefined")
        tables[benchmark] = table
        cohort[benchmark] = {"n_simulations": int(len(keep)),
                             "n_dropped": int(len(dropped)),
                             "dropped_sim_ids": sorted(int(i) for i in dropped)}

    rows, source_rows = [], []
    for window in windows:
        needed = set(range(1, window + 1))
        retained = sorted(s for s, lags in lags_by_source.items() if needed <= lags)
        if len(retained) < MIN_CURVES_FOR_SHAPE:
            continue
        covers_published = published_index in retained
        cells = [(s, k) for s in retained for k in range(1, window + 1)]
        shape = (len(retained), window)
        published_row = retained.index(published_index) if covers_published else None

        for benchmark, table in tables.items():
            # log10 first: the median is order-preserving, so the cohort median
            # of the logs is the log of the cohort median, and the null table
            # below is then a plain subtraction.
            log_values = np.log10(table.loc[cells].to_numpy(dtype=float))
            observed = np.median(log_values, axis=1).reshape(shape)
            stats = _shape_decomposition(observed)
            per_source = stats["per_source_rms"]
            order = np.argsort(per_source)
            # Competition rank, the same count the bootstrap below takes, so
            # curves that agree to floating-point dust all rank first instead of
            # being ordered by argsort's arbitrary tie-breaking -- which would
            # otherwise hand the published curve a last place it did not earn,
            # and put it outside its own interval.
            rank = (1 + int((per_source < per_source[published_row]).sum())
                    if covers_published else None)

            # The rank, not the interaction size, carries the verdict. A
            # resampled median is noisier than the median it is drawn from, so
            # every replicate overstates the interaction and an interval around
            # it would be biased outward. Rank is a comparison made inside one
            # replicate, where that inflation applies to all curves at once and
            # cancels: under curves that genuinely share a shape the interval
            # widens to the whole set, which is the answer being sought.
            n_sims = log_values.shape[1]
            interval = {"rank_ci_lower": None, "rank_ci_upper": None,
                        "rank_span_fraction": None, "interval_method": "none"}
            if covers_published and n_sims >= MIN_SIMS_FOR_CI and n_boot > 0:
                boot = _shape_decomposition(
                    _shape_bootstrap(log_values, shape, n_boot, rng_seed))
                ranks = 1 + (boot["per_source_rms"]
                             < boot["per_source_rms"][published_row]).sum(axis=0)
                lo, hi = (int(v) for v in np.percentile(ranks, [2.5, 97.5]))
                interval.update(
                    rank_ci_lower=lo, rank_ci_upper=hi,
                    rank_span_fraction=float((hi - lo + 1) / len(retained)),
                    interval_method="paired simulation bootstrap (percentile)")

            interaction = float(stats["interaction_rms_log10"])
            rows.append({
                "benchmark": benchmark, "window_lags": int(window),
                "window_lead_time": float(lead_by_lag.loc[window]),
                "n_curves": len(retained), "source_indices": retained,
                "n_simulations": n_sims,
                "n_seeds": len(seeds[benchmark]), "seed_ids": seeds[benchmark],
                "covers_published": covers_published,
                "interaction_rms_log10": interaction,
                "interaction_pct": float(100.0 * (10.0 ** interaction - 1.0)),
                "common_shape_fraction": float(stats["common_shape_fraction"]),
                "n_boot": int(n_boot),
                "level_spread_pct": float(
                    100.0 * (10.0 ** float(stats["level_spread_log10"]) - 1.0)),
                "published_rank": rank,
                "published_rms_log10": (float(per_source[published_row])
                                        if covers_published else None),
                "most_typical_index": int(retained[int(order[0])]),
                "least_typical_index": int(retained[int(order[-1])]),
                **interval,
            })
            for position, index in enumerate(retained):
                source_rows.append({
                    "benchmark": benchmark, "window_lags": int(window),
                    "source_index": int(index),
                    "source_time": float(protocol[protocol[:, 0] == index][0, 2]),
                    "is_published": index == published_index,
                    "residual_rms_log10": float(per_source[position]),
                    "residual_pct": float(100.0 * (10.0 ** float(per_source[position]) - 1.0)),
                    "rank": int(np.where(order == position)[0][0]) + 1,
                })

    if not rows:
        raise ProvenanceError(
            f"no lead window retains {MIN_CURVES_FOR_SHAPE} source times; the "
            "snapshot grid is too short for a shape comparison")

    counts = {b: len(s) for b, s in seeds.items()}
    return {
        "rows": rows, "sources": source_rows,
        "published": {"source_index": published_index, "source_time": published_time,
                      "source_fraction": float(source_fraction),
                      "horizon": horizon, "max_lead_lag": max_lag - published_index},
        "windows": windows, "max_lag": max_lag,
        "lead_by_lag": {int(k): float(v) for k, v in lead_by_lag.items()},
        "cohort": cohort, "degradations": degradations,
        "seeds": seeds, "seed_counts": counts,
        "unequal_seed_counts": len(set(counts.values())) > 1,
        "protocols": protocol_info, "n_boot": n_boot, "rng_seed": rng_seed,
    }


SURFACE_BENCHMARKS = ("interfaces", "source_itr_sin")


def _additive_fit_incomplete(values, rows, cols, n_rows, n_cols):
    """Least-squares ``mu + alpha_row + beta_col`` over an arbitrary cell set.

    :func:`_additive_interaction` takes row and column means, which is the
    least-squares fit only when every cell is observed. The evaluated ``(s, k)``
    domain is triangular -- a late source time reaches no long lag -- so the
    fit has to be solved rather than averaged. On a complete grid this returns
    the same answer.

    The design is rank deficient by two, one redundancy per factor, and
    ``lstsq`` resolves that with its minimum-norm solution; the coefficients are
    then recentered so ``alpha`` and ``beta`` read as offsets. Fitted values are
    invariant to both choices. What is *not* automatic is identifiability: if
    the observed cells split into groups sharing no row and no column, the level
    difference between those groups is unmeasurable and the rank falls short.
    """
    rows, cols = np.asarray(rows), np.asarray(cols)
    design = np.zeros((len(values), 1 + n_rows + n_cols))
    design[:, 0] = 1.0
    design[np.arange(len(rows)), 1 + rows] = 1.0
    design[np.arange(len(cols)), 1 + n_rows + cols] = 1.0
    if np.linalg.matrix_rank(design) != n_rows + n_cols - 1:
        raise ProvenanceError(
            "the evaluated source/lead cells are disconnected; a shared "
            "lead-time shape is not identifiable across them")
    coef = np.linalg.lstsq(design, values, rcond=None)[0]
    alpha, beta = coef[1:1 + n_rows], coef[1 + n_rows:]
    return (float(coef[0] + alpha.mean() + beta.mean()),
            alpha - alpha.mean(), beta - beta.mean())


def source_lead_error_surface(frames, *, metadata=None,
                              benchmarks=SURFACE_BENCHMARKS,
                              source_fraction=DEFAULT_SOURCE_FRACTION):
    """The whole ``log10 RMSE(source time, lead time)`` surface per benchmark.

    F32 draws one column of this surface and
    :func:`source_time_shape_typicality` tests whether that column is
    representative. Both answer the question indirectly. This returns the
    surface itself, together with the additive fit ``mu + alpha_s + beta_k`` and
    its residual, so the two ways a single source time can mislead are separable
    by eye: ``alpha`` is how much the level moves with source time, and the
    residual is how much the *shape* does.

    The domain is triangular, not rectangular, so unlike the windowed
    comparison this is an unbalanced design -- long leads are observed at fewer
    source times than short ones. The fit accounts for that, but the summary
    scalars are weighted by cell coverage and are therefore not comparable to
    the balanced-window values that carry the verdict there. Nothing here is a
    hypothesis test; it is the data the test is run on.
    """
    unknown = [b for b in benchmarks if b not in GLOBAL_FIELD_BENCHMARKS]
    if unknown:
        raise ProvenanceError(f"unknown global-field benchmarks: {sorted(unknown)}")

    prepared, seeds, _, protocol_info, protocol = _prepare_global_field_frames(
        frames, metadata)
    lead_by_lag = _lead_by_lag(protocol)
    lags_by_source = _shape_lags_by_source(protocol)
    max_lag = int(max(lead_by_lag.index))
    published_index, published_time, horizon = _select_source_index(
        protocol, float(source_fraction))

    sources = sorted(lags_by_source)
    lags = list(range(1, max_lag + 1))
    row_of = {s: i for i, s in enumerate(sources)}
    source_time = {int(s): float(t) for s, _, t, _ in protocol}
    cells = [(s, k) for s in sources for k in sorted(lags_by_source[s])]
    cell_rows = [row_of[s] for s, _ in cells]
    cell_cols = [k - 1 for _, k in cells]

    n_pairs = len(protocol)
    out, degradations = {}, []
    for benchmark in benchmarks:
        frame = prepared[benchmark]
        keep, dropped = _complete_shape_cohort(frame, n_pairs)
        if len(keep) < MIN_CURVES_FOR_SHAPE:
            raise ProvenanceError(
                f"{benchmark}: {len(keep)} simulations evaluate the full pair "
                f"protocol; the surface needs at least {MIN_CURVES_FOR_SHAPE}")
        if len(dropped):
            degradations.append(Degradation(
                "incomplete_pair_cohort",
                f"{benchmark}: {len(dropped)} of {len(keep) + len(dropped)} simulations "
                "miss at least one evaluated pair and are excluded so every cell of the "
                "surface is measured on one cohort"))
        part = frame.loc[frame["sim_id"].isin(set(keep.tolist()))].copy()
        part["lag"] = part["j"] - part["s"]
        pooled = part.groupby(["s", "lag", "sim_id", "seed"], sort=True).agg(
            sse=("sse_K2", "sum"), cells=("num_error_cells", "sum"))
        pooled["rmse_K"] = pooled_rmse(pooled["sse"], pooled["cells"])
        table = pooled.groupby(["s", "lag", "sim_id"])["rmse_K"].mean().unstack("sim_id")
        values = table.loc[cells].to_numpy(dtype=float)
        if not np.all(np.isfinite(values) & (values > 0)):
            raise ProvenanceError(
                f"{benchmark}: nonpositive or missing pooled RMSE; the log-space "
                "surface is undefined")
        observed = np.median(np.log10(values), axis=1)

        grand, alpha, beta = _additive_fit_incomplete(
            observed, cell_rows, cell_cols, len(sources), len(lags))
        fitted = grand + alpha[cell_rows] + beta[cell_cols]
        residual = observed - fitted

        blank = np.full((len(sources), len(lags)), np.nan)
        surface, fit_surface, residual_surface = (blank.copy() for _ in range(3))
        surface[cell_rows, cell_cols] = observed
        fit_surface[cell_rows, cell_cols] = fitted
        residual_surface[cell_rows, cell_cols] = residual

        ss_resid = float((residual ** 2).sum())
        ss_beta = float((beta[cell_cols] ** 2).sum())
        interaction = float(np.sqrt(ss_resid / len(cells)))
        level = float(alpha.max() - alpha.min())
        out[benchmark] = {
            "surface_log10": surface,
            "additive_fit_log10": fit_surface,
            "residual_log10": residual_surface,
            "level_log10": alpha,
            "shape_log10": beta,
            "grand_log10": grand,
            "interaction_rms_log10": interaction,
            "interaction_pct": float(100.0 * (10.0 ** interaction - 1.0)),
            "level_spread_pct": float(100.0 * (10.0 ** level - 1.0)),
            "common_shape_fraction": (ss_beta / (ss_beta + ss_resid)
                                      if ss_beta + ss_resid > 0 else 1.0),
            "n_simulations": int(len(keep)),
            "n_dropped": int(len(dropped)),
            "n_cells": len(cells),
            "n_seeds": len(seeds[benchmark]),
            "seed_ids": seeds[benchmark],
        }

    counts = {b: out[b]["n_seeds"] for b in benchmarks}
    return {
        "benchmarks": list(benchmarks), "surfaces": out,
        "source_indices": sources,
        "source_times": [source_time[s] for s in sources],
        "lags": lags,
        "lead_times": [float(lead_by_lag.loc[k]) for k in lags],
        "published": {"source_index": published_index, "source_time": published_time,
                      "source_fraction": float(source_fraction), "horizon": horizon,
                      "max_lead_lag": max_lag - published_index},
        "max_lag": max_lag, "degradations": degradations,
        "seeds": {b: seeds[b] for b in benchmarks}, "seed_counts": counts,
        "unequal_seed_counts": len(set(counts.values())) > 1,
        "protocols": {b: protocol_info[b] for b in benchmarks},
        "domain": "triangular; cell (s, k) exists iff the target snapshot s + k is evaluated",
        "fit": "least squares mu + alpha_s + beta_k over observed cells; unbalanced design",
    }


__all__ = [
    "DEFAULT_SHAPE_N_BOOT",
    "DEFAULT_SHAPE_WINDOW_FRACTIONS",
    "MIN_CURVES_FOR_SHAPE",
    "SURFACE_BENCHMARKS",
    "source_lead_error_surface",
    "source_time_shape_typicality",
    "DEFAULT_SOURCE_FRACTION",
    "GLOBAL_FIELD_BENCHMARKS",
    "SOURCE_FRACTION_SWEEP",
    "fixed_source_lead_summary",
    "LEAD_SPREAD_QUANTILES",
    "LEAD_SPREAD_SOURCE_TIME",
    "lead_error_spread",
    "global_field_error_summary",
    "interface_mean_resistance",
    "ContactJumpCurve",
    "DEFAULT_METRICS",
    "DEFAULT_N_BOOT",
    "DEFAULT_RNG_SEED",
    "MEAN_METRICS",
    "METRICS",
    "MIN_SEEDS_FOR_SPREAD",
    "MIN_SIMS_FOR_CI",
    "MIN_SIMS_FOR_P99",
    "InverseSensorSweepSummary",
    "LaplaceJointRegion",
    "MetricSpec",
    "POOLED_METRICS",
    "PairedDelta",
    "ParityFit",
    "RATIO_METRICS",
    "RankCorrelation",
    "RateCI",
    "RolloutDeltaCurve",
    "SensorSweepCurve",
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
    "contact_jump_curve",
    "ecdf",
    "ecdf_band",
    "exceedance_rate",
    "inverse_profile_l2_error",
    "inverse_sensor_sweep_summaries",
    "laplace_joint_region",
    "metric_spec",
    "representative_inverse_case",
    "paired_seed_delta",
    "parity_fit",
    "per_sim",
    "pooled_rel_l2_pct",
    "pooled_rmse",
    "proportion",
    "rollout_delta_curves",
    "shared_bin_edges",
    "spearman",
    "stratum_order",
    "stratum_spec",
    "summarize",
    "wilson_ci",
]
