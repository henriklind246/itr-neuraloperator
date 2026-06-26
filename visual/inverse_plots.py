"""Inverse-problem paper figures (registry group ``inverse``).

A single generic template renders a coherent "Core 4" set for *both* inversion
benchmarks so they read as siblings (same layout / style / palette), differing
only in parameter dimensionality and the lead deliverable:

- ``forcing``    — scalar interface resistance ``R_c`` (theta_dim=1; lead = R_c).
- ``source_itr`` — Gaussian-void profile ``(R_base, R_amp, y0, sigma)``
  (theta_dim=4; lead = ``excess_int`` = void severity). ``R_amp``/``sigma`` are
  individually non-identifiable; severity is the well-conditioned quantity.

The four functions branch only on the :class:`InversePlotSpec` (column names,
``has_ridge``, parameter count, artifact presence) — never on the benchmark
name. CSV columns mirror the adapter ``*_summary`` methods in
``scripts/inverse_adapters.py`` exactly (the source of truth). Per-sim raw
profile curves, MCMC samples, and the observation Jacobian live in the optional
NPZ artifacts written by ``scripts/invert.py --artifact-dir``; without them Figs
2 and 4 degrade gracefully (printed skip note, no crash).

Residual units: ``invert.py`` stores ``fno_resid``/``fv_resid``/
``fno_vs_fv_resid`` as half-MSE over the same sensors in normalized-temperature
units (``0.5*mean((a-b)^2)``); Fig 3 plots ``RMS = sqrt(2*half-MSE)``.
"""

import csv as _csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from visual._common import PLOT_STYLE, _save_figure


# ============================================================
# PALETTE
# ============================================================

# Okabe-Ito, matched to visual/paper_plots.py so a benchmark keeps its color.
_BENCHMARK_COLORS = {
    "forcing": "#0072B2",     # blue
    "source_itr": "#CC79A7",  # reddish purple
}


# ============================================================
# SPEC — all benchmark-specific data the generic functions need
# ============================================================

@dataclass(frozen=True)
class InversePlotSpec:
    """Column names + structural flags for one inversion benchmark.

    Every field that ends in ``_col``/``_cols`` is a CSV column name (or mapping
    to one) produced by the matching adapter ``*_summary`` method. ``param_ranges``
    are the admissible physical ranges (``profile_bounds``) used to normalize
    per-parameter errors and to scale the MCMC ridge into normalized coordinates.
    """

    benchmark: str
    color: str
    param_names: tuple[str, ...]
    param_ranges: dict[str, tuple[float, float]]
    hat_cols: dict[str, str]
    true_cols: dict[str, str]
    abserr_cols: dict[str, str]
    # Lead deliverable (forcing: R_c; source_itr: excess_int / severity).
    lead_name: str
    lead_hat_col: str
    lead_true_col: str
    lead_abserr_col: str
    lead_unit: str
    # Identifiability (Fig 2).
    sv_cols: tuple[str, ...]
    sens_cols: dict[str, str]
    cond_col: str | None          # None for forcing (cond ~= 1, uninformative)
    align_col: str | None         # ramp/sigma alignment (source_itr only)
    has_ridge: bool
    ridge_pair: tuple[int, int] | None   # theta indices for the MCMC cloud
    # Surrogate-error inflation variance column (Fig 2/4 context).
    sigma_eff2_col: str
    # UQ (Fig 4).
    profile_param_name: str
    profile_ci_low_col: str
    profile_ci_high_col: str
    profile_covered_col: str
    mcmc_lead_mean_col: str
    mcmc_lead_ci_low_col: str
    mcmc_lead_ci_high_col: str
    mcmc_lead_covered_col: str
    # source_itr only: profile-path severity range (a path projection, NOT a CI).
    profile_excess_low_col: str | None = None
    profile_excess_high_col: str | None = None
    # Interpretive second suptitle line for Fig 1 (None => single-line title).
    recovery_subtitle: str | None = None
    # Parameters that trade off and are individually weakly identified; their
    # Fig 1 panels get an "expected parameter trade-off" tag.
    non_identifiable_params: frozenset[str] = frozenset()


FORCING_SPEC = InversePlotSpec(
    benchmark="forcing",
    color=_BENCHMARK_COLORS["forcing"],
    param_names=("R_c",),
    param_ranges={"R_c": (0.05, 1.0)},
    hat_cols={"R_c": "R_c_map"},
    true_cols={"R_c": "R_c_true"},
    abserr_cols={"R_c": "R_c_abs_error"},
    lead_name="R_c",
    lead_hat_col="R_c_map",
    lead_true_col="R_c_true",
    lead_abserr_col="R_c_abs_error",
    lead_unit="",
    sv_cols=("sv_0",),
    sens_cols={"R_c": "sens_R_c"},
    cond_col=None,
    align_col=None,
    has_ridge=False,
    ridge_pair=None,
    sigma_eff2_col="uq_sigma_eff2",
    profile_param_name="R_c",
    profile_ci_low_col="profile_R_c_ci_low",
    profile_ci_high_col="profile_R_c_ci_high",
    profile_covered_col="profile_R_c_covered",
    mcmc_lead_mean_col="mcmc_R_c_mean",
    mcmc_lead_ci_low_col="mcmc_R_c_ci_low",
    mcmc_lead_ci_high_col="mcmc_R_c_ci_high",
    mcmc_lead_covered_col="mcmc_R_c_covered",
)


SOURCE_ITR_SPEC = InversePlotSpec(
    benchmark="source_itr",
    color=_BENCHMARK_COLORS["source_itr"],
    param_names=("R_base", "R_amp", "y0", "sigma"),
    param_ranges={
        "R_base": (0.05, 1.0),
        "R_amp": (0.0, 2.95),
        "y0": (0.1, 0.9),
        "sigma": (0.05, 0.2),
    },
    hat_cols={
        "R_base": "R_base_hat", "R_amp": "R_amp_hat",
        "y0": "y0_hat", "sigma": "sigma_hat",
    },
    true_cols={
        "R_base": "R_base_true", "R_amp": "R_amp_true",
        "y0": "y0_true", "sigma": "sigma_true",
    },
    abserr_cols={
        "R_base": "R_base_abserr", "R_amp": "R_amp_abserr",
        "y0": "y0_abserr", "sigma": "sigma_abserr",
    },
    lead_name="excess_int",
    lead_hat_col="excess_int_hat",
    lead_true_col="excess_int_true",
    lead_abserr_col="excess_int_abserr",
    lead_unit=" (severity)",
    sv_cols=("sv_0", "sv_1", "sv_2", "sv_3"),
    sens_cols={
        "R_base": "sens_R_base", "R_amp": "sens_R_amp",
        "y0": "sens_y0", "sigma": "sens_sigma",
    },
    cond_col="cond_number",
    align_col="ramp_sigma_alignment",
    has_ridge=True,
    ridge_pair=(1, 3),   # (R_amp, sigma)
    sigma_eff2_col="uq_sigma_eff2",
    profile_param_name="R_amp",
    profile_ci_low_col="profile_R_amp_ci_low",
    profile_ci_high_col="profile_R_amp_ci_high",
    profile_covered_col="profile_R_amp_covered",
    mcmc_lead_mean_col="mcmc_excess_mean",
    mcmc_lead_ci_low_col="mcmc_excess_ci_low",
    mcmc_lead_ci_high_col="mcmc_excess_ci_high",
    mcmc_lead_covered_col="mcmc_excess_covered",
    profile_excess_low_col="profile_excess_ci_low",
    profile_excess_high_col="profile_excess_ci_high",
    recovery_subtitle=(
        "R_amp and sigma trade off strongly; "
        "integrated severity (left) is the primary estimand"
    ),
    non_identifiable_params=frozenset({"R_amp", "sigma"}),
)


_SPECS = {"forcing": FORCING_SPEC, "source_itr": SOURCE_ITR_SPEC}


def spec_for(benchmark: str) -> InversePlotSpec:
    """Return the :class:`InversePlotSpec` for ``benchmark``."""
    try:
        return _SPECS[benchmark]
    except KeyError:
        raise ValueError(
            f"No inverse plot spec for benchmark {benchmark!r}; "
            f"known: {sorted(_SPECS)}"
        )


# ============================================================
# CSV LOADING + COERCION
# ============================================================

def _load_inverse_csv(csv_path: str | Path) -> dict[str, list[str]]:
    """Read an inversion summary CSV into a column dict of raw string cells.

    Missing cells are stored as ``""``. Numeric/boolean coercion is deferred to
    :func:`_floats` / :func:`_bools` so columns absent from a particular pass
    (e.g. no ``--mcmc``) degrade to all-NaN instead of raising.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"inverse CSV not found: {path}")
    with path.open(newline="") as f:
        reader = _csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    table: dict[str, list[str]] = {col: [] for col in fieldnames}
    for r in rows:
        for col in fieldnames:
            table[col].append(r.get(col, "") or "")
    table["_n"] = len(rows)  # type: ignore[assignment]
    return table


def _n_rows(table: dict) -> int:
    return int(table.get("_n", 0))


def _floats(table: dict, col: str) -> np.ndarray:
    """Column ``col`` as float64; NaN for blank cells or a missing column."""
    n = _n_rows(table)
    if col not in table:
        return np.full(n, np.nan, dtype=np.float64)
    out = np.empty(n, dtype=np.float64)
    for i, v in enumerate(table[col]):
        try:
            out[i] = float(v) if v not in ("", None) else np.nan
        except (TypeError, ValueError):
            out[i] = np.nan
    return out


def _bools(table: dict, col: str) -> np.ndarray:
    """Column ``col`` as float in {0,1}; NaN for blank/missing.

    ``csv.DictWriter`` serializes Python ``bool`` via ``str`` → ``"True"``/
    ``"False"``; both spellings (and 0/1) are accepted.
    """
    n = _n_rows(table)
    if col not in table:
        return np.full(n, np.nan, dtype=np.float64)
    out = np.full(n, np.nan, dtype=np.float64)
    for i, v in enumerate(table[col]):
        s = str(v).strip().lower()
        if s in ("true", "1", "1.0"):
            out[i] = 1.0
        elif s in ("false", "0", "0.0"):
            out[i] = 0.0
    return out


def _sim_ids(table: dict) -> np.ndarray:
    return _floats(table, "sim_id")


def _require_columns(table: dict, cols: list[str], fig: str) -> None:
    """Raise a clear error if any structural column is absent from the CSV."""
    missing = [c for c in cols if c not in table]
    if missing:
        raise ValueError(
            f"{fig}: CSV is missing required column(s) {missing}. "
            f"Present columns: {sorted(k for k in table if k != '_n')}"
        )


# ============================================================
# ARTIFACT (NPZ) LOADING
# ============================================================

def _load_artifacts(artifact_dir: str | Path | None) -> dict[int, dict]:
    """Load every ``sim_*.npz`` into ``{sim_id: {key: array}}`` (empty if None)."""
    if artifact_dir is None:
        return {}
    root = Path(artifact_dir)
    if not root.exists():
        print(f"  [inverse] artifact dir {root} not found; CSV-only fallback")
        return {}
    arts: dict[int, dict] = {}
    for p in sorted(root.glob("sim_*.npz")):
        with np.load(p, allow_pickle=True) as z:
            d = {k: z[k] for k in z.files}
        try:
            sid = int(np.asarray(d["sim_id"]).item())
        except (KeyError, ValueError):
            continue
        arts[sid] = d
    return arts


def _normalize_theta(theta: np.ndarray, spec: InversePlotSpec) -> np.ndarray:
    """Map physical theta columns to fraction-of-range [0,1] coordinates."""
    lo = np.array([spec.param_ranges[p][0] for p in spec.param_names], dtype=np.float64)
    hi = np.array([spec.param_ranges[p][1] for p in spec.param_names], dtype=np.float64)
    width = np.where((hi - lo) > 0, hi - lo, 1.0)
    return (np.asarray(theta, dtype=np.float64) - lo[None, :]) / width[None, :]


def _ridge_axis_limits(pts: np.ndarray, *, min_width: float = 0.12
                       ) -> tuple[tuple[float, float], tuple[float, float]]:
    """Padded [0,1]-clamped axis limits around a 2D ridge cloud.

    Robust to a collapsed (near-constant) coordinate: ``pad = max(0.15*span,
    0.03)`` keeps a span-0 axis visible, and a too-narrow clamped interval is
    expanded inward from whichever edge it abuts to ``min_width``.
    """
    lims: list[tuple[float, float]] = []
    for k in range(2):
        col = np.asarray(pts[:, k], dtype=np.float64)
        lo_d, hi_d = float(np.min(col)), float(np.max(col))
        span = hi_d - lo_d
        pad = max(0.15 * span, 0.03)
        lo = max(lo_d - pad, 0.0)
        hi = min(hi_d + pad, 1.0)
        if hi - lo < min_width:
            if hi >= 1.0:
                lo = max(0.0, hi - min_width)
            elif lo <= 0.0:
                hi = min(1.0, lo + min_width)
            else:
                mid = 0.5 * (lo + hi)
                lo = max(0.0, mid - 0.5 * min_width)
                hi = min(1.0, mid + 0.5 * min_width)
        lims.append((lo, hi))
    return lims[0], lims[1]


def _centered_direction_segment(center: np.ndarray, direction: np.ndarray,
                                pts: np.ndarray, xlim: tuple[float, float],
                                ylim: tuple[float, float], *,
                                default_len: float = 0.08):
    """Two-sided segment through ``center`` along (unsigned) ``direction``.

    Length scales with the cloud spread (max coordinate IQR, then range, then a
    fixed fallback) and is symmetrically shrunk so both endpoints stay inside the
    displayed limits — preserving centering and direction. Returns ``(p0, p1)``
    or ``None`` when the direction is degenerate or no room exists.
    """
    d = np.asarray(direction, dtype=np.float64)
    norm = float(np.hypot(d[0], d[1]))
    if not np.isfinite(norm) or norm == 0.0:
        return None
    d = d / norm
    c = np.asarray(center, dtype=np.float64)
    iqr = [float(np.subtract(*np.percentile(pts[:, k], [75, 25]))) for k in range(2)]
    rng = [float(np.ptp(pts[:, k])) for k in range(2)]
    L = next((v for v in (max(iqr), max(rng), default_len) if v > 0.0), default_len)
    for k, bound in enumerate((xlim, ylim)):
        dk = float(d[k])
        if abs(dk) <= 1e-12:
            continue
        lo, hi = bound
        room = min(hi - c[k], c[k] - lo)
        if room <= 0.0:
            return None
        L = min(L, room / abs(dk))
    if L <= 0.0:
        return None
    p0 = (float(c[0] - L * d[0]), float(c[1] - L * d[1]))
    p1 = (float(c[0] + L * d[0]), float(c[1] + L * d[1]))
    return p0, p1


# ============================================================
# SHARED PANEL HELPERS
# ============================================================

def _finite_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = np.isfinite(a) & np.isfinite(b)
    return a[m], b[m]


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average-rank transform (ties share the mean of their ranks)."""
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.size, dtype=np.float64)
    ranks[order] = np.arange(1, a.size + 1, dtype=np.float64)
    sa = a[order]
    i = 0
    n = a.size
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def _annotate_spearman(ax, x: np.ndarray, y: np.ndarray) -> None:
    """Corner Spearman-ρ box, only when it is well-defined (else nothing).

    Requires >=3 finite pairs and >=2 unique values on each axis; otherwise
    Spearman is undefined/constant-input and the annotation is omitted.
    """
    x, y = _finite_pair(np.asarray(x, float), np.asarray(y, float))
    if x.size < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return
    rho = float(np.corrcoef(_rankdata(x), _rankdata(y))[0, 1])
    if not np.isfinite(rho):
        return
    ax.text(0.04, 0.96, f"Spearman $\\rho$={rho:.2f}", transform=ax.transAxes,
            va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))


def _scatter_parity(ax, true: np.ndarray, hat: np.ndarray, rng: tuple[float, float],
                    name: str, color: str, *, emphasize: bool = False,
                    note: str | None = None) -> None:
    """Recovered-vs-true scatter with identity line + median |error| annotation."""
    t, h = _finite_pair(np.asarray(true, float), np.asarray(hat, float))
    size = 26 if emphasize else 16
    if t.size:
        ax.scatter(t, h, s=size, color=color, alpha=0.75, edgecolor="none",
                   zorder=3)
        lo = float(min(t.min(), h.min()))
        hi = float(max(t.max(), h.max()))
    else:
        lo, hi = rng
    span = hi - lo if hi > lo else 1.0
    lo -= 0.04 * span
    hi += 0.04 * span
    ax.plot([lo, hi], [lo, hi], color="0.25", linestyle="--", linewidth=1.2,
            zorder=2, label="identity")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    if t.size:
        med_abs = float(np.median(np.abs(h - t)))
        width = rng[1] - rng[0] if rng[1] > rng[0] else 1.0
        ax.text(0.04, 0.96,
                f"med|err|={med_abs:.3g}\n({100.0 * med_abs / width:.1f}% of range)",
                transform=ax.transAxes, va="top", ha="left", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))
    if note is not None:
        ax.text(0.96, 0.04, note, transform=ax.transAxes, va="bottom",
                ha="right", fontsize=8, style="italic", color="0.3",
                bbox=dict(boxstyle="round", fc="0.95", ec="0.8", alpha=0.85))
    ax.set_xlabel(f"true {name}")
    ax.set_ylabel(f"recovered {name}")
    ax.set_title(name + (" — lead deliverable" if emphasize else ""),
                 fontweight="bold" if emphasize else "normal")
    ax.grid(True)


def _rms_from_half_mse(values: np.ndarray) -> np.ndarray:
    """RMS sensor error from stored half-MSE (``0.5*mean((a-b)^2)``)."""
    v = np.asarray(values, dtype=np.float64)
    return np.sqrt(np.clip(2.0 * v, 0.0, None))


def _representative_sim(sim_ids: np.ndarray, lead_abserr: np.ndarray,
                        available: set[int] | None) -> int | None:
    """Sim nearest the median lead abs error; tie-break lowest sim id.

    Picks only from sims in ``available`` (those with artifacts) when given.
    """
    ids = np.asarray(sim_ids, dtype=np.float64)
    err = np.asarray(lead_abserr, dtype=np.float64)
    mask = np.isfinite(ids) & np.isfinite(err)
    if available is not None:
        mask &= np.array([int(s) in available if np.isfinite(s) else False
                          for s in ids], dtype=bool)
    idx = np.where(mask)[0]
    if idx.size == 0:
        return None
    med = float(np.median(err[idx]))
    best = sorted(idx, key=lambda i: (abs(err[i] - med), int(ids[i])))[0]
    return int(ids[best])


def _wilson_ci(k: float, n: float, z: float = 1.959963984540054
               ) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion.

    ``k`` successes out of ``n`` trials. ``k`` is clipped to ``[0, n]``.
    Returns ``(nan, nan)`` when ``n == 0`` (caller omits the error bar).
    """
    n = float(n)
    if n <= 0:
        return (float("nan"), float("nan"))
    k = float(min(max(k, 0.0), n))
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return (float(center - half), float(center + half))


def _inclusion_yerr(rate: float, n: int) -> np.ndarray | None:
    """Asymmetric Wilson-CI distances ``[[rate-lo], [hi-rate]]`` for one bar.

    Returns ``None`` when ``n == 0`` so the caller skips the error bar.
    """
    lo, hi = _wilson_ci(rate * n, n)
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return None
    return np.array([[max(rate - lo, 0.0)], [max(hi - rate, 0.0)]])


def _inclusion_bar(ax, covered: np.ndarray, label: str, color: str) -> None:
    """Single-series inclusion-rate bar with a 95% Wilson CI error bar."""
    vals = covered[np.isfinite(covered)]
    n = int(vals.size)
    if n == 0:
        ax.text(0.5, 0.5, f"{label}\nno UQ rows", transform=ax.transAxes,
                ha="center", va="center", fontsize=9)
        ax.set_axis_off()
        return
    rate = float(np.mean(vals))
    ax.bar([0], [rate], width=0.5, color=color, alpha=0.85)
    yerr = _inclusion_yerr(rate, n)
    if yerr is not None:
        ax.errorbar([0], [rate], yerr=yerr, fmt="none", ecolor="0.2",
                    elinewidth=1.3, capsize=4, zorder=4)
    ax.axhline(0.95, color="0.3", linestyle=":", linewidth=1.1,
               label="nominal 0.95")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([0])
    ax.set_xticklabels([label], fontsize=8)
    ax.set_ylabel("inclusion rate")
    ax.set_title(f"{label}\nrate={rate:.2f}  (n={n}, one case = {1.0 / n:.2f})",
                 fontsize=9)
    ax.legend(loc="lower right")
    ax.grid(True, axis="y")


def _inclusion_grouped(ax, series: list[tuple[str, np.ndarray]], color: str
                       ) -> None:
    """Grouped inclusion-rate bars (one per estimand) with Wilson CI error bars.

    ``series`` is a list of ``(label, covered_array)``. Empty series render an
    ``n=0`` placeholder bar (no error bar). Each bar is annotated ``rate`` + ``n``;
    a shared note states the empirical resolution ``one case = 1/n``.
    """
    positions = list(range(len(series)))
    rates: list[float] = []
    ns: list[int] = []
    for x, (label, covered) in zip(positions, series):
        vals = np.asarray(covered, float)
        vals = vals[np.isfinite(vals)]
        n = int(vals.size)
        rate = float(np.mean(vals)) if n else 0.0
        rates.append(rate)
        ns.append(n)
        ax.bar([x], [rate], width=0.55, color=color, alpha=0.85, zorder=3)
        yerr = _inclusion_yerr(rate, n) if n else None
        if yerr is not None:
            ax.errorbar([x], [rate], yerr=yerr, fmt="none", ecolor="0.2",
                        elinewidth=1.3, capsize=4, zorder=4)
        tag = f"rate={rate:.2f}\nn={n}" if n else "n=0"
        ax.text(x, min(rate + 0.04, 0.98), tag, ha="center", va="bottom",
                fontsize=8)
    ax.axhline(0.95, color="0.3", linestyle=":", linewidth=1.1,
               label="nominal 0.95")
    ax.set_ylim(0.0, 1.05)
    ax.set_xlim(-0.6, len(series) - 0.4)
    ax.set_xticks(positions)
    ax.set_xticklabels([label for label, _ in series], fontsize=8)
    ax.set_ylabel("inclusion rate")
    finite_ns = [n for n in ns if n > 0]
    note = (f"one case = 1/n (n={min(finite_ns)}–{max(finite_ns)})"
            if finite_ns else "no UQ rows")
    ax.text(0.02, 0.04, note, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8, color="0.35")
    ax.set_title("credible-interval inclusion (95% Wilson CI)", fontsize=10)
    ax.legend(loc="lower right", frameon=True, framealpha=0.85)
    ax.grid(True, axis="y")


# ============================================================
# FIG 1 — Parameter recovery parity
# ============================================================

def plot_parameter_recovery(csv_path, spec: InversePlotSpec, *,
                            save_path=None, name=None):
    """Per-parameter recovered-vs-true parity (+ emphasized lead for source_itr)."""
    name = name or f"{spec.benchmark}_parameter_recovery"
    table = _load_inverse_csv(csv_path)

    req = []
    for p in spec.param_names:
        req += [spec.hat_cols[p], spec.true_cols[p]]
    if spec.lead_name not in spec.param_names:
        req += [spec.lead_hat_col, spec.lead_true_col]
    _require_columns(table, req, "plot_parameter_recovery")

    title = f"{spec.benchmark}: parameter recovery"
    if spec.recovery_subtitle:
        title = f"{title}\n{spec.recovery_subtitle}"

    with plt.rc_context(PLOT_STYLE):
        if not spec.has_ridge:
            # Degenerate sibling: a single annotated lead panel.
            fig, ax = plt.subplots(figsize=(4.6, 4.4), constrained_layout=True)
            p = spec.param_names[0]
            _scatter_parity(
                ax, _floats(table, spec.true_cols[p]),
                _floats(table, spec.hat_cols[p]),
                spec.param_ranges[p], p, spec.color, emphasize=True,
            )
            ax.legend(loc="lower right")
            fig.suptitle(title)
        else:
            # Lead severity panel (tall, left) + the four parameter panels.
            fig = plt.figure(figsize=(11.0, 6.2), constrained_layout=True)
            gs = fig.add_gridspec(2, 3)
            ax_lead = fig.add_subplot(gs[:, 0])
            _scatter_parity(
                ax_lead, _floats(table, spec.lead_true_col),
                _floats(table, spec.lead_hat_col),
                _lead_range(table, spec), spec.lead_name, spec.color,
                emphasize=True,
            )
            ax_lead.legend(loc="lower right")
            cells = [(0, 1), (0, 2), (1, 1), (1, 2)]
            for p, (r, c) in zip(spec.param_names, cells):
                ax = fig.add_subplot(gs[r, c])
                tag = ("expected parameter trade-off"
                       if p in spec.non_identifiable_params else None)
                _scatter_parity(
                    ax, _floats(table, spec.true_cols[p]),
                    _floats(table, spec.hat_cols[p]),
                    spec.param_ranges[p], p, spec.color, note=tag,
                )
            fig.suptitle(title)
    return _save_figure(fig, save_path, "inverse", name, layout="constrained")


def _lead_range(table: dict, spec: InversePlotSpec) -> tuple[float, float]:
    """Range for the lead deliverable axis (params have fixed bounds)."""
    if spec.lead_name in spec.param_ranges:
        return spec.param_ranges[spec.lead_name]
    vals = np.concatenate([
        _floats(table, spec.lead_true_col), _floats(table, spec.lead_hat_col),
    ])
    vals = vals[np.isfinite(vals)]
    if vals.size:
        return float(vals.min()), float(vals.max())
    return (0.0, 1.0)


# ============================================================
# FIG 2 — Identifiability / sensitivity (local, honestly labelled)
# ============================================================

def plot_identifiability(csv_path, spec: InversePlotSpec, *,
                         artifact_dir=None, save_path=None, name=None):
    """Local identifiability geometry at theta_hat (+ posterior ridge when rich)."""
    name = name or f"{spec.benchmark}_identifiability"
    table = _load_inverse_csv(csv_path)

    if not spec.has_ridge:
        fig = _identifiability_scalar(table, spec)
    else:
        arts = _load_artifacts(artifact_dir)
        fig = _identifiability_multi(table, spec, arts)
    return _save_figure(fig, save_path, "inverse", name, layout="constrained")


def _identifiability_scalar(table: dict, spec: InversePlotSpec):
    """forcing (1 param): sensitivity, Fisher info, approx SE vs true R_c.

    Condition number is ~1 and uninformative for a scalar, so it is omitted.
    """
    p = spec.param_names[0]
    _require_columns(table, [spec.true_cols[p], spec.sens_cols[p]],
                     "plot_identifiability")
    true = _floats(table, spec.true_cols[p])
    sens = _floats(table, spec.sens_cols[p])          # ||J||_2 = sv_0
    sigma_eff2 = _floats(table, spec.sigma_eff2_col)  # may be all-NaN

    order = np.argsort(true)
    tv = true[order]
    sv = sens[order]
    s2 = sigma_eff2[order]
    have_sigma = np.any(np.isfinite(s2))

    with plt.rc_context(PLOT_STYLE):
        ncol = 3 if have_sigma else 1
        fig, axes = plt.subplots(1, ncol, figsize=(4.6 * ncol, 4.0),
                                 squeeze=False, constrained_layout=True)
        ax = axes[0, 0]
        ax.plot(tv, sv, "o-", color=spec.color, ms=4)
        ax.set_xlabel(f"true {p}")
        ax.set_ylabel(r"$\|J\|_2$ (sensitivity)")
        ax.set_title("scalar sensitivity at $\\hat{\\theta}$")
        ax.grid(True)
        if have_sigma:
            fisher = sv ** 2 / s2
            se = np.sqrt(s2) / np.where(sv > 0, sv, np.nan)
            ax2 = axes[0, 1]
            ax2.plot(tv, fisher, "s-", color=spec.color, ms=4)
            ax2.set_xlabel(f"true {p}")
            ax2.set_ylabel(r"$\|J\|^2 / \sigma_{\rm eff}^2$ (Fisher info)")
            ax2.set_title("Fisher information")
            ax2.grid(True)
            ax3 = axes[0, 2]
            ax3.plot(tv, se, "^-", color=spec.color, ms=4)
            ax3.set_xlabel(f"true {p}")
            ax3.set_ylabel(r"$\sqrt{\sigma_{\rm eff}^2}/\|J\|_2$ (approx SE)")
            ax3.set_title("approximate standard error")
            ax3.grid(True)
        fig.suptitle(
            f"{spec.benchmark}: local identifiability at $\\hat{{\\theta}}$")
    return fig


def _identifiability_multi(table: dict, spec: InversePlotSpec, arts: dict):
    """source_itr: normalized SVD spectrum, cond dist, least-id dir, MCMC ridge."""
    _require_columns(table, list(spec.sv_cols) + [spec.cond_col],
                     "plot_identifiability")
    cond = _floats(table, spec.cond_col)
    cond = cond[np.isfinite(cond)]

    # Normalized SVD from artifact Jacobians when available; else raw CSV svs.
    spectra, least_dirs = [], []
    for d in arts.values():
        if "observation_jacobian" not in d or "param_scales" not in d:
            continue
        J = np.asarray(d["observation_jacobian"], dtype=np.float64)
        scales = np.asarray(d["param_scales"], dtype=np.float64)
        Jn = J * scales[None, :]
        try:
            _, S, Vt = np.linalg.svd(Jn, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        spectra.append(S)
        least_dirs.append(Vt[-1])
    have_norm = len(spectra) > 0
    if not have_norm:
        print("  [inverse] no Jacobian artifacts; Fig 2 uses raw CSV "
              "singular values (no normalized SVD / MCMC ridge)")

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(10.0, 8.4),
                                 constrained_layout=True)

        # (a) singular-value spectrum.
        axa = axes[0, 0]
        if have_norm:
            Smat = np.vstack(spectra)
            k = Smat.shape[1]
            xs = np.arange(k)
            axa.plot(xs, np.median(Smat, axis=0), "o-", color=spec.color)
            axa.fill_between(xs, np.percentile(Smat, 10, axis=0),
                             np.percentile(Smat, 90, axis=0),
                             color=spec.color, alpha=0.2)
            axa.set_title("normalized singular-value spectrum")
        else:
            svm = np.vstack([_floats(table, c) for c in spec.sv_cols]).T
            svm = svm[np.all(np.isfinite(svm), axis=1)]
            xs = np.arange(len(spec.sv_cols))
            if svm.size:
                axa.plot(xs, np.median(svm, axis=0), "o-", color=spec.color)
            axa.set_title("raw singular-value spectrum (CSV)")
        axa.set_yscale("log")
        axa.set_xlabel("singular index")
        axa.set_ylabel("singular value")
        axa.grid(True, which="both")

        # (b) condition-number distribution across sims.
        axb = axes[0, 1]
        if cond.size:
            axb.hist(cond, bins=min(20, max(5, cond.size)), color=spec.color,
                     alpha=0.8)
            axb.axvline(float(np.median(cond)), color="0.2", linestyle="--",
                        label=f"median {np.median(cond):.1f}")
            axb.legend()
        axb.set_xlabel("condition number")
        axb.set_ylabel("count")
        axb.set_title("condition number across sims")
        axb.grid(True, axis="y")

        # (c) least-identified direction (mean |component| in normalized coords).
        axc = axes[1, 0]
        if have_norm:
            L = np.abs(np.vstack(least_dirs))
            comp = np.median(L, axis=0)
            axc.bar(np.arange(len(spec.param_names)), comp, color=spec.color,
                    alpha=0.85)
            axc.set_xticks(np.arange(len(spec.param_names)))
            axc.set_xticklabels(spec.param_names, rotation=20)
            axc.set_ylabel("|component| (normalized)")
            axc.set_title("least-identified direction")
        else:
            axc.text(0.5, 0.5, "needs Jacobian artifacts", transform=axc.transAxes,
                     ha="center", va="center")
            axc.set_axis_off()
        axc.grid(True, axis="y")

        # (d) (param_i, param_j) MCMC cloud (true nonlinear ridge) + local
        #     least-identified direction drawn as a centered two-sided segment.
        axd = axes[1, 1]
        i, j = spec.ridge_pair
        Sn_pair = None
        for d in arts.values():
            if "mcmc_theta_samples" not in d:
                continue
            S = np.asarray(d["mcmc_theta_samples"], dtype=np.float64)
            if S.ndim != 2 or S.shape[1] <= max(i, j):
                continue
            Sn = _normalize_theta(S, spec)
            Sn_pair = Sn[:, [i, j]]
            Sn_pair = Sn_pair[np.all(np.isfinite(Sn_pair), axis=1)]
            if Sn_pair.size:
                axd.scatter(Sn_pair[:, 0], Sn_pair[:, 1], s=4, color=spec.color,
                            alpha=0.12, edgecolor="none", zorder=2)
            break  # representative single-sim ridge keeps the panel legible
        if Sn_pair is not None and Sn_pair.shape[0] >= 1:
            xlim, ylim = _ridge_axis_limits(Sn_pair)
            axd.set_xlim(*xlim)
            axd.set_ylim(*ylim)
            axd.set_aspect("equal", adjustable="box")
            if have_norm:
                direction = np.median(np.vstack(least_dirs), axis=0)[[i, j]]
                center = np.median(Sn_pair, axis=0)
                seg = _centered_direction_segment(center, direction, Sn_pair,
                                                   xlim, ylim)
                if seg is not None:
                    (x0, y0), (x1, y1) = seg
                    axd.plot([x0, x1], [y0, y1], color="0.12", lw=1.8,
                             solid_capstyle="round", zorder=4,
                             label="least-identified direction")
                    axd.legend(loc="upper right", fontsize=8, frameon=True,
                               framealpha=0.85)
        else:
            axd.text(0.5, 0.5, "needs MCMC artifacts", transform=axd.transAxes,
                     ha="center", va="center")
        axd.set_xlabel(f"{spec.param_names[i]} (normalized)")
        axd.set_ylabel(f"{spec.param_names[j]} (normalized)")
        axd.set_title(
            f"({spec.param_names[i]}, {spec.param_names[j]}) posterior ridge\n"
            f"nonlinear ridge (MCMC) + local least-identified direction",
            fontsize=10)
        axd.grid(True)

        fig.suptitle(
            f"{spec.benchmark}: local identifiability geometry at "
            f"$\\hat{{\\theta}}$ + posterior ridge")
    return fig


# ============================================================
# FIG 3 — Surrogate fidelity at theta_hat
# ============================================================

def plot_surrogate_fidelity(csv_path, spec: InversePlotSpec, *,
                            save_path=None, name=None):
    """fno/fv/fno-vs-fv sensor RMS at theta_hat (forward consistency)."""
    name = name or f"{spec.benchmark}_surrogate_fidelity"
    table = _load_inverse_csv(csv_path)
    _require_columns(table, ["fno_resid", "fv_resid", "fno_vs_fv_resid"],
                     "plot_surrogate_fidelity")

    fno = _rms_from_half_mse(_floats(table, "fno_resid"))
    fv = _rms_from_half_mse(_floats(table, "fv_resid"))
    cfno = _rms_from_half_mse(_floats(table, "fno_vs_fv_resid"))
    lead_err = _floats(table, spec.lead_abserr_col)

    series = [("FNO vs data", fno), ("FV vs data", fv), ("C_FNO (FNO vs FV)", cfno)]

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4),
                                 constrained_layout=True)
        ax = axes[0]
        data = [s[np.isfinite(s)] for _, s in series]
        labels = [lbl for lbl, _ in series]
        positions = np.arange(len(series))
        bp = ax.boxplot(data, positions=positions, widths=0.55,
                        patch_artist=True, showfliers=False)
        for patch in bp["boxes"]:
            patch.set(facecolor=spec.color, alpha=0.5)
        for k, s in enumerate(data):
            if s.size:
                ax.scatter(np.full(s.size, positions[k]), s, s=10, color="0.25",
                           alpha=0.5, zorder=3)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=12)
        ax.set_ylabel("sensor RMS (normalized temp)")
        # Log scale only when every plotted finite RMS is strictly positive;
        # never substitute an epsilon for a true zero.
        all_finite = np.concatenate(data) if any(s.size for s in data) else np.array([])
        if all_finite.size and np.all(all_finite > 0.0):
            ax.set_yscale("log")
            ax.set_title("forward consistency at $\\hat{\\theta}$ (log scale)")
        else:
            ax.set_title("forward consistency at $\\hat{\\theta}$")
            if all_finite.size:
                ax.text(0.02, 0.98, "log scale unavailable (zero-valued RMS)",
                        transform=ax.transAxes, va="top", ha="left", fontsize=8,
                        color="0.35")
        ax.grid(True, axis="y")

        ax2 = axes[1]
        x, y = _finite_pair(cfno, lead_err)
        if x.size:
            ax2.scatter(x, y, s=20, color=spec.color, alpha=0.7, edgecolor="none")
        _annotate_spearman(ax2, x, y)
        ax2.set_xlabel("C_FNO RMS (surrogate vs FV)")
        ax2.set_ylabel(f"{spec.lead_name} abs error")
        ax2.set_title("surrogate error vs lead error")
        ax2.grid(True)

        fig.suptitle(
            f"{spec.benchmark}: surrogate fidelity "
            f"(RMS=$\\sqrt{{2\\cdot}}$half-MSE; C_FNO inflates UQ variance)")
    return _save_figure(fig, save_path, "inverse", name, layout="constrained")


# ============================================================
# FIG 4 — Uncertainty quantification (two frameworks, never overlaid)
# ============================================================

def plot_uncertainty(csv_path, spec: InversePlotSpec, *,
                     artifact_dir=None, save_path=None, name=None):
    """Profile (pinned param) + MCMC (lead deliverable) + inclusion-rate strip."""
    name = name or f"{spec.benchmark}_uncertainty"
    table = _load_inverse_csv(csv_path)
    arts = _load_artifacts(artifact_dir)

    lead_err = _floats(table, spec.lead_abserr_col)
    sim_ids = _sim_ids(table)
    rep = _representative_sim(
        sim_ids, lead_err, set(arts) if arts else None,
    )
    rep_art = arts.get(rep) if rep is not None else None

    prof_cov = _bools(table, spec.profile_covered_col)
    mcmc_cov = _bools(table, spec.mcmc_lead_covered_col)

    with plt.rc_context(PLOT_STYLE):
        fig = plt.figure(figsize=(11.0, 7.6), constrained_layout=True)
        gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1.0])

        ax_prof = fig.add_subplot(gs[0, 0])
        _panel_profile(ax_prof, rep_art, spec)

        ax_mc = fig.add_subplot(gs[0, 1])
        _panel_mcmc_posterior(ax_mc, rep_art, spec, table)

        ax_incl = fig.add_subplot(gs[1, :])
        if spec.has_ridge:
            # Two distinct estimands worth comparing: profile (a pinned param)
            # vs MCMC (the lead severity). Grouped bars + Wilson CI.
            _inclusion_grouped(
                ax_incl,
                [(f"profile — {spec.profile_param_name}", prof_cov),
                 (f"MCMC — {spec.lead_name}", mcmc_cov)],
                spec.color,
            )
        else:
            # Scalar benchmark: profile and MCMC both target the same lead, so a
            # single MCMC credible-interval inclusion category is shown.
            _inclusion_bar(
                ax_incl, mcmc_cov,
                f"MCMC credible-interval inclusion — {spec.lead_name}",
                spec.color,
            )

        rep_txt = f"representative sim {rep}" if rep is not None else "no artifacts"
        fig.suptitle(
            f"{spec.benchmark}: uncertainty quantification "
            f"(profile vs MCMC; {rep_txt})")
    return _save_figure(fig, save_path, "inverse", name, layout="constrained")


def _panel_profile(ax, rep_art, spec: InversePlotSpec) -> None:
    """Profile ΔNLL vs the pinned parameter with the Wilks threshold."""
    if rep_art is None or "profile_grid" not in rep_art:
        ax.text(0.5, 0.5, "no profile artifact", transform=ax.transAxes,
                ha="center", va="center")
        ax.set_axis_off()
        return
    grid = np.asarray(rep_art["profile_grid"], dtype=np.float64)
    nll = np.asarray(rep_art["profile_nll"], dtype=np.float64)
    nll_min = float(np.asarray(rep_art.get("profile_nll_min", np.nanmin(nll))).item())
    dnll = nll - nll_min
    thresh = float(np.asarray(rep_art.get("profile_threshold", np.nan)).item())
    ax.plot(grid, dnll, "o-", color=spec.color, ms=4)
    if np.isfinite(thresh):
        ax.axhline(thresh, color="0.3", linestyle="--", linewidth=1.1,
                   label=f"Wilks {thresh:.2f}")
        ax.legend(loc="upper center")
    pname = str(np.asarray(rep_art.get("profile_param_name",
                                       spec.profile_param_name)).item())
    ax.set_xlabel(f"pinned {pname}")
    ax.set_ylabel(r"$\Delta$NLL")
    ax.set_title(f"profile likelihood (pinned {pname})")
    ax.grid(True)


def _panel_mcmc_posterior(ax, rep_art, spec: InversePlotSpec, table: dict) -> None:
    """MCMC posterior of the lead deliverable with its credible interval."""
    samples = None
    if rep_art is not None:
        if spec.has_ridge and "mcmc_excess_int" in rep_art:
            samples = np.asarray(rep_art["mcmc_excess_int"], dtype=np.float64)
        elif (not spec.has_ridge) and "mcmc_theta_samples" in rep_art:
            S = np.asarray(rep_art["mcmc_theta_samples"], dtype=np.float64)
            if S.ndim == 2 and S.shape[1] >= 1:
                samples = S[:, 0]
    if samples is None or samples.size == 0:
        ax.text(0.5, 0.5, "no MCMC artifact", transform=ax.transAxes,
                ha="center", va="center")
        ax.set_axis_off()
        return
    samples = samples[np.isfinite(samples)]
    ax.hist(samples, bins=min(40, max(10, samples.size // 20)), color=spec.color,
            alpha=0.7, density=True)
    lo, hi = np.percentile(samples, [2.5, 97.5])
    ax.axvline(lo, color="0.2", linestyle="--", linewidth=1.1)
    ax.axvline(hi, color="0.2", linestyle="--", linewidth=1.1,
               label=f"95% CI [{lo:.3g}, {hi:.3g}]")
    # source_itr: annotate the profile-path severity range (a path projection).
    if spec.profile_excess_low_col and rep_art is not None:
        plo = rep_art.get("profile_excess_int")
        if plo is not None:
            pe = np.asarray(plo, dtype=np.float64)
            pe = pe[np.isfinite(pe)]
            if pe.size:
                ax.axvspan(float(pe.min()), float(pe.max()), color="0.6",
                           alpha=0.15, label="profile-path severity range")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_xlabel(spec.lead_name)
    ax.set_ylabel("posterior density")
    ax.set_title(f"MCMC posterior — {spec.lead_name}")
    ax.grid(True)


# ============================================================
# CSV MERGE (two-pass optimization path)
# ============================================================

def merge_uq_csv(master_path, uq_path):
    """Merge a UQ-subset CSV into a master CSV by ``sim_id``.

    The master (cheap pass over all sims) carries one row per sim; the UQ subset
    (rich ``--profile/--mcmc`` pass over fewer sims) is written to a *distinct*
    file because ``invert.py`` overwrites ``--out-csv``. This copies only the
    UQ-related columns (``profile_*``, ``mcmc_*``, ``uq_*``, ``excess_*``) onto
    matching ``sim_id`` rows, leaving missing UQ values as ``NaN``.

    Validates that ``benchmark`` (when present in both) and every shared
    ground-truth ``*_true`` column agree on shared ``sim_id``s, and rejects
    duplicate ``sim_id`` within either file. Returns a ``pandas.DataFrame`` and
    prints ``n_total`` / ``n_UQ``.
    """
    import pandas as pd

    master = pd.read_csv(master_path)
    uq = pd.read_csv(uq_path)

    for label, df in (("master", master), ("uq", uq)):
        if "sim_id" not in df.columns:
            raise ValueError(f"{label} CSV has no 'sim_id' column")
        dup = df["sim_id"][df["sim_id"].duplicated()].unique()
        if len(dup):
            raise ValueError(f"duplicate sim_id in {label} CSV: {sorted(dup)}")

    master = master.set_index("sim_id")
    uq = uq.set_index("sim_id")
    shared = master.index.intersection(uq.index)

    if "benchmark" in master.columns and "benchmark" in uq.columns and len(shared):
        bm = master.loc[shared, "benchmark"].astype(str)
        bu = uq.loc[shared, "benchmark"].astype(str)
        if not (bm.values == bu.values).all():
            raise ValueError("benchmark column disagrees between master and uq CSVs")

    true_cols = [c for c in uq.columns if c.endswith("_true") and c in master.columns]
    for c in true_cols:
        if not len(shared):
            break
        a = pd.to_numeric(master.loc[shared, c], errors="coerce")
        b = pd.to_numeric(uq.loc[shared, c], errors="coerce")
        both = a.notna() & b.notna()
        if both.any() and not np.allclose(a[both].values, b[both].values,
                                          rtol=1e-5, atol=1e-8):
            raise ValueError(f"ground-truth column {c!r} disagrees on shared sim_id")

    uq_prefixes = ("profile_", "mcmc_", "uq_", "excess_")
    uq_cols = [c for c in uq.columns
               if c.startswith(uq_prefixes) or c == "profile_param"]
    merged = master.copy()
    for c in uq_cols:
        merged[c] = uq[c].reindex(merged.index)

    merged = merged.reset_index()
    n_total = int(len(merged))
    if uq_cols:
        n_uq = int(merged[uq_cols].notna().any(axis=1).sum())
    else:
        n_uq = 0
    print(f"  [inverse] merge_uq_csv: n_total={n_total}  n_UQ={n_uq}")
    return merged
