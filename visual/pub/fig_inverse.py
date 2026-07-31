"""Inverse-problem figures: F16, F17, F18, F22.

These read an artifact family independent of the forward benchmarks -- inverse
result CSVs rather than ``test_records.csv`` -- so they are not blocked by the
forward eval. They are, however, thin: every surviving inverse CSV in this
workspace has ten cases at a single sensor count and a single noise
realization. Ten is below the twenty-case floor the statistics layer needs for
an interval, so every summary here carries ``TOO_FEW_SIMS`` and is drawn as a
point estimate. The figures are published as case listings, not as
distributions.

F18 is blocked on an experiment that has never been run at all: its abscissa,
sensor count, does not vary in any artifact here.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from visual.pub import panels, records, stats, style
from visual.pub._blocked import blocked

_RECOVERY_KEYS = {
    "forcing": "F16_inverse_forcing_recovery",
    "forcing_itr": "F17_inverse_forcing_itr",
}

# Recovered quantities per inverse benchmark, in the order they are drawn:
# (column stem, axis label). The stem indexes ``<stem>_true`` and the estimate,
# whose suffix differs between the two CSVs -- ``_map`` for the scalar problem,
# ``_hat`` for the profile one.
_FORCING_PARAMS = (("R_c", r"$R_c$"),)

_FORCING_ITR_PARAMS = (
    ("R_base", r"$R_{\mathrm{base}}$"),
    ("R_amp", r"$R_{\mathrm{amp}}$"),
    ("y0", r"$y_0$"),
    ("sigma", r"$\sigma$"),
    ("excess_int", r"$\int (R_c - R_{\mathrm{base}})\,\mathrm{d}y$"),
)

# Components of the least-determined direction of the local sensitivity, i.e.
# the combination of parameters the boundary data constrains worst.
_ITR_DIRECTION = (
    ("least_dir_R_base", r"$R_{\mathrm{base}}$"),
    ("least_dir_R_amp", r"$R_{\mathrm{amp}}$"),
    ("least_dir_y0", r"$y_0$"),
    ("least_dir_sigma", r"$\sigma$"),
)

_DIRECTION_COLORS = {
    r"$R_{\mathrm{base}}$": "#0072B2",
    r"$R_{\mathrm{amp}}$": "#D55E00",
    r"$y_0$": "#009E73",
    r"$\sigma$": "#CC79A7",
}

_NOMINAL_COVERAGE = 0.95


def _table(source, requirement, key: str, detail: str, required):
    """The single inverse CSV a figure reads, or a blocked figure."""
    paths = records.artifact_paths(source, "inverse_csv")
    if not paths:
        blocked(requirement, detail, key=key)
    df = records.load_csv_table(paths[0], required_columns=tuple(required))
    return df, paths[0]


def _estimate_suffix(df, stem: str) -> str:
    for suffix in ("_map", "_hat"):
        if f"{stem}{suffix}" in df.columns:
            return suffix
    raise KeyError(f"no estimate column for {stem!r} in the inverse table")


def _coverage(df, columns):
    """Interval coverage per method, dropping methods the table does not carry."""
    out = {}
    for label, column in columns.items():
        if column in df.columns:
            out[label] = stats.proportion(df[column])
    return out


def _too_few(n: int) -> list[str]:
    return ["TOO_FEW_SIMS"] if n < stats.MIN_SIMS_FOR_CI else []


def recovery(*, benchmark: str, source=None, spec=None, requirement=None):
    """F16 / F17 -- parameter recovery, one benchmark per figure.

    Each recovered quantity gets a scatter against its true value with the
    identity line, MAP and FV-polished estimates both shown so the reader can
    see how much of the error the surrogate contributes. Nothing is filtered on
    outcome: a case that inverted badly is a point far from the diagonal, not a
    dropped row.

    ``forcing`` recovers a scalar :math:`R_c`, so the figure is one scatter, the
    error distribution, and interval coverage.

    ``forcing_itr`` recovers a Gaussian void profile, four parameters plus the
    integrated excess resistance. Its last panel is the identifiability
    statement: the squared components of the least-determined direction of the
    local sensitivity, one bar per case, ordered by condition number. When a bar
    is a single colour the boundary data does not separate that parameter at
    all.

    The second inverse benchmark is ``forcing_itr``, not ``source_itr``; the
    existing ``source_itr`` inverse figures are retired.
    """
    key = _RECOVERY_KEYS.get(benchmark, f"recovery[{benchmark}]")
    params = _FORCING_ITR_PARAMS if benchmark == "forcing_itr" else _FORCING_PARAMS
    df, path = _table(
        source, requirement, key,
        f"needs an inverse result CSV for {benchmark}; none resolved from the "
        "manifest.",
        ("sim_id", f"{params[0][0]}_true"))

    suffix = _estimate_suffix(df, params[0][0])
    n_cases = records.count_simulations(df)
    width = spec.width if spec is not None else "two_col"

    n_panels = len(params) + (2 if benchmark != "forcing_itr" else 1)
    ncols = min(3, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=style.figsize(width, rows=nrows,
                                                   row_height="std"))
    flat = np.atleast_1d(axes).ravel().tolist()
    color = style.benchmark_color(benchmark)

    for ax, (stem, label) in zip(flat, params):
        polish = df.get(f"{stem}_fvpolish")
        panels.recovery_scatter(
            ax, df[f"{stem}_true"], df[f"{stem}{suffix}"],
            polish=polish, color=color,
            xlabel=f"true {label}", ylabel=f"recovered {label}")
    flat[0].legend(fontsize=5.5, loc="lower right")

    used = len(params)

    if benchmark == "forcing_itr":
        order = np.argsort(np.asarray(df["cond_number"], dtype=float))
        weights = {}
        for column, label in _ITR_DIRECTION:
            if column not in df.columns:
                continue
            v = np.asarray(df[column], dtype=float)[order]
            weights[label] = v * v
        panels.component_bars(
            flat[used], weights, colors=_DIRECTION_COLORS,
            xticklabels=[f"{c:.0f}" for c in
                         np.asarray(df["cond_number"], dtype=float)[order]],
            xlabel="condition number", ylabel="squared component",
            title="least-determined direction")
        used += 1
    else:
        stem = params[0][0]
        err = np.abs(np.asarray(df[f"{stem}_true"], dtype=float)
                     - np.asarray(df[f"{stem}{suffix}"], dtype=float))
        panels.ecdf_curves(flat[used], {"MAP": err}, xlabel=f"|error| in {label}",
                           ylabel="fraction of cases",
                           title="recovery error")
        used += 1

        rates = _coverage(df, {"MCMC 95%": f"mcmc_{stem}_covered",
                               "profile 95%": f"profile_{stem}_covered"})
        if rates:
            panels.rate_bars(flat[used], rates, ylabel="coverage",
                             title="interval coverage")
            flat[used].axhline(_NOMINAL_COVERAGE, color="0.4", linestyle="--",
                               linewidth=0.8, zorder=1)
            flat[used].set_ylim(0.0, 1.05)
        used += 1

    for ax in flat[used:]:
        ax.set_visible(False)

    style.panel_letters(flat[:used])
    fig.tight_layout()

    metric_definition = {
        "benchmark": benchmark,
        "table": str(path),
        "replication_unit": "inversion case (sim_id)",
        "n_cases": n_cases,
        "estimator": "MAP" if suffix == "_map" else "MAP (profile parameterization)",
        "polish": "FV-polished estimate overlaid where the table carries it",
        "parameters": [p[0] for p in params],
        "failed_cases": "no case is dropped; every inversion in the table is drawn",
        "coverage": "Wilson 95% interval on the fraction of cases whose stated "
                    "95% interval contains the truth"
                    if benchmark != "forcing_itr" else None,
        "identifiability": (
            "squared components of the unit least-determined direction of the "
            "local sensitivity, cases ordered by condition number"
            if benchmark == "forcing_itr" else None),
        "degradations": _too_few(n_cases),
    }
    return fig, None, metric_definition


def sensor_count(*, source=None, spec=None, requirement=None):
    """F18 -- inversion accuracy against sensor count.

    Entirely new, and the one piece of evidence the audit calls essential that
    no artifact in this repo speaks to: every inverse run here used the default
    ``sensor_n_y = 8``.

    Design, once the sweep exists:

    * Sensor counts 4, 8, 16, 32, 64 on a log-2 axis, with **identical
      simulations, noise realizations and initialization seeds paired across
      counts**. Pairing is what makes the curve a statement about sensor count
      rather than about which cases each arm happened to draw;
      ``stats.paired_seed_delta`` does the matching and the bootstrap resamples
      the case, not the arm.
    * Median with IQR of scalar ``R_c`` recovery error for ``forcing`` and of
      integrated ITR severity plus profile error for ``forcing_itr``.
    * Optionally a success-rate panel with a Wilson interval and a wall-clock
      panel, since the practical question is where the accuracy gain stops
      paying for the instrumentation.

    ``scripts/invert.py`` already exposes the knob (``resolve_sensor_y``,
    ``build_interface_sensor_mask``); what is missing is a sweep driver that
    holds the case set fixed across counts and writes one CSV with an
    ``n_sensors`` column that actually varies.
    """
    blocked(requirement,
            "needs a paired multi-sensor-count inverse sweep; results_inverse."
            "csv has n_sensors = 8 for all 10 of its rows, so the abscissa of "
            "this figure does not exist yet.",
            key="F18_sensor_count")


def surrogate_fidelity(*, source=None, spec=None, requirement=None):
    """F22 -- surrogate fidelity against recovery error (supplemental).

    Two panels, both against the scalar recovery error: the forward FNO-FV
    mismatch at the recovered parameter, and the data-fit residual the
    optimizer actually minimized. The question is whether inversion failures
    are explained by surrogate error or by genuine ill-posedness. A weak rank
    correlation says a better surrogate would not fix them, which makes the
    identifiability panel of F17 the real story.

    Rank rather than linear correlation, and no fitted line: with ten cases a
    regression line would assert more structure than the data carries.
    """
    df, path = _table(
        source, requirement, "F22_surrogate_fidelity",
        "needs an inverse result CSV carrying per-case forward mismatch "
        "alongside recovery error.",
        ("sim_id", "R_c_abs_error"))

    series = [c for c in ("fno_vs_fv_resid", "fno_resid") if c in df.columns]
    if not series:
        blocked(requirement,
                "the inverse table carries no forward-mismatch column "
                "(fno_vs_fv_resid, fno_resid), so there is nothing to correlate "
                "recovery error against.",
                key="F22_surrogate_fidelity")

    titles = {
        "fno_vs_fv_resid": "surrogate mismatch",
        "fno_resid": "data-fit residual",
    }
    labels = {
        "fno_vs_fv_resid": r"$\|u_{\mathrm{FNO}} - u_{\mathrm{FV}}\|$ at MAP",
        "fno_resid": "FNO data residual at MAP",
    }

    width = spec.width if spec is not None else "one_half_col"
    fig, axes = plt.subplots(1, len(series),
                             figsize=style.figsize(width, row_height="std"))
    flat = np.atleast_1d(axes).ravel().tolist()

    err = df["R_c_abs_error"]
    correlations = {}
    for ax, column in zip(flat, series):
        rho = stats.spearman(df[column], err)
        correlations[column] = rho
        panels.association_scatter(
            ax, df[column], err, correlation=rho,
            color=style.benchmark_color("forcing"),
            xlabel=labels[column], ylabel=r"$|R_c$ recovery error$|$",
            title=titles[column])

    style.panel_letters(flat)
    fig.tight_layout()

    n_cases = records.count_simulations(df)
    metric_definition = {
        "table": str(path),
        "replication_unit": "inversion case (sim_id)",
        "n_cases": n_cases,
        "association": "Spearman rank correlation, percentile bootstrap over "
                       "cases",
        "no_fit_line": "monotone association only; no functional form is "
                       "asserted at this case count",
        "rho": {c: float(r.rho) for c, r in correlations.items()},
        "degradations": sorted(
            set(_too_few(n_cases))
            | {d.code for r in correlations.values() for d in r.degradations}),
    }
    return fig, None, metric_definition


__all__ = ["recovery", "sensor_count", "surrogate_fidelity"]
