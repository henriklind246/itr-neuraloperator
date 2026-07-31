"""Out-of-distribution figures: F19, F20.

Both read ``ood_per_sim.csv``, which already carries the correct per-simulation
reduction -- ``scripts/inspect_ood.py`` is where the three-level rollup in
``visual/pub/stats.py`` came from. It is therefore handed to
:func:`visual.pub.stats.as_sim_frame` rather than to ``per_sim``: reducing an
already-reduced table would group singletons and quietly move the replication
unit back down to the pair.

The remaining weakness is the sample, not the aggregation. Every condition in
this workspace has one model seed and four simulations, so no confidence
interval is estimable and every summary carries ``TOO_FEW_SIMS``. The figures
render and say so, rather than withholding a result whose effects span one to
two orders of magnitude.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from visual.pub import panels, records, stats, style
from visual.pub._blocked import blocked

_METRICS = ("rmse_K", "rel_l2_pct", "iface_rmse_K")

# Ordered so the reader meets the naive protocol first and the two controlled
# ones are read against it.
_PROTOCOL_ORDER = ("fixed_initial", "anchored_from_horizon", "ood_local_fixed_lead")

_PROTOCOL_LABELS = {
    "fixed_initial": "fixed initial state",
    "anchored_from_horizon": "anchored at horizon",
    "ood_local_fixed_lead": "fixed lead, shifted window",
}

_PROTOCOL_COLORS = {
    "fixed_initial": "#D55E00",
    "anchored_from_horizon": "#0072B2",
    "ood_local_fixed_lead": "#009E73",
}

_AXIS_TITLES = {
    "interface_x": "interface position $x_I$",
    "rc": "contact resistance $R_c$",
    "family_transfer": "forcing family",
}

_IN_DISTRIBUTION = "in_distribution"


def _table(source, requirement, key: str, required):
    """The single OOD per-simulation CSV a figure reads, or a blocked figure."""
    paths = records.artifact_paths(source, "ood_per_sim")
    if not paths:
        blocked(requirement,
                "needs an ood_per_sim.csv from scripts/run_ood_suite.py.",
                key=key)
    df = records.load_csv_table(paths[0], required_columns=tuple(required))
    return df, paths[0]


def _series(df, group_column: str, order, *, metric: str, x_column: str):
    """One ordered summary series per group, keyed by a numeric coordinate.

    The stratum is always ``ood_value`` -- the condition the suite actually ran.
    ``x_column`` supplies only the coordinate each condition is drawn at, which
    is what lets the same conditions be replotted against a different axis.
    Stratifying on the coordinate instead would merge conditions that happen to
    share one, and the fixed-lead protocol shares its lead across every target
    time by construction.

    Returns ``{group: (x, [Summary, ...])}``. Groups absent from the table are
    dropped rather than drawn empty, so a legend never carries an entry with no
    data behind it.
    """
    out = {}
    for name in order:
        part = df[df[group_column].astype(str) == str(name)]
        if part.empty:
            continue
        unique = part.drop_duplicates("ood_value")
        coord = dict(zip(unique["ood_value"].astype(str),
                         unique[x_column].astype(float)))
        frame = stats.as_sim_frame(part, metrics=_METRICS, strata=("ood_value",))
        summaries = stats.summarize(frame, metric, strata=("ood_value",))
        # summarize keys are (seed, benchmark, ood_value).
        points = sorted(((coord[str(k[-1])], s) for k, s in summaries.items()),
                        key=lambda kv: kv[0])
        out[name] = (np.array([p[0] for p in points], dtype=float),
                     [p[1] for p in points])
    return out


def _in_distribution_level(df, metric: str) -> float:
    """Worst in-distribution control median, used as the reference line.

    Each shift axis carries its own control, and two axes can label theirs with
    the same ``ood_value`` -- ``interface_x = 0.5`` and ``R_c = 0.5`` are both
    "0.5" while being separate runs over the same simulations. The stratum is
    therefore the axis-value pair. The worst of the controls is taken so the
    line never flatters a shifted condition.
    """
    part = df[df["distribution_class"].astype(str) == _IN_DISTRIBUTION]
    if part.empty:
        return float("nan")
    strata = tuple(c for c in ("ood_axis", "ood_value") if c in part.columns)
    frame = stats.as_sim_frame(part, metrics=_METRICS, strata=strata)
    summaries = stats.summarize(frame, metric, strata=strata)
    return max(s.median for s in summaries.values())


def _codes(summaries) -> list[str]:
    return sorted({d.code for s in summaries for d in s.degradations})


def time_protocols(*, source=None, spec=None, requirement=None):
    """F19 -- temporal extrapolation protocols.

    Three protocols push the same model past the same training horizon while
    varying what is held fixed:

    ``fixed_initial``
        source snapshot pinned at ``t = 0``, so the lead grows with the target
        time. This is the naive reading of "extrapolate further".
    ``anchored_from_horizon``
        source pinned at the horizon, so the lead grows from a state the model
        has seen.
    ``ood_local_fixed_lead``
        the whole window slides past the horizon at constant lead.

    Panel (a) plots each against absolute target time and panel (b) against
    lead. The pair is the argument: the protocols separate sharply in (a) and
    very nearly collapse in (b), which identifies lead rather than absolute time
    as the quantity error is a function of. Showing only (a) would license the
    much stronger and less accurate claim that the surrogate fails beyond its
    training horizon as such.
    """
    key = "F19_ood_time_protocols"
    df, path = _table(
        source, requirement, key,
        ("sim_id", "protocol", "target_time_actual", "lead_time_actual",
         "rel_l2_pct", "iface_rmse_K", "distribution_class"))

    order = [p for p in _PROTOCOL_ORDER
             if (df["protocol"].astype(str) == p).any()]
    if not order:
        blocked(requirement, "the OOD table carries no known protocol values.",
                key=key)

    by_target = _series(df, "protocol", order,
                        metric="rel_l2_pct", x_column="target_time_actual")
    by_lead = _series(df, "protocol", order,
                      metric="rel_l2_pct", x_column="lead_time_actual")
    iface_by_lead = _series(df, "protocol", order,
                            metric="iface_rmse_K", x_column="lead_time_actual")

    horizon = float("nan")
    if "time_norm_horizon" in df.columns:
        horizon = float(np.nanmax(np.asarray(df["time_norm_horizon"],
                                             dtype=float)))
    reference = _in_distribution_level(df, "rel_l2_pct")

    width = spec.width if spec is not None else "two_col"
    fig, axes = plt.subplots(1, 3, figsize=style.figsize(width, rows=1,
                                                         row_height="std"))

    layout = (
        (axes[0], by_target, style.axis_label("rel_l2_pct"),
         "target time $t_j$", "separated by absolute time"),
        (axes[1], by_lead, style.axis_label("rel_l2_pct"),
         "lead time $t_j - t_s$", "collapsed by lead"),
        (axes[2], iface_by_lead, style.axis_label("iface_rmse_K"),
         "lead time $t_j - t_s$", "interface region"),
    )
    for ax, series, ylabel, xlabel, title in layout:
        for name, (x, summaries) in series.items():
            panels.summary_points(
                ax, summaries, x=x, connect=True,
                color=_PROTOCOL_COLORS.get(name, "0.3"),
                label=_PROTOCOL_LABELS.get(name, name),
                ylabel=ylabel, xlabel=xlabel, title=title, logy=True)
        ax.set_xlim(left=0.0)

    if np.isfinite(horizon):
        style.add_reference_line(axes[0], horizon, label="training horizon",
                                 axis="x")
    if np.isfinite(reference):
        style.add_reference_line(axes[0], reference, label="in-distribution")
        style.add_reference_line(axes[1], reference, label="in-distribution")

    # Lower left: the panel carries no data below the training horizon, and the
    # rotated horizon label occupies the top of that same corner.
    axes[0].legend(fontsize=5.0, loc="lower left")
    style.panel_letters(axes)
    fig.tight_layout()

    every = []
    for series in (by_target, by_lead, iface_by_lead):
        for _, summaries in series.values():
            every.extend(summaries)

    per_condition = (df.groupby(["protocol", "ood_value"], dropna=False)["sim_id"]
                     .nunique())
    metric_definition = {
        "source": str(path),
        "aggregation": "per-simulation rows as written by run_ood_suite.py; "
                       "median across simulations within protocol and time",
        "global": "rel_l2_pct = 100 * sqrt(sum sse_K2 / sum target_sse_K2)",
        "interface": "iface_rmse_K = sqrt(sum interface_sse_K2 / "
                     "sum num_interface_cells)",
        "protocols": {p: _PROTOCOL_LABELS[p] for p in order},
        "training_horizon": horizon,
        "in_distribution_rel_l2_pct": reference,
        "n_simulations_per_condition": int(per_condition.max()),
        "n_simulations": records.count_simulations(df),
        "degradations": _codes(every),
    }
    return fig, None, metric_definition


def transfer(*, source=None, spec=None, requirement=None):
    """F20 -- parameter and family transfer.

    One panel per shift axis, never pooled into a single "OOD" number: the axes
    fail by different mechanisms and by different orders of magnitude, and an
    average over them would describe none of them.

    Each panel carries a horizontal line at the in-distribution median, so every
    point is read as a multiple of the model's own baseline skill rather than
    against a number remembered from another figure. That line is the
    degradation display the design called for; with a single model seed the
    paired-seed delta it originally specified has nothing to pair.
    """
    key = "F20_ood_transfer"
    df, path = _table(
        source, requirement, key,
        ("sim_id", "ood_axis", "ood_value", "rel_l2_pct", "distribution_class"))

    present = [a for a in ("interface_x", "rc", "family_transfer")
               if (df["ood_axis"].astype(str) == a).any()]
    present += [a for a in sorted(set(df["ood_axis"].astype(str)))
                if a not in present]
    if not present:
        blocked(requirement, "the OOD table carries no ood_axis values.", key=key)

    reference = _in_distribution_level(df, "rel_l2_pct")
    benchmark = str(df["benchmark"].iloc[0]) if "benchmark" in df.columns \
        else "interfaces"
    color = style.benchmark_color(benchmark)

    width = spec.width if spec is not None else "two_col"
    fig, ax_row = plt.subplots(1, len(present),
                               figsize=style.figsize(width, rows=1,
                                                     row_height="std"))
    flat = np.atleast_1d(ax_row).ravel().tolist()

    every = []
    for ax, axis_name in zip(flat, present):
        part = df[df["ood_axis"].astype(str) == axis_name]
        unique = part.drop_duplicates("ood_value")
        names = unique["ood_value"].astype(str)
        classes = dict(zip(names, unique["distribution_class"].astype(str)))
        numeric = ("ood_value_num" in unique.columns
                   and unique["ood_value_num"].notna().all())
        coord = dict(zip(names, unique["ood_value_num"].astype(float))) \
            if numeric else {}

        frame = stats.as_sim_frame(part, metrics=_METRICS, strata=("ood_value",))
        summaries = stats.summarize(frame, "rel_l2_pct", strata=("ood_value",))
        items = [(str(k[-1]), s) for k, s in summaries.items()]
        if numeric:
            items.sort(key=lambda kv: coord[kv[0]])
            x = np.array([coord[name] for name, _ in items], dtype=float)
            labels = None
        else:
            # No numeric coordinate to honour, so the control goes first and the
            # rest ascend by effect size. Alphabetical order would interleave the
            # two shift mechanisms and hide that one costs an order of magnitude
            # more than the other.
            items.sort(key=lambda kv: (classes.get(kv[0], "") != _IN_DISTRIBUTION,
                                       kv[1].median, kv[0]))
            x = None
            labels = [name for name, _ in items]
        ordered = [s for _, s in items]
        every.extend(ordered)

        positions = panels.summary_points(
            ax, ordered, x=x, labels=labels, connect=bool(numeric), color=color,
            ylabel=style.axis_label("rel_l2_pct"),
            xlabel=_AXIS_TITLES.get(axis_name, axis_name),
            logy=True, rotate_xticks=0.0 if numeric else 30.0)

        # Which point is the control is the first thing the reader needs, so it
        # is marked on the curve rather than deferred to a legend.
        for xi, (name, summary) in zip(positions, items):
            kind = classes.get(name, "")
            if kind == _IN_DISTRIBUTION:
                ax.plot([xi], [summary.median], "s", color=color, markersize=7.0,
                        markerfacecolor="none", markeredgewidth=1.1, zorder=6)
            elif kind == "limiting_case":
                ax.plot([xi], [summary.median], "^", color="0.25", markersize=4.5,
                        markerfacecolor="none", markeredgewidth=0.9, zorder=6)

        if np.isfinite(reference):
            style.add_reference_line(ax, reference, label="in-distribution")

    style.panel_letters(flat)
    fig.tight_layout()

    per_condition = (df.groupby(["ood_axis", "ood_value"], dropna=False)["sim_id"]
                     .nunique())
    metric_definition = {
        "source": str(path),
        "aggregation": "per-simulation rows as written by run_ood_suite.py; "
                       "median across simulations within each shift value",
        "global": "rel_l2_pct = 100 * sqrt(sum sse_K2 / sum target_sse_K2)",
        "axes": present,
        "in_distribution_rel_l2_pct": reference,
        "markers": {"open square": "in-distribution control",
                    "open triangle": "limiting case"},
        "n_simulations_per_condition": int(per_condition.max()),
        "n_simulations": records.count_simulations(df),
        "degradations": _codes(every),
    }
    return fig, None, metric_definition


__all__ = ["time_protocols", "transfer"]
