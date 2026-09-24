"""Per-benchmark difficulty and case figures: F08-F15.

Two parametrized functions cover eight figures. ``difficulty(benchmark)`` is
F08 / F10 / F12 / F14 and ``cases(benchmark)`` is F09 / F11 / F13 / F15; the
benchmark comes from ``FigureSpec.params`` so the panel layout is chosen from
the benchmark's own ``ProblemSpec.val_pair_fields`` rather than by branching on
a name inside each figure.

All eight are blocked on schema-v2 ``test_records.csv`` for their benchmark.
The ``cases`` family additionally needs the benchmark's checkpoint, because the
prediction panels are evaluated rather than read from a stored field.
"""

from __future__ import annotations

import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from visual.pub import fields, jump, panels, records, select, stats, style
from visual.pub._blocked import blocked

# Which strata each benchmark's difficulty panels use. Every name here must
# exist in ``stats.STRATUM_SPECS`` and be available for the benchmark under
# ``ProblemSpec.val_pair_fields``; the difficulty figure is exactly a grid over
# these, so the table is the figure's specification.
DIFFICULTY_STRATA = {
    "forcing": ("lead_bin", "temporal_family", "spatial_family"),
    "source": ("lead_bin", "regime", "patch_x_bin", "amplitude_bin", "R_c_bin"),
    "source_itr_sin": ("lead_bin", "itr_amplitude", "R_c_bin"),
    "interfaces": ("lead_bin", "interface_x_bin", "R_c_bin"),
}

# Source uses its fifth panel for global physical error across contact resistance
# instead of repeating the normalized per-simulation error distribution.
DIFFICULTY_ECDF_BENCHMARKS = frozenset({"forcing", "source_itr_sin", "interfaces"})

_DIFFICULTY_KEYS = {
    "forcing": "F08_forcing_difficulty",
    "source": "F10_source_difficulty",
    "source_itr_sin": "F12_source_itr_sin_resistance",
    "interfaces": "F14_interfaces_difficulty",
}

_CASE_KEYS = {
    "forcing": "F09_forcing_cases",
    "source": "F11_source_cases",
    "source_itr_sin": "F13_source_itr_sin_cases",
    "interfaces": "F15_interfaces_profiles",
}


# The metric every difficulty panel is drawn on. Kelvin, because the figure is
# about which physical cases are hard, and a normalized error would let a case
# with a small temperature range look bad for the wrong reason.
DIFFICULTY_METRIC = "rmse_K"

# The interaction each benchmark is worth crossing. Both axes are binned on
# edges shared across the whole panel, so the cells are commensurable.
INTERACTIONS = {
    "forcing": ("temporal_family", "spatial_family"),
    "source": ("regime", "lead_bin"),
    "source_itr_sin": ("itr_amplitude", "lead_bin"),
    "interfaces": ("interface_x_bin", "R_c_bin"),
}

# The stratum each case figure contrasts, and which of its levels to show. A
# categorical stratum names its levels explicitly so the set is a stated
# editorial choice, not whichever families happened to be sampled; an ordinal
# one gives a column count instead, because its level labels are generated from
# bin edges and hardcoding them would break on a reformat.
#
# The counts are not uniform, and deliberately so: each figure shows as many
# columns as its benchmark's variation needs. `forcing` and `source` each carry
# a three-way contrast that is part of the argument, `interfaces` needs three
# interface positions to show the model is not memorising one discontinuity,
# and `source_itr_sin` varies one scalar, for which the two extremes suffice.
CASE_CONTRAST = {
    "forcing": ("temporal_family", ("sin", "exp", "pulse_train")),
    "source": ("regime", ("left", "near", "right")),
    "source_itr_sin": ("itr_amplitude", 2),
    "interfaces": ("interface_x_bin", 3),
}

# A second stratum pinned to one distinct level per contrast column. `forcing`
# crosses four temporal families with four spatial profiles, and selecting on
# the temporal family alone leaves the profile to whichever simulation happens
# to be that family's median -- which drew `gaussian` twice and `triangle`
# twice, so the figure showed half the spatial factor and repeated itself.
#
# Three columns cannot cover a 4x4 product, so the pairing is chosen for
# contrast rather than for coverage: broad and smooth (`sin` / `uniform`),
# localized (`exp` / `gaussian`), and pulsed with transverse structure
# (`pulse_train` / `triangle`).
CASE_COVER = {
    "forcing": ("spatial_family", {
        "sin": "uniform",
        "exp": "gaussian",
        "pulse_train": "triangle",
    }),
}

# Benchmarks whose case columns carry no title. The source_itr_sin levels are
# numeric amplitude-bin intervals, which read as clutter above the maps; the
# level of every column is still recorded in the sidecar.
CASE_UNTITLED = frozenset({"source_itr_sin"})

# Benchmarks whose truth maps carry no sim / t_s / lead-time box. On the source
# benchmarks the box sits over the heated region and hides the field; the same
# values are in the sidecar's `cases` entries.
CASE_UNANNOTATED = frozenset({"source", "source_itr_sin"})

# The type around the case maps. The in-panel annotations stay at their own,
# smaller size: they sit on the maps and would cover them if scaled with these.
CASE_LABEL_PT = 9.5
CASE_TICK_PT = 8.5
_RESIDUAL_LABEL = r"$\boldsymbol{T}_{\mathbf{FNO}}-\boldsymbol{T}_{\mathbf{FV}}$"


def _available_strata(frame, benchmark: str) -> list[str]:
    """The declared strata whose source column actually varies in this frame."""
    out = []
    for name in DIFFICULTY_STRATA.get(benchmark, ()):
        spec = stats.stratum_spec(name)
        col = frame.df.get(spec.source)
        if col is None:
            continue
        if col.dropna().astype(str).nunique() < 2:
            continue
        out.append(name)
    return out


def _stratum_panel(ax, frame, stratum: str, color: str):
    """Median + IQR + bootstrap CI of per-sim RMSE across one stratum's levels."""
    sims = stats.per_sim(frame, strata=(stratum,), metrics=(DIFFICULTY_METRIC,))
    df = sims.df
    if "seed" in df.columns:
        df = df.drop(columns=["seed"])
    pooled = stats.SimFrame(df=df, metrics=sims.metrics, strata=sims.strata,
                            pooled=sims.pooled,
                            degradations=list(sims.degradations))
    summaries = stats.summarize(pooled, DIFFICULTY_METRIC, strata=(stratum,))
    levels = stats.stratum_order(df, stratum)

    # summarize keys are ("benchmark",) + strata, in that order, so the stratum
    # level is the last element; matching on membership could collide with a
    # benchmark name that happens to equal a level.
    by_level = {k[-1]: v for k, v in summaries.items()}
    ordered, labels = [], []
    for level in levels:
        if level not in by_level:
            continue
        ordered.append(by_level[level])
        labels.append(level)
    if not ordered:
        return [], []

    positions = np.arange(len(ordered), dtype=float)
    for xi, summary in zip(positions, ordered):
        panels.summary_points(ax, [summary], color=color, offset=xi)
    panels.annotate_counts(ax, positions, ordered)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=5.5,
                       rotation=30 if max(len(s) for s in labels) > 6 else 0,
                       ha="right" if max(len(s) for s in labels) > 6 else "center")
    ax.set_xlim(-0.6, len(ordered) - 0.4)
    ax.set_title(stats.stratum_spec(stratum).label, fontsize=7)
    return ordered, labels


def _interaction_panel(ax, frame, benchmark: str):
    """One curve per level of axis A against the levels of axis B, shared edges."""
    pair = INTERACTIONS.get(benchmark)
    if pair is None:
        return None
    a, b = pair
    for name in (a, b):
        spec = stats.stratum_spec(name)
        col = frame.df.get(spec.source)
        if col is None or col.dropna().astype(str).nunique() < 2:
            return None

    # Edges computed once on the whole frame, not per level of the other axis:
    # binning inside each level would make the bins a function of the group.
    edges = {}
    for name in (a, b):
        if stats.stratum_spec(name).kind != "categorical":
            edges[name] = stats.shared_bin_edges(frame.df, name)

    sims = stats.per_sim(frame, strata=(a, b), metrics=(DIFFICULTY_METRIC,),
                         edges=edges)
    df = sims.df
    if "seed" in df.columns:
        df = df.drop(columns=["seed"])
    pooled = stats.SimFrame(df=df, metrics=sims.metrics, strata=sims.strata,
                            pooled=sims.pooled,
                            degradations=list(sims.degradations))
    summaries = stats.summarize(pooled, DIFFICULTY_METRIC, strata=(a, b))
    a_levels = stats.stratum_order(df, a)
    b_levels = stats.stratum_order(df, b)
    positions = np.arange(len(b_levels), dtype=float)

    cells_by_key = {(k[-2], k[-1]): v for k, v in summaries.items()}
    cmap = plt.get_cmap(style.CMAP_SEQ_ORDINAL)
    for i, a_level in enumerate(a_levels):
        color = cmap(0.15 + 0.7 * i / max(len(a_levels) - 1, 1))
        row = [cells_by_key.get((a_level, b_level)) for b_level in b_levels]
        xs = [x for x, s in zip(positions, row) if s is not None]
        cells = [s for s in row if s is not None]
        if not cells:
            continue
        for xi, summary in zip(xs, cells):
            panels.summary_points(ax, [summary], color=color,
                                  offset=xi + 0.12 * (i - (len(a_levels) - 1) / 2),
                                  show_ci=False)
        ax.plot(xs, [s.median for s in cells], color=color, linewidth=0.9,
                marker="", label=str(a_level))
    ax.set_xticks(positions)
    ax.set_xticklabels(b_levels, fontsize=5.5, rotation=30, ha="right")
    ax.set_xlim(-0.6, len(b_levels) - 0.4)
    ax.set_title(f"{stats.stratum_spec(a).label} x {stats.stratum_spec(b).label}",
                 fontsize=7)
    ax.legend(fontsize=5, ncol=2, columnspacing=0.7, handletextpad=0.3,
              borderpad=0.2)
    return pair


def _itr_panel(ax, source, frame):
    """``R_c(y)`` and the face conductance it produces, for source_itr_sin.

    This is what makes F12 a combined resistance profile-and-difficulty figure rather than a
    fourth copy of the stratum grid: the resistance profile is the benchmark.
    """
    bundle = fields.bundle(source, "source_itr_sin")
    y = np.asarray(bundle.y_grid, dtype=float)
    amp = np.asarray(frame.df["R_c_A"], dtype=float)
    sim_ids = np.asarray(frame.df["sim_id"], dtype=np.int64)
    order = np.argsort(amp, kind="mergesort")
    picks = dict.fromkeys(int(sim_ids[order[k]])
                          for k in (0, len(order) // 2, len(order) - 1))

    cmap = plt.get_cmap(style.CMAP_SEQ_ORDINAL)
    twin = ax.twinx()
    k_left, k_right = fields.layer_conductivities(bundle)
    for i, sim_id in enumerate(picks):
        color = cmap(0.15 + 0.7 * i / 2.0)
        iface = fields.interface_location(bundle, sim_id)
        R_c = np.broadcast_to(
            np.asarray(fields.resistance(bundle, sim_id), dtype=float), y.shape)
        G = jump.interface_conductance(bundle.x_grid, iface, R_c, k_left, k_right)
        ax.plot(y, R_c, color=color, linewidth=1.0, label=f"sim {sim_id}")
        twin.plot(y, np.broadcast_to(np.asarray(G, dtype=float), y.shape),
                  color=color, linewidth=0.7, linestyle=":")
    ax.set_xlabel("$y$")
    ax.set_ylabel("$R_c(y)$")
    twin.set_ylabel("$G(y)$ (dotted)", fontsize=6)
    twin.tick_params(labelsize=6)
    ax.set_title("Void profile", fontsize=7)
    ax.legend(fontsize=5, borderpad=0.2, handletextpad=0.3)


def _contrast_levels(df, stratum: str, wanted) -> list[str]:
    """The stratum levels a case figure shows, in display order.

    ``wanted`` names them explicitly for a categorical stratum. An integer asks
    for that many levels spread evenly over the declared order: two gives the
    extremes, three the extremes plus a middle. An ordinal stratum is specified
    this way rather than by name because its levels are labelled from bin
    edges, which are a property of the sampled data and not stable enough to
    hardcode.
    """
    if stratum not in df.columns:
        return []
    order = stats.stratum_order(df, stratum)
    if isinstance(wanted, int):
        n = min(wanted, len(order))
        if n < 1:
            return []
        idx = np.unique(np.linspace(0, len(order) - 1, n).round().astype(int))
        return [order[i] for i in idx]
    return [level for level in order if level in wanted]


def _draw_patch(ax, patch) -> None:
    """Outline the volumetric heating patch on a field map.

    ``field_map`` draws in physical ``(x, y)`` with extent ``(0, 1, 0, 1)``, and
    ``w_h``/``h_h`` are full widths (``problems/source.py:221``), so the corner
    is the centre minus a half width.

    Solid white over a dark halo, the same annotation language as the interface
    line in ``panels.field_map``. No single flat colour survives all three
    panels: the patch is by construction the hottest region, so it sits on
    ``inferno``'s near-white top end, and the identical box is redrawn on the
    white-centred ``RdBu_r`` residual. The halo is what keeps it readable in
    both places, so it is load-bearing rather than decoration.

    ``zorder`` sits between the interface line (2) and the default text level
    (3): the box has to clear the image and the interface, but a patch sampled
    near the bottom-left corner lands on the case label, and the label is the
    one that has to win.
    """
    x_h, y_h, w_h, h_h = patch
    ax.add_patch(Rectangle(
        (x_h - 0.5 * w_h, y_h - 0.5 * h_h), w_h, h_h,
        fill=False, edgecolor="w", linewidth=1.1, zorder=2.5,
        path_effects=[patheffects.withStroke(linewidth=2.5, foreground="0.08")]))


def difficulty(*, benchmark: str, source=None, spec=None, requirement=None):
    """F08 / F10 / F12 / F14 -- what makes a case hard, per benchmark.

    One panel per stratum in ``DIFFICULTY_STRATA[benchmark]``, each a median
    with IQR whisker and bootstrap CI over **per-simulation** values, plus an
    optional ECDF over simulations and one interaction panel on shared bin
    edges.

    The current versions of these figures are long-tail histograms over
    ``(sim_id, s, j)`` pairs: the tail crushes the body, and the pair count is a
    pseudo-replicate count. Distributions over simulations fix both.
    """
    key = _DIFFICULTY_KEYS.get(benchmark, f"difficulty[{benchmark}]")
    frames = records.records_by_benchmark(source)
    frame = frames.get(benchmark)
    if frame is None:
        blocked(requirement,
                f"needs schema-v2 test_records.csv for {benchmark}; no records "
                "artifact for that benchmark resolved from the manifest.",
                key=key)

    strata = _available_strata(frame, benchmark)
    wants_ecdf = benchmark in DIFFICULTY_ECDF_BENCHMARKS
    wants_itr = benchmark == "source_itr_sin" and bool(fields.run_dirs(source, benchmark))
    n_panels = len(strata) + (1 if wants_ecdf else 0) \
        + (1 if INTERACTIONS.get(benchmark) else 0) \
        + (1 if wants_itr else 0)
    ncol = min(n_panels, 3)
    nrow = int(np.ceil(n_panels / ncol))

    width = getattr(spec, "width", "two_col")
    fig, axes = plt.subplots(nrow, ncol, squeeze=False,
                             figsize=style.figsize(width, rows=nrow,
                                                   row_height="std"))
    flat = [ax for row in axes for ax in row]
    color = style.benchmark_color(benchmark)
    used, drawn_strata = 0, {}

    for stratum in strata:
        ax = flat[used]; used += 1
        summaries, labels = _stratum_panel(ax, frame, stratum, color)
        ax.set_yscale("log")
        drawn_strata[stratum] = {
            lab: {"n_sims": s.n_sims, "median": s.median,
                  "q25": s.q25, "q75": s.q75, "ci_method": s.ci_method}
            for s, lab in zip(summaries, labels)
        }

    sims_flat = stats.per_sim(frame, metrics=(DIFFICULTY_METRIC, "rel_l2_pct"))
    if wants_ecdf:
        # The exceedance fraction at any threshold is readable directly, which
        # a histogram of a heavy tail is not.
        ax = flat[used]; used += 1
        panels.ecdf_curves(
            ax,
            {benchmark: sims_flat.df["rel_l2_pct"].to_numpy(dtype=float)},
            colors={benchmark: color},
            xlabel=style.axis_label("rel_l2_pct"), logx=True,
            reference=1.0, reference_label="1% target",
            title="Per-simulation relative $L_2$")

    interaction = None
    kelvin_axes = list(flat[:len(strata)])
    if INTERACTIONS.get(benchmark):
        ax = flat[used]
        interaction = _interaction_panel(ax, frame, benchmark)
        if interaction is None:
            ax.set_visible(False)
        else:
            ax.set_yscale("log")
            kelvin_axes.append(ax)
            used += 1

    if wants_itr:
        _itr_panel(flat[used], source, frame)
        used += 1

    for ax in flat[used:]:
        ax.set_visible(False)
    # Only the leftmost panel of each row carries the Kelvin label; the others
    # share its scale and repeating it would cost width for nothing.
    for ax in kelvin_axes:
        if flat.index(ax) % ncol == 0:
            ax.set_ylabel(style.axis_label(DIFFICULTY_METRIC))
    # Stratum labels are rotated and long, so the default subplot spacing lets a
    # row's tick labels run into the next row's title. Lay out before the panel
    # letters are placed, since they sit outside the axes box.
    fig.tight_layout(h_pad=1.9, w_pad=1.3)
    style.panel_letters([ax for ax in flat if ax.get_visible()],
                        loc=(-0.20, 1.06))

    metric_definition = {
        "space": "kelvin",
        "benchmark": benchmark,
        "metric": DIFFICULTY_METRIC,
        "aggregation": "pooled per simulation, then median / IQR / BCa CI over "
                       "simulations; the bootstrap resamples sim_id",
        "strata": drawn_strata,
        "interaction": list(interaction) if interaction else None,
        "n_simulations": sims_flat.n_sims,
        "n_pairs": sims_flat.n_pairs,
    }
    return fig, None, metric_definition


def cases(*, benchmark: str, source=None, spec=None, requirement=None):
    """F09 / F11 / F13 / F15 -- representative cases, per benchmark.

    Design:

    * One column per contrast level, titled with the level; three rows, being FV
      truth, FNO prediction, and residual. There is no contact-jump row: the
      jump evidence is consolidated into its own cross-benchmark figure, where
      four curves can be compared directly instead of being spread one per
      column across four figures on four different y scales.
    * Cases come from ``select.select_case`` at the **stratum** median, one per
      contrasting stratum value, so the figure shows a declared median case in
      each regime rather than whichever case looks most dramatic. The chosen
      ``(sim_id, s, j)``, ``t_s``, ``t_j`` and lead time go on the panel and into
      the sidecar.
    * Truth and prediction share one sequential scale across the whole figure
      via ``style.shared_color_limits(mode="sequential")``; residuals share one
      symmetric diverging scale. Per-column scales would rescale each column
      against itself and hide the amplitude difference between the levels.
    * Selection is restricted twice before the median rule runs, and the rule is
      unchanged inside the restricted pool. ``fields.MIN_TRANSVERSE_FRACTION``
      keeps cases whose truth field varies along ``y``, so the maps are not
      effectively 1-D; ``fields.MIN_FIELD_CONTRAST_K`` keeps cases that span a
      visible temperature range, so no column is a flat 300 K block under the
      figure-wide shared scale. The two are independent -- a field can be
      strongly transverse and almost isothermal -- and a qualitative panel needs
      both. The transverse filter pulls toward earlier targets, where the
      benchmarks still have structure, and the contrast filter pulls away from
      the earliest, where nothing has happened yet.
    * The contrast per benchmark, sized to what that benchmark varies rather
      than to a uniform column count:

      - ``forcing``: three temporal families pinned to three different spatial
        profiles via ``CASE_COVER``, chosen for contrast: broad and smooth,
        localized, and pulsed with transverse structure.
      - ``source``: ``regime`` left / near-interface / right, with the patch
        outline drawn on every panel, so the reader sees the model handle the
        source on both sides of the discontinuity and on top of it.
      - ``interfaces``: three interface positions, low / middle / high, which is
        the demonstration that the model is not memorising a fixed
        discontinuity.
      - ``source_itr_sin``: the two extremes of the resistance amplitude. This
        benchmark varies one scalar on top of ``source``'s geometry, so it needs
        fewer columns than the benchmarks whose variation is the argument.
    """
    key = _CASE_KEYS.get(benchmark, f"cases[{benchmark}]")
    frames = records.records_by_benchmark(source)
    frame = frames.get(benchmark)
    if frame is None or not fields.run_dirs(source, benchmark):
        blocked(requirement,
                f"needs schema-v2 test_records.csv for {benchmark} plus the "
                f"{benchmark} checkpoint, since the prediction panels are "
                "evaluated rather than read.",
                key=key)

    stratum, wanted = CASE_CONTRAST.get(benchmark, ("lead_bin", 3))
    cover_stratum, cover_map = CASE_COVER.get(benchmark, (None, {}))
    strata = (stratum,) + ((cover_stratum,) if cover_stratum else ())
    sims = stats.per_sim(frame, strata=strata,
                         metrics=("rmse_K", "rel_l2_pct"))
    # The pair frame needs the same stratum columns, since case selection filters
    # at both levels; assigning it here rather than reaching into per_sim keeps
    # the two calls on one deterministic set of bin edges.
    pair_df = stats.assign_strata(frame.df, strata)
    levels = _contrast_levels(sims.df, stratum, wanted)
    if not levels:
        blocked(requirement,
                f"{benchmark} records carry no usable {stratum} contrast",
                key=key)

    bundle = fields.bundle(source, benchmark)
    picks, cases_, stat, restricted, filters = [], [], [], [], []
    for level in levels:
        strata_filter = {stratum: level}
        if level in cover_map:
            strata_filter[cover_stratum] = cover_map[level]
        filters.append(strata_filter)
        pick, is_2d = fields.select_transverse_case(
            bundle, sims.df, pair_df, quantile=0.5, metric="rmse_K",
            min_field_contrast_K=fields.MIN_FIELD_CONTRAST_K,
            strata_filter=strata_filter)
        picks.append(pick)
        restricted.append(is_2d)
        case = fields.evaluate_case(bundle, pick.sim_id, pick.s, (pick.j,))
        cases_.append(case)
        stat.append(fields.case_metrics(case)[0])

    # One sequential temperature scale and one symmetric residual scale across
    # the whole figure. Per-case scales would rescale each column against
    # itself and hide the amplitude difference between the levels, which is
    # half of what a contrast figure is for.
    temp_limits = style.shared_color_limits(
        *[c.truth[0] for c in cases_], *[c.pred[0] for c in cases_],
        mode="sequential")
    residual_limits = style.shared_color_limits(
        *[c.residual[0] for c in cases_], mode="diverging")

    width = getattr(spec, "width", "full_page")
    # Columns are the contrast levels and rows are the three fields, so a level
    # reads top to bottom.
    #
    # cbar_label_in above the default: a field spanning 300.00-300.20 puts
    # six-character ticks on the bar, which the default 0.50 in cannot hold.
    #
    # Everything else here is the height budget. These figures span both columns
    # of the page, so what a reader feels as "too big" is height at a fixed
    # width, i.e. the aspect ratio, and the aspect ratio of a grid of *square*
    # maps is close to its column:row count. At three of each it is close to
    # square no matter what, and the only slack is the non-map furniture:
    # `cases` draws no suptitle and ships no footer, so the default top and
    # bottom margins reserve about half an inch of height for things that are
    # not there. Reclaiming it, and closing the row gaps that carry no tick
    # labels (only the last row is labelled), is the whole of the available
    # gain; it does not stretch a single panel.
    #
    # max_map_in then sets absolute size. Every column count here hits the
    # bound, so it is what actually fixes the panel size, and it sits well below
    # the width the page allows on purpose: the annotations, tick labels and row
    # labels are at fixed point sizes, so shrinking the maps raises the type
    # *relative to* the panels. Scaled back up to the text width it is the only
    # way a three-row grid of maps reads at this size.
    last = len(levels) - 1
    titled = benchmark not in CASE_UNTITLED
    top_in = (0.46 if cover_map else 0.30) if titled else 0.08
    grid = panels.map_grid(style.WIDTHS_IN[width], 3, n_maps=len(levels),
                           colorbar_after=(last,),
                           colorbar_rows={last: ((0, 1), (2,))},
                           max_map_in=1.05, cbar_label_in=0.80,
                           left_in=0.95,
                           top_in=top_in, bottom_in=0.50,
                           row_gap_in=0.08)
    fig, axes = grid.fig, grid.maps

    for col, (level, case, pick) in enumerate(zip(levels, cases_, picks)):
        # A pinned second stratum has to be on the panel; otherwise the column
        # reads as "the median sin case" when it is "the median sin case with a
        # uniform profile", and the two are not the same claim.
        title = level
        if level in cover_map:
            # Two lines: at the case label size "pulse_train / triangle" is
            # wider than its map and runs into the neighbouring title.
            title = f"{level} /\n{cover_map[level]}"
        patch = fields.source_patch(case.params)
        for row, (field, limits, name) in enumerate((
                (case.truth[0], temp_limits, "FV truth"),
                (case.pred[0], temp_limits, "FNO prediction"),
                (case.residual[0], residual_limits, _RESIDUAL_LABEL))):
            ax = axes[row][col]
            im = panels.field_map(
                ax, field, limits, interface_x=case.interface_x,
                title=title if row == 0 and titled else "",
                xlabel=r"$\boldsymbol{x}$" if row == 2 else "",
                ylabel=f"{name}\n$\\boldsymbol{{y}}$" if col == 0 else "",
                # Four ticks per axis need about 1.4 in to clear each other at
                # 7 pt; these maps are narrower than that by design.
                tick_bins=3)
            # tick_params, not set_xticklabels: these axes keep an automatic
            # locator, which regenerates labels over any fixed list at draw
            # time.
            if col:
                ax.tick_params(labelleft=False)
            if row != 2:
                ax.tick_params(labelbottom=False)
            style.bold_axis_text(ax, label_size=CASE_LABEL_PT,
                                 tick_size=CASE_TICK_PT, title_size=CASE_LABEL_PT)
            if patch is not None:
                _draw_patch(ax, patch)
            bar = None
            if col == last and row == 0:
                bar = panels.attach_colorbar(fig, im, ax, r"$\boldsymbol{T}$ [K]",
                                             cax=grid.colorbars[last][0])
            elif col == last and row == 2:
                bar = panels.attach_colorbar(fig, im, ax, f"{_RESIDUAL_LABEL} [K]",
                                             cax=grid.colorbars[last][1])
            if bar is not None:
                style.bold_axis_text(bar.ax, label_size=CASE_LABEL_PT,
                                     tick_size=CASE_TICK_PT)

        # Three short lines rather than two long ones: the second line of the
        # two-line form is as wide as the panel itself at this map size, and a
        # label that spans its own panel edge to edge reads as an overlay
        # rather than as an annotation.
        if benchmark not in CASE_UNANNOTATED:
            axes[0][col].text(
                0.02, 0.02,
                f"sim {pick.sim_id} ({pick.rank_in_sims}/{pick.n_sims})\n"
                f"$t_s$={case.t_source:.3g}\n"
                f"$\\bar t$={stat[col]['lead_time']:.3g}",
                transform=axes[0][col].transAxes, fontsize=5.0, color="w",
                ha="left", va="bottom", linespacing=1.2,
                bbox=dict(facecolor="0.15", alpha=0.55, pad=1.0, edgecolor="none"))
        axes[2][col].text(
            0.97, 0.03,
            f"rel. $L_2$={stat[col]['rel_l2_pct']:.2f}%\n"
            f"RMSE={stat[col]['rmse_K']:.3g} K",
            transform=axes[2][col].transAxes, fontsize=5.0, color="0.1",
            ha="right", va="bottom", linespacing=1.2,
            # RdBu_r is pale at zero and saturated at the tails, so unbacked
            # text is legible on some residual maps and not on others.
            bbox=dict(facecolor="w", alpha=0.75, edgecolor="none",
                      boxstyle="square,pad=0.15"))

    # The axes are placed at absolute offsets by `map_grid`; tight_layout would
    # undo that and reintroduce the ragged spacing it is there to prevent.

    metric_definition = {
        "space": "kelvin",
        "benchmark": benchmark,
        "contrast": stratum,
        "levels": list(levels),
        "pinned_stratum": cover_stratum,
        "pinned_levels": {lev: f[cover_stratum] for lev, f in zip(levels, filters)
                          if cover_stratum in f},
        "selection": f"median simulation by pooled rmse_K within each {stratum} "
                     "level, median pair within that simulation"
                     + (f", with {cover_stratum} pinned to one distinct level "
                        "per column" if cover_stratum else ""),
        "transverse_restriction": fields.transverse_note(all(restricted)),
        "transverse_minimum": fields.MIN_TRANSVERSE_FRACTION,
        "field_contrast_restriction": fields.field_contrast_note(),
        "field_contrast_minimum_K": fields.MIN_FIELD_CONTRAST_K,
        "color_scales": "one sequential temperature scale and one symmetric "
                        "diverging residual scale, each shared across every "
                        "panel in the figure",
        "temperature_limits": [temp_limits.vmin, temp_limits.vmax],
        "residual_limits": [residual_limits.vmin, residual_limits.vmax],
        "cases": [{"level": lev, "transverse_restricted": bool(r),
                   **p.to_dict(), **m}
                  for lev, p, m, r in zip(levels, picks, stat, restricted)],
    }
    return fig, picks[0], metric_definition


__all__ = ["DIFFICULTY_STRATA", "cases", "difficulty"]
