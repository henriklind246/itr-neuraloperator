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
import matplotlib.ticker as mticker
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
    "source_itr": ("lead_bin", "void_severity", "R_c_bin"),
    "interfaces": ("lead_bin", "interface_x_bin", "R_c_bin"),
}

# Source uses its fifth panel for global physical error across contact resistance
# instead of repeating the normalized per-simulation error distribution.
DIFFICULTY_ECDF_BENCHMARKS = frozenset({"forcing", "source_itr", "interfaces"})

_DIFFICULTY_KEYS = {
    "forcing": "F08_forcing_difficulty",
    "source": "F10_source_difficulty",
    "source_itr": "F12_source_itr_void",
    "interfaces": "F14_interfaces_difficulty",
}

_CASE_KEYS = {
    "forcing": "F09_forcing_cases",
    "source": "F11_source_cases",
    "source_itr": "F13_source_itr_cases",
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
    "source_itr": ("void_severity", "lead_bin"),
    "interfaces": ("interface_x_bin", "R_c_bin"),
}

# The stratum each case figure contrasts, and which of its levels to show. A
# ``None`` level list means the extremes of the stratum's declared order, which
# is the right contrast for an ordinal stratum; a categorical one names the
# levels explicitly so the set is a stated editorial choice, not whichever
# families happened to be sampled.
CASE_CONTRAST = {
    "forcing": ("temporal_family", ("sin", "exp", "pulse_train", "exp_train")),
    "source": ("regime", ("left", "near", "right")),
    "source_itr": ("void_severity", None),
    "interfaces": ("interface_x_bin", None),
}

# A second stratum pinned to one distinct level per contrast column. `forcing`
# crosses four temporal families with four spatial profiles, and selecting on
# the temporal family alone leaves the profile to whichever simulation happens
# to be that family's median -- which drew `gaussian` twice and `triangle`
# twice, so the figure showed half the spatial factor and repeated itself. The
# pairing is by declared order in ``stats.STRATUM_SPECS``, so one column per
# profile as well as per family.
CASE_COVER = {
    "forcing": ("spatial_family", {
        "sin": "uniform",
        "exp": "patch",
        "pulse_train": "gaussian",
        "exp_train": "triangle",
    }),
}


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


def _void_panel(ax, source, frame):
    """``R_c(y)`` and the face conductance it produces, for source_itr.

    This is what makes F12 a combined void-and-difficulty figure rather than a
    fourth copy of the stratum grid: the void is the benchmark.
    """
    bundle = fields.bundle(source, "source_itr")
    y = np.asarray(bundle.y_grid, dtype=float)
    amp = np.asarray(frame.df["R_c_amp"], dtype=float)
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

    ``wanted`` names them explicitly for a categorical stratum; ``None`` means
    take the two extremes of the declared order, which is the informative
    contrast for an ordinal one (a middle row would only interpolate).
    """
    if stratum not in df.columns:
        return []
    order = stats.stratum_order(df, stratum)
    if wanted is None:
        return [order[0], order[-1]] if len(order) > 1 else list(order)
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
    wants_void = benchmark == "source_itr" and bool(fields.run_dirs(source, benchmark))
    n_panels = len(strata) + (1 if wants_ecdf else 0) \
        + (1 if INTERACTIONS.get(benchmark) else 0) \
        + (1 if wants_void else 0)
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

    if wants_void:
        _void_panel(flat[used], source, frame)
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

    Design, once the artifacts exist:

    * One column per contrast level, titled with the level; rows are FV truth,
      FNO prediction, residual, and the contact-jump profile.
    * Cases come from ``select.select_case`` at the **stratum** median, one per
      contrasting stratum value, so the figure shows a declared median case in
      each regime rather than whichever case looks most dramatic. The chosen
      ``(sim_id, s, j)``, ``t_s``, ``t_j`` and lead time go on the panel and into
      the sidecar.
    * Truth and prediction share one sequential scale across the whole figure
      via ``style.shared_color_limits(mode="sequential")``; residuals share one
      symmetric diverging scale. The jump row does **not**: it is read for shape
      rather than amplitude, and a shared axis flattens the small-amplitude
      columns into a line.
    * Selection is additionally restricted to cases whose truth contact jump
      spans at least ``fields.MIN_JUMP_CONTRAST_K`` along ``y``, so the jump row
      shows transverse structure. The median rule is unchanged inside that pool.
    * The contrast per benchmark:

      - ``forcing``: all four temporal families, smooth (``sin``, ``exp``)
        against pulse-like (``pulse_train``, ``exp_train``), each pinned to a
        different spatial profile via ``CASE_COVER`` so the four columns cover
        both factors of the forcing product rather than repeating a profile.
      - ``source``: ``regime`` left / near-interface / right, with the patch
        outline drawn on every panel so the reader can see where the heat went
        in.
      - ``source_itr``: field plus contact jump, with the localized jump error
        around the void made visible as a zoom or difference strip -- the void
        is the point of the benchmark and it is small.
      - ``interfaces``: temperature and contact-jump **profiles** rather than a
        field grid, since the interesting variation is along ``y`` at the
        interface and across ``x_I``.

    * Every jump curve is the physical ``R_c(y) q_n(y, t)`` from
      ``visual.pub.jump``, never the adjacent-node difference the training loss
      uses.
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

    stratum, wanted = CASE_CONTRAST.get(benchmark, ("lead_bin", None))
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
        # The jump row is a quarter of this figure, and the transverse filter
        # does not protect it: it scores the field, and a field can be strongly
        # two-dimensional while its interface jump is flat. Narrowing on the
        # jump's own spread makes the row show the transverse structure it
        # exists to show.
        pick, is_2d = fields.select_transverse_case(
            bundle, sims.df, pair_df, quantile=0.5, metric="rmse_K",
            strata_filter=strata_filter,
            min_jump_contrast_K=fields.MIN_JUMP_CONTRAST_K)
        picks.append(pick)
        restricted.append(is_2d)
        case = fields.evaluate_case(bundle, pick.sim_id, pick.s, (pick.j,))
        cases_.append(case)
        stat.append(fields.case_metrics(case)[0]
                    | {"contact_jump_ptp_K": float(fields.jump_contrast(
                        bundle, [pick.sim_id], [pick.j])[0])})

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
    # Columns are the contrast levels and rows are the three fields plus a
    # trailing row of contact-jump curves, so a level reads top to bottom.
    # cbar_label_in above the default: a field spanning 300.00-300.20 puts
    # six-character ticks on the bar, which the default 0.50 in cannot hold.
    # max_map_in bounds the two-level benchmarks, where dividing the full width
    # by two columns would otherwise give page-tall maps.
    last = len(levels) - 1
    grid = panels.map_grid(style.WIDTHS_IN[width], 3, n_maps=len(levels),
                           colorbar_after=(last,),
                           colorbar_rows={last: ((0, 1), (2,))},
                           extra_rows=1, extra_row_height=0.78,
                           extra_row_inset_in=0.24,
                           max_map_in=1.65, cbar_label_in=0.60)
    fig, axes = grid.fig, grid.maps
    y_grid = np.asarray(bundle.y_grid, dtype=float)
    color = style.benchmark_color(benchmark)

    for col, (level, case, pick) in enumerate(zip(levels, cases_, picks)):
        # A pinned second stratum has to be on the panel; otherwise the column
        # reads as "the median sin case" when it is "the median sin case with a
        # uniform profile", and the two are not the same claim.
        title = level
        if level in cover_map:
            title = f"{level} / {cover_map[level]}"
        patch = fields.source_patch(case.params)
        for row, (field, limits, name) in enumerate((
                (case.truth[0], temp_limits, "FV truth"),
                (case.pred[0], temp_limits, "FNO prediction"),
                (case.residual[0], residual_limits,
                 "$T_{\\rm FNO}-T_{\\rm FV}$"))):
            ax = axes[row][col]
            im = panels.field_map(
                ax, field, limits, interface_x=case.interface_x,
                title=title if row == 0 else "",
                xlabel="$x$" if row == 2 else "",
                ylabel=f"{name}\n$y$" if col == 0 else "")
            # tick_params, not set_xticklabels: these axes keep an automatic
            # locator, which regenerates labels over any fixed list at draw
            # time.
            if col:
                ax.tick_params(labelleft=False)
            if row != 2:
                ax.tick_params(labelbottom=False)
            if patch is not None:
                _draw_patch(ax, patch)
            if col == last and row == 0:
                panels.attach_colorbar(fig, im, ax, "$T$ [K]",
                                       cax=grid.colorbars[last][0])
            elif col == last and row == 2:
                panels.attach_colorbar(fig, im, ax, "$T_{FNO}-T_{FV}$ [K]",
                                       cax=grid.colorbars[last][1])

        j_true, j_pred = case.contact_jumps()
        ax_j = grid.rows[0][col]
        ax_j.plot(y_grid, j_true[0], color="0.15", linewidth=1.0, label="FV")
        ax_j.plot(y_grid, j_pred[0], color=color, linewidth=1.0,
                  linestyle="--", label="FNO")
        ax_j.set_xlabel("$y$")
        ax_j.tick_params(axis="y", labelsize=5.5)
        if col == 0:
            ax_j.set_ylabel(f"${jump.rc_symbol(benchmark)}\\,q_n$ [K]",
                            fontsize=7)
            ax_j.legend(fontsize=5.5, loc="best", handletextpad=0.3,
                        borderpad=0.2, labelspacing=0.2)
        void = fields.void_profile(case.params)
        if void is not None:
            style.add_void_shading(ax_j, void[0], void[1], label=None)

        axes[0][col].text(
            0.02, 0.02,
            f"sim {pick.sim_id} ({pick.rank_in_sims}/{pick.n_sims})\n"
            f"$t_s$={case.t_source:.3g}  $\\bar t$={stat[col]['lead_time']:.3g}",
            transform=axes[0][col].transAxes, fontsize=5.0, color="w",
            ha="left", va="bottom",
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

    # The jump row gets per-column y limits, unlike the maps. The columns differ
    # by an order of magnitude in jump amplitude, and on a shared axis the small
    # ones collapse onto a line that reads as "no transverse structure" when the
    # profile in fact varies by a third of its own mean. The maps can afford a
    # shared scale because amplitude is part of what they compare; the jump row
    # is there to show *shape*, and shape is what a shared axis destroys. Every
    # panel therefore carries its own tick labels -- the scales differ, so an
    # unlabelled axis would be read off its neighbour and misread.
    for ax_j in grid.rows[0]:
        lo, hi = ax_j.get_ylim()
        if lo <= 0.0 <= hi or min(abs(lo), abs(hi)) < 0.25 * max(hi - lo, 1e-12):
            lo, hi = min(lo, 0.0), max(hi, 0.0)
            ax_j.axhline(0.0, color="0.6", linewidth=0.5, zorder=0)
        # A `uniform` forcing profile has a rigorously constant jump, so the
        # autoscaled range is degenerate and matplotlib falls back to a decade
        # of ticks around it.
        if hi - lo < 1e-9:
            mid = 0.5 * (lo + hi)
            lo, hi = mid - 0.5, mid + 0.5
        ax_j.set_ylim(lo, hi)
        ax_j.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4, prune="upper"))
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
        "jump": jump.JUMP_DEFINITION,
        "jump_contrast_restriction": fields.jump_contrast_note(),
        "jump_contrast_minimum_K": fields.MIN_JUMP_CONTRAST_K,
        "jump_axis_scales": "one y scale per column, so each column's jump "
                            "shape is legible at its own amplitude; the maps "
                            "above still share one scale",
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
