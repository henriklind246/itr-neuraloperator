"""Cross-benchmark figures: F04, F05, F07, F21.

These are the figures the statistical redesign is *for*. Every reduction below
comes from ``visual.pub.stats``: the defect being corrected is that the current
paper figures summarize ``(sim_id, s, j)`` pairs as if they were independent,
which inflates every ``n`` and shrinks every interval. Here the replication
unit is the simulation, always, and the bootstrap resamples ``sim_id``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from visual.pub import fields, panels, records, select, stats, style
from visual.pub._blocked import blocked

_RECORDS_NEEDED = (
    "needs schema-v2 test_records.csv for forcing, source, source_itr and "
    "interfaces; no records artifact resolved from the manifest."
)

_CHECKPOINT_NEEDED = (
    "needs a run directory with a checkpoint for at least one benchmark; the "
    "figure evaluates the model, it does not read a precomputed field."
)

BENCH_ORDER = ("forcing", "source", "source_itr", "interfaces")

# The relative-L2 target the project states for itself. It is the only
# threshold in the repo that was declared in advance rather than read off the
# results, which is why every exceedance rate here is measured against it.
TARGET_REL_L2_PCT = 1.0


def benchmark_order(frames) -> list[str]:
    """Declared benchmark order first, then anything else, alphabetically."""
    present = list(frames)
    known = [b for b in BENCH_ORDER if b in present]
    return known + sorted(b for b in present if b not in BENCH_ORDER)


def _load(source, requirement, key: str, metrics):
    frames = records.records_by_benchmark(source)
    if not frames:
        blocked(requirement, _RECORDS_NEEDED, key=key)
    order = benchmark_order(frames)
    sims = {b: stats.per_sim(frames[b], metrics=metrics) for b in order}
    return frames, order, sims


def _pool_seeds(sim_frame: stats.SimFrame) -> stats.SimFrame:
    """Drop the seed grouping so one summary covers every simulation.

    ``summarize`` keys on ``(seed, benchmark)`` whenever a seed column exists.
    The headline interval is over simulations pooled across seeds, with the
    per-seed values drawn separately as dots, so the seed grouping is removed
    here rather than the summary being stitched back together afterwards.
    """
    df = sim_frame.df
    if "seed" in df.columns:
        df = df.drop(columns=["seed"])
    return stats.SimFrame(df=df, metrics=sim_frame.metrics,
                          strata=sim_frame.strata, pooled=sim_frame.pooled,
                          degradations=list(sim_frame.degradations))


def _summaries(sims, order, metric):
    """One :class:`Summary` per benchmark, plus the per-seed medians."""
    out, seed_values, degradations = [], [], []
    for bench in order:
        pooled = stats.summarize(_pool_seeds(sims[bench]), metric)
        summary = pooled.get((bench,)) or next(iter(pooled.values()))
        out.append(summary)
        degradations.extend(summary.degradations)

        rolls = stats.across_seeds(sims[bench], metric)
        roll = rolls.get((bench,)) or next(iter(rolls.values()), None)
        seed_values.append(list(roll.per_seed.values()) if roll else [])
        if roll:
            degradations.extend(roll.degradations)
    return out, seed_values, degradations


def _degradation_codes(degradations) -> list[str]:
    return sorted({d.code for d in degradations})


def headline_accuracy(*, source=None, spec=None, requirement=None):
    """F04 -- cross-benchmark accuracy, the figure that proves the stats layer.

    Four panels, each a single metric so no axis mixes physical spaces: global
    RMSE, interface-region RMSE, node-jump RMSE (all Kelvin), and relative
    :math:`L_2` with the 1% target line (normalized).

    Every point is the median over **simulations**, with the IQR as a thin
    whisker and a BCa bootstrap CI -- resampled over ``sim_id`` -- as the thick
    bar. Below 20 simulations no CI is drawn at all and ``TOO_FEW_SIMS`` goes
    into the sidecar, because a bar there would be read as an estimate.

    The jump metric here is the adjacent-node difference that training and
    monitoring use. The physical contact jump :math:`R_c(y) q_n(y, t)` requires
    the field, not the record, and is drawn in F06.
    """
    metrics = ("rmse_K", "iface_rmse_K", "node_jump_rmse_K", "rel_l2_pct")
    frames, order, sims = _load(source, requirement, "F04_headline_accuracy",
                                metrics)

    width = spec.width if spec is not None else "two_col"
    fig, axes = plt.subplots(2, 2, figsize=style.figsize(width, rows=2,
                                                         row_height="std"))

    colors = [style.benchmark_color(b) for b in order]
    # One line, not two: rotated multi-line ticks drift sideways line by line
    # and collide with their neighbours at this panel width.
    labels = [b.replace("_", "-") for b in order]
    all_degradations, counts = [], {}

    for ax, metric in zip(axes.flat, metrics):
        summaries, seed_values, degradations = _summaries(sims, order, metric)
        all_degradations.extend(degradations)
        counts[metric] = {b: s.n_sims for b, s in zip(order, summaries)}

        positions = np.arange(len(order), dtype=float)
        for xi, summary, color in zip(positions, summaries, colors):
            panels.summary_points(ax, [summary], color=color, offset=xi,
                                  show_ci=True)
        panels.seed_dots(ax, positions, seed_values)
        panels.annotate_counts(ax, positions, summaries)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=6, rotation=25, ha="right")
        ax.set_xlim(-0.6, len(order) - 0.4)
        ax.set_yscale("log")
        ax.set_ylabel(style.axis_label(metric))
        ax.set_title(stats.metric_spec(metric).label, fontsize=7)
        ax.grid(True, axis="y", alpha=0.25)

    style.add_reference_line(axes[1][1], value=TARGET_REL_L2_PCT,
                             label=f"{TARGET_REL_L2_PCT:g}% target")

    style.panel_letters(axes)
    fig.tight_layout()

    metric_definition = {
        "aggregation": "pair -> simulation (pooled sufficient statistics) -> "
                       "median over simulations",
        "replication_unit": "sim_id",
        "interval": "BCa bootstrap over simulations, 95%",
        "seed_markers": "median over simulations, one marker per model seed",
        "metrics": {m: stats.metric_spec(m).label for m in metrics},
        "spaces": {m: stats.metric_spec(m).space for m in metrics},
        "jump": "adjacent-node temperature difference (training proxy); the "
                "physical contact jump R_c(y) q_n(y,t) is F06",
        "target_rel_l2_pct": TARGET_REL_L2_PCT,
        "benchmarks": order,
        "n_simulations": counts.get("rmse_K", {}),
        "n_pairs": {b: frames[b].n_pairs for b in order},
        "pooled": {b: bool(sims[b].pooled) for b in order},
        "degradations": _degradation_codes(all_degradations),
    }
    return fig, None, metric_definition


def jump_vs_lead(*, source=None, spec=None, requirement=None):
    """F07 -- interface error against lead time.

    The first user of shared bin edges. Lead bins are computed once over the
    pooled records of all four benchmarks and reused for every one of them;
    binning each benchmark on its own quantiles would give each a different set
    of edges and make the curves silently incommensurable.

    Binning happens at pair level, aggregation at simulation level, and the
    bootstrap resamples ``sim_id`` within a bin. Two panels: absolute Kelvin
    jump error, and interface relative :math:`L_2`, so a benchmark is not
    credited for merely having a larger temperature range.
    """
    metrics = ("node_jump_rmse_K", "iface_rel_l2_pct", "rmse_K")
    frames, order, _ = _load(source, requirement, "F07_jump_vs_lead", metrics)

    edges = stats.shared_bin_edges([frames[b].df for b in order], "lead_bin")
    sims = {b: stats.per_sim(frames[b], strata=("lead_bin",), metrics=metrics,
                             edges={"lead_bin": edges})
            for b in order}

    bin_labels = stats.stratum_order(sims[order[0]].df, "lead_bin")
    for bench in order[1:]:
        bin_labels = tuple(dict.fromkeys(
            bin_labels + stats.stratum_order(sims[bench].df, "lead_bin")))

    width = spec.width if spec is not None else "two_col"
    fig, axes = plt.subplots(1, 2, figsize=style.figsize(width, row_height="std"))

    all_degradations = []
    for ax, metric in zip(axes, ("node_jump_rmse_K", "iface_rel_l2_pct")):
        for bench in order:
            pooled = stats.summarize(_pool_seeds(sims[bench]), metric,
                                     strata=("lead_bin",))
            ordered, present = [], []
            for label in bin_labels:
                summary = pooled.get((bench, label))
                if summary is None or not np.isfinite(summary.median):
                    continue
                ordered.append(summary)
                present.append(label)
                all_degradations.extend(summary.degradations)
            if not ordered:
                continue
            offsets = np.array([bin_labels.index(p) for p in present], dtype=float)
            colour = style.benchmark_color(bench)
            for xi, summary in zip(offsets, ordered):
                panels.summary_points(ax, [summary], color=colour, offset=xi)
            ax.plot(offsets, [s.median for s in ordered], "-", color=colour,
                    linewidth=1.0, alpha=0.85, label=bench)

        ax.set_xticks(np.arange(len(bin_labels), dtype=float))
        ax.set_xticklabels(bin_labels, fontsize=5.5, rotation=20, ha="right")
        ax.set_xlim(-0.6, len(bin_labels) - 0.4)
        ax.set_xlabel(r"lead-time bin (quantiles of $\bar{t}$, shared edges)")
        ax.set_ylabel(style.axis_label(metric))
        ax.set_yscale("log")
        ax.set_title(stats.metric_spec(metric).label
                     + f"  [{stats.metric_spec(metric).space}]", fontsize=7)
        ax.grid(True, axis="y", alpha=0.25)

    axes[0].legend(fontsize=5.5, loc="best")
    style.panel_letters(axes)
    fig.tight_layout()

    metric_definition = {
        "aggregation": "pair -> bin -> simulation -> median over simulations",
        "replication_unit": "sim_id within a lead bin",
        "binning": "quantile bins of t_bar, edges computed once over the pooled "
                   "records of every benchmark and reused for all of them",
        "bin_edges": [float(e) for e in edges],
        "bin_labels": list(bin_labels),
        "interval": "BCa bootstrap over simulations, 95%",
        "metrics": {m: stats.metric_spec(m).label
                    for m in ("node_jump_rmse_K", "iface_rel_l2_pct")},
        "spaces": {m: stats.metric_spec(m).space
                   for m in ("node_jump_rmse_K", "iface_rel_l2_pct")},
        "benchmarks": order,
        "degradations": _degradation_codes(all_degradations),
    }
    return fig, None, metric_definition


def tail_reliability(*, source=None, spec=None, requirement=None):
    """F21 -- tail reliability across benchmarks (supplemental).

    A median is not what a surrogate is trusted on; the tail is. Three panels:
    simulation-level order statistics of the global RMSE, the full ECDF of
    relative :math:`L_2` with the 1% target marked, and the fraction of
    simulations exceeding that target with a Wilson interval.

    p99 is drawn only above 100 simulations. Below that it is emitted as NaN
    with ``P99_UNDERDETERMINED`` and annotated on the panel: a 99th percentile
    from 45 simulations is the second-largest value, and plotting it as a
    quantile would be an overclaim.
    """
    metrics = ("rmse_K", "rel_l2_pct")
    frames, order, sims = _load(source, requirement, "F21_tail_reliability",
                                metrics)

    width = spec.width if spec is not None else "two_col"
    fig, axes = plt.subplots(1, 3, figsize=style.figsize(width, row_height="std"))

    colors = [style.benchmark_color(b) for b in order]
    # One line, not two: rotated multi-line ticks drift sideways line by line
    # and collide with their neighbours at this panel width.
    labels = [b.replace("_", "-") for b in order]
    positions = np.arange(len(order), dtype=float)

    summaries, _, all_degradations = _summaries(sims, order, "rmse_K")
    ax = axes[0]
    for stat_name, marker, alpha in (("median", "o", 1.0),
                                     ("p90", "s", 0.85),
                                     ("p99", "^", 0.7)):
        values = np.array([getattr(s, stat_name) for s in summaries], dtype=float)
        ax.plot(positions, values, marker, markersize=4.0, linestyle="none",
                markeredgecolor="k", markeredgewidth=0.4, alpha=alpha,
                label=stat_name, zorder=4,
                markerfacecolor="none" if stat_name != "median" else None,
                color="0.25" if stat_name != "median" else None)
    for xi, summary, colour in zip(positions, summaries, colors):
        ax.vlines(xi, summary.median, summary.max, color=colour, linewidth=0.9,
                  alpha=0.5, zorder=2)
    undetermined = [s for s in summaries if not np.isfinite(s.p99)]
    if undetermined:
        ax.text(0.02, 0.97, f"p99 undetermined (< {stats.MIN_SIMS_FOR_P99} sims)",
                transform=ax.transAxes, ha="left", va="top", fontsize=5.0,
                color="#B22222")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=6, rotation=25, ha="right")
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_yscale("log")
    ax.set_ylabel(style.axis_label("rmse_K"))
    ax.set_title("Order statistics over simulations", fontsize=7)
    ax.legend(fontsize=5.5, loc="upper center", ncol=3, columnspacing=0.8,
              handletextpad=0.3, borderpad=0.2)
    ax.grid(True, axis="y", alpha=0.25)

    series, rates = {}, {}
    for bench in order:
        pooled = _pool_seeds(sims[bench]).df
        values = pooled["rel_l2_pct"].to_numpy(dtype=float)
        series[bench] = values
        rates[bench] = stats.exceedance_rate(values, TARGET_REL_L2_PCT)

    panels.ecdf_curves(
        axes[1], series, colors={b: c for b, c in zip(order, colors)},
        xlabel=style.axis_label("rel_l2_pct"), logx=True,
        title="Per-simulation relative $L_2$",
        reference=TARGET_REL_L2_PCT, reference_label=f"{TARGET_REL_L2_PCT:g}%")

    panels.rate_bars(
        axes[2], rates, labels=labels,
        colors={b: c for b, c in zip(order, colors)},
        ylabel="fraction of simulations", title="Exceedance of the target",
        threshold_label=f"rel. $L_2$ > {TARGET_REL_L2_PCT:g}%",
        rotate_xticks=25.0)

    style.panel_letters(axes)
    fig.tight_layout()

    metric_definition = {
        "aggregation": "pair -> simulation (pooled) -> order statistics over "
                       "simulations",
        "replication_unit": "sim_id",
        "p99_rule": f"reported only at >= {stats.MIN_SIMS_FOR_P99} simulations; "
                    "NaN otherwise with P99_UNDERDETERMINED",
        "exceedance_threshold_rel_l2_pct": TARGET_REL_L2_PCT,
        "exceedance_interval": "Wilson score, 95%",
        "exceedance": {b: {"k": r.k, "n": r.n, "rate": r.rate,
                           "lo": r.lo, "hi": r.hi}
                       for b, r in rates.items()},
        "spaces": {"rmse_K": "kelvin", "rel_l2_pct": "normalized"},
        "benchmarks": order,
        "n_pairs": {b: frames[b].n_pairs for b in order},
        "degradations": _degradation_codes(all_degradations),
    }
    return fig, None, metric_definition


def truth_pred_residual(*, source=None, spec=None, requirement=None):
    """F05 -- truth, prediction and residual, one row per benchmark.

    Physical Kelvin throughout. Truth and prediction share one sequential color
    scale per row; residuals share a single symmetric diverging scale across
    the whole figure, so a row is not flattered by being rescaled against
    itself.

    The case comes from ``select.select_case(quantile=0.5)`` -- a declared
    median case, never the worst -- and the resulting ``(sim_id, s, j)``,
    ``t_s``, ``t_j`` and lead time are printed on the panels and written to the
    sidecar.
    """
    metrics = ("rmse_K", "rel_l2_pct")
    frames, order, sims = _load(source, requirement, "F05_truth_pred_residual",
                                metrics)
    order = [b for b in order if fields.run_dirs(source, b)]
    if not order:
        blocked(requirement, _CHECKPOINT_NEEDED, key="F05_truth_pred_residual")

    cases, selections, restricted = [], [], []
    for bench in order:
        bundle = fields.bundle(source, bench)
        pick, is_2d = fields.select_transverse_case(
            bundle, sims[bench], frames[bench].df, quantile=0.5, metric="rmse_K")
        case = fields.evaluate_case(bundle, pick.sim_id, pick.s, (pick.j,))
        cases.append(case)
        selections.append(pick)
        restricted.append(is_2d)

    residual_limits = style.shared_color_limits(
        *[c.residual[0] for c in cases], mode="diverging")

    width = spec.width if spec is not None else "full_page"
    # The residual scale is shared down the whole figure, so it gets one bar
    # spanning every row; temperature is per row and gets one bar per row.
    grid = panels.map_grid(style.WIDTHS_IN[width], len(order),
                           colorbar_after=(1, 2), colorbar_span=(2,))
    fig, axes = grid.fig, grid.maps

    for row, (bench, case, pick) in enumerate(zip(order, cases, selections)):
        temp_limits = style.shared_color_limits(case.truth[0], case.pred[0],
                                                mode="sequential")
        stat = fields.case_metrics(case)[0]
        for col, (field, limits, name) in enumerate((
                (case.truth[0], temp_limits, "FV truth"),
                (case.pred[0], temp_limits, "FNO prediction"),
                (case.residual[0], residual_limits, "residual"))):
            ax = axes[row][col]
            im = panels.field_map(
                ax, field, limits, interface_x=case.interface_x,
                title=name if row == 0 else "",
                xlabel="$x$" if row == len(order) - 1 else "",
                ylabel=f"{bench}\n$y$" if col == 0 else "")
            if col == 1:
                panels.attach_colorbar(fig, im, ax, "$T$ [K]",
                                       cax=grid.colorbars[1][row])
            elif col == 2 and row == 0:
                panels.attach_colorbar(fig, im, ax, "$T_{FNO}-T_{FV}$ [K]",
                                       cax=grid.colorbars[2][0])
            # tick_params, not set_yticklabels: these axes keep an automatic
            # locator, which regenerates labels over any fixed list at draw
            # time.
            if col != 0:
                ax.tick_params(labelleft=False)
            if row != len(order) - 1:
                ax.tick_params(labelbottom=False)
        axes[row][0].text(
            0.02, 0.02,
            f"sim {pick.sim_id} (rank {pick.rank_in_sims}/{pick.n_sims})\n"
            f"$t_s$={case.t_source:.3g}  $t_j$={stat['t_target']:.3g}  "
            f"$\\bar t$={stat['lead_time']:.3g}",
            transform=axes[row][0].transAxes, fontsize=5.0, color="w",
            ha="left", va="bottom",
            # The corner this sits in is the bright end of `inferno` on three of
            # the four rows, where white text disappears. A dark backing box
            # keeps the case identity readable whatever the field does.
            bbox=dict(facecolor="0.1", alpha=0.55, edgecolor="none",
                      boxstyle="square,pad=0.2"))
        axes[row][2].text(
            0.98, 0.02,
            f"rel. $L_2$={stat['rel_l2_pct']:.2f}%\n"
            f"RMSE={stat['rmse_K']:.3g} K",
            transform=axes[row][2].transAxes, fontsize=5.0, color="0.15",
            ha="right", va="bottom")

    # No panel letters and no tight_layout. The axes sit at absolute offsets, so
    # tight_layout would undo the alignment; and twelve letters over a grid whose
    # rows are already named by benchmark and whose columns are already titled is
    # clutter that buys nothing a caption cannot say.

    metric_definition = {
        "space": "kelvin",
        "selection": "median simulation by pooled rmse_K, median pair within it",
        "transverse_restriction": fields.transverse_note(all(restricted)),
        "transverse_minimum": fields.MIN_TRANSVERSE_FRACTION,
        "color_scales": "truth and prediction share one sequential scale per "
                        "row; every residual panel shares one symmetric "
                        "diverging scale across the figure",
        "residual_limits": [residual_limits.vmin, residual_limits.vmax],
        "cases": [{"transverse_restricted": bool(r),
                   **s.to_dict(), **fields.case_metrics(c)[0]}
                  for s, c, r in zip(selections, cases, restricted)],
        "benchmarks": order,
    }
    return fig, selections[0], metric_definition


__all__ = [
    "BENCH_ORDER",
    "TARGET_REL_L2_PCT",
    "benchmark_order",
    "headline_accuracy",
    "jump_vs_lead",
    "tail_reliability",
    "truth_pred_residual",
]
