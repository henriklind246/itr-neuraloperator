"""Cross-benchmark figures: F04, F05, F07, F21, F26-F28 and F31.

These are the figures the statistical redesign is *for*. Every reduction below
comes from ``visual.pub.stats``: the defect being corrected is that the current
paper figures summarize ``(sim_id, s, j)`` pairs as if they were independent,
which inflates every ``n`` and shrinks every interval. Here the replication
unit is the simulation, always, and the bootstrap resamples ``sim_id``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import (
    FuncFormatter,
    LogLocator,
    MaxNLocator,
    NullFormatter,
    StrMethodFormatter,
)

from visual.pub import fields, panels, records, select, stats, style, tables
from visual.pub._blocked import blocked
from visual.pub.manifest import ProvenanceError

_RECORDS_NEEDED = (
    "needs schema-v2 test_records.csv for forcing, source, source_itr_sin and "
    "interfaces; no records artifact resolved from the manifest."
)

_CHECKPOINT_NEEDED = (
    "needs a run directory with a checkpoint for at least one benchmark; the "
    "figure evaluates the model, it does not read a precomputed field."
)

BENCH_ORDER = ("forcing", "source", "source_itr_sin", "interfaces")

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
    the field, not the record, and is drawn in F06 and F26.
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
                "physical contact jump R_c(y) q_n(y,t) is F06 and F26",
        "target_rel_l2_pct": TARGET_REL_L2_PCT,
        "benchmarks": order,
        "n_simulations": counts.get("rmse_K", {}),
        "n_pairs": {b: frames[b].n_pairs for b in order},
        "pooled": {b: bool(sims[b].pooled) for b in order},
        "degradations": _degradation_codes(all_degradations),
    }
    return fig, None, metric_definition


def jump_vs_lead(*, source=None, spec=None, requirement=None):
    """F07 -- pair-weighted field and node-jump error at all exact leads."""
    frames = records.records_by_benchmark(source)
    if not frames:
        blocked(requirement, _RECORDS_NEEDED, key="F07_jump_vs_lead")
    order = benchmark_order(frames)

    width = spec.width if spec is not None else "two_col"
    fig, axes = plt.subplots(1, 2, figsize=style.figsize(width, row_height="std"))

    counts = {}
    for ax, metric in zip(axes, ("field", "jump")):
        for bench in order:
            curve = tables.lead_time_curve(frames[bench].df, metric)
            colour = style.benchmark_color(bench)
            x = curve["t_bar"].to_numpy()
            ax.fill_between(
                x, curve["test_cluster_ci_lower"], curve["test_cluster_ci_upper"],
                color=colour, alpha=0.10, linewidth=0,
            )
            ax.plot(x, curve["mean"], "-", color=colour, linewidth=1.0,
                    label=bench)
            ax.plot(x, curve["p95"], "--", color=colour, linewidth=0.8,
                    alpha=0.8)
            counts[bench] = curve[["t_bar", "n_pairs_per_seed"]].to_dict("records")
        ax.set_xlabel(r"lead time $\bar{t}$")
        metric_name = "rmse_K" if metric == "field" else "node_jump_rmse_K"
        ax.set_ylabel(style.axis_label(metric_name))
        ax.set_yscale("log")
        ax.set_title(
            "Field RMSE" if metric == "field" else "Node-jump RMSE",
            fontsize=7,
        )
        ax.grid(True, axis="y", alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="0.25", linestyle="--", linewidth=0.8))
    labels.append("P95 error (95th percentile)")
    axes[0].legend(handles, labels, fontsize=5.5, loc="best")
    style.panel_letters(axes)
    fig.tight_layout()

    metric_definition = {
        "aggregation": "pairwise mean and P95 at each exact positive lead; "
                       "seed-specific statistics are averaged",
        "replication_unit": "sim_id within each exact lead",
        "lead_grid": "all 30 positive saved-snapshot lead times",
        "interval": "95% simulation-cluster percentile bootstrap around the mean",
        "interval_interpretation": "finite test-cohort uncertainty conditional "
                                   "on the fixed trained models",
        "line_encoding": "solid=mean with CI ribbon; dashed=P95 without ribbon",
        "metrics": {"field": "Field RMSE", "jump": "Node-jump RMSE"},
        "spaces": {"field": "kelvin", "jump": "kelvin"},
        "benchmarks": order,
        "pair_counts_per_seed": counts,
        "degradations": sorted({d.code for f in frames.values()
                                 for d in f.degradations}),
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


def physical_contact_jump_vs_lead(*, source=None, spec=None, requirement=None):
    """F26 -- physical contact-jump fidelity across benchmarks (supplemental).

    A fixed cohort of simulations is evaluated from the initial snapshot to
    every future saved time. Interface profiles are reduced within each
    simulation first, then the median and IQR are taken across simulations.
    """
    frames = records.records_by_benchmark(source)
    if not frames:
        blocked(requirement, _RECORDS_NEEDED,
                key="F26_contact_jump_vs_lead")
    order = [b for b in benchmark_order(frames) if fields.run_dirs(source, b)]
    if not order:
        blocked(requirement, _CHECKPOINT_NEEDED,
                key="F26_contact_jump_vs_lead")

    width = spec.width if spec is not None else "two_col"
    fig, axes = plt.subplots(1, 3, figsize=style.figsize(width, row_height="std"))
    cohorts, curves = {}, {}

    for bench in order:
        cohort = fields.evaluate_contact_jump_cohort(
            fields.bundle(source, bench), frames[bench],
        )
        curve = stats.contact_jump_curve(
            cohort.lead_times, cohort.truth, cohort.pred,
        )
        cohorts[bench] = cohort
        curves[bench] = curve
        color = style.benchmark_color(bench)
        lead = curve.lead_times

        axes[0].plot(lead, curve.truth_median, color=color, linewidth=1.0,
                     label=bench)
        axes[0].plot(lead, curve.pred_median, color=color, linewidth=1.0,
                     linestyle="--")

        axes[1].fill_between(
            lead, curve.rmse_q25, curve.rmse_q75,
            color=color, alpha=0.12, linewidth=0,
        )
        axes[1].plot(lead, curve.rmse_median, color=color, linewidth=1.0)

        axes[2].fill_between(
            lead, curve.relative_q25_pct, curve.relative_q75_pct,
            color=color, alpha=0.12, linewidth=0,
        )
        axes[2].plot(lead, curve.relative_median_pct, color=color, linewidth=1.0)

    axes[0].set_title("Physical jump magnitude", fontsize=7)
    axes[0].set_ylabel(r"RMS $|R_c q_n|$ [K]")
    axes[1].set_title("Physical jump error", fontsize=7)
    axes[1].set_ylabel("contact-jump RMSE [K]")
    axes[2].set_title("Relative physical jump error", fontsize=7)
    axes[2].set_ylabel("profile relative $L_2$ [%]")
    for ax in axes:
        ax.set_xlabel(r"lead time $\bar{t}$")
        ax.grid(True, alpha=0.25)

    benchmark_handles = [
        Line2D([0], [0], color=style.benchmark_color(b), linewidth=1.2,
               label=b)
        for b in order
    ]
    encoding_handles = [
        Line2D([0], [0], color="0.25", linewidth=1.0, label="FV truth"),
        Line2D([0], [0], color="0.25", linewidth=1.0, linestyle="--",
               label="FNO prediction"),
    ]
    axes[0].legend(handles=[*benchmark_handles, *encoding_handles], fontsize=5.2,
                   loc="best", ncol=2, columnspacing=0.8, handletextpad=0.4)
    style.panel_letters(axes)
    fig.tight_layout()

    metric_definition = {
        "space": "kelvin",
        "physical_quantity": "contact jump R_c(y) q_n(y,t)",
        "protocol": "source snapshot s=0; every subsequent saved target time",
        "cohort_selection": "sample without replacement from record sim_id values",
        "cohort_rng_seed": fields.CONTACT_JUMP_COHORT_SEED,
        "requested_simulations_per_benchmark": fields.CONTACT_JUMP_COHORT_SIZE,
        "replication_unit": "sim_id",
        "aggregation": "profile RMS/RMSE/relative L2 over y within each "
                       "simulation and lead; median and IQR across simulations",
        "interval": "IQR across simulations; descriptive, not a confidence interval",
        "benchmarks": order,
        "cohorts": {
            b: {
                "n_simulations": curves[b].n_sims,
                "sim_ids": list(cohorts[b].sim_ids),
                "source_index": cohorts[b].source_index,
                "target_indices": list(cohorts[b].target_indices),
                "lead_times": cohorts[b].lead_times.tolist(),
            }
            for b in order
        },
    }
    return fig, None, metric_definition


NODE_JUMP_LEAD_QUANTILES = (0.1, 0.5, 0.9)

# y nodes kept per profile when drawing the parity cloud. The cohort is 24
# simulations x ~30 leads x ~100 y nodes, and 72k markers per panel is an
# unreadable slab in the raster and a multi-megabyte path in the PDF. The fit
# below is computed on the *full* cohort regardless, so the decimation changes
# what is drawn and not what is reported; the factor is recorded in the sidecar.
PARITY_Y_SAMPLES = 8


def node_jump_fidelity(*, source=None, spec=None, requirement=None):
    """F31 -- adjacent-node interface-jump fidelity across benchmarks.

    Two rows over the same four benchmarks. The top row is one representative
    case per benchmark, truth solid and prediction dashed at three lead times,
    which shows whether the *shape* of the jump along y is recovered. The bottom
    row is the whole seeded cohort as a parity cloud, which shows whether the
    *magnitude* is, and whether the error is a slope bias or symmetric scatter --
    a distinction the RMSE in F07 cannot make.

    The quantity is the adjacent-node jump ``T(x_R, y) - T(x_L, y)``, which is
    what ``test_records.csv`` reports as ``node_jump_rmse_K``. It is not the
    physical contact jump; F26 shows that one.
    """
    frames = records.records_by_benchmark(source)
    if not frames:
        blocked(requirement, _RECORDS_NEEDED, key="F31_node_jump_fidelity")
    order = [b for b in benchmark_order(frames) if fields.run_dirs(source, b)]
    if not order:
        blocked(requirement, _CHECKPOINT_NEEDED, key="F31_node_jump_fidelity")

    cases, cohorts, fits, picks, restrictions, columns_by_bench = {}, {}, {}, {}, {}, {}
    for bench in order:
        frame = frames[bench]
        bundle = fields.bundle(source, bench)
        sims = stats.per_sim(frame, metrics=("rmse_K", "rel_l2_pct"))
        pick, restricted = fields.select_transverse_case(
            bundle, sims, frame.df, quantile=0.5, metric="rmse_K",
            min_leads=len(NODE_JUMP_LEAD_QUANTILES),
            min_node_jump_contrast_K=fields.MIN_NODE_JUMP_CONTRAST_K)
        # The lead columns have to come from the pool the case was ranked in.
        # When the restriction was dropped, the pick is only guaranteed to span
        # the leads of the unrestricted frame.
        pool = frame.df
        if restricted:
            pool = fields.node_jump_contrast_pairs(
                bundle, fields.transverse_pairs(bundle, frame.df))
        cols = select.select_lead_columns(
            pool, pick, lead_quantiles=NODE_JUMP_LEAD_QUANTILES)
        cases[bench] = [fields.evaluate_case(bundle, pick.sim_id, s, (j,))
                        for s, j in cols]
        cohort = fields.evaluate_node_jump_cohort(bundle, frame)
        cohorts[bench] = cohort
        fits[bench] = stats.parity_fit(cohort.truth, cohort.pred,
                                       n_sims=len(cohort.sim_ids))
        picks[bench] = pick
        restrictions[bench] = restricted
        columns_by_bench[bench] = cols

    # One lead-time scale across every panel. Per-panel normalization would make
    # the darkest marker mean a different lead in each column, and the row would
    # stop being comparable across benchmarks.
    lead_lo = min(float(c.lead_times.min()) for c in cohorts.values())
    lead_hi = max(float(c.lead_times.max()) for c in cohorts.values())
    norm = plt.Normalize(vmin=lead_lo, vmax=lead_hi)

    width = spec.width if spec is not None else "two_col"
    n_cols = len(order)
    fig, axes = plt.subplots(
        2, n_cols, squeeze=False,
        figsize=style.figsize(width, rows=2, row_height="short", extra_in=0.5))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.175, top=0.91,
                        wspace=0.34, hspace=0.38)

    mappable = None
    for col, bench in enumerate(order):
        color = style.benchmark_color(bench)
        case_list = cases[bench]
        y_grid = np.asarray(case_list[0].y_grid, dtype=float)
        truth_profiles = np.stack([c.node_jumps()[0][0] for c in case_list])
        pred_profiles = np.stack([c.node_jumps()[1][0] for c in case_list])
        leads = np.asarray([float(c.lead_times[0]) for c in case_list])

        ax_top = axes[0][col]
        panels.jump_profile_family(
            ax_top, y_grid, truth_profiles, pred_profiles, lead_times=leads,
            color=color, xlabel="$y$", title=bench, legend=True)
        ax_top.axhline(0.0, color="0.6", linewidth=0.5, zorder=0)
        ax_top.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax_top.text(0.97, 0.03, f"sim {picks[bench].sim_id}",
                    transform=ax_top.transAxes, ha="right", va="bottom",
                    fontsize=5.0, color="0.35")

        cohort = cohorts[bench]
        lead_field = np.broadcast_to(
            cohort.lead_times[None, :, None], cohort.truth.shape)
        step = max(1, cohort.truth.shape[2] // PARITY_Y_SAMPLES)
        mappable = panels.parity_scatter(
            axes[1][col], cohort.truth[:, :, ::step], cohort.pred[:, :, ::step],
            fit=fits[bench], color_by=lead_field[:, :, ::step], norm=norm,
            xlabel="FV truth [K]",
            ylabel="FNO prediction [K]" if col == 0 else "")
        axes[1][col].xaxis.set_major_locator(MaxNLocator(nbins=4))
        axes[1][col].yaxis.set_major_locator(MaxNLocator(nbins=4))

    axes[0][0].set_ylabel(r"$\Delta T_{\rm node}$ [K]", fontsize=7)

    # The tint in row A already carries lead time per panel, so the solid/dashed
    # key is the only thing that has to be said once for the whole row.
    fig.legend(handles=[Line2D([], [], color=style.GREY, linestyle="-",
                               linewidth=2.8, alpha=0.40, label="FV truth"),
                        Line2D([], [], color=style.GREY, linewidth=1.0,
                               linestyle=(0, (3.0, 1.4)),
                               label="FNO prediction")],
               loc="lower left", bbox_to_anchor=(0.015, 0.0), ncol=1,
               fontsize=8, frameon=False, handlelength=2.4,
               labelspacing=0.4, handletextpad=0.6, borderaxespad=0.2)

    cax = fig.add_axes([0.52, 0.055, 0.30, 0.022])
    fig.colorbar(mappable, cax=cax, orientation="horizontal")
    cax.set_xlabel(r"lead time $\bar t$", fontsize=7, labelpad=1.0)
    cax.tick_params(labelsize=6)
    style.panel_letters(axes)

    metric_definition = {
        "space": "kelvin",
        "physical_quantity": "adjacent-node interface jump T(x_R, y) - T(x_L, y)",
        "relation_to_contact_jump":
            "not the physical contact jump; this quantity carries both half-cell "
            "bulk drops as well as the contact discontinuity, and is reported "
            "because it is the definition test_records.csv scores as "
            "node_jump_rmse_K. See F26 for the physical jump.",
        "sign_convention": "right minus left, matching src/operators/eval.py",
        "row_a": {
            "content": "one representative case per benchmark at lead-time "
                       f"quantiles {NODE_JUMP_LEAD_QUANTILES} of that "
                       "simulation's own t_bar",
            "selection": "median simulation by pooled rmse_K within the "
                         "restricted pool",
            "node_jump_contrast_restriction":
                fields.node_jump_contrast_note(),
            "node_jump_contrast_minimum_K": fields.MIN_NODE_JUMP_CONTRAST_K,
            "cases": {
                b: {
                    "sim_id": int(picks[b].sim_id),
                    "restricted_pool": bool(restrictions[b]),
                    "columns": [{"s": int(s), "j": int(j)}
                                for s, j in columns_by_bench[b]],
                }
                for b in order
            },
        },
        "row_b": {
            "content": "every (simulation, lead, y) sample of the seeded cohort",
            "protocol": "source snapshot s=0; every subsequent saved target time",
            "cohort_selection": "sample without replacement from record sim_id values",
            "cohort_rng_seed": fields.CONTACT_JUMP_COHORT_SEED,
            "requested_simulations_per_benchmark": fields.CONTACT_JUMP_COHORT_SIZE,
            "slope": "least squares through the origin",
            "drawn_y_subsample": PARITY_Y_SAMPLES,
            "subsampling_note": "markers are decimated along y for legibility; "
                                "slope and RMSE are computed on the full cohort",
            "pooling_note": "RMSE here is a point-wise dispersion over pooled "
                            "samples, not a simulation-level estimate; F07 and "
                            "F26 carry the replicated versions",
            "fits": {
                b: {
                    "slope": fits[b].slope,
                    "rmse_K": fits[b].rmse,
                    "truth_rms_K": fits[b].truth_rms,
                    "relative_pct": fits[b].relative_pct,
                    "n_points": fits[b].n_points,
                    "n_simulations": fits[b].n_sims,
                    "sim_ids": list(cohorts[b].sim_ids),
                    "source_index": cohorts[b].source_index,
                    "lead_times": cohorts[b].lead_times.tolist(),
                }
                for b in order
            },
        },
        "lead_color_limits": [lead_lo, lead_hi],
        "benchmarks": order,
    }
    return fig, None, metric_definition


_FIELD_MARKERS = {"forcing": "o", "source": "s", "source_itr_sin": "^", "interfaces": "D"}
_FIELD_LINES = {"forcing": "-", "source": "--", "source_itr_sin": "-.", "interfaces": ":"}


def _global_field_error_figure(*, axis, source, spec, requirement):
    try:
        frames, metadata = records.load_global_field_records(source)
    except (records.SchemaError, FileNotFoundError) as exc:
        raise ProvenanceError(str(exc)) from exc
    if not frames:
        key = "F27_global_field_error_vs_lead" if axis == "lead" else "F28_global_field_error_vs_itr"
        blocked(requirement, _RECORDS_NEEDED, key=key)
    summary = stats.global_field_error_summary(frames, metadata=metadata)
    rows = summary[axis]
    fig, ax = plt.subplots(figsize=style.figsize("one_col", row_height="std"))
    fig.subplots_adjust(left=0.19, right=0.97, bottom=0.21, top=0.78)
    handles = []
    for benchmark in BENCH_ORDER:
        points = [row for row in rows if row["benchmark"] == benchmark]
        xname = "lead_time" if axis == "lead" else "median_resistance"
        x = np.asarray([row[xname] for row in points], dtype=float)
        y = np.asarray([row["median_rmse_K"] for row in points], dtype=float)
        lo = np.asarray([row["ci_lower_K"] for row in points], dtype=float)
        hi = np.asarray([row["ci_upper_K"] for row in points], dtype=float)
        color = style.benchmark_color(benchmark)
        marker, linestyle = _FIELD_MARKERS[benchmark], _FIELD_LINES[benchmark]
        if axis == "lead":
            ax.fill_between(x, lo, hi, color=color, alpha=0.10, linewidth=0)
            ax.plot(x, y, color=color, linestyle=linestyle, linewidth=1.2,
                    marker=marker, markersize=3.2, markevery=max(1, len(x) // 7),
                    markerfacecolor="white", markeredgewidth=0.8)
        else:
            ax.plot(x, y, color=color, linestyle=linestyle, linewidth=0.6, alpha=0.45)
            # Draw interval endpoints directly: a BCa interval need not contain
            # its sample median, unlike Matplotlib's nonnegative yerr contract.
            valid = np.isfinite(lo) & np.isfinite(hi)
            ax.vlines(x[valid], lo[valid], hi[valid], color=color, linewidth=0.8, zorder=3)
            ax.plot(x[valid], lo[valid], "_", color=color, markersize=4, markeredgewidth=0.8)
            ax.plot(x[valid], hi[valid], "_", color=color, markersize=4, markeredgewidth=0.8)
            ax.plot(x, y, linestyle="none", marker=marker, markersize=4,
                    color=color, markerfacecolor="white", markeredgewidth=1.0, zorder=4)
        handles.append(Line2D([], [], color=color, linestyle=linestyle, marker=marker,
                              markersize=3.5, markerfacecolor="white", linewidth=1.2,
                              label=tables.BENCH_LABEL[benchmark]))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.54, 0.99),
               ncol=2, fontsize=7, frameon=False, handlelength=2.2,
               columnspacing=1.2, handletextpad=0.5, labelspacing=0.45)
    ax.set_xlabel(r"Lead time $\Delta t=t_j-t_s$" if axis == "lead" else
                  r"Interface-mean $\overline{R_c}$ [m$^2$ K/W]", fontsize=8)
    ax.set_ylabel("Median global RMSE [K]", fontsize=8)
    ax.set_xscale("linear")
    ax.set_yscale(summary["yscale"])
    ax.set_ylim(summary["ylim"])
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
    if summary["yscale"] == "log":
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5), numticks=8))
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
    else:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.tick_params(labelsize=7, width=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="y", which="major", color="0.7", alpha=0.35,
            linewidth=0.5, linestyle="-")
    ax.set_axisbelow(True)
    ax.margins(x=0.06)

    title = ("Global field RMSE versus lead time" if axis == "lead" else
             "Global field RMSE stratified by interface-mean resistance")
    seed_note = "; ".join(f"{tables.BENCH_LABEL[b]}: K={summary['seed_counts'][b]}"
                          for b in BENCH_ORDER)
    caption = (
        f"{title}. Median simulation-level global field RMSE over held-out test simulations. "
        "SSE and evaluated cell counts are pooled across eligible pairs within each simulation "
        "and stratum before taking the square root; simulation RMSE is then averaged across "
        "fixed model seeds before the cohort median. Pointwise 95% BCa bootstrap intervals "
        "resample simulations and are not simultaneous confidence bands. Intervals quantify "
        "test-cohort uncertainty conditional on the trained models, not training-seed variability. "
        "Averaging errors across seeds does not evaluate an ensemble-averaged prediction. "
        f"Evaluated model seeds: {seed_note}. "
    )
    if summary["unequal_seed_counts"]:
        caption += "Seed counts differ by benchmark; the amount of training-seed averaging is unequal. "
    if any(row["interval_method"] == "none" and row["n_simulations"] for row in rows):
        caption += f"Intervals are omitted for strata with fewer than {stats.MIN_SIMS_FOR_CI} simulations. "
    if any(row["interval_method"] == "percentile" for row in rows):
        caption += "Degenerate BCa cases use percentile intervals, identified in the statistics. "
    if axis == "itr":
        caption += (
            "Common quantile bins count each simulation once; points sit at each benchmark's "
            "median resistance within its bin. Scalar R_c is used for uniform interfaces; "
            "Source + ITR uses the finite-domain mean of R_c(y), not an effective resistance. "
            "Points represent marginal error stratifications over interface resistance and should "
            "not be interpreted as the isolated effect of resistance because other benchmark "
            "parameters vary concurrently. "
        )
    else:
        caption += (
            "On uniform snapshot grids, equal source-target index lags are pooled "
            "at a common lead time to avoid floating-point splitting. All evaluated "
            "positive leads are retained without smoothing or extrapolation. "
        )
    caption += f"The RMSE axis is {summary['yscale']}; resistance and lead axes are linear."
    representations = {
        b: sorted({m["representation"] for m in metadata.get(b, {}).values()
                   if m.get("representation")}) for b in BENCH_ORDER
    }
    if any(reps != ["temporal_encoder"] for reps in representations.values() if reps):
        caption += " Evaluated representations: " + "; ".join(
            f"{tables.BENCH_LABEL[b]}: {', '.join(reps).replace('_', ' ')}"
            for b, reps in representations.items() if reps) + "."
    if source.degradations:
        caption += " This descriptive comparison has publication-cohort limitations detailed in the provenance sidecar."
    definition = {
        "space": "kelvin", "metric": "rmse_K", "title": title, "caption": caption,
        "aggregation": "cells -> pair sufficient statistics -> pooled simulation RMSE -> seed mean -> cohort median",
        "simulation_rmse": "sqrt(sum_pair(sse_K2) / sum_pair(num_error_cells))",
        "replication_unit": "sim_id within benchmark; equal simulation weights",
        "interval": "pointwise 95% BCa simulation bootstrap; not a simultaneous band",
        "interval_interpretation": "test-cohort uncertainty conditional on fixed trained models; not training-seed variability",
        "minimum_simulations_for_ci": stats.MIN_SIMS_FOR_CI,
        "n_boot": summary["n_boot"], "rng_seed": summary["rng_seed"],
        "seed_counts": summary["seed_counts"], "seed_ids": summary["seeds"],
        "unequal_seed_counts": summary["unequal_seed_counts"],
        "seed_averaging": "mean of simulation errors; not an ensemble-averaged prediction",
        "pair_counts": "eligible_pairs_by_seed counts evaluated pairs before pooling; eligible_pair_rows_total sums across seeds; unique_simulation_pairs counts (sim_id,s,j) once across seeds",
        "resistance_definition": "R_c for scalar interfaces; (d-c)^-1 integral_c^d [R_c + R_c_A sin(pi y)] dy for source_itr_sin",
        "resistance_definitions_by_seed": summary["resistance_definitions"],
        "resistance_bin_edges": summary["resistance_bin_edges"],
        "bin_closure": "left closed, right open; final upper endpoint included; constant values occupy one bin",
        "protocols": summary["protocols"], "record_metadata": metadata,
        "yscale": summary["yscale"], "ylim": summary["ylim"],
        "statistics": rows,
        "degradations": [str(d) for d in source.degradations],
    }
    return fig, None, definition


def global_field_error_vs_lead(*, source=None, spec=None, requirement=None):
    """F27: pointwise uncertainty of simulation-level error at exact leads."""
    return _global_field_error_figure(axis="lead", source=source, spec=spec, requirement=requirement)


def global_field_error_vs_itr(*, source=None, spec=None, requirement=None):
    """F28: marginal simulation-level error strata over mean resistance."""
    return _global_field_error_figure(axis="itr", source=source, spec=spec, requirement=requirement)


def global_field_error_fixed_source(*, source=None, spec=None, requirement=None,
                                    source_fraction=stats.DEFAULT_SOURCE_FRACTION):
    """F32: lead-time curves conditioned on one fixed source snapshot.

    F27 pools every source time at a given lead, so its long leads are drawn
    from a smaller and systematically earlier set of source times than its short
    leads. Fixing the source snapshot removes that confound: the cohort is the
    same at every point on the x axis and only the prediction horizon moves.
    """
    try:
        frames, metadata = records.load_global_field_records(source)
    except (records.SchemaError, FileNotFoundError) as exc:
        raise ProvenanceError(str(exc)) from exc
    if not frames:
        blocked(requirement, _RECORDS_NEEDED, key="F32_global_field_error_fixed_source")
    summary = stats.fixed_source_lead_summary(
        frames, metadata=metadata, source_fraction=source_fraction)
    primary = summary["primary"]
    rows = [row for row in summary["rows"] if row["is_primary"]]

    fig, ax = plt.subplots(figsize=style.figsize("one_col", row_height="std"))
    fig.subplots_adjust(left=0.19, right=0.97, bottom=0.21, top=0.78)
    handles = []
    for benchmark in BENCH_ORDER:
        points = [row for row in rows if row["benchmark"] == benchmark]
        x = np.asarray([row["lead_time"] for row in points], dtype=float)
        y = np.asarray([row["median_rmse_K"] for row in points], dtype=float)
        lo = np.asarray([row["ci_lower_K"] for row in points], dtype=float)
        hi = np.asarray([row["ci_upper_K"] for row in points], dtype=float)
        color = style.benchmark_color(benchmark)
        marker, linestyle = _FIELD_MARKERS[benchmark], _FIELD_LINES[benchmark]
        ax.fill_between(x, lo, hi, color=color, alpha=0.10, linewidth=0)
        ax.plot(x, y, color=color, linestyle=linestyle, linewidth=1.2,
                marker=marker, markersize=3.2, markevery=max(1, len(x) // 7),
                markerfacecolor="white", markeredgewidth=0.8)
        handles.append(Line2D([], [], color=color, linestyle=linestyle, marker=marker,
                              markersize=3.5, markerfacecolor="white", linewidth=1.2,
                              label=tables.BENCH_LABEL[benchmark]))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.54, 0.99),
               ncol=2, fontsize=7, frameon=False, handlelength=2.2,
               columnspacing=1.2, handletextpad=0.5, labelspacing=0.45)
    ax.set_xlabel(r"Lead time $\Delta t=t_j-t_s^\ast$", fontsize=8)
    ax.set_ylabel("Median global RMSE [K]", fontsize=8)
    ax.set_xscale("linear")
    ax.set_yscale(summary["yscale"])
    ax.set_ylim(summary["ylim"])
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
    if summary["yscale"] == "log":
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5), numticks=8))
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
    else:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.tick_params(labelsize=7, width=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="y", which="major", color="0.7", alpha=0.35,
            linewidth=0.5, linestyle="-")
    ax.set_axisbelow(True)
    ax.margins(x=0.06)
    ax.annotate(rf"$t_s^\ast={primary['source_time']:.3g}$", xy=(0.97, 0.04),
                xycoords="axes fraction", ha="right", va="bottom", fontsize=7)

    title = "Global field RMSE against lead time at a fixed source time"
    seed_note = "; ".join(f"{tables.BENCH_LABEL[b]}: K={summary['seed_counts'][b]}"
                          for b in BENCH_ORDER)
    caption = (
        f"{title}. Every prediction starts from the same snapshot "
        f"$t_s^\\ast={primary['source_time']:.3g}$, the evaluated snapshot nearest "
        f"{source_fraction:g} of the {primary['horizon']:.3g} horizon, so each point "
        "varies only the prediction horizon and every lead is measured on the same "
        "simulation cohort. Points are the cohort median of simulation-level global "
        "field RMSE; with one source snapshot each simulation contributes exactly one "
        "pair per lead, which is averaged across fixed model seeds before the median. "
        "Shaded regions are pointwise 95% BCa bootstrap intervals resampling "
        "simulations; they are not simultaneous confidence bands and quantify "
        "test-cohort uncertainty conditional on the trained models, not training-seed "
        "variability. Averaging errors across seeds does not evaluate an "
        "ensemble-averaged prediction. The operator predicts each pair directly, so "
        "any rise with lead time is lead-time-dependent difficulty, not error "
        f"accumulation through a rollout. Evaluated model seeds: {seed_note}. "
    )
    if summary["unequal_seed_counts"]:
        caption += "Seed counts differ by benchmark; the amount of training-seed averaging is unequal. "
    if any(row["interval_method"] == "none" and row["n_simulations"] for row in rows):
        caption += f"Intervals are omitted for leads with fewer than {stats.MIN_SIMS_FOR_CI} simulations. "
    if any(row["interval_method"] == "percentile" for row in rows):
        caption += "Degenerate BCa cases use percentile intervals, identified in the statistics. "
    if summary["field_amplitude_available"]:
        caption += (
            "RMSE is absolute, so vertical offsets between benchmarks partly reflect "
            "differing temperature amplitudes; median_target_rms_K in the statistics "
            "gives each benchmark's field scale for that comparison. "
        )
    caption += (
        f"The RMSE axis is {summary['yscale']}; the lead axis is linear. The "
        "statistics also carry the same reduction at source fractions "
        + ", ".join(f"{f:g}" for f in stats.SOURCE_FRACTION_SWEEP)
        + " so the choice of source time can be checked without re-rendering."
    )
    representations = {
        b: sorted({m["representation"] for m in metadata.get(b, {}).values()
                   if m.get("representation")}) for b in BENCH_ORDER
    }
    if any(reps != ["temporal_encoder"] for reps in representations.values() if reps):
        caption += " Evaluated representations: " + "; ".join(
            f"{tables.BENCH_LABEL[b]}: {', '.join(reps).replace('_', ' ')}"
            for b, reps in representations.items() if reps) + "."
    if source.degradations:
        caption += " This descriptive comparison has publication-cohort limitations detailed in the provenance sidecar."
    definition = {
        "space": "kelvin", "metric": "rmse_K", "title": title, "caption": caption,
        "source_time_selection": primary,
        "source_fraction_sweep": summary["selections"],
        "conditioning": "single fixed source snapshot; the simulation cohort is constant across leads",
        "aggregation": "cells -> pair sufficient statistics -> simulation RMSE at one (t_s*, lead) -> seed mean -> cohort median",
        "simulation_rmse": "sqrt(sum_pair(sse_K2) / sum_pair(num_error_cells)) over the single eligible pair",
        "replication_unit": "sim_id within benchmark; equal simulation weights",
        "prediction_mode": "direct_pair; increases with lead are horizon difficulty, not rollout error accumulation",
        "interval": "pointwise 95% BCa simulation bootstrap; not a simultaneous band",
        "interval_interpretation": "test-cohort uncertainty conditional on fixed trained models; not training-seed variability",
        "field_amplitude": ("median simulation RMS of the true field about the training mean, "
                            "in kelvin" if summary["field_amplitude_available"] else None),
        "minimum_simulations_for_ci": stats.MIN_SIMS_FOR_CI,
        "n_boot": summary["n_boot"], "rng_seed": summary["rng_seed"],
        "seed_counts": summary["seed_counts"], "seed_ids": summary["seeds"],
        "unequal_seed_counts": summary["unequal_seed_counts"],
        "seed_averaging": "mean of simulation errors; not an ensemble-averaged prediction",
        "protocols": summary["protocols"], "record_metadata": metadata,
        "yscale": summary["yscale"], "ylim": summary["ylim"],
        "statistics": summary["rows"],
        "degradations": [str(d) for d in source.degradations],
    }
    return fig, None, definition


def _cell_edges(centers):
    """Edges bracketing a monotone sequence of cell centers.

    ``pcolormesh`` wants edges. Midpoints place every center inside its own
    cell, so a tick reading a source time falls in the column it names rather
    than on the boundary between two of them.
    """
    centers = np.asarray(centers, dtype=float)
    inner = 0.5 * (centers[:-1] + centers[1:])
    return np.concatenate([[2 * centers[0] - inner[0]], inner,
                           [2 * centers[-1] - inner[-1]]])


def _kelvin_ticks(vmin, vmax):
    """Ticks at round kelvin values, on a bar that is linear in log10 kelvin.

    Choosing the ticks in log space puts them at round exponents, whose kelvin
    values read as noise (0.0631, 0.158, 0.251). Choosing them in kelvin and
    mapping back gives a scale the reader can price against a temperature.
    """
    decades = np.arange(np.floor(vmin), np.ceil(vmax) + 1)
    for subs in ((1., 2., 5.), (1., 2., 3., 5., 7.), np.arange(1., 10.)):
        ticks = np.log10(np.concatenate([10.0 ** d * np.asarray(subs) for d in decades]))
        ticks = ticks[(ticks >= vmin) & (ticks <= vmax)]
        if len(ticks) >= 3:
            break
    return ticks


# The shared labels are set for tables and for figures that put several
# benchmarks in a legend, where brevity wins. Here each name has a panel to
# itself, so the one that reads as a bare noun is spelled out.
_SURFACE_TITLES = {"interfaces": "Varying interface"}


def _surface_title(benchmark):
    return _SURFACE_TITLES.get(benchmark, tables.BENCH_LABEL[benchmark])


def source_lead_error_surface(*, source=None, spec=None, requirement=None,
                              benchmarks=stats.SURFACE_BENCHMARKS,
                              source_fraction=stats.DEFAULT_SOURCE_FRACTION):
    """F33: the whole error surface over source time and lead time.

    F32 fixes one source time and draws error against lead. Whether that column
    stands for the others is a question the figure cannot answer about itself,
    and ``scripts/inspect_source_time_sweep.py`` answers it only as a rank. Here
    the surface is drawn whole, one heatmap per benchmark, so the column F32
    draws can be read against its neighbours directly.

    The additive fit and its residual are not drawn. They stay in the exported
    statistics and in the caption's scalars, where they quantify what the
    heatmap shows without costing it two thirds of its width.
    """
    try:
        frames, metadata = records.load_global_field_records(source)
    except (records.SchemaError, FileNotFoundError) as exc:
        raise ProvenanceError(str(exc)) from exc
    if not frames:
        blocked(requirement, _RECORDS_NEEDED, key="F33_source_lead_error_surface")
    surface = stats.source_lead_error_surface(
        frames, metadata=metadata, benchmarks=benchmarks,
        source_fraction=source_fraction)
    order = surface["benchmarks"]
    published = surface["published"]

    x_edges = _cell_edges(surface["source_times"])
    y_edges = _cell_edges(surface["lead_times"])
    sequential = plt.get_cmap(style.CMAP_TEMPERATURE).copy()
    # Most of the (s, k) rectangle is never evaluated. Left to the default those
    # cells take the colormap's lowest color, which reads as a real and very
    # small error rather than as an absence of data.
    sequential.set_bad("0.90")

    fig = plt.figure(figsize=style.figsize("two_col", rows=1, row_height="tall"))
    # A uniform wspace would set the panel-to-bar gap and the bar-to-next-panel
    # gap to the same width, and the second of those has to hold a tick column
    # and an axis label while the first holds nothing. Explicit spacer columns
    # at wspace=0 let the bar sit against its panel.
    ratios, panel_cols, bar_cols = [], [], []
    for index in range(len(order)):
        if index:
            ratios.append(.30)
        panel_cols.append(len(ratios))
        ratios += [1., .03]
        bar_cols.append(len(ratios))
        ratios.append(.045)
    grid = fig.add_gridspec(1, len(ratios), width_ratios=ratios, wspace=0,
                            left=.105, right=.895, bottom=.155, top=.905)
    for column, benchmark in enumerate(order):
        data = surface["surfaces"][benchmark]
        vmin = float(np.nanmin(data["surface_log10"]))
        vmax = float(np.nanmax(data["surface_log10"]))
        ax = fig.add_subplot(grid[0, panel_cols[column]])
        mesh = ax.pcolormesh(x_edges, y_edges,
                             np.ma.masked_invalid(data["surface_log10"]).T,
                             cmap=sequential, vmin=vmin, vmax=vmax,
                             shading="flat", rasterized=True)
        ax.set_xlim(x_edges[0], x_edges[-1])
        ax.set_ylim(y_edges[0], y_edges[-1])
        ax.tick_params(labelsize=9, width=.8, labelleft=column == 0)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(.8)
        ax.set_title(_surface_title(benchmark), fontsize=12, pad=6)
        ax.set_xlabel(r"Source time $t_s$", fontsize=11)
        if column == 0:
            ax.set_ylabel(r"Lead time $\Delta t = t_j - t_s$", fontsize=11)

        # Per panel, not shared: the two benchmarks need not span the same
        # kelvin range, and a shared scale would flatten whichever one is
        # smaller. The comparison the figure invites is of shape across columns
        # within a panel, which a per-panel scale serves.
        bar = fig.colorbar(mesh, cax=fig.add_subplot(grid[0, bar_cols[column]]))
        bar.set_label("Median RMSE [K]", fontsize=10, labelpad=4)
        bar.ax.tick_params(labelsize=9, width=.7, pad=2)
        bar.ax.yaxis.set_ticks(_kelvin_ticks(vmin, vmax))
        bar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{10.0 ** v:g}"))
        bar.outline.set_linewidth(.8)

    title = "Global field RMSE over source time and lead time"

    labels = ", ".join(_surface_title(b) for b in order)
    seed_note = "; ".join(f"{_surface_title(b)}: K={surface['seed_counts'][b]}"
                          for b in order)
    numbers = "; ".join(
        f"{_surface_title(b)} {surface['surfaces'][b]['level_spread_pct']:.0f}% level "
        f"spread, {surface['surfaces'][b]['interaction_pct']:.1f}% residual, "
        f"{surface['surfaces'][b]['common_shape_fraction']:.2f} shared-shape fraction"
        for b in order)
    caption = (
        f"{title}. Each cell is the cohort median over held-out test simulations of the "
        "simulation-level global field RMSE at one evaluated (source snapshot, lead) pair, "
        f"for {labels}. SSE and evaluated cell counts are pooled within a simulation and "
        "averaged across fixed model seeds before the cohort median. Grey cells are not "
        "evaluated: a source snapshot admits only the leads that stay inside the horizon, so "
        "the domain is triangular rather than rectangular. Reading a panel column by column is "
        "reading error against lead at one source time; the scalars below come from the "
        "least-squares fit of log10 RMSE = mu + alpha(source) + beta(lead) over the evaluated "
        "cells, which is the model in which every column traces one shape and differs from the "
        "others only by a constant factor. Its residual, not drawn here but exported alongside "
        "this figure, is the disagreement in shape itself; additive in log space is "
        "multiplicative in kelvin, so that residual is a percentage of the fitted value. F32 "
        f"draws the single source time $t_s={published['source_time']:.3g}$, the evaluated "
        f"snapshot nearest {source_fraction:g} of the {published['horizon']:.3g} horizon: that "
        f"figure is one column of this surface. Over the evaluated cells: {numbers}. Because the domain is "
        "triangular the fit is an unbalanced design and these scalars are weighted by cell "
        "coverage, so they describe this surface and are not the balanced-window quantities "
        "reported by scripts/inspect_source_time_sweep.py, which is where the typicality of "
        "the marked column is tested. No interval is drawn; a per-cell median carries "
        "sampling error that this rendering does not show. The operator predicts each pair "
        "directly from its source snapshot, so variation along the vertical axis is "
        "lead-time-dependent difficulty, not error accumulated through a rollout. Evaluated "
        f"model seeds: {seed_note}. "
    )
    if surface["unequal_seed_counts"]:
        caption += "Seed counts differ by benchmark; the amount of training-seed averaging is unequal. "
    if any(surface["seed_counts"][b] < stats.MIN_SEEDS_FOR_SPREAD for b in order):
        caption += (
            f"Fewer than {stats.MIN_SEEDS_FOR_SPREAD} model seeds are averaged, so the surface "
            "carries the idiosyncrasies of the particular trained models alongside the "
            "behaviour of the operator. ")
    caption += ("Each panel carries its own color scale, so colors are comparable within a "
                "benchmark and not across the two. The scales are linear in log10 RMSE; both "
                "time axes are linear.")
    if source.degradations:
        caption += (" This descriptive comparison has publication-cohort limitations detailed "
                    "in the provenance sidecar.")

    rows = []
    for benchmark in order:
        data = surface["surfaces"][benchmark]
        for i, (index, t_s) in enumerate(zip(surface["source_indices"],
                                             surface["source_times"])):
            for j, (lag, lead) in enumerate(zip(surface["lags"], surface["lead_times"])):
                if not np.isfinite(data["surface_log10"][i, j]):
                    continue
                rows.append({
                    "benchmark": benchmark, "source_index": int(index),
                    "source_time": float(t_s), "lag": int(lag), "lead_time": float(lead),
                    "is_published_source": index == published["source_index"],
                    "median_rmse_K": float(10.0 ** data["surface_log10"][i, j]),
                    "additive_fit_rmse_K": float(10.0 ** data["additive_fit_log10"][i, j]),
                    "residual_pct": float(100.0 * (10.0 ** data["residual_log10"][i, j] - 1.0)),
                    "n_simulations": data["n_simulations"], "n_seeds": data["n_seeds"],
                })
    definition = {
        "space": "kelvin", "metric": "rmse_K", "title": title, "caption": caption,
        "benchmarks": order,
        "benchmark_selection": (
            "one benchmark whose error level moves strongly with source time and one whose "
            "error shape does; the two ways a single source time can fail to stand for the rest"),
        "source_time_marked": published,
        "domain": surface["domain"], "fit": surface["fit"],
        "aggregation": "cells -> pair sufficient statistics -> simulation RMSE at one (s, lag) "
                       "-> seed mean -> cohort median -> log10",
        "replication_unit": "sim_id within benchmark; equal simulation weights",
        "prediction_mode": "direct_pair; increases with lead are horizon difficulty, "
                           "not rollout error accumulation",
        "interval": None,
        "interval_omitted_because": "a per-cell median has sampling error a heatmap cannot "
                                    "display; intervals for the marked column are in F32 and "
                                    "the rank interval for its typicality is in the sweep",
        "summary": {b: {k: surface["surfaces"][b][k] for k in
                        ("interaction_pct", "level_spread_pct", "common_shape_fraction",
                         "n_simulations", "n_dropped", "n_cells")} for b in order},
        "summary_comparability": "weighted by cell coverage over a triangular domain; not the "
                                 "balanced-window values in scripts/inspect_source_time_sweep.py",
        "seed_counts": surface["seed_counts"], "seed_ids": surface["seeds"],
        "unequal_seed_counts": surface["unequal_seed_counts"],
        "seed_averaging": "mean of simulation errors; not an ensemble-averaged prediction",
        "protocols": surface["protocols"], "record_metadata": metadata,
        "statistics": rows,
        "degradations": [str(d) for d in source.degradations]
                        + [str(d) for d in surface["degradations"]],
    }
    return fig, None, definition


__all__ = [
    "global_field_error_vs_lead",
    "global_field_error_vs_itr",
    "global_field_error_fixed_source",
    "source_lead_error_surface",
    "BENCH_ORDER",
    "TARGET_REL_L2_PCT",
    "benchmark_order",
    "headline_accuracy",
    "jump_vs_lead",
    "node_jump_fidelity",
    "physical_contact_jump_vs_lead",
    "tail_reliability",
    "truth_pred_residual",
]
