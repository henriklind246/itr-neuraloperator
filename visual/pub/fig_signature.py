"""The signature figure: F06.

One simulation, three lead times, four rows. The point of the figure is that
the fourth row -- the contact jump ``R_c(y) q_n(y, t)`` -- is the physical
interface quantity, not the adjacent-node temperature difference the training
loss uses. The two are not interchangeable, and the paper should show the
physical one.

The columns are lead-time quantiles *of the selected simulation's own*
``t_bar`` distribution, so early / middle / late are meaningful within the case
rather than against a global scale that the case may not span.
"""

from __future__ import annotations

import numpy as np

from visual.pub import fields, jump, panels, records, select, stats, style
from visual.pub._blocked import blocked

# The benchmark this figure most wants, then the fallbacks.
#
# source and source_itr_sin are last despite having the most interesting R_c(y).
# They drive a localized internal source in the left slab across a resistive
# interface, and over the eval set the right slab's temperature rise never
# exceeds ~9% of the left's (median 2%). Rows 1-2 must share one temperature
# scale to be a truth/prediction comparison at all, so on that scale the entire
# right half of every panel collapses to a single flat block -- half the figure
# carrying no information. interfaces (median 84%) and forcing (median 35%)
# keep both slabs alive, and both still give a y-varying contact jump: the
# interface location moves in one and the flux profile s(y) varies in the other.
PREFERRED = ("interfaces", "forcing", "source_itr_sin", "source")

LEAD_QUANTILES = (0.1, 0.5, 0.9)

_NEEDED = (
    "needs schema-v2 test records plus a checkpoint for one of "
    f"{', '.join(PREFERRED)}; the figure evaluates the model, it does not read "
    "a precomputed field."
)


def _pick_benchmark(frames, source) -> str | None:
    """The most informative benchmark that has *both* records and a checkpoint."""
    for bench in PREFERRED:
        if bench in frames and fields.run_dirs(source, bench):
            return bench
    return None


def signature(*, source=None, spec=None, requirement=None):
    """F06 -- field and contact-jump evolution for one representative case."""
    frames = records.records_by_benchmark(source)
    if not frames:
        blocked(requirement, _NEEDED, key="F06_signature")
    bench = _pick_benchmark(frames, source)
    if bench is None:
        blocked(requirement, _NEEDED, key="F06_signature")

    frame = frames[bench]
    sims = stats.per_sim(frame, metrics=("rmse_K", "rel_l2_pct"))
    bundle = fields.bundle(source, bench)
    pick, restricted = fields.select_transverse_case(
        bundle, sims, frame.df, quantile=0.5, metric="rmse_K",
        min_leads=len(LEAD_QUANTILES))
    # The columns come from the same restricted pool. Transverse structure
    # decays as the solution relaxes toward a 1-D profile, so unrestricted lead
    # quantiles put the late columns exactly where there is nothing left to see
    # -- and the earliest one before the field has developed at all, which under
    # the shared temperature scale renders as a flat dark panel.
    columns = select.select_lead_columns(
        fields.transverse_pairs(bundle, frame.df), pick,
        lead_quantiles=LEAD_QUANTILES)

    cases = [fields.evaluate_case(bundle, pick.sim_id, s, (j,))
             for s, j in columns]
    stat = [fields.case_metrics(c)[0] for c in cases]

    truth = [c.truth[0] for c in cases]
    pred = [c.pred[0] for c in cases]
    resid = [c.residual[0] for c in cases]
    jumps = [c.contact_jumps() for c in cases]

    # One temperature scale for rows 1-2 and one symmetric residual scale for
    # row 3, both across every column: a column that were rescaled against
    # itself would hide exactly the growth with lead time the figure is about.
    temp_limits = style.shared_color_limits(*truth, *pred, mode="sequential")
    resid_limits = style.shared_color_limits(*resid, mode="diverging")

    width = getattr(spec, "width", "full_page")
    # Three rows of square maps plus a trailing row of jump curves. The single
    # colorbar column sits after the last map: rows 0-1 share the temperature
    # scale, row 2 carries the residual scale on its own.
    last = len(cases) - 1
    grid = panels.map_grid(style.WIDTHS_IN[width], 3, n_maps=len(cases),
                           colorbar_after=(last,),
                           colorbar_rows={last: ((0, 1), (2,))},
                           extra_rows=1, extra_row_height=0.78)
    fig, axes = grid.fig, grid.maps
    y_grid = np.asarray(cases[0].y_grid, dtype=float)

    for col, case in enumerate(cases):
        iface = case.interface_x
        t_ax = axes[0][col]
        im_t = panels.field_map(t_ax, truth[col], temp_limits, interface_x=iface,
                                title=f"$\\bar t$ = {case.lead_times[0]:.3f}")
        panels.field_map(axes[1][col], pred[col], temp_limits, interface_x=iface)
        im_r = panels.field_map(axes[2][col], resid[col], resid_limits,
                                interface_x=iface)

        j_true, j_pred = jumps[col]
        ax_j = grid.rows[0][col]
        ax_j.plot(y_grid, j_true[0], color="0.15", linewidth=1.0, label="FV")
        ax_j.plot(y_grid, j_pred[0], color=style.benchmark_color(bench),
                  linewidth=1.0, linestyle="--", label="FNO")
        ax_j.axhline(0.0, color="0.6", linewidth=0.5, zorder=0)
        ax_j.set_xlabel("$y$")

        axes[0][col].text(
            0.03, 0.97,
            f"$t_s$={case.t_source:.3f}  $t_j$={case.t_targets[0]:.3f}",
            transform=axes[0][col].transAxes, va="top", ha="left", fontsize=5.5,
            color="w",
            bbox=dict(facecolor="0.1", alpha=0.55, pad=1.0, edgecolor="none"))
        axes[2][col].text(
            0.03, 0.97,
            f"rel. $L_2$={stat[col]['rel_l2_pct']:.2f}%\n"
            f"RMSE={stat[col]['rmse_K']:.3g} K",
            transform=axes[2][col].transAxes, va="top", ha="left", fontsize=5.5,
            bbox=dict(facecolor="w", alpha=0.7, pad=1.0, edgecolor="none"))
        ax_j.text(
            0.97, 0.03,
            f"jump RMSE={stat[col]['contact_jump_rmse_K']:.3g} K\n"
            f"of {stat[col]['contact_jump_rms_K']:.3g} K rms",
            transform=ax_j.transAxes, va="bottom", ha="right", fontsize=5.5,
            bbox=dict(facecolor="w", alpha=0.7, pad=1.0, edgecolor="none"))

        for row in range(3):
            # Rows 0-2 are the same (x, y) grid, so only the bottom map row
            # carries the x axis; repeating it three times is noise.
            axes[row][col].set_xlabel("$x$" if row == 2 else "")
            # tick_params, not set_xticklabels: these axes keep an automatic
            # locator, which regenerates labels over any fixed list at draw
            # time.
            if row < 2:
                axes[row][col].tick_params(labelbottom=False)
            if col:
                axes[row][col].tick_params(labelleft=False)

    # The jump row shares one y scale for the same reason rows 1-2 share one
    # temperature scale: per-column autoscaling would hide the decay of the
    # jump amplitude with lead time, which is half of what the row shows.
    lo = min(ax.get_ylim()[0] for ax in grid.rows[0])
    hi = max(ax.get_ylim()[1] for ax in grid.rows[0])
    for col, ax_j in enumerate(grid.rows[0]):
        ax_j.set_ylim(lo, hi)
        if col:
            ax_j.tick_params(labelleft=False)

    for row, label in enumerate(("FV truth", "FNO prediction",
                                 "$T_{\\rm FNO}-T_{\\rm FV}$")):
        axes[row][0].set_ylabel(label, fontsize=7)
    grid.rows[0][0].set_ylabel(f"${jump.rc_symbol(bench)}\\,q_n(y,t)$",
                               fontsize=7)
    grid.rows[0][0].legend(fontsize=5.5, loc="upper left", ncol=1,
                           handletextpad=0.3, borderpad=0.2, labelspacing=0.2)

    panels.attach_colorbar(fig, im_t, axes[0][last], "$T$ [K]",
                           cax=grid.colorbars[last][0])
    panels.attach_colorbar(fig, im_r, axes[2][last],
                           "$T_{\\rm FNO}-T_{\\rm FV}$ [K]",
                           cax=grid.colorbars[last][1])

    fig.suptitle(
        f"{bench}: simulation {pick.sim_id} "
        f"(rank {pick.rank_in_sims}/{pick.n_sims} by RMSE)", fontsize=8)

    metric_definition = {
        "space": "kelvin",
        "benchmark": bench,
        "selection": "median simulation by pooled rmse_K; columns at lead-time "
                     f"quantiles {LEAD_QUANTILES} of that simulation's own t_bar",
        "transverse_restriction": fields.transverse_note(restricted),
        "transverse_minimum": fields.MIN_TRANSVERSE_FRACTION,
        "jump": jump.JUMP_DEFINITION,
        "color_scales": "rows 1-2 share one sequential temperature scale across "
                        "all columns; row 3 shares one symmetric diverging "
                        "scale; row 4 shares one y scale",
        "temperature_limits": [temp_limits.vmin, temp_limits.vmax],
        "residual_limits": [resid_limits.vmin, resid_limits.vmax],
        "columns": [
            {"s": int(s), "j": int(j), "sim_id": int(pick.sim_id), **m}
            for (s, j), m in zip(columns, stat)
        ],
    }
    return fig, pick, metric_definition


__all__ = ["signature"]
