"""Axes-level renderers shared by publication figures.

Every function here draws into an axes that the caller owns. Nothing in this
module reads a file or reduces a records frame: loading belongs to
``visual/pub/records.py`` and every statistic comes from ``visual/pub/stats.py``.
A panel receives values already computed and only decides how they look.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _field

import matplotlib as mpl
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np

from visual.pub import style

# ============================================================
# Schematic primitives
# ============================================================


def blank_canvas(ax, *, xlim: tuple[float, float] = (0.0, 1.0),
                 ylim: tuple[float, float] = (0.0, 1.0),
                 equal: bool = False) -> None:
    """Turn an axes into a bare drawing surface for a schematic."""
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if equal:
        ax.set_aspect("equal")


def schematic_box(ax, xy: tuple[float, float], width: float, height: float,
                  label: str = "", *, facecolor: str = "0.93",
                  edgecolor: str = "0.35", fontsize: float = 6.5,
                  radius: float = 0.02, zorder: float = 2
                  ) -> tuple[float, float]:
    """Draw a rounded labelled box; return its center."""
    box = mpl.patches.FancyBboxPatch(
        (xy[0], xy[1]), width, height,
        boxstyle=mpl.patches.BoxStyle("Round", pad=0.0, rounding_size=radius),
        facecolor=facecolor, edgecolor=edgecolor, linewidth=0.8, zorder=zorder,
    )
    ax.add_patch(box)
    center = (xy[0] + width / 2.0, xy[1] + height / 2.0)
    if label:
        ax.text(center[0], center[1], label, ha="center", va="center",
                fontsize=fontsize, zorder=zorder + 1, linespacing=1.25)
    return center


def schematic_arrow(ax, start: tuple[float, float], end: tuple[float, float],
                    *, label: str = "", color: str = "0.35",
                    linewidth: float = 0.9, fontsize: float = 6.0,
                    style_str: str = "-|>", connection: str = "arc3,rad=0.0",
                    label_offset: tuple[float, float] = (0.0, 0.02)) -> None:
    """Draw a labelled arrow between two points in axes data coordinates."""
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style_str, color=color,
                                linewidth=linewidth,
                                connectionstyle=connection,
                                shrinkA=1.0, shrinkB=1.0), zorder=3)
    if label:
        mid = ((start[0] + end[0]) / 2.0 + label_offset[0],
               (start[1] + end[1]) / 2.0 + label_offset[1])
        ax.text(mid[0], mid[1], label, ha="center", va="bottom",
                fontsize=fontsize, color=color, zorder=4)


# ============================================================
# Field maps
# ============================================================


def field_map(ax, field: np.ndarray, limits: style.ColorLimits, *,
              extent: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
              interface_x: float | None = None, title: str = "",
              xlabel: str = "", ylabel: str = "", tick_bins: int = 4):
    """Render an ``(Nx, Ny)`` field with x horizontal and y vertical.

    Trajectory arrays are indexed ``[x, y]``, so the array is transposed once,
    here, rather than at every call site.

    The aspect ratio is locked to the extent. The domain is physically square
    and an anisotropic stretch is not a neutral choice: it changes the apparent
    shape of a heating patch and the apparent angle at which a front meets the
    interface, which are the things these panels exist to show.

    ``tick_bins`` is the tick density, and it has to fall as the panel does.
    Tick labels are at a fixed point size, so a "0.00" is the same 0.21 in wide
    whatever the map measures; four of them are comfortable across 1.4 in and
    touching across 1.0 in. Lower it rather than shrinking the type, which is
    the thing a small panel most needs to keep.
    """
    arr = np.asarray(field, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"field_map expects a 2-D (Nx, Ny) array, got {arr.shape}")
    im = ax.imshow(arr.T, origin="lower", extent=extent, aspect="equal",
                   **limits.imshow_kwargs())
    # Prune the upper tick on both axes. Field maps are tiled edge to edge, and
    # the default locator puts a label at the very end of each axis, so the last
    # label of one panel lands on the first label of its neighbour and reads as
    # "1.000.00". Pruning costs one tick and removes the collision outright.
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(mpl.ticker.MaxNLocator(nbins=tick_bins,
                                                      prune="upper"))
    if interface_x is not None:
        ax.axvline(interface_x, color="w", linewidth=0.8, alpha=0.85)
        ax.axvline(interface_x, color="k", linewidth=0.4, alpha=0.6)
    if title:
        ax.set_title(title, fontsize=7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return im


def attach_colorbar(fig, mappable, ax, label: str, *, pad: float = 0.02,
                    fraction: float = 0.046, cax=None,
                    label_x: float | None = None):
    """Attach a thin labelled colorbar, either beside ``ax`` or into ``cax``.

    Passing ``cax`` is the form to use next to field maps. The ``ax=`` form
    takes its width *out of* the axes it is attached to, so in a grid of maps
    the column carrying a colorbar ends up narrower than its neighbours -- which
    is exactly the misalignment a reader sees as sloppy typesetting.

    ``label_x`` is for a bar that has a *neighbour* to its right. Matplotlib
    places the rotated label past the tick labels, so its position is a function
    of the data range: a bar reading ``300.20`` pushes the label a third of an
    inch further out than one reading ``302``, and once it clears the reserved
    gap the label is drawn on top of the next map. Pass
    :attr:`MapGrid.cbar_label_x` to pin it inside the gap instead.
    """
    if cax is not None:
        cbar = fig.colorbar(mappable, cax=cax)
    else:
        cbar = fig.colorbar(mappable, ax=ax, pad=pad, fraction=fraction)
    cbar.set_label(label, fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    cbar.outline.set_linewidth(0.6)
    if label_x is not None:
        cbar.ax.yaxis.set_label_coords(label_x, 0.5)
    return cbar


# ------------------------------------------------------------ grids of maps

# Absolute inch margins for a grid of square field maps. These are measured
# against rendered output, not derived: they are the smallest values that keep
# the row label, the tick labels and the colorbar labels clear of each other at
# 300 dpi and 7 pt type.
_LEFT_IN = 0.72        # two-line row label + y tick labels
_TOP_IN = 0.46         # suptitle + column titles
_BOTTOM_IN = 0.58      # x tick labels + x label + the provenance footer
_ROW_GAP_IN = 0.14
_COL_GAP_IN = 0.10
_CBAR_W_IN = 0.075
_CBAR_GAP_IN = 0.05    # map -> its own colorbar
_CBAR_LABEL_IN = 0.50  # colorbar tick labels + rotated label + breathing room
_CBAR_LABEL_W_IN = 0.16   # a rotated 7 pt label plus its right-hand margin
_EXTRA_ROW_GAP_IN = 0.34   # last map row's x tick labels + x label


@dataclass(frozen=True)
class MapGrid:
    """Axes for a grid of square field maps, placed at absolute inch offsets."""

    fig: object
    maps: list = _field(default_factory=list)     # maps[row][col]
    # colorbars[map_column] -> one axes per row group that column's bar serves.
    colorbars: dict = _field(default_factory=dict)
    rows: list = _field(default_factory=list)     # trailing rows[r][col]
    map_size_in: float = 0.0
    # The realized figure width, which is below the requested one when
    # ``max_map_in`` bound the map size.
    width_in: float = 0.0
    # Pass to attach_colorbar(label_x=...) for a bar that has a map to its
    # right; see that function for why the automatic placement is not safe.
    cbar_label_x: float = 0.0


def map_grid(width_in: float, n_rows: int, *, n_maps: int = 3,
             colorbar_after: tuple[int, ...] = (1, 2),
             colorbar_span: tuple[int, ...] = (),
             colorbar_rows: dict | None = None,
             extra_rows: int = 0, extra_row_height: float = 1.0,
             extra_row_inset_in: float = 0.0,
             max_map_in: float | None = None,
             left_in: float = _LEFT_IN, right_in: float = 0.10,
             top_in: float = _TOP_IN,
             bottom_in: float = _BOTTOM_IN,
             row_gap_in: float = _ROW_GAP_IN,
             cbar_label_in: float = _CBAR_LABEL_IN) -> MapGrid:
    """Lay out ``n_rows x n_maps`` square field maps with dedicated colorbars.

    ``tight_layout`` cannot do this. It distributes *residual* space after the
    fact, and an ``aspect="equal"`` image does not fill the axes it was given,
    so the constrained dimension collapses toward the panels and the rest of the
    row opens up as whitespace. The result is the ragged, loosely-packed look
    these figures had. Solving the geometry up front -- map width from the
    figure width, then figure height from the map width -- is the only way the
    maps come out both square and flush.

    ``colorbar_after`` names the map columns that carry a colorbar in a column
    of its own, so every map stays the same width. Which rows each bar serves is
    the caller's statement about which panels share a scale:

    - by default one bar per row, i.e. a per-row scale;
    - ``colorbar_span`` names columns whose scale is shared down the whole grid,
      drawn once as a single tall bar;
    - ``colorbar_rows`` maps a column to explicit row groups, for the case where
      *some* rows share a scale -- ``{2: ((0, 1), (2,))}`` is "rows 0 and 1
      share one bar; row 2 gets its own".

    ``extra_rows`` appends full-width rows of panels below the maps, each
    ``extra_row_height`` map-widths tall.

    ``extra_row_inset_in`` narrows those rows from the left by that many inches,
    the same amount in every column so they stay equal in width. Maps need no
    room between columns because only the first carries tick labels; a line plot
    whose columns are on different y scales needs labels on all of them, and
    ``_COL_GAP_IN`` alone would run them into the neighbouring panel.

    ``max_map_in`` caps the solved map size and narrows the figure to match.
    Map size falls out of the width divided by the column count, so the same
    call that is well proportioned at four columns produces page-tall maps at
    two; the cap turns a fixed width into a maximum one for the sparse case.

    ``cbar_label_in`` is the space reserved to the right of each bar for its
    tick labels and its rotated label. The default suits a bar reading two or
    three significant figures; a grid whose bars carry six-character ticks
    (``300.20``) needs more, or the label lands on the next map.

    ``top_in``, ``bottom_in`` and ``row_gap_in`` are the whole of the grid's
    height budget besides the maps themselves, and they set how tall the figure
    is relative to its width. The defaults reserve room for a suptitle and a
    provenance footer; a grid that carries neither is paying for both, and on a
    three-row grid that is close to half an inch of height it does not use.
    """
    cbars = tuple(sorted(set(colorbar_after)))
    unknown = set(colorbar_span) - set(cbars)
    if unknown:
        raise ValueError(f"colorbar_span columns {sorted(unknown)} are not in "
                         f"colorbar_after {list(cbars)}")
    unknown = set(colorbar_rows or {}) - set(cbars)
    if unknown:
        raise ValueError(f"colorbar_rows columns {sorted(unknown)} are not in "
                         f"colorbar_after {list(cbars)}")

    every_row = tuple(range(n_rows))
    groups: dict[int, tuple[tuple[int, ...], ...]] = {}
    for col in cbars:
        if colorbar_rows and col in colorbar_rows:
            groups[col] = tuple(tuple(g) for g in colorbar_rows[col])
        elif col in colorbar_span:
            groups[col] = (every_row,)
        else:
            groups[col] = tuple((r,) for r in every_row)

    # A column plan in two currencies: inches for anything of fixed size, and
    # "map widths" for the flexible columns. Solve for the map width, then read
    # the plan back to place the axes.
    plan: list[tuple[str, float]] = []
    for col in range(n_maps):
        if col:
            # A colorbar's label gap already separates the previous map from
            # this one; adding the usual column gap on top double-spaces it.
            plan.append(("gap", 0.0 if (col - 1) in cbars else _COL_GAP_IN))
        plan.append(("map", 1.0))
        if col in cbars:
            plan.append(("gap", _CBAR_GAP_IN))
            plan.append(("cbar", _CBAR_W_IN))
            plan.append(("gap", cbar_label_in))

    fixed = left_in + right_in + sum(v for kind, v in plan
                                     if kind in ("gap", "cbar"))
    units = sum(v for kind, v in plan if kind == "map")
    size = (width_in - fixed) / units
    if size <= 0.0:
        raise ValueError(
            f"map_grid: {n_maps} maps plus fixed furniture need more than "
            f"{width_in:.2f} in of width")
    if max_map_in is not None and size > max_map_in:
        size = max_map_in
        width_in = fixed + units * size

    tail = 0.0
    if extra_rows:
        tail = (_EXTRA_ROW_GAP_IN + extra_rows * size * extra_row_height
                + (extra_rows - 1) * row_gap_in)
    height_in = (top_in + bottom_in + n_rows * size
                 + (n_rows - 1) * row_gap_in + tail)
    fig = plt.figure(figsize=(width_in, height_in))

    def rect(x_in, y_in, w_in, h_in):
        return [x_in / width_in, y_in / height_in,
                w_in / width_in, h_in / height_in]

    def row_bottom(row: int) -> float:
        # Rows run top-down; matplotlib measures from the bottom.
        return height_in - top_in - (row + 1) * size - row * row_gap_in

    # x offsets are the same on every row, so walk the plan once.
    x = left_in
    map_x: list[float] = []
    cbar_x: list[float] = []
    for kind, value in plan:
        if kind == "gap":
            x += value
        elif kind == "cbar":
            cbar_x.append(x)
            x += value
        else:
            map_x.append(x)
            x += size * value

    maps = [[fig.add_axes(rect(mx, row_bottom(row), size, size))
             for mx in map_x] for row in range(n_rows)]

    colorbars: dict[int, list] = {}
    for col, cx in zip(cbars, cbar_x):
        bars = []
        for group in groups[col]:
            bottom = row_bottom(max(group))
            top = row_bottom(min(group)) + size
            bars.append(fig.add_axes(rect(cx, bottom, _CBAR_W_IN, top - bottom)))
        colorbars[col] = bars

    rows: list[list] = []
    base = row_bottom(n_rows - 1) - _EXTRA_ROW_GAP_IN
    inset = min(max(extra_row_inset_in, 0.0), 0.5 * size)
    for r in range(extra_rows):
        h = size * extra_row_height
        y = base - (r + 1) * h - r * row_gap_in
        rows.append([fig.add_axes(rect(mx + inset, y, size - inset, h))
                     for mx in map_x])

    # set_label_coords anchors the rotated label's left edge, so subtracting a
    # label width puts its far edge at the far edge of the reserved gap.
    label_x = 1.0 + (cbar_label_in - _CBAR_LABEL_W_IN) / _CBAR_W_IN
    return MapGrid(fig=fig, maps=maps, colorbars=colorbars, rows=rows,
                   map_size_in=size, width_in=width_in, cbar_label_x=label_x)


# ============================================================
# Convergence panels
# ============================================================


@dataclass(frozen=True)
class ConvergenceFit:
    """The log-log regression a convergence panel drew."""

    slope: float
    intercept: float


def convergence_loglog(ax, steps, errors, *, xlabel: str,
                       reference_order: int = 2, title: str = "",
                       color: str = "#0072B2", reference_symbol: str = "h",
                       ylabel: str = "$L_2$ error") -> ConvergenceFit:
    """Plot ``error`` against ``step`` in log-log with a fitted slope.

    The fitted slope is the observed order of accuracy; the dashed line is the
    theoretical ``O(h^reference_order)`` anchored at the finest step, so a
    second-order scheme shows two parallel lines.
    """
    steps = np.asarray(steps, dtype=float)
    errors = np.asarray(errors, dtype=float)
    if steps.size != errors.size or steps.size < 2:
        raise ValueError("convergence_loglog needs >= 2 matched (step, error) values")

    log_s, log_e = np.log10(steps), np.log10(errors)
    slope, intercept = np.polyfit(log_s, log_e, 1)

    ax.loglog(steps, errors, "o", color="k", markersize=3.2, zorder=3,
              label="MMS")
    ax.loglog(steps, 10 ** np.polyval([slope, intercept], log_s), "-",
              color=color, linewidth=1.1, zorder=2,
              label=f"fit $p$ = {slope:.2f}")
    ref = errors[-1] * (steps / steps[-1]) ** reference_order
    ax.loglog(steps, ref, "--", color="0.4", linewidth=0.8, zorder=1,
              label=f"$O({reference_symbol}^{reference_order})$")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=7)
    ax.grid(True, which="both")
    ax.legend(fontsize=5.5, loc="best")
    return ConvergenceFit(slope=float(slope), intercept=float(intercept))


def order_estimates(ax, midpoints, orders, *, xlabel: str, target: float = 2.0,
                    title: str = "", color: str = "#0072B2",
                    ylabel: str = "observed order $p$") -> None:
    """Plot pairwise observed orders against a horizontal design order."""
    midpoints = np.asarray(midpoints, dtype=float)
    orders = np.asarray(orders, dtype=float)
    ax.semilogx(midpoints, orders, "o-", color=color, markersize=3.2,
                linewidth=1.0, zorder=3)
    ax.axhline(target, color="0.4", linestyle="--", linewidth=0.8, zorder=1)
    ax.text(0.02, target, f"design $p$ = {target:g}", transform=ax.get_yaxis_transform(),
            ha="left", va="top", fontsize=5.5, color="0.4")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=7)
    ax.grid(True, which="both")
    # A converged scheme puts every estimate within a hair of the design order,
    # which autoscales into a misleadingly dramatic axis; hold a floor on the
    # range so the reader sees "flat at p" rather than a magnified wiggle.
    lo = min(float(np.min(orders)), target)
    hi = max(float(np.max(orders)), target)
    if hi - lo < 0.4:
        mid = 0.5 * (lo + hi)
        lo, hi = mid - 0.2, mid + 0.2
    ax.set_ylim(lo - 0.05, hi + 0.05)
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())


def pairwise_orders(steps, errors) -> tuple[np.ndarray, np.ndarray]:
    """Geometric-mean step midpoints and the order between consecutive levels.

    A local convergence rate, not a statistic over a sample; it lives here
    rather than in ``stats.py`` because it reduces nothing across replicates.
    """
    steps = np.asarray(steps, dtype=float)
    errors = np.asarray(errors, dtype=float)
    mids = np.sqrt(steps[:-1] * steps[1:])
    orders = np.log(errors[:-1] / errors[1:]) / np.log(steps[:-1] / steps[1:])
    return mids, orders


# ============================================================
# Rollout and resolution-study panels
# ============================================================


def rollout_delta_panel(ax, curves, *, title: str = "", ylabel: str = "",
                        legend: bool = False) -> None:
    """Paired autoregressive-minus-direct curves with simulation IQR bands."""
    colors = {2: "#0072B2", 4: "#E69F00", 8: "#D55E00"}
    linestyles = {2: "-", 4: "--", 8: ":"}
    markers = {2: "o", 4: "s", 8: "^"}
    for substeps in (2, 4, 8):
        curve = curves[substeps]
        color = colors[substeps]
        ax.fill_between(
            curve.lead_times, curve.q25_pp, curve.q75_pp,
            color=color, alpha=0.13, linewidth=0, zorder=1,
        )
        ax.plot(
            curve.lead_times, curve.median_pp,
            color=color, linestyle=linestyles[substeps],
            marker=markers[substeps], markersize=3.2,
            markerfacecolor="white", markeredgewidth=0.8,
            linewidth=1.1, label=f"K={substeps}", zorder=3,
        )
    ax.axhline(0.0, color="0.3", linewidth=0.8, zorder=2)
    first = curves[2]
    ax.set_xticks(first.lead_times)
    ax.set_xlabel("Exact lead time")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=7)
    ax.grid(True, alpha=0.25)
    if legend:
        ax.legend(fontsize=5.5, loc="best", title="rollout substeps",
                  title_fontsize=5.5)


def resolution_curve_panel(ax, studies, drift, *, value_attr: str,
                           title: str = "", ylabel: str = "",
                           legend: bool = False) -> None:
    """Two fixed-checkpoint resolution curves and metric-matched FV drift."""
    series = (
        (studies.original, "#0072B2", "-", "o"),
        (studies.material_side, "#E69F00", "--", "s"),
    )
    for study, color, linestyle, marker in series:
        ax.plot(
            study.resolutions, getattr(study, value_attr),
            color=color, linestyle=linestyle, marker=marker,
            markersize=3.4, markerfacecolor="white", markeredgewidth=0.8,
            linewidth=1.1, label=study.label, zorder=3,
        )
    ax.plot(
        drift.resolutions, getattr(drift, value_attr),
        color="0.35", linestyle=":", marker="^", markersize=3.2,
        markerfacecolor="white", markeredgewidth=0.8, linewidth=1.0,
        label=f"FV drift to N={drift.reference_resolution}", zorder=2,
    )
    ax.set_xticks(studies.original.resolutions)
    ax.set_xlabel(r"Evaluation grid $N \times N$")
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0.0)
    if title:
        ax.set_title(title, fontsize=7)
    ax.grid(True, alpha=0.25)
    if legend:
        ax.legend(fontsize=5.3, loc="best")


# ============================================================
# Paired inverse-sweep panels
# ============================================================


def paired_sensor_sweep_panel(
    ax,
    curve,
    *,
    color: str,
    ylabel: str,
    title: str = "",
    reference=None,
    legend: bool = False,
) -> None:
    """Draw paired cases and median/IQR at discrete sensor counts.

    Connecting segments join the same inversion case across adjacent protocol
    levels; they are not a fitted or interpolated sensor-count trend.
    """
    x = np.arange(len(curve.sensor_counts), dtype=float)
    for index, values in enumerate(curve.case_values):
        ax.plot(
            x, values, color="0.72", linewidth=0.55, alpha=0.65,
            marker="o", markersize=1.8, markeredgewidth=0,
            label="paired cases" if index == 0 else None, zorder=1,
        )
    ax.vlines(
        x, curve.q25, curve.q75, color=color, linewidth=2.0,
        alpha=0.75, label="IQR", zorder=3,
    )
    ax.plot(
        x, curve.median, color=color, linewidth=1.15, marker="o",
        markersize=4.2, markeredgecolor="black", markeredgewidth=0.4,
        label="median", zorder=4,
    )

    if reference is not None:
        ax.plot(
            x, reference.median, color="0.25", linestyle="--", linewidth=0.9,
            marker="s", markersize=2.8, markerfacecolor="white",
            label="noise std.", zorder=2,
        )
    if curve.censored is not None and np.any(curve.censored):
        rows, columns = np.where(curve.censored)
        ax.plot(
            x[columns], curve.case_values[rows, columns], linestyle="none",
            marker="x", color="black", markersize=3.2, markeredgewidth=0.8,
            label="CI excluded", zorder=5,
        )

    ax.set_xticks(x, [str(int(value)) for value in curve.sensor_counts])
    ax.set_xlim(-0.35, len(x) - 0.65)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("Sensors")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=7)
    ax.grid(True, axis="y", alpha=0.25)
    if legend:
        ax.legend(fontsize=5.3, loc="best")


def rc_profile_recovery_panel(
    ax,
    y,
    rc_true,
    rc_hat,
    *,
    color: str,
    title: str = "",
    xlabel: str = "$y$",
    ylabel: str = "",
    legend: bool = False,
) -> None:
    """Overlay a true and a recovered interfacial resistance profile.

    Exactly two curves: the shrinking gap between them is the whole message, so
    the caller must set identical ``y`` and ``R_c`` limits on every panel it
    draws. Rescaling between panels would make a worse reconstruction look
    equally good.

    ``xlabel`` is settable so a stacked column of these can carry one shared
    label on its bottom panel instead of repeating it three times.
    """
    y = np.asarray(y, dtype=float)
    rc_true = np.asarray(rc_true, dtype=float)
    rc_hat = np.asarray(rc_hat, dtype=float)
    if y.shape != rc_true.shape or y.shape != rc_hat.shape:
        raise ValueError(
            f"y {y.shape}, rc_true {rc_true.shape} and rc_hat {rc_hat.shape} "
            "must share one shape"
        )

    ax.plot(y, rc_true, color=style.GREY, linewidth=1.3, linestyle="-",
            label="true", zorder=2)
    ax.plot(y, rc_hat, color=color, linewidth=1.3, linestyle="--",
            label="recovered", zorder=3)
    ax.set_xlim(float(y.min()), float(y.max()))
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=7)
    ax.grid(True, alpha=0.25)
    if legend:
        # Larger than the 5.3-5.5 pt legends elsewhere in this module. Those
        # label five or six series and are read as a key; this one labels two
        # curves whose solid/dashed distinction *is* the figure, so it has to
        # survive the reduction to a single column.
        ax.legend(fontsize=7.0, loc="best", handlelength=2.4,
                  borderpad=0.4, labelspacing=0.35)


def profile_likelihood_panel(
    ax,
    curve,
    *,
    color: str,
    xlabel: str,
    title: str = "",
    ylabel: str = "",
    threshold: float,
    legend: bool = False,
) -> None:
    """Draw one parameter's profile likelihood with its threshold and interval.

    ``curve`` is a :class:`~visual.pub.records.ProfileCurve`. A ``bound_limited``
    interval is annotated rather than drawn as a closed span, because a closed
    bracket over an interval whose crossing was never found claims a constraint
    the data did not supply.
    """
    values = np.asarray(curve.values, dtype=float)
    delta = np.asarray(curve.delta_ell, dtype=float)
    censored = bool(curve.bound_limited)

    ax.plot(values, delta, color=color, linewidth=1.2, marker="o",
            markersize=2.4, markeredgewidth=0, label=r"profile $\Delta\ell$",
            zorder=3)
    ax.axhline(threshold, color="0.25", linestyle="--", linewidth=0.9,
               label="95% threshold", zorder=2)
    ax.axvline(curve.theta_true, color=style.GREY, linestyle=":",
               linewidth=1.0, label="true", zorder=2)
    ax.plot([curve.theta_hat], [0.0], marker="D", color=color, markersize=4.0,
            markeredgecolor="black", markeredgewidth=0.4, linestyle="none",
            label="recovered", zorder=5)

    if not censored:
        ax.axvspan(curve.ci_low, curve.ci_high, color=color, alpha=0.12,
                   linewidth=0, label="95% interval", zorder=1)
    else:
        ax.annotate(
            "interval bound-limited", xy=(0.5, 0.94), xycoords="axes fraction",
            ha="center", va="top", fontsize=5.5, color="black",
        )

    ax.set_xlim(float(values.min()), float(values.max()))
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel or r"$\Delta\ell$")
    if title:
        ax.set_title(title, fontsize=7)
    ax.grid(True, alpha=0.25)
    if legend:
        ax.legend(fontsize=5.3, loc="best")


def joint_nll_contour_panel(
    ax,
    region,
    *,
    theta_hat,
    theta_true,
    labels,
    title: str = "",
    legend: bool = False,
    levels: int = 9,
):
    """Contour a 2-parameter joint likelihood region over its parameter plane.

    ``region`` is either a measured :class:`~visual.pub.records.JointNLLGrid` or
    a :class:`~visual.pub.stats.LaplaceJointRegion`; both expose ``axes``,
    ``delta_ell``, ``threshold`` and ``approximate``. ``nan`` entries mark pairs
    the parameterization cannot produce and are left blank, so the admissible set
    reads as the triangle it is instead of a filled rectangle.

    Returns the filled-contour mappable so the caller can attach a colorbar.
    """
    first, second = (np.asarray(axis, dtype=float) for axis in region.axes)
    delta = np.asarray(region.delta_ell, dtype=float)
    if delta.shape != (first.size, second.size):
        raise ValueError(
            f"delta_ell {delta.shape} does not match its axes "
            f"({first.size}, {second.size})"
        )

    # Contour in (first, second) reading order: delta_ell rows index the first
    # parameter, so it is transposed once here rather than at every call site.
    grid_x, grid_y = np.meshgrid(first, second, indexing="xy")
    surface = delta.T
    ceiling = float(np.nanmax(surface))
    top = max(ceiling, 1.5 * region.threshold)
    filled = ax.contourf(
        grid_x, grid_y, surface, levels=np.linspace(0.0, top, int(levels)),
        cmap=style.CMAP_SEQ_ORDINAL, extend="max", zorder=1,
    )
    if ceiling > region.threshold:
        ax.contour(
            grid_x, grid_y, surface, levels=[region.threshold], colors="white",
            linewidths=1.1, zorder=2,
        )
    # Both markers carry a dark outline. They may land on the viridis fill or on
    # the blank inadmissible region, and a marker that vanishes against one of
    # the two backgrounds reads as an absent value rather than a plotted one.
    halo = [patheffects.withStroke(linewidth=2.0, foreground="black")]
    ax.plot(
        [theta_true[0]], [theta_true[1]], marker="x", color="white",
        markersize=5.0, markeredgewidth=1.3, linestyle="none", label="true",
        path_effects=halo, zorder=4,
    )
    ax.plot(
        [theta_hat[0]], [theta_hat[1]], marker="D", color="white",
        markersize=4.0, markeredgecolor="black", markeredgewidth=0.6,
        linestyle="none", label="recovered", zorder=5,
    )

    # Limits cover the grid *and* both markers. A true value that falls outside
    # the evaluated region is the finding, so it has to stay visible; cropping
    # to the grid alone would silently drop the marker and read as agreement.
    for axis, values, marker in (
        (ax.set_xlim, first, (theta_true[0], theta_hat[0])),
        (ax.set_ylim, second, (theta_true[1], theta_hat[1])),
    ):
        low = min(float(values.min()), *(float(m) for m in marker))
        high = max(float(values.max()), *(float(m) for m in marker))
        pad = 0.02 * (high - low)
        axis(low - pad, high + pad)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    if title:
        ax.set_title(title, fontsize=7)
    if legend:
        ax.legend(fontsize=5.3, loc="best", framealpha=0.85)
    return filled


# ============================================================
# Distribution panels
# ============================================================


def _finite(values) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    return arr[np.isfinite(arr)]


def summary_points(ax, summaries, *, labels=None, color: str = "#0072B2",
                   connect: bool = False, marker: str = "o",
                   ylabel: str = "", xlabel: str = "", title: str = "",
                   logy: bool = False, offset: float = 0.0,
                   label: str = "", show_ci: bool = True,
                   rotate_xticks: float = 0.0, x=None) -> np.ndarray:
    """Draw an ordered sequence of :class:`~visual.pub.stats.Summary` objects.

    Each entry gets a median marker, a thin IQR whisker, and a thick confidence
    bar. The CI is drawn only when the bootstrap actually produced one -- below
    :data:`~visual.pub.stats.MIN_SIMS_FOR_CI` simulations ``ci_method`` is
    ``"none"`` and no bar is drawn, so the reader cannot mistake an
    unestimated interval for a narrow one.

    ``x`` places the entries at given numeric coordinates instead of at
    successive integers. Use it whenever the stratum is a real quantity -- a
    time, a resistance, an interface position -- because evenly spacing unevenly
    sampled values distorts every slope the reader takes off the plot.

    Returns the x positions, so a caller can overlay seed dots or a second
    series at the same positions.
    """
    summaries = list(summaries)
    numeric_x = x is not None
    if numeric_x:
        x = np.asarray(x, dtype=float) + offset
        if x.size != len(summaries):
            raise ValueError("x must have one coordinate per summary")
    else:
        x = np.arange(len(summaries), dtype=float) + offset

    for xi, s in zip(x, summaries):
        if not np.isfinite(s.median):
            continue
        ax.vlines(xi, s.q25, s.q75, color=color, linewidth=0.9, alpha=0.55,
                  zorder=2)
        if show_ci and s.ci_method != "none" and np.isfinite(s.ci_lo):
            ax.vlines(xi, s.ci_lo, s.ci_hi, color=color, linewidth=2.6,
                      alpha=0.9, zorder=3)

    medians = np.array([s.median for s in summaries], dtype=float)
    if connect:
        ax.plot(x, medians, "-", color=color, linewidth=1.0, alpha=0.8, zorder=3)
    ax.plot(x, medians, marker, color=color, markersize=4.0,
            markeredgecolor="k", markeredgewidth=0.4, zorder=4,
            label=label or None)

    if labels is not None:
        ax.set_xticks(x if numeric_x else np.arange(len(summaries), dtype=float))
        ax.set_xticklabels(list(labels),
                           rotation=rotate_xticks,
                           ha="right" if rotate_xticks else "center")
    if not numeric_x:
        ax.set_xlim(-0.6, len(summaries) - 0.4)
    if logy:
        ax.set_yscale("log")
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title, fontsize=7)
    ax.grid(True, axis="y", alpha=0.25)
    return x


def seed_dots(ax, positions, values_by_position, *, color: str = "k",
              spread: float = 0.13, marker: str = "d",
              label: str = "") -> None:
    """Overlay one dot per model seed at each categorical position.

    Seed spread is shown as the individual values rather than as an error bar:
    with fewer than three seeds a standard error is not estimable, and this is
    the honest display in either case.
    """
    drawn = False
    for xi, values in zip(positions, values_by_position):
        vals = _finite(values)
        if vals.size == 0:
            continue
        if vals.size == 1:
            xs = np.array([xi])
        else:
            xs = xi + np.linspace(-spread, spread, vals.size)
        ax.plot(xs, vals, marker, color=color, markersize=2.6,
                markerfacecolor="none", markeredgewidth=0.7, linestyle="none",
                zorder=5, label=(label if not drawn else None))
        drawn = True


def ecdf_curves(ax, series, *, colors=None, xlabel: str = "",
                ylabel: str = "fraction of simulations", title: str = "",
                logx: bool = False, reference: float | None = None,
                reference_label: str = "") -> None:
    """Empirical CDFs over **per-simulation** values, one curve per group.

    An ECDF replaces the long-tail histograms of the current paper figures:
    a histogram of a heavy-tailed error distribution crushes the body to
    display the tail, while the ECDF shows both at full resolution and makes
    the exceedance fraction at any threshold directly readable.
    """
    from visual.pub import stats as _stats

    colors = colors or {}
    for name, values in series.items():
        x, f = _stats.ecdf(values)
        if x.size == 0:
            continue
        ax.step(x, f, where="post", linewidth=1.1,
                color=colors.get(name, None), label=f"{name} (n={x.size})")
    if reference is not None:
        ax.axvline(reference, color="0.4", linestyle="--", linewidth=0.8,
                   zorder=1)
        if reference_label:
            ax.text(reference, 0.02, f" {reference_label}", fontsize=5.5,
                    color="0.4", ha="left", va="bottom")
    if logx:
        ax.set_xscale("log")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=7)
    ax.grid(True, alpha=0.25)
    # Upper left, not lower right: an ECDF leaves that corner empty, and the
    # lower right is where the reference-threshold label sits.
    ax.legend(fontsize=5.5, loc="upper left")


def recovery_scatter(ax, truth, estimate, *, polish=None, failed=None,
                     color: str = "#0072B2", xlabel: str = "",
                     ylabel: str = "", title: str = "", logscale: bool = False
                     ) -> None:
    """Recovered parameter against its true value, with the identity line.

    ``failed`` marks non-converged inversions, which are drawn as open red
    circles rather than dropped: reporting only the cases that worked would
    report the accuracy of a set selected on the outcome.
    """
    truth = np.asarray(truth, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    ok = np.isfinite(truth) & np.isfinite(estimate)
    bad = np.zeros(truth.shape, dtype=bool) if failed is None else \
        np.asarray(failed, dtype=bool)

    ax.scatter(truth[ok & ~bad], estimate[ok & ~bad], s=14, color=color,
               edgecolor="k", linewidth=0.3, zorder=3, label="MAP")
    drawn = [truth[ok], estimate[ok]]
    if polish is not None:
        polish = np.asarray(polish, dtype=float)
        good = ok & np.isfinite(polish)
        ax.scatter(truth[good], polish[good], s=10, facecolor="none",
                   edgecolor="0.3", linewidth=0.6, zorder=4, label="FV polish")
        # The polish estimate sets the limits too; leaving it out clips markers
        # against the frame, which reads as a missing case rather than a large
        # error.
        drawn.append(polish[good])
    if bad.any():
        ax.scatter(truth[ok & bad], estimate[ok & bad], s=18, facecolor="none",
                   edgecolor="#D55E00", linewidth=0.8, zorder=5,
                   label="not converged")

    finite = np.concatenate(drawn)
    if finite.size:
        lo, hi = float(finite.min()), float(finite.max())
        pad = 0.08 * (hi - lo) if hi > lo else max(abs(hi), 1.0) * 0.08
        span = (lo - pad, hi + pad)
        ax.plot(span, span, color="0.5", linewidth=0.7, linestyle="--",
                zorder=1)
        ax.set_xlim(*span)
        ax.set_ylim(*span)
    if logscale:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=7)
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")


def jump_profile_family(ax, y, truth, pred, *, lead_times, color: str,
                        xlabel: str = "$y$", ylabel: str = "",
                        title: str = "", legend: bool = False) -> None:
    """A family of truth/prediction interface-jump profiles, one pair per lead.

    Truth is a wide translucent band, prediction a narrow crisp dash drawn on
    top, both in the same tint so a pair reads as one comparison; the tint
    darkens with lead time so error growth is visible inside the panel.
    Deliberately not one colour per series: with three leads that would be six
    colours and the eye would group by lead rather than by
    truth-versus-prediction, which is the comparison being made.

    The width asymmetry is what keeps both readable. Two lines of equal weight
    coincide wherever the model is right, which is most of the domain, and the
    one drawn second erases the other; a thin dash riding inside a wide band
    stays legible whether they agree or not.
    """
    y = np.asarray(y, dtype=float)
    truth = np.atleast_2d(np.asarray(truth, dtype=float))
    pred = np.atleast_2d(np.asarray(pred, dtype=float))
    lead_times = np.asarray(lead_times, dtype=float).ravel()
    if truth.shape != pred.shape:
        raise ValueError(
            f"truth {truth.shape} and pred {pred.shape} must share one shape")
    if truth.shape[0] != lead_times.size:
        raise ValueError(
            f"lead_times has {lead_times.size} entries for {truth.shape[0]} profiles")
    if truth.shape[1] != y.size:
        raise ValueError(
            f"profiles have {truth.shape[1]} nodes for {y.size} y values")

    base = np.asarray(mpl.colors.to_rgb(color))
    n = truth.shape[0]
    for i in range(n):
        # The ramp runs past the base colour into a shaded version of it rather
        # than stopping there, because three steps between white and the base
        # alone put the middle lead too close to both of its neighbours to
        # separate at print size. 0.62 is as near white as the lightest can go
        # before it stops holding against the axes background.
        mix = 0.62 - 0.92 * (i / max(n - 1, 1))
        tint = tuple(base + (1.0 - base) * mix if mix >= 0.0
                     else base * (1.0 + mix))
        # Only the truth line is labelled. Labelling both doubles the legend to
        # say the same thing twice; the solid/dashed key belongs once per figure.
        ax.plot(y, truth[i], color=tint, linewidth=2.8, linestyle="-", zorder=3,
                alpha=0.40, solid_capstyle="round",
                label=f"$\\bar t$={lead_times[i]:.2f}")
        ax.plot(y, pred[i], color=tint, linewidth=1.0, zorder=4,
                linestyle=(0, (3.0, 1.4)),
                label="_nolegend_")
    ax.set_xlim(float(y.min()), float(y.max()))
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=7)
    ax.grid(True, alpha=0.25)
    if legend:
        ax.legend(fontsize=7.0, loc="best", handlelength=1.6, borderpad=0.3,
                  labelspacing=0.2, handletextpad=0.5, framealpha=0.75,
                  ncol=1)


def parity_scatter(ax, truth, pred, *, fit=None, color_by=None,
                   norm=None, cmap: str = style.CMAP_SEQ_ORDINAL,
                   color: str = "#0072B2", xlabel: str = "", ylabel: str = "",
                   title: str = ""):
    """Predicted against true values with the identity line, coloured by lead.

    Returns the mappable so a caller can hang one shared colourbar off a row of
    these; the colour scale has to be shared for the panels to be comparable.

    Limits are square and symmetric about the identity line: an unequal aspect
    turns a systematic slope error into something that looks like scatter.
    """
    truth = np.asarray(truth, dtype=float).ravel()
    pred = np.asarray(pred, dtype=float).ravel()
    ok = np.isfinite(truth) & np.isfinite(pred)
    if color_by is None:
        mappable = ax.scatter(truth[ok], pred[ok], s=1.5, color=color,
                              linewidth=0.0, alpha=0.45, zorder=3)
    else:
        c = np.asarray(color_by, dtype=float).ravel()[ok]
        mappable = ax.scatter(truth[ok], pred[ok], s=1.5, c=c, cmap=cmap,
                              norm=norm, linewidth=0.0, alpha=0.55, zorder=3)

    if ok.any():
        lo = float(min(truth[ok].min(), pred[ok].min()))
        hi = float(max(truth[ok].max(), pred[ok].max()))
        pad = 0.06 * (hi - lo) if hi > lo else max(abs(hi), 1.0) * 0.06
        span = (lo - pad, hi + pad)
        ax.plot(span, span, color="0.4", linewidth=0.7, linestyle="--", zorder=1)
        ax.set_xlim(*span)
        ax.set_ylim(*span)
    if fit is not None:
        ax.text(0.04, 0.96,
                f"slope {fit.slope:.3f}\nRMSE {fit.rmse:.3g} K",
                transform=ax.transAxes, ha="left", va="top", fontsize=7.0,
                color="0.2")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=7)
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")
    return mappable


def _spans_a_decade(values: np.ndarray) -> bool:
    """Whether the positive values cover more than one order of magnitude."""
    v = values[np.isfinite(values) & (values > 0)]
    return bool(v.size and v.max() / v.min() > 10.0)


def association_scatter(ax, x, y, *, correlation=None, color: str = "#0072B2",
                        xlabel: str = "", ylabel: str = "", title: str = "",
                        logx="auto", logy: bool = False) -> None:
    """One point per case, annotated with a rank correlation and its interval.

    No fitted line: the claim under test is monotone association, and a least
    squares line drawn through ten heavy-tailed points would suggest a
    functional form the data does not support.

    ``logx="auto"`` takes a log axis only when the values span more than a
    decade. Forcing one on a narrower range yields a strip of overlapping minor
    tick labels that reads as noise.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    ax.scatter(x[ok], y[ok], s=16, color=color, edgecolor="k", linewidth=0.3,
               zorder=3)
    if logx == "auto":
        logx = _spans_a_decade(x[ok])
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    if correlation is not None:
        text = f"Spearman $\\rho$ = {correlation.rho:.2f}"
        if np.isfinite(correlation.ci_lo):
            text += f"\n[{correlation.ci_lo:.2f}, {correlation.ci_hi:.2f}]"
        else:
            text += "\n(no interval)"
        text += f"\nn = {correlation.n}"
        ax.text(0.97, 0.97, text, transform=ax.transAxes, ha="right", va="top",
                fontsize=5.5, color="0.25")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=7)
    ax.grid(True, alpha=0.25)


def component_bars(ax, weights, *, colors=None, xticklabels=None,
                   ylabel: str = "", title: str = "", xlabel: str = "") -> None:
    """Stacked non-negative components, one bar per case.

    ``weights`` is a mapping of component name to a per-case sequence. The
    intended use is the squared components of a unit direction vector, which
    sum to one, so a bar reaching the top with a single colour says the whole
    direction lies along that one parameter.
    """
    names = list(weights)
    n = len(next(iter(weights.values())))
    x = np.arange(n, dtype=float)
    colors = colors or {}
    bottom = np.zeros(n, dtype=float)
    for name in names:
        w = np.asarray(weights[name], dtype=float)
        ax.bar(x, w, bottom=bottom, width=0.72, label=name, zorder=2,
               color=colors.get(name, None), edgecolor="k", linewidth=0.3)
        bottom = bottom + w
    ax.set_xticks(x)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels, fontsize=5.0, rotation=90)
    ax.set_xlim(-0.7, n - 0.3)
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title, fontsize=7)
    ax.grid(True, axis="y", alpha=0.25)
    # Stacked bars fill the axes, so the legend needs an opaque backing to stay
    # readable; it sits over the bottom of the stack, which is flat by
    # construction and carries no detail.
    ax.legend(fontsize=5.5, loc="lower left", ncol=2, frameon=True,
              framealpha=0.9, facecolor="w", edgecolor="0.7")


def rate_bars(ax, rates, *, labels=None, colors=None, ylabel: str = "",
              title: str = "", threshold_label: str = "",
              rotate_xticks: float = 0.0) -> None:
    """Exceedance rates with Wilson score intervals, one bar per group.

    ``rates`` is an ordered mapping of label to
    :class:`~visual.pub.stats.RateCI`. The interval is Wilson rather than
    normal because at these sample sizes a rate near 0 would otherwise be given
    a symmetric interval crossing zero.
    """
    rates = dict(rates)
    names = list(rates)
    x = np.arange(len(names), dtype=float)
    colors = colors or {}
    values = np.array([rates[n].rate for n in names], dtype=float)

    ax.bar(x, values, width=0.6, zorder=2,
           color=[colors.get(n, "#0072B2") for n in names],
           edgecolor="k", linewidth=0.5, alpha=0.85)
    for xi, name in zip(x, names):
        r = rates[name]
        if np.isfinite(r.lo):
            ax.vlines(xi, r.lo, r.hi, color="k", linewidth=1.0, zorder=4)
        ax.text(xi, max(r.hi if np.isfinite(r.hi) else r.rate, r.rate) + 0.015,
                f"{r.k}/{r.n}", ha="center", va="bottom", fontsize=5.5,
                color="0.25", zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels if labels is not None else names,
                       rotation=rotate_xticks,
                       ha="right" if rotate_xticks else "center")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=7)
    if threshold_label:
        ax.text(0.98, 0.95, threshold_label, transform=ax.transAxes, ha="right",
                va="top", fontsize=5.5, color="0.35")
    ax.set_ylim(0.0, None)
    ax.grid(True, axis="y", alpha=0.25)


def annotate_counts(ax, positions, summaries, *, y: float = 0.01,
                    fontsize: float = 5.0) -> None:
    """Print the simulation count under each categorical position.

    The replication count is the whole point of the statistical redesign, so it
    is drawn on the figure rather than left in the sidecar.
    """
    for xi, s in zip(positions, summaries):
        ax.text(xi, y, f"{s.n_sims}", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=fontsize, color="0.45")


__all__ = [
    "ConvergenceFit",
    "annotate_counts",
    "association_scatter",
    "attach_colorbar",
    "blank_canvas",
    "component_bars",
    "convergence_loglog",
    "ecdf_curves",
    "field_map",
    "joint_nll_contour_panel",
    "jump_profile_family",
    "order_estimates",
    "pairwise_orders",
    "paired_sensor_sweep_panel",
    "parity_scatter",
    "profile_likelihood_panel",
    "rate_bars",
    "rc_profile_recovery_panel",
    "recovery_scatter",
    "resolution_curve_panel",
    "rollout_delta_panel",
    "schematic_arrow",
    "schematic_box",
    "seed_dots",
    "summary_points",
]
