"""Print-quality style, sizing, colors, and provenance footers for paper figures.

Deliberately separate from ``visual/_common.PLOT_STYLE``, which targets on-screen
diagnostics. The values here target a two-column journal page: 8 pt body text at
final size, no top/right spines, TrueType-embedded vector output.
"""

from __future__ import annotations

import datetime as _dt
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from visual._common import _ensure_parent

# ============================================================
# rcParams
# ============================================================

PUB_RCPARAMS: dict = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.2,
    "lines.markersize": 3.5,
    "legend.frameon": False,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "image.interpolation": "nearest",
    # Embed TrueType so figure text stays selectable and editable in the PDF.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


@contextmanager
def pub_style(extra: dict | None = None):
    """Apply publication rcParams for the duration of a figure build."""
    params = dict(PUB_RCPARAMS)
    if extra:
        params.update(extra)
    with mpl.rc_context(params):
        yield


# ============================================================
# Sizing
# ============================================================

WIDTHS_IN: dict[str, float] = {
    "one_col": 3.42,
    "one_half_col": 5.0,
    "two_col": 7.0,
    "full_page": 7.0,
}

HEIGHTS_IN: dict[str, float] = {
    "short": 2.0,
    "std": 2.6,
    "tall": 3.6,
    "grid_row": 1.55,
}


def figsize(width: str, *, rows: int = 1, row_height: str = "std",
            extra_in: float = 0.0) -> tuple[float, float]:
    """Return a (w, h) figure size in inches for a named column width."""
    if width not in WIDTHS_IN:
        raise KeyError(f"Unknown figure width {width!r}. Known: {sorted(WIDTHS_IN)}")
    if row_height not in HEIGHTS_IN:
        raise KeyError(f"Unknown row height {row_height!r}. Known: {sorted(HEIGHTS_IN)}")
    return (WIDTHS_IN[width], rows * HEIGHTS_IN[row_height] + extra_in)


# ============================================================
# Color
# ============================================================

CMAP_TEMPERATURE = "inferno"
CMAP_RESIDUAL = "RdBu_r"
CMAP_JUMP = "cividis"
CMAP_SEQ_ORDINAL = "viridis"

# Okabe-Ito, colorblind-safe.
BENCHMARK_COLORS: dict[str, str] = {
    "forcing": "#0072B2",
    "source": "#E69F00",
    "interfaces": "#009E73",
    "source_itr_sin": "#CC79A7",
    "forcing_itr_sin": "#D55E00",
    "source_itr_sin": "#9467BD",
    "forcing_itr_sin": "#8C564B",
}

GREY = "0.35"


def benchmark_color(benchmark: str) -> str:
    """Return the fixed color for a benchmark, or a neutral grey if unknown."""
    return BENCHMARK_COLORS.get(benchmark, "0.45")


@dataclass(frozen=True)
class ColorLimits:
    """Shared color scale for a set of fields."""

    vmin: float
    vmax: float
    vcenter: float | None
    cmap: str

    def imshow_kwargs(self) -> dict:
        kw = {"vmin": self.vmin, "vmax": self.vmax, "cmap": self.cmap}
        if self.vcenter is not None:
            kw.pop("vmin")
            kw.pop("vmax")
            kw["norm"] = mpl.colors.TwoSlopeNorm(
                vmin=self.vmin, vcenter=self.vcenter, vmax=self.vmax
            )
        return kw


def shared_color_limits(*fields: np.ndarray, mode: str = "sequential",
                        robust: bool = True,
                        pct: tuple[float, float] = (0.5, 99.5)) -> ColorLimits:
    """Compute one color scale spanning every supplied field.

    ``mode="diverging"`` returns symmetric limits centered on zero so a signed
    residual is never visually biased toward one sign. Fields are pooled before
    the percentile so truth and prediction panels genuinely share a scale.
    """
    stacked = np.concatenate([np.asarray(f, dtype=float).ravel() for f in fields])
    finite = stacked[np.isfinite(stacked)]
    if finite.size == 0:
        raise ValueError("shared_color_limits received no finite values")

    if mode == "diverging":
        mag = float(np.percentile(np.abs(finite), pct[1])) if robust \
            else float(np.max(np.abs(finite)))
        if not np.isfinite(mag) or mag <= 0.0:
            mag = 1e-12
        return ColorLimits(vmin=-mag, vmax=mag, vcenter=0.0, cmap=CMAP_RESIDUAL)

    if mode != "sequential":
        raise ValueError(f"Unknown color-limit mode {mode!r}")
    if robust:
        lo = float(np.percentile(finite, pct[0]))
        hi = float(np.percentile(finite, pct[1]))
    else:
        lo, hi = float(np.min(finite)), float(np.max(finite))
    if hi <= lo:
        hi = lo + 1e-12
    return ColorLimits(vmin=lo, vmax=hi, vcenter=None, cmap=CMAP_TEMPERATURE)


# ============================================================
# Units and labels
# ============================================================

UNITS: dict[str, str] = {
    "rmse_K": "K",
    "iface_rmse_K": "K",
    "node_jump_rmse_K": "K",
    "contact_jump_rmse_K": "K",
    "max_err_K": "K",
    "rel_l2_pct": "%",
    "iface_rel_l2_pct": "%",
    "nrmse_pct": "%",
    "gnrmse_pct": "%",
    "node_jump_nrmse_pct": "%",
    "node_jump_gnrmse_pct": "%",
    "mean_pair_rmse_K": "K",
    "t_bar": "-",
    "t_s": "-",
    "R_c": "m$^2$K/W",
    "x_I": "-",
}

LABELS: dict[str, str] = {
    "rmse_K": "Global RMSE",
    "iface_rmse_K": "Interface-region RMSE",
    "node_jump_rmse_K": "Node-jump RMSE",
    "contact_jump_rmse_K": "Contact-jump RMSE",
    "max_err_K": "Max abs. error",
    "rel_l2_pct": "Global rel. $L_2$",
    "iface_rel_l2_pct": "Interface-region rel. $L_2$",
    "nrmse_pct": "NRMSE",
    "gnrmse_pct": "Global-normalized RMSE",
    "node_jump_nrmse_pct": "Node-jump NRMSE",
    "node_jump_gnrmse_pct": "Node-jump global NRMSE",
    "mean_pair_rmse_K": "Mean per-pair RMSE",
    "t_bar": "Lead time",
    "t_s": "Source time",
    "R_c": "Contact resistance $R_c$",
    "x_I": "Interface location $x_I$",
}

# Metric space per metric name; summarize() refuses to mix these.
METRIC_SPACE: dict[str, str] = {
    "rmse_K": "kelvin",
    "iface_rmse_K": "kelvin",
    "node_jump_rmse_K": "kelvin",
    "contact_jump_rmse_K": "kelvin",
    "max_err_K": "kelvin",
    "mean_pair_rmse_K": "kelvin",
    "rel_l2_pct": "normalized",
    "iface_rel_l2_pct": "normalized",
    "nrmse_pct": "normalized",
    "gnrmse_pct": "normalized",
    "node_jump_nrmse_pct": "normalized",
    "node_jump_gnrmse_pct": "normalized",
}


def axis_label(metric: str) -> str:
    """Return a fully-united axis label. Raises on an unknown metric.

    Deliberately strict: an unlabeled or mis-united axis must not be able to ship.
    """
    if metric not in LABELS:
        raise KeyError(
            f"No label registered for metric {metric!r}. "
            f"Add it to visual/pub/style.py LABELS and UNITS. Known: {sorted(LABELS)}"
        )
    unit = UNITS.get(metric, "")
    if unit in ("", "-"):
        return LABELS[metric]
    return f"{LABELS[metric]} [{unit}]"


# ============================================================
# Annotation helpers
# ============================================================

def panel_letters(axes, *, start: int = 0, loc: tuple[float, float] = (-0.14, 1.04),
                  weight: str = "bold") -> None:
    """Label a sequence of axes (a), (b), (c), ... in reading order."""
    flat = np.asarray(axes).ravel()
    for k, ax in enumerate(flat):
        ax.text(loc[0], loc[1], f"({chr(ord('a') + start + k)})",
                transform=ax.transAxes, fontsize=8, fontweight=weight,
                va="bottom", ha="right")


def add_reference_line(ax, value: float = 1.0, label: str = "1% target",
                       axis: str = "y") -> None:
    """Draw a dashed reference line (default: the 1% relative-error target)."""
    draw = ax.axhline if axis == "y" else ax.axvline
    draw(value, color="0.3", linestyle="--", linewidth=0.8, zorder=1)
    if label:
        if axis == "y":
            ax.text(0.99, value, label, transform=ax.get_yaxis_transform(),
                    ha="right", va="bottom", fontsize=6, color="0.3")
        else:
            ax.text(value, 0.99, label, transform=ax.get_xaxis_transform(),
                    ha="left", va="top", fontsize=6, color="0.3", rotation=90)




def _footer_text(figure_key: str, source, selection) -> str:
    """Assemble the one-line provenance string drawn under every figure."""
    parts: list[str] = [figure_key]
    if source is not None:
        parts.extend(source.footer_fields())
    if selection is not None:
        # The inverse figures select an inversion case, not a (sim, s, j) pair,
        # so they hand back a plain dict instead of a select.CaseSelection.
        parts.append(
            selection.footer_field() if hasattr(selection, "footer_field")
            else " ".join(f"{k}={v}" for k, v in dict(selection).items())
        )
    parts.append(_dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d"))
    return "  |  ".join(p for p in parts if p)


def add_provenance_footer(fig, figure_key: str, source=None,
                          selection=None) -> None:
    """Stamp the figure with benchmark, seed, counts, metric space, and git state.

    Opt-in, for reviewing a render against its artifacts. It is not the record
    of provenance -- the sidecar is, and it holds strictly more -- so the shipped
    figures carry no footer.
    """
    text = _footer_text(figure_key, source, selection)
    # Shrink to fit rather than let the line run off the canvas: a truncated
    # footer silently drops whichever field sorts last, and the date and the
    # selection are the two that would go. 0.6 em per character is the DejaVu
    # Sans average advance; the floor keeps it legible at 300 dpi.
    room_pt = fig.get_figwidth() * 72.0 * 0.99
    size = min(5.5, max(4.0, room_pt / (0.6 * max(len(text), 1))))
    fig.text(0.005, 0.002, text, fontsize=size, color="0.45", ha="left",
             va="bottom", family="DejaVu Sans")


# ============================================================
# Output
# ============================================================

def save(fig, out_dir: str | Path, key: str, *,
         formats: tuple[str, ...] = ("png", "pdf")) -> list[Path]:
    """Write a figure in every requested format and close it."""
    out_dir = Path(out_dir)
    written: list[Path] = []
    for fmt in formats:
        path = _ensure_parent(out_dir / f"{key}.{fmt}")
        fig.savefig(path, format=fmt)
        print(f"Saved {key} to: {path}")
        written.append(path)
    plt.close(fig)
    return written
