"""2x2 FV-vs-FNO temperature cross-sections through the interface, one per benchmark.

Each panel plots T(x, y*) over the full x range for the test example whose peak
adjacent-node interface jump max_y |T(x_L, y) - T(x_R, y)| sits closest to the
requested quantile (default P85) of that benchmark's test-set distribution, with
y* the row attaining that peak. FV is solid, FNO dashed, and an inset zooms on
the jump.

Evaluated mode reads each run's ``test_records.csv`` (written by
``scripts/run_eval.py <config_dir> --write-test-records``), whose
``node_jump_abs_max_true_K`` column is exactly that peak jump per test pair. The
quantile is taken over those rows, and only the selected (sim_id, s, j) pair is
re-run through the checkpoint to recover the full truth and prediction fields.

The four runs are discovered under ``--runs-root`` (default ``~/fno_runs``, where
the MSI training jobs copy their runs) by the same rule as
``python -m visual.pub --runs-root``: exactly one evaluated experiment per
benchmark, with ``--run BENCHMARK=<config_dir>`` choosing when there are several.

    python scripts/fig_interface_jump_profiles.py

``--synthetic`` instead draws a watermarked layout preview from
`synthetic_test_set`; no checkpoint or dataset is read.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.physics.internal_source import make_rc_sin_profile
from visual.pub import style

BENCHMARKS = ("forcing", "interfaces", "source", "source_itr_sin")
TITLES = {
    "forcing": "B1: Boundary forcing + ITR",
    "interfaces": "B2: Varying interface + ITR",
    "source": "B3: Internal source + ITR",
    "source_itr_sin": "B4: Spatially varying ITR",
}

# P85 of the FV peak node jump over 2000 sims x 30 saved times, from
# figures/fv_benchmark_table_20261007/<benchmark>_snapshots.npz. Synthetic sets
# are rescaled to these so the panels carry realistic Kelvin magnitudes.
FV_P85_PEAK_JUMP_K = {
    "forcing": 5.75,
    "interfaces": 15.38,
    "source": 8.84,
    "source_itr_sin": 11.95,
}

T_REF = 300.0
INSET_HALF_WIDTH = 0.07


@dataclass(frozen=True)
class Selection:
    index: int
    y_index: int
    y: float
    interface_x: float
    quantile: float
    target_jump_K: float
    selected_jump_K: float
    n_examples: int
    case: dict = field(default_factory=dict)


# ---------------------------------------------------------------- selection

def flanking_nodes(x_grid: np.ndarray, interface_x: float) -> tuple[int, int]:
    """Same convention as visual.dataset_plots: a node on the interface counts as right."""
    left = int(np.searchsorted(x_grid, interface_x, side="left")) - 1
    return left, left + 1


def node_jumps(truth: np.ndarray, x_grid: np.ndarray, interface_x: np.ndarray) -> np.ndarray:
    """``T(x_L, y) - T(x_R, y)`` for every example, ``(N, Ny)``; positive is a drop."""
    out = np.empty((truth.shape[0], truth.shape[2]))
    for n, x_i in enumerate(interface_x):
        left, right = flanking_nodes(x_grid, float(x_i))
        out[n] = truth[n, left] - truth[n, right]
    return out


def select_quantile_example(truth, x_grid, y_grid, interface_x, quantile) -> Selection:
    jumps = np.abs(node_jumps(truth, x_grid, interface_x))
    peak = jumps.max(axis=1)
    target = float(np.quantile(peak, quantile))
    index = int(np.argmin(np.abs(peak - target)))
    y_index = int(np.argmax(jumps[index]))
    return Selection(
        index=index, y_index=y_index, y=float(y_grid[y_index]),
        interface_x=float(interface_x[index]), quantile=float(quantile),
        target_jump_K=target, selected_jump_K=float(peak[index]),
        n_examples=int(truth.shape[0]),
    )


# ----------------------------------------------------- evaluated test sets

RECORD_JUMP = "node_jump_abs_max_true_K"
# The name scripts/run_eval.py --write-test-records uses, and the one
# Manifest.discover_global_field searches for.
RECORDS_NAME = "test_records.csv"


def resolve_seed_dir(run: Path, model_seed: int | None) -> Path:
    """The seed directory holding ``fno2d_best.pt``, given it or its config directory."""
    run = Path(run).expanduser()
    if (run / "fno2d_best.pt").is_file():
        return run
    seeds = [d for d in sorted(run.glob("seed*")) if (d / "fno2d_best.pt").is_file()]
    if model_seed is not None:
        seeds = [d for d in seeds if d.name == f"seed{model_seed}"]
    if len(seeds) != 1:
        raise SystemExit(f"{run}: expected exactly one seed*/fno2d_best.pt, found "
                         f"{[d.name for d in seeds]}; pass --model-seed or the seed directory")
    return seeds[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluated_panel(benchmark: str, run_dir: Path, records_name: str, quantile: float):
    """``(x, T_fv, T_fno, Selection)`` for the quantile test pair of one evaluated run."""
    import pandas as pd
    from data.dataset import assert_dataset_problem_version, problem_from_config
    from visual.pub import fields

    records_path = run_dir / records_name
    if not records_path.is_file():
        raise SystemExit(f"{records_path} not found; run "
                         f"scripts/run_eval.py {run_dir.parent} --write-test-records first")
    records = pd.read_csv(records_path)
    if RECORD_JUMP not in records.columns:
        raise SystemExit(f"{records_path} has no {RECORD_JUMP} column; regenerate it with "
                         "the current scripts/run_eval.py --write-test-records")
    peak = records[RECORD_JUMP].to_numpy(dtype=np.float64)
    target = float(np.quantile(peak, quantile))
    row_index = int(np.argmin(np.abs(peak - target)))
    row = records.iloc[row_index]

    bundle = fields._load(str(run_dir))
    if bundle.benchmark != benchmark:
        raise SystemExit(f"--run {benchmark}=... points at a {bundle.benchmark!r} run: {run_dir}")
    try:
        assert_dataset_problem_version(problem_from_config(bundle.config),
                                       bundle.config["data"]["t_grid_path"])
    except ValueError as exc:
        raise SystemExit(f"{benchmark}: {exc} ({run_dir})") from None
    sim_id, s, j = int(row["sim_id"]), int(row["s"]), int(row["j"])
    case = fields.evaluate_case(bundle, sim_id, s, (j,))
    truth, pred = case.truth[0], case.pred[0]
    jumps = np.abs(node_jumps(truth[None], case.x_grid, np.array([case.interface_x])))[0]
    y_index = int(np.argmax(jumps))
    # Guards against records written for a different dataset or split than the
    # one config_used.yaml now points at.
    if not np.isclose(jumps[y_index], row[RECORD_JUMP], rtol=1e-3, atol=1e-3):
        raise SystemExit(f"{benchmark}: recomputed peak jump {jumps[y_index]:.4f} K does not match "
                         f"the record's {row[RECORD_JUMP]:.4f} K for sim {sim_id} (s={s}, j={j})")

    checkpoint = fields.checkpoint_path(run_dir)
    sel = Selection(
        index=row_index, y_index=y_index, y=float(case.y_grid[y_index]),
        interface_x=float(case.interface_x), quantile=float(quantile),
        target_jump_K=target, selected_jump_K=float(row[RECORD_JUMP]),
        n_examples=int(len(records)),
        case=dict(run_dir=str(run_dir), records=str(records_path),
                  checkpoint=str(checkpoint), checkpoint_sha256=_sha256(checkpoint),
                  sim_id=sim_id, s=s, j=j, t_source=float(case.t_source),
                  t_target=float(case.t_targets[0]), lead_time=float(case.lead_times[0])),
    )
    return case.x_grid, truth[:, y_index], pred[:, y_index], sel


# --------------------------------------------------------- synthetic data

def _implicit_step(x, interface_x, k_left, k_right, c_left, c_right, R, tau,
                   q_left, source, T0):
    """One backward-Euler step of the two-layer FV problem, independently per y row.

    Left face takes flux ``q_left``, the last node is Dirichlet at T_REF, and the
    interface face uses the series conductance ``1/(h_L/k_L + R + h_R/k_R)``.
    One large step from T0 gives a transient-shaped penetration profile; it is
    a stand-in for a solver trajectory, nothing more.
    """
    nx, ny = x.size, T0.shape[1]
    dx = x[1] - x[0]
    left, right = flanking_nodes(x, interface_x)
    R = np.broadcast_to(np.asarray(R, dtype=np.float64), (ny,))
    k_face = np.where(x[:-1] < interface_x, k_left, k_right) / dx
    G = np.broadcast_to(k_face, (ny, nx - 1)).copy()
    G[:, left] = 1.0 / ((interface_x - x[left]) / k_left + R
                        + (x[right] - interface_x) / k_right)
    volume = np.full(nx, dx)
    volume[0] = dx / 2
    cap = np.where(x < interface_x, c_left, c_right) * volume / tau

    M = np.zeros((ny, nx, nx))
    i = np.arange(nx - 1)
    M[:, i, i] += cap[:-1] + G
    M[:, i + 1, i + 1] += G
    M[:, i, i + 1] -= G
    M[:, i + 1, i] -= G
    rhs = (cap[:, None] * T0 + source * volume[:, None]).T.copy()
    rhs[:, 0] += q_left
    M[:, -1, :] = 0.0
    M[:, -1, -1] = 1.0
    rhs[:, -1] = T_REF
    return np.linalg.solve(M, rhs[..., None])[..., 0].T


def _forcing_profile(y, rng):
    kind = rng.integers(4)
    c, w = rng.uniform(0.3, 0.7), rng.uniform(0.08, 0.25)
    if kind == 0:
        return np.ones_like(y)
    if kind == 1:
        return 0.15 + 0.85 * (np.abs(y - c) < w)
    if kind == 2:
        return 0.15 + 0.85 * np.exp(-0.5 * ((y - c) / w) ** 2)
    return 0.15 + 0.85 * np.clip(1.0 - np.abs(y - c) / (2 * w), 0.0, None)


def _synthetic_example(benchmark, x, y, rng):
    X, Y = np.meshgrid(x, y, indexing="ij")
    tau = rng.uniform(0.03, 0.3)
    zero = np.zeros_like(X)
    if benchmark in ("forcing", "interfaces"):
        if benchmark == "forcing":
            x_i, T0, s = 0.5, np.full_like(X, T_REF), _forcing_profile(y, rng)
        else:
            x_i = rng.uniform(0.2, 0.8)
            b = rng.uniform(-0.6, 1.0)
            T0 = T_REF + b * (1.0 - X) * (1.0 + 0.4 * np.cos(2 * np.pi * (Y - rng.uniform())))
            s = np.ones_like(y)
        R = rng.uniform(0.05, 1.0)
        T = _implicit_step(x, x_i, 2.0, 1.0, 1.0, 1.0, R, tau,
                           rng.uniform(0.3, 1.0) * s, zero, T0)
        return T, x_i
    x_h = rng.choice([rng.uniform(0.1, 0.42), rng.uniform(0.58, 0.9)], p=[0.8, 0.2])
    y_h, w_h, h_h = rng.uniform(0.25, 0.75), rng.uniform(0.05, 0.1), rng.uniform(0.1, 0.2)
    g = np.exp(-0.5 * (((X - x_h) / w_h) ** 2 + ((Y - y_h) / h_h) ** 2))
    if benchmark == "source":
        R = rng.uniform(0.1, 2.0)
    else:
        R = make_rc_sin_profile(y, R_base=rng.uniform(0.1, 1.0), A=rng.uniform(0.0, 1.0))
    T = _implicit_step(x, 0.5, 1.0, 16.0, 1.0, 1.4, R, tau, np.zeros_like(y),
                       rng.uniform(0.3, 1.0) * 8.0 * g, np.full_like(X, T_REF))
    return T, 0.5


def _synthetic_prediction(truth, x, interface_x, jump, rng, field_err, jump_err):
    """Truth plus a smooth global error and a smeared, slightly biased interface jump."""
    span = float(np.ptp(truth))
    phase = rng.uniform(0, 2 * np.pi, size=4)
    smooth = sum(np.sin((m + 1) * np.pi * x + phase[m]) / (m + 1) for m in range(3))
    smooth = smooth[:, None] * (1.0 + 0.3 * np.cos(2 * np.pi * np.linspace(0, 1, truth.shape[1]) + phase[3]))
    d = x - interface_x
    kappa = rng.normal(-jump_err, jump_err)
    side = np.where(d < 0, 0.5, -0.5) * np.exp(-np.abs(d) / 0.04)
    ripple = 0.3 * jump_err * np.sin(2 * np.pi * d / 0.06) * np.exp(-np.abs(d) / 0.05)
    return (truth + field_err * span * smooth
            + kappa * side[:, None] * jump[None, :]
            + ripple[:, None] * np.abs(jump)[None, :])


# Relative error scales, ordered after the paper's Table 4: forcing has the
# tightest node jump, interfaces the loosest.
_SYNTHETIC_ERRORS = {
    "forcing": (0.006, 0.02),
    "interfaces": (0.004, 0.06),
    "source": (0.003, 0.03),
    "source_itr_sin": (0.008, 0.04),
}


def synthetic_test_set(benchmark: str, n_examples: int, seed: int):
    """Synthetic (truth, prediction) test set in Kelvin, ``(N, Nx, Ny)`` each."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, 100)
    y = np.linspace(0.0, 1.0, 100)
    examples = [_synthetic_example(benchmark, x, y, rng) for _ in range(n_examples)]
    rise = np.stack([T - T_REF for T, _ in examples])
    interface_x = np.array([x_i for _, x_i in examples])
    peak = np.abs(node_jumps(rise, x, interface_x)).max(axis=1)
    rise *= FV_P85_PEAK_JUMP_K[benchmark] / np.quantile(peak, 0.85)
    truth = rise + T_REF
    jumps = node_jumps(truth, x, interface_x)
    field_err, jump_err = _SYNTHETIC_ERRORS[benchmark]
    pred = np.stack([
        _synthetic_prediction(truth[n], x, interface_x[n], jumps[n], rng, field_err, jump_err)
        for n in range(n_examples)
    ])
    return truth, pred, x, y, interface_x


# ------------------------------------------------------------------ figure

INK = "#1a1a1a"
MUTED = "0.42"
REF = "0.25"
GRID = "0.9"
BAND = "0.91"
FNO_DASH = (0, (3.0, 1.5))
FV_LW, FNO_LW, IFACE_LW = 2.6, 1.4, 1.3

# Each layout pairs inset corners with the y headroom that frees them: upper
# corners with spare room above the data, lower corners with spare room below.
_INSET_LAYOUTS = (
    ((-0.06, 0.42), {"upper right": (0.555, 0.46, 0.43, 0.48),
                     "upper left": (0.03, 0.46, 0.43, 0.48)}),
    ((-0.62, 0.06), {"lower right": (0.555, 0.15, 0.43, 0.42),
                     "lower left": (0.03, 0.15, 0.43, 0.42)}),
)


def _place_inset(ax, x, curves, interface_x):
    """Set the y limits and return the inset rectangle covering the least data.

    Every corner is scored by how many plotted points fall under it, and a corner
    over the interface itself is ruled out, since the jump must stay visible in
    the main panel. Ties keep the first layout, so insets sit top-right by default.
    """
    lo = min(float(np.min(T)) for T in curves)
    hi = max(float(np.max(T)) for T in curves)
    span = hi - lo
    x0, x1 = ax.get_xlim()
    xf = (np.asarray(x) - x0) / (x1 - x0)
    jump_xf = (interface_x - x0) / (x1 - x0)
    best = None
    for (below, above), slots in _INSET_LAYOUTS:
        y0, y1 = lo + below * span, hi + above * span
        for left, bottom, width, height in slots.values():
            cost = 1e3 if left - 0.05 < jump_xf < left + width + 0.03 else 0.0
            for T in curves:
                yf = (np.asarray(T) - y0) / (y1 - y0)
                cost += ((xf > left - 0.03) & (xf < left + width + 0.03)
                         & (yf > bottom - 0.06) & (yf < bottom + height + 0.06)).sum()
            if best is None or cost < best[0]:
                best = (cost, (y0, y1), (left, bottom, width, height))
    ax.set_ylim(*best[1])
    return best[2]


def _fit_inside(text, ax, margin=0.03, min_size=5.4):
    """Shrink ``text`` until it ends inside ``ax`` with ``margin`` to spare.

    Real readouts vary in width ("-12.3%" vs "+1.4%"), so the size is fitted
    rather than fixed.
    """
    renderer = ax.figure.canvas.get_renderer()
    box = ax.get_window_extent(renderer)
    limit = box.x1 - margin * box.width
    while text.get_fontsize() > min_size and text.get_window_extent(renderer).x1 > limit:
        text.set_fontsize(text.get_fontsize() - 0.1)


def _jump_inset(ax, x, T_fv, T_fno, sel, color, jump_fv, jump_fno):
    axins = ax.inset_axes(_place_inset(ax, x, (T_fv, T_fno), sel.interface_x))
    window = np.abs(x - sel.interface_x) <= INSET_HALF_WIDTH
    left, right = flanking_nodes(x, sel.interface_x)
    nodes = [left, right]

    axins.plot(x[window], T_fv[window], color=color, linewidth=FV_LW,
               solid_capstyle="round", zorder=2)
    axins.plot(x[window], T_fno[window], color=INK, linewidth=FNO_LW,
               linestyle=FNO_DASH, zorder=3)
    axins.plot(x[nodes], T_fv[nodes], linestyle="none", marker="o", markersize=6.5,
               markerfacecolor=color, markeredgecolor="white", markeredgewidth=1.1, zorder=4)
    axins.plot(x[nodes], T_fno[nodes], linestyle="none", marker="o", markersize=6.5,
               markerfacecolor="none", markeredgecolor=INK, markeredgewidth=1.3, zorder=5)

    # Bracket on the downstream side, where a drop leaves the space above the curve empty.
    dx = x[1] - x[0]
    x_b = x[right] + 2.0 * dx
    axins.annotate("", xy=(x_b, T_fv[right]), xytext=(x_b, T_fv[left]),
                   arrowprops=dict(arrowstyle="<|-|>", color=INK, linewidth=1.1,
                                   shrinkA=0, shrinkB=0, mutation_scale=7))
    axins.plot([x[left], x_b], [T_fv[left], T_fv[left]], color=REF,
               linewidth=0.9, linestyle=(0, (1.2, 1.4)), zorder=1)

    w_lo = min(T_fv[window].min(), T_fno[window].min())
    w_hi = max(T_fv[window].max(), T_fno[window].max())
    span = w_hi - w_lo
    up = jump_fv > 0
    axins.set_xlim(sel.interface_x - INSET_HALF_WIDTH, sel.interface_x + INSET_HALF_WIDTH)
    y_lo, y_hi = w_lo - (0.1 if up else 0.62) * span, w_hi + (0.62 if up else 0.1) * span
    axins.set_ylim(y_lo, y_hi)
    # Stop the interface line short of the readout band so the text never crosses it.
    data_top, data_bot = (w_hi - y_lo) / (y_hi - y_lo), (w_lo - y_lo) / (y_hi - y_lo)
    axins.axvline(sel.interface_x, color=REF, linestyle=":", linewidth=IFACE_LW, zorder=1,
                  ymin=0.0 if up else data_bot - 0.04, ymax=data_top + 0.04 if up else 1.0)
    # The main axis already carries T, so the inset drops its y labels: the room
    # goes to the readout, and the box can sit close to the interface line.
    axins.xaxis.set_major_locator(MaxNLocator(nbins=3))
    axins.set_yticks([])
    axins.tick_params(axis="x", length=2.5, width=0.8, pad=1.5, colors=REF, labelcolor=INK)
    style.bold_axis_text(axins, label_size=6.8, tick_size=6.8)
    axins.set_facecolor("white")
    for spine in axins.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.9)
        spine.set_color(REF)

    rel = f"{100.0 * (jump_fno - jump_fv) / jump_fv:+.1f}".replace("-", "\u2212")
    label = "$\\mathbf{{\\Delta}}\\boldsymbol{{T}}_{{\\mathbf{{{}}}}}$"
    readout = axins.text(0.04, 0.95 if up else 0.05,
                         f"{label.format('FV')} = {jump_fv:.2f} K\n"
                         f"{label.format('FNO')} = {jump_fno:.2f} K ({rel}%)",
                         transform=axins.transAxes, ha="left", va="top" if up else "bottom",
                         fontsize=6.6, fontweight="bold", color=INK, linespacing=1.4)
    _fit_inside(readout, axins)
    ax.axvspan(*axins.get_xlim(), color=BAND, linewidth=0, zorder=0)


def _panel(ax, benchmark, x, T_fv, T_fno, sel: Selection):
    color = style.benchmark_color(benchmark)
    ax.grid(axis="y", color=GRID, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.axvline(sel.interface_x, color=REF, linestyle=":", linewidth=IFACE_LW, zorder=1)
    ax.plot(x, T_fv, color=color, linewidth=FV_LW, solid_capstyle="round", zorder=2)
    ax.plot(x, T_fno, color=INK, linewidth=FNO_LW, linestyle=FNO_DASH, zorder=3)
    ax.set_xlim(0.0, 1.0)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.tick_params(colors=REF, labelcolor=INK, length=3, labelsize=7.5)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(REF)
    ax.set_title(TITLES[benchmark], loc="left", fontsize=8.8, fontweight="bold",
                 color=INK, pad=6)
    ax.set_title(f"$\\boldsymbol{{y}}$ = {sel.y:.2f}", loc="right", fontsize=8.3,
                 fontweight="bold", color=REF, pad=6)

    left, right = flanking_nodes(x, sel.interface_x)
    jump_fv = float(T_fv[left] - T_fv[right])
    jump_fno = float(T_fno[left] - T_fno[right])
    _jump_inset(ax, x, T_fv, T_fno, sel, color, jump_fv, jump_fno)

    # The headroom exists only for the inset, so the temperature axis stops at
    # the data rather than labelling empty space.
    lo, hi = min(T_fv.min(), T_fno.min()), max(T_fv.max(), T_fno.max())
    slack = 0.02 * (hi - lo)
    ticks = [t for t in MaxNLocator(nbins=5, steps=[1, 2, 5, 10]).tick_values(lo, hi) if lo - slack <= t <= hi + slack]
    ax.set_yticks(ticks)
    ax.spines["left"].set_bounds(min(ticks[0], lo), max(ticks[-1], hi))
    return jump_fv, jump_fno


def render(panels: dict, out_dir: Path, key: str, synthetic: bool) -> list[Path]:
    with style.pub_style():
        fig, axes = plt.subplots(2, 2, figsize=style.figsize("two_col", rows=2, row_height="std",
                                                             extra_in=0.3))
        fig.subplots_adjust(left=0.08, right=0.985, bottom=0.085, top=0.88,
                            wspace=0.18, hspace=0.38)
        for ax, bench in zip(axes.ravel(), BENCHMARKS):
            x, T_fv, T_fno, sel = panels[bench]
            panels[bench] = (*panels[bench], *_panel(ax, bench, x, T_fv, T_fno, sel))
        for ax in axes[1]:
            ax.set_xlabel("$\\boldsymbol{x}$", color=INK, fontsize=9)
        for ax in axes[:, 0]:
            ax.set_ylabel("$\\boldsymbol{T}$ [K]", color=INK, fontsize=9, fontweight="bold")
        style.panel_letters(axes, loc=(-0.03, 1.035))
        for ax in axes.ravel():
            ax.texts[-1].set_fontsize(9.5)

        fv_key = tuple(Line2D([], [], color=style.benchmark_color(b), linewidth=FV_LW)
                       for b in BENCHMARKS)
        fig.legend(
            handles=[fv_key,
                     Line2D([], [], color=INK, linewidth=FNO_LW, linestyle=FNO_DASH),
                     Line2D([], [], color=REF, linewidth=IFACE_LW, linestyle=":"),
                     Patch(facecolor=BAND, edgecolor="none")],
            labels=["FV ground truth", "FNO prediction", "material interface",
                    "region shown in inset"],
            handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)},
            loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=4, frameon=False,
            handlelength=3.0, columnspacing=1.8, prop={"size": 8, "weight": "bold"})
        if synthetic:
            fig.text(0.005, 0.002, "SYNTHETIC DATA: layout preview, not FV solver or "
                     "FNO checkpoint output.", fontsize=5.5, color="#B22222",
                     ha="left", va="bottom")
        return style.save(fig, out_dir, key)


def discover_runs(runs_root: Path, selections: list[str], model_seed: int | None) -> dict[str, Path]:
    """One evaluated seed directory per benchmark, found under ``runs_root``."""
    from visual.pub.manifest import Manifest, ProvenanceError

    chosen = {}
    for item in selections:
        bench, sep, path = item.partition("=")
        if not sep or bench not in BENCHMARKS or bench in chosen:
            raise SystemExit(f"--run expects a unique BENCHMARK=CONFIG_DIR with BENCHMARK in "
                             f"{BENCHMARKS}; got {item!r}")
        chosen[bench] = path
    try:
        manifest = Manifest.discover_global_field(runs_root, selections=chosen)
    except ProvenanceError as exc:
        raise SystemExit(f"run discovery under {runs_root} failed: {exc}") from None
    return {bench: resolve_seed_dir(Path(manifest.sources[f"{bench}_records"][0]["run"]).parent,
                                    model_seed)
            for bench in BENCHMARKS}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs-root", type=Path, default=Path("~/fno_runs"),
                        help="directory searched for evaluated runs (default ~/fno_runs)")
    parser.add_argument("--run", action="append", default=[], metavar="BENCHMARK=CONFIG_DIR",
                        help="choose an experiment when a benchmark has several evaluated ones")
    parser.add_argument("--model-seed", type=int, default=None,
                        help="which seed*/ to use when a config dir holds several")
    parser.add_argument("--synthetic", action="store_true",
                        help="draw the watermarked synthetic layout preview instead")
    parser.add_argument("--out", type=Path, default=None,
                        help="output directory (default figures/, or figures/preview/ with --synthetic)")
    parser.add_argument("--quantile", type=float, default=0.85)
    parser.add_argument("--n-examples", type=int, default=240, help="synthetic only")
    parser.add_argument("--seed", type=int, default=20261007, help="synthetic only")
    args = parser.parse_args(argv)
    if args.synthetic and args.run:
        parser.error("--run selects evaluated runs; it does not apply to --synthetic")

    sidecar = {"synthetic": args.synthetic, "quantile": args.quantile,
               "jump_definition": "peak over y of |T(x_L,y) - T(x_R,y)| across the two "
                                  "nodes flanking the interface (test_records "
                                  f"{RECORD_JUMP})",
               "benchmarks": {}}
    panels = {}
    if args.synthetic:
        sidecar["seed"] = args.seed
        for k, bench in enumerate(BENCHMARKS):
            truth, pred, x, y, interface_x = synthetic_test_set(bench, args.n_examples, args.seed + k)
            sel = select_quantile_example(truth, x, y, interface_x, args.quantile)
            panels[bench] = (x, truth[sel.index, :, sel.y_index], pred[sel.index, :, sel.y_index], sel)
    else:
        for bench, run_dir in discover_runs(args.runs_root, args.run, args.model_seed).items():
            print(f"{bench:15s} {run_dir}")
            panels[bench] = evaluated_panel(bench, run_dir, RECORDS_NAME, args.quantile)
    for bench in BENCHMARKS:
        sidecar["benchmarks"][bench] = asdict(panels[bench][3])

    key = "interface_jump_profiles" + ("_SYNTHETIC" if args.synthetic else "")
    out = args.out or PROJECT_ROOT / "figures" / ("preview" if args.synthetic else "")
    render(panels, out, key, synthetic=args.synthetic)
    for bench in BENCHMARKS:
        sidecar["benchmarks"][bench].update(
            plotted_jump_fv_K=float(panels[bench][4]), plotted_jump_fno_K=float(panels[bench][5]))
    (out / f"{key}.json").write_text(json.dumps(sidecar, indent=2))
    for bench, s in sidecar["benchmarks"].items():
        where = (f"sim {s['case']['sim_id']} t={s['case']['t_target']:.3f}" if s["case"]
                 else f"example {s['index']}")
        print(f"{bench:15s} P{100 * s['quantile']:.0f}={s['target_jump_K']:.2f} K  {where}  "
              f"peak={s['selected_jump_K']:.2f} K  y={s['y']:.2f}  x_I={s['interface_x']:.3f}  "
              f"FV={s['plotted_jump_fv_K']:.2f}  FNO={s['plotted_jump_fno_K']:.2f}  "
              f"(n={s['n_examples']})")


if __name__ == "__main__":
    main()
