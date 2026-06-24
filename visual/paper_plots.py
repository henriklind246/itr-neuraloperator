"""Publication-ready benchmark figures (registry group ``paper``).

Two figure types per benchmark plus two cross-benchmark synthesis figures, all
with capped panel counts:

- ``*_test_error_summary``        2x3 diversity/regime evidence from per-sample
                                  test records (``test_records.csv``).
- ``*_prediction_truth_residual`` 3x3 truth | prediction | residual for three
                                  representative cases.
- ``all_benchmarks_error_summary``            1x3 cross-benchmark error spread.
- ``all_benchmarks_prediction_truth_residual`` 3x3 one case per benchmark.

Per-sample records are produced by ``src.operators.eval.write_test_records``;
metrics are in normalized space (same convention as train/val rel-L2). Fields
are rendered in normalized space so the residual matches the reported metric.
"""

import csv as _csv
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

from data.dataset import (
    SnapshotPairDataset,
    compute_global_stats,
    problem_from_config,
    split_sim_ids,
)
from visual._common import PLOT_STYLE, _save_figure
from visual.dataset_plots import (
    _compute_binned_quantiles,
    _contact_jump_reductions,
    _future_target_indices,
    _interface_conductance_G,
    _interface_contact_jump_map,
    _interface_flanking_nodes_from_grid,
    _interface_jump_map,
    _jump_error_metrics,
    _model_predict_item,
    _plot_field_2d,
    _prepare_prediction_case,
    _resolve_layer_conductivities,
)
from src.physics.internal_source import make_rc_void_profile


TEMPORAL_ORDER = ("sin", "exp", "pulse_train", "exp_train")
SPATIAL_ORDER = ("uniform", "patch", "gaussian", "triangle")
REGIME_ORDER = ("left", "near", "right")

# Colorblind-safe (Okabe-Ito) palette shared across paper figures so a benchmark
# keeps the same color everywhere. The two generalization series use distinct
# hues from the same palette.
_BENCHMARK_COLORS = {
    "forcing": "#0072B2",     # blue
    "source": "#E69F00",      # orange
    "interfaces": "#009E73",  # bluish green
    "source_itr": "#CC79A7",  # reddish purple
}
_SAME_SIM_COLOR = "#0072B2"   # blue
_UNSEEN_COLOR = "#D55E00"     # vermillion

FORCING_ACTIVE_T_MAX = 0.20  # target time t_j below which boundary forcing is active

# Curated prose for the benchmark-overview table figure. ``ProblemSpec`` exposes
# only ``name`` + structured ``ProblemDims`` (no description), so the descriptive
# cells live here. Keyed by benchmark; column order is ``_BENCHMARK_OVERVIEW_COLS``.
# Kept to the four paper benchmarks deliberately (not all of ``REGISTRY``).
_BENCHMARK_OVERVIEW_COLS = (
    "geometry / source",
    "varied parameters",
    "ITR parameterization",
    "prediction task",
    "main stress test",
)
_BENCHMARK_OVERVIEW_ROWS: dict[str, dict[str, str]] = {
    "forcing": {
        "geometry / source": "two slabs, interface fixed at x=0.5; separable left-flux forcing q_L(y,t)=a(t)·s(y)",
        "varied parameters": "4 temporal families × 4 spatial profiles; scalar R_c; uniform 300 K IC",
        "ITR parameterization": "scalar R_c (uniform interface resistance)",
        "prediction task": "next-snapshot temperature field given forcing history",
        "main stress test": "pulse-like forcing, long leads, unseen forcing combinations",
    },
    "source": {
        "geometry / source": "two slabs, interface fixed at x=0.5; internal volumetric chip-heating patch",
        "varied parameters": "patch (x_h, y_h, A, w_h, h_h), stratified regime; scalar R_c; uniform 300 K IC",
        "ITR parameterization": "scalar R_c (uniform interface resistance)",
        "prediction task": "next-snapshot field given internal source channels",
        "main stress test": "near-interface patches and high-amplitude heating",
    },
    "source_itr": {
        "geometry / source": "two slabs, interface fixed at x=0.5; same internal patch as source",
        "varied parameters": "source patch/A/IC stream + Gaussian void profile",
        "ITR parameterization": "spatially-varying R_c(y) Gaussian void (R_c_base, R_c_amp, R_c_y0, R_c_sigma)",
        "prediction task": "next-snapshot field with delamination / air-gap interface",
        "main stress test": "severe voids (max R_c_amp) over the interface zone",
    },
    "interfaces": {
        "geometry / source": "two slabs; interface location x_I sampled per sim; fixed sin/uniform forcing",
        "varied parameters": "interface_x ∈ [0.2, 0.8]; scalar R_c; varying initial conditions",
        "ITR parameterization": "scalar R_c (uniform), per-sample interface location",
        "prediction task": "next-snapshot field with a moving interface position",
        "main stress test": "off-center interfaces (max |x_I − 0.5|)",
    },
}

_INT_COLS = ("sim_id", "s", "j")
_FLOAT_COLS = (
    "t_s", "t_bar", "R_c", "x_h", "y_h", "A", "freq",
    "R_c_amp", "R_c_y0", "R_c_sigma",
    "x_I", "rel_l2_pct", "iface_rel_l2_pct",
    "nrmse_pct", "rmse_K", "gnrmse_pct",
    "node_jump_rmse_K", "node_jump_nrmse_pct", "node_jump_gnrmse_pct",
)
_STR_COLS = ("benchmark", "temporal_family", "spatial_family", "regime")


# ============================================================
# RECORD LOADING + SELECTION
# ============================================================

def _load_test_records(csv_path: str | Path) -> dict[str, np.ndarray]:
    """Read a ``test_records.csv`` into a column dict of numpy arrays.

    Integer columns become int64; float columns become float64 with NaN for
    blank cells (fields that don't apply to a benchmark); string columns become
    object arrays.
    """
    with Path(csv_path).open(newline="") as f:
        rows = list(_csv.DictReader(f))

    out: dict[str, np.ndarray] = {}
    for col in _INT_COLS:
        out[col] = np.array([int(r[col]) for r in rows], dtype=np.int64)
    for col in _FLOAT_COLS:
        out[col] = np.array(
            [float(r[col]) if r.get(col, "") not in ("", None) else np.nan for r in rows],
            dtype=np.float64,
        )
    for col in _STR_COLS:
        out[col] = np.array([r.get(col, "") for r in rows], dtype=object)
    out["_n"] = np.int64(len(rows))
    return out


def _mask_in(col: np.ndarray, allowed) -> np.ndarray:
    """Boolean mask selecting rows whose string column is in ``allowed``."""
    allowed = {allowed} if isinstance(allowed, str) else set(allowed)
    return np.array([str(v) in allowed for v in col], dtype=bool)


def _representative_row(records: dict[str, np.ndarray], mask: np.ndarray) -> int:
    """Return the masked row whose global rel-L2 is closest to the group median.

    Deterministic; returns -1 when no finite-error row matches ``mask``.
    """
    err = records["rel_l2_pct"]
    idx = np.where(mask & np.isfinite(err))[0]
    if idx.size == 0:
        return -1
    median = float(np.median(err[idx]))
    return int(idx[int(np.argmin(np.abs(err[idx] - median)))])


def _row_for_x_I(records: dict[str, np.ndarray], target: float, tol: float = 0.08) -> int:
    """Representative row with ``x_I`` near ``target`` (median error in band).

    Falls back to the single row with the nearest ``x_I`` when the band is empty.
    """
    x_I = records["x_I"]
    band = np.isfinite(x_I) & (np.abs(x_I - target) <= tol)
    row = _representative_row(records, band)
    if row >= 0:
        return row
    finite = np.where(np.isfinite(x_I))[0]
    if finite.size == 0:
        return -1
    return int(finite[int(np.argmin(np.abs(x_I[finite] - target)))])


# ============================================================
# PANEL HELPERS
# ============================================================

def _finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def _hist(ax, values: np.ndarray, xlabel: str, title: str, bins: int = 30) -> None:
    vals = _finite(values)
    if vals.size:
        ax.hist(vals, bins=bins, color="C0", alpha=0.8)
        ax.axvline(float(np.median(vals)), color="0.2", linestyle="--", linewidth=1.2,
                   label=f"median {np.median(vals):.2f}%")
        ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.set_title(title)


def _box_by_category(ax, records, cat_col: str, order, value_col: str,
                     xlabel: str, title: str) -> None:
    cats = records[cat_col]
    err = records[value_col]
    data, labels = [], []
    for name in order:
        vals = _finite(err[_mask_in(cats, name)])
        if vals.size:
            data.append(vals)
            labels.append(name)
    if data:
        ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("rel-L2 (%)")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)


def _binned_line(ax, x: np.ndarray, y: np.ndarray, n_bins: int, xlabel: str,
                 title: str, ylabel: str = "rel-L2 (%)") -> None:
    centers, median, q25, q75, _ = _compute_binned_quantiles(x, y, n_bins)
    if centers.size:
        ax.plot(centers, median, "-o", ms=3, color="C0")
        ax.fill_between(centers, q25, q75, alpha=0.25, color="C0")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def _cat_cat_median(records, row_col, row_order, col_col, col_order, value_col):
    """Median of ``value_col`` over a (row_order x col_order) category grid."""
    rows_v = records[row_col]
    cols_v = records[col_col]
    err = records[value_col]
    grid = np.full((len(row_order), len(col_order)), np.nan, dtype=np.float64)
    for i, rname in enumerate(row_order):
        rmask = _mask_in(rows_v, rname)
        for k, cname in enumerate(col_order):
            vals = _finite(err[rmask & _mask_in(cols_v, cname)])
            if vals.size:
                grid[i, k] = float(np.median(vals))
    return grid


def _draw_heatmap(ax, grid, row_labels, col_labels, title, cbar_label="median rel-L2 (%)"):
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=20)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title)
    for i in range(grid.shape[0]):
        for k in range(grid.shape[1]):
            if np.isfinite(grid[i, k]):
                ax.text(k, i, f"{grid[i, k]:.1f}", ha="center", va="center",
                        color="white", fontsize=7)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)


def _binned_2d_median(x, y, v, n_bins):
    """Median of ``v`` on an ``n_bins`` x ``n_bins`` grid over (x, y)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(v)
    x, y, v = x[valid], y[valid], v[valid]
    grid = np.full((n_bins, n_bins), np.nan)
    if x.size == 0:
        return grid, np.array([]), np.array([])
    x_edges = np.linspace(x.min(), x.max(), n_bins + 1)
    y_edges = np.linspace(y.min(), y.max(), n_bins + 1)
    xb = np.clip(np.digitize(x, x_edges[1:-1]), 0, n_bins - 1)
    yb = np.clip(np.digitize(y, y_edges[1:-1]), 0, n_bins - 1)
    for i in range(n_bins):
        for k in range(n_bins):
            cell = v[(xb == i) & (yb == k)]
            if cell.size:
                grid[k, i] = float(np.median(cell))
    return grid, x_edges, y_edges


def _draw_binned_2d(ax, grid, x_edges, y_edges, xlabel, ylabel, title):
    if x_edges.size and y_edges.size:
        im = ax.pcolormesh(x_edges, y_edges, grid, cmap="viridis", shading="auto")
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="median rel-L2 (%)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


# ============================================================
# PREDICTION-FIGURE HELPERS
# ============================================================

def _records_dataset(model, trajectories, x_grid, y_grid, t_grid, sim_params,
                     config, dt=None) -> SnapshotPairDataset:
    """Build a snapshot-pair dataset over all sims for arbitrary-pair rendering.

    Uses the model's attached global stats when present (training-set stats from
    the checkpoint) and otherwise recomputes them from the training split. Only
    used to call ``problem.build_item`` directly with explicit (sim_id, s, j),
    so a minimal ``n_snapshots`` keeps pair enumeration cheap.
    """
    mu = getattr(model, "_mu_global", None)
    sigma = getattr(model, "_sigma_global", None)
    if mu is None or sigma is None:
        train_ids, _, _ = split_sim_ids(trajectories.shape[0], 0.7, 0.15, seed=0)
        mu, sigma = compute_global_stats(trajectories, train_ids)
    all_ids = np.arange(trajectories.shape[0], dtype=np.int64)
    return SnapshotPairDataset(
        trajectories=trajectories,
        t_grid=t_grid,
        x_grid=x_grid,
        y_grid=y_grid,
        sim_ids=all_ids,
        sim_params=sim_params,
        mu_global=mu,
        sigma_global=sigma,
        n_snapshots=2,
        dt=dt,
        problem=problem_from_config(config),
    )


def _predict_norm_fields(model, ds, sid: int, s: int, j: int):
    """Return (Y_true, Y_pred) normalized (Nx, Ny) fields for one pair."""
    import torch

    item = ds.problem.build_item(ds, int(sid), int(s), int(j))
    tensors = {k: torch.from_numpy(v) for k, v in item.items()}
    y_pred = _model_predict_item(model, tensors)
    Y_true = np.asarray(item["Y"][..., 0], dtype=np.float64)
    Y_pred = y_pred.squeeze(0).squeeze(-1).cpu().numpy().astype(np.float64)
    return Y_true, Y_pred


def _render_tpr_row(axes_row, x_grid, y_grid, Y_true, Y_pred, *,
                    interface_positions=None, patch=None, row_label="",
                    col_titles=False) -> None:
    """Render one truth | prediction | residual row on three axes."""
    vmin = float(min(Y_true.min(), Y_pred.min()))
    vmax = float(max(Y_true.max(), Y_pred.max()))
    residual = Y_pred - Y_true
    rmax = float(np.max(np.abs(residual)))
    if rmax <= 0.0:
        rmax = 1e-6
    res_norm = TwoSlopeNorm(vcenter=0.0, vmin=-rmax, vmax=rmax)

    ax_t, ax_p, ax_r = axes_row
    _plot_field_2d(ax_t, x_grid, y_grid, Y_true, cmap="inferno", vmin=vmin, vmax=vmax,
                   interface_positions=interface_positions)
    pcm_p = _plot_field_2d(ax_p, x_grid, y_grid, Y_pred, cmap="inferno", vmin=vmin, vmax=vmax,
                           interface_positions=interface_positions)
    pcm_r = _plot_field_2d(ax_r, x_grid, y_grid, residual, cmap="RdBu_r", norm=res_norm,
                           interface_positions=interface_positions)
    ax_p.figure.colorbar(pcm_p, ax=ax_p, fraction=0.046, pad=0.04)
    ax_r.figure.colorbar(pcm_r, ax=ax_r, fraction=0.046, pad=0.04)

    if patch is not None:
        x_h, y_h, w, h = patch
        for ax in axes_row:
            ax.add_patch(Rectangle((x_h - 0.5 * w, y_h - 0.5 * h), w, h,
                                   fill=False, edgecolor="lime", linewidth=1.2))

    if col_titles:
        ax_t.set_title("Truth")
        ax_p.set_title("Prediction")
        ax_r.set_title("Residual (pred − truth)")
    if row_label:
        ax_t.set_ylabel(row_label, fontsize=8)


def _row_label(ds, records, row: int) -> str:
    """Compose a case label from the problem's plot_label plus rel-L2 metrics."""
    sid = int(records["sim_id"][row])
    label = ds.problem.plot_label(ds.sim_params[sid])
    rel = records["rel_l2_pct"][row]
    iface = records["iface_rel_l2_pct"][row]
    suffix = []
    if np.isfinite(rel):
        suffix.append(f"L2={rel:.1f}%")
    if np.isfinite(iface):
        suffix.append(f"iface={iface:.1f}%")
    return f"{label}\n" + " ".join(suffix) if suffix else label


def _nearest_index(grid: np.ndarray, value: float) -> int:
    """Return the index of the grid node nearest to ``value``."""
    return int(np.argmin(np.abs(np.asarray(grid, dtype=np.float64) - float(value))))


def _format_coord(value: float) -> str:
    return f"{float(value):.3f}"


def _debug_scalar(value) -> float:
    arr = np.asarray(value, dtype=np.float64)
    return float(arr) if arr.ndim == 0 else float(np.mean(arr))


def _source_itr_rc_profile(params: dict, y_grid: np.ndarray):
    required = ("R_c_base", "R_c_amp", "R_c_y0", "R_c_sigma")
    if all(k in params for k in required):
        return make_rc_void_profile(
            np.asarray(y_grid, dtype=np.float64),
            R_base=float(params["R_c_base"]),
            R_amp=float(params["R_c_amp"]),
            y0=float(params["R_c_y0"]),
            sigma=float(params["R_c_sigma"]),
        )
    return float(params["R_c"])


def _interface_R_for_sim(ds, sid: int):
    params = ds.sim_params[int(sid)]
    if getattr(ds.problem, "name", "") == "source_itr":
        return _source_itr_rc_profile(params, ds.y_grid)
    return float(params["R_c"])


def _selected_forcing_rows(records) -> list[tuple[str, int]]:
    t_j = records["t_s"] + records["t_bar"]
    active = t_j <= FORCING_ACTIVE_T_MAX
    active_relaxed = t_j <= 0.25
    preferred_spatial = {"sin": "uniform", "exp": "gaussian",
                         "pulse_train": "patch", "exp_train": "triangle"}

    selected = []
    for fam in TEMPORAL_ORDER:
        fam_mask = _mask_in(records["temporal_family"], fam)
        sp_mask = _mask_in(records["spatial_family"], preferred_spatial[fam])
        row = -1
        for mask in (fam_mask & active & sp_mask,
                     fam_mask & active,
                     fam_mask & active_relaxed,
                     fam_mask):
            row = _representative_row(records, mask)
            if row >= 0:
                break
        if row >= 0:
            selected.append((fam, row))
    return selected


def _selected_source_rows(records, ds=None, prefer_void: bool = False) -> list[tuple[str, int]]:
    selected = []
    for regime in REGIME_ORDER:
        mask = _mask_in(records["regime"], regime)
        row = _representative_row(records, mask)
        if prefer_void and ds is not None:
            idx = np.where(mask & np.isfinite(records["rel_l2_pct"]))[0]
            if idx.size:
                amps = np.array([
                    float(ds.sim_params[int(records["sim_id"][i])].get("R_c_amp", 0.0))
                    for i in idx
                ])
                row = int(idx[int(np.argmax(amps))])
        if row >= 0:
            selected.append((regime, row))
    return selected


def _selected_interface_rows(records) -> list[tuple[str, int]]:
    selected = []
    for target in (0.25, 0.50, 0.75):
        row = _row_for_x_I(records, target)
        if row >= 0:
            selected.append((f"x_I≈{target:.2f}", row))
    return selected


def _predict_phys_case(model, ds, sid: int, s: int, target_indices: np.ndarray, config, dt):
    return _prepare_prediction_case(
        model, ds.trajectories, ds.x_grid, ds.y_grid, ds.t_grid, ds.sim_params,
        int(sid), int(s), np.asarray(target_indices, dtype=int), config=config, dt=dt,
    )


def _slice_points_for_case(ds, sid: int) -> tuple[float, float]:
    params = ds.sim_params[int(sid)]
    name = getattr(ds.problem, "name", "")
    if name in ("source", "source_itr"):
        return float(params["y_h"]), float(params["x_h"])
    if name == "forcing":
        spatial_params = params.get("spatial_params", {})
        y_value = float(spatial_params.get("y_c", 0.5))
        return y_value, 0.25
    if name == "interfaces":
        interface_x = float(params.get("interface_x", 0.5))
        left_node, _right_node = _interface_flanking_nodes_from_grid(ds.x_grid, interface_x)
        return 0.5, float(ds.x_grid[left_node])
    return 0.5, 0.5


def _patch_for_case(ds, sid: int):
    params = ds.sim_params[int(sid)]
    if all(k in params for k in ("x_h", "y_h", "w_h", "h_h")):
        return (
            float(params["x_h"]), float(params["y_h"]),
            float(params["w_h"]), float(params["h_h"]),
        )
    return None


def _row_metric_text(records, row: int, extra: str | None = None) -> str:
    bits = []
    rel = records["rel_l2_pct"][row]
    iface = records["iface_rel_l2_pct"][row]
    if np.isfinite(rel):
        bits.append(f"L2={rel:.2f}%")
    if np.isfinite(iface):
        bits.append(f"iface={iface:.2f}%")
    if extra:
        bits.append(extra)
    return "\n".join(bits)


# ============================================================
# TEST-ERROR SUMMARIES (2x3)
# ============================================================

def plot_forcing_test_error_summary(records, save_path=None):
    """Forcing benchmark per-sample error summary (2x3)."""
    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(13.5, 8))
        _hist(axes[0, 0], records["rel_l2_pct"], "rel-L2 (%)", "Global error")
        _hist(axes[0, 1], records["iface_rel_l2_pct"], "interface rel-L2 (%)", "Interface-band error")
        _binned_line(axes[0, 2], records["t_bar"], records["rel_l2_pct"], 8,
                     "lead time Δt", "Error vs lead time")
        _box_by_category(axes[1, 0], records, "temporal_family", TEMPORAL_ORDER,
                         "rel_l2_pct", "temporal family", "Error by temporal family")
        _box_by_category(axes[1, 1], records, "spatial_family", SPATIAL_ORDER,
                         "rel_l2_pct", "spatial family", "Error by spatial family")
        grid = _cat_cat_median(records, "temporal_family", TEMPORAL_ORDER,
                               "spatial_family", SPATIAL_ORDER, "rel_l2_pct")
        _draw_heatmap(axes[1, 2], grid, TEMPORAL_ORDER, SPATIAL_ORDER,
                      "Temporal × spatial median")
        return _save_figure(fig, save_path, "paper", "forcing_test_error_summary")


def plot_source_test_error_summary(records, save_path=None, panel6="lead_time"):
    """Source benchmark per-sample error summary (2x3).

    ``panel6`` swaps the sixth panel between ``"lead_time"`` and ``"amplitude"``.
    """
    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(13.5, 8))
        _hist(axes[0, 0], records["rel_l2_pct"], "rel-L2 (%)", "Global error")
        _hist(axes[0, 1], records["iface_rel_l2_pct"], "interface rel-L2 (%)", "Interface-band error")
        _box_by_category(axes[0, 2], records, "regime", REGIME_ORDER,
                         "rel_l2_pct", "regime", "Error by patch regime")
        _binned_line(axes[1, 0], records["x_h"], records["rel_l2_pct"], 8,
                     "patch x_h", "Error vs x_h")
        dist = np.abs(records["x_h"] - 0.5)
        _binned_line(axes[1, 1], dist, records["rel_l2_pct"], 8,
                     "|x_h − 0.5|", "Error vs interface distance")
        if panel6 == "amplitude":
            _binned_line(axes[1, 2], records["A"], records["rel_l2_pct"], 8,
                         "amplitude A", "Error vs amplitude")
        else:
            _binned_line(axes[1, 2], records["t_bar"], records["rel_l2_pct"], 8,
                         "lead time Δt", "Error vs lead time")
        return _save_figure(fig, save_path, "paper", "source_test_error_summary")


def plot_source_itr_test_error_summary(records, save_path=None):
    """source_itr per-sample error summary (2x3), void-severity focused.

    Top row mirrors the source summary (global + interface-band histograms,
    error by patch regime). Bottom row swaps the patch-geometry panels for the
    Gaussian-void resistance parameters: depth ``R_c_amp``, width ``R_c_sigma``,
    and location ``R_c_y0``. These columns are NaN for non-source_itr records;
    the binning helpers drop non-finite values, so the panels stay empty in that
    case rather than erroring.
    """
    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(13.5, 8))
        _hist(axes[0, 0], records["rel_l2_pct"], "rel-L2 (%)", "Global error")
        _hist(axes[0, 1], records["iface_rel_l2_pct"], "interface rel-L2 (%)", "Interface-band error")
        _box_by_category(axes[0, 2], records, "regime", REGIME_ORDER,
                         "rel_l2_pct", "regime", "Error by patch regime")
        _binned_line(axes[1, 0], records["R_c_amp"], records["rel_l2_pct"], 8,
                     "void depth R_c_amp", "Error vs void depth")
        _binned_line(axes[1, 1], records["R_c_sigma"], records["rel_l2_pct"], 8,
                     "void width R_c_sigma", "Error vs void width")
        _binned_line(axes[1, 2], records["R_c_y0"], records["rel_l2_pct"], 8,
                     "void center R_c_y0", "Error vs void location")
        return _save_figure(fig, save_path, "paper", "source_itr_test_error_summary")


def plot_interfaces_test_error_summary(records, save_path=None):
    """Interfaces benchmark per-sample error summary (2x3).

    The interface-band panel uses ``iface_rel_l2_pct``, which the records
    generator computes around each sample's own ``x_I`` (dynamic band).
    """
    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(13.5, 8))
        _hist(axes[0, 0], records["rel_l2_pct"], "rel-L2 (%)", "Global error")
        _hist(axes[0, 1], records["iface_rel_l2_pct"], "interface rel-L2 (%)",
              "Dynamic interface-band error")
        _binned_line(axes[0, 2], records["x_I"], records["rel_l2_pct"], 8,
                     "interface x_I", "Error vs x_I")
        _binned_line(axes[1, 0], records["R_c"], records["rel_l2_pct"], 8,
                     "contact resistance R_c", "Error vs R_c")
        grid, x_edges, y_edges = _binned_2d_median(
            records["x_I"], records["R_c"], records["rel_l2_pct"], 6)
        _draw_binned_2d(axes[1, 1], grid, x_edges, y_edges, "x_I", "R_c", "x_I × R_c median")
        _binned_line(axes[1, 2], records["t_bar"], records["rel_l2_pct"], 8,
                     "lead time Δt", "Error vs lead time")
        return _save_figure(fig, save_path, "paper", "interfaces_test_error_summary")


# ============================================================
# PREDICTION / TRUTH / RESIDUAL (3x3)
# ============================================================

def plot_forcing_prediction_truth_residual(model, ds, records, save_path=None):
    """Forcing truth/prediction/residual, one row per temporal family, sampled
    during the forcing-active window (target time t_j <= FORCING_ACTIVE_T_MAX).

    Selecting the target snapshot from the forcing phase keeps y-direction
    structure (2D coupling) visible instead of the late, near-1D x-diffusion
    relaxation. Spatial family is secondary, so per-row selection prefers a
    rotating spatial profile but relaxes it (then the time window) via fallbacks.
    """
    t_j = records["t_s"] + records["t_bar"]
    active = t_j <= FORCING_ACTIVE_T_MAX
    active_relaxed = t_j <= 0.25
    preferred_spatial = {"sin": "uniform", "exp": "gaussian",
                         "pulse_train": "patch", "exp_train": "triangle"}

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(4, 3, figsize=(12.5, 15.3))
        for r, fam in enumerate(TEMPORAL_ORDER):
            fam_mask = _mask_in(records["temporal_family"], fam)
            sp_mask = _mask_in(records["spatial_family"], preferred_spatial[fam])
            row = -1
            for mask in (fam_mask & active & sp_mask,
                         fam_mask & active,
                         fam_mask & active_relaxed,
                         fam_mask):
                row = _representative_row(records, mask)
                if row >= 0:
                    break
            if row < 0:
                continue
            sid, s, j = (int(records[k][row]) for k in ("sim_id", "s", "j"))
            Y_true, Y_pred = _predict_norm_fields(model, ds, sid, s, j)
            _render_tpr_row(axes[r], ds.x_grid, ds.y_grid, Y_true, Y_pred,
                            interface_positions=[0.5], row_label=_row_label(ds, records, row),
                            col_titles=(r == 0))
        return _save_figure(fig, save_path, "paper", "forcing_prediction_truth_residual")


def plot_source_prediction_truth_residual(model, ds, records, save_path=None):
    """Source benchmark truth/prediction/residual for left/near/right patches."""
    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(3, 3, figsize=(12.5, 11.5))
        for r, regime in enumerate(REGIME_ORDER):
            row = _representative_row(records, _mask_in(records["regime"], regime))
            if row < 0:
                continue
            sid, s, j = (int(records[k][row]) for k in ("sim_id", "s", "j"))
            params = ds.sim_params[sid]
            patch = (
                float(params["x_h"]), float(params["y_h"]),
                float(params["w_h"]), float(params["h_h"]),
            )
            Y_true, Y_pred = _predict_norm_fields(model, ds, sid, s, j)
            _render_tpr_row(axes[r], ds.x_grid, ds.y_grid, Y_true, Y_pred,
                            interface_positions=[float(params.get("interface_x", 0.5))],
                            patch=patch, row_label=_row_label(ds, records, row),
                            col_titles=(r == 0))
        return _save_figure(fig, save_path, "paper", "source_prediction_truth_residual")


def plot_source_itr_prediction_truth_residual(model, ds, records, save_path=None):
    """source_itr truth/prediction/residual, one row per patch regime.

    Within each regime the most severe void (largest ``R_c_amp``) case is chosen
    via ``_selected_source_rows(..., prefer_void=True)`` so the residual panels
    show where the spatially-varying interface resistance stresses the model.
    """
    selected = _selected_source_rows(records, ds, prefer_void=True)
    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(3, 3, figsize=(12.5, 11.5))
        for r, (_regime, row) in enumerate(selected):
            sid, s, j = (int(records[k][row]) for k in ("sim_id", "s", "j"))
            params = ds.sim_params[sid]
            patch = (
                float(params["x_h"]), float(params["y_h"]),
                float(params["w_h"]), float(params["h_h"]),
            )
            Y_true, Y_pred = _predict_norm_fields(model, ds, sid, s, j)
            _render_tpr_row(axes[r], ds.x_grid, ds.y_grid, Y_true, Y_pred,
                            interface_positions=[float(params.get("interface_x", 0.5))],
                            patch=patch, row_label=_row_label(ds, records, row),
                            col_titles=(r == 0))
        return _save_figure(fig, save_path, "paper", "source_itr_prediction_truth_residual")


def plot_interfaces_prediction_truth_residual(model, ds, records, save_path=None):
    """Interfaces benchmark truth/prediction/residual at x_I≈0.25/0.50/0.75."""
    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(3, 3, figsize=(12.5, 11.5))
        for r, target in enumerate((0.25, 0.50, 0.75)):
            row = _row_for_x_I(records, target)
            if row < 0:
                continue
            sid, s, j = (int(records[k][row]) for k in ("sim_id", "s", "j"))
            x_I = float(ds.sim_params[sid].get("interface_x", records["x_I"][row]))
            Y_true, Y_pred = _predict_norm_fields(model, ds, sid, s, j)
            _render_tpr_row(axes[r], ds.x_grid, ds.y_grid, Y_true, Y_pred,
                            interface_positions=[x_I], row_label=_row_label(ds, records, row),
                            col_titles=(r == 0))
        return _save_figure(fig, save_path, "paper", "interfaces_prediction_truth_residual")


# ============================================================
# TEMPERATURE PROFILES (FVM vs FNO line slices)
# ============================================================

def _render_temperature_profile_row(axes_row, ds, records, row: int, label: str,
                                    case, *, col_titles: bool = False) -> None:
    ax_x, ax_y, ax_err = axes_row
    sid = int(records["sim_id"][row])
    y_value, x_value = _slice_points_for_case(ds, sid)
    y_idx = _nearest_index(ds.y_grid, y_value)
    x_idx = _nearest_index(ds.x_grid, x_value)
    y_actual = float(ds.y_grid[y_idx])
    x_actual = float(ds.x_grid[x_idx])

    Y_true = np.asarray(case["Y_true"][0], dtype=np.float64)
    Y_pred = np.asarray(case["Y_pred"][0], dtype=np.float64)
    err = Y_pred - Y_true
    true_x = Y_true[:, y_idx]
    pred_x = Y_pred[:, y_idx]
    true_y = Y_true[x_idx, :]
    pred_y = Y_pred[x_idx, :]
    err_x = err[:, y_idx]
    err_y = err[x_idx, :]
    rmse_x = float(np.sqrt(np.mean(err_x ** 2)))
    rmse_y = float(np.sqrt(np.mean(err_y ** 2)))

    interface_x = float(case["interface_x"])
    patch = _patch_for_case(ds, sid)
    ax_x.plot(ds.x_grid, true_x, color="black", label="FVM")
    ax_x.plot(ds.x_grid, pred_x, color="C3", linestyle="--", label="FNO")
    ax_x.axvline(interface_x, color="0.35", linestyle=":", linewidth=1.1)
    if patch is not None:
        x_h, _y_h, w_h, _h_h = patch
        ax_x.axvspan(x_h - 0.5 * w_h, x_h + 0.5 * w_h, color="C2", alpha=0.12)
    ax_x.set_xlabel("x")
    ax_x.set_ylabel((label + "\n" if label else "") + "T [K]", fontsize=8)
    ax_x.grid(True)

    ax_y.plot(ds.y_grid, true_y, color="black", label="FVM")
    ax_y.plot(ds.y_grid, pred_y, color="C3", linestyle="--", label="FNO")
    if patch is not None:
        _x_h, y_h, _w_h, h_h = patch
        ax_y.axvspan(y_h - 0.5 * h_h, y_h + 0.5 * h_h, color="C2", alpha=0.12)
    ax_y.set_xlabel("y")
    ax_y.set_ylabel("T [K]")
    ax_y.grid(True)

    ax_err.plot(ds.x_grid, err_x, color="C0", label=f"x slice, y={_format_coord(y_actual)}")
    ax_err.plot(ds.y_grid, err_y, color="C1", linestyle="--", label=f"y slice, x={_format_coord(x_actual)}")
    ax_err.axhline(0.0, color="0.35", linestyle=":", linewidth=1.0)
    ax_err.set_xlabel("coordinate")
    ax_err.set_ylabel("FNO - FVM [K]")
    ax_err.grid(True)
    metric_text = _row_metric_text(records, row, extra=f"RMSE_x={rmse_x:.3g}K\nRMSE_y={rmse_y:.3g}K")
    if metric_text:
        ax_err.text(
            0.03, 0.96, metric_text,
            transform=ax_err.transAxes, va="top", ha="left",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.8"},
        )

    if col_titles:
        ax_x.set_title(f"T vs x at y={_format_coord(y_actual)}")
        ax_y.set_title(f"T vs y at x={_format_coord(x_actual)}")
        ax_err.set_title("Signed profile error")
        ax_x.legend(loc="best")
        ax_err.legend(loc="best")


def _build_temperature_profile_figure(model, ds, records, selected, config, dt,
                                      name: str, save_path):
    if not selected:
        with plt.rc_context(PLOT_STYLE):
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            ax.text(0.5, 0.5, "No cases available", ha="center", va="center")
            ax.axis("off")
            return _save_figure(fig, save_path, "paper", name)

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(len(selected), 3, figsize=(15, 4.0 * len(selected)), squeeze=False)
        for r, (label, row) in enumerate(selected):
            sid, s, j = (int(records[k][row]) for k in ("sim_id", "s", "j"))
            case = _predict_phys_case(model, ds, sid, s, np.array([j], dtype=int), config, dt)
            full_label = f"{label}\n{_row_label(ds, records, row)}"
            _render_temperature_profile_row(
                axes[r], ds, records, row, full_label, case, col_titles=(r == 0)
            )
        fig.tight_layout()
        return _save_figure(fig, save_path, "paper", name)


def plot_forcing_temperature_profiles(model, ds, records, config, dt, save_path=None):
    """Forcing FVM-vs-FNO temperature line profiles, one row per temporal family."""
    return _build_temperature_profile_figure(
        model, ds, records, _selected_forcing_rows(records), config, dt,
        "forcing_temperature_profiles", save_path,
    )


def plot_source_temperature_profiles(model, ds, records, config, dt, save_path=None):
    """Source FVM-vs-FNO temperature line profiles for left/near/right patches."""
    return _build_temperature_profile_figure(
        model, ds, records, _selected_source_rows(records), config, dt,
        "source_temperature_profiles", save_path,
    )


def plot_source_itr_temperature_profiles(model, ds, records, config, dt, save_path=None):
    """Source-ITR FVM-vs-FNO temperature line profiles, preferring visible voids."""
    return _build_temperature_profile_figure(
        model, ds, records, _selected_source_rows(records, ds=ds, prefer_void=True), config, dt,
        "source_itr_temperature_profiles", save_path,
    )


def plot_interfaces_temperature_profiles(model, ds, records, config, dt, save_path=None):
    """Interfaces FVM-vs-FNO temperature line profiles at low/mid/high interface x."""
    return _build_temperature_profile_figure(
        model, ds, records, _selected_interface_rows(records), config, dt,
        "interfaces_temperature_profiles", save_path,
    )


# ============================================================
# INTERFACE CONTACT-JUMP OVER LEAD TIME (per benchmark)
# ============================================================
# The FNO predicts nodal temperatures; we postprocess them with the known
# imperfect-interface law to obtain the *implied* physical contact jump in
# Kelvin: dT_contact = R_c * G * (T_left - T_right). These figures headline the
# genuinely new content — the absolute-Kelvin jump magnitude and its absolute
# error over lead time. Per-case RELATIVE L2 is omitted as a headline because for
# a scalar R_c / fixed interface it equals the node-to-node jump rel error (the
# R_c*G scale factor cancels per case).

_JUMP_N_TARGETS = 20  # future targets rolled out per source snapshot


def _jump_case(model, ds, sid: int, s: int, config, dt, n_targets: int = _JUMP_N_TARGETS):
    """Roll one source snapshot forward; return contact-jump truth/pred arrays (K).

    Reuses ``_prepare_prediction_case`` (physical-Kelvin Y_true/Y_pred of shape
    ``(n_targets, Nx, Ny)``) and converts each field to the contact jump with the
    same interface law the FV solver uses. Emits a debug line per case so a
    silently wrong ``G`` (mismatched interface_x / R_c / conductivities) is caught.
    """
    target_indices = _future_target_indices(int(s), n_targets, len(ds.t_grid))
    case = _prepare_prediction_case(
        model, ds.trajectories, ds.x_grid, ds.y_grid, ds.t_grid,
        ds.sim_params, int(sid), int(s), target_indices, config=config, dt=dt,
    )
    interface_x = float(case["interface_x"])
    R_c = float(case["R_c"])
    interface_R = _interface_R_for_sim(ds, int(sid))
    k_left, k_right = _resolve_layer_conductivities(ds, int(sid), config)

    left_node, right_node = _interface_flanking_nodes_from_grid(ds.x_grid, interface_x)
    h_L = float(interface_x) - float(ds.x_grid[left_node])
    h_R = float(ds.x_grid[right_node]) - float(interface_x)
    G = _interface_conductance_G(ds.x_grid, interface_x, interface_R, k_left, k_right)

    true_jump = _interface_contact_jump_map(case["Y_true"], ds.x_grid, interface_x, interface_R, k_left, k_right)
    pred_jump = _interface_contact_jump_map(case["Y_pred"], ds.x_grid, interface_x, interface_R, k_left, k_right)
    node_true = _interface_jump_map(case["Y_true"], left_node, right_node)
    node_peak = float(np.max(np.abs(node_true))) if node_true.size else 0.0
    contact_peak = float(np.max(np.abs(true_jump))) if true_jump.size else 0.0

    print(
        f"[jump] sid={int(sid)} s={int(s)} x_I={interface_x:.4f} R_c={R_c:.4f} "
        f"k_L={k_left:.3f} k_R={k_right:.3f} h_L={h_L:.4f} h_R={h_R:.4f} "
        f"G={_debug_scalar(G):.4f} R_c*G={_debug_scalar(np.asarray(interface_R) * np.asarray(G)):.4f} "
        f"max|node|={node_peak:.4g} max|contact|={contact_peak:.4g}"
    )

    return {
        "t_bars": np.asarray(case["t_bars"], dtype=np.float64),
        "y_grid": np.asarray(ds.y_grid, dtype=np.float64),
        "true_jump": true_jump,
        "pred_jump": pred_jump,
        "true_red": _contact_jump_reductions(true_jump),
        "pred_red": _contact_jump_reductions(pred_jump),
        "metrics": _jump_error_metrics(pred_jump, true_jump),
        "interface_x": interface_x,
        "R_c": R_c,
        "interface_R": interface_R,
    }


def _render_jump_row(axes_row, case, jump_abs: float, mag_ymax: float, *,
                     row_label: str = "", col_titles: bool = False) -> None:
    """Render one truth-jump | pred-jump | magnitude-vs-time row (absolute K)."""
    ax_t, ax_p, ax_c = axes_row
    t_bars = case["t_bars"]
    y_grid = case["y_grid"]

    pcm_t = ax_t.pcolormesh(t_bars, y_grid, case["true_jump"].T, cmap="coolwarm",
                            vmin=-jump_abs, vmax=jump_abs, shading="auto")
    pcm_p = ax_p.pcolormesh(t_bars, y_grid, case["pred_jump"].T, cmap="coolwarm",
                            vmin=-jump_abs, vmax=jump_abs, shading="auto")
    cbar_label = r"$\Delta T_\mathrm{contact}$ [K]"
    ax_t.figure.colorbar(pcm_t, ax=ax_t, fraction=0.046, pad=0.04, label=cbar_label)
    ax_p.figure.colorbar(pcm_p, ax=ax_p, fraction=0.046, pad=0.04, label=cbar_label)
    for ax in (ax_t, ax_p):
        ax.set_xlabel(r"lead time $\bar{t}$")
    ax_t.set_ylabel((row_label + "\n" if row_label else "") + "y", fontsize=8)
    ax_p.set_ylabel("y")

    # Column 3: mean-|jump| magnitude vs lead time (truth vs pred), absolute K.
    # CAVEAT: this y-reduced magnitude can overlap while pointwise error is large;
    # any error claim must lean on abs_rmse_K (annotated), not on curve overlap.
    ax_c.plot(t_bars, case["true_red"]["mean_abs"], "-o", ms=3, color="C0", label="truth")
    ax_c.plot(t_bars, case["pred_red"]["mean_abs"], "--s", ms=3, color="C3", label="pred")
    ax_c.set_xlabel(r"lead time $\bar{t}$")
    ax_c.set_ylabel(r"mean $|\Delta T_\mathrm{contact}|$ [K]")
    if mag_ymax > 0:
        ax_c.set_ylim(0.0, mag_ymax * 1.08)
    ax_c.legend(loc="upper right")
    m = case["metrics"]
    ax_c.text(
        0.03, 0.96,
        f"abs RMSE={m['abs_rmse_K']:.3g} K\n"
        f"peak err={m['peak_abs_err_K']:.3g} K\n"
        f"truth peak={m['truth_peak_jump_K']:.3g} K",
        transform=ax_c.transAxes, va="top", ha="left",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.8"},
    )

    if col_titles:
        ax_t.set_title("Truth contact jump")
        ax_p.set_title("Predicted contact jump")
        ax_c.set_title(r"Mean $|$jump$|$ vs lead time")


def _build_jump_figure(cases, labels, name: str, save_path):
    """Assemble a per-benchmark contact-jump figure with shared symmetric scales."""
    if not cases:
        with plt.rc_context(PLOT_STYLE):
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            ax.text(0.5, 0.5, "No cases available", ha="center", va="center")
            ax.axis("off")
            return _save_figure(fig, save_path, "paper", name)

    jump_abs = max(
        [float(np.max(np.abs(c["true_jump"]))) for c in cases]
        + [float(np.max(np.abs(c["pred_jump"]))) for c in cases]
        + [1e-8]
    )
    mag_ymax = max(
        [float(np.max(c["true_red"]["mean_abs"])) for c in cases]
        + [float(np.max(c["pred_red"]["mean_abs"])) for c in cases]
        + [1e-8]
    )
    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(len(cases), 3, figsize=(15, 4.1 * len(cases)), squeeze=False)
        for r, (case, label) in enumerate(zip(cases, labels)):
            _render_jump_row(axes[r], case, jump_abs, mag_ymax,
                             row_label=label, col_titles=(r == 0))
        return _save_figure(fig, save_path, "paper", name)


def plot_forcing_interface_jump(model, ds, records, config, dt, save_path=None):
    """Forcing contact-jump over lead time, one row per temporal family.

    Picks the same representative cases as the forcing TPR figure (forcing-active
    window, rotating spatial profile), then rolls each forward to show the
    implied imperfect-interface temperature jump in Kelvin: truth map | pred map |
    mean-|jump| vs lead time.
    """
    t_j = records["t_s"] + records["t_bar"]
    active = t_j <= FORCING_ACTIVE_T_MAX
    active_relaxed = t_j <= 0.25
    preferred_spatial = {"sin": "uniform", "exp": "gaussian",
                         "pulse_train": "patch", "exp_train": "triangle"}

    cases, labels = [], []
    for fam in TEMPORAL_ORDER:
        fam_mask = _mask_in(records["temporal_family"], fam)
        sp_mask = _mask_in(records["spatial_family"], preferred_spatial[fam])
        row = -1
        for mask in (fam_mask & active & sp_mask, fam_mask & active,
                     fam_mask & active_relaxed, fam_mask):
            row = _representative_row(records, mask)
            if row >= 0:
                break
        if row < 0:
            continue
        sid, s = int(records["sim_id"][row]), int(records["s"][row])
        case = _jump_case(model, ds, sid, s, config, dt)
        cases.append(case)
        labels.append(f"{fam}\nR_c={case['R_c']:.2f}")
    return _build_jump_figure(cases, labels, "forcing_interface_jump", save_path)


def plot_source_interface_jump(model, ds, records, config, dt, save_path=None):
    """Source contact-jump over lead time for left/near/right patch regimes."""
    cases, labels = [], []
    for regime in REGIME_ORDER:
        row = _representative_row(records, _mask_in(records["regime"], regime))
        if row < 0:
            continue
        sid, s = int(records["sim_id"][row]), int(records["s"][row])
        case = _jump_case(model, ds, sid, s, config, dt)
        cases.append(case)
        labels.append(f"{regime}\nx_I={case['interface_x']:.2f}, R_c={case['R_c']:.2f}")
    return _build_jump_figure(cases, labels, "source_interface_jump", save_path)


def plot_interfaces_interface_jump(model, ds, records, config, dt, save_path=None):
    """Interfaces contact-jump over lead time at x_I≈0.25/0.50/0.75."""
    cases, labels = [], []
    for target in (0.25, 0.50, 0.75):
        row = _row_for_x_I(records, target)
        if row < 0:
            continue
        sid, s = int(records["sim_id"][row]), int(records["s"][row])
        case = _jump_case(model, ds, sid, s, config, dt)
        cases.append(case)
        labels.append(f"x_I≈{target:.2f}\nx_I={case['interface_x']:.2f}, R_c={case['R_c']:.2f}")
    return _build_jump_figure(cases, labels, "interfaces_interface_jump", save_path)


def _jump_profile_case(model, ds, records, row: int, config, dt,
                       n_targets: int = _JUMP_N_TARGETS):
    sid, s, j = (int(records[k][row]) for k in ("sim_id", "s", "j"))
    future = _future_target_indices(s, n_targets, len(ds.t_grid))
    target_indices = np.unique(np.concatenate([future, np.array([j], dtype=int)]))
    case = _predict_phys_case(model, ds, sid, s, target_indices, config, dt)
    interface_x = float(case["interface_x"])
    interface_R = _interface_R_for_sim(ds, sid)
    k_left, k_right = _resolve_layer_conductivities(ds, sid, config)
    true_jump = _interface_contact_jump_map(
        case["Y_true"], ds.x_grid, interface_x, interface_R, k_left, k_right
    )
    pred_jump = _interface_contact_jump_map(
        case["Y_pred"], ds.x_grid, interface_x, interface_R, k_left, k_right
    )
    fixed_idx = int(np.where(target_indices == j)[0][0])
    return {
        "sid": sid,
        "s": s,
        "j": j,
        "t_bars": np.asarray(case["t_bars"], dtype=np.float64),
        "y_grid": np.asarray(ds.y_grid, dtype=np.float64),
        "true_jump": true_jump,
        "pred_jump": pred_jump,
        "fixed_idx": fixed_idx,
        "interface_x": interface_x,
        "R_c": float(case["R_c"]),
    }


def _render_jump_profile_row(axes_row, ds, records, row: int, label: str, case,
                             *, col_titles: bool = False) -> None:
    ax_t, ax_te, ax_y, ax_ye = axes_row
    sid = int(records["sim_id"][row])
    y_value, _x_value = _slice_points_for_case(ds, sid)
    y_idx = _nearest_index(ds.y_grid, y_value)
    y_actual = float(ds.y_grid[y_idx])
    t_bars = case["t_bars"]
    y_grid = case["y_grid"]
    jt = case["true_jump"]
    jp = case["pred_jump"]
    diff = jp - jt
    fixed_idx = int(case["fixed_idx"])
    fixed_t = float(t_bars[fixed_idx])
    rmse_time = float(np.sqrt(np.mean(diff[:, y_idx] ** 2)))
    rmse_y = float(np.sqrt(np.mean(diff[fixed_idx, :] ** 2)))

    ax_t.plot(t_bars, jt[:, y_idx], color="black", label="FVM")
    ax_t.plot(t_bars, jp[:, y_idx], color="C3", linestyle="--", label="FNO")
    ax_t.axvline(fixed_t, color="0.35", linestyle=":", linewidth=1.0)
    ax_t.set_xlabel(r"lead time $\bar{t}$")
    ax_t.set_ylabel((label + "\n" if label else "") + r"$\Delta T_\mathrm{contact}$ [K]", fontsize=8)
    ax_t.grid(True)

    ax_te.plot(t_bars, diff[:, y_idx], color="C0")
    ax_te.axhline(0.0, color="0.35", linestyle=":", linewidth=1.0)
    ax_te.axvline(fixed_t, color="0.35", linestyle=":", linewidth=1.0)
    ax_te.set_xlabel(r"lead time $\bar{t}$")
    ax_te.set_ylabel("FNO - FVM [K]")
    ax_te.grid(True)

    ax_y.plot(y_grid, jt[fixed_idx, :], color="black", label="FVM")
    ax_y.plot(y_grid, jp[fixed_idx, :], color="C3", linestyle="--", label="FNO")
    ax_y.axvline(y_actual, color="0.35", linestyle=":", linewidth=1.0)
    ax_y.set_xlabel("y")
    ax_y.set_ylabel(r"$\Delta T_\mathrm{contact}$ [K]")
    ax_y.grid(True)

    ax_ye.plot(y_grid, diff[fixed_idx, :], color="C1")
    ax_ye.axhline(0.0, color="0.35", linestyle=":", linewidth=1.0)
    ax_ye.axvline(y_actual, color="0.35", linestyle=":", linewidth=1.0)
    ax_ye.set_xlabel("y")
    ax_ye.set_ylabel("FNO - FVM [K]")
    ax_ye.grid(True)
    metric_text = _row_metric_text(records, row, extra=f"RMSE_t={rmse_time:.3g}K\nRMSE_y={rmse_y:.3g}K")
    if metric_text:
        ax_ye.text(
            0.03, 0.96, metric_text,
            transform=ax_ye.transAxes, va="top", ha="left",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.8"},
        )

    if col_titles:
        ax_t.set_title(f"Jump vs time at y={_format_coord(y_actual)}")
        ax_te.set_title("Jump error vs time")
        ax_y.set_title(f"Jump vs y at lead={_format_coord(fixed_t)}")
        ax_ye.set_title("Jump error vs y")
        ax_t.legend(loc="best")
        ax_y.legend(loc="best")


def _build_interface_jump_profile_figure(model, ds, records, selected, config, dt,
                                         name: str, save_path):
    if not selected:
        with plt.rc_context(PLOT_STYLE):
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            ax.text(0.5, 0.5, "No cases available", ha="center", va="center")
            ax.axis("off")
            return _save_figure(fig, save_path, "paper", name)

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(len(selected), 4, figsize=(18, 3.9 * len(selected)), squeeze=False)
        for r, (label, row) in enumerate(selected):
            case = _jump_profile_case(model, ds, records, row, config, dt)
            full_label = f"{label}\n{_row_label(ds, records, row)}"
            _render_jump_profile_row(
                axes[r], ds, records, row, full_label, case, col_titles=(r == 0)
            )
        fig.tight_layout()
        return _save_figure(fig, save_path, "paper", name)


def plot_forcing_interface_jump_profiles(model, ds, records, config, dt, save_path=None):
    """Forcing contact-jump line profiles and errors, one row per temporal family."""
    return _build_interface_jump_profile_figure(
        model, ds, records, _selected_forcing_rows(records), config, dt,
        "forcing_interface_jump_profiles", save_path,
    )


def plot_source_interface_jump_profiles(model, ds, records, config, dt, save_path=None):
    """Source contact-jump line profiles and errors for left/near/right patches."""
    return _build_interface_jump_profile_figure(
        model, ds, records, _selected_source_rows(records), config, dt,
        "source_interface_jump_profiles", save_path,
    )


def plot_source_itr_interface_jump_profiles(model, ds, records, config, dt, save_path=None):
    """Source-ITR contact-jump line profiles using the per-row R_c(y) law."""
    return _build_interface_jump_profile_figure(
        model, ds, records, _selected_source_rows(records, ds=ds, prefer_void=True), config, dt,
        "source_itr_interface_jump_profiles", save_path,
    )


def plot_interfaces_interface_jump_profiles(model, ds, records, config, dt, save_path=None):
    """Interfaces contact-jump line profiles and errors at low/mid/high interface x."""
    return _build_interface_jump_profile_figure(
        model, ds, records, _selected_interface_rows(records), config, dt,
        "interfaces_interface_jump_profiles", save_path,
    )


# ============================================================
# CROSS-BENCHMARK SYNTHESIS
# ============================================================

def plot_all_benchmarks_error_summary(records_by_benchmark, save_path=None):
    """Cross-benchmark *typical + ordinary-high* error (2x2 metric grid).

    Each panel is one metric with benchmarks along the x-axis: bars are the
    **median** (typical performance), an overlaid marker is **p90** (ordinary
    high). Rare-failure statistics (p99, max, tail amplification) deliberately
    live in ``plot_all_benchmarks_tail_errors`` instead — this figure and the
    tail figure must not share statistics. Physical-Kelvin panels (``*_K``) and
    normalized-percent panels (``*_pct``) keep separate y-axes by construction.
    """
    names = list(records_by_benchmark.keys())
    panels = (
        ("rmse_K", "RMSE (K)", "Field error magnitude"),
        ("node_jump_rmse_K", "node-jump RMSE (K)", "Interface-jump error magnitude"),
        ("gnrmse_pct", "gNRMSE (%)", "Scale-normalized field difficulty"),
        ("node_jump_gnrmse_pct", "node-jump gNRMSE (%)", "Scale-normalized interface difficulty"),
    )
    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        for ax, (col, ylabel, title) in zip(axes.ravel(), panels):
            stats_by_name = {
                n: _tail_stats(records_by_benchmark[n].get(col, np.array([np.nan])))
                for n in names
            }
            _grouped_tail_bars(
                ax, names, stats_by_name,
                bar_stat="median", marker_stats=("p90",),
                ylabel=ylabel, title=title,
            )
        fig.suptitle("Cross-benchmark error summary — median (bar) + p90 (marker)")
        return _save_figure(fig, save_path, "paper", "all_benchmarks_error_summary")


def plot_all_benchmarks_prediction_truth_residual(forcing_ctx, source_ctx,
                                                  interfaces_ctx, source_itr_ctx,
                                                  save_path=None):
    """One representative case per benchmark (4x3).

    Each ``*_ctx`` is ``(model, ds, records)``. Row 1 is the forcing hard case
    (max global error), row 2 the source near-interface case, row 3 the
    interfaces off-center case (max ``|x_I − 0.5|``), row 4 the source_itr
    near-interface most-severe-void case (max ``R_c_amp``).
    """
    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(4, 3, figsize=(12.5, 15.3))

        f_model, f_ds, f_rec = forcing_ctx
        err = f_rec["rel_l2_pct"]
        finite = np.where(np.isfinite(err))[0]
        if finite.size:
            row = int(finite[int(np.argmax(err[finite]))])
            sid, s, j = (int(f_rec[k][row]) for k in ("sim_id", "s", "j"))
            Y_true, Y_pred = _predict_norm_fields(f_model, f_ds, sid, s, j)
            _render_tpr_row(axes[0], f_ds.x_grid, f_ds.y_grid, Y_true, Y_pred,
                            interface_positions=[0.5],
                            row_label="forcing: " + _row_label(f_ds, f_rec, row),
                            col_titles=True)

        s_model, s_ds, s_rec = source_ctx
        row = _representative_row(s_rec, _mask_in(s_rec["regime"], "near"))
        if row >= 0:
            sid, s, j = (int(s_rec[k][row]) for k in ("sim_id", "s", "j"))
            params = s_ds.sim_params[sid]
            patch = (float(params["x_h"]), float(params["y_h"]),
                     float(params["w_h"]), float(params["h_h"]))
            Y_true, Y_pred = _predict_norm_fields(s_model, s_ds, sid, s, j)
            _render_tpr_row(axes[1], s_ds.x_grid, s_ds.y_grid, Y_true, Y_pred,
                            interface_positions=[float(params.get("interface_x", 0.5))],
                            patch=patch, row_label="source: " + _row_label(s_ds, s_rec, row))

        i_model, i_ds, i_rec = interfaces_ctx
        x_I = i_rec["x_I"]
        finite = np.where(np.isfinite(x_I))[0]
        if finite.size:
            row = int(finite[int(np.argmax(np.abs(x_I[finite] - 0.5)))])
            sid, s, j = (int(i_rec[k][row]) for k in ("sim_id", "s", "j"))
            x_I_val = float(i_ds.sim_params[sid].get("interface_x", i_rec["x_I"][row]))
            Y_true, Y_pred = _predict_norm_fields(i_model, i_ds, sid, s, j)
            _render_tpr_row(axes[2], i_ds.x_grid, i_ds.y_grid, Y_true, Y_pred,
                            interface_positions=[x_I_val],
                            row_label="interfaces: " + _row_label(i_ds, i_rec, row))

        si_model, si_ds, si_rec = source_itr_ctx
        selected = _selected_source_rows(si_rec, si_ds, prefer_void=True)
        near_rows = [row for regime, row in selected if regime == "near"]
        row = near_rows[0] if near_rows else (selected[0][1] if selected else -1)
        if row >= 0:
            sid, s, j = (int(si_rec[k][row]) for k in ("sim_id", "s", "j"))
            params = si_ds.sim_params[sid]
            patch = (float(params["x_h"]), float(params["y_h"]),
                     float(params["w_h"]), float(params["h_h"]))
            Y_true, Y_pred = _predict_norm_fields(si_model, si_ds, sid, s, j)
            _render_tpr_row(axes[3], si_ds.x_grid, si_ds.y_grid, Y_true, Y_pred,
                            interface_positions=[float(params.get("interface_x", 0.5))],
                            patch=patch, row_label="source_itr: " + _row_label(si_ds, si_rec, row))
        return _save_figure(fig, save_path, "paper", "all_benchmarks_prediction_truth_residual")


def _aggregate_benchmark_jump(ctx, n_sims: int = 24, n_targets: int = _JUMP_N_TARGETS,
                              seed: int = 0):
    """Roll a fixed-protocol sample of test sims; stack per-lead-time jump curves.

    Identical protocol across benchmarks for fairness: same sampled count, same
    rollout start (s=0), same target lead-time indices, same fixed seed. Returns
    per-sim stacks of shape ``(n_sims, n_lead)`` for truth/pred magnitude (K),
    absolute jump RMSE (K), and scale-normalized relative jump error (%).
    """
    model, ds, records, config, dt = ctx
    n_total = len(ds.t_grid)
    if n_total < 2:
        return None
    s_start = 0
    target_indices = _future_target_indices(s_start, n_targets, n_total)

    sim_ids = np.unique(records["sim_id"])
    rng = np.random.default_rng(seed)
    if len(sim_ids) > n_sims:
        sim_ids = np.sort(rng.choice(sim_ids, size=n_sims, replace=False))

    truth_mag, pred_mag, abs_err, rel_err = [], [], [], []
    t_bars_ref = None
    for sid in sim_ids:
        case = _prepare_prediction_case(
            model, ds.trajectories, ds.x_grid, ds.y_grid, ds.t_grid,
            ds.sim_params, int(sid), s_start, target_indices, config=config, dt=dt,
        )
        interface_x = float(case["interface_x"])
        interface_R = _interface_R_for_sim(ds, int(sid))
        k_left, k_right = _resolve_layer_conductivities(ds, int(sid), config)
        tj = _interface_contact_jump_map(case["Y_true"], ds.x_grid, interface_x, interface_R, k_left, k_right)
        pj = _interface_contact_jump_map(case["Y_pred"], ds.x_grid, interface_x, interface_R, k_left, k_right)
        diff = pj - tj
        truth_mag.append(np.mean(np.abs(tj), axis=1))
        pred_mag.append(np.mean(np.abs(pj), axis=1))
        abs_err.append(np.sqrt(np.mean(diff ** 2, axis=1)))
        denom = np.sqrt(np.sum(tj ** 2, axis=1))
        rel_err.append(np.sqrt(np.sum(diff ** 2, axis=1)) / np.maximum(denom, 1e-8) * 100.0)
        if t_bars_ref is None:
            t_bars_ref = np.asarray(case["t_bars"], dtype=np.float64)

    if t_bars_ref is None:
        return None
    return {
        "t_bars": t_bars_ref,
        "truth_mag": np.vstack(truth_mag),
        "pred_mag": np.vstack(pred_mag),
        "abs_err": np.vstack(abs_err),
        "rel_err": np.vstack(rel_err),
    }


def plot_all_benchmarks_interface_jump(forcing_ctx, source_ctx, interfaces_ctx,
                                       source_itr_ctx,
                                       save_path=None, n_sims: int = 24, seed: int = 0):
    """Combined cross-benchmark contact-jump synthesis (1x3), honest per-panel lanes.

    Each ``*_ctx`` is ``(model, ds, records, config, dt)``. The three panels each
    own one question and must not share a caption:

    - (a) Magnitude/bias (absolute K): mean-|contact jump| vs lead time, truth
      (solid) vs pred (dashed) per benchmark. Shows under/over-prediction of jump
      magnitude. No difficulty claim.
    - (b) Absolute jump-error magnitude (absolute K): median jump RMSE + IQR. This
      is a SCALE comparison of error magnitude, NOT a difficulty ranking, because
      benchmarks differ in characteristic jump magnitude.
    - (c) Difficulty ranking (scale-normalized): relative jump error, median + IQR.
      This equals the node-to-node jump rel-L2 per case (R_c*G cancels) — the honest
      "which benchmark is hardest to learn" panel.
    """
    names = ("forcing", "source", "interfaces", "source_itr")
    ctxs = (forcing_ctx, source_ctx, interfaces_ctx, source_itr_ctx)
    colors = _BENCHMARK_COLORS
    aggs = {name: _aggregate_benchmark_jump(ctx, n_sims=n_sims, seed=seed)
            for name, ctx in zip(names, ctxs)}

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

        # (a) magnitude / bias view — absolute Kelvin.
        for name in names:
            a = aggs[name]
            if a is None:
                continue
            c = colors[name]
            axes[0].plot(a["t_bars"], a["truth_mag"].mean(axis=0), "-", color=c, label=f"{name} truth")
            axes[0].plot(a["t_bars"], a["pred_mag"].mean(axis=0), "--", color=c, label=f"{name} pred")
        axes[0].set_xlabel(r"lead time $\bar{t}$")
        axes[0].set_ylabel(r"mean $|\Delta T_\mathrm{contact}|$ [K]")
        axes[0].set_title("(a) Jump magnitude vs lead time\n(truth solid, pred dashed)")
        axes[0].legend(fontsize=7)

        # (b) absolute jump-error magnitude — SCALE comparison, not difficulty.
        for name in names:
            a = aggs[name]
            if a is None:
                continue
            c = colors[name]
            med = np.median(a["abs_err"], axis=0)
            q25 = np.percentile(a["abs_err"], 25, axis=0)
            q75 = np.percentile(a["abs_err"], 75, axis=0)
            axes[1].plot(a["t_bars"], med, "-o", ms=3, color=c, label=name)
            axes[1].fill_between(a["t_bars"], q25, q75, alpha=0.2, color=c)
        axes[1].set_xlabel(r"lead time $\bar{t}$")
        axes[1].set_ylabel("abs jump RMSE [K]")
        axes[1].set_title("(b) Absolute jump-error magnitude\n(scale comparison, NOT difficulty)")
        axes[1].legend(fontsize=7)

        # (c) scale-normalized difficulty ranking = node-to-node jump rel-L2 per case.
        for name in names:
            a = aggs[name]
            if a is None:
                continue
            c = colors[name]
            med = np.median(a["rel_err"], axis=0)
            q25 = np.percentile(a["rel_err"], 25, axis=0)
            q75 = np.percentile(a["rel_err"], 75, axis=0)
            axes[2].plot(a["t_bars"], med, "-o", ms=3, color=c, label=name)
            axes[2].fill_between(a["t_bars"], q25, q75, alpha=0.2, color=c)
        axes[2].set_xlabel(r"lead time $\bar{t}$")
        axes[2].set_ylabel("relative jump error (%)")
        axes[2].set_title("(c) Difficulty ranking (scale-normalized)\n= node-to-node jump rel-L2 per case")
        axes[2].legend(fontsize=7)

        return _save_figure(fig, save_path, "paper", "all_benchmarks_interface_jump")


# ============================================================
# TAIL / MAX ERROR STATS PER BENCHMARK PER METRIC
# ============================================================

# Explicit, unit-bearing metric names (what lands in the table/plots) mapped to
# the underlying per-sample record columns. Both are normalized-space percentages.
TAIL_METRICS = {
    "global_rel_l2_pct": "rel_l2_pct",
    "iface_rel_l2_pct": "iface_rel_l2_pct",
}

# Order in which strata dims appear in plots/table.
_TAIL_STAT_KEYS = ("n", "mean", "median", "p90", "p99", "max")
_TAIL_SUMMARY_FIELDS = (
    "benchmark", "metric", "stratum", "group",
    "n", "mean", "median", "p90", "p99", "max",
)


def _tail_stats(values: np.ndarray) -> dict[str, float]:
    """Tail/summary stats over the finite entries of ``values``.

    Returns ``{n, mean, median, p90, p99, max}``. ``n`` is the finite count;
    when ``n == 0`` every stat (except ``n``) is NaN so callers never crash on
    empty groups. Percentiles use ``method="linear"`` (numpy default) so the
    table and any verification test agree exactly.
    """
    v = _finite(values)
    n = int(v.size)
    if n == 0:
        return {"n": 0, "mean": np.nan, "median": np.nan,
                "p90": np.nan, "p99": np.nan, "max": np.nan}
    p90, p99 = np.percentile(v, [90, 99], method="linear")
    return {
        "n": n,
        "mean": float(np.mean(v)),
        "median": float(np.median(v)),
        "p90": float(p90),
        "p99": float(p99),
        "max": float(np.max(v)),
    }


def _tail_amp(stats: dict[str, float]) -> float:
    """Dimensionless tail-amplification ratio ``p99 / median``.

    NaN when the median is non-finite or numerically ~0 so the ratio never
    explodes. Never stored in the CSV; only annotated on the tail figure.
    """
    median = stats.get("median", np.nan)
    p99 = stats.get("p99", np.nan)
    if np.isfinite(median) and median > 1e-12 and np.isfinite(p99):
        return float(p99 / median)
    return np.nan


# Marker style per stat: (matplotlib marker, color). Used by _grouped_tail_bars.
_TAIL_MARKER_STYLE = {
    "mean": ("o", "0.25"),
    "median": ("o", "0.15"),
    "p90": ("o", "#0072B2"),
    "p99": ("s", "#D55E00"),
    "max": ("D", "0.1"),
}


def _grouped_tail_bars(ax, names, stats_by_name, *, bar_stat, ylabel, title,
                       marker_stats=(), annotate_n=False, hollow_stats=(),
                       annotate_tail_amp=False):
    """One-metric grouped bar panel with benchmarks along the x-axis.

    ``stats_by_name`` maps each benchmark name to a stats dict (the output of
    ``_tail_stats``, optionally carrying a derived ``tail_amp``). ``bar_stat`` is
    drawn as soft-filled bars in the benchmark's shared palette color; each of
    ``marker_stats`` is overlaid as a point marker (those in ``hollow_stats`` are
    drawn open to de-emphasize n-dependent stats such as ``max``).

    A benchmark whose ``bar_stat`` is non-finite **keeps its x-axis slot and tick
    label**, draws no bar or marker, and is annotated ``n/a`` so category order is
    identical across panels. ``annotate_n`` writes the finite sample count per
    benchmark; ``annotate_tail_amp`` writes the dimensionless ``tail_amp`` ratio
    as a text label (e.g. ``×2.7``) — it is never plotted on the unit y-axis.
    """
    x = np.arange(len(names))
    bar_vals = [float(stats_by_name[n].get(bar_stat, np.nan)) for n in names]

    for xi, name, val in zip(x, names, bar_vals):
        if np.isfinite(val):
            ax.bar(xi, val, width=0.62, color=_BENCHMARK_COLORS.get(name, "0.5"),
                   alpha=0.85, edgecolor="0.25", linewidth=0.8, zorder=2)

    for stat in marker_stats:
        ys = [float(stats_by_name[n].get(stat, np.nan)) for n in names]
        marker, color = _TAIL_MARKER_STYLE.get(stat, ("o", "k"))
        hollow = stat in hollow_stats
        ax.plot(x, ys, linestyle="none", marker=marker, ms=6, zorder=3,
                mfc="none" if hollow else color, mec=color, mew=1.3,
                label=f"{stat} (n-dependent)" if hollow else stat)

    # Per-benchmark top of plotted content, for stacking text annotations.
    def _top(name):
        vals = [float(stats_by_name[name].get(bar_stat, np.nan))]
        vals += [float(stats_by_name[name].get(s, np.nan)) for s in marker_stats]
        finite = [v for v in vals if np.isfinite(v)]
        return max(finite) if finite else np.nan

    tops = [_top(n) for n in names]
    finite_tops = [t for t in tops if np.isfinite(t)]
    span = max(finite_tops) if finite_tops else 1.0
    pad = 0.02 * span if span > 0 else 0.02

    for xi, name, top in zip(x, names, tops):
        base = (top if np.isfinite(top) else 0.0) + pad
        if not np.isfinite(bar_vals[names.index(name)]):
            ax.text(xi, base, "n/a", ha="center", va="bottom", fontsize=8, color="0.5")
            continue
        line = 0
        if annotate_tail_amp:
            amp = float(stats_by_name[name].get("tail_amp", np.nan))
            label = f"×{amp:.1f}" if np.isfinite(amp) else "×n/a"
            ax.annotate(label, (xi, base), xytext=(0, 2 + 11 * line),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=7, color="0.2")
            line += 1
        if annotate_n:
            n_val = int(stats_by_name[name].get("n", 0))
            ax.annotate(f"n={n_val}", (xi, base), xytext=(0, 2 + 11 * line),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=6.5, color="0.45")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.margins(y=0.18)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if marker_stats:
        ax.legend(fontsize=7, loc="upper left")
    return ax


def _quantile_bin_strata(values: np.ndarray, n_bins: int = 4,
                         label: str = "lead_time") -> list[tuple[str, str, np.ndarray]]:
    """Quantile-binned strata over a numeric column.

    Bin edges are quantiles over the finite values, de-duplicated with
    ``np.unique``. Degenerate columns (constant or near-constant, fewer than 2
    unique edges) yield no strata — the caller simply omits that dimension. The
    final bin is right-inclusive so the maximum value lands in the last bin.
    Returned masks are over the full-length array (rows with non-finite values
    fall in no bin).
    """
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    if not finite.any():
        return []
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(arr[finite], qs))
    if edges.size < 2:
        return []
    strata: list[tuple[str, str, np.ndarray]] = []
    for k in range(edges.size - 1):
        lo, hi = edges[k], edges[k + 1]
        if k == edges.size - 2:
            mask = finite & (arr >= lo) & (arr <= hi)
        else:
            mask = finite & (arr >= lo) & (arr < hi)
        group = f"[{lo:.3g}, {hi:.3g}{']' if k == edges.size - 2 else ')'}"
        strata.append((label, group, mask))
    return strata


def _benchmark_strata(records: dict[str, np.ndarray],
                      benchmark: str) -> list[tuple[str, str, np.ndarray]]:
    """Strata (dim, group, full-length mask) for one benchmark's records.

    Always emits ``("overall", "all", ones)`` as an all-rows mask (NOT a
    stratum-finite mask) so a row with a valid error but a missing ``t_bar`` or
    ``x_I`` is never dropped from the overall distribution; metric finiteness is
    applied later inside ``_tail_stats``. Benchmark-specific categorical strata
    are only added when their column is present and non-empty (forcing does not
    require ``regime``; source does not require ``temporal_family``; etc.).
    Numeric strata (lead time, interface position) use ``_quantile_bin_strata``.
    """
    n = int(records.get("_n", 0))
    strata: list[tuple[str, str, np.ndarray]] = [
        ("overall", "all", np.ones(n, dtype=bool)),
    ]

    def _has_str(col: str) -> bool:
        return col in records and any(str(v) != "" for v in records[col])

    if benchmark == "forcing":
        if _has_str("temporal_family"):
            for fam in TEMPORAL_ORDER:
                m = _mask_in(records["temporal_family"], fam)
                if m.any():
                    strata.append(("temporal_family", fam, m))
        if _has_str("spatial_family"):
            for fam in SPATIAL_ORDER:
                m = _mask_in(records["spatial_family"], fam)
                if m.any():
                    strata.append(("spatial_family", fam, m))
    elif benchmark == "source":
        if _has_str("regime"):
            for regime in REGIME_ORDER:
                m = _mask_in(records["regime"], regime)
                if m.any():
                    strata.append(("regime", regime, m))
    elif benchmark == "interfaces":
        if "x_I" in records:
            strata.extend(_quantile_bin_strata(records["x_I"], label="x_I"))

    if "t_bar" in records:
        strata.extend(_quantile_bin_strata(records["t_bar"], label="lead_time"))

    return strata


def _slice_records(records: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    """Slice every same-length column of ``records`` down to ``mask`` rows."""
    n = int(records.get("_n", 0))
    out: dict[str, np.ndarray] = {}
    for key, arr in records.items():
        if key == "_n":
            continue
        a = np.asarray(arr)
        if a.shape[:1] == (n,):
            out[key] = a[mask]
        else:
            out[key] = a
    out["_n"] = np.int64(int(mask.sum()))
    return out


def compute_tail_stats(records: dict[str, np.ndarray],
                       benchmark: str | None = None) -> list[dict]:
    """Per-(benchmark, metric, stratum, group) tail stats from test records.

    Each row is ``{benchmark, metric, stratum, group, n, mean, median, p90,
    p99, max}`` for both ``TAIL_METRICS`` across every stratum from
    ``_benchmark_strata``. Metric finiteness is enforced per stat inside
    ``_tail_stats``, so the ``overall/all`` mask stays an all-rows mask.

    Mixed-benchmark handling: if ``benchmark`` is given it is authoritative; if
    ``None`` and the records carry exactly one benchmark, that value is used; if
    ``None`` and several benchmarks are present, the records are first sliced to
    each benchmark's rows (a benchmark-specific dict) and strata are computed on
    the sliced dict — masks built on the full mixed array are never reused.
    Small-``n`` groups are kept; ``n`` is reported in every row so p99/max
    reliability is judgeable.
    """
    if benchmark is None and "benchmark" in records and int(records.get("_n", 0)) > 0:
        uniq = sorted({str(v) for v in records["benchmark"]})
        if len(uniq) == 1:
            benchmark = uniq[0]
        elif len(uniq) > 1:
            rows: list[dict] = []
            for bm in uniq:
                sub = _slice_records(records, _mask_in(records["benchmark"], bm))
                rows.extend(compute_tail_stats(sub, benchmark=bm))
            return rows

    bm_name = "" if benchmark is None else str(benchmark)
    rows: list[dict] = []
    for metric_name, col in TAIL_METRICS.items():
        if col not in records:
            continue
        values = records[col]
        for dim, group, mask in _benchmark_strata(records, bm_name):
            stats = _tail_stats(values[mask])
            row = {"benchmark": bm_name, "metric": metric_name,
                   "stratum": dim, "group": group}
            row.update(stats)
            rows.append(row)
    return rows


def write_tail_summary(records_by_benchmark: dict[str, dict[str, np.ndarray]],
                       out_path: str | Path) -> Path:
    """Write a combined ``tail_summary.csv`` over one or more benchmarks.

    For each ``benchmark -> records`` pair the dict key is authoritative and is
    passed explicitly to ``compute_tail_stats`` so a missing or stale
    ``benchmark`` column in the records never matters. Rows from every benchmark
    are concatenated into a single CSV.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    for benchmark, records in records_by_benchmark.items():
        all_rows.extend(compute_tail_stats(records, benchmark=benchmark))

    def _fmt(key: str, value) -> str:
        if key == "n":
            return str(int(value))
        if key in ("benchmark", "metric", "stratum", "group"):
            return str(value)
        return "" if value is None or (isinstance(value, float) and np.isnan(value)) else f"{float(value):.6g}"

    with out_path.open("w", newline="") as f:
        writer = _csv.writer(f)
        writer.writerow(_TAIL_SUMMARY_FIELDS)
        for row in all_rows:
            writer.writerow([_fmt(k, row.get(k)) for k in _TAIL_SUMMARY_FIELDS])
    return out_path


def plot_benchmark_tail_errors(records: dict[str, np.ndarray], save_path=None):
    """Per-benchmark tail-error figure: rows = metrics, columns = stratum dims.

    The benchmark is auto-detected from the records (single unique value). Each
    panel shows p90/p99 as grouped bars (C0/C3) with max as a black-diamond
    marker overlaid per group, so an outlier max never crushes the p90/p99 bars.
    A log y-scale is used only when every plotted tail value in the panel is
    strictly positive (and the max/p90 spread is large); any zero keeps the
    panel linear, and no epsilon is ever added to the stats themselves. The CSV
    written by ``write_tail_summary`` remains the authoritative numeric result.
    """
    benchmark = ""
    if "benchmark" in records and int(records.get("_n", 0)) > 0:
        uniq = sorted({str(v) for v in records["benchmark"]})
        benchmark = uniq[0] if len(uniq) == 1 else (uniq[0] if uniq else "")

    strata = _benchmark_strata(records, benchmark)
    dims: list[str] = []
    for dim, _group, _mask in strata:
        if dim not in dims:
            dims.append(dim)
    metric_names = list(TAIL_METRICS.keys())

    n_rows = len(metric_names)
    n_cols = max(1, len(dims))
    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.8 * n_rows),
                                 squeeze=False)
        for r, metric_name in enumerate(metric_names):
            col = TAIL_METRICS[metric_name]
            values = records.get(col)
            for c, dim in enumerate(dims):
                ax = axes[r][c]
                groups = [(g, m) for (d, g, m) in strata if d == dim]
                labels, p90s, p99s, maxes = [], [], [], []
                for g, m in groups:
                    stats = _tail_stats(values[m]) if values is not None else _tail_stats(np.array([]))
                    labels.append(g)
                    p90s.append(stats["p90"])
                    p99s.append(stats["p99"])
                    maxes.append(stats["max"])
                x = np.arange(len(labels))
                ax.bar(x - 0.2, p90s, width=0.4, label="p90", color="C0")
                ax.bar(x + 0.2, p99s, width=0.4, label="p99", color="C3")
                ax.plot(x, maxes, "D", color="k", ms=5, label="max")
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=7)
                if r == 0:
                    ax.set_title(dim)
                if c == 0:
                    ax.set_ylabel(f"{metric_name}\nrel-L2 (%)")
                plotted = np.array(p90s + p99s + maxes, dtype=np.float64)
                finite = plotted[np.isfinite(plotted)]
                if finite.size and np.all(finite > 0):
                    pos_p90 = np.array([p for p in p90s if np.isfinite(p) and p > 0], dtype=np.float64)
                    if pos_p90.size and float(np.max(finite)) / float(np.min(pos_p90)) > 50.0:
                        ax.set_yscale("log")
                if r == 0 and c == n_cols - 1:
                    ax.legend(fontsize=7)
        fig.suptitle(f"{benchmark or 'benchmark'}: tail errors (p90/p99/max) by stratum")
        return _save_figure(fig, save_path, "paper", f"{benchmark or 'benchmark'}_tail_errors")


def plot_all_benchmarks_tail_errors(records_by_benchmark, save_path=None):
    """Cross-benchmark *rare-failure* reliability (1x3 metric grid).

    Each panel is one scale-normalized, cross-benchmark-comparable metric with
    benchmarks along the x-axis: bars are **p99** (rare-but-recurring failure),
    an overlaid hollow marker is **max** (drawn open because it is n-dependent
    and not a headline statistic). The dimensionless **tail amplification**
    ``p99 / median`` is written above each benchmark as a text label (e.g.
    ``×2.7``); the finite sample count ``n`` is annotated so max reliability is
    judgeable. This figure deliberately owns no median/p90 — those typical and
    ordinary-high statistics live in ``plot_all_benchmarks_error_summary`` and
    the two figures must not share statistics.
    """
    names = list(records_by_benchmark.keys())
    panels = (
        ("gnrmse_pct", "gNRMSE (%)", "Field tail (scale-normalized)"),
        ("iface_rel_l2_pct", "interface rel-L2 (%)", "Interface-region tail"),
        ("node_jump_gnrmse_pct", "node-jump gNRMSE (%)", "Interface-jump tail"),
    )
    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
        for ax, (col, ylabel, title) in zip(np.atleast_1d(axes), panels):
            stats_by_name = {}
            for n in names:
                stats = _tail_stats(records_by_benchmark[n].get(col, np.array([np.nan])))
                stats["tail_amp"] = _tail_amp(stats)
                stats_by_name[n] = stats
            _grouped_tail_bars(
                ax, names, stats_by_name,
                bar_stat="p99", marker_stats=("max",), hollow_stats=("max",),
                ylabel=ylabel, title=title,
                annotate_n=True, annotate_tail_amp=True,
            )
        fig.suptitle("Cross-benchmark tail reliability — p99 (bar) + max (hollow) + tail amp (×p99/median)")
        return _save_figure(fig, save_path, "paper", "all_benchmarks_tail_errors")


def plot_benchmark_overview(save_path=None):
    """Benchmark overview table figure (what each task tests).

    A pure ``matplotlib`` table built from the curated ``_BENCHMARK_OVERVIEW_ROWS``
    constant — it reads no model or CSV, so it always renders. Rows are the four
    paper benchmarks; columns are ``_BENCHMARK_OVERVIEW_COLS``. Cells are
    ``textwrap``-wrapped and row heights scale with the wrapped line count so no
    text overflows. Each benchmark's row header uses its shared palette color.
    """
    benchmarks = list(_BENCHMARK_OVERVIEW_ROWS.keys())
    col_labels = ["benchmark", *_BENCHMARK_OVERVIEW_COLS]
    wrap_widths = {
        "benchmark": 12,
        "geometry / source": 30,
        "varied parameters": 28,
        "ITR parameterization": 26,
        "prediction task": 24,
        "main stress test": 24,
    }

    cell_text: list[list[str]] = []
    row_line_counts: list[int] = []
    for bm in benchmarks:
        row = [bm]
        for col in _BENCHMARK_OVERVIEW_COLS:
            row.append(_BENCHMARK_OVERVIEW_ROWS[bm][col])
        wrapped = [
            "\n".join(textwrap.wrap(text, width=wrap_widths[label]) or [""])
            for label, text in zip(col_labels, row)
        ]
        cell_text.append(wrapped)
        row_line_counts.append(max(c.count("\n") + 1 for c in wrapped))

    total_lines = sum(row_line_counts) + len(col_labels)  # + header band
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(15, 0.6 + 0.34 * total_lines))
        ax.axis("off")
        col_widths = [0.11, 0.24, 0.21, 0.17, 0.14, 0.13]
        table = ax.table(
            cellText=cell_text,
            colLabels=col_labels,
            colWidths=col_widths,
            cellLoc="left",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)

        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor("0.8")
            cell.set_linewidth(0.6)
            if r == 0:  # header band
                cell.set_facecolor("0.92")
                cell.set_text_props(fontweight="bold")
            else:
                line_count = row_line_counts[r - 1]
                cell.set_height(0.04 * (line_count + 0.8))
                if c == 0:
                    bm = benchmarks[r - 1]
                    cell.set_facecolor(_BENCHMARK_COLORS.get(bm, "0.95"))
                    cell.set_text_props(color="white", fontweight="bold")
                elif r % 2 == 0:
                    cell.set_facecolor("0.975")
        ax.set_title("Benchmark overview — what each task tests", fontweight="bold", pad=12)
        return _save_figure(fig, save_path, "paper", "benchmark_overview", layout="none")


# ============================================================
# GENERALIZATION (same-sim held-out vs unseen-sim)
# ============================================================

def _assert_single_benchmark(records: dict[str, np.ndarray], benchmark: str) -> None:
    """Raise unless every stored ``benchmark`` value equals ``benchmark``.

    The ``benchmark`` kwarg is authoritative and never inferred from a path. If
    the records carry a ``benchmark`` column, a record set spanning more than one
    benchmark, or one whose stored value disagrees with the passed kwarg, is a
    mislabeled/swapped file and raises. Records without the column are accepted
    (the kwarg stands alone).
    """
    if "benchmark" not in records:
        return
    uniq = sorted({str(v) for v in records["benchmark"] if str(v) != ""})
    if not uniq:
        return
    if len(uniq) > 1:
        raise ValueError(
            f"generalization records span multiple benchmarks {uniq}; "
            f"pass a single-benchmark record set matching benchmark='{benchmark}'"
        )
    if uniq[0] != str(benchmark):
        raise ValueError(
            f"stored benchmark '{uniq[0]}' does not match requested "
            f"benchmark='{benchmark}'"
        )


def _compute_shared_lead_bin_edges(same_sim_records: dict[str, np.ndarray],
                                   unseen_records: dict[str, np.ndarray],
                                   *, n_lead_bins: int = 4) -> dict:
    """Quantile lead-time (``t_bar``) bin edges shared by every metric panel.

    Edges are quantiles of finite ``t_bar`` over the **union** of both record
    sets, computed independently of any metric so both panels and both series
    share identical physical lead-time edges. Tied quantiles are de-duplicated
    (``np.unique``); when ties collapse bins the effective bin count drops and is
    reported in ``n_effective_bins``. Raises only when a set has zero finite
    ``t_bar`` (empty input) or the union has fewer than two distinct ``t_bar``
    values (no meaningful lead-time axis).
    """
    same_t = _finite(same_sim_records.get("t_bar", np.array([])))
    unseen_t = _finite(unseen_records.get("t_bar", np.array([])))
    if same_t.size == 0 or unseen_t.size == 0:
        raise ValueError(
            "generalization needs finite t_bar in both record sets; got "
            f"same_sim={same_t.size}, unseen={unseen_t.size}"
        )
    union_t = np.concatenate([same_t, unseen_t])
    if np.unique(union_t).size < 2:
        raise ValueError(
            "fewer than two distinct t_bar values across the union; no lead-time "
            "axis to bin"
        )
    qs = np.linspace(0.0, 1.0, n_lead_bins + 1)
    edges = np.unique(np.quantile(union_t, qs))
    return {"lead_bin_edges": edges, "n_effective_bins": int(edges.size - 1)}


def _lead_bin_masks(t_bar: np.ndarray, edges: np.ndarray) -> list[np.ndarray]:
    """Full-length masks for each lead-time bin (final bin right-inclusive)."""
    arr = np.asarray(t_bar, dtype=np.float64)
    finite = np.isfinite(arr)
    masks: list[np.ndarray] = []
    for k in range(edges.size - 1):
        lo, hi = edges[k], edges[k + 1]
        if k == edges.size - 2:
            masks.append(finite & (arr >= lo) & (arr <= hi))
        else:
            masks.append(finite & (arr >= lo) & (arr < hi))
    return masks


def _compute_generalization_by_lead_bin(same_sim_records: dict[str, np.ndarray],
                                        unseen_records: dict[str, np.ndarray],
                                        *, benchmark: str, metric: str,
                                        lead_bin_edges: np.ndarray) -> dict:
    """Per-lead-bin median + IQR of ``metric`` for both record sets.

    ``lead_bin_edges`` are the shared edges from ``_compute_shared_lead_bin_edges``
    so every metric panel uses identical physical lead-time bins; only the per-bin
    counts differ between metrics/groups. Both record sets are asserted to carry
    the passed ``benchmark`` (never inferred). Empty bins yield count 0 and NaN
    statistics rather than raising. Precondition (enforced upstream, not here):
    ``same_sim`` is held-out time pairs of seen sims; ``unseen`` is disjoint sim
    identities.
    """
    _assert_single_benchmark(same_sim_records, benchmark)
    _assert_single_benchmark(unseen_records, benchmark)

    def _series(records: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        t_bar = records.get("t_bar", np.array([]))
        n = int(np.asarray(t_bar).shape[0])
        vals = np.asarray(records.get(metric, np.full(n, np.nan)), dtype=np.float64)
        if vals.shape[0] != n:
            vals = np.full(n, np.nan)
        masks = _lead_bin_masks(t_bar, lead_bin_edges)
        counts, median, q25, q75 = [], [], [], []
        for m in masks:
            v = vals[m]
            v = v[np.isfinite(v)]
            counts.append(int(v.size))
            if v.size:
                median.append(float(np.median(v)))
                q25.append(float(np.quantile(v, 0.25)))
                q75.append(float(np.quantile(v, 0.75)))
            else:
                median.append(np.nan)
                q25.append(np.nan)
                q75.append(np.nan)
        return {
            "counts": np.array(counts, dtype=np.int64),
            "median": np.array(median, dtype=np.float64),
            "q25": np.array(q25, dtype=np.float64),
            "q75": np.array(q75, dtype=np.float64),
        }

    same = _series(same_sim_records)
    unseen = _series(unseen_records)
    return {
        "same_sim_counts": same["counts"],
        "unseen_counts": unseen["counts"],
        "same_sim_median": same["median"],
        "same_sim_q25": same["q25"],
        "same_sim_q75": same["q75"],
        "unseen_median": unseen["median"],
        "unseen_q25": unseen["q25"],
        "unseen_q75": unseen["q75"],
    }


def plot_generalization_same_vs_unseen(same_sim_records, unseen_records, *,
                                       benchmark,
                                       metrics=("rmse_K", "node_jump_gnrmse_pct"),
                                       n_lead_bins=4, save_path=None):
    """Same-sim held-out vs unseen-sim generalization gap vs lead time.

    One panel per metric, lead-time bins on the x-axis, same-sim and unseen as two
    distinctly-colored series (median line + IQR band). A single lead-time binning
    is computed once over the union and reused for every panel so the physical
    edges are identical across metrics. ``benchmark`` is required and validated
    against each record set's stored ``benchmark`` column. When both sets carry
    ``sim_id``, an accidental swap is caught by a disjointness guard. Returns the
    saved figure ``Path`` (or ``None`` if skipped), matching the package's plot
    API; per-bin metadata is exposed via the private compute helpers, not here.
    """
    if "sim_id" in same_sim_records and "sim_id" in unseen_records:
        same_ids = {int(v) for v in same_sim_records["sim_id"]}
        unseen_ids = {int(v) for v in unseen_records["sim_id"]}
        if same_ids and unseen_ids and not same_ids.isdisjoint(unseen_ids):
            raise ValueError(
                "same-sim and unseen-sim record sets share sim_ids "
                f"{sorted(same_ids & unseen_ids)[:5]}...; the unseen set must hold "
                "disjoint simulation identities"
            )

    edges_info = _compute_shared_lead_bin_edges(
        same_sim_records, unseen_records, n_lead_bins=n_lead_bins
    )
    edges = edges_info["lead_bin_edges"]
    centers = 0.5 * (edges[:-1] + edges[1:])

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(1, len(metrics), figsize=(6.0 * len(metrics), 4.6),
                                 squeeze=False)
        first_stats = None
        for ax, metric in zip(axes[0], metrics):
            stats = _compute_generalization_by_lead_bin(
                same_sim_records, unseen_records,
                benchmark=benchmark, metric=metric, lead_bin_edges=edges,
            )
            if first_stats is None:
                first_stats = stats
            for label, color, key in (
                ("same-sim (held-out time)", _SAME_SIM_COLOR, "same_sim"),
                ("unseen-sim", _UNSEEN_COLOR, "unseen"),
            ):
                med = stats[f"{key}_median"]
                q25 = stats[f"{key}_q25"]
                q75 = stats[f"{key}_q75"]
                ax.fill_between(centers, q25, q75, color=color, alpha=0.18, linewidth=0)
                ax.plot(centers, med, marker="o", ms=5, color=color, label=label)
            ax.set_xlabel("lead time t̄ (bin center)")
            ax.set_ylabel(metric)
            ax.set_title(metric)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.legend(fontsize=7, loc="upper left")

        edge_str = ", ".join(f"{e:.3g}" for e in edges)
        same_counts = "/".join(str(int(c)) for c in first_stats["same_sim_counts"])
        unseen_counts = "/".join(str(int(c)) for c in first_stats["unseen_counts"])
        fig.suptitle(
            f"{benchmark}: same-sim vs unseen-sim by lead time "
            f"(median + IQR)\nt̄ edges [{edge_str}]  ·  "
            f"n same={same_counts}  n unseen={unseen_counts}",
            fontsize=9,
        )
        return _save_figure(fig, save_path, "paper", "generalization_same_vs_unseen")
