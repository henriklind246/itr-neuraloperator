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
    _model_predict_item,
    _plot_field_2d,
)


TEMPORAL_ORDER = ("sin", "exp", "pulse_train", "exp_train")
SPATIAL_ORDER = ("uniform", "patch", "gaussian", "triangle")
REGIME_ORDER = ("left", "near", "right")

FORCING_ACTIVE_T_MAX = 0.20  # target time t_j below which boundary forcing is active

_INT_COLS = ("sim_id", "s", "j")
_FLOAT_COLS = (
    "t_s", "t_bar", "R_c", "x_h", "y_h", "A", "freq",
    "x_I", "rel_l2_pct", "iface_rel_l2_pct",
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
# CROSS-BENCHMARK SYNTHESIS
# ============================================================

def plot_all_benchmarks_error_summary(records_by_benchmark, save_path=None):
    """Cross-benchmark error spread (1x3): global, interface-band, tail."""
    names = list(records_by_benchmark.keys())
    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

        global_data = [_finite(records_by_benchmark[n]["rel_l2_pct"]) for n in names]
        iface_data = [_finite(records_by_benchmark[n]["iface_rel_l2_pct"]) for n in names]
        gd = [(d if d.size else np.array([np.nan])) for d in global_data]
        idd = [(d if d.size else np.array([np.nan])) for d in iface_data]
        axes[0].boxplot(gd, tick_labels=names, showfliers=False)
        axes[0].set_ylabel("rel-L2 (%)")
        axes[0].set_title("Global error by benchmark")
        axes[1].boxplot(idd, tick_labels=names, showfliers=False)
        axes[1].set_ylabel("interface rel-L2 (%)")
        axes[1].set_title("Interface-band error by benchmark")

        x = np.arange(len(names))
        p90 = [float(np.percentile(d, 90)) if d.size else np.nan for d in global_data]
        p99 = [float(np.percentile(d, 99)) if d.size else np.nan for d in global_data]
        axes[2].bar(x - 0.2, p90, width=0.4, label="p90", color="C0")
        axes[2].bar(x + 0.2, p99, width=0.4, label="p99", color="C3")
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(names)
        axes[2].set_ylabel("rel-L2 (%)")
        axes[2].set_title("Tail error by benchmark")
        axes[2].legend()
        for ax in axes:
            ax.tick_params(axis="x", rotation=20)
        return _save_figure(fig, save_path, "paper", "all_benchmarks_error_summary")


def plot_all_benchmarks_prediction_truth_residual(forcing_ctx, source_ctx,
                                                  interfaces_ctx, save_path=None):
    """One representative case per benchmark (3x3).

    Each ``*_ctx`` is ``(model, ds, records)``. Row 1 is the forcing hard case
    (max global error), row 2 the source near-interface case, row 3 the
    interfaces off-center case (max ``|x_I − 0.5|``).
    """
    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(3, 3, figsize=(12.5, 11.5))

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
        return _save_figure(fig, save_path, "paper", "all_benchmarks_prediction_truth_residual")
