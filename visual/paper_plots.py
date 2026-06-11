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
    k_left, k_right = _resolve_layer_conductivities(ds, int(sid), config)

    left_node, right_node = _interface_flanking_nodes_from_grid(ds.x_grid, interface_x)
    h_L = float(interface_x) - float(ds.x_grid[left_node])
    h_R = float(ds.x_grid[right_node]) - float(interface_x)
    G = _interface_conductance_G(ds.x_grid, interface_x, R_c, k_left, k_right)

    true_jump = _interface_contact_jump_map(case["Y_true"], ds.x_grid, interface_x, R_c, k_left, k_right)
    pred_jump = _interface_contact_jump_map(case["Y_pred"], ds.x_grid, interface_x, R_c, k_left, k_right)
    node_true = _interface_jump_map(case["Y_true"], left_node, right_node)
    node_peak = float(np.max(np.abs(node_true))) if node_true.size else 0.0
    contact_peak = float(np.max(np.abs(true_jump))) if true_jump.size else 0.0

    print(
        f"[jump] sid={int(sid)} s={int(s)} x_I={interface_x:.4f} R_c={R_c:.4f} "
        f"k_L={k_left:.3f} k_R={k_right:.3f} h_L={h_L:.4f} h_R={h_R:.4f} "
        f"G={G:.4f} R_c*G={R_c * G:.4f} max|node|={node_peak:.4g} max|contact|={contact_peak:.4g}"
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
        R_c = float(case["R_c"])
        k_left, k_right = _resolve_layer_conductivities(ds, int(sid), config)
        tj = _interface_contact_jump_map(case["Y_true"], ds.x_grid, interface_x, R_c, k_left, k_right)
        pj = _interface_contact_jump_map(case["Y_pred"], ds.x_grid, interface_x, R_c, k_left, k_right)
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
    names = ("forcing", "source", "interfaces")
    ctxs = (forcing_ctx, source_ctx, interfaces_ctx)
    colors = {"forcing": "C0", "source": "C1", "interfaces": "C2"}
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
