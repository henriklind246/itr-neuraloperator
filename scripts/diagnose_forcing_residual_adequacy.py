"""Residual-adequacy vs optimization for the physics-only forcing PINO plateau.

Phase 3 of the forcing-PINO diagnostic suite. Phase 1 localizes *where* the ~5%
gnrmse lives (error surface + image fidelity + pathway probe); Phase 2 tests the
forcing representation's linearity/superposition. This script asks the remaining
question for the sims that are *actually worst*: is the loss the model minimizes
even satisfied there, and if not, *which* loss term is left large?

For the worst-K sims (ranked by gnrmse, reusing the Phase-1 eval) at a few
snapshots, it evaluates every training loss-term residual with the SAME autodiff
operators the trainer uses (``src/physics/pde_residual.py``):

  - interior PDE            ``diffusion_residual``      (r = T_t - (T_xx+T_yy))
  - left forcing BC (x=0)   ``forcing_neumann_residual`` (T_x + q_L/(k*sigma))
  - top/bottom adiabatic    ``neumann_residual``        (dT/dy at y=1 / y=0)
  - right Dirichlet         T(x=1) - T_RIGHT            (skipped when hard-enforced)
  - initial condition       ``ic_residual``             (T(t=0) - IC)

All residuals are in NORMALIZED (T_tilde) units, exactly as the loss sees them;
the physical-T residual is this times ``sigma``. Because the inverse heat
operator spreads a local residual globally, the primary reading is AGGREGATE and
CALIBRATED, not pointwise: each term's residual RMS is compared to the SAME
term's RMS on training-like collocation (``sample_collocation``), so "small"
means small relative to what training actually drove down, not relative to zero.

Interpretation (see ``--help`` and the summary JSON's guardrails field):
  - low interior residual + large left-BC residual (>> its calibration)
      => the boundary constraint was not enforced (needs BC upweight / GradNorm /
         more BC collocation), NOT underdetermination.
  - residuals co-located with the error AND comparable to the training-large
      residual => optimization / capacity limit, not a missing constraint.
Pointwise error<->residual correlation is reported only as a caveated secondary.

Metric: ``gnrmse = rmse_K / sigma_global`` on ``T - T_RIGHT`` (matches
``val_gnrmse`` in ``train_metrics.csv``). CPU-only; small defaults (worst-K=6 x
2 snapshots on a strided eval grid) keep it Mac-cheap. AD (the 2nd-derivative
interior term) is the cost driver, so it is chunked and run one sim at a time.

Example (smoke; numbers not scientifically meaningful):
  .venv/bin/python scripts/diagnose_forcing_residual_adequacy.py \\
    --checkpoint runs/pino_smoke/config0/seed42 \\
    --data-dir data/diffusion_forcing_probe_cvit_best \\
    --num-sims 8 --worst-k 3 --snapshots 2 --out-dir /tmp/diag_res
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from problems.diffusion import T_RIGHT  # noqa: E402
from scripts._forcing_diag_common import (  # noqa: E402
    load_model_and_data,
    load_sim_params,
    predict_grid,
    select_ids,
)
from src.operators.train_pino import (  # noqa: E402
    _ic_targets,
    build_forcing_image,
    left_wall_qL,
    sample_collocation,
)
from src.operators.utils import resolve_device  # noqa: E402
from src.physics.pde_residual import (  # noqa: E402
    diffusion_residual,
    forcing_neumann_residual,
    ic_residual,
    neumann_residual,
)

STATS_FIELDS = [
    "sim_id", "rank", "temporal_family", "spatial_family",
    "t", "lead_frac", "gnrmse",
    "err_rms_K", "err_rms_left_K",
    "interior_res_rms", "interior_res_calib_ratio",
    "left_bc_res_rms", "left_bc_calib_ratio",
    "top_res_rms", "bottom_res_rms",
    "right_dir_res_rms", "right_hard_enforced",
    "ic_res_rms", "ic_calib_ratio",
    "corr_interior_res_err", "corr_caveat",
]

CORR_CAVEAT = (
    "secondary: the inverse heat operator spreads a local residual globally, so "
    "low |residual|<->|error| correlation does NOT by itself imply a deficient "
    "physics loss; read the aggregate calibrated RMS instead."
)

GUARDRAILS = (
    "Residuals are normalized (T_tilde) units, as the loss sees them (physical-T "
    "residual = value * sigma). Calibration ratios are each term's worst-K RMS "
    "over the SAME term's training-like collocation RMS. low interior + large "
    "left-BC ratio => boundary constraint unmet (upweight BC / GradNorm / more BC "
    "samples), not underdetermination. residual co-located and ~training-large => "
    "optimization/capacity. Pointwise correlation is a caveated secondary only."
)


# --------------------------------------------------------------------------- #
# Small autodiff evaluators (one sim, B=1) — mirror the trainer's residuals.  #
# --------------------------------------------------------------------------- #

def _rms(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    if a.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(a ** 2)))


def _leaf(vals: np.ndarray, device: torch.device, grad: bool) -> torch.Tensor:
    t = torch.as_tensor(np.asarray(vals, dtype=np.float32), device=device).reshape(1, -1, 1)
    return t.requires_grad_(grad)


def _interior_res_field(
    model: torch.nn.Module,
    u: torch.Tensor,
    xy: np.ndarray,
    t_val: float,
    device: torch.device,
    chunk: int,
) -> np.ndarray:
    """|interior residual| at every (x,y) in ``xy`` at fixed time; (N,) float64.

    Chunked so each chunk builds and frees its own 2nd-order graph — the peak
    memory is set by ``chunk``, not by the full eval grid.
    """
    out = np.empty(xy.shape[0], dtype=np.float64)
    for s in range(0, xy.shape[0], chunk):
        e = min(s + chunk, xy.shape[0])
        x = _leaf(xy[s:e, 0], device, grad=True)
        y = _leaf(xy[s:e, 1], device, grad=True)
        t = _leaf(np.full(e - s, t_val), device, grad=True)
        r = diffusion_residual(model, u, x, y, t)
        out[s:e] = r.detach().reshape(-1).cpu().numpy()
    return out


def _left_bc_res_profile(
    model: torch.nn.Module,
    u: torch.Tensor,
    params1: list[dict[str, Any]],
    y_pts: np.ndarray,
    t_val: float,
    sigma: float,
    t_ramp: float,
    device: torch.device,
    chunk: int,
) -> np.ndarray:
    """Left-wall forcing residual ``T_x + q_L/(k*sigma)`` along y at x=0; (Ny,)."""
    out = np.empty(y_pts.shape[0], dtype=np.float64)
    for s in range(0, y_pts.shape[0], chunk):
        e = min(s + chunk, y_pts.shape[0])
        x = torch.zeros((1, e - s, 1), device=device, requires_grad=True)
        y = _leaf(y_pts[s:e], device, grad=True)
        t = _leaf(np.full(e - s, t_val), device, grad=True)
        q_L = left_wall_qL(params1, y, t, device, t_ramp)
        r = forcing_neumann_residual(model, u, x, y, t, q_L, sigma, k=1.0)
        out[s:e] = r.detach().reshape(-1).cpu().numpy()
    return out


def _hom_wall_res_profile(
    model: torch.nn.Module,
    u: torch.Tensor,
    free_pts: np.ndarray,
    t_val: float,
    wall: str,
    device: torch.device,
    chunk: int,
) -> np.ndarray:
    """Adiabatic wall residual (dT/dy) along the free coordinate; (N,).

    ``wall`` in {"top" (y=1), "bottom" (y=0)}; ``free_pts`` are the x samples.
    """
    y_fixed = 1.0 if wall == "top" else 0.0
    out = np.empty(free_pts.shape[0], dtype=np.float64)
    for s in range(0, free_pts.shape[0], chunk):
        e = min(s + chunk, free_pts.shape[0])
        x = _leaf(free_pts[s:e], device, grad=True)
        y = torch.full((1, e - s, 1), y_fixed, device=device, requires_grad=True)
        t = _leaf(np.full(e - s, t_val), device, grad=True)
        r = neumann_residual(model, u, x, y, t, wall)
        out[s:e] = r.detach().reshape(-1).cpu().numpy()
    return out


def _right_dirichlet_res(
    model: torch.nn.Module,
    u: torch.Tensor,
    y_pts: np.ndarray,
    t_val: float,
    t_right_tilde: float,
    device: torch.device,
) -> np.ndarray:
    """Right-wall Dirichlet error ``T_tilde(x=1) - T_RIGHT_tilde`` along y; (Ny,).

    No autograd (a pointwise value error), so ``no_grad`` keeps it cheap.
    """
    x = torch.ones((1, y_pts.shape[0], 1), device=device)
    y = torch.as_tensor(y_pts.astype(np.float32), device=device).reshape(1, -1, 1)
    t = torch.full((1, y_pts.shape[0], 1), float(t_val), device=device)
    coords = torch.cat([x, y], dim=-1)
    with torch.no_grad():
        T = model(u, coords, t)
    return (T.reshape(-1).cpu().numpy() - t_right_tilde).astype(np.float64)


def _ic_res_field(
    model: torch.nn.Module,
    u: torch.Tensor,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    ic_field_K: np.ndarray,
    mu: float,
    sigma: float,
    device: torch.device,
) -> np.ndarray:
    """IC residual ``T_tilde(t=0) - IC_tilde`` over the full grid; (Nx*Ny,)."""
    gx, gy = np.meshgrid(x_grid, y_grid, indexing="ij")
    coords = torch.as_tensor(
        np.stack([gx.reshape(-1), gy.reshape(-1)], axis=-1).astype(np.float32),
        device=device,
    ).unsqueeze(0)
    t0 = torch.zeros((1, coords.shape[1], 1), device=device)
    target = ((ic_field_K.reshape(-1) - mu) / (sigma + 1e-8)).astype(np.float32)
    target_t = torch.as_tensor(target, device=device).reshape(1, -1, 1)
    with torch.no_grad():
        r = ic_residual(model, u, coords, t0, target_t)
    return r.reshape(-1).cpu().numpy().astype(np.float64)


# --------------------------------------------------------------------------- #
# Training-like calibration (the "small relative to training" reference).      #
# --------------------------------------------------------------------------- #

def _calibration_residuals(
    model: torch.nn.Module,
    params: list[dict[str, Any]],
    ids: np.ndarray,
    data: dict[str, Any],
    cfg,
    device: torch.device,
    *,
    n_r: int,
    n_bc: int,
    seed: int,
) -> dict[str, float]:
    """Per-term residual RMS on training-like collocation over ``params``.

    Uses the trainer's ``sample_collocation`` (random space-time, dense-grid IC
    nodes) and the exact residual operators, so the returned RMS is the scale the
    optimizer actually drove each term to — the calibration denominator.
    """
    x_grid = torch.as_tensor(np.asarray(data["x_grid"], np.float32), device=device)
    y_grid = torch.as_tensor(np.asarray(data["y_grid"], np.float32), device=device)
    mu, sigma = float(cfg.mu), float(cfg.sigma)
    y_img = np.linspace(cfg.c_dom, cfg.d_dom, cfg.ny_img, dtype=np.float64)
    t_img = np.linspace(0.0, cfg.t_final, cfg.nt_img, dtype=np.float64)
    u = build_forcing_image(params, y_img, t_img, cfg.a_ref, device, cfg.t_ramp)

    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    batch = sample_collocation(
        n_r, 0, n_bc, float(cfg.t_final), x_grid, y_grid, device, gen, dense_ic=True,
    )

    x_r, y_r, t_r = batch["interior"]
    r_int = diffusion_residual(model, u, x_r, y_r, t_r)
    interior = _rms(r_int.detach().cpu().numpy())

    walls = batch["walls"]
    xw, yw, tw = walls["left"]
    q_L = left_wall_qL(params, yw, tw, device, cfg.t_ramp)
    r_left = forcing_neumann_residual(model, u, xw, yw, tw, q_L, sigma, k=1.0)
    left = _rms(r_left.detach().cpu().numpy())

    xt, yt, tt = walls["top"]
    top = _rms(neumann_residual(model, u, xt, yt, tt, "top").detach().cpu().numpy())
    xb, yb, tb = walls["bottom"]
    bottom = _rms(
        neumann_residual(model, u, xb, yb, tb, "bottom").detach().cpu().numpy()
    )

    # calibration IC targets must come from THESE sims' IC fields (calib ids),
    # aligned row-for-row with u; the dense-IC nodes are shared across sims.
    ix, iy = batch["ic"]["ix"], batch["ic"]["iy"]
    ic_tgt = _ic_targets(data["trajectories"], np.asarray(ids), ix, iy, mu, sigma, device)
    r_ic = ic_residual(model, u, batch["ic"]["coords"], batch["ic"]["t"], ic_tgt)
    ic = _rms(r_ic.detach().cpu().numpy())

    return {
        "interior_res_rms": interior,
        "left_bc_res_rms": left,
        "top_res_rms": top,
        "bottom_res_rms": bottom,
        "ic_res_rms": ic,
        "n_calib_sims": int(len(params)),
        "n_r": int(n_r),
        "n_bc": int(n_bc),
    }


# --------------------------------------------------------------------------- #
# Main driver.                                                                 #
# --------------------------------------------------------------------------- #

def _snapshot_indices(nt: int, snapshots: int, snapshot_times, t_grid) -> list[int]:
    """Snapshot grid indices to analyze (exclude t=0, the IC anchor)."""
    if snapshot_times:
        return [int(np.argmin(np.abs(t_grid - float(s)))) for s in snapshot_times]
    usable = max(nt - 1, 1)
    k = min(int(snapshots), usable)
    # evenly spread across 1..nt-1 (skip the IC snapshot 0).
    return sorted({int(round(v)) for v in np.linspace(1, nt - 1, k)})


def _ratio(x: float, ref: float) -> float:
    return float(x / ref) if ref > 1e-12 else float("nan")


def run_residual_adequacy(
    checkpoint: str,
    data_dir: str | None,
    *,
    split: str = "val",
    num_sims: int = 8,
    sim_ids: str | None = None,
    worst_k: int = 6,
    snapshots: int = 2,
    snapshot_times: str | None = None,
    eval_stride: int = 2,
    query_chunk: int = 512,
    calib_sims: int = 8,
    calib_n_r: int = 4096,
    calib_n_bc: int = 1024,
    query_batch: int = 8,
    out_dir: str | None = None,
    device: str = "auto",
    seed: int = 0,
) -> tuple[dict[str, Any], Path]:
    """Worst-K residual adequacy; writes CSV/JSON + per-sim panels.

    Returns ``(summary, out_dir)``.
    """
    t0 = time.perf_counter()
    dev = resolve_device(device) if isinstance(device, str) else device
    model, data, cfg = load_model_and_data(checkpoint, data_dir, dev)
    sim_params = load_sim_params(cfg.config)
    ids = select_ids(data, split=split, num_sims=num_sims, sim_ids=sim_ids)
    params = [dict(sim_params[int(i)]) for i in ids]

    x_grid = np.asarray(data["x_grid"], dtype=np.float64)
    y_grid = np.asarray(data["y_grid"], dtype=np.float64)
    t_grid = np.asarray(data["t_grid"], dtype=np.float64)
    nx, ny, nt = x_grid.shape[0], y_grid.shape[0], t_grid.shape[0]
    mu, sigma = float(cfg.mu), float(cfg.sigma)
    t_final = float(t_grid[-1]) if nt else float(cfg.t_final)
    t_right_tilde = (float(T_RIGHT) - mu) / (sigma + 1e-8)

    # ---- rank the pool by gnrmse (reuse the Phase-1 eval) --------------------
    t_pred = time.perf_counter()
    pred_K = predict_grid(model, params, data, cfg, t_grid, dev, query_batch)
    truth = np.asarray(data["trajectories"][np.asarray(ids)], dtype=np.float32)
    if truth.shape != pred_K.shape:
        raise ValueError(f"shape mismatch pred {pred_K.shape} vs truth {truth.shape}.")
    # per (sim, snapshot) rmse_K -> gnrmse; rank sims by mean gnrmse (skip t=0).
    err2 = (pred_K - truth).astype(np.float64) ** 2  # (B, nt, nx, ny)
    rmse_st = np.sqrt(err2.reshape(len(ids), nt, -1).mean(axis=2))  # (B, nt)
    gnrmse_st = rmse_st / (sigma + 1e-8)
    rank_score = gnrmse_st[:, 1:].mean(axis=1) if nt > 1 else gnrmse_st.mean(axis=1)
    order = np.argsort(-rank_score)  # worst first
    worst = order[: min(int(worst_k), len(ids))]
    pred_time = time.perf_counter() - t_pred

    # ---- training-like calibration ------------------------------------------
    t_cal = time.perf_counter()
    calib_ids = select_ids(
        data, split=split, num_sims=min(int(calib_sims), len(sim_params)), sim_ids=None
    )
    calib_params = [dict(sim_params[int(i)]) for i in calib_ids]
    calib = _calibration_residuals(
        model, calib_params, np.asarray(calib_ids), data, cfg, dev,
        n_r=calib_n_r, n_bc=calib_n_bc, seed=seed,
    )
    calib_time = time.perf_counter() - t_cal

    # ---- worst-K residual evaluation ----------------------------------------
    snap_times_arg = (
        [float(s) for s in snapshot_times.split(",") if s.strip()]
        if snapshot_times else None
    )
    snap_idx = _snapshot_indices(nt, snapshots, snap_times_arg, t_grid)

    xs = x_grid[::eval_stride]
    ys = y_grid[::eval_stride]
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    xy = np.stack([gx.reshape(-1), gy.reshape(-1)], axis=-1)
    left_mask = x_grid < (x_grid.min() + (x_grid.max() - x_grid.min()) / 3.0)

    y_img = np.linspace(cfg.c_dom, cfg.d_dom, cfg.ny_img, dtype=np.float64)
    t_img = np.linspace(0.0, cfg.t_final, cfg.nt_img, dtype=np.float64)

    out = Path(out_dir).expanduser() if out_dir else _checkpoint_out(checkpoint)
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    panels: list[Path] = []
    t_ad = time.perf_counter()
    for rank, b in enumerate(worst):
        p1 = [params[int(b)]]
        u = build_forcing_image(p1, y_img, t_img, cfg.a_ref, dev, cfg.t_ramp)
        ic_rms = _rms(
            _ic_res_field(model, u, x_grid, y_grid, truth[b, 0], mu, sigma, dev)
        )
        panel_fields: list[dict[str, Any]] = []
        for k in snap_idx:
            tv = float(t_grid[k])
            err_K = np.abs(pred_K[b, k] - truth[b, k]).astype(np.float64)  # (nx,ny)
            err_rms_K = float(np.sqrt((err_K ** 2).mean()))
            err_rms_left_K = float(np.sqrt((err_K[left_mask] ** 2).mean()))

            r_int = _interior_res_field(model, u, xy, tv, dev, query_chunk)
            interior_rms = _rms(r_int)
            r_int_field = np.abs(r_int).reshape(xs.shape[0], ys.shape[0])

            r_left = _left_bc_res_profile(
                model, u, p1, y_grid, tv, sigma, cfg.t_ramp, dev, query_chunk
            )
            left_rms = _rms(r_left)
            r_top = _hom_wall_res_profile(model, u, x_grid, tv, "top", dev, query_chunk)
            r_bot = _hom_wall_res_profile(
                model, u, x_grid, tv, "bottom", dev, query_chunk
            )
            if cfg.hard_right_dirichlet:
                right_rms = float("nan")
            else:
                right_rms = _rms(
                    _right_dirichlet_res(model, u, y_grid, tv, t_right_tilde, dev)
                )

            # secondary pointwise |residual|<->|error| correlation (strided).
            err_sub = err_K[::eval_stride, ::eval_stride].reshape(-1)
            ar = np.abs(r_int).reshape(-1)
            if err_sub.shape[0] == ar.shape[0] and np.std(err_sub) > 1e-12 and np.std(ar) > 1e-12:
                corr = float(np.corrcoef(ar, err_sub)[0, 1])
            else:
                corr = float("nan")

            rows.append({
                "sim_id": int(ids[b]),
                "rank": int(rank),
                "temporal_family": str(params[int(b)].get("temporal_family", "")),
                "spatial_family": str(params[int(b)].get("spatial_family", "")),
                "t": tv,
                "lead_frac": float(tv / t_final) if t_final > 0 else 0.0,
                "gnrmse": float(gnrmse_st[b, k]),
                "err_rms_K": err_rms_K,
                "err_rms_left_K": err_rms_left_K,
                "interior_res_rms": interior_rms,
                "interior_res_calib_ratio": _ratio(interior_rms, calib["interior_res_rms"]),
                "left_bc_res_rms": left_rms,
                "left_bc_calib_ratio": _ratio(left_rms, calib["left_bc_res_rms"]),
                "top_res_rms": _rms(r_top),
                "bottom_res_rms": _rms(r_bot),
                "right_dir_res_rms": right_rms,
                "right_hard_enforced": bool(cfg.hard_right_dirichlet),
                "ic_res_rms": ic_rms,
                "ic_calib_ratio": _ratio(ic_rms, calib["ic_res_rms"]),
                "corr_interior_res_err": corr,
                "corr_caveat": CORR_CAVEAT,
            })
            panel_fields.append({
                "t": tv, "err_K": err_K, "res_field": r_int_field,
                "y": y_grid, "x": x_grid,
                "left": r_left, "top": r_top, "bottom": r_bot,
            })
        panels.append(_make_panel(out, int(ids[b]), int(rank), panel_fields))
    ad_time = time.perf_counter() - t_ad

    csv_path = out / "residual_adequacy_stats.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=STATS_FIELDS)
        w.writeheader()
        w.writerows(rows)

    summary = _summarize(
        rows, calib, cfg, checkpoint, data_dir, split, mu, sigma,
        worst_ids=[int(ids[int(b)]) for b in worst],
        timing={
            "predict_rank_s": pred_time,
            "calibration_s": calib_time,
            "worst_k_ad_s": ad_time,
            "total_s": time.perf_counter() - t0,
        },
    )
    summary["csv"] = str(csv_path)
    summary["plots"] = [str(p) for p in panels]

    json_path = out / "residual_adequacy_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps({k: summary[k] for k in (
        "worst_k_sim_ids", "calibration", "aggregate", "by_time",
    )}, indent=2))
    print(f"\nWrote: {csv_path}\n       {json_path}")
    for p in panels:
        print(f"       {p}")
    if cfg.warnings:
        print("\nConfig-recovery warnings:")
        for wmsg in cfg.warnings:
            print(f"  ! {wmsg}")
    return summary, out


def _checkpoint_out(checkpoint: str) -> Path:
    p = Path(checkpoint).expanduser()
    return (p if p.is_dir() else p.parent) / "forcing_diag"


def _summarize(
    rows: list[dict[str, Any]],
    calib: dict[str, float],
    cfg,
    checkpoint: str,
    data_dir: str | None,
    split: str,
    mu: float,
    sigma: float,
    *,
    worst_ids: list[int],
    timing: dict[str, float],
) -> dict[str, Any]:
    def _mean(col: str) -> float:
        vals = [float(r[col]) for r in rows if np.isfinite(float(r[col]))]
        return float(np.mean(vals)) if vals else float("nan")

    # by-lead-bin aggregation (3 bins over lead_frac).
    by_time: dict[str, dict[str, float]] = {}
    if rows:
        leads = np.array([r["lead_frac"] for r in rows])
        bins = np.clip((leads * 3).astype(int), 0, 2)
        for i in range(3):
            m = bins == i
            if m.any():
                by_time[f"lead_bin_{i}"] = {
                    "mean_gnrmse": float(np.mean([rows[j]["gnrmse"] for j in np.where(m)[0]])),
                    "mean_err_rms_K": float(np.mean([rows[j]["err_rms_K"] for j in np.where(m)[0]])),
                    "mean_interior_res_rms": float(
                        np.mean([rows[j]["interior_res_rms"] for j in np.where(m)[0]])
                    ),
                    "mean_left_bc_res_rms": float(
                        np.mean([rows[j]["left_bc_res_rms"] for j in np.where(m)[0]])
                    ),
                }

    return {
        "checkpoint": str(checkpoint),
        "data_dir": str(data_dir) if data_dir else None,
        "split": split,
        "metric": "gnrmse = rmse_K / sigma_global on T - T_RIGHT deviation field",
        "residual_units": (
            "normalized T_tilde (matches training loss); physical-T residual = "
            "value * sigma"
        ),
        "hard_left_flux": bool(cfg.hard_left_flux),
        "hard_right_dirichlet": bool(cfg.hard_right_dirichlet),
        "mu_global": mu,
        "sigma_global": sigma,
        "config_warnings": list(cfg.warnings),
        "config_provenance": cfg.provenance,
        "n_rows": int(len(rows)),
        "worst_k_sim_ids": worst_ids,
        "calibration": calib,
        "aggregate": {
            "mean_gnrmse": _mean("gnrmse"),
            "mean_err_rms_K": _mean("err_rms_K"),
            "mean_interior_res_rms": _mean("interior_res_rms"),
            "mean_interior_res_calib_ratio": _mean("interior_res_calib_ratio"),
            "mean_left_bc_res_rms": _mean("left_bc_res_rms"),
            "mean_left_bc_calib_ratio": _mean("left_bc_calib_ratio"),
            "mean_ic_res_rms": _mean("ic_res_rms"),
            "mean_ic_calib_ratio": _mean("ic_calib_ratio"),
            "boundary_vs_nearwall": {
                "mean_left_bc_res_rms": _mean("left_bc_res_rms"),
                "mean_err_rms_left_K": _mean("err_rms_left_K"),
            },
            "mean_corr_interior_res_err_SECONDARY": _mean("corr_interior_res_err"),
        },
        "by_time": by_time,
        "interpretation_guardrails": GUARDRAILS,
        "timing_s": timing,
    }


def _make_panel(
    out: Path, sim_id: int, rank: int, fields: list[dict[str, Any]]
) -> Path:
    """Per-sim panel: rows=snapshots, cols=(error K | |interior res| | boundary)."""
    nrow = max(len(fields), 1)
    fig, axes = plt.subplots(nrow, 3, figsize=(11.5, 3.1 * nrow), squeeze=False)
    for i, fd in enumerate(fields):
        ax0, ax1, ax2 = axes[i]
        im0 = ax0.imshow(fd["err_K"].T, origin="lower", aspect="auto", cmap="magma")
        ax0.set_title(f"|error| K  (t={fd['t']:.3g})")
        ax0.set_xlabel("x idx"); ax0.set_ylabel("y idx")
        fig.colorbar(im0, ax=ax0, fraction=0.046)

        im1 = ax1.imshow(
            fd["res_field"].T, origin="lower", aspect="auto", cmap="viridis"
        )
        ax1.set_title("|interior residual| (tilde)")
        ax1.set_xlabel("x idx"); ax1.set_ylabel("y idx")
        fig.colorbar(im1, ax=ax1, fraction=0.046)

        ax2.plot(fd["y"], fd["left"], label="left BC (x=0)", color="C3")
        ax2.plot(fd["x"], fd["top"], label="top (y=1)", color="C0", alpha=0.7)
        ax2.plot(fd["x"], fd["bottom"], label="bottom (y=0)", color="C2", alpha=0.7)
        ax2.axhline(0.0, color="k", lw=0.6)
        ax2.set_title("boundary residuals (tilde)")
        ax2.set_xlabel("coord"); ax2.set_ylabel("residual")
        ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.3)
    fig.suptitle(f"sim {sim_id} (rank {rank}, worst-K residual adequacy)")
    fig.tight_layout()
    p = out / f"residual_panel_sim{sim_id}_rank{rank}.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", required=True, help="cvit_best.pt or seed dir.")
    parser.add_argument("--data-dir", default=None, help="Dataset dir; defaults to DATA_DIR.")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--num-sims", type=int, default=8, help="Pool size to rank.")
    parser.add_argument("--sim-ids", default="", help="Comma-separated ids; overrides split.")
    parser.add_argument("--worst-k", type=int, default=6, help="Worst sims to analyze.")
    parser.add_argument("--snapshots", type=int, default=2, help="Snapshots per sim.")
    parser.add_argument("--snapshot-times", default="", help="Explicit t list; overrides --snapshots.")
    parser.add_argument("--eval-stride", type=int, default=2, help="Spatial subsample for AD grid.")
    parser.add_argument("--query-chunk", type=int, default=512, help="AD query chunk size.")
    parser.add_argument("--calib-sims", type=int, default=8)
    parser.add_argument("--calib-n-r", type=int, default=4096)
    parser.add_argument("--calib-n-bc", type=int, default=1024)
    parser.add_argument("--query-batch", type=int, default=8, help="predict_grid batch.")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_residual_adequacy(
        args.checkpoint,
        args.data_dir,
        split=args.split,
        num_sims=args.num_sims,
        sim_ids=args.sim_ids or None,
        worst_k=args.worst_k,
        snapshots=args.snapshots,
        snapshot_times=args.snapshot_times or None,
        eval_stride=args.eval_stride,
        query_chunk=args.query_chunk,
        calib_sims=args.calib_sims,
        calib_n_r=args.calib_n_r,
        calib_n_bc=args.calib_n_bc,
        query_batch=args.query_batch,
        out_dir=args.out_dir,
        device=args.device,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
