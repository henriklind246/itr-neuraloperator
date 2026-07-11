"""Localize the physics-only forcing PINO ~5% plateau across a full error surface.

Highest-leverage Phase-1 diagnostic. The trainer reports a single pooled
``val_gnrmse``; this extends it to a stratified surface so the 5% can be
attributed to a temporal family (e.g. pulse-like forcing), a spatial profile,
long leads, weak/strong forcing severity, or the near-wall region -- and to a
cheap forcing-pathway ablation that says whether the model uses the encoder
image at all. Runs CPU-only on a supplied ``--checkpoint`` + the small local
probe dataset.

Metric: ``gnrmse = rmse_K / sigma_global`` on ``T`` (``T_RIGHT`` cancels in the
deviation error, so this equals the trainer's deviation-field gnrmse). Reported
per (sim, snapshot); comparable to ``val_gnrmse`` in ``train_metrics.csv``.

Example (smoke; numbers not scientifically meaningful):
  .venv/bin/python scripts/diagnose_forcing_error_surface.py \\
    --checkpoint runs/pino_smoke/config0/seed42 \\
    --data-dir data/diffusion_forcing_probe_cvit_best --num-sims 8 --out-dir /tmp/diag_es
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from problems.diffusion import T_RIGHT  # noqa: E402
from scripts._forcing_diag_common import (  # noqa: E402
    forcing_pathway_probe,
    forcing_severity,
    load_model_and_data,
    load_sim_params,
    predict_grid,
    select_ids,
)

CSV_FIELDS = [
    "sim_id", "temporal_family", "spatial_family",
    "q_rms", "Q_in", "amp",
    "t", "lead_frac", "rmse_K", "gnrmse", "rel_l2_dev",
    "rmse_left", "rmse_mid", "rmse_right",
]

PATHWAY_FIELDS = [
    "sim_id", "case", "tensor_changed", "temporal_family", "spatial_family",
    "delta_vs_baseline_K", "gnrmse", "skipped", "skip_reason",
]


def _region_masks(x_grid: np.ndarray) -> dict[str, np.ndarray]:
    """Boolean (Nx,) masks for the near-wall/left, mid, and right thirds."""
    lo, hi = x_grid.min(), x_grid.max()
    a = lo + (hi - lo) / 3.0
    b = lo + 2.0 * (hi - lo) / 3.0
    return {
        "left": x_grid < a,   # x < 1/3: near the forcing wall
        "mid": (x_grid >= a) & (x_grid < b),
        "right": x_grid >= b,
    }


def _region_rmse(err2: np.ndarray, mask: np.ndarray) -> float:
    """RMS-K over an x-region; ``err2`` is squared error ``(Nx, Ny)``."""
    if not mask.any():
        return float("nan")
    return float(np.sqrt(err2[mask].mean()))


def _tercile_labels(values: np.ndarray) -> np.ndarray:
    """Per-item low/mid/high tercile labels by value (ties -> lower band)."""
    out = np.array(["mid"] * len(values), dtype=object)
    if len(values) >= 3:
        q1, q2 = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
        out[values <= q1] = "low"
        out[(values > q1) & (values <= q2)] = "mid"
        out[values > q2] = "high"
    return out


def _mean_by(rows: list[dict[str, Any]], key: str, col: str = "gnrmse") -> dict[str, float]:
    groups: dict[str, list[float]] = {}
    for r in rows:
        groups.setdefault(str(r[key]), []).append(float(r[col]))
    return {k: float(np.mean(v)) for k, v in sorted(groups.items())}


def run_error_surface(
    checkpoint: str,
    data_dir: str | None,
    *,
    split: str = "val",
    num_sims: int = 12,
    sim_ids: str | None = None,
    query_batch: int = 8,
    out_dir: str | None = None,
    device: str = "auto",
    n_lead_bins: int = 5,
    dense_ny: int = 256,
    dense_nt: int = 1024,
) -> tuple[dict[str, Any], Path]:
    """Compute the error surface + pathway probe; write CSV/JSON/PNGs.

    Returns ``(summary, out_dir)``.
    """
    model, data, cfg = load_model_and_data(checkpoint, data_dir, device)
    sim_params = load_sim_params(cfg.config)
    ids = select_ids(data, split=split, num_sims=num_sims, sim_ids=sim_ids)
    params = [dict(sim_params[int(i)]) for i in ids]

    x_grid = np.asarray(data["x_grid"], dtype=np.float64)
    y_grid = np.asarray(data["y_grid"], dtype=np.float64)
    t_grid = np.asarray(data["t_grid"], dtype=np.float64)
    nx, ny, nt = int(x_grid.shape[0]), int(y_grid.shape[0]), int(t_grid.shape[0])
    sigma = float(cfg.sigma)
    t_final = float(t_grid[-1]) if nt else float(cfg.t_final)

    pred_K = predict_grid(model, params, data, cfg, t_grid, device, query_batch)
    truth = np.asarray(data["trajectories"][np.asarray(ids)], dtype=np.float32)
    if truth.shape != pred_K.shape:
        raise ValueError(f"shape mismatch pred {pred_K.shape} vs truth {truth.shape}.")

    # Severity scalars (dense grid) per sim.
    y_dense = np.linspace(cfg.c_dom, cfg.d_dom, dense_ny)
    t_dense = np.linspace(0.0, t_final, dense_nt)
    sev = [forcing_severity(p, y_dense, t_dense, cfg.t_ramp) for p in params]
    q_rms_arr = np.array([s["q_rms"] for s in sev])
    Q_in_arr = np.array([s["Q_in"] for s in sev])
    qrms_terc = _tercile_labels(q_rms_arr)
    Qin_terc = _tercile_labels(np.abs(Q_in_arr))

    masks = _region_masks(x_grid)
    rows: list[dict[str, Any]] = []
    # per-sim carry of severity tercile so per-row summaries can group by it.
    row_qrms_terc: list[str] = []
    row_Qin_terc: list[str] = []
    for b, sid in enumerate(np.asarray(ids)):
        for k in range(nt):
            err2 = (pred_K[b, k] - truth[b, k]).astype(np.float64) ** 2  # (Nx,Ny)
            rmse_k = float(np.sqrt(err2.mean()))
            dev = truth[b, k].astype(np.float64) - float(T_RIGHT)
            den = float(np.sqrt((dev ** 2).sum()))
            rel = float(np.sqrt(err2.sum()) / den) if den > 1e-12 else float("nan")
            rows.append({
                "sim_id": int(sid),
                "temporal_family": str(params[b].get("temporal_family", "")),
                "spatial_family": str(params[b].get("spatial_family", "")),
                "q_rms": float(q_rms_arr[b]),
                "Q_in": float(Q_in_arr[b]),
                "amp": float(sev[b]["amp"]),
                "t": float(t_grid[k]),
                "lead_frac": float(t_grid[k] / t_final) if t_final > 0 else 0.0,
                "rmse_K": rmse_k,
                "gnrmse": rmse_k / (sigma + 1e-8),
                "rel_l2_dev": rel,
                "rmse_left": _region_rmse(err2, masks["left"]),
                "rmse_mid": _region_rmse(err2, masks["mid"]),
                "rmse_right": _region_rmse(err2, masks["right"]),
            })
            row_qrms_terc.append(str(qrms_terc[b]))
            row_Qin_terc.append(str(Qin_terc[b]))

    out = Path(out_dir).expanduser() if out_dir else _checkpoint_out(checkpoint)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "error_surface.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)

    # Pathway probe on the first min(num_sims, 8) sims.
    n_probe = min(len(ids), 8)
    probe_rows = forcing_pathway_probe(
        model, params[:n_probe], np.asarray(ids)[:n_probe], data, cfg, device
    )
    probe_path = out / "pathway_probe.csv"
    with open(probe_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PATHWAY_FIELDS)
        w.writeheader()
        w.writerows(probe_rows)

    # Lead-time binned means.
    lead = np.array([r["lead_frac"] for r in rows])
    gnr = np.array([r["gnrmse"] for r in rows])
    bin_idx = np.clip((lead * n_lead_bins).astype(int), 0, n_lead_bins - 1)
    by_lead = {
        f"bin_{i}": float(gnr[bin_idx == i].mean()) if (bin_idx == i).any() else float("nan")
        for i in range(n_lead_bins)
    }

    cell_groups: dict[str, list[float]] = {}
    for r in rows:
        cell_groups.setdefault(f"{r['temporal_family']}|{r['spatial_family']}", []).append(
            float(r["gnrmse"])
        )
    cell = {k: float(np.mean(v)) for k, v in sorted(cell_groups.items())}

    def _terc_summary(labels: list[str]) -> dict[str, float]:
        g: dict[str, list[float]] = {}
        for r, lab in zip(rows, labels):
            g.setdefault(lab, []).append(float(r["gnrmse"]))
        return {k: float(np.mean(v)) for k, v in sorted(g.items())}

    region_mean = {
        reg: float(np.nanmean([r[f"rmse_{reg}"] for r in rows]))
        for reg in ("left", "mid", "right")
    }

    summary: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "data_dir": str(data_dir) if data_dir else None,
        "split": split,
        "metric": "gnrmse = rmse_K / sigma_global on T - T_RIGHT deviation field",
        "n_sims": int(len(ids)),
        "n_rows": int(len(rows)),
        "sigma_global": sigma,
        "config_warnings": list(cfg.warnings),
        "config_provenance": cfg.provenance,
        "hard_left_flux": bool(cfg.hard_left_flux),
        "mean_gnrmse": float(gnr.mean()),
        "by_temporal_family": _mean_by(rows, "temporal_family"),
        "by_spatial_family": _mean_by(rows, "spatial_family"),
        "by_cell": cell,
        "by_lead_bin": by_lead,
        "by_qrms_tercile": _terc_summary(row_qrms_terc),
        "by_Qin_tercile": _terc_summary(row_Qin_terc),
        "by_region_rmse_K": region_mean,
        "csv": str(csv_path),
        "pathway_probe_csv": str(probe_path),
    }

    plots = _make_plots(rows, out)
    summary["plots"] = [str(p) for p in plots]

    json_path = out / "error_surface_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: summary[k] for k in (
        "mean_gnrmse", "by_temporal_family", "by_spatial_family",
        "by_lead_bin", "by_qrms_tercile", "by_region_rmse_K",
    )}, indent=2))
    print(f"\nWrote: {csv_path}\n       {json_path}\n       {probe_path}")
    for p in plots:
        print(f"       {p}")
    if cfg.warnings:
        print("\nConfig-recovery warnings:")
        for w in cfg.warnings:
            print(f"  ! {w}")
    return summary, out


def _checkpoint_out(checkpoint: str) -> Path:
    p = Path(checkpoint).expanduser()
    return (p if p.is_dir() else p.parent) / "forcing_diag"


def _make_plots(rows: list[dict[str, Any]], out: Path) -> list[Path]:
    """Temporal x spatial mean-gnrmse heatmap + gnrmse-vs-lead per family."""
    paths: list[Path] = []
    tfams = sorted({r["temporal_family"] for r in rows})
    sfams = sorted({r["spatial_family"] for r in rows})

    grid = np.full((len(tfams), len(sfams)), np.nan)
    for i, tf in enumerate(tfams):
        for j, sf in enumerate(sfams):
            vals = [
                r["gnrmse"] for r in rows
                if r["temporal_family"] == tf and r["spatial_family"] == sf
            ]
            if vals:
                grid[i, j] = float(np.mean(vals))
    fig, ax = plt.subplots(figsize=(1.6 + 1.1 * len(sfams), 1.6 + 0.9 * len(tfams)))
    im = ax.imshow(grid * 100.0, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(sfams)), sfams, rotation=30, ha="right")
    ax.set_yticks(range(len(tfams)), tfams)
    ax.set_xlabel("spatial family")
    ax.set_ylabel("temporal family")
    ax.set_title("mean gnrmse (%) by temporal x spatial")
    for i in range(len(tfams)):
        for j in range(len(sfams)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j] * 100:.1f}", ha="center", va="center",
                        color="w", fontsize=8)
    fig.colorbar(im, ax=ax, label="gnrmse (%)")
    fig.tight_layout()
    p1 = out / "heatmap_temporal_spatial.png"
    fig.savefig(p1, dpi=110)
    plt.close(fig)
    paths.append(p1)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for tf in tfams:
        fr = [r for r in rows if r["temporal_family"] == tf]
        leads = sorted({r["lead_frac"] for r in fr})
        means = [
            float(np.mean([r["gnrmse"] for r in fr if r["lead_frac"] == L])) * 100.0
            for L in leads
        ]
        ax.plot(leads, means, marker="o", label=tf)
    ax.set_xlabel("lead fraction t / t_final")
    ax.set_ylabel("mean gnrmse (%)")
    ax.set_title("gnrmse vs lead time by temporal family")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p2 = out / "gnrmse_vs_lead.png"
    fig.savefig(p2, dpi=110)
    plt.close(fig)
    paths.append(p2)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="cvit_best.pt or seed dir.")
    parser.add_argument("--data-dir", default=None, help="Dataset dir; defaults to DATA_DIR.")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--num-sims", type=int, default=12)
    parser.add_argument("--sim-ids", default="", help="Comma-separated ids; overrides split.")
    parser.add_argument("--query-batch", type=int, default=8)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    run_error_surface(
        args.checkpoint,
        args.data_dir,
        split=args.split,
        num_sims=args.num_sims,
        sim_ids=args.sim_ids or None,
        query_batch=args.query_batch,
        out_dir=args.out_dir,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
