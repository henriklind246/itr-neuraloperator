"""Encoder-image aliasing analysis for the forcing PINO (no model required).

The forcing image fed to the ForcingCViT encoder is 2-D in ``(y, t)``, so sampling
it at ``(ny_img, nt_img)`` can alias in BOTH directions -- a narrow spatial
profile (patch / narrow gaussian / triangle) or a sharp temporal pulse can be
attenuated or mislocated before the model ever sees it. This script sweeps both
axes and quantifies, per temporal family x spatial profile, how much a dense
continuous ``q_L`` loses when rendered at each ``(ny_img, nt_img)``.

Two complementary comparisons vs the dense reference:
  1. Bilinear interpolation of the sampled coarse image back onto the dense grid
     -- used ONLY as a documented aliasing proxy. This is NOT a claim to
     reproduce the CViT patch encoder, which consumes the sampled image directly
     and does not interpolate it back to a dense temporal grid.
  2. The exact ``build_forcing_image`` sampled image (what the encoder actually
     ingests), compared on cell-integrated quantities (net impulse ``Q_in``,
     peak) with no interpolation -- a stronger, encoder-faithful metric.

Pure numpy, deterministic. No checkpoint needed; if ``--checkpoint`` is given the
sweeps are anchored to (and always include) the recovered ``ny_img``/``nt_img``.

Example:
  .venv/bin/python scripts/diagnose_forcing_image_fidelity.py \\
    --data-dir data/diffusion_forcing_probe_cvit_best --out-dir /tmp/diag_fid
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

from problems.forcing import A_AMP_REF, SPATIAL_FAMILY_ORDER  # noqa: E402
from src.operators.train_pino import build_forcing_image  # noqa: E402
from src.physics.boundary_forcing import (  # noqa: E402
    SPATIAL_SAMPLERS,
    TEMPORAL_FAMILY_ORDER,
    TEMPORAL_SAMPLERS,
    reconstruct_qL,
)

import torch  # noqa: E402


CSV_FIELDS = [
    "temporal_family", "spatial_family", "ny_img", "nt_img",
    "joint_rel_l2", "temporal_rel_l2", "spatial_rel_l2",
    "peak_t_err", "peak_y_err", "temporal_width_err", "spatial_width_err",
    "peak_ratio", "integrated_impulse_err", "temporal_centroid_err",
    "max_temporal_slope_atten",
    "integrated_impulse_err_sampled", "peak_ratio_sampled",
]


def _bilinear_to_dense(
    img: np.ndarray,
    y_img: np.ndarray,
    t_img: np.ndarray,
    y_dense: np.ndarray,
    t_dense: np.ndarray,
) -> np.ndarray:
    """Separable bilinear interpolation of a coarse ``(ny,nt)`` image to dense.

    Documented aliasing PROXY only; the CViT encoder does not do this.
    """
    # interpolate along t (columns) for each coarse y-row, then along y.
    tmp = np.empty((img.shape[0], t_dense.shape[0]), dtype=np.float64)
    for i in range(img.shape[0]):
        tmp[i] = np.interp(t_dense, t_img, img[i])
    out = np.empty((y_dense.shape[0], t_dense.shape[0]), dtype=np.float64)
    for j in range(t_dense.shape[0]):
        out[:, j] = np.interp(y_dense, y_img, tmp[:, j])
    return out


def _profiles(q: np.ndarray, y: np.ndarray, t: np.ndarray) -> dict[str, Any]:
    """Marginal profiles + moments of a ``q_L`` image ``(Ny, Nt)``."""
    A = np.trapz(q, y, axis=0)          # temporal marginal (integral over y)
    S = np.trapz(q, t, axis=1)          # spatial marginal  (integral over t)
    Q_in = float(np.trapz(A, t))
    absA, absS = np.abs(A), np.abs(S)
    tA = float(np.trapz(absA, t))
    sS = float(np.trapz(absS, y))
    cen_t = float(np.trapz(t * absA, t) / tA) if tA > 1e-30 else float("nan")
    cen_y = float(np.trapz(y * absS, y) / sS) if sS > 1e-30 else float("nan")
    var_t = float(np.trapz((t - cen_t) ** 2 * absA, t) / tA) if tA > 1e-30 else float("nan")
    var_y = float(np.trapz((y - cen_y) ** 2 * absS, y) / sS) if sS > 1e-30 else float("nan")
    slope = np.gradient(A, t)
    return {
        "A": A, "S": S, "Q_in": Q_in,
        "peak_t": float(t[int(np.argmax(absA))]),
        "peak_y": float(y[int(np.argmax(absS))]),
        "centroid_t": cen_t,
        "width_t": float(np.sqrt(var_t)) if np.isfinite(var_t) else float("nan"),
        "width_y": float(np.sqrt(var_y)) if np.isfinite(var_y) else float("nan"),
        "max_slope": float(np.max(np.abs(slope))),
        "peak_abs": float(np.max(np.abs(q))),
    }


def _rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.sqrt(np.sum(b.astype(np.float64) ** 2)))
    if den < 1e-30:
        return float("nan")
    return float(np.sqrt(np.sum((a - b).astype(np.float64) ** 2)) / den)


def fidelity_metrics(
    q_ref: np.ndarray,
    q_rec: np.ndarray,
    q_sampled: np.ndarray,
    y_dense: np.ndarray,
    t_dense: np.ndarray,
    y_img: np.ndarray,
    t_img: np.ndarray,
) -> dict[str, float]:
    """Separable + joint fidelity metrics of ``q_rec``/``q_sampled`` vs ``q_ref``.

    ``q_rec`` is the bilinear-proxy reconstruction on the dense grid; ``q_sampled``
    is the encoder-faithful coarse image on ``(y_img, t_img)``.
    """
    pr = _profiles(q_ref, y_dense, t_dense)
    pc = _profiles(q_rec, y_dense, t_dense)
    ps = _profiles(q_sampled, y_img, t_img)  # cell-integrated on the sampled grid

    ref_slope = pr["max_slope"] if pr["max_slope"] > 1e-30 else float("nan")
    q_ref_peak = pr["peak_abs"] if pr["peak_abs"] > 1e-30 else float("nan")
    Q_ref = pr["Q_in"]
    Q_den = abs(Q_ref) if abs(Q_ref) > 1e-30 else float("nan")

    return {
        "joint_rel_l2": _rel_l2(q_rec, q_ref),
        "temporal_rel_l2": _rel_l2(pc["A"], pr["A"]),
        "spatial_rel_l2": _rel_l2(pc["S"], pr["S"]),
        "peak_t_err": abs(pc["peak_t"] - pr["peak_t"]),
        "peak_y_err": abs(pc["peak_y"] - pr["peak_y"]),
        "temporal_width_err": abs(pc["width_t"] - pr["width_t"]),
        "spatial_width_err": abs(pc["width_y"] - pr["width_y"]),
        "peak_ratio": pc["peak_abs"] / q_ref_peak,
        "integrated_impulse_err": abs(pc["Q_in"] - Q_ref) / Q_den,
        "temporal_centroid_err": abs(pc["centroid_t"] - pr["centroid_t"]),
        "max_temporal_slope_atten": pc["max_slope"] / ref_slope,
        "integrated_impulse_err_sampled": abs(ps["Q_in"] - Q_ref) / Q_den,
        "peak_ratio_sampled": ps["peak_abs"] / q_ref_peak,
    }


def _sample_params(
    temporal_family: str,
    spatial_family: str,
    dt: float,
    t_final: float,
    c_dom: float,
    d_dom: float,
    seed: int,
) -> dict[str, Any]:
    """One deterministic param draw for a (temporal, spatial) family pair."""
    rng = np.random.default_rng(seed)
    tp = TEMPORAL_SAMPLERS[temporal_family](
        rng, dt=dt, t_final=t_final,
        t_on=0.0, t_off=0.2 * t_final / max(t_final, 1e-12), phase=0.0, tukey_alpha=0.5,
    )
    sp = SPATIAL_SAMPLERS[spatial_family](rng, c=c_dom, d=d_dom)
    return {
        "temporal_family": temporal_family, "temporal_params": tp,
        "spatial_family": spatial_family, "spatial_params": sp,
    }


def run_image_fidelity(
    data_dir: str | None = None,
    checkpoint: str | None = None,
    *,
    nt_img_list: tuple[int, ...] = (64, 128, 256, 512),
    ny_img_list: tuple[int, ...] = (32, 64, 128),
    out_dir: str | None = None,
    dense_ny: int = 1024,
    dense_nt: int = 4096,
    t_final: float | None = None,
    c_dom: float = 0.0,
    d_dom: float = 1.0,
    t_ramp: float = 0.0,
    a_ref: float = A_AMP_REF,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], Path]:
    """Sweep (family x profile x ny_img x nt_img); write CSV + plots.

    Returns ``(rows, out_dir)``. ``--checkpoint`` (if given) anchors the sweeps to
    the recovered encoder-image resolution and domain/ramp.
    """
    anchor_ny: int | None = None
    anchor_nt: int | None = None
    if checkpoint:
        from scripts._forcing_diag_common import _checkpoint_path, recover_config
        ckpt = torch.load(_checkpoint_path(checkpoint), map_location="cpu", weights_only=False)
        cfg = recover_config(ckpt, None)
        anchor_ny, anchor_nt = int(cfg.ny_img), int(cfg.nt_img)
        c_dom, d_dom = cfg.c_dom, cfg.d_dom
        t_ramp, a_ref = cfg.t_ramp, cfg.a_ref
        if t_final is None:
            t_final = cfg.t_final
        ny_img_list = tuple(sorted(set(ny_img_list) | {anchor_ny}))
        nt_img_list = tuple(sorted(set(nt_img_list) | {anchor_nt}))

    if data_dir:
        dd = Path(data_dir).expanduser()
        y_grid = np.load(dd / "y_grid.npy")
        t_grid = np.load(dd / "t_grid.npy")
        c_dom, d_dom = float(y_grid[0]), float(y_grid[-1])
        if t_final is None:
            t_final = float(t_grid[-1])
    if t_final is None:
        t_final = 0.3
    dt = float(t_final) / 100.0

    y_dense = np.linspace(c_dom, d_dom, dense_ny)
    t_dense = np.linspace(0.0, t_final, dense_nt)

    rows: list[dict[str, Any]] = []
    for tf in TEMPORAL_FAMILY_ORDER:
        for sf in SPATIAL_FAMILY_ORDER:
            params = _sample_params(tf, sf, dt, t_final, c_dom, d_dom, seed)
            q_image, _ = reconstruct_qL(
                params["temporal_family"], params["temporal_params"],
                params["spatial_family"], params["spatial_params"], t_ramp=t_ramp,
            )
            q_ref = np.asarray(q_image(y_dense, t_dense), dtype=np.float64)
            for ny_img in ny_img_list:
                for nt_img in nt_img_list:
                    y_img = np.linspace(c_dom, d_dom, ny_img)
                    t_img = np.linspace(0.0, t_final, nt_img)
                    q_coarse = np.asarray(q_image(y_img, t_img), dtype=np.float64)
                    q_rec = _bilinear_to_dense(q_coarse, y_img, t_img, y_dense, t_dense)
                    # encoder-faithful sampled image (build_forcing_image / a_ref);
                    # multiply back by a_ref to recover q at the img nodes.
                    u = build_forcing_image(
                        [params], y_img, t_img, a_ref, torch.device("cpu"), t_ramp
                    )
                    q_sampled = u[0, 0].numpy().astype(np.float64) * float(a_ref)
                    m = fidelity_metrics(
                        q_ref, q_rec, q_sampled, y_dense, t_dense, y_img, t_img
                    )
                    rows.append({
                        "temporal_family": tf, "spatial_family": sf,
                        "ny_img": int(ny_img), "nt_img": int(nt_img), **m,
                    })

    out = Path(out_dir).expanduser() if out_dir else Path.cwd() / "forcing_image_fidelity"
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "image_fidelity.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)

    plots = _make_plots(
        rows, out, y_dense, t_dense, dt, t_final, c_dom, d_dom, t_ramp, a_ref,
        anchor_ny, anchor_nt, ny_img_list, nt_img_list, seed,
    )

    summary = {
        "data_dir": str(data_dir) if data_dir else None,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "anchor_ny_img": anchor_ny, "anchor_nt_img": anchor_nt,
        "ny_img_list": list(ny_img_list), "nt_img_list": list(nt_img_list),
        "t_final": float(t_final), "domain": [c_dom, d_dom], "t_ramp": float(t_ramp),
        "n_rows": len(rows),
        "note": (
            "bilinear reconstruction is an ALIASING PROXY, not the CViT encoder; "
            "*_sampled columns are the encoder-faithful cell-integrated metrics."
        ),
        "csv": str(csv_path), "plots": [str(p) for p in plots],
    }
    json_path = out / "image_fidelity_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote: {csv_path}\n       {json_path}")
    for p in plots:
        print(f"       {p}")
    return rows, out


def _make_plots(
    rows, out, y_dense, t_dense, dt, t_final, c_dom, d_dom, t_ramp, a_ref,
    anchor_ny, anchor_nt, ny_img_list, nt_img_list, seed,
) -> list[Path]:
    """Overlay plots (pulse temporal, narrow-gaussian spatial) + joint heatmap."""
    paths: list[Path] = []
    ny0 = anchor_ny if anchor_ny else ny_img_list[0]
    nt0 = anchor_nt if anchor_nt else nt_img_list[0]

    def _overlay(tf, sf, axis, fname, title):
        params = _sample_params(tf, sf, dt, t_final, c_dom, d_dom, seed)
        q_image, _ = reconstruct_qL(
            params["temporal_family"], params["temporal_params"],
            params["spatial_family"], params["spatial_params"], t_ramp=t_ramp,
        )
        q_ref = np.asarray(q_image(y_dense, t_dense), dtype=np.float64)
        y_img = np.linspace(c_dom, d_dom, ny0)
        t_img = np.linspace(0.0, t_final, nt0)
        q_coarse = np.asarray(q_image(y_img, t_img), dtype=np.float64)
        q_rec = _bilinear_to_dense(q_coarse, y_img, t_img, y_dense, t_dense)
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        if axis == "t":
            ax.plot(t_dense, np.trapz(q_ref, y_dense, axis=0), label="dense ref", lw=2)
            ax.plot(t_dense, np.trapz(q_rec, y_dense, axis=0), "--", label=f"recon {ny0}x{nt0}")
            ax.set_xlabel("t")
            ax.set_ylabel(r"$\int q_L\,dy$")
        else:
            ax.plot(y_dense, np.trapz(q_ref, t_dense, axis=1), label="dense ref", lw=2)
            ax.plot(y_dense, np.trapz(q_rec, t_dense, axis=1), "--", label=f"recon {ny0}x{nt0}")
            ax.set_xlabel("y")
            ax.set_ylabel(r"$\int q_L\,dt$")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p = out / fname
        fig.savefig(p, dpi=110)
        plt.close(fig)
        return p

    paths.append(_overlay("pulse_train", "uniform", "t",
                          "overlay_pulse_temporal.png",
                          "pulse_train x uniform: temporal marginal"))
    paths.append(_overlay("sin", "gaussian", "y",
                          "overlay_gaussian_spatial.png",
                          "sin x gaussian: spatial marginal"))

    # joint-error heatmap at the anchor (or first) ny_img x nt_img.
    anchor_rows = [r for r in rows if r["ny_img"] == ny0 and r["nt_img"] == nt0]
    tfams = list(TEMPORAL_FAMILY_ORDER)
    sfams = list(SPATIAL_FAMILY_ORDER)
    grid = np.full((len(tfams), len(sfams)), np.nan)
    for r in anchor_rows:
        i = tfams.index(r["temporal_family"])
        j = sfams.index(r["spatial_family"])
        grid[i, j] = r["joint_rel_l2"] * 100.0
    fig, ax = plt.subplots(figsize=(1.6 + 1.1 * len(sfams), 1.6 + 0.9 * len(tfams)))
    im = ax.imshow(grid, aspect="auto", cmap="magma")
    ax.set_xticks(range(len(sfams)), sfams, rotation=30, ha="right")
    ax.set_yticks(range(len(tfams)), tfams)
    ax.set_title(f"joint (y,t) rel-L2 (%) at {ny0}x{nt0}")
    for i in range(len(tfams)):
        for j in range(len(sfams)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center",
                        color="w", fontsize=8)
    fig.colorbar(im, ax=ax, label="rel-L2 (%)")
    fig.tight_layout()
    p = out / "joint_error_heatmap.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    paths.append(p)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None, help="Dataset dir for y/t grids.")
    parser.add_argument("--checkpoint", default=None, help="Anchor sweeps to recovered res.")
    parser.add_argument("--nt-img-list", default="64,128,256,512")
    parser.add_argument("--ny-img-list", default="32,64,128")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--dense-ny", type=int, default=1024)
    parser.add_argument("--dense-nt", type=int, default=4096)
    args = parser.parse_args()
    run_image_fidelity(
        data_dir=args.data_dir,
        checkpoint=args.checkpoint,
        nt_img_list=tuple(int(x) for x in args.nt_img_list.split(",") if x.strip()),
        ny_img_list=tuple(int(x) for x in args.ny_img_list.split(",") if x.strip()),
        out_dir=args.out_dir,
        dense_ny=args.dense_ny,
        dense_nt=args.dense_nt,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
