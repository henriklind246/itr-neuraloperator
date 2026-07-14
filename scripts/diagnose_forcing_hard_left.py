from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from problems.diffusion import T_RIGHT
from problems.diffusion_forcing import K_SLAB
from problems.forcing import A_AMP_REF
from src.operators.train_pino import (
    build_cvit,
    build_forcing_image,
    load_diffusion_data,
)
from src.operators.utils import resolve_device
from src.physics.boundary_forcing import reconstruct_qL


DATA_FILE_NAMES = {
    "t_grid_path": "t_grid.npy",
    "x_grid_path": "x_grid.npy",
    "y_grid_path": "y_grid.npy",
    "trajectories.npy": "trajectories.npy",
    "sim_params_path": "sim_params.npy",
}


def _checkpoint_path(path: str) -> Path:
    p = Path(path).expanduser()
    return p / "cvit_best.pt" if p.is_dir() else p


def _sync_data_dir(config: dict[str, Any], data_dir: Path) -> None:
    config.setdefault("paths", {})["data_dir"] = str(data_dir)
    config.setdefault("data", {})
    for key, name in DATA_FILE_NAMES.items():
        config["data"][key] = str(data_dir / name)


def _forcing_image_meta(
    ckpt: dict[str, Any],
    config: dict[str, Any],
    data: dict[str, Any],
) -> dict[str, Any]:
    meta = dict(ckpt.get("forcing_image") or {})
    fcfg = config.get("training", {}).get("pino", {}).get("forcing", {}) or {}
    y_grid = np.asarray(data["y_grid"], dtype=np.float64)
    t_grid = np.asarray(data["t_grid"], dtype=np.float64)
    return {
        "ny_img": int(meta.get("ny_img") or fcfg.get("ny_img") or y_grid.shape[0]),
        "nt_img": int(meta.get("nt_img") or fcfg.get("nt_img") or 128),
        "a_ref": float(meta.get("a_ref") or fcfg.get("a_ref") or A_AMP_REF),
        "t_ramp": float(meta.get("t_ramp") or fcfg.get("ramp_seconds") or 0.0),
        "c_dom": float(meta.get("c_dom") or y_grid[0]),
        "d_dom": float(meta.get("d_dom") or y_grid[-1]),
        "t_final": float(meta.get("t_final") or t_grid[-1]),
    }


def _select_ids(data: dict[str, Any], args: argparse.Namespace) -> np.ndarray:
    if args.sim_ids:
        return np.asarray([int(x) for x in args.sim_ids.split(",") if x.strip()], dtype=int)
    key = f"{args.split}_ids"
    ids = np.asarray(data[key], dtype=int)
    return ids[: int(args.num_sims)]


def _q_grid(
    params: list[dict[str, Any]],
    y_grid: np.ndarray,
    t_grid: np.ndarray,
    nx: int,
    t_ramp: float,
) -> tuple[np.ndarray, list[float], list[str]]:
    bsz = len(params)
    nt = int(t_grid.shape[0])
    ny = int(y_grid.shape[0])
    q_all = np.empty((bsz, nt, nx * ny), dtype=np.float32)
    amps: list[float] = []
    fams: list[str] = []
    for b, p in enumerate(params):
        forcing = reconstruct_qL(
            p["temporal_family"],
            p["temporal_params"],
            p["spatial_family"],
            p["spatial_params"],
            t_ramp=t_ramp,
        )
        qg = np.asarray(
            forcing.evaluate_grid(y_grid, t_grid), dtype=np.float32
        )
        q_all[b] = np.tile(qg.T, (1, nx))
        amps.append(float(np.max(np.abs(qg))))
        fams.append(str(p.get("temporal_family", "")))
    return q_all, amps, fams


def _rmse(se: np.ndarray, count: int) -> np.ndarray:
    return np.sqrt(se / max(float(count), 1.0))


@torch.no_grad()
def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    ckpt_path = _checkpoint_path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    data_dir = Path(args.data_dir or os.environ.get("DATA_DIR", "")).expanduser()
    if data_dir:
        _sync_data_dir(config, data_dir)

    data = load_diffusion_data(config)
    mu = float(ckpt.get("mu_global", data["mu_global"]))
    sigma = float(ckpt.get("sigma_global", data["sigma_global"]))
    meta = _forcing_image_meta(ckpt, config, data)

    device = resolve_device(args.device)
    model = build_cvit(
        config,
        mu,
        sigma,
        grid_size=(int(meta["ny_img"]), int(meta["nt_img"])),
        t_final=float(meta["t_final"]),
        variant="forcing",
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    if not bool(getattr(model, "hard_left_flux", False)):
        raise ValueError("Checkpoint model was not built with hard_left_flux=True.")

    x_grid = torch.as_tensor(data["x_grid"], dtype=torch.float32, device=device)
    y_grid = torch.as_tensor(data["y_grid"], dtype=torch.float32, device=device)
    t_grid_np = np.asarray(data["t_grid"], dtype=np.float64)
    x_grid_np = np.asarray(data["x_grid"], dtype=np.float64)
    y_grid_np = np.asarray(data["y_grid"], dtype=np.float64)
    nx, ny, nt = int(x_grid.numel()), int(y_grid.numel()), int(t_grid_np.shape[0])
    gx, gy = torch.meshgrid(x_grid, y_grid, indexing="ij")
    coords_1 = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1).unsqueeze(0)
    x_flat_1 = coords_1[..., 0:1]

    ids = _select_ids(data, args)
    sp_path = Path(config["data"]["trajectories.npy"]).parent / "sim_params.npy"
    sim_params = np.load(str(sp_path), allow_pickle=True)

    y_img = np.linspace(float(meta["c_dom"]), float(meta["d_dom"]), int(meta["ny_img"]))
    t_img = np.linspace(0.0, float(meta["t_final"]), int(meta["nt_img"]))

    rows: list[dict[str, Any]] = []
    count = nt * nx * ny
    for start in range(0, len(ids), int(args.query_batch)):
        chunk = ids[start : start + int(args.query_batch)]
        params = [dict(sim_params[int(i)]) for i in chunk]
        bsz = len(params)
        coords = coords_1.expand(bsz, -1, -1)
        x_flat = x_flat_1.expand(bsz, -1, -1)
        u = build_forcing_image(
            params,
            y_img,
            t_img,
            float(meta["a_ref"]),
            device,
            float(meta["t_ramp"]),
        )
        u_zero = torch.zeros_like(u)
        q_np, amps, fams = _q_grid(params, y_grid_np, t_grid_np, nx, float(meta["t_ramp"]))
        q_all = torch.from_numpy(q_np).unsqueeze(-1).to(device)
        truth = torch.as_tensor(
            np.asarray(data["trajectories"][chunk], dtype=np.float32),
            device=device,
        )

        se_full = np.zeros(bsz, dtype=np.float64)
        se_no_q = np.zeros(bsz, dtype=np.float64)
        se_zero_u = np.zeros(bsz, dtype=np.float64)
        se_analytic = np.zeros(bsz, dtype=np.float64)
        se_full_minus_zero_u = np.zeros(bsz, dtype=np.float64)
        se_full_minus_analytic = np.zeros(bsz, dtype=np.float64)
        se_truth_dev = np.zeros(bsz, dtype=np.float64)

        for k, tk_val in enumerate(t_grid_np):
            tk = torch.full((bsz, nx * ny, 1), float(tk_val), device=device)
            q_left = q_all[:, k]
            full = model(u, coords, tk, q_left=q_left)
            no_q = model(u, coords, tk, q_left=None)
            zero_u = model(u_zero, coords, tk, q_left=q_left)
            g = -q_left * model.left_flux_scale
            analytic = model.t_right_tilde + g * (x_flat - 1.0)

            truth_k = truth[:, k].reshape(bsz, nx * ny, 1)
            for name, pred, acc in (
                ("full", full, se_full),
                ("no_q", no_q, se_no_q),
                ("zero_u", zero_u, se_zero_u),
                ("analytic", analytic, se_analytic),
            ):
                pred_k = pred * sigma + mu
                acc += ((pred_k - truth_k) ** 2).sum(dim=(1, 2)).detach().cpu().numpy()
            se_full_minus_zero_u += (
                ((full - zero_u) * sigma) ** 2
            ).sum(dim=(1, 2)).detach().cpu().numpy()
            se_full_minus_analytic += (
                ((full - analytic) * sigma) ** 2
            ).sum(dim=(1, 2)).detach().cpu().numpy()
            se_truth_dev += (
                (truth_k - float(T_RIGHT)) ** 2
            ).sum(dim=(1, 2)).detach().cpu().numpy()

        metrics = {
            "full": _rmse(se_full, count),
            "no_q": _rmse(se_no_q, count),
            "zero_u": _rmse(se_zero_u, count),
            "analytic": _rmse(se_analytic, count),
            "full_minus_zero_u": _rmse(se_full_minus_zero_u, count),
            "full_minus_analytic": _rmse(se_full_minus_analytic, count),
            "truth_dev": _rmse(se_truth_dev, count),
        }
        for j, sid in enumerate(chunk):
            rows.append({
                "sim_id": int(sid),
                "temporal_family": fams[j],
                "amp": amps[j],
                "full_rmse_K": float(metrics["full"][j]),
                "full_gnrmse_pct": float(metrics["full"][j] / (sigma + 1e-8) * 100.0),
                "analytic_rmse_K": float(metrics["analytic"][j]),
                "zero_u_rmse_K": float(metrics["zero_u"][j]),
                "no_q_rmse_K": float(metrics["no_q"][j]),
                "full_minus_zero_u_rmse_K": float(metrics["full_minus_zero_u"][j]),
                "full_minus_analytic_rmse_K": float(metrics["full_minus_analytic"][j]),
                "truth_dev_rmse_K": float(metrics["truth_dev"][j]),
            })

    out_dir = Path(args.out_dir).expanduser() if args.out_dir else ckpt_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "hard_left_probe.csv"
    json_path = out_dir / "hard_left_probe.json"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, Any] = {
        "checkpoint": str(ckpt_path),
        "data_dir": str(data_dir) if data_dir else None,
        "n_sims": len(rows),
        "sigma": sigma,
        "mean": {},
        "by_family": {},
        "csv": str(csv_path),
    }
    metric_cols = [k for k in rows[0].keys() if k.endswith("_K") or k.endswith("_pct")]
    for col in metric_cols:
        summary["mean"][col] = float(np.mean([float(r[col]) for r in rows]))
    for fam in sorted({str(r["temporal_family"]) for r in rows}):
        fam_rows = [r for r in rows if r["temporal_family"] == fam]
        summary["by_family"][fam] = {
            col: float(np.mean([float(r[col]) for r in fam_rows]))
            for col in metric_cols
        }

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print("\nInterpretation checks:")
    print("- full_rmse_K is the actual hard-left validation error on this subset.")
    print("- full_minus_zero_u_rmse_K near 0 means the forcing image is mostly ignored.")
    print("- full_rmse_K close to analytic_rmse_K means the learned correction adds little.")
    print("- no_q_rmse_K much worse than full_rmse_K means q_left forwarding matters.")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe whether hard-left ForcingCViT uses the forcing image."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to cvit_best.pt or seed dir.")
    parser.add_argument("--data-dir", default=None, help="Dataset dir; defaults to DATA_DIR.")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--num-sims", type=int, default=12)
    parser.add_argument("--sim-ids", default="", help="Comma-separated sim ids; overrides split/num-sims.")
    parser.add_argument("--query-batch", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    run_probe(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
