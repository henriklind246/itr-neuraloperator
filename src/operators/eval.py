import torch
from data.dataset import (
    COND_STATIC_DIM,
    TEMPORAL_SAMPLES,
    T_EPS,
    compute_global_stats,
    create_dataloaders,
    load_sim_data,
    load_solver_dt,
    split_sim_ids,
)
from src.operators.fno2d import FNO2d
from src.operators.losses import build_interface_mask, compute_interface_rel_l2
from src.operators.utils import resolve_device
from pathlib import Path
import math
import json
from datetime import datetime

"""File loads checkpoint, runs test metrics in normalized AND physical space, outputs seed_report.json.

Per-seed result keys:
    test_rel_l2_norm        normalized-space relative L2 (%) — comparable to val_rel_l2
    test_rel_l2             physical-space (Kelvin) relative L2 (%)
    test_iface_rel_l2_norm  normalized-space interface-weighted relative L2 (%)
    test_iface_rel_l2       physical-space interface-weighted relative L2 (%)
"""

# -------- LOAD TEST SET ---------

def build_test_loader(config, mu_global=None, sigma_global=None):
    import numpy as np

    trajectories, x_grid, y_grid, t_grid = load_sim_data(
        sim_traj_path=config["data"]["trajectories.npy"],
        x_grid_path=config["data"]["x_grid_path"],
        y_grid_path=config["data"]["y_grid_path"],
        t_grid_path=config["data"]["t_grid_path"],
    )
    sim_params = np.load(config["data"]["sim_params_path"], allow_pickle=True)
    solver_dt = load_solver_dt(config["data"]["t_grid_path"])

    train_ids, val_ids, test_ids = split_sim_ids(num_sims=trajectories.shape[0], train_frac=0.7, val_frac=0.15, seed=0)

    # Use provided global stats or recompute from training set
    if mu_global is None or sigma_global is None:
        mu_global, sigma_global = compute_global_stats(trajectories, train_ids)

    _, _, testing_set = create_dataloaders(
        trajectories=trajectories, x_grid=x_grid, y_grid=y_grid, t_grid=t_grid,
        train_ids=train_ids, val_ids=val_ids, test_ids=test_ids,
        batch_size=config["training"]["batch_size"],
        sim_params=sim_params,
        mu_global=mu_global,
        sigma_global=sigma_global,
        n_snapshots=10,
        n_snapshots_test=config.get("training", {}).get("n_snapshots_test", 40),
        dt=solver_dt,
        temporal_samples=config["model"]["parameters"].get("temporal_samples", TEMPORAL_SAMPLES),
    )

    return testing_set, x_grid, y_grid

# -------- EVAL MODEL ON TEST SET  ---------

def evaluate(model, test_loader, device, iface_mask=None):
    """Return a dict of test metrics in both normalized and physical space.

    Keys:
        rel_l2_norm:        relative L2 (%) on normalized outputs — directly
                            comparable to train/val_rel_l2 in train_metrics.csv.
        rel_l2_phys:        relative L2 (%) after denormalizing to Kelvin —
                            small because the ~300 K baseline inflates the
                            denominator. Useful for "% of absolute T" intuition.
        iface_rel_l2_norm:  interface-weighted relative L2 (%) on normalized outputs.
        iface_rel_l2_phys:  interface-weighted relative L2 (%) in Kelvin.
    """
    with torch.no_grad():
        model.eval()
        rel_l2_norm = 0.0
        rel_l2_phys = 0.0
        iface_rel_l2_norm = 0.0
        iface_rel_l2_phys = 0.0

        for x_spatial, cond_static, forcing_seq, y_batch, T_stats in test_loader:
            x_spatial = x_spatial.to(device)
            cond_static = cond_static.to(device)
            forcing_seq = forcing_seq.to(device)
            y_batch = y_batch.to(device)
            T_stats = T_stats.to(device)

            y_pred = model(x_spatial, cond_static, forcing_seq)

            # Normalized-space metric (same convention as train/val_rel_l2)
            batch_rel_l2_norm = (torch.mean((y_pred - y_batch) ** 2) / torch.mean(y_batch ** 2)) ** 0.5 * 100
            rel_l2_norm += batch_rel_l2_norm.item()

            # Physical-space metric (denormalized to Kelvin)
            mu_s = T_stats[:, 0]
            sigma_s = T_stats[:, 1]
            y_pred_phys = y_pred * (sigma_s[:, None, None, None] + T_EPS) + mu_s[:, None, None, None]
            y_true_phys = y_batch * (sigma_s[:, None, None, None] + T_EPS) + mu_s[:, None, None, None]
            batch_rel_l2_phys = (torch.mean((y_pred_phys - y_true_phys) ** 2) / torch.mean(y_true_phys ** 2)) ** 0.5 * 100
            rel_l2_phys += batch_rel_l2_phys.item()

            if iface_mask is not None:
                iface_rel_l2_norm += compute_interface_rel_l2(y_pred, y_batch, iface_mask)
                iface_rel_l2_phys += compute_interface_rel_l2(y_pred_phys, y_true_phys, iface_mask)

        n_batches = len(test_loader)
        rel_l2_norm /= n_batches
        rel_l2_phys /= n_batches
        iface_rel_l2_norm /= n_batches
        iface_rel_l2_phys /= n_batches

    return {
        "rel_l2_norm": rel_l2_norm,
        "rel_l2_phys": rel_l2_phys,
        "iface_rel_l2_norm": iface_rel_l2_norm,
        "iface_rel_l2_phys": iface_rel_l2_phys,
    }


# --------- EVAL ALL SEEDS IN RUNS ---------

def eval_all_seeds(run_root: str):
    run_root = Path(run_root)
    results = []

    print("Testing started.")

    for seed_dir in sorted(run_root.glob("seed*")):
        ckpt_path = seed_dir / "fno2d_best.pt"
        if not ckpt_path.exists():
            continue

        ckpt = torch.load(ckpt_path, map_location="cpu")
        config = ckpt['conf']
        device = resolve_device(config.get("training", {}).get("device", "auto"))

        test_loader, x_grid, y_grid = build_test_loader(
            config,
            mu_global=ckpt.get("mu_global"),
            sigma_global=ckpt.get("sigma_global"),
        )

        loss_cfg = config.get("training", {}).get("loss", {})
        iface_mask = build_interface_mask(
            x_grid, y_grid,
            loss_cfg.get("interface_x", 0.5), loss_cfg.get("interface_half_width", 0.05),
        ).to(device)

        model_cfg = config['model']['parameters']
        fno = FNO2d(
            modes1=model_cfg["modes1"],
            modes2=model_cfg["modes2"],
            width=model_cfg["width"],
            in_channels=model_cfg.get("in_channels", 4),
            out_channels=model_cfg.get("out_channels", 1),
            n_layers=model_cfg.get("n_layers", 4),
            cond_static_dim=model_cfg.get("cond_static_dim", COND_STATIC_DIM),
            cond_hidden=model_cfg.get("cond_hidden", 256),
            temporal_token_dim=model_cfg.get("temporal_token_dim", 5),
            temporal_hidden=model_cfg.get("temporal_hidden", 128),
            forcing_embed_dim=model_cfg.get("forcing_embed_dim", 64),
            forcing_spatial_dim=model_cfg.get("forcing_spatial_dim", 16),
        )
        fno.load_state_dict(ckpt['model_state'])
        fno.to(device)

        metrics = evaluate(model=fno, test_loader=test_loader, device=device, iface_mask=iface_mask)

        results.append(
            {
                "seed": ckpt.get("seed", seed_dir.name),
                "best_epoch": ckpt["epoch"],
                "best_val": float(ckpt["best_val"]),
                "test_rel_l2_norm": float(metrics["rel_l2_norm"]),
                "test_rel_l2": float(metrics["rel_l2_phys"]),
                "test_iface_rel_l2_norm": float(metrics["iface_rel_l2_norm"]),
                "test_iface_rel_l2": float(metrics["iface_rel_l2_phys"]),
                "ckpt": str(ckpt_path),
            }
        )

    # sort by normalized test performance (apples-to-apples with val_rel_l2)
    results.sort(key=lambda r: r["test_rel_l2_norm"])
    return results


# ---------- COMPUTE MEAN + STD FOR TEST ERROR ACROSS SEEDS ----------

def mean_std(values: list[float]) -> tuple[float, float]:
    """Sample std for small-n reporting; returns (mean, std)."""
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    mu = sum(values) / n
    if n == 1:
        return mu, 0.0
    # variance formula for a sample
    var = sum((x - mu) ** 2 for x in values) / (n - 1)
    return mu, math.sqrt(var)

def print_seed_report(results: list[dict]) -> dict:
    best_vals = [r["best_val"] for r in results]
    test_rel_l2_norm = [r["test_rel_l2_norm"] for r in results]
    test_rel_l2 = [r["test_rel_l2"] for r in results]
    test_iface_norm = [r["test_iface_rel_l2_norm"] for r in results]
    test_iface = [r["test_iface_rel_l2"] for r in results]

    val_mu, val_std = mean_std(best_vals)
    norm_mu, norm_std = mean_std(test_rel_l2_norm)
    test_mu, test_std = mean_std(test_rel_l2)
    iface_norm_mu, iface_norm_std = mean_std(test_iface_norm)
    iface_mu, iface_std = mean_std(test_iface)

    print("\n===== Seed Report =====")
    print(f"Number of seeds: {len(results)}")
    print(f"best_val_loss            mean, std: ({val_mu}, {val_std})")
    print(f"test_rel_l2_norm         mean, std: ({norm_mu}, {norm_std})   <- comparable to val_rel_l2")
    print(f"test_rel_l2 (physical)   mean, std: ({test_mu}, {test_std})")
    print(f"test_iface_rel_l2_norm   mean, std: ({iface_norm_mu}, {iface_norm_std})")
    print(f"test_iface_rel_l2 (phys) mean, std: ({iface_mu}, {iface_std})")

    best = min(results, key=lambda r: r["test_rel_l2_norm"])
    print(
        f"Best by lowest normalized test error: seed={best['seed']} "
        f"test_rel_l2_norm={best['test_rel_l2_norm']} "
        f"test_rel_l2={best['test_rel_l2']} "
        f"test_iface_rel_l2_norm={best['test_iface_rel_l2_norm']} "
        f"test_iface_rel_l2={best['test_iface_rel_l2']}"
    )

    return {
        "num_seeds": len(results),
        "best_val_loss_mean": val_mu,
        "best_val_loss_std": val_std,
        "test_rel_l2_norm_mean": norm_mu,
        "test_rel_l2_norm_std": norm_std,
        "test_rel_l2_mean": test_mu,
        "test_rel_l2_std": test_std,
        "test_iface_rel_l2_norm_mean": iface_norm_mu,
        "test_iface_rel_l2_norm_std": iface_norm_std,
        "test_iface_rel_l2_mean": iface_mu,
        "test_iface_rel_l2_std": iface_std,
    }

def save_report(run_root: str, results: list[dict], summary: dict) -> None:
    out = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_root": str(run_root),
        "summary": summary,
        "per_seed": results,
    }
    out_path = Path(run_root) / "seed_report.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved report -> {out_path}")
