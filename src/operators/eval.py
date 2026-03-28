import torch
from data.dataset import compute_global_stats, load_sim_data, split_sim_ids, create_dataloaders, T_EPS
from src.operators.fno1d import FNO1d
from src.operators.losses import build_interface_mask, compute_interface_rel_l2
from src.operators.utils import resolve_device
from pathlib import Path
import math
import json
from datetime import datetime

"""File loads checkpoint, runs test metrics test_rel_l2 and test_iface_rel_l2, outputs seed_report.json"""

# -------- LOAD TEST SET ---------

def build_test_loader(config, mu_global=None, sigma_global=None):
    import numpy as np

    trajectories, x_grid, t_grid = load_sim_data(
        sim_traj_path=config["data"]["trajectories.npy"],
        x_grid_path=config["data"]["x_grid_path"],
        t_grid_path=config["data"]["t_grid_path"],
    )
    sim_params = np.load(config["data"]["sim_params_path"], allow_pickle=True)

    train_ids, val_ids, test_ids = split_sim_ids(num_sims=trajectories.shape[0], train_frac=0.7, val_frac=0.15, seed=0)

    # Use provided global stats or recompute from training set
    if mu_global is None or sigma_global is None:
        mu_global, sigma_global = compute_global_stats(trajectories, train_ids)

    _, _, testing_set = create_dataloaders(
        trajectories=trajectories, x_grid=x_grid, t_grid=t_grid,
        train_ids=train_ids, val_ids=val_ids, test_ids=test_ids,
        batch_size=config["training"]["batch_size"],
        sim_params=sim_params,
        mu_global=mu_global,
        sigma_global=sigma_global,
        n_snapshots=10,
        n_snapshots_test=40
    )

    return testing_set, x_grid

# -------- EVAL MODEL ON TEST SET  ---------

def evaluate(model, test_loader, device, iface_mask=None):
    """Return (test_rel_l2, test_iface_rel_l2) after evaluation on test set.

    Metrics are computed in physical (denormalized) space using T_stats
    from the dataset.
    """
    with torch.no_grad():
        model.eval()
        test_loss = 0.0
        test_iface = 0.0

        for x_spatial, cond, y_batch, T_stats in test_loader:
            x_spatial = x_spatial.to(device)
            cond = cond.to(device)
            y_batch = y_batch.to(device)
            T_stats = T_stats.to(device)

            y_pred = model(x_spatial, cond)

            # Denormalize to physical space
            mu_s = T_stats[:, 0]    # (B,)
            sigma_s = T_stats[:, 1]  # (B,)
            y_pred_phys = y_pred * (sigma_s[:, None, None] + T_EPS) + mu_s[:, None, None]
            y_true_phys = y_batch * (sigma_s[:, None, None] + T_EPS) + mu_s[:, None, None]

            test_rel_l2 = (torch.mean((y_pred_phys - y_true_phys) ** 2) / torch.mean(y_true_phys ** 2)) ** 0.5 * 100

            test_loss += test_rel_l2.item()
            if iface_mask is not None:
                test_iface += compute_interface_rel_l2(y_pred_phys, y_true_phys, iface_mask)

        # average test loss across all batches
        test_loss /= len(test_loader)
        test_iface /= len(test_loader)

    return test_loss, test_iface


# --------- EVAL ALL SEEDS IN RUNS ---------

def eval_all_seeds(run_root: str):
    run_root = Path(run_root)
    results = []

    print("Testing started.")

    for seed_dir in sorted(run_root.glob("seed*")):
        ckpt_path = seed_dir / "fno1d_best.pt"
        if not ckpt_path.exists():
            continue

        ckpt = torch.load(ckpt_path, map_location="cpu")
        config = ckpt['conf']
        device = resolve_device(config.get("training", {}).get("device", "auto"))

        test_loader, x_grid = build_test_loader(
            config,
            mu_global=ckpt.get("mu_global"),
            sigma_global=ckpt.get("sigma_global"),
        )

        loss_cfg = config.get("training", {}).get("loss", {})
        iface_mask = build_interface_mask(
            x_grid, loss_cfg.get("interface_x", 0.5), loss_cfg.get("interface_half_width", 0.05),
        ).to(device)

        model_cfg = config['model']['parameters']
        fno = FNO1d(
            modes=model_cfg["modes"],
            width=model_cfg["width"],
            in_channels=model_cfg.get("in_channels", 2),
            out_channels=model_cfg.get("out_channels", 1),
            n_layers=model_cfg.get("n_layers", 4),
            cond_dim=model_cfg.get("cond_dim", 5),
            cond_hidden=model_cfg.get("cond_hidden", 256),
        )
        fno.load_state_dict(ckpt['model_state'])
        fno.to(device)

        test_rel_l2, test_iface_rel_l2 = evaluate(model=fno, test_loader=test_loader, device=device, iface_mask=iface_mask)

        results.append(
            {
                "seed": ckpt.get("seed", seed_dir.name),
                "best_epoch": ckpt["epoch"],
                "best_val": float(ckpt["best_val"]),
                "test_rel_l2": float(test_rel_l2),
                "test_iface_rel_l2": float(test_iface_rel_l2),
                "ckpt": str(ckpt_path)
            }
        )

    # sort by test performance (done by sort() which goes from smallest -> largest)
    results.sort(key=lambda r: r['test_rel_l2'])
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
    best_vals = [r['best_val'] for r in results]
    test_rel_l2 = [r['test_rel_l2'] for r in results]
    test_iface = [r['test_iface_rel_l2'] for r in results]

    # compute the mean and std for validation, test, and interface test
    val_mu, val_std = mean_std(best_vals)
    test_mu, test_std = mean_std(test_rel_l2)
    iface_mu, iface_std = mean_std(test_iface)

    # print seed report
    print("\n===== Seed Report =====")
    print(f"Number of seeds: {len(results)}")
    print(f"best_val_loss mean & standard dist. ({val_mu}, {val_std})")
    print(f"test_rel_l2 mean & standard dist. ({test_mu}, {test_std})")
    print(f"test_iface_rel_l2 mean & standard dist. ({iface_mu}, {iface_std})")

    # also print best seed by lowest test error
    best = min(results, key=lambda r: r['test_rel_l2'])
    print(f"Best by lowest test error: seed={best['seed']} test_rel_l2={best['test_rel_l2']} test_iface_rel_l2={best['test_iface_rel_l2']}")

    # return dict
    return {
        "num_seeds": len(results),
        "best_val_loss_mean": val_mu,
        "best_val_loss_std": val_std,
        "test_rel_l2_mean": test_mu,
        "test_rel_l2_std": test_std,
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

if __name__ == '__main__':
    project_root = Path(__file__).resolve().parents[2]
    run_root = str(project_root / "runs" / "experiment0" / "config0")
    # get results from evaluating model with a conf on all seeds
    results = eval_all_seeds(run_root)

    for r in results:
        print(f"seed={r['seed']} best epoch: {r['best_epoch']} best validation loss: {r['best_val']} test_rel_l2: {r['test_rel_l2']} test_iface_rel_l2: {r['test_iface_rel_l2']}.")

    # print summary evaluation of seeds
    summary = print_seed_report(results)

    # save to disk
    save_report(run_root=run_root, results=results, summary=summary)
