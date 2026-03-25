import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam

from data.dataset import create_dataloaders, load_sim_data, split_sim_ids
from src.operators.fno1d import FNO1d
from src.operators.losses import SpatiallyWeightedMSE, build_interface_mask, compute_interface_rel_l2
from src.operators.utils import resolve_device

from omegaconf import OmegaConf


def load_config(config_path: str | None = None) -> dict:
    # train.py -> src/operators -> project root is parents[2]
    project_root = Path(__file__).resolve().parents[2]
    path = Path(config_path) if config_path else (project_root / "conf" / "config.yaml")
    path = path.expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    import os
    os.environ.setdefault("PROJECT_ROOT", project_root.as_posix())

    cfg = OmegaConf.load(path)
    paths_cfg = OmegaConf.load(project_root / "conf" / "paths" / "default.yaml")
    cfg = OmegaConf.merge({"paths": paths_cfg}, cfg)

    # Remove Hydra-only sections that can't resolve outside Hydra
    for key in ("hydra", "defaults"):
        if key in cfg:
            del cfg[key]

    cfg = OmegaConf.to_container(cfg, resolve=True)

    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a mapping/dict, got: {type(cfg)}")

    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def _is_training_complete(run_path: Path) -> bool:
    """Training is complete if best checkpoint exists and no latest (sentinel) exists."""
    return (run_path / "fno1d_best.pt").exists() and not (run_path / "fno1d_latest.pt").exists()


def _load_completed_result(run_path: Path, seed: int) -> dict[str, float | int | str]:
    """Extract result from a previously completed run's best checkpoint."""
    best_path = run_path / "fno1d_best.pt"
    ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    return {"seed": seed, "best_val": float(ckpt["best_val"]), "best_path": str(best_path)}


def train_one_epoch(model, train_loader, optimizer, loss_fn, device, *, iface_mask=None, grad_clip=None) -> tuple[float, float, float]:
    model.train()
    training_loss = 0.0
    train_rel_l2 = 0.0
    train_iface_rel_l2 = 0.0

    # _T_stats is not used since error metrics are computed in z-score temp. source space
    for x_spatial, cond, y_batch, _T_stats in train_loader:
        x_spatial = x_spatial.to(device)
        cond = cond.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        y_pred = model(x_spatial, cond)
        loss = loss_fn(y_pred, y_batch)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        training_loss += loss.item()

        with torch.no_grad():
            batch_rel_l2 = (torch.mean((y_pred - y_batch) ** 2) / torch.mean(y_batch ** 2)) ** 0.5 * 100
            train_rel_l2 += batch_rel_l2.item()
            if iface_mask is not None:
                train_iface_rel_l2 += compute_interface_rel_l2(y_pred, y_batch, iface_mask)

    training_loss /= len(train_loader)
    train_rel_l2 /= len(train_loader)
    train_iface_rel_l2 /= len(train_loader)
    return training_loss, train_rel_l2, train_iface_rel_l2


def validate(model, val_loader, device, *, iface_mask=None) -> tuple[float, float]:
    """Return (val_rel_l2, val_iface_rel_l2) after validation."""
    with torch.no_grad():
        model.eval()
        val_loss = 0.0
        val_iface = 0.0

        for x_spatial, cond, y_batch, _T_stats in val_loader:
            x_spatial = x_spatial.to(device)
            cond = cond.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(x_spatial, cond)
            val_rel_l2 = (torch.mean((y_pred - y_batch) ** 2) / torch.mean(y_batch ** 2)) ** 0.5 * 100
            val_loss += val_rel_l2.item()
            if iface_mask is not None:
                val_iface += compute_interface_rel_l2(y_pred, y_batch, iface_mask)

        val_loss /= len(val_loader)
        val_iface /= len(val_loader)
        return val_loss, val_iface


def run_one_seed(config: dict, seed: int, run_dir: str | Path) -> dict[str, float | int | str]:
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)

    # --- Skip completed runs ---
    if _is_training_complete(run_path):
        result = _load_completed_result(run_path, seed)
        print(f"Seed {seed}: training already complete (best_val={result['best_val']:.4f}%), skipping.")
        return result

    # --- Check for interrupted run ---
    latest_path = run_path / "fno1d_latest.pt"
    resuming = latest_path.exists()

    set_seed(seed)

    trajectories, x_grid, t_grid = load_sim_data(
        sim_traj_path=config["data"]["trajectories.npy"],
        x_grid_path=config["data"]["x_grid_path"],
        t_grid_path=config["data"]["t_grid_path"],
    )
    sim_params = np.load(config["data"]["sim_params_path"], allow_pickle=True)

    train_ids, val_ids, test_ids = split_sim_ids(num_sims=trajectories.shape[0], train_frac=0.7, val_frac=0.15, seed=0)

    training_set, validation_set, _ = create_dataloaders(
        trajectories=trajectories, x_grid=x_grid, t_grid=t_grid,
        train_ids=train_ids, val_ids=val_ids, test_ids=test_ids,
        batch_size=config["training"]["batch_size"],
        sim_params=sim_params,
        n_snapshots=config["training"].get("n_snapshots", 15),
        n_snapshots_test=config["training"].get("n_snapshots_test", None),
    )

    device = resolve_device(config["training"].get("device", "auto"))
    print(f"Training on: {device}")

    model_cfg = config["model"]["parameters"]
    fno = FNO1d(
        modes=model_cfg["modes"],
        width=model_cfg["width"],
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["out_channels"],
        n_layers=model_cfg.get("n_layers", 4),
        cond_dim=model_cfg.get("cond_dim", 4),
        cond_hidden=model_cfg.get("cond_hidden", 256)
    )

    # --- Resume state ---
    start_epoch = 0
    best_val_loss = float("inf")
    bad_epochs = 0

    if resuming:
        ckpt = torch.load(latest_path, map_location=device, weights_only=False)
        fno.load_state_dict(ckpt["model_state"])
        best_val_loss = ckpt["best_val"]
        bad_epochs = ckpt.get("bad_epochs", 0)
        start_epoch = ckpt["epoch"] + 1

    fno.to(device)

    optimizer = Adam(
        fno.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    sched_cfg = config["training"]["scheduler"]
    sched_type = sched_cfg.get("type", "StepLR")
    if sched_type == "CosineWarmRestarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=sched_cfg.get("T_0", 100),
            T_mult=sched_cfg.get("T_mult", 2),
            eta_min=sched_cfg.get("eta_min", 1e-6),
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=sched_cfg["step_size"],
            gamma=sched_cfg["gamma"],
        )

    if resuming:
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        print(f"Resuming seed {seed} from epoch {start_epoch} (best_val={best_val_loss:.4f}%, bad_epochs={bad_epochs})")

    loss_cfg = config["training"].get("loss", {})
    loss_fn = SpatiallyWeightedMSE(
        x_grid=x_grid,
        interface_x=loss_cfg.get("interface_x", 0.5),
        interface_half_width=loss_cfg.get("interface_half_width", 0.05),
        interface_weight=loss_cfg.get("interface_weight", 1.0),
    ).to(device)

    iface_mask = build_interface_mask(
        x_grid, loss_cfg.get("interface_x", 0.5), loss_cfg.get("interface_half_width", 0.05),
    ).to(device)

    epochs = config["training"]["epochs"]
    validate_every = config["training"]["validate_every"]
    patience = config["training"]["patience"]

    best_path = run_path / "fno1d_best.pt"

    # --- CSV: truncate to start_epoch when resuming, overwrite when fresh ---
    csv_path = run_path / "train_metrics.csv"
    fieldnames = ["epoch", "train_loss", "train_rel_l2", "train_iface_rel_l2", "val_rel_l2", "val_iface_rel_l2", "lr", "is_best"]

    if resuming and csv_path.exists():
        # Keep only rows with epoch < start_epoch (discard stale rows beyond checkpoint)
        with csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            kept_rows = [row for row in reader if int(row["epoch"]) < start_epoch]
        csv_file = csv_path.open("w", newline="")
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()
        csv_writer.writerows(kept_rows)
    else:
        csv_file = csv_path.open("w", newline="")
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()

    grad_clip = config["training"].get("grad_clip", None)
    warmup_epochs = config["training"].get("curriculum_warmup", 0)

    for epoch in range(start_epoch, epochs):
        # Lead-time curriculum: progressively expose longer lead times
        if warmup_epochs > 0:
            frac = min(1.0, (epoch + 1) / warmup_epochs)
            training_set.dataset.set_curriculum_fraction(frac)

        train_loss, train_rel_l2, train_iface_rel_l2 = train_one_epoch(
            model=fno, train_loader=training_set, optimizer=optimizer,
            loss_fn=loss_fn, device=device, iface_mask=iface_mask,
            grad_clip=grad_clip,
        )
        scheduler.step()

        print(f"Epoch {epoch}: train_loss={train_loss:.6f}, train_rel_l2={train_rel_l2:.4f}%, iface_rel_l2={train_iface_rel_l2:.4f}%")
        is_best = 0
        val_loss = None
        val_iface_rel_l2 = None
        should_stop = False

        if (epoch % validate_every) == 0:
            val_loss, val_iface_rel_l2 = validate(model=fno, val_loader=validation_set, device=device, iface_mask=iface_mask)
            print(f"Validation loss for epoch {epoch}: rel_l2={val_loss:.4f}%, iface_rel_l2={val_iface_rel_l2:.4f}%")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                bad_epochs = 0
                is_best = 1
                torch.save(
                    {
                        "epoch": epoch,
                        "conf": config,
                        "seed": seed,
                        "model_state": fno.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "best_val": best_val_loss,
                        "bad_epochs": bad_epochs,
                    },
                    best_path,
                )
                print(f"Saved new best: {best_val_loss} -> {best_path}")
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print(f"Early stopping: no improvement for {patience} evaluations.")
                    should_stop = True

        # Save latest checkpoint (sentinel) every epoch for resume
        torch.save(
            {
                "epoch": epoch,
                "conf": config,
                "seed": seed,
                "model_state": fno.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_val": best_val_loss,
                "bad_epochs": bad_epochs,
            },
            latest_path,
        )

        lr = optimizer.param_groups[0]["lr"]
        csv_writer.writerow(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "train_rel_l2": float(train_rel_l2),
                "train_iface_rel_l2": float(train_iface_rel_l2),
                "val_rel_l2": "" if val_loss is None else float(val_loss),
                "val_iface_rel_l2": "" if val_iface_rel_l2 is None else float(val_iface_rel_l2),
                "lr": float(lr),
                "is_best": int(is_best),
            }
        )
        csv_file.flush()

        if should_stop:
            break

    csv_file.close()

    # Clean completion: remove sentinel
    if latest_path.exists():
        latest_path.unlink()

    return {"seed": seed, "best_val": float(best_val_loss), "best_path": str(best_path)}


def run_config_seeds(config: dict, base_run_dir: str | Path, seeds: list[int] | None = None) -> dict:
    run_root = Path(base_run_dir)
    run_root.mkdir(parents=True, exist_ok=True)

    seed_values = list(seeds if seeds is not None else config["training"]["seeds"])
    per_seed = []
    for seed in seed_values:
        seed_run_dir = run_root / f"seed{seed}"
        result = run_one_seed(config, int(seed), run_dir=seed_run_dir)
        per_seed.append(result)

    mean_best_val = float(sum(item["best_val"] for item in per_seed) / len(per_seed))
    return {"num_seeds": len(per_seed), "mean_best_val": mean_best_val, "per_seed": per_seed}


if __name__ == "__main__":
    config = load_config()
    base_run_dir = Path(config["training"]["run"]["run_dir"])
    summary = run_config_seeds(config, base_run_dir, seeds=config["training"]["seeds"])
    print(summary)
