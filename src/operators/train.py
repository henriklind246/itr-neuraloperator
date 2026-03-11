import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam

from data.dataset import create_dataloaders, load_sim_data, split_sim_ids
from src.operators.fno2d import FNO2d
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
    os.environ.setdefault("PROJECT_ROOT", str(project_root))

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


def train_one_epoch(model, train_loader, optimizer, loss_fn, device) -> float:
    model.train()
    training_loss = 0.0

    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = loss_fn(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        training_loss += loss.item()

    training_loss /= len(train_loader)
    return training_loss


def validate(model, val_loader, device) -> float:
    """Output val_rel_l2 after validation."""
    with torch.no_grad():
        model.eval()
        val_loss = 0.0

        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_pred = model(x_batch)
            val_rel_l2 = (torch.mean((y_pred - y_batch) ** 2) / torch.mean(y_batch ** 2)) ** 0.5 * 100
            val_loss += val_rel_l2.item()

        val_loss /= len(val_loader)
        return val_loss


def run_one_seed(config: dict, seed: int, run_dir: str | Path) -> dict[str, float | int | str]:
    set_seed(seed)

    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)

    csv_path = run_path / "train_metrics.csv"
    csv_file = csv_path.open("w", newline="")
    csv_writer = csv.DictWriter(
        csv_file,
        fieldnames=["epoch", "train_loss", "val_rel_l2", "lr", "is_best"],
    )
    csv_writer.writeheader()


    trajectories, x_grid, t_grid = load_sim_data(sim_traj_path=config["data"]["trajectories.npy"], x_grid_path=config["data"]["x_grid_path"], t_grid_path=config["data"]["t_grid_path"])

    train_ids, val_ids, test_ids = split_sim_ids(num_sims=trajectories.shape[0], train_frac=0.7, val_frac=0.15, seed=0)

    training_set, validation_set, _ = create_dataloaders(trajectories=trajectories, x_grid=x_grid, t_grid=t_grid, train_ids=train_ids, val_ids=val_ids, test_ids=test_ids, batch_size=config["training"]["batch_size"], k=config["training"]["k"], H=config["training"]["H"])

    device = resolve_device(config["training"].get("device", "auto"))

    fno = FNO2d(config["model"]["parameters"]["modes1"], config["model"]["parameters"]["modes2"], config["model"]["parameters"]["width"])
    fno.to(device)

    optimizer = Adam(
        fno.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer,
        step_size=config["training"]["scheduler"]["step_size"],
        gamma=config["training"]["scheduler"]["gamma"],
    )
    loss_fn = torch.nn.MSELoss()

    epochs = config["training"]["epochs"]
    validate_every = config["training"]["validate_every"]
    patience = config["training"]["patience"]

    best_val_loss = float("inf")
    best_path = run_path / "fno2d_best.pt"
    bad_epochs = 0

    for epoch in range(epochs):
        train_loss = train_one_epoch(model=fno, train_loader=training_set, optimizer=optimizer, loss_fn=loss_fn, device=device)
        scheduler.step()

        print(f"Training loss for epoch {epoch} is {train_loss}")
        is_best = 0
        val_loss = None
        should_stop = False

        if (epoch % validate_every) == 0:
            val_loss = validate(model=fno, val_loader=validation_set, device=device)
            print(f"Validation loss for epoch {epoch} is {val_loss}")

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
                    },
                    best_path,
                )
                print(f"Saved new best: {best_val_loss} -> {best_path}")
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print(f"Early stopping: no improvement for {patience} evaluations.")
                    should_stop = True

        lr = optimizer.param_groups[0]["lr"]
        csv_writer.writerow(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_rel_l2": "" if val_loss is None else float(val_loss),
                "lr": float(lr),
                "is_best": int(is_best)
            }
        )
        csv_file.flush()

        if should_stop:
            break

    csv_file.close()
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
