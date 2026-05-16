import csv
import json
import math
import os
import random
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam, AdamW

from data.dataset import (
    COND_STATIC_DIM,
    TEMPORAL_SAMPLES,
    SnapshotPairDataset,
    compute_global_stats,
    create_dataloaders,
    load_sim_data,
    load_solver_dt,
    split_sim_ids,
)
from src.operators.distributed import DistInfo, get_dist_info
from src.operators.fno2d import FNO2d
from src.operators.losses import SpatiallyWeightedMSE, build_interface_mask
from src.operators.utils import resolve_device

from omegaconf import OmegaConf


VAL_PAIR_FIELDNAMES = [
    "epoch",
    "sim_id",
    "temporal_family",
    "spatial_family",
    "t_s",
    "t_bar",
    "R_c",
    "rel_l2",
    "iface_rel_l2",
]


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
    return (run_path / "fno2d_best.pt").exists() and not (run_path / "fno2d_latest.pt").exists()


def _load_completed_result(run_path: Path, seed: int) -> dict[str, float | int | str]:
    """Extract result from a previously completed run's best checkpoint."""
    best_path = run_path / "fno2d_best.pt"
    ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    return {"seed": seed, "best_val": float(ckpt["best_val"]), "best_path": str(best_path)}


def _optimizer_name(config: dict) -> str:
    return str(config.get("training", {}).get("optimizer", "Adam"))


def _scheduler_type(config: dict) -> str:
    return str(config.get("training", {}).get("scheduler", {}).get("type", "StepLR"))


def _validate_resume_compatibility(checkpoint_conf: dict, current_conf: dict) -> None:
    ckpt_optimizer = _optimizer_name(checkpoint_conf)
    curr_optimizer = _optimizer_name(current_conf)
    ckpt_scheduler = _scheduler_type(checkpoint_conf)
    curr_scheduler = _scheduler_type(current_conf)

    if ckpt_optimizer != curr_optimizer or ckpt_scheduler != curr_scheduler:
        raise ValueError(
            "Incompatible resume state: checkpoint uses "
            f"optimizer={ckpt_optimizer}, scheduler={ckpt_scheduler}, "
            f"but current config uses optimizer={curr_optimizer}, scheduler={curr_scheduler}. "
            "Start from a fresh run directory or remove fno2d_latest.pt."
        )


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        return ""
    return result.stdout.rstrip("\n")


def _capture_git_provenance(run_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    head = _git(repo_root, "rev-parse", "HEAD") or "unknown"
    short = _git(repo_root, "rev-parse", "--short", "HEAD") or "unknown"
    status = _git(repo_root, "status", "--short")
    (run_path / "git_commit.txt").write_text(f"{head}\n{short}\n")
    (run_path / "git_status.txt").write_text(status + ("\n" if status else ""))


def _capture_job_context(run_path: Path) -> None:
    slurm_keys = ("SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_NODELIST", "SLURMD_NODENAME")
    slurm_env = {k: os.environ[k] for k in slurm_keys if k in os.environ}
    lines = []
    if slurm_env:
        for k, v in slurm_env.items():
            lines.append(f"{k}={v}")
    else:
        lines.append("slurm=local")
    lines.append(f"hostname={socket.gethostname()}")
    lines.append(f"argv={' '.join(sys.argv)}")
    lines.append(f"start_utc={datetime.now(timezone.utc).isoformat()}")
    (run_path / "slurm_job.txt").write_text("\n".join(lines) + "\n")


def _dump_resolved_config(run_path: Path, config: dict) -> None:
    OmegaConf.save(OmegaConf.create(config), run_path / "config_used.yaml")


def _summarize_metrics_csv(csv_path: Path) -> dict:
    with csv_path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))

    def _f(row, key):
        v = row.get(key, "")
        return float(v) if v not in ("", None) else None

    def _row(row):
        return {
            "epoch": int(row["epoch"]),
            "train_rel_l2": _f(row, "train_rel_l2"),
            "train_iface_rel_l2": _f(row, "train_iface_rel_l2"),
            "val_rel_l2": _f(row, "val_rel_l2"),
            "val_iface_rel_l2": _f(row, "val_iface_rel_l2"),
            "lr": _f(row, "lr"),
        }

    best_rows = [r for r in rows if r.get("is_best") == "1"]
    best_row = _row(best_rows[-1]) if best_rows else _row(rows[-1])
    final_row = _row(rows[-1])
    return {
        "best_row": best_row,
        "final_row": final_row,
        "total_epochs": len(rows),
    }


def _write_final_metrics(
    run_path: Path,
    *,
    seed: int,
    config: dict,
    wall_time_s: float,
    status: str,
) -> None:
    csv_path = run_path / "train_metrics.csv"
    if not csv_path.exists():
        return
    summary = _summarize_metrics_csv(csv_path)
    best = summary["best_row"]
    final = summary["final_row"]

    git_short = "unknown"
    commit_file = run_path / "git_commit.txt"
    if commit_file.exists():
        lines = commit_file.read_text().strip().splitlines()
        if len(lines) >= 2:
            git_short = lines[1]

    model_cfg = config.get("model", {}).get("parameters", {})
    training_cfg = config.get("training", {})
    payload = {
        "status": status,
        "seed": int(seed),
        "experiment_name": str(config.get("experiment", {}).get("name", "")),
        "config_id": config.get("config_id"),
        "best": {
            "epoch": best["epoch"],
            "val_rel_l2": best["val_rel_l2"],
            "val_iface_rel_l2": best["val_iface_rel_l2"],
            "train_rel_l2": best["train_rel_l2"],
            "train_iface_rel_l2": best["train_iface_rel_l2"],
        },
        "final": {
            "epoch": final["epoch"],
            "train_rel_l2": final["train_rel_l2"],
            "train_iface_rel_l2": final["train_iface_rel_l2"],
            "lr": final["lr"],
        },
        "epochs_trained": summary["total_epochs"],
        "epochs_configured": int(training_cfg.get("epochs", 0)),
        "early_stopped": status == "early_stopped",
        "wall_time_seconds": round(float(wall_time_s), 2),
        "wall_time_minutes": round(float(wall_time_s) / 60.0, 2),
        "git_commit_short": git_short,
        "model_params": {
            "modes1": model_cfg.get("modes1"),
            "modes2": model_cfg.get("modes2"),
            "width": model_cfg.get("width"),
            "n_layers": model_cfg.get("n_layers"),
        },
        "training_params": {
            "batch_size": training_cfg.get("batch_size"),
            "learning_rate": training_cfg.get("learning_rate"),
            "n_snapshots": training_cfg.get("n_snapshots"),
        },
    }
    (run_path / "final_metrics.json").write_text(json.dumps(payload, indent=2) + "\n")


def resolve_num_workers(config: dict, world_size: int) -> int:
    """Decide DataLoader worker count. Config takes precedence; otherwise
    derive from `SLURM_CPUS_PER_TASK` divided by `world_size`. Default: 0."""
    cfg_workers = config.get("training", {}).get("num_workers", None)
    if cfg_workers is not None:
        return int(cfg_workers)

    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus is not None:
        return max(0, int(slurm_cpus) // max(1, world_size))

    return 0


def build_optimizer(config: dict, params) -> torch.optim.Optimizer:
    training_cfg = config["training"]
    optimizer_name = training_cfg.get("optimizer", "Adam")
    optimizer_map = {
        "Adam": Adam,
        "AdamW": AdamW,
    }
    optimizer_cls = optimizer_map.get(optimizer_name)
    if optimizer_cls is None:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    return optimizer_cls(
        params,
        lr=training_cfg["learning_rate"],
        weight_decay=training_cfg["weight_decay"],
    )


def _rigno_three_phase_lr_for_epoch(
    epoch_idx: int,
    warmup_epochs: int,
    cosine_epochs: int,
    exp_epochs: int,
    init_lr: float,
    peak_lr: float,
    cosine_floor_lr: float,
    final_lr: float,
) -> float:
    if epoch_idx < 0:
        raise ValueError(f"epoch_idx must be >= 0, got {epoch_idx}")

    if warmup_epochs > 0 and epoch_idx < warmup_epochs:
        if warmup_epochs == 1:
            return peak_lr
        progress = epoch_idx / (warmup_epochs - 1)
        return init_lr + (peak_lr - init_lr) * progress

    epoch_idx -= warmup_epochs
    if cosine_epochs > 0 and epoch_idx < cosine_epochs:
        if cosine_epochs == 1:
            return cosine_floor_lr
        progress = epoch_idx / (cosine_epochs - 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return cosine_floor_lr + (peak_lr - cosine_floor_lr) * cosine

    epoch_idx -= cosine_epochs
    if exp_epochs > 0 and epoch_idx < exp_epochs:
        if exp_epochs == 1:
            return final_lr
        progress = epoch_idx / (exp_epochs - 1)
        decay_ratio = final_lr / cosine_floor_lr
        return cosine_floor_lr * (decay_ratio ** progress)

    return final_lr


def _resolve_rigno_phase_lengths(training_cfg: dict, sched_cfg: dict) -> tuple[int, int, int]:
    total_epochs = int(training_cfg["epochs"])
    fraction_keys = ("warmup_fraction", "cosine_fraction", "exp_fraction")
    epoch_keys = ("warmup_epochs", "cosine_epochs", "exp_epochs")

    if any(key in sched_cfg for key in fraction_keys):
        if not all(key in sched_cfg for key in fraction_keys):
            raise ValueError(
                "RIGNOThreePhase fraction schedule requires warmup_fraction, cosine_fraction, and exp_fraction."
            )

        warmup_fraction = float(sched_cfg["warmup_fraction"])
        cosine_fraction = float(sched_cfg["cosine_fraction"])
        exp_fraction = float(sched_cfg["exp_fraction"])
        if min(warmup_fraction, cosine_fraction, exp_fraction) <= 0:
            raise ValueError("RIGNOThreePhase fractions must all be positive.")
        if not math.isclose(
            warmup_fraction + cosine_fraction + exp_fraction,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("RIGNOThreePhase fractions must sum to 1.0.")

        warmup_epochs = max(1, int(round(total_epochs * warmup_fraction)))
        exp_epochs = max(1, int(round(total_epochs * exp_fraction)))
        cosine_epochs = total_epochs - warmup_epochs - exp_epochs
        if cosine_epochs <= 0:
            raise ValueError("RIGNOThreePhase derived cosine_epochs must be positive.")
        return warmup_epochs, cosine_epochs, exp_epochs

    if any(key in sched_cfg for key in epoch_keys):
        if not all(key in sched_cfg for key in epoch_keys):
            raise ValueError(
                "RIGNOThreePhase explicit phase lengths require warmup_epochs, cosine_epochs, and exp_epochs."
            )

        warmup_epochs = int(sched_cfg["warmup_epochs"])
        cosine_epochs = int(sched_cfg["cosine_epochs"])
        exp_epochs = int(sched_cfg["exp_epochs"])
        if warmup_epochs <= 0 or cosine_epochs <= 0 or exp_epochs <= 0:
            raise ValueError("RIGNOThreePhase requires positive warmup_epochs, cosine_epochs, and exp_epochs.")
        if warmup_epochs + cosine_epochs + exp_epochs != total_epochs:
            raise ValueError("RIGNOThreePhase phase lengths must sum to training.epochs.")
        return warmup_epochs, cosine_epochs, exp_epochs

    raise ValueError(
        "RIGNOThreePhase requires either warmup_fraction/cosine_fraction/exp_fraction "
        "or warmup_epochs/cosine_epochs/exp_epochs."
    )


def _resolve_rigno_lr_values(training_cfg: dict, sched_cfg: dict) -> tuple[float, float, float, float]:
    peak_lr = float(training_cfg["learning_rate"])
    if "peak_lr" in sched_cfg and not math.isclose(
        float(sched_cfg["peak_lr"]),
        peak_lr,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("scheduler.peak_lr must match training.learning_rate when provided explicitly.")

    def _resolve_lr_value(abs_key: str, ratio_key: str) -> float:
        if ratio_key in sched_cfg:
            ratio = float(sched_cfg[ratio_key])
            if ratio <= 0:
                raise ValueError(f"RIGNOThreePhase requires positive {ratio_key}.")
            return peak_lr * ratio
        if abs_key not in sched_cfg:
            raise ValueError(f"RIGNOThreePhase requires {abs_key} or {ratio_key}.")
        return float(sched_cfg[abs_key])

    init_lr = _resolve_lr_value("init_lr", "init_lr_ratio")
    cosine_floor_lr = _resolve_lr_value("cosine_floor_lr", "cosine_floor_lr_ratio")
    final_lr = _resolve_lr_value("final_lr", "final_lr_ratio")

    if min(init_lr, peak_lr, cosine_floor_lr, final_lr) <= 0:
        raise ValueError("RIGNOThreePhase LR values must all be positive.")
    if peak_lr < init_lr:
        raise ValueError("RIGNOThreePhase requires peak_lr >= init_lr.")
    if peak_lr < cosine_floor_lr or cosine_floor_lr < final_lr:
        raise ValueError("RIGNOThreePhase requires peak_lr >= cosine_floor_lr >= final_lr.")
    return peak_lr, init_lr, cosine_floor_lr, final_lr


class RIGNOThreePhaseScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        cosine_epochs: int,
        exp_epochs: int,
        init_lr: float,
        peak_lr: float,
        cosine_floor_lr: float,
        final_lr: float,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = int(warmup_epochs)
        self.cosine_epochs = int(cosine_epochs)
        self.exp_epochs = int(exp_epochs)
        self.init_lr = float(init_lr)
        self.peak_lr = float(peak_lr)
        self.cosine_floor_lr = float(cosine_floor_lr)
        self.final_lr = float(final_lr)
        self.next_epoch_index = 0
        self._set_lr(self._lr_for_epoch(0))

    def _lr_for_epoch(self, epoch_idx: int) -> float:
        return _rigno_three_phase_lr_for_epoch(
            epoch_idx,
            warmup_epochs=self.warmup_epochs,
            cosine_epochs=self.cosine_epochs,
            exp_epochs=self.exp_epochs,
            init_lr=self.init_lr,
            peak_lr=self.peak_lr,
            cosine_floor_lr=self.cosine_floor_lr,
            final_lr=self.final_lr,
        )

    def _set_lr(self, lr: float) -> None:
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def step(self) -> None:
        self.next_epoch_index += 1
        self._set_lr(self._lr_for_epoch(self.next_epoch_index))

    def state_dict(self) -> dict[str, int]:
        return {"next_epoch_index": self.next_epoch_index}

    def load_state_dict(self, state_dict: dict) -> None:
        self.next_epoch_index = int(state_dict.get("next_epoch_index", 0))
        self._set_lr(self._lr_for_epoch(self.next_epoch_index))


def build_scheduler(config: dict, optimizer: torch.optim.Optimizer):
    training_cfg = config["training"]
    sched_cfg = training_cfg.get("scheduler", {})
    sched_type = sched_cfg.get("type", "StepLR")

    if sched_type == "CosineWarmRestarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=sched_cfg.get("T_0", 100),
            T_mult=sched_cfg.get("T_mult", 2),
            eta_min=sched_cfg.get("eta_min", 1e-6),
        )

    if sched_type == "RIGNOThreePhase":
        warmup_epochs, cosine_epochs, exp_epochs = _resolve_rigno_phase_lengths(training_cfg, sched_cfg)
        peak_lr, init_lr, cosine_floor_lr, final_lr = _resolve_rigno_lr_values(training_cfg, sched_cfg)

        return RIGNOThreePhaseScheduler(
            optimizer,
            warmup_epochs=warmup_epochs,
            cosine_epochs=cosine_epochs,
            exp_epochs=exp_epochs,
            init_lr=init_lr,
            peak_lr=peak_lr,
            cosine_floor_lr=cosine_floor_lr,
            final_lr=final_lr,
        )

    if sched_type == "StepLR":
        return torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=sched_cfg["step_size"],
            gamma=sched_cfg["gamma"],
        )

    raise ValueError(f"Unsupported scheduler type: {sched_type}")


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    loss_fn,
    device,
    iface_mask=None,
    grad_clip=None,
    dist_info: DistInfo | None = None,
) -> tuple[float, float, float]:
    """Train one epoch. Under DDP, reduces sums (loss, MSE, ||y||^2, iface MSE,
    ||y_iface||^2, sample counts) across ranks and computes true global metrics
    from those sums — not an average of per-rank ratios."""
    if dist_info is None:
        dist_info = get_dist_info()

    model.train()

    loss_sum = 0.0
    mse_sum = 0.0
    target_sq_sum = 0.0
    iface_mse_sum = 0.0
    iface_target_sq_sum = 0.0
    n_samples = 0
    n_iface_voxels = 0  # number of (sample, iface_voxel) entries summed

    # _T_stats is not used since error metrics are computed in z-score temp. source space
    for x_spatial, cond_static, forcing_seq, y_batch, _T_stats in train_loader:
        x_spatial = x_spatial.to(device)
        cond_static = cond_static.to(device)
        forcing_seq = forcing_seq.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        y_pred = model(x_spatial, cond_static, forcing_seq)
        loss = loss_fn(y_pred, y_batch)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        with torch.no_grad():
            b = x_spatial.shape[0]
            loss_sum += loss.item() * b
            n_samples += b

            mse_sum += torch.sum((y_pred - y_batch) ** 2).item()
            target_sq_sum += torch.sum(y_batch ** 2).item()

            if iface_mask is not None:
                pred_iface = y_pred[:, iface_mask, :]
                true_iface = y_batch[:, iface_mask, :]
                iface_mse_sum += torch.sum((pred_iface - true_iface) ** 2).item()
                iface_target_sq_sum += torch.sum(true_iface ** 2).item()
                n_iface_voxels += pred_iface.numel()

    if dist_info.is_distributed:
        t = torch.tensor(
            [loss_sum, mse_sum, target_sq_sum, iface_mse_sum, iface_target_sq_sum,
             float(n_samples), float(n_iface_voxels)],
            device=device, dtype=torch.float64,
        )
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        loss_sum = t[0].item()
        mse_sum = t[1].item()
        target_sq_sum = t[2].item()
        iface_mse_sum = t[3].item()
        iface_target_sq_sum = t[4].item()
        n_samples = int(t[5].item())
        n_iface_voxels = int(t[6].item())

    training_loss = loss_sum / max(n_samples, 1)
    train_rel_l2 = math.sqrt(mse_sum / max(target_sq_sum, 1e-12)) * 100.0
    if iface_mask is not None and n_iface_voxels > 0:
        train_iface_rel_l2 = math.sqrt(iface_mse_sum / max(iface_target_sq_sum, 1e-12)) * 100.0
    else:
        train_iface_rel_l2 = 0.0
    return training_loss, train_rel_l2, train_iface_rel_l2


def _per_pair_rel_l2_percent(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    numerator = torch.mean((y_pred - y_true) ** 2, dim=(1, 2, 3))
    denominator = torch.mean(y_true ** 2, dim=(1, 2, 3)).clamp_min(1e-12)
    return torch.sqrt(numerator / denominator) * 100.0


def _write_val_pair_rows(
    writer: csv.DictWriter,
    epoch: int,
    dataset: SnapshotPairDataset,
    pairs: list[tuple[int, int, int]],
    rel_l2: torch.Tensor,
    iface_rel_l2: torch.Tensor,
) -> None:
    rows = []
    for row_idx, (sim_id, s, j) in enumerate(pairs):
        params = dataset.sim_params[int(sim_id)]
        rows.append(
            {
                "epoch": int(epoch),
                "sim_id": int(sim_id),
                "temporal_family": params["temporal_family"],
                "spatial_family": params["spatial_family"],
                "t_s": float(dataset.t_grid[s]),
                "t_bar": float(dataset.t_grid[j] - dataset.t_grid[s]),
                "R_c": float(params["R_c"]),
                "rel_l2": float(rel_l2[row_idx].item()),
                "iface_rel_l2": float(iface_rel_l2[row_idx].item()),
            }
        )
    writer.writerows(rows)


def validate(
    model,
    val_loader,
    device,
    *,
    iface_mask=None,
    dataset: SnapshotPairDataset | None = None,
    pair_csv_path: str | Path | None = None,
    epoch: int | None = None,
) -> tuple[float, float]:
    """Return (val_rel_l2, val_iface_rel_l2) after validation.

    Pass the **unwrapped** model (not the DDP wrapper) — see Fix 1: a rank-0-only
    DDP forward would deadlock other ranks waiting on a collective.
    Validation is intended to run on rank 0 only; the caller broadcasts results.

    Metric semantics: global ratio sqrt(sum MSE / sum target_sq), consistent
    with `train_one_epoch`.
    """
    with torch.no_grad():
        model.eval()

        mse_sum = 0.0
        target_sq_sum = 0.0
        iface_mse_sum = 0.0
        iface_target_sq_sum = 0.0

        pair_cursor = 0
        write_pairs = dataset is not None and pair_csv_path is not None and epoch is not None
        pair_file = None
        pair_writer = None
        if write_pairs:
            pair_path = Path(pair_csv_path)
            pair_path.parent.mkdir(parents=True, exist_ok=True)
            write_header = not pair_path.exists() or pair_path.stat().st_size == 0
            pair_file = pair_path.open("a", newline="")
            pair_writer = csv.DictWriter(pair_file, fieldnames=VAL_PAIR_FIELDNAMES)
            if write_header:
                pair_writer.writeheader()

        try:
            for x_spatial, cond_static, forcing_seq, y_batch, _T_stats in val_loader:
                x_spatial = x_spatial.to(device)
                cond_static = cond_static.to(device)
                forcing_seq = forcing_seq.to(device)
                y_batch = y_batch.to(device)

                y_pred = model(x_spatial, cond_static, forcing_seq)
                mse_sum += torch.sum((y_pred - y_batch) ** 2).item()
                target_sq_sum += torch.sum(y_batch ** 2).item()

                if iface_mask is not None:
                    pred_iface = y_pred[:, iface_mask, :]
                    true_iface = y_batch[:, iface_mask, :]
                    iface_mse_sum += torch.sum((pred_iface - true_iface) ** 2).item()
                    iface_target_sq_sum += torch.sum(true_iface ** 2).item()

                if write_pairs and pair_writer is not None and dataset is not None:
                    rel_l2 = _per_pair_rel_l2_percent(y_pred, y_batch).cpu()
                    if iface_mask is not None:
                        pred_iface_b = y_pred[:, iface_mask, :]
                        true_iface_b = y_batch[:, iface_mask, :]
                        iface_rel_l2 = _per_pair_rel_l2_percent(pred_iface_b[:, :, None, :], true_iface_b[:, :, None, :]).cpu()
                    else:
                        iface_rel_l2 = torch.zeros_like(rel_l2)
                    batch_size = x_spatial.shape[0]
                    pairs = dataset._pairs[pair_cursor:pair_cursor + batch_size]
                    _write_val_pair_rows(pair_writer, int(epoch), dataset, pairs, rel_l2, iface_rel_l2)
                    pair_cursor += batch_size
        finally:
            if pair_file is not None:
                pair_file.close()

        val_loss = math.sqrt(mse_sum / max(target_sq_sum, 1e-12)) * 100.0
        if iface_mask is not None:
            val_iface = math.sqrt(iface_mse_sum / max(iface_target_sq_sum, 1e-12)) * 100.0
        else:
            val_iface = 0.0
        return val_loss, val_iface


def run_one_seed(
    config: dict,
    seed: int,
    run_dir: str | Path,
    *,
    train_loader_override=None,
    val_loader_override=None,
) -> dict[str, float | int | str]:
    dist_info = get_dist_info()
    is_main = dist_info.rank == 0

    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)

    _t_start = time.time()
    if is_main:
        _capture_git_provenance(run_path)
        _capture_job_context(run_path)
        _dump_resolved_config(run_path, config)
    if dist_info.is_distributed:
        dist.barrier()

    # --- Skip completed runs ---
    if _is_training_complete(run_path):
        result = _load_completed_result(run_path, seed)
        if is_main:
            print(f"Seed {seed}: training already complete (best_val={result['best_val']:.4f}%), skipping.")
            _write_final_metrics(
                run_path, seed=seed, config=config, wall_time_s=0.0, status="already_complete"
            )
        return result

    # --- Check for interrupted run ---
    latest_path = run_path / "fno2d_latest.pt"
    resuming = latest_path.exists()

    set_seed(seed)

    if (train_loader_override is None) != (val_loader_override is None):
        raise ValueError("train_loader_override and val_loader_override must be provided together.")

    if train_loader_override is None:
        trajectories, x_grid, y_grid, t_grid = load_sim_data(
            sim_traj_path=config["data"]["trajectories.npy"],
            x_grid_path=config["data"]["x_grid_path"],
            y_grid_path=config["data"]["y_grid_path"],
            t_grid_path=config["data"]["t_grid_path"],
        )
        sim_params = np.load(config["data"]["sim_params_path"], allow_pickle=True)
        solver_dt = load_solver_dt(config["data"]["t_grid_path"])

        train_ids, val_ids, test_ids = split_sim_ids(num_sims=trajectories.shape[0], train_frac=0.7, val_frac=0.15, seed=0)
        mu_global, sigma_global = compute_global_stats(trajectories, train_ids)

        num_workers = resolve_num_workers(config, dist_info.world_size)
        training_set, validation_set, _ = create_dataloaders(
            trajectories=trajectories, x_grid=x_grid, y_grid=y_grid, t_grid=t_grid,
            train_ids=train_ids, val_ids=val_ids, test_ids=test_ids,
            batch_size=config["training"]["batch_size"],
            sim_params=sim_params,
            mu_global=mu_global,
            sigma_global=sigma_global,
            n_snapshots=config["training"].get("n_snapshots", 15),
            n_snapshots_test=config["training"].get("n_snapshots_test", None),
            noise_std=config["training"].get("noise_std", 0.0),
            num_workers=num_workers,
            dt=solver_dt,
            temporal_samples=config["model"]["parameters"].get("temporal_samples", TEMPORAL_SAMPLES),
            world_size=dist_info.world_size,
            rank=dist_info.rank,
            sampler_seed=seed,
        )
    else:
        training_set = train_loader_override
        validation_set = val_loader_override
        x_grid = np.asarray(training_set.dataset.x_grid)
        y_grid = np.asarray(training_set.dataset.y_grid)
        mu_global = float(training_set.dataset.mu_global)
        sigma_global = float(training_set.dataset.sigma_global)

    device = resolve_device(
        config["training"].get("device", "auto"),
        local_rank=dist_info.local_rank if dist_info.is_distributed else None,
    )
    if is_main:
        print(f"Training on: {device} (world_size={dist_info.world_size})")

    model_cfg = config["model"]["parameters"]
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
        dropout=model_cfg.get("dropout", 0.0),
        spectral_dropout=model_cfg.get("spectral_dropout", 0.0),
    )

    # --- Resume state ---
    start_epoch = 0
    best_val_loss = float("inf")
    bad_epochs = 0

    if resuming:
        ckpt = torch.load(latest_path, map_location="cpu", weights_only=False)
        _validate_resume_compatibility(ckpt["conf"], config)
        fno.load_state_dict(ckpt["model_state"])
        best_val_loss = ckpt["best_val"]
        bad_epochs = ckpt.get("bad_epochs", 0)
        start_epoch = ckpt["epoch"] + 1

    fno.to(device)

    # Wrap with DDP after .to(device) and after loading checkpoint weights.
    # `fno_unwrapped` is the original module — use it for state_dict() and for
    # rank-0-only validation forward (Fix 1).
    if dist_info.is_distributed:
        fno = DistributedDataParallel(
            fno,
            device_ids=[dist_info.local_rank] if torch.cuda.is_available() else None,
            output_device=dist_info.local_rank if torch.cuda.is_available() else None,
            find_unused_parameters=False,
        )
        fno_unwrapped = fno.module
    else:
        fno_unwrapped = fno

    optimizer = build_optimizer(config, fno.parameters())
    scheduler = build_scheduler(config, optimizer)

    if resuming:
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        if is_main:
            print(f"Resuming seed {seed} from epoch {start_epoch} (best_val={best_val_loss:.4f}%, bad_epochs={bad_epochs})")

    loss_cfg = config["training"].get("loss", {})
    loss_fn = SpatiallyWeightedMSE(
        x_grid=x_grid,
        y_grid=y_grid,
        interface_x=loss_cfg.get("interface_x", 0.5),
        interface_half_width=loss_cfg.get("interface_half_width", 0.05),
        interface_weight=loss_cfg.get("interface_weight", 1.0),
    ).to(device)

    iface_mask = build_interface_mask(
        x_grid, y_grid,
        loss_cfg.get("interface_x", 0.5), loss_cfg.get("interface_half_width", 0.05),
    ).to(device)

    epochs = config["training"]["epochs"]
    validate_every = config["training"]["validate_every"]
    patience = config["training"]["patience"]

    best_path = run_path / "fno2d_best.pt"

    # --- CSV setup (rank 0 only) ---
    csv_path = run_path / "train_metrics.csv"
    val_pairs_path = run_path / "val_pairs.csv"
    fieldnames = ["epoch", "train_loss", "train_rel_l2", "train_iface_rel_l2", "val_rel_l2", "val_iface_rel_l2", "lr", "is_best"]
    csv_file = None
    csv_writer = None

    if is_main:
        if resuming and csv_path.exists():
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

        if resuming and val_pairs_path.exists():
            with val_pairs_path.open("r", newline="") as f:
                reader = csv.DictReader(f)
                kept_rows = [row for row in reader if int(row["epoch"]) < start_epoch]
            with val_pairs_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=VAL_PAIR_FIELDNAMES)
                writer.writeheader()
                writer.writerows(kept_rows)
        elif not resuming and val_pairs_path.exists():
            val_pairs_path.unlink()

    grad_clip = config["training"].get("grad_clip", None)
    warmup_epochs = config["training"].get("curriculum_warmup", 0)

    should_stop = False

    for epoch in range(start_epoch, epochs):
        # Lead-time curriculum: progressively expose longer lead times.
        # Deterministic given `frac` -> identical _active_len on every rank.
        if warmup_epochs > 0:
            frac = min(1.0, (epoch + 1) / warmup_epochs)
            training_set.dataset.set_curriculum_fraction(frac)

        # DistributedSampler shuffle ordering: curriculum first, then set_epoch.
        if dist_info.is_distributed and hasattr(training_set.sampler, "set_epoch"):
            training_set.sampler.set_epoch(epoch)

        lr = optimizer.param_groups[0]["lr"]
        train_loss, train_rel_l2, train_iface_rel_l2 = train_one_epoch(
            model=fno, train_loader=training_set, optimizer=optimizer,
            loss_fn=loss_fn, device=device, iface_mask=iface_mask,
            grad_clip=grad_clip, dist_info=dist_info,
        )
        scheduler.step()

        if is_main:
            print(f"Epoch {epoch}: train_loss={train_loss:.6f}, train_rel_l2={train_rel_l2:.4f}%, iface_rel_l2={train_iface_rel_l2:.4f}%")

        is_best = 0
        val_loss = None
        val_iface_rel_l2 = None

        if (epoch % validate_every) == 0:
            # Validation runs on rank 0 only, using the UNWRAPPED model (Fix 1).
            # All ranks then receive (val_loss, val_iface, is_best, bad_epochs,
            # best_val_loss, should_stop) via broadcast (Fix 2).
            if is_main:
                val_loss, val_iface_rel_l2 = validate(
                    model=fno_unwrapped,
                    val_loader=validation_set,
                    device=device,
                    iface_mask=iface_mask,
                    dataset=validation_set.dataset,
                    pair_csv_path=val_pairs_path,
                    epoch=epoch,
                )
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
                            "model_state": fno_unwrapped.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "scheduler_state": scheduler.state_dict(),
                            "best_val": best_val_loss,
                            "bad_epochs": bad_epochs,
                            "mu_global": mu_global,
                            "sigma_global": sigma_global,
                        },
                        best_path,
                    )
                    print(f"Saved new best: {best_val_loss} -> {best_path}")
                else:
                    bad_epochs += 1
                    if bad_epochs >= patience:
                        print(f"Early stopping: no improvement for {patience} evaluations.")
                        should_stop = True

            # Broadcast control-flow state so every rank stays in lockstep (Fix 2).
            if dist_info.is_distributed:
                state = {
                    "val_loss": val_loss,
                    "val_iface_rel_l2": val_iface_rel_l2,
                    "best_val_loss": best_val_loss,
                    "bad_epochs": bad_epochs,
                    "is_best": is_best,
                    "should_stop": should_stop,
                }
                obj_list = [state if is_main else None]
                dist.broadcast_object_list(obj_list, src=0)
                state = obj_list[0]
                val_loss = state["val_loss"]
                val_iface_rel_l2 = state["val_iface_rel_l2"]
                best_val_loss = state["best_val_loss"]
                bad_epochs = state["bad_epochs"]
                is_best = state["is_best"]
                should_stop = state["should_stop"]

        # Save latest checkpoint (rank 0 only) for resume.
        if is_main:
            torch.save(
                {
                    "epoch": epoch,
                    "conf": config,
                    "seed": seed,
                    "model_state": fno_unwrapped.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_val": best_val_loss,
                    "bad_epochs": bad_epochs,
                    "mu_global": mu_global,
                    "sigma_global": sigma_global,
                },
                latest_path,
            )

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

        if dist_info.is_distributed:
            dist.barrier()

        if should_stop:
            break

    if is_main:
        if csv_file is not None:
            csv_file.close()
        if latest_path.exists():
            latest_path.unlink()

        status = "early_stopped" if should_stop else "completed"
        _write_final_metrics(
            run_path,
            seed=seed,
            config=config,
            wall_time_s=time.time() - _t_start,
            status=status,
        )

    if dist_info.is_distributed:
        dist.barrier()

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
