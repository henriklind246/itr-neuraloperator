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
    TEMPORAL_SAMPLES,
    SnapshotPairDataset,
    collate_fn,
    compute_global_stats,
    create_dataloaders,
    load_ramp_seconds,
    load_sim_data,
    load_solver_dt,
    one_step_physics_view,
    problem_from_config,
    split_sim_ids,
)
from src.operators.distributed import DistInfo, get_dist_info
from src.operators.fno2d import FNO2d
from src.operators.losses import (
    EPS_JUMP,
    SpatiallyWeightedMSE,
    build_interface_band,
    build_interface_mask,
    full_bc_physics_loss,
    get_batch_interface_x,
    interface_flanking_nodes,
    interface_flanking_nodes_per_sample,
    per_sample_node_jump_errors,
    per_sample_nrmse,
    per_sample_sq_rms,
    physics_residual_loss,
    tail_stats,
)
from src.operators.rollout import build_rollout_item_from_base, _q_callable_for_boundary
from src.physics.fv_residual import FullBCData, build_cn_geom_batched
from src.operators.utils import resolve_device

from omegaconf import OmegaConf

from problems.registry import REGISTRY as _PROBLEM_REGISTRY


# Universal val-pair columns written for every benchmark. Benchmark-specific
# conditioning columns are appended from each ProblemSpec.val_pair_fields, so
# one CSV schema (the union) covers all benchmarks with empty cells where a
# column does not apply.
BASE_VAL_PAIR_FIELDNAMES = [
    "epoch",
    "sim_id",
    "benchmark",
    "t_s",
    "t_bar",
    "R_c",
    "rel_l2",
    "iface_rel_l2",
    "nrmse",
    "rmse_K",
    "gnrmse_pct",
    "node_jump_rmse_K",
    "node_jump_nrmse",
    "node_jump_gnrmse_pct",
]


def _build_val_pair_fieldnames() -> list[str]:
    extra: list[str] = []
    for problem in _PROBLEM_REGISTRY.values():
        for field in getattr(problem, "val_pair_fields", ()):
            if field not in extra and field not in BASE_VAL_PAIR_FIELDNAMES:
                extra.append(field)
    return BASE_VAL_PAIR_FIELDNAMES + sorted(extra)


VAL_PAIR_FIELDNAMES = _build_val_pair_fieldnames()


# Fixed seed for selecting which validation pairs are written to val_pairs.csv
# when training.val_pairs_max_rows caps the per-pass row count. Holding the seed
# constant makes the written subset identical on every validation pass, so
# per-pair error can be tracked across epochs.
VAL_PAIRS_SUBSAMPLE_SEED = 0


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

    # Compose the active benchmark group. Hydra entrypoints do this via the
    # `defaults` list; load_config (fixed-run path) bypasses Hydra, so merge the
    # group manually. Name comes from $BENCHMARK, else the `defaults` list,
    # else "forcing". The group file is `# @package _global_`, so its keys are
    # top-level and provide the (benchmark-specific) representational dims.
    benchmark_name = os.environ.get("BENCHMARK")
    if benchmark_name is None:
        benchmark_name = "forcing"
        for entry in cfg.get("defaults", []) or []:
            if not isinstance(entry, str):
                d = OmegaConf.to_container(entry)
                if isinstance(d, dict) and "benchmark" in d:
                    benchmark_name = d["benchmark"]
                    break
    benchmark_path = project_root / "conf" / "benchmark" / f"{benchmark_name}.yaml"
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark config not found: {benchmark_path}")
    cfg = OmegaConf.merge(cfg, OmegaConf.load(benchmark_path))

    # Compose the representation group the same way. Name comes from
    # $REPRESENTATION, else the `defaults` list, else "temporal_encoder". The
    # group file is `# @package _global_` and sets `benchmark.representation`.
    representation_name = os.environ.get("REPRESENTATION")
    if representation_name is None:
        representation_name = "temporal_encoder"
        for entry in cfg.get("defaults", []) or []:
            if not isinstance(entry, str):
                d = OmegaConf.to_container(entry)
                if isinstance(d, dict) and "representation" in d:
                    representation_name = d["representation"]
                    break
    representation_path = project_root / "conf" / "representation" / f"{representation_name}.yaml"
    if not representation_path.exists():
        raise FileNotFoundError(f"Representation config not found: {representation_path}")
    cfg = OmegaConf.merge(cfg, OmegaConf.load(representation_path))

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


def _stamp_resolved_dims(config: dict, dims) -> None:
    """Record the spec-owned model dims into config["model"]["parameters"].

    These dims are resolved from the (benchmark, representation) pair, not the
    YAML, so without this the dumped config_used.yaml leaves them null.
    """
    params = config.setdefault("model", {}).setdefault("parameters", {})
    params["in_channels"] = dims.in_channels
    params["cond_static_dim"] = dims.cond_static_dim
    params["temporal_token_dim"] = dims.temporal_token_dim
    params["use_temporal_encoder"] = dims.use_temporal_encoder
    params["s_y_channel"] = dims.s_y_channel
    params["use_forcing_time_aug"] = dims.use_forcing_time_aug


def _dump_resolved_config(run_path: Path, config: dict) -> None:
    OmegaConf.save(OmegaConf.create(config), run_path / "config_used.yaml")


def _summarize_metrics_csv(csv_path: Path) -> dict:
    with csv_path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))

    def _f(row, key):
        v = row.get(key, "")
        return float(v) if v not in ("", None) else None

    _extra_keys = (
        "train_nrmse", "train_rmse_K", "train_gnrmse_pct", "train_max_err_K",
        "train_node_jump_rmse_K", "train_node_jump_nrmse", "train_node_jump_gnrmse_pct",
        "val_nrmse", "val_nrmse_p50", "val_nrmse_iqr",
        "val_nrmse_p90", "val_nrmse_p99", "val_nrmse_max",
        "val_rmse_K", "val_rmse_K_p90", "val_rmse_K_p99", "val_rmse_K_max",
        "val_gnrmse_pct", "val_gnrmse_pct_p99", "val_max_err_K",
        "val_node_jump_rmse_K", "val_node_jump_nrmse",
        "val_node_jump_nrmse_p90", "val_node_jump_nrmse_p99", "val_node_jump_nrmse_max",
        "val_node_jump_gnrmse_pct", "val_node_jump_gnrmse_pct_p99",
    )

    def _row(row):
        out = {
            "epoch": int(row["epoch"]),
            "train_rel_l2": _f(row, "train_rel_l2"),
            "train_iface_rel_l2": _f(row, "train_iface_rel_l2"),
            "val_rel_l2": _f(row, "val_rel_l2"),
            "val_iface_rel_l2": _f(row, "val_iface_rel_l2"),
            "lr": _f(row, "lr"),
        }
        for k in _extra_keys:
            out[k] = _f(row, k)
        return out

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
            "val_nrmse": best["val_nrmse"],
            "val_nrmse_p50": best["val_nrmse_p50"],
            "val_nrmse_iqr": best["val_nrmse_iqr"],
            "val_nrmse_p90": best["val_nrmse_p90"],
            "val_nrmse_p99": best["val_nrmse_p99"],
            "val_nrmse_max": best["val_nrmse_max"],
            "val_rmse_K": best["val_rmse_K"],
            "val_rmse_K_p90": best["val_rmse_K_p90"],
            "val_rmse_K_p99": best["val_rmse_K_p99"],
            "val_rmse_K_max": best["val_rmse_K_max"],
            "val_gnrmse_pct": best["val_gnrmse_pct"],
            "val_gnrmse_pct_p99": best["val_gnrmse_pct_p99"],
            "val_max_err_K": best["val_max_err_K"],
            "val_node_jump_rmse_K": best["val_node_jump_rmse_K"],
            "val_node_jump_nrmse": best["val_node_jump_nrmse"],
            "val_node_jump_gnrmse_pct": best["val_node_jump_gnrmse_pct"],
            "val_node_jump_gnrmse_pct_p99": best["val_node_jump_gnrmse_pct_p99"],
            "train_nrmse": best["train_nrmse"],
            "train_rmse_K": best["train_rmse_K"],
            "train_gnrmse_pct": best["train_gnrmse_pct"],
            "train_max_err_K": best["train_max_err_K"],
            "train_node_jump_rmse_K": best["train_node_jump_rmse_K"],
            "train_node_jump_nrmse": best["train_node_jump_nrmse"],
            "train_node_jump_gnrmse_pct": best["train_node_jump_gnrmse_pct"],
        },
        "final": {
            "epoch": final["epoch"],
            "train_rel_l2": final["train_rel_l2"],
            "train_iface_rel_l2": final["train_iface_rel_l2"],
            "train_nrmse": final["train_nrmse"],
            "train_rmse_K": final["train_rmse_K"],
            "train_gnrmse_pct": final["train_gnrmse_pct"],
            "train_max_err_K": final["train_max_err_K"],
            "train_node_jump_rmse_K": final["train_node_jump_rmse_K"],
            "train_node_jump_nrmse": final["train_node_jump_nrmse"],
            "train_node_jump_gnrmse_pct": final["train_node_jump_gnrmse_pct"],
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


def _physics_batch_loss(model, physics_loader, phys_iter, geom_cfg, device):
    """Draw one physics batch (cycling the loader), forward the model for the
    one-step prediction ``T^{n+1}``, and return ``(loss, batch_size, phys_iter)``.

    The input snapshot's normalized temperature (spatial channel 0) is the truth
    ``T^n``; the per-sample contact resistance ``R_c`` is recovered from
    ``cond_static`` and used to build the batched CN face conductances so the
    interface jump physics is exact per sample.
    """
    try:
        pbatch = next(phys_iter)
    except StopIteration:
        phys_iter = iter(physics_loader)
        pbatch = next(phys_iter)

    x_spatial = pbatch["spatial"].to(device)
    cond_static = pbatch["cond_static"].to(device)
    forcing_seq = pbatch["forcing_seq"].to(device) if "forcing_seq" in pbatch else None

    T_n = x_spatial[..., 0:1]
    y_pred = model(x_spatial, cond_static, forcing_seq)

    rc_lo, rc_hi = geom_cfg["rc_range"]
    R_c = cond_static[:, geom_cfg["rc_index"]] * (rc_hi - rc_lo) + rc_lo
    geom = build_cn_geom_batched(
        geom_cfg["x_grid"], geom_cfg["y_grid"],
        geom_cfg["k_left"], geom_cfg["k_right"], geom_cfg["interface_x"],
        R_c, geom_cfg["dt"], sigma_global=geom_cfg["sigma_global"],
        device=device, dtype=y_pred.dtype,
    )
    p_loss = physics_residual_loss(T_n, y_pred, geom)
    return p_loss, x_spatial.shape[0], phys_iter


def _build_physics_loader(config, phys_cfg, spec, mu_global, sigma_global, *, batch_size, num_workers):
    """Build the dedicated one-step (W1) physics loader and the geometry config
    used to assemble the per-sample CN residual.

    Reads its own ``save_stride=1`` dataset (``physics.data_dir`` or, as a
    fallback, the main ``data`` paths — only correct if that set is itself
    ``save_stride=1``) so consecutive snapshots are exactly one solver step
    ``dt`` apart. Stage 1 is ``forcing``-only: the two-slab ``k=2/k=1`` interface
    at ``x=0.5`` and scalar ``R_c`` in ``[0.05, 1.0]`` (cond_static index 2).

    Normalizes the physics inputs with the MAIN training set's
    ``(mu_global, sigma_global)`` (passed in) — the model was normalized with
    those stats, so the physics set must reuse them or the model sees
    OOD-scaled inputs and the residual's ``sigma_global`` rescale is wrong.
    """
    from torch.utils.data import DataLoader

    data_dir = phys_cfg.get("data_dir") or None
    if data_dir is not None:
        base = Path(data_dir)
        traj_path = str(base / "trajectories.npy")
        x_path = str(base / "x_grid.npy")
        y_path = str(base / "y_grid.npy")
        t_path = str(base / "t_grid.npy")
        sim_params_path = str(base / "sim_params.npy")
    else:
        traj_path = config["data"]["trajectories.npy"]
        x_path = config["data"]["x_grid_path"]
        y_path = config["data"]["y_grid_path"]
        t_path = config["data"]["t_grid_path"]
        sim_params_path = config["data"]["sim_params_path"]

    trajectories, xg, yg, tg = load_sim_data(
        sim_traj_path=traj_path, x_grid_path=x_path,
        y_grid_path=y_path, t_grid_path=t_path,
    )
    sim_params = np.load(sim_params_path, allow_pickle=True)
    solver_dt = load_solver_dt(t_path)
    ramp_seconds = load_ramp_seconds(t_path)
    train_ids, _, _ = split_sim_ids(
        num_sims=trajectories.shape[0], train_frac=0.7, val_frac=0.15, seed=0
    )

    phys_dataset = SnapshotPairDataset(
        trajectories=trajectories, t_grid=tg, x_grid=xg, y_grid=yg,
        sim_ids=train_ids, sim_params=sim_params,
        mu_global=mu_global, sigma_global=sigma_global,
        n_snapshots=None, noise_std=0.0,
        dt=solver_dt, ramp_seconds=ramp_seconds,
        temporal_samples=config["model"]["parameters"].get("temporal_samples", TEMPORAL_SAMPLES),
        problem=spec,
    )
    phys_dataset = one_step_physics_view(phys_dataset)

    pbs = phys_cfg.get("physics_batch_size") or batch_size
    pin = torch.cuda.is_available()
    workers = num_workers if num_workers is not None else (4 if pin else 0)
    loader = DataLoader(
        phys_dataset, batch_size=pbs, shuffle=True,
        pin_memory=pin, num_workers=workers,
        persistent_workers=workers > 0, collate_fn=collate_fn,
    )

    geom_cfg = {
        "x_grid": np.asarray(xg, dtype=float),
        "y_grid": np.asarray(yg, dtype=float),
        "k_left": 2.0,
        "k_right": 1.0,
        "interface_x": float(phys_cfg.get("interface_x", 0.5)),
        "dt": float(phys_cfg.get("dt", solver_dt)),
        "sigma_global": float(sigma_global),
        "rc_range": (0.05, 1.0),
        "rc_index": 2,
    }
    return loader, geom_cfg


class CollocationSampler:
    """Draws W2 collocation batches from the held-out (validation) sims.

    Each item is a base state ``T(t_s)`` (a real saved snapshot of a held-out
    sim — the under-supervised cross-sim distribution the residual is meant to
    constrain) plus a model query at lead ``L`` (``t = t_s + L``) and at
    ``t + dt``. Forwarding the model at both times yields two fields one solver
    step ``dt`` apart, which the ``full_bc`` CN residual couples. The left-edge
    flux ``q_L`` and right-edge Dirichlet target are reconstructed analytically
    at the collocation times so the boundary closures are exact.

    Uses its OWN ``np.random.Generator`` so enabling the physics term does not
    perturb the global torch/numpy RNG stream (the data loader's shuffling and
    the no-op guarantee are untouched).
    """

    def __init__(self, ds, spec, geom_cfg, *, batch_size, dt, rng_seed):
        self.ds = ds
        self.spec = spec
        self.geom_cfg = geom_cfg
        self.batch_size = int(batch_size)
        self.dt = float(dt)
        self.t_final = float(ds.t_final)
        self.sim_ids = np.asarray(ds.sim_ids)
        self.rng = np.random.default_rng(rng_seed)

    def sample_batch(self, max_lead: float):
        """Sample one collocation batch. Leads are drawn in ``[dt, max_lead]``
        from base snapshots whose ``t_s + max_lead + dt`` stays within the
        forcing time support ``t_final``. Raises if no base snapshot supports
        the requested ``max_lead`` (rather than silently extrapolating the
        forcing past its support) and hard-asserts each ``t + dt`` is in range.
        """
        ds = self.ds
        dt = self.dt
        tol = 1e-9
        max_lead = max(float(max_lead), dt)
        valid_s = [
            int(s) for s in ds.t_indices
            if float(ds.t_grid[s]) + max_lead + dt <= self.t_final + tol
        ]
        if not valid_s:
            raise ValueError(
                f"No base snapshot supports collocation max_lead={max_lead} + "
                f"dt={dt} within t_final={self.t_final}; lower collocation_lead_max."
            )

        items_t, items_tdt = [], []
        R_c_list, qLn_list, qLnp1_list = [], [], []
        for _ in range(self.batch_size):
            sid = int(self.rng.choice(self.sim_ids))
            s = int(self.rng.choice(valid_s))
            lead = float(self.rng.uniform(dt, max_lead)) if max_lead > dt else dt
            t_s = float(ds.t_grid[s])
            t = t_s + lead
            t_dt = t + dt
            if t_dt > self.t_final + tol:
                raise AssertionError(
                    f"collocation time t+dt={t_dt} exceeds forcing support "
                    f"t_final={self.t_final}"
                )

            base_item = ds.problem.build_item(ds, sid, s, s)
            current = base_item["spatial"][..., 0]
            item_t = build_rollout_item_from_base(
                base_item, ds, self.spec, sid, current, t_s, t
            )
            item_tdt = build_rollout_item_from_base(
                base_item, ds, self.spec, sid, current, t_s, t_dt
            )

            params = ds.sim_params[sid]
            a_fn = _q_callable_for_boundary(ds, sid, params)
            s_y = ds.s_y_profiles[sid]
            qLn = (float(a_fn(t)) * s_y).astype(np.float32)
            qLnp1 = (float(a_fn(t_dt)) * s_y).astype(np.float32)

            items_t.append({k: torch.from_numpy(v) for k, v in item_t.items()})
            items_tdt.append({k: torch.from_numpy(v) for k, v in item_tdt.items()})
            R_c_list.append(float(params["R_c"]))
            qLn_list.append(qLn)
            qLnp1_list.append(qLnp1)

        batch_t = collate_fn(items_t)
        batch_tdt = collate_fn(items_tdt)
        R_c = torch.tensor(R_c_list, dtype=torch.float32)
        qL_n = torch.from_numpy(np.stack(qLn_list))
        qL_np1 = torch.from_numpy(np.stack(qLnp1_list))
        return batch_t, batch_tdt, R_c, qL_n, qL_np1


def _build_collocation_sampler(config, phys_cfg, ds, spec, mu_global, sigma_global,
                               *, batch_size, rng_seed):
    """Build the W2 collocation sampler over the held-out (validation) sims.

    ``forcing``-only: two-slab ``k=2/k=1`` interface at ``x=0.5``, scalar
    ``R_c``, constant right-edge ``T_right = 300 K`` (the Dirichlet target,
    normalized with the training stats). The physics step ``dt`` is the
    dedicated collocation residual step (``physics.dt``, default 0.005), which
    may differ from the saved data spacing.
    """
    dt = float(phys_cfg.get("dt", 0.005))
    T_right = 300.0
    geom_cfg = {
        "x_grid": np.asarray(ds.x_grid, dtype=float),
        "y_grid": np.asarray(ds.y_grid, dtype=float),
        "k_left": 2.0,
        "k_right": 1.0,
        "interface_x": float(phys_cfg.get("interface_x", 0.5)),
        "dt": dt,
        "sigma_global": float(sigma_global),
        "T_right_tilde": (T_right - float(mu_global)) / float(sigma_global),
    }
    pbs = phys_cfg.get("physics_batch_size") or batch_size
    return CollocationSampler(
        ds, spec, geom_cfg, batch_size=pbs, dt=dt, rng_seed=rng_seed
    )


def _collocation_batch_loss(model, sampler, max_lead, region_weights, device):
    """Forward the model at the collocation pair ``(t, t+dt)`` and return the
    region-partitioned ``full_bc`` physics loss dict plus the batch size.

    Both fields are model outputs from the same base input, so the right-edge
    Dirichlet closure is anchored at BOTH times (``dirichlet_both_ends``).
    """
    batch_t, batch_tdt, R_c, qL_n, qL_np1 = sampler.sample_batch(max_lead)

    def _fwd(b):
        spatial = b["spatial"].to(device)
        cond = b["cond_static"].to(device)
        fseq = b["forcing_seq"].to(device) if "forcing_seq" in b else None
        return model(spatial, cond, fseq)

    yt = _fwd(batch_t)
    ytdt = _fwd(batch_tdt)

    geom = build_cn_geom_batched(
        sampler.geom_cfg["x_grid"], sampler.geom_cfg["y_grid"],
        sampler.geom_cfg["k_left"], sampler.geom_cfg["k_right"],
        sampler.geom_cfg["interface_x"], R_c.to(device),
        sampler.geom_cfg["dt"], sigma_global=sampler.geom_cfg["sigma_global"],
        device=device, dtype=yt.dtype,
    )
    bc = FullBCData(
        T_right_tilde=sampler.geom_cfg["T_right_tilde"],
        qL_n=qL_n.to(device=device, dtype=yt.dtype),
        qL_np1=qL_np1.to(device=device, dtype=yt.dtype),
    )
    out = full_bc_physics_loss(
        yt, ytdt, geom, bc,
        region_weights=region_weights, dirichlet_both_ends=True,
    )
    return out, R_c.shape[0]


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    loss_fn,
    device,
    iface_mask=None,
    grad_clip=None,
    dist_info: DistInfo | None = None,
    use_per_sample_interface: bool = False,
    x_grid: np.ndarray | None = None,
    interface_x: float = 0.5,
    sigma_global: float = 1.0,
    physics_loader=None,
    physics_geom_cfg=None,
    physics_collocation=None,
    collocation_max_lead=None,
    physics_region_weights=None,
    lambda_data: float = 1.0,
    lambda_physics: float = 0.0,
) -> dict[str, float]:
    """Train one epoch and return a dict of epoch metrics.

    Under DDP, reduces sums (loss, MSE, ||y||^2, iface MSE, ||y_iface||^2, sample
    counts, and the unified per-sample metric sums) across ranks and computes
    true global metrics from those sums — not an average of per-rank ratios.

    The unified headline metrics (``nrmse``, ``rmse_K``, ``node_jump_rmse_K``,
    ``node_jump_nrmse``) are accumulated as **summed per-sample** scalars and
    divided by ``n_samples`` after the reduce, so the train headline equals the
    mean-over-pairs definition used by validate/evaluate (not a pooled RMSE).
    ``max_err_K`` is a worst-case over the split (MAX-reduced under DDP).
    """
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

    # Unified metric accumulators (summed per-sample; mean after the reduce).
    nrmse_sum = 0.0
    rmse_K_sum = 0.0
    gnrmse_sum = 0.0  # sum of per-sample normalized RMS (= rmse_K/sigma); *100 for pct
    node_jump_rmse_K_sum = 0.0
    node_jump_nrmse_sum = 0.0
    node_jump_gnrmse_sum = 0.0  # sum of per-sample normalized jump RMS; *100 for pct
    max_abs_err = 0.0  # normalized; scaled to Kelvin after the reduce

    # Fixed-interface flanking nodes (per-sample handled inside the loop).
    fixed_left = fixed_right = None
    if x_grid is not None and not use_per_sample_interface:
        fixed_left, fixed_right = interface_flanking_nodes(x_grid, interface_x)

    # Physics-residual regularizer. Disabled => the branch below is skipped
    # entirely (no loader build, no batch draw, no residual), so the data stream
    # and RNG order are byte-identical to a data-only run. Two mutually-exclusive
    # sources: a W1 one-step ``physics_loader`` (interior residual) or a W2
    # ``physics_collocation`` sampler (full_bc residual on model-pair outputs).
    phys_collocation = physics_collocation is not None and lambda_physics > 0.0
    phys_onestep = physics_loader is not None and lambda_physics > 0.0
    phys_enabled = phys_onestep or phys_collocation
    phys_iter = iter(physics_loader) if phys_onestep else None
    phys_loss_sum = 0.0
    phys_weighted_sum = 0.0
    phys_allcell_sum = 0.0
    phys_region_sums = {
        "interior": 0.0, "left_neumann": 0.0,
        "right_dirichlet": 0.0, "topbot_adiabatic": 0.0,
    }
    n_phys = 0

    for batch in train_loader:
        x_spatial = batch["spatial"].to(device)
        cond_static = batch["cond_static"].to(device)
        forcing_seq = batch["forcing_seq"].to(device) if "forcing_seq" in batch else None
        y_batch = batch["Y"].to(device)
        iface_x = get_batch_interface_x(
            batch, device, use_per_sample_interface=use_per_sample_interface
        )

        optimizer.zero_grad()
        y_pred = model(x_spatial, cond_static, forcing_seq)
        loss = loss_fn(y_pred, y_batch, iface_x)
        if phys_collocation:
            out, p_b = _collocation_batch_loss(
                model, physics_collocation, collocation_max_lead,
                physics_region_weights, device,
            )
            p_loss = out["physics_loss_weighted"]
            (lambda_data * loss + lambda_physics * p_loss).backward()
            phys_loss_sum += p_loss.item() * p_b
            phys_weighted_sum += out["physics_loss_weighted"].item() * p_b
            phys_allcell_sum += out["physics_loss_allcell_mean"].item() * p_b
            for _r in phys_region_sums:
                phys_region_sums[_r] += out[f"phys_{_r}_mse"].item() * p_b
            n_phys += p_b
        elif phys_onestep:
            p_loss, p_b, phys_iter = _physics_batch_loss(
                model, physics_loader, phys_iter, physics_geom_cfg, device
            )
            (lambda_data * loss + lambda_physics * p_loss).backward()
            phys_loss_sum += p_loss.item() * p_b
            n_phys += p_b
        else:
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

            # --- unified per-sample metrics ---
            rms_i = per_sample_sq_rms(y_pred, y_batch)  # (B,)
            nrmse_sum += torch.sum(per_sample_nrmse(y_pred, y_batch)).item()
            rmse_K_sum += torch.sum(rms_i).item() * sigma_global
            gnrmse_sum += torch.sum(rms_i).item()
            max_abs_err = max(max_abs_err, torch.max(torch.abs(y_pred - y_batch)).item())

            left, right = fixed_left, fixed_right
            if use_per_sample_interface and iface_x is not None:
                left, right = interface_flanking_nodes_per_sample(loss_fn.x_grid_t, iface_x)
            if left is not None and right is not None:
                err_rms_i, true_jump_rms_i = per_sample_node_jump_errors(
                    y_pred, y_batch, left, right
                )
                node_jump_rmse_K_sum += torch.sum(err_rms_i).item() * sigma_global
                node_jump_gnrmse_sum += torch.sum(err_rms_i).item()
                node_jump_nrmse_sum += torch.sum(
                    err_rms_i / true_jump_rms_i.clamp_min(EPS_JUMP)
                ).item()

            if iface_x is not None:
                Ny = y_batch.shape[2]
                band = build_interface_band(
                    loss_fn.x_grid_t, iface_x, loss_fn.interface_half_width
                )  # (B, Nx)
                bf = band.to(y_batch.dtype).reshape(band.shape[0], band.shape[1], 1, 1)
                iface_mse_sum += torch.sum(bf * (y_pred - y_batch) ** 2).item()
                iface_target_sq_sum += torch.sum(bf * y_batch ** 2).item()
                n_iface_voxels += int(band.sum().item()) * Ny
            elif iface_mask is not None:
                pred_iface = y_pred[:, iface_mask, :]
                true_iface = y_batch[:, iface_mask, :]
                iface_mse_sum += torch.sum((pred_iface - true_iface) ** 2).item()
                iface_target_sq_sum += torch.sum(true_iface ** 2).item()
                n_iface_voxels += pred_iface.numel()

    if dist_info.is_distributed:
        t = torch.tensor(
            [loss_sum, mse_sum, target_sq_sum, iface_mse_sum, iface_target_sq_sum,
             float(n_samples), float(n_iface_voxels),
             nrmse_sum, rmse_K_sum, node_jump_rmse_K_sum, node_jump_nrmse_sum,
             gnrmse_sum, node_jump_gnrmse_sum, phys_loss_sum, float(n_phys),
             phys_weighted_sum, phys_allcell_sum,
             phys_region_sums["interior"], phys_region_sums["left_neumann"],
             phys_region_sums["right_dirichlet"], phys_region_sums["topbot_adiabatic"]],
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
        nrmse_sum = t[7].item()
        rmse_K_sum = t[8].item()
        node_jump_rmse_K_sum = t[9].item()
        node_jump_nrmse_sum = t[10].item()
        gnrmse_sum = t[11].item()
        node_jump_gnrmse_sum = t[12].item()
        phys_loss_sum = t[13].item()
        n_phys = int(t[14].item())
        phys_weighted_sum = t[15].item()
        phys_allcell_sum = t[16].item()
        phys_region_sums["interior"] = t[17].item()
        phys_region_sums["left_neumann"] = t[18].item()
        phys_region_sums["right_dirichlet"] = t[19].item()
        phys_region_sums["topbot_adiabatic"] = t[20].item()

        m = torch.tensor([max_abs_err], device=device, dtype=torch.float64)
        dist.all_reduce(m, op=dist.ReduceOp.MAX)
        max_abs_err = m[0].item()

    denom = max(n_samples, 1)
    training_loss = loss_sum / denom
    train_rel_l2 = math.sqrt(mse_sum / max(target_sq_sum, 1e-12)) * 100.0
    if n_iface_voxels > 0:
        train_iface_rel_l2 = math.sqrt(iface_mse_sum / max(iface_target_sq_sum, 1e-12)) * 100.0
    else:
        train_iface_rel_l2 = 0.0
    return {
        "loss": training_loss,
        "rel_l2": train_rel_l2,
        "iface_rel_l2": train_iface_rel_l2,
        "nrmse": (nrmse_sum / denom) * 100.0,
        "rmse_K": rmse_K_sum / denom,
        "gnrmse_pct": (gnrmse_sum / denom) * 100.0,
        "max_err_K": max_abs_err * sigma_global,
        "node_jump_rmse_K": node_jump_rmse_K_sum / denom,
        "node_jump_nrmse": (node_jump_nrmse_sum / denom) * 100.0,
        "node_jump_gnrmse_pct": (node_jump_gnrmse_sum / denom) * 100.0,
        "physics_loss": phys_loss_sum / max(n_phys, 1),
        # Per-region full_bc diagnostics (populated by the Stage-2 collocation
        # path; the interior W1 path leaves them at 0.0).
        "physics_loss_weighted": phys_weighted_sum / max(n_phys, 1),
        "physics_loss_allcell_mean": phys_allcell_sum / max(n_phys, 1),
        "phys_interior_mse": phys_region_sums["interior"] / max(n_phys, 1),
        "phys_left_neumann_mse": phys_region_sums["left_neumann"] / max(n_phys, 1),
        "phys_right_dirichlet_mse": phys_region_sums["right_dirichlet"] / max(n_phys, 1),
        "phys_topbot_adiabatic_mse": phys_region_sums["topbot_adiabatic"] / max(n_phys, 1),
    }


def _per_pair_rel_l2_percent(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    numerator = torch.mean((y_pred - y_true) ** 2, dim=(1, 2, 3))
    denominator = torch.mean(y_true ** 2, dim=(1, 2, 3)).clamp_min(1e-12)
    return torch.sqrt(numerator / denominator) * 100.0


# Allowed checkpoint-selection metrics -> key into the `validate` metric dict.
# All are "lower is better". Default keeps legacy `val_rel_l2` selection.
_CHECKPOINT_METRIC_KEYS = {
    "val_rel_l2": "rel_l2",
    "val_nrmse": "nrmse",
    "val_jump_nrmse": "node_jump_nrmse",
}


def _selection_metric_value(val_metrics: dict[str, float], checkpoint_metric: str) -> float:
    """Resolve the scalar used for checkpoint selection from the val metric dict.

    Unknown names fall back to the legacy ``rel_l2`` headline so a typo never
    silently picks an arbitrary metric.
    """
    key = _CHECKPOINT_METRIC_KEYS.get(checkpoint_metric, "rel_l2")
    return float(val_metrics[key])


def _write_val_pair_rows(
    writer: csv.DictWriter,
    epoch: int,
    dataset: SnapshotPairDataset,
    pairs: list[tuple[int, int, int]],
    rel_l2: torch.Tensor,
    iface_rel_l2: torch.Tensor,
    nrmse: torch.Tensor,
    rmse_K: torch.Tensor,
    gnrmse_pct: torch.Tensor,
    node_jump_rmse_K: torch.Tensor,
    node_jump_nrmse: torch.Tensor,
    node_jump_gnrmse_pct: torch.Tensor,
) -> None:
    problem = getattr(dataset, "problem", None)
    benchmark = getattr(problem, "name", "")
    rows = []
    for row_idx, (sim_id, s, j) in enumerate(pairs):
        sid = int(sim_id)
        params = dataset.sim_params[sid]
        row = {
            "epoch": int(epoch),
            "sim_id": sid,
            "benchmark": benchmark,
            "t_s": float(dataset.t_grid[s]),
            "t_bar": float(dataset.t_grid[j] - dataset.t_grid[s]),
            "R_c": float(params["R_c"]) if "R_c" in params else "",
            "rel_l2": float(rel_l2[row_idx].item()),
            "iface_rel_l2": float(iface_rel_l2[row_idx].item()),
            "nrmse": float(nrmse[row_idx].item()),
            "rmse_K": float(rmse_K[row_idx].item()),
            "gnrmse_pct": float(gnrmse_pct[row_idx].item()),
            "node_jump_rmse_K": float(node_jump_rmse_K[row_idx].item()),
            "node_jump_nrmse": float(node_jump_nrmse[row_idx].item()),
            "node_jump_gnrmse_pct": float(node_jump_gnrmse_pct[row_idx].item()),
        }
        if problem is not None:
            row.update(problem.val_pair_row(dataset, sid, int(s), int(j)))
        rows.append(row)
    writer.writerows(rows)


def validate(
    model,
    val_loader,
    device,
    *,
    iface_mask=None,
    dataset: SnapshotPairDataset | None = None,
    pair_csv_path: str | Path | None = None,
    val_pairs_max_rows: int | None = None,
    epoch: int | None = None,
    x_grid_t: torch.Tensor | None = None,
    interface_half_width: float = 0.05,
    use_per_sample_interface: bool = False,
    interface_x: float = 0.5,
    sigma_global: float = 1.0,
) -> dict[str, float]:
    """Return a metric dict after validation.

    The dict always carries ``rel_l2`` (legacy normalized rel-L2 headline) and
    ``iface_rel_l2`` so existing checkpoint selection keeps working, plus the
    unified metric suite: per-sample ``nrmse`` (mean + p90/p99/max tails),
    Kelvin ``rmse_K`` / ``max_err_K``, and node-jump ``node_jump_rmse_K`` /
    ``node_jump_nrmse`` (mean + tails).

    Pass the **unwrapped** model (not the DDP wrapper) — see Fix 1: a rank-0-only
    DDP forward would deadlock other ranks waiting on a collective.
    Validation is intended to run on rank 0 only; the caller broadcasts results.

    ``rel_l2`` keeps the global-ratio semantics of `train_one_epoch`. The new
    headline metrics use the mean-over-pairs convention (per-sample values
    aggregated), so they compare directly to `evaluate`.
    """
    with torch.no_grad():
        model.eval()

        mse_sum = 0.0
        target_sq_sum = 0.0
        iface_mse_sum = 0.0
        iface_target_sq_sum = 0.0

        nrmse_all: list[torch.Tensor] = []
        rmse_K_all: list[torch.Tensor] = []
        gnrmse_all: list[torch.Tensor] = []  # per-sample normalized RMS (= rms_i)
        node_jump_rmse_K_all: list[torch.Tensor] = []
        node_jump_nrmse_all: list[torch.Tensor] = []
        node_jump_gnrmse_all: list[torch.Tensor] = []  # per-sample normalized jump RMS
        max_abs_err = 0.0

        fixed_left = fixed_right = None
        if x_grid_t is not None and not use_per_sample_interface:
            fixed_left, fixed_right = interface_flanking_nodes(
                x_grid_t.detach().cpu().numpy(), interface_x
            )

        if val_pairs_max_rows is not None and val_pairs_max_rows < 1:
            raise ValueError("training.val_pairs_max_rows must be positive or null")

        pair_cursor = 0
        write_pairs = dataset is not None and pair_csv_path is not None and epoch is not None
        pair_file = None
        pair_writer = None
        # Boolean mask over global dataset._pairs indices selecting which pairs to
        # write. None means "write every pair" (no cap / cap >= total). The val
        # loader is unshuffled with no reordering sampler (data/dataset.py:685-714)
        # and SnapshotPairDataset yields _pairs in order, so batch position maps
        # directly to the global pair index via pair_cursor.
        sel_mask = None
        if write_pairs:
            pair_path = Path(pair_csv_path)
            pair_path.parent.mkdir(parents=True, exist_ok=True)
            write_header = not pair_path.exists() or pair_path.stat().st_size == 0
            pair_file = pair_path.open("a", newline="")
            pair_writer = csv.DictWriter(
                pair_file, fieldnames=VAL_PAIR_FIELDNAMES, restval="", extrasaction="ignore"
            )
            if write_header:
                pair_writer.writeheader()

            total_pairs = len(dataset._pairs)
            if val_pairs_max_rows is not None and val_pairs_max_rows < total_pairs:
                rng = np.random.default_rng(VAL_PAIRS_SUBSAMPLE_SEED)
                chosen = rng.choice(total_pairs, size=val_pairs_max_rows, replace=False)
                sel_mask = np.zeros(total_pairs, dtype=bool)
                sel_mask[chosen] = True
            n_sel = int(sel_mask.sum()) if sel_mask is not None else total_pairs
            print(
                f"[val-pairs] writing {n_sel} / {total_pairs} validation pairs",
                flush=True,
            )

        _t_val_start = time.perf_counter()
        _t_write_total = 0.0
        _n_batches = 0
        try:
            for batch in val_loader:
                if epoch is not None and _n_batches % 50 == 0:
                    print(
                        f"[val-progress] epoch={epoch} batch={_n_batches} "
                        f"elapsed={time.perf_counter() - _t_val_start:.1f}s",
                        flush=True,
                    )
                x_spatial = batch["spatial"].to(device)
                cond_static = batch["cond_static"].to(device)
                forcing_seq = batch["forcing_seq"].to(device) if "forcing_seq" in batch else None
                y_batch = batch["Y"].to(device)
                iface_x = get_batch_interface_x(
                    batch, device, use_per_sample_interface=use_per_sample_interface
                )

                y_pred = model(x_spatial, cond_static, forcing_seq)
                mse_sum += torch.sum((y_pred - y_batch) ** 2).item()
                target_sq_sum += torch.sum(y_batch ** 2).item()
                _n_batches += 1

                # --- Unified per-sample metrics (mean-over-pairs convention) ---
                rms_i = per_sample_sq_rms(y_pred, y_batch)
                nrmse_all.append(per_sample_nrmse(y_pred, y_batch).cpu())
                rmse_K_all.append((rms_i * sigma_global).cpu())
                gnrmse_all.append(rms_i.cpu())
                max_abs_err = max(
                    max_abs_err, torch.max(torch.abs(y_pred - y_batch)).item()
                )

                left, right = fixed_left, fixed_right
                if use_per_sample_interface and iface_x is not None and x_grid_t is not None:
                    left, right = interface_flanking_nodes_per_sample(x_grid_t, iface_x)
                jump_nrmse_i = None
                if left is not None and right is not None:
                    err_rms_i, true_jump_rms_i = per_sample_node_jump_errors(
                        y_pred, y_batch, left, right
                    )
                    node_jump_rmse_K_all.append((err_rms_i * sigma_global).cpu())
                    node_jump_gnrmse_all.append(err_rms_i.cpu())
                    jump_nrmse_i = (err_rms_i / true_jump_rms_i.clamp_min(EPS_JUMP)).cpu()
                    node_jump_nrmse_all.append(jump_nrmse_i)

                bf = None
                if iface_x is not None and x_grid_t is not None:
                    band = build_interface_band(x_grid_t, iface_x, interface_half_width)  # (B, Nx)
                    bf = band.to(y_batch.dtype).reshape(band.shape[0], band.shape[1], 1, 1)
                    iface_mse_sum += torch.sum(bf * (y_pred - y_batch) ** 2).item()
                    iface_target_sq_sum += torch.sum(bf * y_batch ** 2).item()
                elif iface_mask is not None:
                    pred_iface = y_pred[:, iface_mask, :]
                    true_iface = y_batch[:, iface_mask, :]
                    iface_mse_sum += torch.sum((pred_iface - true_iface) ** 2).item()
                    iface_target_sq_sum += torch.sum(true_iface ** 2).item()

                if write_pairs and pair_writer is not None and dataset is not None:
                    _tw = time.perf_counter()
                    batch_size = x_spatial.shape[0]
                    # Select which rows of this batch to write before computing
                    # any per-pair diagnostic metric, so the cap saves compute as
                    # well as I/O. The full-batch aggregation above is untouched.
                    if sel_mask is None:
                        local = None
                    else:
                        local = np.nonzero(sel_mask[pair_cursor:pair_cursor + batch_size])[0]
                    if local is None or len(local) > 0:
                        base_pairs = dataset._pairs[pair_cursor:pair_cursor + batch_size]
                        if local is None:
                            yp, yt, bf_l = y_pred, y_batch, bf
                            pairs = base_pairs
                            nrmse_pct = nrmse_all[-1] * 100.0
                            rmse_K_b = rmse_K_all[-1]
                            gnrmse_pct_b = gnrmse_all[-1] * 100.0
                            jump_src = jump_nrmse_i
                            node_jump_rmse_src = (
                                node_jump_rmse_K_all[-1] if jump_nrmse_i is not None else None
                            )
                            node_jump_gnrmse_src = (
                                node_jump_gnrmse_all[-1] if jump_nrmse_i is not None else None
                            )
                        else:
                            idx_cpu = torch.from_numpy(local).long()
                            idx_dev = idx_cpu.to(y_pred.device)
                            yp, yt = y_pred[idx_dev], y_batch[idx_dev]
                            bf_l = bf[idx_dev] if bf is not None else None
                            pairs = [base_pairs[i] for i in local]
                            nrmse_pct = (nrmse_all[-1] * 100.0)[idx_cpu]
                            rmse_K_b = rmse_K_all[-1][idx_cpu]
                            gnrmse_pct_b = (gnrmse_all[-1] * 100.0)[idx_cpu]
                            jump_src = jump_nrmse_i[idx_cpu] if jump_nrmse_i is not None else None
                            node_jump_rmse_src = (
                                node_jump_rmse_K_all[-1][idx_cpu] if jump_nrmse_i is not None else None
                            )
                            node_jump_gnrmse_src = (
                                node_jump_gnrmse_all[-1][idx_cpu] if jump_nrmse_i is not None else None
                            )
                        rel_l2 = _per_pair_rel_l2_percent(yp, yt).cpu()
                        if bf_l is not None:
                            num = torch.sum(bf_l * (yp - yt) ** 2, dim=(1, 2, 3))
                            den = torch.sum(bf_l * yt ** 2, dim=(1, 2, 3)).clamp_min(1e-12)
                            iface_rel_l2 = (torch.sqrt(num / den) * 100.0).cpu()
                        elif iface_mask is not None:
                            pred_iface_b = yp[:, iface_mask, :]
                            true_iface_b = yt[:, iface_mask, :]
                            iface_rel_l2 = _per_pair_rel_l2_percent(pred_iface_b[:, :, None, :], true_iface_b[:, :, None, :]).cpu()
                        else:
                            iface_rel_l2 = torch.zeros_like(rel_l2)
                        if jump_src is not None:
                            node_jump_rmse_K_b = node_jump_rmse_src
                            node_jump_nrmse_pct = jump_src * 100.0
                            node_jump_gnrmse_pct_b = node_jump_gnrmse_src * 100.0
                        else:
                            node_jump_rmse_K_b = torch.zeros_like(rel_l2)
                            node_jump_nrmse_pct = torch.zeros_like(rel_l2)
                            node_jump_gnrmse_pct_b = torch.zeros_like(rel_l2)
                        _write_val_pair_rows(
                            pair_writer, int(epoch), dataset, pairs, rel_l2, iface_rel_l2,
                            nrmse_pct, rmse_K_b, gnrmse_pct_b, node_jump_rmse_K_b,
                            node_jump_nrmse_pct, node_jump_gnrmse_pct_b,
                        )
                    pair_cursor += batch_size
                    _t_write_total += time.perf_counter() - _tw
        finally:
            if pair_file is not None:
                pair_file.close()

        _t_val_total = time.perf_counter() - _t_val_start
        if epoch is not None:
            print(
                f"[val-timing] epoch={epoch} batches={_n_batches} "
                f"total={_t_val_total:.1f}s forward={_t_val_total - _t_write_total:.1f}s "
                f"val_pairs_write={_t_write_total:.1f}s",
                flush=True,
            )

        val_loss = math.sqrt(mse_sum / max(target_sq_sum, 1e-12)) * 100.0
        if iface_target_sq_sum > 0.0:
            val_iface = math.sqrt(iface_mse_sum / max(iface_target_sq_sum, 1e-12)) * 100.0
        else:
            val_iface = 0.0

        nrmse_stats = tail_stats(torch.cat(nrmse_all)) if nrmse_all else tail_stats(torch.empty(0))
        rmse_K_stats = tail_stats(torch.cat(rmse_K_all)) if rmse_K_all else tail_stats(torch.empty(0))
        gnrmse_stats = tail_stats(torch.cat(gnrmse_all)) if gnrmse_all else tail_stats(torch.empty(0))
        if node_jump_nrmse_all:
            jump_nrmse_stats = tail_stats(torch.cat(node_jump_nrmse_all))
            node_jump_rmse_K_mean = float(torch.cat(node_jump_rmse_K_all).mean())
            node_jump_gnrmse_stats = tail_stats(torch.cat(node_jump_gnrmse_all))
        else:
            jump_nrmse_stats = tail_stats(torch.empty(0))
            node_jump_rmse_K_mean = 0.0
            node_jump_gnrmse_stats = tail_stats(torch.empty(0))

        return {
            "rel_l2": val_loss,
            "iface_rel_l2": val_iface,
            "nrmse": nrmse_stats["mean"] * 100.0,
            "nrmse_p50": nrmse_stats["p50"] * 100.0,
            "nrmse_iqr": nrmse_stats["iqr"] * 100.0,
            "nrmse_p90": nrmse_stats["p90"] * 100.0,
            "nrmse_p99": nrmse_stats["p99"] * 100.0,
            "nrmse_max": nrmse_stats["max"] * 100.0,
            "rmse_K": rmse_K_stats["mean"],
            "rmse_K_p90": rmse_K_stats["p90"],
            "rmse_K_p99": rmse_K_stats["p99"],
            "rmse_K_max": rmse_K_stats["max"],
            "gnrmse_pct": gnrmse_stats["mean"] * 100.0,
            "gnrmse_pct_p99": gnrmse_stats["p99"] * 100.0,
            "max_err_K": max_abs_err * sigma_global,
            "node_jump_rmse_K": node_jump_rmse_K_mean,
            "node_jump_nrmse": jump_nrmse_stats["mean"] * 100.0,
            "node_jump_nrmse_p90": jump_nrmse_stats["p90"] * 100.0,
            "node_jump_nrmse_p99": jump_nrmse_stats["p99"] * 100.0,
            "node_jump_nrmse_max": jump_nrmse_stats["max"] * 100.0,
            "node_jump_gnrmse_pct": node_jump_gnrmse_stats["mean"] * 100.0,
            "node_jump_gnrmse_pct_p99": node_jump_gnrmse_stats["p99"] * 100.0,
        }


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

    # Resolve the ProblemSpec up front and stamp the representation-determined
    # dims into config["model"]["parameters"] so config_used.yaml records the
    # dims the model is actually built with (they are owned by the spec, not the
    # benchmark/representation YAML).
    spec = problem_from_config(config)
    _stamp_resolved_dims(config, spec.dims)

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
        ramp_seconds = load_ramp_seconds(config["data"]["t_grid_path"])

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
            ramp_seconds=ramp_seconds,
            temporal_samples=config["model"]["parameters"].get("temporal_samples", TEMPORAL_SAMPLES),
            world_size=dist_info.world_size,
            rank=dist_info.rank,
            sampler_seed=seed,
            problem=spec,
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
    dims = spec.dims
    fno = FNO2d(
        modes1=model_cfg["modes1"],
        modes2=model_cfg["modes2"],
        width=model_cfg["width"],
        in_channels=dims.in_channels,
        out_channels=model_cfg.get("out_channels", 1),
        n_layers=model_cfg.get("n_layers", 4),
        cond_static_dim=dims.cond_static_dim,
        cond_hidden=model_cfg.get("cond_hidden", 256),
        temporal_token_dim=dims.temporal_token_dim,
        temporal_hidden=model_cfg.get("temporal_hidden", 128),
        forcing_embed_dim=model_cfg.get("forcing_embed_dim", 64),
        forcing_spatial_dim=model_cfg.get("forcing_spatial_dim", 16),
        dropout=model_cfg.get("dropout", 0.0),
        spectral_dropout=model_cfg.get("spectral_dropout", 0.0),
        use_temporal_encoder=dims.use_temporal_encoder,
        use_forcing_time_aug=dims.use_forcing_time_aug,
        s_y_channel=dims.s_y_channel,
        padding_reference_resolution=model_cfg.get("padding_reference_resolution"),
        padding_mode=model_cfg.get("padding_mode", "zeros"),
        cin_exclude_padding=model_cfg.get("cin_exclude_padding", False),
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

    use_per_sample_interface = bool(loss_cfg.get("per_sample_interface_x", False))

    # Physics-residual regularizer. Build the physics supply ONLY when active;
    # lambda_physics=0 is a TRUE no-op (nothing below is constructed, so the data
    # stream and RNG are untouched). Two modes:
    #   W1 (mode=one_step): dedicated consecutive-pair loader + interior residual.
    #   W2 (mode=collocation): a sampler over the HELD-OUT (validation) sims that
    #     forwards the model at (t, t+dt) and enforces the full_bc residual on its
    #     own outputs at long-lead/OOD horizons (the cross-sim under-supervised
    #     distribution). The lead curriculum ramps the max collocation lead.
    phys_cfg = config["training"].get("physics", {})
    lambda_physics = float(phys_cfg.get("lambda_physics", 0.0))
    lambda_data = float(phys_cfg.get("lambda_data", 1.0))
    phys_mode = str(phys_cfg.get("mode", "one_step"))
    physics_loader = None
    physics_geom_cfg = None
    physics_collocation = None
    physics_region_weights = None
    collocation_lead_start = float(phys_cfg.get("collocation_lead_start", 0.01))
    collocation_lead_max = float(phys_cfg.get("collocation_lead_max", 0.01))
    collocation_lead_warmup = int(phys_cfg.get("collocation_lead_warmup_epochs", 0))
    if lambda_physics > 0.0 and phys_mode == "collocation":
        physics_region_weights = phys_cfg.get("full_bc_region_weights", None)
        physics_collocation = _build_collocation_sampler(
            config, phys_cfg, validation_set.dataset, spec,
            mu_global, sigma_global,
            batch_size=config["training"]["batch_size"], rng_seed=seed,
        )
        if is_main:
            print(
                f"Physics regularizer ON (W2 collocation): lambda_data={lambda_data}, "
                f"lambda_physics={lambda_physics}, residual=full_bc, "
                f"lead_start={collocation_lead_start}, lead_max={collocation_lead_max}, "
                f"lead_warmup={collocation_lead_warmup}",
                flush=True,
            )
    elif lambda_physics > 0.0:
        physics_loader, physics_geom_cfg = _build_physics_loader(
            config, phys_cfg, spec, mu_global, sigma_global,
            batch_size=config["training"]["batch_size"],
            num_workers=resolve_num_workers(config, dist_info.world_size),
        )
        if is_main:
            print(
                f"Physics regularizer ON: lambda_data={lambda_data}, "
                f"lambda_physics={lambda_physics}, "
                f"residual={phys_cfg.get('residual', 'interior')}, "
                f"mode={phys_cfg.get('mode', 'one_step')}",
                flush=True,
            )

    epochs = config["training"]["epochs"]
    validate_every = config["training"]["validate_every"]
    patience = config["training"]["patience"]
    checkpoint_metric = config["training"].get("checkpoint_metric", "val_rel_l2")

    best_path = run_path / "fno2d_best.pt"

    # --- CSV setup (rank 0 only) ---
    csv_path = run_path / "train_metrics.csv"
    val_pairs_path = run_path / "val_pairs.csv"
    fieldnames = [
        "epoch", "train_loss", "train_physics_loss",
        "train_physics_loss_weighted", "train_physics_loss_allcell_mean",
        "train_phys_interior_mse", "train_phys_left_neumann_mse",
        "train_phys_right_dirichlet_mse", "train_phys_topbot_adiabatic_mse",
        "train_rel_l2", "train_iface_rel_l2",
        "train_nrmse", "train_rmse_K", "train_gnrmse_pct", "train_max_err_K",
        "train_node_jump_rmse_K", "train_node_jump_nrmse", "train_node_jump_gnrmse_pct",
        "val_rel_l2", "val_iface_rel_l2",
        "val_nrmse", "val_nrmse_p50", "val_nrmse_iqr",
        "val_nrmse_p90", "val_nrmse_p99", "val_nrmse_max",
        "val_rmse_K", "val_rmse_K_p90", "val_rmse_K_p99", "val_rmse_K_max",
        "val_gnrmse_pct", "val_gnrmse_pct_p99", "val_max_err_K",
        "val_node_jump_rmse_K", "val_node_jump_nrmse",
        "val_node_jump_nrmse_p90", "val_node_jump_nrmse_p99", "val_node_jump_nrmse_max",
        "val_node_jump_gnrmse_pct", "val_node_jump_gnrmse_pct_p99",
        "lr", "is_best",
    ]
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
                writer = csv.DictWriter(
                    f, fieldnames=VAL_PAIR_FIELDNAMES, restval="", extrasaction="ignore"
                )
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

        # W2 collocation-lead curriculum: ramp the max lead from start to max over
        # `collocation_lead_warmup` epochs (an optimization-stability device — do
        # not slam a strong residual onto garbage long-lead outputs early). Inert
        # when collocation is off.
        if collocation_lead_warmup > 0:
            cfrac = min(1.0, (epoch + 1) / collocation_lead_warmup)
        else:
            cfrac = 1.0
        collocation_max_lead = (
            collocation_lead_start
            + cfrac * (collocation_lead_max - collocation_lead_start)
        )

        train_metrics = train_one_epoch(
            model=fno, train_loader=training_set, optimizer=optimizer,
            loss_fn=loss_fn, device=device, iface_mask=iface_mask,
            grad_clip=grad_clip, dist_info=dist_info,
            use_per_sample_interface=use_per_sample_interface,
            x_grid=x_grid, interface_x=loss_cfg.get("interface_x", 0.5),
            sigma_global=sigma_global,
            physics_loader=physics_loader,
            physics_geom_cfg=physics_geom_cfg,
            physics_collocation=physics_collocation,
            collocation_max_lead=collocation_max_lead,
            physics_region_weights=physics_region_weights,
            lambda_data=lambda_data,
            lambda_physics=lambda_physics,
        )
        train_loss = train_metrics["loss"]
        train_rel_l2 = train_metrics["rel_l2"]
        train_iface_rel_l2 = train_metrics["iface_rel_l2"]
        scheduler.step()

        if is_main:
            print(
                f"Epoch {epoch}: train_loss={train_loss:.6f}, "
                f"train_rel_l2={train_rel_l2:.4f}%, iface_rel_l2={train_iface_rel_l2:.4f}% | "
                f"nrmse={train_metrics['nrmse']:.3f}% gnrmse={train_metrics['gnrmse_pct']:.3f}% "
                f"rmse_K={train_metrics['rmse_K']:.4f}K max_err_K={train_metrics['max_err_K']:.4f}K "
                f"node_jump_rmse_K={train_metrics['node_jump_rmse_K']:.4f}K",
                flush=True,
            )

        is_best = 0
        val_loss = None
        val_iface_rel_l2 = None
        val_metrics = None

        if (epoch % validate_every) == 0:
            # Validation runs on rank 0 only, using the UNWRAPPED model (Fix 1).
            # All ranks then receive the val metric dict and control-flow state
            # via broadcast (Fix 2).
            if is_main:
                val_metrics = validate(
                    model=fno_unwrapped,
                    val_loader=validation_set,
                    device=device,
                    iface_mask=iface_mask,
                    dataset=validation_set.dataset,
                    pair_csv_path=val_pairs_path,
                    val_pairs_max_rows=config["training"].get("val_pairs_max_rows", None),
                    epoch=epoch,
                    x_grid_t=loss_fn.x_grid_t,
                    interface_half_width=loss_cfg.get("interface_half_width", 0.05),
                    use_per_sample_interface=use_per_sample_interface,
                    interface_x=loss_cfg.get("interface_x", 0.5),
                    sigma_global=sigma_global,
                )
                val_loss = val_metrics["rel_l2"]
                val_iface_rel_l2 = val_metrics["iface_rel_l2"]
                print(f"Validation loss for epoch {epoch}: rel_l2={val_loss:.4f}%, iface_rel_l2={val_iface_rel_l2:.4f}%", flush=True)
                print(
                    "  [val metric suite]\n"
                    f"    Kelvin    : rmse_K mean={val_metrics['rmse_K']:.4f}K "
                    f"p99={val_metrics['rmse_K_p99']:.4f}K "
                    f"max_sample={val_metrics['rmse_K_max']:.4f}K | "
                    f"pointwise max_err_K={val_metrics['max_err_K']:.4f}K\n"
                    f"    dimensionless: gnrmse_pct mean={val_metrics['gnrmse_pct']:.3f}% "
                    f"p99={val_metrics['gnrmse_pct_p99']:.3f}% | "
                    f"nrmse mean={val_metrics['nrmse']:.3f}% median={val_metrics['nrmse_p50']:.3f}% "
                    f"IQR={val_metrics['nrmse_iqr']:.3f}% p99={val_metrics['nrmse_p99']:.3f}% "
                    f"max={val_metrics['nrmse_max']:.3f}%\n"
                    f"    interface : node_jump_rmse_K={val_metrics['node_jump_rmse_K']:.4f}K | "
                    f"node_jump_gnrmse_pct mean={val_metrics['node_jump_gnrmse_pct']:.3f}% "
                    f"p99={val_metrics['node_jump_gnrmse_pct_p99']:.3f}% | "
                    f"node_jump_nrmse mean={val_metrics['node_jump_nrmse']:.3f}% "
                    f"p99={val_metrics['node_jump_nrmse_p99']:.3f}%",
                    flush=True,
                )

                # Checkpoint selection: lower is better for every supported metric.
                selection_value = _selection_metric_value(val_metrics, checkpoint_metric)
                if selection_value < best_val_loss:
                    best_val_loss = selection_value
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
                    print(f"Saved new best ({checkpoint_metric}): {best_val_loss} -> {best_path}", flush=True)
                else:
                    bad_epochs += 1
                    if bad_epochs >= patience:
                        print(f"Early stopping: no improvement for {patience} evaluations.")
                        should_stop = True

            # Broadcast control-flow state so every rank stays in lockstep (Fix 2).
            if dist_info.is_distributed:
                state = {
                    "val_metrics": val_metrics,
                    "best_val_loss": best_val_loss,
                    "bad_epochs": bad_epochs,
                    "is_best": is_best,
                    "should_stop": should_stop,
                }
                obj_list = [state if is_main else None]
                dist.broadcast_object_list(obj_list, src=0)
                state = obj_list[0]
                val_metrics = state["val_metrics"]
                val_loss = None if val_metrics is None else val_metrics["rel_l2"]
                val_iface_rel_l2 = None if val_metrics is None else val_metrics["iface_rel_l2"]
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

            def _vm(key: str) -> str | float:
                return "" if val_metrics is None else float(val_metrics[key])

            csv_writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": float(train_loss),
                    "train_physics_loss": float(train_metrics["physics_loss"]),
                    "train_physics_loss_weighted": float(train_metrics["physics_loss_weighted"]),
                    "train_physics_loss_allcell_mean": float(train_metrics["physics_loss_allcell_mean"]),
                    "train_phys_interior_mse": float(train_metrics["phys_interior_mse"]),
                    "train_phys_left_neumann_mse": float(train_metrics["phys_left_neumann_mse"]),
                    "train_phys_right_dirichlet_mse": float(train_metrics["phys_right_dirichlet_mse"]),
                    "train_phys_topbot_adiabatic_mse": float(train_metrics["phys_topbot_adiabatic_mse"]),
                    "train_rel_l2": float(train_rel_l2),
                    "train_iface_rel_l2": float(train_iface_rel_l2),
                    "train_nrmse": float(train_metrics["nrmse"]),
                    "train_rmse_K": float(train_metrics["rmse_K"]),
                    "train_gnrmse_pct": float(train_metrics["gnrmse_pct"]),
                    "train_max_err_K": float(train_metrics["max_err_K"]),
                    "train_node_jump_rmse_K": float(train_metrics["node_jump_rmse_K"]),
                    "train_node_jump_nrmse": float(train_metrics["node_jump_nrmse"]),
                    "train_node_jump_gnrmse_pct": float(train_metrics["node_jump_gnrmse_pct"]),
                    "val_rel_l2": _vm("rel_l2"),
                    "val_iface_rel_l2": _vm("iface_rel_l2"),
                    "val_nrmse": _vm("nrmse"),
                    "val_nrmse_p50": _vm("nrmse_p50"),
                    "val_nrmse_iqr": _vm("nrmse_iqr"),
                    "val_nrmse_p90": _vm("nrmse_p90"),
                    "val_nrmse_p99": _vm("nrmse_p99"),
                    "val_nrmse_max": _vm("nrmse_max"),
                    "val_rmse_K": _vm("rmse_K"),
                    "val_rmse_K_p90": _vm("rmse_K_p90"),
                    "val_rmse_K_p99": _vm("rmse_K_p99"),
                    "val_rmse_K_max": _vm("rmse_K_max"),
                    "val_gnrmse_pct": _vm("gnrmse_pct"),
                    "val_gnrmse_pct_p99": _vm("gnrmse_pct_p99"),
                    "val_max_err_K": _vm("max_err_K"),
                    "val_node_jump_rmse_K": _vm("node_jump_rmse_K"),
                    "val_node_jump_nrmse": _vm("node_jump_nrmse"),
                    "val_node_jump_nrmse_p90": _vm("node_jump_nrmse_p90"),
                    "val_node_jump_nrmse_p99": _vm("node_jump_nrmse_p99"),
                    "val_node_jump_nrmse_max": _vm("node_jump_nrmse_max"),
                    "val_node_jump_gnrmse_pct": _vm("node_jump_gnrmse_pct"),
                    "val_node_jump_gnrmse_pct_p99": _vm("node_jump_gnrmse_pct_p99"),
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
