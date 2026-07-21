import csv
import json
import math
import os
import random
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
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
from src.operators.soap import SOAP
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
from src.physics.boundary_forcing import integrate_temporal_ramped_signed
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

    if optimizer_name == "SOAP":
        soap_cfg = training_cfg.get("soap", None) or {}
        betas = soap_cfg.get("betas", [0.95, 0.95])
        return SOAP(
            params,
            lr=training_cfg["learning_rate"],
            weight_decay=training_cfg["weight_decay"],
            betas=(float(betas[0]), float(betas[1])),
            shampoo_beta=float(soap_cfg.get("shampoo_beta", 0.95)),
            eps=float(soap_cfg.get("eps", 1e-8)),
            precondition_frequency=int(soap_cfg.get("precondition_frequency", 10)),
            max_precond_dim=int(soap_cfg.get("max_precond_dim", 10000)),
            merge_dims=bool(soap_cfg.get("merge_dims", False)),
            precondition_1d=bool(soap_cfg.get("precondition_1d", False)),
        )

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
    step_unit = "epoch"

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


class PICViTExponentialScheduler:
    step_unit = "update"
    STATE_VERSION = 1

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        peak_lr: float,
        decay_every: int,
        decay_rate: float,
        min_lr: float,
    ):
        self.optimizer = optimizer
        self.peak_lr = float(peak_lr)
        self.decay_every = int(decay_every)
        self.decay_rate = float(decay_rate)
        self.min_lr = float(min_lr)

        if self.peak_lr <= 0:
            raise ValueError("PICViTExponential requires a positive peak_lr.")
        if self.decay_every <= 0:
            raise ValueError("PICViTExponential requires decay_every > 0.")
        if not 0.0 < self.decay_rate < 1.0:
            raise ValueError("PICViTExponential requires 0 < decay_rate < 1.")
        if self.min_lr <= 0 or self.min_lr > self.peak_lr:
            raise ValueError("PICViTExponential requires 0 < min_lr <= peak_lr.")

        for group_idx, param_group in enumerate(self.optimizer.param_groups):
            group_lr = float(param_group["lr"])
            if not math.isclose(group_lr, self.peak_lr, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(
                    "PICViTExponential requires every optimizer parameter group "
                    f"to start at training.learning_rate={self.peak_lr}; group "
                    f"{group_idx} has lr={group_lr}."
                )

        self.next_update = 0
        self._set_lr(self.lr_for_update(self.next_update))

    def lr_for_update(self, update_idx: int) -> float:
        if update_idx < 0:
            raise ValueError(f"update_idx must be >= 0, got {update_idx}")
        lr = self.peak_lr * self.decay_rate ** (update_idx / self.decay_every)
        return max(self.min_lr, lr)

    def _set_lr(self, lr: float) -> None:
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def step(self) -> None:
        self.next_update += 1
        self._set_lr(self.lr_for_update(self.next_update))

    def state_dict(self) -> dict[str, int | float]:
        return {
            "version": self.STATE_VERSION,
            "next_update": self.next_update,
            "peak_lr": self.peak_lr,
            "decay_every": self.decay_every,
            "decay_rate": self.decay_rate,
            "min_lr": self.min_lr,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        version = int(state_dict.get("version", -1))
        if version != self.STATE_VERSION:
            raise ValueError(
                f"Unsupported PICViTExponential state version {version}; "
                f"expected {self.STATE_VERSION}."
            )

        expected = {
            "peak_lr": self.peak_lr,
            "decay_every": self.decay_every,
            "decay_rate": self.decay_rate,
            "min_lr": self.min_lr,
        }
        for key, current_value in expected.items():
            if key not in state_dict:
                raise ValueError(f"PICViTExponential state is missing {key}.")
            saved_value = state_dict[key]
            matches = (
                int(saved_value) == int(current_value)
                if key == "decay_every"
                else math.isclose(
                    float(saved_value), float(current_value), rel_tol=0.0, abs_tol=1e-12
                )
            )
            if not matches:
                raise ValueError(
                    f"PICViTExponential resume mismatch for {key}: "
                    f"checkpoint={saved_value}, current={current_value}."
                )

        next_update = int(state_dict.get("next_update", -1))
        if next_update < 0:
            raise ValueError("PICViTExponential state requires next_update >= 0.")
        self.next_update = next_update
        self._set_lr(self.lr_for_update(self.next_update))


def _set_scheduler_step_unit(scheduler, step_unit: str):
    scheduler.step_unit = step_unit
    return scheduler


def _advance_scheduler(
    scheduler,
    *,
    unit: str,
    successful_updates: int,
) -> bool:
    if unit not in {"update", "epoch"}:
        raise ValueError(f"Unknown scheduler step unit: {unit}")
    scheduler_unit = getattr(scheduler, "step_unit", None)
    if scheduler_unit not in {"update", "epoch"}:
        raise ValueError("Scheduler must define step_unit as 'update' or 'epoch'.")
    if successful_updates <= 0 or scheduler_unit != unit:
        return False
    scheduler.step()
    return True


def build_scheduler(config: dict, optimizer: torch.optim.Optimizer):
    training_cfg = config["training"]
    sched_cfg = training_cfg.get("scheduler", {})
    sched_type = sched_cfg.get("type", "StepLR")

    if sched_type == "CosineWarmRestarts":
        return _set_scheduler_step_unit(
            torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=sched_cfg.get("T_0", 100),
                T_mult=sched_cfg.get("T_mult", 2),
                eta_min=sched_cfg.get("eta_min", 1e-6),
            ),
            "epoch",
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

    if sched_type == "PICViTExponential":
        return PICViTExponentialScheduler(
            optimizer,
            peak_lr=float(training_cfg["learning_rate"]),
            decay_every=int(sched_cfg["decay_every"]),
            decay_rate=float(sched_cfg["decay_rate"]),
            min_lr=float(sched_cfg["min_lr"]),
        )

    if sched_type == "StepLR":
        return _set_scheduler_step_unit(
            torch.optim.lr_scheduler.StepLR(
                optimizer=optimizer,
                step_size=sched_cfg["step_size"],
                gamma=sched_cfg["gamma"],
            ),
            "epoch",
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

    # The one-step (W1) CN residual couples CONSECUTIVE saved snapshots and
    # assumes they are exactly one residual step `dt` apart. If the physics set
    # was saved with `save_stride > 1` its snapshot spacing is `stride * dt`, so
    # the residual would be evaluated against a wrong `dt` and silently mistrain.
    # Validate the saved spacing against the residual dt instead of trusting it.
    geom_dt = float(phys_cfg.get("dt", solver_dt))
    # t_grid is persisted as float32; a uniform grid jitters by the float32
    # quantum (~3e-8 at t=0.3). Compare to the mean spacing with a tolerance
    # that absorbs that yet still catches a real save_stride mismatch.
    tg_arr = np.asarray(tg, dtype=np.float64)
    if tg_arr.size >= 2:
        spacings = np.diff(tg_arr)
        grid_dt = float(spacings.mean())
        if not np.allclose(spacings, grid_dt, rtol=1e-3, atol=1e-9):
            raise ValueError(
                "physics one-step loader requires a uniformly-spaced t_grid; "
                f"got non-uniform spacings (min={float(spacings.min())}, "
                f"max={float(spacings.max())}) from {t_path}."
            )
        if not np.isclose(grid_dt, geom_dt, rtol=1e-3, atol=1e-9):
            raise ValueError(
                f"physics one-step loader snapshot spacing {grid_dt} != residual "
                f"dt {geom_dt}: the one-step CN residual assumes consecutive "
                "snapshots are exactly one solver step apart. Point "
                "physics.data_dir at a save_stride=1 dataset, or set physics.dt "
                "to the snapshot spacing."
            )

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
        "dt": geom_dt,
        "sigma_global": float(sigma_global),
        "rc_range": (0.05, 1.0),
        "rc_index": 2,
    }
    return loader, geom_cfg


@dataclass(frozen=True)
class TemporalBinSpec:
    """One shared temporal-bin definition for lead stratification (Step 1) AND
    causal lead weighting (Step 2), so the two techniques bin identically.

    Built once in ``run_one_seed`` from the built sampler's path-aware lead
    domain ``[lead_lo, lead_hi]`` (on-grid CN interval-midpoint domain for
    diffusion, admissible continuous-lead range for the generic forcing path).
    ``bin_edges`` is ``linspace(lead_lo, lead_hi, n_bins + 1)``. Both the
    sampler (stratified allocation + ``_last_lead_bin_ids``) and the weighter
    receive this same object; neither recomputes edges independently.
    """

    lead_lo: float
    lead_hi: float
    n_bins: int

    @property
    def bin_edges(self) -> np.ndarray:
        return np.linspace(self.lead_lo, self.lead_hi, self.n_bins + 1)

    def interval_to_bin(self, lead):
        """Map a lead (scalar or array) to a bin id in ``[0, n_bins-1]``, with the
        same clamp used by the sampler when it records bin ids and by the weighter
        when it forms per-bin means. ``clamp(bucketize(lead)-1, 0, n_bins-1)``.
        """
        edges = self.bin_edges
        idx = np.searchsorted(edges, np.asarray(lead, dtype=float), side="right") - 1
        return np.clip(idx, 0, self.n_bins - 1).astype(int)

    def active_bins(self, current_lead_min: float, current_lead_max: float) -> np.ndarray:
        """Boolean ``(n_bins,)`` mask of bins that overlap the current lead window
        ``[current_lead_min, current_lead_max]``: ``(bin_lower < current_lead_max)
        & (bin_upper > current_lead_min)``.
        """
        edges = self.bin_edges
        lower = edges[:-1]
        upper = edges[1:]
        return (lower < current_lead_max) & (upper > current_lead_min)


class CausalLeadWeighter:
    """Residual-adaptive causal weighting of the interior CN residual by lead bin
    (Step 2 / technique 1 of the PINO recipe).

    Down-weights a lead bin's interior residual until the earlier bins' residuals
    fall, using ``w_b = exp(-eps * sum_{j<b} L_j)`` over the shared
    ``TemporalBinSpec`` bins (``L_j`` = current-step or EMA per-bin mean interior
    residual). Bin 0 always has weight 1; weights are non-increasing in bin id; an
    empty later bin still inherits the accumulated suppression of the earlier bins
    (it is not reset to weight 1).

    Two phases keep all state mutation OUT of the loss function and make the DDP
    collective explicit:

    - ``prepare`` performs the ONLY collective (a SUM all-reduce of the local
      per-bin residual sums and counts) and returns the detached GLOBAL sums,
      counts, and causal weights. Called inside ``_collocation_batch_loss`` before
      the differentiable objective is formed, so ``current`` mode sees this step's
      global per-bin losses with no one-step lag.
    - ``commit`` performs NO collective; it folds the already-global stats into the
      EMA, advances ``_step``, and steps epsilon (only when every ACTIVE bin is
      populated this step). Called in ``train_one_epoch`` after backward.

    Epsilon grows (``x eps_growth``, capped ``eps_max``) when the last ACTIVE bin's
    weight exceeds ``last_bin_threshold`` and shrinks (``/eps_growth``, floored
    ``eps_min``) when the mean active-bin weight drops below ``mean_weight_floor``
    and ``allow_eps_decrease`` -- both gated on full active-bin coverage so an
    unreachable bin (restricted lead window / curriculum ramp) can never freeze the
    schedule.
    """

    def __init__(
        self, spec: "TemporalBinSpec", *, eps=1.0, loss_history="current",
        ema_alpha=0.9, update_every=1, eps_growth=2.0, eps_max=100.0,
        eps_min=0.01, allow_eps_decrease=True, last_bin_threshold=0.99,
        mean_weight_floor=0.5,
    ):
        if loss_history not in ("current", "ema"):
            raise ValueError(
                f"causal_weighting.loss_history must be 'current' or 'ema', "
                f"got {loss_history!r}."
            )
        self.spec = spec
        self.n_bins = int(spec.n_bins)
        self.eps = float(eps)
        self.loss_history = str(loss_history)
        self.ema_alpha = float(ema_alpha)
        self.update_every = max(1, int(update_every))
        self.eps_growth = float(eps_growth)
        self.eps_max = float(eps_max)
        self.eps_min = float(eps_min)
        self.allow_eps_decrease = bool(allow_eps_decrease)
        self.last_bin_threshold = float(last_bin_threshold)
        self.mean_weight_floor = float(mean_weight_floor)

        self.ema = np.zeros(self.n_bins, dtype=np.float64)
        self._ema_initialized = False
        self._step = 0
        self._last_global_bin_loss = None
        self._last_bin_counts = None
        self._last_weights = np.ones(self.n_bins, dtype=np.float64)
        self._active_mask = None

    def active_mask(self, max_lead) -> np.ndarray:
        """Active-bin mask for the current lead window ``[lead_lo, max_lead]``."""
        return self.spec.active_bins(self.spec.lead_lo, float(max_lead))

    def _current_loss(self, sums, counts) -> np.ndarray:
        sums = np.asarray(sums, dtype=np.float64).reshape(-1)
        counts = np.asarray(counts, dtype=np.float64).reshape(-1)
        return np.where(counts > 0, sums / np.clip(counts, 1.0, None), 0.0)

    def _ema_candidate(self, current_loss, counts) -> np.ndarray:
        pop = np.asarray(counts, dtype=np.float64).reshape(-1) > 0
        if not self._ema_initialized:
            return np.where(pop, current_loss, self.ema)
        blended = self.ema_alpha * self.ema + (1.0 - self.ema_alpha) * current_loss
        return np.where(pop, blended, self.ema)

    def _weights_from_loss(self, bin_loss) -> np.ndarray:
        bin_loss = np.asarray(bin_loss, dtype=np.float64).reshape(-1)
        # Exclusive cumulative sum: entry b is sum of bins strictly before b, so
        # bin 0 -> weight exp(0) = 1 and each later bin inherits earlier suppression.
        excl = np.concatenate([[0.0], np.cumsum(bin_loss)[:-1]])
        return np.exp(-self.eps * excl)

    def _bin_loss(self, current_loss, counts) -> np.ndarray:
        if self.loss_history == "ema":
            return self._ema_candidate(current_loss, counts)
        return current_loss

    def prepare(self, local_bin_sums, local_bin_counts, active_mask, dist_info):
        """SUM-all-reduce the local per-bin residual sums/counts (the only
        collective) and return detached global ``(sums, counts, weights)``.

        ``local_bin_sums`` may be a differentiable tensor; only a DETACHED copy is
        reduced here, so the backward graph is never touched by the collective.
        """
        device = local_bin_sums.device
        packed = torch.stack([
            local_bin_sums.detach().to(torch.float64),
            local_bin_counts.detach().to(device=device, dtype=torch.float64),
        ])
        if dist_info is not None and getattr(dist_info, "is_distributed", False):
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        global_sums = packed[0]
        global_counts = packed[1]
        self._active_mask = np.asarray(active_mask, dtype=bool).reshape(-1)
        current = self._current_loss(
            global_sums.cpu().numpy(), global_counts.cpu().numpy()
        )
        weights = self._weights_from_loss(self._bin_loss(current, global_counts.cpu().numpy()))
        return global_sums, global_counts, weights

    def commit(self, global_bin_sums, global_bin_counts):
        """Fold the already-global stats into the EMA, advance ``_step``, and step
        epsilon when active-bin coverage is complete. Performs NO collective."""
        sums = np.asarray(
            global_bin_sums.cpu().numpy() if torch.is_tensor(global_bin_sums)
            else global_bin_sums, dtype=np.float64,
        ).reshape(-1)
        counts = np.asarray(
            global_bin_counts.cpu().numpy() if torch.is_tensor(global_bin_counts)
            else global_bin_counts, dtype=np.float64,
        ).reshape(-1)
        current = self._current_loss(sums, counts)
        bin_loss = self._bin_loss(current, counts)
        weights = self._weights_from_loss(bin_loss)  # uses THIS step's eps

        pop = counts > 0
        if not self._ema_initialized:
            self.ema = np.where(pop, current, self.ema)
            if pop.any():
                self._ema_initialized = True
        else:
            blended = self.ema_alpha * self.ema + (1.0 - self.ema_alpha) * current
            self.ema = np.where(pop, blended, self.ema)

        self._last_global_bin_loss = bin_loss
        self._last_bin_counts = counts.astype(int)
        self._last_weights = weights
        self._step += 1

        active = (
            self._active_mask if self._active_mask is not None
            else np.ones(self.n_bins, dtype=bool)
        )
        if self._step % self.update_every != 0 or not active.any():
            return
        if not bool(np.all(counts[active] > 0)):
            return  # hold epsilon until every active bin is populated this step
        active_idx = np.nonzero(active)[0]
        last_active = int(active_idx[-1])
        if weights[last_active] > self.last_bin_threshold:
            self.eps = min(self.eps * self.eps_growth, self.eps_max)
        elif self.allow_eps_decrease and float(weights[active].mean()) < self.mean_weight_floor:
            self.eps = max(self.eps / self.eps_growth, self.eps_min)

    def weights(self) -> np.ndarray:
        return self._last_weights

    def state_dict(self) -> dict:
        return {
            "ema": self.ema.tolist(),
            "eps": float(self.eps),
            "_step": int(self._step),
            "_ema_initialized": bool(self._ema_initialized),
            "last_global_bin_loss": (
                None if self._last_global_bin_loss is None
                else np.asarray(self._last_global_bin_loss, dtype=float).tolist()
            ),
            "last_bin_counts": (
                None if self._last_bin_counts is None
                else np.asarray(self._last_bin_counts, dtype=int).tolist()
            ),
            "n_bins": int(self.n_bins),
            "loss_history": self.loss_history,
            "ema_alpha": float(self.ema_alpha),
            "bin_edges": self.spec.bin_edges.tolist(),
        }

    def load_state_dict(self, state: dict):
        if int(state["n_bins"]) != self.n_bins:
            raise ValueError(
                f"causal_state n_bins {int(state['n_bins'])} != current "
                f"{self.n_bins}; checkpoint is incompatible with this config."
            )
        if str(state["loss_history"]) != self.loss_history:
            raise ValueError(
                f"causal_state loss_history {state['loss_history']!r} != current "
                f"{self.loss_history!r}."
            )
        if abs(float(state["ema_alpha"]) - self.ema_alpha) > 1e-12:
            raise ValueError(
                f"causal_state ema_alpha {float(state['ema_alpha'])} != current "
                f"{self.ema_alpha}."
            )
        edges = np.asarray(state["bin_edges"], dtype=float)
        if edges.shape != self.spec.bin_edges.shape or not np.allclose(
            edges, self.spec.bin_edges
        ):
            raise ValueError(
                "causal_state bin_edges do not match the current TemporalBinSpec; "
                "the lead domain or n_bins changed between runs."
            )
        self.ema = np.asarray(state["ema"], dtype=np.float64)
        self.eps = float(state["eps"])
        self._step = int(state["_step"])
        self._ema_initialized = bool(state["_ema_initialized"])
        self._last_global_bin_loss = (
            None if state.get("last_global_bin_loss") is None
            else np.asarray(state["last_global_bin_loss"], dtype=np.float64)
        )
        self._last_bin_counts = (
            None if state.get("last_bin_counts") is None
            else np.asarray(state["last_bin_counts"], dtype=int)
        )


class GradNormBalancer:
    """Per-component GradNorm-style adaptive loss balancing (Step 4 / technique 3).

    Balances the RAW (unscaled) physics/IC/data component losses so no single
    objective dominates the shared backbone gradient. Every ``update_every`` calls
    it measures each active term's gradient norm on the CURRENT step's already-built
    graph via ``torch.autograd.grad(term, unwrapped_params, retain_graph=True,
    allow_unused=True)`` -- against the UNWRAPPED module params so it never fires the
    DDP reducer (which only triggers on ``.backward()``) -- then sets
    ``w_t = mean(g_valid) / g_t`` (EMA-smoothed). The static ``lambda_*`` /
    region-weight coefficients are applied separately at the compose site, so
    GradNorm rebalances the raw magnitudes without cancelling those coefficients.

    Per-term rule when measuring ``g_t``:
      - all grads ``None`` (term disconnected)   -> hold previous multiplier
      - finite norm == 0                          -> hold previous multiplier
      - ``0 < norm < eps``                        -> clamp to ``eps``
      - ``norm > 1/eps``                          -> clamp to ``1/eps``
      - non-finite norm                           -> hold, bump warning counter
    ``mean(g)`` is over VALID terms only; held terms keep their prior multiplier.
    Valid multipliers are renormalized so their mean is ~1 (the bare
    ``mean(g)/g_t`` does not guarantee this), keeping the overall loss scale stable.

    Optional guardrails (``w_min`` / ``w_max`` / per-term ``floors``; all default to
    no-ops) bracket the EMA: the mean-normalized target is floored/clamped, then the
    EMA-blended stored weight is floored/clamped again (a clamp applied only before
    the EMA can be re-violated by the blend). No renormalization runs after the final
    clamp -- so the stored mean can drift from 1 when a bound bites; that mean is
    exposed as ``last_mean_multiplier``. Running ``bound_hit_counts`` and the
    ``last_*`` diagnostic snapshots are non-persistent (reset on resume).
    """

    def __init__(
        self, term_names, *, alpha_w=0.9, update_every=10, eps=1.0e-8,
        w_min=None, w_max=None, floors=None,
    ):
        self.term_names = list(term_names)
        self.alpha_w = float(alpha_w)
        self.update_every = max(1, int(update_every))
        self.eps = float(eps)
        self.w_min = None if w_min is None else float(w_min)
        self.w_max = None if w_max is None else float(w_max)
        self.floors = {str(k): float(v) for k, v in dict(floors or {}).items()}
        if (
            self.w_min is not None and self.w_max is not None
            and self.w_min > self.w_max
        ):
            raise ValueError(
                f"GradNormBalancer w_min ({self.w_min}) > w_max ({self.w_max})."
            )
        for name, fl in self.floors.items():
            if fl < 0.0:
                raise ValueError(
                    f"GradNormBalancer floor[{name}]={fl} must be >= 0."
                )
            if self.w_max is not None and fl > self.w_max:
                raise ValueError(
                    f"GradNormBalancer floor[{name}]={fl} > w_max ({self.w_max})."
                )
        self.multipliers = {name: 1.0 for name in self.term_names}
        self._step = 0
        self._nonfinite_count = 0
        # Running (non-persistent) diagnostics; reset on resume (NOT in state_dict).
        self.bound_hit_counts = {
            name: {"min": 0, "max": 0, "floor": 0} for name in self.term_names
        }
        self.last_raw_norms: dict = {}
        self.last_target_multipliers: dict = {}
        self.last_multipliers: dict = {}
        self.last_mean_multiplier: float = 1.0

    def _apply_bounds(self, name, v):
        """Apply per-term floor then global [w_min, w_max] clamp (each opt-in)."""
        fl = self.floors.get(name)
        if fl is not None:
            v = max(v, fl)
        if self.w_min is not None:
            v = max(v, self.w_min)
        if self.w_max is not None:
            v = min(v, self.w_max)
        return v

    def _count_bound_hit(self, name, final_v):
        """Count once per term per update if the FINAL stored value sits at a bound."""
        fl = self.floors.get(name)
        if fl is not None and fl > 0.0 and final_v == fl:
            self.bound_hit_counts[name]["floor"] += 1
        elif self.w_max is not None and final_v == self.w_max:
            self.bound_hit_counts[name]["max"] += 1
        elif self.w_min is not None and final_v == self.w_min:
            self.bound_hit_counts[name]["min"] += 1

    def multipliers_for(self, names) -> dict:
        return {name: float(self.multipliers.get(name, 1.0)) for name in names}

    @staticmethod
    def _grad_norm(term, params):
        grads = torch.autograd.grad(
            term, params, retain_graph=True, allow_unused=True,
        )
        sq = None
        for g in grads:
            if g is None:
                continue
            contrib = g.detach().pow(2).sum()
            sq = contrib if sq is None else sq + contrib
        if sq is None:
            return None  # term disconnected from every param this step
        return sq

    def maybe_update(self, raw_terms, params, *, dist_info=None):
        """Refresh multipliers from the current graph every ``update_every`` calls.

        ``raw_terms`` maps term name -> the differentiable RAW loss scalar. ``params``
        is the UNWRAPPED module parameter iterable. Returns the (possibly unchanged)
        multiplier dict for ``raw_terms``' keys.
        """
        names = [n for n in self.term_names if n in raw_terms]
        do_update = (self._step % self.update_every == 0)
        self._step += 1
        if not do_update or not names:
            return self.multipliers_for(names)

        params = [p for p in params if p.requires_grad]
        device = params[0].device if params else torch.device("cpu")
        norm_dtype = torch.float32 if device.type == "mps" else torch.float64
        sq_norms = torch.zeros(len(names), dtype=norm_dtype, device=device)
        held = [False] * len(names)
        for i, name in enumerate(names):
            sq = self._grad_norm(raw_terms[name], params)
            if sq is None:
                held[i] = True  # disconnected -> hold
            else:
                sq_norms[i] = sq.to(norm_dtype)
        if dist_info is not None and getattr(dist_info, "is_distributed", False):
            # gloo has no AVG; SUM then divide by world_size for a mean sq-norm.
            dist.all_reduce(sq_norms, op=dist.ReduceOp.SUM)
            sq_norms = sq_norms / float(getattr(dist_info, "world_size", 1) or 1)

        norms = sq_norms.sqrt().cpu().numpy()
        hi = 1.0 / self.eps
        valid, clamped = [], {}
        self.last_raw_norms = {}
        for i, name in enumerate(names):
            g = float(norms[i])
            if held[i]:
                self.last_raw_norms[name] = None  # disconnected -> no measured norm
                continue
            self.last_raw_norms[name] = g
            if not np.isfinite(g):
                self._nonfinite_count += 1  # hold
                continue
            if g == 0.0:
                continue  # hold
            clamped[name] = float(np.clip(g, self.eps, hi))
            valid.append(name)

        if not valid:
            self.last_target_multipliers = {}
            self.last_multipliers = self.multipliers_for(names)
            return self.multipliers_for(names)
        mean_g = float(np.mean([clamped[n] for n in valid]))
        target = {n: mean_g / clamped[n] for n in valid}
        # Renormalize valid multipliers so their mean is ~1 (scale stability),
        # THEN clamp/floor the TARGET so the guardrails bracket the EMA (a clamp
        # applied only before the EMA can be re-violated by the EMA blend).
        mean_t = float(np.mean([target[n] for n in valid]))
        if mean_t > 0.0:
            target = {n: v / mean_t for n, v in target.items()}
        target = {n: self._apply_bounds(n, v) for n, v in target.items()}
        self.last_target_multipliers = dict(target)
        for name in valid:
            prev = float(self.multipliers.get(name, 1.0))
            blended = self.alpha_w * prev + (1.0 - self.alpha_w) * target[name]
            # Clamp/floor the STORED weight again (do NOT renormalize afterwards:
            # renormalizing could re-violate the bounds).
            final_v = self._apply_bounds(name, blended)
            self.multipliers[name] = final_v
            self._count_bound_hit(name, final_v)
        stored_vals = [float(self.multipliers[n]) for n in valid]
        self.last_mean_multiplier = float(np.mean(stored_vals)) if stored_vals else 1.0
        self.last_multipliers = self.multipliers_for(names)
        return self.multipliers_for(names)

    def state_dict(self) -> dict:
        return {
            "term_names": list(self.term_names),
            "multipliers": {k: float(v) for k, v in self.multipliers.items()},
            "_step": int(self._step),
        }

    def load_state_dict(self, state: dict):
        saved = list(state["term_names"])
        if saved != self.term_names:
            raise ValueError(
                f"gradnorm_state term_names {saved} != current {self.term_names}; "
                "the active loss components changed between runs "
                "(hard-BC / IC / data / benchmark). Refusing to reassign "
                "multipliers to the wrong objective."
            )
        # Re-apply floor+clamp so a checkpoint from a bounds-free run cannot
        # reintroduce out-of-bounds weights under newly enabled guardrails.
        self.multipliers = {
            k: self._apply_bounds(k, float(v))
            for k, v in dict(state["multipliers"]).items()
        }
        self._step = int(state["_step"])


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
    the no-op guarantee are untouched). Under DDP the caller passes a
    rank-offset ``rng_seed`` so each rank draws a disjoint collocation batch
    (the DDP-averaged physics gradient then covers world_size x distinct
    samples; see ``run_one_seed``).
    """

    def __init__(
        self, ds, spec, geom_cfg, *, batch_size, dt, rng_seed, base_plan=None,
        anchored_first_step=False, early_oversample=False, early_band_steps=5,
        early_mix=None, stratify_leads=False,
        source_time_min=None, source_time_max=None, lead_bins=None,
        lead_min=None, anchor_lead_time=None, anchor_fraction=0.0,
    ):
        self.ds = ds
        self.spec = spec
        self.geom_cfg = geom_cfg
        self.batch_size = int(batch_size)
        self.dt = float(dt)
        self.t_final = float(ds.t_final)
        self.sim_ids = np.asarray(ds.sim_ids)
        self.rng = np.random.default_rng(rng_seed)

        # ---- long-lead OOD generic-path knobs (forcing collocation) ----
        # All default to the legacy behavior (random source, uniform lead over
        # [dt, max_lead], no bridge rows), so the generic path is byte-identical
        # when these are unset.
        self.source_time_min = (
            float(source_time_min) if source_time_min is not None else None
        )
        self.source_time_max = (
            float(source_time_max) if source_time_max is not None else None
        )
        self.lead_bins = int(lead_bins) if lead_bins else None
        self.lead_min = float(lead_min) if lead_min is not None else None
        self.anchor_lead_time = (
            float(anchor_lead_time) if anchor_lead_time is not None else None
        )
        self.anchor_fraction = float(anchor_fraction)
        # Candidate source snapshots (full-grid indices) restricted to the
        # configured source-time range; their grid times drive feasibility. Built
        # only for the generic (forcing) path -- the on-grid path (base_plan set)
        # draws sources through the spec hooks and never reads t_indices, so its
        # datasets need not expose one.
        self._src_pool = None
        self._src_pool_t = None
        self._src_pool_t_min = None
        self._full_t_grid = None
        if base_plan is None:
            src_idx = np.asarray(list(ds.t_indices), dtype=int)
            src_t = np.asarray([float(ds.t_grid[s]) for s in src_idx], dtype=float)
            keep = np.ones(src_idx.shape, dtype=bool)
            tol = 1e-9
            if self.source_time_min is not None:
                keep &= src_t >= self.source_time_min - tol
            if self.source_time_max is not None:
                keep &= src_t <= self.source_time_max + tol
            self._src_pool = src_idx[keep]
            self._src_pool_t = src_t[keep]
            if self._src_pool.size == 0:
                raise ValueError(
                    "CollocationSampler: no source snapshot falls in the range "
                    f"[{self.source_time_min}, {self.source_time_max}]."
                )
            self._src_pool_t_min = float(self._src_pool_t.min())
            # Full grid for bridge-snapshot lookup (nearest index to t_s+anchor).
            self._full_t_grid = np.asarray(ds.t_grid, dtype=float)
        # Per-batch record of the leads actually drawn, for the collocation lead
        # histogram diagnostic. None until the first draw. Both paths populate it
        # (the on-grid path records the CN interval midpoint per row).
        self._last_leads = None
        # Per-batch bin ids for the drawn leads, via ``bin_spec.interval_to_bin``.
        # Stays None (and nothing else changes -> bit-identical) when no
        # ``TemporalBinSpec`` is attached, i.e. when all Step 1/2 techniques are
        # off. Consumed by CausalLeadWeighter so it never re-derives bins.
        self._last_lead_bin_ids = None
        # Shared temporal-bin definition (attached by run_one_seed). None when the
        # causal/stratify techniques are all disabled.
        self.bin_spec = None
        # When non-None, the benchmark pins the base snapshot and draws on-grid
        # conditioning pairs through the spec's collocation hooks instead of the
        # built-in (forcing) random-source / uniform-lead / analytic-closure path.
        self.base_plan = base_plan
        # ---- causal anchored physics-only knobs (on-grid path only) ----
        self.anchored_first_step = bool(anchored_first_step)
        self.early_oversample = bool(early_oversample)
        self.early_band_steps = int(early_band_steps)
        # Block-structured lead stratification (on-grid path only). When on, each
        # batch draws B // n_active base rows and, for every base row, one CN
        # interval per (populated) shared-``bin_spec`` bin -- so every bin sees the
        # SAME sim-id / base-snapshot multiset and a later bin's higher residual
        # reflects longer lead, not a harder set of input functions. Requires an
        # attached ``bin_spec`` (the shared TemporalBinSpec); raises otherwise.
        self.stratify_leads = bool(stratify_leads)
        self.early_mix = dict(early_mix) if early_mix is not None else {
            "anchor": 0.4, "early": 0.4, "rest": 0.2,
        }
        # Side-channel consumed by _collocation_batch_loss: after an anchored
        # on-grid draw this holds (mask (B,), T0_exact (B,Nx,Ny)) torch tensors
        # marking the n==0 rows whose T_n must be overridden with the exact IC;
        # None on any non-anchored / generic draw. Reset at the top of every
        # sample_batch / _sample_batch_on_grid call so a stale anchor from a
        # prior batch can never be reused.
        self._last_anchor = None
        # ---- warm-up freeze cache (IC-anchoring warm-up; default inert) ----
        # When ``_freeze_enabled`` (set once per epoch by train_one_epoch during
        # warm-up), each fresh draw is cached with its side-channels so that a
        # subsequent ``_frozen`` step reuses the SAME input functions and query
        # coordinates instead of resampling. Both flags stay False when warm-up is
        # off, so ``sample_batch`` is a pure passthrough (bit-identical) and never
        # touches the cache. The cache is epoch-local (reset via
        # ``reset_freeze_cache``); the persistent progress counter lives in
        # run_one_seed as ``global_optimizer_step``.
        self._freeze_enabled = False
        self._frozen = False
        self._batch_cache = None
        self._cache_leads = None
        self._cache_bin_ids = None
        self._cache_anchor = None

    def set_frozen(self, frozen: bool):
        """Warm-up per-step control: when ``frozen`` and a cached draw exists, the
        next ``sample_batch`` reuses it (and its side-channels) instead of drawing
        fresh. No-op unless ``_freeze_enabled`` (set by ``enable_freeze_cache``)."""
        self._frozen = bool(frozen) and self._freeze_enabled

    def enable_freeze_cache(self, enabled: bool):
        """Arm/disarm the warm-up freeze cache for the coming epoch. Always drops
        any cached batch first, so the cache is epoch-local: a fresh epoch never
        reuses the previous epoch's frozen draw (deterministic epoch-boundary
        resume without checkpointing the batch)."""
        self.reset_freeze_cache()
        self._freeze_enabled = bool(enabled)

    def reset_freeze_cache(self):
        """Drop the epoch-local cached batch and un-freeze (epoch boundary reset)."""
        self._frozen = False
        self._batch_cache = None
        self._cache_leads = None
        self._cache_bin_ids = None
        self._cache_anchor = None

    def sample_batch(self, max_lead: float):
        """Sample one collocation batch, with an optional warm-up freeze cache.

        Warm-up (Step 5): when ``_frozen`` and a cached draw exists, return the
        cached tuple and restore the cached side-channels (``_last_leads``,
        ``_last_lead_bin_ids``, ``_last_anchor``) so consecutive frozen steps see an
        identical batch. Otherwise draw fresh via ``_draw_collocation_batch`` and,
        when the cache is armed, store the draw + its side-channels. Both flags are
        False when warm-up is off -> pure passthrough, bit-identical, no caching.
        """
        if self._frozen and self._batch_cache is not None:
            self._last_leads = self._cache_leads
            self._last_lead_bin_ids = self._cache_bin_ids
            self._last_anchor = self._cache_anchor
            return self._batch_cache
        result = self._draw_collocation_batch(max_lead)
        if self._freeze_enabled:
            self._batch_cache = result
            self._cache_leads = self._last_leads
            self._cache_bin_ids = self._last_lead_bin_ids
            self._cache_anchor = self._last_anchor
        return result

    def _draw_collocation_batch(self, max_lead: float):
        """Sample one collocation batch (generic forcing path).

        Legacy behavior (all long-lead knobs unset): the source snapshot is drawn
        uniformly from those whose ``t_s + max_lead + dt <= t_final`` and the lead
        uniformly over ``[dt, max_lead]`` -- byte-identical to before.

        Long-lead OOD knobs layer on three behaviors (defaults preserve legacy):
        - ``source_time_min/max`` restrict the source pool (``_src_pool``) so
          collocation is anchored near early sources that can support long leads.
        - ``lead_min`` raises the non-bridge lead lower bound into the unlabeled
          band ``tau > tc``; ``lead_bins`` switches from source-first draws to
          lead-bin-first draws (a lead bin is chosen uniformly, then a feasible
          source), giving the longest leads equal collocation mass instead of the
          source-first under-sampling of ``tau ~ t_final``.
        - ``anchor_lead_time``/``anchor_fraction`` inject *bridge* rows at
          ``lead = tc``: the residual's first state is later overwritten (in
          ``_collocation_batch_loss``) with the exact true snapshot nearest
          ``t_s + tc`` via ``self._last_anchor``, connecting the labeled boundary
          into the first unlabeled step.

        Hard-asserts each ``t + dt`` stays within the forcing support ``t_final``
        and records the drawn leads in ``self._last_leads`` for the histogram
        diagnostic.
        """
        if self.base_plan is not None:
            return self._sample_batch_on_grid(max_lead)

        # Reset side-channels; only populated below if bridge rows are drawn.
        self._last_anchor = None
        ds = self.ds
        dt = self.dt
        tol = 1e-9
        max_lead = max(float(max_lead), dt)

        pool_idx = self._src_pool
        pool_t = self._src_pool_t
        lead_lo = dt if self.lead_min is None else max(dt, float(self.lead_min))
        if lead_lo > max_lead + tol:
            raise ValueError(
                f"collocation_lead_min={self.lead_min} exceeds the active max "
                f"lead {max_lead}; widen collocation_lead_max/warmup."
            )
        use_bridge = (
            self.anchor_lead_time is not None and self.anchor_fraction > 0.0
        )
        anchor_lead = float(self.anchor_lead_time) if use_bridge else None

        # Source-first candidates depend only on max_lead -> precompute once so
        # the legacy (no-knob) path draws exactly as before.
        sf_cand = pool_idx[pool_t + max_lead + dt <= self.t_final + tol]
        if not use_bridge and sf_cand.size == 0:
            raise ValueError(
                f"No source in range supports collocation max_lead={max_lead} + "
                f"dt={dt} within t_final={self.t_final}; lower collocation_lead_max "
                f"or widen collocation_source_time_max."
            )

        items_t, items_tdt = [], []
        R_c_list, qLn_list, qLnp1_list, qLint_list = [], [], [], []
        anchor_flags, T0_list = [], []
        leads_drawn = []
        for _ in range(self.batch_size):
            sid = int(self.rng.choice(self.sim_ids))
            is_bridge = bool(
                use_bridge and self.rng.random() < self.anchor_fraction
            )
            if is_bridge:
                lead = anchor_lead
                feas = pool_t + lead + dt <= self.t_final + tol
                cand = pool_idx[feas]
                if cand.size == 0:
                    raise ValueError(
                        f"No source in range supports bridge lead {lead} + dt "
                        f"within t_final={self.t_final}; lower anchor_lead_time or "
                        f"widen collocation_source_time_max."
                    )
                s = int(self.rng.choice(cand))
            elif self.lead_bins is not None:
                # Lead-bin-first: equal mass per lead bin over [lead_lo, max_lead],
                # then a feasible source for the drawn lead.
                edges = np.linspace(lead_lo, max_lead, self.lead_bins + 1)
                bi = int(self.rng.integers(0, self.lead_bins))
                lead = float(self.rng.uniform(edges[bi], edges[bi + 1]))
                cand = pool_idx[pool_t + lead + dt <= self.t_final + tol]
                if cand.size == 0:
                    raise ValueError(
                        f"No source in range supports lead={lead} + dt within "
                        f"t_final={self.t_final}."
                    )
                s = int(self.rng.choice(cand))
            else:
                # Source-first (legacy shape): source then uniform lead.
                s = int(self.rng.choice(sf_cand))
                lead = (
                    float(self.rng.uniform(lead_lo, max_lead))
                    if max_lead > lead_lo else lead_lo
                )

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
            # Exact step integral of the separable flux, matching the solver's
            # `q_left_integral` injection (boundary_forcing.build_qL_integral):
            # the ramped temporal integral times the spatial profile s(y).
            a_int = integrate_temporal_ramped_signed(
                params["temporal_family"], params["temporal_params"],
                t, t_dt, float(ds.ramp_seconds),
            )
            qLint = (float(a_int) * s_y).astype(np.float32)

            items_t.append({k: torch.from_numpy(v) for k, v in item_t.items()})
            items_tdt.append({k: torch.from_numpy(v) for k, v in item_tdt.items()})
            R_c_list.append(float(params["R_c"]))
            qLn_list.append(qLn)
            qLnp1_list.append(qLnp1)
            qLint_list.append(qLint)
            leads_drawn.append(float(lead))

            if is_bridge:
                # Exact normalized true snapshot nearest t = t_s + tc, in the same
                # (mu_global, sigma_global) space the model outputs. Reuses the
                # spec's build_item so the normalization matches _sample_batch_on_grid.
                k = int(np.argmin(np.abs(self._full_t_grid - t)))
                anchor_item = ds.problem.build_item(ds, sid, k, k)
                anchor_T0 = anchor_item["spatial"][..., 0]
                anchor_flags.append(True)
                T0_list.append(np.asarray(anchor_T0, dtype=np.float32))
            else:
                anchor_flags.append(False)
                T0_list.append(None)

        batch_t = collate_fn(items_t)
        batch_tdt = collate_fn(items_tdt)
        R_c = torch.tensor(R_c_list, dtype=torch.float32)
        qL_n = torch.from_numpy(np.stack(qLn_list))
        qL_np1 = torch.from_numpy(np.stack(qLnp1_list))
        qL_int = torch.from_numpy(np.stack(qLint_list))
        self._last_leads = np.asarray(leads_drawn, dtype=float)
        self._last_lead_bin_ids = (
            self.bin_spec.interval_to_bin(self._last_leads)
            if self.bin_spec is not None else None
        )
        if any(anchor_flags):
            # Non-bridge rows are zero-filled placeholders shaped like a real
            # bridge snapshot; the injection (`_collocation_batch_loss`) only
            # reads the masked (bridge) rows.
            ref = next(t0 for t0 in T0_list if t0 is not None)
            zeros = np.zeros_like(ref)
            mask = torch.tensor(anchor_flags, dtype=torch.bool)
            T0_exact = torch.from_numpy(
                np.stack([zeros if t0 is None else t0 for t0 in T0_list])
            )
            self._last_anchor = (mask, T0_exact)
        return batch_t, batch_tdt, R_c, qL_n, qL_np1, qL_int

    def _early_buckets(self, n_max: int):
        """Return the (label, lo, hi) inclusive integer ranges over ``[0, n_max]``
        for the early-time oversampling mixture, dropping empty buckets. Uses the
        TARGET-time convention: a pair ``n->n+1`` has target ``(n+1)*dt``, so the
        band ``early_band_steps`` (a target bound) maps to start indices:
        anchor = {0}, early = {1..min(band-1, n_max)}, rest = {band..n_max}.
        """
        band = self.early_band_steps
        buckets = [("anchor", 0, 0)]
        early_hi = min(band - 1, n_max)
        if early_hi >= 1:
            buckets.append(("early", 1, early_hi))
        if band <= n_max:
            buckets.append(("rest", band, n_max))
        return buckets

    def _draw_n_on_grid(self, n_max: int) -> int:
        """Draw a start index ``n`` in ``[0, n_max]``. With early oversampling,
        sample a bucket from ``early_mix`` (renormalized over non-empty buckets)
        then a uniform ``n`` within it; otherwise uniform over ``[0, n_max]``.
        """
        if not self.early_oversample:
            return int(self.rng.integers(0, n_max + 1))
        buckets = self._early_buckets(n_max)
        weights = np.array(
            [max(float(self.early_mix.get(label, 0.0)), 0.0) for label, _, _ in buckets],
            dtype=float,
        )
        total = weights.sum()
        if total <= 0.0:  # degenerate mix -> fall back to uniform over buckets
            weights = np.ones(len(buckets), dtype=float)
            total = weights.sum()
        weights = weights / total
        bi = int(self.rng.choice(len(buckets), p=weights))
        _, lo, hi = buckets[bi]
        return int(self.rng.integers(lo, hi + 1))

    def _stratified_draws_on_grid(self, n_max: int):
        """Block-structured (sid, n) draws for one batch when ``stratify_leads``.

        Chunks ``[0, n_max]`` by the shared ``bin_spec`` so a chunk id is exactly
        the bin the row's midpoint lands in (``interval_to_bin``), not a parallel
        ``linspace`` that could drift from the causal bins. Empty bins (windows
        with fewer valid intervals than bins, e.g. under a curriculum cap) are
        skipped. Draws ``B // n_active`` base rows and, per base row, one interval
        from every populated bin (same sid across bins -> identical per-bin sim-id
        multiset); the ``B % n_active`` remainder rows fill the earliest populated
        bins with fresh sids. Single ``self.rng`` stream, so DDP ranks stay in
        lockstep given their rank-offset seed.
        """
        if self.bin_spec is None:
            raise RuntimeError(
                "stratify_leads requires an attached TemporalBinSpec (bin_spec); "
                "run_one_seed attaches it when stratify_leads/causal_weighting is on."
            )
        n_bins = int(self.bin_spec.n_bins)
        all_n = np.arange(n_max + 1)
        all_bin = np.asarray(self.bin_spec.interval_to_bin((all_n + 0.5) * self.dt))
        chunks = [all_n[all_bin == b] for b in range(n_bins)]
        active = [b for b in range(n_bins) if chunks[b].size > 0]
        n_active = len(active)
        n_base = self.batch_size // n_active
        rem = self.batch_size % n_active
        draws = []
        for _ in range(n_base):
            sid = int(self.rng.choice(self.sim_ids))
            for b in active:
                draws.append((sid, int(self.rng.choice(chunks[b]))))
        for j in range(rem):
            b = active[j]
            sid = int(self.rng.choice(self.sim_ids))
            draws.append((sid, int(self.rng.choice(chunks[b]))))
        return draws

    def _sample_batch_on_grid(self, max_lead: float):
        """Base-plan collocation: pin the base snapshot, draw consecutive on-grid
        conditioning pairs ``(t_s + n*dt, t_s + (n+1)*dt)``, and read the boundary
        closure from the spec hook. Both forwards share the same base state.

        The second prediction is capped at ``max_lead``: ``n`` is drawn in
        ``[0, n_max]`` with ``n_max = min(floor(max_lead/dt) - 1, len(t_grid) - 2)``
        so ``t_grid[n+1]`` always exists (the ``- 1`` is the target-time off-by-one:
        a pair ``n->n+1`` has target ``(n+1)*dt``, so a window through
        ``max_lead`` admits start indices up to ``round(max_lead/dt) - 1``).

        When ``early_oversample`` the draw uses the 3-bucket mixture; when
        ``anchored_first_step`` the ``n==0`` rows are flagged in
        ``self._last_anchor`` so the batch loss can hard-inject the exact IC as
        ``T_n`` on those rows.
        """
        # Side-channel hygiene: clear before (re)building so a raised exception or
        # a non-anchored draw never leaves a stale mask behind.
        self._last_anchor = None
        ds = self.ds
        dt = self.dt
        bp = self.base_plan
        s = int(bp.get("base_snapshot_index", 0))
        t_s = float(ds.t_grid[s])
        t_grid = np.asarray(ds.t_grid, dtype=float)
        n_max = max(0, min(int(math.floor(float(max_lead) / dt)) - 1, len(t_grid) - 2))

        draws = (
            self._stratified_draws_on_grid(n_max) if self.stratify_leads
            else [
                (int(self.rng.choice(self.sim_ids)), self._draw_n_on_grid(n_max))
                for _ in range(self.batch_size)
            ]
        )

        items_t, items_tdt = [], []
        R_c_list, qLn_list, qLnp1_list, qLint_list = [], [], [], []
        anchor_flags, T0_list = [], []
        lead_mid_list = []
        for sid, n in draws:
            t = t_s + n * dt
            t_dt = t + dt
            lead_mid_list.append((n + 0.5) * dt)

            base_item = ds.problem.build_item(ds, sid, s, s)
            current = base_item["spatial"][..., 0]
            item_t = build_rollout_item_from_base(
                base_item, ds, self.spec, sid, current, t_s, t
            )
            item_tdt = build_rollout_item_from_base(
                base_item, ds, self.spec, sid, current, t_s, t_dt
            )

            params = ds.sim_params[sid]
            R_c_val, qLn, qLnp1, qLint = self.spec.collocation_closure(
                ds, sid, params, t, t_dt
            )

            items_t.append({k: torch.from_numpy(v) for k, v in item_t.items()})
            items_tdt.append({k: torch.from_numpy(v) for k, v in item_tdt.items()})
            R_c_list.append(float(R_c_val))
            qLn_list.append(np.asarray(qLn, dtype=np.float32))
            qLnp1_list.append(np.asarray(qLnp1, dtype=np.float32))
            qLint_list.append(np.asarray(qLint, dtype=np.float32))
            anchor_flags.append(bool(self.anchored_first_step and n == 0))
            T0_list.append(np.asarray(current, dtype=np.float32))

        batch_t = collate_fn(items_t)
        batch_tdt = collate_fn(items_tdt)
        R_c = torch.tensor(R_c_list, dtype=torch.float32)
        qL_n = torch.from_numpy(np.stack(qLn_list))
        qL_np1 = torch.from_numpy(np.stack(qLnp1_list))
        qL_int = torch.from_numpy(np.stack(qLint_list))
        # Record the per-row CN interval-midpoint lead (t_n + t_{n+1})/2 = (n+0.5)*dt
        # in draw order (this path previously set neither field -- the pre-existing
        # bug that left phys_lead_hist empty for diffusion runs). Bin ids follow
        # from the shared TemporalBinSpec when one is attached.
        self._last_leads = np.asarray(lead_mid_list, dtype=float)
        self._last_lead_bin_ids = (
            self.bin_spec.interval_to_bin(self._last_leads)
            if self.bin_spec is not None else None
        )
        if self.anchored_first_step:
            mask = torch.tensor(anchor_flags, dtype=torch.bool)
            T0_exact = torch.from_numpy(np.stack(T0_list))
            self._last_anchor = (mask, T0_exact)
        return batch_t, batch_tdt, R_c, qL_n, qL_np1, qL_int

    def sample_anchor_batch(self):
        """IC-anchor batch: the model conditioned at ``(t_s, t_s)`` must reproduce
        the base state ``T(t_s)``. Pins the same base snapshot as the residual
        pair (``base_snapshot_index``); the zero-duration conditioning is NaN-safe
        (the token grid is ``linspace``-based and ``t_bar`` only multiplies).
        Returns the collated anchor batch and the per-item target field ``T(t_s)``.
        Only valid when a ``base_plan`` is present.
        """
        ds = self.ds
        bp = self.base_plan or {}
        s = int(bp.get("base_snapshot_index", 0))
        t_s = float(ds.t_grid[s])

        items, targets = [], []
        for _ in range(self.batch_size):
            sid = int(self.rng.choice(self.sim_ids))
            base_item = ds.problem.build_item(ds, sid, s, s)
            current = base_item["spatial"][..., 0]
            item = build_rollout_item_from_base(
                base_item, ds, self.spec, sid, current, t_s, t_s
            )
            items.append({k: torch.from_numpy(v) for k, v in item.items()})
            targets.append(current[..., None].astype(np.float32))

        batch = collate_fn(items)
        target = torch.from_numpy(np.stack(targets))
        return batch, target


def _resolve_causal_stages(cc_cfg, dt: float, t_final: float) -> list[dict]:
    """Resolve the fixed staged causal curriculum into ``[{epochs, max_lead}]``.

    Each stage carries a cumulative-relative ``epochs`` count (``-1`` = the
    remainder) and exactly one of ``max_lead_steps`` (int * dt) or
    ``max_lead_frac`` (fraction of t_final); both express the max TARGET time of
    the sampled window. Raises if a stage specifies neither or both.
    """
    stages = list(cc_cfg.get("stages", []) or [])
    resolved: list[dict] = []
    for st in stages:
        ep = int(st.get("epochs", -1))
        steps = st.get("max_lead_steps", None)
        frac = st.get("max_lead_frac", None)
        if (steps is None) == (frac is None):
            raise ValueError(
                "each causal_curriculum stage needs exactly one of "
                f"max_lead_steps / max_lead_frac; got steps={steps}, frac={frac}."
            )
        if steps is not None:
            lead = float(int(steps)) * float(dt)
        else:
            lead = float(frac) * float(t_final)
        resolved.append({"epochs": ep, "max_lead": lead})
    if not resolved:
        raise ValueError("causal_curriculum.enabled but no stages were provided.")
    return resolved


def _causal_max_lead(resolved: list[dict], epoch: int) -> float:
    """Pick the max lead (target time) for a 0-indexed ``epoch`` from the resolved
    cumulative stage schedule. A stage with ``epochs < 0`` is the remainder and
    applies to every epoch at or beyond its start.
    """
    cum = 0
    for st in resolved:
        ep = int(st["epochs"])
        if ep < 0:
            return float(st["max_lead"])
        cum += ep
        if epoch < cum:
            return float(st["max_lead"])
    return float(resolved[-1]["max_lead"])


def _build_collocation_sampler(config, phys_cfg, ds, spec, mu_global, sigma_global,
                               *, batch_size, rng_seed):
    """Build the W2 collocation sampler over the held-out (validation) sims.

    Geometry, the per-item boundary closure, and the base-snapshot plan are
    routed through the benchmark's ``ProblemSpec`` collocation hooks. When a hook
    returns ``None`` the built-in (``forcing``) default is used: two-slab
    ``k=2/k=1`` interface at ``x=0.5``, scalar ``R_c``, constant right-edge
    ``T_right = 300 K`` (the Dirichlet target, normalized with the training
    stats), with the physics step ``dt`` taken from ``physics.dt`` (default
    0.005). A benchmark that opts into ``on_grid_pairs`` pins ``dt`` to the FV
    grid spacing so geometry, sampler, and residual all share one ``dt``.
    """
    base_plan = spec.collocation_base_plan(ds, float(phys_cfg.get("dt", 0.005)))
    if base_plan is not None and base_plan.get("on_grid_pairs", False):
        # t_grid is persisted as float32, so a genuinely uniform grid jitters by
        # the float32 quantum near t_final (~3e-8 at t=0.3). Compare to the mean
        # spacing with a tolerance that tolerates that quantization yet still
        # rejects a real save_stride mismatch (e.g. 0.01 vs 0.02).
        t_grid = np.asarray(ds.t_grid, dtype=np.float64)
        diffs = np.diff(t_grid)
        dt_mean = float(diffs.mean()) if diffs.size else 0.0
        if diffs.size == 0 or not np.allclose(diffs, dt_mean, rtol=1e-3, atol=1e-9):
            mn = float(diffs.min()) if diffs.size else float("nan")
            mx = float(diffs.max()) if diffs.size else float("nan")
            raise ValueError(
                "collocation_base_plan requested on_grid_pairs but the dataset "
                "t_grid spacing is non-uniform; cannot pin dt to the FV grid "
                f"step (min={mn}, max={mx}, mean={dt_mean})."
            )
        dt = dt_mean
    else:
        dt = float(phys_cfg.get("dt", 0.005))

    geom_cfg = spec.collocation_geom_cfg(ds, phys_cfg, mu_global, sigma_global, dt)
    if geom_cfg is None:
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
    # The anchor/oversample knobs are only meaningful for the on-grid pair path;
    # leave them off for the generic (forcing) sampler so that path is unchanged.
    on_grid = bool(base_plan is not None and base_plan.get("on_grid_pairs", False))
    anchored = bool(on_grid and phys_cfg.get("anchored_first_step", False))
    early_os = bool(on_grid and phys_cfg.get("early_time_oversample", False))
    early_band = int(phys_cfg.get("early_band_steps", 5))
    # Block-structured lead stratification is on-grid-only. Fail loudly on the
    # generic (forcing) path rather than silently no-op'ing a misconfigured sweep;
    # forcing gets long-lead coverage from its own generic-path knobs.
    stratify_req = bool(phys_cfg.get("stratify_leads", False))
    if stratify_req and not on_grid:
        raise ValueError(
            "stratify_leads is on-grid-only (needs a benchmark whose "
            "collocation_base_plan yields on_grid_pairs=True). For generic/forcing "
            "collocation coverage use collocation_lead_bins / collocation_lead_min / "
            "anchor_lead_time / anchor_fraction instead."
        )
    if stratify_req and early_os:
        raise ValueError(
            "stratify_leads and early_time_oversample are mutually exclusive "
            "temporal-coverage mechanisms; intended diffusion usage is "
            "early_time_oversample=false stratify_leads=true."
        )
    early_mix = phys_cfg.get("early_mix", None)
    if early_mix is not None:
        early_mix = dict(early_mix)
    # Long-lead OOD knobs apply only to the generic (forcing) path; gate them off
    # for on-grid benchmarks so those samplers stay byte-identical.
    src_min = None if on_grid else phys_cfg.get("collocation_source_time_min", None)
    src_max = None if on_grid else phys_cfg.get("collocation_source_time_max", None)
    lead_bins = None if on_grid else phys_cfg.get("collocation_lead_bins", None)
    lead_min = None if on_grid else phys_cfg.get("collocation_lead_min", None)
    anchor_lead = None if on_grid else phys_cfg.get("anchor_lead_time", None)
    anchor_frac = 0.0 if on_grid else float(phys_cfg.get("anchor_fraction", 0.0))
    return CollocationSampler(
        ds, spec, geom_cfg, batch_size=pbs, dt=dt, rng_seed=rng_seed,
        base_plan=base_plan,
        anchored_first_step=anchored, early_oversample=early_os,
        early_band_steps=early_band, early_mix=early_mix,
        stratify_leads=stratify_req,
        source_time_min=src_min, source_time_max=src_max, lead_bins=lead_bins,
        lead_min=lead_min, anchor_lead_time=anchor_lead, anchor_fraction=anchor_frac,
    )


def _make_temporal_bin_spec(sampler, phys_cfg, n_bins):
    """Build the one shared ``TemporalBinSpec`` (Step 0) for a built sampler.

    Path-aware lead domain so the same object bins the on-grid (diffusion) CN
    interval midpoints and the generic (forcing) continuous leads:
    - on-grid: ``[0.5*dt, t_final - 0.5*dt]`` -- exactly the span of the valid
      interval midpoints ``(n + 0.5)*dt`` over the full window.
    - generic: ``[collocation_lead_min or dt, collocation_lead_max or t_final]``.
    Attached to BOTH the sampler (stratified allocation + ``_last_lead_bin_ids``)
    and the ``CausalLeadWeighter`` so techniques 1 and 2 never drift apart.
    """
    dt = float(sampler.dt)
    t_final = float(sampler.t_final)
    on_grid = bool(
        sampler.base_plan is not None
        and sampler.base_plan.get("on_grid_pairs", False)
    )
    if on_grid:
        lead_lo = 0.5 * dt
        lead_hi = t_final - 0.5 * dt
    else:
        lead_min = phys_cfg.get("collocation_lead_min", None)
        lead_lo = float(lead_min) if lead_min is not None else dt
        lead_max = phys_cfg.get("collocation_lead_max", None)
        lead_hi = float(lead_max) if lead_max is not None else t_final
    return TemporalBinSpec(lead_lo=lead_lo, lead_hi=lead_hi, n_bins=int(n_bins))


def _collocation_batch_loss(model, sampler, max_lead, region_weights, device,
                            lambda_ic=0.0, causal_weighter=None, dist_info=None,
                            world_size=1):
    """Forward the model at the collocation pair ``(t, t+dt)`` and return the
    region-partitioned ``full_bc`` physics loss dict, the batch size, and the
    optional IC-anchor loss.

    Both fields are model outputs from the same base input, so the right-edge
    Dirichlet closure is anchored at BOTH times (``dirichlet_both_ends``). When
    ``lambda_ic > 0`` an IC-anchor batch is drawn from the same base snapshot and
    ``ic_loss = MSE(model(anchor), T(t_s))`` is returned; otherwise ``ic_loss``
    is ``None``.

    When ``causal_weighter`` is set the residual is computed per-sample, the
    interior residual is binned by lead (``sampler._last_lead_bin_ids``), and the
    weighter's ``prepare`` phase SUM-all-reduces the per-bin sums/counts (the only
    collective) so a DDP-safe differentiable causal interior objective can be
    formed from LOCAL differentiable sums divided by GLOBAL detached counts. The
    weighter is NOT mutated here (safe for eval / grad-accum / retries); the
    caller ``commit``s the EMA/epsilon after backward using the detached global
    stats returned in ``out``.
    """
    batch_t, batch_tdt, R_c, qL_n, qL_np1, qL_int = sampler.sample_batch(max_lead)

    def _fwd(b):
        spatial = b["spatial"].to(device)
        cond = b["cond_static"].to(device)
        fseq = b["forcing_seq"].to(device) if "forcing_seq" in b else None
        return model(spatial, cond, fseq)

    yt = _fwd(batch_t)
    ytdt = _fwd(batch_tdt)

    # ---- anchored first step: hard-inject the exact IC as T_n on n==0 rows ----
    # The side-channel is reset to None at the start of every sample_batch call,
    # so a stale mask can never be reused; getattr keeps the forcing path and the
    # test stubs (which have no _last_anchor) working.
    anchor = getattr(sampler, "_last_anchor", None)
    anchor_mask = None
    if anchor is not None:
        mask, T0_exact = anchor
        if int(mask.shape[0]) != int(yt.shape[0]):
            raise RuntimeError(
                f"Anchor metadata batch dim {int(mask.shape[0])} does not match "
                f"collocation batch dim {int(yt.shape[0])}."
            )
        anchor_mask = mask.to(yt.device)
        m = anchor_mask.view(-1, 1, 1, 1)
        T0 = T0_exact.to(device=yt.device, dtype=yt.dtype)[..., None]
        # detach: gradient flows only through model(0->dt) on anchored rows.
        yt = torch.where(m, T0.detach(), yt)

    def _build_geom_bc(idx=None):
        rc = R_c if idx is None else R_c[idx]
        qn = qL_n if idx is None else qL_n[idx]
        qnp1 = qL_np1 if idx is None else qL_np1[idx]
        qint = qL_int if idx is None else qL_int[idx]
        geom = build_cn_geom_batched(
            sampler.geom_cfg["x_grid"], sampler.geom_cfg["y_grid"],
            sampler.geom_cfg["k_left"], sampler.geom_cfg["k_right"],
            sampler.geom_cfg["interface_x"], rc.to(device),
            sampler.geom_cfg["dt"], sigma_global=sampler.geom_cfg["sigma_global"],
            device=device, dtype=yt.dtype,
        )
        bc = FullBCData(
            T_right_tilde=sampler.geom_cfg["T_right_tilde"],
            qL_n=qn.to(device=device, dtype=yt.dtype),
            qL_np1=qnp1.to(device=device, dtype=yt.dtype),
            qL_int=qint.to(device=device, dtype=yt.dtype),
        )
        return geom, bc

    geom, bc = _build_geom_bc()
    out = full_bc_physics_loss(
        yt, ytdt, geom, bc,
        region_weights=region_weights, dirichlet_both_ends=True,
        per_sample=causal_weighter is not None,
    )

    # ---- split anchored vs non-anchored weighted residual (diagnostic only) ----
    # The total physics loss can fall on easy late steady-state pairs while the
    # T0 -> T_hat(dt) residual stays poor, so log the two subsets separately. The
    # subset weighted residual is recomputed under no_grad on masked rows (the
    # residual is flattened over batch+cells, so it cannot be split post-hoc).
    B = int(yt.shape[0])
    n_anchor = 0
    anchor_weighted = None
    nonanchor_weighted = None
    if anchor_mask is not None:
        with torch.no_grad():
            n_anchor = int(anchor_mask.sum().item())
            if n_anchor > 0:
                idx_a = torch.nonzero(anchor_mask, as_tuple=False).view(-1)
                g_a, bc_a = _build_geom_bc(idx_a.cpu())
                anchor_weighted = full_bc_physics_loss(
                    yt[idx_a], ytdt[idx_a], g_a, bc_a,
                    region_weights=region_weights, dirichlet_both_ends=True,
                )["physics_loss_weighted"].item()
            if n_anchor < B:
                idx_n = torch.nonzero(~anchor_mask, as_tuple=False).view(-1)
                g_n, bc_n = _build_geom_bc(idx_n.cpu())
                nonanchor_weighted = full_bc_physics_loss(
                    yt[idx_n], ytdt[idx_n], g_n, bc_n,
                    region_weights=region_weights, dirichlet_both_ends=True,
                )["physics_loss_weighted"].item()
    out["_anchor_weighted"] = anchor_weighted
    out["_nonanchor_weighted"] = nonanchor_weighted
    out["_anchor_count"] = n_anchor
    out["_nonanchor_count"] = B - n_anchor

    # ---- residual-adaptive causal lead weighting of the interior residual ----
    # Bin the per-sample interior residual by lead, form the DDP-safe differentiable
    # causal interior objective, and swap it for the plain interior term in a
    # ``_causal_total`` that mirrors ``physics_loss_weighted`` (same region weights,
    # BC/IC terms untouched). The weighter is not mutated here; the detached global
    # bin stats are returned so the caller can ``commit`` after backward.
    if causal_weighter is not None:
        bin_ids = getattr(sampler, "_last_lead_bin_ids", None)
        if bin_ids is None:
            raise RuntimeError(
                "causal weighting requires sampler._last_lead_bin_ids; attach a "
                "TemporalBinSpec to the collocation sampler (run_one_seed does this "
                "when causal_weighting.enabled)."
            )
        n_bins = int(causal_weighter.n_bins)
        idx = torch.as_tensor(np.asarray(bin_ids), device=device, dtype=torch.long)
        per_int = out["interior_per_sample"]  # (B,) differentiable
        local_bin_sums = torch.zeros(
            n_bins, device=device, dtype=per_int.dtype
        ).index_add(0, idx, per_int)
        local_bin_counts = torch.bincount(idx, minlength=n_bins).to(
            device=device, dtype=per_int.dtype
        )
        active_np = causal_weighter.active_mask(max_lead)
        global_sums, global_counts, weights_np = causal_weighter.prepare(
            local_bin_sums, local_bin_counts, active_np, dist_info
        )
        cw = torch.as_tensor(weights_np, device=device, dtype=per_int.dtype)
        gc = global_counts.to(device=device, dtype=per_int.dtype)
        am = torch.as_tensor(active_np, device=device, dtype=torch.bool)
        active_pop = am & (gc > 0)
        if bool(active_pop.any()):
            # LOCAL differentiable sums / GLOBAL detached counts * world_size so the
            # DDP gradient average reproduces the exact global per-bin mean even if
            # per-rank counts differ; == mean of bin means when all weights are 1.
            causal_interior = (
                cw[active_pop]
                * float(world_size)
                * local_bin_sums[active_pop]
                / gc[active_pop].clamp_min(1.0)
            ).sum() / active_pop.sum()
        else:
            causal_interior = out["phys_interior_mse"]

        rw = {n: 1.0 for n in (
            "interior", "left_neumann", "right_dirichlet", "topbot_adiabatic"
        )}
        if region_weights:
            for _k, _v in region_weights.items():
                if _k in rw:
                    rw[_k] = float(_v)
        causal_total = (
            rw["interior"] * causal_interior
            + rw["left_neumann"] * out["phys_left_neumann_mse"]
            + rw["right_dirichlet"] * out["phys_right_dirichlet_mse"]
            + rw["topbot_adiabatic"] * out["phys_topbot_adiabatic_mse"]
        )
        out["_causal_interior"] = causal_interior
        out["_causal_total"] = causal_total
        out["causal_bin_sums"] = global_sums.detach().cpu()
        out["causal_bin_counts"] = global_counts.detach().cpu()
        out["causal_weights_global"] = cw.detach().cpu()

    ic_loss = None
    if lambda_ic > 0.0:
        batch_anchor, target = sampler.sample_anchor_batch()
        y_anchor = _fwd(batch_anchor)
        target = target.to(device=device, dtype=y_anchor.dtype)
        ic_loss = torch.mean((y_anchor - target) ** 2)

    return out, R_c.shape[0], ic_loss


def _flat_grad(model) -> torch.Tensor:
    """Flatten every parameter's ``.grad`` into one vector (zeros for None)."""
    parts = []
    for p in model.parameters():
        if p.grad is None:
            parts.append(torch.zeros(p.numel(), device=p.device, dtype=p.dtype))
        else:
            parts.append(p.grad.detach().reshape(-1))
    return torch.cat(parts)


def _log_grad_balance(
    *, model, fixed_batches, physics_collocation, collocation_max_lead,
    physics_region_weights, loss_fn, device, use_per_sample_interface,
    lambda_data, lambda_physics, lambda_ic, n_colloc, epoch, phase, csv_path,
):
    """Optional diagnostic (``training.physics.log_grad_norms``; default off).

    On a fixed set of data batches and ``n_colloc`` fresh collocation draws,
    separately backprop ``lambda_data * L_data`` and ``lambda_physics * L_phys``
    and log ``||grad L_data||``, ``||grad L_phys||``, their ratio, and the cosine
    ``A = (g_data . g_phys) / (||g_data|| ||g_phys||)``. The norm ratio flags a
    mis-scaled ``lambda_physics``; the cosine flags directional conflict between
    the two objectives. Leaves ``model`` grads cleared on exit; never steps the
    optimizer. Rows append to a sidecar ``grad_balance.csv``.

    WARNING (pre-existing, out of scope): this draws ``n_colloc`` EXTRA collocation
    batches from the shared sampler, advancing its RNG on rank 0 only. Under DDP
    that desyncs the lockstep collocation stream across ranks. Do NOT reuse this
    pattern for GradNorm -- ``GradNormBalancer`` measures on the current step's
    already-built graph via ``autograd.grad`` and never draws new batches.
    """
    model.zero_grad(set_to_none=True)
    data_loss = None
    for batch in fixed_batches:
        x_spatial = batch["spatial"].to(device)
        cond_static = batch["cond_static"].to(device)
        forcing_seq = batch["forcing_seq"].to(device) if "forcing_seq" in batch else None
        y_batch = batch["Y"].to(device)
        iface_x = get_batch_interface_x(
            batch, device, use_per_sample_interface=use_per_sample_interface
        )
        y_pred = model(x_spatial, cond_static, forcing_seq)
        li = loss_fn(y_pred, y_batch, iface_x)
        data_loss = li if data_loss is None else data_loss + li
    if data_loss is not None:
        (lambda_data * data_loss).backward()
    g_data = _flat_grad(model)

    model.zero_grad(set_to_none=True)
    phys_loss = None
    for _ in range(int(n_colloc)):
        out, _pb, ic = _collocation_batch_loss(
            model, physics_collocation, collocation_max_lead,
            physics_region_weights, device, lambda_ic=lambda_ic,
        )
        pw = out["physics_loss_weighted"]
        phys_loss = pw if phys_loss is None else phys_loss + pw
        if ic is not None:
            phys_loss = phys_loss + lambda_ic * ic
    if phys_loss is not None:
        (lambda_physics * phys_loss).backward()
    g_phys = _flat_grad(model)
    model.zero_grad(set_to_none=True)

    nd = float(torch.linalg.vector_norm(g_data))
    npz = float(torch.linalg.vector_norm(g_phys))
    denom = nd * npz
    cos = float(torch.dot(g_data, g_phys) / denom) if denom > 0.0 else float("nan")
    ratio = (nd / npz) if npz > 0.0 else float("nan")

    header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if header:
            w.writerow([
                "epoch", "phase", "grad_norm_data", "grad_norm_phys",
                "grad_norm_ratio", "grad_cosine",
            ])
        w.writerow([epoch, phase, nd, npz, ratio, cos])
    print(
        f"[grad-balance] epoch {epoch} ({phase}): "
        f"||g_data||={nd:.3e} ||g_phys||={npz:.3e} ratio={ratio:.3f} cos={cos:.3f}",
        flush=True,
    )


def _phase_now(sync_cuda=False):
    """Monotonic timestamp for per-phase timing; optionally CUDA-synchronize first
    so the measured span reflects finished device work rather than launch latency."""
    if sync_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def _physics_static_coeffs(
    region_weights, lambda_data, lambda_physics, lambda_ic,
    *, has_data, has_ic, ic_multiplier=1.0,
):
    """Static (non-GradNorm) coefficient for each raw component, defined in ONE
    place so the compose site's legacy and GradNorm branches share it.

    ``right_bc`` is 0 when its region weight is 0 (e.g. the hard right-Dirichlet BC
    turns the ``right_dirichlet`` region into a should-be-~0 diagnostic).
    """
    rw = region_weights or {}
    coeffs = {
        "interior": lambda_physics * float(rw.get("interior", 1.0)),
        "left_bc": lambda_physics * float(rw.get("left_neumann", 1.0)),
        "adiabatic_bc": lambda_physics * float(rw.get("topbot_adiabatic", 1.0)),
        "right_bc": lambda_physics * float(rw.get("right_dirichlet", 1.0)),
    }
    if has_ic:
        coeffs["ic"] = lambda_ic * float(ic_multiplier)
    if has_data:
        coeffs["data"] = lambda_data
    return coeffs


# Deterministic order for the GradNorm term vector / active-term filtering.
_PHYSICS_TERM_ORDER = ("interior", "left_bc", "adiabatic_bc", "right_bc", "ic", "data")


def _active_physics_terms(coeffs):
    """Term names with a strictly positive static coefficient, in canonical order."""
    return [n for n in _PHYSICS_TERM_ORDER if float(coeffs.get(n, 0.0)) > 0.0]


def _compose_physics_total(
    *, out, data_loss, ic, causal_weighter, gradnorm, model_unwrapped, dist_info,
    lambda_data, lambda_physics, lambda_ic, region_weights, ic_multiplier=1.0,
    sync_cuda=False,
):
    """The SINGLE site that forms the optimized physics/BC/IC(/data) total.

    Returns ``(total, gradnorm_mults_or_None, gradnorm_ms)``.

    When ``gradnorm is None`` this reproduces the EXACT legacy expression (so the
    default path is bit-identical). When set, it expands to ``sum(c_t * m_t * L_t)``
    over the active raw components, where ``c_t`` is the static coefficient and
    ``m_t`` the GradNorm multiplier (both 1 -> the plain statically-weighted sum).
    """
    has_data = data_loss is not None
    has_ic = ic is not None
    if gradnorm is None:
        grad_phys = (
            out["_causal_total"] if causal_weighter is not None
            else out["physics_loss_weighted"]
        )
        total = lambda_physics * grad_phys
        if has_data:
            total = lambda_data * data_loss + total
        if has_ic:
            total = total + (lambda_ic * float(ic_multiplier)) * ic
        return total, None, 0.0

    coeffs = _physics_static_coeffs(
        region_weights, lambda_data, lambda_physics, lambda_ic,
        has_data=has_data, has_ic=has_ic, ic_multiplier=ic_multiplier,
    )
    interior_L = (
        out["_causal_interior"] if causal_weighter is not None
        else out["phys_interior_mse"]
    )
    raw = {
        "interior": interior_L,
        "left_bc": out["phys_left_neumann_mse"],
        "adiabatic_bc": out["phys_topbot_adiabatic_mse"],
        "right_bc": out["phys_right_dirichlet_mse"],
    }
    if has_ic:
        raw["ic"] = ic
    if has_data:
        raw["data"] = data_loss
    active = _active_physics_terms(coeffs)
    raw = {n: raw[n] for n in active if n in raw}

    t0 = _phase_now(sync_cuda)
    mults = gradnorm.maybe_update(
        raw, list(model_unwrapped.parameters()), dist_info=dist_info,
    )
    gradnorm_ms = (_phase_now(sync_cuda) - t0) * 1000.0

    total = None
    for name in active:
        if name not in raw:
            continue
        term = coeffs[name] * float(mults.get(name, 1.0)) * raw[name]
        total = term if total is None else total + term
    return total, mults, gradnorm_ms


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
    lambda_ic: float = 0.0,
    causal_weighter=None,
    gradnorm=None,
    model_unwrapped=None,
    warmup=None,
    global_step_start: int = 0,
    scheduler=None,
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

    # --- Fine-tune diagnostics (per-rank; rank 0's values are logged) ---
    _t_train_start = time.perf_counter()
    n_model_forwards = 0
    grad_norm_sum = 0.0
    n_grad_steps = 0
    successful_updates = 0
    lr_first = float("nan")
    lr_last = float("nan")
    phys_leads_epoch: list[float] = []  # collocation leads actually drawn

    def _record_successful_update(lr_used: float) -> None:
        nonlocal successful_updates, lr_first, lr_last
        successful_updates += 1
        if successful_updates == 1:
            lr_first = lr_used
        lr_last = lr_used
        if scheduler is not None:
            _advance_scheduler(
                scheduler, unit="update", successful_updates=1
            )

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
    phys_collocation = physics_collocation is not None and (
        lambda_physics > 0.0 or lambda_ic > 0.0
    )
    phys_onestep = physics_loader is not None and lambda_physics > 0.0
    phys_enabled = phys_onestep or phys_collocation

    # Fail-fast on a configuration with no active loss term: a data-only run with
    # lambda_data=0 and no live physics/anchor objective would silently optimize
    # nothing. The lambda_ic=0 physics-only negative control (lambda_physics>0)
    # and the anchor-only debug run (lambda_ic>0, lambda_physics=0) stay legal.
    if lambda_data == 0.0 and not phys_collocation and not phys_onestep:
        raise ValueError(
            "no active loss term: set at least one of lambda_data/lambda_physics/"
            "lambda_ic > 0 (physics terms also require a built physics supply)."
        )

    # ---- IC-anchoring warm-up schedule (Step 5; default inert) ----
    # For the first ``warmup.steps`` optimizer steps, resample collocation only
    # every ``resample_every`` steps (freeze the sampler in between) and boost the
    # IC coefficient by ``ic_multiplier`` -- applied AFTER the GradNorm multiplier
    # (folded into c_ic at the compose site), so GradNorm cannot normalize it out.
    # Progress is the persistent ``global_optimizer_step`` counter (checkpointed by
    # run_one_seed), not epoch*steps_per_epoch, so the boundary survives a resized
    # dataset / different world size on resume. The epoch-local freeze cache is
    # (re)armed here and dropped when warm-up is off, so the disabled path is
    # bit-identical.
    _wu = warmup or {}
    warmup_on = (
        bool(_wu.get("enabled", False)) and phys_collocation
        and hasattr(physics_collocation, "set_frozen")
    )
    _wu_steps = int(_wu.get("steps", 0))
    _wu_resample_every = max(1, int(_wu.get("resample_every", 1)))
    _wu_ic_multiplier = float(_wu.get("ic_multiplier", 1.0))
    global_step = int(global_step_start)
    if physics_collocation is not None and hasattr(
        physics_collocation, "enable_freeze_cache"
    ):
        physics_collocation.enable_freeze_cache(warmup_on)

    def _apply_warmup(gstep):
        """Set the sampler freeze for this optimizer step and return the IC
        multiplier (1.0 outside warm-up). Called before the collocation draw."""
        if not warmup_on or gstep >= _wu_steps:
            if warmup_on:
                physics_collocation.set_frozen(False)
            return 1.0
        physics_collocation.set_frozen(gstep % _wu_resample_every != 0)
        return _wu_ic_multiplier

    phys_iter = iter(physics_loader) if phys_onestep else None
    phys_loss_sum = 0.0
    phys_weighted_sum = 0.0
    phys_allcell_sum = 0.0
    phys_region_sums = {
        "interior": 0.0, "left_neumann": 0.0,
        "right_dirichlet": 0.0, "topbot_adiabatic": 0.0,
    }
    n_phys = 0
    ic_sum = 0.0
    n_ic = 0
    # Causal-weighting bookkeeping (last committed weights/epsilon; JSON-logged).
    causal_weights_last = None
    causal_eps_last = None
    # Causally-weighted interior objective (rank-0 diagnostic; "" when causal off).
    # Distinct from the unweighted phys_region_sums["interior"] so a run can
    # compare the weighted training signal against the raw interior residual.
    phys_causal_interior_sum = 0.0
    n_causal_interior = 0
    # GradNorm bookkeeping (last multipliers; JSON-logged) + per-phase timing sums.
    gradnorm_weights_last = None
    if model_unwrapped is None:
        model_unwrapped = model.module if hasattr(model, "module") else model
    _sync_timing = torch.cuda.is_available()
    if _sync_timing:
        torch.cuda.reset_peak_memory_stats(device)
    tm_step_sum = 0.0
    tm_gradnorm_sum = 0.0
    tm_backward_sum = 0.0
    tm_optstep_sum = 0.0
    n_tm = 0
    # Anchored vs non-anchored weighted-residual split (on-grid anchored path).
    phys_anchor_sum = 0.0
    phys_nonanchor_sum = 0.0
    n_anchor = 0
    n_nonanchor = 0

    def _accum_phys_split(out):
        nonlocal phys_anchor_sum, phys_nonanchor_sum, n_anchor, n_nonanchor
        aw = out.get("_anchor_weighted")
        nw = out.get("_nonanchor_weighted")
        ac = int(out.get("_anchor_count", 0))
        nc = int(out.get("_nonanchor_count", 0))
        if aw is not None and ac > 0:
            phys_anchor_sum += aw * ac
            n_anchor += ac
        if nw is not None and nc > 0:
            phys_nonanchor_sum += nw * nc
            n_nonanchor += nc

    # Physics-only mode (lambda_data == 0): iterate collocation batches for the
    # same number of optimizer steps as a data epoch. ``train_loader`` is never
    # ``iter()``-ed (only ``__len__`` is read), so the data path cannot leak in;
    # the optimization mechanics match the data path exactly.
    physics_only = lambda_data == 0.0 and phys_collocation
    if physics_only:
        for _ in range(len(train_loader)):
            _ic_mult = _apply_warmup(global_step)
            _t_step0 = _phase_now(_sync_timing)
            optimizer.zero_grad()
            out, p_b, ic = _collocation_batch_loss(
                model, physics_collocation, collocation_max_lead,
                physics_region_weights, device, lambda_ic=lambda_ic,
                causal_weighter=causal_weighter, dist_info=dist_info,
                world_size=dist_info.world_size,
            )
            n_model_forwards += 2  # collocation forwards the model at t and t+dt
            _ll = getattr(physics_collocation, "_last_leads", None)
            if _ll is not None:
                phys_leads_epoch.extend(np.asarray(_ll).ravel().tolist())
            p_loss = out["physics_loss_weighted"]
            # ``p_loss`` stays the unweighted diagnostic for the logging
            # accumulators below; the optimized total comes from the single
            # compose site (causal interior + GradNorm multipliers when active).
            total, gn_mults, gn_ms = _compose_physics_total(
                out=out, data_loss=None, ic=ic, causal_weighter=causal_weighter,
                gradnorm=gradnorm, model_unwrapped=model_unwrapped,
                dist_info=dist_info, lambda_data=lambda_data,
                lambda_physics=lambda_physics, lambda_ic=lambda_ic,
                region_weights=physics_region_weights,
                ic_multiplier=_ic_mult, sync_cuda=_sync_timing,
            )
            if gn_mults is not None:
                gradnorm_weights_last = gn_mults
            _t_bwd0 = _phase_now(_sync_timing)
            total.backward()
            tm_backward_sum += (_phase_now(_sync_timing) - _t_bwd0) * 1000.0
            if grad_clip is not None:
                _gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                if _gn is not None:
                    grad_norm_sum += float(_gn)
                    n_grad_steps += 1
            _t_opt0 = _phase_now(_sync_timing)
            lr_used = float(optimizer.param_groups[0]["lr"])
            optimizer.step()
            _record_successful_update(lr_used)
            tm_optstep_sum += (_phase_now(_sync_timing) - _t_opt0) * 1000.0
            if causal_weighter is not None:
                causal_weighter.commit(
                    out["causal_bin_sums"], out["causal_bin_counts"]
                )
                causal_weights_last = causal_weighter.weights().tolist()
                causal_eps_last = float(causal_weighter.eps)
            tm_gradnorm_sum += gn_ms
            tm_step_sum += (_phase_now(_sync_timing) - _t_step0) * 1000.0
            n_tm += 1
            phys_loss_sum += p_loss.item() * p_b
            phys_weighted_sum += out["physics_loss_weighted"].item() * p_b
            phys_allcell_sum += out["physics_loss_allcell_mean"].item() * p_b
            for _r in phys_region_sums:
                phys_region_sums[_r] += out[f"phys_{_r}_mse"].item() * p_b
            if "_causal_interior" in out:
                phys_causal_interior_sum += out["_causal_interior"].item() * p_b
                n_causal_interior += p_b
            n_phys += p_b
            _accum_phys_split(out)
            if ic is not None:
                ic_sum += ic.item() * p_b
                n_ic += p_b
            global_step += 1

    data_iterable = [] if physics_only else train_loader
    for batch in data_iterable:
        x_spatial = batch["spatial"].to(device)
        cond_static = batch["cond_static"].to(device)
        forcing_seq = batch["forcing_seq"].to(device) if "forcing_seq" in batch else None
        y_batch = batch["Y"].to(device)
        iface_x = get_batch_interface_x(
            batch, device, use_per_sample_interface=use_per_sample_interface
        )

        _t_step0 = _phase_now(_sync_timing) if phys_collocation else 0.0
        optimizer.zero_grad()
        y_pred = model(x_spatial, cond_static, forcing_seq)
        n_model_forwards += 1
        loss = loss_fn(y_pred, y_batch, iface_x)
        causal_commit_stats = None  # set below when causal weighting is active
        _step_gn_ms = 0.0
        _ic_mult = _apply_warmup(global_step) if phys_collocation else 1.0
        if phys_collocation:
            out, p_b, ic = _collocation_batch_loss(
                model, physics_collocation, collocation_max_lead,
                physics_region_weights, device, lambda_ic=lambda_ic,
                causal_weighter=causal_weighter, dist_info=dist_info,
                world_size=dist_info.world_size,
            )
            if causal_weighter is not None:
                causal_commit_stats = (
                    out["causal_bin_sums"], out["causal_bin_counts"]
                )
            n_model_forwards += 2  # collocation forwards the model at t and t+dt
            _ll = getattr(physics_collocation, "_last_leads", None)
            if _ll is not None:
                phys_leads_epoch.extend(np.asarray(_ll).ravel().tolist())
            p_loss = out["physics_loss_weighted"]
            total, gn_mults, _step_gn_ms = _compose_physics_total(
                out=out, data_loss=loss, ic=ic, causal_weighter=causal_weighter,
                gradnorm=gradnorm, model_unwrapped=model_unwrapped,
                dist_info=dist_info, lambda_data=lambda_data,
                lambda_physics=lambda_physics, lambda_ic=lambda_ic,
                region_weights=physics_region_weights,
                ic_multiplier=_ic_mult, sync_cuda=_sync_timing,
            )
            if gn_mults is not None:
                gradnorm_weights_last = gn_mults
            _t_bwd0 = _phase_now(_sync_timing)
            total.backward()
            tm_backward_sum += (_phase_now(_sync_timing) - _t_bwd0) * 1000.0
            phys_loss_sum += p_loss.item() * p_b
            phys_weighted_sum += out["physics_loss_weighted"].item() * p_b
            phys_allcell_sum += out["physics_loss_allcell_mean"].item() * p_b
            for _r in phys_region_sums:
                phys_region_sums[_r] += out[f"phys_{_r}_mse"].item() * p_b
            if "_causal_interior" in out:
                phys_causal_interior_sum += out["_causal_interior"].item() * p_b
                n_causal_interior += p_b
            n_phys += p_b
            _accum_phys_split(out)
            if ic is not None:
                ic_sum += ic.item() * p_b
                n_ic += p_b
        elif phys_onestep:
            p_loss, p_b, phys_iter = _physics_batch_loss(
                model, physics_loader, phys_iter, physics_geom_cfg, device
            )
            n_model_forwards += 1  # one-step physics forwards the model once
            (lambda_data * loss + lambda_physics * p_loss).backward()
            phys_loss_sum += p_loss.item() * p_b
            n_phys += p_b
        else:
            loss.backward()

        if grad_clip is not None:
            _gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            if _gn is not None:
                grad_norm_sum += float(_gn)
                n_grad_steps += 1
        _t_opt0 = _phase_now(_sync_timing) if phys_collocation else 0.0
        lr_used = float(optimizer.param_groups[0]["lr"])
        optimizer.step()
        _record_successful_update(lr_used)
        if phys_collocation:
            tm_optstep_sum += (_phase_now(_sync_timing) - _t_opt0) * 1000.0
        if causal_commit_stats is not None:
            causal_weighter.commit(*causal_commit_stats)
            causal_weights_last = causal_weighter.weights().tolist()
            causal_eps_last = float(causal_weighter.eps)
        if phys_collocation:
            tm_gradnorm_sum += _step_gn_ms
            tm_step_sum += (_phase_now(_sync_timing) - _t_step0) * 1000.0
            n_tm += 1
        global_step += 1

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
             phys_region_sums["right_dirichlet"], phys_region_sums["topbot_adiabatic"],
             ic_sum, float(n_ic),
             phys_anchor_sum, phys_nonanchor_sum, float(n_anchor), float(n_nonanchor)],
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
        ic_sum = t[21].item()
        n_ic = int(t[22].item())
        phys_anchor_sum = t[23].item()
        phys_nonanchor_sum = t[24].item()
        n_anchor = int(t[25].item())
        n_nonanchor = int(t[26].item())

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

    # Item-8 diagnostics (per-rank; rank 0's values are the ones logged). These
    # default to null/0/"" so a normal run without collocation leaves the new CSV
    # columns empty rather than fabricating values.
    epoch_train_s = time.perf_counter() - _t_train_start
    grad_norm = (grad_norm_sum / n_grad_steps) if n_grad_steps > 0 else None
    if phys_leads_epoch:
        # Self-describing fixed-edge histogram over [0, t_final] so the analysis
        # can slice at any tc without re-binning. t_final comes from the sampler.
        t_final = float(getattr(physics_collocation, "t_final", 0.0)) if phys_collocation else 0.0
        if t_final <= 0.0:
            t_final = float(max(phys_leads_epoch))
        edges = np.linspace(0.0, t_final, 13)
        counts, _ = np.histogram(np.asarray(phys_leads_epoch, dtype=float), bins=edges)
        phys_lead_hist = json.dumps({
            "edges": [round(float(e), 6) for e in edges],
            "counts": [int(c) for c in counts],
        })
    else:
        phys_lead_hist = ""
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
        "ic_loss": ic_sum / max(n_ic, 1),
        # Anchored vs non-anchored weighted-residual split (0 when not anchoring).
        # anchor_loss is the primary training-time signal that the T0 -> T_hat(dt)
        # residual is actually decreasing (not masked by easy late-step pairs).
        "train_physics_anchor_loss": phys_anchor_sum / max(n_anchor, 1),
        "train_physics_nonanchor_loss": phys_nonanchor_sum / max(n_nonanchor, 1),
        "train_anchor_fraction": n_anchor / max(n_phys, 1),
        # Item-8 diagnostics: training-only wall time (excludes validation /
        # checkpoint / CSV), forward count, mean clipped grad norm, and the
        # per-epoch collocation-lead histogram (JSON, "" when no collocation).
        "epoch_train_s": epoch_train_s,
        "n_model_forwards": int(n_model_forwards),
        "grad_norm": grad_norm,
        "phys_lead_hist": phys_lead_hist,
        # Causal weighting (Step 2): last committed per-bin weights (JSON) and
        # epsilon. "" / None when causal weighting is off (CSV columns land in
        # Step 7 but the keys are harmless extras today).
        "causal_weights": (
            json.dumps([round(float(w), 6) for w in causal_weights_last])
            if causal_weights_last is not None else ""
        ),
        "causal_eps": causal_eps_last,
        "train_physics_causal_interior": (
            phys_causal_interior_sum / n_causal_interior
            if n_causal_interior > 0 else None
        ),
        # Peak CUDA memory for this epoch (MB; None on CPU). Records SOAP's
        # preconditioner-state footprint for the Step-8 matched-memory comparison.
        "opt_peak_mem_mb": (
            torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
            if torch.cuda.is_available() else None
        ),
        # GradNorm (Step 4): last multipliers (JSON, "" when off) and per-phase
        # timing means (ms; 0 when the collocation/GradNorm path never ran). These
        # are per-rank rank-0 diagnostics, not reduced across ranks.
        "gradnorm_weights": (
            json.dumps({k: round(float(v), 6) for k, v in gradnorm_weights_last.items()})
            if gradnorm_weights_last is not None else ""
        ),
        "train_step_ms": (tm_step_sum / n_tm) if n_tm > 0 else 0.0,
        "gradnorm_ms": (tm_gradnorm_sum / n_tm) if n_tm > 0 else 0.0,
        "backward_ms": (tm_backward_sum / n_tm) if n_tm > 0 else 0.0,
        "optimizer_step_ms": (tm_optstep_sum / n_tm) if n_tm > 0 else 0.0,
        "successful_updates": int(successful_updates),
        "lr_first": lr_first,
        "lr_last": lr_last,
        # Warm-up (Step 5): persistent optimizer-step counter after this epoch, so
        # run_one_seed can checkpoint it and resume the warm-up boundary exactly.
        "global_optimizer_step": int(global_step),
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
    # Long-lead OOD selection: minimize post-cutoff (tau > tc) Kelvin RMSE. Only
    # available when training.lead_cutoff_time is set; selection reads the guard
    # companion `rmse_K_lead_le_tc` separately.
    "val_rmse_K_lead_gt_tc": "rmse_K_lead_gt_tc",
}


def _selection_metric_value(val_metrics: dict[str, float], checkpoint_metric: str) -> float:
    """Resolve the scalar used for checkpoint selection from the val metric dict.

    Raises ``ValueError`` on an unrecognized ``checkpoint_metric`` rather than
    silently falling back to ``rel_l2``: a typo (``val_nrsme``) would otherwise
    select on a different objective than the one the user configured, with no
    signal in the logs.
    """
    if checkpoint_metric not in _CHECKPOINT_METRIC_KEYS:
        valid = ", ".join(sorted(_CHECKPOINT_METRIC_KEYS))
        raise ValueError(
            f"unknown training.checkpoint_metric {checkpoint_metric!r}; "
            f"expected one of: {valid}."
        )
    key = _CHECKPOINT_METRIC_KEYS[checkpoint_metric]
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
    lead_cutoff_time: float | None = None,
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
        rel_l2_all: list[torch.Tensor] = []  # per-pair rel-L2 % (for lead stratification)
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
                if lead_cutoff_time is not None:
                    rel_l2_all.append(_per_pair_rel_l2_percent(y_pred, y_batch).cpu())
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

        # --- Lead-stratified aggregates for long-lead OOD selection/logging ---
        # Split every val pair at the physical prediction lead `tau = t_j - t_s`
        # into <= tc and > tc populations. rmse_K is the project's stable primary
        # metric (Kelvin); rel_l2 is the noisy companion. The val loader is
        # unshuffled and yields dataset._pairs in order, so torch.cat(...) rows
        # align 1:1 with dataset._pairs (same invariant the val-pairs writer uses).
        lead_strat: dict[str, float] = {}
        if lead_cutoff_time is not None and dataset is not None and rmse_K_all:
            t_grid = np.asarray(dataset.t_grid)
            leads = np.array(
                [float(t_grid[j] - t_grid[s]) for (_sid, s, j) in dataset._pairs],
                dtype=np.float64,
            )
            eps = 0.5 * float(getattr(dataset, "dt", 0.0))
            le = leads <= (float(lead_cutoff_time) + eps)
            gt = ~le
            rmse_K_cat = torch.cat(rmse_K_all).numpy()
            rel_l2_cat = torch.cat(rel_l2_all).numpy() if rel_l2_all else np.full_like(rmse_K_cat, np.nan)
            n = min(rmse_K_cat.shape[0], leads.shape[0])
            le, gt, rmse_K_cat, rel_l2_cat = le[:n], gt[:n], rmse_K_cat[:n], rel_l2_cat[:n]

            def _mean(arr: np.ndarray, mask: np.ndarray) -> float:
                return float(arr[mask].mean()) if mask.any() else float("nan")

            lead_strat = {
                "rmse_K_lead_le_tc": _mean(rmse_K_cat, le),
                "rmse_K_lead_gt_tc": _mean(rmse_K_cat, gt),
                "rel_l2_lead_le_tc": _mean(rel_l2_cat, le),
                "rel_l2_lead_gt_tc": _mean(rel_l2_cat, gt),
            }

        return {
            "rel_l2": val_loss,
            "iface_rel_l2": val_iface,
            **lead_strat,
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
        norm_max_time = config["data"].get("norm_max_time", None)
        mu_global, sigma_global = compute_global_stats(
            trajectories, train_ids, t_grid=t_grid, max_time=norm_max_time,
        )

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
            train_max_target_time=config["data"].get("train_max_target_time", None),
            train_max_lead_time=config["data"].get("train_max_lead_time", None),
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
    hard_rd = bool(model_cfg.get("hard_right_dirichlet", False))
    t_right_K = float(model_cfg.get("hard_right_dirichlet_t_right", 300.0))
    t_right_norm = (t_right_K - float(mu_global)) / float(sigma_global) if hard_rd else 0.0
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
        hard_right_dirichlet=hard_rd,
        t_right_norm=t_right_norm,
    )

    # --- Resume state ---
    start_epoch = 0
    best_val_loss = float("inf")
    bad_epochs = 0
    # Persistent optimizer-step counter driving the IC-anchoring warm-up boundary
    # (Step 5). Checkpointed and restored so the boundary survives resume exactly,
    # unlike reconstructing epoch * steps_per_epoch.
    global_optimizer_step = 0

    if resuming:
        ckpt = torch.load(latest_path, map_location="cpu", weights_only=False)
        _validate_resume_compatibility(ckpt["conf"], config)
        fno.load_state_dict(ckpt["model_state"])
        best_val_loss = ckpt["best_val"]
        bad_epochs = ckpt.get("bad_epochs", 0)
        start_epoch = ckpt["epoch"] + 1
        global_optimizer_step = int(ckpt.get("global_optimizer_step", 0))

    # Weights-only warm-start (physics fine-tune). Only when NOT auto-resuming a
    # full run in this dir: load model weights from an external checkpoint and
    # leave optimizer/scheduler/epoch/best_val fresh (peak LR = training.
    # learning_rate, start_epoch = 0). The checkpoint's normalization stats must
    # equal the recomputed (mu_global, sigma_global) — identical since every run
    # uses the same train sims and full-trajectory stats — so warm-started weights
    # see the same normalized fields. _validate_resume_compatibility is
    # deliberately skipped (epochs/LR/physics differ by design).
    init_from_checkpoint = config["training"].get("init_from_checkpoint", None)
    if not resuming and init_from_checkpoint:
        init_ckpt = torch.load(
            init_from_checkpoint, map_location="cpu", weights_only=False,
        )
        # A baseline checkpoint (no hard BC) can warm-start a hard-BC fine-tune:
        # the only tolerated missing key is the t_right_norm buffer, which keeps
        # its build-time value. Any other missing/unexpected key is an error.
        missing, unexpected = fno.load_state_dict(
            init_ckpt["model_state"], strict=False
        )
        if set(missing) - {"t_right_norm"} or unexpected:
            raise RuntimeError(
                "init_from_checkpoint state_dict mismatch: "
                f"missing={list(missing)}, unexpected={list(unexpected)} "
                "(only a missing t_right_norm buffer is allowed)."
            )
        ck_mu = float(init_ckpt["mu_global"])
        ck_sigma = float(init_ckpt["sigma_global"])
        if (abs(ck_mu - float(mu_global)) > 1e-6
                or abs(ck_sigma - float(sigma_global)) > 1e-6):
            raise ValueError(
                "init_from_checkpoint normalization mismatch: checkpoint "
                f"(mu={ck_mu}, sigma={ck_sigma}) != recomputed "
                f"(mu={float(mu_global)}, sigma={float(sigma_global)}). Warm-start "
                "requires identical stats (same train sims, full-trajectory stats)."
            )
        if is_main:
            print(
                f"Warm-start seed {seed}: loaded weights from "
                f"{init_from_checkpoint} (optimizer/scheduler/epoch fresh)."
            )

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
            # The collocation physics path forwards the wrapped model three times
            # per step (data + collocation t + collocation t+dt) before a single
            # combined backward, so each parameter's grad-ready hook fires more
            # than once. With the default reducer (one firing per param expected)
            # the per-bucket all_reduce ordering can diverge across ranks and
            # deadlock NCCL. static_graph records the autograd order on the first
            # iteration and supports multi-forward / params reused across graphs.
            # Valid here because the used-parameter set is fixed per run (forward
            # branches are config/shape-driven, not data-value-driven).
            static_graph=True,
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
    lambda_ic = float(phys_cfg.get("lambda_ic", 0.0))
    phys_mode = str(phys_cfg.get("mode", "one_step"))
    collocation_source = str(phys_cfg.get("collocation_source", "train"))
    physics_loader = None
    physics_geom_cfg = None
    physics_collocation = None
    physics_region_weights = None
    causal_weighter = None  # built below when causal_weighting.enabled (Step 2)
    gradnorm = None  # built below when training.gradnorm.enabled (Step 4)
    # IC-anchoring warm-up config (Step 5). Passed to train_one_epoch unconditionally;
    # inert (bit-identical) unless warmup.enabled is set.
    warmup_cfg = phys_cfg.get("warmup", None) or {}
    collocation_lead_start = float(phys_cfg.get("collocation_lead_start", 0.01))
    collocation_lead_max = float(phys_cfg.get("collocation_lead_max", 0.01))
    collocation_lead_warmup = int(phys_cfg.get("collocation_lead_warmup_epochs", 0))
    # Physics-weight ramp: linearly scale lambda_physics from 0 to its full value
    # over the first `lambda_physics_ramp_epochs` epochs (0 disables the ramp).
    lambda_physics_ramp_epochs = int(phys_cfg.get("lambda_physics_ramp_epochs", 0))
    # Optional grad-balance / cosine diagnostic (default off). Sampled at three
    # checkpoints: fine-tune step 0, just after the ramp completes, near the end.
    log_grad_norms = bool(phys_cfg.get("log_grad_norms", False))
    grad_balance_batches: list = []
    cc_cfg = phys_cfg.get("causal_curriculum", None) or {}
    causal_curriculum_enabled = bool(cc_cfg.get("enabled", False))
    causal_stages = None  # resolved after the sampler pins dt/t_final
    if phys_mode == "collocation" and (lambda_physics > 0.0 or lambda_ic > 0.0):
        physics_region_weights = phys_cfg.get("full_bc_region_weights", None)
        # Source the collocation sims from TRAIN by default so the held-out
        # validation sims stay an honest cross-sim metric; "val" is an explicit
        # transductive-debug option (forcing pins it to keep its W2 source). Each
        # split is an independent SnapshotPairDataset whose sim_ids are already
        # restricted to that split, so the swap alone confines sampling.
        if collocation_source == "train":
            colloc_ds = training_set.dataset
        elif collocation_source == "val":
            colloc_ds = validation_set.dataset
        else:
            raise ValueError(
                f"unknown training.physics.collocation_source {collocation_source!r}; "
                f"expected 'train' or 'val'."
            )
        # Seed the collocation stream IDENTICALLY on every rank (rng_seed=seed, no
        # per-rank offset). A previous version offset by rank to decorrelate the
        # per-rank draws for a larger effective physics batch, but that interacts
        # fatally with static_graph=True (see the DDP wrap below): with per-rank
        # RNG the first iteration draws a different collocation batch and anchor
        # mask on each rank, and the anchored torch.where hard-injection changes
        # which elements of yt carry gradient. That makes the first-iteration
        # gradient-readiness order rank-dependent, so static_graph freezes a
        # DIFFERENT per-bucket all_reduce schedule on each rank and the reducer
        # deadlocks on mismatched-size ALLREDUCEs (NCCL watchdog abort). An
        # identical seed guarantees every rank builds the same static graph and
        # bucket schedule. The DDP-averaged gradient is then over the same
        # collocation batch on all ranks (no effective-batch gain), which is the
        # accepted cost of keeping the physics-only DDP path lockstep-safe.
        physics_collocation = _build_collocation_sampler(
            config, phys_cfg, colloc_ds, spec,
            mu_global, sigma_global,
            batch_size=config["training"]["batch_size"],
            rng_seed=seed,
        )
        # The sampler must only ever draw from its source split's sims.
        assert set(np.asarray(physics_collocation.sim_ids).tolist()).issubset(
            set(np.asarray(colloc_ds.sim_ids).tolist())
        )
        # Hard right-Dirichlet buffer and the collocation residual's right-wall
        # target must agree, else the exact BC and the physics residual pull
        # toward different temperatures.
        if hard_rd:
            geom_t_right = float(physics_collocation.geom_cfg["T_right_tilde"])
            if abs(t_right_norm - geom_t_right) > 1e-6:
                raise ValueError(
                    "hard_right_dirichlet t_right_norm "
                    f"({t_right_norm:.6f}) != collocation geom T_right_tilde "
                    f"({geom_t_right:.6f}); set model.parameters."
                    "hard_right_dirichlet_t_right to the collocation right-wall "
                    "temperature."
                )
        # Shared temporal bins for stratification (Step 1) and causal weighting
        # (Step 2): one object built from the sampler's path-aware lead domain and
        # attached to the sampler so both techniques bin identically. Built only
        # when a bin-consuming technique is on -> spec stays None (bit-identical)
        # otherwise. stratify_bins defaults to causal_weighting.n_bins so the two
        # techniques cannot drift apart.
        causal_cfg = phys_cfg.get("causal_weighting", {}) or {}
        stratify_leads_on = bool(phys_cfg.get("stratify_leads", False))
        causal_on = bool(causal_cfg.get("enabled", False))
        if stratify_leads_on or causal_on:
            causal_n_bins = int(causal_cfg.get("n_bins", 12))
            stratify_bins = phys_cfg.get("stratify_bins", None)
            spec_n_bins = (
                int(stratify_bins) if stratify_bins is not None else causal_n_bins
            )
            physics_collocation.bin_spec = _make_temporal_bin_spec(
                physics_collocation, phys_cfg, spec_n_bins
            )
        # Causal lead weighter (Step 2): consumes the SAME attached bin_spec so its
        # bins are byte-identical to the stratification bins. Built only when
        # causal_weighting.enabled; None otherwise (bit-identical). On resume the
        # committed EMA/epsilon/step are restored with a config-shape check.
        if causal_on:
            causal_weighter = CausalLeadWeighter(
                physics_collocation.bin_spec,
                eps=float(causal_cfg.get("eps", 1.0)),
                loss_history=str(causal_cfg.get("loss_history", "current")),
                ema_alpha=float(causal_cfg.get("ema_alpha", 0.9)),
                update_every=int(causal_cfg.get("update_every", 1)),
                eps_growth=float(causal_cfg.get("eps_growth", 2.0)),
                eps_max=float(causal_cfg.get("eps_max", 100.0)),
                eps_min=float(causal_cfg.get("eps_min", 0.01)),
                allow_eps_decrease=bool(causal_cfg.get("allow_eps_decrease", True)),
                last_bin_threshold=float(causal_cfg.get("last_bin_threshold", 0.99)),
                mean_weight_floor=float(causal_cfg.get("mean_weight_floor", 0.5)),
            )
            if resuming and ckpt.get("causal_state") is not None:
                causal_weighter.load_state_dict(ckpt["causal_state"])
        # GradNorm balancer (Step 4): balances the RAW per-component physics
        # losses (interior + BC regions + IC + optional data) via w_t =
        # mean(g)/g_t. Active terms are those with a positive static coefficient,
        # so forcing (live non-zero left-Neumann flux) balances left_bc while
        # diffusion (zero left flux) does not. Built only when
        # training.gradnorm.enabled; None otherwise (bit-identical). The
        # term_names resume-guard catches cross-benchmark checkpoint reuse.
        gradnorm_cfg = config["training"].get("gradnorm", {}) or {}
        if bool(gradnorm_cfg.get("enabled", False)):
            gn_coeffs = _physics_static_coeffs(
                physics_region_weights,
                lambda_data, lambda_physics, lambda_ic,
                has_data=(lambda_data > 0.0),
                has_ic=(lambda_ic > 0.0),
            )
            gn_terms = _active_physics_terms(gn_coeffs)
            gradnorm = GradNormBalancer(
                gn_terms,
                alpha_w=float(gradnorm_cfg.get("alpha_w", 0.9)),
                update_every=int(gradnorm_cfg.get("update_every", 10)),
                eps=float(gradnorm_cfg.get("eps", 1.0e-8)),
            )
            if resuming and ckpt.get("gradnorm_state") is not None:
                gradnorm.load_state_dict(ckpt["gradnorm_state"])
        if causal_curriculum_enabled:
            causal_stages = _resolve_causal_stages(
                cc_cfg, float(physics_collocation.dt), float(colloc_ds.t_final)
            )
        if is_main:
            print(
                f"Physics regularizer ON (W2 collocation): lambda_data={lambda_data}, "
                f"lambda_physics={lambda_physics}, lambda_ic={lambda_ic}, "
                f"residual=full_bc, source={collocation_source}, "
                f"lead_start={collocation_lead_start}, lead_max={collocation_lead_max}, "
                f"lead_warmup={collocation_lead_warmup}, "
                f"anchored_first_step={bool(physics_collocation.anchored_first_step)}, "
                f"early_oversample={bool(physics_collocation.early_oversample)}, "
                f"causal_curriculum={causal_stages if causal_curriculum_enabled else 'off'}",
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
    if checkpoint_metric not in _CHECKPOINT_METRIC_KEYS:
        valid = ", ".join(sorted(_CHECKPOINT_METRIC_KEYS))
        raise ValueError(
            f"unknown training.checkpoint_metric {checkpoint_metric!r}; "
            f"expected one of: {valid}."
        )

    # Long-lead OOD selection. `lead_cutoff_time` (tc) splits every val pair at
    # its prediction lead tau=t_j-t_s; `checkpoint_forgetting_ratio` bounds the
    # allowed pre-cutoff (tau<=tc) RMSE_K regression relative to model A's step-0
    # reference (`pre_cutoff_ref`, captured below before any optimizer step).
    lead_cutoff_time = config["training"].get("lead_cutoff_time", None)
    checkpoint_forgetting_ratio = config["training"].get("checkpoint_forgetting_ratio", None)
    if checkpoint_metric == "val_rmse_K_lead_gt_tc" and lead_cutoff_time is None:
        raise ValueError(
            "training.checkpoint_metric='val_rmse_K_lead_gt_tc' requires "
            "training.lead_cutoff_time to be set (the tau split point)."
        )
    pre_cutoff_ref = None  # rmse_K^A_{tau<=tc}; set by the step-0 reference pass.

    best_path = run_path / "fno2d_best.pt"

    # --- CSV setup (rank 0 only) ---
    csv_path = run_path / "train_metrics.csv"
    val_pairs_path = run_path / "val_pairs.csv"
    fieldnames = [
        "epoch", "train_loss", "train_physics_loss", "train_ic_loss",
        "train_physics_loss_weighted", "train_physics_loss_allcell_mean",
        "train_phys_interior_mse", "train_phys_left_neumann_mse",
        "train_phys_right_dirichlet_mse", "train_phys_topbot_adiabatic_mse",
        "train_physics_anchor_loss", "train_physics_nonanchor_loss",
        "train_anchor_fraction",
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
        # Long-lead OOD stratification (present but empty when lead_cutoff_time
        # is null). rmse_K_lead_gt_tc is the post-cutoff selection primary.
        "val_rmse_K_lead_le_tc", "val_rmse_K_lead_gt_tc",
        "val_rel_l2_lead_le_tc", "val_rel_l2_lead_gt_tc",
        # Fine-tune diagnostics (empty on non-collocation paths where absent).
        "train_grad_norm", "epoch_train_s", "epoch_wall_s", "n_model_forwards",
        "phys_lead_hist",
        # PINO fine-tune diagnostics (Steps 2/4/5/6). Empty/zero when the
        # corresponding knob is off, so baseline runs stay comparable.
        "train_physics_causal_interior", "causal_weights", "causal_eps",
        "gradnorm_weights", "opt_peak_mem_mb",
        "train_step_ms", "gradnorm_ms", "backward_ms", "optimizer_step_ms",
        "lr_first", "lr_last", "lr", "is_best",
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

    # Step-0 reference (Capability 7). For a warm-started fine-tune with a lead
    # cutoff, run ONE validation pass on the freshly-loaded model A weights BEFORE
    # any optimizer step, so `pre_cutoff_ref` = rmse_K^A_{tau<=tc} is model A
    # exactly (not epoch-0-post-step). Logged as epoch -1 so it is auditable. Only
    # on a fresh warm-start (skip on auto-resume, which already has trained state).
    if is_main and (not resuming) and init_from_checkpoint and lead_cutoff_time is not None:
        ref_metrics = validate(
            model=fno_unwrapped,
            val_loader=validation_set,
            device=device,
            iface_mask=iface_mask,
            dataset=validation_set.dataset,
            pair_csv_path=val_pairs_path,
            val_pairs_max_rows=config["training"].get("val_pairs_max_rows", None),
            epoch=-1,
            x_grid_t=loss_fn.x_grid_t,
            interface_half_width=loss_cfg.get("interface_half_width", 0.05),
            use_per_sample_interface=use_per_sample_interface,
            interface_x=loss_cfg.get("interface_x", 0.5),
            sigma_global=sigma_global,
            lead_cutoff_time=lead_cutoff_time,
        )
        pre_cutoff_ref = float(ref_metrics["rmse_K_lead_le_tc"])
        print(
            f"[step-0 ref] pre_cutoff_ref (rmse_K tau<={lead_cutoff_time}) = "
            f"{pre_cutoff_ref:.4f}K; post-cutoff rmse_K = "
            f"{ref_metrics['rmse_K_lead_gt_tc']:.4f}K",
            flush=True,
        )
        csv_writer.writerow({
            "epoch": -1,
            "lr_first": float("nan"),
            "lr_last": float("nan"),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "is_best": 0,
            "val_rel_l2": float(ref_metrics["rel_l2"]),
            "val_rmse_K": float(ref_metrics["rmse_K"]),
            "val_rmse_K_lead_le_tc": float(ref_metrics["rmse_K_lead_le_tc"]),
            "val_rmse_K_lead_gt_tc": float(ref_metrics["rmse_K_lead_gt_tc"]),
            "val_rel_l2_lead_le_tc": float(ref_metrics["rel_l2_lead_le_tc"]),
            "val_rel_l2_lead_gt_tc": float(ref_metrics["rel_l2_lead_gt_tc"]),
        })
        csv_file.flush()

    should_stop = False

    # Freeze ~8 data batches once for the optional grad-balance diagnostic so the
    # three checkpoints compare gradients on an identical set (not epoch noise).
    if is_main and log_grad_norms and physics_collocation is not None:
        for _i, _b in enumerate(training_set):
            if _i >= 8:
                break
            grad_balance_batches.append(_b)
    grad_balance_path = str(run_path / "grad_balance.csv")

    for epoch in range(start_epoch, epochs):
        _t_epoch_start = time.perf_counter()
        # Lead-time curriculum: progressively expose longer lead times.
        # Deterministic given `frac` -> identical _active_len on every rank.
        if warmup_epochs > 0:
            frac = min(1.0, (epoch + 1) / warmup_epochs)
            training_set.dataset.set_curriculum_fraction(frac)

        # DistributedSampler shuffle ordering: curriculum first, then set_epoch.
        if dist_info.is_distributed and hasattr(training_set.sampler, "set_epoch"):
            training_set.sampler.set_epoch(epoch)

        # W2 collocation-lead curriculum. Two mutually-exclusive schedules:
        #  - staged causal curriculum (diffusion): fixed stages expand the max
        #    TARGET time of the sampled window (epoch is 0-indexed; final stage
        #    with epochs=-1 covers the remainder). Keeps the 0->dt anchor bucket
        #    active in every stage (its weight lives in early_mix).
        #  - else the legacy linear ramp from start to max over lead_warmup epochs
        #    (an optimization-stability device). Inert when collocation is off.
        if causal_curriculum_enabled and causal_stages is not None:
            collocation_max_lead = _causal_max_lead(causal_stages, epoch)
        else:
            if collocation_lead_warmup > 0:
                cfrac = min(1.0, (epoch + 1) / collocation_lead_warmup)
            else:
                cfrac = 1.0
            collocation_max_lead = (
                collocation_lead_start
                + cfrac * (collocation_lead_max - collocation_lead_start)
            )

        # Physics-weight ramp (epoch is 0-indexed; epoch 0 keeps a small nonzero
        # weight, full weight from `lambda_physics_ramp_epochs`).
        if lambda_physics_ramp_epochs > 0:
            lambda_physics_eff = lambda_physics * min(
                1.0, (epoch + 1) / lambda_physics_ramp_epochs
            )
        else:
            lambda_physics_eff = lambda_physics

        # Grad-balance / cosine at three checkpoints (before this epoch's steps):
        # fine-tune start, just after the ramp completes, and near the end.
        if is_main and log_grad_norms and physics_collocation is not None and grad_balance_batches:
            _gb_phase = None
            if epoch == start_epoch:
                _gb_phase = "start"
            elif lambda_physics_ramp_epochs > 0 and epoch == lambda_physics_ramp_epochs:
                _gb_phase = "post_ramp"
            elif epoch == epochs - 1:
                _gb_phase = "near_end"
            if _gb_phase is not None:
                _log_grad_balance(
                    model=fno_unwrapped, fixed_batches=grad_balance_batches,
                    physics_collocation=physics_collocation,
                    collocation_max_lead=collocation_max_lead,
                    physics_region_weights=physics_region_weights,
                    loss_fn=loss_fn, device=device,
                    use_per_sample_interface=use_per_sample_interface,
                    lambda_data=lambda_data, lambda_physics=lambda_physics_eff,
                    lambda_ic=lambda_ic, n_colloc=8, epoch=epoch,
                    phase=_gb_phase, csv_path=grad_balance_path,
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
            lambda_physics=lambda_physics_eff,
            lambda_ic=lambda_ic,
            causal_weighter=causal_weighter,
            gradnorm=gradnorm,
            model_unwrapped=fno_unwrapped,
            warmup=warmup_cfg,
            global_step_start=global_optimizer_step,
            scheduler=scheduler,
        )
        global_optimizer_step = int(train_metrics["global_optimizer_step"])
        train_loss = train_metrics["loss"]
        train_rel_l2 = train_metrics["rel_l2"]
        train_iface_rel_l2 = train_metrics["iface_rel_l2"]
        _advance_scheduler(
            scheduler,
            unit="epoch",
            successful_updates=int(train_metrics["successful_updates"]),
        )

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
                    lead_cutoff_time=lead_cutoff_time,
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
                # No-forgetting guard (Capability 7): when selecting on post-cutoff
                # RMSE_K, only accept an epoch whose pre-cutoff (tau<=tc) RMSE_K
                # stays within `ratio * pre_cutoff_ref` (model A). Skips the guard
                # when the ratio or the step-0 reference is unavailable.
                guard_ok = True
                if (
                    checkpoint_metric == "val_rmse_K_lead_gt_tc"
                    and checkpoint_forgetting_ratio is not None
                    and pre_cutoff_ref is not None
                ):
                    pre_now = float(val_metrics["rmse_K_lead_le_tc"])
                    guard_ok = pre_now <= float(checkpoint_forgetting_ratio) * pre_cutoff_ref
                    if not guard_ok:
                        print(
                            f"  [forgetting guard] epoch {epoch} rejected: "
                            f"pre-cutoff rmse_K={pre_now:.4f}K > "
                            f"{checkpoint_forgetting_ratio}*{pre_cutoff_ref:.4f}K",
                            flush=True,
                        )
                if guard_ok and selection_value < best_val_loss:
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
                            "causal_state": (
                                causal_weighter.state_dict()
                                if causal_weighter is not None else None
                            ),
                            "gradnorm_state": (
                                gradnorm.state_dict()
                                if gradnorm is not None else None
                            ),
                            "global_optimizer_step": int(global_optimizer_step),
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
                    "causal_state": (
                        causal_weighter.state_dict()
                        if causal_weighter is not None else None
                    ),
                    "gradnorm_state": (
                        gradnorm.state_dict()
                        if gradnorm is not None else None
                    ),
                    "global_optimizer_step": int(global_optimizer_step),
                },
                latest_path,
            )

            epoch_wall_s = time.perf_counter() - _t_epoch_start

            def _vm(key: str) -> str | float:
                return "" if val_metrics is None else float(val_metrics[key])

            def _vm_opt(key: str) -> str | float:
                # Tolerant read for keys only present when lead_cutoff_time is set.
                if val_metrics is None or key not in val_metrics:
                    return ""
                return float(val_metrics[key])

            def _tm_opt(key: str) -> str | float:
                # Tolerant read for diagnostic train keys absent on non-physics paths.
                v = train_metrics.get(key, None)
                return "" if v is None else float(v)

            csv_writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": float(train_loss),
                    "train_physics_loss": float(train_metrics["physics_loss"]),
                    "train_ic_loss": float(train_metrics["ic_loss"]),
                    "train_physics_loss_weighted": float(train_metrics["physics_loss_weighted"]),
                    "train_physics_loss_allcell_mean": float(train_metrics["physics_loss_allcell_mean"]),
                    "train_phys_interior_mse": float(train_metrics["phys_interior_mse"]),
                    "train_phys_left_neumann_mse": float(train_metrics["phys_left_neumann_mse"]),
                    "train_phys_right_dirichlet_mse": float(train_metrics["phys_right_dirichlet_mse"]),
                    "train_phys_topbot_adiabatic_mse": float(train_metrics["phys_topbot_adiabatic_mse"]),
                    "train_physics_anchor_loss": float(train_metrics["train_physics_anchor_loss"]),
                    "train_physics_nonanchor_loss": float(train_metrics["train_physics_nonanchor_loss"]),
                    "train_anchor_fraction": float(train_metrics["train_anchor_fraction"]),
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
                    "val_rmse_K_lead_le_tc": _vm_opt("rmse_K_lead_le_tc"),
                    "val_rmse_K_lead_gt_tc": _vm_opt("rmse_K_lead_gt_tc"),
                    "val_rel_l2_lead_le_tc": _vm_opt("rel_l2_lead_le_tc"),
                    "val_rel_l2_lead_gt_tc": _vm_opt("rel_l2_lead_gt_tc"),
                    "train_grad_norm": _tm_opt("grad_norm"),
                    "epoch_train_s": float(train_metrics.get("epoch_train_s", float("nan"))),
                    "epoch_wall_s": float(epoch_wall_s),
                    "n_model_forwards": int(train_metrics.get("n_model_forwards", 0)),
                    "phys_lead_hist": train_metrics.get("phys_lead_hist", ""),
                    "train_physics_causal_interior": _tm_opt("train_physics_causal_interior"),
                    "causal_weights": train_metrics.get("causal_weights", ""),
                    "causal_eps": _tm_opt("causal_eps"),
                    "gradnorm_weights": train_metrics.get("gradnorm_weights", ""),
                    "opt_peak_mem_mb": _tm_opt("opt_peak_mem_mb"),
                    "train_step_ms": _tm_opt("train_step_ms"),
                    "gradnorm_ms": _tm_opt("gradnorm_ms"),
                    "backward_ms": _tm_opt("backward_ms"),
                    "optimizer_step_ms": _tm_opt("optimizer_step_ms"),
                    "lr_first": float(train_metrics["lr_first"]),
                    "lr_last": float(train_metrics["lr_last"]),
                    "lr": float(train_metrics["lr_last"]),
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
