import csv
import copy
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
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam, AdamW

from data.dataset import (
    TEMPORAL_SAMPLES,
    SnapshotPairDataset,
    assert_dataset_problem_version,
    build_normalization_provenance,
    collate_fn,
    compute_global_stats,
    create_dataloaders,
    load_ramp_seconds,
    load_sim_data,
    load_solver_dt,
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
    get_batch_interface_x,
    interface_flanking_nodes,
    interface_flanking_nodes_per_sample,
    per_sample_node_jump_errors,
    per_sample_nrmse,
    per_sample_sq_rms,
    tail_stats,
)
from src.operators.utils import resolve_device

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

DIFFUSION_GEOMETRY_X_NORM = (0.0, 0.25, 0.49, 0.51, 0.75, 1.0)
DIFFUSION_GEOMETRY_T_BAR_NORM = (1.0 / 30.0, 0.25, 0.5, 0.75, 1.0)
DIFFUSION_GEOMETRY_RC_NORM = (0.0, 0.5, 1.0)
DIFFUSION_GEOMETRY_FIELDNAMES = (
    "checkpoint_epoch",
    "block",
    "head",
    "x_norm",
    "t_bar_norm",
    "I_cross",
    "R_c_norm",
    "c",
)


def _load_yaml_mapping(path: Path) -> dict:
    with path.open(encoding="utf-8") as config_file:
        payload = yaml.safe_load(config_file)
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping/dict, got: {type(payload)}")
    return payload


def _merge_config(base: dict, overlay: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_config(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(config_path: str | None = None) -> dict:
    project_root = Path(__file__).resolve().parents[2]
    path = Path(config_path) if config_path else (project_root / "conf" / "config.yaml")
    path = path.expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    os.environ.setdefault("PROJECT_ROOT", project_root.as_posix())

    paths_cfg = _load_yaml_mapping(project_root / "conf" / "paths" / "default.yaml")
    cfg = _merge_config({"paths": paths_cfg}, _load_yaml_mapping(path))

    benchmark_cfg = cfg.get("benchmark", {})
    benchmark_name = os.environ.get("BENCHMARK", str(benchmark_cfg.get("name", "forcing")))
    benchmark_path = project_root / "conf" / "benchmark" / f"{benchmark_name}.yaml"
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark config not found: {benchmark_path}")
    cfg = _merge_config(cfg, _load_yaml_mapping(benchmark_path))

    benchmark_cfg = cfg.get("benchmark", {})
    representation_name = os.environ.get(
        "REPRESENTATION", str(benchmark_cfg.get("representation", "temporal_encoder"))
    )
    representation_path = project_root / "conf" / "representation" / f"{representation_name}.yaml"
    if not representation_path.exists():
        raise FileNotFoundError(f"Representation config not found: {representation_path}")
    cfg = _merge_config(cfg, _load_yaml_mapping(representation_path))

    paths = cfg["paths"]
    configured_project_root = paths.get("project_root")
    resolved_project_root = Path(
        str(configured_project_root or os.environ.get("PROJECT_ROOT", project_root))
    ).expanduser().resolve()
    configured_data_dir = paths.get("data_dir")
    configured_runs_root = paths.get("runs_root")
    data_dir = Path(
        str(configured_data_dir or os.environ.get("DATA_DIR", resolved_project_root / "data"))
    ).expanduser().resolve()
    runs_root = Path(
        str(configured_runs_root or os.environ.get("RUNS_ROOT", resolved_project_root / "runs"))
    ).expanduser().resolve()
    paths.update(
        project_root=str(resolved_project_root),
        data_dir=str(data_dir),
        runs_root=str(runs_root),
    )

    data_paths = {
        "t_grid_path": "t_grid.npy",
        "x_grid_path": "x_grid.npy",
        "y_grid_path": "y_grid.npy",
        "trajectories.npy": "trajectories.npy",
        "sim_params_path": "sim_params.npy",
    }
    for key, file_name in data_paths.items():
        if not cfg["data"].get(key):
            cfg["data"][key] = str(data_dir / file_name)

    experiment_name = os.environ.get("EXPERIMENT_NAME")
    if experiment_name:
        cfg.setdefault("experiment", {})["name"] = experiment_name

    run_cfg = cfg.setdefault("training", {}).setdefault("run", {})
    if not run_cfg.get("run_dir"):
        run_cfg["run_dir"] = str(
            runs_root
            / str(cfg["experiment"]["name"])
            / f"config{cfg['config_id']}"
        )

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


def _trainable_scope(config: dict) -> str:
    return str(config.get("training", {}).get("trainable_scope", "full"))


def _forcing_extender_conditions_on_rc(config: dict) -> bool:
    return bool(
        config.get("model", {}).get("parameters", {}).get(
            "forcing_extender_condition_on_rc", False
        )
    )


def _forcing_spatial_mode(config: dict) -> str:
    return str(
        config.get("model", {}).get("parameters", {}).get(
            "forcing_spatial_mode", "broadcast"
        )
    )


def _physics_extender_signature(config: dict) -> tuple[int, float] | None:
    if _forcing_spatial_mode(config) != "physics_extender":
        return None
    params = config.get("model", {}).get("parameters", {})
    return (
        int(params.get("forcing_extender_physics_hidden", 16)),
        float(params.get("forcing_extender_interface_x_norm", 0.5)),
    )


def _validate_resume_compatibility(checkpoint_conf: dict, current_conf: dict) -> None:
    ckpt_optimizer = _optimizer_name(checkpoint_conf)
    curr_optimizer = _optimizer_name(current_conf)
    ckpt_scheduler = _scheduler_type(checkpoint_conf)
    curr_scheduler = _scheduler_type(current_conf)
    ckpt_scope = _trainable_scope(checkpoint_conf)
    curr_scope = _trainable_scope(current_conf)
    ckpt_extender_rc = _forcing_extender_conditions_on_rc(checkpoint_conf)
    curr_extender_rc = _forcing_extender_conditions_on_rc(current_conf)
    ckpt_spatial_mode = _forcing_spatial_mode(checkpoint_conf)
    curr_spatial_mode = _forcing_spatial_mode(current_conf)
    ckpt_physics = _physics_extender_signature(checkpoint_conf)
    curr_physics = _physics_extender_signature(current_conf)

    if (ckpt_optimizer != curr_optimizer
            or ckpt_scheduler != curr_scheduler
            or ckpt_scope != curr_scope
            or ckpt_extender_rc != curr_extender_rc
            or ckpt_spatial_mode != curr_spatial_mode
            or ckpt_physics != curr_physics):
        raise ValueError(
            "Incompatible resume state: checkpoint uses "
            f"optimizer={ckpt_optimizer}, scheduler={ckpt_scheduler}, "
            f"trainable_scope={ckpt_scope}, "
            f"forcing_spatial_mode={ckpt_spatial_mode}, "
            f"forcing_extender_condition_on_rc={ckpt_extender_rc}, "
            f"physics_extender_signature={ckpt_physics}, but current config uses "
            f"optimizer={curr_optimizer}, scheduler={curr_scheduler}, "
            f"trainable_scope={curr_scope}, "
            f"forcing_spatial_mode={curr_spatial_mode}, "
            f"forcing_extender_condition_on_rc={curr_extender_rc}, "
            f"physics_extender_signature={curr_physics}. "
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
    params["forcing_extender_rc_cond_index"] = (
        dims.forcing_extender_rc_cond_index
    )


def _dump_resolved_config(run_path: Path, config: dict) -> None:
    with (run_path / "config_used.yaml").open("w", encoding="utf-8") as config_file:
        yaml.safe_dump(config, config_file, sort_keys=False)


def write_diffusion_geometry_bias(
    model: torch.nn.Module,
    output_path: str | Path,
    *,
    checkpoint_epoch: int,
) -> bool:
    extender = getattr(model, "boundary_extender", None)
    mlps = getattr(extender, "diffusion_geometry_bias_mlps", None)
    if mlps is None:
        return False

    contexts = [
        (x_norm, t_bar_norm, rc_norm)
        for x_norm in DIFFUSION_GEOMETRY_X_NORM
        for t_bar_norm in DIFFUSION_GEOMETRY_T_BAR_NORM
        for rc_norm in DIFFUSION_GEOMETRY_RC_NORM
    ]
    parameter = next(mlps.parameters())
    x_norm = torch.tensor(
        [item[0] for item in contexts],
        device=parameter.device,
        dtype=parameter.dtype,
    )[:, None]
    t_bar_norm = torch.tensor(
        [item[1] for item in contexts],
        device=parameter.device,
        dtype=parameter.dtype,
    )[:, None]
    rc_norm = torch.tensor(
        [item[2] for item in contexts],
        device=parameter.device,
        dtype=parameter.dtype,
    )[:, None]
    with torch.no_grad():
        coefficients = extender.diffusion_geometry_coefficients(
            x_norm, t_bar_norm, rc_norm
        ).detach().cpu()

    path = Path(output_path)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=DIFFUSION_GEOMETRY_FIELDNAMES)
        writer.writeheader()
        for block in range(coefficients.size(0)):
            for head in range(coefficients.size(-1)):
                for context_index, (x_value, t_value, rc_value) in enumerate(contexts):
                    writer.writerow(
                        {
                            "checkpoint_epoch": int(checkpoint_epoch),
                            "block": block,
                            "head": head,
                            "x_norm": x_value,
                            "t_bar_norm": t_value,
                            "I_cross": int(
                                x_value
                                > extender.diffusion_geometry_interface_x_norm
                            ),
                            "R_c_norm": rc_value,
                            "c": float(
                                coefficients[block, context_index, head]
                            ),
                        }
                    )
    return True


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
            "trainable_scope": training_cfg.get("trainable_scope", "full"),
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


def configure_trainable_scope(model: torch.nn.Module, scope: str) -> tuple[int, int]:
    """Apply the one supported selective-adaptation boundary.

    Returns ``(trainable_parameters, total_parameters)`` for run provenance.
    """
    if scope not in {"full", "boundary_extender"}:
        raise ValueError(
            f"training.trainable_scope must be 'full' or 'boundary_extender', got {scope!r}."
        )

    for parameter in model.parameters():
        parameter.requires_grad_(scope == "full")

    if scope == "boundary_extender":
        extender = getattr(model, "boundary_extender", None)
        if extender is None:
            raise ValueError(
                "training.trainable_scope='boundary_extender' requires "
                "model.parameters.forcing_spatial_mode='boundary_extender' or "
                "'physics_extender'."
            )
        for parameter in extender.parameters():
            parameter.requires_grad_(True)

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    return trainable, total


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
    global_step_start: int = 0,
    scheduler=None,
) -> dict[str, float]:
    """Train one epoch (supervised) and return a dict of epoch metrics.

    Under DDP, reduces sums (loss, MSE, ||y||^2, iface MSE, ||y_iface||^2, sample
    counts, and the unified per-sample metric sums) across ranks and computes
    true global metrics from those sums -- not an average of per-rank ratios.

    The unified headline metrics (``nrmse``, ``rmse_K``, ``node_jump_rmse_K``,
    ``node_jump_nrmse``) are accumulated as **summed per-sample** scalars and
    divided by ``n_samples`` after the reduce, so the train headline equals the
    mean-over-pairs definition used by validate/evaluate (not a pooled RMSE).
    ``max_err_K`` is a worst-case over the split (MAX-reduced under DDP).
    """
    if dist_info is None:
        dist_info = get_dist_info()

    model.train()

    _t_train_start = time.perf_counter()
    n_model_forwards = 0
    grad_norm_sum = 0.0
    n_grad_steps = 0
    successful_updates = 0
    lr_first = float("nan")
    lr_last = float("nan")

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

    global_step = int(global_step_start)
    _sync_timing = torch.device(device).type == "cuda"
    if _sync_timing:
        torch.cuda.reset_peak_memory_stats(device)

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
        n_model_forwards += 1
        loss = loss_fn(y_pred, y_batch, iface_x)
        loss.backward()

        if grad_clip is not None:
            _gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            if _gn is not None:
                grad_norm_sum += float(_gn)
                n_grad_steps += 1
        lr_used = float(optimizer.param_groups[0]["lr"])
        optimizer.step()
        _record_successful_update(lr_used)
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
             gnrmse_sum, node_jump_gnrmse_sum],
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

    epoch_train_s = time.perf_counter() - _t_train_start
    grad_norm = (grad_norm_sum / n_grad_steps) if n_grad_steps > 0 else None
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
        "epoch_train_s": epoch_train_s,
        "n_model_forwards": int(n_model_forwards),
        "grad_norm": grad_norm,
        "opt_peak_mem_mb": (
            torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
            if _sync_timing else None
        ),
        "successful_updates": int(successful_updates),
        "lr_first": lr_first,
        "lr_last": lr_last,
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
    # Per-pair mean Kelvin RMSE. `val_rel_l2` is a POOLED ratio-of-sums,
    # sqrt(sum|err|^2 / sum|y|^2), so it is dominated by the highest-energy
    # pairs and can fall while most pairs get worse. On the N=16 sinusoid
    # fine-tune the two disagreed in sign: pooled read 9.78 -> 9.55 while the
    # per-pair mean over the same val set read 5.82 -> 8.24. Select on this
    # when the reported headline is a per-pair average.
    "val_rmse_K": "rmse_K",
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
        assert_dataset_problem_version(spec, config["data"]["t_grid_path"])
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
        normalization_provenance = build_normalization_provenance(
            trajectories,
            train_ids,
            x_grid,
            y_grid,
            t_grid,
            max_time=norm_max_time,
            data_paths=config["data"],
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
        normalization_provenance = getattr(
            training_set.dataset, "normalization_provenance", None
        )

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
        use_temporal_encoder=dims.use_temporal_encoder,
        use_forcing_time_aug=dims.use_forcing_time_aug,
        forcing_cond_mode=model_cfg.get("forcing_cond_mode", "both"),
        forcing_spatial_mode=model_cfg.get("forcing_spatial_mode", "broadcast"),
        forcing_extender_grid_size=model_cfg.get("forcing_extender_grid_size", 16),
        forcing_extender_heads=model_cfg.get("forcing_extender_heads", 4),
        forcing_extender_depth=model_cfg.get("forcing_extender_depth", 1),
        forcing_extender_condition_on_rc=model_cfg.get(
            "forcing_extender_condition_on_rc", False
        ),
        forcing_extender_rc_cond_index=dims.forcing_extender_rc_cond_index,
        forcing_extender_physics_hidden=model_cfg.get(
            "forcing_extender_physics_hidden", 16
        ),
        forcing_extender_interface_x_norm=model_cfg.get(
            "forcing_extender_interface_x_norm", 0.5
        ),
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
        if _forcing_spatial_mode(init_ckpt["conf"]) != _forcing_spatial_mode(config):
            raise ValueError(
                "init_from_checkpoint cannot cross forcing_spatial_mode "
                "architectures; start this ablation from a fresh initialization."
            )
        if _physics_extender_signature(
            init_ckpt["conf"]
        ) != _physics_extender_signature(config):
            raise ValueError(
                "init_from_checkpoint requires matching physics-extender hidden "
                "width and interface location; start this ablation from a fresh "
                "initialization."
            )
        if _forcing_extender_conditions_on_rc(
            init_ckpt["conf"]
        ) != _forcing_extender_conditions_on_rc(config):
            raise ValueError(
                "init_from_checkpoint cannot cross "
                "forcing_extender_condition_on_rc architectures because the "
                "domain_lift input width differs; start this ablation from a "
                "fresh initialization."
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

    trainable_scope = _trainable_scope(config)
    trainable_parameters, total_parameters = configure_trainable_scope(
        fno, trainable_scope,
    )
    if is_main:
        print(
            f"Trainable scope: {trainable_scope} "
            f"({trainable_parameters:,}/{total_parameters:,} parameters, "
            f"{100.0 * trainable_parameters / total_parameters:.2f}%)."
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
            # Valid because the used-parameter set is fixed per run (forward
            # branches are config/shape-driven, not data-value-driven), so the
            # autograd order recorded on the first iteration holds for all steps.
            static_graph=True,
        )
        fno_unwrapped = fno.module
    else:
        fno_unwrapped = fno

    optimizer = build_optimizer(
        config, (parameter for parameter in fno.parameters() if parameter.requires_grad),
    )
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
        "epoch", "train_loss",
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
        # Training diagnostics.
        "train_grad_norm", "epoch_train_s", "epoch_wall_s", "n_model_forwards",
        "opt_peak_mem_mb",
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

    # Step-0 reference (Capability 7). For a warm start, run ONE validation pass
    # on the freshly-loaded model A weights BEFORE any optimizer step, so the
    # run's own metrics contain the honest "before" row rather than an
    # epoch-0-post-step approximation of it. Logged as epoch -1 so it is
    # auditable. Only on a fresh warm-start (skip on auto-resume, which already
    # has trained state).
    #
    # Unconditional, not gated on `lead_cutoff_time`: a fine-tune's headline
    # result is E_0 -> E_best, and that is worth one validation pass on every
    # warm start. When a cutoff IS set the same pass additionally captures
    # `pre_cutoff_ref` = rmse_K^A_{tau<=tc}, which is model A exactly and is what
    # the long-lead selection guard compares against.
    if is_main and (not resuming) and init_from_checkpoint:
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
        print(
            f"[step-0 ref] model A before any optimizer step: "
            f"val rel_l2={float(ref_metrics['rel_l2']):.4f}%, "
            f"rmse_K={float(ref_metrics['rmse_K']):.4f}K",
            flush=True,
        )
        if lead_cutoff_time is not None:
            pre_cutoff_ref = float(ref_metrics["rmse_K_lead_le_tc"])
            print(
                f"[step-0 ref] pre_cutoff_ref (rmse_K tau<={lead_cutoff_time}) = "
                f"{pre_cutoff_ref:.4f}K; post-cutoff rmse_K = "
                f"{ref_metrics['rmse_K_lead_gt_tc']:.4f}K",
                flush=True,
            )
        # Write every val_* column, not just the headline pair. A fine-tune may
        # legitimately select on `val_rmse_K` or `val_nrmse` instead of
        # `val_rel_l2`, and a reference row that omits the chosen metric leaves
        # the run unable to answer "did this beat model A" in its own units.
        # The lead-split keys exist only when a cutoff is set.
        ref_row = {
            "epoch": -1,
            "lr_first": float("nan"),
            "lr_last": float("nan"),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "is_best": 0,
        }
        for column in fieldnames:
            if not column.startswith("val_"):
                continue
            value = ref_metrics.get(column[len("val_"):], None)
            ref_row[column] = float("nan") if value is None else float(value)
        csv_writer.writerow(ref_row)
        csv_file.flush()

    should_stop = False

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

        train_metrics = train_one_epoch(
            model=fno, train_loader=training_set, optimizer=optimizer,
            loss_fn=loss_fn, device=device, iface_mask=iface_mask,
            grad_clip=grad_clip, dist_info=dist_info,
            use_per_sample_interface=use_per_sample_interface,
            x_grid=x_grid, interface_x=loss_cfg.get("interface_x", 0.5),
            sigma_global=sigma_global,
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
                            "normalization_provenance": normalization_provenance,
                            "global_optimizer_step": int(global_optimizer_step),
                        },
                        best_path,
                    )
                    write_diffusion_geometry_bias(
                        fno_unwrapped,
                        run_path / "diffusion_geometry_bias.csv",
                        checkpoint_epoch=epoch,
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
                    "normalization_provenance": normalization_provenance,
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
                    "opt_peak_mem_mb": _tm_opt("opt_peak_mem_mb"),
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
