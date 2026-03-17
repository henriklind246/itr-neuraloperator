import csv
import json
import os
import re
import shutil
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import yaml
from src.operators.train import run_config_seeds
from src.operators.utils import resolve_device

EXPERIMENT_ENV_VAR = "EXPERIMENT_NAME"

def _next_experiment_name(runs_root: Path) -> str:
    runs_root.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(r"^experiment(\d+)$")
    indices: list[int] = []

    for entry in runs_root.iterdir():
        if not entry.is_dir():
            continue
        match = pattern.fullmatch(entry.name)
        if match:
            indices.append(int(match.group(1)))

    next_index = (max(indices) + 1) if indices else 0
    return f"experiment{next_index}"


def _ensure_experiment_name(project_root: Path) -> str:
    existing = os.environ.get(EXPERIMENT_ENV_VAR)
    if existing:
        return existing

    runs_root = project_root / "runs"
    experiment_name = _next_experiment_name(runs_root)
    os.environ[EXPERIMENT_ENV_VAR] = experiment_name
    return experiment_name


def _job_number() -> int:
    if not HydraConfig.initialized():
        return 0
    job_num = HydraConfig.get().job.get("num")
    if job_num is None:
        return 0
    try:
        return int(job_num)
    except (TypeError, ValueError):
        return 0


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _read_index_rows(index_path: Path) -> list[dict]:
    if not index_path.exists():
        return []
    with index_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _config_sort_key(name: str) -> int:
    match = re.fullmatch(r"conf(\d+)", name)
    return int(match.group(1)) if match else 0


def _update_index_and_best(generated_dir: Path, config_name: str, objective: float, num_seeds: int) -> None:
    generated_dir.mkdir(parents=True, exist_ok=True)
    index_path = generated_dir / "index.csv"
    rows = _read_index_rows(index_path)

    row_map = {row["config_name"]: row for row in rows}
    row_map[config_name] = {
        "config_name": config_name,
        "objective_mean_best_val": f"{objective:.12f}",
        "num_seeds": str(num_seeds),
    }
    merged_rows = sorted(row_map.values(), key=lambda row: _config_sort_key(row["config_name"]))

    with index_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["config_name", "objective_mean_best_val", "num_seeds"])
        writer.writeheader()
        writer.writerows(merged_rows)

    best_row = min(merged_rows, key=lambda row: float(row["objective_mean_best_val"]))
    best_src = generated_dir / f"{best_row['config_name']}.yaml"
    if best_src.exists():
        shutil.copy2(best_src, generated_dir / "best_config.yaml")


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def run_train(cfg: DictConfig) -> float:
    project_root = Path(__file__).resolve().parents[1]
    experiment_name = _ensure_experiment_name(project_root)

    job_num = _job_number()
    config_name = f"conf{job_num}"
    run_dir = project_root / "runs" / experiment_name / config_name
    generated_dir = project_root / "conf" / "generated" / experiment_name

    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(resolved_cfg, dict):
        raise ValueError("Hydra conf must resolve to a dictionary payload.")

    resolved_cfg["config_id"] = job_num
    resolved_cfg.setdefault("experiment", {})
    resolved_cfg["experiment"]["name"] = experiment_name
    resolved_cfg.setdefault("training", {})
    resolved_cfg["training"].setdefault("device", "auto")
    resolved_cfg["training"].setdefault("run", {})
    resolved_cfg["training"]["run"]["run_dir"] = str(run_dir)

    device = resolve_device(resolved_cfg["training"]["device"])
    print(f"Device: {device}")

    seeds = resolved_cfg["training"]["seeds"]
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("training.seeds must be a non-empty list.")

    summary = run_config_seeds(resolved_cfg, base_run_dir=run_dir, seeds=[int(seed) for seed in seeds])
    objective = float(summary["mean_best_val"])

    config_yaml_path = generated_dir / f"{config_name}.yaml"
    _write_yaml(config_yaml_path, resolved_cfg)
    _write_json(
        generated_dir / f"{config_name}.json",
        {
            "experiment": experiment_name,
            "config_name": config_name,
            "objective_mean_best_val": objective,
            "num_seeds": summary["num_seeds"],
            "per_seed": summary["per_seed"],
        },
    )
    _update_index_and_best(
        generated_dir=generated_dir,
        config_name=config_name,
        objective=objective,
        num_seeds=summary["num_seeds"],
    )

    print(f"Finished {experiment_name}/{config_name} objective_mean_best_val={objective:.6f}")
    return objective


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("PROJECT_ROOT", str(project_root))
    experiment_name = _ensure_experiment_name(project_root)
    print(f"Using experiment namespace: {experiment_name}")
    run_train()
