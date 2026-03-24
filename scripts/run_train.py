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


def _patch_optuna_210_fresh_db_bug():
    """Optuna 2.10.0 asserts on fresh SQLite DBs (empty alembic_version).

    Force ``skip_compatibility_check=True`` for all RDBStorage instances
    so the sweeper can create and reuse DBs without hitting the assertion.
    Only activates for Optuna 2.x; 3.x+ fixed the issue upstream.

    Also injects SQLite-specific engine kwargs (StaticPool, busy timeout)
    to prevent ``database is locked`` errors that arise from SQLAlchemy's
    default connection pooling opening multiple file handles.
    """
    try:
        import optuna
        if not optuna.__version__.startswith("2."):
            return
        from optuna.storages._rdb import storage as rdb_mod
        _orig_init = rdb_mod.RDBStorage.__init__

        def _patched_init(self, url, engine_kwargs=None,
                          skip_compatibility_check=False, **kwargs):
            if engine_kwargs is None:
                engine_kwargs = {}
            # SQLite: use a single shared connection (StaticPool) so
            # SQLAlchemy never opens competing file handles, and set a
            # 30-second busy-timeout as a safety net.
            if url and "sqlite" in url:
                from sqlalchemy.pool import StaticPool
                engine_kwargs.setdefault("poolclass", StaticPool)
                ca = engine_kwargs.get("connect_args", {})
                ca.setdefault("timeout", 30)
                ca.setdefault("check_same_thread", False)
                engine_kwargs["connect_args"] = ca
            _orig_init(self, url, engine_kwargs=engine_kwargs,
                       skip_compatibility_check=True, **kwargs)

        rdb_mod.RDBStorage.__init__ = _patched_init
    except Exception:
        pass


_patch_optuna_210_fresh_db_bug()


def _resolve_runs_root(project_root: Path) -> Path:
    """Return RUNS_ROOT from env (HPC) or fall back to project_root/runs (local)."""
    return Path(os.environ.get("RUNS_ROOT", project_root / "runs"))

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


def _validate_optuna_db(db_path: Path) -> bool:
    """Return True if db_path doesn't exist or is a valid Optuna SQLite DB."""
    if not db_path.exists():
        return True
    if db_path.stat().st_size == 0:
        return False
    try:
        import sqlite3
        conn = sqlite3.connect(str(db_path), timeout=5)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT version_num FROM alembic_version LIMIT 1")
            row = cursor.fetchone()
            return row is not None  # table must have at least one version row
        finally:
            conn.close()
    except Exception:
        return False


def _pre_init_optuna_storage(project_root: Path, experiment_name: str) -> None:
    """Clean up corrupt Optuna DBs before the Hydra sweeper runs.

    The monkey-patch ``_patch_optuna_210_fresh_db_bug`` (applied at import
    time) ensures RDBStorage always skips the compatibility check, so the
    sweeper can create fresh DBs on its own. This function only needs to
    remove zero-byte or otherwise corrupt DB files that would confuse
    SQLAlchemy.
    """
    db_path = _resolve_runs_root(project_root) / experiment_name / "optuna_study.db"

    if not db_path.exists():
        return

    if not _validate_optuna_db(db_path):
        try:
            db_path.unlink()
            print(f"Removed corrupt Optuna DB: {db_path}")
        except OSError:
            pass


def _ensure_experiment_name(project_root: Path) -> str:
    experiment_name = os.environ.get(EXPERIMENT_ENV_VAR)
    if not experiment_name:
        runs_root = _resolve_runs_root(project_root)
        experiment_name = _next_experiment_name(runs_root)
        os.environ[EXPERIMENT_ENV_VAR] = experiment_name

    # Create the experiment directory so Optuna can write its SQLite DB
    experiment_dir = _resolve_runs_root(project_root) / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Remove corrupt Optuna DB left by a failed prior run
    db_path = experiment_dir / "optuna_study.db"
    if not _validate_optuna_db(db_path):
        print(f"Warning: removing corrupt Optuna DB at {db_path}")
        db_path.unlink(missing_ok=True)

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


def _next_config_index(experiment_dir: Path) -> int:
    """Find the next config index by scanning existing confN directories.

    Used to offset job_num when resuming a sweep so that new configs
    don't overwrite existing ones.
    """
    if not experiment_dir.exists():
        return 0
    pattern = re.compile(r"^conf(\d+)$")
    indices: list[int] = []
    for entry in experiment_dir.iterdir():
        if entry.is_dir():
            match = pattern.fullmatch(entry.name)
            if match:
                indices.append(int(match.group(1)))
    return (max(indices) + 1) if indices else 0


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

    runs_root = _resolve_runs_root(project_root)
    job_num = _job_number()
    offset = _next_config_index(runs_root / experiment_name)
    config_name = f"conf{job_num + offset}"
    run_dir = runs_root / experiment_name / config_name
    generated_dir = runs_root / experiment_name / "generated"

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
    os.environ.setdefault("PROJECT_ROOT", project_root.as_posix())
    experiment_name = _ensure_experiment_name(project_root)
    print(f"Using experiment namespace: {experiment_name}")
    _pre_init_optuna_storage(project_root, experiment_name)
    run_train()
