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
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT version_num FROM alembic_version LIMIT 1")
        row = cursor.fetchone()
        conn.close()
        return row is not None  # table must have at least one version row
    except Exception:
        return False


def _stamp_optuna_alembic(db_path: Path) -> bool:
    """Manually stamp alembic_version using Optuna's migration scripts.

    Returns True if the DB is valid after stamping.
    """
    try:
        import optuna
        from alembic.config import Config
        from alembic.command import stamp as alembic_stamp

        optuna_rdb = Path(optuna.__file__).parent / "storages" / "_rdb"
        config = Config(str(optuna_rdb / "alembic.ini"))
        config.set_main_option("script_location", str(optuna_rdb / "alembic"))
        config.set_main_option("sqlalchemy.url", f"sqlite:///{db_path.as_posix()}")
        alembic_stamp(config, "head")
        return _validate_optuna_db(db_path)
    except Exception:
        return False


def _pre_init_optuna_storage(project_root: Path, experiment_name: str) -> None:
    """Pre-initialize the Optuna SQLite DB so the Hydra sweeper finds a valid schema.

    On some platforms (notably Windows), Optuna's RDBStorage may fail to properly
    populate the alembic_version table during schema creation. This function
    uses two strategies:

    1. Create the study with skip_compatibility_check=True to bypass the
       premature version assertion.
    2. If the DB is still invalid (empty alembic_version), manually stamp
       the version using Optuna's own alembic migration scripts.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    db_path = _resolve_runs_root(project_root) / experiment_name / "optuna_study.db"
    storage_url = f"sqlite:///{db_path.as_posix()}"

    # Strategy 1: skip_compatibility_check bypasses the premature version
    # assertion that fails on fresh databases with some Optuna versions.
    try:
        from optuna.storages import RDBStorage
        storage = RDBStorage(url=storage_url, skip_compatibility_check=True)
        optuna.create_study(
            storage=storage,
            study_name=experiment_name,
            load_if_exists=True,
        )
        if _validate_optuna_db(db_path):
            return
    except Exception:
        pass

    # Strategy 2: Tables may exist but alembic_version is empty.
    # Manually stamp the version using Optuna's alembic config.
    if db_path.exists() and _stamp_optuna_alembic(db_path):
        return

    # Both strategies failed — print diagnostics.
    print(f"Warning: Could not pre-initialize Optuna DB.")
    print(f"  Optuna version installed: {optuna.__version__}")
    print(f"  Storage URL: {storage_url}")

    # Clean up the corrupt DB so the Hydra sweeper can try fresh.
    # On Windows, SQLite may hold a file lock, so ignore OSError.
    try:
        if db_path.exists():
            db_path.unlink()
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
