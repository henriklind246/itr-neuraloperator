import argparse
import ast
import copy
import json
import os
from pathlib import Path

from src.operators.distributed import cleanup_distributed, init_distributed, is_main_process
from src.operators.train import load_config, run_config_seeds
from src.operators.utils import resolve_device

DATA_FILE_NAMES = {
    "t_grid_path": "t_grid.npy",
    "x_grid_path": "x_grid.npy",
    "y_grid_path": "y_grid.npy",
    "trajectories.npy": "trajectories.npy",
    "sim_params_path": "sim_params.npy",
}


def _parse_override_value(raw: str) -> object:
    lowered = raw.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False

    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw


def _apply_override(config: dict, key_path: str, value: object) -> None:
    parts = key_path.split(".")
    current = config

    for idx, part in enumerate(parts[:-1]):
        if not isinstance(current, dict) or part not in current:
            prefix = ".".join(parts[: idx + 1])
            raise KeyError(f"Unknown config path: {prefix}")
        current = current[part]

    leaf = parts[-1]
    if not isinstance(current, dict) or leaf not in current:
        raise KeyError(f"Unknown config path: {key_path}")
    current[leaf] = value


def _sync_data_paths_from_data_dir(config: dict, overridden_keys: set[str]) -> None:
    if "paths.data_dir" not in overridden_keys:
        return

    data_dir = Path(str(config["paths"]["data_dir"]))
    for key, file_name in DATA_FILE_NAMES.items():
        if f"data.{key}" not in overridden_keys:
            config["data"][key] = str(data_dir / file_name)


def _validate_fixed_run_config(config: dict) -> None:
    experiment_name = str(config.get("experiment", {}).get("name", "")).strip()
    if not experiment_name:
        raise ValueError("Fixed runs require experiment.name to be provided explicitly.")

    config_id = config.get("config_id")
    if config_id is None or str(config_id).strip() == "":
        raise ValueError("Fixed runs require config_id to be provided explicitly.")

    seeds = config.get("training", {}).get("seeds")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("training.seeds must be a non-empty list.")

    run_dir = config.get("training", {}).get("run", {}).get("run_dir")
    if not isinstance(run_dir, str) or not run_dir.strip():
        raise ValueError("training.run.run_dir must be set before launching training.")


def _resolve_run_dir(config: dict) -> Path:
    runs_root = Path(str(config["paths"]["runs_root"]))
    experiment_name = str(config["experiment"]["name"])
    config_id = str(config["config_id"])
    return runs_root / experiment_name / f"config{config_id}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a fixed training config.")
    parser.add_argument("overrides", nargs="*", help="Config overrides in dotted key=value form.")
    args = parser.parse_args()

    dist_info = init_distributed()
    main_rank = is_main_process()

    try:
        project_root = Path(__file__).resolve().parents[1]
        os.environ.setdefault("PROJECT_ROOT", project_root.as_posix())

        config = copy.deepcopy(load_config())
        applied_overrides: list[tuple[str, object]] = []

        for raw_override in args.overrides:
            if "=" not in raw_override:
                raise ValueError(f"Invalid override '{raw_override}'. Expected key=value.")
            key_path, raw_value = raw_override.split("=", 1)
            if not key_path:
                raise ValueError(f"Invalid override '{raw_override}'. Expected non-empty key.")
            value = _parse_override_value(raw_value)
            _apply_override(config, key_path, value)
            applied_overrides.append((key_path, value))

        _sync_data_paths_from_data_dir(config, {key for key, _ in applied_overrides})

        run_dir = _resolve_run_dir(config)
        config.setdefault("training", {})
        config["training"].setdefault("run", {})
        config["training"]["run"]["run_dir"] = str(run_dir)

        _validate_fixed_run_config(config)

        device = resolve_device(
            config["training"].get("device", "auto"),
            local_rank=dist_info.local_rank if dist_info.is_distributed else None,
        )
        if main_rank:
            print(f"Using fixed experiment namespace: {config['experiment']['name']}")
            print(f"Config ID: {config['config_id']}")
            print(f"Run directory: {run_dir}")
            print(f"Device: {device}")
            print(f"World size: {dist_info.world_size}")
            if applied_overrides:
                print("Applied overrides:")
                for key, value in applied_overrides:
                    print(f"  {key}={value!r}")

        summary = run_config_seeds(config, base_run_dir=run_dir, seeds=[int(seed) for seed in config["training"]["seeds"]])
        if main_rank:
            print(json.dumps(summary, indent=2))
        return 0
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    raise SystemExit(main())
