import argparse
import ast
import copy
import json
import os
from pathlib import Path

from src.operators.train import load_config
from src.operators.train_pino import run_config_seeds_pino

DATA_FILE_NAMES = {
    "t_grid_path": "t_grid.npy",
    "x_grid_path": "x_grid.npy",
    "y_grid_path": "y_grid.npy",
    "trajectories.npy": "trajectories.npy",
    "sim_params_path": "sim_params.npy",
}

# Overrides that select a config GROUP (merged inside load_config via env var)
# rather than a leaf value; consumed before load_config, not applied as dotted
# paths. model.type is descriptive-only for the PINO path (always CViT).
GROUP_ENV = {"benchmark": "BENCHMARK", "representation": "REPRESENTATION"}


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


def _resolve_run_dir(config: dict) -> Path:
    runs_root = Path(str(config["paths"]["runs_root"]))
    experiment_name = str(config["experiment"]["name"])
    config_id = str(config["config_id"])
    return runs_root / experiment_name / f"config{config_id}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Physics-only (PINO) CViT training on the diffusion benchmark."
    )
    parser.add_argument("overrides", nargs="*", help="Config overrides in dotted key=value form.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("PROJECT_ROOT", project_root.as_posix())

    # Pull out group-selecting and descriptive overrides before load_config so
    # the correct benchmark/representation group is merged.
    dotted: list[str] = []
    for raw in args.overrides:
        if "=" not in raw:
            raise ValueError(f"Invalid override '{raw}'. Expected key=value.")
        key, val = raw.split("=", 1)
        if key in GROUP_ENV:
            os.environ[GROUP_ENV[key]] = val
        elif key == "model.type":
            continue  # PINO path is always CViT; kept for CLI symmetry
        else:
            dotted.append(raw)

    os.environ.setdefault("BENCHMARK", "diffusion")
    os.environ.setdefault("REPRESENTATION", "temporal_encoder")

    config = copy.deepcopy(load_config())

    applied: list[tuple[str, object]] = []
    for raw in dotted:
        key_path, raw_value = raw.split("=", 1)
        if not key_path:
            raise ValueError(f"Invalid override '{raw}'. Expected non-empty key.")
        value = _parse_override_value(raw_value)
        _apply_override(config, key_path, value)
        applied.append((key_path, value))

    _sync_data_paths_from_data_dir(config, {k for k, _ in applied})

    experiment_name = str(config.get("experiment", {}).get("name", "")).strip()
    if not experiment_name:
        raise ValueError("PINO runs require experiment.name to be provided explicitly.")

    seeds = config.get("training", {}).get("seeds")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("training.seeds must be a non-empty list.")

    run_dir = _resolve_run_dir(config)
    config.setdefault("training", {}).setdefault("run", {})
    config["training"]["run"]["run_dir"] = str(run_dir)

    print(f"Benchmark: {os.environ['BENCHMARK']} / {os.environ['REPRESENTATION']}")
    print(f"Experiment: {experiment_name}  config{config['config_id']}")
    print(f"Run directory: {run_dir}")
    if applied:
        print("Applied overrides:")
        for key, value in applied:
            print(f"  {key}={value!r}")

    summary = run_config_seeds_pino(
        config, base_run_dir=run_dir, seeds=[int(s) for s in seeds]
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
