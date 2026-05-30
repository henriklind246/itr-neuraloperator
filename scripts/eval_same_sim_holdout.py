import argparse
import ast
import copy
import csv
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


def _resolve_run_dir(config: dict) -> Path:
    runs_root = Path(str(config["paths"]["runs_root"]))
    experiment_name = str(config["experiment"]["name"])
    config_id = str(config["config_id"])
    return runs_root / experiment_name / f"config{config_id}"


def _latest_validation_value(metrics_path: Path) -> float | None:
    if not metrics_path.exists():
        return None
    with metrics_path.open("r", newline="") as f:
        rows = [row for row in csv.DictReader(f) if row.get("val_rel_l2")]
    if not rows:
        return None
    return float(rows[-1]["val_rel_l2"])


def _build_same_sim_loaders(config: dict, val_pair_frac: float, split_seed: int):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from data.dataset import (
        SnapshotPairDataset,
        compute_global_stats,
        load_sim_data,
        load_solver_dt,
        problem_from_config,
        split_pairs_within_sims,
        split_sim_ids,
    )

    trajectories, x_grid, y_grid, t_grid = load_sim_data(
        sim_traj_path=config["data"]["trajectories.npy"],
        x_grid_path=config["data"]["x_grid_path"],
        y_grid_path=config["data"]["y_grid_path"],
        t_grid_path=config["data"]["t_grid_path"],
    )
    sim_params = np.load(config["data"]["sim_params_path"], allow_pickle=True)
    solver_dt = load_solver_dt(config["data"]["t_grid_path"])
    train_ids, _, _ = split_sim_ids(trajectories.shape[0], train_frac=0.7, val_frac=0.15, seed=0)
    mu_global, sigma_global = compute_global_stats(trajectories, train_ids)

    train_dataset = SnapshotPairDataset(
        trajectories=trajectories,
        t_grid=t_grid,
        x_grid=x_grid,
        y_grid=y_grid,
        sim_ids=train_ids,
        sim_params=sim_params,
        mu_global=mu_global,
        sigma_global=sigma_global,
        n_snapshots=config["training"].get("n_snapshots", 15),
        noise_std=config["training"].get("noise_std", 0.0),
        dt=solver_dt,
        problem=problem_from_config(config),
    )
    train_pairs, val_pairs = split_pairs_within_sims(train_dataset, val_pair_frac=val_pair_frac, seed=split_seed)

    pin = torch.cuda.is_available()
    workers_cfg = config["training"].get("num_workers", None)
    workers = workers_cfg if workers_cfg is not None else (4 if pin else 0)
    batch_size = config["training"]["batch_size"]

    train_loader = DataLoader(train_pairs, batch_size=batch_size, shuffle=True, pin_memory=pin, num_workers=workers)
    val_loader = DataLoader(val_pairs, batch_size=batch_size, shuffle=False, pin_memory=pin, num_workers=workers)
    return train_loader, val_loader


def main(load_config_fn=None, build_loaders_fn=None, run_one_seed_fn=None) -> None:
    parser = argparse.ArgumentParser(description="Train with same-simulation held-out snapshot pairs as validation.")
    parser.add_argument("--config", default=None, help="Optional config path. Defaults to conf/config.yaml.")
    parser.add_argument("--seed", type=int, default=None, help="Model initialization seed. Defaults to the first configured seed.")
    parser.add_argument("--run-dir", default=None, help="Output directory for this held-out-pair seed run.")
    parser.add_argument("--val-pair-frac", type=float, default=0.1, help="Fraction of pairs per training sim held out for validation.")
    parser.add_argument("--split-seed", type=int, default=0, help="RNG seed for the within-simulation pair split.")
    parser.add_argument("--cross-sim-metrics", default=None, help="Optional prior cross-sim train_metrics.csv path.")
    parser.add_argument("overrides", nargs="*", help="Config overrides in dotted key=value form.")
    args = parser.parse_args()

    if load_config_fn is None or run_one_seed_fn is None:
        from src.operators.train import load_config as _load_config, run_one_seed as _run_one_seed
        load_config_fn = load_config_fn or _load_config
        run_one_seed_fn = run_one_seed_fn or _run_one_seed
    build_loaders_fn = build_loaders_fn or _build_same_sim_loaders

    config = copy.deepcopy(load_config_fn(args.config))
    applied_overrides = []
    for raw_override in args.overrides:
        if "=" not in raw_override:
            raise ValueError(f"Invalid override '{raw_override}'. Expected key=value.")
        key_path, raw_value = raw_override.split("=", 1)
        value = _parse_override_value(raw_value)
        _apply_override(config, key_path, value)
        applied_overrides.append((key_path, value))

    _sync_data_paths_from_data_dir(config, {key for key, _ in applied_overrides})

    base_run_dir = _resolve_run_dir(config)
    config.setdefault("training", {})
    config["training"].setdefault("run", {})
    config["training"]["run"]["run_dir"] = str(base_run_dir)
    seed = int(args.seed if args.seed is not None else config["training"]["seeds"][0])
    run_dir = Path(args.run_dir) if args.run_dir else base_run_dir / f"seed{seed}"

    print(f"Using same-sim experiment namespace: {config['experiment']['name']}")
    print(f"Config ID: {config['config_id']}")
    print(f"Run directory: {run_dir}")
    if applied_overrides:
        print("Applied overrides:")
        for key, value in applied_overrides:
            print(f"  {key}={value!r}")

    train_loader, val_loader = build_loaders_fn(config, args.val_pair_frac, args.split_seed)
    result = run_one_seed_fn(
        config,
        seed=seed,
        run_dir=run_dir,
        train_loader_override=train_loader,
        val_loader_override=val_loader,
    )

    same_sim_final = _latest_validation_value(run_dir / "train_metrics.csv")
    cross_metrics = Path(args.cross_sim_metrics) if args.cross_sim_metrics else base_run_dir / f"seed{seed}" / "train_metrics.csv"
    cross_sim_final = _latest_validation_value(cross_metrics)

    print("\n===== Same-Sim Held-Out Pair Diagnostic =====")
    print(f"seed: {seed}")
    print(f"same-sim held-out best_val_rel_l2: {result['best_val']:.6f}")
    if same_sim_final is not None:
        print(f"same-sim held-out final_val_rel_l2: {same_sim_final:.6f}")
    if cross_sim_final is not None:
        print(f"cross-sim final_val_rel_l2: {cross_sim_final:.6f} ({cross_metrics})")
    else:
        print(f"cross-sim final_val_rel_l2: unavailable ({cross_metrics})")


if __name__ == "__main__":
    main()
