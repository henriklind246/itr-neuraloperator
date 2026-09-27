"""Score one fixed checkpoint across a ladder of evaluation grid resolutions.

Generates one held-out dataset per N x N grid from the same draw family, so every
grid sees the same simulation latents, evaluates the unchanged checkpoint on each
with the production ``eval_all_seeds`` path, and reports only two numbers per
grid:

- ``field_gnrmse_pct``: pooled field RMSE (K) / training temperature-rise scale.
- ``node_jump_gnrmse_pct``: mean per-pair interface node-jump RMSE / training sigma.

The node jump is the difference between the two grid nodes adjacent to the
interface, so its target itself changes with grid spacing.

    python scripts/run_resolution_study.py <run_root or .pt> \\
        --out-dir resolution_studies/<name> --device mps
"""

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_rollout_study import (  # noqa: E402
    _git_commit,
    _sha256_file,
    check_disjoint,
    generate_dataset,
    materialize_checkpoint,
    resolve_checkpoint,
    training_rng_seed,
)

METRICS = ("field_gnrmse_pct", "node_jump_gnrmse_pct")
_REPORT_KEYS = {
    "field_gnrmse_pct": "test_gnrmse_pct",
    "node_jump_gnrmse_pct": "test_node_jump_gnrmse_pct",
}
METRIC_DEFINITIONS = {
    "field_gnrmse_pct": (
        "100 * sqrt(sum squared physical error / total cells) / "
        "sqrt(sigma_train^2 + (mu_train - 300 K)^2), pooled over every pair"
    ),
    "node_jump_gnrmse_pct": (
        "100 * mean over pairs of the interface node-jump RMSE / sigma_train"
    ),
}


def _grid_sampled(value, n):
    return isinstance(value, np.ndarray) and (value.ndim >= 2 or n in value.shape)


def _latent_signature(value, n):
    """Grid-independent part of one simulation's parameters on an n x n grid.

    Fields sampled on the grid (initial temperature, per-row profiles) are
    dropped; every other latent, including short parameter arrays, must then
    match across resolutions for the ladder to compare the same simulations.
    """
    if isinstance(value, dict):
        return {
            k: _latent_signature(v, n) for k, v in sorted(value.items())
            if not _grid_sampled(v, n)
        }
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_latent_signature(v, n) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def check_matched_ladder(data_dirs):
    """Refuse a ladder whose grids differ in anything but spatial resolution."""
    reference_n, reference_dir = next(iter(data_dirs.items()))
    ref_t = np.load(reference_dir / "t_grid.npy")
    ref_latents = [
        _latent_signature(p, reference_n)
        for p in np.load(reference_dir / "sim_params.npy", allow_pickle=True)
    ]
    for n, data_dir in data_dirs.items():
        shape = np.load(data_dir / "trajectories.npy", mmap_mode="r").shape
        if tuple(shape[2:]) != (n, n):
            raise SystemExit(f"{data_dir} holds a {shape[2:]} grid, expected ({n}, {n})")
        if not np.array_equal(np.load(data_dir / "t_grid.npy"), ref_t):
            raise SystemExit(f"r{n} time grid differs from r{reference_n}")
        latents = [
            _latent_signature(p, n)
            for p in np.load(data_dir / "sim_params.npy", allow_pickle=True)
        ]
        if latents != ref_latents:
            raise SystemExit(
                f"r{n} simulation latents differ from r{reference_n}; the grids "
                "would not score the same simulations"
            )


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Field and node-jump GNRMSE of one checkpoint per grid resolution.",
    )
    p.add_argument("checkpoint", help="A run_root containing seed*/ or a bare .pt file.")
    p.add_argument("--seed", default=None,
                   help="Seed to evaluate when checkpoint is a run_root (e.g. 32).")
    p.add_argument("--out-dir", default=None,
                   help="Study directory. Default resolution_studies/<benchmark>_<timestamp>/.")
    p.add_argument("--resolutions", default="100,150,200,256",
                   help="Comma-separated N for the N x N evaluation grids.")
    p.add_argument("--num-sims", type=int, default=24,
                   help="Simulations per grid; every one is evaluated.")
    p.add_argument("--n-snapshots-test", type=int, default=10,
                   help="Snapshots per simulation; pairs per sim are S*(S-1)/2.")
    p.add_argument("--rng-seed", type=int, default=7,
                   help="Draw family for the generated ladder; must differ from the "
                        "checkpoint's training family.")
    p.add_argument("--data-root", default=None,
                   help="Reuse existing r<N>/ dataset dirs under this root instead of "
                        "generating them.")
    p.add_argument("--padding-reference-resolution", type=int, default=None,
                   help="Training grid N used to scale FNO padding. Defaults to the "
                        "checkpoint's own value.")
    p.add_argument("--batch-size", type=int, default=4,
                   help="Evaluation batch size; metrics are pooled and do not depend on it.")
    p.add_argument("--device", default=None, help="cpu, mps, cuda, or auto.")
    p.add_argument("--symlink-checkpoint", action="store_true",
                   help="Permit symlinking the checkpoint when hard-linking fails.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    resolutions = sorted({int(n) for n in args.resolutions.split(",") if n.strip()})
    if not resolutions:
        raise SystemExit("--resolutions must name at least one grid")

    ckpt_path, ckpt, seed_name = resolve_checkpoint(args.checkpoint, args.seed)
    config = ckpt["conf"]
    benchmark = config.get("benchmark", {}).get("name", "forcing")
    representation = config.get("benchmark", {}).get("representation", "temporal_encoder")
    padding_ref = args.padding_reference_resolution or config.get("model", {}).get(
        "parameters", {}
    ).get("padding_reference_resolution")
    if padding_ref is None:
        raise SystemExit(
            "the checkpoint records no padding_reference_resolution; pass "
            "--padding-reference-resolution with its training grid N, or the FNO "
            "padding width would change physical size across grids"
        )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else (
        PROJECT_ROOT / "resolution_studies" / f"{benchmark}_{stamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    run_root = out_dir / "run"
    seed_dir = run_root / seed_name
    placement = materialize_checkpoint(
        ckpt_path, seed_dir / "fno2d_best.pt", args.symlink_checkpoint
    )
    ckpt_sha = _sha256_file(seed_dir / "fno2d_best.pt")

    if args.data_root is not None:
        data_root = Path(args.data_root).expanduser()
        dataset_rng_seed = None
    else:
        train_seed, train_seed_source = training_rng_seed(config)
        check_disjoint(args.rng_seed, train_seed, train_seed_source)
        data_root = out_dir / "data"
        dataset_rng_seed = args.rng_seed
    data_dirs = {n: data_root / f"r{n}" for n in resolutions}

    n_snap = args.n_snapshots_test
    pairs_per_sim = n_snap * (n_snap - 1) // 2
    print(f"benchmark   : {benchmark} / {representation}")
    print(f"checkpoint  : {ckpt_path} ({placement}, sha256 {ckpt_sha[:12]})")
    print(f"study dir   : {out_dir}")
    print(f"grids       : {', '.join(f'{n}x{n}' for n in resolutions)}")
    print(f"pairs/grid  : {args.num_sims} sims x {pairs_per_sim}")
    print(flush=True)

    t0 = time.time()
    if dataset_rng_seed is not None:
        for n, data_dir in data_dirs.items():
            print(f"[{time.time() - t0:7.1f}s] generating r{n}", flush=True)
            generate_dataset(benchmark, data_dir, args.num_sims, n, n, dataset_rng_seed)
    missing = [str(d) for d in data_dirs.values() if not (d / "trajectories.npy").exists()]
    if missing:
        raise SystemExit(f"missing datasets: {missing}")
    check_matched_ladder(data_dirs)

    from src.operators.eval import eval_all_seeds

    rows = []
    n_simulations = None
    for n, data_dir in data_dirs.items():
        print(f"[{time.time() - t0:7.1f}s] evaluating r{n}", flush=True)
        results = eval_all_seeds(
            str(run_root),
            data_dir=str(data_dir),
            padding_reference_resolution=int(padding_ref),
            eval_all_sims=True,
            n_snapshots_test=n_snap,
            batch_size=args.batch_size,
            device=args.device,
        )
        if len(results) != 1:
            raise SystemExit(f"expected one checkpoint under {run_root}, scored {len(results)}")
        result = results[0]
        n_simulations = int(result["num_sims"])
        row = {"resolution": n}
        for metric in METRICS:
            value = float(result[_REPORT_KEYS[metric]])
            if not math.isfinite(value):
                raise SystemExit(f"r{n} {metric} is not finite ({value})")
            row[metric] = value
        rows.append(row)

    with open(out_dir / "resolution_study.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["resolution", *METRICS])
        w.writeheader()
        w.writerows(rows)

    spatial_input = config.get("benchmark", {}).get("spatial_input", {}) or {}
    payload = {
        "study": "resolution invariance",
        "created_utc": stamp,
        "argv": sys.argv,
        "git_commit": _git_commit(),
        "benchmark": benchmark,
        "representation": representation,
        "material_side": bool(spatial_input.get("material_side", False)),
        "checkpoint_source_path": str(ckpt_path),
        "checkpoint_sha256": ckpt_sha,
        "checkpoint_placement": placement,
        "checkpoint_seed": str(ckpt.get("seed", seed_name.removeprefix("seed"))),
        "checkpoint_epoch": int(ckpt["epoch"]),
        "checkpoint_best_val": float(ckpt.get("best_val", float("nan"))),
        "weights_unchanged": True,
        "dataset_root": str(data_root),
        "dataset_rng_seed": dataset_rng_seed,
        "num_sims": n_simulations,
        "n_snapshots_test": n_snap,
        "pairs_per_sim": pairs_per_sim,
        "evaluate_all_sims": True,
        "padding_reference_resolution": int(padding_ref),
        "batch_size": args.batch_size,
        "device": args.device,
        "space": "physical temperature (K), normalized by training-set statistics",
        "metric_definitions": METRIC_DEFINITIONS,
        "resolutions": resolutions,
        "per_resolution": rows,
    }
    (out_dir / "resolution_study.json").write_text(json.dumps(payload, indent=2))

    print(f"\n{'grid':>9}  {'field GNRMSE (%)':>17}  {'node-jump GNRMSE (%)':>21}")
    for row in rows:
        n = row["resolution"]
        print(f"{f'{n}x{n}':>9}  {row['field_gnrmse_pct']:>17.4f}  "
              f"{row['node_jump_gnrmse_pct']:>21.4f}")
    print(f"\nStudy written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
