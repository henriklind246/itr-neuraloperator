import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


def _solver_self_difference(base_dir: Path, refined_dir: Path) -> dict:
    """Pooled solver self-difference between the base and refined-grid solves.

    Both datasets hold the SAME CRN sims (identical rng_seed/repeats/values, so
    sim ordering matches) at the same saved physical times (a halved dt is paired
    with a doubled save_stride). The refined grid is the reference; each base
    (coarse) target field is bilinearly lifted onto the refined grid and compared,
    pooling squared error over every sim and every non-initial saved snapshot.
    """
    from scripts.fv_convergence_baseline import interpolate_to_ref

    traj_b = np.load(base_dir / "trajectories.npy", mmap_mode="r")
    traj_r = np.load(refined_dir / "trajectories.npy", mmap_mode="r")
    x_b = np.load(base_dir / "x_grid.npy")
    y_b = np.load(base_dir / "y_grid.npy")
    x_r = np.load(refined_dir / "x_grid.npy")
    y_r = np.load(refined_dir / "y_grid.npy")

    if traj_b.shape[0] != traj_r.shape[0]:
        raise ValueError(
            f"sim-count mismatch: base {traj_b.shape[0]} vs refined {traj_r.shape[0]}"
        )
    if traj_b.shape[1] != traj_r.shape[1]:
        raise ValueError(
            f"saved-time mismatch: base {traj_b.shape[1]} vs refined {traj_r.shape[1]} "
            "(pair --dt refinement with a matching --save-stride)"
        )

    ref_shape = (len(x_r), len(y_r))
    gx, gy = np.meshgrid(x_r, y_r, indexing="ij")
    ref_points = np.stack([gx.ravel(), gy.ravel()], axis=-1)

    num_sims, num_t = int(traj_b.shape[0]), int(traj_b.shape[1])
    sq_diff = sq_ref = 0.0
    n_cells = 0
    for sim_id in range(num_sims):
        for j in range(1, num_t):
            ref_field = np.asarray(traj_r[sim_id, j], dtype=np.float64)
            coarse = np.asarray(traj_b[sim_id, j], dtype=np.float64)
            lifted = interpolate_to_ref(coarse, x_b, y_b, ref_points, ref_shape)
            diff = lifted - ref_field
            sq_diff += float(np.sum(diff ** 2))
            sq_ref += float(np.sum(ref_field ** 2))
            n_cells += ref_field.size

    rel_l2_pct = float(np.sqrt(sq_diff / sq_ref) * 100.0) if sq_ref > 0 else float("nan")
    rmse_k = float(np.sqrt(sq_diff / n_cells)) if n_cells > 0 else float("nan")
    return {
        "global_rel_l2_pct": rel_l2_pct,
        "global_rmse_K": rmse_k,
        "num_sims": num_sims,
        "num_snapshots_compared": num_t - 1,
        "base_grid": [int(len(x_b)), int(len(y_b))],
        "refined_grid": [int(len(x_r)), int(len(y_r))],
    }


def _axis_field(axis_cfg: dict, key: str, default):
    val = axis_cfg.get(key)
    return default if val is None else val


def run_axis(
    suite: dict,
    axis_cfg: dict,
    run_root: Path,
    save_root: Path,
    suite_yaml_path: Path,
    seed: int | None,
    repeats: int,
    run_convergence: bool,
    base_save_stride: int,
    rng_seed: int,
    verbose: bool = True,
) -> Path:
    """Generate, evaluate, and (optionally) convergence-check one OOD axis.

    Returns the path to the written per-pair ``test_records`` CSV (under the
    selected seed dir of ``run_root``).
    """
    from data.generate_ood_dataset import generate_ood_dataset
    from src.operators.eval import write_test_records

    benchmark = suite["benchmark"]
    representation = suite.get("representation", "temporal_encoder")
    axis_name = axis_cfg["name"]

    values = _axis_field(axis_cfg, "values", None)
    dataset_t_final = _axis_field(axis_cfg, "dataset_t_final", None)
    time_norm_horizon = _axis_field(axis_cfg, "time_norm_horizon", 0.30)
    target_times = _axis_field(axis_cfg, "target_times", None)
    protocols = _axis_field(axis_cfg, "protocols", None)

    # For the time axis the generation sweep IS the set of target times (one
    # extended sim per repeat, expanded into these buckets at eval). Suite
    # descriptors declare them once as `target_times`; route them into the
    # generator's `values` when an explicit `values` override is absent.
    gen_values = values
    if axis_name == "time" and gen_values is None:
        gen_values = target_times

    save_dir = save_root / axis_name
    if verbose:
        print(f"\n=== axis {axis_name!r} -> {save_dir} ===", flush=True)

    generate_ood_dataset(
        benchmark=benchmark,
        axis_name=axis_name,
        save_dir=save_dir,
        representation=representation,
        values=gen_values,
        repeats=repeats,
        dataset_t_final=dataset_t_final,
        time_norm_horizon=time_norm_horizon,
        save_stride=base_save_stride,
        rng_seed=rng_seed,
        suite_yaml_path=str(suite_yaml_path),
        verbose=verbose,
    )

    out_name = f"test_records_ood_{axis_name}.csv"
    records_path = write_test_records(
        str(run_root),
        seed=seed,
        out_name=out_name,
        data_dir=str(save_dir),
        eval_all_sims=True,
        time_norm_horizon=time_norm_horizon,
        target_times=target_times,
        protocols=protocols,
    )
    if verbose:
        print(f"[eval] {axis_name}: wrote {records_path}", flush=True)

    convergence = None
    refine = axis_cfg.get("refine")
    if run_convergence and refine:
        nx = int(refine.get("nx", 200))
        ny = int(refine.get("ny", 200))
        dt_factor = int(refine.get("dt_factor", 1))
        refined_dt = 0.005 / dt_factor
        refined_stride = base_save_stride * dt_factor
        refined_dir = save_root / f"{axis_name}__refined"
        if verbose:
            print(
                f"[convergence] {axis_name}: regen {nx}x{ny} dt={refined_dt} "
                f"stride={refined_stride}",
                flush=True,
            )
        generate_ood_dataset(
            benchmark=benchmark,
            axis_name=axis_name,
            save_dir=refined_dir,
            representation=representation,
            values=gen_values,
            repeats=repeats,
            dataset_t_final=dataset_t_final,
            time_norm_horizon=time_norm_horizon,
            save_stride=refined_stride,
            nx=nx,
            ny=ny,
            dt=refined_dt,
            rng_seed=rng_seed,
            suite_yaml_path=str(suite_yaml_path),
            verbose=verbose,
        )
        diff = _solver_self_difference(save_dir, refined_dir)
        convergence = {
            "refined_dir": str(refined_dir),
            "refine": {"nx": nx, "ny": ny, "dt_factor": dt_factor,
                       "refined_dt": refined_dt, "refined_save_stride": refined_stride},
            **diff,
        }
        with open(save_dir / "ood_convergence.json", "w") as f:
            json.dump(convergence, f, indent=2)
        if verbose:
            print(
                f"[convergence] {axis_name}: solver self-diff "
                f"rel_l2={diff['global_rel_l2_pct']:.4f}% "
                f"rmse={diff['global_rmse_K']:.4f}K",
                flush=True,
            )

    _write_eval_manifest(
        records_path=Path(records_path),
        base_dir=save_dir,
        eval_flags={
            "eval_all_sims": True,
            "time_norm_horizon": time_norm_horizon,
            "target_times": target_times,
            "protocols": protocols,
        },
        out_name=out_name,
        convergence=convergence,
    )
    return Path(records_path)


def _write_eval_manifest(records_path, base_dir, eval_flags, out_name, convergence):
    """Link the (reusable) dataset to THIS evaluation (checkpoint/seed/flags)."""
    seed_dir = records_path.parent
    ckpt_path = seed_dir / "fno2d_best.pt"
    model_seed = None
    if seed_dir.name.startswith("seed"):
        suffix = seed_dir.name[len("seed"):]
        if suffix.isdigit():
            model_seed = int(suffix)

    manifest = {
        "ood_manifest_sha256": _sha256(base_dir / "ood_manifest.json"),
        "dataset_dir": str(base_dir),
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": _sha256(ckpt_path) if ckpt_path.exists() else None,
        "model_seed": model_seed,
        "eval_flags": eval_flags,
        "report_name": out_name,
        "git_commit": _git_commit(),
        "convergence": convergence,
    }
    eval_manifest_path = seed_dir / f"ood_eval_manifest_{base_dir.name}.json"
    with open(eval_manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return eval_manifest_path


def aggregate_records(records: list[Path], out_dir: Path) -> None:
    from scripts.inspect_ood import (
        aggregate,
        load_records,
        plot_axis_sweep,
        plot_ood_consolidated,
        plot_time_protocols,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_records([str(p) for p in records], seed_from_path=True)
    frames = aggregate(df)
    frames["sim"].to_csv(out_dir / "ood_per_sim.csv", index=False)
    frames["value"].to_csv(out_dir / "ood_per_seed.csv", index=False)
    frames["cell"].to_csv(out_dir / "ood_summary.csv", index=False)

    n_plots = 0
    for axis in sorted(a for a in frames["cell"]["ood_axis"].unique() if a):
        if plot_axis_sweep(frames["cell"], axis, out_dir) is not None:
            n_plots += 1
    if plot_time_protocols(frames["cell"], out_dir) is not None:
        n_plots += 1
    n_plots += len(plot_ood_consolidated(frames["cell"], out_dir))
    print(f"Aggregated {len(df)} pair rows -> {out_dir} ({n_plots} plot(s)).")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run an OOD suite (generate -> eval -> aggregate) against a "
                    "trained run.",
    )
    parser.add_argument("suite_yaml", type=Path,
                        help="Path to a conf/ood/*.yaml suite descriptor.")
    parser.add_argument("run_root", type=Path,
                        help="Trained run dir containing seed*/fno2d_best.pt.")
    parser.add_argument("--save-root", type=Path, default=None,
                        help="Where to write OOD datasets (default: "
                             "data/ood/<benchmark>).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Checkpoint seed to evaluate (default: lowest best_val).")
    parser.add_argument("--repeats", type=int, default=None,
                        help="Override the suite's repeats.")
    parser.add_argument("--rng-seed", type=int, default=0,
                        help="CRN base seed for the paired draws.")
    parser.add_argument("--save-stride", type=int, default=2,
                        help="Base solver save stride.")
    parser.add_argument("--axes", default=None,
                        help="Comma list to restrict to a subset of suite axes.")
    parser.add_argument("--skip-convergence", action="store_true",
                        help="Skip the opt-in refined-grid convergence pass.")
    parser.add_argument("--no-aggregate", action="store_true",
                        help="Generate + eval only; skip inspect_ood aggregation.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Aggregation output dir (default: <repo>/visual/ood/).")
    args = parser.parse_args(argv)

    if not args.suite_yaml.is_file():
        print(f"error: suite yaml not found: {args.suite_yaml}", file=sys.stderr)
        return 1
    if not args.run_root.is_dir():
        print(f"error: run_root not found: {args.run_root}", file=sys.stderr)
        return 1

    with open(args.suite_yaml) as f:
        suite = yaml.safe_load(f)

    benchmark = suite["benchmark"]
    repeats = args.repeats if args.repeats is not None else int(suite.get("repeats", 2))
    save_root = args.save_root if args.save_root is not None else (
        PROJECT_ROOT / "data" / "ood" / benchmark
    )
    save_root.mkdir(parents=True, exist_ok=True)

    axes = suite.get("axes", [])
    if args.axes:
        wanted = {a.strip() for a in args.axes.split(",")}
        axes = [a for a in axes if a["name"] in wanted]
        if not axes:
            print(f"error: none of --axes {sorted(wanted)} in suite.", file=sys.stderr)
            return 1

    print(f"Suite: {args.suite_yaml} (benchmark={benchmark}, repeats={repeats})")
    print(f"Run root: {args.run_root}")
    print(f"Save root: {save_root}")
    print(f"Axes: {[a['name'] for a in axes]}")

    records: list[Path] = []
    for axis_cfg in axes:
        rec = run_axis(
            suite=suite,
            axis_cfg=axis_cfg,
            run_root=args.run_root,
            save_root=save_root,
            suite_yaml_path=args.suite_yaml,
            seed=args.seed,
            repeats=repeats,
            run_convergence=not args.skip_convergence,
            base_save_stride=args.save_stride,
            rng_seed=args.rng_seed,
        )
        records.append(rec)

    if records and not args.no_aggregate:
        out_dir = args.out if args.out is not None else (
            PROJECT_ROOT / "visual" / "ood"
        )
        aggregate_records(records, out_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
