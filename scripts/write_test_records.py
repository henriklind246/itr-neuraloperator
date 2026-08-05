import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write per-sample test-set records (CSV) for paper figures.",
    )
    parser.add_argument(
        "run_root",
        help="Path to a config directory containing seed*/ subdirs (e.g. $HOME/fno_runs/exp/config0).",
    )
    parser.add_argument(
        "--seed",
        default=None,
        help="Seed to evaluate (e.g. 42 -> seed42/). Defaults to the lowest-best_val seed.",
    )
    parser.add_argument(
        "--all-seeds",
        action="store_true",
        help=(
            "Evaluate every declared seed*/fno2d_best.pt under run_root. "
            "Use this for publication cohorts; it never selects a seed by test error."
        ),
    )
    parser.add_argument(
        "--out-name",
        default="test_records.csv",
        help="Output CSV filename, written under the selected seed directory.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help=(
            "Optional dataset directory to evaluate against instead of the paths "
            "baked into the checkpoint. Points the checkpoint at this dataset's "
            "test split (e.g. a freshly generated benchmark set); the "
            "checkpoint's training-distribution normalization is reused."
        ),
    )
    parser.add_argument(
        "--rollout-num-substeps",
        type=int,
        default=None,
        help=(
            "Opt into homogeneous autoregressive rollout with this many substeps. "
            "Omit to use checkpoint/config defaults; values greater than 1 enable rollout."
        ),
    )
    parser.add_argument(
        "--eval-all-sims",
        action="store_true",
        help=(
            "Record every simulation in the (redirected) dataset instead of the "
            "15%% held-out test slice. Use for OOD datasets so all sims get rows."
        ),
    )
    parser.add_argument(
        "--time-norm-horizon",
        type=float,
        default=None,
        help=(
            "Trained normalization horizon (e.g. 0.30) for temporal model-input "
            "features ONLY; the physical horizon stays the dataset's t_final."
        ),
    )
    parser.add_argument(
        "--target-times",
        default=None,
        help=(
            "Comma-separated target times for the OOD time pair protocols "
            "(e.g. 0.25,0.30,0.35,0.40,0.45)."
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Override the checkpoint's training device (cpu, mps, cuda, auto). "
            "'auto' resolves to CPU without CUDA, so re-scoring a checkpoint on "
            "a workstation needs this to reach an accelerator."
        ),
    )
    parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=1,
        help=(
            "Number of direct pairs forwarded together while preserving one "
            "output row per pair. Keep at 1 for autoregressive rollout."
        ),
    )
    parser.add_argument(
        "--n-snapshots-test",
        type=int,
        default=None,
        help=(
            "Number of snapshots subsampled per simulation for the test split. "
            "Pairs per simulation are S*(S-1)/2, so this governs runtime: "
            "6 -> 15 pairs/sim, the config default 40 -> 780."
        ),
    )
    parser.add_argument(
        "--protocol",
        action="append",
        default=None,
        choices=["fixed_initial", "anchored_from_horizon", "ood_local_fixed_lead"],
        help="OOD time pair protocol(s); repeatable.",
    )
    args = parser.parse_args()

    if args.all_seeds and args.seed is not None:
        parser.error("--all-seeds and --seed are mutually exclusive")

    run_root = Path(args.run_root).expanduser()
    if not run_root.is_dir():
        print(f"error: run_root does not exist or is not a directory: {run_root}", file=sys.stderr)
        return 1

    from src.operators.eval import write_test_records

    rollout_enabled = None
    if args.rollout_num_substeps is not None:
        rollout_enabled = args.rollout_num_substeps > 1

    target_times = None
    if args.target_times is not None:
        target_times = [float(t) for t in args.target_times.split(",") if t.strip()]

    seeds = [args.seed]
    if args.all_seeds:
        seeds = [
            path.name.removeprefix("seed")
            for path in sorted(run_root.glob("seed*"))
            if (path / "fno2d_best.pt").exists()
        ]
        if not seeds:
            print(f"error: no seed*/fno2d_best.pt found under {run_root}", file=sys.stderr)
            return 1
    for seed in seeds:
        write_test_records(
            str(run_root),
            seed=seed,
            out_name=args.out_name,
            data_dir=args.data_dir,
            rollout_enabled=rollout_enabled,
            rollout_num_substeps=args.rollout_num_substeps,
            eval_all_sims=args.eval_all_sims,
            time_norm_horizon=args.time_norm_horizon,
            target_times=target_times,
            protocols=args.protocol,
            device=args.device,
            inference_batch_size=args.inference_batch_size,
            n_snapshots_test=args.n_snapshots_test,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
