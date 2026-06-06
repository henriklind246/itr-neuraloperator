import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate trained FNO2D checkpoints on the test split.",
    )
    parser.add_argument(
        "run_root",
        help="Path to a config directory containing seed*/ subdirs (e.g. $HOME/fno_runs/exp/config0).",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help=(
            "Optional dataset directory to evaluate against instead of the paths "
            "baked into the checkpoint. Used for cross-resolution eval: point a "
            "100x100-trained checkpoint at a finer-grid dataset (only the test "
            "split is consumed; training-distribution normalization is reused)."
        ),
    )
    parser.add_argument(
        "--report-name",
        default="seed_report.json",
        help="Filename for the saved report under run_root (e.g. seed_report_r200.json).",
    )
    args = parser.parse_args()

    run_root = Path(args.run_root).expanduser()
    if not run_root.is_dir():
        print(f"error: run_root does not exist or is not a directory: {run_root}", file=sys.stderr)
        return 1

    seed_dirs = sorted(run_root.glob("seed*"))
    if not seed_dirs:
        print(f"error: no seed*/ subdirectories found under {run_root}", file=sys.stderr)
        return 1

    print(f"Evaluating run_root: {run_root}")
    print(f"Found {len(seed_dirs)} seed dir(s): {[d.name for d in seed_dirs]}")

    from src.operators.eval import eval_all_seeds, print_seed_report, save_report

    results = eval_all_seeds(str(run_root), data_dir=args.data_dir, report_name=args.report_name)
    if not results:
        print(f"error: no fno2d_best.pt checkpoints found under {run_root}", file=sys.stderr)
        return 1

    for r in results:
        print(
            f"seed={r['seed']} best_epoch={r['best_epoch']} "
            f"best_val={r['best_val']:.6f} "
            f"test_rel_l2_norm={r['test_rel_l2_norm']:.6f} "
            f"test_rel_l2_phys={r['test_rel_l2']:.6f} "
            f"test_iface_rel_l2_norm={r['test_iface_rel_l2_norm']:.6f} "
            f"test_iface_rel_l2_phys={r['test_iface_rel_l2']:.6f} "
            f"test_boundary_rel_l2_norm={r['test_boundary_rel_l2_norm']:.6f} "
            f"test_boundary_rel_l2_phys={r['test_boundary_rel_l2']:.6f} "
            f"num_sims={r['num_sims']}"
        )

    summary = print_seed_report(results)
    save_report(run_root=str(run_root), results=results, summary=summary, report_name=args.report_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
