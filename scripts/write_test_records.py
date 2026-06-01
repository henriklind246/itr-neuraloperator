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
        "--out-name",
        default="test_records.csv",
        help="Output CSV filename, written under the selected seed directory.",
    )
    args = parser.parse_args()

    run_root = Path(args.run_root).expanduser()
    if not run_root.is_dir():
        print(f"error: run_root does not exist or is not a directory: {run_root}", file=sys.stderr)
        return 1

    from src.operators.eval import write_test_records

    write_test_records(str(run_root), seed=args.seed, out_name=args.out_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
