import argparse
import copy
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_train_fixed import _apply_override, _parse_override_value
from src.operators.train import load_config
from src.operators.train_pino import run_screen, resume_screen


def main():
    parser = argparse.ArgumentParser(description="Physics screens and online CViT–MG evidence runs")
    parser.add_argument("--config")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stage", choices=("fixed", "online"), default="fixed")
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--phase0-dir")
    parser.add_argument("--fixed-screen-dir")
    parser.add_argument("--exact-screen-dir")
    parser.add_argument("--normalization")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    if args.resume:
        if args.overrides or args.config:
            parser.error("Resume uses the frozen config and does not accept overrides")
        result = resume_screen(args.output_dir)
        print(result)
        return result_exit_code(result)
    if args.data_dir is None or args.phase0_dir is None:
        parser.error("Fresh screens require --data-dir and --phase0-dir")
    config = copy.deepcopy(load_config(args.config))
    for raw in args.overrides:
        key, value = raw.split("=", 1)
        _apply_override(config, key, _parse_override_value(value))
    if config["benchmark"]["name"] != "diffusion_forcing_single":
        raise ValueError("Select the homogeneous benchmark with BENCHMARK=diffusion_forcing_single")
    try:
        result = run_screen(config, args.data_dir, args.output_dir, args.phase0_dir, args.stage,
                            args.fixed_screen_dir, args.normalization, args.smoke, args.exact_screen_dir)
    except BaseException as error:
        status_path = Path(args.output_dir) / "status.json"
        if status_path.parent.exists() and not isinstance(error, FileExistsError):
            status_path.write_text(json.dumps(dict(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                                                   error_type=type(error).__name__, reason=str(error)), indent=2) + "\n")
        raise
    print(result)
    return result_exit_code(result)


def result_exit_code(result):
    if "status" in result:
        return 0 if result["status"] == "complete" else 75
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
