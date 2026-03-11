import argparse
import csv
import json
import re
import shutil
from pathlib import Path


def _latest_experiment_dir(root: Path) -> Path:
    pattern = re.compile(r"^experiment(\d+)$")
    candidates: list[tuple[int, Path]] = []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        match = pattern.fullmatch(entry.name)
        if match:
            candidates.append((int(match.group(1)), entry))

    if not candidates:
        raise FileNotFoundError(f"No experiment directories found in: {root}")

    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def _collect_rows(experiment_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for json_file in sorted(experiment_dir.glob("conf*.json")):
        with json_file.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        rows.append(
            {
                "config_name": payload["config_name"],
                "objective_mean_best_val": f"{float(payload['objective_mean_best_val']):.12f}",
                "num_seeds": str(int(payload["num_seeds"])),
            }
        )
    def _key(row: dict) -> int:
        match = re.fullmatch(r"conf(\d+)", row["config_name"])
        return int(match.group(1)) if match else 0

    rows.sort(key=_key)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild index.csv and best_config.yaml for generated configs.")
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Experiment name (e.g. experiment3). Defaults to latest experiment directory.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    generated_root = project_root / "conf" / "generated"
    if not generated_root.exists():
        raise FileNotFoundError(f"Generated conf root not found: {generated_root}")

    experiment_dir = generated_root / args.experiment if args.experiment else _latest_experiment_dir(generated_root)
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")

    rows = _collect_rows(experiment_dir)
    if not rows:
        raise FileNotFoundError(f"No conf*.json files found in: {experiment_dir}")

    index_path = experiment_dir / "index.csv"
    with index_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["config_name", "objective_mean_best_val", "num_seeds"])
        writer.writeheader()
        writer.writerows(rows)

    best_row = min(rows, key=lambda row: float(row["objective_mean_best_val"]))
    best_src = experiment_dir / f"{best_row['config_name']}.yaml"
    if not best_src.exists():
        raise FileNotFoundError(f"Best conf YAML not found: {best_src}")
    shutil.copy2(best_src, experiment_dir / "best_config.yaml")

    print(f"Rebuilt index + best conf for {experiment_dir.name}")
    print(f"best_config.yaml -> {experiment_dir / 'best_config.yaml'}")


if __name__ == "__main__":
    main()
