"""Run the fixed reviewer inverse sweep for forcing or forcing_itr.

The scientific protocol is intentionally fixed. Reviewers select the inverse
benchmark and attach its trained checkpoint; the runner calibrates and evaluates
the same eight cases at 8, 16, and 32 interface-band sensors.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import split_sim_ids  # noqa: E402
from scripts.invert import (  # noqa: E402
    _checkpoint_fingerprint,
    build_interface_sensor_mask,
)
from visual.inverse_plots import (  # noqa: E402
    _wilson_ci,
    plot_sensor_sweep_recovery,
    plot_sensor_sweep_uq,
    spec_for,
)


SENSOR_COUNTS = (8, 16, 32)
SENSOR_X_HALFWIDTH = 0.005
NOISE_STD = 0.05
N_STARTS = 8
UQ_LEVEL = 0.95
SPLIT_SEED = 0
INIT_SEED = 0
NOISE_SEED = 0
N_CASES = 8

DEFAULT_DATA_DIRS = {
    "forcing": PROJECT_ROOT / "data" / "forcinginverse",
    "forcing_itr": PROJECT_ROOT / "data" / "forcing_itr_inverse_70",
}
REQUIRED_DATA_FILES = (
    "trajectories.npy",
    "x_grid.npy",
    "y_grid.npy",
    "t_grid.npy",
    "dt.npy",
    "sim_params.npy",
)


class CalibrationGateError(RuntimeError):
    pass


class PlotGenerationError(RuntimeError):
    pass


class SummaryGenerationError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        with path.open() as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _load_checkpoint_config(checkpoint: Path) -> dict:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload.get("conf")
    if not isinstance(config, dict):
        raise ValueError("checkpoint does not contain a dictionary under 'conf'.")
    return config


def _checkpoint_benchmark(config: dict) -> tuple[str, str]:
    benchmark_cfg = config.get("benchmark", {})
    if isinstance(benchmark_cfg, dict):
        name = str(benchmark_cfg.get("name", "forcing"))
        representation = str(
            benchmark_cfg.get("representation", "temporal_encoder")
        )
    else:
        name = str(benchmark_cfg)
        representation = str(
            config.get("representation", {}).get("name", "temporal_encoder")
        )
    return name, representation


def _validate_data_dir(data_dir: Path) -> None:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"inverse data directory not found: {data_dir}")
    missing = [name for name in REQUIRED_DATA_FILES if not (data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"inverse data directory {data_dir} is missing {missing}."
        )


def select_case_ids(data_dir: Path, config: dict) -> tuple[list[int], list[int]]:
    trajectories = np.load(data_dir / "trajectories.npy", mmap_mode="r")
    num_sims = int(trajectories.shape[0])
    data_cfg = config.get("data", {})
    train_ids, val_ids, test_ids = split_sim_ids(
        num_sims,
        train_frac=float(data_cfg.get("train_split", 0.7)),
        val_frac=float(data_cfg.get("val_split", 0.15)),
        seed=SPLIT_SEED,
    )
    calibration_ids = sorted(int(sid) for sid in val_ids)
    if not calibration_ids:
        raise ValueError("inverse dataset has an empty validation calibration split.")
    candidate_ids = sorted(int(sid) for sid in np.concatenate([train_ids, test_ids]))
    if len(candidate_ids) < N_CASES:
        raise ValueError(
            f"inverse dataset has only {len(candidate_ids)} non-calibration cases; "
            f"{N_CASES} are required."
        )
    return candidate_ids[:N_CASES], calibration_ids


def validate_sensor_geometry(data_dir: Path) -> None:
    x_grid = np.load(data_dir / "x_grid.npy")
    y_grid = np.load(data_dir / "y_grid.npy")
    for count in SENSOR_COUNTS:
        requested_y = np.linspace(float(y_grid[0]), float(y_grid[-1]), count)
        mask = build_interface_sensor_mask(
            x_grid,
            y_grid,
            interface_x=0.5,
            x_halfwidth=SENSOR_X_HALFWIDTH,
            n_y=count,
            requested_y=requested_y,
        )
        realized = int(mask.sum())
        if realized != count:
            raise ValueError(
                f"sensor-x-halfwidth={SENSOR_X_HALFWIDTH:g} realizes {realized} "
                f"pixels for requested count {count}; this reviewer protocol "
                "requires exactly one x location per y sensor."
            )


def build_calibration_command(
    *,
    checkpoint: Path,
    data_dir: Path,
    device: str,
    sensor_count: int,
    calibration_path: Path,
) -> list[str]:
    return [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "invert.py"),
        "--checkpoint",
        str(checkpoint),
        "--data-dir",
        str(data_dir),
        "--sensor-layout",
        "interface_band",
        "--sensor-x-halfwidth",
        str(SENSOR_X_HALFWIDTH),
        "--sensor-n-y",
        str(sensor_count),
        "--noise-std",
        str(NOISE_STD),
        "--noise-seed",
        str(NOISE_SEED),
        "--seed",
        str(INIT_SEED),
        "--device",
        device,
        "--calibration-out",
        str(calibration_path),
    ]


def build_inversion_command(
    *,
    checkpoint: Path,
    data_dir: Path,
    device: str,
    sensor_count: int,
    calibration_path: Path,
    result_path: Path,
    artifact_dir: Path,
    sim_ids: list[int],
) -> list[str]:
    return [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "invert.py"),
        "--checkpoint",
        str(checkpoint),
        "--data-dir",
        str(data_dir),
        "--sim-ids",
        *(str(sid) for sid in sim_ids),
        "--sensor-layout",
        "interface_band",
        "--sensor-x-halfwidth",
        str(SENSOR_X_HALFWIDTH),
        "--sensor-n-y",
        str(sensor_count),
        "--noise-std",
        str(NOISE_STD),
        "--noise-seed",
        str(NOISE_SEED),
        "--n-starts",
        str(N_STARTS),
        "--seed",
        str(INIT_SEED),
        "--device",
        device,
        "--uq-level",
        str(UQ_LEVEL),
        "--calibration-artifact",
        str(calibration_path),
        "--out-csv",
        str(result_path),
        "--artifact-dir",
        str(artifact_dir),
    ]


def _run_command(command: list[str]) -> None:
    print(f"\n$ {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _calibration_summary(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as stored:
        return {
            "schema_version": int(stored["calibration_schema_version"]),
            "benchmark": str(stored["benchmark"]),
            "split": str(stored["split"]),
            "checkpoint_fingerprint": str(stored["checkpoint_fingerprint"]),
            "dataset_fingerprint": str(stored["dataset_fingerprint"]),
            "observation_fingerprint": str(stored["observation_fingerprint"]),
            "local_gate_pass": bool(stored["local_gate_pass"]),
            "simulation_count": int(stored["simulation_count"]),
            "sigma_fno_norm": float(stored["sigma_fno_norm"]),
            "sigma_fno_K": float(stored["sigma_fno_K"]),
            "sensor_rms_median_K": float(stored["sensor_rms_median_K"]),
            "sensor_rms_p90_K": float(stored["sensor_rms_p90_K"]),
            "sigma_meas_K": float(stored["sigma_meas_K"]),
        }


def _validate_calibration(
    calibration: dict,
    *,
    benchmark: str,
    checkpoint_fingerprint: str,
    calibration_count: int,
) -> None:
    if calibration["benchmark"] != benchmark:
        raise ValueError(
            f"calibration benchmark {calibration['benchmark']!r} does not match "
            f"{benchmark!r}."
        )
    if calibration["split"] != "val":
        raise ValueError("surrogate calibration was not computed on the validation split.")
    if calibration["checkpoint_fingerprint"] != checkpoint_fingerprint:
        raise ValueError("surrogate calibration checkpoint fingerprint does not match.")
    if calibration["simulation_count"] != calibration_count:
        raise ValueError(
            f"surrogate calibration used {calibration['simulation_count']} simulations; "
            f"expected all {calibration_count} validation simulations."
        )
    if not calibration["local_gate_pass"]:
        raise CalibrationGateError(
            "frozen surrogate calibration gate failed: "
            f"median={calibration['sensor_rms_median_K']:.6g} K, "
            f"p90={calibration['sensor_rms_p90_K']:.6g} K, "
            f"measurement sigma={calibration['sigma_meas_K']:.6g} K."
        )


def _resume_provenance_matches(previous: dict | None, current: dict) -> bool:
    if previous is None:
        return False
    keys = (
        "benchmark",
        "representation",
        "checkpoint",
        "checkpoint_fingerprint",
        "data_dir",
        "out_dir",
        "split_seed",
        "calibration_sim_ids",
        "evaluation_sim_ids",
        "fixed_protocol",
    )
    return all(previous.get(key) == current.get(key) for key in keys)


def _required_stage_columns(benchmark: str) -> tuple[str, ...]:
    """Columns the three reported statistics must have populated per sim."""
    shared = ("fv_resid_rms_K", "fv_resid_over_noise")
    if benchmark == "forcing":
        return shared + ("profile_R_c_ci_low", "profile_R_c_ci_high")
    return shared + ("profile_excess_ci_low", "profile_excess_ci_high")


def validate_and_annotate_rows(
    result_path: Path,
    *,
    benchmark: str,
    sensor_count: int,
    sim_ids: list[int],
) -> list[dict[str, str]]:
    with result_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if len(rows) != N_CASES:
        raise ValueError(
            f"{result_path} has {len(rows)} rows; expected {N_CASES}."
        )
    actual_ids = sorted(int(row["sim_id"]) for row in rows)
    if actual_ids != sorted(sim_ids):
        raise ValueError(
            f"{result_path} sim IDs {actual_ids} do not match {sorted(sim_ids)}."
        )

    required_columns = _required_stage_columns(benchmark)
    for row in rows:
        row_benchmark = row.get("benchmark", "")
        if row_benchmark and row_benchmark != benchmark:
            raise ValueError(
                f"{result_path} contains benchmark {row_benchmark!r}, expected "
                f"{benchmark!r}."
            )
        row["benchmark"] = benchmark
        row["init_seed"] = str(INIT_SEED)
        realized = int(float(row["n_sensors"]))
        if realized != sensor_count:
            raise ValueError(
                f"{result_path} reports {realized} sensors; expected {sensor_count}."
            )
        sid = int(row["sim_id"])
        if int(row["noise_seed"]) != NOISE_SEED + sid:
            raise ValueError(f"{result_path} has an unpaired noise seed for sim {sid}.")
        for column in required_columns:
            value = row.get(column, "")
            if value is None or not value.strip():
                raise ValueError(f"{result_path} is missing populated column {column!r}.")
            if not math.isfinite(float(value)):
                raise ValueError(f"{result_path} has non-finite {column}={value!r}.")

    _write_csv(result_path, rows)
    return rows


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    leading = ["benchmark", "sim_id", "n_sensors", "noise_seed", "init_seed"]
    all_columns = {key for row in rows for key in row}
    fieldnames = [name for name in leading if name in all_columns]
    fieldnames.extend(sorted(all_columns - set(fieldnames)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# One block per reported statistic. Spread is median/Q1/Q3/IQR only: at
# N_CASES = 8 a ddof=1 SD carries ~27% relative uncertainty and min/max are
# single order statistics, so the earlier mean/SD/min/max block reported
# precision the sample cannot support.
_ROBUST_STATISTICS = ("n", "median", "q1", "q3", "iqr")

_PAPER_SUMMARY_COLUMNS = (
    "benchmark",
    "estimand",
    "unit",
    "n_sensors",
    "n_simulations",
    # Statistic 1: lead-estimand recovery error.
    "absolute_error_n",
    "absolute_error_median",
    "absolute_error_q1",
    "absolute_error_q3",
    "absolute_error_iqr",
    "recovery_rmse",
    # Statistic 2: FV-verified sensor residual (Kelvin) vs the noise floor.
    "fv_resid_K_n",
    "fv_resid_K_median",
    "fv_resid_K_q1",
    "fv_resid_K_q3",
    "fv_resid_K_iqr",
    "fv_resid_over_noise_median",
    # Statistic 3: profile-likelihood interval width on the lead estimand.
    "profile_width_n",
    "profile_width_median",
    "profile_width_q1",
    "profile_width_q3",
    "profile_width_iqr",
    "profile_bound_limited_cases",
)


def _summary_float(row: dict[str, str], column: str) -> float:
    value = row.get(column, "")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"paper summary requires numeric column {column!r}; got {value!r}."
        ) from exc
    if not math.isfinite(parsed):
        raise ValueError(
            f"paper summary requires finite column {column!r}; got {value!r}."
        )
    return parsed


def _summary_bool(row: dict[str, str], column: str) -> bool:
    value = str(row.get(column, "")).strip().lower()
    if value in {"true", "1", "1.0"}:
        return True
    if value in {"false", "0", "0.0"}:
        return False
    raise ValueError(
        f"paper summary requires boolean column {column!r}; got {value!r}."
    )


def _robust_summary(values: np.ndarray) -> dict[str, float | int]:
    """Median/quartile spread of a per-simulation quantity across one arm."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError(
            "paper summary distributions require at least two finite values."
        )
    q1, median, q3 = np.percentile(values, [25.0, 50.0, 75.0])
    return {
        "n": int(values.size),
        "median": float(median),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
    }


def _flat_paper_row(
    *,
    benchmark: str,
    estimand: str,
    unit: str,
    arm: dict,
) -> dict[str, object]:
    row: dict[str, object] = {
        "benchmark": benchmark,
        "estimand": estimand,
        "unit": unit,
        "n_sensors": arm["n_sensors"],
        "n_simulations": arm["n_simulations"],
        "recovery_rmse": arm["recovery"]["rmse"],
        "fv_resid_over_noise_median": arm["fv_verification"]["over_noise_median"],
        "profile_bound_limited_cases": arm["profile_interval"]["bound_limited_cases"],
    }
    for prefix, summary in (
        ("absolute_error", arm["recovery"]["absolute_error"]),
        ("fv_resid_K", arm["fv_verification"]["residual_K"]),
        ("profile_width", arm["profile_interval"]["width"]),
    ):
        for statistic in _ROBUST_STATISTICS:
            row[f"{prefix}_{statistic}"] = summary[statistic]
    return row


def generate_paper_summary(
    combined_path: Path,
    *,
    benchmark: str,
    out_dir: Path,
) -> dict:
    spec = spec_for(benchmark)
    with combined_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required_columns = (
        "benchmark",
        "sim_id",
        "n_sensors",
        spec.lead_hat_col,
        spec.lead_true_col,
        spec.lead_abserr_col,
        "fv_resid_rms_K",
        "fv_resid_over_noise",
        spec.lead_profile_ci_low_col,
        spec.lead_profile_ci_high_col,
        "profile_bound_limited",
    )
    missing = [
        column
        for column in required_columns
        if not rows or any(column not in row for row in rows)
    ]
    if missing:
        raise ValueError(f"paper summary is missing required columns {missing}.")

    row_by_key: dict[tuple[int, int], dict[str, str]] = {}
    for row in rows:
        if row["benchmark"] != benchmark:
            raise ValueError(
                f"paper summary found benchmark {row['benchmark']!r}, expected "
                f"{benchmark!r}."
            )
        sensor_count = int(_summary_float(row, "n_sensors"))
        sim_id = int(_summary_float(row, "sim_id"))
        key = (sensor_count, sim_id)
        if key in row_by_key:
            raise ValueError(f"paper summary contains duplicate row {key}.")
        row_by_key[key] = row

    counts = tuple(sorted({count for count, _ in row_by_key}))
    if counts != SENSOR_COUNTS:
        raise ValueError(
            f"paper summary sensor counts {counts} do not match {SENSOR_COUNTS}."
        )
    ids_by_count = {
        count: tuple(sorted(sim_id for row_count, sim_id in row_by_key if row_count == count))
        for count in counts
    }
    sim_ids = ids_by_count[counts[0]]
    if len(sim_ids) != N_CASES or any(ids != sim_ids for ids in ids_by_count.values()):
        raise ValueError(
            f"paper summary requires the same {N_CASES} paired simulations; "
            f"got {ids_by_count}."
        )

    by_sensor_count = []
    absolute_errors_by_count: dict[int, np.ndarray] = {}
    for count in counts:
        arm_rows = [row_by_key[(count, sim_id)] for sim_id in sim_ids]
        truth = np.asarray(
            [_summary_float(row, spec.lead_true_col) for row in arm_rows]
        )
        estimate = np.asarray(
            [_summary_float(row, spec.lead_hat_col) for row in arm_rows]
        )
        signed_error = estimate - truth
        absolute_error = np.asarray(
            [_summary_float(row, spec.lead_abserr_col) for row in arm_rows]
        )
        if not np.allclose(
            absolute_error, np.abs(signed_error), rtol=1e-6, atol=1e-12
        ):
            raise ValueError(
                f"paper summary absolute-error column is inconsistent at {count} sensors."
            )

        fv_resid_K = np.asarray(
            [_summary_float(row, "fv_resid_rms_K") for row in arm_rows]
        )
        fv_over_noise = np.asarray(
            [_summary_float(row, "fv_resid_over_noise") for row in arm_rows]
        )
        if np.any(fv_resid_K < 0.0):
            raise ValueError(
                f"paper summary found a negative FV residual at {count} sensors."
            )

        profile_low = np.asarray(
            [_summary_float(row, spec.lead_profile_ci_low_col) for row in arm_rows]
        )
        profile_high = np.asarray(
            [_summary_float(row, spec.lead_profile_ci_high_col) for row in arm_rows]
        )
        if np.any(profile_high < profile_low):
            raise ValueError(
                f"paper summary found negative interval width at {count} sensors."
            )
        # Censored intervals are counted, not silently averaged in: an interval
        # that stopped at the edge of the physical range never closed on a
        # Wilks crossing, so its width is a lower bound, not a measurement.
        bound_limited = sum(
            1 for row in arm_rows if _summary_bool(row, "profile_bound_limited")
        )

        absolute_errors_by_count[count] = absolute_error
        by_sensor_count.append(
            {
                "n_sensors": count,
                "n_simulations": len(sim_ids),
                "simulation_ids": list(sim_ids),
                "recovery": {
                    "absolute_error": _robust_summary(absolute_error),
                    "rmse": float(np.sqrt(np.mean(np.square(signed_error)))),
                },
                "fv_verification": {
                    "residual_K": _robust_summary(fv_resid_K),
                    "over_noise_median": float(np.median(fv_over_noise)),
                },
                "profile_interval": {
                    "width": _robust_summary(profile_high - profile_low),
                    "bound_limited_cases": int(bound_limited),
                },
            }
        )

    source_columns = list(required_columns[3:])
    summary = {
        "schema_version": 1,
        "benchmark": benchmark,
        "estimand": spec.lead_name,
        "estimand_description": spec.lead_description,
        "unit": spec.lead_unit_text,
        "confidence_level": UQ_LEVEL,
        "sensor_counts": list(counts),
        "n_paired_simulations": len(sim_ids),
        "simulation_ids": list(sim_ids),
        "source_csv": str(combined_path.resolve()),
        "source_columns": source_columns,
        "reported_statistics": [
            "recovery_error", "fv_verification", "profile_interval"
        ],
        "statistics_conventions": {
            "spread": (
                "median with Q1/Q3/IQR; no mean/SD, which n=8 cannot support"
            ),
            "quantiles": "numpy percentile with the default linear method",
            "rmse": "sqrt(mean((estimate - truth)^2)) across simulations",
            "fv_residual": (
                "physical RMS sigma_global*sqrt(2*fv_resid) at theta_hat under "
                "the real FVSolver2D, and its ratio to the measurement noise std"
            ),
            "profile_interval": (
                "Wilks interval on the lead estimand at the frozen "
                "validation-calibrated sigma_eff; bound_limited_cases counts "
                "intervals censored by the parameter range rather than closed "
                "by a likelihood crossing"
            ),
        },
        "by_sensor_count": by_sensor_count,
    }

    json_path = (out_dir / "inverse_sensor_sweep_summary.json").resolve()
    csv_path = (out_dir / "inverse_sensor_sweep_summary.csv").resolve()
    _write_json(json_path, summary)
    flat_rows = [
        _flat_paper_row(
            benchmark=benchmark,
            estimand=spec.lead_name,
            unit=spec.lead_unit_text,
            arm=arm,
        )
        for arm in by_sensor_count
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_PAPER_SUMMARY_COLUMNS))
        writer.writeheader()
        writer.writerows(flat_rows)
    return {
        "status": "complete",
        "json": str(json_path),
        "csv": str(csv_path),
        **summary,
    }


def _print_paper_summary(summary: dict) -> None:
    print("\nPaper summary (median [Q1, Q3] across simulations):", flush=True)
    for arm in summary["by_sensor_count"]:
        absolute = arm["recovery"]["absolute_error"]
        residual = arm["fv_verification"]["residual_K"]
        width = arm["profile_interval"]["width"]
        print(
            f"  {arm['n_sensors']} sensors (n={arm['n_simulations']}):",
            flush=True,
        )
        print(
            f"    recovery error: {absolute['median']:.6g} "
            f"[{absolute['q1']:.6g}, {absolute['q3']:.6g}], "
            f"RMSE={arm['recovery']['rmse']:.6g}",
            flush=True,
        )
        print(
            f"    FV residual:    {residual['median']:.6g} K "
            f"[{residual['q1']:.6g}, {residual['q3']:.6g}], "
            f"{arm['fv_verification']['over_noise_median']:.3g}x noise std",
            flush=True,
        )
        print(
            f"    profile width:  {width['median']:.6g} "
            f"[{width['q1']:.6g}, {width['q3']:.6g}], "
            f"{arm['profile_interval']['bound_limited_cases']}"
            f"/{arm['n_simulations']} bound-limited",
            flush=True,
        )


def _validate_artifacts(artifact_dir: Path, sim_ids: list[int]) -> list[str]:
    expected = [artifact_dir / f"sim_{sid:05d}.npz" for sid in sim_ids]
    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing inverse simulation artifacts: {missing}")
    return [str(path) for path in expected]


def generate_sweep_plots(
    combined_path: Path,
    *,
    benchmark: str,
    out_dir: Path,
) -> dict:
    plot_dir = out_dir / "plots"
    spec = spec_for(benchmark)
    recovery = plot_sensor_sweep_recovery(
        combined_path,
        spec,
        out_dir=plot_dir,
    )
    uncertainty = plot_sensor_sweep_uq(
        combined_path,
        spec,
        out_dir=plot_dir,
    )
    return {
        "status": "complete",
        "directory": str(plot_dir.resolve()),
        "recovery": recovery,
        "uncertainty": uncertainty,
    }


def run_sweep(
    *,
    benchmark: str,
    checkpoint: Path,
    data_dir: Path,
    out_dir: Path,
    device: str,
) -> Path:
    checkpoint = checkpoint.expanduser().resolve()
    data_dir = data_dir.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()
    config = _load_checkpoint_config(checkpoint)
    checkpoint_benchmark, representation = _checkpoint_benchmark(config)
    if checkpoint_benchmark != benchmark:
        raise ValueError(
            f"selected benchmark {benchmark!r} does not match checkpoint benchmark "
            f"{checkpoint_benchmark!r}."
        )
    _validate_data_dir(data_dir)
    sim_ids, calibration_ids = select_case_ids(data_dir, config)
    validate_sensor_geometry(data_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "sweep_manifest.json"
    combined_path = out_dir / "inverse_sensor_sweep.csv"
    previous_manifest = _load_json(manifest_path)
    checkpoint_fingerprint = _checkpoint_fingerprint(str(checkpoint))
    manifest = {
        "schema_version": 2,
        "status": "running",
        "started_utc": _utc_now(),
        "benchmark": benchmark,
        "representation": representation,
        "checkpoint": str(checkpoint),
        "checkpoint_fingerprint": checkpoint_fingerprint,
        "data_dir": str(data_dir),
        "out_dir": str(out_dir),
        "split_seed": SPLIT_SEED,
        "calibration_sim_ids": calibration_ids,
        "evaluation_sim_ids": sim_ids,
        "fixed_protocol": {
            "sensor_counts": list(SENSOR_COUNTS),
            "sensor_layout": "interface_band",
            "sensor_x_halfwidth": SENSOR_X_HALFWIDTH,
            "noise_std_norm": NOISE_STD,
            "n_starts": N_STARTS,
            "init_seed": INIT_SEED,
            "noise_seed": NOISE_SEED,
            "fv_refine": True,
            "uq_level": UQ_LEVEL,
            "reported_statistics": [
                "recovery_error", "fv_verification", "profile_interval"
            ],
        },
        "arms": {},
    }
    resume_provenance_matches = _resume_provenance_matches(
        previous_manifest, manifest
    )
    _write_json(manifest_path, manifest)

    combined_rows: list[dict[str, str]] = []
    try:
        for sensor_count in SENSOR_COUNTS:
            arm_dir = out_dir / f"sensors_{sensor_count:02d}"
            calibration_path = arm_dir / "surrogate_calibration.npz"
            result_path = arm_dir / "inverse_results.csv"
            artifact_dir = arm_dir / "artifacts"
            arm_dir.mkdir(parents=True, exist_ok=True)

            calibration_command = build_calibration_command(
                checkpoint=checkpoint,
                data_dir=data_dir,
                device=device,
                sensor_count=sensor_count,
                calibration_path=calibration_path,
            )
            inversion_command = build_inversion_command(
                checkpoint=checkpoint,
                data_dir=data_dir,
                device=device,
                sensor_count=sensor_count,
                calibration_path=calibration_path,
                result_path=result_path,
                artifact_dir=artifact_dir,
                sim_ids=sim_ids,
            )
            arm = {
                "status": "calibrating",
                "sensor_count": sensor_count,
                "calibration_path": str(calibration_path),
                "result_csv": str(result_path),
                "artifact_dir": str(artifact_dir),
                "calibration_command": calibration_command,
                "inversion_command": inversion_command,
            }
            manifest["arms"][str(sensor_count)] = arm
            _write_json(manifest_path, manifest)

            previous_arm = (
                previous_manifest.get("arms", {}).get(str(sensor_count), {})
                if previous_manifest is not None
                else {}
            )
            resume_commands_match = (
                previous_arm.get("calibration_command") == calibration_command
                and previous_arm.get("inversion_command") == inversion_command
            )
            if (
                resume_provenance_matches
                and resume_commands_match
                and result_path.is_file()
                and calibration_path.is_file()
            ):
                try:
                    rows = validate_and_annotate_rows(
                        result_path,
                        benchmark=benchmark,
                        sensor_count=sensor_count,
                        sim_ids=sim_ids,
                    )
                    calibration = _calibration_summary(calibration_path)
                    _validate_calibration(
                        calibration,
                        benchmark=benchmark,
                        checkpoint_fingerprint=checkpoint_fingerprint,
                        calibration_count=len(calibration_ids),
                    )
                    arm["calibration"] = calibration
                    arm["artifacts"] = _validate_artifacts(artifact_dir, sim_ids)
                except CalibrationGateError:
                    arm["status"] = "calibration_gate_failed"
                    manifest["status"] = "calibration_gate_failed"
                    manifest["failed_sensor_count"] = sensor_count
                    manifest["completed_utc"] = _utc_now()
                    _write_json(manifest_path, manifest)
                    raise
                except Exception as exc:
                    arm["resume_validation_error"] = (
                        f"{type(exc).__name__}: {exc}"
                    )
                else:
                    arm["resumed"] = True
                    arm["row_count"] = len(rows)
                    arm["status"] = "complete"
                    combined_rows.extend(rows)
                    _write_json(manifest_path, manifest)
                    print(
                        f"Reusing validated {sensor_count}-sensor arm from "
                        f"{result_path}",
                        flush=True,
                    )
                    continue

            _run_command(calibration_command)

            calibration = _calibration_summary(calibration_path)
            arm["calibration"] = calibration
            try:
                _validate_calibration(
                    calibration,
                    benchmark=benchmark,
                    checkpoint_fingerprint=checkpoint_fingerprint,
                    calibration_count=len(calibration_ids),
                )
            except CalibrationGateError as exc:
                arm["status"] = "calibration_gate_failed"
                manifest["status"] = "calibration_gate_failed"
                manifest["failed_sensor_count"] = sensor_count
                manifest["completed_utc"] = _utc_now()
                _write_json(manifest_path, manifest)
                raise CalibrationGateError(
                    f"sensor count {sensor_count} failed the {exc}"
                ) from exc

            arm["status"] = "inverting"
            _write_json(manifest_path, manifest)
            _run_command(inversion_command)

            rows = validate_and_annotate_rows(
                result_path,
                benchmark=benchmark,
                sensor_count=sensor_count,
                sim_ids=sim_ids,
            )
            arm["artifacts"] = _validate_artifacts(artifact_dir, sim_ids)
            arm["row_count"] = len(rows)
            arm["status"] = "complete"
            combined_rows.extend(rows)
            _write_json(manifest_path, manifest)

        paired = {
            count: sorted(
                int(row["sim_id"])
                for row in combined_rows
                if int(float(row["n_sensors"])) == count
            )
            for count in SENSOR_COUNTS
        }
        if any(ids != sorted(sim_ids) for ids in paired.values()):
            raise ValueError(f"sensor-count arms are not paired: {paired}")
        if len(combined_rows) != len(SENSOR_COUNTS) * N_CASES:
            raise ValueError(
                f"combined sweep has {len(combined_rows)} rows; expected "
                f"{len(SENSOR_COUNTS) * N_CASES}."
            )
        dataset_fingerprints = {
            arm["calibration"]["dataset_fingerprint"]
            for arm in manifest["arms"].values()
        }
        if len(dataset_fingerprints) != 1:
            raise ValueError(
                "sensor-count calibrations do not share one dataset fingerprint: "
                f"{sorted(dataset_fingerprints)}"
            )
        _write_csv(combined_path, combined_rows)
        manifest["dataset_fingerprint"] = dataset_fingerprints.pop()
        manifest["combined_csv"] = str(combined_path)
        manifest["combined_row_count"] = len(combined_rows)
        manifest["status"] = "summarizing"
        _write_json(manifest_path, manifest)
        try:
            manifest["paper_summary"] = generate_paper_summary(
                combined_path,
                benchmark=benchmark,
                out_dir=out_dir,
            )
        except Exception as exc:
            manifest["status"] = "summary_failed"
            manifest["summary_error"] = f"{type(exc).__name__}: {exc}"
            manifest["completed_utc"] = _utc_now()
            _write_json(manifest_path, manifest)
            raise SummaryGenerationError(
                f"inverse sweep completed but paper-summary generation failed: {exc}"
            ) from exc
        _print_paper_summary(manifest["paper_summary"])
        manifest["status"] = "plotting"
        _write_json(manifest_path, manifest)
        try:
            manifest["plots"] = generate_sweep_plots(
                combined_path,
                benchmark=benchmark,
                out_dir=out_dir,
            )
        except Exception as exc:
            manifest["status"] = "plotting_failed"
            manifest["plot_error"] = f"{type(exc).__name__}: {exc}"
            manifest["completed_utc"] = _utc_now()
            _write_json(manifest_path, manifest)
            raise PlotGenerationError(
                f"inverse sweep completed but manuscript plot generation failed: {exc}"
            ) from exc
        manifest["status"] = "complete"
        manifest["completed_utc"] = _utc_now()
        _write_json(manifest_path, manifest)
    except CalibrationGateError:
        raise
    except SummaryGenerationError:
        raise
    except PlotGenerationError:
        raise
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        manifest["completed_utc"] = _utc_now()
        _write_json(manifest_path, manifest)
        raise

    print(f"\nCompleted inverse sensor sweep: {combined_path}")
    print(f"Paper summary JSON: {manifest['paper_summary']['json']}")
    print(f"Paper summary CSV: {manifest['paper_summary']['csv']}")
    return combined_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark", required=True, choices=sorted(DEFAULT_DATA_DIRS)
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override the bundled benchmark-specific inverse dataset.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Override runs/inverse_sensor_sweeps/<benchmark>/<checkpoint fingerprint>.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Torch device passed to scripts/invert.py "
            "(default preference: cuda, mps, then cpu)."
        ),
    )
    return parser


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        checkpoint = args.checkpoint.expanduser().resolve()
        data_dir = (
            args.data_dir.expanduser().resolve()
            if args.data_dir is not None
            else DEFAULT_DATA_DIRS[args.benchmark].resolve()
        )
        fingerprint = _checkpoint_fingerprint(str(checkpoint))
        out_dir = (
            args.out_dir.expanduser().resolve()
            if args.out_dir is not None
            else PROJECT_ROOT
            / "runs"
            / "inverse_sensor_sweeps"
            / args.benchmark
            / fingerprint
        )
        device = args.device or default_device()
        run_sweep(
            benchmark=args.benchmark,
            checkpoint=checkpoint,
            data_dir=data_dir,
            out_dir=out_dir,
            device=device,
        )
    except CalibrationGateError as exc:
        print(f"calibration gate failure: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


# ===========================================================================
# RETIRED AGGREGATIONS - not called by run_sweep() / generate_paper_summary().
#
# The companion appendix to the retired block in scripts/invert.py. These
# summarized statistics the entry point no longer computes, so they have no
# input columns to read even if re-wired; they are kept for reference.
#
#   * _sample_distribution - the eight-number block (mean, ddof=1 SD, median,
#     Q1, Q3, IQR, min, max) that was applied to absolute error, signed error,
#     both interval widths, and MCMC acceptance: 40 of the old 58 summary
#     columns. At N_CASES = 8 the SD carries ~27% relative uncertainty and
#     min/max are single order statistics. Live code uses _robust_summary.
#   * _coverage_summary - the nine-field coverage block with a Wilson score
#     interval. Retired with coverage itself: one noise draw per simulation
#     aggregated across simulations with different truths is not coverage over
#     noise replications, and at n=8 the Wilson interval (6/8 -> [41%, 93%])
#     cannot discriminate any true rate. Validating calibration properly means
#     a separate experiment: one fixed theta, ~200 noise replications, one
#     number.
#   * Signed-error mean bias, and the paired 8->16->32 absolute-error change
#     block (three pairs x eleven fields, including a percent change on an
#     8-sample mean), were dropped inline rather than kept as functions: an
#     8-sample mean bias is not distinguishable from zero, and the sensor
#     trend is already visible in the per-arm medians and the recovery figure.
# ===========================================================================


def _sample_distribution(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError(
            "paper summary distributions require at least two finite values."
        )
    q1, median, q3 = np.percentile(values, [25.0, 50.0, 75.0])
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "sample_sd": float(np.std(values, ddof=1)),
        "median": float(median),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _coverage_summary(values: np.ndarray) -> dict[str, float | int]:
    indicators = np.asarray(values, dtype=np.float64)
    distribution = _sample_distribution(indicators)
    total = int(indicators.size)
    successes = int(np.sum(indicators))
    rate = successes / total
    low, high = _wilson_ci(successes, total)
    return {
        "successes": successes,
        "total": total,
        "rate": float(rate),
        "percent": float(100.0 * rate),
        "indicator_sample_sd": float(distribution["sample_sd"]),
        "wilson95_low": float(low),
        "wilson95_high": float(high),
        "wilson95_low_percent": float(100.0 * low),
        "wilson95_high_percent": float(100.0 * high),
    }


if __name__ == "__main__":
    raise SystemExit(main())
