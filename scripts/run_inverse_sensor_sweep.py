"""Run the fixed reviewer inverse sweep for supported inverse benchmarks.

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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.inverse_adapters import (
    InverseAdapter, RELATIVE_ERROR_FLOOR_FRACTION, relative_error_percent,
)
from scripts.invert import (  # noqa: E402
    CALIBRATION_FLOOR,
    INVERSION_DATASET_SEED,
    PROFILE_STEP_FRACTION, PROFILE_WALK_MAX_STEPS,
    PROFILE_ROOT_RTOL, PROFILE_ROOT_MAX_STEPS, PROFILE_OPTIMUM_ATOL,
    _checkpoint_fingerprint,
    build_interface_sensor_mask,
    prepare_inversion_dataset,
)


SENSOR_COUNTS = (8, 16, 32)
SENSOR_X_HALFWIDTH = 0.005
NOISE_STD = 0.01
N_STARTS = 8
OPTIMIZER = "nelder_mead"
NM_MAXITER = 60
NM_XATOL = 1e-4
NM_FATOL = 1e-14
UQ_LEVEL = 0.95
# Measured joint likelihood surface over both parameters, for the identifiability
# figure. Only defined where the benchmark recovers exactly two parameters; the
# per-parameter profiles carry the uncertainty reporting on their own.
JOINT_NLL_GRID = 25
INIT_SEED = 0
NOISE_SEED = 0
# Fixed reviewer protocol: invert eight held-out cases, calibrate on a disjoint
# floor of sims. Both slices come from one dataset generated disjoint from the
# training corpus (see scripts/invert.prepare_inversion_dataset).
N_CASES = 8

BENCHMARKS = ("forcing", "forcing_itr_sin", "source_itr_sin")
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


class SummaryGenerationError(RuntimeError):
    pass


@dataclass(frozen=True)
class SweepMetricSpec:
    parameter: str
    estimate_col: str
    truth_col: str
    absolute_error_col: str
    relative_error_col: str
    parameter_range: float
    unit: str = "m² K/W"


def specs_for(benchmark: str) -> tuple[SweepMetricSpec, ...]:
    adapter = InverseAdapter.from_config({"benchmark": {"name": benchmark}})
    return tuple(
        SweepMetricSpec(
            parameter=name,
            estimate_col=f"{name}_map" if name == "R_c" else f"{name}_hat",
            truth_col=f"{name}_true",
            absolute_error_col=f"{name}_abs_error" if name == "R_c" else f"{name}_abserr",
            relative_error_col=f"{name}_rel_error_pct",
            parameter_range=float(adapter.param_scales[i]),
        )
        for i, name in enumerate(adapter.param_names)
    )


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


def prepare_sweep_dataset(
    *,
    data_dir: Path | None,
    benchmark: str,
    config: dict,
    out_dir: Path,
) -> tuple[Path, list[int], list[int]]:
    """Resolve the disjoint inversion/calibration dataset for the sweep.

    With no ``data_dir`` a fresh dataset disjoint from the training corpus is
    generated once and reused across every sensor arm. With a ``data_dir``
    override the existing directory is partitioned in place. Either way the
    first ``N_CASES`` sims are the inversion slice and the next
    ``CALIBRATION_FLOOR`` are the calibration slice, matching the partition
    scripts/invert.py rebuilds from the same ``--n-invert``/``--n-calibration``
    so both share one dataset fingerprint.
    """
    total = N_CASES + CALIBRATION_FLOOR
    if data_dir is None:
        data_cfg = config.get("data", {})
        partition = prepare_inversion_dataset(
            benchmark=benchmark,
            n_invert=N_CASES,
            n_calibration=CALIBRATION_FLOOR,
            out_dir=str(out_dir / "inversion_data"),
            nx=int(data_cfg.get("nx", 100)),
            ny=int(data_cfg.get("ny", 100)),
            save_stride=int(data_cfg.get("save_stride", 2)),
            dataset_seed=INVERSION_DATASET_SEED,
        )
        resolved = Path(partition["data_dir"]).resolve()
        return resolved, list(partition["invert_ids"]), list(partition["calibration_ids"])
    resolved = data_dir.expanduser().resolve()
    _validate_data_dir(resolved)
    trajectories = np.load(resolved / "trajectories.npy", mmap_mode="r")
    num_sims = int(trajectories.shape[0])
    if num_sims < total:
        raise ValueError(
            f"inverse dataset {resolved} has {num_sims} sims but the fixed sweep "
            f"protocol needs {N_CASES} inversion + {CALIBRATION_FLOOR} calibration "
            f"= {total}."
        )
    invert_ids = list(range(N_CASES))
    calibration_ids = list(range(N_CASES, total))
    return resolved, invert_ids, calibration_ids


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
        "--n-invert",
        str(N_CASES),
        "--n-calibration",
        str(CALIBRATION_FLOOR),
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
    benchmark: str,
    checkpoint: Path,
    data_dir: Path,
    device: str,
    sensor_count: int,
    calibration_path: Path,
    result_path: Path,
    artifact_dir: Path,
    sim_ids: list[int],
) -> list[str]:
    joint = (
        ["--joint-nll-grid", str(JOINT_NLL_GRID)]
        if len(specs_for(benchmark)) == 2
        else []
    )
    return [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "invert.py"),
        "--checkpoint",
        str(checkpoint),
        "--data-dir",
        str(data_dir),
        "--n-invert",
        str(N_CASES),
        "--n-calibration",
        str(CALIBRATION_FLOOR),
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
        "--optimizer",
        OPTIMIZER,
        "--nm-maxiter",
        str(NM_MAXITER),
        "--nm-xatol",
        str(NM_XATOL),
        "--nm-fatol",
        str(NM_FATOL),
        "--seed",
        str(INIT_SEED),
        "--device",
        device,
        "--uq-level",
        str(UQ_LEVEL),
        *joint,
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
        "dataset_seed",
        "calibration_sim_ids",
        "evaluation_sim_ids",
        "fixed_protocol",
    )
    return all(previous.get(key) == current.get(key) for key in keys)


def _required_stage_columns(benchmark: str) -> tuple[str, ...]:
    columns = ["fv_resid_rms_K", "fv_resid_over_noise"]
    for spec in specs_for(benchmark):
        columns.extend((spec.estimate_col, spec.truth_col, spec.absolute_error_col))
        columns.extend(f"profile_{spec.parameter}_{suffix}" for suffix in (
            "ci_low", "ci_high", "ci_width", "bound_limited",
        ))
    return tuple(columns)


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
            if column.endswith("_bound_limited"):
                _summary_bool(row, column)
            elif not math.isfinite(float(value)):
                raise ValueError(f"{result_path} has non-finite {column}={value!r}.")
        for spec in specs_for(benchmark):
            _relative_error_value(row, spec)

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


_ROBUST_STATISTICS = ("n", "median", "q1", "q3", "iqr")
_PAPER_SUMMARY_COLUMNS = (
    "benchmark", "parameter", "unit", "n_sensors", "n_simulations",
    *(f"{prefix}_{stat}" for prefix in ("absolute_error", "relative_error_pct", "profile_width", "fv_resid_K")
      for stat in _ROBUST_STATISTICS),
    "relative_error_undefined_cases", "recovery_rmse", "fv_resid_over_noise_median",
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


def _robust_summary(values: np.ndarray) -> dict[str, float | int | None]:
    """Median/quartile spread of a per-simulation quantity across one arm."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("paper summary distributions require finite values")
    if not values.size:
        return {"n": 0, "median": None, "q1": None, "q3": None, "iqr": None}
    q1, median, q3 = np.percentile(values, [25.0, 50.0, 75.0])
    return {
        "n": int(values.size),
        "median": float(median),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
    }


def _relative_error_value(row: dict, spec: SweepMetricSpec) -> float:
    expected = relative_error_percent(
        _summary_float(row, spec.estimate_col), _summary_float(row, spec.truth_col),
        spec.parameter_range,
    )
    if spec.relative_error_col not in row:
        raise ValueError(f"missing {spec.relative_error_col}")
    raw = row[spec.relative_error_col]
    actual = float(raw) if raw not in (None, "") else float("nan")
    if not np.isclose(actual, expected, rtol=1e-6, atol=1e-12, equal_nan=True):
        raise ValueError(f"inconsistent relative-error column {spec.relative_error_col}")
    return actual


def _flat_paper_row(*, benchmark, parameter, arm):
    metrics = arm["parameters"][parameter]
    row = {
        "benchmark": benchmark, "parameter": parameter, "unit": metrics["unit"],
        "n_sensors": arm["n_sensors"], "n_simulations": arm["n_simulations"],
        "recovery_rmse": metrics["recovery"]["rmse"],
        "relative_error_undefined_cases": metrics["recovery"]["relative_error_undefined_cases"],
        "fv_resid_over_noise_median": arm["fv_verification"]["over_noise_median"],
        "profile_bound_limited_cases": metrics["profile_interval"]["bound_limited_cases"],
    }
    for prefix, stats in (
        ("absolute_error", metrics["recovery"]["absolute_error"]),
        ("relative_error_pct", metrics["recovery"]["relative_error_pct"]),
        ("profile_width", metrics["profile_interval"]["width"]),
        ("fv_resid_K", arm["fv_verification"]["residual_K"]),
    ):
        row.update({f"{prefix}_{stat}": stats[stat] for stat in _ROBUST_STATISTICS})
    return row


def generate_paper_summary(
    combined_path: Path,
    *,
    benchmark: str,
    out_dir: Path,
) -> dict:
    specs = specs_for(benchmark)
    with combined_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required_columns = (
        "benchmark", "sim_id", "n_sensors", "noise_seed", "init_seed",
        *_required_stage_columns(benchmark), *(spec.relative_error_col for spec in specs),
    )
    missing = [column for column in required_columns if not rows or any(column not in row for row in rows)]
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
    pairing_by_count = {
        count: tuple(sorted(
            (
                int(_summary_float(row_by_key[(count, sim_id)], "sim_id")),
                int(_summary_float(row_by_key[(count, sim_id)], "noise_seed")),
                int(_summary_float(row_by_key[(count, sim_id)], "init_seed")),
            )
            for sim_id in sim_ids
        ))
        for count in counts
    }
    if any(keys != pairing_by_count[counts[0]] for keys in pairing_by_count.values()):
        raise ValueError(
            "paper summary requires identical (sim_id, noise_seed, init_seed) "
            f"pairing across sensor counts; got {pairing_by_count}."
        )

    by_sensor_count = []
    for count in counts:
        arm_rows = [row_by_key[(count, sim_id)] for sim_id in sim_ids]
        fv = np.asarray([_summary_float(row, "fv_resid_rms_K") for row in arm_rows])
        ratios = np.asarray([_summary_float(row, "fv_resid_over_noise") for row in arm_rows])
        if np.any(fv < 0) or np.any(ratios < 0):
            raise ValueError("negative FV residual")
        parameters = {}
        for spec in specs:
            truth = np.asarray([_summary_float(row, spec.truth_col) for row in arm_rows])
            reference_truth = np.asarray([_summary_float(row_by_key[(counts[0], sid)], spec.truth_col) for sid in sim_ids])
            if not np.allclose(truth, reference_truth, rtol=1e-9, atol=1e-12):
                raise ValueError(f"{spec.parameter} truth changes across sensor counts")
            estimate = np.asarray([_summary_float(row, spec.estimate_col) for row in arm_rows])
            error = np.asarray([_summary_float(row, spec.absolute_error_col) for row in arm_rows])
            if not np.allclose(error, np.abs(estimate - truth), rtol=1e-6, atol=1e-12):
                raise ValueError(f"paper summary absolute-error column is inconsistent at {count} sensors")
            relative = np.asarray([_relative_error_value(row, spec) for row in arm_rows])
            stem = f"profile_{spec.parameter}"
            low = np.asarray([_summary_float(row, f"{stem}_ci_low") for row in arm_rows])
            high = np.asarray([_summary_float(row, f"{stem}_ci_high") for row in arm_rows])
            widths = np.asarray([_summary_float(row, f"{stem}_ci_width") for row in arm_rows])
            if np.any(high < low) or not np.allclose(widths, high - low, rtol=1e-6, atol=1e-12):
                raise ValueError(f"inconsistent interval width for {spec.parameter}")
            limited = np.asarray([_summary_bool(row, f"{stem}_bound_limited") for row in arm_rows])
            parameters[spec.parameter] = {
                "unit": spec.unit,
                "recovery": {
                    "absolute_error": _robust_summary(error),
                    "relative_error_pct": _robust_summary(relative[np.isfinite(relative)]),
                    "relative_error_undefined_cases": int(np.count_nonzero(~np.isfinite(relative))),
                    "rmse": float(np.sqrt(np.mean((estimate - truth) ** 2))),
                },
                "profile_interval": {
                    "width": _robust_summary(widths[~limited]),
                    "bound_limited_cases": int(limited.sum()),
                },
            }
        by_sensor_count.append({
            "n_sensors": count, "n_simulations": len(sim_ids),
            "simulation_ids": list(sim_ids), "parameters": parameters,
            "fv_verification": {"residual_K": _robust_summary(fv), "over_noise_median": float(np.median(ratios))},
        })
    summary = {
        "schema_version": 2, "benchmark": benchmark,
        "parameters": [spec.parameter for spec in specs], "confidence_level": UQ_LEVEL,
        "sensor_counts": list(counts), "n_paired_simulations": len(sim_ids),
        "simulation_ids": list(sim_ids),
        "pairing_fields": ["benchmark", "sim_id", "noise_seed", "init_seed"],
        "source_csv": str(combined_path.resolve()), "source_columns": list(required_columns),
        "reported_statistics": ["parameter_recovery", "fv_verification", "parameter_profile_intervals"],
        "statistics_conventions": {
            "spread": "median with Q1/Q3/IQR across simulations; no mean/SD",
            "quantiles": "numpy percentile with the default linear method",
            "relative_error_pct": "100 * abs(estimate - truth) / abs(truth)",
            "relative_error_floor_fraction": RELATIVE_ERROR_FLOOR_FRACTION,
            "relative_error_floor": "undefined when abs(truth) <= floor_fraction * admissible parameter range",
            "rmse": "sqrt(mean((estimate - truth)^2)) across simulations",
            "profile_interval": "95% parameter profile at frozen calibrated variance, endpoints bracketed and bisected with the nuisance re-optimized at every pinned value; width statistics use only intervals with both likelihood crossings inside the admissible range; counts report exclusions",
        },
        "by_sensor_count": by_sensor_count,
    }
    json_path = (out_dir / "inverse_sensor_sweep_summary.json").resolve()
    csv_path = (out_dir / "inverse_sensor_sweep_summary.csv").resolve()
    _write_json(json_path, summary)
    flat_rows = [_flat_paper_row(benchmark=benchmark, parameter=spec.parameter, arm=arm)
                 for arm in by_sensor_count for spec in specs]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_PAPER_SUMMARY_COLUMNS))
        writer.writeheader()
        writer.writerows(flat_rows)
    return {"status": "complete", "json": str(json_path), "csv": str(csv_path), **summary}


def _print_paper_summary(summary: dict) -> None:
    def format_stats(stats):
        if stats["n"] == 0:
            return "unavailable (n=0)"
        return f"{stats['median']:.6g} [{stats['q1']:.6g}, {stats['q3']:.6g}] (n={stats['n']})"

    print("\nParameter summary (median [Q1, Q3] across simulations):", flush=True)
    for arm in summary["by_sensor_count"]:
        print(f"  {arm['n_sensors']} sensors (n={arm['n_simulations']}):", flush=True)
        for name, metrics in arm["parameters"].items():
            recovery, profile = metrics["recovery"], metrics["profile_interval"]
            print(f"    {name}: absolute error {format_stats(recovery['absolute_error'])} {metrics['unit']}")
            print(f"      relative error [%]: {format_stats(recovery['relative_error_pct'])}; "
                  f"undefined={recovery['relative_error_undefined_cases']}")
            print(f"      95% profile width: {format_stats(profile['width'])} {metrics['unit']}; "
                  f"bound-limited={profile['bound_limited_cases']}")
        fv = arm["fv_verification"]
        print(f"    FV residual: {format_stats(fv['residual_K'])} K; {fv['over_noise_median']:.3g}x noise std", flush=True)


def _validate_artifacts(artifact_dir: Path, sim_ids: list[int]) -> list[str]:
    expected = [artifact_dir / f"sim_{sid:05d}.npz" for sid in sim_ids]
    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing inverse simulation artifacts: {missing}")
    return [str(path) for path in expected]


def run_sweep(
    *,
    benchmark: str,
    checkpoint: Path,
    data_dir: Path | None,
    out_dir: Path,
    device: str,
) -> Path:
    checkpoint = checkpoint.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()
    config = _load_checkpoint_config(checkpoint)
    checkpoint_benchmark, representation = _checkpoint_benchmark(config)
    if checkpoint_benchmark != benchmark:
        raise ValueError(
            f"selected benchmark {benchmark!r} does not match checkpoint benchmark "
            f"{checkpoint_benchmark!r}."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir, sim_ids, calibration_ids = prepare_sweep_dataset(
        data_dir=data_dir,
        benchmark=benchmark,
        config=config,
        out_dir=out_dir,
    )
    validate_sensor_geometry(data_dir)

    manifest_path = out_dir / "sweep_manifest.json"
    combined_path = out_dir / "inverse_sensor_sweep.csv"
    previous_manifest = _load_json(manifest_path)
    checkpoint_fingerprint = _checkpoint_fingerprint(str(checkpoint))
    manifest = {
        "schema_version": 4,
        "status": "running",
        "started_utc": _utc_now(),
        "benchmark": benchmark,
        "representation": representation,
        "checkpoint": str(checkpoint),
        "checkpoint_fingerprint": checkpoint_fingerprint,
        "data_dir": str(data_dir),
        "out_dir": str(out_dir),
        "dataset_seed": INVERSION_DATASET_SEED,
        "calibration_sim_ids": calibration_ids,
        "evaluation_sim_ids": sim_ids,
        "fixed_protocol": {
            "sensor_counts": list(SENSOR_COUNTS),
            "sensor_layout": "interface_band",
            "sensor_x_halfwidth": SENSOR_X_HALFWIDTH,
            "noise_std_norm": NOISE_STD,
            "n_starts": N_STARTS,
            "optimizer": OPTIMIZER,
            "nm_maxiter": NM_MAXITER,
            "nm_xatol": NM_XATOL,
            "nm_fatol": NM_FATOL,
            "init_seed": INIT_SEED,
            "noise_seed": NOISE_SEED,
            "fv_refine": True,
            "uq_level": UQ_LEVEL,
            "parameter_reporting_version": 2,
            "profile_parameters": [spec.parameter for spec in specs_for(benchmark)],
            "profile_step_fraction": PROFILE_STEP_FRACTION,
            "profile_walk_max_steps": PROFILE_WALK_MAX_STEPS,
            "joint_nll_grid": (
                JOINT_NLL_GRID if len(specs_for(benchmark)) == 2 else 0
            ),
            "profile_root_rtol": PROFILE_ROOT_RTOL,
            "profile_root_max_steps": PROFILE_ROOT_MAX_STEPS,
            "profile_optimum_atol": PROFILE_OPTIMUM_ATOL,
            "relative_error_floor_fraction": RELATIVE_ERROR_FLOOR_FRACTION,
            "reported_statistics": [
                "parameter_recovery", "fv_verification", "parameter_profile_intervals"
            ],
            "pairing": (
                "identical (benchmark, sim_id, noise_seed, init_seed) cases "
                "across n_sensors = 8, 16, 32"
            ),
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
                benchmark=benchmark,
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
        manifest["canonical_csv"] = str(combined_path)
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
        manifest["status"] = "complete"
        manifest["completed_utc"] = _utc_now()
        _write_json(manifest_path, manifest)
    except CalibrationGateError:
        raise
    except SummaryGenerationError:
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
        "--benchmark", required=True, choices=sorted(BENCHMARKS)
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=(
            "Optional existing disjoint inverse dataset. If omitted, one is "
            "generated (disjoint from training) into <out-dir>/inversion_data."
        ),
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
            else None
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


if __name__ == "__main__":
    raise SystemExit(main())
