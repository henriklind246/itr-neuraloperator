"""Compare inverse runtime for a forcing_itr_sin FNO and the FV solver.

The diagnostic generates three held-out finite-volume simulations and solves
the same noisy inverse problem with three pathways: FNO + Adam/L-BFGS,
FNO + Nelder-Mead, and FV + Nelder-Mead.  It is intentionally a small paired
runtime diagnostic rather than a calibration, UQ, or profiling workflow.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from scipy.optimize import minimize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.generate_dataset import generate_sim_data  # noqa: E402
from scripts.invert import (  # noqa: E402
    INVERSION_DATASET_SEED,
    PAPER_OBSERVATION_TIMES,
    _checkpoint_fingerprint,
    _data_loss,
    _masked_half_mse,
    add_measurement_noise,
    apply_sensor_mask,
    build_dataset_from_dir,
    build_observation_set,
    fv_predict_masked,
    load_checkpoint,
    resolve_observation_times,
)
from scripts.inverse_adapters import ForcingItrSinAdapter  # noqa: E402


BENCHMARK = "forcing_itr_sin"
REPRESENTATION = "temporal_encoder"
METHODS = (
    "fno_adam_lbfgs",
    "fno_nelder_mead",
    "fv_nelder_mead",
)

N_CASES = 3
N_STARTS = 3
INIT_SEED = 0
NOISE_SEED = 0
DEFAULT_NOISE_STD = 0.01

SENSOR_COUNT = 8
SENSOR_LAYOUT = "interface_band"
SENSOR_X_HALFWIDTH = 0.005
INTERFACE_X = 0.5

ADAM_STEPS = 400
ADAM_LR = 0.05
LBFGS_STEPS = 50
NM_MAXITER = 60
NM_XATOL = 1e-4
NM_FATOL = 1e-14

CASE_COLUMNS = (
    "method",
    "forward_model",
    "optimizer",
    "sim_id",
    "runtime_s",
    "objective_half_mse",
    "nfev",
    "nit",
    "converged",
    "termination_reason",
    "R_base_hat",
    "R_base_true",
    "R_base_abs_error",
    "S_R_hat",
    "S_R_true",
    "S_R_abs_error",
)

SUMMARY_COLUMNS = (
    "method",
    "median_runtime_s",
    "speedup_vs_fv_nelder_mead",
    "median_S_R_abs_error",
    "median_R_base_abs_error",
)


@dataclass
class MethodOutcome:
    theta_hat: torch.Tensor
    loss: float
    nfev: int
    nit: int
    converged: bool
    termination_reason: str


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def validate_checkpoint_config(config: dict) -> None:
    benchmark_cfg = config.get("benchmark", {})
    if not isinstance(benchmark_cfg, dict):
        raise ValueError(
            f"runtime comparison requires benchmark {BENCHMARK!r}; "
            f"checkpoint stores {benchmark_cfg!r}."
        )
    name = str(benchmark_cfg.get("name", ""))
    representation = str(
        benchmark_cfg.get(
            "representation",
            config.get("representation", {}).get("name", REPRESENTATION),
        )
    )
    if name != BENCHMARK or representation != REPRESENTATION:
        raise ValueError(
            f"runtime comparison requires {BENCHMARK}/{REPRESENTATION}; "
            f"checkpoint stores {name or '<missing>'}/{representation}."
        )


def generate_ground_truth_dataset(
    config: dict,
    data_dir: Path,
    *,
    generate_fn: Callable = generate_sim_data,
) -> None:
    data_cfg = config.get("data", {})
    generate_fn(
        num_sims=N_CASES,
        save_dir=data_dir,
        benchmark=BENCHMARK,
        nx=int(data_cfg.get("nx", 100)),
        ny=int(data_cfg.get("ny", 100)),
        save_stride=int(data_cfg.get("save_stride", 2)),
        rng_seed=INVERSION_DATASET_SEED,
    )


def lhs_starts(adapter: ForcingItrSinAdapter) -> np.ndarray:
    return adapter.lhs_starts_unconstrained(
        N_STARTS, np.random.default_rng(INIT_SEED)
    )


def _finite_objective(value: float) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"inverse objective is non-finite: {parsed}.")
    return parsed


def run_fno_adam_lbfgs(
    model,
    obs,
    adapter: ForcingItrSinAdapter,
    starts_u: np.ndarray,
    *,
    device: str,
) -> MethodOutcome:
    best: Optional[MethodOutcome] = None
    total_nfev = 0
    total_nit = 0

    for start_u in starts_u:
        u = torch.from_numpy(np.asarray(start_u, dtype=np.float32)).to(device)
        u = u.clone().requires_grad_(True)
        start_nfev = 0

        adam = torch.optim.Adam([u], lr=ADAM_LR)
        for _ in range(ADAM_STEPS):
            adam.zero_grad()
            loss = _data_loss(model, obs, u, 0.0, adapter)
            start_nfev += 1
            loss.backward()
            adam.step()

        lbfgs = torch.optim.LBFGS(
            [u], max_iter=LBFGS_STEPS, line_search_fn="strong_wolfe"
        )

        def closure():
            nonlocal start_nfev
            lbfgs.zero_grad()
            loss = _data_loss(model, obs, u, 0.0, adapter)
            start_nfev += 1
            loss.backward()
            return loss

        lbfgs.step(closure)
        state = lbfgs.state[u]
        lbfgs_nit = int(state.get("n_iter", 0))
        lbfgs_nfev = int(state.get("func_evals", 0))
        max_eval = int(lbfgs.param_groups[0]["max_eval"])

        with torch.no_grad():
            final_loss = _finite_objective(
                _data_loss(model, obs, u, 0.0, adapter).detach().cpu().item()
            )
            start_nfev += 1
            theta = adapter.theta_from_unconstrained(u).detach().cpu()

        if lbfgs_nit >= LBFGS_STEPS:
            converged = False
            reason = "derived: torch LBFGS reached iteration cap"
        elif lbfgs_nfev >= max_eval:
            converged = False
            reason = "derived: torch LBFGS reached evaluation cap"
        else:
            converged = True
            reason = "derived: torch LBFGS stopped before iteration/evaluation caps"

        total_nfev += start_nfev
        total_nit += ADAM_STEPS + lbfgs_nit
        outcome = MethodOutcome(
            theta_hat=theta,
            loss=final_loss,
            nfev=0,
            nit=0,
            converged=converged,
            termination_reason=reason,
        )
        if best is None or outcome.loss < best.loss:
            best = outcome

    assert best is not None
    best.nfev = total_nfev
    best.nit = total_nit
    return best


def run_multistart_nelder_mead(
    adapter: ForcingItrSinAdapter,
    starts_u: np.ndarray,
    objective: Callable[[np.ndarray], float],
) -> MethodOutcome:
    best_result = None
    total_nfev = 0
    total_nit = 0

    def checked_objective(u: np.ndarray) -> float:
        return _finite_objective(objective(np.asarray(u, dtype=np.float64)))

    for start_u in starts_u:
        result = minimize(
            checked_objective,
            np.asarray(start_u, dtype=np.float64),
            method="Nelder-Mead",
            options={
                "maxiter": NM_MAXITER,
                "xatol": NM_XATOL,
                "fatol": NM_FATOL,
            },
        )
        _finite_objective(result.fun)
        total_nfev += int(result.nfev)
        total_nit += int(result.nit)
        if best_result is None or float(result.fun) < float(best_result.fun):
            best_result = result

    assert best_result is not None
    theta = adapter.theta_from_unconstrained(
        torch.from_numpy(np.asarray(best_result.x, dtype=np.float32))
    ).detach().cpu()
    return MethodOutcome(
        theta_hat=theta,
        loss=float(best_result.fun),
        nfev=total_nfev,
        nit=total_nit,
        converged=bool(best_result.success),
        termination_reason=str(best_result.message),
    )


def run_fno_nelder_mead(
    model,
    obs,
    adapter: ForcingItrSinAdapter,
    starts_u: np.ndarray,
    *,
    device: str,
) -> MethodOutcome:
    def objective(u_np: np.ndarray) -> float:
        u = torch.from_numpy(u_np.astype(np.float32)).to(device)
        with torch.no_grad():
            return float(_data_loss(model, obs, u, 0.0, adapter).detach().cpu())

    return run_multistart_nelder_mead(adapter, starts_u, objective)


def run_fv_nelder_mead(
    ds,
    base_kwargs: dict,
    obs,
    adapter: ForcingItrSinAdapter,
    starts_u: np.ndarray,
    *,
    mu_global: float,
    sigma_global: float,
) -> MethodOutcome:
    mask_cpu = None if obs.mask is None else obs.mask.detach().cpu()
    target_obs = apply_sensor_mask(obs.targets.detach().cpu(), mask_cpu)

    def objective(u_np: np.ndarray) -> float:
        theta = adapter.theta_from_unconstrained(
            torch.from_numpy(u_np.astype(np.float32))
        )
        pred_obs = fv_predict_masked(
            ds,
            base_kwargs,
            obs.sid,
            theta,
            obs.time_indices,
            mu_global=mu_global,
            sigma_global=sigma_global,
            mask=mask_cpu,
            adapter=adapter,
            obs=obs,
        )
        return _masked_half_mse(pred_obs, target_obs)

    return run_multistart_nelder_mead(adapter, starts_u, objective)


def _synchronize(device: str) -> None:
    device_type = torch.device(device).type
    if device_type == "cuda":
        torch.cuda.synchronize(torch.device(device))
    elif device_type == "mps":
        torch.mps.synchronize()


def _timed(
    call: Callable[[], MethodOutcome], *, device: Optional[str]
) -> tuple[MethodOutcome, float]:
    if device is not None:
        _synchronize(device)
    started = time.perf_counter()
    outcome = call()
    if device is not None:
        _synchronize(device)
    return outcome, time.perf_counter() - started


def warm_fno(
    model,
    obs,
    adapter: ForcingItrSinAdapter,
    start_u: np.ndarray,
    *,
    device: str,
) -> None:
    u = torch.from_numpy(np.asarray(start_u, dtype=np.float32)).to(device)
    u = u.clone().requires_grad_(True)
    loss = _data_loss(model, obs, u, 0.0, adapter)
    loss.backward()
    _synchronize(device)


def method_order(case_index: int) -> tuple[str, ...]:
    shift = int(case_index) % len(METHODS)
    return METHODS[shift:] + METHODS[:shift]


def outcome_row(
    method: str,
    sim_id: int,
    runtime_s: float,
    outcome: MethodOutcome,
    theta_true: torch.Tensor,
    y_grid: torch.Tensor,
    adapter: ForcingItrSinAdapter,
) -> dict[str, object]:
    labels = {
        "fno_adam_lbfgs": ("FNO", "Adam -> L-BFGS"),
        "fno_nelder_mead": ("FNO", "Nelder-Mead"),
        "fv_nelder_mead": ("finite_volume", "Nelder-Mead"),
    }
    forward_model, optimizer = labels[method]
    theta_hat = outcome.theta_hat.detach().cpu()
    theta_true = theta_true.detach().cpu()
    severity_hat = adapter.uq_quantity(theta_hat, y_grid.detach().cpu())
    severity_true = adapter.uq_quantity(theta_true, y_grid.detach().cpu())
    base_hat = float(theta_hat[0])
    base_true = float(theta_true[0])
    return {
        "method": method,
        "forward_model": forward_model,
        "optimizer": optimizer,
        "sim_id": int(sim_id),
        "runtime_s": float(runtime_s),
        "objective_half_mse": float(outcome.loss),
        "nfev": int(outcome.nfev),
        "nit": int(outcome.nit),
        "converged": bool(outcome.converged),
        "termination_reason": outcome.termination_reason,
        "R_base_hat": base_hat,
        "R_base_true": base_true,
        "R_base_abs_error": abs(base_hat - base_true),
        "S_R_hat": severity_hat,
        "S_R_true": severity_true,
        "S_R_abs_error": abs(severity_hat - severity_true),
    }


def summarize_rows(case_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped = {
        method: [row for row in case_rows if row["method"] == method]
        for method in METHODS
    }
    if any(len(rows) != N_CASES for rows in grouped.values()):
        counts = {method: len(rows) for method, rows in grouped.items()}
        raise ValueError(f"expected {N_CASES} rows per method, got {counts}.")

    fv_runtime = float(
        np.median([float(row["runtime_s"]) for row in grouped["fv_nelder_mead"]])
    )
    summary = []
    for method in METHODS:
        rows = grouped[method]
        median_runtime = float(np.median([float(row["runtime_s"]) for row in rows]))
        summary.append(
            {
                "method": method,
                "median_runtime_s": median_runtime,
                "speedup_vs_fv_nelder_mead": fv_runtime / median_runtime,
                "median_S_R_abs_error": float(
                    np.median([float(row["S_R_abs_error"]) for row in rows])
                ),
                "median_R_base_abs_error": float(
                    np.median([float(row["R_base_abs_error"]) for row in rows])
                ),
            }
        )
    return summary


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary_rows: list[dict[str, object]]) -> None:
    headers = SUMMARY_COLUMNS
    formatted = []
    for row in summary_rows:
        formatted.append(
            (
                str(row["method"]),
                f"{float(row['median_runtime_s']):.6g}",
                f"{float(row['speedup_vs_fv_nelder_mead']):.4g}",
                f"{float(row['median_S_R_abs_error']):.6g}",
                f"{float(row['median_R_base_abs_error']):.6g}",
            )
        )
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in formatted))
        for i in range(len(headers))
    ]
    print("  ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("  ".join("-" * width for width in widths))
    for row in formatted:
        print("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def _default_out_dir(checkpoint: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return (
        PROJECT_ROOT
        / "runs"
        / "inverse_runtime_comparisons"
        / BENCHMARK
        / _checkpoint_fingerprint(str(checkpoint))
        / timestamp
    )


def _prepare_out_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _protocol(
    *,
    checkpoint: Path,
    checkpoint_fingerprint: str,
    data_dir: Path,
    device: str,
    noise_std: float,
    sigma_global: float,
    ds,
    requested_y: np.ndarray,
    observation_times: np.ndarray,
    starts_u: np.ndarray,
) -> dict:
    return {
        "schema_version": 1,
        "diagnostic": "representative paired runtime comparison",
        "benchmark": BENCHMARK,
        "representation": REPRESENTATION,
        "checkpoint": str(checkpoint),
        "checkpoint_fingerprint": checkpoint_fingerprint,
        "data_dir": str(data_dir),
        "devices": {"fno": device, "finite_volume": "cpu"},
        "grid": {
            "nx": int(ds.Nx),
            "ny": int(ds.Ny),
            "save_stride": int(round(float(ds.t_grid[1] - ds.t_grid[0]) / float(ds.dt))),
        },
        "cases": {"count": N_CASES, "dataset_seed": INVERSION_DATASET_SEED},
        "observations": {
            "times": [float(value) for value in observation_times],
            "sensor_layout": SENSOR_LAYOUT,
            "sensor_count": SENSOR_COUNT,
            "sensor_x_halfwidth": SENSOR_X_HALFWIDTH,
            "requested_y": [float(value) for value in requested_y],
            "noise_seed": NOISE_SEED,
            "noise_std_normalized": float(noise_std),
            "noise_std_K": float(noise_std) * float(sigma_global),
            "temperature_scale": "checkpoint sigma_global",
        },
        "initialization": {
            "type": "lhs",
            "seed": INIT_SEED,
            "n_starts": N_STARTS,
            "starts_u": np.asarray(starts_u, dtype=np.float64).tolist(),
        },
        "optimizers": {
            "fno_adam_lbfgs": {
                "adam_steps": ADAM_STEPS,
                "adam_lr": ADAM_LR,
                "lbfgs_max_iter": LBFGS_STEPS,
                "line_search": "strong_wolfe",
            },
            "nelder_mead": {
                "max_iter": NM_MAXITER,
                "xatol": NM_XATOL,
                "fatol": NM_FATOL,
            },
        },
        "timing": {
            "includes": "all optimization work across all starts",
            "excludes": [
                "checkpoint loading",
                "FV ground-truth generation",
                "observation and noise construction",
                "FNO warm-up",
                "aggregation and file writing",
            ],
            "method_order": [list(method_order(i)) for i in range(N_CASES)],
        },
    }


def compare_case(
    *,
    case_index: int,
    obs,
    model,
    adapter: ForcingItrSinAdapter,
    starts_u: np.ndarray,
    ds,
    fv_base_kwargs: dict,
    device: str,
    mu_global: float,
    sigma_global: float,
) -> list[dict[str, object]]:
    calls = {
        "fno_adam_lbfgs": lambda: run_fno_adam_lbfgs(
            model, obs, adapter, starts_u, device=device
        ),
        "fno_nelder_mead": lambda: run_fno_nelder_mead(
            model, obs, adapter, starts_u, device=device
        ),
        "fv_nelder_mead": lambda: run_fv_nelder_mead(
            ds,
            fv_base_kwargs,
            obs,
            adapter,
            starts_u,
            mu_global=mu_global,
            sigma_global=sigma_global,
        ),
    }
    rows = []
    for method in method_order(case_index):
        outcome, elapsed = _timed(
            calls[method], device=device if method.startswith("fno_") else None
        )
        rows.append(
            outcome_row(
                method,
                obs.sid,
                elapsed,
                outcome,
                obs.theta_true,
                obs.y_grid,
                adapter,
            )
        )
    return rows


def run_comparison(
    *,
    checkpoint: Path,
    out_dir: Path,
    device: str,
    noise_std: float,
    generate_fn: Callable = generate_sim_data,
) -> tuple[Path, Path]:
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if not math.isfinite(noise_std) or noise_std < 0.0:
        raise ValueError(f"noise_std must be finite and non-negative, got {noise_std}.")

    loaded = load_checkpoint(str(checkpoint), device=device)
    validate_checkpoint_config(loaded.config)
    adapter = ForcingItrSinAdapter()
    adapter.validate_model(loaded)

    _prepare_out_dir(out_dir)
    data_dir = out_dir / "inversion_data"
    generate_ground_truth_dataset(loaded.config, data_dir, generate_fn=generate_fn)
    ds = build_dataset_from_dir(
        str(data_dir),
        loaded.config,
        mu_global=loaded.mu_global,
        sigma_global=loaded.sigma_global,
        n_invert=N_CASES,
        n_calibration=0,
    )
    adapter.validate_dataset(ds)

    time_indices, observation_times = resolve_observation_times(
        ds.t_grid, PAPER_OBSERVATION_TIMES
    )
    requested_y = np.linspace(
        float(ds.y_grid[0]), float(ds.y_grid[-1]), SENSOR_COUNT
    )
    observations = []
    for sid in ds._split_ids["test"]:
        obs = build_observation_set(
            ds,
            int(sid),
            time_indices,
            device=device,
            adapter=adapter,
            interface_x=INTERFACE_X,
            sensor_x_halfwidth=SENSOR_X_HALFWIDTH,
            sensor_n_y=SENSOR_COUNT,
            sensor_layout=SENSOR_LAYOUT,
            sensor_y=requested_y,
            observation_times=observation_times,
        )
        if obs.mask is None or int(obs.mask.sum()) != SENSOR_COUNT:
            realized = None if obs.mask is None else int(obs.mask.sum())
            raise ValueError(
                f"eight-sensor protocol resolved {realized} sensor pixels."
            )
        add_measurement_noise(obs, noise_std, seed=NOISE_SEED + int(sid))
        observations.append(obs)

    starts_u = lhs_starts(adapter)
    warm_fno(
        loaded.model,
        observations[0],
        adapter,
        starts_u[0],
        device=device,
    )
    fv_base_kwargs = adapter.fv_base_kwargs(ds)

    case_rows = []
    for case_index, obs in enumerate(observations):
        print(f"[sim {obs.sid}] running paired inverse methods", flush=True)
        case_rows.extend(
            compare_case(
                case_index=case_index,
                obs=obs,
                model=loaded.model,
                adapter=adapter,
                starts_u=starts_u,
                ds=ds,
                fv_base_kwargs=fv_base_kwargs,
                device=device,
                mu_global=loaded.mu_global,
                sigma_global=loaded.sigma_global,
            )
        )

    summary_rows = summarize_rows(case_rows)
    case_path = out_dir / "inverse_runtime_cases.csv"
    summary_path = out_dir / "inverse_runtime_summary.csv"
    protocol_path = out_dir / "inverse_runtime_protocol.json"
    _write_csv(case_path, CASE_COLUMNS, case_rows)
    _write_csv(summary_path, SUMMARY_COLUMNS, summary_rows)
    protocol = _protocol(
        checkpoint=checkpoint,
        checkpoint_fingerprint=_checkpoint_fingerprint(str(checkpoint)),
        data_dir=data_dir,
        device=device,
        noise_std=noise_std,
        sigma_global=loaded.sigma_global,
        ds=ds,
        requested_y=requested_y,
        observation_times=observation_times,
        starts_u=starts_u,
    )
    with protocol_path.open("w") as handle:
        json.dump(protocol, handle, indent=2, sort_keys=True)

    print("\nRepresentative paired runtime comparison (n=3):")
    print_summary(summary_rows)
    print(f"\nCase results: {case_path}")
    print(f"Summary: {summary_path}")
    print(f"Protocol: {protocol_path}")
    return case_path, summary_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Unique output directory (default: checkpoint-scoped timestamped run).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="FNO device (default preference: cuda, mps, then cpu).",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=DEFAULT_NOISE_STD,
        help="IID measurement-noise standard deviation in normalized temperature.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        checkpoint = args.checkpoint.expanduser().resolve()
        out_dir = (
            args.out_dir.expanduser().resolve()
            if args.out_dir is not None
            else _default_out_dir(checkpoint)
        )
        run_comparison(
            checkpoint=checkpoint,
            out_dir=out_dir,
            device=args.device or default_device(),
            noise_std=args.noise_std,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
