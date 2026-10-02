"""CN algebra and matched direct-state Adam gates for raw, exact, and MG losses."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from scipy.sparse.linalg import eigsh
import torch
import yaml

from src.physics.boundary_forcing import build_qL, build_qL_integral
from src.physics.fv_preconditioner import ExactCNInverse, MGOneCycleInverse
from src.physics.fv_residual import (
    FullBCData, build_homogeneous_cn_geom, cn_step_residual, explicit_cn_rhs,
    implicit_cn_action,
)
from src.physics.fv_solver_2d import FVSolver2D, Layer2D
from src.physics.init_conditions import IC_FAMILIES, IC_SAMPLERS, build_ic


PROBES = ("smooth", "oscillatory", "mixed", "boundary_local")


def build_problem(grid_size: int, config: dict):
    grid = np.linspace(0, 1, grid_size)
    temporal = dict(A=config["flux_amplitude"], f=config["flux_frequency"],
                    t_on=0.0, t_off=0.2, phase=0.0, tukey_alpha=0.5, rectified=True)
    q, _ = build_qL("sin", temporal, "uniform", {}, grid, config["ramp_seconds"])
    qint, _ = build_qL_integral("sin", temporal, "uniform", {}, grid,
                               config["ramp_seconds"])
    solver = FVSolver2D(
        a=0, b=1, c=0, d=1, Nx=grid_size, Ny=grid_size,
        layers=[Layer2D(0, 1, 1, 1, 1)], interface_R=None,
        lam_target=0.5, dt=config["dt"], t_final=config["t_final"],
        flux_f=0, flux_A=0, t_on=0, t_off=0.2, phase=0,
        q_left_fn=q, q_left_integral_fn=qint,
    )
    geom = build_homogeneous_cn_geom(grid, grid, 1, solver.dt,
                                     sigma_global=config["sigma_global"])
    return solver, geom, ExactCNInverse(geom)


def boundary_data(solver, config, time_n):
    return FullBCData(
        T_right_tilde=torch.tensor((300 - config["mu_global"]) / config["sigma_global"],
                                   dtype=torch.float64),
        qL_n=torch.as_tensor(solver.q_left(time_n), dtype=torch.float64).unsqueeze(0),
        qL_np1=torch.as_tensor(solver.q_left(time_n + solver.dt),
                              dtype=torch.float64).unsqueeze(0),
        qL_int=torch.as_tensor(solver.q_left_integral(time_n, time_n + solver.dt),
                              dtype=torch.float64).unsqueeze(0),
    )


def probe_fields(grid_size, seed, rms):
    x = torch.linspace(0, 1, grid_size, dtype=torch.float64)[:-1, None]
    y = torch.linspace(0, 1, grid_size, dtype=torch.float64)[None, :]
    smooth = torch.cos(torch.pi * x / 2) * torch.cos(torch.pi * y)
    oscillatory = torch.cos(torch.pi * (grid_size - 2) * x) * torch.cos(
        torch.pi * (grid_size - 1) * y)
    generator = torch.Generator().manual_seed(seed)
    mixed = smooth + 0.5 * oscillatory + 0.2 * torch.randn(
        smooth.shape, generator=generator, dtype=torch.float64)
    boundary = torch.exp(-x * (grid_size - 1) / 2) * (1 + torch.cos(torch.pi * y))
    return dict(zip(PROBES, [field / field.square().mean().sqrt() * rms
                            for field in (smooth, oscillatory, mixed, boundary)]))


def audit_algebra(solver, geom, inverse, config):
    n = solver.Nx
    active = np.arange(n * n).reshape(n, n, order="F")[:-1].ravel(order="F")
    diff = inverse.matrix - solver.A_csr[active][:, active]
    matrix_error = float(np.max(np.abs(diff.data), initial=0))
    generator = torch.Generator().manual_seed(config["seed"])
    field = torch.randn((2, n, n), dtype=torch.float64, generator=generator)
    field[:, -1] = 0
    action = implicit_cn_action(field, geom).transpose(-2, -1).reshape(2, -1)
    expected = inverse.matrix @ field[:, :-1].transpose(-2, -1).reshape(2, -1).numpy().T
    action_error = float(np.max(np.abs(action.numpy().T - expected)))
    bc = boundary_data(solver, config, 0)
    rhs = explicit_cn_rhs(field, geom, bc)
    physical = field[0].numpy() * config["sigma_global"] + 300
    solver_next = (solver.cn_step(physical, 0) - 300) / config["sigma_global"]
    rhs_error = float(np.max(np.abs(inverse(rhs)[0].numpy() - solver_next[:-1])))
    residual = implicit_cn_action(field, geom) - rhs
    identity_error = float((inverse(residual) - (field[:, :-1] - inverse(rhs))).abs().max())
    capacity = (geom.rho_cp[:-1] * geom.dx[:-1, None] * geom.dy[None, :]).numpy()
    weighted = inverse.matrix.multiply(capacity.ravel(order="F")[:, None]).tocsr()
    asym = weighted - weighted.T
    symmetry_error = float(np.max(np.abs(asym.data), initial=0))
    hessian = (inverse.matrix.T @ inverse.matrix).tocsr()
    if hessian.shape[0] <= 256:
        spectrum = np.linalg.eigvalsh(hessian.toarray())[[0, -1]]
    else:
        spectrum = [float(eigsh(hessian, k=1, which=which, tol=1e-6,
                                maxiter=20000, return_eigenvectors=False)[0])
                    for which in ("SA", "LA")]
    # Audit the gradient of A^-1(Ax-b), not only the forward solve identity.
    free = field[:, :-1].clone().requires_grad_()
    full = torch.cat((free, torch.zeros_like(field[:, -1:])), dim=1)
    transformed = inverse(implicit_cn_action(full, geom) - rhs)
    gradient = torch.autograd.grad(transformed.square().mean(), free)[0]
    gradient_error = float((gradient - 2 * (free - inverse(rhs)) / free.numel()).abs().max())
    result = dict(matrix_max_error=matrix_error, action_max_error=action_error,
                  solver_step_max_error=rhs_error, exact_identity_max_error=identity_error,
                  gradient_max_error=gradient_error, capacity_symmetry_max_error=symmetry_error,
                  raw_hessian_lambda_min=float(spectrum[0]),
                  raw_hessian_lambda_max=float(spectrum[1]),
                  raw_hessian_condition=float(spectrum[1] / spectrum[0]),
                  spectral_estimator="eigvalsh" if hessian.shape[0] <= 256 else "eigsh")
    result["passed"] = all(result[key] <= 1e-10 for key in (
        "matrix_max_error", "action_max_error", "solver_step_max_error",
        "exact_identity_max_error", "gradient_max_error", "capacity_symmetry_max_error"))
    return result


def optimize_step(source, target, perturbation, geom, bc, inverse, config, objective,
                  row_context, writer, mg=None):
    right = bc.T_right_tilde
    initial = target[:, :-1] + perturbation.unsqueeze(0)
    free = initial.clone().requires_grad_()
    optimizer = torch.optim.Adam([free], lr=config["learning_rate"], weight_decay=0)
    tolerance = max(config["relative_error_tolerance"] * float(perturbation.square().mean().sqrt()),
                    100 * row_context["solver_floor"])
    reached = None
    start = time.perf_counter()
    for step in range(config["updates"] + 1):
        prediction = torch.cat((free, torch.full_like(source[:, -1:], right.item())), dim=1)
        residual = cn_step_residual(source, prediction, geom, bc)
        correction = mg(residual) if objective == "mg" else inverse(residual) if objective == "exact" else residual
        loss = correction.square().mean()
        with torch.no_grad():
            error = float((free - target[:, :-1]).square().mean().sqrt())
        if reached is None and error <= tolerance:
            reached = step
        if step % config["log_every"] == 0 or step == config["updates"]:
            grad = torch.autograd.grad(loss, free, retain_graph=True)[0]
            writer.writerow({**row_context, "objective": objective, "update": step,
                             "loss": float(loss.detach()), "state_rmse_normalized": error,
                             "state_rmse_K": error * config["sigma_global"],
                             "raw_residual_rms": float(residual.detach().square().mean().sqrt()),
                             "exact_correction_rms": float(inverse(residual.detach()).square().mean().sqrt()),
                             "gradient_norm": float(grad.norm()),
                             "elapsed_seconds": time.perf_counter() - start})
        if step == config["updates"]:
            break
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite {objective} loss at update {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return dict(final_rmse_normalized=error, tolerance=tolerance,
                steps_to_tolerance=reached, elapsed_seconds=time.perf_counter() - start)


def gate_result(cases, config, candidate="exact"):
    fine = [case for case in cases if case["grid_size"] == max(config["grid_sizes"])]
    exact_success = all(case[candidate]["steps_to_tolerance"] is not None for case in fine)
    ratios = [case["raw"]["final_rmse_normalized"] /
              max(case[candidate]["final_rmse_normalized"], case["solver_floor"], 1e-15)
              for case in fine]
    def steps(result):
        value = result["steps_to_tolerance"]
        return config["updates"] + 1 if value is None else value

    speedups = [steps(case["raw"]) / max(steps(case[candidate]), 1) for case in fine]
    speedup, ratio = float(np.median(speedups)), float(np.median(ratios))
    result = dict(passed=exact_success and speedup >= config["min_step_speedup"]
                and ratio >= config["min_final_error_ratio"],
                fine_grid_median_step_speedup=speedup, fine_grid_median_final_error_ratio=ratio)
    result[f"fine_grid_{candidate}_all_reached_tolerance"] = exact_success
    return result


def audit_mg(geom, config):
    mg = MGOneCycleInverse(geom, **config["mg"]["solver"])
    probes = probe_fields(geom.Nx, config["seed"], config["perturbation_rms"])
    errors = {}
    for name, field in probes.items():
        full = torch.cat((field, torch.zeros(1, geom.Ny, dtype=field.dtype)), dim=0)[None]
        correction = mg(implicit_cn_action(full, geom))[0]
        errors[name] = float((correction - field).norm() / field.norm())
    return dict(level_shapes=mg.level_shapes, probe_relative_errors=errors,
                cycles_per_application=1,
                passed=max(errors.values()) <= config["mg"]["max_probe_relative_error"])


def run_gate(config, output_dir: Path):
    if output_dir.exists():
        raise FileExistsError(f"Use a new isolated run directory: {output_dir}")
    if not config["grid_sizes"] or min(config["grid_sizes"]) < 4:
        raise ValueError("grid_sizes must be nonempty and each grid must have at least 4 nodes")
    if config["updates"] < 1 or config["log_every"] < 1:
        raise ValueError("updates and log_every must be positive")
    if config["learning_rate"] <= 0 or config["sigma_global"] <= 0:
        raise ValueError("learning_rate and sigma_global must be positive")
    torch.set_num_threads(1)
    output_dir.mkdir(parents=True)
    with_mg = "mg" in config.get("controls", ["raw", "exact"])
    stage = "phase4_direct_state_mg" if with_mg else "phase0_direct_state"
    frozen = dict(config=config, stage=stage, problem="homogeneous_sin_uniform",
                  representation="direct_temperature_state", protocol_role="development",
                  optimizer="Adam", weight_decay=0, dtype="float64", device="cpu",
                  ic_families=list(IC_FAMILIES), probes=list(PROBES),
                  initialization="exact_step_plus_fixed_probe_diagnostic_only")
    encoded = json.dumps(frozen, sort_keys=True).encode()
    frozen["protocol_sha256"] = hashlib.sha256(encoded).hexdigest()
    (output_dir / "config_used.yaml").write_text(yaml.safe_dump(frozen, sort_keys=False))
    for filename, args in (("git_commit.txt", ["git", "rev-parse", "HEAD"]),
                           ("git_status.txt", ["git", "status", "--short"])):
        (output_dir / filename).write_text(subprocess.check_output(args, cwd=PROJECT_ROOT, text=True))
    (output_dir / "source_hashes.json").write_text(json.dumps({str(path.relative_to(PROJECT_ROOT)):
        hashlib.sha256(path.read_bytes()).hexdigest() for path in (
            Path(__file__), PROJECT_ROOT / "src/physics/fv_residual.py",
            PROJECT_ROOT / "src/physics/fv_preconditioner.py",
            PROJECT_ROOT / "src/physics/fv_solver_2d.py")}, indent=2))
    audits, cases = {}, []
    fieldnames = ["grid_size", "ic_family", "probe", "time_n", "solver_floor", "objective",
                  "update", "loss", "state_rmse_normalized", "state_rmse_K", "raw_residual_rms",
                  "exact_correction_rms", "gradient_norm", "elapsed_seconds"]
    with (output_dir / "diagnostics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        problems = {}
        for n in config["grid_sizes"]:
            solver, geom, inverse = build_problem(n, config)
            problems[n] = (solver, geom, inverse)
            audits[str(n)] = audit_algebra(solver, geom, inverse, config)
            if with_mg:
                audits[str(n)]["mg"] = audit_mg(geom, config)
                audits[str(n)]["passed"] &= audits[str(n)]["mg"]["passed"]
            print(f"grid={n} algebra={audits[str(n)]['passed']} "
                  f"raw_hessian_condition={audits[str(n)]['raw_hessian_condition']:.3g}", flush=True)
        (output_dir / "algebra_audit.json").write_text(json.dumps(audits, indent=2))
        algebra_ok = all(audit["passed"] for audit in audits.values())
        if algebra_ok:
            cases = run_cases(problems, config, writer, stream, time_n=0)
        candidate = "mg" if with_mg else "exact"
        first_gate = gate_result(cases, config, candidate) if cases else {"passed": False}
        (output_dir / "first_interval_gate.json").write_text(json.dumps(first_gate, indent=2))
        later = []
        if first_gate["passed"]:
            later = run_cases(problems, config, writer, stream, time_n=config["later_time"])
        later_gate = gate_result(later, config, candidate) if later else {"passed": False, "skipped": True}
    summary = dict(stage=stage, protocol_sha256=frozen["protocol_sha256"],
                   candidate=candidate,
                   algebra_passed=algebra_ok, first_interval_gate=first_gate,
                   later_time_gate=later_gate, cases=cases, later_cases=later,
                   passed=algebra_ok and first_gate["passed"] and later_gate["passed"])
    (output_dir / "final_metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({key: value for key, value in summary.items()
                      if key not in ("cases", "later_cases")}, indent=2), flush=True)
    return summary


def run_cases(problems, config, writer, stream, *, time_n):
    cases = []
    for n, (solver, geom, inverse) in problems.items():
        probes = probe_fields(n, config["seed"], config["perturbation_rms"])
        mg = MGOneCycleInverse(geom, **config["mg"]["solver"]) if "mg" in config.get("controls", []) else None
        for family_idx, family in enumerate(IC_FAMILIES):
            rng = np.random.default_rng(config["seed"] + family_idx)
            physical = build_ic(family, IC_SAMPLERS[family](rng), solver.X, solver.Y, T_right=300)
            for step in range(round(time_n / solver.dt)):
                physical = solver.cn_step(physical, step * solver.dt)
            bc = boundary_data(solver, config, time_n)
            source = torch.from_numpy((physical.astype(float) - config["mu_global"])
                                      / config["sigma_global"]).unsqueeze(0)
            target = torch.from_numpy((solver.cn_step(physical, time_n) - config["mu_global"])
                                      / config["sigma_global"]).unsqueeze(0)
            floor = float(inverse(cn_step_residual(source, target, geom, bc)).square().mean().sqrt())
            for name, perturbation in probes.items():
                context = dict(grid_size=n, ic_family=family, probe=name, time_n=time_n,
                               solver_floor=floor)
                result = dict(context)
                for objective in config.get("controls", ["raw", "exact"]):
                    result[objective] = optimize_step(source, target, perturbation, geom, bc,
                                                      inverse, config, objective, context, writer, mg)
                cases.append(result)
                stream.flush()
                print(f"grid={n} t={time_n} ic={family} probe={name} "
                      f"raw={result['raw']['final_rmse_normalized']:.3g} "
                      f"exact={result['exact']['final_rmse_normalized']:.3g}" +
                      (f" mg={result['mg']['final_rmse_normalized']:.3g}" if mg is not None else ""), flush=True)
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "conf/config.yaml")
    parser.add_argument("--mg", action="store_true")
    args = parser.parse_args()
    physics = yaml.safe_load(args.config.read_text())["physics_test"]
    config = physics["phase0"]
    if args.mg:
        config.update(controls=["raw", "exact", "mg"], mg=physics["mg"])
    summary = run_gate(config, args.output_dir)
    raise SystemExit(0 if summary["passed"] else 2)


if __name__ == "__main__":
    main()
