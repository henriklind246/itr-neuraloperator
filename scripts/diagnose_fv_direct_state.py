#!/usr/bin/env python3
"""Causal direct-state reachability matrix for the fully discrete FV objective."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.physics.boundary_forcing import (
    SPATIAL_SAMPLERS,
    TEMPORAL_SAMPLERS,
    reconstruct_qL,
)
from src.physics.fv_solver_2d import FVSolver2D, Layer2D
from src.physics.init_conditions import IC_SAMPLERS, build_ic
from src.physics.one_step_objective import (
    build_cn_tensors,
    cn_free_diagonal,
    explicit_cn_rhs,
    implicit_cn_action,
)


OBJECTIVES = ("raw_ls", "variational", "defect")
OPTIMIZERS = ("adam", "lbfgs")
HOMOGENEOUS_GATE_CASES = (
    ("uniform_2d", 0.000),
    ("random_sinusoid_2d", 0.075),
    ("grf_2d", 0.150),
    ("hot_spot_2d", 0.225),
)
GATE_METRICS = (
    "overall_field",
    "longest_lead",
    "physical_defect",
    "spatial_mean_bias",
    "lowest_4x4_modes",
)


def build_problem(
    *, grid_size: int, dt: float, t_final: float, flux_amplitude: float,
    flux_frequency: float, interface_x: float, resistance: float,
) -> FVSolver2D:
    return FVSolver2D(
        a=0.0,
        b=1.0,
        c=0.0,
        d=1.0,
        Nx=grid_size,
        Ny=grid_size,
        layers=[
            Layer2D(0.0, interface_x, rho=1.0, cp=1.0, k=2.0),
            Layer2D(interface_x, 1.0, rho=1.0, cp=1.0, k=1.0),
        ],
        interface_R=[resistance],
        lam_target=0.5,
        dt=dt,
        flux_f=flux_frequency,
        flux_A=flux_amplitude,
        t_on=0.0,
        t_off=min(0.2, t_final),
        t_final=t_final,
        phase=0.0,
    )


def active_system(sim: FVSolver2D) -> tuple[sp.csr_matrix, np.ndarray]:
    active = np.asarray(
        [i + j * sim.Nx for j in range(sim.Ny) for i in range(sim.Nx - 1)],
        dtype=np.int64,
    )
    matrix = sim.A_csr[active][:, active].tocsr()
    capacity = (
        sim.rho_nodes[: sim.Nx - 1]
        * sim.cp_nodes[: sim.Nx - 1]
        * sim.dx[: sim.Nx - 1, None]
        * sim.dy[None, :]
    ).ravel(order="F")
    return matrix, capacity


def normalized_rhs(
    sim: FVSolver2D,
    z_n: np.ndarray,
    t_n: float,
    *,
    sigma: float,
) -> np.ndarray:
    """Assemble the active normalized CN right-hand side without solving it."""
    Nx, Ny = sim.Nx, sim.Ny
    act = slice(0, Nx - 1)
    rhs = np.zeros((Nx, Ny), dtype=np.float64)
    rhs[act] = (
        1.0
        - sim.r_w[act]
        - sim.r_e[act]
        - sim.r_s[act]
        - sim.r_n[act]
    ) * z_n[act]
    rhs[1 : Nx - 1] += sim.r_w[1 : Nx - 1] * z_n[0 : Nx - 2]
    rhs[act] += sim.r_e[act] * z_n[1:Nx]
    rhs[act, 1:] += sim.r_s[act, 1:] * z_n[act, 0 : Ny - 1]
    rhs[act, 0 : Ny - 1] += sim.r_n[act, 0 : Ny - 1] * z_n[act, 1:Ny]

    rho_cp_left = sim.rho_nodes[0] * sim.cp_nodes[0]
    if sim.q_left_integral is not None:
        q_integral = np.asarray(
            sim.q_left_integral(t_n, t_n + sim.dt), dtype=np.float64
        )
        rhs[0] += 2.0 * q_integral / (rho_cp_left * sim.hx * sigma)
    else:
        q_n = np.asarray(sim.q_left(t_n), dtype=np.float64)
        q_np1 = np.asarray(sim.q_left(t_n + sim.dt), dtype=np.float64)
        rhs[0] += sim.dt * (q_n + q_np1) / (
            rho_cp_left * sim.hx * sigma
        )
    return rhs[act].ravel(order="F")


def scipy_to_torch_sparse(matrix: sp.csr_matrix) -> torch.Tensor:
    coo = matrix.tocoo()
    indices = torch.as_tensor(
        np.vstack((coo.row, coo.col)), dtype=torch.long
    )
    values = torch.as_tensor(coo.data, dtype=torch.float64)
    return torch.sparse_coo_tensor(
        indices, values, size=coo.shape, dtype=torch.float64
    ).coalesce()


def sparse_mv(matrix: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    return torch.sparse.mm(matrix, vector[:, None])[:, 0]


def raw_region_loss(
    residual: torch.Tensor, *, nx: int, ny: int
) -> torch.Tensor:
    field = residual.view(ny, nx - 1).transpose(0, 1)
    interior = field[1 : nx - 1, 1 : ny - 1].square().mean()
    left = field[0].square().mean()
    top_bottom = torch.cat((field[1 : nx - 1, 0], field[1 : nx - 1, -1]))
    return interior + left + top_bottom.square().mean()


def variational_loss(
    z: torch.Tensor,
    matrix: torch.Tensor,
    rhs: torch.Tensor,
    capacity: torch.Tensor,
) -> torch.Tensor:
    Az = sparse_mv(matrix, z)
    return (
        0.5 * torch.dot(z, capacity * Az)
        - torch.dot(z, capacity * rhs)
    ) / capacity.sum()


def approximate_defect(
    residual: torch.Tensor,
    matrix: torch.Tensor,
    diagonal: torch.Tensor,
    *,
    sweeps: int,
    omega: float,
) -> torch.Tensor:
    correction = torch.zeros_like(residual)
    for _ in range(sweeps):
        correction = correction + omega * (
            residual - sparse_mv(matrix, correction)
        ) / diagonal
    return correction


def objective_value(
    name: str,
    z: torch.Tensor,
    matrix: torch.Tensor,
    rhs: torch.Tensor,
    capacity: torch.Tensor,
    diagonal: torch.Tensor,
    *,
    nx: int,
    ny: int,
    defect_sweeps: int,
    defect_omega: float,
) -> torch.Tensor:
    residual = sparse_mv(matrix, z) - rhs
    if name == "raw_ls":
        return raw_region_loss(residual, nx=nx, ny=ny)
    if name == "variational":
        return variational_loss(z, matrix, rhs, capacity)
    if name == "defect":
        defect = approximate_defect(
            residual,
            matrix,
            diagonal,
            sweeps=defect_sweeps,
            omega=defect_omega,
        )
        return defect.square().mean()
    raise ValueError(f"unknown objective {name!r}")


def optimize_step(
    z_initial: np.ndarray,
    matrix: torch.Tensor,
    rhs: torch.Tensor,
    capacity: torch.Tensor,
    *,
    objective: str,
    optimizer_name: str,
    nx: int,
    ny: int,
    adam_updates: int,
    adam_lr: float,
    lbfgs_iterations: int,
    defect_sweeps: int,
    defect_omega: float,
) -> tuple[np.ndarray, dict[str, float]]:
    z = torch.tensor(z_initial, dtype=torch.float64, requires_grad=True)
    diagonal = torch.zeros(matrix.shape[0], dtype=torch.float64)
    diagonal_mask = matrix.indices()[0] == matrix.indices()[1]
    diagonal[matrix.indices()[0, diagonal_mask]] = matrix.values()[diagonal_mask]
    closure_calls = 0

    def loss_fn() -> torch.Tensor:
        return objective_value(
            objective,
            z,
            matrix,
            rhs,
            capacity,
            diagonal,
            nx=nx,
            ny=ny,
            defect_sweeps=defect_sweeps,
            defect_omega=defect_omega,
        )

    initial_loss = float(loss_fn().detach())
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam([z], lr=adam_lr)
        for _ in range(adam_updates):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn()
            loss.backward()
            optimizer.step()
            closure_calls += 1
    elif optimizer_name == "lbfgs":
        optimizer = torch.optim.LBFGS(
            [z],
            lr=1.0,
            max_iter=lbfgs_iterations,
            max_eval=max(lbfgs_iterations * 5 // 4, lbfgs_iterations + 1),
            tolerance_grad=1.0e-11,
            tolerance_change=1.0e-14,
            history_size=50,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            nonlocal closure_calls
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn()
            loss.backward()
            closure_calls += 1
            return loss

        optimizer.step(closure)
    else:
        raise ValueError(f"unknown optimizer {optimizer_name!r}")

    final_loss = float(loss_fn().detach())
    return z.detach().numpy(), {
        "initial_objective": initial_loss,
        "final_objective": final_loss,
        "closure_calls": float(closure_calls),
    }


def rollout_cell(
    sim: FVSolver2D,
    matrix_scipy: sp.csr_matrix,
    capacity_np: np.ndarray,
    truth: np.ndarray,
    *,
    objective: str,
    optimizer_name: str,
    sigma: float,
    adam_updates: int,
    adam_lr: float,
    lbfgs_iterations: int,
    defect_sweeps: int,
    defect_omega: float,
) -> tuple[dict[str, float | str], np.ndarray]:
    matrix = scipy_to_torch_sparse(matrix_scipy)
    capacity = torch.as_tensor(capacity_np, dtype=torch.float64)
    prediction = np.zeros_like(truth)
    prediction[0] = truth[0]
    total_calls = 0.0
    start = time.perf_counter()

    for n in range(len(sim.t) - 1):
        rhs_np = normalized_rhs(sim, prediction[n], float(sim.t[n]), sigma=sigma)
        rhs = torch.as_tensor(rhs_np, dtype=torch.float64)
        initial = prediction[n, : sim.Nx - 1].ravel(order="F")
        solved, info = optimize_step(
            initial,
            matrix,
            rhs,
            capacity,
            objective=objective,
            optimizer_name=optimizer_name,
            nx=sim.Nx,
            ny=sim.Ny,
            adam_updates=adam_updates,
            adam_lr=adam_lr,
            lbfgs_iterations=lbfgs_iterations,
            defect_sweeps=defect_sweeps,
            defect_omega=defect_omega,
        )
        prediction[n + 1, : sim.Nx - 1] = solved.reshape(
            (sim.Nx - 1, sim.Ny), order="F"
        )
        prediction[n + 1, sim.Nx - 1] = 0.0
        total_calls += info["closure_calls"]

    elapsed = time.perf_counter() - start
    residuals = []
    raw_losses = []
    matrix_t = matrix
    for n in range(len(sim.t) - 1):
        z = torch.as_tensor(
            prediction[n + 1, : sim.Nx - 1].ravel(order="F"),
            dtype=torch.float64,
        )
        rhs = torch.as_tensor(
            normalized_rhs(sim, prediction[n], float(sim.t[n]), sigma=sigma),
            dtype=torch.float64,
        )
        residual = sparse_mv(matrix_t, z) - rhs
        residuals.append(float(residual.square().mean().sqrt()))
        raw_losses.append(float(raw_region_loss(residual, nx=sim.Nx, ny=sim.Ny)))

    error_K = sigma * (prediction - truth)
    face = next(iter(sim.interface_face_map))
    pred_jump = sigma * (prediction[:, face] - prediction[:, face + 1])
    true_jump = sigma * (truth[:, face] - truth[:, face + 1])
    row: dict[str, float | str] = {
        "optimizer": optimizer_name,
        "objective": objective,
        "field_rmse_K": float(np.sqrt(np.mean(error_K**2))),
        "final_field_rmse_K": float(np.sqrt(np.mean(error_K[-1] ** 2))),
        "jump_rmse_K": float(np.sqrt(np.mean((pred_jump - true_jump) ** 2))),
        "pred_jump_rms_K": float(np.sqrt(np.mean(pred_jump**2))),
        "true_jump_rms_K": float(np.sqrt(np.mean(true_jump**2))),
        "residual_rms_normalized": float(np.sqrt(np.mean(np.square(residuals)))),
        "raw_region_loss_mean": float(np.mean(raw_losses)),
        "closure_calls": total_calls,
        "wall_seconds": elapsed,
    }
    return row, prediction


def symmetry_audit(
    matrix: sp.csr_matrix, capacity: np.ndarray
) -> dict[str, float]:
    weighted = sp.diags(capacity) @ matrix
    asymmetry = weighted - weighted.T
    max_asymmetry = (
        0.0 if asymmetry.nnz == 0 else float(np.max(np.abs(asymmetry.data)))
    )
    smallest = float(
        spla.eigsh(weighted, k=1, which="SA", return_eigenvectors=False)[0]
    )
    largest = float(
        spla.eigsh(weighted, k=1, which="LA", return_eigenvectors=False)[0]
    )
    return {
        "capacity_weighted_max_asymmetry": max_asymmetry,
        "capacity_weighted_lambda_min": smallest,
        "capacity_weighted_lambda_max": largest,
        "capacity_weighted_condition_estimate": largest / smallest,
    }


def calibrated_gate_thresholds(
    copy_error: float,
    floor_error: float,
    oracle_error: float,
    *,
    separation: float = 1.0e3,
) -> dict[str, float | bool]:
    values = np.asarray(
        [copy_error, floor_error, oracle_error], dtype=np.float64,
    )
    finite = bool(np.isfinite(values).all())
    valid = bool(
        finite
        and copy_error >= 0.0
        and floor_error >= 0.0
        and oracle_error >= 0.0
        and copy_error >= float(separation) * floor_error
    )
    oracle_threshold = max(
        100.0 * float(floor_error), 1.0e-3 * float(copy_error),
    )
    adam_threshold = max(
        25.0 * float(oracle_error),
        1000.0 * float(floor_error),
        1.0e-2 * float(copy_error),
    )
    return {
        "valid": valid,
        "copy_to_floor_ratio": (
            float("inf") if floor_error == 0.0 and copy_error > 0.0
            else (
                0.0 if floor_error == 0.0
                else float(copy_error / floor_error)
            )
        ),
        "oracle_threshold": float(oracle_threshold),
        "oracle_qualified": bool(valid and oracle_error <= oracle_threshold),
        "adam_threshold": float(adam_threshold),
    }


def preregistered_max_lead(
    completed_update: int,
    total_updates: int,
    feasible_horizon: float,
) -> float:
    if total_updates <= 0:
        raise ValueError("total_updates must be positive")
    fraction = (int(completed_update) + 1) / int(total_updates)
    if fraction <= 0.10:
        requested = 0.05
    elif fraction <= 0.25:
        requested = 0.10
    elif fraction <= 0.50:
        requested = 0.20
    else:
        requested = 0.30
    return min(float(requested), float(feasible_horizon))


def _fixed_gate_parameters(
    family: str,
    *,
    dt: float,
    seed: int,
) -> tuple[dict, dict]:
    ic_rng = np.random.default_rng(int(seed))
    forcing_rng = np.random.default_rng(int(seed) + 10_000)
    ic_params = IC_SAMPLERS[family](ic_rng)
    forcing_params = TEMPORAL_SAMPLERS["sin"](
        forcing_rng,
        dt=float(dt),
        t_final=0.3,
        t_on=0.0,
        t_off=0.2,
        phase=0.0,
        tukey_alpha=0.5,
    )
    return ic_params, forcing_params


def build_homogeneous_gate_case(
    *,
    family: str,
    source_time: float,
    grid_size: int,
    dt: float = 0.005,
    sigma: float = 10.0,
    seed: int = 0,
) -> tuple[FVSolver2D, np.ndarray, dict]:
    horizon = 0.3 - float(source_time)
    n_steps = int(round(horizon / float(dt)))
    if n_steps <= 0 or not math.isclose(
        horizon / float(dt), n_steps, rel_tol=0.0, abs_tol=1.0e-10,
    ):
        raise ValueError("source time must leave an integral positive dt horizon")
    axis = np.linspace(0.0, 1.0, int(grid_size), dtype=np.float64)
    X, Y = np.meshgrid(axis, axis, indexing="ij")
    ic_params, forcing_params = _fixed_gate_parameters(
        family, dt=dt, seed=seed,
    )
    source = np.asarray(
        build_ic(family, ic_params, X, Y, T_right=300.0),
        dtype=np.float64,
    )
    spatial_params = SPATIAL_SAMPLERS["uniform"](
        np.random.default_rng(int(seed) + 20_000), c=0.0, d=1.0,
    )
    forcing = reconstruct_qL(
        "sin",
        forcing_params,
        "uniform",
        spatial_params,
        t_ramp=2.0 * float(dt),
    )

    def q_local(tau: float) -> np.ndarray:
        return forcing.evaluate_points(axis, float(source_time) + float(tau))

    def q_integral_local(tau_lo: float, tau_hi: float) -> np.ndarray:
        return forcing.integral(
            axis,
            float(source_time) + float(tau_lo),
            float(source_time) + float(tau_hi),
        )

    sim = FVSolver2D(
        a=0.0,
        b=1.0,
        c=0.0,
        d=1.0,
        Nx=int(grid_size),
        Ny=int(grid_size),
        lam_target=0.5,
        layers=[Layer2D(0.0, 1.0, rho=1.0, cp=1.0, k=1.0)],
        t_final=horizon,
        flux_f=float(forcing_params["f"]),
        flux_A=0.0,
        t_on=0.0,
        t_off=horizon,
        phase=0.0,
        dt=float(dt),
        q_left_fn=q_local,
        q_left_integral_fn=q_integral_local,
        interface_R=None,
    )
    _, _, _, truth_K = sim.solve(T0=source, store_trajectory=True)
    truth = (np.asarray(truth_K, dtype=np.float64) - 300.0) / float(sigma)
    metadata = {
        "family": str(family),
        "source_time": float(source_time),
        "horizon": float(horizon),
        "n_intervals": int(n_steps),
        "seed": int(seed),
        "ic_params": _jsonable(ic_params),
        "forcing": {
            "temporal_family": "sin",
            "temporal_params": _jsonable(forcing_params),
            "spatial_family": "uniform",
            "spatial_params": _jsonable(spatial_params),
            "absolute_time_offset": float(source_time),
        },
    }
    return sim, truth, metadata


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _resolve_gate_normalizer(
    path: str | None,
    fallback_sigma: float,
    *,
    smoke: bool,
) -> tuple[float, dict[str, str | float | bool | None]]:
    if path is None:
        if not smoke:
            raise ValueError(
                "the registered homogeneous gate requires "
                "--training-normalizer-path"
            )
        return float(fallback_sigma), {
            "uses_training_trajectories": False,
            "source_path": None,
            "source_sha256": None,
            "sigma_global": float(fallback_sigma),
        }
    source = Path(path).expanduser().resolve()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if source.suffix == ".json":
        with source.open() as stream:
            payload = json.load(stream)
    elif source.suffix in {".pt", ".pth"}:
        payload = torch.load(source, map_location="cpu", weights_only=False)
    elif source.suffix == ".npz":
        with np.load(source) as archive:
            payload = {key: archive[key] for key in archive.files}
    else:
        raise ValueError(
            "training normalizer must be a JSON, Torch checkpoint, or NPZ"
        )
    sigma = payload.get("sigma_global")
    if sigma is None and isinstance(payload.get("normalizer"), dict):
        sigma = payload["normalizer"].get("sigma_global")
    if sigma is None:
        raise ValueError("training normalizer artifact has no sigma_global")
    sigma = float(np.asarray(sigma))
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("training sigma_global must be finite and positive")
    return sigma, {
        "uses_training_trajectories": True,
        "source_path": str(source),
        "source_sha256": digest,
        "sigma_global": sigma,
    }


def audit_cn_hessian(
    sim: FVSolver2D,
    *,
    sigma: float,
    seed: int,
) -> dict[str, float | bool]:
    matrix, capacity = active_system(sim)
    hessian = (sp.diags(capacity) @ matrix).tocsr()
    asymmetry = hessian - hessian.T
    h_norm = float(spla.norm(hessian))
    relative_asymmetry = (
        0.0 if asymmetry.nnz == 0
        else float(spla.norm(asymmetry) / max(h_norm, np.finfo(float).tiny))
    )
    lambda_min = float(
        spla.eigsh(hessian, k=1, which="SA", return_eigenvectors=False)[0]
    )
    lambda_max = float(
        spla.eigsh(hessian, k=1, which="LA", return_eigenvectors=False)[0]
    )
    rng = np.random.default_rng(int(seed))
    z = rng.standard_normal(matrix.shape[0])
    full = np.zeros((sim.Nx, sim.Ny), dtype=np.float64)
    full[: sim.Nx - 1] = z.reshape(
        (sim.Nx - 1, sim.Ny), order="F",
    )
    zero = np.zeros_like(full)
    rhs = normalized_rhs(sim, full, 0.0, sigma=sigma)
    forcing_rhs = normalized_rhs(sim, zero, 0.0, sigma=sigma)
    rhs_linear = rhs - forcing_rhs
    rhs_matrix = (2.0 * sp.eye(matrix.shape[0], format="csr") - matrix)
    rhs_action_error = float(
        np.linalg.norm(rhs_linear - rhs_matrix @ z)
        / max(np.linalg.norm(rhs_linear), np.finfo(float).tiny)
    )

    matrix32 = scipy_to_torch_sparse(matrix.astype(np.float32)).to(
        dtype=torch.float32,
    )
    rhs32 = scipy_to_torch_sparse(rhs_matrix.astype(np.float32)).to(
        dtype=torch.float32,
    )
    z32 = torch.as_tensor(z, dtype=torch.float32)
    implicit32 = sparse_mv(matrix32, z32).double().numpy()
    explicit32 = sparse_mv(rhs32, z32).double().numpy()
    implicit_reference = matrix @ z
    explicit_reference = rhs_matrix @ z
    implicit_action_error = float(
        np.linalg.norm(implicit32 - implicit_reference)
        / max(np.linalg.norm(implicit_reference), np.finfo(float).tiny)
    )
    explicit_action_error = float(
        np.linalg.norm(explicit32 - explicit_reference)
        / max(np.linalg.norm(explicit_reference), np.finfo(float).tiny)
    )

    hessian32 = scipy_to_torch_sparse(hessian.astype(np.float32)).to(
        dtype=torch.float32,
    )
    rayleigh = []
    for _ in range(8):
        sample = torch.as_tensor(
            rng.standard_normal(matrix.shape[0]), dtype=torch.float32,
        )
        rayleigh.append(
            float(torch.dot(sample, sparse_mv(hessian32, sample)).item())
        )

    solved = spla.spsolve(matrix, rhs)
    physical = 300.0 + float(sigma) * full
    solver_step = (sim.cn_step(physical, 0.0) - 300.0) / float(sigma)
    step_reference = solver_step[: sim.Nx - 1].ravel(order="F")
    normalization_error = float(
        np.linalg.norm(solved - step_reference)
        / max(np.linalg.norm(step_reference), np.finfo(float).tiny)
    )

    q_exact = np.asarray(sim.q_left_integral(0.0, sim.dt), dtype=np.float64)
    t_fine = np.linspace(0.0, sim.dt, 4097, dtype=np.float64)
    q_values = np.stack(
        [np.asarray(sim.q_left(float(t)), dtype=np.float64) for t in t_fine],
        axis=0,
    )
    q_trap = np.trapezoid(q_values, t_fine, axis=0)
    quadrature_error = float(
        np.linalg.norm(q_exact - q_trap)
        / max(np.linalg.norm(q_exact), np.finfo(float).tiny)
    )
    passed = bool(
        relative_asymmetry <= 1.0e-12
        and lambda_min > 0.0
        and implicit_action_error <= 1.0e-5
        and explicit_action_error <= 1.0e-5
        and rhs_action_error <= 1.0e-12
        and normalization_error <= 1.0e-12
        and min(rayleigh) > 0.0
    )
    return {
        "relative_asymmetry": relative_asymmetry,
        "lambda_min": lambda_min,
        "lambda_max": lambda_max,
        "condition_estimate": lambda_max / lambda_min,
        "implicit_float32_action_relative_error": implicit_action_error,
        "rhs_float32_action_relative_error": explicit_action_error,
        "rhs_matrix_action_relative_error": rhs_action_error,
        "forcing_quadrature_relative_error": quadrature_error,
        "normalization_and_indexing_relative_error": normalization_error,
        "minimum_float32_rayleigh_quotient": min(rayleigh),
        "passed": passed,
    }


def _table_from_free(
    free: torch.Tensor,
    source: torch.Tensor,
) -> torch.Tensor:
    wall = free.new_zeros((free.shape[0], 1, free.shape[-1]))
    future = torch.cat((free, wall), dim=1)
    return torch.cat((source.unsqueeze(0), future), dim=0)


def _dense_table_loss(
    objective: str,
    table: torch.Tensor,
    cn: dict[str, torch.Tensor],
    step_indices: torch.Tensor,
    *,
    defect_sweeps: int,
    defect_omega: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    previous = table.index_select(0, step_indices)
    later = table.index_select(0, step_indices + 1)
    implicit = implicit_cn_action(later, cn)
    if objective == "raw_ls":
        state = previous
        nx = state.shape[-2]
        active = state[..., : nx - 1, :]
        rw, re, rs, rn = (cn[name] for name in ("r_w", "r_e", "r_s", "r_n"))
        rhs = (
            1.0
            - rw[..., : nx - 1, :]
            - re[..., : nx - 1, :]
            - rs[..., : nx - 1, :]
            - rn[..., : nx - 1, :]
        ) * active
        rhs[..., 1 : nx - 1, :] = rhs[..., 1 : nx - 1, :] + (
            rw[..., 1 : nx - 1, :] * state[..., : nx - 2, :]
        )
        rhs = rhs + re[..., : nx - 1, :] * state[..., 1:nx, :]
        rhs[..., 1:] = rhs[..., 1:] + (
            rs[..., : nx - 1, 1:] * active[..., :-1]
        )
        rhs[..., :-1] = rhs[..., :-1] + (
            rn[..., : nx - 1, :-1] * active[..., 1:]
        )
        rhs[..., 0, :] = rhs[..., 0, :] + cn["forcing"].index_select(
            0, step_indices,
        )
    else:
        rhs = explicit_cn_rhs(previous, step_indices, cn)
    residual = implicit - rhs
    if objective == "raw_ls":
        return residual.square().mean(), residual
    if objective == "variational":
        capacity = cn["capacity"]
        active_later = later[..., :-1, :]
        energy = (
            0.5 * (capacity * active_later * implicit).sum(dim=(-2, -1))
            - (capacity * active_later * rhs).sum(dim=(-2, -1))
        ) / capacity.sum()
        return energy.mean(), residual
    if objective == "defect":
        diagonal = cn_free_diagonal(cn, table.shape[-2]).clamp_min(1.0e-12)
        if int(defect_sweeps) <= 1:
            return (residual / diagonal.sqrt()).square().mean(), residual
        correction = torch.zeros_like(residual)
        for _ in range(int(defect_sweeps)):
            wall = correction.new_zeros(
                correction.shape[:-2] + (1, correction.shape[-1]),
            )
            action = implicit_cn_action(
                torch.cat((correction, wall), dim=-2), cn,
            )
            correction = correction + float(defect_omega) * (
                residual - action
            ) / diagonal
        return (diagonal * correction.square()).mean(), residual
    raise ValueError(f"unknown objective {objective!r}")


def optimize_dense_table(
    sim: FVSolver2D,
    source: np.ndarray,
    *,
    sigma: float,
    objective: str,
    optimizer_name: str,
    updates: int,
    learning_rate: float,
    lbfgs_iterations: int,
    defect_sweeps: int,
    defect_omega: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, float]]:
    torch.manual_seed(int(seed))
    cn = build_cn_tensors(
        sim, sigma=sigma, device=torch.device("cpu"), dtype=torch.float64,
    )
    source_t = torch.as_tensor(source, dtype=torch.float64)
    initial = source_t[: sim.Nx - 1].unsqueeze(0).expand(
        len(sim.t) - 1, -1, -1,
    ).clone()
    free = torch.nn.Parameter(initial)
    closure_calls = 0
    start = time.perf_counter()

    def current_loss(max_lead: float | None = None) -> torch.Tensor:
        table = _table_from_free(free, source_t)
        if max_lead is None:
            count = len(sim.t) - 1
        else:
            count = max(
                1, min(len(sim.t) - 1, int(np.floor(
                    (float(max_lead) + 0.5 * sim.dt) / sim.dt,
                ))),
            )
        steps = torch.arange(count, dtype=torch.long)
        loss, _ = _dense_table_loss(
            objective,
            table,
            cn,
            steps,
            defect_sweeps=defect_sweeps,
            defect_omega=defect_omega,
        )
        return loss

    initial_loss = float(current_loss().detach())
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam([free], lr=float(learning_rate))
        for update in range(int(updates)):
            optimizer.zero_grad(set_to_none=True)
            max_lead = preregistered_max_lead(
                update, updates, float(sim.t[-1]),
            )
            loss = current_loss(max_lead)
            loss.backward()
            optimizer.step()
            closure_calls += 1
    elif optimizer_name == "lbfgs":
        if objective != "raw_ls":
            raise ValueError("joint LBFGS oracle is valid only for raw_ls")
        optimizer = torch.optim.LBFGS(
            [free],
            lr=1.0,
            max_iter=int(lbfgs_iterations),
            max_eval=max(int(lbfgs_iterations) * 5 // 4, int(lbfgs_iterations) + 1),
            tolerance_grad=1.0e-13,
            tolerance_change=1.0e-15,
            history_size=100,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            nonlocal closure_calls
            optimizer.zero_grad(set_to_none=True)
            loss = current_loss()
            loss.backward()
            closure_calls += 1
            return loss

        optimizer.step(closure)
    else:
        raise ValueError(f"unknown optimizer {optimizer_name!r}")

    table = _table_from_free(free, source_t).detach().numpy()
    return table, {
        "initial_objective": initial_loss,
        "final_objective": float(current_loss().detach()),
        "closure_calls": float(closure_calls),
        "wall_seconds": float(time.perf_counter() - start),
    }


def optimize_sparse_table(
    sim: FVSolver2D,
    source: np.ndarray,
    *,
    sigma: float,
    objective: str,
    updates: int,
    learning_rate: float,
    intervals_per_update: int,
    defect_sweeps: int,
    defect_omega: float,
    seed: int,
) -> tuple[np.ndarray, dict]:
    if intervals_per_update <= 0:
        raise ValueError("intervals_per_update must be positive")
    rng = np.random.default_rng(int(seed))
    torch.manual_seed(int(seed))
    cn = build_cn_tensors(
        sim, sigma=sigma, device=torch.device("cpu"), dtype=torch.float64,
    )
    source_t = torch.as_tensor(source, dtype=torch.float64)
    free = torch.nn.Parameter(
        source_t[: sim.Nx - 1].unsqueeze(0).expand(
            len(sim.t) - 1, -1, -1,
        ).clone(),
    )
    optimizer = torch.optim.Adam([free], lr=float(learning_rate))
    visits = np.zeros(len(sim.t) - 1, dtype=np.int64)
    first_visit = np.full(len(sim.t) - 1, -1, dtype=np.int64)
    for update in range(int(updates)):
        max_lead = preregistered_max_lead(update, updates, float(sim.t[-1]))
        count = max(
            1, min(len(visits), int(np.floor(
                (max_lead + 0.5 * sim.dt) / sim.dt,
            ))),
        )
        size = min(int(intervals_per_update), count)
        chosen = np.sort(rng.choice(count, size=size, replace=False))
        visits[chosen] += 1
        first_visit[chosen[first_visit[chosen] < 0]] = update
        optimizer.zero_grad(set_to_none=True)
        table = _table_from_free(free, source_t)
        loss, _ = _dense_table_loss(
            objective,
            table,
            cn,
            torch.as_tensor(chosen, dtype=torch.long),
            defect_sweeps=defect_sweeps,
            defect_omega=defect_omega,
        )
        loss.backward()
        optimizer.step()
    return _table_from_free(free, source_t).detach().numpy(), {
        "visits": visits.tolist(),
        "first_visit_update": first_visit.tolist(),
        "all_intervals_visited": bool(np.all(visits > 0)),
        "updates": int(updates),
        "intervals_per_update": int(intervals_per_update),
    }


def sequential_exact_trajectory(
    sim: FVSolver2D,
    source: np.ndarray,
    *,
    sigma: float,
) -> np.ndarray:
    matrix, _ = active_system(sim)
    factor = spla.splu(matrix.tocsc())
    trajectory = np.zeros(
        (len(sim.t), sim.Nx, sim.Ny), dtype=np.float64,
    )
    trajectory[0] = source
    for step in range(len(sim.t) - 1):
        rhs = normalized_rhs(
            sim, trajectory[step], float(sim.t[step]), sigma=sigma,
        )
        trajectory[step + 1, : sim.Nx - 1] = factor.solve(rhs).reshape(
            (sim.Nx - 1, sim.Ny), order="F",
        )
    return trajectory


def _lowest_mode_error(error: np.ndarray) -> float:
    transformed = np.fft.fft2(error, axes=(-2, -1), norm="ortho")
    low = transformed[..., :4, :4]
    return float(np.sqrt(np.mean(np.abs(low) ** 2)))


def trajectory_gate_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    sim: FVSolver2D,
    *,
    sigma: float,
) -> dict[str, float]:
    error_K = float(sigma) * (
        np.asarray(prediction, dtype=np.float64)
        - np.asarray(truth, dtype=np.float64)
    )
    feasible_bins = (
        (0.0, 0.05),
        (0.05, 0.10),
        (0.10, 0.20),
        (0.20, 0.30),
    )
    populated = [
        (lo, min(hi, float(sim.t[-1])))
        for lo, hi in feasible_bins
        if np.any((sim.t > lo + 1.0e-12) & (sim.t <= hi + 1.0e-12))
    ]
    longest_lo, longest_hi = populated[-1]
    longest = (
        (sim.t > longest_lo + 1.0e-12)
        & (sim.t <= longest_hi + 1.0e-12)
    )
    matrix, _ = active_system(sim)
    defects = []
    for step in range(len(sim.t) - 1):
        later = prediction[step + 1, : sim.Nx - 1].ravel(order="F")
        rhs = normalized_rhs(
            sim, prediction[step], float(sim.t[step]), sigma=sigma,
        )
        defects.append(matrix @ later - rhs)
    defect = np.concatenate(defects)
    mean_bias = error_K.mean(axis=(-2, -1))
    return {
        "overall_field": float(np.sqrt(np.mean(error_K[1:] ** 2))),
        "longest_lead": float(np.sqrt(np.mean(error_K[longest] ** 2))),
        "physical_defect": float(np.sqrt(np.mean(defect ** 2))),
        "spatial_mean_bias": float(np.sqrt(np.mean(mean_bias[1:] ** 2))),
        "lowest_4x4_modes": _lowest_mode_error(error_K[1:]),
    }


def run_homogeneous_gate(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    grid_size = int(args.grid_size)
    registered_updates = 2_000 if grid_size == 20 else 10_000
    if grid_size not in {20, 100} and not args.smoke:
        raise ValueError("registered dense gate grids are 20 and 100")
    adam_updates = (
        int(args.gate_adam_updates)
        if args.gate_adam_updates is not None
        else registered_updates
    )
    if not args.smoke and adam_updates != registered_updates:
        raise ValueError(
            f"grid {grid_size} requires the registered {registered_updates} updates"
        )
    lbfgs_iterations = (
        int(args.gate_lbfgs_iterations)
        if args.gate_lbfgs_iterations is not None
        else 2_000
    )
    objectives = tuple(args.gate_objectives or OBJECTIVES)
    unknown = sorted(set(objectives) - set(OBJECTIVES))
    if unknown:
        raise ValueError(f"unknown gate objectives {unknown}")
    sigma, normalizer = _resolve_gate_normalizer(
        args.training_normalizer_path,
        float(args.sigma),
        smoke=bool(args.smoke),
    )

    cases = []
    for case_index, (family, source_time) in enumerate(HOMOGENEOUS_GATE_CASES):
        sim, truth, metadata = build_homogeneous_gate_case(
            family=family,
            source_time=source_time,
            grid_size=grid_size,
            dt=float(args.dt),
            sigma=sigma,
            seed=int(args.seed) + case_index,
        )
        audit = audit_cn_hessian(
            sim, sigma=sigma, seed=int(args.seed) + case_index,
        )
        floor = sequential_exact_trajectory(
            sim, truth[0], sigma=sigma,
        )
        copy_trajectory = np.broadcast_to(
            truth[0], truth.shape,
        ).copy()
        floor_metrics = trajectory_gate_metrics(
            floor, truth, sim, sigma=sigma,
        )
        copy_metrics = trajectory_gate_metrics(
            copy_trajectory, truth, sim, sigma=sigma,
        )
        objective_rows = []
        for objective_index, objective in enumerate(objectives):
            if objective == "raw_ls":
                oracle, oracle_info = optimize_dense_table(
                    sim,
                    truth[0],
                    sigma=sigma,
                    objective=objective,
                    optimizer_name="lbfgs",
                    updates=0,
                    learning_rate=0.01,
                    lbfgs_iterations=lbfgs_iterations,
                    defect_sweeps=int(args.defect_sweeps),
                    defect_omega=float(args.defect_omega),
                    seed=int(args.seed) + objective_index,
                )
                oracle_kind = "joint_lbfgs"
            else:
                oracle = floor.copy()
                oracle_info = {
                    "initial_objective": float("nan"),
                    "final_objective": 0.0,
                    "closure_calls": float(len(sim.t) - 1),
                    "wall_seconds": 0.0,
                }
                oracle_kind = "causal_sequential_exact_solves"
            oracle_metrics = trajectory_gate_metrics(
                oracle, truth, sim, sigma=sigma,
            )
            adam, adam_info = optimize_dense_table(
                sim,
                truth[0],
                sigma=sigma,
                objective=objective,
                optimizer_name="adam",
                updates=adam_updates,
                learning_rate=0.01,
                lbfgs_iterations=0,
                defect_sweeps=int(args.defect_sweeps),
                defect_omega=float(args.defect_omega),
                seed=int(args.seed) + objective_index,
            )
            adam_metrics = trajectory_gate_metrics(
                adam, truth, sim, sigma=sigma,
            )
            metric_decisions = {}
            for name in GATE_METRICS:
                decision = calibrated_gate_thresholds(
                    copy_metrics[name],
                    floor_metrics[name],
                    oracle_metrics[name],
                )
                decision["copy_error"] = copy_metrics[name]
                decision["floor_error"] = floor_metrics[name]
                decision["oracle_error"] = oracle_metrics[name]
                decision["adam_error"] = adam_metrics[name]
                decision["adam_qualified"] = bool(
                    decision["valid"]
                    and decision["oracle_qualified"]
                    and adam_metrics[name] <= decision["adam_threshold"]
                )
                metric_decisions[name] = decision
            qualified = bool(
                audit["passed"]
                and all(
                    bool(decision["adam_qualified"])
                    for decision in metric_decisions.values()
                )
            )
            row = {
                "objective": objective,
                "oracle_kind": oracle_kind,
                "oracle_info": oracle_info,
                "adam_info": adam_info,
                "metrics": metric_decisions,
                "qualified": qualified,
            }
            if args.run_sparse_diagnostic:
                sparse, sparse_info = optimize_sparse_table(
                    sim,
                    truth[0],
                    sigma=sigma,
                    objective=objective,
                    updates=adam_updates,
                    learning_rate=0.01,
                    intervals_per_update=int(args.sparse_intervals_per_update),
                    defect_sweeps=int(args.defect_sweeps),
                    defect_omega=float(args.defect_omega),
                    seed=int(args.seed) + objective_index,
                )
                row["sparse_diagnostic"] = {
                    **sparse_info,
                    "metrics": trajectory_gate_metrics(
                        sparse, truth, sim, sigma=sigma,
                    ),
                    "is_gate": False,
                    "claim_scope": "bootstrap_speed_and_dense_sparse_degradation",
                }
            objective_rows.append(row)
            print(
                json.dumps(
                    {
                        "case": family,
                        "source_time": source_time,
                        "objective": objective,
                        "qualified": qualified,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        np.savez_compressed(
            output_dir / f"{case_index:02d}_{family}_references.npz",
            truth=truth,
            floor=floor,
            copy=copy_trajectory,
        )
        cases.append({
            "case": metadata,
            "audit": audit,
            "floor_metrics": floor_metrics,
            "copy_metrics": copy_metrics,
            "objectives": objective_rows,
        })
    qualified_objectives = [
        objective
        for objective in objectives
        if all(
            next(
                row for row in case["objectives"]
                if row["objective"] == objective
            )["qualified"]
            for case in cases
        )
    ]
    all_passed = bool(qualified_objectives)
    summary = {
        "schema_version": 1,
        "stage": "dense_direct_state_gate",
        "benchmark": "diffusion_forcing_single",
        "coordinate_convention": (
            "intervals use source-relative lead tau; forcing and CN closures "
            "use absolute time source_time + tau"
        ),
        "configuration": {
            **vars(args),
            "gate_objectives": list(objectives),
            "registered_adam_updates": registered_updates,
            "resolved_adam_updates": adam_updates,
            "resolved_lbfgs_iterations": lbfgs_iterations,
            "dtype": "float64",
            "learning_rate": 0.01,
            "scheduler": None,
            "floor_separation": 1.0e3,
            "resolved_sigma_global": sigma,
            "normalizer": normalizer,
        },
        "cases": cases,
        "qualified_objectives": qualified_objectives,
        "passed": bool(all_passed),
        "next_stage_authorized": bool(all_passed and not args.smoke),
        "optimization_uses_solution_fields": False,
        "normalization_uses_training_trajectories": bool(
            normalizer["uses_training_trajectories"]
        ),
        "checkpoint_selection_uses_validation_targets": True,
        "direct_state_gate_uses_fv_reference_solutions": True,
        "network_gate_uses_supervised_capacity_baseline": True,
        "physics_only_claim_scope": "optimization_objective",
    }
    canonical = json.dumps(summary, sort_keys=True, separators=(",", ":"))
    summary["gate_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    with (output_dir / "gate_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary


def run_matrix(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    sim = build_problem(
        grid_size=args.grid_size,
        dt=args.dt,
        t_final=args.t_final,
        flux_amplitude=args.flux_amplitude,
        flux_frequency=args.flux_frequency,
        interface_x=args.interface_x,
        resistance=args.resistance,
    )
    _, _, _, truth_K = sim.solve(
        T0=np.full((sim.Nx, sim.Ny), 300.0, dtype=np.float64),
        store_trajectory=True,
    )
    truth = (truth_K - 300.0) / args.sigma
    matrix, capacity = active_system(sim)
    audit = symmetry_audit(matrix, capacity)
    if audit["capacity_weighted_max_asymmetry"] > 1.0e-12:
        raise RuntimeError("capacity-weighted active CN operator is not symmetric")
    if audit["capacity_weighted_lambda_min"] <= 0.0:
        raise RuntimeError("capacity-weighted active CN operator is not positive definite")

    rows = []
    trajectories = {"truth": truth}
    for optimizer_name in OPTIMIZERS:
        for objective in OBJECTIVES:
            row, prediction = rollout_cell(
                sim,
                matrix,
                capacity,
                truth,
                objective=objective,
                optimizer_name=optimizer_name,
                sigma=args.sigma,
                adam_updates=args.adam_updates,
                adam_lr=args.adam_lr,
                lbfgs_iterations=args.lbfgs_iterations,
                defect_sweeps=args.defect_sweeps,
                defect_omega=args.defect_omega,
            )
            rows.append(row)
            trajectories[f"{optimizer_name}_{objective}"] = prediction
            print(json.dumps(row, sort_keys=True), flush=True)

    with (output_dir / "matrix.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(output_dir / "trajectories_normalized.npz", **trajectories)
    summary = {
        "configuration": vars(args),
        "audit": audit,
        "rows": rows,
        "interpretation": {
            "raw_ls": "legacy region-balanced squared CN residual",
            "variational": "capacity-weighted SPD CN functional with fixed previous state",
            "defect": f"state correction from {args.defect_sweeps} weighted-Jacobi sweeps",
        },
    }
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--homogeneous-gate", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--gate-objectives", nargs="+", choices=OBJECTIVES, default=None,
    )
    parser.add_argument("--gate-adam-updates", type=int, default=None)
    parser.add_argument("--gate-lbfgs-iterations", type=int, default=None)
    parser.add_argument("--run-sparse-diagnostic", action="store_true")
    parser.add_argument("--sparse-intervals-per-update", type=int, default=4)
    parser.add_argument("--grid-size", type=int, default=20)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--t-final", type=float, default=0.08)
    parser.add_argument("--flux-amplitude", type=float, default=300.0)
    parser.add_argument("--flux-frequency", type=float, default=2.0)
    parser.add_argument("--interface-x", type=float, default=0.5)
    parser.add_argument("--resistance", type=float, default=0.5)
    parser.add_argument("--sigma", type=float, default=10.0)
    parser.add_argument("--training-normalizer-path", default=None)
    parser.add_argument("--adam-updates", type=int, default=500)
    parser.add_argument("--adam-lr", type=float, default=0.01)
    parser.add_argument("--lbfgs-iterations", type=int, default=100)
    parser.add_argument("--defect-sweeps", type=int, default=16)
    parser.add_argument("--defect-omega", type=float, default=2.0 / 3.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.homogeneous_gate:
        run_homogeneous_gate(parsed)
    else:
        run_matrix(parsed)
