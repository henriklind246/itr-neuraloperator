#!/usr/bin/env python3
"""Causal direct-state reachability matrix for the fully discrete FV objective."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from src.physics.fv_solver_2d import FVSolver2D, Layer2D


OBJECTIVES = ("raw_ls", "variational", "defect")
OPTIMIZERS = ("adam", "lbfgs")


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
    parser.add_argument("--grid-size", type=int, default=20)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--t-final", type=float, default=0.08)
    parser.add_argument("--flux-amplitude", type=float, default=300.0)
    parser.add_argument("--flux-frequency", type=float, default=2.0)
    parser.add_argument("--interface-x", type=float, default=0.5)
    parser.add_argument("--resistance", type=float, default=0.5)
    parser.add_argument("--sigma", type=float, default=10.0)
    parser.add_argument("--adam-updates", type=int, default=500)
    parser.add_argument("--adam-lr", type=float, default=0.01)
    parser.add_argument("--lbfgs-iterations", type=int, default=100)
    parser.add_argument("--defect-sweeps", type=int, default=16)
    parser.add_argument("--defect-omega", type=float, default=2.0 / 3.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    run_matrix(parse_args())
