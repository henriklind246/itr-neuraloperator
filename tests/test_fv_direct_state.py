import argparse

import numpy as np
import scipy.sparse.linalg as spla
import torch

from scripts.diagnose_fv_direct_state import (
    active_system,
    approximate_defect,
    build_problem,
    normalized_rhs,
    run_matrix,
    scipy_to_torch_sparse,
    sparse_mv,
    symmetry_audit,
    variational_loss,
)


def _problem(grid_size=10, t_final=0.02):
    return build_problem(
        grid_size=grid_size,
        dt=0.005,
        t_final=t_final,
        flux_amplitude=300.0,
        flux_frequency=2.0,
        interface_x=0.5,
        resistance=0.5,
    )


def test_capacity_weighted_active_operator_is_spd():
    sim = _problem()
    matrix, capacity = active_system(sim)
    audit = symmetry_audit(matrix, capacity)
    assert audit["capacity_weighted_max_asymmetry"] < 1.0e-14
    assert audit["capacity_weighted_lambda_min"] > 0.0


def test_normalized_rhs_matches_solver_step():
    sim = _problem()
    matrix, _ = active_system(sim)
    sigma = 11.0
    rng = np.random.default_rng(4)
    z_n = rng.normal(scale=0.2, size=(sim.Nx, sim.Ny))
    z_n[-1] = 0.0
    rhs = normalized_rhs(sim, z_n, float(sim.t[1]), sigma=sigma)
    solved = spla.spsolve(matrix, rhs).reshape(
        (sim.Nx - 1, sim.Ny), order="F"
    )
    expected = (sim.cn_step(300.0 + sigma * z_n, float(sim.t[1])) - 300.0) / sigma
    np.testing.assert_allclose(solved, expected[: sim.Nx - 1], rtol=0.0, atol=2e-12)
    np.testing.assert_allclose(expected[-1], 0.0, rtol=0.0, atol=2e-12)


def test_variational_gradient_is_capacity_weighted_cn_residual():
    sim = _problem()
    matrix_np, capacity_np = active_system(sim)
    matrix = scipy_to_torch_sparse(matrix_np)
    capacity = torch.as_tensor(capacity_np, dtype=torch.float64)
    rng = np.random.default_rng(5)
    z_n = rng.normal(scale=0.1, size=(sim.Nx, sim.Ny))
    z_n[-1] = 0.0
    rhs = torch.as_tensor(
        normalized_rhs(sim, z_n, float(sim.t[0]), sigma=10.0),
        dtype=torch.float64,
    )
    z = torch.randn(matrix.shape[0], dtype=torch.float64, requires_grad=True)
    loss = variational_loss(z, matrix, rhs, capacity)
    gradient = torch.autograd.grad(loss, z)[0]
    expected = capacity * (sparse_mv(matrix, z) - rhs) / capacity.sum()
    torch.testing.assert_close(gradient, expected, rtol=1e-11, atol=1e-12)


def test_jacobi_defect_is_zero_at_solution_and_reduces_residual():
    sim = _problem()
    matrix_np, _ = active_system(sim)
    matrix = scipy_to_torch_sparse(matrix_np)
    diagonal = torch.as_tensor(matrix_np.diagonal(), dtype=torch.float64)
    z_n = np.zeros((sim.Nx, sim.Ny), dtype=np.float64)
    rhs_np = normalized_rhs(sim, z_n, float(sim.t[1]), sigma=10.0)
    rhs = torch.as_tensor(rhs_np, dtype=torch.float64)
    exact = torch.as_tensor(spla.spsolve(matrix_np, rhs_np), dtype=torch.float64)
    exact_residual = sparse_mv(matrix, exact) - rhs
    exact_defect = approximate_defect(
        exact_residual, matrix, diagonal, sweeps=8, omega=2.0 / 3.0
    )
    assert float(exact_defect.abs().max()) < 1e-12

    trial = torch.zeros_like(rhs)
    residual = sparse_mv(matrix, trial) - rhs
    defect = approximate_defect(
        residual, matrix, diagonal, sweeps=8, omega=2.0 / 3.0
    )
    corrected_residual = sparse_mv(matrix, trial - defect) - rhs
    assert float(corrected_residual.norm()) < float(residual.norm())


def test_direct_state_matrix_smoke(tmp_path):
    output_dir = tmp_path / "matrix"
    args = argparse.Namespace(
        output_dir=str(output_dir),
        grid_size=8,
        dt=0.005,
        t_final=0.01,
        flux_amplitude=300.0,
        flux_frequency=2.0,
        interface_x=0.5,
        resistance=0.5,
        sigma=10.0,
        adam_updates=2,
        adam_lr=0.01,
        lbfgs_iterations=2,
        defect_sweeps=2,
        defect_omega=2.0 / 3.0,
        seed=0,
    )
    summary = run_matrix(args)
    assert len(summary["rows"]) == 6
    assert (output_dir / "matrix.csv").is_file()
    assert (output_dir / "summary.json").is_file()
    assert (output_dir / "trajectories_normalized.npz").is_file()
