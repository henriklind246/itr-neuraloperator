import numpy as np
import pytest
import torch

from src.physics.fv_preconditioner import ExactCNInverse, MGOneCycleInverse, assemble_active_cn_matrix
from src.physics.fv_residual import (
    FullBCData, build_cn_geom, build_homogeneous_cn_geom, cn_step_residual,
    explicit_cn_rhs, full_bc_cn_residual, implicit_cn_action,
)
from src.physics.fv_solver_2d import FVSolver2D, Layer2D


def solver_and_geom(n=7, *, integral=True, layered=False, sigma=7.25):
    q = lambda t: 13.0 + 9.0 * t * t
    qint = lambda lo, hi: 13.0 * (hi - lo) + 3.0 * (hi ** 3 - lo ** 3)
    layers = (
        [Layer2D(0, 0.5, 1, 1, 2), Layer2D(0.5, 1, 1, 1, 1)]
        if layered else [Layer2D(0, 1, 1, 1, 1)]
    )
    sim = FVSolver2D(
        a=0, b=1, c=0, d=1, Nx=n, Ny=n, layers=layers,
        interface_R=[0.3] if layered else None,
        lam_target=0.5, dt=0.005, t_final=0.3,
        flux_f=0, flux_A=0, t_on=0, t_off=0.2, phase=0,
        q_left_fn=q, q_left_integral_fn=qint if integral else None,
    )
    if layered:
        geom = build_cn_geom(sim.grid_x, sim.grid_y, 2, 1, 0.5, 0.3, sim.dt,
                             sigma_global=sigma)
    else:
        geom = build_homogeneous_cn_geom(sim.grid_x, sim.grid_y, 1, sim.dt,
                                         sigma_global=sigma)
    return sim, geom


def boundary(sim, *, mu=296.0, sigma=7.25, time=0.075):
    return FullBCData(
        T_right_tilde=torch.tensor((300 - mu) / sigma, dtype=torch.float64),
        qL_n=torch.full((1, sim.Ny), sim.q_left(time), dtype=torch.float64),
        qL_np1=torch.full((1, sim.Ny), sim.q_left(time + sim.dt), dtype=torch.float64),
        qL_int=(None if sim.q_left_integral is None else torch.full(
            (1, sim.Ny), sim.q_left_integral(time, time + sim.dt), dtype=torch.float64)),
    )


@pytest.mark.parametrize("layered,n", [(False, 7), (True, 8)])
def test_active_matrix_and_cn_actions_match_solver(layered, n):
    sim, geom = solver_and_geom(n, layered=layered)
    matrix = assemble_active_cn_matrix(geom)
    active = np.arange(n * n).reshape(n, n, order="F")[:-1].ravel(order="F")
    np.testing.assert_allclose(matrix.toarray(), sim.A_csr[active][:, active].toarray(),
                               atol=1e-12, rtol=0)
    field = torch.randn((2, n, n), generator=torch.Generator().manual_seed(7),
                         dtype=torch.float64)
    field[:, -1] = 0
    action = implicit_cn_action(field, geom)
    for b in range(2):
        expected = matrix @ field[b, :-1].numpy().ravel(order="F")
        np.testing.assert_allclose(action[b].numpy().ravel(order="F"), expected,
                                   atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("integral", [False, True])
def test_normalized_rhs_and_complete_residual_match_solver_and_regions(integral):
    sim, geom = solver_and_geom(integral=integral)
    bc = boundary(sim)
    mu, sigma = 296.0, geom.sigma_global
    field = torch.randn((2, sim.Nx, sim.Ny), dtype=torch.float64) * 5 + 300
    field[:, -1] = 300
    normalized = (field - mu) / sigma
    rhs = explicit_cn_rhs(normalized - bc.T_right_tilde, geom, bc)
    inverse = ExactCNInverse(geom)
    next_free = inverse(rhs) + bc.T_right_tilde
    for b in range(2):
        expected = (sim.cn_step(field[b].numpy(), 0.075) - mu) / sigma
        np.testing.assert_allclose(next_free[b].numpy(), expected[:-1], atol=1e-12)
    next_field = normalized + torch.randn_like(normalized)
    next_field[:, -1] = bc.T_right_tilde
    residual = cn_step_residual(normalized, next_field, geom, bc)
    regions = full_bc_cn_residual(normalized, next_field, geom, bc, keep_batch=True)
    torch.testing.assert_close(residual[:, 0], regions["left_neumann"])
    torch.testing.assert_close(residual[:, 1:, 1:-1], regions["interior"])
    torch.testing.assert_close(torch.cat((residual[:, 1:, 0], residual[:, 1:, -1]), dim=1),
                               regions["topbot_adiabatic"])
    correct = torch.cat((next_free, normalized[:, -1:]), dim=1)
    torch.testing.assert_close(cn_step_residual(normalized, correct, geom, bc),
                               torch.zeros_like(residual), atol=1e-12, rtol=0)


def test_integral_closure_is_not_replaced_by_endpoint_trapezoid():
    sim, geom = solver_and_geom()
    bc = boundary(sim)
    bc.qL_n.zero_()
    bc.qL_np1.zero_()
    dev = torch.zeros((1, sim.Nx, sim.Ny), dtype=torch.float64)
    rhs = explicit_cn_rhs(dev, geom, bc)
    torch.testing.assert_close(rhs[:, 0], 2 * bc.qL_int / (geom.hx * geom.sigma_global))
    assert rhs[:, 0].abs().min() > 0


def test_exact_identity_values_gradients_and_detached_source():
    sim, geom = solver_and_geom()
    bc = boundary(sim)
    inverse = ExactCNInverse(geom)
    previous = torch.randn((2, sim.Nx, sim.Ny), dtype=torch.float64, requires_grad=True)
    next_free = torch.randn((2, sim.Nx - 1, sim.Ny), dtype=torch.float64, requires_grad=True)
    wall = torch.full((2, 1, sim.Ny), bc.T_right_tilde.item(), dtype=torch.float64)
    prediction = torch.cat((next_free, wall), dim=1)
    rhs = explicit_cn_rhs(previous - bc.T_right_tilde, geom, bc)
    correction = inverse(cn_step_residual(previous, prediction, geom, bc))
    target_error = next_free - bc.T_right_tilde - inverse(rhs)
    torch.testing.assert_close(correction, target_error, atol=1e-12, rtol=1e-12)
    actual = torch.autograd.grad(correction.square().mean(), (next_free, previous),
                                 allow_unused=True, retain_graph=True)
    expected = torch.autograd.grad(target_error.square().mean(), next_free)[0]
    torch.testing.assert_close(actual[0], expected, atol=1e-12, rtol=1e-12)
    assert actual[1] is None


def test_sparse_inverse_transpose_and_higher_derivative():
    _, geom = solver_and_geom(4)
    inverse = ExactCNInverse(geom)
    residual = torch.randn((2, 3, 4), dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(inverse, (residual,))
    assert torch.autograd.gradgradcheck(inverse, (residual,))
    matrix = torch.tensor(inverse.matrix.toarray(), dtype=torch.float64)
    assert not torch.allclose(matrix, matrix.T)
    weights = torch.randn_like(residual)
    actual = torch.autograd.grad((inverse(residual) * weights).sum(), residual)[0]
    expected = torch.linalg.solve(matrix.T, weights.transpose(-2, -1).reshape(2, -1).T)
    expected = expected.T.reshape(2, 4, 3).transpose(-2, -1)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


def test_sparse_inverse_handles_noncontiguous_fields_and_float32():
    _, geom = solver_and_geom(5)
    inverse = ExactCNInverse(geom)
    residual = torch.randn((2, 5, 4)).transpose(-2, -1).requires_grad_()
    actual = inverse(residual)
    expected = inverse(residual.double()).float()
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    assert residual.grad is not None and torch.isfinite(residual.grad).all()


def test_equilibrium_and_first_step_hessian():
    sim, geom = solver_and_geom(4)
    inverse = ExactCNInverse(geom)
    bc = boundary(sim)
    bc.qL_n.zero_()
    bc.qL_np1.zero_()
    bc.qL_int.zero_()
    wall = torch.full((1, 1, sim.Ny), bc.T_right_tilde.item(), dtype=torch.float64)
    previous = wall.expand(1, sim.Nx, sim.Ny)
    flat = previous[:, :-1].clone().reshape(-1)

    def loss(values):
        field = torch.cat((values.reshape(1, sim.Nx - 1, sim.Ny), wall), dim=1)
        return inverse(cn_step_residual(previous, field, geom, bc)).square().mean()

    assert loss(flat) == 0
    hessian = torch.autograd.functional.hessian(loss, flat)
    torch.testing.assert_close(hessian, 2 * torch.eye(flat.numel(), dtype=flat.dtype)
                               / flat.numel(), atol=1e-12, rtol=1e-12)


def test_fixed_inverse_rejects_batched_geometry():
    _, geom = solver_and_geom()
    geom.r_w = geom.r_w.unsqueeze(0)
    with pytest.raises(ValueError, match="batch-independent"):
        ExactCNInverse(geom)


@pytest.mark.parametrize('layered,n', [(False,7), (True,8)])
@pytest.mark.parametrize('integral', [False,True])
def test_space_time_inverse_recovers_trajectory_error_and_mse_gradient(layered,n,integral):
    sim, geom = solver_and_geom(n, layered=layered, integral=integral)
    inverse = ExactCNInverse(geom)
    rng = np.random.default_rng(19)
    initial = 300 + rng.normal(size=(n,n))
    initial[-1] = 300
    reference = [initial]
    for step in range(5):
        reference.append(sim.cn_step(reference[-1], step*sim.dt))
    reference = torch.tensor((np.stack(reference)-296)/geom.sigma_global,dtype=torch.float64)
    errors = (torch.randn(2,5,n-1,n,generator=torch.Generator().manual_seed(31),
                          dtype=torch.float64)*.2).requires_grad_()
    predicted = reference[None,1:,:-1] + errors
    wall = reference[None,1:,-1:].expand(2,-1,-1,-1)
    states = torch.cat((reference[None,:1].expand(2,-1,-1,-1),
                        torch.cat((predicted,wall),dim=-2)),dim=1)
    residuals = torch.stack([cn_step_residual(states[:,step],states[:,step+1],geom,
                    boundary(sim,time=step*sim.dt),detach_previous=False) for step in range(5)],dim=1)
    corrections = inverse.space_time(residuals)
    torch.testing.assert_close(corrections,errors,rtol=1e-12,atol=1e-12)
    actual = torch.autograd.grad(.5*corrections.square().sum(),errors,retain_graph=True)[0]
    torch.testing.assert_close(actual,errors,rtol=1e-12,atol=1e-12)
    assert not torch.allclose(inverse(residuals)[:,1:],errors[:,1:],rtol=1e-3,atol=1e-3)


def test_space_time_sparse_inverse_transpose_and_higher_derivative():
    _,geom = solver_and_geom(4)
    inverse = ExactCNInverse(geom)
    residual = torch.randn(1,3,3,4,dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(inverse.space_time,(residual,))
    assert torch.autograd.gradgradcheck(inverse.space_time,(residual,))
    matrix = torch.tensor(inverse.matrix.toarray(),dtype=torch.float64)
    explicit = torch.tensor(inverse.explicit_matrix.toarray(),dtype=torch.float64)
    block = torch.kron(torch.eye(3,dtype=torch.float64),matrix)
    block -= torch.kron(torch.diag(torch.ones(2,dtype=torch.float64),diagonal=-1),explicit)
    weights = torch.randn_like(residual)
    gradient = torch.autograd.grad((inverse.space_time(residual)*weights).sum(),residual)[0]
    packed = weights.transpose(-2,-1).reshape(-1)
    expected = torch.linalg.solve(block.T,packed).reshape(1,3,4,3).transpose(-2,-1)
    torch.testing.assert_close(gradient,expected,rtol=1e-12,atol=1e-12)


def test_space_time_inverse_requires_a_nonempty_time_axis():
    _,geom = solver_and_geom(4)
    inverse = ExactCNInverse(geom)
    with pytest.raises(ValueError,match='time dimension'):
        inverse.space_time(torch.zeros(1,0,3,4))


def test_mg_hierarchy_actual_adjoint_and_second_derivative():
    _, geom = solver_and_geom(7)
    mg = MGOneCycleInverse(geom, coarse_max_unknowns=4)
    n = 42
    basis = torch.eye(n, dtype=torch.float64).reshape(n, 6, 7)
    matrix = mg(basis).reshape(n, n).T
    symmetric_cycle = matrix / mg.capacity.flatten()[None]
    torch.testing.assert_close(symmetric_cycle, symmetric_cycle.T, rtol=1e-12, atol=1e-12)
    assert torch.linalg.eigvalsh(symmetric_cycle).min() > 0
    assert not torch.allclose(matrix, matrix.T)
    residual = torch.randn(1, 6, 7, dtype=torch.float64, requires_grad=True)
    weights = torch.randn_like(residual)
    gradient = torch.autograd.grad((mg(residual) * weights).sum(), residual)[0]
    torch.testing.assert_close(gradient.flatten(), matrix.T @ weights.flatten(), rtol=1e-12, atol=1e-12)
    assert torch.autograd.gradcheck(mg, (residual,))
    assert torch.autograd.gradgradcheck(mg, (residual,))


@pytest.mark.parametrize('n', [20, 40, 100])
def test_one_mg_cycle_recovers_smooth_oscillatory_mixed_and_boundary_errors(n):
    from scripts.diagnose_fv_direct_state import probe_fields
    _, geom = solver_and_geom(n)
    mg = MGOneCycleInverse(geom)
    for field in probe_fields(n, 1701, .2).values():
        full = torch.cat((field, torch.zeros(1, n, dtype=field.dtype)))[None]
        correction = mg(implicit_cn_action(full, geom))[0]
        assert (correction - field).norm() / field.norm() < .1
    if n == 100:
        assert mg.level_shapes == [(99,100), (50,51), (25,26), (13,14), (7,8)]


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='MPS unavailable')
def test_mg_mps_forward_backward_stay_on_device(monkeypatch):
    _, geom = solver_and_geom(100)
    cpu = MGOneCycleInverse(geom).float()
    mg = MGOneCycleInverse(geom).to(device='mps', dtype=torch.float32)
    residual = torch.randn(2,99,100, requires_grad=True)
    expected = cpu(residual)
    expected_grad = torch.autograd.grad(expected.square().mean(), residual)[0]
    gpu_input = residual.detach().to('mps').requires_grad_()
    def forbidden(*args, **kwargs):
        raise AssertionError('MG application transferred a tensor to CPU/NumPy')
    with monkeypatch.context() as guarded:
        guarded.setattr(torch.Tensor, 'cpu', forbidden)
        guarded.setattr(torch.Tensor, 'numpy', forbidden)
        actual = mg(gpu_input)
        actual_grad = torch.autograd.grad(actual.square().mean(), gpu_input)[0]
    torch.testing.assert_close(actual.cpu(), expected, rtol=3e-5, atol=1e-6)
    torch.testing.assert_close(actual_grad.cpu(), expected_grad, rtol=3e-5, atol=1e-8)
    assert all(b.device.type == 'mps' for b in mg.buffers())
