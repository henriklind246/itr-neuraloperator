import numpy as np
import torch

from scripts.diagnose_fv_direct_state import active_system, build_problem, normalized_rhs
from scripts.diagnose_fv_variational_cvit import (
    build_cn_tensors,
    causal_variational_terms,
    explicit_cn_rhs,
    implicit_cn_action,
)


def _setup():
    solver = build_problem(
        grid_size=10,
        dt=0.005,
        t_final=0.02,
        flux_amplitude=300.0,
        flux_frequency=2.0,
        interface_x=0.5,
        resistance=0.5,
    )
    cn = build_cn_tensors(
        solver, sigma=10.0, device=torch.device("cpu"), dtype=torch.float64
    )
    return solver, cn


def test_dense_implicit_action_matches_active_sparse_matrix():
    solver, cn = _setup()
    matrix, _ = active_system(solver)
    generator = torch.Generator().manual_seed(3)
    state = torch.randn(
        2, solver.Nx, solver.Ny, generator=generator, dtype=torch.float64
    )
    state[:, -1] = 0.0
    actual = implicit_cn_action(state, cn)
    for batch in range(2):
        expected = matrix @ state[batch, :-1].numpy().ravel(order="F")
        np.testing.assert_allclose(
            actual[batch].numpy().ravel(order="F"), expected, rtol=0.0, atol=1e-12
        )


def test_dense_explicit_rhs_matches_reference_assembly():
    solver, cn = _setup()
    rng = np.random.default_rng(5)
    state = rng.normal(scale=0.1, size=(2, solver.Nx, solver.Ny))
    state[:, -1] = 0.0
    steps = torch.tensor([0, 2], dtype=torch.long)
    actual = explicit_cn_rhs(
        torch.as_tensor(state, dtype=torch.float64), steps, cn
    )
    for batch, step in enumerate(steps.tolist()):
        expected = normalized_rhs(
            solver, state[batch], float(solver.t[step]), sigma=10.0
        )
        np.testing.assert_allclose(
            actual[batch].numpy().ravel(order="F"), expected, rtol=0.0, atol=1e-12
        )


def test_variational_gradient_is_weighted_residual_and_detaches_previous_state():
    solver, cn = _setup()
    generator = torch.Generator().manual_seed(7)
    prediction_n_values = torch.randn(
        1, solver.Nx, solver.Ny, generator=generator, dtype=torch.float64
    )
    prediction_np1_values = torch.randn(
        1, solver.Nx, solver.Ny, generator=generator, dtype=torch.float64
    )
    prediction_n_values[:, -1] = 0.0
    prediction_np1_values[:, -1] = 0.0
    prediction_n = prediction_n_values.requires_grad_()
    prediction_np1 = prediction_np1_values.requires_grad_()
    energy, residual_rms = causal_variational_terms(
        prediction_np1,
        prediction_n,
        torch.tensor([1]),
        cn,
        right_value=0.0,
    )
    gradient_n, gradient_np1 = torch.autograd.grad(
        energy, (prediction_n, prediction_np1), allow_unused=True
    )
    assert gradient_n is None
    implicit = implicit_cn_action(prediction_np1, cn)
    rhs = explicit_cn_rhs(prediction_n, torch.tensor([1]), cn)
    expected = cn["capacity"] * (implicit - rhs) / cn["capacity"].sum()
    torch.testing.assert_close(gradient_np1[0, :-1], expected[0])
    assert float(gradient_np1[0, -1].abs().max()) > 0.0
    assert float(residual_rms) > 0.0
