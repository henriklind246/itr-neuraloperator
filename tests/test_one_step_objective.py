"""Physics/objective tests for the one-step CN module (plan Section 7).

These lock in the mathematical contract of `src/physics/one_step_objective.py`
before the production runner is built on top of it:

  - the capacity-weighted CN operator ``C A`` is symmetric (so the variational
    energy is a valid convex objective),
  - the variational gradient is the capacity-weighted CN residual ``C (A u - b)``,
  - the variational and single-sweep-defect objectives share the discrete FV
    minimizer,
  - the differentiable defect matches an independent reference,
  - the conservative storage projection never moves the right Dirichlet wall,
  - per-sample geometry produces distinct CN operators, and
  - the objective is quiescent on the zero-forcing equilibrium yet responds to a
    nonzero forcing increment.
"""

from __future__ import annotations

import numpy as np
import torch

from src.physics.fv_residual import build_cn_geom_per_interface
from src.physics.one_step_objective import (
    _apply_free_operator,
    build_cn_tensors_from_geom,
    cn_free_diagonal,
    conservative_energy_storage_projection,
    defect_terms,
    explicit_cn_rhs_one_step,
    implicit_cn_action,
    one_step_objective,
    variational_objective,
)

DTYPE = torch.float64
K_LEFT = 2.0
K_RIGHT = 1.0
RIGHT_VALUE = 0.0
DT = 1.0e-3


def _grids(nx: int = 7, ny: int = 5):
    x_grid = np.linspace(0.0, 1.0, nx)
    y_grid = np.linspace(0.0, 1.0, ny)
    return x_grid, y_grid


def _make_cn(interface_x, r_c, *, forcing=None, nx=7, ny=5):
    x_grid, y_grid = _grids(nx, ny)
    iface = np.atleast_1d(np.asarray(interface_x, dtype=float))
    rc = np.atleast_1d(np.asarray(r_c, dtype=float))
    geom = build_cn_geom_per_interface(
        x_grid, y_grid, K_LEFT, K_RIGHT, iface, rc, DT,
        sigma_global=1.0, dtype=DTYPE,
    )
    b = iface.shape[0]
    if forcing is None:
        forcing = torch.zeros((b, geom.Ny), dtype=DTYPE)
    cn = build_cn_tensors_from_geom(geom, forcing, dtype=DTYPE)
    return geom, cn, x_grid, y_grid


def _dense_free_operator(cn, nx, ny):
    """Assemble the dense free-DOF operator A for a single-sample cn dict."""
    m = (nx - 1) * ny
    columns = []
    for k in range(m):
        basis = torch.zeros((1, nx - 1, ny), dtype=DTYPE)
        basis.view(1, -1)[0, k] = 1.0
        columns.append(_apply_free_operator(basis, cn).reshape(-1))
    return torch.stack(columns, dim=1)


def test_capacity_weighted_operator_is_symmetric():
    # Moving interface + distinct R_c: the raw CN operator is NOT symmetric, but
    # C A is (the shared face conductance makes C_i r_w[i] == C_{i-1} r_e[i-1]).
    for interface_x, r_c in ((0.45, 0.3), (0.6, 0.7), (0.3, 1.0)):
        _, cn, _, _ = _make_cn(interface_x, r_c)
        nx = cn["r_w"].shape[-2]
        ny = cn["r_w"].shape[-1]
        a_free = _dense_free_operator(cn, nx, ny)
        capacity = cn["capacity"].reshape(-1)
        ca = capacity[:, None] * a_free
        asym = (ca - ca.T).abs().max()
        scale = ca.abs().max()
        assert asym <= 1e-10 * scale, (interface_x, r_c, float(asym), float(scale))


def test_capacity_weighted_operator_is_positive_definite():
    for interface_x, r_c in ((0.45, 0.3), (0.6, 0.7)):
        _, cn, _, _ = _make_cn(interface_x, r_c)
        nx = cn["r_w"].shape[-2]
        ny = cn["r_w"].shape[-1]
        a_free = _dense_free_operator(cn, nx, ny)
        capacity = cn["capacity"].reshape(-1)
        ca = capacity[:, None] * a_free
        sym = 0.5 * (ca + ca.T)
        eigvals = torch.linalg.eigvalsh(sym)
        assert float(eigvals.min()) > 0.0, (interface_x, r_c, float(eigvals.min()))


def test_variational_gradient_is_capacity_weighted_residual():
    torch.manual_seed(0)
    geom, cn, _, _ = _make_cn(0.45, 0.3)
    nx, ny = geom.Nx, geom.Ny
    state_n = torch.full((1, nx, ny), RIGHT_VALUE, dtype=DTYPE)
    state_n[:, :2] += 0.5  # nonuniform on-manifold-ish state

    free = torch.randn((1, nx - 1, ny), dtype=DTYPE, requires_grad=True)
    wall = torch.full((1, 1, ny), RIGHT_VALUE, dtype=DTYPE)
    pred = torch.cat((free + RIGHT_VALUE, wall), dim=-2)

    energy, residual = variational_objective(pred, state_n, cn, right_value=RIGHT_VALUE)
    energy.sum().backward()

    norm = cn["capacity"].sum(dim=(-2, -1))
    expected = (cn["capacity"] * residual.detach()) / norm
    assert torch.allclose(free.grad, expected, atol=1e-9, rtol=1e-6)


def test_variational_and_defect_share_fv_minimizer():
    # Solve A u* = b densely; both objectives must have ~zero residual/gradient
    # there (minimizer parity), for several moving-interface geometries.
    torch.manual_seed(1)
    for interface_x, r_c in ((0.45, 0.3), (0.6, 0.7)):
        geom, cn, _, _ = _make_cn(
            interface_x, r_c,
            forcing=torch.randn((1, 5), dtype=DTYPE) * 1e-2,
        )
        nx, ny = geom.Nx, geom.Ny
        state_n = torch.randn((1, nx, ny), dtype=DTYPE) * 0.1 + RIGHT_VALUE
        state_n[:, -1] = RIGHT_VALUE

        deviation_n = state_n - RIGHT_VALUE
        b_free = explicit_cn_rhs_one_step(deviation_n, cn).reshape(-1)
        a_free = _dense_free_operator(cn, nx, ny)
        u_star = torch.linalg.solve(a_free, b_free).reshape(1, nx - 1, ny)

        wall = torch.full((1, 1, ny), RIGHT_VALUE, dtype=DTYPE)
        pred_star = torch.cat((u_star + RIGHT_VALUE, wall), dim=-2)

        _, res_var = variational_objective(pred_star, state_n, cn, right_value=RIGHT_VALUE)
        _, res_def = defect_terms(pred_star, state_n, cn, right_value=RIGHT_VALUE)
        assert res_var.abs().max() <= 1e-8
        assert res_def.abs().max() <= 1e-8

        pred_leaf = pred_star.detach().clone().requires_grad_(True)
        e_var, _ = variational_objective(pred_leaf, state_n, cn, right_value=RIGHT_VALUE)
        e_var.sum().backward()
        assert pred_leaf.grad[:, :-1].abs().max() <= 1e-7

        pred_leaf2 = pred_star.detach().clone().requires_grad_(True)
        e_def, _ = defect_terms(pred_leaf2, state_n, cn, right_value=RIGHT_VALUE)
        e_def.sum().backward()
        assert pred_leaf2.grad[:, :-1].abs().max() <= 1e-7


def test_defect_single_sweep_matches_reference():
    torch.manual_seed(2)
    geom, cn, _, _ = _make_cn(0.6, 0.7, forcing=torch.randn((1, 5), dtype=DTYPE) * 1e-2)
    nx, ny = geom.Nx, geom.Ny
    state_n = torch.randn((1, nx, ny), dtype=DTYPE) * 0.1
    state_n[:, -1] = RIGHT_VALUE
    pred = torch.randn((1, nx, ny), dtype=DTYPE) * 0.1
    pred[:, -1] = RIGHT_VALUE

    energy, residual = defect_terms(pred, state_n, cn, right_value=RIGHT_VALUE)
    diag = cn_free_diagonal(cn, nx)
    reference = (residual / diag.sqrt()).square().mean(dim=(-2, -1))
    assert torch.allclose(energy, reference, atol=1e-12)


def test_projection_preserves_right_dirichlet_wall():
    geom, cn, x_grid, _ = _make_cn(0.45, 0.3)
    nx, ny = geom.Nx, geom.Ny
    state_n = torch.full((1, nx, ny), RIGHT_VALUE, dtype=DTYPE)
    state_n[:, :2] += 0.4
    smooth = torch.randn((1, nx, ny), dtype=DTYPE) * 0.1 + RIGHT_VALUE
    smooth[:, -1] = RIGHT_VALUE
    q_left_integral = torch.full((1, ny), 5.0, dtype=DTYPE)

    _, projected, correction, _ = conservative_energy_storage_projection(
        state_n, smooth, geom, q_left_integral,
        resistance=0.3, right_value=RIGHT_VALUE, sigma=1.0,
    )
    # The correction bubble vanishes at (and past) the right wall.
    assert correction[:, -1].abs().max() <= 1e-12
    assert torch.allclose(projected[:, -1], smooth[:, -1], atol=1e-12)


def test_per_sample_geometry_distinct_operators():
    _, cn, _, _ = _make_cn([0.35, 0.65], [0.2, 0.9])
    # Different interface_x / R_c must give different CN coefficients per sample.
    assert not torch.allclose(cn["r_w"][0], cn["r_w"][1])
    assert not torch.allclose(cn["r_e"][0], cn["r_e"][1])


def test_zero_forcing_equilibrium_is_quiescent():
    geom, cn, _, _ = _make_cn(0.45, 0.3)  # forcing defaults to zeros
    nx, ny = geom.Nx, geom.Ny
    equilibrium = torch.full((1, nx, ny), RIGHT_VALUE, dtype=DTYPE, requires_grad=True)
    energy, residual = variational_objective(
        equilibrium, equilibrium, cn, right_value=RIGHT_VALUE
    )
    assert residual.abs().max() <= 1e-12
    energy.sum().backward()
    assert equilibrium.grad.abs().max() <= 1e-12


def test_nonzero_forcing_moves_constant_prediction():
    forcing = torch.full((1, 5), 1.0, dtype=DTYPE)
    geom, cn, _, _ = _make_cn(0.45, 0.3, forcing=forcing)
    nx, ny = geom.Nx, geom.Ny
    const = torch.full((1, nx, ny), RIGHT_VALUE, dtype=DTYPE, requires_grad=True)
    energy, residual = defect_terms(const, const, cn, right_value=RIGHT_VALUE)
    assert residual.abs().max() > 0.0
    energy.sum().backward()
    assert const.grad.abs().max() > 0.0


def test_dispatcher_matches_direct_objectives():
    torch.manual_seed(3)
    geom, cn, _, _ = _make_cn(0.6, 0.7, forcing=torch.randn((1, 5), dtype=DTYPE) * 1e-2)
    nx, ny = geom.Nx, geom.Ny
    state_n = torch.randn((1, nx, ny), dtype=DTYPE) * 0.1
    state_n[:, -1] = RIGHT_VALUE
    pred = torch.randn((1, nx, ny), dtype=DTYPE) * 0.1
    pred[:, -1] = RIGHT_VALUE

    e_var, _ = variational_objective(pred, state_n, cn, right_value=RIGHT_VALUE)
    loss_var, metrics_var = one_step_objective(
        "variational", pred, state_n, cn, right_value=RIGHT_VALUE
    )
    assert torch.allclose(loss_var, e_var.mean())
    assert "defect_rms" in metrics_var and "energy_per_case" in metrics_var

    e_def, _ = defect_terms(pred, state_n, cn, right_value=RIGHT_VALUE)
    loss_def, _ = one_step_objective(
        "defect", pred, state_n, cn, right_value=RIGHT_VALUE
    )
    assert torch.allclose(loss_def, e_def.mean())


def test_state_n_detached_from_target_step():
    geom, cn, _, _ = _make_cn(0.45, 0.3, forcing=torch.full((1, 5), 0.5, dtype=DTYPE))
    nx, ny = geom.Nx, geom.Ny
    state_n = (torch.randn((1, nx, ny), dtype=DTYPE) * 0.1).requires_grad_(True)
    pred = torch.zeros((1, nx, ny), dtype=DTYPE, requires_grad=True)
    loss, _ = one_step_objective("defect", pred, state_n, cn, right_value=RIGHT_VALUE)
    loss.backward()
    assert state_n.grad is None or state_n.grad.abs().max() == 0.0
    assert pred.grad is not None and pred.grad.abs().max() > 0.0
