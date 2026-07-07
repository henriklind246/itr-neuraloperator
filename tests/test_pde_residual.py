import math

import torch

from src.physics.pde_residual import (
    diffusion_residual,
    ic_residual,
    neumann_residual,
)

PI = math.pi


class _ExactHeat:
    """Analytic heat solution (alpha=1): exp(-2 pi^2 t) cos(pi x) cos(pi y).

    Satisfies T_t = T_xx + T_yy exactly, and zero-Neumann on all four walls
    (dT/dx = -pi sin(pi x) -> 0 at x=0,1; likewise dT/dy at y=0,1). Used to
    validate the autodiff wiring independent of any learned weights.
    """

    def __call__(self, u, coords, t):
        x = coords[..., 0:1]
        y = coords[..., 1:2]
        out = torch.exp(-2.0 * PI * PI * t) * torch.cos(PI * x) * torch.cos(PI * y)
        # Mimic CViT.forward: expand a dim-1 query set across the sim batch.
        if out.shape[0] == 1 and u.shape[0] > 1:
            out = out.expand(u.shape[0], -1, -1)
        return out


def test_residual_analytic_zero():
    m = _ExactHeat()
    u = torch.zeros(2, 1, 20, 20)
    x = torch.rand(2, 128, 1, requires_grad=True)
    y = torch.rand(2, 128, 1, requires_grad=True)
    t = (torch.rand(2, 128, 1) * 0.3).requires_grad_(True)
    r = diffusion_residual(m, u, x, y, t, alpha=1.0)
    assert r.shape == (2, 128, 1)
    assert r.abs().max().item() < 1e-4


def test_neumann_zero_on_walls():
    m = _ExactHeat()
    u = torch.zeros(1, 1, 20, 20)
    n = 64
    # left x=0
    x0 = torch.zeros(1, n, 1, requires_grad=True)
    y = torch.rand(1, n, 1, requires_grad=True)
    t = (torch.rand(1, n, 1) * 0.3).requires_grad_(True)
    assert neumann_residual(m, u, x0, y, t, "left").abs().max().item() < 1e-5
    # bottom y=0
    x = torch.rand(1, n, 1, requires_grad=True)
    y0 = torch.zeros(1, n, 1, requires_grad=True)
    assert neumann_residual(m, u, x, y0, t, "bottom").abs().max().item() < 1e-5
    # top y=1
    y1 = torch.ones(1, n, 1, requires_grad=True)
    assert neumann_residual(m, u, x, y1, t, "top").abs().max().item() < 1e-5


def test_neumann_unknown_wall_raises():
    m = _ExactHeat()
    u = torch.zeros(1, 4, 1)
    x = torch.zeros(1, 4, 1, requires_grad=True)
    y = torch.zeros(1, 4, 1, requires_grad=True)
    t = torch.zeros(1, 4, 1, requires_grad=True)
    try:
        neumann_residual(m, u, x, y, t, "right")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown wall")


def test_ic_residual_shape_and_value():
    m = _ExactHeat()
    u = torch.zeros(2, 1, 20, 20)
    coords = torch.rand(1, 32, 2)
    t0 = torch.zeros(1, 32, 1)
    target = m(u, coords, t0)  # exact -> residual identically zero
    r = ic_residual(m, u, coords, t0, target)
    assert r.shape == (2, 32, 1)
    assert r.abs().max().item() < 1e-6
