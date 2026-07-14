import math

import numpy as np
import torch

from src.physics.boundary_forcing import reconstruct_qL
from src.physics.pde_residual import (
    diffusion_residual,
    forcing_neumann_residual,
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

    def __call__(self, u, coords, t, q_left=None):
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


class _PerSimLinearX:
    """T(x,y,t) = c_b * x with c_b = mean(u_b), so dT/dx = c_b differs per sim.

    Used to pin that a derivative residual computed from a SHARED (1,N,1)
    collocation leaf stays per-sim (row b == c_b), not the batch sum sum_b c_b.
    """

    def __call__(self, u, coords, t, q_left=None):
        x = coords[..., 0:1]
        c = u.mean(dim=(1, 2, 3)).reshape(-1, 1, 1)
        return c * x


def test_shared_leaf_derivative_residual_is_per_sim_not_batch_summed():
    m = _PerSimLinearX()
    B, n = 3, 5
    u = torch.randn(B, 1, 8, 8)
    # Shared (1, n, 1) leaves, exactly as sample_collocation emits them.
    x0 = torch.zeros(1, n, 1, requires_grad=True)
    y = torch.rand(1, n, 1, requires_grad=True)
    t = (torch.rand(1, n, 1) * 0.3).requires_grad_(True)
    r = neumann_residual(m, u, x0, y, t, "left")  # dT/dx = c_b
    assert r.shape == (B, n, 1)
    c = u.mean(dim=(1, 2, 3)).reshape(B, 1, 1)
    # Per-sim: every point of row b carries c_b (not the batch sum sum_b c_b).
    assert torch.allclose(r, c.expand(B, n, 1), atol=1e-5)
    batch_sum = c.sum()
    assert (r - batch_sum).abs().min().item() > 1e-3  # decisively not the sum


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


class _LinearLeftFlux:
    """Field with a known inward left flux: T_tilde(x,y,t) = -q(y,t)/(k*sigma)*x.

    Then dT_tilde/dx = -q/(k*sigma) everywhere, so the inhomogeneous left-Neumann
    residual  dT_tilde/dx + q_L/(k*sigma)  is identically zero when q_L == q. This
    pins the sign, the 1/(k*sigma) normalization, and that autograd sees the
    PHYSICAL x leaf with no rescale.
    """

    def __init__(self, k: float, sigma: float):
        self.k = float(k)
        self.sigma = float(sigma)

    def q(self, y, t):
        # Arbitrary smooth q(y, t) > 0; must match the q_L fed to the residual.
        return 2.0 + y + t

    def __call__(self, u, coords, t, q_left=None):
        x = coords[..., 0:1]
        y = coords[..., 1:2]
        out = -(self.q(y, t) / (self.k * self.sigma)) * x
        if out.shape[0] == 1 and u.shape[0] > 1:
            out = out.expand(u.shape[0], -1, -1)
        return out


def test_forcing_left_neumann_analytic_zero():
    k, sigma = 1.0, 4.0
    m = _LinearLeftFlux(k=k, sigma=sigma)
    u = torch.zeros(2, 1, 16, 20)
    n = 64
    x0 = torch.zeros(2, n, 1, requires_grad=True)  # left wall x=0
    y = torch.rand(2, n, 1, requires_grad=True)
    t = (torch.rand(2, n, 1) * 0.3).requires_grad_(True)
    q_L = m.q(y.detach(), t.detach())  # matched inward flux at (y, t)
    r = forcing_neumann_residual(m, u, x0, y, t, q_L=q_L, sigma=sigma, k=k)
    assert r.shape == (2, n, 1)
    assert r.abs().max().item() < 1e-5


def test_forcing_left_neumann_wrong_sign_is_nonzero():
    # Sanity guard: feeding -q_L (wrong sign) must NOT cancel; residual = 2q/(k*sigma).
    k, sigma = 1.0, 4.0
    m = _LinearLeftFlux(k=k, sigma=sigma)
    u = torch.zeros(1, 1, 16, 20)
    n = 32
    x0 = torch.zeros(1, n, 1, requires_grad=True)
    y = torch.rand(1, n, 1, requires_grad=True)
    t = (torch.rand(1, n, 1) * 0.3).requires_grad_(True)
    r = forcing_neumann_residual(m, u, x0, y, t, q_L=-m.q(y.detach(), t.detach()),
                                 sigma=sigma, k=k)
    assert r.abs().max().item() > 1e-2


def test_qL_image_equals_pointwise_on_grid():
    # Mandatory single-source check: the encoder-image q_L (q_image) and the
    # residual q_L (q_at) come from the SAME reconstruct_qL helper, so they must
    # agree exactly on shared (y, t) grid nodes.
    forcing = reconstruct_qL(
        temporal_family="sin",
        temporal_params=dict(A=1.0, f=3.0, t_on=0.0, t_off=0.2,
                             phase=0.0, tukey_alpha=0.5),
        spatial_family="gaussian",
        spatial_params=dict(y_c=0.6, sigma_y=0.15),
        t_ramp=0.0,
    )
    y_grid = np.linspace(0.0, 1.0, 12)
    t_axis = np.linspace(0.0, 0.3, 16)
    img = forcing.evaluate_grid(y_grid, t_axis)
    assert img.shape == (12, 16)
    # Compare against q_at at the matched meshgrid points.
    YY, TT = np.meshgrid(y_grid, t_axis, indexing="ij")
    pointwise = forcing.evaluate_points(YY.ravel(), TT.ravel()).reshape(12, 16)
    assert np.allclose(img, pointwise, atol=1e-9)


def test_qL_image_resolves_sharpest_pulse():
    # Aliasing guard: the image time axis (Nt over [0, t_final]) must resolve the
    # narrowest sampled pulse, else two distinct forcings alias to one image.
    rng = np.random.default_rng(0)
    from src.physics.boundary_forcing import TEMPORAL_SAMPLERS
    t_final = 1.0
    tp = TEMPORAL_SAMPLERS["pulse_train"](
        rng, dt=0.001, t_final=t_final, t_on=0.0, t_off=0.2, phase=0.0, tukey_alpha=0.5
    )
    dt_list = np.asarray(tp["dt_list"], dtype=float)
    t_list = np.asarray(tp["t_list"], dtype=float)
    A_list = np.asarray(tp["A_list"], dtype=float)
    i_star = int(np.argmin(dt_list))
    width, t_c, A = dt_list[i_star], t_list[i_star], A_list[i_star]

    # Pick Nt so the grid spacing resolves the narrowest pulse (>= 2 samples in it).
    Nt = int(np.ceil(t_final / width)) * 4 + 1
    forcing = reconstruct_qL(
        temporal_family="pulse_train", temporal_params=tp,
        spatial_family="uniform", spatial_params={}, t_ramp=0.0,
    )
    y_grid = np.array([0.5])
    t_axis = np.linspace(0.0, t_final, Nt)
    a_axis = forcing.evaluate_grid(y_grid, t_axis)[0]

    in_support = (t_axis >= t_c) & (t_axis < t_c + width)
    assert in_support.sum() >= 2  # grid actually resolves the pulse
    # pulse_train is rectangular with positive amplitudes, so any in-support
    # sample carries at least this pulse's amplitude A (overlaps only add).
    assert a_axis[in_support].min() >= A - 1e-6
