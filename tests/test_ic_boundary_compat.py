"""Core-computation tests for scripts/diffusion_ic_boundary_compat.py.

Covers the finite-difference wall-normal derivatives, the sign convention of the
left-wall mismatch, and that the real IC builders (with their boundary taper)
produce t=0 fields whose wall-normal derivatives are ~0 -- consistent with the
ramp guaranteeing q_L(y, 0) = 0.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.diffusion_ic_boundary_compat import (
    boundary_normal_derivs,
    ic_boundary_mismatch,
    qL_at_t0,
)
from src.operators.train_pino import sample_forcing_params
from src.physics.boundary_forcing import default_ramp_seconds
from src.physics.init_conditions import IC_SAMPLERS, build_ic

_FAMILIES = ("uniform_2d", "random_sinusoid_2d", "grf_2d", "hot_spot_2d")


def _sin_uniform_param(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return sample_forcing_params(
        rng, 1, dt=1e-3, t_final=0.2,
        temporal_family="sin", spatial_family="uniform",
    )[0]


def _grid(n=41):
    axis = np.linspace(0.0, 1.0, n)
    return axis, axis


def test_boundary_normal_derivs_linear_x_field():
    # T0 = 2 x  => d_x T0 = 2 everywhere, d_y T0 = 0 everywhere.
    x, y = _grid()
    X, _Y = np.meshgrid(x, y, indexing="ij")
    T0 = 2.0 * X
    d = boundary_normal_derivs(T0, x, y)
    assert np.allclose(d["dTdx_left"], 2.0)
    assert np.allclose(d["dTdy_top"], 0.0)
    assert np.allclose(d["dTdy_bottom"], 0.0)


def test_boundary_normal_derivs_linear_y_field():
    # T0 = 3 y  => d_y T0 = 3 at top and bottom, d_x T0 = 0 at left.
    x, y = _grid()
    _X, Y = np.meshgrid(x, y, indexing="ij")
    T0 = 3.0 * Y
    d = boundary_normal_derivs(T0, x, y)
    assert np.allclose(d["dTdx_left"], 0.0)
    assert np.allclose(d["dTdy_top"], 3.0)
    assert np.allclose(d["dTdy_bottom"], 3.0)


def test_boundary_normal_derivs_rejects_wrong_shape():
    x, y = _grid(10)
    with pytest.raises(ValueError):
        boundary_normal_derivs(np.zeros((9, 10)), x, y)


def test_left_mismatch_uses_inward_flux_sign():
    # -k d_x T0|_{x=0} must match q_L exactly cancels when q_L = -k d_x T0.
    x, y = _grid()
    X, _Y = np.meshgrid(x, y, indexing="ij")
    k = 2.0
    T0 = 5.0 * X  # d_x T0 = 5 at the left wall
    qL_perfect = np.full(y.shape[0], -k * 5.0)  # = -k d_x T0
    m = ic_boundary_mismatch(T0, x, y, qL_perfect, k=k)
    assert m["left"] == pytest.approx(0.0, abs=1e-9)
    # Flipping the sign of q_L doubles the residual (proves sign is load-bearing).
    m_flipped = ic_boundary_mismatch(T0, x, y, -qL_perfect, k=k)
    assert m_flipped["left"] == pytest.approx(2.0 * k * 5.0, rel=1e-9)


def test_qL_at_t0_is_zero_under_ramp():
    # Ramped sin/uniform forcing: ramp_envelope(0) = 0 => q_L(y, 0) = 0.
    _x, y = _grid(50)
    param = _sin_uniform_param()
    t_ramp = default_ramp_seconds(1e-3)
    q0 = qL_at_t0(param, y, t_ramp)
    assert np.max(np.abs(q0)) == pytest.approx(0.0, abs=1e-12)


def test_qL_at_t0_nonzero_without_ramp():
    # Sanity: with t_ramp = 0 and a small positive query time inside the window,
    # the flux is a genuine nonzero sin amplitude, so the "= 0" ramp result above
    # is really the startup ramp, not an accident of the evaluator. (At exactly
    # t=0 the Tukey window itself can be 0, so probe a small in-window time.)
    _x, y = _grid(50)
    param = _sin_uniform_param()
    forcing_t = 0.05
    from src.physics.boundary_forcing import reconstruct_qL

    ev = reconstruct_qL(
        param["temporal_family"], param["temporal_params"],
        param["spatial_family"], param["spatial_params"], t_ramp=0.0,
    )
    q = np.asarray(ev.evaluate_points(y, np.full_like(y, forcing_t)), dtype=float)
    assert np.max(np.abs(q)) > 0.0


@pytest.mark.parametrize("family", _FAMILIES)
def test_real_ic_builder_normal_derivative_converges(family):
    # The boundary taper (boundary_taper in init_conditions.py) drives every
    # deviation's value AND wall-normal derivative to zero CONTINUOUSLY at each
    # wall. The one-sided finite difference therefore does not vanish on a finite
    # grid -- it picks up the near-wall curvature -- but it must SHRINK as the
    # grid refines (the continuous derivative it approximates is 0). Refining the
    # grid must reduce the top/bottom/left wall-normal derivative magnitudes.
    def wall_maxes(n: int) -> dict[str, float]:
        x, y = _grid(n)
        X, Y = np.meshgrid(x, y, indexing="ij")
        rng = np.random.default_rng(0)
        params = IC_SAMPLERS[family](rng, Nx=n, Ny=n)
        T0 = build_ic(family, params, X, Y, T_right=300.0)
        m = ic_boundary_mismatch(T0, x, y, np.zeros(n), k=1.0)
        return m

    coarse = wall_maxes(51)
    fine = wall_maxes(201)
    for wall in ("top", "bottom", "left"):
        # uniform_2d has a constant deviation whose tapered top/bottom derivative
        # is already tiny; require monotone non-increase with a small tolerance.
        assert fine[wall] <= coarse[wall] * 0.9 + 1e-6
