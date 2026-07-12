import numpy as np
import pytest

from src.physics.boundary_forcing import (
    build_interface_forcing,
    ramped_temporal,
    default_ramp_seconds,
    A_REF_FLUX,
    FORCING_TEMPORAL_SAMPLES,
)

"""
Stage 2 units tests for the canonical interface forcing helper
(`build_interface_forcing`, §0). One helper emits BOTH the sampled waveform the
model sees (`forcing_seq`) and the physical FV boundary flux (`qL_n/qL_np1/qL_int`)
from the same parameters, so they cannot drift. These tests pin:
  - `qL_int` is the exact SIGNED step INTEGRAL (not average), matching the
    solver's exact left-Neumann RHS, verified against a fine quadrature.
  - the model input is NORMALIZED (`a/A_ref`, absolute normalized time) while the
    flux is PHYSICAL (`a*s(y)`), and the two agree at shared sample times.
"""

T_FINAL = 0.2
DT = 0.005


def _sin_params():
    return {
        "A": 200.0, "f": 5.0, "t_on": 0.0, "t_off": T_FINAL,
        "phase": 0.0, "tukey_alpha": 0.5, "rectified": True,
    }


def _build(t_n, t_np1, y_grid=None, ramp=None):
    if y_grid is None:
        y_grid = np.linspace(0.0, 1.0, 100)
    if ramp is None:
        ramp = default_ramp_seconds(DT)
    return build_interface_forcing(
        "sin", _sin_params(), "uniform", {}, y_grid,
        t_n, t_np1, T_FINAL, ramp,
    )


def test_shapes_and_dtypes():
    fseq, qLn, qLnp1, qLint = _build(0.05, 0.055)
    assert fseq.shape == (FORCING_TEMPORAL_SAMPLES, 2)
    assert fseq.dtype == np.float32
    for v in (qLn, qLnp1, qLint):
        assert v.shape == (100,)
        assert v.dtype == np.float64


def test_forcing_seq_absolute_normalized_time():
    fseq, *_ = _build(0.05, 0.055)
    tau_norm = fseq[:, 0]
    # Absolute normalized time on the WHOLE schedule: 0 -> 1 monotone increasing.
    assert tau_norm[0] == pytest.approx(0.0)
    assert tau_norm[-1] == pytest.approx(1.0)
    assert np.all(np.diff(tau_norm) > 0)


def test_forcing_seq_amplitude_is_normalized_ramped_waveform():
    """Token 1 equals ramp(t)*a(t)/A_ref at each absolute sample time."""
    ramp = default_ramp_seconds(DT)
    fseq, *_ = _build(0.05, 0.055, ramp=ramp)
    a_fn = ramped_temporal("sin", _sin_params(), ramp)
    tau = np.linspace(0.0, T_FINAL, FORCING_TEMPORAL_SAMPLES)
    expected = np.array([a_fn(float(tm)) for tm in tau]) / A_REF_FLUX
    np.testing.assert_allclose(fseq[:, 1], expected, atol=1e-6)
    # Normalized: sampled amplitude token is O(1), never the physical ~O(200).
    assert float(np.abs(fseq[:, 1]).max()) < 1.5


def test_ramp_zeros_flux_at_t0():
    """The startup ramp forces q_L(0) = 0 in both the waveform and the flux."""
    fseq, qLn, _, _ = _build(0.0, DT)
    assert fseq[0, 1] == pytest.approx(0.0, abs=1e-12)
    assert float(np.abs(qLn).max()) == pytest.approx(0.0, abs=1e-12)


def test_uniform_spatial_profile_flux_constant_in_y():
    """s(y)=1 for interfaces: physical flux is uniform across the left wall."""
    _, qLn, qLnp1, qLint = _build(0.05, 0.055)
    for v in (qLn, qLnp1, qLint):
        assert float(v.max() - v.min()) == pytest.approx(0.0, abs=1e-12)


def test_flux_matches_ramped_waveform_at_endpoints():
    """`qL_n`/`qL_np1` (physical, s=1) equal ramp*a at the interval endpoints —
    the model waveform (a/A_ref) and the residual flux (a*s) share one a(t)."""
    ramp = default_ramp_seconds(DT)
    t_n, t_np1 = 0.05, 0.055
    _, qLn, qLnp1, _ = _build(t_n, t_np1, ramp=ramp)
    a_fn = ramped_temporal("sin", _sin_params(), ramp)
    assert float(qLn[0]) == pytest.approx(a_fn(t_n), abs=1e-9)
    assert float(qLnp1[0]) == pytest.approx(a_fn(t_np1), abs=1e-9)


@pytest.mark.parametrize("t_n,t_np1", [(0.0, DT), (0.05, 0.055), (0.1, 0.11)])
def test_qL_int_is_exact_signed_integral(t_n, t_np1):
    """`qL_int` matches a fine trapezoid quadrature of ramp*a over [t_n, t_np1]
    (an integral with units K*s-equivalent flux*time, NOT an interval average)."""
    ramp = default_ramp_seconds(DT)
    _, _, _, qLint = _build(t_n, t_np1, ramp=ramp)
    a_fn = ramped_temporal("sin", _sin_params(), ramp)
    tt = np.linspace(t_n, t_np1, 20001)
    ref = np.trapezoid([a_fn(float(x)) for x in tt], tt)  # s(y)=1
    assert float(qLint[0]) == pytest.approx(float(ref), rel=1e-4, abs=1e-8)
    # Integral, not average: dividing by dt gives back an O(a) mean, not qL_int.
    assert abs(float(qLint[0])) < abs(float(ref)) + 1e-12
    mean = float(qLint[0]) / (t_np1 - t_n)
    assert abs(mean) > abs(float(qLint[0]))  # qL_int << its own time-average*1


def test_consistency_across_repeated_builds():
    """Deterministic: same params -> bitwise-identical waveform and flux."""
    a = _build(0.05, 0.055)
    b = _build(0.05, 0.055)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)
