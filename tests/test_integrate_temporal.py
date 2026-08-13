import numpy as np
import pytest

from src.physics.boundary_forcing import (
    FORCING_BINS,
    integrate_temporal,
    integrate_temporal_bins,
    integrate_temporal_signed,
    integrate_temporal_bins_signed,
    build_qL,
    build_qL_integral,
)


def test_integrate_sin_trapezoid_matches_positive_half_wave():
    params = {"A": 2.0, "f": 0.25, "t_on": 0.0, "t_off": 1.0, "phase": 0.0, "tukey_alpha": 0.0}
    expected = 2.0 * params["A"] / np.pi
    assert integrate_temporal("sin", params, 0.0, 1.0) == pytest.approx(expected, abs=1e-3)


def test_integrate_exp_matches_analytic_and_zero_before_t0():
    params = {"A": 3.0, "t0": 0.2, "tau": 0.5}
    assert integrate_temporal("exp", params, 0.0, 0.1) == 0.0

    a, b = 0.3, 0.8
    expected = params["A"] * params["tau"] * (
        np.exp(-(a - params["t0"]) / params["tau"])
        - np.exp(-(b - params["t0"]) / params["tau"])
    )
    assert integrate_temporal("exp", params, a, b) == pytest.approx(expected, abs=1e-6)


def test_integrate_pulse_train_overlap_and_outside():
    params = {
        "Np": 2,
        "A_list": [10.0, 20.0],
        "t_list": [0.1, 0.5],
        "dt_list": [0.2, 0.1],
    }
    expected = 10.0 * 0.1 + 20.0 * 0.05
    assert integrate_temporal("pulse_train", params, 0.2, 0.55) == pytest.approx(expected, abs=1e-6)
    assert integrate_temporal("pulse_train", params, 0.31, 0.49) == 0.0


def test_integrate_exp_train_matches_sum_of_analytics_and_outside():
    params = {
        "Np": 2,
        "A_list": [2.0, 5.0],
        "t_list": [0.1, 0.7],
        "tau_list": [0.2, 0.3],
    }
    assert integrate_temporal("exp_train", params, 0.0, 0.05) == 0.0

    a, b = 0.2, 1.0
    expected = 0.0
    for A, t0, tau in zip(params["A_list"], params["t_list"], params["tau_list"]):
        lo = max(a, t0)
        if b > t0:
            expected += A * tau * (np.exp(-(lo - t0) / tau) - np.exp(-(b - t0) / tau))
    assert integrate_temporal("exp_train", params, a, b) == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize(
    ("family", "params", "tol"),
    [
        ("sin", {"A": 2.0, "f": 0.25, "t_on": 0.0, "t_off": 1.0, "phase": 0.0, "tukey_alpha": 0.0}, 1e-3),
        ("exp", {"A": 3.0, "t0": 0.2, "tau": 0.5}, 1e-6),
        ("pulse_train", {"Np": 2, "A_list": [10.0, 20.0], "t_list": [0.1, 0.5], "dt_list": [0.2, 0.1]}, 1e-6),
        ("exp_train", {"Np": 2, "A_list": [2.0, 5.0], "t_list": [0.1, 0.7], "tau_list": [0.2, 0.3]}, 1e-6),
    ],
)
def test_bin_sum_matches_total_integral(family, params, tol):
    bins = integrate_temporal_bins(family, params, 0.0, 1.0, K=FORCING_BINS)
    total = integrate_temporal(family, params, 0.0, 1.0)
    assert bins.shape == (FORCING_BINS,)
    assert float(bins.sum()) == pytest.approx(total, abs=tol)


def test_pulse_train_bins_discriminate_early_and_late_pulses():
    early = {"Np": 1, "A_list": [8.0], "t_list": [0.0], "dt_list": [0.25]}
    late = {"Np": 1, "A_list": [8.0], "t_list": [0.75], "dt_list": [0.25]}

    early_bins = integrate_temporal_bins("pulse_train", early, 0.0, 1.0, K=FORCING_BINS)
    late_bins = integrate_temporal_bins("pulse_train", late, 0.0, 1.0, K=FORCING_BINS)

    assert early_bins[0] == pytest.approx(late_bins[-1], abs=1e-6)
    assert early_bins[-1] == pytest.approx(late_bins[0], abs=1e-6)
    assert not np.allclose(early_bins, late_bins)


def test_integrate_signed_sin_is_half_wave_rectified():
    # The "sin" family is a half-wave-rectified sine. The signed helper is
    # signed for train families, but does not reintroduce negative sine lobes.
    params = {"A": 1.0, "f": 1.0, "t_on": 0.0, "t_off": 1.0, "phase": 0.0, "tukey_alpha": 0.0}
    expected = 1.0 / np.pi
    assert integrate_temporal_signed("sin", params, 0.0, 1.0) == pytest.approx(expected, abs=1e-2)
    assert integrate_temporal("sin", params, 0.0, 1.0) == pytest.approx(expected, abs=1e-2)


def test_integrate_signed_pulse_train_negative_amplitude():
    # Negative-amplitude pulse: signed integral is negative; clipped helper
    # (current integrate_temporal) treats pulse_train identically and is also
    # signed for that family, so both agree here.
    params = {"Np": 1, "A_list": [-4.0], "t_list": [0.1], "dt_list": [0.3]}
    assert integrate_temporal_signed("pulse_train", params, 0.0, 1.0) == pytest.approx(-4.0 * 0.3, abs=1e-12)


def test_signed_bins_sum_matches_total_signed_integral():
    params = {"A": 1.0, "f": 1.0, "t_on": 0.0, "t_off": 1.0, "phase": 0.0, "tukey_alpha": 0.0}
    bins = integrate_temporal_bins_signed("sin", params, 0.0, 1.0, K=FORCING_BINS)
    total = integrate_temporal_signed("sin", params, 0.0, 1.0)
    assert bins.shape == (FORCING_BINS,)
    assert float(bins.sum()) == pytest.approx(total, abs=1e-3)
    assert np.all(bins >= -1e-12)


def test_build_qL_integral_separable_pulse_train():
    y_grid = np.linspace(0.0, 1.0, 9)
    temporal_params = {"Np": 1, "A_list": [5.0], "t_list": [0.1], "dt_list": [0.4]}
    spatial_params = {"y_c": 0.5, "w": 0.3}

    q_int_fn, s_vec = build_qL_integral(
        "pulse_train", temporal_params, "patch", spatial_params, y_grid
    )
    t_lo, t_hi = 0.0, 0.6
    scalar_int = integrate_temporal_signed("pulse_train", temporal_params, t_lo, t_hi)
    expected = scalar_int * s_vec
    np.testing.assert_allclose(q_int_fn(t_lo, t_hi), expected, atol=1e-12)


def test_build_qL_integral_matches_high_res_trapezoid_on_smooth_sin():
    # On smooth sin the signed integral should agree with a high-resolution
    # trapezoid of q_left(t) — sanity check for the analytic/quadrature path.
    y_grid = np.linspace(0.0, 1.0, 5)
    temporal_params = {"A": 100.0, "f": 2.0, "t_on": 0.0, "t_off": 1.0, "phase": 0.0, "tukey_alpha": 0.2}
    spatial_params = {"y_c": 0.5, "w": 0.6}

    q_left, _ = build_qL("sin", temporal_params, "patch", spatial_params, y_grid)
    q_int_fn, _ = build_qL_integral("sin", temporal_params, "patch", spatial_params, y_grid)

    t_lo, t_hi = 0.1, 0.4
    t_fine = np.linspace(t_lo, t_hi, 2049)
    q_fine = np.array([q_left(float(tk)) for tk in t_fine])
    _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    expected = _trapz(q_fine, t_fine, axis=0)
    got = q_int_fn(t_lo, t_hi)
    np.testing.assert_allclose(got, expected, atol=1e-3, rtol=1e-3)


def _independent_exp_integral(A, t0, tau, t_lo, t_hi):
    if t_hi <= t0:
        return 0.0
    lo = max(float(t_lo), float(t0))
    hi = float(t_hi)
    return float(A) * float(tau) * (
        np.exp(-(lo - float(t0)) / float(tau))
        - np.exp(-(hi - float(t0)) / float(tau))
    )


def _independent_pulse_integral(A_list, t_list, dt_list, t_lo, t_hi):
    total = 0.0
    for A, t0, width in zip(A_list, t_list, dt_list):
        lo = max(float(t_lo), float(t0))
        hi = min(float(t_hi), float(t0) + float(width))
        if hi > lo:
            total += float(A) * (hi - lo)
    return float(total)


def _independent_temporal_integral(family, params, t_lo, t_hi):
    if family == "exp":
        return _independent_exp_integral(
            params["A"], params["t0"], params["tau"], t_lo, t_hi
        )
    if family == "pulse_train":
        return _independent_pulse_integral(
            params["A_list"], params["t_list"], params["dt_list"], t_lo, t_hi
        )
    if family == "exp_train":
        return sum(
            _independent_exp_integral(A, t0, tau, t_lo, t_hi)
            for A, t0, tau in zip(params["A_list"], params["t_list"], params["tau_list"])
        )
    raise ValueError(f"no closed-form independent integral for {family}")


@pytest.mark.parametrize(
    ("spatial_family", "spatial_params"),
    [
        ("uniform", {}),
        ("patch", {"y_c": 0.5, "w": 0.4}),
        ("gaussian", {"y_c": 0.35, "sigma_y": 0.12}),
        ("triangle", {"y_c": 0.65, "ell": 0.25}),
        ("sinusoid", {"c0": 0.7, "c1": 0.3, "f": 2.5, "phase": 0.9}),
    ],
)
@pytest.mark.parametrize(
    ("temporal_family", "temporal_params", "intervals"),
    [
        (
            "sin",
            {"A": 120.0, "f": 3.0, "t_on": 0.0, "t_off": 0.6, "phase": 0.2, "tukey_alpha": 0.25},
            [(0.0, 0.05), (0.04, 0.17), (0.21, 0.44), (0.0, 0.6)],
        ),
        (
            "exp",
            {"A": 130.0, "t0": 0.13, "tau": 0.07},
            [(0.0, 0.08), (0.08, 0.18), (0.18, 0.41), (0.0, 0.6)],
        ),
        (
            "pulse_train",
            {"Np": 2, "A_list": [90.0, -40.0], "t_list": [0.11, 0.33], "dt_list": [0.07, 0.09]},
            [(0.0, 0.08), (0.08, 0.14), (0.14, 0.36), (0.32, 0.45), (0.0, 0.6)],
        ),
        (
            "exp_train",
            {"Np": 2, "A_list": [75.0, 115.0], "t_list": [0.09, 0.31], "tau_list": [0.05, 0.11]},
            [(0.0, 0.08), (0.08, 0.14), (0.14, 0.36), (0.32, 0.5), (0.0, 0.6)],
        ),
    ],
)
def test_build_qL_integral_production_profiles_match_independent_expected(
    temporal_family, temporal_params, intervals, spatial_family, spatial_params
):
    y_grid = np.linspace(0.0, 1.0, 17)
    q_left, s_vec = build_qL(
        temporal_family, temporal_params, spatial_family, spatial_params, y_grid
    )
    q_int_fn, _ = build_qL_integral(
        temporal_family, temporal_params, spatial_family, spatial_params, y_grid
    )

    for t_lo, t_hi in intervals:
        got = q_int_fn(t_lo, t_hi)
        if temporal_family == "sin":
            t_fine = np.linspace(t_lo, t_hi, 8193)
            q_fine = np.array([q_left(float(tk)) for tk in t_fine])
            _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
            expected = _trapz(q_fine, t_fine, axis=0)
            np.testing.assert_allclose(got, expected, atol=2e-4, rtol=2e-4)
        else:
            scalar = _independent_temporal_integral(
                temporal_family, temporal_params, t_lo, t_hi
            )
            np.testing.assert_allclose(got, scalar * s_vec, atol=1e-12, rtol=1e-12)
