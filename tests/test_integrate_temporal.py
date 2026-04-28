import numpy as np
import pytest

from src.physics.boundary_forcing import (
    FORCING_BINS,
    integrate_temporal,
    integrate_temporal_bins,
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
