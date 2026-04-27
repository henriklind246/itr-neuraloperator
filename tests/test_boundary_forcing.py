"""Tests for src/physics/boundary_forcing.py."""

import numpy as np
import pytest

from src.physics.boundary_forcing import (
    SPATIAL_FAMILIES,
    PATCH_W_RANGE,
    GAUSS_SIGMA_RANGE,
    GAUSS_CENTER_RANGE,
    TRIANGLE_ELL_RANGE,
    SPATIAL_SAMPLERS,
    SPATIAL_BUILDERS,
    TEMPORAL_FAMILIES,
    TEMPORAL_BUILDERS,
    TEMPORAL_SAMPLERS,
    SIN_AMP_RANGE,
    SIN_FREQ_RANGE,
    PULSE_AMP_RANGE,
    EXP_T0_FRAC,
    TAU_FRAC_LO,
    TAU_EXP_FRAC_HI,
    TAU_TRAIN_FRAC_HI,
    DT_PULSE_FRAC_LO,
    DT_PULSE_FRAC_HI,
    NP_CHOICES,
    NP_MAX,
    PULSE_SLOTS,
    spatial_uniform,
    spatial_patch,
    spatial_gaussian,
    spatial_triangle,
    sample_spatial_family,
    sample_temporal_family,
    sample_patch_params,
    sample_gauss_params,
    sample_triangle_params,
    sample_sin_params,
    sample_exp_params,
    sample_pulse_train_params,
    sample_exp_train_params,
    temporal_exp,
    temporal_pulse_train,
    temporal_exp_train,
    build_qL,
    encode_spatial_params,
    encode_temporal_params,
)
from src.physics.fv_solver_1d import windowed_sin_flux


Y = np.linspace(0.0, 1.0, 101)

# ------- SPATIAL PROFILES -------

class TestSpatialProfiles:

    def test_uniform_is_ones(self):
        s = spatial_uniform(Y)
        assert s.shape == Y.shape
        assert np.allclose(s, 1.0)
        assert s.max() == 1.0

    def test_patch_support_and_peak(self):
        y_c, w = 0.5, 0.2
        s = spatial_patch(Y, y_c=y_c, w=w)
        assert s.shape == Y.shape
        # Inside support: equal to 1
        inside = (Y >= y_c - w / 2) & (Y <= y_c + w / 2)
        assert np.allclose(s[inside], 1.0)
        # Outside support: equal to 0
        assert np.allclose(s[~inside], 0.0)
        assert s.max() == 1.0

    def test_gaussian_peak_on_grid_node(self):
        # Pick y_c that lies exactly on the grid (Y has 0.5 since 101 points span [0, 1]).
        y_c, sigma_y = 0.5, 0.1
        assert 0.5 in Y
        s = spatial_gaussian(Y, y_c=y_c, sigma_y=sigma_y)
        assert s.shape == Y.shape
        # Peak value at y_c is exactly 1
        idx = int(np.where(Y == y_c)[0][0])
        assert s[idx] == pytest.approx(1.0, abs=1e-12)

    def test_gaussian_peak_off_grid_node(self):
        # y_c between two grid nodes — discrete max < 1 by a bounded amount.
        h = Y[1] - Y[0]
        y_c, sigma_y = 0.5 + 0.5 * h, 0.1
        s = spatial_gaussian(Y, y_c=y_c, sigma_y=sigma_y)
        # Worst-case offset between y_c and nearest node is h/2.
        bound = float(np.exp(-((h / 2) ** 2) / (2 * sigma_y ** 2)))
        assert s.max() <= 1.0
        assert s.max() >= bound - 1e-12
        # Discrete maximum occurs at the node nearest y_c.
        nearest = int(np.argmin(np.abs(Y - y_c)))
        assert int(np.argmax(s)) == nearest

    def test_gaussian_symmetric_decay(self):
        y_c, sigma_y = 0.5, 0.1
        s = spatial_gaussian(Y, y_c=y_c, sigma_y=sigma_y)
        # Symmetric around y=0.5
        assert np.allclose(s, s[::-1], atol=1e-12)

    def test_triangle_support_and_peak(self):
        y_c, ell = 0.5, 0.2
        s = spatial_triangle(Y, y_c=y_c, ell=ell)
        assert s.shape == Y.shape
        # Outside support strictly zero
        outside = (Y < y_c - ell) | (Y > y_c + ell)
        assert np.allclose(s[outside], 0.0)
        # Linear decay: at y = y_c ± ell/2, value = 0.5
        for y_query in (y_c - ell / 2, y_c + ell / 2):
            idx = int(np.argmin(np.abs(Y - y_query)))
            assert s[idx] == pytest.approx(0.5, abs=1e-2)
        # Peak at y_c is 1
        idx_c = int(np.argmin(np.abs(Y - y_c)))
        assert s[idx_c] == pytest.approx(1.0, abs=1e-12)

# --------- SAMPLERS ---------

class TestSamplers:

    def test_patch_sampler_in_bounds(self):
        rng = np.random.default_rng(0)
        for _ in range(1000):
            p = sample_patch_params(rng)
            assert PATCH_W_RANGE[0] <= p["w"] <= PATCH_W_RANGE[1]
            assert p["y_c"] - p["w"] / 2 >= 0.0 - 1e-12
            assert p["y_c"] + p["w"] / 2 <= 1.0 + 1e-12

    def test_triangle_sampler_in_bounds(self):
        rng = np.random.default_rng(0)
        for _ in range(1000):
            p = sample_triangle_params(rng)
            assert TRIANGLE_ELL_RANGE[0] <= p["ell"] <= TRIANGLE_ELL_RANGE[1]
            assert p["y_c"] - p["ell"] >= 0.0 - 1e-12
            assert p["y_c"] + p["ell"] <= 1.0 + 1e-12

    def test_gauss_sampler_center_and_sigma_bounds(self):
        rng = np.random.default_rng(0)
        for _ in range(1000):
            p = sample_gauss_params(rng)
            assert GAUSS_CENTER_RANGE[0] <= p["y_c"] <= GAUSS_CENTER_RANGE[1]
            assert GAUSS_SIGMA_RANGE[0] - 1e-12 <= p["sigma_y"] <= GAUSS_SIGMA_RANGE[1] + 1e-12

    def test_spatial_family_sampler_covers_all(self):
        rng = np.random.default_rng(0)
        seen = {sample_spatial_family(rng) for _ in range(1000)}
        assert seen == set(SPATIAL_FAMILIES.keys())

    def test_temporal_family_sampler_covers_all(self):
        rng = np.random.default_rng(0)
        seen = {sample_temporal_family(rng) for _ in range(1000)}
        assert seen == set(TEMPORAL_FAMILIES.keys())

# --------- FLUX BUILDER ---------

class TestBuildQL:

    def _temp_params(self):
        return dict(A=50.0, f=2.0, t_on=0.0, t_off=0.2, phase=0.0, tukey_alpha=0.5)

    def test_qL_shape_uniform(self):
        q_left, s_vec = build_qL("sin", self._temp_params(),
                                 "uniform", {}, Y)
        assert s_vec.shape == Y.shape
        assert q_left(0.1).shape == Y.shape

    def test_qL_shape_patch(self):
        q_left, s_vec = build_qL("sin", self._temp_params(),
                                 "patch", {"y_c": 0.5, "w": 0.3}, Y)
        assert s_vec.shape == Y.shape
        assert q_left(0.1).shape == Y.shape

    def test_separability(self):
        # For fixed (a, s), q_L(y, t1)/q_L(y, t2) = a(t1)/a(t2) at every y where s != 0.
        q_left, s_vec = build_qL("sin", self._temp_params(),
                                 "gaussian", {"y_c": 0.5, "sigma_y": 0.1}, Y)
        t1, t2 = 0.05, 0.15
        q1, q2 = q_left(t1), q_left(t2)
        nz = s_vec > 1e-12
        ratios = q1[nz] / q2[nz]
        # All ratios identical (== a(t1)/a(t2))
        assert np.allclose(ratios, ratios[0], atol=1e-12)

    def test_uniform_reproduces_scalar_flux(self):
        # uniform spatial × sin temporal == scalar windowed_sin_flux broadcast to (Ny,)
        tp = self._temp_params()
        q_left, _ = build_qL("sin", tp, "uniform", {}, Y)
        scalar_q = windowed_sin_flux(tp["f"], tp["A"], tp["t_on"], tp["t_off"],
                                     tp["phase"], tp["tukey_alpha"])
        for t in (0.0, 0.05, 0.1, 0.15, 0.2, 0.25):
            scalar_val = scalar_q(t)
            vec_val = q_left(t)
            assert vec_val.shape == Y.shape
            assert np.allclose(vec_val, scalar_val * np.ones_like(Y), atol=1e-14)

# -------- PARAMETER ENCODING ---------

class TestEncodeSpatialParams:

    def test_uniform_all_zeros(self):
        enc = encode_spatial_params("uniform", {})
        assert np.allclose(enc, [0.0, 0.0, 0.0, 0.0])

    def test_patch_zero_padding(self):
        enc = encode_spatial_params("patch", {"y_c": 0.4, "w": 0.2})
        assert enc[0] == 0.4 and enc[1] == 0.2
        assert enc[2] == 0.0 and enc[3] == 0.0

    def test_gaussian_zero_padding(self):
        enc = encode_spatial_params("gaussian", {"y_c": 0.6, "sigma_y": 0.05})
        assert enc[0] == 0.6 and enc[2] == 0.05
        assert enc[1] == 0.0 and enc[3] == 0.0

    def test_triangle_zero_padding(self):
        enc = encode_spatial_params("triangle", {"y_c": 0.5, "ell": 0.1})
        assert enc[0] == 0.5 and enc[3] == 0.1
        assert enc[1] == 0.0 and enc[2] == 0.0

    def test_unknown_family_raises(self):
        with pytest.raises(ValueError):
            encode_spatial_params("bogus", {})


# --------- TEMPORAL BUILDERS ---------

DT = 0.005
T_FINAL = 0.3


class TestTemporalBuilders:

    def test_exp_zero_before_t0_then_decay(self):
        q = temporal_exp(A=200.0, t0=0.05, tau=0.1)
        assert q(0.0) == 0.0
        assert q(0.04999) == 0.0
        assert q(0.05) == pytest.approx(200.0)
        assert q(0.05 + 0.1) == pytest.approx(200.0 * np.exp(-1.0))
        assert q(0.05 + 0.2) == pytest.approx(200.0 * np.exp(-2.0))

    def test_pulse_train_active_window_and_overlap(self):
        # Two non-overlapping pulses
        q = temporal_pulse_train(A_list=[100.0, 200.0],
                                 t_list=[0.0, 0.2],
                                 dt_list=[0.05, 0.05])
        assert q(0.0) == 100.0
        assert q(0.04) == 100.0
        # Right-open interval — at t == t_n + dt_n the pulse is off
        assert q(0.05) == 0.0
        assert q(0.10) == 0.0
        assert q(0.20) == 200.0
        assert q(0.249) == 200.0
        assert q(0.25) == 0.0

        # Overlapping pulses sum amplitudes
        q2 = temporal_pulse_train(A_list=[100.0, 50.0],
                                  t_list=[0.0, 0.02],
                                  dt_list=[0.05, 0.05])
        assert q2(0.03) == pytest.approx(150.0)

    def test_exp_train_components_sum(self):
        q = temporal_exp_train(A_list=[100.0, 200.0],
                               t_list=[0.0, 0.1],
                               tau_list=[0.05, 0.05])
        # Before any pulse activates
        assert q(-0.01) == 0.0
        # Only first pulse active, exact value
        assert q(0.0) == pytest.approx(100.0)
        # After 0.1, both pulses active
        v = 100.0 * np.exp(-0.1 / 0.05) + 200.0 * np.exp(0.0)
        assert q(0.1) == pytest.approx(v)

    def test_all_builders_finite_scalar(self):
        rng = np.random.default_rng(0)
        for fam in TEMPORAL_FAMILIES:
            p = TEMPORAL_SAMPLERS[fam](rng, dt=DT, t_final=T_FINAL,
                                       t_on=0.0, t_off=0.2)
            q = TEMPORAL_BUILDERS[fam](**p)
            for t in np.linspace(0.0, T_FINAL, 11):
                v = q(float(t))
                assert np.isfinite(v)
                assert isinstance(v, float)


# --------- TEMPORAL SAMPLERS ---------

class TestTemporalSamplers:

    def test_sin_in_bounds(self):
        rng = np.random.default_rng(0)
        for _ in range(1000):
            p = sample_sin_params(rng, dt=DT, t_final=T_FINAL,
                                  t_on=0.0, t_off=0.2)
            assert SIN_AMP_RANGE[0] <= p["A"] <= SIN_AMP_RANGE[1]
            assert SIN_FREQ_RANGE[0] - 1e-12 <= p["f"] <= SIN_FREQ_RANGE[1] + 1e-12
            assert p["t_on"] == 0.0
            assert p["t_off"] == 0.2

    def test_exp_in_bounds(self):
        rng = np.random.default_rng(0)
        tau_lo = TAU_FRAC_LO * DT
        tau_hi = TAU_EXP_FRAC_HI * T_FINAL
        for _ in range(1000):
            p = sample_exp_params(rng, dt=DT, t_final=T_FINAL)
            assert PULSE_AMP_RANGE[0] <= p["A"] <= PULSE_AMP_RANGE[1]
            assert 0.0 <= p["t0"] <= EXP_T0_FRAC * T_FINAL
            assert tau_lo - 1e-12 <= p["tau"] <= tau_hi + 1e-12

    def test_pulse_train_in_bounds_and_sorted(self):
        rng = np.random.default_rng(0)
        dt_lo = DT_PULSE_FRAC_LO * DT
        dt_hi = DT_PULSE_FRAC_HI * T_FINAL
        for _ in range(1000):
            p = sample_pulse_train_params(rng, dt=DT, t_final=T_FINAL)
            Np = p["Np"]
            assert Np in NP_CHOICES
            assert len(p["A_list"]) == Np
            assert len(p["t_list"]) == Np
            assert len(p["dt_list"]) == Np
            assert p["t_list"] == sorted(p["t_list"])
            for n in range(Np):
                assert PULSE_AMP_RANGE[0] <= p["A_list"][n] <= PULSE_AMP_RANGE[1]
                assert dt_lo - 1e-12 <= p["dt_list"][n] <= dt_hi + 1e-12
                assert 0.0 <= p["t_list"][n] <= T_FINAL - p["dt_list"][n] + 1e-12

    def test_exp_train_in_bounds_and_sorted(self):
        rng = np.random.default_rng(0)
        tau_lo = TAU_FRAC_LO * DT
        tau_hi = TAU_TRAIN_FRAC_HI * T_FINAL
        for _ in range(1000):
            p = sample_exp_train_params(rng, dt=DT, t_final=T_FINAL)
            Np = p["Np"]
            assert Np in NP_CHOICES
            assert len(p["A_list"]) == len(p["t_list"]) == len(p["tau_list"]) == Np
            assert p["t_list"] == sorted(p["t_list"])
            for n in range(Np):
                assert PULSE_AMP_RANGE[0] <= p["A_list"][n] <= PULSE_AMP_RANGE[1]
                assert 0.0 <= p["t_list"][n] <= EXP_T0_FRAC * T_FINAL
                assert tau_lo - 1e-12 <= p["tau_list"][n] <= tau_hi + 1e-12

    def test_sin_frequency_log_uniform(self):
        # log-uniform sampler should produce a roughly uniform distribution in log-space
        rng = np.random.default_rng(0)
        fs = np.array([sample_sin_params(rng, dt=DT, t_final=T_FINAL,
                                         t_on=0.0, t_off=0.2)["f"]
                       for _ in range(20000)])
        log_fs = np.log(fs)
        log_lo = np.log(SIN_FREQ_RANGE[0])
        log_hi = np.log(SIN_FREQ_RANGE[1])
        # Check that mean is close to the log-uniform mean (midpoint of log range)
        assert log_fs.mean() == pytest.approx(0.5 * (log_lo + log_hi), abs=0.05)


# --------- TEMPORAL PARAM ENCODING ---------

class TestEncodeTemporalParams:

    def test_sin_encoding(self):
        # mid-range A and f
        A = 0.5 * (SIN_AMP_RANGE[0] + SIN_AMP_RANGE[1])
        f = float(np.sqrt(SIN_FREQ_RANGE[0] * SIN_FREQ_RANGE[1]))  # log-mid
        out = encode_temporal_params("sin", {"A": A, "f": f}, dt=DT, t_final=T_FINAL)
        assert out.shape == (1 + PULSE_SLOTS * 3,)
        assert out[0] == 0.0
        assert out[1] == pytest.approx(0.5)
        assert out[2] == pytest.approx(0.5, abs=1e-12)
        # Slots beyond first pulse must be zero
        assert np.allclose(out[3:], 0.0)

    def test_exp_encoding(self):
        t0 = 0.5 * EXP_T0_FRAC * T_FINAL
        tau_lo = TAU_FRAC_LO * DT
        tau_hi = TAU_EXP_FRAC_HI * T_FINAL
        tau = float(np.sqrt(tau_lo * tau_hi))
        A = 0.5 * (PULSE_AMP_RANGE[0] + PULSE_AMP_RANGE[1])
        out = encode_temporal_params("exp", {"A": A, "t0": t0, "tau": tau},
                                     dt=DT, t_final=T_FINAL)
        assert out[0] == 0.0
        # slot 0 = (t0_norm, tau_norm_log, A_norm)
        assert out[1] == pytest.approx(t0 / T_FINAL)
        assert out[2] == pytest.approx(0.5, abs=1e-12)
        assert out[3] == pytest.approx(0.5)
        # other slots zero
        assert np.allclose(out[4:], 0.0)

    def test_pulse_train_encoding_partial(self):
        # Np=2 — slots 0 and 1 populated, slots 2 and 3 zero
        Np = 2
        params = {
            "Np": Np,
            "A_list":  [50.0, 300.0],
            "t_list":  [0.05, 0.20],
            "dt_list": [DT_PULSE_FRAC_LO * DT, DT_PULSE_FRAC_HI * T_FINAL],
        }
        out = encode_temporal_params("pulse_train", params, dt=DT, t_final=T_FINAL)
        assert out[0] == pytest.approx((Np - 1) / (NP_MAX - 1))
        # slot 0
        assert out[1] == pytest.approx(0.05 / T_FINAL)
        assert out[2] == pytest.approx(0.0)
        assert out[3] == pytest.approx(0.0, abs=1e-12)
        # slot 1
        assert out[4] == pytest.approx(0.20 / T_FINAL)
        assert out[5] == pytest.approx(1.0)
        assert out[6] == pytest.approx(1.0, abs=1e-12)
        # slots 2, 3 unused
        assert np.allclose(out[7:], 0.0)

    def test_exp_train_encoding_full(self):
        Np = NP_MAX
        tau_lo = TAU_FRAC_LO * DT
        tau_hi = TAU_TRAIN_FRAC_HI * T_FINAL
        params = {
            "Np": Np,
            "A_list":   [PULSE_AMP_RANGE[0]] * Np,
            "t_list":   [0.0, 0.05, 0.1, 0.15],
            "tau_list": [tau_lo, tau_lo, tau_hi, tau_hi],
        }
        out = encode_temporal_params("exp_train", params, dt=DT, t_final=T_FINAL)
        assert out[0] == pytest.approx(1.0)
        # All Np slots populated; check the tau normalization slot
        assert out[3] == pytest.approx(0.0, abs=1e-12)
        assert out[6] == pytest.approx(0.0, abs=1e-12)
        assert out[9] == pytest.approx(1.0, abs=1e-12)
        assert out[12] == pytest.approx(1.0, abs=1e-12)

    def test_unknown_temporal_family_raises(self):
        with pytest.raises(ValueError):
            encode_temporal_params("bogus", {}, dt=DT, t_final=T_FINAL)


# --------- BUILD_QL ALL FAMILIES ---------

class TestBuildQLAllFamilies:

    def test_all_temporal_families_with_uniform(self):
        rng = np.random.default_rng(0)
        for fam in TEMPORAL_FAMILIES:
            p = TEMPORAL_SAMPLERS[fam](rng, dt=DT, t_final=T_FINAL,
                                       t_on=0.0, t_off=0.2)
            q_left, s_vec = build_qL(fam, p, "uniform", {}, Y)
            assert s_vec.shape == Y.shape
            for t in (0.0, 0.5 * T_FINAL, T_FINAL):
                v = q_left(float(t))
                assert v.shape == Y.shape
                assert np.all(np.isfinite(v))
