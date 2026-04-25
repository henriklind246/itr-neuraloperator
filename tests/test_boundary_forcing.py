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
    spatial_uniform,
    spatial_patch,
    spatial_gaussian,
    spatial_triangle,
    sample_spatial_family,
    sample_temporal_family,
    sample_patch_params,
    sample_gauss_params,
    sample_triangle_params,
    build_qL,
    encode_spatial_params,
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

    def test_temporal_family_sampler_returns_known(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            assert sample_temporal_family(rng) == "sin"

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
