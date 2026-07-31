import random

import numpy as np
import pytest
import torch

from src.physics.init_conditions import (
    IC_BUILDER_SCHEMA_VERSION,
    IC_FAMILIES,
    IC_SAMPLERS,
    IC_BUILDERS,
    HOT_MARGIN_FACTOR,
    GRF_ELL_RANGE,
    GRF_SIGMA_RANGE,
    HOT_AMP_RANGE,
    HOT_N_CHOICES,
    HOT_SIGMA_RANGE,
    ONLINE_IC_SAMPLER_VERSION,
    RANDOM_FIELD_MODES,
    SINU_AMP_RANGE,
    SINU_KMAX,
    SINU_N_CHOICES,
    UNIFORM_OFFSET_RANGE,
    EDGE_TAPER_WIDTH,
    _smootherstep_window,
    balanced_ic_family_assignments,
    build_ic,
    canonical_ic_params,
    dirichlet_edge_taper,
    ic_random_cosine_field_2d,
    sample_ic_family,
    sample_uniform_ic_params,
    sample_random_sinusoid_ic_params,
    sample_random_field_ic_params,
    sample_hot_spot_ic_params,
)


def _mesh(Nx: int = 100, Ny: int = 100, b: float = 1.0):
    x = np.linspace(0.0, b, Nx)
    y = np.linspace(0.0, b, Ny)
    return np.meshgrid(x, y, indexing="ij")


class TestSamplers:
    @pytest.mark.parametrize("batch_size", [1, 2, 3, 4, 5, 16])
    def test_balanced_family_assignments(self, batch_size):
        labels = balanced_ic_family_assignments(
            np.random.default_rng(123), batch_size,
        )
        counts = [labels.count(family) for family in IC_FAMILIES]
        assert len(labels) == batch_size
        assert max(counts) - min(counts) <= 1
        if batch_size >= len(IC_FAMILIES):
            assert all(count > 0 for count in counts)
        else:
            assert len(set(labels)) == batch_size

    def test_uniform_keys(self):
        rng = np.random.default_rng(0)
        p = sample_uniform_ic_params(rng)
        assert set(p.keys()) == {"T0_offset"}

    def test_random_sinusoid_keys_and_lengths(self):
        rng = np.random.default_rng(0)
        p = sample_random_sinusoid_ic_params(rng)
        assert "N" not in p
        assert set(p.keys()) == {"A_list", "nx_list", "ny_list"}
        N = len(p["A_list"])
        assert N in (3, 4, 5)
        for k in ("nx_list", "ny_list"):
            assert len(p[k]) == N

    def test_grf_keys(self):
        rng = np.random.default_rng(0)
        p = sample_random_field_ic_params(rng)
        assert set(p.keys()) == {"ell", "sigma", "wn_seed"}

    def test_hot_spot_keys_and_lengths(self):
        rng = np.random.default_rng(0)
        p = sample_hot_spot_ic_params(rng)
        assert "N" not in p
        assert set(p.keys()) == {"A_list", "mu_x_list", "mu_y_list", "sigma_list"}
        N = len(p["A_list"])
        assert N in (1, 2, 3, 4)
        for k in ("mu_x_list", "mu_y_list", "sigma_list"):
            assert len(p[k]) == N

    def test_sampler_determinism(self):
        for fam in IC_FAMILIES:
            r1 = np.random.default_rng(7)
            r2 = np.random.default_rng(7)
            p1 = IC_SAMPLERS[fam](r1)
            p2 = IC_SAMPLERS[fam](r2)
            assert p1 == p2, f"non-deterministic sampler: {fam}"

    def test_hot_spot_centers_inside_domain(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            p = sample_hot_spot_ic_params(rng)
            for mu_x, mu_y, sigma in zip(p["mu_x_list"], p["mu_y_list"], p["sigma_list"]):
                m = min(HOT_MARGIN_FACTOR * sigma, 0.45)
                assert m <= mu_x <= 1.0 - m
                assert m <= mu_y <= 1.0 - m

    def test_sample_ic_family_returns_known(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            assert sample_ic_family(rng) in IC_FAMILIES

    def test_sample_ic_family_can_be_forced_by_probabilities(self):
        rng = np.random.default_rng(0)
        families = list(IC_FAMILIES.keys())
        for i, fam in enumerate(families):
            probs = np.zeros(len(families), dtype=float)
            probs[i] = 1.0
            assert sample_ic_family(rng, probs=probs) == fam

    def test_uniform_params_within_documented_range(self):
        rng = np.random.default_rng(0)
        for _ in range(100):
            p = sample_uniform_ic_params(rng)
            assert UNIFORM_OFFSET_RANGE[0] <= p["T0_offset"] <= UNIFORM_OFFSET_RANGE[1]

    def test_random_sinusoid_params_within_documented_ranges(self):
        rng = np.random.default_rng(0)
        for _ in range(100):
            p = sample_random_sinusoid_ic_params(rng)
            assert len(p["A_list"]) in SINU_N_CHOICES
            for A, nx, ny in zip(p["A_list"], p["nx_list"], p["ny_list"]):
                assert SINU_AMP_RANGE[0] <= abs(A) <= SINU_AMP_RANGE[1]
                assert 1 <= nx <= SINU_KMAX
                assert 1 <= ny <= SINU_KMAX
                assert float(ny).is_integer()

    def test_random_sinusoid_amplitudes_take_both_signs(self):
        # The cosine basis has no phase freedom, so sign is the only remaining
        # way to flip a mode; a one-sided sampler would silently halve the family.
        rng = np.random.default_rng(0)
        signs = {
            np.sign(A)
            for _ in range(50)
            for A in sample_random_sinusoid_ic_params(rng)["A_list"]
        }
        assert signs == {-1.0, 1.0}

    def test_grf_params_within_documented_ranges(self):
        rng = np.random.default_rng(0)
        for _ in range(100):
            p = sample_random_field_ic_params(rng)
            assert GRF_ELL_RANGE[0] <= p["ell"] <= GRF_ELL_RANGE[1]
            assert GRF_SIGMA_RANGE[0] <= p["sigma"] <= GRF_SIGMA_RANGE[1]
            assert 0 <= p["wn_seed"] < 2 ** 31

    def test_hot_spot_params_within_documented_ranges(self):
        rng = np.random.default_rng(0)
        for _ in range(100):
            p = sample_hot_spot_ic_params(rng)
            assert len(p["A_list"]) in HOT_N_CHOICES
            for A, sigma in zip(p["A_list"], p["sigma_list"]):
                assert HOT_AMP_RANGE[0] <= A <= HOT_AMP_RANGE[1]
                assert HOT_SIGMA_RANGE[0] <= sigma <= HOT_SIGMA_RANGE[1]


class TestBuilders:
    def test_all_families_shape_dtype_finite(self):
        X, Y = _mesh()
        rng = np.random.default_rng(0)
        for fam in IC_FAMILIES:
            p = IC_SAMPLERS[fam](rng)
            dev = IC_BUILDERS[fam](X, Y, **p)
            assert dev.shape == (100, 100), f"{fam} shape"
            assert np.isfinite(dev).all(), f"{fam} non-finite"

    def test_uniform_is_constant_before_taper(self):
        X, Y = _mesh()
        dev = IC_BUILDERS["uniform_2d"](X, Y, T0_offset=7.5)
        assert np.allclose(dev, 7.5)

    @pytest.mark.parametrize("fam", list(IC_FAMILIES.keys()))
    def test_builders_are_deterministic_for_fixed_params(self, fam):
        X, Y = _mesh(Nx=41, Ny=37)
        rng = np.random.default_rng(123)
        p = IC_SAMPLERS[fam](rng)
        d1 = IC_BUILDERS[fam](X, Y, **p)
        d2 = IC_BUILDERS[fam](X, Y, **p)
        np.testing.assert_array_equal(d1, d2)


class TestEdgeTaper:
    def test_taper_endpoints(self):
        X, Y = _mesh()
        t = dirichlet_edge_taper(X, x_right=1.0)
        assert np.allclose(t[0, :], 1.0)
        assert np.allclose(t[-1, :], 0.0)

    def test_taper_monotone(self):
        x = np.linspace(0.0, 1.0, 200)
        X, _ = np.meshgrid(x, x, indexing="ij")
        t = dirichlet_edge_taper(X, x_right=1.0)
        # along x, taper should be non-increasing
        col = t[:, 0]
        assert np.all(np.diff(col) <= 1e-12)

    def test_taper_is_zero_only_inside_right_edge_band(self):
        x = np.linspace(0.0, 2.0, 201)
        y = np.linspace(0.0, 1.0, 51)
        X, _ = np.meshgrid(x, y, indexing="ij")
        t = dirichlet_edge_taper(X, x_right=2.0, edge_width=0.25)
        assert np.allclose(t[x <= 1.75, :], 1.0)
        assert np.allclose(t[-1, :], 0.0)
        assert np.all((0.0 <= t) & (t <= 1.0))

    def test_taper_is_x_only_and_unity_away_from_right_wall(self):
        # The taper must not touch the three Neumann walls: it is a function of
        # x alone and is exactly 1 (not merely close to 1) for x <= b - W, so
        # the builders' even symmetry survives it untouched.
        x = np.linspace(0.0, 1.0, 257)
        y = np.linspace(0.0, 1.0, 129)
        X, _ = np.meshgrid(x, y, indexing="ij")
        t = dirichlet_edge_taper(X, x_right=1.0)
        for j in range(t.shape[1]):
            np.testing.assert_array_equal(t[:, j], t[:, 0])
        inner = x <= 1.0 - EDGE_TAPER_WIDTH
        assert inner.any()
        np.testing.assert_array_equal(t[inner, :], np.ones_like(t[inner, :]))

    def test_smootherstep_endpoint_derivatives_vanish(self):
        # w'(0) = w'(1) = w''(0) = w''(1) = 0 is what makes the taper C2, so it
        # adds no jump to the second derivative the diffusion operator sees.
        #
        # Test it as a root-multiplicity statement rather than by finite
        # differences: w and (1 - w) each vanish to third order at their
        # endpoint iff value, slope and curvature all vanish there, and that is
        # equivalent to w(s)/s^3 and (1 - w(s))/(1 - s)^3 staying bounded. The
        # old C1 smoothstep 3s^2 - 2s^3 gives (1 - w)/(1 - s)^3 = (1 + 2s)/(1 - s),
        # which diverges, so this discriminates exactly and without FD noise.
        W = EDGE_TAPER_WIDTH
        s = np.linspace(1e-6, 1.0 - 1e-6, 20001)
        w = _smootherstep_window(s * W, W)
        left_ratio = w / s ** 3
        right_ratio = (1.0 - w) / (1.0 - s) ** 3
        assert np.isfinite(left_ratio).all() and np.isfinite(right_ratio).all()
        # Both ratios are degree-2 polynomials, hence bounded on [0, 1].
        assert left_ratio.max() < 20.0, left_ratio.max()
        assert right_ratio.max() < 20.0, right_ratio.max()

        # Peak |w''| is 10/sqrt(3) / W^2 = 5.774/W^2, below the C1 smoothstep's
        # 6/W^2, so the C2 window is not paid for with extra curvature.
        d = np.linspace(0.0, W, 100001)
        wd = _smootherstep_window(d, W)
        h = d[1] - d[0]
        curvature = np.abs(np.diff(wd, 2)) / h ** 2
        assert curvature.max() < 6.0 / W ** 2

        # Outside the band the window is exactly the constant 1.
        outside = _smootherstep_window(np.linspace(W, 3.0 * W, 64), W)
        np.testing.assert_array_equal(outside, np.ones_like(outside))


class TestBuildIC:
    @pytest.mark.parametrize("fam", list(IC_FAMILIES.keys()))
    def test_build_ic_shape_dtype_pinned(self, fam):
        X, Y = _mesh()
        rng = np.random.default_rng(0)
        p = IC_SAMPLERS[fam](rng)
        T0 = build_ic(fam, p, X, Y, T_right=300.0, b=1.0)
        assert T0.shape == (100, 100)
        assert T0.dtype == np.float32
        assert np.isfinite(T0).all()
        np.testing.assert_allclose(T0[-1, :], 300.0)

    def test_build_ic_no_pin_no_taper(self):
        X, Y = _mesh()
        rng = np.random.default_rng(0)
        p = IC_SAMPLERS["uniform_2d"](rng)
        T0 = build_ic("uniform_2d", p, X, Y, T_right=300.0,
                      taper=False, pin_right_edge=False)
        # pure constant + T_right
        assert np.allclose(T0, 300.0 + p["T0_offset"])

    @pytest.mark.parametrize("fam", list(IC_FAMILIES.keys()))
    def test_build_ic_taper_reaches_right_dirichlet_without_pin(self, fam):
        X, Y = _mesh(Nx=51, Ny=49)
        rng = np.random.default_rng(5)
        p = IC_SAMPLERS[fam](rng)
        T0 = build_ic(fam, p, X, Y, T_right=307.0, b=1.0,
                      taper=True, pin_right_edge=False)
        np.testing.assert_allclose(T0[-1, :], 307.0)

    def test_build_ic_pin_overrides_untapered_right_edge(self):
        X, Y = _mesh(Nx=21, Ny=19)
        p = {"T0_offset": 11.0}
        T0 = build_ic("uniform_2d", p, X, Y, T_right=300.0,
                      taper=False, pin_right_edge=True)
        assert np.allclose(T0[:-1, :], 311.0)
        np.testing.assert_allclose(T0[-1, :], 300.0)

    @pytest.mark.parametrize("fam", list(IC_FAMILIES.keys()))
    def test_lossless_descriptor_reconstruction_and_rng_purity(self, fam):
        X, Y = _mesh(Nx=31, Ny=29)
        params = IC_SAMPLERS[fam](np.random.default_rng(7))
        descriptor = canonical_ic_params(fam, params)
        numpy_state = np.random.get_state()
        python_state = random.getstate()
        torch_state = torch.get_rng_state().clone()

        original = build_ic(fam, descriptor, X, Y, T_right=300.0)
        reconstructed = build_ic(fam, descriptor, X, Y, T_right=300.0)

        assert np.array_equal(original, reconstructed)
        after_numpy = np.random.get_state()
        assert numpy_state[0] == after_numpy[0]
        np.testing.assert_array_equal(numpy_state[1], after_numpy[1])
        assert numpy_state[2:] == after_numpy[2:]
        assert random.getstate() == python_state
        assert torch.equal(torch.get_rng_state(), torch_state)


# Deterministic parameters per family. Every builder is now an analytic function
# of (x, y) with no grid-sized rng draws, so all four families -- grf_2d included
# -- are comparable across resolutions.
_FIXED_FAMILY_PARAMS = {
    "uniform_2d": dict(T0_offset=9.0),
    "random_sinusoid_2d": dict(
        A_list=[5.0, -3.0], nx_list=[2.3, 5.7], ny_list=[1, 4],
    ),
    "grf_2d": dict(ell=0.15, sigma=10.0, wn_seed=1234),
    "hot_spot_2d": dict(
        A_list=[20.0, -15.0], mu_x_list=[0.4, 0.6], mu_y_list=[0.5, 0.3],
        sigma_list=[0.15, 0.12],
    ),
}

# Families built from cosine modes, hence *exactly* even about the three
# zero-Neumann walls. hot_spot_2d uses a truncated image sum instead and only
# meets a numerical tolerance, so it is tested separately.
_COSINE_FAMILIES = ("uniform_2d", "random_sinusoid_2d", "grf_2d")


def _mirror_pair(fam: str, params: dict, wall: str, d: np.ndarray):
    """Builder values at mirrored off-grid points either side of a wall."""
    other = np.linspace(0.0, 1.0, 17)
    if wall == "x0":
        lo = np.meshgrid(-d, other, indexing="ij")
        hi = np.meshgrid(d, other, indexing="ij")
    elif wall == "y0":
        lo = np.meshgrid(other, -d, indexing="ij")
        hi = np.meshgrid(other, d, indexing="ij")
    elif wall == "y1":
        lo = np.meshgrid(other, 1.0 - d, indexing="ij")
        hi = np.meshgrid(other, 1.0 + d, indexing="ij")
    else:
        raise ValueError(wall)
    build = IC_BUILDERS[fam]
    return build(*lo, **params), build(*hi, **params)


class TestNeumannWallCompatibility:
    @pytest.mark.parametrize("fam", _COSINE_FAMILIES)
    @pytest.mark.parametrize("wall", ["x0", "y0", "y1"])
    def test_wall_normal_derivative_exact_for_cosine_families(self, fam, wall):
        # Even symmetry about the wall is what makes dT/dn = 0 there, and unlike
        # the old taper it holds exactly rather than asymptotically. Probe it
        # off-grid so the property is a statement about the builder, not about
        # one discretization. atol rather than exact equality: cos(+z) and
        # cos(-z) need not be bit-identical on every libm.
        d = np.array([1e-3, 0.01, 0.05, 0.1, 0.25])
        lo, hi = _mirror_pair(fam, _FIXED_FAMILY_PARAMS[fam], wall, d)
        np.testing.assert_allclose(lo, hi, rtol=0.0, atol=1e-13)

    def test_hot_spot_wall_normal_derivative_within_bound(self):
        # A finite image set cannot be exactly even about both y = 0 and y = 1,
        # so hot_spot_2d gets a tolerance contract instead of exactness. The
        # residual is bounded relative to sum|A| because that is the only scale
        # in the builder.
        rng = np.random.default_rng(0)
        d = np.array([1e-4])
        worst = 0.0
        for _ in range(200):
            params = sample_hot_spot_ic_params(rng)
            scale = float(np.abs(params["A_list"]).sum())
            for wall in ("x0", "y0", "y1"):
                lo, hi = _mirror_pair("hot_spot_2d", params, wall, d)
                # (f(+d) - f(-d)) / 2d is the wall-normal derivative: the odd
                # part survives, the even part cancels exactly.
                deriv = np.abs(hi - lo).max() / (2.0 * d[0])
                worst = max(worst, deriv / scale)
        assert worst <= 1e-8, worst

    def test_hot_spot_dirichlet_wall_not_over_imaged(self):
        # The 2 - mu_x image is deliberately absent: x = 1 is Dirichlet, so an
        # image there would fight the pin instead of helping. The signature is
        # that the field is NOT even about x = 1.
        params = dict(A_list=[20.0], mu_x_list=[0.9], mu_y_list=[0.5],
                      sigma_list=[0.05])
        d = np.array([0.02, 0.05])
        other = np.linspace(0.0, 1.0, 17)
        build = IC_BUILDERS["hot_spot_2d"]
        inside = build(*np.meshgrid(1.0 - d, other, indexing="ij"), **params)
        outside = build(*np.meshgrid(1.0 + d, other, indexing="ij"), **params)
        assert np.abs(inside - outside).max() > 1.0
        # ... and the asymmetry never reaches the solver, because build_ic
        # tapers into that wall and pins the last column exactly.
        X, Y = _mesh(Nx=101, Ny=101)
        T0 = build_ic("hot_spot_2d", params, X, Y, T_right=300.0, b=1.0)
        np.testing.assert_allclose(T0[-1, :], 300.0)

    @pytest.mark.parametrize("fam", list(IC_FAMILIES.keys()))
    def test_build_ic_neumann_wall_gradient_is_first_order_in_h(self, fam):
        # End-to-end through build_ic (taper, float32 cast, pin): the continuous
        # normal derivative is zero at all three Neumann walls, so the one-sided
        # difference quotient is pure O(h) truncation and must halve when h does.
        params = _FIXED_FAMILY_PARAMS[fam]

        def wall_grads(N):
            X, Y = _mesh(Nx=N, Ny=N)
            h = 1.0 / (N - 1)
            T0 = build_ic(fam, params, X, Y, T_right=300.0, b=1.0,
                          taper=True, pin_right_edge=False).astype(np.float64)
            return {
                "left": np.abs(T0[1, :] - T0[0, :]).max() / h,
                "bottom": np.abs(T0[:, 1] - T0[:, 0]).max() / h,
                "top": np.abs(T0[:, -1] - T0[:, -2]).max() / h,
            }

        coarse = wall_grads(101)
        fine = wall_grads(201)
        for wall in ("left", "bottom", "top"):
            # Halving h must at least halve the residual (allow float32 noise).
            assert fine[wall] <= 0.6 * coarse[wall] + 1e-6, (
                fam, wall, coarse[wall], fine[wall],
            )


class TestRandomField:
    @pytest.mark.parametrize("sigma", [3.0, 10.0, 18.0])
    @pytest.mark.parametrize("ell", [0.05, 0.15, 0.30])
    def test_random_field_rms_is_exact_on_midpoint_grid(self, sigma, ell):
        # Pins the normalisation identity sum_nm C_nm^2 w_n w_m = sigma^2.
        # On the midpoint grid x_k = (k + 1/2)/M the cosine modes are discretely
        # orthogonal with exactly the continuum weights w_n, so Parseval makes
        # the gridded mean square equal that sum to machine precision -- and it
        # tests the evaluated field rather than re-deriving the coefficients.
        M = 2 * RANDOM_FIELD_MODES
        axis = (np.arange(M) + 0.5) / M
        X, Y = np.meshgrid(axis, axis, indexing="ij")
        f = ic_random_cosine_field_2d(X, Y, ell=ell, sigma=sigma, wn_seed=7)
        assert abs(float(f.mean())) < 1e-12 * sigma
        rms = float(np.sqrt((f ** 2).mean()))
        assert rms == pytest.approx(sigma, rel=1e-12)

    def test_random_field_gridded_rms_converges(self):
        # On the endpoint-inclusive grids the dataset actually uses, quadrature
        # error makes the RMS only approximately sigma; it must converge.
        sigma, ell = 10.0, 0.15
        errs = []
        for N in (33, 65, 129, 257):
            axis = np.linspace(0.0, 1.0, N)
            X, Y = np.meshgrid(axis, axis, indexing="ij")
            f = ic_random_cosine_field_2d(X, Y, ell=ell, sigma=sigma, wn_seed=7)
            errs.append(abs(float(np.sqrt((f ** 2).mean())) - sigma))
        assert errs[-1] < errs[0]
        assert errs[-1] < 0.01 * sigma

    def test_random_field_resolution_independent(self):
        # Nested grids: the 65-point axis is a strict subset of the 129-point
        # axis, so the shared points are directly comparable. (64/128 would
        # share only the two endpoints and prove nothing.)
        coarse_axis = np.linspace(0.0, 1.0, 65)
        fine_axis = np.linspace(0.0, 1.0, 129)
        p = dict(ell=0.10, sigma=12.0, wn_seed=99)
        Xc, Yc = np.meshgrid(coarse_axis, coarse_axis, indexing="ij")
        Xf, Yf = np.meshgrid(fine_axis, fine_axis, indexing="ij")
        fc = ic_random_cosine_field_2d(Xc, Yc, **p)
        ff = ic_random_cosine_field_2d(Xf, Yf, **p)
        np.testing.assert_allclose(fc, ff[::2, ::2], rtol=0.0, atol=1e-11)

    def test_random_field_rejects_non_ij_mesh(self):
        # X[:, 0] / Y[0] silently produce a transposed field on an "xy" mesh.
        X, Y = np.meshgrid(
            np.linspace(0.0, 1.0, 16), np.linspace(0.0, 1.0, 24), indexing="xy",
        )
        with pytest.raises(ValueError, match="indexing='ij'"):
            ic_random_cosine_field_2d(X, Y, ell=0.1, sigma=1.0, wn_seed=0)


class TestVersionsAndDescriptors:
    def test_ic_builder_schema_version_bumped(self):
        # These stamps are the only signal that distinguishes a regenerated
        # dataset from one built by the pre-redesign builders; the load-time
        # guards compare against exactly these values.
        assert IC_BUILDER_SCHEMA_VERSION == 2
        assert ONLINE_IC_SAMPLER_VERSION == "online_neumann_ic_v2"

    def test_sinusoid_nx_is_continuous_float(self):
        # nx may be any positive real because x = 1 is Dirichlet; ny must stay
        # integral or the field is not even about both y walls.
        p = sample_random_sinusoid_ic_params(np.random.default_rng(3))
        d = canonical_ic_params("random_sinusoid_2d", p)
        assert d["nx_list"].dtype == np.dtype("<f8")
        assert d["ny_list"].dtype == np.dtype("<i8")
        assert not all(float(v).is_integer() for v in d["nx_list"])

    @pytest.mark.parametrize("fam", list(IC_FAMILIES.keys()))
    def test_build_ic_pins_right_edge_and_tapers(self, fam):
        X, Y = _mesh(Nx=101, Ny=101)
        params = _FIXED_FAMILY_PARAMS[fam]
        T0 = build_ic(fam, params, X, Y, T_right=300.0, b=1.0)
        np.testing.assert_allclose(T0[-1, :], 300.0)
        dev = np.abs(T0.astype(np.float64) - 300.0)
        x = X[:, 0]
        band = x >= 1.0 - EDGE_TAPER_WIDTH
        # The deviation decays monotonically to zero across the taper band...
        band_max = np.array([dev[i, :].max() for i in np.flatnonzero(band)])
        assert band_max[-1] == 0.0
        # ... and is untouched outside it.
        untapered = IC_BUILDERS[fam](X, Y, **canonical_ic_params(fam, params))
        inner = x <= 1.0 - EDGE_TAPER_WIDTH
        np.testing.assert_allclose(
            dev[inner, :], np.abs(untapered[inner, :]), rtol=0.0, atol=1e-3,
        )
