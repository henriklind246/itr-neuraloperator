import random

import numpy as np
import pytest
import torch

from src.physics.init_conditions import (
    IC_FAMILIES,
    IC_SAMPLERS,
    IC_BUILDERS,
    HOT_MARGIN_FACTOR,
    GRF_ELL_RANGE,
    GRF_SIGMA_RANGE,
    HOT_AMP_RANGE,
    HOT_N_CHOICES,
    HOT_SIGMA_RANGE,
    SINU_AMP_RANGE,
    SINU_KMAX,
    SINU_N_CHOICES,
    UNIFORM_OFFSET_RANGE,
    EDGE_TAPER_WIDTH,
    boundary_taper,
    balanced_ic_family_assignments,
    build_ic,
    canonical_ic_params,
    right_edge_taper,
    sample_ic_family,
    sample_uniform_ic_params,
    sample_random_sinusoid_ic_params,
    sample_grf_ic_params,
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
        assert set(p.keys()) == {"A_list", "nx_list", "ny_list", "phi_list"}
        N = len(p["A_list"])
        assert N in (3, 4, 5)
        for k in ("nx_list", "ny_list", "phi_list"):
            assert len(p[k]) == N

    def test_grf_keys(self):
        rng = np.random.default_rng(0)
        p = sample_grf_ic_params(rng, Nx=64, Ny=64)
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
            p1 = IC_SAMPLERS[fam](r1, Nx=64, Ny=64)
            p2 = IC_SAMPLERS[fam](r2, Nx=64, Ny=64)
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
            for A, nx, ny, phi in zip(p["A_list"], p["nx_list"], p["ny_list"], p["phi_list"]):
                assert SINU_AMP_RANGE[0] <= A <= SINU_AMP_RANGE[1]
                assert 1 <= nx <= SINU_KMAX
                assert 1 <= ny <= SINU_KMAX
                assert 0.0 <= phi <= 2.0 * np.pi

    def test_grf_params_within_documented_ranges(self):
        rng = np.random.default_rng(0)
        for _ in range(100):
            p = sample_grf_ic_params(rng, Nx=64, Ny=64)
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
            p = IC_SAMPLERS[fam](rng, Nx=100, Ny=100)
            dev = IC_BUILDERS[fam](X, Y, **p)
            assert dev.shape == (100, 100), f"{fam} shape"
            assert np.isfinite(dev).all(), f"{fam} non-finite"

    def test_uniform_is_constant_before_taper(self):
        X, Y = _mesh()
        dev = IC_BUILDERS["uniform_2d"](X, Y, T0_offset=7.5)
        assert np.allclose(dev, 7.5)

    def test_grf_empirical_std_matches_sigma(self):
        X, Y = _mesh(Nx=100, Ny=100)
        for sigma in (3.0, 12.0, 18.0):
            for ell in (0.05, 0.15, 0.30):
                p = {"ell": ell, "sigma": sigma, "wn_seed": 42}
                dev = IC_BUILDERS["grf_2d"](X, Y, **p)
                assert abs(dev.std() - sigma) / sigma < 0.05
                assert abs(dev.mean()) < 1e-6

    @pytest.mark.parametrize("fam", list(IC_FAMILIES.keys()))
    def test_builders_are_deterministic_for_fixed_params(self, fam):
        X, Y = _mesh(Nx=41, Ny=37)
        rng = np.random.default_rng(123)
        p = IC_SAMPLERS[fam](rng, Nx=41, Ny=37)
        d1 = IC_BUILDERS[fam](X, Y, **p)
        d2 = IC_BUILDERS[fam](X, Y, **p)
        np.testing.assert_array_equal(d1, d2)


class TestEdgeTaper:
    def test_taper_endpoints(self):
        X, Y = _mesh()
        t = right_edge_taper(X, b=1.0)
        assert np.allclose(t[0, :], 1.0)
        assert np.allclose(t[-1, :], 0.0)

    def test_taper_monotone(self):
        x = np.linspace(0.0, 1.0, 200)
        X, _ = np.meshgrid(x, x, indexing="ij")
        t = right_edge_taper(X, b=1.0)
        # along x, taper should be non-increasing
        col = t[:, 0]
        assert np.all(np.diff(col) <= 1e-12)

    def test_taper_is_zero_only_inside_right_edge_band(self):
        x = np.linspace(0.0, 2.0, 201)
        y = np.linspace(0.0, 1.0, 51)
        X, _ = np.meshgrid(x, y, indexing="ij")
        t = right_edge_taper(X, b=2.0, edge_width=0.25)
        assert np.allclose(t[x <= 1.75, :], 1.0)
        assert np.allclose(t[-1, :], 0.0)
        assert np.all((0.0 <= t) & (t <= 1.0))


class TestBuildIC:
    @pytest.mark.parametrize("fam", list(IC_FAMILIES.keys()))
    def test_build_ic_shape_dtype_pinned(self, fam):
        X, Y = _mesh()
        rng = np.random.default_rng(0)
        p = IC_SAMPLERS[fam](rng, Nx=100, Ny=100)
        T0 = build_ic(fam, p, X, Y, T_right=300.0, b=1.0)
        assert T0.shape == (100, 100)
        assert T0.dtype == np.float32
        assert np.isfinite(T0).all()
        np.testing.assert_allclose(T0[-1, :], 300.0)

    def test_build_ic_no_pin_no_taper(self):
        X, Y = _mesh()
        rng = np.random.default_rng(0)
        p = IC_SAMPLERS["uniform_2d"](rng, Nx=100, Ny=100)
        T0 = build_ic("uniform_2d", p, X, Y, T_right=300.0,
                      taper=False, pin_right_edge=False)
        # pure constant + T_right
        assert np.allclose(T0, 300.0 + p["T0_offset"])

    @pytest.mark.parametrize("fam", list(IC_FAMILIES.keys()))
    def test_build_ic_taper_reaches_right_dirichlet_without_pin(self, fam):
        X, Y = _mesh(Nx=51, Ny=49)
        rng = np.random.default_rng(5)
        p = IC_SAMPLERS[fam](rng, Nx=51, Ny=49)
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
        params = IC_SAMPLERS[fam](np.random.default_rng(7), Nx=31, Ny=29)
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


# Non-uniform IC families whose deviation is a deterministic continuous function
# of (x, y) and therefore comparable across grid resolutions. grf_2d is excluded:
# its FFT realization depends on Nx*Ny rng draws so it cannot be reproduced under
# refinement (see plan / generate_dataset --exclude-ic).
_GRID_COMPARABLE_FAMILIES = {
    "random_sinusoid_2d": dict(
        A_list=[5.0, 3.0], nx_list=[2, 3], ny_list=[1, 2], phi_list=[0.3, 1.1],
    ),
    "hot_spot_2d": dict(
        A_list=[20.0, -15.0], mu_x_list=[0.4, 0.6], mu_y_list=[0.5, 0.3],
        sigma_list=[0.15, 0.12],
    ),
}


class TestBoundaryTaper:
    def test_zero_on_all_outer_node_lines(self):
        X, Y = _mesh(Nx=100, Ny=100)
        w = boundary_taper(X, Y)
        assert np.allclose(w[0, :], 0.0)
        assert np.allclose(w[-1, :], 0.0)
        assert np.allclose(w[:, 0], 0.0)
        assert np.allclose(w[:, -1], 0.0)

    def test_window_in_unit_interval(self):
        X, Y = _mesh(Nx=64, Ny=80)
        w = boundary_taper(X, Y)
        assert np.all((0.0 <= w) & (w <= 1.0 + 1e-12))

    def test_interior_saturates_to_one(self):
        # Points more than edge_width from every boundary have all four smoothstep
        # factors equal to 1, so the window is exactly 1 there.
        X, Y = _mesh(Nx=100, Ny=100)
        w = boundary_taper(X, Y)
        interior = (
            (X > X.min() + EDGE_TAPER_WIDTH) & (X < X.max() - EDGE_TAPER_WIDTH)
            & (Y > Y.min() + EDGE_TAPER_WIDTH) & (Y < Y.max() - EDGE_TAPER_WIDTH)
        )
        assert np.allclose(w[interior], 1.0)
        assert w.max() == pytest.approx(1.0)

    def test_separable_product_structure(self):
        # boundary_taper factorizes as wx(X) * wy(Y).
        X, Y = _mesh(Nx=41, Ny=37)
        w = boundary_taper(X, Y)
        from src.physics.init_conditions import _smoothstep_window
        x_lo, x_hi = X.min(), X.max()
        y_lo, y_hi = Y.min(), Y.max()
        wx = _smoothstep_window(X - x_lo, EDGE_TAPER_WIDTH) * _smoothstep_window(x_hi - X, EDGE_TAPER_WIDTH)
        wy = _smoothstep_window(Y - y_lo, EDGE_TAPER_WIDTH) * _smoothstep_window(y_hi - Y, EDGE_TAPER_WIDTH)
        assert np.allclose(w, wx * wy)


class TestBuildICBoundaryCompatibility:
    @pytest.mark.parametrize("fam", list(_GRID_COMPARABLE_FAMILIES.keys()) + ["grf_2d"])
    def test_all_four_outer_node_lines_equal_t_right_no_pin(self, fam):
        # With the 4-sided taper every outer node line collapses to T_right even
        # without the right-edge pin (W=0 on all outer node lines).
        X, Y = _mesh(Nx=51, Ny=49)
        rng = np.random.default_rng(5)
        p = IC_SAMPLERS[fam](rng, Nx=51, Ny=49)
        T0 = build_ic(fam, p, X, Y, T_right=305.0, b=1.0,
                      taper=True, pin_right_edge=False)
        np.testing.assert_allclose(T0[0, :], 305.0, atol=1e-4)
        np.testing.assert_allclose(T0[-1, :], 305.0, atol=1e-4)
        np.testing.assert_allclose(T0[:, 0], 305.0, atol=1e-4)
        np.testing.assert_allclose(T0[:, -1], 305.0, atol=1e-4)

    @pytest.mark.parametrize("fam", list(_GRID_COMPARABLE_FAMILIES.keys()))
    def test_left_top_bottom_normal_gradient_shrinks_under_refinement(self, fam):
        # Smoothstep has phi(0)=phi'(0)=0, so near each outer node line the
        # tapered deviation is O(dist^2) and the one-sided discrete normal
        # gradient is O(h): it must shrink under grid refinement and stay far
        # below the interior gradient magnitude. No fixed absolute threshold.
        params = _GRID_COMPARABLE_FAMILIES[fam]

        def boundary_and_interior_grads(N):
            X, Y = _mesh(Nx=N, Ny=N)
            h = 1.0 / (N - 1)
            T0 = build_ic(fam, params, X, Y, T_right=300.0, b=1.0,
                          taper=True, pin_right_edge=False)
            left = np.abs(T0[1, :] - T0[0, :]).max() / h
            bottom = np.abs(T0[:, 1] - T0[:, 0]).max() / h
            top = np.abs(T0[:, -1] - T0[:, -2]).max() / h
            interior = np.abs(np.diff(T0, axis=0)).max() / h
            return dict(left=left, bottom=bottom, top=top, interior=interior)

        coarse = boundary_and_interior_grads(51)
        fine = boundary_and_interior_grads(101)
        for edge in ("left", "bottom", "top"):
            # Boundary normal gradient decreases under refinement (O(h)).
            assert fine[edge] < coarse[edge], (fam, edge, fine[edge], coarse[edge])
            # ... and is much smaller than a representative interior gradient.
            assert fine[edge] < 0.25 * fine["interior"], (fam, edge)
