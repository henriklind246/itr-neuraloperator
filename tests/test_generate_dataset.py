import numpy as np

from data.generate_dataset import (
    build_sim_params,
    generate_lhs_samples,
    random_ic,
)


AMP_RANGE = (50.0, 300.0)
FREQ_RANGE = (1.0, 20.0)
RC_RANGE = (0.05, 1.0)


def _mesh(a: float, b: float, c: float, d: float, Nx: int, Ny: int):
    x = np.linspace(a, b, Nx)
    y = np.linspace(c, d, Ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    return X, Y


class TestRandomIC:
    def test_shape(self):
        X, Y = _mesh(0.0, 1.0, 0.0, 1.0, 25, 17)
        rng = np.random.default_rng(0)
        result = random_ic(0.0, 1.0, 0.0, 1.0, X, Y, rng)
        assert result.shape == (25, 17)

    def test_deterministic_same_seed(self):
        X, Y = _mesh(0.0, 1.0, 0.0, 1.0, 11, 13)
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        r1 = random_ic(0.0, 1.0, 0.0, 1.0, X, Y, rng1)
        r2 = random_ic(0.0, 1.0, 0.0, 1.0, X, Y, rng2)
        np.testing.assert_array_equal(r1, r2)

    def test_varies_with_different_seed(self):
        X, Y = _mesh(0.0, 1.0, 0.0, 1.0, 11, 13)
        r1 = random_ic(0.0, 1.0, 0.0, 1.0, X, Y, np.random.default_rng(0))
        r2 = random_ic(0.0, 1.0, 0.0, 1.0, X, Y, np.random.default_rng(1))
        assert not np.array_equal(r1, r2)

    def test_values_finite(self):
        X, Y = _mesh(0.0, 1.0, 0.0, 1.0, 25, 17)
        rng = np.random.default_rng(0)
        result = random_ic(0.0, 1.0, 0.0, 1.0, X, Y, rng)
        assert np.all(np.isfinite(result))

    def test_dtype_float(self):
        X, Y = _mesh(0.0, 1.0, 0.0, 1.0, 11, 13)
        rng = np.random.default_rng(0)
        result = random_ic(0.0, 1.0, 0.0, 1.0, X, Y, rng)
        assert np.issubdtype(result.dtype, np.floating)

    def test_different_domains(self):
        # Same grid but different declared Lx changes the trig arguments -> different output.
        # (Match the 1D-era test: grid spans [0, 2] while a=0 is fixed and b toggles between 1 and 2.)
        X, Y = _mesh(0.0, 2.0, 0.0, 2.0, 11, 13)
        r1 = random_ic(0.0, 1.0, 0.0, 1.0, X, Y, np.random.default_rng(0))
        r2 = random_ic(0.0, 2.0, 0.0, 2.0, X, Y, np.random.default_rng(0))
        assert not np.array_equal(r1, r2)

    def test_nontrivial_y_variation(self):
        """IC must actually vary along the y axis (otherwise 2D becomes 1D)."""
        X, Y = _mesh(0.0, 1.0, 0.0, 1.0, 25, 17)
        rng = np.random.default_rng(0)
        result = random_ic(0.0, 1.0, 0.0, 1.0, X, Y, rng)
        # std along y-axis (axis=1) at each x-row: at least one row must vary
        assert float(np.std(result, axis=1).max()) > 1e-6


class TestGenerateLHSSamples:
    def test_shape_and_dtype(self):
        samples = generate_lhs_samples(num_sims=64, seed=0)
        assert samples.shape == (64, 3)
        assert samples.dtype == np.float32

    def test_columns_within_bounds(self):
        samples = generate_lhs_samples(num_sims=128, seed=0)
        amps, freqs, rcs = samples[:, 0], samples[:, 1], samples[:, 2]
        assert np.all(amps >= AMP_RANGE[0]) and np.all(amps <= AMP_RANGE[1])
        assert np.all(freqs >= FREQ_RANGE[0]) and np.all(freqs <= FREQ_RANGE[1])
        assert np.all(rcs >= RC_RANGE[0]) and np.all(rcs <= RC_RANGE[1])

    def test_deterministic_same_seed(self):
        s1 = generate_lhs_samples(num_sims=32, seed=7)
        s2 = generate_lhs_samples(num_sims=32, seed=7)
        np.testing.assert_array_equal(s1, s2)

    def test_different_seed_differs(self):
        s1 = generate_lhs_samples(num_sims=32, seed=0)
        s2 = generate_lhs_samples(num_sims=32, seed=1)
        assert not np.array_equal(s1, s2)

    def test_frequency_is_log_uniform(self):
        """Frequency is sampled log-uniformly: log(f) should be approximately
        uniform on [log(lo), log(hi)], i.e. geometric mean ≈ sqrt(lo*hi)."""
        samples = generate_lhs_samples(num_sims=4000, seed=0)
        freqs = samples[:, 1]
        log_mean = np.mean(np.log(freqs))
        expected = 0.5 * (np.log(FREQ_RANGE[0]) + np.log(FREQ_RANGE[1]))
        assert abs(log_mean - expected) < 0.05

    def test_amplitude_is_uniform_not_log(self):
        """Amplitude is uniformly scaled: arithmetic mean ≈ (lo + hi) / 2."""
        samples = generate_lhs_samples(num_sims=4000, seed=0)
        amps = samples[:, 0]
        expected = 0.5 * (AMP_RANGE[0] + AMP_RANGE[1])
        assert abs(np.mean(amps) - expected) < 5.0


class TestBuildSimParams:
    def test_length_and_tuple_structure(self):
        X, Y = _mesh(0.0, 1.0, 0.0, 1.0, 9, 7)
        rng = np.random.default_rng(0)
        params = build_sim_params(0.0, 1.0, 0.0, 1.0, X, Y, num_sims=8, rng=rng, lhs_seed=0)
        assert len(params) == 8
        for entry in params:
            assert len(entry) == 4
            amp, freq, T0, R_c = entry
            assert AMP_RANGE[0] <= float(amp) <= AMP_RANGE[1]
            assert FREQ_RANGE[0] <= float(freq) <= FREQ_RANGE[1]
            assert RC_RANGE[0] <= float(R_c) <= RC_RANGE[1]
            assert isinstance(T0, np.ndarray)

    def test_T0_is_2d_with_grid_shape(self):
        """Regression guard: catches any silent revert to a 1D IC."""
        Nx, Ny = 13, 9
        X, Y = _mesh(0.0, 1.0, 0.0, 1.0, Nx, Ny)
        rng = np.random.default_rng(0)
        params = build_sim_params(0.0, 1.0, 0.0, 1.0, X, Y, num_sims=4, rng=rng, lhs_seed=0)
        for _, _, T0, _ in params:
            assert T0.shape == (Nx, Ny)
            assert np.issubdtype(T0.dtype, np.floating)

    def test_deterministic_same_rng_and_lhs_seed(self):
        X, Y = _mesh(0.0, 1.0, 0.0, 1.0, 11, 11)
        p1 = build_sim_params(0.0, 1.0, 0.0, 1.0, X, Y, num_sims=5,
                              rng=np.random.default_rng(123), lhs_seed=0)
        p2 = build_sim_params(0.0, 1.0, 0.0, 1.0, X, Y, num_sims=5,
                              rng=np.random.default_rng(123), lhs_seed=0)
        for (a1, f1, T1, r1), (a2, f2, T2, r2) in zip(p1, p2):
            assert float(a1) == float(a2)
            assert float(f1) == float(f2)
            assert float(r1) == float(r2)
            np.testing.assert_array_equal(T1, T2)

    def test_T0_varies_across_sims(self):
        """Each sim should get its own random IC."""
        X, Y = _mesh(0.0, 1.0, 0.0, 1.0, 15, 15)
        rng = np.random.default_rng(0)
        params = build_sim_params(0.0, 1.0, 0.0, 1.0, X, Y, num_sims=4, rng=rng, lhs_seed=0)
        T0_stack = np.stack([p[2] for p in params])
        # No two ICs should be identical
        for i in range(len(params)):
            for j in range(i + 1, len(params)):
                assert not np.array_equal(T0_stack[i], T0_stack[j])
