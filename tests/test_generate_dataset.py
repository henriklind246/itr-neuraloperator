import numpy as np

from data.generate_dataset import random_ic


class TestRandomIC:
    def test_shape(self):
        grid = np.linspace(0, 1, 101)
        rng = np.random.default_rng(0)
        result = random_ic(0.0, 1.0, grid, rng)
        assert result.shape == grid.shape

    def test_deterministic_same_seed(self):
        grid = np.linspace(0, 1, 11)
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        r1 = random_ic(0.0, 1.0, grid, rng1)
        r2 = random_ic(0.0, 1.0, grid, rng2)
        np.testing.assert_array_equal(r1, r2)

    def test_varies_with_different_seed(self):
        grid = np.linspace(0, 1, 11)
        r1 = random_ic(0.0, 1.0, grid, np.random.default_rng(0))
        r2 = random_ic(0.0, 1.0, grid, np.random.default_rng(1))
        assert not np.array_equal(r1, r2)

    def test_values_finite(self):
        grid = np.linspace(0, 1, 101)
        rng = np.random.default_rng(0)
        result = random_ic(0.0, 1.0, grid, rng)
        assert np.all(np.isfinite(result))

    def test_dtype_float(self):
        grid = np.linspace(0, 1, 11)
        rng = np.random.default_rng(0)
        result = random_ic(0.0, 1.0, grid, rng)
        assert np.issubdtype(result.dtype, np.floating)

    def test_different_domains(self):
        grid = np.linspace(0, 2, 11)
        rng1 = np.random.default_rng(0)
        r1 = random_ic(0.0, 1.0, grid, rng1)
        rng2 = np.random.default_rng(0)
        r2 = random_ic(0.0, 2.0, grid, rng2)
        # Same random coefficients but different L changes the trig arguments
        assert not np.array_equal(r1, r2)
