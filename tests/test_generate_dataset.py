import numpy as np

from data.generate_dataset import (
    build_sim_params,
    generate_lhs_samples,
    random_ic,
)
from src.physics.boundary_forcing import TEMPORAL_FAMILIES


RC_RANGE = (0.05, 1.0)
DT = 0.005
T_FINAL = 0.3


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
        X, Y = _mesh(0.0, 2.0, 0.0, 2.0, 11, 13)
        r1 = random_ic(0.0, 1.0, 0.0, 1.0, X, Y, np.random.default_rng(0))
        r2 = random_ic(0.0, 2.0, 0.0, 2.0, X, Y, np.random.default_rng(0))
        assert not np.array_equal(r1, r2)

    def test_nontrivial_y_variation(self):
        """IC must actually vary along the y axis (otherwise 2D becomes 1D)."""
        X, Y = _mesh(0.0, 1.0, 0.0, 1.0, 25, 17)
        rng = np.random.default_rng(0)
        result = random_ic(0.0, 1.0, 0.0, 1.0, X, Y, rng)
        assert float(np.std(result, axis=1).max()) > 1e-6


class TestGenerateLHSSamples:
    def test_shape_and_dtype(self):
        samples = generate_lhs_samples(num_sims=64, seed=0)
        assert samples.shape == (64, 1)
        assert samples.dtype == np.float32

    def test_columns_within_bounds(self):
        samples = generate_lhs_samples(num_sims=128, seed=0)
        rcs = samples[:, 0]
        assert np.all(rcs >= RC_RANGE[0]) and np.all(rcs <= RC_RANGE[1])

    def test_deterministic_same_seed(self):
        s1 = generate_lhs_samples(num_sims=32, seed=7)
        s2 = generate_lhs_samples(num_sims=32, seed=7)
        np.testing.assert_array_equal(s1, s2)

    def test_different_seed_differs(self):
        s1 = generate_lhs_samples(num_sims=32, seed=0)
        s2 = generate_lhs_samples(num_sims=32, seed=1)
        assert not np.array_equal(s1, s2)


class TestBuildSimParams:
    def test_length_and_dict_structure(self):
        X, Y = _mesh(0.0, 1.0, 0.0, 1.0, 9, 7)
        rng = np.random.default_rng(0)
        rng_profile = np.random.default_rng(1)
        params = build_sim_params(0.0, 1.0, 0.0, 1.0, X, Y, num_sims=8,
                                  rng=rng, rng_profile=rng_profile,
                                  dt=DT, t_final=T_FINAL, lhs_seed=0)
        assert len(params) == 8
        required_keys = {"R_c", "T0", "temporal_family",
                         "temporal_params", "spatial_family", "spatial_params"}
        for entry in params:
            assert isinstance(entry, dict)
            assert required_keys.issubset(entry.keys())
            assert RC_RANGE[0] <= float(entry["R_c"]) <= RC_RANGE[1]
            assert isinstance(entry["T0"], np.ndarray)
            assert entry["temporal_family"] in TEMPORAL_FAMILIES
            assert entry["spatial_family"] in {"uniform", "patch", "gaussian", "triangle"}

    def test_temporal_params_match_family_schema(self):
        X, Y = _mesh(0.0, 1.0, 0.0, 1.0, 9, 7)
        rng = np.random.default_rng(0)
        rng_profile = np.random.default_rng(1)
        params = build_sim_params(0.0, 1.0, 0.0, 1.0, X, Y, num_sims=64,
                                  rng=rng, rng_profile=rng_profile,
                                  dt=DT, t_final=T_FINAL, lhs_seed=0)
        for entry in params:
            tp = entry["temporal_params"]
            fam = entry["temporal_family"]
            if fam == "sin":
                assert {"A", "f", "t_on", "t_off", "phase", "tukey_alpha"} <= tp.keys()
            elif fam == "exp":
                assert {"A", "t0", "tau"} <= tp.keys()
            elif fam == "pulse_train":
                assert {"Np", "A_list", "t_list", "dt_list"} <= tp.keys()
                assert len(tp["A_list"]) == tp["Np"]
            elif fam == "exp_train":
                assert {"Np", "A_list", "t_list", "tau_list"} <= tp.keys()
                assert len(tp["A_list"]) == tp["Np"]

    def test_T0_is_2d_with_grid_shape(self):
        Nx, Ny = 13, 9
        X, Y = _mesh(0.0, 1.0, 0.0, 1.0, Nx, Ny)
        rng = np.random.default_rng(0)
        rng_profile = np.random.default_rng(1)
        params = build_sim_params(0.0, 1.0, 0.0, 1.0, X, Y, num_sims=4,
                                  rng=rng, rng_profile=rng_profile,
                                  dt=DT, t_final=T_FINAL, lhs_seed=0)
        for entry in params:
            T0 = entry["T0"]
            assert T0.shape == (Nx, Ny)
            assert np.issubdtype(T0.dtype, np.floating)

    def test_deterministic_same_rng_and_lhs_seed(self):
        X, Y = _mesh(0.0, 1.0, 0.0, 1.0, 11, 11)
        p1 = build_sim_params(0.0, 1.0, 0.0, 1.0, X, Y, num_sims=5,
                              rng=np.random.default_rng(123),
                              rng_profile=np.random.default_rng(7),
                              dt=DT, t_final=T_FINAL, lhs_seed=0)
        p2 = build_sim_params(0.0, 1.0, 0.0, 1.0, X, Y, num_sims=5,
                              rng=np.random.default_rng(123),
                              rng_profile=np.random.default_rng(7),
                              dt=DT, t_final=T_FINAL, lhs_seed=0)
        for e1, e2 in zip(p1, p2):
            assert float(e1["R_c"]) == float(e2["R_c"])
            np.testing.assert_array_equal(e1["T0"], e2["T0"])
            assert e1["spatial_family"] == e2["spatial_family"]
            assert e1["spatial_params"] == e2["spatial_params"]
            assert e1["temporal_family"] == e2["temporal_family"]
            assert e1["temporal_params"] == e2["temporal_params"]

    def test_T0_varies_across_sims(self):
        """Each sim should get its own random IC."""
        X, Y = _mesh(0.0, 1.0, 0.0, 1.0, 15, 15)
        rng = np.random.default_rng(0)
        rng_profile = np.random.default_rng(1)
        params = build_sim_params(0.0, 1.0, 0.0, 1.0, X, Y, num_sims=4,
                                  rng=rng, rng_profile=rng_profile,
                                  dt=DT, t_final=T_FINAL, lhs_seed=0)
        T0_stack = np.stack([p["T0"] for p in params])
        for i in range(len(params)):
            for j in range(i + 1, len(params)):
                assert not np.array_equal(T0_stack[i], T0_stack[j])
