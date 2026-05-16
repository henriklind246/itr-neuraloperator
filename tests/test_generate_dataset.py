import numpy as np

from data.generate_dataset import (
    build_sim_params,
    generate_sim_data,
    generate_lhs_samples,
    main as generate_main,
)
from src.physics.boundary_forcing import TEMPORAL_FAMILIES
from src.physics.init_conditions import IC_FAMILIES


RC_RANGE = (0.05, 1.0)
DT = 0.005
T_FINAL = 0.3


def _mesh(a: float, b: float, c: float, d: float, Nx: int, Ny: int):
    x = np.linspace(a, b, Nx)
    y = np.linspace(c, d, Ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    return X, Y


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
                         "temporal_params", "spatial_family", "spatial_params",
                         "ic_family", "ic_params"}
        for entry in params:
            assert isinstance(entry, dict)
            assert required_keys.issubset(entry.keys())
            assert RC_RANGE[0] <= float(entry["R_c"]) <= RC_RANGE[1]
            assert isinstance(entry["T0"], np.ndarray)
            assert entry["temporal_family"] in TEMPORAL_FAMILIES
            assert entry["spatial_family"] in {"uniform", "patch", "gaussian", "triangle"}
            assert entry["ic_family"] in IC_FAMILIES
            assert isinstance(entry["ic_params"], dict)

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
            assert T0.dtype == np.float32
            assert np.isfinite(T0).all()

    def test_T0_right_boundary_matches_dirichlet(self):
        Nx, Ny = 13, 9
        T_right = 312.5
        X, Y = _mesh(0.0, 1.0, 0.0, 1.0, Nx, Ny)
        rng = np.random.default_rng(0)
        rng_profile = np.random.default_rng(1)
        params = build_sim_params(0.0, 1.0, 0.0, 1.0, X, Y, num_sims=4,
                                  rng=rng, rng_profile=rng_profile,
                                  dt=DT, t_final=T_FINAL, lhs_seed=0,
                                  T_right=T_right)
        for entry in params:
            np.testing.assert_allclose(entry["T0"][-1, :], T_right)

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
            assert e1["ic_family"] == e2["ic_family"]
            assert e1["ic_params"] == e2["ic_params"]

    def test_T0_varies_across_sims(self):
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

    def test_ic_family_distribution_uses_rng_not_rng_profile(self):
        """Changing rng_profile alone must not change IC family/params draws."""
        X, Y = _mesh(0.0, 1.0, 0.0, 1.0, 9, 7)
        # same rng (data RNG), different rng_profile
        p1 = build_sim_params(0.0, 1.0, 0.0, 1.0, X, Y, num_sims=6,
                              rng=np.random.default_rng(42),
                              rng_profile=np.random.default_rng(1),
                              dt=DT, t_final=T_FINAL, lhs_seed=0)
        p2 = build_sim_params(0.0, 1.0, 0.0, 1.0, X, Y, num_sims=6,
                              rng=np.random.default_rng(42),
                              rng_profile=np.random.default_rng(2),
                              dt=DT, t_final=T_FINAL, lhs_seed=0)
        for e1, e2 in zip(p1, p2):
            assert e1["ic_family"] == e2["ic_family"]
            assert e1["ic_params"] == e2["ic_params"]
            np.testing.assert_array_equal(e1["T0"], e2["T0"])

    def test_every_ic_family_is_accepted_by_build_sim_params(self, monkeypatch):
        X, Y = _mesh(0.0, 1.0, 0.0, 1.0, 17, 13)
        families = list(IC_FAMILIES.keys())
        family_iter = iter(families)
        monkeypatch.setattr(
            "data.generate_dataset.sample_ic_family",
            lambda rng: next(family_iter),
        )

        params = build_sim_params(0.0, 1.0, 0.0, 1.0, X, Y,
                                  num_sims=len(families),
                                  rng=np.random.default_rng(11),
                                  rng_profile=np.random.default_rng(12),
                                  dt=DT, t_final=T_FINAL, lhs_seed=0)

        assert [p["ic_family"] for p in params] == families
        for entry in params:
            assert entry["T0"].shape == (17, 13)
            assert entry["T0"].dtype == np.float32
            assert np.isfinite(entry["T0"]).all()
            np.testing.assert_allclose(entry["T0"][-1, :], 300.0)


class TestGenerateSimData:
    def test_main_passes_cli_overrides_to_generator(self, tmp_path):
        calls = {}

        def fake_generate(**kwargs):
            calls.update(kwargs)

        save_dir = tmp_path / "custom_data"
        exit_code = generate_main(
            ["--num-sims", "17", "--save-dir", str(save_dir)],
            generate_fn=fake_generate,
        )

        assert exit_code == 0
        assert calls["num_sims"] == 17
        assert calls["save_dir"] == save_dir

    def test_generate_sim_data_smoke_writes_outputs_to_data_dir_with_ic_metadata(
        self, tmp_path, monkeypatch, capsys,
    ):
        cwd = tmp_path / "cwd"
        data_dir = tmp_path / "data"
        cwd.mkdir()
        monkeypatch.chdir(cwd)
        monkeypatch.setattr("data.generate_dataset.DATA_DIR", data_dir)

        generate_sim_data(num_sims=2, save_stride=2)

        out = capsys.readouterr().out
        assert "Building simulation parameters." in out
        assert "Finished simulation 0" in out
        assert "Finished simulation 1" in out
        assert "Saved to:" in out

        expected_files = [
            "x_grid.npy",
            "y_grid.npy",
            "t_grid.npy",
            "dt.npy",
            "trajectories.npy",
            "sim_params.npy",
        ]
        for name in expected_files:
            assert (data_dir / name).exists()
            assert not (cwd / name).exists()

        trajectories = np.load(data_dir / "trajectories.npy")
        x_grid = np.load(data_dir / "x_grid.npy")
        y_grid = np.load(data_dir / "y_grid.npy")
        t_grid = np.load(data_dir / "t_grid.npy")
        dt = np.load(data_dir / "dt.npy")
        sim_params = np.load(data_dir / "sim_params.npy", allow_pickle=True)

        assert trajectories.shape == (2, 31, 100, 100)
        assert trajectories.dtype == np.float32
        assert x_grid.shape == (100,)
        assert y_grid.shape == (100,)
        assert t_grid.shape == (31,)
        assert float(dt) == DT
        assert sim_params.shape == (2,)
        for entry in sim_params:
            assert entry["ic_family"] in IC_FAMILIES
            assert isinstance(entry["ic_params"], dict)
            assert entry["T0"].shape == (100, 100)
            np.testing.assert_allclose(entry["T0"][-1, :], 300.0)
