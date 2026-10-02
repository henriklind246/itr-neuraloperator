import numpy as np

from data.generate_dataset import (
    IC_ASSIGN_SEED,
    build_sim_params,
    generate_sim_data,
    generate_lhs_samples,
    main as generate_main,
    resolve_seed_streams,
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
            assert entry["ic_family"] == "uniform_2d"
            assert entry["ic_params"] == {"T0_offset": 0.0}

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
            np.testing.assert_allclose(T0, 300.0)

    def test_T0_is_fixed_300_even_with_T_right_override(self):
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
            np.testing.assert_allclose(entry["T0"], 300.0)

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

    def test_T0_is_uniform_300_across_sims(self):
        X, Y = _mesh(0.0, 1.0, 0.0, 1.0, 15, 15)
        rng = np.random.default_rng(0)
        rng_profile = np.random.default_rng(1)
        params = build_sim_params(0.0, 1.0, 0.0, 1.0, X, Y, num_sims=4,
                                  rng=rng, rng_profile=rng_profile,
                                  dt=DT, t_final=T_FINAL, lhs_seed=0)
        T0_stack = np.stack([p["T0"] for p in params])
        np.testing.assert_allclose(T0_stack, 300.0)
        assert {p["ic_family"] for p in params} == {"uniform_2d"}
        assert all(p["ic_params"] == {"T0_offset": 0.0} for p in params)

    def test_ic_metadata_does_not_depend_on_rng_streams(self):
        X, Y = _mesh(0.0, 1.0, 0.0, 1.0, 9, 7)
        p1 = build_sim_params(0.0, 1.0, 0.0, 1.0, X, Y, num_sims=6,
                              rng=np.random.default_rng(42),
                              rng_profile=np.random.default_rng(1),
                              dt=DT, t_final=T_FINAL, lhs_seed=0)
        p2 = build_sim_params(0.0, 1.0, 0.0, 1.0, X, Y, num_sims=6,
                              rng=np.random.default_rng(99),
                              rng_profile=np.random.default_rng(2),
                              dt=DT, t_final=T_FINAL, lhs_seed=0)
        for e1, e2 in zip(p1, p2):
            assert e1["ic_family"] == e2["ic_family"] == "uniform_2d"
            assert e1["ic_params"] == e2["ic_params"] == {"T0_offset": 0.0}
            np.testing.assert_allclose(e1["T0"], 300.0)
            np.testing.assert_array_equal(e1["T0"], e2["T0"])

    def test_build_sim_params_uses_only_uniform_300_ic(self):
        X, Y = _mesh(0.0, 1.0, 0.0, 1.0, 17, 13)

        params = build_sim_params(0.0, 1.0, 0.0, 1.0, X, Y,
                                  num_sims=4,
                                  rng=np.random.default_rng(11),
                                  rng_profile=np.random.default_rng(12),
                                  dt=DT, t_final=T_FINAL, lhs_seed=0)

        for entry in params:
            assert entry["ic_family"] == "uniform_2d"
            assert entry["ic_params"] == {"T0_offset": 0.0}
            assert entry["T0"].shape == (17, 13)
            assert entry["T0"].dtype == np.float32
            assert np.isfinite(entry["T0"]).all()
            np.testing.assert_allclose(entry["T0"], 300.0)


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
            "ramp_seconds.npy",
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
        ramp_seconds = np.load(data_dir / "ramp_seconds.npy")
        sim_params = np.load(data_dir / "sim_params.npy", allow_pickle=True)

        assert trajectories.shape == (2, 31, 100, 100)
        assert trajectories.dtype == np.float32
        assert x_grid.shape == (100,)
        assert y_grid.shape == (100,)
        assert t_grid.shape == (31,)
        assert float(dt) == DT
        assert float(ramp_seconds) == 2.0 * DT
        assert sim_params.shape == (2,)
        for entry in sim_params:
            assert entry["ic_family"] == "uniform_2d"
            assert entry["ic_params"] == {"T0_offset": 0.0}
            assert entry["T0"].shape == (100, 100)
            np.testing.assert_allclose(entry["T0"], 300.0)


def _forcing_latents(sim_params) -> list[tuple]:
    """Forcing/IC identity of each sim, ignoring the LHS-drawn R_c."""
    return [
        (
            entry["temporal_family"],
            tuple(sorted((k, float(np.asarray(v).ravel()[0]))
                         for k, v in entry["temporal_params"].items()
                         if np.isscalar(v) or np.asarray(v).size)),
            entry["spatial_family"],
        )
        for entry in sim_params
    ]


def _generate(tmp_path, name, **kwargs):
    out = tmp_path / name
    generate_sim_data(num_sims=kwargs.pop("num_sims", 3), save_stride=2,
                      nx=12, ny=12, save_dir=out, **kwargs)
    return np.load(out / "sim_params.npy", allow_pickle=True), out


class TestResolveSeedStreams:
    def test_family_zero_reproduces_the_historical_seeds(self):
        assert resolve_seed_streams(0) == {
            "ic_params": 0, "forcing_profile": 1, "lhs": 0,
            "ic_assign": IC_ASSIGN_SEED,
        }

    def test_distinct_families_share_no_seed(self):
        a = set(resolve_seed_streams(0).values())
        b = set(resolve_seed_streams(7).values())
        assert not (a & b)

    def test_main_forwards_the_flag(self, tmp_path):
        calls = {}
        generate_main(["--rng-seed", "7", "--save-dir", str(tmp_path)],
                      generate_fn=lambda **kw: calls.update(kw))
        assert calls["rng_seed"] == 7

    def test_flag_defaults_to_the_historical_family(self, tmp_path):
        calls = {}
        generate_main(["--save-dir", str(tmp_path)],
                      generate_fn=lambda **kw: calls.update(kw))
        assert calls["rng_seed"] == 0


class TestDrawFamilyDisjointness:
    def test_default_family_repeats_forcing_draws_across_set_sizes(self, tmp_path):
        """Why --rng-seed exists.

        The per-sim forcing/IC streams are consumed sequentially, so sim i of a
        small default-family set is the same forcing draw as sim i of the large
        default-family training set. Regenerating locally at a different
        num_sims therefore does NOT produce held-out simulations.
        """
        small, _ = _generate(tmp_path, "small", num_sims=3)
        large, _ = _generate(tmp_path, "large", num_sims=6)
        assert _forcing_latents(small) == _forcing_latents(large)[:3]

    def test_non_default_family_shares_no_forcing_draw(self, tmp_path):
        base, _ = _generate(tmp_path, "base", num_sims=4, rng_seed=0)
        held, _ = _generate(tmp_path, "held", num_sims=4, rng_seed=7)
        overlap = set(_forcing_latents(base)) & set(_forcing_latents(held))
        assert not overlap

    def test_non_default_family_shares_no_R_c(self, tmp_path):
        base, _ = _generate(tmp_path, "base", num_sims=4, rng_seed=0)
        held, _ = _generate(tmp_path, "held", num_sims=4, rng_seed=7)
        assert not (
            {round(float(e["R_c"]), 9) for e in base}
            & {round(float(e["R_c"]), 9) for e in held}
        )

    def test_a_family_is_reproducible(self, tmp_path):
        a, _ = _generate(tmp_path, "a", num_sims=3, rng_seed=7)
        b, _ = _generate(tmp_path, "b", num_sims=3, rng_seed=7)
        assert _forcing_latents(a) == _forcing_latents(b)
        np.testing.assert_allclose([e["R_c"] for e in a], [e["R_c"] for e in b])

    def test_R_c_stays_in_range_off_the_default_family(self, tmp_path):
        held, _ = _generate(tmp_path, "held", num_sims=8, rng_seed=7)
        values = np.array([float(e["R_c"]) for e in held])
        assert values.min() >= RC_RANGE[0]
        assert values.max() <= RC_RANGE[1]


class TestSeedProvenance:
    def test_non_default_family_records_its_seed(self, tmp_path):
        _, out = _generate(tmp_path, "held", num_sims=2, rng_seed=7)
        meta = np.load(out / "meta.npy", allow_pickle=True).item()
        assert meta["rng_seed"] == 7
        assert meta["seed_streams"] == resolve_seed_streams(7)

    def test_default_family_writes_no_meta_for_an_unversioned_spec(self, tmp_path):
        _, out = _generate(tmp_path, "base", num_sims=2, rng_seed=0)
        assert not (out / "meta.npy").exists()


def test_homogeneous_generation_preserves_varying_ic_versions_and_balancing(tmp_path):
    from src.physics.init_conditions import IC_BUILDER_SCHEMA_VERSION, ONLINE_IC_SAMPLER_VERSION
    generate_sim_data(num_sims=8,save_stride=2,save_dir=tmp_path,
                      benchmark='diffusion_forcing_single',nx=4,ny=4)
    meta=np.load(tmp_path/'meta.npy',allow_pickle=True).item()
    assert meta['problem_version'] == 'forcing_single_varying_ic_v2'
    assert meta['ic_mode'] == 'varying'
    assert meta['online_sampler_version'] == ONLINE_IC_SAMPLER_VERSION
    assert meta['ic_builder_version'] == IC_BUILDER_SCHEMA_VERSION
    params=np.load(tmp_path/'sim_params.npy',allow_pickle=True)
    families,counts=np.unique([p['ic_family'] for p in params],return_counts=True)
    assert len(families) == 4
    assert np.all(counts == 2)
    assert all('R_c' not in p and 'interface_x' not in p for p in params)
    assert all(p['temporal_family'] == 'sin' and p['spatial_family'] == 'uniform' for p in params)
    assert np.load(tmp_path/'trajectories.npy').shape == (8,31,4,4)
