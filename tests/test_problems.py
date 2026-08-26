import numpy as np
import pytest

from data.dataset import SnapshotPairDataset, problem_from_config
from problems.registry import get_problem
from problems.forcing import (
    COND_STATIC_DIM as FORCING_COND_STATIC_DIM,
    FORCING_SPATIAL_DESCRIPTOR_SLICE,
    FORCING_TEMPORAL_SAMPLES,
    FORCING_TEMPORAL_TOKEN_DIM,
    RC_RANGE,
    SPATIAL_PROFILE_BINS,
    ForcingProblem,
    build_cond_vector,
    build_spatial_profile_bin_averages,
)
from problems.forcing_itr import ForcingItrProblem
from problems.interfaces import InterfacesProblem
from problems.source import SourceProblem
from problems.source_itr import SourceItrProblem
from src.physics.fv_solver_2d import FVSolver2D, Layer2D

_SYNTH_MU = 0.0
_SYNTH_SIGMA = 1.0

# Target (benchmark x representation) tensor contract. Single source of truth for
# the dims/shape tests below. `temporal_encoder` is the only representation.
CONTRACTS = {
    ("forcing", "temporal_encoder"): dict(
        in_ch=4, cond=10, token=3, t_stats=2, s_y=3, aug=True,
    ),
    ("forcing_itr", "temporal_encoder"): dict(
        in_ch=5, cond=13, token=3, t_stats=2, s_y=3, aug=True,
    ),
    ("forcing_itr_sin", "temporal_encoder"): dict(
        in_ch=5, cond=11, token=3, t_stats=2, s_y=3, aug=True,
    ),
    ("source", "temporal_encoder"): dict(
        in_ch=4, cond=6, token=3, t_stats=3, s_y=3, aug=False,
    ),
    ("source_itr", "temporal_encoder"): dict(
        in_ch=5, cond=9, token=3, t_stats=3, s_y=3, aug=False,
    ),
    ("source_itr_sin", "temporal_encoder"): dict(
        in_ch=5, cond=7, token=3, t_stats=3, s_y=3, aug=False,
    ),
    ("interfaces", "temporal_encoder"): dict(
        in_ch=6, cond=3, token=3, t_stats=3, s_y=5, aug=True,
    ),
}


def _make_dataset(trajectories, x_grid, y_grid, t_grid, sim_params, spec):
    sim_ids = np.arange(trajectories.shape[0])
    return SnapshotPairDataset(
        trajectories=trajectories,
        t_grid=t_grid,
        x_grid=x_grid,
        y_grid=y_grid,
        sim_ids=sim_ids,
        sim_params=sim_params,
        mu_global=_SYNTH_MU,
        sigma_global=_SYNTH_SIGMA,
        n_snapshots=6,
        noise_std=0.0,
        problem=spec,
    )


def _adapter_sim_params(spec, trajectories, x_grid, y_grid, t_grid, **extra):
    """Generate valid sim_params for `spec` at the synthetic grid size."""
    X, Y = np.meshgrid(x_grid, y_grid, indexing="ij")
    time_cfg = dict(
        num_sims=trajectories.shape[0],
        dt=float(t_grid[1] - t_grid[0]),
        t_final=float(t_grid[-1]),
        lhs_seed=0,
        t_on=0.0, t_off=0.2, phase=0.0, tukey_alpha=0.5,
        T_right=300.0, b=1.0,
    )
    time_cfg.update(extra)
    params = spec.sample_sim_params(
        rng=np.random.default_rng(0),
        rng_profile=np.random.default_rng(1),
        grids={"X": X, "Y": Y, "x_grid": x_grid, "y_grid": y_grid},
        time_cfg=time_cfg,
    )
    return np.array(params, dtype=object)


@pytest.fixture
def forcing_dataset(synthetic_trajectories, synthetic_sim_params):
    trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
    return _make_dataset(
        trajectories, x_grid, y_grid, t_grid, synthetic_sim_params,
        get_problem("forcing"),
    )


@pytest.fixture
def interfaces_dataset(synthetic_trajectories):
    trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
    spec = get_problem("interfaces")
    sim_params = _adapter_sim_params(spec, trajectories, x_grid, y_grid, t_grid)
    return _make_dataset(trajectories, x_grid, y_grid, t_grid, sim_params, spec)


@pytest.fixture
def source_dataset(synthetic_trajectories):
    trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
    spec = get_problem("source")
    sim_params = _adapter_sim_params(spec, trajectories, x_grid, y_grid, t_grid)
    return _make_dataset(trajectories, x_grid, y_grid, t_grid, sim_params, spec)


@pytest.fixture
def source_itr_dataset(synthetic_trajectories):
    trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
    spec = get_problem("source_itr")
    sim_params = _adapter_sim_params(spec, trajectories, x_grid, y_grid, t_grid)
    return _make_dataset(trajectories, x_grid, y_grid, t_grid, sim_params, spec)


@pytest.fixture
def source_itr_sin_dataset(synthetic_trajectories):
    trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
    spec = get_problem("source_itr_sin")
    sim_params = _adapter_sim_params(spec, trajectories, x_grid, y_grid, t_grid)
    return _make_dataset(trajectories, x_grid, y_grid, t_grid, sim_params, spec)


# ===================== registry / dims =====================

class TestRegistry:
    def test_forcing_registered(self):
        assert isinstance(get_problem("forcing"), ForcingProblem)

    def test_forcing_itr_registered(self):
        assert isinstance(get_problem("forcing_itr"), ForcingItrProblem)

    def test_interfaces_registered(self):
        assert isinstance(get_problem("interfaces"), InterfacesProblem)

    def test_source_registered(self):
        assert isinstance(get_problem("source"), SourceProblem)

    def test_source_itr_registered(self):
        assert isinstance(get_problem("source_itr"), SourceItrProblem)

    def test_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown benchmark"):
            get_problem("does_not_exist")

    def test_default_representation_is_temporal_encoder(self):
        assert get_problem("forcing").representation == "temporal_encoder"

    def test_unknown_representation_raises(self):
        with pytest.raises(ValueError, match="representation"):
            get_problem("forcing", "does_not_exist")

    @pytest.mark.parametrize("key", list(CONTRACTS))
    def test_dims_per_benchmark_representation(self, key):
        name, representation = key
        c = CONTRACTS[key]
        dims = get_problem(name, representation).dims
        assert dims.in_channels == c["in_ch"]
        assert dims.cond_static_dim == c["cond"]
        assert dims.temporal_token_dim == c["token"]
        assert dims.t_stats_dim == c["t_stats"]
        assert dims.s_y_channel == c["s_y"]
        assert dims.use_forcing_time_aug is c["aug"]

    def test_scalar_rc_extender_capability_is_narrow(self):
        eligible = {
            ("forcing", "temporal_encoder"),
            ("interfaces", "temporal_encoder"),
        }
        for key in CONTRACTS:
            dims = get_problem(*key).dims
            expected = 1 if key in eligible else None
            assert dims.forcing_extender_rc_cond_index == expected

    def test_diffusion_geometry_extender_capability_is_forcing_temporal_only(self):
        for key in CONTRACTS:
            dims = get_problem(*key).dims
            assert dims.supports_diffusion_geometry_extender is (
                key == ("forcing", "temporal_encoder")
            )

    @pytest.mark.parametrize(
        ("benchmark", "representation"),
        [("source", "temporal_encoder"), ("interfaces", "temporal_encoder")],
    )
    def test_problem_config_rejects_physics_extender_outside_v1_scope(
        self, benchmark, representation
    ):
        config = {
            "benchmark": {"name": benchmark, "representation": representation},
            "model": {"parameters": {"forcing_spatial_mode": "physics_extender"}},
        }
        with pytest.raises(ValueError, match="fixed-interface"):
            problem_from_config(config)


# ===================== forcing item shape contract =====================

class TestForcingItem:
    """forcing/temporal_encoder item shapes: lean 4-channel spatial, cond 10,
    a 128x3 forcing_seq, T_stats 2."""

    def test_item_keys_shapes(self, forcing_dataset):
        ds = forcing_dataset
        spec = get_problem("forcing")
        sim_id, s, j = ds._pairs[0]
        item = spec.build_item(ds, sim_id, s, j)
        assert set(item) == {"spatial", "cond_static", "forcing_seq", "Y", "T_stats"}
        assert item["spatial"].shape == (ds.Nx, ds.Ny, 4)
        assert item["cond_static"].shape == (FORCING_COND_STATIC_DIM,)
        assert item["forcing_seq"].shape == (
            FORCING_TEMPORAL_SAMPLES, FORCING_TEMPORAL_TOKEN_DIM
        )
        assert item["Y"].shape == (ds.Nx, ds.Ny, 1)
        assert item["T_stats"].shape == (2,)

    def test_getitem_matches_build_item(self, forcing_dataset):
        ds = forcing_dataset
        spec = get_problem("forcing")
        ref = ds[0]
        sim_id, s, j = ds._pairs[0]
        item = spec.build_item(ds, sim_id, s, j)
        assert set(item) == set(ref.keys())
        for k in item:
            np.testing.assert_allclose(item[k], ref[k].numpy(), rtol=0, atol=0)

    def test_static_conditioning_is_invariant_to_source_time_at_fixed_lead(
        self, forcing_dataset
    ):
        ds = forcing_dataset
        spec = get_problem("forcing")
        first = spec.build_item(ds, 0, 0, 1)
        shifted = spec.build_item(ds, 0, 1, 2)
        np.testing.assert_allclose(
            first["cond_static"], shifted["cond_static"], rtol=0.0, atol=1e-7
        )

    @pytest.mark.parametrize(
        ("include_itr", "include_lead_time", "expected_channels"),
        [(True, False, 5), (False, True, 5), (True, True, 6)],
    )
    def test_optional_scalar_spatial_channels(
        self,
        forcing_dataset,
        include_itr,
        include_lead_time,
        expected_channels,
    ):
        ds = forcing_dataset
        spec = problem_from_config({
            "benchmark": {
                "name": "forcing",
                "spatial_input": {
                    "itr": include_itr,
                    "lead_time": include_lead_time,
                },
            }
        })
        ds.problem = spec
        sid, s, j = ds._pairs[-1]
        item = spec.build_item(ds, sid, s, j)

        assert spec.dims.in_channels == expected_channels
        assert item["spatial"].shape == (ds.Nx, ds.Ny, expected_channels)
        next_channel = 4
        if include_itr:
            np.testing.assert_allclose(
                item["spatial"][..., next_channel],
                item["cond_static"][1],
                rtol=0.0,
                atol=0.0,
            )
            next_channel += 1
        if include_lead_time:
            np.testing.assert_allclose(
                item["spatial"][..., next_channel],
                item["cond_static"][0],
                rtol=0.0,
                atol=0.0,
            )


class TestForcingUnknownSpatialFamily:
    """An unregistered family is represented by measurements of its profile."""

    _SINU = {"c0": 0.7, "c1": 0.3, "f": 2.0, "phase": 0.5}

    def test_cond_vector_is_family_independent(self):
        bins = np.linspace(0.2, 0.9, SPATIAL_PROFILE_BINS, dtype=np.float32)
        np.testing.assert_allclose(
            build_cond_vector(
                t_bar_norm=0.1,
                R_c=0.5,
                spatial_profile_bins=bins,
            ),
            np.concatenate([
                np.array(
                    [0.1, (0.5 - RC_RANGE[0]) / (RC_RANGE[1] - RC_RANGE[0])],
                    dtype=np.float32,
                ),
                bins,
            ]),
            rtol=0.0, atol=1e-7,
        )

    def test_eight_bin_averages_are_grid_independent_for_linear_profile(self):
        y = np.linspace(-2.0, 3.0, 17)
        profile = 2.0 * y + 4.0
        got = build_spatial_profile_bin_averages(y, profile)
        edges = np.linspace(y[0], y[-1], SPATIAL_PROFILE_BINS + 1)
        expected = 2.0 * 0.5 * (edges[:-1] + edges[1:]) + 4.0
        np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-6)

    def test_bin_builder_rejects_invalid_arrays(self):
        with pytest.raises(ValueError, match="matching one-dimensional"):
            build_spatial_profile_bin_averages(
                np.linspace(0.0, 1.0, 4), np.ones(3)
            )
        with pytest.raises(ValueError, match="strictly increasing"):
            build_spatial_profile_bin_averages(
                np.array([0.0, 0.5, 0.5, 1.0]), np.ones(4)
            )

    def _sinusoid_dataset(self, synthetic_trajectories, synthetic_sim_params, spec):
        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        params = np.array([dict(p) for p in synthetic_sim_params], dtype=object)
        for p in params:
            p["spatial_family"] = "sinusoid"
            p["spatial_params"] = dict(self._SINU)
        return _make_dataset(trajectories, x_grid, y_grid, t_grid, params, spec)

    def test_build_item_succeeds_for_unregistered_family(
        self, synthetic_trajectories, synthetic_sim_params
    ):
        spec = get_problem("forcing")
        ds = self._sinusoid_dataset(synthetic_trajectories, synthetic_sim_params, spec)
        sim_id, s, j = ds._pairs[0]
        item = spec.build_item(ds, sim_id, s, j)
        assert item["cond_static"].shape == (FORCING_COND_STATIC_DIM,)
        s_y = item["spatial"][..., 3]
        assert float(s_y.max()) > float(s_y.min())
        expected = build_spatial_profile_bin_averages(ds.y_grid, s_y[0])
        np.testing.assert_allclose(
            item["cond_static"][FORCING_SPATIAL_DESCRIPTOR_SLICE],
            expected,
            rtol=0.0,
            atol=0.0,
        )


# ===================== schema validation =====================

class TestForcingSchema:
    def test_accepts_valid(self, synthetic_sim_params):
        spec = get_problem("forcing")
        spec.validate_schema(synthetic_sim_params, np.arange(len(synthetic_sim_params)))

    def test_rejects_missing_key(self):
        spec = get_problem("forcing")
        bad = np.array([{"R_c": 0.5}], dtype=object)
        with pytest.raises(ValueError, match="missing keys"):
            spec.validate_schema(bad, np.array([0]))


# ===================== solver wiring =====================

class TestForcingSolver:
    def test_configure_solver_wires_q_left(self, synthetic_sim_params):
        spec = get_problem("forcing")
        params = synthetic_sim_params[0]
        y_grid = np.linspace(0.0, 1.0, 20)
        layers = [
            Layer2D(x_left=0.0, x_right=0.5, rho=1.0, cp=1.0, k=2.0),
            Layer2D(x_left=0.5, x_right=1.0, rho=1.0, cp=1.0, k=1.0),
        ]
        base_kwargs = dict(
            a=0.0, b=1.0, c=0.0, d=1.0, Nx=20, Ny=20,
            lam_target=0.8, layers=layers, t_final=0.3,
            flux_f=0.0, flux_A=0.0, t_on=0.0, t_off=0.2, phase=0.0,
            dt=0.005, tukey_alpha=0.5, y_grid=y_grid,
        )
        solver = spec.configure_solver(params, base_kwargs)
        assert isinstance(solver, FVSolver2D)
        assert solver.q_left is not None
        assert solver.q_left_integral is not None
        assert solver.interface_R == [params["R_c"]]


# ===================== sample_sim_params parity =====================

class TestForcingSampleParity:
    def test_matches_generate_dataset(self):
        """Adapter sampling reproduces generate_dataset.build_sim_params stream."""
        from data import generate_dataset as gd

        Nx = Ny = 12
        x_grid = np.linspace(0.0, 1.0, Nx)
        y_grid = np.linspace(0.0, 1.0, Ny)
        X, Y = np.meshgrid(x_grid, y_grid, indexing="ij")
        dt, t_final = 0.005, 0.3
        num_sims = 5

        ref = gd.build_sim_params(
            a=0.0, b=1.0, c=0.0, d=1.0, X=X, Y=Y,
            num_sims=num_sims,
            rng=np.random.default_rng(0),
            rng_profile=np.random.default_rng(1),
            dt=dt, t_final=t_final, lhs_seed=0,
            t_on=0.0, t_off=0.2, phase=0.0, tukey_alpha=0.5,
        )

        spec = get_problem("forcing")
        got = spec.sample_sim_params(
            rng=np.random.default_rng(0),
            rng_profile=np.random.default_rng(1),
            grids={"X": X, "Y": Y, "x_grid": x_grid, "y_grid": y_grid},
            time_cfg=dict(
                num_sims=num_sims, dt=dt, t_final=t_final, lhs_seed=0,
                t_on=0.0, t_off=0.2, phase=0.0, tukey_alpha=0.5,
                T_right=300.0, b=1.0,
            ),
        )

        assert len(got) == len(ref) == num_sims
        for g, r in zip(got, ref):
            assert g["R_c"] == r["R_c"]
            assert g["temporal_family"] == r["temporal_family"]
            assert g["spatial_family"] == r["spatial_family"]
            assert g["ic_family"] == r["ic_family"] == "uniform_2d"
            assert g["ic_params"] == r["ic_params"] == {"T0_offset": 0.0}
            np.testing.assert_allclose(g["T0"], r["T0"], rtol=0, atol=0)
            np.testing.assert_allclose(g["T0"], 300.0)


# ===================== interfaces adapter =====================

class TestInterfacesItem:
    def test_item_keys_shapes(self, interfaces_dataset):
        ds = interfaces_dataset
        spec = get_problem("interfaces")
        sim_id, s, j = ds._pairs[0]
        item = spec.build_item(ds, sim_id, s, j)
        assert set(item) == {"spatial", "cond_static", "forcing_seq", "Y", "T_stats"}
        assert item["spatial"].shape == (ds.Nx, ds.Ny, 6)
        assert item["cond_static"].shape == (3,)
        assert item["forcing_seq"].shape == (
            FORCING_TEMPORAL_SAMPLES, FORCING_TEMPORAL_TOKEN_DIM
        )
        assert item["Y"].shape == (ds.Nx, ds.Ny, 1)
        assert item["T_stats"].shape == (3,)

    def test_t_stats_carries_interface_x(self, interfaces_dataset):
        ds = interfaces_dataset
        spec = get_problem("interfaces")
        sim_id, s, j = ds._pairs[0]
        item = spec.build_item(ds, sim_id, s, j)
        expected = np.float32(ds.sim_params[int(sim_id)]["interface_x"])
        assert item["T_stats"][2] == expected

    def test_getitem_matches_build_item(self, interfaces_dataset):
        ds = interfaces_dataset
        spec = get_problem("interfaces")
        ref = ds[0]
        sim_id, s, j = ds._pairs[0]
        item = spec.build_item(ds, sim_id, s, j)
        assert set(item) == set(ref.keys())
        for k in item:
            np.testing.assert_allclose(item[k], ref[k].numpy(), rtol=0, atol=0)


class TestInterfacesSchema:
    def test_accepts_valid(self, interfaces_dataset):
        ds = interfaces_dataset
        get_problem("interfaces").validate_schema(ds.sim_params, ds.sim_ids)

    def test_admits_family_transfer_families(self):
        # The family_transfer OOD axis drives exp/patch, so the schema now
        # admits the {sin,exp} x {uniform,patch} grid (not sin/uniform only).
        spec = get_problem("interfaces")
        for tf, sf in [("sin", "uniform"), ("exp", "uniform"),
                       ("sin", "patch"), ("exp", "patch")]:
            ok = np.array([{
                "R_c": 0.5, "interface_x": 0.5,
                "temporal_family": tf, "temporal_params": {},
                "spatial_family": sf, "spatial_params": {},
            }], dtype=object)
            spec.validate_schema(ok, np.array([0]))

    def test_rejects_out_of_set_families(self):
        spec = get_problem("interfaces")
        bad_temporal = np.array([{
            "R_c": 0.5, "interface_x": 0.5,
            "temporal_family": "pulse_train", "temporal_params": {},
            "spatial_family": "uniform", "spatial_params": {},
        }], dtype=object)
        with pytest.raises(ValueError, match="admits only"):
            spec.validate_schema(bad_temporal, np.array([0]))
        bad_spatial = np.array([{
            "R_c": 0.5, "interface_x": 0.5,
            "temporal_family": "sin", "temporal_params": {},
            "spatial_family": "gaussian", "spatial_params": {},
        }], dtype=object)
        with pytest.raises(ValueError, match="admits only"):
            spec.validate_schema(bad_spatial, np.array([0]))

    def test_rejects_missing_interface_x(self):
        spec = get_problem("interfaces")
        bad = np.array([{
            "R_c": 0.5,
            "temporal_family": "sin", "temporal_params": {},
            "spatial_family": "uniform", "spatial_params": {},
        }], dtype=object)
        with pytest.raises(ValueError, match="missing keys"):
            spec.validate_schema(bad, np.array([0]))


class TestInterfacesSolver:
    def test_layers_wired_from_interface_x(self, interfaces_dataset):
        ds = interfaces_dataset
        spec = get_problem("interfaces")
        params = ds.sim_params[0]
        base_kwargs = dict(
            a=0.0, b=1.0, c=0.0, d=1.0, Nx=ds.Nx, Ny=ds.Ny,
            lam_target=0.8, layers=None, t_final=0.3,
            flux_f=0.0, flux_A=0.0, t_on=0.0, t_off=0.2, phase=0.0,
            dt=0.005, tukey_alpha=0.5, y_grid=ds.y_grid,
        )
        solver = spec.configure_solver(params, base_kwargs)
        assert isinstance(solver, FVSolver2D)
        # interfaces wires a vector q_left but no closed-form integral, and no
        # volumetric source -> distinguishes it from forcing and source.
        assert solver.q_left is not None
        assert solver.q_left_integral is None
        assert solver.source is None
        assert solver.interface_R == [params["R_c"]]


class TestInterfacesSampleParity:
    def test_sin_uniform_and_ranges(self, interfaces_dataset):
        ds = interfaces_dataset
        for p in ds.sim_params:
            assert p["temporal_family"] == "sin"
            assert p["spatial_family"] == "uniform"
            assert 0.05 <= p["R_c"] <= 1.0
            assert 0.2 <= p["interface_x"] <= 0.8

    def test_interfaces_keep_varying_initial_conditions(self, interfaces_dataset):
        ds = interfaces_dataset
        assert any(not np.allclose(p["T0"], 300.0) for p in ds.sim_params)


# ===================== source adapter =====================

class TestSourceItem:
    def test_item_keys_shapes(self, source_dataset):
        ds = source_dataset
        spec = get_problem("source")
        sim_id, s, j = ds._pairs[0]
        item = spec.build_item(ds, sim_id, s, j)
        # source/temporal_encoder: lean 4-channel spatial, cond 6, pulse 128x3.
        assert set(item) == {"spatial", "cond_static", "forcing_seq", "Y", "T_stats"}
        assert item["spatial"].shape == (ds.Nx, ds.Ny, 4)
        assert item["cond_static"].shape == (6,)
        assert item["forcing_seq"].shape == (
            FORCING_TEMPORAL_SAMPLES, FORCING_TEMPORAL_TOKEN_DIM
        )
        assert item["Y"].shape == (ds.Nx, ds.Ny, 1)
        assert item["T_stats"].shape == (3,)

    def test_t_stats_carries_interface_x(self, source_dataset):
        ds = source_dataset
        spec = get_problem("source")
        sim_id, s, j = ds._pairs[0]
        item = spec.build_item(ds, sim_id, s, j)
        assert item["T_stats"][2] == np.float32(0.5)

    def test_getitem_matches_build_item(self, source_dataset):
        ds = source_dataset
        spec = get_problem("source")
        ref = ds[0]
        sim_id, s, j = ds._pairs[0]
        item = spec.build_item(ds, sim_id, s, j)
        assert set(item) == set(ref.keys())
        for k in item:
            np.testing.assert_allclose(item[k], ref[k].numpy(), rtol=0, atol=0)


class TestSourceSchema:
    def test_accepts_valid(self, source_dataset):
        ds = source_dataset
        get_problem("source").validate_schema(ds.sim_params, ds.sim_ids)

    def test_rejects_missing_patch_keys(self):
        spec = get_problem("source")
        bad = np.array([{"R_c": 0.5, "interface_x": 0.5}], dtype=object)
        with pytest.raises(ValueError, match="missing keys"):
            spec.validate_schema(bad, np.array([0]))

    def test_rejects_legacy_forcing_keys(self):
        spec = get_problem("source")
        bad = np.array([{
            "R_c": 0.5, "interface_x": 0.5, "x_h": 0.3, "y_h": 0.5,
            "A": 5000.0, "t_off": 0.225, "w_h": 0.1, "h_h": 0.1,
            "temporal_family": "sin", "temporal_params": {},
        }], dtype=object)
        with pytest.raises(ValueError, match="legacy keys"):
            spec.validate_schema(bad, np.array([0]))


class TestSourceSolver:
    def test_wires_volumetric_source(self, source_dataset):
        ds = source_dataset
        spec = get_problem("source")
        params = ds.sim_params[0]
        # Independent Nx=Ny=20 grid so the fixed interface at x=0.5 lands
        # off-node (the solver rejects on-node interfaces); production uses
        # Nx=100 where 0.5 is likewise off-node.
        Nx = Ny = 20
        x_grid = np.linspace(0.0, 1.0, Nx)
        y_grid = np.linspace(0.0, 1.0, Ny)
        X, Y = np.meshgrid(x_grid, y_grid, indexing="ij")
        base_kwargs = dict(
            a=0.0, b=1.0, c=0.0, d=1.0, Nx=Nx, Ny=Ny,
            lam_target=0.8, layers=None, t_final=0.3,
            flux_f=0.0, flux_A=0.0, t_on=0.0, t_off=0.2, phase=0.0,
            dt=0.005, tukey_alpha=0.5, y_grid=y_grid, X=X, Y=Y,
        )
        solver = spec.configure_solver(params, base_kwargs)
        assert isinstance(solver, FVSolver2D)
        # source benchmark drives heating via a volumetric source, not a
        # closed-form flux integral.
        assert solver.source is not None
        assert solver.q_left_integral is None
        assert solver.interface_R == [params["R_c"]]


class TestSourceSampleParity:
    def test_regimes_and_ranges(self, source_dataset):
        ds = source_dataset
        regimes = set()
        for p in ds.sim_params:
            assert 0.05 <= p["R_c"] <= 1.0
            assert p["interface_x"] == 0.5
            assert p["A"] > 0.0
            regimes.add(p["regime"])
        # stratified x_h sampling should populate more than one regime
        assert regimes <= {"left", "near", "right"}
        assert len(regimes) >= 2

    def test_uniform_300_initial_conditions(self, source_dataset):
        ds = source_dataset
        for p in ds.sim_params:
            assert p["ic_family"] == "uniform_2d"
            assert p["ic_params"] == {"T0_offset": 0.0}
            assert p["T0"].shape == (ds.Nx, ds.Ny)
            assert p["T0"].dtype == np.float32
            np.testing.assert_allclose(p["T0"], 300.0)


# ===================== source_itr adapter =====================

class TestSourceItrItem:
    def test_item_keys_shapes(self, source_itr_dataset):
        ds = source_itr_dataset
        spec = get_problem("source_itr")
        sim_id, s, j = ds._pairs[0]
        item = spec.build_item(ds, sim_id, s, j)
        # source_itr/temporal_encoder: 5-channel spatial (adds Rc_y), cond 10.
        assert set(item) == {"spatial", "cond_static", "forcing_seq", "Y", "T_stats"}
        assert item["spatial"].shape == (ds.Nx, ds.Ny, 5)
        assert item["cond_static"].shape == (9,)
        assert item["forcing_seq"].shape == (
            FORCING_TEMPORAL_SAMPLES, FORCING_TEMPORAL_TOKEN_DIM
        )
        assert item["Y"].shape == (ds.Nx, ds.Ny, 1)
        assert item["T_stats"].shape == (3,)

    def test_rc_channel_index_is_4(self, source_itr_dataset):
        # The normalized R_c(y) channel is spatial index 4; under broadcast it is
        # constant in x (one column of the y-profile reproduces the whole field).
        ds = source_itr_dataset
        spec = get_problem("source_itr")
        sim_id, s, j = ds._pairs[0]
        item = spec.build_item(ds, sim_id, s, j)
        rc_channel = item["spatial"][:, :, 4]
        # broadcast mode: every x-row equals the y-profile.
        np.testing.assert_allclose(
            rc_channel,
            np.broadcast_to(rc_channel[0:1, :], rc_channel.shape),
            rtol=0, atol=0,
        )

    def test_rc_channel_matches_log_norm_profile(self, source_itr_dataset):
        from problems.source_itr import rc_log_norm
        from src.physics.internal_source import make_rc_void_profile

        ds = source_itr_dataset
        spec = get_problem("source_itr")
        sim_id, s, j = ds._pairs[0]
        p = ds.sim_params[int(sim_id)]
        item = spec.build_item(ds, sim_id, s, j)
        expected = rc_log_norm(
            make_rc_void_profile(
                ds.y_grid,
                R_base=float(p["R_c_base"]), R_amp=float(p["R_c_amp"]),
                y0=float(p["R_c_y0"]), sigma=float(p["R_c_sigma"]),
            )
        )
        np.testing.assert_allclose(item["spatial"][0, :, 4], expected, rtol=0, atol=0)

    def test_localized_mode_decays_away_from_interface(self, synthetic_trajectories):
        # localized mode multiplies the y-profile by exp(-((x - x_I)/ell)^2), so
        # the channel magnitude at x far from the interface is below that at x_I.
        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        spec = SourceItrProblem("temporal_encoder")
        spec.rc_channel_mode = "localized"
        spec.rc_ell = 0.05
        sim_params = _adapter_sim_params(spec, trajectories, x_grid, y_grid, t_grid)
        ds = _make_dataset(trajectories, x_grid, y_grid, t_grid, sim_params, spec)
        sim_id, s, j = ds._pairs[0]
        item = spec.build_item(ds, sim_id, s, j)
        rc_channel = item["spatial"][:, :, 4]
        i_near = int(np.argmin(np.abs(ds.x_grid - 0.5)))
        col = int(np.argmax(np.abs(rc_channel[i_near, :])))
        assert abs(rc_channel[0, col]) < abs(rc_channel[i_near, col])

    def test_t_stats_carries_interface_x(self, source_itr_dataset):
        ds = source_itr_dataset
        spec = get_problem("source_itr")
        sim_id, s, j = ds._pairs[0]
        item = spec.build_item(ds, sim_id, s, j)
        assert item["T_stats"][2] == np.float32(0.5)

    def test_getitem_matches_build_item(self, source_itr_dataset):
        ds = source_itr_dataset
        spec = get_problem("source_itr")
        ref = ds[0]
        sim_id, s, j = ds._pairs[0]
        item = spec.build_item(ds, sim_id, s, j)
        assert set(item) == set(ref.keys())
        for k in item:
            np.testing.assert_allclose(item[k], ref[k].numpy(), rtol=0, atol=0)


class TestSourceItrSchema:
    def test_accepts_valid(self, source_itr_dataset):
        ds = source_itr_dataset
        get_problem("source_itr").validate_schema(ds.sim_params, ds.sim_ids)

    def test_rejects_missing_void_keys(self):
        spec = get_problem("source_itr")
        # All source keys present but the four void keys are missing.
        bad = np.array([{
            "R_c": 0.5, "interface_x": 0.5, "x_h": 0.3, "y_h": 0.5,
            "A": 5000.0, "t_off": 0.225, "w_h": 0.1, "h_h": 0.1,
        }], dtype=object)
        with pytest.raises(ValueError, match="missing keys"):
            spec.validate_schema(bad, np.array([0]))


class TestSourceItrSampleParity:
    def test_void_params_present_and_bounded(self, source_itr_dataset):
        from src.physics.internal_source import RC_VOID_RANGES, R_PEAK_MAX
        ds = source_itr_dataset
        b_lo, b_hi = RC_VOID_RANGES["R_base"]
        y0_lo, y0_hi = RC_VOID_RANGES["y0"]
        s_lo, s_hi = RC_VOID_RANGES["sigma"]
        for p in ds.sim_params:
            assert b_lo <= p["R_c_base"] <= b_hi
            assert 0.0 <= p["R_c_amp"] <= R_PEAK_MAX - p["R_c_base"] + 1e-9
            assert y0_lo <= p["R_c_y0"] <= y0_hi
            assert s_lo <= p["R_c_sigma"] <= s_hi
            # R_c mirrors R_c_base so the universal column stays valid.
            assert p["R_c"] == pytest.approx(p["R_c_base"])

    def test_patch_sampling_matches_source(self, synthetic_trajectories):
        # source_itr must reuse source's patch/A/IC stream unchanged.
        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        src = get_problem("source")
        itr = get_problem("source_itr")
        p_src = _adapter_sim_params(src, trajectories, x_grid, y_grid, t_grid)
        p_itr = _adapter_sim_params(itr, trajectories, x_grid, y_grid, t_grid)
        for a, b in zip(p_src, p_itr):
            assert a["x_h"] == pytest.approx(b["x_h"])
            assert a["y_h"] == pytest.approx(b["y_h"])
            assert a["A"] == pytest.approx(b["A"])
            assert a["regime"] == b["regime"]


class TestSourceItrSolver:
    def test_wires_per_row_interface_resistance(self, synthetic_trajectories):
        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        spec = get_problem("source_itr")
        sim_params = _adapter_sim_params(spec, trajectories, x_grid, y_grid, t_grid)
        params = sim_params[0]
        Nx = Ny = 20
        xg = np.linspace(0.0, 1.0, Nx)
        yg = np.linspace(0.0, 1.0, Ny)
        X, Y = np.meshgrid(xg, yg, indexing="ij")
        base_kwargs = dict(
            a=0.0, b=1.0, c=0.0, d=1.0, Nx=Nx, Ny=Ny,
            lam_target=0.8, layers=None, t_final=0.3,
            flux_f=0.0, flux_A=0.0, t_on=0.0, t_off=0.2, phase=0.0,
            dt=0.005, tukey_alpha=0.5, y_grid=yg, X=X, Y=Y,
        )
        solver = spec.configure_solver(params, base_kwargs)
        assert isinstance(solver, FVSolver2D)
        assert solver.source is not None
        # interface_R is a per-row (Ny,) profile, not a scalar.
        rc = solver.interface_R[0]
        assert np.ndim(rc) == 1
        assert np.shape(rc) == (Ny,)


class TestSourceItrValPairRow:
    def test_fields_and_row(self, source_itr_dataset):
        ds = source_itr_dataset
        spec = get_problem("source_itr")
        assert spec.val_pair_fields == (
            "x_h", "y_h", "A", "regime", "R_c_amp", "R_c_y0", "R_c_sigma"
        )
        sim_id, s, j = ds._pairs[0]
        row = spec.val_pair_row(ds, sim_id, s, j)
        assert set(row) == set(spec.val_pair_fields)
        p = ds.sim_params[int(sim_id)]
        assert row["R_c_amp"] == pytest.approx(float(p["R_c_amp"]))
        assert row["R_c_y0"] == pytest.approx(float(p["R_c_y0"]))
        assert row["R_c_sigma"] == pytest.approx(float(p["R_c_sigma"]))
        # void benchmark still forbids family columns.
        assert "temporal_family" not in row
        assert "spatial_family" not in row


# ===================== source_itr_sin adapter =====================

class TestSourceItrSinItem:
    def test_item_keys_shapes(self, source_itr_sin_dataset):
        ds = source_itr_sin_dataset
        spec = get_problem("source_itr_sin")
        sim_id, s, j = ds._pairs[0]
        item = spec.build_item(ds, sim_id, s, j)
        # source_itr_sin/temporal_encoder: 5-channel spatial (Rc_y at 4), cond 7.
        assert set(item) == {"spatial", "cond_static", "forcing_seq", "Y", "T_stats"}
        assert item["spatial"].shape == (ds.Nx, ds.Ny, 5)
        assert item["cond_static"].shape == (7,)
        assert item["forcing_seq"].shape == (
            FORCING_TEMPORAL_SAMPLES, FORCING_TEMPORAL_TOKEN_DIM
        )
        assert item["Y"].shape == (ds.Nx, ds.Ny, 1)
        assert item["T_stats"].shape == (3,)

    def test_rc_channel_index_is_4(self, source_itr_sin_dataset):
        ds = source_itr_sin_dataset
        spec = get_problem("source_itr_sin")
        sim_id, s, j = ds._pairs[0]
        item = spec.build_item(ds, sim_id, s, j)
        rc_channel = item["spatial"][:, :, 4]
        np.testing.assert_allclose(
            rc_channel,
            np.broadcast_to(rc_channel[0:1, :], rc_channel.shape),
            rtol=0, atol=0,
        )

    def test_rc_channel_matches_sin_log_norm_profile(self, source_itr_sin_dataset):
        from problems.source_itr import rc_log_norm
        from src.physics.internal_source import make_rc_sin_profile

        ds = source_itr_sin_dataset
        spec = get_problem("source_itr_sin")
        sim_id, s, j = ds._pairs[0]
        p = ds.sim_params[int(sim_id)]
        item = spec.build_item(ds, sim_id, s, j)
        expected = rc_log_norm(
            make_rc_sin_profile(
                ds.y_grid, R_base=float(p["R_c_base"]), A=float(p["R_c_A"]),
            )
        )
        np.testing.assert_allclose(item["spatial"][0, :, 4], expected, rtol=0, atol=0)

    def test_getitem_matches_build_item(self, source_itr_sin_dataset):
        ds = source_itr_sin_dataset
        spec = get_problem("source_itr_sin")
        ref = ds[0]
        sim_id, s, j = ds._pairs[0]
        item = spec.build_item(ds, sim_id, s, j)
        assert set(item) == set(ref.keys())
        for k in item:
            np.testing.assert_allclose(item[k], ref[k].numpy(), rtol=0, atol=0)


class TestSourceItrSinSchema:
    def test_accepts_valid(self, source_itr_sin_dataset):
        ds = source_itr_sin_dataset
        get_problem("source_itr_sin").validate_schema(ds.sim_params, ds.sim_ids)

    def test_rejects_void_keys(self):
        spec = get_problem("source_itr_sin")
        bad = np.array([{
            "R_c": 0.5, "interface_x": 0.5, "x_h": 0.3, "y_h": 0.5,
            "A": 5000.0, "t_off": 0.225, "w_h": 0.1, "h_h": 0.1,
            "R_c_base": 0.5, "R_c_A": 0.5,
            "R_c_amp": 0.5, "R_c_y0": 0.5, "R_c_sigma": 0.1,
        }], dtype=object)
        with pytest.raises(ValueError, match="forbidden keys"):
            spec.validate_schema(bad, np.array([0]))


class TestSourceItrSinSampleParity:
    def test_sin_params_present_and_bounded(self, source_itr_sin_dataset):
        from src.physics.internal_source import RC_SIN_RANGES, R_PEAK_MAX
        ds = source_itr_sin_dataset
        b_lo, b_hi = RC_SIN_RANGES["R_base"]
        for p in ds.sim_params:
            assert b_lo <= p["R_c_base"] <= b_hi
            assert 0.0 <= p["R_c_A"] <= R_PEAK_MAX - p["R_c_base"] + 1e-9
            assert p["R_c"] == pytest.approx(p["R_c_base"])
            # Void keys must be absent (popped after the parent sampler).
            assert "R_c_amp" not in p
            assert "R_c_y0" not in p
            assert "R_c_sigma" not in p

    def test_patch_sampling_matches_source(self, synthetic_trajectories):
        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        src = get_problem("source")
        sin = get_problem("source_itr_sin")
        p_src = _adapter_sim_params(src, trajectories, x_grid, y_grid, t_grid)
        p_sin = _adapter_sim_params(sin, trajectories, x_grid, y_grid, t_grid)
        for a, b in zip(p_src, p_sin):
            assert a["x_h"] == pytest.approx(b["x_h"])
            assert a["y_h"] == pytest.approx(b["y_h"])
            assert a["A"] == pytest.approx(b["A"])
            assert a["regime"] == b["regime"]


class TestSourceItrSinSolver:
    def test_wires_per_row_interface_resistance(self, synthetic_trajectories):
        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        spec = get_problem("source_itr_sin")
        sim_params = _adapter_sim_params(spec, trajectories, x_grid, y_grid, t_grid)
        params = sim_params[0]
        Nx = Ny = 20
        xg = np.linspace(0.0, 1.0, Nx)
        yg = np.linspace(0.0, 1.0, Ny)
        X, Y = np.meshgrid(xg, yg, indexing="ij")
        base_kwargs = dict(
            a=0.0, b=1.0, c=0.0, d=1.0, Nx=Nx, Ny=Ny,
            lam_target=0.8, layers=None, t_final=0.3,
            flux_f=0.0, flux_A=0.0, t_on=0.0, t_off=0.2, phase=0.0,
            dt=0.005, tukey_alpha=0.5, y_grid=yg, X=X, Y=Y,
        )
        solver = spec.configure_solver(params, base_kwargs)
        assert isinstance(solver, FVSolver2D)
        assert solver.source is not None
        rc = solver.interface_R[0]
        assert np.ndim(rc) == 1
        assert np.shape(rc) == (Ny,)


class TestSourceItrSinValPairRow:
    def test_fields_and_row(self, source_itr_sin_dataset):
        ds = source_itr_sin_dataset
        spec = get_problem("source_itr_sin")
        assert spec.val_pair_fields == ("x_h", "y_h", "A", "regime", "R_c_A")
        sim_id, s, j = ds._pairs[0]
        row = spec.val_pair_row(ds, sim_id, s, j)
        assert set(row) == set(spec.val_pair_fields)
        p = ds.sim_params[int(sim_id)]
        assert row["R_c_A"] == pytest.approx(float(p["R_c_A"]))
        assert "R_c_amp" not in row
        assert "temporal_family" not in row


class TestProblemFromConfig:
    @pytest.mark.parametrize(
        "name,cls",
        [
            ("forcing", ForcingProblem),
            ("forcing_itr", ForcingItrProblem),
            ("interfaces", InterfacesProblem),
            ("source", SourceProblem),
        ],
    )
    def test_resolves_active_benchmark(self, name, cls):
        spec = problem_from_config({"benchmark": {"name": name}})
        assert isinstance(spec, cls)
        assert spec.name == name

    def test_defaults_to_forcing_when_missing(self):
        assert isinstance(problem_from_config({}), ForcingProblem)
        assert isinstance(problem_from_config({"benchmark": {}}), ForcingProblem)

    def test_rejects_retired_representation(self):
        with pytest.raises(ValueError, match="representation"):
            problem_from_config(
                {"benchmark": {"name": "forcing", "representation": "bins"}}
            )

    def test_defaults_to_temporal_encoder_when_representation_missing(self):
        spec = problem_from_config({"benchmark": {"name": "source"}})
        assert spec.representation == "temporal_encoder"

    @pytest.mark.parametrize("name", ("forcing", "forcing_itr", "forcing_itr_sin"))
    def test_forcing_family_accepts_eight_profile_bins(self, name):
        spec = problem_from_config({
            "benchmark": {"name": name, "spatial_profile_bins": 8}
        })
        assert spec.spatial_profile_bins == 8

    def test_forcing_rejects_other_profile_bin_width(self):
        with pytest.raises(ValueError, match="requires spatial_profile_bins=8"):
            problem_from_config({
                "benchmark": {"name": "forcing", "spatial_profile_bins": 7}
            })

    def test_source_itr_defaults_broadcast_channel_mode(self):
        spec = problem_from_config({"benchmark": {"name": "source_itr"}})
        assert isinstance(spec, SourceItrProblem)
        assert spec.rc_channel_mode == "broadcast"

    def test_source_itr_reads_rc_channel_knobs(self):
        spec = problem_from_config({
            "benchmark": {
                "name": "source_itr",
                "rc_channel_mode": "localized",
                "rc_ell": 0.08,
            }
        })
        assert spec.rc_channel_mode == "localized"
        assert spec.rc_ell == pytest.approx(0.08)

    def test_forcing_reads_optional_spatial_input_knobs(self):
        spec = problem_from_config({
            "benchmark": {
                "name": "forcing",
                "spatial_input": {"itr": True, "lead_time": True},
            }
        })
        assert spec.spatial_input_itr is True
        assert spec.spatial_input_lead_time is True
        assert spec.dims.in_channels == 6

    @pytest.mark.parametrize("value", [1, "true", None])
    def test_forcing_rejects_non_boolean_spatial_input_knobs(self, value):
        with pytest.raises(ValueError, match="must be a boolean"):
            problem_from_config({
                "benchmark": {
                    "name": "forcing",
                    "spatial_input": {"itr": value},
                }
            })

    def test_other_benchmarks_reject_enabled_scalar_spatial_input(self):
        with pytest.raises(ValueError, match="only by the forcing benchmark"):
            problem_from_config({
                "benchmark": {
                    "name": "source",
                    "spatial_input": {"lead_time": True},
                }
            })


class TestGeometryAwareProblemSampling:
    def _grids(self, *, a=2.0, b=4.0, c=-1.0, d=1.0, Nx=40, Ny=40):
        x_grid = np.linspace(a, b, Nx)
        y_grid = np.linspace(c, d, Ny)
        X, Y = np.meshgrid(x_grid, y_grid, indexing="ij")
        return x_grid, y_grid, X, Y

    def test_forcing_spatial_params_are_physical_and_bounded(self):
        x_grid, y_grid, X, Y = self._grids(c=-1.0, d=2.0, Nx=39, Ny=58)
        spec = get_problem("forcing")
        params = spec.sample_sim_params(
            rng=np.random.default_rng(0),
            rng_profile=np.random.default_rng(1),
            grids={"X": X, "Y": Y, "x_grid": x_grid, "y_grid": y_grid},
            time_cfg=dict(num_sims=20, dt=0.005, t_final=0.3, lhs_seed=0,
                          t_on=0.0, t_off=0.2, phase=0.0, tukey_alpha=0.5),
        )
        c, d = float(y_grid[0]), float(y_grid[-1])
        for p in params:
            sp = p["spatial_params"]
            if p["spatial_family"] == "patch":
                assert sp["y_c"] - 0.5 * sp["w"] >= c - 1e-12
                assert sp["y_c"] + 0.5 * sp["w"] <= d + 1e-12
            elif p["spatial_family"] == "triangle":
                assert sp["y_c"] - sp["ell"] >= c - 1e-12
                assert sp["y_c"] + sp["ell"] <= d + 1e-12
            elif p["spatial_family"] == "gaussian":
                assert c <= sp["y_c"] <= d

    def test_interfaces_metadata_is_physical_and_solver_layers_use_domain(self):
        x_grid, y_grid, X, Y = self._grids()
        spec = get_problem("interfaces")
        params = spec.sample_sim_params(
            rng=np.random.default_rng(0),
            rng_profile=np.random.default_rng(1),
            grids={"X": X, "Y": Y, "x_grid": x_grid, "y_grid": y_grid},
            time_cfg=dict(num_sims=5, dt=0.005, t_final=0.3, lhs_seed=0,
                          t_on=0.0, t_off=0.2, phase=0.0, tukey_alpha=0.5,
                          T_right=300.0, b=float(x_grid[-1])),
        )
        assert all(2.4 <= p["interface_x"] <= 3.6 for p in params)
        base_kwargs = dict(
            a=2.0, b=4.0, c=-1.0, d=1.0, Nx=40, Ny=40,
            lam_target=0.8, layers=None, t_final=0.3,
            flux_f=0.0, flux_A=0.0, t_on=0.0, t_off=0.2, phase=0.0,
            dt=0.005, tukey_alpha=0.5, y_grid=y_grid,
        )
        solver = spec.configure_solver(params[0], base_kwargs)
        assert solver.layers[0].x_left == pytest.approx(2.0)
        assert solver.layers[-1].x_right == pytest.approx(4.0)
        assert solver.interface_positions[0] == pytest.approx(params[0]["interface_x"])

    def test_source_patch_metadata_is_physical_and_bounded(self):
        x_grid, y_grid, X, Y = self._grids()
        spec = get_problem("source")
        params = spec.sample_sim_params(
            rng=np.random.default_rng(0),
            rng_profile=np.random.default_rng(1),
            grids={"X": X, "Y": Y, "x_grid": x_grid, "y_grid": y_grid},
            time_cfg=dict(num_sims=20, dt=0.005, t_final=0.3, lhs_seed=0),
        )
        a, b = float(x_grid[0]), float(x_grid[-1])
        c, d = float(y_grid[0]), float(y_grid[-1])
        for p in params:
            assert p["interface_x"] == pytest.approx(3.0)
            assert p["x_h"] - 0.5 * p["w_h"] >= a - 1e-12
            assert p["x_h"] + 0.5 * p["w_h"] <= b + 1e-12
            assert p["y_h"] - 0.5 * p["h_h"] >= c - 1e-12
            assert p["y_h"] + 0.5 * p["h_h"] <= d + 1e-12


# ===================== representation x benchmark contracts =====================

def _dataset_for(name, representation, synthetic_trajectories, forcing_sim_params):
    """Build a dataset for any (benchmark, representation) at the synthetic grid."""
    trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
    spec = get_problem(name, representation)
    if name == "forcing":
        sim_params = forcing_sim_params
    else:
        sim_params = _adapter_sim_params(spec, trajectories, x_grid, y_grid, t_grid)
    return spec, _make_dataset(trajectories, x_grid, y_grid, t_grid, sim_params, spec)


class TestRepresentationContracts:
    @pytest.mark.parametrize("key", list(CONTRACTS))
    def test_item_shapes(self, key, synthetic_trajectories, synthetic_sim_params):
        name, representation = key
        c = CONTRACTS[key]
        spec, ds = _dataset_for(
            name, representation, synthetic_trajectories, synthetic_sim_params
        )
        sim_id, s, j = ds._pairs[0]
        item = spec.build_item(ds, sim_id, s, j)

        assert item["spatial"].shape == (ds.Nx, ds.Ny, c["in_ch"])
        assert item["cond_static"].shape == (c["cond"],)
        assert item["Y"].shape == (ds.Nx, ds.Ny, 1)
        assert item["T_stats"].shape == (c["t_stats"],)
        assert item["forcing_seq"].shape == (FORCING_TEMPORAL_SAMPLES, c["token"])

    @pytest.mark.parametrize("key", list(CONTRACTS))
    def test_getitem_matches_build_item(self, key, synthetic_trajectories, synthetic_sim_params):
        name, representation = key
        spec, ds = _dataset_for(
            name, representation, synthetic_trajectories, synthetic_sim_params
        )
        ref = ds[0]
        sim_id, s, j = ds._pairs[0]
        item = spec.build_item(ds, sim_id, s, j)
        assert set(item) == set(ref.keys())
        for k in item:
            np.testing.assert_allclose(item[k], ref[k].numpy(), rtol=0, atol=0)


class TestRepresentationAbsence:
    def test_source_cond_has_no_A_norm(self, synthetic_trajectories, synthetic_sim_params):
        # source cond is base 2 + (x_h, y_h, w_h, h_h) = 6; A_norm is dropped.
        spec, ds = _dataset_for(
            "source", "temporal_encoder", synthetic_trajectories, synthetic_sim_params
        )
        sim_id, s, j = ds._pairs[0]
        item = spec.build_item(ds, sim_id, s, j)
        assert item["cond_static"].shape == (6,)

    def test_interfaces_temporal_reads_s_y_channel_5(self):
        # The learned spatial forcing must read s_y at channel 5, not K_norm at 3.
        assert get_problem("interfaces", "temporal_encoder").dims.s_y_channel == 5
        assert get_problem("forcing", "temporal_encoder").dims.s_y_channel == 3
        assert get_problem("source", "temporal_encoder").dims.s_y_channel == 3


# ===================== val-pair logging hooks =====================

class TestValPairRow:
    """Each benchmark contributes exactly its declared val_pair_fields to
    val_pairs.csv, so the superset header stays consistent and no benchmark
    leaks columns it forbids elsewhere (e.g. source must not emit families)."""

    def test_forcing_fields_and_row(self, forcing_dataset):
        ds = forcing_dataset
        spec = get_problem("forcing")
        assert spec.val_pair_fields == ("temporal_family", "spatial_family")
        sim_id, s, j = ds._pairs[0]
        row = spec.val_pair_row(ds, sim_id, s, j)
        assert set(row) == set(spec.val_pair_fields)
        p = ds.sim_params[int(sim_id)]
        assert row["temporal_family"] == p["temporal_family"]
        assert row["spatial_family"] == p["spatial_family"]

    def test_interfaces_fields_and_row(self, interfaces_dataset):
        ds = interfaces_dataset
        spec = get_problem("interfaces")
        assert spec.val_pair_fields == ("interface_x",)
        sim_id, s, j = ds._pairs[0]
        row = spec.val_pair_row(ds, sim_id, s, j)
        assert set(row) == set(spec.val_pair_fields)
        assert row["interface_x"] == pytest.approx(
            float(ds.sim_params[int(sim_id)]["interface_x"])
        )

    def test_source_fields_and_row(self, source_dataset):
        ds = source_dataset
        spec = get_problem("source")
        assert spec.val_pair_fields == ("x_h", "y_h", "A", "regime")
        sim_id, s, j = ds._pairs[0]
        row = spec.val_pair_row(ds, sim_id, s, j)
        assert set(row) == set(spec.val_pair_fields)
        p = ds.sim_params[int(sim_id)]
        assert row["x_h"] == pytest.approx(float(p["x_h"]))
        assert row["y_h"] == pytest.approx(float(p["y_h"]))
        assert row["A"] == pytest.approx(float(p["A"]))
        assert row["regime"] == p["regime"]
        # source forbids family columns everywhere; the val-pair hook must not
        # reintroduce them.
        assert "temporal_family" not in row
        assert "spatial_family" not in row

    def test_base_default_is_empty(self):
        from problems.base import ProblemSpec

        assert ProblemSpec.val_pair_fields == ()


# OOD-axis contract: the exact set of axes each benchmark declares plus each
# axis's kind. Mirrors the plan's OOD axis catalog; the per-axis coherence test
# below enforces the shared invariants (CRN-paired sweep ordering, kind/compound
# agreement, evaluation-parameter axes carry no sim field).
OOD_AXIS_CONTRACTS = {
    "forcing": {
        "rc": "simulation_parameter",
        "sin_freq": "simulation_parameter",
        "sin_amp": "simulation_parameter",
        "pulse_count": "simulation_parameter",
        "fast_timescale": "simulation_parameter",
        "slow_decay": "simulation_parameter",
        "patch_width": "simulation_parameter",
        "gaussian_sigma_y": "simulation_parameter",
        "triangle_ell": "simulation_parameter",
    },
    "source": {
        "rc": "simulation_parameter",
        "patch_size": "simulation_parameter",
        "t_off": "simulation_parameter",
    },
    "source_itr": {
        "rc_base": "simulation_parameter",
        "rc_amp": "simulation_parameter",
        "rc_sigma": "simulation_parameter",
        "rc_y0": "simulation_parameter",
        "rc_severity": "compound",
    },
    "source_itr_sin": {
        "rc_base": "simulation_parameter",
        "rc_A": "simulation_parameter",
        "rc_severity": "compound",
    },
    "interfaces": {
        "rc": "simulation_parameter",
        "interface_x": "simulation_parameter",
        "family_transfer": "compound",
    },
}


class TestOODAxisContract:
    @pytest.mark.parametrize("benchmark", sorted(OOD_AXIS_CONTRACTS))
    def test_declared_axes_match_contract(self, benchmark):
        spec = get_problem(benchmark, "temporal_encoder")
        axes = spec.ood_axes()
        expected = OOD_AXIS_CONTRACTS[benchmark]
        assert set(axes) == set(expected)
        for name, kind in expected.items():
            assert axes[name].kind == kind

    @pytest.mark.parametrize("benchmark", sorted(OOD_AXIS_CONTRACTS))
    def test_axis_invariants(self, benchmark):
        spec = get_problem(benchmark, "temporal_encoder")
        for name, axis in spec.ood_axes().items():
            assert axis.name == name
            # `compound` flag and `kind` must agree.
            assert axis.compound == (axis.kind == "compound")
            # Sweep leads with the in-distribution reference(s).
            assert axis.sweep_values()[: len(axis.id_reference)] == tuple(axis.id_reference)
            assert len(axis.ood_values) >= 1
            if axis.kind == "evaluation_parameter":
                # The time axis carries no sim-param field.
                assert axis.field is None
            assert 0.0 < axis.time_norm_horizon <= axis.dataset_t_final


# ===================== spatial-descriptor conditioning ablation =====================

_ABLATION_KEYS = [
    ("forcing", "temporal_encoder"),
    ("forcing_itr", "temporal_encoder"),
    ("forcing_itr_sin", "temporal_encoder"),
    ("source", "temporal_encoder"),
    ("source_itr", "temporal_encoder"),
    ("source_itr_sin", "temporal_encoder"),
]

_ALL_MODES = ("full", "spatial_field_only")


def _items_across_modes(spec, ds, sid, s, j, modes=_ALL_MODES):
    """Build the same (sid, s, j) item under each mode on identical inputs.

    Toggles the mode on a single spec so the only thing that varies between
    returned items is the ablation mask; resets to ``full`` afterwards.
    """
    out = {}
    for mode in modes:
        spec.set_spatial_conditioning(mode)
        out[mode] = spec.build_item(ds, sid, s, j)
    spec.set_spatial_conditioning("full")
    return out


def _assert_slice_masked(full_cond, ablated_cond, sl):
    """Value-level mask check that reads the slice from the spec (never a
    re-hardcoded index): zeroed entries are exactly 0, every other entry is
    byte-identical to the full baseline. A ``None`` slice must be a no-op."""
    n = full_cond.shape[0]
    if sl is None:
        np.testing.assert_array_equal(ablated_cond, full_cond)
        return
    zeroed = np.zeros(n, dtype=bool)
    zeroed[sl] = True
    assert np.all(ablated_cond[zeroed] == 0.0)
    np.testing.assert_array_equal(ablated_cond[~zeroed], full_cond[~zeroed])


class TestSpatialConditioningMask:
    """Value-level ablation contract: masking zeros exactly the declared
    descriptor slice, leaves the spatial field and all other cond entries
    byte-identical, and never changes cond_static_dim. The slice is read from
    ``spec.spatial_descriptor_cond_slice`` so a wrong slice cannot pass by
    matching a duplicated literal at the test site."""

    def test_default_mode_is_full(self):
        for name, representation in _ABLATION_KEYS:
            assert get_problem(name, representation).spatial_conditioning == "full"

    @pytest.mark.parametrize("key", _ABLATION_KEYS)
    def test_spatial_field_and_target_unchanged_across_modes(
        self, key, synthetic_trajectories, synthetic_sim_params
    ):
        name, representation = key
        spec, ds = _dataset_for(
            name, representation, synthetic_trajectories, synthetic_sim_params
        )
        sid, s, j = ds._pairs[0]
        items = _items_across_modes(spec, ds, sid, s, j)
        base = items["full"]
        for mode in _ALL_MODES:
            for k in ("spatial", "Y", "T_stats", "forcing_seq"):
                np.testing.assert_array_equal(
                    items[mode][k], base[k],
                    err_msg=f"{name}/{representation}: {k} changed under {mode}",
                )

    @pytest.mark.parametrize("key", _ABLATION_KEYS)
    def test_cond_dim_unchanged_across_modes(
        self, key, synthetic_trajectories, synthetic_sim_params
    ):
        name, representation = key
        spec, ds = _dataset_for(
            name, representation, synthetic_trajectories, synthetic_sim_params
        )
        sid, s, j = ds._pairs[0]
        items = _items_across_modes(spec, ds, sid, s, j)
        dim = items["full"]["cond_static"].shape
        assert dim == (spec.dims.cond_static_dim,)
        for mode in _ALL_MODES:
            assert items[mode]["cond_static"].shape == dim

    @pytest.mark.parametrize("key", _ABLATION_KEYS)
    def test_spatial_field_only_zeros_full_descriptor(
        self, key, synthetic_trajectories, synthetic_sim_params
    ):
        name, representation = key
        spec, ds = _dataset_for(
            name, representation, synthetic_trajectories, synthetic_sim_params
        )
        sid, s, j = ds._pairs[0]
        items = _items_across_modes(spec, ds, sid, s, j)
        full_cond = items["full"]["cond_static"]
        sl = spec.spatial_descriptor_cond_slice
        # Guard against a vacuous test: the descriptor must actually carry
        # information in the full baseline, otherwise zeroing proves nothing.
        assert sl is not None
        assert np.any(full_cond[sl] != 0.0)
        _assert_slice_masked(
            full_cond, items["spatial_field_only"]["cond_static"], sl
        )

    @pytest.mark.parametrize("key", _ABLATION_KEYS)
    def test_full_mode_is_identity(
        self, key, synthetic_trajectories, synthetic_sim_params
    ):
        # The default/`full` path must be byte-identical to the unmasked cond a
        # fresh spec builds (i.e. today's behavior with no ablation configured).
        name, representation = key
        spec, ds = _dataset_for(
            name, representation, synthetic_trajectories, synthetic_sim_params
        )
        sid, s, j = ds._pairs[0]
        spec.set_spatial_conditioning("full")
        masked = spec.build_item(ds, sid, s, j)["cond_static"]
        # `full` returns the input unchanged; build against an independent fresh
        # spec (default mode) and require exact equality.
        fresh, ds2 = _dataset_for(
            name, representation, synthetic_trajectories, synthetic_sim_params
        )
        ref = fresh.build_item(ds2, sid, s, j)["cond_static"]
        np.testing.assert_array_equal(masked, ref)

    def test_itr_prefix_blocks_are_preserved(
        self, synthetic_trajectories, synthetic_sim_params
    ):
        # source_itr keeps t_bar[0] + 4 void params[1:5] ahead of the masked
        # descriptor. Explicit entry-order guard against an off-by-one that
        # would clobber R_base/R_amp/y0/sigma.
        spec, ds = _dataset_for(
            "source_itr", "temporal_encoder", synthetic_trajectories,
            synthetic_sim_params,
        )
        sid, s, j = ds._pairs[0]
        items = _items_across_modes(spec, ds, sid, s, j)
        prefix = slice(0, 5)  # [t_bar, void x4]; sits before the descriptor
        for mode in _ALL_MODES:
            np.testing.assert_array_equal(
                items[mode]["cond_static"][prefix],
                items["full"]["cond_static"][prefix],
                err_msg=f"source_itr: void prefix changed under {mode}",
            )
        # The prefix must lie strictly before the masked descriptor slice.
        assert spec.spatial_descriptor_cond_slice.start >= 5

    def test_interfaces_is_noop_control_across_modes(self, interfaces_dataset):
        # interfaces declares no descriptor slice -> every mode reproduces
        # `full` exactly (the empty-mask regression control).
        ds = interfaces_dataset
        spec = get_problem("interfaces")
        assert spec.spatial_descriptor_cond_slice is None
        sid, s, j = ds._pairs[0]
        items = _items_across_modes(spec, ds, sid, s, j)
        for mode in _ALL_MODES:
            for k in items[mode]:
                np.testing.assert_array_equal(items[mode][k], items["full"][k])

    def test_invalid_mode_raises_immediately(self):
        spec = get_problem("forcing")
        with pytest.raises(ValueError, match="Unknown spatial_conditioning"):
            spec.set_spatial_conditioning("bogus")
        # A rejected mode must not have mutated the spec.
        assert spec.spatial_conditioning == "full"

    def test_noop_modes_warn_at_resolution(self):
        # A mode that masks nothing for the benchmark must surface a warning so a
        # duplicate run is never mistaken for an independent sweep condition.
        for name in ("interfaces",):
            with pytest.warns(UserWarning, match="spatial_field_only has no effect"):
                problem_from_config(
                    {"benchmark": {"name": name,
                                   "spatial_conditioning": "spatial_field_only"}}
                )

    def test_effective_modes_do_not_warn(self, recwarn):
        # A mode that actually masks must not emit the no-op warning.
        for name in ("forcing", "forcing_itr", "forcing_itr_sin", "source"):
            problem_from_config(
                {"benchmark": {"name": name,
                               "spatial_conditioning": "spatial_field_only"}}
            )
        assert not any("has no effect" in str(w.message) for w in recwarn.list)

    def test_mask_helper_returns_copy_not_alias(self):
        # Ablated modes must never alias the caller's array (guards against
        # mutating a reused dict or shared-memory tensor in place).
        spec = get_problem("source")
        spec.set_spatial_conditioning("spatial_field_only")
        cond = np.ones(spec.dims.cond_static_dim, dtype=np.float32)
        out = spec._apply_spatial_conditioning_mask(cond)
        assert out is not cond
        # Source untouched; descriptor entries still 1.0 in the input.
        assert np.all(cond == 1.0)
        assert np.all(out[spec.spatial_descriptor_cond_slice] == 0.0)

    def test_spec_independence_no_shared_mode_leak(
        self, synthetic_trajectories, synthetic_sim_params
    ):
        # get_problem constructs a fresh spec per call, so mutating one spec's
        # mode must not leak into another (guards against singleton/class-attr
        # aliasing that would silently ablate an intended-baseline run).
        a, ds_a = _dataset_for(
            "source", "temporal_encoder", synthetic_trajectories, synthetic_sim_params
        )
        sid, s, j = ds_a._pairs[0]
        a_full = a.build_item(ds_a, sid, s, j)["cond_static"].copy()

        b = get_problem("source", "temporal_encoder")
        b.set_spatial_conditioning("spatial_field_only")
        assert b.spatial_conditioning == "spatial_field_only"
        assert a.spatial_conditioning == "full"

        a_again = a.build_item(ds_a, sid, s, j)["cond_static"]
        np.testing.assert_array_equal(a_again, a_full)
        # And the descriptor block is genuinely non-zero for the untouched spec.
        assert np.any(a_again[a.spatial_descriptor_cond_slice] != 0.0)


class TestSpatialConditioningApprovalGates:
    """The three approval gates: (1) the ablation leaves the initialized model
    bit-identical (only the dataset conditioning values differ); (2) the mode
    round-trips through the persisted config and is restored on the eval path,
    producing masked cond at eval time; (3) the mode is recorded in the
    resolved config provenance so seed-matched runs are auditable."""

    def _fno(self, dims):
        from src.operators.fno2d import FNO2d

        return FNO2d(
            modes1=2, modes2=2, width=8,
            in_channels=dims.in_channels,
            out_channels=1, n_layers=2,
            cond_static_dim=dims.cond_static_dim,
            cond_hidden=16,
            temporal_token_dim=dims.temporal_token_dim,
            temporal_hidden=16, forcing_embed_dim=16, forcing_spatial_dim=4,
            use_forcing_time_aug=dims.use_forcing_time_aug,
            s_y_channel=dims.s_y_channel,
        )

    def test_gate1_initial_state_dict_parity(self):
        # Same seed reset immediately before each construction: the ablated run
        # must start from a bit-identical model, since the mode never touches
        # cond_static_dim (or any other model-facing dim).
        import torch

        full = problem_from_config(
            {"benchmark": {"name": "source", "spatial_conditioning": "full"}}
        )
        abl = problem_from_config(
            {"benchmark": {"name": "source",
                           "spatial_conditioning": "spatial_field_only"}}
        )
        assert full.dims == abl.dims  # ablation is dim-invariant by construction

        torch.manual_seed(1234)
        m_full = self._fno(full.dims)
        torch.manual_seed(1234)
        m_abl = self._fno(abl.dims)

        sd_full, sd_abl = m_full.state_dict(), m_abl.state_dict()
        assert sd_full.keys() == sd_abl.keys()
        for k in sd_full:
            assert torch.equal(sd_full[k], sd_abl[k]), f"init differs at {k}"

    def test_gate2_checkpoint_to_eval_mode_restoration(
        self, tmp_path, synthetic_trajectories
    ):
        # Persist a resolved config carrying the ablation mode (as config_used.yaml
        # does), then reconstruct the spec exactly as the eval path does
        # (problem_from_config on the reloaded config) and require the restored
        # spec to (a) report the same mode and (b) produce masked cond at eval.
        import yaml

        cfg = {"benchmark": {"name": "source_itr",
                             "spatial_conditioning": "spatial_field_only"}}
        cfg_path = tmp_path / "config_used.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg))

        reloaded = yaml.safe_load(cfg_path.read_text())
        spec = problem_from_config(reloaded)
        assert spec.spatial_conditioning == "spatial_field_only"

        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        sim_params = _adapter_sim_params(spec, trajectories, x_grid, y_grid, t_grid)
        ds = _make_dataset(trajectories, x_grid, y_grid, t_grid, sim_params, spec)
        sid, s, j = ds._pairs[0]
        cond = ds[0]["cond_static"].numpy()  # eval-path item build (via __getitem__)
        sl = spec.spatial_descriptor_cond_slice
        assert np.all(cond[sl] == 0.0)

        # A `full` checkpoint reloads unmasked -> a training-masked / eval-full
        # mismatch would surface as a different cond, not pass silently.
        full_spec = problem_from_config(
            {"benchmark": {"name": "source_itr", "spatial_conditioning": "full"}}
        )
        ds_full = _make_dataset(
            trajectories, x_grid, y_grid, t_grid, sim_params, full_spec
        )
        assert np.any(ds_full[0]["cond_static"].numpy()[sl] != 0.0)

    def test_gate3_config_provenance_round_trip(self, tmp_path):
        # The mode must survive a YAML dump/load of config_used.yaml and re-resolve
        # to the same spec mode for every benchmark (auditable seed-matched runs).
        import yaml

        for name in (
            "forcing", "forcing_itr", "forcing_itr_sin",
            "source", "source_itr", "source_itr_sin", "interfaces",
        ):
            for mode in _ALL_MODES:
                cfg = {"benchmark": {"name": name, "spatial_conditioning": mode}}
                path = tmp_path / f"config_used_{name}_{mode}.yaml"
                path.write_text(yaml.safe_dump(cfg))
                loaded = yaml.safe_load(path.read_text())
                assert loaded["benchmark"]["spatial_conditioning"] == mode
                spec = problem_from_config(loaded)
                assert spec.spatial_conditioning == mode
