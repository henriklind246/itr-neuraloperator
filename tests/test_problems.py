import numpy as np
import pytest

from data.dataset import SnapshotPairDataset, problem_from_config
from problems.registry import get_problem
from problems.forcing import ForcingProblem
from problems.interfaces import InterfacesProblem
from problems.source import SourceProblem
from src.physics.fv_solver_2d import FVSolver2D, Layer2D

_SYNTH_MU = 0.0
_SYNTH_SIGMA = 1.0


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


# ===================== registry / dims =====================

class TestRegistry:
    def test_forcing_registered(self):
        assert isinstance(get_problem("forcing"), ForcingProblem)

    def test_interfaces_registered(self):
        assert isinstance(get_problem("interfaces"), InterfacesProblem)

    def test_source_registered(self):
        assert isinstance(get_problem("source"), SourceProblem)

    def test_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown benchmark"):
            get_problem("does_not_exist")

    def test_forcing_dims(self):
        dims = get_problem("forcing").dims
        assert dims.in_channels == 20
        assert dims.cond_static_dim == 23
        assert dims.has_forcing_seq is True
        assert dims.temporal_token_dim == 5
        assert dims.t_stats_dim == 2
        assert dims.use_temporal_encoder is True

    @pytest.mark.parametrize(
        "name,in_ch,cond,has_fseq,t_stats,encoder",
        [
            ("forcing", 20, 23, True, 2, True),
            ("interfaces", 6, 4, True, 3, True),
            ("source", 20, 8, False, 3, False),
        ],
    )
    def test_dims_per_benchmark(self, name, in_ch, cond, has_fseq, t_stats, encoder):
        dims = get_problem(name).dims
        assert dims.in_channels == in_ch
        assert dims.cond_static_dim == cond
        assert dims.has_forcing_seq is has_fseq
        assert dims.t_stats_dim == t_stats
        assert dims.use_temporal_encoder is encoder


# ===================== item parity (independent oracle) =====================

class TestForcingItemParity:
    """The adapter must reproduce dataset.__getitem__ bit-for-bit.

    The dataset and the adapter implement the 20/23 representation
    independently, so allclose agreement proves the abstraction did not change
    the numerics.
    """

    def test_build_item_matches_getitem(self, forcing_dataset):
        spec = get_problem("forcing")
        # Independent caches so neither path warms the other's lazily.
        ds_a = forcing_dataset
        for idx in range(0, len(ds_a), max(1, len(ds_a) // 12)):
            ref = ds_a[idx]
            sim_id, s, j = ds_a._pairs[idx]
            item = spec.build_item(ds_a, sim_id, s, j)

            assert item["spatial"].shape == (ds_a.Nx, ds_a.Ny, 20)
            assert item["cond_static"].shape == (23,)
            assert item["forcing_seq"].shape == (ds_a.temporal_samples, 5)
            assert item["Y"].shape == (ds_a.Nx, ds_a.Ny, 1)
            assert item["T_stats"].shape == (2,)

            np.testing.assert_allclose(item["spatial"], ref["spatial"].numpy(), rtol=0, atol=0)
            np.testing.assert_allclose(item["cond_static"], ref["cond_static"].numpy(), rtol=0, atol=0)
            np.testing.assert_allclose(item["forcing_seq"], ref["forcing_seq"].numpy(), rtol=0, atol=0)
            np.testing.assert_allclose(item["Y"], ref["Y"].numpy(), rtol=0, atol=0)
            np.testing.assert_allclose(item["T_stats"], ref["T_stats"].numpy(), rtol=0, atol=0)

    def test_item_keys(self, forcing_dataset):
        spec = get_problem("forcing")
        sim_id, s, j = forcing_dataset._pairs[0]
        item = spec.build_item(forcing_dataset, sim_id, s, j)
        assert set(item) == {"spatial", "cond_static", "forcing_seq", "Y", "T_stats"}


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
            assert g["ic_family"] == r["ic_family"]
            np.testing.assert_allclose(g["T0"], r["T0"], rtol=0, atol=0)


# ===================== interfaces adapter =====================

class TestInterfacesItem:
    def test_item_keys_shapes(self, interfaces_dataset):
        ds = interfaces_dataset
        spec = get_problem("interfaces")
        sim_id, s, j = ds._pairs[0]
        item = spec.build_item(ds, sim_id, s, j)
        assert set(item) == {"spatial", "cond_static", "forcing_seq", "Y", "T_stats"}
        assert item["spatial"].shape == (ds.Nx, ds.Ny, 6)
        assert item["cond_static"].shape == (4,)
        assert item["forcing_seq"].shape == (ds.temporal_samples, 5)
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

    def test_rejects_non_sin(self):
        spec = get_problem("interfaces")
        bad = np.array([{
            "R_c": 0.5, "interface_x": 0.5,
            "temporal_family": "exp", "temporal_params": {},
            "spatial_family": "uniform", "spatial_params": {},
        }], dtype=object)
        with pytest.raises(ValueError, match="sin-only"):
            spec.validate_schema(bad, np.array([0]))

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


# ===================== source adapter =====================

class TestSourceItem:
    def test_item_keys_shapes(self, source_dataset):
        ds = source_dataset
        spec = get_problem("source")
        sim_id, s, j = ds._pairs[0]
        item = spec.build_item(ds, sim_id, s, j)
        # No forcing_seq for the encoder-off source benchmark.
        assert set(item) == {"spatial", "cond_static", "Y", "T_stats"}
        assert item["spatial"].shape == (ds.Nx, ds.Ny, 20)
        assert item["cond_static"].shape == (8,)
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


class TestProblemFromConfig:
    @pytest.mark.parametrize(
        "name,cls",
        [("forcing", ForcingProblem), ("interfaces", InterfacesProblem), ("source", SourceProblem)],
    )
    def test_resolves_active_benchmark(self, name, cls):
        spec = problem_from_config({"benchmark": {"name": name}})
        assert isinstance(spec, cls)
        assert spec.name == name

    def test_defaults_to_forcing_when_missing(self):
        assert isinstance(problem_from_config({}), ForcingProblem)
        assert isinstance(problem_from_config({"benchmark": {}}), ForcingProblem)
