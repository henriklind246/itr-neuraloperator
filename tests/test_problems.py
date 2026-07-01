import numpy as np
import pytest

from data.dataset import SnapshotPairDataset, problem_from_config
from problems.registry import get_problem
from problems.diffusion import DiffusionProblem
from problems.forcing import FORCING_TEMPORAL_SAMPLES, ForcingProblem
from problems.interfaces import InterfacesProblem
from problems.source import SourceProblem
from problems.source_itr import SourceItrProblem
from src.physics.fv_solver_2d import FVSolver2D, Layer2D

_SYNTH_MU = 0.0
_SYNTH_SIGMA = 1.0

# Target (benchmark x representation) tensor contract. Single source of truth for
# the dims/shape/absence tests below; mirrors the plan's contract table.
CONTRACTS = {
    ("forcing", "temporal_encoder"): dict(
        in_ch=4, cond=11, has_fseq=True, token=2, t_stats=2,
        encoder=True, s_y=3, aug=True,
    ),
    ("forcing", "bins"): dict(
        in_ch=20, cond=11, has_fseq=False, token=2, t_stats=2,
        encoder=False, s_y=3, aug=False,
    ),
    ("source", "temporal_encoder"): dict(
        in_ch=4, cond=7, has_fseq=True, token=2, t_stats=3,
        encoder=True, s_y=3, aug=False,
    ),
    ("source", "bins"): dict(
        in_ch=20, cond=7, has_fseq=False, token=2, t_stats=3,
        encoder=False, s_y=3, aug=False,
    ),
    ("source_itr", "temporal_encoder"): dict(
        in_ch=5, cond=10, has_fseq=True, token=2, t_stats=3,
        encoder=True, s_y=3, aug=False,
    ),
    ("source_itr", "bins"): dict(
        in_ch=21, cond=10, has_fseq=False, token=2, t_stats=3,
        encoder=False, s_y=3, aug=False,
    ),
    ("interfaces", "temporal_encoder"): dict(
        in_ch=6, cond=4, has_fseq=True, token=2, t_stats=3,
        encoder=True, s_y=5, aug=True,
    ),
    ("interfaces", "bins"): dict(
        in_ch=22, cond=4, has_fseq=False, token=2, t_stats=3,
        encoder=False, s_y=5, aug=False,
    ),
    # diffusion supports only temporal_encoder (bins raises in __init__).
    ("diffusion", "temporal_encoder"): dict(
        in_ch=4, cond=2, has_fseq=True, token=2, t_stats=2,
        encoder=True, s_y=3, aug=True,
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


# ===================== registry / dims =====================

class TestRegistry:
    def test_forcing_registered(self):
        assert isinstance(get_problem("forcing"), ForcingProblem)

    def test_interfaces_registered(self):
        assert isinstance(get_problem("interfaces"), InterfacesProblem)

    def test_source_registered(self):
        assert isinstance(get_problem("source"), SourceProblem)

    def test_source_itr_registered(self):
        assert isinstance(get_problem("source_itr"), SourceItrProblem)

    def test_diffusion_registered(self):
        assert isinstance(get_problem("diffusion"), DiffusionProblem)

    def test_diffusion_rejects_bins(self):
        # diffusion supports only temporal_encoder; bins must raise in __init__.
        with pytest.raises(ValueError, match="temporal_encoder"):
            get_problem("diffusion", "bins")

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
        assert dims.has_forcing_seq is c["has_fseq"]
        assert dims.temporal_token_dim == c["token"]
        assert dims.t_stats_dim == c["t_stats"]
        assert dims.use_temporal_encoder is c["encoder"]
        assert dims.s_y_channel == c["s_y"]
        assert dims.use_forcing_time_aug is c["aug"]


# ===================== forcing item shape contract =====================

class TestForcingItem:
    """forcing/temporal_encoder item shapes: lean 4-channel spatial, cond 11,
    a 128x2 forcing_seq, T_stats 2."""

    def test_item_keys_shapes(self, forcing_dataset):
        ds = forcing_dataset
        spec = get_problem("forcing")
        sim_id, s, j = ds._pairs[0]
        item = spec.build_item(ds, sim_id, s, j)
        assert set(item) == {"spatial", "cond_static", "forcing_seq", "Y", "T_stats"}
        assert item["spatial"].shape == (ds.Nx, ds.Ny, 4)
        assert item["cond_static"].shape == (11,)
        assert item["forcing_seq"].shape == (FORCING_TEMPORAL_SAMPLES, 2)
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
        assert item["cond_static"].shape == (4,)
        assert item["forcing_seq"].shape == (FORCING_TEMPORAL_SAMPLES, 2)
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
        # source/temporal_encoder: lean 4-channel spatial, cond 7, pulse 128x2.
        assert set(item) == {"spatial", "cond_static", "forcing_seq", "Y", "T_stats"}
        assert item["spatial"].shape == (ds.Nx, ds.Ny, 4)
        assert item["cond_static"].shape == (7,)
        assert item["forcing_seq"].shape == (FORCING_TEMPORAL_SAMPLES, 2)
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
        assert item["cond_static"].shape == (10,)
        assert item["forcing_seq"].shape == (FORCING_TEMPORAL_SAMPLES, 2)
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

    def test_reads_representation_axis(self):
        spec = problem_from_config(
            {"benchmark": {"name": "forcing", "representation": "bins"}}
        )
        assert spec.representation == "bins"
        assert spec.dims.in_channels == 20
        assert spec.dims.use_temporal_encoder is False

    def test_defaults_to_temporal_encoder_when_representation_missing(self):
        spec = problem_from_config({"benchmark": {"name": "source"}})
        assert spec.representation == "temporal_encoder"

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
    """Build a dataset for any (benchmark, representation) at the synthetic grid.

    The representation does not change sim_params (both modes derive from the
    same trajectories + sim_params), so the per-benchmark sim_params are built
    once and reused across representations.
    """
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

        if c["has_fseq"]:
            assert item["forcing_seq"].shape == (FORCING_TEMPORAL_SAMPLES, c["token"])
        else:
            # bins mode emits an explicit empty forcing_seq (fixed item shape).
            assert item["forcing_seq"].shape[-2] == 0

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
    @pytest.mark.parametrize(
        "name", ["forcing", "source", "interfaces"]
    )
    def test_temporal_has_no_integral_bins(self, name, synthetic_trajectories, synthetic_sim_params):
        """temporal_encoder spatial channels carry no Q-integral bins; bins mode
        adds exactly the extra bin channels on top of the lean stack."""
        spec_t, ds_t = _dataset_for(
            name, "temporal_encoder", synthetic_trajectories, synthetic_sim_params
        )
        spec_b, ds_b = _dataset_for(
            name, "bins", synthetic_trajectories, synthetic_sim_params
        )
        lean = CONTRACTS[(name, "temporal_encoder")]["in_ch"]
        bins = CONTRACTS[(name, "bins")]["in_ch"]
        # 16 integral bins are present only in bins mode.
        assert bins - lean == 16

    def test_bins_forcing_seq_is_empty(self, synthetic_trajectories, synthetic_sim_params):
        spec, ds = _dataset_for(
            "forcing", "bins", synthetic_trajectories, synthetic_sim_params
        )
        sim_id, s, j = ds._pairs[0]
        item = spec.build_item(ds, sim_id, s, j)
        assert item["forcing_seq"].size == 0

    def test_source_cond_has_no_A_norm(self, synthetic_trajectories, synthetic_sim_params):
        # source cond is base 3 + (x_h, y_h, w_h, h_h) = 7; A_norm is dropped.
        for representation in ("temporal_encoder", "bins"):
            spec, ds = _dataset_for(
                "source", representation, synthetic_trajectories, synthetic_sim_params
            )
            sim_id, s, j = ds._pairs[0]
            item = spec.build_item(ds, sim_id, s, j)
            assert item["cond_static"].shape == (7,)

    def test_interfaces_temporal_reads_s_y_channel_5(self):
        # The learned spatial forcing must read s_y at channel 5, not K_norm at 3.
        assert get_problem("interfaces", "temporal_encoder").dims.s_y_channel == 5
        assert get_problem("forcing", "temporal_encoder").dims.s_y_channel == 3
        assert get_problem("source", "temporal_encoder").dims.s_y_channel == 3


class TestRepresentationReconstruction:
    """Both representations are derivable from the same trajectories + sim_params;
    building one then the other from the same (sid, s, j) must succeed with the
    contracted shapes (catches benchmarks whose stored metadata is insufficient
    for the new representation)."""

    @pytest.mark.parametrize("name", ["forcing", "source", "interfaces"])
    def test_both_representations_from_same_pair(self, name, synthetic_trajectories, synthetic_sim_params):
        spec_t, ds_t = _dataset_for(
            name, "temporal_encoder", synthetic_trajectories, synthetic_sim_params
        )
        spec_b, ds_b = _dataset_for(
            name, "bins", synthetic_trajectories, synthetic_sim_params
        )
        sim_id, s, j = ds_t._pairs[0]
        item_t = spec_t.build_item(ds_t, sim_id, s, j)
        item_b = spec_b.build_item(ds_b, sim_id, s, j)

        ct = CONTRACTS[(name, "temporal_encoder")]
        cb = CONTRACTS[(name, "bins")]
        assert item_t["spatial"].shape[-1] == ct["in_ch"]
        assert item_b["spatial"].shape[-1] == cb["in_ch"]
        # Y target is representation-invariant.
        np.testing.assert_allclose(
            item_t["Y"], item_b["Y"], rtol=0, atol=0
        )


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
    "interfaces": {
        "rc": "simulation_parameter",
        "interface_x": "simulation_parameter",
        "family_transfer": "compound",
    },
    "diffusion": {},
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
