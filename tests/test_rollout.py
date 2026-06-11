import numpy as np
import pytest
import torch

from data.dataset import SnapshotPairDataset
from problems import forcing as forcing_problem
from problems import interfaces as interfaces_problem
from problems import source as source_problem
from problems.registry import get_problem
from src.operators.rollout import (
    RolloutOptions,
    build_homogeneous_rollout_times,
    build_rollout_item_from_base,
    predict_autoregressive,
    rollout_is_active,
    rollout_options_from_config,
)


def _make_dataset(trajectories, x_grid, y_grid, t_grid, sim_params, spec):
    return SnapshotPairDataset(
        trajectories=trajectories,
        t_grid=t_grid,
        x_grid=x_grid,
        y_grid=y_grid,
        sim_ids=np.arange(trajectories.shape[0]),
        sim_params=sim_params,
        mu_global=0.0,
        sigma_global=1.0,
        n_snapshots=6,
        noise_std=0.0,
        problem=spec,
    )


def _adapter_sim_params(spec, trajectories, x_grid, y_grid, t_grid):
    X, Y = np.meshgrid(x_grid, y_grid, indexing="ij")
    return np.array(
        spec.sample_sim_params(
            rng=np.random.default_rng(0),
            rng_profile=np.random.default_rng(1),
            grids={"X": X, "Y": Y, "x_grid": x_grid, "y_grid": y_grid},
            time_cfg={
                "num_sims": trajectories.shape[0],
                "dt": float(t_grid[1] - t_grid[0]),
                "t_final": float(t_grid[-1]),
                "lhs_seed": 0,
                "t_on": 0.0,
                "t_off": 0.2,
                "phase": 0.0,
                "tukey_alpha": 0.5,
                "T_right": 300.0,
                "b": 1.0,
            },
        ),
        dtype=object,
    )


def _direct_model_pred(model, item):
    spatial = torch.from_numpy(item["spatial"]).unsqueeze(0)
    cond = torch.from_numpy(item["cond_static"]).unsqueeze(0)
    forcing_seq = item.get("forcing_seq")
    if bool(getattr(model, "use_temporal_encoder", True)) and forcing_seq is not None:
        return model(spatial, cond, torch.from_numpy(forcing_seq).unsqueeze(0))
    return model(spatial, cond)


class TestRolloutOptions:
    def test_missing_config_is_default_off(self):
        opts = rollout_options_from_config({})
        assert opts == RolloutOptions(enabled=False, num_substeps=1, partition="homogeneous")
        assert not rollout_is_active(opts)

    def test_num_substeps_one_is_inactive_even_when_enabled(self):
        assert not rollout_is_active(RolloutOptions(enabled=True, num_substeps=1))

    def test_num_substeps_override_opts_in(self):
        opts = rollout_options_from_config({}, num_substeps=4)
        assert opts.enabled
        assert opts.num_substeps == 4
        assert rollout_is_active(opts)


class TestHomogeneousPartition:
    def test_single_substep_is_direct_interval(self):
        assert build_homogeneous_rollout_times(0.2, 0.8, 1) == [(0.2, 0.8)]

    def test_substeps_are_equal_duration(self):
        intervals = build_homogeneous_rollout_times(0.0, 1.0, 4)
        widths = [hi - lo for lo, hi in intervals]
        np.testing.assert_allclose(widths, [0.25, 0.25, 0.25, 0.25])
        assert intervals[-1][1] == pytest.approx(1.0)


class TestRolloutItemRecompute:
    def test_forcing_temporal_recomputes_interval_tokens_and_preserves_stats(
        self, synthetic_trajectories, synthetic_sim_params
    ):
        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        spec = get_problem("forcing", "temporal_encoder")
        ds = _make_dataset(trajectories, x_grid, y_grid, t_grid, synthetic_sim_params, spec)
        sid, s, k, j = 0, 0, 10, 20
        base = spec.build_item(ds, sid, s, j)
        direct_sub = spec.build_item(ds, sid, s, k)
        current = base["spatial"][..., 0] + np.float32(2.0)

        rollout = build_rollout_item_from_base(
            base, ds, spec, sid, current, float(t_grid[s]), float(t_grid[k])
        )

        np.testing.assert_allclose(rollout["spatial"][..., 0], current)
        np.testing.assert_allclose(rollout["spatial"][..., 1:], base["spatial"][..., 1:])
        np.testing.assert_allclose(rollout["cond_static"], direct_sub["cond_static"])
        np.testing.assert_allclose(rollout["forcing_seq"], direct_sub["forcing_seq"])
        np.testing.assert_allclose(rollout["T_stats"], base["T_stats"])

    def test_forcing_bins_recomputes_signed_bins_with_existing_convention(
        self, synthetic_trajectories, synthetic_sim_params
    ):
        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        spec = get_problem("forcing", "bins")
        ds = _make_dataset(trajectories, x_grid, y_grid, t_grid, synthetic_sim_params, spec)
        sid, s, k, j = 0, 0, 10, 20
        base = spec.build_item(ds, sid, s, j)
        direct_sub = spec.build_item(ds, sid, s, k)
        current = base["spatial"][..., 0] - np.float32(1.0)

        rollout = build_rollout_item_from_base(
            base, ds, spec, sid, current, float(t_grid[s]), float(t_grid[k])
        )

        np.testing.assert_allclose(rollout["spatial"][..., 0], current)
        np.testing.assert_allclose(
            rollout["spatial"][..., forcing_problem.SPATIAL_CHANNELS_TEMPORAL:],
            direct_sub["spatial"][..., forcing_problem.SPATIAL_CHANNELS_TEMPORAL:],
        )
        np.testing.assert_allclose(rollout["forcing_seq"], np.zeros((0, 0), dtype=np.float32))
        np.testing.assert_allclose(rollout["T_stats"], base["T_stats"])

    def test_interfaces_preserves_material_distance_sy_and_interface_stats(
        self, synthetic_trajectories
    ):
        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        spec = get_problem("interfaces", "temporal_encoder")
        sim_params = _adapter_sim_params(spec, trajectories, x_grid, y_grid, t_grid)
        ds = _make_dataset(trajectories, x_grid, y_grid, t_grid, sim_params, spec)
        sid, s, k, j = 0, 0, 10, 20
        base = spec.build_item(ds, sid, s, j)
        direct_sub = spec.build_item(ds, sid, s, k)
        current = base["spatial"][..., 0] + np.float32(0.5)

        rollout = build_rollout_item_from_base(
            base, ds, spec, sid, current, float(t_grid[s]), float(t_grid[k])
        )

        assert spec.dims.s_y_channel == interfaces_problem.S_Y_CHANNEL
        np.testing.assert_allclose(rollout["spatial"][..., 3:6], base["spatial"][..., 3:6])
        np.testing.assert_allclose(rollout["cond_static"], direct_sub["cond_static"])
        np.testing.assert_allclose(rollout["forcing_seq"], direct_sub["forcing_seq"])
        np.testing.assert_allclose(rollout["T_stats"], base["T_stats"])

    def test_source_bins_recomputes_source_bins_without_amplitude_cond_leak(
        self, synthetic_trajectories, synthetic_source_sim_params
    ):
        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        spec = get_problem("source", "bins")
        ds = _make_dataset(trajectories, x_grid, y_grid, t_grid, synthetic_source_sim_params, spec)
        sid, s, k, j = 0, 0, 10, 20
        base = spec.build_item(ds, sid, s, j)
        direct_sub = spec.build_item(ds, sid, s, k)
        current = base["spatial"][..., 0] + np.float32(0.25)

        rollout = build_rollout_item_from_base(
            base, ds, spec, sid, current, float(t_grid[s]), float(t_grid[k])
        )

        assert rollout["cond_static"].shape == (source_problem.COND_STATIC_DIM,)
        np.testing.assert_allclose(rollout["cond_static"], direct_sub["cond_static"])
        np.testing.assert_allclose(
            rollout["spatial"][..., source_problem.SPATIAL_CHANNELS_TEMPORAL:],
            direct_sub["spatial"][..., source_problem.SPATIAL_CHANNELS_TEMPORAL:],
        )
        np.testing.assert_allclose(rollout["T_stats"], base["T_stats"])


class TestPredictAutoregressive:
    def test_single_substep_matches_direct_model_prediction(
        self, synthetic_trajectories, synthetic_sim_params, small_fno2d
    ):
        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        spec = get_problem("forcing", "temporal_encoder")
        ds = _make_dataset(trajectories, x_grid, y_grid, t_grid, synthetic_sim_params, spec)
        sid, s, j = 0, 0, 20
        item = spec.build_item(ds, sid, s, j)
        small_fno2d.eval()

        with torch.no_grad():
            direct = _direct_model_pred(small_fno2d, item)
            rollout = predict_autoregressive(
                small_fno2d,
                ds,
                sim_id=sid,
                s=s,
                j=j,
                num_substeps=1,
                device=torch.device("cpu"),
            )

        torch.testing.assert_close(rollout, direct)
