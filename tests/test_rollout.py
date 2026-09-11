import math
import types

import numpy as np
import pytest
import torch

from data.dataset import SnapshotPairDataset
from problems import forcing as forcing_problem
from problems import interfaces as interfaces_problem
from problems import source as source_problem
from problems import source_itr_sin as source_itr_sin_problem
from problems.registry import get_problem
from src.operators.eval import evaluate
from src.operators.fno2d import FNO2d
from src.operators.rollout import (
    RolloutOptions,
    build_homogeneous_rollout_times,
    build_rollout_item_from_base,
    predict_autoregressive,
    rollout_is_active,
    rollout_options_from_config,
)

# (benchmark, representation) pairs the rollout dispatch supports.
# forcing_itr_sin has no rollout branch.
ROLLOUT_CASES = [
    ("forcing", "temporal_encoder"),
    ("interfaces", "temporal_encoder"),
    ("source", "temporal_encoder"),
    ("source_itr_sin", "temporal_encoder"),
    ("source_itr_sin", "temporal_encoder"),
]


def _make_dataset(trajectories, x_grid, y_grid, t_grid, sim_params, spec,
                  time_norm_horizon=None):
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
        time_norm_horizon=time_norm_horizon,
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


def _sim_params_for(benchmark, spec, trajectories, x_grid, y_grid, t_grid, fixtures):
    """Pick the sim_params source for a benchmark.

    forcing/source/source_itr_sin use hand-built fixtures whose schema is pinned to
    the live sampler; interfaces round-trips through the spec's own sampler
    because its params (interface_x, sampled ICs) have no shortcut.
    """
    if benchmark in fixtures:
        return fixtures[benchmark]
    return _adapter_sim_params(spec, trajectories, x_grid, y_grid, t_grid)


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

    def test_source_recomputes_cond_without_amplitude_leak(
        self, synthetic_trajectories, synthetic_source_sim_params
    ):
        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        spec = get_problem("source", "temporal_encoder")
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
        np.testing.assert_allclose(rollout["forcing_seq"], direct_sub["forcing_seq"])
        np.testing.assert_allclose(rollout["T_stats"], base["T_stats"])


class TestSingleSubstepItemIdentity:
    """K=1 rollout must reconstruct the direct item, for every dispatch branch.

    Non-trivial because the rollout path does not copy the conditioning: it
    recomputes the cond vector, the forcing_seq, and the Q-bin channels from
    sim_params. Any mis-wired argument in a branch shows up here rather than as
    a silently-worse rollout number.
    """

    @pytest.mark.parametrize("benchmark,representation", ROLLOUT_CASES)
    def test_k1_item_matches_direct_item(
        self, benchmark, representation, synthetic_trajectories,
        synthetic_sim_params, synthetic_source_sim_params,
        synthetic_source_itr_sin_sim_params,
    ):
        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        spec = get_problem(benchmark, representation)
        sim_params = _sim_params_for(
            benchmark, spec, trajectories, x_grid, y_grid, t_grid,
            {
                "forcing": synthetic_sim_params,
                "source": synthetic_source_sim_params,
                "source_itr_sin": synthetic_source_itr_sin_sim_params,
            },
        )
        ds = _make_dataset(trajectories, x_grid, y_grid, t_grid, sim_params, spec)
        sid, s, j = 0, 10, 30
        t_s, t_j = float(t_grid[s]), float(t_grid[j])

        # The endpoints are assigned, not derived, so they must be bit-exact.
        intervals = build_homogeneous_rollout_times(t_s, t_j, 1)
        assert intervals == [(t_s, t_j)]

        direct = spec.build_item(ds, sid, s, j)
        rollout = build_rollout_item_from_base(
            direct, ds, spec, sid, direct["spatial"][..., 0], t_s, t_j
        )

        assert set(rollout) == set(direct)
        # Copied verbatim by _copy_item, so exact.
        split = get_problem(benchmark, "temporal_encoder").dims.in_channels
        np.testing.assert_array_equal(
            rollout["spatial"][..., :split], direct["spatial"][..., :split]
        )
        np.testing.assert_array_equal(rollout["Y"], direct["Y"])
        np.testing.assert_array_equal(rollout["T_stats"], direct["T_stats"])
        # Recomputed through linspace / horizon division / range normalization.
        np.testing.assert_allclose(
            rollout["cond_static"], direct["cond_static"], rtol=1e-6
        )
        np.testing.assert_allclose(
            rollout["forcing_seq"], direct["forcing_seq"], rtol=1e-6
        )
        np.testing.assert_allclose(
            rollout["spatial"][..., split:], direct["spatial"][..., split:], rtol=1e-6
        )


class TestShortSubintervalIntegrity:
    """K=8 on the shortest lead: the partition and its features stay well-formed.

    Deliberately mechanical. Homogeneous subdivision at large K produces
    subinterval *leads* far shorter than any training pair, which is a plausible
    mechanism for large-K degradation and therefore a result to measure, not an
    implementation error. Asserting it here would encode a scientific claim as a
    correctness contract.
    """

    NUM_SUBSTEPS = 8

    @pytest.mark.parametrize("benchmark,representation", ROLLOUT_CASES)
    def test_shortest_lead_partition_and_features_are_well_formed(
        self, benchmark, representation, synthetic_trajectories,
        synthetic_sim_params, synthetic_source_sim_params,
        synthetic_source_itr_sin_sim_params,
    ):
        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        spec = get_problem(benchmark, representation)
        sim_params = _sim_params_for(
            benchmark, spec, trajectories, x_grid, y_grid, t_grid,
            {
                "forcing": synthetic_sim_params,
                "source": synthetic_source_sim_params,
                "source_itr_sin": synthetic_source_itr_sin_sim_params,
            },
        )
        ds = _make_dataset(trajectories, x_grid, y_grid, t_grid, sim_params, spec)
        # Shortest lead available on the n_snapshots=6 partition.
        s, j = int(ds.t_indices[0]), int(ds.t_indices[1])
        sid = 0
        t_s, t_j = float(t_grid[s]), float(t_grid[j])

        intervals = build_homogeneous_rollout_times(t_s, t_j, self.NUM_SUBSTEPS)
        assert len(intervals) == self.NUM_SUBSTEPS
        edges = [lo for lo, _ in intervals] + [intervals[-1][1]]
        assert all(math.isfinite(e) for e in edges)
        assert all(hi > lo for lo, hi in intervals)
        assert all(b > a for a, b in zip(edges, edges[1:]))
        assert edges[0] == t_s
        assert edges[-1] == t_j

        base = spec.build_item(ds, sid, s, j)
        for t_lo, t_hi in intervals:
            item = build_rollout_item_from_base(
                base, ds, spec, sid, base["spatial"][..., 0], t_lo, t_hi
            )
            assert np.all(np.isfinite(item["cond_static"]))
            assert np.all(np.isfinite(item["spatial"]))
            if spec.representation == "temporal_encoder":
                seq = item["forcing_seq"]
                assert seq.shape == (ds.temporal_samples, spec.dims.temporal_token_dim)
                assert np.all(np.isfinite(seq))
                # Position tokens must still span the subinterval, however short.
                np.testing.assert_allclose(seq[0, 0], 0.0, atol=1e-7)
                np.testing.assert_allclose(seq[-1, 0], 1.0, atol=1e-7)


class TestSourceItrRollout:
    def test_rc_y_channel_survives_and_forcing_seq_recomputes(
        self, synthetic_trajectories, synthetic_source_itr_sin_sim_params
    ):
        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        spec = get_problem("source_itr_sin", "temporal_encoder")
        ds = _make_dataset(
            trajectories, x_grid, y_grid, t_grid, synthetic_source_itr_sin_sim_params, spec
        )
        sid, s, k, j = 0, 0, 10, 20
        base = spec.build_item(ds, sid, s, j)
        direct_sub = spec.build_item(ds, sid, s, k)
        current = base["spatial"][..., 0] + np.float32(0.25)

        rollout = build_rollout_item_from_base(
            base, ds, spec, sid, current, float(t_grid[s]), float(t_grid[k])
        )

        assert rollout["cond_static"].shape == (source_itr_sin_problem.COND_STATIC_DIM,)
        assert source_itr_sin_problem.COND_STATIC_DIM == 7
        np.testing.assert_allclose(rollout["cond_static"], direct_sub["cond_static"])

        # R_c(y) is time-invariant, so _copy_item carries it through untouched.
        rc = rollout["spatial"][..., source_itr_sin_problem.RC_Y_CHANNEL]
        np.testing.assert_array_equal(
            rc, base["spatial"][..., source_itr_sin_problem.RC_Y_CHANNEL]
        )
        # Non-constant along y: a plain `source` item has no void profile here.
        assert float(rc[0].std()) > 0.0

        seq = rollout["forcing_seq"]
        np.testing.assert_allclose(seq, direct_sub["forcing_seq"])
        # Recomputed for [t_s, t_k], not copied from the [t_s, t_j] base item.
        assert not np.allclose(seq, base["forcing_seq"])
        np.testing.assert_allclose(rollout["T_stats"], base["T_stats"])


class TestRolloutTimeNormalizationHorizon:
    """Rollout subinterval lead time must normalize by time_norm_horizon.

    They previously divided by t_final. The two coincide on the standard test
    partition, so the divergence only appears when the normalization horizon
    differs from the dataset's final time.
    """

    HORIZON = 0.6

    def _dataset(self, spec, trajectories, x_grid, y_grid, t_grid, sim_params):
        ds = _make_dataset(
            trajectories, x_grid, y_grid, t_grid, sim_params, spec,
            time_norm_horizon=self.HORIZON,
        )
        assert ds.time_norm_horizon != ds.t_final
        return ds

    @pytest.mark.parametrize(
        "benchmark", ["forcing", "interfaces", "source", "source_itr_sin"]
    )
    def test_subinterval_time_features_match_direct_item_past_horizon(
        self, benchmark, synthetic_trajectories, synthetic_sim_params,
        synthetic_source_sim_params, synthetic_source_itr_sin_sim_params,
    ):
        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        spec = get_problem(benchmark, "temporal_encoder")
        sim_params = _sim_params_for(
            benchmark, spec, trajectories, x_grid, y_grid, t_grid,
            {
                "forcing": synthetic_sim_params,
                "source": synthetic_source_sim_params,
                "source_itr_sin": synthetic_source_itr_sin_sim_params,
            },
        )
        ds = self._dataset(spec, trajectories, x_grid, y_grid, t_grid, sim_params)

        # t_s past the trained horizon: t_grid[35] = 0.70 > 0.60.
        sid, s, k, j = 0, 35, 40, 45
        t_s, t_k = float(t_grid[s]), float(t_grid[k])
        assert t_s > self.HORIZON

        base = spec.build_item(ds, sid, s, j)
        direct_sub = spec.build_item(ds, sid, s, k)
        rollout = build_rollout_item_from_base(
            base, ds, spec, sid, base["spatial"][..., 0], t_s, t_k
        )

        np.testing.assert_allclose(
            rollout["cond_static"][0], direct_sub["cond_static"][0], rtol=1e-6
        )
        np.testing.assert_allclose(
            rollout["cond_static"][0], (t_k - t_s) / self.HORIZON, rtol=1e-6
        )


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

    def test_single_substep_matches_direct_for_source_itr_sin(
        self, synthetic_trajectories, synthetic_source_itr_sin_sim_params
    ):
        """Same equivalence through the new source_itr_sin branch, end to end.

        rtol is loosened relative to the item-level tests because this compares
        two forward passes, not two arrays.
        """
        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        spec = get_problem("source_itr_sin", "temporal_encoder")
        ds = _make_dataset(
            trajectories, x_grid, y_grid, t_grid, synthetic_source_itr_sin_sim_params, spec
        )
        model = FNO2d(
            modes1=2,
            modes2=2,
            width=8,
            in_channels=spec.dims.in_channels,
            out_channels=1,
            n_layers=2,
            cond_static_dim=spec.dims.cond_static_dim,
            temporal_token_dim=spec.dims.temporal_token_dim,
            temporal_hidden=16,
            forcing_embed_dim=16,
            use_forcing_time_aug=spec.dims.use_forcing_time_aug,
            s_y_channel=spec.dims.s_y_channel,
        )
        model.eval()
        sid, s, j = 0, 0, 20
        item = spec.build_item(ds, sid, s, j)

        with torch.no_grad():
            direct = _direct_model_pred(model, item)
            rollout = predict_autoregressive(
                model,
                ds,
                sim_id=sid,
                s=s,
                j=j,
                num_substeps=1,
                device=torch.device("cpu"),
            )

        torch.testing.assert_close(rollout, direct, rtol=1e-5, atol=1e-6)


class TestRolloutEvalMetricContract:
    def test_rollout_evaluate_omits_unified_metrics(
        self, synthetic_trajectories, synthetic_sim_params, small_fno2d
    ):
        """The rollout path returns only the rel/iface/boundary keys.

        eval_all_seeds relies on this contract: the unified per-sample metrics
        (nrmse, rmse_K, ...) are simply not computed under rollout, so a consumer
        must treat them as missing (NaN), never as a real 0.0 score.
        """
        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        spec = get_problem("forcing", "temporal_encoder")
        ds = _make_dataset(trajectories, x_grid, y_grid, t_grid, synthetic_sim_params, spec)
        loader = types.SimpleNamespace(dataset=ds)
        small_fno2d.eval()

        metrics = evaluate(
            small_fno2d,
            loader,
            torch.device("cpu"),
            rollout_options=RolloutOptions(enabled=True, num_substeps=2),
            x_grid=x_grid,
        )

        for present in (
            "rel_l2_norm", "rel_l2_phys",
            "iface_rel_l2_norm", "iface_rel_l2_phys",
            "boundary_rel_l2_norm", "boundary_rel_l2_phys",
        ):
            assert present in metrics
        for absent in ("nrmse", "rmse_K", "gnrmse_pct", "max_err_K", "node_jump_nrmse"):
            assert absent not in metrics
        # The eval_all_seeds default surfaces the gap as NaN, not a fake 0.0.
        assert math.isnan(float(metrics.get("nrmse", float("nan"))))
