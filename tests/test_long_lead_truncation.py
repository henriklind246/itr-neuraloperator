"""TRAIN-split pair truncation for long-lead OOD adaptation (Capability 1).

Exercises the real forcing ``SnapshotPairDataset`` truncation knobs
``max_lead_time`` / ``max_target_time`` and the ``+eps = 0.5*dt`` boundary
tolerance (a pair landing exactly on the cutoff is kept, the next one dropped).
"""

import numpy as np
import pytest

from data.dataset import SnapshotPairDataset

_MU = 0.0
_SIGMA = 1.0
# synthetic_trajectories uses t_grid = linspace(0, 1, 51) -> dt = 0.02, so the
# truncation tolerance is eps = 0.5*dt = 0.01 and leads are multiples of 0.02.
_DT = 0.02
_EPS = 0.5 * _DT
_TC = 0.2


def _build(synthetic_trajectories, synthetic_sim_params, **kwargs):
    trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
    return SnapshotPairDataset(
        trajectories=trajectories,
        t_grid=t_grid,
        x_grid=x_grid,
        y_grid=y_grid,
        sim_ids=np.arange(4),
        sim_params=synthetic_sim_params,
        mu_global=_MU,
        sigma_global=_SIGMA,
        **kwargs,
    )


def _target_times(ds):
    return np.array([float(ds.t_grid[j]) for (_, _, j) in ds._pairs], dtype=np.float64)


class TestMaxLeadTruncation:
    def test_no_pair_exceeds_lead_cap(self, synthetic_trajectories, synthetic_sim_params):
        ds = _build(synthetic_trajectories, synthetic_sim_params, max_lead_time=_TC)
        assert ds._lead_times.size > 0
        assert ds._lead_times.max() <= _TC + _EPS + 1e-9

    def test_boundary_lead_kept_next_dropped(self, synthetic_trajectories, synthetic_sim_params):
        """A pair at exactly tau == tc survives the +eps tolerance; tc+dt drops."""
        ds = _build(synthetic_trajectories, synthetic_sim_params, max_lead_time=_TC)
        leads = ds._lead_times
        assert np.any(np.isclose(leads, _TC, atol=1e-6)), "lead == tc must be kept"
        assert not np.any(np.isclose(leads, _TC + _DT, atol=1e-6)), "lead == tc+dt must drop"

    def test_uncapped_spans_full_leads(self, synthetic_trajectories, synthetic_sim_params):
        ds = _build(synthetic_trajectories, synthetic_sim_params)
        # Longest possible lead is t_grid[-1] - t_grid[0] = 1.0; well past tc.
        assert ds._lead_times.max() == pytest.approx(1.0, abs=1e-6)
        assert np.any(ds._lead_times > _TC + _EPS)


class TestMaxTargetTruncation:
    def test_no_pair_exceeds_target_cap(self, synthetic_trajectories, synthetic_sim_params):
        ds = _build(synthetic_trajectories, synthetic_sim_params, max_target_time=_TC)
        targets = _target_times(ds)
        assert targets.size > 0
        assert targets.max() <= _TC + _EPS + 1e-9

    def test_boundary_target_kept_next_dropped(self, synthetic_trajectories, synthetic_sim_params):
        ds = _build(synthetic_trajectories, synthetic_sim_params, max_target_time=_TC)
        targets = _target_times(ds)
        assert np.any(np.isclose(targets, _TC, atol=1e-6)), "target == tc must be kept"
        assert not np.any(np.isclose(targets, _TC + _DT, atol=1e-6)), "target == tc+dt must drop"

    def test_uncapped_spans_full_targets(self, synthetic_trajectories, synthetic_sim_params):
        ds = _build(synthetic_trajectories, synthetic_sim_params)
        assert _target_times(ds).max() == pytest.approx(1.0, abs=1e-6)


class TestTruncationComposition:
    def test_capped_is_strict_subset_of_uncapped(
        self, synthetic_trajectories, synthetic_sim_params
    ):
        full = _build(synthetic_trajectories, synthetic_sim_params)
        capped = _build(synthetic_trajectories, synthetic_sim_params, max_lead_time=_TC)
        assert len(capped._pairs) < len(full._pairs)
        assert set(capped._pairs).issubset(set(full._pairs))

    def test_pairs_remain_lead_sorted(self, synthetic_trajectories, synthetic_sim_params):
        ds = _build(synthetic_trajectories, synthetic_sim_params, max_lead_time=_TC)
        leads = ds._lead_times
        assert np.all(np.diff(leads) >= -1e-9), "pairs must stay lead-sorted for curriculum"
