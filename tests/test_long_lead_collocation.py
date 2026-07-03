"""Long-lead OOD collocation sampling (Capabilities 3 and 4).

Directly constructs the real forcing ``CollocationSampler`` (the generic path,
which never touches ``geom_cfg``) and checks:
- source-time range restriction + per-source lead feasibility (guard never trips),
- lead-bin-first sampling gives every lead bin nonzero mass (incl. the longest),
- ``collocation_lead_min`` lifts non-bridge leads into the unlabeled band,
- the cutoff-bridge injects the exact true snapshot as a detached first state and
  keeps the (T_n earlier, T_{n+1} later) time order.
"""

import numpy as np
import pytest

from data.dataset import SnapshotPairDataset
from src.operators.train import CollocationSampler

_MU = 0.0
_SIGMA = 1.0
# Full synthetic grid: t_grid = linspace(0, 1, 51) -> dt = 0.02, t_final = 1.0.
_DT = 0.02
_TOL = 1e-9


def _forcing_dataset(synthetic_trajectories, synthetic_sim_params):
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
    )


def _sampler(ds, *, batch_size, **knobs):
    # geom_cfg is unused on the generic forcing path -> pass an empty dict.
    return CollocationSampler(
        ds, ds.problem, {},
        batch_size=batch_size, dt=ds.dt, rng_seed=0, **knobs,
    )


def _spy_sources(monkeypatch, ds):
    """Capture the source index build_item is called with, one per batch row
    (plus one anchor-index call per bridge row). Order matches _last_leads."""
    orig = ds.problem.build_item
    seen = []

    def spy(ds_, sim_id, s_idx, j_idx):
        seen.append((int(sim_id), int(s_idx), int(j_idx)))
        return orig(ds_, sim_id, s_idx, j_idx)

    monkeypatch.setattr(ds.problem, "build_item", spy)
    return seen


class TestSourceTimeRange:
    def test_sources_in_range_and_feasible(
        self, monkeypatch, synthetic_trajectories, synthetic_sim_params
    ):
        ds = _forcing_dataset(synthetic_trajectories, synthetic_sim_params)
        src_min, src_max, max_lead = 0.1, 0.4, 0.5
        sampler = _sampler(
            ds, batch_size=64,
            source_time_min=src_min, source_time_max=src_max,
        )
        seen = _spy_sources(monkeypatch, ds)
        sampler.sample_batch(max_lead)

        # No bridge -> exactly one (source) build_item per row, in draw order.
        assert len(seen) == 64
        leads = sampler._last_leads
        assert leads.shape == (64,)
        for (_, s_idx, _), lead in zip(seen, leads):
            t_s = float(ds.t_grid[s_idx])
            assert src_min - _TOL <= t_s <= src_max + _TOL
            assert t_s + lead + ds.dt <= ds.t_final + 1e-6
        # Legacy lead lower bound is dt when lead_min is unset.
        assert leads.min() >= ds.dt - _TOL
        assert leads.max() <= max_lead + _TOL

    def test_empty_source_range_raises(self, synthetic_trajectories, synthetic_sim_params):
        ds = _forcing_dataset(synthetic_trajectories, synthetic_sim_params)
        with pytest.raises(ValueError, match="no source snapshot falls in the range"):
            _sampler(ds, batch_size=8, source_time_min=5.0, source_time_max=6.0)


class TestLeadBinFirst:
    def test_every_bin_gets_mass(self, synthetic_trajectories, synthetic_sim_params):
        ds = _forcing_dataset(synthetic_trajectories, synthetic_sim_params)
        lead_min, max_lead, n_bins = 0.2, 0.6, 5
        sampler = _sampler(
            ds, batch_size=400,
            source_time_min=0.0, source_time_max=0.1,
            lead_min=lead_min, lead_bins=n_bins,
        )
        sampler.sample_batch(max_lead)
        leads = sampler._last_leads

        assert leads.min() >= lead_min - _TOL
        assert leads.max() <= max_lead + _TOL
        edges = np.linspace(lead_min, max_lead, n_bins + 1)
        counts, _ = np.histogram(leads, bins=edges)
        assert (counts > 0).all(), f"each lead bin must get mass; got {counts}"
        # The longest bin (hardest leads) must be populated, not starved.
        assert counts[-1] > 0

    def test_all_draws_feasible(
        self, monkeypatch, synthetic_trajectories, synthetic_sim_params
    ):
        ds = _forcing_dataset(synthetic_trajectories, synthetic_sim_params)
        sampler = _sampler(
            ds, batch_size=200,
            source_time_min=0.0, source_time_max=0.1,
            lead_min=0.2, lead_bins=5,
        )
        seen = _spy_sources(monkeypatch, ds)
        sampler.sample_batch(0.6)
        for (_, s_idx, _), lead in zip(seen, sampler._last_leads):
            t_s = float(ds.t_grid[s_idx])
            assert t_s + lead + ds.dt <= ds.t_final + 1e-6


class TestCollocationLeadMin:
    def test_non_bridge_leads_in_band(self, synthetic_trajectories, synthetic_sim_params):
        ds = _forcing_dataset(synthetic_trajectories, synthetic_sim_params)
        lead_min, max_lead = 0.4, 0.6
        sampler = _sampler(
            ds, batch_size=64,
            source_time_min=0.0, source_time_max=0.4,
            lead_min=lead_min,
        )
        sampler.sample_batch(max_lead)
        leads = sampler._last_leads
        assert leads.min() >= lead_min - _TOL
        assert leads.max() <= max_lead + _TOL
        # No bridge configured -> no anchor side-channel.
        assert sampler._last_anchor is None

    def test_lead_min_above_max_raises(self, synthetic_trajectories, synthetic_sim_params):
        ds = _forcing_dataset(synthetic_trajectories, synthetic_sim_params)
        sampler = _sampler(
            ds, batch_size=8,
            source_time_min=0.0, source_time_max=0.2,
            lead_min=0.5,
        )
        with pytest.raises(ValueError, match="collocation_lead_min"):
            sampler.sample_batch(0.3)


class TestCutoffBridge:
    def test_bridge_injects_detached_snapshot_and_time_order(
        self, synthetic_trajectories, synthetic_sim_params
    ):
        ds = _forcing_dataset(synthetic_trajectories, synthetic_sim_params)
        anchor_lead, max_lead = 0.4, 0.6
        sampler = _sampler(
            ds, batch_size=128,
            source_time_min=0.0, source_time_max=0.3,
            lead_min=anchor_lead,
            anchor_lead_time=anchor_lead, anchor_fraction=0.5,
        )
        batch_t, batch_tdt, *_ = sampler.sample_batch(max_lead)

        assert sampler._last_anchor is not None
        mask, T0 = sampler._last_anchor
        assert tuple(mask.shape) == (128,)
        assert tuple(T0.shape) == (128, ds.Nx, ds.Ny)

        m = mask.numpy()
        assert m.dtype == np.bool_
        assert m.sum() > 0, "some bridge rows expected at anchor_fraction=0.5"
        assert m.sum() < 128, "some non-bridge rows expected at anchor_fraction=0.5"

        leads = sampler._last_leads
        # Bridge rows sit exactly at the cutoff lead; non-bridge rows at tau > tc.
        np.testing.assert_allclose(leads[m], anchor_lead, atol=1e-6)
        assert leads[~m].min() >= anchor_lead - _TOL

        # Injected true snapshot: masked rows carry the exact (nonzero) field;
        # non-bridge rows are zero-filled placeholders.
        assert T0[~mask].abs().sum().item() == pytest.approx(0.0, abs=1e-9)
        assert T0[mask].abs().sum().item() > 0.0

        # Time order: T_{n+1} is exactly dt later than T_n for every row (the FV
        # transient term is ordered). cond_static[:, 0] is the normalized lead.
        horizon = float(ds.time_norm_horizon)
        lead_t = batch_t["cond_static"][:, 0].numpy()
        lead_tdt = batch_tdt["cond_static"][:, 0].numpy()
        np.testing.assert_allclose(lead_t, leads / horizon, atol=1e-5)
        np.testing.assert_allclose(lead_tdt, (leads + ds.dt) / horizon, atol=1e-5)

    def test_bridge_snapshot_matches_nearest_grid_index(
        self, synthetic_trajectories, synthetic_sim_params
    ):
        """A masked bridge row's injected T0 equals build_item at the grid index
        nearest t_s + anchor_lead (normalized), detached from the source."""
        ds = _forcing_dataset(synthetic_trajectories, synthetic_sim_params)
        anchor_lead = 0.4
        # anchor_fraction = 1.0 -> every row is a bridge row (deterministic check).
        sampler = _sampler(
            ds, batch_size=16,
            source_time_min=0.0, source_time_max=0.3,
            lead_min=anchor_lead,
            anchor_lead_time=anchor_lead, anchor_fraction=1.0,
        )
        seen = []
        orig = ds.problem.build_item

        def spy(ds_, sim_id, s_idx, j_idx):
            seen.append((int(sim_id), int(s_idx), int(j_idx)))
            return orig(ds_, sim_id, s_idx, j_idx)

        ds.problem.build_item = spy
        try:
            sampler.sample_batch(0.6)
        finally:
            ds.problem.build_item = orig

        mask, T0 = sampler._last_anchor
        assert mask.all(), "anchor_fraction=1.0 should make every row a bridge"
        # With all-bridge rows, build_item is called twice per row: the source
        # (s, s) then the anchor (k, k). Verify each anchor index is the grid
        # index nearest t_s + anchor_lead.
        source_calls = seen[0::2]
        anchor_calls = seen[1::2]
        assert len(source_calls) == 16 and len(anchor_calls) == 16
        for (sid_s, s_idx, _), (sid_a, k_idx, k_idx2) in zip(source_calls, anchor_calls):
            assert sid_a == sid_s
            assert k_idx == k_idx2, "anchor is an on-grid (k, k) snapshot"
            t_target = float(ds.t_grid[s_idx]) + anchor_lead
            expected_k = int(np.argmin(np.abs(np.asarray(ds.t_grid, float) - t_target)))
            assert k_idx == expected_k
