"""DDP guard for the collocation physics path: cross-rank RNG seed parity.

Regression (E1_diffusion_50ep DDP hang, job 12379348): the physics-only
collocation step forwards the DDP-wrapped model three times per iteration under
``static_graph=True``. static_graph freezes the per-bucket all-reduce schedule
from the FIRST iteration's autograd order. If the first iteration draws a
different collocation batch / anchor mask on each rank, the anchored
``torch.where`` hard-injection changes which rows of ``yt`` carry gradient, the
frozen bucket schedule diverges across ranks, and the reducer deadlocks on
mismatched-size ALLREDUCEs (NCCL watchdog abort, exit -6).

The fix in ``train.py`` seeds the collocation sampler IDENTICALLY on every rank
(``rng_seed=seed``, no ``+ rank`` offset). This test guards the invariant that
fix relies on WITHOUT needing a real NCCL group: two samplers built with the
same seed must draw byte-identical collocation batches and anchor masks (so
every rank builds the same static graph), while different seeds -- the old
per-rank offset -- decorrelate the draw and would have diverged the graph.
"""

import numpy as np
import torch

from data.dataset import SnapshotPairDataset
from src.operators.train import CollocationSampler

_MU = 0.0
_SIGMA = 1.0


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


def _anchored_sampler(ds, *, rng_seed):
    # Cutoff-bridge knobs give a data-dependent anchor mask (anchor_fraction=0.5),
    # the generic-path analogue of the diffusion `anchored_first_step` n==0 mask.
    return CollocationSampler(
        ds, ds.problem, {},
        batch_size=128, dt=ds.dt, rng_seed=rng_seed,
        source_time_min=0.0, source_time_max=0.3,
        lead_min=0.4, anchor_lead_time=0.4, anchor_fraction=0.5,
    )


def _draw(sampler, max_lead=0.6):
    batch_t, batch_tdt, R_c, qL_n, qL_np1, qL_int = sampler.sample_batch(max_lead)
    mask, T0 = sampler._last_anchor
    return batch_t, batch_tdt, R_c, mask, T0


def test_same_seed_draws_identical_batch_and_anchor_mask(
    synthetic_trajectories, synthetic_sim_params
):
    """Two independently-built samplers sharing rng_seed draw byte-identical
    collocation batches and anchor masks -- the property that makes the DDP
    static graph identical across ranks."""
    ds_a = _forcing_dataset(synthetic_trajectories, synthetic_sim_params)
    ds_b = _forcing_dataset(synthetic_trajectories, synthetic_sim_params)
    bt_a, btdt_a, rc_a, mask_a, t0_a = _draw(_anchored_sampler(ds_a, rng_seed=0))
    bt_b, btdt_b, rc_b, mask_b, t0_b = _draw(_anchored_sampler(ds_b, rng_seed=0))

    assert torch.equal(mask_a, mask_b), "anchor mask must match across ranks"
    assert torch.equal(t0_a, t0_b)
    assert torch.equal(bt_a["spatial"], bt_b["spatial"])
    assert torch.equal(btdt_a["spatial"], btdt_b["spatial"])
    assert torch.equal(bt_a["cond_static"], bt_b["cond_static"])
    assert torch.equal(rc_a, rc_b)


def test_per_rank_offset_seed_decorrelates_draw(
    synthetic_trajectories, synthetic_sim_params
):
    """The old ``seed + rank`` scheme yields a different draw per rank; capturing
    that here documents why it diverged the static-graph bucket schedule."""
    ds_r0 = _forcing_dataset(synthetic_trajectories, synthetic_sim_params)
    ds_r1 = _forcing_dataset(synthetic_trajectories, synthetic_sim_params)
    bt_r0, _, _, _, _ = _draw(_anchored_sampler(ds_r0, rng_seed=0))
    bt_r1, _, _, _, _ = _draw(_anchored_sampler(ds_r1, rng_seed=1))

    assert not torch.equal(bt_r0["spatial"], bt_r1["spatial"])
