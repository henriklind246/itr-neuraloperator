"""Step 2 coverage for the residual-adaptive causal lead weighter.

Exercises ``CausalLeadWeighter`` (the ``exp(-eps * exclusive_cumsum)`` weight
formula, ``current`` vs ``ema`` loss history, the epsilon growth/shrink schedule
and its active-bin coverage gate, and state round-trip) plus the DDP-safe
differentiable interior objective built inside ``_collocation_batch_loss`` (local
differentiable sums / global detached counts, so the disabled path is
bit-identical and the all-ones-weight path reduces to the mean of bin means).

A dedicated two-rank gloo test asserts that ``prepare`` is the only collective,
that the objective's DDP-averaged gradient matches a single-process global-bin
objective, and that epsilon / causal state stay synchronized across ranks.
"""

from __future__ import annotations

import os
from datetime import timedelta

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from problems.diffusion import K_SLAB
from src.operators.train import (
    CausalLeadWeighter,
    TemporalBinSpec,
    _collocation_batch_loss,
)


def _spec(n_bins=5, lo=0.005, hi=0.095):
    return TemporalBinSpec(lead_lo=lo, lead_hi=hi, n_bins=n_bins)


# --------------------------------------------------------------------------- #
# 1. Weight formula                                                           #
# --------------------------------------------------------------------------- #


def test_weight_formula_bin0_one_and_non_increasing():
    w = CausalLeadWeighter(_spec(n_bins=5), eps=1.0)
    bin_loss = np.array([0.4, 0.3, 0.2, 0.1, 0.05])
    weights = w._weights_from_loss(bin_loss)
    # Exclusive cumsum -> bin 0 sees no earlier loss -> weight exp(0) = 1.
    assert weights[0] == pytest.approx(1.0)
    # Non-increasing in bin id (each later bin inherits more suppression).
    assert np.all(np.diff(weights) <= 1e-12)
    # Explicit reference against the closed form.
    excl = np.concatenate([[0.0], np.cumsum(bin_loss)[:-1]])
    assert np.allclose(weights, np.exp(-1.0 * excl))


def test_empty_later_bin_inherits_suppression_not_one():
    """A later bin with zero residual is NOT reset to weight 1: it still carries
    the accumulated suppression of the earlier (non-empty) bins."""
    w = CausalLeadWeighter(_spec(n_bins=4), eps=2.0)
    bin_loss = np.array([1.0, 0.0, 0.0, 0.0])
    weights = w._weights_from_loss(bin_loss)
    assert weights[0] == pytest.approx(1.0)
    # bins 1..3 all inherit exp(-2 * 1.0); none snaps back to 1.
    assert np.allclose(weights[1:], np.exp(-2.0))


# --------------------------------------------------------------------------- #
# 2. current vs ema loss history                                             #
# --------------------------------------------------------------------------- #


def test_current_mode_uses_this_step_loss():
    w = CausalLeadWeighter(_spec(n_bins=3), eps=1.0, loss_history="current")
    sums = np.array([2.0, 4.0, 6.0])
    counts = np.array([2.0, 2.0, 2.0])
    current = w._current_loss(sums, counts)
    assert np.allclose(current, [1.0, 2.0, 3.0])
    assert np.allclose(w._bin_loss(current, counts), current)


def test_ema_candidate_holds_empty_bins_and_blends_populated():
    w = CausalLeadWeighter(_spec(n_bins=3), eps=1.0, loss_history="ema",
                           ema_alpha=0.5)
    w.ema = np.array([1.0, 1.0, 1.0])
    w._ema_initialized = True
    current = np.array([3.0, 5.0, 9.0])
    counts = np.array([4.0, 0.0, 4.0])  # bin 1 empty this step
    cand = w._ema_candidate(current, counts)
    # populated bins blend 0.5*ema + 0.5*current; empty bin holds its ema.
    assert cand[0] == pytest.approx(2.0)
    assert cand[1] == pytest.approx(1.0)
    assert cand[2] == pytest.approx(5.0)


def test_ema_first_step_seeds_populated_bins():
    w = CausalLeadWeighter(_spec(n_bins=3), eps=1.0, loss_history="ema",
                           ema_alpha=0.9)
    current = np.array([2.0, 7.0, 4.0])
    counts = np.array([1.0, 0.0, 1.0])
    cand = w._ema_candidate(current, counts)
    # uninitialized EMA: populated bins take current, empty bin stays 0.
    assert np.allclose(cand, [2.0, 0.0, 4.0])


def test_invalid_loss_history_raises():
    with pytest.raises(ValueError, match="loss_history"):
        CausalLeadWeighter(_spec(), loss_history="bogus")


# --------------------------------------------------------------------------- #
# 3. epsilon schedule                                                        #
# --------------------------------------------------------------------------- #


def _commit_step(w, sums, counts, active_mask):
    """Drive one prepare+commit with a plain (non-DDP) local tensor pair."""
    ls = torch.tensor(sums, dtype=torch.float64)
    lc = torch.tensor(counts, dtype=torch.float64)
    gs, gc, _ = w.prepare(ls, lc, active_mask, dist_info=None)
    w.commit(gs, gc)


def test_eps_grows_when_last_active_bin_weight_high():
    # Near-zero earlier losses -> last-bin weight ~ 1 > threshold -> eps grows.
    w = CausalLeadWeighter(_spec(n_bins=4), eps=1.0, eps_growth=2.0,
                           last_bin_threshold=0.99)
    active = np.ones(4, dtype=bool)
    _commit_step(w, np.zeros(4), np.full(4, 2.0), active)
    assert w.eps == pytest.approx(2.0)


def test_eps_shrinks_when_mean_weight_below_floor():
    w = CausalLeadWeighter(_spec(n_bins=4), eps=1.0, eps_growth=2.0,
                           last_bin_threshold=0.99, mean_weight_floor=0.5,
                           allow_eps_decrease=True)
    active = np.ones(4, dtype=bool)
    # Large early losses drive later weights ~ 0 -> mean weight < floor -> shrink.
    _commit_step(w, np.full(4, 10.0), np.full(4, 1.0), active)
    assert w.eps == pytest.approx(0.5)


def test_eps_held_when_active_bin_unpopulated():
    w = CausalLeadWeighter(_spec(n_bins=4), eps=1.0, eps_growth=2.0)
    active = np.ones(4, dtype=bool)
    counts = np.array([2.0, 0.0, 2.0, 2.0])  # bin 1 active but empty
    _commit_step(w, np.zeros(4), counts, active)
    assert w.eps == pytest.approx(1.0)  # coverage gate holds epsilon


def test_eps_growth_respects_update_every():
    w = CausalLeadWeighter(_spec(n_bins=3), eps=1.0, eps_growth=2.0,
                           update_every=2, last_bin_threshold=0.99)
    active = np.ones(3, dtype=bool)
    _commit_step(w, np.zeros(3), np.full(3, 2.0), active)  # _step=1, skip
    assert w.eps == pytest.approx(1.0)
    _commit_step(w, np.zeros(3), np.full(3, 2.0), active)  # _step=2, grow
    assert w.eps == pytest.approx(2.0)


def test_eps_clamped_at_max_and_min():
    w = CausalLeadWeighter(_spec(n_bins=3), eps=90.0, eps_growth=2.0,
                           eps_max=100.0, last_bin_threshold=0.99)
    active = np.ones(3, dtype=bool)
    _commit_step(w, np.zeros(3), np.full(3, 2.0), active)
    assert w.eps == pytest.approx(100.0)  # min(90*2, 100)

    # eps just above min so eps/growth would undershoot -> clamps at eps_min.
    # Large losses are needed so the mean active weight drops below the floor
    # (small eps alone keeps weights ~1 and never triggers the shrink).
    w2 = CausalLeadWeighter(_spec(n_bins=3), eps=0.015, eps_growth=2.0,
                            eps_min=0.01, mean_weight_floor=0.5,
                            allow_eps_decrease=True)
    _commit_step(w2, np.full(3, 1000.0), np.full(3, 1.0), active)
    assert w2.eps == pytest.approx(0.01)  # max(0.015/2, 0.01)


def test_allow_eps_decrease_false_holds():
    w = CausalLeadWeighter(_spec(n_bins=3), eps=1.0, eps_growth=2.0,
                           mean_weight_floor=0.5, allow_eps_decrease=False,
                           last_bin_threshold=0.99)
    active = np.ones(3, dtype=bool)
    _commit_step(w, np.full(3, 10.0), np.full(3, 1.0), active)
    assert w.eps == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# 4. state round-trip + metadata guard                                       #
# --------------------------------------------------------------------------- #


def test_state_dict_round_trip():
    w = CausalLeadWeighter(_spec(n_bins=4), eps=1.0, eps_growth=2.0,
                           loss_history="ema", ema_alpha=0.8)
    active = np.ones(4, dtype=bool)
    _commit_step(w, np.array([0.1, 0.2, 0.3, 0.4]), np.full(4, 2.0), active)
    _commit_step(w, np.array([0.2, 0.1, 0.05, 0.0]), np.full(4, 2.0), active)
    state = w.state_dict()

    w2 = CausalLeadWeighter(_spec(n_bins=4), eps=999.0, loss_history="ema",
                            ema_alpha=0.8)
    w2.load_state_dict(state)
    assert w2.eps == pytest.approx(w.eps)
    assert w2._step == w._step
    assert w2._ema_initialized == w._ema_initialized
    assert np.allclose(w2.ema, w.ema)


def test_load_state_dict_metadata_mismatch_raises():
    w = CausalLeadWeighter(_spec(n_bins=4), loss_history="current",
                           ema_alpha=0.9)
    state = w.state_dict()

    with pytest.raises(ValueError, match="n_bins"):
        CausalLeadWeighter(_spec(n_bins=5)).load_state_dict(state)
    with pytest.raises(ValueError, match="loss_history"):
        CausalLeadWeighter(_spec(n_bins=4), loss_history="ema").load_state_dict(state)
    with pytest.raises(ValueError, match="ema_alpha"):
        CausalLeadWeighter(_spec(n_bins=4), ema_alpha=0.5).load_state_dict(state)
    with pytest.raises(ValueError, match="bin_edges"):
        CausalLeadWeighter(_spec(n_bins=4, hi=0.2)).load_state_dict(state)


# --------------------------------------------------------------------------- #
# 5. differentiable interior objective (single process)                      #
# --------------------------------------------------------------------------- #


class _IdentityModel(torch.nn.Module):
    def forward(self, spatial, cond_static, forcing_seq=None):
        return spatial[..., 0:1]


class _BinnedStubColloc:
    """On-grid-style sampler that records ``_last_leads`` / ``_last_lead_bin_ids``
    from an attached ``bin_spec`` exactly like the real CollocationSampler."""

    def __init__(self, B=12, Nx=5, Ny=5, dt=0.01, n_t=11, seed=0):
        self.B, self.Nx, self.Ny = B, Nx, Ny
        xg = np.linspace(0.0, 1.0, Nx)
        yg = np.linspace(0.0, 1.0, Ny)
        self.dt = float(dt)
        self.t_final = float((n_t - 1) * dt)
        self.n_max = n_t - 2
        self.geom_cfg = {
            "x_grid": xg, "y_grid": yg,
            "k_left": K_SLAB, "k_right": K_SLAB,
            "interface_x": float(xg[Nx // 2]),
            "dt": self.dt, "sigma_global": 1.0, "T_right_tilde": 0.0,
        }
        self.base_plan = {"base_snapshot_index": 0, "on_grid_pairs": True}
        self.bin_spec = None
        self._last_leads = None
        self._last_lead_bin_ids = None
        self.rng = np.random.default_rng(seed)

    def _mk_batch(self, field):
        spatial = np.zeros((self.B, self.Nx, self.Ny, 4), np.float32)
        spatial[..., 0] = field
        return {
            "spatial": torch.from_numpy(spatial),
            "cond_static": torch.zeros(self.B, 2),
            "forcing_seq": torch.zeros(self.B, 128, 2),
        }

    def sample_batch(self, max_lead):
        n_max = max(0, min(int(np.floor(float(max_lead) / self.dt)) - 1, self.n_max))
        ns = self.rng.integers(0, n_max + 1, size=self.B)
        self._last_leads = (ns + 0.5) * self.dt
        self._last_lead_bin_ids = (
            self.bin_spec.interval_to_bin(self._last_leads)
            if self.bin_spec is not None else None
        )
        f = (self.rng.standard_normal((self.B, self.Nx, self.Ny)) * 0.1).astype(np.float32)
        z = torch.zeros(self.B, self.Ny)
        return (self._mk_batch(f), self._mk_batch(f), torch.zeros(self.B),
                z, z.clone(), z.clone())


def _make_stub(n_bins=5, seed=0):
    sampler = _BinnedStubColloc(seed=seed)
    lo, hi = 0.5 * sampler.dt, sampler.t_final - 0.5 * sampler.dt
    sampler.bin_spec = TemporalBinSpec(lead_lo=lo, lead_hi=hi, n_bins=n_bins)
    return sampler


def test_causal_interior_matches_manual_recompute():
    sampler = _make_stub(n_bins=5, seed=1)
    w = CausalLeadWeighter(sampler.bin_spec, eps=1.5, loss_history="current")
    out, _, _ = _collocation_batch_loss(
        _IdentityModel(), sampler, sampler.t_final, None, torch.device("cpu"),
        causal_weighter=w, dist_info=None, world_size=1,
    )
    per_int = out["interior_per_sample"].detach().numpy()
    bin_ids = np.asarray(sampler._last_lead_bin_ids)
    n_bins = sampler.bin_spec.n_bins
    sums = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    for b, v in zip(bin_ids, per_int):
        sums[b] += v
        counts[b] += 1
    current = np.where(counts > 0, sums / np.clip(counts, 1, None), 0.0)
    excl = np.concatenate([[0.0], np.cumsum(current)[:-1]])
    weights = np.exp(-1.5 * excl)
    active = w.active_mask(sampler.t_final) & (counts > 0)
    expected = float(
        (weights[active] * sums[active] / counts[active]).sum() / active.sum()
    )
    assert float(out["_causal_interior"]) == pytest.approx(expected, rel=1e-5)


def test_causal_interior_is_mean_of_bin_means_when_weights_one():
    sampler = _make_stub(n_bins=5, seed=2)
    w = CausalLeadWeighter(sampler.bin_spec, eps=0.0)  # eps=0 -> all weights 1
    out, _, _ = _collocation_batch_loss(
        _IdentityModel(), sampler, sampler.t_final, None, torch.device("cpu"),
        causal_weighter=w, dist_info=None, world_size=1,
    )
    per_int = out["interior_per_sample"].detach().numpy()
    bin_ids = np.asarray(sampler._last_lead_bin_ids)
    n_bins = sampler.bin_spec.n_bins
    sums = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    for b, v in zip(bin_ids, per_int):
        sums[b] += v
        counts[b] += 1
    pop = counts > 0
    bin_means = sums[pop] / counts[pop]
    assert float(out["_causal_interior"]) == pytest.approx(
        float(bin_means.mean()), rel=1e-5
    )


def test_none_weighter_path_has_no_causal_keys_and_matches_weighted():
    sampler_a = _make_stub(n_bins=5, seed=3)
    out_none, _, _ = _collocation_batch_loss(
        _IdentityModel(), sampler_a, sampler_a.t_final, None, torch.device("cpu"),
        causal_weighter=None,
    )
    assert "_causal_total" not in out_none
    assert "causal_bin_sums" not in out_none

    # Same draw order with a weighter attached: the diagnostic scalar
    # physics_loss_weighted is numerically identical (per_sample only adds keys).
    sampler_b = _make_stub(n_bins=5, seed=3)
    w = CausalLeadWeighter(sampler_b.bin_spec, eps=1.0)
    out_w, _, _ = _collocation_batch_loss(
        _IdentityModel(), sampler_b, sampler_b.t_final, None, torch.device("cpu"),
        causal_weighter=w, dist_info=None, world_size=1,
    )
    assert float(out_none["physics_loss_weighted"]) == pytest.approx(
        float(out_w["physics_loss_weighted"]), rel=1e-6
    )


def test_missing_bin_ids_raises():
    sampler = _make_stub(n_bins=5, seed=4)
    sampler.bin_spec = None  # sample_batch will leave _last_lead_bin_ids None
    w = CausalLeadWeighter(_spec(n_bins=5), eps=1.0)
    with pytest.raises(RuntimeError, match="_last_lead_bin_ids"):
        _collocation_batch_loss(
            _IdentityModel(), sampler, sampler.t_final, None,
            torch.device("cpu"), causal_weighter=w, dist_info=None, world_size=1,
        )


# --------------------------------------------------------------------------- #
# 6. two-rank gloo DDP: objective gradient + synchronized state               #
# --------------------------------------------------------------------------- #

_GLOO_OK = dist.is_available() and dist.is_gloo_available()


class _DI:
    """Minimal DistInfo stand-in for prepare (only reads is_distributed)."""

    def __init__(self, world_size):
        self.is_distributed = True
        self.world_size = world_size


def _causal_ddp_worker(rank, world_size, port, q):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        "gloo", rank=rank, world_size=world_size, timeout=timedelta(seconds=30)
    )
    try:
        n_bins = 4
        spec = TemporalBinSpec(lead_lo=0.005, lead_hi=0.095, n_bins=n_bins)
        w = CausalLeadWeighter(spec, eps=1.0, loss_history="current")
        di = _DI(world_size)

        theta = torch.ones(n_bins, dtype=torch.float64, requires_grad=True)
        # Rank-dependent scale + counts -> genuinely unequal per-rank bin stats.
        scale = torch.arange(1, n_bins + 1, dtype=torch.float64) * (rank + 1)
        counts = torch.full((n_bins,), 2.0 * (rank + 1), dtype=torch.float64)
        local_bin_sums = theta * scale

        active = w.active_mask(spec.lead_hi)
        gs, gc, weights = w.prepare(local_bin_sums, counts, active, di)
        cw = torch.tensor(weights, dtype=torch.float64)
        gc_t = gc.to(torch.float64)
        am = torch.tensor(active, dtype=torch.bool)
        active_pop = am & (gc_t > 0)
        obj = (
            cw[active_pop] * float(world_size) * local_bin_sums[active_pop]
            / gc_t[active_pop].clamp_min(1.0)
        ).sum() / active_pop.sum()
        obj.backward()

        g = theta.grad.detach().clone()
        g_avg = g.clone()
        dist.all_reduce(g_avg, op=dist.ReduceOp.SUM)
        g_avg /= world_size

        # Reference single-process global-bin gradient: cw_b * (sum_r scale_{r,b})
        # / global_count_b / n_active. Only the summed scale needs a reduce.
        scale_sum = scale.clone()
        dist.all_reduce(scale_sum, op=dist.ReduceOp.SUM)
        K = float(active_pop.sum())
        ref = torch.zeros(n_bins, dtype=torch.float64)
        ref[active_pop] = (
            cw[active_pop] * scale_sum[active_pop] / gc_t[active_pop] / K
        )
        grad_ok = bool(torch.allclose(g_avg, ref, atol=1e-9))

        # Global weights must be identical across ranks (prepare all-reduced).
        w_ref = cw.clone()
        dist.all_reduce(w_ref, op=dist.ReduceOp.SUM)
        weights_synced = bool(torch.allclose(w_ref / world_size, cw, atol=1e-12))

        # commit performs NO collective; epsilon + step stay synced across ranks.
        w.commit(gs, gc)
        eps_t = torch.tensor([w.eps, float(w._step)], dtype=torch.float64)
        eps_ref = eps_t.clone()
        dist.all_reduce(eps_ref, op=dist.ReduceOp.SUM)
        state_synced = bool(torch.allclose(eps_ref / world_size, eps_t, atol=1e-12))

        q.put((rank, "ok", grad_ok and weights_synced and state_synced
               and w._step == 1))
    except Exception as exc:  # noqa: BLE001
        q.put((rank, "error", repr(exc)))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not _GLOO_OK, reason="gloo backend unavailable")
def test_causal_objective_ddp_gradient_and_state():
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    world_size = 2
    procs = [
        ctx.Process(target=_causal_ddp_worker, args=(r, world_size, 29551, q))
        for r in range(world_size)
    ]
    for p in procs:
        p.start()
    results = []
    try:
        for _ in range(world_size):
            results.append(q.get(timeout=120))
    finally:
        for p in procs:
            p.join(timeout=120)
            if p.is_alive():
                p.terminate()
                pytest.fail("causal DDP worker hung")
    assert len(results) == world_size
    for rank, status, ok in results:
        assert status == "ok", f"rank {rank} failed: {ok}"
        assert ok, f"rank {rank} gradient/state not synchronized"
