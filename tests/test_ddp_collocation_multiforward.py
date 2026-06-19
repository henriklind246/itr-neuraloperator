"""DDP guard for the collocation physics path: multiple forward passes before a
single backward.

Regression (E8_forcingPI DDP hang): the W2 collocation step forwards the
DDP-wrapped model three times per iteration (data + collocation ``t`` +
collocation ``t+dt``) and then runs ONE combined ``.backward()``. With the
default reducer (``find_unused_parameters=False`` and no ``static_graph``) the
per-bucket all-reduces are launched as buckets fill, and the fill order depends
on the (rank-dependent) data shard and collocation batch shapes. Across ranks
that order can diverge, so NCCL ends up matching different collectives and
deadlocks. ``static_graph=True`` records the autograd order on the first
iteration and supports multi-forward / params reused across graphs, which is the
fix applied in ``train.py``.

This test reproduces the *pattern* with a real 2-rank ``gloo`` group and a tiny
*real-valued* model and asserts that ``static_graph=True`` runs it to completion
with gradients synchronized across ranks. It cannot reproduce the NCCL hang
itself: that failure is NCCL-collective-ordering specific and needs the real
multi-bucket FNO, whose ``torch.cfloat`` spectral weights ``gloo`` cannot
all-reduce (the same constraint that forces ``test_ddp_parity.py`` to emulate
DDP). So this is a correctness/safety guard for the fix, not a hang repro.

Runs on CPU; spawns real processes. Skipped if gloo is unavailable.
"""

from __future__ import annotations

import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

_GLOO_OK = dist.is_available() and dist.is_gloo_available()

pytestmark = pytest.mark.skipif(not _GLOO_OK, reason="gloo backend unavailable")


class _TinyModel(nn.Module):
    """Real-valued stand-in whose parameters are reused across forwards."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 4))

    def forward(self, x):
        return self.net(x)


def _multiforward_worker(rank, world_size, static_graph, port, result_queue):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        "gloo", rank=rank, world_size=world_size, timeout=timedelta(seconds=30)
    )
    try:
        torch.manual_seed(0)
        model = _TinyModel()
        ddp = DistributedDataParallel(
            model, find_unused_parameters=False, static_graph=static_graph
        )

        # Per-rank shard so the synced gradient is a non-trivial average.
        x = torch.randn(8, 4) + rank

        # Three forward passes through the wrapped model, ONE backward — the
        # collocation physics pattern (data + collocation t + collocation t+dt).
        try:
            out = ddp(x) + ddp(x) + ddp(x)
            out.pow(2).mean().backward()
        except RuntimeError as exc:
            result_queue.put((rank, "error", str(exc)))
            return

        # If DDP synced, each rank's grad already equals the cross-rank mean.
        grads = [p.grad.detach().clone() for p in model.parameters()]
        finite = all(bool(torch.isfinite(g).all()) for g in grads)
        synced = True
        for g in grads:
            summed = g.clone()
            dist.all_reduce(summed, op=dist.ReduceOp.SUM)
            if not torch.allclose(g, summed / world_size, atol=1e-6):
                synced = False
                break
        result_queue.put((rank, "ok", bool(synced and finite)))
    finally:
        dist.destroy_process_group()


def _run(world_size, static_graph, port):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [
        ctx.Process(
            target=_multiforward_worker,
            args=(r, world_size, static_graph, port, q),
        )
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
                pytest.fail(
                    "DDP worker hung on the multi-forward / single-backward step"
                )
    return results


def test_static_graph_allows_multiforward_single_backward():
    """static_graph=True: the 3-forward/1-backward step completes and every
    parameter's gradient is synchronized (averaged) across both ranks."""
    results = _run(world_size=2, static_graph=True, port=29541)
    assert len(results) == 2
    for rank, status, ok in results:
        assert status == "ok", f"rank {rank} failed under static_graph=True: {ok}"
        assert ok, f"rank {rank} gradient not synced/finite under static_graph=True"
