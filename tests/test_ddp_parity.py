"""Gradient-parity check: single-process vs DDP-style gradient averaging.

Motivation (forcing-itr DDP regression): a 4-GPU DDP run and a 1-GPU run with
identical config/data/seed/batch diverged badly (DDP train rel-L2 ~5.7% vs
~0.5%). The training code is *designed* so that DDP matches single-GPU:
`per_gpu_bs = batch_size // world_size` keeps the global batch fixed, the loss
is a `mean`, and DDP all-reduces (averages) per-rank gradients. This test
verifies that mathematical claim directly.

It builds a fixed model + fixed global batch, computes the gradient over the
whole batch in one pass (the single-GPU reference), then computes each rank's
gradient on its shard (per-rank batch = N // world_size) and averages them the
way DDP's gradient all-reduce does. With equal shards and a mean-reduction loss
the two must agree up to float summation order.

Why emulate rather than spawn real `gloo` workers: the SpectralConv2d weights
are ``torch.cfloat`` (complex). gloo cannot broadcast or all-reduce complex
tensors ("Invalid scalar type"), so a real CPU DDP wrap crashes at init. The
production runs use the NCCL/CUDA backend, which handles complex fine. The
gradient-averaging arithmetic this test checks is backend-independent, so the
single-process emulation faithfully reproduces what NCCL does to the gradients.

A failure here localizes the regression to gradient computation/reduction. A
pass pushes the investigation toward data coverage / SGD trajectory instead.

Runs on CPU; no GPU or cluster required.
"""

import numpy as np
import pytest
import torch

from src.operators.fno2d import FNO2d
from src.operators.losses import SpatiallyWeightedMSE


IN_CHANNELS = 20
COND_STATIC_DIM = 23
TEMPORAL_TOKEN_DIM = 5
TEMPORAL_SAMPLES = 64
NX = NY = 11
MODEL_SEED = 1234
BATCH_SEED = 5678


def _build_model() -> FNO2d:
    torch.manual_seed(MODEL_SEED)
    return FNO2d(
        modes1=2,
        modes2=2,
        width=8,
        in_channels=IN_CHANNELS,
        out_channels=1,
        n_layers=2,
        cond_static_dim=COND_STATIC_DIM,
        temporal_token_dim=TEMPORAL_TOKEN_DIM,
        temporal_hidden=16,
        forcing_embed_dim=16,
        forcing_spatial_dim=4,
    )


def _make_batch(n: int):
    g = torch.Generator().manual_seed(BATCH_SEED)
    spatial = torch.randn(n, NX, NY, IN_CHANNELS, generator=g)
    cond = torch.randn(n, COND_STATIC_DIM, generator=g)
    forcing = torch.randn(n, TEMPORAL_SAMPLES, TEMPORAL_TOKEN_DIM, generator=g)
    y = torch.randn(n, NX, NY, 1, generator=g)
    return spatial, cond, forcing, y


def _make_batch_seeded(n: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    spatial = torch.randn(n, NX, NY, IN_CHANNELS, generator=g)
    cond = torch.randn(n, COND_STATIC_DIM, generator=g)
    forcing = torch.randn(n, TEMPORAL_SAMPLES, TEMPORAL_TOKEN_DIM, generator=g)
    y = torch.randn(n, NX, NY, 1, generator=g)
    return spatial, cond, forcing, y


def _grids():
    x_grid = np.linspace(0.0, 1.0, NX).astype(np.float32)
    y_grid = np.linspace(0.0, 1.0, NY).astype(np.float32)
    return x_grid, y_grid


def _loss_fn():
    x_grid, y_grid = _grids()
    return SpatiallyWeightedMSE(
        x_grid=x_grid,
        y_grid=y_grid,
        interface_x=0.5,
        interface_half_width=0.05,
        interface_weight=10.0,
    )


def _grads_on(model_state, batch, interface_x=None):
    """Gradient of the mean-reduction loss over the given batch.

    ``interface_x is None`` exercises the byte-identical fixed-band path; a
    ``(n,)`` tensor exercises the per-sample band path.
    """
    model = _build_model()
    model.load_state_dict(model_state)
    model.eval()  # disable spectral dropout so the grad is deterministic
    loss_fn = _loss_fn()
    spatial, cond, forcing, y = batch

    model.zero_grad(set_to_none=False)
    pred = model(spatial, cond, forcing)
    loss = loss_fn(pred, y, interface_x)
    loss.backward()
    return {name: p.grad.detach().clone() for name, p in model.named_parameters()}


def _ddp_averaged_grads(world_size, model_state, batch, interface_x=None):
    """Emulate DDP: per-rank grad on an equal shard, then average across ranks.

    DDP's gradient all-reduce uses ReduceOp.AVG (== SUM / world_size). Each rank
    runs the *same* loss on its shard, so the all-reduced gradient is the mean of
    the per-rank gradients.
    """
    spatial, cond, forcing, y = batch
    n = spatial.shape[0]
    per = n // world_size
    assert per * world_size == n, "global batch must divide evenly across ranks"

    accum = None
    for rank in range(world_size):
        sl = slice(rank * per, (rank + 1) * per)
        shard = (spatial[sl], cond[sl], forcing[sl], y[sl])
        ix = None if interface_x is None else interface_x[sl]
        g = _grads_on(model_state, shard, ix)
        if accum is None:
            accum = {k: v.clone() for k, v in g.items()}
        else:
            for k in accum:
                accum[k] += g[k]
    return {k: v / world_size for k, v in accum.items()}


@pytest.mark.parametrize("world_size", [2, 4])
def test_ddp_gradient_matches_single_process(world_size):
    """DDP-averaged gradient must equal the single-process gradient on the same
    global batch when shards are equal (per-rank batch = N // world_size)."""
    n = 16 * world_size  # evenly divisible; keeps each shard a clean size
    model = _build_model()
    model_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    batch = _make_batch(n)

    single = _grads_on(model_state, batch)
    ddp = _ddp_averaged_grads(world_size, model_state, batch)

    assert set(single.keys()) == set(ddp.keys())

    mismatches = []
    for name in single:
        if not torch.allclose(single[name], ddp[name], atol=1e-6, rtol=1e-4):
            max_abs = (single[name] - ddp[name]).abs().max().item()
            ref = single[name].abs().max().item()
            mismatches.append((name, max_abs, ref))

    assert not mismatches, (
        "DDP-averaged gradient differs from single-process gradient on identical "
        f"data (world_size={world_size}). Worst offenders "
        "(param, max_abs_diff, ref_scale): "
        + "; ".join(f"{n}: {d:.3e} vs scale {r:.3e}" for n, d, r in mismatches[:8])
    )


@pytest.mark.parametrize("world_size", [2, 4])
def test_ddp_gradient_matches_single_process_per_sample(world_size):
    """Per-sample interface band must preserve DDP/single-process gradient parity.

    The per-sample weight is mean-1-normalized per sample, so the loss stays a
    mean reduction that is linear across equal shards. DDP-averaging the per-rank
    gradients must therefore still equal the single-process gradient on the same
    global batch with the same per-sample ``interface_x`` vector."""
    n = 16 * world_size
    model = _build_model()
    model_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    batch = _make_batch(n)
    g = torch.Generator().manual_seed(4321)
    interface_x = 0.2 + 0.6 * torch.rand(n, generator=g)  # sampled in [0.2, 0.8]

    single = _grads_on(model_state, batch, interface_x)
    ddp = _ddp_averaged_grads(world_size, model_state, batch, interface_x)

    mismatches = []
    for name in single:
        if not torch.allclose(single[name], ddp[name], atol=1e-6, rtol=1e-4):
            max_abs = (single[name] - ddp[name]).abs().max().item()
            ref = single[name].abs().max().item()
            mismatches.append((name, max_abs, ref))

    assert not mismatches, (
        "Per-sample interface DDP-averaged gradient differs from single-process "
        f"gradient (world_size={world_size}). Worst offenders "
        "(param, max_abs_diff, ref_scale): "
        + "; ".join(f"{n}: {d:.3e} vs scale {r:.3e}" for n, d, r in mismatches[:8])
    )


def _adamw():
    """Match the production optimizer (config: AdamW, wd=3e-5, lr=2e-3)."""
    return dict(lr=2e-3, weight_decay=3e-5)


def _step(model, optimizer, loss_fn, batch, grad_clip):
    spatial, cond, forcing, y = batch
    optimizer.zero_grad()
    pred = model(spatial, cond, forcing)
    loss = loss_fn(pred, y)
    loss.backward()
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
    optimizer.step()


def _run_trajectory(world_size, optim, grad_clip, n_steps, per, base_seed=9000):
    """Run single-process vs DDP-emulated training on *identical* per-step data.

    Returns the max abs parameter difference after ``n_steps`` optimizer steps.
    DDP-emulation averages per-shard gradients exactly as DDP's all-reduce does.
    """
    n = per * world_size
    state = {k: v.detach().clone() for k, v in _build_model().state_dict().items()}

    def make_opt(m):
        if optim == "sgd":
            return torch.optim.SGD(m.parameters(), lr=2e-3)
        return torch.optim.AdamW(m.parameters(), **_adamw())

    ref = _build_model(); ref.load_state_dict(state); ref.eval()
    ddp = _build_model(); ddp.load_state_dict(state); ddp.eval()
    ref_opt, ddp_opt = make_opt(ref), make_opt(ddp)
    ref_loss, ddp_loss = _loss_fn(), _loss_fn()

    for step in range(n_steps):
        batch = _make_batch_seeded(n, seed=base_seed + step)
        _step(ref, ref_opt, ref_loss, batch, grad_clip)

        spatial, cond, forcing, y = batch
        ddp_opt.zero_grad()
        accum = None
        for rank in range(world_size):
            sl = slice(rank * per, (rank + 1) * per)
            loss = ddp_loss(ddp(spatial[sl], cond[sl], forcing[sl]), y[sl])
            grads = torch.autograd.grad(loss, list(ddp.parameters()))
            accum = ([g.clone() for g in grads] if accum is None
                     else [a + g for a, g in zip(accum, grads)])
        for p, g in zip(ddp.parameters(), accum):
            p.grad = g / world_size
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(ddp.parameters(), max_norm=grad_clip)
        ddp_opt.step()

    ref_params = dict(ref.named_parameters())
    return max((p - ref_params[name]).abs().max().item()
               for name, p in ddp.named_parameters())


@pytest.mark.parametrize("world_size", [2, 4])
def test_ddp_sgd_trajectory_matches_single_process(world_size):
    """End-to-end update-rule parity under a *linear* optimizer (SGD + grad_clip).

    On identical per-step data, DDP gradient-averaging + grad_clip + SGD must
    reproduce the single-process trajectory to float32 epsilon. This proves the
    full DDP update path is correct, not just the first gradient.
    """
    max_diff = _run_trajectory(world_size, "sgd", grad_clip=1.0, n_steps=15, per=12)
    assert max_diff < 1e-5, (
        f"SGD DDP trajectory diverged from single-process (world_size={world_size}, "
        f"max|Δparam|={max_diff:.3e}). A linear optimizer on identical data must "
        "match to roundoff; divergence here indicates a real reduction bug."
    )


@pytest.mark.parametrize("world_size", [2, 4])
def test_adamw_is_not_reproducible_across_world_size(world_size):
    """Documents the actual forcing-itr regression mechanism.

    With AdamW, *identical* per-step data still produces macroscopically different
    trajectories across world sizes. AdamW's step is ~lr*sign(g) when the second
    moment is small, so the ~1e-7 reduction-order roundoff (4 per-rank means vs one
    global mean) flips the sign of near-zero gradient coordinates -> ~lr-scale
    per-step update differences. SGD does not have this problem (see the SGD test).

    This is not a bug in the DDP code; it is inherent optimizer non-determinism
    across GPU counts, and it compounds over a run. Combined with the different
    data *ordering* a DistributedSampler produces, it explains why the 4-GPU run
    diverged from the 1-GPU run despite identical config/data/seed.
    """
    sgd_diff = _run_trajectory(world_size, "sgd", grad_clip=1.0, n_steps=10, per=12)
    adamw_diff = _run_trajectory(world_size, "adamw", grad_clip=1.0, n_steps=10, per=12)

    # AdamW diverges by orders of magnitude more than SGD, on the scale of lr.
    assert adamw_diff > 1e-4, (
        f"Expected AdamW reduction non-determinism (world_size={world_size}), "
        f"got max|Δparam|={adamw_diff:.3e}"
    )
    assert adamw_diff > 100 * sgd_diff, (
        f"AdamW divergence ({adamw_diff:.3e}) should dwarf SGD divergence "
        f"({sgd_diff:.3e})"
    )
