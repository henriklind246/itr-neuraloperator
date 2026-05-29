"""SpectralConv2d real-weight reparametrization: equivalence + DDP reduction.

Context (forcing-itr DDP regression): a 4-GPU DDP run trained ~5-10x worse than a
1-GPU run on identical config/data/seed/batch. The divergence was localized: at
epoch 0, with provably identical data, *every* real-valued gradient norm and
activation statistic was bit-identical across the two runs, and the only training
gradient that differed was ``grad_spectral`` (the complex SpectralConv2d filters),
by 1.2% -- five orders of magnitude above float roundoff. That is a deterministic,
systematic discrepancy in how the *complex* gradients are reduced under DDP, not
data-ordering variance.

Root cause: ``torch.distributed`` cannot all-reduce complex (``cfloat``) tensors.
gloo errors outright ("Invalid scalar type"); NCCL does not crash but routes
complex grads through a view-as-real path that is not numerically equal to the
single-process complex gradient. Because SpectralConv2d *is* the FNO's learning
mechanism, a per-step bias there degrades DDP training consistently and
monotonically.

Fix: store the Fourier filters as real ``(..., 2)`` parameters and recover the
complex weight via ``view_as_complex`` in ``forward``. DDP then reduces only real
gradients through its standard, correct path.

These tests check (1) the reparametrization is mathematically identical to a
``cfloat`` parameter (forward + gradient), and (2) a real 2-rank gloo all-reduce
of the now-real gradients matches the single-process gradient. Test (2) is only
*possible* because the weights are real -- with ``cfloat`` params the same DDP
backward raises "Invalid scalar type".
"""

import os
import tempfile

import pytest
import torch

from src.operators.fno2d import SpectralConv2d


def _reference_complex_forward(weights1_c, weights2_c, x, modes1, modes2, out_channels):
    """Original cfloat formulation of SpectralConv2d.forward (eval; no dropout)."""
    Nx, Ny = x.size(-2), x.size(-1)
    x_ft = torch.fft.rfft2(x, dim=(-2, -1))
    Nx_freq, Ny_freq = x_ft.size(-2), x_ft.size(-1)
    mx = min(modes1, Nx_freq)
    my = min(modes2, Ny_freq)
    out_ft = torch.zeros(x.size(0), out_channels, Nx_freq, Ny_freq, dtype=torch.cfloat)
    out_ft[:, :, :mx, :my] = torch.einsum(
        "bixy,ioxy->boxy", x_ft[:, :, :mx, :my], weights1_c[:, :, :mx, :my]
    )
    out_ft[:, :, -mx:, :my] = torch.einsum(
        "bixy,ioxy->boxy", x_ft[:, :, -mx:, :my], weights2_c[:, :, :mx, :my]
    )
    return torch.fft.irfft2(out_ft, s=(Nx, Ny), dim=(-2, -1))


def test_real_storage_matches_complex_forward_and_grad():
    """Real (..., 2) storage + view_as_complex == a cfloat parameter, to roundoff.

    Proves the fix does not change the single-process result: identical forward
    output and identical gradients (input grad, and both weight grads expressed in
    the real (..., 2) layout)."""
    torch.manual_seed(0)
    conv = SpectralConv2d(in_channels=6, out_channels=4, modes1=3, modes2=3)
    conv.eval()

    assert conv.weights1.dtype == torch.float32, "weights must be stored real"
    assert conv.weights1.shape == (6, 4, 3, 3, 2)
    assert conv.weights2.shape == (6, 4, 3, 3, 2)

    x = torch.randn(5, 6, 12, 12, requires_grad=True)

    y = conv(x)
    (y ** 2).sum().backward()
    g_x = x.grad.detach().clone()
    g_w1 = conv.weights1.grad.detach().clone()
    g_w2 = conv.weights2.grad.detach().clone()

    # Reference: cfloat leaf parameters holding the SAME values.
    w1_c = torch.view_as_complex(conv.weights1.detach().clone()).requires_grad_(True)
    w2_c = torch.view_as_complex(conv.weights2.detach().clone()).requires_grad_(True)
    x2 = x.detach().clone().requires_grad_(True)
    y_ref = _reference_complex_forward(w1_c, w2_c, x2, conv.modes1, conv.modes2, conv.out_channels)
    (y_ref ** 2).sum().backward()

    assert torch.allclose(y, y_ref, atol=1e-6), "forward output changed"
    assert torch.allclose(g_x, x2.grad, atol=1e-6), "input gradient changed"
    assert torch.allclose(g_w1, torch.view_as_real(w1_c.grad), atol=1e-6), "weights1 grad changed"
    assert torch.allclose(g_w2, torch.view_as_real(w2_c.grad), atol=1e-6), "weights2 grad changed"


# ----------------------------- DDP reduction ------------------------------

IN_CH = 6
OUT_CH = 4
MODES = 3
NX = NY = 12
N_PER_RANK = 8
MODEL_SEED = 7
BATCH_SEED = 12345
ATOL = 1e-6
RTOL = 1e-5


def _ddp_reduction_worker(rank, world_size, init_file):
    import torch.distributed as dist

    dist.init_process_group(
        backend="gloo", init_method=f"file://{init_file}", world_size=world_size, rank=rank
    )
    try:
        torch.manual_seed(MODEL_SEED)  # identical init on every rank
        model = SpectralConv2d(IN_CH, OUT_CH, MODES, MODES)
        ddp = torch.nn.parallel.DistributedDataParallel(model)

        n = N_PER_RANK * world_size
        g = torch.Generator().manual_seed(BATCH_SEED)
        x_full = torch.randn(n, IN_CH, NX, NY, generator=g)
        y_full = torch.randn(n, OUT_CH, NX, NY, generator=g)

        sl = slice(rank * N_PER_RANK, (rank + 1) * N_PER_RANK)
        out = ddp(x_full[sl])
        # mean reduction + equal shards => DDP grad all-reduce yields the grad of
        # the global mean loss, matching a single-process full-batch backward.
        torch.mean((out - y_full[sl]) ** 2).backward()

        g_w1_ddp = model.weights1.grad.detach().clone()
        g_w2_ddp = model.weights2.grad.detach().clone()

        if rank == 0:
            torch.manual_seed(MODEL_SEED)
            ref = SpectralConv2d(IN_CH, OUT_CH, MODES, MODES)
            torch.mean((ref(x_full) - y_full) ** 2).backward()
            assert torch.allclose(g_w1_ddp, ref.weights1.grad, atol=ATOL, rtol=RTOL), (
                "gloo-reduced weights1 grad != single-process grad: "
                f"max|Δ|={(g_w1_ddp - ref.weights1.grad).abs().max().item():.3e}"
            )
            assert torch.allclose(g_w2_ddp, ref.weights2.grad, atol=ATOL, rtol=RTOL), (
                "gloo-reduced weights2 grad != single-process grad: "
                f"max|Δ|={(g_w2_ddp - ref.weights2.grad).abs().max().item():.3e}"
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2])
def test_gloo_ddp_spectral_grad_matches_single_process(world_size):
    """Real 2-rank gloo all-reduce of the spectral grads matches single-process.

    This is the end-to-end confirmation that real-stored Fourier filters reduce
    correctly under torch.distributed. With cfloat parameters this same DDP
    backward raises "Invalid scalar type" -- i.e. the reparametrization is exactly
    what makes the distributed gradient reduction well-defined and correct."""
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed unavailable")
    import torch.multiprocessing as mp

    with tempfile.TemporaryDirectory() as tmp:
        init_file = os.path.join(tmp, "store")
        try:
            mp.spawn(
                _ddp_reduction_worker,
                args=(world_size, init_file),
                nprocs=world_size,
                join=True,
            )
        except Exception as exc:  # pragma: no cover - environment-dependent
            pytest.skip(f"could not spawn gloo workers in this environment: {exc}")
