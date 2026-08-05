"""Tests for the vendored SOAP optimizer (Step 6) and its build_optimizer wiring.

Covers:
  - build_optimizer maps "SOAP" -> SOAP with the configured knobs; unknown names raise.
  - ~30 steps on a tiny FNO2d strictly decrease a synthetic regression loss and keep
    params finite (exercises the 5-D spectral weights past precondition_frequency).
  - optimizer state_dict round-trip reproduces identical params on CPU.
  - _validate_resume_compatibility raises on an AdamW -> SOAP mismatch.
  - a dedicated multi-rank gloo test: params match across ranks after several
    preconditioner updates, losses finite, SOAP state tensors finite.
"""

from __future__ import annotations

import io
import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from src.operators.fno2d import FNO2d
from src.operators.soap import SOAP
from src.operators.train import build_optimizer, _validate_resume_compatibility

FORCING_IN_CHANNELS = 4
FORCING_COND_STATIC_DIM = 10
FORCING_TEMPORAL_TOKEN_DIM = 2


def _tiny_fno2d() -> FNO2d:
    return FNO2d(
        modes1=2,
        modes2=2,
        width=8,
        in_channels=FORCING_IN_CHANNELS,
        out_channels=1,
        n_layers=2,
        cond_static_dim=FORCING_COND_STATIC_DIM,
        temporal_token_dim=FORCING_TEMPORAL_TOKEN_DIM,
        temporal_hidden=16,
        forcing_embed_dim=16,
        use_forcing_time_aug=True,
        s_y_channel=3,
    )


def _tiny_inputs(batch: int = 3, n: int = 8, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    spatial = torch.randn(batch, n, n, FORCING_IN_CHANNELS, generator=g)
    cond_static = torch.randn(batch, FORCING_COND_STATIC_DIM, generator=g)
    forcing_seq = torch.randn(batch, 128, FORCING_TEMPORAL_TOKEN_DIM, generator=g)
    target = torch.randn(batch, n, n, 1, generator=g)
    return spatial, cond_static, forcing_seq, target


def _base_config(optimizer: str = "SOAP") -> dict:
    return {
        "training": {
            "optimizer": optimizer,
            "learning_rate": 2.0e-3,
            "weight_decay": 0.0,
            "soap": {
                "betas": [0.95, 0.95],
                "shampoo_beta": 0.95,
                "eps": 1.0e-8,
                "precondition_frequency": 5,
                "max_precond_dim": 10000,
                "merge_dims": False,
                "precondition_1d": False,
            },
        }
    }


# ---------- build_optimizer wiring ----------

def test_build_optimizer_maps_soap():
    model = _tiny_fno2d()
    opt = build_optimizer(_base_config("SOAP"), model.parameters())
    assert isinstance(opt, SOAP)
    group = opt.param_groups[0]
    assert group["lr"] == pytest.approx(2.0e-3)
    assert group["betas"] == (0.95, 0.95)
    assert group["shampoo_beta"] == pytest.approx(0.95)
    assert group["precondition_frequency"] == 5
    assert group["precondition_1d"] is False


def test_build_optimizer_unknown_name_raises():
    model = _tiny_fno2d()
    with pytest.raises(ValueError, match="Unsupported optimizer"):
        build_optimizer(
            {"training": {"optimizer": "Nope", "learning_rate": 1e-3, "weight_decay": 0.0}},
            model.parameters(),
        )


def test_build_optimizer_soap_defaults_when_block_absent():
    model = _tiny_fno2d()
    cfg = {"training": {"optimizer": "SOAP", "learning_rate": 1e-3, "weight_decay": 0.0}}
    opt = build_optimizer(cfg, model.parameters())
    assert isinstance(opt, SOAP)
    assert opt.param_groups[0]["precondition_frequency"] == 10


# ---------- optimization behavior ----------

def test_soap_decreases_loss_on_tiny_fno2d():
    torch.manual_seed(0)
    model = _tiny_fno2d()
    opt = build_optimizer(_base_config("SOAP"), model.parameters())
    spatial, cond_static, forcing_seq, target = _tiny_inputs()

    def _loss():
        pred = model(spatial, cond_static, forcing_seq)
        return (pred - target).pow(2).mean()

    first = _loss().item()
    for _ in range(30):  # > precondition_frequency=5, exercises the QR update path
        opt.zero_grad()
        loss = _loss()
        loss.backward()
        opt.step()
    last = _loss().item()

    assert last < first
    for p in model.parameters():
        assert torch.isfinite(p).all()


def test_soap_state_dict_round_trip_reproduces_params():
    torch.manual_seed(1)
    model_a = _tiny_fno2d()
    opt_a = build_optimizer(_base_config("SOAP"), model_a.parameters())
    spatial, cond_static, forcing_seq, target = _tiny_inputs(seed=2)

    def _step(model, opt):
        opt.zero_grad()
        loss = (model(spatial, cond_static, forcing_seq) - target).pow(2).mean()
        loss.backward()
        opt.step()

    for _ in range(8):  # past the first preconditioner update
        _step(model_a, opt_a)

    # Round-trip through a byte buffer, exactly like the checkpoint resume path
    # (torch.save -> torch.load). Loading a *live* state_dict directly would alias
    # the source optimizer's tensors (standard torch.optim behavior), so serialize.
    model_buf = io.BytesIO()
    torch.save(model_a.state_dict(), model_buf)
    opt_buf = io.BytesIO()
    torch.save(opt_a.state_dict(), opt_buf)

    model_b = _tiny_fno2d()
    model_buf.seek(0)
    model_b.load_state_dict(torch.load(model_buf, weights_only=False))
    opt_b = build_optimizer(_base_config("SOAP"), model_b.parameters())
    opt_buf.seek(0)
    opt_b.load_state_dict(torch.load(opt_buf, weights_only=False))

    for _ in range(6):
        _step(model_a, opt_a)
        _step(model_b, opt_b)

    for pa, pb in zip(model_a.parameters(), model_b.parameters()):
        assert torch.allclose(pa, pb, atol=1e-6)


# ---------- resume compatibility ----------

def test_resume_compat_raises_on_adamw_to_soap():
    ckpt_conf = {"training": {"optimizer": "AdamW", "scheduler": {"type": "StepLR"}}}
    curr_conf = {"training": {"optimizer": "SOAP", "scheduler": {"type": "StepLR"}}}
    with pytest.raises(ValueError, match="Incompatible resume state"):
        _validate_resume_compatibility(ckpt_conf, curr_conf)


# ---------- multi-rank gloo ----------

_GLOO_OK = dist.is_available() and dist.is_gloo_available()


class _TinyReal(nn.Module):
    """Real-valued model with a 5-D parameter so SOAP's N-D preconditioner path
    (analogous to the FNO2d spectral weights) is exercised across ranks."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.spectral = nn.Parameter(torch.randn(2, 2, 2, 2, 2) * 0.1)

    def forward(self, x):
        y = self.lin(x)
        scale = self.spectral.reshape(-1).sum()
        return y * (1.0 + 0.0 * scale) + scale


def _soap_worker(rank, world_size, port, result_queue):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        "gloo", rank=rank, world_size=world_size, timeout=timedelta(seconds=30)
    )
    try:
        torch.manual_seed(0)  # identical init on every rank
        model = _TinyReal()
        opt = SOAP(
            model.parameters(),
            lr=1e-3,
            weight_decay=0.0,
            betas=(0.95, 0.95),
            shampoo_beta=0.95,
            precondition_frequency=5,
        )
        x = torch.randn(8, 4) + rank  # per-rank shard
        target = torch.zeros(8, 4)

        losses = []
        for _ in range(15):  # >= 15 steps, several preconditioner updates
            opt.zero_grad()
            out = model(x)
            loss = (out - target).pow(2).mean()
            loss.backward()
            # Manually average grads across ranks (no DDP wrapper needed here).
            for p in model.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                    p.grad /= world_size
            opt.step()
            losses.append(loss.item())

        finite_loss = all(torch.isfinite(torch.tensor(l)) for l in losses)
        params_finite = all(bool(torch.isfinite(p).all()) for p in model.parameters())

        # Params must match rank 0 (identical init + averaged grads => lockstep).
        params_match = True
        for p in model.parameters():
            ref = p.detach().clone()
            dist.broadcast(ref, src=0)
            if not torch.allclose(p.detach(), ref, atol=1e-6):
                params_match = False
                break

        state_finite = True
        for st in opt.state.values():
            for key in ("exp_avg", "exp_avg_sq"):
                if key in st and not bool(torch.isfinite(st[key]).all()):
                    state_finite = False
        result_queue.put(
            (rank, "ok", bool(finite_loss and params_finite and params_match and state_finite))
        )
    except Exception as exc:  # pragma: no cover - surfaced via queue
        result_queue.put((rank, "error", str(exc)))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not _GLOO_OK, reason="gloo backend unavailable")
def test_soap_multirank_gloo_consistency():
    world_size = 2
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [
        ctx.Process(target=_soap_worker, args=(r, world_size, 29553, q))
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
                pytest.fail("SOAP multi-rank worker hung")
    assert len(results) == world_size
    for rank, status, ok in results:
        assert status == "ok", f"rank {rank} failed: {ok}"
        assert ok, f"rank {rank} SOAP state/params not consistent across ranks"
