import inspect

import numpy as np
import pytest
import torch

from problems.interfaces import (
    build_cond_vector,
    normalize_interface_scalars,
    INTERFACE_X_RANGE,
    RC_RANGE,
)
from src.operators.cvit import CViT, ForcingCViT, InterfaceCViT
from src.operators.train_pino import build_cvit

"""
Stage 3 deterministic architecture / wiring tests for `InterfaceCViT` (§2/§5).
These are exact, seed-free assertions on how the three token streams are wired
(spatial field / sampled forcing waveform / interface scalars) and on the
function-value information contract `(x_Gamma, R_c, a(.)) -> T` — NOT claims about
learned behavior. They pin:
  - the model consumes only `(u_spatial, forcing_seq, params=[x_hat, Rc_hat])`,
    never the raw generator parameters;
  - forcing / parameter tokens depend deterministically on their inputs and are
    on the autograd path;
  - `x_hat`/`Rc_hat` share the linear `build_cond_vector` normalization;
  - the untouched `CViT`/`ForcingCViT` classes still construct and run.
"""

GRID = (20, 20)
EMB = 32


def _model(**kw):
    defaults = dict(
        spatial_in_ch=3, emb_dim=EMB, patch_size=10, grid_size=GRID,
        depth_enc=1, depth_dec=1, num_heads=4, t_final=0.2,
        num_forcing_tokens=1, num_param_tokens=1,
    )
    defaults.update(kw)
    m = InterfaceCViT(**defaults)
    return m.eval()


def _inputs(B=2, Nq=7, seed=0):
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(B, 3, *GRID, generator=g)
    fseq = torch.randn(B, 128, 2, generator=g)
    params = torch.rand(B, 2, generator=g)
    coords = torch.rand(B, Nq, 2, generator=g)
    t = torch.rand(B, Nq, 1, generator=g) * 0.2
    return u, fseq, params, coords, t


def test_model_smoke_shapes():
    m = _model()
    u, fseq, params, coords, t = _inputs()
    out = m(u, coords, t, fseq, params)
    assert out.shape == (2, 7, 1)
    lat = m.encode(u, fseq, params)
    # 100/patch^2 spatial tokens (10x10) + 1 forcing + 1 param... grid 20x20 p10 -> 4
    assert lat.shape == (2, 4 + 1 + 1, EMB)


def test_encode_signature_excludes_generator_params():
    """The function-value contract: `encode` takes only the spatial field, the
    sampled waveform, and the two interface scalars — there is no argument path
    for amplitude/frequency/phase."""
    sig = list(inspect.signature(InterfaceCViT.encode).parameters)
    assert sig == ["self", "u_spatial", "forcing_seq", "params"]
    for bad in ("amplitude", "frequency", "phase", "temporal_params"):
        assert bad not in sig


def test_forcing_tokens_track_sampled_sequence():
    m = _model()
    u, fseq, params, _, _ = _inputs()
    zf_a = m.forcing_encoder(fseq)
    zf_b = m.forcing_encoder(fseq.clone())
    assert torch.equal(zf_a, zf_b)  # identical seq -> bitwise identical token
    fseq2 = fseq.clone()
    fseq2[0, :, 1] += 1.0  # different waveform for sim 0 only
    zf_c = m.forcing_encoder(fseq2)
    assert not torch.allclose(zf_a[0], zf_c[0])  # sim 0 token changed
    assert torch.equal(zf_a[1], zf_c[1])         # sim 1 untouched


def test_param_token_tracks_scalars():
    m = _model()
    p = torch.rand(3, 2)
    zp = m.param_encoder(p)
    assert torch.equal(zp, m.param_encoder(p.clone()))
    p2 = p.clone()
    p2[0, 1] += 0.3  # change Rc_hat of sim 0
    zp2 = m.param_encoder(p2)
    assert not torch.allclose(zp[0], zp2[0])
    assert torch.equal(zp[1:], zp2[1:])


def test_gradients_reach_forcing_and_param_branches():
    m = _model().train()
    u, fseq, params, coords, t = _inputs()
    out = m(u, coords, t, fseq, params)
    out.sum().backward()
    fp = list(m.forcing_encoder.parameters())
    pp = list(m.param_encoder.parameters())
    assert all(p.grad is not None for p in fp)
    assert all(p.grad is not None for p in pp)
    assert m.modality.grad is not None


def test_double_backward_through_decode():
    """The PDE residual differentiates the decoder twice w.r.t. coords with
    create_graph=True; the explicit-attention decoder must survive it."""
    m = _model()
    u, fseq, params, _, _ = _inputs()
    lat = m.encode(u, fseq, params)
    B = u.shape[0]
    x = torch.rand(B, 5, 1, requires_grad=True)
    y = torch.rand(B, 5, 1, requires_grad=True)
    t = torch.rand(B, 5, 1, requires_grad=True) * 0.2
    T = m.decode(lat, torch.cat([x, y], dim=-1), t)
    T_x = torch.autograd.grad(T, x, torch.ones_like(T), create_graph=True)[0]
    T_xx = torch.autograd.grad(T_x, x, torch.ones_like(T_x), create_graph=True)[0]
    assert T_xx.shape == (B, 5, 1)
    (T_xx ** 2).mean().backward()  # third backward into params must not raise


def test_hard_right_dirichlet_exact_at_x1():
    m = _model()
    u, fseq, params, coords, t = _inputs()
    lat = m.encode(u, fseq, params)
    coords1 = coords.clone()
    coords1[..., 0] = 1.0  # x = 1 wall
    out = m.decode(lat, coords1, t)
    assert float(out.abs().max()) == pytest.approx(0.0, abs=1e-6)  # t_right_tilde=0


def test_encode_once_decode_twice_matches_single_shot():
    m = _model()
    u, fseq, params, coords, t = _inputs()
    lat = m.encode(u, fseq, params)
    d = m.decode(lat, coords, t)
    f = m(u, coords, t, fseq, params)
    assert torch.allclose(d, f, atol=1e-6)


def test_batch_permutation_equivariance():
    m = _model()
    u, fseq, params, coords, t = _inputs(B=3)
    out = m(u, coords, t, fseq, params)
    perm = torch.tensor([2, 0, 1])
    out_p = m(u[perm], coords[perm], t[perm], fseq[perm], params[perm])
    assert torch.allclose(out_p, out[perm], atol=1e-6)


def test_changing_one_sim_forcing_isolates_to_that_sim():
    m = _model()
    u, fseq, params, coords, t = _inputs(B=3)
    out = m(u, coords, t, fseq, params)
    fseq2 = fseq.clone()
    fseq2[1, :, 1] += 2.0  # perturb only sim 1's waveform
    out2 = m(u, coords, t, fseq2, params)
    assert not torch.allclose(out[1], out2[1])           # sim 1 responds
    assert torch.allclose(out[0], out2[0], atol=1e-6)    # sims 0, 2 unchanged
    assert torch.allclose(out[2], out2[2], atol=1e-6)


@pytest.mark.parametrize(
    "ix,rc",
    [(0.2, 0.05), (0.5, 0.5), (0.8, 1.0), (0.35, 0.7)],
)
def test_scalar_normalization_matches_cond_vector(ix, rc):
    """`normalize_interface_scalars` reproduces the R_c / interface_x entries of
    `build_cond_vector` (indices 2, 3)."""
    xhat, rchat = normalize_interface_scalars(ix, rc)
    cv = build_cond_vector(0.0, 0.0, rc, ix)
    assert rchat == pytest.approx(cv[2], abs=1e-6)  # R_c_norm
    assert xhat == pytest.approx(cv[3], abs=1e-6)   # interface_x_norm
    assert 0.0 - 1e-6 <= xhat <= 1.0 + 1e-6
    assert 0.0 - 1e-6 <= rchat <= 1.0 + 1e-6


def test_normalize_interface_scalars_vectorized():
    ix = np.array([0.2, 0.5, 0.8], dtype=np.float64)  # over (0.2, 0.8)
    rc = np.array([0.05, 1.0], dtype=np.float64)       # endpoints of (0.05, 1.0)
    out_x = normalize_interface_scalars(ix, np.full_like(ix, 0.05))
    assert out_x.shape == (3, 2)
    np.testing.assert_allclose(out_x[:, 0], [0.0, 0.5, 1.0], atol=1e-6)
    out_rc = normalize_interface_scalars(np.full_like(rc, 0.5), rc)
    np.testing.assert_allclose(out_rc[:, 1], [0.0, 1.0], atol=1e-6)


def test_build_cvit_interfaces_variant():
    config = {
        "model": {
            "cvit": {
                "emb_dim": EMB, "patch_size": 10, "depth_enc": 1, "depth_dec": 1,
                "num_heads": 4, "mlp_ratio": 2.0,
            },
            "interface_cvit": {
                "spatial_in_ch": 3, "num_forcing_tokens": 1, "num_param_tokens": 1,
            },
        }
    }
    m = build_cvit(config, mu=300.0, sigma=10.0, grid_size=GRID, t_final=0.2,
                   variant="interfaces")
    assert isinstance(m, InterfaceCViT)
    u, fseq, params, coords, t = _inputs()
    out = m(u, coords, t, fseq, params)
    assert out.shape == (2, 7, 1)


def test_untouched_cvit_forcingcvit_still_build():
    """`InterfaceCViT` is a separate class; the legacy paths are unchanged."""
    c = CViT(in_ch=1, emb_dim=EMB, patch_size=10, grid_size=GRID,
             depth_enc=1, depth_dec=1, num_heads=4)
    f = ForcingCViT(in_ch=1, emb_dim=EMB, patch_size=10, grid_size=GRID,
                    depth_enc=1, depth_dec=1, num_heads=4)
    assert hasattr(c, "encoder") and hasattr(c, "decoder")
    assert isinstance(f, CViT)  # ForcingCViT subclasses CViT
    assert not isinstance(InterfaceCViT(spatial_in_ch=3, emb_dim=EMB,
                                        patch_size=10, grid_size=GRID,
                                        depth_enc=1, depth_dec=1, num_heads=4), CViT)
