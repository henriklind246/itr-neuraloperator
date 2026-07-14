import inspect

import pytest
import torch

from src.operators.cvit import CViT, ForcingCViT, ForcingICCViT, InterfaceCViT
from src.operators.train_pino import build_cvit

"""
Deterministic architecture / wiring tests for `ForcingICCViT`, the two-branch
token-fusion surrogate for `diffusion_forcing_single` with a VARYING initial
condition. These are exact, seed-free assertions on how the two token streams
(forcing space-time image / IC field) are wired and on the function-value
contract `(q_L(y,t), T0_tilde) -> T` — NOT claims about learned behavior. They
pin:
  - the model consumes only `(u_forcing, u_ic)` at encode time (no raw params);
  - both encoder branches sit on the autograd path and the learned 2-way
    modality embedding is trained;
  - the decoder survives the PDE residual's double-differentiation w.r.t. coords;
  - the hard right-Dirichlet ansatz keeps the x=1 wall exact;
  - decode does not re-scale its output — decoded space, IC encoder input, and
    the IC loss target share one normalized space (frozen sigma_global);
  - the untouched `CViT`/`ForcingCViT`/`InterfaceCViT` classes still build.
"""

IC_GRID = (20, 20)          # (Nx, Ny); ic_patch_size 10 -> 2x2 = 4 IC tokens
FORCING_GRID = (16, 32)     # (Ny_img, Nt_img); patch 8 -> 2x4 = 8 forcing tokens
EMB = 32
N_FORCING_TOK = 8
N_IC_TOK = 4


def _model(**kw):
    defaults = dict(
        forcing_in_ch=1, ic_in_ch=1, emb_dim=EMB,
        ic_patch_size=10, ic_grid_size=IC_GRID,
        forcing_patch_size=8, forcing_grid_size=FORCING_GRID,
        depth_enc=1, depth_dec=1, num_heads=4, t_final=0.2,
    )
    defaults.update(kw)
    m = ForcingICCViT(**defaults)
    return m.eval()


def _inputs(B=2, Nq=7, seed=0):
    g = torch.Generator().manual_seed(seed)
    u_forcing = torch.randn(B, 1, *FORCING_GRID, generator=g)
    u_ic = torch.randn(B, 1, *IC_GRID, generator=g)
    coords = torch.rand(B, Nq, 2, generator=g)
    t = torch.rand(B, Nq, 1, generator=g) * 0.2
    return u_forcing, u_ic, coords, t


def test_model_smoke_shapes_and_token_counts():
    m = _model()
    u_forcing, u_ic, coords, t = _inputs()
    out = m(u_forcing, u_ic, coords, t)
    assert out.shape == (2, 7, 1)
    lat = m.encode(u_forcing, u_ic)
    assert m.num_forcing_tokens == N_FORCING_TOK
    assert m.num_ic_tokens == N_IC_TOK
    assert lat.shape == (2, N_FORCING_TOK + N_IC_TOK, EMB)


def test_encode_signature_is_two_branch_only():
    """The function-value contract: `encode` takes only the forcing image and
    the IC field — no argument path for amplitude/frequency/interface params."""
    sig = list(inspect.signature(ForcingICCViT.encode).parameters)
    assert sig == ["self", "u_forcing", "u_ic"]
    for bad in ("amplitude", "frequency", "phase", "params", "interface_x", "q_left"):
        assert bad not in sig
    # decode is a pure coord+time query with no q_left (left wall stays soft).
    dsig = list(inspect.signature(ForcingICCViT.decode).parameters)
    assert dsig == ["self", "latent", "coords", "t"]
    assert "q_left" not in dsig


def test_forcing_tokens_track_image():
    m = _model()
    u_forcing, _, _, _ = _inputs()
    zf_a = m.forcing_encoder(u_forcing)
    assert torch.equal(zf_a, m.forcing_encoder(u_forcing.clone()))
    image2 = u_forcing.clone()
    image2[0] += 1.0
    zf_c = m.forcing_encoder(image2)
    assert not torch.allclose(zf_a[0], zf_c[0])  # sim 0 token changed
    assert torch.equal(zf_a[1], zf_c[1])         # sim 1 untouched


def test_ic_tokens_track_field():
    m = _model()
    _, u_ic, _, _ = _inputs()
    zi_a = m.ic_encoder(u_ic)
    assert torch.equal(zi_a, m.ic_encoder(u_ic.clone()))
    ic2 = u_ic.clone()
    ic2[1] += 0.5
    zi_c = m.ic_encoder(ic2)
    assert not torch.allclose(zi_a[1], zi_c[1])  # sim 1 token changed
    assert torch.equal(zi_a[0], zi_c[0])         # sim 0 untouched


def test_gradients_reach_both_encoder_branches_and_modality():
    m = _model().train()
    u_forcing, u_ic, coords, t = _inputs()
    out = m(u_forcing, u_ic, coords, t)
    out.sum().backward()
    fp = list(m.forcing_encoder.parameters())
    ip = list(m.ic_encoder.parameters())
    assert all(p.grad is not None for p in fp)
    assert all(p.grad is not None for p in ip)
    assert any(p.grad.abs().sum() > 0 for p in fp)
    assert any(p.grad.abs().sum() > 0 for p in ip)
    assert m.modality.grad is not None
    assert m.modality.grad.abs().sum() > 0  # both modality rows are used


def test_double_backward_through_decode():
    """The PDE residual differentiates the decoder twice w.r.t. coords with
    create_graph=True (T_xx, T_yy). The explicit-attention decoder over the
    fused latent must survive it and still backprop into params."""
    m = _model()
    u_forcing, u_ic, _, _ = _inputs()
    lat = m.encode(u_forcing, u_ic)
    B = u_forcing.shape[0]
    x = torch.rand(B, 5, 1, requires_grad=True)
    y = torch.rand(B, 5, 1, requires_grad=True)
    t = torch.rand(B, 5, 1, requires_grad=True) * 0.2
    T = m.decode(lat, torch.cat([x, y], dim=-1), t)
    T_x = torch.autograd.grad(T, x, torch.ones_like(T), create_graph=True)[0]
    T_xx = torch.autograd.grad(T_x, x, torch.ones_like(T_x), create_graph=True)[0]
    T_y = torch.autograd.grad(T, y, torch.ones_like(T), create_graph=True)[0]
    T_yy = torch.autograd.grad(T_y, y, torch.ones_like(T_y), create_graph=True)[0]
    assert T_xx.shape == (B, 5, 1)
    assert T_yy.shape == (B, 5, 1)
    (T_xx ** 2 + T_yy ** 2).mean().backward()  # third backward into params


def test_shared_latent_reuse_across_decodes_no_freed_graph():
    """The runner encodes once and calls the decoder for interior, BC, and IC
    query sets against the SAME latent. Reusing one latent across multiple
    decode+grad calls (graph retained) must not raise a freed-graph error."""
    m = _model().train()
    u_forcing, u_ic, _, _ = _inputs()
    lat = m.encode(u_forcing, u_ic)
    B = u_forcing.shape[0]
    total = 0.0
    for k in range(3):
        x = torch.rand(B, 4, 1, requires_grad=True)
        coords = torch.cat([x, torch.rand(B, 4, 1)], dim=-1)
        t = torch.rand(B, 4, 1, requires_grad=True) * 0.2
        T = m.decode(lat, coords, t)
        T_x = torch.autograd.grad(T, x, torch.ones_like(T), create_graph=True)[0]
        total = total + (T_x ** 2).mean() + T.mean()
    total.backward()  # single backward through the shared-latent subgraph
    assert all(
        p.grad is not None for p in m.forcing_encoder.parameters()
    )
    assert all(p.grad is not None for p in m.ic_encoder.parameters())


def test_hard_right_dirichlet_exact_at_x1():
    m = _model()
    u_forcing, u_ic, coords, t = _inputs()
    lat = m.encode(u_forcing, u_ic)
    coords1 = coords.clone()
    coords1[..., 0] = 1.0  # x = 1 wall
    out = m.decode(lat, coords1, t)
    assert float(out.abs().max()) == pytest.approx(0.0, abs=1e-6)  # t_right_tilde=0


def test_hard_right_dirichlet_uses_normalized_t_right():
    """The x=1 clamp is applied in the SAME normalized space as the IC / output;
    t_right_tilde is a scalar buffer, not a hidden re-scaling of the field."""
    m = _model(t_right_tilde=1.5)
    u_forcing, u_ic, coords, t = _inputs()
    lat = m.encode(u_forcing, u_ic)
    coords1 = coords.clone()
    coords1[..., 0] = 1.0
    out = m.decode(lat, coords1, t)
    assert float(m.t_right_tilde) == pytest.approx(1.5)
    assert torch.allclose(out, torch.full_like(out, 1.5), atol=1e-6)


def test_soft_decode_returns_unscaled_raw_output():
    """With the ansatz off, decode returns the raw decoder output verbatim (no
    extra scaling) — decoded space == IC encoder input space == IC target space,
    so the caller's frozen sigma_global normalization is the only scale."""
    m = _model(hard_right_dirichlet=False)
    u_forcing, u_ic, coords, t = _inputs()
    lat = m.encode(u_forcing, u_ic)
    raw = m.decoder(lat, coords, t / m.t_norm)
    out = m.decode(lat, coords, t)
    assert torch.allclose(out, raw, atol=0.0, rtol=0.0)


def test_encode_once_decode_matches_single_shot():
    m = _model()
    u_forcing, u_ic, coords, t = _inputs()
    lat = m.encode(u_forcing, u_ic)
    d = m.decode(lat, coords, t)
    f = m(u_forcing, u_ic, coords, t)
    assert torch.allclose(d, f, atol=1e-6)


def test_batch_permutation_equivariance():
    m = _model()
    u_forcing, u_ic, coords, t = _inputs(B=3)
    out = m(u_forcing, u_ic, coords, t)
    perm = torch.tensor([2, 0, 1])
    out_p = m(u_forcing[perm], u_ic[perm], coords[perm], t[perm])
    assert torch.allclose(out_p, out[perm], atol=1e-6)


def test_changing_one_sim_forcing_isolates_to_that_sim():
    m = _model()
    u_forcing, u_ic, coords, t = _inputs(B=3)
    out = m(u_forcing, u_ic, coords, t)
    image2 = u_forcing.clone()
    image2[1] += 2.0
    out2 = m(image2, u_ic, coords, t)
    assert not torch.allclose(out[1], out2[1])           # sim 1 responds
    assert torch.allclose(out[0], out2[0], atol=1e-6)    # sims 0, 2 unchanged
    assert torch.allclose(out[2], out2[2], atol=1e-6)


def test_changing_one_sim_ic_isolates_to_that_sim():
    m = _model()
    u_forcing, u_ic, coords, t = _inputs(B=3)
    out = m(u_forcing, u_ic, coords, t)
    ic2 = u_ic.clone()
    ic2[2] += 2.0
    out2 = m(u_forcing, ic2, coords, t)
    assert not torch.allclose(out[2], out2[2])           # sim 2 responds to IC
    assert torch.allclose(out[0], out2[0], atol=1e-6)    # sims 0, 1 unchanged
    assert torch.allclose(out[1], out2[1], atol=1e-6)


def test_modality_distinguishes_streams():
    """The 2-way modality embedding is added per token before concatenation, so
    the forcing tokens and IC tokens carry different learned offsets."""
    m = _model()
    assert m.modality.shape == (2, EMB)
    assert not torch.allclose(m.modality[0], m.modality[1])


def test_production_token_contract_384_plus_100():
    m = ForcingICCViT(
        emb_dim=16, ic_patch_size=10, ic_grid_size=(100, 100),
        forcing_patch_size=8, forcing_grid_size=(96, 256),
        depth_enc=0, depth_dec=1, num_heads=4,
    )
    assert m.forcing_encoder.pos_emb.shape[1] == 384    # (96/8)*(256/8)
    assert m.ic_encoder.pos_emb.shape[1] == 100         # (100/10)*(100/10)
    assert m.num_forcing_tokens == 384
    assert m.num_ic_tokens == 100


def test_build_cvit_forcing_ic_variant():
    config = {
        "model": {
            "cvit": {
                "emb_dim": EMB, "patch_size": 10, "depth_enc": 1, "depth_dec": 1,
                "num_heads": 4, "mlp_ratio": 2.0,
            },
            "forcing_ic_cvit": {
                "forcing_in_ch": 1, "ic_in_ch": 1,
                "forcing_patch_size": 8, "ic_patch_size": 10,
            },
        },
        "training": {"pino": {"forcing": {
            "ny_img": FORCING_GRID[0], "nt_img": FORCING_GRID[1], "a_ref": 300.0,
        }}},
    }
    m = build_cvit(config, mu=300.0, sigma=10.0, grid_size=IC_GRID, t_final=0.2,
                   variant="forcing_ic")
    assert isinstance(m, ForcingICCViT)
    assert not isinstance(m, CViT)  # standalone module, not a CViT subclass
    u_forcing, u_ic, coords, t = _inputs()
    out = m(u_forcing, u_ic, coords, t)
    assert out.shape == (2, 7, 1)
    # t_right_tilde comes from the frozen (mu, sigma): (300 - 300)/10 = 0.
    assert float(m.t_right_tilde) == pytest.approx(0.0, abs=1e-6)


def test_build_cvit_forcing_ic_bakes_normalized_t_right():
    """Right-wall clamp is stored in normalized space from the checkpoint's
    frozen (mu, sigma), not as a raw Kelvin value."""
    config = {
        "model": {
            "cvit": {"emb_dim": EMB, "patch_size": 10, "depth_enc": 1,
                     "depth_dec": 1, "num_heads": 4},
            "forcing_ic_cvit": {"forcing_patch_size": 8, "ic_patch_size": 10},
        },
        "training": {"pino": {"forcing": {
            "ny_img": FORCING_GRID[0], "nt_img": FORCING_GRID[1],
        }}},
    }
    m = build_cvit(config, mu=280.0, sigma=20.0, grid_size=IC_GRID, t_final=0.2,
                   variant="forcing_ic")
    assert float(m.t_right_tilde) == pytest.approx((300.0 - 280.0) / (20.0 + 1e-8), abs=1e-5)


def test_rejects_invalid_contracts():
    with pytest.raises(ValueError, match="forcing_grid_size dimensions must be divisible"):
        _model(forcing_grid_size=(15, 32))
    with pytest.raises(ValueError, match="ic_grid_size dimensions"):
        _model(ic_grid_size=(21, 20))
    with pytest.raises(ValueError, match="exactly one forcing-image channel"):
        _model(forcing_in_ch=2)
    m = _model()
    u_forcing, u_ic, _, _ = _inputs()
    with pytest.raises(ValueError, match="u_forcing must have shape"):
        m.encode(u_forcing[..., :-1], u_ic)
    with pytest.raises(ValueError, match="u_ic must have shape"):
        m.encode(u_forcing, u_ic[..., :-1])


def test_untouched_cvit_family_still_builds():
    """`ForcingICCViT` is a standalone module; the legacy classes are unchanged."""
    c = CViT(in_ch=1, emb_dim=EMB, patch_size=10, grid_size=IC_GRID,
             depth_enc=1, depth_dec=1, num_heads=4)
    f = ForcingCViT(in_ch=1, emb_dim=EMB, patch_size=10, grid_size=IC_GRID,
                    depth_enc=1, depth_dec=1, num_heads=4)
    assert hasattr(c, "encoder") and hasattr(c, "decoder")
    assert isinstance(f, CViT)  # ForcingCViT subclasses CViT
    assert not isinstance(_model(), CViT)
    assert not isinstance(_model(), InterfaceCViT)
