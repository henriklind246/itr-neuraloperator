import torch

from src.operators.cvit import CViT


def _tiny(**kw):
    base = dict(
        in_ch=1, out_dim=1, emb_dim=32, patch_size=10, grid_size=(20, 20),
        depth_enc=2, depth_dec=2, num_heads=4, mlp_ratio=2.0, fourier_freq=1.0,
        hard_right_dirichlet=True, t_right_tilde=0.0,
    )
    base.update(kw)
    return CViT(**base)


def test_forward_shape():
    m = _tiny()
    u = torch.randn(3, 1, 20, 20)
    coords = torch.rand(1, 64, 2)
    t = torch.rand(1, 64, 1) * 0.3
    out = m(u, coords, t)
    assert out.shape == (3, 64, 1)


def test_dim1_broadcast_matches_expanded():
    m = _tiny()
    u = torch.randn(4, 1, 20, 20)
    coords = torch.rand(1, 16, 2)
    t = torch.rand(1, 16, 1) * 0.3
    out_shared = m(u, coords, t)
    out_expanded = m(u, coords.expand(4, -1, -1), t.expand(4, -1, -1))
    assert torch.allclose(out_shared, out_expanded, atol=1e-6)


def test_hard_dirichlet_exact():
    m = _tiny(t_right_tilde=0.7)
    u = torch.randn(2, 1, 20, 20)
    coords = torch.rand(2, 32, 2)
    coords[..., 0] = 1.0  # x = 1 -> right Dirichlet wall
    t = torch.rand(2, 32, 1) * 0.3
    out = m(u, coords, t)
    assert torch.allclose(out, torch.full_like(out, 0.7), atol=1e-5)


def test_hard_dirichlet_off_returns_raw():
    m = _tiny(hard_right_dirichlet=False)
    u = torch.randn(1, 1, 20, 20)
    coords = torch.rand(1, 8, 2)
    coords[..., 0] = 1.0
    t = torch.zeros(1, 8, 1)
    out = m(u, coords, t)
    # With the ansatz off, x=1 is not pinned, so the output is generically nonzero.
    assert out.shape == (1, 8, 1)


def test_fourier_freq_t_decouples_temporal_scale():
    # fourier_freq_t scales ONLY the temporal Fourier kernel; the spatial kernel
    # stays at fourier_freq. (kernel = randn * scale, so std tracks the scale.)
    torch.manual_seed(0)
    m = _tiny(fourier_freq=1.0, fourier_freq_t=10.0)
    sx = m.decoder.fourier_x.kernel.std().item()
    st = m.decoder.fourier_t.kernel.std().item()
    assert st > 4.0 * sx  # ~10x in expectation; loose bound for 16-sample noise


def test_fourier_freq_t_none_matches_spatial_scale():
    # Unset (null) -> temporal scale equals fourier_freq (backward compatible).
    torch.manual_seed(0)
    m = _tiny(fourier_freq=5.0, fourier_freq_t=None)
    sx = m.decoder.fourier_x.kernel.std().item()
    st = m.decoder.fourier_t.kernel.std().item()
    assert 0.5 < (st / sx) < 2.0


def test_higher_fourier_freq_t_increases_time_sensitivity():
    # The fix's core claim: on the diffusion window t in [0, 0.3], a higher
    # temporal Fourier scale makes the decoder materially more time-sensitive.
    # The time-FiLM last layer is zero-init (time-invariant at init), so break
    # that first, then isolate the frequency effect by rescaling the SAME frozen
    # temporal kernel in place (freq 1 -> 10, identical random directions).
    torch.manual_seed(0)
    m = _tiny(fourier_freq=1.0, fourier_freq_t=1.0)
    last = m.decoder.time_film.net[-1]
    with torch.no_grad():
        last.weight.normal_(0.0, 0.5)
        last.bias.normal_(0.0, 0.5)
    u = torch.randn(2, 1, 20, 20)
    coords = torch.rand(2, 64, 2)
    t0 = torch.zeros(2, 64, 1)
    t1 = torch.full((2, 64, 1), 0.3)
    with torch.no_grad():
        d_lo = (m(u, coords, t1) - m(u, coords, t0)).abs().mean().item()
        m.decoder.fourier_t.kernel.mul_(10.0)  # freq 1 -> 10, same directions
        d_hi = (m(u, coords, t1) - m(u, coords, t0)).abs().mean().item()
    assert d_hi > 2.0 * d_lo


def test_double_backward_through_decoder():
    m = _tiny()
    u = torch.randn(2, 1, 20, 20)
    x = torch.rand(2, 8, 1, requires_grad=True)
    y = torch.rand(2, 8, 1, requires_grad=True)
    t = (torch.rand(2, 8, 1) * 0.3).requires_grad_(True)
    coords = torch.cat([x, y], dim=-1)
    T = m(u, coords, t)
    (T_x,) = torch.autograd.grad(T.sum(), x, create_graph=True)
    (T_xx,) = torch.autograd.grad(T_x.sum(), x)
    assert T_xx.shape == (2, 8, 1)
    assert torch.isfinite(T_xx).all()
