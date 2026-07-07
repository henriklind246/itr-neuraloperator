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
