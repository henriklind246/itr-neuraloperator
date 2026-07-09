import torch

from src.operators.cvit import CViT, ForcingCViT
from src.physics.pde_residual import forcing_neumann_residual


def _tiny(**kw):
    base = dict(
        in_ch=1, out_dim=1, emb_dim=32, patch_size=10, grid_size=(20, 20),
        depth_enc=2, depth_dec=2, num_heads=4, mlp_ratio=2.0, fourier_freq=1.0,
        hard_right_dirichlet=True, t_right_tilde=0.0,
    )
    base.update(kw)
    return CViT(**base)


def _tiny_forcing(**kw):
    # grid_size is (Ny, Nt): the forcing space-time image resolution.
    base = dict(
        in_ch=1, out_dim=1, emb_dim=32, patch_size=4, grid_size=(16, 20),
        depth_enc=2, depth_dec=2, num_heads=4, mlp_ratio=2.0, fourier_freq=1.0,
        hard_right_dirichlet=True, t_right_tilde=0.0,
    )
    base.update(kw)
    return ForcingCViT(**base)


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


def test_t_final_normalizes_decoder_time_in_graph():
    # t is divided by t_final INSIDE forward, so a model with t_final=0.3 queried
    # at physical t must equal an unnormalized model (t_final=1.0) queried at the
    # pre-divided t/0.3. Same seed -> identical weights, isolating the scaling.
    torch.manual_seed(0)
    m_norm = _tiny(t_final=0.3)
    torch.manual_seed(0)
    m_ref = _tiny(t_final=1.0)
    u = torch.randn(2, 1, 20, 20)
    coords = torch.rand(2, 16, 2)
    t = torch.rand(2, 16, 1) * 0.3
    with torch.no_grad():
        out_norm = m_norm(u, coords, t)
        out_ref = m_ref(u, coords, t / 0.3)
    assert torch.allclose(out_norm, out_ref, atol=1e-6)


def test_t_final_default_is_noop():
    # Default t_final=1.0 divides by one: output must match an explicit-1.0 model.
    torch.manual_seed(0)
    m_default = _tiny()
    torch.manual_seed(0)
    m_one = _tiny(t_final=1.0)
    u = torch.randn(1, 1, 20, 20)
    coords = torch.rand(1, 8, 2)
    t = torch.rand(1, 8, 1) * 0.3
    with torch.no_grad():
        assert torch.allclose(m_default(u, coords, t), m_one(u, coords, t), atol=0)


def test_t_final_scales_physical_time_derivative():
    # Normalizing in-graph means autograd's dT/dt picks up the 1/t_final chain
    # factor: the physical T_t of the t_final=0.3 model is (1/0.3)x that of the
    # unnormalized model evaluated at the matching normalized time.
    torch.manual_seed(0)
    m_norm = _tiny(t_final=0.3)
    torch.manual_seed(0)
    m_ref = _tiny(t_final=1.0)
    u = torch.randn(2, 1, 20, 20)
    coords = torch.rand(2, 8, 2)
    t_phys = (torch.rand(2, 8, 1) * 0.3).requires_grad_(True)
    T = m_norm(u, coords, t_phys)
    (T_t_norm,) = torch.autograd.grad(T.sum(), t_phys)

    t_ref = (t_phys.detach() / 0.3).requires_grad_(True)
    T2 = m_ref(u, coords, t_ref)
    (T_t_ref,) = torch.autograd.grad(T2.sum(), t_ref)
    assert torch.allclose(T_t_norm, T_t_ref / 0.3, atol=1e-5)


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


def test_forcing_forward_shape():
    # Encoder input is the forcing space-time image u:(B, 1, Ny, Nt).
    m = _tiny_forcing()
    u = torch.randn(3, 1, 16, 20)
    coords = torch.rand(1, 64, 2)
    t = torch.rand(1, 64, 1) * 0.3
    out = m(u, coords, t)
    assert out.shape == (3, 64, 1)


def test_forcing_hard_dirichlet_exact():
    # The right wall (x=1) is pinned by the inherited hard-Dirichlet ansatz.
    m = _tiny_forcing(t_right_tilde=0.7)
    u = torch.randn(2, 1, 16, 20)
    coords = torch.rand(2, 32, 2)
    coords[..., 0] = 1.0
    t = torch.rand(2, 32, 1) * 0.3
    out = m(u, coords, t)
    assert torch.allclose(out, torch.full_like(out, 0.7), atol=1e-5)


def test_hard_left_flux_off_ignores_q_left():
    # Default off: the lifting is inert and q_left has no effect; the output is
    # the legacy (1 - x) * raw ansatz whether or not q_left is supplied.
    m = _tiny_forcing()  # hard_left_flux defaults False
    u = torch.randn(2, 1, 16, 20)
    coords = torch.rand(2, 16, 2)
    t = torch.rand(2, 16, 1) * 0.3
    q_left = torch.rand(2, 16, 1)
    with torch.no_grad():
        out_q = m(u, coords, t, q_left=q_left)
        out_none = m(u, coords, t)
    assert torch.allclose(out_q, out_none, atol=0)


def test_hard_left_flux_preserves_right_dirichlet():
    # With the lifting on, x=1 must still collapse to t_right: g*(x-1) and
    # (1 - x^2)*raw both vanish there, so the right Dirichlet wall is untouched.
    m = _tiny_forcing(t_right_tilde=0.4, hard_left_flux=True, left_flux_scale=0.2)
    u = torch.randn(2, 1, 16, 20)
    coords = torch.rand(2, 32, 2)
    coords[..., 0] = 1.0
    t = torch.rand(2, 32, 1) * 0.3
    q_left = torch.rand(2, 32, 1)
    out = m(u, coords, t, q_left=q_left)
    assert torch.allclose(out, torch.full_like(out, 0.4), atol=1e-5)


def test_hard_left_flux_none_preserves_right_dirichlet():
    # IC / eval path passes q_left=None -> g=0; x=1 stays pinned to t_right.
    m = _tiny_forcing(t_right_tilde=0.3, hard_left_flux=True, left_flux_scale=0.2)
    u = torch.randn(1, 1, 16, 20)
    coords = torch.rand(1, 16, 2)
    coords[..., 0] = 1.0
    t = torch.rand(1, 16, 1) * 0.3
    out = m(u, coords, t, q_left=None)
    assert torch.allclose(out, torch.full_like(out, 0.3), atol=1e-5)


def test_hard_left_flux_collapses_bc_left_to_forcing_independent():
    # left_flux_scale = 1/(k*sigma): the lifting's -q_L/(k*sigma)*(x-1) term makes
    # the left-wall residual dT_tilde/dx + q_L/(k*sigma) reduce to raw_x(0), which
    # does not depend on q_L. Two different q_L over the SAME (x=0, y, t) leaves
    # must give the same residual.
    sigma, k = 5.0, 1.0
    m = _tiny_forcing(hard_left_flux=True, left_flux_scale=1.0 / (k * sigma))
    u = torch.randn(2, 1, 16, 20)
    x = torch.zeros(2, 8, 1, requires_grad=True)
    y = torch.rand(2, 8, 1, requires_grad=True)
    t = (torch.rand(2, 8, 1) * 0.3).requires_grad_(True)
    q1 = torch.rand(2, 8, 1)
    q2 = q1 + 0.5
    r1 = forcing_neumann_residual(m, u, x, y, t, q_L=q1, sigma=sigma, k=k)
    r2 = forcing_neumann_residual(m, u, x, y, t, q_L=q2, sigma=sigma, k=k)
    assert torch.allclose(r1, r2, atol=1e-5)


def test_hard_left_flux_double_backward_left_neumann():
    # The collapsed left residual must remain twice differentiable (grad w.r.t. x
    # for the Neumann term, then w.r.t. params through the loss).
    sigma, k = 5.0, 1.0
    m = _tiny_forcing(hard_left_flux=True, left_flux_scale=1.0 / (k * sigma))
    u = torch.randn(2, 1, 16, 20)
    x = torch.zeros(2, 8, 1, requires_grad=True)
    y = torch.rand(2, 8, 1, requires_grad=True)
    t = (torch.rand(2, 8, 1) * 0.3).requires_grad_(True)
    q_L = torch.rand(2, 8, 1)
    r = forcing_neumann_residual(m, u, x, y, t, q_L=q_L, sigma=sigma, k=k)
    r.pow(2).mean().backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads, "no parameter received a gradient"
    assert all(torch.isfinite(g).all() for g in grads)


def test_forcing_double_backward_left_neumann():
    # The left-wall forcing residual dT_tilde/dx + q_L/(k*sigma) must be twice
    # differentiable: grad w.r.t. x (first order) and w.r.t. the model params
    # (second order, through the loss) both finite.
    m = _tiny_forcing()
    u = torch.randn(2, 1, 16, 20)
    x = torch.zeros(2, 8, 1, requires_grad=True)  # left wall x=0
    y = torch.rand(2, 8, 1, requires_grad=True)
    t = (torch.rand(2, 8, 1) * 0.3).requires_grad_(True)
    q_L = torch.rand(2, 8, 1)  # precomputed constant inward flux
    r = forcing_neumann_residual(m, u, x, y, t, q_L=q_L, sigma=5.0, k=1.0)
    assert r.shape == (2, 8, 1)
    loss = r.pow(2).mean()
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads, "no parameter received a gradient"
    assert all(torch.isfinite(g).all() for g in grads)
