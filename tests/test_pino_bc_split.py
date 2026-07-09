import torch

from src.operators.cvit import ForcingCViT
from src.operators.train_pino import pino_losses, sample_collocation


def _tiny_forcing(**kw):
    base = dict(
        in_ch=1, out_dim=1, emb_dim=32, patch_size=4, grid_size=(16, 20),
        depth_enc=2, depth_dec=2, num_heads=4, mlp_ratio=2.0, fourier_freq=1.0,
        hard_right_dirichlet=True, t_right_tilde=0.0,
    )
    base.update(kw)
    return ForcingCViT(**base)


def _setup(batch: int = 2, **model_kw):
    torch.manual_seed(0)
    model = _tiny_forcing(**model_kw)
    u = torch.randn(batch, 1, 16, 20)
    x_grid = torch.linspace(0.0, 1.0, 10)
    y_grid = torch.linspace(0.0, 1.0, 12)
    gen = torch.Generator().manual_seed(0)
    coll = sample_collocation(
        64, 32, 24, 0.3, x_grid, y_grid, torch.device("cpu"), gen
    )
    ic_target = torch.zeros(batch, coll["ic"]["coords"].shape[1], 1)
    _, yw_left, tw_left = coll["walls"]["left"]
    q_L = torch.rand(batch, yw_left.shape[1], 1)
    return model, u, coll, ic_target, q_L


def test_pino_losses_reports_bc_hom():
    model, u, coll, ic_target, q_L = _setup()
    losses = pino_losses(
        model, u, coll, ic_target, alpha=1.0, t_final=0.3,
        left_qL=q_L, sigma=5.0, k_slab=1.0,
    )
    assert "bc_hom" in losses
    assert losses["bc_hom"].shape == ()


def test_bc_bucket_is_left_plus_two_hom_over_three():
    # bc averages all three walls; bc_hom averages the two homogeneous walls, so
    # bc == (bc_left + 2*bc_hom)/3 exactly. This is the identity the forcing loop
    # relies on to pull the left wall out without changing the homogeneous terms.
    model, u, coll, ic_target, q_L = _setup()
    losses = pino_losses(
        model, u, coll, ic_target, alpha=1.0, t_final=0.3,
        left_qL=q_L, sigma=5.0, k_slab=1.0,
    )
    expected = (losses["bc_left"] + 2.0 * losses["bc_hom"]) / 3.0
    assert torch.allclose(losses["bc"], expected, atol=1e-6)


def test_lifting_makes_bc_left_forcing_independent_in_losses():
    # With the lifting on and left_flux_scale = 1/(k*sigma), the reported bc_left
    # collapses to a forcing-independent term, so scaling q_L leaves it unchanged.
    # The cancellation is per-sim and holds at batch>1 because the residual now
    # expands the shared collocation leaf to per-sim leaves (no batch-sum).
    sigma, k = 5.0, 1.0
    model, u, coll, ic_target, q_L = _setup(
        batch=2, hard_left_flux=True, left_flux_scale=1.0 / (k * sigma)
    )
    kw = dict(alpha=1.0, t_final=0.3, sigma=sigma, k_slab=k)
    l1 = pino_losses(model, u, coll, ic_target, left_qL=q_L, **kw)
    l2 = pino_losses(model, u, coll, ic_target, left_qL=q_L + 0.5, **kw)
    assert torch.allclose(l1["bc_left"], l2["bc_left"], atol=1e-5)
