import numpy as np
import pytest
import torch

from src.operators.cvit import ForcingCViT
from src.operators.train_pino import (
    forcing_data_loss,
    pino_losses,
    sample_collocation,
    sample_forcing_params,
    validate_forcing_gnrmse,
)


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


def test_invalid_global_left_flux_lift_is_rejected():
    with pytest.raises(ValueError, match="interior PDE residual"):
        _setup(batch=2, hard_left_flux=True, left_flux_scale=0.2)


def test_soft_left_bc_is_forcing_dependent_in_losses():
    # The left wall is a genuine soft Neumann penalty
    # (dT_tilde/dx|_0 + q_L/(k*sigma))^2, so it must move with q_L.
    sigma, k = 5.0, 1.0
    model, u, coll, ic_target, q_L = _setup(batch=2, hard_left_flux=False)
    kw = dict(alpha=1.0, t_final=0.3, sigma=sigma, k_slab=k)
    l1 = pino_losses(model, u, coll, ic_target, left_qL=q_L, **kw)
    l2 = pino_losses(model, u, coll, ic_target, left_qL=q_L + 0.5, **kw)
    assert not torch.allclose(l1["bc_left"], l2["bc_left"], atol=1e-5)


def _data_setup(n_sims: int = 4, Nt: int = 8, Nx: int = 10, Ny: int = 12):
    torch.manual_seed(0)
    model = _tiny_forcing()
    rng = np.random.default_rng(0)
    params = sample_forcing_params(rng, n_sims, dt=0.003, t_final=0.3)
    sim_params = np.array(params, dtype=object)
    # Saved FV field on the (Nt, Nx, Ny) grid; a deviation off the 300 K baseline.
    trajectories = (300.0 + rng.standard_normal((n_sims, Nt, Nx, Ny)) * 5.0).astype(
        np.float32
    )
    kw = dict(
        y_img=np.linspace(0.0, 1.0, 16),
        t_img=np.linspace(0.0, 0.3, 20),
        a_ref=300.0,
        t_ramp=0.003,
        x_grid=torch.linspace(0.0, 1.0, Nx),
        y_grid=torch.linspace(0.0, 1.0, Ny),
        t_grid=np.linspace(0.0, 0.3, Nt),
        mu=300.0,
        sigma=5.0,
        n_sims=3,
        n_pts=50,
        device=torch.device("cpu"),
    )
    return model, sim_params, trajectories, np.arange(n_sims), kw


def test_forcing_data_loss_is_differentiable_and_truth_dependent():
    # Lever #1: the supervised data term must be a finite, positive, grad-enabled
    # scalar that actually depends on the FV truth (so it can pin the field level
    # the soft Neumann BC cannot). A fresh generator each call isolates the change
    # to the truth, not the sampled nodes.
    model, sim_params, traj, ids, kw = _data_setup()

    def _loss(trajectories):
        return forcing_data_loss(
            model, sim_params, trajectories, ids,
            rng=np.random.default_rng(0),
            gen=torch.Generator().manual_seed(0),
            **kw,
        )

    loss = _loss(traj)
    assert loss.shape == ()
    assert torch.isfinite(loss) and float(loss) > 0.0
    assert loss.requires_grad
    loss.backward()
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters()
    )
    # Shifting the FV truth by a constant must change the value MSE.
    shifted = _loss(traj + 25.0)
    assert not torch.allclose(loss.detach(), shifted.detach(), atol=1e-6)


class _SpyModel(torch.nn.Module):
    """Records the ``q_left`` each forward receives; returns a zero field."""

    def __init__(self):
        super().__init__()
        self.seen: list = []

    def forward(self, u, coords, t, q_left=None):
        self.seen.append(q_left)
        return torch.zeros(u.shape[0], coords.shape[1], 1)


def _val_data(Nx=6, Ny=5, Nt=4, n_sims=3):
    x_grid = np.linspace(0.0, 1.0, Nx).astype(np.float32)
    y_grid = np.linspace(0.0, 1.0, Ny).astype(np.float32)
    t_grid = np.linspace(0.0, 0.3, Nt).astype(np.float32)
    data = {
        "x_grid": x_grid, "y_grid": y_grid, "t_grid": t_grid,
        "mu_global": 300.0, "sigma_global": 5.0,
        "trajectories": np.zeros((n_sims, Nt, Nx, Ny), dtype=np.float32),
    }
    rng = np.random.default_rng(0)
    records = sample_forcing_params(rng, n_sims, dt=0.3 / (Nt - 1), t_final=0.3)
    sim_params = np.array(records, dtype=object)
    return data, sim_params, np.arange(n_sims)


def test_validation_never_injects_boundary_flux_into_field_ansatz():
    data, sim_params, ids = _val_data()
    model = _SpyModel()
    validate_forcing_gnrmse(
        model, data, ids, sim_params, data["y_grid"].copy(),
        data["t_grid"].copy(), a_ref=300.0, t_ramp=0.02,
        device=torch.device("cpu"),
    )
    assert model.seen and all(q is None for q in model.seen)
