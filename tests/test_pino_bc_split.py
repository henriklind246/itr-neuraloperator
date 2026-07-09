import numpy as np
import torch

from src.operators.cvit import ForcingCViT
from src.operators.train_pino import (
    left_wall_qL,
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


class _SpyModel(torch.nn.Module):
    """Records the ``q_left`` each forward receives; returns a zero field."""

    def __init__(self, hard_left_flux: bool):
        super().__init__()
        self.hard_left_flux = hard_left_flux
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


def test_validation_feeds_q_left_when_hard_left_flux():
    # Regression: with hard_left_flux the eval MUST pass q_L into the ansatz.
    # A None q_left drops the analytic g*(x-1) forcing term and scores an
    # insulated wall, which looks catastrophic across every forcing family.
    data, sim_params, ids = _val_data()
    y_img = data["y_grid"].copy()
    t_img = data["t_grid"].copy()
    model = _SpyModel(hard_left_flux=True)
    validate_forcing_gnrmse(
        model, data, ids, sim_params, y_img, t_img,
        a_ref=300.0, t_ramp=0.02, device=torch.device("cpu"),
    )
    assert model.seen, "model was never queried"
    assert all(q is not None for q in model.seen)
    Nx, Ny = data["x_grid"].size, data["y_grid"].size
    for q in model.seen:
        assert q.shape == (len(ids), Nx * Ny, 1)

    # The passed q_L must equal the shared reconstruction at the mesh y for the
    # first slice time (t_grid[0]); this ties eval to the training residual path.
    gx, gy = np.meshgrid(data["x_grid"], data["y_grid"], indexing="ij")
    mesh_y = torch.as_tensor(gy.reshape(-1), dtype=torch.float32)
    params = [dict(sim_params[int(i)]) for i in ids]
    t_pts = torch.full_like(mesh_y, float(data["t_grid"][0]))
    expected0 = left_wall_qL(params, mesh_y, t_pts, torch.device("cpu"), 0.02)
    assert torch.allclose(model.seen[0], expected0, atol=1e-5)


def test_validation_omits_q_left_when_flux_off():
    # The cheap legacy path is preserved: hard_left_flux=False never builds q_L.
    data, sim_params, ids = _val_data()
    model = _SpyModel(hard_left_flux=False)
    validate_forcing_gnrmse(
        model, data, ids, sim_params, data["y_grid"].copy(),
        data["t_grid"].copy(), a_ref=300.0, t_ramp=0.02,
        device=torch.device("cpu"),
    )
    assert model.seen and all(q is None for q in model.seen)
