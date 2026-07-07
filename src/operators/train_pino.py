from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from data.dataset import (
    compute_global_stats,
    load_sim_data,
    problem_from_config,
    split_sim_ids,
)
from problems.diffusion import T_RIGHT
from src.operators.cvit import CViT
from src.operators.train import (
    build_optimizer,
    build_scheduler,
    load_config,
    set_seed,
)
from src.operators.utils import resolve_device
from src.physics.pde_residual import (
    diffusion_residual,
    ic_residual,
    neumann_residual,
)

# ---- Physics-only (PINO) training of a CViT on the diffusion benchmark -------
#
# Self-contained trainer: it reuses the data loaders, optimizer/scheduler
# builders, and seeding from the FNO path by import, but never constructs an
# FNO or a SnapshotPairDataset. The objective is purely physical — interior heat
# residual + soft zero-Neumann walls + an IC anchor — so no paired (source,
# target) snapshots are used; validation compares against the saved trajectories.

WALLS = ("left", "top", "bottom")


# --------- collocation sampling ---------

def _leaf(shape, device, generator, scale: float = 1.0) -> torch.Tensor:
    t = torch.rand(shape, device=device, generator=generator) * scale
    return t.requires_grad_(True)


def sample_collocation(
    n_r: int,
    n_ic: int,
    n_bc: int,
    t_final: float,
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> dict[str, Any]:
    """Free space-time collocation for one training step (shared across sims).

    Returns leaf tensors (requires_grad where a derivative is taken), all with a
    leading batch dim of 1 so they broadcast across the sim minibatch (the model
    expands dim-1 query sets explicitly). Interior/wall points are random;
    IC points are drawn on native grid NODES (as index pairs) so per-sim IC
    targets can be read from the trajectories exactly, with no interpolation.
    The right wall (x=1) is omitted — it is enforced by the hard ansatz.
    """
    # interior: x, y ~ U(0,1); t ~ U(0, t_final)
    x_r = _leaf((1, n_r, 1), device, generator)
    y_r = _leaf((1, n_r, 1), device, generator)
    t_r = _leaf((1, n_r, 1), device, generator, scale=t_final)

    # IC: grid-node indices -> exact coords + a t=0 column
    Nx = int(x_grid.numel())
    Ny = int(y_grid.numel())
    ix = torch.randint(0, Nx, (n_ic,), device=device, generator=generator)
    iy = torch.randint(0, Ny, (n_ic,), device=device, generator=generator)
    ic_x = x_grid[ix].view(1, n_ic, 1)
    ic_y = y_grid[iy].view(1, n_ic, 1)
    ic_coords = torch.cat([ic_x, ic_y], dim=-1)  # (1, n_ic, 2)
    ic_t = torch.zeros((1, n_ic, 1), device=device)

    # walls: free coordinate ~ U(0,1); the pinned coordinate is fixed. All three
    # kept as separate leaves so neumann_residual differentiates unambiguously.
    def _wall(pin: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        free = _leaf((1, n_bc, 1), device, generator)
        tw = _leaf((1, n_bc, 1), device, generator, scale=t_final)
        if pin == "left":       # x = 0
            xw = torch.zeros((1, n_bc, 1), device=device, requires_grad=True)
            return xw, free, tw
        if pin == "top":        # y = 1
            yw = torch.ones((1, n_bc, 1), device=device, requires_grad=True)
            return free, yw, tw
        # bottom: y = 0
        yw = torch.zeros((1, n_bc, 1), device=device, requires_grad=True)
        return free, yw, tw

    walls = {w: _wall(w) for w in WALLS}

    return {
        "interior": (x_r, y_r, t_r),
        "ic": {"coords": ic_coords, "t": ic_t, "ix": ix, "iy": iy},
        "walls": walls,
    }


# --------- data helpers ---------

def load_diffusion_data(config: dict) -> dict[str, Any]:
    """Load raw trajectories + grids and the train/val/test split + global stats.

    Mirrors the FNO data-load block but skips create_dataloaders /
    SnapshotPairDataset: PINO needs raw ICs and the saved trajectory grid, not
    snapshot pairs.
    """
    trajectories, x_grid, y_grid, t_grid = load_sim_data(
        sim_traj_path=config["data"]["trajectories.npy"],
        x_grid_path=config["data"]["x_grid_path"],
        y_grid_path=config["data"]["y_grid_path"],
        t_grid_path=config["data"]["t_grid_path"],
    )
    train_ids, val_ids, test_ids = split_sim_ids(
        num_sims=trajectories.shape[0], train_frac=0.7, val_frac=0.15, seed=0,
    )
    norm_max_time = config["data"].get("norm_max_time", None)
    mu_global, sigma_global = compute_global_stats(
        trajectories, train_ids, t_grid=t_grid, max_time=norm_max_time,
    )
    return {
        "trajectories": trajectories,
        "x_grid": np.asarray(x_grid),
        "y_grid": np.asarray(y_grid),
        "t_grid": np.asarray(t_grid),
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
        "mu_global": float(mu_global),
        "sigma_global": float(sigma_global),
    }


def build_ic_batch(
    trajectories: np.ndarray,
    ids: np.ndarray,
    mu: float,
    sigma: float,
    device: torch.device,
) -> torch.Tensor:
    """Encoder input u = normalized IC field (snapshot 0); (B, 1, Nx, Ny)."""
    ic = np.asarray(trajectories[ids, 0, :, :], dtype=np.float32)
    ic = (ic - mu) / (sigma + 1e-8)
    return torch.from_numpy(ic).unsqueeze(1).to(device)


def _ic_targets(
    trajectories: np.ndarray,
    ids: np.ndarray,
    ix: torch.Tensor,
    iy: torch.Tensor,
    mu: float,
    sigma: float,
    device: torch.device,
) -> torch.Tensor:
    """Per-sim normalized IC values at the sampled grid nodes; (B, n_ic, 1)."""
    ix_np = ix.detach().cpu().numpy()
    iy_np = iy.detach().cpu().numpy()
    vals = np.asarray(trajectories[np.asarray(ids)][:, 0, :, :], dtype=np.float32)
    vals = vals[:, ix_np, iy_np]  # (B, n_ic)
    vals = (vals - mu) / (sigma + 1e-8)
    return torch.from_numpy(vals).unsqueeze(-1).to(device)


# --------- loss ---------

def pino_losses(
    model: CViT,
    u: torch.Tensor,
    batch: dict[str, Any],
    ic_target: torch.Tensor,
    alpha: float,
) -> dict[str, torch.Tensor]:
    x_r, y_r, t_r = batch["interior"]
    r = diffusion_residual(model, u, x_r, y_r, t_r, alpha=alpha)
    loss_r = (r ** 2).mean()

    ic = ic_residual(model, u, batch["ic"]["coords"], batch["ic"]["t"], ic_target)
    loss_ic = (ic ** 2).mean()

    bc_sq = 0.0
    for w in WALLS:
        xw, yw, tw = batch["walls"][w]
        nb = neumann_residual(model, u, xw, yw, tw, w)
        bc_sq = bc_sq + (nb ** 2).mean()
    loss_bc = bc_sq / len(WALLS)

    return {"r": loss_r, "ic": loss_ic, "bc": loss_bc}


# --------- validation ---------

@torch.no_grad()
def validate_rel_l2(
    model: CViT,
    data: dict[str, Any],
    ids: np.ndarray,
    device: torch.device,
    query_batch: int = 8,
) -> dict[str, float]:
    """Grid-query rel-L2 (normalized) + Kelvin RMSE over held-out sims.

    Queries the model on the full (x_grid, y_grid) mesh at every saved t_grid
    time and compares to the stored trajectories.
    """
    x_grid = torch.as_tensor(data["x_grid"], dtype=torch.float32, device=device)
    y_grid = torch.as_tensor(data["y_grid"], dtype=torch.float32, device=device)
    t_grid = data["t_grid"]
    mu, sigma = data["mu_global"], data["sigma_global"]
    Nx, Ny, Nt = x_grid.numel(), y_grid.numel(), len(t_grid)

    gx, gy = torch.meshgrid(x_grid, y_grid, indexing="ij")
    mesh = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1).unsqueeze(0)  # (1, Nx*Ny, 2)

    sq_err = 0.0
    sq_ref = 0.0
    se_K = 0.0
    n_pts = 0
    ids = np.asarray(ids)
    for start in range(0, len(ids), query_batch):
        chunk = ids[start:start + query_batch]
        u = build_ic_batch(data["trajectories"], chunk, mu, sigma, device)
        B = u.shape[0]
        coords = mesh.expand(B, -1, -1)
        pred = torch.empty((B, Nt, Nx, Ny), device=device)
        for k in range(Nt):
            tk = torch.full((B, Nx * Ny, 1), float(t_grid[k]), device=device)
            out = model(u, coords, tk)              # (B, Nx*Ny, 1)
            pred[:, k] = out[..., 0].view(B, Nx, Ny)
        truth = np.asarray(data["trajectories"][chunk], dtype=np.float32)  # (B,Nt,Nx,Ny)
        truth_t = torch.from_numpy(truth).to(device)
        truth_norm = (truth_t - mu) / (sigma + 1e-8)
        sq_err += ((pred - truth_norm) ** 2).sum().item()
        sq_ref += (truth_norm ** 2).sum().item()
        pred_K = pred * sigma + mu
        se_K += ((pred_K - truth_t) ** 2).sum().item()
        n_pts += truth_t.numel()

    rel_l2 = float(np.sqrt(sq_err / max(sq_ref, 1e-30)))
    rmse_K = float(np.sqrt(se_K / max(n_pts, 1)))
    return {"val_rel_l2": rel_l2, "val_rmse_K": rmse_K}


# --------- single-seed training ---------

def build_cvit(config: dict, mu: float, sigma: float, grid_size: tuple[int, int]) -> CViT:
    c = config["model"]["cvit"]
    t_right_K = float(c.get("hard_right_dirichlet_t_right", T_RIGHT))
    hard_rd = bool(c.get("hard_right_dirichlet", True))
    t_right_tilde = (t_right_K - mu) / (sigma + 1e-8) if hard_rd else 0.0
    return CViT(
        in_ch=int(c.get("in_ch", 1)),
        out_dim=int(c.get("out_dim", 1)),
        emb_dim=int(c.get("emb_dim", 256)),
        dec_emb_dim=c.get("dec_emb_dim", None),
        patch_size=int(c.get("patch_size", 10)),
        grid_size=grid_size,
        depth_enc=int(c.get("depth_enc", 4)),
        depth_dec=int(c.get("depth_dec", 2)),
        num_heads=int(c.get("num_heads", 8)),
        mlp_ratio=float(c.get("mlp_ratio", 2.0)),
        fourier_freq=float(c.get("fourier_freq", 1.0)),
        activation=str(c.get("activation", "gelu")),
        hard_right_dirichlet=hard_rd,
        t_right_tilde=t_right_tilde,
    )


def run_one_seed_pino(config: dict, seed: int, run_dir: Path) -> dict[str, Any]:
    set_seed(seed)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(config["training"].get("device", "auto"))
    data = load_diffusion_data(config)
    mu, sigma = data["mu_global"], data["sigma_global"]
    Nx, Ny = int(data["x_grid"].shape[0]), int(data["y_grid"].shape[0])
    t_final = float(data["t_grid"][-1])

    model = build_cvit(config, mu, sigma, grid_size=(Nx, Ny)).to(device)
    optimizer = build_optimizer(config, model.parameters())
    scheduler = build_scheduler(config, optimizer)

    pino = config["training"]["pino"]
    lam_r = float(pino["lambda_r"])
    lam_ic = float(pino["lambda_ic"])
    lam_bc = float(pino["lambda_bc"])
    n_r = int(pino["n_r"])
    n_ic = int(pino["n_ic"])
    n_bc = int(pino["n_bc"])
    sim_batch = int(pino["sim_batch"])
    alpha = float(pino.get("alpha", 1.0))

    epochs = int(config["training"]["epochs"])
    validate_every = int(config["training"].get("validate_every", 10))

    x_grid_t = torch.as_tensor(data["x_grid"], dtype=torch.float32, device=device)
    y_grid_t = torch.as_tensor(data["y_grid"], dtype=torch.float32, device=device)

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    rng = np.random.default_rng(seed)
    train_ids = np.asarray(data["train_ids"])

    metrics_path = run_dir / "train_metrics.csv"
    fieldnames = ["epoch", "loss", "loss_r", "loss_ic", "loss_bc", "val_rel_l2", "val_rmse_K"]
    with open(metrics_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    best_val = float("inf")
    history: list[dict[str, float]] = []

    for epoch in range(epochs):
        model.train()
        batch_ids = rng.choice(train_ids, size=min(sim_batch, len(train_ids)), replace=False)
        u = build_ic_batch(data["trajectories"], batch_ids, mu, sigma, device)

        coll = sample_collocation(n_r, n_ic, n_bc, t_final, x_grid_t, y_grid_t, device, gen)
        ic_target = _ic_targets(
            data["trajectories"], batch_ids, coll["ic"]["ix"], coll["ic"]["iy"], mu, sigma, device,
        )

        optimizer.zero_grad(set_to_none=True)
        losses = pino_losses(model, u, coll, ic_target, alpha)
        loss = lam_r * losses["r"] + lam_ic * losses["ic"] + lam_bc * losses["bc"]
        loss.backward()
        optimizer.step()
        scheduler.step()

        row = {
            "epoch": epoch,
            "loss": float(loss.detach().cpu()),
            "loss_r": float(losses["r"].detach().cpu()),
            "loss_ic": float(losses["ic"].detach().cpu()),
            "loss_bc": float(losses["bc"].detach().cpu()),
            "val_rel_l2": "",
            "val_rmse_K": "",
        }

        do_val = (epoch % validate_every == 0) or (epoch == epochs - 1)
        if do_val:
            model.eval()
            val = validate_rel_l2(model, data, data["val_ids"], device)
            row["val_rel_l2"] = val["val_rel_l2"]
            row["val_rmse_K"] = val["val_rmse_K"]
            if val["val_rel_l2"] < best_val:
                best_val = val["val_rel_l2"]
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "mu_global": mu,
                        "sigma_global": sigma,
                        "config": config,
                        "epoch": epoch,
                        "best_val": best_val,
                    },
                    run_dir / "cvit_best.pt",
                )

        history.append({k: (v if v != "" else None) for k, v in row.items()})
        with open(metrics_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(row)

    summary = {"seed": seed, "best_val_rel_l2": best_val, "epochs": epochs}
    with open(run_dir / "final_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def run_config_seeds_pino(
    config: dict, base_run_dir: Path, seeds: list[int]
) -> dict[str, Any]:
    base_run_dir = Path(base_run_dir)
    results = {}
    for seed in seeds:
        seed_dir = base_run_dir / f"seed{seed}"
        results[str(seed)] = run_one_seed_pino(config, int(seed), seed_dir)
    return {"run_dir": str(base_run_dir), "seeds": results}


__all__ = [
    "sample_collocation",
    "load_diffusion_data",
    "build_ic_batch",
    "pino_losses",
    "validate_rel_l2",
    "build_cvit",
    "run_one_seed_pino",
    "run_config_seeds_pino",
]
