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
    GradNormBalancer,
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
    dense_ic: bool = False,
) -> dict[str, Any]:
    """Free space-time collocation for one training step (shared across sims).

    Returns leaf tensors (requires_grad where a derivative is taken), all with a
    leading batch dim of 1 so they broadcast across the sim minibatch (the model
    expands dim-1 query sets explicitly). Interior/wall points are random;
    IC points are drawn on native grid NODES (as index pairs) so per-sim IC
    targets can be read from the trajectories exactly, with no interpolation.
    The right wall (x=1) is omitted — it is enforced by the hard ansatz.

    ``dense_ic`` replaces the ``n_ic`` random IC nodes with EVERY grid node
    (``Nx*Ny`` points, no RNG draw): the constant field is the physics attractor
    for this benchmark, so a dense per-sim IC anchor is the primary stabilizer.
    The default (``False``) keeps the legacy random-node draw byte-identical.
    """
    # interior: x, y ~ U(0,1); t ~ U(0, t_final)
    x_r = _leaf((1, n_r, 1), device, generator)
    y_r = _leaf((1, n_r, 1), device, generator)
    t_r = _leaf((1, n_r, 1), device, generator, scale=t_final)

    # IC: grid-node indices -> exact coords + a t=0 column
    Nx = int(x_grid.numel())
    Ny = int(y_grid.numel())
    if dense_ic:
        # Every node, row-major (ix varies slowest) -> exact full-grid anchor.
        ix = torch.arange(Nx, device=device).repeat_interleave(Ny)
        iy = torch.arange(Ny, device=device).repeat(Nx)
    else:
        ix = torch.randint(0, Nx, (n_ic,), device=device, generator=generator)
        iy = torch.randint(0, Ny, (n_ic,), device=device, generator=generator)
    n_ic_eff = int(ix.numel())
    ic_x = x_grid[ix].view(1, n_ic_eff, 1)
    ic_y = y_grid[iy].view(1, n_ic_eff, 1)
    ic_coords = torch.cat([ic_x, ic_y], dim=-1)  # (1, n_ic_eff, 2)
    ic_t = torch.zeros((1, n_ic_eff, 1), device=device)

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


# --------- causal residual weighting + time-bin diagnostics ---------

def _time_bin_index(t: torch.Tensor, t_final: float, n_bins: int) -> torch.Tensor:
    """Long bin index in [0, n_bins-1] for times t in [0, t_final]."""
    frac = t / max(float(t_final), 1e-12)
    idx = (frac * n_bins).long()
    return idx.clamp_(0, n_bins - 1)


def _bin_residual(
    r: torch.Tensor, t_r: torch.Tensor, t_final: float, n_bins: int
) -> torch.Tensor:
    """Detached per-time-bin mean squared residual, (n_bins,); empty bins -> 0."""
    sq = (r ** 2).mean(dim=0).reshape(-1).detach()
    idx = _time_bin_index(t_r.reshape(-1).detach(), t_final, n_bins)
    bin_sum = torch.zeros(n_bins, device=r.device, dtype=sq.dtype)
    bin_cnt = torch.zeros(n_bins, device=r.device, dtype=sq.dtype)
    bin_sum = bin_sum.index_add(0, idx, sq)
    bin_cnt = bin_cnt.index_add(0, idx, torch.ones_like(sq))
    return bin_sum / bin_cnt.clamp_min(1.0)


def _causal_weights(bin_mean: torch.Tensor, eps_causal: float) -> torch.Tensor:
    """Causal weights w_i = exp(-eps * sum_{j<i} L_j) from detached bin losses.

    Later time bins are only penalized once the earlier bins are resolved
    (Wang, Sankaran & Perdikaris 2022; Chen et al. arXiv:2606.06164): a small
    time-averaged residual that nonetheless violates the causal time evolution is
    the failure mode this counteracts.
    """
    cum_prev = torch.cumsum(bin_mean, 0) - bin_mean
    return torch.exp(-float(eps_causal) * cum_prev)


def _causal_residual_loss(
    r: torch.Tensor, t_r: torch.Tensor, t_final: float, n_bins: int, eps_causal: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Causally weighted interior residual loss + detached per-bin residual.

    Bins the interior residual by query time, forms a differentiable per-bin mean
    squared residual, and returns a weighted mean over occupied bins with causal
    weights (detached, so they act as a mask, not an extra gradient path).
    """
    sq = (r ** 2).mean(dim=0).reshape(-1)  # (n_r,) differentiable
    idx = _time_bin_index(t_r.reshape(-1).detach(), t_final, n_bins)
    bin_sum = torch.zeros(n_bins, device=r.device, dtype=sq.dtype)
    bin_cnt = torch.zeros(n_bins, device=r.device, dtype=sq.dtype)
    bin_sum = bin_sum.index_add(0, idx, sq)
    bin_cnt = bin_cnt.index_add(0, idx, torch.ones_like(sq))
    bin_mean = bin_sum / bin_cnt.clamp_min(1.0)  # (n_bins,) differentiable
    w = _causal_weights(bin_mean.detach(), eps_causal) * (bin_cnt > 0)
    denom = w.sum().clamp_min(1e-12)
    loss = (w * bin_mean).sum() / denom
    return loss, bin_mean.detach()


# --------- loss ---------

def _ic_loss(
    ic: torch.Tensor,
    ic_target: torch.Tensor,
    *,
    mode: str = "mse",
    t_right_tilde: float = 0.0,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """IC term from the raw IC residual ``ic = T_hat_norm(0) - ic_target``.

    ``mode="mse"`` (legacy): plain ``mean(ic**2)`` -- a raw normalized MSE that
    goes numerically small on a near-300 K field even when the *relative* error
    in the nonconstant thermal signal is large (t0 rel-L2 ~0.75 on diffusion, the
    IC that seeds the whole trajectory).

    ``mode="rel"``: per-sim relative L2,
    ``mean_b[ sum_i ic**2 / (sum_i (ic_target - t_right_tilde)**2 + eps) ]``. The
    denominator is the IC's energy measured as a *deviation from the constant
    300 K field* (``t_right_tilde = (300 - mu)/sigma``), so the loss targets
    relative error in the signal validation rel-L2 measures, not absolute Kelvin
    MSE. Per-sim (dim-0) so it is invariant to each trajectory's IC amplitude;
    for a single sim it reduces to one ratio.
    """
    if mode == "rel":
        num = (ic ** 2).sum(dim=(1, 2))                    # (B,)
        dev = ic_target - t_right_tilde
        den = (dev ** 2).sum(dim=(1, 2)) + eps             # (B,)
        return (num / den).mean()
    return (ic ** 2).mean()


def pino_losses(
    model: CViT,
    u: torch.Tensor,
    batch: dict[str, Any],
    ic_target: torch.Tensor,
    alpha: float,
    *,
    causal_cfg: dict | None = None,
    t_final: float | None = None,
    res_bins: int = 0,
    ic_loss: str = "mse",
    t_right_tilde: float = 0.0,
    ic_eps: float = 1.0e-6,
) -> dict[str, torch.Tensor]:
    """Raw physics/IC/BC losses. ``causal_cfg.enabled`` swaps the plain
    ``mean(r^2)`` interior term for a causally time-weighted one (needs
    ``t_final``); ``res_bins > 0`` also returns a detached ``res_bins`` per-time
    residual vector for diagnostics. ``ic_loss="rel"`` (with ``t_right_tilde``)
    swaps the raw IC MSE for the per-sim relative IC L2 (see ``_ic_loss``). All
    default off -> legacy behavior.
    """
    x_r, y_r, t_r = batch["interior"]
    r = diffusion_residual(model, u, x_r, y_r, t_r, alpha=alpha)
    causal_on = bool(causal_cfg and causal_cfg.get("enabled", False))
    bin_mean = None
    if causal_on:
        if t_final is None:
            raise ValueError("causal residual weighting requires t_final")
        loss_r, bin_mean = _causal_residual_loss(
            r, t_r, float(t_final),
            int(causal_cfg.get("n_bins", 16)),
            float(causal_cfg.get("eps_causal", 1.0)),
        )
    else:
        loss_r = (r ** 2).mean()

    ic = ic_residual(model, u, batch["ic"]["coords"], batch["ic"]["t"], ic_target)
    loss_ic = _ic_loss(
        ic, ic_target, mode=ic_loss, t_right_tilde=t_right_tilde, eps=ic_eps,
    )

    bc_sq = 0.0
    for w in WALLS:
        xw, yw, tw = batch["walls"][w]
        nb = neumann_residual(model, u, xw, yw, tw, w)
        bc_sq = bc_sq + (nb ** 2).mean()
    loss_bc = bc_sq / len(WALLS)

    out = {"r": loss_r, "ic": loss_ic, "bc": loss_bc}
    if res_bins:
        if bin_mean is not None and bin_mean.numel() == res_bins:
            out["res_bins"] = bin_mean
        else:
            out["res_bins"] = _bin_residual(r, t_r, float(t_final), int(res_bins))
    return out


# --------- loss weighting (curriculum + GradNorm) ---------

def _curriculum_weights(
    epoch: int,
    lam_r: float,
    lam_ic: float,
    lam_bc: float,
    cfg: dict | None,
) -> tuple[float, float, float]:
    """Static per-term weights for this epoch under the IC-first curriculum.

    Returns the base ``lambda_*`` unchanged when the curriculum is disabled.

    ``mode="ic_only"`` (default, legacy): an IC-only phase (``w_r = w_bc = 0``)
    for ``ic_only_epochs`` steps, then a linear ramp of the residual/BC weights up
    to their ``lambda_*`` targets over ``ramp_epochs`` steps, then full weights.

    ``mode="ic_heavy"``: the residual/BC stay ON through the warm-up (Chen et al.
    arXiv:2606.06164 — the residual must see the IC-anchored field from ``t=0`` to
    propagate it forward, so zeroing it re-opens the constant-field basin). For
    ``warmup_epochs`` steps the weights are ``(warmup_lambda_r, lambda_ic,
    warmup_lambda_bc)``; then over ``decay_epochs`` the IC weight linearly relaxes
    ``lambda_ic -> lambda_ic_final`` while the residual/BC weights ramp
    ``warmup_lambda_* -> lambda_*``. The IC weight is always active.
    """
    if not cfg or not cfg.get("enabled", False):
        return lam_r, lam_ic, lam_bc
    mode = str(cfg.get("mode", "ic_only"))
    if mode == "ic_heavy":
        warm = int(cfg.get("warmup_epochs", 0))
        wr0 = float(cfg.get("warmup_lambda_r", lam_r))
        wbc0 = float(cfg.get("warmup_lambda_bc", lam_bc))
        decay = max(1, int(cfg.get("decay_epochs", 1)))
        ic_final = float(cfg.get("lambda_ic_final", lam_ic))
        if epoch < warm:
            return wr0, lam_ic, wbc0
        s = min(1.0, (epoch - warm) / decay)
        w_ic = lam_ic + s * (ic_final - lam_ic)
        w_r = wr0 + s * (lam_r - wr0)
        w_bc = wbc0 + s * (lam_bc - wbc0)
        return w_r, w_ic, w_bc
    ic_only = int(cfg.get("ic_only_epochs", 0))
    ramp = max(1, int(cfg.get("ramp_epochs", 1)))
    if epoch < ic_only:
        return 0.0, lam_ic, 0.0
    s = min(1.0, (epoch - ic_only) / ramp)
    return s * lam_r, lam_ic, s * lam_bc


def build_gradnorm(config: dict, lam_r: float, lam_ic: float, lam_bc: float):
    """GradNormBalancer over the active PINO terms, or None when disabled.

    Reuses the FNO path's balancer (inverse gradient-norm multipliers, EMA
    smoothed). Term set is the base terms with a positive static ``lambda_*`` (an
    IC-first curriculum may zero ``r``/``bc`` for early epochs, but those terms
    still come online during the ramp, so they belong in ``term_names``).
    """
    gn_cfg = config["training"].get("gradnorm", {}) or {}
    if not bool(gn_cfg.get("enabled", False)):
        return None
    terms = [n for n, w in (("r", lam_r), ("ic", lam_ic), ("bc", lam_bc)) if w > 0.0]
    return GradNormBalancer(
        terms,
        alpha_w=float(gn_cfg.get("alpha_w", 0.9)),
        update_every=int(gn_cfg.get("update_every", 10)),
        eps=float(gn_cfg.get("eps", 1.0e-8)),
    )


def _term_grad_norms(terms: dict[str, torch.Tensor], params) -> dict[str, float]:
    """L2 gradient norm of each raw loss term w.r.t. ``params`` on the live graph.

    Diagnostic only: exposes which objective dominates the shared backbone
    gradient (the quantity GradNorm balances). Uses ``retain_graph=True`` so the
    caller's subsequent ``loss.backward()`` still runs; call BEFORE that backward.
    """
    params = [p for p in params if p.requires_grad]
    out: dict[str, float] = {}
    for name, term in terms.items():
        grads = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
        sq = torch.zeros((), device=params[0].device) if params else torch.zeros(())
        for g in grads:
            if g is not None:
                sq = sq + g.detach().pow(2).sum()
        out[name] = float(torch.sqrt(sq))
    return out


# --------- validation ---------

@torch.no_grad()
def validate_rel_l2(
    model: CViT,
    data: dict[str, Any],
    ids: np.ndarray,
    device: torch.device,
    query_batch: int = 8,
    return_per_time: bool = False,
) -> dict[str, float]:
    """Grid-query rel-L2 (normalized) + Kelvin RMSE over held-out sims.

    Queries the model on the full (x_grid, y_grid) mesh at every saved t_grid
    time and compares to the stored trajectories. ``return_per_time=True`` adds
    honest per-time / banded diagnostics for a near-uniform late-time field where
    the plain normalized rel-L2 denominator collapses toward the constant-300 K
    reference:
      - ``per_time``        normalized rel-L2 at each saved time (length Nt),
      - ``per_time_rmse_K`` Kelvin RMSE at each saved time,
      - ``per_time_rel_dev`` rel-L2 measured on the deviation (T - 300 K); sigma
        cancels so this equals the Kelvin rel-L2 on ``T - T_RIGHT``,
      - ``t0_rel_l2``       the t=0 rel-L2 (the IC pass/fail gate),
      - band scalars ``rel_l2_{early,mid,late}``, ``rmse_K_{early,mid,late}``,
        ``rel_dev_{early,mid,late}`` over t<=0.05 / 0.05<t<=0.15 / t>0.15.
    ``t_right_tilde = (T_RIGHT - mu)/sigma`` is the normalized constant-300 field.
    """
    x_grid = torch.as_tensor(data["x_grid"], dtype=torch.float32, device=device)
    y_grid = torch.as_tensor(data["y_grid"], dtype=torch.float32, device=device)
    t_grid = np.asarray(data["t_grid"], dtype=np.float64)
    mu, sigma = data["mu_global"], data["sigma_global"]
    t_right_tilde = (T_RIGHT - mu) / (sigma + 1e-8)
    Nx, Ny, Nt = x_grid.numel(), y_grid.numel(), len(t_grid)

    gx, gy = torch.meshgrid(x_grid, y_grid, indexing="ij")
    mesh = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1).unsqueeze(0)  # (1, Nx*Ny, 2)

    sq_err = 0.0
    sq_ref = 0.0
    se_K = 0.0
    n_pts = 0
    sq_err_t = np.zeros(Nt, dtype=np.float64)
    sq_ref_t = np.zeros(Nt, dtype=np.float64)
    dev_ref_t = np.zeros(Nt, dtype=np.float64)
    se_K_t = np.zeros(Nt, dtype=np.float64)
    n_pts_t = np.zeros(Nt, dtype=np.float64)
    ids = np.asarray(ids)
    with torch.no_grad():
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
            if return_per_time:
                err_bt = ((pred - truth_norm) ** 2).sum(dim=(0, 2, 3))
                ref_bt = (truth_norm ** 2).sum(dim=(0, 2, 3))
                dev_bt = ((truth_norm - t_right_tilde) ** 2).sum(dim=(0, 2, 3))
                seK_bt = ((pred_K - truth_t) ** 2).sum(dim=(0, 2, 3))
                sq_err_t += err_bt.detach().cpu().numpy().astype(np.float64)
                sq_ref_t += ref_bt.detach().cpu().numpy().astype(np.float64)
                dev_ref_t += dev_bt.detach().cpu().numpy().astype(np.float64)
                se_K_t += seK_bt.detach().cpu().numpy().astype(np.float64)
                n_pts_t += float(B * Nx * Ny)

    rel_l2 = float(np.sqrt(sq_err / max(sq_ref, 1e-30)))
    rmse_K = float(np.sqrt(se_K / max(n_pts, 1)))
    out = {"val_rel_l2": rel_l2, "val_rmse_K": rmse_K}
    if return_per_time:
        per_time = np.sqrt(sq_err_t / np.maximum(sq_ref_t, 1e-30))
        per_time_rel_dev = np.sqrt(sq_err_t / np.maximum(dev_ref_t, 1e-30))
        per_time_rmse_K = np.sqrt(se_K_t / np.maximum(n_pts_t, 1.0))
        out["per_time"] = per_time
        out["per_time_rel_dev"] = per_time_rel_dev
        out["per_time_rmse_K"] = per_time_rmse_K
        out["t0_rel_l2"] = float(per_time[0])
        bands = {
            "early": t_grid <= 0.05,
            "mid": (t_grid > 0.05) & (t_grid <= 0.15),
            "late": t_grid > 0.15,
        }
        for name, mask in bands.items():
            if not mask.any():
                out[f"rel_l2_{name}"] = float("nan")
                out[f"rel_dev_{name}"] = float("nan")
                out[f"rmse_K_{name}"] = float("nan")
                continue
            out[f"rel_l2_{name}"] = float(
                np.sqrt(sq_err_t[mask].sum() / max(sq_ref_t[mask].sum(), 1e-30))
            )
            out[f"rel_dev_{name}"] = float(
                np.sqrt(sq_err_t[mask].sum() / max(dev_ref_t[mask].sum(), 1e-30))
            )
            out[f"rmse_K_{name}"] = float(
                np.sqrt(se_K_t[mask].sum() / max(n_pts_t[mask].sum(), 1.0))
            )
    return out


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
        fourier_freq_t=(
            None if c.get("fourier_freq_t", None) is None
            else float(c["fourier_freq_t"])
        ),
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

    # IC term norm: "mse" (raw normalized MSE, legacy) or "rel" (per-sim relative
    # L2 vs the constant-300 K deviation). The relative form makes the IC anchor
    # target the same signal the validation rel-L2 measures on a near-300 K field.
    ic_loss = str(pino.get("ic_loss", "mse"))
    ic_eps = float(pino.get("ic_eps", 1.0e-6))
    t_right_tilde_ic = (T_RIGHT - mu) / (sigma + 1e-8)

    dense_ic = bool(pino.get("dense_ic", False))
    resample_every = max(1, int(pino.get("resample_every", 1)))
    causal_cfg = pino.get("causal", {}) or {}
    causal_on = bool(causal_cfg.get("enabled", False))
    res_n_bins = (
        int(causal_cfg.get("n_bins", 16)) if causal_on
        else int(pino.get("diag_time_bins", 16))
    )

    curr_cfg = pino.get("curriculum", {}) or {}
    gradnorm = build_gradnorm(config, lam_r, lam_ic, lam_bc)

    epochs = int(config["training"]["epochs"])
    validate_every = int(config["training"].get("validate_every", 10))

    x_grid_t = torch.as_tensor(data["x_grid"], dtype=torch.float32, device=device)
    y_grid_t = torch.as_tensor(data["y_grid"], dtype=torch.float32, device=device)

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    rng = np.random.default_rng(seed)
    train_ids = np.asarray(data["train_ids"])

    metrics_path = run_dir / "train_metrics.csv"
    # Banded / t0 columns give an honest read on a near-300 K late field where the
    # plain normalized rel-L2 denominator collapses: val_t0_rel_l2 is the IC gate,
    # rel_dev_* is rel-L2 on (T-300 K), rmse_K_* is the Kelvin band error.
    band_cols = [
        f"{stem}_{band}"
        for stem in ("rel_l2", "rel_dev", "rmse_K")
        for band in ("early", "mid", "late")
    ]
    fieldnames = [
        "epoch", "loss", "loss_r", "loss_ic", "loss_bc",
        "w_r", "w_ic", "w_bc", "grad_norm_r", "grad_norm_ic", "grad_norm_bc",
        "val_rel_l2", "val_rmse_K", "val_t0_rel_l2",
    ] + band_cols
    with open(metrics_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    # Per-time diagnostics (written on validation epochs only): rel-L2 at each
    # saved time exposes IC-fit-but-trajectory-drift; per-time-bin residual shows
    # whether the residual is uniformly small or concentrated at late times. The
    # rel-dev (rel-L2 on T-300 K) and RMSE_K per-time CSVs are the metric-honest
    # companions for the near-uniform late field.
    t_grid = np.asarray(data["t_grid"])
    Nt = int(t_grid.shape[0])
    relt_path = run_dir / "rel_l2_per_time.csv"
    relt_fields = ["epoch"] + [f"t{k}" for k in range(Nt)]
    with open(relt_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=relt_fields).writeheader()
    reldevt_path = run_dir / "rel_l2_dev_per_time.csv"
    with open(reldevt_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=relt_fields).writeheader()
    rmseKt_path = run_dir / "rmse_K_per_time.csv"
    with open(rmseKt_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=relt_fields).writeheader()
    resbin_path = run_dir / "residual_per_time_bin.csv"
    resbin_fields = ["epoch"] + [f"bin{b}" for b in range(res_n_bins)]
    with open(resbin_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=resbin_fields).writeheader()

    best_val = float("inf")
    history: list[dict[str, float]] = []

    curr_on = bool(curr_cfg.get("enabled", False))
    if curr_on and str(curr_cfg.get("mode", "ic_only")) == "ic_heavy":
        curr_desc = (
            f"ic_heavy warm={int(curr_cfg.get('warmup_epochs', 0))} "
            f"decay={int(curr_cfg.get('decay_epochs', 1))} "
            f"ic:{lam_ic}->{float(curr_cfg.get('lambda_ic_final', lam_ic))}"
        )
    elif curr_on:
        curr_desc = (
            f"ic_only={int(curr_cfg.get('ic_only_epochs', 0))} "
            f"ramp={int(curr_cfg.get('ramp_epochs', 1))}"
        )
    else:
        curr_desc = "off"
    gn_desc = (
        f"on(terms={gradnorm.term_names})" if gradnorm is not None else "off"
    )
    causal_desc = (
        f"on(n_bins={res_n_bins},eps={float(causal_cfg.get('eps_causal', 1.0))})"
        if causal_on else "off"
    )
    print(
        f"[pino] seed={seed} device={device} epochs={epochs} "
        f"validate_every={validate_every} | grid={Nx}x{Ny} t_final={t_final:.4f} | "
        f"lambda_r={lam_r} lambda_ic={lam_ic} lambda_bc={lam_bc} | "
        f"n_r={n_r} n_ic={n_ic} n_bc={n_bc} sim_batch={sim_batch} alpha={alpha} | "
        f"ic_loss={ic_loss} dense_ic={dense_ic} resample_every={resample_every} | "
        f"curriculum={curr_desc} gradnorm={gn_desc} causal={causal_desc}",
        flush=True,
    )

    coll = None
    for epoch in range(epochs):
        model.train()
        batch_ids = rng.choice(train_ids, size=min(sim_batch, len(train_ids)), replace=False)
        u = build_ic_batch(data["trajectories"], batch_ids, mu, sigma, device)

        # Slower warm-up resampling: redraw the interior/wall collocation only
        # every ``resample_every`` steps. Per-step resampling can keep the IC loss
        # from converging (Chen et al.); reusing the collocation set lets the
        # anchor settle. IC targets are re-read every step (the sim minibatch
        # rotates); resample_every=1 restores per-step sampling (legacy).
        do_val = (epoch % validate_every == 0) or (epoch == epochs - 1)
        if coll is None or (epoch % resample_every == 0):
            coll = sample_collocation(
                n_r, n_ic, n_bc, t_final, x_grid_t, y_grid_t, device, gen,
                dense_ic=dense_ic,
            )
        ic_target = _ic_targets(
            data["trajectories"], batch_ids, coll["ic"]["ix"], coll["ic"]["iy"], mu, sigma, device,
        )

        optimizer.zero_grad(set_to_none=True)
        losses = pino_losses(
            model, u, coll, ic_target, alpha,
            causal_cfg=causal_cfg, t_final=t_final,
            res_bins=(res_n_bins if do_val else 0),
            ic_loss=ic_loss, t_right_tilde=t_right_tilde_ic, ic_eps=ic_eps,
        )

        # Static per-term weights for this epoch (IC-first curriculum or plain
        # lambda_*), then GradNorm multipliers on the RAW magnitudes. Only terms
        # with a positive static weight are measured; a zero-weighted term (IC-only
        # phase) is neither balanced nor added to the total.
        w = {"r": 0.0, "ic": 0.0, "bc": 0.0}
        w["r"], w["ic"], w["bc"] = _curriculum_weights(
            epoch, lam_r, lam_ic, lam_bc, curr_cfg,
        )
        if gradnorm is not None:
            active = {k: losses[k] for k in ("r", "ic", "bc") if w[k] > 0.0}
            mults = gradnorm.maybe_update(active, model.parameters())
        else:
            mults = {}
        w_eff = {k: w[k] * float(mults.get(k, 1.0)) for k in ("r", "ic", "bc")}

        # Per-term gradient norms (diagnostic; val epochs only to bound the extra
        # backward passes). Measured on the live graph before the combined backward.
        gnorms: dict[str, float] = {}
        if do_val:
            active_terms = {k: losses[k] for k in ("r", "ic", "bc") if w[k] > 0.0}
            if active_terms:
                gnorms = _term_grad_norms(active_terms, model.parameters())

        loss = (
            w_eff["r"] * losses["r"]
            + w_eff["ic"] * losses["ic"]
            + w_eff["bc"] * losses["bc"]
        )
        loss.backward()
        lr = float(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

        row = {
            "epoch": epoch,
            "loss": float(loss.detach().cpu()),
            "loss_r": float(losses["r"].detach().cpu()),
            "loss_ic": float(losses["ic"].detach().cpu()),
            "loss_bc": float(losses["bc"].detach().cpu()),
            "w_r": w_eff["r"],
            "w_ic": w_eff["ic"],
            "w_bc": w_eff["bc"],
            "grad_norm_r": gnorms.get("r", ""),
            "grad_norm_ic": gnorms.get("ic", ""),
            "grad_norm_bc": gnorms.get("bc", ""),
            "val_rel_l2": "",
            "val_rmse_K": "",
            "val_t0_rel_l2": "",
            **{c: "" for c in band_cols},
        }

        print(
            f"Epoch {epoch}: loss={row['loss']:.6f} "
            f"(r={row['loss_r']:.6f}, ic={row['loss_ic']:.6f}, bc={row['loss_bc']:.6f}) "
            f"w=({w_eff['r']:.3g},{w_eff['ic']:.3g},{w_eff['bc']:.3g}) "
            f"lr={lr:.2e}",
            flush=True,
        )

        if do_val:
            model.eval()
            val = validate_rel_l2(
                model, data, data["val_ids"], device, return_per_time=True,
            )
            row["val_rel_l2"] = val["val_rel_l2"]
            row["val_rmse_K"] = val["val_rmse_K"]
            row["val_t0_rel_l2"] = val["t0_rel_l2"]
            for c in band_cols:
                row[c] = val[c]

            # per-time rel-L2 (normalized), rel-L2 on (T-300 K), RMSE_K, and
            # per-time-bin residual diagnostics
            with open(relt_path, "a", newline="") as f:
                rr = {"epoch": epoch}
                rr.update({f"t{k}": float(val["per_time"][k]) for k in range(Nt)})
                csv.DictWriter(f, fieldnames=relt_fields).writerow(rr)
            with open(reldevt_path, "a", newline="") as f:
                rr = {"epoch": epoch}
                rr.update({f"t{k}": float(val["per_time_rel_dev"][k]) for k in range(Nt)})
                csv.DictWriter(f, fieldnames=relt_fields).writerow(rr)
            with open(rmseKt_path, "a", newline="") as f:
                rr = {"epoch": epoch}
                rr.update({f"t{k}": float(val["per_time_rmse_K"][k]) for k in range(Nt)})
                csv.DictWriter(f, fieldnames=relt_fields).writerow(rr)
            if "res_bins" in losses:
                rb = losses["res_bins"].detach().cpu().numpy()
                with open(resbin_path, "a", newline="") as f:
                    rr = {"epoch": epoch}
                    rr.update({f"bin{b}": float(rb[b]) for b in range(res_n_bins)})
                    csv.DictWriter(f, fieldnames=resbin_fields).writerow(rr)
            if gnorms:
                gn_txt = " ".join(f"{k}={gnorms[k]:.3e}" for k in ("r", "ic", "bc") if k in gnorms)
                print(f"  grad_norms: {gn_txt}", flush=True)
            is_best = val["val_rel_l2"] < best_val
            if is_best:
                best_val = val["val_rel_l2"]
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "mu_global": mu,
                        "sigma_global": sigma,
                        "config": config,
                        "epoch": epoch,
                        "best_val": best_val,
                        "gradnorm_state": (
                            gradnorm.state_dict() if gradnorm is not None else None
                        ),
                    },
                    run_dir / "cvit_best.pt",
                )
            print(
                f"Validation for epoch {epoch}: "
                f"val_rel_l2={val['val_rel_l2'] * 100:.4f}% "
                f"val_rmse_K={val['val_rmse_K']:.4f}K "
                f"(best={best_val * 100:.4f}%)"
                + ("  [new best -> cvit_best.pt]" if is_best else ""),
                flush=True,
            )
            print(
                f"  t0_rel_l2={val['t0_rel_l2'] * 100:.2f}%  "
                f"rel_l2[e/m/l]={val['rel_l2_early'] * 100:.1f}/"
                f"{val['rel_l2_mid'] * 100:.1f}/{val['rel_l2_late'] * 100:.1f}%  "
                f"rel_dev[e/m/l]={val['rel_dev_early'] * 100:.1f}/"
                f"{val['rel_dev_mid'] * 100:.1f}/{val['rel_dev_late'] * 100:.1f}%  "
                f"rmse_K[e/m/l]={val['rmse_K_early']:.3f}/"
                f"{val['rmse_K_mid']:.3f}/{val['rmse_K_late']:.3f}K",
                flush=True,
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
    "_ic_loss",
    "_curriculum_weights",
    "_causal_weights",
    "_causal_residual_loss",
    "_bin_residual",
    "_term_grad_norms",
    "build_gradnorm",
    "validate_rel_l2",
    "build_cvit",
    "run_one_seed_pino",
    "run_config_seeds_pino",
]
