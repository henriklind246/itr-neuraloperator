from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from data.dataset import (
    compute_global_stats,
    load_ramp_seconds,
    load_sim_data,
    load_solver_dt,
    problem_from_config,
    split_sim_ids,
)
from problems.diffusion import T_RIGHT
from problems.diffusion_forcing import K_SLAB
from problems.forcing import A_AMP_REF
from src.operators.cvit import CViT, ForcingCViT
from src.operators.train import (
    GradNormBalancer,
    build_optimizer,
    build_scheduler,
    load_config,
    set_seed,
)
from src.operators.utils import resolve_device
from src.physics.boundary_forcing import (
    SPATIAL_SAMPLERS,
    TEMPORAL_SAMPLERS,
    default_ramp_seconds,
    reconstruct_qL,
    sample_spatial_family,
    sample_temporal_family,
)
from src.physics.pde_residual import (
    diffusion_residual,
    forcing_neumann_residual,
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


def _lhs_unit(n: int, dim: int, device, generator) -> torch.Tensor:
    """Latin-hypercube sample in the unit cube; (n, dim) in [0, 1).

    Each column is a stratified permutation of ``n`` equal bins with a uniform
    jitter inside the bin, so the marginal coverage of every axis is uniform for
    any ``n`` (broad per-iteration domain coverage; Chen et al. arXiv:2606.06164).
    Columns are permuted independently so the joint sample decorrelates.
    """
    cols = []
    for _ in range(dim):
        perm = torch.randperm(n, device=device, generator=generator).to(torch.float32)
        jitter = torch.rand(n, device=device, generator=generator)
        cols.append((perm + jitter) / float(n))
    return torch.stack(cols, dim=-1)


def _lhs_leaf(col: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Shape an LHS column (n,) into a (1, n, 1) leaf with requires_grad."""
    return (col.reshape(1, -1, 1) * scale).detach().requires_grad_(True)


def _collocation_bias_counts(
    n_r: int, wall_frac: float, lead_frac: float
) -> tuple[int, int, int]:
    """Interior sub-population sizes (n_wall, n_lead, n_full) summing to n_r.

    ``n_full`` absorbs the integer-rounding remainder so the total is exactly
    ``n_r`` (count-preserving). Pure integer arithmetic; tested in isolation.
    """
    n_wall = int(wall_frac * n_r)
    n_lead = int(lead_frac * n_r)
    n_full = n_r - n_wall - n_lead
    return n_wall, n_lead, n_full


def _validate_collocation_bias(bias: dict) -> tuple[float, float, float, float]:
    """Validate the collocation-bias spec and return (wall_frac, wall_x_cut,
    lead_frac, lead_t_lo). Raises ``ValueError`` on any out-of-range / non-finite
    value; the ``wall_frac + lead_frac <= 1`` guard prevents a negative n_full.
    """
    wall_frac = float(bias.get("wall_frac", 0.0))
    wall_x_cut = float(bias.get("wall_x_cut", 0.333))
    lead_frac = float(bias.get("lead_frac", 0.0))
    lead_t_lo = float(bias.get("lead_t_lo", 0.5))
    for name, v in (
        ("wall_frac", wall_frac), ("lead_frac", lead_frac),
        ("wall_x_cut", wall_x_cut), ("lead_t_lo", lead_t_lo),
    ):
        if not math.isfinite(v):
            raise ValueError(f"collocation_bias.{name} must be finite, got {v}")
    if not 0.0 <= wall_frac <= 1.0:
        raise ValueError(f"collocation_bias.wall_frac must be in [0,1], got {wall_frac}")
    if not 0.0 <= lead_frac <= 1.0:
        raise ValueError(f"collocation_bias.lead_frac must be in [0,1], got {lead_frac}")
    if wall_frac + lead_frac > 1.0:
        raise ValueError(
            "collocation_bias.wall_frac + lead_frac must be <= 1, got "
            f"{wall_frac} + {lead_frac}"
        )
    if not 0.0 < wall_x_cut <= 1.0:
        raise ValueError(f"collocation_bias.wall_x_cut must be in (0,1], got {wall_x_cut}")
    if not 0.0 <= lead_t_lo < 1.0:
        raise ValueError(f"collocation_bias.lead_t_lo must be in [0,1), got {lead_t_lo}")
    return wall_frac, wall_x_cut, lead_frac, lead_t_lo


def _resolve_collocation_bias(pino_cfg: dict) -> dict | None:
    """Build the validated collocation-bias spec from the ``training.pino`` config,
    or ``None`` when disabled. Returns ``None`` unless ``collocation_bias.enabled``
    is truthy AND at least one of ``wall_frac`` / ``lead_frac`` is > 0 (so the
    default and the enabled-but-zero cases both keep the legacy RNG-identical
    draw). Validates eagerly, so a bad enabled spec raises before training.
    """
    bias_cfg = pino_cfg.get("collocation_bias", {}) or {}
    if not bias_cfg.get("enabled"):
        return None
    wall_frac = float(bias_cfg.get("wall_frac", 0.0))
    lead_frac = float(bias_cfg.get("lead_frac", 0.0))
    if wall_frac <= 0.0 and lead_frac <= 0.0:
        return None
    coll_bias = {
        "wall_frac": wall_frac,
        "wall_x_cut": float(bias_cfg.get("wall_x_cut", 0.333)),
        "lead_frac": lead_frac,
        "lead_t_lo": float(bias_cfg.get("lead_t_lo", 0.5)),
    }
    _validate_collocation_bias(coll_bias)
    return coll_bias


def _biased_interior(
    n_r: int, t_final: float, device, generator, lhs: bool, bias: dict
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Interior x/y/t leaves with a wall/lead/full mixture (static collocation
    biasing). Each subgroup is drawn WITHOUT gradient (``_lhs_unit`` or
    ``torch.rand``); the biased axis is rescaled into its band; the three groups
    are concatenated (order wall->lead->full) and ONLY THEN made leaves via a
    single ``requires_grad_`` per axis. Subgroup-wise LHS (each group stratified
    in its own cube), not one global Latin hypercube.
    """
    wall_frac, wall_x_cut, lead_frac, lead_t_lo = _validate_collocation_bias(bias)
    n_wall, n_lead, n_full = _collocation_bias_counts(n_r, wall_frac, lead_frac)
    lead_lo = lead_t_lo * t_final
    lead_span = t_final - lead_lo

    def _draw(n: int, dim: int) -> torch.Tensor:
        if n == 0:
            return torch.empty((n, dim), device=device)
        if lhs:
            return _lhs_unit(n, dim, device, generator)
        return torch.rand((n, dim), device=device, generator=generator)

    xs, ys, ts = [], [], []
    # wall group: x in [0, wall_x_cut]; y, t full range.
    wall = _draw(n_wall, 3)
    xs.append(wall[:, 0] * wall_x_cut); ys.append(wall[:, 1]); ts.append(wall[:, 2] * t_final)
    # lead group: t in [lead_lo, t_final]; x, y full range.
    lead = _draw(n_lead, 3)
    xs.append(lead[:, 0]); ys.append(lead[:, 1]); ts.append(lead_lo + lead[:, 2] * lead_span)
    # full group: legacy full cube.
    full = _draw(n_full, 3)
    xs.append(full[:, 0]); ys.append(full[:, 1]); ts.append(full[:, 2] * t_final)

    x_r = torch.cat(xs).reshape(1, n_r, 1).requires_grad_(True)
    y_r = torch.cat(ys).reshape(1, n_r, 1).requires_grad_(True)
    t_r = torch.cat(ts).reshape(1, n_r, 1).requires_grad_(True)
    return x_r, y_r, t_r


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
    sampler: str = "uniform",
    bias: dict | None = None,
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

    ``bias`` (optional) applies static, diagnostic-informed nonuniform biasing to
    the ``n_r`` interior points only (an implicit residual reweighting; total
    count preserved). When ``None`` — or when both ``wall_frac`` and ``lead_frac``
    are <= 0 — the legacy interior draw runs untouched (RNG-identical). IC/wall
    blocks are never biased.
    """
    lhs = sampler == "lhs"

    _wf = float(bias.get("wall_frac", 0.0)) if bias is not None else 0.0
    _lf = float(bias.get("lead_frac", 0.0)) if bias is not None else 0.0
    _biased = bias is not None and (_wf > 0.0 or _lf > 0.0)

    # interior: x, y ~ U(0,1); t ~ U(0, t_final). LHS stratifies the (x, y, t)
    # cube jointly for broader per-iteration coverage.
    if _biased:
        x_r, y_r, t_r = _biased_interior(
            n_r, t_final, device, generator, lhs, bias
        )
    elif lhs:
        cube = _lhs_unit(n_r, 3, device, generator)
        x_r = _lhs_leaf(cube[:, 0])
        y_r = _lhs_leaf(cube[:, 1])
        t_r = _lhs_leaf(cube[:, 2], scale=t_final)
    else:
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
        if lhs:
            wc = _lhs_unit(n_bc, 2, device, generator)
            free = _lhs_leaf(wc[:, 0])
            tw = _lhs_leaf(wc[:, 1], scale=t_final)
        else:
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


# --------- online forcing sampling (physics-only, no saved sim_params) ---------

def sample_forcing_params(
    rng: np.random.Generator,
    n: int,
    dt: float,
    t_final: float,
    *,
    c: float = 0.0,
    d: float = 1.0,
    temporal_window: dict | None = None,
) -> list[dict]:
    """Draw ``n`` fresh separable-forcing parameter sets ``q_L = a(t)*s(y)``.

    Uses the exact ProblemSpec samplers (``sample_temporal_family`` /
    ``sample_spatial_family`` + ``TEMPORAL_SAMPLERS`` / ``SPATIAL_SAMPLERS``) so
    the online training distribution matches the FV validation set. No
    ``sim_params.npy`` is read: physics-only training is not tied to a finite
    saved set (Chen et al. arXiv:2606.06164). ``c, d`` bound the y-domain the
    spatial profile lives on; ``temporal_window`` supplies the sin on/off window.
    """
    tw = temporal_window or {}
    win = dict(
        t_on=float(tw.get("t_on", 0.0)),
        t_off=float(tw.get("t_off", 0.2)),
        phase=float(tw.get("phase", 0.0)),
        tukey_alpha=float(tw.get("tukey_alpha", 0.5)),
    )
    out: list[dict] = []
    for _ in range(int(n)):
        tf = sample_temporal_family(rng)
        tp = TEMPORAL_SAMPLERS[tf](rng, dt=dt, t_final=t_final, **win)
        sf = sample_spatial_family(rng)
        sp = SPATIAL_SAMPLERS[sf](rng, c=c, d=d)
        out.append({
            "temporal_family": tf, "temporal_params": tp,
            "spatial_family": sf, "spatial_params": sp,
        })
    return out


def build_forcing_image(
    params: list[dict],
    y_img: np.ndarray,
    t_img: np.ndarray,
    a_ref: float,
    device: torch.device,
    t_ramp: float,
) -> torch.Tensor:
    """Encoder input: the forcing rendered as a space-time image; (B, 1, Ny, Nt).

    ``u_enc(y, t) = q_L(y, t) / a_ref`` with ``q_L`` reconstructed by the SAME
    ``reconstruct_qL`` helper the left-wall residual uses, so the encoder image
    and the residual forcing can never diverge. Axes are (rows = y, cols = t).
    """
    B = len(params)
    Ny, Nt = int(y_img.shape[0]), int(t_img.shape[0])
    img = np.empty((B, 1, Ny, Nt), dtype=np.float32)
    for b, p in enumerate(params):
        q_image, _ = reconstruct_qL(
            p["temporal_family"], p["temporal_params"],
            p["spatial_family"], p["spatial_params"], t_ramp=t_ramp,
        )
        img[b, 0] = np.asarray(q_image(y_img, t_img), dtype=np.float32) / float(a_ref)
    return torch.from_numpy(img).to(device)


def left_wall_qL(
    params: list[dict],
    y_pts: torch.Tensor,
    t_pts: torch.Tensor,
    device: torch.device,
    t_ramp: float,
) -> torch.Tensor:
    """Precomputed inward flux ``q_L(y_w, t_w)`` at the left-wall points; (B, N, 1).

    ``y_pts`` / ``t_pts`` are the shared left-wall collocation leaves (any shape
    with ``N`` elements); ``q_L`` is evaluated per sim via the shared
    ``reconstruct_qL`` ``q_at`` at exactly those points. Returned detached (the
    residual only needs autograd through ``dT/dx``, never through ``q_L``).
    """
    yv = y_pts.detach().reshape(-1).cpu().numpy()
    tv = t_pts.detach().reshape(-1).cpu().numpy()
    B, N = len(params), int(yv.shape[0])
    q = np.empty((B, N), dtype=np.float32)
    for b, p in enumerate(params):
        _, q_at = reconstruct_qL(
            p["temporal_family"], p["temporal_params"],
            p["spatial_family"], p["spatial_params"], t_ramp=t_ramp,
        )
        q[b] = np.asarray(q_at(yv, tv), dtype=np.float32)
    return torch.from_numpy(q).unsqueeze(-1).to(device)


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
    den: torch.Tensor | None = None,
) -> torch.Tensor:
    """IC term from the raw IC residual ``ic = T_hat_norm(0) - ic_target``.

    ``mode="mse"`` (legacy): plain ``mean(ic**2)`` -- a raw normalized MSE that
    goes numerically small on a near-300 K field even when the *relative* error
    in the nonconstant thermal signal is large (t0 rel-L2 ~0.75 on diffusion, the
    IC that seeds the whole trajectory).

    ``mode="rel"`` with ``den=None`` (legacy sampled): per-sim relative L2 with a
    *sampled* denominator ``sum_i (ic_target_i - t_right_tilde)**2``. This is the
    unstable form -- when the sampled IC nodes land in near-flat regions the
    denominator collapses and the ratio explodes, corrupting SOAP's second-moment
    preconditioner (observed loss_ic spikes to >2000).

    ``mode="rel"`` with ``den`` provided (stabilized full-grid): the denominator
    is a *fixed per-sim signal norm* precomputed on the whole grid,
    ``D_sim = mean_{x,y}[(T0_tilde - t_right_tilde)**2]`` (already floored by the
    caller). The numerator is the per-point mean squared IC residual over the
    sampled nodes, so numerator and denominator are both per-point mean-squares
    (dimensionally consistent regardless of ``n_ic``) and the ratio is bounded by
    the floor. ``den`` has shape ``(B,)``.
    """
    if mode == "rel":
        if den is not None:
            num = (ic ** 2).mean(dim=(1, 2))               # (B,) per-point MS
            return (num / den).mean()
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
    ic_den: torch.Tensor | None = None,
    compute_r: bool = True,
    compute_bc: bool = True,
    left_qL: torch.Tensor | None = None,
    sigma: float = 1.0,
    k_slab: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Raw physics/IC/BC losses. ``causal_cfg.enabled`` swaps the plain
    ``mean(r^2)`` interior term for a causally time-weighted one (needs
    ``t_final``); ``res_bins > 0`` also returns a detached ``res_bins`` per-time
    residual vector for diagnostics. ``ic_loss="rel"`` (with ``t_right_tilde``)
    swaps the raw IC MSE for the per-sim relative IC L2 (see ``_ic_loss``);
    passing ``ic_den`` (shape ``(B,)``) selects the stabilized fixed full-grid
    denominator path. ``compute_r=False`` / ``compute_bc=False`` skip the
    (expensive, double-backward) residual / BC autodiff entirely and return a
    zero placeholder -- used for the IC-only diagnostic. All default off/on ->
    legacy behavior.
    """
    bin_mean = None
    r = None
    if compute_r:
        x_r, y_r, t_r = batch["interior"]
        r = diffusion_residual(model, u, x_r, y_r, t_r, alpha=alpha)
        causal_on = bool(causal_cfg and causal_cfg.get("enabled", False))
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
    else:
        loss_r = u.new_zeros(())

    ic = ic_residual(model, u, batch["ic"]["coords"], batch["ic"]["t"], ic_target)
    loss_ic = _ic_loss(
        ic, ic_target, mode=ic_loss, t_right_tilde=t_right_tilde, eps=ic_eps,
        den=ic_den,
    )

    loss_bc_left = u.new_zeros(())
    loss_bc_hom = u.new_zeros(())
    if compute_bc:
        bc_sq = 0.0
        hom_sq = 0.0
        n_hom = 0
        for w in WALLS:
            xw, yw, tw = batch["walls"][w]
            if w == "left" and left_qL is not None:
                # Inhomogeneous forcing residual dT_tilde/dx + q_L/(k*sigma); the
                # left wall is now the actual forcing signal (top/bottom stay
                # homogeneous adiabatic). q_L is a precomputed constant tensor.
                nb = forcing_neumann_residual(
                    model, u, xw, yw, tw, left_qL, sigma, k=k_slab,
                )
                loss_bc_left = (nb ** 2).mean()
                bc_sq = bc_sq + loss_bc_left
            else:
                nb = neumann_residual(model, u, xw, yw, tw, w)
                wall_sq = (nb ** 2).mean()
                bc_sq = bc_sq + wall_sq
                hom_sq = hom_sq + wall_sq
                n_hom += 1
        loss_bc = bc_sq / len(WALLS)
        if n_hom > 0:
            loss_bc_hom = hom_sq / n_hom
    else:
        loss_bc = u.new_zeros(())

    out = {
        "r": loss_r,
        "ic": loss_ic,
        "bc": loss_bc,
        "bc_left": loss_bc_left,
        "bc_hom": loss_bc_hom,
    }
    if res_bins and r is not None:
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


def build_gradnorm(
    config: dict,
    lam_r: float | None = None,
    lam_ic: float | None = None,
    lam_bc: float | None = None,
    *,
    term_weights: dict[str, float] | None = None,
):
    """GradNormBalancer over the active PINO terms, or None when disabled.

    Reuses the FNO path's balancer (inverse gradient-norm multipliers, EMA
    smoothed). ``term_weights`` gives an explicit ordered term -> static-weight map
    (its insertion order fixes ``term_names``); the forcing trainer passes the split
    set ``{r, ic, bc_left, bc_hom}`` so the left forcing wall is isolated from the
    near-satisfied adiabatic walls. When ``term_weights`` is None the legacy generic
    set ``{r, ic, bc}`` is used. A term is included when its static weight > 0 (an
    IC-first curriculum may zero ``r``/``bc`` early, but those still ramp on later,
    so they belong in ``term_names``). Opt-in ``w_min`` / ``w_max`` / ``floor``
    guardrails are threaded through from ``config["training"]["gradnorm"]``.
    """
    gn_cfg = config["training"].get("gradnorm", {}) or {}
    if not bool(gn_cfg.get("enabled", False)):
        return None
    if term_weights is None:
        term_weights = {"r": lam_r, "ic": lam_ic, "bc": lam_bc}
    terms = [n for n, w in term_weights.items() if w is not None and w > 0.0]
    floor_cfg = gn_cfg.get("floor", {}) or {}
    return GradNormBalancer(
        terms,
        alpha_w=float(gn_cfg.get("alpha_w", 0.9)),
        update_every=int(gn_cfg.get("update_every", 10)),
        eps=float(gn_cfg.get("eps", 1.0e-8)),
        w_min=None if gn_cfg.get("w_min") is None else float(gn_cfg["w_min"]),
        w_max=None if gn_cfg.get("w_max") is None else float(gn_cfg["w_max"]),
        floors={str(k): float(v) for k, v in dict(floor_cfg).items()},
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


def _term_grad_cosines(
    terms: dict[str, torch.Tensor],
    params,
    pairs: list[tuple[str, str]] | None = None,
    *,
    tiny: float = 1.0e-12,
) -> dict[str, float | None]:
    """Pairwise cosine similarity of raw-loss gradients over ``params``.

    Diagnostic: a near-zero cosine means the terms are mostly a MAGNITUDE problem
    (GradNorm-friendly); a strongly negative cosine means a DIRECTION conflict that
    GradNorm alone will not resolve. Dot products and squared norms are accumulated
    tensorwise (never concatenating one giant vector). A pair maps to ``None`` when
    either gradient norm < ``tiny`` (e.g. a near-satisfied ``bc_hom``) so the log
    carries an honest null instead of a NaN or an artificial 0. Uses
    ``retain_graph=True``; call BEFORE the caller's ``loss.backward()``.
    """
    params = [p for p in params if p.requires_grad]
    grads = {
        name: torch.autograd.grad(
            term, params, retain_graph=True, allow_unused=True,
        )
        for name, term in terms.items()
    }

    def _sq(g):
        s = None
        for t in g:
            if t is None:
                continue
            c = t.detach().pow(2).sum()
            s = c if s is None else s + c
        return s

    def _dot(ga, gb):
        s = None
        for a, b in zip(ga, gb):
            if a is None or b is None:
                continue
            c = (a.detach() * b.detach()).sum()
            s = c if s is None else s + c
        return s

    if pairs is None:
        keys = list(terms.keys())
        pairs = [
            (keys[i], keys[j])
            for i in range(len(keys))
            for j in range(i + 1, len(keys))
        ]
    sq_norms = {name: _sq(g) for name, g in grads.items()}
    out: dict[str, float | None] = {}
    for a, b in pairs:
        key = f"{a}|{b}"
        sa, sb = sq_norms.get(a), sq_norms.get(b)
        if sa is None or sb is None:
            out[key] = None
            continue
        na, nb = float(torch.sqrt(sa)), float(torch.sqrt(sb))
        if na < tiny or nb < tiny:
            out[key] = None
            continue
        d = _dot(grads[a], grads[b])
        out[key] = None if d is None else float(d) / (na * nb)
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


@torch.no_grad()
def validate_forcing_gnrmse(
    model: CViT,
    data: dict[str, Any],
    ids: np.ndarray,
    sim_params: np.ndarray,
    y_img: np.ndarray,
    t_img: np.ndarray,
    a_ref: float,
    t_ramp: float,
    device: torch.device,
    query_batch: int = 8,
) -> dict[str, float]:
    """Deviation-field globally-normalized RMSE over held-out forcing sims.

    The single-slab forcing field sits on a fixed 300 K baseline, so a raw
    rel-L2 denominator is dominated by that baseline (a constant-300 prediction
    scores deceptively well) and a per-sim normalized denominator explodes on
    weak forcings. This judges on the deviation field ``T - T_RIGHT`` with the
    FROZEN global ``sigma``:

        gnrmse = sqrt(mean_pts[(T_pred - T_true) ** 2]) / sigma_global

    (``T_RIGHT`` cancels in the deviation error, so this is ``rmse_K / sigma``),
    which is amplitude-fair. Per-sim results are stratified by forcing-amplitude
    tercile and by ``temporal_family`` so weak-forcing and pulse-family failures
    stay visible rather than averaged away. ``sim_params`` are the per-sim
    forcing records; each sim's encoder image is reconstructed by the SAME
    ``build_forcing_image`` / ``reconstruct_qL`` path the trainer uses.
    """
    x_grid = torch.as_tensor(data["x_grid"], dtype=torch.float32, device=device)
    y_grid = torch.as_tensor(data["y_grid"], dtype=torch.float32, device=device)
    t_grid = np.asarray(data["t_grid"], dtype=np.float64)
    mu, sigma = data["mu_global"], data["sigma_global"]
    Nx, Ny, Nt = x_grid.numel(), y_grid.numel(), len(t_grid)

    gx, gy = torch.meshgrid(x_grid, y_grid, indexing="ij")
    mesh = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1).unsqueeze(0)
    # For hard_left_flux the ansatz carries the forcing via the analytic term
    # g*(x-1), g = -q_L(y,t)/(k*sigma). It is dropped when q_left is None, so the
    # eval MUST feed q_L at every query point or it scores an insulated wall
    # (T_x(0)=raw_x(0)~0) and looks catastrophically wrong regardless of family.
    use_left_flux = bool(getattr(model, "hard_left_flux", False))
    y_grid_np = np.asarray(data["y_grid"], dtype=np.float64)
    M = Nx * Ny

    ids = np.asarray(ids)
    per_sim_rmse: list[float] = []
    per_sim_amp: list[float] = []
    per_sim_fam: list[str] = []
    for start in range(0, len(ids), query_batch):
        chunk = ids[start:start + query_batch]
        params = [dict(sim_params[int(i)]) for i in chunk]
        u = build_forcing_image(params, y_img, t_img, a_ref, device, t_ramp)
        B = u.shape[0]
        coords = mesh.expand(B, -1, -1)
        q_all = None
        if use_left_flux:
            # g depends on (y, t) only, so evaluate q_L on the distinct (Ny x Nt)
            # grid with the vectorized q_image (one reconstruct per sim), then tile
            # across x into mesh order (flat index = ix*Ny + iy, y fastest). This
            # avoids the Nx-redundant pointwise Nt*Nx*Ny evaluation.
            q_np = np.empty((B, Nt, M), dtype=np.float32)
            for b, p in enumerate(params):
                q_image, _ = reconstruct_qL(
                    p["temporal_family"], p["temporal_params"],
                    p["spatial_family"], p["spatial_params"], t_ramp=t_ramp,
                )
                qg = np.asarray(q_image(y_grid_np, t_grid), dtype=np.float32)  # (Ny,Nt)
                q_np[b] = np.tile(qg.T, (1, Nx))  # (Nt, M), y fastest
            q_all = torch.from_numpy(q_np).unsqueeze(-1).to(device)  # (B,Nt,M,1)
        pred = torch.empty((B, Nt, Nx, Ny), device=device)
        for k in range(Nt):
            tk = torch.full((B, Nx * Ny, 1), float(t_grid[k]), device=device)
            q_left = q_all[:, k] if use_left_flux else None
            out = model(u, coords, tk, q_left=q_left)
            pred[:, k] = out[..., 0].view(B, Nx, Ny)
        pred_K = pred * sigma + mu
        truth = np.asarray(data["trajectories"][chunk], dtype=np.float32)
        truth_t = torch.from_numpy(truth).to(device)
        se = ((pred_K - truth_t) ** 2).sum(dim=(1, 2, 3))  # (B,)
        rmse = torch.sqrt(se / float(Nt * Nx * Ny)).detach().cpu().numpy()
        for b, p in enumerate(params):
            per_sim_rmse.append(float(rmse[b]))
            q_image, _ = reconstruct_qL(
                p["temporal_family"], p["temporal_params"],
                p["spatial_family"], p["spatial_params"], t_ramp=t_ramp,
            )
            per_sim_amp.append(float(np.max(np.abs(q_image(y_img, t_img)))))
            per_sim_fam.append(str(p.get("temporal_family", "")))

    rmse_arr = np.asarray(per_sim_rmse, dtype=np.float64)
    amp_arr = np.asarray(per_sim_amp, dtype=np.float64)
    fam_arr = np.asarray(per_sim_fam)
    gnrmse = rmse_arr / (float(sigma) + 1e-8)

    out = {
        "val_gnrmse": float(gnrmse.mean()),
        "val_rmse_K": float(rmse_arr.mean()),
    }
    if len(amp_arr) >= 3:
        q1, q2 = np.quantile(amp_arr, [1.0 / 3.0, 2.0 / 3.0])
        strata = {
            "amp_low": amp_arr <= q1,
            "amp_mid": (amp_arr > q1) & (amp_arr <= q2),
            "amp_high": amp_arr > q2,
        }
        for name, mask in strata.items():
            out[f"gnrmse_{name}"] = (
                float(gnrmse[mask].mean()) if mask.any() else float("nan")
            )
    for fam in np.unique(fam_arr):
        out[f"gnrmse_fam_{fam}"] = float(gnrmse[fam_arr == fam].mean())
    return out


def forcing_data_loss(
    model: CViT,
    sim_params: np.ndarray,
    trajectories: np.ndarray,
    ids: np.ndarray,
    *,
    y_img: np.ndarray,
    t_img: np.ndarray,
    a_ref: float,
    t_ramp: float,
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    t_grid: np.ndarray,
    mu: float,
    sigma: float,
    n_sims: int,
    n_pts: int,
    device: torch.device,
    rng: np.random.Generator,
    gen: torch.Generator,
) -> torch.Tensor:
    """Supervised interior-field MSE against the saved FV forcing trajectories.

    Lever #1 for the diffusion_forcing gap. The pure-physics forcing objective
    couples ``q_L`` to the trainable field ONLY through the single soft left-wall
    Neumann residual, which competes with the flat-``T=300 K`` basin that
    minimizes every other term (interior residual, IC, homogeneous walls). This
    draws ``n_sims`` saved TRAIN sims, reconstructs each encoder image with the
    SAME :func:`build_forcing_image` path the physics batch uses, and matches
    ``model(u, x, y, t)`` to the FV field at ``n_pts`` grid nodes shared across
    the sim batch (normalized-space MSE). It is a plain value penalty -- no
    autodiff through the output -- so it directly pins the field level the
    derivative BC cannot, giving the reference benchmarks' value-supervised
    conditioning. Intended for the soft path (``hard_left_flux=false``); the
    ansatz's analytic forcing term is not needed, so ``q_left`` is left ``None``.
    """
    ids = np.asarray(ids)
    k = min(int(n_sims), int(len(ids)))
    chunk = rng.choice(ids, size=k, replace=False)
    params = [dict(sim_params[int(i)]) for i in chunk]
    u = build_forcing_image(params, y_img, t_img, a_ref, device, t_ramp)
    B = u.shape[0]
    Nx, Ny, Nt = int(x_grid.numel()), int(y_grid.numel()), int(len(t_grid))

    it = torch.randint(0, Nt, (n_pts,), device=device, generator=gen)
    ix = torch.randint(0, Nx, (n_pts,), device=device, generator=gen)
    iy = torch.randint(0, Ny, (n_pts,), device=device, generator=gen)
    coords = torch.stack([x_grid[ix], y_grid[iy]], dim=-1).view(1, n_pts, 2)
    coords = coords.expand(B, -1, -1)
    it_np = it.detach().cpu().numpy()
    ix_np = ix.detach().cpu().numpy()
    iy_np = iy.detach().cpu().numpy()
    t_np = np.asarray(t_grid, dtype=np.float32)[it_np]
    t = torch.from_numpy(t_np).to(device).view(1, n_pts, 1).expand(B, -1, -1)

    traj = np.asarray(trajectories[np.asarray(chunk)], dtype=np.float32)  # (B,Nt,Nx,Ny)
    truth = traj[:, it_np, ix_np, iy_np]  # (B, n_pts)
    truth = (truth - float(mu)) / (float(sigma) + 1e-8)
    truth_t = torch.from_numpy(truth).to(device).unsqueeze(-1)  # (B, n_pts, 1)

    pred = model(u, coords, t)  # (B, n_pts, 1)
    return ((pred - truth_t) ** 2).mean()


# --------- single-seed training ---------

def build_cvit(
    config: dict,
    mu: float,
    sigma: float,
    grid_size: tuple[int, int],
    t_final: float = 1.0,
    variant: str = "cvit",
) -> CViT:
    """Construct the PINO surrogate. ``variant="cvit"`` (default) builds the
    diffusion :class:`CViT` conditioned on the IC field over ``grid_size =
    (Nx, Ny)``. ``variant="forcing"`` builds a :class:`ForcingCViT` whose encoder
    ingests the forcing space-time image over ``grid_size = (Ny_img, Nt_img)``;
    it reads ``model.forcing_cvit`` when present, falling back to ``model.cvit``.
    """
    if variant == "forcing":
        c = {**config["model"]["cvit"], **config["model"].get("forcing_cvit", {})}
    else:
        c = config["model"]["cvit"]
    t_right_K = float(c.get("hard_right_dirichlet_t_right", T_RIGHT))
    hard_rd = bool(c.get("hard_right_dirichlet", True))
    t_right_tilde = (t_right_K - mu) / (sigma + 1e-8) if hard_rd else 0.0
    # Decoder time normalization horizon. Default to the data-derived t_final so
    # the temporal Fourier features live on [0, 1]; a config override wins.
    t_norm = float(c.get("t_final", None) if c.get("t_final", None) is not None else t_final)
    # Hard left-flux lifting (opt-in): converts the inward flux q_L into the
    # normalized slope it must produce via left_flux_scale = 1/(k*sigma). Only
    # meaningful for the forcing variant, which carries the q_L signal.
    hard_lf = bool(c.get("hard_left_flux", False))
    left_flux_scale = 1.0 / (float(K_SLAB) * (float(sigma) + 1e-8))
    cls = ForcingCViT if variant == "forcing" else CViT
    return cls(
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
        t_final=t_norm,
        hard_left_flux=hard_lf,
        left_flux_scale=left_flux_scale,
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

    model = build_cvit(config, mu, sigma, grid_size=(Nx, Ny), t_final=t_final).to(device)
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

    # Relative-IC denominator source. "sampled" (legacy): per-step sum over the
    # sampled IC nodes -- collapses when nodes land in the near-flat interior,
    # blowing the ratio up and corrupting the optimizer's second moment.
    # "full_grid": a FIXED per-sim signal norm D_sim = mean_{x,y}[(T0_tilde -
    # t_right_tilde)**2] precomputed on the whole grid and floored at a fraction
    # of the train-set median, so the ratio is bounded and amplitude-invariant.
    ic_denom = str(pino.get("ic_denom", "sampled"))
    ic_denom_floor_frac = float(pino.get("ic_denom_floor_frac", 0.01))
    d_sim_all: torch.Tensor | None = None
    ic_floor = 0.0
    if ic_loss == "rel" and ic_denom == "full_grid":
        T0_tilde = (
            np.asarray(data["trajectories"][:, 0, :, :], dtype=np.float64) - mu
        ) / (sigma + 1e-8)
        dev0 = T0_tilde - float(t_right_tilde_ic)
        d_np = (dev0 ** 2).reshape(dev0.shape[0], -1).mean(axis=1)   # (S,)
        _train_ids = np.asarray(data["train_ids"])
        ic_floor = ic_denom_floor_frac * float(np.median(d_np[_train_ids]))
        d_np = np.maximum(d_np, ic_floor)
        d_sim_all = torch.as_tensor(d_np, dtype=torch.float32, device=device)

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

    # IC-only fast path: when a term can never carry weight (static lambda 0 and
    # no curriculum that could ramp it up), skip its autodiff entirely. Dropping
    # the residual removes the double-backward that dominates step cost, so the
    # IC-only diagnostic runs at interactive speed.
    _curr_on_flags = bool(curr_cfg.get("enabled", False))
    compute_r = _curr_on_flags or lam_r > 0.0
    compute_bc = _curr_on_flags or lam_bc > 0.0

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
        f"ic_loss={ic_loss} ic_denom={ic_denom}"
        + (f"(floor={ic_floor:.3e})" if d_sim_all is not None else "")
        + f" dense_ic={dense_ic} resample_every={resample_every} | "
        f"compute_r={compute_r} compute_bc={compute_bc} | "
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

        ic_den = None
        if d_sim_all is not None:
            ic_den = d_sim_all[torch.as_tensor(batch_ids, device=device)]

        optimizer.zero_grad(set_to_none=True)
        losses = pino_losses(
            model, u, coll, ic_target, alpha,
            causal_cfg=causal_cfg, t_final=t_final,
            res_bins=(res_n_bins if do_val else 0),
            ic_loss=ic_loss, t_right_tilde=t_right_tilde_ic, ic_eps=ic_eps,
            ic_den=ic_den, compute_r=compute_r, compute_bc=compute_bc,
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


def run_one_seed_forcing_pino(config: dict, seed: int, run_dir: Path) -> dict[str, Any]:
    """Physics-only training of a :class:`ForcingCViT` on the single-slab forcing
    benchmark. Run-0 recipe: static Adam weights, ONLINE forcing sampling, LHS
    collocation, inhomogeneous left-wall Neumann forcing, and a fixed 300 K IC
    anchor. No GradNorm / causal / SOAP (layer those in later, one per run).

    The encoder conditions on the forcing rendered as a ``(Ny_img, Nt_img)``
    space-time image. Each step samples FRESH forcing params and FRESH
    collocation, so training is not tied to a finite saved set (Chen et al.
    arXiv:2606.06164). Saved FV trajectories + ``sim_params`` are used for
    VALIDATION only, judged by the deviation-field gnRMSE with a frozen global
    ``sigma`` (see :func:`validate_forcing_gnrmse`).
    """
    set_seed(seed)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(config["training"].get("device", "auto"))
    data = load_diffusion_data(config)
    mu, sigma = data["mu_global"], data["sigma_global"]
    Nx, Ny = int(data["x_grid"].shape[0]), int(data["y_grid"].shape[0])
    t_final = float(data["t_grid"][-1])
    y_grid_np = np.asarray(data["y_grid"], dtype=np.float64)
    c_dom, d_dom = float(y_grid_np[0]), float(y_grid_np[-1])

    # Saved FV forcing records are VALIDATION-only. sim_params.npy lives beside
    # trajectories.npy (same directory the data generator writes to).
    sp_path = Path(config["data"]["trajectories.npy"]).parent / "sim_params.npy"
    sim_params = np.load(str(sp_path), allow_pickle=True)

    pino = config["training"]["pino"]
    fcfg = pino.get("forcing", {}) or {}
    # `null` in YAML resolves to a dynamic default here (Ny/A_AMP_REF/t_final are
    # not knowable statically), so coalesce None rather than trusting .get's
    # absent-key fallback.
    a_ref = float(fcfg.get("a_ref") if fcfg.get("a_ref") is not None else A_AMP_REF)
    ny_img = int(fcfg.get("ny_img") if fcfg.get("ny_img") is not None else Ny)
    nt_img = int(fcfg.get("nt_img") if fcfg.get("nt_img") is not None else 128)
    # Forcing image axes: rows = left-wall y-nodes, cols = time over [0, t_final].
    y_img = np.linspace(c_dom, d_dom, ny_img, dtype=np.float64)
    t_img = np.linspace(0.0, t_final, nt_img, dtype=np.float64)

    # Frozen startup ramp: pin an ABSOLUTE constant shared by the online training
    # q_L AND the FV-baked validation q_L so early-time forcing matches exactly.
    # Config override wins; else reuse the ramp stored with the dataset; else fall
    # back to the dt-derived default.
    ramp_cfg = fcfg.get("ramp_seconds", None)
    if ramp_cfg is not None:
        t_ramp = float(ramp_cfg)
    else:
        t_ramp = load_ramp_seconds(config["data"]["t_grid_path"])
        if t_ramp is None:
            dt = load_solver_dt(config["data"]["t_grid_path"])
            t_ramp = default_ramp_seconds(dt if dt is not None else t_final / 100.0)

    temporal_window = dict(
        t_on=float(fcfg.get("t_on", 0.0)),
        t_off=float(fcfg.get("t_off", 0.2)),
        phase=float(fcfg.get("phase", 0.0)),
        tukey_alpha=float(fcfg.get("tukey_alpha", 0.5)),
    )
    dt_sample = (
        float(fcfg["dt_sample"])
        if fcfg.get("dt_sample") is not None
        else t_final / max(nt_img - 1, 1)
    )
    sampler = str(fcfg.get("collocation") or "lhs")

    # Static, diagnostic-informed collocation biasing (implicit residual
    # reweighting of the interior points). Disabled by default -> coll_bias is
    # None, so sample_collocation runs the legacy RNG-identical interior draw.
    # The helper validates eagerly (raises before the training loop).
    coll_bias = _resolve_collocation_bias(pino)

    model = build_cvit(
        config, mu, sigma, grid_size=(ny_img, nt_img), t_final=t_final,
        variant="forcing",
    ).to(device)
    optimizer = build_optimizer(config, model.parameters())
    scheduler = build_scheduler(config, optimizer)

    lam_r = float(pino["lambda_r"])
    lam_ic = float(pino["lambda_ic"])
    lam_bc = float(pino["lambda_bc"])
    # Optional per-term forcing-wall weight. None (default) keeps the legacy
    # single bc bucket (left+top+bottom averaged, weighted by lam_bc). When set,
    # the forcing left wall is pulled out and weighted by lam_bc_left, while the
    # homogeneous top/bottom walls stay on lam_bc via the bc_hom bucket.
    lam_bc_left_cfg = pino.get("lambda_bc_left", None)
    lam_bc_left = None if lam_bc_left_cfg is None else float(lam_bc_left_cfg)
    # Lever #1: optional supervised interior-field data term drawn from the saved
    # TRAIN forcing sims. 0.0 (default) = pure physics (off).
    lam_data = float(pino.get("lambda_data", 0.0))
    n_data_sims = int(pino.get("n_data_sims", 8))
    n_data_pts = int(pino.get("n_data_pts", 1024))
    t_grid_np = np.asarray(data["t_grid"], dtype=np.float64)
    n_r = int(pino["n_r"])
    n_ic = int(pino["n_ic"])
    n_bc = int(pino["n_bc"])
    sim_batch = int(pino["sim_batch"])
    alpha = float(pino.get("alpha", 1.0))
    # Fixed uniform 300 K IC in normalized space (mu ~ 300 -> ~0).
    t_right_tilde_ic = (T_RIGHT - mu) / (sigma + 1e-8)

    # GradNorm balances the RAW per-term gradient magnitudes so the forcing
    # left-wall residual is not swamped by the interior PDE gradient (SOAP only
    # sees the combined gradient). The term set matches the loss composition: the
    # split {r, ic, bc_left, bc_hom} isolates the forcing wall from the
    # near-satisfied adiabatic walls; the legacy single-bucket set is {r, ic, bc}.
    # None (disabled) is an exact no-op (multipliers == 1). Guardrails come from
    # config["training"]["gradnorm"] (w_min / w_max / floor).
    if lam_bc_left is None:
        gn_term_weights = {"r": lam_r, "ic": lam_ic, "bc": lam_bc}
        gn_cos_pairs = None
    else:
        gn_term_weights = {
            "r": lam_r, "ic": lam_ic, "bc_left": lam_bc_left, "bc_hom": lam_bc,
        }
        gn_cos_pairs = [("r", "bc_left"), ("ic", "bc_left"), ("bc_hom", "bc_left")]
    gradnorm = build_gradnorm(config, term_weights=gn_term_weights)

    # Warm-up (5b): hold the forcing batch + collocation fixed and upweight the IC
    # anchor for the first ``warmup.epochs`` steps so the fixed T=300 solution is
    # learned before the PDE/BC residuals dominate. Run-0 defaults are inert.
    wcfg = fcfg.get("warmup", {}) or {}
    warmup_epochs = int(wcfg.get("epochs", 0))
    warmup_ic_mult = float(wcfg.get("ic_mult", 1.0))
    warmup_resample_every = max(1, int(wcfg.get("resample_every", 1)))

    epochs = int(config["training"]["epochs"])
    validate_every = int(config["training"].get("validate_every", 10))

    x_grid_t = torch.as_tensor(data["x_grid"], dtype=torch.float32, device=device)
    y_grid_t = torch.as_tensor(data["y_grid"], dtype=torch.float32, device=device)

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    rng = np.random.default_rng(seed)

    metrics_path = run_dir / "train_metrics.csv"
    # loss_bc_left is logged as its OWN column: the left wall is the forcing
    # signal and must be watchable independently of the homogeneous top/bottom.
    # GradNorm diagnostics keep the existing w_* columns at their STATIC meaning
    # and add explicit new columns for the split term set {r, ic, bc_left, bc_hom}:
    # gn_mult_* (multiplier), w_eff_* (static x multiplier == what SOAP sees),
    # grad_norm_* (raw per-term gradient norm) and grad_norm_eff_* (effective).
    # These + JSON supplements are populated on validation epochs only.
    gn_cols = ["r", "ic", "bc_left", "bc_hom"]
    fieldnames = [
        "epoch", "loss", "loss_r", "loss_ic", "loss_bc", "loss_bc_left",
        "loss_data",
        "w_r", "w_ic", "w_bc", "w_bc_left", "w_data",
        "val_gnrmse", "val_rmse_K",
        "gnrmse_amp_low", "gnrmse_amp_mid", "gnrmse_amp_high",
        *[f"gn_mult_{c}" for c in gn_cols],
        *[f"w_eff_{c}" for c in gn_cols],
        *[f"grad_norm_{c}" for c in gn_cols],
        *[f"grad_norm_eff_{c}" for c in gn_cols],
        "gradnorm_mean_mult", "gradnorm_ms",
        "gradnorm_weights", "grad_cosines", "gradnorm_bound_hits",
    ]
    with open(metrics_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore").writeheader()

    print(
        f"[pino-forcing] seed={seed} device={device} epochs={epochs} "
        f"validate_every={validate_every} | grid={Nx}x{Ny} t_final={t_final:.4f} | "
        f"img=({ny_img}x{nt_img}) a_ref={a_ref} t_ramp={t_ramp:.4g} "
        f"sampler={sampler} coll_bias={coll_bias} | lambda_r={lam_r} lambda_ic={lam_ic} "
        f"lambda_bc={lam_bc} lambda_bc_left={lam_bc_left} "
        f"lambda_data={lam_data} (n_data_sims={n_data_sims},n_data_pts={n_data_pts}) "
        f"hard_left_flux={bool(getattr(model, 'hard_left_flux', False))} | "
        f"n_r={n_r} n_ic={n_ic} n_bc={n_bc} "
        f"sim_batch={sim_batch} alpha={alpha} k_slab={K_SLAB} | "
        f"warmup(epochs={warmup_epochs},ic_mult={warmup_ic_mult},"
        f"resample_every={warmup_resample_every})",
        flush=True,
    )

    best_val = float("inf")
    history: list[dict[str, float]] = []
    coll = None
    params_batch: list[dict] | None = None
    for epoch in range(epochs):
        model.train()
        warming = epoch < warmup_epochs
        # Computed up front: the cosine diagnostic runs on the training step that
        # coincides with a validation epoch, before backward.
        do_val = (epoch % validate_every == 0) or (epoch == epochs - 1)
        re = warmup_resample_every if warming else 1
        # Online resampling: fresh forcing batch + fresh collocation. During
        # warm-up they are held for ``re`` steps to let the IC anchor settle.
        if params_batch is None or (epoch % re == 0):
            params_batch = sample_forcing_params(
                rng, sim_batch, dt_sample, t_final,
                c=c_dom, d=d_dom, temporal_window=temporal_window,
            )
            coll = sample_collocation(
                n_r, n_ic, n_bc, t_final, x_grid_t, y_grid_t, device, gen,
                sampler=sampler, bias=coll_bias,
            )

        u = build_forcing_image(params_batch, y_img, t_img, a_ref, device, t_ramp)
        B = u.shape[0]
        ic_target = torch.full(
            (B, coll["ic"]["coords"].shape[1], 1), float(t_right_tilde_ic),
            device=device,
        )
        _, yw_left, tw_left = coll["walls"]["left"]
        left_qL = left_wall_qL(params_batch, yw_left, tw_left, device, t_ramp)

        optimizer.zero_grad(set_to_none=True)
        losses = pino_losses(
            model, u, coll, ic_target, alpha,
            t_final=t_final, ic_loss="mse", t_right_tilde=t_right_tilde_ic,
            left_qL=left_qL, sigma=float(sigma), k_slab=K_SLAB,
        )
        w_ic = lam_ic * (warmup_ic_mult if warming else 1.0)
        # Static per-term weights ``sw`` (name -> lambda). Dict order preserves the
        # ORIGINAL left-to-right summation so the disabled path (gn_mults empty,
        # multiplier == 1.0) reproduces the previous ``loss`` exactly.
        if lam_bc_left is None:
            w_bc = lam_bc
            w_bc_left = lam_bc  # reported effective weight; left sits inside bc
            sw = {"r": lam_r, "ic": w_ic, "bc": lam_bc}
        else:
            # Split BC: homogeneous top/bottom on lam_bc, forcing left on its own
            # weight so the forcing residual is not diluted by the 1/len(WALLS)
            # bucket average.
            w_bc = lam_bc
            w_bc_left = lam_bc_left
            sw = {"r": lam_r, "ic": w_ic, "bc_hom": lam_bc, "bc_left": lam_bc_left}

        # GradNorm rebalances the RAW per-term gradient magnitudes BEFORE backward
        # (one extra autograd.grad per active term every ``update_every`` steps).
        # ``w_eff[k] = sw[k] * m_k`` -- static lambda priorities are preserved; the
        # multiplier equalizes the raw magnitudes SOAP sees. Disabled -> m_k == 1.
        gn_mults: dict[str, float] = {}
        gradnorm_ms: float | str = ""
        if gradnorm is not None:
            active = {k: losses[k] for k in gradnorm.term_names if k in losses}
            gn_params = [p for p in model.parameters() if p.requires_grad]
            if device.type == "cuda":
                torch.cuda.synchronize()
            _t0 = time.perf_counter()
            gn_mults = gradnorm.maybe_update(active, gn_params, dist_info=None)
            if device.type == "cuda":
                torch.cuda.synchronize()
            gradnorm_ms = (time.perf_counter() - _t0) * 1000.0
        # Cosine diagnostic on the decoder subset, only on the training step that
        # coincides with a validation epoch, on the LIVE graph before backward.
        grad_cosines: dict[str, float | None] | None = None
        if do_val and gradnorm is not None:
            grad_cosines = _term_grad_cosines(
                {k: losses[k] for k in sw}, model.decoder.parameters(),
                pairs=gn_cos_pairs,
            )
        w_eff = {k: sw[k] * float(gn_mults.get(k, 1.0)) for k in sw}
        loss = None
        for k, wk in w_eff.items():
            term = wk * losses[k]
            loss = term if loss is None else loss + term

        loss_data = None
        if lam_data > 0.0:
            loss_data = forcing_data_loss(
                model, sim_params, data["trajectories"], data["train_ids"],
                y_img=y_img, t_img=t_img, a_ref=a_ref, t_ramp=t_ramp,
                x_grid=x_grid_t, y_grid=y_grid_t, t_grid=t_grid_np,
                mu=mu, sigma=sigma, n_sims=n_data_sims, n_pts=n_data_pts,
                device=device, rng=rng, gen=gen,
            )
            loss = loss + lam_data * loss_data
        loss.backward()
        lr = float(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

        row: dict[str, Any] = {
            "epoch": epoch,
            "loss": float(loss.detach().cpu()),
            "loss_r": float(losses["r"].detach().cpu()),
            "loss_ic": float(losses["ic"].detach().cpu()),
            "loss_bc": float(losses["bc"].detach().cpu()),
            "loss_bc_left": float(losses["bc_left"].detach().cpu()),
            "loss_data": (
                float(loss_data.detach().cpu()) if loss_data is not None else ""
            ),
            "w_r": lam_r, "w_ic": w_ic, "w_bc": w_bc, "w_bc_left": w_bc_left,
            "w_data": lam_data,
            "val_gnrmse": "", "val_rmse_K": "",
            "gnrmse_amp_low": "", "gnrmse_amp_mid": "", "gnrmse_amp_high": "",
        }
        data_txt = (
            f", data={row['loss_data']:.6f}" if loss_data is not None else ""
        )
        print(
            f"Epoch {epoch}: loss={row['loss']:.6f} "
            f"(r={row['loss_r']:.6f}, ic={row['loss_ic']:.6f}, "
            f"bc={row['loss_bc']:.6f}, bc_left={row['loss_bc_left']:.6f}"
            f"{data_txt}) "
            f"lr={lr:.2e}" + ("  [warmup]" if warming else ""),
            flush=True,
        )

        if do_val:
            model.eval()
            val = validate_forcing_gnrmse(
                model, data, data["val_ids"], sim_params,
                y_img, t_img, a_ref, t_ramp, device,
            )
            row["val_gnrmse"] = val["val_gnrmse"]
            row["val_rmse_K"] = val["val_rmse_K"]
            for c in ("gnrmse_amp_low", "gnrmse_amp_mid", "gnrmse_amp_high"):
                row[c] = val.get(c, "")
            if gradnorm is not None:
                # GradNorm diagnostics logged on val epochs only. Raw per-term norms
                # come from the balancer's most-recent-update snapshot (fresh within
                # ``update_every``); effective = w_eff * raw is what SOAP sees.
                raw_norms = gradnorm.last_raw_norms
                for k in sw:
                    m = float(gn_mults.get(k, 1.0))
                    row[f"gn_mult_{k}"] = m
                    row[f"w_eff_{k}"] = w_eff[k]
                    g_raw = raw_norms.get(k)
                    if g_raw is not None:
                        row[f"grad_norm_{k}"] = g_raw
                        row[f"grad_norm_eff_{k}"] = w_eff[k] * g_raw
                row["gradnorm_mean_mult"] = gradnorm.last_mean_multiplier
                row["gradnorm_ms"] = gradnorm_ms
                row["gradnorm_weights"] = json.dumps(
                    {k: float(v) for k, v in gn_mults.items()}
                )
                row["gradnorm_bound_hits"] = json.dumps(gradnorm.bound_hit_counts)
                if grad_cosines is not None:
                    row["grad_cosines"] = json.dumps(grad_cosines)
            is_best = val["val_gnrmse"] < best_val
            if is_best:
                best_val = val["val_gnrmse"]
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "mu_global": mu, "sigma_global": sigma,
                        "config": config, "epoch": epoch, "best_val": best_val,
                        "gradnorm_state": (
                            gradnorm.state_dict() if gradnorm is not None else None
                        ),
                        "forcing_image": {
                            "ny_img": ny_img, "nt_img": nt_img,
                            "a_ref": a_ref, "t_ramp": t_ramp,
                            "c_dom": c_dom, "d_dom": d_dom, "t_final": t_final,
                        },
                    },
                    run_dir / "cvit_best.pt",
                )
            fam_txt = " ".join(
                f"{k.split('gnrmse_fam_')[1]}={v * 100:.2f}%"
                for k, v in val.items() if k.startswith("gnrmse_fam_")
            )
            print(
                f"Validation for epoch {epoch}: "
                f"val_gnrmse={val['val_gnrmse'] * 100:.4f}% "
                f"val_rmse_K={val['val_rmse_K']:.4f}K (best={best_val * 100:.4f}%)"
                + ("  [new best -> cvit_best.pt]" if is_best else ""),
                flush=True,
            )
            print(
                f"  gnrmse[amp low/mid/high]="
                f"{val.get('gnrmse_amp_low', float('nan')) * 100:.2f}/"
                f"{val.get('gnrmse_amp_mid', float('nan')) * 100:.2f}/"
                f"{val.get('gnrmse_amp_high', float('nan')) * 100:.2f}%  "
                f"fam[{fam_txt}]",
                flush=True,
            )

        history.append({k: (v if v != "" else None) for k, v in row.items()})
        with open(metrics_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore").writerow(row)

    summary = {"seed": seed, "best_val_gnrmse": best_val, "epochs": epochs}
    with open(run_dir / "final_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def run_config_seeds_pino(
    config: dict, base_run_dir: Path, seeds: list[int]
) -> dict[str, Any]:
    base_run_dir = Path(base_run_dir)
    # Dispatch on the benchmark: the single-slab forcing benchmark trains a
    # ForcingCViT physics-only on an online-sampled forcing image; every other
    # benchmark uses the IC-conditioned diffusion CViT path.
    bench = str(config.get("benchmark", {}).get("name", "diffusion"))
    runner = (
        run_one_seed_forcing_pino if bench == "diffusion_forcing"
        else run_one_seed_pino
    )
    results = {}
    for seed in seeds:
        seed_dir = base_run_dir / f"seed{seed}"
        results[str(seed)] = runner(config, int(seed), seed_dir)
    return {"run_dir": str(base_run_dir), "seeds": results}


__all__ = [
    "sample_collocation",
    "sample_forcing_params",
    "build_forcing_image",
    "left_wall_qL",
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
    "validate_forcing_gnrmse",
    "build_cvit",
    "run_one_seed_pino",
    "run_one_seed_forcing_pino",
    "run_config_seeds_pino",
]
