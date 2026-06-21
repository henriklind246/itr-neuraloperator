"""Inverse solver for the ``source_itr`` Gaussian-void interface resistance.

Use the *frozen forward FNO* as a
differentiable surrogate to recover the four void scalars

    theta = (R_base, R_amp, y0, sigma),   R_c(y) = R_base + R_amp e^{-((y-y0)/sigma)^2}

from observed temperatures, via a MAP / least-squares fit. The forward code is
untouched; this script only *imports* it.

What Stages 1-5 do (deliberately narrow):
  * direct-from-IC observation operator (known uniform-300 K IC, known source
    patch / amplitude, so G_theta is a pure function of theta),
  * observations at several times spanning early/mid/late, either full-field
    (Stage 1) or through a masked interface-proximal *sensor operator* M
    (Stage 2 — a band of x-columns near x = 0.5 crossed with a subset of y
    rows; full-field is the all-pixels special case),
  * LHS multistart Adam -> L-BFGS over an unconstrained reparameterization that
    keeps every iterate inside the physical box and honors the R_base-dependent
    R_amp ceiling (so the FNO is never queried out-of-distribution) (Stage 3),
  * a sensitivity / identifiability report (Stage 3): the SVD of the observation
    Jacobian ``dG_theta/dtheta`` at the estimate, which exposes ill-conditioned
    directions — chiefly the coupled R_amp/sigma void-"mass" ridge the plan
    predicts. Diagnostic only; no UQ intervals here.
  * an FV refinement / surrogate-error check (Stage 4): the FNO is the fast
    *exploration and sensitivity engine*, but the conservative Crank-Nicolson
    finite-volume solver is the *accuracy engine*. We re-evaluate the FNO MAP
    estimate ``theta_hat`` with the **real** ``FVSolver2D`` (rebuilt via
    ``ds.problem.configure_solver`` so the operator is bit-identical to data
    generation: k=3/35 layers, q_left=0, the patch source, ``interface_R=[Rc(y)]``),
    report the FV sensor residual and the **theta-local** FNO-vs-FV discrepancy
    ``C_FNO(theta_hat)`` (a single-point, signal-correlated surrogate error — used
    for conservative interval *inflation* later, not a calibrated covariance, and
    never a population diagonal), and optionally **polish** ``theta_hat`` against
    the FV sensor residual with derivative-free Nelder-Mead (the FV solve is not
    autodiff). Cheap because it starts from the FNO MAP.
  * measurement noise + uncertainty quantification (Stage 5): iid Gaussian
    sensor noise is added to the observations, and the per-scalar measurement
    variance is inflated to ``C_total = C_meas + C_FNO(theta_hat)`` (the Stage-4
    theta-local discrepancy). UQ is led by an interval on the well-conditioned
    integrated void *severity* (excess-resistance integral), via two primary
    routes: a **profile likelihood** (frequentist, Wilks interval over the
    R_amp/sigma ridge or the R_amp~0 boundary) and a **short RW-Metropolis
    MCMC** (Bayesian, uniform-theta prior pushed through the reparameterization
    Jacobian). The **Laplace** Gauss-Newton eigen-spectrum is reported only as a
    degeneracy *diagnostic* (it is least valid along the ridge / at the
    boundary), never as the reported interval.

theta enters the FNO in *two* places, and both must agree with the data
pipeline exactly:
  1. ``cond_static[2:6]`` — linear-normalized over ``RC_VOID_RANGES``
     (mirrors ``problems.source_itr.build_cond_vector_itr``).
  2. spatial channel ``RC_Y_CHANNEL`` (=4) — ``rc_log_norm(make_rc_void_profile(...))``
     broadcast across x (mirrors ``problems.source_itr._rc_channel`` in
     ``broadcast`` mode).

Everything Stage 1 needs that is *not* theta (IC snapshot, x/y channels, patch
mask, forcing sequence, the cond time/patch slots) is taken verbatim from the
real ``problem.build_item(ds, sid, s=0, j=n)`` so the scaffolding cannot drift
from training. Only channel 4 and cond[2:6] are replaced by torch functions of
theta, so autograd produces dT/dtheta through the frozen model.

Honesty caveats that the writeup must keep: the profile / MCMC intervals are
*conditional on* the inflated ``C_total``, which is a single-/few-point,
signal-correlated surrogate-error estimate, not a calibrated covariance; and the
recoverable quantity under sparse / far-field sensing is the integrated severity,
not the individual R_amp/sigma shape scalars (the Laplace diagnostic exists to
make that degeneracy explicit rather than hide it).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import (  # noqa: E402
    SnapshotPairDataset,
    T_EPS,
    load_sim_data,
    problem_from_config,
    split_sim_ids,
)
from problems.source_itr import RC_Y_CHANNEL  # noqa: E402
from src.operators.fno2d import FNO2d  # noqa: E402
from src.physics.internal_source import (  # noqa: E402
    RC_MIN,
    R_PEAK_MAX,
    RC_VOID_RANGES,
)

VOID_PARAM_NAMES = ("R_base", "R_amp", "y0", "sigma")


# ---------------------------------------------------------------------------
# Reparameterization: unconstrained u  <->  physical theta  (the heart of the
# operator wiring; checkpoint-independent and unit-tested separately).
# ---------------------------------------------------------------------------

def theta_from_unconstrained(u: torch.Tensor) -> torch.Tensor:
    """Map unconstrained ``u`` (..., 4) to physical theta (..., 4).

    Columns of both ``u`` and the result are ordered ``(R_base, R_amp, y0,
    sigma)``. The box and the R_base-dependent amplitude ceiling mirror data
    generation in ``problems.source_itr.sample_sim_params`` exactly:

        R_base  = 0.05 + 0.95 * sigmoid(u0)
        R_amp   = sigmoid(u1) * (R_PEAK_MAX - R_base)      # dependent ceiling
        y0      = 0.10 + 0.80 * sigmoid(u2)
        sigma   = 0.05 + 0.15 * sigmoid(u3)

    so any iterate stays inside ``RC_VOID_RANGES`` with ``R_c(y) in [0.05, 3.0]``
    and the FNO is never queried out-of-distribution.
    """
    base_lo, base_hi = RC_VOID_RANGES["R_base"]
    y0_lo, y0_hi = RC_VOID_RANGES["y0"]
    sig_lo, sig_hi = RC_VOID_RANGES["sigma"]

    s = torch.sigmoid(u)
    R_base = base_lo + (base_hi - base_lo) * s[..., 0]
    amp_unit = s[..., 1]
    R_amp = amp_unit * (R_PEAK_MAX - R_base)
    y0 = y0_lo + (y0_hi - y0_lo) * s[..., 2]
    sigma = sig_lo + (sig_hi - sig_lo) * s[..., 3]
    return torch.stack([R_base, R_amp, y0, sigma], dim=-1)


def unconstrained_from_theta(theta: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Inverse of :func:`theta_from_unconstrained` (logit of each box fraction).

    Used to seed a start *at* a known theta (tests, warm starts). The amplitude
    fraction is taken against the R_base-dependent ceiling so the round trip is
    exact.
    """
    base_lo, base_hi = RC_VOID_RANGES["R_base"]
    y0_lo, y0_hi = RC_VOID_RANGES["y0"]
    sig_lo, sig_hi = RC_VOID_RANGES["sigma"]

    R_base = theta[..., 0]
    R_amp = theta[..., 1]
    y0 = theta[..., 2]
    sigma = theta[..., 3]

    def _logit(p: torch.Tensor) -> torch.Tensor:
        p = p.clamp(eps, 1.0 - eps)
        return torch.log(p) - torch.log1p(-p)

    p_base = (R_base - base_lo) / (base_hi - base_lo)
    ceil = (R_PEAK_MAX - R_base).clamp_min(eps)
    p_amp = R_amp / ceil
    p_y0 = (y0 - y0_lo) / (y0_hi - y0_lo)
    p_sigma = (sigma - sig_lo) / (sig_hi - sig_lo)
    return torch.stack(
        [_logit(p_base), _logit(p_amp), _logit(p_y0), _logit(p_sigma)], dim=-1
    )


def theta_to_cond_slice(theta: torch.Tensor) -> torch.Tensor:
    """Linear-normalized ``cond_static[2:6]`` from theta (injection point 1).

    Mirrors ``build_cond_vector_itr``: each void scalar is normalized over its
    own ``RC_VOID_RANGES`` entry. ``R_amp`` is normalized over the *global*
    ``(0, R_PEAK_MAX - RC_MIN)`` span (not the per-sample dependent ceiling),
    exactly as the data pipeline does.
    """
    base_lo, base_hi = RC_VOID_RANGES["R_base"]
    amp_lo, amp_hi = RC_VOID_RANGES["R_amp"]
    y0_lo, y0_hi = RC_VOID_RANGES["y0"]
    sig_lo, sig_hi = RC_VOID_RANGES["sigma"]

    R_base = theta[..., 0]
    R_amp = theta[..., 1]
    y0 = theta[..., 2]
    sigma = theta[..., 3]

    R_base_norm = (R_base - base_lo) / (base_hi - base_lo)
    R_amp_norm = (R_amp - amp_lo) / (amp_hi - amp_lo)
    y0_norm = (y0 - y0_lo) / (y0_hi - y0_lo)
    sigma_norm = (sigma - sig_lo) / (sig_hi - sig_lo)
    return torch.stack([R_base_norm, R_amp_norm, y0_norm, sigma_norm], dim=-1)


def theta_to_rc_channel(
    theta: torch.Tensor, y_grid: torch.Tensor, Nx: int
) -> torch.Tensor:
    """Normalized R_c(y) spatial channel from theta (injection point 2).

    Mirrors ``_rc_channel`` in ``broadcast`` mode: build the physical profile
    ``R_c(y) = R_base + R_amp exp(-((y - y0)/sigma)^2)``, apply ``rc_log_norm``
    (log-spaced map to roughly [-1, 1] over ``[RC_MIN, R_PEAK_MAX]``), then
    broadcast across the ``Nx`` rows.

    Returns ``(..., Nx, Ny)`` (a leading batch dim of ``theta`` is preserved).
    """
    R_base = theta[..., 0:1]
    R_amp = theta[..., 1:2]
    y0 = theta[..., 2:3]
    sigma = theta[..., 3:4]

    y = y_grid.to(theta.dtype)  # (Ny,)
    Rc_y = R_base + R_amp * torch.exp(-(((y - y0) / sigma) ** 2))  # (..., Ny)

    log_min = float(np.log(RC_MIN))
    log_max = float(np.log(R_PEAK_MAX))
    Rc_y_norm = 2.0 * (torch.log(Rc_y) - log_min) / (log_max - log_min) - 1.0

    channel = Rc_y_norm.unsqueeze(-2).expand(*Rc_y_norm.shape[:-1], Nx, y.shape[0])
    return channel


# ---------------------------------------------------------------------------
# Sensor operator M (Stage 2): a sparse, interface-proximal pixel mask.
# ---------------------------------------------------------------------------

def build_interface_sensor_mask(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    *,
    interface_x: float = 0.5,
    x_halfwidth: float = 0.05,
    n_y: int = 8,
) -> torch.Tensor:
    """Boolean ``(Nx, Ny)`` mask of interface-proximal sensor pixels.

    Sensors sit where ``R_c(y)`` is identifiable: a band of x-columns within
    ``|x - interface_x| <= x_halfwidth`` crossed with ``n_y`` (roughly) evenly
    spaced y rows. If no column falls inside the band, the single nearest column
    is used. ``n_y >= Ny`` selects every row (a full y line of the band).

    Full-field observation is recovered by selecting all pixels, which callers
    express by passing ``mask=None`` (see :func:`apply_sensor_mask`); this helper
    always returns a *sparse* mask.
    """
    x = np.asarray(x_grid, dtype=np.float64)
    y = np.asarray(y_grid, dtype=np.float64)
    Nx, Ny = x.shape[0], y.shape[0]

    col_sel = np.abs(x - interface_x) <= x_halfwidth
    if not col_sel.any():
        col_sel = np.zeros(Nx, dtype=bool)
        col_sel[int(np.argmin(np.abs(x - interface_x)))] = True

    if n_y >= Ny:
        row_idx = np.arange(Ny)
    else:
        row_idx = np.unique(np.linspace(0, Ny - 1, n_y).round().astype(int))

    mask = np.zeros((Nx, Ny), dtype=bool)
    cols = np.where(col_sel)[0]
    mask[np.ix_(cols, row_idx)] = True
    return torch.from_numpy(mask)


def apply_sensor_mask(
    field: torch.Tensor, mask: Optional[torch.Tensor]
) -> torch.Tensor:
    """Gather a ``(N, Nx, Ny, C)`` field at sensor pixels.

    ``mask`` is a boolean ``(Nx, Ny)`` tensor; the result is
    ``(N, n_sensors, C)``. ``mask=None`` means full-field: the field is returned
    flattened over space as ``(N, Nx*Ny, C)`` so the loss reduction matches the
    masked path elementwise (same mean-of-squares convention, no spatial weight
    change).
    """
    N, Nx, Ny, C = field.shape
    if mask is None:
        return field.reshape(N, Nx * Ny, C)
    return field[:, mask]  # boolean index over dims (1, 2) -> (N, n_sensors, C)


# ---------------------------------------------------------------------------
# Checkpoint loading (mirrors src/operators/eval.py construction).
# ---------------------------------------------------------------------------

@dataclass
class LoadedModel:
    model: FNO2d
    config: dict
    dims: object
    mu_global: float
    sigma_global: float


def load_checkpoint(ckpt_path: str, device: str = "cpu") -> LoadedModel:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["conf"]
    bench = config.get("benchmark", {})
    name = bench.get("name")
    if name != "source_itr":
        raise ValueError(
            f"Checkpoint benchmark is {name!r}, expected 'source_itr'. "
            f"This inverse solver targets the source_itr void parameters."
        )
    dims = problem_from_config(config).dims
    model_cfg = config["model"]["parameters"]
    model = FNO2d(
        modes1=model_cfg["modes1"],
        modes2=model_cfg["modes2"],
        width=model_cfg["width"],
        in_channels=dims.in_channels,
        out_channels=model_cfg.get("out_channels", 1),
        n_layers=model_cfg.get("n_layers", 4),
        cond_static_dim=dims.cond_static_dim,
        cond_hidden=model_cfg.get("cond_hidden", 256),
        temporal_token_dim=dims.temporal_token_dim,
        temporal_hidden=model_cfg.get("temporal_hidden", 128),
        forcing_embed_dim=model_cfg.get("forcing_embed_dim", 64),
        forcing_spatial_dim=model_cfg.get("forcing_spatial_dim", 16),
        use_temporal_encoder=dims.use_temporal_encoder,
        use_forcing_time_aug=dims.use_forcing_time_aug,
        s_y_channel=dims.s_y_channel,
        padding_reference_resolution=model_cfg.get("padding_reference_resolution"),
        padding_mode=model_cfg.get("padding_mode", "zeros"),
        cin_exclude_padding=model_cfg.get("cin_exclude_padding", False),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    mu = float(ckpt.get("mu_global"))
    sigma = float(ckpt.get("sigma_global"))
    return LoadedModel(model=model, config=config, dims=dims, mu_global=mu, sigma_global=sigma)


# ---------------------------------------------------------------------------
# Dataset construction over a local source_itr artifact directory.
# ---------------------------------------------------------------------------

def build_dataset_from_dir(
    data_dir: str,
    config: dict,
    *,
    mu_global: float,
    sigma_global: float,
    seed: int = 0,
) -> SnapshotPairDataset:
    """Construct a SnapshotPairDataset over ``data_dir`` with the checkpoint's stats.

    Uses the checkpoint's baked ``(mu_global, sigma_global)`` so normalization
    matches training exactly (never re-normalized per-sample).
    """
    traj_path = os.path.join(data_dir, "trajectories.npy")
    x_path = os.path.join(data_dir, "x_grid.npy")
    y_path = os.path.join(data_dir, "y_grid.npy")
    t_path = os.path.join(data_dir, "t_grid.npy")
    sim_params_path = os.path.join(data_dir, "sim_params.npy")

    trajectories, x_grid, y_grid, t_grid = load_sim_data(traj_path, x_path, y_path, t_path)
    sim_params = np.load(sim_params_path, allow_pickle=True)
    num_sims = trajectories.shape[0]
    train_ids, val_ids, test_ids = split_sim_ids(
        num_sims,
        train_frac=config["data"].get("train_split", 0.7),
        val_frac=config["data"].get("val_split", 0.15),
        seed=seed,
    )

    dt_path = os.path.join(data_dir, "dt.npy")
    ramp_path = os.path.join(data_dir, "ramp_seconds.npy")
    dt = float(np.load(dt_path)) if os.path.exists(dt_path) else None
    ramp_seconds = float(np.load(ramp_path)) if os.path.exists(ramp_path) else 0.0

    problem = problem_from_config(config)
    # theta_to_rc_channel / predict_fullfield inject channel RC_Y_CHANNEL with the
    # broadcast formula only. A localized-trained checkpoint would see a channel-4
    # the model never trained on, biasing every recovered theta with no error.
    rc_mode = getattr(problem, "rc_channel_mode", "broadcast")
    if rc_mode != "broadcast":
        raise ValueError(
            f"inversion only supports rc_channel_mode='broadcast'; checkpoint "
            f"trained with {rc_mode!r}"
        )
    ds = SnapshotPairDataset(
        trajectories=trajectories,
        sim_params=sim_params,
        t_grid=t_grid,
        x_grid=x_grid,
        y_grid=y_grid,
        sim_ids=np.arange(num_sims),
        mu_global=mu_global,
        sigma_global=sigma_global,
        problem=problem,
        dt=dt,
        ramp_seconds=ramp_seconds,
    )
    ds._split_ids = {"train": train_ids, "val": val_ids, "test": test_ids}
    return ds


# ---------------------------------------------------------------------------
# Direct-from-IC observation operator.
# ---------------------------------------------------------------------------

@dataclass
class ObservationSet:
    """theta-independent scaffolding + full-field targets for one sim.

    Built once from ``problem.build_item(ds, sid, s=0, j=n)`` for each
    observation time index ``n``. ``spatial`` carries the four theta-independent
    channels verbatim; channel ``RC_Y_CHANNEL`` is overwritten per-theta in the
    loss. ``cond`` carries the time/patch slots; columns 2:6 are overwritten.
    """

    sid: int
    time_indices: list[int]
    spatial: torch.Tensor       # (N, Nx, Ny, C)
    cond: torch.Tensor          # (N, cond_dim)
    forcing_seq: torch.Tensor   # (N, M, token_dim)
    targets: torch.Tensor       # (N, Nx, Ny, 1)  normalized full-field T
    y_grid: torch.Tensor        # (Ny,)
    Nx: int
    theta_true: Optional[torch.Tensor] = None  # (4,) physical, if known
    mask: Optional[torch.Tensor] = None  # bool (Nx, Ny) sensor pixels; None=full-field


def build_observation_set(
    ds: SnapshotPairDataset,
    sid: int,
    time_indices: list[int],
    device: str = "cpu",
    *,
    interface_x: float = 0.5,
    sensor_x_halfwidth: Optional[float] = None,
    sensor_n_y: int = 8,
) -> ObservationSet:
    """Assemble direct-from-IC observations for ``sid`` at the given time indices.

    For each ``n`` we call the real ``build_item(ds, sid, s=0, j=n)``: with
    ``s=0`` the input snapshot *is* the uniform-300 K IC and ``t_s=0``, so the
    operator is a pure function of theta. The returned spatial/cond/forcing are
    the exact training-pipeline tensors; ``targets`` are the FV trajectory
    snapshots (normalized).

    When ``sensor_x_halfwidth`` is ``None`` the observation is full-field
    (Stage 1). Otherwise a sparse interface-proximal sensor mask M (Stage 2) is
    attached: a band of x-columns within ``sensor_x_halfwidth`` of
    ``interface_x`` crossed with ``sensor_n_y`` y rows. The full ``targets``
    field is still stored; the loss restricts to the masked pixels.
    """
    spatials, conds, fseqs, targets = [], [], [], []
    for n in time_indices:
        item = ds.problem.build_item(ds, int(sid), 0, int(n))
        spatials.append(torch.from_numpy(item["spatial"]))
        conds.append(torch.from_numpy(item["cond_static"]))
        fseqs.append(torch.from_numpy(item["forcing_seq"]))
        targets.append(torch.from_numpy(item["Y"]))

    spatial = torch.stack(spatials).to(device)
    cond = torch.stack(conds).to(device)
    forcing_seq = torch.stack(fseqs).to(device)
    target = torch.stack(targets).to(device)
    y_grid = torch.from_numpy(np.asarray(ds.y_grid, dtype=np.float32)).to(device)

    mask = None
    if sensor_x_halfwidth is not None:
        mask = build_interface_sensor_mask(
            np.asarray(ds.x_grid),
            np.asarray(ds.y_grid),
            interface_x=interface_x,
            x_halfwidth=float(sensor_x_halfwidth),
            n_y=int(sensor_n_y),
        ).to(device)

    params = ds.sim_params[int(sid)]
    theta_true = torch.tensor(
        [
            float(params["R_c_base"]),
            float(params["R_c_amp"]),
            float(params["R_c_y0"]),
            float(params["R_c_sigma"]),
        ],
        dtype=torch.float32,
        device=device,
    )
    return ObservationSet(
        sid=int(sid),
        time_indices=list(time_indices),
        spatial=spatial,
        cond=cond,
        forcing_seq=forcing_seq,
        targets=target,
        y_grid=y_grid,
        Nx=spatial.shape[1],
        theta_true=theta_true,
        mask=mask,
    )


def predict_fullfield(
    model: FNO2d, obs: ObservationSet, theta: torch.Tensor
) -> torch.Tensor:
    """G_theta over all observation times: returns (N, Nx, Ny, 1) normalized T.

    ``theta`` is a single physical (4,) vector. Both injection points are
    rebuilt from theta and spliced into the (otherwise detached) scaffolding so
    autograd flows only through cond[2:6] and channel RC_Y_CHANNEL.

    The splice is done out-of-place (``torch.where`` / ``torch.cat`` rather than
    indexed assignment) so the function is safe under ``vmap`` — needed for the
    Stage 3 ``torch.autograd.functional.jacobian`` sensitivity diagnostic.
    """
    N = obs.spatial.shape[0]
    Nx, Ny, C = (obs.spatial.shape[1], obs.spatial.shape[2], obs.spatial.shape[3])

    rc_channel = theta_to_rc_channel(theta, obs.y_grid, Nx)  # (Nx, Ny)
    rc_b = rc_channel.unsqueeze(0).unsqueeze(-1).expand(N, Nx, Ny, C)
    chan_sel = torch.arange(C, device=obs.spatial.device) == RC_Y_CHANNEL
    spatial = torch.where(chan_sel, rc_b, obs.spatial)

    cond_slice = theta_to_cond_slice(theta)  # (4,)
    cond = torch.cat(
        [obs.cond[:, :2], cond_slice.unsqueeze(0).expand(N, 4), obs.cond[:, 6:]],
        dim=1,
    )

    return model(spatial, cond, obs.forcing_seq)


# ---------------------------------------------------------------------------
# Sensitivity / identifiability diagnostics (Stage 3): J = dG_theta/dtheta at a
# point, and the SVD of J that exposes ill-conditioned directions (notably the
# coupled R_amp/sigma void-"mass" ridge). This is *diagnostic only* — no UQ,
# no Laplace/Hessian intervals (those are Stage 5).
# ---------------------------------------------------------------------------

def observation_jacobian(
    model: FNO2d, obs: ObservationSet, theta: torch.Tensor
) -> torch.Tensor:
    """Jacobian ``J = d(masked G_theta)/d theta`` at ``theta``: ``(m, 4)``.

    ``m`` is the number of observed scalars (sensor pixels times observation
    times times channels). The model is frozen; only the four physical void
    scalars are differentiated, through both injection points.
    """
    theta0 = theta.detach().to(obs.spatial.device)

    def obs_vec(th: torch.Tensor) -> torch.Tensor:
        pred = predict_fullfield(model, obs, th)
        return apply_sensor_mask(pred, obs.mask).reshape(-1)

    try:
        J = torch.autograd.functional.jacobian(obs_vec, theta0, vectorize=True)
    except RuntimeError:
        # vmap may not support every op (e.g. some FFT paths); fall back to the
        # slower row-by-row reverse-mode Jacobian, which is always correct.
        J = torch.autograd.functional.jacobian(obs_vec, theta0, vectorize=False)
    return J.detach()


def svd_identifiability(J: np.ndarray) -> dict:
    """Identifiability summary from the observation Jacobian ``J`` (m, 4).

    Pure linear algebra (unit-tested independently of the FNO):
      * ``singular_values`` — descending; small values flag directions the data
        cannot constrain.
      * ``right_vectors`` — (4, 4), columns are orthonormal directions in
        theta-space ``(R_base, R_amp, y0, sigma)``.
      * ``param_sensitivity`` — per-parameter column norms ``||dG/dtheta_k||``.
      * ``least_identified_dir`` — right singular vector of the *smallest*
        singular value (the worst-constrained combination).
      * ``ramp_sigma_alignment`` — fraction of that direction's norm lying in the
        ``(R_amp, sigma)`` plane; ~1 means the ill-conditioned direction *is* the
        void-mass ridge the plan predicts.
      * ``cond_number`` — s_max / s_min.
    """
    Jn = np.asarray(J, dtype=np.float64)
    param_sensitivity = np.linalg.norm(Jn, axis=0)
    _, S, Vt = np.linalg.svd(Jn, full_matrices=False)
    V = Vt.T
    least = V[:, -1]
    least = least / (np.linalg.norm(least) + 1e-300)
    ramp_sigma_alignment = float(
        np.linalg.norm(least[[1, 3]]) / (np.linalg.norm(least) + 1e-300)
    )
    cond_number = float(S[0] / S[-1]) if S[-1] > 0 else float("inf")
    return {
        "singular_values": S,
        "right_vectors": V,
        "param_sensitivity": param_sensitivity,
        "least_identified_dir": least,
        "ramp_sigma_alignment": ramp_sigma_alignment,
        "cond_number": cond_number,
    }


def sensitivity_report(
    model: FNO2d, obs: ObservationSet, theta: torch.Tensor
) -> dict:
    """``svd_identifiability`` of the observation Jacobian at ``theta``."""
    J = observation_jacobian(model, obs, theta)
    return svd_identifiability(J.cpu().numpy())


def sensitivity_summary(report: dict) -> dict:
    """Flatten a :func:`sensitivity_report` into scalar CSV columns."""
    S = report["singular_values"]
    least = report["least_identified_dir"]
    sens = report["param_sensitivity"]
    out = {
        "cond_number": report["cond_number"],
        "ramp_sigma_alignment": report["ramp_sigma_alignment"],
    }
    for i, name in enumerate(VOID_PARAM_NAMES):
        out[f"sv_{i}"] = float(S[i]) if i < len(S) else float("nan")
        out[f"sens_{name}"] = float(sens[i])
        out[f"least_dir_{name}"] = float(least[i])
    return out


# ---------------------------------------------------------------------------
# Multistart sampling (Stage 3): Latin-hypercube starts over the theta-box.
# ---------------------------------------------------------------------------

def lhs_unit_samples(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Jittered Latin-hypercube sample on the open unit cube ``(0, 1)^d``.

    Each of the ``d`` columns is stratified into ``n`` equal bins with one
    sample jittered inside each bin, then the bins are permuted independently
    per column. Returns ``(n, d)``.
    """
    cut = np.linspace(0.0, 1.0, n + 1)
    lo = cut[:n]
    hi = cut[1:]
    pts = lo[:, None] + rng.uniform(size=(n, d)) * (hi - lo)[:, None]
    for j in range(d):
        rng.shuffle(pts[:, j])
    return pts


def lhs_starts_u(n_starts: int, rng: np.random.Generator, eps: float = 1e-3) -> np.ndarray:
    """LHS starts in unconstrained ``u``-space, stratified over the theta-box.

    LHS is taken on the sigmoid-fraction cube (which *is* the physical box via
    :func:`theta_from_unconstrained`), then mapped to ``u`` with the logit so the
    starts evenly tile every void scalar's range. Returns ``(n_starts, 4)``.
    """
    f = lhs_unit_samples(n_starts, 4, rng)
    f = np.clip(f, eps, 1.0 - eps)
    u = np.log(f / (1.0 - f))  # logit: box fraction -> unconstrained
    return u.astype(np.float32)


# ---------------------------------------------------------------------------
# MAP optimization: multistart Adam -> L-BFGS.
# ---------------------------------------------------------------------------

@dataclass
class InversionConfig:
    n_starts: int = 8
    adam_steps: int = 400
    adam_lr: float = 0.05
    lbfgs_steps: int = 50
    seed: int = 0
    reg_weight: float = 0.0  # weak Gaussian-in-u optimization regularizer
    device: str = "cpu"
    start_sampling: str = "lhs"  # "lhs" (Stage 3) or "normal" (legacy N(0,1))


@dataclass
class InversionResult:
    theta_hat: torch.Tensor
    loss: float
    theta_true: Optional[torch.Tensor]
    per_start_losses: list[float] = field(default_factory=list)
    per_start_theta: list[torch.Tensor] = field(default_factory=list)


def _data_loss(model: FNO2d, obs: ObservationSet, u: torch.Tensor, reg_weight: float):
    theta = theta_from_unconstrained(u)
    pred = predict_fullfield(model, obs, theta)
    pred_obs = apply_sensor_mask(pred, obs.mask)
    target_obs = apply_sensor_mask(obs.targets, obs.mask)
    resid = pred_obs - target_obs
    loss = 0.5 * torch.mean(resid ** 2)
    if reg_weight > 0:
        loss = loss + 0.5 * reg_weight * torch.mean(u ** 2)
    return loss


def invert_sim(
    model: FNO2d, obs: ObservationSet, cfg: InversionConfig
) -> InversionResult:
    device = cfg.device
    rng = np.random.default_rng(cfg.seed)
    per_start_losses: list[float] = []
    per_start_theta: list[torch.Tensor] = []
    best_loss = float("inf")
    best_theta = None

    if cfg.start_sampling == "lhs":
        starts_u = lhs_starts_u(cfg.n_starts, rng)
    elif cfg.start_sampling == "normal":
        starts_u = rng.normal(0.0, 1.0, size=(cfg.n_starts, 4)).astype(np.float32)
    else:
        raise ValueError(f"Unknown start_sampling {cfg.start_sampling!r}")

    for start in range(cfg.n_starts):
        u0 = torch.from_numpy(starts_u[start]).to(device)
        u = u0.clone().requires_grad_(True)

        adam = torch.optim.Adam([u], lr=cfg.adam_lr)
        for _ in range(cfg.adam_steps):
            adam.zero_grad()
            loss = _data_loss(model, obs, u, cfg.reg_weight)
            loss.backward()
            adam.step()

        lbfgs = torch.optim.LBFGS(
            [u], max_iter=cfg.lbfgs_steps, line_search_fn="strong_wolfe"
        )

        def closure():
            lbfgs.zero_grad()
            loss = _data_loss(model, obs, u, cfg.reg_weight)
            loss.backward()
            return loss

        lbfgs.step(closure)

        with torch.no_grad():
            final_loss = float(_data_loss(model, obs, u, cfg.reg_weight))
            theta = theta_from_unconstrained(u).detach().cpu()
        per_start_losses.append(final_loss)
        per_start_theta.append(theta)
        if final_loss < best_loss:
            best_loss = final_loss
            best_theta = theta

    return InversionResult(
        theta_hat=best_theta,
        loss=best_loss,
        theta_true=None if obs.theta_true is None else obs.theta_true.detach().cpu(),
        per_start_losses=per_start_losses,
        per_start_theta=per_start_theta,
    )


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------

def integrated_excess_resistance(theta: torch.Tensor, y_grid: torch.Tensor) -> float:
    """Trapezoidal integral of (R_c(y) - R_base) over y: total void "mass"."""
    R_base, R_amp, y0, sigma = (theta[i].item() for i in range(4))
    y = y_grid.detach().cpu().numpy().astype(np.float64)
    excess = R_amp * np.exp(-(((y - y0) / sigma) ** 2))
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(trapz(excess, y))


def summarize(result: InversionResult, y_grid: torch.Tensor) -> dict:
    out = {
        "loss": result.loss,
        "R_base_hat": float(result.theta_hat[0]),
        "R_amp_hat": float(result.theta_hat[1]),
        "y0_hat": float(result.theta_hat[2]),
        "sigma_hat": float(result.theta_hat[3]),
        "excess_int_hat": integrated_excess_resistance(result.theta_hat, y_grid),
    }
    if result.theta_true is not None:
        t = result.theta_true
        out.update(
            {
                "R_base_true": float(t[0]),
                "R_amp_true": float(t[1]),
                "y0_true": float(t[2]),
                "sigma_true": float(t[3]),
                "excess_int_true": integrated_excess_resistance(t, y_grid),
                "R_base_abserr": abs(float(result.theta_hat[0]) - float(t[0])),
                "R_amp_abserr": abs(float(result.theta_hat[1]) - float(t[1])),
                "y0_abserr": abs(float(result.theta_hat[2]) - float(t[2])),
                "sigma_abserr": abs(float(result.theta_hat[3]) - float(t[3])),
            }
        )
        out["excess_int_abserr"] = abs(out["excess_int_hat"] - out["excess_int_true"])
    return out


def _default_time_indices(Nt: int) -> list[int]:
    """Early / mid / late observation snapshots (skip t=0, which is the IC)."""
    early = max(1, Nt // 4)
    mid = Nt // 2
    late = Nt - 1
    return sorted(set([early, mid, late]))


# ---------------------------------------------------------------------------
# Stage 4: FV refinement + theta-local surrogate error.
#
# The FNO is the *exploration / sensitivity* engine; the conservative FV solver
# is the *accuracy* engine. We rebuild the EXACT data-generation operator via
# ``ds.problem.configure_solver`` (k=3/35 layers, q_left=0, the patch source,
# ``interface_R=[Rc(y)]``) and re-evaluate the FNO MAP estimate with it. We
# report the FV sensor residual and the theta-local FNO-vs-FV discrepancy
# ``C_FNO(theta_hat)``, and optionally polish theta_hat against the FV residual
# with derivative-free Nelder-Mead (FV is not autodiff). No UQ here (Stage 5).
# ---------------------------------------------------------------------------

# Domain / forcing constants hardcoded in ``data/generate_dataset.py`` (they are
# NOT persisted with the dataset, so the FV operator can only be reproduced by
# reusing the exact generation values). ``configure_solver`` reads the rest
# (t_off, the patch, the void scalars) from the per-sim ``sim_params`` and the
# resolution / dt / t_final from the dataset.
_GEN_DOMAIN = dict(a=0.0, b=1.0, c=0.0, d=1.0)
_GEN_LAM_TARGET = 0.8
_GEN_FLUX_F = 0.0
_GEN_FLUX_A = 0.0
_GEN_T_ON = 0.0
_GEN_PHASE = 0.0
_GEN_TUKEY_ALPHA = 0.5


def build_fv_base_kwargs(ds: SnapshotPairDataset) -> dict:
    """Reconstruct the ``base_kwargs`` ``configure_solver`` consumes.

    Mirrors ``data/generate_dataset.py``: the domain box, ``lam_target``, the
    (zero) left flux, ``t_on``/``phase``/``tukey_alpha`` are the generation
    constants above; ``Nx``/``Ny`` and ``y_grid`` come from the dataset; ``dt``
    is the *solver* dt persisted in ``dt.npy`` (decoupled from the decimated
    ``t_grid`` spacing) and ``t_final`` is the dataset horizon. ``configure_solver``
    reads ``t_off`` and the source patch from ``params``, so they are absent here.
    """
    if ds.dt is None:
        raise ValueError(
            "FV refinement needs the solver dt; dt.npy is missing from the data dir."
        )
    # ds.t_final is t_grid[-1] read back from a float32 .npy, so it carries a
    # ~1e-7 cast error (e.g. 0.3 -> 0.30000001) that fails build_time_grid's
    # explicit-dt divisibility check (atol/rtol 1e-12). The solver dt is the
    # authoritative spacing, so snap t_final to the nearest whole number of dt
    # steps to recover the exact generation horizon.
    dt = float(ds.dt)
    n_steps = int(round(float(ds.t_final) / dt))
    t_final = n_steps * dt
    return dict(
        a=_GEN_DOMAIN["a"], b=_GEN_DOMAIN["b"],
        c=_GEN_DOMAIN["c"], d=_GEN_DOMAIN["d"],
        Nx=int(ds.Nx), Ny=int(ds.Ny),
        lam_target=_GEN_LAM_TARGET,
        t_final=t_final,
        flux_f=_GEN_FLUX_F, flux_A=_GEN_FLUX_A,
        t_on=_GEN_T_ON, phase=_GEN_PHASE,
        dt=dt, tukey_alpha=_GEN_TUKEY_ALPHA,
        y_grid=np.asarray(ds.y_grid, dtype=np.float64),
    )


def _theta_to_numpy(theta) -> np.ndarray:
    if torch.is_tensor(theta):
        return theta.detach().cpu().numpy().astype(np.float64)
    return np.asarray(theta, dtype=np.float64)


def fv_predict_masked(
    ds: SnapshotPairDataset,
    base_kwargs: dict,
    sid: int,
    theta,
    time_indices: list[int],
    *,
    mu_global: float,
    sigma_global: float,
    mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """FV forward of ``theta`` at the observation sensors/times (normalized).

    Overrides only the four void scalars (and the mirrored ``R_c`` slot) in a
    copy of the stored ``sim_params[sid]`` so the patch / IC / interface_x are
    bit-identical to data generation, rebuilds the solver, integrates from the
    stored IC, selects the observation snapshots by nearest solver time, and
    normalizes with the checkpoint's baked global stats. Returns a CPU tensor
    ``(N, n_sensors, 1)`` (or ``(N, Nx*Ny, 1)`` full-field when ``mask`` is None).
    """
    th = _theta_to_numpy(theta)
    params = dict(ds.sim_params[int(sid)])
    params["R_c_base"] = float(th[0])
    params["R_c_amp"] = float(th[1])
    params["R_c_y0"] = float(th[2])
    params["R_c_sigma"] = float(th[3])
    params["R_c"] = float(th[0])

    solver = ds.problem.configure_solver(params, base_kwargs)
    T0 = np.asarray(params["T0"], dtype=np.float64)
    t_solver, _x, _y, T_hist = solver.solve(T0=T0, store_trajectory=True)

    t_obs = np.asarray(ds.t_grid, dtype=np.float64)[list(time_indices)]
    t_solver = np.asarray(t_solver, dtype=np.float64)
    sel = [int(np.argmin(np.abs(t_solver - tt))) for tt in t_obs]

    T_sel = np.asarray(T_hist)[sel]  # (N, Nx, Ny) physical Kelvin
    T_norm = (T_sel - mu_global) / (sigma_global + T_EPS)
    field = torch.from_numpy(T_norm.astype(np.float32)).unsqueeze(-1)  # (N,Nx,Ny,1)
    return apply_sensor_mask(field, mask)


def _masked_half_mse(pred_obs: torch.Tensor, target_obs: torch.Tensor) -> float:
    """0.5 * mean squared residual over the observed pixels.

    Same convention as the MAP data loss (:func:`_data_loss` with reg_weight=0),
    so FV and FNO residuals are directly comparable to the inversion ``loss``.
    """
    resid = pred_obs - target_obs
    return 0.5 * float(torch.mean(resid ** 2))


@dataclass
class FVRefineResult:
    fno_resid: float            # 0.5*mean (FNO(theta_hat) - obs)^2  over sensors
    fv_resid: float             # 0.5*mean (FV(theta_hat)  - obs)^2  over sensors
    fno_vs_fv_resid: float      # C_FNO(theta_hat): 0.5*mean (FNO - FV)^2
    theta_fv_polish: Optional[torch.Tensor] = None
    fv_polish_resid: Optional[float] = None
    fv_polish_evals: Optional[int] = None


def fv_polish(
    ds: SnapshotPairDataset,
    base_kwargs: dict,
    obs: ObservationSet,
    theta_hat: torch.Tensor,
    *,
    mu_global: float,
    sigma_global: float,
    maxiter: int = 60,
) -> tuple[torch.Tensor, float, int]:
    """Derivative-free FV polish of ``theta_hat`` against the FV sensor residual.

    Optimizes in the unconstrained ``u``-space (so every FV query stays inside
    the physical box and honors the R_base-dependent amplitude ceiling) with
    Nelder-Mead, starting from the FNO MAP. Returns ``(theta_polished,
    final_half_mse, n_func_evals)``.
    """
    from scipy.optimize import minimize

    mask_cpu = None if obs.mask is None else obs.mask.cpu()
    target_obs = apply_sensor_mask(obs.targets.detach().cpu(), mask_cpu)
    u0 = unconstrained_from_theta(theta_hat.detach().cpu()).numpy().astype(np.float64)

    def objective(u_np: np.ndarray) -> float:
        u_t = torch.from_numpy(u_np.astype(np.float32))
        theta = theta_from_unconstrained(u_t)
        pred_obs = fv_predict_masked(
            ds, base_kwargs, obs.sid, theta, obs.time_indices,
            mu_global=mu_global, sigma_global=sigma_global, mask=mask_cpu,
        )
        return _masked_half_mse(pred_obs, target_obs)

    res = minimize(
        objective, u0, method="Nelder-Mead",
        options={"maxiter": int(maxiter), "xatol": 1e-4, "fatol": 1e-14},
    )
    theta_polished = theta_from_unconstrained(
        torch.from_numpy(np.asarray(res.x, dtype=np.float32))
    )
    return theta_polished, float(res.fun), int(res.nfev)


def fv_refine_report(
    model: FNO2d,
    ds: SnapshotPairDataset,
    base_kwargs: dict,
    obs: ObservationSet,
    theta_hat: torch.Tensor,
    *,
    mu_global: float,
    sigma_global: float,
    polish: bool = False,
    polish_maxiter: int = 60,
) -> FVRefineResult:
    """FNO-vs-FV residual report at ``theta_hat`` (+ optional FV polish)."""
    mask_cpu = None if obs.mask is None else obs.mask.cpu()
    target_obs = apply_sensor_mask(obs.targets.detach().cpu(), mask_cpu)

    with torch.no_grad():
        fno_pred = predict_fullfield(model, obs, theta_hat.to(obs.spatial.device))
    fno_obs = apply_sensor_mask(fno_pred.detach().cpu(), mask_cpu)
    fv_obs = fv_predict_masked(
        ds, base_kwargs, obs.sid, theta_hat, obs.time_indices,
        mu_global=mu_global, sigma_global=sigma_global, mask=mask_cpu,
    )

    out = FVRefineResult(
        fno_resid=_masked_half_mse(fno_obs, target_obs),
        fv_resid=_masked_half_mse(fv_obs, target_obs),
        fno_vs_fv_resid=_masked_half_mse(fno_obs, fv_obs),
    )
    if polish:
        theta_p, fv_polish_resid, nfev = fv_polish(
            ds, base_kwargs, obs, theta_hat,
            mu_global=mu_global, sigma_global=sigma_global, maxiter=polish_maxiter,
        )
        out.theta_fv_polish = theta_p
        out.fv_polish_resid = fv_polish_resid
        out.fv_polish_evals = nfev
    return out


def fv_refine_summary(
    res: FVRefineResult, obs: ObservationSet, y_grid: torch.Tensor
) -> dict:
    """Flatten a :func:`fv_refine_report` into scalar CSV columns."""
    out = {
        "fno_resid": res.fno_resid,
        "fv_resid": res.fv_resid,
        "fno_vs_fv_resid": res.fno_vs_fv_resid,  # C_FNO(theta_hat)
    }
    if res.theta_fv_polish is not None:
        tp = res.theta_fv_polish
        out["fv_polish_resid"] = res.fv_polish_resid
        out["fv_polish_evals"] = res.fv_polish_evals
        for i, name in enumerate(VOID_PARAM_NAMES):
            out[f"{name}_fvpolish"] = float(tp[i])
        out["excess_int_fvpolish"] = integrated_excess_resistance(tp, y_grid)
        if obs.theta_true is not None:
            t = obs.theta_true
            for i, name in enumerate(VOID_PARAM_NAMES):
                out[f"{name}_fvpolish_abserr"] = abs(float(tp[i]) - float(t[i]))
            out["excess_int_fvpolish_abserr"] = abs(
                out["excess_int_fvpolish"]
                - integrated_excess_resistance(t, y_grid)
            )
    return out


# ---------------------------------------------------------------------------
# Stage 5: measurement noise + uncertainty quantification.
#
# The honesty constraints from the plan are load-bearing here:
#   * Profile-likelihood (frequentist) and a short RW-Metropolis MCMC (Bayesian,
#     uniform-theta prior) are the PRIMARY UQ. Both lead with an interval on the
#     well-conditioned integrated void "severity" (excess-resistance integral),
#     not on R_amp/sigma separately.
#   * Laplace (Gauss-Newton from the observation Jacobian) is DEMOTED to a fast
#     degeneracy *diagnostic*: its eigen-spectrum identifies the coupled
#     R_amp/sigma ridge and the R_amp~0 boundary, where a Gaussian-around-MAP is
#     least trustworthy. We never report final intervals from it.
#   * The surrogate error is theta-local: the per-scalar measurement variance is
#     C_total = C_meas + C_FNO(theta_hat), with C_meas = noise_std^2 and the FNO
#     error variance = 2 * fno_vs_fv_resid (fno_vs_fv_resid is a half-MSE). This
#     inflates intervals conservatively; it is not claimed to be a calibrated
#     covariance.
# ---------------------------------------------------------------------------

# Wilks thresholds on (NLL - NLL_min): 0.5 * chi2_inv(level, df=1). A profile
# confidence interval at `level` is where 2*(NLL - NLL_min) <= chi2_inv, i.e.
# (NLL - NLL_min) <= the value below.
_CHI2_HALF_THRESH = {0.68: 0.5, 0.90: 1.352772, 0.95: 1.920729, 0.99: 3.317448}


def effective_sigma2(noise_std: float, c_fno: float = 0.0) -> float:
    """Per-scalar effective measurement variance ``C_total = C_meas + C_FNO``.

    ``C_meas = noise_std**2`` is the iid Gaussian sensor noise variance (in
    normalized temperature units). ``c_fno`` is the Stage-4 half-MSE
    ``fno_vs_fv_resid`` (``0.5*mean (FNO-FV)^2``), so the FNO error *variance* is
    ``2*c_fno``. Both are in the same normalized units as the residual, so the
    Gaussian NLL below is consistent.
    """
    return float(noise_std) ** 2 + 2.0 * float(c_fno)


def _masked_residual(
    model: FNO2d, obs: ObservationSet, theta: torch.Tensor
) -> torch.Tensor:
    """Flat residual vector ``(m,)`` ``= masked(G_theta) - masked(obs)``."""
    pred = predict_fullfield(model, obs, theta)
    pred_obs = apply_sensor_mask(pred, obs.mask)
    target_obs = apply_sensor_mask(obs.targets, obs.mask)
    return (pred_obs - target_obs).reshape(-1)


def neg_log_likelihood(
    model: FNO2d, obs: ObservationSet, theta: torch.Tensor, sigma_eff2: float
) -> torch.Tensor:
    """Gaussian negative log-likelihood (sum convention) at ``theta``.

    ``NLL = 0.5 * sum_resid^2 / sigma_eff2`` up to a theta-independent constant.
    This is the proper likelihood used by both the profile-likelihood and the
    MCMC target; it differs from the MAP ``_data_loss`` (a *mean*-of-squares) by
    the count ``m`` and the variance ``sigma_eff2``, so the two are not the same
    scale — UQ needs the sum/variance form to get the chi-square calibration
    right.
    """
    resid = _masked_residual(model, obs, theta)
    return 0.5 * torch.sum(resid ** 2) / float(sigma_eff2)


def add_measurement_noise(
    obs: ObservationSet, noise_std: float, seed: int = 0
) -> None:
    """Add iid ``N(0, noise_std^2)`` noise to ``obs.targets`` in place.

    Noise is added to the full target field before masking; since the mask is a
    deterministic pixel selection, this is exactly iid noise on the sensors. A
    non-positive ``noise_std`` is a no-op. The generator is seeded for
    reproducibility across a sweep (callers pass ``noise_seed + sid``).
    """
    if noise_std is None or noise_std <= 0:
        return
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    noise = torch.randn(obs.targets.shape, generator=gen, dtype=torch.float32)
    obs.targets = obs.targets + (noise * float(noise_std)).to(obs.targets)


def theta_logabsdet_du(u: torch.Tensor) -> torch.Tensor:
    """``log|det d theta / d u|`` of :func:`theta_from_unconstrained`.

    The map is lower-triangular in the ordering ``(R_base, R_amp, y0, sigma)`` vs
    ``(u0, u1, u2, u3)`` — only ``R_amp`` is coupled (to ``u0`` through the
    R_base-dependent ceiling), and that coupling sits *below* the diagonal — so
    the determinant is the product of the four diagonal partials:

        dR_base/du0 = (base_hi-base_lo) * s0(1-s0)
        dR_amp/du1  = (R_PEAK_MAX - R_base) * s1(1-s1)
        dy0/du2     = (y0_hi-y0_lo) * s2(1-s2)
        dsigma/du3  = (sig_hi-sig_lo) * s3(1-s3)

    Used to push a *uniform-theta* prior through to ``u`` for the MCMC target
    (``log p(u) = -NLL + logabsdet``). Implemented with softplus for stable
    ``log s = -softplus(-u)`` / ``log(1-s) = -softplus(u)``.
    """
    base_lo, base_hi = RC_VOID_RANGES["R_base"]
    y0_lo, y0_hi = RC_VOID_RANGES["y0"]
    sig_lo, sig_hi = RC_VOID_RANGES["sigma"]

    sp = torch.nn.functional.softplus
    u0, u1, u2, u3 = u[..., 0], u[..., 1], u[..., 2], u[..., 3]
    R_base = base_lo + (base_hi - base_lo) * torch.sigmoid(u0)

    log_diag0 = float(np.log(base_hi - base_lo)) - sp(-u0) - sp(u0)
    log_diag1 = torch.log(R_PEAK_MAX - R_base) - sp(-u1) - sp(u1)
    log_diag2 = float(np.log(y0_hi - y0_lo)) - sp(-u2) - sp(u2)
    log_diag3 = float(np.log(sig_hi - sig_lo)) - sp(-u3) - sp(u3)
    return log_diag0 + log_diag1 + log_diag2 + log_diag3


# --- Profile likelihood (primary, frequentist) -----------------------------

def _theta_profile(
    u: torch.Tensor, fixed_index: int, fixed_value: float
) -> torch.Tensor:
    """``theta_from_unconstrained(u)`` with physical coord ``fixed_index`` pinned.

    The dependent ``R_amp`` ceiling is honored in both directions: pinning
    ``R_base`` (index 0) recomputes the free ``R_amp`` against the pinned base,
    and pinning ``R_amp`` (index 1) caps the free ``R_base`` at
    ``min(base_hi, R_PEAK_MAX - R_amp)`` so ``R_base + R_amp <= R_PEAK_MAX``
    always holds and the FNO is never queried outside ``R_c(y) in [RC_MIN,
    R_PEAK_MAX]``. The pinned coordinate's ``u`` entry is ignored, so the
    optimizer effectively re-fits the other three.
    """
    base_lo, base_hi = RC_VOID_RANGES["R_base"]
    y0_lo, y0_hi = RC_VOID_RANGES["y0"]
    sig_lo, sig_hi = RC_VOID_RANGES["sigma"]
    s = torch.sigmoid(u)

    if fixed_index == 1:
        R_amp = torch.as_tensor(fixed_value, dtype=u.dtype, device=u.device)
        base_ceiling = min(base_hi, R_PEAK_MAX - float(fixed_value))
        R_base = base_lo + (base_ceiling - base_lo) * s[..., 0]
    else:
        R_base = (
            torch.as_tensor(fixed_value, dtype=u.dtype, device=u.device)
            if fixed_index == 0
            else base_lo + (base_hi - base_lo) * s[..., 0]
        )
        R_amp = s[..., 1] * (R_PEAK_MAX - R_base)
    y0 = (
        torch.as_tensor(fixed_value, dtype=u.dtype, device=u.device)
        if fixed_index == 2
        else y0_lo + (y0_hi - y0_lo) * s[..., 2]
    )
    sigma = (
        torch.as_tensor(fixed_value, dtype=u.dtype, device=u.device)
        if fixed_index == 3
        else sig_lo + (sig_hi - sig_lo) * s[..., 3]
    )
    return torch.stack([R_base, R_amp, y0, sigma], dim=-1)


def _profile_refit(
    model: FNO2d,
    obs: ObservationSet,
    sigma_eff2: float,
    fixed_index: int,
    fixed_value: float,
    u_seed: torch.Tensor,
    *,
    adam_steps: int = 150,
    adam_lr: float = 0.05,
    lbfgs_steps: int = 30,
) -> tuple[float, torch.Tensor]:
    """Re-optimize the free three void scalars at a pinned ``fixed_value``.

    Minimizes the Gaussian NLL over ``u`` (the pinned coord's ``u`` is inert),
    Adam then an L-BFGS polish, returning ``(nll_min, theta)``.
    """
    u = u_seed.clone().detach().to(obs.spatial.device).requires_grad_(True)

    def nll():
        theta = _theta_profile(u, fixed_index, fixed_value)
        return neg_log_likelihood(model, obs, theta, sigma_eff2)

    adam = torch.optim.Adam([u], lr=adam_lr)
    for _ in range(adam_steps):
        adam.zero_grad()
        loss = nll()
        loss.backward()
        adam.step()

    lbfgs = torch.optim.LBFGS([u], max_iter=lbfgs_steps, line_search_fn="strong_wolfe")

    def closure():
        lbfgs.zero_grad()
        loss = nll()
        loss.backward()
        return loss

    lbfgs.step(closure)

    with torch.no_grad():
        theta = _theta_profile(u, fixed_index, fixed_value).detach().cpu()
        final = float(neg_log_likelihood(model, obs, theta, sigma_eff2))
    return final, theta


@dataclass
class ProfileResult:
    param_index: int
    param_name: str
    level: float
    grid: np.ndarray            # (G,) pinned physical values
    nll: np.ndarray             # (G,) profiled NLL at each grid value
    excess_int: np.ndarray      # (G,) integrated severity along the profile
    nll_min: float
    ci_low: float               # physical param CI (Wilks)
    ci_high: float
    excess_ci_low: float        # severity CI (lead deliverable)
    excess_ci_high: float


def _threshold_crossings(
    grid: np.ndarray, delta: np.ndarray, thresh: float
) -> tuple[float, float]:
    """Linear-interpolated ``[lo, hi]`` where ``delta(grid) <= thresh``.

    ``delta`` is ``NLL - NLL_min`` over the grid (assumed roughly U-shaped). The
    interval is the outermost crossings of ``thresh``; if the boundary itself is
    below ``thresh`` the interval is open there (clamped to the grid edge).
    """
    inside = delta <= thresh
    if not inside.any():
        k = int(np.argmin(delta))
        return float(grid[k]), float(grid[k])
    idx = np.where(inside)[0]
    i0, i1 = idx[0], idx[-1]

    def interp(a, b):
        da, db = delta[a], delta[b]
        if db == da:
            return float(grid[a])
        w = (thresh - da) / (db - da)
        return float(grid[a] + w * (grid[b] - grid[a]))

    lo = float(grid[i0]) if i0 == 0 else interp(i0, i0 - 1)
    hi = float(grid[i1]) if i1 == len(grid) - 1 else interp(i1, i1 + 1)
    return lo, hi


def profile_likelihood(
    model: FNO2d,
    obs: ObservationSet,
    theta_hat: torch.Tensor,
    sigma_eff2: float,
    *,
    param_index: int = 1,
    n_grid: int = 11,
    span: float = 0.6,
    level: float = 0.95,
    adam_steps: int = 150,
    lbfgs_steps: int = 30,
) -> ProfileResult:
    """Profile-likelihood interval for one void scalar (default ``R_amp``).

    Pins ``theta[param_index]`` to each of ``n_grid`` physical values spanning
    ``theta_hat[param_index] +/- span`` (clamped to the parameter's physical
    range), re-fits the other three by minimizing the Gaussian NLL, and traces
    the profile. The confidence interval uses Wilks' theorem
    (``2*(NLL - NLL_min) <= chi2_{1,level}``). Crucially we also record the
    integrated severity along the profile and report its min/max inside the CI:
    that severity interval is the well-conditioned lead deliverable, while the
    ``R_amp`` CI itself is expected to be wide / boundary-limited.
    """
    name = VOID_PARAM_NAMES[param_index]
    if param_index == 0:
        lo_b, hi_b = RC_VOID_RANGES["R_base"]
    elif param_index == 1:
        lo_b, hi_b = 0.0, R_PEAK_MAX - RC_MIN
    elif param_index == 2:
        lo_b, hi_b = RC_VOID_RANGES["y0"]
    else:
        lo_b, hi_b = RC_VOID_RANGES["sigma"]

    center = float(theta_hat[param_index])
    grid = np.linspace(
        max(lo_b, center - span), min(hi_b, center + span), int(n_grid)
    )
    u_seed = unconstrained_from_theta(theta_hat.detach().cpu())

    nll = np.empty(len(grid), dtype=np.float64)
    excess = np.empty(len(grid), dtype=np.float64)
    for i, v in enumerate(grid):
        val, theta = _profile_refit(
            model, obs, sigma_eff2, param_index, float(v), u_seed,
            adam_steps=adam_steps, lbfgs_steps=lbfgs_steps,
        )
        nll[i] = val
        excess[i] = integrated_excess_resistance(theta, obs.y_grid)

    nll_min = float(nll.min())
    thresh = _CHI2_HALF_THRESH.get(level, 1.920729)
    delta = nll - nll_min
    ci_low, ci_high = _threshold_crossings(grid, delta, thresh)
    inside = delta <= thresh
    exc_in = excess[inside] if inside.any() else excess[np.argmin(delta):np.argmin(delta) + 1]
    return ProfileResult(
        param_index=param_index, param_name=name, level=level,
        grid=grid, nll=nll, excess_int=excess, nll_min=nll_min,
        ci_low=ci_low, ci_high=ci_high,
        excess_ci_low=float(exc_in.min()), excess_ci_high=float(exc_in.max()),
    )


def profile_summary(res: ProfileResult, obs: ObservationSet) -> dict:
    """Flatten a :func:`profile_likelihood` result into CSV columns."""
    out = {
        "profile_param": res.param_name,
        "profile_level": res.level,
        f"profile_{res.param_name}_ci_low": res.ci_low,
        f"profile_{res.param_name}_ci_high": res.ci_high,
        "profile_excess_ci_low": res.excess_ci_low,
        "profile_excess_ci_high": res.excess_ci_high,
        "profile_excess_ci_width": res.excess_ci_high - res.excess_ci_low,
    }
    if obs.theta_true is not None:
        t = obs.theta_true
        excess_true = integrated_excess_resistance(t, obs.y_grid)
        out["profile_excess_covered"] = bool(
            res.excess_ci_low <= excess_true <= res.excess_ci_high
        )
        out[f"profile_{res.param_name}_covered"] = bool(
            res.ci_low <= float(t[res.param_index]) <= res.ci_high
        )
    return out


# --- Laplace degeneracy diagnostic (diagnostic only) -----------------------

def laplace_spectrum(
    model: FNO2d, obs: ObservationSet, theta_hat: torch.Tensor, sigma_eff2: float
) -> dict:
    """Gauss-Newton eigen-spectrum at ``theta_hat`` (degeneracy diagnostic).

    Builds the observation Jacobian ``J = dG/dtheta`` and the Gauss-Newton
    Hessian of the NLL ``H = J^T J / sigma_eff2``. Its eigenvalues are
    ``S_k^2 / sigma_eff2`` and eigenvectors the right singular vectors of ``J``,
    so this reuses :func:`svd_identifiability`. Small eigenvalues + an
    ``(R_amp, sigma)``-aligned least-identified direction confirm the ridge the
    profile/MCMC must handle. Reported as a diagnostic; NOT used for intervals
    (the Gaussian-around-MAP is least valid exactly along this degenerate
    direction and at the R_amp~0 boundary).
    """
    J = observation_jacobian(model, obs, theta_hat).cpu().numpy()
    ident = svd_identifiability(J)
    S = ident["singular_values"]
    eig = (S ** 2) / float(sigma_eff2)
    return {
        "eigenvalues": eig,                       # descending
        "right_vectors": ident["right_vectors"],
        "least_identified_dir": ident["least_identified_dir"],
        "ramp_sigma_alignment": ident["ramp_sigma_alignment"],
        "cond_number": float(eig[0] / eig[-1]) if eig[-1] > 0 else float("inf"),
    }


def laplace_summary(spec: dict) -> dict:
    """Flatten a :func:`laplace_spectrum` into CSV columns."""
    eig = spec["eigenvalues"]
    least = spec["least_identified_dir"]
    out = {
        "laplace_cond": spec["cond_number"],
        "laplace_ramp_sigma_align": spec["ramp_sigma_alignment"],
    }
    for i in range(len(eig)):
        out[f"laplace_eig_{i}"] = float(eig[i])
    for i, nm in enumerate(VOID_PARAM_NAMES):
        out[f"laplace_least_dir_{nm}"] = float(least[i])
    return out


# --- Short MCMC (primary, Bayesian with a uniform-theta prior) --------------

@dataclass
class MCMCResult:
    theta_samples: np.ndarray   # (S, 4) physical
    excess_int: np.ndarray      # (S,) integrated severity per sample
    accept_rate: float
    level: float


def run_mcmc(
    model: FNO2d,
    obs: ObservationSet,
    theta_hat: torch.Tensor,
    sigma_eff2: float,
    *,
    n_samples: int = 2000,
    burn: int = 500,
    step_size: float = 0.05,
    level: float = 0.95,
    seed: int = 0,
) -> MCMCResult:
    """Random-walk Metropolis in ``u`` with a uniform-theta prior.

    The target is ``log p(u) = -NLL(theta(u)) + log|det d theta/du|``: the
    Jacobian term pushes a *uniform prior in physical theta* (matching data
    generation) through the unconstrained reparameterization, so the box
    constraints are respected automatically and the boundary at ``R_amp ~ 0`` is
    handled honestly (no Gaussian-around-MAP assumption). Captures the
    non-Gaussian, boundary-coupled posterior the Laplace diagnostic flags.
    """
    device = obs.spatial.device
    rng = np.random.default_rng(int(seed))
    u = unconstrained_from_theta(theta_hat.detach().cpu()).to(device)

    def log_target(u_t: torch.Tensor) -> float:
        with torch.no_grad():
            theta = theta_from_unconstrained(u_t)
            nll = float(neg_log_likelihood(model, obs, theta, sigma_eff2))
            ladj = float(theta_logabsdet_du(u_t))
        return -nll + ladj

    cur_lp = log_target(u)
    samples, n_acc = [], 0
    total = int(burn) + int(n_samples)
    for it in range(total):
        prop = u + torch.from_numpy(
            (rng.standard_normal(4) * step_size).astype(np.float32)
        ).to(device)
        prop_lp = log_target(prop)
        if np.log(rng.uniform()) < (prop_lp - cur_lp):
            u, cur_lp = prop, prop_lp
            n_acc += 1
        if it >= burn:
            samples.append(theta_from_unconstrained(u).detach().cpu().numpy())

    theta_samples = np.asarray(samples, dtype=np.float64)
    excess = np.array(
        [
            integrated_excess_resistance(
                torch.from_numpy(s.astype(np.float32)), obs.y_grid
            )
            for s in theta_samples
        ]
    )
    return MCMCResult(
        theta_samples=theta_samples,
        excess_int=excess,
        accept_rate=n_acc / max(1, total),
        level=level,
    )


def mcmc_summary(res: MCMCResult, obs: ObservationSet) -> dict:
    """Posterior summaries from an MCMC run, led by the severity interval."""
    alpha = (1.0 - res.level) / 2.0
    qlo, qhi = 100.0 * alpha, 100.0 * (1.0 - alpha)
    exc_lo, exc_hi = np.percentile(res.excess_int, [qlo, qhi])
    out = {
        "mcmc_accept_rate": res.accept_rate,
        "mcmc_excess_mean": float(np.mean(res.excess_int)),
        "mcmc_excess_ci_low": float(exc_lo),
        "mcmc_excess_ci_high": float(exc_hi),
        "mcmc_excess_ci_width": float(exc_hi - exc_lo),
    }
    for i, nm in enumerate(VOID_PARAM_NAMES):
        col = res.theta_samples[:, i]
        lo, hi = np.percentile(col, [qlo, qhi])
        out[f"mcmc_{nm}_mean"] = float(np.mean(col))
        out[f"mcmc_{nm}_ci_low"] = float(lo)
        out[f"mcmc_{nm}_ci_high"] = float(hi)
    if obs.theta_true is not None:
        t = obs.theta_true
        excess_true = integrated_excess_resistance(t, obs.y_grid)
        out["mcmc_excess_covered"] = bool(exc_lo <= excess_true <= exc_hi)
    return out


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="Path to source_itr fno2d_best.pt")
    ap.add_argument("--data-dir", required=True, help="Dir with trajectories/sim_params/grids")
    ap.add_argument("--sim-ids", type=int, nargs="*", default=None,
                    help="Sim ids to invert (default: first few test-split sims)")
    ap.add_argument("--n-sims", type=int, default=3, help="How many test sims if --sim-ids omitted")
    ap.add_argument("--time-indices", type=int, nargs="*", default=None,
                    help="Observation time indices (default: early/mid/late)")
    ap.add_argument("--n-starts", type=int, default=8)
    ap.add_argument("--adam-steps", type=int, default=400)
    ap.add_argument("--adam-lr", type=float, default=0.05)
    ap.add_argument("--lbfgs-steps", type=int, default=50)
    ap.add_argument("--reg-weight", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-csv", default=None)
    # Stage 2 sensor operator. Omit --sensor-x-halfwidth for full-field (Stage 1).
    ap.add_argument("--sensor-x-halfwidth", type=float, default=None,
                    help="Half-width of the x-band around the interface; omit for full-field")
    ap.add_argument("--sensor-n-y", type=int, default=8,
                    help="Number of y sensor rows in the interface band")
    ap.add_argument("--interface-x", type=float, default=0.5,
                    help="Interface x location for sensor placement (source_itr: 0.5)")
    ap.add_argument("--self-consistency", action="store_true",
                    help="Invert against the FNO's own output at theta_true "
                         "(isolates operator wiring from surrogate accuracy)")
    # Stage 3 diagnostics.
    ap.add_argument("--start-sampling", choices=["lhs", "normal"], default="lhs",
                    help="Multistart sampling over the theta-box (default: lhs)")
    ap.add_argument("--sensitivity", action="store_true",
                    help="Report dG/dtheta SVD identifiability at theta_hat "
                         "(condition number + R_amp/sigma ridge alignment)")
    # Stage 4: FV refinement. The FNO MAP is re-checked against the real solver.
    ap.add_argument("--fv-refine", action="store_true",
                    help="Re-evaluate theta_hat with the real FVSolver2D and "
                         "report fno/fv/fno-vs-fv sensor residuals (C_FNO at theta_hat)")
    ap.add_argument("--fv-polish", action="store_true",
                    help="Derivative-free (Nelder-Mead) FV polish of theta_hat "
                         "against the FV sensor residual (implies --fv-refine)")
    ap.add_argument("--fv-polish-maxiter", type=int, default=60,
                    help="Max Nelder-Mead iterations for --fv-polish (default 60)")
    # Stage 5: measurement noise + uncertainty quantification.
    ap.add_argument("--noise-std", type=float, default=0.0,
                    help="Std of iid Gaussian sensor noise (normalized temp units); "
                         "0 disables noise")
    ap.add_argument("--noise-seed", type=int, default=0,
                    help="Base seed for measurement noise (per-sim seed = noise-seed + sim_id)")
    ap.add_argument("--uq-level", type=float, default=0.95,
                    help="Confidence/credible level for profile + MCMC intervals")
    ap.add_argument("--profile", action="store_true",
                    help="Profile-likelihood UQ (primary); leads with the integrated-"
                         "severity interval. Needs a noise model (--noise-std or --fv-refine)")
    ap.add_argument("--profile-index", type=int, default=1,
                    help="Void scalar to profile: 0=R_base 1=R_amp 2=y0 3=sigma (default R_amp)")
    ap.add_argument("--profile-grid", type=int, default=11,
                    help="Number of pinned grid points for --profile (default 11)")
    ap.add_argument("--profile-span", type=float, default=0.6,
                    help="Half-width of the profiled-parameter grid around theta_hat")
    ap.add_argument("--laplace", action="store_true",
                    help="Laplace Gauss-Newton eigen-spectrum (degeneracy DIAGNOSTIC only, "
                         "not reported as an interval)")
    ap.add_argument("--mcmc", action="store_true",
                    help="Short RW-Metropolis MCMC UQ (primary, uniform-theta prior). "
                         "Needs a noise model (--noise-std or --fv-refine)")
    ap.add_argument("--mcmc-samples", type=int, default=2000)
    ap.add_argument("--mcmc-burn", type=int, default=500)
    ap.add_argument("--mcmc-step", type=float, default=0.05,
                    help="RW-Metropolis proposal std in u-space (default 0.05)")
    args = ap.parse_args(argv)

    loaded = load_checkpoint(args.checkpoint, device=args.device)
    ds = build_dataset_from_dir(
        args.data_dir, loaded.config,
        mu_global=loaded.mu_global, sigma_global=loaded.sigma_global,
    )

    if args.sim_ids is not None:
        sim_ids = list(args.sim_ids)
    else:
        sim_ids = list(ds._split_ids["test"][: args.n_sims])

    time_indices = args.time_indices or _default_time_indices(ds.Nt)
    cfg = InversionConfig(
        n_starts=args.n_starts, adam_steps=args.adam_steps, adam_lr=args.adam_lr,
        lbfgs_steps=args.lbfgs_steps, reg_weight=args.reg_weight, seed=args.seed,
        device=args.device, start_sampling=args.start_sampling,
    )

    do_uq = args.profile or args.laplace or args.mcmc
    do_fv = args.fv_refine or args.fv_polish
    fv_base_kwargs = build_fv_base_kwargs(ds) if do_fv else None

    rows = []
    for sid in sim_ids:
        obs = build_observation_set(
            ds, int(sid), time_indices, device=args.device,
            interface_x=args.interface_x,
            sensor_x_halfwidth=args.sensor_x_halfwidth,
            sensor_n_y=args.sensor_n_y,
        )
        if args.self_consistency:
            with torch.no_grad():
                obs.targets = predict_fullfield(
                    loaded.model, obs, obs.theta_true
                ).detach()
        if args.noise_std > 0:
            add_measurement_noise(obs, args.noise_std, seed=args.noise_seed + int(sid))
        result = invert_sim(loaded.model, obs, cfg)
        summary = summarize(result, obs.y_grid)
        summary["sim_id"] = int(sid)
        summary["time_indices"] = ";".join(str(t) for t in time_indices)
        n_sensors = (
            int(obs.spatial.shape[1] * obs.spatial.shape[2])
            if obs.mask is None
            else int(obs.mask.sum())
        )
        summary["n_sensors"] = n_sensors
        summary["self_consistency"] = bool(args.self_consistency)
        if args.sensitivity:
            theta_eval = torch.tensor(
                [summary["R_base_hat"], summary["R_amp_hat"],
                 summary["y0_hat"], summary["sigma_hat"]],
                dtype=torch.float32, device=args.device,
            )
            report = sensitivity_report(loaded.model, obs, theta_eval)
            summary.update(sensitivity_summary(report))
        if do_fv:
            theta_hat = torch.tensor(
                [summary["R_base_hat"], summary["R_amp_hat"],
                 summary["y0_hat"], summary["sigma_hat"]],
                dtype=torch.float32, device=args.device,
            )
            fv_res = fv_refine_report(
                loaded.model, ds, fv_base_kwargs, obs, theta_hat,
                mu_global=loaded.mu_global, sigma_global=loaded.sigma_global,
                polish=args.fv_polish, polish_maxiter=args.fv_polish_maxiter,
            )
            summary.update(fv_refine_summary(fv_res, obs, obs.y_grid))
        if do_uq:
            theta_hat = torch.tensor(
                [summary["R_base_hat"], summary["R_amp_hat"],
                 summary["y0_hat"], summary["sigma_hat"]],
                dtype=torch.float32, device=args.device,
            )
            # theta-local surrogate error C_FNO(theta_hat) from Stage 4 (0 if FV
            # refinement was not run); C_total = C_meas + C_FNO.
            c_fno = float(summary.get("fno_vs_fv_resid", 0.0))
            sigma_eff2 = effective_sigma2(args.noise_std, c_fno)
            summary["uq_sigma_eff2"] = sigma_eff2
            summary["uq_c_fno"] = 2.0 * c_fno
            if sigma_eff2 <= 0:
                print(
                    f"[sim {sid}] UQ skipped: sigma_eff2<=0 "
                    f"(supply --noise-std and/or --fv-refine for a noise model)"
                )
            else:
                if args.profile:
                    prof = profile_likelihood(
                        loaded.model, obs, theta_hat, sigma_eff2,
                        param_index=args.profile_index, n_grid=args.profile_grid,
                        span=args.profile_span, level=args.uq_level,
                    )
                    summary.update(profile_summary(prof, obs))
                if args.laplace:
                    spec = laplace_spectrum(loaded.model, obs, theta_hat, sigma_eff2)
                    summary.update(laplace_summary(spec))
                if args.mcmc:
                    mc = run_mcmc(
                        loaded.model, obs, theta_hat, sigma_eff2,
                        n_samples=args.mcmc_samples, burn=args.mcmc_burn,
                        step_size=args.mcmc_step, level=args.uq_level,
                        seed=args.seed + int(sid),
                    )
                    summary.update(mcmc_summary(mc, obs))
        rows.append(summary)
        print(
            f"[sim {sid}] loss={summary['loss']:.3e}  n_sensors={n_sensors}  "
            f"R_base {summary['R_base_hat']:.3f}/{summary.get('R_base_true', float('nan')):.3f}  "
            f"R_amp {summary['R_amp_hat']:.3f}/{summary.get('R_amp_true', float('nan')):.3f}  "
            f"y0 {summary['y0_hat']:.3f}/{summary.get('y0_true', float('nan')):.3f}  "
            f"sigma {summary['sigma_hat']:.3f}/{summary.get('sigma_true', float('nan')):.3f}  "
            f"excess_int {summary['excess_int_hat']:.4f}/{summary.get('excess_int_true', float('nan')):.4f}"
            + (
                f"  cond={summary['cond_number']:.2e}  "
                f"ramp_sigma_align={summary['ramp_sigma_alignment']:.3f}"
                if args.sensitivity else ""
            )
            + (
                f"  fno_resid={summary['fno_resid']:.3e}  "
                f"fv_resid={summary['fv_resid']:.3e}  "
                f"C_FNO={summary['fno_vs_fv_resid']:.3e}"
                + (f"  fv_polish_resid={summary['fv_polish_resid']:.3e}"
                   if args.fv_polish else "")
                if do_fv else ""
            )
            + (
                f"  prof_excess_CI=[{summary['profile_excess_ci_low']:.4f},"
                f"{summary['profile_excess_ci_high']:.4f}]"
                if args.profile and "profile_excess_ci_low" in summary else ""
            )
            + (
                f"  mcmc_excess_CI=[{summary['mcmc_excess_ci_low']:.4f},"
                f"{summary['mcmc_excess_ci_high']:.4f}]"
                f" acc={summary['mcmc_accept_rate']:.2f}"
                if args.mcmc and "mcmc_excess_ci_low" in summary else ""
            )
            + (
                f"  laplace_cond={summary['laplace_cond']:.2e}"
                if args.laplace and "laplace_cond" in summary else ""
            )
        )

    if args.out_csv and rows:
        fieldnames = sorted({k for r in rows for k in r})
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} rows to {args.out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
