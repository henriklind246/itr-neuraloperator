"""Shared loader + helpers for the forcing-PINO diagnostic suite.

These scripts localize why the physics-only diffusion-forcing benchmark plateaus
around 5% val gnrmse (vs the data-driven forcing-with-interface run below 1%).
Every model-loading diagnostic MUST recover its config FROM THE CHECKPOINT and
its run directory, never from the current ``conf/config.yaml`` defaults: a
diagnostic that silently rebuilds the model or FV problem from live defaults may
not match the checkpoint and is invalid.

Metric convention (report in every script): the trainer's val number is
``gnrmse = rmse_K / sigma_global`` on the DEVIATION field ``T - T_RIGHT``
(``validate_forcing_gnrmse``, ``train_pino.py``), not a raw rel-L2. ``T_RIGHT``
cancels in the deviation error, so ``gnrmse = rmse_K / sigma``. Numbers here are
therefore directly comparable to ``val_gnrmse`` in ``train_metrics.csv``.

The loader/meta scaffolding (``DATA_FILE_NAMES``, ``_checkpoint_path``,
``_sync_data_dir``, ``_forcing_image_meta``, ``_q_grid``) mirrors
``scripts/diagnose_forcing_hard_left.py``; that script is left untouched.
"""

from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from problems.forcing import A_AMP_REF
from src.operators.train_pino import (
    build_cvit,
    build_forcing_image,
    load_diffusion_data,
)
from src.operators.utils import resolve_device
from src.physics.boundary_forcing import reconstruct_qL


DATA_FILE_NAMES = {
    "t_grid_path": "t_grid.npy",
    "x_grid_path": "x_grid.npy",
    "y_grid_path": "y_grid.npy",
    "trajectories.npy": "trajectories.npy",
    "sim_params_path": "sim_params.npy",
}


def _checkpoint_path(path: str) -> Path:
    """Resolve a seed dir OR an explicit ``.pt`` file to the checkpoint file."""
    p = Path(path).expanduser()
    return p / "cvit_best.pt" if p.is_dir() else p


def _sync_data_dir(config: dict[str, Any], data_dir: Path) -> None:
    """Point the checkpoint's stored config at a local dataset directory."""
    config.setdefault("paths", {})["data_dir"] = str(data_dir)
    config.setdefault("data", {})
    for key, name in DATA_FILE_NAMES.items():
        config["data"][key] = str(data_dir / name)


# --------------------------------------------------------------------------- #
# Exact checkpoint-config recovery (two tiers).                               #
# --------------------------------------------------------------------------- #

# Prediction-critical fields are required NOW: without them a forward pass is
# either impossible or silently wrong. Solver-critical fields are only needed
# when a Phase 2 fresh FV re-solve is invoked; they are optional here and the
# Phase 2 script validates their presence and fails loudly only then.
_SOLVER_CRITICAL_KEYS = ("fv_dt", "k_slab", "cp", "forcing_ranges")


@dataclass(frozen=True)
class CkptConfig:
    """Frozen record of the config recovered from a checkpoint + its run dir.

    Prediction-critical fields (``config``, ``mu``/``sigma``, forcing-image
    metadata, hard-constraint flags) are always populated or ``recover_config``
    raises. ``solver`` holds the optional solver-critical values (all may be
    ``None`` for an old checkpoint); Phase 2 validates them. ``provenance`` maps
    each resolved field to the source it came from ("checkpoint", "config",
    "data", "default") and ``warnings`` records loud fallbacks so no rebuild is
    silent.
    """

    config: dict[str, Any]
    mu: float
    sigma: float
    ny_img: int
    nt_img: int
    a_ref: float
    t_ramp: float
    c_dom: float
    d_dom: float
    t_final: float
    hard_left_flux: bool
    hard_right_dirichlet: bool
    provenance: dict[str, str] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    solver: dict[str, Any] = field(default_factory=dict)

    @property
    def grid_size(self) -> tuple[int, int]:
        """Encoder-image ``(Ny_img, Nt_img)`` grid for ``build_cvit``."""
        return (int(self.ny_img), int(self.nt_img))


def _forcing_cvit_cfg(config: dict[str, Any]) -> dict[str, Any]:
    """Merged forcing-variant CViT config: ``forcing_cvit`` overrides ``cvit``."""
    model = config.get("model")
    if not isinstance(model, dict) or "cvit" not in model:
        raise KeyError(
            "Checkpoint config is missing model.cvit; cannot recover model "
            "architecture. This checkpoint predates the CViT config block or is "
            "not a forcing-CViT checkpoint."
        )
    return {**model["cvit"], **(model.get("forcing_cvit") or {})}


def recover_config(
    ckpt: dict[str, Any],
    data: dict[str, Any] | None = None,
) -> CkptConfig:
    """Recover a :class:`CkptConfig` from a loaded checkpoint dict.

    Prediction-critical values are resolved with this precedence and a loud
    warning whenever the authoritative checkpoint source is absent:
    ``ckpt["forcing_image"]`` (authoritative) -> stored config's forcing block
    -> dataset grids / constants. ``model_state`` and ``model.cvit`` must exist
    or this raises. ``data`` (from :func:`load_diffusion_data`) is only needed
    to supply grid-derived fallbacks when the checkpoint omits the
    ``forcing_image`` block; pass it whenever available.
    """
    if "model_state" not in ckpt:
        raise KeyError("Checkpoint has no 'model_state'; cannot rebuild the model.")
    config = ckpt.get("config")
    if not isinstance(config, dict):
        raise KeyError("Checkpoint has no 'config' dict; cannot recover architecture.")

    ccfg = _forcing_cvit_cfg(config)  # raises if model.cvit is missing
    provenance: dict[str, str] = {"model_state": "checkpoint", "model.cvit": "config"}
    warns: list[str] = []

    # Normalization (mu, sigma): checkpoint is authoritative; else the dataset's
    # global stats; else fail. Per-sample normalization is never used.
    def _resolve_norm(key: str, data_key: str) -> float:
        if key in ckpt and ckpt[key] is not None:
            provenance[key] = "checkpoint"
            return float(ckpt[key])
        if data is not None and data_key in data:
            provenance[key] = "data"
            warns.append(
                f"{key} not in checkpoint; using dataset {data_key}. If this "
                "dataset differs from the training set the normalization is wrong."
            )
            return float(data[data_key])
        raise KeyError(
            f"Cannot resolve {key}: absent from checkpoint and no dataset provided."
        )

    mu = _resolve_norm("mu_global", "mu_global")
    sigma = _resolve_norm("sigma_global", "sigma_global")

    fcfg = (
        config.get("training", {}).get("pino", {}).get("forcing", {}) or {}
    )
    fimg = dict(ckpt.get("forcing_image") or {})
    have_block = bool(fimg)
    if not have_block:
        warns.append(
            "Checkpoint has no 'forcing_image' block; inferring encoder-image "
            "metadata (ny_img, nt_img, a_ref, t_ramp, domain, t_final) from the "
            "stored config and dataset grids. Verify this matches the run."
        )

    y_grid = (
        np.asarray(data["y_grid"], dtype=np.float64) if data is not None else None
    )
    t_grid = (
        np.asarray(data["t_grid"], dtype=np.float64) if data is not None else None
    )

    def _resolve_img(name: str, cfg_key: str, data_default):
        if name in fimg and fimg[name] is not None:
            provenance[name] = "checkpoint"
            return fimg[name]
        if cfg_key is not None and fcfg.get(cfg_key) is not None:
            provenance[name] = "config"
            return fcfg[cfg_key]
        if data_default is None:
            raise KeyError(
                f"Cannot resolve prediction-critical '{name}': absent from "
                "checkpoint 'forcing_image', config, and no dataset fallback."
            )
        provenance[name] = "default"
        return data_default() if callable(data_default) else data_default

    ny_img = int(_resolve_img("ny_img", "ny_img", (lambda: int(y_grid.shape[0])) if y_grid is not None else None))
    nt_img = int(_resolve_img("nt_img", "nt_img", 128))
    a_ref = float(_resolve_img("a_ref", "a_ref", A_AMP_REF))
    t_ramp = float(_resolve_img("t_ramp", "ramp_seconds", 0.0))
    c_dom = float(_resolve_img("c_dom", None, (lambda: float(y_grid[0])) if y_grid is not None else None))
    d_dom = float(_resolve_img("d_dom", None, (lambda: float(y_grid[-1])) if y_grid is not None else None))
    t_final = float(_resolve_img("t_final", None, (lambda: float(t_grid[-1])) if t_grid is not None else None))

    hard_lf = bool(ccfg.get("hard_left_flux", False))
    hard_rd = bool(ccfg.get("hard_right_dirichlet", True))
    provenance["hard_left_flux"] = "config"
    provenance["hard_right_dirichlet"] = "config"

    # Solver-critical (optional here): only Phase 2 fresh FV solves need these.
    # ``fv_dt`` is the FV Crank-Nicolson timestep. It is rarely stored under the
    # forcing block; fall back to ``training.physics.dt`` (the solve dt the data
    # generator used) then the forcing ``dt_sample`` mirror, recording which
    # source won. ``k_slab``/``cp`` are hardcoded to 1.0 in the diffusion_forcing
    # ``configure_solver`` (single homogeneous slab), so they stay ``None`` here
    # and Phase 2 does not require them.
    solver: dict[str, Any] = {}
    solver_prov: dict[str, str] = {}
    phys = config.get("training", {}).get("physics", {}) or {}
    fv_dt_candidates = (
        ("forcing_image.fv_dt", fimg.get("fv_dt")),
        ("training.pino.forcing.fv_dt", fcfg.get("fv_dt")),
        ("training.physics.dt", phys.get("dt")),
        ("training.pino.forcing.dt_sample", fcfg.get("dt_sample")),
    )
    fv_dt = None
    for src, val in fv_dt_candidates:
        if val is not None:
            fv_dt = float(val)
            solver_prov["fv_dt"] = src
            break
    solver["fv_dt"] = fv_dt
    for k in ("k_slab", "cp", "forcing_ranges"):
        val = fcfg.get(k)
        solver[k] = val if val is not None else None
        if val is not None:
            solver_prov[k] = "training.pino.forcing"
    provenance.update(solver_prov)

    for w in warns:
        warnings.warn(w, RuntimeWarning, stacklevel=2)

    return CkptConfig(
        config=config,
        mu=mu,
        sigma=sigma,
        ny_img=ny_img,
        nt_img=nt_img,
        a_ref=a_ref,
        t_ramp=t_ramp,
        c_dom=c_dom,
        d_dom=d_dom,
        t_final=t_final,
        hard_left_flux=hard_lf,
        hard_right_dirichlet=hard_rd,
        provenance=provenance,
        warnings=tuple(warns),
        solver=solver,
    )


def load_model_and_data(
    checkpoint: str,
    data_dir: str | None,
    device: str = "auto",
) -> tuple[torch.nn.Module, dict[str, Any], CkptConfig]:
    """Load the forcing CViT + its dataset with exact checkpoint-config recovery.

    Returns ``(model.eval(), data, cfg)``. ``data_dir`` (or the ``DATA_DIR`` env
    var) supplies the local probe dataset the checkpoint's config is re-pointed
    at; ``load_diffusion_data`` then reads grids/trajectories/global stats.
    """
    ckpt_path = _checkpoint_path(checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config")
    if not isinstance(config, dict):
        raise KeyError("Checkpoint has no 'config' dict; cannot recover architecture.")

    ddir = Path(data_dir or os.environ.get("DATA_DIR", "")).expanduser()
    if str(ddir):
        _sync_data_dir(config, ddir)
    data = load_diffusion_data(config)

    cfg = recover_config(ckpt, data)
    dev = resolve_device(device)
    model = build_cvit(
        cfg.config,
        cfg.mu,
        cfg.sigma,
        grid_size=cfg.grid_size,
        t_final=cfg.t_final,
        variant="forcing",
    ).to(dev)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, data, cfg


# --------------------------------------------------------------------------- #
# Encoder-image / q_left tensor construction (mesh order: flat = ix*Ny + iy).  #
# --------------------------------------------------------------------------- #

def _mesh_coords(data: dict[str, Any], device: torch.device) -> torch.Tensor:
    """Query coords ``(1, Nx*Ny, 2)`` in mesh order (y fastest)."""
    x_grid = torch.as_tensor(data["x_grid"], dtype=torch.float32, device=device)
    y_grid = torch.as_tensor(data["y_grid"], dtype=torch.float32, device=device)
    gx, gy = torch.meshgrid(x_grid, y_grid, indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1).unsqueeze(0)


def _q_left_grid(
    params: list[dict[str, Any]],
    y_grid_np: np.ndarray,
    t_values: np.ndarray,
    nx: int,
    t_ramp: float,
    device: torch.device,
) -> torch.Tensor:
    """Hard-left ``q_L`` tiled into mesh order for each queried time.

    Returns ``(B, T, Nx*Ny, 1)``. ``q_L`` depends only on ``(y, t)``, so it is
    evaluated on the ``(Ny, T)`` grid via the vectorized ``q_image`` (one
    ``reconstruct_qL`` per sim) then tiled across x with ``np.tile(qg.T, (1,
    Nx))`` (flat index ``ix*Ny + iy``, y fastest) exactly as the trainer does.
    """
    bsz = len(params)
    ny = int(y_grid_np.shape[0])
    nt = int(t_values.shape[0])
    q_np = np.empty((bsz, nt, nx * ny), dtype=np.float32)
    for b, p in enumerate(params):
        forcing = reconstruct_qL(
            p["temporal_family"], p["temporal_params"],
            p["spatial_family"], p["spatial_params"], t_ramp=t_ramp,
        )
        qg = np.asarray(
            forcing.evaluate_grid(y_grid_np, t_values), dtype=np.float32
        )
        q_np[b] = np.tile(qg.T, (1, nx))  # (T, Nx*Ny)
    return torch.from_numpy(q_np).unsqueeze(-1).to(device)


def _forward_grid(
    model: torch.nn.Module,
    u: torch.Tensor,
    coords: torch.Tensor,
    q_all: torch.Tensor | None,
    t_values: np.ndarray,
    device: torch.device,
    mu: float,
    sigma: float,
    nx: int,
    ny: int,
) -> torch.Tensor:
    """Single forward implementation shared by predict_grid + the pathway probe.

    ``q_all`` is ``(B, T, Nx*Ny, 1)`` when the model uses hard-left flux, else
    ``None``. Returns physical-K predictions ``(B, T, Nx, Ny)``.
    """
    B = u.shape[0]
    T = len(t_values)
    coords_e = coords.expand(B, -1, -1)
    use_left = q_all is not None
    pred = torch.empty((B, T, nx, ny), device=device)
    with torch.no_grad():
        for k, tv in enumerate(t_values):
            tk = torch.full((B, nx * ny, 1), float(tv), device=device)
            q_left = q_all[:, k] if use_left else None
            out = model(u, coords_e, tk, q_left=q_left)
            pred[:, k] = out[..., 0].view(B, nx, ny)
    return pred * sigma + mu


def predict_grid(
    model: torch.nn.Module,
    params: list[dict[str, Any]],
    data: dict[str, Any],
    cfg: CkptConfig,
    t_values: np.ndarray | None,
    device: torch.device | str = "cpu",
    query_batch: int = 8,
) -> np.ndarray:
    """Physical-K predictions ``(B, len(t_values), Nx, Ny)`` for ``params``.

    Factored out of ``validate_forcing_gnrmse``: the encoder image is built at
    the recovered ``(ny_img, nt_img)`` over ``[c_dom, d_dom] x [0, t_final]``,
    the hard-left ``q_left`` is tiled in mesh order when ``hard_left_flux`` is
    set, and predictions are denormalized with the recovered ``(mu, sigma)``.
    ``t_values`` defaults to every saved snapshot ``data["t_grid"]``.
    """
    dev = resolve_device(device) if isinstance(device, str) else device
    tv = np.asarray(data["t_grid"] if t_values is None else t_values, dtype=np.float64)
    x_grid_np = np.asarray(data["x_grid"], dtype=np.float64)
    y_grid_np = np.asarray(data["y_grid"], dtype=np.float64)
    nx, ny = int(x_grid_np.shape[0]), int(y_grid_np.shape[0])
    coords = _mesh_coords(data, dev)

    y_img = np.linspace(cfg.c_dom, cfg.d_dom, cfg.ny_img, dtype=np.float64)
    t_img = np.linspace(0.0, cfg.t_final, cfg.nt_img, dtype=np.float64)

    out = np.empty((len(params), len(tv), nx, ny), dtype=np.float32)
    for start in range(0, len(params), int(query_batch)):
        chunk = params[start : start + int(query_batch)]
        u = build_forcing_image(chunk, y_img, t_img, cfg.a_ref, dev, cfg.t_ramp)
        q_all = (
            _q_left_grid(chunk, y_grid_np, tv, nx, cfg.t_ramp, dev)
            if cfg.hard_left_flux
            else None
        )
        pred_K = _forward_grid(
            model, u, coords, q_all, tv, dev, cfg.mu, cfg.sigma, nx, ny
        )
        out[start : start + len(chunk)] = pred_K.detach().cpu().numpy()
    return out


# --------------------------------------------------------------------------- #
# Forcing-severity metrics (physically comparable across temporal families).  #
# --------------------------------------------------------------------------- #

def severity_from_grid(
    q: np.ndarray, y_grid: np.ndarray, t_dense: np.ndarray
) -> dict[str, float]:
    """Severity scalars from a dense ``q_L`` image ``q`` of shape ``(Ny, Nt)``.

    - ``q_rms`` = sqrt(mean over Gamma_L x [0,T] of q_L^2).
    - ``Q_in``  = integral over Gamma_L x [0,T] of q_L dy dt (net heat input),
      trapezoidal over y then t.
    - ``amp``   = max |q_L| (nominal peak; metadata only).
    """
    q = np.asarray(q, dtype=np.float64)
    q_rms = float(np.sqrt(np.mean(q ** 2)))
    Q_in = float(np.trapz(np.trapz(q, y_grid, axis=0), t_dense))
    amp = float(np.max(np.abs(q)))
    return {"q_rms": q_rms, "Q_in": Q_in, "amp": amp}


def forcing_severity(
    params: dict[str, Any],
    y_grid: np.ndarray,
    t_dense: np.ndarray,
    t_ramp: float = 0.0,
) -> dict[str, float]:
    """Severity scalars for one sim via ``reconstruct_qL`` on a dense grid.

    Nominal amplitude keys are not comparable across temporal families (a ``sin``
    peak and a ``pulse_train`` peak deposit very different heat); ``q_rms`` and
    ``Q_in`` are. ``amp`` is retained as metadata only.
    """
    forcing = reconstruct_qL(
        params["temporal_family"], params["temporal_params"],
        params["spatial_family"], params["spatial_params"], t_ramp=t_ramp,
    )
    q = np.asarray(forcing.evaluate_grid(y_grid, t_dense), dtype=np.float64)
    return severity_from_grid(q, np.asarray(y_grid, float), np.asarray(t_dense, float))


# --------------------------------------------------------------------------- #
# Cheap forcing-pathway ablation (no FV).                                      #
# --------------------------------------------------------------------------- #

# Each case perturbs an explicit input TENSOR (the encoder image ``u`` and/or the
# hard-left ``q_left``), because the model only consumes those two tensors -- not
# an abstract forcing-parameter dict. The CSV records exactly which tensor was
# changed. Boundary cases are flagged when the model has no hard-left flux path.
PATHWAY_CASES = (
    "baseline",
    "image_zero",
    "image_shuffled",
    "boundary_zero",
    "boundary_scaled",
    "both_shuffled",
)


def _rms_field(a: np.ndarray) -> np.ndarray:
    """Per-sim RMS over the (T, Nx, Ny) grid; ``a`` is ``(B, T, Nx, Ny)``."""
    return np.sqrt(np.mean(a.astype(np.float64) ** 2, axis=(1, 2, 3)))


def forcing_pathway_probe(
    model: torch.nn.Module,
    params: list[dict[str, Any]],
    ids: np.ndarray,
    data: dict[str, Any],
    cfg: CkptConfig,
    device: torch.device | str = "cpu",
    *,
    scale: float = 0.5,
) -> list[dict[str, Any]]:
    """Six-case forcing-pathway ablation with dual reporting.

    For each case and sim, records BOTH ``delta_vs_baseline`` (RMS-K distance of
    the ablated prediction from the baseline prediction: how much the output
    moved) AND ``gnrmse`` (ablated prediction error vs FV truth: whether the move
    helped) -- a pathway can move the output substantially in an unhelpful
    direction. Partner sims for the shuffled cases are the batch neighbours
    (roll by 1), so each sim gets a distinct other sim's image / q_left.

    Interpretation: image-zero/shuffled ~ baseline => the encoder ignores the
    image (forcing is carried by the hard-left q_left); boundary-zero/scaled ~
    baseline => q_left is unused; all cases moving little while gnrmse stays ~5%
    => an insensitive / averaged response.
    """
    dev = resolve_device(device) if isinstance(device, str) else device
    tv = np.asarray(data["t_grid"], dtype=np.float64)
    x_grid_np = np.asarray(data["x_grid"], dtype=np.float64)
    y_grid_np = np.asarray(data["y_grid"], dtype=np.float64)
    nx, ny = int(x_grid_np.shape[0]), int(y_grid_np.shape[0])
    coords = _mesh_coords(data, dev)

    y_img = np.linspace(cfg.c_dom, cfg.d_dom, cfg.ny_img, dtype=np.float64)
    t_img = np.linspace(0.0, cfg.t_final, cfg.nt_img, dtype=np.float64)

    u = build_forcing_image(params, y_img, t_img, cfg.a_ref, dev, cfg.t_ramp)
    B = u.shape[0]
    partner = np.roll(np.arange(B), -1)  # each sim's neighbour; distinct for B>=2
    u_zero = torch.zeros_like(u)
    u_shuf = u[torch.as_tensor(partner, device=dev)]

    use_left = cfg.hard_left_flux
    q_all = (
        _q_left_grid(params, y_grid_np, tv, nx, cfg.t_ramp, dev) if use_left else None
    )
    q_shuf = q_all[torch.as_tensor(partner, device=dev)] if use_left else None

    truth = np.asarray(data["trajectories"][np.asarray(ids)], dtype=np.float32)
    if truth.shape[1] != len(tv):
        raise ValueError(
            f"trajectories snapshots ({truth.shape[1]}) != t_grid ({len(tv)})."
        )

    def _run(u_in, q_in):
        return _forward_grid(
            model, u_in, coords, q_in, tv, dev, cfg.mu, cfg.sigma, nx, ny
        ).detach().cpu().numpy()

    base = _run(u, q_all)
    sigma = float(cfg.sigma)

    case_preds: dict[str, np.ndarray | None] = {
        "baseline": base,
        "image_zero": _run(u_zero, q_all),
        "image_shuffled": _run(u_shuf, q_all),
        "boundary_zero": _run(u, torch.zeros_like(q_all)) if use_left else None,
        "boundary_scaled": _run(u, q_all * float(scale)) if use_left else None,
        "both_shuffled": _run(u_shuf, q_shuf) if use_left else _run(u_shuf, None),
    }

    rows: list[dict[str, Any]] = []
    for case in PATHWAY_CASES:
        pred = case_preds[case]
        boundary_case = case in ("boundary_zero", "boundary_scaled")
        # which tensor(s) this case perturbs, recorded verbatim in the CSV.
        changed = {
            "baseline": "none",
            "image_zero": "u",
            "image_shuffled": "u",
            "boundary_zero": "q_left",
            "boundary_scaled": "q_left",
            "both_shuffled": "u+q_left" if use_left else "u",
        }[case]
        for b, sid in enumerate(np.asarray(ids)):
            if pred is None:
                rows.append({
                    "sim_id": int(sid),
                    "case": case,
                    "tensor_changed": changed,
                    "temporal_family": str(params[b].get("temporal_family", "")),
                    "spatial_family": str(params[b].get("spatial_family", "")),
                    "delta_vs_baseline_K": float("nan"),
                    "gnrmse": float("nan"),
                    "skipped": True,
                    "skip_reason": "model has no hard_left_flux path",
                })
                continue
            delta = float(_rms_field((pred[b : b + 1] - base[b : b + 1]))[0])
            gnr = float(
                np.sqrt(np.mean((pred[b] - truth[b]).astype(np.float64) ** 2)) / (sigma + 1e-8)
            )
            rows.append({
                "sim_id": int(sid),
                "case": case,
                "tensor_changed": changed,
                "temporal_family": str(params[b].get("temporal_family", "")),
                "spatial_family": str(params[b].get("spatial_family", "")),
                "delta_vs_baseline_K": delta,
                "gnrmse": gnr,
                "skipped": False,
                "skip_reason": "",
            })
    return rows


def load_sim_params(config: dict[str, Any]) -> np.ndarray:
    """Load the per-sim forcing records that live beside ``trajectories.npy``."""
    sp_path = Path(config["data"]["trajectories.npy"]).parent / "sim_params.npy"
    return np.load(str(sp_path), allow_pickle=True)


def select_ids(
    data: dict[str, Any],
    split: str = "val",
    num_sims: int = 8,
    sim_ids: str | None = None,
) -> np.ndarray:
    """Pick sim ids: explicit ``sim_ids`` override, else the split's first N."""
    if sim_ids:
        return np.asarray(
            [int(x) for x in str(sim_ids).split(",") if x.strip()], dtype=int
        )
    ids = np.asarray(data[f"{split}_ids"], dtype=int)
    return ids[: int(num_sims)]
