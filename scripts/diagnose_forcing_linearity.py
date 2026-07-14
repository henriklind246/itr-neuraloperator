"""Phase 2: forcing linearity / superposition vs a fresh FV truth.

If Phase 1 (error surface + image fidelity) does not pin the ~5% physics-only
forcing plateau on the encoder-image representation, the next sharpest probe
exploits the fact that the heat equation is LINEAR at fixed coefficients:

  * Amplitude linearity: ``G(a*q) - T0 ~ a*(G(q) - T0)``. The FV deviation field
    scales exactly with the forcing amplitude (verified rel-err ~1e-12), so a
    sublinear / saturating MODEL slope of ``||T_pred - T_RIGHT||`` vs ``a`` flags
    a weak forcing encoder rather than an optimization limit.
  * Superposition: ``G(q1 + q2) - T0 ~ (G(q1) - T0) + (G(q2) - T0)``. FV is
    exactly additive; the model's non-additivity is a direct nonlinearity gauge.

The two regimes are reported separately and NOT mixed:
  * ``id_local``  -- amplitude scaling within the trained range + superposition
    of two in-distribution forcings.
  * ``ood``       -- temporal-family and spatial-profile swaps (grafted from
    other dataset sims), a generalization stress test.

Every variant gets a FRESH FV truth via ``DiffusionForcingProblem.configure_solver``
+ ``FVSolver2D.solve`` on the recovered grid / dt / ramp (amplitude scales, and
superposition sums, BOTH the ``q_left`` and ``q_left_integral`` solver callables;
the integral term matters for pulse forcing). Because ``build_forcing_image`` is
linear in ``q_L`` (``img = q_L / a_ref``), the matching model variant is produced
by scaling / summing the encoder-image tensor (and the hard-left ``q_left`` tensor
when the checkpoint uses it), so model and FV represent the same physical forcing.

Metric convention (as everywhere in this suite): deviations are ``T - T_RIGHT``
in Kelvin; ``model_rmse_K`` is the model's error vs the fresh FV truth. The
response-match columns are ``delta_model_K = ||T_pred(var) - T_pred(base)||``,
``delta_fv_K = ||T_FV(var) - T_FV(base)||``, ``delta_ratio = delta_model_K /
max(delta_fv_K, eps)`` (flagged invalid when ``delta_fv_K < eps``), and the cosine
between the two delta fields.

Runs CPU-only against a supplied ``--checkpoint`` + the local probe dataset.

Example (smoke; numbers not scientifically meaningful):
  .venv/bin/python scripts/diagnose_forcing_linearity.py \\
    --checkpoint /path/to/cvit_best.pt \\
    --data-dir data/diffusion_forcing_probe_cvit_best --num-sims 4 --out-dir /tmp/diag_lin
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.generate_dataset import build_base_setup  # noqa: E402
from problems.diffusion_forcing import DiffusionForcingProblem, T_RIGHT  # noqa: E402
from scripts._forcing_diag_common import (  # noqa: E402
    _forward_grid,
    _mesh_coords,
    _q_left_grid,
    forcing_severity,
    load_model_and_data,
    load_sim_params,
    select_ids,
    severity_from_grid,
)
from src.operators.train_pino import build_forcing_image  # noqa: E402
from src.operators.utils import resolve_device  # noqa: E402
from src.physics.boundary_forcing import reconstruct_qL  # noqa: E402

CSV_FIELDS = [
    "test", "regime", "sim_id", "partner_id", "variant",
    "temporal_family", "spatial_family", "alpha",
    "q_rms", "Q_in",
    "dev_model_K", "dev_fv_K", "model_rmse_K",
    "delta_model_K", "delta_fv_K", "delta_ratio", "delta_ratio_valid", "cosine",
]


# --------------------------------------------------------------------------- #
# small numeric helpers                                                        #
# --------------------------------------------------------------------------- #

def _rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(a, dtype=np.float64) ** 2)))


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _match_indices(t_fv: np.ndarray, t_model: np.ndarray) -> np.ndarray:
    """Nearest FV time index for each model snapshot time."""
    t_fv = np.asarray(t_fv, dtype=np.float64)
    return np.array([int(np.argmin(np.abs(t_fv - tm))) for tm in t_model], dtype=int)


def _delta_metrics(
    dm_var: np.ndarray,
    df_var: np.ndarray,
    dm_base: np.ndarray,
    df_base: np.ndarray,
    eps: float,
) -> dict[str, Any]:
    """Response-match metrics between a variant and its base (deviation fields)."""
    delta_model = _rms(dm_var - dm_base)
    delta_fv = _rms(df_var - df_base)
    valid = delta_fv >= eps
    ratio = float(delta_model / delta_fv) if valid else float("nan")
    cos = _cos(dm_var - dm_base, df_var - df_base) if valid else float("nan")
    return {
        "dev_model_K": _rms(dm_var),
        "dev_fv_K": _rms(df_var),
        "model_rmse_K": _rms(dm_var - df_var),
        "delta_model_K": delta_model,
        "delta_fv_K": delta_fv,
        "delta_ratio": ratio,
        "delta_ratio_valid": bool(valid),
        "cosine": cos,
    }


# --------------------------------------------------------------------------- #
# FV re-solve (fresh truth per variant)                                        #
# --------------------------------------------------------------------------- #

def _fv_dev(
    spec: DiffusionForcingProblem,
    base_kwargs: dict[str, Any],
    params: dict[str, Any],
    fv_idx_ref: dict[str, Any],
    *,
    alpha: float = 1.0,
    add_params: dict[str, Any] | None = None,
) -> np.ndarray:
    """FV deviation ``T - T_RIGHT`` at the model snapshot times, ``(T, Nx, Ny)``.

    ``alpha`` scales the single-forcing ``q_left`` (and its integral); when
    ``add_params`` is given the solver's forcing is the SUM of ``params`` and
    ``add_params`` (superposition). Both the flux and its time-integral callables
    are scaled/summed so pulse forcing is handled correctly.
    """
    sim = spec.configure_solver(params, base_kwargs)
    if add_params is not None:
        other = spec.configure_solver(add_params, base_kwargs)
        qa, qia = sim.q_left, sim.q_left_integral
        qb, qib = other.q_left, other.q_left_integral
        sim.q_left = lambda t, _a=qa, _b=qb: _a(t) + _b(t)
        sim.q_left_integral = lambda t0, t1, _a=qia, _b=qib: _a(t0, t1) + _b(t0, t1)
    elif alpha != 1.0:
        q0, qi0 = sim.q_left, sim.q_left_integral
        sim.q_left = lambda t, _a=alpha, _f=q0: _a * _f(t)
        sim.q_left_integral = lambda t0, t1, _a=alpha, _f=qi0: _a * _f(t0, t1)
    t_fv, _, _, T_hist = sim.solve(T0=params["T0"], store_trajectory=True)
    if fv_idx_ref["idx"] is None:
        fv_idx_ref["idx"] = _match_indices(np.asarray(t_fv), fv_idx_ref["t_model"])
    T_sel = np.asarray(T_hist, dtype=np.float64)[fv_idx_ref["idx"]]
    return T_sel - float(T_RIGHT)


# --------------------------------------------------------------------------- #
# model deviation (image tensor drives the forward pass)                       #
# --------------------------------------------------------------------------- #

def _model_dev(
    model: torch.nn.Module,
    cfg,
    coords: torch.Tensor,
    t_model: np.ndarray,
    dev: torch.device,
    nx: int,
    ny: int,
    u: torch.Tensor,
    q_all: torch.Tensor | None,
) -> np.ndarray:
    """Model deviation ``T_pred - T_RIGHT`` for a single sim, ``(T, Nx, Ny)``."""
    pred_K = _forward_grid(
        model, u, coords, q_all, t_model, dev, cfg.mu, cfg.sigma, nx, ny
    )
    return pred_K[0].detach().cpu().numpy().astype(np.float64) - float(T_RIGHT)


def _build_image(params: dict[str, Any], cfg, dev: torch.device) -> torch.Tensor:
    y_img = np.linspace(cfg.c_dom, cfg.d_dom, cfg.ny_img, dtype=np.float64)
    t_img = np.linspace(0.0, cfg.t_final, cfg.nt_img, dtype=np.float64)
    return build_forcing_image([params], y_img, t_img, cfg.a_ref, dev, cfg.t_ramp)


def _build_qleft(
    params: dict[str, Any], cfg, y_grid_np: np.ndarray, t_model: np.ndarray,
    nx: int, dev: torch.device,
) -> torch.Tensor | None:
    if not cfg.hard_left_flux:
        return None
    return _q_left_grid([params], y_grid_np, t_model, nx, cfg.t_ramp, dev)


def _find_partner(params: list[dict[str, Any]], idx: int, key: str) -> int:
    """First sim whose ``key`` (temporal/spatial family) differs; else next idx."""
    n = len(params)
    fam = params[idx].get(key)
    for off in range(1, n):
        j = (idx + off) % n
        if params[j].get(key) != fam:
            return j
    return (idx + 1) % n


# --------------------------------------------------------------------------- #
# driver                                                                       #
# --------------------------------------------------------------------------- #

def run_linearity(
    checkpoint: str,
    data_dir: str | None,
    *,
    split: str = "val",
    num_sims: int = 6,
    sim_ids: str | None = None,
    alphas: tuple[float, ...] = (0.25, 0.5, 1.0, 1.5, 2.0),
    max_pairs: int = 4,
    do_ood: bool = True,
    delta_eps: float = 1e-2,
    out_dir: str | None = None,
    device: str = "auto",
    dense_ny: int = 128,
    dense_nt: int = 256,
) -> tuple[dict[str, Any], Path]:
    model, data, cfg = load_model_and_data(checkpoint, data_dir, device)

    fv_dt = cfg.solver.get("fv_dt")
    if fv_dt is None:
        raise SystemExit(
            "Phase 2 requires the FV timestep 'fv_dt' for fresh re-solves, but it "
            "could not be recovered from the checkpoint config (checked "
            "forcing_image.fv_dt, training.pino.forcing.fv_dt, training.physics.dt, "
            "training.pino.forcing.dt_sample). Supply a checkpoint that stores one."
        )
    fv_dt = float(fv_dt)

    dev = resolve_device(device) if isinstance(device, str) else device
    spec = DiffusionForcingProblem()

    x_grid = np.asarray(data["x_grid"], dtype=np.float64)
    y_grid = np.asarray(data["y_grid"], dtype=np.float64)
    t_model = np.asarray(data["t_grid"], dtype=np.float64)
    nx, ny = int(x_grid.shape[0]), int(y_grid.shape[0])
    coords = _mesh_coords(data, dev)

    # The recovered t_final can carry float32 noise (e.g. 0.30000001) that fails
    # build_time_grid's exact dt-divisibility check; snap the FV solve horizon to
    # the nearest dt multiple. The model encoder image still uses the trained
    # cfg.t_final; FV snapshots are matched back to the model times by index.
    t_final_fv = round(float(cfg.t_final) / fv_dt) * fv_dt
    base_kwargs = build_base_setup(
        num_sims=1, save_stride=1, nx=nx, ny=ny,
        t_final=t_final_fv, dt=fv_dt, ramp_seconds=float(cfg.t_ramp),
    )["base_kwargs"]
    fv_idx_ref: dict[str, Any] = {"t_model": t_model, "idx": None}

    sim_params = load_sim_params(cfg.config)
    ids = select_ids(data, split=split, num_sims=num_sims, sim_ids=sim_ids)
    params = [dict(sim_params[int(i)]) for i in ids]
    n = len(params)

    y_dense = np.linspace(cfg.c_dom, cfg.d_dom, dense_ny)
    t_dense = np.linspace(0.0, float(cfg.t_final), dense_nt)

    # --- cache per-sim base (alpha=1) model + FV deviation, image, severity ---
    cache: dict[int, dict[str, Any]] = {}
    for b in range(n):
        p = params[b]
        u = _build_image(p, cfg, dev)
        q_all = _build_qleft(p, cfg, y_grid, t_model, nx, dev)
        dm = _model_dev(model, cfg, coords, t_model, dev, nx, ny, u, q_all)
        df = _fv_dev(spec, base_kwargs, p, fv_idx_ref)
        cache[b] = {
            "u": u, "q_all": q_all, "dm": dm, "df": df,
            "sev": forcing_severity(p, y_dense, t_dense, cfg.t_ramp),
        }

    rows: list[dict[str, Any]] = []

    # --- amplitude linearity (id_local) ---
    amp_track: dict[int, dict[str, list[float]]] = {}
    for b in range(n):
        p = params[b]
        base = cache[b]
        amp_track[b] = {"alpha": [], "dev_model": [], "dev_fv": []}
        for a in alphas:
            if a == 1.0:
                dm, df = base["dm"], base["df"]
            else:
                u_a = base["u"] * float(a)
                q_a = base["q_all"] * float(a) if base["q_all"] is not None else None
                dm = _model_dev(model, cfg, coords, t_model, dev, nx, ny, u_a, q_a)
                df = _fv_dev(spec, base_kwargs, p, fv_idx_ref, alpha=float(a))
            m = _delta_metrics(dm, df, base["dm"], base["df"], delta_eps)
            amp_track[b]["alpha"].append(float(a))
            amp_track[b]["dev_model"].append(m["dev_model_K"])
            amp_track[b]["dev_fv"].append(m["dev_fv_K"])
            rows.append({
                "test": "amplitude", "regime": "id_local",
                "sim_id": int(ids[b]), "partner_id": -1,
                "variant": f"alpha={a:g}",
                "temporal_family": str(p.get("temporal_family", "")),
                "spatial_family": str(p.get("spatial_family", "")),
                "alpha": float(a),
                "q_rms": float(base["sev"]["q_rms"] * a),
                "Q_in": float(base["sev"]["Q_in"] * a),
                **m,
            })

    # --- superposition (id_local) ---
    n_pairs = min(max_pairs, max(0, n - 1))
    for k in range(n_pairs):
        i, j = k, k + 1
        pi, pj = params[i], params[j]
        u_ij = cache[i]["u"] + cache[j]["u"]
        if cache[i]["q_all"] is not None:
            q_ij = cache[i]["q_all"] + cache[j]["q_all"]
        else:
            q_ij = None
        dm_joint = _model_dev(model, cfg, coords, t_model, dev, nx, ny, u_ij, q_ij)
        df_joint = _fv_dev(spec, base_kwargs, pi, fv_idx_ref, add_params=pj)
        dm_add = cache[i]["dm"] + cache[j]["dm"]
        df_add = cache[i]["df"] + cache[j]["df"]
        m = _delta_metrics(dm_joint, df_joint, dm_add, df_add, delta_eps)
        # combined severity of q_i + q_j on the dense grid.
        forcing_i = reconstruct_qL(
            pi["temporal_family"], pi["temporal_params"],
            pi["spatial_family"], pi["spatial_params"], t_ramp=cfg.t_ramp,
        )
        forcing_j = reconstruct_qL(
            pj["temporal_family"], pj["temporal_params"],
            pj["spatial_family"], pj["spatial_params"], t_ramp=cfg.t_ramp,
        )
        q_sum = (
            np.asarray(forcing_i.evaluate_grid(y_dense, t_dense), dtype=np.float64)
            + np.asarray(forcing_j.evaluate_grid(y_dense, t_dense), dtype=np.float64)
        )
        sev = severity_from_grid(q_sum, y_dense, t_dense)
        rows.append({
            "test": "superposition", "regime": "id_local",
            "sim_id": int(ids[i]), "partner_id": int(ids[j]),
            "variant": "q_i+q_j",
            "temporal_family": f"{pi.get('temporal_family','')}+{pj.get('temporal_family','')}",
            "spatial_family": f"{pi.get('spatial_family','')}+{pj.get('spatial_family','')}",
            "alpha": float("nan"),
            "q_rms": float(sev["q_rms"]), "Q_in": float(sev["Q_in"]),
            **m,
        })

    # --- OOD family swaps (ood) ---
    if do_ood and n >= 2:
        for b in range(n):
            p = params[b]
            for key, tag in (("temporal_family", "ood_swap_temporal"),
                             ("spatial_family", "ood_swap_spatial")):
                j = _find_partner(params, b, key)
                graft = dict(p)
                if key == "temporal_family":
                    graft["temporal_family"] = params[j]["temporal_family"]
                    graft["temporal_params"] = params[j]["temporal_params"]
                else:
                    graft["spatial_family"] = params[j]["spatial_family"]
                    graft["spatial_params"] = params[j]["spatial_params"]
                u_g = _build_image(graft, cfg, dev)
                q_g = _build_qleft(graft, cfg, y_grid, t_model, nx, dev)
                dm_g = _model_dev(model, cfg, coords, t_model, dev, nx, ny, u_g, q_g)
                df_g = _fv_dev(spec, base_kwargs, graft, fv_idx_ref)
                m = _delta_metrics(dm_g, df_g, cache[b]["dm"], cache[b]["df"], delta_eps)
                sev = forcing_severity(graft, y_dense, t_dense, cfg.t_ramp)
                rows.append({
                    "test": tag, "regime": "ood",
                    "sim_id": int(ids[b]), "partner_id": int(ids[j]),
                    "variant": f"{key}:{p.get(key,'')}->{params[j].get(key,'')}",
                    "temporal_family": str(graft.get("temporal_family", "")),
                    "spatial_family": str(graft.get("spatial_family", "")),
                    "alpha": float("nan"),
                    "q_rms": float(sev["q_rms"]), "Q_in": float(sev["Q_in"]),
                    **m,
                })

    out = Path(out_dir).expanduser() if out_dir else _checkpoint_out(checkpoint)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "linearity.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)

    summary = _summarize(rows, amp_track, params, ids, cfg, checkpoint, data_dir,
                         split, fv_dt, delta_eps)
    plots = _make_plots(amp_track, rows, params, ids, out)
    summary["plots"] = [str(p) for p in plots]
    summary["csv"] = str(csv_path)

    json_path = out / "linearity_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps({k: summary[k] for k in (
        "amplitude", "superposition", "ood",
    ) if k in summary}, indent=2))
    print(f"\nWrote: {csv_path}\n       {json_path}")
    for p in plots:
        print(f"       {p}")
    if cfg.warnings:
        print("\nConfig-recovery warnings:")
        for w in cfg.warnings:
            print(f"  ! {w}")
    return summary, out


def _checkpoint_out(checkpoint: str) -> Path:
    p = Path(checkpoint).expanduser()
    return (p if p.is_dir() else p.parent) / "forcing_diag"


def _linfit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Least-squares slope (through the points) and R^2 of ``y ~ slope*x + b``."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or np.ptp(x) < 1e-12:
        return float("nan"), float("nan")
    slope, b = np.polyfit(x, y, 1)
    pred = slope * x + b
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    return float(slope), r2


def _summarize(
    rows: list[dict[str, Any]],
    amp_track: dict[int, dict[str, list[float]]],
    params: list[dict[str, Any]],
    ids: np.ndarray,
    cfg,
    checkpoint: str,
    data_dir: str | None,
    split: str,
    fv_dt: float,
    delta_eps: float,
) -> dict[str, Any]:
    amp_rows = []
    slope_ratios, model_r2s = [], []
    for b, tr in amp_track.items():
        m_slope, m_r2 = _linfit(tr["alpha"], tr["dev_model"])
        f_slope, _ = _linfit(tr["alpha"], tr["dev_fv"])
        ratio = float(m_slope / f_slope) if (f_slope and np.isfinite(f_slope)
                                             and abs(f_slope) > 1e-12) else float("nan")
        amp_rows.append({
            "sim_id": int(ids[b]),
            "temporal_family": str(params[b].get("temporal_family", "")),
            "spatial_family": str(params[b].get("spatial_family", "")),
            "model_slope": m_slope, "fv_slope": f_slope,
            "slope_ratio": ratio, "model_fit_r2": m_r2,
        })
        if np.isfinite(ratio):
            slope_ratios.append(ratio)
        if np.isfinite(m_r2):
            model_r2s.append(m_r2)

    sup = [r for r in rows if r["test"] == "superposition"]
    sup_summary = {
        "n": len(sup),
        "mean_model_nonadditivity_K": float(np.mean([r["delta_model_K"] for r in sup])) if sup else None,
        "mean_rel_nonadditivity": float(np.mean([
            r["delta_model_K"] / r["dev_model_K"] for r in sup if r["dev_model_K"] > 1e-9
        ])) if sup else None,
        "mean_fv_nonadditivity_K": float(np.mean([r["delta_fv_K"] for r in sup])) if sup else None,
        "mean_model_rmse_K": float(np.mean([r["model_rmse_K"] for r in sup])) if sup else None,
    }

    def _ood_group(tag: str) -> dict[str, Any]:
        g = [r for r in rows if r["test"] == tag]
        valid = [r for r in g if r["delta_ratio_valid"]]
        return {
            "n": len(g),
            "mean_delta_ratio": float(np.mean([r["delta_ratio"] for r in valid])) if valid else None,
            "mean_cosine": float(np.mean([r["cosine"] for r in valid])) if valid else None,
            "mean_model_rmse_K": float(np.mean([r["model_rmse_K"] for r in g])) if g else None,
        }

    return {
        "checkpoint": str(checkpoint),
        "data_dir": str(data_dir) if data_dir else None,
        "split": split,
        "fv_dt": fv_dt,
        "delta_eps": delta_eps,
        "hard_left_flux": bool(cfg.hard_left_flux),
        "metric": "deviations are T - T_RIGHT (K); model_rmse_K vs fresh FV truth",
        "config_provenance": cfg.provenance,
        "config_warnings": list(cfg.warnings),
        "amplitude": {
            "note": ("FV deviation scales exactly with amplitude (slope_ratio~1); "
                     "a model slope_ratio<1 or model_fit_r2<1 flags a sublinear / "
                     "saturating forcing encoder."),
            "mean_slope_ratio": float(np.mean(slope_ratios)) if slope_ratios else None,
            "mean_model_fit_r2": float(np.mean(model_r2s)) if model_r2s else None,
            "per_sim": amp_rows,
        },
        "superposition": sup_summary,
        "ood": {
            "temporal_swap": _ood_group("ood_swap_temporal"),
            "spatial_swap": _ood_group("ood_swap_spatial"),
        },
    }


def _make_plots(
    amp_track: dict[int, dict[str, list[float]]],
    rows: list[dict[str, Any]],
    params: list[dict[str, Any]],
    ids: np.ndarray,
    out: Path,
) -> list[Path]:
    paths: list[Path] = []
    try:
        # amplitude: dev vs alpha, model (solid) vs FV (dashed) per sim.
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        cmap = plt.get_cmap("tab10")
        for idx, (b, tr) in enumerate(amp_track.items()):
            c = cmap(idx % 10)
            lab = f"sim {int(ids[b])} {params[b].get('temporal_family','')}"
            ax.plot(tr["alpha"], tr["dev_model"], "-o", color=c, label=lab)
            ax.plot(tr["alpha"], tr["dev_fv"], "--x", color=c, alpha=0.7)
        ax.set_xlabel("amplitude scale alpha")
        ax.set_ylabel("||T - T_RIGHT|| RMS (K)")
        ax.set_title("Amplitude linearity: model (solid) vs FV (dashed)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p1 = out / "amplitude_linearity.png"
        fig.savefig(p1, dpi=110)
        plt.close(fig)
        paths.append(p1)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] amplitude plot skipped: {exc}")

    try:
        sup = [r for r in rows if r["test"] == "superposition"]
        if sup:
            fig, ax = plt.subplots(figsize=(6.4, 4.0))
            labels = [f"{r['sim_id']}+{r['partner_id']}" for r in sup]
            rel = [r["delta_model_K"] / r["dev_model_K"] if r["dev_model_K"] > 1e-9 else 0.0
                   for r in sup]
            ax.bar(range(len(sup)), np.array(rel) * 100.0, color="slateblue")
            ax.set_xticks(range(len(sup)), labels, rotation=30, ha="right")
            ax.set_ylabel("model non-additivity / ||dev|| (%)")
            ax.set_title("Superposition error (FV is exactly additive)")
            ax.grid(True, axis="y", alpha=0.3)
            fig.tight_layout()
            p2 = out / "superposition.png"
            fig.savefig(p2, dpi=110)
            plt.close(fig)
            paths.append(p2)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] superposition plot skipped: {exc}")

    try:
        ood = [r for r in rows if r["regime"] == "ood" and r["delta_ratio_valid"]]
        if ood:
            fig, ax = plt.subplots(figsize=(7.0, 4.0))
            labels = [f"{r['sim_id']} {r['test'].split('_')[-1]}" for r in ood]
            xs = np.arange(len(ood))
            ax.bar(xs - 0.2, [r["delta_ratio"] for r in ood], width=0.4,
                   label="delta_ratio (model/FV response)", color="teal")
            ax.bar(xs + 0.2, [r["cosine"] for r in ood], width=0.4,
                   label="cosine(model, FV)", color="salmon")
            ax.axhline(1.0, color="k", lw=0.8, ls=":")
            ax.set_xticks(xs, labels, rotation=40, ha="right", fontsize=7)
            ax.set_title("OOD family swaps: response magnitude + direction")
            ax.legend(fontsize=8)
            ax.grid(True, axis="y", alpha=0.3)
            fig.tight_layout()
            p3 = out / "ood_swap.png"
            fig.savefig(p3, dpi=110)
            plt.close(fig)
            paths.append(p3)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] ood plot skipped: {exc}")
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="cvit_best.pt or seed dir.")
    parser.add_argument("--data-dir", default=None, help="Dataset dir; defaults to DATA_DIR.")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--num-sims", type=int, default=6)
    parser.add_argument("--sim-ids", default="", help="Comma-separated ids; overrides split.")
    parser.add_argument("--alphas", default="0.25,0.5,1.0,1.5,2.0",
                        help="Comma-separated amplitude scales (1.0 is the base).")
    parser.add_argument("--max-pairs", type=int, default=4,
                        help="Number of consecutive sim pairs for superposition.")
    parser.add_argument("--no-ood", action="store_true", help="Skip OOD family swaps.")
    parser.add_argument("--delta-eps", type=float, default=1e-2,
                        help="delta_fv floor (K) below which delta_ratio is flagged invalid.")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    alphas = tuple(float(a) for a in str(args.alphas).split(",") if a.strip())
    run_linearity(
        args.checkpoint,
        args.data_dir,
        split=args.split,
        num_sims=args.num_sims,
        sim_ids=args.sim_ids or None,
        alphas=alphas,
        max_pairs=args.max_pairs,
        do_ood=not args.no_ood,
        delta_eps=args.delta_eps,
        out_dir=args.out_dir,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
