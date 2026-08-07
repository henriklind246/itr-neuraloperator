import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# eps (physical K) below which the FV one-step change is treated as
# near-stationary, so Q1/C1 are ill-conditioned and reported as NaN.
STATIONARY_EPS_K = 1e-6


def _select_seed_dirs(run_root: Path, seed: int | None) -> list[Path]:
    """Return seed*/ dirs holding fno2d_best.pt, or [run_root] if it is one."""
    if (run_root / "fno2d_best.pt").exists():
        return [run_root]
    if seed is not None:
        cand = run_root / f"seed{seed}"
        return [cand] if (cand / "fno2d_best.pt").exists() else []
    return [d for d in sorted(run_root.glob("seed*")) if (d / "fno2d_best.pt").exists()]


def _build_dataset(config: dict, mu_global, sigma_global, data_dir: str | None):
    from data.dataset import (
        SnapshotPairDataset,
        load_ramp_seconds,
        load_sim_data,
        load_solver_dt,
        problem_from_config,
        split_sim_ids,
    )

    if data_dir is not None:
        dd = Path(data_dir)
        config["data"]["trajectories.npy"] = str(dd / "trajectories.npy")
        config["data"]["x_grid_path"] = str(dd / "x_grid.npy")
        config["data"]["y_grid_path"] = str(dd / "y_grid.npy")
        config["data"]["t_grid_path"] = str(dd / "t_grid.npy")
        config["data"]["sim_params_path"] = str(dd / "sim_params.npy")

    trajectories, x_grid, y_grid, t_grid = load_sim_data(
        sim_traj_path=config["data"]["trajectories.npy"],
        x_grid_path=config["data"]["x_grid_path"],
        y_grid_path=config["data"]["y_grid_path"],
        t_grid_path=config["data"]["t_grid_path"],
    )
    sim_params = np.load(config["data"]["sim_params_path"], allow_pickle=True)
    solver_dt = load_solver_dt(config["data"]["t_grid_path"])
    ramp_seconds = load_ramp_seconds(config["data"]["t_grid_path"])

    _, val_ids, _ = split_sim_ids(trajectories.shape[0], train_frac=0.7, val_frac=0.15, seed=0)

    ds = SnapshotPairDataset(
        trajectories=trajectories,
        t_grid=t_grid,
        x_grid=x_grid,
        y_grid=y_grid,
        sim_ids=val_ids,
        sim_params=sim_params,
        mu_global=mu_global,
        sigma_global=sigma_global,
        n_snapshots=config["training"].get("n_snapshots", 15),
        noise_std=0.0,
        dt=solver_dt,
        ramp_seconds=ramp_seconds,
        problem=problem_from_config(config),
    )
    return ds, val_ids


def _build_model(config: dict, model_state, device):
    from data.dataset import problem_from_config
    from src.operators.fno2d import FNO2d

    model_cfg = config["model"]["parameters"]
    dims = problem_from_config(config).dims
    fno = FNO2d(
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
        forcing_cond_mode=model_cfg.get("forcing_cond_mode", "both"),
        s_y_channel=dims.s_y_channel,
        padding_reference_resolution=model_cfg.get("padding_reference_resolution"),
        padding_mode=model_cfg.get("padding_mode", "zeros"),
        cin_exclude_padding=model_cfg.get("cin_exclude_padding", False),
        hard_right_dirichlet=model_cfg.get("hard_right_dirichlet", False),
    )
    fno.load_state_dict(model_state)
    fno.to(device)
    fno.eval()
    return fno


def _selected_time_indices(t_grid: np.ndarray, t_final: float) -> list[int]:
    """Snap {0, dt, 2dt, 5dt, t_final/2, t_final} to unique grid indices."""
    grid_dt = float(t_grid[1] - t_grid[0])
    nt = len(t_grid)
    want = [0.0, grid_dt, 2.0 * grid_dt, 5.0 * grid_dt, 0.5 * t_final, t_final]
    idxs: list[int] = []
    for tv in want:
        idx = int(round(tv / grid_dt))
        idx = max(0, min(nt - 1, idx))
        if idx not in idxs:
            idxs.append(idx)
    return sorted(idxs)


def _predict_single_shot(model, base_item, ds, problem, sid, ic_norm, t_hi, device):
    import torch

    from src.operators.rollout import _forward_numpy_item, build_rollout_item_from_base

    item = build_rollout_item_from_base(base_item, ds, problem, int(sid), ic_norm, 0.0, float(t_hi))
    with torch.no_grad():
        y = _forward_numpy_item(model, item, device)
    return y.squeeze(0).squeeze(-1).detach().cpu().numpy().astype(np.float64)


def _run_seed(seed_dir: Path, data_dir: str | None, num_sims: int) -> None:
    import torch

    from src.operators.utils import resolve_device

    ckpt = torch.load(seed_dir / "fno2d_best.pt", map_location="cpu", weights_only=False)
    config = ckpt["conf"]
    if config.get("benchmark", {}).get("name") != "diffusion":
        print(
            f"warning: checkpoint benchmark is "
            f"{config.get('benchmark', {}).get('name')!r}, not 'diffusion'; "
            f"the continuity diagnostic assumes the diffusion relaxation problem."
        )
    mu_global = float(ckpt["mu_global"])
    sigma_global = float(ckpt["sigma_global"])
    device = resolve_device(config.get("training", {}).get("device", "auto"))

    ds, val_ids = _build_dataset(config, mu_global, sigma_global, data_dir)
    model = _build_model(config, ckpt["model_state"], device)
    problem = ds.problem

    t_grid = np.asarray(ds.t_grid, dtype=np.float64)
    t_final = float(ds.t_final)
    time_idxs = _selected_time_indices(t_grid, t_final)
    dt_idx = 1 if len(t_grid) > 1 else 0  # one grid step == Δt for J1/Q1/C1

    sids = [int(s) for s in val_ids[: int(num_sims)]]
    out_dir = seed_dir / "continuity"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_sim_rows: list[dict] = []
    q1_valid: list[float] = []
    c1_valid: list[float] = []

    for sid in sids:
        base_item = problem.build_item(ds, sid, 0, 0)
        ic_norm = np.asarray(base_item["spatial"][..., 0], dtype=np.float64)
        t0_phys = ic_norm * sigma_global + mu_global  # == FV(0)

        times = []
        model_fields = []
        fv_fields = []
        for idx in time_idxs:
            yhat_norm = _predict_single_shot(
                model, base_item, ds, problem, sid, ic_norm, float(t_grid[idx]), device
            )
            model_fields.append(yhat_norm * sigma_global + mu_global)
            fv_fields.append(np.asarray(ds.trajectories[sid, idx], dtype=np.float64))
            times.append(float(t_grid[idx]))

        # One-step (Δt) jump magnitudes / direction (physical K).
        pred_dt = _predict_single_shot(
            model, base_item, ds, problem, sid, ic_norm, float(t_grid[dt_idx]), device
        ) * sigma_global + mu_global
        fv_dt = np.asarray(ds.trajectories[sid, dt_idx], dtype=np.float64)

        pred_step_vec = (pred_dt - t0_phys).ravel()
        fv_step_vec = (fv_dt - t0_phys).ravel()
        j1 = float(np.linalg.norm(pred_step_vec))
        j1_fv = float(np.linalg.norm(fv_step_vec))
        if j1_fv > STATIONARY_EPS_K:
            q1 = j1 / j1_fv
            if j1 > STATIONARY_EPS_K:
                c1 = float(np.dot(pred_step_vec, fv_step_vec) / (j1 * j1_fv))
            else:
                c1 = float("nan")
        else:
            q1 = float("nan")
            c1 = float("nan")
        if not np.isnan(q1):
            q1_valid.append(q1)
        if not np.isnan(c1):
            c1_valid.append(c1)

        _save_field_rows(out_dir, sid, times, ic_norm, model_fields, fv_fields, mu_global)
        _save_curves(out_dir, sid, times, model_fields, fv_fields)

        per_sim_rows.append(
            {
                "sim_id": sid,
                "ic_family": str(ds.sim_params[sid].get("ic_family", "")),
                "J1": j1,
                "J1_FV": j1_fv,
                "Q1": q1,
                "C1": c1,
            }
        )
        print(
            f"[seed={seed_dir.name} sid={sid}] "
            f"J1={j1:.4g} J1_FV={j1_fv:.4g} "
            f"Q1={q1:.4g} C1={c1:.4g}"
        )

    summary = {
        "seed_dir": str(seed_dir),
        "best_epoch": int(ckpt.get("epoch", -1)),
        "best_val": float(ckpt.get("best_val", float("nan"))),
        "num_sims_inspected": len(sids),
        "num_valid_Q1": len(q1_valid),
        "num_valid_C1": len(c1_valid),
        "mean_Q1": float(np.mean(q1_valid)) if q1_valid else float("nan"),
        "mean_C1": float(np.mean(c1_valid)) if c1_valid else float("nan"),
        "selected_time_indices": time_idxs,
        "selected_times": [float(t_grid[i]) for i in time_idxs],
        "per_sim": per_sim_rows,
    }
    with (out_dir / "continuity_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    with (out_dir / "continuity_per_sim.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sim_id", "ic_family", "J1", "J1_FV", "Q1", "C1"])
        writer.writeheader()
        writer.writerows(per_sim_rows)

    print(
        f"[seed={seed_dir.name}] valid Q1={len(q1_valid)}/{len(sids)} "
        f"mean Q1={summary['mean_Q1']:.4g} mean C1={summary['mean_C1']:.4g} "
        f"-> {out_dir}"
    )


def _save_field_rows(out_dir, sid, times, ic_norm, model_fields, fv_fields, mu_global):
    import matplotlib.pyplot as plt

    from visual._common import PLOT_STYLE, _save_figure

    plt.rcParams.update(PLOT_STYLE)
    n = len(times)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n), squeeze=False)
    for r, (t, mf, ff) in enumerate(zip(times, model_fields, fv_fields)):
        resid = mf - ff
        vmin = min(mf.min(), ff.min())
        vmax = max(mf.max(), ff.max())
        rlim = float(np.max(np.abs(resid))) if np.any(resid) else 1.0
        im0 = axes[r][0].imshow(mf.T, origin="lower", vmin=vmin, vmax=vmax, cmap="magma")
        im1 = axes[r][1].imshow(ff.T, origin="lower", vmin=vmin, vmax=vmax, cmap="magma")
        im2 = axes[r][2].imshow(resid.T, origin="lower", vmin=-rlim, vmax=rlim, cmap="coolwarm")
        axes[r][0].set_ylabel(f"t={t:.4g}")
        fig.colorbar(im0, ax=axes[r][0], fraction=0.046)
        fig.colorbar(im1, ax=axes[r][1], fraction=0.046)
        fig.colorbar(im2, ax=axes[r][2], fraction=0.046)
    axes[0][0].set_title("model $\\hat{T}(t)$")
    axes[0][1].set_title("FV $T(t)$")
    axes[0][2].set_title("residual")
    fig.suptitle(f"sim {sid}: single-shot prediction vs FV from the IC")
    _save_figure(fig, out_dir / f"fields_sim{sid}.png", "physics", f"fields_sim{sid}", layout="tight")


def _save_curves(out_dir, sid, times, model_fields, fv_fields):
    import matplotlib.pyplot as plt

    from visual._common import PLOT_STYLE, _save_figure

    plt.rcParams.update(PLOT_STYLE)
    t = np.asarray(times, dtype=np.float64)

    def _stats(fields):
        mean = np.array([f.mean() for f in fields])
        std = np.array([f.std() for f in fields])
        signed = np.array([float(np.sum(f - 300.0)) for f in fields])
        unsigned = np.array([float(np.linalg.norm(f - 300.0)) for f in fields])
        return mean, std, signed, unsigned

    m_mean, m_std, m_signed, m_unsigned = _stats(model_fields)
    f_mean, f_std, f_signed, f_unsigned = _stats(fv_fields)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    panels = [
        (axes[0][0], "spatial mean (K)", m_mean, f_mean),
        (axes[0][1], "spatial std (K)", m_std, f_std),
        (axes[1][0], "signed energy $\\Sigma(T-300)$", m_signed, f_signed),
        (axes[1][1], "unsigned $\\|T-300\\|_2$", m_unsigned, f_unsigned),
    ]
    for ax, title, mvals, fvals in panels:
        ax.plot(t, mvals, "o-", label="model", color="C3")
        ax.plot(t, fvals, "s--", label="FV", color="C0")
        ax.set_title(title)
        ax.set_xlabel("t")
        ax.grid(True)
        ax.legend()
    fig.suptitle(f"sim {sid}: continuity of the single-shot transient")
    _save_figure(fig, out_dir / f"curves_sim{sid}.png", "physics", f"curves_sim{sid}", layout="tight")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Diffusion physics-only continuity diagnostic: verify the model "
            "produces a genuine diffusive transient from the IC rather than an "
            "instantaneous collapse to uniform 300 K."
        ),
    )
    parser.add_argument(
        "run_root",
        help="A config dir with seed*/ subdirs, or a single seed dir holding fno2d_best.pt.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Restrict to one seed index.")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Optional dataset dir override (else use the checkpoint-baked paths).",
    )
    parser.add_argument(
        "--num-sims",
        type=int,
        default=4,
        help="Number of validation sims to inspect (default 4).",
    )
    args = parser.parse_args()

    run_root = Path(args.run_root).expanduser()
    if not run_root.is_dir():
        print(f"error: run_root does not exist or is not a directory: {run_root}", file=sys.stderr)
        return 1

    seed_dirs = _select_seed_dirs(run_root, args.seed)
    if not seed_dirs:
        print(f"error: no fno2d_best.pt checkpoints found under {run_root}", file=sys.stderr)
        return 1

    print(f"Found {len(seed_dirs)} seed dir(s): {[d.name for d in seed_dirs]}")
    for seed_dir in seed_dirs:
        _run_seed(seed_dir, args.data_dir, args.num_sims)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
