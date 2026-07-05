"""Step 0.5 baseline diagnostic for the long-lead physics-only fine-tune.

Read-only over an existing checkpoint: it flips no config and mutates no
training state. It measures the *failure signature* the PINO plan targets
before any technique is enabled, so the causal-violation reading can be
confirmed against the alternatives (weak IC/early-time anchoring; snapshot
decoupling) and so Step 8 has a pre-fix reference.

Reported (both benchmarks, one large fixed-seed collocation draw):
  - unweighted interior CN residual per lead bin (the shared ``TemporalBinSpec``
    bins) — flat/low across bins while solution error grows is the causal
    fingerprint;
  - each BC region residual (left-Neumann, top/bottom adiabatic, right-Dirichlet)
    per lead bin;
  - IC anchor error (model at ``(t_s, t_s)`` vs ``T(t_s)``) and the first-CN-
    interval interior residual (bin 0) — a large value there points to
    anchoring/collapse rather than pure causal ordering;
  - solution error vs lead against the CN-FV reference trajectory (``rmse_K``,
    deviation from the 300 K equilibrium ``||T-300||_2``, and the
    perturbation-relative error ``||T_hat-T||_2 / ||T-300||_2`` which does not
    collapse toward 0 as the field relaxes).

Outputs a per-bin CSV, a solution-vs-lead CSV, a summary JSON, and a two-panel
plot under ``<seed_dir>/causal_diagnostic/``.
"""

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Region keys as emitted by full_bc_physics_loss(per_sample=True); each has a
# companion "<region>_per_sample" (B,) tensor.
REGION_KEYS = ("interior", "left_neumann", "topbot_adiabatic", "right_dirichlet")

# Physical equilibrium the diffusion field relaxes toward (Kelvin); the
# perturbation-relative error is measured against ||T - EQUILIBRIUM_K||.
EQUILIBRIUM_K = 300.0


def build_temporal_bin_spec(sampler, n_bins):
    """Build the shared ``TemporalBinSpec`` over the sampler's path-aware lead
    domain (the same domain Step 2's weighter uses):

      - on-grid (diffusion): ``[0.5*dt, t_final - 0.5*dt]`` — the CN interval-
        midpoint domain;
      - generic (forcing): ``[dt or collocation_lead_min, t_final]`` — the
        admissible continuous-lead range.
    """
    from src.operators.train import TemporalBinSpec

    base_plan = getattr(sampler, "base_plan", None)
    on_grid = bool(base_plan is not None and base_plan.get("on_grid_pairs", False))
    dt = float(sampler.dt)
    t_final = float(sampler.t_final)
    if on_grid:
        lead_lo = 0.5 * dt
        lead_hi = t_final - 0.5 * dt
    else:
        lead_min = getattr(sampler, "lead_min", None)
        lead_lo = dt if lead_min is None else max(dt, float(lead_min))
        lead_hi = t_final
    return TemporalBinSpec(lead_lo=lead_lo, lead_hi=lead_hi, n_bins=int(n_bins))


def _forward(model, batch, device):
    import torch  # noqa: F401 (kept local so the core stays import-light)

    spatial = batch["spatial"].to(device)
    cond = batch["cond_static"].to(device)
    fseq = batch["forcing_seq"].to(device) if "forcing_seq" in batch else None
    return model(spatial, cond, fseq)


def _build_geom_bc(sampler, R_c, qL_n, qL_np1, qL_int, device, dtype):
    """Mirror ``_collocation_batch_loss``'s geom/BC build (whole batch) so the
    residual the diagnostic measures is byte-identical to the training residual.
    """
    from src.physics.fv_residual import FullBCData, build_cn_geom_batched

    gc = sampler.geom_cfg
    geom = build_cn_geom_batched(
        gc["x_grid"], gc["y_grid"], gc["k_left"], gc["k_right"],
        gc["interface_x"], R_c.to(device), gc["dt"],
        sigma_global=gc["sigma_global"], device=device, dtype=dtype,
    )
    bc = FullBCData(
        T_right_tilde=gc["T_right_tilde"],
        qL_n=qL_n.to(device=device, dtype=dtype),
        qL_np1=qL_np1.to(device=device, dtype=dtype),
        qL_int=qL_int.to(device=device, dtype=dtype),
    )
    return geom, bc


def accumulate_bin_residuals(model, sampler, bin_spec, device, n_batches, max_lead):
    """Draw ``n_batches`` collocation batches and accumulate the per-region
    per-sample CN residual into the shared bins. Returns ``(means, counts)``
    where ``means[region]`` is ``(n_bins,)`` and ``counts`` is ``(n_bins,)``.

    Bins rows via ``sampler._last_lead_bin_ids`` (populated by the same
    ``TemporalBinSpec`` the weighter uses), so the diagnostic never re-derives
    bins from raw leads. Read-only: no grad, no weighter/EMA mutation.
    """
    import torch

    from src.operators.losses import full_bc_physics_loss

    n_bins = int(bin_spec.n_bins)
    sums = {k: np.zeros(n_bins, dtype=np.float64) for k in REGION_KEYS}
    counts = np.zeros(n_bins, dtype=np.int64)

    sampler.bin_spec = bin_spec  # attach so sample_batch records bin ids
    model.eval()
    with torch.no_grad():
        for _ in range(int(n_batches)):
            batch_t, batch_tdt, R_c, qL_n, qL_np1, qL_int = sampler.sample_batch(max_lead)
            yt = _forward(model, batch_t, device)
            ytdt = _forward(model, batch_tdt, device)
            geom, bc = _build_geom_bc(sampler, R_c, qL_n, qL_np1, qL_int, device, yt.dtype)
            out = full_bc_physics_loss(
                yt, ytdt, geom, bc, dirichlet_both_ends=True, per_sample=True
            )
            bin_ids = sampler._last_lead_bin_ids
            if bin_ids is None:
                raise RuntimeError(
                    "sampler did not record _last_lead_bin_ids; a TemporalBinSpec "
                    "must be attached before sampling (accumulate_bin_residuals "
                    "attaches it, so the sampler is likely a stub that ignores it)."
                )
            bin_ids = np.asarray(bin_ids, dtype=np.int64)
            for k in REGION_KEYS:
                vals = out[f"{k}_per_sample"].detach().cpu().numpy().astype(np.float64)
                np.add.at(sums[k], bin_ids, vals)
            np.add.at(counts, bin_ids, 1)

    denom = np.maximum(counts, 1)
    means = {k: sums[k] / denom for k in REGION_KEYS}
    return means, counts


def ic_anchor_error(model, sampler, device):
    """MSE of the model conditioned at ``(t_s, t_s)`` against the base state
    ``T(t_s)`` — the IC-anchoring error. Returns NaN if the sampler exposes no
    anchor batch (generic paths without a base plan)."""
    import torch

    if not hasattr(sampler, "sample_anchor_batch"):
        return float("nan")
    model.eval()
    with torch.no_grad():
        batch, target = sampler.sample_anchor_batch()
        y = _forward(model, batch, device)
        target = target.to(device=y.device, dtype=y.dtype)
        return float(torch.mean((y - target) ** 2).item())


def solution_error_by_lead(predict_fn, ds, sids, time_idxs, mu_global, sigma_global):
    """Single-shot solution error vs the CN-FV reference per on-grid lead.

    ``predict_fn(sid, t_hi) -> normalized field (Nx, Ny)``. For each selected
    time index the field is compared (in Kelvin) to ``ds.trajectories[sid, idx]``
    and averaged over ``sids``. Returns a list of row dicts:
    ``{lead, rmse_K, dev_from_eq_K, pert_rel_err, n_sims}``.
    """
    rows = []
    t_grid = np.asarray(ds.t_grid, dtype=np.float64)
    for idx in time_idxs:
        t_hi = float(t_grid[idx])
        rmse_list, dev_list, pert_list = [], [], []
        for sid in sids:
            yhat_norm = predict_fn(int(sid), t_hi)
            yhat_K = np.asarray(yhat_norm, dtype=np.float64) * sigma_global + mu_global
            ref_K = np.asarray(ds.trajectories[sid, idx], dtype=np.float64)
            diff = yhat_K - ref_K
            dev = float(np.linalg.norm(ref_K - EQUILIBRIUM_K))
            rmse_list.append(float(np.sqrt(np.mean(diff ** 2))))
            dev_list.append(dev)
            err = float(np.linalg.norm(diff))
            pert_list.append(err / dev if dev > 1e-12 else float("nan"))
        rows.append({
            "lead": t_hi,
            "rmse_K": float(np.mean(rmse_list)),
            "dev_from_eq_K": float(np.mean(dev_list)),
            "pert_rel_err": float(np.nanmean(pert_list)) if pert_list else float("nan"),
            "n_sims": len(sids),
        })
    return rows


def _slope(x, y):
    """Least-squares slope of ``y`` vs ``x`` over the finite pairs; NaN if < 2."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 2:
        return float("nan")
    return float(np.polyfit(x[m], y[m], 1)[0])


def write_bin_csv(path, bin_spec, means, counts):
    """Per-bin residual CSV. Columns: bin_id, lead_lo/hi/mid, count, and each
    region's mean per-sample CN residual."""
    edges = np.asarray(bin_spec.bin_edges, dtype=np.float64)
    fieldnames = (
        ["bin_id", "lead_lo", "lead_hi", "lead_mid", "count"]
        + [f"{k}_resid" for k in REGION_KEYS]
    )
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(int(bin_spec.n_bins)):
            row = {
                "bin_id": i,
                "lead_lo": float(edges[i]),
                "lead_hi": float(edges[i + 1]),
                "lead_mid": float(0.5 * (edges[i] + edges[i + 1])),
                "count": int(counts[i]),
            }
            for k in REGION_KEYS:
                row[f"{k}_resid"] = float(means[k][i])
            writer.writerow(row)
    return fieldnames


def write_solution_csv(path, rows):
    fieldnames = ["lead", "rmse_K", "dev_from_eq_K", "pert_rel_err", "n_sims"]
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return fieldnames


def write_plot(path, bin_spec, means, counts, solution_rows):
    """Two-panel figure: interior CN residual per bin (left) and solution error
    vs lead (right). Best-effort; a headless failure is non-fatal."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment-dependent
        print(f"warning: skipping plot ({exc})")
        return

    edges = np.asarray(bin_spec.bin_edges, dtype=np.float64)
    mids = 0.5 * (edges[:-1] + edges[1:])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax0 = axes[0]
    ax0.bar(mids, means["interior"], width=(edges[1] - edges[0]) * 0.9,
            color="C0", alpha=0.8)
    ax0.set_xlabel("lead (bin midpoint)")
    ax0.set_ylabel("interior CN residual (unweighted)")
    ax0.set_title("interior residual per lead bin")
    ax0.grid(True, alpha=0.3)
    for x, c in zip(mids, counts):
        ax0.annotate(str(int(c)), (x, 0), textcoords="offset points",
                     xytext=(0, 2), ha="center", fontsize=7, color="0.3")

    ax1 = axes[1]
    if solution_rows:
        leads = [r["lead"] for r in solution_rows]
        ax1.plot(leads, [r["rmse_K"] for r in solution_rows], "o-",
                 color="C3", label="rmse_K")
        ax1.set_xlabel("lead")
        ax1.set_ylabel("rmse (K)", color="C3")
        ax1.tick_params(axis="y", labelcolor="C3")
        ax1b = ax1.twinx()
        ax1b.plot(leads, [r["pert_rel_err"] for r in solution_rows], "s--",
                  color="C0", label="pert_rel_err")
        ax1b.set_ylabel("pert_rel_err", color="C0")
        ax1b.tick_params(axis="y", labelcolor="C0")
        ax1.set_title("solution error vs lead (vs CN-FV)")
        ax1.grid(True, alpha=0.3)
    else:
        ax1.set_axis_off()
        ax1.text(0.5, 0.5, "no solution-error rows", ha="center", va="center")

    fig.tight_layout()
    fig.savefig(str(path), dpi=120)
    plt.close(fig)


def _load_sibling_loaders():
    """Load the dataset/model/single-shot helpers from
    ``scripts/diffusion_continuity_check.py`` without a ``scripts`` package."""
    path = PROJECT_ROOT / "scripts" / "diffusion_continuity_check.py"
    spec = importlib.util.spec_from_file_location("_diffusion_continuity_check", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _selected_time_indices(t_grid, n_leads):
    """``n_leads`` on-grid target indices spread over ``[1, nt-1]`` (single-shot
    leads from the IC). Deduplicated and sorted."""
    nt = len(t_grid)
    if nt <= 1:
        return [0]
    want = np.unique(np.round(np.linspace(1, nt - 1, int(n_leads))).astype(int))
    return [int(i) for i in want if 0 < i < nt]


def run_seed(seed_dir, data_dir, *, n_bins, n_batches, num_sims, n_leads,
             batch_size, rng_seed):
    import torch

    from src.operators.train import _build_collocation_sampler
    from src.operators.utils import resolve_device

    sib = _load_sibling_loaders()

    ckpt = torch.load(seed_dir / "fno2d_best.pt", map_location="cpu", weights_only=False)
    config = ckpt["conf"]
    bench = config.get("benchmark", {}).get("name")
    mu_global = float(ckpt["mu_global"])
    sigma_global = float(ckpt["sigma_global"])
    device = resolve_device(config.get("training", {}).get("device", "auto"))

    ds, _val_ids = sib._build_dataset(config, mu_global, sigma_global, data_dir)
    model = sib._build_model(config, ckpt["model_state"], device)
    spec = ds.problem

    phys_cfg = config.get("training", {}).get("physics", {}) or {}
    bs = int(batch_size or phys_cfg.get("physics_batch_size")
             or config.get("training", {}).get("batch_size", 64))
    sampler = _build_collocation_sampler(
        config, phys_cfg, ds, spec, mu_global, sigma_global,
        batch_size=bs, rng_seed=int(rng_seed),
    )
    bin_spec = build_temporal_bin_spec(sampler, n_bins)
    max_lead = float(ds.t_final)

    means, counts = accumulate_bin_residuals(
        model, sampler, bin_spec, device, n_batches, max_lead
    )
    ic_mse = ic_anchor_error(model, sampler, device)

    t_grid = np.asarray(ds.t_grid, dtype=np.float64)
    time_idxs = _selected_time_indices(t_grid, n_leads)
    sids = [int(s) for s in np.asarray(ds.sim_ids)[: int(num_sims)]]

    def _predict(sid, t_hi):
        base_item = spec.build_item(ds, sid, 0, 0)
        ic_norm = np.asarray(base_item["spatial"][..., 0], dtype=np.float64)
        return sib._predict_single_shot(
            model, base_item, ds, spec, sid, ic_norm, t_hi, device
        )

    solution_rows = solution_error_by_lead(
        _predict, ds, sids, time_idxs, mu_global, sigma_global
    )

    out_dir = seed_dir / "causal_diagnostic"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_bin_csv(out_dir / "bin_residuals.csv", bin_spec, means, counts)
    write_solution_csv(out_dir / "solution_error.csv", solution_rows)
    write_plot(out_dir / "diagnostic.png", bin_spec, means, counts, solution_rows)

    mids = 0.5 * (np.asarray(bin_spec.bin_edges[:-1]) + np.asarray(bin_spec.bin_edges[1:]))
    populated = counts > 0
    summary = {
        "seed_dir": str(seed_dir),
        "benchmark": bench,
        "best_epoch": int(ckpt.get("epoch", -1)),
        "best_val": float(ckpt.get("best_val", float("nan"))),
        "n_bins": int(n_bins),
        "n_batches": int(n_batches),
        "batch_size": bs,
        "lead_domain": [float(bin_spec.lead_lo), float(bin_spec.lead_hi)],
        "ic_anchor_mse": ic_mse,
        "first_interval_interior_resid": float(means["interior"][0]),
        "interior_resid_by_bin": [float(v) for v in means["interior"]],
        "bin_counts": [int(c) for c in counts],
        "interior_resid_populated_mean": (
            float(means["interior"][populated].mean()) if populated.any() else float("nan")
        ),
        "interior_resid_slope_vs_lead": _slope(mids[populated], means["interior"][populated]),
        "bc_resid_by_bin": {
            k: [float(v) for v in means[k]] for k in REGION_KEYS if k != "interior"
        },
        "rmse_K_slope_vs_lead": _slope(
            [r["lead"] for r in solution_rows], [r["rmse_K"] for r in solution_rows]
        ),
        "pert_rel_err_slope_vs_lead": _slope(
            [r["lead"] for r in solution_rows], [r["pert_rel_err"] for r in solution_rows]
        ),
        "solution_error": solution_rows,
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"[{seed_dir.name}] bench={bench} ic_mse={ic_mse:.4g} "
        f"bin0_interior={means['interior'][0]:.4g} "
        f"interior_slope={summary['interior_resid_slope_vs_lead']:.4g} "
        f"rmse_K_slope={summary['rmse_K_slope_vs_lead']:.4g} -> {out_dir}"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_root",
        help="A config dir with seed*/ subdirs, or a single seed dir holding fno2d_best.pt.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Restrict to one seed index.")
    parser.add_argument("--data-dir", default=None,
                        help="Optional dataset dir override (else checkpoint-baked paths).")
    parser.add_argument("--n-bins", type=int, default=12, help="Lead bins (default 12).")
    parser.add_argument("--n-batches", type=int, default=32,
                        help="Collocation batches to accumulate (default 32).")
    parser.add_argument("--num-sims", type=int, default=4,
                        help="Validation sims for the solution-error curve (default 4).")
    parser.add_argument("--n-leads", type=int, default=10,
                        help="On-grid target times for the solution-error curve (default 10).")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Collocation batch size (default: physics_batch_size or training.batch_size).")
    parser.add_argument("--rng-seed", type=int, default=0, help="Sampler RNG seed (default 0).")
    args = parser.parse_args()

    run_root = Path(args.run_root).expanduser()
    if not run_root.is_dir():
        print(f"error: run_root does not exist or is not a directory: {run_root}", file=sys.stderr)
        return 1

    sib = _load_sibling_loaders()
    seed_dirs = sib._select_seed_dirs(run_root, args.seed)
    if not seed_dirs:
        print(f"error: no fno2d_best.pt checkpoints found under {run_root}", file=sys.stderr)
        return 1

    print(f"Found {len(seed_dirs)} seed dir(s): {[d.name for d in seed_dirs]}")
    for seed_dir in seed_dirs:
        run_seed(
            seed_dir, args.data_dir,
            n_bins=args.n_bins, n_batches=args.n_batches, num_sims=args.num_sims,
            n_leads=args.n_leads, batch_size=args.batch_size, rng_seed=args.rng_seed,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
