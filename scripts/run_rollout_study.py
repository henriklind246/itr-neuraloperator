"""Score a checkpoint with direct prediction and homogeneous autoregressive rollout.

Generates a fresh held-out dataset, runs the production ``write_test_records``
path once per arm (direct, then one per requested substep count), and reduces the
per-pair records to a simulation-clustered paired comparison.

All uncertainty is over simulations, never over pairs: pairs drawn from the same
simulation share latents and a trajectory, so a pair-level SEM is
pseudo-replication and is optimistic by a large factor.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Benchmarks with a branch in rollout.build_rollout_item_from_base. Anything else
# would raise mid-study, after dataset generation has already been paid for.
ROLLOUT_BENCHMARKS = ("forcing", "interfaces", "source", "source_itr", "diffusion")


def lead_of(row):
    """Physical lead of a pair. eval.py stores t_j - t_s directly in `t_bar`."""
    return float(row["t_bar"])


# ---------- provenance helpers ----------

def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, OSError):
        return None


# ---------- checkpoint resolution ----------

def resolve_checkpoint(target, seed):
    """Return (checkpoint_path, ckpt, seed_name) for a run_root or a bare .pt."""
    import torch

    from src.operators.eval import _select_seed_checkpoint

    target = Path(target).expanduser()
    if target.is_dir():
        seed_dir, ckpt = _select_seed_checkpoint(target, seed)
        return seed_dir / "fno2d_best.pt", ckpt, seed_dir.name
    if target.is_file():
        if seed is not None:
            raise SystemExit("--seed applies to a run_root directory, not a bare .pt file")
        return target, torch.load(target, map_location="cpu", weights_only=True), "seed0"
    raise SystemExit(f"checkpoint not found: {target}")


def materialize_checkpoint(src, dst, allow_symlink):
    """Place the checkpoint under the study dir and return how it was placed.

    Every arm then runs with ``run_root=<out_dir>/ckpt``, so the record CSVs and
    provenance sidecars that ``write_test_records`` writes beside the checkpoint
    land inside the study directory with no output-path changes in eval.py.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        pass
    if allow_symlink:
        os.symlink(os.path.abspath(src), dst)
        return "symlink"
    shutil.copy2(src, dst)
    return "copy"


# ---------- dataset generation ----------

def training_rng_seed(config):
    """Return (seed, source) for the draw family the checkpoint trained on."""
    traj = config.get("data", {}).get("trajectories.npy")
    if not traj:
        return None, "unavailable"
    data_dir = Path(traj).parent
    if not data_dir.is_dir():
        return None, "unavailable"
    meta_path = data_dir / "meta.npy"
    if not meta_path.exists():
        # generate_sim_data writes meta.npy whenever rng_seed != 0, so an absent
        # sidecar beside a readable training dataset pins the family at 0.
        return 0, "inferred_from_absent_meta"
    meta = np.load(meta_path, allow_pickle=True).item()
    seed = meta.get("rng_seed")
    if seed is None:
        return None, "unavailable"
    return int(seed), "meta.npy"


def check_disjoint(study_seed, train_seed, train_seed_source):
    if train_seed is not None:
        if study_seed == train_seed:
            raise SystemExit(
                f"--rng-seed {study_seed} matches the checkpoint's training draw "
                f"family (read from {train_seed_source}). The study set would not "
                f"be held out. Pick a different --rng-seed."
            )
        return
    if study_seed == 0:
        raise SystemExit(
            "--rng-seed 0 rejected because production training datasets to date "
            "were generated with seed 0; the training seed could not be read from "
            "the checkpoint, so disjointness cannot be verified. Pass an explicit "
            "non-zero --rng-seed, or --data-dir to supply your own held-out set."
        )


def generate_dataset(benchmark, save_dir, num_sims, nx, ny, rng_seed):
    from data.generate_dataset import generate_sim_data

    generate_sim_data(
        num_sims=num_sims,
        save_dir=save_dir,
        benchmark=benchmark,
        nx=nx,
        ny=ny,
        rng_seed=rng_seed,
    )
    return save_dir


# ---------- records ----------

def read_records(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def pair_key(row):
    return (int(row["sim_id"]), int(row["s"]), int(row["j"]))


def write_pair_keys(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sim_id", "s", "j", "t_s", "lead"])
        for r in rows:
            w.writerow([r["sim_id"], r["s"], r["j"], float(r["t_s"]), lead_of(r)])


def assert_same_pairs(arm, reference, rows):
    keys = [pair_key(r) for r in rows]
    if keys != reference:
        raise SystemExit(
            f"arm {arm!r} enumerated a different pair set than the direct arm "
            f"({len(keys)} vs {len(reference)} pairs). The paired comparison "
            f"would be invalid; aborting."
        )


# ---------- aggregation ----------

def lead_axis(rows, dt_grid):
    """Return (lead -> step-count key, ordered exact lead values).

    Snapshot times are grid points, so every lead is an integer number of saved
    grid steps. Keying on that integer collapses the float32 round-trip noise
    that otherwise splits one physical lead into several near-duplicates, without
    imposing an arbitrary rounding tolerance. Distinct leads are grouped by their
    exact shared value, never binned.
    """
    steps = defaultdict(list)
    for r in rows:
        steps[int(round(lead_of(r) / dt_grid))].append(lead_of(r))
    ordered = sorted(steps)
    index = {k: i for i, k in enumerate(ordered)}
    leads = [float(np.mean(steps[k])) for k in ordered]
    return index, leads


def lead_key(row, dt_grid):
    return int(round(lead_of(row) / dt_grid))


def pool_cells(rows, lead_index, dt_grid):
    """Pool sufficient statistics per (sim_id, lead) for one arm.

    sse_K2 and target_sse_K2 are both computed on normalized truths and scaled by
    the same sigma^2, so sigma^2 cancels in the ratio and E stays in the same
    normalized-percentage convention as the per-row rel_l2_pct.
    """
    cells = defaultdict(lambda: {"sse": 0.0, "target": 0.0, "cells": 0.0, "n": 0})
    for r in rows:
        c = cells[(int(r["sim_id"]), lead_index[lead_key(r, dt_grid)])]
        c["sse"] += float(r["sse_K2"])
        c["target"] += float(r["target_sse_K2"])
        c["cells"] += float(r["num_error_cells"])
        c["n"] += 1
    for c in cells.values():
        c["E"] = 100.0 * math.sqrt(c["sse"] / c["target"])
        c["rmse_K"] = math.sqrt(c["sse"] / c["cells"])
    return dict(cells)


def mean_sem(values):
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return float("nan"), float("nan")
    if v.size == 1:
        return float(v[0]), float("nan")
    return float(v.mean()), float(v.std(ddof=1) / math.sqrt(v.size))


def build_summary(arms, cells_by_arm, leads, sim_ids, time_norm_horizon):
    """Long-format summary rows. Every SEM and win rate is over simulations."""
    out = []
    direct = cells_by_arm["direct"]
    for arm, substeps in arms:
        cells = cells_by_arm[arm]
        for li, lead_phys in enumerate(leads):
            present = [i for i in sim_ids if (i, li) in cells]
            if not present:
                continue
            E = [cells[(i, li)]["E"] for i in present]
            n_pairs = sum(cells[(i, li)]["n"] for i in present)
            lead_norm = (lead_phys / time_norm_horizon
                         if time_norm_horizon else float("nan"))
            base = {
                "arm": arm, "substeps": substeps, "lead_index": li,
                "lead_physical": lead_phys, "lead_normalized": lead_norm,
                "n_sims": len(present), "n_pairs": n_pairs,
            }
            m, s = mean_sem(E)
            out.append({**base, "metric": "rel_l2_pct", "aggregation": "simulation_mean",
                        "estimate": m, "sem": s, "median": float(np.median(E)),
                        "win_rate": ""})

            sse = sum(cells[(i, li)]["sse"] for i in present)
            target = sum(cells[(i, li)]["target"] for i in present)
            out.append({**base, "metric": "rel_l2_pct", "aggregation": "globally_pooled",
                        "estimate": 100.0 * math.sqrt(sse / target),
                        # An ordinary SEM over sims does not apply to a pooled
                        # ratio; populate only with a cluster bootstrap.
                        "sem": "", "median": "", "win_rate": ""})

            cell_count = sum(cells[(i, li)]["cells"] for i in present)
            rm, rs = mean_sem([cells[(i, li)]["rmse_K"] for i in present])
            out.append({**base, "metric": "rmse_K", "aggregation": "simulation_mean",
                        "estimate": rm, "sem": rs, "win_rate": "",
                        "median": float(np.median([cells[(i, li)]["rmse_K"] for i in present]))})
            out.append({**base, "metric": "rmse_K", "aggregation": "globally_pooled",
                        "estimate": math.sqrt(sse / cell_count),
                        "sem": "", "median": "", "win_rate": ""})

            if arm == "direct":
                continue
            paired = [i for i in present if (i, li) in direct]
            deltas = [cells[(i, li)]["E"] - direct[(i, li)]["E"] for i in paired]
            dm, ds = mean_sem(deltas)
            # Strict inequality: ties count as losses. Continuous errors make ties
            # essentially impossible, but the convention is fixed here so it can
            # never be read two ways later.
            wins = sum(1 for d in deltas if d < 0.0)
            out.append({**base, "metric": "rel_l2_pct",
                        "aggregation": "paired_simulation_delta",
                        "estimate": dm, "sem": ds, "median": float(np.median(deltas)),
                        "win_rate": wins / len(deltas), "n_sims": len(paired)})
    return out


def write_paired_table(path, arms, cells_by_arm, leads, sim_ids):
    direct = cells_by_arm["direct"]
    fields = [
        "sim_id", "arm", "substeps", "lead_index", "lead_physical",
        "n_pairs", "E_rel_l2_pct", "rmse_K", "delta_vs_direct",
        "sum_sse_K2", "sum_target_sse_K2", "sum_num_error_cells",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for arm, substeps in arms:
            cells = cells_by_arm[arm]
            for li, lead_phys in enumerate(leads):
                for sid in sim_ids:
                    c = cells.get((sid, li))
                    if c is None:
                        continue
                    d = direct.get((sid, li))
                    w.writerow({
                        "sim_id": sid, "arm": arm, "substeps": substeps,
                        "lead_index": li, "lead_physical": lead_phys,
                        "n_pairs": c["n"], "E_rel_l2_pct": c["E"],
                        "rmse_K": c["rmse_K"],
                        "delta_vs_direct": "" if (d is None or arm == "direct")
                                           else c["E"] - d["E"],
                        "sum_sse_K2": c["sse"], "sum_target_sse_K2": c["target"],
                        "sum_num_error_cells": c["cells"],
                    })


# ---------- diagnostics ----------

def minimum_lead(t_grid, n_snapshots):
    """Smallest gap between consecutive snapshots under the dataset's selection."""
    Nt = len(t_grid)
    if n_snapshots is None or n_snapshots >= Nt:
        idx = np.arange(Nt)
    else:
        idx = np.round(np.linspace(0, Nt - 1, n_snapshots)).astype(int)
    times = np.asarray(t_grid, dtype=np.float64)[idx]
    return float(np.min(np.diff(times)))


def subinterval_diagnostics(leads, substeps, train_min_lead):
    """Per-K subinterval durations vs the shortest lead the model trained on.

    Homogeneous subdivision at large K produces subintervals far shorter than any
    training pair. That is a plausible mechanism for large-K degradation, so it is
    recorded as a measurement rather than enforced as a constraint.
    """
    out = {}
    for K in substeps:
        durations = [lead / K for lead in leads]
        below = (
            None if train_min_lead is None
            else sum(1 for d in durations if d < train_min_lead) / len(durations)
        )
        out[str(K)] = {
            "minimum_rollout_subinterval": min(durations),
            "fraction_of_subintervals_below_training_minimum": below,
        }
    return out


# ---------- figure ----------

# Okabe-Ito, the same palette as visual/pub/style.BENCHMARK_COLORS. Arms are
# ordinal in substep count, so `direct` takes the first slot and each rollout arm
# follows in order.
ARM_COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00")


def write_figure(path, arms, summary, leads):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from visual.pub import style

    def series(arm, aggregation):
        rows = [
            r for r in summary
            if r["arm"] == arm and r["metric"] == "rel_l2_pct"
            and r["aggregation"] == aggregation
        ]
        rows.sort(key=lambda r: r["lead_index"])
        x = np.array([r["lead_physical"] for r in rows])
        y = np.array([r["estimate"] for r in rows], dtype=float)
        e = np.array([r["sem"] if r["sem"] != "" else np.nan for r in rows], dtype=float)
        return x, y, e

    # One colour per arm across both panels; the delta panel drops `direct`, so
    # letting each axis run its own cycle would recolour every rollout arm.
    colors = {arm: ARM_COLORS[i % len(ARM_COLORS)] for i, (arm, _) in enumerate(arms)}

    with style.pub_style():
        fig, (ax0, ax1) = plt.subplots(
            1, 2, figsize=style.figsize("two_col", row_height="std")
        )

        for arm, _ in arms:
            x, y, e = series(arm, "simulation_mean")
            colour = colors[arm]
            # A symmetric SEM band can reach zero or below, which a log axis
            # cannot draw; floor it instead of losing the ribbon.
            lower = np.maximum(y - e, 0.05 * np.nanmin(y))
            ax0.fill_between(x, lower, y + e, color=colour, alpha=0.10, linewidth=0)
            ax0.plot(x, y, "-o", color=colour, linewidth=1.0, markersize=3,
                     label=arm)
        ax0.set_xlabel(r"lead time $\bar{t}$")
        ax0.set_ylabel(style.axis_label("rel_l2_pct"))
        ax0.set_yscale("log")
        # Errors usually span well under a decade, so the default log axis labels
        # a single power of ten. Label the 2/3/5 subdivisions as plain numbers.
        ax0.yaxis.set_minor_locator(
            matplotlib.ticker.LogLocator(base=10.0, subs=(2.0, 3.0, 5.0), numticks=12)
        )
        for setter in (ax0.yaxis.set_major_formatter, ax0.yaxis.set_minor_formatter):
            setter(matplotlib.ticker.ScalarFormatter())
        ax0.tick_params(axis="y", which="minor", labelsize=6)
        ax0.set_title("Simulation-mean error", fontsize=7)
        ax0.grid(True, axis="y", alpha=0.25)
        ax0.legend(loc="best", fontsize=6.5)

        style.add_reference_line(ax1, value=0.0, label="")
        for arm, _ in arms:
            if arm == "direct":
                continue
            x, y, e = series(arm, "paired_simulation_delta")
            colour = colors[arm]
            ax1.fill_between(x, y - e, y + e, color=colour, alpha=0.10, linewidth=0)
            ax1.plot(x, y, "-o", color=colour, linewidth=1.0, markersize=3,
                     label=arm)
        ax1.set_xlabel(r"lead time $\bar{t}$")
        ax1.set_ylabel(r"$\Delta$ rel. $L_2$ vs direct [%]")
        ax1.set_title("Paired delta (negative = rollout better)", fontsize=7)
        ax1.grid(True, axis="y", alpha=0.25)
        ax1.legend(loc="best", fontsize=6.5)

        style.panel_letters([ax0, ax1])
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)


# ---------- printing ----------

def print_summary(summary, arms, leads):
    print("\nSimulation-mean rel-L2 (%) +- SEM over simulations")
    header = "  lead     " + "".join(f"{arm:>20}" for arm, _ in arms)
    print(header)
    for li, lead in enumerate(leads):
        cells = []
        for arm, _ in arms:
            row = next(
                (r for r in summary if r["arm"] == arm and r["lead_index"] == li
                 and r["metric"] == "rel_l2_pct"
                 and r["aggregation"] == "simulation_mean"), None
            )
            cells.append("".rjust(20) if row is None
                         else f"{row['estimate']:>13.4f} +-{row['sem']:>5.4f}")
        print(f"  {lead:<8.4f}" + "".join(cells))

    print("\nPaired delta vs direct (%) +- SEM, and simulation win rate")
    for arm, _ in arms:
        if arm == "direct":
            continue
        print(f"  {arm}")
        for li, lead in enumerate(leads):
            row = next(
                (r for r in summary if r["arm"] == arm and r["lead_index"] == li
                 and r["aggregation"] == "paired_simulation_delta"), None
            )
            if row is None:
                continue
            print(f"    lead {lead:<8.4f} delta {row['estimate']:>8.4f} "
                  f"+-{row['sem']:>7.4f}  win {row['win_rate']:.3f} "
                  f"(n_sims={row['n_sims']}, n_pairs={row['n_pairs']})")

    print("\nBest arm per lead (simulation mean)")
    for li, lead in enumerate(leads):
        rows = [r for r in summary if r["lead_index"] == li
                and r["metric"] == "rel_l2_pct"
                and r["aggregation"] == "simulation_mean"]
        if rows:
            best = min(rows, key=lambda r: r["estimate"])
            print(f"  lead {lead:<8.4f} -> {best['arm']} ({best['estimate']:.4f}%)")


# ---------- main ----------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Compare direct prediction against homogeneous autoregressive "
                    "rollout for a trained checkpoint.",
    )
    p.add_argument("checkpoint", help="A run_root containing seed*/ or a bare .pt file.")
    p.add_argument("--seed", default=None,
                   help="Seed to evaluate when checkpoint is a run_root (e.g. 42).")
    p.add_argument("--out-dir", default=None,
                   help="Study directory. Default rollout_studies/<benchmark>_<timestamp>/.")
    p.add_argument("--num-sims", type=int, default=24,
                   help="Simulations to generate. These are the statistical clusters.")
    p.add_argument("--n-snapshots-test", type=int, default=6,
                   help="Snapshots per simulation; pairs per sim are S*(S-1)/2.")
    p.add_argument("--substeps", default="2,4,8",
                   help="Comma-separated autoregressive substep counts.")
    p.add_argument("--data-dir", default=None,
                   help="Use an existing dataset directory instead of generating one.")
    p.add_argument("--rng-seed", type=int, default=7,
                   help="Draw-family selector for the generated set; must differ "
                        "from the checkpoint's training family.")
    p.add_argument("--device", default=None, help="cpu, mps, cuda, or auto.")
    p.add_argument("--nx", type=int, default=100)
    p.add_argument("--ny", type=int, default=100)
    p.add_argument("--symlink-checkpoint", action="store_true",
                   help="Permit symlinking the checkpoint when hard-linking fails.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    substeps = [int(k) for k in args.substeps.split(",") if k.strip()]
    if any(k < 2 for k in substeps):
        raise SystemExit("--substeps values must be >= 2; the direct arm is not a "
                         "one-step rollout and is always scored separately.")

    ckpt_path, ckpt, seed_name = resolve_checkpoint(args.checkpoint, args.seed)
    config = ckpt["conf"]
    benchmark = config.get("benchmark", {}).get("name", "forcing")
    representation = config.get("benchmark", {}).get("representation", "temporal_encoder")
    if benchmark not in ROLLOUT_BENCHMARKS:
        raise SystemExit(
            f"benchmark {benchmark!r} has no rollout branch; supported: "
            f"{', '.join(ROLLOUT_BENCHMARKS)}."
        )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else (
        PROJECT_ROOT / "rollout_studies" / f"{benchmark}_{stamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    run_root = out_dir / "ckpt"
    seed_dir = run_root / seed_name
    placement = materialize_checkpoint(
        ckpt_path, seed_dir / "fno2d_best.pt", args.symlink_checkpoint
    )
    ckpt_sha = _sha256_file(seed_dir / "fno2d_best.pt")

    train_seed, train_seed_source = training_rng_seed(config)
    if args.data_dir is not None:
        data_dir = Path(args.data_dir).expanduser()
        dataset_rng_seed = None
    else:
        check_disjoint(args.rng_seed, train_seed, train_seed_source)
        data_dir = out_dir / "data"
        dataset_rng_seed = args.rng_seed

    n_snap = args.n_snapshots_test
    pairs_per_sim = n_snap * (n_snap - 1) // 2
    n_pairs = args.num_sims * pairs_per_sim
    passes = n_pairs * (1 + sum(substeps))
    print(f"benchmark        : {benchmark} / {representation}")
    print(f"checkpoint       : {ckpt_path} ({placement}, sha256 {ckpt_sha[:12]})")
    print(f"study dir        : {out_dir}")
    print(f"arms             : direct, " + ", ".join(f"k{k}" for k in substeps))
    print(f"pairs            : {args.num_sims} sims x {pairs_per_sim} = {n_pairs}")
    print(f"forward passes   : {passes}  (~{passes * 0.091 / 60:.1f} min at 0.091 s/pass)")
    print(flush=True)

    if dataset_rng_seed is not None:
        generate_dataset(benchmark, data_dir, args.num_sims, args.nx, args.ny,
                         dataset_rng_seed)

    from src.operators.eval import write_test_records

    arms = [("direct", "")] + [(f"k{k}", k) for k in substeps]
    records = {}
    reference_keys = None
    t0 = time.time()
    for arm, substep in arms:
        print(f"[{time.time() - t0:7.1f}s] scoring arm {arm}", flush=True)
        out_csv = write_test_records(
            str(run_root),
            seed=seed_name.removeprefix("seed"),
            out_name=f"records_{arm}.csv",
            data_dir=str(data_dir),
            rollout_enabled=(arm != "direct"),
            rollout_num_substeps=(None if arm == "direct" else substep),
            eval_all_sims=True,
            n_snapshots_test=n_snap,
            device=args.device,
            inference_batch_size=1,
        )
        rows = read_records(out_csv)
        if reference_keys is None:
            reference_keys = [pair_key(r) for r in rows]
            write_pair_keys(out_dir / "pair_keys.csv", rows)
        else:
            assert_same_pairs(arm, reference_keys, rows)
        records[arm] = rows

    t_grid = np.load(data_dir / "t_grid.npy")
    dt_grid = float(np.min(np.diff(np.asarray(t_grid, dtype=np.float64))))
    lead_index, leads = lead_axis(records["direct"], dt_grid)
    sim_ids = sorted({int(r["sim_id"]) for r in records["direct"]})
    cells_by_arm = {
        arm: pool_cells(rows, lead_index, dt_grid) for arm, rows in records.items()
    }

    first = records["direct"][0]
    horizon = float(first["time_norm_horizon"])
    summary = build_summary(arms, cells_by_arm, leads, sim_ids, horizon)
    write_paired_table(out_dir / "paired_by_sim_lead.csv", arms, cells_by_arm,
                       leads, sim_ids)
    summary_fields = ["arm", "substeps", "lead_index", "lead_physical",
                      "lead_normalized", "metric", "aggregation", "estimate",
                      "sem", "median", "win_rate", "n_sims", "n_pairs"]
    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        w.writerows(summary)

    train_n_snap = config.get("training", {}).get("n_snapshots")
    train_t_grid_path = Path(config.get("data", {}).get("t_grid_path", ""))
    if train_n_snap is None:
        train_min_lead, train_min_lead_source = None, "unavailable"
    elif train_t_grid_path.is_file():
        train_min_lead = minimum_lead(np.load(train_t_grid_path), train_n_snap)
        train_min_lead_source = "checkpoint_training_t_grid"
    else:
        train_min_lead = minimum_lead(t_grid, train_n_snap)
        train_min_lead_source = "inferred_from_study_grid"

    manifest = {
        "created_utc": stamp,
        "argv": sys.argv,
        "git_commit": _git_commit(),
        "checkpoint_source_path": str(ckpt_path),
        "checkpoint_path": str(seed_dir / "fno2d_best.pt"),
        "checkpoint_sha256": ckpt_sha,
        "checkpoint_placement": placement,
        "best_validation_value": float(ckpt.get("best_val", float("nan"))),
        "benchmark": benchmark,
        "representation": representation,
        "dataset_dir": str(data_dir),
        "dataset_rng_seed": dataset_rng_seed,
        "training_rng_seed": train_seed,
        "training_rng_seed_source": train_seed_source,
        "sim_params_sha256": _sha256_file(data_dir / "sim_params.npy"),
        "n_simulations": len(sim_ids),
        "n_snapshots_test": n_snap,
        "n_pairs_per_arm": len(reference_keys),
        "snapshot_indices": sorted({int(r["s"]) for r in records["direct"]}
                                   | {int(r["j"]) for r in records["direct"]}),
        "lead_values": leads,
        "substeps": substeps,
        "device": args.device,
        "nx": args.nx,
        "ny": args.ny,
        "time_norm_horizon": horizon,
        "dataset_t_final": float(first["dataset_t_final"]),
        "torch_version": __import__("torch").__version__,
        "numpy_version": np.__version__,
        "training_n_snapshots": train_n_snap,
        "training_minimum_lead": train_min_lead,
        "training_minimum_lead_source": train_min_lead_source,
        "subinterval_diagnostics": subinterval_diagnostics(leads, substeps, train_min_lead),
    }
    (out_dir / "study_manifest.json").write_text(json.dumps(manifest, indent=2))

    write_figure(out_dir / "rollout_crossover.png", arms, summary, leads)

    print_summary(summary, arms, leads)
    print(f"\nStudy written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
