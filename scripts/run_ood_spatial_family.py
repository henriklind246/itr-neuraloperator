"""Zero-shot OOD test: an unseen sinusoidal spatial forcing family.

The forcing benchmark trains on q_L(y, t) = a(t) * s(y) with s(y) drawn from
four spatial families (uniform, patch, gaussian, triangle). This script scores a
trained checkpoint on a spatial profile it has never seen,

    s(y) = c0 + c1 * sin(2*pi*f*y + phi),   c0 = 1 - m, c1 = m, m in [0, 0.5],

while holding the temporal forcing, R_c, and the IC at exactly the trained
distribution.

Three CRN-paired arms share identical latents per sim_id and differ only in s(y):

  sinusoid       the OOD test
  id_baseline    a fresh draw from the four trained families (matched-physics
                 reference; the paired delta is degradation relative to a
                 *typical* in-distribution profile, not a shape-only
                 counterfactual)
  uniform_anchor the sinusoid family at m = 0 exactly, which is numerically
                 identical to the trained uniform family. A correctness anchor
                 on the new code path: read it before anything else.

The forcing cond_static carries eight equal-y-interval averages of s(y), so the
same unseen profile reaches the model honestly through both the spatial field
and, for checkpoints trained with ``spatial_conditioning=full``, the
family-independent conditioning pathway. Checkpoints trained with
``spatial_conditioning=spatial_field_only`` receive zeros in those eight slots.

All uncertainty is over simulations, never over pairs: pairs from one simulation
share latents and a trajectory, so a pair-level SEM is pseudo-replication.
"""

import argparse
import copy
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.generate_ood_dataset import stable_hash
from src.physics.boundary_forcing import (
    SPATIAL_BUILDERS,
    SPATIAL_SAMPLERS,
    SINUSOID_FREQ_RANGE,
    SINUSOID_MOD_DEPTH_RANGE,
    sample_sinusoid_params,
    sample_spatial_family,
)

OOD_AXIS = "spatial_family_sinusoid"
ARMS = ("sinusoid", "id_baseline", "uniform_anchor")
# m = 0 collapses the sinusoid to s(y) == 1; f/phase are then inert.
ANCHOR_PARAMS = {"c0": 1.0, "c1": 0.0, "f": 1.0, "phase": 0.0}
# Offset for the spatial RNG stream so overwriting s(y) cannot perturb the
# temporal/IC/R_c draws that must stay at the trained distribution.
SPATIAL_STREAM_OFFSET = 500_000
ENVELOPE_DRAWS = 20_000


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
    if dst.exists():
        return "existing"
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


def check_preconditions(ckpt):
    """Reject checkpoints for which this study would be dishonest."""
    bench = ckpt.get("conf", {}).get("benchmark", {})
    name = bench.get("name")
    if name != "forcing":
        raise SystemExit(
            f"this study is defined for the forcing benchmark; checkpoint "
            f"reports benchmark.name={name!r}."
        )
    if bench.get("spatial_profile_bins") != 8:
        raise SystemExit(
            "this study requires a checkpoint trained with the eight-bin "
            "spatial-profile conditioning contract "
            "(benchmark.spatial_profile_bins=8)."
        )


# ---------- spatial diagnostics ----------

def spatial_load(spatial_family, spatial_params, y_grid):
    """I_s = int s(y) dy over the domain, by trapezoid on the dataset grid."""
    s_y = SPATIAL_BUILDERS[spatial_family](y_grid, **spatial_params)
    return float(np.trapezoid(np.asarray(s_y, dtype=float), y_grid))


def modulation_depth(spatial_params):
    c0 = float(spatial_params["c0"])
    c1 = float(spatial_params["c1"])
    total = c0 + c1
    return c1 / total if total != 0.0 else float("nan")


def load_envelope(values):
    v = np.asarray(sorted(float(x) for x in values), dtype=float)
    if v.size == 0:
        return {}
    q = [0.0, 5.0, 25.0, 50.0, 75.0, 95.0, 100.0]
    keys = ["min", "p05", "p25", "median", "p75", "p95", "max"]
    return {k: float(np.percentile(v, p)) for k, p in zip(keys, q)}


def trained_family_envelope(y_grid, seed, n_draws=ENVELOPE_DRAWS):
    """I_s spread of the four trained families: the reference envelope."""
    rng = np.random.default_rng(seed)
    loads = []
    for _ in range(n_draws):
        fam = sample_spatial_family(rng)
        params = SPATIAL_SAMPLERS[fam](rng, c=float(y_grid[0]), d=float(y_grid[-1]))
        loads.append(spatial_load(fam, params, y_grid))
    return load_envelope(loads)


# ---------- arm construction ----------

def build_arm_params(id_params, arm, spatial_seed, anchor_sims):
    """Return the sim_params list for one arm, sharing every non-spatial latent."""
    if arm == "id_baseline":
        return copy.deepcopy(id_params)
    if arm == "sinusoid":
        params = copy.deepcopy(id_params)
        rng = np.random.default_rng(spatial_seed)
        for p in params:
            p["spatial_family"] = "sinusoid"
            p["spatial_params"] = sample_sinusoid_params(rng, c=0.0, d=1.0)
        return params
    if arm == "uniform_anchor":
        params = copy.deepcopy(id_params[:anchor_sims])
        for p in params:
            p["spatial_family"] = "sinusoid"
            p["spatial_params"] = dict(ANCHOR_PARAMS)
        return params
    raise ValueError(f"unknown arm {arm!r}")


def latents_hash(params):
    """CRN join key: everything the arms hold in common."""
    return stable_hash({
        "R_c": float(params["R_c"]),
        "temporal_family": params["temporal_family"],
        "temporal_params": params["temporal_params"],
    })


def write_ood_sidecar(path, arm, sim_params, y_grid):
    distribution_class = (
        "out_of_distribution" if arm == "sinusoid" else "in_distribution"
    )
    with Path(path).open("w") as f:
        for sim_id, p in enumerate(sim_params):
            fam = p["spatial_family"]
            sp = p["spatial_params"]
            m = modulation_depth(sp) if fam == "sinusoid" else 0.0
            rec = {
                "sim_id": sim_id,
                "ood_axis": OOD_AXIS,
                "ood_value": float(m),
                "ood_repeat": sim_id,
                "latents_hash": latents_hash(p),
                "distribution_class": distribution_class,
                "arm": arm,
                "spatial_family": fam,
                "spatial_load": spatial_load(fam, sp, y_grid),
            }
            f.write(json.dumps(rec) + "\n")


# ---------- generation ----------

def generate_arms(args, out_dir, ckpt_info=None):
    from data.generate_dataset import (
        build_base_setup, resolve_seed_streams, run_solves, save_dataset,
    )
    from problems.registry import get_problem

    spec = get_problem("forcing")
    seeds = resolve_seed_streams(args.rng_seed)
    setup = build_base_setup(
        num_sims=args.num_sims, save_stride=args.save_stride,
        nx=args.nx, ny=args.ny, t_final=args.t_final, dt=args.dt,
        lhs_seed=seeds["lhs"], forcing_profile_seed=seeds["forcing_profile"],
    )
    y_grid = setup["grids"]["y_grid"]

    # sample_sim_params is reused verbatim, so R_c (LHS), the IC, temporal_family
    # and temporal_params come from exactly the trained distribution in exactly
    # the trained draw order. Only s(y) is overwritten afterwards.
    id_params = spec.sample_sim_params(
        rng=np.random.default_rng(seeds["ic_params"]),
        rng_profile=np.random.default_rng(seeds["forcing_profile"]),
        grids=setup["grids"], time_cfg=setup["time_cfg"],
    )

    arms = list(args.arms)
    anchor_sims = min(args.anchor_sims, args.num_sims)
    arm_dirs = {}
    arm_params = {}
    for arm in arms:
        arm_dir = out_dir / f"data_{arm}"
        arm_dirs[arm] = arm_dir
        params = build_arm_params(
            id_params, arm,
            spatial_seed=seeds["forcing_profile"] + SPATIAL_STREAM_OFFSET,
            anchor_sims=anchor_sims,
        )
        arm_params[arm] = params

        if args.reuse_data and (arm_dir / "trajectories.npy").exists():
            print(f"[{arm}] reusing existing dataset at {arm_dir}", flush=True)
            write_ood_sidecar(arm_dir / "ood_metadata.jsonl", arm, params, y_grid)
            continue

        print(f"[{arm}] solving {len(params)} simulations", flush=True)
        trajectories, t, x, y = run_solves(
            spec, params, setup["base_kwargs"], args.save_stride,
            setup["Nt_saved"], setup["Nx"], setup["Ny"], verbose=False,
        )
        save_dataset(
            arm_dir, x, y, t, args.save_stride, setup["dt"], setup["t_ramp"],
            trajectories, params,
            meta={
                "problem_version": None,
                "ic_mode": None,
                "rng_seed": int(args.rng_seed),
                "seed_streams": seeds,
                "ood_arm": arm,
            },
        )
        write_ood_sidecar(arm_dir / "ood_metadata.jsonl", arm, params, y_grid)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "checkpoint": ckpt_info,
        "ood_axis": OOD_AXIS,
        "arms": arms,
        "num_sims": args.num_sims,
        "anchor_sims": anchor_sims,
        "rng_seed": args.rng_seed,
        "seed_streams": seeds,
        "spatial_stream_seed": seeds["forcing_profile"] + SPATIAL_STREAM_OFFSET,
        "grid": {"nx": args.nx, "ny": args.ny, "t_final": args.t_final,
                 "dt": setup["dt"], "save_stride": args.save_stride,
                 "t_ramp": setup["t_ramp"]},
        "sampler_ranges": {
            "freq_cycles_per_unit_y": list(SINUSOID_FREQ_RANGE),
            "freq_distribution": "log_uniform",
            "modulation_depth": list(SINUSOID_MOD_DEPTH_RANGE),
            "modulation_distribution": "uniform",
            "phase": [0.0, 2.0 * math.pi],
            "phase_distribution": "uniform",
        },
        "spatial_load": {
            "definition": "I_s = int_0^1 s(y) dy (trapezoid on the dataset y grid)",
            "trained_four_family_envelope": trained_family_envelope(
                y_grid, seed=seeds["forcing_profile"] + 2 * SPATIAL_STREAM_OFFSET),
            "trained_envelope_draws": ENVELOPE_DRAWS,
            "arms": {
                arm: load_envelope([
                    spatial_load(p["spatial_family"], p["spatial_params"], y_grid)
                    for p in arm_params[arm]
                ]) for arm in arms
            },
        },
    }
    (out_dir / "ood_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return arm_dirs, arm_params, manifest


# ---------- aggregation ----------

_SUM_COLS = (
    "sse_K2", "num_error_cells", "target_sse_K2",
    "interface_sse_K2", "num_interface_cells", "interface_target_sse_K2",
)


def read_records(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _safe_ratio_sqrt(num, den, scale=1.0):
    if den <= 0.0:
        return float("nan")
    return scale * math.sqrt(num / den)


def pool_to_sims(rows):
    """Pair -> simulation, pooling the sufficient statistics in the records."""
    acc = defaultdict(lambda: {k: 0.0 for k in _SUM_COLS} | {"n_pairs": 0,
                                                             "pair_rmse_sum": 0.0})
    for r in rows:
        c = acc[int(r["sim_id"])]
        for k in _SUM_COLS:
            v = r.get(k, "")
            c[k] += float(v) if v not in ("", None) else 0.0
        c["n_pairs"] += 1
        c["pair_rmse_sum"] += float(r["rmse_K"])
    out = {}
    for sim_id, c in acc.items():
        out[sim_id] = {
            "sim_id": sim_id,
            "n_pairs": c["n_pairs"],
            "rmse_K": _safe_ratio_sqrt(c["sse_K2"], c["num_error_cells"]),
            "rel_l2_pct": _safe_ratio_sqrt(c["sse_K2"], c["target_sse_K2"], 100.0),
            "iface_rmse_K": _safe_ratio_sqrt(
                c["interface_sse_K2"], c["num_interface_cells"]),
            "iface_rel_l2_pct": _safe_ratio_sqrt(
                c["interface_sse_K2"], c["interface_target_sse_K2"], 100.0),
            "mean_pair_rmse_K": c["pair_rmse_sum"] / max(c["n_pairs"], 1),
            **{k: c[k] for k in _SUM_COLS},
        }
    return out


def _lead_key(row):
    # t_bar is a difference of float32 grid times, so one physical lead appears
    # as several values differing at ~1e-8 (0.10000000149 vs 0.10000000894).
    # Snapshot spacing is save_stride*dt >= 1e-4, so 1e-6 separates real leads
    # while collapsing the round-trip noise.
    return round(float(row["t_bar"]), 6)


def lead_bin_edges(rows, n_lead_bins=4):
    """Contiguous chunks of the *distinct* lead values.

    Snapshot times are grid points, so leads are discrete and heavily repeated.
    Quantiles of the raw rows let one physical lead straddle an edge and show up
    as two bins with the same printed range; chunking distinct values cannot.
    """
    leads = sorted({_lead_key(r) for r in rows})
    if not leads:
        return []
    n = min(n_lead_bins, len(leads))
    chunks = np.array_split(np.asarray(leads, dtype=float), n)
    return [float(c[-1]) for c in chunks[:-1]]


def pool_to_sim_lead(rows, edges):
    """Pair -> (simulation, lead bin). Lead is a pair attribute, not a sim one.

    Edges are shared across arms so the bin index is a valid CRN join key.
    """
    grouped = defaultdict(list)
    for r in rows:
        b = int(np.searchsorted(edges, _lead_key(r), side="left"))
        grouped[b].append(r)
    out = {}
    labels = []
    for b in sorted(grouped):
        bin_leads = [_lead_key(r) for r in grouped[b]]
        labels.append((b, f"[{min(bin_leads):.3f}, {max(bin_leads):.3f}]"))
        for sim_id, stats in pool_to_sims(grouped[b]).items():
            out[(sim_id, b)] = stats
    return out, labels


ANCHOR_RTOL = 1e-6


def anchor_check(anchor_sims, id_sims, id_attrs):
    """Compare the m=0 anchor against the id_baseline sims that drew 'uniform'.

    Only sim_ids present in both sets are used. On those, the two arms have
    identical latents *and* an identical s(y) == 1, so the solver trajectories
    and the model inputs are the same object: rmse_K must agree to float
    round-trip, not merely to some statistical tolerance. Comparing means over
    two different sim sets instead would just measure latent spread.
    """
    if not id_sims:
        return None
    ids = [i for i in sorted(anchor_sims)
           if i in id_sims and id_attrs.get(i, {}).get("spatial_family") == "uniform"]
    if not ids:
        return None
    a = np.array([anchor_sims[i]["rmse_K"] for i in ids], dtype=float)
    b = np.array([id_sims[i]["rmse_K"] for i in ids], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.abs(np.where(b > 0, (a - b) / b, np.inf))
    max_rel = float(np.max(rel))
    return {
        "n": len(ids),
        "anchor_mean": float(a.mean()),
        "ref_mean": float(b.mean()),
        "max_rel": max_rel,
        "status": "PASS" if max_rel < ANCHOR_RTOL else "FAIL",
    }


def bootstrap_ci(values, n_boot=2000, alpha=0.05, seed=0):
    """Simulation-level bootstrap CI of the mean. Resamples simulations."""
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if v.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    means = v[idx].mean(axis=1)
    return (float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def summarize(values, seed=0):
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return {"n_sims": 0}
    lo, hi = bootstrap_ci(v, seed=seed)
    return {
        "n_sims": int(v.size),
        "mean": float(v.mean()),
        "ci_lo": lo,
        "ci_hi": hi,
        "median": float(np.median(v)),
        "p90": float(np.percentile(v, 90)),
        "max": float(v.max()),
    }


def quartile_bins(values, labels_fmt="{lo:.3g}-{hi:.3g}"):
    """Return a {sim_id: label} map from a {sim_id: value} map, by quartile."""
    items = [(k, v) for k, v in values.items() if np.isfinite(v)]
    if not items:
        return {}
    vals = np.asarray([v for _, v in items], dtype=float)
    edges = np.quantile(vals, [0.25, 0.5, 0.75])
    bounds = np.concatenate([[vals.min()], edges, [vals.max()]])
    out = {}
    for k, v in items:
        b = int(np.searchsorted(edges, v, side="right"))
        out[k] = f"Q{b + 1} " + labels_fmt.format(lo=bounds[b], hi=bounds[b + 1])
    return out


def sim_attributes(sim_params, y_grid):
    """Per-sim stratification keys derived from the arm's own sim_params."""
    attrs = {}
    for sim_id, p in enumerate(sim_params):
        fam = p["spatial_family"]
        sp = p["spatial_params"]
        rec = {
            "temporal_family": p["temporal_family"],
            "spatial_family": fam,
            "R_c": float(p["R_c"]),
            "I_s": spatial_load(fam, sp, y_grid),
        }
        if fam == "sinusoid":
            rec["m"] = modulation_depth(sp)
            rec["f"] = float(sp["f"])
            rec["phi"] = float(sp["phase"])
        attrs[sim_id] = rec
    return attrs


def build_strata(sims_by_arm, attrs, seed=0):
    """Stratified rows: sinusoid error plus the paired ID delta per stratum."""
    sinus = sims_by_arm.get("sinusoid", {})
    ident = sims_by_arm.get("id_baseline", {})
    if not sinus:
        return []

    keyed = {}
    keyed["temporal_family"] = {i: attrs[i]["temporal_family"] for i in sinus}
    keyed["R_c"] = quartile_bins({i: attrs[i]["R_c"] for i in sinus})
    keyed["I_s"] = quartile_bins({i: attrs[i]["I_s"] for i in sinus})
    if all("m" in attrs[i] for i in sinus):
        keyed["m"] = quartile_bins({i: attrs[i]["m"] for i in sinus})
        keyed["f"] = quartile_bins({i: attrs[i]["f"] for i in sinus})
        keyed["phi"] = {
            i: f"Q{int(attrs[i]['phi'] / (math.pi / 2)) + 1}" for i in sinus
        }
        keyed["temporal_family_x_m"] = {
            i: f"{attrs[i]['temporal_family']}|{keyed['m'][i]}" for i in sinus
        }

    rows = []
    for stratum, mapping in keyed.items():
        groups = defaultdict(list)
        for sim_id, label in mapping.items():
            groups[label].append(sim_id)
        for label in sorted(groups):
            ids = groups[label]
            rmse = [sinus[i]["rmse_K"] for i in ids]
            rel = [sinus[i]["rel_l2_pct"] for i in ids]
            deltas = [sinus[i]["rmse_K"] - ident[i]["rmse_K"]
                      for i in ids if i in ident]
            s_rmse = summarize(rmse, seed=seed)
            s_rel = summarize(rel, seed=seed)
            s_delta = summarize(deltas, seed=seed) if deltas else {"n_sims": 0}
            rows.append({
                "stratum": stratum,
                "level": label,
                "n_sims": len(ids),
                "n_pairs": sum(sinus[i]["n_pairs"] for i in ids),
                "sinusoid_rmse_K_mean": s_rmse.get("mean", float("nan")),
                "sinusoid_rmse_K_median": s_rmse.get("median", float("nan")),
                "sinusoid_rel_l2_pct_mean": s_rel.get("mean", float("nan")),
                "paired_delta_rmse_K_mean": s_delta.get("mean", float("nan")),
                "paired_delta_ci_lo": s_delta.get("ci_lo", float("nan")),
                "paired_delta_ci_hi": s_delta.get("ci_hi", float("nan")),
            })
    return rows


def build_lead_strata(rows_by_arm, seed=0):
    sinus_rows = rows_by_arm.get("sinusoid")
    if not sinus_rows:
        return []
    edges = lead_bin_edges(sinus_rows)
    sinus, labels = pool_to_sim_lead(sinus_rows, edges)
    ident = {}
    if "id_baseline" in rows_by_arm:
        ident, _ = pool_to_sim_lead(rows_by_arm["id_baseline"], edges)
    out = []
    for b, label in labels:
        ids = [k for k in sinus if k[1] == b]
        rmse = [sinus[k]["rmse_K"] for k in ids]
        deltas = [sinus[k]["rmse_K"] - ident[k]["rmse_K"] for k in ids if k in ident]
        s_rmse = summarize(rmse, seed=seed)
        s_delta = summarize(deltas, seed=seed) if deltas else {"n_sims": 0}
        out.append({
            "stratum": "lead_time",
            "level": label,
            "n_sims": len(ids),
            "n_pairs": sum(sinus[k]["n_pairs"] for k in ids),
            "sinusoid_rmse_K_mean": s_rmse.get("mean", float("nan")),
            "sinusoid_rmse_K_median": s_rmse.get("median", float("nan")),
            "sinusoid_rel_l2_pct_mean": float("nan"),
            "paired_delta_rmse_K_mean": s_delta.get("mean", float("nan")),
            "paired_delta_ci_lo": s_delta.get("ci_lo", float("nan")),
            "paired_delta_ci_hi": s_delta.get("ci_hi", float("nan")),
        })
    return out


# ---------- reporting ----------

def _fmt(v, w=10, p=4):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-".rjust(w)
    if isinstance(v, float):
        return f"{v:.{p}g}".rjust(w)
    return str(v).rjust(w)


def report(out_dir, arm_dirs, arm_params, manifest, args):
    y_grid = np.load(next(iter(arm_dirs.values())) / "y_grid.npy")
    rows_by_arm = {}
    sims_by_arm = {}
    for arm in arm_dirs:
        path = out_dir / "ckpt" / args.seed_name / f"records_{arm}.csv"
        if not path.exists():
            continue
        rows_by_arm[arm] = read_records(path)
        sims_by_arm[arm] = pool_to_sims(rows_by_arm[arm])

    attrs = sim_attributes(arm_params["sinusoid"], y_grid)
    id_attrs = (sim_attributes(arm_params["id_baseline"], y_grid)
                if "id_baseline" in arm_params else {})

    # ---- per-sim CSV ----
    per_sim_path = out_dir / "per_sim.csv"
    fields = ["arm", "sim_id", "n_pairs", "rmse_K", "rel_l2_pct", "iface_rmse_K",
              "iface_rel_l2_pct", "mean_pair_rmse_K", "temporal_family",
              "spatial_family", "R_c", "I_s", "m", "f", "phi"]
    with per_sim_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for arm, sims in sims_by_arm.items():
            src = sim_attributes(arm_params[arm], y_grid)
            for sim_id in sorted(sims):
                w.writerow({"arm": arm, **sims[sim_id], **src.get(sim_id, {})})

    # ---- overall ----
    overall = []
    for arm in ARMS:
        if arm not in sims_by_arm:
            continue
        sims = sims_by_arm[arm]
        for metric in ("rmse_K", "rel_l2_pct", "iface_rmse_K"):
            s = summarize([sims[i][metric] for i in sims], seed=args.rng_seed)
            overall.append({"arm": arm, "metric": metric, "scope": "all_sims",
                            "n_pairs": sum(sims[i]["n_pairs"] for i in sims), **s})

    # Paired CRN delta, headline statistic.
    paired = []
    if "id_baseline" in sims_by_arm and "sinusoid" in sims_by_arm:
        a, b = sims_by_arm["sinusoid"], sims_by_arm["id_baseline"]
        shared = sorted(set(a) & set(b))
        for metric in ("rmse_K", "rel_l2_pct"):
            d = [a[i][metric] - b[i][metric] for i in shared]
            r = [a[i][metric] / b[i][metric] for i in shared
                 if b[i][metric] > 0.0]
            s = summarize(d, seed=args.rng_seed)
            paired.append({"arm": "sinusoid_minus_id", "metric": metric,
                           "scope": "paired_delta", "n_pairs": "", **s})
            sr = summarize(r, seed=args.rng_seed)
            paired.append({"arm": "sinusoid_over_id", "metric": metric,
                           "scope": "paired_ratio", "n_pairs": "", **sr})
    overall.extend(paired)

    overall_path = out_dir / "summary_overall.csv"
    keys = ["arm", "metric", "scope", "n_sims", "n_pairs", "mean", "ci_lo",
            "ci_hi", "median", "p90", "max"]
    with overall_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(overall)

    # ---- strata ----
    strata = build_strata(sims_by_arm, attrs, seed=args.rng_seed)
    strata += build_lead_strata(rows_by_arm, seed=args.rng_seed)
    strata_path = out_dir / "summary_strata.csv"
    with strata_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(strata[0]) if strata else ["stratum"])
        w.writeheader()
        w.writerows(strata)

    from visual.forcing_plots import (
        generate_zero_shot_sinusoid_field_and_jump,
        plot_forcing_sinusoid_temporal_panels,
    )

    forcing_plot_path = out_dir / "sinusoid_temporal_forcing.png"
    plot_forcing_sinusoid_temporal_panels(
        seed=args.rng_seed,
        save_path=forcing_plot_path,
        dt=float(manifest["grid"]["dt"]),
        t_final=float(manifest["grid"]["t_final"]),
    )
    case_plot_base = out_dir / "zero_shot_field_and_jump"
    case_plot = generate_zero_shot_sinusoid_field_and_jump(
        checkpoint_path=out_dir / "ckpt" / args.seed_name / "fno2d_best.pt",
        data_dir=arm_dirs["sinusoid"],
        save_path=case_plot_base,
    )

    # ---- console ----
    print()
    print("=" * 78)
    print("ZERO-SHOT OOD: unseen sinusoidal spatial forcing family")
    print("=" * 78)

    # Anchor first: a blown anchor invalidates everything downstream.
    if "uniform_anchor" in sims_by_arm:
        anc = sims_by_arm["uniform_anchor"]
        ids = sorted(anc)
        print()
        print("[1] ANCHOR CHECK (read this first)")
        print(f"    uniform_anchor (sinusoid at m=0) mean rmse_K = "
              f"{np.mean([anc[i]['rmse_K'] for i in ids]):.5g} over {len(ids)} sims")
        chk = anchor_check(anc, sims_by_arm.get("id_baseline"), id_attrs)
        if chk is None:
            print("    (no id_baseline sim drew 'uniform' among the anchor sim_ids)")
        else:
            print(f"    CRN-paired subset: {chk['n']} sim_ids where the id_baseline arm "
                  f"also drew 'uniform'")
            print(f"    anchor {chk['anchor_mean']:.8f}  vs  id_baseline "
                  f"{chk['ref_mean']:.8f}  (identical s(y), identical latents)")
            print(f"    max per-sim relative difference = {chk['max_rel']:.3e}  "
                  f"-> {chk['status']}")
            if chk["status"] == "FAIL":
                print("    A blown anchor means the new profile/conditioning/data path "
                      "is broken.\n    Do not interpret the OOD numbers below.")

    print()
    print("[2] OVERALL (simulation-level; CI is a simulation bootstrap)")
    print(f"    {'arm':<18}{'metric':<14}{'n_sims':>8}{'mean':>11}"
          f"{'ci_lo':>11}{'ci_hi':>11}{'median':>11}{'p90':>11}")
    for r in overall:
        print(f"    {r['arm']:<18}{r['metric']:<14}"
              f"{_fmt(r.get('n_sims'), 8)}{_fmt(r.get('mean'), 11)}"
              f"{_fmt(r.get('ci_lo'), 11)}{_fmt(r.get('ci_hi'), 11)}"
              f"{_fmt(r.get('median'), 11)}{_fmt(r.get('p90'), 11)}")
    print()
    print("    The paired delta is matched-physics degradation relative to a")
    print("    TYPICAL in-distribution spatial profile (a fresh draw across the")
    print("    four trained families), not a shape-matched counterfactual.")

    print()
    print("[3] STRATIFIED (sinusoid arm; delta vs the CRN-paired ID sim)")
    print(f"    {'stratum':<22}{'level':<26}{'n_sims':>7}{'n_pairs':>9}"
          f"{'rmse_K':>10}{'delta':>10}{'ci_lo':>10}{'ci_hi':>10}")
    for r in strata:
        print(f"    {r['stratum']:<22}{str(r['level'])[:25]:<26}"
              f"{_fmt(r['n_sims'], 7)}{_fmt(r['n_pairs'], 9)}"
              f"{_fmt(r['sinusoid_rmse_K_mean'], 10)}"
              f"{_fmt(r['paired_delta_rmse_K_mean'], 10)}"
              f"{_fmt(r['paired_delta_ci_lo'], 10)}"
              f"{_fmt(r['paired_delta_ci_hi'], 10)}")

    print()
    print("[4] SPATIAL LOAD I_s = int s(y) dy  (total-load confound check)")
    env = manifest["spatial_load"]
    cols = ["min", "p05", "p25", "median", "p75", "p95", "max"]
    print(f"    {'source':<28}" + "".join(c.rjust(9) for c in cols))
    tr = env["trained_four_family_envelope"]
    print(f"    {'trained 4 families':<28}" + "".join(_fmt(tr.get(c), 9) for c in cols))
    for arm, e in env["arms"].items():
        print(f"    {arm:<28}" + "".join(_fmt(e.get(c), 9) for c in cols))
    sin_env = env["arms"].get("sinusoid", {})
    if sin_env and tr:
        inside = (sin_env["min"] >= tr["min"] - 1e-9
                  and sin_env["max"] <= tr["max"] + 1e-9)
        if inside:
            print("    Sinusoid I_s sits inside the trained envelope: total load is")
            print("    not a confound for the shape reading.")
        else:
            print("    Sinusoid I_s escapes the trained envelope. The result is NOT")
            print("    attributable to shape alone; report the load confound.")

    print()
    print(f"Wrote {overall_path}")
    print(f"Wrote {strata_path}")
    print(f"Wrote {per_sim_path}")
    print(f"Wrote {forcing_plot_path}")
    for path in case_plot["paths"]:
        print(f"Wrote {path}")


# ---------- main ----------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", help="Path to a run_root or a bare fno2d_best.pt")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=None,
                   help="Seed subdir when checkpoint is a run_root")
    p.add_argument("--num-sims", type=int, default=500)
    p.add_argument("--anchor-sims", type=int, default=64)
    # 16 of the 31 saved frames: 120 pairs/sim, min lead 0.02. Halving the
    # minimum lead from 8 -> 16 snapshots moved the headline rmse_K by ~3%
    # (0.5335 -> 0.5183) and the ID ratio from 13.0x to 12.8x, so the remaining
    # 4x of forward passes to reach all 31 frames buys precision on a number
    # that is not in doubt.
    p.add_argument("--n-snapshots-test", type=int, default=16)
    p.add_argument("--rng-seed", type=int, default=7)
    p.add_argument("--device", default=None)
    p.add_argument("--inference-batch-size", type=int, default=32)
    p.add_argument("--no-id-baseline", action="store_true")
    p.add_argument("--no-uniform-anchor", action="store_true")
    p.add_argument("--reuse-data", action="store_true")
    p.add_argument("--skip-eval", action="store_true",
                   help="Generate the arms and stop (no scoring)")
    p.add_argument("--nx", type=int, default=100)
    p.add_argument("--ny", type=int, default=100)
    p.add_argument("--t-final", type=float, default=0.30)
    p.add_argument("--dt", type=float, default=0.005)
    p.add_argument("--save-stride", type=int, default=2)
    p.add_argument("--allow-symlink", action="store_true")
    args = p.parse_args(argv)
    args.arms = ["sinusoid"]
    if not args.no_id_baseline:
        args.arms.append("id_baseline")
    if not args.no_uniform_anchor:
        args.arms.append("uniform_anchor")
    return args


def main(argv=None):
    args = parse_args(argv)
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path, ckpt, seed_name = resolve_checkpoint(args.checkpoint, args.seed)
    check_preconditions(ckpt)
    args.seed_name = seed_name
    seed_num = int(seed_name.replace("seed", "")) if seed_name.startswith("seed") else 0

    staged = out_dir / "ckpt" / seed_name / "fno2d_best.pt"
    how = materialize_checkpoint(ckpt_path, staged, args.allow_symlink)
    print(f"Checkpoint {ckpt_path} -> {staged} ({how})", flush=True)
    print(f"  experiment: {ckpt['conf'].get('experiment', {}).get('name')}", flush=True)

    arm_dirs, arm_params, manifest = generate_arms(args, out_dir, ckpt_info={
        "source_path": str(ckpt_path),
        "experiment": ckpt["conf"].get("experiment", {}).get("name"),
        "epoch": ckpt.get("epoch"),
        "best_val": ckpt.get("best_val"),
        "config_id": ckpt["conf"].get("config_id"),
        "spatial_conditioning": ckpt["conf"]["benchmark"].get("spatial_conditioning"),
        "spatial_profile_bins": ckpt["conf"]["benchmark"].get(
            "spatial_profile_bins"
        ),
        "representation": ckpt["conf"]["benchmark"].get("representation")
                          or ckpt["conf"].get("representation"),
        "mu_global": ckpt.get("mu_global"),
        "sigma_global": ckpt.get("sigma_global"),
    })
    if args.skip_eval:
        return 0

    from src.operators.eval import write_test_records

    for arm, arm_dir in arm_dirs.items():
        print(f"[{arm}] scoring", flush=True)
        write_test_records(
            run_root=out_dir / "ckpt",
            seed=seed_num,
            out_name=f"records_{arm}.csv",
            data_dir=str(arm_dir),
            eval_all_sims=True,
            n_snapshots_test=args.n_snapshots_test,
            device=args.device,
            inference_batch_size=args.inference_batch_size,
        )

    report(out_dir, arm_dirs, arm_params, manifest, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
