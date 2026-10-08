"""Reproduce the B1--B4 thermal-range table from fresh, unique FV snapshots."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

import numpy as np
import scipy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.generate_dataset import (
    build_balanced_ic_families,
    build_base_setup,
    resolve_seed_streams,
)
from problems.registry import get_problem
from src.physics.init_conditions import IC_FAMILIES

BENCHMARKS = ("forcing", "interfaces", "source", "source_itr_sin")


def snapshot_statistics(fields, face):
    fields = np.asarray(fields, dtype=np.float64)
    node_jump = fields[:, face, :] - fields[:, face + 1, :]
    return {
        "T_min_K": fields.min(axis=(1, 2)),
        "T_max_K": fields.max(axis=(1, 2)),
        "field_range_K": np.ptp(fields, axis=(1, 2)),
        "node_jump_min_K": node_jump.min(axis=1),
        "node_jump_max_K": node_jump.max(axis=1),
        "node_jump_abs_max_K": np.abs(node_jump).max(axis=1),
    }


def summarize(records):
    return {
        "T_min_K": float(records["T_min_K"].min()),
        "T_max_K": float(records["T_max_K"].max()),
        "field_range_p95_K": float(np.percentile(records["field_range_K"], 95)),
        "node_jump_min_K": float(records["node_jump_min_K"].min()),
        "node_jump_max_K": float(records["node_jump_max_K"].max()),
        "node_jump_abs_max_p95_K": float(np.percentile(records["node_jump_abs_max_K"], 95)),
    }


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def solve_benchmark(name, num_sims, rng_seed, output_dir):
    start = time.perf_counter()
    spec = get_problem(name)
    seeds = resolve_seed_streams(rng_seed)
    setup = build_base_setup(
        num_sims=num_sims, save_stride=2, lhs_seed=seeds["lhs"],
        forcing_profile_seed=seeds["forcing_profile"],
    )
    if getattr(spec, "ic_mode", None) == "varying":
        setup["time_cfg"]["ic_family_assignment"] = build_balanced_ic_families(
            num_sims, list(IC_FAMILIES), seed=seeds["ic_assign"],
        )
    params = spec.sample_sim_params(
        np.random.default_rng(seeds["ic_params"]),
        np.random.default_rng(seeds["forcing_profile"]),
        setup["grids"], setup["time_cfg"],
    )
    records = {}
    metadata = []
    temperature_hash = hashlib.sha256()
    extrema = {}
    for sid, param in enumerate(params):
        solver = spec.configure_solver(param, setup["base_kwargs"])
        t, x, y, history = solver.solve(T0=param["T0"], store_trajectory=True)
        # Match the dataset generator's persisted precision and snapshot cadence.
        fields = history[::2].astype(np.float32)
        temperature_hash.update(fields.tobytes())
        face = next(iter(solver.interface_offsets))
        values = snapshot_statistics(fields, face)
        if not records:
            records = {key: np.empty((num_sims, len(t[::2]))) for key in values}
        for key, value in values.items():
            records[key][sid] = value
        for key, direction in (("T_min_K", "min"), ("T_max_K", "max"),
                               ("node_jump_min_K", "min"), ("node_jump_max_K", "max")):
            index = int(np.argmin(values[key]) if direction == "min" else np.argmax(values[key]))
            value = float(values[key][index])
            previous = extrema.get(key)
            if previous is None or (value < previous["value"] if direction == "min" else value > previous["value"]):
                extrema[key] = {"sim_id": sid, "snapshot_index": index, "time_s": float(t[::2][index]), "value": value}
        if sid == 0:
            physical_setup = {
                "layers": [asdict(layer) for layer in solver.layers],
                "solver_dt": float(solver.dt),
                "ramp_seconds": setup["t_ramp"],
                "grid_shape": [len(x), len(y)],
                "saved_times_s": t[::2].tolist(),
            }
        metadata.append({
            **{key: value for key, value in param.items() if key != "T0"},
            "T0_sha256": hashlib.sha256(param["T0"].tobytes()).hexdigest(),
            "interface_x": float(solver.interface_positions[0]),
            "left_node_index": int(face),
            "right_node_index": int(face + 1),
        })
        if (sid + 1) % 250 == 0 or sid + 1 == num_sims:
            print(f"{name}: {sid + 1}/{num_sims} ({time.perf_counter() - start:.1f}s)", flush=True)
    destination = Path(output_dir)
    np.savez_compressed(destination / f"{name}_snapshots.npz", **records, t_grid=t[::2])
    (destination / f"{name}_parameters.json").write_text(json.dumps(metadata, default=json_value))
    result = {
        "benchmark": name,
        "representation": spec.representation,
        "problem_version": getattr(spec, "problem_version", None),
        "dims": asdict(spec.dims),
        "num_simulations": num_sims,
        "num_unique_snapshots": int(records["T_min_K"].size),
        "rng_seed": rng_seed,
        "seed_streams": seeds,
        "setup": physical_setup,
        "metrics": summarize(records),
        "extrema_locations": extrema,
        "saved_temperature_sha256": temperature_hash.hexdigest(),
        "wall_time_seconds": time.perf_counter() - start,
    }
    (destination / f"{name}_summary.json").write_text(json.dumps(result, indent=2))
    return result


def write_table(results, destination, num_sims, rng_seed):
    keys = ("T_min_K", "T_max_K", "field_range_p95_K", "node_jump_min_K",
            "node_jump_max_K", "node_jump_abs_max_p95_K")
    rows = []
    for i, result in enumerate(results, 1):
        values = [f"{result['metrics'][key]:.2f}" for key in keys]
        values = ["0.00" if value == "-0.00" else value for value in values]
        rows.append("B" + str(i) + " & " + " & ".join(values) + r" \\")
    table = r"""\begin{table}[htbp]
\centering
\small
\setlength{\tabcolsep}{5pt}
\begin{tabular}{lrrrrrr}
\toprule
Benchmark & $T_{\min}$ & $T_{\max}$ & $P_{95}(\Delta T_{\mathrm{field}})$
& $\Delta T_{\mathrm{node},\min}$ & $\Delta T_{\mathrm{node},\max}$ & $P_{95}(J_{\max})$ \\
& (K) & (K) & (K) & (K) & (K) & (K) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\caption{Temperature and signed adjacent-node jump statistics from fresh
finite-volume simulations using the benchmark data-generation distributions.
Each benchmark contains NUMSIMS trajectories and NUMSNAPS unique saved snapshots
(31 per trajectory, including $t=0$), on a $100\times100$ grid with
$\Delta t=0.005\,\mathrm{s}$, saved every $0.01\,\mathrm{s}$ through
$t=0.3\,\mathrm{s}$. Extrema are taken over all sampled simulations,
saved times and relevant spatial nodes; percentiles are taken over the
unique simulation--snapshot population with equal snapshot weights.
B1: boundary forcing; B2: varying interface; B3: internal source;
B4: internal source with sinusoidal ITR. These are sample statistics,
not extrema or percentiles of the original training datasets.}
\label{tab:fv-thermal-ranges}
\end{table}
"""
    table = table.replace("NUMSIMS", f"{num_sims:,}").replace("NUMSNAPS", f"{num_sims * 31:,}")
    (destination / "table.tex").write_text(table)
    definitions = r"""
For each saved field, define
\[
\Delta T_{\mathrm{field}}=\max_{x,y}T-\min_{x,y}T,
\qquad
\Delta T_{\mathrm{node}}(y,t)=T(x_L,y,t)-T(x_R,y,t),
\qquad
J_{\max}=\max_y|\Delta T_{\mathrm{node}}(y,t)|.
\]
Here $x_L<x_\Gamma<x_R$ are the two adjacent grid nodes that flank the
simulation's interface. A positive jump denotes a temperature drop from
left to right. This is the adjacent-node jump, including the two bulk
conduction contributions; it is not the reconstructed interface-face contact
jump. The evaluation code uses the opposite signed subtraction, which gives
the same jump-error magnitudes.

The finite-volume fields are rounded to float32, as in dataset generation,
before statistics are computed in float64. The 95th percentiles use linear
interpolation (NumPy's default). No source--target pair weighting or
temperature normalization is applied. Sampling uses generator RNG family
RNGSEED. Material properties, source strengths, initial conditions and ITR
profiles are taken from the current benchmark ProblemSpecs; B3 and B4 use
the titanium/brass material setup and B4 uses the sinusoidal ITR profile.
""".replace("RNGSEED", str(rng_seed))
    document = r"""\documentclass[10pt]{article}
\usepackage[margin=0.65in]{geometry}
\usepackage{amsmath,booktabs}
\begin{document}
\pagestyle{empty}
""" + table + definitions + "\n\\end{document}\n"
    (destination / "fv_benchmark_table.tex").write_text(document)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-sims", type=int, default=2000)
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.num_sims < 1 or args.workers < 1:
        parser.error("num-sims and workers must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_paths = [Path(__file__), PROJECT_ROOT / "data/generate_dataset.py"]
    source_paths += sorted((PROJECT_ROOT / "problems").glob("*.py"))
    source_paths += sorted((PROJECT_ROOT / "src/physics").glob("*.py"))
    provenance = {
        "command": shlex.join([sys.executable, *sys.argv]),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip(),
        "source_sha256": {str(p.relative_to(PROJECT_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        "numpy_version": np.__version__, "scipy_version": scipy.__version__,
        "population": "fresh FV samples; all saved snapshots including t=0; no pair multiplicity",
        "jump": "adjacent left node minus adjacent right node",
        "percentile": "95th percentile across simulation-snapshot values; linear interpolation",
    }
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(solve_benchmark, name, args.num_sims, args.rng_seed, str(args.output_dir)) for name in BENCHMARKS]
        results = [job.result() for job in jobs]
    provenance["benchmarks"] = results
    (args.output_dir / "results.json").write_text(json.dumps(provenance, indent=2))
    write_table(results, args.output_dir, args.num_sims, args.rng_seed)
    print(json.dumps({result["benchmark"]: result["metrics"] for result in results}, indent=2))


if __name__ == "__main__":
    main()
