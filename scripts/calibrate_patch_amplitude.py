"""Screen source amplitudes and interface jumps on both source benchmarks."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.generate_dataset import build_base_setup
from problems.registry import get_problem
from problems.source import CP_LEFT, CP_RIGHT, K_LEFT, K_RIGHT, RHO_LEFT, RHO_RIGHT
from src.physics.internal_source import PATCH_A_RANGE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=100)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--num-samples", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-rise", type=float, default=5.0)
    parser.add_argument("--max-rise", type=float, default=150.0)
    parser.add_argument("--output", type=Path, default=Path("tmp/source_material_calibration"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    setup = build_base_setup(
        num_sims=args.num_samples, save_stride=1, nx=args.nx, ny=args.nx,
        dt=args.dt, lhs_seed=args.seed,
    )
    rows = []
    checks = []
    for name in ("source", "source_itr_sin"):
        spec = get_problem(name, "temporal_encoder")
        params = spec.sample_sim_params(
            np.random.default_rng(args.seed), np.random.default_rng(args.seed + 1),
            setup["grids"], setup["time_cfg"],
        )
        cases = [(f"sample_{i}", p) for i, p in enumerate(params)]
        resistance_cases = [(0.05, 0.0), (1.0, 0.0)]
        if name == "source_itr_sin":
            resistance_cases += [(0.05, 2.95), (1.0, 2.0)]
        for x_h in (0.05, 0.25, 0.45, 0.5, 0.55, 0.75, 0.95):
            for y_h in (0.05, 0.5, 0.95):
                for rc, rc_amp in resistance_cases:
                    p = dict(params[0], x_h=x_h, y_h=y_h, R_c=rc)
                    if name == "source_itr_sin":
                        p.update(R_c_base=rc, R_c_A=rc_amp)
                    cases.append((f"anchor_{x_h}_{y_h}_{rc}_{rc_amp}", p))
        for index, (label, p) in enumerate(cases):
            # The PDE is linear in amplitude with fixed material properties and
            # a uniform 300 K initial/boundary temperature.
            p = dict(p, A=1.0)
            solver = spec.configure_solver(p, setup["base_kwargs"])
            t, x, y, history = solver.solve(T0=p["T0"], store_trajectory=True)
            rise = history - 300.0
            left = int(np.searchsorted(x, p["interface_x"]) - 1)
            node_jump = history[:, left, :] - history[:, left + 1, :]
            rc = np.broadcast_to(solver.interface_R[0], y.shape)
            contact_jump = node_jump * (rc * solver.G_x[left, :])[None, :]
            peak_index = np.unravel_index(np.argmax(rise), rise.shape)
            row = dict(
                benchmark=name, case=label, x_h=p["x_h"], y_h=p["y_h"],
                R_c=p["R_c"], R_c_A=p.get("R_c_A", 0.0),
                peak_rise_per_unit_A=float(np.max(rise)),
                minimum_rise_per_unit_A=float(np.min(rise)),
                peak_time_s=float(t[peak_index[0]]),
                max_node_jump_per_unit_A=float(np.max(np.abs(node_jump))),
                max_contact_jump_per_unit_A=float(np.max(np.abs(contact_jump))),
                max_right_boundary_error_K=float(np.max(np.abs(rise[:, -1, :]))),
            )
            if not np.isfinite(history).all():
                raise RuntimeError(f"Nonfinite trajectory: {name}/{label}")
            rows.append(row)
            if index % 20 == 0:
                print(f"{name}: {index + 1}/{len(cases)} cases", flush=True)
        benchmark_rows = [r for r in rows if r["benchmark"] == name]
        for key in ("peak_rise_per_unit_A", "max_contact_jump_per_unit_A"):
            for choose in (min, max):
                selected = choose(benchmark_rows, key=lambda r: r[key])
                label, p = next(c for c in cases if c[0] == selected["case"])
                amplitude = PATCH_A_RANGE[0 if choose is min else 1]
                solver = spec.configure_solver(dict(p, A=amplitude), setup["base_kwargs"])
                t, x, y, history = solver.solve(T0=p["T0"], store_trajectory=True)
                measured = float(np.max(history - 300.0))
                predicted = amplitude * selected["peak_rise_per_unit_A"]
                np.testing.assert_allclose(measured, predicted, rtol=1e-7, atol=1e-7)
                checks.append(dict(benchmark=name, case=label, amplitude=amplitude,
                                   measured_peak_rise_K=measured,
                                   linearity_error_K=abs(measured - predicted)))
                np.savez_compressed(
                    args.output / f"{name}_{key}_{choose.__name__}.npz",
                    t=t, x=x, y=y, temperature=history, amplitude=amplitude,
                    interface_R=solver.interface_R[0],
                )
    gains = [r["peak_rise_per_unit_A"] for r in rows]
    proposed = [float(np.ceil(args.min_rise / min(gains) * 10) / 10),
                float(np.floor(args.max_rise / max(gains) * 10) / 10)]
    decade_max = float(np.floor(proposed[1] / 10) * 10)
    summary = dict(
        benchmarks=["source", "source_itr_sin"], representation="temporal_encoder",
        problem_version=get_problem("source").problem_version,
        units=dict(length="mm", time="s", density="kg/mm^3", cp="J/(kg K)",
                   conductivity="W/(mm K)", source="W/mm^3", resistance="mm^2 K/W"),
        materials=dict(left=dict(k=K_LEFT, rho=RHO_LEFT, cp=CP_LEFT),
                       right=dict(k=K_RIGHT, rho=RHO_RIGHT, cp=CP_RIGHT)),
        nx=args.nx, dt=args.dt, t_final=0.3, seed=args.seed,
        num_cases=len(rows), target_peak_rise_K=[args.min_rise, args.max_rise],
        amplitude_bounds_for_all_cases=proposed,
        feasible_common_range=proposed[0] <= proposed[1],
        ceiling_limited_decade_range=[decade_max / 10, decade_max],
        configured_amplitude_range=list(PATCH_A_RANGE),
        configured_peak_rise_envelope_K=[min(gains) * PATCH_A_RANGE[0],
                                        max(gains) * PATCH_A_RANGE[1]],
        minimum_temperature_K=300.0 + min(r["minimum_rise_per_unit_A"] for r in rows) * PATCH_A_RANGE[1],
        maximum_right_boundary_error_K=max(r["max_right_boundary_error_K"] for r in rows),
        direct_amplitude_checks=checks,
        scope="Finite screening set; range is not a bound over every possible patch location.",
    )
    with (args.output / "cases.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
