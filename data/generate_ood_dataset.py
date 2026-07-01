"""CRN-paired out-of-distribution dataset generator.

Sweeps a single declared OOD axis of one benchmark while holding every other
factor at a shared, in-distribution background. For each ``repeat`` the spec
draws one shared-latent bundle (``draw_latents``) and maps it through every
swept value (``apply_ood_value``), so equal-``repeat`` sims are exactly paired
(common random numbers). The leading value(s) of the sweep are the axis's
in-distribution reference, produced inside the same sweep.

The standard dataset .npy files are written via the shared helpers in
``data/generate_dataset.py`` (byte-for-byte the same layout the trainer/eval
expect). OOD identity lives in two dedicated sidecars so ``sim_params`` stays
exactly what validation/numeric code consume:

- ``ood_metadata.jsonl`` -- one JSON record per sim id, joined by ``sim_id``.
- ``ood_manifest.json`` -- dataset-only provenance, reusable across checkpoints.

The ``time`` axis is special (``kind="evaluation_parameter"``): one extended
trajectory is generated per repeat (solved to ``dataset_t_final``), and the
swept target times are applied later by the eval stage, not by spawning sims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from problems.base import OODAxis
from problems.registry import get_problem
from data.generate_dataset import build_base_setup, run_solves, save_dataset

DATA_DIR = Path(__file__).resolve().parent

# Per-repeat seed offsets keep the IC/forcing streams independent and the
# background reproducible across runs.
_RNG_BASE = 0
_RNG_PROFILE_OFFSET = 100_000


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _update_hash(h: "hashlib._Hash", obj) -> None:
    if isinstance(obj, np.ndarray):
        h.update(b"ndarray")
        h.update(np.ascontiguousarray(obj).tobytes())
        h.update(str(obj.shape).encode())
        h.update(str(obj.dtype).encode())
    elif isinstance(obj, dict):
        for k in sorted(obj.keys(), key=str):
            h.update(str(k).encode())
            _update_hash(h, obj[k])
    elif isinstance(obj, (list, tuple)):
        h.update(b"seq")
        for x in obj:
            _update_hash(h, x)
    else:
        h.update(repr(obj).encode())


def stable_hash(obj) -> str:
    """Stable short fingerprint of a latents/params bundle (CRN pairing key)."""
    h = hashlib.sha256()
    _update_hash(h, obj)
    return h.hexdigest()[:16]


def classify_value(axis: OODAxis, value, atol: float = 1e-9) -> str:
    """4-level distribution class of a swept value vs the trained range.

    A zero of a strictly-positive trained quantity is a ``limiting_case``
    (e.g. perfect contact R_c=0); endpoints are ``boundary``; strictly inside
    is ``in_distribution``; otherwise ``out_of_distribution``. The leading
    ``id_reference`` value is always in-distribution by construction.
    """
    if any(value == ref for ref in axis.id_reference):
        return "in_distribution"
    tr = axis.trained_range
    if tr is None:
        # Compound / non-numeric OOD value with no numeric range to compare.
        return "out_of_distribution"
    lo, hi = float(tr[0]), float(tr[1])
    v = float(value)
    if v == 0.0 and lo > 0.0:
        return "limiting_case"
    if abs(v - lo) <= atol or abs(v - hi) <= atol:
        return "boundary"
    if lo < v < hi:
        return "in_distribution"
    return "out_of_distribution"


def _value_to_json(value):
    """JSON-friendly rendering of a swept value (tuples -> '+'-joined)."""
    if isinstance(value, (tuple, list)):
        return "+".join(str(v) for v in value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _parse_values(axis_name: str, raw: list[str]):
    """Parse CLI --values for an axis (family tuples vs floats)."""
    if axis_name == "family_transfer":
        out = []
        for v in raw:
            parts = v.split("+")
            if len(parts) != 2:
                raise ValueError(
                    f"family_transfer value {v!r} must be 'temporal+spatial'."
                )
            out.append((parts[0], parts[1]))
        return out
    return [float(v) for v in raw]


def _resolve_axis(spec, axis_name, values, dataset_t_final, time_norm_horizon):
    """Return (axis, sweep_values, t_final) for the requested axis."""
    if axis_name == "time":
        if not values:
            raise ValueError("--values (target times) required for the time axis.")
        t_final = dataset_t_final if dataset_t_final is not None else max(values)
        axis = OODAxis(
            name="time",
            kind="evaluation_parameter",
            id_reference=(),
            ood_values=tuple(values),
            field=None,
            dataset_t_final=t_final,
            time_norm_horizon=time_norm_horizon,
        )
        return axis, list(values), t_final

    axes = spec.ood_axes()
    if axis_name not in axes:
        raise KeyError(
            f"Benchmark {spec.name!r} has no OOD axis {axis_name!r}. "
            f"Available: {sorted(axes)}"
        )
    axis = axes[axis_name]
    sweep = list(values) if values else list(axis.sweep_values())
    t_final = dataset_t_final if dataset_t_final is not None else axis.dataset_t_final
    return axis, sweep, t_final


def generate_ood_dataset(
    benchmark: str,
    axis_name: str,
    save_dir: Path | str,
    representation: str = "temporal_encoder",
    values: list | None = None,
    repeats: int = 2,
    dataset_t_final: float | None = None,
    time_norm_horizon: float = 0.30,
    save_stride: int = 2,
    nx: int = 100,
    ny: int = 100,
    dt: float | None = None,
    ramp_seconds: float | None = None,
    rng_seed: int = 0,
    suite_yaml_path: str | None = None,
    verbose: bool = True,
) -> Path:
    """Generate one CRN OOD dataset for a single axis. Returns the save dir.

    ``nx``/``ny``/``dt``/``save_stride`` are exposed so the suite's convergence
    pass can regenerate the SAME CRN params (identical ``rng_seed``/``repeats``/
    ``values``) on a refined grid; pairing a halved ``dt`` with a doubled
    ``save_stride`` keeps the saved snapshots at the same physical times across
    resolutions so the solver self-difference is a pure discretization signal.
    """
    spec = get_problem(benchmark, representation)
    axis, sweep, t_final = _resolve_axis(
        spec, axis_name, values, dataset_t_final, time_norm_horizon,
    )
    is_time = axis.kind == "evaluation_parameter"

    num_sims = repeats if is_time else repeats * len(sweep)
    setup_kwargs = dict(
        num_sims=num_sims, save_stride=save_stride, nx=nx, ny=ny,
        t_final=t_final, ramp_seconds=ramp_seconds,
    )
    if dt is not None:
        setup_kwargs["dt"] = dt
    setup = build_base_setup(**setup_kwargs)
    grids = setup["grids"]
    time_cfg = setup["time_cfg"]
    base_kwargs = setup["base_kwargs"]
    y_grid = grids["y_grid"]
    dy = float(y_grid[1] - y_grid[0])
    dt_solver = float(setup["dt"])

    sim_params: list[dict] = []
    records: list[dict] = []

    for r in range(repeats):
        rng = np.random.default_rng(rng_seed + _RNG_BASE + r)
        rng_profile = np.random.default_rng(rng_seed + _RNG_PROFILE_OFFSET + r)

        if is_time:
            # One in-distribution sim per repeat, solved to the extended horizon.
            tcfg = dict(time_cfg)
            tcfg["num_sims"] = 1
            tcfg["lhs_seed"] = rng_seed + r
            params = spec.sample_sim_params(
                rng=rng, rng_profile=rng_profile, grids=grids, time_cfg=tcfg,
            )[0]
            lat_hash = stable_hash(params)
            sim_params.append(params)
            records.append(_make_record(
                sim_id=len(sim_params) - 1, axis=axis, value=None, repeat=r,
                latents_hash=lat_hash, realized={},
            ))
            if verbose:
                print(f"[time] repeat {r}: 1 sim -> t_final={t_final}", flush=True)
            continue

        latents = spec.draw_latents(rng, rng_profile, grids, time_cfg, axis)
        lat_hash = stable_hash(latents)
        for value in sweep:
            params = spec.apply_ood_value(latents, axis, value)
            realized = params.pop("_ood_realized", {})
            accounting = _resolution_accounting(spec, axis, params, dy, dt_solver)
            sim_params.append(params)
            records.append(_make_record(
                sim_id=len(sim_params) - 1, axis=axis, value=value, repeat=r,
                latents_hash=lat_hash, realized=realized, accounting=accounting,
            ))
        if verbose:
            print(
                f"[{axis.name}] repeat {r}: {len(sweep)} sims, "
                f"latents_hash={lat_hash}", flush=True,
            )

    spec.validate_schema(np.array(sim_params, dtype=object), np.arange(len(sim_params)))

    trajectories, t, x, y = run_solves(
        spec, sim_params, base_kwargs, save_stride,
        setup["Nt_saved"], setup["Nx"], setup["Ny"], verbose=verbose,
    )

    save_path = save_dataset(
        save_dir, x, y, t, save_stride, setup["dt"], setup["t_ramp"],
        trajectories, sim_params,
    )

    _write_sidecar(save_path, records)
    _write_manifest(
        save_path, benchmark=benchmark, representation=representation,
        axis=axis, sweep=sweep, repeats=repeats, t_final=t_final,
        time_norm_horizon=time_norm_horizon, rng_seed=rng_seed,
        setup=setup, save_stride=save_stride, suite_yaml_path=suite_yaml_path,
    )
    return save_path


# Per-feature-type adequacy criteria for the swept feature's effective
# discretization, keyed by ``OODAxis.resolution_kind``. The tuple is
# ``(domain, criterion_name, ok_threshold, under_threshold)``: ``domain`` picks
# cells (length / dy) or steps (timescale / dt); a feature is "ok" at or above
# ``ok_threshold``, else "watch", else "under_resolved" below ``under_threshold``.
# Spatial features use ``under_threshold = None`` so they cap at "watch" and are
# never auto-rejected on the cell count alone (the convergence pass decides);
# the temporally tight features can fall to "under_resolved".
_RES_CRITERIA: dict[str, tuple[str, str, float, float | None]] = {
    "spatial_patch_edge": ("cells", "cells_per_patch_width", 4.0, None),
    "spatial_gaussian_sigma": ("cells", "cells_per_sigma", 4.0, None),
    "spatial_triangle": ("cells", "cells_per_half_width", 4.0, None),
    "temporal_period": ("steps", "steps_per_period", 10.0, 5.0),
    "temporal_timescale": ("steps", "steps_per_timescale", 8.0, 4.0),
    "temporal_pulse": ("steps", "steps_per_min_pulse", 8.0, 4.0),
}

_EMPTY_ACCOUNTING = {
    "eff_cells_per_scale": None,
    "eff_steps_per_scale": None,
    "resolution_flag": None,
    "resolution_criterion": None,
}


def _resolution_accounting(spec, axis, params, dy: float, dt: float) -> dict:
    """Effective cells/steps for the swept feature plus a per-type adequacy flag.

    Reads the realized physical scale from ``spec.resolution_scale`` (units set
    by ``axis.resolution_kind``) and divides by ``dy`` (spatial) or ``dt``
    (temporal). Axes with no resolution stake (``resolution_kind == "none"`` or
    an unmapped kind) or a ``None`` scale record empty accounting.
    """
    crit = _RES_CRITERIA.get(axis.resolution_kind)
    if crit is None:
        return dict(_EMPTY_ACCOUNTING)
    scale = spec.resolution_scale(axis, params)
    if scale is None:
        return dict(_EMPTY_ACCOUNTING)
    domain, crit_name, ok_thr, under_thr = crit
    eff = float(scale) / (dy if domain == "cells" else dt)
    if eff >= ok_thr:
        flag = "ok"
    elif under_thr is None or eff >= under_thr:
        flag = "watch"
    else:
        flag = "under_resolved"
    return {
        "eff_cells_per_scale": eff if domain == "cells" else None,
        "eff_steps_per_scale": eff if domain == "steps" else None,
        "resolution_flag": flag,
        "resolution_criterion": crit_name,
    }


def _make_record(
    sim_id, axis, value, repeat, latents_hash, realized, accounting=None,
) -> dict:
    is_time = axis.kind == "evaluation_parameter"
    acc = accounting if accounting is not None else dict(_EMPTY_ACCOUNTING)
    rec = {
        "sim_id": int(sim_id),
        "ood_axis": axis.name,
        "ood_value": None if is_time else _value_to_json(value),
        "ood_repeat": int(repeat),
        "latents_hash": latents_hash,
        "distribution_class": None if is_time else classify_value(axis, value),
        "compound": bool(axis.compound),
        "pinned_family": axis.pinned_family,
        "dataset_t_final": float(axis.dataset_t_final),
        "time_norm_horizon": float(axis.time_norm_horizon),
        "eff_cells_per_scale": acc["eff_cells_per_scale"],
        "eff_steps_per_scale": acc["eff_steps_per_scale"],
        "resolution_flag": acc["resolution_flag"],
        "resolution_criterion": acc["resolution_criterion"],
    }
    for k, v in realized.items():
        rec[f"realized_{k}"] = _value_to_json(v)
    return rec


def _write_sidecar(save_path: Path, records: list[dict]) -> None:
    with open(save_path / "ood_metadata.jsonl", "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _write_manifest(
    save_path, benchmark, representation, axis, sweep, repeats, t_final,
    time_norm_horizon, rng_seed, setup, save_stride, suite_yaml_path,
) -> None:
    manifest = {
        "benchmark": benchmark,
        "representation": representation,
        "axis": axis.name,
        "axis_kind": axis.kind,
        "values": [_value_to_json(v) for v in sweep],
        "repeats": int(repeats),
        "dataset_t_final": float(t_final),
        "time_norm_horizon": float(time_norm_horizon),
        "rng_seed": int(rng_seed),
        "git_commit": _git_commit(),
        "solver_grid": {
            "Nx": int(setup["Nx"]),
            "Ny": int(setup["Ny"]),
            "dt": float(setup["dt"]),
        },
        "save_stride": int(save_stride),
        "suite_yaml_path": suite_yaml_path,
    }
    with open(save_path / "ood_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a CRN OOD dataset.")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--representation", default="temporal_encoder")
    parser.add_argument("--axis", required=True, help="OOD axis name (or 'time').")
    parser.add_argument(
        "--values", default=None,
        help="Comma list of swept values (ID reference first). Floats, or "
             "'temporal+spatial' for family_transfer, or target times for "
             "the time axis. Omit to use the axis's declared sweep.",
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--dataset-t-final", type=float, default=None)
    parser.add_argument("--time-norm-horizon", type=float, default=0.30)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--save-stride", type=int, default=2)
    parser.add_argument("--nx", type=int, default=100)
    parser.add_argument("--ny", type=int, default=100)
    parser.add_argument("--dt", type=float, default=None,
                        help="Solver step (default 0.005). Refine with a doubled "
                             "--save-stride to hold saved times fixed.")
    parser.add_argument("--ramp-seconds", type=float, default=None)
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument("--suite-yaml-path", type=str, default=None)
    args = parser.parse_args(argv)

    raw = [v.strip() for v in args.values.split(",")] if args.values else []
    values = _parse_values(args.axis, raw) if raw else None

    generate_ood_dataset(
        benchmark=args.benchmark,
        axis_name=args.axis,
        save_dir=args.save_dir,
        representation=args.representation,
        values=values,
        repeats=args.repeats,
        dataset_t_final=args.dataset_t_final,
        time_norm_horizon=args.time_norm_horizon,
        save_stride=args.save_stride,
        nx=args.nx,
        ny=args.ny,
        dt=args.dt,
        ramp_seconds=args.ramp_seconds,
        rng_seed=args.rng_seed,
        suite_yaml_path=args.suite_yaml_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
