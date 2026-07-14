"""Initial-boundary compatibility diagnostic for diffusion_forcing_single.

Reports, per IC family, how well the sampled t=0 initial condition satisfies the
benchmark's spatial boundary conditions, using the code's OWN sign convention:

  - adiabatic top/bottom walls (y=1, y=0):  d_y T0 ~ 0
  - inhomogeneous left wall (x=0):          -k d_x T0|_{x=0} ~ q_L(y, 0)

The left-wall convention comes straight from ``forcing_neumann_residual``
(src/physics/pde_residual.py): the FV solver injects ``q_L`` as the INWARD flux,
i.e. ``-k d_x T|_{x=0} = q_L``. ``q_L(y, 0)`` is evaluated from the ACTUAL ramped
forcing via ``reconstruct_qL(...).evaluate_points(y, 0.0)`` -- not assumed zero.
Because the startup ramp is a smoothstep with ``ramp_envelope(0)=0`` whenever
``t_ramp>0``, ``q_L(y,0)`` is expected to be identically 0; this script checks
that explicitly, so the left-wall condition then reduces to ``d_x T0|_{x=0} ~ 0``.

The IC builders taper every deviation to zero (value AND normal derivative) at all
four walls (``boundary_taper`` in src/physics/init_conditions.py), so all three
mismatches should be small and no family should stand out. This diagnostic is the
empirical check that the sampled fields actually honor that.

Usage:
  PYTHONPATH=. python scripts/diffusion_ic_boundary_compat.py \
      --data-dir data/diffusion_forcing_probe_cvit_best
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import load_ramp_seconds, load_solver_dt  # noqa: E402
from problems.diffusion_forcing import K_SLAB  # noqa: E402
from src.physics.boundary_forcing import (  # noqa: E402
    default_ramp_seconds,
    reconstruct_qL,
)

DATA_FILE_NAMES = {
    "trajectories": "trajectories.npy",
    "x_grid": "x_grid.npy",
    "y_grid": "y_grid.npy",
    "t_grid": "t_grid.npy",
    "sim_params": "sim_params.npy",
}


def boundary_normal_derivs(
    T0: np.ndarray, x_grid: np.ndarray, y_grid: np.ndarray
) -> dict[str, np.ndarray]:
    """One-sided finite-difference wall-normal derivatives of ``T0``.

    ``T0`` is ``(Nx, Ny)`` with axis 0 = x (x=0 at index 0, x=1 at index -1) and
    axis 1 = y (y=0 at index 0, y=1 at index -1), matching the saved trajectory
    layout. Returns the wall-normal derivative along each wall:
      - ``dTdx_left``  ``(Ny,)``  at x=0
      - ``dTdy_bottom`` ``(Nx,)`` at y=0
      - ``dTdy_top``    ``(Nx,)`` at y=1
    """
    x = np.asarray(x_grid, dtype=float).reshape(-1)
    y = np.asarray(y_grid, dtype=float).reshape(-1)
    T = np.asarray(T0, dtype=float)
    if T.shape != (x.shape[0], y.shape[0]):
        raise ValueError(
            f"T0 shape {T.shape} != (Nx, Ny) = ({x.shape[0]}, {y.shape[0]})."
        )
    dTdx_left = (T[1, :] - T[0, :]) / (x[1] - x[0])
    dTdy_bottom = (T[:, 1] - T[:, 0]) / (y[1] - y[0])
    dTdy_top = (T[:, -1] - T[:, -2]) / (y[-1] - y[-2])
    return {
        "dTdx_left": dTdx_left,
        "dTdy_bottom": dTdy_bottom,
        "dTdy_top": dTdy_top,
    }


def qL_at_t0(param: dict, y_grid: np.ndarray, t_ramp: float) -> np.ndarray:
    """Inward left-wall flux ``q_L(y, 0)`` for one sim; shape ``(Ny,)``.

    Uses the exact ramped forcing evaluator, so the ramp envelope is applied.
    """
    forcing = reconstruct_qL(
        param["temporal_family"], param["temporal_params"],
        param["spatial_family"], param["spatial_params"], t_ramp=float(t_ramp),
    )
    y = np.asarray(y_grid, dtype=float).reshape(-1)
    return np.asarray(
        forcing.evaluate_points(y, np.zeros_like(y)), dtype=float
    )


def ic_boundary_mismatch(
    T0: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    qL_t0: np.ndarray,
    k: float = K_SLAB,
) -> dict[str, float]:
    """Max-abs boundary-condition mismatches for one sim (physical Kelvin units).

    left   : max | -k d_x T0|_{x=0} - q_L(y, 0) |   (inhomogeneous Neumann)
    top    : max |  d_y T0|_{y=1} |                  (adiabatic)
    bottom : max |  d_y T0|_{y=0} |                  (adiabatic)
    qL_t0  : max |  q_L(y, 0) |                       (should be ~0 under the ramp)
    """
    d = boundary_normal_derivs(T0, x_grid, y_grid)
    left_residual = (-float(k) * d["dTdx_left"]) - np.asarray(qL_t0, dtype=float)
    return {
        "left": float(np.max(np.abs(left_residual))),
        "top": float(np.max(np.abs(d["dTdy_top"]))),
        "bottom": float(np.max(np.abs(d["dTdy_bottom"]))),
        "qL_t0": float(np.max(np.abs(qL_t0))),
    }


def _resolve_t_ramp(data_dir: Path) -> float:
    t_grid_path = data_dir / DATA_FILE_NAMES["t_grid"]
    ramp = load_ramp_seconds(t_grid_path)
    if ramp is not None:
        return float(ramp)
    dt = load_solver_dt(t_grid_path)
    if dt is not None:
        return float(default_ramp_seconds(dt))
    return 0.0


def run_diagnostic(data_dir: Path, k: float, limit: int | None) -> dict[str, dict]:
    trajectories = np.load(data_dir / DATA_FILE_NAMES["trajectories"])
    x_grid = np.load(data_dir / DATA_FILE_NAMES["x_grid"])
    y_grid = np.load(data_dir / DATA_FILE_NAMES["y_grid"])
    sim_params = np.load(data_dir / DATA_FILE_NAMES["sim_params"], allow_pickle=True)
    num_sims = int(trajectories.shape[0])
    if len(sim_params) != num_sims:
        raise ValueError(
            f"sim_params length {len(sim_params)} != num_sims {num_sims}."
        )
    if limit is not None:
        num_sims = min(num_sims, int(limit))

    t_ramp = _resolve_t_ramp(data_dir)

    per_family: dict[str, dict[str, list[float]]] = {}
    for i in range(num_sims):
        param = dict(sim_params[i])
        fam = str(param["ic_family"])
        T0 = trajectories[i, 0, :, :]
        qL0 = qL_at_t0(param, y_grid, t_ramp)
        m = ic_boundary_mismatch(T0, x_grid, y_grid, qL0, k=k)
        bucket = per_family.setdefault(
            fam, {"left": [], "top": [], "bottom": [], "qL_t0": []}
        )
        for key in ("left", "top", "bottom", "qL_t0"):
            bucket[key].append(m[key])

    summary: dict[str, dict] = {}
    for fam, bucket in sorted(per_family.items()):
        summary[fam] = {
            "count": len(bucket["left"]),
            "left_mean": float(np.mean(bucket["left"])),
            "left_max": float(np.max(bucket["left"])),
            "top_mean": float(np.mean(bucket["top"])),
            "top_max": float(np.max(bucket["top"])),
            "bottom_mean": float(np.mean(bucket["bottom"])),
            "bottom_max": float(np.max(bucket["bottom"])),
            "qL_t0_max": float(np.max(bucket["qL_t0"])),
        }
    return {"t_ramp": t_ramp, "num_sims": num_sims, "per_family": summary}


def _print_report(result: dict) -> None:
    t_ramp = result["t_ramp"]
    print(
        f"IC boundary-compatibility (t=0) | sims={result['num_sims']} "
        f"t_ramp={t_ramp:.6g} k={K_SLAB} | units: Kelvin per unit length"
    )
    if t_ramp > 0.0:
        print(
            "  ramp>0 => ramp_envelope(0)=0 => q_L(y,0)=0 expected; "
            "left check reduces to |k*d_x T0|_{x=0}| ~ 0"
        )
    header = (
        f"  {'family':<20} {'n':>5} "
        f"{'left_mean':>11} {'left_max':>11} "
        f"{'top_max':>11} {'bottom_max':>11} {'qL0_max':>11}"
    )
    print(header)
    for fam, s in result["per_family"].items():
        print(
            f"  {fam:<20} {s['count']:>5d} "
            f"{s['left_mean']:>11.3e} {s['left_max']:>11.3e} "
            f"{s['top_max']:>11.3e} {s['bottom_max']:>11.3e} {s['qL_t0_max']:>11.3e}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-IC-family initial-boundary compatibility diagnostic "
        "for diffusion_forcing_single (t=0 wall-normal mismatches).",
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory holding trajectories.npy, x_grid.npy, y_grid.npy, "
        "t_grid.npy, sim_params.npy (a diffusion_forcing_single dataset).",
    )
    parser.add_argument(
        "--k", type=float, default=float(K_SLAB),
        help=f"Slab conductivity for the left-wall flux (default K_SLAB={K_SLAB}).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Optional cap on the number of sims scanned.",
    )
    args = parser.parse_args()

    result = run_diagnostic(Path(args.data_dir), k=args.k, limit=args.limit)
    _print_report(result)


if __name__ == "__main__":
    main()
