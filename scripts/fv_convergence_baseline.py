"""FV ground-truth self-convergence baseline (read-only; no model).

The FNO learned the 100x100 FV operator (including its discretization error) but
the cross-resolution eval scores it against 150/200/256 FV -- a different, more
accurate target. Part of any resolution trend in the FNO metrics is therefore
just the FV reference moving, not the network. This script quantifies that drift.

For the same test split and snapshot set the evaluator uses, it interpolates each
coarse-grid FV field onto the finest (reference) grid and reports

    ||FV_N - FV_ref|| / ||FV_ref||

globally and restricted to the interface band and the boundary band (the same
coordinate-based masks eval uses). Overlay the resulting curve on the FNO's
`test_rel_l2_norm` vs. resolution: the residual above this curve is the network's
own resolution behavior; what tracks the curve is the reference drifting.

Usage
-----
    python scripts/fv_convergence_baseline.py --resinv-root data/resinv
    # explicit reference grid + custom snapshot count:
    python scripts/fv_convergence_baseline.py --resinv-root data/resinv \
        --ref-res 256 --n-snapshots-test 40

Expects `--resinv-root` to contain per-resolution subdirs named `r<N>` (e.g.
`r100`, `r150`, `r200`, `r256`), each holding `trajectories.npy`, `x_grid.npy`,
`y_grid.npy`, `t_grid.npy` as written by `data/generate_dataset.py`.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scipy.interpolate import RegularGridInterpolator

from data.dataset import split_sim_ids


def discover_resolution_dirs(resinv_root: Path) -> dict[int, Path]:
    """Map resolution N -> directory for every `r<N>` subdir under resinv_root."""
    res_dirs: dict[int, Path] = {}
    for d in sorted(resinv_root.glob("r*")):
        if not d.is_dir():
            continue
        m = re.fullmatch(r"r(\d+)", d.name)
        if m is None:
            continue
        res_dirs[int(m.group(1))] = d
    return res_dirs


def load_grids(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_grid = np.load(data_dir / "x_grid.npy")
    y_grid = np.load(data_dir / "y_grid.npy")
    t_grid = np.load(data_dir / "t_grid.npy")
    return x_grid, y_grid, t_grid


def build_interface_band_np(x_grid: np.ndarray, y_grid: np.ndarray,
                            interface_x: float, half_width: float) -> np.ndarray:
    """(Nx, Ny) bool: True on full columns within half_width of interface_x."""
    Ny = len(y_grid)
    x_mask = np.abs(x_grid - interface_x) <= half_width  # (Nx,)
    return np.broadcast_to(x_mask[:, None], (len(x_grid), Ny)).copy()


def build_boundary_band_np(x_grid: np.ndarray, y_grid: np.ndarray,
                           width: float) -> np.ndarray:
    """(Nx, Ny) bool: True within `width` of any domain edge (coordinate-based)."""
    a, b = float(x_grid[0]), float(x_grid[-1])
    c, d = float(y_grid[0]), float(y_grid[-1])
    x_edge = (np.abs(x_grid - a) <= width) | (np.abs(x_grid - b) <= width)
    y_edge = (np.abs(y_grid - c) <= width) | (np.abs(y_grid - d) <= width)
    return x_edge[:, None] | y_edge[None, :]


def snapshot_indices(Nt: int, n_snapshots_test: int) -> np.ndarray:
    """Mirror Dataset.t_indices: rounded linspace over [0, Nt-1] (deterministic)."""
    if n_snapshots_test is not None and n_snapshots_test < Nt:
        return np.round(np.linspace(0, Nt - 1, n_snapshots_test)).astype(int)
    return np.arange(Nt)


def target_indices(t_indices: np.ndarray) -> np.ndarray:
    """Snapshot indices that appear as the *target* j of some (s, j) pair.

    Eval enumerates all pairs s<j over t_indices; the field being predicted is the
    target state. The union of targets is every t_index except the first.
    """
    return t_indices[1:]


def interpolate_to_ref(field: np.ndarray, x_src: np.ndarray, y_src: np.ndarray,
                       ref_points: np.ndarray, ref_shape: tuple[int, int]) -> np.ndarray:
    """Bilinearly interpolate a (Nx_src, Ny_src) field onto the reference grid.

    Domains match (unit square, shared endpoints), so no extrapolation occurs;
    fill_value=None guards the boundary float round-off case.
    """
    interp = RegularGridInterpolator(
        (x_src, y_src), field, method="linear", bounds_error=False, fill_value=None,
    )
    return interp(ref_points).reshape(ref_shape)


def compute_drift(res_dir: Path, ref_dir: Path,
                  x_ref: np.ndarray, y_ref: np.ndarray, ref_points: np.ndarray,
                  test_ids: np.ndarray, t_idx_target: np.ndarray,
                  iface_mask: np.ndarray, bnd_mask: np.ndarray) -> dict:
    """Aggregate ||FV_N - FV_ref|| / ||FV_ref|| (global / interface / boundary)."""
    ref_shape = (len(x_ref), len(y_ref))
    traj_n = np.load(res_dir / "trajectories.npy", mmap_mode="r")
    traj_ref = np.load(ref_dir / "trajectories.npy", mmap_mode="r")
    x_n = np.load(res_dir / "x_grid.npy")
    y_n = np.load(res_dir / "y_grid.npy")

    is_ref = res_dir.resolve() == ref_dir.resolve()

    sq_diff = sq_ref = 0.0
    sq_diff_if = sq_ref_if = 0.0
    sq_diff_bd = sq_ref_bd = 0.0

    for sim_id in test_ids:
        for j in t_idx_target:
            ref_field = np.asarray(traj_ref[int(sim_id), int(j)], dtype=np.float64)
            if is_ref:
                src_on_ref = ref_field
            else:
                coarse_field = np.asarray(traj_n[int(sim_id), int(j)], dtype=np.float64)
                src_on_ref = interpolate_to_ref(coarse_field, x_n, y_n, ref_points, ref_shape)

            diff = src_on_ref - ref_field
            sq_diff += float(np.sum(diff ** 2))
            sq_ref += float(np.sum(ref_field ** 2))

            sq_diff_if += float(np.sum(diff[iface_mask] ** 2))
            sq_ref_if += float(np.sum(ref_field[iface_mask] ** 2))

            sq_diff_bd += float(np.sum(diff[bnd_mask] ** 2))
            sq_ref_bd += float(np.sum(ref_field[bnd_mask] ** 2))

    def rel(num: float, den: float) -> float:
        return float(np.sqrt(num / den) * 100.0) if den > 0 else float("nan")

    return {
        "global_rel_l2_pct": rel(sq_diff, sq_ref),
        "interface_rel_l2_pct": rel(sq_diff_if, sq_ref_if),
        "boundary_rel_l2_pct": rel(sq_diff_bd, sq_ref_bd),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="FV ground-truth self-convergence baseline for the resolution sweep.",
    )
    parser.add_argument("--resinv-root", type=Path, default=Path("data/resinv"),
                        help="Directory containing per-resolution r<N>/ subdirs.")
    parser.add_argument("--ref-res", type=int, default=None,
                        help="Reference resolution N (default: the largest found).")
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--n-snapshots-test", type=int, default=40,
                        help="Must match build_test_loader's n_snapshots_test.")
    parser.add_argument("--interface-x", type=float, default=0.5)
    parser.add_argument("--interface-half-width", type=float, default=0.05)
    parser.add_argument("--boundary-width", type=float, default=0.05)
    parser.add_argument("--out-json", type=Path, default=None,
                        help="Output JSON path (default: <resinv-root>/fv_drift_baseline.json).")
    parser.add_argument("--out-csv", type=Path, default=None,
                        help="Output CSV path (default: <resinv-root>/fv_drift_baseline.csv).")
    args = parser.parse_args(argv)

    resinv_root = args.resinv_root.expanduser()
    if not resinv_root.is_dir():
        print(f"error: resinv-root does not exist or is not a directory: {resinv_root}",
              file=sys.stderr)
        return 1

    res_dirs = discover_resolution_dirs(resinv_root)
    if not res_dirs:
        print(f"error: no r<N>/ subdirs found under {resinv_root}", file=sys.stderr)
        return 1

    resolutions = sorted(res_dirs)
    ref_res = args.ref_res if args.ref_res is not None else max(resolutions)
    if ref_res not in res_dirs:
        print(f"error: reference resolution r{ref_res} not found under {resinv_root}",
              file=sys.stderr)
        return 1
    ref_dir = res_dirs[ref_res]

    # Reference grid + the snapshot set / test split that eval will use. Every
    # resolution must share num_sims so split_sim_ids(seed) yields the same
    # test_ids and the deterministic t_indices line up (guard below).
    x_ref, y_ref, t_ref = load_grids(ref_dir)
    Nt_ref = len(t_ref)

    num_sims_ref = int(np.load(ref_dir / "trajectories.npy", mmap_mode="r").shape[0])
    Nt_by_res: dict[int, int] = {}
    for N, d in res_dirs.items():
        tr = np.load(d / "trajectories.npy", mmap_mode="r")
        n_sims = int(tr.shape[0])
        if n_sims != num_sims_ref:
            print(
                f"error: num_sims mismatch -- r{N} has {n_sims}, r{ref_res} has "
                f"{num_sims_ref}. Regenerate the sweep with identical --num-sims so "
                f"the test split aligns across resolutions.",
                file=sys.stderr,
            )
            return 1
        Nt_by_res[N] = int(tr.shape[1])
        if Nt_by_res[N] != Nt_ref:
            print(
                f"error: Nt mismatch -- r{N} has {Nt_by_res[N]} snapshots, r{ref_res} "
                f"has {Nt_ref}. The time grid must match across resolutions.",
                file=sys.stderr,
            )
            return 1

    _, _, test_ids = split_sim_ids(
        num_sims=num_sims_ref, train_frac=args.train_frac,
        val_frac=args.val_frac, seed=args.split_seed,
    )

    t_indices = snapshot_indices(Nt_ref, args.n_snapshots_test)
    t_idx_target = target_indices(t_indices)

    iface_mask = build_interface_band_np(
        x_ref, y_ref, args.interface_x, args.interface_half_width)
    bnd_mask = build_boundary_band_np(x_ref, y_ref, args.boundary_width)

    Xr, Yr = np.meshgrid(x_ref, y_ref, indexing="ij")
    ref_points = np.stack([Xr.ravel(), Yr.ravel()], axis=-1)

    print(f"Reference grid: r{ref_res}  ({len(x_ref)}x{len(y_ref)}), Nt={Nt_ref}")
    print(f"num_sims={num_sims_ref}  |test_ids|={len(test_ids)}  "
          f"target snapshots/sim={len(t_idx_target)}")
    print(f"Resolutions: {resolutions}")

    rows = []
    for N in resolutions:
        print(f"Computing drift for r{N} ...", flush=True)
        drift = compute_drift(
            res_dir=res_dirs[N], ref_dir=ref_dir,
            x_ref=x_ref, y_ref=y_ref, ref_points=ref_points,
            test_ids=test_ids, t_idx_target=t_idx_target,
            iface_mask=iface_mask, bnd_mask=bnd_mask,
        )
        row = {"resolution": N, "is_reference": (N == ref_res), **drift}
        rows.append(row)
        print(
            f"  r{N}: global={drift['global_rel_l2_pct']:.6f}%  "
            f"interface={drift['interface_rel_l2_pct']:.6f}%  "
            f"boundary={drift['boundary_rel_l2_pct']:.6f}%"
        )

    out_json = args.out_json or (resinv_root / "fv_drift_baseline.json")
    out_csv = args.out_csv or (resinv_root / "fv_drift_baseline.csv")

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "resinv_root": str(resinv_root),
        "ref_res": ref_res,
        "num_sims": num_sims_ref,
        "n_test_ids": int(len(test_ids)),
        "n_snapshots_test": int(args.n_snapshots_test),
        "n_target_snapshots": int(len(t_idx_target)),
        "split": {"train_frac": args.train_frac, "val_frac": args.val_frac,
                  "seed": args.split_seed},
        "masks": {"interface_x": args.interface_x,
                  "interface_half_width": args.interface_half_width,
                  "boundary_width": args.boundary_width},
        "per_resolution": rows,
    }
    with out_json.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved JSON -> {out_json}")

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["resolution", "is_reference", "global_rel_l2_pct",
                        "interface_rel_l2_pct", "boundary_rel_l2_pct"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved CSV  -> {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
