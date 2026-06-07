"""Spatial resolution-invariance visualization.

The experiment evaluates a single 100x100-trained checkpoint against finer-grid
FV datasets (see `scripts/run_eval.py --data-dir` and
`scripts/fv_convergence_baseline.py`). The headline question is whether the
FNO's test error stays *flat* as the grid refines -- a flat curve is the
signature of discretization invariance.

`plot_resolution_invariance` draws Rel. L2 (%) vs. grid resolution N for three
metrics (global / interface-band / boundary-band), each with:

  * the FNO error (mean +/- std across seeds) as solid line + shaded band,
  * the FV ground-truth self-convergence drift as a dashed line -- the part of
    any upward trend that is just the *reference* moving, not the network,
  * a dotted horizontal line at the training-resolution (smallest N) FNO error,
    the "perfect invariance" reference. The gap between the solid curve and this
    line is the loss of invariance; the gap above the dashed curve is the
    network's own resolution behavior net of FV drift.

Inputs are the JSON artifacts the experiment already writes:
  * `<run_root>/seed_report_r<N>.json`  (one per resolution, from run_eval)
  * `<resinv_root>/fv_drift_baseline.json`  (optional, from the baseline script)

Run directly:
    python -m visual.resinv_plots --run-root <run_root> --resinv-root data/resinv
or via the plot CLI:
    python -m visual.cli --group resinv --resinv-run-root <run_root> \
        --resinv-root data/resinv --out visual/
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from visual._common import PLOT_STYLE, _save_figure

# (panel key, per-seed report field, FV-drift field, panel title)
_METRICS = [
    ("global", "test_rel_l2_norm", "global_rel_l2_pct", "Global"),
    ("interface", "test_iface_rel_l2_norm", "interface_rel_l2_pct", "Interface band"),
    ("boundary", "test_boundary_rel_l2_norm", "boundary_rel_l2_pct", "Boundary band"),
]

_FNO_COLOR = "#1f6feb"
_FV_COLOR = "#d1495b"
_REF_COLOR = "#555555"


def _load_seed_reports(run_root: Path,
                       report_glob: str = "seed_report_r*.json") -> dict[int, dict]:
    """Load per-resolution eval reports keyed by resolution N.

    Returns {N: {"num_sims": int, "<field>_mean": float, "<field>_std": float, ...}}
    with mean/std computed across seeds for every metric in `_METRICS`.
    """
    pat = re.compile(r"seed_report_r(\d+)\.json$")
    out: dict[int, dict] = {}
    for path in sorted(run_root.glob(report_glob)):
        m = pat.search(path.name)
        if m is None:
            continue
        N = int(m.group(1))
        with path.open("r") as f:
            data = json.load(f)
        per_seed = data.get("per_seed", [])
        if not per_seed:
            print(f"Warning: {path.name} has no per_seed entries, skipping.")
            continue

        entry: dict = {"num_sims": int(per_seed[0].get("num_sims", -1))}
        for _key, field, _fv, _title in _METRICS:
            vals = np.array([float(s[field]) for s in per_seed if field in s], dtype=float)
            if vals.size == 0:
                entry[f"{field}_mean"] = np.nan
                entry[f"{field}_std"] = np.nan
            else:
                entry[f"{field}_mean"] = float(vals.mean())
                entry[f"{field}_std"] = float(vals.std())  # population std, matches sweep_plots
        out[N] = entry
    return out


def _load_fv_drift(resinv_root: Path | None) -> dict[int, dict]:
    """Load FV self-convergence drift keyed by resolution N (empty if absent)."""
    if resinv_root is None:
        return {}
    path = Path(resinv_root) / "fv_drift_baseline.json"
    if not path.exists():
        print(f"Note: no FV-drift baseline at {path} (overlay omitted).")
        return {}
    with path.open("r") as f:
        data = json.load(f)
    return {int(r["resolution"]): r for r in data.get("per_resolution", [])}


def plot_resolution_invariance(
    run_root: str | Path,
    resinv_root: str | Path | None = None,
    report_glob: str = "seed_report_r*.json",
    save_path: str | Path | None = None,
):
    """Plot FNO test error vs. grid resolution with the FV-drift baseline overlaid.

    A flat FNO curve (tracking the dotted training-resolution reference) indicates
    discretization invariance. Any rise that follows the dashed FV-drift curve is
    the moving ground truth rather than the network.
    """
    run_root = Path(run_root)
    reports = _load_seed_reports(run_root, report_glob)

    if not reports:
        print(f"No seed_report_r*.json found under {run_root} — skipping "
              f"resolution_invariance.")
        return

    resolutions = sorted(reports)
    x = np.array(resolutions, dtype=float)

    # Guard (plan Change 5): the cross-resolution comparison is only valid when
    # every report used the same number of sims, so the seed=0 test split aligns.
    num_sims = {N: reports[N]["num_sims"] for N in resolutions}
    unique_ns = {n for n in num_sims.values() if n >= 0}
    num_sims_ok = len(unique_ns) <= 1
    if not num_sims_ok:
        print(f"WARNING: num_sims differs across reports {num_sims} — the test "
              f"splits are NOT aligned; the comparison is invalid. Regenerate the "
              f"sweep with identical --num-sims.")

    fv_drift = _load_fv_drift(resinv_root)
    train_res = resolutions[0]  # smallest grid == the training resolution

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True)

        for ax, (_key, field, fv_field, title) in zip(axes, _METRICS):
            mean = np.array([reports[N][f"{field}_mean"] for N in resolutions])
            std = np.array([reports[N][f"{field}_std"] for N in resolutions])

            # FNO error: mean line + +/- std band across seeds
            ax.plot(x, mean, marker="o", color=_FNO_COLOR, linewidth=1.8,
                    zorder=4, label="FNO (mean over seeds)")
            if np.any(std > 0):
                ax.fill_between(x, mean - std, mean + std, color=_FNO_COLOR,
                                alpha=0.18, linewidth=0, zorder=2, label="+/- 1 std")

            # Perfect-invariance reference: the training-resolution FNO error
            ref_val = reports[train_res][f"{field}_mean"]
            ax.axhline(ref_val, color=_REF_COLOR, linestyle=":", linewidth=1.3,
                       zorder=1, label=f"invariance ref (r{train_res})")

            # FV ground-truth self-convergence drift overlay
            if fv_drift:
                fx, fy = [], []
                for N in resolutions:
                    if N in fv_drift and fv_field in fv_drift[N]:
                        fx.append(float(N))
                        fy.append(float(fv_drift[N][fv_field]))
                if fx:
                    ax.plot(fx, fy, marker="s", color=_FV_COLOR, linestyle="--",
                            linewidth=1.6, zorder=3, label="FV drift (ref moves)")

            ax.set_title(title)
            ax.set_xlabel("Grid resolution N  (N x N)")
            ax.set_xticks(resolutions)
            ax.set_ylim(bottom=0.0)
            ax.grid(True)

        axes[0].set_ylabel("Rel. L2 (%)")
        axes[0].legend(loc="best")

        suptitle = "Spatial resolution invariance — test error vs. grid refinement"
        if not num_sims_ok:
            suptitle += "  [WARNING: num_sims mismatch — splits not aligned]"
        fig.suptitle(suptitle, fontsize=13)

        default = run_root / "resolution_invariance.png"
        _save_figure(fig, save_path if save_path is not None else default,
                     "resinv", "resolution_invariance")


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot spatial resolution-invariance (error vs. grid resolution).",
    )
    parser.add_argument("--run-root", type=Path, required=True,
                        help="Directory containing seed_report_r<N>.json files "
                             "(written by scripts/run_eval.py --report-name).")
    parser.add_argument("--resinv-root", type=Path, default=None,
                        help="Directory with fv_drift_baseline.json for the FV-drift "
                             "overlay (written by scripts/fv_convergence_baseline.py).")
    parser.add_argument("--report-glob", type=str, default="seed_report_r*.json",
                        help="Glob for the per-resolution reports under --run-root.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output PNG path (default: <run-root>/resolution_invariance.png).")
    args = parser.parse_args(argv)

    plot_resolution_invariance(
        run_root=args.run_root,
        resinv_root=args.resinv_root,
        report_glob=args.report_glob,
        save_path=args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
