"""Autoregressive partition-size visualization.

The rollout experiment evaluates a single trained checkpoint under autoregressive
inference: the interval ``[t_s, t_j]`` is split into ``num_substeps`` equal
sub-intervals and the model is applied recursively, each prediction seeding the
next (see ``src/operators/rollout.py`` and ``scripts/run_eval.py
--rollout-num-substeps``). ``num_substeps == 1`` is the one-shot baseline.

``plot_rollout_partition_error`` draws Rel. L2 (%) vs. partition size for three
metrics (global / interface-band / boundary-band), each with:

  * the test error (mean +/- std across seeds) as solid line + shaded band,
  * a dotted horizontal line at the one-shot baseline (``num_substeps == 1``)
    error, so the gain or loss from autoregression is visually obvious.

The figure answers whether error *increases*, *decreases*, or follows a
*U-shaped* tradeoff as the partition is refined: shorter subintervals are each an
easier prediction, but recursive application accumulates error. It does not
presuppose monotonic degradation.

Inputs are the per-evaluation JSON reports ``scripts/run_eval.py`` already writes
into a config directory (one per ``--rollout-num-substeps`` / ``--report-name``),
each carrying its own ``rollout_num_substeps`` and the test metrics:
  * ``<run_root>/directeval.json``    (num_substeps = 1, baseline)
  * ``<run_root>/rollouteval*.json``  (num_substeps > 1)

Run directly:
    python -m visual.rollout_plots --run-root <run_root>
or via the plot CLI:
    python -m visual.cli --group rollout --rollout-run-root <run_root> --out visual/
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from visual._common import PLOT_STYLE, _save_figure

# (panel key, per-seed report field, panel title)
_METRICS = [
    ("global", "test_rel_l2_norm", "Global"),
    ("interface", "test_iface_rel_l2_norm", "Interface band"),
    ("boundary", "test_boundary_rel_l2_norm", "Boundary band"),
]

_REQUIRED_FIELDS = [field for _key, field, _title in _METRICS]

_FNO_COLOR = "#1f6feb"
_REF_COLOR = "#555555"


def _parse_timestamp(value) -> float:
    """Return a sortable epoch from an ISO timestamp string, or -inf if absent."""
    if not isinstance(value, str):
        return float("-inf")
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return float("-inf")


def _report_priority(path: Path, data: dict) -> tuple[float, float, str]:
    """Tie-breaker key for duplicate partition sizes: timestamp, mtime, filename."""
    ts = _parse_timestamp(data.get("timestamp"))
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = float("-inf")
    return (ts, mtime, path.name)


def _substeps_of(data: dict) -> int | None:
    """Resolve the partition size, trusting ``summary`` over ``per_seed``."""
    summary = data.get("summary", {}) or {}
    per_seed = data.get("per_seed", []) or []
    n = summary.get("rollout_num_substeps")
    if n is None and per_seed:
        n = per_seed[0].get("rollout_num_substeps")
    if n is None:
        return None
    return int(n)


def _load_rollout_reports(
    run_root: Path,
    report_glob: str = "*eval*.json",
    allow_missing: bool = False,
) -> dict[int, dict]:
    """Load per-partition eval reports keyed by ``num_substeps``.

    Returns ``{n: {"num_sims": int, "seed_signature": tuple[int, ...],
    "<field>_mean": float, "<field>_std": float, ...}}`` with population mean/std
    across seeds for every metric in ``_METRICS``. Applies the robustness rules:
    trust ``summary`` for the partition size, skip malformed/incomparable reports,
    and de-duplicate same-partition reports by recency.
    """
    chosen: dict[int, tuple[tuple[float, float, str], Path, dict]] = {}
    for path in sorted(run_root.glob(report_glob)):
        try:
            with path.open("r") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Warning: {path.name} is not readable JSON ({exc}); skipping.")
            continue

        per_seed = data.get("per_seed", []) or []
        if not per_seed:
            print(f"Warning: {path.name} has no per_seed entries; skipping.")
            continue

        n = _substeps_of(data)
        if n is None:
            print(f"Warning: {path.name} has no rollout_num_substeps; skipping.")
            continue

        summary_n = (data.get("summary", {}) or {}).get("rollout_num_substeps")
        seed_ns = {int(s["rollout_num_substeps"]) for s in per_seed
                   if "rollout_num_substeps" in s}
        if summary_n is not None and seed_ns and seed_ns != {int(summary_n)}:
            print(f"Warning: {path.name} per_seed substeps {sorted(seed_ns)} disagree "
                  f"with summary {summary_n}; using summary.")

        missing = [field for field in _REQUIRED_FIELDS
                   if not any(field in s for s in per_seed)]
        if missing and not allow_missing:
            raise ValueError(
                f"{path.name} is missing required metric(s) {missing}. Pass "
                f"allow_missing=True (CLI --rollout-allow-missing) to plot NaN instead."
            )
        if missing:
            print(f"Warning: {path.name} missing {missing}; those points will be NaN.")

        priority = _report_priority(path, data)
        if n in chosen:
            kept_priority, kept_path, _ = chosen[n]
            if priority <= kept_priority:
                print(f"Warning: duplicate num_substeps={n}; keeping "
                      f"{kept_path.name}, dropping {path.name}.")
                continue
            print(f"Warning: duplicate num_substeps={n}; keeping {path.name}, "
                  f"dropping {kept_path.name}.")
        chosen[n] = (priority, path, data)

    out: dict[int, dict] = {}
    for n, (_priority, _path, data) in chosen.items():
        per_seed = data["per_seed"]
        signature = tuple(sorted(int(s["seed"]) for s in per_seed if "seed" in s))
        entry: dict = {
            "num_sims": int(per_seed[0].get("num_sims", -1)),
            "seed_signature": signature,
        }
        for _key, field, _title in _METRICS:
            vals = np.array([float(s[field]) for s in per_seed if field in s], dtype=float)
            if vals.size == 0:
                entry[f"{field}_mean"] = np.nan
                entry[f"{field}_std"] = np.nan
            else:
                entry[f"{field}_mean"] = float(vals.mean())
                entry[f"{field}_std"] = float(vals.std())  # population std, matches resinv
        out[n] = entry
    return out


def plot_rollout_partition_error(
    run_root: str | Path,
    report_glob: str = "*eval*.json",
    allow_missing: bool = False,
    save_path: str | Path | None = None,
):
    """Plot test error vs. autoregressive partition size with the one-shot baseline.

    Reads every ``report_glob`` report under ``run_root``, places each at its
    ``num_substeps`` on the x-axis, and overlays a dotted reference at the
    one-shot (``num_substeps == 1``) error. The trend may rise, fall, or be
    U-shaped; the figure does not assume monotonic degradation.
    """
    run_root = Path(run_root)
    reports = _load_rollout_reports(run_root, report_glob, allow_missing=allow_missing)

    if not reports:
        print(f"No {report_glob} reports found under {run_root} — skipping "
              f"rollout_partition_error.")
        return

    partitions = sorted(reports)
    x = np.array(partitions, dtype=float)

    # The comparison is only valid when every report evaluated the same test split:
    # same seed set and same num_sims so the per-seed pairs align across partitions.
    num_sims = {n: reports[n]["num_sims"] for n in partitions}
    unique_ns = {v for v in num_sims.values() if v >= 0}
    signatures = {reports[n]["seed_signature"] for n in partitions}
    num_sims_ok = len(unique_ns) <= 1
    seeds_ok = len(signatures) <= 1
    if not num_sims_ok:
        print(f"WARNING: num_sims differs across reports {num_sims} — test splits are "
              f"NOT aligned; the comparison is invalid.", file=sys.stderr)
    if not seeds_ok:
        print(f"WARNING: seed sets differ across reports {sorted(signatures)} — the "
              f"comparison may be unaligned.", file=sys.stderr)

    has_baseline = 1 in reports

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True)

        for ax, (_key, field, title) in zip(axes, _METRICS):
            mean = np.array([reports[n][f"{field}_mean"] for n in partitions])
            std = np.array([reports[n][f"{field}_std"] for n in partitions])

            ax.plot(x, mean, marker="o", color=_FNO_COLOR, linewidth=1.8,
                    zorder=4, label="test error (mean over seeds)")
            if np.any(std > 0):
                ax.fill_between(x, mean - std, mean + std, color=_FNO_COLOR,
                                alpha=0.18, linewidth=0, zorder=2, label="+/- 1 std")

            if has_baseline:
                ref_val = reports[1][f"{field}_mean"]
                if np.isfinite(ref_val):
                    ax.axhline(ref_val, color=_REF_COLOR, linestyle=":", linewidth=1.3,
                               zorder=1, label="one-shot baseline (substeps=1)")

            ax.set_title(title)
            ax.set_xlabel("Autoregressive partition size (num_substeps)")
            ax.set_xticks(partitions)
            ax.set_ylim(bottom=0.0)
            ax.grid(True)

        if not has_baseline:
            print("Note: no num_substeps=1 report found; baseline reference omitted.")

        axes[0].set_ylabel("Rel. L2 (%)")
        axes[0].legend(loc="best")

        suptitle = "Autoregressive partition size — test error vs. num_substeps"
        warnings = []
        if not num_sims_ok:
            warnings.append("num_sims mismatch")
        if not seeds_ok:
            warnings.append("seed-set mismatch")
        if warnings:
            suptitle += f"  [WARNING: {', '.join(warnings)} — splits not aligned]"
        fig.suptitle(suptitle, fontsize=13)

        default = run_root / "rollout_partition_error.png"
        _save_figure(fig, save_path if save_path is not None else default,
                     "rollout", "rollout_partition_error")


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot test error vs. autoregressive partition size (num_substeps).",
    )
    parser.add_argument("--run-root", type=Path, required=True,
                        help="Config directory containing *eval.json reports "
                             "(written by scripts/run_eval.py --report-name).")
    parser.add_argument("--report-glob", type=str, default="*eval*.json",
                        help="Glob for the per-partition reports under --run-root.")
    parser.add_argument("--allow-missing", action="store_true",
                        help="Plot NaN for reports missing a required metric instead "
                             "of failing.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output PNG path (default: <run-root>/rollout_partition_error.png).")
    args = parser.parse_args(argv)

    plot_rollout_partition_error(
        run_root=args.run_root,
        report_glob=args.report_glob,
        allow_missing=args.allow_missing,
        save_path=args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
