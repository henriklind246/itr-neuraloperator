"""Check whether F32's plotted source time has a typical error-vs-lead shape.

Run:
    python scripts/inspect_source_time_sweep.py --runs-root <dir>
    python scripts/inspect_source_time_sweep.py --manifest visual/pub/manifest.yaml

F32 plots global field error against lead time at one fixed source snapshot
``t_s*``. That snapshot is chosen as a fraction of the horizon, not because it
is representative, so the figure owes a check that its shape is not peculiar to
the source time it was measured at. This script is that check, over every
source time on the grid and all four benchmarks, and it writes nothing: the
report is stdout.

The comparison fits ``log10 RMSE(s, k) = mu + alpha_s + beta_k`` over the
source times that span a common lead window. Additive in log space is
multiplicative in kelvin, so the fit says every source time traces one shared
lead-time shape ``beta`` and differs only by a constant factor ``alpha``.
Whatever the fit cannot reach is the disagreement in shape, reported as
``interaction``. A source time whose curve merely sits higher or lower than the
others is not atypical in shape, and the decomposition keeps that in ``alpha``
where it belongs.

Windows matter because a source time at index ``s`` only reaches lag ``30 - s``.
Ranking curves measured over unequal lead windows ranks window length, not
shape, so each window here retains only the source times that span all of it,
and the report shows several windows to expose the trade between window length
and how many curves it admits.

Reading the verdict: ``interaction`` is descriptive and in percent of RMSE. The
published curve's rank among the others is the calibrated part -- a paired
simulation bootstrap gives its interval, which widens to the whole set exactly
when the curves are indistinguishable in shape.

Every pair is predicted directly from its source snapshot, so a rising curve is
lead-time-dependent degradation, not error accumulated over a rollout.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visual.pub import records, stats  # noqa: E402
from visual.pub.manifest import DEFAULT_MANIFEST, Manifest, ProvenanceError  # noqa: E402

FIGURE_KEY = "F32_global_field_error_fixed_source"

# Below this the additive fit already explains the curves and no source time
# could have been unrepresentative, so the rank carries nothing worth reading.
NEGLIGIBLE_INTERACTION_PCT = 1.0
# A rank interval covering this much of the compared set is not locating the
# published curve anywhere in particular.
UNINFORMATIVE_RANK_SPAN = 0.5


def build_parser():
    p = argparse.ArgumentParser(
        prog="python scripts/inspect_source_time_sweep.py",
        description="Is F32's plotted source time representative of the rest?")
    source = p.add_mutually_exclusive_group()
    source.add_argument("--runs-root", type=Path,
                        help="discover test_records.csv recursively under this directory")
    source.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                        help=f"manifest.yaml path (default: {DEFAULT_MANIFEST})")
    p.add_argument("--run", action="append", default=[], metavar="BENCHMARK=CONFIG_DIR",
                   help="select an experiment when discovery finds several (repeatable)")
    p.add_argument("--source-fraction", type=float, default=stats.DEFAULT_SOURCE_FRACTION,
                   help=f"source time F32 plots, as a fraction of the horizon "
                        f"(default: {stats.DEFAULT_SOURCE_FRACTION})")
    p.add_argument("--window-fraction", type=float, action="append", default=[],
                   metavar="F", help="lead window as a fraction of the longest lag "
                                     "(repeatable; default: "
                                     f"{', '.join(str(f) for f in stats.DEFAULT_SHAPE_WINDOW_FRACTIONS)})")
    p.add_argument("--n-boot", type=int, default=stats.DEFAULT_SHAPE_N_BOOT,
                   help=f"bootstrap replicates (default: {stats.DEFAULT_SHAPE_N_BOOT}; "
                        "0 skips the rank interval)")
    p.add_argument("--rng-seed", type=int, default=stats.DEFAULT_RNG_SEED,
                   help=f"bootstrap seed (default: {stats.DEFAULT_RNG_SEED})")
    return p


def load_manifest(args, parser):
    if args.runs_root is None:
        return Manifest.load(args.manifest)
    selections = {}
    for value in args.run:
        benchmark, separator, directory = value.partition("=")
        if not separator or not directory or benchmark in selections:
            parser.error("--run requires a unique BENCHMARK=CONFIG_DIR selection")
        selections[benchmark] = directory
    return Manifest.discover_global_field(args.runs_root, selections=selections)


def report(result):
    published = result["published"]
    print("\n=== 1. Protocol ===")
    print(f"  snapshot lags evaluated  1..{result['max_lag']}")
    print(f"  published source index   {published['source_index']} "
          f"(t_s = {published['source_time']:.4f}, "
          f"{published['source_fraction']:g} of horizon {published['horizon']:.4f})")
    print(f"  its longest lag          {published['max_lead_lag']}")
    print(f"  lead windows compared    {result['windows']}")

    print("\n=== 2. Cohort ===")
    for benchmark, info in result["cohort"].items():
        note = (f"  ({info['n_dropped']} excluded for missing pairs: "
                f"{info['dropped_sim_ids']})" if info["n_dropped"] else "")
        print(f"  {benchmark:<16} {info['n_simulations']:>5} simulations"
              f"  x {len(result['seeds'][benchmark])} seed(s){note}")
    if result["unequal_seed_counts"]:
        print("  NOTE benchmarks carry unequal seed counts")

    print("\n=== 3. Shape decomposition ===")
    print("  interaction  RMSE spread around the shared lead-time shape")
    print("  level        largest RMSE ratio between source times (not shape)")
    print("  common       fraction of post-level lead variation that is the shared shape")
    header = (f"  {'benchmark':<16} {'K':>3} {'curves':>6} {'lead':>7} "
              f"{'interaction':>12} {'level':>8} {'common':>8}")
    print(header)
    for row in result["rows"]:
        print(f"  {row['benchmark']:<16} {row['window_lags']:>3} {row['n_curves']:>6} "
              f"{row['window_lead_time']:>7.4f} {row['interaction_pct']:>11.2f}% "
              f"{row['level_spread_pct']:>7.1f}% {row['common_shape_fraction']:>8.4f}")

    print("\n=== 4. Where the published curve sits ===")
    print(f"  {'benchmark':<16} {'K':>3} {'rank':>10} {'95% CI':>12} {'span':>7} "
          f"{'least typical':>14}")
    for row in result["rows"]:
        if not row["covers_published"]:
            print(f"  {row['benchmark']:<16} {row['window_lags']:>3} "
                  f"{'not spanned':>10}")
            continue
        rank = f"{row['published_rank']}/{row['n_curves']}"
        interval = ("n/a" if row["rank_ci_lower"] is None
                    else f"[{row['rank_ci_lower']}, {row['rank_ci_upper']}]")
        span = ("n/a" if row["rank_span_fraction"] is None
                else f"{row['rank_span_fraction']:.2f}")
        print(f"  {row['benchmark']:<16} {row['window_lags']:>3} {rank:>10} "
              f"{interval:>12} {span:>7} {row['least_typical_index']:>14}")

    print("\n=== 5. Verdict ===")
    atypical = []
    for benchmark in result["cohort"]:
        rows = [r for r in result["rows"] if r["benchmark"] == benchmark]
        material = [r for r in rows if r["interaction_pct"] >= NEGLIGIBLE_INTERACTION_PCT]
        if not material:
            worst = max(r["interaction_pct"] for r in rows)
            print(f"  {benchmark:<16} shapes agree (interaction <= {worst:.2f}%); "
                  "no source time is unrepresentative")
            continue
        located = [r for r in material if r["covers_published"]
                   and r["rank_span_fraction"] is not None
                   and r["rank_span_fraction"] <= UNINFORMATIVE_RANK_SPAN]
        if not located:
            worst = max(r["interaction_pct"] for r in material)
            print(f"  {benchmark:<16} shapes differ (interaction up to {worst:.2f}%) "
                  "but the published curve is not placed among them")
            continue
        outlying = [r for r in located
                    if r["published_rank"] > 0.75 * r["n_curves"]]
        verdict = "ATYPICAL" if outlying else "typical"
        ranks = ", ".join(f"K={r['window_lags']} rank {r['published_rank']}/{r['n_curves']}"
                          for r in located)
        print(f"  {benchmark:<16} {verdict}: {ranks}")
        atypical.extend(outlying)

    if atypical:
        print("\n  The plotted source time ranks among the least typical curves for "
              f"{sorted({r['benchmark'] for r in atypical})}.")
        print("  Report the sweep, or pick a source time nearer the centre of the set.")
    else:
        print("\n  Nothing contradicts F32's choice of source time.")
    for degradation in result["degradations"]:
        print(f"  DEGRADED {degradation}")
    return 2 if atypical else 0


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.run and args.runs_root is None:
        parser.error("--run requires --runs-root")

    try:
        manifest = load_manifest(args, parser)
        source = manifest.resolve(FIGURE_KEY, strict=False)
        frames, metadata = records.load_global_field_records(source)
        if not frames:
            raise ProvenanceError(
                "no global-field test records; nothing to compare source times over")
        result = stats.source_time_shape_typicality(
            frames, metadata=metadata,
            source_fraction=args.source_fraction,
            window_fractions=tuple(args.window_fraction)
            or stats.DEFAULT_SHAPE_WINDOW_FRACTIONS,
            n_boot=args.n_boot, rng_seed=args.rng_seed)
    except (ProvenanceError, records.SchemaError, FileNotFoundError) as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    if source.is_degraded:
        print(f"WARNING resolved {FIGURE_KEY} degraded: "
              f"{'; '.join(str(d) for d in source.degradations)}", file=sys.stderr)
    return report(result)


if __name__ == "__main__":
    raise SystemExit(main())
