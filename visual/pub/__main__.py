"""``python -m visual.pub`` -- render, list, verify, and audit publication figures.

A separate entry point from ``visual/cli.py`` because that file's dispatch policy
is skip-on-missing, which this package forbids, and its registry has no notion of
required artifacts, seeds, or tiers.

Exit codes:

- ``0`` every requested figure rendered cleanly
- ``1`` a provenance requirement failed under ``--strict`` (the default)
- ``2`` at least one figure rendered degraded under ``--allow-missing``

``--list`` and ``--verify`` never import a figure module, so neither imports torch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from visual.pub.manifest import DEFAULT_MANIFEST, Manifest, ProvenanceError
from visual.pub.registry import (
    DEFAULT_OUT_DIR,
    FIGURES,
    check_registry_matches,
    figures_at_tier,
    get_figure,
    render,
)

EXIT_OK = 0
EXIT_PROVENANCE = 1
EXIT_DEGRADED = 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m visual.pub",
        description="Render publication figures from provenance-verified artifacts.",
    )
    p.add_argument("--figure", nargs="+", metavar="KEY",
                   help="figure keys to render")
    p.add_argument("--all", action="store_true",
                   help="render every figure at or below --tier")
    p.add_argument("--tier", type=int, default=2, choices=(1, 2, 3),
                   help="tier ceiling for --all (default: 2)")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                   help=f"manifest.yaml path (default: {DEFAULT_MANIFEST})")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR,
                   help=f"output directory (default: {DEFAULT_OUT_DIR})")

    strictness = p.add_mutually_exclusive_group()
    strictness.add_argument("--strict", dest="strict", action="store_true",
                            default=True,
                            help="fail on any unmet requirement (default)")
    strictness.add_argument("--allow-missing", dest="strict", action="store_false",
                            help="render degraded figures into out/degraded/")

    p.add_argument("--formats", default="png,pdf",
                   help="comma-separated output formats (default: png,pdf)")
    p.add_argument("--footer", dest="footer", action="store_true", default=False,
                   help="stamp the provenance line under the figure; off by "
                        "default, since the sidecar already carries it")
    p.add_argument("--force", action="store_true",
                   help="overwrite a render made at a different git commit")

    p.add_argument("--verify", action="store_true",
                   help="resolve every figure, render nothing, print the "
                        "satisfiability table")
    p.add_argument("--audit", type=Path, metavar="DIR",
                   help="re-hash the sources of rendered figures and report drift")
    p.add_argument("--list", dest="do_list", action="store_true",
                   help="list figure keys, tiers, and requirements")
    p.add_argument("--format", choices=("text", "json"), default="text",
                   help="output format for --list, --verify and --audit")
    return p


def _requirements(manifest_path: Path):
    """Load figures.yaml only. Used by --list, which must not touch manifest.yaml."""
    mf = Manifest.load(manifest_path)
    check_registry_matches(mf.requirements)
    return mf


def cmd_list(mf: Manifest, fmt: str) -> int:
    rows = []
    for key in sorted(FIGURES):
        spec = FIGURES[key]
        req = mf.requirements.figures[key]
        rows.append({
            "key": key,
            "tier": spec.tier,
            "title": spec.title,
            "width": spec.width,
            "metric_space": req.metric_space,
            "requires": list(req.requires),
            "params": spec.params,
        })
    if fmt == "json":
        print(json.dumps(rows, indent=2))
        return EXIT_OK
    width = max(len(r["key"]) for r in rows)
    for tier in (2, 1, 3):
        tier_rows = [r for r in rows if r["tier"] == tier]
        if not tier_rows:
            continue
        print(f"\n--- tier {tier} " + "-" * 60)
        for r in tier_rows:
            requires = ", ".join(r["requires"]) or "(none)"
            print(f"  {r['key']:<{width}}  {r['metric_space']:<10}  {requires}")
            print(f"  {'':<{width}}  {r['title']}")
    return EXIT_OK


def cmd_verify(mf: Manifest, fmt: str) -> int:
    rows = mf.satisfiability()
    if fmt == "json":
        print(json.dumps(rows, indent=2))
        return EXIT_OK if all(r["satisfied"] for r in rows) else EXIT_DEGRADED

    if mf.path is None:
        print(f"No manifest at {DEFAULT_MANIFEST}; every requirement is unmet.\n")
    width = max(len(r["key"]) for r in rows)
    n_ok = 0
    for row in rows:
        mark = "OK  " if row["satisfied"] else "BLOCK"
        n_ok += bool(row["satisfied"])
        counts = (f"seeds={row['n_seeds']} sims={row['n_simulations']} "
                  f"pairs={row['n_pairs']}")
        print(f"{mark} {row['key']:<{width}}  {counts}")
        for blocker in row["blocking"]:
            print(f"       {blocker}")
    print(f"\n{n_ok}/{len(rows)} figures satisfiable.")
    return EXIT_OK if n_ok == len(rows) else EXIT_DEGRADED


def cmd_audit(out_dir: Path, fmt: str) -> int:
    from visual.pub.manifest import audit

    findings = audit(out_dir)
    if fmt == "json":
        print(json.dumps(findings, indent=2))
    else:
        if not findings:
            print(f"No rendered figures with sidecars under {out_dir}.")
        for f in findings:
            print(f"{f['status']:<8} {f['figure_key']:<28} {f['path']}")
    drift = [f for f in findings if f["status"] != "OK"]
    return EXIT_DEGRADED if drift else EXIT_OK


def _selected_keys(args) -> list[str]:
    if args.figure:
        for key in args.figure:
            get_figure(key)
        return list(args.figure)
    if args.all:
        return figures_at_tier(args.tier)
    return []


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.audit is not None:
        return cmd_audit(args.audit, args.format)

    mf = _requirements(args.manifest)

    if args.do_list:
        return cmd_list(mf, args.format)
    if args.verify:
        return cmd_verify(mf, args.format)

    keys = _selected_keys(args)
    if not keys:
        build_parser().print_help()
        return EXIT_OK

    formats = tuple(f.strip() for f in args.formats.split(",") if f.strip())
    any_degraded = False
    for key in keys:
        try:
            result = render(key, manifest=mf, out_dir=args.out, strict=args.strict,
                            formats=formats, footer=args.footer, force=args.force)
        except (ProvenanceError, NotImplementedError) as exc:
            # NotImplementedError is a registered-but-unbuilt figure raising from
            # blocked(). It is the same class of failure as a missing artifact --
            # the run does not exist yet -- so it gets the same exit code rather
            # than a traceback.
            print(f"FAILED {key}: {exc}", file=sys.stderr)
            return EXIT_PROVENANCE
        any_degraded |= result.degraded

    return EXIT_DEGRADED if any_degraded else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
