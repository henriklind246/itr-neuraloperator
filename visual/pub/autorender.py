"""Render a benchmark's case figure from the evaluation that produced it.

``scripts/run_eval.py --write-test-records`` leaves a ``test_records.csv`` beside
every seed checkpoint, which together with that checkpoint is exactly what the
per-benchmark case figure reads. This module closes the gap between the two, so
the figure is a product of the evaluation rather than a step someone has to
remember afterwards, and so it behaves identically whether the evaluation ran on
a cluster or on a laptop.

Provenance is still resolved and the usual sidecar is still written, so an
auto-rendered figure is traceable in the same way as one from
``python -m visual.pub``. Resolution is deliberately non-strict: a single-seed
evaluation does not meet the three-seed publication bar, and the right answer
there is a figure under ``degraded/`` that says so, not no figure at all. Only
the cross-benchmark figures are left out, because no single run can satisfy them.
"""

from __future__ import annotations

from pathlib import Path

from visual.pub.manifest import Manifest

# The case figure each benchmark's own evaluation can produce unaided.
CASE_FIGURES = {
    "forcing": "F09_forcing_cases",
    "source": "F11_source_cases",
    "source_itr_sin": "F13_source_itr_sin_cases",
    "interfaces": "F15_interfaces_profiles",
}

# Beside seed_report.json rather than in visual/pub_out, because the figure
# describes this evaluation: it belongs with the run, and writing into the repo
# would silently overwrite the manuscript's copy whenever anyone evaluated
# anything.
DEFAULT_SUBDIR = "figures"


def render_run(run_root: str | Path, *, records_name: str = "test_records.csv",
               out_dir: str | Path | None = None) -> list[Path]:
    """Render the case figure for the benchmark evaluated under ``run_root``.

    ``run_root`` is the config directory holding ``seed*/``, the same path
    ``scripts/run_eval.py`` takes. Returns the written paths; an empty list means
    the benchmark has no case figure.
    """
    root = Path(run_root).expanduser().resolve()
    manifest, benchmark = Manifest.for_run(root, records_name=records_name)
    key = CASE_FIGURES.get(benchmark)
    if key is None:
        return []

    from visual.pub.registry import render

    target = Path(out_dir) if out_dir is not None else root / DEFAULT_SUBDIR
    result = render(key, manifest=manifest, out_dir=target, strict=False,
                    force=True)
    return list(result.paths)


__all__ = ["CASE_FIGURES", "DEFAULT_SUBDIR", "render_run"]
