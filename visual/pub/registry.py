"""The figure table and the single render entry point.

``FIGURES`` maps every publication figure key to the function that draws it.
The *requirement* side of each key lives in ``figures.yaml``, not here, so the
two cannot drift: :data:`FIGURES` and ``figures.yaml`` must declare exactly the
same key set, and a test asserts it in both directions.

:func:`render` is the only way a figure is produced. It resolves provenance
first and refuses to draw anything it cannot trace, which is the deliberate
opposite of ``visual/_common.py:_print_skip``: a figure never quietly fails to
appear. Under ``--allow-missing`` it still draws, but into ``degraded/`` with a
red border and a non-empty ``degradations`` list in the sidecar.

Figure modules are imported lazily so ``--list`` costs nothing and never pulls
in torch.
"""

from __future__ import annotations

import csv
import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from visual.pub.manifest import (
    DEFAULT_MANIFEST,
    FigureSource,
    Manifest,
    ProvenanceError,
    read_sidecar,
    write_sidecar,
)

DEFAULT_OUT_DIR = Path("visual/pub_out")
DEGRADED_SUBDIR = "degraded"

WIDTH_NAMES = ("one_col", "one_half_col", "two_col", "full_page")


@dataclass(frozen=True)
class FigureSpec:
    """How one publication figure is drawn and how big it is.

    ``requires`` and ``metric_space`` are *not* stored here; they are read from
    ``figures.yaml`` through :class:`~visual.pub.manifest.FigureRequirements` so
    the requirement declaration has exactly one home.
    """

    key: str
    title: str
    module: str
    func: str
    tier: int
    width: str = "two_col"
    params: dict = field(default_factory=dict)
    formats: tuple[str, ...] = ("png", "pdf")
    preserve_size: bool = False

    def __post_init__(self) -> None:
        if self.tier not in (1, 2, 3):
            raise ValueError(f"{self.key}: tier must be 1, 2 or 3, got {self.tier}")
        if self.width not in WIDTH_NAMES:
            raise ValueError(f"{self.key}: unknown width {self.width!r}")

    def load(self):
        """Import the drawing function. Deferred so ``--list`` stays cheap."""
        module = importlib.import_module(self.module)
        try:
            return getattr(module, self.func)
        except AttributeError:
            raise ImportError(
                f"{self.key}: {self.module} has no function {self.func!r}"
            ) from None


def _spec(key, title, module, func, tier, width="two_col", *,
          formats=("png", "pdf"), preserve_size=False, **params) -> FigureSpec:
    return FigureSpec(key=key, title=title, module=module, func=func, tier=tier,
                      width=width, params=params, formats=formats, preserve_size=preserve_size)


_SPECS: tuple[FigureSpec, ...] = (
    # ------------------------------------------------------ tier 2: unblocked
    _spec("F01_problem_schematic", "Governing problem and benchmark suite",
          "visual.pub.fig_schematic", "problem_schematic", 2, "two_col"),
    _spec("F02_architecture", "FNO with temporal forcing encoder",
          "visual.pub.fig_schematic", "architecture", 2, "two_col"),
    _spec("F25_mms_convergence", "MMS verification of the finite-volume solver",
          "visual.pub.fig_supplement", "mms_convergence", 2, "one_half_col"),

    # -------------------------------------- tier 1: blocked on canonical runs
    _spec("F03_learning_curves", "Learning curves across the four benchmarks",
          "visual.pub.fig_training", "learning_curves", 1, "two_col"),
    _spec("F04_headline_accuracy", "Cross-benchmark accuracy",
          "visual.pub.fig_crossbench", "headline_accuracy", 1, "two_col"),
    _spec("F05_truth_pred_residual", "Truth, prediction and residual by benchmark",
          "visual.pub.fig_crossbench", "truth_pred_residual", 1, "full_page"),
    _spec("F06_signature", "Field and contact-jump evolution for one case",
          "visual.pub.fig_signature", "signature", 1, "full_page"),
    _spec("F07_jump_vs_lead", "Interface-jump error against lead time",
          "visual.pub.fig_crossbench", "jump_vs_lead", 1, "two_col"),
    _spec("F08_forcing_difficulty", "Forcing: what makes a case hard",
          "visual.pub.fig_benchmark", "difficulty", 1, "two_col",
          benchmark="forcing"),
    _spec("F09_forcing_cases", "Forcing: representative cases",
          "visual.pub.fig_benchmark", "cases", 1, "two_col",
          benchmark="forcing"),
    _spec("F10_source_difficulty", "Source: what makes a case hard",
          "visual.pub.fig_benchmark", "difficulty", 1, "two_col",
          benchmark="source"),
    _spec("F11_source_cases", "Source: representative cases",
          "visual.pub.fig_benchmark", "cases", 1, "two_col",
          benchmark="source"),
    _spec("F12_source_itr_sin_resistance", "Source + ITR: resistance profile profile and difficulty",
          "visual.pub.fig_benchmark", "difficulty", 1, "two_col",
          benchmark="source_itr_sin"),
    _spec("F13_source_itr_sin_cases", "Source + ITR: representative cases",
          "visual.pub.fig_benchmark", "cases", 1, "two_col",
          benchmark="source_itr_sin"),
    _spec("F14_interfaces_difficulty", "Interfaces: what makes a case hard",
          "visual.pub.fig_benchmark", "difficulty", 1, "two_col",
          benchmark="interfaces"),
    _spec("F15_interfaces_profiles", "Interfaces: representative interface positions",
          "visual.pub.fig_benchmark", "cases", 1, "two_col",
          benchmark="interfaces"),
    _spec("F21_tail_reliability", "Tail reliability across benchmarks",
          "visual.pub.fig_crossbench", "tail_reliability", 1, "two_col"),
    _spec("F26_contact_jump_vs_lead",
          "Physical contact-jump fidelity across benchmarks",
          "visual.pub.fig_crossbench", "physical_contact_jump_vs_lead", 1,
          "two_col"),
    _spec("F27_global_field_error_vs_lead", "Global field RMSE versus lead time",
          "visual.pub.fig_crossbench", "global_field_error_vs_lead", 1, "one_col",
          formats=("png", "pdf", "svg"), preserve_size=True),
    _spec("F28_global_field_error_vs_itr", "Global field RMSE stratified by interface-mean resistance",
          "visual.pub.fig_crossbench", "global_field_error_vs_itr", 1, "one_col",
          formats=("png", "pdf", "svg"), preserve_size=True),
    # F27 pools every source time at a given lead, so its long leads come from a
    # smaller and systematically earlier cohort. This one conditions on a single
    # source snapshot, holding the cohort fixed along the whole lead axis.
    _spec("F32_global_field_error_fixed_source",
          "Global field RMSE against lead time at a fixed source time",
          "visual.pub.fig_crossbench", "global_field_error_fixed_source", 1, "one_col",
          formats=("png", "pdf", "svg"), preserve_size=True, source_fraction=0.2),
    # The adjacent-node companion to F26. Same four benchmarks, the other jump
    # definition -- this is the one test_records.csv scores.
    _spec("F31_node_jump_fidelity",
          "Adjacent-node interface-jump fidelity across benchmarks",
          "visual.pub.fig_crossbench", "node_jump_fidelity", 1, "two_col"),

    # ------------------------------- tier 3: blocked on experiments not yet run
    _spec("F16_inverse_forcing_recovery", "Parameter recovery: forcing",
          "visual.pub.fig_inverse", "recovery", 3, "two_col",
          benchmark="forcing"),
    _spec("F17_inverse_forcing_itr_sin", "Parameter recovery: forcing_itr_sin",
          "visual.pub.fig_inverse", "recovery", 3, "two_col",
          benchmark="forcing_itr_sin"),
    _spec("F18_sensor_count", "Inversion accuracy against sensor count",
          "visual.pub.fig_inverse", "sensor_count", 3, "two_col"),
    _spec("F19_ood_time_protocols", "Temporal extrapolation protocols",
          "visual.pub.fig_ood", "time_protocols", 3, "two_col"),
    _spec("F20_ood_transfer", "Parameter and family transfer",
          "visual.pub.fig_ood", "transfer", 3, "two_col"),
    _spec("F22_surrogate_fidelity", "Surrogate fidelity against recovery error",
          "visual.pub.fig_inverse", "surrogate_fidelity", 3, "one_half_col"),
    _spec("F23_direct_vs_autoregressive", "Direct against autoregressive rollout",
          "visual.pub.fig_supplement", "direct_vs_autoregressive", 3, "two_col"),
    _spec("F24_resolution_invariance", "Resolution invariance",
          "visual.pub.fig_supplement", "resolution_invariance", 3, "two_col"),
    # one_col, not two_col: both are stacked 3x1 so they set into a single
    # column of the two-column page rather than spanning it.
    _spec("F29_inverse_rc_profile_recovery",
          "Recovery of the spatially varying interfacial resistance",
          "visual.pub.fig_inverse", "rc_profile_recovery", 3, "one_col"),
    _spec("F30_inverse_identifiability",
          "Identifiability of the interfacial resistance parameters",
          "visual.pub.fig_inverse", "identifiability", 3, "one_col",
          n_sensors=32),
)

FIGURES: dict[str, FigureSpec] = {s.key: s for s in _SPECS}


def get_figure(key: str) -> FigureSpec:
    try:
        return FIGURES[key]
    except KeyError:
        raise KeyError(
            f"unknown figure key {key!r}; known keys: {sorted(FIGURES)}"
        ) from None


def figures_at_tier(tier: int) -> list[str]:
    """Every key at or below ``tier``, in key order."""
    return sorted(k for k, s in FIGURES.items() if s.tier <= tier)


def check_registry_matches(requirements) -> None:
    """Assert :data:`FIGURES` and ``figures.yaml`` declare the same keys."""
    declared = set(requirements.figures)
    registered = set(FIGURES)
    if declared != registered:
        raise ValueError(
            "figures.yaml and registry.FIGURES disagree: "
            f"only in yaml={sorted(declared - registered)}, "
            f"only in registry={sorted(registered - declared)}"
        )


@dataclass
class RenderResult:
    """What one :func:`render` call produced."""

    key: str
    paths: list[Path]
    sidecar: Path
    source: FigureSource
    degraded: bool

    @property
    def degradation_codes(self) -> list[str]:
        return [d.code for d in self.source.degradations]


def _commit_changed(out_dir: Path, key: str) -> str | None:
    """Return the sidecar's generator commit when it differs from the current one."""
    from visual.pub.manifest import git_state

    previous = read_sidecar(out_dir, key)
    if not previous:
        return None
    old = previous.get("generator_git_commit")
    new, _ = git_state()
    if old and new and old != new:
        return old
    return None


def render(key: str, *, manifest: Manifest | str | Path | None = None,
           out_dir: str | Path = DEFAULT_OUT_DIR, strict: bool = True,
           formats: tuple[str, ...] | None = None,
           footer: bool = False, force: bool = False) -> RenderResult:
    """Resolve provenance, draw one figure, and optionally stamp it.

    Provenance is resolved *before* the drawing module is imported, so a figure
    whose artifacts are missing fails without paying the import cost and without
    leaving a partial file behind.

    The on-figure footer is off by default. Provenance lives in the sidecar and
    in the ``degraded/`` directory split, which survive a figure being dropped
    into a manuscript; a line of 5 pt grey text under the artwork does not, and
    it is dead weight in the printed figure.
    """
    spec = get_figure(key)
    if isinstance(manifest, Manifest):
        mf = manifest
    else:
        mf = Manifest.load(manifest if manifest is not None else DEFAULT_MANIFEST)
    check_registry_matches(mf.requirements)

    source = mf.resolve(key, strict=strict)
    requirement = mf.requirements.figures[key]
    degraded = source.is_degraded

    target = Path(out_dir) / DEGRADED_SUBDIR if degraded else Path(out_dir)
    if not force:
        old_commit = _commit_changed(target, key)
        if old_commit is not None:
            raise ProvenanceError(
                f"{key}: an existing render in {target} was generated at git "
                f"{old_commit[:7]}, which is not the current commit. Pass "
                f"--force to overwrite."
            )

    if degraded:
        for degradation in source.degradations:
            print(f"DEGRADED {key}: {degradation}", flush=True)

    draw = spec.load()
    from visual.pub import style

    with style.pub_style():
        result = draw(source=source, spec=spec, requirement=requirement, **spec.params)

    fig, selection, metric_definition = _unpack(key, result)

    if footer:
        style.add_provenance_footer(fig, key, source=source, selection=selection)
    save_style = {"savefig.bbox": None, "svg.fonttype": "none"} if spec.preserve_size else {}
    with style.pub_style(extra=save_style):
        paths = style.save(fig, target, key, formats=spec.formats if formats is None else formats)
    if "statistics" in metric_definition:
        statistics_path = target / f"{key}.statistics.csv"
        rows = metric_definition["statistics"]
        with statistics_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows({k: json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v
                              for k, v in row.items()} for row in rows)
        metric_definition["statistics_file"] = statistics_path.name
        paths.append(statistics_path)
    sidecar = write_sidecar(
        target, source,
        metric_space=requirement.metric_space,
        metric_definition=metric_definition,
        selection=_selection_dict(selection),
        params=spec.params,
    )
    return RenderResult(key=key, paths=paths, sidecar=sidecar, source=source,
                        degraded=degraded)


def _selection_dict(selection) -> dict:
    """Normalize whatever the drawing function chose to describe its case with.

    Most qualitative figures pick a (sim, s, j) pair and return a
    ``select.CaseSelection``. The inverse figures pick a whole inversion case,
    which has no pair, no quantile rank, and no pair metric, so they return a
    plain dict rather than fill two thirds of a CaseSelection with nulls.
    """
    if selection is None:
        return {}
    if hasattr(selection, "to_dict"):
        return selection.to_dict()
    return dict(selection)


def _unpack(key: str, result):
    """Accept ``fig``, ``(fig, selection)``, or ``(fig, selection, metric_def)``."""
    if isinstance(result, tuple):
        if len(result) == 2:
            fig, selection = result
            return fig, selection, {}
        if len(result) == 3:
            return result
        raise TypeError(
            f"{key}: drawing function returned a {len(result)}-tuple; expected "
            f"fig, (fig, selection), or (fig, selection, metric_definition)"
        )
    return result, None, {}


def render_many(keys, **kwargs) -> list[RenderResult]:
    return [render(key, **kwargs) for key in keys]


__all__ = [
    "DEFAULT_OUT_DIR",
    "DEGRADED_SUBDIR",
    "FIGURES",
    "FigureSpec",
    "RenderResult",
    "check_registry_matches",
    "figures_at_tier",
    "get_figure",
    "render",
    "render_many",
]
