"""Provenance layer: what a figure requires, which runs satisfy it, and proof.

Three layers:

1. ``figures.yaml`` -- version controlled. Declares named ``requirement_sets``
   (kind, benchmarks, representation, split, min_seeds, schema_min,
   required_columns) and binds each figure key to the sets it needs.
2. ``manifest.yaml`` -- user supplied. Maps each requirement-set name onto the
   concrete run directories or tables that satisfy it.
3. :class:`FigureSource` -- the resolved, hashed, verified result, written next
   to every rendered figure as a ``.provenance.json`` sidecar.

Strict resolution is the default: a figure that cannot name its sources raises
:class:`ProvenanceError` rather than rendering. ``--allow-missing`` downgrades
the failure to a :class:`Degradation`, which is carried all the way into the
figure footer. This module must not import matplotlib.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PUB_DIR = Path(__file__).resolve().parent
FIGURES_YAML = PUB_DIR / "figures.yaml"
DEFAULT_MANIFEST = PUB_DIR / "manifest.yaml"

SIDECAR_SCHEMA = "visual.pub.provenance/1"

ARTIFACT_KINDS = (
    "test_records", "train_metrics", "ood_per_sim", "inverse_csv",
    "checkpoint", "json_report", "none",
)

METRIC_SPACES = ("kelvin", "normalized", "mixed", "none")

# Filename each artifact kind takes inside a run directory. ``inverse_csv`` and
# ``ood_per_sim`` are always given as explicit paths instead.
_RUN_ARTIFACT_NAME = {
    "test_records": "test_records.csv",
    "train_metrics": "train_metrics.csv",
}


class ProvenanceError(RuntimeError):
    """A figure's declared requirements are not satisfied by the manifest."""


@dataclass(frozen=True)
class Degradation:
    """A named, human-readable reason a figure's numbers are not fully trusted."""

    code: str
    detail: str

    def __str__(self) -> str:
        return f"{self.code}: {self.detail}"


@dataclass(frozen=True)
class ArtifactRef:
    """A hashed reference to one file on disk, with its measured contents."""

    path: str
    kind: str
    sha256: str
    mtime_iso: str
    n_bytes: int
    schema_version: int | None = None
    n_rows: int | None = None
    n_simulations: int | None = None
    seed: str | None = None
    benchmarks: tuple[str, ...] = ()


@dataclass(frozen=True)
class RunSource:
    """The training run an artifact came from, as recorded by that run itself."""

    run_dir: str
    benchmark: str | None
    representation: str | None
    seed: str | None
    git_commit: str | None
    git_dirty: bool
    config_sha256: str | None
    checkpoint_sha256: str | None
    dataset_dir: str | None
    final_metrics: dict


@dataclass
class FigureSource:
    """Everything one figure is allowed to know about where its numbers came from."""

    figure_key: str
    requirement_sets: tuple[str, ...] = ()
    runs: list[RunSource] = field(default_factory=list)
    artifacts: list[ArtifactRef] = field(default_factory=list)
    degradations: list[Degradation] = field(default_factory=list)

    @property
    def is_degraded(self) -> bool:
        return bool(self.degradations)

    @property
    def seeds(self) -> tuple[str, ...]:
        seen = {str(a.seed) for a in self.artifacts if a.seed is not None} | {
            str(r.seed) for r in self.runs if r.seed is not None
        }
        return tuple(sorted(seen))

    @property
    def n_seeds(self) -> int:
        return len(self.seeds)

    @property
    def n_simulations(self) -> int:
        return sum(a.n_simulations or 0 for a in self.artifacts)

    @property
    def n_pairs(self) -> int:
        return sum(a.n_rows or 0 for a in self.artifacts)

    @property
    def benchmarks(self) -> tuple[str, ...]:
        seen: set[str] = set()
        for a in self.artifacts:
            seen.update(a.benchmarks)
        for r in self.runs:
            if r.benchmark:
                seen.add(r.benchmark)
        return tuple(sorted(seen))

    @property
    def git_commit(self) -> str | None:
        commits = {r.git_commit for r in self.runs if r.git_commit}
        return sorted(commits)[0] if len(commits) == 1 else None

    @property
    def git_dirty(self) -> bool:
        return any(r.git_dirty for r in self.runs)

    @property
    def run_label(self) -> str:
        names = sorted({Path(r.run_dir).parts[-2] if len(Path(r.run_dir).parts) > 1
                        else r.run_dir for r in self.runs})
        return ",".join(names) if names else "no run"

    def add_degradation(self, code: str, detail: str) -> None:
        self.degradations.append(Degradation(code, detail))

    def paths(self) -> list[Path]:
        return [Path(a.path) for a in self.artifacts]

    def footer_fields(self) -> list[str]:
        """The provenance fields ``style.add_provenance_footer`` prints."""
        fields = []
        if self.benchmarks:
            fields.append("+".join(self.benchmarks))
        fields.append(self.run_label)
        if self.seeds:
            fields.append(f"seed {','.join(self.seeds)}")
        counts = []
        if self.n_simulations:
            counts.append(f"n_sim={self.n_simulations}")
        if self.n_pairs:
            counts.append(f"n_pair={self.n_pairs}")
        if counts:
            fields.append(" ".join(counts))
        if self.git_commit:
            fields.append(f"git {self.git_commit[:7]}{'*' if self.git_dirty else ''}")
        if self.degradations:
            fields.append("DEGRADED: "
                          + ",".join(sorted({d.code for d in self.degradations})))
        return fields


def sha256_file(path: str | Path) -> str:
    """Chunked SHA-256 of a file, matching ``scripts/run_ood_suite.py``."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_state(cwd: Path = PROJECT_ROOT) -> tuple[str | None, bool]:
    """Return ``(commit, dirty)`` for the generating checkout."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return None, False
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=cwd,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        status = ""
    return commit, bool(status)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _mtime_iso(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(
        timespec="seconds")


# ---------------------------------------------------------------------------
# figures.yaml
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RequirementSet:
    """One named class of artifact a figure can require."""

    name: str
    kind: str
    benchmarks: tuple[str, ...] = ()
    representation: str = "temporal_encoder"
    split: str = "test"
    min_seeds: int = 1
    schema_min: int = 2
    required_columns: tuple[str, ...] = ()
    note: str = ""
    regenerate: str = ""


@dataclass(frozen=True)
class FigureRequirement:
    """The requirement side of one figure key, as declared in ``figures.yaml``."""

    key: str
    requires: tuple[str, ...]
    metric_space: str


@dataclass(frozen=True)
class FigureRequirements:
    """Parsed ``figures.yaml``."""

    requirement_sets: dict[str, RequirementSet]
    figures: dict[str, FigureRequirement]

    @classmethod
    def load(cls, path: str | Path = FIGURES_YAML) -> FigureRequirements:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"figures.yaml not found: {path}")
        raw = yaml.safe_load(path.read_text()) or {}

        sets: dict[str, RequirementSet] = {}
        for name, spec in (raw.get("requirement_sets") or {}).items():
            spec = spec or {}
            kind = spec.get("kind", "none")
            if kind not in ARTIFACT_KINDS:
                raise ValueError(
                    f"requirement_set {name!r} has unknown kind {kind!r}; "
                    f"allowed: {list(ARTIFACT_KINDS)}"
                )
            sets[name] = RequirementSet(
                name=name,
                kind=kind,
                benchmarks=tuple(spec.get("benchmarks") or ()),
                representation=spec.get("representation", "temporal_encoder"),
                split=spec.get("split", "test"),
                min_seeds=int(spec.get("min_seeds", 1)),
                schema_min=int(spec.get("schema_min", 2)),
                required_columns=tuple(spec.get("required_columns") or ()),
                note=spec.get("note", ""),
                regenerate=spec.get("regenerate", ""),
            )

        figures: dict[str, FigureRequirement] = {}
        for key, spec in (raw.get("figures") or {}).items():
            spec = spec or {}
            requires = tuple(spec.get("requires") or ())
            unknown = [r for r in requires if r not in sets]
            if unknown:
                raise ValueError(
                    f"figure {key!r} requires undeclared requirement_sets: {unknown}"
                )
            metric_space = spec.get("metric_space", "none")
            if metric_space not in METRIC_SPACES:
                raise ValueError(
                    f"figure {key!r} has unknown metric_space {metric_space!r}"
                )
            figures[key] = FigureRequirement(key, requires, metric_space)

        return cls(requirement_sets=sets, figures=figures)


# ---------------------------------------------------------------------------
# manifest.yaml
# ---------------------------------------------------------------------------


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def _run_metadata(run_dir: Path) -> RunSource:
    config_path = run_dir / "config_used.yaml"
    config = _read_yaml(config_path)
    metrics_path = run_dir / "final_metrics.json"
    final_metrics = {}
    if metrics_path.exists():
        try:
            final_metrics = json.loads(metrics_path.read_text())
        except json.JSONDecodeError:
            final_metrics = {}

    commit = None
    commit_path = run_dir / "git_commit.txt"
    if commit_path.exists():
        commit = commit_path.read_text().strip() or None
    status_path = run_dir / "git_status.txt"
    dirty = bool(status_path.exists() and status_path.read_text().strip())

    checkpoints = sorted(run_dir.glob("*_best.pt"))
    seed = run_dir.name[len("seed"):] if run_dir.name.startswith("seed") else None
    if seed is None:
        seed = str(final_metrics.get("seed") or "") or None

    data_cfg = config.get("data", {}) if isinstance(config, dict) else {}
    # Training configs store the benchmark group as a nested block; flat keys
    # are accepted too because a hand-written config_used.yaml is a legitimate way
    # to describe a run that was trained elsewhere.
    bench_cfg = config.get("benchmark") if isinstance(config, dict) else None
    if isinstance(bench_cfg, dict):
        benchmark = bench_cfg.get("name")
        representation = bench_cfg.get("representation")
    else:
        benchmark = bench_cfg
        representation = config.get("representation") if isinstance(config, dict) else None

    return RunSource(
        run_dir=str(run_dir),
        benchmark=benchmark,
        representation=representation,
        seed=seed,
        git_commit=commit,
        git_dirty=dirty,
        config_sha256=sha256_file(config_path) if config_path.exists() else None,
        checkpoint_sha256=sha256_file(checkpoints[0]) if checkpoints else None,
        dataset_dir=data_cfg.get("data_dir") if isinstance(data_cfg, dict) else None,
        final_metrics=final_metrics,
    )


def _inspect_artifact(path: Path, kind: str, *, seed: str | None) -> ArtifactRef:
    """Hash a file and measure whatever its kind lets us measure."""
    # Deferred: records.py imports Degradation from this module.
    from visual.pub import records as _records

    schema_version = n_rows = n_simulations = None
    benchmarks: tuple[str, ...] = ()

    if kind in ("test_records", "ood_per_sim", "inverse_csv", "train_metrics"):
        import pandas as pd

        df = pd.read_csv(path)
        n_rows = int(len(df))
        if kind == "test_records":
            schema_version = _records.detect_schema_version(df.columns)
        if "sim_id" in df.columns:
            n_simulations = int(df["sim_id"].nunique())
        if "benchmark" in df.columns:
            benchmarks = tuple(sorted(df["benchmark"].dropna().astype(str).unique()))
        if seed is None and "seed" in df.columns:
            seeds = sorted(df["seed"].dropna().astype(str).unique())
            seed = seeds[0] if len(seeds) == 1 else None

    return ArtifactRef(
        path=str(path),
        kind=kind,
        sha256=sha256_file(path),
        mtime_iso=_mtime_iso(path),
        n_bytes=path.stat().st_size,
        schema_version=schema_version,
        n_rows=n_rows,
        n_simulations=n_simulations,
        seed=None if seed is None else str(seed),
        benchmarks=benchmarks,
    )


@dataclass
class Manifest:
    """Which concrete runs and tables satisfy each declared requirement set."""

    sources: dict[str, list[dict]]
    requirements: FigureRequirements
    root: Path = PROJECT_ROOT
    path: Path | None = None

    @classmethod
    def discover_global_field(cls, runs_root: str | Path, *,
                              selections: dict[str, str] | None = None) -> Manifest:
        """Discover complete seed records, requiring one experiment per benchmark."""
        benchmarks = ("forcing", "source", "source_itr_sin", "interfaces")
        root = Path(runs_root).expanduser().resolve()
        if not root.is_dir():
            raise ProvenanceError(f"Runs directory does not exist: {root}")
        selections = selections or {}
        unknown = set(selections) - set(benchmarks)
        if unknown:
            raise ProvenanceError(f"Unknown benchmark selection: {sorted(unknown)}")
        candidates = {b: {} for b in benchmarks}
        for record in sorted(root.rglob("test_records.csv")):
            run = record.parent.resolve()
            config_path = run / "config_used.yaml"
            if not config_path.is_file():
                raise ProvenanceError(f"Missing run configuration: {config_path}")
            try:
                config = _read_yaml(config_path)
            except yaml.YAMLError as exc:
                raise ProvenanceError(f"Invalid run configuration: {config_path}: {exc}") from exc
            if not isinstance(config, dict):
                raise ProvenanceError(f"Expected a configuration mapping: {config_path}")
            benchmark = config.get("benchmark")
            if isinstance(benchmark, dict):
                benchmark = benchmark.get("name")
            if benchmark not in benchmarks:
                continue
            if not run.name.startswith("seed") or not run.name[4:].isdigit():
                raise ProvenanceError(f"Expected a seed<number> directory: {run}")
            candidates[benchmark].setdefault(run.parent, {})[run] = {
                "run": str(run), "seed": int(run.name[4:]),
            }
        sources = {}
        for benchmark, experiments in candidates.items():
            if benchmark in selections:
                selected = Path(selections[benchmark]).expanduser().resolve()
                experiments = {p: entries for p, entries in experiments.items() if p == selected}
            if len(experiments) != 1:
                choices = "\n".join(f"  --run {benchmark}={p}" for p in candidates[benchmark])
                raise ProvenanceError(
                    f"{benchmark}: expected one experiment with test_records.csv under {root}; "
                    f"found {len(experiments)} matching experiments. "
                    "Select its config directory with --run benchmark=/absolute/path/config0.\n"
                    + (choices or "  No eligible records found.")
                )
            sources[f"{benchmark}_records"] = list(next(iter(experiments.values())).values())
        return cls(sources=sources, requirements=FigureRequirements.load())

    @classmethod
    def load(cls, path: str | Path | None = None, *,
             figures_yaml: str | Path = FIGURES_YAML,
             root: Path = PROJECT_ROOT) -> Manifest:
        requirements = FigureRequirements.load(figures_yaml)
        resolved = Path(path) if path is not None else DEFAULT_MANIFEST
        raw = _read_yaml(resolved)
        sources = {name: list(entries or [])
                   for name, entries in (raw.get("sources") or {}).items()}
        return cls(sources=sources, requirements=requirements, root=root,
                   path=resolved if resolved.exists() else None)

    def _resolve_entry(self, entry: dict, decl: RequirementSet
                       ) -> tuple[RunSource | None, Path | None]:
        if "run" in entry:
            status = str(entry.get("status", "completed"))
            allowed = {
                "completed", "infrastructure_invalid", "invalid_checkpoint",
                "representation_incompatible", "pending_rerun",
            }
            if status not in allowed:
                raise ProvenanceError(
                    f"{decl.name}: unknown declared-run status {status!r}; "
                    f"expected one of {sorted(allowed)}"
                )
            if status != "completed":
                seed = entry.get("seed", "unspecified")
                rerun = entry.get("rerun_of")
                lineage = f"; rerun_of={rerun}" if rerun else ""
                raise ProvenanceError(
                    f"{decl.name}: declared seed {seed} has status {status}{lineage}. "
                    "Infrastructure-invalid attempts may be rerun only with the "
                    "same declared seed/configuration; completed poor runs may not "
                    "be silently replaced."
                )
            run_dir = self.root / entry["run"] if not Path(entry["run"]).is_absolute() \
                else Path(entry["run"])
            if not run_dir.is_dir():
                raise ProvenanceError(f"run directory does not exist: {run_dir}")
            run = _run_metadata(run_dir)
            name = entry.get("artifact") or _RUN_ARTIFACT_NAME.get(decl.kind)
            if name is None:
                return run, None
            artifact_path = run_dir / name
            if not artifact_path.exists():
                raise ProvenanceError(
                    f"{decl.name}: run {run_dir} has no {name}. "
                    f"{decl.regenerate or 'Re-run evaluation for this run.'}"
                )
            return run, artifact_path
        if "table" in entry:
            table = Path(entry["table"])
            table = table if table.is_absolute() else self.root / table
            if not table.exists():
                raise ProvenanceError(f"{decl.name}: table does not exist: {table}")
            return None, table
        raise ProvenanceError(
            f"{decl.name}: manifest entry must have a 'run' or 'table' key, got "
            f"{sorted(entry)}"
        )

    def resolve(self, figure_key: str, *, strict: bool = True) -> FigureSource:
        """Resolve every requirement set of one figure into a hashed source.

        Under ``strict`` any unmet requirement raises. Otherwise it is recorded
        as a :class:`Degradation` and the figure renders into ``degraded/``.
        """
        try:
            req = self.requirements.figures[figure_key]
        except KeyError:
            raise ProvenanceError(
                f"unknown figure key {figure_key!r}; not declared in figures.yaml"
            ) from None

        source = FigureSource(figure_key=figure_key, requirement_sets=req.requires)

        for name in req.requires:
            decl = self.requirements.requirement_sets[name]
            if decl.kind == "none":
                continue
            entries = self.sources.get(name) or []
            if not entries:
                self._fail(source, strict, "MANIFEST_ENTRY_MISSING",
                           f"requirement set {name!r} ({decl.kind}) has no manifest "
                           f"entry. {decl.regenerate or decl.note}".strip())
                continue

            set_artifacts: list[ArtifactRef] = []
            set_runs: list[RunSource] = []
            for entry in entries:
                try:
                    run, artifact_path = self._resolve_entry(entry, decl)
                except ProvenanceError as exc:
                    self._fail(source, strict, "ARTIFACT_MISSING", str(exc))
                    continue
                if run is not None:
                    set_runs.append(run)
                    source.runs.append(run)
                if artifact_path is None:
                    continue
                ref = _inspect_artifact(
                    artifact_path, decl.kind,
                    seed=entry.get("seed") or (run.seed if run else None),
                )
                set_artifacts.append(ref)
                source.artifacts.append(ref)

            self._validate_set(source, decl, set_artifacts, set_runs, strict)

        return source

    def _fail(self, source: FigureSource, strict: bool, code: str, detail: str) -> None:
        if strict:
            raise ProvenanceError(f"{source.figure_key}: {code}: {detail}")
        source.add_degradation(code, detail)

    def _validate_set(self, source: FigureSource, decl: RequirementSet,
                      artifacts: list[ArtifactRef], runs: list[RunSource],
                      strict: bool) -> None:
        if not artifacts:
            return

        for ref in artifacts:
            if ref.schema_version is not None and ref.schema_version < decl.schema_min:
                self._fail(
                    source, strict, "SCHEMA_BELOW_REQUIRED",
                    f"{ref.path} is schema v{ref.schema_version}; {decl.name} "
                    f"requires v{decl.schema_min}. "
                    f"{decl.regenerate or 'Re-run scripts/write_test_records.py.'}",
                )
            if decl.required_columns:
                missing = _missing_columns(Path(ref.path), decl.required_columns)
                if missing:
                    self._fail(source, strict, "COLUMNS_MISSING",
                               f"{ref.path} is missing columns {missing}")

        seeds = {ref.seed for ref in artifacts if ref.seed}
        if len(seeds) < decl.min_seeds:
            self._fail(
                source, strict, "TOO_FEW_SEEDS",
                f"{decl.name} has {len(seeds)} seed(s); {decl.min_seeds} required. "
                f"{decl.regenerate or ''}".strip(),
            )

        if decl.benchmarks:
            present: set[str] = set()
            for ref in artifacts:
                present.update(ref.benchmarks)
            missing = [b for b in decl.benchmarks if b not in present]
            if missing and present:
                self._fail(source, strict, "BENCHMARK_MISSING",
                           f"{decl.name} covers {sorted(present)}; missing {missing}")

        for run in runs:
            if run.representation and run.representation != decl.representation:
                self._fail(
                    source, strict, "REPRESENTATION_MISMATCH",
                    f"{run.run_dir} is representation {run.representation!r}; "
                    f"{decl.name} requires {decl.representation!r}",
                )

    def satisfiability(self) -> list[dict]:
        """Resolve every declared figure and report what blocks it.

        This is the table ``python -m visual.pub --verify`` prints. It is the
        honest statement of which quantitative figures the repo can currently
        support.
        """
        rows = []
        for key in sorted(self.requirements.figures):
            source = self.resolve(key, strict=False)
            rows.append({
                "key": key,
                "requires": list(self.requirements.figures[key].requires),
                "satisfied": not source.is_degraded,
                "n_seeds": source.n_seeds,
                "n_simulations": source.n_simulations,
                "n_pairs": source.n_pairs,
                "blocking": [str(d) for d in source.degradations],
            })
        return rows


def _missing_columns(path: Path, required: tuple[str, ...]) -> list[str]:
    import pandas as pd

    header = pd.read_csv(path, nrows=0).columns
    return [c for c in required if c not in header]


# ---------------------------------------------------------------------------
# sidecars
# ---------------------------------------------------------------------------


def sidecar_path(out_dir: str | Path, figure_key: str) -> Path:
    return Path(out_dir) / f"{figure_key}.provenance.json"


def build_sidecar(source: FigureSource, *, metric_space: str,
                  metric_definition: dict | None = None,
                  selection: dict | None = None,
                  params: dict | None = None) -> dict:
    commit, dirty = git_state()
    return {
        "schema": SIDECAR_SCHEMA,
        "figure_key": source.figure_key,
        "generated_utc": _utc_now(),
        "generator_git_commit": commit,
        "generator_git_dirty": dirty,
        "metric_space": metric_space,
        "metric_definition": metric_definition or {},
        "params": params or {},
        "requirement_sets": list(source.requirement_sets),
        "runs": [asdict(r) for r in source.runs],
        "sources": [asdict(a) for a in source.artifacts],
        "counts": {
            "n_simulations": source.n_simulations,
            "n_pairs": source.n_pairs,
            "n_seeds": source.n_seeds,
        },
        "selection": selection or {},
        "degradations": [asdict(d) for d in source.degradations],
    }


def write_sidecar(out_dir: str | Path, source: FigureSource, *, metric_space: str,
                  metric_definition: dict | None = None,
                  selection: dict | None = None,
                  params: dict | None = None) -> Path:
    path = sidecar_path(out_dir, source.figure_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_sidecar(source, metric_space=metric_space,
                            metric_definition=metric_definition,
                            selection=selection, params=params)
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def read_sidecar(out_dir: str | Path, figure_key: str) -> dict | None:
    path = sidecar_path(out_dir, figure_key)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def audit(out_dir: str | Path) -> list[dict]:
    """Re-hash the sources of already-rendered figures and report drift.

    A schematic figure has no hashed sources, but it still gets a row: the
    absence of a row would be indistinguishable from the figure never having
    been rendered, and this command exists to say what it actually inspected.
    """
    out_dir = Path(out_dir)
    findings = []
    for path in sorted(out_dir.glob("**/*.provenance.json")):
        payload = json.loads(path.read_text())
        artifact_key = (
            payload.get("figure_key") or payload.get("table_key") or path.stem
        )
        sources = payload.get("sources", [])
        if not sources:
            findings.append({"figure_key": artifact_key,
                             "path": str(path), "status": "OK",
                             "detail": "no hashed sources"})
            continue
        for entry in sources:
            src = Path(entry["path"])
            if not src.exists():
                findings.append({"figure_key": artifact_key,
                                 "path": str(src), "status": "MISSING"})
                continue
            digest = sha256_file(src)
            findings.append({
                "figure_key": artifact_key,
                "path": str(src),
                "status": "OK" if digest == entry.get("sha256") else "DRIFT",
                "expected_sha256": entry.get("sha256"),
                "actual_sha256": digest,
            })
    return findings


__all__ = [
    "ARTIFACT_KINDS",
    "METRIC_SPACES",
    "SIDECAR_SCHEMA",
    "ArtifactRef",
    "Degradation",
    "FigureRequirement",
    "FigureRequirements",
    "FigureSource",
    "Manifest",
    "ProvenanceError",
    "RequirementSet",
    "RunSource",
    "audit",
    "build_sidecar",
    "git_state",
    "read_sidecar",
    "sha256_file",
    "sidecar_path",
    "write_sidecar",
]
