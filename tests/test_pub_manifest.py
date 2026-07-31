"""Tests for ``visual.pub.manifest``, ``visual.pub.records`` and the render gate.

The policy under test is that a figure never quietly fails to appear. The
existing diagnostic pipeline skips a plot whose inputs are missing and prints a
note (``visual/_common.py:_print_skip``); this package refuses instead. So the
central test here is :class:`TestNoSilentSkip`, which asserts for *every*
registered key that an empty manifest produces an exception and no file on disk.
"""

from __future__ import annotations

import json

import matplotlib
import pandas as pd
import pytest
import yaml

matplotlib.use("Agg")

from visual.pub import records as records_mod  # noqa: E402
from visual.pub import registry  # noqa: E402
from visual.pub.manifest import (  # noqa: E402
    FigureRequirements,
    FigureSource,
    Manifest,
    ProvenanceError,
    audit,
    read_sidecar,
    sha256_file,
    write_sidecar,
)

SCHEMATIC_KEYS = ("F01_problem_schematic", "F02_architecture", "F25_mms_convergence")


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def requirements() -> FigureRequirements:
    return FigureRequirements.load()


@pytest.fixture
def empty_manifest(tmp_path) -> Manifest:
    """A manifest that declares nothing, i.e. the current state of this repo."""
    return Manifest.load(tmp_path / "absent.yaml")


def write_records(path, *, version: int, n_sims: int = 3, n_pairs: int = 4,
                  benchmark: str = "source_itr", seed: str = "42"):
    """A test_records.csv at the requested schema version."""
    columns = list(records_mod.SCHEMA_V1_COLUMNS)
    if version >= 2:
        columns += list(records_mod.SCHEMA_V2_REQUIRED)
    if version >= 3:
        columns += list(records_mod.SCHEMA_V3_REQUIRED)

    rows = []
    for sim in range(n_sims):
        for k in range(n_pairs):
            row = {c: 0.0 for c in columns}
            row.update({"sim_id": sim, "s": k // 2, "j": k % 2 + 1,
                        "benchmark": benchmark})
            if version >= 2:
                row.update({"sse_K2": 1.0, "num_error_cells": 256.0,
                            "target_sse_K2": 400.0, "interface_sse_K2": 0.25,
                            "num_interface_cells": 16.0,
                            "interface_target_sse_K2": 100.0})
            rows.append(row)
    frame = pd.DataFrame(rows, columns=columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def make_run(root, *, name="pub_source_itr", seed="42", version=2,
             benchmark="source_itr", representation="temporal_encoder"):
    """A minimal run directory carrying the metadata the manifest reads."""
    run_dir = root / "runs" / name / "config0" / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config_used.yaml").write_text(yaml.safe_dump({
        "benchmark": benchmark,
        "representation": representation,
        "data": {"data_dir": "data/fake"},
    }))
    (run_dir / "final_metrics.json").write_text(json.dumps({"val_rel_l2": 1.0}))
    (run_dir / "git_commit.txt").write_text("0f988c4" + "0" * 33)
    (run_dir / "git_status.txt").write_text("")
    write_records(run_dir / "test_records.csv", version=version,
                  benchmark=benchmark, seed=seed)
    return run_dir


def manifest_with(tmp_path, sources: dict) -> Manifest:
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump({"sources": sources}))
    return Manifest.load(path, root=tmp_path)


# ---------------------------------------------------------------------------
# figures.yaml <-> registry
# ---------------------------------------------------------------------------


class TestCompleteness:
    def test_key_sets_agree_in_both_directions(self, requirements):
        registry.check_registry_matches(requirements)
        assert set(requirements.figures) == set(registry.FIGURES)

    def test_every_requirement_name_is_declared(self, requirements):
        for key, figure in requirements.figures.items():
            for name in figure.requires:
                assert name in requirements.requirement_sets, (key, name)

    def test_every_requirement_set_is_used(self, requirements):
        used = {name for f in requirements.figures.values() for name in f.requires}
        unused = set(requirements.requirement_sets) - used
        assert not unused, f"requirement sets declared but never required: {unused}"

    def test_every_figure_declares_a_known_metric_space(self, requirements):
        from visual.pub.manifest import METRIC_SPACES
        for key, figure in requirements.figures.items():
            assert figure.metric_space in METRIC_SPACES, key

    def test_every_registered_module_and_function_import(self):
        for key in registry.FIGURES:
            fn = registry.get_figure(key).load()
            assert callable(fn), key

    def test_schematic_figures_require_nothing(self, requirements):
        """The three tier-2 figures must be renderable with no artifacts at all.

        If one of them acquires a requirement, it stops being the thing that
        validates the style layer while the canonical runs are queued.
        """
        for key in SCHEMATIC_KEYS:
            assert requirements.figures[key].requires == ()

    def test_registry_disagreement_raises(self, requirements):
        broken = FigureRequirements(
            requirement_sets=requirements.requirement_sets,
            figures={k: v for k, v in requirements.figures.items()
                     if k != "F04_headline_accuracy"},
        )
        with pytest.raises(ValueError, match="F04_headline_accuracy"):
            registry.check_registry_matches(broken)

    def test_unknown_requirement_name_is_rejected(self, tmp_path):
        path = tmp_path / "figures.yaml"
        path.write_text(yaml.safe_dump({
            "requirement_sets": {"a": {"kind": "test_records"}},
            "figures": {"F99": {"requires": ["nope"], "metric_space": "kelvin"}},
        }))
        with pytest.raises(ValueError, match="undeclared"):
            FigureRequirements.load(path)

    def test_unknown_kind_is_rejected(self, tmp_path):
        path = tmp_path / "figures.yaml"
        path.write_text(yaml.safe_dump({
            "requirement_sets": {"a": {"kind": "vibes"}}, "figures": {}}))
        with pytest.raises(ValueError, match="unknown kind"):
            FigureRequirements.load(path)

    def test_unknown_metric_space_is_rejected(self, tmp_path):
        path = tmp_path / "figures.yaml"
        path.write_text(yaml.safe_dump({
            "requirement_sets": {},
            "figures": {"F99": {"requires": [], "metric_space": "furlongs"}},
        }))
        with pytest.raises(ValueError, match="metric_space"):
            FigureRequirements.load(path)


# ---------------------------------------------------------------------------
# strict resolution
# ---------------------------------------------------------------------------


class TestStrictResolution:
    def test_missing_entry_names_the_requirement_set(self, empty_manifest):
        with pytest.raises(ProvenanceError) as exc:
            empty_manifest.resolve("F04_headline_accuracy", strict=True)
        message = str(exc.value)
        assert "F04_headline_accuracy" in message
        assert "MANIFEST_ENTRY_MISSING" in message
        assert "forcing_records" in message

    def test_message_carries_the_regeneration_command(self, empty_manifest):
        with pytest.raises(ProvenanceError) as exc:
            empty_manifest.resolve("F08_forcing_difficulty", strict=True)
        message = str(exc.value)
        assert "scripts/run_train_fixed.py" in message

    def test_unknown_figure_key_raises(self, empty_manifest):
        with pytest.raises(ProvenanceError, match="unknown figure key"):
            empty_manifest.resolve("F99_not_a_figure", strict=True)

    def test_requirement_free_figures_resolve_clean(self, empty_manifest):
        for key in SCHEMATIC_KEYS:
            source = empty_manifest.resolve(key, strict=True)
            assert not source.is_degraded
            assert source.artifacts == []

    def test_missing_run_directory_raises(self, tmp_path):
        mf = manifest_with(tmp_path, {"forcing_records": [{"run": "runs/nope"}]})
        with pytest.raises(ProvenanceError, match="run directory does not exist"):
            mf.resolve("F08_forcing_difficulty", strict=True)

    def test_entry_without_run_or_table_raises(self, tmp_path):
        mf = manifest_with(tmp_path, {"forcing_records": [{"path": "x.csv"}]})
        with pytest.raises(ProvenanceError, match="must have a 'run' or 'table'"):
            mf.resolve("F08_forcing_difficulty", strict=True)

    def test_schema_below_required_raises(self, tmp_path):
        """A v1 records file cannot satisfy a v2 requirement.

        This is the guard that stops the one surviving run in this repo from
        silently backing a figure whose statistics need pooled sufficient
        statistics it does not carry.
        """
        run = make_run(tmp_path, name="pub_source_itr", version=1)
        mf = manifest_with(tmp_path, {
            "source_itr_records": [{"run": str(run.relative_to(tmp_path))}]})
        with pytest.raises(ProvenanceError, match="SCHEMA_BELOW_REQUIRED"):
            mf.resolve("F12_source_itr_void", strict=True)

    def test_too_few_seeds_raises(self, tmp_path):
        run = make_run(tmp_path, name="pub_source_itr", version=2)
        mf = manifest_with(tmp_path, {
            "source_itr_records": [{"run": str(run.relative_to(tmp_path))}]})
        with pytest.raises(ProvenanceError, match="TOO_FEW_SEEDS"):
            mf.resolve("F12_source_itr_void", strict=True)

    def test_representation_mismatch_raises(self, tmp_path):
        runs = [make_run(tmp_path, seed=s, version=2) for s in ("1", "2")]
        runs.append(make_run(tmp_path, seed="3", version=2,
                             representation="bins"))
        mf = manifest_with(tmp_path, {
            "source_itr_records": [{"run": str(r.relative_to(tmp_path))}
                                   for r in runs]})
        with pytest.raises(ProvenanceError, match="REPRESENTATION_MISMATCH"):
            mf.resolve("F12_source_itr_void", strict=True)

    def test_three_good_seeds_resolve_clean(self, tmp_path):
        runs = [make_run(tmp_path, seed=s, version=2) for s in ("1", "2", "3")]
        mf = manifest_with(tmp_path, {
            "source_itr_records": [{"run": str(r.relative_to(tmp_path))}
                                   for r in runs]})
        source = mf.resolve("F12_source_itr_void", strict=True)
        assert not source.is_degraded
        assert source.n_seeds == 3
        assert source.n_simulations == 9
        assert source.n_pairs == 36
        assert source.benchmarks == ("source_itr",)


# ---------------------------------------------------------------------------
# degraded mode
# ---------------------------------------------------------------------------


class TestDegraded:
    def test_allow_missing_records_rather_than_raises(self, empty_manifest):
        source = empty_manifest.resolve("F04_headline_accuracy", strict=False)
        assert source.is_degraded
        codes = {d.code for d in source.degradations}
        assert codes == {"MANIFEST_ENTRY_MISSING"}
        # One per unmet requirement set, so the report is not just "something".
        assert len(source.degradations) == 4

    def test_degradation_appears_in_the_footer(self, empty_manifest):
        source = empty_manifest.resolve("F04_headline_accuracy", strict=False)
        fields = source.footer_fields()
        assert any("DEGRADED" in f for f in fields)

    def test_v1_records_degrade_with_a_named_code(self, tmp_path):
        run = make_run(tmp_path, version=1)
        mf = manifest_with(tmp_path, {
            "source_itr_records": [{"run": str(run.relative_to(tmp_path))}]})
        source = mf.resolve("F12_source_itr_void", strict=False)
        codes = {d.code for d in source.degradations}
        assert "SCHEMA_BELOW_REQUIRED" in codes
        assert source.artifacts[0].schema_version == 1

    def test_satisfiability_table_covers_every_figure(self, empty_manifest):
        rows = empty_manifest.satisfiability()
        assert len(rows) == len(registry.FIGURES)
        satisfied = {r["key"] for r in rows if r["satisfied"]}
        assert satisfied == set(SCHEMATIC_KEYS)
        for row in rows:
            if not row["satisfied"]:
                assert row["blocking"], row["key"]


# ---------------------------------------------------------------------------
# the no-silent-skip policy
# ---------------------------------------------------------------------------


class TestNoSilentSkip:
    """A figure is either rendered and traceable, or it is an exception.

    Parametrizing over every registered key is deliberate: adding a new figure
    that quietly returns ``None`` when its data is missing fails here, which is
    the whole point of building a separate package rather than extending
    ``PLOT_REGISTRY``.
    """

    @pytest.mark.parametrize("key", sorted(registry.FIGURES))
    def test_strict_render_is_all_or_nothing(self, key, tmp_path, empty_manifest):
        out = tmp_path / "out"
        if key in SCHEMATIC_KEYS:
            pytest.skip("requires nothing; covered by tests/test_pub_figures.py")
        with pytest.raises((ProvenanceError, NotImplementedError)):
            registry.render(key, manifest=empty_manifest, out_dir=out,
                            strict=True, formats=("png",))
        assert not out.exists() or not list(out.rglob("*"))

    @pytest.mark.parametrize("key", sorted(registry.FIGURES))
    def test_allow_missing_render_still_writes_nothing_for_blocked_figures(
            self, key, tmp_path, empty_manifest, capsys):
        """Under ``--allow-missing`` the provenance gate opens; the design gate holds.

        Every quantitative figure in this repo is blocked twice over: it has no
        manifest entry, and its drawing function has no data to draw. Relaxing
        the first must not produce an empty or fabricated figure.
        """
        out = tmp_path / "out"
        if key in SCHEMATIC_KEYS:
            pytest.skip("renderable today; covered by tests/test_pub_figures.py")
        with pytest.raises(NotImplementedError) as exc:
            registry.render(key, manifest=empty_manifest, out_dir=out,
                            strict=False, formats=("png",))
        assert key in str(exc.value)
        assert "--verify" in str(exc.value)
        assert "DEGRADED" in capsys.readouterr().out
        assert not out.exists() or not list(out.rglob("*.png"))

    def test_a_footerless_degraded_figure_is_not_refused(self, tmp_path,
                                                         empty_manifest, capsys):
        """The footer is not what declares a figure degraded.

        The directory split, the stderr line, and the sidecar are, and all three
        survive a figure being cropped into a manuscript. Rendering without a
        footer must therefore fail only for the reason it would have failed
        anyway -- here, no artifacts at all.
        """
        with pytest.raises(NotImplementedError):
            registry.render("F04_headline_accuracy", manifest=empty_manifest,
                            out_dir=tmp_path / "out", strict=False, footer=False)
        assert "DEGRADED" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# sidecars and audit
# ---------------------------------------------------------------------------


class TestSidecar:
    @pytest.fixture
    def source(self, tmp_path) -> tuple[Manifest, FigureSource]:
        runs = [make_run(tmp_path, seed=s, version=2) for s in ("1", "2", "3")]
        mf = manifest_with(tmp_path, {
            "source_itr_records": [{"run": str(r.relative_to(tmp_path))}
                                   for r in runs]})
        return mf, mf.resolve("F12_source_itr_void", strict=True)

    def test_round_trip_carries_every_required_key(self, source, tmp_path):
        _, resolved = source
        out = tmp_path / "out"
        write_sidecar(out, resolved, metric_space="kelvin",
                      metric_definition={"quantity": "pooled RMSE"},
                      selection={"sim_id": 7}, params={"benchmark": "source_itr"})
        payload = read_sidecar(out, "F12_source_itr_void")
        required = {"schema", "figure_key", "generated_utc", "generator_git_commit",
                    "metric_space", "metric_definition", "requirement_sets", "runs",
                    "sources", "counts", "selection", "degradations", "params"}
        assert required <= set(payload)
        assert payload["schema"] == "visual.pub.provenance/1"
        assert payload["counts"] == {"n_simulations": 9, "n_pairs": 36,
                                     "n_seeds": 3}

    def test_sources_carry_hashes(self, source, tmp_path):
        _, resolved = source
        out = tmp_path / "out"
        write_sidecar(out, resolved, metric_space="kelvin")
        payload = read_sidecar(out, "F12_source_itr_void")
        assert len(payload["sources"]) == 3
        for entry in payload["sources"]:
            assert len(entry["sha256"]) == 64
            assert entry["sha256"] == sha256_file(entry["path"])

    def test_missing_sidecar_reads_as_none(self, tmp_path):
        assert read_sidecar(tmp_path, "F01_problem_schematic") is None

    def test_audit_reports_drift_after_one_byte_changes(self, source, tmp_path):
        """Hash sensitivity: a figure must not silently outlive its inputs."""
        _, resolved = source
        out = tmp_path / "out"
        write_sidecar(out, resolved, metric_space="kelvin")
        assert all(f["status"] == "OK" for f in audit(out))

        target = resolved.artifacts[0].path
        with open(target, "a") as handle:
            handle.write("\n")
        findings = audit(out)
        drifted = [f for f in findings if f["status"] == "DRIFT"]
        assert len(drifted) == 1
        assert drifted[0]["path"] == target
        assert drifted[0]["expected_sha256"] != drifted[0]["actual_sha256"]

    def test_audit_reports_a_deleted_source(self, source, tmp_path):
        _, resolved = source
        out = tmp_path / "out"
        write_sidecar(out, resolved, metric_space="kelvin")
        __import__("os").remove(resolved.artifacts[0].path)
        assert any(f["status"] == "MISSING" for f in audit(out))


# ---------------------------------------------------------------------------
# schema detection
# ---------------------------------------------------------------------------


class TestSchema:
    def test_v1_column_set_detects_as_v1(self):
        assert records_mod.detect_schema_version(
            records_mod.SCHEMA_V1_COLUMNS) == 1

    def test_full_field_list_detects_as_v2_or_better(self):
        """``TEST_RECORD_FIELDS`` is imported from the eval module, not copied.

        If the writer gains the pooled sufficient statistics this must follow
        automatically; that is why the constant is not restated here.
        """
        version = records_mod.detect_schema_version(
            records_mod.TEST_RECORD_FIELDS)
        assert version >= 2

    def test_v2_and_v3_ladder(self):
        v2 = list(records_mod.SCHEMA_V1_COLUMNS) + list(
            records_mod.SCHEMA_V2_REQUIRED)
        v3 = v2 + list(records_mod.SCHEMA_V3_REQUIRED)
        assert records_mod.detect_schema_version(v2) == 2
        assert records_mod.detect_schema_version(v3) == 3

    def test_a_frame_that_is_not_test_records_raises(self, tmp_path):
        """A mis-pointed manifest entry must fail at load, not draw an empty figure."""
        path = tmp_path / "not_records.csv"
        path.write_text("a,b,c\n1,2,3\n")
        with pytest.raises(records_mod.SchemaError):
            records_mod.load_test_records(path)

    def test_v1_load_is_marked_degraded(self, tmp_path):
        """``require_version`` is a soft floor: v1 loads, but carries its defect.

        Raising would make the one surviving real run unusable as a development
        fixture; returning it silently would let it produce unpooled numbers that
        read as if they were pooled. The degradation is the third option.
        """
        path = write_records(tmp_path / "test_records.csv", version=1)
        frame = records_mod.load_test_records(path, require_version=2)
        assert frame.schema_version == 1
        codes = {d.code for d in frame.degradations}
        assert "SCHEMA_V1_NO_POOLED_STATS" in codes

    def test_v1_at_its_own_floor_is_clean(self, tmp_path):
        path = write_records(tmp_path / "test_records.csv", version=1)
        frame = records_mod.load_test_records(path, require_version=1)
        assert frame.degradations == []

    def test_v2_load_is_clean(self, tmp_path):
        path = write_records(tmp_path / "test_records.csv", version=2)
        frame = records_mod.load_test_records(path, require_version=2)
        assert frame.schema_version == 2
        assert frame.degradations == []
        assert records_mod.count_simulations(frame.df) == 3

    def test_seed_is_read_from_the_run_path(self, tmp_path):
        run = make_run(tmp_path, seed="1234", version=2)
        frame = records_mod.load_test_records(run / "test_records.csv")
        assert frame.seed == "1234"


# ---------------------------------------------------------------------------
# the real artifact in this workspace
# ---------------------------------------------------------------------------


REAL_RUN = "runs/source_itr_smoke3/config0/seed42"


class TestRealArtifact:
    """The one surviving test_records.csv, used as an end-to-end fixture.

    It is schema v1 at a single seed, so it must fail a v2 requirement in both
    of the ways the manifest can report. Skipped rather than failed when the run
    is absent, since this file is workspace state, not version-controlled.
    """

    @pytest.fixture
    def run_dir(self):
        from visual.pub.manifest import PROJECT_ROOT
        path = PROJECT_ROOT / REAL_RUN
        if not (path / "test_records.csv").exists():
            pytest.skip(f"{REAL_RUN} is not present in this workspace")
        return path

    def test_detects_as_schema_v1(self, run_dir):
        header = pd.read_csv(run_dir / "test_records.csv", nrows=0).columns
        assert records_mod.detect_schema_version(header) == 1

    def test_blocks_a_v2_figure_in_strict_mode(self, run_dir, tmp_path):
        from visual.pub.manifest import PROJECT_ROOT
        path = tmp_path / "manifest.yaml"
        path.write_text(yaml.safe_dump({"sources": {
            "source_itr_records": [{"run": REAL_RUN}]}}))
        mf = Manifest.load(path, root=PROJECT_ROOT)
        with pytest.raises(ProvenanceError, match="SCHEMA_BELOW_REQUIRED"):
            mf.resolve("F12_source_itr_void", strict=True)

    def test_degrades_with_both_codes(self, run_dir, tmp_path):
        from visual.pub.manifest import PROJECT_ROOT
        path = tmp_path / "manifest.yaml"
        path.write_text(yaml.safe_dump({"sources": {
            "source_itr_records": [{"run": REAL_RUN}]}}))
        mf = Manifest.load(path, root=PROJECT_ROOT)
        source = mf.resolve("F12_source_itr_void", strict=False)
        codes = {d.code for d in source.degradations}
        assert {"SCHEMA_BELOW_REQUIRED", "TOO_FEW_SEEDS"} <= codes
        assert source.n_seeds == 1
        assert source.artifacts[0].benchmarks == ("source_itr",)
