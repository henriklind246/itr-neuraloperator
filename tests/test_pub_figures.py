"""End-to-end render tests and the CLI contract for ``visual/pub``.

Three things are checked here that no other module can check.

**A tier-2 figure really renders.** F01, F02 and F25 require no run artifacts, so
they are the only figures that can be drawn against today's repo. Each must
produce a non-trivial PNG and PDF, a sidecar that validates, and no leaked
matplotlib figure. F25 additionally runs the manufactured-solution harness, so
it is the one test in the package that checks a *number* end to end: the fitted
convergence order must come out near the design order of 2.

**A blocked figure is blocked as a checked fact.** Every tier-1 and tier-3 key is
asserted to raise. Writing that down as a test rather than a comment is what
stops a half-built figure from shipping as an empty axes, and it turns the
question "which figures does this paper actually have?" into something the test
suite answers.

**The CLI honors its documented exit codes.** ``0`` clean, ``1`` provenance
failure, ``2`` degraded, with ``--list`` and ``--verify`` never importing torch.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import yaml

import visual.pub as pub
from visual.pub import __main__ as cli
from visual.pub import fields, fig_benchmark, fig_crossbench, registry
from visual.pub.manifest import SIDECAR_SCHEMA, Manifest, ProvenanceError

REPO_ROOT = Path(pub.__file__).resolve().parents[2]

# The only figures that require no run artifacts at all: they are analytic or
# they solve for their own data at render time. Everything else needs a manifest.
# Note this is not the same as "everything else is unbuilt" -- most of the
# registry now draws against real runs and is blocked only by the seed count, so
# it renders under --allow-missing. `--verify` is the live statement of which
# figures clear strict mode; these tests deliberately pin the no-manifest case.
RENDERABLE = ("F01_problem_schematic", "F02_architecture", "F25_mms_convergence")

# F25 solves the MMS ladder at render time; ~20 s against ~1 s for the others.
SLOW_KEYS = {"F25_mms_convergence"}

BLOCKED = tuple(sorted(set(registry.FIGURES) - set(RENDERABLE)))

SIDECAR_KEYS = {
    "schema", "figure_key", "generated_utc", "generator_git_commit",
    "generator_git_dirty", "metric_space", "metric_definition", "params",
    "requirement_sets", "runs", "sources", "counts", "selection", "degradations",
}


def test_source_difficulty_replaces_rel_l2_ecdf_with_contact_resistance():
    assert fig_benchmark.DIFFICULTY_STRATA["source"] == (
        "lead_bin", "regime", "patch_x_bin", "amplitude_bin", "R_c_bin")
    assert "source" not in fig_benchmark.DIFFICULTY_ECDF_BENCHMARKS
    assert fig_benchmark.DIFFICULTY_ECDF_BENCHMARKS == frozenset({
        "forcing", "source_itr", "interfaces"})


def test_contact_jump_cohort_is_reproducible(monkeypatch):
    bundle = SimpleNamespace(benchmark="forcing", n_sims=40, n_times=4)
    frame = pd.DataFrame({"sim_id": np.repeat(np.arange(30), 2)})

    class Case:
        lead_times = np.array([0.1, 0.2, 0.3])

        def __init__(self, sim_id):
            self.sim_id = sim_id

        def contact_jumps(self):
            truth = np.full((3, 5), float(self.sim_id))
            return truth, truth + 1.0

    monkeypatch.setattr(
        fields, "evaluate_case",
        lambda _bundle, sim_id, _source, targets: Case(sim_id),
    )
    first = fields.evaluate_contact_jump_cohort(
        bundle, frame, n_sims=8, rng_seed=7,
    )
    second = fields.evaluate_contact_jump_cohort(
        bundle, frame, n_sims=8, rng_seed=7,
    )

    assert first.sim_ids == second.sim_ids
    assert len(first.sim_ids) == 8
    assert first.target_indices == (1, 2, 3)
    assert first.truth.shape == first.pred.shape == (8, 3, 5)


def test_contact_jump_figure_uses_simulation_level_curves(monkeypatch):
    frames = {b: pd.DataFrame({"sim_id": [0, 1, 2]})
              for b in fig_crossbench.BENCH_ORDER}
    monkeypatch.setattr(fig_crossbench.records, "records_by_benchmark",
                        lambda _source: frames)
    monkeypatch.setattr(fields, "run_dirs", lambda _source, _benchmark: [Path("run")])
    monkeypatch.setattr(fields, "bundle", lambda _source, benchmark: benchmark)

    def cohort(benchmark, _records):
        truth = np.ones((3, 2, 4), dtype=float)
        pred = truth + 0.1
        return fields.ContactJumpCohort(
            benchmark=benchmark, sim_ids=(0, 1, 2), source_index=0,
            target_indices=(1, 2), lead_times=np.array([0.1, 0.2]),
            truth=truth, pred=pred,
        )

    monkeypatch.setattr(fields, "evaluate_contact_jump_cohort", cohort)
    fig, selection, definition = fig_crossbench.physical_contact_jump_vs_lead(
        source=object(), spec=SimpleNamespace(width="two_col"), requirement=None,
    )

    assert selection is None
    assert len(fig.axes) == 3
    assert definition["replication_unit"] == "sim_id"
    assert definition["requested_simulations_per_benchmark"] == 24
    assert set(definition["cohorts"]) == set(fig_crossbench.BENCH_ORDER)


@pytest.fixture(autouse=True)
def _no_leaked_figures():
    plt.close("all")
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def empty_manifest(tmp_path_factory) -> Manifest:
    """A manifest naming no runs -- which is the true state of this repo.

    Deliberately not the checked-in ``visual/pub/manifest.yaml``: that file does
    not exist yet, and a test that silently depended on a developer having
    created one locally would pass on one machine and fail on another.
    """
    return Manifest.load(tmp_path_factory.mktemp("mf") / "manifest.yaml")


def run_cli(*argv: str) -> tuple[int, str, str]:
    """Invoke the CLI in-process and capture its streams."""
    import contextlib
    import io

    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(list(argv))
    return code, out.getvalue(), err.getvalue()


def write_inverse_sensor_sweep(path: Path, benchmark: str) -> Path:
    rows = []
    for sim_id in range(8):
        truth = 0.4 + 0.02 * sim_id
        for count in (8, 16, 32):
            error = 0.01 * (sim_id + 1) * (8.0 / count)
            width = 0.12 * (8.0 / count) + 0.002 * sim_id
            fv = 0.04 * (8.0 / count) + 0.001 * sim_id
            row = {
                "benchmark": benchmark,
                "sim_id": sim_id,
                "n_sensors": count,
                "noise_seed": 100 + sim_id,
                "init_seed": 0,
                "noise_std_K": 0.05,
                "fv_resid_rms_K": fv,
                "fv_resid_over_noise": fv / 0.05,
                "profile_bound_limited": sim_id == 0 and count == 8,
            }
            if benchmark == "forcing":
                row.update({
                    "R_c_true": truth,
                    "R_c_map": truth + error,
                    "R_c_abs_error": error,
                    "profile_R_c_ci_low": truth - width / 2.0,
                    "profile_R_c_ci_high": truth + width / 2.0,
                })
            else:
                row.update({
                    "excess_int_true": truth,
                    "excess_int_hat": truth - error,
                    "excess_int_abserr": error,
                    "profile_excess_ci_low": truth - width / 2.0,
                    "profile_excess_ci_high": truth + width / 2.0,
                })
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def write_inverse_result(path: Path, benchmark: str) -> Path:
    rows = []
    for sim_id in range(10):
        if benchmark == "forcing":
            truth = 0.1 + 0.08 * sim_id
            estimate = truth + 0.01 * (sim_id + 1)
            rows.append({
                "benchmark": benchmark,
                "sim_id": sim_id,
                "n_sensors": 8,
                "R_c_true": truth,
                "R_c_map": estimate,
                "R_c_abs_error": abs(estimate - truth),
                "fno_vs_fv_resid": 0.002 * (sim_id + 1),
                "fno_resid": 0.003 * (sim_id + 1),
                "profile_R_c_covered": sim_id < 9,
            })
        else:
            row = {
                "benchmark": benchmark,
                "sim_id": sim_id,
                "cond_number": 2.0 + sim_id,
                "least_dir_R_base": 0.2,
                "least_dir_R_amp": 0.8,
                "least_dir_y0": 0.3,
                "least_dir_sigma": 0.45,
            }
            for offset, stem in enumerate(
                ("R_base", "R_amp", "y0", "sigma", "excess_int")
            ):
                truth = 0.1 * (offset + 1) + 0.01 * sim_id
                row[f"{stem}_true"] = truth
                row[f"{stem}_hat"] = truth + 0.005 * (sim_id + 1)
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


class TestRenderable:
    @pytest.mark.parametrize(
        "key",
        [pytest.param(k, marks=pytest.mark.slow) if k in SLOW_KEYS else k
         for k in RENDERABLE])
    def test_render_writes_both_formats(self, key, tmp_path, empty_manifest):
        out = tmp_path / "out"
        result = registry.render(key, manifest=empty_manifest, out_dir=out,
                                 strict=True, formats=("png", "pdf"))

        assert not result.degraded
        assert result.degradation_codes == []
        assert [p.name for p in result.paths] == [f"{key}.png", f"{key}.pdf"]
        for path in result.paths:
            assert path.exists()
            # A blank axes still renders a few kB; this catches an empty canvas.
            assert path.stat().st_size > 5000, f"{path.name} looks empty"

    @pytest.mark.parametrize(
        "key",
        [pytest.param(k, marks=pytest.mark.slow) if k in SLOW_KEYS else k
         for k in RENDERABLE])
    def test_sidecar_validates(self, key, tmp_path, empty_manifest):
        out = tmp_path / "out"
        result = registry.render(key, manifest=empty_manifest, out_dir=out,
                                 strict=True, formats=("png",))
        payload = json.loads(result.sidecar.read_text())

        assert set(payload) == SIDECAR_KEYS
        assert payload["schema"] == SIDECAR_SCHEMA
        assert payload["figure_key"] == key
        assert payload["degradations"] == []
        assert payload["metric_space"] in ("none", "kelvin", "normalized", "mixed")

    @pytest.mark.parametrize(
        "key",
        [pytest.param(k, marks=pytest.mark.slow) if k in SLOW_KEYS else k
         for k in RENDERABLE])
    def test_render_leaks_no_figures(self, key, tmp_path, empty_manifest):
        """``style.save`` closes the figure; a leak here becomes a memory leak
        when ``--all`` renders the full registered figure suite in one process."""
        registry.render(key, manifest=empty_manifest, out_dir=tmp_path,
                        strict=True, formats=("png",))
        assert plt.get_fignums() == []

    @pytest.mark.parametrize(
        "key",
        [pytest.param(k, marks=pytest.mark.slow) if k in SLOW_KEYS else k
         for k in RENDERABLE])
    def test_render_writes_nothing_outside_the_output_directory(
            self, key, tmp_path, empty_manifest):
        out = tmp_path / "out"
        before = {p for p in tmp_path.rglob("*")}
        registry.render(key, manifest=empty_manifest, out_dir=out, strict=True,
                        formats=("png",))
        new = {p for p in tmp_path.rglob("*")} - before
        assert new and all(out in p.parents or p == out for p in new)

    @pytest.mark.parametrize(
        "key",
        [pytest.param(k, marks=pytest.mark.slow) if k in SLOW_KEYS else k
         for k in RENDERABLE])
    def test_figure_size_matches_its_declared_width(self, key, tmp_path,
                                                    empty_manifest, monkeypatch):
        """The declared column width has to be the width actually drawn.

        A figure that says ``one_half_col`` and draws seven inches gets scaled by
        the typesetter, and every font size in it silently changes.
        """
        from visual.pub import style

        captured = {}
        real_save = style.save

        def spy(fig, out_dir, k, **kw):
            captured["width_in"] = fig.get_size_inches()[0]
            return real_save(fig, out_dir, k, **kw)

        monkeypatch.setattr(style, "save", spy)
        registry.render(key, manifest=empty_manifest, out_dir=tmp_path,
                        strict=True, formats=("png",))

        expected = style.WIDTHS_IN[registry.FIGURES[key].width]
        assert captured["width_in"] == pytest.approx(expected, abs=0.01)

    def test_footer_is_opt_in(self, tmp_path, empty_manifest):
        """Off by default; the sidecar, not the artwork, carries provenance."""
        for footer in (False, True):
            result = registry.render(
                "F01_problem_schematic", manifest=empty_manifest,
                out_dir=tmp_path / str(footer), strict=True, formats=("png",),
                footer=footer)
            assert result.paths[0].exists()

    def test_render_many_returns_one_result_per_key(self, tmp_path,
                                                    empty_manifest):
        results = registry.render_many(
            ["F01_problem_schematic", "F02_architecture"],
            manifest=empty_manifest, out_dir=tmp_path, strict=True,
            formats=("png",))
        assert [r.key for r in results] == ["F01_problem_schematic",
                                            "F02_architecture"]

    def test_f18_renders_paired_8_16_32_sweep_in_strict_mode(self, tmp_path):
        forcing = write_inverse_sensor_sweep(tmp_path / "forcing.csv", "forcing")
        forcing_itr = write_inverse_sensor_sweep(
            tmp_path / "forcing_itr.csv", "forcing_itr"
        )
        manifest_path = tmp_path / "manifest.yaml"
        manifest_path.write_text(yaml.safe_dump({
            "sources": {
                "inverse_sensor_sweep": [
                    {"table": str(forcing), "seed": "0"},
                    {"table": str(forcing_itr), "seed": "0"},
                ]
            }
        }))
        manifest = Manifest.load(manifest_path, root=tmp_path)
        result = registry.render(
            "F18_sensor_count", manifest=manifest, out_dir=tmp_path / "out",
            strict=True, formats=("png", "pdf"),
        )

        assert not result.degraded
        assert all(path.stat().st_size > 5000 for path in result.paths)
        definition = json.loads(result.sidecar.read_text())["metric_definition"]
        assert definition["sensor_counts"] == [8, 16, 32]
        assert definition["pairing_fields"] == [
            "benchmark", "sim_id", "noise_seed", "init_seed"
        ]
        assert definition["n_cases"] == {"forcing": 8, "forcing_itr": 8}
        assert definition["inferential_interval"].startswith("none")

    @pytest.mark.parametrize(
        ("key", "requirement", "benchmark"),
        [
            ("F16_inverse_forcing_recovery", "inverse_forcing", "forcing"),
            ("F17_inverse_forcing_itr", "inverse_forcing_itr", "forcing_itr"),
            ("F22_surrogate_fidelity", "inverse_forcing", "forcing"),
        ],
    )
    def test_inverse_publication_figures_render_without_legacy_module(
        self, tmp_path, key, requirement, benchmark
    ):
        table = write_inverse_result(tmp_path / f"{benchmark}.csv", benchmark)
        manifest_path = tmp_path / "manifest.yaml"
        manifest_path.write_text(yaml.safe_dump({
            "sources": {requirement: [{"table": str(table), "seed": "0"}]}
        }))
        manifest = Manifest.load(manifest_path, root=tmp_path)
        result = registry.render(
            key, manifest=manifest, out_dir=tmp_path / "out", strict=True,
            formats=("png",),
        )

        assert not result.degraded
        assert result.paths[0].stat().st_size > 5000
        definition = json.loads(result.sidecar.read_text())["metric_definition"]
        assert definition["n_cases"] == 10
        if key == "F17_inverse_forcing_itr":
            assert "least-determined direction" in definition["identifiability"]


class TestMmsNumbers:
    """F25 is the one figure whose content can be checked, not just its bytes.

    It regenerates its own data, so the fitted convergence orders in its sidecar
    are a genuine verification of the finite-volume interface discretization the
    training data comes from -- not a plot of a stored number.
    """

    @pytest.mark.slow
    def test_fitted_orders_are_near_second(self, tmp_path, empty_manifest):
        result = registry.render("F25_mms_convergence", manifest=empty_manifest,
                                 out_dir=tmp_path, strict=True, formats=("png",))
        definition = json.loads(result.sidecar.read_text())["metric_definition"]

        assert definition["design_order"] == 2
        assert definition["fitted_space_order"] == pytest.approx(2.0, abs=0.35)
        assert definition["fitted_time_order"] == pytest.approx(2.0, abs=0.35)

    @pytest.mark.slow
    def test_sidecar_names_the_harness_it_ran(self, tmp_path, empty_manifest):
        """A convergence claim has to say which solver produced it."""
        result = registry.render("F25_mms_convergence", manifest=empty_manifest,
                                 out_dir=tmp_path, strict=True, formats=("png",))
        definition = json.loads(result.sidecar.read_text())["metric_definition"]
        assert definition["mms_runner"] == "src.physics.mms_2d.run_mms_2d_interface"
        assert "Crank-Nicolson" in definition["solver"]


class TestBlocked:
    """A figure with nothing behind it must fail loudly and leave no trace.

    Parametrized over the twenty-two keys that need a manifest, against an empty
    one. This pins the no-silent-skip policy: a missing artifact raises and
    writes nothing, rather than producing a plausible-looking plot from whatever
    happened to be on disk. It says nothing about whether those figures are
    implemented -- most are, and render under --allow-missing. `--verify` is the
    live report of what clears strict mode.
    """

    @pytest.mark.parametrize("key", BLOCKED)
    def test_strict_render_raises_and_writes_nothing(self, key, tmp_path,
                                                     empty_manifest):
        out = tmp_path / "out"
        with pytest.raises((ProvenanceError, NotImplementedError)):
            registry.render(key, manifest=empty_manifest, out_dir=out,
                            strict=True, formats=("png", "pdf"))
        assert not out.exists() or not list(out.rglob("*"))

    @pytest.mark.parametrize("key", BLOCKED)
    def test_blocked_figures_leak_no_partial_state(self, key, tmp_path,
                                                   empty_manifest):
        with pytest.raises((ProvenanceError, NotImplementedError)):
            registry.render(key, manifest=empty_manifest, out_dir=tmp_path,
                            strict=True, formats=("png",))
        assert plt.get_fignums() == []

    def test_the_blocked_set_is_the_whole_registry_minus_three(self):
        """If this changes, a figure was built or a key was added; both are news."""
        assert len(BLOCKED) == len(registry.FIGURES) - 3
        assert len(registry.FIGURES) == 28

    def test_every_renderable_key_is_tier_two(self):
        for key in RENDERABLE:
            assert registry.FIGURES[key].tier == 2

    def test_no_tier_two_figure_is_blocked(self):
        """Tier 2 means "buildable today". A blocked tier-2 key is a mis-tiering."""
        tier2 = {k for k, s in registry.FIGURES.items() if s.tier == 2}
        assert tier2 == set(RENDERABLE)


class TestOverwriteGuard:
    def test_rerender_at_the_same_commit_is_allowed(self, tmp_path,
                                                    empty_manifest):
        for _ in range(2):
            registry.render("F01_problem_schematic", manifest=empty_manifest,
                            out_dir=tmp_path, strict=True, formats=("png",))

    def test_rerender_across_a_commit_change_is_refused(self, tmp_path,
                                                        empty_manifest):
        """Two figures in one paper must not come from two different commits."""
        key = "F01_problem_schematic"
        registry.render(key, manifest=empty_manifest, out_dir=tmp_path,
                        strict=True, formats=("png",))

        sidecar = tmp_path / f"{key}.provenance.json"
        payload = json.loads(sidecar.read_text())
        payload["generator_git_commit"] = "0" * 40
        sidecar.write_text(json.dumps(payload))

        with pytest.raises(ProvenanceError, match="--force"):
            registry.render(key, manifest=empty_manifest, out_dir=tmp_path,
                            strict=True, formats=("png",))

    def test_force_overrides_the_commit_guard(self, tmp_path, empty_manifest):
        key = "F01_problem_schematic"
        registry.render(key, manifest=empty_manifest, out_dir=tmp_path,
                        strict=True, formats=("png",))
        sidecar = tmp_path / f"{key}.provenance.json"
        payload = json.loads(sidecar.read_text())
        payload["generator_git_commit"] = "0" * 40
        sidecar.write_text(json.dumps(payload))

        result = registry.render(key, manifest=empty_manifest, out_dir=tmp_path,
                                 strict=True, formats=("png",), force=True)
        assert result.paths[0].exists()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCliList:
    def test_list_covers_every_figure(self):
        code, out, _ = run_cli("--list")
        assert code == cli.EXIT_OK
        for key in registry.FIGURES:
            assert key in out

    def test_list_json_is_machine_readable(self):
        code, out, _ = run_cli("--list", "--format", "json")
        rows = json.loads(out)
        assert code == cli.EXIT_OK
        assert {r["key"] for r in rows} == set(registry.FIGURES)
        assert all(r["tier"] in (1, 2, 3) for r in rows)

    def test_list_reports_the_blocking_requirements(self):
        """``--list`` is how a reader learns what a figure is waiting on."""
        rows = json.loads(run_cli("--list", "--format", "json")[1])
        by_key = {r["key"]: r for r in rows}
        assert by_key["F04_headline_accuracy"]["requires"]
        assert by_key["F01_problem_schematic"]["requires"] == []

    def test_list_does_not_import_torch(self):
        """A documentation command must not pay for a deep-learning import."""
        script = (
            "import sys; sys.argv = ['x', '--list', '--format', 'json']; "
            "from visual.pub.__main__ import main; main(sys.argv[1:]); "
            "sys.stderr.write('TORCH' if 'torch' in sys.modules else 'NOTORCH')"
        )
        proc = subprocess.run([sys.executable, "-c", script], cwd=REPO_ROOT,
                              capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
        assert proc.stderr.strip().endswith("NOTORCH")


class TestCliVerify:
    def test_verify_reports_the_repo_as_mostly_unsatisfiable(self, tmp_path):
        """The honest satisfiability table is the deliverable of the manifest layer.

        With no canonical runs in the workspace, only the three requirement-free
        figures can be satisfied. If this ever passes with a larger number, real
        runs have landed and the tier assignments should be revisited.
        """
        code, out, _ = run_cli("--verify", "--manifest", str(tmp_path / "none.yaml"))
        assert code == cli.EXIT_DEGRADED
        assert f"3/{len(registry.FIGURES)} figures satisfiable." in out

    def test_verify_json_marks_each_figure(self, tmp_path):
        code, out, _ = run_cli("--verify", "--format", "json",
                               "--manifest", str(tmp_path / "none.yaml"))
        rows = json.loads(out)
        assert code == cli.EXIT_DEGRADED
        satisfied = {r["key"] for r in rows if r["satisfied"]}
        assert satisfied == set(RENDERABLE)

    def test_verify_names_the_regeneration_command(self, tmp_path):
        rows = json.loads(run_cli("--verify", "--format", "json", "--manifest",
                                  str(tmp_path / "none.yaml"))[1])
        blocking = " ".join(
            b for r in rows for b in r["blocking"])
        assert "scripts/run_train_fixed.py" in blocking

    def test_verify_renders_nothing(self, tmp_path):
        out_dir = tmp_path / "out"
        run_cli("--verify", "--out", str(out_dir),
                "--manifest", str(tmp_path / "none.yaml"))
        assert not out_dir.exists()


class TestCliRender:
    def test_clean_render_exits_zero(self, tmp_path):
        code, _, _ = run_cli("--figure", "F01_problem_schematic",
                             "--out", str(tmp_path), "--formats", "png",
                             "--manifest", str(tmp_path / "none.yaml"))
        assert code == cli.EXIT_OK
        assert (tmp_path / "F01_problem_schematic.png").exists()

    def test_blocked_render_exits_one(self, tmp_path):
        """A blocked figure is a provenance failure, not a traceback."""
        code, _, err = run_cli("--figure", "F04_headline_accuracy",
                               "--out", str(tmp_path), "--formats", "png",
                               "--manifest", str(tmp_path / "none.yaml"))
        assert code == cli.EXIT_PROVENANCE
        assert "FAILED F04_headline_accuracy" in err

    def test_blocked_render_under_allow_missing_still_exits_one(self, tmp_path):
        """``--allow-missing`` relaxes provenance; it does not invent a figure."""
        code, _, err = run_cli("--figure", "F04_headline_accuracy",
                               "--allow-missing", "--out", str(tmp_path),
                               "--formats", "png",
                               "--manifest", str(tmp_path / "none.yaml"))
        assert code == cli.EXIT_PROVENANCE
        assert "FAILED F04_headline_accuracy" in err
        assert not list(tmp_path.rglob("*.png"))

    def test_unknown_key_is_rejected_before_anything_renders(self, tmp_path):
        with pytest.raises(KeyError, match="unknown figure key"):
            run_cli("--figure", "F99_nope", "--out", str(tmp_path),
                    "--manifest", str(tmp_path / "none.yaml"))
        assert not list(tmp_path.rglob("*.png"))

    def test_no_arguments_prints_help_and_exits_zero(self, tmp_path):
        code, out, _ = run_cli("--manifest", str(tmp_path / "none.yaml"))
        assert code == cli.EXIT_OK
        assert "--verify" in out


class TestCliAudit:
    def test_audit_reports_ok_for_a_fresh_render(self, tmp_path):
        run_cli("--figure", "F01_problem_schematic", "--out", str(tmp_path),
                "--formats", "png", "--manifest", str(tmp_path / "none.yaml"))
        code, out, _ = run_cli("--audit", str(tmp_path))
        assert code == cli.EXIT_OK
        assert "OK" in out

    def test_audit_of_an_empty_directory_says_so(self, tmp_path):
        code, out, _ = run_cli("--audit", str(tmp_path))
        assert code == cli.EXIT_OK
        assert "No rendered figures" in out
