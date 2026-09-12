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
import re
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

# The per-sim NPZ writer lives with the reader tests that pin its key set.
# F29/F30 have to consume the same bytes those tests validate, so it is shared
# rather than restated -- two synthetic writers drifting apart would let a
# figure pass against a layout the reader rejects.
from tests.test_pub_manifest import write_inverse_artifact

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
        "forcing", "source_itr_sin", "interfaces"})


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
                    "R_base_true": truth,
                    "R_base_hat": truth - error,
                    "R_base_abserr": error,
                    "profile_R_base_ci_low": truth - width / 2.0,
                    "profile_R_base_ci_high": truth + width / 2.0,
                })
            if benchmark != "forcing":
                row.update({
                    "A_true": row["R_base_true"], "A_hat": row["R_base_hat"],
                    "A_abserr": row["R_base_abserr"],
                    "profile_A_ci_low": row["profile_R_base_ci_low"],
                    "profile_A_ci_high": row["profile_R_base_ci_high"],
                })
            limited = row.pop("profile_bound_limited")
            for name in (("R_c",) if benchmark == "forcing" else ("R_base", "A")):
                error_col = f"{name}_abs_error" if name == "R_c" else f"{name}_abserr"
                row[f"{name}_rel_error_pct"] = 100 * row[error_col] / abs(row[f"{name}_true"])
                row[f"profile_{name}_bound_limited"] = limited
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
                "least_dir_A": 0.8,
            }
            for offset, stem in enumerate(
                ("R_base", "A")
            ):
                truth = 0.1 * (offset + 1) + 0.01 * sim_id
                row[f"{stem}_true"] = truth
                row[f"{stem}_hat"] = truth + 0.005 * (sim_id + 1)
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# The case ``stats.representative_inverse_case`` must pick out of the synthetic
# sweep above. Both parameter errors are -error with error = 0.01*(sim_id+1)*
# (8/n_sensors), so the profile L2 is error*sqrt(1 + 2*(2/pi) + 1/2): monotone
# in sim_id at every arm. The ranking is therefore the sim order whichever arm
# is used, and the lower median of eight cases is index 3.
REPRESENTATIVE_SIM = 3


def write_itr_sin_sweep_with_artifacts(root: Path, *, joint: bool = True,
                                       jacobian: bool = True,
                                       artifacts: bool = True) -> Path:
    """A forcing_itr_sin sweep plus the per-sim NPZ tree F30 descends into.

    The NPZ is placed at whatever ``records.inverse_artifact_path`` derives from
    the CSV rather than at a path this test picks, so the layout contract with
    ``run_inverse_sensor_sweep.py`` is what gets exercised.

    ``theta`` matches the CSV row for the representative case at the 8-sensor
    arm. Nothing in the figure cross-checks the two, but a fixture that
    disagreed with itself would make any later mismatch unreadable.
    """
    root.mkdir(parents=True, exist_ok=True)
    csv = write_inverse_sensor_sweep(root / "inverse_sensor_sweep.csv",
                                     "forcing_itr_sin")
    if not artifacts:
        return csv

    from visual.pub import records

    truth = 0.4 + 0.02 * REPRESENTATIVE_SIM
    for count in records.INVERSE_SENSOR_COUNTS:
        estimate = truth - 0.01 * (REPRESENTATIVE_SIM + 1) * (8.0 / count)
        path = records.inverse_artifact_path(csv, n_sensors=count,
                                             sim_id=REPRESENTATIVE_SIM)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_inverse_artifact(
            path, joint=joint,
            theta_hat=(estimate, estimate), theta_true=(truth, truth),
            # None is dropped by the writer, which is how the "no measured grid
            # and no sensitivity either" case is built.
            observation_jacobian=(
                np.asarray([[1.0, 0.2], [0.3, 1.4], [0.9, 0.1]])
                if jacobian else None
            ),
        )
    return csv


def itr_sin_manifest(tmp_path: Path, csv: Path) -> Manifest:
    """A manifest for the ``inverse_sensor_sweep_itr_sin`` requirement set.

    One CSV, because that set declares one benchmark. F29 and F30 read no row of
    the forcing arm, so requiring it would block them on a sweep they do not
    depend on -- which is exactly what lets ``run_inverse_sensor_sweep.py``
    render them strictly at the end of a single-benchmark run.
    """
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump({
        "sources": {"inverse_sensor_sweep_itr_sin": [
            {"table": str(csv), "seed": "0"},
        ]}
    }))
    return Manifest.load(manifest_path, root=tmp_path)


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
        forcing_itr_sin = write_inverse_sensor_sweep(
            tmp_path / "forcing_itr_sin.csv", "forcing_itr_sin"
        )
        manifest_path = tmp_path / "manifest.yaml"
        manifest_path.write_text(yaml.safe_dump({
            "sources": {
                "inverse_sensor_sweep": [
                    {"table": str(forcing), "seed": "0"},
                    {"table": str(forcing_itr_sin), "seed": "0"},
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
        assert definition["n_cases"] == {"forcing": 8, "forcing_itr_sin": 8}
        assert definition["inferential_interval"].startswith("none")

    @pytest.mark.parametrize(
        ("key", "requirement", "benchmark"),
        [
            ("F16_inverse_forcing_recovery", "inverse_forcing", "forcing"),
            ("F17_inverse_forcing_itr_sin", "inverse_forcing_itr_sin", "forcing_itr_sin"),
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
        if key == "F17_inverse_forcing_itr_sin":
            assert "least-determined direction" in definition["identifiability"]


class TestInverseProfileFigures:
    """F29 and F30 show one case, so the case had better not be chosen by hand.

    Both figures are structural: they make the inverse errors legible rather
    than re-establishing them. That only holds if the case is fixed by a rule
    before anything is drawn, so the rule is pinned here, and so is the
    separation of concerns that makes these two figures honest -- F29 draws no
    uncertainty band, F30 labels an approximated joint region as approximated.

    Everything runs against synthetic tables: no ``forcing_itr_sin`` sweep
    exists on disk yet, and waiting for one would mean shipping these untested.
    """

    F29 = "F29_inverse_rc_profile_recovery"
    F30 = "F30_inverse_identifiability"

    def draw(self, key: str, manifest: Manifest, **params):
        """Call the drawing function directly, to inspect the axes it made."""
        from visual.pub import style

        spec = registry.FIGURES[key]
        source = manifest.resolve(key, strict=True)
        with style.pub_style():
            return spec.load()(
                source=source, spec=spec,
                requirement=manifest.requirements.figures[key],
                **{**spec.params, **params},
            )

    # ------------------------------------------------------------- F29

    def test_f29_renders_one_case_at_three_sensor_counts(self, tmp_path):
        csv = write_itr_sin_sweep_with_artifacts(tmp_path / "sweep")
        result = registry.render(
            self.F29, manifest=itr_sin_manifest(tmp_path, csv),
            out_dir=tmp_path / "out", strict=True, formats=("png",),
        )

        assert not result.degraded
        assert result.paths[0].stat().st_size > 5000
        definition = json.loads(result.sidecar.read_text())["metric_definition"]
        assert definition["n_cases"] == 1
        assert definition["sim_id"] == REPRESENTATIVE_SIM
        assert definition["sensor_counts"] == [8, 16, 32]

    def test_f29_case_is_the_pre_registered_median_not_the_best(self, tmp_path):
        """A figure that picked its own best case would be advertising, not evidence."""
        from visual.pub import records, stats

        csv = write_itr_sin_sweep_with_artifacts(tmp_path / "sweep")
        manifest = itr_sin_manifest(tmp_path, csv)
        table = records.load_inverse_sensor_sweep(
            manifest.resolve(self.F29), benchmarks=("forcing_itr_sin",))
        # Default arm on purpose: the point is that the figure applies the
        # module's rule, not that it agrees with a rule this test invented.
        expected = stats.representative_inverse_case(
            table, benchmark="forcing_itr_sin")

        _, selection, definition = self.draw(self.F29, manifest)

        assert selection["sim_id"] == expected == REPRESENTATIVE_SIM
        # The synthetic errors rise with sim_id, so the best case is sim 0.
        assert expected != 0

    def test_f29_panels_share_one_axis_and_one_true_curve(self, tmp_path):
        """The shrinking gap is the whole message; rescaling per panel erases it."""
        csv = write_itr_sin_sweep_with_artifacts(tmp_path / "sweep")
        fig, _, _ = self.draw(self.F29, itr_sin_manifest(tmp_path, csv))

        axes = fig.axes
        assert len(axes) == 3
        assert all(len(ax.lines) == 2 for ax in axes)
        assert len({ax.get_ylim() for ax in axes}) == 1

        truth = [ax.lines[0].get_ydata() for ax in axes]
        assert all(np.array_equal(truth[0], other) for other in truth[1:])

    def test_f29_and_f30_are_stacked_single_column_figures(self, tmp_path):
        """Both set into one column of a two-column page, so both are 3x1.

        Pinned because the layout is a typesetting constraint the figure cannot
        express: a 1x3 renders perfectly well and would only be caught at
        submission, by which point the panels are illegible at column width.
        """
        from visual.pub import style

        csv = write_itr_sin_sweep_with_artifacts(tmp_path / "sweep")
        manifest = itr_sin_manifest(tmp_path, csv)

        for key in (self.F29, self.F30):
            fig, _, _ = self.draw(key, manifest)
            assert registry.FIGURES[key].width == "one_col"
            assert fig.get_size_inches()[0] == pytest.approx(
                style.WIDTHS_IN["one_col"])
            # A colorbar is an axes, and it inherits a subplotspec from the
            # host it was split off, so it has to be excluded by name.
            panels_ = [ax for ax in fig.axes if ax.get_label() != "<colorbar>"]
            assert len(panels_) == 3
            rows, cols = panels_[0].get_subplotspec().get_gridspec().get_geometry()
            assert (rows, cols) == (3, 1)
            assert fig.get_size_inches()[1] > fig.get_size_inches()[0]

    def test_f29_labels_y_once_at_the_bottom(self, tmp_path):
        """A shared x axis repeated three times is three times the ink."""
        csv = write_itr_sin_sweep_with_artifacts(tmp_path / "sweep")
        fig, _, _ = self.draw(self.F29, itr_sin_manifest(tmp_path, csv))

        assert [ax.get_xlabel() for ax in fig.axes] == ["", "", "$y$"]
        # The R_c axis is *not* shared, so every panel keeps its own label.
        assert all(ax.get_ylabel() for ax in fig.axes)

    def test_f29_recovered_peak_is_the_recovered_r_base_plus_a(self, tmp_path):
        """R_c(y) peaks at R_base + A; if it does not, the curve is not the model."""
        csv = write_itr_sin_sweep_with_artifacts(tmp_path / "sweep")
        fig, _, _ = self.draw(self.F29, itr_sin_manifest(tmp_path, csv))

        truth = 0.4 + 0.02 * REPRESENTATIVE_SIM
        peaks = [float(ax.lines[1].get_ydata().max()) for ax in fig.axes]
        for peak, count in zip(peaks, (8, 16, 32)):
            estimate = truth - 0.01 * (REPRESENTATIVE_SIM + 1) * (8.0 / count)
            assert peak == pytest.approx(2 * estimate, abs=1e-6)

        # More sensors, closer to the truth: the figure's entire claim.
        assert peaks[0] < peaks[1] < peaks[2] < 2 * truth

    def test_f29_draws_no_uncertainty_band(self, tmp_path):
        """Two marginal intervals cannot be propagated into a joint band."""
        csv = write_itr_sin_sweep_with_artifacts(tmp_path / "sweep")
        fig, _, definition = self.draw(self.F29, itr_sin_manifest(tmp_path, csv))

        # A band would be a PolyCollection from fill_between; two Line2D and
        # nothing else is the whole content of every panel.
        assert all(len(ax.collections) == 0 for ax in fig.axes)
        assert definition["inferential_interval"].startswith("none")
        assert "F30" in definition["no_uncertainty_band"]

    def test_f29_is_blocked_when_only_the_forcing_arm_is_supplied(self, tmp_path):
        """The sweep spans two benchmarks; only one of them carries R_base and A.

        Built as a bare source rather than through the manifest: under
        ``--allow-missing`` a half-populated sweep reaches the drawing function,
        and it has to name the arm it is missing instead of dying inside pandas.
        """
        from visual.pub import fig_inverse

        csv = write_inverse_sensor_sweep(tmp_path / "forcing.csv", "forcing")
        source = SimpleNamespace(artifacts=[SimpleNamespace(
            kind="inverse_csv", path=str(csv), benchmarks=("forcing",))])

        with pytest.raises(NotImplementedError, match="forcing_itr_sin"):
            fig_inverse.rc_profile_recovery(
                source=source, spec=registry.FIGURES[self.F29], requirement=None)

    def test_f29_reads_the_itr_sin_arm_out_of_a_combined_table(self, tmp_path):
        """One CSV carrying both benchmarks is a documented layout; honour it."""
        from visual.pub import fig_inverse, style

        root = tmp_path / "sweep"
        csv = write_itr_sin_sweep_with_artifacts(root)
        combined = pd.concat([
            pd.read_csv(write_inverse_sensor_sweep(tmp_path / "f.csv", "forcing")),
            pd.read_csv(csv),
        ], ignore_index=True, sort=False)
        combined.to_csv(csv, index=False)

        source = SimpleNamespace(artifacts=[SimpleNamespace(
            kind="inverse_csv", path=str(csv),
            benchmarks=("forcing", "forcing_itr_sin"))])
        with style.pub_style():
            _, selection, definition = fig_inverse.rc_profile_recovery(
                source=source, spec=registry.FIGURES[self.F29], requirement=None)

        assert selection["sim_id"] == REPRESENTATIVE_SIM
        assert definition["n_cases"] == 1

    # ------------------------------------------------------------- F30

    def test_f30_prefers_the_measured_joint_grid(self, tmp_path):
        csv = write_itr_sin_sweep_with_artifacts(tmp_path / "sweep", joint=True)
        result = registry.render(
            self.F30, manifest=itr_sin_manifest(tmp_path, csv),
            out_dir=tmp_path / "out", strict=True, formats=("png",),
        )

        definition = json.loads(result.sidecar.read_text())["metric_definition"]
        assert "measured" in definition["joint_region_source"]
        assert "approximation" not in definition["joint_region_source"]
        assert "JOINT_REGION_LAPLACE_APPROXIMATION" not in definition["degradations"]
        assert definition["sim_id"] == REPRESENTATIVE_SIM
        # The densest arm: F30 shows the method at its intended operating point,
        # so a tilted region there is a property of the parameterization rather
        # than of thin data. Read off the spec so the two cannot drift apart.
        assert definition["n_sensors"] == 32
        assert registry.FIGURES[self.F30].params["n_sensors"] == 32

    def test_f30_falls_back_to_the_quadratic_and_says_so(self, tmp_path):
        """An assumed ellipse must not be presentable as a measured shape."""
        csv = write_itr_sin_sweep_with_artifacts(tmp_path / "sweep", joint=False)
        fig, _, definition = self.draw(self.F30, itr_sin_manifest(tmp_path, csv))

        assert "approximation" in definition["joint_region_source"]
        assert "measured" not in definition["joint_region_source"]
        assert "JOINT_REGION_LAPLACE_APPROXIMATION" in definition["degradations"]
        assert "approx" in fig.axes[2].get_title()

    def test_f30_is_blocked_without_a_grid_or_a_sensitivity(self, tmp_path):
        csv = write_itr_sin_sweep_with_artifacts(
            tmp_path / "sweep", joint=False, jacobian=False)
        with pytest.raises(NotImplementedError, match="joint"):
            self.draw(self.F30, itr_sin_manifest(tmp_path, csv))

    def test_f30_is_blocked_when_the_per_sim_artifact_is_absent(self, tmp_path):
        csv = write_itr_sin_sweep_with_artifacts(tmp_path / "sweep",
                                                 artifacts=False)
        with pytest.raises(NotImplementedError, match="artifact"):
            self.draw(self.F30, itr_sin_manifest(tmp_path, csv))

    def test_f30_rejects_a_sensor_count_the_sweep_never_ran(self, tmp_path):
        csv = write_itr_sin_sweep_with_artifacts(tmp_path / "sweep")
        with pytest.raises(ValueError, match="12"):
            self.draw(self.F30, itr_sin_manifest(tmp_path, csv), n_sensors=12)

    def test_f30_uses_the_two_parameter_threshold_for_the_joint_region(self, tmp_path):
        """chi2(0.95, 2)/2, not the marginal chi2(0.95, 1)/2 reused twice."""
        csv = write_itr_sin_sweep_with_artifacts(tmp_path / "sweep")
        _, _, definition = self.draw(self.F30, itr_sin_manifest(tmp_path, csv))

        # The definitions are prose, so the number is read back out of them --
        # the claim under test is that the two thresholds carry different
        # degrees of freedom, and a definition that said "1" while the panel
        # drew 2 would be the failure worth catching.
        marginal = definition["marginal_threshold"]
        joint = definition["joint_threshold"]
        assert "chi2.ppf(0.95, 1) / 2" in marginal
        assert "chi2.ppf(0.95, 2) / 2" in joint
        assert float(re.search(r"= ([\d.]+);", marginal).group(1)) \
            == pytest.approx(1.92, abs=0.01)
        assert float(re.search(r"= ([\d.]+);", joint).group(1)) \
            == pytest.approx(3.00, abs=0.01)

    def test_f29_and_f30_describe_the_same_inversion(self, tmp_path):
        """Two figures about one case are only comparable if it is one case."""
        csv = write_itr_sin_sweep_with_artifacts(tmp_path / "sweep")
        manifest = itr_sin_manifest(tmp_path, csv)

        _, recovery_selection, _ = self.draw(self.F29, manifest)
        _, joint_selection, _ = self.draw(self.F30, manifest)

        assert recovery_selection["sim_id"] == joint_selection["sim_id"]


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
        assert len(registry.FIGURES) == 30

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
