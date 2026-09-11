"""Style contract and architectural guards for ``visual/pub``.

Two halves.

The first half pins the parts of ``style.py`` that a reader of the printed paper
depends on and cannot check for themselves: that a residual color scale is
symmetric about zero, that no axis can ship without a unit, and that a named
column width really is that many inches.

The second half is a set of AST and import guards on the package as a whole. The
plan for this package assigns each module one job -- ``records`` loads,
``stats`` reduces, ``panels`` draws, ``fig_*`` composes -- and the reason for
that split is the defect the package exists to fix: the previous paper figures
computed their own means over pair-level rows, which is where the
pseudo-replication entered. A grep would find today's violations; an AST walk
finds tomorrow's. :class:`TestNoInlineReduction` is the single most valuable
test in this file, and its whitelist is deliberately empty.
"""

from __future__ import annotations

import ast
import importlib
import pkgutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import visual.pub as pub
from visual.pub import manifest as manifest_mod
from visual.pub import registry, style

PUB_DIR = Path(pub.__file__).resolve().parent
REPO_ROOT = PUB_DIR.parents[1]


@pytest.fixture(autouse=True)
def _no_leaked_figures():
    """Start every test with a clean figure registry.

    Without this a failure that skips its own ``plt.close`` makes the next test
    that counts open figures fail too, which hides the real defect.
    """
    plt.close("all")
    yield
    plt.close("all")


def pub_sources() -> list[Path]:
    return sorted(p for p in PUB_DIR.glob("*.py") if p.name != "__init__.py")


def parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


# ---------------------------------------------------------------------------
# Color
# ---------------------------------------------------------------------------


class TestColorLimits:
    def test_diverging_limits_are_symmetric_about_zero(self):
        """A signed residual drawn on an asymmetric scale reads as biased.

        The eye maps color saturation to magnitude, so if ``vmin`` and ``vmax``
        differ in magnitude an equal over- and under-prediction print as
        different-sized errors.
        """
        field = np.array([-1.0, 0.5, 8.0, -3.0])
        lim = style.shared_color_limits(field, mode="diverging", robust=False)
        assert lim.vcenter == 0.0
        assert lim.vmin == pytest.approx(-lim.vmax)
        assert lim.vmax == pytest.approx(8.0)

    def test_diverging_uses_the_residual_colormap(self):
        lim = style.shared_color_limits(np.array([-1.0, 1.0]), mode="diverging")
        assert lim.cmap == style.CMAP_RESIDUAL

    def test_diverging_stays_symmetric_for_one_signed_input(self):
        """All-positive residuals must still center on zero, not on their own min."""
        lim = style.shared_color_limits(np.array([2.0, 3.0, 4.0]), mode="diverging",
                                        robust=False)
        assert lim.vmin == pytest.approx(-4.0)
        assert lim.vmax == pytest.approx(4.0)
        assert lim.vcenter == 0.0

    def test_sequential_spans_every_field(self):
        """Truth and prediction share a scale only if both are pooled first."""
        truth = np.array([300.0, 310.0])
        pred = np.array([295.0, 320.0])
        lim = style.shared_color_limits(truth, pred, mode="sequential", robust=False)
        assert lim.vmin == pytest.approx(295.0)
        assert lim.vmax == pytest.approx(320.0)
        assert lim.vcenter is None

    def test_sequential_of_one_field_is_not_the_shared_scale(self):
        """Guards the per-row-norm pattern this function replaces.

        The legacy field renderer normalized each row on
        its own values, so two panels printed side by side used different scales.
        """
        truth = np.array([300.0, 310.0])
        pred = np.array([295.0, 320.0])
        alone = style.shared_color_limits(truth, mode="sequential", robust=False)
        shared = style.shared_color_limits(truth, pred, mode="sequential",
                                           robust=False)
        assert alone.vmax != shared.vmax

    def test_constant_field_does_not_produce_a_degenerate_scale(self):
        lim = style.shared_color_limits(np.full(8, 300.0), mode="sequential")
        assert lim.vmax > lim.vmin

    def test_nonfinite_values_are_dropped(self):
        field = np.array([1.0, np.nan, np.inf, 5.0])
        lim = style.shared_color_limits(field, mode="sequential", robust=False)
        assert lim.vmin == pytest.approx(1.0)
        assert lim.vmax == pytest.approx(5.0)

    def test_an_all_nan_field_raises(self):
        with pytest.raises(ValueError, match="no finite values"):
            style.shared_color_limits(np.full(4, np.nan))

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown color-limit mode"):
            style.shared_color_limits(np.array([1.0]), mode="loglog")

    def test_imshow_kwargs_carry_a_centered_norm_when_diverging(self):
        lim = style.shared_color_limits(np.array([-2.0, 2.0]), mode="diverging")
        kw = lim.imshow_kwargs()
        assert "norm" in kw and "vmin" not in kw and "vmax" not in kw
        assert kw["norm"].vcenter == 0.0

    def test_imshow_kwargs_are_plain_limits_when_sequential(self):
        lim = style.shared_color_limits(np.array([1.0, 2.0]), mode="sequential")
        kw = lim.imshow_kwargs()
        assert "norm" not in kw
        assert kw["vmin"] < kw["vmax"]


class TestBenchmarkColors:
    def test_every_benchmark_has_a_distinct_color(self):
        """A family keeps one color across the whole paper only if it is unique."""
        colors = list(style.BENCHMARK_COLORS.values())
        assert len(set(colors)) == len(colors)

    def test_unknown_benchmark_falls_back_to_neutral(self):
        assert style.benchmark_color("not_a_benchmark") == "0.45"

    def test_registered_benchmarks_keep_their_color(self):
        for name, color in style.BENCHMARK_COLORS.items():
            assert style.benchmark_color(name) == color


# ---------------------------------------------------------------------------
# Units and labels
# ---------------------------------------------------------------------------


class TestLabels:
    def test_axis_label_raises_on_an_unknown_metric(self):
        """An unlabeled or mis-united axis must not be able to ship."""
        with pytest.raises(KeyError, match="No label registered"):
            style.axis_label("rmse")

    def test_kelvin_metrics_carry_their_unit(self):
        assert style.axis_label("rmse_K").endswith("[K]")

    def test_percent_metrics_carry_their_unit(self):
        assert style.axis_label("rel_l2_pct").endswith("[%]")

    def test_dimensionless_metrics_carry_no_bracket(self):
        assert "[" not in style.axis_label("t_bar")

    def test_every_labelled_metric_has_a_unit(self):
        missing = sorted(set(style.LABELS) - set(style.UNITS))
        assert missing == []

    def test_every_united_metric_has_a_label(self):
        missing = sorted(set(style.UNITS) - set(style.LABELS))
        assert missing == []

    def test_metric_space_is_declared_for_every_error_metric(self):
        """Only the covariates -- lead time, ``R_c``, ``x_I`` -- may lack a space.

        Everything that could land on an error axis must declare kelvin or
        normalized, because that declaration is what stops a normalized training
        curve being co-plotted with a physical Kelvin eval metric.
        """
        covariates = {"t_bar", "t_s", "R_c", "x_I"}
        undeclared = sorted(set(style.LABELS) - set(style.METRIC_SPACE) - covariates)
        assert undeclared == []

    def test_metric_spaces_are_from_the_allowed_set(self):
        allowed = set(manifest_mod.METRIC_SPACES)
        assert set(style.METRIC_SPACE.values()) <= allowed

    def test_kelvin_metrics_agree_with_their_unit(self):
        """The ``_K`` suffix, the ``K`` unit, and ``kelvin`` must not disagree."""
        for metric, space in style.METRIC_SPACE.items():
            if space == "kelvin":
                assert style.UNITS[metric] == "K", metric
            elif space == "normalized":
                assert style.UNITS[metric] == "%", metric


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------


class TestSizing:
    def test_figsize_matches_the_named_width(self):
        for name, inches in style.WIDTHS_IN.items():
            assert style.figsize(name)[0] == pytest.approx(inches)

    def test_rows_multiply_the_row_height(self):
        one = style.figsize("two_col", rows=1, row_height="std")[1]
        three = style.figsize("two_col", rows=3, row_height="std")[1]
        assert three == pytest.approx(3 * one)

    def test_extra_inches_are_added_after_the_row_product(self):
        base = style.figsize("one_col", rows=2, row_height="short")[1]
        padded = style.figsize("one_col", rows=2, row_height="short",
                               extra_in=0.4)[1]
        assert padded == pytest.approx(base + 0.4)

    def test_unknown_width_raises(self):
        with pytest.raises(KeyError, match="Unknown figure width"):
            style.figsize("half_col")

    def test_unknown_row_height_raises(self):
        with pytest.raises(KeyError, match="Unknown row height"):
            style.figsize("two_col", row_height="enormous")

    def test_registry_widths_are_all_sizeable(self):
        """``FigureSpec`` validates the name; this checks it also has a number."""
        for spec in registry.FIGURES.values():
            assert spec.width in style.WIDTHS_IN

    def test_registry_width_names_match_style(self):
        assert set(registry.WIDTH_NAMES) == set(style.WIDTHS_IN)

    def test_no_figure_is_wider_than_the_page(self):
        assert max(style.WIDTHS_IN.values()) == style.WIDTHS_IN["full_page"]


class TestRcParams:
    def test_fonts_are_embedded_as_truetype(self):
        """Type-3 text is not editable and several publishers reject it."""
        assert style.PUB_RCPARAMS["pdf.fonttype"] == 42
        assert style.PUB_RCPARAMS["ps.fonttype"] == 42

    def test_saved_output_is_at_print_resolution(self):
        assert style.PUB_RCPARAMS["savefig.dpi"] >= 300

    def test_image_interpolation_is_nearest(self):
        """A smoothed field map invents structure between cells."""
        assert style.PUB_RCPARAMS["image.interpolation"] == "nearest"

    def test_pub_style_is_scoped_to_the_context(self):
        before = matplotlib.rcParams["font.size"]
        with style.pub_style():
            assert matplotlib.rcParams["font.size"] == 8
            assert matplotlib.rcParams["pdf.fonttype"] == 42
        assert matplotlib.rcParams["font.size"] == before

    def test_extra_overrides_the_defaults(self):
        with style.pub_style(extra={"font.size": 11}):
            assert matplotlib.rcParams["font.size"] == 11

    def test_pub_rcparams_differs_from_the_screen_diagnostic_style(self):
        """These are separate on purpose; the screen style is not print-ready."""
        from visual._common import PLOT_STYLE

        assert PLOT_STYLE.get("font.size") != style.PUB_RCPARAMS["font.size"]


# ---------------------------------------------------------------------------
# Annotation and output
# ---------------------------------------------------------------------------


class TestAnnotation:
    def test_panel_letters_run_in_reading_order(self):
        fig, axes = plt.subplots(2, 2)
        style.panel_letters(axes)
        letters = [ax.texts[0].get_text() for ax in axes.ravel()]
        assert letters == ["(a)", "(b)", "(c)", "(d)"]
        plt.close(fig)

    def test_panel_letters_can_start_partway_through(self):
        fig, axes = plt.subplots(1, 2)
        style.panel_letters(axes, start=3)
        assert [ax.texts[0].get_text() for ax in axes] == ["(d)", "(e)"]
        plt.close(fig)


    def test_reference_line_sits_at_the_requested_value(self):
        fig, ax = plt.subplots()
        style.add_reference_line(ax, value=1.0)
        assert ax.lines[0].get_ydata()[0] == pytest.approx(1.0)
        plt.close(fig)

    def test_the_footer_never_draws_a_border(self):
        """A frame around the artwork is lost the moment the figure is cropped.

        Degradation is carried by the footer wording and by the separate output
        directory, both of which travel with the file.
        """
        fig = plt.figure()
        style.add_provenance_footer(fig, "F04_headline_accuracy")
        assert fig.patches == []
        plt.close(fig)

    def test_the_footer_is_neutral_grey(self):
        fig = plt.figure()
        style.add_provenance_footer(fig, "F04_headline_accuracy")
        assert fig.texts[0].get_color() == "0.45"
        plt.close(fig)

    def test_footer_always_names_the_figure_key(self):
        fig = plt.figure()
        style.add_provenance_footer(fig, "F06_signature")
        assert "F06_signature" in fig.texts[0].get_text()
        plt.close(fig)


class TestSave:
    def test_requested_formats_are_written_and_the_figure_is_closed(self, tmp_path):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        paths = style.save(fig, tmp_path, "F01_problem_schematic",
                           formats=("png", "pdf"))
        assert [p.name for p in paths] == ["F01_problem_schematic.png",
                                           "F01_problem_schematic.pdf"]
        assert all(p.stat().st_size > 1000 for p in paths)
        assert plt.get_fignums() == []

    def test_missing_parent_directories_are_created(self, tmp_path):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        (path,) = style.save(fig, tmp_path / "deep" / "nested", "F02_architecture",
                             formats=("png",))
        assert path.exists()


# ---------------------------------------------------------------------------
# Architectural guards
# ---------------------------------------------------------------------------

REDUCTION_FUNCS = {"mean", "median", "percentile", "quantile", "average",
                   "std", "var", "nanmean", "nanmedian", "nanpercentile",
                   "nanquantile", "nanstd"}

# Deliberately empty. A figure module that needs a reduction needs it in
# stats.py, where the resampling unit is a simulation rather than a snapshot
# pair. If this list ever grows, the pseudo-replication defect has come back.
REDUCTION_WHITELIST: dict[str, set[str]] = {}


class TestNoInlineReduction:
    """No ``fig_*`` module may reduce a data frame itself.

    The previous paper figures called ``np.mean``/``np.percentile`` directly on
    rows of ``test_records.csv``. Each row is one ``(sim_id, s, j)`` snapshot
    pair and a single simulation contributes ``S x J`` strongly correlated rows,
    so those reductions counted pseudo-replicates: every reported ``n`` was
    inflated and every interval was anticonservative. Routing every reduction
    through ``stats.py`` is what fixes it, and this test is what keeps it fixed.
    """

    @pytest.mark.parametrize(
        "path", [p for p in pub_sources() if p.name.startswith("fig_")],
        ids=lambda p: p.name)
    def test_figure_module_performs_no_reduction(self, path):
        offenders = []
        for node in ast.walk(parse(path)):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr in REDUCTION_FUNCS:
                offenders.append(f"{func.attr} at line {node.lineno}")
        allowed = REDUCTION_WHITELIST.get(path.name, set())
        unexpected = [o for o in offenders if o.split()[0] not in allowed]
        assert unexpected == [], (
            f"{path.name} reduces inline: {unexpected}. Every reduction belongs "
            f"in visual/pub/stats.py, where the resampling unit is a simulation."
        )

    def test_panels_module_performs_no_reduction(self):
        """``panels.py`` draws onto an axes; it receives numbers already reduced."""
        offenders = [
            f"{n.func.attr} at line {n.lineno}"
            for n in ast.walk(parse(PUB_DIR / "panels.py"))
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr in REDUCTION_FUNCS
        ]
        assert offenders == []

    def test_the_whitelist_is_empty(self):
        assert REDUCTION_WHITELIST == {}


class TestModuleOwnership:
    """Each module has one job, and the boundaries are checked rather than stated."""

    def test_manifest_does_not_import_matplotlib(self):
        """Provenance resolution must be cheap and must never open a display."""
        assert not self._imports(PUB_DIR / "manifest.py", "matplotlib")

    def test_stats_reads_no_files(self):
        """``stats.py`` takes frames. File access belongs to ``records.py``."""
        source = parse(PUB_DIR / "stats.py")
        assert not self._imports_any(source, {"pathlib", "os", "json", "csv"})
        readers = [
            n.func.attr for n in ast.walk(source)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr in {"read_csv", "read_json", "open", "read_text"}
        ]
        assert readers == []

    def test_stats_does_not_import_matplotlib(self):
        assert not self._imports(PUB_DIR / "stats.py", "matplotlib")

    def test_records_does_not_aggregate(self):
        """``records.py`` loads and validates; the reduction is a separate step."""
        offenders = [
            n.func.attr for n in ast.walk(parse(PUB_DIR / "records.py"))
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr in {"groupby", "agg", "pivot_table"}
        ]
        assert offenders == []

    def test_records_does_not_import_matplotlib(self):
        assert not self._imports(PUB_DIR / "records.py", "matplotlib")

    def test_select_does_not_import_matplotlib(self):
        assert not self._imports(PUB_DIR / "select.py", "matplotlib")

    def test_panels_does_not_load_files(self):
        source = parse(PUB_DIR / "panels.py")
        assert not self._imports_any(source, {"pathlib"})
        assert not self._imports(PUB_DIR / "panels.py", "visual.pub.records")

    def test_style_does_not_know_the_statistics_layer(self):
        """``style.py`` is presentation only; a cycle here would be a design smell."""
        for module in ("visual.pub.stats", "visual.pub.records",
                       "visual.pub.manifest", "visual.pub.select"):
            assert not self._imports(PUB_DIR / "style.py", module)

    @staticmethod
    def _imported_names(tree: ast.Module) -> set[str]:
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module)
                names.update(f"{node.module}.{a.name}" for a in node.names)
        return names

    @classmethod
    def _imports(cls, path: Path, module: str) -> bool:
        return any(n == module or n.startswith(module + ".")
                   for n in cls._imported_names(parse(path)))

    @classmethod
    def _imports_any(cls, tree: ast.Module, modules: set[str]) -> bool:
        names = cls._imported_names(tree)
        return any(n == m or n.startswith(m + ".") for n in names for m in modules)


class TestPackageBoundaries:
    def test_legacy_publication_modules_are_deleted_and_unreferenced(self):
        """The superseded pair-weighted publication paths must not return."""
        for filename in (
            "paper_plots.py",
            "inverse_plots.py",
            "rollout_plots.py",
            "resinv_plots.py",
        ):
            assert not (PUB_DIR.parent / filename).exists()
        offenders = [
            p.name for p in pub_sources()
            if any(TestModuleOwnership._imports(p, module) for module in (
                "visual.paper_plots",
                "visual.inverse_plots",
                "visual.rollout_plots",
                "visual.resinv_plots",
            ))
        ]
        assert offenders == []

    def test_every_module_imports_cleanly(self):
        """Catches a syntax error or a circular import in a module no test loads."""
        for info in pkgutil.iter_modules([str(PUB_DIR)]):
            importlib.import_module(f"visual.pub.{info.name}")

    def test_listing_figures_does_not_import_torch(self):
        """``--list`` and ``--verify`` must stay cheap; drawing modules load lazily."""
        source = parse(PUB_DIR / "registry.py")
        assert not TestModuleOwnership._imports_any(source, {"torch"})
        for spec in registry.FIGURES.values():
            assert spec.module.startswith("visual.pub.")


class TestSingleJumpImplementation:
    """The contact jump ``R_c(y) q_n(y, t)`` exists in exactly one place.

    The paper's claim rests on plotting the physical contact discontinuity rather
    than the adjacent-node difference the training loss uses. Two
    implementations of that formula is two chances to plot the wrong one, so
    ``visual/pub/jump.py`` imports the existing helper instead of restating it.
    """

    def test_jump_module_imports_rather_than_restates(self):
        assert TestModuleOwnership._imports(PUB_DIR / "jump.py",
                                            "visual.dataset_plots")

    def test_no_module_recomputes_the_conductance_formula(self):
        """``G(y) = 1 / (h_L/k_L + R_c + h_R/k_R)`` is written once, elsewhere."""
        offenders = []
        for path in pub_sources():
            text = path.read_text()
            if "k_left" in text and "k_right" in text and "/ (" in text:
                if path.name != "jump.py":
                    offenders.append(path.name)
        assert offenders == []

    def test_jump_users_go_through_the_jump_module(self):
        """No figure may reach past ``jump.py`` into ``dataset_plots`` internals."""
        offenders = [
            p.name for p in pub_sources()
            if p.name != "jump.py"
            and "_interface_contact_jump_map" in p.read_text()
        ]
        assert offenders == []

    def test_the_definition_names_what_it_is_not(self):
        """The sidecar has to say the plotted jump is not the training quantity."""
        from visual.pub.jump import JUMP_DEFINITION

        assert "adjacent-node" in JUMP_DEFINITION["not"]
        assert JUMP_DEFINITION["unit"] == "K"

    def test_flux_is_the_resistance_free_factor_of_the_jump(self):
        """``dT_contact = R_c * q_n`` must hold numerically, not just in the docstring."""
        from visual.pub import jump

        rng = np.random.default_rng(0)
        x_grid = np.linspace(0.0, 1.0, 33)
        fields = 300.0 + rng.normal(size=(2, 33, 8)) * 5.0
        R_c = 0.05

        flux = jump.contact_flux_map(fields, x_grid, 0.5, R_c, 2.0, 1.0)
        dT = jump.contact_jump_map(fields, x_grid, 0.5, R_c, 2.0, 1.0)
        np.testing.assert_allclose(dT, R_c * flux, rtol=1e-12, atol=1e-12)

    def test_a_vector_resistance_profile_is_accepted(self):
        """``source_itr_sin`` is the whole reason the jump must be a function of y."""
        from visual.pub import jump

        x_grid = np.linspace(0.0, 1.0, 17)
        fields = np.zeros((1, 17, 6))
        fields[:, :8, :] = 310.0
        fields[:, 8:, :] = 300.0
        profile = np.linspace(0.01, 0.2, 6)

        dT = jump.contact_jump_map(fields, x_grid, 0.5, profile, 2.0, 1.0)
        assert dT.shape == (1, 6)
        # A larger local resistance holds a larger share of the same total drop.
        assert np.all(np.diff(dT[0]) > 0)
