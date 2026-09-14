"""Behavioural tests for the inverse-extension panels.

``tests/test_pub_style.py`` already checks that ``panels.py`` stays a drawing
layer. These tests check what it draws: that a censored interval is never shown
as a closed bracket, that an inadmissible parameter pair stays blank, and that a
marker outside the evaluated grid stays on the canvas. Each of those is a claim
about the figure's honesty, so each is pinned rather than eyeballed.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from visual.pub import panels, records, stats

COLOR = "#8C564B"


@pytest.fixture
def ax():
    figure, axes = plt.subplots()
    yield axes
    plt.close(figure)


def make_curve(**overrides) -> records.ProfileCurve:
    values = np.linspace(0.2, 0.7, 11)
    fields = dict(
        param_name="R_base", param_index=0, values=values,
        delta_ell=40.0 * (values - 0.45) ** 2,
        theta_hat=0.45, theta_true=0.40, ci_low=0.23, ci_high=0.67,
        bound_limited=False,
    )
    fields.update(overrides)
    return records.ProfileCurve(**fields)


def make_region(**overrides) -> records.JointNLLGrid:
    base = np.linspace(0.2, 0.9, 8)
    amp = np.linspace(0.4, 2.6, 6)
    surface = 40.0 * ((base[:, None] - 0.45) ** 2 + (amp[None, :] - 1.05) ** 2)
    surface[amp[None, :] > 3.0 - base[:, None]] = np.nan
    fields = dict(
        param_names=("R_base", "A"), axes=(base, amp), delta_ell=surface,
        threshold=2.9957,
    )
    fields.update(overrides)
    return records.JointNLLGrid(**fields)


class TestRcProfileRecoveryPanel:
    def test_it_draws_exactly_two_curves_distinguished_by_dash_pattern(self, ax):
        y = np.linspace(0.0, 1.0, 50)
        panels.rc_profile_recovery_panel(
            ax, y, 0.4 + 1.2 * np.sin(np.pi * y), 0.45 + 1.05 * np.sin(np.pi * y),
            color=COLOR,
        )
        assert len(ax.lines) == 2
        true_line, hat_line = ax.lines
        assert true_line.get_label() == "true"
        assert hat_line.get_label() == "recovered"
        # Solid vs dashed is the only channel separating them in print, so a
        # colour-only distinction would not survive a greyscale reproduction.
        assert true_line.get_linestyle() != hat_line.get_linestyle()
        assert hat_line.get_linestyle() in {"--", (0, (6.4, 1.6))}

    def test_it_leaves_the_shared_y_scale_to_the_caller(self, ax):
        """Rescaling per panel would make a worse reconstruction look as good."""
        y = np.linspace(0.0, 1.0, 50)
        ax.set_ylim(0.0, 4.0)
        panels.rc_profile_recovery_panel(
            ax, y, np.full_like(y, 0.4), np.full_like(y, 0.5), color=COLOR,
        )
        assert ax.get_ylim() == (0.0, 4.0)
        assert ax.get_xlim() == (0.0, 1.0)

    def test_mismatched_shapes_raise_rather_than_broadcast(self, ax):
        with pytest.raises(ValueError, match="must share one shape"):
            panels.rc_profile_recovery_panel(
                ax, np.linspace(0.0, 1.0, 50), np.zeros(50), np.zeros(49),
                color=COLOR,
            )


class TestProfileLikelihoodPanel:
    def test_a_closed_interval_is_drawn_as_a_span(self, ax):
        panels.profile_likelihood_panel(
            ax, make_curve(), color=COLOR, xlabel="$R$", threshold=1.9207,
        )
        spans = [p for p in ax.patches if p.get_label() == "95% interval"]
        assert len(spans) == 1
        assert len(ax.texts) == 0

    def test_a_censored_interval_is_annotated_not_bracketed(self, ax):
        """A closed bracket over a censored interval overstates the constraint."""
        panels.profile_likelihood_panel(
            ax, make_curve(bound_limited=True), color=COLOR, xlabel="$R$",
            threshold=1.9207,
        )
        assert [p for p in ax.patches if p.get_label() == "95% interval"] == []
        assert [t.get_text() for t in ax.texts] == ["interval bound-limited"]

    def test_the_threshold_and_true_value_are_drawn_where_asked(self, ax):
        panels.profile_likelihood_panel(
            ax, make_curve(), color=COLOR, xlabel="$R$", threshold=1.9207,
        )
        horizontals = [
            line.get_ydata()[0] for line in ax.lines
            if len(set(np.asarray(line.get_ydata(), dtype=float))) == 1
            and len(line.get_xdata()) == 2
        ]
        assert pytest.approx(1.9207) in horizontals
        verticals = [
            float(np.asarray(line.get_xdata(), dtype=float)[0])
            for line in ax.lines if line.get_label() == "true"
        ]
        assert verticals == [pytest.approx(0.40)]

    def test_the_baseline_is_pinned_at_zero(self, ax):
        """``delta_ell`` is a distance from the optimum; a floating base hides it."""
        panels.profile_likelihood_panel(
            ax, make_curve(), color=COLOR, xlabel="$R$", threshold=1.9207,
        )
        assert ax.get_ylim()[0] == pytest.approx(0.0)


class TestJointNllContourPanel:
    def test_it_consumes_a_measured_grid_and_a_laplace_region_alike(self, ax):
        """The two producers are duck typed; a branch here would be a design smell."""
        measured = make_region()
        approximate = stats.laplace_joint_region(
            np.array([[1.0, 0.2], [0.3, 1.4], [0.9, 0.1]]), 4e-4, [0.45, 1.05],
            param_names=("R_base", "A"), peak_max=3.0,
        )
        assert measured.approximate is False
        assert approximate.approximate is True
        for region in (measured, approximate):
            mappable = panels.joint_nll_contour_panel(
                ax, region, theta_hat=[0.45, 1.05], theta_true=[0.40, 1.20],
                labels=("$R$", "$A$"),
            )
            assert mappable is not None

    def test_the_inadmissible_corner_is_left_blank(self, ax):
        """``A > R_PEAK_MAX - R_base`` cannot occur, so it must not be shaded."""
        region = make_region()
        panels.joint_nll_contour_panel(
            ax, region, theta_hat=[0.45, 1.05], theta_true=[0.40, 1.20],
            labels=("$R$", "$A$"),
        )
        assert np.isnan(region.delta_ell).any()
        # The contour set never received the masked entries, so the highest
        # level it drew is bounded by the finite part of the surface.
        assert np.nanmax(region.delta_ell) < np.inf

    def test_a_marker_outside_the_grid_stays_visible(self, ax):
        """A true value outside the evaluated region is the finding, not a gap."""
        region = make_region()
        panels.joint_nll_contour_panel(
            ax, region, theta_hat=[0.45, 1.05], theta_true=[0.05, 2.90],
            labels=("$R$", "$A$"),
        )
        low_x, high_x = ax.get_xlim()
        low_y, high_y = ax.get_ylim()
        assert low_x <= 0.05 and high_x >= 0.9
        assert low_y <= 0.4 and high_y >= 2.90

    def test_the_true_marker_carries_an_outline(self, ax):
        """White on the blank corner would read as an absent value."""
        panels.joint_nll_contour_panel(
            ax, make_region(), theta_hat=[0.45, 1.05], theta_true=[0.40, 1.20],
            labels=("$R$", "$A$"),
        )
        true_marker = next(l for l in ax.lines if l.get_label() == "true")
        assert true_marker.get_path_effects() != []

    def test_axes_and_surface_shape_must_agree(self, ax):
        region = make_region(delta_ell=np.zeros((3, 3)))
        with pytest.raises(ValueError, match="does not match its axes"):
            panels.joint_nll_contour_panel(
                ax, region, theta_hat=[0.45, 1.05], theta_true=[0.40, 1.20],
                labels=("$R$", "$A$"),
            )


class TestJumpProfileFamily:
    def test_each_lead_draws_a_truth_and_a_prediction_line(self, ax):
        y = np.linspace(0.0, 1.0, 20)
        truth = np.stack([np.sin(np.pi * y), 2.0 * np.sin(np.pi * y)])
        panels.jump_profile_family(ax, y, truth, truth + 0.1,
                                   lead_times=[0.2, 0.8], color=COLOR)
        assert len(ax.lines) == 4

    def test_truth_and_prediction_are_separated_by_dash_pattern_not_colour(self, ax):
        """Greyscale reproduction has to preserve the comparison being made."""
        y = np.linspace(0.0, 1.0, 20)
        truth = np.sin(np.pi * y)[None, :]
        panels.jump_profile_family(ax, y, truth, truth + 0.1,
                                   lead_times=[0.5], color=COLOR)
        truth_line, pred_line = ax.lines
        assert truth_line.get_linestyle() != pred_line.get_linestyle()
        assert truth_line.get_color() == pred_line.get_color()

    def test_the_tint_darkens_with_lead_time(self, ax):
        y = np.linspace(0.0, 1.0, 8)
        truth = np.zeros((3, 8))
        panels.jump_profile_family(ax, y, truth, truth, lead_times=[0.1, 0.5, 0.9],
                                   color=COLOR)
        brightness = [sum(matplotlib.colors.to_rgb(line.get_color()))
                      for line in ax.lines[::2]]
        assert brightness[0] > brightness[1] > brightness[2]

    def test_mismatched_shapes_raise_rather_than_broadcast(self, ax):
        with pytest.raises(ValueError, match="must share one shape"):
            panels.jump_profile_family(ax, np.linspace(0, 1, 8), np.zeros((2, 8)),
                                       np.zeros((2, 7)), lead_times=[0.1, 0.2],
                                       color=COLOR)

    def test_a_lead_without_a_profile_raises(self, ax):
        with pytest.raises(ValueError, match="lead_times has"):
            panels.jump_profile_family(ax, np.linspace(0, 1, 8), np.zeros((2, 8)),
                                       np.zeros((2, 8)), lead_times=[0.1, 0.2, 0.3],
                                       color=COLOR)

    def test_a_profile_must_match_the_y_grid(self, ax):
        with pytest.raises(ValueError, match="y values"):
            panels.jump_profile_family(ax, np.linspace(0, 1, 7), np.zeros((1, 8)),
                                       np.zeros((1, 8)), lead_times=[0.1],
                                       color=COLOR)


class TestParityScatter:
    def test_the_identity_line_is_drawn_and_the_axes_are_square(self, ax):
        """An unequal aspect turns a slope bias into what looks like scatter."""
        rng = np.random.default_rng(0)
        truth = rng.standard_normal(200)
        panels.parity_scatter(ax, truth, 0.8 * truth)
        assert len(ax.lines) == 1
        assert ax.get_xlim() == ax.get_ylim()
        assert ax.get_aspect() == 1.0

    def test_it_returns_a_mappable_so_one_colourbar_can_be_shared(self, ax):
        truth = np.linspace(0.0, 1.0, 30)
        mappable = panels.parity_scatter(ax, truth, truth, color_by=truth)
        assert mappable.get_array() is not None

    def test_non_finite_samples_are_dropped_rather_than_ruining_the_limits(self, ax):
        truth = np.array([0.0, 1.0, np.nan])
        pred = np.array([0.0, 1.0, 5.0])
        panels.parity_scatter(ax, truth, pred)
        low, high = ax.get_xlim()
        assert high < 2.0

    def test_the_fit_is_annotated_when_supplied(self, ax):
        truth = np.linspace(1.0, 5.0, 20)
        fit = stats.parity_fit(truth, 0.9 * truth, n_sims=3)
        panels.parity_scatter(ax, truth, 0.9 * truth, fit=fit)
        assert any("slope" in t.get_text() for t in ax.texts)
