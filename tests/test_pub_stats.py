"""Tests for ``visual.pub.stats``.

The defect this module exists to correct is that the previous paper figures
summarized ``(sim_id, s, j)`` snapshot pairs as if they were independent
observations. Several tests here are regression guards against exactly that:
:class:`TestReplicationUnit` asserts that the reported ``n`` counts simulations
rather than pairs, and :meth:`TestPooling.test_pooled_differs_from_pair_mean`
asserts that pooling is not the same number as averaging per-pair values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from visual.pub import stats


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def make_records(*, n_sims: int, n_pairs: int, seed: str = "42",
                 benchmark: str = "forcing", rng_seed: int = 0,
                 cells: int = 256, sim_scale=None) -> pd.DataFrame:
    """A schema-v2-shaped pair-level frame with pooled sufficient statistics.

    ``sim_scale(i)`` sets the error magnitude of simulation ``i`` so a test can
    control the per-sim distribution exactly.
    """
    rng = np.random.default_rng(rng_seed)
    rows = []
    for sim in range(n_sims):
        scale = 1.0 if sim_scale is None else float(sim_scale(sim))
        for k in range(n_pairs):
            sse = scale ** 2 * cells * (1.0 + 0.05 * rng.standard_normal())
            rows.append({
                "sim_id": sim,
                "s": k // 4,
                "j": k % 4 + 1,
                "seed": seed,
                "benchmark": benchmark,
                "t_bar": 0.05 + 0.9 * (k + 1) / n_pairs,
                "sse_K2": abs(sse),
                "num_error_cells": cells,
                "target_sse_K2": cells * 400.0,
                "interface_sse_K2": abs(sse) * 0.25,
                "num_interface_cells": 16,
                "interface_target_sse_K2": 16 * 400.0,
                "node_jump_rmse_K": scale * 0.5,
            })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def records() -> pd.DataFrame:
    return make_records(n_sims=45, n_pairs=12)


def only(mapping):
    """The single entry of a keyed reduction, asserting there is exactly one.

    ``summarize`` and ``across_seeds`` key by ``seed`` and ``benchmark`` even
    with no strata, because a summary pooled over seeds would hide exactly the
    seed spread the figures are supposed to show.
    """
    assert len(mapping) == 1, f"expected one group, got {sorted(mapping)}"
    return next(iter(mapping.values()))


# ---------------------------------------------------------------------------
# pooling identities
# ---------------------------------------------------------------------------


class TestPooling:
    def test_pooled_rmse_identity(self):
        sse = np.array([4.0, 16.0, 36.0])
        n = np.array([1.0, 4.0, 9.0])
        got = stats.pooled_rmse(sse.sum(), n.sum())
        assert got == pytest.approx(np.sqrt(56.0 / 14.0))

    def test_pooled_rel_l2_is_percent(self):
        got = stats.pooled_rel_l2_pct(1.0, 4.0)
        assert got == pytest.approx(50.0)

    def test_pooled_differs_from_pair_mean(self):
        """The regression guard against the original defect.

        When pairs carry different cell counts, the pooled RMSE and the mean of
        per-pair RMSEs are different statistics. A test that only checked the
        equal-weight case would pass against the buggy implementation.
        """
        sse = np.array([100.0, 100.0])
        n = np.array([1.0, 99.0])
        pooled = stats.pooled_rmse(sse.sum(), n.sum())
        pair_mean = float(np.mean(np.sqrt(sse / n)))
        assert pooled == pytest.approx(np.sqrt(200.0 / 100.0))
        assert abs(pooled - pair_mean) > 1.0

    def test_rel_l2_is_sigma_invariant(self):
        """Scaling target and error together must not move the ratio."""
        base = stats.pooled_rel_l2_pct(9.0, 400.0)
        for sigma in (0.5, 2.0, 17.0):
            scaled = stats.pooled_rel_l2_pct(9.0 * sigma ** 2, 400.0 * sigma ** 2)
            assert scaled == pytest.approx(base)

    def test_non_finite_and_zero_denominator_give_nan(self):
        assert np.isnan(stats.pooled_rmse(1.0, 0.0))
        assert np.isnan(stats.pooled_rel_l2_pct(1.0, 0.0))
        assert np.isnan(stats.pooled_rmse(np.nan, 4.0))

    def test_per_sim_uses_the_pooled_identity(self, records):
        sims = stats.per_sim(records, metrics=("rmse_K",))
        assert sims.pooled
        row = sims.df.iloc[0]
        raw = records[records["sim_id"] == row["sim_id"]]
        expected = np.sqrt(raw["sse_K2"].sum() / raw["num_error_cells"].sum())
        assert row["rmse_K"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# replication unit
# ---------------------------------------------------------------------------


class TestReplicationUnit:
    """``n`` must count simulations. This is the whole point of the module."""

    def test_n_sims_is_not_n_pairs(self, records):
        sims = stats.per_sim(records, metrics=("rmse_K",))
        assert sims.n_sims == 45
        assert sims.n_pairs == 45 * 12
        assert sims.n_sims != sims.n_pairs

    def test_summary_reports_sims_not_pairs(self, records):
        summary = only(stats.summarize(stats.per_sim(records, metrics=("rmse_K",)),
                                       "rmse_K"))
        assert summary.n_sims == 45
        assert summary.n_pairs == 540

    def test_few_sims_many_pairs_gives_a_wider_interval(self):
        """5 sims x 100 pairs and 100 sims x 5 pairs, identical per-sim values.

        Treating pairs as replicates would make the first case look like the
        better-determined one. Aggregating to sims first makes it the worse one,
        which is the truth.
        """
        def scale(i):
            return 1.0 + 0.4 * np.sin(i)

        wide = stats.per_sim(make_records(n_sims=25, n_pairs=100, sim_scale=scale),
                             metrics=("rmse_K",))
        narrow = stats.per_sim(make_records(n_sims=100, n_pairs=5, sim_scale=scale),
                               metrics=("rmse_K",))
        assert wide.n_sims == 25 and narrow.n_sims == 100
        assert wide.n_pairs > narrow.n_pairs

        a = only(stats.summarize(wide, "rmse_K", n_boot=2000))
        b = only(stats.summarize(narrow, "rmse_K", n_boot=2000))
        assert (a.ci_hi - a.ci_lo) > (b.ci_hi - b.ci_lo)

    def test_one_row_per_sim(self, records):
        sims = stats.per_sim(records, metrics=("rmse_K",))
        assert sims.df["sim_id"].is_unique


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------


class TestGuards:
    def test_too_few_sims_suppresses_the_interval(self):
        sims = stats.per_sim(make_records(n_sims=8, n_pairs=6), metrics=("rmse_K",))
        summary = only(stats.summarize(sims, "rmse_K"))
        assert summary.ci_method == "none"
        assert np.isnan(summary.ci_lo) and np.isnan(summary.ci_hi)
        assert any(d.code == "TOO_FEW_SIMS" for d in summary.degradations)

    def test_p99_is_nan_below_the_threshold(self, records):
        summary = only(stats.summarize(stats.per_sim(records, metrics=("rmse_K",)),
                                       "rmse_K", n_boot=500))
        assert np.isnan(summary.p99)
        assert any(d.code == "P99_UNDERDETERMINED" for d in summary.degradations)
        assert np.isfinite(summary.p90)

    def test_p99_is_reported_above_the_threshold(self):
        sims = stats.per_sim(make_records(n_sims=120, n_pairs=3), metrics=("rmse_K",))
        summary = only(stats.summarize(sims, "rmse_K", n_boot=500))
        assert np.isfinite(summary.p99)
        assert not any(d.code == "P99_UNDERDETERMINED" for d in summary.degradations)

    def test_seed_spread_undetermined_below_three_seeds(self):
        frames = [make_records(n_sims=30, n_pairs=4, seed=s, rng_seed=i)
                  for i, s in enumerate(("1", "2"))]
        sims = stats.per_sim(pd.concat(frames, ignore_index=True),
                             metrics=("rmse_K",))
        roll = only(stats.across_seeds(sims, "rmse_K"))
        assert np.isnan(roll.std)
        assert any(d.code == "SEED_SPREAD_UNDERDETERMINED" for d in roll.degradations)

    def test_seed_spread_reported_at_three_seeds(self):
        frames = [make_records(n_sims=30, n_pairs=4, seed=s, rng_seed=i)
                  for i, s in enumerate(("1", "2", "3"))]
        sims = stats.per_sim(pd.concat(frames, ignore_index=True),
                             metrics=("rmse_K",))
        roll = only(stats.across_seeds(sims, "rmse_K"))
        assert np.isfinite(roll.std)
        assert len(roll.per_seed) == 3

    def test_v1_frame_emits_the_schema_degradation(self, records):
        v1 = records.drop(columns=list(stats._SUM_COLS))
        sims = stats.per_sim(v1, metrics=("node_jump_rmse_K",))
        assert not sims.pooled
        assert any(d.code == "SCHEMA_V1_NO_POOLED_STATS" for d in sims.degradations)

    def test_v1_frame_cannot_fake_a_pooled_metric(self, records):
        v1 = records.drop(columns=list(stats._SUM_COLS))
        with pytest.raises(KeyError):
            stats.per_sim(v1, metrics=("rmse_K",))


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------


class TestBootstrap:
    def test_deterministic_under_a_fixed_seed(self):
        values = np.random.default_rng(3).lognormal(size=60)
        a = stats.bootstrap_ci(values, n_boot=1000, rng_seed=7)
        b = stats.bootstrap_ci(values, n_boot=1000, rng_seed=7)
        assert a == b

    def test_different_seeds_give_different_intervals(self):
        values = np.random.default_rng(3).lognormal(size=60)
        a = stats.bootstrap_ci(values, n_boot=1000, rng_seed=7)
        b = stats.bootstrap_ci(values, n_boot=1000, rng_seed=8)
        assert a[:2] != b[:2]

    def test_interval_brackets_the_point_estimate(self):
        values = np.random.default_rng(11).lognormal(size=80)
        lo, hi, method = stats.bootstrap_ci(values, n_boot=2000)
        assert method == "bca"
        assert lo <= float(np.median(values)) <= hi

    def test_small_sample_returns_no_method(self):
        lo, hi, method = stats.bootstrap_ci(np.arange(5.0))
        assert method == "none"
        assert np.isnan(lo) and np.isnan(hi)

    @pytest.mark.slow
    def test_coverage_is_approximately_nominal(self):
        """A slow sanity check that the interval is not badly miscalibrated."""
        rng = np.random.default_rng(101)
        truth = float(np.median(rng.lognormal(size=2_000_000)))
        hits = 0
        trials = 120
        for t in range(trials):
            sample = rng.lognormal(size=60)
            lo, hi, _ = stats.bootstrap_ci(sample, n_boot=800, rng_seed=1000 + t)
            hits += int(lo <= truth <= hi)
        assert 0.80 <= hits / trials <= 1.0


# ---------------------------------------------------------------------------
# distributions and rates
# ---------------------------------------------------------------------------


class TestDistributions:
    def test_ecdf_is_monotone_and_ends_at_one(self):
        values = np.random.default_rng(5).normal(size=37)
        x, f = stats.ecdf(values)
        assert x.size == f.size == 37
        assert np.all(np.diff(x) >= 0)
        assert np.all(np.diff(f) > 0)
        assert f[-1] == pytest.approx(1.0)

    def test_ecdf_length_is_n_sims_not_n_pairs(self, records):
        sims = stats.per_sim(records, metrics=("rmse_K",))
        x, _ = stats.ecdf(sims.df["rmse_K"])
        assert x.size == sims.n_sims

    def test_ecdf_band_is_ordered(self):
        by_seed = {s: np.random.default_rng(i).normal(size=40)
                   for i, s in enumerate("abc")}
        grid, lo, hi = stats.ecdf_band(by_seed)
        assert grid.size == lo.size == hi.size
        assert np.all(hi >= lo - 1e-12)

    def test_wilson_interval_brackets_the_rate(self):
        lo, hi = stats.wilson_ci(3, 20)
        assert lo < 3 / 20 < hi
        assert 0.0 <= lo and hi <= 1.0

    def test_wilson_handles_zero_events(self):
        lo, hi = stats.wilson_ci(0, 20)
        assert lo == pytest.approx(0.0, abs=1e-12)
        assert 0.0 < hi < 1.0

    def test_exceedance_rate_counts_correctly(self):
        rate = stats.exceedance_rate([1.0, 2.0, 3.0, 4.0], threshold=2.5)
        assert rate.k == 2 and rate.n == 4
        assert rate.rate == pytest.approx(0.5)
        assert rate.lo < 0.5 < rate.hi


# ---------------------------------------------------------------------------
# strata
# ---------------------------------------------------------------------------


class TestStrata:
    def test_shared_edges_are_identical_across_frames(self):
        a = make_records(n_sims=20, n_pairs=8, rng_seed=1)
        b = make_records(n_sims=20, n_pairs=8, rng_seed=2)
        b["t_bar"] = b["t_bar"] * 0.4          # a genuinely different spread
        shared = stats.shared_bin_edges([a, b], "lead_bin")
        only_a = stats.shared_bin_edges([a], "lead_bin")
        assert not np.allclose(shared, only_a)
        # The point: one call, one set of edges, reused for both frames.
        again = stats.shared_bin_edges([a, b], "lead_bin")
        np.testing.assert_allclose(shared, again)

    def test_fixed_edges_ignore_the_data(self):
        a = make_records(n_sims=5, n_pairs=4)
        a["x_I"] = 0.5
        np.testing.assert_allclose(stats.shared_bin_edges([a], "interface_x_bin"),
                                   (0.2, 0.35, 0.5, 0.65, 0.8))

    def test_categorical_stratum_has_no_edges(self):
        with pytest.raises(ValueError, match="categorical"):
            stats.shared_bin_edges([make_records(n_sims=2, n_pairs=2)],
                                   "temporal_family")

    def test_unknown_stratum_names_the_known_ones(self):
        with pytest.raises(KeyError, match="unknown stratum"):
            stats.stratum_spec("not_a_stratum")

    def test_strata_partition_the_sims(self, records):
        sims = stats.per_sim(records, strata=("lead_bin",), metrics=("rmse_K",))
        assert "lead_bin" in sims.df.columns
        assert sims.n_pairs == len(records)
        assert len(sims.df) > records["sim_id"].nunique()

    def test_summarize_keys_by_stratum(self, records):
        sims = stats.per_sim(records, strata=("lead_bin",), metrics=("rmse_K",))
        out = stats.summarize(sims, "rmse_K", strata=("lead_bin",), n_boot=200)
        assert len(out) == sims.df["lead_bin"].nunique()
        assert all(k[-1] in set(sims.df["lead_bin"]) for k in out)
        assert sum(s.n_sims for s in out.values()) == len(sims.df)


# ---------------------------------------------------------------------------
# paired deltas and metric spaces
# ---------------------------------------------------------------------------


class TestPairedDelta:
    def test_identical_frames_give_exactly_zero(self, records):
        sims = stats.per_sim(records, metrics=("rmse_K",))
        delta = stats.paired_seed_delta(sims, sims, "rmse_K", n_boot=200)
        assert delta.median == pytest.approx(0.0, abs=1e-12)
        assert np.allclose(delta.values, 0.0)

    def test_constant_offset_is_recovered(self, records):
        a = stats.per_sim(records, metrics=("rmse_K",))
        b = stats.per_sim(records, metrics=("rmse_K",))
        b.df = b.df.copy()
        b.df["rmse_K"] = b.df["rmse_K"] + 2.0
        delta = stats.paired_seed_delta(a, b, "rmse_K", n_boot=200)
        assert abs(delta.median) == pytest.approx(2.0, rel=1e-9)

    def test_matching_drops_unpaired_sims(self, records):
        a = stats.per_sim(records, metrics=("rmse_K",))
        b = stats.per_sim(records[records["sim_id"] < 20], metrics=("rmse_K",))
        delta = stats.paired_seed_delta(a, b, "rmse_K", n_boot=200)
        assert delta.n_pairs == 20


class TestMetricSpace:
    def test_same_space_is_accepted(self):
        assert stats.assert_same_space(("rmse_K", "iface_rmse_K")) == "kelvin"

    def test_mixing_spaces_raises(self):
        with pytest.raises(ValueError):
            stats.assert_same_space(("rmse_K", "rel_l2_pct"))

    def test_unknown_metric_raises(self):
        with pytest.raises(KeyError):
            stats.metric_spec("not_a_metric")

    def test_every_metric_declares_a_known_space(self):
        assert {s.space for s in stats.METRICS.values()} <= {"kelvin", "normalized"}

    def test_pooling_families_partition_the_metrics(self):
        families = set(stats.POOLED_METRICS) | set(stats.RATIO_METRICS) \
            | set(stats.MEAN_METRICS)
        assert families == set(stats.METRICS)
