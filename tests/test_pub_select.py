"""Tests for ``visual.pub.select``.

Which case a qualitative figure shows is a claim about the model, so the
selection has to be a pre-registered rule rather than an ad hoc choice. These
tests pin the three properties that makes it one: it is deterministic, it ranks
*simulations* rather than correlated pairs, and it is versioned -- the golden
test at the bottom fails if the protocol changes without a version bump.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

from visual.pub import select, stats

REPO_ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1])


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def make_pairs(sim_values, *, n_pairs: int = 6, seed: str = "42",
               benchmark: str = "forcing", cells: int = 256,
               pair_offsets=None, strata=None) -> pd.DataFrame:
    """Pair-level frame where simulation ``i`` has ``rmse_K`` exactly ``sim_values[i]``.

    Every pair of a simulation carries the same ``sse_K2 / num_error_cells``
    ratio, so the pooled per-simulation value is that number exactly and a test
    can state the expected ranking without reproducing the pooling arithmetic.
    ``pair_offsets`` perturbs pair ``k`` when a test needs the pair rule to
    actually discriminate.
    """
    rows = []
    for sim, value in enumerate(sim_values):
        for k in range(n_pairs):
            v = float(value)
            if pair_offsets is not None:
                v += float(pair_offsets[k])
            row = {
                "sim_id": sim,
                "s": k // 3,
                "j": k % 3 + 1,
                "seed": seed,
                "benchmark": benchmark,
                "t_bar": 0.05 + 0.9 * (k + 1) / n_pairs,
                "rmse_K": v,
                "sse_K2": v * v * cells,
                "num_error_cells": cells,
                "target_sse_K2": cells * 400.0,
                "interface_sse_K2": v * v * 16.0,
                "num_interface_cells": 16,
                "interface_target_sse_K2": 16 * 400.0,
            }
            if strata is not None:
                row.update(strata(sim, k))
            rows.append(row)
    return pd.DataFrame(rows)


def sims_of(records: pd.DataFrame, *, strata=()) -> stats.SimFrame:
    return stats.per_sim(records, strata=strata, metrics=("rmse_K",))


@pytest.fixture(scope="module")
def ladder() -> pd.DataFrame:
    """101 simulations with ``rmse_K`` 1.0 .. 101.0, so quantiles are checkable."""
    return make_pairs([1.0 + i for i in range(101)])


# ---------------------------------------------------------------------------
# quantile correctness
# ---------------------------------------------------------------------------


class TestQuantile:
    def test_median_of_101_sims_is_the_51st(self, ladder):
        sims = sims_of(ladder)
        sel = select.select_case(sims, ladder, quantile=0.5)
        assert sel.n_sims == 101
        assert sel.rank_in_sims == 51
        assert sel.sim_id == 50
        assert sel.sim_metric_value == pytest.approx(51.0)

    def test_p90_is_the_91st(self, ladder):
        sel = select.select_case(sims_of(ladder), ladder, quantile=0.9)
        assert sel.rank_in_sims == 91
        assert sel.sim_metric_value == pytest.approx(91.0)

    def test_endpoints_are_best_and_worst(self, ladder):
        sims = sims_of(ladder)
        best = select.select_case(sims, ladder, quantile=0.0)
        worst = select.select_case(sims, ladder, quantile=1.0)
        assert best.rank_in_sims == 1
        assert best.sim_metric_value == pytest.approx(1.0)
        assert worst.rank_in_sims == 101
        assert worst.sim_metric_value == pytest.approx(101.0)

    def test_rank_is_monotone_in_quantile(self, ladder):
        sims = sims_of(ladder)
        ranks = [select.select_case(sims, ladder, quantile=q).rank_in_sims
                 for q in (0.1, 0.25, 0.5, 0.75, 0.9)]
        assert ranks == sorted(ranks)
        assert len(set(ranks)) == len(ranks)

    def test_out_of_range_quantile_raises(self, ladder):
        with pytest.raises(select.SelectionError):
            select.select_case(sims_of(ladder), ladder, quantile=1.5)

    def test_unknown_metric_raises(self, ladder):
        with pytest.raises(select.SelectionError):
            select.select_case(sims_of(ladder), ladder, metric="not_a_metric")

    def test_grid_returns_one_selection_per_quantile(self, ladder):
        sims = sims_of(ladder)
        grid = select.select_case_grid(sims, ladder, quantiles=(0.1, 0.5, 0.9))
        assert [s.quantile for s in grid] == [0.1, 0.5, 0.9]
        assert len({s.sim_id for s in grid}) == 3


# ---------------------------------------------------------------------------
# the replication unit
# ---------------------------------------------------------------------------


class TestReplicationUnit:
    def test_n_sims_counts_simulations_not_pairs(self):
        """``_representative_row`` ranked pairs; ranking pairs is the defect.

        With 20 simulations of 50 pairs each, a pair-ranking selector would
        report 1000 candidates and land on whichever pair sat at the middle of a
        correlated cloud. The reported ``n_sims`` must be 20.
        """
        records = make_pairs([1.0 + i for i in range(20)], n_pairs=50)
        sel = select.select_case(sims_of(records), records, quantile=0.5)
        assert sel.n_sims == 20
        assert len(records) == 1000

    def test_selection_is_invariant_to_pair_count(self):
        """Duplicating every pair must not move the selected simulation.

        Pairs are pseudo-replicates, so doubling them carries no new evidence
        about which simulation is the median one.
        """
        values = [1.0 + i for i in range(31)]
        few = make_pairs(values, n_pairs=4)
        many = make_pairs(values, n_pairs=40)
        a = select.select_case(sims_of(few), few, quantile=0.5)
        b = select.select_case(sims_of(many), many, quantile=0.5)
        assert a.sim_id == b.sim_id
        assert a.rank_in_sims == b.rank_in_sims


# ---------------------------------------------------------------------------
# determinism and tie-breaking
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_fifty_calls_agree(self, ladder):
        sims = sims_of(ladder)
        first = select.select_case(sims, ladder, quantile=0.5)
        for _ in range(50):
            again = select.select_case(sims, ladder, quantile=0.5)
            assert again == first

    def test_row_order_does_not_matter(self, ladder):
        sims = sims_of(ladder)
        shuffled = ladder.sample(frac=1.0, random_state=7).reset_index(drop=True)
        a = select.select_case(sims, ladder, quantile=0.5)
        b = select.select_case(sims_of(shuffled), shuffled, quantile=0.5)
        assert (a.sim_id, a.s, a.j) == (b.sim_id, b.s, b.j)

    def test_ties_break_on_the_lower_sim_id(self):
        """All ten simulations have identical error; the choice must be stated.

        Without an explicit tie-break the winner would depend on frame order,
        which is exactly the kind of silent instability that makes a figure
        irreproducible.
        """
        records = make_pairs([3.0] * 10)
        sims = sims_of(records)
        assert select.select_case(sims, records, quantile=0.0).sim_id == 0
        # A tied block is ordered by sim_id, so the q-th element of the block is
        # the q-th sim_id, not an arbitrary one.
        assert select.select_case(sims, records, quantile=0.5).sim_id == 5
        assert select.select_case(sims, records, quantile=1.0).sim_id == 9

    def test_tie_break_seed_is_recorded(self, ladder):
        sel = select.select_case(sims_of(ladder), ladder, tie_break_seed=12345)
        assert sel.tie_break_seed == 12345

    def test_stable_under_python_hash_seed(self, ladder, tmp_path):
        """Selection must not depend on ``PYTHONHASHSEED``.

        The tie-break PRNG is keyed by a sha256 of the request rather than by
        ``hash()`` precisely so this holds; the test is what stops someone
        replacing it with a dict ordering or a set iteration.
        """
        script = textwrap.dedent(
            f"""
            import sys
            sys.path.insert(0, {REPO_ROOT!r})
            sys.path.insert(0, {REPO_ROOT!r} + "/tests")
            from test_pub_select import make_pairs, sims_of
            from visual.pub import select
            records = make_pairs([1.0 + i for i in range(101)])
            sel = select.select_case(sims_of(records), records, quantile=0.5)
            print(sel.sim_id, sel.s, sel.j, sel.rank_in_sims)
            """
        )
        path = tmp_path / "probe.py"
        path.write_text(script)
        outputs = []
        for hash_seed in ("0", "1"):
            proc = subprocess.run(
                [sys.executable, str(path)], cwd=REPO_ROOT, check=True,
                capture_output=True, text=True,
                env={**__import__("os").environ, "PYTHONHASHSEED": hash_seed},
            )
            outputs.append(proc.stdout.strip())
        assert outputs[0] == outputs[1]


# ---------------------------------------------------------------------------
# pair rules
# ---------------------------------------------------------------------------


class TestPairRules:
    def test_median_within_sim_picks_a_middle_pair(self):
        offsets = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        records = make_pairs([10.0] * 5, pair_offsets=offsets)
        sel = select.select_case(sims_of(records), records, quantile=0.5,
                                 pair_rule="median_within_sim")
        assert sel.pair_metric_value == pytest.approx(10.2)

    def test_longest_lead_picks_the_largest_t_bar(self):
        records = make_pairs([10.0] * 5)
        sel = select.select_case(sims_of(records), records, quantile=0.5,
                                 pair_rule="longest_lead")
        part = records[records["sim_id"] == sel.sim_id]
        row = part[(part["s"] == sel.s) & (part["j"] == sel.j)].iloc[0]
        assert row["t_bar"] == pytest.approx(part["t_bar"].max())

    def test_unknown_pair_rule_raises(self, ladder):
        with pytest.raises(select.SelectionError):
            select.select_case(sims_of(ladder), ladder, pair_rule="whichever")

    def test_longest_lead_without_t_bar_raises(self):
        records = make_pairs([10.0] * 5).drop(columns=["t_bar"])
        with pytest.raises(select.SelectionError):
            select.select_case(sims_of(records), records,
                               pair_rule="longest_lead")

    def test_pair_rule_is_recorded(self, ladder):
        sel = select.select_case(sims_of(ladder), ladder,
                                 pair_rule="longest_lead")
        assert sel.pair_rule == "longest_lead"


# ---------------------------------------------------------------------------
# strata
# ---------------------------------------------------------------------------


def _family(sim: int, _pair: int) -> dict:
    return {"temporal_family": "sin" if sim % 2 == 0 else "pulse_train"}


class TestStrata:
    @pytest.fixture(scope="class")
    def records(self) -> pd.DataFrame:
        return make_pairs([1.0 + i for i in range(20)], strata=_family)

    def test_filter_never_leaks_across_strata(self, records):
        sims = sims_of(records, strata=("temporal_family",))
        for family, parity in (("sin", 0), ("pulse_train", 1)):
            sel = select.select_case(
                sims, records, quantile=0.5,
                strata_filter={"temporal_family": family})
            assert sel.sim_id % 2 == parity
            assert sel.n_sims == 10
            assert sel.strata["temporal_family"] == family

    def test_filtered_ranking_is_within_the_stratum(self, records):
        sims = sims_of(records, strata=("temporal_family",))
        sel = select.select_case(sims, records, quantile=1.0,
                                 strata_filter={"temporal_family": "sin"})
        # The worst even simulation is 18, not the worst overall (19).
        assert sel.sim_id == 18
        assert sel.sim_metric_value == pytest.approx(19.0)

    def test_unknown_stratum_column_raises(self, records):
        sims = sims_of(records, strata=("temporal_family",))
        with pytest.raises(select.SelectionError):
            select.select_case(sims, records,
                               strata_filter={"not_a_column": "x"})

    def test_empty_stratum_raises(self, records):
        sims = sims_of(records, strata=("temporal_family",))
        with pytest.raises(select.SelectionError):
            select.select_case(sims, records,
                               strata_filter={"temporal_family": "exp_train"})

    def test_strata_of_the_chosen_pair_are_reported(self, records):
        sims = sims_of(records, strata=("temporal_family",))
        sel = select.select_case(sims, records, quantile=0.5)
        assert sel.strata["temporal_family"] in ("sin", "pulse_train")


# ---------------------------------------------------------------------------
# lead columns and context
# ---------------------------------------------------------------------------


class TestLeadColumns:
    def test_columns_are_ordered_by_lead_and_exist(self):
        records = make_pairs([1.0 + i for i in range(11)], n_pairs=12)
        sel = select.select_case(sims_of(records), records, quantile=0.5)
        columns = select.select_lead_columns(records, sel,
                                             lead_quantiles=(0.1, 0.5, 0.9))
        assert len(columns) == 3
        assert len(set(columns)) == 3

        part = records[records["sim_id"] == sel.sim_id]
        leads = []
        for s, j in columns:
            row = part[(part["s"] == s) & (part["j"] == j)]
            assert len(row) == 1
            leads.append(float(row["t_bar"].iloc[0]))
        assert leads == sorted(leads)

    def test_missing_t_bar_raises(self):
        records = make_pairs([1.0 + i for i in range(11)])
        sel = select.select_case(sims_of(records), records, quantile=0.5)
        with pytest.raises(select.SelectionError):
            select.select_lead_columns(records.drop(columns=["t_bar"]), sel)

    def test_context_holds_the_selected_value(self, ladder):
        sims = sims_of(ladder)
        sel = select.select_case(sims, ladder, quantile=0.5)
        context = select.selection_context(sims, "rmse_K", sel)
        assert context.shape == (101,)
        assert np.isfinite(context).all()
        assert np.isclose(context, sel.sim_metric_value).any()


# ---------------------------------------------------------------------------
# the record the sidecar and footer carry
# ---------------------------------------------------------------------------


class TestRecord:
    def test_dict_round_trip_keeps_every_field(self, ladder):
        sel = select.select_case(sims_of(ladder), ladder, quantile=0.5)
        payload = sel.to_dict()
        assert set(payload) == set(select.CaseSelection.__dataclass_fields__)
        assert select.CaseSelection(**payload) == sel

    def test_label_names_the_quantile_and_the_rank(self, ladder):
        sel = select.select_case(sims_of(ladder), ladder, quantile=0.5)
        assert sel.label == "sim 50, rank 51/101 (median of rmse_K)"
        assert "sel v1" in sel.footer_field()

    def test_seed_and_benchmark_are_carried(self, ladder):
        sel = select.select_case(sims_of(ladder), ladder, quantile=0.5)
        assert sel.seed == "42"
        assert sel.benchmark == "forcing"

    def test_selection_is_frozen(self, ladder):
        sel = select.select_case(sims_of(ladder), ladder, quantile=0.5)
        with pytest.raises(Exception):
            sel.sim_id = 0


# ---------------------------------------------------------------------------
# golden
# ---------------------------------------------------------------------------


class TestGolden:
    """A change to any of these numbers is a protocol change.

    Published figures name the case they show. If the rule that picks it moves
    without :data:`select.PROTOCOL_VERSION` moving with it, two papers can print
    the same selection string for different data, so this test is deliberately
    brittle: update it and bump the version in the same commit.
    """

    def test_protocol_version_is_pinned(self):
        assert select.PROTOCOL_VERSION == 1
        assert select.DEFAULT_TIE_BREAK_SEED == 20260728
        assert select.PAIR_RULES == (
            "median_within_sim", "longest_lead", "fixed_leads")

    def test_golden_case(self):
        records = make_pairs([1.0 + i for i in range(101)],
                             pair_offsets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        sel = select.select_case(sims_of(records), records, quantile=0.5)
        assert (sel.sim_id, sel.s, sel.j) == (50, 0, 3)
        assert sel.rank_in_sims == 51
        assert sel.n_sims == 101
        assert sel.protocol_version == 1
