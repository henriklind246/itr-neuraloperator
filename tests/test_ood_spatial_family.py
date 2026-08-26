"""Tests for scripts/run_ood_spatial_family.py (zero-shot sinusoid OOD study)."""

import math

import numpy as np
import pytest

from scripts.run_ood_spatial_family import (
    ANCHOR_PARAMS,
    anchor_check,
    build_arm_params,
    check_preconditions,
    latents_hash,
    lead_bin_edges,
    modulation_depth,
    pool_to_sim_lead,
    pool_to_sims,
    spatial_load,
)
from src.physics.boundary_forcing import SPATIAL_BUILDERS, spatial_uniform


def _id_params(n=6):
    rng = np.random.default_rng(0)
    out = []
    for i in range(n):
        out.append({
            "R_c": float(rng.uniform(0.05, 1.0)),
            "temporal_family": "sin",
            "temporal_params": {"A": 100.0 + i, "f": 2.0, "t_on": 0.0,
                                "t_off": 0.2, "phase": 0.0, "tukey_alpha": 0.5},
            "spatial_family": "gaussian",
            "spatial_params": {"y_c": 0.5, "sigma_y": 0.1},
            "T0": np.full((4, 4), 300.0, dtype=np.float32),
        })
    return out


class TestArmConstruction:
    def test_arms_share_every_non_spatial_latent(self):
        base = _id_params()
        sinus = build_arm_params(base, "sinusoid", spatial_seed=11, anchor_sims=3)
        ident = build_arm_params(base, "id_baseline", spatial_seed=11, anchor_sims=3)
        anchor = build_arm_params(base, "uniform_anchor", spatial_seed=11, anchor_sims=3)

        assert len(sinus) == len(ident) == len(base)
        assert len(anchor) == 3
        for i, src in enumerate(base):
            for arm in (sinus, ident, anchor):
                if i >= len(arm):
                    continue
                assert arm[i]["R_c"] == src["R_c"]
                assert arm[i]["temporal_family"] == src["temporal_family"]
                assert arm[i]["temporal_params"] == src["temporal_params"]
                assert latents_hash(arm[i]) == latents_hash(src)

    def test_only_the_spatial_profile_differs(self):
        base = _id_params()
        sinus = build_arm_params(base, "sinusoid", spatial_seed=11, anchor_sims=3)
        ident = build_arm_params(base, "id_baseline", spatial_seed=11, anchor_sims=3)
        for a, b in zip(sinus, ident):
            assert a["spatial_family"] == "sinusoid"
            assert b["spatial_family"] == "gaussian"
            assert a["spatial_params"] != b["spatial_params"]

    def test_sinusoid_arm_does_not_perturb_the_source(self):
        base = _id_params()
        before = [dict(p["temporal_params"]) for p in base]
        build_arm_params(base, "sinusoid", spatial_seed=11, anchor_sims=3)
        assert [dict(p["temporal_params"]) for p in base] == before

    def test_anchor_profile_is_exactly_uniform(self):
        base = _id_params()
        anchor = build_arm_params(base, "uniform_anchor", spatial_seed=11, anchor_sims=3)
        y = np.linspace(0.0, 1.0, 101)
        for p in anchor:
            assert p["spatial_params"] == ANCHOR_PARAMS
            s = SPATIAL_BUILDERS["sinusoid"](y, **p["spatial_params"])
            np.testing.assert_array_equal(s, spatial_uniform(y))

    def test_unknown_arm_rejected(self):
        with pytest.raises(ValueError, match="unknown arm"):
            build_arm_params(_id_params(), "nope", spatial_seed=1, anchor_sims=1)


class TestSpatialLoad:
    def test_matches_closed_form(self):
        # I_s = (1-m) + (m / (2 pi f)) * [cos(phi) - cos(2 pi f + phi)]
        y = np.linspace(0.0, 1.0, 20001)
        for m, f, phi in [(0.3, 2.5, 0.7), (0.5, 0.5, 0.0), (0.1, 4.0, 3.1)]:
            params = {"c0": 1.0 - m, "c1": m, "f": f, "phase": phi}
            got = spatial_load("sinusoid", params, y)
            want = (1.0 - m) + (m / (2.0 * math.pi * f)) * (
                math.cos(phi) - math.cos(2.0 * math.pi * f + phi)
            )
            assert got == pytest.approx(want, rel=1e-6, abs=1e-9)

    def test_anchor_load_is_one(self):
        y = np.linspace(0.0, 1.0, 501)
        assert spatial_load("sinusoid", ANCHOR_PARAMS, y) == pytest.approx(1.0)

    def test_modulation_depth(self):
        assert modulation_depth({"c0": 0.7, "c1": 0.3}) == pytest.approx(0.3)
        assert modulation_depth(ANCHOR_PARAMS) == pytest.approx(0.0)


class TestPrecondition:
    def test_accepts_full_spatial_conditioning(self):
        check_preconditions({"conf": {"benchmark": {
            "name": "forcing", "spatial_conditioning": "full",
            "spatial_profile_bins": 8}}})

    def test_rejects_other_benchmark(self):
        ckpt = {"conf": {"benchmark": {"name": "source",
                                       "spatial_conditioning": "spatial_field_only"}}}
        with pytest.raises(SystemExit, match="forcing benchmark"):
            check_preconditions(ckpt)

    def test_accepts_valid(self):
        check_preconditions({"conf": {"benchmark": {
            "name": "forcing", "spatial_conditioning": "spatial_field_only",
            "spatial_profile_bins": 8}}})

    def test_rejects_checkpoint_without_eight_bin_contract(self):
        with pytest.raises(SystemExit, match="spatial_profile_bins=8"):
            check_preconditions({"conf": {"benchmark": {
                "name": "forcing", "spatial_conditioning": "full"}}})


class TestPooling:
    def test_reproduces_hand_computed_statistics(self):
        rows = [
            {"sim_id": "0", "rmse_K": "1.0", "sse_K2": "8.0",
             "num_error_cells": "2", "target_sse_K2": "200.0",
             "interface_sse_K2": "2.0", "num_interface_cells": "1",
             "interface_target_sse_K2": "50.0"},
            {"sim_id": "0", "rmse_K": "3.0", "sse_K2": "10.0",
             "num_error_cells": "2", "target_sse_K2": "300.0",
             "interface_sse_K2": "3.0", "num_interface_cells": "1",
             "interface_target_sse_K2": "50.0"},
            {"sim_id": "1", "rmse_K": "2.0", "sse_K2": "4.0",
             "num_error_cells": "1", "target_sse_K2": "100.0",
             "interface_sse_K2": "1.0", "num_interface_cells": "1",
             "interface_target_sse_K2": "25.0"},
        ]
        sims = pool_to_sims(rows)
        assert sims[0]["n_pairs"] == 2
        # Pooled, not averaged: sqrt(18/4), not mean(1, 3).
        assert sims[0]["rmse_K"] == pytest.approx(math.sqrt(18.0 / 4.0))
        assert sims[0]["mean_pair_rmse_K"] == pytest.approx(2.0)
        assert sims[0]["rel_l2_pct"] == pytest.approx(
            100.0 * math.sqrt(18.0 / 500.0))
        assert sims[0]["iface_rmse_K"] == pytest.approx(math.sqrt(5.0 / 2.0))
        assert sims[1]["rmse_K"] == pytest.approx(2.0)


class TestAnchorCheck:
    ANCHOR = {0: {"rmse_K": 0.10}, 1: {"rmse_K": 0.20}, 2: {"rmse_K": 0.30}}
    ATTRS = {0: {"spatial_family": "uniform"}, 1: {"spatial_family": "gaussian"},
             2: {"spatial_family": "uniform"}}

    def test_compares_only_the_crn_matched_uniform_sims(self):
        # sim 1 is gaussian in the ID arm, so it is not a matched pair; including
        # it would compare different physics and manufacture a spurious FAIL.
        ident = {0: {"rmse_K": 0.10}, 1: {"rmse_K": 0.99}, 2: {"rmse_K": 0.30}}
        chk = anchor_check(self.ANCHOR, ident, self.ATTRS)
        assert chk["n"] == 2
        assert chk["status"] == "PASS"
        assert chk["max_rel"] == pytest.approx(0.0)

    def test_flags_a_real_mismatch(self):
        ident = {0: {"rmse_K": 0.10}, 1: {"rmse_K": 0.20}, 2: {"rmse_K": 0.33}}
        chk = anchor_check(self.ANCHOR, ident, self.ATTRS)
        assert chk["status"] == "FAIL"
        assert chk["max_rel"] == pytest.approx(0.03 / 0.33)

    def test_none_when_no_matched_uniform_sim(self):
        attrs = {i: {"spatial_family": "patch"} for i in range(3)}
        assert anchor_check(self.ANCHOR, {0: {"rmse_K": 0.1}}, attrs) is None
        assert anchor_check(self.ANCHOR, None, self.ATTRS) is None


class TestLeadBinning:
    @staticmethod
    def _rows(leads, sims=3):
        out = []
        for s in range(sims):
            for t in leads:
                out.append({"sim_id": str(s), "t_bar": str(t), "rmse_K": "1.0",
                            "sse_K2": "1.0", "num_error_cells": "1",
                            "target_sse_K2": "100.0", "interface_sse_K2": "1.0",
                            "num_interface_cells": "1",
                            "interface_target_sse_K2": "50.0"})
        return out

    def test_float32_noise_does_not_split_one_physical_lead(self):
        # Verbatim t_bar strings from a real records CSV: three physical leads
        # appear as five float64 values differing at ~1e-8.
        rows = self._rows([0.10000000149011612, 0.10000000894069672,
                           0.20000000298023224, 0.20000001043081284,
                           0.30000001192092896])
        edges = lead_bin_edges(rows)
        _, labels = pool_to_sim_lead(rows, edges)
        assert [lab for _, lab in labels] == [
            "[0.100, 0.100]", "[0.200, 0.200]", "[0.300, 0.300]"]

    def test_a_repeated_lead_never_splits_across_bins(self):
        rows = self._rows([0.1, 0.1, 0.2, 0.2, 0.3, 0.3])
        edges = lead_bin_edges(rows)
        _, labels = pool_to_sim_lead(rows, edges)
        assert len({lab for _, lab in labels}) == len(labels)
        seen = {}
        for r in rows:
            b = int(np.searchsorted(edges, round(float(r["t_bar"]), 6), side="left"))
            seen.setdefault(round(float(r["t_bar"]), 6), set()).add(b)
        assert all(len(v) == 1 for v in seen.values())

    def test_bins_capped_by_distinct_lead_count(self):
        rows = self._rows([0.1, 0.1, 0.1])
        edges = lead_bin_edges(rows)
        assert edges == []
        _, labels = pool_to_sim_lead(rows, edges)
        assert len(labels) == 1

    def test_edges_shared_across_arms_give_matching_keys(self):
        a = self._rows([0.1, 0.2, 0.3, 0.4])
        b = self._rows([0.1, 0.2, 0.3, 0.4])
        edges = lead_bin_edges(a)
        ka, _ = pool_to_sim_lead(a, edges)
        kb, _ = pool_to_sim_lead(b, edges)
        assert set(ka) == set(kb)
