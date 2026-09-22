"""Statistical and publication-export contracts for F27/F28/F32."""

import json
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import yaml
from PIL import Image
from scipy.integrate import quad

from visual.pub import records, registry, stats, style, tables
from visual.pub.manifest import FigureSource, Manifest, ProvenanceError, audit
from src.physics.internal_source import make_rc_sin_profile


KEYS = ("F27_global_field_error_vs_lead", "F28_global_field_error_vs_itr")
FIXED_SOURCE_KEY = "F32_global_field_error_fixed_source"
SURFACE_KEY = "F33_source_lead_error_surface"


def field_records(n_sims=25, seed_counts=None, grid=None):
    grid = np.asarray([0.0, 0.04, 0.10, 0.16] if grid is None else grid, dtype=float)
    frames = {}
    seed_counts = seed_counts or {}
    for bidx, benchmark in enumerate(stats.GLOBAL_FIELD_BENCHMARKS):
        rows = []
        for seed in range(seed_counts.get(benchmark, 1)):
            for sim in range(n_sims):
                for s in range(len(grid) - 1):
                    for j in range(s + 1, len(grid)):
                        cells = 10 + s + j
                        error = (0.03 + sim / 200) * (1 + grid[j] - grid[s]) * (1 + bidx) * (seed + 1)
                        row = {name: np.nan for name in records.SCHEMA_V1_COLUMNS}
                        row.update({"seed": str(seed), "benchmark": benchmark, "sim_id": sim,
                                    "s": s, "j": j, "t_s": grid[s], "t_bar": grid[j] - grid[s],
                                    "R_c": 0.05 + 0.95 * sim / max(n_sims - 1, 1),
                                    "R_c_A": 0.8 if benchmark == "source_itr_sin" else np.nan,
                                    "rmse_K": error, "sse_K2": error ** 2 * cells,
                                    "num_error_cells": cells, "target_sse_K2": 100.0,
                                    "interface_sse_K2": error ** 2,
                                    "num_interface_cells": 1, "interface_target_sse_K2": 100.0})
                        rows.append(row)
        frames[benchmark] = pd.DataFrame(rows)
    return frames


def summarize(frames, **kwargs):
    return stats.global_field_error_summary(frames, n_boot=250, **kwargs)


def test_pooling_precedes_seed_mean_and_cohort_median():
    frames = field_records(n_sims=3, seed_counts={"forcing": 2})
    f = frames["forcing"]
    f.loc[:, "sse_K2"] = 0.0
    # Unequal cell counts make sqrt(sum SSE / sum N) differ from mean pair RMSE.
    for seed, multiplier in (("0", 1), ("1", 9)):
        mask = (f.seed == seed) & (f.sim_id == 1)
        f.loc[mask, "sse_K2"] = np.array([100, 1, 4, 9, 16, 25]) * multiplier
        f.loc[mask, "num_error_cells"] = [1, 99, 2, 3, 4, 5]
        f.loc[(f.seed == seed) & (f.sim_id == 2), "sse_K2"] = 10000 * multiplier
    summary = summarize(frames)
    row = next(r for r in summary["lead"] if r["benchmark"] == "forcing" and np.isclose(r["lead_time"], .06))
    pairs = f[(f.seed == "0") & (f.sim_id == 1) & np.isclose(f.t_bar, .06)]
    expected = 2 * np.sqrt(pairs.sse_K2.sum() / pairs.num_error_cells.sum())
    assert row["median_rmse_K"] == pytest.approx(expected)
    assert row["eligible_pairs_by_seed"] == {"0": 6, "1": 6}
    assert row["eligible_pair_rows_total"] == 12
    assert row["unique_simulation_pairs"] == 6
    assert row["n_simulations"] == 3
    assert summary["unequal_seed_counts"] is True
    assert summary["seed_counts"]["forcing"] == 2
    assert row["interval_method"] == "none"
    assert row["ci_lower_K"] is None
    # The ITR reduction uses all six eligible pairs, not a mean of their RMSEs.
    f.loc[:, "R_c"] = 0.0
    row = next(r for r in summarize(frames)["itr"] if r["benchmark"] == "forcing" and r["n_simulations"])
    pairs = f[(f.seed == "0") & (f.sim_id == 1)]
    assert row["median_rmse_K"] == pytest.approx(2 * np.sqrt(pairs.sse_K2.sum() / pairs.num_error_cells.sum()))


def test_unequal_pair_counts_do_not_weight_cohort_median():
    frames = field_records(n_sims=3)
    f = frames["forcing"]
    # One simulation has one pair at .06, while two have two pairs each.
    f = f.loc[~((f.sim_id == 0) & (f.s == 1) & (f.j == 2))].copy()
    f["sse_K2"] = (f.sim_id.map({0: 1.0, 1: 2.0, 2: 10.0}) ** 2) * f.num_error_cells
    frames["forcing"] = f
    row = next(r for r in summarize(frames)["lead"] if r["benchmark"] == "forcing" and np.isclose(r["lead_time"], .06))
    assert row["median_rmse_K"] == 2.0
    assert row["n_simulations"] == 3
    assert row["eligible_pair_rows_total"] == 5


def test_bootstrap_is_over_seed_averaged_simulations_and_reproducible():
    frames = field_records(seed_counts={"forcing": 2})
    first = summarize(frames)
    assert first == summarize({b: f.sample(frac=1, random_state=3) for b, f in frames.items()})
    row = next(r for r in first["lead"] if r["benchmark"] == "forcing" and np.isclose(r["lead_time"], .16))
    values = (0.03 + np.arange(25) / 200) * 1.16 * 1.5
    lo, hi, method = stats.bootstrap_ci(values, n_boot=250)
    assert row["ci_lower_K"] == pytest.approx(lo)
    assert row["ci_upper_K"] == pytest.approx(hi)
    assert row["interval_method"] == method


@pytest.mark.parametrize("defect,match", [
    ("cohort", "cohorts"), ("duplicates", "duplicate"),
    ("protocol", "protocol"), ("time", "snapshot"),
    ("resistance", "resistance changes"), ("v1", "missing columns"),
])
def test_rejects_incompatible_record_inputs(defect, match):
    frames = field_records(n_sims=3, seed_counts={"forcing": 2})
    f = frames["forcing"]
    if defect == "cohort":
        frames["forcing"] = f.loc[~((f.seed == "1") & (f.sim_id == 2))]
    elif defect == "duplicates":
        frames["forcing"] = pd.concat([f, f.iloc[:1]])
    elif defect == "protocol":
        f["protocol"] = "autoregressive"
    elif defect == "time":
        f.loc[0, "t_bar"] += .01
    elif defect == "resistance":
        f.loc[0, "R_c"] += .1
    else:
        frames["forcing"] = f.drop(columns="sse_K2")
    with pytest.raises(ProvenanceError, match=match):
        summarize(frames)


def test_grid_resolves_float_duplicates_and_rejects_protocol_mismatch():
    frames = field_records(n_sims=3, seed_counts={"forcing": 2})
    frames["forcing"].loc[frames["forcing"].seed == "1", "t_bar"] += 1e-10
    metadata = {b: {seed: {"t_grid": [0, .04, .1, .16]} for seed in f.seed.unique()}
                for b, f in frames.items()}
    summary = summarize(frames, metadata=metadata)
    lead = [r for r in summary["lead"] if r["benchmark"] == "forcing"]
    assert len(lead) == 5
    assert all(r["n_simulations"] == 3 for r in lead)
    frames["source"] = frames["source"].loc[frames["source"].s == 0]
    with pytest.raises(ProvenanceError, match="protocols"):
        summarize(frames, metadata=metadata)


@pytest.mark.parametrize("with_grid_metadata", [False, True])
def test_float32_uniform_leads_pool_all_start_times(with_grid_metadata):
    grid = np.linspace(0, .3, 31, dtype=np.float32)
    frames = field_records(n_sims=3, seed_counts={"forcing": 2}, grid=grid)
    for frame in frames.values():
        # Start-dependent errors expose accidental pooling of only a subset of
        # source times at each nominal lead, even when every sim is present.
        frame["sse_K2"] = (1 + frame.s) ** 2 * frame.num_error_cells
    metadata = ({b: {seed: {"t_grid": grid.tolist()} for seed in f.seed.unique()}
                 for b, f in frames.items()} if with_grid_metadata else None)
    result = summarize(frames, metadata=metadata)
    for benchmark, frame in frames.items():
        rows = [r for r in result["lead"] if r["benchmark"] == benchmark]
        assert len(rows) == 30
        assert [r["lead_time"] for r in rows] == pytest.approx(np.arange(1, 31) * .01)
        for lag, row in enumerate(rows, start=1):
            pairs = frame[(frame.seed == "0") & (frame.sim_id == 0)
                          & (frame.j - frame.s == lag)]
            expected = np.sqrt(pairs.sse_K2.sum() / pairs.num_error_cells.sum())
            assert row["median_rmse_K"] == pytest.approx(expected)
            assert row["eligible_pairs_by_seed"] == {
                seed: 3 * (31 - lag) for seed in frame.seed.unique()}
    assert sum(r["eligible_pair_rows_total"] for r in result["lead"]) == sum(
        len(frame) for frame in frames.values())


def test_nonuniform_grid_preserves_distinct_leads_with_equal_index_lags():
    grid = np.array([0, .01, .021, .04], dtype=np.float32)
    frames = field_records(n_sims=3, grid=grid)
    result = summarize(frames)
    rows = [r for r in result["lead"] if r["benchmark"] == "forcing"]
    assert len(rows) == 6
    assert [r["lead_time"] for r in rows] == pytest.approx([.01, .011, .019, .021, .03, .04])


@pytest.mark.parametrize("bounds,amp", [((0, 1), 0), ((0, 1), 2), ((-.2, .8), 1)])
def test_sinusoidal_mean_matches_finite_domain_quadrature(bounds, amp):
    f = pd.DataFrame({"R_c": [.2], "R_c_A": [amp]})
    expected = quad(lambda y: float(make_rc_sin_profile(np.array(y), .2, amp)), *bounds)[0]
    expected /= bounds[1] - bounds[0]
    assert stats.interface_mean_resistance(f, "source_itr_sin", bounds)[0] == pytest.approx(expected)


def test_bins_are_common_seed_invariant_and_preserve_fixed_zero_resistance():
    frames = field_records(n_sims=25)
    frames["forcing"]["R_c"] = 0.0
    result = summarize(frames)
    forcing = [r for r in result["itr"] if r["benchmark"] == "forcing"]
    assert sum(r["n_simulations"] > 0 for r in forcing) == 1
    assert next(r for r in forcing if r["n_simulations"])["median_resistance"] == 0
    extra = frames["source"].assign(seed="1")
    frames["source"] = pd.concat([frames["source"], extra])
    assert summarize(frames)["resistance_bin_edges"] == result["resistance_bin_edges"]
    for f in frames.values():
        f["R_c"] = 0.0
        f["R_c_A"] = 0.0
    result = summarize(frames)
    assert result["resistance_bin_edges"] == [0.0, 0.0]
    assert len(result["itr"]) == 4


def test_shared_rmse_scale_falls_back_to_linear_without_clipping_zero():
    frames = field_records(n_sims=3)
    frames["forcing"]["sse_K2"] = 0
    result = summarize(frames)
    assert result["yscale"] == "linear"
    assert result["ylim"][0] == 0


def test_fixed_source_conditions_on_one_snapshot_and_keeps_the_cohort_constant():
    frames = field_records(n_sims=25)
    for frame in frames.values():
        # Error depends only on the source snapshot, so pooling s=1 with s=2 at
        # lead .06 -- which F27 does -- cannot reproduce the conditioned value.
        frame["sse_K2"] = (1.0 + frame.s) ** 2 * frame.num_error_cells
    result = stats.fixed_source_lead_summary(frames, n_boot=250)
    assert result["primary"]["source_index"] == 1
    assert result["primary"]["source_time"] == pytest.approx(.04)
    assert result["primary"]["requested_time"] == pytest.approx(.032)
    rows = [r for r in result["rows"] if r["is_primary"] and r["benchmark"] == "forcing"]
    assert [r["lead_time"] for r in rows] == pytest.approx([.06, .12])
    assert all(r["n_simulations"] == 25 for r in rows)
    assert all(r["unique_simulation_pairs"] == 25 for r in rows)
    assert all(r["median_rmse_K"] == pytest.approx(2.0) for r in rows)
    pooled = next(r for r in summarize(frames)["lead"]
                  if r["benchmark"] == "forcing" and np.isclose(r["lead_time"], .06))
    assert pooled["median_rmse_K"] != pytest.approx(2.0)


def test_fixed_source_sweep_exports_every_fraction_with_a_single_primary():
    frames = field_records(n_sims=3)
    result = stats.fixed_source_lead_summary(frames, source_fraction=.6, n_boot=50)
    assert result["primary"]["source_index"] == 2
    assert sorted(result["selections"]) == ["0.1", "0.2", "0.4", "0.6"]
    assert [k for k, v in result["selections"].items() if v["is_primary"]] == ["0.6"]
    assert {r["source_fraction"] for r in result["rows"]} == set(stats.SOURCE_FRACTION_SWEEP)
    assert {r["source_fraction"] for r in result["rows"] if r["is_primary"]} == {.6}
    # The sweep still reaches the initial condition the default deliberately avoids.
    assert result["selections"]["0.1"]["source_index"] == 0
    with pytest.raises(ProvenanceError, match="source fraction"):
        stats.fixed_source_lead_summary(frames, source_fraction=1.5)


def shape_records(*, n_sims=30, n_snap=12, dt=.02, log_level=None, log_shape=None,
                  noise=0.0, rng_seed=0):
    """Records whose log10 error is ``log_level(s) + log_shape(s, k)`` per simulation.

    The shape reduction needs a grid long enough to hold several lead windows,
    which the four-snapshot ``field_records`` grid is not. Writing the error in
    log space lets a test plant a pure level change or a pure shape change and
    assert the decomposition separates them.
    """
    log_level = log_level or (lambda s: 0.0)
    log_shape = log_shape or (lambda s, k: .5 * np.log10(k))
    rng = np.random.default_rng(rng_seed)
    frames = {}
    for benchmark in stats.GLOBAL_FIELD_BENCHMARKS:
        offsets = rng.normal(0, .08, n_sims)
        rows = []
        for s in range(n_snap - 1):
            for j in range(s + 1, n_snap):
                k, cells = j - s, 16
                error = 10.0 ** (np.log10(.05) + log_level(s) + log_shape(s, k) + offsets
                                 + noise * rng.normal(0, 1, n_sims))
                for sim in range(n_sims):
                    row = {name: np.nan for name in records.SCHEMA_V1_COLUMNS}
                    row.update({"seed": "0", "benchmark": benchmark, "sim_id": sim,
                                "s": s, "j": j, "t_s": s * dt, "t_bar": k * dt,
                                "R_c": .05 + .95 * sim / max(n_sims - 1, 1),
                                "R_c_A": .8 if benchmark == "source_itr_sin" else np.nan,
                                "rmse_K": error[sim], "sse_K2": error[sim] ** 2 * cells,
                                "num_error_cells": cells, "target_sse_K2": 100.0,
                                "interface_sse_K2": error[sim] ** 2,
                                "num_interface_cells": 1, "interface_target_sse_K2": 100.0})
                    rows.append(row)
        frames[benchmark] = pd.DataFrame(rows)
    return frames


def typicality(frames, **kwargs):
    kwargs.setdefault("n_boot", 200)
    return stats.source_time_shape_typicality(frames, **kwargs)


def forcing_rows(result):
    return [r for r in result["rows"] if r["benchmark"] == "forcing"]


def test_shape_windows_share_support_and_shrink_as_the_window_grows():
    result = typicality(shape_records())
    assert result["max_lag"] == 11
    assert result["windows"] == [9, 4, 2]
    assert result["published"]["source_index"] == 2
    rows = forcing_rows(result)
    assert [r["n_curves"] for r in rows] == [3, 8, 10]
    for row in rows:
        # Every retained curve must reach every lead in the window, or the
        # comparison would read window length as shape.
        assert row["source_indices"] == list(range(result["max_lag"] - row["window_lags"] + 1))
        assert row["window_lead_time"] == pytest.approx(row["window_lags"] * .02)
    # All four benchmarks are compared over the same source times per window.
    for window in result["windows"]:
        supports = {tuple(r["source_indices"]) for r in result["rows"]
                    if r["window_lags"] == window}
        assert len(supports) == 1


def test_identical_shapes_leave_no_interaction_whatever_the_level():
    # One source time's curve is lifted bodily by a constant factor. In log
    # space that is alpha, not interaction, so the shape verdict must ignore it.
    result = typicality(shape_records(log_level=lambda s: .3 * (s == 2)))
    for row in forcing_rows(result):
        assert row["interaction_pct"] == pytest.approx(0.0, abs=1e-9)
        assert row["common_shape_fraction"] == pytest.approx(1.0)
        assert row["level_spread_pct"] > 90


def test_planted_bend_is_detected_and_attributed_to_its_source_time():
    bent = 3
    result = typicality(shape_records(
        log_shape=lambda s, k: .5 * np.log10(k) + .4 * np.log10(k) * (s == bent)))
    rows = [r for r in forcing_rows(result) if bent in r["source_indices"]]
    assert rows
    for row in rows:
        assert row["interaction_pct"] > 1.0
        assert row["least_typical_index"] == bent
        assert row["common_shape_fraction"] < 1.0
    worst = max((r for r in result["sources"] if r["benchmark"] == "forcing"
                 and r["window_lags"] == rows[0]["window_lags"]),
                key=lambda r: r["residual_rms_log10"])
    assert worst["source_index"] == bent


def test_noise_alone_neither_inflates_with_window_nor_singles_out_a_curve():
    # The failure this guards against: a per-curve statistic measured over a
    # window whose length is a function of the source index will rank the
    # curves even when they are identical.
    result = typicality(shape_records(noise=.02, rng_seed=5))
    rows = forcing_rows(result)
    interactions = [r["interaction_pct"] for r in rows]
    assert max(interactions) < 2 * min(interactions)
    covering = [r for r in rows if r["covers_published"]]
    assert covering
    for row in covering:
        assert row["rank_span_fraction"] > .8
        assert row["rank_ci_lower"] <= row["published_rank"] <= row["rank_ci_upper"]


def test_tied_curves_all_rank_first_rather_than_being_ordered_by_dust():
    # Noiseless identical curves separate only by floating-point residue.
    # Ranking on that names a least-typical source time that does not exist and
    # strands the published rank outside its own interval.
    for row in forcing_rows(typicality(shape_records(n_sims=25))):
        assert row["interaction_pct"] == pytest.approx(0.0, abs=1e-6)
        assert row["published_rank"] == 1
        assert row["rank_ci_lower"] <= row["published_rank"] <= row["rank_ci_upper"]


def test_incomplete_pair_cohort_is_excluded_and_disclosed():
    frames = shape_records(n_sims=25)
    f = frames["forcing"]
    frames["forcing"] = f.drop(f.index[(f.sim_id == 7) & (f.s == 1) & (f.j == 5)])
    result = typicality(frames)
    assert result["cohort"]["forcing"] == {"n_simulations": 24, "n_dropped": 1,
                                           "dropped_sim_ids": [7]}
    assert result["cohort"]["source"]["n_dropped"] == 0
    assert [d.code for d in result["degradations"]] == ["incomplete_pair_cohort"]
    assert all(r["n_simulations"] == 24 for r in forcing_rows(result))


def test_bootstrap_rank_interval_is_reproducible_and_seed_dependent():
    frames = shape_records(noise=.02, rng_seed=3)
    span = lambda r: [(x["rank_ci_lower"], x["rank_ci_upper"]) for x in forcing_rows(r)]
    assert span(typicality(frames)) == span(typicality(frames))
    assert span(typicality(frames, rng_seed=stats.DEFAULT_RNG_SEED + 1)) != []


def test_shape_comparison_refuses_short_and_nonuniform_grids():
    short = field_records(n_sims=25, grid=np.arange(4) * .02)
    with pytest.raises(ProvenanceError, match="too short"):
        stats.source_time_shape_typicality(short, n_boot=0)
    grid = np.array([0, .01, .021, .04, .07, .11, .16, .22], dtype=float)
    frames = field_records(n_sims=25, grid=grid)
    with pytest.raises(ProvenanceError, match="non-uniform snapshot grid"):
        stats.source_time_shape_typicality(frames, n_boot=0)


def write_field_manifest(root, n_sims=25, frames=None):
    sources = {}
    for benchmark, f in (field_records(n_sims=n_sims) if frames is None else frames).items():
        run = root / benchmark / "seed0"
        run.mkdir(parents=True)
        f.to_csv(run / "test_records.csv", index=False)
        (run / "config_used.yaml").write_text(yaml.safe_dump({
            "benchmark": {"name": benchmark, "representation": "temporal_encoder"},
            "evaluation": {"rollout": {"enabled": False}}}))
        sources[f"{benchmark}_records"] = [{"run": str(run), "seed": 0}]
    path = root / "manifest.yaml"
    path.write_text(yaml.safe_dump({"sources": sources}))
    return Manifest.load(path)


def test_builders_share_scale_and_keep_itr_markers_dominant(monkeypatch):
    frames = field_records()
    frames["forcing"]["R_c"] = 0.0
    reduced = summarize(frames)
    monkeypatch.setattr(records, "load_global_field_records", lambda source: (frames, {}))
    monkeypatch.setattr(stats, "global_field_error_summary", lambda *a, **k: reduced)
    figures = []
    for key in KEYS:
        fig, _, definition = registry.get_figure(key).load()(source=FigureSource(key))
        figures.append(fig)
        assert len(fig.axes) == 1
        assert fig.get_size_inches() == pytest.approx([3.42, 2.6])
        assert len(fig.legends[0].get_texts()) == 4
        assert definition["statistics"]
        assert "pointwise" in definition["interval"]
        assert "not training-seed variability" in definition["interval_interpretation"]
    assert figures[0].axes[0].get_ylim() == figures[1].axes[0].get_ylim()
    itr = figures[1].axes[0]
    assert itr.get_xscale() == "linear"
    connectors = [line for line in itr.lines if line.get_alpha() == .45]
    assert len(connectors) == 4
    assert all(line.get_linewidth() == .6 for line in connectors)
    assert any(np.isnan(line.get_xdata()).any() for line in connectors)
    for fig in figures:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for text in [*fig.axes[0].get_xticklabels(), *fig.axes[0].get_yticklabels(),
                     fig.axes[0].xaxis.label, fig.axes[0].yaxis.label, *fig.legends[0].get_texts()]:
            if not text.get_visible() or not text.get_text():
                continue
            bbox = text.get_window_extent(renderer)
            # Locators may instantiate invisible, out-of-range tick labels.
            if text in fig.axes[0].get_yticklabels() and not fig.axes[0].get_ylim()[0] <= text.get_position()[1] <= fig.axes[0].get_ylim()[1]:
                continue
            if text in fig.axes[0].get_xticklabels() and not fig.axes[0].get_xlim()[0] <= text.get_position()[0] <= fig.axes[0].get_xlim()[1]:
                continue
            assert bbox.x0 >= 0 and bbox.y0 >= 0
            assert bbox.x1 <= fig.bbox.width and bbox.y1 <= fig.bbox.height
        plt.close(fig)


def test_render_exports_vectors_dimensions_statistics_and_provenance(tmp_path):
    manifest = write_field_manifest(tmp_path)
    for key in KEYS:
        result = registry.render(key, manifest=manifest, out_dir=tmp_path / "out", strict=False)
        assert result.degraded
        assert [p.suffix for p in result.paths] == [".png", ".pdf", ".svg", ".csv"]
        png, pdf, svg, table = result.paths
        with Image.open(png) as image:
            assert image.size == (1026, 780)
        assert b"/FontFile2" in pdf.read_bytes()
        assert b"/Subtype /Type3" not in pdf.read_bytes()
        box = re.search(rb"/MediaBox\s*\[([^\]]+)\]", pdf.read_bytes())
        assert list(map(float, box[1].split())) == pytest.approx([0, 0, 246.24, 187.2])
        assert "<text" in svg.read_text()
        payload = json.loads(result.sidecar.read_text())
        definition = payload["metric_definition"]
        plotted = pd.read_csv(table)
        assert len(plotted) == len(definition["statistics"])
        assert set(plotted.benchmark) == set(stats.GLOBAL_FIELD_BENCHMARKS)
        assert set(definition["seed_counts"].values()) == {1}
        assert "pointwise" in definition["caption"].lower()
        assert payload["degradations"]
    assert all(row["status"] == "OK" for row in audit(tmp_path / "out"))
    assert {"F08_forcing_difficulty", "F10_source_difficulty", "F12_source_itr_sin_resistance", "F14_interfaces_difficulty"} <= set(registry.FIGURES)


def test_fixed_source_figure_draws_only_the_primary_fraction(monkeypatch):
    frames = field_records()
    reduced = stats.fixed_source_lead_summary(frames)
    monkeypatch.setattr(records, "load_global_field_records", lambda source: (frames, {}))
    fig, _, definition = registry.get_figure(FIXED_SOURCE_KEY).load()(
        source=FigureSource(FIXED_SOURCE_KEY))
    ax = fig.axes[0]
    assert fig.get_size_inches() == pytest.approx([3.42, 2.6])
    assert len(fig.legends[0].get_texts()) == 4
    assert len(ax.lines) == 4 and len(ax.collections) == 4
    primary = [r for r in reduced["rows"] if r["is_primary"] and r["benchmark"] == "forcing"]
    assert ax.lines[0].get_xdata() == pytest.approx([r["lead_time"] for r in primary])
    assert len(definition["statistics"]) == len(reduced["rows"])
    assert sorted(definition["source_fraction_sweep"]) == ["0.1", "0.2", "0.4", "0.6"]
    assert definition["prediction_mode"].startswith("direct_pair")
    assert "not error accumulation" in definition["caption"]
    plt.close(fig)


def test_fixed_source_render_exports_the_whole_sweep(tmp_path):
    manifest = write_field_manifest(tmp_path)
    result = registry.render(FIXED_SOURCE_KEY, manifest=manifest,
                             out_dir=tmp_path / "out", strict=False)
    assert [p.suffix for p in result.paths] == [".png", ".pdf", ".svg", ".csv"]
    with Image.open(result.paths[0]) as image:
        assert image.size == (1026, 780)
    plotted = pd.read_csv(result.paths[3])
    assert set(plotted.source_fraction) == set(stats.SOURCE_FRACTION_SWEEP)
    drawn = plotted.loc[plotted.is_primary]
    assert drawn.source_time.nunique() == 1
    assert drawn.n_simulations.nunique() == 1
    definition = json.loads(result.sidecar.read_text())["metric_definition"]
    assert definition["source_time_selection"]["source_index"] == 1


def surface(frames=None, **kwargs):
    return stats.source_lead_error_surface(
        shape_records(**kwargs) if frames is None else frames)


def test_incomplete_fit_reduces_to_row_and_column_means_on_a_full_grid():
    """The whole reason the surface solves rather than averages.

    Row and column means are the least-squares additive fit only when every
    cell is observed, which is the windowed comparison's balanced design but
    never the surface's triangular one. If the two disagree on a full grid the
    solver is wrong, not merely different.
    """
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(5, 7))
    rows, cols = np.divmod(np.arange(matrix.size), 7)
    grand, alpha, beta = stats._additive_fit_incomplete(matrix.ravel(), rows, cols, 5, 7)
    _, expected_alpha, expected_beta = stats._additive_interaction(matrix)
    assert grand == pytest.approx(matrix.mean())
    assert alpha == pytest.approx(expected_alpha)
    assert beta == pytest.approx(expected_beta)


def test_disconnected_cells_are_refused_rather_than_fitted():
    # Two blocks sharing no row and no column: the level difference between them
    # is unmeasurable, and lstsq would silently return one anyway.
    values = np.arange(4.0)
    rows, cols = [0, 0, 1, 1], [0, 1, 2, 3]
    with pytest.raises(ProvenanceError, match="disconnected"):
        stats._additive_fit_incomplete(values, rows, cols, 2, 4)


def test_surface_separates_a_level_shift_from_a_shape_change():
    # n_snap=10 leaves source indices 0..8, so the planted level spans .2 * 8.
    level = surface(n_snap=10, log_level=lambda s: .2 * s)["surfaces"]["interfaces"]
    assert level["level_spread_pct"] == pytest.approx(100 * (10 ** 1.6 - 1), rel=1e-6)
    assert level["interaction_pct"] == pytest.approx(0, abs=1e-6)
    assert level["common_shape_fraction"] == pytest.approx(1.0)

    # A bend that only the late source times have cannot be absorbed by a
    # per-source constant, so it lands in the residual and nowhere else.
    bent = surface(n_snap=10,
                   log_shape=lambda s, k: .5 * np.log10(k) + .4 * (s >= 5) * np.log10(k),
                   )["surfaces"]["interfaces"]
    assert bent["interaction_pct"] > 5
    assert bent["common_shape_fraction"] < level["common_shape_fraction"]
    assert bent["level_spread_pct"] < level["level_spread_pct"]


def test_surface_is_triangular_and_its_fit_reproduces_the_observed_cells():
    result = surface(n_snap=10)
    data = result["surfaces"]["source_itr_sin"]
    observed = data["surface_log10"]
    assert observed.shape == (len(result["source_indices"]), len(result["lags"]))
    evaluated = np.argwhere(np.isfinite(observed))
    # Cell (s, k) exists exactly when the target snapshot s + k is on the grid.
    assert {(int(s), int(k) + 1) for s, k in evaluated} == {
        (s, k) for s in result["source_indices"]
        for k in result["lags"] if s + k <= result["max_lag"]}
    assert data["n_cells"] == len(evaluated)
    assert np.isnan(data["additive_fit_log10"][~np.isfinite(observed)]).all()
    residual = observed - data["additive_fit_log10"]
    assert np.nanmax(np.abs(residual - data["residual_log10"])) == pytest.approx(0, abs=1e-12)
    # Recentered offsets, so the fit reads as a grand level plus two deviations.
    assert data["level_log10"].mean() == pytest.approx(0, abs=1e-12)
    assert data["shape_log10"].mean() == pytest.approx(0, abs=1e-12)


def test_surface_refuses_an_unknown_benchmark_and_a_thin_cohort():
    with pytest.raises(ProvenanceError, match="unknown global-field"):
        stats.source_lead_error_surface(shape_records(), benchmarks=("typo",))
    with pytest.raises(ProvenanceError, match="at least"):
        stats.source_lead_error_surface(shape_records(n_sims=2))


def test_surface_discloses_an_incomplete_cohort_without_dropping_a_cell():
    frames = shape_records(n_sims=20, n_snap=8)
    f = frames["interfaces"]
    frames["interfaces"] = f.loc[~((f.sim_id == 3) & (f.s == 1) & (f.j == 4))]
    result = stats.source_lead_error_surface(frames)
    assert result["surfaces"]["interfaces"]["n_simulations"] == 19
    assert result["surfaces"]["interfaces"]["n_dropped"] == 1
    # The other benchmark keeps its full cohort; the exclusion is not global.
    assert result["surfaces"]["source_itr_sin"]["n_simulations"] == 20
    assert any(d.code == "incomplete_pair_cohort" for d in result["degradations"])


def test_surface_figure_titles_each_panel_and_masks_the_unevaluated_cells(monkeypatch):
    frames = shape_records(n_sims=20, n_snap=10)
    reduced = stats.source_lead_error_surface(frames)
    monkeypatch.setattr(records, "load_global_field_records", lambda source: (frames, {}))
    fig, _, definition = registry.get_figure(SURFACE_KEY).load()(
        source=FigureSource(SURFACE_KEY))
    # Colorbar axes carry a collection too, so the title distinguishes a panel.
    panels = [ax for ax in fig.axes if ax.get_title()]
    assert len(panels) == len(stats.SURFACE_BENCHMARKS)
    assert len(fig.axes) == len(stats.SURFACE_BENCHMARKS) + 1
    assert fig.get_size_inches() == pytest.approx([8.4, 4.32])
    assert not fig.texts, "the panel titles carry the naming; no figure-level title"
    # The panel title overrides the shared table label for interfaces only.
    assert [ax.get_title() for ax in panels] == [
        "Varying interface", tables.BENCH_LABEL["source_itr_sin"]]
    assert not any(ax.lines for ax in panels), "no source-time marker is drawn"

    all_values = np.concatenate([
        reduced["surfaces"][b]["surface_log10"].ravel()
        for b in stats.SURFACE_BENCHMARKS])
    shared_limits = (np.nanmin(all_values), np.nanmax(all_values))
    for ax, benchmark in zip(panels, stats.SURFACE_BENCHMARKS):
        values = reduced["surfaces"][benchmark]["surface_log10"]
        mesh = ax.collections[0]
        assert mesh.get_clim() == pytest.approx(shared_limits)
        assert mesh.norm is panels[0].collections[0].norm
        assert mesh.get_array().compressed() == pytest.approx(values.T[np.isfinite(values.T)])
    assert fig.axes[-1].get_ylabel() == "Median RMSE [K]"
    assert "Both panels share one color scale" in definition["caption"]
    grey = panels[0].collections[0].get_cmap()(np.ma.masked_invalid([np.nan]))[0]
    assert grey[:3] == pytest.approx((.90, .90, .90), abs=.01)

    total = sum(s["n_cells"] for s in reduced["surfaces"].values())
    assert len(definition["statistics"]) == total
    assert sum(r["is_published_source"] for r in definition["statistics"]) > 0
    assert definition["benchmarks"] == list(stats.SURFACE_BENCHMARKS)
    assert definition["interval"] is None
    assert "not error accumulated through a rollout" in definition["caption"]
    assert "accumulat" not in definition["caption"].replace(
        "not error accumulated through a rollout", "")
    plt.close(fig)


def test_surface_render_exports_one_row_per_evaluated_cell(tmp_path):
    # The default field_records grid is non-uniform, which the surface refuses:
    # lead time has to be a function of snapshot lag for a column to mean
    # anything. shape_records is evenly spaced.
    manifest = write_field_manifest(
        tmp_path, frames=shape_records(n_sims=20, n_snap=10))
    result = registry.render(SURFACE_KEY, manifest=manifest,
                             out_dir=tmp_path / "out", strict=False)
    assert [p.suffix for p in result.paths] == [".png", ".pdf", ".svg", ".csv"]
    plotted = pd.read_csv(result.paths[3])
    assert set(plotted.benchmark) == set(stats.SURFACE_BENCHMARKS)
    assert (plotted.source_index + plotted.lag <= plotted.lag.max() + 1).all()
    assert plotted.loc[plotted.is_published_source].source_time.nunique() == 1
    # The residual is a percentage of the fit, which is what the colorbar claims.
    assert (100 * (plotted.median_rmse_K / plotted.additive_fit_rmse_K - 1)
            ).to_numpy() == pytest.approx(plotted.residual_pct.to_numpy(), abs=1e-6)
    definition = json.loads(result.sidecar.read_text())["metric_definition"]
    assert "triangular domain" in definition["summary_comparability"]


def test_loader_rejects_legacy_rollout_and_reads_grid_metadata(tmp_path):
    manifest = write_field_manifest(tmp_path, n_sims=3)
    run = tmp_path / "forcing" / "seed0"
    path = run / "config_used.yaml"
    config = yaml.safe_load(path.read_text())
    tgrid = run / "t_grid.npy"
    np.save(tgrid, [0, .04, .1, .16])
    config["data"] = {"t_grid_path": str(tgrid)}
    path.write_text(yaml.safe_dump(config))
    source = manifest.resolve(KEYS[0], strict=False)
    _, metadata = records.load_global_field_records(source)
    assert metadata["forcing"]["0"]["t_grid"] == [0, .04, .1, .16]
    assert str(tgrid) in [ref.path for ref in source.artifacts]
    config["evaluation"]["rollout"]["enabled"] = True
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(records.SchemaError, match="rollout"):
        records.load_global_field_records(source)


@pytest.mark.parametrize("defect", [None, "rollout", "identity", "representation", "grid_hash"])
def test_loader_checks_v4_provenance(tmp_path, defect):
    manifest = write_field_manifest(tmp_path, n_sims=3)
    run = tmp_path / "forcing" / "seed0"
    path = run / "test_records.csv"
    frame = pd.read_csv(path)
    for name in records.SCHEMA_V3_REQUIRED + records.SCHEMA_V4_REQUIRED:
        if name not in frame:
            frame[name] = np.nan
    frame["provenance_id"] = "test-population"
    frame.to_csv(path, index=False)
    provenance = {
        "schema": "test-records-provenance/v1", "provenance_id": "test-population",
        "benchmark": "forcing", "seed": "0", "representation": "temporal_encoder",
        "prediction_mode": "direct_pair", "rollout_enabled": False,
        "evaluation_population_hash": "same-simulations",
        "evaluation_population": {"t_grid": [0, .04, .1, .16], "dataset_file_hashes": {}},
    }
    if defect == "rollout":
        provenance["prediction_mode"] = "autoregressive_2_substeps"
    elif defect == "identity":
        provenance["seed"] = "99"
    elif defect == "representation":
        provenance["representation"] = "bins"
    elif defect == "grid_hash":
        ypath = run / "y_grid.npy"
        np.save(ypath, np.linspace(0, 1, 10))
        config_path = run / "config_used.yaml"
        config = yaml.safe_load(config_path.read_text())
        config["data"] = {"y_grid_path": str(ypath)}
        config_path.write_text(yaml.safe_dump(config))
        provenance["evaluation_population"]["dataset_file_hashes"]["y_grid"] = "wrong"
    records.provenance_path_for(path).write_text(json.dumps(provenance))
    source = manifest.resolve(KEYS[0], strict=False)
    if defect is not None:
        with pytest.raises(records.SchemaError):
            records.load_global_field_records(source)
    else:
        frames, metadata = records.load_global_field_records(source)
        assert metadata["forcing"]["0"]["prediction_mode"] == "direct_pair"
        assert summarize(frames, metadata=metadata)["seed_counts"]["forcing"] == 1


def test_legacy_float_roundoff_does_not_split_seeds_into_separate_strata():
    frames = field_records(n_sims=3, seed_counts={"forcing": 2})
    frames["forcing"].loc[frames["forcing"].seed == "1", "t_bar"] += 1e-10
    result = summarize(frames)
    lead = [r for r in result["lead"] if r["benchmark"] == "forcing"]
    assert len(lead) == 5
    assert all(r["eligible_pair_rows_total"] == 2 * r["unique_simulation_pairs"] for r in lead)


def test_seed_population_identity_is_checked_before_averaging():
    frames = field_records(n_sims=3, seed_counts={"forcing": 2})
    metadata = {"forcing": {"0": {"evaluation_population_hash": "A"},
                             "1": {"evaluation_population_hash": "B"}}}
    with pytest.raises(ProvenanceError, match="evaluation populations"):
        summarize(frames, metadata=metadata)


def test_recorded_simulation_parameters_must_match_across_seeds():
    frames = field_records(n_sims=3, seed_counts={"source": 2})
    frames["source"]["x_h"] = .4
    frames["source"].loc[frames["source"].seed == "1", "x_h"] = .6
    with pytest.raises(ProvenanceError, match="simulation parameters"):
        summarize(frames)


def test_seed_with_no_eligible_pairs_is_not_silently_dropped():
    frames = field_records(n_sims=3, seed_counts={"forcing": 2})
    frame = frames["forcing"]
    frame.loc[frame.seed == "1", "j"] = frame.loc[frame.seed == "1", "s"]
    frame.loc[frame.seed == "1", "t_bar"] = 0
    with pytest.raises(ProvenanceError, match="seed has no eligible"):
        summarize(frames)


def test_global_field_discovery_selects_all_seeds_and_rejects_ambiguity(tmp_path):
    import shutil
    manifest = write_field_manifest(tmp_path / 'runs', n_sims=3)
    root = tmp_path / 'runs'
    shutil.copytree(root / 'forcing' / 'seed0', root / 'forcing' / 'seed1')
    discovered = Manifest.discover_global_field(root)
    assert [e['seed'] for e in discovered.sources['forcing_records']] == [0, 1]
    assert discovered.path is None
    assert len(discovered.sources) == 4
    shutil.copytree(root / 'forcing' / 'seed0', root / 'another' / 'seed0')
    with pytest.raises(ProvenanceError, match='--run forcing='):
        Manifest.discover_global_field(root)
    selected = Manifest.discover_global_field(root, selections={'forcing': str(root / 'forcing')})
    assert len(selected.sources['forcing_records']) == 2
    with pytest.raises(ProvenanceError, match='found 0'):
        Manifest.discover_global_field(root, selections={'forcing': str(root / 'absent')})
    with pytest.raises(ProvenanceError, match='Unknown benchmark'):
        Manifest.discover_global_field(root, selections={'typo': str(root)})


def test_global_field_discovery_missing_records_and_config(tmp_path):
    with pytest.raises(ProvenanceError, match='does not exist'):
        Manifest.discover_global_field(tmp_path / 'absent')
    with pytest.raises(ProvenanceError, match='No eligible records'):
        Manifest.discover_global_field(tmp_path)
    (tmp_path / 'test_records.csv').write_text('sim_id\n1\n')
    with pytest.raises(ProvenanceError, match='Missing run configuration'):
        Manifest.discover_global_field(tmp_path)


def test_global_field_shortcut_routes_both_figures_and_preserves_strict(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace
    from visual.pub import __main__ as cli
    write_field_manifest(tmp_path / 'runs', n_sims=3)
    calls = []
    def fake_render(key, **kwargs):
        calls.append((key, kwargs))
        return SimpleNamespace(degraded=not kwargs['strict'], paths=[], sidecar=tmp_path/'preview.json')
    monkeypatch.setattr(cli, 'render', fake_render)
    command = ['--global-field', '--runs-root', str(tmp_path / 'runs')]
    assert cli.main(command) == 2
    assert [key for key, _ in calls] == [*KEYS, FIXED_SOURCE_KEY, SURFACE_KEY]
    assert all(not kw['strict'] and str(kw['out_dir']) == 'figures/global_field' for _, kw in calls)
    assert 'Selected forcing_records' in capsys.readouterr().out
    calls.clear()
    assert cli.main(command + ['--strict', '--out', str(tmp_path / 'figures')]) == 0
    assert all(kw['strict'] and kw['out_dir'] == tmp_path / 'figures' for _, kw in calls)
    for invalid in (['--runs-root', str(tmp_path)], command + ['--all'], command + ['--manifest', 'custom.yaml']):
        with pytest.raises(SystemExit):
            cli.main(invalid)


def test_verify_reports_satisfiability_of_discovered_runs(tmp_path, capsys):
    # The preflight for a cluster: ask what the records already on disk can
    # support, and what each gap needs, before spending a job rendering them.
    from visual.pub import __main__ as cli
    write_field_manifest(tmp_path / 'runs', n_sims=3)
    assert cli.main(['--runs-root', str(tmp_path / 'runs'), '--verify']) == 2
    out = capsys.readouterr().out
    assert 'BLOCK F32_global_field_error_fixed_source' in out
    assert 'figures satisfiable.' in out


def test_discovered_runs_can_back_the_primary_table(tmp_path, monkeypatch, capsys):
    # T01 draws on the same four benchmarks discovery finds, so a cluster with
    # records but no manifest.yaml can write the table as well as the figures.
    from visual.pub import __main__ as cli
    from visual.pub import tables
    write_field_manifest(tmp_path / 'runs', n_sims=3)
    seen = {}
    def fake_write(source, out_dir):
        seen.update(degraded=source.is_degraded, out=out_dir)
        return [out_dir / 'T01.tex']
    monkeypatch.setattr(tables, 'write_descriptive_results', fake_write)
    command = ['--runs-root', str(tmp_path / 'runs'), '--out', str(tmp_path / 'out')]
    assert cli.main([*command, '--table', 'T01_primary_results_descriptive']) == 2
    assert seen == {'degraded': True, 'out': tmp_path / 'out'}
    assert 'Selected forcing_records' in capsys.readouterr().out
    # The strict table still refuses a single-seed cohort rather than shipping it.
    assert cli.main([*command, '--table', 'T01_primary_results']) == 1
    assert 'FAILED T01_primary_results' in capsys.readouterr().err


def test_sinusoidal_records_accept_legacy_empty_gaussian_columns(tmp_path):
    manifest = write_field_manifest(tmp_path, n_sims=3)
    csv = tmp_path / 'source_itr_sin' / 'seed0' / 'test_records.csv'
    frame = pd.read_csv(csv)
    for column in ('R_c_amp', 'R_c_y0', 'R_c_sigma'):
        frame[column] = np.nan
    frame.to_csv(csv, index=False)
    found = Manifest.discover_global_field(tmp_path)
    source = found.resolve(KEYS[0], strict=False)
    frames, metadata = records.load_global_field_records(source)
    result = summarize(frames, metadata=metadata)
    assert result['resistance_definitions']['source_itr_sin']['0']['kind'] == 'finite-domain sinusoidal mean'
