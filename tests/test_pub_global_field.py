"""Statistical and publication-export contracts for F27/F28."""

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

from visual.pub import records, registry, stats, style
from visual.pub.manifest import FigureSource, Manifest, ProvenanceError, audit
from src.physics.internal_source import make_rc_void_profile


KEYS = ("F27_global_field_error_vs_lead", "F28_global_field_error_vs_itr")


def field_records(n_sims=25, seed_counts=None):
    grid = np.array([0.0, 0.04, 0.10, 0.16])
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
                                    "R_c_amp": 0.8 if benchmark == "source_itr" else np.nan,
                                    "R_c_y0": 0.2, "R_c_sigma": 0.1,
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


@pytest.mark.parametrize("bounds,center,amp", [((0, 1), .5, 0), ((0, 1), .01, 2), ((-.2, .8), .79, 1)])
def test_gaussian_mean_matches_finite_domain_quadrature(bounds, center, amp):
    f = pd.DataFrame({"R_c": [.2], "R_c_amp": [amp], "R_c_y0": [center], "R_c_sigma": [.1]})
    expected = quad(lambda y: float(make_rc_void_profile(np.array(y), .2, amp, center, .1)), *bounds)[0]
    expected /= bounds[1] - bounds[0]
    assert stats.interface_mean_resistance(f, "source_itr", bounds)[0] == pytest.approx(expected)


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
        f["R_c_amp"] = 0.0
    result = summarize(frames)
    assert result["resistance_bin_edges"] == [0.0, 0.0]
    assert len(result["itr"]) == 4


def test_shared_rmse_scale_falls_back_to_linear_without_clipping_zero():
    frames = field_records(n_sims=3)
    frames["forcing"]["sse_K2"] = 0
    result = summarize(frames)
    assert result["yscale"] == "linear"
    assert result["ylim"][0] == 0


def write_field_manifest(root, n_sims=25):
    sources = {}
    for benchmark, f in field_records(n_sims=n_sims).items():
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
    assert {"F08_forcing_difficulty", "F10_source_difficulty", "F12_source_itr_void", "F14_interfaces_difficulty"} <= set(registry.FIGURES)


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
