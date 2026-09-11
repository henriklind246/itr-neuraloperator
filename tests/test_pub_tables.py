from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.dataset import build_normalization_provenance, compute_global_stats
from visual.pub import records, tables
from visual.pub.__main__ import build_parser
from visual.pub.manifest import (
    ArtifactRef, FigureSource, Manifest, ProvenanceError, sha256_file,
)


def _write_seed(root: Path, benchmark: str, seed: str, *, mode="direct_pair",
                training_hash=None, representation="temporal_encoder") -> Path:
    path = root / benchmark / f"seed{seed}" / "test_records.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    provenance_id = f"{benchmark}-{seed}-provenance"
    scale_factor = 1.0 + 0.1 * int(seed)
    rows = []
    pairs = ((0, 1, 1.0), (0, 2, 2.0), (1, 2, 1.0))
    for sim_id in range(4):
        for s, j, lead in pairs:
            rmse = scale_factor * (1.0 + sim_id + 2.0 * lead)
            jump = scale_factor * (0.5 + sim_id + lead)
            truth_max = 4.0 + sim_id + j
            pred_max = truth_max + scale_factor * (j - 1.5)
            row = {field: "" for field in records.TEST_RECORD_FIELDS}
            row.update({
                "provenance_id": provenance_id,
                "sim_id": sim_id,
                "s": s,
                "j": j,
                "t_s": float(s),
                "t_bar": lead,
                "benchmark": benchmark,
                "rmse_K": rmse,
                "node_jump_rmse_K": jump,
                "node_jump_abs_max_pred_K": pred_max,
                "node_jump_abs_max_true_K": truth_max,
                "sse_K2": rmse ** 2 * 10.0,
                "num_error_cells": 10,
                "target_sse_K2": 1000.0,
                "interface_sse_K2": jump ** 2 * 2.0,
                "num_interface_cells": 2,
                "interface_target_sse_K2": 100.0,
                "distribution_class": "in_distribution",
                "protocol": "",
                "ood_axis": "",
                "lead_time_actual": lead,
                "time_norm_horizon": 2.0,
            })
            rows.append(row)
    pd.DataFrame(rows, columns=records.TEST_RECORD_FIELDS).to_csv(path, index=False)
    mu, sigma = 304.0, 3.0
    rise = 5.0
    payload = {
        "schema": "test-records-provenance/v1",
        "provenance_id": provenance_id,
        "training_mean_K": mu,
        "training_population_std_K": sigma,
        "training_temperature_rise_rms_K": rise,
        "temperature_reference_K": 300.0,
        "training_population_hash": training_hash or f"train-{benchmark}",
        "normalization_definition_hash": "norm-v1",
        "evaluation_population_hash": f"eval-{benchmark}",
        "prediction_mode": mode,
        "benchmark": benchmark,
        "representation": representation,
        "seed": seed,
    }
    records.provenance_path_for(path).write_text(json.dumps(payload))
    return path


def _source(paths) -> FigureSource:
    source = FigureSource("F04_headline_accuracy")
    for path in paths:
        frame = pd.read_csv(path)
        source.artifacts.append(ArtifactRef(
            path=str(path), kind="test_records", sha256=sha256_file(path),
            mtime_iso="", n_bytes=path.stat().st_size, schema_version=4,
            n_rows=len(frame), n_simulations=frame.sim_id.nunique(),
            seed=path.parent.name.removeprefix("seed"),
            benchmarks=(str(frame.benchmark.iloc[0]),),
        ))
    return source


@pytest.fixture
def publication_source(tmp_path):
    paths = [
        _write_seed(tmp_path, benchmark, seed)
        for benchmark in tables.BENCH_ORDER
        for seed in ("1", "2", "3")
    ]
    return _source(paths)


def test_population_std_identity_is_exact_but_sample_std_is_not():
    values = np.array([[[[299.0]], [[301.0]], [[305.0]]]], dtype=np.float64)
    mu, sigma = compute_global_stats(values, np.array([0]))
    direct = float(np.sqrt(np.mean((values - 300.0) ** 2)))
    assert np.sqrt(sigma ** 2 + (mu - 300.0) ** 2) == pytest.approx(direct)
    sample_sigma = float(values.std(ddof=1))
    assert np.sqrt(sample_sigma ** 2 + (mu - 300.0) ** 2) != pytest.approx(direct)


def test_training_population_hash_tracks_split_and_normalization_definition(tmp_path):
    trajectories = np.arange(4 * 3 * 2 * 2, dtype=np.float32).reshape(4, 3, 2, 2)
    grids = np.array([0.0, 1.0])
    times = np.array([0.0, 0.5, 1.0])
    a = build_normalization_provenance(
        trajectories, np.array([0, 1]), grids, grids, times
    )
    b = build_normalization_provenance(
        trajectories, np.array([0, 1]), grids, grids, times
    )
    c = build_normalization_provenance(
        trajectories, np.array([0, 2]), grids, grids, times
    )
    grid_path = tmp_path / "x_grid.npy"
    np.save(grid_path, grids)
    with_config = build_normalization_provenance(
        trajectories,
        np.array([0, 1]),
        grids,
        grids,
        times,
        data_paths={"x_grid_path": str(grid_path), "train_split": 0.7},
    )
    assert a["training_population_hash"] == b["training_population_hash"]
    assert a["training_population_hash"] != c["training_population_hash"]
    assert a["normalization_definition"]["numpy_ddof"] == 0
    assert "x_grid_path" in with_config["training_population"][
        "compact_dataset_file_hashes"
    ]


def test_schema_v4_and_provenance_link(publication_source):
    path = Path(publication_source.artifacts[0].path)
    assert records.detect_schema_version(pd.read_csv(path, nrows=0).columns) == 4
    payload = records.load_test_record_provenance(path)
    assert payload["provenance_id"] == pd.read_csv(path).provenance_id.iloc[0]


def test_primary_table_blocks_missing_run_provenance(publication_source, tmp_path):
    path = Path(publication_source.artifacts[0].path)
    records.provenance_path_for(path).unlink()
    with pytest.raises(ProvenanceError, match="provenance not found"):
        tables.write_primary_results(
            publication_source, tmp_path / "out", n_boot=5, strict_counts=False
        )


def test_primary_table_outputs_and_seed_averaging(publication_source, tmp_path):
    compact_path, detail_path, provenance_path = tables.write_primary_results(
        publication_source, tmp_path / "out", n_boot=80, rng_seed=7,
        strict_counts=False,
    )
    compact = pd.read_csv(compact_path)
    assert list(compact.columns) == ["Benchmark"] + [label for _, label in tables.PRIMARY_SPECS]
    assert list(compact.Benchmark) == [tables.BENCH_LABEL[b] for b in tables.BENCH_ORDER]

    forcing_paths = [Path(a.path) for a in publication_source.artifacts
                     if a.benchmarks == ("forcing",)]
    seed_p95 = [np.quantile(pd.read_csv(path).rmse_K, 0.95, method="linear")
                for path in forcing_paths]
    got = compact.loc[compact.Benchmark == "Forcing", "Field RMSE P95 (K)"].iloc[0]
    assert got == pytest.approx(np.mean(seed_p95))

    detail = pd.read_csv(detail_path)
    aggregate = detail[detail.aggregation_level == "across_seed"]
    assert {"test_cluster_ci_lower", "test_cluster_ci_upper", "across_seed_sd"} <= set(detail)
    assert np.all(aggregate.test_cluster_ci_lower <= aggregate.estimate)
    assert np.all(aggregate.estimate <= aggregate.test_cluster_ci_upper)
    assert (detail.aggregation_level == "lead_balanced_across_seed").any()

    provenance = json.loads(provenance_path.read_text())
    assert "conditional on the fixed trained models" in provenance["bootstrap"]["interpretation"]
    assert "not a percentile of pooled seed predictions" in provenance["seed_aggregation"]


def test_gnrmse_is_pooled_not_mean_rmse(publication_source, tmp_path):
    compact_path, _, _ = tables.write_primary_results(
        publication_source, tmp_path / "out", n_boot=20, strict_counts=False,
    )
    compact = pd.read_csv(compact_path).set_index("Benchmark")
    expected = []
    for artifact in publication_source.artifacts:
        if artifact.benchmarks != ("forcing",):
            continue
        frame = pd.read_csv(artifact.path)
        expected.append(100.0 * np.sqrt(frame.sse_K2.sum()
                                        / frame.num_error_cells.sum()) / 5.0)
    assert compact.loc["Forcing", "Field GNRMSE (%)"] == pytest.approx(np.mean(expected))


def test_peak_uses_only_initial_state_rows(publication_source, tmp_path):
    compact_path, _, _ = tables.write_primary_results(
        publication_source, tmp_path / "out", n_boot=20, strict_counts=False,
    )
    compact = pd.read_csv(compact_path).set_index("Benchmark")
    expected = []
    for artifact in publication_source.artifacts:
        if artifact.benchmarks != ("forcing",):
            continue
        frame = pd.read_csv(artifact.path)
        peak = tables._peak_frame(frame)
        expected.append(peak.peak_jump_error_K.mean())
    assert compact.loc["Forcing", "Peak-jump error mean (K)"] == pytest.approx(
        np.mean(expected)
    )


def test_mismatched_training_hash_blocks(publication_source, tmp_path):
    path = Path(publication_source.artifacts[0].path)
    payload = json.loads(records.provenance_path_for(path).read_text())
    payload["training_population_hash"] = "different"
    records.provenance_path_for(path).write_text(json.dumps(payload))
    with pytest.raises(ProvenanceError, match="hashes"):
        tables.write_primary_results(
            publication_source, tmp_path / "out", n_boot=5, strict_counts=False
        )


def test_rollout_records_block(publication_source, tmp_path):
    path = Path(publication_source.artifacts[0].path)
    payload = json.loads(records.provenance_path_for(path).read_text())
    payload["prediction_mode"] = "autoregressive_4_substeps"
    records.provenance_path_for(path).write_text(json.dumps(payload))
    with pytest.raises(ProvenanceError, match="direct_pair"):
        tables.write_primary_results(
            publication_source, tmp_path / "out", n_boot=5, strict_counts=False
        )


def test_strict_table_still_blocks_bins_representation(publication_source, tmp_path):
    path = next(
        Path(artifact.path)
        for artifact in publication_source.artifacts
        if artifact.benchmarks == ("source_itr_sin",)
    )
    payload = json.loads(records.provenance_path_for(path).read_text())
    payload["representation"] = "bins"
    records.provenance_path_for(path).write_text(json.dumps(payload))
    with pytest.raises(ProvenanceError, match="temporal_encoder"):
        tables.write_primary_results(
            publication_source, tmp_path / "out", n_boot=5, strict_counts=False
        )


def test_lead_curve_uses_exact_leads_and_cluster_counts(publication_source):
    paths = [Path(a.path) for a in publication_source.artifacts
             if a.benchmarks == ("forcing",)]
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["seed"] = path.parent.name.removeprefix("seed")
        frames.append(frame)
    curve = tables.lead_time_curve(pd.concat(frames), "field", n_boot=50)
    assert list(curve.t_bar) == [1.0, 2.0]
    assert list(curve.n_pairs_per_seed) == [8, 4]
    assert np.all(curve.test_cluster_ci_lower <= curve["mean"])
    assert np.all(curve["mean"] <= curve.test_cluster_ci_upper)


def test_lead_grouping_uses_integer_snapshot_steps_not_float_equality(
    publication_source,
):
    paths = [
        Path(artifact.path)
        for artifact in publication_source.artifacts
        if artifact.benchmarks == ("forcing",)
    ]
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["seed"] = path.parent.name.removeprefix("seed")
        frame["t_bar"] += frame["sim_id"] * np.finfo(np.float64).eps
        frames.append(frame)
    curve = tables.lead_time_curve(pd.concat(frames), "field", n_boot=20)
    assert len(curve) == 2
    assert list(curve.n_pairs_per_seed) == [8, 4]


def test_declared_invalid_run_status_blocks_before_path_resolution(tmp_path):
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        "sources:\n"
        "  forcing_records:\n"
        "    - run: runs/missing/config0/seed32\n"
        "      seed: 32\n"
        "      status: infrastructure_invalid\n"
    )
    manifest = Manifest.load(manifest_path, root=tmp_path)
    with pytest.raises(ProvenanceError, match="same declared seed/configuration"):
        manifest.resolve("F04_headline_accuracy", strict=True)


def test_cli_registers_primary_table():
    args = build_parser().parse_args(["--table", "T01_primary_results"])
    assert args.table == "T01_primary_results"


def test_descriptive_table_accepts_one_checkpoint_and_labels_mixed_representation(
    tmp_path,
):
    paths = [
        _write_seed(
            tmp_path,
            benchmark,
            "7",
            representation="bins" if benchmark == "source_itr_sin" else "temporal_encoder",
        )
        for benchmark in tables.BENCH_ORDER
    ]
    compact_path, detail_path, provenance_path = tables.write_descriptive_results(
        _source(paths), tmp_path / "out", n_boot=40, rng_seed=9,
        strict_counts=False,
    )
    compact = pd.read_csv(compact_path)
    assert list(compact.Benchmark) == [
        tables.BENCH_LABEL[benchmark] for benchmark in tables.BENCH_ORDER
    ]
    assert compact.set_index("Benchmark").loc["Source + ITR", "Representation"] == "bins"
    assert "Field training-centered pooled rel-L2 (%)" in compact

    detail = pd.read_csv(detail_path)
    assert set(detail.aggregation_level) == {
        "single_checkpoint", "lead_balanced_single_checkpoint"
    }
    assert np.all(detail.test_cluster_ci_lower <= detail.estimate)
    assert np.all(detail.estimate <= detail.test_cluster_ci_upper)

    provenance = json.loads(provenance_path.read_text())
    assert provenance["publication_readiness"] == "descriptive_only"
    assert provenance["mixed_representation"] is True
    assert provenance["strict_table_unchanged"] is True


def test_cli_registers_descriptive_table():
    args = build_parser().parse_args(
        ["--table", "T01_primary_results_descriptive"]
    )
    assert args.table == "T01_primary_results_descriptive"
