"""Tests for scripts/run_resolution_study.py and the F24 figure that reads it."""

import csv
import json
import math
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from scripts import run_resolution_study as study
from tests.conftest import (
    FORCING_COND_STATIC_DIM,
    FORCING_IN_CHANNELS,
    FORCING_TEMPORAL_SAMPLES,
    FORCING_TEMPORAL_TOKEN_DIM,
)

RESOLUTIONS = (16, 24)


@pytest.fixture
def forcing_run(tmp_path, small_fno2d):
    config = {
        "benchmark": {"name": "forcing", "representation": "temporal_encoder"},
        "data": {},
        "training": {"batch_size": 512, "device": "cpu", "n_snapshots_test": 40},
        "model": {
            "parameters": {
                "modes1": 2,
                "modes2": 2,
                "width": 8,
                "in_channels": FORCING_IN_CHANNELS,
                "out_channels": 1,
                "n_layers": 2,
                "cond_static_dim": FORCING_COND_STATIC_DIM,
                "cond_hidden": 256,
                "temporal_token_dim": FORCING_TEMPORAL_TOKEN_DIM,
                "temporal_samples": FORCING_TEMPORAL_SAMPLES,
                "temporal_hidden": 16,
                "forcing_embed_dim": 16,
                "padding_reference_resolution": RESOLUTIONS[0],
            }
        },
    }
    seed_dir = tmp_path / "run" / "seed42"
    seed_dir.mkdir(parents=True)
    torch.save(
        {
            "model_state": small_fno2d.state_dict(),
            "conf": config,
            "mu_global": 303.0,
            "sigma_global": 7.0,
            "best_val": 1.0,
            "epoch": 5,
            "seed": 42,
        },
        seed_dir / "fno2d_best.pt",
    )
    return tmp_path / "run"


@pytest.fixture
def finished_study(tmp_path, forcing_run):
    out_dir = tmp_path / "study"
    assert study.main([
        str(forcing_run), "--seed", "42", "--out-dir", str(out_dir),
        "--resolutions", ",".join(str(n) for n in reversed(RESOLUTIONS)),
        "--num-sims", "2", "--n-snapshots-test", "3", "--device", "cpu",
    ]) == 0
    return out_dir


def test_outputs_only_field_and_node_jump_gnrmse_per_grid(finished_study):
    with open(finished_study / "resolution_study.csv", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert reader.fieldnames == ["resolution", "field_gnrmse_pct", "node_jump_gnrmse_pct"]
    assert [int(r["resolution"]) for r in rows] == list(RESOLUTIONS)
    for row in rows:
        assert math.isfinite(float(row["field_gnrmse_pct"]))
        assert math.isfinite(float(row["node_jump_gnrmse_pct"]))

    payload = json.loads((finished_study / "resolution_study.json").read_text())
    assert payload["per_resolution"] == [
        {k: (int(v) if k == "resolution" else float(v)) for k, v in row.items()}
        for row in rows
    ]
    assert set(payload["metric_definitions"]) == {"field_gnrmse_pct", "node_jump_gnrmse_pct"}
    assert payload["num_sims"] == 2
    assert payload["pairs_per_sim"] == 3
    assert payload["checkpoint_seed"] == "42"
    assert payload["checkpoint_epoch"] == 5
    assert payload["padding_reference_resolution"] == RESOLUTIONS[0]
    assert payload["dataset_rng_seed"] == 7
    assert not list(finished_study.rglob("seed_report*.json"))


def test_interfaces_study_scores_the_jump_at_each_sims_own_interface(
        tmp_path, monkeypatch):
    """The varying-interface ladder must flank every sim's own interface_x."""
    import src.operators.eval as eval_mod
    from problems.registry import get_problem
    from src.operators.fno2d import FNO2d

    dims = get_problem("interfaces").dims
    params = {
        "modes1": 2, "modes2": 2, "width": 8, "out_channels": 1, "n_layers": 2,
        "cond_hidden": 32, "temporal_hidden": 16, "forcing_embed_dim": 16,
        "padding_reference_resolution": RESOLUTIONS[0],
    }
    model = FNO2d(
        in_channels=dims.in_channels, cond_static_dim=dims.cond_static_dim,
        temporal_token_dim=dims.temporal_token_dim,
        use_forcing_time_aug=dims.use_forcing_time_aug, s_y_channel=dims.s_y_channel,
        **{k: v for k, v in params.items() if k != "padding_reference_resolution"},
    )
    config = {
        "benchmark": {"name": "interfaces", "representation": "temporal_encoder"},
        "data": {},
        "training": {"batch_size": 512, "device": "cpu", "n_snapshots_test": 40,
                     "loss": {"interface_x": 0.5, "interface_half_width": 0.05,
                              "per_sample_interface_x": True}},
        "model": {"parameters": params},
    }
    seed_dir = tmp_path / "run" / "seed32"
    seed_dir.mkdir(parents=True)
    torch.save({"model_state": model.state_dict(), "conf": config,
                "mu_global": 308.0, "sigma_global": 12.0, "best_val": 1.0,
                "epoch": 7, "seed": 32}, seed_dir / "fno2d_best.pt")

    flanked = []
    original = eval_mod.interface_flanking_nodes_per_sample

    def spy(x_grid, interface_x):
        flanked.append(interface_x.detach().cpu().numpy().copy())
        return original(x_grid, interface_x)

    monkeypatch.setattr(eval_mod, "interface_flanking_nodes_per_sample", spy)
    out_dir = tmp_path / "study"
    assert study.main([
        str(tmp_path / "run"), "--out-dir", str(out_dir),
        "--resolutions", ",".join(map(str, RESOLUTIONS)),
        "--num-sims", "2", "--n-snapshots-test", "3", "--device", "cpu",
    ]) == 0

    payload = json.loads((out_dir / "resolution_study.json").read_text())
    assert payload["benchmark"] == "interfaces"
    assert [r["resolution"] for r in payload["per_resolution"]] == list(RESOLUTIONS)
    for row in payload["per_resolution"]:
        assert set(row) == {"resolution", "field_gnrmse_pct", "node_jump_gnrmse_pct"}
        assert math.isfinite(row["field_gnrmse_pct"])
        assert math.isfinite(row["node_jump_gnrmse_pct"])
    sampled = np.load(out_dir / "data" / "r16" / "sim_params.npy", allow_pickle=True)
    expected = {round(float(p["interface_x"]), 5) for p in sampled}
    assert len(expected) == 2 and 0.5 not in expected
    assert {round(float(x), 5) for batch in flanked for x in batch} == expected


def test_ladder_with_different_latents_is_refused(tmp_path):
    root = tmp_path / "data"
    study.generate_dataset("forcing", root / "r16", 2, 16, 16, 7)
    study.generate_dataset("forcing", root / "r24", 2, 24, 24, 8)
    with pytest.raises(SystemExit, match="latents differ"):
        study.check_matched_ladder({16: root / "r16", 24: root / "r24"})


def test_latent_signature_ignores_only_grid_sampled_arrays():
    a = {"R_c": np.float64(0.1), "T0": np.zeros((4, 4)), "row": np.zeros(4),
         "p": {"A_list": np.array([1.0, 2.0])}}
    b = {"R_c": 0.1, "T0": np.ones((8, 8)), "row": np.ones(8),
         "p": {"A_list": np.array([1.0, 2.0])}}
    assert study._latent_signature(a, 4) == study._latent_signature(b, 8)
    b["p"]["A_list"] = np.array([1.0, 2.5])
    assert study._latent_signature(a, 4) != study._latent_signature(b, 8)


def _source(path):
    return SimpleNamespace(artifacts=[SimpleNamespace(
        kind="json_report", path=str(path), benchmarks=())])


def test_f24_plots_the_two_gnrmse_metrics_from_the_study(finished_study):
    from visual.pub import fig_supplement, records, style

    report = finished_study / "resolution_study.json"
    loaded = records.load_resolution_study(_source(report))
    payload = json.loads(report.read_text())
    assert loaded.resolutions == RESOLUTIONS
    assert loaded.field_gnrmse_pct == tuple(
        r["field_gnrmse_pct"] for r in payload["per_resolution"])
    assert loaded.node_jump_gnrmse_pct == tuple(
        r["node_jump_gnrmse_pct"] for r in payload["per_resolution"])

    with style.pub_style():
        fig, _, definition = fig_supplement.resolution_invariance(
            source=_source(report))
    try:
        axes = fig.axes
        assert len(axes) == 2
        for ax, metric in zip(axes, ("field_gnrmse_pct", "node_jump_gnrmse_pct")):
            (line,) = ax.get_lines()
            assert tuple(line.get_xdata()) == RESOLUTIONS
            assert tuple(line.get_ydata()) == getattr(loaded, metric)
    finally:
        plt.close(fig)
    assert definition["resolutions"] == list(RESOLUTIONS)
    assert set(definition) >= {"field_gnrmse_pct", "node_jump_gnrmse_pct"}


def test_loader_rejects_a_report_missing_a_metric(tmp_path):
    from visual.pub import records

    report = tmp_path / "resolution_study.json"
    report.write_text(json.dumps({
        "checkpoint_epoch": 1, "num_sims": 2, "pairs_per_sim": 3,
        "per_resolution": [{"resolution": 100, "field_gnrmse_pct": 1.0}],
    }))
    with pytest.raises(records.SchemaError, match="node_jump_gnrmse_pct"):
        records.load_resolution_study(_source(report))
