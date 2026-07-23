import json
import math

import numpy as np
import pytest
import torch
import torch.nn as nn

import src.operators.train_pino as train_pino
from src.operators.cvit import (
    ForcingTransitionCViT,
    TransitionEncoding,
    canonicalize_query_coords,
    sample_source_at_queries,
)
from src.operators.train_pino import (
    TransitionPairSchedule,
    aggregate_transition_metrics,
    build_cvit,
    build_forcing_transition_image,
    run_one_seed_forcing_transition_physics,
    run_one_seed_forcing_transition,
    sample_forcing_params,
    transition_epoch_order,
    transition_manifest_hash,
    transition_pair_metrics,
    write_transition_manifest,
)


SOURCE_GRID = (8, 8)
FORCING_GRID = (8, 8)


def _model(**overrides):
    values = {
        "emb_dim": 16,
        "dec_emb_dim": 16,
        "source_patch_size": 4,
        "source_grid_size": SOURCE_GRID,
        "forcing_patch_size": 4,
        "forcing_grid_size": FORCING_GRID,
        "depth_enc": 1,
        "depth_dec": 2,
        "num_heads": 4,
        "t_final": 0.3,
    }
    values.update(overrides)
    return ForcingTransitionCViT(**values)


def _inputs(batch=2, queries=7):
    generator = torch.Generator().manual_seed(12)
    forcing = torch.randn(
        batch, 3, *FORCING_GRID, generator=generator,
    )
    source = torch.randn(
        batch, 1, *SOURCE_GRID, generator=generator,
    )
    coords = torch.rand(batch, queries, 2, generator=generator)
    source_time = torch.tensor([[0.0], [0.1]])[:batch]
    lead_time = torch.tensor([[0.1], [0.2]])[:batch]
    return forcing, source, coords, source_time, lead_time


def test_structured_encoding_and_cached_source_equivalence():
    model = _model().eval()
    forcing, source, coords, source_time, lead_time = _inputs()
    source_tokens = model.encode_source(source)
    forcing_tokens = model.encode_forcing(forcing)
    encoding = model.fuse(source_tokens, forcing_tokens)
    direct = model.encode(forcing, source)
    assert isinstance(encoding, TransitionEncoding)
    assert encoding.memory_tokens.shape == (2, 8, 16)
    assert encoding.forcing_context.shape == (2, 16)
    torch.testing.assert_close(
        encoding.memory_tokens, direct.memory_tokens,
    )
    torch.testing.assert_close(
        encoding.forcing_context, direct.forcing_context,
    )
    cached = model.decode(
        encoding, source, coords, source_time, lead_time,
    )
    single_shot = model(
        forcing, source, coords, source_time, lead_time,
    )
    torch.testing.assert_close(cached, single_shot)


def test_forcing_context_is_separate_and_forcing_specific():
    model = _model().eval()
    forcing, source, _, _, _ = _inputs()
    source_tokens = model.encode_source(source)
    first = model.fuse(source_tokens, model.encode_forcing(forcing))
    changed = forcing.clone()
    changed[0, 0] += 2.0
    second = model.fuse(source_tokens, model.encode_forcing(changed))
    assert not torch.allclose(
        first.forcing_context[0], second.forcing_context[0],
    )
    torch.testing.assert_close(
        first.forcing_context[1], second.forcing_context[1],
    )


def test_film_initialization_and_first_backward_gradients():
    torch.manual_seed(3)
    model = _model(film_init_std=1.0e-3).train()
    head = model.decoder.time_film.net[-1]
    assert isinstance(head, nn.Linear)
    assert abs(float(head.weight.mean())) < 5.0e-4
    assert 2.0e-4 < float(head.weight.std()) < 2.0e-3
    assert torch.count_nonzero(head.bias) == 0
    forcing, source, coords, source_time, lead_time = _inputs()
    model(
        forcing, source, coords, source_time, lead_time,
    ).square().mean().backward()
    assert head.weight.grad is not None
    assert torch.isfinite(head.weight.grad).all()
    assert head.weight.grad.abs().sum() > 0
    first = model.decoder.time_film.net[0]
    assert isinstance(first, nn.Linear)
    assert first.weight.grad is not None
    assert torch.isfinite(first.weight.grad).all()
    assert first.weight.grad.abs().sum() > 0
    query_projection = model.decoder.query_time_projection
    assert query_projection.weight.grad is not None
    assert query_projection.weight.grad.abs().sum() > 0


def test_initial_film_is_small_but_not_identically_disabled():
    torch.manual_seed(21)
    conditioned = _model(
        query_time_conditioning=True,
        film_time_conditioning=True,
    ).eval()
    unconditioned = _model(
        query_time_conditioning=True,
        film_time_conditioning=False,
    ).eval()
    unconditioned.load_state_dict(
        conditioned.state_dict(), strict=False,
    )
    forcing, source, coords, source_time, lead_time = _inputs()
    with torch.no_grad():
        output_conditioned = conditioned(
            forcing, source, coords, source_time, lead_time,
        )
        output_unconditioned = unconditioned(
            forcing, source, coords, source_time, lead_time,
        )
        encoding = conditioned.encode(forcing, source)
        source_norm = source_time / 0.3
        lead_norm = lead_time / 0.3
        time_raw = torch.stack(
            [lead_norm[:, 0], source_norm[:, 0]], dim=-1,
        ).unsqueeze(1)
        time_context = conditioned.decoder.time_conditioner(
            torch.cat(
                [
                    time_raw,
                    conditioned.decoder.fourier_lead(
                        lead_norm.unsqueeze(1)
                    ),
                    conditioned.decoder.fourier_source(
                        source_norm.unsqueeze(1)
                    ),
                ],
                dim=-1,
            )
        )
        film = conditioned.decoder.time_film(
            torch.cat([time_context[:, 0], encoding.forcing_context], dim=-1)
        )
    difference = (output_conditioned - output_unconditioned).abs().mean()
    assert 0.0 < float(difference) < 0.05
    assert float(film.abs().max()) < 0.05


def test_conditioning_switch_contract():
    with pytest.raises(ValueError, match="At least one"):
        _model(
            query_time_conditioning=False,
            film_time_conditioning=False,
        )
    assert not hasattr(
        _model(
            query_time_conditioning=True,
            film_time_conditioning=False,
        ).decoder,
        "time_film",
    )
    assert not hasattr(
        _model(
            query_time_conditioning=False,
            film_time_conditioning=True,
        ).decoder,
        "query_time_projection",
    )


def test_zero_lead_is_exact_identity_without_nans():
    model = _model().eval()
    forcing, source, coords, source_time, _ = _inputs()
    output = model(
        forcing,
        source,
        coords,
        source_time,
        torch.zeros_like(source_time),
    )
    sampled, _ = sample_source_at_queries(source, coords)
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, sampled, rtol=0.0, atol=0.0)


def test_coordinate_canonicalization_interpolation_and_boundary_gate():
    grid_x, grid_y = torch.meshgrid(
        torch.linspace(0.0, 1.0, 8),
        torch.linspace(0.0, 1.0, 8),
        indexing="ij",
    )
    source = (2.0 * grid_x + 3.0 * grid_y).view(1, 1, 8, 8)
    coords = torch.tensor(
        [[[0.25, 0.75], [1.0 + 5.0e-7, 0.4]]],
    )
    sampled, canonical = sample_source_at_queries(
        source, coords, tolerance=1.0e-6,
    )
    torch.testing.assert_close(
        sampled[0, 0, 0], torch.tensor(2.75), atol=1.0e-6, rtol=0.0,
    )
    assert canonical[0, 1, 0].item() == 1.0
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        canonicalize_query_coords(
            torch.tensor([[[1.01, 0.5]]]), tolerance=1.0e-6,
        )

    model = _model().eval()
    forcing = torch.randn(1, 3, *FORCING_GRID)
    output = model(
        forcing,
        source,
        coords[:, 1:],
        torch.tensor([[0.0]]),
        torch.tensor([[0.1]]),
    )
    expected, _ = sample_source_at_queries(
        source, coords[:, 1:], tolerance=1.0e-6,
    )
    torch.testing.assert_close(output, expected, rtol=0.0, atol=0.0)


class _Schedule:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float64)

    def evaluate_grid(self, y, t):
        indices = np.rint(np.asarray(t) * 100).astype(int)
        return np.broadcast_to(
            self.values[indices][None, :],
            (len(y), len(indices)),
        )


def test_zero_lead_forcing_image_and_suffix_invariance(monkeypatch):
    schedules = {
        "first": np.arange(31, dtype=np.float64),
        "second": np.arange(31, dtype=np.float64),
    }
    schedules["second"][11:] += 100.0

    monkeypatch.setattr(
        train_pino,
        "reconstruct_qL",
        lambda temporal_family, *args, **kwargs: _Schedule(
            schedules[temporal_family]
        ),
    )
    params = {
        "temporal_family": "first",
        "temporal_params": {},
        "spatial_family": "uniform",
        "spatial_params": {},
    }
    zero = build_forcing_transition_image(
        [params],
        np.linspace(0.0, 1.0, 8),
        [0.1],
        [0.0],
        8,
        10.0,
        torch.device("cpu"),
        0.01,
        0.3,
    )
    assert torch.isfinite(zero).all()
    assert torch.all(zero[:, 0] == zero[:, 0, :, 0:1])
    torch.testing.assert_close(
        zero[:, 2],
        torch.full_like(zero[:, 2], 1.0 / 3.0),
    )
    assert torch.unique(zero[:, 1]).numel() == 8

    other = dict(params, temporal_family="second")
    first_segment = build_forcing_transition_image(
        [params],
        np.linspace(0.0, 1.0, 8),
        [0.0],
        [0.1],
        8,
        10.0,
        torch.device("cpu"),
        0.01,
        0.3,
    )
    second_segment = build_forcing_transition_image(
        [other],
        np.linspace(0.0, 1.0, 8),
        [0.0],
        [0.1],
        8,
        10.0,
        torch.device("cpu"),
        0.01,
        0.3,
    )
    torch.testing.assert_close(first_segment, second_segment)
    model = _model().eval()
    source = torch.randn(1, 1, *SOURCE_GRID)
    coords = torch.rand(1, 5, 2)
    first_prediction = model(
        first_segment,
        source,
        coords,
        torch.tensor([[0.0]]),
        torch.tensor([[0.1]]),
    )
    second_prediction = model(
        second_segment,
        source,
        coords,
        torch.tensor([[0.0]]),
        torch.tensor([[0.1]]),
    )
    torch.testing.assert_close(
        first_prediction, second_prediction, rtol=0.0, atol=0.0,
    )


def test_source_and_in_window_forcing_paths_affect_output():
    model = _model().eval()
    forcing, source, coords, source_time, lead_time = _inputs()
    baseline = model(
        forcing, source, coords, source_time, lead_time,
    )
    source_changed = source.clone()
    source_changed[0] += 1.0
    forcing_changed = forcing.clone()
    forcing_changed[0, 0] += 1.0
    by_source = model(
        forcing, source_changed, coords, source_time, lead_time,
    )
    by_forcing = model(
        forcing_changed, source, coords, source_time, lead_time,
    )
    assert not torch.allclose(baseline[0], by_source[0])
    assert not torch.allclose(baseline[0], by_forcing[0])
    torch.testing.assert_close(baseline[1], by_source[1])
    torch.testing.assert_close(baseline[1], by_forcing[1])


def test_pair_schedule_is_deterministic_balanced_and_nonduplicating():
    t_grid = np.linspace(0.0, 0.3, 31)
    schedule = TransitionPairSchedule(
        t_grid,
        20,
        [0.0, 0.05, 0.10, 0.20, 0.30],
        [0.0, 0.075, 0.15, 0.225, 0.30],
        [0.0, 0.075, 0.15, 0.225, 0.30],
    )
    first = schedule.sample_pair(7, 2, 11)
    assert first == schedule.sample_pair(7, 2, 11)
    assert any(
        schedule.sample_pair(7, 2, sim_id)
        != schedule.sample_pair(7, 3, sim_id)
        for sim_id in range(20)
    )
    ids = np.arange(200)
    order = transition_epoch_order(ids, base_seed=5, epoch=2)
    assert len(order) == len(np.unique(order)) == len(ids)
    np.testing.assert_array_equal(
        order, transition_epoch_order(ids, base_seed=5, epoch=2),
    )
    lead_counts = np.zeros(4, dtype=int)
    conditional: dict[int, list[int]] = {}
    for sim_id in range(4000):
        _, _, source_bin, lead_bin = schedule.sample_pair(9, 1, sim_id)
        lead_counts[lead_bin] += 1
        conditional.setdefault(lead_bin, []).append(source_bin)
    assert lead_counts.max() / lead_counts.min() < 1.12
    for lead_bin, values in conditional.items():
        counts = np.unique(values, return_counts=True)[1]
        assert counts.max() / counts.min() < 1.2
        assert lead_bin in schedule.eligible_lead_bins

    records = schedule.validation_records(
        np.array([4]), pairs_per_cell=100, base_seed=1,
    )
    keys = {
        (row["source_index"], row["target_index"]) for row in records
    }
    assert len(keys) == len(records)
    assert (3, 3) not in schedule.cells


def test_metric_identity_signal_gate_macro_and_manifest(tmp_path):
    source = torch.zeros(2, 4)
    target = torch.tensor(
        [[1.0, 1.0, 1.0, 1.0], [0.001, 0.001, 0.001, 0.001]],
    )
    prediction = torch.tensor(
        [[0.5, 0.5, 0.5, 0.5], [0.0, 0.0, 0.0, 0.0]],
    )
    first = transition_pair_metrics(
        prediction[0],
        target[0],
        source[0],
        sigma_global=10.0,
        signal_floor_fraction=0.01,
    )
    assert first["target_rmse_K"] == pytest.approx(5.0)
    assert first["increment_rel_l2"] == pytest.approx(0.5)
    assert first["copy_skill"] == pytest.approx(0.75)
    second = transition_pair_metrics(
        prediction[1],
        target[1],
        source[1],
        sigma_global=10.0,
        signal_floor_fraction=0.01,
    )
    assert second["increment_gated"] is False
    assert second["increment_rel_l2"] is None

    rows = []
    for source_bin, lead_bin, value in ((0, 0, 1.0), (0, 0, 3.0), (1, 1, 9.0)):
        rows.append({
            "source_bin": source_bin,
            "lead_bin": lead_bin,
            "target_bin": lead_bin,
            "source_lead_cell": f"s{source_bin}_l{lead_bin}",
            "ic_family": "uniform_2d",
            "forcing_amplitude_tercile": "low",
            "target_rmse_K": value * 10.0,
            "target_gnrmse": value,
            "increment_gated": True,
            "increment_rel_l2": 0.5,
            "copy_skill": 0.75,
        })
    aggregate = aggregate_transition_metrics(rows)
    assert aggregate["target_gnrmse"] == pytest.approx(13.0 / 3.0)
    assert aggregate["macro_cell_gnrmse"] == pytest.approx(5.5)
    records = [{"sim_id": 2, "source_index": 0, "target_index": 1}]
    digest = transition_manifest_hash(records)
    path = tmp_path / "manifest.json"
    assert write_transition_manifest(path, records) == digest
    payload = json.loads(path.read_text())
    assert payload["sha256"] == digest
    assert payload["records"] == records


def test_build_cvit_transition_variant():
    config = {
        "model": {
            "cvit": {
                "emb_dim": 16,
                "patch_size": 4,
                "depth_enc": 1,
                "depth_dec": 1,
                "num_heads": 4,
            },
            "forcing_transition_cvit": {
                "forcing_patch_size": 4,
                "source_patch_size": 4,
            },
        },
        "training": {
            "pino": {
                "forcing": {"ny_img": 8, "nt_img": 8, "a_ref": 300.0}
            }
        },
    }
    model = build_cvit(
        config,
        mu=300.0,
        sigma=10.0,
        grid_size=(8, 8),
        t_final=0.3,
        variant="forcing_transition",
    )
    assert isinstance(model, ForcingTransitionCViT)


def test_dispatch_selects_transition_runner(tmp_path, monkeypatch):
    config = {
        "benchmark": {"name": "diffusion_forcing_single"},
        "model": {"cvit": {}},
        "training": {"pino": {"variant": "forcing_transition"}},
    }
    called = []

    def runner(received_config, seed, run_dir):
        called.append((received_config, seed, run_dir))
        return {"variant": "forcing_transition"}

    monkeypatch.setattr(
        train_pino, "run_one_seed_forcing_transition", runner,
    )
    result = train_pino.run_config_seeds_pino(
        config, tmp_path, [7],
    )
    assert result["seeds"]["7"]["variant"] == "forcing_transition"
    assert called[0][1] == 7


IC_FAMILIES = (
    "uniform_2d",
    "random_sinusoid_2d",
    "grf_2d",
    "hot_spot_2d",
)


def _write_transition_dataset(path, num_sims=16):
    rng = np.random.default_rng(13)
    x_grid = np.linspace(0.0, 1.0, 8, dtype=np.float32)
    y_grid = np.linspace(0.0, 1.0, 8, dtype=np.float32)
    t_grid = np.linspace(0.0, 0.3, 7, dtype=np.float32)
    gx, gy = np.meshgrid(x_grid, y_grid, indexing="ij")
    trajectories = np.empty((num_sims, 7, 8, 8), dtype=np.float32)
    for sim_id in range(num_sims):
        source = 300.0 + (1.0 - gx) * (
            np.sin(2.0 * np.pi * gy + sim_id * 0.1)
        )
        for time_index, time_value in enumerate(t_grid):
            trajectories[sim_id, time_index] = (
                source
                + (1.0 - gx) * time_value * (sim_id + 1) * 0.2
            )
    records = sample_forcing_params(
        rng,
        num_sims,
        dt=0.05,
        t_final=0.3,
        temporal_family="sin",
        spatial_family="uniform",
    )
    for sim_id, record in enumerate(records):
        record["ic_family"] = IC_FAMILIES[sim_id % 4]
        record["ic_params"] = {}
        record["T0"] = trajectories[sim_id, 0].copy()
    np.save(path / "trajectories.npy", trajectories)
    np.save(path / "x_grid.npy", x_grid)
    np.save(path / "y_grid.npy", y_grid)
    np.save(path / "t_grid.npy", t_grid)
    np.save(path / "sim_params.npy", np.asarray(records, dtype=object))
    np.save(path / "dt.npy", np.asarray(0.05))
    np.save(path / "ramp_seconds.npy", np.asarray(0.01))
    np.save(
        path / "meta.npy",
        np.asarray(
            {
                "problem_version": "forcing_single_varying_ic_v1",
                "ic_mode": "varying",
                "ic_families": list(IC_FAMILIES),
            },
            dtype=object,
        ),
        allow_pickle=True,
    )


def _runner_config(path, epochs=1):
    return {
        "benchmark": {"name": "diffusion_forcing_single"},
        "data": {
            "trajectories.npy": str(path / "trajectories.npy"),
            "x_grid_path": str(path / "x_grid.npy"),
            "y_grid_path": str(path / "y_grid.npy"),
            "t_grid_path": str(path / "t_grid.npy"),
        },
        "model": {
            "cvit": {
                "out_dim": 1,
                "emb_dim": 16,
                "dec_emb_dim": 16,
                "patch_size": 4,
                "depth_enc": 0,
                "depth_dec": 1,
                "num_heads": 4,
                "mlp_ratio": 2.0,
                "fourier_freq": 1.0,
                "activation": "gelu",
            },
            "forcing_transition_cvit": {
                "forcing_patch_size": 4,
                "source_patch_size": 4,
                "film_init_std": 1.0e-3,
            },
        },
        "training": {
            "device": "cpu",
            "epochs": epochs,
            "n_snapshots": 4,
            "validate_every": 1,
            "learning_rate": 5.0e-4,
            "weight_decay": 1.0e-5,
            "noise_std": 0.0,
            "grad_clip": 1.0,
            "optimizer": "AdamW",
            "scheduler": {
                "type": "StepLR",
                "step_size": 10,
                "gamma": 0.5,
            },
            "pino": {
                "variant": "forcing_transition",
                "forcing": {
                    "ny_img": 8,
                    "nt_img": 8,
                    "a_ref": 300.0,
                    "ramp_seconds": 0.01,
                },
                "transition": {
                    "sim_batch": 32,
                    "n_queries": 4,
                    "query_chunk": 0,
                    "lead_edges": [0.0, 0.05, 0.10, 0.20, 0.30],
                    "source_edges": [0.0, 0.075, 0.15, 0.225, 0.30],
                    "target_edges": [0.0, 0.075, 0.15, 0.225, 0.30],
                    "fast_val_max_sims": 2,
                    "fast_val_pairs_per_cell": 1,
                    "final_val_pairs_per_cell": 1,
                    "validation_batch_size": 2,
                    "validation_query_chunk": 64,
                    "save_latest_every_updates": 1,
                },
            },
        },
    }


def test_tiny_end_to_end_and_best_checkpoint_provenance(tmp_path):
    _write_transition_dataset(tmp_path)
    run_dir = tmp_path / "run"
    summary = run_one_seed_forcing_transition(
        _runner_config(tmp_path), seed=5, run_dir=run_dir,
    )
    assert summary["test_set_evaluated"] is False
    assert summary["best_checkpoint_epoch"] == 0
    assert summary["final_validation"]["num_pairs"] > 0
    assert math.isfinite(
        summary["final_validation"]["macro_cell_gnrmse"]
    )
    for filename in (
        "cvit_best.pt",
        "cvit_latest.pt",
        "cvit_final.pt",
        "fast_validation_manifest.json",
        "final_validation_manifest.json",
        "final_val_pairs.csv",
        "final_metrics.json",
        "RUN_COMPLETE",
    ):
        assert (run_dir / filename).exists()
    best = torch.load(
        run_dir / "cvit_best.pt", map_location="cpu", weights_only=False,
    )
    final = torch.load(
        run_dir / "cvit_final.pt", map_location="cpu", weights_only=False,
    )
    assert best["epoch"] == summary["best_checkpoint_epoch"]
    for name, value in best["model_state"].items():
        torch.testing.assert_close(value, final["model_state"][name])
    assert final["checkpoint_compatibility"]["forcing_channels"] == 3
    assert final["checkpoint_compatibility"]["align_corners"] is True


def _write_gate_summary(path, payload):
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload = dict(payload)
    payload["gate_sha256"] = __import__("hashlib").sha256(
        canonical.encode()
    ).hexdigest()
    path.write_text(json.dumps(payload))
    return payload


def _physics_runner_config(path, epochs=1):
    direct = _write_gate_summary(
        path / "direct_gate.json",
        {
            "schema_version": 1,
            "stage": "dense_direct_state_gate",
            "passed": True,
            "qualified_objectives": ["variational"],
            "cases": [{
                "floor_metrics": {
                    "overall_field": 1.0e-12,
                    "longest_lead": 1.0e-12,
                },
            }],
        },
    )
    network = _write_gate_summary(
        path / "network_gate.json",
        {
            "schema_version": 1,
            "stage": "single_instance_network_gate",
            "passed": True,
            "direct_state_gate_sha256": direct["gate_sha256"],
            "selected_objective": "variational",
            "matched_budget": 1,
            "consecutive_intervals": {
                "enabled": False,
                "probability": 0.25,
                "min_length": 2,
                "max_length": 4,
            },
        },
    )
    config = _runner_config(path, epochs=epochs)
    config["training"]["gradnorm"] = {"enabled": True}
    transition = config["training"]["pino"]["transition"]
    transition["objective"] = "physics_only"
    transition["sim_batch"] = 4
    transition["physics"] = {
        "gate_summary": str(path / "direct_gate.json"),
        "network_gate_summary": str(path / "network_gate.json"),
        "objective": "variational",
        "residual_method": "finite_volume",
        "dt": 0.005,
        "updates_per_epoch": 1,
        "source_state_mode": "online_local_ivp",
        "time_bundle": "lead_bins",
        "include_anchor_interval": True,
        "intervals_per_cell": 1,
        "consecutive_intervals": network["consecutive_intervals"],
        "production_screen": {
            "enabled": False,
            "updates": 2000,
            "anti_collapse_every": 100,
            "validate_every": 250,
            "collapse_update": 500,
            "minimum_skill_improvement": 0.10,
        },
    }
    return config, direct, network


def test_tiny_physics_only_transition_end_to_end(tmp_path):
    _write_transition_dataset(tmp_path)
    config, direct, network = _physics_runner_config(tmp_path)
    summary = run_one_seed_forcing_transition_physics(
        config, seed=19, run_dir=tmp_path / "physics_run",
    )
    assert summary["objective"] == "physics_only_transition"
    assert summary["completed_updates"] == 1
    assert summary["selected_physics_objective"] == "variational"
    assert summary["optimization_uses_solution_fields"] is False
    assert summary["direct_state_gate_sha256"] == direct["gate_sha256"]
    assert summary["network_gate_sha256"] == network["gate_sha256"]
    assert summary["gradnorm_mode"] == "inert_single_term"


def test_physics_transition_exact_resume(tmp_path, monkeypatch):
    _write_transition_dataset(tmp_path)
    config, _, _ = _physics_runner_config(tmp_path, epochs=2)
    full_dir = tmp_path / "physics_full"
    split_dir = tmp_path / "physics_split"
    run_one_seed_forcing_transition_physics(
        config, seed=23, run_dir=full_dir,
    )

    original_save = train_pino._atomic_torch_save
    interrupted = False

    def interrupt_after_first_update(payload, path):
        nonlocal interrupted
        original_save(payload, path)
        if (
            not interrupted
            and path.name == "cvit_latest.pt"
            and payload.get("objective") == "physics_only_transition"
            and payload["completed_updates"] == 1
        ):
            interrupted = True
            raise RuntimeError("simulated physics interruption")

    monkeypatch.setattr(
        train_pino, "_atomic_torch_save", interrupt_after_first_update,
    )
    with pytest.raises(RuntimeError, match="simulated physics interruption"):
        run_one_seed_forcing_transition_physics(
            config, seed=23, run_dir=split_dir,
        )
    monkeypatch.setattr(
        train_pino, "_atomic_torch_save", original_save,
    )
    run_one_seed_forcing_transition_physics(
        config, seed=23, run_dir=split_dir,
    )
    full = torch.load(
        full_dir / "cvit_latest.pt", map_location="cpu", weights_only=False,
    )
    split = torch.load(
        split_dir / "cvit_latest.pt", map_location="cpu", weights_only=False,
    )
    assert full["completed_updates"] == split["completed_updates"] == 2
    for name, value in full["model_state"].items():
        torch.testing.assert_close(
            value, split["model_state"][name], rtol=0.0, atol=0.0,
        )


def test_mid_epoch_resume_parity(tmp_path, monkeypatch):
    _write_transition_dataset(tmp_path)
    full_dir = tmp_path / "full"
    split_dir = tmp_path / "split"
    config = _runner_config(tmp_path, epochs=2)
    config["training"]["pino"]["transition"]["sim_batch"] = 4
    run_one_seed_forcing_transition(
        config, seed=17, run_dir=full_dir,
    )

    original_save = train_pino._atomic_torch_save
    interrupted = False

    def interrupt_after_first_batch(payload, path):
        nonlocal interrupted
        original_save(payload, path)
        if (
            not interrupted
            and path.name == "cvit_latest.pt"
            and payload["next_epoch"] == 0
            and payload["next_batch"] == 1
        ):
            interrupted = True
            raise RuntimeError("simulated interruption")

    monkeypatch.setattr(
        train_pino, "_atomic_torch_save", interrupt_after_first_batch,
    )
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_one_seed_forcing_transition(
            config, seed=17, run_dir=split_dir,
        )
    monkeypatch.setattr(train_pino, "_atomic_torch_save", original_save)
    run_one_seed_forcing_transition(
        config, seed=17, run_dir=split_dir,
    )
    full = torch.load(
        full_dir / "cvit_latest.pt", map_location="cpu", weights_only=False,
    )
    split = torch.load(
        split_dir / "cvit_latest.pt", map_location="cpu", weights_only=False,
    )
    assert full["completed_updates"] == split["completed_updates"]
    for name, value in full["model_state"].items():
        torch.testing.assert_close(
            value, split["model_state"][name], rtol=0.0, atol=0.0,
        )
