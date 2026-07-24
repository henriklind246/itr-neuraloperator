import hashlib
import json

import numpy as np
import pytest
import torch

import src.operators.train_pino as train_pino
from scripts.diagnose_fv_direct_state import (
    _dense_table_loss,
    _table_from_free,
    audit_cn_hessian,
    build_homogeneous_gate_case,
    calibrated_gate_thresholds,
    preregistered_max_lead,
    sequential_exact_trajectory,
)
from scripts.diagnose_fv_variational_cvit import (
    _contrast_metrics,
    repeat_transition_intervals,
    transition_multiforcing_probe_params,
    transition_multisource_multiforcing_specs,
)
from src.physics.one_step_objective import build_cn_tensors
from src.operators.cvit import ForcingTransitionCViT
from src.operators.train_pino import (
    forcing_transition_physics_loss,
    load_verified_transition_gate,
    sample_transition_physics_intervals,
    sample_transition_source_steps,
    transition_consecutive_interval_trigger,
    transition_curriculum_max_lead,
    transition_production_screen_decision,
    transition_screen_skill,
)


def _write_gate(path, stage="dense_direct_state_gate", passed=True):
    payload = {"schema_version": 1, "stage": stage, "passed": passed}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["gate_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    path.write_text(json.dumps(payload))
    return payload


def test_gate_hash_and_copy_floor_validity(tmp_path):
    path = tmp_path / "gate.json"
    expected = _write_gate(path)
    loaded, digest = load_verified_transition_gate(
        path, expected_stage="dense_direct_state_gate",
    )
    assert loaded == expected
    assert digest == expected["gate_sha256"]

    invalid = calibrated_gate_thresholds(1.0, 0.01, 0.001)
    assert invalid["valid"] is False
    valid = calibrated_gate_thresholds(1.0, 1.0e-6, 1.0e-5)
    assert valid["valid"] is True
    assert valid["oracle_qualified"] is True

    payload = json.loads(path.read_text())
    payload["passed"] = False
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_verified_transition_gate(
            path, expected_stage="dense_direct_state_gate",
        )


def test_homogeneous_hessian_audit_and_sequential_oracle():
    sim, truth, metadata = build_homogeneous_gate_case(
        family="grf_2d",
        source_time=0.15,
        grid_size=20,
        seed=11,
    )
    assert metadata["n_intervals"] == 30
    audit = audit_cn_hessian(sim, sigma=10.0, seed=11)
    assert audit["passed"]
    assert audit["relative_asymmetry"] <= 1.0e-12
    assert audit["lambda_min"] > 0.0
    assert audit["implicit_float32_action_relative_error"] <= 1.0e-5
    reconstructed = sequential_exact_trajectory(
        sim, truth[0], sigma=10.0,
    )
    np.testing.assert_allclose(reconstructed, truth, rtol=0.0, atol=1.0e-10)


def test_registered_curriculum_boundaries():
    expected = (0.05, 0.10, 0.20, 0.30)
    updates = (0, 10, 25, 50)
    for index, update in enumerate(updates):
        direct = preregistered_max_lead(update, 100, 0.3)
        production = transition_curriculum_max_lead(update, 100, 0.3)
        assert direct == production
        if index:
            assert direct >= expected[index - 1]


@pytest.mark.parametrize(
    ("objective", "previous_has_gradient"),
    [("raw_ls", True), ("variational", False), ("defect", False)],
)
def test_dense_objective_endpoint_detachment(objective, previous_has_gradient):
    sim, truth, _ = build_homogeneous_gate_case(
        family="uniform_2d",
        source_time=0.225,
        grid_size=20,
        seed=8,
    )
    cn = build_cn_tensors(
        sim,
        sigma=10.0,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    source = torch.as_tensor(truth[0], dtype=torch.float64)
    free = torch.nn.Parameter(
        source[:-1].unsqueeze(0).expand(len(sim.t) - 1, -1, -1).clone()
    )
    table = _table_from_free(free, source)
    loss, _ = _dense_table_loss(
        objective,
        table,
        cn,
        torch.tensor([1]),
        defect_sweeps=1,
        defect_omega=2.0 / 3.0,
    )
    gradient = torch.autograd.grad(loss, free)[0]
    previous_norm = float(gradient[0].abs().sum())
    later_norm = float(gradient[1].abs().sum())
    assert (previous_norm > 0.0) is previous_has_gradient
    assert later_norm > 0.0


def test_two_dimensional_interval_sampling_is_deterministic_and_direct():
    source_rng = np.random.default_rng(4)
    source_steps, source_bins = sample_transition_source_steps(
        source_rng,
        batch_size=8,
        dt=0.005,
        t_final=0.3,
        source_edges=[0.0, 0.075, 0.15, 0.225, 0.30],
    )
    assert set(source_bins) == {0, 1, 2, 3}
    assert np.all(source_steps < 60)

    kwargs = {
        "source_steps": source_steps,
        "dt": 0.005,
        "t_final": 0.3,
        "lead_edges": [0.0, 0.05, 0.10, 0.20, 0.30],
        "include_anchor_interval": True,
        "intervals_per_cell": 1,
        "consecutive_intervals": {
            "enabled": True,
            "probability": 1.0,
            "min_length": 3,
            "max_length": 3,
        },
    }
    first = sample_transition_physics_intervals(
        np.random.default_rng(9), **kwargs,
    )
    second = sample_transition_physics_intervals(
        np.random.default_rng(9), **kwargs,
    )
    for key in first:
        torch.testing.assert_close(first[key], second[key])
    rows = list(zip(
        first["sim_local"].tolist(),
        first["start_step"].tolist(),
        first["lead_bin"].tolist(),
        first["is_anchor"].tolist(),
    ))
    assert len(rows) == len(set(rows))
    for sim_local in range(len(source_steps)):
        assert any(
            row[0] == sim_local and row[1] == 0 and row[3]
            for row in rows
        )


def test_multiforcing_probe_is_adversarial_and_synchronizes_endpoints():
    specs = transition_multiforcing_probe_params()
    assert [spec["name"] for spec in specs] == [
        "zero_uniform",
        "sin_low_uniform",
        "sin_high_uniform",
        "pulse_early_uniform",
        "pulse_late_uniform",
        "sin_high_gaussian",
    ]
    intervals = {
        "sim_local": torch.zeros(3, dtype=torch.long),
        "start_step": torch.tensor([0, 7, 19]),
        "lead_bin": torch.tensor([0, 1, 2]),
        "is_anchor": torch.tensor([True, False, False]),
    }
    repeated = repeat_transition_intervals(intervals, 4)
    assert repeated["sim_local"].tolist() == [
        0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3,
    ]
    for sim_local in range(4):
        mask = repeated["sim_local"] == sim_local
        torch.testing.assert_close(
            repeated["start_step"][mask], intervals["start_step"],
        )
        torch.testing.assert_close(
            repeated["lead_bin"][mask], intervals["lead_bin"],
        )


def test_multisource_multiforcing_specs_form_complete_cartesian_product():
    specs = transition_multisource_multiforcing_specs()
    assert len(specs) == 24
    sources = {
        "uniform_2d",
        "random_sinusoid_2d",
        "grf_2d",
        "hot_spot_2d",
    }
    forcings = {
        "zero_uniform",
        "sin_low_uniform",
        "sin_high_uniform",
        "pulse_early_uniform",
        "pulse_late_uniform",
        "sin_high_gaussian",
    }
    assert {row["source_family"] for row in specs} == sources
    assert {row["forcing_name"] for row in specs} == forcings
    assert {
        (row["source_family"], row["forcing_name"]) for row in specs
    } == {
        (source, forcing)
        for source in sources
        for forcing in forcings
    }


def test_multiforcing_contrast_metrics_detect_blind_and_correct_response():
    truth_a = np.zeros((3, 2, 2))
    truth_b = np.ones((3, 2, 2))
    perfect = _contrast_metrics(
        truth_a, truth_b, truth_a, truth_b, sigma=2.0,
    )
    assert perfect["response_gain"] == pytest.approx(1.0)
    assert perfect["relative_contrast_error"] == pytest.approx(0.0)
    assert perfect["response_cosine"] == pytest.approx(1.0)

    forcing_blind = _contrast_metrics(
        truth_a, truth_a, truth_a, truth_b, sigma=2.0,
    )
    assert forcing_blind["response_gain"] == pytest.approx(0.0)
    assert forcing_blind["relative_contrast_error"] == pytest.approx(1.0)
    assert forcing_blind["response_cosine"] == pytest.approx(0.0)


def test_screen_skill_progress_and_consecutive_trigger_are_operational():
    assert transition_screen_skill(0.4, 0.0, 1.0) == pytest.approx(0.6)
    with pytest.raises(ValueError, match="separation"):
        transition_screen_skill(0.01, 0.001, 0.5)
    records = {
        500: {"overall_skill": 0.05, "long_skill": 0.01},
        750: {"overall_skill": 0.10, "long_skill": 0.05},
        1000: {"overall_skill": 0.15, "long_skill": 0.10},
        1500: {"overall_skill": 0.25, "long_skill": 0.20},
        1750: {"overall_skill": 0.30, "long_skill": 0.25},
        2000: {"overall_skill": 0.35, "long_skill": 0.30},
    }
    decision = transition_production_screen_decision(records)
    assert decision["passed"]
    assert transition_consecutive_interval_trigger(
        overall_passed=True,
        longest_passed=False,
        copy_skills=[0.1, 0.2, 0.1],
        forcing_response_ratio=0.2,
        source_departure_ratio=0.3,
        finite_and_defect_passed=True,
        low_frequency_fractions=[0.1, 0.2, 0.20, 0.20],
    )
    assert not transition_consecutive_interval_trigger(
        overall_passed=True,
        longest_passed=True,
        copy_skills=[0.1],
        forcing_response_ratio=0.2,
        source_departure_ratio=0.3,
        finite_and_defect_passed=True,
        low_frequency_fractions=[0.1, 0.2, 0.3],
    )


@pytest.mark.parametrize(
    ("objective", "detached"),
    [("raw_ls", False), ("variational", True), ("defect", True)],
)
def test_transition_physics_loss_has_direct_endpoints_and_gradients(
    objective, detached,
):
    model = ForcingTransitionCViT(
        emb_dim=16,
        dec_emb_dim=16,
        source_patch_size=4,
        source_grid_size=(8, 8),
        forcing_patch_size=4,
        forcing_grid_size=(8, 8),
        depth_enc=1,
        depth_dec=1,
        num_heads=4,
        t_final=0.3,
    )
    source = torch.zeros(1, 1, 8, 8)
    params = [{
        "temporal_family": "sin",
        "temporal_params": {
            "A": 175.0,
            "f": 4.0,
            "t_on": 0.0,
            "t_off": 0.2,
            "phase": 0.0,
            "tukey_alpha": 0.5,
            "rectified": True,
        },
        "spatial_family": "uniform",
        "spatial_params": {},
    }]
    intervals = {
        "sim_local": torch.tensor([0, 0]),
        "start_step": torch.tensor([0, 1]),
        "lead_bin": torch.tensor([0, 0]),
        "is_anchor": torch.tensor([True, False]),
    }
    axis = torch.linspace(0.0, 1.0, 8)
    result = forcing_transition_physics_loss(
        model=model,
        source_fields=source,
        params=params,
        source_steps=np.asarray([0]),
        source_bins=np.asarray([0]),
        intervals=intervals,
        x_grid=axis,
        y_grid=axis,
        y_img=np.linspace(0.0, 1.0, 8),
        nt_img=8,
        a_ref=300.0,
        t_ramp=0.01,
        t_final=0.3,
        dt=0.005,
        sigma_global=10.0,
        right_value=0.0,
        objective=objective,
        causal_epsilon=0.01,
        query_chunk=0,
    )
    assert result["finite"]
    assert result["no_rollout"]
    assert result["previous_endpoint_detached"] is detached
    assert int(result["decoded_endpoint_count"]) == 3
    assert 0.0 <= float(result["local_forcing_active_fraction"]) <= 1.0
    assert 0.0 <= float(result["history_forcing_active_fraction"]) <= 1.0
    assert torch.isfinite(result["boundary_defect_mse"])
    assert torch.isfinite(result["interior_defect_mse"])
    if objective == "raw_ls":
        assert set(result["raw_terms"]) == {
            "interior",
            "left_neumann",
            "topbot_adiabatic",
            "right_dirichlet",
        }
    else:
        assert "raw_terms" not in result
    result["loss"].backward()
    gradients = [
        parameter.grad for parameter in model.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0.0


def test_dispatch_selects_physics_transition_runner(tmp_path, monkeypatch):
    config = {
        "benchmark": {"name": "diffusion_forcing_single"},
        "model": {"cvit": {}},
        "training": {
            "pino": {
                "variant": "forcing_transition",
                "transition": {"objective": "physics_only"},
            },
        },
    }
    called = []

    def runner(received_config, seed, run_dir):
        called.append((received_config, seed, run_dir))
        return {"variant": "forcing_transition_physics"}

    monkeypatch.setattr(
        train_pino, "run_one_seed_forcing_transition_physics", runner,
    )
    result = train_pino.run_config_seeds_pino(config, tmp_path, [5])
    assert result["seeds"]["5"]["variant"] == "forcing_transition_physics"
    assert called[0][1] == 5
