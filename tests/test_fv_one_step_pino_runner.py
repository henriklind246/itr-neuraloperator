"""End-to-end + unit tests for the one-step Markov InterfaceCViT PINO runner.

The runner (``run_one_seed_interfaces_one_step_pino``) trains the 2D-image
``InterfaceCViT`` as a state-conditioned operator ``T_n -> T_{n+1}`` on a
physics-only CN objective, drawing online descriptors from a continuously
refreshed FV-state pool. These tests write a tiny synthetic interfaces dataset
(only mu/sigma/grids + the jump-scale sim_params are read; the FV manifold is
solved online), then assert the plan's Section 7 runner contracts:

- smoke completion + finite metrics + checkpoints (variational and defect),
- the startup temporal contract rejects ``step_stride != 1``,
- validation emits every required rollout/teacher/strata/conservation key,
- per-case gradient averaging equals the mean gradient,
- the objective detaches the previous state ``T_n`` from its RHS,
- the state pool yields distinct, bounded-reuse descriptors per update.
"""

import copy
import csv
import json
import math

import numpy as np
import pytest
import torch

from problems.interfaces import K_LEFT, K_RIGHT
from src.physics.boundary_forcing import A_REF_FLUX
from src.operators.train_pino import (
    T_RIGHT,
    _fixed_interval_indices,
    _resolve_interfaces_one_step_config,
    _through_origin_slope,
    anti_collapse_eligible,
    build_cvit,
    interfaces_one_step_response_slopes,
    load_interface_cvit_checkpoint,
    load_diffusion_data,
    run_one_seed_interfaces_pino,
    run_one_seed_interfaces_one_step_pino,
    validate_interfaces_one_step_gnrmse,
)
import src.operators.one_step as one_step_mod
from src.operators.one_step import (
    OneStepCase,
    OneStepStatePool,
    _one_step_candidate,
    _remove_current_contact_jump,
    predict_one_step_field,
    prepare_interfaces_one_step_case,
)
from src.physics.one_step_objective import one_step_objective
from data.dataset import problem_from_config


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _write_synthetic_interfaces(tmp_path, num_sims=8, Nt=6, Nx=20, Ny=20, t_final=0.25):
    rng = np.random.default_rng(0)
    traj = 300.0 + 5.0 * rng.standard_normal((num_sims, Nt, Nx, Ny)).astype(np.float32)
    x_grid = np.linspace(0.0, 1.0, Nx).astype(np.float32)
    y_grid = np.linspace(0.0, 1.0, Ny).astype(np.float32)
    t_grid = np.linspace(0.0, t_final, Nt).astype(np.float32)
    np.save(tmp_path / "trajectories.npy", traj)
    np.save(tmp_path / "x_grid.npy", x_grid)
    np.save(tmp_path / "y_grid.npy", y_grid)
    np.save(tmp_path / "t_grid.npy", t_grid)
    np.save(tmp_path / "dt.npy", np.array(float(t_grid[1] - t_grid[0])))
    faces = [(x_grid[8] + x_grid[9]) / 2.0, (x_grid[10] + x_grid[11]) / 2.0]
    sim_params = []
    for i in range(num_sims):
        sim_params.append({
            "T0": traj[i, 0].copy(),
            "interface_x": float(faces[i % 2]),
            "R_c": 0.2 if i % 2 == 0 else 0.7,
            "temporal_family": "sin",
            "temporal_params": {
                "A": 100.0 + i, "f": 5.0, "t_on": 0.0, "t_off": 0.2,
                "phase": 0.0, "tukey_alpha": 0.5, "rectified": True,
            },
            "spatial_family": "uniform",
            "spatial_params": {},
        })
    np.save(tmp_path / "sim_params.npy", np.asarray(sim_params, dtype=object))


def _one_step_config(tmp_path, *, objective="variational", n_cases=2):
    return {
        "data": {
            "trajectories.npy": str(tmp_path / "trajectories.npy"),
            "x_grid_path": str(tmp_path / "x_grid.npy"),
            "y_grid_path": str(tmp_path / "y_grid.npy"),
            "t_grid_path": str(tmp_path / "t_grid.npy"),
        },
        "benchmark": {"name": "interfaces", "representation": "temporal_encoder"},
        "model": {
            "cvit": {
                "in_ch": 1,
                "out_dim": 1,
                "emb_dim": 16,
                "dec_emb_dim": None,
                "patch_size": 10,
                "depth_enc": 1,
                "depth_dec": 1,
                "num_heads": 2,
                "mlp_ratio": 2.0,
                "fourier_freq": 1.0,
                "activation": "gelu",
                "hard_right_dirichlet": True,
                "hard_right_dirichlet_t_right": 300.0,
            },
            "interface_cvit": {
                "spatial_in_ch": 3,
                "forcing_in_ch": 1,
                "forcing_patch_size": 8,
                "n_param_scalars": 2,
                "num_param_tokens": 1,
                "param_hidden": 16,
            },
        },
        "training": {
            "device": "cpu",
            "epochs": 2,
            "validate_every": 1,
            "learning_rate": 0.001,
            "weight_decay": 1e-5,
            "optimizer": "Adam",
            "scheduler": {"type": "StepLR", "step_size": 1, "gamma": 0.9},
            "pino": {
                "mode": "one_step",
                "chunk_r": 0,
                "dt": 0.0625,
                "forcing": {
                    "ny_img": 16,
                    "nt_img": 32,
                    "a_ref": 300.0,
                    "ramp_seconds": 0.01,
                },
                "one_step": {
                    "step_stride": 1,
                    "objective": objective,
                    "defect_sweeps": 1,
                    "defect_omega": 0.6667,
                    "n_cases": n_cases,
                    "state_pool_size": 4,
                    "state_pool_replace_per_update": 2,
                    "max_uses_per_case": 4,
                    "time_mixture": {"uniform": 1.0},
                    "output_parameterization": "absolute",
                    "conservation": "learned",
                    "updates": 2,
                    "validate_every": 1,
                    "fast_validation_cases": 3,
                },
            },
        },
    }


def _rows(run_dir):
    with (run_dir / "train_metrics.csv").open("r", newline="") as f:
        return list(csv.DictReader(f))


VAL_KEYS = (
    "val_gnrmse", "val_gnrmse_rollout", "val_gnrmse_teacher",
    "val_rmse_K", "val_rmse_K_teacher", "node_jump_rmse_K",
    "E_model", "E_zero", "rollout_max_stepwise_gnrmse",
    "rollout_final_gnrmse", "rollout_teacher_ratio",
    "rollout_max_abs_T_K", "rollout_nonfinite",
    "hard_storage_active",
    "gnrmse_Rc_low", "gnrmse_Rc_mid", "gnrmse_Rc_high",
    "gnrmse_ix_low", "gnrmse_ix_mid", "gnrmse_ix_high",
)


def test_one_step_config_is_authoritative_for_enrichment_mode(tmp_path):
    config = _one_step_config(tmp_path)
    config["training"]["pino"]["one_step"].update({
        "output_parameterization": "rate",
        "conservation": "conservative_storage_projection",
    })
    resolved = _resolve_interfaces_one_step_config(config)
    interface = resolved["model"]["interface_cvit"]
    assert interface["jump_enrichment"] is True
    assert interface["jump_flux_mode"] == "conservative_storage_projection"
    assert config["model"]["interface_cvit"].get("jump_enrichment") is None


def test_one_step_default_uses_y_dependent_learned_flux(tmp_path):
    config = _one_step_config(tmp_path)
    del config["training"]["pino"]["one_step"]["conservation"]
    resolved = _resolve_interfaces_one_step_config(config)
    interface = resolved["model"]["interface_cvit"]
    assert resolved["training"]["pino"]["one_step"]["conservation"] == "learned"
    assert interface["jump_enrichment"] is True
    assert interface["jump_flux_mode"] == "learned"


def test_aligned_domains_require_learned_rate_operator(tmp_path):
    config = _one_step_config(tmp_path)
    config["model"]["interface_cvit"]["interface_aligned_domains"] = True
    with pytest.raises(ValueError, match="requires output_parameterization='rate'"):
        _resolve_interfaces_one_step_config(config)
    config["training"]["pino"]["one_step"]["output_parameterization"] = "rate"
    resolved = _resolve_interfaces_one_step_config(config)
    assert resolved["model"]["interface_cvit"]["jump_enrichment"] is True


def test_aligned_domain_runner_smoke(tmp_path):
    _write_synthetic_interfaces(tmp_path)
    config = _one_step_config(tmp_path, objective="defect", n_cases=1)
    config["model"]["interface_cvit"]["interface_aligned_domains"] = True
    one_step = config["training"]["pino"]["one_step"]
    one_step.update({
        "output_parameterization": "rate",
        "updates": 1,
        "fast_validation_cases": 1,
        "probe_pairs": 1,
    })
    run_dir = tmp_path / "aligned_domains"
    summary = run_one_seed_interfaces_one_step_pino(
        config, seed=3, run_dir=run_dir
    )
    checkpoint = torch.load(
        run_dir / "cvit_last.pt", map_location="cpu", weights_only=False
    )
    assert summary["objective"] == "defect"
    assert checkpoint["config"]["model"]["interface_cvit"][
        "interface_aligned_domains"
    ] is True
    architecture = json.loads((run_dir / "architecture.json").read_text())
    assert architecture["interface_aligned_domains"] is True
    restored, _ = load_interface_cvit_checkpoint(run_dir / "cvit_last.pt")
    assert restored.interface_aligned_domains is True


def test_collapse_runner_rejects_disconnected_jump_enrichment():
    config = {"model": {"interface_cvit": {"jump_enrichment": True}}}
    with pytest.raises(ValueError, match="not consumed by the collapse"):
        run_one_seed_interfaces_pino(config, seed=0, run_dir="unused")


def test_rate_parameterization_removes_current_jump_before_residual_update(tmp_path):
    _write_synthetic_interfaces(tmp_path)
    env = _build_env(_one_step_config(tmp_path), "cpu")
    case = env["make_case"](0, np.random.default_rng(3))
    state = case.truth_states[1:2]
    base = _remove_current_contact_jump(
        state,
        env["coords"],
        case.interface_x,
        case.jump_scale,
        case.geom,
        env["sigma"],
        env["a_ref"],
    )
    face = int(case.geom.face_idx.reshape(-1)[0])
    conductance = case.geom.G_x[:, face]
    q_norm = conductance * env["sigma"] * (
        state[:, face] - state[:, face + 1]
    ) / env["a_ref"]
    left = (
        env["coords"][..., 0].view_as(state)
        < case.interface_x.reshape(-1, 1, 1)
    ).to(state)
    reconstructed = base + left * case.jump_scale[:, None, None] * q_norm[:, None]
    assert torch.allclose(reconstructed, state, atol=1e-6, rtol=1e-6)

    right = torch.tensor(env["t_right_tilde"])
    zero_rate_decoded = torch.full_like(state, right)
    assert torch.allclose(
        _one_step_candidate(zero_rate_decoded, base, right, env["dt"], "rate"),
        base,
        atol=0,
    )


def test_learned_flux_rejects_storage_projection_diagnostic(tmp_path):
    _write_synthetic_interfaces(tmp_path)
    config = _resolve_interfaces_one_step_config(_one_step_config(tmp_path))
    env = _build_env(config, "cpu")
    case = env["make_case"](0, np.random.default_rng(4))
    with pytest.raises(RuntimeError, match="storage-projection diagnostics"):
        predict_one_step_field(
            env["model"],
            case.truth_states[0:1],
            case.interval_images[0:1],
            case.scalars,
            case.fixed_channels,
            env["coords"],
            dt=env["dt"],
            query_chunk=env["chunk_r"],
            interface_x=case.interface_x,
            jump_scale=case.jump_scale,
            closure_geom=case.geom,
            q_left_integral=case.q_left_integrals[0:1],
            resistance=case.resistance,
            sigma=env["sigma"],
            q_ref=env["a_ref"],
            return_storage_projection=True,
        )


def _build_env(config, device):
    """Replicate the runner's model + case setup for the unit-level tests."""
    config = _resolve_interfaces_one_step_config(config)
    dev = torch.device(device)
    data = load_diffusion_data(config)
    mu, sigma = data["mu_global"], data["sigma_global"]
    x_grid_np = np.asarray(data["x_grid"], dtype=np.float64)
    y_grid_np = np.asarray(data["y_grid"], dtype=np.float64)
    t_grid_np = np.asarray(data["t_grid"], dtype=np.float64)
    Nx, Ny = int(x_grid_np.shape[0]), int(y_grid_np.shape[0])
    t_final = float(t_grid_np[-1])
    pino = config["training"]["pino"]
    os_cfg = pino["one_step"]
    fcfg = pino["forcing"]
    a_ref = float(fcfg["a_ref"])
    ny_img, nt_img = int(fcfg["ny_img"]), int(fcfg["nt_img"])
    t_ramp = float(fcfg["ramp_seconds"])
    dt = float(pino["dt"])

    model = build_cvit(
        config, mu, sigma, grid_size=(Nx, Ny), t_final=t_final,
        variant="interfaces",
    ).to(dev)
    t_right_tilde = float((T_RIGHT - mu) / (sigma + 1e-8))

    spec = problem_from_config(config)
    gxm, gym = np.meshgrid(x_grid_np, y_grid_np, indexing="ij")
    online_grids = {"X": gxm, "Y": gym, "x_grid": x_grid_np, "y_grid": y_grid_np}
    online_time_cfg = {
        "dt": dt, "t_final": t_final, "b": 1.0, "T_right": float(T_RIGHT),
        "t_on": 0.0, "t_off": 0.2, "phase": 0.0, "tukey_alpha": 0.5,
    }
    base_kwargs = {
        "a": float(x_grid_np[0]), "b": float(x_grid_np[-1]),
        "c": float(y_grid_np[0]), "d": float(y_grid_np[-1]),
        "Nx": Nx, "Ny": Ny, "lam_target": 0.8,
        "t_final": t_final, "flux_f": 0.0, "flux_A": 0.0,
        "t_on": 0.0, "t_off": 0.2, "phase": 0.0, "tukey_alpha": 0.5,
        "dt": dt, "y_grid": y_grid_np, "ramp_seconds": t_ramp,
    }

    def case_from_params(key, params):
        return prepare_interfaces_one_step_case(
            key, params, lambda p: spec.configure_solver(p, base_kwargs),
            x_grid=x_grid_np, y_grid=y_grid_np, mu=mu, sigma=sigma,
            t_ramp=t_ramp, q_ref=a_ref, k_left=K_LEFT, k_right=K_RIGHT,
            ny_image=ny_img, image_time_points=nt_img,
            right_value=t_right_tilde, device=dev,
        )

    def make_case(key, rng):
        params = spec.sample_online_params(rng, 1, online_grids, online_time_cfg)[0]
        return case_from_params(key, params)

    def sample_params(rng):
        return spec.sample_online_params(rng, 1, online_grids, online_time_cfg)[0]

    x_grid_t = torch.as_tensor(x_grid_np, dtype=torch.float32, device=dev)
    y_grid_t = torch.as_tensor(y_grid_np, dtype=torch.float32, device=dev)
    gx, gy = torch.meshgrid(x_grid_t, y_grid_t, indexing="ij")
    coords = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1).unsqueeze(0)

    return {
        "model": model, "make_case": make_case,
        "case_from_params": case_from_params, "sample_params": sample_params,
        "coords": coords,
        "mu": mu, "sigma": sigma, "dt": dt, "a_ref": a_ref,
        "t_right_tilde": t_right_tilde, "chunk_r": Nx * Ny,
    }


# --------------------------------------------------------------------------- #
# End-to-end runner tests
# --------------------------------------------------------------------------- #
def test_one_step_runner_smoke_variational(tmp_path):
    _write_synthetic_interfaces(tmp_path)
    config = _one_step_config(tmp_path, objective="variational")
    run_dir = tmp_path / "one_step_var"

    summary = run_one_seed_interfaces_one_step_pino(config, seed=0, run_dir=run_dir)

    rows = _rows(run_dir)
    assert len(rows) == 2
    for r in rows:
        for col in ("loss", "energy", "defect_rms"):
            assert r[col] not in ("", None), col
            assert math.isfinite(float(r[col])), (col, r[col])
        # pool diversity logged every update
        assert float(r["pool_size"]) == 4.0
        assert float(r["pool_distinct_keys"]) == 4.0
    # validation ran every update (validate_every=1) -> keys populated + finite
    for r in rows:
        for key in VAL_KEYS:
            assert key in r, key
            assert r[key] not in ("", None), key
            assert math.isfinite(float(r[key])), (key, r[key])

    assert (run_dir / "cvit_best_global.pt").exists()
    assert (run_dir / "cvit_last.pt").exists()
    assert math.isfinite(summary["best_val_gnrmse"])
    assert summary["objective"] == "variational"
    final = json.loads((run_dir / "final_metrics.json").read_text())
    assert final["seed"] == 0

    # fresh-experiment guard: rerunning the same run_dir refuses stale artifacts
    with pytest.raises(FileExistsError, match="fresh experiment name"):
        run_one_seed_interfaces_one_step_pino(config, seed=0, run_dir=run_dir)


def test_rate_parameterization_with_storage_projection_runs(tmp_path):
    _write_synthetic_interfaces(tmp_path)
    config = _one_step_config(tmp_path, objective="variational", n_cases=1)
    one_step = config["training"]["pino"]["one_step"]
    one_step.update({
        "updates": 1,
        "fast_validation_cases": 1,
        "probe_pairs": 1,
        "output_parameterization": "rate",
        "conservation": "conservative_storage_projection",
    })
    run_dir = tmp_path / "one_step_rate_projection"
    run_one_seed_interfaces_one_step_pino(config, seed=0, run_dir=run_dir)
    row = _rows(run_dir)[0]
    assert float(row["hard_storage_active"]) == 1.0
    assert math.isfinite(float(row["hard_global_storage_error"]))
    assert float(row["hard_global_storage_error"]) < 1e-5


def test_one_step_runner_defect_objective(tmp_path):
    _write_synthetic_interfaces(tmp_path)
    config = _one_step_config(tmp_path, objective="defect")
    run_dir = tmp_path / "one_step_defect"

    summary = run_one_seed_interfaces_one_step_pino(config, seed=1, run_dir=run_dir)

    rows = _rows(run_dir)
    assert rows
    for r in rows:
        for col in ("loss", "energy", "defect_rms"):
            assert math.isfinite(float(r[col])), (col, r[col])
    assert summary["objective"] == "defect"
    assert math.isfinite(summary["best_val_gnrmse"])


def test_fixed_interval_indices_span_the_fv_trajectory():
    assert _fixed_interval_indices(60, 1) == [0]
    indices = _fixed_interval_indices(60, 16)
    assert indices[0] == 0
    assert indices[-1] == 59
    assert len(indices) == len(set(indices)) == 16
    with pytest.raises(ValueError, match="cannot exceed"):
        _fixed_interval_indices(3, 4)


def test_one_step_fixed_simulation_trains_multiple_states(tmp_path):
    _write_synthetic_interfaces(tmp_path)
    config = _one_step_config(tmp_path, objective="defect")
    one_step = config["training"]["pino"]["one_step"]
    one_step.update({
        "fixed_simulation": {"enabled": True, "num_states": 3},
        "updates": 2,
        "validate_every": 1,
        "fast_validation_cases": 1,
        "probe_pairs": 1,
    })
    run_dir = tmp_path / "fixed_multistate"

    summary = run_one_seed_interfaces_one_step_pino(
        config, seed=3, run_dir=run_dir
    )

    rows = _rows(run_dir)
    assert len(rows) == 2
    for row in rows:
        assert float(row["pool_distinct_keys"]) == 1.0
        assert float(row["pool_size"]) == 1.0
        assert math.isfinite(float(row["fixed_train_rmse_K"]))
        assert math.isfinite(float(row["fixed_train_transition_rel"]))
    with (run_dir / "fixed_state_metrics.csv").open(newline="") as f:
        state_rows = list(csv.DictReader(f))
    assert len(state_rows) == 2 * 3
    assert {int(float(row["interval"])) for row in state_rows} == {0, 2, 3}
    assert (run_dir / "cvit_best_fixed.pt").exists()
    assert summary["fixed_simulation"] is True
    assert summary["fixed_intervals"] == [0, 2, 3]
    assert math.isfinite(summary["best_fixed_transition_rel"])


def test_fixed_set_trains_multiple_simulations(tmp_path):
    _write_synthetic_interfaces(tmp_path)
    config = _one_step_config(tmp_path, objective="defect")
    one_step = config["training"]["pino"]["one_step"]
    one_step.update({
        "fixed_simulation": {
            "enabled": True,
            "num_simulations": 2,
            "num_states": 2,
            "normalize_per_state_defect": True,
        },
        "updates": 1,
        "validate_every": 1,
        "fast_validation_cases": 1,
        "probe_pairs": 1,
    })
    run_dir = tmp_path / "fixed_multi_simulation"

    summary = run_one_seed_interfaces_one_step_pino(
        config, seed=8, run_dir=run_dir
    )

    row = _rows(run_dir)[0]
    assert float(row["pool_distinct_keys"]) == 2.0
    assert float(row["pool_size"]) == 2.0
    with (run_dir / "fixed_state_metrics.csv").open(newline="") as f:
        state_rows = list(csv.DictReader(f))
    assert len(state_rows) == 2 * 2
    assert {int(row["simulation"]) for row in state_rows} == {0, 1}
    checkpoint = torch.load(
        run_dir / "cvit_last.pt", map_location="cpu", weights_only=False
    )
    assert len(checkpoint["fixed_simulation_state"]["case_params"]) == 2
    assert tuple(checkpoint["fixed_case_weights"].shape) == (2, 2)
    assert summary["fixed_num_simulations"] == 2


def test_fixed_simulation_warm_start_reuses_states_and_freezes_weights(tmp_path):
    _write_synthetic_interfaces(tmp_path)

    source_config = _one_step_config(tmp_path, objective="defect")
    source_one_step = source_config["training"]["pino"]["one_step"]
    source_one_step.update({
        "fixed_simulation": {"enabled": True, "num_states": 3},
        "updates": 1,
        "validate_every": 1,
        "fast_validation_cases": 1,
        "probe_pairs": 1,
    })
    source_dir = tmp_path / "fixed_source"
    run_one_seed_interfaces_one_step_pino(
        source_config, seed=4, run_dir=source_dir
    )
    source_path = source_dir / "cvit_best_fixed.pt"
    source = torch.load(source_path, map_location="cpu", weights_only=False)

    continuation = copy.deepcopy(source_config)
    continuation["training"]["learning_rate"] = 1.0e-4
    continuation_one_step = continuation["training"]["pino"]["one_step"]
    continuation_one_step["init_from_checkpoint"] = str(source_path)
    continuation_one_step["fixed_simulation"].update({
        "normalize_per_state_defect": True,
        "normalization_floor_fraction": 0.01,
    })
    continuation_dir = tmp_path / "fixed_continuation"
    summary = run_one_seed_interfaces_one_step_pino(
        continuation, seed=99, run_dir=continuation_dir
    )

    continued = torch.load(
        continuation_dir / "cvit_last.pt", map_location="cpu", weights_only=False
    )
    np.testing.assert_array_equal(
        continued["fixed_simulation_state"]["params"]["T0"],
        source["fixed_simulation_state"]["params"]["T0"],
    )
    assert continued["fixed_simulation_state"]["intervals"] == [0, 2, 3]
    weights = continued["fixed_case_weights"].numpy()
    assert np.isfinite(weights).all()
    assert (weights > 0.0).all()
    assert float(weights.mean()) == pytest.approx(1.0)
    weight_rows = json.loads(
        (continuation_dir / "fixed_state_weights.json").read_text()
    )
    assert [row["interval"] for row in weight_rows] == [0, 2, 3]
    assert summary["fixed_state_defect_normalization"] is True
    assert summary["source_checkpoint"] == str(source_path.resolve())


def test_one_step_runner_rejects_step_stride(tmp_path):
    _write_synthetic_interfaces(tmp_path)
    config = _one_step_config(tmp_path)
    config["training"]["pino"]["one_step"]["step_stride"] = 2
    run_dir = tmp_path / "bad_stride"
    with pytest.raises(ValueError, match="step_stride must be 1"):
        run_one_seed_interfaces_one_step_pino(config, seed=0, run_dir=run_dir)


def test_one_step_runner_rejects_unknown_objective(tmp_path):
    _write_synthetic_interfaces(tmp_path)
    config = _one_step_config(tmp_path, objective="residual_mse")
    run_dir = tmp_path / "bad_obj"
    with pytest.raises(ValueError, match="objective must be"):
        run_one_seed_interfaces_one_step_pino(config, seed=0, run_dir=run_dir)


# --------------------------------------------------------------------------- #
# Validation contract
# --------------------------------------------------------------------------- #
def test_validation_emits_all_required_keys(tmp_path):
    _write_synthetic_interfaces(tmp_path)
    config = _one_step_config(tmp_path)
    env = _build_env(config, "cpu")
    rng = np.random.default_rng(7)
    cases = [env["make_case"](-1 - i, rng) for i in range(3)]

    out = validate_interfaces_one_step_gnrmse(
        env["model"], cases,
        mu=env["mu"], sigma=env["sigma"], sigma_dT_train=1.0,
        dt=env["dt"], coords=env["coords"], query_chunk=env["chunk_r"],
        a_ref=env["a_ref"], right_value=env["t_right_tilde"],
        device=torch.device("cpu"),
    )
    for key in VAL_KEYS:
        assert key in out, key
    assert "hard_global_storage_error" in out
    assert math.isnan(out["hard_global_storage_error"])
    # val_gnrmse mirrors the rollout gate; teacher-forced is a separate scalar
    assert out["val_gnrmse"] == out["val_gnrmse_rollout"]
    assert math.isfinite(out["val_gnrmse"])
    assert math.isfinite(out["val_gnrmse_teacher"])
    assert out["rollout_nonfinite"] == 0.0


# --------------------------------------------------------------------------- #
# Gradient averaging + detachment
# --------------------------------------------------------------------------- #
def _case_loss(env, case, step):
    model = env["model"]
    state_n = case.truth_states[step : step + 1].to("cpu")
    pred = predict_one_step_field(
        model, state_n, case.interval_images[step : step + 1],
        case.scalars, case.fixed_channels, env["coords"],
        dt=env["dt"], query_chunk=env["chunk_r"],
        interface_x=case.interface_x, jump_scale=case.jump_scale,
        closure_geom=case.geom,
        q_left_integral=case.q_left_integrals[step : step + 1],
        resistance=case.resistance, sigma=env["sigma"], q_ref=env["a_ref"],
    )
    cn_step = {**case.cn, "forcing": case.cn["forcing"][step : step + 1]}
    loss, _ = one_step_objective(
        "variational", pred, state_n.detach(), cn_step,
        right_value=env["t_right_tilde"],
    )
    return loss


def test_gradient_average_equals_mean(tmp_path):
    _write_synthetic_interfaces(tmp_path)
    config = _one_step_config(tmp_path)
    env = _build_env(config, "cpu")
    rng = np.random.default_rng(3)
    cases = [env["make_case"](i, rng) for i in range(2)]
    model = env["model"]
    n_cases = len(cases)

    # (a) per-case pre-divided backward accumulation (the runner's mechanism)
    model.zero_grad(set_to_none=True)
    for c in cases:
        (_case_loss(env, c, 0) / n_cases).backward()
    accum = [p.grad.detach().clone() for p in model.parameters() if p.grad is not None]

    # (b) single backward of the explicit mean loss
    model.zero_grad(set_to_none=True)
    mean_loss = sum(_case_loss(env, c, 0) for c in cases) / n_cases
    mean_loss.backward()
    direct = [p.grad.detach().clone() for p in model.parameters() if p.grad is not None]

    assert len(accum) == len(direct) and accum
    for ga, gd in zip(accum, direct):
        assert torch.allclose(ga, gd, atol=1e-6, rtol=1e-5)


def test_objective_detaches_previous_state(tmp_path):
    _write_synthetic_interfaces(tmp_path)
    config = _one_step_config(tmp_path)
    env = _build_env(config, "cpu")
    rng = np.random.default_rng(5)
    case = env["make_case"](0, rng)

    # prediction_np1 carries the trainable path; prediction_n is the previous
    # state whose RHS contribution the objective must detach.
    pred_np1 = case.truth_states[1:2].to("cpu").clone().requires_grad_(True)
    prev = case.truth_states[0:1].to("cpu").clone().requires_grad_(True)
    cn_step = {**case.cn, "forcing": case.cn["forcing"][0:1]}
    loss, _ = one_step_objective(
        "variational", pred_np1, prev, cn_step, right_value=env["t_right_tilde"],
    )
    loss.backward()
    assert prev.grad is None
    assert pred_np1.grad is not None
    assert torch.isfinite(pred_np1.grad).all()


# --------------------------------------------------------------------------- #
# State pool invariants
# --------------------------------------------------------------------------- #
def _fake_case(key, n_intervals=6):
    z = torch.zeros(1)
    scores = np.ones(n_intervals, dtype=np.float64)
    return OneStepCase(
        key=int(key), params={}, n_intervals=int(n_intervals),
        truth_states=z, interval_images=z, scalars=z, interface_x=z,
        jump_scale=z, cn={}, geom=None, q_left_integrals=z, resistance=0.5,
        fixed_channels=z, forcing_norm=scores.copy(),
        storage_change=scores.copy(), interface_jump=scores.copy(),
    )


def test_state_pool_distinct_and_bounded():
    keys_made = []

    def factory(key):
        keys_made.append(key)
        return _fake_case(key)

    pool = OneStepStatePool(
        factory, pool_size=4, n_cases=3, replace_per_update=2,
        max_uses_per_case=4, time_mixture={"uniform": 1.0},
        rng=np.random.default_rng(0),
    )
    for _ in range(6):
        pool.refresh()
        selection = pool.select()
        assert len(selection) == 3
        chosen_keys = [case.key for case, _ in selection]
        # no descriptor selected twice within one update
        assert len(set(chosen_keys)) == 3
        # each pairing uses a fresh (not-yet-used) interval where available
        for case, interval in selection:
            assert 0 <= interval < case.n_intervals
        stats = pool.diversity_stats()
        assert stats["pool_size"] == 4.0
        assert stats["pool_distinct_keys"] == 4.0

    # bounded reuse: no case ever exceeds max_uses_per_case while pooled
    for case in pool.cases:
        assert case.uses <= 4


def test_state_pool_rejects_n_cases_gt_pool():
    with pytest.raises(ValueError, match="n_cases cannot exceed pool_size"):
        OneStepStatePool(
            lambda k: _fake_case(k), pool_size=2, n_cases=4,
            replace_per_update=1, max_uses_per_case=4,
            time_mixture={"uniform": 1.0}, rng=np.random.default_rng(0),
        )


# --------------------------------------------------------------------------- #
# Anti-collapse gate (plan Section 6)
# --------------------------------------------------------------------------- #
def test_through_origin_slope_recovers_known_slope():
    true = np.arange(1.0, 11.0)
    # exact through-origin relationship pred = 2.5 * true -> slope 2.5
    assert _through_origin_slope(true, 2.5 * true) == pytest.approx(2.5)
    # a constant-field collapse (pred ~ 0) -> slope 0, not nan
    assert _through_origin_slope(true, np.zeros_like(true)) == pytest.approx(0.0)
    # degenerate true signal -> nan (gate treats as ineligible)
    assert math.isnan(_through_origin_slope(np.zeros(5), np.arange(5.0)))


def test_anti_collapse_gate_rejects_collapse_accepts_transport():
    healthy = {
        "forcing_response_slope": 0.8,
        "jump_response_slope": 0.4,
        "rollout_nonfinite": 0.0,
        "val_gnrmse": 0.05,
        "rollout_max_abs_T_K": 350.0,
        "hard_global_storage_error": 1e-9,
        "hard_storage_active": 1.0,
    }
    assert anti_collapse_eligible(
        healthy, forcing_slope_min=0.25, jump_slope_min=0.10,
        max_abs_T_K=1000.0, hard_storage_tol=1e-6, require_finite=True,
    )

    inactive_storage = {
        **healthy,
        "hard_storage_active": 0.0,
        "hard_global_storage_error": float("nan"),
    }
    assert anti_collapse_eligible(
        inactive_storage,
        forcing_slope_min=0.25,
        jump_slope_min=0.10,
        hard_storage_tol=1e-6,
    )

    # 300 K constant-field collapse: no forcing/jump response
    collapsed = {**healthy, "forcing_response_slope": 0.0,
                 "jump_response_slope": 0.0}
    assert not anti_collapse_eligible(
        collapsed, forcing_slope_min=0.25, jump_slope_min=0.10,
    )

    # jump alone below threshold is disqualifying
    weak_jump = {**healthy, "jump_response_slope": 0.05}
    assert not anti_collapse_eligible(
        weak_jump, forcing_slope_min=0.25, jump_slope_min=0.10,
    )

    # non-finite rollout is ineligible when require_finite
    bad_roll = {**healthy, "rollout_nonfinite": 3.0}
    assert not anti_collapse_eligible(
        bad_roll, forcing_slope_min=0.25, jump_slope_min=0.10,
    )

    # exceeding the temperature bound is ineligible
    hot = {**healthy, "rollout_max_abs_T_K": 5000.0}
    assert not anti_collapse_eligible(
        hot, forcing_slope_min=0.25, jump_slope_min=0.10, max_abs_T_K=1000.0,
    )

    # None thresholds skip that condition: a huge max|T| passes when unbounded
    assert anti_collapse_eligible(
        hot, forcing_slope_min=0.25, jump_slope_min=0.10, max_abs_T_K=None,
        hard_storage_tol=None,
    )


def test_response_slopes_reject_constant_field(tmp_path, monkeypatch):
    import copy as _copy

    _write_synthetic_interfaces(tmp_path)
    config = _one_step_config(tmp_path)
    env = _build_env(config, "cpu")
    rng = np.random.default_rng(11)

    # Frozen paired-amplitude probe: one descriptor, two forcing amplitudes that
    # share the IC / interface_x / R_c so differencing isolates the response.
    def make_pair(key):
        base = env["sample_params"](rng)
        a_hi = float(base["temporal_params"]["A"])
        pair = []
        for i, amp in enumerate((0.5 * a_hi, a_hi)):
            p = _copy.deepcopy(base)
            p["temporal_params"] = dict(p["temporal_params"])
            p["temporal_params"]["A"] = float(amp)
            pair.append(env["case_from_params"]((key * 2) + i, p))
        return (pair[0], pair[1])

    probe_pairs = [make_pair(-2000 - i) for i in range(2)]

    # A collapsed operator returns the SAME field regardless of forcing.
    def _constant_predict(model, state_n, *args, **kwargs):
        return state_n

    monkeypatch.setattr(one_step_mod, "predict_one_step_field", _constant_predict)
    slopes = interfaces_one_step_response_slopes(
        env["model"], probe_pairs,
        dt=env["dt"], coords=env["coords"], query_chunk=env["chunk_r"],
        a_ref=env["a_ref"], sigma=env["sigma"], device=torch.device("cpu"),
    )
    for key in ("forcing_response_slope", "jump_response_slope",
                "jump_rmse_K", "jump_correlation", "jump_sign_fraction"):
        assert key in slopes, key
    # constant field -> zero response -> gate rejects
    assert slopes["forcing_response_slope"] == pytest.approx(0.0, abs=1e-9)
    assert slopes["jump_response_slope"] == pytest.approx(0.0, abs=1e-9)
    assert not anti_collapse_eligible(
        slopes, forcing_slope_min=0.25, jump_slope_min=0.10,
    )


def test_runner_logs_anti_collapse_columns(tmp_path):
    _write_synthetic_interfaces(tmp_path)
    config = _one_step_config(tmp_path)
    config["training"]["pino"]["one_step"]["probe_pairs"] = 2
    run_dir = tmp_path / "gate_cols"
    run_one_seed_interfaces_one_step_pino(config, seed=0, run_dir=run_dir)

    rows = _rows(run_dir)
    assert rows
    gate_cols = (
        "hard_storage_active", "hard_global_storage_error", "forcing_response_slope",
        "jump_response_slope", "jump_rmse_K", "jump_correlation",
        "jump_sign_fraction", "anti_collapse_eligible",
    )
    for r in rows:
        for col in gate_cols:
            assert col in r, col
            assert r[col] not in ("", None), col
        assert int(r["anti_collapse_eligible"]) in (0, 1)
        # learned jump-flux mode has no structural storage projection.
        assert float(r["hard_storage_active"]) == 0.0
        assert math.isnan(float(r["hard_global_storage_error"]))


# --------------------------------------------------------------------------- #
# Exact-resume reproducibility (plan Section 9)
# --------------------------------------------------------------------------- #
def test_case_regeneration_is_deterministic(tmp_path):
    """FV manifolds are regenerated bit-identically from a stored descriptor.

    The resume path never serializes trajectories; it re-solves them from
    ``params``. This is the checksum guarantee underpinning pool resume.
    """
    _write_synthetic_interfaces(tmp_path)
    config = _one_step_config(tmp_path)
    env = _build_env(config, "cpu")
    params = env["sample_params"](np.random.default_rng(123))

    c1 = env["case_from_params"](7, params)
    c2 = env["case_from_params"](7, params)
    assert torch.equal(c1.truth_states, c2.truth_states)
    assert torch.equal(c1.interval_images, c2.interval_images)
    assert torch.equal(c1.q_left_integrals, c2.q_left_integrals)


def _build_resumable_pool(env, seed):
    """A pool wired exactly like the runner: separate factory + selection rng."""
    factory_rng = np.random.default_rng(seed + 101)
    pool = OneStepStatePool(
        lambda key: env["make_case"](key, factory_rng),
        pool_size=4, n_cases=2, replace_per_update=2,
        max_uses_per_case=4, time_mixture={"uniform": 1.0},
        rng=np.random.default_rng(seed + 202),
    )
    return pool, factory_rng


def test_state_pool_resume_matches_uninterrupted(tmp_path):
    """Plan 7.7: after resume, the next descriptor batch and the selected
    ``n`` indices match an uninterrupted run.

    The pool serializes only descriptors + rng state; the factory rng is stored
    alongside (as the runner does). Rebuilding the pool, loading the state, and
    restoring the factory rng must reproduce the identical future stream.
    """
    _write_synthetic_interfaces(tmp_path)
    config = _one_step_config(tmp_path)
    env = _build_env(config, "cpu")

    def cycle(pool):
        pool.refresh()
        return [(int(c.key), int(step)) for c, step in pool.select()]

    # Uninterrupted reference stream.
    ref_pool, _ = _build_resumable_pool(env, seed=0)
    ref = [cycle(ref_pool) for _ in range(6)]

    # Interrupted: 3 cycles, snapshot, rebuild fresh, restore, continue.
    a_pool, a_frng = _build_resumable_pool(env, seed=0)
    for _ in range(3):
        cycle(a_pool)
    pool_state = a_pool.state_dict()
    frng_state = a_frng.bit_generator.state

    b_pool, b_frng = _build_resumable_pool(env, seed=0)  # _fill advances b_frng
    b_pool.load_state_dict(pool_state, env["case_from_params"])
    b_frng.bit_generator.state = frng_state  # restore factory rng post-construction
    resumed = [cycle(b_pool) for _ in range(3)]

    assert resumed == ref[3:]


def _model_state_close(a, b):
    ka, kb = set(a.keys()), set(b.keys())
    assert ka == kb
    for k in ka:
        assert torch.equal(a[k], b[k]), k


def test_runner_resume_matches_uninterrupted(tmp_path):
    """A split (train -> checkpoint -> resume) run reproduces the uninterrupted
    run's final weights and metrics exactly (plan Section 9)."""
    _write_synthetic_interfaces(tmp_path)

    def cfg():
        c = _one_step_config(tmp_path)
        os_cfg = c["training"]["pino"]["one_step"]
        os_cfg["updates"] = 4
        os_cfg["validate_every"] = 2
        os_cfg["probe_pairs"] = 2
        return c

    # Uninterrupted 4-update run.
    ref_dir = tmp_path / "resume_ref"
    run_one_seed_interfaces_one_step_pino(cfg(), seed=0, run_dir=ref_dir)
    ref_last = torch.load(ref_dir / "cvit_last.pt", map_location="cpu",
                          weights_only=False)["model_state"]

    # Split run: 2 updates, then resume for the remaining 2.
    split_dir = tmp_path / "resume_split"
    phase1 = cfg()
    phase1["training"]["pino"]["one_step"]["updates"] = 2
    run_one_seed_interfaces_one_step_pino(phase1, seed=0, run_dir=split_dir)
    assert (split_dir / "resume_state.pt").exists()

    phase2 = cfg()
    phase2["training"]["pino"]["one_step"]["resume"] = True
    run_one_seed_interfaces_one_step_pino(phase2, seed=0, run_dir=split_dir)
    split_last = torch.load(split_dir / "cvit_last.pt", map_location="cpu",
                            weights_only=False)["model_state"]

    _model_state_close(ref_last, split_last)

    # The resumed run continues update indices rather than restarting them.
    updates = [int(r["update"]) for r in _rows(split_dir)]
    assert max(updates) == 3
    assert updates[-1] == 3


def test_runner_resume_requires_existing_state(tmp_path):
    """resume=true without a resume_state.pt is treated as a fresh run and still
    honors the stale-artifact guard."""
    _write_synthetic_interfaces(tmp_path)
    config = _one_step_config(tmp_path)
    config["training"]["pino"]["one_step"]["resume"] = True
    run_dir = tmp_path / "resume_missing"

    # No prior state -> behaves as a fresh run (no crash, artifacts written).
    run_one_seed_interfaces_one_step_pino(config, seed=0, run_dir=run_dir)
    assert (run_dir / "cvit_last.pt").exists()

    # A second resume=true call now finds stale artifacts AND a resume_state.pt,
    # so it resumes (already complete) instead of raising.
    run_one_seed_interfaces_one_step_pino(config, seed=0, run_dir=run_dir)


def test_runner_rejects_non_interface_cvit(tmp_path, monkeypatch):
    """The encoder guard (plan Section 10) rejects a model that is not the 2D
    forcing-image InterfaceCViT before any training runs."""
    import torch.nn as nn
    import src.operators.train_pino as train_pino_mod

    _write_synthetic_interfaces(tmp_path)
    config = _one_step_config(tmp_path)
    run_dir = tmp_path / "guard_reject_class"

    class _NotCViT(nn.Module):
        def __init__(self):
            super().__init__()
            self.p = nn.Parameter(torch.zeros(1))

    monkeypatch.setattr(train_pino_mod, "build_cvit", lambda *a, **k: _NotCViT())
    with pytest.raises(ValueError, match="forcing-image InterfaceCViT"):
        run_one_seed_interfaces_one_step_pino(config, seed=0, run_dir=run_dir)
    assert not (run_dir / "architecture.json").exists()


def test_runner_rejects_forcing_grid_mismatch(tmp_path, monkeypatch):
    """A real InterfaceCViT built at the wrong forcing_grid_size is rejected by
    the encoder guard (patch-token / grid contract)."""
    import src.operators.train_pino as train_pino_mod

    _write_synthetic_interfaces(tmp_path)
    config = _one_step_config(tmp_path)
    run_dir = tmp_path / "guard_reject_grid"

    real_build = train_pino_mod.build_cvit

    def _wrong_grid(cfg, *a, **k):
        cfg = copy.deepcopy(cfg)
        fcfg = cfg["training"]["pino"]["forcing"]
        fcfg["ny_img"] = int(fcfg["ny_img"]) + 8
        return real_build(cfg, *a, **k)

    monkeypatch.setattr(train_pino_mod, "build_cvit", _wrong_grid)
    with pytest.raises(ValueError, match="forcing_grid_size"):
        run_one_seed_interfaces_one_step_pino(config, seed=0, run_dir=run_dir)


def test_runner_writes_architecture_metadata(tmp_path):
    """A normal run records the concrete encoder class + architecture version in
    ``architecture.json`` (plan Section 10)."""
    from src.operators.cvit import CViTEncoder

    _write_synthetic_interfaces(tmp_path)
    config = _one_step_config(tmp_path)
    run_dir = tmp_path / "guard_arch_meta"

    run_one_seed_interfaces_one_step_pino(config, seed=0, run_dir=run_dir)

    arch = json.loads((run_dir / "architecture.json").read_text())
    assert arch["model_class"] == "InterfaceCViT"
    assert arch["forcing_encoder_class"] == CViTEncoder.__name__
    fcfg = config["training"]["pino"]["forcing"]
    assert arch["forcing_grid_size"] == [int(fcfg["ny_img"]), int(fcfg["nt_img"])]
    expected_tokens = (
        (arch["forcing_grid_size"][0] // arch["forcing_patch_size"])
        * (arch["forcing_grid_size"][1] // arch["forcing_patch_size"])
    )
    assert arch["num_forcing_tokens"] == expected_tokens
    assert isinstance(arch["version"], int)


def test_runner_tolerates_float32_t_final_quantization(tmp_path):
    """t_final reconstructed from the float32-saved t_grid carries ~1e-8 noise
    (0.3 -> 0.30000001192). With an explicit dt that mathematically divides the
    clean t_final, the runner must snap the noise out instead of failing the
    solver's exact-divisibility check."""
    # t_final=0.3 is not exactly representable in float32; dt=0.1 divides the
    # clean value but not the quantized one (0.30000001192 / 0.1 = 3.0000001).
    _write_synthetic_interfaces(tmp_path, t_final=0.3)
    t_grid = np.load(tmp_path / "t_grid.npy")
    assert float(t_grid[-1]) != 0.3  # float32 quantization noise is present

    config = _one_step_config(tmp_path)
    config["training"]["pino"]["dt"] = 0.1
    run_dir = tmp_path / "t_final_quant"

    run_one_seed_interfaces_one_step_pino(config, seed=0, run_dir=run_dir)
    assert (run_dir / "cvit_last.pt").exists()
