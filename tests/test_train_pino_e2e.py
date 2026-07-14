"""Tiny end-to-end run of the physics-only CViT (PINO) trainer.

Writes a small synthetic diffusion dataset to a tmp dir, builds a minimal
plain-dict config (small CViT, SOAP, tiny collocation counts), runs two epochs
through the real ``run_one_seed_pino``, and asserts:

- the loss and its residual/IC/Neumann components are finite,
- a validation rel-L2 is computed against the saved trajectories,
- the best checkpoint carries the global (mu, sigma) baked in, and
- ``train_metrics.csv`` / ``final_metrics.json`` are written.

The synthetic grid is 20x20 so ``patch_size=10`` divides it (the fixture's
11x11 grid is prime and cannot be patched cleanly).
"""

import csv
import json
import math

import numpy as np
import pytest
import torch
import src.operators.train_pino as train_pino_mod

from src.operators.train_pino import (
    _adapt_causal_eps,
    _causal_bin_stats,
    _causal_residual_loss,
    _causal_weights,
    _curriculum_weights,
    _ic_loss,
    _resolve_forcing_causal,
    _time_bin_index,
    load_interface_cvit_checkpoint,
    run_one_seed_forcing_pino,
    run_one_seed_interfaces_pino,
    run_one_seed_pino,
    sample_forcing_params,
)


def _write_synthetic_diffusion(tmp_path, num_sims=16, Nt=6, Nx=20, Ny=20):
    rng = np.random.default_rng(0)
    # Smooth-ish fields around 300 K so the normalized targets are well posed.
    traj = 300.0 + 5.0 * rng.standard_normal((num_sims, Nt, Nx, Ny)).astype(np.float32)
    x_grid = np.linspace(0.0, 1.0, Nx).astype(np.float32)
    y_grid = np.linspace(0.0, 1.0, Ny).astype(np.float32)
    t_grid = np.linspace(0.0, 0.3, Nt).astype(np.float32)
    np.save(tmp_path / "trajectories.npy", traj)
    np.save(tmp_path / "x_grid.npy", x_grid)
    np.save(tmp_path / "y_grid.npy", y_grid)
    np.save(tmp_path / "t_grid.npy", t_grid)
    return traj


def _config(tmp_path):
    return {
        "data": {
            "trajectories.npy": str(tmp_path / "trajectories.npy"),
            "x_grid_path": str(tmp_path / "x_grid.npy"),
            "y_grid_path": str(tmp_path / "y_grid.npy"),
            "t_grid_path": str(tmp_path / "t_grid.npy"),
        },
        "model": {
            "cvit": {
                "in_ch": 1,
                "out_dim": 1,
                "emb_dim": 32,
                "dec_emb_dim": None,
                "patch_size": 10,
                "depth_enc": 1,
                "depth_dec": 1,
                "num_heads": 4,
                "mlp_ratio": 2.0,
                "fourier_freq": 1.0,
                "activation": "gelu",
                "hard_right_dirichlet": True,
                "hard_right_dirichlet_t_right": 300.0,
            }
        },
        "training": {
            "device": "cpu",
            "epochs": 2,
            "validate_every": 1,
            "learning_rate": 0.001,
            "weight_decay": 1e-5,
            "optimizer": "SOAP",
            "soap": {
                "betas": [0.95, 0.95],
                "shampoo_beta": 0.95,
                "eps": 1.0e-8,
                "precondition_frequency": 3,
                "max_precond_dim": 10000,
                "merge_dims": False,
                "precondition_1d": False,
            },
            "scheduler": {"type": "StepLR", "step_size": 1, "gamma": 0.9},
            "pino": {
                "lambda_r": 1.0,
                "lambda_ic": 1.0,
                "lambda_bc": 1.0,
                "n_r": 64,
                "n_ic": 32,
                "n_bc": 16,
                "sim_batch": 4,
                "alpha": 1.0,
            },
        },
    }


def _write_synthetic_interfaces(tmp_path, num_sims=12, Nt=3, Nx=20, Ny=20):
    traj = _write_synthetic_diffusion(tmp_path, num_sims, Nt, Nx, Ny)
    x_grid = np.load(tmp_path / "x_grid.npy")
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


def _interface_config(tmp_path):
    cfg = _config(tmp_path)
    cfg["model"]["cvit"].update({"emb_dim": 16, "num_heads": 2})
    cfg["model"]["interface_cvit"] = {
        "spatial_in_ch": 3,
        "forcing_in_ch": 1,
        "forcing_patch_size": 8,
        "n_param_scalars": 2,
        "num_param_tokens": 1,
        "param_hidden": 16,
    }
    cfg["training"].update({"epochs": 1, "optimizer": "Adam"})
    cfg["training"]["pino"] = {
        "lambda_r": 1.0,
        "lambda_ic": 1.0,
        "lambda_bc": 1.0,
        "lambda_bc_left": 1.0,
        "sim_batch": 2,
        "chunk_r": 0,
        "collocation_source": "saved_train",
        "dt": 0.005,
        "intervals_per_sim": 1,
        "stratified_time_sampling": False,
        "causal": {"enabled": False},
        "interface_band": {"enabled": False},
        "region_standardize": {"enabled": False},
        "forcing": {
            "ny_img": 16,
            "nt_img": 32,
            "a_ref": 300.0,
            "ramp_seconds": 0.01,
        },
    }
    return cfg


def _rows(run_dir):
    with (run_dir / "train_metrics.csv").open("r", newline="") as f:
        return list(csv.DictReader(f))


def _named_rows(run_dir, name):
    with (run_dir / name).open("r", newline="") as f:
        return list(csv.DictReader(f))


def test_e2e_runs_logs_and_checkpoints(tmp_path):
    _write_synthetic_diffusion(tmp_path)
    cfg = _config(tmp_path)
    run_dir = tmp_path / "run"

    summary = run_one_seed_pino(cfg, seed=0, run_dir=run_dir)

    # ---- loss components are finite every epoch ----
    rows = _rows(run_dir)
    assert len(rows) == 2
    for r in rows:
        for col in ("loss", "loss_r", "loss_ic", "loss_bc"):
            assert r[col] not in ("", None), col
            assert math.isfinite(float(r[col])), (col, r[col])
            assert float(r[col]) >= 0.0

    # ---- validation computed (validate_every=1 -> every epoch) ----
    for r in rows:
        assert r["val_rel_l2"] not in ("", None)
        assert math.isfinite(float(r["val_rel_l2"]))
        assert float(r["val_rel_l2"]) >= 0.0

    # ---- checkpoint carries the global stats ----
    ckpt = torch.load(run_dir / "cvit_best.pt", map_location="cpu", weights_only=False)
    assert "mu_global" in ckpt and "sigma_global" in ckpt
    assert math.isfinite(ckpt["mu_global"]) and ckpt["sigma_global"] > 0.0
    assert "model_state" in ckpt and ckpt["model_state"]

    # ---- summary + artifacts ----
    assert math.isfinite(summary["best_val_rel_l2"])
    assert summary["epochs"] == 2
    final = json.loads((run_dir / "final_metrics.json").read_text())
    assert final["seed"] == 0

    # ---- default path: static weights, no gradnorm state ----
    for r in rows:
        assert float(r["w_r"]) == 1.0
        assert float(r["w_ic"]) == 1.0
        assert float(r["w_bc"]) == 1.0
    assert ckpt.get("gradnorm_state") is None


def test_interface_image_e2e_has_finite_metrics_and_complete_metadata(tmp_path):
    _write_synthetic_interfaces(tmp_path)
    config = _interface_config(tmp_path)
    run_dir = tmp_path / "interface_run"

    summary = run_one_seed_interfaces_pino(config, seed=0, run_dir=run_dir)
    rows = _rows(run_dir)
    assert len(rows) == 1
    for key in (
        "loss", "loss_interior", "loss_left_neumann", "loss_topbot",
        "loss_ic", "loss_right_dir", "val_gnrmse", "val_rmse_K",
        "node_jump_rmse_K", "E_model", "E_zero",
    ):
        assert math.isfinite(float(rows[0][key])), key
    assert math.isfinite(summary["best_val_gnrmse"])

    checkpoint_path = run_dir / "cvit_best_global.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    spec = checkpoint["interface_forcing"]
    assert spec == {
        "representation": "space_time_image",
        "version": 1,
        "forcing_schema_version": 1,
        "axis_order": "channel_y_time",
        "dtype": "float32",
        "ny_img": 16,
        "nt_img": 32,
        "patch_size": 8,
        "include_endpoints": True,
        "y_min": 0.0,
        "y_max": 1.0,
        "t_min": 0.0,
        "t_final": float(np.float32(0.3)),
        "sign_convention": "positive_inward_left_flux",
        "normalization": "fixed_division",
        "a_ref": 300.0,
        "clipping": False,
        "ramp": {"type": "cubic_smoothstep", "version": 1, "duration": 0.01},
        "fv_dt": 0.005,
        "spatial_grid_size": [20, 20],
    }
    restored, _ = load_interface_cvit_checkpoint(checkpoint_path)
    assert restored.forcing_grid_size == (16, 32)
    with pytest.raises(FileExistsError, match="fresh experiment name"):
        run_one_seed_interfaces_pino(config, seed=0, run_dir=run_dir)


# --------- IC-first curriculum ---------

def test_curriculum_weights_schedule():
    cfg = {"enabled": True, "ic_only_epochs": 1, "ramp_epochs": 2}
    # phase 1: IC-only.
    assert _curriculum_weights(0, 1.0, 1.0, 1.0, cfg) == (0.0, 1.0, 0.0)
    # phase 2: ramp start (s=0) then linear.
    assert _curriculum_weights(1, 1.0, 1.0, 1.0, cfg) == (0.0, 1.0, 0.0)
    assert _curriculum_weights(2, 2.0, 1.0, 4.0, cfg) == (1.0, 1.0, 2.0)  # s=0.5
    # phase 3: full weights, clamped at s=1.
    assert _curriculum_weights(3, 2.0, 1.0, 4.0, cfg) == (2.0, 1.0, 4.0)
    assert _curriculum_weights(9, 2.0, 1.0, 4.0, cfg) == (2.0, 1.0, 4.0)


def test_curriculum_weights_disabled_is_noop():
    for cfg in ({}, None, {"enabled": False, "ic_only_epochs": 5}):
        assert _curriculum_weights(0, 0.5, 2.0, 3.0, cfg) == (0.5, 2.0, 3.0)


def test_curriculum_weights_ic_heavy_schedule():
    # IC-heavy: residual/BC stay ON (small) through warm-up; then lambda_ic decays
    # to lambda_ic_final while r/bc ramp to their targets. lambda_* = (1, 100, 1).
    cfg = {
        "enabled": True, "mode": "ic_heavy",
        "warmup_epochs": 2, "warmup_lambda_r": 0.1, "warmup_lambda_bc": 1.0,
        "decay_epochs": 2, "lambda_ic_final": 1.0,
    }
    # phase 1 (warm-up): residual/BC never zero, IC at the heavy weight.
    assert _curriculum_weights(0, 1.0, 100.0, 1.0, cfg) == (0.1, 100.0, 1.0)
    assert _curriculum_weights(1, 1.0, 100.0, 1.0, cfg) == (0.1, 100.0, 1.0)
    # phase 2 (decay, s=0 then s=0.5): IC relaxes 100 -> 1, r ramps 0.1 -> 1.
    assert _curriculum_weights(2, 1.0, 100.0, 1.0, cfg) == (0.1, 100.0, 1.0)
    w_r, w_ic, w_bc = _curriculum_weights(3, 1.0, 100.0, 1.0, cfg)  # s=0.5
    assert w_r == 0.55 and w_ic == 50.5 and w_bc == 1.0
    # phase 3 (clamped s=1): full targets with lambda_ic_final.
    assert _curriculum_weights(4, 1.0, 100.0, 1.0, cfg) == (1.0, 1.0, 1.0)
    assert _curriculum_weights(9, 1.0, 100.0, 1.0, cfg) == (1.0, 1.0, 1.0)


# --------- causal residual weighting ---------

def test_causal_weights_downweight_later_bins():
    # Increasing per-bin losses -> monotonically decreasing causal weights; the
    # first bin is always weight 1 (no prior loss). w_i = exp(-eps*sum_{j<i} L_j).
    bin_mean = torch.tensor([1.0, 2.0, 3.0, 4.0])
    w = _causal_weights(bin_mean, eps_causal=1.0)
    assert float(w[0]) == 1.0
    assert torch.all(w[1:] < w[:-1])
    # eps=0 disables the mask (all ones).
    w0 = _causal_weights(bin_mean, eps_causal=0.0)
    assert torch.allclose(w0, torch.ones_like(w0))


def test_causal_residual_loss_masks_late_violation():
    # r is (B, n_r, 1) (dim 0 = sim minibatch, averaged out); t_r is (1, n_r, 1).
    # Causal weighting only masks a late bin when the EARLIER bins still carry
    # substantial loss (large cumulative prefix). With O(1) early residuals and a
    # huge late residual, the late bin is suppressed and the reported loss tracks
    # the (small) early bins rather than the unweighted mean.
    t_final, n_bins = 1.0, 4
    t_r = torch.tensor([0.0, 0.3, 0.6, 1.0]).view(1, -1, 1)
    r = torch.tensor([1.0, 1.0, 1.0, 10.0]).view(1, -1, 1)
    loss, bin_mean = _causal_residual_loss(r, t_r, t_final, n_bins, eps_causal=5.0)
    plain = (r ** 2).mean()
    assert float(loss) < float(plain)
    assert float(loss) < 2.0  # dominated by the early O(1) bins, not the 100 outlier
    # the late bin's raw residual is still visible in the (detached) diagnostic.
    assert bin_mean.shape == (n_bins,)
    assert float(bin_mean[-1]) > float(bin_mean[0])


def test_causal_empty_bin_falls_back_to_pointwise_mse():
    t_r = torch.tensor([0.1, 0.9]).view(1, -1, 1)
    r = torch.tensor([1.0, 10.0]).view(1, -1, 1)
    loss, _ = _causal_residual_loss(r, t_r, 1.0, 4, eps_causal=5.0)
    assert torch.equal(loss, (r ** 2).mean())


def test_causal_bins_average_per_input_and_clamp_endpoints():
    t_r = torch.tensor([0.0, 0.2, 0.8, 1.0]).view(1, -1, 1)
    idx = _time_bin_index(t_r, 1.0, 2)
    assert idx.reshape(-1).tolist() == [0, 0, 1, 1]
    r = torch.tensor([
        [1.0, 3.0, 2.0, 4.0],
        [2.0, 4.0, 1.0, 3.0],
    ]).unsqueeze(-1)
    means, counts, pointwise = _causal_bin_stats(r, t_r, 1.0, 2)
    expected = torch.tensor([(5.0 + 10.0) / 2.0, (10.0 + 5.0) / 2.0])
    assert torch.allclose(means, expected)
    assert counts.tolist() == [2.0, 2.0]
    assert torch.allclose(pointwise, (r ** 2).mean())


def test_causal_config_alias_validation_and_adaptation():
    cfg = _resolve_forcing_causal({"enabled": True, "eps_causal": 0.1})
    assert cfg["initial_eps"] == pytest.approx(0.1)
    cfg2 = _resolve_forcing_causal({
        "enabled": True, "initial_eps": 0.2, "eps_causal": 0.1,
        "min_eps": 1e-3, "max_eps": 1.0, "step_size": 5.0,
    })
    assert cfg2["initial_eps"] == pytest.approx(0.2)
    increased, action = _adapt_causal_eps(
        0.2, torch.tensor([1.0, 1.0]), cfg2, populated=True,
    )
    assert (increased, action) == (1.0, "increase")
    decreased, action = _adapt_causal_eps(
        1e-3, torch.tensor([0.2, 0.0]), cfg2, populated=True,
    )
    assert (decreased, action) == (1e-3, "decrease")
    with pytest.raises(ValueError, match="step_size"):
        _resolve_forcing_causal({"step_size": 1.0})


def test_e2e_curriculum_ic_first(tmp_path):
    _write_synthetic_diffusion(tmp_path)
    cfg = _config(tmp_path)
    cfg["training"]["epochs"] = 4
    cfg["training"]["pino"]["curriculum"] = {
        "enabled": True, "ic_only_epochs": 1, "ramp_epochs": 2,
    }
    run_dir = tmp_path / "run_curr"

    run_one_seed_pino(cfg, seed=0, run_dir=run_dir)
    rows = _rows(run_dir)
    assert len(rows) == 4

    # IC-only epoch: residual/BC weights are zero, IC weight active.
    assert float(rows[0]["w_r"]) == 0.0
    assert float(rows[0]["w_bc"]) == 0.0
    assert float(rows[0]["w_ic"]) == 1.0
    # Ramp midpoint (epoch 2, s=0.5) and full weight (epoch 3).
    assert float(rows[2]["w_r"]) == 0.5
    assert float(rows[3]["w_r"]) == 1.0
    for r in rows:
        for col in ("loss", "loss_r", "loss_ic", "loss_bc"):
            assert math.isfinite(float(r[col]))


# --------- GradNorm balancing ---------

def test_e2e_gradnorm_enabled(tmp_path):
    _write_synthetic_diffusion(tmp_path)
    cfg = _config(tmp_path)
    cfg["training"]["epochs"] = 3
    cfg["training"]["gradnorm"] = {
        "enabled": True, "alpha_w": 0.5, "update_every": 1, "eps": 1e-8,
    }
    run_dir = tmp_path / "run_gn"

    run_one_seed_pino(cfg, seed=0, run_dir=run_dir)
    rows = _rows(run_dir)
    assert len(rows) == 3
    for r in rows:
        for col in ("loss", "loss_r", "loss_ic", "loss_bc", "w_r", "w_ic", "w_bc"):
            assert math.isfinite(float(r[col]))

    # GradNorm state is persisted with all three active terms.
    ckpt = torch.load(run_dir / "cvit_best.pt", map_location="cpu", weights_only=False)
    gn = ckpt.get("gradnorm_state")
    assert gn is not None
    assert set(gn["term_names"]) == {"r", "ic", "bc"}
    assert set(gn["multipliers"]) == {"r", "ic", "bc"}
    for v in gn["multipliers"].values():
        assert math.isfinite(float(v)) and float(v) > 0.0


# --------- dense IC + causal weighting + ic_heavy warm-up (staged pipeline) ---------

def test_e2e_staged_pipeline_and_diagnostics(tmp_path):
    _write_synthetic_diffusion(tmp_path)
    cfg = _config(tmp_path)
    cfg["training"]["epochs"] = 4
    cfg["training"]["validate_every"] = 1
    cfg["training"]["pino"]["dense_ic"] = True
    cfg["training"]["pino"]["resample_every"] = 2
    cfg["training"]["pino"]["curriculum"] = {
        "enabled": True, "mode": "ic_heavy",
        "warmup_epochs": 1, "warmup_lambda_r": 0.1, "warmup_lambda_bc": 1.0,
        "decay_epochs": 2, "lambda_ic_final": 1.0,
    }
    cfg["training"]["pino"]["lambda_ic"] = 100.0
    cfg["training"]["pino"]["causal"] = {
        "enabled": True, "n_bins": 4, "eps_causal": 1.0,
    }
    run_dir = tmp_path / "run_staged"

    run_one_seed_pino(cfg, seed=0, run_dir=run_dir)
    rows = _rows(run_dir)
    assert len(rows) == 4

    # ic_heavy warm-up: residual/BC stay ON (never zero), IC is the heavy weight.
    assert float(rows[0]["w_r"]) == 0.1
    assert float(rows[0]["w_ic"]) == 100.0
    assert float(rows[0]["w_bc"]) == 1.0
    for r in rows:
        for col in ("loss", "loss_r", "loss_ic", "loss_bc"):
            assert math.isfinite(float(r[col]))
    # grad-norm diagnostics logged on every (validate_every=1) epoch.
    for r in rows:
        for col in ("grad_norm_r", "grad_norm_ic", "grad_norm_bc"):
            assert r[col] not in ("", None)
            assert math.isfinite(float(r[col]))

    # per-time diagnostics written to their own CSVs, one row per val epoch.
    relt = _named_rows(run_dir, "rel_l2_per_time.csv")
    assert len(relt) == 4
    assert "t0" in relt[0] and "t5" in relt[0]  # Nt = 6
    for row in relt:
        for k, v in row.items():
            assert math.isfinite(float(v))
    resbin = _named_rows(run_dir, "residual_per_time_bin.csv")
    assert len(resbin) == 4
    assert set(f"bin{b}" for b in range(4)).issubset(resbin[0].keys())
    for row in resbin:
        for k, v in row.items():
            assert math.isfinite(float(v))


# --------- relative IC loss ---------

def test_ic_loss_mse_is_mean_square():
    # Legacy path: mode="mse" is exactly mean(ic**2), independent of the target.
    ic = torch.tensor([[[0.1], [0.2], [0.3]]])          # (1, 3, 1)
    ic_target = torch.tensor([[[5.0], [-4.0], [2.0]]])
    got = _ic_loss(ic, ic_target, mode="mse")
    assert torch.allclose(got, (ic ** 2).mean())


def test_ic_loss_rel_is_squared_relative_ratio():
    # rel: sum_i ic**2 / (sum_i (ic_target - t_right_tilde)**2 + eps). With
    # t_right_tilde=0 and ic = 0.1 * ic_target, the ratio is 0.1**2 = 0.01.
    ic_target = torch.tensor([[[2.0], [-3.0], [1.5]]])   # (1, 3, 1)
    ic = 0.1 * ic_target
    got = _ic_loss(ic, ic_target, mode="rel", t_right_tilde=0.0, eps=0.0)
    assert torch.allclose(got, torch.tensor(0.01), atol=1e-6)


def test_ic_loss_rel_is_per_sim_amplitude_invariant():
    # Two sims, same 10% relative IC error but different amplitudes; per-sim
    # normalization makes each ratio 0.01, so the mean is 0.01 (a raw MSE would be
    # dominated by the larger-amplitude sim).
    ic_target = torch.tensor([
        [[1.0], [1.0]],       # small-amplitude sim
        [[100.0], [100.0]],   # large-amplitude sim
    ])                                                   # (2, 2, 1)
    ic = 0.1 * ic_target
    got = _ic_loss(ic, ic_target, mode="rel", t_right_tilde=0.0, eps=0.0)
    assert torch.allclose(got, torch.tensor(0.01), atol=1e-6)


def test_ic_loss_rel_uses_deviation_from_t_right():
    # The denominator measures the IC as a deviation from the constant field
    # t_right_tilde, so an IC equal to t_right_tilde has ~zero energy and the eps
    # guard keeps the ratio finite.
    t_right = 2.5
    ic_target = torch.full((1, 4, 1), t_right)
    ic = torch.full((1, 4, 1), 0.3)
    got = _ic_loss(ic, ic_target, mode="rel", t_right_tilde=t_right, eps=1.0e-6)
    assert math.isfinite(float(got))
    # num = 4 * 0.09 = 0.36; den = 0 + 1e-6 -> large but finite ratio.
    assert float(got) > 1.0


def test_ic_loss_rel_fullgrid_denom_is_per_point_mean_ratio():
    # Stabilized path: passing an explicit per-sim denominator makes the numerator
    # a per-point MEAN square (not a sum), so the loss is mean(ic**2)/den. ic=0.1
    # everywhere, den=1.0 -> 0.01. The sampled ic_target is ignored on this path.
    ic = torch.full((1, 4, 1), 0.1)
    ic_target = torch.zeros((1, 4, 1))
    den = torch.tensor([1.0])
    got = _ic_loss(ic, ic_target, mode="rel", den=den)
    assert torch.allclose(got, torch.tensor(0.01), atol=1e-6)


def test_ic_loss_rel_fullgrid_denom_is_n_ic_independent():
    # The whole point of the mean-based numerator: the loss is invariant to how
    # many collocation nodes were sampled. The legacy sum-based sampled path would
    # scale linearly with the node count.
    den = torch.tensor([2.0])
    ic4 = torch.full((1, 4, 1), 0.3)
    ic8 = torch.full((1, 8, 1), 0.3)
    g4 = _ic_loss(ic4, torch.zeros_like(ic4), mode="rel", den=den)
    g8 = _ic_loss(ic8, torch.zeros_like(ic8), mode="rel", den=den)
    assert torch.allclose(g4, g8, atol=1e-6)
    assert torch.allclose(g4, torch.tensor(0.045), atol=1e-6)   # 0.09 / 2.0


def test_ic_loss_rel_fullgrid_denom_bounded_by_floor():
    # A floored (larger) denominator caps the ratio: the near-flat-sim explosion
    # the sampled path suffers cannot happen once den >= floor.
    ic = torch.full((1, 4, 1), 0.5)
    tgt = torch.zeros_like(ic)
    unfloored = _ic_loss(ic, tgt, mode="rel", den=torch.tensor([1.0e-4]))
    floored = _ic_loss(ic, tgt, mode="rel", den=torch.tensor([1.0]))
    assert float(floored) < float(unfloored)
    assert math.isfinite(float(floored))


def test_e2e_ic_only_raw_mse_skips_residual(tmp_path):
    # IC-only diagnostic A: lambda_r=0, lambda_bc=0, dense IC, AdamW flat LR.
    # compute_r/compute_bc are off, so loss_r/loss_bc are exact zeros and no
    # residual double-backward runs.
    _write_synthetic_diffusion(tmp_path)
    cfg = _config(tmp_path)
    cfg["training"]["epochs"] = 2
    cfg["training"]["validate_every"] = 1
    cfg["training"]["optimizer"] = "AdamW"
    cfg["training"]["learning_rate"] = 3.0e-4
    cfg["training"]["scheduler"] = {"type": "StepLR", "step_size": 1, "gamma": 1.0}
    cfg["training"]["pino"]["lambda_r"] = 0.0
    cfg["training"]["pino"]["lambda_bc"] = 0.0
    cfg["training"]["pino"]["ic_loss"] = "mse"
    cfg["training"]["pino"]["dense_ic"] = True
    run_dir = tmp_path / "run_ic_mse"

    run_one_seed_pino(cfg, seed=0, run_dir=run_dir)
    rows = _rows(run_dir)
    assert len(rows) == 2
    for r in rows:
        assert float(r["loss_r"]) == 0.0
        assert float(r["loss_bc"]) == 0.0
        assert math.isfinite(float(r["loss_ic"]))
        assert float(r["loss_ic"]) > 0.0


def test_e2e_ic_only_fullgrid_rel_is_stable(tmp_path):
    # IC-only diagnostic B: stabilized relative loss with the fixed full-grid
    # per-sim denominator. Asserts the loss stays finite/bounded (no sampled-denom
    # explosion) and the residual is skipped.
    _write_synthetic_diffusion(tmp_path)
    cfg = _config(tmp_path)
    cfg["training"]["epochs"] = 2
    cfg["training"]["validate_every"] = 1
    cfg["training"]["optimizer"] = "AdamW"
    cfg["training"]["learning_rate"] = 3.0e-4
    cfg["training"]["scheduler"] = {"type": "StepLR", "step_size": 1, "gamma": 1.0}
    cfg["training"]["pino"]["lambda_r"] = 0.0
    cfg["training"]["pino"]["lambda_bc"] = 0.0
    cfg["training"]["pino"]["ic_loss"] = "rel"
    cfg["training"]["pino"]["ic_denom"] = "full_grid"
    cfg["training"]["pino"]["ic_denom_floor_frac"] = 0.01
    cfg["training"]["pino"]["dense_ic"] = True
    run_dir = tmp_path / "run_ic_rel_fg"

    run_one_seed_pino(cfg, seed=0, run_dir=run_dir)
    rows = _rows(run_dir)
    assert len(rows) == 2
    for r in rows:
        assert float(r["loss_r"]) == 0.0
        assert float(r["loss_bc"]) == 0.0
        assert math.isfinite(float(r["loss_ic"]))
        # A relative IC loss on a per-sim-normalized denominator starts O(1); the
        # floor guarantees it cannot blow past a small multiple of that.
        assert float(r["loss_ic"]) < 100.0


def test_e2e_ic_loss_rel_runs_and_reports_bands(tmp_path):
    _write_synthetic_diffusion(tmp_path)
    cfg = _config(tmp_path)
    cfg["training"]["epochs"] = 2
    cfg["training"]["validate_every"] = 1
    cfg["training"]["pino"]["ic_loss"] = "rel"
    cfg["training"]["pino"]["ic_eps"] = 1.0e-6
    run_dir = tmp_path / "run_ic_rel"

    run_one_seed_pino(cfg, seed=0, run_dir=run_dir)
    rows = _rows(run_dir)
    assert len(rows) == 2

    band_cols = [
        f"{stem}_{band}"
        for stem in ("rel_l2", "rel_dev", "rmse_K")
        for band in ("early", "mid", "late")
    ]
    for r in rows:
        for col in ("loss", "loss_r", "loss_ic", "loss_bc"):
            assert math.isfinite(float(r[col]))
        # new honest-metric columns populated on every (validate_every=1) epoch
        assert r["val_t0_rel_l2"] not in ("", None)
        assert math.isfinite(float(r["val_t0_rel_l2"]))
        for col in band_cols:
            assert col in r and r[col] not in ("", None)
            assert math.isfinite(float(r[col]))

    # companion per-time CSVs for rel-dev and RMSE_K, one row per val epoch, Nt=6.
    for name in ("rel_l2_dev_per_time.csv", "rmse_K_per_time.csv"):
        prt = _named_rows(run_dir, name)
        assert len(prt) == 2
        assert "t0" in prt[0] and "t5" in prt[0]
        for row in prt:
            for _, v in row.items():
                assert math.isfinite(float(v))


# --------- forcing PINO trainer: split-BC GradNorm ---------

def _write_synthetic_forcing(tmp_path, num_sims=8, Nt=6, Nx=10, Ny=16):
    # ForcingCViT trains physics-only on ONLINE-sampled forcing; the saved FV
    # trajectories + sim_params are VALIDATION-only (deviation-field gnRMSE).
    rng = np.random.default_rng(0)
    traj = (300.0 + 5.0 * rng.standard_normal((num_sims, Nt, Nx, Ny))).astype(
        np.float32
    )
    x_grid = np.linspace(0.0, 1.0, Nx).astype(np.float32)
    y_grid = np.linspace(0.0, 1.0, Ny).astype(np.float32)
    t_grid = np.linspace(0.0, 0.3, Nt).astype(np.float32)
    # sim_params.npy MUST live beside trajectories.npy (the trainer loads it there).
    records = sample_forcing_params(rng, num_sims, dt=0.3 / (Nt - 1), t_final=0.3)
    sim_params = np.array(records, dtype=object)
    np.save(tmp_path / "trajectories.npy", traj)
    np.save(tmp_path / "x_grid.npy", x_grid)
    np.save(tmp_path / "y_grid.npy", y_grid)
    np.save(tmp_path / "t_grid.npy", t_grid)
    np.save(tmp_path / "sim_params.npy", sim_params)
    return traj


def test_sample_forcing_params_pins_sin_uniform():
    # diffusion_forcing_single pins the online sampler to sin/uniform; every
    # drawn record must use that single family.
    rng = np.random.default_rng(0)
    records = sample_forcing_params(
        rng, 32, dt=0.3 / 5, t_final=0.3,
        temporal_family="sin", spatial_family="uniform",
    )
    assert len(records) == 32
    for r in records:
        assert r["temporal_family"] == "sin"
        assert r["spatial_family"] == "uniform"


def test_sample_forcing_params_default_path_is_deterministic():
    # Adding the family kwargs must not perturb the default (all-family) RNG
    # consumption: two freshly-seeded draws with default kwargs match exactly.
    def _draw():
        rng = np.random.default_rng(1234)
        return sample_forcing_params(rng, 16, dt=0.3 / 5, t_final=0.3)

    a, b = _draw(), _draw()
    assert [r["temporal_family"] for r in a] == [r["temporal_family"] for r in b]
    assert [r["spatial_family"] for r in a] == [r["spatial_family"] for r in b]
    # Same seed => byte-identical param sequence; repr captures floats/arrays alike.
    assert repr(a) == repr(b)


def test_sample_forcing_params_default_path_draws_multiple_families():
    # Default (unpinned) sampling still spans more than one temporal family, so
    # the pin above is a real restriction rather than a no-op.
    rng = np.random.default_rng(7)
    records = sample_forcing_params(rng, 64, dt=0.3 / 5, t_final=0.3)
    families = {r["temporal_family"] for r in records}
    assert len(families) > 1


def _forcing_config(tmp_path):
    # Forcing image grid = (ny_img=Ny=16, nt_img=20); patch_size=4 divides both.
    return {
        "benchmark": {"name": "diffusion_forcing"},
        "data": {
            "trajectories.npy": str(tmp_path / "trajectories.npy"),
            "x_grid_path": str(tmp_path / "x_grid.npy"),
            "y_grid_path": str(tmp_path / "y_grid.npy"),
            "t_grid_path": str(tmp_path / "t_grid.npy"),
        },
        "model": {
            "cvit": {
                "in_ch": 1,
                "out_dim": 1,
                "emb_dim": 32,
                "dec_emb_dim": None,
                "patch_size": 4,
                "depth_enc": 1,
                "depth_dec": 1,
                "num_heads": 4,
                "mlp_ratio": 2.0,
                "fourier_freq": 1.0,
                "activation": "gelu",
                "hard_right_dirichlet": True,
                "hard_right_dirichlet_t_right": 300.0,
                "hard_left_flux": True,
            }
        },
        "training": {
            "device": "cpu",
            "epochs": 2,
            "validate_every": 1,
            "learning_rate": 5.0e-4,
            "weight_decay": 1e-5,
            "optimizer": "SOAP",
            "soap": {
                "betas": [0.95, 0.95],
                "shampoo_beta": 0.95,
                "eps": 1.0e-8,
                "precondition_frequency": 3,
                "max_precond_dim": 10000,
                "merge_dims": False,
                "precondition_1d": False,
            },
            "scheduler": {
                "type": "PICViTExponential",
                "decay_every": 500,
                "decay_rate": 0.95,
                "min_lr": 1.0e-5,
            },
            "pino": {
                "lambda_r": 1.0,
                "lambda_ic": 1.0,
                "lambda_bc": 1.0,
                "lambda_bc_left": 6.0,
                "n_r": 64,
                "n_ic": 32,
                "n_bc": 16,
                "sim_batch": 4,
                "alpha": 1.0,
                "forcing": {"nt_img": 20, "ramp_seconds": 0.003},
            },
        },
    }


def test_e2e_forcing_gradnorm_enabled(tmp_path):
    _write_synthetic_forcing(tmp_path)
    cfg = _forcing_config(tmp_path)
    cfg["training"]["gradnorm"] = {
        "enabled": True, "alpha_w": 0.5, "update_every": 1, "eps": 1e-8,
        "w_min": 0.1, "w_max": 5.0, "floor": {"bc_left": 0.25},
    }
    run_dir = tmp_path / "run_forcing_gn"

    run_one_seed_forcing_pino(cfg, seed=0, run_dir=run_dir)
    rows = _rows(run_dir)
    assert len(rows) == 2

    gn_cols = ["r", "ic", "bc_left", "bc_hom"]
    for r in rows:
        for col in ("loss", "loss_r", "loss_ic", "loss_bc", "loss_bc_left"):
            assert math.isfinite(float(r[col]))
        # Static weight columns keep their STATIC meaning (not the multiplier).
        assert float(r["w_r"]) == 1.0
        assert float(r["w_bc_left"]) == 6.0
        # validate_every=1 -> GradNorm diagnostics on every epoch.
        for c in gn_cols:
            assert math.isfinite(float(r[f"gn_mult_{c}"]))
            assert float(r[f"gn_mult_{c}"]) > 0.0
            assert math.isfinite(float(r[f"w_eff_{c}"]))
        # Raw / effective grad norms populated for the terms that carry gradient
        # (a held near-zero term logs empty, not NaN -- allowed here).
        for c in gn_cols:
            for stem in (f"grad_norm_{c}", f"grad_norm_eff_{c}"):
                if r[stem] not in ("", None):
                    assert math.isfinite(float(r[stem]))
        assert r["grad_norm_r"] not in ("", None)
        assert r["grad_norm_bc_left"] not in ("", None)
        # Effective left weight respects the multiplier floor: w_eff_bc_left =
        # lambda_bc_left * m_bc_left >= 6.0 * 0.25 = 1.5.
        assert float(r["w_eff_bc_left"]) >= 6.0 * 0.25 - 1e-6
        # JSON supplements parse and carry the split term set.
        weights = json.loads(r["gradnorm_weights"])
        assert set(weights) == {"r", "ic", "bc_left", "bc_hom"}
        cos = json.loads(r["grad_cosines"])
        assert set(cos) == {"r|bc_left", "ic|bc_left", "bc_hom|bc_left"}
        for v in cos.values():
            assert v is None or math.isfinite(float(v))
        hits = json.loads(r["gradnorm_bound_hits"])
        assert set(hits) == {"r", "ic", "bc_left", "bc_hom"}

    # GradNorm state persists the EXACT ordered split term set.
    ckpt = torch.load(run_dir / "cvit_best.pt", map_location="cpu", weights_only=False)
    gn = ckpt.get("gradnorm_state")
    assert gn is not None
    assert gn["term_names"] == ["r", "ic", "bc_left", "bc_hom"]
    assert set(gn["multipliers"]) == {"r", "ic", "bc_left", "bc_hom"}
    for v in gn["multipliers"].values():
        assert math.isfinite(float(v)) and float(v) > 0.0
        # Restored multipliers sit inside the enabled clamp [0.1, 5.0].
        assert 0.1 - 1e-6 <= float(v) <= 5.0 + 1e-6


def test_e2e_forcing_gradnorm_disabled_is_noop(tmp_path):
    _write_synthetic_forcing(tmp_path)
    cfg = _forcing_config(tmp_path)
    run_dir = tmp_path / "run_forcing_nogn"

    run_one_seed_forcing_pino(cfg, seed=0, run_dir=run_dir)
    rows = _rows(run_dir)
    assert len(rows) == 2

    assert float(rows[0]["lr_first"]) == pytest.approx(5.0e-4)
    assert float(rows[0]["lr_last"]) == pytest.approx(5.0e-4)
    expected_second_lr = 5.0e-4 * 0.95 ** (1.0 / 500.0)
    assert float(rows[1]["lr_first"]) == pytest.approx(expected_second_lr)
    assert float(rows[1]["lr_last"]) == pytest.approx(expected_second_lr)

    for r in rows:
        for col in ("loss", "loss_r", "loss_ic", "loss_bc", "loss_bc_left"):
            assert math.isfinite(float(r[col]))
        # Static weights unchanged; GradNorm columns stay empty when disabled.
        assert float(r["w_r"]) == 1.0
        assert float(r["w_ic"]) == 1.0
        assert float(r["w_bc"]) == 1.0
        assert float(r["w_bc_left"]) == 6.0
        for c in ("r", "ic", "bc_left", "bc_hom"):
            assert r[f"gn_mult_{c}"] in ("", None)
            assert r[f"w_eff_{c}"] in ("", None)
        assert r["gradnorm_weights"] in ("", None)
        assert r["grad_cosines"] in ("", None)

    ckpt = torch.load(run_dir / "cvit_best.pt", map_location="cpu", weights_only=False)
    assert ckpt["config"]["training"]["optimizer"] == "SOAP"
    assert ckpt.get("gradnorm_state") is None


def test_e2e_forcing_causal_warmup_and_completion_state(tmp_path):
    _write_synthetic_forcing(tmp_path)
    cfg = _forcing_config(tmp_path)
    cfg["training"]["pino"]["causal"] = {
        "enabled": True, "n_bins": 4, "initial_eps": 1e-2,
        "eps_causal": None, "min_eps": 1e-12, "max_eps": 100.0,
        "step_size": 5.0, "min_mean_weight": 0.4, "max_min_weight": 0.99,
    }
    cfg["training"]["pino"]["forcing"].update({
        "save_latest_every": 1,
        "warmup": {
            "steps": 1, "epochs": 0, "resample_every": 25,
            "r_mult": 2.0, "ic_mult": 1.0,
            "bc_left_mult": 2.0, "bc_hom_mult": 1.0,
        },
    })
    run_dir = tmp_path / "run_forcing_causal"
    run_one_seed_forcing_pino(cfg, seed=0, run_dir=run_dir)
    rows = _rows(run_dir)
    assert float(rows[0]["warm_mult_r"]) == 2.0
    assert float(rows[0]["warm_mult_bc_left"]) == 2.0
    assert float(rows[1]["warm_mult_r"]) == 1.0
    for row in rows:
        assert row["causal_adaptation_action"] in {"increase", "decrease", "hold"}
        assert math.isfinite(float(row["causal_log_last_weight"]))
        for i in range(4):
            assert float(row[f"causal_count_bin_{i:02d}"]) > 0
    assert (run_dir / "cvit_latest.pt").exists()
    assert (run_dir / "cvit_best.pt").exists()
    assert (run_dir / "cvit_final.pt").exists()
    assert (run_dir / "RUN_COMPLETE").exists()
    ckpt = torch.load(run_dir / "cvit_latest.pt", map_location="cpu", weights_only=False)
    assert ckpt["completed_updates"] == 2
    assert ckpt["causal_state"]["n_bins"] == 4
    assert "rng_state" in ckpt and "forcing_cache" in ckpt


def test_e2e_forcing_resume_matches_uninterrupted(tmp_path):
    _write_synthetic_forcing(tmp_path)
    full_cfg = _forcing_config(tmp_path)
    full_cfg["training"]["epochs"] = 4
    full_cfg["training"]["pino"]["forcing"]["save_latest_every"] = 1
    full_dir = tmp_path / "full"
    run_one_seed_forcing_pino(full_cfg, seed=4, run_dir=full_dir)

    split_cfg = _forcing_config(tmp_path)
    split_cfg["training"]["pino"]["forcing"]["save_latest_every"] = 1
    split_dir = tmp_path / "split"
    run_one_seed_forcing_pino(split_cfg, seed=4, run_dir=split_dir)
    (split_dir / "RUN_COMPLETE").unlink()
    split_cfg["training"]["epochs"] = 4
    run_one_seed_forcing_pino(split_cfg, seed=4, run_dir=split_dir)

    full = torch.load(full_dir / "cvit_final.pt", map_location="cpu", weights_only=False)
    split = torch.load(split_dir / "cvit_final.pt", map_location="cpu", weights_only=False)
    assert full["completed_updates"] == split["completed_updates"] == 4
    for key, value in full["model_state"].items():
        assert torch.equal(value, split["model_state"][key]), key
    assert full["scheduler_state"] == split["scheduler_state"]
    assert full["rng_state"]["numpy_local"] == split["rng_state"]["numpy_local"]


def test_e2e_forcing_explicit_disabled_options_match_legacy(tmp_path):
    _write_synthetic_forcing(tmp_path)
    legacy_cfg = _forcing_config(tmp_path)
    explicit_cfg = _forcing_config(tmp_path)
    explicit_cfg["training"]["pino"]["causal"] = {"enabled": False}
    explicit_cfg["training"]["pino"]["forcing"].update({
        "grad_clip": None,
        "warmup": {
            "steps": None, "epochs": 0, "resample_every": 1,
            "r_mult": 1.0, "ic_mult": 1.0,
            "bc_left_mult": 1.0, "bc_hom_mult": 1.0,
        },
    })
    run_one_seed_forcing_pino(legacy_cfg, seed=8, run_dir=tmp_path / "legacy")
    run_one_seed_forcing_pino(explicit_cfg, seed=8, run_dir=tmp_path / "explicit")
    legacy = torch.load(tmp_path / "legacy/cvit_final.pt", map_location="cpu", weights_only=False)
    explicit = torch.load(tmp_path / "explicit/cvit_final.pt", map_location="cpu", weights_only=False)
    for key, value in legacy["model_state"].items():
        assert torch.equal(value, explicit["model_state"][key]), key
    assert legacy["optimizer_state"].keys() == explicit["optimizer_state"].keys()
    assert legacy["scheduler_state"] == explicit["scheduler_state"]
    assert torch.equal(
        legacy["rng_state"]["train_generator"],
        explicit["rng_state"]["train_generator"],
    )
    assert legacy["rng_state"]["numpy_local"] == explicit["rng_state"]["numpy_local"]


def test_e2e_forcing_nonfinite_loss_does_not_commit(tmp_path, monkeypatch):
    _write_synthetic_forcing(tmp_path)
    cfg = _forcing_config(tmp_path)
    original = train_pino_mod.pino_losses

    def nonfinite_losses(*args, **kwargs):
        out = original(*args, **kwargs)
        out["r"] = out["r"] * torch.tensor(float("nan"))
        return out

    monkeypatch.setattr(train_pino_mod, "pino_losses", nonfinite_losses)
    run_dir = tmp_path / "nonfinite"
    with pytest.raises(FloatingPointError, match="Non-finite forcing PINO loss"):
        run_one_seed_forcing_pino(cfg, seed=0, run_dir=run_dir)
    assert not (run_dir / "cvit_latest.pt").exists()
    assert not (run_dir / "RUN_COMPLETE").exists()
    assert _rows(run_dir) == []
