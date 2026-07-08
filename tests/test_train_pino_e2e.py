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
import torch

from src.operators.train_pino import (
    _causal_residual_loss,
    _causal_weights,
    _curriculum_weights,
    _ic_loss,
    run_one_seed_pino,
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
    t_r = torch.tensor([0.1, 0.4, 0.9]).view(1, -1, 1)
    r = torch.tensor([1.0, 1.0, 10.0]).view(1, -1, 1)
    loss, bin_mean = _causal_residual_loss(r, t_r, t_final, n_bins, eps_causal=5.0)
    plain = (r ** 2).mean()
    assert float(loss) < float(plain)
    assert float(loss) < 2.0  # dominated by the early O(1) bins, not the 100 outlier
    # the late bin's raw residual is still visible in the (detached) diagnostic.
    assert bin_mean.shape == (n_bins,)
    assert float(bin_mean[-1]) > float(bin_mean[0])


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
