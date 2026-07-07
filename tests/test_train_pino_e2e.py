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

from src.operators.train_pino import _curriculum_weights, run_one_seed_pino


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
