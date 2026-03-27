import csv
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.operators.fno1d import FNO1d
from src.operators.losses import SpatiallyWeightedMSE, build_interface_mask

from src.operators.train import (
    load_config,
    set_seed,
    train_one_epoch,
    validate,
    _is_training_complete,
    _load_completed_result,
    _rigno_three_phase_lr_for_epoch,
    _resolve_rigno_lr_values,
    _resolve_rigno_phase_lengths,
    build_optimizer,
    build_scheduler,
    run_one_seed,
    run_config_seeds,
)


# ===================== set_seed =====================

class TestSetSeed:
    def test_torch_determinism(self):
        set_seed(42)
        a = torch.randn(5)
        set_seed(42)
        b = torch.randn(5)
        assert torch.equal(a, b)

    def test_numpy_determinism(self):
        set_seed(42)
        a = np.random.rand(5)
        set_seed(42)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)

    def test_python_random_determinism(self):
        set_seed(42)
        a = random.random()
        set_seed(42)
        b = random.random()
        assert a == b

    def test_different_seeds_differ(self):
        set_seed(0)
        a = torch.randn(5)
        set_seed(1)
        b = torch.randn(5)
        assert not torch.equal(a, b)


# ===================== load_config =====================

class TestLoadConfig:
    def test_returns_dict(self):
        cfg = load_config()
        assert isinstance(cfg, dict)

    def test_has_required_keys(self):
        cfg = load_config()
        assert "model" in cfg
        assert "training" in cfg
        assert "data" in cfg

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")

    def test_resolves_variables(self):
        cfg = load_config()
        # paths should be resolved, not contain ${...}
        traj_path = cfg["data"]["trajectories.npy"]
        assert "${" not in traj_path


# ===================== helpers for training tests =====================

def _make_4tuple_loader(Nx=11, n_samples=4, batch_size=2):
    """Create a DataLoader yielding (x_spatial, cond, Y, T_stats) 4-tuples."""
    x_spatial = torch.randn(n_samples, Nx, 2)
    cond = torch.rand(n_samples, 7)
    Y = torch.randn(n_samples, Nx, 1)
    T_stats = torch.randn(n_samples, 2)
    return DataLoader(TensorDataset(x_spatial, cond, Y, T_stats), batch_size=batch_size)


@pytest.fixture
def tiny_training_setup():
    """Tiny model + synthetic 4-tuple dataloader for fast training tests."""
    Nx = 11
    model = FNO1d(modes=2, width=8, in_channels=2, out_channels=1, n_layers=2, cond_dim=7)
    loader = _make_4tuple_loader(Nx=Nx)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    x_grid_np = np.linspace(0.0, 1.0, Nx).astype(np.float32)
    loss_fn = SpatiallyWeightedMSE(x_grid_np, interface_weight=1.0)
    iface_mask = build_interface_mask(x_grid_np)
    device = torch.device("cpu")

    return model, loader, optimizer, loss_fn, device, iface_mask


# ===================== train_one_epoch =====================

class TestTrainOneEpoch:
    def test_returns_tuple_of_floats(self, tiny_training_setup):
        model, loader, optimizer, loss_fn, device, iface_mask = tiny_training_setup
        result = train_one_epoch(model, loader, optimizer, loss_fn, device, iface_mask=iface_mask)
        assert isinstance(result, tuple) and len(result) == 3
        loss, rel_l2, iface_rel_l2 = result
        assert isinstance(loss, float)
        assert isinstance(rel_l2, float)
        assert isinstance(iface_rel_l2, float)

    def test_loss_is_finite(self, tiny_training_setup):
        model, loader, optimizer, loss_fn, device, iface_mask = tiny_training_setup
        loss, rel_l2, iface_rel_l2 = train_one_epoch(model, loader, optimizer, loss_fn, device, iface_mask=iface_mask)
        assert np.isfinite(loss)
        assert np.isfinite(rel_l2)
        assert np.isfinite(iface_rel_l2)

    def test_loss_is_nonnegative(self, tiny_training_setup):
        model, loader, optimizer, loss_fn, device, iface_mask = tiny_training_setup
        loss, rel_l2, iface_rel_l2 = train_one_epoch(model, loader, optimizer, loss_fn, device, iface_mask=iface_mask)
        assert loss >= 0
        assert rel_l2 >= 0
        assert iface_rel_l2 >= 0

    def test_updates_parameters(self, tiny_training_setup):
        model, loader, optimizer, loss_fn, device, iface_mask = tiny_training_setup
        params_before = {n: p.clone() for n, p in model.named_parameters()}
        train_one_epoch(model, loader, optimizer, loss_fn, device, iface_mask=iface_mask)
        changed = any(
            not torch.equal(params_before[n], p)
            for n, p in model.named_parameters()
        )
        assert changed

    def test_no_iface_mask_returns_zero_iface_metric(self, tiny_training_setup):
        model, loader, optimizer, loss_fn, device, _ = tiny_training_setup
        _, _, iface_rel_l2 = train_one_epoch(model, loader, optimizer, loss_fn, device)
        assert iface_rel_l2 == 0.0

    def test_grad_clip_runs_without_error(self, tiny_training_setup):
        model, loader, optimizer, loss_fn, device, iface_mask = tiny_training_setup
        loss, _, _ = train_one_epoch(
            model, loader, optimizer, loss_fn, device,
            iface_mask=iface_mask, grad_clip=0.01,
        )
        assert np.isfinite(loss)

    def test_grad_clip_none_is_noop(self, tiny_training_setup):
        model, loader, optimizer, loss_fn, device, iface_mask = tiny_training_setup
        loss, _, _ = train_one_epoch(
            model, loader, optimizer, loss_fn, device,
            iface_mask=iface_mask, grad_clip=None,
        )
        assert np.isfinite(loss)


# ===================== validate =====================

class TestValidate:
    def test_returns_tuple(self, tiny_training_setup):
        model, loader, _, _, device, iface_mask = tiny_training_setup
        result = validate(model, loader, device, iface_mask=iface_mask)
        assert isinstance(result, tuple) and len(result) == 2
        val_rel_l2, val_iface = result
        assert isinstance(val_rel_l2, float)
        assert isinstance(val_iface, float)

    def test_loss_is_nonnegative(self, tiny_training_setup):
        model, loader, _, _, device, iface_mask = tiny_training_setup
        val_rel_l2, val_iface = validate(model, loader, device, iface_mask=iface_mask)
        assert val_rel_l2 >= 0
        assert val_iface >= 0

    def test_no_gradient_accumulation(self, tiny_training_setup):
        model, loader, _, _, device, iface_mask = tiny_training_setup
        model.zero_grad()
        validate(model, loader, device, iface_mask=iface_mask)
        for p in model.parameters():
            assert p.grad is None or torch.all(p.grad == 0)

    def test_does_not_change_parameters(self, tiny_training_setup):
        model, loader, _, _, device, iface_mask = tiny_training_setup
        params_before = {n: p.clone() for n, p in model.named_parameters()}
        validate(model, loader, device, iface_mask=iface_mask)
        for n, p in model.named_parameters():
            assert torch.equal(params_before[n], p)


# ===================== RIGNO scheduler =====================

class TestRIGNOThreePhaseSchedule:
    def test_matches_requested_sentinel_epochs(self):
        kwargs = {
            "warmup_epochs": 30,
            "cosine_epochs": 1320,
            "exp_epochs": 150,
            "init_lr": 1.0e-4,
            "peak_lr": 2.0e-3,
            "cosine_floor_lr": 1.0e-4,
            "final_lr": 1.0e-5,
        }
        expected = {
            1: 1.0e-4,
            2: 1.65517241379e-4,
            15: 1.01724137931e-3,
            30: 2.0e-3,
            31: 2.0e-3,
            100: 1.98719957948e-3,
            500: 1.46640739533e-3,
            1350: 1.0e-4,
            1351: 1.0e-4,
            1450: 2.16556124063e-5,
            1500: 1.0e-5,
        }

        for human_epoch, target_lr in expected.items():
            lr = _rigno_three_phase_lr_for_epoch(human_epoch - 1, **kwargs)
            assert lr == pytest.approx(target_lr, rel=1e-10, abs=1e-12)

    def test_fraction_based_phase_lengths_match_default_schedule(self):
        training_cfg = {"epochs": 1500}
        sched_cfg = {
            "warmup_fraction": 0.02,
            "cosine_fraction": 0.88,
            "exp_fraction": 0.10,
        }

        assert _resolve_rigno_phase_lengths(training_cfg, sched_cfg) == (30, 1320, 150)

    def test_fraction_based_phase_lengths_support_epoch_override(self):
        training_cfg = {"epochs": 900}
        sched_cfg = {
            "warmup_fraction": 0.02,
            "cosine_fraction": 0.88,
            "exp_fraction": 0.10,
        }

        assert _resolve_rigno_phase_lengths(training_cfg, sched_cfg) == (18, 792, 90)

    @pytest.mark.parametrize(
        ("learning_rate", "expected"),
        [
            (2.0e-3, (2.0e-3, 1.0e-4, 1.0e-4, 1.0e-5)),
            (1.0e-3, (1.0e-3, 5.0e-5, 5.0e-5, 5.0e-6)),
        ],
    )
    def test_ratio_based_lr_values_track_learning_rate(self, learning_rate, expected):
        training_cfg = {"learning_rate": learning_rate}
        sched_cfg = {
            "init_lr_ratio": 0.05,
            "cosine_floor_lr_ratio": 0.05,
            "final_lr_ratio": 0.005,
        }

        assert _resolve_rigno_lr_values(training_cfg, sched_cfg) == pytest.approx(expected)

    def test_fraction_ratio_scheduler_accepts_swept_lr_and_epochs(self):
        config = {
            "training": {
                "learning_rate": 1.0e-3,
                "weight_decay": 1.0e-5,
                "epochs": 900,
                "optimizer": "AdamW",
                "scheduler": {
                    "type": "RIGNOThreePhase",
                    "warmup_fraction": 0.02,
                    "cosine_fraction": 0.88,
                    "exp_fraction": 0.10,
                    "init_lr_ratio": 0.05,
                    "cosine_floor_lr_ratio": 0.05,
                    "final_lr_ratio": 0.005,
                },
            }
        }
        model = FNO1d(modes=2, width=8, in_channels=2, out_channels=1, n_layers=2, cond_dim=7)
        optimizer = build_optimizer(config, model.parameters())

        scheduler = build_scheduler(config, optimizer)

        assert scheduler is not None
        assert optimizer.param_groups[0]["lr"] == pytest.approx(5.0e-5)

    def test_peak_lr_mismatch_still_raises(self):
        config = {
            "training": {
                "learning_rate": 1.0e-3,
                "weight_decay": 1.0e-5,
                "epochs": 6,
                "optimizer": "AdamW",
                "scheduler": {
                    "type": "RIGNOThreePhase",
                    "warmup_epochs": 2,
                    "cosine_epochs": 2,
                    "exp_epochs": 2,
                    "peak_lr": 2.0e-3,
                    "init_lr": 5.0e-5,
                    "cosine_floor_lr": 1.0e-4,
                    "final_lr": 1.0e-5,
                },
            }
        }
        model = FNO1d(modes=2, width=8, in_channels=2, out_channels=1, n_layers=2, cond_dim=7)
        optimizer = build_optimizer(config, model.parameters())

        with pytest.raises(ValueError, match="scheduler.peak_lr must match training.learning_rate"):
            build_scheduler(config, optimizer)


class TestSearchSpaceConfig:
    def test_search_space_drops_stale_cosine_restart_params(self):
        search_space_path = Path(__file__).resolve().parents[1] / "conf" / "search_space" / "medium.yaml"
        text = search_space_path.read_text(encoding="utf-8")

        assert "training.scheduler.T_0" not in text
        assert "training.scheduler.T_mult" not in text


# ===================== _is_training_complete =====================

class TestIsTrainingComplete:
    def test_complete_when_best_exists_no_latest(self, tmp_path):
        (tmp_path / "fno1d_best.pt").touch()
        assert _is_training_complete(tmp_path) is True

    def test_incomplete_when_latest_exists(self, tmp_path):
        (tmp_path / "fno1d_best.pt").touch()
        (tmp_path / "fno1d_latest.pt").touch()
        assert _is_training_complete(tmp_path) is False

    def test_incomplete_when_nothing_exists(self, tmp_path):
        assert _is_training_complete(tmp_path) is False

    def test_incomplete_when_only_latest(self, tmp_path):
        (tmp_path / "fno1d_latest.pt").touch()
        assert _is_training_complete(tmp_path) is False


# ===================== _load_completed_result =====================

class TestLoadCompletedResult:
    def test_returns_correct_fields(self, tmp_path):
        best_path = tmp_path / "fno1d_best.pt"
        torch.save({"best_val": 0.05, "epoch": 100, "seed": 0}, best_path)
        result = _load_completed_result(tmp_path, seed=0)
        assert result["seed"] == 0
        assert result["best_val"] == pytest.approx(0.05)
        assert result["best_path"] == str(best_path)

    def test_different_seed_value(self, tmp_path):
        torch.save({"best_val": 1.23, "epoch": 50, "seed": 7}, tmp_path / "fno1d_best.pt")
        result = _load_completed_result(tmp_path, seed=7)
        assert result["seed"] == 7
        assert result["best_val"] == pytest.approx(1.23)


# ===================== run_one_seed (skip & resume) =====================

class TestRunOneSeedResume:
    """Test that run_one_seed skips completed runs and resumes interrupted ones."""

    @pytest.fixture
    def seed_config(self, tmp_path, synthetic_trajectories, synthetic_sim_params):
        """Build a minimal config pointing at synthetic data files."""
        trajectories, x_grid, t_grid = synthetic_trajectories
        np.save(tmp_path / "trajectories.npy", trajectories)
        np.save(tmp_path / "x_grid.npy", x_grid)
        np.save(tmp_path / "t_grid.npy", t_grid)
        np.save(tmp_path / "sim_params.npy", synthetic_sim_params)

        return {
            "data": {
                "trajectories.npy": str(tmp_path / "trajectories.npy"),
                "x_grid_path": str(tmp_path / "x_grid.npy"),
                "t_grid_path": str(tmp_path / "t_grid.npy"),
                "sim_params_path": str(tmp_path / "sim_params.npy"),
            },
            "model": {
                "parameters": {
                    "modes": 2,
                    "width": 8,
                    "in_channels": 2,
                    "out_channels": 1,
                    "n_layers": 2,
                    "cond_dim": 7,
                    "cond_hidden": 32,
                }
            },
            "training": {
                "batch_size": 4,
                "n_snapshots": 5,
                "device": "cpu",
                "epochs": 4,
                "learning_rate": 0.001,
                "weight_decay": 1e-5,
                "scheduler": {"step_size": 2, "gamma": 0.5},
                "validate_every": 2,
                "patience": 50,
                "seeds": [0],
            },
        }

    def test_skip_completed_run(self, tmp_path, seed_config):
        """A completed run (best exists, no latest) should be skipped."""
        run_dir = tmp_path / "seed0"
        run_dir.mkdir()
        # Simulate a completed run
        torch.save({"best_val": 0.042, "epoch": 99, "seed": 0}, run_dir / "fno1d_best.pt")

        result = run_one_seed(seed_config, seed=0, run_dir=run_dir)
        assert result["best_val"] == pytest.approx(0.042)
        # No sentinel should exist
        assert not (run_dir / "fno1d_latest.pt").exists()

    def test_fresh_run_produces_checkpoint(self, tmp_path, seed_config):
        """A fresh run should produce best checkpoint and clean up sentinel."""
        run_dir = tmp_path / "seed0"
        result = run_one_seed(seed_config, seed=0, run_dir=run_dir)

        assert "best_val" in result
        assert (run_dir / "fno1d_best.pt").exists()
        assert not (run_dir / "fno1d_latest.pt").exists()  # sentinel removed
        assert (run_dir / "train_metrics.csv").exists()

    def test_cosine_warm_restarts_scheduler(self, tmp_path, seed_config):
        """CosineWarmRestarts scheduler should complete training without error."""
        cosine_config = {
            **seed_config,
            "training": {
                **seed_config["training"],
                "scheduler": {"type": "CosineWarmRestarts", "T_0": 2, "T_mult": 1, "eta_min": 1e-6},
            },
        }
        run_dir = tmp_path / "seed0_cosine"
        result = run_one_seed(cosine_config, seed=0, run_dir=run_dir)
        assert "best_val" in result
        assert (run_dir / "fno1d_best.pt").exists()

    def test_adamw_rigno_three_phase_scheduler(self, tmp_path, seed_config):
        """AdamW + RIGNOThreePhase should complete training with fraction/ratio config."""
        rigno_config = {
            **seed_config,
            "training": {
                **seed_config["training"],
                "optimizer": "AdamW",
                "epochs": 10,
                "scheduler": {
                    "type": "RIGNOThreePhase",
                    "warmup_fraction": 0.2,
                    "cosine_fraction": 0.6,
                    "exp_fraction": 0.2,
                    "init_lr_ratio": 0.05,
                    "cosine_floor_lr_ratio": 0.05,
                    "final_lr_ratio": 0.005,
                },
            },
        }
        run_dir = tmp_path / "seed0_rigno"
        result = run_one_seed(rigno_config, seed=0, run_dir=run_dir)

        assert "best_val" in result
        assert (run_dir / "fno1d_best.pt").exists()

        csv_path = run_dir / "train_metrics.csv"
        with csv_path.open("r", newline="") as f:
            rows = list(csv.DictReader(f))

        assert float(rows[0]["lr"]) == pytest.approx(5.0e-5)
        assert float(rows[-1]["lr"]) == pytest.approx(5.0e-6)

    def test_legacy_adamw_rigno_three_phase_scheduler(self, tmp_path, seed_config):
        """Explicit RIGNO phase lengths and LR endpoints should remain supported."""
        rigno_config = {
            **seed_config,
            "training": {
                **seed_config["training"],
                "optimizer": "AdamW",
                "epochs": 6,
                "scheduler": {
                    "type": "RIGNOThreePhase",
                    "warmup_epochs": 2,
                    "cosine_epochs": 2,
                    "exp_epochs": 2,
                    "init_lr": 5.0e-5,
                    "peak_lr": 1.0e-3,
                    "cosine_floor_lr": 1.0e-4,
                    "final_lr": 1.0e-5,
                },
            },
        }
        run_dir = tmp_path / "seed0_rigno"
        result = run_one_seed(rigno_config, seed=0, run_dir=run_dir)

        assert "best_val" in result
        assert (run_dir / "fno1d_best.pt").exists()

        csv_path = run_dir / "train_metrics.csv"
        with csv_path.open("r", newline="") as f:
            rows = list(csv.DictReader(f))

        assert float(rows[0]["lr"]) == pytest.approx(5.0e-5)
        assert float(rows[-1]["lr"]) == pytest.approx(1.0e-5)

    def test_grad_clip_in_full_run(self, tmp_path, seed_config):
        """grad_clip config should be picked up and run without error."""
        clip_config = {
            **seed_config,
            "training": {**seed_config["training"], "grad_clip": 0.5},
        }
        run_dir = tmp_path / "seed0_clip"
        result = run_one_seed(clip_config, seed=0, run_dir=run_dir)
        assert "best_val" in result

    def test_curriculum_warmup_in_full_run(self, tmp_path, seed_config):
        """curriculum_warmup config should complete training without error."""
        curriculum_config = {
            **seed_config,
            "training": {**seed_config["training"], "curriculum_warmup": 2},
        }
        run_dir = tmp_path / "seed0_curriculum"
        result = run_one_seed(curriculum_config, seed=0, run_dir=run_dir)
        assert "best_val" in result

    def test_resume_interrupted_run(self, tmp_path, seed_config):
        """An interrupted run (sentinel exists) should resume and complete."""
        run_dir = tmp_path / "seed0"
        run_dir.mkdir()

        # Do a fresh short run first (2 epochs, validate_every=2 so epoch 0 validates)
        short_config = {**seed_config, "training": {**seed_config["training"], "epochs": 2}}
        run_one_seed(short_config, seed=0, run_dir=run_dir)

        # The run completed cleanly, so simulate interruption by recreating sentinel
        best_ckpt = torch.load(run_dir / "fno1d_best.pt", map_location="cpu", weights_only=False)
        torch.save(best_ckpt, run_dir / "fno1d_latest.pt")

        # Now resume with more epochs
        resume_config = {**seed_config, "training": {**seed_config["training"], "epochs": 6}}
        result = run_one_seed(resume_config, seed=0, run_dir=run_dir)

        assert "best_val" in result
        assert (run_dir / "fno1d_best.pt").exists()
        assert not (run_dir / "fno1d_latest.pt").exists()  # sentinel cleaned

    def test_csv_no_duplicate_header_on_resume(self, tmp_path, seed_config):
        """Resuming should append to CSV without writing a second header row."""
        run_dir = tmp_path / "seed0"

        # Run 2 epochs, then simulate interruption
        short_config = {**seed_config, "training": {**seed_config["training"], "epochs": 2}}
        run_one_seed(short_config, seed=0, run_dir=run_dir)

        best_ckpt = torch.load(run_dir / "fno1d_best.pt", map_location="cpu", weights_only=False)
        torch.save(best_ckpt, run_dir / "fno1d_latest.pt")

        # Resume with more epochs
        resume_config = {**seed_config, "training": {**seed_config["training"], "epochs": 6}}
        run_one_seed(resume_config, seed=0, run_dir=run_dir)

        csv_path = run_dir / "train_metrics.csv"
        with csv_path.open("r", newline="") as f:
            lines = f.readlines()

        header_count = sum(1 for line in lines if line.startswith("epoch,"))
        assert header_count == 1, f"Expected 1 header row, found {header_count}"

    def test_csv_epochs_continue_after_resume(self, tmp_path, seed_config):
        """Resumed epochs should start after the last epoch in the existing CSV."""
        run_dir = tmp_path / "seed0"

        short_config = {**seed_config, "training": {**seed_config["training"], "epochs": 2}}
        run_one_seed(short_config, seed=0, run_dir=run_dir)

        best_ckpt = torch.load(run_dir / "fno1d_best.pt", map_location="cpu", weights_only=False)
        torch.save(best_ckpt, run_dir / "fno1d_latest.pt")

        resume_config = {**seed_config, "training": {**seed_config["training"], "epochs": 6}}
        run_one_seed(resume_config, seed=0, run_dir=run_dir)

        csv_path = run_dir / "train_metrics.csv"
        with csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            epochs = [int(row["epoch"]) for row in reader]

        # Epochs should be monotonically increasing with no repeats
        assert epochs == sorted(set(epochs))
        # Should have epochs from the resumed portion (starting after epoch from checkpoint)
        assert len(epochs) > 2

    def test_resume_rejects_optimizer_scheduler_mismatch(self, tmp_path, seed_config):
        """Resuming should fail clearly when optimizer or scheduler family changes."""
        run_dir = tmp_path / "seed0_mismatch"

        short_config = {**seed_config, "training": {**seed_config["training"], "epochs": 2}}
        run_one_seed(short_config, seed=0, run_dir=run_dir)

        best_ckpt = torch.load(run_dir / "fno1d_best.pt", map_location="cpu", weights_only=False)
        torch.save(best_ckpt, run_dir / "fno1d_latest.pt")

        mismatch_config = {
            **seed_config,
            "training": {
                **seed_config["training"],
                "optimizer": "AdamW",
                "epochs": 6,
                "scheduler": {
                    "type": "RIGNOThreePhase",
                    "warmup_epochs": 2,
                    "cosine_epochs": 2,
                    "exp_epochs": 2,
                    "init_lr": 5.0e-5,
                    "peak_lr": 1.0e-3,
                    "cosine_floor_lr": 1.0e-4,
                    "final_lr": 1.0e-5,
                },
            },
        }

        with pytest.raises(ValueError, match="fresh run directory|remove fno1d_latest.pt"):
            run_one_seed(mismatch_config, seed=0, run_dir=run_dir)


# ===================== run_config_seeds =====================

class TestRunConfigSeeds:
    """Test multi-seed orchestration with skip/resume."""

    @pytest.fixture
    def seed_config(self, tmp_path, synthetic_trajectories, synthetic_sim_params):
        """Build a minimal config pointing at synthetic data files."""
        trajectories, x_grid, t_grid = synthetic_trajectories
        np.save(tmp_path / "trajectories.npy", trajectories)
        np.save(tmp_path / "x_grid.npy", x_grid)
        np.save(tmp_path / "t_grid.npy", t_grid)
        np.save(tmp_path / "sim_params.npy", synthetic_sim_params)

        return {
            "data": {
                "trajectories.npy": str(tmp_path / "trajectories.npy"),
                "x_grid_path": str(tmp_path / "x_grid.npy"),
                "t_grid_path": str(tmp_path / "t_grid.npy"),
                "sim_params_path": str(tmp_path / "sim_params.npy"),
            },
            "model": {
                "parameters": {
                    "modes": 2,
                    "width": 8,
                    "in_channels": 2,
                    "out_channels": 1,
                    "n_layers": 2,
                    "cond_dim": 7,
                    "cond_hidden": 32,
                }
            },
            "training": {
                "batch_size": 4,
                "n_snapshots": 5,
                "device": "cpu",
                "epochs": 4,
                "learning_rate": 0.001,
                "weight_decay": 1e-5,
                "scheduler": {"step_size": 2, "gamma": 0.5},
                "validate_every": 2,
                "patience": 50,
                "seeds": [0, 1],
            },
        }

    def test_returns_correct_structure(self, tmp_path, seed_config):
        run_dir = tmp_path / "run"
        summary = run_config_seeds(seed_config, base_run_dir=run_dir, seeds=[0, 1])

        assert summary["num_seeds"] == 2
        assert "mean_best_val" in summary
        assert len(summary["per_seed"]) == 2
        for entry in summary["per_seed"]:
            assert "seed" in entry
            assert "best_val" in entry
            assert "best_path" in entry

    def test_creates_seed_directories(self, tmp_path, seed_config):
        run_dir = tmp_path / "run"
        run_config_seeds(seed_config, base_run_dir=run_dir, seeds=[0, 1])

        assert (run_dir / "seed0" / "fno1d_best.pt").exists()
        assert (run_dir / "seed1" / "fno1d_best.pt").exists()
        # Sentinels should be cleaned up
        assert not (run_dir / "seed0" / "fno1d_latest.pt").exists()
        assert not (run_dir / "seed1" / "fno1d_latest.pt").exists()

    def test_mean_best_val_is_average(self, tmp_path, seed_config):
        run_dir = tmp_path / "run"
        summary = run_config_seeds(seed_config, base_run_dir=run_dir, seeds=[0, 1])

        per_seed_vals = [entry["best_val"] for entry in summary["per_seed"]]
        expected_mean = sum(per_seed_vals) / len(per_seed_vals)
        assert summary["mean_best_val"] == pytest.approx(expected_mean)

    def test_skips_already_completed_seeds(self, tmp_path, seed_config):
        run_dir = tmp_path / "run"

        # Pre-populate seed0 as complete
        seed0_dir = run_dir / "seed0"
        seed0_dir.mkdir(parents=True)
        torch.save({"best_val": 0.01, "epoch": 99, "seed": 0}, seed0_dir / "fno1d_best.pt")

        summary = run_config_seeds(seed_config, base_run_dir=run_dir, seeds=[0, 1])

        assert summary["num_seeds"] == 2
        # seed0 should have the pre-populated value
        seed0_result = next(r for r in summary["per_seed"] if r["seed"] == 0)
        assert seed0_result["best_val"] == pytest.approx(0.01)
        # seed1 should have been trained fresh
        assert (run_dir / "seed1" / "fno1d_best.pt").exists()

    def test_uses_config_seeds_when_none_provided(self, tmp_path, seed_config):
        run_dir = tmp_path / "run"
        summary = run_config_seeds(seed_config, base_run_dir=run_dir, seeds=None)
        # config has seeds: [0, 1]
        assert summary["num_seeds"] == 2
