import random

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.operators.fno2d import FNO2d
import csv

from src.operators.train import (
    load_config,
    set_seed,
    train_one_epoch,
    validate,
    _is_training_complete,
    _load_completed_result,
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

@pytest.fixture
def tiny_training_setup():
    """Tiny model + synthetic dataloader for fast training tests."""
    Nx, H, k = 11, 10, 5
    in_channels = k + 5  # 10 (history + x + t + q + k_field + rcp_field)
    model = FNO2d(modes1=2, modes2=2, width=8, in_channels=in_channels)

    # Synthetic data: 4 samples
    X = torch.randn(4, Nx, H, in_channels)
    Y = torch.randn(4, Nx, H, 1)
    loader = DataLoader(TensorDataset(X, Y), batch_size=2)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()
    device = torch.device("cpu")

    return model, loader, optimizer, loss_fn, device


# ===================== train_one_epoch =====================

class TestTrainOneEpoch:
    def test_returns_tuple_of_floats(self, tiny_training_setup):
        model, loader, optimizer, loss_fn, device = tiny_training_setup
        result = train_one_epoch(model, loader, optimizer, loss_fn, device)
        assert isinstance(result, tuple) and len(result) == 2
        loss, rel_l2 = result
        assert isinstance(loss, float)
        assert isinstance(rel_l2, float)

    def test_loss_is_finite(self, tiny_training_setup):
        model, loader, optimizer, loss_fn, device = tiny_training_setup
        loss, rel_l2 = train_one_epoch(model, loader, optimizer, loss_fn, device)
        assert np.isfinite(loss)
        assert np.isfinite(rel_l2)

    def test_loss_is_nonnegative(self, tiny_training_setup):
        model, loader, optimizer, loss_fn, device = tiny_training_setup
        loss, rel_l2 = train_one_epoch(model, loader, optimizer, loss_fn, device)
        assert loss >= 0
        assert rel_l2 >= 0

    def test_updates_parameters(self, tiny_training_setup):
        model, loader, optimizer, loss_fn, device = tiny_training_setup
        params_before = {n: p.clone() for n, p in model.named_parameters()}
        train_one_epoch(model, loader, optimizer, loss_fn, device)
        changed = any(
            not torch.equal(params_before[n], p)
            for n, p in model.named_parameters()
        )
        assert changed


# ===================== validate =====================

class TestValidate:
    def test_returns_float(self, tiny_training_setup):
        model, loader, _, _, device = tiny_training_setup
        val = validate(model, loader, device)
        assert isinstance(val, float)

    def test_loss_is_nonnegative(self, tiny_training_setup):
        model, loader, _, _, device = tiny_training_setup
        val = validate(model, loader, device)
        assert val >= 0

    def test_no_gradient_accumulation(self, tiny_training_setup):
        model, loader, _, _, device = tiny_training_setup
        # Zero all grads first
        model.zero_grad()
        validate(model, loader, device)
        for p in model.parameters():
            assert p.grad is None or torch.all(p.grad == 0)

    def test_does_not_change_parameters(self, tiny_training_setup):
        model, loader, _, _, device = tiny_training_setup
        params_before = {n: p.clone() for n, p in model.named_parameters()}
        validate(model, loader, device)
        for n, p in model.named_parameters():
            assert torch.equal(params_before[n], p)


# ===================== _is_training_complete =====================

class TestIsTrainingComplete:
    def test_complete_when_best_exists_no_latest(self, tmp_path):
        (tmp_path / "fno2d_best.pt").touch()
        assert _is_training_complete(tmp_path) is True

    def test_incomplete_when_latest_exists(self, tmp_path):
        (tmp_path / "fno2d_best.pt").touch()
        (tmp_path / "fno2d_latest.pt").touch()
        assert _is_training_complete(tmp_path) is False

    def test_incomplete_when_nothing_exists(self, tmp_path):
        assert _is_training_complete(tmp_path) is False

    def test_incomplete_when_only_latest(self, tmp_path):
        (tmp_path / "fno2d_latest.pt").touch()
        assert _is_training_complete(tmp_path) is False


# ===================== _load_completed_result =====================

class TestLoadCompletedResult:
    def test_returns_correct_fields(self, tmp_path):
        best_path = tmp_path / "fno2d_best.pt"
        torch.save({"best_val": 0.05, "epoch": 100, "seed": 0}, best_path)
        result = _load_completed_result(tmp_path, seed=0)
        assert result["seed"] == 0
        assert result["best_val"] == pytest.approx(0.05)
        assert result["best_path"] == str(best_path)

    def test_different_seed_value(self, tmp_path):
        torch.save({"best_val": 1.23, "epoch": 50, "seed": 7}, tmp_path / "fno2d_best.pt")
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

        Nx = x_grid.shape[0]
        k, H = 5, 10
        in_channels = k + 5  # history + x + t + q + k_field + rcp_field

        return {
            "data": {
                "trajectories.npy": str(tmp_path / "trajectories.npy"),
                "x_grid_path": str(tmp_path / "x_grid.npy"),
                "t_grid_path": str(tmp_path / "t_grid.npy"),
                "sim_params_path": str(tmp_path / "sim_params.npy"),
            },
            "model": {
                "parameters": {
                    "modes1": 2,
                    "modes2": 2,
                    "width": 8,
                    "in_channels": in_channels,
                    "out_channels": 1,
                }
            },
            "training": {
                "batch_size": 4,
                "k": k,
                "H": H,
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
        torch.save({"best_val": 0.042, "epoch": 99, "seed": 0}, run_dir / "fno2d_best.pt")

        result = run_one_seed(seed_config, seed=0, run_dir=run_dir)
        assert result["best_val"] == pytest.approx(0.042)
        # No sentinel should exist
        assert not (run_dir / "fno2d_latest.pt").exists()

    def test_fresh_run_produces_checkpoint(self, tmp_path, seed_config):
        """A fresh run should produce best checkpoint and clean up sentinel."""
        run_dir = tmp_path / "seed0"
        result = run_one_seed(seed_config, seed=0, run_dir=run_dir)

        assert "best_val" in result
        assert (run_dir / "fno2d_best.pt").exists()
        assert not (run_dir / "fno2d_latest.pt").exists()  # sentinel removed
        assert (run_dir / "train_metrics.csv").exists()

    def test_resume_interrupted_run(self, tmp_path, seed_config):
        """An interrupted run (sentinel exists) should resume and complete."""
        run_dir = tmp_path / "seed0"
        run_dir.mkdir()

        # Do a fresh short run first (2 epochs, validate_every=2 so epoch 0 validates)
        short_config = {**seed_config, "training": {**seed_config["training"], "epochs": 2}}
        run_one_seed(short_config, seed=0, run_dir=run_dir)

        # The run completed cleanly, so simulate interruption by recreating sentinel
        best_ckpt = torch.load(run_dir / "fno2d_best.pt", map_location="cpu", weights_only=False)
        torch.save(best_ckpt, run_dir / "fno2d_latest.pt")

        # Now resume with more epochs
        resume_config = {**seed_config, "training": {**seed_config["training"], "epochs": 6}}
        result = run_one_seed(resume_config, seed=0, run_dir=run_dir)

        assert "best_val" in result
        assert (run_dir / "fno2d_best.pt").exists()
        assert not (run_dir / "fno2d_latest.pt").exists()  # sentinel cleaned

    def test_csv_no_duplicate_header_on_resume(self, tmp_path, seed_config):
        """Resuming should append to CSV without writing a second header row."""
        run_dir = tmp_path / "seed0"

        # Run 2 epochs, then simulate interruption
        short_config = {**seed_config, "training": {**seed_config["training"], "epochs": 2}}
        run_one_seed(short_config, seed=0, run_dir=run_dir)

        best_ckpt = torch.load(run_dir / "fno2d_best.pt", map_location="cpu", weights_only=False)
        torch.save(best_ckpt, run_dir / "fno2d_latest.pt")

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

        best_ckpt = torch.load(run_dir / "fno2d_best.pt", map_location="cpu", weights_only=False)
        torch.save(best_ckpt, run_dir / "fno2d_latest.pt")

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

        k, H = 5, 10
        in_channels = k + 5

        return {
            "data": {
                "trajectories.npy": str(tmp_path / "trajectories.npy"),
                "x_grid_path": str(tmp_path / "x_grid.npy"),
                "t_grid_path": str(tmp_path / "t_grid.npy"),
                "sim_params_path": str(tmp_path / "sim_params.npy"),
            },
            "model": {
                "parameters": {
                    "modes1": 2,
                    "modes2": 2,
                    "width": 8,
                    "in_channels": in_channels,
                    "out_channels": 1,
                }
            },
            "training": {
                "batch_size": 4,
                "k": k,
                "H": H,
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

        assert (run_dir / "seed0" / "fno2d_best.pt").exists()
        assert (run_dir / "seed1" / "fno2d_best.pt").exists()
        # Sentinels should be cleaned up
        assert not (run_dir / "seed0" / "fno2d_latest.pt").exists()
        assert not (run_dir / "seed1" / "fno2d_latest.pt").exists()

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
        torch.save({"best_val": 0.01, "epoch": 99, "seed": 0}, seed0_dir / "fno2d_best.pt")

        summary = run_config_seeds(seed_config, base_run_dir=run_dir, seeds=[0, 1])

        assert summary["num_seeds"] == 2
        # seed0 should have the pre-populated value
        seed0_result = next(r for r in summary["per_seed"] if r["seed"] == 0)
        assert seed0_result["best_val"] == pytest.approx(0.01)
        # seed1 should have been trained fresh
        assert (run_dir / "seed1" / "fno2d_best.pt").exists()

    def test_uses_config_seeds_when_none_provided(self, tmp_path, seed_config):
        run_dir = tmp_path / "run"
        summary = run_config_seeds(seed_config, base_run_dir=run_dir, seeds=None)
        # config has seeds: [0, 1]
        assert summary["num_seeds"] == 2
