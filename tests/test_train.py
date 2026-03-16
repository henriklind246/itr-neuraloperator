import random

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.operators.fno2d import FNO2d
from src.operators.train import load_config, set_seed, train_one_epoch, validate


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
    in_channels = k + 3  # 8 (history + x + t + q)
    model = FNO2d(modes1=2, modes2=2, width=8)
    # Override the lift layer to accept 8 channels instead of 13
    model.linear_p = torch.nn.Linear(in_channels, 8)

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
