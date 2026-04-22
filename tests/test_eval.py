import math

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.operators.eval import evaluate, mean_std, print_seed_report
from src.operators.fno2d import FNO2d


# ===================== evaluate =====================

@pytest.fixture
def eval_setup():
    """Tiny model + 4-tuple loader for eval tests."""
    Nx = 11
    Ny = 11
    model = FNO2d(modes1=2, modes2=2, width=8, in_channels=3, out_channels=1, n_layers=2, cond_dim=5)
    model.eval()

    x_spatial = torch.randn(4, Nx, Ny, 3)
    cond = torch.rand(4, 5)
    Y = torch.randn(4, Nx, Ny, 1)
    T_stats = torch.stack([
        torch.randn(4),           # mu_s
        torch.abs(torch.randn(4)),  # sigma_s (positive)
    ], dim=-1)  # (4, 2)
    loader = DataLoader(TensorDataset(x_spatial, cond, Y, T_stats), batch_size=2)
    device = torch.device("cpu")
    return model, loader, device


class TestEvaluate:
    def test_returns_tuple(self, eval_setup):
        model, loader, device = eval_setup
        result = evaluate(model, loader, device)
        assert isinstance(result, tuple)
        assert len(result) == 2
        test_rel_l2, test_iface_rel_l2 = result
        assert isinstance(test_rel_l2, float)
        assert isinstance(test_iface_rel_l2, float)

    def test_nonnegative(self, eval_setup):
        model, loader, device = eval_setup
        test_rel_l2, test_iface_rel_l2 = evaluate(model, loader, device)
        assert test_rel_l2 >= 0
        assert test_iface_rel_l2 >= 0

    def test_with_iface_mask(self, eval_setup):
        model, loader, device = eval_setup
        Nx = 11
        Ny = 11
        iface_mask = torch.zeros(Nx, Ny, dtype=torch.bool)
        iface_mask[4:7, :] = True  # mark 3 columns as interface
        test_rel_l2, test_iface_rel_l2 = evaluate(model, loader, device, iface_mask=iface_mask)
        assert isinstance(test_iface_rel_l2, float)
        assert test_iface_rel_l2 >= 0

    def test_model_stays_eval(self, eval_setup):
        model, loader, device = eval_setup
        evaluate(model, loader, device)
        assert not model.training


# ===================== mean_std =====================

class TestMeanStd:
    def test_empty_list(self):
        mu, std = mean_std([])
        assert math.isnan(mu)
        assert math.isnan(std)

    def test_single_value(self):
        mu, std = mean_std([5.0])
        assert mu == 5.0
        assert std == 0.0

    def test_known_values(self):
        values = [2, 4, 4, 4, 5, 5, 7, 9]
        mu, std = mean_std(values)
        expected_mu = sum(values) / len(values)
        expected_std = (sum((x - expected_mu) ** 2 for x in values) / (len(values) - 1)) ** 0.5
        assert pytest.approx(mu) == expected_mu
        assert pytest.approx(std) == expected_std

    def test_identical_values(self):
        mu, std = mean_std([5.0, 5.0, 5.0])
        assert mu == 5.0
        assert std == 0.0


# ===================== print_seed_report =====================

class TestPrintSeedReport:
    def test_returns_dict(self):
        results = [
            {"seed": 0, "best_epoch": 10, "best_val": 0.5, "test_rel_l2": 0.6, "test_iface_rel_l2": 0.8, "ckpt": "x"},
            {"seed": 1, "best_epoch": 20, "best_val": 0.4, "test_rel_l2": 0.5, "test_iface_rel_l2": 0.7, "ckpt": "y"},
        ]
        summary = print_seed_report(results)
        assert isinstance(summary, dict)
        assert "num_seeds" in summary
        assert "best_val_loss_mean" in summary
        assert "test_rel_l2_mean" in summary
        assert "test_iface_rel_l2_mean" in summary
        assert "test_iface_rel_l2_std" in summary

    def test_correct_num_seeds(self):
        results = [
            {"seed": 0, "best_epoch": 10, "best_val": 0.5, "test_rel_l2": 0.6, "test_iface_rel_l2": 0.8, "ckpt": "x"},
        ]
        summary = print_seed_report(results)
        assert summary["num_seeds"] == 1
