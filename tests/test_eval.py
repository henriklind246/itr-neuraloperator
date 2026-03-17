import math

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.operators.eval import evaluate, mean_std, print_seed_report
from src.operators.fno2d import FNO2d


# ===================== evaluate =====================

@pytest.fixture
def eval_setup():
    """Tiny model + loader for eval tests."""
    Nx, H, in_ch = 11, 10, 7
    model = FNO2d(modes1=2, modes2=2, width=8, in_channels=in_ch)
    model.eval()

    X = torch.randn(4, Nx, H, in_ch)
    Y = torch.randn(4, Nx, H, 1)
    loader = DataLoader(TensorDataset(X, Y), batch_size=2)
    device = torch.device("cpu")
    return model, loader, device


class TestEvaluate:
    def test_returns_float(self, eval_setup):
        model, loader, device = eval_setup
        result = evaluate(model, loader, device)
        assert isinstance(result, float)

    def test_nonnegative(self, eval_setup):
        model, loader, device = eval_setup
        result = evaluate(model, loader, device)
        assert result >= 0

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
            {"seed": 0, "best_epoch": 10, "best_val": 0.5, "test_rel_l2": 0.6, "ckpt": "x"},
            {"seed": 1, "best_epoch": 20, "best_val": 0.4, "test_rel_l2": 0.5, "ckpt": "y"},
        ]
        summary = print_seed_report(results)
        assert isinstance(summary, dict)
        assert "num_seeds" in summary
        assert "best_val_loss_mean" in summary
        assert "test_rel_l2_mean" in summary

    def test_correct_num_seeds(self):
        results = [
            {"seed": 0, "best_epoch": 10, "best_val": 0.5, "test_rel_l2": 0.6, "ckpt": "x"},
        ]
        summary = print_seed_report(results)
        assert summary["num_seeds"] == 1
