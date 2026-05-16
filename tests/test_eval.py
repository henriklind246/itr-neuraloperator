import math

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.operators.eval import evaluate, mean_std, print_seed_report
from src.operators.fno2d import FNO2d

SPATIAL_IN_CHANNELS = 4
COND_STATIC_DIM = 11
TEMPORAL_TOKEN_DIM = 5
TEMPORAL_SAMPLES = 64


# ===================== evaluate =====================

@pytest.fixture
def eval_setup():
    """Tiny model + 5-tuple loader for eval tests."""
    Nx = 11
    Ny = 11
    model = FNO2d(
        modes1=2, modes2=2, width=8,
        in_channels=SPATIAL_IN_CHANNELS, out_channels=1, n_layers=2,
        cond_static_dim=COND_STATIC_DIM,
        temporal_token_dim=TEMPORAL_TOKEN_DIM,
        temporal_hidden=16,
        forcing_embed_dim=16,
    )
    model.eval()

    x_spatial = torch.randn(4, Nx, Ny, SPATIAL_IN_CHANNELS)
    cond_static = torch.rand(4, COND_STATIC_DIM)
    forcing_seq = torch.randn(4, TEMPORAL_SAMPLES, TEMPORAL_TOKEN_DIM)
    Y = torch.randn(4, Nx, Ny, 1)
    T_stats = torch.stack([
        torch.randn(4),             # mu_s
        torch.abs(torch.randn(4)),  # sigma_s (positive)
    ], dim=-1)  # (4, 2)
    loader = DataLoader(
        TensorDataset(x_spatial, cond_static, forcing_seq, Y, T_stats),
        batch_size=2,
    )
    device = torch.device("cpu")
    return model, loader, device


EXPECTED_METRIC_KEYS = {"rel_l2_norm", "rel_l2_phys", "iface_rel_l2_norm", "iface_rel_l2_phys"}


class TestEvaluate:
    def test_returns_dict(self, eval_setup):
        model, loader, device = eval_setup
        result = evaluate(model, loader, device)
        assert isinstance(result, dict)
        assert set(result.keys()) == EXPECTED_METRIC_KEYS
        for v in result.values():
            assert isinstance(v, float)

    def test_nonnegative(self, eval_setup):
        model, loader, device = eval_setup
        result = evaluate(model, loader, device)
        for k in ("rel_l2_norm", "rel_l2_phys"):
            assert result[k] >= 0

    def test_with_iface_mask(self, eval_setup):
        model, loader, device = eval_setup
        Nx = 11
        Ny = 11
        iface_mask = torch.zeros(Nx, Ny, dtype=torch.bool)
        iface_mask[4:7, :] = True  # mark 3 columns as interface
        result = evaluate(model, loader, device, iface_mask=iface_mask)
        assert isinstance(result["iface_rel_l2_norm"], float)
        assert isinstance(result["iface_rel_l2_phys"], float)
        assert result["iface_rel_l2_norm"] >= 0
        assert result["iface_rel_l2_phys"] >= 0

    def test_iface_zero_without_mask(self, eval_setup):
        model, loader, device = eval_setup
        result = evaluate(model, loader, device)
        assert result["iface_rel_l2_norm"] == 0.0
        assert result["iface_rel_l2_phys"] == 0.0

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

def _seed_result(seed, best_val, norm, phys, iface_norm, iface_phys):
    return {
        "seed": seed,
        "best_epoch": 10,
        "best_val": best_val,
        "test_rel_l2_norm": norm,
        "test_rel_l2": phys,
        "test_iface_rel_l2_norm": iface_norm,
        "test_iface_rel_l2": iface_phys,
        "ckpt": "x",
    }


class TestPrintSeedReport:
    def test_returns_dict(self):
        results = [
            _seed_result(0, 0.5, 2.6, 0.06, 2.8, 0.08),
            _seed_result(1, 0.4, 2.5, 0.05, 2.7, 0.07),
        ]
        summary = print_seed_report(results)
        assert isinstance(summary, dict)
        for key in (
            "num_seeds",
            "best_val_loss_mean",
            "test_rel_l2_norm_mean",
            "test_rel_l2_norm_std",
            "test_rel_l2_mean",
            "test_iface_rel_l2_norm_mean",
            "test_iface_rel_l2_mean",
            "test_iface_rel_l2_std",
        ):
            assert key in summary

    def test_correct_num_seeds(self):
        results = [_seed_result(0, 0.5, 2.6, 0.06, 2.8, 0.08)]
        summary = print_seed_report(results)
        assert summary["num_seeds"] == 1
