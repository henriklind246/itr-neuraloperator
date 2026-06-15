import math

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.operators.eval import evaluate, mean_std, print_seed_report
from src.operators.fno2d import FNO2d

SPATIAL_IN_CHANNELS = 20
COND_STATIC_DIM = 23
TEMPORAL_TOKEN_DIM = 5
TEMPORAL_SAMPLES = 64

_ITEM_KEYS = ("spatial", "cond_static", "forcing_seq", "Y", "T_stats")


def _dict_collate(batch):
    """Map TensorDataset tuples -> dict batches matching the dataset item schema."""
    stacked = [torch.stack([sample[i] for sample in batch]) for i in range(len(_ITEM_KEYS))]
    return dict(zip(_ITEM_KEYS, stacked))


# ===================== evaluate =====================

@pytest.fixture
def eval_setup():
    """Tiny model + dict-batch loader for eval tests."""
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
        collate_fn=_dict_collate,
    )
    device = torch.device("cpu")
    return model, loader, device


EXPECTED_METRIC_KEYS = {
    "rel_l2_norm", "rel_l2_phys",
    "iface_rel_l2_norm", "iface_rel_l2_phys",
    "boundary_rel_l2_norm", "boundary_rel_l2_phys",
    "nrmse", "nrmse_p50", "nrmse_iqr", "nrmse_p90", "nrmse_p99", "nrmse_max",
    "rmse_K", "rmse_K_p90", "rmse_K_p99", "rmse_K_max",
    "gnrmse_pct", "gnrmse_pct_p99", "max_err_K",
    "node_jump_rmse_K",
    "node_jump_nrmse", "node_jump_nrmse_p90",
    "node_jump_nrmse_p99", "node_jump_nrmse_max",
    "node_jump_gnrmse_pct", "node_jump_gnrmse_pct_p99",
}


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

    def test_new_metrics_finite_and_nonnegative(self, eval_setup):
        model, loader, device = eval_setup
        result = evaluate(model, loader, device)
        for k in (
            "nrmse", "nrmse_p50", "nrmse_iqr", "nrmse_p90", "nrmse_p99", "nrmse_max",
            "rmse_K", "rmse_K_p90", "rmse_K_p99", "rmse_K_max",
            "gnrmse_pct", "gnrmse_pct_p99", "max_err_K",
            "node_jump_rmse_K",
            "node_jump_nrmse", "node_jump_nrmse_p90",
            "node_jump_nrmse_p99", "node_jump_nrmse_max",
            "node_jump_gnrmse_pct", "node_jump_gnrmse_pct_p99",
        ):
            assert math.isfinite(result[k]), k
            assert result[k] >= 0.0, k

    def test_nrmse_tail_ordering(self, eval_setup):
        model, loader, device = eval_setup
        result = evaluate(model, loader, device)
        assert result["nrmse"] <= result["nrmse_p90"] + 1e-6
        assert result["nrmse_p90"] <= result["nrmse_p99"] + 1e-6
        assert result["nrmse_p99"] <= result["nrmse_max"] + 1e-6


# ===================== evaluate nRMSE invariance / batching =====================


def _fixed_eval_loader(sigma, n=8, Nx=11, Ny=11, batch_size=2, seed=0):
    """Deterministic loader; T_stats sigma_s column set to `sigma` for Kelvin scaling."""
    g = torch.Generator().manual_seed(seed)
    x_spatial = torch.randn(n, Nx, Ny, SPATIAL_IN_CHANNELS, generator=g)
    cond_static = torch.rand(n, COND_STATIC_DIM, generator=g)
    forcing_seq = torch.randn(n, TEMPORAL_SAMPLES, TEMPORAL_TOKEN_DIM, generator=g)
    Y = torch.randn(n, Nx, Ny, 1, generator=g)
    T_stats = torch.stack([
        torch.zeros(n),
        torch.full((n,), float(sigma)),
    ], dim=-1)
    return DataLoader(
        TensorDataset(x_spatial, cond_static, forcing_seq, Y, T_stats),
        batch_size=batch_size,
        collate_fn=_dict_collate,
    )


class TestEvaluateNRMSEInvariance:
    def test_nrmse_invariant_to_sigma(self, eval_setup):
        """nRMSE is normalization-invariant: same value for any sigma_global (z-scored == Kelvin)."""
        model, _, device = eval_setup
        model.eval()
        a = evaluate(model, _fixed_eval_loader(1.0), device)
        b = evaluate(model, _fixed_eval_loader(7.5), device)
        assert a["nrmse"] == pytest.approx(b["nrmse"], rel=1e-5)
        assert a["node_jump_nrmse"] == pytest.approx(b["node_jump_nrmse"], rel=1e-5)

    def test_rmse_K_scales_with_sigma(self, eval_setup):
        """Kelvin RMSE is a scalar multiple of sigma_global."""
        model, _, device = eval_setup
        model.eval()
        a = evaluate(model, _fixed_eval_loader(1.0), device)
        b = evaluate(model, _fixed_eval_loader(7.5), device)
        assert b["rmse_K"] == pytest.approx(a["rmse_K"] * 7.5, rel=1e-5)

    def test_gnrmse_pct_is_rmse_K_over_sigma(self, eval_setup):
        """gnrmse_pct == rmse_K / sigma_global * 100 (dimensionless restatement)."""
        model, _, device = eval_setup
        model.eval()
        sigma = 7.5
        r = evaluate(model, _fixed_eval_loader(sigma), device)
        assert r["gnrmse_pct"] == pytest.approx(r["rmse_K"] / sigma * 100.0, rel=1e-5)

    def test_gnrmse_pct_invariant_to_sigma(self, eval_setup):
        """gnrmse_pct is normalized (= rms*100), so independent of sigma_global."""
        model, _, device = eval_setup
        model.eval()
        a = evaluate(model, _fixed_eval_loader(1.0), device)
        b = evaluate(model, _fixed_eval_loader(7.5), device)
        assert a["gnrmse_pct"] == pytest.approx(b["gnrmse_pct"], rel=1e-5)

    def test_nrmse_independent_of_batch_size(self, eval_setup):
        """Per-sample mean nRMSE does not depend on how batches are chunked."""
        model, _, device = eval_setup
        model.eval()
        a = evaluate(model, _fixed_eval_loader(1.0, batch_size=2), device)
        b = evaluate(model, _fixed_eval_loader(1.0, batch_size=4), device)
        assert a["nrmse"] == pytest.approx(b["nrmse"], rel=1e-5)
        assert a["node_jump_nrmse"] == pytest.approx(b["node_jump_nrmse"], rel=1e-5)


# ===================== evaluate per-sample interface =====================

_ITEM_KEYS_IFACE = ("spatial", "cond_static", "forcing_seq", "Y", "T_stats")


def _make_iface_loader(interface_x_col, n=4, Nx=11, Ny=11):
    """Loader whose T_stats carries a slot-2 interface_x column."""
    x_spatial = torch.randn(n, Nx, Ny, SPATIAL_IN_CHANNELS)
    cond_static = torch.rand(n, COND_STATIC_DIM)
    forcing_seq = torch.randn(n, TEMPORAL_SAMPLES, TEMPORAL_TOKEN_DIM)
    Y = torch.randn(n, Nx, Ny, 1)
    T_stats = torch.stack([
        torch.zeros(n),            # mu_s
        torch.ones(n),             # sigma_s
        interface_x_col,           # slot 2: interface_x
    ], dim=-1)
    return DataLoader(
        TensorDataset(x_spatial, cond_static, forcing_seq, Y, T_stats),
        batch_size=2,
        collate_fn=_dict_collate,
    )


@pytest.fixture
def iface_model():
    Nx = Ny = 11
    model = FNO2d(
        modes1=2, modes2=2, width=8,
        in_channels=SPATIAL_IN_CHANNELS, out_channels=1, n_layers=2,
        cond_static_dim=COND_STATIC_DIM,
        temporal_token_dim=TEMPORAL_TOKEN_DIM,
        temporal_hidden=16,
        forcing_embed_dim=16,
    )
    model.eval()
    return model


class TestEvaluatePerSampleInterface:
    def _grids(self, Nx=11, Ny=11):
        import numpy as np
        return (
            np.linspace(0.0, 1.0, Nx).astype("float32"),
            np.linspace(0.0, 1.0, Ny).astype("float32"),
        )

    def test_dynamic_half_matches_fixed_mask(self, iface_model):
        """interface_x == 0.5 everywhere: per-sample band matches the fixed x=0.5 mask."""
        from src.operators.losses import build_interface_mask
        torch.manual_seed(0)
        x_grid, y_grid = self._grids()
        loader = _make_iface_loader(torch.full((4,), 0.5))
        mask = build_interface_mask(x_grid, y_grid, 0.5, 0.05)

        fixed = evaluate(iface_model, loader, torch.device("cpu"), iface_mask=mask)
        dynamic = evaluate(
            iface_model, loader, torch.device("cpu"),
            iface_mask=mask, x_grid=x_grid, interface_half_width=0.05,
            use_per_sample_interface=True,
        )
        assert dynamic["iface_rel_l2_norm"] == pytest.approx(fixed["iface_rel_l2_norm"], rel=1e-5)

    def test_flag_off_uses_fixed_mask_even_with_slot2(self, iface_model):
        """source-like: slot 2 present but flag off -> fixed mask path (ignores x_grid)."""
        from src.operators.losses import build_interface_mask
        torch.manual_seed(0)
        x_grid, y_grid = self._grids()
        loader = _make_iface_loader(torch.full((4,), 0.5))
        mask = build_interface_mask(x_grid, y_grid, 0.5, 0.05)

        fixed = evaluate(iface_model, loader, torch.device("cpu"), iface_mask=mask)
        flag_off = evaluate(
            iface_model, loader, torch.device("cpu"),
            iface_mask=mask, x_grid=x_grid, interface_half_width=0.05,
            use_per_sample_interface=False,
        )
        assert flag_off["iface_rel_l2_norm"] == pytest.approx(fixed["iface_rel_l2_norm"], rel=1e-6)

    def test_dynamic_differs_from_fixed_when_interface_off_center(self, iface_model):
        """interface_x away from 0.5 should drive a different interface metric than the fixed mask."""
        from src.operators.losses import build_interface_mask
        torch.manual_seed(3)
        x_grid, y_grid = self._grids()
        loader = _make_iface_loader(torch.full((4,), 0.2))
        mask = build_interface_mask(x_grid, y_grid, 0.5, 0.05)

        fixed = evaluate(iface_model, loader, torch.device("cpu"), iface_mask=mask)
        dynamic = evaluate(
            iface_model, loader, torch.device("cpu"),
            iface_mask=mask, x_grid=x_grid, interface_half_width=0.05,
            use_per_sample_interface=True,
        )
        assert dynamic["iface_rel_l2_norm"] != pytest.approx(fixed["iface_rel_l2_norm"], rel=1e-6)

    def test_forcing_like_two_col_tstats_falls_back(self, iface_model):
        """forcing has t_stats_dim=2: flag on but no slot 2 -> fixed mask fallback."""
        from src.operators.losses import build_interface_mask
        torch.manual_seed(0)
        x_grid, y_grid = self._grids()
        # 2-column T_stats loader (no interface slot)
        x_spatial = torch.randn(4, 11, 11, SPATIAL_IN_CHANNELS)
        cond_static = torch.rand(4, COND_STATIC_DIM)
        forcing_seq = torch.randn(4, TEMPORAL_SAMPLES, TEMPORAL_TOKEN_DIM)
        Y = torch.randn(4, 11, 11, 1)
        T_stats = torch.stack([torch.zeros(4), torch.ones(4)], dim=-1)
        loader = DataLoader(
            TensorDataset(x_spatial, cond_static, forcing_seq, Y, T_stats),
            batch_size=2, collate_fn=_dict_collate,
        )
        mask = build_interface_mask(x_grid, y_grid, 0.5, 0.05)

        fixed = evaluate(iface_model, loader, torch.device("cpu"), iface_mask=mask)
        dynamic = evaluate(
            iface_model, loader, torch.device("cpu"),
            iface_mask=mask, x_grid=x_grid, interface_half_width=0.05,
            use_per_sample_interface=True,
        )
        assert dynamic["iface_rel_l2_norm"] == pytest.approx(fixed["iface_rel_l2_norm"], rel=1e-6)


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

def _seed_result(seed, best_val, norm, phys, iface_norm, iface_phys,
                 bnd_norm=0.1, bnd_phys=0.01):
    return {
        "seed": seed,
        "best_epoch": 10,
        "best_val": best_val,
        "test_rel_l2_norm": norm,
        "test_rel_l2": phys,
        "test_iface_rel_l2_norm": iface_norm,
        "test_iface_rel_l2": iface_phys,
        "test_boundary_rel_l2_norm": bnd_norm,
        "test_boundary_rel_l2": bnd_phys,
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
