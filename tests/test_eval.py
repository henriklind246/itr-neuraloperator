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
        torch.full((4,), 300.0),    # global training mean
        torch.full((4,), 5.0),      # global population std
    ], dim=-1)  # (4, 2)
    loader = DataLoader(
        TensorDataset(x_spatial, cond_static, forcing_seq, Y, T_stats),
        batch_size=2,
        collate_fn=_dict_collate,
    )
    device = torch.device("cpu")
    return model, loader, device


EXPECTED_METRIC_KEYS = {
    "peak_jump_error_mean_K", "peak_jump_error_p95_K",
    "rel_l2_norm", "rel_l2_phys",
    "iface_rel_l2_norm", "iface_rel_l2_phys",
    "boundary_rel_l2_norm", "boundary_rel_l2_phys",
    "nrmse", "nrmse_p50", "nrmse_iqr", "nrmse_p90", "nrmse_p99", "nrmse_max",
    "rmse_K", "rmse_K_p90", "rmse_K_p95", "rmse_K_p99", "rmse_K_max",
    "gnrmse_pct", "gnrmse_pct_p99", "temperature_rise_scale_K", "max_err_K",
    "node_jump_rmse_K", "node_jump_rmse_K_p95",
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
            "rmse_K", "rmse_K_p90", "rmse_K_p95", "rmse_K_p99", "rmse_K_max",
            "gnrmse_pct", "gnrmse_pct_p99", "temperature_rise_scale_K", "max_err_K",
            "node_jump_rmse_K", "node_jump_rmse_K_p95",
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


def _fixed_eval_loader(sigma, n=8, Nx=11, Ny=11, batch_size=2, seed=0,
                       mu=300.0):
    """Deterministic loader; T_stats sigma_s column set to `sigma` for Kelvin scaling."""
    g = torch.Generator().manual_seed(seed)
    x_spatial = torch.randn(n, Nx, Ny, SPATIAL_IN_CHANNELS, generator=g)
    cond_static = torch.rand(n, COND_STATIC_DIM, generator=g)
    forcing_seq = torch.randn(n, TEMPORAL_SAMPLES, TEMPORAL_TOKEN_DIM, generator=g)
    Y = torch.randn(n, Nx, Ny, 1, generator=g)
    T_stats = torch.stack([
        torch.full((n,), float(mu)),
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

    def test_gnrmse_pct_uses_training_temperature_rise_scale(self, eval_setup):
        model, _, device = eval_setup
        model.eval()
        sigma = 7.5
        mu = 307.0
        loader = _fixed_eval_loader(sigma, mu=mu)
        r = evaluate(model, loader, device)
        sse = cells = 0
        with torch.no_grad():
            for batch in loader:
                pred = model(batch["spatial"], batch["cond_static"], batch["forcing_seq"])
                sse += torch.sum(((pred - batch["Y"]) * sigma) ** 2).item()
                cells += batch["Y"].numel()
        expected_scale = math.sqrt(sigma ** 2 + (mu - 300.0) ** 2)
        assert r["temperature_rise_scale_K"] == pytest.approx(expected_scale)
        assert r["gnrmse_pct"] == pytest.approx(
            100.0 * math.sqrt(sse / cells) / expected_scale, rel=1e-5
        )

    def test_gnrmse_pct_invariant_to_sigma(self, eval_setup):
        """With mu=300 K, error and training-rise scale both scale with sigma."""
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

    def test_rel_l2_norm_independent_of_batch_size(self, eval_setup):
        """Global sum-of-squares rel_l2 is invariant to batch chunking.

        n=8 with batch_size=3 yields uneven batches (3, 3, 2); the old
        mean-of-per-batch-ratios convention would disagree with a single batch.
        """
        model, _, device = eval_setup
        model.eval()
        uneven = evaluate(model, _fixed_eval_loader(1.0, batch_size=3), device)
        single = evaluate(model, _fixed_eval_loader(1.0, batch_size=8), device)
        assert uneven["rel_l2_norm"] == pytest.approx(single["rel_l2_norm"], rel=1e-5)
        assert uneven["rel_l2_phys"] == pytest.approx(single["rel_l2_phys"], rel=1e-5)

    def test_rel_l2_norm_matches_global_formula(self, eval_setup):
        """evaluate's rel_l2_norm equals sqrt(sum_sq_err/sum_sq_true)*100."""
        model, _, device = eval_setup
        model.eval()
        loader = _fixed_eval_loader(1.0, batch_size=3)
        result = evaluate(model, loader, device)
        sse = 0.0
        sst = 0.0
        with torch.no_grad():
            for batch in loader:
                yp = model(
                    batch["spatial"].to(device),
                    batch["cond_static"].to(device),
                    batch["forcing_seq"].to(device),
                )
                yb = batch["Y"].to(device)
                sse += torch.sum((yp - yb) ** 2).item()
                sst += torch.sum(yb ** 2).item()
        expected = (sse / sst) ** 0.5 * 100
        assert result["rel_l2_norm"] == pytest.approx(expected, rel=1e-5)


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

    def test_best_seed_selected_on_val_not_test(self, capsys):
        # seed 0 wins on test error, seed 1 wins on validation loss. Selecting
        # on test would report a minimum over seeds as if it were a draw.
        results = [
            _seed_result(0, 0.50, 2.0, 0.04, 2.2, 0.05),
            _seed_result(1, 0.30, 2.9, 0.09, 3.1, 0.10),
        ]
        print_seed_report(results)
        line = next(
            ln for ln in capsys.readouterr().out.splitlines() if "Best by" in ln
        )
        assert "validation loss" in line
        assert "seed=1" in line


# ===================== write_test_records =====================

@pytest.fixture
def records_run(tmp_path, synthetic_trajectories, synthetic_sim_params, small_fno2d):
    """A minimal run_root + dataset that write_test_records can actually score.

    forcing/temporal_encoder, because it is the only benchmark whose model
    fixture already exists. `training.n_snapshots_test` is deliberately set to a
    non-default value so the config-fallback path is distinguishable from both
    the hardcoded 40 and the explicit override.
    """
    import numpy as np

    trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    np.save(data_dir / "trajectories.npy", trajectories)
    np.save(data_dir / "x_grid.npy", x_grid)
    np.save(data_dir / "y_grid.npy", y_grid)
    np.save(data_dir / "t_grid.npy", t_grid)
    np.save(data_dir / "sim_params.npy", synthetic_sim_params, allow_pickle=True)

    from tests.conftest import (
        FORCING_COND_STATIC_DIM,
        FORCING_IN_CHANNELS,
        FORCING_TEMPORAL_SAMPLES,
        FORCING_TEMPORAL_TOKEN_DIM,
    )

    config = {
        "benchmark": {"name": "forcing", "representation": "temporal_encoder"},
        "data": {
            "trajectories.npy": str(data_dir / "trajectories.npy"),
            "x_grid_path": str(data_dir / "x_grid.npy"),
            "y_grid_path": str(data_dir / "y_grid.npy"),
            "t_grid_path": str(data_dir / "t_grid.npy"),
            "sim_params_path": str(data_dir / "sim_params.npy"),
        },
        "training": {"batch_size": 4, "device": "cpu", "n_snapshots_test": 4},
        "model": {
            "parameters": {
                "modes1": 2,
                "modes2": 2,
                "width": 8,
                "in_channels": FORCING_IN_CHANNELS,
                "out_channels": 1,
                "n_layers": 2,
                "cond_static_dim": FORCING_COND_STATIC_DIM,
                "cond_hidden": 256,
                "temporal_token_dim": FORCING_TEMPORAL_TOKEN_DIM,
                "temporal_samples": FORCING_TEMPORAL_SAMPLES,
                "temporal_hidden": 16,
                "forcing_embed_dim": 16,
            }
        },
    }

    run_root = tmp_path / "run"
    seed_dir = run_root / "seed42"
    seed_dir.mkdir(parents=True)
    torch.save(
        {
            "model_state": small_fno2d.state_dict(),
            "conf": config,
            "mu_global": 300.0,
            "sigma_global": 12.5,
            "best_val": 1.0,
        },
        seed_dir / "fno2d_best.pt",
    )
    # split_sim_ids(20, 0.7, 0.15) -> 14 train / 3 val / 3 test.
    return run_root, seed_dir, 3


def _read_records(path):
    import csv

    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


class TestWriteTestRecordsSnapshotCount:
    """n_snapshots_test governs the pair count, which governs study runtime.

    Pairs per simulation are S*(S-1)/2, so this is the only lever that makes a
    multi-arm rollout study affordable. Before it was exposed, the value was
    read from config and could not be overridden per call.
    """

    def test_config_value_is_used_when_override_is_absent(self, records_run):
        from src.operators.eval import write_test_records

        run_root, _seed_dir, n_test_sims = records_run
        out = write_test_records(str(run_root), seed="42", out_name="cfg.csv")
        # config n_snapshots_test=4 -> 4*3/2 = 6 pairs per sim.
        assert len(_read_records(out)) == n_test_sims * 6

    def test_override_changes_the_pair_count(self, records_run):
        from src.operators.eval import write_test_records

        run_root, _seed_dir, n_test_sims = records_run
        out = write_test_records(
            str(run_root), seed="42", out_name="six.csv", n_snapshots_test=6
        )
        # 6*5/2 = 15 pairs per sim.
        assert len(_read_records(out)) == n_test_sims * 15

    def test_provenance_records_the_effective_snapshot_count(self, records_run):
        import json

        from src.operators.eval import write_test_records

        run_root, seed_dir, n_test_sims = records_run
        write_test_records(
            str(run_root), seed="42", out_name="six.csv", n_snapshots_test=6
        )
        payload = json.loads((seed_dir / "six.provenance.json").read_text())
        assert payload["n_snapshots_test"] == 6
        assert payload["n_simulations"] == n_test_sims
        assert payload["n_pairs"] == n_test_sims * 15


class TestTestRecordSufficientStatistics:
    """The pooled statistics must reconstruct the per-row metrics exactly.

    A rollout study aggregates across pairs by summing sse_K2 / target_sse_K2 /
    num_error_cells rather than averaging per-pair percentages. That is only
    valid if these two identities hold, so they are pinned here rather than
    inferred by reading _batch_metrics (which is a closure and cannot be
    imported).
    """

    def test_row_statistics_reproduce_rmse_and_rel_l2(self, records_run):
        from src.operators.eval import write_test_records

        run_root, _seed_dir, _n = records_run
        out = write_test_records(
            str(run_root), seed="42", out_name="stats.csv", n_snapshots_test=4
        )
        rows = _read_records(out)
        assert rows
        for row in rows:
            sse = float(row["sse_K2"])
            target_sse = float(row["target_sse_K2"])
            cells = float(row["num_error_cells"])
            assert cells > 0
            assert target_sse > 0
            assert math.sqrt(sse / cells) == pytest.approx(
                float(row["rmse_K"]), rel=1e-5
            )
            # sigma^2 cancels in the ratio, so this stays in normalized space.
            assert 100.0 * math.sqrt(sse / target_sse) == pytest.approx(
                float(row["rel_l2_pct"]), rel=1e-5
            )


class TestEvalAllSeedsProtocolOverrides:
    """A cross-resolution study fixes the pair count and shrinks the batch.

    Both used to require editing a copy of the checkpoint's ``conf``; the
    overrides must reach the loader without touching the saved checkpoint.
    """

    @pytest.fixture
    def scored_run(self, records_run, monkeypatch):
        import src.operators.eval as eval_mod

        run_root, seed_dir, n_test_sims = records_run
        ckpt_path = seed_dir / "fno2d_best.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ckpt["epoch"] = 3
        torch.save(ckpt, ckpt_path)

        loaders = []
        original = eval_mod.build_test_loader

        def spy(*args, **kwargs):
            out = original(*args, **kwargs)
            loaders.append(out[0])
            return out

        monkeypatch.setattr(eval_mod, "build_test_loader", spy)
        return run_root, ckpt_path, n_test_sims, loaders

    def test_config_values_are_used_when_overrides_are_absent(self, scored_run):
        from src.operators.eval import eval_all_seeds

        run_root, _ckpt_path, n_test_sims, loaders = scored_run
        eval_all_seeds(str(run_root), device="cpu")
        # config n_snapshots_test=4 -> 6 pairs per sim; config batch_size=4.
        assert len(loaders[0].dataset) == n_test_sims * 6
        assert loaders[0].batch_size == 4

    def test_overrides_reach_the_loader(self, scored_run):
        from src.operators.eval import eval_all_seeds

        run_root, ckpt_path, n_test_sims, loaders = scored_run
        results = eval_all_seeds(
            str(run_root), device="cpu", n_snapshots_test=6, batch_size=1,
        )
        assert len(loaders[0].dataset) == n_test_sims * 15
        assert loaders[0].batch_size == 1
        assert math.isfinite(results[0]["test_gnrmse_pct"])
        assert math.isfinite(results[0]["test_node_jump_gnrmse_pct"])
        saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert saved["conf"]["training"]["n_snapshots_test"] == 4
        assert saved["conf"]["training"]["batch_size"] == 4

    def test_nonpositive_batch_size_is_rejected(self, scored_run):
        from src.operators.eval import eval_all_seeds

        run_root, _ckpt_path, _n, _loaders = scored_run
        with pytest.raises(ValueError, match="batch_size"):
            eval_all_seeds(str(run_root), device="cpu", batch_size=0)


class _MetricPairs(list):
    @property
    def _pairs(self):
        return [(i, 0, 1) for i in range(len(self))]


class _MetricPrediction(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, spatial, cond_static, forcing_seq=None):
        return spatial[..., :1] + self.offset


def _heterogeneous_metric_pairs():
    x = torch.linspace(0, 1, 6).reshape(6, 1, 1)
    y = torch.linspace(0, 1, 3).reshape(1, 3, 1)
    pairs = _MetricPairs()
    for amplitude, error, center in zip(
        [1.0, 3.0, 10.0, 30.0, 100.0],
        [8.0, 0.2, 4.0, 30.0, 1.0],
        [0.1, 0.2, 0.5, 0.6, 0.9],
    ):
        target = amplitude * (1 + x + y)
        pairs.append({
            "spatial": target + error * (1 + 2 * x + y),
            "Y": target,
            "cond_static": torch.zeros(2),
            "forcing_seq": torch.zeros(128, 3),
            "T_stats": torch.tensor([307.0, 5.0, center]),
        })
    return pairs


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("batch_size", [1, 2, 3, 5])
def test_region_metrics_pool_squared_errors(dynamic, batch_size):
    from src.operators.losses import build_boundary_mask, build_interface_mask
    from src.operators.train import validate
    from data.dataset import T_EPS

    dataset = _heterogeneous_metric_pairs()
    loader = DataLoader(dataset, batch_size=batch_size)
    grid = torch.linspace(0, 1, 6)
    y_grid = torch.linspace(0, 1, 3)
    iface = build_interface_mask(grid.numpy(), y_grid.numpy(), interface_half_width=0.12)
    boundary = build_boundary_mask(grid.numpy(), y_grid.numpy())
    model = _MetricPrediction()
    metrics = evaluate(
        model, loader, "cpu", iface_mask=iface, boundary_mask=boundary,
        x_grid=grid, interface_half_width=0.12, use_per_sample_interface=dynamic,
    )
    batch = next(iter(DataLoader(dataset, batch_size=len(dataset))))
    pred, target = batch["spatial"], batch["Y"]
    if dynamic:
        centers = batch["T_stats"][:, 2, None]
        band = (abs(grid[None, :] - centers) <= 0.12)[:, :, None, None]
    else:
        band = iface[None, :, :, None]
    for region, mask in [("iface", band), ("boundary", boundary[None, :, :, None])]:
        for space in ["norm", "phys"]:
            yp, yt = pred, target
            if space == "phys":
                yp, yt = yp * (5.0 + T_EPS) + 307.0, yt * (5.0 + T_EPS) + 307.0
            expected = 100 * torch.sqrt(((yp - yt).square() * mask).sum() / (yt.square() * mask).sum())
            assert metrics[f"{region}_rel_l2_{space}"] == pytest.approx(expected.item(), rel=1e-6)
    val = validate(
        model, loader, "cpu", iface_mask=iface, x_grid_t=grid,
        interface_half_width=0.12, use_per_sample_interface=dynamic,
    )
    assert metrics["iface_rel_l2_norm"] == pytest.approx(val["iface_rel_l2"], rel=1e-6)


@pytest.mark.parametrize("dynamic", [False, True])
def test_rollout_region_metrics_match_direct_pooling(monkeypatch, dynamic):
    from src.operators.losses import build_boundary_mask, build_interface_mask
    from src.operators.rollout import RolloutOptions

    dataset = _heterogeneous_metric_pairs()
    loader = DataLoader(dataset, batch_size=2)
    grid = torch.linspace(0, 1, 6).numpy()
    y_grid = torch.linspace(0, 1, 3).numpy()
    kwargs = dict(
        iface_mask=build_interface_mask(grid, y_grid, interface_half_width=0.12),
        boundary_mask=build_boundary_mask(grid, y_grid), x_grid=grid,
        interface_half_width=0.12, use_per_sample_interface=dynamic,
    )
    monkeypatch.setattr(
        "src.operators.eval.predict_autoregressive",
        lambda **kw: kw["dataset"][kw["sim_id"]]["spatial"].unsqueeze(0),
    )
    model = _MetricPrediction()
    direct = evaluate(model, loader, "cpu", **kwargs)
    rollout = evaluate(
        model, loader, "cpu", rollout_options=RolloutOptions(enabled=True, num_substeps=2),
        **kwargs,
    )
    for key, value in rollout.items():
        assert value == pytest.approx(direct[key], rel=1e-6)


def test_empty_regions_contribute_zero():
    mask = torch.zeros(6, 3, dtype=torch.bool)
    metrics = evaluate(
        _MetricPrediction(), DataLoader(_heterogeneous_metric_pairs(), batch_size=2),
        "cpu", iface_mask=mask, boundary_mask=mask,
    )
    for region in ["iface", "boundary"]:
        for space in ["norm", "phys"]:
            assert metrics[f"{region}_rel_l2_{space}"] == 0.0


def test_sigma_nrmse_and_gnrmse_have_distinct_names_and_definitions():
    from src.operators.losses import SpatiallyWeightedMSE
    from src.operators.train import train_one_epoch, validate

    dataset = _heterogeneous_metric_pairs()
    loader = DataLoader(dataset, batch_size=2)
    model = _MetricPrediction()
    grid = torch.linspace(0, 1, 6).numpy()
    y_grid = torch.linspace(0, 1, 3).numpy()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    train = train_one_epoch(
        model, loader, optimizer, SpatiallyWeightedMSE(grid, y_grid), "cpu",
        sigma_global=5.0,
    )
    val = validate(model, loader, "cpu", sigma_global=5.0)
    test = evaluate(model, loader, "cpu")
    errors = torch.stack([row["spatial"] - row["Y"] for row in dataset])
    pair_rms = errors.square().mean(dim=(1, 2, 3)).sqrt()
    for metrics in [train, val]:
        assert "gnrmse_pct" not in metrics
        assert metrics["sigma_nrmse_pct"] == pytest.approx(100 * pair_rms.mean().item())
    assert val["sigma_nrmse_pct_p99"] == pytest.approx(100 * torch.quantile(pair_rms, 0.99).item())
    assert "sigma_nrmse_pct" not in test
    expected = 100 * 5.0 * errors.square().mean().sqrt().item() / math.sqrt(5.0**2 + 7.0**2)
    assert test["gnrmse_pct"] == pytest.approx(expected, rel=1e-6)
    assert test["gnrmse_pct"] != pytest.approx(val["sigma_nrmse_pct"])


@pytest.mark.parametrize('options', [{}, {'long_lead_only': True}, {'rollout_num_substeps': 2}])
def test_combined_evaluation_reuses_predictions_and_preserves_outputs(records_run, monkeypatch, options):
    import json
    import src.operators.eval as evaluation

    run_root, seed_dir, n_test_sims = records_run
    checkpoint = torch.load(seed_dir / 'fno2d_best.pt', map_location='cpu')
    checkpoint.update(epoch=3, seed=42)
    torch.save(checkpoint, seed_dir / 'fno2d_best.pt')
    original_forward = evaluation.FNO2d.forward
    forward_calls = []

    def counted_forward(self, *args, **kwargs):
        forward_calls.append(args[0].shape[0])
        return original_forward(self, *args, **kwargs)

    monkeypatch.setattr(evaluation.FNO2d, 'forward', counted_forward)
    baseline = evaluation.eval_all_seeds(run_root, device='cpu', **options)
    expected_calls = list(forward_calls)
    forward_calls.clear()
    combined = evaluation.eval_all_seeds(run_root, device='cpu', write_records=True, **options)
    assert forward_calls == expected_calls
    assert len(combined) == 1
    for key, expected in baseline[0].items():
        actual = combined[0][key]
        if isinstance(expected, float):
            assert actual == pytest.approx(expected, nan_ok=True, rel=1e-6, abs=1e-7), key
        else:
            assert actual == expected, key
    rows = _read_records(seed_dir / 'test_records.csv')
    assert len(rows) == n_test_sims * (1 if options.get('long_lead_only') else 6)
    import pandas as pd
    from visual.pub.tables import _peak_frame
    peaks = _peak_frame(pd.read_csv(seed_dir / 'test_records.csv'))
    assert combined[0]['test_peak_jump_error_mean_K'] == pytest.approx(
        peaks.peak_jump_error_K.mean(), abs=1e-7
    )
    assert combined[0]['test_peak_jump_error_p95_K'] == pytest.approx(
        peaks.peak_jump_error_K.quantile(0.95), abs=1e-7
    )
    provenance = json.loads((seed_dir / 'test_records.provenance.json').read_text())
    assert provenance['n_pairs'] == len(rows)
    assert provenance['rollout_enabled'] == (options.get('rollout_num_substeps', 1) > 1)
    assert combined[0]['test_records'] == str(seed_dir / 'test_records.csv')
    assert {r['provenance_id'] for r in rows} == {provenance['provenance_id']}
    if not options.get('long_lead_only'):
        standalone = evaluation.write_test_records(
            run_root, seed=42, device='cpu', out_name='standalone.csv',
            inference_batch_size=4, **options,
        )
        for actual, expected in zip(rows, _read_records(standalone), strict=True):
            assert {k: v for k, v in actual.items() if k != 'provenance_id'} == {
                k: v for k, v in expected.items() if k != 'provenance_id'
            }


def test_run_eval_combined_cli_writes_report_and_all_seed_records(records_run, monkeypatch):
    import json
    from scripts import run_eval
    run_root, seed_dir, _ = records_run
    checkpoint = torch.load(seed_dir / 'fno2d_best.pt', map_location='cpu')
    checkpoint.update(epoch=3, seed=42)
    torch.save(checkpoint, seed_dir / 'fno2d_best.pt')
    other_seed = run_root / 'seed43'
    other_seed.mkdir()
    checkpoint['seed'] = 43
    torch.save(checkpoint, other_seed / 'fno2d_best.pt')
    monkeypatch.setattr('sys.argv', ['run_eval.py', str(run_root), '--write-test-records',
                                   '--device', 'cpu', '--rollout-num-substeps', '1',
                                   '--records-name', 'custom.csv', '--report-name', 'custom.json'])
    assert run_eval.main() == 0
    report = json.loads((run_root / 'custom.json').read_text())
    assert report
    for seed in (seed_dir, other_seed):
        assert _read_records(seed / 'custom.csv')
        assert (seed / 'custom.provenance.json').exists()


@pytest.mark.parametrize("batch_size", [1, 3, 5])
@pytest.mark.parametrize("dynamic", [False, True])
def test_peak_jump_errors_use_separate_initial_state_maxima(batch_size, dynamic):
    from data.dataset import T_EPS
    from src.operators.train import validate

    pairs = [(10, 0, 1), (20, 0, 1), (10, 1, 2), (10, 0, 2), (20, 0, 2)]
    predicted = [5., 1., 999., 2., 3.]
    truth = [1., 7., 0., 4., 2.]
    x_grid = torch.linspace(0, 1, 5)
    items = []
    for i, (pred, true) in enumerate(zip(predicted, truth)):
        interface = 0.625 if dynamic and pairs[i][0] == 20 else 0.375
        right = 3 if interface == 0.625 else 2
        spatial = torch.zeros(5, 2, 1)
        target = torch.zeros_like(spatial)
        spatial[right, :, 0] = -pred / (2.0 + T_EPS)
        target[right, :, 0] = -true / (2.0 + T_EPS)
        items.append(dict(spatial=spatial, Y=target, cond_static=torch.zeros(1),
                          forcing_seq=torch.empty(0, 0),
                          T_stats=torch.tensor([300., 2., interface])))
    class Pairs(list):
        _pairs = pairs
    dataset = Pairs(items)
    loader = DataLoader(dataset, batch_size=batch_size)
    model = _MetricPrediction()
    metrics = evaluate(model, loader, "cpu", x_grid=x_grid,
                       interface_x=0.375, use_per_sample_interface=dynamic)
    validation = validate(model, loader, "cpu", x_grid_t=x_grid,
                          interface_x=0.375, use_per_sample_interface=dynamic,
                          sigma_global=2.0)
    for result in (metrics, validation):
        assert result["peak_jump_error_mean_K"] == pytest.approx(2.5)
        assert result["peak_jump_error_p95_K"] == pytest.approx(3.85)


@pytest.mark.parametrize("missing_geometry", [False, True])
def test_peak_jump_unavailable_without_initial_state_pairs(missing_geometry):
    class LaterPairs(list):
        @property
        def _pairs(self):
            return [(i, 1, 2) for i in range(len(self))]
    dataset = LaterPairs(_heterogeneous_metric_pairs())
    loader = DataLoader(dataset, batch_size=2)
    result = evaluate(_MetricPrediction(), loader, "cpu",
                      x_grid=None if missing_geometry else torch.linspace(0, 1, 4))
    assert math.isnan(result["peak_jump_error_mean_K"])
    assert math.isnan(result["peak_jump_error_p95_K"])


def test_seed_report_aggregates_peak_errors_and_persists_them(tmp_path):
    import json
    from src.operators.eval import save_report

    rows = [_seed_result(i, 1., 1., 1., 1., 1.) for i in (1, 2)]
    for row, mean, p95 in zip(rows, [1., 3.], [2., 6.]):
        row.update(test_peak_jump_error_mean_K=mean, test_peak_jump_error_p95_K=p95)
    summary = print_seed_report(rows)
    save_report(tmp_path, rows, summary)
    report = json.loads((tmp_path / "seed_report.json").read_text())
    assert report["summary"]["test_peak_jump_error_mean_K_mean"] == 2.
    assert report["summary"]["test_peak_jump_error_p95_K_mean"] == 4.
    assert report["summary"]["test_peak_jump_error_mean_K_std"] == pytest.approx(2**0.5)
    assert report["per_seed"] == rows
