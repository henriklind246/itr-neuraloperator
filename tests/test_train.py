import copy
import csv
import math
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from data.dataset import SnapshotPairDataset
from src.operators.fno2d import FNO2d
from src.operators.losses import SpatiallyWeightedMSE, build_interface_mask

from src.operators.train import (
    _advance_scheduler,
    _per_pair_rel_l2_percent,
    _selection_metric_value,
    _validate_resume_compatibility,
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
    configure_trainable_scope,
    run_one_seed,
    run_config_seeds,
)
import src.operators.train as train_mod

# Active default = forcing benchmark, temporal_encoder representation:
# 4 spatial channels [T_tilde, x, y, s_y], 10 static cond dims, (128, 3) tokens.
SPATIAL_IN_CHANNELS = 4
COND_STATIC_DIM = 10
TEMPORAL_TOKEN_DIM = 3
TEMPORAL_SAMPLES = 128


class ZeroModel(torch.nn.Module):
    def forward(self, spatial, cond_static, forcing_seq):
        return torch.zeros(
            spatial.shape[0],
            spatial.shape[1],
            spatial.shape[2],
            1,
            dtype=spatial.dtype,
            device=spatial.device,
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

    def test_forcing_spatial_input_defaults_are_disabled(self):
        cfg = load_config()
        assert cfg["benchmark"]["spatial_input"] == {
            "itr": False,
            "lead_time": False,
            "material_side": False,
        }

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")

    def test_resolves_variables(self):
        cfg = load_config()
        traj_path = cfg["data"]["trajectories.npy"]
        assert "${" not in traj_path

    def test_resolves_environment_paths(self, tmp_path, monkeypatch):
        data_dir = tmp_path / "data"
        runs_root = tmp_path / "runs"
        monkeypatch.setenv("DATA_DIR", str(data_dir))
        monkeypatch.setenv("RUNS_ROOT", str(runs_root))

        cfg = load_config()

        assert cfg["paths"]["data_dir"] == str(data_dir)
        assert cfg["paths"]["runs_root"] == str(runs_root)
        assert cfg["data"]["trajectories.npy"] == str(data_dir / "trajectories.npy")
        assert cfg["training"]["run"]["run_dir"] == str(
            runs_root / "experiment0" / "config0"
        )

    def test_selects_benchmark_and_representation_from_environment(self, monkeypatch):
        monkeypatch.setenv("BENCHMARK", "interfaces")
        monkeypatch.setenv("REPRESENTATION", "temporal_encoder")

        cfg = load_config()

        assert cfg["benchmark"]["name"] == "interfaces"
        assert cfg["benchmark"]["representation"] == "temporal_encoder"
        assert cfg["training"]["loss"]["per_sample_interface_x"] is True


class TestResumeArchitectureCompatibility:
    @staticmethod
    def _config(mode="boundary_extender", hidden=16, interface_x=0.5):
        return {
            "model": {
                "parameters": {
                    "forcing_spatial_mode": mode,
                    "forcing_extender_physics_hidden": hidden,
                    "forcing_extender_interface_x_norm": interface_x,
                }
            },
            "training": {
                "optimizer": "AdamW",
                "scheduler": {"type": "StepLR"},
                "trainable_scope": "full",
            },
        }

    def test_resume_rejects_cross_mode_extender_state(self):
        with pytest.raises(ValueError, match="forcing_spatial_mode"):
            _validate_resume_compatibility(
                self._config("boundary_extender"),
                self._config("physics_extender"),
            )

    @pytest.mark.parametrize(("hidden", "interface_x"), [(24, 0.5), (16, 0.45)])
    def test_resume_rejects_changed_physics_extender_signature(
        self, hidden, interface_x
    ):
        with pytest.raises(ValueError, match="physics_extender_signature"):
            _validate_resume_compatibility(
                self._config("physics_extender"),
                self._config("physics_extender", hidden, interface_x),
            )


# ===================== helpers for training tests =====================

_ITEM_KEYS = ("spatial", "cond_static", "forcing_seq", "Y", "T_stats")


def _dict_collate(batch):
    """Stack a list of (spatial, cond_static, forcing_seq, Y, T_stats) tuples
    into the dict batch contract the train/val loops now consume."""
    return {
        k: torch.stack([item[i] for item in batch])
        for i, k in enumerate(_ITEM_KEYS)
    }


def _make_dict_loader(Nx=11, Ny=11, n_samples=4, batch_size=2):
    """Create a DataLoader yielding dict batches with the forcing item keys."""
    x_spatial = torch.randn(n_samples, Nx, Ny, SPATIAL_IN_CHANNELS)
    cond_static = torch.rand(n_samples, COND_STATIC_DIM)
    forcing_seq = torch.randn(n_samples, TEMPORAL_SAMPLES, TEMPORAL_TOKEN_DIM)
    Y = torch.randn(n_samples, Nx, Ny, 1)
    T_stats = torch.randn(n_samples, 2)
    return DataLoader(
        TensorDataset(x_spatial, cond_static, forcing_seq, Y, T_stats),
        batch_size=batch_size,
        collate_fn=_dict_collate,
    )


def _make_tiny_fno():
    return FNO2d(
        modes1=2, modes2=2, width=8,
        in_channels=SPATIAL_IN_CHANNELS, out_channels=1, n_layers=2,
        cond_static_dim=COND_STATIC_DIM,
        temporal_token_dim=TEMPORAL_TOKEN_DIM,
        temporal_hidden=16,
        forcing_embed_dim=16,
    )


def _make_tiny_extender_fno():
    return FNO2d(
        modes1=2, modes2=2, width=8,
        in_channels=SPATIAL_IN_CHANNELS, out_channels=1, n_layers=2,
        cond_static_dim=COND_STATIC_DIM,
        temporal_token_dim=TEMPORAL_TOKEN_DIM,
        temporal_hidden=16, forcing_embed_dim=16, forcing_spatial_dim=8,
        use_forcing_time_aug=True,
        forcing_cond_mode="spatial_only",
        forcing_spatial_mode="boundary_extender",
        forcing_extender_grid_size=8,
        forcing_extender_heads=4,
        forcing_extender_depth=2,
    )


def _make_tiny_physics_extender_fno():
    return FNO2d(
        modes1=2, modes2=2, width=8,
        in_channels=SPATIAL_IN_CHANNELS, out_channels=1, n_layers=2,
        cond_static_dim=COND_STATIC_DIM,
        temporal_token_dim=TEMPORAL_TOKEN_DIM,
        temporal_hidden=16, forcing_embed_dim=16, forcing_spatial_dim=8,
        use_forcing_time_aug=True,
        forcing_cond_mode="spatial_only",
        forcing_spatial_mode="physics_extender",
        forcing_extender_grid_size=8,
        forcing_extender_heads=4,
        forcing_extender_depth=2,
        forcing_extender_rc_cond_index=1,
    )


class TestTrainableScope:
    def test_full_scope_leaves_every_parameter_trainable(self):
        model = _make_tiny_extender_fno()
        trainable, total = configure_trainable_scope(model, "full")
        assert trainable == total
        assert all(parameter.requires_grad for parameter in model.parameters())

    def test_boundary_extender_scope_changes_only_extender_weights(self):
        model = _make_tiny_extender_fno()
        trainable, total = configure_trainable_scope(model, "boundary_extender")
        assert 0 < trainable < total
        assert {
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        } == {
            name for name, _ in model.named_parameters()
            if name.startswith("boundary_extender.")
        }

        before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=1e-3,
        )
        prediction = model(
            torch.randn(2, 11, 11, SPATIAL_IN_CHANNELS),
            torch.randn(2, COND_STATIC_DIM),
            torch.randn(2, 16, TEMPORAL_TOKEN_DIM),
        )
        loss = (prediction - torch.randn_like(prediction)).square().mean()
        loss.backward()
        optimizer.step()

        changed = {
            name for name, parameter in model.named_parameters()
            if not torch.equal(before[name], parameter)
        }
        assert changed
        assert all(name.startswith("boundary_extender.") for name in changed)

    def test_boundary_extender_scope_includes_physics_bias(self):
        model = _make_tiny_physics_extender_fno()
        trainable, total = configure_trainable_scope(model, "boundary_extender")
        trainable_names = {
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

        assert 0 < trainable < total
        assert trainable_names
        assert all(name.startswith("boundary_extender.") for name in trainable_names)
        assert any("diffusion_geometry_bias_mlps" in name for name in trainable_names)

    def test_boundary_extender_scope_requires_that_module(self):
        with pytest.raises(ValueError, match="forcing_spatial_mode"):
            configure_trainable_scope(_make_tiny_fno(), "boundary_extender")

    def test_unknown_scope_is_rejected(self):
        with pytest.raises(ValueError, match="trainable_scope"):
            configure_trainable_scope(_make_tiny_extender_fno(), "head")


@pytest.fixture
def tiny_training_setup():
    """Tiny model + synthetic 5-tuple dataloader for fast training tests."""
    Nx = 11
    Ny = 11
    model = _make_tiny_fno()
    loader = _make_dict_loader(Nx=Nx, Ny=Ny)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    x_grid_np = np.linspace(0.0, 1.0, Nx).astype(np.float32)
    y_grid_np = np.linspace(0.0, 1.0, Ny).astype(np.float32)
    loss_fn = SpatiallyWeightedMSE(x_grid_np, y_grid_np, interface_weight=1.0)
    iface_mask = build_interface_mask(x_grid_np, y_grid_np)
    device = torch.device("cpu")

    return model, loader, optimizer, loss_fn, device, iface_mask


# ===================== train_one_epoch =====================

class TestTrainOneEpoch:
    def test_returns_metric_dict(self, tiny_training_setup):
        model, loader, optimizer, loss_fn, device, iface_mask = tiny_training_setup
        result = train_one_epoch(model, loader, optimizer, loss_fn, device, iface_mask=iface_mask)
        assert isinstance(result, dict)
        expected = {
            "loss", "rel_l2", "iface_rel_l2", "nrmse", "rmse_K",
            "max_err_K", "node_jump_rmse_K", "node_jump_nrmse",
        }
        assert expected.issubset(result)
        for k in expected:
            assert isinstance(result[k], float)

    def test_loss_is_finite(self, tiny_training_setup):
        model, loader, optimizer, loss_fn, device, iface_mask = tiny_training_setup
        result = train_one_epoch(model, loader, optimizer, loss_fn, device, iface_mask=iface_mask)
        assert np.isfinite(result["loss"])
        assert np.isfinite(result["rel_l2"])
        assert np.isfinite(result["iface_rel_l2"])
        assert np.isfinite(result["nrmse"])
        assert np.isfinite(result["rmse_K"])

    def test_loss_is_nonnegative(self, tiny_training_setup):
        model, loader, optimizer, loss_fn, device, iface_mask = tiny_training_setup
        result = train_one_epoch(model, loader, optimizer, loss_fn, device, iface_mask=iface_mask)
        assert result["loss"] >= 0
        assert result["rel_l2"] >= 0
        assert result["iface_rel_l2"] >= 0
        assert result["nrmse"] >= 0

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
        result = train_one_epoch(model, loader, optimizer, loss_fn, device)
        assert result["iface_rel_l2"] == 0.0

    def test_grad_clip_runs_without_error(self, tiny_training_setup):
        model, loader, optimizer, loss_fn, device, iface_mask = tiny_training_setup
        result = train_one_epoch(
            model, loader, optimizer, loss_fn, device,
            iface_mask=iface_mask, grad_clip=0.01,
        )
        assert np.isfinite(result["loss"])

    def test_grad_clip_none_is_noop(self, tiny_training_setup):
        model, loader, optimizer, loss_fn, device, iface_mask = tiny_training_setup
        result = train_one_epoch(
            model, loader, optimizer, loss_fn, device,
            iface_mask=iface_mask, grad_clip=None,
        )
        assert np.isfinite(result["loss"])


# ===================== validate =====================

class TestValidate:
    def test_per_pair_rel_l2_uses_rms_over_rms(self):
        y_true = torch.arange(1.0, 7.0).view(1, 2, 3, 1)
        y_pred = torch.zeros_like(y_true)

        rel_l2 = _per_pair_rel_l2_percent(y_pred, y_true)

        assert rel_l2.item() == pytest.approx(100.0)

    def test_returns_metric_dict(self, tiny_training_setup):
        model, loader, _, _, device, iface_mask = tiny_training_setup
        result = validate(model, loader, device, iface_mask=iface_mask)
        assert isinstance(result, dict)
        expected = {
            "rel_l2", "iface_rel_l2", "nrmse", "nrmse_p90", "nrmse_p99",
            "nrmse_max", "rmse_K", "max_err_K", "node_jump_rmse_K",
            "node_jump_nrmse",
        }
        assert expected.issubset(result)
        for k in expected:
            assert isinstance(result[k], float)

    def test_loss_is_nonnegative(self, tiny_training_setup):
        model, loader, _, _, device, iface_mask = tiny_training_setup
        result = validate(model, loader, device, iface_mask=iface_mask)
        assert result["rel_l2"] >= 0
        assert result["iface_rel_l2"] >= 0
        assert result["nrmse"] >= 0

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

    def test_writes_per_pair_validation_csv(self, tmp_path, synthetic_trajectories, synthetic_sim_params):
        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        dataset = SnapshotPairDataset(
            trajectories=trajectories,
            t_grid=t_grid,
            x_grid=x_grid,
            y_grid=y_grid,
            sim_ids=np.array([0]),
            sim_params=synthetic_sim_params,
            mu_global=0.0,
            sigma_global=1.0,
            n_snapshots=3,
        )
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        model = _make_tiny_fno()
        iface_mask = build_interface_mask(x_grid, y_grid)
        csv_path = tmp_path / "val_pairs.csv"

        validate(
            model,
            loader,
            torch.device("cpu"),
            iface_mask=iface_mask,
            dataset=dataset,
            pair_csv_path=csv_path,
            epoch=7,
        )

        with csv_path.open("r", newline="") as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == len(dataset)
        # Header is the full superset (union of every benchmark's columns).
        assert set(rows[0]) == {
            "epoch",
            "sim_id",
            "benchmark",
            "t_s",
            "t_bar",
            "R_c",
            "rel_l2",
            "iface_rel_l2",
            "nrmse",
            "rmse_K",
            "sigma_nrmse_pct",
            "node_jump_rmse_K",
            "node_jump_nrmse",
            "node_jump_gnrmse_pct",
            "A",
            "interface_x",
            "regime",
            "spatial_family",
            "temporal_family",
            "x_h",
            "y_h",
            "R_c_A",
            "R_c_A",
            "R_c_base",
        }
        assert {int(row["epoch"]) for row in rows} == {7}
        # The fixture is a forcing-style dataset, so only forcing's extra
        # columns are populated; every other benchmark's columns stay empty.
        for row in rows:
            assert row["benchmark"] == "forcing"
            assert row["temporal_family"] != ""
            assert row["spatial_family"] != ""
            for empty_col in (
                "A",
                "interface_x",
                "regime",
                "x_h",
                "y_h",
                "R_c_A",
                "R_c_A",
            ):
                assert row[empty_col] == ""

    @pytest.mark.parametrize("legacy_header", [False, True])
    def test_per_pair_csv_metrics_use_normalized_tensors(self, tmp_path, synthetic_trajectories, synthetic_sim_params, legacy_header):
        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        dataset = SnapshotPairDataset(
            trajectories=trajectories,
            t_grid=t_grid,
            x_grid=x_grid,
            y_grid=y_grid,
            sim_ids=np.array([0]),
            sim_params=synthetic_sim_params,
            mu_global=10.0,
            sigma_global=2.0,
            n_snapshots=3,
        )
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        iface_mask = build_interface_mask(x_grid, y_grid)
        csv_path = tmp_path / "val_pairs.csv"

        if legacy_header:
            from src.operators.train import VAL_PAIR_FIELDNAMES
            with csv_path.open("w", newline="") as f:
                csv.writer(f).writerow([
                    "gnrmse_pct" if key == "sigma_nrmse_pct" else key
                    for key in VAL_PAIR_FIELDNAMES
                ])

        validate(
            ZeroModel(),
            loader,
            torch.device("cpu"),
            iface_mask=iface_mask,
            dataset=dataset,
            pair_csv_path=csv_path,
            epoch=7,
            sigma_global=2.0,
        )

        with csv_path.open("r", newline="") as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == len(dataset)
        for row in rows:
            assert float(row["rel_l2"]) == pytest.approx(100.0)
            assert float(row["iface_rel_l2"]) == pytest.approx(100.0)
            key = "gnrmse_pct" if legacy_header else "sigma_nrmse_pct"
            assert float(row[key]) == pytest.approx(float(row["rmse_K"]) / 2.0 * 100)

    def _capped_dataset(self, synthetic_trajectories, synthetic_sim_params):
        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        dataset = SnapshotPairDataset(
            trajectories=trajectories,
            t_grid=t_grid,
            x_grid=x_grid,
            y_grid=y_grid,
            sim_ids=np.array([0, 1]),
            sim_params=synthetic_sim_params,
            mu_global=0.0,
            sigma_global=1.0,
            n_snapshots=4,
        )
        return dataset, x_grid, y_grid

    def test_val_pairs_max_rows_caps_written_rows(self, tmp_path, synthetic_trajectories, synthetic_sim_params):
        dataset, x_grid, y_grid = self._capped_dataset(synthetic_trajectories, synthetic_sim_params)
        cap = 5
        assert cap < len(dataset)
        loader = DataLoader(dataset, batch_size=3, shuffle=False)
        csv_path = tmp_path / "val_pairs.csv"

        validate(
            _make_tiny_fno(),
            loader,
            torch.device("cpu"),
            iface_mask=build_interface_mask(x_grid, y_grid),
            dataset=dataset,
            pair_csv_path=csv_path,
            val_pairs_max_rows=cap,
            epoch=3,
        )

        with csv_path.open("r", newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == cap

    def test_val_pairs_cap_does_not_change_metrics(self, tmp_path, synthetic_trajectories, synthetic_sim_params):
        dataset, x_grid, y_grid = self._capped_dataset(synthetic_trajectories, synthetic_sim_params)
        iface_mask = build_interface_mask(x_grid, y_grid)
        model = _make_tiny_fno()

        def run(cap, name):
            loader = DataLoader(dataset, batch_size=3, shuffle=False)
            return validate(
                model,
                loader,
                torch.device("cpu"),
                iface_mask=iface_mask,
                x_grid_t=torch.as_tensor(x_grid),
                dataset=dataset,
                pair_csv_path=tmp_path / name,
                val_pairs_max_rows=cap,
                epoch=3,
            )

        uncapped = run(None, "uncapped.csv")
        capped = run(5, "capped.csv")

        assert set(uncapped) == set(capped)
        for k in uncapped:
            assert capped[k] == pytest.approx(uncapped[k])

    def test_val_pairs_subset_stable_across_passes(self, tmp_path, synthetic_trajectories, synthetic_sim_params):
        dataset, x_grid, y_grid = self._capped_dataset(synthetic_trajectories, synthetic_sim_params)
        iface_mask = build_interface_mask(x_grid, y_grid)
        model = _make_tiny_fno()

        def written_identities(name):
            loader = DataLoader(dataset, batch_size=3, shuffle=False)
            csv_path = tmp_path / name
            validate(
                model,
                loader,
                torch.device("cpu"),
                iface_mask=iface_mask,
                dataset=dataset,
                pair_csv_path=csv_path,
                val_pairs_max_rows=5,
                epoch=3,
            )
            with csv_path.open("r", newline="") as f:
                rows = list(csv.DictReader(f))
            # (sim_id, t_s, t_bar) maps 1:1 to (sim_id, s, j) since t_grid is monotonic.
            return {(r["sim_id"], r["t_s"], r["t_bar"]) for r in rows}

        first = written_identities("pass1.csv")
        second = written_identities("pass2.csv")
        assert len(first) == 5
        assert first == second

    def test_val_pairs_cap_geq_total_writes_all(self, tmp_path, synthetic_trajectories, synthetic_sim_params):
        dataset, x_grid, y_grid = self._capped_dataset(synthetic_trajectories, synthetic_sim_params)
        loader = DataLoader(dataset, batch_size=3, shuffle=False)
        csv_path = tmp_path / "val_pairs.csv"

        validate(
            _make_tiny_fno(),
            loader,
            torch.device("cpu"),
            iface_mask=build_interface_mask(x_grid, y_grid),
            dataset=dataset,
            pair_csv_path=csv_path,
            val_pairs_max_rows=len(dataset) + 100,
            epoch=3,
        )

        with csv_path.open("r", newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == len(dataset)

    def test_val_pairs_max_rows_rejects_nonpositive(self, tmp_path, synthetic_trajectories, synthetic_sim_params):
        dataset, x_grid, y_grid = self._capped_dataset(synthetic_trajectories, synthetic_sim_params)
        loader = DataLoader(dataset, batch_size=3, shuffle=False)
        with pytest.raises(ValueError):
            validate(
                _make_tiny_fno(),
                loader,
                torch.device("cpu"),
                iface_mask=build_interface_mask(x_grid, y_grid),
                dataset=dataset,
                pair_csv_path=tmp_path / "val_pairs.csv",
                val_pairs_max_rows=0,
                epoch=3,
            )


# ===================== per-sample interface metric =====================

class TestPerSampleInterfaceMetric:
    """interfaces benchmark: loss/metric follow each sample's interface_x."""

    def _setup(self, interface_x_value=0.5, n=4, Nx=11, Ny=11):
        model = _make_tiny_fno()
        x_spatial = torch.randn(n, Nx, Ny, SPATIAL_IN_CHANNELS)
        cond_static = torch.rand(n, COND_STATIC_DIM)
        forcing_seq = torch.randn(n, TEMPORAL_SAMPLES, TEMPORAL_TOKEN_DIM)
        Y = torch.randn(n, Nx, Ny, 1)
        # 3-column T_stats: [mu, sigma, interface_x] (interfaces/source schema)
        T_stats = torch.stack(
            [torch.zeros(n), torch.ones(n), torch.full((n,), float(interface_x_value))],
            dim=-1,
        )
        loader = DataLoader(
            TensorDataset(x_spatial, cond_static, forcing_seq, Y, T_stats),
            batch_size=2,
            collate_fn=_dict_collate,
        )
        x_grid = np.linspace(0.0, 1.0, Nx).astype(np.float32)
        y_grid = np.linspace(0.0, 1.0, Ny).astype(np.float32)
        loss_fn = SpatiallyWeightedMSE(
            x_grid, y_grid, interface_x=0.5, interface_half_width=0.05, interface_weight=1.0
        )
        iface_mask = build_interface_mask(x_grid, y_grid, 0.5, 0.05)
        return model, loader, loss_fn, iface_mask, torch.from_numpy(x_grid)

    def test_validate_persample_matches_fixed_when_all_half(self):
        """All interface_x == 0.5: per-sample band metric equals the fixed-mask metric."""
        model, loader, _, iface_mask, x_grid_t = self._setup(0.5)
        device = torch.device("cpu")
        fixed = validate(model, loader, device, iface_mask=iface_mask)["iface_rel_l2"]
        dyn = validate(
            model, loader, device, iface_mask=iface_mask,
            x_grid_t=x_grid_t, interface_half_width=0.05, use_per_sample_interface=True,
        )["iface_rel_l2"]
        assert dyn == pytest.approx(fixed, rel=1e-5)

    def test_validate_persample_differs_when_off_center(self):
        """interface_x == 0.2 must drive the metric off the fixed x=0.5 band."""
        model, loader, _, iface_mask, x_grid_t = self._setup(0.2)
        device = torch.device("cpu")
        fixed = validate(model, loader, device, iface_mask=iface_mask)["iface_rel_l2"]
        dyn = validate(
            model, loader, device, iface_mask=iface_mask,
            x_grid_t=x_grid_t, interface_half_width=0.05, use_per_sample_interface=True,
        )["iface_rel_l2"]
        assert dyn != pytest.approx(fixed, rel=1e-6)

    def test_train_one_epoch_persample_matches_fixed_when_all_half(self):
        """lr=0 freezes params: per-sample masked-sum metric == boolean-slice metric at 0.5."""
        model, loader, loss_fn, iface_mask, _ = self._setup(0.5)
        device = torch.device("cpu")
        opt = torch.optim.SGD(model.parameters(), lr=0.0)
        fixed = train_one_epoch(
            model, loader, opt, loss_fn, device,
            iface_mask=iface_mask, use_per_sample_interface=False,
        )["iface_rel_l2"]
        dyn = train_one_epoch(
            model, loader, opt, loss_fn, device,
            iface_mask=iface_mask, use_per_sample_interface=True,
        )["iface_rel_l2"]
        assert dyn == pytest.approx(fixed, rel=1e-5)


# ===================== per-sample interface config flag =====================

class TestPerSampleInterfaceConfig:
    """Guard the config gating so a refactor can't silently disable the fix."""

    def _load_loss_cfg(self, rel):
        import yaml
        path = Path(__file__).resolve().parents[1] / rel
        with path.open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cfg["training"]["loss"]

    def test_interfaces_enables_per_sample(self):
        assert self._load_loss_cfg("conf/benchmark/interfaces.yaml")["per_sample_interface_x"] is True

    def test_forcing_disables_per_sample(self):
        assert self._load_loss_cfg("conf/benchmark/forcing.yaml")["per_sample_interface_x"] is False

    def test_source_disables_per_sample(self):
        assert self._load_loss_cfg("conf/benchmark/source.yaml")["per_sample_interface_x"] is False

    def test_code_default_off_keeps_fixed_path(self):
        """Code default (flag off) returns None even when slot 2 exists (source case)."""
        from src.operators.losses import get_batch_interface_x
        batch = {"T_stats": torch.tensor([[0.0, 1.0, 0.5]])}
        assert get_batch_interface_x(batch, torch.device("cpu"), use_per_sample_interface=False) is None


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
        model = _make_tiny_fno()
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
        model = _make_tiny_fno()
        optimizer = build_optimizer(config, model.parameters())

        with pytest.raises(ValueError, match="scheduler.peak_lr must match training.learning_rate"):
            build_scheduler(config, optimizer)


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
        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        np.save(tmp_path / "trajectories.npy", trajectories)
        np.save(tmp_path / "x_grid.npy", x_grid)
        np.save(tmp_path / "y_grid.npy", y_grid)
        np.save(tmp_path / "t_grid.npy", t_grid)
        np.save(tmp_path / "sim_params.npy", synthetic_sim_params)

        return {
            "data": {
                "trajectories.npy": str(tmp_path / "trajectories.npy"),
                "x_grid_path": str(tmp_path / "x_grid.npy"),
                "y_grid_path": str(tmp_path / "y_grid.npy"),
                "t_grid_path": str(tmp_path / "t_grid.npy"),
                "sim_params_path": str(tmp_path / "sim_params.npy"),
            },
            "model": {
                "parameters": {
                    "modes1": 2,
                    "modes2": 2,
                    "width": 8,
                    "in_channels": SPATIAL_IN_CHANNELS,
                    "out_channels": 1,
                    "n_layers": 2,
                    "cond_static_dim": COND_STATIC_DIM,
                    "cond_hidden": 32,
                    "temporal_token_dim": TEMPORAL_TOKEN_DIM,
                    "temporal_samples": TEMPORAL_SAMPLES,
                    "temporal_hidden": 16,
                    "forcing_embed_dim": 16,
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

    def test_physics_extender_run_writes_best_checkpoint_bias_grid(
        self, tmp_path, seed_config
    ):
        cfg = copy.deepcopy(seed_config)
        cfg["model"]["parameters"].update(
            {
                "forcing_spatial_mode": "physics_extender",
                "forcing_extender_grid_size": 4,
                "forcing_extender_heads": 4,
                "forcing_extender_depth": 2,
                "forcing_extender_physics_hidden": 8,
                "forcing_extender_interface_x_norm": 0.5,
            }
        )
        run_dir = tmp_path / "seed0_physics"
        run_one_seed(cfg, seed=0, run_dir=run_dir)

        checkpoint = torch.load(
            run_dir / "fno2d_best.pt", map_location="cpu", weights_only=False
        )
        with (run_dir / "diffusion_geometry_bias.csv").open(
            "r", newline="", encoding="utf-8"
        ) as csv_file:
            rows = list(csv.DictReader(csv_file))

        assert len(rows) == 6 * 5 * 3 * 2 * 4
        assert set(rows[0]) == {
            "checkpoint_epoch", "block", "head", "x_norm", "t_bar_norm",
            "I_cross", "R_c_norm", "c",
        }
        assert {int(row["checkpoint_epoch"]) for row in rows} == {
            int(checkpoint["epoch"])
        }
        assert {int(row["I_cross"]) for row in rows} == {0, 1}

    def test_train_metrics_csv_has_new_metric_columns(self, tmp_path, seed_config):
        """End-to-end run emits the new nRMSE / Kelvin / node-jump columns, finite."""
        run_dir = tmp_path / "seed0"
        run_one_seed(seed_config, seed=0, run_dir=run_dir)

        csv_path = run_dir / "train_metrics.csv"
        with csv_path.open("r", newline="") as f:
            rows = list(csv.DictReader(f))
        assert rows

        train_cols = (
            "train_nrmse", "train_rmse_K", "train_max_err_K",
            "train_node_jump_rmse_K", "train_node_jump_nrmse",
            "train_sigma_nrmse_pct", "train_node_jump_gnrmse_pct",
        )
        for col in train_cols:
            assert col in rows[0], col
            assert math.isfinite(float(rows[0][col])), col

        # Validation columns are only populated on validation epochs.
        val_rows = [r for r in rows if r.get("val_nrmse", "") != ""]
        assert val_rows
        val_cols = (
            "val_nrmse", "val_nrmse_p50", "val_nrmse_iqr",
            "val_nrmse_p90", "val_nrmse_p99", "val_nrmse_max",
            "val_rmse_K", "val_rmse_K_p90", "val_rmse_K_p99", "val_rmse_K_max",
            "val_sigma_nrmse_pct", "val_sigma_nrmse_pct_p99", "val_max_err_K",
            "val_node_jump_rmse_K", "val_node_jump_nrmse",
            "val_node_jump_nrmse_p90", "val_node_jump_nrmse_p99", "val_node_jump_nrmse_max",
            "val_node_jump_gnrmse_pct", "val_node_jump_gnrmse_pct_p99",
        )
        for col in val_cols:
            assert col in val_rows[0], col
            assert math.isfinite(float(val_rows[0][col])), col

    def test_checkpoint_metric_nrmse_selects_on_nrmse(self, tmp_path, seed_config):
        """Setting checkpoint_metric=val_nrmse drives best_val selection off nRMSE."""
        cfg = copy.deepcopy(seed_config)
        cfg["training"]["checkpoint_metric"] = "val_nrmse"
        run_dir = tmp_path / "seed0_nrmse"
        result = run_one_seed(cfg, seed=0, run_dir=run_dir)

        ckpt = torch.load(run_dir / "fno2d_best.pt", map_location="cpu", weights_only=False)
        best_epoch = int(ckpt["epoch"])

        csv_path = run_dir / "train_metrics.csv"
        with csv_path.open("r", newline="") as f:
            rows = list(csv.DictReader(f))
        val_rows = [r for r in rows if r.get("val_nrmse", "") != ""]
        best_val_nrmse = min(float(r["val_nrmse"]) for r in val_rows)
        chosen = next(r for r in rows if int(r["epoch"]) == best_epoch)
        assert float(chosen["val_nrmse"]) == pytest.approx(best_val_nrmse, rel=1e-6)
        assert result["best_val"] == pytest.approx(best_val_nrmse, rel=1e-6)

    def test_selection_metric_value_rejects_unknown_metric(self):
        """A typo'd checkpoint_metric raises instead of silently using rel_l2."""
        val_metrics = {"rel_l2": 1.0, "nrmse": 2.0, "node_jump_nrmse": 3.0}
        with pytest.raises(ValueError, match="unknown training.checkpoint_metric"):
            _selection_metric_value(val_metrics, "val_nrsme")
        # Known metrics still resolve to the mapped key.
        assert _selection_metric_value(val_metrics, "val_nrmse") == 2.0
        assert _selection_metric_value(val_metrics, "val_rel_l2") == 1.0

    def test_run_one_seed_rejects_unknown_checkpoint_metric(self, tmp_path, seed_config):
        """run_one_seed fails fast (before training) on a bad checkpoint_metric."""
        cfg = copy.deepcopy(seed_config)
        cfg["training"]["checkpoint_metric"] = "val_bogus"
        run_dir = tmp_path / "seed0_bogus"
        with pytest.raises(ValueError, match="unknown training.checkpoint_metric"):
            run_one_seed(cfg, seed=0, run_dir=run_dir)
        # Fail-fast: no checkpoint should have been written.
        assert not (run_dir / "fno2d_best.pt").exists()

    def test_fresh_run_writes_no_diagnostics_csv(self, tmp_path, seed_config):
        """diagnostics.csv must not be written; val_pairs.csv must still be.

        A stale diagnostics block is injected to confirm the logic is gone, not
        merely disabled by config.
        """
        run_dir = tmp_path / "seed0"
        cfg = copy.deepcopy(seed_config)
        cfg["training"]["diagnostics"] = {
            "enabled": True,
            "every": 1,
            "run_first_epoch": True,
            "activation_batches": 2,
            "sensitivity_batches": 2,
        }

        run_one_seed(cfg, seed=0, run_dir=run_dir)

        assert not (run_dir / "diagnostics.csv").exists()
        assert not (run_dir / "diffusion_geometry_bias.csv").exists()
        assert (run_dir / "val_pairs.csv").exists()

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
        assert (run_dir / "fno2d_best.pt").exists()

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
        assert (run_dir / "fno2d_best.pt").exists()

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
        assert (run_dir / "fno2d_best.pt").exists()

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

    @pytest.mark.parametrize("legacy_metrics", [False, True])
    def test_csv_epochs_continue_after_resume(self, tmp_path, seed_config, legacy_metrics):
        """Resumed epochs should start after the last epoch in the existing CSV."""
        run_dir = tmp_path / "seed0"

        short_config = {**seed_config, "training": {**seed_config["training"], "epochs": 2}}
        run_one_seed(short_config, seed=0, run_dir=run_dir)

        best_ckpt = torch.load(run_dir / "fno2d_best.pt", map_location="cpu", weights_only=False)
        torch.save(best_ckpt, run_dir / "fno2d_latest.pt")

        retained = {}
        for name in ("train_metrics.csv", "val_pairs.csv"):
            path = run_dir / name
            with path.open(newline="") as f:
                retained[name] = next(csv.DictReader(f))
            if legacy_metrics:
                path.write_text(path.read_text().replace("sigma_nrmse", "gnrmse"))

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
        for name, expected in retained.items():
            with (run_dir / name).open(newline="") as f:
                row = next(csv.DictReader(f))
            for key in expected:
                if "sigma_nrmse" in key:
                    assert row[key] == expected[key]
                    assert key.replace("sigma_nrmse", "gnrmse") not in row

    def test_resume_rejects_optimizer_scheduler_mismatch(self, tmp_path, seed_config):
        """Resuming should fail clearly when optimizer or scheduler family changes."""
        run_dir = tmp_path / "seed0_mismatch"

        short_config = {**seed_config, "training": {**seed_config["training"], "epochs": 2}}
        run_one_seed(short_config, seed=0, run_dir=run_dir)

        best_ckpt = torch.load(run_dir / "fno2d_best.pt", map_location="cpu", weights_only=False)
        torch.save(best_ckpt, run_dir / "fno2d_latest.pt")

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

        with pytest.raises(ValueError, match="fresh run directory|remove fno2d_latest.pt"):
            run_one_seed(mismatch_config, seed=0, run_dir=run_dir)


# ===================== run_config_seeds =====================

class TestRunConfigSeeds:
    """Test multi-seed orchestration with skip/resume."""

    @pytest.fixture
    def seed_config(self, tmp_path, synthetic_trajectories, synthetic_sim_params):
        """Build a minimal config pointing at synthetic data files."""
        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        np.save(tmp_path / "trajectories.npy", trajectories)
        np.save(tmp_path / "x_grid.npy", x_grid)
        np.save(tmp_path / "y_grid.npy", y_grid)
        np.save(tmp_path / "t_grid.npy", t_grid)
        np.save(tmp_path / "sim_params.npy", synthetic_sim_params)

        return {
            "data": {
                "trajectories.npy": str(tmp_path / "trajectories.npy"),
                "x_grid_path": str(tmp_path / "x_grid.npy"),
                "y_grid_path": str(tmp_path / "y_grid.npy"),
                "t_grid_path": str(tmp_path / "t_grid.npy"),
                "sim_params_path": str(tmp_path / "sim_params.npy"),
            },
            "model": {
                "parameters": {
                    "modes1": 2,
                    "modes2": 2,
                    "width": 8,
                    "in_channels": SPATIAL_IN_CHANNELS,
                    "out_channels": 1,
                    "n_layers": 2,
                    "cond_static_dim": COND_STATIC_DIM,
                    "cond_hidden": 32,
                    "temporal_token_dim": TEMPORAL_TOKEN_DIM,
                    "temporal_samples": TEMPORAL_SAMPLES,
                    "temporal_hidden": 16,
                    "forcing_embed_dim": 16,
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


@pytest.mark.parametrize("legacy", [False, True])
def test_metrics_summary_preserves_sigma_definition(tmp_path, legacy):
    from src.operators.train import _summarize_metrics_csv

    row = {
        "epoch": 0, "is_best": 1,
        "train_sigma_nrmse_pct": 2.5,
        "val_sigma_nrmse_pct": 4.5,
        "val_sigma_nrmse_pct_p99": 9.0,
        "val_node_jump_gnrmse_pct": 6.0,
    }
    if legacy:
        row = {key.replace("sigma_nrmse", "gnrmse"): value for key, value in row.items()}
    path = tmp_path / "train_metrics.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    summary = _summarize_metrics_csv(path)
    for key in ("best_row", "final_row"):
        assert summary[key]["train_sigma_nrmse_pct"] == 2.5
        assert summary[key]["val_sigma_nrmse_pct"] == 4.5
        assert summary[key]["val_sigma_nrmse_pct_p99"] == 9.0
        assert summary[key]["val_node_jump_gnrmse_pct"] == 6.0
        assert "val_gnrmse_pct" not in summary[key]


@pytest.mark.parametrize("legacy", [False, True])
def test_final_metrics_preserve_best_and_final_peak_errors(tmp_path, legacy):
    import json
    from src.operators.train import _write_final_metrics

    rows = [dict(epoch=0, is_best=1), dict(epoch=1, is_best=0)]
    if not legacy:
        rows[0].update(val_peak_jump_error_mean_K=0.2, val_peak_jump_error_p95_K=0.5)
        rows[1].update(val_peak_jump_error_mean_K=0.3, val_peak_jump_error_p95_K=0.7)
    with (tmp_path / "train_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _write_final_metrics(tmp_path, seed=42, config={}, wall_time_s=1., status="completed")
    report = json.loads((tmp_path / "final_metrics.json").read_text())
    assert report["best"]["val_peak_jump_error_mean_K"] == (None if legacy else 0.2)
    assert report["best"]["val_peak_jump_error_p95_K"] == (None if legacy else 0.5)
    assert report["final"]["val_peak_jump_error_mean_K"] == (None if legacy else 0.3)
    assert report["final"]["val_peak_jump_error_p95_K"] == (None if legacy else 0.7)
