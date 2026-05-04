import inspect
from math import comb

import matplotlib
import numpy as np
import pytest
import torch

matplotlib.use("Agg")

from data.dataset import split_sim_ids
from visual import _common, dataset_plots, physics_plots


@pytest.fixture
def plot_config():
    return {
        "training": {
            "n_snapshots": 6,
            "n_snapshots_test": 10,
            "curriculum_warmup": 20,
            "loss": {
                "interface_x": 0.5,
                "interface_half_width": 0.05,
            },
        },
        "data": {
            "train_split": 0.7,
            "val_split": 0.15,
        },
    }


class TestLoadPlotData:
    def test_loads_saved_arrays(self, tmp_npy_data):
        traj_path, x_path, y_path, t_path = tmp_npy_data
        trajectories, x_grid, y_grid, t_grid = dataset_plots._load_plot_data(
            traj_path, x_path, y_path, t_path
        )
        assert trajectories.shape == (20, 51, 11, 11)
        assert x_grid.shape == (11,)
        assert y_grid.shape == (11,)
        assert t_grid.shape == (51,)

    def test_mismatched_grid_raises(self, tmp_path, synthetic_trajectories):
        trajectories, _x_grid, y_grid, t_grid = synthetic_trajectories
        traj_path = tmp_path / "trajectories.npy"
        x_path = tmp_path / "x_grid.npy"
        y_path = tmp_path / "y_grid.npy"
        t_path = tmp_path / "t_grid.npy"
        np.save(traj_path, trajectories)
        np.save(x_path, np.linspace(0.0, 1.0, 7, dtype=np.float32))
        np.save(y_path, y_grid)
        np.save(t_path, t_grid)
        with pytest.raises(ValueError, match="x_grid length"):
            dataset_plots._load_plot_data(traj_path, x_path, y_path, t_path)


class TestPlotRegistry:
    def test_new_plot_names_registered(self):
        for name in [
            "prediction_vs_truth",
            "snapshot_pair_samples",
            "lead_time_coverage",
            "lead_time_error",
            "parameter_error_slices",
            "dataset_summary",
            "interface_jump_summary",
        ]:
            assert _common.PLOT_REGISTRY[name] == "data"

    def test_retired_plot_names_removed(self):
        for name in [
            "dataset_samples",
            "trajectory_comparison_grid",
            "boundary_temperature",
            "parameter_response",
        ]:
            assert name not in _common.PLOT_REGISTRY

    def test_should_run_uses_new_registry(self):
        assert _common._should_run("prediction_vs_truth", ["data"], None) is True
        assert _common._should_run("prediction_vs_truth", ["training"], None) is False


class TestSnapshotPairSamples:
    def test_signature_has_no_legacy_window_params(self):
        signature = inspect.signature(dataset_plots.plot_snapshot_pair_samples)
        assert "k" not in signature.parameters
        assert "H" not in signature.parameters

    def test_smoke_writes_png(
        self,
        tmp_path,
        synthetic_trajectories,
        synthetic_sim_params,
        plot_config,
    ):
        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        out_path = tmp_path / "snapshot_pair_samples.png"
        dataset_plots.plot_snapshot_pair_samples(
            trajectories,
            x_grid,
            y_grid,
            t_grid,
            synthetic_sim_params,
            config=plot_config,
            n_samples=2,
            save_path=out_path,
        )
        assert out_path.exists()


class TestPlotHelpers:
    def test_future_target_indices_are_unique_and_increasing(self):
        target_indices = dataset_plots._future_target_indices(start_idx=2, n_requested=5, n_total=9)
        assert np.array_equal(target_indices, np.unique(target_indices))
        assert np.all(np.diff(target_indices) > 0)
        assert target_indices[0] > 2

    def test_future_target_indices_cap_to_available_horizon(self):
        target_indices = dataset_plots._future_target_indices(start_idx=7, n_requested=10, n_total=10)
        assert np.array_equal(target_indices, np.array([8, 9]))

    def test_future_target_indices_short_horizon_has_no_duplicates(self):
        target_indices = dataset_plots._future_target_indices(start_idx=46, n_requested=10, n_total=51)
        assert np.array_equal(target_indices, np.array([47, 48, 49, 50]))

    def test_future_target_indices_without_future_targets_raises(self):
        with pytest.raises(ValueError, match="future target"):
            dataset_plots._future_target_indices(start_idx=4, n_requested=3, n_total=5)

    def test_resolve_interface_metadata_uses_solver_positions(self, plot_config):
        class DummySolver:
            interface_positions = [0.25, 0.75]

        meta = _common._resolve_interface_metadata(config=plot_config, solver=DummySolver())
        assert meta["positions"] == [0.25, 0.75]
        assert meta["interface_x"] == pytest.approx(0.5)
        assert meta["interface_half_width"] == pytest.approx(0.05)

    def test_resolve_interface_metadata_uses_config_when_solver_missing(self, plot_config):
        meta = _common._resolve_interface_metadata(config=plot_config)
        assert meta["positions"] == [0.5]
        assert meta["interface_x"] == pytest.approx(0.5)
        assert meta["interface_half_width"] == pytest.approx(0.05)

    def test_resolve_interface_metadata_falls_back_when_missing(self):
        """Missing config+solver should yield defaults (no raise) so older checkpoints still plot."""
        meta = _common._resolve_interface_metadata(config=None, solver=None)
        assert meta["interface_x"] == pytest.approx(0.5)
        assert meta["interface_half_width"] == pytest.approx(0.05)
        assert meta["positions"] == [0.5]


class TestLeadTimeCoverage:
    def test_counts_match_all_pairs(self, synthetic_trajectories, synthetic_sim_params, plot_config):
        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        coverage = dataset_plots._lead_time_coverage_counts(
            trajectories,
            x_grid,
            y_grid,
            t_grid,
            synthetic_sim_params,
            plot_config,
        )
        train_ids, val_ids, test_ids = split_sim_ids(
            trajectories.shape[0],
            plot_config["data"]["train_split"],
            plot_config["data"]["val_split"],
            seed=0,
        )
        assert int(coverage["train_count"][0]) == len(train_ids) * comb(plot_config["training"]["n_snapshots"], 2)
        assert int(coverage["val_count"][0]) == len(val_ids) * comb(plot_config["training"]["n_snapshots"], 2)
        assert int(coverage["test_count"][0]) == len(test_ids) * comb(plot_config["training"]["n_snapshots_test"], 2)

    def test_smoke_writes_png(
        self,
        tmp_path,
        synthetic_trajectories,
        synthetic_sim_params,
        plot_config,
    ):
        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        out_path = tmp_path / "lead_time_coverage.png"
        dataset_plots.plot_lead_time_coverage(
            trajectories,
            x_grid,
            y_grid,
            t_grid,
            synthetic_sim_params,
            config=plot_config,
            save_path=out_path,
        )
        assert out_path.exists()


class TestDataFieldPlots:
    def test_trajectory_heatmap_smoke(
        self,
        tmp_path,
        synthetic_trajectories,
    ):
        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        out_path = tmp_path / "trajectory_heatmap.png"
        dataset_plots.plot_trajectory_heatmap(
            trajectories,
            sim_id=0,
            x_grid=x_grid,
            y_grid=y_grid,
            t_grid=t_grid,
            save_path=out_path,
        )
        assert out_path.exists()

    def test_initial_conditions_smoke(
        self,
        tmp_path,
        synthetic_trajectories,
    ):
        trajectories, x_grid, y_grid, _t_grid = synthetic_trajectories
        out_path = tmp_path / "initial_conditions.png"
        dataset_plots.plot_initial_conditions(
            trajectories,
            x_grid=x_grid,
            y_grid=y_grid,
            n_samples=4,
            save_path=out_path,
        )
        assert out_path.exists()

    def test_flux_profiles_smoke(self, tmp_path):
        out_path = tmp_path / "flux_profiles.png"
        dataset_plots.plot_flux_profiles(t_final=0.2, save_path=out_path)
        assert out_path.exists()


class TestModelDiagnosticPlots:
    def test_prediction_vs_truth_smoke(
        self,
        tmp_path,
        small_fno2d,
        synthetic_trajectories,
        synthetic_sim_params,
        plot_config,
    ):
        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        out_path = tmp_path / "prediction_vs_truth.png"
        dataset_plots.plot_prediction_vs_truth(
            small_fno2d,
            trajectories,
            x_grid,
            y_grid,
            t_grid,
            synthetic_sim_params,
            sim_id=0,
            s=1,
            n_steps=5,
            config=plot_config,
            save_path=out_path,
        )
        assert out_path.exists()

    def test_prediction_vs_truth_short_horizon_smoke(
        self,
        tmp_path,
        small_fno2d,
        synthetic_trajectories,
        synthetic_sim_params,
        plot_config,
    ):
        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        out_path = tmp_path / "prediction_vs_truth_short.png"
        dataset_plots.plot_prediction_vs_truth(
            small_fno2d,
            trajectories,
            x_grid,
            y_grid,
            t_grid,
            synthetic_sim_params,
            sim_id=0,
            s=len(t_grid) - 4,
            n_steps=10,
            config=plot_config,
            save_path=out_path,
        )
        assert out_path.exists()

    def test_interface_error_short_horizon_smoke(
        self,
        tmp_path,
        small_fno2d,
        synthetic_trajectories,
        synthetic_sim_params,
        plot_config,
    ):
        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        out_path = tmp_path / "interface_error.png"
        dataset_plots.plot_interface_error(
            small_fno2d,
            trajectories,
            x_grid,
            y_grid,
            t_grid,
            synthetic_sim_params,
            np.array([0, 1, 2]),
            n_samples=2,
            n_targets=10,
            config=plot_config,
            save_path=out_path,
        )
        assert out_path.exists()

    def test_lead_time_error_smoke(
        self,
        tmp_path,
        small_fno2d,
        synthetic_trajectories,
        synthetic_sim_params,
        plot_config,
    ):
        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        datasets = dataset_plots._build_split_datasets(trajectories, x_grid, y_grid, t_grid, synthetic_sim_params, plot_config)
        out_path = tmp_path / "lead_time_error.png"
        dataset_plots.plot_lead_time_error(
            small_fno2d,
            datasets["test"],
            x_grid,
            y_grid,
            config=plot_config,
            max_samples=16,
            save_path=out_path,
        )
        assert out_path.exists()

    def test_parameter_error_slices_smoke(
        self,
        tmp_path,
        small_fno2d,
        synthetic_trajectories,
        synthetic_sim_params,
        plot_config,
    ):
        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        datasets = dataset_plots._build_split_datasets(trajectories, x_grid, y_grid, t_grid, synthetic_sim_params, plot_config)
        out_path = tmp_path / "parameter_error_slices.png"
        dataset_plots.plot_parameter_error_slices(
            small_fno2d,
            datasets["test"],
            x_grid,
            y_grid,
            config=plot_config,
            max_samples=16,
            save_path=out_path,
        )
        assert out_path.exists()

    def test_dataset_summary_smoke(
        self,
        tmp_path,
        synthetic_trajectories,
        synthetic_sim_params,
        plot_config,
    ):
        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        out_path = tmp_path / "dataset_summary.png"
        dataset_plots.plot_dataset_summary(
            trajectories,
            x_grid,
            y_grid,
            t_grid,
            synthetic_sim_params,
            config=plot_config,
            save_path=out_path,
        )
        assert out_path.exists()

    def test_dataset_summary_handles_empty_split(
        self,
        tmp_path,
        plot_config,
    ):
        """With num_sims=4, split_sim_ids gives n_val=0 — guards must prevent .max() on empty arrays."""
        rng = np.random.default_rng(1)
        num_sims, Nt, Nx, Ny = 4, 21, 9, 9
        trajectories = rng.standard_normal((num_sims, Nt, Nx, Ny)).astype(np.float32)
        x_grid = np.linspace(0.0, 1.0, Nx).astype(np.float32)
        y_grid = np.linspace(0.0, 1.0, Ny).astype(np.float32)
        t_grid = np.linspace(0.0, 1.0, Nt).astype(np.float32)
        sim_params = np.array(
            [
                {
                    "amp": np.float32(rng.uniform(50.0, 300.0)),
                    "freq": np.float32(rng.uniform(1.0, 20.0)),
                    "T0": rng.standard_normal((Nx, Ny)).astype(np.float32),
                    "R_c": np.float32(rng.uniform(0.05, 1.0)),
                    "temporal_family": "sin",
                    "temporal_params": {"A": 100.0, "f": 5.0, "t_on": 0.0, "t_off": 0.2,
                                         "phase": 0.0, "tukey_alpha": 0.5},
                    "spatial_family": "uniform",
                    "spatial_params": {},
                }
                for _ in range(num_sims)
            ],
            dtype=object,
        )

        # Sanity check: split really does produce an empty val split at this size.
        _, val_ids, _ = split_sim_ids(num_sims, 0.7, 0.15, seed=0)
        assert len(val_ids) == 0

        out_path = tmp_path / "dataset_summary_empty_split.png"
        dataset_plots.plot_dataset_summary(
            trajectories,
            x_grid,
            y_grid,
            t_grid,
            sim_params,
            config=plot_config,
            save_path=out_path,
        )
        assert out_path.exists()

    def test_interface_jump_summary_smoke(
        self,
        tmp_path,
        small_fno2d,
        synthetic_trajectories,
        synthetic_sim_params,
        plot_config,
    ):
        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        datasets = dataset_plots._build_split_datasets(trajectories, x_grid, y_grid, t_grid, synthetic_sim_params, plot_config)
        out_path = tmp_path / "interface_jump_summary.png"
        dataset_plots.plot_interface_jump_summary(
            small_fno2d,
            datasets["test"],
            x_grid,
            y_grid,
            config=plot_config,
            max_samples=16,
            save_path=out_path,
        )
        assert out_path.exists()

    def test_load_checkpoint_model_loads_2d_checkpoint(self, small_fno2d_checkpoint):
        model, conf = dataset_plots._load_checkpoint_model(small_fno2d_checkpoint)
        assert conf["model"]["parameters"]["modes1"] == 2
        assert conf["model"]["parameters"]["modes2"] == 2
        assert getattr(model, "_mu_global") == 0.0
        assert getattr(model, "_sigma_global") == 1.0

    def test_load_checkpoint_model_rejects_1d_checkpoint(self, tmp_path):
        ckpt_path = tmp_path / "legacy_fno1d.pt"
        torch.save(
            {
                "conf": {
                    "model": {
                        "parameters": {
                            "modes": 2,
                            "width": 8,
                            "in_channels": 2,
                            "out_channels": 1,
                            "n_layers": 2,
                            "cond_dim": 5,
                        }
                    }
                }
            },
            ckpt_path,
        )
        with pytest.raises(ValueError, match="missing modes1/modes2"):
            dataset_plots._load_checkpoint_model(ckpt_path)

    def test_checkpoint_loaded_model_prediction_smoke(
        self,
        tmp_path,
        small_fno2d_checkpoint,
        synthetic_trajectories,
        synthetic_sim_params,
        plot_config,
    ):
        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        model, _ = dataset_plots._load_checkpoint_model(small_fno2d_checkpoint)
        out_path = tmp_path / "prediction_vs_truth_from_ckpt.png"
        dataset_plots.plot_prediction_vs_truth(
            model,
            trajectories,
            x_grid,
            y_grid,
            t_grid,
            synthetic_sim_params,
            sim_id=0,
            s=1,
            n_steps=5,
            config=plot_config,
            save_path=out_path,
        )
        assert out_path.exists()


class TestPhysicsPlots:
    def test_multilayer_evolution_smoke(self, tmp_path):
        solver = physics_plots.create_demo_multilayer_solver()
        T0 = np.full((solver.Nx, solver.Ny), solver.T_right(0.0))
        _, _, _, T_hist = solver.solve(T0=T0, store_trajectory=True)
        out_path = tmp_path / "multilayer_evolution.png"
        physics_plots.plot_multilayer_evolution(solver, T_hist, save_path=out_path)
        assert out_path.exists()
