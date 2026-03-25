import inspect
from math import comb

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from data.dataset import split_sim_ids
from visual import plots


@pytest.fixture
def plot_config():
    return {
        "training": {
            "n_snapshots": 6,
            "n_snapshots_test": 10,
            "curriculum_warmup": 20,
        },
        "data": {
            "train_split": 0.7,
            "val_split": 0.15,
        },
    }


class TestLoadPlotData:
    def test_loads_saved_arrays(self, tmp_npy_data):
        traj_path, x_path, t_path = tmp_npy_data
        trajectories, x_grid, t_grid = plots._load_plot_data(traj_path, x_path, t_path)
        assert trajectories.shape == (20, 51, 11)
        assert x_grid.shape == (11,)
        assert t_grid.shape == (51,)

    def test_mismatched_grid_raises(self, tmp_path, synthetic_trajectories):
        trajectories, _x_grid, t_grid = synthetic_trajectories
        traj_path = tmp_path / "trajectories.npy"
        x_path = tmp_path / "x_grid.npy"
        t_path = tmp_path / "t_grid.npy"
        np.save(traj_path, trajectories)
        np.save(x_path, np.linspace(0.0, 1.0, 7, dtype=np.float32))
        np.save(t_path, t_grid)
        with pytest.raises(ValueError, match="x_grid length"):
            plots._load_plot_data(traj_path, x_path, t_path)


class TestPlotRegistry:
    def test_new_plot_names_registered(self):
        for name in [
            "prediction_vs_truth",
            "snapshot_pair_samples",
            "lead_time_coverage",
            "lead_time_error",
            "parameter_error_slices",
        ]:
            assert plots.PLOT_REGISTRY[name] == "data"

    def test_retired_plot_names_removed(self):
        for name in [
            "dataset_samples",
            "trajectory_comparison_grid",
            "boundary_temperature",
            "parameter_response",
        ]:
            assert name not in plots.PLOT_REGISTRY

    def test_should_run_uses_new_registry(self):
        assert plots._should_run("prediction_vs_truth", ["data"], None) is True
        assert plots._should_run("prediction_vs_truth", ["training"], None) is False


class TestSnapshotPairSamples:
    def test_signature_has_no_legacy_window_params(self):
        signature = inspect.signature(plots.plot_snapshot_pair_samples)
        assert "k" not in signature.parameters
        assert "H" not in signature.parameters

    def test_smoke_writes_png(
        self,
        tmp_path,
        synthetic_trajectories,
        synthetic_sim_params,
        plot_config,
    ):
        trajectories, x_grid, t_grid = synthetic_trajectories
        out_path = tmp_path / "snapshot_pair_samples.png"
        plots.plot_snapshot_pair_samples(
            trajectories,
            x_grid,
            t_grid,
            synthetic_sim_params,
            config=plot_config,
            n_samples=2,
            save_path=out_path,
        )
        assert out_path.exists()


class TestLeadTimeCoverage:
    def test_counts_match_all_pairs(self, synthetic_trajectories, synthetic_sim_params, plot_config):
        trajectories, x_grid, t_grid = synthetic_trajectories
        coverage = plots._lead_time_coverage_counts(
            trajectories,
            x_grid,
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
        trajectories, x_grid, t_grid = synthetic_trajectories
        out_path = tmp_path / "lead_time_coverage.png"
        plots.plot_lead_time_coverage(
            trajectories,
            x_grid,
            t_grid,
            synthetic_sim_params,
            config=plot_config,
            save_path=out_path,
        )
        assert out_path.exists()


class TestModelDiagnosticPlots:
    def test_prediction_vs_truth_smoke(
        self,
        tmp_path,
        small_fno,
        synthetic_trajectories,
        synthetic_sim_params,
    ):
        trajectories, x_grid, t_grid = synthetic_trajectories
        out_path = tmp_path / "prediction_vs_truth.png"
        plots.plot_prediction_vs_truth(
            small_fno,
            trajectories,
            x_grid,
            t_grid,
            synthetic_sim_params,
            sim_id=0,
            s=1,
            n_steps=5,
            save_path=out_path,
        )
        assert out_path.exists()

    def test_lead_time_error_smoke(
        self,
        tmp_path,
        small_fno,
        synthetic_trajectories,
        synthetic_sim_params,
        plot_config,
    ):
        trajectories, x_grid, t_grid = synthetic_trajectories
        datasets = plots._build_split_datasets(trajectories, x_grid, t_grid, synthetic_sim_params, plot_config)
        out_path = tmp_path / "lead_time_error.png"
        plots.plot_lead_time_error(
            small_fno,
            datasets["test"],
            x_grid,
            max_samples=16,
            save_path=out_path,
        )
        assert out_path.exists()

    def test_parameter_error_slices_smoke(
        self,
        tmp_path,
        small_fno,
        synthetic_trajectories,
        synthetic_sim_params,
        plot_config,
    ):
        trajectories, x_grid, t_grid = synthetic_trajectories
        datasets = plots._build_split_datasets(trajectories, x_grid, t_grid, synthetic_sim_params, plot_config)
        out_path = tmp_path / "parameter_error_slices.png"
        plots.plot_parameter_error_slices(
            small_fno,
            datasets["test"],
            x_grid,
            max_samples=16,
            save_path=out_path,
        )
        assert out_path.exists()
