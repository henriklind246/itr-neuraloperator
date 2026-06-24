import csv
import inspect
import json
from math import comb
from pathlib import Path

import matplotlib
import numpy as np
import pytest
import torch

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from data.dataset import split_sim_ids
from src.operators.eval import TEST_RECORD_FIELDS
from visual import _common, dataset_plots, forcing_plots, paper_plots, physics_plots


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

    def test_new_paper_profile_plot_names_registered(self):
        for name in [
            "forcing_temperature_profiles",
            "source_temperature_profiles",
            "source_itr_temperature_profiles",
            "interfaces_temperature_profiles",
            "forcing_interface_jump_profiles",
            "source_interface_jump_profiles",
            "source_itr_interface_jump_profiles",
            "interfaces_interface_jump_profiles",
            "source_itr_test_error_summary",
            "source_itr_prediction_truth_residual",
        ]:
            assert _common.PLOT_REGISTRY[name] == "paper"

    def test_retired_plot_names_removed(self):
        for name in [
            "dataset_samples",
            "trajectory_comparison_grid",
            "boundary_temperature",
            "parameter_response",
            "flux_profiles",
            "lhs_scatter",
            "y_perturbation",
        ]:
            assert name not in _common.PLOT_REGISTRY

    def test_should_run_uses_new_registry(self):
        assert _common._should_run("prediction_vs_truth", ["data"], None) is True
        assert _common._should_run("prediction_vs_truth", ["training"], None) is False

    def test_forcing_plot_names_registered(self):
        for name in [
            "forcing_temporal_families",
            "forcing_spatial_profiles",
            "forcing_separable_assembly",
            "forcing_bin_encoding",
            "forcing_seq_tokens",
            "forcing_summary_scalars",
            "forcing_param_distributions_design",
            "forcing_param_distributions_empirical",
        ]:
            assert _common.PLOT_REGISTRY[name] == "forcing"
        assert "forcing" in _common.GROUPS

    def test_source_plot_names_registered(self):
        for name in [
            "source_dataset_summary",
            "patch_param_scatter",
            "source_temporal_profile",
            "source_field_snapshots",
            "source_input_channels",
            "patch_overlay_trajectory",
            "energy_budget",
            "regime_error_breakdown",
            "patch_error_slices",
            "patch_region_error_map",
            "source_error_vs_params",
            "source_interface_zone_error",
            "source_itr_void_profiles",
            "source_itr_error_vs_void_params",
        ]:
            assert _common.PLOT_REGISTRY[name] == "source"
        assert "source" in _common.GROUPS

    def test_interface_plot_names_registered(self):
        for name in [
            "interface_y_perturbation",
            "interface_lhs_scatter",
            "vary_interface_lhs_scatter",
            "interface_flux_profiles",
            "sin_forcing_profiles",
            "interface_x_breakdown",
            "ic_family_trajectory_breakdown",
            "vary_interface_dataset_summary",
        ]:
            assert _common.PLOT_REGISTRY[name] == "interfaces"
        assert "interfaces" in _common.GROUPS

    def test_bc_verification_and_sweep_hyperparams_registered(self):
        assert _common.PLOT_REGISTRY["bc_verification"] == "physics"
        assert _common.PLOT_REGISTRY["sweep_hyperparams"] == "sweep"

    def test_itr_temperature_jump_sweep_registered(self):
        assert _common.PLOT_REGISTRY["itr_temperature_jump_sweep"] == "physics"
        assert "itr_temperature_jump_sweep" in _common._plots_for_group("physics")

    def test_tail_error_plot_names_registered(self):
        for name in [
            "forcing_tail_errors",
            "source_tail_errors",
            "interfaces_tail_errors",
        ]:
            assert _common.PLOT_REGISTRY[name] == "paper"

    def test_consolidated_paper_plot_names_registered(self):
        for name in [
            "benchmark_overview",
            "all_benchmarks_tail_errors",
            "generalization_same_vs_unseen",
        ]:
            assert _common.PLOT_REGISTRY[name] == "paper"
            assert name in _common._plots_for_group("paper")

    def test_rollout_plot_name_registered(self):
        assert _common.PLOT_REGISTRY["rollout_partition_error"] == "rollout"
        assert "rollout" in _common.GROUPS


class TestItrTemperatureJumpSweep:
    def test_forcing_smoke(self, tmp_path):
        result = physics_plots.plot_itr_temperature_jump_sweep(
            benchmarks=("forcing",),
            Nx=20, Ny=20, t_final=0.1,
            scalar_rc_values=(0.10, 0.70),
            requested_times=(0.05, 0.10),
            save_path=tmp_path / "itr_temperature_jump_sweep.png",
        )
        assert result["png"].exists()
        assert result["csv"].exists()
        records = result["records"]
        # 2 R_c values x 2 requested times.
        assert len(records) == 4
        for row in records:
            assert row["benchmark"] == "forcing"
            assert row["itr_kind"] == "scalar_Rc"
            assert np.isfinite(row["mean_abs_jump_K"])
            assert np.isfinite(row["rms_jump_K"])
            assert np.isfinite(row["peak_abs_jump_K"])
            assert row["mean_abs_jump_K"] >= 0.0

    def test_source_itr_metadata(self, tmp_path):
        result = physics_plots.plot_itr_temperature_jump_sweep(
            benchmarks=("source_itr",),
            Nx=20, Ny=20, t_final=0.1,
            rc_peak_values=(0.05, 1.00),
            requested_times=(0.05, 0.10),
            save_path=tmp_path / "itr_temperature_jump_sweep.png",
        )
        records = result["records"]
        assert len(records) == 4
        for row in records:
            assert np.isfinite(row["mean_abs_jump_K"])
            assert row["itr_kind"] == "Rc_peak"
            assert row["R_c_base"] == 0.05
            assert row["R_c_peak"] == row["itr_value"]
            assert row["R_c_amp"] == pytest.approx(row["R_c_peak"] - 0.05)

        # The R_c_peak == 0.05 endpoint is the "no void excess" baseline.
        base_rows = [r for r in records if r["itr_value"] == 0.05]
        assert base_rows
        for row in base_rows:
            assert row["R_c_amp"] == pytest.approx(0.0)

    def test_forcing_jump_grows_with_resistance(self, tmp_path):
        # Physical sanity, scoped to the canonical forcing case at the final
        # requested time only: contact-jump magnitude is larger-or-comparable as
        # R_c grows. source/source_itr/interfaces trends are intentionally not
        # asserted (patch placement and local R_c(y) can break monotonicity).
        rc_values = (0.05, 1.00)
        final_time = 0.10
        result = physics_plots.plot_itr_temperature_jump_sweep(
            benchmarks=("forcing",),
            Nx=24, Ny=24, t_final=0.1,
            scalar_rc_values=rc_values,
            requested_times=(final_time,),
            save_path=tmp_path / "itr_temperature_jump_sweep.png",
        )
        by_rc = {
            row["R_c"]: row["mean_abs_jump_K"]
            for row in result["records"]
            if row["time_requested"] == final_time
        }
        tol = 1e-6
        assert by_rc[1.00] >= by_rc[0.05] - tol


def _write_rollout_report(path, num_substeps, global_norm):
    report = {
        "timestamp": "2026-06-11T16:00:00",
        "summary": {"rollout_num_substeps": num_substeps},
        "per_seed": [
            {
                "seed": 42,
                "num_sims": 50,
                "rollout_num_substeps": num_substeps,
                "test_rel_l2_norm": global_norm,
                "test_iface_rel_l2_norm": global_norm * 1.1,
                "test_boundary_rel_l2_norm": global_norm * 1.05,
            }
        ],
    }
    with open(path, "w") as f:
        json.dump(report, f)


class TestRolloutPlots:
    def test_loader_keys_by_num_substeps_no_scaling(self, tmp_path):
        from visual import rollout_plots

        _write_rollout_report(tmp_path / "directeval.json", 1, 0.51)
        _write_rollout_report(tmp_path / "rollouteval_s5.json", 5, 1.90)

        reports = rollout_plots._load_rollout_reports(tmp_path)

        assert sorted(reports) == [1, 5]
        # rel_l2_norm fields are already percentages: stored verbatim, no x100.
        assert reports[1]["test_rel_l2_norm_mean"] == pytest.approx(0.51)
        assert reports[5]["test_rel_l2_norm_mean"] == pytest.approx(1.90)
        assert reports[1]["seed_signature"] == (42,)

    def test_loader_fails_on_missing_metric(self, tmp_path):
        from visual import rollout_plots

        report = {
            "timestamp": "2026-06-11T16:00:00",
            "summary": {"rollout_num_substeps": 3},
            "per_seed": [
                {
                    "seed": 42,
                    "num_sims": 50,
                    "rollout_num_substeps": 3,
                    "test_rel_l2_norm": 1.0,
                    "test_iface_rel_l2_norm": 1.1,
                }
            ],
        }
        with open(tmp_path / "rollouteval_s3.json", "w") as f:
            json.dump(report, f)

        with pytest.raises(ValueError, match="test_boundary_rel_l2_norm"):
            rollout_plots._load_rollout_reports(tmp_path)

        reports = rollout_plots._load_rollout_reports(tmp_path, allow_missing=True)
        assert np.isnan(reports[3]["test_boundary_rel_l2_norm_mean"])

    def test_plot_smoke_writes_png(self, tmp_path):
        from visual import rollout_plots

        _write_rollout_report(tmp_path / "directeval.json", 1, 0.51)
        _write_rollout_report(tmp_path / "rollouteval_s5.json", 5, 1.90)
        out_path = tmp_path / "rollout_partition_error.png"

        rollout_plots.plot_rollout_partition_error(tmp_path, save_path=out_path)

        assert out_path.exists()


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

class TestForcingPlots:
    def test_temporal_families_smoke(self, tmp_path):
        out_path = tmp_path / "forcing_temporal_families.png"
        forcing_plots.plot_forcing_temporal_families(n_curves=3, save_path=out_path)
        assert out_path.exists()

    def test_spatial_profiles_smoke(self, tmp_path):
        out_path = tmp_path / "forcing_spatial_profiles.png"
        forcing_plots.plot_forcing_spatial_profiles(n_curves=3, save_path=out_path)
        assert out_path.exists()

    def test_separable_assembly_smoke(self, tmp_path):
        out_path = tmp_path / "forcing_separable_assembly.png"
        forcing_plots.plot_forcing_separable_assembly(save_path=out_path)
        assert out_path.exists()

    def test_bin_encoding_smoke(self, tmp_path):
        out_path = tmp_path / "forcing_bin_encoding.png"
        forcing_plots.plot_forcing_bin_encoding(save_path=out_path)
        assert out_path.exists()

    def test_seq_tokens_smoke(self, tmp_path):
        out_path = tmp_path / "forcing_seq_tokens.png"
        forcing_plots.plot_forcing_seq_tokens(save_path=out_path)
        assert out_path.exists()

    def test_summary_scalars_smoke(self, tmp_path):
        out_path = tmp_path / "forcing_summary_scalars.png"
        forcing_plots.plot_forcing_summary_scalars(n_samples=40, save_path=out_path)
        assert out_path.exists()

    def test_param_distributions_design_smoke(self, tmp_path):
        out_path = tmp_path / "forcing_param_distributions_design.png"
        forcing_plots.plot_forcing_param_distributions_design(
            n_samples=40,
            save_path=out_path,
            spatial_save_path=tmp_path / "forcing_param_distributions_design_spatial.png",
        )
        assert out_path.exists()

    def test_param_distributions_empirical_smoke(self, tmp_path):
        sim_params = np.array(
            [
                {
                    "temporal_family": "sin",
                    "temporal_params": {
                        "A": 200.0, "f": 5.0, "t_on": 0.0, "t_off": 0.2,
                        "phase": 0.0, "tukey_alpha": 0.5,
                    },
                    "spatial_family": "uniform",
                    "spatial_params": {},
                },
                {
                    "temporal_family": "exp",
                    "temporal_params": {"A": 150.0, "t0": 0.05, "tau": 0.03},
                    "spatial_family": "gaussian",
                    "spatial_params": {"y_c": 0.5, "sigma_y": 0.08},
                },
                {
                    "temporal_family": "pulse_train",
                    "temporal_params": {
                        "Np": 2,
                        "A_list": [100.0, 80.0],
                        "t_list": [0.05, 0.12],
                        "dt_list": [0.01, 0.015],
                    },
                    "spatial_family": "patch",
                    "spatial_params": {"y_c": 0.4, "w": 0.2},
                },
                {
                    "temporal_family": "exp_train",
                    "temporal_params": {
                        "Np": 2,
                        "A_list": [120.0, 90.0],
                        "t_list": [0.04, 0.10],
                        "tau_list": [0.01, 0.02],
                    },
                    "spatial_family": "triangle",
                    "spatial_params": {"y_c": 0.5, "ell": 0.25},
                },
            ],
            dtype=object,
        )
        out_path = tmp_path / "forcing_param_distributions_empirical.png"
        forcing_plots.plot_forcing_param_distributions_empirical(
            sim_params,
            save_path=out_path,
            spatial_save_path=tmp_path / "forcing_param_distributions_empirical_spatial.png",
        )
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

    def test_bc_verification_smoke(self, tmp_path):
        solver = physics_plots.create_demo_multilayer_solver()
        T0 = np.full((solver.Nx, solver.Ny), solver.T_right(0.0))
        _, _, _, T_hist = solver.solve(T0=T0, store_trajectory=True)
        out_path = tmp_path / "bc_verification.png"
        physics_plots.plot_bc_verification(solver, T_hist, save_path=out_path)
        assert out_path.exists()


class TestInterfacePlots:
    def test_interface_flux_profiles_smoke(self, tmp_path):
        out_path = tmp_path / "interface_flux_profiles.png"
        dataset_plots.plot_interface_flux_profiles(save_path=out_path)
        assert out_path.exists()

    def test_sin_forcing_profiles_smoke(self, tmp_path):
        out_path = tmp_path / "sin_forcing_profiles.png"
        dataset_plots.plot_sin_forcing_profiles(save_path=out_path)
        assert out_path.exists()


def _source_itr_params_from_source(synthetic_source_sim_params):
    params = []
    for i, p in enumerate(synthetic_source_sim_params):
        q = dict(p)
        R_base = 0.08 + 0.035 * (i % 12)
        R_amp = 0.0 if i % 6 == 0 else 0.25 + 0.08 * (i % 5)
        q["R_c_base"] = float(R_base)
        q["R_c_amp"] = float(R_amp)
        q["R_c_y0"] = float(0.15 + 0.7 * ((i * 3) % 19) / 18.0)
        q["R_c_sigma"] = float(0.05 + 0.15 * ((i * 5) % 19) / 18.0)
        q["R_c"] = float(R_base)
        params.append(q)
    return np.array(params, dtype=object)


def _interfaces_params_from_forcing(synthetic_sim_params):
    params = []
    x_values = np.linspace(0.2, 0.8, len(synthetic_sim_params))
    for i, p in enumerate(synthetic_sim_params):
        q = dict(p)
        q["interface_x"] = float(x_values[i])
        q["temporal_family"] = "sin"
        q["temporal_params"] = {
            "A": 100.0,
            "f": 5.0,
            "t_on": 0.0,
            "t_off": 0.2,
            "phase": 0.0,
            "tukey_alpha": 0.5,
            "rectified": True,
        }
        q["spatial_family"] = "uniform"
        q["spatial_params"] = {}
        params.append(q)
    return np.array(params, dtype=object)


def _small_source_itr_model():
    from problems.source_itr import COND_STATIC_DIM, SPATIAL_CHANNELS_BINS
    from src.operators.fno2d import FNO2d

    return FNO2d(
        modes1=2,
        modes2=2,
        width=8,
        in_channels=SPATIAL_CHANNELS_BINS,
        out_channels=1,
        n_layers=2,
        cond_static_dim=COND_STATIC_DIM,
        use_temporal_encoder=False,
    )


def _small_interfaces_model():
    from problems.interfaces import (
        COND_STATIC_DIM,
        FORCING_TEMPORAL_TOKEN_DIM,
        S_Y_CHANNEL,
        SPATIAL_CHANNELS_TEMPORAL,
    )
    from src.operators.fno2d import FNO2d

    return FNO2d(
        modes1=2,
        modes2=2,
        width=8,
        in_channels=SPATIAL_CHANNELS_TEMPORAL,
        out_channels=1,
        n_layers=2,
        cond_static_dim=COND_STATIC_DIM,
        temporal_token_dim=FORCING_TEMPORAL_TOKEN_DIM,
        temporal_hidden=16,
        forcing_embed_dim=16,
        use_forcing_time_aug=True,
        s_y_channel=S_Y_CHANNEL,
    )


class TestSourcePlots:
    def test_patch_param_scatter_smoke(self, tmp_path, synthetic_source_sim_params):
        out_path = tmp_path / "patch_param_scatter.png"
        dataset_plots.plot_patch_param_scatter(
            synthetic_source_sim_params, save_path=out_path
        )
        assert out_path.exists()

    def test_source_itr_void_severity_handles_flat_profile(self, synthetic_source_sim_params):
        y_grid = np.linspace(0.0, 1.0, 101, dtype=np.float32)
        params = _source_itr_params_from_source(synthetic_source_sim_params[:3])
        params[0]["R_c_amp"] = 0.0
        params[1]["R_c_amp"] = 0.4
        params[1]["R_c_sigma"] = 0.05
        params[2]["R_c_amp"] = 0.4
        params[2]["R_c_sigma"] = 0.16

        severity = dataset_plots._void_severity_arrays(params, y_grid)

        assert severity["R_c_excess_integral"][0] == pytest.approx(0.0)
        assert severity["conductance_deficit"][0] == pytest.approx(0.0)
        assert severity["is_void_active"][0] == np.bool_(False)
        assert severity["R_c_excess_integral"][2] > severity["R_c_excess_integral"][1]

    def test_source_itr_void_profiles_smoke(
        self,
        tmp_path,
        synthetic_trajectories,
        synthetic_source_sim_params,
    ):
        _trajectories, x_grid, y_grid, _t_grid = synthetic_trajectories
        sim_params = _source_itr_params_from_source(synthetic_source_sim_params)
        out_path = tmp_path / "source_itr_void_profiles.png"

        dataset_plots.plot_source_itr_void_profiles(
            sim_params,
            y_grid,
            x_grid=x_grid,
            save_path=out_path,
        )

        assert out_path.exists()

    def test_source_dataset_summary_smoke(
        self, tmp_path, synthetic_trajectories, synthetic_source_sim_params, plot_config
    ):
        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        source_config = {**plot_config, "benchmark": {"name": "source", "representation": "bins"}}
        out_path = tmp_path / "source_dataset_summary.png"
        dataset_plots.plot_source_dataset_summary(
            trajectories,
            x_grid,
            y_grid,
            t_grid,
            synthetic_source_sim_params,
            config=source_config,
            save_path=out_path,
        )
        assert out_path.exists()

    def test_source_checkpoint_loads_with_temporal_encoder_off(
        self, small_source_fno2d_checkpoint
    ):
        model, conf = dataset_plots._load_checkpoint_model(small_source_fno2d_checkpoint)
        assert conf["model"]["parameters"]["cond_static_dim"] == 7
        assert getattr(model, "use_temporal_encoder") is False

    def test_patch_error_slices_smoke(
        self,
        tmp_path,
        small_source_fno2d_checkpoint,
        synthetic_trajectories,
        synthetic_source_sim_params,
        plot_config,
    ):
        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        model, conf = dataset_plots._load_checkpoint_model(small_source_fno2d_checkpoint)
        source_config = {**plot_config, "benchmark": {"name": "source", "representation": "bins"}}
        datasets = dataset_plots._build_split_datasets(
            trajectories, x_grid, y_grid, t_grid, synthetic_source_sim_params, source_config
        )
        out_path = tmp_path / "patch_error_slices.png"
        dataset_plots.plot_patch_error_slices(
            model,
            datasets["test"],
            x_grid,
            y_grid,
            config=source_config,
            sim_params=synthetic_source_sim_params,
            max_samples=16,
            save_path=out_path,
        )
        assert out_path.exists()

    def test_source_itr_error_vs_void_params_smoke(
        self,
        tmp_path,
        synthetic_trajectories,
        synthetic_source_sim_params,
        plot_config,
    ):
        from problems.source_itr import COND_STATIC_DIM, SPATIAL_CHANNELS_BINS
        from src.operators.fno2d import FNO2d

        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        sim_params = _source_itr_params_from_source(synthetic_source_sim_params)
        source_itr_config = {
            **plot_config,
            "benchmark": {"name": "source_itr", "representation": "bins"},
        }
        datasets = dataset_plots._build_split_datasets(
            trajectories, x_grid, y_grid, t_grid, sim_params, source_itr_config
        )
        model = FNO2d(
            modes1=2,
            modes2=2,
            width=8,
            in_channels=SPATIAL_CHANNELS_BINS,
            out_channels=1,
            n_layers=2,
            cond_static_dim=COND_STATIC_DIM,
            use_temporal_encoder=False,
        )
        out_path = tmp_path / "source_itr_error_vs_void_params.png"

        dataset_plots.plot_source_itr_error_vs_void_params(
            model,
            datasets["test"],
            x_grid,
            y_grid,
            config=source_itr_config,
            max_samples=16,
            save_path=out_path,
        )

        assert out_path.exists()


# ---------- paper-figure helpers ----------

def _write_records_csv(path, rows):
    """Write a minimal test_records.csv (blank cells for absent fields)."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TEST_RECORD_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in TEST_RECORD_FIELDS})


def _rich_metric_cells(rng):
    """The six physical-K / scale-normalized metrics eval writes per record row.

    Kept together so every benchmark fixture emits the same rich-metric columns the
    paper loader (`_FLOAT_COLS`) now reads; blanks would otherwise load as NaN.
    """
    return {
        "rmse_K": float(rng.uniform(0.5, 12.0)),
        "gnrmse_pct": float(rng.uniform(0.5, 10.0)),
        "nrmse_pct": float(rng.uniform(0.5, 10.0)),
        "node_jump_rmse_K": float(rng.uniform(0.2, 6.0)),
        "node_jump_nrmse_pct": float(rng.uniform(0.5, 14.0)),
        "node_jump_gnrmse_pct": float(rng.uniform(0.5, 14.0)),
    }


def _forcing_record_rows():
    """One row per temporal x spatial family combo (covers every box/heatmap cell)."""
    rng = np.random.default_rng(0)
    rows = []
    for ti, temporal in enumerate(paper_plots.TEMPORAL_ORDER):
        for si, spatial in enumerate(paper_plots.SPATIAL_ORDER):
            sim_id = (ti * len(paper_plots.SPATIAL_ORDER) + si) % 20
            rows.append({
                "sim_id": sim_id, "s": 1, "j": 8,
                "t_s": 0.02, "t_bar": 0.14 + 0.01 * si,
                "R_c": float(rng.uniform(0.05, 1.0)), "benchmark": "forcing",
                "temporal_family": temporal, "spatial_family": spatial,
                "x_I": 0.5,
                "rel_l2_pct": float(rng.uniform(0.5, 8.0)),
                "iface_rel_l2_pct": float(rng.uniform(0.5, 12.0)),
                **_rich_metric_cells(rng),
            })
    return rows


def _source_record_rows():
    rng = np.random.default_rng(1)
    rows = []
    for i in range(15):
        regime = paper_plots.REGIME_ORDER[i % len(paper_plots.REGIME_ORDER)]
        x_h = {"left": 0.25, "near": 0.5, "right": 0.75}[regime]
        rows.append({
            "sim_id": i % 20, "s": 1, "j": 8,
            "t_s": 0.02, "t_bar": 0.10 + 0.01 * i,
            "R_c": float(rng.uniform(0.05, 1.0)), "benchmark": "source",
            "x_h": x_h + float(rng.uniform(-0.05, 0.05)),
            "y_h": float(rng.uniform(0.3, 0.7)),
            "A": float(rng.uniform(50.0, 300.0)), "regime": regime,
            "x_I": 0.5,
            "rel_l2_pct": float(rng.uniform(0.5, 8.0)),
            "iface_rel_l2_pct": float(rng.uniform(0.5, 12.0)),
            **_rich_metric_cells(rng),
        })
    return rows


def _interfaces_record_rows():
    rng = np.random.default_rng(2)
    rows = []
    for i in range(15):
        x_I = float(np.linspace(0.2, 0.8, 15)[i])
        rows.append({
            "sim_id": i % 20, "s": 1, "j": 8,
            "t_s": 0.02, "t_bar": 0.10 + 0.01 * i,
            "R_c": float(rng.uniform(0.05, 1.0)), "benchmark": "interfaces",
            "temporal_family": "sin", "spatial_family": "uniform",
            "A": float(rng.uniform(50.0, 300.0)), "freq": float(rng.uniform(1.0, 10.0)),
            "x_I": x_I,
            "rel_l2_pct": float(rng.uniform(0.5, 8.0)),
            "iface_rel_l2_pct": float(rng.uniform(0.5, 12.0)),
            **_rich_metric_cells(rng),
        })
    return rows


@pytest.fixture
def forcing_records(tmp_path):
    path = tmp_path / "forcing_records.csv"
    _write_records_csv(path, _forcing_record_rows())
    return paper_plots._load_test_records(path)


@pytest.fixture
def source_records(tmp_path):
    path = tmp_path / "source_records.csv"
    _write_records_csv(path, _source_record_rows())
    return paper_plots._load_test_records(path)


@pytest.fixture
def source_itr_records(tmp_path):
    path = tmp_path / "source_itr_records.csv"
    rng = np.random.default_rng(7)
    rows = []
    for row in _source_record_rows():
        updated = dict(row)
        updated["benchmark"] = "source_itr"
        updated["R_c_amp"] = float(rng.uniform(0.0, 2.95))
        updated["R_c_y0"] = float(rng.uniform(0.1, 0.9))
        updated["R_c_sigma"] = float(rng.uniform(0.05, 0.2))
        rows.append(updated)
    _write_records_csv(path, rows)
    return paper_plots._load_test_records(path)


@pytest.fixture
def interfaces_records(tmp_path):
    path = tmp_path / "interfaces_records.csv"
    _write_records_csv(path, _interfaces_record_rows())
    return paper_plots._load_test_records(path)


class TestPaperRecordLoading:
    def test_void_columns_in_schema(self):
        for col in ("R_c_amp", "R_c_y0", "R_c_sigma"):
            assert col in TEST_RECORD_FIELDS
            assert col in paper_plots._FLOAT_COLS

    def test_void_columns_load_for_source_itr(self, source_itr_records):
        for col in ("R_c_amp", "R_c_y0", "R_c_sigma"):
            assert np.all(np.isfinite(source_itr_records[col]))

    def test_void_columns_blank_for_source(self, source_records):
        # Source rows omit void params -> NaN, so void panels stay empty.
        for col in ("R_c_amp", "R_c_y0", "R_c_sigma"):
            assert np.all(np.isnan(source_records[col]))

    _RICH_METRIC_COLS = (
        "rmse_K", "gnrmse_pct", "nrmse_pct",
        "node_jump_rmse_K", "node_jump_nrmse_pct", "node_jump_gnrmse_pct",
    )

    def test_rich_metrics_in_float_cols(self):
        for col in self._RICH_METRIC_COLS:
            assert col in TEST_RECORD_FIELDS
            assert col in paper_plots._FLOAT_COLS

    def test_rich_metrics_load_finite_when_present(self, forcing_records, interfaces_records):
        for records in (forcing_records, interfaces_records):
            for col in self._RICH_METRIC_COLS:
                assert np.all(np.isfinite(records[col]))

    def test_rich_metrics_blank_become_nan(self, tmp_path):
        # A row that omits the rich metric columns loads them as NaN, never 0.
        path = tmp_path / "thin_records.csv"
        _write_records_csv(path, [{
            "sim_id": 0, "s": 1, "j": 8, "t_s": 0.02, "t_bar": 0.1,
            "R_c": 0.5, "benchmark": "forcing", "x_I": 0.5,
            "rel_l2_pct": 2.0, "iface_rel_l2_pct": 3.0,
        }])
        thin = paper_plots._load_test_records(path)
        for col in self._RICH_METRIC_COLS:
            assert np.all(np.isnan(thin[col]))

    def test_blank_floats_become_nan(self, forcing_records):
        # Forcing rows leave x_h/A blank -> NaN; rel_l2_pct is always present.
        assert np.all(np.isnan(forcing_records["x_h"]))
        assert np.all(np.isfinite(forcing_records["rel_l2_pct"]))
        assert int(forcing_records["_n"]) == 16

    def test_representative_row_is_median_pick(self, forcing_records):
        mask = np.ones(int(forcing_records["_n"]), dtype=bool)
        row = paper_plots._representative_row(forcing_records, mask)
        assert row >= 0
        err = forcing_records["rel_l2_pct"]
        median = float(np.median(err))
        assert np.argmin(np.abs(err - median)) == row

    def test_representative_row_empty_mask_returns_negative(self, forcing_records):
        mask = np.zeros(int(forcing_records["_n"]), dtype=bool)
        assert paper_plots._representative_row(forcing_records, mask) == -1


class TestPaperSummaryPlots:
    def test_forcing_summary_smoke(self, tmp_path, forcing_records):
        out_path = tmp_path / "forcing_test_error_summary.png"
        paper_plots.plot_forcing_test_error_summary(forcing_records, save_path=out_path)
        assert out_path.exists()

    def test_source_summary_smoke(self, tmp_path, source_records):
        out_path = tmp_path / "source_test_error_summary.png"
        paper_plots.plot_source_test_error_summary(source_records, save_path=out_path)
        assert out_path.exists()

    def test_source_summary_amplitude_panel_smoke(self, tmp_path, source_records):
        out_path = tmp_path / "source_test_error_summary_amp.png"
        paper_plots.plot_source_test_error_summary(
            source_records, save_path=out_path, panel6="amplitude"
        )
        assert out_path.exists()

    def test_interfaces_summary_smoke(self, tmp_path, interfaces_records):
        out_path = tmp_path / "interfaces_test_error_summary.png"
        paper_plots.plot_interfaces_test_error_summary(interfaces_records, save_path=out_path)
        assert out_path.exists()

    def test_source_itr_summary_smoke(self, tmp_path, source_itr_records):
        out_path = tmp_path / "source_itr_test_error_summary.png"
        paper_plots.plot_source_itr_test_error_summary(source_itr_records, save_path=out_path)
        assert out_path.exists()

    def test_all_benchmarks_summary_smoke(
        self, tmp_path, forcing_records, source_records, interfaces_records,
        source_itr_records,
    ):
        out_path = tmp_path / "all_benchmarks_test_error_summary.png"
        paper_plots.plot_all_benchmarks_error_summary(
            {"forcing": forcing_records, "source": source_records,
             "interfaces": interfaces_records, "source_itr": source_itr_records},
            save_path=out_path,
        )
        assert out_path.exists()


class TestTailErrorStats:
    def test_tail_stats_percentiles_exact(self):
        v = np.linspace(0.0, 100.0, 101)
        stats = paper_plots._tail_stats(v)
        p90, p99 = np.percentile(v, [90, 99], method="linear")
        assert stats["n"] == 101
        assert stats["p90"] == pytest.approx(float(p90))
        assert stats["p99"] == pytest.approx(float(p99))
        assert stats["max"] == pytest.approx(float(np.max(v)))
        assert stats["mean"] == pytest.approx(float(np.mean(v)))
        assert stats["median"] == pytest.approx(float(np.median(v)))

    def test_tail_stats_empty_returns_nans(self):
        stats = paper_plots._tail_stats(np.array([np.nan, np.inf, -np.inf]))
        assert stats["n"] == 0
        for key in ("mean", "median", "p90", "p99", "max"):
            assert np.isnan(stats[key])

    def test_tail_stats_ignores_nans_and_counts_finite(self):
        # Mixed finite/NaN: n is the finite count, stats use only finite entries.
        v = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        stats = paper_plots._tail_stats(v)
        assert stats["n"] == 3
        assert stats["median"] == pytest.approx(3.0)
        assert stats["max"] == pytest.approx(5.0)

    def test_tail_stats_unequal_sample_counts(self):
        # Two benchmarks with different finite counts both report n correctly.
        a = paper_plots._tail_stats(np.linspace(0.0, 1.0, 10))
        b = paper_plots._tail_stats(np.linspace(0.0, 1.0, 25))
        assert a["n"] == 10
        assert b["n"] == 25

    def test_tail_amp_zero_median_is_nan(self):
        stats = {"median": 0.0, "p99": 4.0}
        assert np.isnan(paper_plots._tail_amp(stats))

    def test_tail_amp_positive_median_is_finite_ratio(self):
        stats = {"median": 2.0, "p99": 5.0}
        assert paper_plots._tail_amp(stats) == pytest.approx(2.5)

    def test_tail_amp_nonfinite_inputs_are_nan(self):
        assert np.isnan(paper_plots._tail_amp({"median": np.nan, "p99": 4.0}))
        assert np.isnan(paper_plots._tail_amp({"median": 2.0, "p99": np.nan}))

    def test_grouped_tail_bars_all_nan_metric_no_crash(self):
        # A benchmark whose metric is all-NaN keeps its slot and annotates n/a;
        # the panel must render without raising.
        fig, ax = plt.subplots()
        names = ["forcing", "source"]
        stats_by_name = {
            "forcing": {**paper_plots._tail_stats(np.linspace(1.0, 9.0, 9)),
                        "tail_amp": 1.4},
            "source": {**paper_plots._tail_stats(np.array([np.nan, np.nan])),
                       "tail_amp": np.nan},
        }
        paper_plots._grouped_tail_bars(
            ax, names, stats_by_name,
            bar_stat="p99", marker_stats=("max",), hollow_stats=("max",),
            ylabel="m", title="t", annotate_n=True, annotate_tail_amp=True,
        )
        ticklabels = [t.get_text() for t in ax.get_xticklabels()]
        assert ticklabels == names
        plt.close(fig)

    def test_all_benchmarks_tail_errors_smoke(
        self, tmp_path, forcing_records, source_records, interfaces_records,
        source_itr_records,
    ):
        out_path = tmp_path / "all_benchmarks_tail_errors.png"
        result = paper_plots.plot_all_benchmarks_tail_errors(
            {"forcing": forcing_records, "source": source_records,
             "interfaces": interfaces_records, "source_itr": source_itr_records},
            save_path=out_path,
        )
        assert out_path.exists()
        assert result == out_path

    def test_forcing_tail_errors_smoke(self, tmp_path, forcing_records):
        out_path = tmp_path / "forcing_tail_errors.png"
        paper_plots.plot_benchmark_tail_errors(forcing_records, save_path=out_path)
        assert out_path.exists()

    def test_source_tail_errors_smoke(self, tmp_path, source_records):
        out_path = tmp_path / "source_tail_errors.png"
        paper_plots.plot_benchmark_tail_errors(source_records, save_path=out_path)
        assert out_path.exists()

    def test_interfaces_tail_errors_smoke(self, tmp_path, interfaces_records):
        out_path = tmp_path / "interfaces_tail_errors.png"
        paper_plots.plot_benchmark_tail_errors(interfaces_records, save_path=out_path)
        assert out_path.exists()

    def test_write_tail_summary(
        self, tmp_path, forcing_records, source_records, interfaces_records
    ):
        out_path = tmp_path / "tail_summary.csv"
        paper_plots.write_tail_summary(
            {"forcing": forcing_records, "source": source_records,
             "interfaces": interfaces_records},
            out_path,
        )
        assert out_path.exists()
        with open(out_path, newline="") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == list(paper_plots._TAIL_SUMMARY_FIELDS)
            rows = list(reader)

        benchmarks = {r["benchmark"] for r in rows}
        assert benchmarks == {"forcing", "source", "interfaces"}
        metrics = {r["metric"] for r in rows}
        assert metrics == set(paper_plots.TAIL_METRICS.keys())

        for bm in ("forcing", "source", "interfaces"):
            for metric in paper_plots.TAIL_METRICS:
                overall = [r for r in rows if r["benchmark"] == bm
                           and r["metric"] == metric
                           and r["stratum"] == "overall" and r["group"] == "all"]
                assert len(overall) == 1

        for r in rows:
            if int(r["n"]) == 0:
                continue
            p90, p99, mx = float(r["p90"]), float(r["p99"]), float(r["max"])
            assert p90 <= p99 <= mx

    def test_quantile_bin_degenerate(self):
        # Constant x_I and constant t_bar must omit the degenerate numeric strata
        # while still emitting overall/all and not crashing.
        records = {
            "_n": np.int64(6),
            "benchmark": np.array(["interfaces"] * 6, dtype=object),
            "x_I": np.full(6, 0.5, dtype=np.float64),
            "t_bar": np.full(6, 0.1, dtype=np.float64),
            "rel_l2_pct": np.linspace(1.0, 6.0, 6),
            "iface_rel_l2_pct": np.linspace(2.0, 7.0, 6),
        }
        strata = paper_plots._benchmark_strata(records, "interfaces")
        dims = {dim for dim, _g, _m in strata}
        assert ("overall", "all") in [(d, g) for d, g, _m in strata]
        assert "x_I" not in dims
        assert "lead_time" not in dims
        rows = paper_plots.compute_tail_stats(records, benchmark="interfaces")
        assert any(r["stratum"] == "overall" for r in rows)

    def test_benchmark_strata_missing_columns(self, forcing_records, source_records):
        # Forcing has no regime; source has no temporal_family. Strata must build
        # without requiring the absent column.
        f_strata = paper_plots._benchmark_strata(forcing_records, "forcing")
        f_dims = {dim for dim, _g, _m in f_strata}
        assert "temporal_family" in f_dims
        assert "regime" not in f_dims

        s_strata = paper_plots._benchmark_strata(source_records, "source")
        s_dims = {dim for dim, _g, _m in s_strata}
        assert "regime" in s_dims
        assert "temporal_family" not in s_dims

    def test_compute_tail_stats_mixed_benchmark(
        self, forcing_records, source_records
    ):
        # A mixed-benchmark records dict must be split per benchmark internally.
        n_f = int(forcing_records["_n"])
        n_s = int(source_records["_n"])
        mixed = {"_n": np.int64(n_f + n_s)}
        keys = set(forcing_records) | set(source_records)
        keys.discard("_n")
        for key in keys:
            f_arr = forcing_records.get(key)
            s_arr = source_records.get(key)
            if f_arr is None:
                f_arr = np.full(n_f, np.nan) if key in paper_plots._FLOAT_COLS \
                    else np.array([""] * n_f, dtype=object)
            if s_arr is None:
                s_arr = np.full(n_s, np.nan) if key in paper_plots._FLOAT_COLS \
                    else np.array([""] * n_s, dtype=object)
            mixed[key] = np.concatenate([np.asarray(f_arr), np.asarray(s_arr)])
        rows = paper_plots.compute_tail_stats(mixed, benchmark=None)
        assert {r["benchmark"] for r in rows} == {"forcing", "source"}
        # Source overall n must equal the source row count (no mixed-mask leakage).
        src_overall = [r for r in rows if r["benchmark"] == "source"
                       and r["stratum"] == "overall"
                       and r["metric"] == "global_rel_l2_pct"]
        assert len(src_overall) == 1
        assert src_overall[0]["n"] == n_s


class TestPaperPredictionPlots:
    def test_forcing_prediction_smoke(
        self,
        tmp_path,
        small_fno2d_checkpoint,
        synthetic_trajectories,
        synthetic_sim_params,
        plot_config,
        forcing_records,
    ):
        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        model, _ = dataset_plots._load_checkpoint_model(small_fno2d_checkpoint)
        ds = paper_plots._records_dataset(
            model, trajectories, x_grid, y_grid, t_grid, synthetic_sim_params, plot_config
        )
        out_path = tmp_path / "forcing_prediction_truth_residual.png"
        paper_plots.plot_forcing_prediction_truth_residual(
            model, ds, forcing_records, save_path=out_path
        )
        assert out_path.exists()

    def test_source_prediction_smoke(
        self,
        tmp_path,
        small_source_fno2d_checkpoint,
        synthetic_trajectories,
        synthetic_source_sim_params,
        plot_config,
        source_records,
    ):
        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        model, _ = dataset_plots._load_checkpoint_model(small_source_fno2d_checkpoint)
        source_config = {**plot_config, "benchmark": {"name": "source", "representation": "bins"}}
        ds = paper_plots._records_dataset(
            model, trajectories, x_grid, y_grid, t_grid,
            synthetic_source_sim_params, source_config
        )
        out_path = tmp_path / "source_prediction_truth_residual.png"
        paper_plots.plot_source_prediction_truth_residual(
            model, ds, source_records, save_path=out_path
        )
        assert out_path.exists()

    def test_source_itr_prediction_smoke(
        self,
        tmp_path,
        synthetic_trajectories,
        synthetic_source_sim_params,
        plot_config,
        source_itr_records,
    ):
        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        model = _small_source_itr_model()
        source_itr_config = {**plot_config, "benchmark": {"name": "source_itr", "representation": "bins"}}
        sim_params = _source_itr_params_from_source(synthetic_source_sim_params)
        ds = paper_plots._records_dataset(
            model, trajectories, x_grid, y_grid, t_grid, sim_params, source_itr_config
        )
        out_path = tmp_path / "source_itr_prediction_truth_residual.png"
        paper_plots.plot_source_itr_prediction_truth_residual(
            model, ds, source_itr_records, save_path=out_path
        )
        assert out_path.exists()

    def test_forcing_temperature_profiles_smoke(
        self,
        tmp_path,
        small_fno2d_checkpoint,
        synthetic_trajectories,
        synthetic_sim_params,
        plot_config,
        forcing_records,
    ):
        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        model, _ = dataset_plots._load_checkpoint_model(small_fno2d_checkpoint)
        ds = paper_plots._records_dataset(
            model, trajectories, x_grid, y_grid, t_grid, synthetic_sim_params, plot_config
        )
        out_path = tmp_path / "forcing_temperature_profiles.png"
        paper_plots.plot_forcing_temperature_profiles(
            model, ds, forcing_records, plot_config, None, save_path=out_path
        )
        assert out_path.exists()

    def test_source_temperature_profiles_smoke(
        self,
        tmp_path,
        small_source_fno2d_checkpoint,
        synthetic_trajectories,
        synthetic_source_sim_params,
        plot_config,
        source_records,
    ):
        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        model, _ = dataset_plots._load_checkpoint_model(small_source_fno2d_checkpoint)
        source_config = {**plot_config, "benchmark": {"name": "source", "representation": "bins"}}
        ds = paper_plots._records_dataset(
            model, trajectories, x_grid, y_grid, t_grid,
            synthetic_source_sim_params, source_config
        )
        out_path = tmp_path / "source_temperature_profiles.png"
        paper_plots.plot_source_temperature_profiles(
            model, ds, source_records, source_config, None, save_path=out_path
        )
        assert out_path.exists()

    def test_source_itr_temperature_profiles_smoke(
        self,
        tmp_path,
        synthetic_trajectories,
        synthetic_source_sim_params,
        plot_config,
        source_itr_records,
    ):
        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        model = _small_source_itr_model()
        source_itr_config = {**plot_config, "benchmark": {"name": "source_itr", "representation": "bins"}}
        sim_params = _source_itr_params_from_source(synthetic_source_sim_params)
        ds = paper_plots._records_dataset(
            model, trajectories, x_grid, y_grid, t_grid, sim_params, source_itr_config
        )
        out_path = tmp_path / "source_itr_temperature_profiles.png"
        paper_plots.plot_source_itr_temperature_profiles(
            model, ds, source_itr_records, source_itr_config, None, save_path=out_path
        )
        assert out_path.exists()

    def test_interfaces_temperature_profiles_smoke(
        self,
        tmp_path,
        synthetic_trajectories,
        synthetic_sim_params,
        plot_config,
        interfaces_records,
    ):
        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        model = _small_interfaces_model()
        interfaces_config = {**plot_config, "benchmark": {"name": "interfaces", "representation": "temporal_encoder"}}
        sim_params = _interfaces_params_from_forcing(synthetic_sim_params)
        ds = paper_plots._records_dataset(
            model, trajectories, x_grid, y_grid, t_grid, sim_params, interfaces_config
        )
        out_path = tmp_path / "interfaces_temperature_profiles.png"
        paper_plots.plot_interfaces_temperature_profiles(
            model, ds, interfaces_records, interfaces_config, None, save_path=out_path
        )
        assert out_path.exists()

    def test_forcing_interface_jump_profiles_smoke(
        self,
        tmp_path,
        small_fno2d_checkpoint,
        synthetic_trajectories,
        synthetic_sim_params,
        plot_config,
        forcing_records,
    ):
        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        model, _ = dataset_plots._load_checkpoint_model(small_fno2d_checkpoint)
        ds = paper_plots._records_dataset(
            model, trajectories, x_grid, y_grid, t_grid, synthetic_sim_params, plot_config
        )
        out_path = tmp_path / "forcing_interface_jump_profiles.png"
        paper_plots.plot_forcing_interface_jump_profiles(
            model, ds, forcing_records, plot_config, None, save_path=out_path
        )
        assert out_path.exists()

    def test_source_interface_jump_profiles_smoke(
        self,
        tmp_path,
        small_source_fno2d_checkpoint,
        synthetic_trajectories,
        synthetic_source_sim_params,
        plot_config,
        source_records,
    ):
        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        model, _ = dataset_plots._load_checkpoint_model(small_source_fno2d_checkpoint)
        source_config = {**plot_config, "benchmark": {"name": "source", "representation": "bins"}}
        ds = paper_plots._records_dataset(
            model, trajectories, x_grid, y_grid, t_grid,
            synthetic_source_sim_params, source_config
        )
        out_path = tmp_path / "source_interface_jump_profiles.png"
        paper_plots.plot_source_interface_jump_profiles(
            model, ds, source_records, source_config, None, save_path=out_path
        )
        assert out_path.exists()

    def test_source_itr_interface_jump_profiles_smoke(
        self,
        tmp_path,
        synthetic_trajectories,
        synthetic_source_sim_params,
        plot_config,
        source_itr_records,
    ):
        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        model = _small_source_itr_model()
        source_itr_config = {**plot_config, "benchmark": {"name": "source_itr", "representation": "bins"}}
        sim_params = _source_itr_params_from_source(synthetic_source_sim_params)
        ds = paper_plots._records_dataset(
            model, trajectories, x_grid, y_grid, t_grid, sim_params, source_itr_config
        )
        out_path = tmp_path / "source_itr_interface_jump_profiles.png"
        paper_plots.plot_source_itr_interface_jump_profiles(
            model, ds, source_itr_records, source_itr_config, None, save_path=out_path
        )
        assert out_path.exists()

    def test_interfaces_interface_jump_profiles_smoke(
        self,
        tmp_path,
        synthetic_trajectories,
        synthetic_sim_params,
        plot_config,
        interfaces_records,
    ):
        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        model = _small_interfaces_model()
        interfaces_config = {**plot_config, "benchmark": {"name": "interfaces", "representation": "temporal_encoder"}}
        sim_params = _interfaces_params_from_forcing(synthetic_sim_params)
        ds = paper_plots._records_dataset(
            model, trajectories, x_grid, y_grid, t_grid, sim_params, interfaces_config
        )
        out_path = tmp_path / "interfaces_interface_jump_profiles.png"
        paper_plots.plot_interfaces_interface_jump_profiles(
            model, ds, interfaces_records, interfaces_config, None, save_path=out_path
        )
        assert out_path.exists()

    def test_contact_jump_map_accepts_vector_resistance(self):
        x_grid = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        fields = np.zeros((1, 3, 3), dtype=np.float64)
        fields[:, 0, :] = np.array([4.0, 4.0, 4.0])
        fields[:, 1, :] = np.array([1.0, 1.0, 1.0])
        scalar = dataset_plots._interface_contact_jump_map(
            fields, x_grid, 0.25, 0.2, 2.0, 1.0
        )
        vector = dataset_plots._interface_contact_jump_map(
            fields, x_grid, 0.25, np.array([0.1, 0.2, 0.4]), 2.0, 1.0
        )
        assert vector.shape == scalar.shape == (1, 3)
        assert not np.allclose(vector, scalar)
        assert vector[0, 2] > vector[0, 0]


class TestWriteTestRecords:
    def test_writes_csv_with_expected_header(
        self,
        tmp_path,
        small_fno2d,
        synthetic_trajectories,
        synthetic_sim_params,
    ):
        from src.operators.eval import write_test_records

        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        traj_path = tmp_path / "trajectories.npy"
        x_path = tmp_path / "x_grid.npy"
        y_path = tmp_path / "y_grid.npy"
        t_path = tmp_path / "t_grid.npy"
        params_path = tmp_path / "sim_params.npy"
        np.save(traj_path, trajectories)
        np.save(x_path, x_grid)
        np.save(y_path, y_grid)
        np.save(t_path, t_grid)
        np.save(params_path, synthetic_sim_params)

        conf = {
            "data": {
                "trajectories.npy": str(traj_path),
                "x_grid_path": str(x_path),
                "y_grid_path": str(y_path),
                "t_grid_path": str(t_path),
                "sim_params_path": str(params_path),
            },
            "training": {
                "batch_size": 4,
                "n_snapshots_test": 3,
                "device": "cpu",
                "loss": {"interface_half_width": 0.05},
            },
            "model": {
                "parameters": {
                    "modes1": 2,
                    "modes2": 2,
                    "width": 8,
                    "in_channels": 4,
                    "out_channels": 1,
                    "n_layers": 2,
                    "cond_static_dim": 11,
                    "cond_hidden": 256,
                    "temporal_token_dim": 2,
                    "temporal_samples": 128,
                    "temporal_hidden": 16,
                    "forcing_embed_dim": 16,
                }
            },
        }
        seed_dir = tmp_path / "seed42"
        seed_dir.mkdir()
        torch.save(
            {
                "model_state": small_fno2d.state_dict(),
                "conf": conf,
                "best_val": 1.23,
                "mu_global": 0.0,
                "sigma_global": 1.0,
            },
            seed_dir / "fno2d_best.pt",
        )

        out_path = write_test_records(tmp_path)
        assert out_path.exists()
        with open(out_path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            data_rows = list(reader)
        assert header == TEST_RECORD_FIELDS
        assert len(data_rows) > 0
        records = paper_plots._load_test_records(out_path)
        assert np.all(records["benchmark"] == "forcing")
        assert np.all(np.isfinite(records["rel_l2_pct"]))


class TestBenchmarkOverview:
    def test_overview_rows_exactly_four_benchmarks(self):
        # The curated set is the four paper benchmarks, not all of REGISTRY (which
        # may later acquire utility/MMS problems with no paper figure).
        assert set(paper_plots._BENCHMARK_OVERVIEW_ROWS) == {
            "forcing", "source", "source_itr", "interfaces"
        }

    def test_benchmark_overview_smoke(self, tmp_path):
        out_path = tmp_path / "benchmark_overview.png"
        result = paper_plots.plot_benchmark_overview(save_path=out_path)
        assert out_path.exists()
        assert result == out_path


def _gen_records(t_bar, *, benchmark="forcing", sim_ids=None, seed=0,
                 metrics=("rmse_K", "node_jump_gnrmse_pct")):
    """Build a loaded-records-style dict for the generalization helpers."""
    t_bar = np.asarray(t_bar, dtype=np.float64)
    n = int(t_bar.size)
    rng = np.random.default_rng(seed)
    rec = {
        "_n": np.int64(n),
        "t_bar": t_bar,
        "benchmark": np.array([benchmark] * n, dtype=object),
    }
    for m in metrics:
        rec[m] = rng.uniform(1.0, 5.0, n)
    if sim_ids is not None:
        rec["sim_id"] = np.asarray(sim_ids)
    return rec


class TestGeneralizationHelpers:
    def test_shared_edges_drive_both_metric_panels(self):
        same = _gen_records(np.linspace(0.0, 0.30, 12), seed=1)
        unseen = _gen_records(np.linspace(0.05, 0.35, 12), seed=2)
        info = paper_plots._compute_shared_lead_bin_edges(same, unseen, n_lead_bins=4)
        edges = info["lead_bin_edges"]

        union = np.concatenate([same["t_bar"], unseen["t_bar"]])
        expected = np.unique(np.quantile(union, np.linspace(0.0, 1.0, 5)))
        assert np.allclose(edges, expected)
        assert info["n_effective_bins"] == edges.size - 1

        # One binning shared across metrics: identical edges -> identical per-bin
        # counts for two fully-present metrics.
        s1 = paper_plots._compute_generalization_by_lead_bin(
            same, unseen, benchmark="forcing", metric="rmse_K", lead_bin_edges=edges)
        s2 = paper_plots._compute_generalization_by_lead_bin(
            same, unseen, benchmark="forcing", metric="node_jump_gnrmse_pct",
            lead_bin_edges=edges)
        assert np.array_equal(s1["same_sim_counts"], s2["same_sim_counts"])
        assert np.array_equal(s1["unseen_counts"], s2["unseen_counts"])

    def test_tied_quantiles_collapse_bins(self):
        tied = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        info = paper_plots._compute_shared_lead_bin_edges(
            _gen_records(tied, seed=3), _gen_records(tied, seed=4), n_lead_bins=4)
        assert info["n_effective_bins"] < 4
        assert info["lead_bin_edges"].size == info["n_effective_bins"] + 1

    def test_fewer_than_two_distinct_tbar_raises(self):
        const = np.full(6, 0.2)
        with pytest.raises(ValueError, match="two distinct t_bar"):
            paper_plots._compute_shared_lead_bin_edges(
                _gen_records(const, seed=5), _gen_records(const, seed=6))

    def test_empty_record_set_raises(self):
        ok = _gen_records(np.linspace(0.0, 0.3, 8), seed=7)
        empty = _gen_records(np.array([]), seed=8)
        with pytest.raises(ValueError, match="finite t_bar"):
            paper_plots._compute_shared_lead_bin_edges(ok, empty)

    def test_empty_bin_yields_zero_count_and_nan(self):
        # Same-sim t_bar all in [0, 1); hand-made edges put bins 2,3 empty.
        same = _gen_records(np.linspace(0.0, 0.9, 10), seed=9)
        unseen = _gen_records(np.linspace(0.0, 2.9, 10), seed=10)
        edges = np.array([0.0, 1.0, 2.0, 3.0])
        stats = paper_plots._compute_generalization_by_lead_bin(
            same, unseen, benchmark="forcing", metric="rmse_K", lead_bin_edges=edges)
        assert stats["same_sim_counts"][1] == 0
        assert np.isnan(stats["same_sim_median"][1])

    def test_benchmark_mismatch_raises(self):
        same = _gen_records(np.linspace(0.0, 0.3, 6), benchmark="source", seed=11)
        unseen = _gen_records(np.linspace(0.0, 0.3, 6), benchmark="forcing", seed=12)
        edges = np.array([0.0, 0.15, 0.3])
        with pytest.raises(ValueError, match="does not match requested"):
            paper_plots._compute_generalization_by_lead_bin(
                same, unseen, benchmark="forcing", metric="rmse_K", lead_bin_edges=edges)

    def test_record_set_spanning_multiple_benchmarks_raises(self):
        mixed = _gen_records(np.linspace(0.0, 0.3, 6), seed=13)
        mixed["benchmark"] = np.array(
            ["forcing", "source", "forcing", "source", "forcing", "source"], dtype=object)
        with pytest.raises(ValueError, match="span multiple benchmarks"):
            paper_plots._assert_single_benchmark(mixed, "forcing")

    def test_missing_benchmark_column_is_accepted(self):
        rec = _gen_records(np.linspace(0.0, 0.3, 6), seed=14)
        del rec["benchmark"]
        # No column -> kwarg stands alone, no raise.
        paper_plots._assert_single_benchmark(rec, "forcing")


class TestGeneralizationFigure:
    def test_returns_path_not_tuple(self, tmp_path):
        same = _gen_records(np.linspace(0.0, 0.30, 14), sim_ids=range(14), seed=20)
        unseen = _gen_records(np.linspace(0.05, 0.35, 14),
                              sim_ids=range(100, 114), seed=21)
        out_path = tmp_path / "generalization_same_vs_unseen.png"
        result = paper_plots.plot_generalization_same_vs_unseen(
            same, unseen, benchmark="forcing", save_path=out_path)
        assert isinstance(result, Path)
        assert out_path.exists()

    def test_sim_id_overlap_raises(self, tmp_path):
        same = _gen_records(np.linspace(0.0, 0.3, 10), sim_ids=range(10), seed=22)
        unseen = _gen_records(np.linspace(0.05, 0.35, 10), sim_ids=range(5, 15), seed=23)
        with pytest.raises(ValueError, match="share sim_ids"):
            paper_plots.plot_generalization_same_vs_unseen(
                same, unseen, benchmark="forcing",
                save_path=tmp_path / "gen.png")

    def test_sim_id_absent_skips_guard(self, tmp_path):
        # No sim_id column on either set -> disjointness guard skipped, renders.
        same = _gen_records(np.linspace(0.0, 0.30, 12), seed=24)
        unseen = _gen_records(np.linspace(0.05, 0.35, 12), seed=25)
        out_path = tmp_path / "gen_no_simid.png"
        result = paper_plots.plot_generalization_same_vs_unseen(
            same, unseen, benchmark="forcing", save_path=out_path)
        assert result == out_path
        assert out_path.exists()


class TestGeneralizationCLI:
    def test_single_records_flag_prints_skip(self, tmp_path, monkeypatch, capsys):
        import sys

        from visual import cli

        argv = [
            "cli.py", "--plots", "generalization_same_vs_unseen",
            "--out", str(tmp_path),
            "--records-same-sim", str(tmp_path / "same.csv"),
        ]
        monkeypatch.setattr(sys, "argv", argv)
        cli.main()
        out = capsys.readouterr().out
        assert "Skipping generalization_same_vs_unseen" in out
        assert "--records-unseen" in out
        assert "--generalization-benchmark" in out
