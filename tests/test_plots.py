import csv
import inspect
import json
from math import comb

import matplotlib
import numpy as np
import pytest
import torch

matplotlib.use("Agg")

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

    def test_tail_error_plot_names_registered(self):
        for name in [
            "forcing_tail_errors",
            "source_tail_errors",
            "interfaces_tail_errors",
        ]:
            assert _common.PLOT_REGISTRY[name] == "paper"

    def test_rollout_plot_name_registered(self):
        assert _common.PLOT_REGISTRY["rollout_partition_error"] == "rollout"
        assert "rollout" in _common.GROUPS


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


class TestSourcePlots:
    def test_patch_param_scatter_smoke(self, tmp_path, synthetic_source_sim_params):
        out_path = tmp_path / "patch_param_scatter.png"
        dataset_plots.plot_patch_param_scatter(
            synthetic_source_sim_params, save_path=out_path
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


# ---------- paper-figure helpers ----------

def _write_records_csv(path, rows):
    """Write a minimal test_records.csv (blank cells for absent fields)."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TEST_RECORD_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in TEST_RECORD_FIELDS})


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
def interfaces_records(tmp_path):
    path = tmp_path / "interfaces_records.csv"
    _write_records_csv(path, _interfaces_record_rows())
    return paper_plots._load_test_records(path)


class TestPaperRecordLoading:
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

    def test_all_benchmarks_summary_smoke(
        self, tmp_path, forcing_records, source_records, interfaces_records
    ):
        out_path = tmp_path / "all_benchmarks_test_error_summary.png"
        paper_plots.plot_all_benchmarks_error_summary(
            {"forcing": forcing_records, "source": source_records,
             "interfaces": interfaces_records},
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
