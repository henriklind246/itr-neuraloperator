import csv
import json
from math import comb

import matplotlib
import numpy as np
import pytest
import torch

matplotlib.use("Agg")

from data.dataset import split_sim_ids
from src.operators.eval import TEST_RECORD_FIELDS
from visual import _common, dataset_plots, forcing_plots, physics_plots


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
    def test_registry_is_the_recurring_diagnostic_core(self):
        expected = {
            "bc_verification": "physics",
            "itr_temperature_jump_sweep": "physics",
            "mms_convergence": "mms",
            "mms_order_estimation": "mms",
            "training_curves": "training",
            "seed_comparison": "training",
            "prediction_vs_truth": "data",
            "interface_error": "data",
            "lead_time_coverage": "data",
            "initial_conditions": "data",
            "forcing_temporal_families": "forcing",
            "forcing_spatial_profiles": "forcing",
            "forcing_separable_assembly": "forcing",
            "forcing_seq_tokens": "forcing",
            "energy_budget": "source",
            "patch_region_error_map": "source",
            "source_interface_zone_error": "source",
            "source_itr_sin_resistance_profiles": "source",
            "interface_x_breakdown": "interfaces",
        }
        assert _common.PLOT_REGISTRY == expected
        assert set(_common.PLOT_RERUN_TRIGGERS) == set(expected)
        assert all(_common.PLOT_RERUN_TRIGGERS.values())

    def test_retired_plot_names_removed(self):
        retired = {
            "final_temperature", "layer_geometry", "face_conductance",
            "multilayer_evolution", "heat_flux_profile", "trajectory_heatmap",
            "trajectory_deviation_heatmap", "spatial_family_breakdown",
            "ic_uniform_progression",
            "ic_random_sinusoid_progression", "ic_grf_progression",
            "ic_hot_spot_progression", "snapshot_pair_samples",
            "lead_time_error", "parameter_error_slices", "dataset_summary",
            "interface_jump_summary", "forcing_sinusoid_temporal_panels",
            "forcing_zero_shot_field_jump", "forcing_param_distributions_design",
            "forcing_param_distributions_empirical", "source_dataset_summary",
            "patch_param_scatter", "source_temporal_profile",
            "source_field_snapshots", "patch_overlay_trajectory",
            "regime_error_breakdown", "patch_error_slices",
            "source_error_vs_params", "source_itr_sin_error_vs_void_params",
            "interface_y_perturbation", "interface_lhs_scatter",
            "vary_interface_lhs_scatter", "interface_flux_profiles",
            "sin_forcing_profiles", "ic_family_trajectory_breakdown",
            "vary_interface_dataset_summary", "forcing_summary_scalars",
            "benchmark_overview", "rollout_partition_error",
            "resolution_invariance",
        }
        for name in retired:
            assert name not in _common.PLOT_REGISTRY

    def test_should_run_uses_new_registry(self):
        assert _common._should_run("prediction_vs_truth", ["data"], None) is True
        assert _common._should_run("prediction_vs_truth", ["training"], None) is False

    def test_only_diagnostic_groups_remain(self):
        assert _common.GROUPS == {
            "physics", "mms", "training", "data", "forcing", "source", "interfaces"
        }


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

    def test_source_itr_sin_metadata(self, tmp_path):
        result = physics_plots.plot_itr_temperature_jump_sweep(
            benchmarks=("source_itr_sin",),
            Nx=20, Ny=20, t_final=0.1,
            rc_peak_values=(17.5, 350.0),
            requested_times=(0.05, 0.10),
            save_path=tmp_path / "itr_temperature_jump_sweep.png",
        )
        records = result["records"]
        assert len(records) == 4
        for row in records:
            assert np.isfinite(row["mean_abs_jump_K"])
            assert row["itr_kind"] == "Rc_peak"
            assert row["R_c_base"] == 17.5
            assert row["R_c_peak"] == row["itr_value"]
            assert row["R_c_A"] == pytest.approx(row["R_c_peak"] - 17.5)

        # The R_c_peak == 17.5 endpoint is the "no void excess" baseline.
        base_rows = [r for r in records if r["itr_value"] == 17.5]
        assert base_rows
        for row in base_rows:
            assert row["R_c_A"] == pytest.approx(0.0)

    def test_forcing_jump_grows_with_resistance(self, tmp_path):
        # Physical sanity, scoped to the canonical forcing case at the final
        # requested time only: contact-jump magnitude is larger-or-comparable as
        # R_c grows. source/source_itr_sin/interfaces trends are intentionally not
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
        self, tmp_path, synthetic_trajectories, synthetic_sim_params, plot_config
    ):
        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        out_path = tmp_path / "lead_time_coverage.png"
        dataset_plots.plot_lead_time_coverage(
            trajectories,
            x_grid,
            y_grid,
            t_grid,
            synthetic_sim_params,
            plot_config,
            save_path=out_path,
        )
        assert out_path.exists()


class TestInitialConditions:
    def test_smoke_writes_png(self, tmp_path, synthetic_trajectories):
        trajectories, x_grid, y_grid, _t_grid = synthetic_trajectories
        out_path = tmp_path / "initial_conditions.png"
        dataset_plots.plot_initial_conditions(
            trajectories, x_grid, y_grid, n_samples=6, save_path=out_path
        )
        assert out_path.exists()

    def test_sample_count_caps_to_available_sims(self, tmp_path, synthetic_trajectories):
        trajectories, x_grid, y_grid, _t_grid = synthetic_trajectories
        out_path = tmp_path / "initial_conditions.png"
        dataset_plots.plot_initial_conditions(
            trajectories, x_grid, y_grid,
            n_samples=trajectories.shape[0] + 5, save_path=out_path,
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


    def test_seq_tokens_smoke(self, tmp_path):
        out_path = tmp_path / "forcing_seq_tokens.png"
        forcing_plots.plot_forcing_seq_tokens(save_path=out_path)
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


    def test_load_checkpoint_model_loads_2d_checkpoint(self, small_fno2d_checkpoint):
        model, conf = dataset_plots._load_checkpoint_model(small_fno2d_checkpoint)
        assert conf["model"]["parameters"]["modes1"] == 2
        assert conf["model"]["parameters"]["modes2"] == 2
        assert getattr(model, "_mu_global") == 0.0
        assert getattr(model, "_sigma_global") == 1.0

    def test_load_checkpoint_model_rebuilds_rc_conditioned_extender(
        self, small_rc_extender_checkpoint
    ):
        model, conf = dataset_plots._load_checkpoint_model(
            small_rc_extender_checkpoint
        )

        assert conf["model"]["parameters"][
            "forcing_extender_condition_on_rc"
        ] is True
        assert model.forcing_extender_condition_on_rc is True
        assert model.boundary_extender.domain_lift[0].in_features == 4

    def test_load_checkpoint_model_rebuilds_physics_extender(
        self, small_physics_extender_checkpoint
    ):
        model, conf = dataset_plots._load_checkpoint_model(
            small_physics_extender_checkpoint
        )

        params = conf["model"]["parameters"]
        assert params["forcing_spatial_mode"] == "physics_extender"
        assert len(model.boundary_extender.diffusion_geometry_bias_mlps) == 2
        assert model.boundary_extender.diffusion_geometry_bias_mlps[0][0].out_features == 7
        assert model.boundary_extender.diffusion_geometry_interface_x_norm == pytest.approx(0.4)

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
    def test_bc_verification_smoke(self, tmp_path):
        solver = physics_plots.create_demo_multilayer_solver()
        T0 = np.full((solver.Nx, solver.Ny), solver.T_right(0.0))
        _, _, _, T_hist = solver.solve(T0=T0, store_trajectory=True)
        out_path = tmp_path / "bc_verification.png"
        physics_plots.plot_bc_verification(solver, T_hist, save_path=out_path)
        assert out_path.exists()


def _source_itr_sin_params_from_source(synthetic_source_sim_params):
    params = []
    for i, p in enumerate(synthetic_source_sim_params):
        q = dict(p)
        R_base = 0.08 + 0.035 * (i % 12)
        R_amp = 0.0 if i % 6 == 0 else 0.25 + 0.08 * (i % 5)
        q["R_c_base"] = float(R_base)
        q["R_c_A"] = float(R_amp)
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


class TestSourcePlots:
    def test_source_itr_sin_itr_amplitude_handles_flat_profile(self, synthetic_source_sim_params):
        y_grid = np.linspace(0.0, 1.0, 101, dtype=np.float32)
        params = _source_itr_sin_params_from_source(synthetic_source_sim_params[:3])
        params[0]["R_c_A"] = 0.0
        params[1]["R_c_A"] = 0.4
        params[1]["R_c_sigma"] = 0.05
        params[2]["R_c_A"] = 0.4
        params[2]["R_c_sigma"] = 0.16

        severity = dataset_plots._itr_amplitude_arrays(params, y_grid)

        assert severity["R_c_A"][0] == pytest.approx(0.0)
        assert severity["conductance_deficit"][0] == pytest.approx(0.0)
        assert severity["is_itr_active"][0] == np.bool_(False)
        assert severity["R_c_A"][2] == pytest.approx(severity["R_c_A"][1])

    def test_source_itr_sin_resistance_profiles_smoke(
        self,
        tmp_path,
        synthetic_trajectories,
        synthetic_source_sim_params,
    ):
        _trajectories, x_grid, y_grid, _t_grid = synthetic_trajectories
        sim_params = _source_itr_sin_params_from_source(synthetic_source_sim_params)
        out_path = tmp_path / "source_itr_sin_resistance_profiles.png"

        dataset_plots.plot_source_itr_sin_resistance_profiles(
            sim_params,
            y_grid,
            x_grid=x_grid,
            save_path=out_path,
        )

        assert out_path.exists()

    def test_source_checkpoint_loads(self, small_source_fno2d_checkpoint):
        model, conf = dataset_plots._load_checkpoint_model(small_source_fno2d_checkpoint)
        assert conf["model"]["parameters"]["cond_static_dim"] == 6
        assert model.cond_static_dim == 6

    def test_energy_budget_smoke(
        self, tmp_path, synthetic_trajectories, synthetic_source_sim_params
    ):
        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        out_path = tmp_path / "energy_budget.png"
        dataset_plots.plot_energy_budget(
            trajectories,
            x_grid,
            y_grid,
            t_grid,
            synthetic_source_sim_params,
            sim_id=0,
            save_path=out_path,
        )
        assert out_path.exists()

    def test_patch_region_error_map_smoke(
        self,
        tmp_path,
        small_source_fno2d,
        synthetic_trajectories,
        synthetic_source_sim_params,
        plot_config,
    ):
        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        source_config = {
            **plot_config,
            "benchmark": {"name": "source", "representation": "temporal_encoder"},
        }
        dataset = dataset_plots._build_split_datasets(
            trajectories,
            x_grid,
            y_grid,
            t_grid,
            synthetic_source_sim_params,
            source_config,
        )["test"]
        out_path = tmp_path / "patch_region_error_map.png"
        dataset_plots.plot_patch_region_error_map(
            small_source_fno2d,
            dataset,
            x_grid,
            y_grid,
            synthetic_source_sim_params,
            config=source_config,
            max_samples=4,
            save_path=out_path,
        )
        assert out_path.exists()

    def test_source_interface_zone_error_smoke(
        self,
        tmp_path,
        small_source_fno2d,
        synthetic_trajectories,
        synthetic_source_sim_params,
        plot_config,
    ):
        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        source_config = {
            **plot_config,
            "benchmark": {"name": "source", "representation": "temporal_encoder"},
        }
        dataset = dataset_plots._build_split_datasets(
            trajectories,
            x_grid,
            y_grid,
            t_grid,
            synthetic_source_sim_params,
            source_config,
        )["test"]
        out_path = tmp_path / "source_interface_zone_error.png"
        dataset_plots.plot_source_interface_zone_error(
            small_source_fno2d,
            dataset,
            x_grid,
            y_grid,
            source_config,
            max_samples=4,
            save_path=out_path,
        )
        assert out_path.exists()


class TestInterfacesPlots:
    def test_interface_x_breakdown_smoke(
        self, tmp_path, synthetic_trajectories, synthetic_sim_params
    ):
        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        sim_params = _interfaces_params_from_forcing(synthetic_sim_params)
        out_path = tmp_path / "interface_x_breakdown.png"
        dataset_plots.plot_interface_x_breakdown(
            trajectories,
            sim_params,
            x_grid,
            y_grid,
            t_grid,
            n_snaps=2,
            save_path=out_path,
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
        from src.operators.eval import _canonical_hash, write_test_records

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
                    "cond_static_dim": 10,
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
        normalization_sidecar = seed_dir / "normalization_provenance.json"
        normalization_sidecar.write_text(json.dumps({
            "normalization_definition": {"version": "test/v1"},
            "normalization_definition_hash": "definition-hash",
            "training_population_hash": "population-hash",
            "training_population": {"train_ids": [0, 1]},
        }))

        out_path = write_test_records(tmp_path)
        batched_path = write_test_records(
            tmp_path, out_name="test_records_batched.csv", inference_batch_size=2
        )
        assert out_path.exists()
        with open(out_path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            data_rows = list(reader)
        assert header == TEST_RECORD_FIELDS
        assert len(data_rows) > 0
        with open(out_path, newline="") as f:
            record_rows = list(csv.DictReader(f))
        with open(batched_path, newline="") as f:
            batched_rows = list(csv.DictReader(f))
        assert {row["benchmark"] for row in record_rows} == {"forcing"}
        assert all(np.isfinite(float(row["rel_l2_pct"])) for row in record_rows)
        for row, batched_row in zip(record_rows, batched_rows, strict=True):
            for key, value in row.items():
                if key == "provenance_id":
                    continue
                try:
                    expected = float(value)
                    actual = float(batched_row[key])
                except ValueError:
                    assert batched_row[key] == value
                else:
                    assert actual == pytest.approx(expected, rel=1e-5, abs=1e-7)
        assert all(np.isfinite(float(row["node_jump_abs_max_pred_K"])) for row in record_rows)
        assert all(np.isfinite(float(row["node_jump_abs_max_true_K"])) for row in record_rows)

        provenance_path = out_path.with_name(f"{out_path.stem}.provenance.json")
        assert provenance_path.exists()
        provenance = json.loads(provenance_path.read_text())
        assert provenance["provenance_id"]
        assert {row["provenance_id"] for row in record_rows} == {
            provenance["provenance_id"]
        }
        assert provenance["normalization_standard_deviation_convention"] == (
            "population (ddof=0)"
        )
        assert provenance["normalization_definition_hash"] == "definition-hash"
        assert provenance["training_population_hash"] == "population-hash"
        assert provenance["normalization_provenance_source"] == str(
            normalization_sidecar
        )
        assert provenance["normalization_provenance_source_sha256"]
        evaluation_population = provenance["evaluation_population"]
        assert set(evaluation_population["dataset_file_hashes"]) == {
            "sim_params", "t_grid", "trajectories", "x_grid", "y_grid"
        }
        assert provenance["evaluation_population_hash"] == _canonical_hash(
            evaluation_population
        )
