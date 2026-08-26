"""CLI entry point for plot generation.

Run as: ``python -m visual.cli ...`` (see --help for full flag list).

Replaces the old ``python -m visual.plots`` entry. Dispatches to per-group
plot modules based on --group / --plots flags.
"""

from pathlib import Path

import numpy as np

from visual._common import _plots_for_group, _print_skip, _should_run
from visual import (
    dataset_plots,
    forcing_plots,
    inverse_plots,
    mms_plots,
    paper_plots,
    physics_plots,
    resinv_plots,
    rollout_plots,
    training_plots,
)


def _parse_benchmark_path_tokens(tokens, flag):
    """Parse ``benchmark=path`` CLI tokens into a dict (validates the form)."""
    if not tokens:
        return {}
    parsed = {}
    for tok in tokens:
        if "=" not in tok:
            raise ValueError(
                f"{flag} expects benchmark=path tokens; got {tok!r}")
        bm, path = tok.split("=", 1)
        bm, path = bm.strip(), path.strip()
        if not bm or not path:
            raise ValueError(
                f"{flag} expects benchmark=path tokens; got {tok!r}")
        if bm not in inverse_plots._SPECS:
            raise ValueError(
                f"{flag}: unknown benchmark {bm!r} "
                f"(expected one of {sorted(inverse_plots._SPECS)})")
        parsed[bm] = path
    return parsed


def main():
    """
    Usage:
      python -m visual.cli --out visual/                          # all plots (default)
      python -m visual.cli --group physics --out visual/          # physics diagnostics only
      python -m visual.cli --group mms --out visual/              # MMS convergence only
      python -m visual.cli --group training --csv <path> --out visual/
      python -m visual.cli --group data --data <path> --x-grid <path> --y-grid <path> --t-grid <path> --params <path> --out visual/
      python -m visual.cli --group paper --records <test_records.csv> --out visual/
      python -m visual.cli --plots prediction_vs_truth lead_time_error --checkpoint <path> --out visual/

    Optional data flags:
      --data        Path to trajectories .npy file
      --x-grid      Path to x_grid .npy file
      --y-grid      Path to y_grid .npy file
      --t-grid      Path to t_grid .npy file
      --params      Path to sim_params .npy file
      --csv         Path to train_metrics.csv
      --report      Path to seed_report.json
      --checkpoint  Path to model checkpoint .pt (for model diagnostic plots)
    """

    import argparse

    parser = argparse.ArgumentParser(description="Generate plots for the ITR project")
    parser.add_argument("--data", type=str, default=None, help="Path to trajectories .npy file")
    parser.add_argument("--x-grid", type=str, default=None, help="Path to x_grid .npy file")
    parser.add_argument("--y-grid", type=str, default=None, help="Path to y_grid .npy file")
    parser.add_argument("--t-grid", type=str, default=None, help="Path to t_grid .npy file")
    parser.add_argument("--params", type=str, default=None, help="Path to sim_params .npy file")
    parser.add_argument("--csv", type=str, default=None, help="Path to train_metrics.csv")
    parser.add_argument("--report", type=str, default=None, help="Path to seed_report.json")
    parser.add_argument("--resinv-run-root", type=str, default=None,
                        help="Directory with seed_report_r<N>.json files (resolution_invariance)")
    parser.add_argument("--resinv-root", type=str, default=None,
                        help="Directory with fv_drift_baseline.json (resolution_invariance overlay)")
    parser.add_argument("--rollout-run-root", type=str, default=None,
                        help="Config dir with *eval.json reports (rollout_partition_error)")
    parser.add_argument("--rollout-report-glob", type=str, default="*eval*.json",
                        help="Glob for rollout eval reports under --rollout-run-root")
    parser.add_argument("--rollout-allow-missing", action="store_true",
                        help="Plot NaN for rollout reports missing a required metric")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (for interface_error plot)")
    parser.add_argument("--breakdown", type=str, default=None,
                        help="Path to breakdown.json (for regime_error_breakdown)")
    parser.add_argument("--records", type=str, default=None,
                        help="Path to test_records.csv (paper figures)")
    parser.add_argument("--records-forcing", type=str, default=None,
                        help="Path to forcing test_records.csv (combined paper figures)")
    parser.add_argument("--records-source", type=str, default=None,
                        help="Path to source test_records.csv (combined paper figures)")
    parser.add_argument("--records-interfaces", type=str, default=None,
                        help="Path to interfaces test_records.csv (combined paper figures)")
    parser.add_argument("--records-source-itr", type=str, default=None,
                        help="Path to source_itr test_records.csv (combined paper figures)")
    parser.add_argument("--inverse-csv", type=str, nargs="+", default=None,
                        help="Inverse-problem summary CSV(s) as benchmark=path tokens "
                             "(e.g. --inverse-csv forcing=a.csv source_itr=b.csv)")
    parser.add_argument("--inverse-artifacts", type=str, nargs="+", default=None,
                        help="Inverse-problem NPZ artifact dir(s) as benchmark=path tokens "
                             "(e.g. --inverse-artifacts forcing=dirA source_itr=dirB)")
    parser.add_argument("--benchmark", type=str, nargs="+", default=None,
                        help="Restrict inverse figures to these benchmark(s)")
    parser.add_argument("--out", type=str, default=None, help="Output directory for plots")
    parser.add_argument("--group", type=str, nargs="+", default=["all"],
                        choices=["all", "physics", "mms", "training", "data", "forcing", "source", "interfaces", "paper", "resinv", "rollout", "inverse"],
                        help="Which plot group(s) to generate (default: all)")
    parser.add_argument("--plots", type=str, nargs="+", default=None,
                        help="Individual plot names to generate (overrides --group)")
    args = parser.parse_args()

    out_dir = Path(args.out).resolve() if args.out else Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = args.group
    individual = args.plots

    # ---- PHYSICS GROUP ----
    # itr_temperature_jump_sweep builds its own per-benchmark solvers, so it is
    # excluded from the shared demo-solver gate and dispatched standalone below.
    physics_plot_names = [
        p for p in _plots_for_group("physics") if p != "itr_temperature_jump_sweep"
    ]
    need_physics = any(_should_run(p, groups, individual) for p in physics_plot_names)

    if need_physics:
        print("=== PHYSICS GROUP ===")
        solver = physics_plots.create_demo_multilayer_solver()
        T0_demo = np.full((solver.Nx, solver.Ny), solver.T_right(0.0), dtype=float)
        t_sol, gx_sol, gy_sol, T_hist = solver.solve(T0=T0_demo, store_trajectory=True)

        physics_dir = out_dir / "physics"

        if _should_run("final_temperature", groups, individual):
            print("--- final_temperature ---")
            physics_plots.plot_final_temperature(solver, save_path=physics_dir / "final_temperature.png")

        if _should_run("layer_geometry", groups, individual):
            print("--- layer_geometry ---")
            physics_plots.plot_layer_geometry(solver, save_path=physics_dir / "layer_geometry.png")

        if _should_run("face_conductance", groups, individual):
            print("--- face_conductance ---")
            physics_plots.plot_face_conductance(solver, save_path=physics_dir / "face_conductance.png")

        if _should_run("multilayer_evolution", groups, individual):
            print("--- multilayer_evolution ---")
            physics_plots.plot_multilayer_evolution(solver, T_hist, save_path=physics_dir / "multilayer_evolution.png")

        if _should_run("heat_flux_profile", groups, individual):
            print("--- heat_flux_profile ---")
            physics_plots.plot_heat_flux_profile(solver, T_hist, save_path=physics_dir / "heat_flux_profile.png")

        if _should_run("bc_verification", groups, individual):
            print("--- bc_verification ---")
            physics_plots.plot_bc_verification(solver, T_hist, save_path=physics_dir / "bc_verification.png")

    # ---- PHYSICS (self-contained ITR sweep; builds its own solvers) ----
    if _should_run("itr_temperature_jump_sweep", groups, individual):
        print("=== itr_temperature_jump_sweep ===")
        physics_plots.plot_itr_temperature_jump_sweep(
            save_path=out_dir / "physics" / "itr_temperature_jump_sweep.png"
        )

    # ---- MMS GROUP ----
    mms_plot_names = _plots_for_group("mms")
    if any(_should_run(p, groups, individual) for p in mms_plot_names):
        print("=== MMS GROUP ===")

        mms_dir = out_dir / "mms"

        if _should_run("mms_convergence", groups, individual):
            print("--- mms_convergence ---")
            mms_plots.plot_mms_convergence(save_path=mms_dir / "mms_convergence.png")

        if _should_run("mms_order_estimation", groups, individual):
            print("--- mms_order_estimation ---")
            mms_plots.plot_mms_order_estimation(save_path=mms_dir / "mms_order_estimation.png")

    # ---- TRAINING GROUP ----
    training_plot_names = _plots_for_group("training")
    if any(_should_run(p, groups, individual) for p in training_plot_names):
        print("=== TRAINING GROUP ===")

        training_dir = out_dir / "training"

        if _should_run("training_curves", groups, individual):
            if args.csv:
                print("--- training_curves ---")
                training_plots.plot_training_curves(args.csv, save_path=training_dir / "training_curves.png")
            else:
                _print_skip("training_curves", "need --csv")

        if _should_run("seed_comparison", groups, individual):
            if args.report:
                print("--- seed_comparison ---")
                training_plots.plot_seed_comparison(args.report, save_path=training_dir / "seed_comparison.png")
            else:
                _print_skip("seed_comparison", "need --report")

    # ---- DATA GROUP ----
    data_plot_names = _plots_for_group("data")
    need_data = any(_should_run(p, groups, individual) for p in data_plot_names)

    if need_data:
        print("=== DATA GROUP ===")

        data_dir = out_dir / "data"
        checkpoint_conf = None
        model = None

        # IC family progression plots — pure synthetic, no data files needed
        ic_progression_plots = [
            ("ic_uniform_progression",         dataset_plots.plot_ic_uniform_progression),
            ("ic_random_sinusoid_progression", dataset_plots.plot_ic_random_sinusoid_progression),
            ("ic_grf_progression",             dataset_plots.plot_ic_grf_progression),
            ("ic_hot_spot_progression",        dataset_plots.plot_ic_hot_spot_progression),
        ]
        for name, fn in ic_progression_plots:
            if _should_run(name, groups, individual):
                print(f"--- {name} ---")
                fn(save_path=data_dir / f"{name}.png")

        sim_params = None
        if args.params:
            sim_params = np.load(args.params, allow_pickle=True)

        grid_data_plots = {
            "trajectory_heatmap",
            "trajectory_deviation_heatmap",
            "spatial_family_breakdown",
            "initial_conditions",
            "snapshot_pair_samples",
            "prediction_vs_truth",
            "interface_error",
            "lead_time_coverage",
            "lead_time_error",
            "parameter_error_slices",
            "dataset_summary",
            "interface_jump_summary",
        }
        needs_grid_data = any(_should_run(name, groups, individual) for name in grid_data_plots)
        trajectories = None
        x_grid = None
        y_grid = None
        t_grid = None
        plot_config = None
        split_datasets = None

        solver_dt = None
        if needs_grid_data:
            if args.data and args.x_grid and args.y_grid and args.t_grid:
                print(f"Loading trajectories and grids from {args.data}, {args.x_grid}, {args.y_grid}, {args.t_grid} ...")
                trajectories, x_grid, y_grid, t_grid = dataset_plots._load_plot_data(args.data, args.x_grid, args.y_grid, args.t_grid)
                from data.dataset import load_solver_dt
                solver_dt = load_solver_dt(args.t_grid)
            else:
                for name in grid_data_plots:
                    if _should_run(name, groups, individual):
                        _print_skip(name, "need --data, --x-grid, --y-grid, and --t-grid")

        model_plot_names = {
            "prediction_vs_truth",
            "interface_error",
            "lead_time_error",
            "parameter_error_slices",
            "interface_jump_summary",
        }
        needs_model = any(_should_run(name, groups, individual) for name in model_plot_names)
        if needs_model and args.checkpoint:
            try:
                model, checkpoint_conf = dataset_plots._load_checkpoint_model(args.checkpoint)
            except ValueError as exc:
                for name in model_plot_names:
                    if _should_run(name, groups, individual):
                        _print_skip(name, str(exc))
        elif needs_model:
            for name in model_plot_names:
                if _should_run(name, groups, individual):
                    _print_skip(name, "need --checkpoint")

        if trajectories is not None and sim_params is not None:
            plot_config = dataset_plots._resolve_plot_config(checkpoint_conf)
            split_datasets = dataset_plots._build_split_datasets(
                trajectories, x_grid, y_grid, t_grid, sim_params, plot_config,
                dt=solver_dt,
            )

        if trajectories is not None:
            highlight_sid = (
                dataset_plots._select_localized_sim_id(sim_params) if sim_params is not None else 0
            )

            if _should_run("trajectory_heatmap", groups, individual):
                print("--- trajectory_heatmap ---")
                dataset_plots.plot_trajectory_heatmap(
                    trajectories,
                    sim_id=highlight_sid,
                    x_grid=x_grid, y_grid=y_grid, t_grid=t_grid,
                    sim_params=sim_params,
                    t_window=(0.0, 0.20) if sim_params is not None else None,
                    save_path=data_dir / "trajectory_heatmap.png",
                )

            if _should_run("trajectory_deviation_heatmap", groups, individual):
                print("--- trajectory_deviation_heatmap ---")
                dataset_plots.plot_trajectory_deviation_heatmap(
                    trajectories,
                    sim_id=highlight_sid,
                    x_grid=x_grid, y_grid=y_grid, t_grid=t_grid,
                    sim_params=sim_params,
                    t_window=(0.0, 0.20) if sim_params is not None else None,
                    save_path=data_dir / "trajectory_deviation_heatmap.png",
                )

            if _should_run("initial_conditions", groups, individual):
                print("--- initial_conditions ---")
                dataset_plots.plot_initial_conditions(trajectories, x_grid=x_grid, y_grid=y_grid,
                                                     save_path=data_dir / "initial_conditions.png")

            # plots requiring sim_params
            if sim_params is not None:
                if _should_run("spatial_family_breakdown", groups, individual):
                    print("--- spatial_family_breakdown ---")
                    dataset_plots.plot_spatial_family_breakdown(
                        trajectories,
                        sim_params,
                        x_grid=x_grid, y_grid=y_grid, t_grid=t_grid,
                        save_path=data_dir / "spatial_family_breakdown.png",
                    )
                if split_datasets is not None and _should_run("dataset_summary", groups, individual):
                    print("--- dataset_summary ---")
                    dataset_plots.plot_dataset_summary(
                        trajectories,
                        x_grid,
                        y_grid,
                        t_grid,
                        sim_params,
                        config=plot_config,
                        save_path=data_dir / "dataset_summary.png",
                    )

                if split_datasets is not None and _should_run("snapshot_pair_samples", groups, individual):
                    print("--- snapshot_pair_samples ---")
                    dataset_plots.plot_snapshot_pair_samples(
                        trajectories,
                        x_grid,
                        y_grid,
                        t_grid,
                        sim_params,
                        config=plot_config,
                        save_path=data_dir / "snapshot_pair_samples.png",
                    )

                if split_datasets is not None and _should_run("lead_time_coverage", groups, individual):
                    print("--- lead_time_coverage ---")
                    dataset_plots.plot_lead_time_coverage(
                        trajectories,
                        x_grid,
                        y_grid,
                        t_grid,
                        sim_params,
                        config=plot_config,
                        save_path=data_dir / "lead_time_coverage.png",
                    )

                if model is not None and split_datasets is not None:
                    test_dataset = split_datasets["test"]
                    if _should_run("prediction_vs_truth", groups, individual):
                        print("--- prediction_vs_truth ---")
                        representative_idx = min(len(test_dataset) // 2, len(test_dataset) - 1)
                        sim_id, s, _ = test_dataset._pairs[representative_idx]
                        n_steps = max(2, min(40, len(t_grid) - s - 1))
                        dataset_plots.plot_prediction_vs_truth(
                            model,
                            trajectories,
                            x_grid,
                            y_grid,
                            t_grid,
                            sim_params,
                            sim_id=sim_id,
                            s=s,
                            n_steps=n_steps,
                            config=plot_config,
                            save_path=data_dir / "prediction_vs_truth.png",
                            dt=solver_dt,
                        )

                    if _should_run("interface_error", groups, individual):
                        print("--- interface_error ---")
                        dataset_plots.plot_interface_error(
                            model,
                            trajectories,
                            x_grid,
                            y_grid,
                            t_grid,
                            sim_params,
                            test_dataset.sim_ids,
                            config=plot_config,
                            dt=solver_dt,
                            save_path=data_dir / "interface_error.png",
                        )

                    if _should_run("lead_time_error", groups, individual):
                        print("--- lead_time_error ---")
                        dataset_plots.plot_lead_time_error(
                            model,
                            test_dataset,
                            x_grid,
                            y_grid,
                            config=plot_config,
                            save_path=data_dir / "lead_time_error.png",
                        )

                    if _should_run("parameter_error_slices", groups, individual):
                        print("--- parameter_error_slices ---")
                        dataset_plots.plot_parameter_error_slices(
                            model,
                            test_dataset,
                            x_grid,
                            y_grid,
                            config=plot_config,
                            save_path=data_dir / "parameter_error_slices.png",
                        )

                    if _should_run("interface_jump_summary", groups, individual):
                        print("--- interface_jump_summary ---")
                        dataset_plots.plot_interface_jump_summary(
                            model,
                            test_dataset,
                            x_grid,
                            y_grid,
                            config=plot_config,
                            save_path=data_dir / "interface_jump_summary.png",
                        )
            else:
                for name in {
                    "snapshot_pair_samples",
                    "lead_time_coverage",
                    "dataset_summary",
                    "prediction_vs_truth",
                    "interface_error",
                    "lead_time_error",
                    "parameter_error_slices",
                    "interface_jump_summary",
                }:
                    if _should_run(name, groups, individual):
                        _print_skip(name, "need --params")
        else:
            if needs_grid_data:
                for name in grid_data_plots:
                    if _should_run(name, groups, individual):
                        _print_skip(name, "trajectory grids are unavailable")

    # ---- FORCING GROUP ----
    forcing_plot_names = _plots_for_group("forcing")
    need_forcing = any(_should_run(p, groups, individual) for p in forcing_plot_names)

    if need_forcing:
        print("=== FORCING GROUP ===")
        forcing_dir = out_dir / "forcing"

        if _should_run("forcing_temporal_families", groups, individual):
            print("--- forcing_temporal_families ---")
            forcing_plots.plot_forcing_temporal_families(
                save_path=forcing_dir / "forcing_temporal_families.png",
            )

        if _should_run("forcing_spatial_profiles", groups, individual):
            print("--- forcing_spatial_profiles ---")
            forcing_plots.plot_forcing_spatial_profiles(
                save_path=forcing_dir / "forcing_spatial_profiles.png",
            )

        if _should_run("forcing_separable_assembly", groups, individual):
            print("--- forcing_separable_assembly ---")
            forcing_plots.plot_forcing_separable_assembly(
                save_path=forcing_dir / "forcing_separable_assembly.png",
            )

        if _should_run("forcing_sinusoid_temporal_panels", groups, individual):
            print("--- forcing_sinusoid_temporal_panels ---")
            forcing_plots.plot_forcing_sinusoid_temporal_panels(
                save_path=forcing_dir / "forcing_sinusoid_temporal_panels.png",
            )

        if _should_run("forcing_zero_shot_field_jump", groups, individual):
            if args.checkpoint and args.data:
                print("--- forcing_zero_shot_field_jump ---")
                forcing_plots.generate_zero_shot_sinusoid_field_and_jump(
                    checkpoint_path=args.checkpoint,
                    data_dir=Path(args.data).parent,
                    save_path=forcing_dir / "forcing_zero_shot_field_jump",
                )
            else:
                _print_skip(
                    "forcing_zero_shot_field_jump", "need --checkpoint and --data"
                )

        if _should_run("forcing_seq_tokens", groups, individual):
            print("--- forcing_seq_tokens ---")
            forcing_plots.plot_forcing_seq_tokens(
                save_path=forcing_dir / "forcing_seq_tokens.png",
            )

        if _should_run("forcing_summary_scalars", groups, individual):
            print("--- forcing_summary_scalars ---")
            forcing_plots.plot_forcing_summary_scalars(
                save_path=forcing_dir / "forcing_summary_scalars.png",
            )

        if _should_run("forcing_param_distributions_design", groups, individual):
            print("--- forcing_param_distributions_design ---")
            forcing_plots.plot_forcing_param_distributions_design(
                save_path=forcing_dir / "forcing_param_distributions_design.png",
            )

        if _should_run("forcing_param_distributions_empirical", groups, individual):
            if args.params:
                sim_params_for_forcing = (
                    sim_params
                    if "sim_params" in locals() and sim_params is not None
                    else np.load(args.params, allow_pickle=True)
                )
                print("--- forcing_param_distributions_empirical ---")
                forcing_plots.plot_forcing_param_distributions_empirical(
                    sim_params_for_forcing,
                    save_path=forcing_dir / "forcing_param_distributions_empirical.png",
                )
            else:
                _print_skip("forcing_param_distributions_empirical", "need --params")

    # ---- SOURCE GROUP ----
    source_plot_names = _plots_for_group("source")
    need_source = any(_should_run(p, groups, individual) for p in source_plot_names)

    if need_source:
        print("=== SOURCE GROUP ===")
        source_dir = out_dir / "source"
        checkpoint_conf = None
        model = None

        sim_params = None
        if args.params:
            sim_params = np.load(args.params, allow_pickle=True)

        grid_data_plots = {
            "source_dataset_summary",
            "source_temporal_profile",
            "source_field_snapshots",
            "patch_overlay_trajectory",
            "energy_budget",
            "patch_param_scatter",
            "source_error_vs_params",
            "source_interface_zone_error",
            "patch_error_slices",
            "patch_region_error_map",
            "source_itr_error_vs_void_params",
        }
        y_grid_only_plots = {"source_itr_void_profiles"}
        needs_grid_data = any(_should_run(name, groups, individual) for name in grid_data_plots)
        needs_y_grid_only = any(_should_run(name, groups, individual) for name in y_grid_only_plots)
        trajectories = x_grid = y_grid = t_grid = None
        profile_x_grid = profile_y_grid = None
        plot_config = None
        split_datasets = None
        solver_dt = None
        if needs_y_grid_only:
            if args.y_grid:
                profile_y_grid = np.load(args.y_grid)
                profile_x_grid = np.load(args.x_grid) if args.x_grid else None
            else:
                for name in y_grid_only_plots:
                    if _should_run(name, groups, individual):
                        _print_skip(name, "need --y-grid")

        if needs_grid_data:
            if args.data and args.x_grid and args.y_grid and args.t_grid:
                print(f"Loading trajectories and grids from {args.data}, {args.x_grid}, {args.y_grid}, {args.t_grid} ...")
                trajectories, x_grid, y_grid, t_grid = dataset_plots._load_plot_data(args.data, args.x_grid, args.y_grid, args.t_grid)
                from data.dataset import load_solver_dt
                solver_dt = load_solver_dt(args.t_grid)
                if profile_y_grid is None:
                    profile_y_grid = y_grid
                    profile_x_grid = x_grid
            else:
                for name in grid_data_plots:
                    if _should_run(name, groups, individual):
                        _print_skip(name, "need --data, --x-grid, --y-grid, and --t-grid")

        model_plot_names = {
            "source_error_vs_params",
            "source_interface_zone_error",
            "patch_error_slices",
            "patch_region_error_map",
            "source_itr_error_vs_void_params",
        }
        needs_model = any(_should_run(name, groups, individual) for name in model_plot_names)
        if needs_model and args.checkpoint:
            try:
                model, checkpoint_conf = dataset_plots._load_checkpoint_model(args.checkpoint)
            except ValueError as exc:
                for name in model_plot_names:
                    if _should_run(name, groups, individual):
                        _print_skip(name, str(exc))
        elif needs_model:
            for name in model_plot_names:
                if _should_run(name, groups, individual):
                    _print_skip(name, "need --checkpoint")

        if trajectories is not None and sim_params is not None:
            plot_config = dataset_plots._resolve_plot_config(checkpoint_conf)
            split_datasets = dataset_plots._build_split_datasets(
                trajectories, x_grid, y_grid, t_grid, sim_params, plot_config,
                dt=solver_dt,
            )

        if trajectories is not None and sim_params is not None:
            if _should_run("patch_param_scatter", groups, individual):
                print("--- patch_param_scatter ---")
                dataset_plots.plot_patch_param_scatter(
                    sim_params, save_path=source_dir / "patch_param_scatter.png")

            if _should_run("source_itr_void_profiles", groups, individual) and profile_y_grid is not None:
                print("--- source_itr_void_profiles ---")
                dataset_plots.plot_source_itr_void_profiles(
                    sim_params,
                    profile_y_grid,
                    x_grid=profile_x_grid,
                    save_path=source_dir / "source_itr_void_profiles.png",
                )

            if _should_run("source_temporal_profile", groups, individual):
                print("--- source_temporal_profile ---")
                dataset_plots.plot_source_temporal_profile(
                    sim_params, t_grid, save_path=source_dir / "source_temporal_profile.png")

            if _should_run("source_field_snapshots", groups, individual):
                print("--- source_field_snapshots ---")
                dataset_plots.plot_source_field_snapshots(
                    trajectories, x_grid, y_grid, t_grid, sim_params,
                    save_path=source_dir / "source_field_snapshots.png")

            if _should_run("patch_overlay_trajectory", groups, individual):
                print("--- patch_overlay_trajectory ---")
                dataset_plots.plot_patch_overlay_trajectory(
                    trajectories, x_grid, y_grid, t_grid, sim_params,
                    save_path=source_dir / "patch_overlay_trajectory.png")

            if _should_run("energy_budget", groups, individual):
                print("--- energy_budget ---")
                dataset_plots.plot_energy_budget(
                    trajectories, x_grid, y_grid, t_grid, sim_params,
                    save_path=source_dir / "energy_budget.png")

            if split_datasets is not None and _should_run("source_dataset_summary", groups, individual):
                print("--- source_dataset_summary ---")
                dataset_plots.plot_source_dataset_summary(
                    trajectories, x_grid, y_grid, t_grid, sim_params,
                    config=plot_config, save_path=source_dir / "source_dataset_summary.png")

            if model is not None and split_datasets is not None:
                test_dataset = split_datasets["test"]
                if _should_run("source_error_vs_params", groups, individual):
                    print("--- source_error_vs_params ---")
                    dataset_plots.plot_source_error_vs_params(
                        model, test_dataset, x_grid, y_grid,
                        config=plot_config, save_path=source_dir / "source_error_vs_params.png")

                if _should_run("source_interface_zone_error", groups, individual):
                    print("--- source_interface_zone_error ---")
                    dataset_plots.plot_source_interface_zone_error(
                        model, test_dataset, x_grid, y_grid,
                        config=plot_config, save_path=source_dir / "source_interface_zone_error.png")

                if _should_run("patch_error_slices", groups, individual):
                    print("--- patch_error_slices ---")
                    dataset_plots.plot_patch_error_slices(
                        model, test_dataset, x_grid, y_grid,
                        config=plot_config, sim_params=sim_params,
                        save_path=source_dir / "patch_error_slices.png")

                if _should_run("patch_region_error_map", groups, individual):
                    print("--- patch_region_error_map ---")
                    dataset_plots.plot_patch_region_error_map(
                        model, test_dataset, x_grid, y_grid, sim_params,
                        config=plot_config, save_path=source_dir / "patch_region_error_map.png")

                if _should_run("source_itr_error_vs_void_params", groups, individual):
                    print("--- source_itr_error_vs_void_params ---")
                    dataset_plots.plot_source_itr_error_vs_void_params(
                        model, test_dataset, x_grid, y_grid,
                        config=plot_config,
                        save_path=source_dir / "source_itr_error_vs_void_params.png")
        elif needs_grid_data and trajectories is not None and sim_params is None:
            for name in grid_data_plots:
                if _should_run(name, groups, individual):
                    _print_skip(name, "need --params")

        if trajectories is None and sim_params is not None and _should_run("source_itr_void_profiles", groups, individual):
            if profile_y_grid is not None:
                print("--- source_itr_void_profiles ---")
                dataset_plots.plot_source_itr_void_profiles(
                    sim_params,
                    profile_y_grid,
                    x_grid=profile_x_grid,
                    save_path=source_dir / "source_itr_void_profiles.png",
                )
        elif trajectories is None and sim_params is None and _should_run("source_itr_void_profiles", groups, individual):
            _print_skip("source_itr_void_profiles", "need --params")

        if _should_run("regime_error_breakdown", groups, individual):
            breakdown_path = args.breakdown
            if breakdown_path is None and args.checkpoint:
                candidate = Path(args.checkpoint).parent / "breakdown.json"
                if candidate.is_file():
                    breakdown_path = str(candidate)
            if breakdown_path is not None:
                print("--- regime_error_breakdown ---")
                dataset_plots.plot_regime_error_breakdown(
                    breakdown_path, save_path=source_dir / "regime_error_breakdown.png")
            else:
                _print_skip("regime_error_breakdown", "need --breakdown or checkpoint dir")

    # ---- INTERFACES GROUP ----
    interfaces_plot_names = _plots_for_group("interfaces")
    need_interfaces = any(_should_run(p, groups, individual) for p in interfaces_plot_names)

    if need_interfaces:
        print("=== INTERFACES GROUP ===")
        interfaces_dir = out_dir / "interfaces"

        # Self-contained plots — no data files needed.
        if _should_run("interface_flux_profiles", groups, individual):
            print("--- interface_flux_profiles ---")
            dataset_plots.plot_interface_flux_profiles(
                save_path=interfaces_dir / "interface_flux_profiles.png")

        if _should_run("sin_forcing_profiles", groups, individual):
            print("--- sin_forcing_profiles ---")
            dataset_plots.plot_sin_forcing_profiles(
                save_path=interfaces_dir / "sin_forcing_profiles.png")

        sim_params = None
        if args.params:
            sim_params = np.load(args.params, allow_pickle=True)

        grid_data_plots = {
            "interface_y_perturbation",
            "interface_x_breakdown",
            "ic_family_trajectory_breakdown",
            "vary_interface_dataset_summary",
        }
        param_only_plots = {"interface_lhs_scatter", "vary_interface_lhs_scatter"}
        needs_grid_data = any(_should_run(name, groups, individual) for name in grid_data_plots)
        trajectories = x_grid = y_grid = t_grid = None
        plot_config = None
        split_datasets = None
        solver_dt = None
        if needs_grid_data:
            if args.data and args.x_grid and args.y_grid and args.t_grid:
                print(f"Loading trajectories and grids from {args.data}, {args.x_grid}, {args.y_grid}, {args.t_grid} ...")
                trajectories, x_grid, y_grid, t_grid = dataset_plots._load_plot_data(args.data, args.x_grid, args.y_grid, args.t_grid)
                from data.dataset import load_solver_dt
                solver_dt = load_solver_dt(args.t_grid)
            else:
                for name in grid_data_plots:
                    if _should_run(name, groups, individual):
                        _print_skip(name, "need --data, --x-grid, --y-grid, and --t-grid")

        if sim_params is not None:
            if _should_run("interface_lhs_scatter", groups, individual):
                print("--- interface_lhs_scatter ---")
                dataset_plots.plot_interface_lhs_scatter(
                    sim_params, save_path=interfaces_dir / "interface_lhs_scatter.png")

            if _should_run("vary_interface_lhs_scatter", groups, individual):
                print("--- vary_interface_lhs_scatter ---")
                dataset_plots.plot_vary_interface_lhs_scatter(
                    sim_params, save_path=interfaces_dir / "vary_interface_lhs_scatter.png")
        else:
            for name in param_only_plots:
                if _should_run(name, groups, individual):
                    _print_skip(name, "need --params")

        if trajectories is not None and sim_params is not None:
            plot_config = dataset_plots._resolve_plot_config(None)
            highlight_sid = dataset_plots._select_localized_sim_id(sim_params)

            if _should_run("interface_y_perturbation", groups, individual):
                print("--- interface_y_perturbation ---")
                dataset_plots.plot_interface_y_perturbation(
                    trajectories, highlight_sid, x_grid, y_grid, t_grid,
                    sim_params=sim_params,
                    save_path=interfaces_dir / "interface_y_perturbation.png")

            if _should_run("interface_x_breakdown", groups, individual):
                print("--- interface_x_breakdown ---")
                dataset_plots.plot_interface_x_breakdown(
                    trajectories, sim_params, x_grid, y_grid, t_grid,
                    save_path=interfaces_dir / "interface_x_breakdown.png")

            if _should_run("ic_family_trajectory_breakdown", groups, individual):
                print("--- ic_family_trajectory_breakdown ---")
                dataset_plots.plot_ic_family_trajectory_breakdown(
                    trajectories, sim_params, x_grid, y_grid, t_grid,
                    save_path=interfaces_dir / "ic_family_trajectory_breakdown.png")

            if _should_run("vary_interface_dataset_summary", groups, individual):
                print("--- vary_interface_dataset_summary ---")
                dataset_plots.plot_vary_interface_dataset_summary(
                    trajectories, x_grid, y_grid, t_grid, sim_params,
                    config=plot_config,
                    save_path=interfaces_dir / "vary_interface_dataset_summary.png")
        elif needs_grid_data and trajectories is not None and sim_params is None:
            for name in grid_data_plots:
                if _should_run(name, groups, individual):
                    _print_skip(name, "need --params")

    # ---- PAPER GROUP ----
    paper_plot_names = _plots_for_group("paper")
    need_paper = any(_should_run(p, groups, individual) for p in paper_plot_names)

    if need_paper:
        print("=== PAPER GROUP ===")
        paper_dir = out_dir / "paper"

        # benchmark_overview reads no model or CSV, so it always renders.
        if _should_run("benchmark_overview", groups, individual):
            print("--- benchmark_overview ---")
            paper_dir.mkdir(parents=True, exist_ok=True)
            paper_plots.plot_benchmark_overview(
                save_path=paper_dir / "benchmark_overview.png")

        active_records = None
        records_path = args.records or args.csv
        if records_path:
            active_records = paper_plots._load_test_records(records_path)

        summary_plots = {
            "forcing_test_error_summary": paper_plots.plot_forcing_test_error_summary,
            "source_test_error_summary": paper_plots.plot_source_test_error_summary,
            "source_itr_test_error_summary": paper_plots.plot_source_itr_test_error_summary,
            "interfaces_test_error_summary": paper_plots.plot_interfaces_test_error_summary,
        }
        for name, fn in summary_plots.items():
            if _should_run(name, groups, individual):
                if active_records is not None:
                    print(f"--- {name} ---")
                    fn(active_records, save_path=paper_dir / f"{name}.png")
                else:
                    _print_skip(name, "need --records (or --csv) pointing to test_records.csv")

        tail_plot_names = ("forcing_tail_errors", "source_tail_errors", "interfaces_tail_errors")
        for name in tail_plot_names:
            if _should_run(name, groups, individual):
                if active_records is not None:
                    print(f"--- {name} ---")
                    paper_plots.plot_benchmark_tail_errors(
                        active_records, save_path=paper_dir / f"{name}.png")
                else:
                    _print_skip(name, "need --records (or --csv) pointing to test_records.csv")

        # tail_summary.csv is decoupled from plot selection: write it whenever
        # single-records data is available, keyed by the benchmark in the records.
        if active_records is not None and int(active_records.get("_n", 0)) > 0:
            uniq = sorted({str(v) for v in active_records["benchmark"]}) \
                if "benchmark" in active_records else []
            if len(uniq) > 1:
                records_by_benchmark = {
                    bm: paper_plots._slice_records(
                        active_records, paper_plots._mask_in(active_records["benchmark"], bm))
                    for bm in uniq
                }
            else:
                records_by_benchmark = {(uniq[0] if uniq else "all"): active_records}
            paper_dir.mkdir(parents=True, exist_ok=True)
            out_csv = paper_plots.write_tail_summary(
                records_by_benchmark, paper_dir / "tail_summary.csv")
            print(f"Wrote tail summary to: {out_csv}")

        prediction_plots = {
            "forcing_prediction_truth_residual": paper_plots.plot_forcing_prediction_truth_residual,
            "source_prediction_truth_residual": paper_plots.plot_source_prediction_truth_residual,
            "source_itr_prediction_truth_residual": paper_plots.plot_source_itr_prediction_truth_residual,
            "interfaces_prediction_truth_residual": paper_plots.plot_interfaces_prediction_truth_residual,
        }
        temperature_profile_plots = {
            "forcing_temperature_profiles": paper_plots.plot_forcing_temperature_profiles,
            "source_temperature_profiles": paper_plots.plot_source_temperature_profiles,
            "source_itr_temperature_profiles": paper_plots.plot_source_itr_temperature_profiles,
            "interfaces_temperature_profiles": paper_plots.plot_interfaces_temperature_profiles,
        }
        # Per-benchmark contact-jump plots share the same model/ds inputs but take
        # the extra (config, dt) needed to compute the physical interface law.
        jump_plots = {
            "forcing_interface_jump": paper_plots.plot_forcing_interface_jump,
            "source_interface_jump": paper_plots.plot_source_interface_jump,
            "interfaces_interface_jump": paper_plots.plot_interfaces_interface_jump,
        }
        jump_profile_plots = {
            "forcing_interface_jump_profiles": paper_plots.plot_forcing_interface_jump_profiles,
            "source_interface_jump_profiles": paper_plots.plot_source_interface_jump_profiles,
            "source_itr_interface_jump_profiles": paper_plots.plot_source_itr_interface_jump_profiles,
            "interfaces_interface_jump_profiles": paper_plots.plot_interfaces_interface_jump_profiles,
        }
        needs_pred = any(
            _should_run(name, groups, individual)
            for name in (*prediction_plots, *temperature_profile_plots, *jump_plots, *jump_profile_plots)
        )
        if needs_pred:
            have_inputs = bool(
                active_records is not None and args.checkpoint and args.data
                and args.x_grid and args.y_grid and args.t_grid and args.params
            )
            if not have_inputs:
                for name in (*prediction_plots, *temperature_profile_plots, *jump_plots, *jump_profile_plots):
                    if _should_run(name, groups, individual):
                        _print_skip(name, "need --records, --checkpoint, --data, --x-grid, --y-grid, --t-grid, --params")
            else:
                try:
                    model, checkpoint_conf = dataset_plots._load_checkpoint_model(args.checkpoint)
                except ValueError as exc:
                    model = None
                    for name in (*prediction_plots, *temperature_profile_plots, *jump_plots, *jump_profile_plots):
                        if _should_run(name, groups, individual):
                            _print_skip(name, str(exc))
                if model is not None:
                    trajectories, x_grid, y_grid, t_grid = dataset_plots._load_plot_data(
                        args.data, args.x_grid, args.y_grid, args.t_grid)
                    from data.dataset import load_solver_dt
                    solver_dt = load_solver_dt(args.t_grid)
                    sim_params = np.load(args.params, allow_pickle=True)
                    plot_config = dataset_plots._resolve_plot_config(checkpoint_conf)
                    ds = paper_plots._records_dataset(
                        model, trajectories, x_grid, y_grid, t_grid, sim_params,
                        plot_config, dt=solver_dt)
                    for name, fn in prediction_plots.items():
                        if _should_run(name, groups, individual):
                            print(f"--- {name} ---")
                            fn(model, ds, active_records, save_path=paper_dir / f"{name}.png")
                    for name, fn in temperature_profile_plots.items():
                        if _should_run(name, groups, individual):
                            print(f"--- {name} ---")
                            fn(model, ds, active_records, plot_config, solver_dt,
                               save_path=paper_dir / f"{name}.png")
                    for name, fn in jump_plots.items():
                        if _should_run(name, groups, individual):
                            print(f"--- {name} ---")
                            fn(model, ds, active_records, plot_config, solver_dt,
                               save_path=paper_dir / f"{name}.png")
                    for name, fn in jump_profile_plots.items():
                        if _should_run(name, groups, individual):
                            print(f"--- {name} ---")
                            fn(model, ds, active_records, plot_config, solver_dt,
                               save_path=paper_dir / f"{name}.png")

        csv_map = {
            "forcing": args.records_forcing,
            "source": args.records_source,
            "interfaces": args.records_interfaces,
            "source_itr": args.records_source_itr,
        }
        supplied = {name: path for name, path in csv_map.items() if path}
        combined_records = {
            name: paper_plots._load_test_records(path) for name, path in supplied.items()
        }

        if _should_run("all_benchmarks_error_summary", groups, individual):
            if len(combined_records) >= 2:
                print("--- all_benchmarks_error_summary ---")
                paper_plots.plot_all_benchmarks_error_summary(
                    combined_records, save_path=paper_dir / "all_benchmarks_error_summary.png")
            else:
                _print_skip("all_benchmarks_error_summary",
                            "need at least two of --records-forcing, --records-source, "
                            "--records-interfaces, --records-source-itr")

        if _should_run("all_benchmarks_tail_errors", groups, individual):
            if len(combined_records) >= 2:
                print("--- all_benchmarks_tail_errors ---")
                paper_plots.plot_all_benchmarks_tail_errors(
                    combined_records, save_path=paper_dir / "all_benchmarks_tail_errors.png")
            else:
                _print_skip("all_benchmarks_tail_errors",
                            "need at least two of --records-forcing, --records-source, "
                            "--records-interfaces, --records-source-itr")

        # In combined mode, write one tail_summary.csv covering whichever
        # benchmark CSVs were supplied, regardless of which plots were requested.
        if combined_records:
            paper_dir.mkdir(parents=True, exist_ok=True)
            out_csv = paper_plots.write_tail_summary(
                combined_records, paper_dir / "tail_summary.csv")
            print(f"Wrote tail summary to: {out_csv}")

        if _should_run("all_benchmarks_prediction_truth_residual", groups, individual):
            _print_skip(
                "all_benchmarks_prediction_truth_residual",
                "render programmatically: needs a (checkpoint, trajectories, params, records) bundle "
                "per benchmark (forcing, source, interfaces, source_itr)",
            )

        if _should_run("all_benchmarks_interface_jump", groups, individual):
            _print_skip(
                "all_benchmarks_interface_jump",
                "render programmatically via paper_plots.plot_all_benchmarks_interface_jump: "
                "needs a (model, ds, records, config, dt) context per benchmark "
                "(forcing, source, interfaces, source_itr)",
            )

    # ---- RESINV GROUP ----
    if _should_run("resolution_invariance", groups, individual):
        print("=== RESINV GROUP ===")
        if args.resinv_run_root:
            print("--- resolution_invariance ---")
            resinv_plots.plot_resolution_invariance(
                run_root=args.resinv_run_root,
                resinv_root=args.resinv_root,
                save_path=out_dir / "resinv" / "resolution_invariance.png",
            )
        else:
            _print_skip("resolution_invariance", "need --resinv-run-root")

    # ---- ROLLOUT GROUP ----
    if _should_run("rollout_partition_error", groups, individual):
        print("=== ROLLOUT GROUP ===")
        if args.rollout_run_root:
            print("--- rollout_partition_error ---")
            rollout_plots.plot_rollout_partition_error(
                run_root=args.rollout_run_root,
                report_glob=args.rollout_report_glob,
                allow_missing=args.rollout_allow_missing,
                save_path=out_dir / "rollout" / "rollout_partition_error.png",
            )
        else:
            _print_skip("rollout_partition_error", "need --rollout-run-root")

    # ---- INVERSE GROUP ----
    inverse_plot_names = _plots_for_group("inverse")
    need_inverse = any(_should_run(p, groups, individual) for p in inverse_plot_names)

    if need_inverse:
        print("=== INVERSE GROUP ===")
        inverse_dir = out_dir / "inverse"

        # benchmark=path tokens so a forcing spec can never pair with a
        # source_itr CSV/artifact dir.
        csv_by_bench = _parse_benchmark_path_tokens(args.inverse_csv, "--inverse-csv")
        art_by_bench = _parse_benchmark_path_tokens(
            args.inverse_artifacts, "--inverse-artifacts")

        bench_filter = set(args.benchmark) if args.benchmark else None
        fn_by_kind = {
            "parameter_recovery": inverse_plots.plot_parameter_recovery,
            "identifiability": inverse_plots.plot_identifiability,
            "surrogate_fidelity": inverse_plots.plot_surrogate_fidelity,
            "uncertainty": inverse_plots.plot_uncertainty,
        }
        # artifact_dir is only consumed by these two (Figs 2 & 4).
        takes_artifacts = {"identifiability", "uncertainty"}

        for benchmark in sorted(inverse_plots._SPECS):
            if bench_filter is not None and benchmark not in bench_filter:
                continue
            spec = inverse_plots.spec_for(benchmark)
            csv_path = csv_by_bench.get(benchmark)
            art_dir = art_by_bench.get(benchmark)
            # One probe column per figure whose statistic scripts/invert.py no
            # longer reports. The identifiability figure reads the retired
            # Jacobian-SVD block, and the uncertainty figure needs a profile
            # interval (present only when a calibration artifact was supplied).
            # Skip with a reason rather than crashing on a missing column; see
            # the RETIRED STATISTICS section of scripts/invert.py.
            probe_by_kind = {
                "identifiability": spec.sv_cols[0],
                "uncertainty": spec.lead_profile_ci_low_col,
            }
            available = (
                set(inverse_plots._load_inverse_csv(csv_path))
                if csv_path is not None
                else set()
            )
            for kind, fn in fn_by_kind.items():
                name = f"{benchmark}_{kind}"
                if not _should_run(name, groups, individual):
                    continue
                if csv_path is None:
                    _print_skip(name, f"need --inverse-csv {benchmark}=<path>")
                    continue
                probe = probe_by_kind.get(kind)
                if probe is not None and probe not in available:
                    _print_skip(
                        name,
                        f"{csv_path} has no {probe!r} column; this figure reads "
                        "a statistic scripts/invert.py no longer reports",
                    )
                    continue
                print(f"--- {name} ---")
                save_path = inverse_dir / f"{name}.png"
                if kind in takes_artifacts:
                    fn(csv_path, spec, artifact_dir=art_dir,
                       save_path=save_path, name=name)
                else:
                    fn(csv_path, spec, save_path=save_path, name=name)

    print(f"\nAll requested plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
