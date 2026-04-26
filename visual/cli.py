"""CLI entry point for plot generation.

Run as: ``python -m visual.cli ...`` (see --help for full flag list).

Replaces the old ``python -m visual.plots`` entry. Dispatches to per-group
plot modules based on --group / --plots flags.
"""

from pathlib import Path

import numpy as np

from visual._common import _plots_for_group, _print_skip, _should_run
from visual import dataset_plots, mms_plots, physics_plots, sweep_plots, training_plots


def main():
    """
    Usage:
      python -m visual.cli --out visual/                          # all plots (default)
      python -m visual.cli --group physics --out visual/          # physics diagnostics only
      python -m visual.cli --group mms --out visual/              # MMS convergence only
      python -m visual.cli --group training --csv <path> --out visual/
      python -m visual.cli --group data --data <path> --x-grid <path> --y-grid <path> --t-grid <path> --params <path> --out visual/
      python -m visual.cli --group sweep --experiment <path> --runs <path> --out visual/
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
      --experiment  Path to conf/generated/experiment{N}/ directory (sweep plots)
      --runs        Path to runs/experiment{N}/ directory (sweep convergence)
      --sweep-seed  Which seed to show in convergence plot (default: 0)
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
    parser.add_argument("--experiment", type=str, default=None,
                        help="Path to conf/generated/experiment{N}/ directory (sweep plots)")
    parser.add_argument("--runs", type=str, default=None,
                        help="Path to runs/experiment{N}/ directory (sweep convergence)")
    parser.add_argument("--sweep-seed", type=int, default=0,
                        help="Which seed to show in convergence plot (default: 0)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (for interface_error plot)")
    parser.add_argument("--out", type=str, default=None, help="Output directory for plots")
    parser.add_argument("--group", type=str, nargs="+", default=["all"],
                        choices=["all", "physics", "mms", "training", "data", "sweep"],
                        help="Which plot group(s) to generate (default: all)")
    parser.add_argument("--plots", type=str, nargs="+", default=None,
                        help="Individual plot names to generate (overrides --group)")
    args = parser.parse_args()

    out_dir = Path(args.out).resolve() if args.out else Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = args.group
    individual = args.plots

    # ---- PHYSICS GROUP ----
    physics_plot_names = _plots_for_group("physics")
    need_physics = any(_should_run(p, groups, individual) for p in physics_plot_names)

    if need_physics:
        print("=== PHYSICS GROUP ===")
        solver = physics_plots.create_demo_multilayer_solver()
        t_sol, gx_sol, gy_sol, T_hist = solver.solve(store_trajectory=True)

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

        # flux_profiles needs no data files
        if _should_run("flux_profiles", groups, individual):
            print("--- flux_profiles ---")
            dataset_plots.plot_flux_profiles(save_path=data_dir / "flux_profiles.png")

        sim_params = None
        if args.params:
            sim_params = np.load(args.params, allow_pickle=True)

        grid_data_plots = {
            "trajectory_heatmap",
            "y_perturbation",
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

        if needs_grid_data:
            if args.data and args.x_grid and args.y_grid and args.t_grid:
                print(f"Loading trajectories and grids from {args.data}, {args.x_grid}, {args.y_grid}, {args.t_grid} ...")
                trajectories, x_grid, y_grid, t_grid = dataset_plots._load_plot_data(args.data, args.x_grid, args.y_grid, args.t_grid)
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
            split_datasets = dataset_plots._build_split_datasets(trajectories, x_grid, y_grid, t_grid, sim_params, plot_config)

        if sim_params is not None and _should_run("lhs_scatter", groups, individual):
            print("--- lhs_scatter ---")
            dataset_plots.plot_lhs_scatter(sim_params, save_path=data_dir / "lhs_scatter.png")

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

            if _should_run("initial_conditions", groups, individual):
                print("--- initial_conditions ---")
                dataset_plots.plot_initial_conditions(trajectories, x_grid=x_grid, y_grid=y_grid,
                                                     save_path=data_dir / "initial_conditions.png")

            # plots requiring sim_params
            if sim_params is not None:
                if _should_run("y_perturbation", groups, individual):
                    print("--- y_perturbation ---")
                    dataset_plots.plot_y_perturbation(
                        trajectories,
                        sim_id=highlight_sid,
                        x_grid=x_grid, y_grid=y_grid, t_grid=t_grid,
                        sim_params=sim_params,
                        save_path=data_dir / "y_perturbation.png",
                    )

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
                    "lhs_scatter",
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

    # ---- SWEEP GROUP ----
    sweep_plot_names = _plots_for_group("sweep")
    need_sweep = any(_should_run(p, groups, individual) for p in sweep_plot_names)

    if need_sweep:
        print("=== SWEEP GROUP ===")

        if args.experiment:
            experiment_dir = Path(args.experiment)
            sweep_dir = out_dir / "sweep"

            if _should_run("sweep_ranking", groups, individual):
                print("--- sweep_ranking ---")
                sweep_plots.plot_sweep_ranking(experiment_dir, save_path=sweep_dir / "sweep_ranking.png")

            if _should_run("sweep_convergence", groups, individual):
                if args.runs:
                    print("--- sweep_convergence ---")
                    sweep_plots.plot_sweep_convergence(
                        experiment_dir,
                        args.runs,
                        seed=args.sweep_seed,
                        save_path=sweep_dir / "sweep_convergence.png",
                    )
                else:
                    _print_skip("sweep_convergence", "need --runs")

            if _should_run("sweep_hyperparams", groups, individual):
                print("--- sweep_hyperparams ---")
                sweep_plots.plot_sweep_hyperparams(experiment_dir, save_path=sweep_dir / "sweep_hyperparams.png")
        else:
            for name in sweep_plot_names:
                if _should_run(name, groups, individual):
                    _print_skip(name, "need --experiment")

    print(f"\nAll requested plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
