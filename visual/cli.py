"""CLI entry point for recurring development and correctness diagnostics.

Publication figures are generated separately with 'python -m visual.pub'.
Run 'python -m visual.cli --help' for the supported diagnostic inputs.
"""

from pathlib import Path

import numpy as np

from data.dataset import load_solver_dt, split_sim_ids
from visual import dataset_plots, forcing_plots, mms_plots, physics_plots, training_plots
from visual._common import (
    PLOT_REGISTRY,
    PLOT_RERUN_TRIGGERS,
    _plots_for_group,
    _print_skip,
    _should_run,
)


def main():
    import argparse

    trigger_help = "\n".join(
        f"  {name}: {PLOT_RERUN_TRIGGERS[name]}"
        for name in PLOT_REGISTRY
    )
    parser = argparse.ArgumentParser(
        description="Generate recurring diagnostics for the ITR project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Rerun triggers:\n" + trigger_help,
    )
    parser.add_argument("--data", type=str, default=None, help="Path to trajectories .npy file")
    parser.add_argument("--x-grid", type=str, default=None, help="Path to x_grid .npy file")
    parser.add_argument("--y-grid", type=str, default=None, help="Path to y_grid .npy file")
    parser.add_argument("--t-grid", type=str, default=None, help="Path to t_grid .npy file")
    parser.add_argument("--params", type=str, default=None, help="Path to sim_params .npy file")
    parser.add_argument("--csv", type=str, default=None, help="Path to train_metrics.csv")
    parser.add_argument("--report", type=str, default=None, help="Path to seed_report.json")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint .pt")
    parser.add_argument("--out", type=str, default=None, help="Output directory for plots")
    parser.add_argument(
        "--group",
        type=str,
        nargs="+",
        default=["all"],
        choices=["all", "physics", "mms", "training", "data", "forcing", "source", "interfaces"],
        help="Which diagnostic group(s) to generate (default: all)",
    )
    parser.add_argument(
        "--plots",
        type=str,
        nargs="+",
        default=None,
        choices=sorted(PLOT_REGISTRY),
        help="Individual plot names to generate (overrides --group)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out).resolve() if args.out else Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = args.group
    individual = args.plots

    def requested(name: str) -> bool:
        return _should_run(name, groups, individual)

    if requested("bc_verification"):
        print("=== bc_verification ===")
        solver = physics_plots.create_demo_multilayer_solver()
        T0 = np.full((solver.Nx, solver.Ny), solver.T_right(0.0), dtype=float)
        _, _, _, history = solver.solve(T0=T0, store_trajectory=True)
        physics_plots.plot_bc_verification(
            solver,
            history,
            save_path=out_dir / "physics" / "bc_verification.png",
        )
    if requested("itr_temperature_jump_sweep"):
        print("=== itr_temperature_jump_sweep ===")
        physics_plots.plot_itr_temperature_jump_sweep(
            save_path=out_dir / "physics" / "itr_temperature_jump_sweep.png"
        )

    if any(requested(name) for name in _plots_for_group("mms")):
        print("=== MMS GROUP ===")
        mms_dir = out_dir / "mms"
        if requested("mms_convergence"):
            mms_plots.plot_mms_convergence(save_path=mms_dir / "mms_convergence.png")
        if requested("mms_order_estimation"):
            mms_plots.plot_mms_order_estimation(
                save_path=mms_dir / "mms_order_estimation.png"
            )

    if any(requested(name) for name in _plots_for_group("training")):
        print("=== TRAINING GROUP ===")
        training_dir = out_dir / "training"
        if requested("training_curves"):
            if args.csv:
                training_plots.plot_training_curves(
                    args.csv, save_path=training_dir / "training_curves.png"
                )
            else:
                _print_skip("training_curves", "need --csv")
        if requested("seed_comparison"):
            if args.report:
                training_plots.plot_seed_comparison(
                    args.report, save_path=training_dir / "seed_comparison.png"
                )
            else:
                _print_skip("seed_comparison", "need --report")

    if any(requested(name) for name in _plots_for_group("forcing")):
        print("=== FORCING GROUP ===")
        forcing_dir = out_dir / "forcing"
        forcing_dispatch = {
            "forcing_temporal_families": forcing_plots.plot_forcing_temporal_families,
            "forcing_spatial_profiles": forcing_plots.plot_forcing_spatial_profiles,
            "forcing_separable_assembly": forcing_plots.plot_forcing_separable_assembly,
            "forcing_seq_tokens": forcing_plots.plot_forcing_seq_tokens,
        }
        for name, plot_fn in forcing_dispatch.items():
            if requested(name):
                print(f"--- {name} ---")
                plot_fn(save_path=forcing_dir / f"{name}.png")

    full_array_names = {
        "prediction_vs_truth",
        "interface_error",
        "lead_time_coverage",
        "initial_conditions",
        "energy_budget",
        "patch_region_error_map",
        "source_interface_zone_error",
        "interface_x_breakdown",
    }
    requested_full_names = {name for name in full_array_names if requested(name)}
    trajectories = x_grid = y_grid = t_grid = sim_params = solver_dt = None
    if requested_full_names:
        missing = [
            flag
            for flag, value in (
                ("--data", args.data),
                ("--x-grid", args.x_grid),
                ("--y-grid", args.y_grid),
                ("--t-grid", args.t_grid),
                ("--params", args.params),
            )
            if value is None
        ]
        if missing:
            reason = "need " + ", ".join(missing)
            for name in sorted(requested_full_names):
                _print_skip(name, reason)
        else:
            trajectories, x_grid, y_grid, t_grid = dataset_plots._load_plot_data(
                args.data, args.x_grid, args.y_grid, args.t_grid
            )
            sim_params = np.load(args.params, allow_pickle=True)
            solver_dt = load_solver_dt(args.t_grid)

    if requested("source_itr_sin_resistance_profiles") and y_grid is None:
        missing = [
            flag
            for flag, value in (("--y-grid", args.y_grid), ("--params", args.params))
            if value is None
        ]
        if missing:
            _print_skip("source_itr_sin_resistance_profiles", "need " + ", ".join(missing))
        else:
            y_grid = np.load(args.y_grid)
            x_grid = np.load(args.x_grid) if args.x_grid else None
            sim_params = np.load(args.params, allow_pickle=True)

    model_names = {
        "prediction_vs_truth",
        "interface_error",
        "patch_region_error_map",
        "source_interface_zone_error",
    }
    requested_model_names = {name for name in model_names if requested(name)}
    model = checkpoint_conf = None
    if requested_model_names and trajectories is not None:
        if args.checkpoint:
            try:
                model, checkpoint_conf = dataset_plots._load_checkpoint_model(
                    args.checkpoint
                )
            except ValueError as exc:
                for name in sorted(requested_model_names):
                    _print_skip(name, str(exc))
        else:
            for name in sorted(requested_model_names):
                _print_skip(name, "need --checkpoint")

    plot_config = dataset_plots._resolve_plot_config(checkpoint_conf)
    split_datasets = None
    split_names = {"patch_region_error_map", "source_interface_zone_error"}
    if (
        trajectories is not None
        and sim_params is not None
        and model is not None
        and any(requested(name) for name in split_names)
    ):
        split_datasets = dataset_plots._build_split_datasets(
            trajectories,
            x_grid,
            y_grid,
            t_grid,
            sim_params,
            plot_config,
            dt=solver_dt,
        )

    data_dir = out_dir / "data"
    if trajectories is not None:
        if requested("initial_conditions"):
            dataset_plots.plot_initial_conditions(
                trajectories,
                x_grid,
                y_grid,
                save_path=data_dir / "initial_conditions.png",
            )
        if requested("lead_time_coverage"):
            dataset_plots.plot_lead_time_coverage(
                trajectories,
                x_grid,
                y_grid,
                t_grid,
                sim_params,
                plot_config,
                save_path=data_dir / "lead_time_coverage.png",
            )
        if requested("prediction_vs_truth") and model is not None:
            dataset_plots.plot_prediction_vs_truth(
                model,
                trajectories,
                x_grid,
                y_grid,
                t_grid,
                sim_params,
                sim_id=0,
                s=0,
                config=plot_config,
                dt=solver_dt,
                save_path=data_dir / "prediction_vs_truth.png",
            )
        if requested("interface_error") and model is not None:
            _train_ids, _val_ids, test_ids = split_sim_ids(
                len(trajectories),
                plot_config.get("data", {}).get("train_split", 0.7),
                plot_config.get("data", {}).get("val_split", 0.15),
                seed=0,
            )
            dataset_plots.plot_interface_error(
                model,
                trajectories,
                x_grid,
                y_grid,
                t_grid,
                sim_params,
                test_ids,
                config=plot_config,
                dt=solver_dt,
                save_path=data_dir / "interface_error.png",
            )

    source_dir = out_dir / "source"
    if sim_params is not None and y_grid is not None and requested("source_itr_sin_resistance_profiles"):
        dataset_plots.plot_source_itr_sin_resistance_profiles(
            sim_params,
            y_grid,
            x_grid=x_grid,
            save_path=source_dir / "source_itr_sin_resistance_profiles.png",
        )
    if trajectories is not None:
        if requested("energy_budget"):
            dataset_plots.plot_energy_budget(
                trajectories,
                x_grid,
                y_grid,
                t_grid,
                sim_params,
                save_path=source_dir / "energy_budget.png",
            )
        if requested("patch_region_error_map") and model is not None:
            dataset_plots.plot_patch_region_error_map(
                model,
                split_datasets["test"],
                x_grid,
                y_grid,
                sim_params,
                config=plot_config,
                save_path=source_dir / "patch_region_error_map.png",
            )
        if requested("source_interface_zone_error") and model is not None:
            dataset_plots.plot_source_interface_zone_error(
                model,
                split_datasets["test"],
                x_grid,
                y_grid,
                plot_config,
                save_path=source_dir / "source_interface_zone_error.png",
            )

    if trajectories is not None and requested("interface_x_breakdown"):
        dataset_plots.plot_interface_x_breakdown(
            trajectories,
            sim_params,
            x_grid,
            y_grid,
            t_grid,
            save_path=out_dir / "interfaces" / "interface_x_breakdown.png",
        )


if __name__ == "__main__":
    main()
