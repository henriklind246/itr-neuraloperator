"""Supplemental rollout, resolution, and solver-verification figures."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from src.physics.mms_2d import run_mms_2d_interface
from visual.pub import panels, records, stats, style

# The refinement ladders already used by ``visual/mms_plots.py``. The spatial
# study fixes dt small enough that the time error is negligible against the
# space error; the temporal study fixes N for the mirror reason.
SPATIAL_N = (50, 100, 200)
SPATIAL_DT = 1e-4
# build_time_grid rejects a dt that does not divide t_final, which is 0.37 for
# the interface MMS; these are t_final / {20, 40, 80}.
TEMPORAL_DT = (0.0185, 0.00925, 0.004625)
TEMPORAL_N = 200


def _spatial_ladder(n_list, dt):
    steps, errors = [], []
    for n in n_list:
        h, _, _, l2 = run_mms_2d_interface(int(n), dt=dt)
        steps.append(float(h))
        errors.append(float(l2))
    return np.asarray(steps), np.asarray(errors)


def _temporal_ladder(dt_list, n):
    steps, errors = [], []
    for dt in dt_list:
        _, dt_used, _, l2 = run_mms_2d_interface(int(n), dt=float(dt))
        steps.append(float(dt_used))
        errors.append(float(l2))
    return np.asarray(steps), np.asarray(errors)


def mms_convergence(*, source=None, spec=None, requirement=None,
                    n_list=SPATIAL_N, spatial_dt=SPATIAL_DT,
                    dt_list=TEMPORAL_DT, temporal_n=TEMPORAL_N):
    """F25 -- second-order space and time convergence of the interface solver.

    The manufactured solution is the piecewise two-layer field with a thermal
    contact resistance, so this exercises the same interface discretization the
    training data is generated with, not a simplified proxy.
    """
    h, err_h = _spatial_ladder(n_list, spatial_dt)
    dt, err_dt = _temporal_ladder(dt_list, temporal_n)

    order = np.argsort(h)
    h, err_h = h[order], err_h[order]
    order = np.argsort(dt)
    dt, err_dt = dt[order], err_dt[order]

    width = spec.width if spec is not None else "one_half_col"
    fig, axes = plt.subplots(
        2, 2, figsize=style.figsize(width, rows=2, row_height="std"))

    fit_h = panels.convergence_loglog(
        axes[0, 0], h, err_h, xlabel=r"grid spacing $h$", reference_order=2,
        title=f"space  ($\\Delta t$ = {spatial_dt:g})",
        color=style.BENCHMARK_COLORS["forcing"])
    mids_h, orders_h = panels.pairwise_orders(h, err_h)
    panels.order_estimates(
        axes[0, 1], mids_h, orders_h, xlabel=r"grid spacing $h$", target=2.0,
        color=style.BENCHMARK_COLORS["forcing"])

    fit_dt = panels.convergence_loglog(
        axes[1, 0], dt, err_dt, xlabel=r"time step $\Delta t$",
        reference_order=2, title=f"time  ($N$ = {int(temporal_n)})",
        reference_symbol=r"\Delta t", color=style.BENCHMARK_COLORS["interfaces"])
    mids_dt, orders_dt = panels.pairwise_orders(dt, err_dt)
    panels.order_estimates(
        axes[1, 1], mids_dt, orders_dt, xlabel=r"time step $\Delta t$",
        target=2.0, color=style.BENCHMARK_COLORS["interfaces"])

    style.panel_letters(axes.ravel())
    fig.tight_layout()

    metric_definition = {
        "quantity": "discrete $L_2$ error against the manufactured solution",
        "solver": "FVSolver2D, Crank-Nicolson, two-layer interface with $R_c$",
        "mms_runner": "src.physics.mms_2d.run_mms_2d_interface",
        "spatial_N": [int(n) for n in n_list],
        "spatial_dt": float(spatial_dt),
        "temporal_dt": [float(v) for v in dt_list],
        "temporal_N": int(temporal_n),
        "fitted_space_order": round(fit_h.slope, 4),
        "fitted_time_order": round(fit_dt.slope, 4),
        "design_order": 2,
    }
    return fig, None, metric_definition


def direct_vs_autoregressive(*, source=None, spec=None, requirement=None):
    """F23 -- paired rollout change against direct prediction by exact lead."""
    arms_by_benchmark = records.load_rollout_arms(source)
    order = ("forcing", "source", "interfaces")
    missing = [benchmark for benchmark in order if benchmark not in arms_by_benchmark]
    if missing:
        raise records.SchemaError(f"F23 is missing rollout studies for {missing}")
    curves = {
        benchmark: stats.rollout_delta_curves(arms_by_benchmark[benchmark])
        for benchmark in order
    }

    width = spec.width if spec is not None else "two_col"
    fig, axes = plt.subplots(
        1, 3, figsize=style.figsize(width, row_height="std"), squeeze=False,
    )
    labels = {"forcing": "forcing", "source": "source", "interfaces": "interfaces"}
    for index, benchmark in enumerate(order):
        reference = curves[benchmark][2]
        panels.rollout_delta_panel(
            axes[0, index], curves[benchmark],
            title=f"{labels[benchmark]}  (n={reference.n_sims} simulations)",
            ylabel=("Autoregressive minus direct\n"
                    "global rel. $L_2$ [percentage points]" if index == 0 else ""),
            legend=index == 0,
        )
    fig.suptitle(
        "Paired rollout error relative to direct prediction",
        fontsize=8, y=0.995,
    )
    style.panel_letters(axes.ravel())
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))

    metric_definition = {
        "quantity": "autoregressive minus direct global relative L2 error",
        "space": "checkpoint-normalized temperature",
        "unit": "percentage points",
        "replication_unit": "sim_id",
        "within_simulation_pooling": (
            "100 * sqrt(sum sse_K2 / sum target_sse_K2) within each exact lead"
        ),
        "pairing": "same seed, benchmark, sim_id, source index, and target index",
        "summary": "median with interquartile range across simulations",
        "model_seed": "32",
        "studies": {
            benchmark: {
                "n_simulations": curves[benchmark][2].n_sims,
                "snapshot_pairs_per_arm": curves[benchmark][2].n_snapshot_pairs,
                "exact_lead_times": curves[benchmark][2].lead_times.tolist(),
                "direct_median_rel_l2_pct": (
                    curves[benchmark][2].direct_median_pct.tolist()
                ),
            }
            for benchmark in order
        },
    }
    return fig, None, metric_definition


def resolution_invariance(*, source=None, spec=None, requirement=None):
    """F24 -- E32/E33 resolution transfer against metric-matched FV drift."""
    studies = records.load_resolution_studies(source)
    width = spec.width if spec is not None else "two_col"
    fig, axes = plt.subplots(
        1, 3, figsize=style.figsize(width, row_height="std"), squeeze=False,
    )
    panel_specs = (
        ("global_rel_l2_pct", "global field"),
        ("interface_rel_l2_pct", "interface band"),
        ("boundary_rel_l2_pct", "boundary band"),
    )
    for index, (value_attr, title) in enumerate(panel_specs):
        panels.resolution_curve_panel(
            axes[0, index], studies, studies.fv_drift,
            value_attr=value_attr, title=title,
            ylabel=("Relative $L_2$ [%]" if index == 0 else ""),
            legend=index == 0,
        )
    fig.suptitle(
        f"Forcing benchmark · fixed seed {studies.original.seed} · "
        f"{studies.original.n_simulations} fresh simulations per grid",
        fontsize=8, y=0.995,
    )
    style.panel_letters(axes.ravel())
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))

    metric_definition = {
        "quantity": "global, interface-band, and boundary-band relative L2 error",
        "space": "checkpoint-normalized temperature",
        "unit": "%",
        "comparison": [studies.original.label, studies.material_side.label],
        "model_seed": studies.original.seed,
        "n_simulations_per_resolution": studies.original.n_simulations,
        "snapshot_pairs_per_simulation": studies.original.pairs_per_simulation,
        "resolutions": list(studies.original.resolutions),
        "uncertainty": (
            "none; each curve is a point estimate for one fixed checkpoint"
        ),
        "fv_reference": (
            "metric-matched finite-volume discretization drift to "
            f"N={studies.fv_drift.reference_resolution}"
        ),
        "comparison_caveat": studies.material_side.comparison_caveat,
        "checkpoint_best_epochs_zero_based": {
            studies.original.label: studies.original.checkpoint_epoch,
            studies.material_side.label: studies.material_side.checkpoint_epoch,
        },
    }
    return fig, None, metric_definition


__all__ = ["direct_vs_autoregressive", "mms_convergence", "resolution_invariance"]
