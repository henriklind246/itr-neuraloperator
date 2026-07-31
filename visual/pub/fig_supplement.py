"""Supplemental figures: F23, F24, F25.

F25 is self-generating -- it runs the method-of-manufactured-solutions harness
in ``src/physics/mms_2d.py`` at render time and needs no training artifact, which
is why it is the one quantitative figure that is unblocked today. F23 and F24
are registered so ``--list`` documents exactly what blocks them; they raise
until the runs they need exist.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from src.physics.mms_2d import run_mms_2d_interface
from visual.pub import panels, style
from visual.pub._blocked import blocked

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
    """F23 -- direct against autoregressive rollout.

    Design, once the artifacts exist: error against rollout step for both
    prediction modes on the same simulations, so the crossover point where
    accumulated autoregressive error overtakes direct prediction is visible.
    Per-sim aggregation, with the rollout step as the stratum.

    Conditional, and the first figure to cut if the page budget is tight.
    """
    blocked(requirement,
            "needs rollout-partition test records; evaluation.rollout.enabled "
            "is false in every surviving config.",
            key="F23_direct_vs_autoregressive")


def resolution_invariance(*, source=None, spec=None, requirement=None):
    """F24 -- resolution invariance.

    Design, once the artifacts exist: error against evaluation resolution for a
    model trained at one resolution, with the **FV discretization drift** drawn
    as a reference band. Without that band the figure cannot distinguish
    operator-learning transfer failure from the solver's own truncation error
    changing under the reader's feet.

    Conditional, and the second figure to cut.
    """
    blocked(requirement,
            "needs the seed_report_r*.json set across resolutions plus the FV "
            "drift baseline from scripts/fv_convergence_baseline.py.",
            key="F24_resolution_invariance")


__all__ = ["direct_vs_autoregressive", "mms_convergence", "resolution_invariance"]
