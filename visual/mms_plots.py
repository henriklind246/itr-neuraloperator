"""MMS verification: spatial + temporal convergence and order estimation for the 2D solver."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from visual._common import PLOT_STYLE, _save_figure


def plot_mms_convergence(save_path: str | Path | None = None):
    """MMS convergence for the 2D solver in log-log space, by direction.

    3x3 layout (rows = measurement axis, cols = case):
        Row 0  x-axis spatial:  y-independent (x-isolated), Full 2D, Interface
        Row 1  y-axis spatial:  x-linear (y-isolated), [off], [off]
        Row 2  temporal:        y-independent, Full 2D, Interface

    Only the single-layer directional cases separate x vs y cleanly: the
    y-independent solution is flat in y (isolates x) and the x-linear solution
    has zero x-curvature (isolates y). The combined cases (Full 2D, Interface)
    refine x and y together on the enforced isotropic grid, so they appear once
    (tagged "combined x+y") and the two y-row cells for them are turned off.
    """
    from src.physics.mms_2d import (
        run_mms_y_independent,
        run_mms_x_linear,
        run_mms_2d,
        run_mms_2d_interface,
    )

    # helper for log-log regression + plotting
    def _plot_convergence(ax, refinement, errors, xlabel, title):
        log_r = np.log10(refinement)
        log_e = np.log10(errors)
        coeffs = np.polyfit(log_r, log_e, 1)
        slope = coeffs[0]
        fit_line = 10 ** np.polyval(coeffs, log_r)

        ax.loglog(refinement, errors, "ko", markersize=7, label="MMS data")
        ax.loglog(refinement, fit_line, "C0-", linewidth=1.5, label=f"Fit: slope = {slope:.2f}")
        ref = errors[-1] * (refinement / refinement[-1]) ** 2
        ax.loglog(refinement, ref, "k--", alpha=0.4, linewidth=1, label="$O(h^2)$ reference")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("L2 error")
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(True, which="both", linestyle="--", alpha=0.3)

    def _spatial(ax, run_fn, N_list, title):
        h_vals, h_l2 = [], []
        for N in N_list:
            h, _, _, l2 = run_fn(N, dt=fixed_dt)
            h_vals.append(h)
            h_l2.append(l2)
        _plot_convergence(ax, np.array(h_vals), np.array(h_l2), r"$\Delta x = \Delta y$", title)

    def _temporal(ax, run_fn, fixed_N, title):
        dt_vals, dt_l2 = [], []
        for dt_val in dt_list:
            _, _, _, l2 = run_fn(fixed_N, dt=dt_val)
            dt_vals.append(dt_val)
            dt_l2.append(l2)
        _plot_convergence(ax, np.array(dt_vals), np.array(dt_l2), r"$\Delta t$", title)

    fixed_dt = 0.0001
    dt_list = [0.02, 0.01, 0.005]
    N_full = [21, 41, 81, 161]
    N_interface = [50, 100, 200]

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(3, 3, figsize=(16, 13))

        # Row 0 — x-axis spatial
        _spatial(axes[0, 0], run_mms_y_independent, N_full, "x-isolated: y-independent")
        _spatial(axes[0, 1], run_mms_2d, N_full, "combined x+y: Full 2D")
        _spatial(axes[0, 2], run_mms_2d_interface, N_interface, "combined x+y: Interface")

        # Row 1 — y-axis spatial (only the y-isolated case is meaningful)
        _spatial(axes[1, 0], run_mms_x_linear, N_full, "y-isolated: x-linear")
        axes[1, 1].axis("off")
        axes[1, 2].axis("off")

        # Row 2 — temporal
        _temporal(axes[2, 0], run_mms_y_independent, 201, "temporal: y-independent")
        _temporal(axes[2, 1], run_mms_2d, 201, "temporal: Full 2D")
        _temporal(axes[2, 2], run_mms_2d_interface, 200, "temporal: Interface")

        fig.suptitle("2D MMS Convergence — Crank-Nicolson FV Solver "
                     "(rows: x-axis spatial / y-axis spatial / temporal)")
        _save_figure(fig, save_path, "mms", "mms_convergence")


def plot_mms_order_estimation(save_path: str | Path | None = None):
    """Pairwise observed-order estimates for the 2D solver, by direction.

    3x3 layout (rows = measurement axis, cols = case):
        Row 0  x-axis spatial p_x:  y-independent (x-isolated), Full 2D, Interface
        Row 1  y-axis spatial p_y:  x-linear (y-isolated), [off], [off]
        Row 2  temporal p_t:        y-independent, Full 2D, Interface

    See plot_mms_convergence for why only the single-layer directional cases
    yield a true per-axis order; combined cases refine x and y together.
    """
    from src.physics.mms_2d import (
        run_mms_y_independent,
        run_mms_x_linear,
        run_mms_2d,
        run_mms_2d_interface,
    )

    def _pairwise_order(steps, errs):
        vals, mids = [], []
        for i in range(len(steps) - 1):
            p = np.log(errs[i] / errs[i + 1]) / np.log(steps[i] / steps[i + 1])
            vals.append(p)
            mids.append(np.sqrt(steps[i] * steps[i + 1]))
        return mids, vals

    def _order_panel(ax, mids, vals, xlabel, ylabel, title):
        ax.plot(mids, vals, "ko-", markersize=7)
        ax.axhline(2.0, color="C3", linestyle="--", alpha=0.6, label="Expected order = 2")
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(True)
        ax.set_ylim(0, 4)

    def _spatial_order(ax, run_fn, N_list, ylabel, title):
        h_vals, h_l2 = [], []
        for N in N_list:
            h, _, _, l2 = run_fn(N, dt=fixed_dt)
            h_vals.append(h)
            h_l2.append(l2)
        mids, vals = _pairwise_order(h_vals, h_l2)
        _order_panel(ax, mids, vals, r"$\Delta x = \Delta y$ (geometric midpoint)", ylabel, title)

    def _temporal_order(ax, run_fn, fixed_N, title):
        dt_vals, dt_l2 = [], []
        for dt_val in dt_list:
            _, _, _, l2 = run_fn(fixed_N, dt=dt_val)
            dt_vals.append(dt_val)
            dt_l2.append(l2)
        mids, vals = _pairwise_order(dt_vals, dt_l2)
        _order_panel(ax, mids, vals, r"$\Delta t$ (geometric midpoint)",
                     "Estimated order $p_t$", title)

    fixed_dt = 0.0001
    dt_list = [0.02, 0.01, 0.005]
    N_full = [21, 41, 81, 161]
    N_interface = [50, 100, 200]

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(3, 3, figsize=(16, 13))

        # Row 0 — x-axis spatial order p_x
        _spatial_order(axes[0, 0], run_mms_y_independent, N_full,
                       "Estimated order $p_x$", "x-isolated: y-independent")
        _spatial_order(axes[0, 1], run_mms_2d, N_full,
                       "Estimated order $p_x$", "combined x+y: Full 2D")
        _spatial_order(axes[0, 2], run_mms_2d_interface, N_interface,
                       "Estimated order $p_x$", "combined x+y: Interface")

        # Row 1 — y-axis spatial order p_y (only the y-isolated case is meaningful)
        _spatial_order(axes[1, 0], run_mms_x_linear, N_full,
                       "Estimated order $p_y$", "y-isolated: x-linear")
        axes[1, 1].axis("off")
        axes[1, 2].axis("off")

        # Row 2 — temporal order p_t
        _temporal_order(axes[2, 0], run_mms_y_independent, 201, "temporal: y-independent")
        _temporal_order(axes[2, 1], run_mms_2d, 201, "temporal: Full 2D")
        _temporal_order(axes[2, 2], run_mms_2d_interface, 200, "temporal: Interface")

        fig.suptitle("2D MMS Order Estimation — Successive Refinement Pairs "
                     "(rows: x-axis / y-axis / temporal)")
        _save_figure(fig, save_path, "mms", "mms_order_estimation")
