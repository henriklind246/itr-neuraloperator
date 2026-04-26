"""MMS verification: spatial + temporal convergence and order estimation for the 2D solver."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from visual._common import PLOT_STYLE, _save_figure


def plot_mms_convergence(save_path: str | Path | None = None):
    """MMS convergence for all three 2D test cases in log-log space.

    2x3 layout:
        Top row: spatial convergence (varying N, fixed dt) for y-independent, full 2D, interface
        Bottom row: temporal convergence (varying dt, fixed N) for the same 3 cases
    """
    from src.physics.mms_2d import (
        run_mms_y_independent,
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

    cases = [
        ("y-independent", run_mms_y_independent),
        ("Full 2D", run_mms_2d),
        ("Interface", run_mms_2d_interface),
    ]

    # N lists: interface requires even N
    N_lists = {
        "y-independent": [21, 41, 81, 161],
        "Full 2D": [21, 41, 81, 161],
        "Interface": [50, 100, 200],
    }
    fixed_dt = 0.0001
    dt_list = [0.02, 0.01, 0.005]
    fixed_N_map = {
        "y-independent": 201,
        "Full 2D": 201,
        "Interface": 200,
    }

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))

        for col, (name, run_fn) in enumerate(cases):
            h_vals, h_l2 = [], []
            for N in N_lists[name]:
                h, _, _, l2 = run_fn(N, dt=fixed_dt)
                h_vals.append(h)
                h_l2.append(l2)
            _plot_convergence(axes[0, col], np.array(h_vals), np.array(h_l2),
                              r"$\Delta x$", f"{name} — Spatial (dt={fixed_dt})")

            dt_vals, dt_l2 = [], []
            fixed_N = fixed_N_map[name]
            for dt_val in dt_list:
                _, _, _, l2 = run_fn(fixed_N, dt=dt_val)
                dt_vals.append(dt_val)
                dt_l2.append(l2)
            _plot_convergence(axes[1, col], np.array(dt_vals), np.array(dt_l2),
                              r"$\Delta t$", f"{name} — Temporal (N={fixed_N})")

        fig.suptitle("2D MMS Convergence — Crank-Nicolson FV Solver")
        _save_figure(fig, save_path, "mms", "mms_convergence")


def plot_mms_order_estimation(save_path: str | Path | None = None):
    """Pairwise order estimates for all three 2D MMS cases.

    2x3 layout:
        Top row: estimated spatial order p_x vs dx for 3 cases
        Bottom row: estimated temporal order p_t vs dt for 3 cases
    """
    from src.physics.mms_2d import (
        run_mms_y_independent,
        run_mms_2d,
        run_mms_2d_interface,
    )

    cases = [
        ("y-independent", run_mms_y_independent),
        ("Full 2D", run_mms_2d),
        ("Interface", run_mms_2d_interface),
    ]

    N_lists = {
        "y-independent": [21, 41, 81, 161],
        "Full 2D": [21, 41, 81, 161],
        "Interface": [50, 100, 200],
    }
    fixed_dt = 0.0001
    dt_list = [0.02, 0.01, 0.005]
    fixed_N_map = {
        "y-independent": 201,
        "Full 2D": 201,
        "Interface": 200,
    }

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))

        for col, (name, run_fn) in enumerate(cases):
            h_vals, h_l2 = [], []
            for N in N_lists[name]:
                h, _, _, l2 = run_fn(N, dt=fixed_dt)
                h_vals.append(h)
                h_l2.append(l2)

            px_vals, px_mid = [], []
            for i in range(len(h_vals) - 1):
                p = np.log(h_l2[i] / h_l2[i + 1]) / np.log(h_vals[i] / h_vals[i + 1])
                px_vals.append(p)
                px_mid.append(np.sqrt(h_vals[i] * h_vals[i + 1]))

            ax = axes[0, col]
            ax.plot(px_mid, px_vals, "ko-", markersize=7)
            ax.axhline(2.0, color="C3", linestyle="--", alpha=0.6, label="Expected order = 2")
            ax.set_xscale("log")
            ax.set_xlabel(r"$\Delta x$ (geometric midpoint)")
            ax.set_ylabel("Estimated order $p_x$")
            ax.set_title(f"{name} — Spatial (dt={fixed_dt})")
            ax.legend(fontsize=7)
            ax.grid(True)
            ax.set_ylim(0, 4)

            fixed_N = fixed_N_map[name]
            dt_vals, dt_l2 = [], []
            for dt_val in dt_list:
                _, _, _, l2 = run_fn(fixed_N, dt=dt_val)
                dt_vals.append(dt_val)
                dt_l2.append(l2)

            pt_vals, pt_mid = [], []
            for i in range(len(dt_vals) - 1):
                p = np.log(dt_l2[i] / dt_l2[i + 1]) / np.log(dt_vals[i] / dt_vals[i + 1])
                pt_vals.append(p)
                pt_mid.append(np.sqrt(dt_vals[i] * dt_vals[i + 1]))

            ax = axes[1, col]
            ax.plot(pt_mid, pt_vals, "ko-", markersize=7)
            ax.axhline(2.0, color="C3", linestyle="--", alpha=0.6, label="Expected order = 2")
            ax.set_xscale("log")
            ax.set_xlabel(r"$\Delta t$ (geometric midpoint)")
            ax.set_ylabel("Estimated order $p_t$")
            ax.set_title(f"{name} — Temporal (N={fixed_N})")
            ax.legend(fontsize=7)
            ax.grid(True)
            ax.set_ylim(0, 4)

        fig.suptitle("2D MMS Order Estimation — Successive Refinement Pairs")
        _save_figure(fig, save_path, "mms", "mms_order_estimation")
