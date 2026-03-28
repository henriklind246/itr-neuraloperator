"""
Temporal resolution analysis for the Crank-Nicolson FV solver.

Answers: "What dt does the CN solver need for high-fidelity applications?"

Uses MMS verification with T*(x,t) = 300 + A*sin(omega*t)*(1-x)^4 to isolate
temporal error by fixing a very fine spatial grid (N=801, dx ~ 0.00125).
"""
import numpy as np
import matplotlib.pyplot as plt
from src.physics.mms_1d import run_mms_once

# ---- CONFIG ----
N_FINE = 801  # fine spatial grid so spatial error is negligible
DT_LIST = [2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 8e-4, 1e-3, 2e-3, 5e-3, 0.01, 0.02]
DT_CURRENT = 0.005  # current dataset generation value


def run_temporal_sweep(dt_list: list[float], N: int = N_FINE) -> dict:
    results = {"dt": [], "l2_error": [], "max_error": [], "Fo": [], "rel_l2_pct": []}

    # compute dx for Fourier number
    dx = 1.0 / (N - 1)
    alpha = 1.0  # k/(rho*cp) for default material

    # reference L2 norm of exact solution for relative error
    # T* = 300 + 12.5*sin(4pi*t_final)*(1-x)^4 at t=1.0
    x_grid = np.linspace(0, 1, N)
    omega = 2.0 * np.pi * 2.0
    A = 50.0 / (4.0 * 1.0 * 1.0**3)
    T_exact_final = 300.0 + A * np.sin(omega * 1.0) * (1.0 - x_grid) ** 4
    l2_norm_exact = np.sqrt(np.mean(T_exact_final**2))

    for dt in dt_list:
        h, dt_actual, max_err, l2_err = run_mms_once(N=N, dt=dt)
        Fo = alpha * dt / dx**2
        rel_l2 = (l2_err / l2_norm_exact) * 100.0

        results["dt"].append(dt)
        results["l2_error"].append(l2_err)
        results["max_error"].append(max_err)
        results["Fo"].append(Fo)
        results["rel_l2_pct"].append(rel_l2)

        print(f"  dt={dt:.1e}  Fo={Fo:8.2f}  L2={l2_err:.4e}  max={max_err:.4e}  rel={rel_l2:.6f}%")

    return results


def compute_pairwise_order(dt_list: list[float], errors: list[float]) -> list[float]:
    orders = [None]  # first point has no predecessor
    for i in range(1, len(dt_list)):
        if errors[i] > 0 and errors[i - 1] > 0:
            p = np.log(errors[i - 1] / errors[i]) / np.log(dt_list[i - 1] / dt_list[i])
            orders.append(p)
        else:
            orders.append(None)
    return orders


def find_dt_for_threshold(dt_vals, rel_errors, threshold_pct: float) -> float | None:
    """Find largest dt where error is still below threshold (ascending dt/error)."""
    dt_arr = np.array(dt_vals)
    err_arr = np.array(rel_errors)
    # check if all errors are below threshold
    if all(e < threshold_pct for e in rel_errors):
        return dt_arr[-1]  # even coarsest dt is fine
    # check if all errors are above threshold
    if all(e >= threshold_pct for e in rel_errors):
        return None
    # find crossing from below threshold to above (ascending dt order)
    for i in range(len(err_arr) - 1):
        if err_arr[i] < threshold_pct and err_arr[i + 1] >= threshold_pct:
            # log-log interpolation to find crossover dt
            log_dt = np.interp(
                np.log10(threshold_pct),
                [np.log10(err_arr[i]), np.log10(err_arr[i + 1])],
                [np.log10(dt_arr[i]), np.log10(dt_arr[i + 1])],
            )
            return 10**log_dt
    return None


def print_summary_table(results: dict, orders: list[float]) -> None:
    print("\n" + "=" * 85)
    print("                     TEMPORAL CONVERGENCE SUMMARY")
    print("=" * 85)
    print(f"{'dt':>10s} | {'Fo':>8s} | {'L2 Error':>12s} | {'Max Error':>12s} | {'Rel L2 (%)':>11s} | {'Order':>5s}")
    print("-" * 85)

    for i in range(len(results["dt"])):
        dt = results["dt"][i]
        Fo = results["Fo"][i]
        l2 = results["l2_error"][i]
        mx = results["max_error"][i]
        rel = results["rel_l2_pct"][i]
        order = orders[i]

        marker = " <-- current dataset" if abs(dt - DT_CURRENT) < 1e-10 else ""
        order_str = f"{order:.2f}" if order is not None else "--"

        print(f"{dt:10.1e} | {Fo:8.2f} | {l2:12.4e} | {mx:12.4e} | {rel:11.6f} | {order_str:>5s}{marker}")

    print("=" * 85)


def print_recommendations(results: dict) -> None:
    thresholds = [
        (0.001, "High-fidelity (MMS verification baseline)"),
        (0.01, "Production quality"),
        (0.1, "Good training data"),
        (1.0, "Coarse but usable"),
    ]

    print("\n" + "=" * 70)
    print("                  RECOMMENDED dt FOR ACCURACY TARGETS")
    print("=" * 70)

    # sort by ascending dt for interpolation
    idx_sorted = np.argsort(results["dt"])
    dt_sorted = [results["dt"][i] for i in idx_sorted]
    rel_sorted = [results["rel_l2_pct"][i] for i in idx_sorted]

    for threshold, label in thresholds:
        dt_rec = find_dt_for_threshold(dt_sorted, rel_sorted, threshold)
        if dt_rec is not None:
            dx = 1.0 / (N_FINE - 1)
            Fo_rec = 1.0 * dt_rec / dx**2
            Nt_rec = int(np.ceil(1.0 / dt_rec))
            print(f"  < {threshold:>6.3f}%  | dt ~ {dt_rec:.1e}  | Fo ~ {Fo_rec:6.1f}  | ~{Nt_rec:>6,} steps  | {label}")
        else:
            print(f"  < {threshold:>6.3f}%  | NOT achievable in tested range              | {label}")

    # report current dataset error
    dt_idx = None
    for i, dt in enumerate(results["dt"]):
        if abs(dt - DT_CURRENT) < 1e-10:
            dt_idx = i
            break
    if dt_idx is not None:
        print(f"\n  Current dataset (dt={DT_CURRENT}): relative L2 error = {results['rel_l2_pct'][dt_idx]:.4f}%")

    print("=" * 70)


def plot_results(results: dict, orders: list[float], save_path: str = "visual/mms/dt_convergence_study.png") -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    dt_arr = np.array(results["dt"])
    l2_arr = np.array(results["l2_error"])
    Fo_arr = np.array(results["Fo"])
    rel_arr = np.array(results["rel_l2_pct"])

    # ---- Panel (a): log-log dt vs L2 error ----
    ax = axes[0, 0]
    ax.loglog(dt_arr, l2_arr, "o-", color="tab:blue", markersize=6, label="L2 error")

    # O(dt^2) reference line
    ref_idx = 3  # use a mid-range point as anchor
    ref_dt, ref_err = dt_arr[ref_idx], l2_arr[ref_idx]
    dt_ref_line = np.array([dt_arr[0], dt_arr[-1]])
    err_ref_line = ref_err * (dt_ref_line / ref_dt) ** 2
    ax.loglog(dt_ref_line, err_ref_line, "--", color="gray", alpha=0.6, label=r"$O(\Delta t^2)$ reference")

    # mark current dataset dt
    ax.axvline(DT_CURRENT, color="red", linestyle=":", alpha=0.7, label=f"Current dt={DT_CURRENT}")

    ax.set_xlabel(r"$\Delta t$")
    ax.set_ylabel("L2 Error")
    ax.set_title("(a) Temporal Convergence")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    # ---- Panel (b): pairwise order vs dt ----
    ax = axes[0, 1]
    valid_dt = [dt_arr[i] for i in range(len(orders)) if orders[i] is not None]
    valid_orders = [o for o in orders if o is not None]
    ax.plot(valid_dt, valid_orders, "s-", color="tab:green", markersize=6)
    ax.axhline(2.0, color="gray", linestyle="--", alpha=0.6, label="Expected order = 2")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\Delta t$")
    ax.set_ylabel("Estimated Order")
    ax.set_title("(b) Order of Accuracy")
    ax.set_ylim(0, 3)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    # ---- Panel (c): relative error vs Fourier number ----
    ax = axes[1, 0]
    ax.loglog(Fo_arr, rel_arr, "D-", color="tab:orange", markersize=6)
    ax.axvline(0.5, color="green", linestyle=":", alpha=0.7, label="Fo = 0.5 (textbook)")
    ax.axvline(1.0, color="blue", linestyle=":", alpha=0.5, label="Fo = 1.0")

    # threshold lines
    for thr, ls in [(0.001, ":"), (0.01, "--"), (0.1, "-.")]:
        ax.axhline(thr, color="gray", linestyle=ls, alpha=0.4, label=f"{thr}%")

    ax.set_xlabel("Fourier Number (Fo)")
    ax.set_ylabel("Relative L2 Error (%)")
    ax.set_title("(c) Error vs Fourier Number")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, which="both", alpha=0.3)

    # ---- Panel (d): summary text ----
    ax = axes[1, 1]
    ax.axis("off")

    # sort by dt ascending for the table
    idx_sorted = np.argsort(dt_arr)

    lines = ["TEMPORAL RESOLUTION RECOMMENDATIONS\n"]
    lines.append(f"{'dt':>10s}  {'Fo':>7s}  {'Rel Err%':>10s}  {'Steps':>7s}")
    lines.append("-" * 42)
    for i in idx_sorted:
        dt = dt_arr[i]
        Fo = Fo_arr[i]
        rel = rel_arr[i]
        Nt = int(np.ceil(1.0 / dt))
        marker = " *" if abs(dt - DT_CURRENT) < 1e-10 else ""
        lines.append(f"{dt:10.1e}  {Fo:7.1f}  {rel:10.5f}  {Nt:>7,}{marker}")
    lines.append("-" * 42)
    lines.append("* = current dataset dt")
    lines.append(f"\nScheme: Crank-Nicolson (2nd order)")
    lines.append(f"Spatial grid: N={N_FINE}, dx={1.0/(N_FINE-1):.5f}")

    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
            fontsize=8, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    ax.set_title("(d) Summary")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {save_path}")


def run_dataset_grid_comparison(dt_list: list[float]) -> dict:
    """Run MMS at the actual dataset grid (N=101) to show combined spatial+temporal error."""
    N_dataset = 101
    dx_dataset = 1.0 / (N_dataset - 1)

    print(f"\n{'='*70}")
    print(f"  DATASET GRID COMPARISON (N={N_dataset}, dx={dx_dataset:.4f})")
    print(f"{'='*70}")
    print(f"{'dt':>10s} | {'Fo':>8s} | {'L2 Error':>12s} | {'Max Error':>12s} | {'Note':>20s}")
    print("-" * 70)

    for dt in dt_list:
        h, dt_actual, max_err, l2_err = run_mms_once(N=N_dataset, dt=dt)
        Fo = 1.0 * dt / dx_dataset**2
        note = "CURRENT" if abs(dt - DT_CURRENT) < 1e-10 else ""
        print(f"{dt:10.1e} | {Fo:8.2f} | {l2_err:12.4e} | {max_err:12.4e} | {note:>20s}")

    # also run with very small dt to show spatial error floor
    _, _, max_floor, l2_floor = run_mms_once(N=N_dataset, dt=1e-5)
    print(f"{'1.0e-05':>10s} | {'0.10':>8s} | {l2_floor:12.4e} | {max_floor:12.4e} | {'SPATIAL FLOOR':>20s}")
    print(f"{'='*70}")
    print(f"  Spatial error floor at N={N_dataset}: L2 = {l2_floor:.4e}")
    print(f"  -> Any dt where temporal error << {l2_floor:.1e} is dominated by spatial error")
    print(f"{'='*70}")


if __name__ == "__main__":
    print("Running temporal convergence sweep...")
    print(f"Spatial grid: N={N_FINE}, dx={1.0/(N_FINE-1):.5f}")
    print(f"Material: rho=1, cp=1, k=1 (alpha=1)\n")

    results = run_temporal_sweep(DT_LIST, N=N_FINE)
    orders = compute_pairwise_order(results["dt"], results["l2_error"])

    print_summary_table(results, orders)
    print_recommendations(results)

    # show what happens on the actual dataset grid
    run_dataset_grid_comparison([5e-5, 5e-4, 1e-3, 5e-3, 0.01])

    plot_results(results, orders)
