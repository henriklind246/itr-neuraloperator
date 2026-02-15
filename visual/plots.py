from pathlib import Path
import importlib.util
import sys
from typing import Optional

import matplotlib.pyplot as plt

# TODO: expose a function to plot stored temperature history (heatmaps or time series)

# Utility to load the solver module by file path so this script works regardless of package layout.
def load_fd_solver_module() -> object:
    # locate fd_solver_1d relative to this file: ../src/physics/fd_solver_1d.py
    base = Path(__file__).resolve().parent.parent
    solver_path = base / "src" / "physics" / "fd_solver_1d.py"
    if not solver_path.exists():
        raise FileNotFoundError(f"Could not find fd_solver_1d.py at expected location: {solver_path}")

    spec = importlib.util.spec_from_file_location("fd_solver_1d_for_plots", str(solver_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def plot_final_temperature(
    solver: object,
    save_path: Optional[Path] = None,
    title: str = "Final temperature vs spatial nodes",
):
    """Run solver to final time and plot spatial grid (x) vs final temperature (T).

    - solver: an instance of FDSolver1D (must have .solve(store_history=False) -> (t, x, T_final)).
    - save_path: where to save the figure. If None, saves to ./final_temperature.png next to this file.
    """
    # run solver without storing full history to get final T
    t, x, T_final = solver.solve(store_history=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, T_final, marker="o", linestyle="-", color="C0", markersize=5)
    ax.set_xlabel("x (spatial nodes)")
    ax.set_ylabel("Temperature")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.5)

    if save_path is None:
        save_path = Path(__file__).resolve().parent / "final_temperature.png"

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    print(f"Saved final temperature plot to: {save_path}")
    plt.close(fig)


def demo_and_save():
    """Simple demo that loads the solver, runs a short sim, and saves the plot."""
    mod = load_fd_solver_module()
    # get the class
    FDSolver1D = getattr(mod, "FDSolver1D")

    # create a solver with reasonable defaults that mirror the example in the solver file
    solver = FDSolver1D(
        a=0.0,
        b=1.0,
        N=101,
        cp=1.0,
        rho=1.0,
        k=1.0,
        lam_target=0.5,
        t_final=0.5,
        flux_f=2.0,
        flux_A=50.0,
        t_on=0.0,
        t_off=0.5,
        phase=0.0,
    )

    out_path = Path(__file__).resolve().parent / "final_temperature.png"
    plot_final_temperature(solver, save_path=out_path)


if __name__ == "__main__":
    demo_and_save()
