"""Physics solver diagnostics: temperature fields, geometry, conductance, and flux profiles."""

import csv
import math
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from src.physics.fv_solver_2d import FVSolver2D, Layer2D

from visual._common import (
    PLOT_STYLE,
    _add_interface_lines,
    _evaluate_q_left_field,
    _resolve_interface_metadata,
    _save_figure,
)


def create_demo_multilayer_solver() -> FVSolver2D:
    """Create a 2-layer 2D demo solver for physics visualisation plots.

    Matches the layer configuration in generate_dataset.py:
      Layer 1: [0.0, 0.5], rho=1.0, cp=1.0, k=2.0
      Layer 2: [0.5, 1.0], rho=1.0, cp=1.0, k=1.0
    Nx=100 (even) ensures x=0.5 lies on a cell face (required by the solver).
    Ny=100 matches for isotropic grid on [0,1]x[0,1].
    """
    layers = [
        Layer2D(x_left=0.0, x_right=0.5, rho=1.0, cp=1.0, k=2.0),
        Layer2D(x_left=0.5, x_right=1.0, rho=1.0, cp=1.0, k=1.0),
    ]
    return FVSolver2D(
        a=0.0, b=1.0, c=0.0, d=1.0,
        Nx=100, Ny=100,
        lam_target=0.8, layers=layers, interface_R=[0.5],
        t_final=0.3, flux_f=2.0, flux_A=50.0,
        t_on=0.0, t_off=0.2, phase=0.0, dt=0.005,
    )


def _interface_flanking_nodes(solver: FVSolver2D) -> list[tuple[int, int, float]]:
    """Return (left_node, right_node, interface_x) for each internal interface."""
    return [
        (f, f + 1, solver.face_positions_x[f])
        for f in sorted(solver.interface_face_map.keys())
    ]


def plot_final_temperature(
    solver: FVSolver2D,
    save_path: Optional[Path] = None,
):
    """2D temperature field at final time as a pcolormesh heatmap."""
    T0 = np.full((solver.Nx, solver.Ny), solver.T_right(0.0), dtype=float)
    _, _, _, T_final = solver.solve(T0=T0, store_trajectory=False)
    interface_meta = _resolve_interface_metadata(solver=solver)

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(8, 6))
        pc = ax.pcolormesh(solver.X, solver.Y, T_final, cmap="inferno", shading="auto")
        fig.colorbar(pc, ax=ax, label="Temperature")
        _add_interface_lines(ax, interface_meta["positions"])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Final Temperature at t = {solver.t[-1]:.3f}")
        ax.set_aspect("equal")
        _save_figure(fig, save_path, "physics", "final_temperature")


def plot_layer_geometry(
    solver: FVSolver2D,
    save_path: str | Path | None = None,
):
    """Layer geometry and material properties for a 2D multilayer domain.

    3-panel layout:
        (a) Domain schematic with colored layers, subsampled grid nodes, interface lines
        (b) Material properties k, rho, cp vs x (1D slice at mid-y)
        (c) Thermal diffusivity alpha = k/(rho*cp) vs x
    """
    interface_meta = _resolve_interface_metadata(solver=solver)

    with plt.rc_context(PLOT_STYLE):
        fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(14, 4.5))

        layer_colors = plt.cm.Pastel1(np.linspace(0, 1, max(len(solver.layers), 3)))

        for j, layer in enumerate(solver.layers):
            ax_a.axvspan(
                layer.x_left,
                layer.x_right,
                alpha=0.4,
                color=layer_colors[j],
                label=f"Layer {j + 1} (k={layer.k})",
            )
        step = max(1, solver.Nx // 20)
        ax_a.plot(
            solver.X[::step, ::step].ravel(),
            solver.Y[::step, ::step].ravel(),
            ".",
            color="black",
            markersize=2,
            alpha=0.35,
        )
        _add_interface_lines(ax_a, interface_meta["positions"])
        ax_a.set_xlabel("x")
        ax_a.set_ylabel("y")
        ax_a.set_title("Domain Schematic")
        ax_a.legend(loc="upper right")
        ax_a.set_xlim(solver.a, solver.b)
        ax_a.set_ylim(solver.c, solver.d)
        ax_a.set_aspect("equal")

        ax_b.plot(solver.grid_x, solver.k_nodes[:, 0], drawstyle="steps-mid", label="k")
        ax_b.plot(solver.grid_x, solver.rho_nodes[:, 0], drawstyle="steps-mid", label=r"$\rho$")
        ax_b.plot(solver.grid_x, solver.cp_nodes[:, 0], drawstyle="steps-mid", label=r"$c_p$")
        _add_interface_lines(ax_b, interface_meta["positions"])
        ax_b.set_xlabel("x")
        ax_b.set_ylabel("Property value")
        ax_b.set_title("Material Properties")
        ax_b.legend()
        ax_b.grid(True)

        alpha_nodes = solver.k_nodes[:, 0] / (solver.rho_nodes[:, 0] * solver.cp_nodes[:, 0])
        ax_c.plot(solver.grid_x, alpha_nodes, drawstyle="steps-mid", color="C2")
        _add_interface_lines(ax_c, interface_meta["positions"])
        ax_c.set_xlabel("x")
        ax_c.set_ylabel(r"$\alpha = k / (\rho \, c_p)$")
        ax_c.set_title("Thermal Diffusivity")
        ax_c.grid(True)

        _save_figure(fig, save_path, "physics", "layer_geometry")


def plot_face_conductance(
    solver: FVSolver2D,
    save_path: str | Path | None = None,
):
    """Face conductance and CN coefficients for the 2D solver.

    2x2 layout:
        (a) G_x heatmap (x-direction face conductance)
        (b) G_y heatmap (y-direction face conductance)
        (c) CN coefficients r_w, r_e at mid-y slice (1D view)
        (d) Diagonal dominance margin heatmap
    """
    interface_meta = _resolve_interface_metadata(solver=solver)

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
        ax_a, ax_b = axes[0]
        ax_c, ax_d = axes[1]

        face_x = solver.face_positions_x
        pc_a = ax_a.pcolormesh(face_x, solver.grid_y, solver.G_x.T, cmap="viridis", shading="nearest")
        fig.colorbar(pc_a, ax=ax_a, label=r"$G_x$ [W/(m$^2$K)]", shrink=0.85)
        _add_interface_lines(ax_a, interface_meta["positions"])
        ax_a.set_xlabel("x (face position)")
        ax_a.set_ylabel("y")
        ax_a.set_title("x-Face Conductance")

        face_y = (solver.grid_y[:-1] + solver.grid_y[1:]) / 2.0
        pc_b = ax_b.pcolormesh(solver.grid_x, face_y, solver.G_y.T, cmap="viridis", shading="nearest")
        fig.colorbar(pc_b, ax=ax_b, label=r"$G_y$ [W/(m$^2$K)]", shrink=0.85)
        ax_b.set_xlabel("x")
        ax_b.set_ylabel("y (face position)")
        ax_b.set_title("y-Face Conductance")

        j_mid = solver.Ny // 2
        interior_idx = np.arange(0, solver.Nx - 1)
        ax_c.plot(interior_idx, solver.r_w[:solver.Nx - 1, j_mid], color="C0", label=r"$r_w$")
        ax_c.plot(interior_idx, solver.r_e[:solver.Nx - 1, j_mid], color="C3", label=r"$r_e$")
        for xi in interface_meta["positions"]:
            node_approx = (xi - solver.a) / solver.hx
            ax_c.axvline(node_approx, color="0.35", linestyle=":", alpha=0.8)
        ax_c.set_xlabel("Node index i")
        ax_c.set_ylabel("CN coefficient")
        ax_c.set_title(f"CN Coefficients at j={j_mid}")
        ax_c.legend()
        ax_c.grid(True)

        margin = 1.0 - solver.r_w - solver.r_e - solver.r_s - solver.r_n
        margin_active = margin[:solver.Nx - 1, :]
        vmax = float(np.max(np.abs(margin_active)))
        pc_d = ax_d.pcolormesh(
            solver.grid_x[:solver.Nx - 1],
            solver.grid_y,
            margin_active.T,
            cmap="RdYlGn",
            vmin=-vmax,
            vmax=vmax,
            shading="nearest",
        )
        fig.colorbar(pc_d, ax=ax_d, label=r"$1 - r_w - r_e - r_s - r_n$", shrink=0.85)
        ax_d.set_xlabel("x")
        ax_d.set_ylabel("y")
        ax_d.set_title("Diagonal Dominance Margin")

        _save_figure(fig, save_path, "physics", "face_conductance", layout="constrained")


def plot_multilayer_evolution(
    solver: FVSolver2D,
    T_hist: np.ndarray,
    save_path: str | Path | None = None,
):
    """Temperature evolution as 2D snapshots with one shared temperature scale per figure."""
    Nt = len(solver.t)
    n_snaps = 6
    snap_indices = np.linspace(0, Nt - 1, n_snaps, dtype=int)
    snapshots = T_hist[snap_indices]
    vmin = float(np.min(snapshots))
    vmax = float(np.max(snapshots))
    interface_meta = _resolve_interface_metadata(solver=solver)

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
        pcm = None
        for ax, t_idx, snap in zip(axes.ravel(), snap_indices, snapshots):
            pcm = ax.pcolormesh(
                solver.X,
                solver.Y,
                snap,
                cmap="inferno",
                shading="auto",
                vmin=vmin,
                vmax=vmax,
            )
            _add_interface_lines(ax, interface_meta["positions"])
            ax.set_title(f"t = {solver.t[t_idx]:.3f}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_aspect("equal")

        fig.colorbar(pcm, ax=axes.ravel().tolist(), label="Temperature", shrink=0.9)
        fig.suptitle("Temperature Evolution")
        _save_figure(fig, save_path, "physics", "multilayer_evolution", layout="constrained")


def plot_heat_flux_profile(
    solver: FVSolver2D,
    T_hist: np.ndarray,
    save_path: str | Path | None = None,
):
    """Applied boundary flux and numerical x-direction flux at mid-y.

    2x2 layout:
        (a) Temporal trace a(t) at the y of peak |s(y)|
        (b) Spatial profile s(y) at the time of peak |q|
        (c) Heatmap q_left(y, t)
        (d) Numerical x-flux at y=mid at selected time snapshots
    """
    Nt = len(solver.t)
    j_mid = solver.Ny // 2
    interface_meta = _resolve_interface_metadata(solver=solver)

    Q = _evaluate_q_left_field(solver)  # (Nt, Ny)
    is_vector = bool(getattr(solver, "_q_left_is_vector", False))

    abs_max_per_y = np.max(np.abs(Q), axis=0)
    j_peak = int(np.argmax(abs_max_per_y)) if is_vector else j_mid
    abs_max_per_t = np.max(np.abs(Q), axis=1)
    t_peak = int(np.argmax(abs_max_per_t))
    q_abs = max(float(np.max(np.abs(Q))), 1e-12)

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        ax_a, ax_b = axes[0, 0], axes[0, 1]
        ax_c, ax_d = axes[1, 0], axes[1, 1]

        ax_a.plot(solver.t, Q[:, j_peak], color="C3")
        ax_a.axvline(solver.t_on, color="C2", linestyle=":", alpha=0.8, label=f"t_on={solver.t_on}")
        ax_a.axvline(solver.t_off, color="C1", linestyle=":", alpha=0.8, label=f"t_off={solver.t_off}")
        ax_a.set_xlabel("Time")
        ax_a.set_ylabel("q_left(t)")
        title_a = f"Temporal trace at y = {solver.grid_y[j_peak]:.2f}" if is_vector else "Applied Boundary Flux (uniform)"
        ax_a.set_title(title_a)
        ax_a.legend()
        ax_a.grid(True)

        if is_vector:
            ax_b.plot(solver.grid_y, Q[t_peak, :], color="C0")
            ax_b.set_title(f"Spatial profile at t = {solver.t[t_peak]:.3f}")
        else:
            ax_b.plot(solver.grid_y, np.ones_like(solver.grid_y), color="C0")
            ax_b.set_ylim(-0.1, 1.5)
            ax_b.set_title("Spatial profile (uniform)")
        ax_b.set_xlabel("y")
        ax_b.set_ylabel("q_left(y) at t_peak")
        ax_b.grid(True)

        pcm = ax_c.pcolormesh(
            solver.t,
            solver.grid_y,
            Q.T,
            cmap="coolwarm",
            vmin=-q_abs,
            vmax=q_abs,
            shading="auto",
        )
        ax_c.axvline(solver.t_on, color="0.2", linestyle=":", alpha=0.8)
        ax_c.axvline(solver.t_off, color="0.2", linestyle=":", alpha=0.8)
        ax_c.set_xlabel("Time")
        ax_c.set_ylabel("y")
        ax_c.set_title("q_left(y, t)")
        fig.colorbar(pcm, ax=ax_c, label=r"$q_{left}$", shrink=0.9)

        snap_indices = np.linspace(1, Nt - 1, min(5, Nt - 1), dtype=int)
        cmap_snap = plt.cm.viridis
        for i, t_idx in enumerate(snap_indices):
            dT = T_hist[t_idx, 1:, j_mid] - T_hist[t_idx, :-1, j_mid]
            q_face = -solver.G_x[:, j_mid] * dT
            ax_d.plot(
                solver.face_positions_x,
                q_face,
                color=cmap_snap(i / max(len(snap_indices) - 1, 1)),
                label=f"t={solver.t[t_idx]:.3f}",
            )

        _add_interface_lines(ax_d, interface_meta["positions"])
        ax_d.set_xlabel("Face position (x)")
        ax_d.set_ylabel("Heat flux q")
        ax_d.set_title(f"Numerical x-Flux at y = {solver.grid_y[j_mid]:.2f}")
        ax_d.legend(loc="best")
        ax_d.grid(True)

        _save_figure(fig, save_path, "physics", "heat_flux_profile")


def plot_bc_verification(
    solver: FVSolver2D,
    T_hist: np.ndarray,
    save_path: str | Path | None = None,
):
    """Postprocessed boundary-condition diagnostics for the 2D solver.

    2x2 layout:
        (0,0) T(x=1, y, t) heatmap with max |T - T_right| annotation
        (0,1) Left-wall one-sided estimate of d_x T(0, y, t)
        (1,0) Bottom-wall one-sided estimate of d_y T(x, 0, t)
        (1,1) Top-wall one-sided estimate of d_y T(x, 1, t)

    The derivatives are postprocessed from cell-center samples; the solver
    enforces the BC fluxes to machine precision via its FV stencil.
    """
    T_hist = np.asarray(T_hist)
    t = np.asarray(solver.t)
    hx = float(solver.hx)
    hy = float(solver.grid_y[1] - solver.grid_y[0])
    T_right_target = float(solver.T_right(0.0))

    T_right_field = T_hist[:, -1, :]
    left_deriv = (T_hist[:, 1, :] - T_hist[:, 0, :]) / hx
    bottom_deriv = (T_hist[:, :, 1] - T_hist[:, :, 0]) / hy
    top_deriv = (T_hist[:, :, -1] - T_hist[:, :, -2]) / hy

    max_right_err = float(np.max(np.abs(T_right_field - T_right_target)))
    max_left = float(np.max(np.abs(left_deriv)))
    max_bottom = float(np.max(np.abs(bottom_deriv)))
    max_top = float(np.max(np.abs(top_deriv)))

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

        ax = axes[0, 0]
        pc = ax.pcolormesh(t, solver.grid_y, T_right_field.T, cmap="inferno", shading="auto")
        fig.colorbar(pc, ax=ax, label="T(x=1, y, t)")
        ax.set_xlabel("t")
        ax.set_ylabel("y")
        ax.set_title(
            f"Right Dirichlet: max |T - {T_right_target:.1f}| = {max_right_err:.2e}"
        )

        ax = axes[0, 1]
        vlim = max(max_left, 1e-12)
        pc = ax.pcolormesh(
            t, solver.grid_y, left_deriv.T,
            cmap="RdBu_r", shading="auto", vmin=-vlim, vmax=vlim,
        )
        fig.colorbar(pc, ax=ax, label=r"$\partial_x T(0, y, t)$")
        ax.set_xlabel("t")
        ax.set_ylabel("y")
        ax.set_title(f"Left Neumann: max |dT/dx| = {max_left:.2e}")

        ax = axes[1, 0]
        vlim = max(max_bottom, 1e-12)
        pc = ax.pcolormesh(
            t, solver.grid_x, bottom_deriv.T,
            cmap="RdBu_r", shading="auto", vmin=-vlim, vmax=vlim,
        )
        fig.colorbar(pc, ax=ax, label=r"$\partial_y T(x, 0, t)$")
        ax.set_xlabel("t")
        ax.set_ylabel("x")
        ax.set_title(f"Bottom adiabatic: max |dT/dy| = {max_bottom:.2e}")

        ax = axes[1, 1]
        vlim = max(max_top, 1e-12)
        pc = ax.pcolormesh(
            t, solver.grid_x, top_deriv.T,
            cmap="RdBu_r", shading="auto", vmin=-vlim, vmax=vlim,
        )
        fig.colorbar(pc, ax=ax, label=r"$\partial_y T(x, 1, t)$")
        ax.set_xlabel("t")
        ax.set_ylabel("x")
        ax.set_title(f"Top adiabatic: max |dT/dy| = {max_top:.2e}")

        fig.suptitle(
            "BC verification (postprocessed cell-center differences; "
            "solver-imposed fluxes are machine precision)"
        )
        _save_figure(fig, save_path, "physics", "bc_verification", layout="constrained")


# ============================================================
# Controlled ITR temperature-jump sweep (solver-truth, no model)
# ============================================================
# A no-model diagnostic: solve the production FV operator for each benchmark with
# the thermal driver held fixed while interface thermal resistance (ITR) is
# swept, and report the physical contact temperature-jump magnitude
# |dT_contact| = R_c * G * (T_L - T_R). Every solve routes through the
# benchmark's ProblemSpec.configure_solver so geometry, conductivities, forcing,
# and the (scalar or (Ny,)) interface_R are always correct per benchmark.

# Cap mirrors src/physics/internal_source.py:R_PEAK_MAX so source_itr void peaks
# stay inside the sampled R_c(y) range.
_ITR_R_PEAK_MAX = 3.0

# Canonical sin temporal driver shared by the forcing/interfaces panels. Keys
# match temporal_sin(A, f, t_on, t_off, phase, tukey_alpha, rectified).
_ITR_SIN_TEMPORAL_PARAMS = {
    "A": 200.0, "f": 5.0, "t_on": 0.0, "t_off": 0.2,
    "phase": 0.0, "tukey_alpha": 0.5, "rectified": True,
}

_ITR_SWEEP_FIELDS = (
    "benchmark", "itr_kind", "itr_value", "R_c", "R_c_base", "R_c_amp",
    "R_c_peak", "R_c_y0", "R_c_sigma", "interface_x", "time_requested",
    "time_actual", "time_index", "mean_abs_jump_K", "rms_jump_K",
    "peak_abs_jump_K",
)


def _itr_canonical_params(benchmark: str, itr_value: float,
                          X: np.ndarray, Y: np.ndarray, base_kwargs: dict) -> tuple[dict, dict]:
    """Return (solver_params, record_meta) for one (benchmark, itr_value).

    ``solver_params`` keys match the benchmark's ``configure_solver``; a wrong or
    missing key raises immediately when round-tripped, so these dicts cannot
    silently drift. ``record_meta`` carries the ITR bookkeeping columns for the
    CSV/records (kind, swept value, and void params for source_itr).
    """
    T_right = 300.0
    if benchmark == "forcing":
        params = {
            "R_c": float(itr_value),
            "T0": np.full(X.shape, T_right, dtype=np.float32),
            "temporal_family": "sin",
            "temporal_params": dict(_ITR_SIN_TEMPORAL_PARAMS),
            "spatial_family": "uniform",
            "spatial_params": {},
        }
        meta = {"itr_kind": "scalar_Rc", "itr_value": float(itr_value), "R_c": float(itr_value)}
        return params, meta

    if benchmark in ("source", "source_itr", "source_itr_sin"):
        t_off = 0.75 * float(base_kwargs["t_final"])
        params = {
            "interface_x": 0.5,
            "x_h": 0.45, "y_h": 0.5, "w_h": 0.1, "h_h": 0.1,
            "A": 8000.0, "t_off": t_off,
            "T0": np.full(X.shape, T_right, dtype=np.float32),
        }
        if benchmark == "source":
            params["R_c"] = float(itr_value)
            meta = {"itr_kind": "scalar_Rc", "itr_value": float(itr_value), "R_c": float(itr_value)}
            return params, meta
        R_c_base = 0.05
        R_c_peak = min(float(itr_value), _ITR_R_PEAK_MAX)
        if benchmark == "source_itr_sin":
            # source_itr_sin: itr_value is the sinusoid peak R_c,peak = R_base + A
            # (capped), so the amplitude is the excess over the base. Universal
            # column rule holds: store R_c == R_c_base.
            R_c_A = R_c_peak - R_c_base
            params.update({
                "R_c_base": R_c_base, "R_c_A": R_c_A, "R_c": R_c_base,
            })
            meta = {
                "itr_kind": "Rc_peak", "itr_value": R_c_peak, "R_c": R_c_base,
                "R_c_base": R_c_base, "R_c_A": R_c_A, "R_c_peak": R_c_peak,
            }
            return params, meta
        # source_itr: itr_value is the void peak R_c,peak (capped); amp is the
        # excess over the base, mirroring the benchmark's universal-column rule
        # of storing R_c == R_c_base.
        R_c_amp = R_c_peak - R_c_base
        params.update({
            "R_c_base": R_c_base, "R_c_amp": R_c_amp,
            "R_c_y0": 0.5, "R_c_sigma": 0.12, "R_c": R_c_base,
        })
        meta = {
            "itr_kind": "Rc_peak", "itr_value": R_c_peak, "R_c": R_c_base,
            "R_c_base": R_c_base, "R_c_amp": R_c_amp, "R_c_peak": R_c_peak,
            "R_c_y0": 0.5, "R_c_sigma": 0.12,
        }
        return params, meta

    if benchmark == "interfaces":
        from src.physics.init_conditions import build_ic
        # Asymmetric two-bump IC (positive bump left of the interface, negative
        # bump right of it) keeps this panel visibly distinct from the others.
        ic_params = {
            "A_list": [25.0, -15.0],
            "mu_x_list": [0.35, 0.80],
            "mu_y_list": [0.40, 0.65],
            "sigma_list": [0.12, 0.10],
        }
        T0 = build_ic("hot_spot_2d", ic_params, X, Y,
                      T_right=T_right, b=float(base_kwargs["b"]))
        params = {
            "R_c": float(itr_value),
            "interface_x": 0.65,
            "T0": T0,
            "temporal_family": "sin",
            "temporal_params": dict(_ITR_SIN_TEMPORAL_PARAMS),
            "spatial_family": "uniform",
            "spatial_params": {},
        }
        meta = {"itr_kind": "scalar_Rc", "itr_value": float(itr_value), "R_c": float(itr_value)}
        return params, meta

    raise ValueError(f"Unknown benchmark for ITR sweep: {benchmark!r}")


def _write_itr_sweep_csv(records: list[dict], out_path: Path) -> Path:
    """Write the ITR-sweep records to CSV (one row per benchmark x itr x time)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _fmt(key: str, value) -> str:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return ""
        if key == "time_index":
            return str(int(value))
        if key in ("benchmark", "itr_kind"):
            return str(value)
        return f"{float(value):.6g}"

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(_ITR_SWEEP_FIELDS)
        for row in records:
            writer.writerow([_fmt(k, row.get(k)) for k in _ITR_SWEEP_FIELDS])
    return out_path


def plot_itr_temperature_jump_sweep(
    benchmarks: tuple[str, ...] = ("forcing", "source", "source_itr", "interfaces"),
    Nx: int = 40,
    Ny: int = 40,
    dt: float = 0.005,
    t_final: float = 0.3,
    requested_times: tuple[float, ...] = (0.10, 0.20, 0.30),
    scalar_rc_values: tuple[float, ...] = (0.05, 0.15, 0.35, 0.70, 1.00),
    rc_peak_values: tuple[float, ...] = (0.05, 0.15, 0.35, 0.70, 1.00, 2.00, 3.00),
    save_path: str | Path | None = None,
) -> dict:
    """Solver-truth sweep of contact temperature-jump magnitude vs ITR.

    For each benchmark, the thermal driver is held fixed while interface thermal
    resistance is swept (scalar ``R_c`` for forcing/source/interfaces; void peak
    ``R_c,peak`` for source_itr). Every solve is wired through the production
    ``ProblemSpec.configure_solver`` so per-benchmark physics is never re-derived
    here. Returns ``{"png", "csv", "records"}``; ``records`` is one dict per
    (benchmark, itr_value, requested_time).
    """
    # Reusing the underscore-prefixed contact-jump helpers across visual modules
    # is acceptable for now; if a third caller appears these should move to a
    # shared visual/jump_utils.py. Lazy-imported here to keep physics_plots light
    # and avoid an import cycle with dataset_plots.
    from visual.dataset_plots import (
        _interface_contact_jump_map,
        _contact_jump_reductions,
    )
    from problems.registry import get_problem
    from src.physics.boundary_forcing import default_ramp_seconds

    a, b, c, d = 0.0, 1.0, 0.0, 1.0
    x_grid = np.linspace(a, b, Nx)
    y_grid = np.linspace(c, d, Ny)
    X, Y = np.meshgrid(x_grid, y_grid, indexing="ij")

    # base_kwargs intentionally mirrors data/generate_dataset.py:146-152 so the
    # sweep solves use the same domain/material/forcing defaults as the dataset.
    # If those defaults change, re-sync here.
    t_ramp = default_ramp_seconds(dt)
    layers = [
        Layer2D(x_left=a, x_right=0.5, rho=1.0, cp=1.0, k=2.0),
        Layer2D(x_left=0.5, x_right=b, rho=1.0, cp=1.0, k=1.0),
    ]
    base_kwargs = dict(
        a=a, b=b, c=c, d=d, Nx=Nx, Ny=Ny,
        lam_target=0.8, layers=layers, t_final=t_final,
        flux_f=0.0, flux_A=0.0, t_on=0.0, t_off=0.2, phase=0.0,
        dt=dt, tukey_alpha=0.5, y_grid=y_grid, X=X, Y=Y,
        ramp_seconds=t_ramp,
    )

    requested_times = tuple(float(t) for t in requested_times)
    # panel_data[benchmark][time_req] -> {"x": [...], "y": [...]}; line x-axis is
    # ITR value, y-axis is mean |dT_contact| at the nearest saved time.
    panel_data: dict[str, dict[float, dict[str, list[float]]]] = {}
    records: list[dict] = []

    for benchmark in benchmarks:
        spec = get_problem(benchmark)
        sweep_values = (
            rc_peak_values
            if benchmark in ("source_itr", "source_itr_sin")
            else scalar_rc_values
        )
        panel_data[benchmark] = {t_req: {"x": [], "y": []} for t_req in requested_times}

        for itr_value in sweep_values:
            params, meta = _itr_canonical_params(benchmark, itr_value, X, Y, base_kwargs)
            sim = spec.configure_solver(params, base_kwargs)
            t, x_solved, y_solved, T_hist = sim.solve(
                T0=params["T0"], store_trajectory=True
            )
            t = np.asarray(t)

            interface_x = float(sim.interface_positions[0])
            k_left = float(sim.layers[0].k)
            k_right = float(sim.layers[1].k)
            R_c = sim.interface_R[0]

            jump_map = _interface_contact_jump_map(
                T_hist, np.asarray(x_solved), interface_x, R_c, k_left, k_right
            )  # (Nt, Ny)
            red = _contact_jump_reductions(jump_map)

            for t_req in requested_times:
                idx = int(np.argmin(np.abs(t - t_req)))
                mean_abs = float(red["mean_abs"][idx])
                rms = float(red["rms"][idx])
                peak_abs = float(np.max(np.abs(jump_map[idx])))
                time_actual = float(t[idx])

                panel_data[benchmark][t_req]["x"].append(float(meta["itr_value"]))
                panel_data[benchmark][t_req]["y"].append(mean_abs)

                records.append({
                    "benchmark": benchmark,
                    "itr_kind": meta["itr_kind"],
                    "itr_value": meta["itr_value"],
                    "R_c": meta.get("R_c"),
                    "R_c_base": meta.get("R_c_base"),
                    "R_c_amp": meta.get("R_c_amp"),
                    "R_c_peak": meta.get("R_c_peak"),
                    "R_c_y0": meta.get("R_c_y0"),
                    "R_c_sigma": meta.get("R_c_sigma"),
                    "interface_x": interface_x,
                    "time_requested": t_req,
                    "time_actual": time_actual,
                    "time_index": idx,
                    "mean_abs_jump_K": mean_abs,
                    "rms_jump_K": rms,
                    "peak_abs_jump_K": peak_abs,
                })

    # Figure: near-square grid sized to the number of benchmarks.
    n = len(benchmarks)
    ncols = max(1, math.ceil(math.sqrt(n)))
    nrows = max(1, math.ceil(n / ncols))
    cmap = plt.cm.viridis
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    n_times = max(len(requested_times), 1)

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 4.6 * nrows),
                                 squeeze=False)
        flat_axes = axes.ravel()

        for panel_idx, benchmark in enumerate(benchmarks):
            ax = flat_axes[panel_idx]
            for ti, t_req in enumerate(requested_times):
                series = panel_data[benchmark][t_req]
                color = cmap(ti / max(n_times - 1, 1))
                ax.plot(
                    series["x"], series["y"],
                    marker=markers[ti % len(markers)],
                    color=color, label=f"t = {t_req:.2f}",
                )
            ax.set_title(benchmark)
            ax.set_xlabel(
                r"$R_{c,\mathrm{peak}}$"
                if benchmark in ("source_itr", "source_itr_sin")
                else r"$R_c$"
            )
            ax.set_ylabel(r"mean $|\Delta T_{\mathrm{contact}}|$ [K]")
            ax.grid(True)

        for hidden_idx in range(n, len(flat_axes)):
            flat_axes[hidden_idx].axis("off")

        # Reserve a top band for the suptitle + shared legend so neither overlaps
        # the panels (tight_layout does not account for figure-level artists).
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.90])
        handles, labels = flat_axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=n_times,
                   title="requested time", bbox_to_anchor=(0.5, 0.95))
        fig.suptitle("Contact temperature-jump magnitude vs interface resistance",
                     y=0.995)

        png_path = _save_figure(fig, save_path, "physics", "itr_temperature_jump_sweep",
                                layout="none")

    csv_path = png_path.with_name("itr_temperature_jump_sweep.csv")
    _write_itr_sweep_csv(records, csv_path)
    print(f"Saved itr_temperature_jump_sweep CSV to: {csv_path}")

    return {"png": png_path, "csv": csv_path, "records": records}
