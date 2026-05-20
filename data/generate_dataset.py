import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from src.physics.fv_solver_2d import FVSolver2D, Layer2D
from src.physics.boundary_forcing import (
    SPATIAL_FAMILIES,
    TEMPORAL_FAMILIES,
    SPATIAL_SAMPLERS,
    TEMPORAL_SAMPLERS,
    sample_spatial_family,
    sample_temporal_family,
    build_qL,
    build_qL_integral,
)
from src.physics.init_conditions import (
    IC_SAMPLERS,
    sample_ic_family,
    build_ic,
)


DATA_DIR = Path(__file__).resolve().parent


def generate_lhs_samples(num_sims: int, seed: int = 0) -> np.ndarray:
    # R_c is the only material parameter sampled here; forcing-family params
    # are drawn per-sim by the boundary-forcing samplers (see TEMPORAL_SAMPLERS).
    param_ranges = {"R_c": (0.05, 1.0)}
    param_names = list(param_ranges.keys())
    sample_dim = len(param_names)
    lower_bounds = np.array([param_ranges[name][0] for name in param_names], dtype=np.float32)
    upper_bounds = np.array([param_ranges[name][1] for name in param_names], dtype=np.float32)

    rng = np.random.default_rng(seed)
    edges = np.linspace(0.0, 1.0, num_sims + 1, dtype=np.float64)
    widths = edges[1:] - edges[:-1]
    samples_unit = edges[:-1, None] + widths[:, None] * rng.random((num_sims, sample_dim))
    for j in range(sample_dim):
        rng.shuffle(samples_unit[:, j])

    samples_scaled = lower_bounds + samples_unit * (upper_bounds - lower_bounds)
    samples_scaled = samples_scaled.astype(np.float32)
    return samples_scaled


def build_sim_params(a: float, b: float, c: float, d: float, X: np.ndarray, Y: np.ndarray,
                     num_sims: int, rng, rng_profile,
                     dt: float, t_final: float, lhs_seed: int = 0,
                     t_on: float = 0.0, t_off: float = 0.2, phase: float = 0.0,
                     tukey_alpha: float = 0.5,
                     T_right: float = 300.0) -> list:

    samples_scaled = generate_lhs_samples(num_sims=num_sims, seed=lhs_seed)
    R_c_values = samples_scaled[:, 0]

    temporal_window = dict(t_on=t_on, t_off=t_off, phase=phase, tukey_alpha=tukey_alpha)

    # building full list of simulation parameters
    sim_params = []
    for i in range(num_sims):
        R_c = float(R_c_values[i])

        # IC family + params per sim. The smoothstep taper inside build_ic
        # drives the deviation to zero at the right Dirichlet edge so the
        # exact pin in the last column does not introduce a discontinuity.
        ic_family = sample_ic_family(rng)
        ic_params = IC_SAMPLERS[ic_family](rng, Nx=X.shape[0], Ny=X.shape[1])
        T0 = build_ic(ic_family, ic_params, X, Y, T_right=T_right, b=b)

        temporal_family = sample_temporal_family(rng_profile)
        temporal_params = TEMPORAL_SAMPLERS[temporal_family](rng_profile, dt=dt, t_final=t_final, **temporal_window)

        spatial_family = sample_spatial_family(rng_profile)
        spatial_params = SPATIAL_SAMPLERS[spatial_family](rng_profile)

        sim_params.append({
            "R_c": R_c,
            "T0": T0,
            "ic_family": ic_family,
            "ic_params": ic_params,
            "temporal_family": temporal_family,
            "temporal_params": temporal_params,
            "spatial_family": spatial_family,
            "spatial_params": spatial_params,
        })

    return sim_params


def generate_sim_data(
    num_sims: int = 2000,
    save_stride: int = 2,
    save_dir: Path | str | None = None,
) -> None:
    rng = np.random.default_rng(0)
    rng_profile = np.random.default_rng(1)

    a, b, c, d = 0.0, 1.0, 0.0, 1.0
    Nx, Ny = 100, 100
    dt = 0.005
    t_final = 0.3

    t_on, t_off = 0.0, 0.2
    phase = 0.0
    tukey_alpha = 0.5

    x_grid = np.linspace(a, b, Nx)
    y_grid = np.linspace(c, d, Ny)
    X, Y = np.meshgrid(x_grid, y_grid, indexing="ij")

    t_grid_template = np.arange(0.0, t_final + 1e-12, dt)
    Nt = len(t_grid_template)
    Nt_saved = len(t_grid_template[::save_stride])

    layers = [
        Layer2D(x_left=0.0, x_right=0.5, rho=1.0, cp=1.0, k=2.0),
        Layer2D(x_left=0.5, x_right=1.0, rho=1.0, cp=1.0, k=1.0),
    ]

    print("Building simulation parameters.", flush=True)

    sim_params = build_sim_params(
        a=a, b=b, c=c, d=d, X=X, Y=Y,
        num_sims=num_sims, rng=rng, rng_profile=rng_profile,
        dt=dt, t_final=t_final, lhs_seed=0,
        t_on=t_on, t_off=t_off, phase=phase, tukey_alpha=tukey_alpha,
    )

    trajectories = np.zeros((num_sims, Nt_saved, Nx, Ny), dtype=np.float32)

    # generate all simulations with varying parameters from simulation parameters
    for i, params in enumerate(sim_params):
        q_left_fn, _ = build_qL(
            temporal_family=params["temporal_family"],
            temporal_params=params["temporal_params"],
            spatial_family=params["spatial_family"],
            spatial_params=params["spatial_params"],
            y_grid=y_grid,
        )
        q_left_integral_fn, _ = build_qL_integral(
            temporal_family=params["temporal_family"],
            temporal_params=params["temporal_params"],
            spatial_family=params["spatial_family"],
            spatial_params=params["spatial_params"],
            y_grid=y_grid,
        )

        sim = FVSolver2D(
            a=a, b=b, c=c, d=d, Nx=Nx, Ny=Ny,
            lam_target=0.8, layers=layers,
            t_final=t_final,
            flux_f=0.0, flux_A=0.0,
            t_on=t_on, t_off=t_off, phase=phase,
            dt=dt, tukey_alpha=tukey_alpha,
            interface_R=[params["R_c"]],
            q_left_fn=q_left_fn,
            q_left_integral_fn=q_left_integral_fn,
        )

        t, x, y, T_hist = sim.solve(T0=params["T0"], store_trajectory=True)

        trajectories[i] = T_hist[::save_stride].astype(np.float32)

        print(f"Finished simulation {i}", flush=True)

    x_grid = x.astype(np.float32)
    y_grid = y.astype(np.float32)
    t_grid = t[::save_stride].astype(np.float32)

    save_path = Path(save_dir) if save_dir is not None else DATA_DIR
    save_path.mkdir(parents=True, exist_ok=True)

    np.save(save_path / "x_grid.npy", x_grid)
    np.save(save_path / "y_grid.npy", y_grid)
    np.save(save_path / "t_grid.npy", t_grid)
    # Saved t_grid spacing != solver dt when save_stride > 1; persist solver dt
    # so the dataset can normalize tau / dt_n cond slots against the same bounds
    # used during sampling.
    np.save(save_path / "dt.npy", np.float64(dt))
    np.save(save_path / "trajectories.npy", trajectories)
    np.save(save_path / "sim_params.npy", np.array(sim_params, dtype=object), allow_pickle=True)
    print("Saved to:", save_path, x_grid.shape, y_grid.shape, t_grid.shape, trajectories.shape, flush=True)

def main(argv: list[str] | None = None, generate_fn=generate_sim_data) -> int:
    parser = argparse.ArgumentParser(description="Generate FNO training data.")
    parser.add_argument("--num-sims", type=int, default=8000, help="Number of simulations to generate.")
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Directory for generated .npy files. Defaults to this script's data directory.",
    )
    args = parser.parse_args(argv)

    generate_fn(num_sims=args.num_sims, save_dir=args.save_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
