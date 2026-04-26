import numpy as np
from src.physics.fv_solver_2d import FVSolver2D, Layer2D
from src.physics.boundary_forcing import (
    SPATIAL_FAMILIES,
    TEMPORAL_FAMILIES,
    SPATIAL_SAMPLERS,
    sample_spatial_family,
    sample_temporal_family,
    build_qL,
)
from scipy.stats import qmc


def random_ic(a: float, b: float, c: float, d: float,
              X: np.ndarray, Y: np.ndarray, rng) -> np.ndarray:
    Lx = b - a
    Ly = d - c

    # x-direction random combination
    cx0 = rng.uniform(0.5, 1.5)
    cx1 = rng.uniform(-0.3, 0.3)
    cx2 = rng.uniform(-0.3, 0.3)
    Xn = X - a
    Yn = Y - c
    fx = (cx0 * np.cos((np.pi * Xn) / (2.0 * Lx)) + cx1 * np.sin((np.pi * Xn) / Lx) + cx2 * np.cos((2.0 * np.pi * Xn) / Lx))

    # y-direction random combination
    cy0 = rng.uniform(0.5, 1.5)
    cy1 = rng.uniform(-0.3, 0.3)
    cy2 = rng.uniform(-0.3, 0.3)
    fy = (cy0 * np.cos((np.pi * Yn) / (2.0 * Ly)) + cy1 * np.sin((np.pi * Yn) / Ly) + cy2 * np.cos((2.0 * np.pi * Yn) / Ly))

    return (fx * fy).astype(np.float32)


def generate_lhs_samples(num_sims: int, seed: int = 0) -> np.ndarray:

    # -------- DEFINE PARAM. RANGES ------
    param_ranges = {
        "amplitude": (50.0, 300.0),
        "frequency": (1.0, 20.0),
        "R_c":       (0.05, 1.0),
    }

    # Parameters sampled log-uniformly instead of uniformly.
    # Frequency: penetration depth delta ~ 1/sqrt(f), so log-uniform gives
    # more uniform coverage of the physics space (deep vs shallow penetration).
    log_uniform_params = {"frequency"}

    # create list of parameter names to help calculate sample dim. and lower & upper bounds
    param_names = list(param_ranges.keys())
    sample_dim = len(param_names)
    lower_bounds = np.array([param_ranges[name][0] for name in param_names], dtype=np.float32)
    upper_bounds = np.array([param_ranges[name][1] for name in param_names], dtype=np.float32)

    # ------- GENERATE LHS SAMPLES -------
    sampler = qmc.LatinHypercube(d=sample_dim, seed=seed)
    samples_unit = sampler.random(n=num_sims)

    # uniform scaling for all parameters first
    samples_scaled = qmc.scale(samples_unit, lower_bounds, upper_bounds).astype(np.float32)

    # override log-uniform parameters: x = lo * (hi/lo)^u, u in [0, 1]
    for idx, name in enumerate(param_names):
        if name in log_uniform_params:
            lo, hi = lower_bounds[idx], upper_bounds[idx]
            samples_scaled[:, idx] = lo * (hi / lo) ** samples_unit[:, idx]

    return samples_scaled


def build_sim_params(a: float, b: float, c: float, d: float, X: np.ndarray, Y: np.ndarray,
                     num_sims: int, rng, rng_profile, lhs_seed: int = 0,
                     t_on: float = 0.0, t_off: float = 0.2, phase: float = 0.0,
                     tukey_alpha: float = 0.5) -> list:

    samples_scaled = generate_lhs_samples(num_sims=num_sims, seed=lhs_seed)

    amplitudes  = samples_scaled[:, 0]
    frequencies = samples_scaled[:, 1]
    R_c_values  = samples_scaled[:, 2]

    sim_params = []
    for i in range(num_sims):
        amp = float(amplitudes[i])
        freq = float(frequencies[i])
        R_c = float(R_c_values[i])
        T0 = random_ic(a, b, c, d, X, Y, rng)

        # Sample temporal family (currently only "sin", but kept for future-proofing)
        temporal_family = sample_temporal_family(rng_profile)
        temporal_params = {
            "A": amp, "f": freq,
            "t_on": t_on, "t_off": t_off,
            "phase": phase, "tukey_alpha": tukey_alpha,
        }

        # Sample spatial family + its parameters
        spatial_family = sample_spatial_family(rng_profile)
        spatial_params = SPATIAL_SAMPLERS[spatial_family](rng_profile)

        sim_params.append({
            "amp": amp,
            "freq": freq,
            "R_c": R_c,
            "T0": T0,
            "temporal_family": temporal_family,
            "temporal_params": temporal_params,
            "spatial_family": spatial_family,
            "spatial_params": spatial_params,
        })

    return sim_params


def generate_sim_data(num_sims: int = 2000, save_stride: int = 2) -> None:
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

    print("Building simulation parameters.")

    sim_params = build_sim_params(
        a=a, b=b, c=c, d=d, X=X, Y=Y,
        num_sims=num_sims, rng=rng, rng_profile=rng_profile, lhs_seed=0,
        t_on=t_on, t_off=t_off, phase=phase, tukey_alpha=tukey_alpha,
    )

    trajectories = np.zeros((num_sims, Nt_saved, Nx, Ny), dtype=np.float32)

    for i, params in enumerate(sim_params):
        q_left_fn, _ = build_qL(
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
            flux_f=params["freq"], flux_A=params["amp"],
            t_on=t_on, t_off=t_off, phase=phase,
            dt=dt, tukey_alpha=tukey_alpha,
            interface_R=[params["R_c"]],
            q_left_fn=q_left_fn,
        )

        t, x, y, T_hist = sim.solve(T0=params["T0"], store_trajectory=True)

        trajectories[i] = T_hist[::save_stride].astype(np.float32)

        print(f"Finished simulation {i}")

    x_grid = x.astype(np.float32)
    y_grid = y.astype(np.float32)
    t_grid = t[::save_stride].astype(np.float32)

    np.save("x_grid.npy", x_grid)
    np.save("y_grid.npy", y_grid)
    np.save("t_grid.npy", t_grid)
    np.save("trajectories.npy", trajectories)
    np.save("sim_params.npy", np.array(sim_params, dtype=object), allow_pickle=True)
    print("Saved:", x_grid.shape, y_grid.shape, t_grid.shape, trajectories.shape)

if __name__ == '__main__':
    generate_sim_data(num_sims=2000)
