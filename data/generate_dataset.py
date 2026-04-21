import numpy as np
from src.physics.fv_solver_2d import FVSolver2D, Layer2D
from scipy.stats import qmc


def random_ic(a: float, b: float, c: float, d: float,
              X: np.ndarray, Y: np.ndarray, rng) -> np.ndarray:
    Lx = b - a
    Ly = d - c

    # x-direction random combination
    cx0 = rng.uniform(0.5, 1.5)
    cx1 = rng.uniform(-0.3, 0.3)
    cx2 = rng.uniform(-0.3, 0.3)
    fx = (cx0 * np.cos((np.pi * X) / (2.0 * Lx)) + cx1 * np.sin((np.pi * X) / Lx) + cx2 * np.cos((2.0 * np.pi * X) / Lx))

    # y-direction random combination
    cy0 = rng.uniform(0.5, 1.5)
    cy1 = rng.uniform(-0.3, 0.3)
    cy2 = rng.uniform(-0.3, 0.3)
    fy = (cy0 * np.cos((np.pi * Y) / (2.0 * Ly)) + cy1 * np.sin((np.pi * Y) / Ly) + cy2 * np.cos((2.0 * np.pi * Y) / Ly))

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


def build_sim_params(a: float, b: float, c: float, d: float, X: np.ndarray, Y: np.ndarray, num_sims: int, rng, lhs_seed: int = 0) -> list:

    samples_scaled = generate_lhs_samples(num_sims=num_sims, seed=lhs_seed)

    # -------- EXTRACT PARAMS FROM LHS SAMPLES --------
    amplitudes  = samples_scaled[:, 0]
    frequencies = samples_scaled[:, 1]
    R_c_values  = samples_scaled[:, 2]

    T0_list = [random_ic(a, b, c, d, X, Y, rng) for _ in range(num_sims)]

    # shape: [(amp, freq, T0, R_c), ...]
    sim_params = list(zip(amplitudes, frequencies, T0_list, R_c_values))

    return sim_params


def generate_sim_data(num_sims: int = 1024, save_stride: int = 5) -> None:
    # seed 0 for reproducibility after I generate all simulations
    rng = np.random.default_rng(0)

    # fixed grid geometry and time parameters
    a, b, c, d = 0.0, 1.0, 0.0, 1.0
    Nx, Ny = 100, 100
    dt = 0.005
    t_final = 1.0

    # fixed flux window parameters
    t_on, t_off = 0.0, 0.2
    phase = 0.0
    tukey_alpha = 0.5

    x_grid = np.linspace(a, b, Nx)
    y_grid = np.linspace(c, d, Ny)
    X, Y = np.meshgrid(x_grid, y_grid, indexing="ij")

    # compute Nt from the time grid (same for all sims since dt and t_final are fixed)
    t_grid_template = np.arange(0.0, t_final + 1e-12, dt)
    Nt = len(t_grid_template)
    # Subsample every `save_stride` steps on disk to cap file size. The solver
    # still integrates at full dt
    Nt_saved = len(t_grid_template[::save_stride])

    # Fixed materials: k1=2, k2=1, rho=cp=1 for both layers
    layers = [
        Layer2D(x_left=0.0, x_right=0.5, rho=1.0, cp=1.0, k=2.0),
        Layer2D(x_left=0.5, x_right=1.0, rho=1.0, cp=1.0, k=1.0),
    ]

    print("Building simulation parameters.")

    # shape: [(amp, freq, T0, R_c), ...]
    sim_params = build_sim_params(a=a, b=b, c=c, d=d, X=X, Y=Y, num_sims=num_sims, rng=rng, lhs_seed=0)

    trajectories = np.zeros((num_sims, Nt_saved, Nx, Ny), dtype=np.float32)

    for i, (amp, freq, T0, R_c) in enumerate(sim_params):
        sim = FVSolver2D(
            a=a, b=b, c=c, d=d, Nx=Nx, Ny=Ny,
            lam_target=0.8, layers=layers,
            t_final=t_final, flux_f=float(freq), flux_A=float(amp),
            t_on=t_on, t_off=t_off, phase=phase,
            dt=dt, tukey_alpha=tukey_alpha,
            interface_R=[float(R_c)],
        )

        t, x, y, T_hist = sim.solve(T0=T0, store_trajectory=True)

        trajectories[i] = T_hist[::save_stride].astype(np.float32)

        print(f"Finished simulation {i}")

    # x, y, and t are the same for all simulations, so just use the last ones
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
    generate_sim_data(num_sims=4000)
