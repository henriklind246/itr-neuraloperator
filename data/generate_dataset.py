import numpy as np
from src.physics.fd_solver_1d import FDSolver1D, Layer1D
from scipy.stats import qmc


def random_ic(a: float, b: float, grid: np.ndarray, rng) -> np.ndarray:
    L = b - a

    # random linear combination, simple but random
    c0 = rng.uniform(0.5, 1.5)
    c1 = rng.uniform(-0.3, 0.3)
    c2 = rng.uniform(-0.3, 0.3)

    return (
        c0 * np.cos((np.pi * grid)/(2.0*L)) + c1 * np.sin((np.pi * grid)/L) + c2 * np.cos((2*np.pi*grid)/L)
    ).astype(np.float32)

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


def build_sim_params(a: float, b: float, grid: np.ndarray, num_sims: int, rng, lhs_seed: int = 0) -> list:

    samples_scaled = generate_lhs_samples(num_sims=num_sims, seed=lhs_seed)

    # -------- EXTRACT PARAMS FROM LHS SAMPLES --------
    amplitudes  = samples_scaled[:, 0]
    frequencies = samples_scaled[:, 1]
    R_c_values  = samples_scaled[:, 2]

    T0_list = []

    for _ in range(num_sims):
        T0 = random_ic(a=a, b=b, grid=grid, rng=rng)
        T0_list.append(T0)

    # shape: [(amp, freq, T0, R_c), ...]
    sim_params = list(zip(amplitudes, frequencies, T0_list, R_c_values))

    return sim_params


def generate_sim_data(num_sims: int = 1024) -> None:
    # seed 0 for reproducibility after I generate all simulations
    rng = np.random.default_rng(0)

    # fixed grid geometry and time parameters
    a, b, N = 0.0, 1.0, 100
    grid = np.linspace(a, b, N)
    dt = 0.005
    t_final = 1.0

    # fixed flux window parameters
    t_on, t_off = 0.0, 0.2
    phase = 0.0
    tukey_alpha = 0.5

    # compute Nt from the time grid (same for all sims since dt and t_final are fixed)
    t_grid_template = np.arange(0.0, t_final + 1e-12, dt)
    Nx = N
    Nt = len(t_grid_template)

    # Fixed materials: k1=2, k2=1, rho=cp=1 for both layers
    layers = [
        Layer1D(x_left=0.0, x_right=0.5, rho=1.0, cp=1.0, k=2.0),
        Layer1D(x_left=0.5, x_right=1.0, rho=1.0, cp=1.0, k=1.0),
    ]

    print("Building simulation parameters.")

    # shape: [(amp, freq, T0, R_c), ...]
    sim_params = build_sim_params(a=a, b=b, grid=grid, num_sims=num_sims, rng=rng, lhs_seed=0)

    trajectories = np.zeros((num_sims, Nt, Nx), dtype=np.float32)

    for i, (amp, freq, T0, R_c) in enumerate(sim_params):
        sim = FDSolver1D(
            a=a, b=b, N=N, lam_target=0.8, layers=layers,
            t_final=t_final, flux_f=float(freq), flux_A=float(amp),
            t_on=t_on, t_off=t_off, phase=phase, dt=dt, tukey_alpha=tukey_alpha,
            interface_R=[float(R_c)],
        )

        t, x, T_hist = sim.solve(T0=T0, store_trajectory=True)

        trajectories[i, :, :] = T_hist.astype(np.float32)

        print(f"Finished simulation {i}")

    # x and t are the same for all simulations, so just use the last ones
    x_grid = x.astype(np.float32)
    t_grid = t.astype(np.float32)

    np.save("x_grid.npy", x_grid)
    np.save("t_grid.npy", t_grid)
    np.save("trajectories.npy", trajectories)
    np.save("sim_params.npy", np.array(sim_params, dtype=object), allow_pickle=True)
    print("Saved:", x_grid.shape, t_grid.shape, trajectories.shape)

if __name__ == '__main__':
    generate_sim_data(num_sims=1024)
