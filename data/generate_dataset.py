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
        "k1":        (0.5, 5.0),
        "k2":        (0.5, 5.0),
        "rho_cp1":   (0.5, 5.0),
        "rho_cp2":   (0.5, 5.0),
    }

    # create list of parameter names to help calculate sample dim. and lower & upper bounds
    param_names = list(param_ranges.keys())
    sample_dim = len(param_names)
    lower_bounds = np.array([param_ranges[name][0] for name in param_names], dtype=np.float32)
    upper_bounds = np.array([param_ranges[name][1] for name in param_names], dtype=np.float32)

    # ------- GENERATE LHS SAMPLES -------
    sampler = qmc.LatinHypercube(d=sample_dim, seed=seed)
    samples_unit = sampler.random(n=num_sims)

    # samples_scaled is a np.array with shape (num_samples, sample_dim), in this case (num_samples, 6)
    samples_scaled = qmc.scale(samples_unit, lower_bounds, upper_bounds).astype(np.float32)

    return samples_scaled


def build_sim_params(a: float, b: float, grid: np.ndarray, num_sims: int, rng, lhs_seed: int = 0) -> list:

    samples_scaled = generate_lhs_samples(num_sims=num_sims, seed=lhs_seed)

    # -------- EXTRACT PARAMS FROM LHS SAMPLES --------
    amplitudes   = samples_scaled[:, 0]
    frequencies  = samples_scaled[:, 1]
    k1_vals      = samples_scaled[:, 2]
    k2_vals      = samples_scaled[:, 3]
    rho_cp1_vals = samples_scaled[:, 4]
    rho_cp2_vals = samples_scaled[:, 5]

    T0_list = []

    for _ in range(num_sims):
        T0 = random_ic(a=a, b=b, grid=grid, rng=rng)
        T0_list.append(T0)

    # shape: [(a0, f0, k1_0, k2_0, rcp1_0, rcp2_0, (Nx,)), ...]
    sim_params = list(zip(amplitudes, frequencies, k1_vals, k2_vals, rho_cp1_vals, rho_cp2_vals, T0_list))

    return sim_params


def generate_sim_data(num_sims: int = 2500) -> None:
    # seed 0 for reproducibility after I generate all simulations
    rng = np.random.default_rng(0)

    # fixed grid geometry and time parameters (material properties vary per sim)
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

    print("Building simulation parameters.")

    # shape: [(a0, f0, k1_0, k2_0, rcp1_0, rcp2_0, (Nx,)), ...]
    sim_params = build_sim_params(a=a, b=b, grid=grid, num_sims=num_sims, rng=rng, lhs_seed=0)

    trajectories = np.zeros((num_sims, Nt, Nx), dtype=np.float32) # (num_sims, Nt, Nx)

    # enumerate sim_params to yield (index, element) and use tuple unpacking
    for i, (amp, freq, k1, k2, rcp1, rcp2, T0) in enumerate(sim_params):
        # construct per-sim layers with sampled material properties
        # rho = rho_cp (since cp=1), so rho*cp = rho_cp
        layers = [
            Layer1D(x_left=0.0, x_right=0.5, rho=float(rcp1), cp=1.0, k=float(k1)),
            Layer1D(x_left=0.5, x_right=1.0, rho=float(rcp2), cp=1.0, k=float(k2)),
        ]

        sim = FDSolver1D(
            a=a, b=b, N=N, lam_target=0.8, layers=layers,
            t_final=t_final, flux_f=float(freq), flux_A=float(amp),
            t_on=t_on, t_off=t_off, phase=phase, dt=dt, tukey_alpha=tukey_alpha,
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
    generate_sim_data(num_sims=2500)
