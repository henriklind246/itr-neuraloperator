import numpy as np
from src.physics.fd_solver_1d import windowed_sin_flux
from src.physics.fd_solver_1d import FDSolver1D
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
        "frequency": (1.0, 20.0)
    }

    # create list of parameter names to help calculate sample dim. and lower & upper bounds
    param_names = list(param_ranges.keys())
    sample_dim = len(param_names)
    lower_bounds = np.array([param_ranges[name][0] for name in param_names], dtype=np.float32)
    upper_bounds = np.array([param_ranges[name][1] for name in param_names], dtype=np.float32)

    # ------- GENERATE LHS SAMPLES -------
    sampler = qmc.LatinHypercube(d=sample_dim, seed=seed)
    samples_unit = sampler.random(n=num_sims)

    # samples_scaled is a np.array with shape (num_samples, sample_dim), in this case (num_samples, 2)
    samples_scaled = qmc.scale(samples_unit, lower_bounds, upper_bounds).astype(np.float32)

    return samples_scaled


def build_sim_params(sim: FDSolver1D, num_sims: int, rng, lhs_seed: int = 0) -> list:

    samples_scaled = generate_lhs_samples(num_sims=num_sims, seed=lhs_seed)

    # -------- EXTRACT PARAMS FROM LHS SAMPLES --------
    amplitudes = samples_scaled[:, 0]
    frequencies = samples_scaled[:, 1]

    T0_list = []

    for _ in range(num_sims):
        T0 = random_ic(a=sim.a, b=sim.b, grid=sim.grid, rng=rng)
        T0_list.append(T0)

    # shape: [(a0, f0, (Nx0,)), (a1, f1, (Nx1,)), ...]
    sim_params = list(zip(amplitudes, frequencies, T0_list))

    return sim_params


def generate_sim_data(num_sims: int = 1000) -> None:
    # seed 0 for repoduciability after I generate all simulations
    rng = np.random.default_rng(0)

    # fixed PDE + BCs parameters + Nt = 200 (so mapping is learnable)
    sim = FDSolver1D(
        a=0.0,
        b=1.0,
        N=101,
        cp=1,
        rho=1,
        k=1,
        lam_target=0.8,
        t_final=1.0,
        flux_f=2.0,
        flux_A=1.0,
        t_on=0.0,
        t_off=0.2,
        phase=0.0,
        dt=.005,
        tukey_alpha=0.5
    )
    Nx = sim.N
    Nt = len(sim.t)

    print("Building simulation parameters.")

    # shape: [(a0, f0, (Nx0,), (a1, f1, (Nx1,)), ...]
    sim_params = build_sim_params(sim=sim, num_sims=num_sims, rng=rng, lhs_seed=0)

    trajectories = np.zeros((num_sims, Nt, Nx), dtype=np.float32) # (num_sims, Nt, Nx)

    # enumerate sim_params to yield (index, element) and use tuple unpacking
    for i, (amp, freq, T0) in enumerate(sim_params):
        sim.q_left = windowed_sin_flux(float(freq), float(amp), sim.t_on, sim.t_off, sim.phase, sim.tukey_alpha)

        t, x, T_hist = sim.solve(T0=T0, store_trajectory=True)

        trajectories[i, :, :] = T_hist.astype(np.float32)
        print(f"Finished {i} simulation with trajectories shape {trajectories.shape}.")

    # x and t are the same for all simulations, so just use the last ones
    x_grid = x.astype(np.float32)
    t_grid = t.astype(np.float32)

    np.save("x_grid.npy", x_grid)
    np.save("t_grid.npy", t_grid)
    np.save("trajectories.npy", trajectories)
    np.save("sim_params.npy", np.array(sim_params, dtype=object), allow_pickle=True)
    print("Saved:", x_grid.shape, t_grid.shape, trajectories.shape)

if __name__ == '__main__':
    generate_sim_data(num_sims=1000)