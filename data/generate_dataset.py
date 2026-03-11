import numpy as np
from src.physics.fd_solver_1d import FDSolver1D

# Training samples are NOT entire simulations, they are windows extracted from simulations

# TODO: implement LHS distribution for heat flux, material properties, and ICs?

def random_ic(a: float, b: float, grid: np.ndarray, rng) -> np.ndarray:
    L = b - a

    # random linear combination, simple but random
    c0 = rng.uniform(0.5, 1.5)
    c1 = rng.uniform(-0.3, 0.3)
    c2 = rng.uniform(-0.3, 0.3)

    return (
        c0 * np.cos((np.pi * grid)/(2.0*L)) + c1 * np.sin((np.pi * grid)/L) + c2 * np.cos((2*np.pi*grid)/L)
    )


def main():
    # seed 0 for repoduciability after I generate all simulations
    rng = np.random.default_rng(0)

    # make sure to generate the same # of simulations as train/val/test split in fno
    n_sims = 1000

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
        dt=.005
    )
    Nx = sim.N
    Nt = len(sim.t)

    trajectories = np.zeros((n_sims, Nt, Nx), dtype=np.float32) # (num_sims, Nt, Nx)

    for i in range(n_sims):
        T0 = random_ic(a=sim.a, b=sim.b, grid=sim.grid, rng=rng)
        t, x, T_hist = sim.solve(T0=T0, store_trajectory=True)

        trajectories[i, :, :] = T_hist.astype(np.float32)
        print(f"Finished {i} simulation with trajectories shape {trajectories.shape}.")

    # x and t are the same for all simulations, so just use the last ones
    x_grid = x.astype(np.float32)
    t_grid = t.astype(np.float32)

    np.save("x_grid.npy", x_grid)
    np.save("t_grid.npy", t_grid)
    np.save("trajectories.npy", trajectories)
    print("Saved:", x_grid.shape, t_grid.shape, trajectories.shape)

if __name__ == '__main__':
    main()