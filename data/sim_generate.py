import numpy as np
from physics.fd_solver_1d import FDSolver1D

# data consists of temp at time_final

# Input: X[i, :, 0] = T0, X[i, :, 1] = x, that is the initial condition and the spatial points

# I want to take the linear combination of the initial condition because thats all i'm going to vary

def random_ic(a: float, b: float, grid: np.ndarray, rng) -> np.ndarray:
    L = a - b

    # random linear combination, simple but random
    c0 = rng.uniform(0.5, 1.5)
    c1 = rng.uniform(-0.3, 0.3)
    c2 = rng.uniform(-0.3, 0.3)

    return (
        c0 * np.cos((np.pi * grid)/(2.0*L)) + c1 * np.sin((np.pi * grid)/L) + c2 * np.cos((2*np.pi*grid)/L)
    )

# create main function that will generate the number of simulations for training and populate x_data & y_data data sets
def main():
    # seed 0 for repoduciability after I generate all simulations
    rng = np.random.default_rng(0)

    # make sure to generate the same # of simulations as train/val/test split in fno
    n_train, n_val, n_test = 128, 32, 256
    n_samples = n_train + n_val + n_test


    # fixed PDE + BCs parameters (so mapping is learnable from u0)
    sim = FDSolver1D(
        a=0.0,
        b=1.0,
        N=128,
        cp=1,
        rho=1,
        k=1,
        lam_target=0.8,
        t_final=0.2,
        flux_f=2.0,
        flux_A=1.0,
        t_on=0.0,
        t_off=0.2,
        phase=0.0
    )
    N = sim.N

    X = np.zeros((n_samples, N, 2), dtype=float) # (# of sims, # of grid points, (x0, x))
    Y = np.zeros((n_samples, N, 1), dtype=float) # (# of sims, # of grid points, T_final)

    for i in range(n_samples):
        T0 = random_ic(a=sim.a, b=sim.b, grid=sim.grid, rng=rng)
        t, x, T_final = sim.solve(T0=T0, store_history=False)

        X[i, :, 0] = T0.astype(np.float32)
        X[i, :, 1] = x.astype(np.float32)
        Y[i, :, 0] = T_final.astype(np.float32)

    np.save("x_data.npy", X)
    np.save("y_data.npy", Y)
    print("Saved:", X.shape, Y.shape)

if __name__ == '__main__':
    main()