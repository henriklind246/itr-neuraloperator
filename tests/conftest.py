import numpy as np
import pytest
import torch

from src.physics.fd_solver_1d import FDSolver1D, Layer1D
from src.operators.fno2d import FNO2d


# ---------- solver fixtures ----------

SINGLE_LAYER = [Layer1D(x_left=0.0, x_right=1.0, rho=1.0, cp=1.0, k=1.0)]

SOLVER_DEFAULTS = dict(
    a=0.0, b=1.0,
    layers=SINGLE_LAYER,
    lam_target=0.5, t_final=0.5,
    flux_f=2.0, flux_A=50.0,
    t_on=0.0, t_off=0.5, phase=0.0,
)


@pytest.fixture
def small_solver():
    """N=11 solver for fast tests."""
    return FDSolver1D(N=11, **SOLVER_DEFAULTS)


@pytest.fixture
def default_solver():
    """N=101 solver for accuracy tests."""
    return FDSolver1D(N=101, **SOLVER_DEFAULTS)


# ---------- synthetic data fixtures ----------

@pytest.fixture
def synthetic_trajectories():
    """Small synthetic trajectories (20 sims, 51 timesteps, 11 spatial nodes)."""
    rng = np.random.default_rng(0)
    num_sims, Nt, Nx = 20, 51, 11
    trajectories = rng.standard_normal((num_sims, Nt, Nx)).astype(np.float32)
    x_grid = np.linspace(0.0, 1.0, Nx).astype(np.float32)
    t_grid = np.linspace(0.0, 1.0, Nt).astype(np.float32)
    return trajectories, x_grid, t_grid


@pytest.fixture
def synthetic_sim_params(synthetic_trajectories):
    """Synthetic sim_params matching synthetic_trajectories (20 sims)."""
    trajectories, x_grid, _ = synthetic_trajectories
    num_sims = trajectories.shape[0]
    Nx = x_grid.shape[0]
    rng = np.random.default_rng(42)
    params = []
    for _ in range(num_sims):
        amp = np.float32(rng.uniform(50.0, 300.0))
        freq = np.float32(rng.uniform(1.0, 20.0))
        k1 = np.float32(rng.uniform(0.5, 5.0))
        k2 = np.float32(rng.uniform(0.5, 5.0))
        rcp1 = np.float32(rng.uniform(0.5, 5.0))
        rcp2 = np.float32(rng.uniform(0.5, 5.0))
        T0 = rng.standard_normal(Nx).astype(np.float32)
        params.append((amp, freq, k1, k2, rcp1, rcp2, T0))
    return np.array(params, dtype=object)


@pytest.fixture
def tmp_npy_data(tmp_path, synthetic_trajectories):
    """Save synthetic data as .npy files and return paths."""
    trajectories, x_grid, t_grid = synthetic_trajectories
    traj_path = tmp_path / "trajectories.npy"
    x_path = tmp_path / "x_grid.npy"
    t_path = tmp_path / "t_grid.npy"
    np.save(traj_path, trajectories)
    np.save(x_path, x_grid)
    np.save(t_path, t_grid)
    return str(traj_path), str(x_path), str(t_path)


# ---------- model fixture ----------

@pytest.fixture
def small_fno():
    """Tiny FNO2d for fast tests."""
    return FNO2d(modes1=2, modes2=2, width=8, in_channels=15)
