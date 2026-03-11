import numpy as np
import pytest
import torch

from src.physics.fd_solver_1d import FDSolver1D
from src.operators.fno2d import FNO2d


# ---------- solver fixtures ----------

SOLVER_DEFAULTS = dict(
    a=0.0, b=1.0, rho=1.0, cp=1.0, k=1.0,
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
    return FNO2d(modes1=2, modes2=2, width=8)
