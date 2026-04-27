import numpy as np
import pytest
import torch

from src.physics.fv_solver_1d import FVSolver1D, Layer1D
from src.operators.fno1d import FNO1d
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
    return FVSolver1D(N=11, **SOLVER_DEFAULTS)


@pytest.fixture
def default_solver():
    """N=101 solver for accuracy tests."""
    return FVSolver1D(N=101, **SOLVER_DEFAULTS)


# ---------- synthetic data fixtures ----------

@pytest.fixture
def synthetic_trajectories():
    """Small synthetic 2D trajectories (20 sims, 51 timesteps, 11x11 spatial grid)."""
    rng = np.random.default_rng(0)
    num_sims, Nt, Nx, Ny = 20, 51, 11, 11
    trajectories = rng.standard_normal((num_sims, Nt, Nx, Ny)).astype(np.float32)
    x_grid = np.linspace(0.0, 1.0, Nx).astype(np.float32)
    y_grid = np.linspace(0.0, 1.0, Ny).astype(np.float32)
    t_grid = np.linspace(0.0, 1.0, Nt).astype(np.float32)
    return trajectories, x_grid, y_grid, t_grid


@pytest.fixture
def synthetic_sim_params(synthetic_trajectories):
    """Synthetic sim_params matching synthetic_trajectories (20 sims, dict schema).

    Uses the real per-family samplers so the temporal_params dict shape always
    matches what TEMPORAL_BUILDERS / encode_temporal_params expect.
    """
    from src.physics.boundary_forcing import (
        SPATIAL_SAMPLERS,
        TEMPORAL_SAMPLERS,
        sample_spatial_family,
        sample_temporal_family,
    )

    trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
    num_sims = trajectories.shape[0]
    Nx = x_grid.shape[0]
    Ny = y_grid.shape[0]
    dt = float(t_grid[1] - t_grid[0])
    t_final = float(t_grid[-1])
    rng = np.random.default_rng(42)
    params = []
    for _ in range(num_sims):
        R_c = float(rng.uniform(0.05, 1.0))
        T0 = rng.standard_normal((Nx, Ny)).astype(np.float32)
        temporal_family = sample_temporal_family(rng)
        temporal_params = TEMPORAL_SAMPLERS[temporal_family](
            rng, dt=dt, t_final=t_final, t_on=0.0, t_off=0.2,
        )
        spatial_family = sample_spatial_family(rng)
        spatial_params = SPATIAL_SAMPLERS[spatial_family](rng)
        params.append({
            "R_c": R_c,
            "T0": T0,
            "temporal_family": temporal_family,
            "temporal_params": temporal_params,
            "spatial_family": spatial_family,
            "spatial_params": spatial_params,
        })
    return np.array(params, dtype=object)


@pytest.fixture
def tmp_npy_data(tmp_path, synthetic_trajectories):
    """Save synthetic data as .npy files and return paths."""
    trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
    traj_path = tmp_path / "trajectories.npy"
    x_path = tmp_path / "x_grid.npy"
    y_path = tmp_path / "y_grid.npy"
    t_path = tmp_path / "t_grid.npy"
    np.save(traj_path, trajectories)
    np.save(x_path, x_grid)
    np.save(y_path, y_grid)
    np.save(t_path, t_grid)
    return str(traj_path), str(x_path), str(y_path), str(t_path)


# ---------- model fixture ----------

@pytest.fixture
def small_fno():
    """Tiny FNO1d for fast tests."""
    return FNO1d(modes=2, width=8, in_channels=2, out_channels=1, n_layers=2, cond_dim=5)


@pytest.fixture
def small_fno2d():
    """Tiny FNO2d for fast plot tests."""
    return FNO2d(
        modes1=2,
        modes2=2,
        width=8,
        in_channels=3,
        out_channels=1,
        n_layers=2,
        cond_dim=28,
    )


@pytest.fixture
def small_fno2d_checkpoint(tmp_path, small_fno2d):
    """Save a minimal 2D checkpoint for plot-loader smoke tests."""
    ckpt_path = tmp_path / "small_fno2d.pt"
    torch.save(
        {
            "model_state": small_fno2d.state_dict(),
            "conf": {
                "model": {
                    "parameters": {
                        "modes1": 2,
                        "modes2": 2,
                        "width": 8,
                        "in_channels": 3,
                        "out_channels": 1,
                        "n_layers": 2,
                        "cond_dim": 28,
                        "cond_hidden": 256,
                        "dropout": 0.0,
                        "spectral_dropout": 0.0,
                    }
                }
            },
            "mu_global": 0.0,
            "sigma_global": 1.0,
        },
        ckpt_path,
    )
    return ckpt_path


# ---------- conditioning fixture ----------

@pytest.fixture
def synthetic_cond():
    """Random conditioning vectors (4, 5) for testing."""
    return torch.rand(4, 5)
