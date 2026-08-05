import numpy as np
import pytest
import torch

from src.physics.fv_solver_1d import FVSolver1D, Layer1D
from src.operators.fno1d import FNO1d
from src.operators.fno2d import FNO2d

# Forcing benchmark, temporal_encoder representation (dataset/model default).
FORCING_IN_CHANNELS = 4
FORCING_COND_STATIC_DIM = 10
FORCING_TEMPORAL_TOKEN_DIM = 2
FORCING_TEMPORAL_SAMPLES = 128

# Source benchmark, bins representation (temporal encoder off, no A_norm leak).
SOURCE_IN_CHANNELS = 20
SOURCE_COND_STATIC_DIM = 6


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
        T0 = np.full((Nx, Ny), 300.0, dtype=np.float32)
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
            "ic_family": "uniform_2d",
            "ic_params": {"T0_offset": 0.0},
        })
    return np.array(params, dtype=object)


@pytest.fixture
def synthetic_source_sim_params(synthetic_trajectories):
    """Synthetic sim_params matching the source benchmark schema (20 sims).

    Mirrors problems/source.py: rectangular patch params, fixed interface at
    x = 0.5, no temporal/spatial forcing keys, plus a T0 IC field for plotting.
    """
    from problems.source import INTERFACE_X, _classify_regime
    from src.physics.internal_source import (
        PATCH_A_RANGE,
        PATCH_H,
        PATCH_W,
        PATCH_X_RANGE,
        PATCH_Y_RANGE,
    )

    trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
    num_sims = trajectories.shape[0]
    Nx = x_grid.shape[0]
    Ny = y_grid.shape[0]
    t_final = float(t_grid[-1])
    t_off = 0.75 * t_final
    A_min, A_max = PATCH_A_RANGE
    rng = np.random.default_rng(7)
    params = []
    for _ in range(num_sims):
        x_h = float(rng.uniform(*PATCH_X_RANGE))
        y_h = float(rng.uniform(*PATCH_Y_RANGE))
        A = float(np.exp(rng.uniform(np.log(A_min), np.log(A_max))))
        params.append({
            "R_c": float(rng.uniform(0.05, 1.0)),
            "interface_x": INTERFACE_X,
            "x_h": x_h,
            "y_h": y_h,
            "w_h": PATCH_W,
            "h_h": PATCH_H,
            "A": A,
            "t_off": t_off,
            "regime": _classify_regime(x_h, INTERFACE_X, PATCH_W),
            "T0": np.full((Nx, Ny), 300.0, dtype=np.float32),
            "ic_family": "uniform_2d",
            "ic_params": {"T0_offset": 0.0},
        })
    return np.array(params, dtype=object)


@pytest.fixture
def synthetic_source_itr_sim_params(synthetic_source_sim_params):
    """Synthetic sim_params for source_itr: source params plus the void scalars.

    The R_amp bound is dependent on R_base (as in sample_sim_params) so the
    profile peak stays under R_PEAK_MAX without clipping.
    """
    from src.physics.internal_source import RC_VOID_RANGES, R_PEAK_MAX

    base_lo, base_hi = RC_VOID_RANGES["R_base"]
    y0_lo, y0_hi = RC_VOID_RANGES["y0"]
    sig_lo, sig_hi = RC_VOID_RANGES["sigma"]

    rng = np.random.default_rng(11)
    params = []
    for entry in synthetic_source_sim_params:
        p = dict(entry)
        R_base = float(rng.uniform(base_lo, base_hi))
        p["R_c_base"] = R_base
        p["R_c_amp"] = float(rng.uniform(0.0, 1.0)) * (R_PEAK_MAX - R_base)
        p["R_c_y0"] = float(rng.uniform(y0_lo, y0_hi))
        p["R_c_sigma"] = float(rng.uniform(sig_lo, sig_hi))
        p["R_c"] = R_base
        params.append(p)
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
    """Tiny forcing/temporal_encoder FNO2d for fast plot tests."""
    return FNO2d(
        modes1=2,
        modes2=2,
        width=8,
        in_channels=FORCING_IN_CHANNELS,
        out_channels=1,
        n_layers=2,
        cond_static_dim=FORCING_COND_STATIC_DIM,
        temporal_token_dim=FORCING_TEMPORAL_TOKEN_DIM,
        temporal_hidden=16,
        forcing_embed_dim=16,
        use_forcing_time_aug=True,
        s_y_channel=3,
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
                        "in_channels": FORCING_IN_CHANNELS,
                        "out_channels": 1,
                        "n_layers": 2,
                        "cond_static_dim": FORCING_COND_STATIC_DIM,
                        "cond_hidden": 256,
                        "temporal_token_dim": FORCING_TEMPORAL_TOKEN_DIM,
                        "temporal_samples": FORCING_TEMPORAL_SAMPLES,
                        "temporal_hidden": 16,
                        "forcing_embed_dim": 16,
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


@pytest.fixture
def small_source_fno2d():
    """Tiny source-benchmark FNO2d (bins representation: encoder off, 7 static dims)."""
    return FNO2d(
        modes1=2,
        modes2=2,
        width=8,
        in_channels=SOURCE_IN_CHANNELS,
        out_channels=1,
        n_layers=2,
        cond_static_dim=SOURCE_COND_STATIC_DIM,
        use_temporal_encoder=False,
    )


@pytest.fixture
def small_source_fno2d_checkpoint(tmp_path, small_source_fno2d):
    """Save a minimal source-benchmark checkpoint for plot-loader smoke tests."""
    ckpt_path = tmp_path / "small_source_fno2d.pt"
    torch.save(
        {
            "model_state": small_source_fno2d.state_dict(),
            "conf": {
                "benchmark": {"name": "source", "representation": "bins"},
                "model": {
                    "parameters": {
                        "modes1": 2,
                        "modes2": 2,
                        "width": 8,
                        "in_channels": SOURCE_IN_CHANNELS,
                        "out_channels": 1,
                        "n_layers": 2,
                        "cond_static_dim": SOURCE_COND_STATIC_DIM,
                        "use_temporal_encoder": False,
                        "dropout": 0.0,
                        "spectral_dropout": 0.0,
                    }
                },
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
