import copy
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

from src.physics.boundary_forcing import (
    SPATIAL_FAMILIES,
    SPATIAL_BUILDERS,
    PATCH_W_RANGE,
    GAUSS_SIGMA_RANGE,
    TRIANGLE_ELL_RANGE,
    TEMPORAL_FAMILIES,
    TEMPORAL_FAMILY_ORDER,
    PULSE_SLOTS,
    FORCING_BINS,
    SIN_AMP_RANGE,
    encode_temporal_params,
    integrate_temporal_bins,
)

RC_RANGE = (0.05, 1.0)
T_EPS = 1e-6  # epsilon for temperature normalization

# Conditioning-vector layout (28 dims):
#   [0:3]    base:           t_bar_norm, t_s_norm, R_c_norm
#   [3:7]    spatial onehot: uniform, patch, gaussian, triangle
#   [7:11]   spatial params: y_c_norm, w_norm, sigma_y_norm, ell_norm
#   [11:15]  temporal onehot: sin, exp, pulse_train, exp_train
#   [15:28]  temporal params (1 + PULSE_SLOTS*3 = 13 dims), family-meaning
SPATIAL_FAMILY_ORDER = ("uniform", "patch", "gaussian", "triangle")

_BASE_DIM            = 3
_SPATIAL_ONEHOT_DIM  = len(SPATIAL_FAMILY_ORDER)
_SPATIAL_PARAM_DIM   = 4
_TEMPORAL_ONEHOT_DIM = len(TEMPORAL_FAMILY_ORDER)
_TEMPORAL_PARAM_DIM  = 1 + PULSE_SLOTS * 3
COND_DIM = (_BASE_DIM + _SPATIAL_ONEHOT_DIM + _SPATIAL_PARAM_DIM
            + _TEMPORAL_ONEHOT_DIM + _TEMPORAL_PARAM_DIM)

# log-uniform sigma_y is min-max normalized in log-space so coverage matches
# the sampler's log-uniform distribution.
_LOG_SIGMA_LO = float(np.log(GAUSS_SIGMA_RANGE[0]))
_LOG_SIGMA_HI = float(np.log(GAUSS_SIGMA_RANGE[1]))


def build_cond_vector(t_bar_norm: float, t_s_norm: float, R_c: float,
                      spatial_family: str, spatial_params: dict,
                      temporal_family: str, temporal_params: dict,
                      dt: float, t_final: float) -> np.ndarray:
    """Assemble the 28-dim conditioning vector. Used by both the dataset and
    the inference plotting paths so they cannot drift apart."""
    R_c_norm = (R_c - RC_RANGE[0]) / (RC_RANGE[1] - RC_RANGE[0])
    base = np.array([t_bar_norm, t_s_norm, R_c_norm], dtype=np.float32)

    spatial_oh = np.zeros(_SPATIAL_ONEHOT_DIM, dtype=np.float32)
    spatial_oh[SPATIAL_FAMILY_ORDER.index(spatial_family)] = 1.0
    y_c_norm = w_norm = sigma_y_norm = ell_norm = 0.0
    if spatial_family == "patch":
        y_c_norm = float(spatial_params["y_c"])
        w_norm = (spatial_params["w"] - PATCH_W_RANGE[0]) / (PATCH_W_RANGE[1] - PATCH_W_RANGE[0])
    elif spatial_family == "gaussian":
        y_c_norm = float(spatial_params["y_c"])
        sigma_y_norm = (np.log(spatial_params["sigma_y"]) - _LOG_SIGMA_LO) / (_LOG_SIGMA_HI - _LOG_SIGMA_LO)
    elif spatial_family == "triangle":
        y_c_norm = float(spatial_params["y_c"])
        ell_norm = (spatial_params["ell"] - TRIANGLE_ELL_RANGE[0]) / (TRIANGLE_ELL_RANGE[1] - TRIANGLE_ELL_RANGE[0])
    spatial_p = np.array([y_c_norm, w_norm, sigma_y_norm, ell_norm], dtype=np.float32)

    temporal_oh = np.zeros(_TEMPORAL_ONEHOT_DIM, dtype=np.float32)
    temporal_oh[TEMPORAL_FAMILY_ORDER.index(temporal_family)] = 1.0
    temporal_p = encode_temporal_params(
        temporal_family, temporal_params, dt, t_final
    ).astype(np.float32)

    return np.concatenate([base, spatial_oh, spatial_p, temporal_oh, temporal_p]).astype(np.float32)

# --------- SNAPSHOT PAIR DATASET ---------

class SnapshotPairDataset(Dataset):
    """All-to-all snapshot-pair dataset for time-conditioned FNO.

    Each sample is a (input, target) pair: given the temperature field at
    time t_s, predict the field at a future time t_j > t_s.

    Uses **global normalization**: T̃ = (T - μ_global) / σ_global, where
    μ_global and σ_global are computed once from the training set. This
    preserves absolute temperature scale in the field (unlike per-sample
    z-score) so the model can learn BC- and IC-dependent dynamics.

    When n_snapshots is provided, uniformly subsamples that many time steps
    from the full trajectory and enumerates all possible pairs per
    simulation.  Pairs are sorted by lead time to support curriculum slicing.

    Returns 4-tuple: (spatial, cond, Y, T_stats)
        spatial : (Nx, Ny, 4 + FORCING_BINS)  — [T̃_source, x_norm, y_norm, s_y,
                                   Q_y_bin_0, ..., Q_y_bin_{FORCING_BINS-1}]
        cond      : (28,)    — see COND_DIM layout above
        Y         : (Nx, Ny, 1)  — T̃_target (globally normalized)
        T_stats   : (2,)     — [μ_global, σ_global] for denormalization
    """

    def __init__(
        self,
        trajectories: np.ndarray,
        t_grid: np.ndarray,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        sim_ids: np.ndarray,
        sim_params: np.ndarray,
        mu_global: float,
        sigma_global: float,
        n_snapshots: int | None = None,
        noise_std: float = 0.0,
        dt: float | None = None,
        t_final: float | None = None,
    ):
        self.trajectories = trajectories  # (num_sims, Nt, Nx, Ny)
        self.sim_params = sim_params
        self.t_grid = t_grid.astype(np.float32)
        self.x_grid = x_grid.astype(np.float32)
        self.y_grid = y_grid.astype(np.float32)
        self.sim_ids = sim_ids.astype(np.int64)
        self.mu_global = np.float32(mu_global)
        self.sigma_global = np.float32(sigma_global)
        self.noise_std = noise_std

        self.num_sims, self.Nt, self.Nx, self.Ny = trajectories.shape

        # dt and t_final are needed by encode_temporal_params for log-space
        # normalization of tau / dt_n. Defaults derived from t_grid match the
        # generation pipeline whenever the saved t_grid is uniform.
        self.t_final = float(t_final) if t_final is not None else float(self.t_grid[-1])
        self.dt = float(dt) if dt is not None else float(self.t_grid[1] - self.t_grid[0])
        self.q_ref = np.float32(SIN_AMP_RANGE[1] * self.t_final / FORCING_BINS)

        # Normalized spatial coordinates (fixed for all samples)
        self.x_norm = (
            (self.x_grid - self.x_grid[0]) / (self.x_grid[-1] - self.x_grid[0])
        ).astype(np.float32)

        self.y_norm = (
            (self.y_grid - self.y_grid[0]) / (self.y_grid[-1] - self.y_grid[0])
        ).astype(np.float32)

        # Broadcast to (Nx, Ny) so they can be stacked with T_source at sample time
        self.X_norm = np.broadcast_to(self.x_norm[:, None], (self.Nx, self.Ny)).astype(np.float32)
        self.Y_norm = np.broadcast_to(self.y_norm[None, :], (self.Nx, self.Ny)).astype(np.float32)
        self.s_y_profiles = self._build_spatial_profiles()

        # Determine which time indices to use
        if n_snapshots is not None and n_snapshots < self.Nt:
            self.t_indices = np.round(
                np.linspace(0, self.Nt - 1, n_snapshots)
            ).astype(int)
        else:
            self.t_indices = np.arange(self.Nt)

        # Build ALL valid (sim_id, s_idx, j_idx) pairs, sorted by lead time
        self._build_all_pairs()
        # active pairs represents the amount of pairs actually seen during a certain epoch
        self._active_len = len(self._pairs)

    def _build_all_pairs(self):
        """Enumerate all valid (sim_id, s, j) pairs from subsampled time indices."""
        pairs = []
        for sim_id in self.sim_ids:
            for i, s_idx in enumerate(self.t_indices):
                for j_idx in self.t_indices[i + 1:]:
                    lead = float(self.t_grid[j_idx] - self.t_grid[s_idx])
                    pairs.append((int(sim_id), int(s_idx), int(j_idx), lead))
        # Sort by lead time for curriculum slicing
        pairs.sort(key=lambda p: p[3])
        self._pairs = [(p[0], p[1], p[2]) for p in pairs]
        self._lead_times = np.array([p[3] for p in pairs], dtype=np.float32)

    def _build_spatial_profiles(self) -> dict[int, np.ndarray]:
        profiles = {}
        for sim_id in self.sim_ids:
            params = self.sim_params[int(sim_id)]
            s_vec = SPATIAL_BUILDERS[params["spatial_family"]](
                self.y_grid, **params["spatial_params"]
            )
            profiles[int(sim_id)] = np.asarray(s_vec, dtype=np.float32)
        return profiles

    def set_curriculum_fraction(self, frac: float):
        """Expose only pairs with lead time <= frac * max_lead_time.
        frac=1.0 means all pairs (no curriculum restriction)."""
        if frac >= 1.0:
            self._active_len = len(self._pairs)
        else:
            max_lead = self._lead_times[-1]
            cutoff = frac * max_lead
            self._active_len = max(1, int(np.searchsorted(self._lead_times, cutoff, side='right')))

    def __len__(self):
        return self._active_len

    def __getitem__(self, idx):
        sim_id, s, j = self._pairs[idx]

        params = self.sim_params[sim_id]
        R_c = np.float32(params["R_c"])
        spatial_family = params["spatial_family"]
        spatial_params = params["spatial_params"]
        temporal_family = params["temporal_family"]
        temporal_params = params["temporal_params"]

        T_source = self.trajectories[sim_id, s, :, :]  # (Nx, Ny)
        T_target = self.trajectories[sim_id, j, :, :]  # (Nx, Ny)

        T_source_norm = (T_source - self.mu_global) / (self.sigma_global + T_EPS)
        T_target_norm = (T_target - self.mu_global) / (self.sigma_global + T_EPS)

        if self.noise_std > 0:
            T_source_norm = T_source_norm + np.random.randn(*T_source_norm.shape).astype(np.float32) * self.noise_std

        t_bar_norm = (self.t_grid[j] - self.t_grid[s]) / self.t_grid[-1]
        t_s_norm = self.t_grid[s] / self.t_grid[-1]
        bins = integrate_temporal_bins(
            temporal_family,
            temporal_params,
            float(self.t_grid[s]),
            float(self.t_grid[j]),
            K=FORCING_BINS,
        ).astype(np.float32)

        s_y = self.s_y_profiles[sim_id]
        S_y = np.broadcast_to(s_y[None, :], (self.Nx, self.Ny))
        Q_y_bins = (s_y[None, :, None] * bins[None, None, :] / self.q_ref).astype(np.float32)
        Q_y_bins_2d = np.broadcast_to(Q_y_bins, (self.Nx, self.Ny, FORCING_BINS))
        spatial_base = np.stack([T_source_norm, self.X_norm, self.Y_norm, S_y], axis=-1)
        spatial = np.concatenate([spatial_base, Q_y_bins_2d], axis=-1).astype(np.float32)
        cond = build_cond_vector(
            t_bar_norm=float(t_bar_norm), t_s_norm=float(t_s_norm), R_c=float(R_c),
            spatial_family=spatial_family, spatial_params=spatial_params,
            temporal_family=temporal_family, temporal_params=temporal_params,
            dt=self.dt, t_final=self.t_final,
        )

        # Target: (Nx, Ny, 1)
        Y = T_target_norm[:, :, None].astype(np.float32)

        # Stats for denormalization at eval time: (2,) — global, constant across samples
        T_stats = np.array([self.mu_global, self.sigma_global], dtype=np.float32)

        return (
            torch.from_numpy(spatial),
            torch.from_numpy(cond),
            torch.from_numpy(Y),
            torch.from_numpy(T_stats),
        )


def _dataset_with_pairs(dataset: SnapshotPairDataset, pairs: list[tuple[int, int, int]]) -> SnapshotPairDataset:
    out = copy.copy(dataset)
    out._pairs = list(pairs)
    out._lead_times = np.array(
        [float(out.t_grid[j] - out.t_grid[s]) for _, s, j in out._pairs],
        dtype=np.float32,
    )
    out._active_len = len(out._pairs)
    return out


def split_pairs_within_sims(
    dataset: SnapshotPairDataset,
    val_pair_frac: float = 0.1,
    seed: int = 0,
) -> tuple[Dataset, Dataset]:
    rng = np.random.default_rng(seed)
    val_indices = set()
    for sim_id in dataset.sim_ids:
        sim_pair_indices = [i for i, pair in enumerate(dataset._pairs) if pair[0] == int(sim_id)]
        n_val = int(round(len(sim_pair_indices) * val_pair_frac))
        if val_pair_frac > 0.0 and len(sim_pair_indices) > 0:
            n_val = max(1, n_val)
        if n_val > 0:
            chosen = rng.choice(sim_pair_indices, size=n_val, replace=False)
            val_indices.update(int(i) for i in chosen)

    train_pairs = [pair for i, pair in enumerate(dataset._pairs) if i not in val_indices]
    val_pairs = [pair for i, pair in enumerate(dataset._pairs) if i in val_indices]
    train_dataset = _dataset_with_pairs(dataset, train_pairs)
    val_dataset = _dataset_with_pairs(dataset, val_pairs)
    val_dataset.noise_std = 0.0
    return train_dataset, val_dataset


# --------- LOAD RAW SIM. DATA --------

def load_sim_data(
    sim_traj_path: str,
    x_grid_path: str,
    y_grid_path: str,
    t_grid_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    trajectories = np.load(sim_traj_path)  # (num_sims, Nt, Nx, Ny)
    x_grid = np.load(x_grid_path) # (Nx,)
    y_grid = np.load(y_grid_path) # (Ny,)
    t_grid = np.load(t_grid_path) # (Nt,)

    # size safeguards
    if trajectories.ndim != 4:
        raise ValueError(f"Expected trajectories with ndim=4, got shape {trajectories.shape}")
    if x_grid.ndim != 1:
        raise ValueError(f"Expected x_grid with ndim=1, got shape {x_grid.shape}")
    if y_grid.ndim != 1:
        raise ValueError(f"Expected y_grid with ndim=1, got shape {y_grid.shape}")
    if t_grid.ndim != 1:
        raise ValueError(f"Expected t_grid with ndim=1, got shape {t_grid.shape}")

    # length safeguards
    n_sims, Nt_total, Nx, Ny = trajectories.shape
    if x_grid.shape[0] != Nx:
        raise ValueError(f"x_grid length {x_grid.shape[0]} does not match Nx={Nx}")
    if y_grid.shape[0] != Ny:
        raise ValueError(f"y_grid length {y_grid.shape[0]} does not match Ny={Ny}")
    if t_grid.shape[0] != Nt_total:
        raise ValueError(f"t_grid length {t_grid.shape[0]} does not match Nt_total={Nt_total}")

    return trajectories, x_grid, y_grid, t_grid


def load_solver_dt(t_grid_path: str | Path) -> float | None:
    """Return the solver dt saved alongside the trajectories, or None if absent.

    Looks for `dt.npy` in the same directory as `t_grid_path`. Needed because
    `t_grid[1] - t_grid[0]` is the saved snapshot cadence (= solver dt × save_stride),
    not the solver dt that the boundary-forcing samplers used to set tau / dt_n
    bounds. Mismatched dt produces cond-vec slots outside [0, 1].
    """
    dt_path = Path(t_grid_path).parent / "dt.npy"
    if not dt_path.exists():
        return None
    return float(np.load(dt_path))


# ------- SLICE ALL SIMS INTO TRAIN/VAL/TEST SPLITS -------

def split_sim_ids(
    num_sims: int,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    ids = np.arange(num_sims)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)

    n_train = int(num_sims * train_frac)
    n_val = int(num_sims * val_frac)
    n_test = num_sims - n_train - n_val

    train_ids = ids[:n_train]
    val_ids = ids[n_train:n_train + n_val]
    test_ids = ids[n_train + n_val:n_train + n_val + n_test]

    return train_ids, val_ids, test_ids


# ------- GLOBAL NORMALIZATION STATISTICS -------

def compute_global_stats(
    trajectories: np.ndarray,
    train_ids: np.ndarray,
) -> tuple[float, float]:
    """Compute global mean/std from training simulations only (avoids data leakage).

    Returns (mu_global, sigma_global) as Python floats.
    """
    train_data = trajectories[train_ids]  # (N_train, Nt, Nx, Ny)
    mu_global = float(train_data.mean())
    sigma_global = float(train_data.std())
    return mu_global, sigma_global


# ------- CREATE DATALOADERS -------

def create_dataloaders(
    trajectories: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    t_grid: np.ndarray,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    test_ids: np.ndarray,
    batch_size: int,
    sim_params: np.ndarray,
    mu_global: float,
    sigma_global: float,
    n_snapshots: int = 15,
    n_snapshots_test: int | None = None,
    noise_std: float = 0.0,
    num_workers: int | None = None,
    dt: float | None = None,
    t_final: float | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:

    n_test = n_snapshots_test if n_snapshots_test is not None else n_snapshots

    train_dataset = SnapshotPairDataset(
        trajectories=trajectories,
        x_grid=x_grid,
        y_grid=y_grid,
        t_grid=t_grid,
        sim_ids=train_ids,
        sim_params=sim_params,
        mu_global=mu_global,
        sigma_global=sigma_global,
        n_snapshots=n_snapshots,
        noise_std=noise_std,
        dt=dt,
        t_final=t_final,
    )

    val_dataset = SnapshotPairDataset(
        trajectories=trajectories,
        x_grid=x_grid,
        y_grid=y_grid,
        t_grid=t_grid,
        sim_ids=val_ids,
        sim_params=sim_params,
        mu_global=mu_global,
        sigma_global=sigma_global,
        n_snapshots=n_snapshots,
        dt=dt,
        t_final=t_final,
    )

    test_dataset = SnapshotPairDataset(
        trajectories=trajectories,
        x_grid=x_grid,
        y_grid=y_grid,
        t_grid=t_grid,
        sim_ids=test_ids,
        sim_params=sim_params,
        mu_global=mu_global,
        sigma_global=sigma_global,
        n_snapshots=n_test,
        dt=dt,
        t_final=t_final,
    )

    pin = torch.cuda.is_available()
    workers = num_workers if num_workers is not None else (4 if pin else 0)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, pin_memory=pin, num_workers=workers)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, pin_memory=pin, num_workers=workers)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False, pin_memory=pin, num_workers=workers)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    project_root = Path(__file__).resolve().parents[1]
    trajectories, x_grid, y_grid, t_grid = load_sim_data(
        sim_traj_path=str(project_root / "data" / "trajectories.npy"),
        x_grid_path=str(project_root / "data" / "x_grid.npy"),
        y_grid_path=str(project_root / "data" / "y_grid.npy"),
        t_grid_path=str(project_root / "data" / "t_grid.npy"),
    )
    sim_params = np.load(str(project_root / "data" / "sim_params.npy"), allow_pickle=True)

    train_ids, val_ids, test_ids = split_sim_ids(num_sims=trajectories.shape[0], train_frac=0.7, val_frac=0.15, seed=0)
    mu_global, sigma_global = compute_global_stats(trajectories, train_ids)
    print(f"Global stats: mu={mu_global:.4f}, sigma={sigma_global:.4f}")

    batch_size = 64

    train_loader, val_loader, test_loader = create_dataloaders(
        trajectories=trajectories, x_grid=x_grid, y_grid=y_grid, t_grid=t_grid,
        train_ids=train_ids, val_ids=val_ids, test_ids=test_ids,
        batch_size=batch_size, sim_params=sim_params,
        mu_global=mu_global, sigma_global=sigma_global,
    )

    # Verify shapes
    x_spatial, cond, yb, t_stats = next(iter(train_loader))
    print(f"spatial: {x_spatial.shape}")  # (B, Nx, Ny, 4 + FORCING_BINS)
    print(f"cond: {cond.shape}")            # (B, 28)
    print(f"Y: {yb.shape}")                 # (B, Nx, Ny, 1)
    print(f"T_stats: {t_stats.shape}")      # (B, 2)
