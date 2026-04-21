from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

# --------- NORMALIZATION CONSTANTS ---------

RC_RANGE = (0.05, 1.0)
AMP_RANGE = (50.0, 300.0)
FREQ_RANGE = (1.0, 20.0)
T_EPS = 1e-6  # epsilon for temperature normalization

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
        spatial : (Nx, Ny, 3)  — [T̃_source, x_norm, y_norm]
        cond      : (5,)     — [t̄_norm, t_s_norm, A_norm, f_norm, R_c_norm]
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

        # Unpack sim params: (amp, freq, T0, R_c)
        amp, freq, _T0, R_c = self.sim_params[sim_id]
        amp = np.float32(amp)
        freq = np.float32(freq)

        # Source and target snapshots
        T_source = self.trajectories[sim_id, s, :, :]  # (Nx, Ny)
        T_target = self.trajectories[sim_id, j, :, :]  # (Nx, Ny)

        # Global temperature normalization (training-set statistics)
        T_source_norm = (T_source - self.mu_global) / (self.sigma_global + T_EPS)
        T_target_norm = (T_target - self.mu_global) / (self.sigma_global + T_EPS)

        # Input noise augmentation (training only, controlled by noise_std)
        if self.noise_std > 0:
            T_source_norm = T_source_norm + np.random.randn(*T_source_norm.shape).astype(np.float32) * self.noise_std

        # Spatial input: (Nx, Ny, 3) — [T̃_source, x_norm, y_norm]
        spatial = np.stack([T_source_norm, self.X_norm, self.Y_norm], axis=-1).astype(np.float32)

        # Conditioning vector: (5,) — [t̄_norm, t_s_norm, A_norm, f_norm, R_c_norm]
        t_bar = self.t_grid[j] - self.t_grid[s]
        t_bar_norm = t_bar / self.t_grid[-1]
        t_s_norm = self.t_grid[s] / self.t_grid[-1]
        A_norm = (amp - AMP_RANGE[0]) / (AMP_RANGE[1] - AMP_RANGE[0])
        f_norm = (freq - FREQ_RANGE[0]) / (FREQ_RANGE[1] - FREQ_RANGE[0])
        R_c_norm = (R_c - RC_RANGE[0]) / (RC_RANGE[1] - RC_RANGE[0])
        cond = np.array([t_bar_norm, t_s_norm, A_norm, f_norm, R_c_norm], dtype=np.float32)

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
    print(f"spatial: {x_spatial.shape}")  # (B, Nx, Ny, 3)
    print(f"cond: {cond.shape}")            # (B, 5)
    print(f"Y: {yb.shape}")                 # (B, Nx, Ny, 1)
    print(f"T_stats: {t_stats.shape}")      # (B, 2)
