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

    Each sample is a (source, target) pair: given the temperature field at
    time t_s, predict the field at a future time t_j > t_s.

    Returns 4-tuple: (x_spatial, cond, Y, T_stats)
        x_spatial : (Nx, 2)  — [T̃_source, x_norm]
        cond      : (4,)     — [t̄_norm, A_norm, f_norm, R_c_norm]
        Y         : (Nx, 1)  — T̃_target (normalized)
        T_stats   : (2,)     — [μ_s, σ_s] for denormalization
    """

    def __init__(
        self,
        trajectories: np.ndarray,
        t_grid: np.ndarray,
        x_grid: np.ndarray,
        sim_ids: np.ndarray,
        sim_params: np.ndarray,
        pairs_per_sim: int = 50,
        random_pairs: bool = True,
        seed: int = 0,
        stride: int = 1,
    ):
        self.trajectories = trajectories
        self.sim_params = sim_params
        self.t_grid = t_grid.astype(np.float32)
        self.x_grid = x_grid.astype(np.float32)
        self.sim_ids = sim_ids.astype(np.int64)

        self.pairs_per_sim = pairs_per_sim
        self.random_pairs = random_pairs
        self.stride = stride
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.num_sims, self.Nt, self.Nx = trajectories.shape

        # Normalized spatial coordinates (fixed for all samples)
        self.x_norm = (
            (self.x_grid - self.x_grid[0]) / (self.x_grid[-1] - self.x_grid[0])
        ).astype(np.float32)

        # Pre-compute all valid pairs for deterministic mode
        if not self.random_pairs:
            self._build_deterministic_pairs()

    def _build_deterministic_pairs(self):
        """Enumerate all valid (s, j) pairs at given stride for each sim."""
        pairs = []
        for sim_pos, sim_id in enumerate(self.sim_ids):
            for s in range(0, self.Nt, self.stride):
                for j in range(s + 1, self.Nt, self.stride):
                    pairs.append((int(sim_id), s, j))
        self._det_pairs = pairs

    def __len__(self):
        if self.random_pairs:
            return len(self.sim_ids) * self.pairs_per_sim
        return len(self._det_pairs)

    def __getitem__(self, idx):
        if self.random_pairs:
            sim_id = int(self.sim_ids[idx % len(self.sim_ids)])
            s = int(self.rng.integers(0, self.Nt - 1))
            j = int(self.rng.integers(s + 1, self.Nt))
        else:
            sim_id, s, j = self._det_pairs[idx]

        # Unpack sim params: (amp, freq, T0, R_c)
        amp, freq, _T0, R_c = self.sim_params[sim_id]
        amp = np.float32(amp)
        freq = np.float32(freq)

        # Source and target snapshots
        T_source = self.trajectories[sim_id, s, :]  # (Nx,)
        T_target = self.trajectories[sim_id, j, :]  # (Nx,)

        # Per-sample temperature normalization (source statistics)
        mu_s = T_source.mean()
        sigma_s = T_source.std()
        T_source_norm = (T_source - mu_s) / (sigma_s + T_EPS)
        T_target_norm = (T_target - mu_s) / (sigma_s + T_EPS)

        # Spatial input: (Nx, 2) — [T̃_source, x_norm]
        x_spatial = np.stack([T_source_norm, self.x_norm], axis=-1).astype(np.float32)

        # Conditioning vector: (4,) — [t̄_norm, A_norm, f_norm, R_c_norm]
        t_bar = self.t_grid[j] - self.t_grid[s]
        t_bar_norm = t_bar / self.t_grid[-1]
        A_norm = (amp - AMP_RANGE[0]) / (AMP_RANGE[1] - AMP_RANGE[0])
        f_norm = (freq - FREQ_RANGE[0]) / (FREQ_RANGE[1] - FREQ_RANGE[0])
        R_c_norm = (R_c - RC_RANGE[0]) / (RC_RANGE[1] - RC_RANGE[0])
        cond = np.array([t_bar_norm, A_norm, f_norm, R_c_norm], dtype=np.float32)

        # Target: (Nx, 1)
        Y = T_target_norm[:, None].astype(np.float32)

        # Stats for denormalization at eval time: (2,)
        T_stats = np.array([mu_s, sigma_s], dtype=np.float32)

        return (
            torch.from_numpy(x_spatial),
            torch.from_numpy(cond),
            torch.from_numpy(Y),
            torch.from_numpy(T_stats),
        )


# --------- LOAD RAW SIM. DATA --------

def load_sim_data(
    sim_traj_path: str,
    x_grid_path: str,
    t_grid_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    trajectories = np.load(sim_traj_path)  # (num_sims, Nt, Nx)
    x_grid = np.load(x_grid_path)          # (Nx,)
    t_grid = np.load(t_grid_path)          # (Nt,)

    # size safeguards
    if trajectories.ndim != 3:
        raise ValueError(f"Expected trajectories with ndim=3, got shape {trajectories.shape}")
    if x_grid.ndim != 1:
        raise ValueError(f"Expected x_grid with ndim=1, got shape {x_grid.shape}")
    if t_grid.ndim != 1:
        raise ValueError(f"Expected t_grid with ndim=1, got shape {t_grid.shape}")

    # length safeguards
    n_sims, Nt_total, Nx = trajectories.shape
    if x_grid.shape[0] != Nx:
        raise ValueError(f"x_grid length {x_grid.shape[0]} does not match Nx={Nx}")
    if t_grid.shape[0] != Nt_total:
        raise ValueError(f"t_grid length {t_grid.shape[0]} does not match Nt_total={Nt_total}")

    return trajectories, x_grid, t_grid


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


# ------- CREATE DATALOADERS -------

def create_dataloaders(
    trajectories: np.ndarray,
    x_grid: np.ndarray,
    t_grid: np.ndarray,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    test_ids: np.ndarray,
    batch_size: int,
    sim_params: np.ndarray,
    pairs_per_sim_train: int = 50,
    pairs_per_sim_val: int = 20,
    test_stride: int = 5,
) -> tuple[DataLoader, DataLoader, DataLoader]:

    train_dataset = SnapshotPairDataset(
        trajectories=trajectories,
        x_grid=x_grid,
        t_grid=t_grid,
        sim_ids=train_ids,
        sim_params=sim_params,
        pairs_per_sim=pairs_per_sim_train,
        random_pairs=True,
    )

    val_dataset = SnapshotPairDataset(
        trajectories=trajectories,
        x_grid=x_grid,
        t_grid=t_grid,
        sim_ids=val_ids,
        sim_params=sim_params,
        pairs_per_sim=pairs_per_sim_val,
        random_pairs=True,
    )

    test_dataset = SnapshotPairDataset(
        trajectories=trajectories,
        x_grid=x_grid,
        t_grid=t_grid,
        sim_ids=test_ids,
        sim_params=sim_params,
        random_pairs=False,
        stride=test_stride,
    )

    pin = torch.cuda.is_available()
    workers = 2 if pin else 0

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, pin_memory=pin, num_workers=workers)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, pin_memory=pin, num_workers=workers)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False, pin_memory=pin, num_workers=workers)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    project_root = Path(__file__).resolve().parents[1]
    trajectories, x_grid, t_grid = load_sim_data(
        sim_traj_path=str(project_root / "data" / "trajectories.npy"),
        x_grid_path=str(project_root / "data" / "x_grid.npy"),
        t_grid_path=str(project_root / "data" / "t_grid.npy"),
    )
    sim_params = np.load(str(project_root / "data" / "sim_params.npy"), allow_pickle=True)

    train_ids, val_ids, test_ids = split_sim_ids(num_sims=trajectories.shape[0], train_frac=0.7, val_frac=0.15, seed=0)

    batch_size = 64

    train_loader, val_loader, test_loader = create_dataloaders(
        trajectories=trajectories, x_grid=x_grid, t_grid=t_grid,
        train_ids=train_ids, val_ids=val_ids, test_ids=test_ids,
        batch_size=batch_size, sim_params=sim_params,
    )

    # Verify shapes
    x_spatial, cond, yb, t_stats = next(iter(train_loader))
    print(f"x_spatial: {x_spatial.shape}")  # (B, Nx, 2)
    print(f"cond: {cond.shape}")            # (B, 4)
    print(f"Y: {yb.shape}")                 # (B, Nx, 1)
    print(f"T_stats: {t_stats.shape}")      # (B, 2)
