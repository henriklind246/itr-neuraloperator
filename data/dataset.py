from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from src.physics.fd_solver_1d import windowed_sin_flux

# Important constraint: s + k + H <= Nt

# output for one sample (window): (Nx, H, 1)
# input for one sample (window): (Nx, H, 13)
# history tensor shape: (Nx, H, 10)

# Take raw data from simulations (x_grid, t_grid, trajectories) and create samples
# from them defined by the sim_id and a valid starting index s (sim_id, s)

# trajectories.shape = (num_sims, Nt, Nx)
# t_grid.shape = (Nt,)
# x_grid.shape = (Nx,)

class WindowedForecastDataset(Dataset):
    def __init__(
            self,
            trajectories: np.ndarray,
            t_grid: np.ndarray,
            x_grid: np.ndarray,
            sim_ids: np.ndarray,
            sim_params: np.ndarray,
            k: int = 10,
            H: int = 40,
            random_window: bool = True,
            windows_per_sim_per_epoch: int = 1,
            normalize_time: bool = True,
            seed: int = 0
    ):
        self.trajectories = trajectories
        self.sim_params = sim_params
        self.t_grid = t_grid.astype(dtype=np.float32)
        self.x_grid = x_grid.astype(dtype=np.float32)
        self.sim_ids = sim_ids.astype(dtype=np.int64)

        self.k = k
        self.H = H
        self.random_window = random_window
        self.windows_per_sim_per_epoch = windows_per_sim_per_epoch
        self.normalize_time = normalize_time
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.num_sims, self.Nt, self.Nx = self.trajectories.shape

        # largest starting index that satisfies the restriction
        self.max_s = self.Nt - self.H - self.k

        # np array with all valid starting indices
        self.all_s = np.arange(self.max_s + 1, dtype=np.int64)

    def __len__(self):
        """
        Function that determines the total length of the dataset
        """
        if self.random_window:
            return len(self.sim_ids) * self.windows_per_sim_per_epoch
        return len(self.sim_ids) * len(self.all_s)


    def __getitem__(self, idx):
        """
        Function that fetches a data sample given a key (index)

        Chooses sim -> choose or decode valid starting index -> slice history -> slice target -> normalize coords. -> broadcast k/x/t/q -> concatenate to (Nx, H, 13)
        """
        if self.random_window:
            # use mod in case idx > len(sim_ids)
            sim_id = self.sim_ids[idx % len(self.sim_ids)]
            s = self.rng.integers(0, self.max_s + 1)
        else:
            # only use the slice of sim_ids == max_s in size
            sim_pos = idx // len(self.all_s)
            start_pos = idx % len(self.all_s)
            sim_id = self.sim_ids[sim_pos]
            s = self.all_s[start_pos]

        T_hist = self.trajectories[sim_id] # shape (Nt, Nx)

        # history with shape (Nx, k)
        history = T_hist[s:s + self.k, :].T

        # target with shape (Nx, H)
        target = T_hist[s + self.k:s + self.k + self.H, :].T

        # future time coords. for prediction slab: shape (H,)
        t_future = self.t_grid[s + self.k: s + self.k + self.H]


        # future boundary force heat flux values
        # q_future = q(t_{s+k}:t_{s+k+H})
        amp, freq, _ = self.sim_params[sim_id]
        amp = np.float32(amp)
        freq = np.float32(freq)

        q_left = windowed_sin_flux(f=float(freq), A=float(amp), t_on=0.0, t_off=0.2, phase=0.0, tukey_alpha=0.5)
        # shape: (H,)
        q_future = np.array([q_left(t) for t in t_future], dtype=np.float32)

        # normalize spatial domain: shape (Nx,)
        x_norm = (self.x_grid - self.x_grid[0]) / (self.x_grid[-1] - self.x_grid[0])

        if self.normalize_time:
            t_norm = (t_future - self.t_grid[0]) / (self.t_grid[-1] - self.t_grid[0])

        # Broadcast to (Nx, H, Channels)
        history_grid = np.broadcast_to(history[:, None, :], (self.Nx, self.H, self.k)) # None adds a dim. before broadcasting
        x_channel = np.broadcast_to(x_norm[:, None, None], (self.Nx, self.H, 1))
        t_channel = np.broadcast_to(t_norm[None, :, None], (self.Nx, self.H, 1))
        q_channel = np.broadcast_to(q_future[None, :, None], (self.Nx, self.H, 1))

        # Concatenate into X: shape (Nx, H, 13)
        X = np.concatenate([history_grid, x_channel, t_channel, q_channel], axis=-1).astype(np.float32)

        # Y: shape (Nx, H, 1)
        Y = target[:, :, None].astype(np.float32)

        return torch.from_numpy(X), torch.from_numpy(Y)


# --------- LOAD RAW SIM. DATA --------

def load_sim_data(
        sim_traj_path: str,
        x_grid_path: str,
        t_grid_path: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    trajectories = np.load(sim_traj_path) # (num_sims, Nt, Nx)
    x_grid = np.load(x_grid_path) # (Nx,)
    t_grid = np.load(t_grid_path) # (Nt,)

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
        seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    ids = np.arange(num_sims)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)

    n_train = int(num_sims*train_frac)
    n_val = int(num_sims*val_frac)
    n_test = num_sims - n_train - n_val

    train_ids = ids[:n_train]
    val_ids = ids[n_train:n_train + n_val]
    test_ids = ids[n_train + n_val: n_train + n_val + n_test]

    return train_ids, val_ids, test_ids

# ------- CREATE INDEX SAMPLERS FOR TRAIN, VAL, AND TEST -------

def create_dataloaders(
        trajectories: np.ndarray,
        x_grid: np.ndarray,
        t_grid: np.ndarray,
        train_ids: np.ndarray,
        val_ids: np.ndarray,
        test_ids: np.ndarray,
        batch_size: int,
        sim_params: np.ndarray,
        k: int = 10,
        H: int = 40
) -> tuple[DataLoader, DataLoader, DataLoader]:

    train_dataset = WindowedForecastDataset(
        trajectories=trajectories,
        x_grid=x_grid,
        t_grid=t_grid,
        sim_ids=train_ids,
        sim_params=sim_params,
        k=k,
        H=H,
        random_window=True
    )

    val_dataset = WindowedForecastDataset(
        trajectories=trajectories,
        x_grid=x_grid,
        t_grid=t_grid,
        sim_ids=val_ids,
        sim_params=sim_params,
        k=k,
        H=H,
        windows_per_sim_per_epoch=8,
        random_window=True
    )

    test_dataset = WindowedForecastDataset(
        trajectories=trajectories,
        x_grid=x_grid,
        t_grid=t_grid,
        sim_ids=test_ids,
        sim_params=sim_params,
        k=k,
        H=H,
        random_window=False
    )

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    project_root = Path(__file__).resolve().parents[1]
    trajectories, x_grid, t_grid = load_sim_data(sim_traj_path=str(project_root/"data"/"trajectories.npy"), x_grid_path=str(project_root/"data"/"x_grid.npy"), t_grid_path=str(project_root/"data"/"t_grid.npy"))
    sim_params = np.load(str(project_root/"data"/"sim_params.npy"), allow_pickle=True)

    train_ids, val_ids, test_ids = split_sim_ids(num_sims=trajectories.shape[0], train_frac=0.7, val_frac=0.15, seed=0)

    batch_size = 10
    k = 10
    H = 40

    train_loader, val_loader, test_loader = create_dataloaders(trajectories=trajectories, x_grid=x_grid, t_grid=t_grid, train_ids=train_ids, val_ids=val_ids, test_ids=test_ids, batch_size=batch_size, sim_params=sim_params, k=k, H=H)

    # Add batches to tensors
    xb, yb = next(iter(train_loader))
    print(f"Train batch X shape: {xb.shape}") # (B, Nx, H, 13)
    print(f"Train batch Y shape: {yb.shape}") # (B, Nx, H, 1)




