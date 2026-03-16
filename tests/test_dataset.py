import numpy as np
import pytest
import torch

from data.dataset import (
    load_sim_data,
    split_sim_ids,
    WindowedForecastDataset,
    create_dataloaders,
)


# ===================== load_sim_data =====================

class TestLoadSimData:
    def test_correct_shapes(self, tmp_npy_data):
        traj_path, x_path, t_path = tmp_npy_data
        trajectories, x_grid, t_grid = load_sim_data(traj_path, x_path, t_path)
        assert trajectories.shape == (20, 51, 11)
        assert x_grid.shape == (11,)
        assert t_grid.shape == (51,)

    def test_wrong_traj_ndim_raises(self, tmp_path):
        bad = np.zeros((10, 5))
        x = np.zeros(5)
        t = np.zeros(10)
        np.save(tmp_path / "traj.npy", bad)
        np.save(tmp_path / "x.npy", x)
        np.save(tmp_path / "t.npy", t)
        with pytest.raises(ValueError, match="ndim=3"):
            load_sim_data(str(tmp_path / "traj.npy"), str(tmp_path / "x.npy"), str(tmp_path / "t.npy"))

    def test_wrong_xgrid_ndim_raises(self, tmp_path):
        traj = np.zeros((5, 10, 4))
        bad_x = np.zeros((4, 2))
        t = np.zeros(10)
        np.save(tmp_path / "traj.npy", traj)
        np.save(tmp_path / "x.npy", bad_x)
        np.save(tmp_path / "t.npy", t)
        with pytest.raises(ValueError, match="x_grid.*ndim=1"):
            load_sim_data(str(tmp_path / "traj.npy"), str(tmp_path / "x.npy"), str(tmp_path / "t.npy"))

    def test_xgrid_length_mismatch_raises(self, tmp_path):
        traj = np.zeros((5, 10, 4))
        bad_x = np.zeros(7)  # should be 4
        t = np.zeros(10)
        np.save(tmp_path / "traj.npy", traj)
        np.save(tmp_path / "x.npy", bad_x)
        np.save(tmp_path / "t.npy", t)
        with pytest.raises(ValueError, match="x_grid length"):
            load_sim_data(str(tmp_path / "traj.npy"), str(tmp_path / "x.npy"), str(tmp_path / "t.npy"))

    def test_tgrid_length_mismatch_raises(self, tmp_path):
        traj = np.zeros((5, 10, 4))
        x = np.zeros(4)
        bad_t = np.zeros(7)  # should be 10
        np.save(tmp_path / "traj.npy", traj)
        np.save(tmp_path / "x.npy", x)
        np.save(tmp_path / "t.npy", bad_t)
        with pytest.raises(ValueError, match="t_grid length"):
            load_sim_data(str(tmp_path / "traj.npy"), str(tmp_path / "x.npy"), str(tmp_path / "t.npy"))


# ===================== split_sim_ids =====================

class TestSplitSimIds:
    def test_sizes(self):
        train, val, test = split_sim_ids(100, 0.7, 0.15, seed=0)
        assert len(train) == 70
        assert len(val) == 15
        assert len(test) == 15

    def test_no_overlap(self):
        train, val, test = split_sim_ids(100, 0.7, 0.15, seed=0)
        all_ids = set(train) | set(val) | set(test)
        assert len(all_ids) == len(train) + len(val) + len(test)

    def test_covers_all(self):
        train, val, test = split_sim_ids(100, 0.7, 0.15, seed=0)
        assert set(train) | set(val) | set(test) == set(range(100))

    def test_deterministic_same_seed(self):
        a1, b1, c1 = split_sim_ids(100, 0.7, 0.15, seed=0)
        a2, b2, c2 = split_sim_ids(100, 0.7, 0.15, seed=0)
        np.testing.assert_array_equal(a1, a2)
        np.testing.assert_array_equal(b1, b2)
        np.testing.assert_array_equal(c1, c2)

    def test_different_seed_differs(self):
        a1, _, _ = split_sim_ids(100, 0.7, 0.15, seed=0)
        a2, _, _ = split_sim_ids(100, 0.7, 0.15, seed=42)
        assert not np.array_equal(a1, a2)

    def test_small_dataset(self):
        train, val, test = split_sim_ids(3, 0.7, 0.15, seed=0)
        # int(3*0.7)=2, int(3*0.15)=0, rest=1
        assert len(train) == 2
        assert len(val) == 0
        assert len(test) == 1

    def test_all_ids_valid_range(self):
        train, val, test = split_sim_ids(50, 0.7, 0.15, seed=0)
        all_ids = np.concatenate([train, val, test])
        assert np.all(all_ids >= 0)
        assert np.all(all_ids < 50)


# ===================== WindowedForecastDataset =====================

class TestWindowedForecastDataset:
    @pytest.fixture
    def dataset_random(self, synthetic_trajectories, synthetic_sim_params):
        trajectories, x_grid, t_grid = synthetic_trajectories
        sim_ids = np.arange(5)
        return WindowedForecastDataset(
            trajectories=trajectories, t_grid=t_grid, x_grid=x_grid,
            sim_ids=sim_ids, sim_params=synthetic_sim_params,
            k=5, H=10, random_window=True,
            windows_per_sim_per_epoch=1,
        )

    @pytest.fixture
    def dataset_exhaustive(self, synthetic_trajectories, synthetic_sim_params):
        trajectories, x_grid, t_grid = synthetic_trajectories
        sim_ids = np.arange(5)
        return WindowedForecastDataset(
            trajectories=trajectories, t_grid=t_grid, x_grid=x_grid,
            sim_ids=sim_ids, sim_params=synthetic_sim_params,
            k=5, H=10, random_window=False,
        )

    def test_len_random(self, dataset_random):
        assert len(dataset_random) == 5  # 5 sims * 1 window

    def test_len_exhaustive(self, dataset_exhaustive):
        # max_s = 51 - 10 - 5 = 36, all_s = [0..36] = 37 entries
        assert len(dataset_exhaustive) == 5 * 37

    def test_getitem_shapes(self, dataset_random):
        X, Y = dataset_random[0]
        Nx = 11
        H = 10
        assert X.shape == (Nx, H, 5 + 1 + 1 + 1 + 1 + 1)  # k=5 history + x + t + q + k_field + rcp_field = 10
        assert Y.shape == (Nx, H, 1)

    def test_getitem_dtypes(self, dataset_random):
        X, Y = dataset_random[0]
        assert X.dtype == torch.float32
        assert Y.dtype == torch.float32

    def test_x_coord_normalized(self, dataset_random):
        X, _ = dataset_random[0]
        x_channel = X[:, 0, -5]  # fifth-to-last channel (x), first time step
        assert x_channel.min() >= -1e-6
        assert x_channel.max() <= 1.0 + 1e-6

    def test_t_coord_normalized(self, dataset_random):
        X, _ = dataset_random[0]
        t_channel = X[0, :, -4]  # fourth-to-last channel (t), first spatial point
        assert t_channel.min() >= -1e-6
        assert t_channel.max() <= 1.0 + 1e-6

    def test_seed_determinism(self, synthetic_trajectories, synthetic_sim_params):
        trajectories, x_grid, t_grid = synthetic_trajectories
        sim_ids = np.arange(5)
        ds1 = WindowedForecastDataset(trajectories=trajectories, t_grid=t_grid, x_grid=x_grid,
                                       sim_ids=sim_ids, sim_params=synthetic_sim_params,
                                       k=5, H=10, random_window=True, seed=0)
        ds2 = WindowedForecastDataset(trajectories=trajectories, t_grid=t_grid, x_grid=x_grid,
                                       sim_ids=sim_ids, sim_params=synthetic_sim_params,
                                       k=5, H=10, random_window=True, seed=0)
        X1, Y1 = ds1[0]
        X2, Y2 = ds2[0]
        assert torch.equal(X1, X2)
        assert torch.equal(Y1, Y2)

    def test_max_s_constraint(self, dataset_random):
        assert dataset_random.max_s == 51 - 10 - 5  # Nt - H - k


# ===================== create_dataloaders =====================

class TestCreateDataloaders:
    def test_returns_three(self, synthetic_trajectories, synthetic_sim_params):
        trajectories, x_grid, t_grid = synthetic_trajectories
        train_ids, val_ids, test_ids = split_sim_ids(20, 0.7, 0.15, seed=0)
        loaders = create_dataloaders(trajectories, x_grid, t_grid,
                                     train_ids, val_ids, test_ids,
                                     batch_size=4, sim_params=synthetic_sim_params,
                                     k=5, H=10)
        assert len(loaders) == 3

    def test_batch_shapes(self, synthetic_trajectories, synthetic_sim_params):
        trajectories, x_grid, t_grid = synthetic_trajectories
        train_ids, val_ids, test_ids = split_sim_ids(20, 0.7, 0.15, seed=0)
        train_loader, _, _ = create_dataloaders(trajectories, x_grid, t_grid,
                                                 train_ids, val_ids, test_ids,
                                                 batch_size=4, sim_params=synthetic_sim_params,
                                                 k=5, H=10)
        X, Y = next(iter(train_loader))
        assert X.shape[0] <= 4
        assert X.shape[1] == 11  # Nx
        assert X.shape[2] == 10  # H
        assert X.shape[3] == 10  # k + 5 (history + x + t + q + k_field + rcp_field)
        assert Y.shape[-1] == 1

    def test_no_data_leakage(self, synthetic_trajectories, synthetic_sim_params):
        trajectories, x_grid, t_grid = synthetic_trajectories
        train_ids, val_ids, test_ids = split_sim_ids(20, 0.7, 0.15, seed=0)
        train_loader, val_loader, test_loader = create_dataloaders(
            trajectories, x_grid, t_grid, train_ids, val_ids, test_ids,
            batch_size=4, sim_params=synthetic_sim_params, k=5, H=10)
        train_sims = set(train_loader.dataset.sim_ids.tolist())
        val_sims = set(val_loader.dataset.sim_ids.tolist())
        test_sims = set(test_loader.dataset.sim_ids.tolist())
        assert train_sims.isdisjoint(val_sims)
        assert train_sims.isdisjoint(test_sims)
        assert val_sims.isdisjoint(test_sims)
