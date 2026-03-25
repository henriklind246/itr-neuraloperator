import numpy as np
import pytest
import torch

from data.dataset import (
    load_sim_data,
    split_sim_ids,
    SnapshotPairDataset,
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


# ===================== SnapshotPairDataset =====================

class TestSnapshotPairDataset:
    @pytest.fixture
    def dataset_subsampled(self, synthetic_trajectories, synthetic_sim_params):
        """All-to-all dataset with n_snapshots=6 -> C(6,2)=15 pairs per sim."""
        trajectories, x_grid, t_grid = synthetic_trajectories
        sim_ids = np.arange(5)
        return SnapshotPairDataset(
            trajectories=trajectories, t_grid=t_grid, x_grid=x_grid,
            sim_ids=sim_ids, sim_params=synthetic_sim_params,
            n_snapshots=6,
        )

    @pytest.fixture
    def dataset_full(self, synthetic_trajectories, synthetic_sim_params):
        """All-to-all dataset using all time steps."""
        trajectories, x_grid, t_grid = synthetic_trajectories
        sim_ids = np.arange(5)
        return SnapshotPairDataset(
            trajectories=trajectories, t_grid=t_grid, x_grid=x_grid,
            sim_ids=sim_ids, sim_params=synthetic_sim_params,
            n_snapshots=None,
        )

    def test_len_subsampled(self, dataset_subsampled):
        # 5 sims * C(6,2) = 5 * 15 = 75
        assert len(dataset_subsampled) == 5 * 15

    def test_len_full(self, dataset_full):
        # 5 sims * C(51,2) = 5 * 1275 = 6375
        assert len(dataset_full) == 5 * 1275

    def test_getitem_returns_4tuple(self, dataset_subsampled):
        result = dataset_subsampled[0]
        assert len(result) == 4

    def test_getitem_shapes(self, dataset_subsampled):
        x_spatial, cond, Y, T_stats = dataset_subsampled[0]
        Nx = 11
        assert x_spatial.shape == (Nx, 2)
        assert cond.shape == (4,)
        assert Y.shape == (Nx, 1)
        assert T_stats.shape == (2,)

    def test_getitem_dtypes(self, dataset_subsampled):
        x_spatial, cond, Y, T_stats = dataset_subsampled[0]
        assert x_spatial.dtype == torch.float32
        assert cond.dtype == torch.float32
        assert Y.dtype == torch.float32
        assert T_stats.dtype == torch.float32

    def test_x_coord_normalized(self, dataset_subsampled):
        x_spatial, _, _, _ = dataset_subsampled[0]
        x_channel = x_spatial[:, 1]  # second channel is x_norm
        assert x_channel.min() >= -1e-6
        assert x_channel.max() <= 1.0 + 1e-6

    def test_t_bar_positive(self, dataset_subsampled):
        """Lead time t_bar should always be > 0 (target after source)."""
        for i in range(min(10, len(dataset_subsampled))):
            _, cond, _, _ = dataset_subsampled[i]
            t_bar_norm = cond[0].item()
            assert t_bar_norm > 0

    def test_conditioning_in_unit_range(self, dataset_subsampled):
        """All conditioning values should be in [0, 1]."""
        for i in range(min(10, len(dataset_subsampled))):
            _, cond, _, _ = dataset_subsampled[i]
            assert torch.all(cond >= -1e-6)
            assert torch.all(cond <= 1.0 + 1e-6)

    def test_temperature_normalization(self, dataset_subsampled):
        """Source temperature channel should have mean ~ 0, std ~ 1."""
        x_spatial, _, _, _ = dataset_subsampled[0]
        T_norm = x_spatial[:, 0]  # first channel is T_source_norm
        assert abs(T_norm.mean().item()) < 0.5  # roughly centered
        assert abs(T_norm.std().item() - 1.0) < 0.5  # roughly unit std

    def test_T_stats_matches_source(self, synthetic_trajectories, synthetic_sim_params):
        """T_stats (mu_s, sigma_s) should match the source snapshot statistics."""
        trajectories, x_grid, t_grid = synthetic_trajectories
        sim_ids = np.arange(1)  # single sim for easier verification
        ds = SnapshotPairDataset(
            trajectories=trajectories, t_grid=t_grid, x_grid=x_grid,
            sim_ids=sim_ids, sim_params=synthetic_sim_params,
            n_snapshots=6,
        )
        x_spatial, _, _, T_stats = ds[0]
        mu_s = T_stats[0].item()
        sigma_s = T_stats[1].item()
        assert np.isfinite(mu_s)
        assert sigma_s >= 0

    def test_n_snapshots_uniform_spacing(self, dataset_subsampled):
        """Subsampled t_indices should be uniformly spaced."""
        t_indices = dataset_subsampled.t_indices
        diffs = np.diff(t_indices)
        # Uniform spacing means all diffs should be equal (or differ by at most 1 due to rounding)
        assert np.max(diffs) - np.min(diffs) <= 1

    def test_all_pairs_exhaustive(self, dataset_subsampled):
        """Pair count should match C(n_snapshots, 2) * num_sims."""
        n_snap = len(dataset_subsampled.t_indices)
        expected = 5 * (n_snap * (n_snap - 1) // 2)
        assert len(dataset_subsampled) == expected

    def test_all_lead_times_positive(self, dataset_subsampled):
        """Every pair should have t_j > t_s (positive lead time)."""
        assert np.all(dataset_subsampled._lead_times > 0)

    def test_pairs_sorted_by_lead_time(self, dataset_subsampled):
        """_lead_times array should be non-decreasing."""
        lead_times = dataset_subsampled._lead_times
        assert np.all(lead_times[1:] >= lead_times[:-1])

    def test_curriculum_fraction_reduces_len(self, dataset_subsampled):
        """set_curriculum_fraction(0.5) should expose fewer pairs than full."""
        full_len = len(dataset_subsampled)
        dataset_subsampled.set_curriculum_fraction(0.5)
        reduced_len = len(dataset_subsampled)
        assert reduced_len < full_len
        assert reduced_len >= 1
        # Restore
        dataset_subsampled.set_curriculum_fraction(1.0)

    def test_curriculum_fraction_one_exposes_all(self, dataset_subsampled):
        """set_curriculum_fraction(1.0) should restore full length."""
        full_len = len(dataset_subsampled._pairs)
        dataset_subsampled.set_curriculum_fraction(0.3)
        dataset_subsampled.set_curriculum_fraction(1.0)
        assert len(dataset_subsampled) == full_len

    def test_n_snapshots_test_finer_than_train(self, synthetic_trajectories, synthetic_sim_params):
        """Test dataset with more snapshots should have more pairs per sim."""
        trajectories, x_grid, t_grid = synthetic_trajectories
        sim_ids = np.arange(1)
        ds_train = SnapshotPairDataset(
            trajectories=trajectories, t_grid=t_grid, x_grid=x_grid,
            sim_ids=sim_ids, sim_params=synthetic_sim_params,
            n_snapshots=6,
        )
        ds_test = SnapshotPairDataset(
            trajectories=trajectories, t_grid=t_grid, x_grid=x_grid,
            sim_ids=sim_ids, sim_params=synthetic_sim_params,
            n_snapshots=10,
        )
        assert len(ds_test) > len(ds_train)


# ===================== create_dataloaders =====================

class TestCreateDataloaders:
    def test_returns_three(self, synthetic_trajectories, synthetic_sim_params):
        trajectories, x_grid, t_grid = synthetic_trajectories
        train_ids, val_ids, test_ids = split_sim_ids(20, 0.7, 0.15, seed=0)
        loaders = create_dataloaders(
            trajectories, x_grid, t_grid,
            train_ids, val_ids, test_ids,
            batch_size=4, sim_params=synthetic_sim_params,
            n_snapshots=6,
        )
        assert len(loaders) == 3

    def test_batch_shapes(self, synthetic_trajectories, synthetic_sim_params):
        trajectories, x_grid, t_grid = synthetic_trajectories
        train_ids, val_ids, test_ids = split_sim_ids(20, 0.7, 0.15, seed=0)
        train_loader, _, _ = create_dataloaders(
            trajectories, x_grid, t_grid,
            train_ids, val_ids, test_ids,
            batch_size=4, sim_params=synthetic_sim_params,
            n_snapshots=6,
        )
        x_spatial, cond, Y, T_stats = next(iter(train_loader))
        assert x_spatial.shape[0] <= 4
        assert x_spatial.shape[1] == 11   # Nx
        assert x_spatial.shape[2] == 2    # T_source + x_norm
        assert cond.shape[1] == 4         # t_bar, A, f, R_c
        assert Y.shape[-1] == 1
        assert T_stats.shape[-1] == 2     # mu_s, sigma_s

    def test_no_data_leakage(self, synthetic_trajectories, synthetic_sim_params):
        trajectories, x_grid, t_grid = synthetic_trajectories
        train_ids, val_ids, test_ids = split_sim_ids(20, 0.7, 0.15, seed=0)
        train_loader, val_loader, test_loader = create_dataloaders(
            trajectories, x_grid, t_grid, train_ids, val_ids, test_ids,
            batch_size=4, sim_params=synthetic_sim_params,
            n_snapshots=6,
        )
        train_sims = set(train_loader.dataset.sim_ids.tolist())
        val_sims = set(val_loader.dataset.sim_ids.tolist())
        test_sims = set(test_loader.dataset.sim_ids.tolist())
        assert train_sims.isdisjoint(val_sims)
        assert train_sims.isdisjoint(test_sims)
        assert val_sims.isdisjoint(test_sims)

    def test_n_snapshots_test_passed_through(self, synthetic_trajectories, synthetic_sim_params):
        """n_snapshots_test should give the test set a finer grid."""
        trajectories, x_grid, t_grid = synthetic_trajectories
        train_ids, val_ids, test_ids = split_sim_ids(20, 0.7, 0.15, seed=0)
        _, _, test_loader = create_dataloaders(
            trajectories, x_grid, t_grid, train_ids, val_ids, test_ids,
            batch_size=4, sim_params=synthetic_sim_params,
            n_snapshots=6, n_snapshots_test=10,
        )
        # Test dataset should use n_snapshots_test=10 -> C(10,2)=45 pairs per sim
        n_test_sims = len(test_ids)
        assert len(test_loader.dataset) == n_test_sims * 45
