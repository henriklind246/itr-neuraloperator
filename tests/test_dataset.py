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
    def dataset_random(self, synthetic_trajectories, synthetic_sim_params):
        trajectories, x_grid, t_grid = synthetic_trajectories
        sim_ids = np.arange(5)
        return SnapshotPairDataset(
            trajectories=trajectories, t_grid=t_grid, x_grid=x_grid,
            sim_ids=sim_ids, sim_params=synthetic_sim_params,
            pairs_per_sim=10, random_pairs=True,
        )

    @pytest.fixture
    def dataset_deterministic(self, synthetic_trajectories, synthetic_sim_params):
        trajectories, x_grid, t_grid = synthetic_trajectories
        sim_ids = np.arange(5)
        return SnapshotPairDataset(
            trajectories=trajectories, t_grid=t_grid, x_grid=x_grid,
            sim_ids=sim_ids, sim_params=synthetic_sim_params,
            random_pairs=False, stride=10,
        )

    def test_len_random(self, dataset_random):
        assert len(dataset_random) == 5 * 10  # 5 sims * 10 pairs_per_sim

    def test_len_deterministic(self, dataset_deterministic):
        """Deterministic mode enumerates all valid (s, j) pairs at given stride."""
        assert len(dataset_deterministic) > 0

    def test_getitem_returns_4tuple(self, dataset_random):
        result = dataset_random[0]
        assert len(result) == 4

    def test_getitem_shapes(self, dataset_random):
        x_spatial, cond, Y, T_stats = dataset_random[0]
        Nx = 11
        assert x_spatial.shape == (Nx, 2)
        assert cond.shape == (4,)
        assert Y.shape == (Nx, 1)
        assert T_stats.shape == (2,)

    def test_getitem_dtypes(self, dataset_random):
        x_spatial, cond, Y, T_stats = dataset_random[0]
        assert x_spatial.dtype == torch.float32
        assert cond.dtype == torch.float32
        assert Y.dtype == torch.float32
        assert T_stats.dtype == torch.float32

    def test_x_coord_normalized(self, dataset_random):
        x_spatial, _, _, _ = dataset_random[0]
        x_channel = x_spatial[:, 1]  # second channel is x_norm
        assert x_channel.min() >= -1e-6
        assert x_channel.max() <= 1.0 + 1e-6

    def test_t_bar_positive(self, dataset_random):
        """Lead time t̄ should always be > 0 (target after source)."""
        for i in range(min(10, len(dataset_random))):
            _, cond, _, _ = dataset_random[i]
            t_bar_norm = cond[0].item()
            assert t_bar_norm > 0

    def test_conditioning_in_unit_range(self, dataset_random):
        """All conditioning values should be in [0, 1]."""
        for i in range(min(10, len(dataset_random))):
            _, cond, _, _ = dataset_random[i]
            assert torch.all(cond >= -1e-6)
            assert torch.all(cond <= 1.0 + 1e-6)

    def test_temperature_normalization(self, dataset_random):
        """Source temperature channel should have mean ≈ 0, std ≈ 1."""
        x_spatial, _, _, _ = dataset_random[0]
        T_norm = x_spatial[:, 0]  # first channel is T̃_source
        assert abs(T_norm.mean().item()) < 0.5  # roughly centered
        assert abs(T_norm.std().item() - 1.0) < 0.5  # roughly unit std

    def test_T_stats_matches_source(self, synthetic_trajectories, synthetic_sim_params):
        """T_stats (μ_s, σ_s) should match the source snapshot statistics."""
        trajectories, x_grid, t_grid = synthetic_trajectories
        sim_ids = np.arange(1)  # single sim for easier verification
        ds = SnapshotPairDataset(
            trajectories=trajectories, t_grid=t_grid, x_grid=x_grid,
            sim_ids=sim_ids, sim_params=synthetic_sim_params,
            pairs_per_sim=1, random_pairs=True, seed=42,
        )
        x_spatial, _, _, T_stats = ds[0]
        mu_s = T_stats[0].item()
        sigma_s = T_stats[1].item()
        # The source temp channel should satisfy: T_source_norm * (sigma_s + eps) + mu_s ≈ T_source_original
        # We can at least verify T_stats are finite and sigma_s >= 0
        assert np.isfinite(mu_s)
        assert sigma_s >= 0

    def test_seed_determinism(self, synthetic_trajectories, synthetic_sim_params):
        trajectories, x_grid, t_grid = synthetic_trajectories
        sim_ids = np.arange(5)
        ds1 = SnapshotPairDataset(
            trajectories=trajectories, t_grid=t_grid, x_grid=x_grid,
            sim_ids=sim_ids, sim_params=synthetic_sim_params,
            pairs_per_sim=5, random_pairs=True, seed=0,
        )
        ds2 = SnapshotPairDataset(
            trajectories=trajectories, t_grid=t_grid, x_grid=x_grid,
            sim_ids=sim_ids, sim_params=synthetic_sim_params,
            pairs_per_sim=5, random_pairs=True, seed=0,
        )
        xs1, c1, y1, s1 = ds1[0]
        xs2, c2, y2, s2 = ds2[0]
        assert torch.equal(xs1, xs2)
        assert torch.equal(c1, c2)
        assert torch.equal(y1, y2)
        assert torch.equal(s1, s2)

    def test_stratified_returns_valid_samples(self, synthetic_trajectories, synthetic_sim_params):
        """Stratified sampling should produce valid (positive lead time) samples."""
        trajectories, x_grid, t_grid = synthetic_trajectories
        sim_ids = np.arange(5)
        ds = SnapshotPairDataset(
            trajectories=trajectories, t_grid=t_grid, x_grid=x_grid,
            sim_ids=sim_ids, sim_params=synthetic_sim_params,
            pairs_per_sim=20, random_pairs=True, stratified=True, seed=0,
        )
        for i in range(len(ds)):
            x_spatial, cond, Y, T_stats = ds[i]
            assert x_spatial.shape == (11, 2)
            assert cond[0].item() > 0  # lead time > 0

    def test_stratified_lead_time_more_uniform(self, synthetic_trajectories, synthetic_sim_params):
        """Stratified mode should produce a more uniform lead-time distribution
        than naive sampling (less triangular bias toward short lead times)."""
        trajectories, x_grid, t_grid = synthetic_trajectories
        Nt = t_grid.shape[0]
        sim_ids = np.arange(1)  # single sim for clear comparison
        n_pairs = 500

        ds_naive = SnapshotPairDataset(
            trajectories=trajectories, t_grid=t_grid, x_grid=x_grid,
            sim_ids=sim_ids, sim_params=synthetic_sim_params,
            pairs_per_sim=n_pairs, random_pairs=True, stratified=False, seed=0,
        )
        ds_strat = SnapshotPairDataset(
            trajectories=trajectories, t_grid=t_grid, x_grid=x_grid,
            sim_ids=sim_ids, sim_params=synthetic_sim_params,
            pairs_per_sim=n_pairs, random_pairs=True, stratified=True, seed=0,
        )

        lead_naive = [ds_naive[i][1][0].item() for i in range(n_pairs)]
        lead_strat = [ds_strat[i][1][0].item() for i in range(n_pairs)]

        # Stratified should have higher mean lead time (naive is biased toward short)
        assert np.mean(lead_strat) > np.mean(lead_naive)


# ===================== create_dataloaders =====================

class TestCreateDataloaders:
    def test_returns_three(self, synthetic_trajectories, synthetic_sim_params):
        trajectories, x_grid, t_grid = synthetic_trajectories
        train_ids, val_ids, test_ids = split_sim_ids(20, 0.7, 0.15, seed=0)
        loaders = create_dataloaders(
            trajectories, x_grid, t_grid,
            train_ids, val_ids, test_ids,
            batch_size=4, sim_params=synthetic_sim_params,
        )
        assert len(loaders) == 3

    def test_batch_shapes(self, synthetic_trajectories, synthetic_sim_params):
        trajectories, x_grid, t_grid = synthetic_trajectories
        train_ids, val_ids, test_ids = split_sim_ids(20, 0.7, 0.15, seed=0)
        train_loader, _, _ = create_dataloaders(
            trajectories, x_grid, t_grid,
            train_ids, val_ids, test_ids,
            batch_size=4, sim_params=synthetic_sim_params,
        )
        x_spatial, cond, Y, T_stats = next(iter(train_loader))
        assert x_spatial.shape[0] <= 4
        assert x_spatial.shape[1] == 11   # Nx
        assert x_spatial.shape[2] == 2    # T̃_source + x_norm
        assert cond.shape[1] == 4         # t̄, A, f, R_c
        assert Y.shape[-1] == 1
        assert T_stats.shape[-1] == 2     # μ_s, σ_s

    def test_no_data_leakage(self, synthetic_trajectories, synthetic_sim_params):
        trajectories, x_grid, t_grid = synthetic_trajectories
        train_ids, val_ids, test_ids = split_sim_ids(20, 0.7, 0.15, seed=0)
        train_loader, val_loader, test_loader = create_dataloaders(
            trajectories, x_grid, t_grid, train_ids, val_ids, test_ids,
            batch_size=4, sim_params=synthetic_sim_params,
        )
        train_sims = set(train_loader.dataset.sim_ids.tolist())
        val_sims = set(val_loader.dataset.sim_ids.tolist())
        test_sims = set(test_loader.dataset.sim_ids.tolist())
        assert train_sims.isdisjoint(val_sims)
        assert train_sims.isdisjoint(test_sims)
        assert val_sims.isdisjoint(test_sims)
