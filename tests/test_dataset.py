import numpy as np
import pytest
import torch

from data.dataset import (
    compute_global_stats,
    load_sim_data,
    split_sim_ids,
    SnapshotPairDataset,
    create_dataloaders,
)

# Global stats for synthetic test data (standard_normal → mu≈0, sigma≈1)
_SYNTH_MU = 0.0
_SYNTH_SIGMA = 1.0


# ===================== load_sim_data =====================

class TestLoadSimData:
    def test_correct_shapes(self, tmp_npy_data):
        traj_path, x_path, y_path, t_path = tmp_npy_data
        trajectories, x_grid, y_grid, t_grid = load_sim_data(traj_path, x_path, y_path, t_path)
        assert trajectories.shape == (20, 51, 11, 11)
        assert x_grid.shape == (11,)
        assert y_grid.shape == (11,)
        assert t_grid.shape == (51,)

    def test_wrong_traj_ndim_raises(self, tmp_path):
        bad = np.zeros((10, 5, 4))  # 3D instead of 4D
        x = np.zeros(5)
        y = np.zeros(4)
        t = np.zeros(10)
        np.save(tmp_path / "traj.npy", bad)
        np.save(tmp_path / "x.npy", x)
        np.save(tmp_path / "y.npy", y)
        np.save(tmp_path / "t.npy", t)
        with pytest.raises(ValueError, match="ndim=4"):
            load_sim_data(
                str(tmp_path / "traj.npy"),
                str(tmp_path / "x.npy"),
                str(tmp_path / "y.npy"),
                str(tmp_path / "t.npy"),
            )

    def test_wrong_xgrid_ndim_raises(self, tmp_path):
        traj = np.zeros((5, 10, 4, 3))
        bad_x = np.zeros((4, 2))
        y = np.zeros(3)
        t = np.zeros(10)
        np.save(tmp_path / "traj.npy", traj)
        np.save(tmp_path / "x.npy", bad_x)
        np.save(tmp_path / "y.npy", y)
        np.save(tmp_path / "t.npy", t)
        with pytest.raises(ValueError, match="x_grid.*ndim=1"):
            load_sim_data(
                str(tmp_path / "traj.npy"),
                str(tmp_path / "x.npy"),
                str(tmp_path / "y.npy"),
                str(tmp_path / "t.npy"),
            )

    def test_xgrid_length_mismatch_raises(self, tmp_path):
        traj = np.zeros((5, 10, 4, 3))
        bad_x = np.zeros(7)  # should be 4
        y = np.zeros(3)
        t = np.zeros(10)
        np.save(tmp_path / "traj.npy", traj)
        np.save(tmp_path / "x.npy", bad_x)
        np.save(tmp_path / "y.npy", y)
        np.save(tmp_path / "t.npy", t)
        with pytest.raises(ValueError, match="x_grid length"):
            load_sim_data(
                str(tmp_path / "traj.npy"),
                str(tmp_path / "x.npy"),
                str(tmp_path / "y.npy"),
                str(tmp_path / "t.npy"),
            )

    def test_ygrid_length_mismatch_raises(self, tmp_path):
        traj = np.zeros((5, 10, 4, 3))
        x = np.zeros(4)
        bad_y = np.zeros(7)  # should be 3
        t = np.zeros(10)
        np.save(tmp_path / "traj.npy", traj)
        np.save(tmp_path / "x.npy", x)
        np.save(tmp_path / "y.npy", bad_y)
        np.save(tmp_path / "t.npy", t)
        with pytest.raises(ValueError, match="y_grid length"):
            load_sim_data(
                str(tmp_path / "traj.npy"),
                str(tmp_path / "x.npy"),
                str(tmp_path / "y.npy"),
                str(tmp_path / "t.npy"),
            )

    def test_tgrid_length_mismatch_raises(self, tmp_path):
        traj = np.zeros((5, 10, 4, 3))
        x = np.zeros(4)
        y = np.zeros(3)
        bad_t = np.zeros(7)  # should be 10
        np.save(tmp_path / "traj.npy", traj)
        np.save(tmp_path / "x.npy", x)
        np.save(tmp_path / "y.npy", y)
        np.save(tmp_path / "t.npy", bad_t)
        with pytest.raises(ValueError, match="t_grid length"):
            load_sim_data(
                str(tmp_path / "traj.npy"),
                str(tmp_path / "x.npy"),
                str(tmp_path / "y.npy"),
                str(tmp_path / "t.npy"),
            )


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
        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        sim_ids = np.arange(5)
        return SnapshotPairDataset(
            trajectories=trajectories, t_grid=t_grid, x_grid=x_grid, y_grid=y_grid,
            sim_ids=sim_ids, sim_params=synthetic_sim_params,
            mu_global=_SYNTH_MU, sigma_global=_SYNTH_SIGMA,
            n_snapshots=6,
        )

    @pytest.fixture
    def dataset_full(self, synthetic_trajectories, synthetic_sim_params):
        """All-to-all dataset using all time steps."""
        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        sim_ids = np.arange(5)
        return SnapshotPairDataset(
            trajectories=trajectories, t_grid=t_grid, x_grid=x_grid, y_grid=y_grid,
            sim_ids=sim_ids, sim_params=synthetic_sim_params,
            mu_global=_SYNTH_MU, sigma_global=_SYNTH_SIGMA,
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
        Nx, Ny = 11, 11
        assert x_spatial.shape == (Nx, Ny, 3)
        assert cond.shape == (5,)
        assert Y.shape == (Nx, Ny, 1)
        assert T_stats.shape == (2,)

    def test_getitem_dtypes(self, dataset_subsampled):
        x_spatial, cond, Y, T_stats = dataset_subsampled[0]
        assert x_spatial.dtype == torch.float32
        assert cond.dtype == torch.float32
        assert Y.dtype == torch.float32
        assert T_stats.dtype == torch.float32

    def test_x_coord_normalized(self, dataset_subsampled):
        x_spatial, _, _, _ = dataset_subsampled[0]
        x_channel = x_spatial[:, :, 1]  # second channel is x_norm
        assert x_channel.min() >= -1e-6
        assert x_channel.max() <= 1.0 + 1e-6

    def test_ynorm_channel_varies_along_y(self, dataset_subsampled):
        """y_norm channel should be constant along x and monotone along y."""
        x_spatial, _, _, _ = dataset_subsampled[0]
        y_channel = x_spatial[:, :, 2]  # third channel is y_norm
        # Constant along x (axis=0)
        assert torch.allclose(y_channel[0, :], y_channel[-1, :])
        assert torch.allclose(y_channel.std(dim=0), torch.zeros(y_channel.shape[1]), atol=1e-6)
        # Monotone along y
        diffs = y_channel[0, 1:] - y_channel[0, :-1]
        assert torch.all(diffs > 0)

    def test_ynorm_channel_unit_range(self, dataset_subsampled):
        """y_norm endpoints should be 0 and 1."""
        x_spatial, _, _, _ = dataset_subsampled[0]
        y_channel = x_spatial[:, :, 2]
        assert y_channel[:, 0].abs().max().item() < 1e-6
        assert (y_channel[:, -1] - 1.0).abs().max().item() < 1e-6

    def test_t_bar_positive(self, dataset_subsampled):
        """Lead time t_bar should always be > 0 (target after source)."""
        for i in range(min(10, len(dataset_subsampled))):
            _, cond, _, _ = dataset_subsampled[i]
            t_bar_norm = cond[0].item()
            assert t_bar_norm > 0

    def test_conditioning_in_unit_range(self, dataset_subsampled):
        """All 5 conditioning values (min-max normalized) should be in [0, 1]."""
        for i in range(min(10, len(dataset_subsampled))):
            _, cond, _, _ = dataset_subsampled[i]
            assert torch.all(cond >= -1e-6)
            assert torch.all(cond <= 1.0 + 1e-6)

    def test_temperature_normalization(self, dataset_subsampled):
        """Source temperature should be globally normalized (finite values)."""
        x_spatial, _, _, _ = dataset_subsampled[0]
        T_norm = x_spatial[:, :, 0]  # first channel is T_source_norm
        assert torch.all(torch.isfinite(T_norm))

    def test_T_stats_returns_global_stats(self, synthetic_trajectories, synthetic_sim_params):
        """T_stats should return (mu_global, sigma_global), constant across samples."""
        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        sim_ids = np.arange(1)  # single sim for easier verification
        ds = SnapshotPairDataset(
            trajectories=trajectories, t_grid=t_grid, x_grid=x_grid, y_grid=y_grid,
            sim_ids=sim_ids, sim_params=synthetic_sim_params,
            mu_global=_SYNTH_MU, sigma_global=_SYNTH_SIGMA,
            n_snapshots=6,
        )
        _, _, _, T_stats_0 = ds[0]
        _, _, _, T_stats_1 = ds[1]
        # T_stats should be constant (global stats, not per-sample)
        assert torch.equal(T_stats_0, T_stats_1)
        assert T_stats_0[0].item() == pytest.approx(_SYNTH_MU)
        assert T_stats_0[1].item() == pytest.approx(_SYNTH_SIGMA)

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
        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        sim_ids = np.arange(1)
        ds_train = SnapshotPairDataset(
            trajectories=trajectories, t_grid=t_grid, x_grid=x_grid, y_grid=y_grid,
            sim_ids=sim_ids, sim_params=synthetic_sim_params,
            mu_global=_SYNTH_MU, sigma_global=_SYNTH_SIGMA,
            n_snapshots=6,
        )
        ds_test = SnapshotPairDataset(
            trajectories=trajectories, t_grid=t_grid, x_grid=x_grid, y_grid=y_grid,
            sim_ids=sim_ids, sim_params=synthetic_sim_params,
            mu_global=_SYNTH_MU, sigma_global=_SYNTH_SIGMA,
            n_snapshots=10,
        )
        assert len(ds_test) > len(ds_train)


# ===================== create_dataloaders =====================

class TestCreateDataloaders:
    def test_returns_three(self, synthetic_trajectories, synthetic_sim_params):
        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        train_ids, val_ids, test_ids = split_sim_ids(20, 0.7, 0.15, seed=0)
        loaders = create_dataloaders(
            trajectories, x_grid, y_grid, t_grid,
            train_ids, val_ids, test_ids,
            batch_size=4, sim_params=synthetic_sim_params,
            mu_global=_SYNTH_MU, sigma_global=_SYNTH_SIGMA,
            n_snapshots=6,
        )
        assert len(loaders) == 3

    def test_batch_shapes(self, synthetic_trajectories, synthetic_sim_params):
        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        train_ids, val_ids, test_ids = split_sim_ids(20, 0.7, 0.15, seed=0)
        train_loader, _, _ = create_dataloaders(
            trajectories, x_grid, y_grid, t_grid,
            train_ids, val_ids, test_ids,
            batch_size=4, sim_params=synthetic_sim_params,
            mu_global=_SYNTH_MU, sigma_global=_SYNTH_SIGMA,
            n_snapshots=6,
        )
        x_spatial, cond, Y, T_stats = next(iter(train_loader))
        assert x_spatial.shape[0] <= 4
        assert x_spatial.shape[1] == 11   # Nx
        assert x_spatial.shape[2] == 11   # Ny
        assert x_spatial.shape[3] == 3    # T_source + x_norm + y_norm
        assert cond.shape[1] == 5         # t_bar, t_s, A, f, R_c
        assert Y.shape[-1] == 1
        assert T_stats.shape[-1] == 2     # mu_global, sigma_global

    def test_no_data_leakage(self, synthetic_trajectories, synthetic_sim_params):
        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        train_ids, val_ids, test_ids = split_sim_ids(20, 0.7, 0.15, seed=0)
        train_loader, val_loader, test_loader = create_dataloaders(
            trajectories, x_grid, y_grid, t_grid, train_ids, val_ids, test_ids,
            batch_size=4, sim_params=synthetic_sim_params,
            mu_global=_SYNTH_MU, sigma_global=_SYNTH_SIGMA,
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
        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        train_ids, val_ids, test_ids = split_sim_ids(20, 0.7, 0.15, seed=0)
        _, _, test_loader = create_dataloaders(
            trajectories, x_grid, y_grid, t_grid, train_ids, val_ids, test_ids,
            batch_size=4, sim_params=synthetic_sim_params,
            mu_global=_SYNTH_MU, sigma_global=_SYNTH_SIGMA,
            n_snapshots=6, n_snapshots_test=10,
        )
        # Test dataset should use n_snapshots_test=10 -> C(10,2)=45 pairs per sim
        n_test_sims = len(test_ids)
        assert len(test_loader.dataset) == n_test_sims * 45

    def test_noise_std_only_on_train(self, synthetic_trajectories, synthetic_sim_params):
        """noise_std should only be applied to training set, not val/test."""
        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        train_ids, val_ids, test_ids = split_sim_ids(20, 0.7, 0.15, seed=0)
        train_loader, val_loader, test_loader = create_dataloaders(
            trajectories, x_grid, y_grid, t_grid, train_ids, val_ids, test_ids,
            batch_size=4, sim_params=synthetic_sim_params,
            mu_global=_SYNTH_MU, sigma_global=_SYNTH_SIGMA,
            n_snapshots=6, noise_std=0.1,
        )
        assert train_loader.dataset.noise_std == 0.1
        assert val_loader.dataset.noise_std == 0.0
        assert test_loader.dataset.noise_std == 0.0

    def test_noise_augmentation_changes_source(self, synthetic_trajectories, synthetic_sim_params):
        """With noise_std > 0, two reads of the same sample should differ."""
        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        sim_ids = np.arange(5)
        ds = SnapshotPairDataset(
            trajectories=trajectories, t_grid=t_grid, x_grid=x_grid, y_grid=y_grid,
            sim_ids=sim_ids, sim_params=synthetic_sim_params,
            mu_global=_SYNTH_MU, sigma_global=_SYNTH_SIGMA,
            n_snapshots=6, noise_std=0.5,
        )
        x1, _, _, _ = ds[0]
        x2, _, _, _ = ds[0]
        # Source temperature channel should differ due to noise
        assert not torch.equal(x1[:, :, 0], x2[:, :, 0])
        # x_norm channel should be identical (no noise on spatial coords)
        assert torch.equal(x1[:, :, 1], x2[:, :, 1])
        # y_norm channel should be identical too
        assert torch.equal(x1[:, :, 2], x2[:, :, 2])


# ===================== Solver -> Dataset integration =====================

class TestSolverDatasetIntegration:
    """End-to-end smoke: real FVSolver2D output flows through SnapshotPairDataset
    with matching shapes. Guards the shape contract between physics and ML pipelines."""

    def test_tiny_solver_feeds_dataset(self, tmp_path):
        from src.physics.fv_solver_2d import FVSolver2D, Layer2D

        a, b, c, d = 0.0, 1.0, 0.0, 1.0
        # Nx=Ny=12 puts face 5|6 exactly at x=0.5 (interface between layers).
        Nx = Ny = 12
        dt = 0.01
        t_final = 0.1
        num_sims = 2

        layers = [
            Layer2D(x_left=0.0, x_right=0.5, rho=1.0, cp=1.0, k=2.0),
            Layer2D(x_left=0.5, x_right=1.0, rho=1.0, cp=1.0, k=1.0),
        ]

        sim_params_rows = []
        T_hist_list = []
        x_grid = y_grid = t_grid = None
        for i in range(num_sims):
            amp = 100.0 + 50.0 * i
            freq = 2.0 + i
            R_c = 0.1 + 0.1 * i
            T0 = np.full((Nx, Ny), 300.0, dtype=np.float64)
            solver = FVSolver2D(
                a=a, b=b, c=c, d=d, Nx=Nx, Ny=Ny,
                lam_target=0.5, layers=layers,
                t_final=t_final, flux_f=freq, flux_A=amp,
                t_on=0.0, t_off=t_final, phase=0.0,
                dt=dt, interface_R=[float(R_c)],
            )
            t, x_grid, y_grid, T_hist = solver.solve(T0=T0, store_trajectory=True)
            t_grid = t
            T_hist_list.append(T_hist.astype(np.float32))
            sim_params_rows.append((np.float32(amp), np.float32(freq), T0.astype(np.float32), np.float32(R_c)))

        trajectories = np.stack(T_hist_list, axis=0)  # (num_sims, Nt, Nx, Ny)
        sim_params = np.array(sim_params_rows, dtype=object)

        # Shape contract: solver output matches the 4D trajectory the dataset expects
        Nt = len(t_grid)
        assert trajectories.shape == (num_sims, Nt, Nx, Ny)
        assert x_grid.shape == (Nx,)
        assert y_grid.shape == (Ny,)

        mu_global = float(trajectories.mean())
        sigma_global = float(trajectories.std())

        ds = SnapshotPairDataset(
            trajectories=trajectories,
            t_grid=t_grid.astype(np.float32),
            x_grid=x_grid.astype(np.float32),
            y_grid=y_grid.astype(np.float32),
            sim_ids=np.arange(num_sims),
            sim_params=sim_params,
            mu_global=mu_global, sigma_global=sigma_global,
            n_snapshots=5,
        )
        assert len(ds) == num_sims * (5 * 4 // 2)

        spatial, cond, Y, T_stats = ds[0]
        assert spatial.shape == (Nx, Ny, 3)
        assert cond.shape == (5,)
        assert Y.shape == (Nx, Ny, 1)
        assert T_stats.shape == (2,)
        assert torch.all(torch.isfinite(spatial))
        assert torch.all(torch.isfinite(Y))
