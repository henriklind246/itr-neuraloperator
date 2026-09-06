import numpy as np
import pytest
import torch

from data.dataset import (
    compute_global_stats,
    load_sim_data,
    split_sim_ids,
    long_lead_pairs,
    SnapshotPairDataset,
    create_dataloaders,
    collate_fn,
)
from problems.forcing import (
    A_AMP_REF,
    COND_STATIC_DIM,
    FORCING_SPATIAL_DESCRIPTOR_SLICE,
    FORCING_TEMPORAL_SAMPLES,
    FORCING_TEMPORAL_TOKEN_DIM,
    SPATIAL_CHANNELS_TEMPORAL,
    _forcing_seq_3tok_from_samples,
    _sample_a,
    build_spatial_profile_bin_averages,
)
from src.physics.boundary_forcing import (
    SPATIAL_BUILDERS,
    TEMPORAL_BUILDERS,
    integrate_temporal_ramped_signed,
    ramped_temporal,
)

# Global stats for synthetic test data (standard_normal → mu≈0, sigma≈1)
_SYNTH_MU = 0.0
_SYNTH_SIGMA = 1.0
# Active dataset default = forcing benchmark, temporal_encoder representation:
# 4 spatial channels [T_tilde, x, y, s_y], 10 static cond dims, (128, 3) tokens.
SPATIAL_IN_CHANNELS = SPATIAL_CHANNELS_TEMPORAL


def _unpack(item):
    """Adapt a dict dataset item / batched dict to the legacy 5-tuple order.

    `SnapshotPairDataset.__getitem__` and the `collate_fn`-backed DataLoader both
    return dicts now; the forcing benchmark always carries `forcing_seq`.
    """
    return (
        item["spatial"], item["cond_static"], item["forcing_seq"],
        item["Y"], item["T_stats"],
    )


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
        assert len(dataset_subsampled) == 5 * 15

    def test_len_full(self, dataset_full):
        assert len(dataset_full) == 5 * 1275

    def test_getitem_returns_dict(self, dataset_subsampled):
        result = dataset_subsampled[0]
        assert isinstance(result, dict)
        assert set(result.keys()) == {
            "spatial", "cond_static", "forcing_seq", "Y", "T_stats",
        }

    def test_getitem_shapes(self, dataset_subsampled):
        spatial, cond_static, forcing_seq, Y, T_stats = _unpack(dataset_subsampled[0])
        Nx, Ny = 11, 11
        assert spatial.shape == (Nx, Ny, SPATIAL_IN_CHANNELS)
        assert cond_static.shape == (COND_STATIC_DIM,)
        assert forcing_seq.shape == (FORCING_TEMPORAL_SAMPLES, FORCING_TEMPORAL_TOKEN_DIM)
        assert Y.shape == (Nx, Ny, 1)
        assert T_stats.shape == (2,)

    def test_getitem_dtypes(self, dataset_subsampled):
        spatial, cond_static, forcing_seq, Y, T_stats = _unpack(dataset_subsampled[0])
        assert spatial.dtype == torch.float32
        assert cond_static.dtype == torch.float32
        assert forcing_seq.dtype == torch.float32
        assert Y.dtype == torch.float32
        assert T_stats.dtype == torch.float32

    def test_x_coord_normalized(self, dataset_subsampled):
        spatial, _, _, _, _ = _unpack(dataset_subsampled[0])
        x_channel = spatial[:, :, 1]
        assert x_channel.min() >= -1e-6
        assert x_channel.max() <= 1.0 + 1e-6

    def test_ynorm_channel_varies_along_y(self, dataset_subsampled):
        spatial, _, _, _, _ = _unpack(dataset_subsampled[0])
        y_channel = spatial[:, :, 2]
        assert torch.allclose(y_channel[0, :], y_channel[-1, :])
        assert torch.allclose(y_channel.std(dim=0), torch.zeros(y_channel.shape[1]), atol=1e-6)
        diffs = y_channel[0, 1:] - y_channel[0, :-1]
        assert torch.all(diffs > 0)

    def test_ynorm_channel_unit_range(self, dataset_subsampled):
        spatial, _, _, _, _ = _unpack(dataset_subsampled[0])
        y_channel = spatial[:, :, 2]
        assert y_channel[:, 0].abs().max().item() < 1e-6
        assert (y_channel[:, -1] - 1.0).abs().max().item() < 1e-6

    def test_sy_channel_matches_spatial_builder(self, dataset_subsampled):
        spatial, _, _, _, _ = _unpack(dataset_subsampled[0])
        sim_id, _, _ = dataset_subsampled._pairs[0]
        params = dataset_subsampled.sim_params[sim_id]
        expected_vec = SPATIAL_BUILDERS[params["spatial_family"]](
            dataset_subsampled.y_grid, **params["spatial_params"]
        )
        expected = torch.from_numpy(
            np.broadcast_to(
                np.asarray(expected_vec, dtype=np.float32)[None, :],
                (dataset_subsampled.Nx, dataset_subsampled.Ny),
            ).copy()
        )
        assert torch.allclose(spatial[:, :, 3], expected)

    def test_sy_channel_constant_along_x(self, dataset_subsampled):
        spatial, _, _, _, _ = _unpack(dataset_subsampled[0])
        sy_channel = spatial[:, :, 3]
        assert torch.allclose(sy_channel[0, :], sy_channel[-1, :])
        assert torch.allclose(sy_channel.std(dim=0), torch.zeros(sy_channel.shape[1]), atol=1e-6)

    def test_forcing_seq_r_endpoints(self, dataset_subsampled):
        _, _, forcing_seq, _, _ = _unpack(dataset_subsampled[0])
        # r_m = m / (M-1), so first is 0 and last is 1
        assert forcing_seq[0, 0].item() == pytest.approx(0.0, abs=1e-6)
        assert forcing_seq[-1, 0].item() == pytest.approx(1.0, abs=1e-6)

    def test_forcing_seq_tok1_matches_amplitude(self, dataset_subsampled):
        """temporal_encoder tok1 == a(t) sampled over [t_s, t_j], scaled by A_AMP_REF."""
        _, _, forcing_seq, _, _ = _unpack(dataset_subsampled[0])
        sim_id, s, j = dataset_subsampled._pairs[0]
        params = dataset_subsampled.sim_params[sim_id]
        q = TEMPORAL_BUILDERS[params["temporal_family"]](**params["temporal_params"])
        t_s = float(dataset_subsampled.t_grid[s])
        t_j = float(dataset_subsampled.t_grid[j])
        _, a_m = _sample_a(q, t_s, t_j, FORCING_TEMPORAL_SAMPLES)
        expected_tok1 = (a_m / A_AMP_REF).astype(np.float32)
        np.testing.assert_allclose(forcing_seq[:, 1].numpy(), expected_tok1, atol=1e-5)

    def test_forcing_seq_tok2_matches_interval_average(self, dataset_subsampled):
        _, _, forcing_seq, _, _ = _unpack(dataset_subsampled[0])
        sim_id, s, j = dataset_subsampled._pairs[0]
        params = dataset_subsampled.sim_params[sim_id]
        t_s = float(dataset_subsampled.t_grid[s])
        t_j = float(dataset_subsampled.t_grid[j])
        q = ramped_temporal(
            params["temporal_family"],
            params["temporal_params"],
            dataset_subsampled.ramp_seconds,
        )
        t_samples, _ = _sample_a(q, t_s, t_j, FORCING_TEMPORAL_SAMPLES)
        expected = np.array(
            [
                integrate_temporal_ramped_signed(
                    params["temporal_family"],
                    params["temporal_params"],
                    float(a),
                    float(b),
                    dataset_subsampled.ramp_seconds,
                )
                / (float(b) - float(a))
                / A_AMP_REF
                for a, b in zip(t_samples[:-1], t_samples[1:])
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(forcing_seq[:-1, 2].numpy(), expected, atol=1e-5)
        assert forcing_seq[-1, 2].item() == pytest.approx(
            forcing_seq[-2, 2].item(), abs=1e-7
        )


    @staticmethod
    def _pulse(onset: float, duration: float, amplitude: float):
        def value(t):
            values = np.asarray(t)
            result = amplitude * (
                (values >= onset) & (values < onset + duration)
            )
            return float(result) if result.ndim == 0 else result.astype(float)

        def integral(a: float, b: float) -> float:
            return amplitude * max(0.0, min(b, onset + duration) - max(a, onset))

        return value, integral

    def test_known_point_collision_is_separated_by_interval_average(self):
        sequences = []
        point_channels = []
        for onset in (0.272639125, 0.2740155):
            params = {
                "Np": 1,
                "A_list": [300.0],
                "t_list": [onset],
                "dt_list": [0.025],
            }
            q = ramped_temporal("pulse_train", params, 0.01)
            t_samples, a_m = _sample_a(q, 0.0, 0.3, 128)
            sequence = _forcing_seq_3tok_from_samples(
                t_samples,
                a_m,
                interval_integral_fn=lambda a, b, p=params: integrate_temporal_ramped_signed(
                    "pulse_train", p, a, b, 0.01
                ),
                A_amp_ref=A_AMP_REF,
            )
            sequences.append(sequence)
            point_channels.append(sequence[:, :2])

        np.testing.assert_array_equal(point_channels[0], point_channels[1])
        assert not np.array_equal(sequences[0], sequences[1])

    def test_subcell_edge_sweep_tracks_occupancy_without_changing_points(self):
        t_samples = np.linspace(0.0, 1.0, 5, dtype=np.float32)
        amplitude = 150.0
        sequences = []
        for occupancy in np.linspace(0.2, 0.8, 7):
            q, integral = self._pulse(0.25, occupancy * 0.25, amplitude)
            a_m = np.asarray(q(t_samples), dtype=np.float32)
            sequences.append(
                _forcing_seq_3tok_from_samples(
                    t_samples,
                    a_m,
                    interval_integral_fn=integral,
                    A_amp_ref=A_AMP_REF,
                )
            )

        for sequence in sequences[1:]:
            np.testing.assert_array_equal(sequence[:, :2], sequences[0][:, :2])
        cell_averages = np.array([sequence[1, 2] for sequence in sequences])
        assert np.all(np.diff(cell_averages) > 0.0)
        assert cell_averages[0] == pytest.approx(0.2 * amplitude / A_AMP_REF)
        assert cell_averages[-1] == pytest.approx(0.8 * amplitude / A_AMP_REF)

    def test_constant_average_and_final_token_convention(self):
        t_samples = np.linspace(0.0, 0.5, 128, dtype=np.float32)
        amplitude = 75.0
        a_m = np.full_like(t_samples, amplitude)
        sequence = _forcing_seq_3tok_from_samples(
            t_samples,
            a_m,
            interval_integral_fn=lambda a, b: amplitude * (b - a),
            A_amp_ref=A_AMP_REF,
        )
        np.testing.assert_allclose(sequence[:, 1], amplitude / A_AMP_REF)
        np.testing.assert_allclose(sequence[:, 2], amplitude / A_AMP_REF)
        assert sequence.dtype == np.float32
        assert sequence[-1, 2] == sequence[-2, 2]

    def test_t_bar_positive(self, dataset_subsampled):
        for i in range(min(10, len(dataset_subsampled))):
            _, cond_static, _, _, _ = _unpack(dataset_subsampled[i])
            t_bar_norm = cond_static[0].item()
            assert t_bar_norm > 0

    def test_cond_static_in_unit_range(self, dataset_subsampled):
        for i in range(min(10, len(dataset_subsampled))):
            _, cond_static, _, _, _ = _unpack(dataset_subsampled[i])
            assert torch.all(cond_static >= -1e-6)
            assert torch.all(cond_static <= 1.0 + 1e-6)

    def test_temperature_normalization(self, dataset_subsampled):
        spatial, _, _, _, _ = _unpack(dataset_subsampled[0])
        T_norm = spatial[:, :, 0]
        assert torch.all(torch.isfinite(T_norm))

    def test_T_stats_returns_global_stats(self, synthetic_trajectories, synthetic_sim_params):
        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        sim_ids = np.arange(1)
        ds = SnapshotPairDataset(
            trajectories=trajectories, t_grid=t_grid, x_grid=x_grid, y_grid=y_grid,
            sim_ids=sim_ids, sim_params=synthetic_sim_params,
            mu_global=_SYNTH_MU, sigma_global=_SYNTH_SIGMA,
            n_snapshots=6,
        )
        _, _, _, _, T_stats_0 = _unpack(ds[0])
        _, _, _, _, T_stats_1 = _unpack(ds[1])
        assert torch.equal(T_stats_0, T_stats_1)
        assert T_stats_0[0].item() == pytest.approx(_SYNTH_MU)
        assert T_stats_0[1].item() == pytest.approx(_SYNTH_SIGMA)

    def test_n_snapshots_uniform_spacing(self, dataset_subsampled):
        t_indices = dataset_subsampled.t_indices
        diffs = np.diff(t_indices)
        assert np.max(diffs) - np.min(diffs) <= 1

    def test_all_pairs_exhaustive(self, dataset_subsampled):
        n_snap = len(dataset_subsampled.t_indices)
        expected = 5 * (n_snap * (n_snap - 1) // 2)
        assert len(dataset_subsampled) == expected

    def test_all_lead_times_positive(self, dataset_subsampled):
        assert np.all(dataset_subsampled._lead_times > 0)

    def test_pairs_sorted_by_lead_time(self, dataset_subsampled):
        lead_times = dataset_subsampled._lead_times
        assert np.all(lead_times[1:] >= lead_times[:-1])

    def test_curriculum_fraction_reduces_len(self, dataset_subsampled):
        full_len = len(dataset_subsampled)
        dataset_subsampled.set_curriculum_fraction(0.5)
        reduced_len = len(dataset_subsampled)
        assert reduced_len < full_len
        assert reduced_len >= 1
        dataset_subsampled.set_curriculum_fraction(1.0)

    def test_curriculum_fraction_one_exposes_all(self, dataset_subsampled):
        full_len = len(dataset_subsampled._pairs)
        dataset_subsampled.set_curriculum_fraction(0.3)
        dataset_subsampled.set_curriculum_fraction(1.0)
        assert len(dataset_subsampled) == full_len

    def test_n_snapshots_test_finer_than_train(self, synthetic_trajectories, synthetic_sim_params):
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

    def test_long_lead_pairs_full_span_only(self, dataset_subsampled):
        s_first = int(dataset_subsampled.t_indices[0])
        j_last = int(dataset_subsampled.t_indices[-1])
        pairs = long_lead_pairs(dataset_subsampled)

        # one full-span pair per sim, all at (t=0 -> t_final)
        assert len(pairs) == len(dataset_subsampled.sim_ids)
        assert all(s == s_first and j == j_last for _, s, j in pairs)
        assert {sim for sim, _, _ in pairs} == {int(s) for s in dataset_subsampled.sim_ids}
        # the full-span pair is the maximum-lead pair (pairs are lead-sorted)
        assert dataset_subsampled._pairs[-1] in pairs


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
        spatial, cond_static, forcing_seq, Y, T_stats = _unpack(next(iter(train_loader)))
        assert spatial.shape[0] <= 4
        assert spatial.shape[1] == 11
        assert spatial.shape[2] == 11
        assert spatial.shape[3] == SPATIAL_IN_CHANNELS
        assert cond_static.shape[1] == COND_STATIC_DIM
        assert forcing_seq.shape[1] == FORCING_TEMPORAL_SAMPLES
        assert forcing_seq.shape[2] == FORCING_TEMPORAL_TOKEN_DIM
        assert Y.shape[-1] == 1
        assert T_stats.shape[-1] == 2

    def test_temporal_samples_pinned_in_temporal_mode(self, synthetic_trajectories, synthetic_sim_params):
        """temporal_encoder forcing pins the token grid to 128 regardless of config."""
        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        train_ids, val_ids, test_ids = split_sim_ids(20, 0.7, 0.15, seed=0)
        train_loader, _, _ = create_dataloaders(
            trajectories, x_grid, y_grid, t_grid,
            train_ids, val_ids, test_ids,
            batch_size=4, sim_params=synthetic_sim_params,
            mu_global=_SYNTH_MU, sigma_global=_SYNTH_SIGMA,
            n_snapshots=6,
            temporal_samples=32,
        )
        _, _, forcing_seq, _, _ = _unpack(next(iter(train_loader)))
        assert forcing_seq.shape[1] == FORCING_TEMPORAL_SAMPLES

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
        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        train_ids, val_ids, test_ids = split_sim_ids(20, 0.7, 0.15, seed=0)
        _, _, test_loader = create_dataloaders(
            trajectories, x_grid, y_grid, t_grid, train_ids, val_ids, test_ids,
            batch_size=4, sim_params=synthetic_sim_params,
            mu_global=_SYNTH_MU, sigma_global=_SYNTH_SIGMA,
            n_snapshots=6, n_snapshots_test=10,
        )
        n_test_sims = len(test_ids)
        assert len(test_loader.dataset) == n_test_sims * 45

    def test_noise_std_only_on_train(self, synthetic_trajectories, synthetic_sim_params):
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

    def test_ddp_val_loader_built_on_nonzero_rank(self, synthetic_trajectories, synthetic_sim_params):
        """Under DDP the val loader must exist on every rank: the W2 collocation
        sampler reads `val_loader.dataset` on all ranks, so a None loader on
        rank>0 crashes the physics path during setup and hangs the job."""
        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        train_ids, val_ids, test_ids = split_sim_ids(20, 0.7, 0.15, seed=0)
        for rank in (0, 1):
            _, val_loader, _ = create_dataloaders(
                trajectories, x_grid, y_grid, t_grid, train_ids, val_ids, test_ids,
                batch_size=4, sim_params=synthetic_sim_params,
                mu_global=_SYNTH_MU, sigma_global=_SYNTH_SIGMA,
                n_snapshots=6, world_size=2, rank=rank, sampler_seed=0,
            )
            assert val_loader is not None, f"val_loader is None on rank={rank}"
            assert set(val_loader.dataset.sim_ids.tolist()) == set(val_ids.tolist())

    def test_noise_augmentation_changes_source(self, synthetic_trajectories, synthetic_sim_params):
        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        sim_ids = np.arange(5)
        ds = SnapshotPairDataset(
            trajectories=trajectories, t_grid=t_grid, x_grid=x_grid, y_grid=y_grid,
            sim_ids=sim_ids, sim_params=synthetic_sim_params,
            mu_global=_SYNTH_MU, sigma_global=_SYNTH_SIGMA,
            n_snapshots=6, noise_std=0.5,
        )
        x1, _, _, _, _ = _unpack(ds[0])
        x2, _, _, _, _ = _unpack(ds[0])
        assert not torch.equal(x1[:, :, 0], x2[:, :, 0])
        assert torch.equal(x1[:, :, 1], x2[:, :, 1])
        assert torch.equal(x1[:, :, 2], x2[:, :, 2])
        assert torch.equal(x1[:, :, 3], x2[:, :, 3])


# ===================== Solver -> Dataset integration =====================

class TestSolverDatasetIntegration:
    """End-to-end smoke: real FVSolver2D output flows through SnapshotPairDataset
    with matching shapes. Guards the shape contract between physics and ML pipelines."""

    def test_tiny_solver_feeds_dataset(self, tmp_path):
        from src.physics.fv_solver_2d import FVSolver2D, Layer2D
        from src.physics.boundary_forcing import build_qL

        a, b, c, d = 0.0, 1.0, 0.0, 1.0
        Nx = Ny = 12
        dt = 0.01
        t_final = 0.1
        num_sims = 2

        layers = [
            Layer2D(x_left=0.0, x_right=0.5, rho=1.0, cp=1.0, k=2.0),
            Layer2D(x_left=0.5, x_right=1.0, rho=1.0, cp=1.0, k=1.0),
        ]
        y_grid_eval = np.linspace(c, d, Ny)

        sim_params_rows = []
        T_hist_list = []
        x_grid = y_grid = t_grid = None
        for i in range(num_sims):
            amp = 100.0 + 50.0 * i
            freq = 2.0 + i
            R_c = 0.1 + 0.1 * i
            T0 = np.full((Nx, Ny), 300.0, dtype=np.float64)
            temporal_params = dict(A=amp, f=freq, t_on=0.0, t_off=t_final, phase=0.0, tukey_alpha=0.5)
            spatial_family = "uniform"
            spatial_params = {}
            q_left_fn, _ = build_qL("sin", temporal_params, spatial_family, spatial_params, y_grid_eval)
            solver = FVSolver2D(
                a=a, b=b, c=c, d=d, Nx=Nx, Ny=Ny,
                lam_target=0.5, layers=layers,
                t_final=t_final, flux_f=freq, flux_A=amp,
                t_on=0.0, t_off=t_final, phase=0.0,
                q_left_fn=q_left_fn,
                dt=dt, interface_R=[float(R_c)],
            )
            t, x_grid, y_grid, T_hist = solver.solve(T0=T0, store_trajectory=True)
            t_grid = t
            T_hist_list.append(T_hist.astype(np.float32))
            sim_params_rows.append({
                "amp": np.float32(amp),
                "freq": np.float32(freq),
                "R_c": np.float32(R_c),
                "T0": T0.astype(np.float32),
                "temporal_family": "sin",
                "temporal_params": temporal_params,
                "spatial_family": spatial_family,
                "spatial_params": spatial_params,
            })

        trajectories = np.stack(T_hist_list, axis=0)
        sim_params = np.array(sim_params_rows, dtype=object)

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

        spatial, cond_static, forcing_seq, Y, T_stats = _unpack(ds[0])
        assert spatial.shape == (Nx, Ny, SPATIAL_IN_CHANNELS)
        assert cond_static.shape == (COND_STATIC_DIM,)
        assert forcing_seq.shape == (FORCING_TEMPORAL_SAMPLES, FORCING_TEMPORAL_TOKEN_DIM)
        assert Y.shape == (Nx, Ny, 1)
        assert T_stats.shape == (2,)
        assert torch.all(torch.isfinite(spatial))
        assert torch.all(torch.isfinite(forcing_seq))
        assert torch.all(torch.isfinite(Y))


# ===================== Static conditioning vector layout =====================


class TestCondStaticLayout:
    """The forcing condition appends eight profile-bin averages to its scalars."""

    def _make_dataset(self, spatial_family, spatial_params,
                      temporal_family, temporal_params,
                      synthetic_trajectories):
        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        sim_params = np.array([{
            "R_c": 0.5,
            "T0": np.zeros((trajectories.shape[2], trajectories.shape[3]), dtype=np.float32),
            "temporal_family": temporal_family,
            "temporal_params": temporal_params,
            "spatial_family": spatial_family,
            "spatial_params": spatial_params,
        }], dtype=object)
        return SnapshotPairDataset(
            trajectories=trajectories[:1], t_grid=t_grid, x_grid=x_grid, y_grid=y_grid,
            sim_ids=np.array([0]), sim_params=sim_params,
            mu_global=_SYNTH_MU, sigma_global=_SYNTH_SIGMA,
            n_snapshots=4,
        )

    _DEFAULT_TEMPORAL = dict(
        A=175.0, f=10.0, t_on=0.0, t_off=0.2, phase=0.0, tukey_alpha=0.5,
    )

    @pytest.mark.parametrize(
        ("spatial_family", "spatial_params"),
        [
            ("uniform", {}),
            ("patch", {"y_c": 0.5, "w": 0.4}),
            ("gaussian", {"y_c": 0.5, "sigma_y": 0.05}),
            ("triangle", {"y_c": 0.5, "ell": 0.2}),
        ],
    )
    def test_spatial_profile_channel_matches_family(self, synthetic_trajectories,
                                                    spatial_family, spatial_params):
        ds = self._make_dataset(spatial_family, spatial_params, "sin", self._DEFAULT_TEMPORAL,
                                synthetic_trajectories)
        spatial, _, _, _, _ = _unpack(ds[0])
        expected_vec = SPATIAL_BUILDERS[spatial_family](ds.y_grid, **spatial_params)
        expected = torch.from_numpy(
            np.broadcast_to(
                np.asarray(expected_vec, dtype=np.float32)[None, :],
                (ds.Nx, ds.Ny),
            ).copy()
        )
        assert torch.allclose(spatial[:, :, 3], expected)

    def test_cond_static_bins_match_each_spatial_family(self, synthetic_trajectories):
        families = [("uniform", {}), ("patch", {"y_c": 0.5, "w": 0.2}),
                    ("gaussian", {"y_c": 0.5, "sigma_y": 0.05}),
                    ("triangle", {"y_c": 0.5, "ell": 0.1})]
        conds, s_ys = [], []
        for sf, sp in families:
            ds = self._make_dataset(sf, sp, "sin", self._DEFAULT_TEMPORAL, synthetic_trajectories)
            spatial, cond_static, _, _, _ = _unpack(ds[0])
            assert cond_static.shape == (COND_STATIC_DIM,)
            expected_bins = build_spatial_profile_bin_averages(
                ds.y_grid, spatial[0, :, 3].numpy()
            )
            np.testing.assert_allclose(
                cond_static[FORCING_SPATIAL_DESCRIPTOR_SLICE].numpy(),
                expected_bins,
                rtol=0.0,
                atol=0.0,
            )
            conds.append(cond_static)
            s_ys.append(spatial[:, :, 3])
        for other in conds[1:]:
            torch.testing.assert_close(conds[0][:2], other[:2])
            assert not torch.allclose(
                conds[0][FORCING_SPATIAL_DESCRIPTOR_SLICE],
                other[FORCING_SPATIAL_DESCRIPTOR_SLICE],
            )
        for other in s_ys[1:]:
            assert not torch.allclose(s_ys[0], other)


# ===================== collate_fn =====================


class TestCollateFn:
    """The dict collate stacks present keys and tolerates benchmarks that omit
    optional keys (e.g. source has no `forcing_seq`)."""

    def test_stacks_all_keys_with_forcing_seq(self, synthetic_trajectories,
                                              synthetic_sim_params):
        trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
        ds = SnapshotPairDataset(
            trajectories=trajectories, t_grid=t_grid, x_grid=x_grid, y_grid=y_grid,
            sim_ids=np.arange(5), sim_params=synthetic_sim_params,
            mu_global=_SYNTH_MU, sigma_global=_SYNTH_SIGMA,
            n_snapshots=6,
        )
        batch = [ds[i] for i in range(4)]
        out = collate_fn(batch)
        assert set(out.keys()) == {
            "spatial", "cond_static", "forcing_seq", "Y", "T_stats",
        }
        assert out["spatial"].shape[0] == 4
        assert out["forcing_seq"].shape[0] == 4
        assert out["T_stats"].shape == (4, 2)

    def test_omitted_optional_key_not_required(self):
        batch = [
            {"spatial": torch.zeros(3, 3, 6), "cond_static": torch.zeros(4),
             "Y": torch.zeros(3, 3, 1), "T_stats": torch.zeros(3)},
            {"spatial": torch.ones(3, 3, 6), "cond_static": torch.ones(4),
             "Y": torch.ones(3, 3, 1), "T_stats": torch.ones(3)},
        ]
        out = collate_fn(batch)
        assert "forcing_seq" not in out
        assert out["spatial"].shape == (2, 3, 3, 6)
        assert out["T_stats"].shape == (2, 3)
