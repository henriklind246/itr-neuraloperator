import numpy as np
import pytest
import torch

from src.operators.losses import (
    EPS_JUMP,
    EPS_STD,
    SpatiallyWeightedMSE,
    build_interface_band,
    build_interface_mask,
    compute_interface_rel_l2,
    get_batch_interface_x,
    interface_flanking_nodes,
    interface_flanking_nodes_per_sample,
    per_sample_contact_jump_rmse,
    per_sample_node_jump_errors,
    per_sample_nrmse,
    per_sample_sq_rms,
    tail_stats,
)


# ===================== SpatiallyWeightedMSE =====================

class TestSpatiallyWeightedMSE:
    @pytest.fixture
    def x_grid(self):
        return np.linspace(0.0, 1.0, 101).astype(np.float32)

    @pytest.fixture
    def y_grid(self):
        return np.linspace(0.0, 1.0, 21).astype(np.float32)

    def test_uniform_weight_matches_mse(self, x_grid, y_grid):
        """interface_weight=1.0 should produce the same result as plain MSE."""
        loss_fn = SpatiallyWeightedMSE(x_grid, y_grid, interface_weight=1.0)
        mse_fn = torch.nn.MSELoss()

        y_pred = torch.randn(2, 101, 21, 1)
        y_true = torch.randn(2, 101, 21, 1)

        weighted = loss_fn(y_pred, y_true)
        plain = mse_fn(y_pred, y_true)
        assert weighted.item() == pytest.approx(plain.item(), rel=1e-5)

    def test_weight_normalization(self, x_grid, y_grid):
        """Weights should have mean == 1.0 regardless of interface_weight."""
        for w in [1.0, 5.0, 10.0, 50.0]:
            loss_fn = SpatiallyWeightedMSE(x_grid, y_grid, interface_weight=w)
            assert loss_fn.weights.mean().item() == pytest.approx(1.0, abs=1e-6)

    def test_higher_weight_increases_interface_loss(self, x_grid, y_grid):
        """Error concentrated at the interface should produce higher loss with larger weight."""
        y_true = torch.zeros(1, 101, 21, 1)
        y_pred = torch.zeros(1, 101, 21, 1)
        # Place error only at the interface column (x=0.5 -> index 50)
        y_pred[:, 50, :, :] = 1.0

        loss_w1 = SpatiallyWeightedMSE(x_grid, y_grid, interface_weight=1.0)(y_pred, y_true)
        loss_w10 = SpatiallyWeightedMSE(x_grid, y_grid, interface_weight=10.0)(y_pred, y_true)
        assert loss_w10.item() > loss_w1.item()

    def test_buffer_moves_to_device(self, x_grid, y_grid):
        """The weight buffer should move with .to(device)."""
        loss_fn = SpatiallyWeightedMSE(x_grid, y_grid)
        device = torch.device("cpu")
        loss_fn = loss_fn.to(device)
        assert loss_fn.weights.device == device

    def test_weight_shape(self, x_grid, y_grid):
        """Weights should broadcast against (B, Nx, Ny, 1)."""
        loss_fn = SpatiallyWeightedMSE(x_grid, y_grid)
        assert loss_fn.weights.shape == (1, 101, 21, 1)

    def test_weights_constant_along_y(self, x_grid, y_grid):
        """Interface is a vertical slab: weights must be invariant along the y axis."""
        loss_fn = SpatiallyWeightedMSE(x_grid, y_grid, interface_weight=10.0)
        w = loss_fn.weights[0, :, :, 0]  # (Nx, Ny)
        # Every row (fixed x) should be constant across y
        row_std = w.std(dim=1)
        assert torch.all(row_std < 1e-6)

    def test_gradient_flows(self, x_grid, y_grid):
        """Loss should be differentiable."""
        loss_fn = SpatiallyWeightedMSE(x_grid, y_grid, interface_weight=10.0)
        y_pred = torch.randn(2, 101, 21, 1, requires_grad=True)
        y_true = torch.randn(2, 101, 21, 1)
        loss = loss_fn(y_pred, y_true)
        loss.backward()
        assert y_pred.grad is not None
        assert y_pred.grad.shape == y_pred.shape


# ===================== build_interface_band (per-sample) =====================

class TestBuildInterfaceBand:
    @pytest.fixture
    def x_grid_t(self):
        return torch.linspace(0.0, 1.0, 101)

    def test_band_true_within_half_width(self, x_grid_t):
        """Band True exactly where |x - interface_x| <= half_width."""
        interface_x = torch.tensor([0.5])
        hw = 0.05
        band = build_interface_band(x_grid_t, interface_x, hw)  # (1, Nx)
        assert band.shape == (1, 101)
        assert band.dtype == torch.bool
        expected = (torch.abs(x_grid_t - 0.5) <= hw)
        assert torch.equal(band[0], expected)

    def test_band_shifts_with_interface_x(self, x_grid_t):
        """Center of the band tracks interface_x per sample."""
        hw = 0.05
        for cx in [0.2, 0.35, 0.8]:
            band = build_interface_band(x_grid_t, torch.tensor([cx]), hw)[0]
            true_idx = torch.nonzero(band).flatten()
            center_x = x_grid_t[true_idx].mean().item()
            assert center_x == pytest.approx(cx, abs=hw)

    def test_band_batched(self, x_grid_t):
        """Each row of the batch follows its own interface_x."""
        interface_x = torch.tensor([0.2, 0.5, 0.8])
        hw = 0.05
        band = build_interface_band(x_grid_t, interface_x, hw)  # (3, Nx)
        assert band.shape == (3, 101)
        for b, cx in enumerate([0.2, 0.5, 0.8]):
            expected = (torch.abs(x_grid_t - cx) <= hw)
            assert torch.equal(band[b], expected)


# ===================== SpatiallyWeightedMSE per-sample forward =====================

class TestSpatiallyWeightedMSEPerSample:
    @pytest.fixture
    def x_grid(self):
        return np.linspace(0.0, 1.0, 101).astype(np.float32)

    @pytest.fixture
    def y_grid(self):
        return np.linspace(0.0, 1.0, 21).astype(np.float32)

    def test_dynamic_half_matches_fixed_buffer(self, x_grid, y_grid):
        """Per-sample interface_x == 0.5 must match the fixed-buffer path numerically."""
        loss_fn = SpatiallyWeightedMSE(
            x_grid, y_grid, interface_x=0.5, interface_half_width=0.05, interface_weight=10.0
        )
        torch.manual_seed(0)
        y_pred = torch.randn(4, 101, 21, 1)
        y_true = torch.randn(4, 101, 21, 1)

        fixed = loss_fn(y_pred, y_true)  # interface_x=None -> fixed buffer
        iface_x = torch.full((4,), 0.5)
        dynamic = loss_fn(y_pred, y_true, iface_x)
        torch.testing.assert_close(dynamic, fixed)

    def test_per_sample_weights_mean_one(self, x_grid, y_grid):
        """Per-sample weights normalize to mean 1.0 for each sample independently."""
        loss_fn = SpatiallyWeightedMSE(
            x_grid, y_grid, interface_half_width=0.05, interface_weight=7.0
        )
        iface_x = torch.tensor([0.2, 0.5, 0.8])
        band = build_interface_band(loss_fn.x_grid_t, iface_x, loss_fn.interface_half_width)
        raw = 1.0 + (loss_fn.interface_weight - 1.0) * band.float()
        w = raw / raw.mean(dim=1, keepdim=True)
        means = w.mean(dim=1)
        assert torch.allclose(means, torch.ones_like(means), atol=1e-6)

    def test_none_path_is_fixed_buffer(self, x_grid, y_grid):
        """interface_x=None is the byte-identical regression guard (fixed buffer)."""
        loss_fn = SpatiallyWeightedMSE(x_grid, y_grid, interface_weight=10.0)
        torch.manual_seed(1)
        y_pred = torch.randn(3, 101, 21, 1)
        y_true = torch.randn(3, 101, 21, 1)
        expected = torch.mean(loss_fn.weights * (y_pred - y_true) ** 2)
        assert loss_fn(y_pred, y_true).item() == expected.item()

    def test_per_sample_broadcasts_against_output(self, x_grid, y_grid):
        """Per-sample forward returns a scalar over (B, Nx, Ny, 1) predictions."""
        loss_fn = SpatiallyWeightedMSE(x_grid, y_grid, interface_weight=10.0)
        y_pred = torch.randn(5, 101, 21, 1)
        y_true = torch.randn(5, 101, 21, 1)
        iface_x = torch.linspace(0.2, 0.8, 5)
        out = loss_fn(y_pred, y_true, iface_x)
        assert out.dim() == 0
        assert out.item() >= 0.0

    def test_works_after_to_device(self, x_grid, y_grid):
        """x_grid_t buffer moves with .to(device); per-sample forward still runs."""
        loss_fn = SpatiallyWeightedMSE(x_grid, y_grid, interface_weight=10.0).to("cpu")
        assert loss_fn.x_grid_t.device == torch.device("cpu")
        y_pred = torch.randn(2, 101, 21, 1)
        y_true = torch.randn(2, 101, 21, 1)
        iface_x = torch.tensor([0.3, 0.7])
        out = loss_fn(y_pred, y_true, iface_x)
        assert torch.isfinite(out)


# ===================== get_batch_interface_x (selector) =====================

class TestGetBatchInterfaceX:
    def test_returns_none_when_flag_off(self):
        """Flag off keeps source/forcing on the fixed path even with slot 2 present."""
        batch = {"T_stats": torch.tensor([[1.0, 2.0, 0.5], [1.0, 2.0, 0.5]])}
        out = get_batch_interface_x(batch, torch.device("cpu"), use_per_sample_interface=False)
        assert out is None

    def test_returns_none_when_no_slot2(self):
        """forcing has t_stats_dim=2; no slot 2 -> None even with flag on."""
        batch = {"T_stats": torch.tensor([[1.0, 2.0], [1.0, 2.0]])}
        out = get_batch_interface_x(batch, torch.device("cpu"), use_per_sample_interface=True)
        assert out is None

    def test_returns_slot2_when_flag_on(self):
        """interfaces: flag on AND slot 2 present -> per-sample interface_x vector."""
        batch = {"T_stats": torch.tensor([[1.0, 2.0, 0.3], [1.0, 2.0, 0.7]])}
        out = get_batch_interface_x(batch, torch.device("cpu"), use_per_sample_interface=True)
        assert out is not None
        assert torch.equal(out, torch.tensor([0.3, 0.7]))

    def test_returns_none_when_no_t_stats(self):
        out = get_batch_interface_x({}, torch.device("cpu"), use_per_sample_interface=True)
        assert out is None


# ===================== build_interface_mask =====================

class TestBuildInterfaceMask:
    def test_correct_column_count(self):
        """Mask should select the right number of x-columns within ±half_width."""
        x_grid = np.linspace(0.0, 1.0, 101).astype(np.float32)
        y_grid = np.linspace(0.0, 1.0, 21).astype(np.float32)
        mask = build_interface_mask(x_grid, y_grid, interface_x=0.5, interface_half_width=0.05)
        # Per-column count along x (any row): ~9-11 nodes given float32 rounding.
        col_count = mask[:, 0].sum().item()
        assert 9 <= col_count <= 11
        # Total True cells: col_count * Ny
        assert mask.sum().item() == col_count * len(y_grid)

    def test_mask_dtype_is_bool(self):
        x_grid = np.linspace(0.0, 1.0, 11).astype(np.float32)
        y_grid = np.linspace(0.0, 1.0, 7).astype(np.float32)
        mask = build_interface_mask(x_grid, y_grid)
        assert mask.dtype == torch.bool

    def test_all_false_when_no_overlap(self):
        """If interface is outside the grid, mask should be all False."""
        x_grid = np.linspace(0.0, 0.3, 31).astype(np.float32)
        y_grid = np.linspace(0.0, 1.0, 11).astype(np.float32)
        mask = build_interface_mask(x_grid, y_grid, interface_x=0.5, interface_half_width=0.05)
        assert mask.sum().item() == 0

    def test_mask_shape(self):
        x_grid = np.linspace(0.0, 1.0, 101).astype(np.float32)
        y_grid = np.linspace(0.0, 1.0, 21).astype(np.float32)
        mask = build_interface_mask(x_grid, y_grid)
        assert mask.shape == (101, 21)

    def test_vertical_slab_geometry(self):
        """Each column (fixed x) in the interface band must be fully True."""
        x_grid = np.linspace(0.0, 1.0, 101).astype(np.float32)
        y_grid = np.linspace(0.0, 1.0, 21).astype(np.float32)
        mask = build_interface_mask(x_grid, y_grid, interface_x=0.5, interface_half_width=0.05)
        # A column is either entirely True or entirely False
        col_sums = mask.sum(dim=1)
        for s in col_sums:
            assert int(s) in (0, len(y_grid))


# ===================== compute_interface_rel_l2 =====================

class TestComputeInterfaceRelL2:
    def test_perfect_prediction_gives_zero(self):
        """Zero error should give 0% rel L2."""
        mask = torch.zeros(5, 3, dtype=torch.bool)
        mask[1, :] = True
        mask[2, :] = True
        y = torch.randn(2, 5, 3, 1)
        result = compute_interface_rel_l2(y, y, mask)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_nonzero_error(self):
        """Non-zero error should give positive rel L2."""
        mask = torch.zeros(5, 3, dtype=torch.bool)
        mask[1, :] = True
        mask[2, :] = True
        y_true = torch.ones(2, 5, 3, 1)
        y_pred = torch.ones(2, 5, 3, 1) * 1.1
        result = compute_interface_rel_l2(y_pred, y_true, mask)
        assert result > 0.0

    def test_returns_percentage(self):
        """Result should be in percent (multiplied by 100)."""
        mask = torch.zeros(5, 3, dtype=torch.bool)
        mask[1, :] = True
        mask[2, :] = True
        y_true = torch.ones(2, 5, 3, 1)
        # 10% error everywhere
        y_pred = torch.ones(2, 5, 3, 1) * 1.1
        result = compute_interface_rel_l2(y_pred, y_true, mask)
        assert result == pytest.approx(10.0, rel=0.01)

    def test_only_uses_masked_nodes(self):
        """Errors outside the mask should not affect the result."""
        mask = torch.zeros(3, 2, dtype=torch.bool)
        mask[1, :] = True
        y_true = torch.ones(1, 3, 2, 1)
        y_pred = y_true.clone()
        # Large error only outside the masked column
        y_pred[:, 0, :, :] = 100.0
        y_pred[:, 2, :, :] = 100.0
        result = compute_interface_rel_l2(y_pred, y_true, mask)
        assert result == pytest.approx(0.0, abs=1e-6)


# ===================== interface_flanking_nodes =====================

class TestInterfaceFlankingNodes:
    def test_fixed_midpoint(self):
        """Interface at 0.5 on a 101-node grid flanks nodes 49/50 or 50/51."""
        x_grid = np.linspace(0.0, 1.0, 101).astype(np.float32)
        left, right = interface_flanking_nodes(x_grid, 0.5)
        assert right == left + 1
        assert x_grid[left] <= 0.5 <= x_grid[right]

    def test_left_right_straddle_interface(self):
        """For several interface_x the flanking pair must straddle it."""
        x_grid = np.linspace(0.0, 1.0, 101).astype(np.float32)
        for cx in [0.2, 0.37, 0.63, 0.8]:
            left, right = interface_flanking_nodes(x_grid, cx)
            assert x_grid[left] <= cx <= x_grid[right]
            assert right == left + 1

    def test_outside_grid_raises(self):
        x_grid = np.linspace(0.0, 0.3, 31).astype(np.float32)
        with pytest.raises(ValueError):
            interface_flanking_nodes(x_grid, 0.9)

    def test_per_sample_matches_scalar(self):
        """Vectorized per-sample flanking nodes match the scalar form row-wise."""
        x_grid = np.linspace(0.0, 1.0, 101).astype(np.float32)
        x_grid_t = torch.as_tensor(x_grid)
        centers = [0.2, 0.5, 0.8]
        left_v, right_v = interface_flanking_nodes_per_sample(
            x_grid_t, torch.tensor(centers)
        )
        for i, cx in enumerate(centers):
            ls, rs = interface_flanking_nodes(x_grid, cx)
            assert int(left_v[i]) == ls
            assert int(right_v[i]) == rs


# ===================== per_sample nRMSE (headline) =====================

class TestPerSampleNRMSE:
    def test_normalization_invariance(self):
        """Scaling pred and true by the same constant leaves nRMSE unchanged."""
        torch.manual_seed(0)
        y_true = torch.randn(4, 21, 11, 1) * 3.0 + 7.0
        y_pred = y_true + 0.1 * torch.randn(4, 21, 11, 1)
        base = per_sample_nrmse(y_pred, y_true)
        for c in [2.0, 50.0, 300.0]:
            scaled = per_sample_nrmse(y_pred * c, y_true * c)
            torch.testing.assert_close(scaled, base, rtol=1e-5, atol=1e-7)

    def test_eps_std_floor_keeps_finite(self):
        """A near-constant target (std≈0) must not blow up nRMSE."""
        y_true = torch.full((2, 21, 11, 1), 5.0)
        y_pred = y_true + 1e-3
        out = per_sample_nrmse(y_pred, y_true)
        assert torch.all(torch.isfinite(out))
        # Denominator is floored at EPS_STD, so nrmse ≈ rms / EPS_STD.
        rms = per_sample_sq_rms(y_pred, y_true)
        torch.testing.assert_close(out, rms / EPS_STD, rtol=1e-4, atol=1e-4)

    def test_perfect_prediction_is_zero(self):
        y = torch.randn(3, 21, 11, 1)
        out = per_sample_nrmse(y, y)
        assert torch.allclose(out, torch.zeros(3), atol=1e-7)

    def test_matches_manual_definition(self):
        torch.manual_seed(1)
        y_true = torch.randn(5, 9, 7, 1) * 2.0
        y_pred = y_true + 0.2 * torch.randn(5, 9, 7, 1)
        rms = torch.sqrt(((y_pred - y_true) ** 2).mean(dim=(1, 2, 3)))
        std = y_true.std(dim=(1, 2, 3), unbiased=False)
        expected = rms / std.clamp_min(EPS_STD)
        torch.testing.assert_close(per_sample_nrmse(y_pred, y_true), expected)


# ===================== node-jump errors (interface fidelity) =====================

class TestPerSampleNodeJumpErrors:
    def test_non_circular_uses_own_fields(self):
        """Predicted jump comes from y_pred, true jump from y_true."""
        B, Nx, Ny = 2, 11, 5
        y_true = torch.zeros(B, Nx, Ny, 1)
        y_pred = torch.zeros(B, Nx, Ny, 1)
        left, right = 4, 5
        # Build a known step across the interface for each field.
        y_true[:, right] = 2.0
        y_pred[:, right] = 2.5  # pred over-predicts the jump by 0.5
        err_rms, true_rms = per_sample_node_jump_errors(y_pred, y_true, left, right)
        torch.testing.assert_close(err_rms, torch.full((B,), 0.5))
        torch.testing.assert_close(true_rms, torch.full((B,), 2.0))

    def test_zero_error_when_fields_equal(self):
        y = torch.randn(3, 11, 5, 1)
        err_rms, true_rms = per_sample_node_jump_errors(y, y, 4, 5)
        assert torch.allclose(err_rms, torch.zeros(3), atol=1e-7)

    def test_per_sample_left_right_tensor(self):
        """Long-tensor (per-sample) flanking nodes select per-row columns."""
        B, Nx, Ny = 3, 11, 4
        y_true = torch.zeros(B, Nx, Ny, 1)
        y_pred = torch.zeros(B, Nx, Ny, 1)
        left = torch.tensor([2, 4, 6])
        right = left + 1
        for b in range(B):
            y_true[b, right[b]] = 1.0
            y_pred[b, right[b]] = 1.0 + 0.3 * (b + 1)
        err_rms, _ = per_sample_node_jump_errors(y_pred, y_true, left, right)
        torch.testing.assert_close(
            err_rms, torch.tensor([0.3, 0.6, 0.9])
        )

    def test_normalized_jump_offset_free(self):
        """A constant additive offset on both fields leaves the jump error unchanged."""
        B, Nx, Ny = 2, 11, 5
        y_true = torch.randn(B, Nx, Ny, 1)
        y_pred = torch.randn(B, Nx, Ny, 1)
        e0, t0 = per_sample_node_jump_errors(y_pred, y_true, 4, 5)
        e1, t1 = per_sample_node_jump_errors(y_pred + 300.0, y_true + 300.0, 4, 5)
        torch.testing.assert_close(e0, e1)
        torch.testing.assert_close(t0, t1)


# ===================== contact-jump RMSE (source_itr) =====================

class TestPerSampleContactJumpRMSE:
    def test_matches_weighted_reference(self):
        """Contact jump equals the R_c(y)·G(y)-weighted node-jump difference."""
        B, Nx, Ny = 2, 11, 6
        left, right = 4, 5
        torch.manual_seed(0)
        y_true = torch.randn(B, Nx, Ny, 1)
        y_pred = torch.randn(B, Nx, Ny, 1)
        weight_y = torch.linspace(0.5, 2.0, Ny)  # stand-in for R_c(y)·G(y)
        out = per_sample_contact_jump_rmse(y_pred, y_true, left, right, weight_y)

        pl, pr = y_pred[:, left, :, 0], y_pred[:, right, :, 0]
        tl, tr = y_true[:, left, :, 0], y_true[:, right, :, 0]
        cj_pred = weight_y * (pl - pr)
        cj_true = weight_y * (tl - tr)
        expected = torch.sqrt(((cj_pred - cj_true) ** 2).mean(dim=1))
        torch.testing.assert_close(out, expected)

    def test_zero_weight_gives_zero(self):
        B, Nx, Ny = 2, 11, 6
        y_true = torch.randn(B, Nx, Ny, 1)
        y_pred = torch.randn(B, Nx, Ny, 1)
        out = per_sample_contact_jump_rmse(
            y_pred, y_true, 4, 5, torch.zeros(Ny)
        )
        assert torch.allclose(out, torch.zeros(B), atol=1e-7)


# ===================== aggregation: train-mean == val/test mean-of-pairs =====================

class TestMeanOfPairsAggregation:
    def test_summed_per_sample_equals_concat_mean(self):
        """Σ metric_i / n over minibatches equals the mean over all pairs."""
        torch.manual_seed(0)
        y_true = torch.randn(12, 9, 7, 1)
        y_pred = y_true + 0.1 * torch.randn(12, 9, 7, 1)

        # Pooled-over-all-pairs reference (val/test convention).
        all_nrmse = per_sample_nrmse(y_pred, y_true)
        pooled_mean = all_nrmse.mean()

        # Train convention: sum per-sample over minibatches, divide by n.
        running_sum = 0.0
        n = 0
        for start in range(0, 12, 4):
            sl = slice(start, start + 4)
            vals = per_sample_nrmse(y_pred[sl], y_true[sl])
            running_sum += float(vals.sum())
            n += vals.numel()
        train_mean = running_sum / n
        assert train_mean == pytest.approx(float(pooled_mean), rel=1e-6)

    def test_differs_from_pooled_rmse(self):
        """Mean-of-per-sample-RMSE is not the same as a single pooled RMSE."""
        torch.manual_seed(3)
        y_true = torch.zeros(8, 9, 7, 1)
        # Per-sample error magnitudes span two orders so the two aggregations
        # diverge (pooled RMSE is dominated by the largest-error samples).
        err_scale = torch.logspace(-1, 2, 8).reshape(-1, 1, 1, 1)
        y_pred = y_true + err_scale * torch.randn(8, 9, 7, 1)
        per_sample = per_sample_sq_rms(y_pred, y_true).mean()
        pooled = torch.sqrt(((y_pred - y_true) ** 2).mean())
        assert abs(float(per_sample) - float(pooled)) > 1.0


# ===================== tail_stats =====================

class TestTailStats:
    def test_matches_numpy_percentile(self):
        torch.manual_seed(0)
        v = torch.rand(1000)
        out = tail_stats(v)
        vn = v.double().numpy()
        assert out["mean"] == pytest.approx(float(np.mean(vn)), rel=1e-6)
        assert out["max"] == pytest.approx(float(np.max(vn)), rel=1e-6)
        for q, key in [(25, "p25"), (50, "p50"), (75, "p75"), (90, "p90"), (99, "p99")]:
            assert out[key] == pytest.approx(
                float(np.percentile(vn, q, method="linear")), rel=1e-6
            )
        assert out["iqr"] == pytest.approx(out["p75"] - out["p25"], rel=1e-6)

    def test_empty_returns_zeros(self):
        out = tail_stats(torch.empty(0))
        assert out == {
            "mean": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "iqr": 0.0,
            "p90": 0.0, "p99": 0.0, "max": 0.0,
        }

    def test_ordering_mean_le_p90_le_p99_le_max(self):
        v = torch.rand(500)
        out = tail_stats(v)
        assert out["p25"] <= out["p50"] <= out["p75"] <= out["p90"] <= out["p99"] <= out["max"]
        assert out["mean"] <= out["max"]
        assert out["iqr"] >= 0.0


# ===================== gnrmse_pct identities (dimensionless restatements) =====================

class TestGnrmsePctIdentities:
    def test_gnrmse_pct_equals_rmse_K_over_sigma(self):
        """gnrmse_pct == rmse_K / sigma_global * 100 (unit restatement of rmse_K)."""
        torch.manual_seed(0)
        sigma_global = 4.2
        y_true = torch.randn(6, 9, 7, 1) * 2.0 + 5.0
        y_pred = y_true + 0.1 * torch.randn(6, 9, 7, 1)

        rms_i = per_sample_sq_rms(y_pred, y_true)
        gnrmse_pct = rms_i * 100.0
        rmse_K = rms_i * sigma_global
        torch.testing.assert_close(gnrmse_pct, rmse_K / sigma_global * 100.0)

    def test_node_jump_gnrmse_pct_equals_jump_rmse_K_over_sigma(self):
        """node_jump_gnrmse_pct == node_jump_rmse_K / sigma_global * 100."""
        torch.manual_seed(1)
        sigma_global = 3.7
        y_true = torch.randn(5, 11, 5, 1)
        y_pred = y_true + 0.2 * torch.randn(5, 11, 5, 1)
        left, right = 4, 5

        err_rms_i, _ = per_sample_node_jump_errors(y_pred, y_true, left, right)
        node_jump_gnrmse_pct = err_rms_i * 100.0
        node_jump_rmse_K = err_rms_i * sigma_global
        torch.testing.assert_close(
            node_jump_gnrmse_pct, node_jump_rmse_K / sigma_global * 100.0
        )
