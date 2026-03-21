import numpy as np
import pytest
import torch

from src.operators.losses import SpatiallyWeightedMSE, build_interface_mask, compute_interface_rel_l2


# ===================== SpatiallyWeightedMSE =====================

class TestSpatiallyWeightedMSE:
    @pytest.fixture
    def x_grid(self):
        return np.linspace(0.0, 1.0, 101).astype(np.float32)

    def test_uniform_weight_matches_mse(self, x_grid):
        """interface_weight=1.0 should produce the same result as plain MSE."""
        loss_fn = SpatiallyWeightedMSE(x_grid, interface_weight=1.0)
        mse_fn = torch.nn.MSELoss()

        y_pred = torch.randn(2, 101, 40, 1)
        y_true = torch.randn(2, 101, 40, 1)

        weighted = loss_fn(y_pred, y_true)
        plain = mse_fn(y_pred, y_true)
        assert weighted.item() == pytest.approx(plain.item(), rel=1e-5)

    def test_weight_normalization(self, x_grid):
        """Weights should have mean == 1.0 regardless of interface_weight."""
        for w in [1.0, 5.0, 10.0, 50.0]:
            loss_fn = SpatiallyWeightedMSE(x_grid, interface_weight=w)
            assert loss_fn.weights.mean().item() == pytest.approx(1.0, abs=1e-6)

    def test_higher_weight_increases_interface_loss(self, x_grid):
        """Error concentrated at the interface should produce higher loss with larger weight."""
        y_true = torch.zeros(1, 101, 10, 1)
        y_pred = torch.zeros(1, 101, 10, 1)
        # Place error only at the interface node (index 50 for x=0.5)
        y_pred[:, 50, :, :] = 1.0

        loss_w1 = SpatiallyWeightedMSE(x_grid, interface_weight=1.0)(y_pred, y_true)
        loss_w10 = SpatiallyWeightedMSE(x_grid, interface_weight=10.0)(y_pred, y_true)
        assert loss_w10.item() > loss_w1.item()

    def test_buffer_moves_to_device(self, x_grid):
        """The weight buffer should move with .to(device)."""
        loss_fn = SpatiallyWeightedMSE(x_grid)
        device = torch.device("cpu")
        loss_fn = loss_fn.to(device)
        assert loss_fn.weights.device == device

    def test_weight_shape(self, x_grid):
        """Weights should broadcast against (B, Nx, H, 1)."""
        loss_fn = SpatiallyWeightedMSE(x_grid)
        assert loss_fn.weights.shape == (1, 101, 1, 1)

    def test_gradient_flows(self, x_grid):
        """Loss should be differentiable."""
        loss_fn = SpatiallyWeightedMSE(x_grid, interface_weight=10.0)
        y_pred = torch.randn(2, 101, 10, 1, requires_grad=True)
        y_true = torch.randn(2, 101, 10, 1)
        loss = loss_fn(y_pred, y_true)
        loss.backward()
        assert y_pred.grad is not None
        assert y_pred.grad.shape == y_pred.shape


# ===================== build_interface_mask =====================

class TestBuildInterfaceMask:
    def test_correct_node_count(self):
        """Mask should select the right number of nodes within ±half_width."""
        x_grid = np.linspace(0.0, 1.0, 101).astype(np.float32)
        mask = build_interface_mask(x_grid, interface_x=0.5, interface_half_width=0.05)
        # Float32 rounding at boundaries (0.45/0.55) may exclude edge nodes.
        # With dx=0.01 and half_width=0.05, expect ~9-11 nodes.
        assert 9 <= mask.sum().item() <= 11

    def test_mask_dtype_is_bool(self):
        x_grid = np.linspace(0.0, 1.0, 11).astype(np.float32)
        mask = build_interface_mask(x_grid)
        assert mask.dtype == torch.bool

    def test_all_false_when_no_overlap(self):
        """If interface is outside the grid, mask should be all False."""
        x_grid = np.linspace(0.0, 0.3, 31).astype(np.float32)
        mask = build_interface_mask(x_grid, interface_x=0.5, interface_half_width=0.05)
        assert mask.sum().item() == 0

    def test_mask_shape(self):
        x_grid = np.linspace(0.0, 1.0, 101).astype(np.float32)
        mask = build_interface_mask(x_grid)
        assert mask.shape == (101,)


# ===================== compute_interface_rel_l2 =====================

class TestComputeInterfaceRelL2:
    def test_perfect_prediction_gives_zero(self):
        """Zero error should give 0% rel L2."""
        mask = torch.tensor([False, True, True, False, False])
        y = torch.randn(2, 5, 10, 1)
        result = compute_interface_rel_l2(y, y, mask)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_nonzero_error(self):
        """Non-zero error should give positive rel L2."""
        mask = torch.tensor([False, True, True, False, False])
        y_true = torch.ones(2, 5, 10, 1)
        y_pred = torch.ones(2, 5, 10, 1) * 1.1
        result = compute_interface_rel_l2(y_pred, y_true, mask)
        assert result > 0.0

    def test_returns_percentage(self):
        """Result should be in percent (multiplied by 100)."""
        mask = torch.tensor([False, True, True, False, False])
        y_true = torch.ones(2, 5, 10, 1)
        # 10% error everywhere
        y_pred = torch.ones(2, 5, 10, 1) * 1.1
        result = compute_interface_rel_l2(y_pred, y_true, mask)
        assert result == pytest.approx(10.0, rel=0.01)

    def test_only_uses_masked_nodes(self):
        """Errors outside the mask should not affect the result."""
        mask = torch.tensor([False, True, False])
        y_true = torch.ones(1, 3, 5, 1)
        y_pred = y_true.clone()
        # Large error only outside the mask
        y_pred[:, 0, :, :] = 100.0
        y_pred[:, 2, :, :] = 100.0
        result = compute_interface_rel_l2(y_pred, y_true, mask)
        assert result == pytest.approx(0.0, abs=1e-6)
