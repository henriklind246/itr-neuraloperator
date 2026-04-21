import pytest

from src.physics.mms_2d import (
    run_mms_2d_interface,
    space_order_test_2d_off_center_interface,
    time_order_test_2d_off_center_interface,
)


pytestmark = pytest.mark.slow


class TestOffCenterInterfaceMMS:
    def test_runs_at_off_center_x_I(self):
        # Smoke: x_I=0.4734 with Nx=100 gives face_idx=46, h_L/hx ~ 0.87
        h, dt, max_err, l2_err = run_mms_2d_interface(N=100, dt=0.0001, x_I=0.4734)
        assert h > 0 and dt > 0
        assert max_err > 0 and l2_err > 0
        assert max_err < 1e-2
        assert l2_err < 5e-3

    def test_spatial_convergence_is_second_order(self):
        order = space_order_test_2d_off_center_interface([50, 100, 200], x_I=0.4734)
        assert order >= 1.90, f"expected >= 1.90, got {order:.3f}"

    def test_temporal_convergence_is_second_order(self):
        order = time_order_test_2d_off_center_interface([0.02, 0.01, 0.005], x_I=0.4734)
        assert order >= 1.90, f"expected >= 1.90, got {order:.3f}"
