import pytest

from src.physics.mms_2d import (
    run_mms_2d_interface,
    run_mms_2d_patch_source,
    run_mms_2d_smooth_forcing_integral,
    run_mms_2d_yflux,
    run_mms_x_linear,
    smooth_forcing_left_flux_sign_sample,
    space_order_test_2d_off_center_interface,
    space_order_test_2d_patch_source,
    space_order_test_2d_yflux,
    space_order_test_x_linear,
    time_order_test_2d_off_center_interface,
    time_order_test_2d_patch_source,
    time_order_test_2d_smooth_forcing_integral,
    time_order_test_2d_yflux,
    time_order_test_x_linear,
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


class TestYFluxMMS:
    """Verifies the vector-valued left Neumann flux path q_L(y, t) = a(t) s(y)."""

    def test_runs_at_default_grid(self):
        h, dt, max_err, l2_err = run_mms_2d_yflux(N=101, dt=0.0001)
        assert h > 0 and dt > 0
        assert max_err > 0 and l2_err > 0
        assert max_err < 1e-2
        assert l2_err < 5e-3

    def test_spatial_convergence_is_second_order(self):
        order = space_order_test_2d_yflux([21, 41, 81])
        assert order >= 1.90, f"expected >= 1.90, got {order:.3f}"

    def test_temporal_convergence_is_second_order(self):
        order = time_order_test_2d_yflux([0.02, 0.01, 0.005])
        assert order >= 1.90, f"expected >= 1.90, got {order:.3f}"


class TestXLinearMMS:
    """Verifies the y-direction stencil in isolation. The manufactured solution
    is linear in x (zero x-curvature) so spatial truncation comes only from the
    y-Laplacian; the observed spatial order is therefore the y-direction order.
    On the enforced isotropic grid this catches a y-Laplacian curvature bug, not
    an hx != hy bug."""

    def test_runs_at_default_grid(self):
        h, dt, max_err, l2_err = run_mms_x_linear(N=101, dt=0.0001)
        assert h > 0 and dt > 0
        assert max_err > 0 and l2_err > 0
        assert max_err < 1e-2
        assert l2_err < 5e-3

    def test_spatial_convergence_is_second_order(self):
        order = space_order_test_x_linear([21, 41, 81])
        assert order >= 1.90, f"expected >= 1.90, got {order:.3f}"

    def test_temporal_convergence_is_second_order(self):
        order = time_order_test_x_linear([0.02, 0.01, 0.005])
        assert order >= 1.90, f"expected >= 1.90, got {order:.3f}"


class TestSmoothForcingIntegralMMS:
    """Strict MMS for smooth-in-time left forcing through q_left_integral_fn."""

    def test_runs_at_default_grid(self):
        h, dt, max_err, l2_err = run_mms_2d_smooth_forcing_integral(N=101, dt=0.0001)
        assert h > 0 and dt > 0
        assert max_err > 0 and l2_err > 0
        assert max_err < 1e-2
        assert l2_err < 5e-3

    def test_temporal_convergence_is_second_order(self):
        order = time_order_test_2d_smooth_forcing_integral([0.02, 0.01, 0.005])
        assert order >= 1.85, f"expected >= 1.85, got {order:.3f}"

    def test_left_flux_sign_convention_matches_mms_derivative(self):
        q_left, minus_k_Tx, plus_k_Tx = smooth_forcing_left_flux_sign_sample()
        assert q_left == pytest.approx(minus_k_Tx, rel=0.0, abs=1e-12)
        assert q_left == pytest.approx(-plus_k_Tx, rel=0.0, abs=1e-12)


class TestPatchSourceMMS:
    """Verifies the internal volumetric source path (source benchmark) with a
    smooth Gaussian-envelope manufactured solution."""

    def test_runs_at_default_grid(self):
        h, dt, max_err, l2_err = run_mms_2d_patch_source(N=101, dt=0.0001)
        assert h > 0 and dt > 0
        assert max_err > 0 and l2_err > 0
        assert max_err < 1e-2
        assert l2_err < 5e-3

    def test_spatial_convergence_is_second_order(self):
        order = space_order_test_2d_patch_source([21, 41, 81])
        assert order >= 1.90, f"expected >= 1.90, got {order:.3f}"

    def test_temporal_convergence_is_second_order(self):
        order = time_order_test_2d_patch_source([0.04, 0.02, 0.01])
        assert order >= 1.85, f"expected >= 1.85, got {order:.3f}"
