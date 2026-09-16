import math

import pytest

from src.physics.mms_2d import (
    run_interface_trace_reconstruction_mms,
    run_interface_alignment_study,
    near_node_interface_x_I,
    patch_source_neumann_residual,
    run_mms_2d_interface,
    run_mms_2d_patch_source,
    run_mms_2d_smooth_forcing_integral,
    run_mms_2d_yflux,
    run_mms_x_linear,
    smooth_forcing_left_flux_sign_sample,
    space_order_test_2d_near_node_interface,
    space_order_test_2d_off_center_interface,
    space_order_test_2d_patch_source,
    space_order_test_2d_yflux,
    space_order_test_x_linear,
    time_order_test_2d_near_node_interface,
    time_order_test_2d_off_center_interface,
    time_order_test_2d_patch_source,
    time_order_test_2d_smooth_forcing_integral,
    time_order_test_2d_yflux,
    time_order_test_x_linear,
)


def test_interface_trace_reconstruction_is_second_order():
    coarse_h, coarse_error = run_interface_trace_reconstruction_mms(20)
    fine_h, fine_error = run_interface_trace_reconstruction_mms(40)
    order = math.log(coarse_error / fine_error) / math.log(coarse_h / fine_h)
    assert order > 1.9


pytestmark = pytest.mark.slow

MMS_DT_LIST = [0.0185, 0.00925, 0.004625]
MMS_PATCH_DT_LIST = [0.037, 0.0185, 0.00925]


@pytest.fixture(scope="module")
def interface_alignment_study():
    return run_interface_alignment_study()


@pytest.mark.parametrize("name, fraction", [
    ("controlled_left", 0.8),
    ("controlled_middle", 0.2),
    ("controlled_right", 0.6),
])
def test_controlled_interface_alignment_converges_pairwise(interface_alignment_study, name, fraction):
    case = next(case for case in interface_alignment_study["cases"] if case["name"] == name)
    records = case["records"]
    assert len({row["interface_x"] for row in records}) == 1
    for row in records:
        assert row["interface_fraction"] == pytest.approx(fraction, abs=1e-12)
    for coarse, fine in zip(records, records[1:]):
        assert fine["rms_error_K"] < coarse["rms_error_K"]
        assert 1.8 < fine["order_from_previous"] < 2.2


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
        order = time_order_test_2d_off_center_interface(MMS_DT_LIST, x_I=0.4734)
        assert order >= 1.90, f"expected >= 1.90, got {order:.3f}"


class TestNearNodeInterfaceMMS:
    """Interface placed 1% of a cell from a grid node — near-node stress for the
    asymmetric face conductance and control volumes. Geometry held fixed
    (h_min/hx = 0.01) across resolutions."""

    def test_runs_near_node(self):
        x_I = near_node_interface_x_I(N=100, eps=0.01)
        h, dt, max_err, l2_err = run_mms_2d_interface(N=100, dt=0.0001, x_I=x_I)
        assert h > 0 and dt > 0
        assert max_err > 0 and l2_err > 0
        assert max_err < 1e-2
        assert l2_err < 5e-3

    def test_spatial_convergence_is_second_order(self):
        order = space_order_test_2d_near_node_interface([50, 100, 200], eps=0.01)
        assert order >= 1.90, f"expected >= 1.90, got {order:.3f}"

    def test_temporal_convergence_is_second_order(self):
        order = time_order_test_2d_near_node_interface(MMS_DT_LIST, eps=0.01)
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
        order = time_order_test_2d_yflux(MMS_DT_LIST)
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
        order = time_order_test_x_linear(MMS_DT_LIST)
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
        order = time_order_test_2d_smooth_forcing_integral(MMS_DT_LIST)
        assert order >= 1.85, f"expected >= 1.85, got {order:.3f}"

    def test_left_flux_sign_convention_matches_mms_derivative(self):
        q_left, minus_k_Tx, plus_k_Tx = smooth_forcing_left_flux_sign_sample()
        assert q_left == pytest.approx(minus_k_Tx, rel=0.0, abs=1e-12)
        assert q_left == pytest.approx(-plus_k_Tx, rel=0.0, abs=1e-12)


class TestPatchSourceMMS:
    """Verifies the internal volumetric source path (source benchmark) with a
    smooth Gaussian-envelope manufactured solution."""

    def test_neumann_bc_satisfied_exactly(self):
        # F_y(x, c) and F_y(x, d) must vanish to roundoff (exact top/bottom
        # Neumann). This is the failure mode of the old isotropic Gaussian.
        res_c, res_d = patch_source_neumann_residual(N=101)
        assert res_c < 1e-10, f"F_y at y=c not zero: {res_c:.3e}"
        assert res_d < 1e-10, f"F_y at y=d not zero: {res_d:.3e}"

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
        order = time_order_test_2d_patch_source(MMS_PATCH_DT_LIST)
        assert order >= 1.85, f"expected >= 1.85, got {order:.3f}"
