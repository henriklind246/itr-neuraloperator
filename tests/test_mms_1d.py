import numpy as np
import pytest

from src.physics.mms_1d import run_mms_once, run_mms_interface


pytestmark = pytest.mark.slow


class TestRunMMSOnce:
    def test_returns_four_floats(self):
        result = run_mms_once(51)
        assert len(result) == 4
        assert all(isinstance(v, float) for v in result)

    def test_errors_are_small(self):
        _, _, max_err, l2_err = run_mms_once(101)
        assert max_err < 1e-2
        assert l2_err < 5e-3

    def test_error_decreases_with_refinement(self):
        _, _, _, l2_coarse = run_mms_once(51)
        _, _, _, l2_fine = run_mms_once(101)
        assert l2_fine < l2_coarse

    def test_positive_errors(self):
        _, _, max_err, l2_err = run_mms_once(51)
        assert max_err > 0
        assert l2_err > 0


class TestSpaceOrderTest:
    def test_approximately_two(self):
        """Verify spatial convergence order ≈ 2.

        Uses a small fixed dt so temporal error is negligible,
        and four grid levels to estimate mean order.
        """
        N_list = [21, 41, 81, 161]
        fixed_dt = 0.0005
        hs, errs = [], []
        for N in N_list:
            h, _, _, l2 = run_mms_once(N, dt=fixed_dt)
            hs.append(h)
            errs.append(l2)
        orders = []
        for i in range(len(errs) - 1):
            p = np.log(errs[i] / errs[i + 1]) / np.log(hs[i] / hs[i + 1])
            orders.append(p)
        mean_order = np.mean(orders)
        assert 1.9 < mean_order < 2.1, f"Expected order ~2, got {mean_order:.3f}"


class TestTimeOrderTest:
    def test_approximately_two(self):
        """Verify temporal convergence order ≈ 2.

        Uses a fine grid (N=801) so spatial error is negligible,
        and three dt values to estimate mean temporal order.

        Note: the Neumann BC evaluates flux at t^{n+1} rather than
        t^{n+1/2}, but this does not degrade the observed order because
        the boundary node is a single point in the L2 norm.
        Stops at dt=0.005 to avoid spatial-floor contamination.
        """
        N_fixed = 801
        dt_list = [0.02, 0.01, 0.005]
        errs = []
        for dt in dt_list:
            _, _, _, l2 = run_mms_once(N_fixed, dt=dt)
            errs.append(l2)
        orders = []
        for i in range(len(errs) - 1):
            p = np.log(errs[i] / errs[i + 1]) / np.log(dt_list[i] / dt_list[i + 1])
            orders.append(p)
        mean_order = np.mean(orders)
        assert 1.9 < mean_order < 2.1, f"Expected order ~2, got {mean_order:.3f}"


class TestConvergenceTableOutput:
    """Print convergence tables for audit / CI logs.

    These tests always pass; their purpose is to capture the raw
    convergence data in the test output so it can be inspected.
    Run with ``pytest -s`` to see the tables.
    """

    def test_print_spatial_convergence_table(self):
        """Spatial convergence table for single-layer MMS."""
        N_list = [21, 41, 81, 161]
        fixed_dt = 0.0005
        hs, errs = [], []
        for N in N_list:
            h, _, _, l2 = run_mms_once(N, dt=fixed_dt)
            hs.append(h)
            errs.append(l2)

        print("\n--- Single-layer spatial convergence (dt=0.0005) ---")
        print(f"{'N':>6}  {'h':>12}  {'L2 error':>12}  {'order':>8}")
        print("-" * 46)
        print(f"{N_list[0]:>6}  {hs[0]:>12.6f}  {errs[0]:>12.4e}  {'--':>8}")
        for i in range(1, len(N_list)):
            p = np.log(errs[i - 1] / errs[i]) / np.log(hs[i - 1] / hs[i])
            print(f"{N_list[i]:>6}  {hs[i]:>12.6f}  {errs[i]:>12.4e}  {p:>8.3f}")

    def test_print_temporal_convergence_table(self):
        """Temporal convergence table for single-layer MMS."""
        dt_list = [0.02, 0.01, 0.005]
        N_fixed = 801
        errs = []
        for dt in dt_list:
            _, _, _, l2 = run_mms_once(N_fixed, dt=dt)
            errs.append(l2)

        print("\n--- Single-layer temporal convergence (N=801) ---")
        print(f"{'dt':>10}  {'L2 error':>12}  {'order':>8}")
        print("-" * 34)
        print(f"{dt_list[0]:>10.4f}  {errs[0]:>12.4e}  {'--':>8}")
        for i in range(1, len(dt_list)):
            p = np.log(errs[i - 1] / errs[i]) / np.log(dt_list[i - 1] / dt_list[i])
            print(f"{dt_list[i]:>10.4f}  {errs[i]:>12.4e}  {p:>8.3f}")
