import numpy as np
import pytest

from src.physics.mms_1d import run_mms_once


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
        and three grid levels to estimate mean order.
        """
        N_list = [21, 41, 81]
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
        assert 1.5 < mean_order < 2.5, f"Expected order ~2, got {mean_order:.2f}"


class TestTimeOrderTest:
    def test_approximately_two(self):
        """Verify temporal convergence order ≈ 2.

        Uses a fine grid (N=801) so spatial error is negligible,
        and two dt values to estimate temporal order.
        """
        N_fixed = 801
        dt_list = [0.02, 0.01]
        errs = []
        for dt in dt_list:
            _, _, _, l2 = run_mms_once(N_fixed, dt=dt)
            errs.append(l2)
        p_t = np.log(errs[0] / errs[1]) / np.log(dt_list[0] / dt_list[1])
        assert 1.5 < p_t < 2.5, f"Expected order ~2, got {p_t:.2f}"
