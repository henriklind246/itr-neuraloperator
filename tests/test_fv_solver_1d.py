import numpy as np
import pytest

from src.physics.fv_solver_1d import FVSolver1D, Layer1D, ic, windowed_sin_flux, compute_dt


# ===================== ic() =====================

class TestIC:
    def test_ic_shape(self):
        x = np.linspace(0, 1, 101)
        assert ic(x, 0, 1).shape == (101,)

    def test_ic_values_at_x_zero(self):
        x = np.array([0.0])
        result = ic(x, 0.0, 1.0)
        expected = np.cos(0) + 0.1 * np.sin(0)  # = 1.0
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_ic_dtype_is_float(self):
        x = np.linspace(0, 1, 11)
        assert np.issubdtype(ic(x, 0, 1).dtype, np.floating)


# ===================== windowed_sin_flux() =====================

class TestWindowedSinFlux:
    """Tests with alpha=0 (rectangular window, backward-compatible behavior)."""

    def test_inside_window(self):
        q = windowed_sin_flux(f=2.0, A=50.0, t_on=0.0, t_off=1.0, phase=0.0, tukey_alpha=0.0)
        t = 0.25
        expected = 50.0 * np.sin(2 * np.pi * 2.0 * t)
        assert pytest.approx(q(t), abs=1e-12) == expected

    def test_outside_window_before(self):
        q = windowed_sin_flux(f=2.0, A=50.0, t_on=0.5, t_off=1.0, tukey_alpha=0.0)
        assert q(0.1) == 0.0

    def test_outside_window_after(self):
        q = windowed_sin_flux(f=2.0, A=50.0, t_on=0.0, t_off=0.5, tukey_alpha=0.0)
        assert q(0.6) == 0.0

    def test_at_window_boundaries(self):
        q = windowed_sin_flux(f=2.0, A=50.0, t_on=0.2, t_off=0.8, tukey_alpha=0.0)
        # t_on and t_off are inside the window (<=)
        assert q(0.2) == 50.0 * np.sin(2 * np.pi * 2.0 * 0.2)
        assert q(0.8) == 50.0 * np.sin(2 * np.pi * 2.0 * 0.8)

    def test_phase_shift(self):
        q = windowed_sin_flux(f=2.0, A=50.0, t_on=0.0, t_off=1.0, phase=np.pi / 2, tukey_alpha=0.0)
        t = 0.0
        expected = 50.0 * np.sin(np.pi / 2)  # = 50.0
        assert pytest.approx(q(t), abs=1e-12) == expected


class TestWindowedSinFluxTukey:
    """Tests for the Tukey (tapered cosine) window envelope."""

    def test_alpha_zero_is_rectangular(self):
        q_rect = windowed_sin_flux(f=2.0, A=50.0, t_on=0.0, t_off=1.0, tukey_alpha=0.0)
        q_tukey = windowed_sin_flux(f=2.0, A=50.0, t_on=0.0, t_off=1.0, tukey_alpha=0.0)
        for t in np.linspace(0.0, 1.0, 50):
            assert pytest.approx(q_rect(t), abs=1e-14) == q_tukey(t)

    def test_smooth_taper_region(self):
        q = windowed_sin_flux(f=1.0, A=1.0, t_on=0.0, t_off=1.0, tukey_alpha=0.5)
        # at t=0 (start of taper), envelope should be 0
        assert q(0.0) == 0.0
        # slightly inside, envelope should be between 0 and 1
        val = q(0.1)  # tau=0.1, inside taper (alpha/2=0.25)
        sin_val = np.sin(2 * np.pi * 1.0 * 0.1)
        assert abs(val) < abs(sin_val)  # attenuated by envelope < 1
        assert abs(val) > 0.0  # but not zero

    def test_flat_top_is_unity(self):
        q = windowed_sin_flux(f=1.0, A=1.0, t_on=0.0, t_off=1.0, tukey_alpha=0.4)
        # tau=0.5 is well inside the flat region (alpha/2=0.2 to 1-alpha/2=0.8)
        t = 0.5
        expected = np.sin(2 * np.pi * 1.0 * t)  # envelope = 1.0
        assert pytest.approx(q(t), abs=1e-14) == expected

    def test_outside_window_still_zero(self):
        q = windowed_sin_flux(f=2.0, A=50.0, t_on=0.2, t_off=0.8, tukey_alpha=0.5)
        assert q(0.1) == 0.0
        assert q(0.9) == 0.0


# ===================== compute_dt() =====================

class TestComputeDt:
    def test_rule_a_dominates(self):
        # lam*h^2/alpha is small → rule_a < rule_b
        h, alpha, lam, f = 0.01, 1.0, 0.5, 2.0
        rule_a = lam * h**2 / alpha  # = 0.00005
        rule_b = 1 / (20 * f)  # = 0.025
        assert compute_dt(h, alpha, lam, f) == rule_a

    def test_rule_b_dominates(self):
        # large h → rule_a > rule_b
        h, alpha, lam, f = 1.0, 1.0, 0.5, 100.0
        rule_a = lam * h**2 / alpha  # = 0.5
        rule_b = 1 / (20 * f)  # = 0.0005
        assert compute_dt(h, alpha, lam, f) == rule_b

    def test_always_positive(self):
        assert compute_dt(0.01, 1.0, 0.5, 2.0) > 0


# ===================== FVSolver1D.__init__ =====================

class TestSolverInit:
    def test_grid_shape(self, small_solver):
        assert small_solver.grid.shape == (11,)

    def test_grid_endpoints(self, small_solver):
        assert small_solver.grid[0] == 0.0
        assert small_solver.grid[-1] == 1.0

    def test_h_value(self, small_solver):
        expected_h = 1.0 / (11 - 1)
        np.testing.assert_allclose(small_solver.h, expected_h)

    def test_dt_explicit_override(self):
        layers = [Layer1D(x_left=0.0, x_right=1.0, rho=1.0, cp=1.0, k=1.0)]
        solver = FVSolver1D(N=11, dt=0.001, a=0, b=1, layers=layers,
                            lam_target=0.5, t_final=0.5, flux_f=2.0, flux_A=50.0,
                            t_on=0.0, t_off=0.5, phase=0.0)
        assert solver.dt == 0.001

    def test_diffusivity_at_nodes(self, small_solver):
        expected_alpha = 1.0 / (1.0 * 1.0)  # k/(rho*cp) for uniform material
        alpha_nodes = small_solver.k_nodes / (small_solver.rho_nodes * small_solver.cp_nodes)
        np.testing.assert_allclose(alpha_nodes, expected_alpha)


# ===================== build_A_banded =====================

class TestBuildABanded:
    def test_shape(self, small_solver):
        assert small_solver.ab.shape == (4, 11)

    def test_N_less_than_3_raises(self):
        layers = [Layer1D(x_left=0.0, x_right=1.0, rho=1.0, cp=1.0, k=1.0)]
        with pytest.raises(ValueError, match="N must be >= 3"):
            FVSolver1D(N=2, a=0, b=1, layers=layers,
                        lam_target=0.5, t_final=0.1, flux_f=2.0, flux_A=50.0,
                        t_on=0.0, t_off=0.1, phase=0.0)

    def test_N_equals_3_succeeds(self):
        layers = [Layer1D(x_left=0.0, x_right=1.0, rho=1.0, cp=1.0, k=1.0)]
        solver = FVSolver1D(N=3, a=0, b=1, layers=layers,
                            lam_target=0.5, t_final=0.1, flux_f=2.0, flux_A=50.0,
                            t_on=0.0, t_off=0.1, phase=0.0)
        assert solver.ab.shape == (4, 3)

    def test_neumann_row(self, small_solver):
        ab = small_solver.ab
        assert ab[2, 0] == 3
        assert ab[1, 1] == -4
        assert ab[0, 2] == 1

    def test_dirichlet_row(self, small_solver):
        ab = small_solver.ab
        N = small_solver.N
        assert ab[2, N - 1] == 1.0
        assert ab[3, N - 2] == 0.0

    def test_interior_main_diagonal(self, small_solver):
        ab = small_solver.ab
        N = small_solver.N
        for i in range(1, N - 1):
            rm = small_solver.r_minus[i]
            rp = small_solver.r_plus[i]
            np.testing.assert_allclose(ab[2, i], 1.0 + rm + rp)

    def test_interior_off_diagonals(self, small_solver):
        ab = small_solver.ab
        N = small_solver.N
        for i in range(1, N - 1):
            np.testing.assert_allclose(ab[3, i - 1], -small_solver.r_minus[i])
            np.testing.assert_allclose(ab[1, i + 1], -small_solver.r_plus[i])


# ===================== cn_step_banded =====================

class TestCNStep:
    def test_output_shape(self, small_solver):
        Tn = np.ones(small_solver.N) * 300.0
        result = small_solver.cn_step_banded(Tn, 0.0)
        assert result.shape == (small_solver.N,)

    def test_preserves_dirichlet(self, small_solver):
        Tn = np.ones(small_solver.N) * 300.0
        result = small_solver.cn_step_banded(Tn, 0.0)
        assert result[-1] == 300.0


# ===================== solve() =====================

class TestSolve:
    def test_final_only_shapes(self, small_solver):
        t, x, T_final = small_solver.solve(store_trajectory=False)
        assert T_final.shape == (small_solver.N,)
        assert t.shape == (len(small_solver.t),)
        assert x.shape == (small_solver.N,)

    def test_trajectory_shapes(self, small_solver):
        t, x, T_hist = small_solver.solve(store_trajectory=True)
        Nt = len(small_solver.t)
        assert T_hist.shape == (Nt, small_solver.N)

    def test_trajectory_first_row_is_ic(self, small_solver):
        t, x, T_hist = small_solver.solve(store_trajectory=True)
        expected_ic = ic(small_solver.grid, small_solver.a, small_solver.b) + 300.0
        np.testing.assert_allclose(T_hist[0], expected_ic)

    def test_custom_T0(self, small_solver):
        T0 = np.ones(small_solver.N) * 500.0
        t, x, T_hist = small_solver.solve(T0=T0, store_trajectory=True)
        np.testing.assert_allclose(T_hist[0], T0)

    def test_wrong_T0_shape_raises(self, small_solver):
        T0 = np.ones(small_solver.N + 5) * 300.0
        with pytest.raises(ValueError):
            small_solver.solve(T0=T0)

    def test_trajectory_final_matches_final_only(self, small_solver):
        T0 = np.ones(small_solver.N) * 300.0
        _, _, T_final = small_solver.solve(T0=T0, store_trajectory=False)
        _, _, T_hist = small_solver.solve(T0=T0, store_trajectory=True)
        np.testing.assert_allclose(T_final, T_hist[-1], atol=1e-12)

    def test_right_bc_maintained(self, small_solver):
        _, _, T_hist = small_solver.solve(store_trajectory=True)
        np.testing.assert_allclose(T_hist[:, -1], 300.0)

    def test_temperatures_finite(self, small_solver):
        _, _, T_hist = small_solver.solve(store_trajectory=True)
        assert np.all(np.isfinite(T_hist))
