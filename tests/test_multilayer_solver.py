"""
Multilayer FD Solver Validation Tests
=====================================

eEight test groups validating the conservative multilayer Crank-Nicolson solver:

1. Constant solution preservation
2. Geometry / interface validation
3. Regression: single-layer MMS verification
4. One layer vs two identical layers
5. Coefficient and indexing sanity
6. Material-jump physics sanity
7. Grid refinement / self-convergence
8. Interface thermal resistance
"""

import numpy as np
import pytest

from src.physics.fd_solver_1d import FDSolver1D, Layer1D


# ==================== SHARED METRICS ====================

def err_inf(A: np.ndarray, B: np.ndarray) -> float:
    """Absolute max error."""
    return float(np.max(np.abs(A - B)))


def err_rel_l2(A: np.ndarray, B: np.ndarray) -> float:
    """Relative L2 error."""
    return float(np.linalg.norm(A - B) / max(np.linalg.norm(B), 1e-14))


# ==================== SHARED SOLVER PARAMS ====================

# Common parameters for multilayer tests (no material properties).
COMMON_PARAMS = dict(
    a=0.0, b=1.0,
    lam_target=0.5,
    flux_f=2.0, flux_A=50.0,
    t_on=0.0, t_off=0.5, phase=0.0,
)


# ==================== TEST 1: CONSTANT SOLUTION PRESERVATION ====================

class TestConstantSolution:
    """
    Test 1: Verify that a constant T=300 field is preserved exactly
    when BCs and source imply no evolution.
    """

    @staticmethod
    def _make_const_solver(layers, N, t_final=0.1):
        """Build a solver with zero flux, constant right BC, no source."""
        return FDSolver1D(
            a=0.0, b=1.0, N=N, lam_target=0.5,
            layers=layers, t_final=t_final,
            flux_f=0.0, flux_A=0.0,
            t_on=0.0, t_off=t_final, phase=0.0,
            q_left_fn=lambda t: 0.0,
            T_right_fn=lambda t: 300.0,
        )

    def test_constant_single_layer(self):
        """Single uniform layer: T=300 must stay constant."""
        layers = [Layer1D(0.0, 1.0, rho=1.0, cp=1.0, k=1.0)]
        sim = self._make_const_solver(layers, N=51)
        T0 = np.full(51, 300.0)
        _, _, T_hist = sim.solve(T0=T0, store_trajectory=True)
        max_err = np.max(np.abs(T_hist - 300.0))
        assert max_err < 1e-10, f"Constant solution drifted: max error = {max_err}"

    def test_constant_multilayer(self):
        """Two layers with different materials: T=300 must stay constant."""
        layers = [
            Layer1D(0.0, 0.5, rho=8000.0, cp=500.0, k=50.0),
            Layer1D(0.5, 1.0, rho=1.0, cp=1.0, k=5.0),
        ]
        sim = self._make_const_solver(layers, N=100)
        T0 = np.full(100, 300.0)
        _, _, T_hist = sim.solve(T0=T0, store_trajectory=True)
        max_err = np.max(np.abs(T_hist - 300.0))
        assert max_err < 1e-10, f"Constant solution drifted: max error = {max_err}"


# ==================== TEST 2: GEOMETRY / INTERFACE VALIDATION ====================

class TestGeometryValidation:
    """
    Test 2: Verify that the solver correctly accepts or rejects
    layer/interface configurations.
    """

    COMMON = dict(
        a=0.0, b=1.0, lam_target=0.5, t_final=0.1,
        flux_f=2.0, flux_A=50.0,
        t_on=0.0, t_off=0.1, phase=0.0,
    )

    def test_2a_valid_face_aligned_interface(self):
        """N=100, interface at 0.5 -> face-aligned. Should succeed."""
        layers = [
            Layer1D(0.0, 0.5, 1.0, 1.0, 1.0),
            Layer1D(0.5, 1.0, 1.0, 1.0, 1.0),
        ]
        sim = FDSolver1D(N=100, layers=layers, **self.COMMON)
        assert 49 in sim.interface_face_map
        assert sim.interface_face_map[49] == (0, 1)
        assert len(sim.interface_positions) == 1
        assert sim.interface_positions[0] == pytest.approx(0.5)

    def test_2b_interface_on_node_raises(self):
        """N=101, interface at 0.5 -> lies on grid node. Should fail."""
        layers = [
            Layer1D(0.0, 0.5, 1.0, 1.0, 1.0),
            Layer1D(0.5, 1.0, 1.0, 1.0, 1.0),
        ]
        with pytest.raises(ValueError, match="lies on a grid node"):
            FDSolver1D(N=101, layers=layers, **self.COMMON)

    def test_2c_off_face_interface_raises(self):
        """N=100, interface at 0.37 -> not face-aligned. Should fail."""
        layers = [
            Layer1D(0.0, 0.37, 1.0, 1.0, 1.0),
            Layer1D(0.37, 1.0, 1.0, 1.0, 1.0),
        ]
        with pytest.raises(ValueError, match="not face-aligned"):
            FDSolver1D(N=100, layers=layers, **self.COMMON)

    def test_2d_gap_between_layers_raises(self):
        """Layer1 ends at 0.49, layer2 starts at 0.50 -> gap. Should fail."""
        layers = [
            Layer1D(0.0, 0.49, 1.0, 1.0, 1.0),
            Layer1D(0.50, 1.0, 1.0, 1.0, 1.0),
        ]
        with pytest.raises(ValueError, match="not contiguous"):
            FDSolver1D(N=100, layers=layers, **self.COMMON)

    def test_2e_overlap_between_layers_raises(self):
        """Layer1 ends at 0.55, layer2 starts at 0.50 -> overlap. Should fail."""
        layers = [
            Layer1D(0.0, 0.55, 1.0, 1.0, 1.0),
            Layer1D(0.50, 1.0, 1.0, 1.0, 1.0),
        ]
        with pytest.raises(ValueError, match="not contiguous"):
            FDSolver1D(N=100, layers=layers, **self.COMMON)

    def test_2f_domain_not_covered_raises(self):
        """First layer starts at 0.1 or last layer ends at 0.9. Should fail."""
        layers_bad_left = [Layer1D(0.1, 1.0, 1.0, 1.0, 1.0)]
        with pytest.raises(ValueError, match="First layer must start"):
            FDSolver1D(N=100, layers=layers_bad_left, **self.COMMON)

        layers_bad_right = [Layer1D(0.0, 0.9, 1.0, 1.0, 1.0)]
        with pytest.raises(ValueError, match="Last layer must end"):
            FDSolver1D(N=100, layers=layers_bad_right, **self.COMMON)


# ==================== TEST 3: MMS REGRESSION ====================

@pytest.mark.slow
class TestMMSRegression:
    """
    Test 3: Regression test ensuring the refactored solver still matches
    the MMS exact analytical solution for a single uniform layer.

    Since the old solver code was replaced in-place, the MMS analytical
    solution serves as the reference ("old solver").
    """

    def test_mms_error_bounded(self):
        """L2 and max errors at N=101 are within known thresholds."""
        from src.physics.mms_1d import run_mms_once

        h, dt, max_err, l2_err = run_mms_once(N=101)
        assert l2_err < 5e-3, f"L2 error {l2_err:.6e} exceeds threshold"
        assert max_err < 1e-2, f"Max error {max_err:.6e} exceeds threshold"

    def test_mms_convergence_rate(self):
        """Spatial convergence order should be ~2 (CN is second-order)."""
        from src.physics.mms_1d import run_mms_once

        hs, errs = [], []
        for N in [21, 41, 81]:
            h, _, _, l2 = run_mms_once(N, dt=0.0005)
            hs.append(h)
            errs.append(l2)

        orders = []
        for i in range(len(errs) - 1):
            p = np.log(errs[i] / errs[i + 1]) / np.log(hs[i] / hs[i + 1])
            orders.append(p)

        mean_order = np.mean(orders)
        assert 1.5 < mean_order < 2.5, f"Expected order ~2, got {mean_order:.2f}"

    def test_solve_properties_new_api(self):
        """Basic solve output properties with the new Layer1D API."""
        layer = Layer1D(0.0, 1.0, rho=1.0, cp=1.0, k=1.0)
        sim = FDSolver1D(
            a=0.0, b=1.0, N=51, lam_target=0.5,
            layers=[layer], t_final=0.5,
            flux_f=2.0, flux_A=50.0,
            t_on=0.0, t_off=0.5, phase=0.0,
        )
        t, x, T_hist = sim.solve(store_trajectory=True)

        # Shape checks
        assert T_hist.shape == (len(t), 51)
        assert x.shape == (51,)

        # All values finite
        assert np.all(np.isfinite(T_hist)), "Non-finite values in trajectory"

        # Right Dirichlet BC maintained at all timesteps
        np.testing.assert_allclose(T_hist[:, -1], 300.0)


# ==================== TEST 4: ONE LAYER VS TWO IDENTICAL LAYERS ====================

class TestIdenticalLayerEquivalence:
    """
    Test 4: Verify that the solver gives the same answer whether the domain
    is one full layer or two adjacent layers with identical properties.

    Any difference would indicate the interface machinery introduces artifacts.
    """

    @pytest.fixture
    def one_layer_solver(self):
        layers = [Layer1D(0.0, 1.0, 1.0, 1.0, 1.0)]
        return FDSolver1D(
            N=100, layers=layers, t_final=0.2, **COMMON_PARAMS,
        )

    @pytest.fixture
    def two_layer_solver(self):
        layers = [
            Layer1D(0.0, 0.5, 1.0, 1.0, 1.0),
            Layer1D(0.5, 1.0, 1.0, 1.0, 1.0),
        ]
        return FDSolver1D(
            N=100, layers=layers, t_final=0.2, **COMMON_PARAMS,
        )

    def test_final_solution_matches(self, one_layer_solver, two_layer_solver):
        T0 = np.full(100, 300.0)
        _, _, T_final_1 = one_layer_solver.solve(T0=T0, store_trajectory=False)
        _, _, T_final_2 = two_layer_solver.solve(T0=T0, store_trajectory=False)

        inf_err = err_inf(T_final_1, T_final_2)
        rel_err = err_rel_l2(T_final_1, T_final_2)
        assert inf_err < 1e-10, f"Final T mismatch: err_inf = {inf_err}"
        assert rel_err < 1e-10, f"Final T mismatch: err_rel_l2 = {rel_err}"

    def test_trajectory_matches(self, one_layer_solver, two_layer_solver):
        T0 = np.full(100, 300.0)
        _, _, T_hist_1 = one_layer_solver.solve(T0=T0, store_trajectory=True)
        _, _, T_hist_2 = two_layer_solver.solve(T0=T0, store_trajectory=True)

        inf_err = err_inf(T_hist_1, T_hist_2)
        rel_err = err_rel_l2(T_hist_1, T_hist_2)
        assert inf_err < 1e-10, f"Trajectory mismatch: err_inf = {inf_err}"
        assert rel_err < 1e-10, f"Trajectory mismatch: err_rel_l2 = {rel_err}"


# ==================== TEST 5: COEFFICIENT AND INDEXING SANITY ====================

class TestCoefficientIndexing:
    """
    Test 5: Verify that the internal arrays defining multilayer physics
    are correctly built for a two-layer problem.

    Setup: [0, 0.5] rho=1, cp=1, k=1  |  [0.5, 1] rho=2, cp=1.5, k=0.2
    N=100  ->  h = 1/99, interface at face index 49.
    """

    @pytest.fixture
    def two_layer_solver(self):
        layers = [
            Layer1D(0.0, 0.5, rho=1.0, cp=1.0, k=1.0),
            Layer1D(0.5, 1.0, rho=2.0, cp=1.5, k=0.2),
        ]
        return FDSolver1D(
            N=100, layers=layers, t_final=0.1, **COMMON_PARAMS,
        )

    def test_layer_index_at_nodes(self, two_layer_solver):
        """Nodes < 0.5 -> layer 0, nodes >= 0.5 -> layer 1."""
        idx = two_layer_solver.layer_index_at_nodes
        assert idx.shape == (100,)
        # With N=100, h=1/99, node i has x = i/99
        # Node 50 has x = 50/99 ~ 0.505 > 0.5 -> layer 1
        # Node 49 has x = 49/99 ~ 0.4949 < 0.5 -> layer 0
        np.testing.assert_array_equal(idx[:50], 0)
        np.testing.assert_array_equal(idx[50:], 1)

    def test_material_arrays(self, two_layer_solver):
        """rho, cp, k at nodes should match layer assignment."""
        # Layer 0: rho=1, cp=1, k=1
        np.testing.assert_allclose(two_layer_solver.rho_nodes[:50], 1.0)
        np.testing.assert_allclose(two_layer_solver.cp_nodes[:50], 1.0)
        np.testing.assert_allclose(two_layer_solver.k_nodes[:50], 1.0)
        # Layer 1: rho=2, cp=1.5, k=0.2
        np.testing.assert_allclose(two_layer_solver.rho_nodes[50:], 2.0)
        np.testing.assert_allclose(two_layer_solver.cp_nodes[50:], 1.5)
        np.testing.assert_allclose(two_layer_solver.k_nodes[50:], 0.2)

    def test_C_node(self, two_layer_solver):
        """Storage coefficient: C_node = rho * cp * h."""
        h = two_layer_solver.h
        C = two_layer_solver.C_node
        np.testing.assert_allclose(C[:50], 1.0 * 1.0 * h)
        np.testing.assert_allclose(C[50:], 2.0 * 1.5 * h)

    def test_G_face_interior(self, two_layer_solver):
        """Interior (non-interface) faces: G = k/h within each material."""
        h = two_layer_solver.h
        G = two_layer_solver.G_face
        # Faces 0..48 are in layer 0 (k=1): G = 1.0/h
        np.testing.assert_allclose(G[:49], 1.0 / h)
        # Faces 50..98 are in layer 1 (k=0.2): G = 0.2/h
        np.testing.assert_allclose(G[50:], 0.2 / h)

    def test_G_face_interface(self, two_layer_solver):
        """Interface face: harmonic mean conductance G = 1/(h/(2*kL) + h/(2*kR))."""
        h = two_layer_solver.h
        G = two_layer_solver.G_face
        kL, kR = 1.0, 0.2
        expected_G_int = 1.0 / (h / (2.0 * kL) + h / (2.0 * kR))
        np.testing.assert_allclose(G[49], expected_G_int)

    def test_interface_face_map(self, two_layer_solver):
        """Exactly one interface at face index 49, mapping layers (0, 1)."""
        ifm = two_layer_solver.interface_face_map
        assert len(ifm) == 1
        assert 49 in ifm
        assert ifm[49] == (0, 1)


# ==================== TEST 6: MATERIAL-JUMP SANITY ====================

@pytest.mark.slow
class TestMaterialJumpSanity:
    """
    Test 6: Qualitative physics check with a strong conductivity contrast.

    k=1.0 vs k=0.05 with a heat flux pulse. Verifies no blow-up,
    smooth solution within each layer, and physically plausible behavior.
    """

    @pytest.fixture
    def high_contrast_solver(self):
        layers = [
            Layer1D(0.0, 0.5, rho=1.0, cp=1.0, k=1.0),
            Layer1D(0.5, 1.0, rho=1.0, cp=1.0, k=0.05),
        ]
        return FDSolver1D(
            N=100, layers=layers, t_final=0.5,
            a=0.0, b=1.0, lam_target=0.5,
            flux_f=2.0, flux_A=50.0,
            t_on=0.0, t_off=0.2, phase=0.0,
        )

    def test_no_nan_or_inf(self, high_contrast_solver):
        """Entire trajectory must contain only finite values."""
        T0 = np.full(100, 300.0)
        _, _, T_hist = high_contrast_solver.solve(T0=T0, store_trajectory=True)
        assert np.all(np.isfinite(T_hist)), "Non-finite values in trajectory"

    def test_solution_smooth_within_layers(self, high_contrast_solver):
        """No wild oscillations: max node-to-node jump within each layer
        should be a small fraction of the overall temperature range."""
        T0 = np.full(100, 300.0)
        _, _, T_final = high_contrast_solver.solve(T0=T0, store_trajectory=False)

        diffs = np.abs(np.diff(T_final))
        T_range = np.max(T_final) - np.min(T_final)

        if T_range > 1e-10:
            # Interior diffs within each layer (exclude interface at nodes 49-50)
            max_diff_layer0 = np.max(diffs[:49])
            max_diff_layer1 = np.max(diffs[50:])
            assert max_diff_layer0 / T_range < 0.2, (
                f"Layer 0 not smooth: max interior diff / range = "
                f"{max_diff_layer0 / T_range:.3f}"
            )
            assert max_diff_layer1 / T_range < 0.2, (
                f"Layer 1 not smooth: max interior diff / range = "
                f"{max_diff_layer1 / T_range:.3f}"
            )

    def test_temperatures_bounded(self, high_contrast_solver):
        """Solution should stay within a physically sensible range.
        No temperatures wildly above or below initial/boundary values."""
        T0 = np.full(100, 300.0)
        _, _, T_hist = high_contrast_solver.solve(T0=T0, store_trajectory=True)

        # Heat enters from the left, so temperatures may rise above 300
        # but should not be extreme (e.g., not > 1000 K or < 0 K)
        assert np.min(T_hist) > 0.0, f"Unphysical negative temperature: {np.min(T_hist)}"
        assert np.max(T_hist) < 1000.0, f"Unphysical temperature spike: {np.max(T_hist)}"


# ==================== TEST 7: GRID REFINEMENT / SELF-CONVERGENCE ====================

@pytest.mark.slow
class TestGridRefinementConvergence:
    """
    Test 7: Self-convergence under grid refinement for a two-layer problem.

    Run on N=50, 100, 200 with fixed small dt to isolate spatial convergence.
    Errors (vs finest grid) must decrease as the grid is refined.
    """

    @staticmethod
    def _make_solver(N):
        layers = [
            Layer1D(0.0, 0.5, rho=1.0, cp=1.0, k=1.0),
            Layer1D(0.5, 1.0, rho=2.0, cp=1.5, k=0.5),
        ]
        return FDSolver1D(
            a=0.0, b=1.0, N=N, lam_target=0.5,
            layers=layers, t_final=0.2,
            flux_f=2.0, flux_A=50.0,
            t_on=0.0, t_off=0.2, phase=0.0,
            dt=0.0001,  # fixed small dt so temporal error is negligible
        )

    def test_errors_decrease_with_refinement(self):
        """Interpolated error from fine to coarse grid should decrease."""
        N_values = [50, 100, 200]
        solutions = {}
        grids = {}

        for N in N_values:
            sim = self._make_solver(N)
            T0 = np.full(N, 300.0)
            _, x, T_final = sim.solve(T0=T0, store_trajectory=False)
            solutions[N] = T_final
            grids[N] = x

        # Use finest grid (N=200) as reference
        x_ref = grids[200]
        T_ref = solutions[200]

        errors_inf = []
        errors_l2 = []
        for N in [50, 100]:
            x_coarse = grids[N]
            # Interpolate reference solution to coarse grid
            T_ref_on_coarse = np.interp(x_coarse, x_ref, T_ref)
            e_inf = err_inf(solutions[N], T_ref_on_coarse)
            e_l2 = err_rel_l2(solutions[N], T_ref_on_coarse)
            errors_inf.append(e_inf)
            errors_l2.append(e_l2)

        # Error at N=50 should be larger than error at N=100
        assert errors_inf[0] > errors_inf[1], (
            f"err_inf did not decrease with refinement: "
            f"N=50 -> {errors_inf[0]:.6e}, N=100 -> {errors_inf[1]:.6e}"
        )
        assert errors_l2[0] > errors_l2[1], (
            f"err_rel_l2 did not decrease with refinement: "
            f"N=50 -> {errors_l2[0]:.6e}, N=100 -> {errors_l2[1]:.6e}"
        )


# ==================== TEST 8: INTERFACE THERMAL RESISTANCE ====================

class TestInterfaceResistance:
    """
    Test 8: Verify the interface thermal resistance (interface_R) parameter.

    Adding R_c (m²·K/W) to the face conductance at an interface:
        G = 1 / (h/(2*k_L) + R_c + h/(2*k_R))
    """

    # --- 8a: R_c=0 matches perfect contact (bitwise identical) ---

    def test_8a_zero_resistance_matches_default(self):
        """interface_R=[0.0] must produce identical results to default (None)."""
        layers = [
            Layer1D(0.0, 0.5, rho=1.0, cp=1.0, k=1.0),
            Layer1D(0.5, 1.0, rho=2.0, cp=1.5, k=0.5),
        ]
        sim_default = FDSolver1D(
            N=100, layers=layers, t_final=0.2, **COMMON_PARAMS,
        )
        sim_zero = FDSolver1D(
            N=100, layers=layers, t_final=0.2, interface_R=[0.0], **COMMON_PARAMS,
        )
        T0 = np.full(100, 300.0)
        _, _, T_default = sim_default.solve(T0=T0, store_trajectory=False)
        _, _, T_zero = sim_zero.solve(T0=T0, store_trajectory=False)

        np.testing.assert_array_equal(T_default, T_zero)

    # --- 8b: Constant field preserved with R_c > 0 ---

    def test_8b_constant_preserved_with_resistance(self):
        """T=300 with zero flux and R_c=1.0 stays constant (no flux = no jump)."""
        layers = [
            Layer1D(0.0, 0.5, rho=1.0, cp=1.0, k=1.0),
            Layer1D(0.5, 1.0, rho=2.0, cp=1.5, k=0.5),
        ]
        sim = FDSolver1D(
            a=0.0, b=1.0, N=100, lam_target=0.5,
            layers=layers, t_final=0.1,
            flux_f=0.0, flux_A=0.0,
            t_on=0.0, t_off=0.1, phase=0.0,
            q_left_fn=lambda t: 0.0,
            T_right_fn=lambda t: 300.0,
            interface_R=[1.0],
        )
        T0 = np.full(100, 300.0)
        _, _, T_hist = sim.solve(T0=T0, store_trajectory=True)
        max_err = np.max(np.abs(T_hist - 300.0))
        assert max_err < 1e-10, f"Constant solution drifted: max error = {max_err}"

    # --- 8c: G_face coefficient with R_c ---

    def test_8c_face_conductance_with_resistance(self):
        """White-box: G_face at the interface face must equal
        1 / (h/(2*k_L) + R_c + h/(2*k_R))."""
        k1, k2, Rc = 1.0, 0.5, 0.05
        layers = [
            Layer1D(0.0, 0.5, rho=1.0, cp=1.0, k=k1),
            Layer1D(0.5, 1.0, rho=1.0, cp=1.0, k=k2),
        ]
        sim = FDSolver1D(
            N=100, layers=layers, t_final=0.1, interface_R=[Rc], **COMMON_PARAMS,
        )
        h = sim.h
        expected_G = 1.0 / (h / (2.0 * k1) + Rc + h / (2.0 * k2))
        np.testing.assert_allclose(sim.G_face[49], expected_G)

    # --- 8d: Steady-state analytical (strongest verification) ---

    def test_8d_steady_state_preserved(self):
        """Start at exact piecewise-linear steady state with constant flux q,
        two layers, and R_c. The solver must preserve this profile.

        Exact steady-state (constant flux q, right Dirichlet T_right):
            Layer 1: T(x) = T_right + q * [(L1 - x)/k1 + R_c + L2/k2]
            Layer 2: T(x) = T_right + q * [(1 - x)/k2]

        Temperature drop across the interface face (nodes 49 → 50):
            ΔT = q * (h/(2*k1) + R_c + h/(2*k2))
        """
        k1, k2 = 2.0, 0.5
        Rc = 0.1
        q = 100.0  # constant left flux
        T_right = 300.0
        L1, L2 = 0.5, 0.5
        N = 100

        layers = [
            Layer1D(0.0, L1, rho=1.0, cp=1.0, k=k1),
            Layer1D(L1, 1.0, rho=1.0, cp=1.0, k=k2),
        ]

        sim = FDSolver1D(
            a=0.0, b=1.0, N=N, lam_target=0.5,
            layers=layers, t_final=0.5,
            flux_f=0.0, flux_A=0.0,
            t_on=0.0, t_off=1.0, phase=0.0,
            q_left_fn=lambda t: q,
            T_right_fn=lambda t: T_right,
            interface_R=[Rc],
            dt=0.001,
        )

        # Build exact steady-state IC on the grid
        x = sim.grid
        T0 = np.zeros(N, dtype=float)
        for i, xi in enumerate(x):
            if xi < L1 - sim.tol:
                # Layer 1
                T0[i] = T_right + q * ((L1 - xi) / k1 + Rc + L2 / k2)
            else:
                # Layer 2
                T0[i] = T_right + q * ((1.0 - xi) / k2)

        _, _, T_final = sim.solve(T0=T0, store_trajectory=False)

        # Solver should preserve the steady state
        max_err = err_inf(T_final, T0)
        assert max_err < 1e-6, f"Steady state not preserved: max error = {max_err:.2e}"

        # Verify temperature drop across interface face (nodes 49 → 50)
        h = sim.h
        expected_dT = q * (h / (2.0 * k1) + Rc + h / (2.0 * k2))
        actual_dT = T0[49] - T0[50]
        np.testing.assert_allclose(actual_dT, expected_dT, rtol=1e-10)

    # --- 8e: Validation errors ---

    def test_8e_wrong_length_raises(self):
        """interface_R with wrong length must raise ValueError."""
        layers = [
            Layer1D(0.0, 0.5, rho=1.0, cp=1.0, k=1.0),
            Layer1D(0.5, 1.0, rho=1.0, cp=1.0, k=1.0),
        ]
        # Two-layer has 1 interface, but we pass 2 resistances
        with pytest.raises(ValueError, match="interface_R has length 2"):
            FDSolver1D(
                N=100, layers=layers, t_final=0.1, interface_R=[0.0, 0.0],
                **COMMON_PARAMS,
            )

    def test_8e_negative_resistance_raises(self):
        """Negative R_c must raise ValueError."""
        layers = [
            Layer1D(0.0, 0.5, rho=1.0, cp=1.0, k=1.0),
            Layer1D(0.5, 1.0, rho=1.0, cp=1.0, k=1.0),
        ]
        with pytest.raises(ValueError, match="negative"):
            FDSolver1D(
                N=100, layers=layers, t_final=0.1, interface_R=[-0.01],
                **COMMON_PARAMS,
            )

    def test_8e_single_layer_defaults_to_empty(self):
        """Single layer has no interfaces, so interface_R defaults to []."""
        layers = [Layer1D(0.0, 1.0, rho=1.0, cp=1.0, k=1.0)]
        sim = FDSolver1D(
            N=100, layers=layers, t_final=0.1, **COMMON_PARAMS,
        )
        assert sim.interface_R == []

    # --- 8f: Large R_c sanity ---

    def test_8f_large_resistance_no_blowup(self):
        """R_c=10.0: no NaN/Inf, finite temperatures, solution bounded."""
        layers = [
            Layer1D(0.0, 0.5, rho=1.0, cp=1.0, k=1.0),
            Layer1D(0.5, 1.0, rho=1.0, cp=1.0, k=1.0),
        ]
        sim = FDSolver1D(
            N=100, layers=layers, t_final=0.2, interface_R=[10.0], **COMMON_PARAMS,
        )
        T0 = np.full(100, 300.0)
        _, _, T_hist = sim.solve(T0=T0, store_trajectory=True)

        assert np.all(np.isfinite(T_hist)), "Non-finite values with large R_c"
        assert np.min(T_hist) > 0.0, f"Unphysical negative temperature: {np.min(T_hist)}"
        assert np.max(T_hist) < 2000.0, f"Unphysical temperature spike: {np.max(T_hist)}"

    # --- 8g: Three layers, two interfaces ---

    def test_8g_three_layers_two_interfaces(self):
        """Three layers with different R_c at each interface runs without error.

        Domain [0, 1], N=99 (h=1/98). Interfaces at x=0.25 and x=0.75.
        Face alignment: 0.25*98 - 0.5 = 24 (integer), 0.75*98 - 0.5 = 73 (integer).
        Node check: 0.25*98 = 24.5, 0.75*98 = 73.5 (not on nodes).
        """
        layers_3 = [
            Layer1D(0.0, 0.25, rho=1.0, cp=1.0, k=2.0),
            Layer1D(0.25, 0.75, rho=1.5, cp=1.0, k=0.5),
            Layer1D(0.75, 1.0, rho=1.0, cp=1.0, k=1.0),
        ]
        sim = FDSolver1D(
            a=0.0, b=1.0, N=99, lam_target=0.5,
            layers=layers_3, t_final=0.1,
            flux_f=2.0, flux_A=50.0,
            t_on=0.0, t_off=0.1, phase=0.0,
            interface_R=[0.05, 0.1],
        )

        assert len(sim.interface_R) == 2
        assert len(sim.interface_positions) == 2

        T0 = np.full(99, 300.0)
        _, _, T_final = sim.solve(T0=T0, store_trajectory=False)
        assert np.all(np.isfinite(T_final)), "Non-finite values with three layers"


# ============================================================
# 9. MMS VERIFICATION — interface thermal resistance
#    Piecewise manufactured solution with R_c, verifying
#    2nd-order spatial and temporal convergence.
# ============================================================

from src.physics.mms_1d import (
    run_mms_interface,
    space_order_test_interface,
    time_order_test_interface,
)


class TestMmsInterfaceResistance:
    """MMS convergence verification for the CN scheme with interface resistance.

    Uses a piecewise manufactured solution T_L*(x,t), T_R*(x,t) that
    satisfies flux continuity and the temperature jump condition
    ΔT = R_c · q_I at the interface.  Source terms are derived analytically.
    """

    def test_9a_moderate_grid_low_error(self):
        """N=100 gives L2 error well below 1e-2 (≈7.5e-4 expected)."""
        _, _, max_err, l2_err = run_mms_interface(N=100)
        assert l2_err < 5e-3, f"L2 error {l2_err:.4e} too large for N=100"
        assert max_err < 5e-3, f"Max error {max_err:.4e} too large for N=100"

    def test_9b_fine_grid_very_low_error(self):
        """N=400, dt=0.001 gives L2 error < 1e-3 (≈5.6e-5 expected)."""
        _, _, max_err, l2_err = run_mms_interface(N=400, dt=0.001)
        assert l2_err < 1e-3, f"L2 error {l2_err:.4e} too large for fine grid"
        assert max_err < 1e-3, f"Max error {max_err:.4e} too large for fine grid"

    def test_9c_spatial_order_two(self):
        """Spatial convergence order ≈ 2 (dt fixed small, vary N)."""
        p = space_order_test_interface(N_list=[50, 100])
        assert 1.8 < p < 2.2, f"Spatial order {p:.3f} outside [1.8, 2.2]"

    def test_9d_temporal_order_two(self):
        """Temporal convergence order ≈ 2 (N fixed large, vary dt)."""
        p = time_order_test_interface(dt_list=[0.02, 0.01])
        assert 1.8 < p < 2.2, f"Temporal order {p:.3f} outside [1.8, 2.2]"
