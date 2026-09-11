"""
2D FV Solver Tests
==================

Test groups:
1. Grid and data model validation
2. Tiny-grid matrix verification (Nx=4, Ny=3)
3. Constant solution preservation
4. 1D equivalence (y-independent problem)
5. Adiabatic BC verification
6. Time-dependent Dirichlet
7. Multilayer with R_c constant solution
"""

import numpy as np
import pytest

from src.physics.fv_solver_1d import FVSolver1D, Layer1D, compute_dt_2d
from src.physics.fv_solver_2d import FVSolver2D, Layer2D


# ==================== HELPERS ====================

def make_single_layer_2d(**overrides):
    """Build a single-layer 2D solver with sensible defaults."""
    defaults = dict(
        a=0.0, b=1.0, c=0.0, d=1.0,
        Nx=11, Ny=11,
        lam_target=0.5,
        layers=[Layer2D(x_left=0.0, x_right=1.0, rho=1.0, cp=1.0, k=1.0)],
        t_final=0.1,
        flux_f=2.0, flux_A=50.0,
        t_on=0.0, t_off=0.1, phase=0.0,
    )
    defaults.update(overrides)
    return FVSolver2D(**defaults)


# ==================== TEST 1: GRID AND DATA MODEL ====================

class TestGridAndDataModel:

    def test_grid_shapes(self):
        # Nx=11 on [0,1] -> hx=0.1; Ny=6 on [0,0.5] -> hy=0.1. Isotropic.
        sim = make_single_layer_2d(Nx=11, Ny=6, d=0.5)
        assert sim.grid_x.shape == (11,)
        assert sim.grid_y.shape == (6,)
        assert sim.X.shape == (11, 6)
        assert sim.Y.shape == (11, 6)

    def test_grid_endpoints(self):
        sim = make_single_layer_2d(a=0.0, b=1.0, c=0.0, d=0.5, Nx=11, Ny=6)
        assert sim.grid_x[0] == 0.0
        assert sim.grid_x[-1] == 1.0
        assert sim.grid_y[0] == 0.0
        assert sim.grid_y[-1] == 0.5

    def test_isotropic_grid_enforced(self):
        """hx != hy must raise."""
        with pytest.raises(ValueError, match="not isotropic"):
            # Nx=11 on [0,1] gives hx=0.1, Ny=6 on [0,1] gives hy=0.2
            make_single_layer_2d(Nx=11, Ny=6)

    def test_material_array_shapes(self):
        sim = make_single_layer_2d(Nx=11, Ny=11)
        assert sim.k_nodes.shape == (11, 11)
        assert sim.rho_nodes.shape == (11, 11)
        assert sim.cp_nodes.shape == (11, 11)

    def test_face_conductance_shapes(self):
        sim = make_single_layer_2d(Nx=11, Ny=11)
        assert sim.G_x.shape == (10, 11)
        assert sim.G_y.shape == (11, 10)

    def test_cn_coefficient_shapes(self):
        sim = make_single_layer_2d(Nx=11, Ny=11)
        for arr in [sim.r_w, sim.r_e, sim.r_s, sim.r_n]:
            assert arr.shape == (11, 11)

    def test_dx_dy_half_cells(self):
        sim = make_single_layer_2d(Nx=5, Ny=5)
        h = sim.hx
        assert sim.dx[0] == pytest.approx(h / 2)
        assert sim.dx[-1] == pytest.approx(h / 2)
        assert sim.dx[2] == pytest.approx(h)
        assert sim.dy[0] == pytest.approx(h / 2)
        assert sim.dy[-1] == pytest.approx(h / 2)
        assert sim.dy[2] == pytest.approx(h)

    def test_layer2d_is_layer1d(self):
        assert Layer2D is Layer1D

    def test_explicit_dt_must_reach_t_final(self):
        with pytest.raises(ValueError, match="does not divide t_final"):
            make_single_layer_2d(dt=0.3, t_final=1.0)

    def test_computed_dt_uses_2d_raw_heuristic_as_max_step(self):
        sim = make_single_layer_2d(
            Nx=11, Ny=11, lam_target=0.3, flux_f=0.0, t_final=0.5,
        )
        dt_raw = compute_dt_2d(sim.hx, 1.0, 0.3, 0.0)
        Nt = len(sim.t) - 1
        assert dt_raw == pytest.approx(0.3 * sim.hx**2 / 2.0)
        assert sim.dt <= dt_raw
        assert sim.t[-1] == pytest.approx(0.5)
        assert sim.dt == pytest.approx(0.5 / Nt)

    def test_nonunit_domain_with_isotropic_spacing(self):
        # Lx=2, Ly=3. Choose node counts so hx=hy=1/19.
        sim = make_single_layer_2d(
            a=2.0, b=4.0, c=-1.0, d=2.0,
            Nx=39, Ny=58,
            layers=[Layer2D(2.0, 4.0, 1.0, 1.0, 1.0)],
            dt=0.005,
            t_final=0.1,
        )
        assert sim.grid_x[0] == pytest.approx(2.0)
        assert sim.grid_x[-1] == pytest.approx(4.0)
        assert sim.grid_y[0] == pytest.approx(-1.0)
        assert sim.grid_y[-1] == pytest.approx(2.0)
        assert sim.hx == pytest.approx(sim.grid_y[1] - sim.grid_y[0])


# ==================== TEST 2: TINY-GRID MATRIX (Nx=4, Ny=3) ====================

class TestInterfaceValidation2D:

    def test_duplicate_off_midpoint_interfaces_in_same_face_slot_raise(self):
        layers = [
            Layer2D(0.0, 0.43, 1.0, 1.0, 2.0),
            Layer2D(0.43, 0.47, 1.0, 1.0, 1.5),
            Layer2D(0.47, 1.0, 1.0, 1.0, 1.0),
        ]
        with pytest.raises(ValueError, match="Multiple interfaces map to face slot 4"):
            FVSolver2D(
                a=0.0, b=1.0, c=0.0, d=1.0,
                Nx=11, Ny=11,
                lam_target=0.5,
                layers=layers,
                t_final=0.1,
                flux_f=0.0, flux_A=0.0,
                t_on=0.0, t_off=0.1, phase=0.0,
                dt=0.001,
            )

    def test_duplicate_slot_error_reports_both_locations(self):
        layers = [
            Layer2D(0.0, 0.40000000001, 1.0, 1.0, 2.0),
            Layer2D(0.40000000001, 0.49999999999, 1.0, 1.0, 1.5),
            Layer2D(0.49999999999, 1.0, 1.0, 1.0, 1.0),
        ]
        with pytest.raises(ValueError) as excinfo:
            FVSolver2D(
                a=0.0, b=1.0, c=0.0, d=1.0,
                Nx=11, Ny=11,
                lam_target=0.5,
                layers=layers,
                t_final=0.1,
                flux_f=0.0, flux_A=0.0,
                t_on=0.0, t_off=0.1, phase=0.0,
                dt=0.001,
            )
        msg = str(excinfo.value)
        assert "face slot 4" in msg
        assert "0.40000000001" in msg
        assert "0.49999999999" in msg


class TestTinyGridMatrix:
    """
    Nx=4, Ny=3 (12 nodes), single material k=1, rho=1, cp=1.
    Domain [0,1]x[0, 2/3] -> hx = hy = 1/3.

    Node layout (i,j) -> global index k = i + j*Nx:
        (0,2)(1,2)(2,2)(3,2)    8  9  10 11
        (0,1)(1,1)(2,1)(3,1)    4  5   6  7
        (0,0)(1,0)(2,0)(3,0)    0  1   2  3
    """

    @pytest.fixture
    def tiny_sim(self):
        layer = Layer2D(x_left=0.0, x_right=1.0, rho=1.0, cp=1.0, k=1.0)
        return FVSolver2D(
            a=0.0, b=1.0, c=0.0, d=2.0 / 3.0,
            Nx=4, Ny=3,
            lam_target=0.5,
            layers=[layer],
            t_final=0.1,
            flux_f=0.0, flux_A=0.0,
            t_on=0.0, t_off=0.1, phase=0.0,
            dt=0.001,
        )

    def test_matrix_shape(self, tiny_sim):
        A = tiny_sim.A_csr
        assert A.shape == (12, 12)

    def test_dirichlet_rows(self, tiny_sim):
        """Nodes with i=Nx-1=3 (k=3,7,11) must be identity rows."""
        A = tiny_sim.A_csr.toarray()
        for j in range(3):
            k = 3 + j * 4  # i=3, j=0,1,2 -> k=3,7,11
            row = A[k, :]
            assert row[k] == pytest.approx(1.0)
            assert np.sum(np.abs(row)) == pytest.approx(1.0), (
                f"Dirichlet row k={k} has off-diagonal entries"
            )

    def test_corner_00_row(self, tiny_sim):
        """Node (0,0), k=0: bottom-left corner. Has east and north neighbors only."""
        A = tiny_sim.A_csr.toarray()
        re = tiny_sim.r_e[0, 0]
        rn = tiny_sim.r_n[0, 0]
        # Main diagonal
        assert A[0, 0] == pytest.approx(1.0 + re + rn)
        # East neighbor: k=1
        assert A[0, 1] == pytest.approx(-re)
        # North neighbor: k=4
        assert A[0, 4] == pytest.approx(-rn)
        # No other nonzeros
        nnz = np.count_nonzero(A[0, :])
        assert nnz == 3

    def test_corner_02_row(self, tiny_sim):
        """Node (0,2), k=8: top-left corner. Has east and south neighbors only."""
        A = tiny_sim.A_csr.toarray()
        re = tiny_sim.r_e[0, 2]
        rs = tiny_sim.r_s[0, 2]
        assert A[8, 8] == pytest.approx(1.0 + re + rs)
        assert A[8, 9] == pytest.approx(-re)
        assert A[8, 4] == pytest.approx(-rs)
        nnz = np.count_nonzero(A[8, :])
        assert nnz == 3

    def test_interior_11_row(self, tiny_sim):
        """Node (1,1), k=5: interior node. Has 4 neighbors."""
        A = tiny_sim.A_csr.toarray()
        rw = tiny_sim.r_w[1, 1]
        re = tiny_sim.r_e[1, 1]
        rs = tiny_sim.r_s[1, 1]
        rn = tiny_sim.r_n[1, 1]
        assert A[5, 5] == pytest.approx(1.0 + rw + re + rs + rn)
        assert A[5, 4] == pytest.approx(-rw)   # west: k=4
        assert A[5, 6] == pytest.approx(-re)   # east: k=6
        assert A[5, 1] == pytest.approx(-rs)   # south: k=1
        assert A[5, 9] == pytest.approx(-rn)   # north: k=9
        nnz = np.count_nonzero(A[5, :])
        assert nnz == 5

    def test_left_edge_01_row(self, tiny_sim):
        """Node (0,1), k=4: left edge (not corner). Has E, S, N neighbors."""
        A = tiny_sim.A_csr.toarray()
        re = tiny_sim.r_e[0, 1]
        rs = tiny_sim.r_s[0, 1]
        rn = tiny_sim.r_n[0, 1]
        assert A[4, 4] == pytest.approx(1.0 + re + rs + rn)
        assert A[4, 5] == pytest.approx(-re)
        assert A[4, 0] == pytest.approx(-rs)
        assert A[4, 8] == pytest.approx(-rn)
        nnz = np.count_nonzero(A[4, :])
        assert nnz == 4

    def test_bottom_edge_10_row(self, tiny_sim):
        """Node (1,0), k=1: bottom edge (not corner). Has W, E, N neighbors."""
        A = tiny_sim.A_csr.toarray()
        rw = tiny_sim.r_w[1, 0]
        re = tiny_sim.r_e[1, 0]
        rn = tiny_sim.r_n[1, 0]
        assert A[1, 1] == pytest.approx(1.0 + rw + re + rn)
        assert A[1, 0] == pytest.approx(-rw)
        assert A[1, 2] == pytest.approx(-re)
        assert A[1, 5] == pytest.approx(-rn)
        nnz = np.count_nonzero(A[1, :])
        assert nnz == 4

    def test_half_cell_doubles_coefficients(self, tiny_sim):
        """At boundary nodes, half-cell dx or dy doubles the coupling r."""
        h = tiny_sim.hx
        k = 1.0
        G = k / h
        rho_cp = 1.0

        # Interior node (1,1): r_w = dt * G / (2 * rho_cp * h)
        r_interior = tiny_sim.dt * G / (2.0 * rho_cp * h)

        # Left boundary (0,1): dx[0] = h/2, so r_e = dt * G / (2 * rho_cp * h/2)
        r_left_e = tiny_sim.dt * G / (2.0 * rho_cp * (h / 2.0))

        assert r_left_e == pytest.approx(2.0 * r_interior)

        # Bottom boundary (1,0): dy[0] = h/2, so r_n = dt * G / (2 * rho_cp * h/2)
        r_bot_n = tiny_sim.dt * G / (2.0 * rho_cp * (h / 2.0))
        assert r_bot_n == pytest.approx(2.0 * r_interior)

    def test_dense_reference_matches_sparse(self, tiny_sim):
        """Build a fully explicit dense 12x12 matrix and compare to sparse."""
        Nx, Ny = tiny_sim.Nx, tiny_sim.Ny
        n = Nx * Ny
        A_dense = np.zeros((n, n))

        for j in range(Ny):
            for i in range(Nx):
                k = i + j * Nx
                if i == Nx - 1:
                    A_dense[k, k] = 1.0
                    continue
                rw = tiny_sim.r_w[i, j]
                re = tiny_sim.r_e[i, j]
                rs = tiny_sim.r_s[i, j]
                rn = tiny_sim.r_n[i, j]
                A_dense[k, k] = 1.0 + rw + re + rs + rn
                if i > 0:
                    A_dense[k, i - 1 + j * Nx] = -rw
                if i < Nx - 1:
                    A_dense[k, i + 1 + j * Nx] = -re
                if j > 0:
                    A_dense[k, i + (j - 1) * Nx] = -rs
                if j < Ny - 1:
                    A_dense[k, i + (j + 1) * Nx] = -rn

        np.testing.assert_allclose(
            tiny_sim.A_csr.toarray(), A_dense, atol=1e-15,
            err_msg="Sparse matrix does not match dense reference"
        )

    def test_loop_rhs_vs_explicit(self, tiny_sim):
        """Verify cn_step RHS matches hand-built RHS for arbitrary T^n."""
        Nx, Ny = tiny_sim.Nx, tiny_sim.Ny
        rng = np.random.default_rng(42)
        Tn = rng.standard_normal((Nx, Ny)) + 300.0

        # Get the RHS by solving A * T^{n+1} = rhs => rhs = A * T^{n+1}
        # But easier: rebuild RHS from scratch and compare with cn_step output.
        # Since cn_step returns T^{n+1} = A^{-1} rhs, we can check via:
        # A * cn_step(Tn, t) should equal the explicitly built RHS.

        # Use zero flux and no source for clean comparison.
        sim = FVSolver2D(
            a=0.0, b=1.0, c=0.0, d=2.0 / 3.0,
            Nx=4, Ny=3,
            lam_target=0.5,
            layers=[Layer2D(0.0, 1.0, 1.0, 1.0, 1.0)],
            t_final=0.1,
            flux_f=0.0, flux_A=0.0,
            t_on=0.0, t_off=0.1, phase=0.0,
            dt=tiny_sim.dt,
            q_left_fn=lambda t: 0.0,
        )

        Tnp1 = sim.cn_step(Tn, 0.0)
        # Verify A * T^{n+1} = rhs by rebuilding rhs explicitly
        rhs_explicit = np.zeros((Nx, Ny))
        for j in range(Ny):
            for i in range(Nx):
                if i == Nx - 1:
                    rhs_explicit[i, j] = sim.T_right(sim.dt)
                    continue
                rw = sim.r_w[i, j]
                re = sim.r_e[i, j]
                rs = sim.r_s[i, j]
                rn = sim.r_n[i, j]
                T_W = Tn[i - 1, j] if i > 0 else 0.0
                T_E = Tn[i + 1, j] if i < Nx - 1 else 0.0
                T_S = Tn[i, j - 1] if j > 0 else 0.0
                T_N = Tn[i, j + 1] if j < Ny - 1 else 0.0
                rhs_explicit[i, j] = (
                    rw * T_W
                    + (1.0 - rw - re - rs - rn) * Tn[i, j]
                    + re * T_E
                    + rs * T_S
                    + rn * T_N
                )

        # A * T^{n+1} should equal rhs_explicit
        A = sim.A_csr.toarray()
        lhs = A @ Tnp1.ravel(order="F")
        np.testing.assert_allclose(
            lhs, rhs_explicit.ravel(order="F"), atol=1e-12,
            err_msg="A * T^{n+1} != RHS from explicit loop"
        )


# ==================== TEST 3: CONSTANT SOLUTION ====================

class TestConstantSolution2D:

    def test_constant_single_layer(self):
        """T=300 with zero flux must stay constant."""
        sim = FVSolver2D(
            a=0.0, b=1.0, c=0.0, d=1.0,
            Nx=11, Ny=11,
            lam_target=0.5,
            layers=[Layer2D(0.0, 1.0, 1.0, 1.0, 1.0)],
            t_final=0.1,
            flux_f=0.0, flux_A=0.0,
            t_on=0.0, t_off=0.1, phase=0.0,
            q_left_fn=lambda t: 0.0,
            T_right_fn=lambda t: 300.0,
        )
        T0 = np.full((11, 11), 300.0)
        _, _, _, T_final = sim.solve(T0=T0, store_trajectory=False)
        max_err = np.max(np.abs(T_final - 300.0))
        assert max_err < 1e-10, f"Constant solution drifted: max error = {max_err}"

    def test_constant_multilayer(self):
        """Two layers, T=300, zero flux: must stay constant."""
        layers = [
            Layer2D(0.0, 0.5, rho=8000.0, cp=500.0, k=50.0),
            Layer2D(0.5, 1.0, rho=1.0, cp=1.0, k=5.0),
        ]
        # Need even Nx so x=0.5 is face-aligned.
        # Nx=20: h=1/19. face_coord = 0.5*19 - 0.5 = 9. Integer. Good.
        # Ny=20: hy=1/19 = hx. Good.
        Nx = Ny = 20
        sim = FVSolver2D(
            a=0.0, b=1.0, c=0.0, d=1.0,
            Nx=Nx, Ny=Ny,
            lam_target=0.5,
            layers=layers,
            t_final=0.1,
            flux_f=0.0, flux_A=0.0,
            t_on=0.0, t_off=0.1, phase=0.0,
            q_left_fn=lambda t: 0.0,
            T_right_fn=lambda t: 300.0,
        )
        T0 = np.full((Nx, Ny), 300.0)
        _, _, _, T_final = sim.solve(T0=T0, store_trajectory=False)
        max_err = np.max(np.abs(T_final - 300.0))
        assert max_err < 1e-8, f"Constant multilayer drifted: max error = {max_err}"

    def test_constant_multilayer_with_resistance(self):
        """Two layers + R_c, T=300, zero flux: must stay constant."""
        layers = [
            Layer2D(0.0, 0.5, rho=1.0, cp=1.0, k=2.0),
            Layer2D(0.5, 1.0, rho=1.0, cp=1.0, k=1.0),
        ]
        Nx = Ny = 20
        sim = FVSolver2D(
            a=0.0, b=1.0, c=0.0, d=1.0,
            Nx=Nx, Ny=Ny,
            lam_target=0.5,
            layers=layers,
            t_final=0.1,
            flux_f=0.0, flux_A=0.0,
            t_on=0.0, t_off=0.1, phase=0.0,
            q_left_fn=lambda t: 0.0,
            T_right_fn=lambda t: 300.0,
            interface_R=[0.5],
        )
        T0 = np.full((Nx, Ny), 300.0)
        _, _, _, T_final = sim.solve(T0=T0, store_trajectory=False)
        max_err = np.max(np.abs(T_final - 300.0))
        assert max_err < 1e-10, f"Constant+R_c drifted: max error = {max_err}"


# ============= TEST 3b: PER-ROW (SPATIALLY-VARYING) R_c(y) =============

class TestPerRowInterfaceResistance:
    """Spatially-varying interface resistance R_c(y) along a vertical interface.

    A scalar R_c must remain byte-for-byte identical to the legacy uniform path;
    a (Ny,) array applies the series-resistance formula per row using the actual
    one-sided distances to the interface.
    """

    @staticmethod
    def _two_layer_kwargs(Nx=20, Ny=20):
        return dict(
            a=0.0, b=1.0, c=0.0, d=1.0,
            Nx=Nx, Ny=Ny,
            lam_target=0.5,
            layers=[
                Layer2D(0.0, 0.5, rho=1.0, cp=1.0, k=2.0),
                Layer2D(0.5, 1.0, rho=1.0, cp=1.0, k=1.0),
            ],
            t_final=0.1,
            flux_f=0.0, flux_A=0.0,
            t_on=0.0, t_off=0.1, phase=0.0,
            q_left_fn=lambda t: 0.0,
            T_right_fn=lambda t: 300.0,
        )

    def test_uniform_array_matches_scalar_bitwise(self):
        """A constant (Ny,) R_c array reproduces the scalar G_x exactly."""
        Rc = 0.37
        kw = self._two_layer_kwargs()
        sim_scalar = FVSolver2D(interface_R=[Rc], **kw)
        sim_array = FVSolver2D(interface_R=[np.full(kw["Ny"], Rc)], **kw)
        G_scalar = sim_scalar._build_face_conductance_x()
        G_array = sim_array._build_face_conductance_x()
        assert G_scalar.shape == (kw["Nx"] - 1, kw["Ny"])
        assert np.array_equal(G_scalar, G_array)

    def test_per_row_uses_actual_one_sided_distances(self):
        """The interface row equals 1/((x_I-x_L)/k_L + Rc[j] + (x_R-x_I)/k_R)."""
        Ny = 20
        kw = self._two_layer_kwargs(Ny=Ny)
        rng = np.random.default_rng(0)
        Rc = 0.1 + 0.5 * rng.random(Ny)
        sim = FVSolver2D(interface_R=[Rc], **kw)
        G = sim._build_face_conductance_x()

        # Locate the single interface face and pull the solver's own geometry.
        assert len(sim.interface_face_map) == 1
        i = next(iter(sim.interface_face_map))
        left_idx, right_idx = sim.interface_face_map[i]
        h_L, h_R = sim.interface_offsets[i]
        kL = sim.layers[left_idx].k
        kR = sim.layers[right_idx].k
        expected_row = 1.0 / (h_L / kL + Rc + h_R / kR)
        np.testing.assert_allclose(G[i, :], expected_row, rtol=0, atol=0)

        # Non-interface faces stay row-uniform (independent of j).
        for ii in range(sim.Nx - 1):
            if ii == i:
                continue
            assert np.allclose(G[ii, :], G[ii, 0], rtol=0, atol=0)

    def test_flat_void_profile_full_solve_matches_scalar(self):
        """A=0 (flat profile) full solve equals the constant-R_c solve.

        Proves existing constant-R_c benchmarks are unaffected by the per-row
        code path: a flat make_rc_sin_profile is bit-for-bit the scalar solve.
        """
        from src.physics.internal_source import make_rc_sin_profile

        Nx = Ny = 24
        kw = self._two_layer_kwargs(Nx=Nx, Ny=Ny)
        R_base = 0.42
        sim_scalar = FVSolver2D(interface_R=[R_base], **kw)
        flat = make_rc_sin_profile(
            sim_scalar.grid_y, R_base=R_base, A=0.0
        )
        sim_array = FVSolver2D(interface_R=[flat], **kw)

        # A non-trivial transient: warm left half, cool right half.
        T0 = np.full((Nx, Ny), 300.0)
        T0[: Nx // 2, :] = 350.0
        _, _, _, T_scalar = sim_scalar.solve(T0=T0.copy(), store_trajectory=False)
        _, _, _, T_array = sim_array.solve(T0=T0.copy(), store_trajectory=False)
        assert np.array_equal(T_scalar, T_array)

    def test_void_increases_local_resistance(self):
        """A central void (higher R_c) lowers interface conductance there."""
        from src.physics.internal_source import make_rc_sin_profile

        Ny = 40
        kw = self._two_layer_kwargs(Nx=Ny, Ny=Ny)
        prof = make_rc_sin_profile(
            np.linspace(0.0, 1.0, Ny), R_base=0.1, A=1.0
        )
        sim = FVSolver2D(interface_R=[prof], **kw)
        G = sim._build_face_conductance_x()
        i = next(iter(sim.interface_face_map))
        # Center row (peak R_c) must have strictly lower conductance than edges.
        center = Ny // 2
        assert G[i, center] < G[i, 0]
        assert G[i, center] < G[i, -1]

    def test_wrong_length_array_rejected(self):
        kw = self._two_layer_kwargs(Ny=20)
        with pytest.raises(ValueError):
            FVSolver2D(interface_R=[np.ones(19)], **kw)

    def test_two_d_array_rejected(self):
        kw = self._two_layer_kwargs(Ny=20)
        with pytest.raises(ValueError):
            FVSolver2D(interface_R=[np.ones((20, 1))], **kw)

    def test_negative_entry_in_array_rejected(self):
        kw = self._two_layer_kwargs(Ny=20)
        bad = np.full(20, 0.1)
        bad[5] = -0.01
        with pytest.raises(ValueError):
            FVSolver2D(interface_R=[bad], **kw)


# ==================== TEST 4: 1D EQUIVALENCE ====================

class TestOneDEquivalence:
    """Run a y-independent problem in 2D and compare to 1D solver output."""

    def test_y_independent_matches_1d(self):
        """2D with y-independent IC and BCs must match 1D at every x-node."""
        N = 21
        layer_1d = Layer1D(0.0, 1.0, rho=1.0, cp=1.0, k=1.0)
        sim_1d = FVSolver1D(
            a=0.0, b=1.0, N=N,
            lam_target=0.5,
            layers=[layer_1d],
            t_final=0.1,
            flux_f=2.0, flux_A=50.0,
            t_on=0.0, t_off=0.1, phase=0.0,
            dt=0.001,
        )

        # 2D: same x-grid, any Ny with matching h
        # h_x = 1/(N-1) = 1/20 = 0.05
        # Need d-c = h_x * (Ny-1) for some integer Ny.
        # Choose d-c = 1.0, Ny = N = 21 -> hy = 0.05 = hx.
        sim_2d = FVSolver2D(
            a=0.0, b=1.0, c=0.0, d=1.0,
            Nx=N, Ny=N,
            lam_target=0.5,
            layers=[Layer2D(0.0, 1.0, 1.0, 1.0, 1.0)],
            t_final=0.1,
            flux_f=2.0, flux_A=50.0,
            t_on=0.0, t_off=0.1, phase=0.0,
            dt=0.001,
        )

        # Use same IC: broadcast 1D IC across y
        T0_1d = np.full(N, 300.0)
        T0_2d = np.tile(T0_1d[:, np.newaxis], (1, N))

        _, _, T_final_1d = sim_1d.solve(T0=T0_1d, store_trajectory=False)
        _, _, _, T_final_2d = sim_2d.solve(T0=T0_2d, store_trajectory=False)

        # Each column of 2D should match the 1D solution
        for j in range(N):
            np.testing.assert_allclose(
                T_final_2d[:, j], T_final_1d, atol=1e-10,
                err_msg=f"2D column j={j} does not match 1D solution"
            )

    def test_y_independent_multilayer_matches_1d(self):
        """Two-layer y-independent 2D matches 1D."""
        N = 20  # even, so interface at x=0.5 is face-aligned
        layers_1d = [
            Layer1D(0.0, 0.5, rho=1.0, cp=1.0, k=2.0),
            Layer1D(0.5, 1.0, rho=1.0, cp=1.0, k=1.0),
        ]
        layers_2d = [
            Layer2D(0.0, 0.5, rho=1.0, cp=1.0, k=2.0),
            Layer2D(0.5, 1.0, rho=1.0, cp=1.0, k=1.0),
        ]
        # h = 1/19... no, N=20 on [0,1] gives h = 1/19. That won't have
        # x=0.5 face-aligned. Need N such that 0.5 falls on a face.
        # face_idx = (0.5 - 0)/h - 0.5 must be integer.
        # h = 1/(N-1). 0.5*(N-1) - 0.5 = (N-2)/2 must be integer => N even.
        # N=20: h=1/19. 0.5*19 - 0.5 = 9. OK, face_idx=9. But 0.5/h=9.5, not
        # on a node. Good.
        # But Ny must give hy = hx = 1/19. d-c = 1.0, Ny: 1/(Ny-1) = 1/19 => Ny=20.
        sim_1d = FVSolver1D(
            a=0.0, b=1.0, N=N,
            lam_target=0.5,
            layers=layers_1d,
            t_final=0.1,
            flux_f=2.0, flux_A=50.0,
            t_on=0.0, t_off=0.1, phase=0.0,
            interface_R=[0.1],
            dt=0.001,
        )
        sim_2d = FVSolver2D(
            a=0.0, b=1.0, c=0.0, d=1.0,
            Nx=N, Ny=N,
            lam_target=0.5,
            layers=layers_2d,
            t_final=0.1,
            flux_f=2.0, flux_A=50.0,
            t_on=0.0, t_off=0.1, phase=0.0,
            interface_R=[0.1],
            dt=0.001,
        )

        T0_1d = np.full(N, 300.0)
        T0_2d = np.tile(T0_1d[:, np.newaxis], (1, N))

        _, _, T_final_1d = sim_1d.solve(T0=T0_1d, store_trajectory=False)
        _, _, _, T_final_2d = sim_2d.solve(T0=T0_2d, store_trajectory=False)

        for j in range(N):
            np.testing.assert_allclose(
                T_final_2d[:, j], T_final_1d, atol=1e-10,
                err_msg=f"Multilayer 2D column j={j} != 1D"
            )


# ==================== TEST 5: ADIABATIC BC ====================

class TestAdiabaticBC:

    def test_adiabatic_top_bottom_symmetry(self):
        """Heat from left with symmetric y-IC: solution must be y-independent."""
        N = 11
        sim = make_single_layer_2d(
            Nx=N, Ny=N, t_final=0.1, dt=0.001,
            flux_A=50.0,
        )
        T0 = np.full((N, N), 300.0)
        _, _, _, T_final = sim.solve(T0=T0, store_trajectory=False)

        # All columns should be identical (y-independent)
        for j in range(1, N):
            np.testing.assert_allclose(
                T_final[:, j], T_final[:, 0], atol=1e-10,
                err_msg=f"Column {j} differs from column 0 — adiabatic BC broken"
            )

    def test_y_varying_ic_damps(self):
        """Start with y-varying perturbation: it should damp over time.
        With adiabatic top/bottom, a cosine mode in y should decay."""
        Nx, Ny = 11, 11
        sim = FVSolver2D(
            a=0.0, b=1.0, c=0.0, d=1.0,
            Nx=Nx, Ny=Ny,
            lam_target=0.5,
            layers=[Layer2D(0.0, 1.0, 1.0, 1.0, 1.0)],
            t_final=0.5,
            flux_f=0.0, flux_A=0.0,
            t_on=0.0, t_off=0.5, phase=0.0,
            q_left_fn=lambda t: 0.0,
            T_right_fn=lambda t: 300.0,
            dt=0.001,
        )
        # IC: 300 + perturbation that varies in y
        T0 = np.full((Nx, Ny), 300.0)
        for j in range(Ny):
            y = sim.grid_y[j]
            T0[:Nx - 1, j] += 10.0 * np.cos(np.pi * y)
        # Right BC nodes stay at 300
        T0[Nx - 1, :] = 300.0

        _, _, _, T_final = sim.solve(T0=T0, store_trajectory=False)

        # The y-variation should have damped
        y_variation_init = np.max(np.ptp(T0[:Nx - 1, :], axis=1))
        y_variation_final = np.max(np.ptp(T_final[:Nx - 1, :], axis=1))
        assert y_variation_final < y_variation_init * 0.5, (
            f"y-variation did not damp: init={y_variation_init:.4f}, "
            f"final={y_variation_final:.4f}"
        )


# ==================== TEST 6: TIME-DEPENDENT DIRICHLET ====================

class TestTimeDependentDirichlet:

    def test_right_bc_updated_each_step(self):
        """With T_right(t) = 300 + 10*sin(t), right column must track the BC."""
        N = 11
        sim = FVSolver2D(
            a=0.0, b=1.0, c=0.0, d=1.0,
            Nx=N, Ny=N,
            lam_target=0.5,
            layers=[Layer2D(0.0, 1.0, 1.0, 1.0, 1.0)],
            t_final=0.5,
            flux_f=0.0, flux_A=0.0,
            t_on=0.0, t_off=0.5, phase=0.0,
            q_left_fn=lambda t: 0.0,
            T_right_fn=lambda t: 300.0 + 10.0 * np.sin(t),
            dt=0.01,
        )
        T0 = np.full((N, N), 300.0)
        _, _, _, T_hist = sim.solve(T0=T0, store_trajectory=True)

        # Check that the right column tracks T_right at each time step
        for n, tn in enumerate(sim.t):
            expected = 300.0 + 10.0 * np.sin(tn)
            np.testing.assert_allclose(
                T_hist[n, N - 1, :], expected, atol=1e-10,
                err_msg=f"Right BC wrong at step {n}, t={tn:.4f}"
            )


# ==================== TEST 7: SOLVE OUTPUT ====================

class TestSolveOutput:

    def test_final_only_shapes(self):
        sim = make_single_layer_2d(Nx=11, Ny=11)
        T0 = np.full((11, 11), sim.T_right(0.0))
        t, gx, gy, T_final = sim.solve(T0=T0, store_trajectory=False)
        assert T_final.shape == (11, 11)
        assert gx.shape == (11,)
        assert gy.shape == (11,)

    def test_trajectory_shapes(self):
        sim = make_single_layer_2d(Nx=11, Ny=11)
        T0 = np.full((11, 11), sim.T_right(0.0))
        t, gx, gy, T_hist = sim.solve(T0=T0, store_trajectory=True)
        Nt = len(sim.t)
        assert T_hist.shape == (Nt, 11, 11)

    def test_wrong_T0_shape_raises(self):
        sim = make_single_layer_2d(Nx=11, Ny=11)
        T0 = np.ones((11, 5)) * 300.0
        with pytest.raises(ValueError, match="T0 shape"):
            sim.solve(T0=T0)

    def test_trajectory_final_matches_final_only(self):
        sim = make_single_layer_2d(Nx=11, Ny=11)
        T0 = np.full((11, 11), 300.0)
        _, _, _, T_final = sim.solve(T0=T0, store_trajectory=False)
        _, _, _, T_hist = sim.solve(T0=T0, store_trajectory=True)
        np.testing.assert_allclose(T_final, T_hist[-1], atol=1e-12)

    def test_temperatures_finite(self):
        sim = make_single_layer_2d(Nx=11, Ny=11, flux_A=50.0)
        T0 = np.full((11, 11), sim.T_right(0.0))
        _, _, _, T_hist = sim.solve(T0=T0, store_trajectory=True)
        assert np.all(np.isfinite(T_hist))


# ==================== TEST 8: LOOP VS VECTORIZED CN STEP ====================

class TestLoopVsVectorized:
    """Verify vectorized cn_step matches loop-based cn_step_loop to machine precision."""

    def test_single_layer_random_ic(self):
        """Single-layer with random IC and nonzero flux."""
        Nx, Ny = 11, 11
        sim = FVSolver2D(
            a=0.0, b=1.0, c=0.0, d=1.0,
            Nx=Nx, Ny=Ny,
            lam_target=0.5,
            layers=[Layer2D(0.0, 1.0, 1.0, 1.0, 1.0)],
            t_final=0.1,
            flux_f=2.0, flux_A=50.0,
            t_on=0.0, t_off=0.1, phase=0.0,
            dt=0.001,
        )
        rng = np.random.default_rng(42)
        Tn = rng.standard_normal((Nx, Ny)) + 300.0

        T_loop = sim.cn_step_loop(Tn, 0.01)
        T_vec = sim.cn_step(Tn, 0.01)
        np.testing.assert_allclose(T_vec, T_loop, atol=1e-12)

    def test_multilayer_with_source(self):
        """Two-layer + source + flux: loop and vectorized must agree."""
        Nx = Ny = 20
        layers = [
            Layer2D(0.0, 0.5, rho=1.5, cp=1.0, k=2.0),
            Layer2D(0.5, 1.0, rho=0.8, cp=1.0, k=1.0),
        ]

        def src(X, Y, t):
            return 10.0 * np.sin(2 * np.pi * t) * np.ones_like(X)

        sim = FVSolver2D(
            a=0.0, b=1.0, c=0.0, d=1.0,
            Nx=Nx, Ny=Ny,
            lam_target=0.5,
            layers=layers,
            t_final=0.1,
            flux_f=2.0, flux_A=50.0,
            t_on=0.0, t_off=0.1, phase=0.0,
            interface_R=[0.1],
            source=src,
            dt=0.001,
        )
        rng = np.random.default_rng(123)
        Tn = rng.standard_normal((Nx, Ny)) + 300.0

        T_loop = sim.cn_step_loop(Tn, 0.02)
        T_vec = sim.cn_step(Tn, 0.02)
        np.testing.assert_allclose(T_vec, T_loop, atol=1e-12)

    def test_time_dep_dirichlet(self):
        """Time-dependent right BC: loop and vectorized must agree."""
        Nx = Ny = 11
        sim = FVSolver2D(
            a=0.0, b=1.0, c=0.0, d=1.0,
            Nx=Nx, Ny=Ny,
            lam_target=0.5,
            layers=[Layer2D(0.0, 1.0, 1.0, 1.0, 1.0)],
            t_final=0.5,
            flux_f=2.0, flux_A=50.0,
            t_on=0.0, t_off=0.5, phase=0.0,
            T_right_fn=lambda t: 300.0 + 10.0 * np.sin(t),
            dt=0.01,
        )
        rng = np.random.default_rng(99)
        Tn = rng.standard_normal((Nx, Ny)) + 300.0

        T_loop = sim.cn_step_loop(Tn, 0.05)
        T_vec = sim.cn_step(Tn, 0.05)
        np.testing.assert_allclose(T_vec, T_loop, atol=1e-12)

    def test_with_q_left_integral_fn(self):
        """Loop vs vectorized must agree to 1e-12 when q_left_integral_fn is used."""
        from src.physics.boundary_forcing import build_qL, build_qL_integral

        Nx = Ny = 13
        y_grid = np.linspace(0.0, 1.0, Ny)
        temporal_params = {"Np": 2, "A_list": [120.0, -80.0],
                           "t_list": [0.005, 0.04], "dt_list": [0.01, 0.015]}
        spatial_params = {"y_c": 0.5, "sigma_y": 0.15}

        q_left, _ = build_qL("pulse_train", temporal_params,
                             "gaussian", spatial_params, y_grid)
        q_int, _ = build_qL_integral("pulse_train", temporal_params,
                                     "gaussian", spatial_params, y_grid)

        sim = FVSolver2D(
            a=0.0, b=1.0, c=0.0, d=1.0,
            Nx=Nx, Ny=Ny,
            lam_target=0.5,
            layers=[Layer2D(0.0, 1.0, 1.0, 1.0, 1.0)],
            t_final=0.1,
            flux_f=0.0, flux_A=0.0,
            t_on=0.0, t_off=0.1, phase=0.0,
            dt=0.001,
            q_left_fn=q_left,
            q_left_integral_fn=q_int,
        )
        rng = np.random.default_rng(7)
        Tn = rng.standard_normal((Nx, Ny)) + 300.0

        T_loop = sim.cn_step_loop(Tn, 0.02)
        T_vec = sim.cn_step(Tn, 0.02)
        np.testing.assert_allclose(T_vec, T_loop, atol=1e-12)


# ==================== TEST 8b: INTEGRAL LEFT-FLUX FORCING =========

class TestIntegralLeftFlux:
    """Verify that q_left_integral_fn captures sub-step pulses that endpoint sampling misses."""

    def _make_solvers(self, dt, pulse_t, pulse_dt, A):
        """Build two identical solvers; one with endpoint sampling, one with integral fn."""
        from src.physics.boundary_forcing import build_qL, build_qL_integral

        Nx = Ny = 9
        y_grid = np.linspace(0.0, 1.0, Ny)
        temporal_params = {"Np": 1, "A_list": [A], "t_list": [pulse_t], "dt_list": [pulse_dt]}
        spatial_params = {}

        q_left, _ = build_qL("pulse_train", temporal_params,
                             "uniform", spatial_params, y_grid)
        q_int, _ = build_qL_integral("pulse_train", temporal_params,
                                     "uniform", spatial_params, y_grid)

        common = dict(
            a=0.0, b=1.0, c=0.0, d=1.0, Nx=Nx, Ny=Ny,
            lam_target=0.5, layers=[Layer2D(0.0, 1.0, 1.0, 1.0, 1.0)],
            t_final=10.0 * dt,
            flux_f=0.0, flux_A=0.0,
            t_on=0.0, t_off=10.0 * dt, phase=0.0,
            dt=dt,
        )
        sim_endpoint = FVSolver2D(q_left_fn=q_left, **common)
        sim_integral = FVSolver2D(q_left_fn=q_left, q_left_integral_fn=q_int, **common)
        return sim_endpoint, sim_integral

    def test_pulse_inside_step_endpoint_misses_integral_captures(self):
        """Pulse strictly inside (tn, tn+dt) with q_left(tn)=q_left(tn+dt)=0 must
        produce zero injected energy under endpoint sampling and strictly positive
        injection under the integral path."""
        dt = 0.01
        # Pulse centered at tn=0 + dt/2, width 0.2*dt — fully inside (0, dt).
        pulse_t = 0.4 * dt
        pulse_dt = 0.2 * dt
        A = 500.0
        sim_endpoint, sim_integral = self._make_solvers(dt, pulse_t, pulse_dt, A)

        # Sanity: q_left at both endpoints is zero.
        assert sim_endpoint.q_left(0.0)[0] == pytest.approx(0.0)
        assert sim_endpoint.q_left(dt)[0] == pytest.approx(0.0)

        T0 = np.full((sim_endpoint.Nx, sim_endpoint.Ny), 300.0)
        T_end_endpoint = sim_endpoint.cn_step(T0.copy(), 0.0)
        T_end_integral = sim_integral.cn_step(T0.copy(), 0.0)

        # Endpoint: no flux contribution and no source → field stays exactly T0.
        np.testing.assert_allclose(T_end_endpoint, T0, atol=1e-12)
        # Integral path: row 0 (and everywhere via diffusion) must rise above T0.
        assert T_end_integral[0, :].min() > 300.0
        assert T_end_integral.max() > T_end_endpoint.max()

    def test_smooth_sin_integral_agrees_with_endpoint(self):
        """On slowly varying sin (f·dt ≪ 1), integral and endpoint paths should
        agree to a loose tolerance after several steps — regression sanity check."""
        from src.physics.boundary_forcing import build_qL, build_qL_integral

        Nx = Ny = 11
        y_grid = np.linspace(0.0, 1.0, Ny)
        temporal_params = {"A": 50.0, "f": 1.0, "t_on": 0.0, "t_off": 1.0,
                           "phase": 0.0, "tukey_alpha": 0.2}
        spatial_params = {}

        q_left, _ = build_qL("sin", temporal_params, "uniform", spatial_params, y_grid)
        q_int, _ = build_qL_integral("sin", temporal_params, "uniform", spatial_params, y_grid)

        common = dict(
            a=0.0, b=1.0, c=0.0, d=1.0, Nx=Nx, Ny=Ny,
            lam_target=0.5, layers=[Layer2D(0.0, 1.0, 1.0, 1.0, 1.0)],
            t_final=0.2, flux_f=0.0, flux_A=0.0,
            t_on=0.0, t_off=0.2, phase=0.0,
            dt=0.001,  # f * dt = 1e-3, very smooth at this resolution
        )
        sim_endpoint = FVSolver2D(q_left_fn=q_left, **common)
        sim_integral = FVSolver2D(q_left_fn=q_left, q_left_integral_fn=q_int, **common)

        T0 = np.full((Nx, Ny), 300.0)
        _, _, _, T_hist_e = sim_endpoint.solve(T0=T0.copy(), store_trajectory=True)
        _, _, _, T_hist_i = sim_integral.solve(T0=T0.copy(), store_trajectory=True)

        # Loose tolerance: both paths are O(dt^2) for smooth forcing but not identical.
        np.testing.assert_allclose(T_hist_i[-1], T_hist_e[-1], atol=1e-4, rtol=1e-3)


class TestLeftFluxSignConvention:
    """Positive q_left is inward heat flux at the left boundary."""

    def test_positive_left_flux_heats_and_negative_left_flux_cools(self):
        Nx = Ny = 9
        T0 = np.full((Nx, Ny), 300.0)

        common = dict(
            Nx=Nx, Ny=Ny,
            t_final=0.01,
            dt=0.01,
            flux_A=0.0,
            q_left_integral_fn=None,
        )
        sim_pos = make_single_layer_2d(q_left_fn=lambda t: 100.0, **common)
        sim_neg = make_single_layer_2d(q_left_fn=lambda t: -100.0, **common)

        T_pos = sim_pos.cn_step(T0.copy(), 0.0)
        T_neg = sim_neg.cn_step(T0.copy(), 0.0)

        assert T_pos[0, :].mean() > 300.0
        assert T_neg[0, :].mean() < 300.0
        assert T_pos[0, :].mean() > T_neg[0, :].mean()


class TestNonsmoothForcingReference:
    """Production nonsmooth forcings converge toward smaller-dt integral-path references."""

    @staticmethod
    def _solve_family(temporal_family, temporal_params, dt):
        from src.physics.boundary_forcing import build_qL, build_qL_integral

        Nx = Ny = 9
        t_final = 0.08
        y_grid = np.linspace(0.0, 1.0, Ny)
        q_left, _ = build_qL(temporal_family, temporal_params, "uniform", {}, y_grid)
        q_int, _ = build_qL_integral(temporal_family, temporal_params, "uniform", {}, y_grid)
        sim = FVSolver2D(
            a=0.0, b=1.0, c=0.0, d=1.0,
            Nx=Nx, Ny=Ny,
            lam_target=0.5,
            layers=[Layer2D(0.0, 1.0, 1.0, 1.0, 1.0)],
            t_final=t_final,
            flux_f=0.0, flux_A=0.0,
            t_on=0.0, t_off=t_final, phase=0.0,
            dt=dt,
            q_left_fn=q_left,
            q_left_integral_fn=q_int,
        )
        T0 = np.full((Nx, Ny), 300.0)
        _, _, _, T_final = sim.solve(T0=T0, store_trajectory=False)
        return T_final

    @pytest.mark.parametrize(
        ("temporal_family", "temporal_params"),
        [
            ("exp", {"A": 160.0, "t0": 0.013, "tau": 0.012}),
            (
                "exp_train",
                {"Np": 2, "A_list": [130.0, 80.0], "t_list": [0.011, 0.037], "tau_list": [0.009, 0.017]},
            ),
            (
                "pulse_train",
                {"Np": 2, "A_list": [150.0, 90.0], "t_list": [0.013, 0.041], "dt_list": [0.014, 0.017]},
            ),
        ],
    )
    def test_integral_path_converges_toward_refined_reference(self, temporal_family, temporal_params):
        dts = [0.01, 0.005, 0.0025]
        reference = self._solve_family(temporal_family, temporal_params, dt=0.000625)

        errors = [
            float(np.sqrt(np.mean((self._solve_family(temporal_family, temporal_params, dt) - reference) ** 2)))
            for dt in dts
        ]

        assert errors[1] < errors[0], errors
        assert errors[2] < errors[1], errors
        assert errors[2] < 0.5 * errors[0], errors


class TestSpatialTemporalCoupling:
    """The left-boundary temperature response follows production s_y profiles."""

    @staticmethod
    def _one_step_response(spatial_family, spatial_params):
        from src.physics.boundary_forcing import build_qL, build_qL_integral

        Nx = Ny = 41
        y_grid = np.linspace(0.0, 1.0, Ny)
        temporal_params = {
            "Np": 1,
            "A_list": [300.0],
            "t_list": [0.0013],
            "dt_list": [0.0127],
        }
        q_left, s_vec = build_qL(
            "pulse_train", temporal_params, spatial_family, spatial_params, y_grid
        )
        q_int, _ = build_qL_integral(
            "pulse_train", temporal_params, spatial_family, spatial_params, y_grid
        )
        sim = FVSolver2D(
            a=0.0, b=1.0, c=0.0, d=1.0,
            Nx=Nx, Ny=Ny,
            lam_target=0.5,
            layers=[Layer2D(0.0, 1.0, 1.0, 1.0, 1.0)],
            t_final=0.005,
            flux_f=0.0, flux_A=0.0,
            t_on=0.0, t_off=0.005, phase=0.0,
            dt=0.005,
            q_left_fn=q_left,
            q_left_integral_fn=q_int,
        )
        T0 = np.full((Nx, Ny), 300.0)
        T1 = sim.cn_step(T0.copy(), 0.0)
        return y_grid, s_vec, T1[0, :] - 300.0

    def test_uniform_profile_heats_left_boundary_uniformly(self):
        _, _, response = self._one_step_response("uniform", {})
        np.testing.assert_allclose(response, response.mean(), rtol=0.0, atol=1e-12)

    def test_patch_profile_heats_inside_band_more_than_outside(self):
        y, _, response = self._one_step_response("patch", {"y_c": 0.5, "w": 0.3})
        inside = (y >= 0.35) & (y <= 0.65)
        outside = ~inside
        assert response[inside].mean() > response[outside].mean() + 1e-3

    @pytest.mark.parametrize(
        ("spatial_family", "spatial_params", "expected_peak_y"),
        [
            ("gaussian", {"y_c": 0.35, "sigma_y": 0.10}, 0.35),
            ("triangle", {"y_c": 0.65, "ell": 0.25}, 0.65),
        ],
    )
    def test_smooth_localized_profiles_peak_and_correlate_with_response(
        self, spatial_family, spatial_params, expected_peak_y
    ):
        y, s_vec, response = self._one_step_response(spatial_family, spatial_params)
        assert abs(float(y[np.argmax(response)]) - expected_peak_y) <= y[1] - y[0]

        corr = np.corrcoef(s_vec, response)[0, 1]
        assert corr > 0.98


# ==================== TEST 9: MMS VERIFICATION ====================

from src.physics.mms_2d import (
    run_mms_y_independent,
    run_mms_2d,
    run_mms_2d_interface,
    space_order_test_y_independent,
    time_order_test_y_independent,
    space_order_test_2d,
    time_order_test_2d,
    space_order_test_2d_interface,
    time_order_test_2d_interface,
)


MMS_DT_LIST = [0.0185, 0.00925, 0.004625]


class TestMMS2D:
    """MMS convergence verification for the 2D solver."""

    # --- y-independent (same as 1D, validates 2D infrastructure) ---

    def test_y_independent_error_bounded(self):
        _, _, max_err, l2_err = run_mms_y_independent(N=101)
        assert l2_err < 5e-3
        assert max_err < 1e-2

    def test_y_independent_spatial_order(self):
        p = space_order_test_y_independent([21, 41, 81, 161])
        assert 1.9 < p < 2.1, f"Spatial order {p:.3f}"

    def test_y_independent_temporal_order(self):
        p = time_order_test_y_independent(MMS_DT_LIST)
        assert 1.9 < p < 2.1, f"Temporal order {p:.3f}"

    # --- Full 2D (y-dependent correction) ---

    def test_2d_error_bounded(self):
        _, _, max_err, l2_err = run_mms_2d(N=101)
        assert l2_err < 5e-3
        assert max_err < 5e-3

    def test_2d_spatial_order(self):
        p = space_order_test_2d([21, 41, 81, 161])
        assert 1.9 < p < 2.1, f"Spatial order {p:.3f}"

    def test_2d_temporal_order(self):
        p = time_order_test_2d(MMS_DT_LIST)
        assert 1.9 < p < 2.1, f"Temporal order {p:.3f}"

    # --- Interface (piecewise 2D with R_c) ---

    def test_interface_error_bounded(self):
        _, _, max_err, l2_err = run_mms_2d_interface(N=100)
        assert l2_err < 5e-3
        assert max_err < 5e-3

    def test_interface_spatial_order(self):
        p = space_order_test_2d_interface([50, 100, 200])
        assert 1.9 < p < 2.1, f"Spatial order {p:.3f}"

    def test_interface_temporal_order(self):
        p = time_order_test_2d_interface(MMS_DT_LIST)
        assert 1.9 < p < 2.1, f"Temporal order {p:.3f}"


# ==================== VECTOR LEFT-FLUX (q_L(y, t)) ====================

class TestVectorLeftFlux:
    """Vector-valued q_left_fn(t) -> (Ny,) — separable y-dependent boundary flux."""

    def test_uniform_vector_matches_scalar(self):
        """Constant vector flux gives bit-identical result to scalar flux."""
        Ny = 11
        flux_A, flux_f = 50.0, 2.0
        # Use windowed sin to mimic the default flux exactly.
        from src.physics.fv_solver_1d import windowed_sin_flux
        scalar_q = windowed_sin_flux(flux_f, flux_A, 0.0, 0.1, 0.0, 0.5)

        sim_scalar = make_single_layer_2d(Nx=11, Ny=Ny, q_left_fn=scalar_q)
        T0 = np.full((11, Ny), 300.0)
        _, _, _, T_scalar = sim_scalar.solve(T0=T0, store_trajectory=False)

        sim_vec = make_single_layer_2d(
            Nx=11, Ny=Ny,
            q_left_fn=lambda t: np.full(Ny, scalar_q(t)),
        )
        _, _, _, T_vec = sim_vec.solve(T0=T0, store_trajectory=False)

        assert np.allclose(T_scalar, T_vec, atol=1e-13)

    def test_qleft_shape_validation(self):
        """A wrong-sized array should raise during __init__."""
        Ny = 11
        with pytest.raises(ValueError, match=f"shape \\({Ny},\\)"):
            make_single_layer_2d(
                Nx=11, Ny=Ny,
                q_left_fn=lambda t: np.zeros(Ny - 1),
            )

    def test_nonuniform_qleft_localizes_heating(self):
        """A patch-localized flux raises temperature near its center more than far away."""
        N = 41
        y_grid = np.linspace(0.0, 1.0, N)
        sim = make_single_layer_2d(
            Nx=N, Ny=N, t_final=0.05,
            q_left_fn=lambda t: 100.0 * (
                # narrow patch around y=0.5
                (y_grid >= 0.45) & (y_grid <= 0.55)
            ).astype(float),
        )
        T0 = np.full((N, N), 300.0)
        _, _, _, T_final = sim.solve(T0=T0, store_trajectory=False)

        j_center = N // 2  # y = 0.5
        j_edge = 1          # near y = 0
        # Heating should be larger near the patch than far from it (at x=0).
        assert T_final[0, j_center] > T_final[0, j_edge] + 1e-6
