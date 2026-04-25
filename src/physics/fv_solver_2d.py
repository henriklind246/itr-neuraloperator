import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from src.physics.fv_solver_1d import (
    Layer1D,
    windowed_sin_flux,
    compute_dt,
)

# ---------- SOLVER RESTRICTIONS -----------
#
# Uniform isotropic grid (hx = hy = h).  Deliberate simplification, not a
# theoretical requirement — the FV method generalizes to non-uniform grids.
#
# Layers are vertical slabs (interfaces at x = const, spanning full height).
# Interface locations must lie on x-direction cell faces (midpoints between
# x-nodes).  Off-face and on-node interfaces are NOT allowed.
# Interface thermal resistance supported via interface_R parameter.
#
# Boundary conditions: Neumann left, Dirichlet right, adiabatic top/bottom.

# Alias for clarity — layers are identical to 1D (x-slabs).
Layer2D = Layer1D


def ic(X: np.ndarray, Y: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    # returns a 2d array with shape (Nx, Ny)
    Lx = b - a
    Ly = d - c
    Xn = X - a
    Yn = Y - c
    fx = np.cos((np.pi * Xn) / (2 * Lx)) + 0.1 * np.sin((np.pi * Xn) / Lx)
    fy = np.cos((np.pi * Yn) / (2 * Ly)) + 0.1 * np.sin((np.pi * Yn) / Ly)
    return fx * fy


class FVSolver2D:
    """
    2D Conservative Multilayer FV Solver (Crank-Nicolson)

    Rectangular domain [a,b]×[c,d], uniform isotropic grid h = hx = hy.
    BCs: Neumann left (prescribed flux), Dirichlet right,
         adiabatic (zero flux) top and bottom.
    Layers: vertical slabs with interfaces at x = const.
    """

    def __init__(
        self,
        a: float,
        b: float,
        c: float,
        d: float,
        Nx: int,
        Ny: int,
        lam_target: float,
        layers: list[Layer1D],
        t_final: float,
        flux_f: float,
        flux_A: float,
        t_on: float,
        t_off: float,
        phase: float,
        tukey_alpha: float = 0.5,
        dt=None,
        source=None,
        q_left_fn=None,
        T_right_fn=None,
        interface_R: list[float] | None = None,
        tol: float = 1e-12,
    ):
        self.a, self.b = a, b
        self.c, self.d = c, d
        self.Nx, self.Ny = Nx, Ny
        self.layers = sorted(layers, key=lambda L: L.x_left)
        self.lam_target = float(lam_target)
        self.flux_f = flux_f
        self.flux_A = flux_A
        self.t_on = t_on
        self.t_off = t_off
        self.t_final = t_final
        self.phase = phase
        self.tukey_alpha = tukey_alpha
        self.tol = float(tol)

        # -------- GRID --------
        self.grid_x = np.linspace(a, b, Nx)
        self.grid_y = np.linspace(c, d, Ny)
        self.hx = self.grid_x[1] - self.grid_x[0]
        hy = self.grid_y[1] - self.grid_y[0]

        if not np.isclose(self.hx, hy, rtol=1e-12):
            raise ValueError(
                f"Grid is not isotropic: hx={self.hx:.15e}, hy={hy:.15e}. "
                f"Uniform isotropic grid required (hx == hy)."
            )

        self.X, self.Y = np.meshgrid(self.grid_x, self.grid_y, indexing="ij")
        # X.shape = Y.shape = (Nx, Ny)
        # X stores the physical x-position of node ((i,j))
        # Y stores the physical y-position of node ((i,j))

        # x-direction face positions (same as 1D)
        self.face_positions_x = self.a + (np.arange(Nx - 1) + 0.5) * self.hx

        # -------- VALIDATE GEOMETRY / LAYERS --------
        self._validate_layers_cover_domain()
        self.interface_positions = self._extract_internal_interfaces()
        self.interface_face_map = self._validate_and_index_interfaces()

        # -------- INTERFACE THERMAL RESISTANCE --------
        num_interfaces = len(self.interface_positions)
        if interface_R is None:
            self.interface_R = [0.0] * num_interfaces
        else:
            if len(interface_R) != num_interfaces:
                raise ValueError(
                    f"interface_R has length {len(interface_R)} but there are "
                    f"{num_interfaces} interface(s). Must match exactly."
                )
            for j, Rc in enumerate(interface_R):
                if Rc < 0.0:
                    raise ValueError(
                        f"interface_R[{j}] = {Rc} is negative. "
                        f"Contact resistance must be >= 0."
                    )
            self.interface_R = list(interface_R)

        # -------- BCs --------
        if q_left_fn is not None:
            self.q_left = q_left_fn
        else:
            self.q_left = windowed_sin_flux(
                flux_f, flux_A, t_on, t_off, phase, tukey_alpha
            )

        # Detect whether q_left returns a scalar or a (Ny,) vector. The vector
        # form is needed for separable q_L(y, t) = a(t) s(y); the scalar form
        # is the legacy uniform-flux path. Both broadcast correctly into the
        # rhs[0, :] update in cn_step.
        q_probe = self.q_left(0.0)
        q_arr = np.asarray(q_probe)
        if q_arr.ndim == 0:
            self._q_left_is_vector = False
        else:
            if q_arr.shape != (self.Ny,):
                raise ValueError(
                    f"q_left_fn(t) must return scalar or array of shape "
                    f"({self.Ny},); got shape {q_arr.shape}"
                )
            self._q_left_is_vector = True

        self.T_right = T_right_fn if T_right_fn is not None else (lambda t: 300.0)

        # source term: source(X, Y, t) -> (Nx, Ny) in [W/m^3]
        self.source = source

        # -------- MATERIAL ARRAYS (Nx, Ny) --------
        self._layer_index_1d = self._build_node_layer_indices_1d()
        self.k_nodes = np.zeros((Nx, Ny), dtype=float)
        self.rho_nodes = np.zeros((Nx, Ny), dtype=float)
        self.cp_nodes = np.zeros((Nx, Ny), dtype=float)
        # fill material arrays given the layer each point lives in
        for i in range(Nx):
            j_layer = self._layer_index_1d[i]
            self.k_nodes[i, :] = self.layers[j_layer].k
            self.rho_nodes[i, :] = self.layers[j_layer].rho
            self.cp_nodes[i, :] = self.layers[j_layer].cp

        # -------- CONTROL VOLUME DIMENSIONS --------
        # dx[i] = h/2 at boundaries, h otherwise
        # create an array with length Nx with sizes of control volumes in the x-dir.
        self.dx = np.full(Nx, self.hx)
        self.dx[0] = self.hx / 2.0
        self.dx[Nx - 1] = self.hx / 2.0

        # Off-center interface adjustment: when an interface at face slot face_idx shifts from the node midpoint, the cells flanking it get
        # wider/narrower. Interfaces are forbidden in face slots 0 and Nx-2,
        # so only interior full-width cells are ever touched here.
        for face_idx, (h_L, h_R) in self.interface_offsets.items():
            self.dx[face_idx]     = 0.5 * self.hx + h_L
            self.dx[face_idx + 1] = h_R + 0.5 * self.hx

        # dy[j] = h/2 at boundaries, h otherwise
        # create an array with length Ny with sizes of control volumes in the y-dir.
        self.dy = np.full(Ny, self.hx)
        self.dy[0] = self.hx / 2.0
        self.dy[Ny - 1] = self.hx / 2.0

        # Patch face_positions_x at interface slots to reflect the actual
        # x_int rather than the midpoint (cosmetic: used by diagnostic plots).
        for face_idx, (layer_L, _) in self.interface_face_map.items():
            self.face_positions_x[face_idx] = self.interface_positions[layer_L]

        # -------- FACE CONDUCTANCE --------
        # x-face conductance's allow for contact resistance, that is, interface handling R_c
        self.G_x = self._build_face_conductance_x()  # (Nx-1, Ny)
        # y-face conductance are only defined for non-interfaces (k/h) since interfaces are vertical (x=const)
        self.G_y = self._build_face_conductance_y()  # (Nx, Ny-1)

        # -------- TIME STEP --------
        alpha_nodes = self.k_nodes / (self.rho_nodes * self.cp_nodes)
        alpha_max = float(np.max(alpha_nodes))

        if dt is not None:
            self.dt = float(dt)
        else:
            self.dt = compute_dt(self.hx, alpha_max, self.lam_target, self.flux_f)

        self.t = np.arange(0.0, t_final + 1e-12, self.dt)

        # -------- CN COEFFICIENTS --------
        self.r_w, self.r_e, self.r_s, self.r_n = (
            self._build_local_cn_coefficients()
        )

        # -------- SPARSE MATRIX (build + factor once) --------
        self.A_csr = self.build_A_sparse()
        # compute an LU factorization once from the compressed sparse row matrix
        self.A_factor = splu(self.A_csr.tocsc())

    # ================================================================
    #                    GEOMETRY / VALIDATION
    # ================================================================
    # the geometry / validation is x-direction only because interfaces are vertical lines.

    def _validate_layers_cover_domain(self) -> None:
        if len(self.layers) == 0:
            raise ValueError("At least one layer must be provided.")

        if not np.isclose(
            self.layers[0].x_left, self.a, atol=self.tol, rtol=0.0
        ):
            raise ValueError(
                f"First layer must start at a={self.a}, got {self.layers[0].x_left}"
            )

        if not np.isclose(
            self.layers[-1].x_right, self.b, atol=self.tol, rtol=0.0
        ):
            raise ValueError(
                f"Last layer must end at b={self.b}, got {self.layers[-1].x_right}"
            )

        for j, layer in enumerate(self.layers):
            if layer.x_right <= layer.x_left:
                raise ValueError(f"Layer {j} has non-positive width.")
            if layer.rho <= 0.0 or layer.cp <= 0.0 or layer.k <= 0.0:
                raise ValueError(
                    f"Layer {j} must have rho > 0, cp > 0, and k > 0."
                )

        for j in range(len(self.layers) - 1):
            if not np.isclose(
                self.layers[j].x_right,
                self.layers[j + 1].x_left,
                atol=self.tol,
                rtol=0.0,
            ):
                raise ValueError(
                    f"Layers {j} and {j+1} are not contiguous: "
                    f"{self.layers[j].x_right} != {self.layers[j+1].x_left}"
                )

    def _extract_internal_interfaces(self) -> list[float]:
        return [self.layers[j].x_right for j in range(len(self.layers) - 1)]

    def _validate_and_index_interfaces(self) -> dict[int, tuple[int, int]]:
        """
        Validate interfaces and assign each to a face slot.

        Interfaces may lie anywhere strictly between two adjacent nodes
        (face slot face_idx lies between nodes face_idx and face_idx+1).
        Off-midpoint interfaces are supported via per-interface offsets
        (h_L, h_R) stored in self.interface_offsets, where h_L + h_R = hx.

        Restrictions:
        (1) interfaces on nodes are invalid (degenerate),
        (2) interfaces in the first or last face slot are invalid — those
            slots touch the Neumann/Dirichlet half-cell BCs.
        """
        interface_face_map: dict[int, tuple[int, int]] = {}
        interface_offsets: dict[int, tuple[float, float]] = {}

        # first validate the interface location
        for j, x_int in enumerate(self.interface_positions):
            # Reject on-node interfaces (degenerate).
            node_coord = (x_int - self.a) / self.hx
            if np.isclose(
                node_coord, round(node_coord), atol=self.tol, rtol=0.0
            ):
                raise ValueError(
                    f"Internal interface x={x_int} lies on a grid node. "
                    f"That is not allowed."
                )

            # Locate the face slot: interface sits between nodes
            # face_idx and face_idx+1.
            face_idx = int(np.floor((x_int - self.a) / self.hx))
            if face_idx < 1 or face_idx > self.Nx - 3:
                raise ValueError(
                    f"Interface x={x_int} falls in a boundary-adjacent face "
                    f"slot (face_idx={face_idx}); must satisfy "
                    f"1 <= face_idx <= {self.Nx - 3} so the Neumann/Dirichlet "
                    f"half-cell BCs stay intact."
                )

            x_left_node = self.a + face_idx * self.hx
            x_right_node = self.a + (face_idx + 1) * self.hx
            h_L = x_int - x_left_node
            h_R = x_right_node - x_int
            if h_L <= self.tol or h_R <= self.tol:
                raise ValueError(
                    f"Interface x={x_int} is too close to a node "
                    f"(h_L={h_L}, h_R={h_R})."
                )

            # after validation, create face map and dict of interface offsets
            interface_face_map[face_idx] = (j, j + 1)
            interface_offsets[face_idx] = (float(h_L), float(h_R))

        self.interface_offsets = interface_offsets
        return interface_face_map

    def _layer_index_for_position(self, x: float) -> int:
        for j, layer in enumerate(self.layers):
            is_last = j == len(self.layers) - 1
            if is_last:
                if (x >= layer.x_left - self.tol) and (
                    x <= layer.x_right + self.tol
                ):
                    return j
            else:
                if (x >= layer.x_left - self.tol) and (
                    x < layer.x_right - self.tol
                ):
                    return j
        raise ValueError(f"Could not assign x={x} to any layer.")

    def _build_node_layer_indices_1d(self) -> np.ndarray:
        """
        Builds array of numbers per-grid node that reflects which layer each point belongs to
        """
        idx = np.zeros(self.Nx, dtype=int)
        for i, x in enumerate(self.grid_x):
            idx[i] = self._layer_index_for_position(x)
        return idx

    def _layer_index_for_face_interior(self, x_face: float) -> int:
        for j, layer in enumerate(self.layers):
            if (x_face > layer.x_left + self.tol) and (
                x_face < layer.x_right - self.tol
            ):
                return j
        raise ValueError(
            f"Non-interface face at x={x_face} could not be assigned "
            f"to a single layer."
        )

    # ================================================================
    #                    FACE CONDUCTANCE
    # ================================================================

    def _build_face_conductance_x(self) -> np.ndarray:
        """
        x-direction face conductance G_x[i, j] between nodes (i,j) and (i+1,j).
        Shape: (Nx-1, Ny).

        Independent of j (vertical interfaces only) — computed as 1D and broadcast.
        """
        G_x_1d = np.zeros(self.Nx - 1, dtype=float)

        for i, x_face in enumerate(self.face_positions_x):
            if i in self.interface_face_map:
                left_idx, right_idx = self.interface_face_map[i]
                h_L, h_R = self.interface_offsets[i]
                kL = self.layers[left_idx].k
                kR = self.layers[right_idx].k
                Rc = self.interface_R[left_idx]
                G_x_1d[i] = 1.0 / (h_L / kL + Rc + h_R / kR)
            else:
                layer_idx = self._layer_index_for_face_interior(x_face)
                G_x_1d[i] = self.layers[layer_idx].k / self.hx

        return np.tile(G_x_1d[:, np.newaxis], (1, self.Ny))

    def _build_face_conductance_y(self) -> np.ndarray:
        """
        y-direction face conductance G_y[i, j] between nodes (i,j) and (i,j+1).
        Shape: (Nx, Ny-1).

        No interfaces in y-direction, so G_y = k/h everywhere.
        """
        G_y = np.zeros((self.Nx, self.Ny - 1), dtype=float)
        for i in range(self.Nx):
            G_y[i, :] = self.k_nodes[i, 0] / self.hx
        return G_y

    # ================================================================
    #                    CN COEFFICIENTS
    # ================================================================

    def _build_local_cn_coefficients(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Build CN coupling coefficients for all four face directions.

        For x-direction faces the face length (dy) cancels with V:
            r = dt * G / (2 * rho*cp * dx[i])

        For y-direction faces the face length (dx) cancels with V:
            r = dt * G / (2 * rho*cp * dy[j])

        Coefficients are zero where no face exists (boundary edges).
        """
        Nx, Ny = self.Nx, self.Ny
        r_w = np.zeros((Nx, Ny), dtype=float)
        r_e = np.zeros((Nx, Ny), dtype=float)
        r_s = np.zeros((Nx, Ny), dtype=float)
        r_n = np.zeros((Nx, Ny), dtype=float)

        rho_cp = self.rho_nodes * self.cp_nodes  # (Nx, Ny)

        for i in range(Nx):
            for j in range(Ny):
                # Dirichlet nodes: no coefficients needed
                if i == Nx - 1:
                    continue

                rc = rho_cp[i, j]
                dxi = self.dx[i]
                dyj = self.dy[j]

                # West face (exists if i > 0)
                if i > 0:
                    r_w[i, j] = self.dt * self.G_x[i - 1, j] / (2.0 * rc * dxi)

                # East face (exists if i < Nx-1)
                if i < Nx - 1:
                    r_e[i, j] = self.dt * self.G_x[i, j] / (2.0 * rc * dxi)

                # South face (exists if j > 0)
                if j > 0:
                    r_s[i, j] = self.dt * self.G_y[i, j - 1] / (2.0 * rc * dyj)

                # North face (exists if j < Ny-1)
                if j < Ny - 1:
                    r_n[i, j] = self.dt * self.G_y[i, j] / (2.0 * rc * dyj)

        return r_w, r_e, r_s, r_n

    # ================================================================
    #                    SPARSE MATRIX ASSEMBLY
    # ================================================================

    def _ij_to_k(self, i: int, j: int) -> int:
        """Convert (i, j) grid index to global linear index k = i + j*Nx."""
        return i + j * self.Nx

    def build_A_sparse(self) -> sp.csr_matrix:
        """
        Build the implicit-side CN matrix A as a sparse CSR matrix.

        Size: (Nx*Ny, Nx*Ny).
        Stencil: 5-point (center + E/W/N/S neighbors).
        Dirichlet rows (i=Nx-1): identity row.
        """
        Nx, Ny = self.Nx, self.Ny
        n = Nx * Ny
        A = sp.lil_matrix((n, n), dtype=float)

        for j in range(Ny):
            for i in range(Nx):
                k = self._ij_to_k(i, j)

                # Dirichlet nodes: right edge + right corners
                if i == Nx - 1:
                    A[k, k] = 1.0
                    continue

                # fetch the local face-based CN coefficient values
                rw = self.r_w[i, j]
                re = self.r_e[i, j]
                rs = self.r_s[i, j]
                rn = self.r_n[i, j]

                # Main diagonal stores the nodes coefficient
                A[k, k] = 1.0 + rw + re + rs + rn

                # West neighbor
                if i > 0:
                    A[k, self._ij_to_k(i - 1, j)] = -rw

                # East neighbor
                if i < Nx - 1:
                    A[k, self._ij_to_k(i + 1, j)] = -re

                # South neighbor
                if j > 0:
                    A[k, self._ij_to_k(i, j - 1)] = -rs

                # North neighbor
                if j < Ny - 1:
                    A[k, self._ij_to_k(i, j + 1)] = -rn

        return A.tocsr()

    # ================================================================
    #                    CN TIME STEP (loop-based reference)
    # ================================================================

    def cn_step_loop(self, Tn: np.ndarray, tn: float) -> np.ndarray:
        """
        One Crank-Nicolson step (loop-based reference implementation).

        Tn has shape (Nx, Ny).  Returns T^{n+1} of shape (Nx, Ny).
        """
        Nx, Ny = self.Nx, self.Ny
        h = self.hx
        dt = self.dt
        rhs = np.zeros((Nx, Ny), dtype=float)

        for j in range(Ny):
            for i in range(Nx):
                # --- Dirichlet nodes (right edge) ---
                if i == Nx - 1:
                    rhs[i, j] = self.T_right(tn + dt)
                    continue

                # fetch local face-based CN coefficients for the current nodes (i,j)
                rw = self.r_w[i, j]
                re = self.r_e[i, j]
                rs = self.r_s[i, j]
                rn = self.r_n[i, j]

                # Gather neighbor temperatures (0 where no neighbor exists,
                # but the corresponding r coefficient is also 0).
                T_W = Tn[i - 1, j] if i > 0 else 0.0
                T_E = Tn[i + 1, j] if i < Nx - 1 else 0.0
                T_S = Tn[i, j - 1] if j > 0 else 0.0
                T_N = Tn[i, j + 1] if j < Ny - 1 else 0.0
                T_C = Tn[i, j]

                # west, center, east, south, and north contributions
                rhs[i, j] = (
                    rw * T_W
                    + (1.0 - rw - re - rs - rn) * T_C
                    + re * T_E
                    + rs * T_S
                    + rn * T_N
                )

                # --- Left Neumann BC flux contribution (i=0) ---
                if i == 0:
                    qn = self.q_left(tn)
                    qnp1 = self.q_left(tn + dt)
                    if self._q_left_is_vector:
                        qn_j = qn[j]
                        qnp1_j = qnp1[j]
                    else:
                        qn_j = qn
                        qnp1_j = qnp1
                    rho_cp = self.rho_nodes[0, j] * self.cp_nodes[0, j]
                    rhs[i, j] += dt * (qn_j + qnp1_j) / (rho_cp * h)

        # --- Source term (CN time-averaged, V cancels for all cell types) ---
        if self.source is not None:
            s_n = self.source(self.X[:Nx - 1, :], self.Y[:Nx - 1, :], tn)
            s_np1 = self.source(
                self.X[:Nx - 1, :], self.Y[:Nx - 1, :], tn + dt
            )
            rho_cp_active = (
                self.rho_nodes[:Nx - 1, :] * self.cp_nodes[:Nx - 1, :]
            )
            rhs[:Nx - 1, :] += dt * 0.5 * (s_n + s_np1) / rho_cp_active

        # --- Solve ---
        # flatten array to column-major indexing, that is, rows change fastest and columns change slowest
        rhs_flat = rhs.ravel(order="F")
        Tnp1_flat = self.A_factor.solve(rhs_flat)
        # reshape back into 2d array
        return Tnp1_flat.reshape((Nx, Ny), order="F")

    # ================================================================
    #                    CN TIME STEP (vectorized)
    # ================================================================

    def cn_step(self, Tn: np.ndarray, tn: float) -> np.ndarray:
        """
        One Crank-Nicolson step (vectorized).

        Tn has shape (Nx, Ny).  Returns T^{n+1} of shape (Nx, Ny).
        """
        Nx, Ny = self.Nx, self.Ny
        h = self.hx
        dt = self.dt
        rhs = np.zeros((Nx, Ny), dtype=float)

        # Active nodes: i = 0..Nx-2 (all except Dirichlet column)
        # Slices for active region
        # this just represents the slice 0:Nx-1
        # for example: rhs[0:Nx-1, :] = rhs[act, :]
        act = slice(0, Nx - 1)  # i = 0..Nx-2

        # --- Center contribution: (1 - rw - re - rs - rn) * T_C ---
        rhs[act, :] = (
            (1.0 - self.r_w[act, :] - self.r_e[act, :]
             - self.r_s[act, :] - self.r_n[act, :])
            * Tn[act, :]
        )

        # --- West: r_w * T_W  (exists for i >= 1) ---
        rhs[1:Nx - 1, :] += self.r_w[1:Nx - 1, :] * Tn[0:Nx - 2, :]

        # --- East: r_e * T_E  (all active nodes i=0..Nx-2 have east neighbor) ---
        # East neighbor of i is i+1; for i=Nx-2, this is the Dirichlet column.
        rhs[act, :] += self.r_e[act, :] * Tn[1:Nx, :]

        # --- South: r_s * T_S  (exists for j >= 1) ---
        rhs[act, 1:] += self.r_s[act, 1:] * Tn[act, 0:Ny - 1]

        # --- North: r_n * T_N  (exists for j <= Ny-2) ---
        rhs[act, 0:Ny - 1] += self.r_n[act, 0:Ny - 1] * Tn[act, 1:Ny]

        # --- Left Neumann BC flux (i=0 row) ---
        qn = self.q_left(tn)
        qnp1 = self.q_left(tn + dt)
        rho_cp_left = self.rho_nodes[0, :] * self.cp_nodes[0, :]
        rhs[0, :] += dt * (qn + qnp1) / (rho_cp_left * h)

        # --- Dirichlet column (i=Nx-1) ---
        rhs[Nx - 1, :] = self.T_right(tn + dt)

        # --- Source term (CN time-averaged) ---
        if self.source is not None:
            s_n = self.source(self.X[act, :], self.Y[act, :], tn)
            s_np1 = self.source(self.X[act, :], self.Y[act, :], tn + dt)
            rho_cp_active = (
                self.rho_nodes[act, :] * self.cp_nodes[act, :]
            )
            rhs[act, :] += dt * 0.5 * (s_n + s_np1) / rho_cp_active

        # --- Solve ---
        rhs_flat = rhs.ravel(order="F")
        Tnp1_flat = self.A_factor.solve(rhs_flat)
        return Tnp1_flat.reshape((Nx, Ny), order="F")

    # ================================================================
    #                    SOLVE (time loop)
    # ================================================================

    def solve(
        self,
        T0: np.ndarray | None = None,
        store_trajectory: bool = False,
    ) -> tuple:
        """
        Run the full time integration.

        Returns
        -------
        If store_trajectory:
            (t, grid_x, grid_y, T_hist) with T_hist shape (Nt, Nx, Ny)
        else:
            (t, grid_x, grid_y, T_final) with T_final shape (Nx, Ny)
        """
        Nt = len(self.t)
        Nx, Ny = self.Nx, self.Ny

        if T0 is None:
            T = ic(self.X, self.Y, self.a, self.b, self.c, self.d).astype(float)
            T += self.T_right(0.0)
        else:
            T = np.asarray(T0, dtype=float)
            if T.shape != (Nx, Ny):
                raise ValueError(
                    f"T0 shape must be ({Nx}, {Ny}), got {T.shape}"
                )

        if store_trajectory:
            T_hist = np.zeros((Nt, Nx, Ny), dtype=float)
            T_hist[0] = T

        for n in range(Nt - 1):
            T = self.cn_step(T, self.t[n])
            if store_trajectory:
                T_hist[n + 1] = T

        if store_trajectory:
            return self.t, self.grid_x, self.grid_y, T_hist
        else:
            return self.t, self.grid_x, self.grid_y, T


if __name__ == '__main__':

    layers = [
        Layer2D(x_left=0.0, x_right=0.5, rho=1, cp=1, k=2),
        Layer2D(x_left=0.5, x_right=1.0, rho=1, cp=1, k=1)
    ]

    contact_r = [0.5]

    sim = FVSolver2D(
        a=0.0,
        b=1.0,
        c=0.0,
        d=1.0,
        Nx=100,
        Ny=100,
        layers=layers,
        interface_R=contact_r,
        lam_target=0.5,
        dt=0.005,
        flux_f=2.0,
        flux_A=50.0,
        t_on=0.0,
        t_off=0.2,
        t_final=0.5,
        phase=0.0
    )

    t, x, y, T_final = sim.solve(store_trajectory=False)

    print(f"t shape: {t.shape}")
    print(f"x shape: {x.shape}")
    print(f"y shape: {y.shape}")
    print(f"T_final shape: {T_final.shape}")
    print(f"interface positions: {sim.interface_positions}")
    print(f"interface face map: {sim.interface_face_map}")

