import numpy as np
from scipy.linalg import solve_banded
from dataclasses import dataclass

# ---------- SOLVER RESTRICTIONS -----------

# Uniform grid
# Interface locations must lie on the midpoint between nodes, that is, middle of cell faces
# Interface locations on nodes are NOT allowed
# arbitrary off-face interfaces are NOT allowed
# Interface thermal resistance supported via interface_R parameter (default: perfect contact)

# initial condition
def ic(x : np.ndarray,  a: float, b: float) -> np.ndarray:
    L = b - a
    return np.cos((np.pi*x)/(2*L)) + 0.1*np.sin((np.pi*x)/L)

# create sinusoid function with a Tukey (tapered cosine) window
def windowed_sin_flux(f : float, A : float, t_on : float, t_off: float, phase = 0.0, tukey_alpha: float = 0.5):
    def q(t):
        if t < t_on or t > t_off:
            return 0.0
        W = t_off - t_on
        tau = (t - t_on) / W  # normalized position in time domain
        # tukey envelope: tukey_alpha=0 -> rectangular, tukey_alpha=1 -> Hann
        if tukey_alpha <= 0.0:
            w = 1.0
        elif tau < tukey_alpha / 2:
            w = 0.5 * (1 - np.cos(2 * np.pi * tau / tukey_alpha))
        elif tau > 1 - tukey_alpha / 2:
            w = 0.5 * (1 + np.cos(2 * np.pi * (tau - 1 + tukey_alpha / 2) / tukey_alpha))
        else:
            w = 1.0
        return w * A * np.sin(2 * np.pi * f * t + phase)
    return q

def compute_dt(h : float, alpha_val : float, lam : float, flux_frequency : float) -> float:
   rule_a = (lam * h**2) / alpha_val
   rule_b = np.inf if flux_frequency <= 0.0 else 1.0 /(20.0 * flux_frequency)
   return min(rule_a, rule_b)

# data for creating a generic layer in 1D
@dataclass(frozen=True)
class Layer1D:
    x_left: float
    x_right: float
    rho: float
    cp: float
    k: float

# create class FVSolver1D so it contains all functions above and implementation is straight forward
class FVSolver1D:
    """
    1D Conservative Multilayer FV Solver

    Baseline Qualities:
    (1) Uniform grid
    (2) Interfaces lie on cell faces
    (3) Interfaces cannot be off-center
    (4) Interfaces cannot lie on nodes
    (5) Local storage coefficients stored in nodes
    """
    def __init__(
            self,
            a: float,
            b: float,
            N: int,
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
            tol: float = 1e-12
    ):
        self.a = a
        self.b = b
        self.N = N
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

        # -------- SPACE & TIME GRID + DT & DX ---------
        self.grid = np.linspace(self.a, self.b, self.N)
        self.h = self.grid[1] - self.grid[0]
        self.face_positions = self.a + (np.arange(self.N - 1) + 0.5) * self.h

        # -------- VALIDATE GEOMETRY / LAYERS ----------
        self._validate_layers_cover_domain()
        self.interface_positions = self._extract_internal_interfaces()
        self.interface_face_map = self._validate_and_index_interfaces()

        # --------- INTERFACE THERMAL RESISTANCE ----------
        num_interfaces = len(self.interface_positions)
        # if there does not exist of list of interface resistance values then zero-out the affect of R_c
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

        # --------- BCs -------------
        # boundary flux function q(t) with optional MMS flux input
        if q_left_fn is not None:
            self.q_left = q_left_fn
        else:
            self.q_left = windowed_sin_flux(flux_f, flux_A, t_on, t_off, phase, tukey_alpha)

        # make T_right_fn always callable
        self.T_right = T_right_fn if T_right_fn is not None else (lambda t: 300.0)

        # source term, default is None
        self.source = source

        # ------- BUILD MATERIAL/STORAGE/COUPLING FIELDS -------
        self.layer_index_at_nodes = self._build_node_layer_indices()
        self.rho_nodes = np.array([self.layers[j].rho for j in self.layer_index_at_nodes], dtype=float)
        self.cp_nodes = np.array([self.layers[j].cp for j in self.layer_index_at_nodes], dtype=float)
        self.k_nodes = np.array([self.layers[j].k for j in self.layer_index_at_nodes], dtype=float)

        # heat storage per unit cross-sectional area
        self.C_node = self.rho_nodes * self.cp_nodes * self.h

        # face conductance per unit cross-sectional area
        # G_{i + 1/2} = k / h (same material)
        # G_{i + 1/2} = 1 / ((h/2kl) + (h/2kr)) if face is an interface
        self.G_face = self._build_face_conductance()

        # ------- TIME STEP ------
        alpha_nodes = self.k_nodes / (self.rho_nodes * self.cp_nodes)
        alpha_max = float(np.max(alpha_nodes))

        # allow for dt to be passed int explicitly
        if dt is not None:
            self.dt = float(dt)
        else:
            self.dt = compute_dt(self.h, alpha_max, self.lam_target, self.flux_f)

        self.t = (np.arange(0.0, t_final + 1e-12, self.dt))  # + 1e-12 because np.arnage() does NOT include the stop point

        # -------- LOCAL CN COEFFICIENTS ------
        self.r_minus, self.r_plus = self._build_local_cn_coefficients()

        # -------- HALF-CELL BC COEFFICIENT ------
        # r0_bc = dt * G_face[0] / (rho_0 * cp_0 * h)
        # This is exactly twice the equivalent interior coefficient because the
        # half-cell (width h/2) has half the thermal capacity of a full cell.
        self.r0_bc = self.dt * self.G_face[0] / (self.rho_nodes[0] * self.cp_nodes[0] * self.h)

        # --------- BANDED MATRIX -------
        self.ab = self.build_A_banded()

    # ---------- GEOMETRY / VALIDATION ---------
    def _validate_layers_cover_domain(self) -> None:
        """
        Determines whether predefined layers cover the whole domain and other checks
        """
        if len(self.layers) == 0:
            raise ValueError("At least one layer must be provided.")

        if not np.isclose(self.layers[0].x_left, self.a, atol=self.tol, rtol=0.0):
            raise ValueError(f"First layer must start at a={self.a}, got {self.layers[0].x_left}")

        if not np.isclose(self.layers[-1].x_right, self.b, atol=self.tol, rtol=0.0):
            raise ValueError(f"Last layer must end at b={self.b}, got {self.layers[-1].x_right}")

        for j, layer in enumerate(self.layers):
            if layer.x_right <= layer.x_left:
                raise ValueError(f"Layer {j} has non-positive width.")

            if layer.rho <= 0.0 or layer.cp <= 0.0 or layer.k <= 0.0:
                raise ValueError(f"Layer {j} must have rho > 0, cp > 0, and k > 0.")

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
        # only the internal interfaces are wanted
        return [self.layers[j].x_right for j in range(len(self.layers) - 1)]

    def _validate_and_index_interfaces(self) -> dict[int, tuple[int, int]]:
        """
        Validate interfaces across three rules and index interfaces

        Rules:
        (1) every internal interface must lie exactly on a grid face
        (2) interfaces on nodes are invalid
        (3) off-face interfaces are invalid
        """
        interface_face_map: dict[int, tuple[int, int]] = {}

        for j, x_int in enumerate(self.interface_positions):
            # node test: x_int = a + i h
            # testing whether node_coord is an integer in grid index form
            node_coord = (x_int - self.a) / self.h
            if np.isclose(node_coord, round(node_coord), atol=self.tol, rtol=0.0):
                raise ValueError(f"Internal interface x={x_int} lies on a grid node. That is not allowed.")

            # face test: x_int = a + (i + 1/2) h
            face_coord = (x_int - self.a) / self.h - 0.5
            face_idx = int(round(face_coord))
            if not np.isclose(face_coord, face_idx, atol=self.tol, rtol=0.0):
                raise ValueError(f"Internal interface x={x_int} is not face-aligned. That is not allowed.")

            if face_idx < 0 or face_idx > self.N - 2:
                raise ValueError(f"Computed face index {face_idx} out of range for interface x={x_int}")

            interface_face_map[face_idx] = (j, j+1)

        return interface_face_map

    def _layer_index_for_position(self, x: float) -> int:
        """
        Return the layer index containing x
        """
        for j, layer in enumerate(self.layers):
            is_last = (j == len(self.layers) - 1)
            if is_last:
                if (x >= layer.x_left - self.tol) and (x <= layer.x_right + self.tol):
                    return j
            else:
                if (x >= layer.x_left - self.tol) and (x < layer.x_right - self.tol):
                    return j

        raise ValueError(f"Could not assign x={x} to any layer.")

    def _build_node_layer_indices(self) -> np.ndarray:
        """
        Build array of layer indices representing which layer each spatial point lies
        """
        idx = np.zeros(self.N, dtype=int)
        for i, x in enumerate(self.grid):
            idx[i] = self._layer_index_for_position(x)
        return idx

    # -------- STORAGE / CONDUCTANCE BUILDERS ---------

    def _layer_index_for_face_interior(self, x_face: float) -> int:
        """
        Used only for non-interface faces since all are already explicitly handled for
        """
        for j, layer in enumerate(self.layers):
            if (x_face > layer.x_left + self.tol) and (x_face < layer.x_right - self.tol):
                return j

        raise ValueError(f"Non-interface face at x={x_face} could not be assigned to a single layer.")

    def _build_face_conductance(self) -> np.ndarray:
        """
        Build face conductance G_face[i] between node i and node i + 1

        At an interface with contact resistance R_c:
            G = 1 / (h/(2*k_L) + R_c + h/(2*k_R))
        When R_c = 0 this reduces to the harmonic mean (perfect contact).
        """
        # allocate array of size N-1 (which is the total number of faces)
        G = np.zeros(self.N - 1, dtype=float)

        for i, x_face in enumerate(self.face_positions):
            if i in self.interface_face_map:
                left_layer_idx, right_layer_idx = self.interface_face_map[i]
                kL = self.layers[left_layer_idx].k
                kR = self.layers[right_layer_idx].k
                Rc = self.interface_R[left_layer_idx]
                G[i] = 1.0 / ((self.h / (2.0 * kL)) + Rc + (self.h / (2.0 * kR)))
            else:
                # not an interface, so get layer index
                layer_idx = self._layer_index_for_face_interior(x_face)
                k = self.layers[layer_idx].k
                G[i] = k / self.h

        return G

    def _build_local_cn_coefficients(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Interior coefficients:
        r_minus[i] = dt * G_{i-1/2} / (2 * C_i)
        r_plus[i] = dt * G_{i + 1/2} / (2 * C_i)

        where G_face = face conductance per unit area
              C_i = rho_i * cp_i * h
        """
        r_minus = np.zeros(self.N, dtype=float)
        r_plus = np.zeros(self.N, dtype=float)

        for i in range(1, self.N - 1):
            r_minus[i] = self.dt * self.G_face[i-1] / (2.0 * self.C_node[i])
            r_plus[i] = self.dt * self.G_face[i] / (2.0 * self.C_node[i])

        return r_minus, r_plus

    def build_A_banded(self) -> np.ndarray:
        """Build tridiagonal banded matrix ab with shape (3, N).

        Uses half-cell energy balance at node 0 (left Neumann BC),
        standard conservative CN for interior nodes, and Dirichlet at node N-1.
        """
        N = self.N

        if N<3:
            raise ValueError("N must be >= 3 for second-order accurate.")

        l, u = 1, 1
        ab = np.zeros((l + u + 1, N), dtype=float)

        # ------- ROW 0: Half-cell energy balance (left Neumann BC) -------
        # (1 + r0) T_0^{n+1}  -  r0 T_1^{n+1}  =  RHS
        r0 = self.r0_bc
        ab[1, 0] = 1.0 + r0  # main diagonal  A[0, 0]
        ab[0, 1] = -r0  # super-diagonal A[0, 1]

        # ------- INTERIOR ROWS: CONSERVATIVE CN ----------
        for i in range(1, N-1):
            rm = self.r_minus[i]
            rp = self.r_plus[i]

            ab[1, i]     = 1.0 + rm + rp  # main diagonal  A[i, i]
            ab[2, i - 1] = -rm             # sub-diagonal   A[i, i-1]
            ab[0, i + 1] = -rp             # super-diagonal A[i, i+1]

        # ----- ROW N-1: Right side Dirichlet BC -------
        ab[1, N-1] = 1.0  # A[N-1, N-1] = 1
        ab[2, N-2] = 0.0  # removing coupling A[N-1, N-2]

        return ab

    def cn_step_banded(self, Tn: np.ndarray, tn: float) -> np.ndarray:
        """One CN step: solve A T^{n+1} = rhs, with optional source term."""
        rhs = np.zeros(self.N, dtype=float)

        # row 0: half-cell energy balance (left Neumann BC, CN time-averaged)
        # RHS = (1 - r0) T_0^n  +  r0 T_1^n  +  dt (q^n + q^{n+1}) / (rho_0 cp_0 h)
        r0 = self.r0_bc
        qn = self.q_left(tn)
        qnp1 = self.q_left(tn + self.dt)
        C_denom = self.rho_nodes[0] * self.cp_nodes[0] * self.h  # = 2 * C_half
        rhs[0] = (1.0 - r0) * Tn[0] + r0 * Tn[1] + self.dt * (qn + qnp1) / C_denom

        # interior rows: conservative CN
        for i in range(1, self.N - 1):
            rm = self.r_minus[i]
            rp = self.r_plus[i]
            rhs[i] = rm * Tn[i - 1] + (1.0 - rm - rp) * Tn[i] + rp * Tn[i + 1]

        # source term s(x, t): CN time-averaging on nodes 0 through N-2
        if self.source is not None:
            # evaluate source at all nodes except the Dirichlet node (N-1)
            active_x = self.grid[:-1]
            s_n = np.asarray(self.source(active_x, tn), dtype=float)
            s_np1 = np.asarray(self.source(active_x, tn + self.dt), dtype=float)

            if s_n.shape != (self.N - 1,) or s_np1.shape != (self.N - 1,):
                raise ValueError(
                    f"source(x, t) must return array of shape ({self.N - 1},) "
                    f"for active nodes, got {s_n.shape}."
                )

            # node 0 (half-cell): dt * 0.5 * (s_n + s_np1) / (rho_0 * cp_0)
            # the h/2 cell width cancels with C_half = rho*cp*h/2
            rhs[0] += self.dt * 0.5 * (s_n[0] + s_np1[0]) / (self.rho_nodes[0] * self.cp_nodes[0])

            # interior nodes 1..N-2 (full cells): dt * h * 0.5 * (s_n + s_np1) / C_node
            rhs[1:-1] += self.dt * self.h * 0.5 * (s_n[1:] + s_np1[1:]) / self.C_node[1:-1]

        # row N-1: Dirichlet BC
        rhs[-1] = self.T_right(tn + self.dt)

        # solve tridiagonal system (l=1, u=1)
        Tnp1 = solve_banded((1, 1), self.ab, rhs)
        return Tnp1

    def solve(self, T0: np.ndarray | None = None, store_trajectory: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        If store_trajectory:
            return (t, grid, T_hist) with T_hist shape = (Nt, N).
        else:
            return (t, grid, T_final) with T_final shape = (N,)
        """
        Nt = len(self.t)
        N = self.N

        T_hist = None

        if T0 is None:
            T = ic(self.grid, self.a, self.b).astype(float)
            T += self.T_right(0.0)
        else:
            T = np.asarray(T0, dtype=float)
            if T.shape != (N,):
                raise ValueError(f"The shape of T0 msut be ({N}), but got {T.shape}")

        if store_trajectory:
            T_hist = np.zeros((Nt, N), dtype=float)
            T_hist[0, :] = T # its also equal to T_hist[0] because that returns the whole row (column) of nodes at time step 0

        # time stepping
        for n in range(Nt - 1):
            T = self.cn_step_banded(T, self.t[n])
            if store_trajectory:
                T_hist[n+1, :] = T

        if store_trajectory:
            return self.t, self.grid, T_hist
        else:
            return self.t, self.grid, T


if __name__ == '__main__':
    # IMPORTANT:
    # for domain [0, 1] and interface x=0.5, using a uniform node grid:
    # - N must be even for x=0.5 to lie on a face
    # - N = 100 makes x = 0.5 a face

    # define layers
    layers = [
        Layer1D(x_left=0.0, x_right=0.5, rho=1.0, cp=1.0, k=1.0),
        Layer1D(x_left=0.5, x_right=1.0, rho=2.0, cp=1.5, k=1.5)
    ]

    # define list of contact resistance
    contact_resistance = [0.5]

    # init simulation
    sim = FVSolver1D(
        a=0.0,
        b=1.0,
        N=100,
        layers=layers,
        interface_R=contact_resistance,
        lam_target=0.5,
        t_final=1.0,
        flux_f=2.0,
        flux_A=50.0,
        t_on=0.0,
        t_off=0.2,
        phase=0.0,
        dt=0.005
    )

    t, x, T_hist = sim.solve(store_trajectory=True)
    print(f"t shape: {t.shape}")
    print(f"x shape: {x.shape}")
    print(f"T_history shape: {T_hist.shape}")
    print(f"interface positions: {sim.interface_positions}")
    print(f"interface face map: {sim.interface_face_map}")
