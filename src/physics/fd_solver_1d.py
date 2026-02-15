import numpy as np
from scipy.linalg import solve_banded
from numpy.typing import ArrayLike



# initial condition
def ic(x : np.ndarray,  a: float, b: float) -> np.ndarray:
    L = b - a
    return np.cos((np.pi*x)/(2*L)) + 0.1*np.sin((np.pi*x)/L)

# create sinusoid function with a window
def windowed_sin_flux(f : float, A : float, t_on : float, t_off: float, phase = 0.0):
    def q(t):
        if (t_on <= t) and (t <= t_off):
            return A * np.sin(2 * np.pi * f * t + phase)
        return 0.0
    return q

def compute_dt(h : float, alpha_val : float, lam : float, flux_frequency : float) -> float:
   rule_a = (lam * h**2) / alpha_val
   rule_b = 1/(20 * flux_frequency)
   return min(rule_a, rule_b)

# create class FDSolver1D so it contains all functions above and implementation is straight forward
class FDSolver1D:
    def __init__(
            self,
            a: float,
            b: float,
            N: int,
            rho: float,
            cp: float,
            k: float,
            lam_target: float,
            t_final: float,
            flux_f: float,
            flux_A: float,
            t_on: float,
            t_off: float,
            phase: float
    ):

        self.a = a
        self.b = b
        self.N = N
        self.k = k
        self.alpha = k/(rho*cp)
        self.flux_f = flux_f
        self.flux_A = flux_A
        self.t_on = t_on
        self.t_off = t_off
        self.phase = phase

        self.grid = np.linspace(self.a, self.b, self.N)
        self.h = self.grid[1] - self.grid[0]

        self.t_final = t_final
        self.dt = compute_dt(self.h, self.alpha, lam_target, flux_f)
        self.lam = self.alpha * self.dt / (self.h**2)

        # time grid
        self.t = np.arange(0.0, t_final + 1e-12, self.dt) # + 1e-12 because np.arnage() does NOT include the stop point

        # boundary flux function q(t)
        self.q_left = windowed_sin_flux(flux_f, flux_A, t_on, t_off, phase)

        # build ab once
        self.ab = self.build_A_banded()

        # right Dirichlet function T_R(t)
        # for now, i'm just going to have T_right be a constant
        self.T_right = 300.0 # in Kelvin

    def build_A_banded(self) -> np.ndarray:
        """build banded matrix ab with shape (l + u + 1, N)"""
        N = self.N

        if N<3:
            raise ValueError("N must be >= 3 for second-order accurate.")

        l, u = 1, 2
        # init ab by creating a numpy array with size (l + u + 1, N) filled with 0 floats
        ab = np.zeros((l + u + 1, N), dtype=float)


        # ------- ROW 0: Left-side Neumann BC ( -------
        ab[2, 0] = 3  # A_0,0
        ab[1, 1] = -4  # A_0,1
        ab[0, 2] = 1  # A_0,2

        # ------- INTERIOR CN NODES (1, ...., N-2) ----------
        ab[2, 1:N-1] = 1+self.lam # main diagonal
        ab[1, 2:N] = -self.lam/2 # +1 diagonal
        ab[3, 0:N-2] = -self.lam/2 # -1 diagonal

        # ----- ROW N-1: Right side Dirichlet BC -------
        ab[2, N-1] = 1.0 # A_N-1,N-1 = 1
        ab[3, N-2] = 0.0 # removing coupling A_N-1, N-2

        return ab

    # alpha is constant for now so ab just needs to be built once
    def cn_step_banded(self, Tn: np.ndarray, tn: float) -> np.ndarray:
        """One CN solving AT^{n+1} = rhs"""
        N = self.N

        rhs = np.zeros(N, dtype=float)

        # row 0: Neumann BC with time-averaged flux
        qn = self.q_left(tn)
        qnp1 = self.q_left(tn + self.dt)
        q_half = (qnp1+qn)/2
        rhs[0] = (2*self.h*q_half)/self.k

        # interior rows cn nodes: left neighbors + centers + right neighbors
        rhs[1:-1] = (self.lam/2)*Tn[:-2] + (1.0 - self.lam)*Tn[1:-1] + (self.lam/2)*Tn[2:]

        # row N-1
        rhs[-1] = self.T_right

        # solve banded system with (l=1, u=2)
        Tnp1 = solve_banded((1, 2), self.ab, rhs)
        return Tnp1

    def solve(self, T0: np.ndarray | None = None, store_history: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        If store_history:
            return (t, grid, T_hist) with T_hist shape = (Nt, N).
        else:
            return (t, grid, T_final) with T_final shape = (N,)
        """
        Nt = len(self.t)
        N = self.N

        if T0 is None:
            T = ic(self.grid, self.a, self.b).astype(float)
            T += self.T_right
        else:
            T = np.asarray(T0, dtype=float)
            if T.shape != (N,):
                raise ValueError(f"The shape of T0 msut be ({N}), but got {T.shape}")

        if store_history:
            T_hist = np.zeros((Nt, N), dtype=float)
            T_hist[0, :] = T # its also equal to T_hist[0] because that returns the whole row (column) of nodes at time step 0

        # time stepping
        for n in range(Nt - 1):
            T = self.cn_step_banded(T, self.t[n])
            if store_history:
                T_hist[n+1, :] = T

        if store_history:
            return self.t, self.grid, T_hist
        else:
            return self.t, self.grid, T


if __name__ == '__main__':
    # s will stand for simulation
    sim = FDSolver1D(
        a=0.0,
        b=1.0,
        N=101,
        cp=1,
        rho=1,
        k=1,
        lam_target=0.5,
        t_final=1.0,
        flux_f=2.0,
        flux_A=50,
        t_on=0.0,
        t_off=1.0,
        phase=0.0
    )

    t, x, T_final = sim.solve(store_history=False)
    print("Ok:", t.shape, x.shape, T_final.shape)







