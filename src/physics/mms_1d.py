from src.physics.fv_solver_1d import FVSolver1D, Layer1D
import numpy as np

def run_mms_once(N: int, dt=None) -> tuple[float, float, float, float]:
    a, b = 0.0, 1.0
    L = b-a
    rho, cp, k = 1.0, 1.0, 1.0

    flux_f = 2.0
    omega = 2.0 * np.pi * flux_f
    phase = 0.0
    flux_A = 50.0

    # choose A such that flux amplitude matches flux_A:
    # q_star = 4*k*A*L^3*sin(...), so |q_star|_max = flux_A => A = flux_A/(4*k*L^3)
    A = flux_A / (4.0 * k * L**3)

    # define T*(x,t)
    def T_star(x: np.ndarray, t: float) -> np.ndarray:
        return 300.0 + A * np.sin(omega * t + phase) * (b - x)**4

    # left flux: q = -k * dT*/dx |_{x=a} = 4*k*A*L^3*sin(omega*t + phase)
    def q_star(t: float) -> float:
        return 4.0 * k * A * L**3 * np.sin(omega * t + phase)

    # source: s = rho*cp*dT*/dt - k*d^2T*/dx^2
    def s_star(x: np.ndarray, t: float) -> np.ndarray:
        return (rho * cp * A * omega * np.cos(omega * t + phase) * (b - x)**4
                - k * 12.0 * A * np.sin(omega * t + phase) * (b - x)**2)

    # create instance of fv solver class to solve forcing equation
    layer = Layer1D(x_left=a, x_right=b, rho=rho, cp=cp, k=k)
    sim = FVSolver1D(
        a=a,
        b=b,
        N=N,
        lam_target=0.5,
        layers=[layer],
        t_final=1.0,
        flux_f=flux_f,
        flux_A=flux_A,
        dt=dt,
        t_on=0.0,
        t_off=0.2,
        phase=0.0,
        source=s_star,
        q_left_fn=q_star,
    )

    T0 = T_star(sim.grid, sim.t[0])
    t, x, T_final_num = sim.solve(T0=T0, store_trajectory=False)

    T_final_exact = T_star(sim.grid, sim.t[-1])

    error  = T_final_num-T_final_exact
    max_abs_err = np.max(np.abs(error))
    l2_err = np.sqrt(np.mean(error**2))

    return sim.h, sim.dt, max_abs_err, l2_err

# fix dt — must be small enough that temporal error << spatial error at all grid levels
def space_order_test(N_list: list) -> float:
    results = []
    hs = []
    for N in N_list:
        h, _, _, l2_error = run_mms_once(N, dt=0.0005)

        results.append(l2_error)
        hs.append(h)

    # compute pairwise orders across all adjacent levels and return the mean
    orders = []
    for i in range(len(results) - 1):
        p = np.log(results[i] / results[i + 1]) / np.log(hs[i] / hs[i + 1])
        orders.append(p)
    return float(np.mean(orders))

# fix dx — must be fine enough that spatial error << temporal error at all dt levels
def time_order_test(dt_list: list) -> float:
    """Keep N high so that spatial error is negligible."""
    t_results = []
    for t in dt_list:
        _, _, _, l2_error = run_mms_once(N=801, dt=t)

        t_results.append(l2_error)

    # compute pairwise orders across all adjacent levels and return the mean
    orders = []
    for i in range(len(t_results) - 1):
        p = np.log(t_results[i] / t_results[i + 1]) / np.log(dt_list[i] / dt_list[i + 1])
        orders.append(p)
    return float(np.mean(orders))


# ============================================================
# MMS with interface thermal resistance (piecewise solution)
# ============================================================
#
# Two-layer domain [0, 1] with interface at x_I = 0.5.
# Layer 1 (left):  k1=2, rho1=1.5, cp1=1
# Layer 2 (right): k2=1, rho2=0.8, cp2=1
# Interface resistance: Rc = 0.1
#
# Manufactured solution:
#   T_L*(x, t) = 300 + A_L sin(wt) [(x_I - x)^4 + D_L (x_I - x)] + C_L sin(wt)
#   T_R*(x, t) = 300 + A_R sin(wt) (1 - x)^4
#
# Constraints (flux continuity + temperature jump ΔT = Rc·q_I):
#   A_R = 2 k1 A_L D_L / k2
#   C_L = k1 A_L D_L (1/(8 k2) + Rc)
#
# With A_L=10, D_L=1:  A_R=40, C_L=4.5
# q_I(t) = 20 sin(wt),  jump = 2 sin(wt) = 0.1 × 20 sin(wt)  ✓

def run_mms_interface(N: int, dt=None) -> tuple[float, float, float, float]:
    """MMS verification for two-layer solver with interface thermal resistance.

    Returns (h, dt, max_abs_error, l2_error) — same signature as run_mms_once.
    N must be even so that the interface at x=0.5 lands on a cell face.
    """
    a, b, x_I = 0.0, 1.0, 0.5
    k1, k2 = 2.0, 1.0
    rho1, cp1 = 1.5, 1.0
    rho2, cp2 = 0.8, 1.0
    Rc = 0.1

    flux_f = 2.0
    omega = 2.0 * np.pi * flux_f
    A_L, D_L = 10.0, 1.0
    A_R = 2.0 * k1 * A_L * D_L / k2            # = 40
    C_L = k1 * A_L * D_L * (1.0 / (8 * k2) + Rc)  # = 4.5

    # --- manufactured solution (piecewise) ---
    def T_star(x: np.ndarray, t: float) -> np.ndarray:
        T = np.full_like(x, 300.0)
        g = np.sin(omega * t)
        left = x < x_I
        T[left] += A_L * g * ((x_I - x[left])**4 + D_L * (x_I - x[left])) + C_L * g
        T[~left] += A_R * g * (b - x[~left])**4
        return T

    # --- left flux: q = -k1 dT_L/dx|_{x=0} ---
    def q_left(t: float) -> float:
        return k1 * A_L * np.sin(omega * t) * (4.0 * (x_I - a)**3 + D_L)

    # --- piecewise source: s = rho cp dT*/dt - k d^2T*/dx^2 ---
    def source(x: np.ndarray, t: float) -> np.ndarray:
        s = np.empty_like(x)
        g = np.sin(omega * t)
        gp = omega * np.cos(omega * t)
        left = x < x_I
        xl = x[left]
        xr = x[~left]
        s[left] = (rho1 * cp1 * gp * (A_L * ((x_I - xl)**4 + D_L * (x_I - xl)) + C_L)
                   - k1 * A_L * g * 12.0 * (x_I - xl)**2)
        s[~left] = (rho2 * cp2 * A_R * gp * (b - xr)**4
                    - k2 * A_R * g * 12.0 * (b - xr)**2)
        return s

    # --- solver setup ---
    layers = [
        Layer1D(x_left=a, x_right=x_I, rho=rho1, cp=cp1, k=k1),
        Layer1D(x_left=x_I, x_right=b, rho=rho2, cp=cp2, k=k2),
    ]
    sim = FVSolver1D(
        a=a, b=b, N=N,
        lam_target=0.5,
        layers=layers,
        interface_R=[Rc],
        t_final=1.0,
        flux_f=flux_f, flux_A=0.0,
        dt=dt,
        t_on=0.0, t_off=0.2, phase=0.0,
        source=source,
        q_left_fn=q_left,
    )

    T0 = T_star(sim.grid, sim.t[0])
    _, _, T_final_num = sim.solve(T0=T0, store_trajectory=False)
    T_final_exact = T_star(sim.grid, sim.t[-1])

    error = T_final_num - T_final_exact
    max_abs_err = float(np.max(np.abs(error)))
    l2_err = float(np.sqrt(np.mean(error**2)))
    return sim.h, sim.dt, max_abs_err, l2_err


def space_order_test_interface(N_list: list) -> float:
    """Spatial convergence order for the interface-resistance MMS.

    All N values must be even so that x=0.5 lands on a face.
    Fixes dt=0.0005 so temporal error is negligible.
    """
    results, hs = [], []
    for N in N_list:
        h, _, _, l2 = run_mms_interface(N, dt=0.0005)
        results.append(l2)
        hs.append(h)
    orders = []
    for i in range(len(results) - 1):
        p = np.log(results[i] / results[i + 1]) / np.log(hs[i] / hs[i + 1])
        orders.append(p)
    return float(np.mean(orders))


def time_order_test_interface(dt_list: list) -> float:
    """Temporal convergence order for the interface-resistance MMS.

    Fixes N=800 (even) so spatial error is negligible.
    """
    results = []
    for dt_val in dt_list:
        _, _, _, l2 = run_mms_interface(N=800, dt=dt_val)
        results.append(l2)
    orders = []
    for i in range(len(results) - 1):
        p = np.log(results[i] / results[i + 1]) / np.log(dt_list[i] / dt_list[i + 1])
        orders.append(p)
    return float(np.mean(orders))


if __name__ == '__main__':
    # --- Single-layer MMS ---
    print("=== Single-Layer MMS ===")
    h, dt, max_error, l2_error = run_mms_once(N=101)
    print(f"N=101: h={h:.5f}, dt={dt:.6f}, max_err={max_error:.6e}, l2_err={l2_error:.6e}")
    N_list = [21, 41, 81, 161]
    order_x = space_order_test(N_list=N_list)
    print(f"Spatial order (mean): {order_x:.3f}")

    dt_list = [0.02, 0.01, 0.005]
    order_t = time_order_test(dt_list=dt_list)
    print(f"Temporal order (mean): {order_t:.3f}")

    # --- Interface resistance MMS ---
    print("\n=== Interface Resistance MMS ===")
    h, dt, max_error, l2_error = run_mms_interface(N=100)
    print(f"N=100: h={h:.5f}, dt={dt:.6f}, max_err={max_error:.6e}, l2_err={l2_error:.6e}")

    p_x = space_order_test_interface([50, 100, 200])
    print(f"Spatial order (mean): {p_x:.3f}")

    p_t = time_order_test_interface([0.02, 0.01, 0.005])
    print(f"Temporal order (mean): {p_t:.3f}")

