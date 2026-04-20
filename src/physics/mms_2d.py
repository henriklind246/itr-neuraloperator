"""
Method of Manufactured Solutions (MMS) for the 2D FV Solver.

Three test cases:
1. y-independent MMS — same as 1D, validates 2D infrastructure
2. Full 2D MMS — y-dependent correction validates y-Laplacian terms
3. Interface MMS — piecewise 2D with R_c at x=0.5
"""

import numpy as np
from src.physics.fv_solver_2d import FVSolver2D, Layer2D


# ==============================================================
# 1. y-independent MMS (identical physics to mms_1d.run_mms_once)
# ==============================================================

def run_mms_y_independent(N: int, dt=None) -> tuple[float, float, float, float]:
    """
    y-independent manufactured solution on 2D grid.

    T*(x,y,t) = 300 + A sin(wt) (b-x)^4   (no y dependence)

    Returns (h, dt, max_abs_error, l2_error).
    """
    a, b, c, d = 0.0, 1.0, 0.0, 1.0
    rho, cp, k = 1.0, 1.0, 1.0

    flux_f = 2.0
    omega = 2.0 * np.pi * flux_f
    phase = 0.0
    flux_A = 50.0

    A = flux_A / (4.0 * k * (b - a) ** 3)

    def T_star(X, Y, t):
        return 300.0 + A * np.sin(omega * t + phase) * (b - X) ** 4

    def q_star(t):
        return 4.0 * k * A * (b - a) ** 3 * np.sin(omega * t + phase)

    def s_star(X, Y, t):
        return (rho * cp * A * omega * np.cos(omega * t + phase) * (b - X) ** 4
                - k * 12.0 * A * np.sin(omega * t + phase) * (b - X) ** 2)

    layer = Layer2D(x_left=a, x_right=b, rho=rho, cp=cp, k=k)
    sim = FVSolver2D(
        a=a, b=b, c=c, d=d,
        Nx=N, Ny=N,
        lam_target=0.5,
        layers=[layer],
        t_final=0.37,
        flux_f=flux_f, flux_A=flux_A,
        dt=dt,
        t_on=0.0, t_off=0.2, phase=0.0,
        source=s_star,
        q_left_fn=q_star,
    )

    T0 = T_star(sim.X, sim.Y, sim.t[0])
    _, _, _, T_final_num = sim.solve(T0=T0, store_trajectory=False)
    T_final_exact = T_star(sim.X, sim.Y, sim.t[-1])

    error = T_final_num - T_final_exact
    max_abs_err = float(np.max(np.abs(error)))
    l2_err = float(np.sqrt(np.mean(error ** 2)))
    return sim.hx, sim.dt, max_abs_err, l2_err


# ==============================================================
# 2. Full 2D MMS (y-dependent single layer)
# ==============================================================

def run_mms_2d(N: int, dt=None) -> tuple[float, float, float, float]:
    """
    Full 2D manufactured solution with y-dependent correction.

    T*(x,y,t) = 300 + A sin(wt)(b-x)^4 + B sin(wt)(x-a)^2(b-x)^2 cos(pi*y/(d-c))

    Boundary conditions:
      Left flux:  q = -k dT*/dx|_{x=a} = 4kA(b-a)^3 sin(wt)  (correction vanishes at x=a)
      Right:      T*(b,y,t) = 300                               (correction vanishes at x=b)
      Top/bottom: dT*/dy = 0 at y=c,d                           (cos -> sin derivative = 0)

    Returns (h, dt, max_abs_error, l2_error).
    """
    a, b, c, d = 0.0, 1.0, 0.0, 1.0
    L = b - a
    rho, cp, k = 1.0, 1.0, 1.0

    flux_f = 2.0
    omega = 2.0 * np.pi * flux_f
    phase = 0.0
    flux_A = 50.0
    A = flux_A / (4.0 * k * L ** 3)
    B = 20.0  # amplitude of y-dependent correction

    kappa = np.pi / (d - c)  # wave number for y-mode

    def T_star(X, Y, t):
        g = np.sin(omega * t + phase)
        T_1d = 300.0 + A * g * (b - X) ** 4
        phi = (X - a) ** 2 * (b - X) ** 2
        T_corr = B * g * phi * np.cos(kappa * (Y - c))
        return T_1d + T_corr

    def q_star(t):
        # Left flux: correction phi(a) = 0, so same as 1D
        return 4.0 * k * A * L ** 3 * np.sin(omega * t + phase)

    def s_star(X, Y, t):
        g = np.sin(omega * t + phase)
        gp = omega * np.cos(omega * t + phase)

        # 1D part: dT/dt - k d²T/dx²
        s_1d = (rho * cp * A * gp * (b - X) ** 4
                - k * 12.0 * A * g * (b - X) ** 2)

        # Correction phi = (x-a)^2 (b-x)^2
        phi = (X - a) ** 2 * (b - X) ** 2
        # phi'' = d²phi/dx² = 2(b-x)^2 + 2(x-a)^2 - 8(x-a)(b-x)
        phi_xx = 2.0 * (b - X) ** 2 + 2.0 * (X - a) ** 2 - 8.0 * (X - a) * (b - X)
        cos_y = np.cos(kappa * (Y - c))

        # dT_corr/dt = B omega cos(wt) phi cos(ky)
        s_corr_t = rho * cp * B * gp * phi * cos_y

        # k d²T_corr/dx² = k B sin(wt) phi'' cos(ky)
        s_corr_xx = k * B * g * phi_xx * cos_y

        # k d²T_corr/dy² = -k B sin(wt) phi kappa² cos(ky)
        s_corr_yy = -k * B * g * phi * kappa ** 2 * cos_y

        return s_1d + s_corr_t - s_corr_xx - s_corr_yy

    layer = Layer2D(x_left=a, x_right=b, rho=rho, cp=cp, k=k)
    sim = FVSolver2D(
        a=a, b=b, c=c, d=d,
        Nx=N, Ny=N,
        lam_target=0.5,
        layers=[layer],
        t_final=0.37,
        flux_f=flux_f, flux_A=flux_A,
        dt=dt,
        t_on=0.0, t_off=0.2, phase=0.0,
        source=s_star,
        q_left_fn=q_star,
    )

    T0 = T_star(sim.X, sim.Y, sim.t[0])
    _, _, _, T_final_num = sim.solve(T0=T0, store_trajectory=False)
    T_final_exact = T_star(sim.X, sim.Y, sim.t[-1])

    error = T_final_num - T_final_exact
    max_abs_err = float(np.max(np.abs(error)))
    l2_err = float(np.sqrt(np.mean(error ** 2)))
    return sim.hx, sim.dt, max_abs_err, l2_err


# ==============================================================
# 3. Interface MMS (piecewise 2D with R_c)
# ==============================================================

def run_mms_2d_interface(N: int, dt=None) -> tuple[float, float, float, float]:
    """
    2D MMS with two-layer interface and thermal resistance.

    Base 1D solution (from mms_1d.py):
      T_L,1D(x,t) = 300 + A_L sin(wt)[(x_I-x)^4 + D_L(x_I-x)] + C_L sin(wt)
      T_R,1D(x,t) = 300 + A_R sin(wt)(b-x)^4

    2D correction (vanishes at interface with zero x-derivative):
      T_L*(x,y,t) = T_L,1D + B_L sin(wt)(x-a)^2(x_I-x)^2 cos(kappa(y-c))
      T_R*(x,y,t) = T_R,1D + B_R sin(wt)(x-x_I)^2(b-x)^2  cos(kappa(y-c))

    N must be even so x=0.5 lands on a cell face.

    Returns (h, dt, max_abs_error, l2_error).
    """
    a, b, c, d = 0.0, 1.0, 0.0, 1.0
    x_I = 0.5
    k1, k2 = 2.0, 1.0
    rho1, cp1 = 1.5, 1.0
    rho2, cp2 = 0.8, 1.0
    Rc = 0.1

    flux_f = 2.0
    omega = 2.0 * np.pi * flux_f
    A_L, D_L = 10.0, 1.0
    A_R = 2.0 * k1 * A_L * D_L / k2           # = 40
    C_L = k1 * A_L * D_L * (1.0 / (8 * k2) + Rc)  # = 4.5

    B_L = 15.0   # y-correction amplitude, left
    B_R = 10.0   # y-correction amplitude, right
    kappa = np.pi / (d - c)

    # --- manufactured solution ---
    def T_star(X, Y, t):
        T = np.full_like(X, 300.0)
        g = np.sin(omega * t)
        cos_y = np.cos(kappa * (Y - c))
        left = X < x_I

        # 1D base
        T[left] += (A_L * g * ((x_I - X[left]) ** 4 + D_L * (x_I - X[left]))
                     + C_L * g)
        T[~left] += A_R * g * (b - X[~left]) ** 4

        # 2D corrections
        phi_L = (X[left] - a) ** 2 * (x_I - X[left]) ** 2
        phi_R = (X[~left] - x_I) ** 2 * (b - X[~left]) ** 2
        T[left] += B_L * g * phi_L * cos_y[left]
        T[~left] += B_R * g * phi_R * cos_y[~left]
        return T

    # --- left boundary flux: q = -k1 dT_L*/dx|_{x=a} ---
    # Correction phi_L(a) = 0 and phi_L'(a) = 0, so same as 1D
    def q_left(t):
        return k1 * A_L * np.sin(omega * t) * (4.0 * (x_I - a) ** 3 + D_L)

    # --- source ---
    def source(X, Y, t):
        s = np.empty_like(X)
        g = np.sin(omega * t)
        gp = omega * np.cos(omega * t)
        cos_y = np.cos(kappa * (Y - c))
        left = X < x_I
        xl = X[left]
        xr = X[~left]

        # 1D source
        s_L_1d = (rho1 * cp1 * gp * (A_L * ((x_I - xl) ** 4 + D_L * (x_I - xl)) + C_L)
                   - k1 * A_L * g * 12.0 * (x_I - xl) ** 2)
        s_R_1d = (rho2 * cp2 * A_R * gp * (b - xr) ** 4
                   - k2 * A_R * g * 12.0 * (b - xr) ** 2)

        # Left correction source
        phi_L = (xl - a) ** 2 * (x_I - xl) ** 2
        phi_L_xx = (2.0 * (x_I - xl) ** 2 + 2.0 * (xl - a) ** 2
                    - 8.0 * (xl - a) * (x_I - xl))
        cos_yL = cos_y[left]

        s_corr_L = (rho1 * cp1 * B_L * gp * phi_L * cos_yL
                     - k1 * B_L * g * phi_L_xx * cos_yL
                     - k1 * B_L * g * phi_L * (-kappa ** 2) * cos_yL)

        # Right correction source
        phi_R = (xr - x_I) ** 2 * (b - xr) ** 2
        phi_R_xx = (2.0 * (b - xr) ** 2 + 2.0 * (xr - x_I) ** 2
                    - 8.0 * (xr - x_I) * (b - xr))
        cos_yR = cos_y[~left]

        s_corr_R = (rho2 * cp2 * B_R * gp * phi_R * cos_yR
                     - k2 * B_R * g * phi_R_xx * cos_yR
                     - k2 * B_R * g * phi_R * (-kappa ** 2) * cos_yR)

        s[left] = s_L_1d + s_corr_L
        s[~left] = s_R_1d + s_corr_R
        return s

    # --- solver ---
    layers = [
        Layer2D(x_left=a, x_right=x_I, rho=rho1, cp=cp1, k=k1),
        Layer2D(x_left=x_I, x_right=b, rho=rho2, cp=cp2, k=k2),
    ]
    sim = FVSolver2D(
        a=a, b=b, c=c, d=d,
        Nx=N, Ny=N,
        lam_target=0.5,
        layers=layers,
        interface_R=[Rc],
        t_final=0.37,
        flux_f=flux_f, flux_A=0.0,
        dt=dt,
        t_on=0.0, t_off=0.2, phase=0.0,
        source=source,
        q_left_fn=q_left,
    )

    T0 = T_star(sim.X, sim.Y, sim.t[0])
    _, _, _, T_final_num = sim.solve(T0=T0, store_trajectory=False)
    T_final_exact = T_star(sim.X, sim.Y, sim.t[-1])

    error = T_final_num - T_final_exact
    max_abs_err = float(np.max(np.abs(error)))
    l2_err = float(np.sqrt(np.mean(error ** 2)))
    return sim.hx, sim.dt, max_abs_err, l2_err


# ==============================================================
# Order-of-convergence helpers
# ==============================================================

def _pairwise_orders(hs, errs):
    orders = []
    for i in range(len(errs) - 1):
        p = np.log(errs[i] / errs[i + 1]) / np.log(hs[i] / hs[i + 1])
        orders.append(p)
    return orders


def space_order_test_y_independent(N_list: list) -> float:
    results, hs = [], []
    for N in N_list:
        h, _, _, l2 = run_mms_y_independent(N, dt=0.0001)
        results.append(l2)
        hs.append(h)
    return float(np.mean(_pairwise_orders(hs, results)))


def time_order_test_y_independent(dt_list: list) -> float:
    results = []
    for dt_val in dt_list:
        _, _, _, l2 = run_mms_y_independent(N=201, dt=dt_val)
        results.append(l2)
    return float(np.mean(_pairwise_orders(dt_list, results)))


def space_order_test_2d(N_list: list) -> float:
    results, hs = [], []
    for N in N_list:
        h, _, _, l2 = run_mms_2d(N, dt=0.0001)
        results.append(l2)
        hs.append(h)
    return float(np.mean(_pairwise_orders(hs, results)))


def time_order_test_2d(dt_list: list) -> float:
    results = []
    for dt_val in dt_list:
        _, _, _, l2 = run_mms_2d(N=201, dt=dt_val)
        results.append(l2)
    return float(np.mean(_pairwise_orders(dt_list, results)))


def space_order_test_2d_interface(N_list: list) -> float:
    results, hs = [], []
    for N in N_list:
        h, _, _, l2 = run_mms_2d_interface(N, dt=0.0001)
        results.append(l2)
        hs.append(h)
    return float(np.mean(_pairwise_orders(hs, results)))


def time_order_test_2d_interface(dt_list: list) -> float:
    results = []
    for dt_val in dt_list:
        _, _, _, l2 = run_mms_2d_interface(N=200, dt=dt_val)
        results.append(l2)
    return float(np.mean(_pairwise_orders(dt_list, results)))


# ==============================================================
# Main: print convergence tables
# ==============================================================

if __name__ == '__main__':
    print("=== y-independent MMS (2D grid) ===")
    h, dt, me, l2 = run_mms_y_independent(N=101)
    print(f"N=101: h={h:.5f}, dt={dt:.6f}, max_err={me:.6e}, l2_err={l2:.6e}")
    p = space_order_test_y_independent([21, 41, 81, 161])
    print(f"Spatial order: {p:.3f}")
    p = time_order_test_y_independent([0.02, 0.01, 0.005])
    print(f"Temporal order: {p:.3f}")

    print("\n=== Full 2D MMS ===")
    h, dt, me, l2 = run_mms_2d(N=101)
    print(f"N=101: h={h:.5f}, dt={dt:.6f}, max_err={me:.6e}, l2_err={l2:.6e}")
    p = space_order_test_2d([21, 41, 81, 161])
    print(f"Spatial order: {p:.3f}")
    p = time_order_test_2d([0.02, 0.01, 0.005])
    print(f"Temporal order: {p:.3f}")

    print("\n=== 2D Interface MMS ===")
    h, dt, me, l2 = run_mms_2d_interface(N=100)
    print(f"N=100: h={h:.5f}, dt={dt:.6f}, max_err={me:.6e}, l2_err={l2:.6e}")
    p = space_order_test_2d_interface([50, 100, 200])
    print(f"Spatial order: {p:.3f}")
    p = time_order_test_2d_interface([0.02, 0.01, 0.005])
    print(f"Temporal order: {p:.3f}")
