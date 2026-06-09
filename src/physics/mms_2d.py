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
# 2b. y-dependent left-flux MMS (verifies vector q_left path)
# ==============================================================

def run_mms_2d_yflux(N: int, dt=None) -> tuple[float, float, float, float]:
    """
    Manufactured solution with separable y-dependent left Neumann flux.

    T*(x, y, t) = 300 + A sin(wt) (b-x)^4 cos(pi (y-c)/(d-c))

    Boundary conditions:
      Left:       q*(y,t) = -k dT*/dx|_{x=a}
                          = 4 k A sin(wt) (b-a)^3 cos(pi (y-c)/(d-c))
                          = a*(t) * s*(y)   (separable)
      Right:      T*(b,y,t) = 300            (factor (b-x)^4 vanishes)
      Top/bottom: dT*/dy = 0 at y=c, y=d     (sin(0) = sin(pi) = 0)

    This case exists specifically to verify the vector-valued q_left path —
    the only existing MMS cases use scalar q*(t).
    """
    a, b, c, d = 0.0, 1.0, 0.0, 1.0
    L = b - a
    Ly = d - c
    rho, cp, k = 1.0, 1.0, 1.0

    flux_f = 2.0
    omega = 2.0 * np.pi * flux_f
    phase = 0.0
    A = 10.0
    kappa = np.pi / Ly

    def cos_y(Y):
        return np.cos(kappa * (Y - c))

    def T_star(X, Y, t):
        return 300.0 + A * np.sin(omega * t + phase) * (b - X) ** 4 * cos_y(Y)

    def q_star(t):
        # Vector-valued: shape (Ny,)
        return 4.0 * k * A * L ** 3 * np.sin(omega * t + phase) * cos_y(np.linspace(c, d, N))

    def s_star(X, Y, t):
        g = np.sin(omega * t + phase)
        gp = omega * np.cos(omega * t + phase)
        cy = cos_y(Y)
        # dT/dt
        s_t = rho * cp * A * gp * (b - X) ** 4 * cy
        # k d²T/dx²
        s_xx = k * 12.0 * A * g * (b - X) ** 2 * cy
        # k d²T/dy² = -k A g (b-x)^4 kappa^2 cos_y
        s_yy = -k * A * g * (b - X) ** 4 * kappa ** 2 * cy
        return s_t - s_xx - s_yy

    layer = Layer2D(x_left=a, x_right=b, rho=rho, cp=cp, k=k)
    sim = FVSolver2D(
        a=a, b=b, c=c, d=d,
        Nx=N, Ny=N,
        lam_target=0.5,
        layers=[layer],
        t_final=0.37,
        flux_f=flux_f, flux_A=0.0,
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
# 2d. x-linear MMS (isolates the y-Laplacian truncation)
# ==============================================================

def run_mms_x_linear(N: int, dt=None) -> tuple[float, float, float, float]:
    """
    y-direction-isolating manufactured solution: linear in x, curved in y.

    T*(x, y, t) = 300 + A sin(wt) (b-x) cos(pi (y-c)/(d-c))

    The (b-x) factor is linear in x, so d²T*/dx² = 0 exactly and the FV
    x-stencil is reproduced with no truncation error. The only spatial
    truncation comes from the y-Laplacian acting on cos(pi (y-c)/(d-c)), so the
    observed spatial order measures the y-direction stencil. This is the mirror
    image of run_mms_y_independent (curved in x, flat in y), which isolates x.

    Scope: because the solver enforces an isotropic grid (hx == hy) and N drives
    both, this isolates the y diffusion stencil on a uniform grid. It catches a
    y-Laplacian curvature bug that would otherwise be masked by the correct
    x-error; it cannot catch a bug that only manifests at hy != hx.

    Boundary conditions:
      Left:       q*(y,t) = -k dT*/dx|_{x=a} = k A sin(wt) cos(pi (y-c)/(d-c))
                            (separable a(t) s(y); no (b-a)^3 factor since x is linear)
      Right:      T*(b,y,t) = 300            (factor (b-x) vanishes)
      Top/bottom: dT*/dy = 0 at y=c, y=d     (sin(0) = sin(pi) = 0)

    Returns (h, dt, max_abs_error, l2_error).
    """
    a, b, c, d = 0.0, 1.0, 0.0, 1.0
    Ly = d - c
    rho, cp, k = 1.0, 1.0, 1.0

    flux_f = 2.0
    omega = 2.0 * np.pi * flux_f
    phase = 0.0
    A = 10.0
    kappa = np.pi / Ly

    def cos_y(Y):
        return np.cos(kappa * (Y - c))

    def T_star(X, Y, t):
        return 300.0 + A * np.sin(omega * t + phase) * (b - X) * cos_y(Y)

    def q_star(t):
        # Vector-valued: shape (Ny,). -k dT*/dx|_{x=a} with d/dx[(b-x)] = -1.
        return k * A * np.sin(omega * t + phase) * cos_y(np.linspace(c, d, N))

    def s_star(X, Y, t):
        g = np.sin(omega * t + phase)
        gp = omega * np.cos(omega * t + phase)
        P = (b - X)
        cy = cos_y(Y)
        # dT/dt
        s_t = rho * cp * A * gp * P * cy
        # k d²T/dx² = 0 (linear in x)
        # k d²T/dy² = -k A g (b-x) kappa^2 cos_y
        s_yy = -k * A * g * P * kappa ** 2 * cy
        return s_t - s_yy

    layer = Layer2D(x_left=a, x_right=b, rho=rho, cp=cp, k=k)
    sim = FVSolver2D(
        a=a, b=b, c=c, d=d,
        Nx=N, Ny=N,
        lam_target=0.5,
        layers=[layer],
        t_final=0.37,
        flux_f=flux_f, flux_A=0.0,
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
# 2c. smooth-in-time left-flux MMS with exact step-integral forcing
# ==============================================================

def run_mms_2d_smooth_forcing_integral(N: int, dt=None) -> tuple[float, float, float, float]:
    """
    Manufactured solution for the exact-integral left-forcing path.

    T*(x, y, t) = 300 + A sin(omega t + phase) (b-x)^4 cos(pi (y-c)/(d-c))

    The matching source is f = rho*cp*T_t - div(k grad T). The left flux uses
    the solver's positive-inward convention:

      q_L(y,t) = -k dT*/dx|_{x=a}
               = 4 k A sin(omega t + phase) (b-a)^3 cos(pi (y-c)/(d-c)).

    This case is intentionally smooth in time. Production exp/exp_train
    activation and pulse_train discontinuities belong in refined-reference
    tests, not strict second-order MMS.
    """
    a, b, c, d = 0.0, 1.0, 0.0, 1.0
    L = b - a
    Ly = d - c
    rho, cp, k = 1.0, 1.0, 1.0

    flux_f = 2.0
    omega = 2.0 * np.pi * flux_f
    phase = 0.37
    A = 10.0
    kappa = np.pi / Ly
    y_vec = np.linspace(c, d, N)
    cos_y_left = np.cos(kappa * (y_vec - c))

    def cos_y(Y):
        return np.cos(kappa * (Y - c))

    def g(t):
        return np.sin(omega * t + phase)

    def g_t(t):
        return omega * np.cos(omega * t + phase)

    def T_star(X, Y, t):
        return 300.0 + A * g(t) * (b - X) ** 4 * cos_y(Y)

    def q_left(t):
        return 4.0 * k * A * L ** 3 * g(t) * cos_y_left

    def q_left_integral(t_lo, t_hi):
        g_int = (np.cos(omega * t_lo + phase) - np.cos(omega * t_hi + phase)) / omega
        return 4.0 * k * A * L ** 3 * g_int * cos_y_left

    def source(X, Y, t):
        P = (b - X) ** 4
        cy = cos_y(Y)
        T_t = A * g_t(t) * P * cy
        T_xx = 12.0 * A * g(t) * (b - X) ** 2 * cy
        T_yy = -A * g(t) * P * kappa ** 2 * cy
        return rho * cp * T_t - k * (T_xx + T_yy)

    layer = Layer2D(x_left=a, x_right=b, rho=rho, cp=cp, k=k)
    sim = FVSolver2D(
        a=a, b=b, c=c, d=d,
        Nx=N, Ny=N,
        lam_target=0.5,
        layers=[layer],
        t_final=0.37,
        flux_f=flux_f, flux_A=0.0,
        dt=dt,
        t_on=0.0, t_off=0.2, phase=0.0,
        source=source,
        q_left_fn=q_left,
        q_left_integral_fn=q_left_integral,
    )

    T0 = T_star(sim.X, sim.Y, sim.t[0])
    _, _, _, T_final_num = sim.solve(T0=T0, store_trajectory=False)
    T_final_exact = T_star(sim.X, sim.Y, sim.t[-1])

    error = T_final_num - T_final_exact
    max_abs_err = float(np.max(np.abs(error)))
    l2_err = float(np.sqrt(np.mean(error ** 2)))
    return sim.hx, sim.dt, max_abs_err, l2_err


def smooth_forcing_left_flux_sign_sample() -> tuple[float, float, float]:
    """Return (q_L, -k*T_x, +k*T_x) for the smooth forcing MMS at x=a."""
    a, b, c, d = 0.0, 1.0, 0.0, 1.0
    k = 1.0
    flux_f = 2.0
    omega = 2.0 * np.pi * flux_f
    phase = 0.37
    A = 10.0
    t = 0.043
    y = 0.25
    kappa = np.pi / (d - c)
    g = np.sin(omega * t + phase)
    cos_y = np.cos(kappa * (y - c))
    T_x_left = -4.0 * A * g * (b - a) ** 3 * cos_y
    q_left = 4.0 * k * A * (b - a) ** 3 * g * cos_y
    return float(q_left), float(-k * T_x_left), float(k * T_x_left)


# ==============================================================
# 3. Interface MMS (piecewise 2D with R_c)
# ==============================================================

def run_mms_2d_interface(N: int, dt=None, x_I: float = 0.5) -> tuple[float, float, float, float]:
    """
    2D MMS with two-layer interface, thermal resistance, and y-dependent
    left flux — exercises the production physics path end-to-end.

    Manufactured solution carries a single shared y-mode cos(kappa*(y-c)):
      T_L*(x,y,t) = 300 + cos(kappa*(y-c)) * V_L(x,t)
      T_R*(x,y,t) = 300 + cos(kappa*(y-c)) * V_R(x,t)
    with
      V_L(x,t) = A_L g[(x_I-x)^4 + D_L(x_I-x)] + C_L g + B_L g (x-a)^2(x_I-x)^2
      V_R(x,t) = A_R g (b-x)^4                        + B_R g (x-x_I)^2(b-x)^2
      g = sin(omega t),  kappa = pi / (d - c)

    The shared cos(kappa*(y-c)) factors out of every interface condition, so the
    A_R / C_L relations (flux continuity, jump T_L - T_R = R_c q_I) are
    unchanged from the y-independent case.

    Boundary conditions:
      Left:       q*(y,t) = -k1 dT_L*/dx|_{x=a}
                          = k1 A_L cos(kappa*(y-c)) g [4(x_I-a)^3 + D_L]
                            (B_L correction has phi_L'(a)=0, contributes nothing)
      Right:      T_R*(b,y,t) = 300                (V_R(b,t) = 0)
      Top/bottom: dT*/dy = 0 at y=c,d              (cos -> sin derivative = 0)

    The default x_I = 0.5 places the interface at a cell-face midpoint
    (requires even N). Any other x_I is an off-center interface; choose N so x_I doesn't land on a node.

    Returns (h, dt, max_abs_error, l2_error).
    """
    a, b, c, d = 0.0, 1.0, 0.0, 1.0
    k1, k2 = 2.0, 1.0
    rho1, cp1 = 1.5, 1.0
    rho2, cp2 = 0.8, 1.0
    Rc = 0.1

    flux_f = 2.0
    omega = 2.0 * np.pi * flux_f
    A_L, D_L = 10.0, 1.0
    # Flux continuity at x_I:  k1 A_L D_L = 4 k2 A_R (b - x_I)^3
    A_R = k1 * A_L * D_L / (4.0 * k2 * (b - x_I) ** 3)
    # Jump condition T_L - T_R = R_c * q_I at x_I
    C_L = k1 * A_L * D_L * ((b - x_I) / (4.0 * k2) + Rc)

    B_L = 15.0   # y-correction amplitude, left
    B_R = 10.0   # y-correction amplitude, right
    kappa = np.pi / (d - c)

    def cos_y_arr(Y):
        return np.cos(kappa * (Y - c))

    # --- manufactured solution ---
    def T_star(X, Y, t):
        T = np.full_like(X, 300.0)
        g = np.sin(omega * t)
        cy = cos_y_arr(Y)
        left = X < x_I
        xl = X[left]
        xr = X[~left]

        V_L = (A_L * g * ((x_I - xl) ** 4 + D_L * (x_I - xl))
               + C_L * g
               + B_L * g * (xl - a) ** 2 * (x_I - xl) ** 2)
        V_R = (A_R * g * (b - xr) ** 4
               + B_R * g * (xr - x_I) ** 2 * (b - xr) ** 2)

        T[left] += cy[left] * V_L
        T[~left] += cy[~left] * V_R
        return T

    # --- left boundary flux: q = -k1 dT_L*/dx|_{x=a}, vector-valued (Ny,) ---
    y_vec = np.linspace(c, d, N)
    cos_y_left = np.cos(kappa * (y_vec - c))

    def q_left(t):
        return k1 * A_L * np.sin(omega * t) * (4.0 * (x_I - a) ** 3 + D_L) * cos_y_left

    # --- source: ρcp ∂V/∂t - k ∂²V/∂x² + k κ² V, all multiplied by cos(κ(y-c)) ---
    def source(X, Y, t):
        s = np.empty_like(X)
        g = np.sin(omega * t)
        gp = omega * np.cos(omega * t)
        cy = cos_y_arr(Y)
        left = X < x_I
        xl = X[left]
        xr = X[~left]

        # Left side: V_L and its t-derivative and xx-derivative
        phi_L = (xl - a) ** 2 * (x_I - xl) ** 2
        phi_L_xx = (2.0 * (x_I - xl) ** 2 + 2.0 * (xl - a) ** 2
                    - 8.0 * (xl - a) * (x_I - xl))
        V_L = (A_L * g * ((x_I - xl) ** 4 + D_L * (x_I - xl))
               + C_L * g
               + B_L * g * phi_L)
        Vt_L = (A_L * gp * ((x_I - xl) ** 4 + D_L * (x_I - xl))
                + C_L * gp
                + B_L * gp * phi_L)
        Vxx_L = A_L * g * 12.0 * (x_I - xl) ** 2 + B_L * g * phi_L_xx

        s[left] = cy[left] * (rho1 * cp1 * Vt_L - k1 * Vxx_L + k1 * kappa ** 2 * V_L)

        # Right side
        phi_R = (xr - x_I) ** 2 * (b - xr) ** 2
        phi_R_xx = (2.0 * (b - xr) ** 2 + 2.0 * (xr - x_I) ** 2
                    - 8.0 * (xr - x_I) * (b - xr))
        V_R = A_R * g * (b - xr) ** 4 + B_R * g * phi_R
        Vt_R = A_R * gp * (b - xr) ** 4 + B_R * gp * phi_R
        Vxx_R = A_R * g * 12.0 * (b - xr) ** 2 + B_R * g * phi_R_xx

        s[~left] = cy[~left] * (rho2 * cp2 * Vt_R - k2 * Vxx_R + k2 * kappa ** 2 * V_R)
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


# ======================================================
# 4. Patch source MMS (internal source with R_c)
# ======================================================

def run_mms_2d_patch_source(N: int, dt=None) -> tuple[float, float, float, float]:
    """
    Smooth-source MMS for the internal-source pathway.

    Manufactured solution:
      T*(x,y,t) = 300 + g(t) * phi(x) * psi(y)
      g(t)  = sin(omega * t),  omega = 6*pi
      phi(x) = (x-a)^2 (b-x)^2
      psi(y) = cos(pi*(y-c)/(d-c))

    The factors are chosen so that:
      - phi(a) = phi(b) = 0 and phi'(a) = phi'(b) = 0 → left flux is identically
        zero AND right boundary stays at T = 300. Matches the
        zero-Neumann / 300-Dirichlet wall conditions used by the internal-source
        experiment.
      - psi'(c) = psi'(d) = 0 → top/bottom Neumann zero-flux conditions hold.

    The required source is globally smooth (NOT a rectangular indicator):
      s*(X, Y, t) = rho*cp * dT/dt - k * d^2T/dx^2 - k * d^2T/dy^2

    To localize the source like the real patch experiment, we additionally
    modulate by a Gaussian envelope centered at (x_h, y_h). The Gaussian is
    smooth, so convergence orders are preserved, but the source is concentrated
    in a region of size sigma, mimicking the patch.

      env(X, Y) = exp(-((X - x_h)^2 + (Y - y_h)^2) / (2 sigma^2))

    NOTE: The manufactured T* is the *full* analytic solution including the
    Gaussian-shaped source. We therefore compute s* directly from derivatives
    of T*_full(X, Y, t) = 300 + g(t) * env(X, Y) * phi(X) * psi(Y).

    This verifies the source pathway end-to-end with the expected
    second-order spatial and temporal convergence.

    Returns (h, dt, max_abs_error, l2_error).
    """
    a, b, c, d = 0.0, 1.0, 0.0, 1.0
    L = b - a
    Ly = d - c
    rho, cp, k = 1.0, 1.0, 1.0

    # omega=6*pi keeps the CN temporal truncation above the spatial error floor
    # at N=201 across dt in [0.04, 0.02, 0.01], so the temporal study is not
    # contaminated by spatial error (the solution amplitude here is small).
    omega = 6.0 * np.pi
    kappa = np.pi / Ly

    x_h, y_h = 0.5, 0.5
    # sigma must keep the source negligible at the Neumann walls (x=a, y=c, y=d).
    # The solver point-samples the source at boundary half-cells, so a source that
    # is non-trivial there leaves an O(1) half-cell quadrature residual that does
    # not vanish under refinement (sigma=0.15 stalls at ~3.5e-4, order ~0.1).
    # sigma=0.10 gives env ~ 4e-6 at the walls, recovering clean 2nd order, and
    # still matches the strictly-interior real patch source.
    sigma = 0.10
    inv_2sig2 = 1.0 / (2.0 * sigma ** 2)

    def env(X, Y):
        return np.exp(-((X - x_h) ** 2 + (Y - y_h) ** 2) * inv_2sig2)

    def env_x(X, Y):
        return -((X - x_h) / sigma ** 2) * env(X, Y)

    def env_xx(X, Y):
        return ((X - x_h) ** 2 / sigma ** 4 - 1.0 / sigma ** 2) * env(X, Y)

    def env_y(X, Y):
        return -((Y - y_h) / sigma ** 2) * env(X, Y)

    def env_yy(X, Y):
        return ((Y - y_h) ** 2 / sigma ** 4 - 1.0 / sigma ** 2) * env(X, Y)

    def phi(X):
        return (X - a) ** 2 * (b - X) ** 2

    def phi_x(X):
        return 2.0 * (X - a) * (b - X) ** 2 - 2.0 * (X - a) ** 2 * (b - X)

    def phi_xx(X):
        return 2.0 * (b - X) ** 2 - 8.0 * (X - a) * (b - X) + 2.0 * (X - a) ** 2

    def psi(Y):
        return np.cos(kappa * (Y - c))

    def psi_y(Y):
        return -kappa * np.sin(kappa * (Y - c))

    def psi_yy(Y):
        return -kappa ** 2 * np.cos(kappa * (Y - c))

    def F(X, Y):
        return env(X, Y) * phi(X) * psi(Y)

    def F_xx(X, Y):
        e = env(X, Y)
        ex = env_x(X, Y)
        exx = env_xx(X, Y)
        return (exx * phi(X) + 2.0 * ex * phi_x(X) + e * phi_xx(X)) * psi(Y)

    def F_yy(X, Y):
        e = env(X, Y)
        ey = env_y(X, Y)
        eyy = env_yy(X, Y)
        return (eyy * psi(Y) + 2.0 * ey * psi_y(Y) + e * psi_yy(Y)) * phi(X)

    def T_star(X, Y, t):
        return 300.0 + np.sin(omega * t) * F(X, Y)

    def s_star(X, Y, t):
        g = np.sin(omega * t)
        gp = omega * np.cos(omega * t)
        return rho * cp * gp * F(X, Y) - k * g * (F_xx(X, Y) + F_yy(X, Y))

    y_vec = np.linspace(c, d, N)
    zero_q = np.zeros_like(y_vec)

    def q_left(t):
        return zero_q

    layer = Layer2D(x_left=a, x_right=b, rho=rho, cp=cp, k=k)
    sim = FVSolver2D(
        a=a, b=b, c=c, d=d,
        Nx=N, Ny=N,
        lam_target=0.5,
        layers=[layer],
        t_final=0.37,
        flux_f=1.0, flux_A=0.0,
        dt=dt,
        t_on=0.0, t_off=0.2, phase=0.0,
        source=s_star,
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


def space_order_test_2d_yflux(N_list: list) -> float:
    results, hs = [], []
    for N in N_list:
        h, _, _, l2 = run_mms_2d_yflux(N, dt=0.0001)
        results.append(l2)
        hs.append(h)
    return float(np.mean(_pairwise_orders(hs, results)))


def time_order_test_2d_yflux(dt_list: list) -> float:
    results = []
    for dt_val in dt_list:
        _, _, _, l2 = run_mms_2d_yflux(N=201, dt=dt_val)
        results.append(l2)
    return float(np.mean(_pairwise_orders(dt_list, results)))


def space_order_test_x_linear(N_list: list) -> float:
    results, hs = [], []
    for N in N_list:
        h, _, _, l2 = run_mms_x_linear(N, dt=0.0001)
        results.append(l2)
        hs.append(h)
    return float(np.mean(_pairwise_orders(hs, results)))


def time_order_test_x_linear(dt_list: list) -> float:
    results = []
    for dt_val in dt_list:
        _, _, _, l2 = run_mms_x_linear(N=201, dt=dt_val)
        results.append(l2)
    return float(np.mean(_pairwise_orders(dt_list, results)))


def time_order_test_2d_smooth_forcing_integral(dt_list: list) -> float:
    results = []
    for dt_val in dt_list:
        _, _, _, l2 = run_mms_2d_smooth_forcing_integral(N=201, dt=dt_val)
        results.append(l2)
    return float(np.mean(_pairwise_orders(dt_list, results)))


def space_order_test_2d_off_center_interface(N_list: list, x_I: float = 0.4734) -> float:
    """Spatial convergence test with interface not on a cell-face midpoint."""
    results, hs = [], []
    for N in N_list:
        h, _, _, l2 = run_mms_2d_interface(N, dt=0.0001, x_I=x_I)
        results.append(l2)
        hs.append(h)
    return float(np.mean(_pairwise_orders(hs, results)))


def time_order_test_2d_off_center_interface(dt_list: list, x_I: float = 0.4734) -> float:
    """Temporal convergence test with interface not on a cell-face midpoint."""
    results = []
    for dt_val in dt_list:
        _, _, _, l2 = run_mms_2d_interface(N=200, dt=dt_val, x_I=x_I)
        results.append(l2)
    return float(np.mean(_pairwise_orders(dt_list, results)))


def space_order_test_2d_patch_source(N_list: list) -> float:
    results, hs = [], []
    for N in N_list:
        h, _, _, l2 = run_mms_2d_patch_source(N, dt=0.0001)
        results.append(l2)
        hs.append(h)
    return float(np.mean(_pairwise_orders(hs, results)))


def time_order_test_2d_patch_source(dt_list: list) -> float:
    results = []
    for dt_val in dt_list:
        _, _, _, l2 = run_mms_2d_patch_source(N=201, dt=dt_val)
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

    print("\n=== y-dependent Left-Flux MMS (vector q_left) ===")
    h, dt, me, l2 = run_mms_2d_yflux(N=101)
    print(f"N=101: h={h:.5f}, dt={dt:.6f}, max_err={me:.6e}, l2_err={l2:.6e}")
    p = space_order_test_2d_yflux([21, 41, 81, 161])
    print(f"Spatial order: {p:.3f}")
    p = time_order_test_2d_yflux([0.02, 0.01, 0.005])
    print(f"Temporal order: {p:.3f}")

    print("\n=== x-linear MMS (isolates y-Laplacian) ===")
    h, dt, me, l2 = run_mms_x_linear(N=101)
    print(f"N=101: h={h:.5f}, dt={dt:.6f}, max_err={me:.6e}, l2_err={l2:.6e}")
    p = space_order_test_x_linear([21, 41, 81, 161])
    print(f"Spatial order: {p:.3f}")
    p = time_order_test_x_linear([0.02, 0.01, 0.005])
    print(f"Temporal order: {p:.3f}")

    print("\n=== 2D Interface MMS ===")
    h, dt, me, l2 = run_mms_2d_interface(N=100)
    print(f"N=100: h={h:.5f}, dt={dt:.6f}, max_err={me:.6e}, l2_err={l2:.6e}")
    p = space_order_test_2d_interface([50, 100, 200])
    print(f"Spatial order: {p:.3f}")
    p = time_order_test_2d_interface([0.02, 0.01, 0.005])
    print(f"Temporal order: {p:.3f}")

    print("\n=== 2D Off-Center Interface MMS (x_I=0.4734) ===")
    h, dt, me, l2 = run_mms_2d_interface(N=100, dt=0.0001, x_I=0.4734)
    print(f"N=100: h={h:.5f}, dt={dt:.6f}, max_err={me:.6e}, l2_err={l2:.6e}")
    p = space_order_test_2d_off_center_interface([50, 100, 200])
    print(f"Spatial order: {p:.3f}")
    p = time_order_test_2d_off_center_interface([0.02, 0.01, 0.005])
    print(f"Temporal order: {p:.3f}")

    print("\n=== 2D Patch Source MMS ===")
    h, dt, me, l2 = run_mms_2d_patch_source(N=101)
    print(f"N=101: h={h:.5f}, dt={dt:.6f}, max_err={me:.6e}, l2_err={l2:.6e}")
    p = space_order_test_2d_patch_source([21, 41, 81, 161])
    print(f"Spatial order: {p:.3f}")
    p = time_order_test_2d_patch_source([0.04, 0.02, 0.01])
    print(f"Temporal order: {p:.3f}")
