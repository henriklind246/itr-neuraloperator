from src.physics.fd_solver_1d import FDSolver1D
import numpy as np

# T_star input is (x_grid, t_grid) output is


def run_mms_once(N: int, dt=None) -> tuple[float, float, float, float]:
    a, b = 0.0, 1.0
    L = b-a
    rho, cp, k = 1.0, 1.0, 1.0

    flux_f = 2.0
    omega = 2.0 * np.pi * flux_f
    phase = 0.0
    flux_A = 50.0

    # choose A such that it matches flux amplitude
    A = flux_A/(2.0 * k * L)

    # define T*(x,t)
    def T_star(x: np.ndarray, t: float) -> np.ndarray:
        return 300.0 + A * np.sin(omega * t + phase) * (b - x)**4

    def q_star(t: float) -> float:
        return 2.0 * k * A * L * np.sin(omega * t + phase)

    def s_star(x: np.ndarray, t: float) -> np.ndarray:
        return rho * cp * (A* omega*  np.cos(omega * t + phase)*(b-x)**2) - k * (2.0 * A * np.sin(omega * t + phase))

    # create instance of fd solver class to solve forcing equation
    sim = FDSolver1D(
        a=a,
        b=b,
        N=N,
        rho=rho,
        cp=cp,
        k=k,
        lam_target=0.5,
        t_final=1.0,
        flux_f=flux_f,
        flux_A=flux_A,
        dt=dt,
        t_on=0.0,
        t_off=1.0,
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

# fix dt
def space_order_test(N_list: list) -> float:
    results = []
    hs = []
    for N in N_list:
        h, _, _, l2_error = run_mms_once(N, dt=.00000002)

        results.append(l2_error)
        hs.append(h)

    # use generic formula for estimating the order
    p_x = np.log(results[0]/results[1])/np.log(hs[0]/hs[1])
    return p_x

# fix dx
def time_order_test(dt_list: list) -> float:
    """Keep N high so that dt is small"""
    t_results = []
    for t in dt_list:
        _, _, _, l2_error = run_mms_once(N = 404, dt=t)

        t_results.append(l2_error)

    # use generic formula for estimating the order
    p_t = np.log(t_results[0]/t_results[1])/np.log(dt_list[0]/dt_list[1])
    return p_t


if __name__ == '__main__':
    h, dt, max_error, l2_error = run_mms_once(N = 101)
    print(f"With h={h} and dt={dt}, max abs. error between exact soln and numerical soln is {max_error} and l2 error is {l2_error}")
    N_list = [51, 101]
    order_x = space_order_test(N_list=N_list)
    print(f"Spatial order of FD solver is roughly {order_x}")

    dt_list = [.005, .0025]
    order_t = time_order_test(dt_list=dt_list)
    print(f"Temporal order of FD solver is roughly {order_t}")

