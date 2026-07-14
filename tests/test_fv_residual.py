import numpy as np
import torch
import pytest

from src.physics.fv_solver_2d import FVSolver2D, Layer2D
from src.physics.fv_residual import (
    build_face_conductances,
    build_cn_geom,
    build_homogeneous_cn_geom,
    build_cn_geom_per_interface,
    locate_interface,
    interior_cn_residual,
    full_bc_cn_residual,
    FullBCData,
)
from src.operators.losses import full_bc_physics_loss

"""
Stage 1 correctness gates for `src/physics/fv_residual.py`.

Two independent checks:
  1. `build_face_conductances` / `build_cn_geom` reproduce the solver's
     `G_x`/`G_y`/`dx`/`dy` and `r_w/r_e/r_s/r_n` bit-for-bit (incl. the
     interface x-face) for the `forcing` geometry.
  2. The interior CN residual on consecutive ground-truth `save_stride=1`
     snapshots is at the solver's linear-solve floor (~0) in float64, and the
     float32 evaluation is characterized (a looser, documented floor) so the
     live training residual is judged against the right number.
"""


def _build_forcing_solver(dt: float = 0.005, t_final: float = 0.3):
    """The core `forcing` geometry: two slabs k=2/k=1, interface at x=0.5,
    scalar R_c, Nx=Ny=100. Mirrors `fv_solver_2d.py.__main__`. The default
    windowed-sin left flux is active, but it only touches the i=0 row, which the
    interior residual excludes."""
    layers = [
        Layer2D(x_left=0.0, x_right=0.5, rho=1, cp=1, k=2),
        Layer2D(x_left=0.5, x_right=1.0, rho=1, cp=1, k=1),
    ]
    sim = FVSolver2D(
        a=0.0, b=1.0, c=0.0, d=1.0,
        Nx=100, Ny=100,
        layers=layers,
        interface_R=[0.5],
        lam_target=0.5,
        dt=dt,
        flux_f=2.0, flux_A=50.0,
        t_on=0.0, t_off=0.2,
        t_final=t_final, phase=0.0,
    )
    return sim


def _build_homogeneous_solver(N: int = 10, dt: float = 0.005, t_final: float = 0.01):
    return FVSolver2D(
        a=0.0, b=1.0, c=0.0, d=1.0,
        Nx=N, Ny=N,
        layers=[Layer2D(0.0, 1.0, rho=1.0, cp=1.0, k=1.0)],
        interface_R=None,
        lam_target=0.5,
        dt=dt,
        flux_f=1.0, flux_A=0.0,
        t_on=0.0, t_off=0.2,
        t_final=t_final, phase=0.0,
    )


def test_homogeneous_geometry_matches_one_layer_solver():
    sim = _build_homogeneous_solver()
    geom = build_homogeneous_cn_geom(
        sim.grid_x, sim.grid_y, k=1.0, dt=sim.dt,
        rho=1.0, cp=1.0, dtype=torch.float64,
    )
    for actual, expected in (
        (geom.G_x, sim.G_x), (geom.G_y, sim.G_y),
        (geom.dx, sim.dx), (geom.dy, sim.dy),
        (geom.r_w, sim.r_w), (geom.r_e, sim.r_e),
        (geom.r_s, sim.r_s), (geom.r_n, sim.r_n),
    ):
        np.testing.assert_allclose(actual.numpy(), expected, rtol=0, atol=1e-12)


def test_homogeneous_geometry_matches_zero_resistance_midpoint_interface():
    sim = _build_homogeneous_solver()
    direct = build_homogeneous_cn_geom(
        sim.grid_x, sim.grid_y, k=1.0, dt=sim.dt, dtype=torch.float64,
    )
    equivalent = build_cn_geom(
        sim.grid_x, sim.grid_y,
        k_left=1.0, k_right=1.0, interface_x=0.5, R_c=0.0,
        dt=sim.dt, dtype=torch.float64,
    )
    for name in ("G_x", "G_y", "dx", "dy", "r_w", "r_e", "r_s", "r_n"):
        assert torch.allclose(getattr(direct, name), getattr(equivalent, name), atol=1e-12)


def test_homogeneous_geometry_rejects_anisotropic_grid():
    with pytest.raises(ValueError, match="isotropic"):
        build_homogeneous_cn_geom(
            np.linspace(0.0, 1.0, 10), np.linspace(0.0, 1.0, 12),
            k=1.0, dt=0.005,
        )


def test_homogeneous_regions_normalize_as_physical_residual_over_sigma():
    sim = _build_homogeneous_solver()
    sigma = 7.25
    mu = 296.0
    torch.manual_seed(17)
    T_n = 300.0 + torch.randn((2, sim.Nx, sim.Ny), dtype=torch.float64)
    T_np1 = 300.0 + torch.randn((2, sim.Nx, sim.Ny), dtype=torch.float64)
    qn = torch.randn((2, sim.Ny), dtype=torch.float64)
    qnp1 = torch.randn((2, sim.Ny), dtype=torch.float64)
    qint = torch.randn((2, sim.Ny), dtype=torch.float64) * sim.dt

    physical = full_bc_cn_residual(
        T_n, T_np1,
        build_homogeneous_cn_geom(
            sim.grid_x, sim.grid_y, 1.0, sim.dt,
            sigma_global=1.0, dtype=torch.float64,
        ),
        FullBCData(
            T_right_tilde=torch.tensor(300.0, dtype=torch.float64),
            qL_n=qn, qL_np1=qnp1, qL_int=qint,
        ),
        dirichlet_both_ends=True,
    )
    normalized = full_bc_cn_residual(
        (T_n - mu) / sigma, (T_np1 - mu) / sigma,
        build_homogeneous_cn_geom(
            sim.grid_x, sim.grid_y, 1.0, sim.dt,
            sigma_global=sigma, dtype=torch.float64,
        ),
        FullBCData(
            T_right_tilde=torch.tensor((300.0 - mu) / sigma, dtype=torch.float64),
            qL_n=qn, qL_np1=qnp1, qL_int=qint,
        ),
        dirichlet_both_ends=True,
    )
    for name in physical:
        assert torch.allclose(normalized[name], physical[name] / sigma, atol=1e-12), name


def test_homogeneous_constant_state_is_zero_in_every_region():
    sim = _build_homogeneous_solver()
    geom = build_homogeneous_cn_geom(
        sim.grid_x, sim.grid_y, 1.0, sim.dt,
        sigma_global=5.0, dtype=torch.float64,
    )
    field = torch.full((1, sim.Nx, sim.Ny), (300.0 - 295.0) / 5.0, dtype=torch.float64)
    zero = torch.zeros((1, sim.Ny), dtype=torch.float64)
    parts = full_bc_cn_residual(
        field, field, geom,
        FullBCData(
            T_right_tilde=torch.tensor(1.0, dtype=torch.float64),
            qL_n=zero, qL_np1=zero, qL_int=zero,
        ),
    )
    assert all(float(value.abs().max()) == 0.0 for value in parts.values())


def test_homogeneous_solver_truth_requires_matching_exact_flux_integral():
    N = 10
    dt = 0.01
    amp = 40.0
    omega = 70.0

    def q_left(t):
        return amp * np.sin(omega * t) * np.ones(N)

    def q_left_integral(t0, t1):
        value = amp * (np.cos(omega * t0) - np.cos(omega * t1)) / omega
        return value * np.ones(N)

    sim = FVSolver2D(
        a=0.0, b=1.0, c=0.0, d=1.0,
        Nx=N, Ny=N,
        layers=[Layer2D(0.0, 1.0, rho=1.0, cp=1.0, k=1.0)],
        interface_R=None,
        lam_target=0.5,
        dt=dt,
        flux_f=1.0, flux_A=0.0,
        t_on=0.0, t_off=0.03,
        t_final=0.03, phase=0.0,
        q_left_fn=q_left,
        q_left_integral_fn=q_left_integral,
    )
    gx, gy = np.meshgrid(sim.grid_x, sim.grid_y, indexing="ij")
    T0 = 300.0 + 4.0 * (1.0 - gx) * np.sin(2.0 * np.pi * gy)
    _, _, _, history = sim.solve(T0=T0, store_trajectory=True)
    geom = build_homogeneous_cn_geom(
        sim.grid_x, sim.grid_y, 1.0, dt, dtype=torch.float64,
    )

    n = 1
    tn, tnp1 = float(sim.t[n]), float(sim.t[n + 1])
    exact = full_bc_cn_residual(
        torch.as_tensor(history[n], dtype=torch.float64),
        torch.as_tensor(history[n + 1], dtype=torch.float64),
        geom,
        FullBCData(
            T_right_tilde=torch.tensor(300.0, dtype=torch.float64),
            qL_n=torch.as_tensor(q_left(tn), dtype=torch.float64),
            qL_np1=torch.as_tensor(q_left(tnp1), dtype=torch.float64),
            qL_int=torch.as_tensor(q_left_integral(tn, tnp1), dtype=torch.float64),
        ),
    )
    assert all(float(value.abs().max()) < 1e-7 for value in exact.values())

    endpoint = full_bc_cn_residual(
        torch.as_tensor(history[n], dtype=torch.float64),
        torch.as_tensor(history[n + 1], dtype=torch.float64),
        geom,
        FullBCData(
            T_right_tilde=torch.tensor(300.0, dtype=torch.float64),
            qL_n=torch.as_tensor(q_left(tn), dtype=torch.float64),
            qL_np1=torch.as_tensor(q_left(tnp1), dtype=torch.float64),
        ),
    )
    assert float(endpoint["left_neumann"].abs().max()) > 1e-5


def test_left_corner_is_complete_control_volume_balance():
    sim = _build_homogeneous_solver()
    geom = build_homogeneous_cn_geom(
        sim.grid_x, sim.grid_y, 1.0, sim.dt, dtype=torch.float64,
    )
    base = torch.zeros((1, sim.Nx, sim.Ny), dtype=torch.float64)
    qzero = torch.zeros((1, sim.Ny), dtype=torch.float64)
    bc = FullBCData(
        T_right_tilde=torch.tensor(0.0), qL_n=qzero, qL_np1=qzero, qL_int=qzero,
    )

    y_neighbor = base.clone()
    y_neighbor[:, 0, 1] = 1.0
    x_neighbor = base.clone()
    x_neighbor[:, 1, 0] = 1.0
    y_corner = full_bc_cn_residual(base, y_neighbor, geom, bc)["left_neumann"][0]
    x_corner = full_bc_cn_residual(base, x_neighbor, geom, bc)["left_neumann"][0]
    assert torch.allclose(y_corner, -geom.r_n[0, 0])
    assert torch.allclose(x_corner, -geom.r_e[0, 0])


def test_face_conductances_match_solver():
    sim = _build_forcing_solver()
    G_x, G_y, dx, dy = build_face_conductances(
        sim.grid_x, sim.grid_y,
        k_left=2.0, k_right=1.0, interface_x=0.5, R_c=0.5,
        dtype=torch.float64,
    )
    np.testing.assert_allclose(G_x.numpy(), sim.G_x, rtol=0, atol=1e-12)
    np.testing.assert_allclose(G_y.numpy(), sim.G_y, rtol=0, atol=1e-12)
    np.testing.assert_allclose(dx.numpy(), sim.dx, rtol=0, atol=1e-12)
    np.testing.assert_allclose(dy.numpy(), sim.dy, rtol=0, atol=1e-12)


def test_interface_face_conductance_value():
    """The interface x-face (slot 49 for Nx=100, h_L=h_R=h/2) must equal the
    series resistance 1/(h_L/k_L + R_c + h_R/k_R), not a single-layer k/h."""
    sim = _build_forcing_solver()
    G_x, _, _, _ = build_face_conductances(
        sim.grid_x, sim.grid_y,
        k_left=2.0, k_right=1.0, interface_x=0.5, R_c=0.5,
        dtype=torch.float64,
    )
    h = float(sim.grid_x[1] - sim.grid_x[0])
    face_idx = int(np.floor((0.5 - 0.0) / h))
    expected = 1.0 / ((h / 2.0) / 2.0 + 0.5 + (h / 2.0) / 1.0)
    np.testing.assert_allclose(G_x.numpy()[face_idx, :], expected,
                               rtol=0, atol=1e-12)


def test_cn_coefficients_match_solver():
    sim = _build_forcing_solver()
    geom = build_cn_geom(
        sim.grid_x, sim.grid_y,
        k_left=2.0, k_right=1.0, interface_x=0.5, R_c=0.5,
        dt=sim.dt, dtype=torch.float64,
    )
    np.testing.assert_allclose(geom.r_w.numpy(), sim.r_w, rtol=0, atol=1e-12)
    np.testing.assert_allclose(geom.r_e.numpy(), sim.r_e, rtol=0, atol=1e-12)
    np.testing.assert_allclose(geom.r_s.numpy(), sim.r_s, rtol=0, atol=1e-12)
    np.testing.assert_allclose(geom.r_n.numpy(), sim.r_n, rtol=0, atol=1e-12)


def test_interior_residual_zero_on_truth_float64():
    """The interior CN residual on consecutive solver snapshots must be at the
    direct-solve floor. `splu` is essentially exact, so float64 residuals sit
    near machine precision relative to the field scale (~300 K)."""
    sim = _build_forcing_solver()
    T0 = np.full((sim.Nx, sim.Ny), sim.T_right(0.0), dtype=np.float64)
    _, _, _, T_hist = sim.solve(T0=T0, store_trajectory=True)

    geom = build_cn_geom(
        sim.grid_x, sim.grid_y,
        k_left=2.0, k_right=1.0, interface_x=0.5, R_c=0.5,
        dt=sim.dt, dtype=torch.float64,
    )

    # Sample several consecutive pairs spanning the forced ramp where interior
    # gradients are nontrivial.
    max_abs = 0.0
    for n in (1, 5, 10, 20, 40):
        T_n = torch.as_tensor(T_hist[n], dtype=torch.float64)
        T_np1 = torch.as_tensor(T_hist[n + 1], dtype=torch.float64)
        res, mask = interior_cn_residual(T_n, T_np1, geom)
        max_abs = max(max_abs, float(res.abs().max()))

    # Field scale ~300 K; direct-solve residual should be many orders below.
    assert max_abs < 1e-7, f"float64 interior residual floor too high: {max_abs:.3e}"


def test_interior_residual_float32_noise_floor():
    """Characterize (not gate hard) the float32 floor. `res` divides the small
    one-step change by dt=0.005 and differences fluxes, so float32 snapshots
    amplify ~7-digit cancellation; the live float32 training residual must be
    judged against this elevated floor, not the float64 one."""
    sim = _build_forcing_solver()
    T0 = np.full((sim.Nx, sim.Ny), sim.T_right(0.0), dtype=np.float64)
    _, _, _, T_hist = sim.solve(T0=T0, store_trajectory=True)

    geom32 = build_cn_geom(
        sim.grid_x, sim.grid_y,
        k_left=2.0, k_right=1.0, interface_x=0.5, R_c=0.5,
        dt=sim.dt, dtype=torch.float32,
    )

    max_abs = 0.0
    for n in (1, 5, 10, 20, 40):
        T_n = torch.as_tensor(T_hist[n].astype(np.float32), dtype=torch.float32)
        T_np1 = torch.as_tensor(T_hist[n + 1].astype(np.float32), dtype=torch.float32)
        res, _ = interior_cn_residual(T_n, T_np1, geom32)
        max_abs = max(max_abs, float(res.abs().max()))

    # Float32 floor is far above float64 but still small vs the field scale.
    # This is a documented characterization bound, not a tight gate.
    assert max_abs < 1e-1, f"float32 interior residual floor unexpectedly high: {max_abs:.3e}"
    # Sanity: float32 floor is strictly worse than float64 (cancellation), so it
    # should be well above machine-zero — guards against a silently-zeroed mask.
    assert max_abs > 1e-6, f"float32 floor suspiciously low (mask zeroed?): {max_abs:.3e}"


def test_interior_residual_batched_and_grad():
    """Residual accepts a batch dim, is differentiable wrt the prediction, and
    zeros every non-interior cell (boundary/Dirichlet rows)."""
    sim = _build_forcing_solver()
    geom = build_cn_geom(
        sim.grid_x, sim.grid_y,
        k_left=2.0, k_right=1.0, interface_x=0.5, R_c=0.5,
        dt=sim.dt, dtype=torch.float64,
    )
    Nx, Ny = sim.Nx, sim.Ny
    T_n = torch.zeros((2, Nx, Ny), dtype=torch.float64)
    T_np1 = torch.randn((2, Nx, Ny), dtype=torch.float64, requires_grad=True)

    res, mask = interior_cn_residual(T_n, T_np1, geom)
    assert res.shape == (2, Nx, Ny)

    # Boundary rows/cols are masked out.
    assert torch.all(res[:, 0, :] == 0)
    assert torch.all(res[:, -1, :] == 0)
    assert torch.all(res[:, :, 0] == 0)
    assert torch.all(res[:, :, -1] == 0)
    assert bool(mask[1:-1, 1:-1].all()) and not bool(mask[0, 0])

    loss = (res ** 2).mean()
    loss.backward()
    assert T_np1.grad is not None
    assert torch.isfinite(T_np1.grad).all()
    # Gradient is nonzero only on cells coupled to the interior.
    assert float(T_np1.grad.abs().sum()) > 0.0


# --- Stage 2: full-BC residual ---------------------------------------------
#
# The full-BC residual adds the solver's three boundary closures (right
# Dirichlet, top/bottom adiabatic, left time-dependent Neumann forcing) so it is
# ~0 across EVERY region on consecutive ground-truth snapshots — including the
# forced left row. The gate uses the sim's real `q_left`/`T_right`, so a missing
# or wrong Neumann reconstruction surfaces as a nonzero left-row residual rather
# than passing silently. Evaluated in float64 on raw (sigma_global=1) fields.


def _full_bc_data_for_step(sim, n: int, dtype=torch.float64) -> FullBCData:
    """Build `FullBCData` for the consecutive pair (snapshot n -> n+1) from the
    solver's own `q_left`/`T_right`, mirroring `cn_step`'s callable branch
    (`rhs[0]+=dt*(q(tn)+q(tn+dt))/(rho_cp_left*h)`; `rhs[Nx-1]=T_right(tn+dt)`).
    Raw physical fields => sigma_global=1, so T_right_tilde = T_right itself."""
    tn = float(sim.t[n])
    tnp1 = float(sim.t[n + 1])
    qL_n = torch.as_tensor(sim.q_left(tn), dtype=dtype)
    qL_np1 = torch.as_tensor(sim.q_left(tnp1), dtype=dtype)
    T_right_tilde = torch.as_tensor(float(sim.T_right(tnp1)), dtype=dtype)
    return FullBCData(T_right_tilde=T_right_tilde, qL_n=qL_n, qL_np1=qL_np1)


def test_full_bc_residual_zero_on_truth_float64():
    """`full_bc_cn_residual` on consecutive solver snapshots is at the
    direct-solve floor in EVERY region (interior, top/bottom adiabatic, left
    Neumann, right Dirichlet), using the sim's real forced left flux."""
    sim = _build_forcing_solver()
    T0 = np.full((sim.Nx, sim.Ny), sim.T_right(0.0), dtype=np.float64)
    _, _, _, T_hist = sim.solve(T0=T0, store_trajectory=True)

    geom = build_cn_geom(
        sim.grid_x, sim.grid_y,
        k_left=2.0, k_right=1.0, interface_x=0.5, R_c=0.5,
        dt=sim.dt, sigma_global=1.0, dtype=torch.float64,
    )

    region_max = {}
    for n in (1, 5, 10, 20, 40):
        T_n = torch.as_tensor(T_hist[n], dtype=torch.float64)
        T_np1 = torch.as_tensor(T_hist[n + 1], dtype=torch.float64)
        bc = _full_bc_data_for_step(sim, n)
        parts = full_bc_cn_residual(T_n, T_np1, geom, bc)
        for name, vec in parts.items():
            region_max[name] = max(region_max.get(name, 0.0),
                                   float(vec.abs().max()))

    for name, m in region_max.items():
        assert m < 1e-7, f"full_bc region {name} residual floor too high: {m:.3e}"


def test_full_bc_region_mse_zero_on_truth():
    """The loss reducer's per-region MSEs and the weighted/all-cell aggregates
    are all at the float64 floor on truth — the Stage-2 well-posedness gate as
    the training path will compute it."""
    sim = _build_forcing_solver()
    T0 = np.full((sim.Nx, sim.Ny), sim.T_right(0.0), dtype=np.float64)
    _, _, _, T_hist = sim.solve(T0=T0, store_trajectory=True)

    geom = build_cn_geom(
        sim.grid_x, sim.grid_y,
        k_left=2.0, k_right=1.0, interface_x=0.5, R_c=0.5,
        dt=sim.dt, sigma_global=1.0, dtype=torch.float64,
    )

    n = 10
    T_n = torch.as_tensor(T_hist[n], dtype=torch.float64)
    T_np1 = torch.as_tensor(T_hist[n + 1], dtype=torch.float64)
    bc = _full_bc_data_for_step(sim, n)
    out = full_bc_physics_loss(T_n, T_np1, geom, bc)

    for key in ("phys_interior_mse", "phys_left_neumann_mse",
                "phys_right_dirichlet_mse", "phys_topbot_adiabatic_mse",
                "physics_loss_weighted", "physics_loss_allcell_mean"):
        assert float(out[key]) < 1e-13, f"{key} too high on truth: {float(out[key]):.3e}"


def test_full_bc_left_neumann_required_on_forcing():
    """Negative control: dropping the left flux (assuming q_L=0) must blow up the
    left-Neumann region while the other regions stay at the floor — i.e. the
    Neumann reconstruction is REQUIRED on `forcing`, not optional."""
    sim = _build_forcing_solver()
    T0 = np.full((sim.Nx, sim.Ny), sim.T_right(0.0), dtype=np.float64)
    _, _, _, T_hist = sim.solve(T0=T0, store_trajectory=True)

    geom = build_cn_geom(
        sim.grid_x, sim.grid_y,
        k_left=2.0, k_right=1.0, interface_x=0.5, R_c=0.5,
        dt=sim.dt, sigma_global=1.0, dtype=torch.float64,
    )

    n = 10  # forcing window is t_on=0..t_off=0.2, so the ramp is active here
    T_n = torch.as_tensor(T_hist[n], dtype=torch.float64)
    T_np1 = torch.as_tensor(T_hist[n + 1], dtype=torch.float64)

    bc_real = _full_bc_data_for_step(sim, n)
    zero = torch.zeros_like(bc_real.qL_n)
    bc_zero = FullBCData(T_right_tilde=bc_real.T_right_tilde, qL_n=zero, qL_np1=zero)

    real = full_bc_physics_loss(T_n, T_np1, geom, bc_real)
    dropped = full_bc_physics_loss(T_n, T_np1, geom, bc_zero)

    # Only the left row should change; it must move far above the float64 floor.
    assert float(dropped["phys_left_neumann_mse"]) > 1e-6
    assert float(dropped["phys_left_neumann_mse"]) > 1e6 * float(real["phys_left_neumann_mse"])
    assert float(dropped["phys_interior_mse"]) == float(real["phys_interior_mse"])
    assert float(dropped["phys_right_dirichlet_mse"]) == float(real["phys_right_dirichlet_mse"])


def test_full_bc_shape_grad_and_both_ends():
    """Full-BC residual accepts a batch, is differentiable wrt the prediction,
    partitions the grid exactly, and exposes the both-ends Dirichlet term."""
    sim = _build_forcing_solver()
    geom = build_cn_geom(
        sim.grid_x, sim.grid_y,
        k_left=2.0, k_right=1.0, interface_x=0.5, R_c=0.5,
        dt=sim.dt, sigma_global=1.0, dtype=torch.float64,
    )
    Nx, Ny = sim.Nx, sim.Ny
    B = 3
    T_n = torch.zeros((B, Nx, Ny), dtype=torch.float64)
    T_np1 = torch.randn((B, Nx, Ny), dtype=torch.float64, requires_grad=True)
    bc = FullBCData(
        T_right_tilde=torch.zeros((B, Ny), dtype=torch.float64),
        qL_n=torch.zeros((B, Ny), dtype=torch.float64),
        qL_np1=torch.zeros((B, Ny), dtype=torch.float64),
    )

    parts = full_bc_cn_residual(T_n, T_np1, geom, bc, dirichlet_both_ends=True)
    # Region cell counts partition the full grid (single-end Dirichlet).
    n_cells = (parts["interior"].numel() + parts["left_neumann"].numel()
               + parts["right_dirichlet"].numel() + parts["topbot_adiabatic"].numel())
    assert n_cells == B * Nx * Ny
    assert "right_dirichlet_n" in parts
    assert parts["right_dirichlet_n"].numel() == B * Ny

    out = full_bc_physics_loss(T_n, T_np1, geom, bc, dirichlet_both_ends=True)
    loss = out["physics_loss_weighted"]
    loss.backward()
    assert T_np1.grad is not None and torch.isfinite(T_np1.grad).all()
    assert float(T_np1.grad.abs().sum()) > 0.0


def test_full_bc_keep_batch_shapes_and_flatten_equiv():
    """`keep_batch=True` retains the leading batch dim per region; the flattened
    default equals `reshape(-1)` of the kept-batch tensors bit-for-bit."""
    sim = _build_forcing_solver()
    geom = build_cn_geom(
        sim.grid_x, sim.grid_y,
        k_left=2.0, k_right=1.0, interface_x=0.5, R_c=0.5,
        dt=sim.dt, sigma_global=1.0, dtype=torch.float64,
    )
    Nx, Ny = sim.Nx, sim.Ny
    B = 3
    torch.manual_seed(0)
    T_n = torch.randn((B, Nx, Ny), dtype=torch.float64)
    T_np1 = torch.randn((B, Nx, Ny), dtype=torch.float64)
    bc = FullBCData(
        T_right_tilde=torch.randn((B, Ny), dtype=torch.float64),
        qL_n=torch.randn((B, Ny), dtype=torch.float64),
        qL_np1=torch.randn((B, Ny), dtype=torch.float64),
    )

    kept = full_bc_cn_residual(T_n, T_np1, geom, bc,
                               dirichlet_both_ends=True, keep_batch=True)
    flat = full_bc_cn_residual(T_n, T_np1, geom, bc, dirichlet_both_ends=True)

    assert kept["interior"].shape == (B, Nx - 2, Ny - 2)
    assert kept["topbot_adiabatic"].shape == (B, 2 * (Nx - 2))
    assert kept["left_neumann"].shape == (B, Ny)
    assert kept["right_dirichlet"].shape == (B, Ny)
    assert kept["right_dirichlet_n"].shape == (B, Ny)

    for name in kept:
        assert torch.equal(kept[name].reshape(-1), flat[name]), name


def test_full_bc_per_sample_means_match_scalar_regions():
    """`per_sample=True` per-region `(B,)` means average to the scalar region
    MSEs, and the scalar keys are numerically identical to the default path."""
    sim = _build_forcing_solver()
    geom = build_cn_geom(
        sim.grid_x, sim.grid_y,
        k_left=2.0, k_right=1.0, interface_x=0.5, R_c=0.5,
        dt=sim.dt, sigma_global=1.0, dtype=torch.float64,
    )
    Nx, Ny = sim.Nx, sim.Ny
    B = 4
    torch.manual_seed(1)
    T_n = torch.randn((B, Nx, Ny), dtype=torch.float64)
    T_np1 = torch.randn((B, Nx, Ny), dtype=torch.float64)
    bc = FullBCData(
        T_right_tilde=torch.randn((B, Ny), dtype=torch.float64),
        qL_n=torch.randn((B, Ny), dtype=torch.float64),
        qL_np1=torch.randn((B, Ny), dtype=torch.float64),
    )

    base = full_bc_physics_loss(T_n, T_np1, geom, bc)
    ps = full_bc_physics_loss(T_n, T_np1, geom, bc, per_sample=True)

    region_to_ps = {
        "phys_interior_mse": "interior_per_sample",
        "phys_left_neumann_mse": "left_neumann_per_sample",
        "phys_topbot_adiabatic_mse": "topbot_adiabatic_per_sample",
        "phys_right_dirichlet_mse": "right_dirichlet_per_sample",
    }
    for scalar_key, ps_key in region_to_ps.items():
        assert ps[ps_key].shape == (B,)
        # scalar == batch mean of the per-sample tensor.
        assert torch.allclose(ps[ps_key].mean(), ps[scalar_key], atol=0, rtol=0)
        # scalar keys identical to the default (flattened) path.
        assert torch.allclose(ps[scalar_key], base[scalar_key], atol=1e-12)

    for key in ("physics_loss_weighted", "physics_loss_allcell_mean"):
        assert torch.allclose(ps[key], base[key], atol=1e-12), key


def test_full_bc_per_sample_manual_two_rows():
    """A hand-checked 2-row case: the interior per-sample residual mean for each
    row equals the mean of that row's own interior cells (rows are independent)."""
    sim = _build_forcing_solver()
    geom = build_cn_geom(
        sim.grid_x, sim.grid_y,
        k_left=2.0, k_right=1.0, interface_x=0.5, R_c=0.5,
        dt=sim.dt, sigma_global=1.0, dtype=torch.float64,
    )
    Nx, Ny = sim.Nx, sim.Ny
    torch.manual_seed(2)
    T_n = torch.randn((2, Nx, Ny), dtype=torch.float64)
    T_np1 = torch.randn((2, Nx, Ny), dtype=torch.float64)
    bc = FullBCData(
        T_right_tilde=torch.zeros((2, Ny), dtype=torch.float64),
        qL_n=torch.zeros((2, Ny), dtype=torch.float64),
        qL_np1=torch.zeros((2, Ny), dtype=torch.float64),
    )

    kept = full_bc_cn_residual(T_n, T_np1, geom, bc, keep_batch=True)
    ps = full_bc_physics_loss(T_n, T_np1, geom, bc, per_sample=True)

    for row in (0, 1):
        manual = kept["interior"][row].pow(2).mean()
        assert torch.allclose(ps["interior_per_sample"][row], manual, atol=1e-12)

    # And each row uses only its own data: swapping row order permutes the
    # per-sample vector identically.
    kept_swap = full_bc_cn_residual(
        T_n.flip(0), T_np1.flip(0), geom,
        FullBCData(T_right_tilde=bc.T_right_tilde, qL_n=bc.qL_n, qL_np1=bc.qL_np1),
        keep_batch=True,
    )
    assert torch.allclose(
        kept_swap["interior"][0].pow(2).mean(),
        ps["interior_per_sample"][1], atol=1e-12,
    )


def test_per_interface_geom_exposes_face_idx():
    """`build_cn_geom_per_interface` publishes the per-sample `face_idx` `(B,)`
    equal to `locate_interface(...).face_idx` for each interface; the scalar
    `build_cn_geom` leaves `face_idx` unset (None)."""
    sim = _build_forcing_solver()
    xg, yg = sim.grid_x, sim.grid_y
    interface_x = [0.3, 0.5, 0.7, 0.42]
    R_c = [0.5, 0.4, 0.6, 0.5]
    geom = build_cn_geom_per_interface(
        xg, yg, k_left=2.0, k_right=1.0,
        interface_x_batch=interface_x, R_c_batch=R_c,
        dt=sim.dt, sigma_global=1.0, dtype=torch.float64,
    )
    assert geom.face_idx is not None
    assert geom.face_idx.shape == (len(interface_x),)
    assert geom.face_idx.dtype == torch.long
    expected = torch.tensor(
        [locate_interface(xg, ix).face_idx for ix in interface_x], dtype=torch.long
    )
    assert torch.equal(geom.face_idx, expected)

    scalar_geom = build_cn_geom(
        xg, yg, k_left=2.0, k_right=1.0, interface_x=0.5, R_c=0.5,
        dt=sim.dt, sigma_global=1.0, dtype=torch.float64,
    )
    assert scalar_geom.face_idx is None


def _build_pulse_train_solver(dt: float = 0.005, t_final: float = 0.3, Ny: int = 100):
    """A `forcing`-geometry solver driven by a single rectangular pulse that
    lives strictly inside one solver step. The solver injects the EXACT step
    integral of the flux (`q_left_integral_fn`), so a residual that reconstructs
    the left flux from the step ENDPOINTS misses the pulse entirely — the
    discriminating case for the exact-vs-trapezoid left Neumann closure."""
    from src.physics.boundary_forcing import (
        ramped_temporal,
        integrate_temporal_ramped_signed,
    )

    layers = [
        Layer2D(x_left=0.0, x_right=0.5, rho=1, cp=1, k=2),
        Layer2D(x_left=0.5, x_right=1.0, rho=1, cp=1, k=1),
    ]
    fam = "pulse_train"
    # Pulse [0.051, 0.053) is interior to step n=10 ([0.050, 0.055)).
    tparams = {"A_list": [200.0], "t_list": [0.051], "dt_list": [0.002]}
    t_ramp = 0.0
    a_fn = ramped_temporal(fam, tparams, t_ramp)

    def q_left_fn(t):
        return float(a_fn(t)) * np.ones(Ny)

    def q_left_integral_fn(t_lo, t_hi):
        val = integrate_temporal_ramped_signed(fam, tparams, t_lo, t_hi, t_ramp)
        return float(val) * np.ones(Ny)

    sim = FVSolver2D(
        a=0.0, b=1.0, c=0.0, d=1.0,
        Nx=Ny, Ny=Ny,
        layers=layers,
        interface_R=[0.5],
        lam_target=0.5,
        dt=dt,
        flux_f=2.0, flux_A=50.0,
        t_on=0.0, t_off=0.2,
        t_final=t_final, phase=0.0,
        q_left_fn=q_left_fn,
        q_left_integral_fn=q_left_integral_fn,
    )
    return sim


def test_full_bc_left_neumann_exact_integral_required_for_pulse():
    """Regression for the exact-integral left Neumann closure. When the solver
    injects the EXACT step flux integral (its `q_left_integral` path), the
    residual must use `2*qL_int`, not the trapezoid of the step endpoints. On a
    pulse that lives strictly inside one step the endpoints are both zero, so the
    trapezoid closure silently drops the forcing and the left row blows up, while
    the `qL_int` closure stays at the float64 solve floor."""
    sim = _build_pulse_train_solver()
    T0 = np.full((sim.Nx, sim.Ny), sim.T_right(0.0), dtype=np.float64)
    _, _, _, T_hist = sim.solve(T0=T0, store_trajectory=True)

    geom = build_cn_geom(
        sim.grid_x, sim.grid_y,
        k_left=2.0, k_right=1.0, interface_x=0.5, R_c=0.5,
        dt=sim.dt, sigma_global=1.0, dtype=torch.float64,
    )

    n = 10
    tn = float(sim.t[n])
    tnp1 = float(sim.t[n + 1])
    T_n = torch.as_tensor(T_hist[n], dtype=torch.float64)
    T_np1 = torch.as_tensor(T_hist[n + 1], dtype=torch.float64)

    qL_n = torch.as_tensor(sim.q_left(tn), dtype=torch.float64)
    qL_np1 = torch.as_tensor(sim.q_left(tnp1), dtype=torch.float64)
    qL_int = torch.as_tensor(sim.q_left_integral(tn, tnp1), dtype=torch.float64)
    T_right_tilde = torch.as_tensor(float(sim.T_right(tnp1)), dtype=torch.float64)

    # The discriminating setup: endpoints miss the interior pulse, the exact
    # integral does not.
    assert float(qL_n.abs().max()) == 0.0
    assert float(qL_np1.abs().max()) == 0.0
    assert float(qL_int.abs().max()) > 0.0

    bc_exact = FullBCData(
        T_right_tilde=T_right_tilde, qL_n=qL_n, qL_np1=qL_np1, qL_int=qL_int
    )
    bc_trap = FullBCData(T_right_tilde=T_right_tilde, qL_n=qL_n, qL_np1=qL_np1)

    exact = full_bc_cn_residual(T_n, T_np1, geom, bc_exact)
    trap = full_bc_cn_residual(T_n, T_np1, geom, bc_trap)

    # Exact integral matches the solver's RHS injection -> left row at the floor.
    assert float(exact["left_neumann"].abs().max()) < 1e-7
    # Trapezoid endpoints drop the interior pulse -> left row badly wrong.
    assert float(trap["left_neumann"].abs().max()) > 1e-3
    # The fix only touches the left row; every other region is byte-identical.
    for name in ("interior", "topbot_adiabatic", "right_dirichlet"):
        assert float((exact[name] - trap[name]).abs().max()) == 0.0


# --- no-op / RNG guard ------------------------------------------------------
#
# The `lambda_physics=0` contract is a TRUE no-op: when the physics path is
# disabled, `train_one_epoch` must NOT draw a physics batch or run the residual
# forward, so the global RNG (and therefore the data stream) is byte-identical to
# a data-only run. These tests gate that the branch is SKIPPED, not merely
# weighted by zero — a "weighted by zero" implementation would still forward the
# model on the physics batch and advance the RNG, which the positive control
# below detects.


class _DropoutModel(torch.nn.Module):
    """Tiny stand-in whose forward consumes the global RNG (via dropout) so an
    extra (erroneous) physics forward is observable in the post-epoch RNG state.
    Output is the input snapshot (channel 0) plus a zero-weighted dropout term —
    deterministic value, but each forward draws from the RNG."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self.drop = torch.nn.Dropout(0.5)

    def forward(self, x_spatial, cond_static, forcing_seq=None):
        base = x_spatial[..., 0:1]
        return base + self.w * self.drop(torch.ones_like(base))


def _data_loss(y_pred, y_batch, iface_x):
    return ((y_pred - y_batch) ** 2).mean()


def _make_batches(n_batches=3, B=4, Nx=6, Ny=6, seed=7):
    rng = np.random.default_rng(seed)
    batches = []
    for _ in range(n_batches):
        spatial = torch.as_tensor(
            rng.standard_normal((B, Nx, Ny, 1)), dtype=torch.float32
        )
        cond_static = torch.as_tensor(
            rng.uniform(0.0, 1.0, (B, 3)), dtype=torch.float32
        )
        Y = torch.as_tensor(
            rng.standard_normal((B, Nx, Ny, 1)), dtype=torch.float32
        )
        batches.append({"spatial": spatial, "cond_static": cond_static, "Y": Y})
    return batches


def _run_epoch(physics_loader, lambda_physics, geom_cfg, epoch_seed=1234):
    from src.operators.train import train_one_epoch

    torch.manual_seed(0)
    model = _DropoutModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    batches = _make_batches()

    torch.manual_seed(epoch_seed)
    metrics = train_one_epoch(
        model=model, train_loader=batches, optimizer=optimizer,
        loss_fn=_data_loss, device=torch.device("cpu"),
        physics_loader=physics_loader, physics_geom_cfg=geom_cfg,
        lambda_data=1.0, lambda_physics=lambda_physics,
    )
    return metrics, torch.get_rng_state()


def test_physics_lambda_zero_is_true_noop():
    """`lambda_physics=0` with a physics loader PRESENT reproduces the data-only
    run exactly — identical metrics and identical post-epoch RNG state — proving
    the physics branch is skipped, not run-then-zeroed."""
    physics_loader = _make_batches(seed=99)  # present but must be untouched
    geom_cfg = {
        "x_grid": np.linspace(0.0, 1.0, 6), "y_grid": np.linspace(0.0, 1.0, 6),
        "k_left": 2.0, "k_right": 1.0, "interface_x": 0.5, "dt": 0.005,
        "sigma_global": 1.0, "rc_range": (0.05, 1.0), "rc_index": 2,
    }

    base_metrics, base_rng = _run_epoch(None, 0.0, None)
    noop_metrics, noop_rng = _run_epoch(physics_loader, 0.0, geom_cfg)

    assert torch.equal(base_rng, noop_rng), "physics branch perturbed the RNG at lambda=0"
    # Wall-clock timing columns are inherently non-deterministic; compare the
    # numeric training metrics only.
    _timing = {"epoch_train_s", "epoch_wall_s"}
    for k in base_metrics:
        if k in _timing:
            continue
        assert base_metrics[k] == noop_metrics[k], f"metric {k} changed at lambda=0"
    assert noop_metrics["physics_loss"] == 0.0


def test_physics_lambda_positive_fires_branch():
    """Positive control: with `lambda_physics>0` the physics forward runs, drawing
    an extra dropout sample, so the post-epoch RNG state MUST differ from the
    no-op run and a finite physics_loss is recorded. Guards against a no-op test
    that would pass even if the branch were dead."""
    physics_loader = _make_batches(seed=99)
    geom_cfg = {
        "x_grid": np.linspace(0.0, 1.0, 6), "y_grid": np.linspace(0.0, 1.0, 6),
        "k_left": 2.0, "k_right": 1.0, "interface_x": 0.5, "dt": 0.005,
        "sigma_global": 1.0, "rc_range": (0.05, 1.0), "rc_index": 2,
    }

    _, base_rng = _run_epoch(None, 0.0, None)
    on_metrics, on_rng = _run_epoch(physics_loader, 0.5, geom_cfg)

    assert not torch.equal(base_rng, on_rng), "enabled physics branch did not advance the RNG"
    assert np.isfinite(on_metrics["physics_loss"])
    assert on_metrics["physics_loss"] > 0.0


# --- W2 collocation sampler: forcing-support hard assertion -----------------
#
# The left-Neumann closure needs q_L(y, t) and q_L(y, t+dt) at the COLLOCATION
# times, so no sampled collocation time may exceed the forcing time support
# (t_final). The sampler must raise rather than silently extrapolate the forcing
# past its support — otherwise the left-row residual becomes silently wrong.


class _StubDS:
    """Minimal `ds` exposing only what `CollocationSampler.sample_batch` touches
    before any item is built: the saved-snapshot index set, the time grid, the
    final time, and the sim ids. The out-of-support filter (`valid_s`) runs first
    and raises, so the heavier item-building machinery is never reached."""

    def __init__(self, t_grid, t_indices, t_final, sim_ids):
        self.t_grid = np.asarray(t_grid, dtype=float)
        self.t_indices = list(t_indices)
        self.t_final = float(t_final)
        self.sim_ids = np.asarray(sim_ids)


def _stub_sampler(max_t_final=0.3, n_saved=31, dt=0.005):
    from src.operators.train import CollocationSampler

    t_grid = np.linspace(0.0, max_t_final, n_saved)
    ds = _StubDS(
        t_grid=t_grid, t_indices=range(n_saved),
        t_final=max_t_final, sim_ids=np.array([0, 1, 2]),
    )
    geom_cfg = {  # unused before the support filter raises
        "x_grid": np.linspace(0.0, 1.0, 6), "y_grid": np.linspace(0.0, 1.0, 6),
        "k_left": 2.0, "k_right": 1.0, "interface_x": 0.5, "dt": dt,
        "sigma_global": 1.0, "T_right_tilde": 0.0,
    }
    return CollocationSampler(
        ds, spec=None, geom_cfg=geom_cfg, batch_size=4, dt=dt, rng_seed=0
    )


def test_collocation_sampler_raises_out_of_support():
    """An OOD max_lead that pushes t + max_lead + dt past t_final for EVERY base
    snapshot must raise (no base supports it), not silently extrapolate q_L."""
    sampler = _stub_sampler(max_t_final=0.3, dt=0.005)
    # 0.0 (earliest base) + 0.5 + 0.005 = 0.505 > 0.3 for every snapshot.
    with pytest.raises(ValueError, match="collocation_lead_max"):
        sampler.sample_batch(max_lead=0.5)


def test_collocation_sampler_support_boundary():
    """Just past the support raises; the filter is the gate. A max_lead whose
    earliest-base reach exceeds t_final (but is < the whole-grid extent) still
    raises because the latest valid base must satisfy t_s + max_lead + dt <=
    t_final, and here even t_s=0 fails."""
    sampler = _stub_sampler(max_t_final=0.3, dt=0.005)
    # t_final - dt = 0.295 is the largest lead any base at t_s=0 could support;
    # anything strictly larger leaves no valid base.
    with pytest.raises(ValueError):
        sampler.sample_batch(max_lead=0.296)


def test_collocation_rank_offset_seed_draws_disjoint_batches(monkeypatch):
    """Under DDP each rank passes ``rng_seed = seed + rank`` (Option 1), so the
    per-rank collocation draws — (sim id, base snapshot, lead) — diverge. Same
    seed reproduces the draw sequence exactly (determinism / no-op guarantee);
    a rank-offset seed yields a different sequence. That decorrelation is what
    lets the DDP-averaged physics gradient cover ``world_size x`` distinct
    collocation samples instead of averaging identical per-rank gradients.

    The heavy item builders are stubbed so the test exercises only the sampler's
    RNG-driven selection (the part that the per-rank seed controls).
    """
    import src.operators.train as train_mod
    from src.operators.train import CollocationSampler

    n_saved = 31
    t_final = 0.3
    t_grid = np.linspace(0.0, t_final, n_saved)

    class _DS:
        def __init__(self):
            self.t_grid = t_grid
            self.t_indices = list(range(n_saved))
            self.t_final = t_final
            self.sim_ids = np.array([0, 1, 2, 3, 4])
            self.ramp_seconds = 0.0
            self.sim_params = {
                i: {
                    "R_c": 0.1,
                    "temporal_family": "sin",
                    "temporal_params": {},
                }
                for i in self.sim_ids
            }
            self.s_y_profiles = {
                i: np.ones(4, dtype=np.float32) for i in self.sim_ids
            }
            self.problem = self

        def build_item(self, ds, sid, s, s2):
            return {"spatial": np.zeros((4, 4, 1), dtype=np.float32)}

    draws = []

    def _fake_build_rollout(base_item, ds, spec, sid, current, t_s, t):
        draws.append((int(sid), round(float(t), 9)))
        return {"x": np.zeros(1, dtype=np.float32)}

    monkeypatch.setattr(
        train_mod, "build_rollout_item_from_base", _fake_build_rollout
    )
    monkeypatch.setattr(
        train_mod, "_q_callable_for_boundary",
        lambda ds, sid, params: (lambda t: 1.0),
    )
    monkeypatch.setattr(
        train_mod, "integrate_temporal_ramped_signed",
        lambda *a, **k: 0.0,
    )
    monkeypatch.setattr(train_mod, "collate_fn", lambda items: items)

    geom_cfg = {
        "x_grid": np.linspace(0.0, 1.0, 4), "y_grid": np.linspace(0.0, 1.0, 4),
        "k_left": 2.0, "k_right": 1.0, "interface_x": 0.5, "dt": 0.005,
        "sigma_global": 1.0, "T_right_tilde": 0.0,
    }

    def run(seed):
        draws.clear()
        sampler = CollocationSampler(
            _DS(), spec=None, geom_cfg=geom_cfg,
            batch_size=4, dt=0.005, rng_seed=seed,
        )
        sampler.sample_batch(max_lead=0.05)
        return list(draws)

    rank0 = run(seed=42)
    rank0_again = run(seed=42)
    rank1 = run(seed=43)

    assert rank0 == rank0_again, "same rng_seed must reproduce the draw sequence"
    assert rank0 != rank1, "rank-offset rng_seed must decorrelate the draws"
