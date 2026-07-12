import numpy as np
import torch
import pytest

from src.physics.fv_solver_2d import FVSolver2D, Layer2D
from src.physics.fv_residual import (
    build_cn_geom,
    build_cn_geom_batched,
    build_cn_geom_per_interface,
    locate_interface,
    full_bc_cn_residual,
    interior_cn_residual,
    FullBCData,
)

"""
Stage 1 physics gates for the per-sample interface geometry
(`build_cn_geom_per_interface` + `locate_interface`) that the `interfaces`
CViT-PINO residual will use. These are correctness gates, not equivalence-only
checks: two of them (uniform equilibrium, exact steady contact-jump) assert the
residual is ~0 on an ANALYTIC solution, so a shared bug in the old and new
geometry cannot pass silently.

Gates:
  1. Geometry equivalence: `build_cn_geom_per_interface` reduces bit-for-bit to
     `build_cn_geom` / `build_cn_geom_batched` when interfaces coincide, and a
     distinct-interface batch matches independent `build_cn_geom` calls.
  2. Uniform equilibrium: constant T_tilde + zero flux -> zero residual in every
     region for arbitrary interface_x / R_c.
  3. Exact steady contact-jump: the analytic constant-flux piecewise-linear field
     with contact jump q_in*R_c is the exact discrete steady state -> zero
     residual in every region (units-precise: normalized T, physical flux).
  4. Discrete consistency: the residual on consecutive REAL solver snapshots at an
     OFF-CENTER interface is at the direct-solve floor for every tested dt.
  5. Truncation convergence: on a manufactured smooth solution the interior
     residual shrinks ~linearly with dt at fixed hx.
"""

K_LEFT = 2.0
K_RIGHT = 1.0
F64 = torch.float64


def _grids(N: int = 100):
    return np.linspace(0.0, 1.0, N), np.linspace(0.0, 1.0, N)


def _build_interface_solver(interface_x: float, R_c: float,
                            dt: float = 0.005, t_final: float = 0.2,
                            Nx: int = 100, Ny: int = 100):
    """Two-slab solver (k=2 left, k=1 right) with the interface at an arbitrary
    off-node `interface_x`, scalar `R_c`. The default windowed-sin left flux is
    active (it exercises the left-Neumann closure); `lam_target` is inert because
    an explicit `dt` is passed."""
    layers = [
        Layer2D(x_left=0.0, x_right=interface_x, rho=1, cp=1, k=K_LEFT),
        Layer2D(x_left=interface_x, x_right=1.0, rho=1, cp=1, k=K_RIGHT),
    ]
    return FVSolver2D(
        a=0.0, b=1.0, c=0.0, d=1.0,
        Nx=Nx, Ny=Ny,
        layers=layers,
        interface_R=[R_c],
        lam_target=0.5,
        dt=dt,
        flux_f=2.0, flux_A=50.0,
        t_on=0.0, t_off=0.2,
        t_final=t_final, phase=0.0,
    )


def _full_bc_data_for_step(sim, n: int, dtype=F64) -> FullBCData:
    """`FullBCData` for the consecutive pair (n -> n+1) from the solver's own
    `q_left`/`T_right` (trapezoid left-Neumann path; raw fields => sigma=1)."""
    tn = float(sim.t[n])
    tnp1 = float(sim.t[n + 1])
    qL_n = torch.as_tensor(sim.q_left(tn), dtype=dtype)
    qL_np1 = torch.as_tensor(sim.q_left(tnp1), dtype=dtype)
    T_right_tilde = torch.as_tensor(float(sim.T_right(tnp1)), dtype=dtype)
    return FullBCData(T_right_tilde=T_right_tilde, qL_n=qL_n, qL_np1=qL_np1)


# --- 1. Geometry equivalence -----------------------------------------------


def test_locate_interface_exact_and_discrete_fields():
    xg, _ = _grids(100)
    hx = float(xg[1] - xg[0])
    ix = 0.345
    loc = locate_interface(xg, ix)
    assert loc.interface_x_exact == ix
    assert loc.face_idx == int(np.floor(ix / hx))
    # Sub-cell widths sum to hx and place x_Gamma between the flanking nodes.
    assert loc.h_left > 0 and loc.h_right > 0
    np.testing.assert_allclose(loc.h_left + loc.h_right, hx, atol=1e-12)
    left_node = xg[loc.face_idx]
    assert left_node < ix < xg[loc.face_idx + 1]
    # Masks agree (interface is off-node) and split at the face.
    assert loc.left_node_mask[loc.face_idx] and not loc.left_node_mask[loc.face_idx + 1]
    assert bool(np.all(loc.left_node_mask == loc.left_cell_mask))


def test_locate_interface_rejects_out_of_band():
    xg, _ = _grids(100)
    hx = float(xg[1] - xg[0])
    # Interface in the boundary-adjacent face slots is invalid (matches the solver
    # rule 1 <= face_idx <= Nx-3): face slot 0 leaves no left cell, the last face
    # slot no right cell.
    with pytest.raises(ValueError):
        locate_interface(xg, xg[0] + 0.4 * hx)   # face slot 0 (face_idx=0)
    with pytest.raises(ValueError):
        locate_interface(xg, xg[-1] - 0.4 * hx)  # last face slot (face_idx=Nx-2)


def test_geom_per_interface_matches_single_when_coincident():
    xg, yg = _grids(100)
    ix, rc, dt = 0.42, 0.4, 0.005
    single = build_cn_geom(xg, yg, K_LEFT, K_RIGHT, ix, rc, dt, sigma_global=3.0)
    B = 4
    per = build_cn_geom_per_interface(
        xg, yg, K_LEFT, K_RIGHT, [ix] * B, [rc] * B, dt, sigma_global=3.0
    )
    for b in range(B):
        assert torch.equal(per.G_x[b], single.G_x)
        assert torch.equal(per.G_y[b], single.G_y)
        assert torch.equal(per.dx[b], single.dx)
        for name in ("r_w", "r_e", "r_s", "r_n"):
            assert torch.equal(getattr(per, name)[b], getattr(single, name)), name
    assert torch.equal(per.dy, single.dy)
    assert torch.equal(per.rho_cp, single.rho_cp)
    assert torch.equal(per.interior_mask, single.interior_mask)
    assert per.hx == single.hx and per.dt == single.dt
    assert per.sigma_global == single.sigma_global


def test_geom_per_interface_matches_batched_fixed_interface():
    """At a FIXED interface with varying R_c, the general per-interface builder
    reproduces the R_c-only `build_cn_geom_batched` bit-for-bit."""
    xg, yg = _grids(100)
    ix, dt = 0.5, 0.005
    rcs = [0.1, 0.5, 1.0]
    batched = build_cn_geom_batched(xg, yg, K_LEFT, K_RIGHT, ix, rcs, dt)
    per = build_cn_geom_per_interface(
        xg, yg, K_LEFT, K_RIGHT, [ix] * len(rcs), rcs, dt
    )
    for b in range(len(rcs)):
        assert torch.equal(per.G_x[b], batched.G_x[b])
        assert torch.equal(per.G_y[b], batched.G_y[b])
        for name in ("r_w", "r_e", "r_s", "r_n"):
            assert torch.equal(getattr(per, name)[b], getattr(batched, name)[b]), name
    # dx is R_c-independent and 1-D in the batched builder; broadcast-compare.
    for b in range(len(rcs)):
        assert torch.equal(per.dx[b], batched.dx)


def test_geom_per_interface_distinct_matches_separate_calls():
    """A two-sim batch with DISTINCT interfaces matches two independent
    single-sim `build_cn_geom` calls on every geometry tensor."""
    xg, yg = _grids(100)
    dt = 0.005
    ifaces = [0.345, 0.62]
    rcs = [0.3, 0.8]
    per = build_cn_geom_per_interface(xg, yg, K_LEFT, K_RIGHT, ifaces, rcs, dt)
    for b, (ix, rc) in enumerate(zip(ifaces, rcs)):
        single = build_cn_geom(xg, yg, K_LEFT, K_RIGHT, ix, rc, dt)
        assert torch.equal(per.G_x[b], single.G_x)
        assert torch.equal(per.G_y[b], single.G_y)
        assert torch.equal(per.dx[b], single.dx)
        for name in ("r_w", "r_e", "r_s", "r_n"):
            assert torch.equal(getattr(per, name)[b], getattr(single, name)), name


def test_geom_per_interface_batch_permutation_equivariant():
    """Permuting the batch permutes the geometry identically (no cross-sim
    coupling in the builder)."""
    xg, yg = _grids(100)
    dt = 0.005
    ifaces = [0.345, 0.5, 0.62]
    rcs = [0.3, 0.7, 0.9]
    per = build_cn_geom_per_interface(xg, yg, K_LEFT, K_RIGHT, ifaces, rcs, dt)
    perm = [2, 0, 1]
    per_p = build_cn_geom_per_interface(
        xg, yg, K_LEFT, K_RIGHT, [ifaces[i] for i in perm], [rcs[i] for i in perm], dt
    )
    for out_row, in_row in enumerate(perm):
        for name in ("G_x", "G_y", "dx", "r_w", "r_e", "r_s", "r_n"):
            assert torch.equal(getattr(per_p, name)[out_row],
                               getattr(per, name)[in_row]), name


# --- 2. Uniform equilibrium ------------------------------------------------


@pytest.mark.parametrize("ix,rc", [(0.345, 0.2), (0.5, 0.7), (0.62, 1.0)])
def test_uniform_equilibrium_zero_residual(ix, rc):
    xg, yg = _grids(100)
    Nx = Ny = 100
    dt = 0.005
    geom = build_cn_geom_per_interface(
        xg, yg, K_LEFT, K_RIGHT, [ix], [rc], dt, sigma_global=1.0
    )
    c = 0.37  # arbitrary constant normalized temperature
    T = torch.full((1, Nx, Ny), c, dtype=F64)
    bc = FullBCData(
        T_right_tilde=torch.tensor(c, dtype=F64),
        qL_n=torch.zeros(Ny, dtype=F64),
        qL_np1=torch.zeros(Ny, dtype=F64),
        qL_int=torch.zeros(Ny, dtype=F64),
    )
    parts = full_bc_cn_residual(T, T, geom, bc)
    for name, vec in parts.items():
        assert float(vec.abs().max()) < 1e-12, f"{name}: {float(vec.abs().max()):.3e}"


# --- 3. Exact steady contact-jump ------------------------------------------


def _steady_jump_field(xg, Nx, Ny, ix, rc, q_in, kL, kR, T_wall):
    """Analytic constant-flux steady field for inward left flux `q_in`.

    Convention (repository): the left wall injects `q_in` inward,
    `-k dT/dx|_{x=0} = q_in`, so steady conduction has slope `-q_in/k` in each
    material and a contact jump `T(Gamma^-) - T(Gamma^+) = q_in * R_c`. Returns a
    `(Nx, Ny)` field uniform in y.
    """
    xt = torch.as_tensor(xg, dtype=F64)
    T_gamma_plus = T_wall + (q_in / kR) * (1.0 - ix)
    T_gamma_minus = T_gamma_plus + q_in * rc
    Tx = torch.where(
        xt <= ix,
        T_gamma_minus + (q_in / kL) * (ix - xt),
        T_wall + (q_in / kR) * (1.0 - xt),
    )
    return Tx[:, None].expand(Nx, Ny).contiguous()


@pytest.mark.parametrize(
    "ix,rc,q_in,sigma",
    [(0.345, 0.3, 20.0, 1.0), (0.5, 0.7, 35.0, 40.0), (0.62, 1.0, 50.0, 25.0)],
)
def test_steady_contact_jump_zero_residual(ix, rc, q_in, sigma):
    xg, yg = _grids(100)
    Nx = Ny = 100
    dt = 0.005
    mu, T_wall = 290.0, 300.0
    T_phys = _steady_jump_field(xg, Nx, Ny, ix, rc, q_in, K_LEFT, K_RIGHT, T_wall)
    T_tilde = ((T_phys - mu) / sigma)[None]  # (1, Nx, Ny)

    geom = build_cn_geom_per_interface(
        xg, yg, K_LEFT, K_RIGHT, [ix], [rc], dt, sigma_global=sigma
    )
    bc = FullBCData(
        T_right_tilde=torch.tensor((T_wall - mu) / sigma, dtype=F64),
        qL_n=torch.full((Ny,), q_in, dtype=F64),
        qL_np1=torch.full((Ny,), q_in, dtype=F64),
        qL_int=torch.full((Ny,), q_in * dt, dtype=F64),  # exact step integral
    )
    parts = full_bc_cn_residual(T_tilde, T_tilde, geom, bc)
    for name, vec in parts.items():
        assert float(vec.abs().max()) < 1e-8, f"{name}: {float(vec.abs().max()):.3e}"


def test_steady_jump_units_contract_requires_sigma():
    """Negative control on the normalized-T / physical-flux contract: if the
    field is normalized by sigma but the geometry's `sigma_global` is left at 1,
    the physical left flux is mis-scaled and ONLY the left-Neumann row blows up;
    the interior/adiabatic/Dirichlet rows (no flux term) stay at the floor."""
    xg, yg = _grids(100)
    Nx = Ny = 100
    dt = 0.005
    ix, rc, q_in, sigma = 0.5, 0.7, 35.0, 40.0
    mu, T_wall = 290.0, 300.0
    T_phys = _steady_jump_field(xg, Nx, Ny, ix, rc, q_in, K_LEFT, K_RIGHT, T_wall)
    T_tilde = ((T_phys - mu) / sigma)[None]

    geom_wrong = build_cn_geom_per_interface(
        xg, yg, K_LEFT, K_RIGHT, [ix], [rc], dt, sigma_global=1.0
    )
    bc = FullBCData(
        T_right_tilde=torch.tensor((T_wall - mu) / sigma, dtype=F64),
        qL_n=torch.full((Ny,), q_in, dtype=F64),
        qL_np1=torch.full((Ny,), q_in, dtype=F64),
        qL_int=torch.full((Ny,), q_in * dt, dtype=F64),
    )
    parts = full_bc_cn_residual(T_tilde, T_tilde, geom_wrong, bc)
    assert float(parts["left_neumann"].abs().max()) > 1e-2
    assert float(parts["interior"].abs().max()) < 1e-8
    assert float(parts["right_dirichlet"].abs().max()) < 1e-8
    assert float(parts["topbot_adiabatic"].abs().max()) < 1e-8


# --- 4. Discrete consistency (every dt) ------------------------------------


@pytest.mark.parametrize("dt", [0.002, 0.005, 0.01])
def test_discrete_consistency_offcenter_interface(dt):
    """The per-interface residual on consecutive REAL solver snapshots at an
    off-center interface is at the direct-solve floor for every tested dt. This
    validates the whole batched geometry against the actual FV solver at an
    interface the existing 0.5-only tests never exercise."""
    ix, rc = 0.345, 0.4
    sim = _build_interface_solver(ix, rc, dt=dt, t_final=0.2, Nx=100, Ny=100)
    T0 = np.full((sim.Nx, sim.Ny), sim.T_right(0.0), dtype=np.float64)
    _, _, _, T_hist = sim.solve(T0=T0, store_trajectory=True)

    geom = build_cn_geom_per_interface(
        sim.grid_x, sim.grid_y, K_LEFT, K_RIGHT, [ix], [rc], sim.dt,
        sigma_global=1.0,
    )
    # Sample steps spread across the trajectory; the count of snapshots depends on
    # dt (t_final/dt + 1), so derive valid indices instead of hard-coding them.
    nsteps = len(T_hist) - 1
    steps = sorted({1, nsteps // 4, nsteps // 2, nsteps - 1})
    region_max = {}
    for n in steps:
        T_n = torch.as_tensor(T_hist[n], dtype=F64)
        T_np1 = torch.as_tensor(T_hist[n + 1], dtype=F64)
        bc = _full_bc_data_for_step(sim, n)
        parts = full_bc_cn_residual(T_n, T_np1, geom, bc)
        for name, vec in parts.items():
            region_max[name] = max(region_max.get(name, 0.0),
                                   float(vec.abs().max()))
    for name, m in region_max.items():
        assert m < 1e-7, f"{name} floor too high at dt={dt}: {m:.3e}"


# --- 5. Truncation convergence (MMS) ---------------------------------------


def _mms_field(xg, yg, t):
    """T = exp(-2 pi^2 t) cos(pi x) cos(pi y): satisfies dT/dt = lap(T) with
    alpha=1, so its discrete CN residual is pure truncation error."""
    X, Y = np.meshgrid(xg, yg, indexing="ij")
    return np.exp(-2.0 * np.pi**2 * t) * np.cos(np.pi * X) * np.cos(np.pi * Y)


def test_truncation_interior_residual_third_order_in_dt():
    """On a homogeneous slab (k_left=k_right=1, R_c=0) built through the
    per-interface geometry, the interior CN residual of a smooth manufactured
    solution shrinks with dt at the Crank-Nicolson local-truncation rate O(dt^3)
    at fixed hx: plugging the exact continuous solution into the discrete equation
    leaves R/T ~ (lam*dt)^3/12 per step (the O(dt^2) terms cancel by CN symmetry).
    This confirms the batched geometry yields a genuinely consistent discretization
    (residual -> 0 with refinement), not merely one that equals the old routine.

    Measured on the temporal-truncation-dominated regime (coarse dt); at very small
    dt the fixed-hx spatial truncation term (~dt*hx^2) and the temporal term cross
    and cancel, so the finest steps are excluded from the order estimate.
    """
    xg, yg = _grids(64)
    ix = 0.5  # off-node on this grid; homogeneous material => no real jump
    t0 = 0.05
    dts = [0.04, 0.02, 0.01]

    res_rms = []
    for dt in dts:
        geom = build_cn_geom_per_interface(
            xg, yg, 1.0, 1.0, [ix], [0.0], dt, sigma_global=1.0
        )
        T_n = torch.as_tensor(_mms_field(xg, yg, t0), dtype=F64)
        T_np1 = torch.as_tensor(_mms_field(xg, yg, t0 + dt), dtype=F64)
        res, mask = interior_cn_residual(T_n, T_np1, geom)
        n_int = float(mask.sum())
        res_rms.append(float((res.pow(2).sum() / n_int).sqrt()))

    # Strictly decreasing with dt.
    for a, b in zip(res_rms, res_rms[1:]):
        assert b < a, f"residual not decreasing with dt: {res_rms}"
    # Third order: halving dt reduces the residual ~8x (nominal 2^3). Band allows
    # for the sub-leading spatial term without admitting first/second order.
    for i in range(len(dts) - 1):
        ratio = res_rms[i] / res_rms[i + 1]
        assert 5.0 < ratio < 11.0, f"not O(dt^3): ratio={ratio:.3f} ({res_rms})"
