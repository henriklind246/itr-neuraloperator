import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional

"""
Differentiable torch mirror of the Crank-Nicolson FV solver's discrete cell
balance (`src/physics/fv_solver_2d.py`), used as a physics-informed residual
loss for the forward FNO surrogate.

Stage 1 scope: interior conduction only. We reproduce the solver's face
conductances (`fv_solver_2d.py:390-433`) and CN coupling coefficients
(`fv_solver_2d.py:439-485`), then evaluate the interior CN cell-balance residual

    res_ij = (T^{n+1} - T^n)_ij - ( Lap(T^{n+1})_ij + Lap(T^n)_ij )

where  Lap(T)_ij = rw(T_W-T_C) + re(T_E-T_C) + rs(T_S-T_C) + rn(T_N-T_C)
and the r-coefficients already fold in dt, the CN 1/2, and the cell capacity
(`r_w = dt*G_x[i-1,j] / (2*rho_cp*dx[i])`, etc.). Because every `T` appears only
in differences, the interior residual is level-invariant: evaluating it on
globally-normalized `T_tilde = (T - mu)/sigma` gives exactly `res(T_phys)/sigma`
(`mu_global` cancels). The r-coefficients are dimensionless, so `res_ij` is the
nondimensional per-step (normalized) temperature residual directly — no separate
capacity-scale division is needed for the interior path.

The interface / contact-resistance physics enters through the interface x-face
conductance `G_x = 1/(h_L/k_L + R_c[j] + h_R/k_R)`, so the residual encodes the
temperature-jump condition with no jump-specific term.

`full_bc_cn_residual` (Stage 2) extends the cell balance to every cell and adds
the solver's three boundary closures (`fv_solver_2d.py:671-682`): the right
Dirichlet column (`T^{n+1}_{Nx-1} = T_right`), the top/bottom adiabatic rows
(zero external face flux, which the r-coefficients already encode), and the left
time-dependent Neumann forcing (CN-averaged `0.5*(q_L(t)+q_L(t+dt))`). Every
region's residual is a normalized per-step temperature error (the balance rows
inherit the capacity folded into the r-coefficients; the algebraic Dirichlet row
is a normalized-temperature mismatch), so the four regions are commensurate and
can be summed/weighted directly. The optional volumetric `source_term` (Stage 3)
is intentionally not implemented here.
"""


@dataclass
class FVGeom:
    """Precomputed, batch-independent CN geometry for the residual.

    All tensors live on a single device/dtype. The `r_*` coefficients mirror
    `FVSolver2D._build_local_cn_coefficients`. `interior_mask` marks the cells
    where all four faces exist and no boundary closure applies
    (i in [1, Nx-2], j in [1, Ny-2]).
    """
    G_x: torch.Tensor          # (Nx-1, Ny)
    G_y: torch.Tensor          # (Nx, Ny-1)
    dx: torch.Tensor           # (Nx,)
    dy: torch.Tensor           # (Ny,)
    rho_cp: torch.Tensor       # (Nx, Ny)
    r_w: torch.Tensor          # (Nx, Ny)
    r_e: torch.Tensor          # (Nx, Ny)
    r_s: torch.Tensor          # (Nx, Ny)
    r_n: torch.Tensor          # (Nx, Ny)
    interior_mask: torch.Tensor  # (Nx, Ny) bool
    dt: float
    sigma_global: float
    hx: float                    # base grid spacing (left-Neumann flux uses h=hx)

    @property
    def Nx(self) -> int:
        return self.r_w.shape[-2]

    @property
    def Ny(self) -> int:
        return self.r_w.shape[-1]


def _node_k(x_grid: np.ndarray, k_left: float, k_right: float,
            interface_x: float) -> np.ndarray:
    """Per-node conductivity for a single vertical interface (mirrors
    `FVSolver2D._build_node_layer_indices_1d` + `k_nodes`). Nodes never lie on
    the interface (it is a face), so a strict comparison is unambiguous."""
    return np.where(x_grid < interface_x, float(k_left), float(k_right))


def build_face_conductances(x_grid, y_grid, k_left: float, k_right: float,
                            interface_x: float, R_c,
                            device=None, dtype=torch.float64):
    """Mirror the solver's `G_x` / `G_y` assembly for a single x-interface.

    Parameters mirror the two-slab `forcing` geometry: `k_left`/`k_right` are the
    layer conductivities, `interface_x` the interface location (must fall on an
    x-face), and `R_c` the contact resistance — scalar (uniform) or `(Ny,)` array
    (`R_c(y)`), matching `FVSolver2D.interface_R[0]`.

    Returns `(G_x (Nx-1,Ny), G_y (Nx,Ny-1), dx (Nx,), dy (Ny,))`.
    """
    x_grid = np.asarray(x_grid, dtype=float)
    y_grid = np.asarray(y_grid, dtype=float)
    Nx = x_grid.shape[0]
    Ny = y_grid.shape[0]
    a = float(x_grid[0])
    hx = float(x_grid[1] - x_grid[0])

    # Interface face slot + off-midpoint distances (mirror
    # `_validate_and_index_interfaces`).
    face_idx = int(np.floor((interface_x - a) / hx))
    x_left_node = a + face_idx * hx
    x_right_node = a + (face_idx + 1) * hx
    h_L = interface_x - x_left_node
    h_R = x_right_node - interface_x

    Rc = np.asarray(R_c, dtype=float)  # scalar -> 0-d; profile -> (Ny,)

    # --- G_x (Nx-1, Ny) ---
    G_x = np.zeros((Nx - 1, Ny), dtype=float)
    face_pos = a + (np.arange(Nx - 1) + 0.5) * hx
    for i in range(Nx - 1):
        if i == face_idx:
            G_x[i, :] = 1.0 / (h_L / k_left + Rc + h_R / k_right)
        else:
            k_face = k_left if face_pos[i] < interface_x else k_right
            G_x[i, :] = k_face / hx

    # --- G_y (Nx, Ny-1): k of the node's layer / hx ---
    k_nodes = _node_k(x_grid, k_left, k_right, interface_x)
    G_y = np.repeat((k_nodes / hx)[:, None], Ny - 1, axis=1)

    # --- Control-volume widths (mirror the dx/dy construction) ---
    dx = np.full(Nx, hx)
    dx[0] = hx / 2.0
    dx[Nx - 1] = hx / 2.0
    dx[face_idx] = 0.5 * hx + h_L
    dx[face_idx + 1] = h_R + 0.5 * hx

    dy = np.full(Ny, hx)
    dy[0] = hx / 2.0
    dy[Ny - 1] = hx / 2.0

    t = lambda arr: torch.as_tensor(arr, device=device, dtype=dtype)
    return t(G_x), t(G_y), t(dx), t(dy)


def build_cn_geom(x_grid, y_grid, k_left: float, k_right: float,
                  interface_x: float, R_c, dt: float,
                  sigma_global: float = 1.0,
                  rho: float = 1.0, cp: float = 1.0,
                  device=None, dtype=torch.float64) -> FVGeom:
    """Assemble the CN coupling coefficients (`r_w/r_e/r_s/r_n`) and interior
    mask from the face conductances. Mirrors
    `FVSolver2D._build_local_cn_coefficients`: coefficients are zero where the
    corresponding face does not exist and on the Dirichlet row `i = Nx-1`."""
    G_x, G_y, dx, dy = build_face_conductances(
        x_grid, y_grid, k_left, k_right, interface_x, R_c,
        device=device, dtype=dtype,
    )
    Nx = dx.shape[0]
    Ny = dy.shape[0]
    rho_cp = torch.full((Nx, Ny), float(rho) * float(cp),
                        device=device, dtype=dtype)

    r_w = torch.zeros((Nx, Ny), device=device, dtype=dtype)
    r_e = torch.zeros((Nx, Ny), device=device, dtype=dtype)
    r_s = torch.zeros((Nx, Ny), device=device, dtype=dtype)
    r_n = torch.zeros((Nx, Ny), device=device, dtype=dtype)

    dxi = dx[:, None]          # (Nx,1)
    dyj = dy[None, :]          # (1,Ny)
    two_rc = 2.0 * rho_cp

    # Active rows i = 0..Nx-2 (Dirichlet row i=Nx-1 stays zero).
    # West face exists for i >= 1: G_x[i-1, j].
    r_w[1:Nx - 1, :] = dt * G_x[0:Nx - 2, :] / (two_rc[1:Nx - 1, :] * dxi[1:Nx - 1, :])
    # East face exists for i <= Nx-2: G_x[i, j].
    r_e[0:Nx - 1, :] = dt * G_x[0:Nx - 1, :] / (two_rc[0:Nx - 1, :] * dxi[0:Nx - 1, :])
    # South face exists for j >= 1: G_y[i, j-1].
    r_s[0:Nx - 1, 1:Ny] = dt * G_y[0:Nx - 1, 0:Ny - 1] / (two_rc[0:Nx - 1, 1:Ny] * dyj[:, 1:Ny])
    # North face exists for j <= Ny-2: G_y[i, j].
    r_n[0:Nx - 1, 0:Ny - 1] = dt * G_y[0:Nx - 1, 0:Ny - 1] / (two_rc[0:Nx - 1, 0:Ny - 1] * dyj[:, 0:Ny - 1])

    interior_mask = torch.zeros((Nx, Ny), dtype=torch.bool, device=device)
    interior_mask[1:Nx - 1, 1:Ny - 1] = True

    hx = float(np.asarray(x_grid, dtype=float)[1] - np.asarray(x_grid, dtype=float)[0])
    return FVGeom(
        G_x=G_x, G_y=G_y, dx=dx, dy=dy, rho_cp=rho_cp,
        r_w=r_w, r_e=r_e, r_s=r_s, r_n=r_n,
        interior_mask=interior_mask, dt=float(dt),
        sigma_global=float(sigma_global), hx=hx,
    )


def build_cn_geom_batched(x_grid, y_grid, k_left: float, k_right: float,
                          interface_x: float, R_c_batch, dt: float,
                          sigma_global: float = 1.0,
                          rho: float = 1.0, cp: float = 1.0,
                          device=None, dtype=torch.float64) -> FVGeom:
    """Per-sample CN geometry for a batch of scalar contact resistances
    `R_c_batch` (shape `(B,)`). Only the interface x-face conductance depends on
    `R_c`, so the R_c-independent faces are built once and the interface column is
    set per sample. Returns an `FVGeom` whose `r_w/r_e/r_s/r_n` are `(B, Nx, Ny)`;
    `interior_mask` stays 2D (the residual broadcasts it over the batch).

    This is the second caller of the CN assembly (W1 physics residual, where R_c
    varies per sim and the interface conductance spans ~17x over R_c in [0.05, 1]).
    It reduces to `build_cn_geom` exactly at `B = 1`.
    """
    R_c_t = torch.as_tensor(R_c_batch, device=device, dtype=dtype).reshape(-1)
    B = R_c_t.shape[0]

    # R_c-independent faces (R_c=0 placeholder; interface column overridden below).
    G_x0, G_y0, dx, dy = build_face_conductances(
        x_grid, y_grid, k_left, k_right, interface_x, 0.0,
        device=device, dtype=dtype,
    )
    Nx = dx.shape[0]
    Ny = dy.shape[0]

    xg = np.asarray(x_grid, dtype=float)
    x0 = float(xg[0])
    hx = float(xg[1] - xg[0])
    face_idx = int(np.floor((interface_x - x0) / hx))
    h_L = interface_x - (x0 + face_idx * hx)
    h_R = (x0 + (face_idx + 1) * hx) - interface_x

    G_x = G_x0.unsqueeze(0).expand(B, Nx - 1, Ny).clone()
    G_x[:, face_idx, :] = (1.0 / (h_L / k_left + R_c_t + h_R / k_right))[:, None]
    G_y = G_y0.unsqueeze(0).expand(B, Nx, Ny - 1)

    rho_cp_val = float(rho) * float(cp)
    r_w = torch.zeros((B, Nx, Ny), device=device, dtype=dtype)
    r_e = torch.zeros((B, Nx, Ny), device=device, dtype=dtype)
    r_s = torch.zeros((B, Nx, Ny), device=device, dtype=dtype)
    r_n = torch.zeros((B, Nx, Ny), device=device, dtype=dtype)

    dxi = dx[None, :, None]    # (1,Nx,1)
    dyj = dy[None, None, :]    # (1,1,Ny)
    two_rc = 2.0 * rho_cp_val

    r_w[:, 1:Nx - 1, :] = dt * G_x[:, 0:Nx - 2, :] / (two_rc * dxi[:, 1:Nx - 1, :])
    r_e[:, 0:Nx - 1, :] = dt * G_x[:, 0:Nx - 1, :] / (two_rc * dxi[:, 0:Nx - 1, :])
    r_s[:, 0:Nx - 1, 1:Ny] = dt * G_y[:, 0:Nx - 1, 0:Ny - 1] / (two_rc * dyj[:, :, 1:Ny])
    r_n[:, 0:Nx - 1, 0:Ny - 1] = dt * G_y[:, 0:Nx - 1, 0:Ny - 1] / (two_rc * dyj[:, :, 0:Ny - 1])

    interior_mask = torch.zeros((Nx, Ny), dtype=torch.bool, device=device)
    interior_mask[1:Nx - 1, 1:Ny - 1] = True

    rho_cp = torch.full((Nx, Ny), rho_cp_val, device=device, dtype=dtype)
    return FVGeom(
        G_x=G_x, G_y=G_y, dx=dx, dy=dy, rho_cp=rho_cp,
        r_w=r_w, r_e=r_e, r_s=r_s, r_n=r_n,
        interior_mask=interior_mask, dt=float(dt),
        sigma_global=float(sigma_global), hx=float(hx),
    )


def _cn_laplacian(T: torch.Tensor, geom: FVGeom) -> torch.Tensor:
    """CN flux-balance operator on the interior, returning a (B, Nx-2, Ny-2)
    tensor. `Lap(T)_ij = rw(T_W-T_C)+re(T_E-T_C)+rs(T_S-T_C)+rn(T_N-T_C)` with the
    solver's r-coefficients (dt, 1/2, capacity already folded in)."""
    Tc = T[:, 1:-1, 1:-1]
    Tw = T[:, 0:-2, 1:-1]
    Te = T[:, 2:, 1:-1]
    Ts = T[:, 1:-1, 0:-2]
    Tn = T[:, 1:-1, 2:]

    rw = geom.r_w[..., 1:-1, 1:-1]
    re = geom.r_e[..., 1:-1, 1:-1]
    rs = geom.r_s[..., 1:-1, 1:-1]
    rn = geom.r_n[..., 1:-1, 1:-1]

    return (rw * (Tw - Tc) + re * (Te - Tc)
            + rs * (Ts - Tc) + rn * (Tn - Tc))


def interior_cn_residual(T_n: torch.Tensor, T_np1: torch.Tensor,
                         geom: FVGeom, extra_mask: torch.Tensor | None = None):
    """Interior CN cell-balance residual.

    `T_n`, `T_np1` are normalized temperature fields one solver step `dt` apart,
    shape `(B, Nx, Ny)` (a missing batch dim is added). Returns
    `(residual (B, Nx, Ny), mask (Nx, Ny) bool)` with the residual zero outside
    the interior mask. `extra_mask` (broadcastable to `(Nx, Ny)`, True = keep)
    further removes cells (e.g. an active source patch in Phase A).

    The residual is `res = (T_np1 - T_n) - (Lap(T_np1) + Lap(T_n))` on interior
    cells, evaluated in normalized `T` (so it equals the physical per-step
    residual divided by `sigma_global`).
    """
    if T_n.dim() == 2:
        T_n = T_n[None]
        T_np1 = T_np1[None]
    if T_n.shape != T_np1.shape:
        raise ValueError(
            f"T_n shape {tuple(T_n.shape)} != T_np1 shape {tuple(T_np1.shape)}"
        )

    B = T_n.shape[0]
    res = torch.zeros_like(T_n)
    dT = T_np1[:, 1:-1, 1:-1] - T_n[:, 1:-1, 1:-1]
    res[:, 1:-1, 1:-1] = dT - (_cn_laplacian(T_np1, geom)
                               + _cn_laplacian(T_n, geom))

    mask = geom.interior_mask
    if extra_mask is not None:
        mask = mask & extra_mask.to(mask.device, torch.bool)
    res = res * mask[None].to(res.dtype)
    return res, mask


@dataclass
class FullBCData:
    """Boundary data for `full_bc_cn_residual`, all in the SAME space as the
    fields passed to the residual (normalized when the model outputs are
    normalized).

    `T_right_tilde` is the (normalized) right-edge Dirichlet target
    `(T_right - mu_global) / sigma_global`; scalar or broadcastable to `(B, Ny)`.
    `qL_n` / `qL_np1` are the PHYSICAL left-edge fluxes `q_L(y, t_n)` and
    `q_L(y, t_n + dt)`, broadcastable to `(B, Ny)`; the residual divides them by
    `sigma_global` to land in normalized-temperature units.

    `qL_int`, when provided, is the PHYSICAL exact time integral of the left flux
    over the step `int_{t_n}^{t_n+dt} q_L(y, t) dt`, broadcastable to `(B, Ny)`.
    The solver injects this exact integral (not the trapezoid endpoint average)
    into its RHS, so when it is available the residual uses it directly and the
    `qL_n`/`qL_np1` endpoints are ignored for the left-Neumann row. When `None`
    the residual falls back to the CN trapezoid `dt*(qL_n+qL_np1)/2`.
    """
    T_right_tilde: torch.Tensor
    qL_n: torch.Tensor
    qL_np1: torch.Tensor
    qL_int: Optional[torch.Tensor] = None


def _cn_laplacian_full(T: torch.Tensor, geom: FVGeom) -> torch.Tensor:
    """Full-grid CN flux-balance operator `(B, Nx, Ny)`. Identical form to
    `_cn_laplacian` but over every cell: at boundary cells the missing-face
    r-coefficient is zero (mirroring `_build_local_cn_coefficients`), so the
    out-of-domain neighbor placeholder never contributes. The Dirichlet row
    `i = Nx-1` has all r-coefficients zero and is handled algebraically by
    `full_bc_cn_residual`, not by this operator."""
    Tw = torch.zeros_like(T)
    Tw[:, 1:, :] = T[:, :-1, :]
    Te = torch.zeros_like(T)
    Te[:, :-1, :] = T[:, 1:, :]
    Ts = torch.zeros_like(T)
    Ts[:, :, 1:] = T[:, :, :-1]
    Tn = torch.zeros_like(T)
    Tn[:, :, :-1] = T[:, :, 1:]

    return (geom.r_w * (Tw - T) + geom.r_e * (Te - T)
            + geom.r_s * (Ts - T) + geom.r_n * (Tn - T))


def full_bc_cn_residual(T_n: torch.Tensor, T_np1: torch.Tensor,
                        geom: FVGeom, bc: FullBCData,
                        dirichlet_both_ends: bool = False) -> dict:
    """Full-boundary CN residual, partitioned into the solver's four regions.

    `T_n`, `T_np1` are normalized temperature fields one solver step `dt` apart,
    shape `(B, Nx, Ny)` (a missing batch dim is added). Returns a dict of 1-D
    residual vectors (flattened over batch and the region's cells), each a
    normalized per-step temperature error:

      - ``interior``          : CN cell balance, `i in [1,Nx-2], j in [1,Ny-2]`.
      - ``topbot_adiabatic``  : CN cell balance on rows `j in {0,Ny-1}` for
                                `i in [1,Nx-2]` (zero top/bottom face flux baked
                                into the r-coefficients).
      - ``left_neumann``      : CN cell balance on row `i=0` minus the normalized
                                left flux. With `bc.qL_int` set this is the exact
                                step integral `2*qL_int / (rho_cp_left * hx *
                                sigma_global)` (the solver's exact path); without
                                it, the CN trapezoid `dt*(qL_n+qL_np1) /
                                (rho_cp_left * hx * sigma_global)`.
      - ``right_dirichlet``   : algebraic `T_np1[Nx-1,:] - T_right_tilde`.

    The cell-balance regions reuse the solver's r-coefficients (dt, CN 1/2, and
    cell capacity folded in), so they are nondimensional per-step temperature
    residuals; the Dirichlet row is a normalized-temperature mismatch on the same
    footing. With ``dirichlet_both_ends`` the dict also carries
    ``right_dirichlet_n`` = `T_n[Nx-1,:] - T_right_tilde` so the W2 collocation
    loss can anchor the right edge at BOTH model-output times (in W1 `T_n` is
    truth, so only the `T_np1` end is needed).
    """
    if T_n.dim() == 2:
        T_n = T_n[None]
        T_np1 = T_np1[None]
    if T_n.shape != T_np1.shape:
        raise ValueError(
            f"T_n shape {tuple(T_n.shape)} != T_np1 shape {tuple(T_np1.shape)}"
        )

    Nx, Ny = geom.Nx, geom.Ny

    # CN cell balance on every active row (i = 0..Nx-2). The Dirichlet row
    # i = Nx-1 carries zero r-coefficients here; it is overwritten algebraically.
    bal = ((T_np1 - T_n)
           - (_cn_laplacian_full(T_np1, geom) + _cn_laplacian_full(T_n, geom)))

    # Left Neumann row i=0: subtract the normalized left-flux forcing, mirroring
    # the solver's RHS injection (`fv_solver_2d.py` left-Neumann block). When the
    # exact step integral is supplied use `2*qL_int` (the solver's exact path);
    # otherwise fall back to the CN trapezoid `dt*(qL_n+qL_np1)`.
    rho_cp_left = geom.rho_cp[0, :]                          # (Ny,)
    denom = rho_cp_left[None, :] * geom.hx * geom.sigma_global
    if bc.qL_int is not None:
        f_norm = 2.0 * bc.qL_int / denom                    # (B,Ny) after bcast
    else:
        flux = bc.qL_n + bc.qL_np1                          # (B,Ny) after bcast
        f_norm = geom.dt * flux / denom
    left_res = bal[:, 0, :] - f_norm                        # (B, Ny)

    # Right Dirichlet column i=Nx-1 (algebraic).
    right_res = T_np1[:, Nx - 1, :] - bc.T_right_tilde       # (B, Ny)

    interior_res = bal[:, 1:Nx - 1, 1:Ny - 1]               # (B, Nx-2, Ny-2)
    topbot_res = torch.cat(
        (bal[:, 1:Nx - 1, 0], bal[:, 1:Nx - 1, Ny - 1]), dim=1
    )                                                       # (B, 2*(Nx-2))

    out = {
        "interior": interior_res.reshape(-1),
        "topbot_adiabatic": topbot_res.reshape(-1),
        "left_neumann": left_res.reshape(-1),
        "right_dirichlet": right_res.reshape(-1),
    }
    if dirichlet_both_ends:
        out["right_dirichlet_n"] = (T_n[:, Nx - 1, :] - bc.T_right_tilde).reshape(-1)
    return out
