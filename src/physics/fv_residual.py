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
    face_idx: torch.Tensor | None = None  # (B,) per-sample interface face; None for scalar geom

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


def _solver_face_coefficient(k: float, h: float) -> float:
    return float(k) / float(h)


def _solver_control_widths(n: int, h: float) -> np.ndarray:
    widths = np.full(int(n), float(h))
    widths[0] = widths[-1] = float(h) / 2.0
    return widths


@dataclass
class InterfaceLocation:
    """Where a single vertical interface sits on the FV grid.

    Separates the EXACT physical position (`interface_x_exact`, never quantized)
    from its discrete FV face (`face_idx` and the sub-cell widths `h_left`/
    `h_right` to that face). Continuous quantities (signed distance, the K-map,
    query-relative side) must use `interface_x_exact` so they never snap to a
    face center; the discrete conductance / dx adjustment / band masks use
    `face_idx`/`h_left`/`h_right`.

    Nodes never coincide with the interface (it is a face; the solver forbids an
    on-node interface via `1 <= face_idx <= Nx-3`), so `left_node_mask`
    (`x <= x_Gamma`, matching `problems/interfaces._material_channel`) and
    `left_cell_mask` (node index `<= face_idx`) are identical here; both are
    provided so the K-map side and the discrete band masks read from one source.
    """
    interface_x_exact: float
    face_idx: int
    h_left: float
    h_right: float
    left_node_mask: np.ndarray   # (Nx,) bool, x <= x_Gamma
    left_cell_mask: np.ndarray   # (Nx,) bool, node index <= face_idx


def locate_interface(x_grid, interface_x: float) -> InterfaceLocation:
    """Locate `interface_x` on the uniform x-grid, mirroring the solver's face
    indexing (`fv_solver_2d._validate_and_index_interfaces`).

    `face_idx = floor((x_Gamma - a) / hx)` with sub-cell widths
    `h_left = x_Gamma - x_node[face_idx]`, `h_right = hx - h_left`. Raises if the
    interface lands outside the solver's admissible face band
    `1 <= face_idx <= Nx-3` (so the Neumann/Dirichlet closures stay clear of it).
    """
    xg = np.asarray(x_grid, dtype=float)
    Nx = xg.shape[0]
    a = float(xg[0])
    hx = float(xg[1] - xg[0])
    x_exact = float(interface_x)

    face_idx = int(np.floor((x_exact - a) / hx))
    if face_idx < 1 or face_idx > Nx - 3:
        raise ValueError(
            f"interface_x={x_exact} maps to face slot {face_idx}; must satisfy "
            f"1 <= face_idx <= {Nx - 3} (Nx={Nx})."
        )
    x_left_node = a + face_idx * hx
    h_left = x_exact - x_left_node
    h_right = hx - h_left

    node_idx = np.arange(Nx)
    return InterfaceLocation(
        interface_x_exact=x_exact,
        face_idx=face_idx,
        h_left=float(h_left),
        h_right=float(h_right),
        left_node_mask=(xg <= x_exact),
        left_cell_mask=(node_idx <= face_idx),
    )


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
            G_x[i, :] = _solver_face_coefficient(k_face, hx)

    # --- G_y (Nx, Ny-1): k of the node's layer / hx ---
    k_nodes = _node_k(x_grid, k_left, k_right, interface_x)
    G_y = np.repeat(
        np.asarray([_solver_face_coefficient(k, hx) for k in k_nodes])[:, None],
        Ny - 1,
        axis=1,
    )

    # --- Control-volume widths (mirror the dx/dy construction) ---
    dx = _solver_control_widths(Nx, hx)
    dx[face_idx] = 0.5 * hx + h_L
    dx[face_idx + 1] = h_R + 0.5 * hx

    dy = _solver_control_widths(Ny, hx)

    t = lambda arr: torch.as_tensor(arr, device=device, dtype=dtype)
    return t(G_x), t(G_y), t(dx), t(dy)


def _assemble_scalar_cn_geom(
    G_x: torch.Tensor,
    G_y: torch.Tensor,
    dx: torch.Tensor,
    dy: torch.Tensor,
    *,
    dt: float,
    sigma_global: float,
    rho: float,
    cp: float,
    hx: float,
) -> FVGeom:
    """Assemble scalar-geometry CN coefficients from solver-style face data."""
    Nx = dx.shape[0]
    Ny = dy.shape[0]
    rho_cp = torch.full(
        (Nx, Ny), float(rho) * float(cp), device=dx.device, dtype=dx.dtype,
    )

    r_w = torch.zeros((Nx, Ny), device=dx.device, dtype=dx.dtype)
    r_e = torch.zeros((Nx, Ny), device=dx.device, dtype=dx.dtype)
    r_s = torch.zeros((Nx, Ny), device=dx.device, dtype=dx.dtype)
    r_n = torch.zeros((Nx, Ny), device=dx.device, dtype=dx.dtype)

    dxi = dx[:, None]
    dyj = dy[None, :]
    two_rc = 2.0 * rho_cp

    r_w[1:Nx - 1, :] = (
        dt * G_x[0:Nx - 2, :] / (two_rc[1:Nx - 1, :] * dxi[1:Nx - 1, :])
    )
    r_e[0:Nx - 1, :] = (
        dt * G_x[0:Nx - 1, :] / (two_rc[0:Nx - 1, :] * dxi[0:Nx - 1, :])
    )
    r_s[0:Nx - 1, 1:Ny] = (
        dt * G_y[0:Nx - 1, 0:Ny - 1]
        / (two_rc[0:Nx - 1, 1:Ny] * dyj[:, 1:Ny])
    )
    r_n[0:Nx - 1, 0:Ny - 1] = (
        dt * G_y[0:Nx - 1, 0:Ny - 1]
        / (two_rc[0:Nx - 1, 0:Ny - 1] * dyj[:, 0:Ny - 1])
    )

    interior_mask = torch.zeros((Nx, Ny), dtype=torch.bool, device=dx.device)
    interior_mask[1:Nx - 1, 1:Ny - 1] = True
    return FVGeom(
        G_x=G_x,
        G_y=G_y,
        dx=dx,
        dy=dy,
        rho_cp=rho_cp,
        r_w=r_w,
        r_e=r_e,
        r_s=r_s,
        r_n=r_n,
        interior_mask=interior_mask,
        dt=float(dt),
        sigma_global=float(sigma_global),
        hx=float(hx),
    )


def build_homogeneous_cn_geom(
    x_grid,
    y_grid,
    k: float,
    dt: float,
    sigma_global: float = 1.0,
    rho: float = 1.0,
    cp: float = 1.0,
    device=None,
    dtype=torch.float64,
) -> FVGeom:
    """Assemble the direct one-layer CN geometry used by ``FVSolver2D``.

    The solver's face coefficients are per unit orthogonal face length. On the
    required isotropic grid this gives ``k / h`` in both directions; the
    orthogonal width cancels against the control-volume area in the local CN
    coefficients. Boundary nodes use half-width one-dimensional control-volume
    widths, so corner volumes are one quarter of an interior volume.
    """
    xg = np.asarray(x_grid, dtype=float)
    yg = np.asarray(y_grid, dtype=float)
    if xg.ndim != 1 or yg.ndim != 1 or xg.size < 2 or yg.size < 2:
        raise ValueError("x_grid and y_grid must be one-dimensional with at least 2 nodes")
    hx = float((xg[-1] - xg[0]) / (xg.size - 1))
    hy = float((yg[-1] - yg[0]) / (yg.size - 1))
    if not np.allclose(np.diff(xg), hx, rtol=1e-5, atol=1e-8):
        raise ValueError("x_grid must be uniform for homogeneous CN geometry")
    if not np.allclose(np.diff(yg), hy, rtol=1e-5, atol=1e-8):
        raise ValueError("y_grid must be uniform for homogeneous CN geometry")
    if not np.isclose(hx, hy, rtol=1e-12):
        raise ValueError(
            f"Grid is not isotropic: hx={hx:.15e}, hy={hy:.15e}. "
            "Uniform isotropic grid required (hx == hy)."
        )
    if not np.isfinite(sigma_global) or float(sigma_global) <= 0.0:
        raise ValueError("sigma_global must be finite and > 0")

    Nx, Ny = xg.size, yg.size
    G_x = torch.full(
        (Nx - 1, Ny), _solver_face_coefficient(k, hx), device=device, dtype=dtype,
    )
    G_y = torch.full(
        (Nx, Ny - 1), _solver_face_coefficient(k, hx), device=device, dtype=dtype,
    )
    dx = torch.as_tensor(_solver_control_widths(Nx, hx), device=device, dtype=dtype)
    dy = torch.as_tensor(_solver_control_widths(Ny, hx), device=device, dtype=dtype)
    return _assemble_scalar_cn_geom(
        G_x,
        G_y,
        dx,
        dy,
        dt=dt,
        sigma_global=sigma_global,
        rho=rho,
        cp=cp,
        hx=hx,
    )


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
    hx = float(np.asarray(x_grid, dtype=float)[1] - np.asarray(x_grid, dtype=float)[0])
    return _assemble_scalar_cn_geom(
        G_x,
        G_y,
        dx,
        dy,
        dt=dt,
        sigma_global=sigma_global,
        rho=rho,
        cp=cp,
        hx=hx,
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


def build_cn_geom_per_interface(x_grid, y_grid, k_left: float, k_right: float,
                                interface_x_batch, R_c_batch, dt: float,
                                sigma_global: float = 1.0,
                                rho: float = 1.0, cp: float = 1.0,
                                device=None, dtype=torch.float64) -> FVGeom:
    """Per-sample CN geometry for a batch of DISTINCT interface locations and
    contact resistances.

    Unlike `build_cn_geom_batched` (which fixes `interface_x` and only varies the
    single interface x-face column with `R_c`), moving `interface_x` per sample
    shifts `face_idx`, the sub-cell widths `h_L`/`h_R`, the material split of
    every x-face and y-node conductance, and the two interface-adjacent `dx`
    widths. So EVERY material-dependent quantity is batched here:
      - `G_x` `(B, Nx-1, Ny)`: non-interface faces `k_face/hx` split by each
        sample's interface, interface column the series conductance
        `1/(h_L/k_L + R_c + h_R/k_R)`.
      - `G_y` `(B, Nx, Ny-1)`: each node's `k/hx` per the sample's interface.
      - `dx` `(B, Nx)`: the two interface-flanking widths shift with `h_L`/`h_R`.
      - `r_w/r_e/r_s/r_n` `(B, Nx, Ny)`: derived from the batched `G_x/G_y/dx`.
    `rho_cp` (uniform), `dy`, `interior_mask`, and `hx` are batch-independent.

    Reduces bit-for-bit to `build_cn_geom` / `build_cn_geom_batched` when all
    `interface_x` (and `R_c`) are equal (the interface is off-node, so the K-map
    `<=` and the solver's strict `<` agree). Mirrors
    `build_face_conductances` + `build_cn_geom`.
    """
    iface = torch.as_tensor(interface_x_batch, device=device, dtype=dtype).reshape(-1)
    R_c_t = torch.as_tensor(R_c_batch, device=device, dtype=dtype).reshape(-1)
    if iface.shape[0] != R_c_t.shape[0]:
        raise ValueError(
            f"interface_x_batch ({iface.shape[0]}) and R_c_batch "
            f"({R_c_t.shape[0]}) must have the same length."
        )
    B = iface.shape[0]

    xg = np.asarray(x_grid, dtype=float)
    yg = np.asarray(y_grid, dtype=float)
    Nx = xg.shape[0]
    Ny = yg.shape[0]
    a = float(xg[0])
    hx = float(xg[1] - xg[0])

    # Validate every interface once via the canonical locator (raises on an
    # out-of-band or on-node interface) and gather the discrete indices.
    locs = [locate_interface(xg, float(v)) for v in iface.tolist()]
    face_idx = torch.as_tensor([loc.face_idx for loc in locs],
                               device=device, dtype=torch.long)          # (B,)
    # Mirror `build_face_conductances`'s h_L/h_R construction bit-for-bit
    # (h_R = x_right_node - interface_x, NOT hx - h_L) so the per-interface
    # geometry reduces exactly to `build_cn_geom` when interfaces coincide.
    h_L = iface - (a + face_idx.to(dtype) * hx)                          # (B,)
    h_R = (a + (face_idx.to(dtype) + 1.0) * hx) - iface                 # (B,)

    x_nodes = torch.as_tensor(xg, device=device, dtype=dtype)           # (Nx,)
    face_pos = torch.as_tensor(a + (np.arange(Nx - 1) + 0.5) * hx,
                               device=device, dtype=dtype)              # (Nx-1,)
    kL = torch.as_tensor(float(k_left), device=device, dtype=dtype)
    kR = torch.as_tensor(float(k_right), device=device, dtype=dtype)

    # --- G_x (B, Nx-1, Ny) ---
    left_face = face_pos[None, :] < iface[:, None]                      # (B, Nx-1)
    G_x2d = torch.where(left_face, kL, kR) / hx                         # (B, Nx-1)
    G_series = 1.0 / (h_L / float(k_left) + R_c_t + h_R / float(k_right))  # (B,)
    is_iface = (torch.arange(Nx - 1, device=device)[None, :]
                == face_idx[:, None])                                    # (B, Nx-1)
    G_x2d = torch.where(is_iface, G_series[:, None], G_x2d)             # (B, Nx-1)
    G_x = G_x2d[:, :, None].expand(B, Nx - 1, Ny).contiguous()

    # --- G_y (B, Nx, Ny-1) ---
    left_node = x_nodes[None, :] < iface[:, None]                       # (B, Nx)
    k_nodes = torch.where(left_node, kL, kR)                            # (B, Nx)
    G_y = (k_nodes / hx)[:, :, None].expand(B, Nx, Ny - 1).contiguous()

    # --- dx (B, Nx): boundary halves, then interface-flanking widths ---
    dx = torch.full((B, Nx), hx, device=device, dtype=dtype)
    dx[:, 0] = hx / 2.0
    dx[:, Nx - 1] = hx / 2.0
    brow = torch.arange(B, device=device)
    dx[brow, face_idx] = 0.5 * hx + h_L
    dx[brow, face_idx + 1] = h_R + 0.5 * hx

    dy = torch.full((Ny,), hx, device=device, dtype=dtype)
    dy[0] = hx / 2.0
    dy[Ny - 1] = hx / 2.0

    rho_cp_val = float(rho) * float(cp)
    two_rc = 2.0 * rho_cp_val
    r_w = torch.zeros((B, Nx, Ny), device=device, dtype=dtype)
    r_e = torch.zeros((B, Nx, Ny), device=device, dtype=dtype)
    r_s = torch.zeros((B, Nx, Ny), device=device, dtype=dtype)
    r_n = torch.zeros((B, Nx, Ny), device=device, dtype=dtype)

    dxi = dx[:, :, None]        # (B, Nx, 1)
    dyj = dy[None, None, :]     # (1, 1, Ny)

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
        face_idx=face_idx,
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


def _interface_face_indices(geom: FVGeom, batch_size: int) -> torch.Tensor:
    face_idx = geom.face_idx
    if face_idx is None:
        raise ValueError(
            "interface rate residual requires geom.face_idx; build geometry "
            "with build_cn_geom_per_interface"
        )
    face_idx = face_idx.to(device=geom.r_w.device, dtype=torch.long).reshape(-1)
    if face_idx.numel() == 1 and batch_size > 1:
        face_idx = face_idx.expand(batch_size)
    if face_idx.numel() != batch_size:
        raise ValueError(
            f"geom.face_idx has {face_idx.numel()} entries for batch size {batch_size}"
        )
    if int(face_idx.min()) < 1 or int(face_idx.max()) > geom.Nx - 3:
        raise ValueError(
            "interface cells must stay clear of the physical boundary closures"
        )
    return face_idx


def _full_cn_rate_balance(
    T_n: torch.Tensor, T_np1: torch.Tensor, geom: FVGeom,
) -> torch.Tensor:
    """Normalized-temperature CN energy balance in temperature-rate units."""
    if T_n.dim() == 2:
        T_n = T_n[None]
        T_np1 = T_np1[None]
    if T_n.shape != T_np1.shape:
        raise ValueError(
            f"T_n shape {tuple(T_n.shape)} != T_np1 shape {tuple(T_np1.shape)}"
        )
    if T_n.shape[-2:] != (geom.Nx, geom.Ny):
        raise ValueError(
            f"full-grid fields must end in {(geom.Nx, geom.Ny)}, got {tuple(T_n.shape)}"
        )
    dt = float(geom.dt)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"geom.dt must be finite and positive, got {dt!r}")
    return (
        (T_np1 - T_n) / dt
        - (_cn_laplacian_full(T_np1, geom) + _cn_laplacian_full(T_n, geom)) / dt
    )


def _gather_x(tensor: torch.Tensor, columns: torch.Tensor) -> torch.Tensor:
    """Gather per-batch x columns from a 2D or batched 3D geometry tensor."""
    batch_size, n_columns = columns.shape
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0).expand(batch_size, -1, -1)
    elif tensor.dim() == 3 and tensor.shape[0] == 1 and batch_size > 1:
        tensor = tensor.expand(batch_size, -1, -1)
    if tensor.dim() != 3 or tensor.shape[0] != batch_size:
        raise ValueError("geometry coefficient batch does not match the temperature batch")
    return tensor.gather(
        1, columns[:, :, None].expand(batch_size, n_columns, tensor.shape[-1])
    )


def _quadratic_trace_weights(
    nodes: np.ndarray, x_eval: float,
) -> tuple[np.ndarray, np.ndarray]:
    nodes = np.asarray(nodes, dtype=np.float64)
    vandermonde = np.stack((np.ones(3), nodes, nodes ** 2), axis=1)
    inverse = np.linalg.inv(vandermonde)
    return (
        np.array([1.0, x_eval, x_eval ** 2]) @ inverse,
        np.array([0.0, 1.0, 2.0 * x_eval]) @ inverse,
    )


def interface_residual_rate(
    T_n: torch.Tensor, T_np1: torch.Tensor, geom: FVGeom,
) -> torch.Tensor:
    """Rate-form FV residual in the two interface-adjacent columns.

    Inputs may be full fields ``(B,Nx,Ny)`` or the local four-column stencil
    ``(B,4,Ny)`` ordered ``[f-1,f,f+1,f+2]``. The result is always
    ``(B,2,Ny)`` and includes the top and bottom interface cells.
    """
    if T_n.dim() == 2:
        T_n = T_n[None]
        T_np1 = T_np1[None]
    if T_n.shape != T_np1.shape:
        raise ValueError(
            f"T_n shape {tuple(T_n.shape)} != T_np1 shape {tuple(T_np1.shape)}"
        )
    if T_n.dim() != 3 or T_n.shape[-1] != geom.Ny:
        raise ValueError("interface fields must have shape (B,Nx_or_4,Ny)")

    batch_size = T_n.shape[0]
    face_idx = _interface_face_indices(geom, batch_size)
    stencil_cols = face_idx[:, None] + torch.tensor(
        [-1, 0, 1, 2], device=face_idx.device, dtype=torch.long,
    )[None, :]
    if T_n.shape[1] == geom.Nx:
        T_n = T_n.gather(
            1, stencil_cols[:, :, None].expand(batch_size, 4, geom.Ny)
        )
        T_np1 = T_np1.gather(
            1, stencil_cols[:, :, None].expand(batch_size, 4, geom.Ny)
        )
    elif T_n.shape[1] != 4:
        raise ValueError(
            f"interface fields must contain Nx={geom.Nx} or 4 columns, got {T_n.shape[1]}"
        )

    target_cols = face_idx[:, None] + torch.tensor(
        [0, 1], device=face_idx.device, dtype=torch.long,
    )[None, :]
    rw = _gather_x(geom.r_w, target_cols)
    re = _gather_x(geom.r_e, target_cols)
    rs = _gather_x(geom.r_s, target_cols)
    rn = _gather_x(geom.r_n, target_cols)

    def _local_laplacian(T: torch.Tensor) -> torch.Tensor:
        center = T[:, 1:3, :]
        west = T[:, 0:2, :]
        east = T[:, 2:4, :]
        south = torch.zeros_like(center)
        south[:, :, 1:] = center[:, :, :-1]
        north = torch.zeros_like(center)
        north[:, :, :-1] = center[:, :, 1:]
        return (
            rw * (west - center)
            + re * (east - center)
            + rs * (south - center)
            + rn * (north - center)
        )

    dt = float(geom.dt)
    return (
        (T_np1[:, 1:3, :] - T_n[:, 1:3, :]) / dt
        - (_local_laplacian(T_np1) + _local_laplacian(T_n)) / dt
    )


def block_energy_residual(
    T_n: torch.Tensor,
    T_np1: torch.Tensor,
    geom: FVGeom,
    qL_int: torch.Tensor,
    *,
    x_blocks_per_layer: int,
    y_blocks: int,
    q_ref: float,
    scale_floor: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Dimensionless conservative balances on deterministic rectangular blocks.

    The block numerators are sums of the solver's existing per-cell CN balance,
    converted back to physical energy. Internal face contributions therefore
    telescope exactly. The algebraic right-Dirichlet node is excluded from the
    energy cells; the face from ``Nx-2`` to that node is the discrete heat sink.
    """
    if T_n.dim() == 2:
        T_n = T_n[None]
        T_np1 = T_np1[None]
    if T_n.shape != T_np1.shape or T_n.shape[-2:] != (geom.Nx, geom.Ny):
        raise ValueError("block energy fields must have shape (B,Nx,Ny)")
    if x_blocks_per_layer < 1 or y_blocks < 1:
        raise ValueError("block counts must be positive")
    if not np.isfinite(q_ref) or float(q_ref) <= 0.0:
        raise ValueError("q_ref must be finite and positive")
    if not np.isfinite(scale_floor) or float(scale_floor) <= 0.0:
        raise ValueError("scale_floor must be finite and positive")

    batch_size, Nx, Ny = T_n.shape
    face_idx = _interface_face_indices(geom, batch_size)
    if not bool((face_idx == face_idx[0]).all().item()):
        raise ValueError("block energy currently requires one shared interface face")
    face = int(face_idx[0].item())
    left_cells = np.arange(0, face + 1, dtype=np.int64)
    right_cells = np.arange(face + 1, Nx - 1, dtype=np.int64)
    if min(left_cells.size, right_cells.size) < x_blocks_per_layer:
        raise ValueError("too many x blocks for the available material cells")
    if Ny < y_blocks:
        raise ValueError("too many y blocks for the available rows")
    x_parts = [
        *np.array_split(left_cells, x_blocks_per_layer),
        *np.array_split(right_cells, x_blocks_per_layer),
    ]
    y_parts = np.array_split(np.arange(Ny, dtype=np.int64), y_blocks)

    balance = (
        (T_np1 - T_n)
        - (_cn_laplacian_full(T_np1, geom) + _cn_laplacian_full(T_n, geom))
    )
    rho_cp_left = geom.rho_cp[0, :]
    left_denom = rho_cp_left[None, :] * float(geom.hx) * float(geom.sigma_global)
    forcing_increment = 2.0 * qL_int.to(balance) / left_denom
    balance[:, 0, :] = balance[:, 0, :] - forcing_increment

    dx = geom.dx.to(balance)
    if dx.dim() == 1:
        dx = dx.unsqueeze(0).expand(batch_size, -1)
    rho_cp = geom.rho_cp.to(balance)
    if rho_cp.dim() == 2:
        rho_cp = rho_cp.unsqueeze(0).expand(batch_size, -1, -1)
    dy = geom.dy.to(balance)
    capacity = rho_cp * dx[:, :, None] * dy[None, None, :]
    physical_cell_residual = (
        capacity * float(geom.sigma_global) * balance
    )

    residuals = []
    numerators = []
    denominators = []
    interface_mask = []
    block_x = []
    block_y = []
    for x_block, x_ids_np in enumerate(x_parts):
        x_ids = torch.as_tensor(x_ids_np, device=balance.device, dtype=torch.long)
        touches_left_wall = int(x_ids_np[0]) == 0
        touches_interface = (
            x_block == x_blocks_per_layer - 1
            or x_block == x_blocks_per_layer
        )
        for y_block, y_ids_np in enumerate(y_parts):
            y_ids = torch.as_tensor(y_ids_np, device=balance.device, dtype=torch.long)
            numerator = physical_cell_residual.index_select(1, x_ids).index_select(
                2, y_ids
            ).sum(dim=(1, 2))
            block_height = dy.index_select(0, y_ids).sum()
            prescribed = (
                (qL_int.to(balance).index_select(1, y_ids)
                 * dy.index_select(0, y_ids)[None, :]).sum(dim=1).abs()
                if touches_left_wall else torch.zeros_like(numerator)
            )
            characteristic = (
                float(q_ref) * float(geom.dt) * block_height * float(scale_floor)
            )
            denominator = (prescribed + characteristic).clamp_min(1.0e-12)
            residuals.append(numerator / denominator)
            numerators.append(numerator)
            denominators.append(denominator)
            interface_mask.append(touches_interface)
            block_x.append(x_block)
            block_y.append(y_block)
    residual = torch.stack(residuals, dim=1)
    mask = torch.as_tensor(interface_mask, device=residual.device, dtype=torch.bool)
    return {
        "all": residual,
        "numerator": torch.stack(numerators, dim=1),
        "denominator": torch.stack(denominators, dim=1),
        "interface": residual[:, mask],
        "far": residual[:, ~mask],
        "interface_mask": mask,
        "block_x": torch.as_tensor(block_x, device=residual.device),
        "block_y": torch.as_tensor(block_y, device=residual.device),
    }


def interface_trace_constraint_residuals(
    T_n: torch.Tensor,
    T_np1: torch.Tensor,
    geom: FVGeom,
    x_grid,
    R_c,
    *,
    k_left: float,
    k_right: float,
    sigma_global: float,
    q_ref: float,
    jump_floor_K: float = 1.0,
) -> dict[str, torch.Tensor]:
    """One-sided quadratic interface flux and contact-law residuals."""
    if T_n.dim() == 2:
        T_n = T_n[None]
        T_np1 = T_np1[None]
    if T_n.shape != T_np1.shape or T_n.shape[-2:] != (geom.Nx, geom.Ny):
        raise ValueError("interface trace fields must have shape (B,Nx,Ny)")
    if min(k_left, k_right, sigma_global, q_ref, jump_floor_K) <= 0.0:
        raise ValueError("interface trace scales and conductivities must be positive")
    batch_size, _, Ny = T_n.shape
    x = np.asarray(x_grid, dtype=np.float64)
    face_idx = _interface_face_indices(geom, batch_size)
    rc = torch.as_tensor(R_c, device=T_n.device, dtype=T_n.dtype).reshape(-1)
    if rc.numel() == 1 and batch_size > 1:
        rc = rc.expand(batch_size)
    if rc.numel() != batch_size:
        raise ValueError("R_c batch does not match the temperature batch")

    value_left = []
    deriv_left = []
    value_right = []
    deriv_right = []
    for b in range(batch_size):
        face = int(face_idx[b].item())
        h_left = float(geom.dx[b, face].item() if geom.dx.dim() == 2 else geom.dx[face].item()) - 0.5 * float(geom.hx)
        interface_x = float(x[face] + h_left)
        lv, ld = _quadratic_trace_weights(x[face - 2:face + 1], interface_x)
        rv, rd = _quadratic_trace_weights(x[face + 1:face + 4], interface_x)
        value_left.append(lv)
        deriv_left.append(ld)
        value_right.append(rv)
        deriv_right.append(rd)
    weights = [
        torch.as_tensor(np.asarray(values), device=T_n.device, dtype=T_n.dtype)
        for values in (value_left, deriv_left, value_right, deriv_right)
    ]
    value_left_t, deriv_left_t, value_right_t, deriv_right_t = weights

    endpoint_flux = []
    endpoint_contact = []
    raw = []
    for field in (T_n, T_np1):
        left_nodes = []
        right_nodes = []
        series_left = []
        series_right = []
        conductance = []
        for b in range(batch_size):
            face = int(face_idx[b].item())
            left_nodes.append(field[b, face - 2:face + 1])
            right_nodes.append(field[b, face + 1:face + 4])
            series_left.append(field[b, face])
            series_right.append(field[b, face + 1])
            conductance.append(geom.G_x[b, face] if geom.G_x.dim() == 3 else geom.G_x[face])
        left_nodes_t = torch.stack(left_nodes)
        right_nodes_t = torch.stack(right_nodes)
        T_minus = float(sigma_global) * torch.einsum(
            "bi,bij->bj", value_left_t, left_nodes_t
        )
        T_plus = float(sigma_global) * torch.einsum(
            "bi,bij->bj", value_right_t, right_nodes_t
        )
        q_minus = -float(k_left) * float(sigma_global) * torch.einsum(
            "bi,bij->bj", deriv_left_t, left_nodes_t
        )
        q_plus = -float(k_right) * float(sigma_global) * torch.einsum(
            "bi,bij->bj", deriv_right_t, right_nodes_t
        )
        g_face = torch.stack(conductance)
        q_series = g_face * float(sigma_global) * (
            torch.stack(series_left) - torch.stack(series_right)
        )
        endpoint_flux.append(torch.stack(
            ((q_minus - q_series) / float(q_ref),
             (q_plus - q_series) / float(q_ref)), dim=1,
        ))
        jump_scale = torch.maximum(
            rc * float(q_ref), torch.full_like(rc, float(jump_floor_K))
        )
        endpoint_contact.append(
            (T_minus - T_plus - rc[:, None] * q_series) / jump_scale[:, None]
        )
        raw.append((q_minus, q_plus, q_series, T_minus - T_plus))
    return {
        "flux": torch.stack(endpoint_flux, dim=1),
        "contact": torch.stack(endpoint_contact, dim=1),
        "q_minus": torch.stack([value[0] for value in raw], dim=1),
        "q_plus": torch.stack([value[1] for value in raw], dim=1),
        "q_series": torch.stack([value[2] for value in raw], dim=1),
        "jump_K": torch.stack([value[3] for value in raw], dim=1),
    }


def full_bc_cn_residual_rate(
    T_n: torch.Tensor,
    T_np1: torch.Tensor,
    geom: FVGeom,
    bc: FullBCData,
    dirichlet_both_ends: bool = False,
    keep_batch: bool = False,
) -> dict:
    """Full-boundary CN residual in normalized-temperature-rate units.

    The energy-balance regions are the rate-form counterpart of
    :func:`full_bc_cn_residual`. The algebraic right-Dirichlet residual remains
    a normalized-temperature mismatch and is intentionally not divided by time.
    """
    if T_n.dim() == 2:
        T_n = T_n[None]
        T_np1 = T_np1[None]
    balance = _full_cn_rate_balance(T_n, T_np1, geom)
    Nx, Ny = geom.Nx, geom.Ny

    rho_cp_left = geom.rho_cp[0, :]
    denom = rho_cp_left[None, :] * geom.hx * geom.sigma_global
    if bc.qL_int is not None:
        forcing_rate = 2.0 * bc.qL_int / (float(geom.dt) * denom)
    else:
        forcing_rate = (bc.qL_n + bc.qL_np1) / denom
    left = balance[:, 0, :] - forcing_rate
    right = T_np1[:, Nx - 1, :] - bc.T_right_tilde
    interior = balance[:, 1:Nx - 1, 1:Ny - 1]
    top = balance[:, 1:Nx - 1, 0]
    bottom = balance[:, 1:Nx - 1, Ny - 1]

    out = {
        "interior": interior,
        "top_adiabatic": top,
        "bottom_adiabatic": bottom,
        "topbot_adiabatic": torch.cat((top, bottom), dim=1),
        "left_neumann": left,
        "right_dirichlet": right,
    }
    if dirichlet_both_ends:
        out["right_dirichlet_n"] = T_n[:, Nx - 1, :] - bc.T_right_tilde
    if keep_batch:
        return out
    return {key: value.reshape(-1) for key, value in out.items()}


def region_balanced_fv_rate_residual(
    T_n: torch.Tensor,
    T_np1: torch.Tensor,
    geom: FVGeom,
    bc: FullBCData,
    dirichlet_both_ends: bool = False,
    keep_batch: bool = False,
) -> dict:
    """Disjoint rate residuals for bulk, interface, and physical boundaries."""
    if T_n.dim() == 2:
        T_n = T_n[None]
        T_np1 = T_np1[None]
    balance = _full_cn_rate_balance(T_n, T_np1, geom)
    batch_size, Nx, Ny = balance.shape
    face_idx = _interface_face_indices(geom, batch_size)

    interface_cols = face_idx[:, None] + torch.tensor(
        [0, 1], device=face_idx.device, dtype=torch.long,
    )[None, :]
    interface = balance.gather(
        1, interface_cols[:, :, None].expand(batch_size, 2, Ny)
    )

    active = torch.arange(1, Nx - 1, device=balance.device)[None, :].expand(
        batch_size, -1
    )
    keep = (active != face_idx[:, None]) & (active != (face_idx + 1)[:, None])
    bulk_cols = active[keep].view(batch_size, Nx - 4)
    bulk = balance.gather(
        1, bulk_cols[:, :, None].expand(batch_size, Nx - 4, Ny)
    )

    rho_cp_left = geom.rho_cp[0, :]
    denom = rho_cp_left[None, :] * geom.hx * geom.sigma_global
    if bc.qL_int is not None:
        forcing_rate = 2.0 * bc.qL_int / (float(geom.dt) * denom)
    else:
        forcing_rate = (bc.qL_n + bc.qL_np1) / denom
    left = balance[:, 0, :] - forcing_rate
    right = T_np1[:, Nx - 1, :] - bc.T_right_tilde

    out = {
        "interior": bulk[:, :, 1:Ny - 1],
        "interface": interface,
        "left_neumann": left,
        "top_adiabatic": bulk[:, :, 0],
        "bottom_adiabatic": bulk[:, :, Ny - 1],
        "right_dirichlet": right,
    }
    out["topbot_adiabatic"] = torch.cat(
        (out["top_adiabatic"], out["bottom_adiabatic"]), dim=1
    )
    if dirichlet_both_ends:
        out["right_dirichlet_n"] = T_n[:, Nx - 1, :] - bc.T_right_tilde
    if keep_batch:
        return out
    return {key: value.reshape(-1) for key, value in out.items()}


def full_bc_cn_residual(T_n: torch.Tensor, T_np1: torch.Tensor,
                        geom: FVGeom, bc: FullBCData,
                        dirichlet_both_ends: bool = False,
                        keep_batch: bool = False) -> dict:
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

    With ``keep_batch`` the four (five) regions retain their leading batch dim
    instead of being flattened, so callers can reduce per-sample:
    ``interior (B, Nx-2, Ny-2)``, ``topbot_adiabatic (B, 2*(Nx-2))``,
    ``left_neumann (B, Ny)``, ``right_dirichlet (B, Ny)`` (+ ``right_dirichlet_n
    (B, Ny)``). The flattened default is ``reshape(-1)`` of these.
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

    right_dirichlet_n = None
    if dirichlet_both_ends:
        right_dirichlet_n = T_n[:, Nx - 1, :] - bc.T_right_tilde   # (B, Ny)

    if keep_batch:
        out = {
            "interior": interior_res,
            "topbot_adiabatic": topbot_res,
            "left_neumann": left_res,
            "right_dirichlet": right_res,
        }
        if right_dirichlet_n is not None:
            out["right_dirichlet_n"] = right_dirichlet_n
        return out

    out = {
        "interior": interior_res.reshape(-1),
        "topbot_adiabatic": topbot_res.reshape(-1),
        "left_neumann": left_res.reshape(-1),
        "right_dirichlet": right_res.reshape(-1),
    }
    if right_dirichlet_n is not None:
        out["right_dirichlet_n"] = right_dirichlet_n.reshape(-1)
    return out
