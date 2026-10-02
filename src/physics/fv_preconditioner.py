"""Fixed CN operators with exact and Torch geometric multigrid inverses."""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
import torch
from torch import nn

from src.physics.fv_residual import FVGeom


def assemble_active_cn_matrix(geom: FVGeom) -> sp.csr_matrix:
    """Free-node CN matrix in the solver's i+j*(Nx-1) order."""
    if geom.r_w.ndim != 2:
        raise ValueError("A fixed inverse requires one batch-independent FV geometry")
    nx, ny = geom.Nx - 1, geom.Ny
    index = np.arange(nx * ny).reshape((nx, ny), order="F")
    rw, re, rs, rn = (
        getattr(geom, name).detach().cpu().double().numpy()[:nx]
        for name in ("r_w", "r_e", "r_s", "r_n")
    )
    rows = [index.ravel()]
    cols = [index.ravel()]
    values = [(1.0 + rw + re + rs + rn).ravel()]
    for row, col, coefficient in (
        (index[1:], index[:-1], rw[1:]),
        (index[:-1], index[1:], re[:-1]),
        (index[:, 1:], index[:, :-1], rs[:, 1:]),
        (index[:, :-1], index[:, 1:], rn[:, :-1]),
    ):
        rows.append(row.ravel())
        cols.append(col.ravel())
        values.append(-coefficient.ravel())
    return sp.coo_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
        shape=(nx * ny, nx * ny),
    ).tocsr()


class _SparseSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, residual, factor, transpose):
        ctx.factor = factor
        ctx.transpose = transpose
        nx, ny = residual.shape[-2:]
        # The public tensor is (x,y); SuperLU consumes the solver's Fortran order.
        packed = residual.detach().cpu().double().transpose(-2, -1).reshape(-1, nx * ny)
        solved = factor.solve(packed.numpy().T, trans="T" if transpose else "N")
        result = torch.from_numpy(np.ascontiguousarray(solved.T)).reshape(
            *residual.shape[:-2], ny, nx,
        ).transpose(-2, -1)
        return result.to(device=residual.device, dtype=residual.dtype)

    @staticmethod
    def backward(ctx, gradient):
        return _SparseSolve.apply(gradient, ctx.factor, not ctx.transpose), None, None


class _SparseMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, field, matrix, transpose):
        ctx.matrix = matrix
        ctx.transpose = transpose
        nx, ny = field.shape[-2:]
        packed = field.detach().cpu().double().transpose(-2, -1).reshape(-1, nx * ny)
        operator = matrix.T if transpose else matrix
        product = operator @ packed.numpy().T
        result = torch.from_numpy(np.ascontiguousarray(product.T)).reshape(
            *field.shape[:-2], ny, nx,
        ).transpose(-2, -1)
        return result.to(device=field.device, dtype=field.dtype)

    @staticmethod
    def backward(ctx, gradient):
        return _SparseMultiply.apply(gradient, ctx.matrix, not ctx.transpose), None, None


class ExactCNInverse(nn.Module):
    """Diagnostic A_hat inverse with one cached CPU factorization and transpose backward."""

    def __init__(self, geom: FVGeom):
        super().__init__()
        self.free_shape = (geom.Nx - 1, geom.Ny)
        self.matrix = assemble_active_cn_matrix(geom)
        self.factor = splu(self.matrix.tocsc())
        # Capacity division makes A and B nonsymmetric; backward needs true transposes.
        self.explicit_matrix = 2.0 * sp.eye(self.matrix.shape[0], format="csr") - self.matrix

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        if residual.shape[-2:] != self.free_shape:
            raise ValueError(f"Expected free field {self.free_shape}, got {tuple(residual.shape)}")
        return _SparseSolve.apply(residual, self.factor, False)

    def space_time(self, residuals: torch.Tensor) -> torch.Tensor:
        """Solve the anchored CN block system on (...,time,Nx-1,Ny)."""
        if residuals.ndim < 3 or residuals.shape[-3] < 1:
            raise ValueError("A space-time residual requires a nonempty time dimension")
        previous = torch.zeros_like(residuals[..., 0, :, :])
        corrections = []
        for residual in residuals.unbind(dim=-3):
            coupled = _SparseMultiply.apply(previous, self.explicit_matrix, False)
            previous = self(residual + coupled)
            corrections.append(previous)
        return torch.stack(corrections, dim=-3)


class MGOneCycleInverse(nn.Module):
    """One zero-start geometric V-cycle for the capacity-scaled SPD CN system.

    Setup uses SciPy once. Application and its true transpose gradient use only
    Torch buffers, including the cached small coarse-grid inverse.
    """

    def __init__(self, geom: FVGeom, pre_smooth=2, post_smooth=2,
                 omega=2 / 3, coarse_max_unknowns=64):
        super().__init__()
        if pre_smooth < 1 or post_smooth < 1 or not 0 < omega < 1 or coarse_max_unknowns < 4:
            raise ValueError("MG requires positive smoothing, 0 < omega < 1, and a coarse size >= 4")
        self.free_shape = (geom.Nx - 1, geom.Ny)
        self.pre_smooth, self.post_smooth, self.omega = pre_smooth, post_smooth, omega
        capacity = (geom.rho_cp[:-1] * geom.dx[:-1, None] * geom.dy[None, :]).detach().cpu().double().numpy()
        matrix = sp.diags(capacity.ravel(order="F")) @ assemble_active_cn_matrix(geom)
        asymmetry = matrix - matrix.T
        if np.any(capacity <= 0) or np.max(np.abs(asymmetry.data), initial=0) > 1e-12:
            raise ValueError("MG requires positive capacity and a symmetric capacity-scaled CN matrix")
        self.register_buffer("capacity", torch.from_numpy(capacity.copy()))
        x, y = np.arange(geom.Nx, dtype=float), np.arange(geom.Ny, dtype=float)
        self.level_shapes = []
        while True:
            nx, ny = len(x) - 1, len(y)
            self.level_shapes.append((nx, ny))
            level = len(self.level_shapes) - 1
            diagonal = matrix.diagonal().reshape(nx, ny, order="F").copy()
            self.register_buffer(f"diagonal_{level}", torch.from_numpy(diagonal))
            if nx * ny <= coarse_max_unknowns or min(nx, ny) <= 2:
                # Sparse matrices use Fortran order; the Torch coarse action uses (x,y).reshape.
                permutation = np.arange(nx * ny).reshape(nx, ny, order="F").ravel()
                dense = matrix.tocsr()[permutation][:, permutation].toarray()
                chol = np.linalg.cholesky(dense)
                coarse_inverse = np.linalg.solve(chol.T, np.linalg.solve(chol, np.eye(nx * ny)))
                self.register_buffer("coarse_inverse", torch.from_numpy(coarse_inverse))
                break
            entries = matrix.tocoo()
            row_x, row_y = entries.row % nx, entries.row // nx
            col_x, col_y = entries.col % nx, entries.col // nx
            coefficients = np.zeros((9, nx, ny))
            offsets = (col_x - row_x + 1) * 3 + col_y - row_y + 1
            if np.any(np.abs(col_x - row_x) > 1) or np.any(np.abs(col_y - row_y) > 1):
                raise ValueError("MG Galerkin hierarchy requires a local nine-point stencil")
            coefficients[offsets, row_x, row_y] = entries.data
            self.register_buffer(f"stencil_{level}", torch.from_numpy(coefficients))
            coarse_x = x[np.unique(np.r_[np.arange(0, len(x), 2), len(x) - 1])]
            coarse_y = y[np.unique(np.r_[np.arange(0, len(y), 2), len(y) - 1])]
            px = self._interpolation(x, coarse_x)[:-1, :-1]
            py = self._interpolation(y, coarse_y)
            self.register_buffer(f"px_{level}", torch.from_numpy(px))
            self.register_buffer(f"py_{level}", torch.from_numpy(py))
            prolongation = sp.kron(sp.csr_matrix(py), sp.csr_matrix(px), format="csr")
            matrix = (prolongation.T @ matrix @ prolongation).tocsr()
            x, y = coarse_x, coarse_y

    @staticmethod
    def _interpolation(fine, coarse):
        right = np.searchsorted(coarse, fine, side="right").clip(1, len(coarse) - 1)
        left = right - 1
        fraction = (fine - coarse[left]) / (coarse[right] - coarse[left])
        result = np.zeros((len(fine), len(coarse)))
        result[np.arange(len(fine)), left] = 1 - fraction
        result[np.arange(len(fine)), right] = fraction
        return result

    def _action(self, field, level):
        nx, ny = self.level_shapes[level]
        padded = torch.nn.functional.pad(field, (1, 1, 1, 1))
        neighbors = torch.stack([padded[..., i:i + nx, j:j + ny]
                                 for i in range(3) for j in range(3)], dim=-3)
        return (neighbors * getattr(self, f"stencil_{level}")).sum(dim=-3)

    def _cycle(self, rhs, level):
        if level == len(self.level_shapes) - 1:
            return (rhs.flatten(-2) @ self.coarse_inverse.T).reshape(rhs.shape)
        diagonal = getattr(self, f"diagonal_{level}")
        result = self.omega * rhs / diagonal
        for _ in range(self.pre_smooth - 1):
            result = result + self.omega * (rhs - self._action(result, level)) / diagonal
        px, py = getattr(self, f"px_{level}"), getattr(self, f"py_{level}")
        coarse_rhs = px.T @ (rhs - self._action(result, level)) @ py
        result = result + px @ self._cycle(coarse_rhs, level + 1) @ py.T
        for _ in range(self.post_smooth):
            result = result + self.omega * (rhs - self._action(result, level)) / diagonal
        return result

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        if residual.shape[-2:] != self.free_shape:
            raise ValueError(f"Expected free field {self.free_shape}, got {tuple(residual.shape)}")
        return self._cycle(residual * self.capacity, 0)
