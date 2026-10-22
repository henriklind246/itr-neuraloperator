"""Solution-space conditioning of the forcing_homogeneous physics-only CN loss.

Over N intervals the CN residuals form one space-time system r = Du - b, and the
loss 0.5*||P(Du - b)||^2 has Hessian H = (PD)^T PD. D (A on the block diagonal,
-B below it) is read from the training residual and P from the training
preconditioners, so no network, checkpoint, or trajectory is involved.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from scipy.linalg import eigvalsh_tridiagonal, lu_factor, lu_solve
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, aslinearoperator, eigsh, splu, svds
import torch
import yaml

from data.dataset import problem_from_config
from data.generate_dataset import build_base_setup
from src.operators.train import load_config
from src.operators.train_pino import prefix_cn_objective
from src.physics.fv_preconditioner import (
    ExactCNInverse, MGOneCycleInverse, assemble_active_cn_matrix,
)
from src.physics.fv_residual import (
    FullBCData, build_homogeneous_cn_geom, cn_step_residual, explicit_cn_rhs,
    implicit_cn_action,
)


SPATIAL_CONFIGURATIONS = ("A_none", "B_spatial_mg", "C_spatial_exact")
IDENTITY_TOLERANCE = 1e-10


def configurations(lags):
    return (*SPATIAL_CONFIGURATIONS,
            *(f"D_truncated_L{lag}" for lag in lags),
            *(f"D_truncated_mg_L{lag}" for lag in lags),
            "D_sequential_mg", "exact_space_time")


def build_benchmark(spec, grid_size, study):
    """The benchmark's solver and CN geometry; grid_size None keeps the dataset grid."""
    setup = build_base_setup(1, 1, **({} if grid_size is None else {"nx": grid_size, "ny": grid_size}),
                             **({} if study["dt"] is None else {"dt": study["dt"]}))
    rng = np.random.default_rng(study["seed"])
    params = spec.sample_sim_params(rng, rng, setup["grids"], setup["time_cfg"])[0]
    solver = spec.configure_solver(params, setup["base_kwargs"])
    grids = SimpleNamespace(x_grid=setup["grids"]["x_grid"], y_grid=setup["grids"]["y_grid"])
    cfg = spec.collocation_geom_cfg(grids, {}, 300.0, 1.0, setup["dt"])
    if cfg["geometry_kind"] != "homogeneous":
        raise ValueError(f"{spec.name} has no fixed homogeneous CN operator")
    geom = build_homogeneous_cn_geom(cfg["x_grid"], cfg["y_grid"], cfg["k"], cfg["dt"],
                                     rho=cfg["rho"], cp=cfg["cp"])
    return solver, geom, cfg


def zero_boundary(geom):
    zero = torch.zeros((1, geom.Ny), dtype=torch.float64)
    return FullBCData(T_right_tilde=torch.tensor(0.0, dtype=torch.float64),
                      qL_n=zero, qL_np1=zero, qL_int=zero)


def _capacity(geom):
    return (geom.rho_cp[:-1] * geom.dx[:-1, None] * geom.dy[None, :]).numpy().ravel()


def _c_order_matrix(geom):
    nx, ny = geom.Nx - 1, geom.Ny
    order = np.arange(nx * ny).reshape((nx, ny), order="F").ravel()
    return assemble_active_cn_matrix(geom)[order][:, order].tocsr()


def dense_blocks(geom, mg_settings):
    """Dense free-node blocks in (x, y) C order.

    A and B are the residual's Jacobian blocks for the current and previous
    state, and P_mg is one zero-start V-cycle, all from the training modules.
    """
    nx, ny = geom.Nx - 1, geom.Ny
    n = nx * ny
    bc = zero_boundary(geom)
    basis = torch.eye(n, dtype=torch.float64).reshape(n, nx, ny)
    padded = torch.cat((basis, torch.zeros((n, 1, ny), dtype=torch.float64)), dim=1)
    rest = torch.zeros_like(padded)
    # Zero forcing and zero right-wall deviation make the residual linear, so
    # basis responses are exactly its Jacobian columns.
    implicit = cn_step_residual(rest, padded, geom, bc, detach_previous=False)
    explicit = -cn_step_residual(padded, rest, geom, bc, detach_previous=False)
    with torch.no_grad():
        cycle = MGOneCycleInverse(geom, **mg_settings)(basis)
    implicit, explicit, cycle = (np.ascontiguousarray(t.reshape(n, n).T.numpy())
                                 for t in (implicit, explicit, cycle))
    inverse = np.linalg.inv(implicit)
    return dict(A=implicit, B=explicit, A_inv=inverse, K=inverse @ explicit,
                P_mg=cycle, P_mg_inv=np.linalg.inv(cycle), K_mg=cycle @ explicit,
                capacity=_capacity(geom), shape=(nx, ny))


def operator_blocks(geom, mg_settings, chunk=256):
    """Sparse, LU, and V-cycle blocks for grids too large for dense spatial blocks.

    P_mg runs the training V-cycle; only P_mg^{-1}, which no training path
    needs, is factorized densely.
    """
    nx, ny = geom.Nx - 1, geom.Ny
    n = nx * ny
    implicit = _c_order_matrix(geom)
    explicit = (2.0 * sp.eye(n, format="csr") - implicit).tocsr()
    factor = splu(implicit.tocsc())
    inverse = LinearOperator((n, n), matvec=factor.solve, matmat=factor.solve,
                             rmatvec=lambda x: factor.solve(x, trans="T"),
                             rmatmat=lambda x: factor.solve(x, trans="T"), dtype=np.float64)
    mg = MGOneCycleInverse(geom, **mg_settings)

    def fields(x):
        return torch.from_numpy(np.ascontiguousarray(np.reshape(x, (n, -1)).T)).reshape(-1, nx, ny)

    def unfields(tensor, like):
        return tensor.reshape(-1, n).T.numpy().reshape(np.shape(like))

    def cycle(x):
        with torch.no_grad():
            return unfields(mg(fields(x)), x)

    def cycle_transpose(x):
        # The zero-start V-cycle V is symmetric, so P_mg^T = M V = M P_mg M^{-1}
        # at a third of the autograd cost; audit_blocks checks it against autograd.
        with torch.no_grad():
            return unfields(mg.capacity * mg(fields(x) / mg.capacity), x)

    dense = np.empty((n, n))
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        basis = np.zeros((n, stop - start))
        basis[np.arange(start, stop), np.arange(stop - start)] = 1.0
        dense[:, start:stop] = cycle(basis)
    lu = lu_factor(dense, overwrite_a=True, check_finite=False)
    cycle_inverse = LinearOperator(
        (n, n), matvec=lambda x: lu_solve(lu, x), matmat=lambda x: lu_solve(lu, x),
        rmatvec=lambda x: lu_solve(lu, x, trans=1), rmatmat=lambda x: lu_solve(lu, x, trans=1),
        dtype=np.float64)
    cycle_operator = LinearOperator((n, n), matvec=cycle, matmat=cycle, rmatvec=cycle_transpose,
                                    rmatmat=cycle_transpose, dtype=np.float64)
    return dict(A=implicit, B=explicit, A_inv=inverse, K=inverse @ aslinearoperator(explicit),
                P_mg=cycle_operator, P_mg_inv=cycle_inverse,
                K_mg=cycle_operator @ aslinearoperator(explicit),
                capacity=_capacity(geom), shape=(nx, ny))


def spatial_modes(geom, count=5):
    """Slowest and fastest decay modes, L phi = lambda M phi with M-orthonormal phi.

    A = I + dt/2 M^{-1} L with L symmetric, so K = A^{-1}B acts on the same
    modes with eigenvalues (1 - lambda dt/2) / (1 + lambda dt/2).
    """
    capacity = _capacity(geom)
    n = len(capacity)
    stiffness = (sp.diags(capacity) @ (_c_order_matrix(geom) - sp.eye(n))) * (2.0 / geom.dt)
    asymmetry = float(abs(stiffness - stiffness.T).max() / abs(stiffness).max())
    stiffness = (0.5 * (stiffness + stiffness.T)).tocsc()
    mass = sp.diags(capacity).tocsc()
    slow_rates, slow = eigsh(stiffness, k=count, M=mass, sigma=0.0, which="LM")
    fast_rates, fast = eigsh(stiffness, k=count, M=mass, which="LA")
    slow_order, fast_order = np.argsort(slow_rates), np.argsort(fast_rates)
    return dict(rates=np.concatenate((slow_rates[slow_order], fast_rates[fast_order])),
                slow=slow[:, slow_order], fast=fast[:, fast_order], asymmetry=asymmetry)


def _block(block, x, transpose=False):
    steps, n, k = x.shape
    flat = np.moveaxis(x, 1, 0).reshape(n, steps * k)
    product = (block.T if transpose else block) @ flat
    return np.moveaxis(np.asarray(product).reshape(n, steps, k), 0, 1)


def _shifted(blocks, x, transpose=False):
    out = np.zeros_like(x)
    steps = len(x)
    for lag, block in blocks.items():
        if lag >= steps:
            continue
        if transpose:
            out[:steps - lag] += _block(block, x[lag:], True)
        else:
            out[lag:] += _block(block, x[:steps - lag])
    return out


def _substitute(blocks, diagonal_inverse, y, transpose=False):
    """Solve sum_lag blocks[lag] S^lag x = y in time (identity diagonal if None).

    Steps closer together than the smallest nonzero lag are independent, so
    they are solved together as one multi-column block.
    """
    steps = len(y)
    stride = min((lag for lag in blocks if lag), default=steps)
    segments = [(start, min(start + stride, steps)) for start in range(0, steps, stride)]
    x = np.zeros_like(y)
    for start, stop in (reversed(segments) if transpose else segments):
        acc = y[start:stop].copy()
        for lag, block in blocks.items():
            if not lag:
                continue
            if transpose:
                end = min(stop, steps - lag)
                if end > start:
                    acc[:end - start] -= _block(block, x[start + lag:end + lag], True)
            elif stop > lag:
                first = max(start, lag)
                acc[first - start:] -= _block(block, x[first - lag:stop - lag])
        x[start:stop] = acc if diagonal_inverse is None else _block(diagonal_inverse, acc, transpose)
    return x


class TimeToeplitz:
    """Block lower-triangular Toeplitz operator Q^{-1} R on (steps, n, k) fields.

    Q has an identity diagonal and blocks q[lag > 0]; R has blocks r[lag >= 0]
    with r[0] invertible. Spatial blocks may be arrays, sparse matrices, or
    LinearOperators.
    """

    def __init__(self, q, r, r0_inverse):
        self.q, self.r, self.r0_inverse = q, r, r0_inverse

    def apply(self, x):
        return _substitute(self.q, None, _shifted(self.r, x))

    def apply_transpose(self, x):
        return _shifted(self.r, _substitute(self.q, None, x, transpose=True), transpose=True)

    def solve(self, y):
        return _substitute(self.r, self.r0_inverse, y + _shifted(self.q, y))

    def solve_transpose(self, y):
        z = _substitute(self.r, self.r0_inverse, y, transpose=True)
        return z + _shifted(self.q, z, transpose=True)


class SpaceTimeOperator:
    """T = P D with the transpose and inverse actions Lanczos needs."""

    def __init__(self, residual, preconditioner, steps, n):
        self.factors = [f for f in (preconditioner, residual) if f is not None]
        self.steps, self.n = steps, n
        self.size = steps * n

    def apply(self, x):
        for factor in reversed(self.factors):
            x = factor.apply(x)
        return x

    def apply_transpose(self, x):
        for factor in self.factors:
            x = factor.apply_transpose(x)
        return x

    def solve(self, y):
        for factor in self.factors:
            y = factor.solve(y)
        return y

    def solve_transpose(self, y):
        for factor in reversed(self.factors):
            y = factor.solve_transpose(y)
        return y

    def linear_operator(self, action, counter):
        def matmat(v):
            counter[0] += 1 if np.ndim(v) == 1 else np.shape(v)[1]
            x = np.asarray(v, dtype=np.float64).reshape(self.steps, self.n, -1)
            return action(x).reshape(np.shape(v))
        return LinearOperator((self.size, self.size), matvec=matmat, matmat=matmat,
                              dtype=np.float64)


def residual_operator(blocks):
    return TimeToeplitz({}, {0: blocks["A"], 1: -blocks["B"]}, blocks["A_inv"])


def _power(blocks, key, exponent):
    cache = blocks.setdefault("powers", {})
    if (key, exponent) not in cache:
        block = blocks[key]
        cache[key, exponent] = (np.linalg.matrix_power(block, exponent)
                                if isinstance(block, np.ndarray) else block ** exponent)
    return cache[key, exponent]


def preconditioner(name, blocks, steps):
    """P = Q^{-1} R for one configuration, or None for the raw loss.

    D_truncated_L keeps the first L+1 terms of D^{-1} = sum_j S^j K^j A^{-1}, so
    each corrected residual couples the L preceding intervals; the mg variants
    replace A^{-1} by the V-cycle, and D_sequential_mg keeps every term.
    """
    if name == "A_none":
        return None
    exact = "mg" not in name
    spatial, spatial_inverse = ((blocks["A_inv"], blocks["A"]) if exact
                                else (blocks["P_mg"], blocks["P_mg_inv"]))
    if name in SPATIAL_CONFIGURATIONS:
        return TimeToeplitz({}, {0: spatial}, spatial_inverse)
    step = "K" if exact else "K_mg"
    r = {0: spatial}
    if name.startswith("D_truncated"):
        lag = int(name.rsplit("_L", 1)[1]) + 1
        if lag < steps:
            r[lag] = -(_power(blocks, step, lag) @ spatial)
    return TimeToeplitz({1: -blocks[step]}, r, spatial_inverse)


def space_time_operator(name, blocks, steps):
    return SpaceTimeOperator(residual_operator(blocks), preconditioner(name, blocks, steps),
                             steps, blocks["A"].shape[0])


def identity_deviation(operator, seed):
    x = np.random.default_rng(seed).standard_normal((operator.steps, operator.n, 3))
    return float(np.abs(operator.apply(x) - x).max() / np.abs(x).max())


def estimate_spectrum(operator, modes, study, seed):
    """Extreme eigenpairs of H = T^T T.

    The largest comes from Lanczos on H and the smallest from Lanczos on
    H^{-1} = T^{-1} T^{-T}. Each estimate is repeated from a second random
    start, and every returned pair carries its residual norm against H itself,
    which bounds the distance to the nearest true eigenvalue.
    """
    counts = dict(hessian=[0], inverse=[0])
    hessian = operator.linear_operator(
        lambda x: operator.apply_transpose(operator.apply(x)), counts["hessian"])
    inverse = operator.linear_operator(
        lambda x: operator.solve(operator.solve_transpose(x)), counts["inverse"])
    starts = np.random.default_rng(seed).standard_normal((4, operator.size))
    common = dict(which="LA", ncv=min(operator.size - 1, 40))
    tic = time.perf_counter()
    top, top_vector = eigsh(hessian, k=1, tol=study["max_tolerance"], v0=starts[0], **common)
    top_restart = eigsh(hessian, k=1, tol=study["max_tolerance"], v0=starts[1],
                        return_eigenvectors=False, **common)[0]
    inverse_values, vectors = eigsh(inverse, k=min(modes, operator.size - 2),
                                    tol=study["min_tolerance"], v0=starts[2], **common)
    low_restart = eigsh(inverse, k=1, tol=study["min_tolerance"], v0=starts[3],
                        return_eigenvectors=False, **common)[0]
    elapsed = time.perf_counter() - tic
    order = np.argsort(inverse_values)[::-1]
    lowest, vectors = 1.0 / inverse_values[order], vectors[:, order]
    pairs = np.column_stack((top_vector, vectors))
    values = np.concatenate((top, lowest))
    residuals = np.linalg.norm(hessian.matmat(pairs) - pairs * values, axis=0)
    if not np.isfinite(np.concatenate((values, residuals, [top_restart, low_restart]))).all():
        raise FloatingPointError("Lanczos returned non-finite eigenpairs")
    return dict(lambda_max=float(top[0]), lambda_min=float(lowest[0]), lowest=lowest,
                vectors=vectors,
                lambda_max_residual=float(residuals[0]),
                lambda_min_residual=float(residuals[1]),
                lambda_max_restart_rel_diff=float(abs(top_restart - top[0]) / top[0]),
                lambda_min_restart_rel_diff=float(abs(1.0 / low_restart - lowest[0]) / lowest[0]),
                hessian_applications=counts["hessian"][0],
                inverse_applications=counts["inverse"][0], elapsed_seconds=elapsed)


def bidiagonal_singular_values(contraction, length):
    """Singular values of the length x length matrix I - c S (S the time shift)."""
    diagonal = np.full(length, 1.0 + contraction ** 2)
    diagonal[-1] = 1.0
    values = eigvalsh_tridiagonal(diagonal, np.full(length - 1, -contraction))
    return np.sqrt(np.clip(values, 0.0, None))


def energy_model(name, rates, dt, steps):
    """Singular-value range of PD in the capacity-weighted norm.

    When P commutes with A, M^{1/2}-weighting makes every spatial block
    symmetric with shared eigenvectors, so PD splits into one scalar bidiagonal
    time matrix per spatial mode. The extremes come from the slowest and
    fastest modes, so those rates suffice. Returns None for preconditioners
    (the V-cycle) that do not commute with A.
    """
    half = 0.5 * rates * dt
    contraction = (1.0 - half) / (1.0 + half)
    scale = np.ones_like(rates)
    lag = 1
    if name == "exact_space_time":
        return 1.0, 1.0
    if name == "A_none":
        scale = 1.0 + half
    elif name.startswith("D_truncated_L"):
        lag = int(name.rsplit("_L", 1)[1]) + 1
        contraction = contraction ** lag
    elif name != "C_spatial_exact":
        return None
    lengths = sorted({len(range(start, steps, lag)) for start in range(min(lag, steps))})
    low, high = np.inf, 0.0
    for factor, c in zip(scale, contraction):
        for length in lengths:
            values = factor * bidiagonal_singular_values(c, length)
            low, high = min(low, values.min()), max(high, values.max())
    return float(low), float(high)


def _sign_changes(trace):
    kept = trace[np.abs(trace) > 1e-8 * np.abs(trace).max()]
    return int(np.count_nonzero(np.diff(np.sign(kept))))


def mode_summaries(vectors, eigenvalues, modes, capacity, steps, dt):
    """Spatial content and time evolution of the softest error directions."""
    rows, norms = [], []
    for rank in range(vectors.shape[1]):
        field = vectors[:, rank].reshape(steps, -1)
        weighted = field * capacity
        total = (weighted * field).sum()
        slow, fast = weighted @ modes["slow"], weighted @ modes["fast"]
        step_norm = np.linalg.norm(field, axis=1)
        norms.append(step_norm)
        rows.append(dict(rank=rank + 1, lambda_H=float(eigenvalues[rank]),
                         singular_value=float(np.sqrt(eigenvalues[rank])),
                         slowest_mode_energy_fraction=float((slow[:, 0] ** 2).sum() / total),
                         slowest_five_energy_fraction=float((slow ** 2).sum() / total),
                         fastest_five_energy_fraction=float((fast ** 2).sum() / total),
                         slowest_mode_time_sign_changes=_sign_changes(slow[:, 0]),
                         fastest_mode_time_sign_changes=_sign_changes(fast[:, -1]),
                         peak_time=float((np.argmax(step_norm) + 1) * dt),
                         first_over_peak=float(step_norm[0] / step_norm.max()),
                         final_over_peak=float(step_norm[-1] / step_norm.max())))
    return rows, np.array(norms)


def _relative(actual, expected):
    return float(np.abs(actual - expected).max() / np.abs(expected).max())


def audit_blocks(solver, geom, blocks, mg_settings, seed):
    """Tie the blocks to the benchmark solver and the training residual and MG modules."""
    nx, ny = blocks["shape"]
    n = nx * ny
    rng = np.random.default_rng(seed)
    probes = rng.standard_normal((n, 3))
    order = np.arange(n).reshape((nx, ny), order="F").ravel()
    active = np.arange(geom.Nx * ny).reshape((geom.Nx, ny), order="F")[:-1].ravel(order="F")
    solver_matrix = solver.A_csr[active][:, active][order][:, order]
    implicit = blocks["A"] @ probes
    padded = torch.cat((torch.from_numpy(np.ascontiguousarray(probes.T)).reshape(3, nx, ny),
                        torch.zeros((3, 1, ny), dtype=torch.float64)), dim=1)
    training = implicit_cn_action(padded, geom).reshape(3, n).T.numpy()
    base = np.full((geom.Nx, ny), 300.0)
    perturbed = base.copy()
    perturbed[:-1] += probes[:, 0].reshape(nx, ny)
    # The solver step is affine, so the difference of two steps isolates K = A^{-1}B.
    step = (solver.cn_step(perturbed, 0.0) - solver.cn_step(base, 0.0))[:-1].ravel()
    mg = MGOneCycleInverse(geom, **mg_settings)
    field = torch.zeros((1, nx, ny), dtype=torch.float64, requires_grad=True)
    weights = rng.standard_normal(n)
    vjp = torch.autograd.grad((mg(field) * torch.from_numpy(weights).reshape(1, nx, ny)).sum(),
                              field)[0].numpy().ravel()
    cycle = blocks["P_mg"] @ probes
    error = aslinearoperator(sp.eye(n)) - aslinearoperator(blocks["P_mg"]) @ aslinearoperator(blocks["A"])
    return dict(
        residual_vs_solver_matrix_max_error=_relative(implicit, solver_matrix @ probes),
        residual_vs_training_action_max_error=_relative(implicit, training),
        explicit_is_two_minus_implicit_max_error=_relative(blocks["B"] @ probes, 2 * probes - implicit),
        solver_step_max_error=_relative(blocks["K"] @ probes[:, 0], step),
        mg_transpose_vs_autograd_max_error=_relative(blocks["P_mg"].T @ weights, vjp),
        mg_adjoint_dot_test=float(abs(weights @ cycle[:, 0] - (blocks["P_mg"].T @ weights) @ probes[:, 0])
                                  / abs(weights @ cycle[:, 0])),
        mg_inverse_max_error=_relative(blocks["P_mg_inv"] @ cycle, probes),
        mg_level_shapes=[list(shape) for shape in mg.level_shapes],
        mg_one_cycle_error_norm=float(svds(error, k=1, return_singular_vectors=False)[0]))


def dense_validation(spec, study, mg_settings, lags):
    """Check the matrix-free operators against dense products of training modules.

    On a grid small enough for dense SVD, every PD is rebuilt independently by
    pushing basis trajectories through the stacked training residual and the
    MG/exact modules, the loss Hessians come from autograd, and the sparse
    operator backend must reproduce the dense one.
    """
    grid, steps = study["validation_grid_size"], study["validation_horizon"]
    solver, geom, _ = build_benchmark(spec, grid, study)
    blocks = dense_blocks(geom, mg_settings)
    operators = operator_blocks(geom, mg_settings)
    nx, ny = blocks["shape"]
    n = nx * ny
    size = steps * n
    exact = ExactCNInverse(geom)
    mg = MGOneCycleInverse(geom, **mg_settings)
    bc = zero_boundary(geom)

    def states(u):
        batch = u.shape[0]
        free = torch.cat((u.new_zeros(batch, 1, nx, ny), u), dim=1)
        return torch.cat((free, u.new_zeros(batch, steps + 1, 1, ny)), dim=2)

    def residuals(u):
        full = states(u)
        return cn_step_residual(full[:, :-1].reshape(-1, nx + 1, ny),
                                full[:, 1:].reshape(-1, nx + 1, ny), geom, bc,
                                detach_previous=False).reshape(u.shape)

    def explicit(x):
        full = torch.cat((x, x.new_zeros(*x.shape[:-2], 1, ny)), dim=-2)
        return explicit_cn_rhs(full.reshape(-1, nx + 1, ny), geom, bc,
                               detach_previous=False).reshape(x.shape)

    def truncated(r, inverse, lag):
        term = inverse(r)
        corrected = term.clone()
        for j in range(1, min(lag, steps - 1) + 1):
            term = inverse(explicit(term))
            corrected[:, j:] += term[:, :steps - j]
        return corrected

    # D_sequential_mg is the training preconditioner itself, not a rebuild from its parts.
    transforms = {"A_none": lambda r: r, "B_spatial_mg": mg, "C_spatial_exact": exact,
                  "D_sequential_mg": mg.space_time,
                  "exact_space_time": exact.space_time}
    for lag in lags:
        transforms[f"D_truncated_L{lag}"] = lambda r, lag=lag: truncated(r, exact, lag)
        transforms[f"D_truncated_mg_L{lag}"] = lambda r, lag=lag: truncated(r, mg, lag)

    basis = torch.eye(size, dtype=torch.float64).reshape(size, steps, nx, ny)
    identity = np.eye(size).reshape(steps, n, size)
    probes = np.random.default_rng(study["seed"]).standard_normal((steps, n, 3))
    results, passed = {}, True
    with torch.no_grad():
        stacked = residuals(basis)
    for name in configurations(lags):
        with torch.no_grad():
            reference = transforms[name](stacked).reshape(size, size).T.numpy()
        operator = space_time_operator(name, blocks, steps)
        sparse = space_time_operator(name, operators, steps)
        matrix = operator.apply(identity).reshape(size, size)
        record = dict(
            operator_max_error=_relative(matrix, reference),
            transpose_max_error=_relative(operator.apply_transpose(identity).reshape(size, size),
                                          reference.T),
            inverse_max_error=float(np.abs(reference @ operator.solve(identity).reshape(size, size)
                                           - np.eye(size)).max()),
            inverse_transpose_max_error=float(np.abs(
                reference.T @ operator.solve_transpose(identity).reshape(size, size)
                - np.eye(size)).max()),
            operator_backend_max_error=max(
                _relative(getattr(sparse, action)(probes), getattr(operator, action)(probes))
                for action in ("apply", "apply_transpose", "solve", "solve_transpose")))
        singular = np.linalg.svd(reference, compute_uv=False)
        record.update(dense_sigma_max=float(singular[0]), dense_sigma_min=float(singular[-1]),
                      dense_condition_H=float((singular[0] / singular[-1]) ** 2))
        if identity_deviation(operator, study["seed"]) > IDENTITY_TOLERANCE:
            # This check validates the estimator itself, so it runs at a tight tolerance.
            estimate = estimate_spectrum(operator, study["modes"],
                                         dict(study, max_tolerance=1e-10, min_tolerance=1e-10),
                                         study["seed"])
            lowest = np.sqrt(estimate["lowest"])
            smallest = singular[::-1][:len(lowest)]
            record.update(
                sigma_max_rel_error=abs(np.sqrt(estimate["lambda_max"]) - singular[0]) / singular[0],
                sigma_min_rel_error=abs(lowest[0] - singular[-1]) / singular[-1],
                lowest_sigmas_max_rel_error=float(np.max(np.abs(lowest - smallest) / smallest)))
            accuracy = max(record["sigma_max_rel_error"], record["sigma_min_rel_error"],
                           record["lowest_sigmas_max_rel_error"])
        else:
            accuracy = abs(singular[0] - 1) + abs(singular[-1] - 1)
            record["identity_sigma_error"] = accuracy
        record["passed"] = bool(max(record["operator_max_error"], record["transpose_max_error"],
                                    record["inverse_max_error"],
                                    record["inverse_transpose_max_error"],
                                    record["operator_backend_max_error"]) <= 1e-10
                                and accuracy <= 1e-7)
        passed &= record["passed"]
        results[name] = record

    hessians = {}
    for name, loss in (
            ("A_none", lambda u: prefix_cn_objective(states(u[None]), geom, [bc] * steps,
                                                     exact, "raw")[0]),
            ("exact_space_time", lambda u: prefix_cn_objective(states(u[None]), geom, [bc] * steps,
                                                               exact, "exact")[0]),
            ("D_sequential_mg", lambda u: prefix_cn_objective(states(u[None]), geom, [bc] * steps,
                                                              mg, "mg")[0]),
            ("B_spatial_mg", lambda u: mg(residuals(u[None])).square().mean()),
            ("C_spatial_exact", lambda u: exact(residuals(u[None])).square().mean())):
        start = torch.zeros((steps, nx, ny), dtype=torch.float64)
        actual = torch.autograd.functional.hessian(loss, start).reshape(size, size).numpy()
        operator = space_time_operator(name, blocks, steps)
        expected = 2.0 * operator.apply_transpose(operator.apply(identity)).reshape(size, size) / size
        hessians[name] = _relative(actual, expected)
    passed &= max(hessians.values()) <= 1e-10
    audit = audit_blocks(solver, geom, blocks, mg_settings, study["seed"])
    passed &= max(value for key, value in audit.items() if key.endswith("_error")) <= 1e-10
    return dict(grid_size=grid, steps=steps, unknowns=size, configurations=results,
                loss_hessian_max_error=hessians, block_audit=audit, passed=bool(passed))


def model_columns(name, rates, dt, steps):
    model = energy_model(name, rates, dt, steps)
    if model is None:
        return {}
    return dict(model_sigma_min=model[0], model_sigma_max=model[1],
                model_condition_H=(model[1] / model[0]) ** 2)


def run_grid(spec, grid, horizons, builder, names, study, mg_settings, prefix):
    solver, geom, _ = build_benchmark(spec, grid, study)
    tic = time.perf_counter()
    blocks = builder(geom, mg_settings)
    modes = spatial_modes(geom)
    rates, dt = modes["rates"], geom.dt
    audit = dict(audit_blocks(solver, geom, blocks, mg_settings, study["seed"]),
                 backend=builder.__name__, setup_seconds=time.perf_counter() - tic,
                 capacity_stiffness_asymmetry=modes["asymmetry"],
                 slowest_decay_rates=rates[:5].tolist(), fastest_decay_rates=rates[5:].tolist(),
                 slowest_step_contraction=float((1 - rates[0] * dt / 2) / (1 + rates[0] * dt / 2)),
                 fastest_step_contraction=float((1 - rates[-1] * dt / 2) / (1 + rates[-1] * dt / 2)))
    n = blocks["A"].shape[0]
    rows, mode_rows, saved = [], [], {}
    for steps in horizons:
        for name in names:
            operator = space_time_operator(name, blocks, steps)
            row = dict(grid_size=geom.Nx, backend=builder.__name__, unknowns_per_step=n,
                       steps=steps, horizon_seconds=round(steps * dt, 10), unknowns=steps * n,
                       configuration=name, **model_columns(name, rates, dt, steps))
            deviation = identity_deviation(operator, study["seed"])
            row["identity_max_deviation"] = deviation
            if deviation <= IDENTITY_TOLERANCE:
                row.update(lambda_min=1.0, lambda_max=1.0, sigma_min=1.0, sigma_max=1.0,
                           condition_H=1.0, condition_PD=1.0)
                rows.append(row)
                continue
            estimate = estimate_spectrum(operator, study["modes"], study, study["seed"] + steps)
            row.update({key: estimate[key] for key in (
                "lambda_min", "lambda_max", "lambda_min_residual", "lambda_max_residual",
                "lambda_min_restart_rel_diff", "lambda_max_restart_rel_diff",
                "hessian_applications", "inverse_applications", "elapsed_seconds")})
            row.update(sigma_min=float(np.sqrt(estimate["lambda_min"])),
                       sigma_max=float(np.sqrt(estimate["lambda_max"])),
                       condition_H=estimate["lambda_max"] / estimate["lambda_min"],
                       condition_PD=float(np.sqrt(estimate["lambda_max"] / estimate["lambda_min"])))
            rows.append(row)
            print(f"grid={geom.Nx} steps={steps} {name}: kappa(H)={row['condition_H']:.4g} "
                  f"sigma_min={row['sigma_min']:.4g} sigma_max={row['sigma_max']:.4g} "
                  f"({estimate['elapsed_seconds']:.1f}s)", flush=True)
            if steps == prefix:
                summaries, norms = mode_summaries(estimate["vectors"], estimate["lowest"], modes,
                                                  blocks["capacity"], steps, dt)
                mode_rows += [dict(grid_size=geom.Nx, steps=steps, configuration=name, **summary)
                              for summary in summaries]
                saved[f"grid{geom.Nx}_{name}_step_norms"] = norms
                saved[f"grid{geom.Nx}_{name}_lambda"] = estimate["lowest"]
                if name in SPATIAL_CONFIGURATIONS:
                    saved[f"grid{geom.Nx}_{name}_vectors"] = estimate["vectors"].astype(np.float32)
    saved[f"grid{geom.Nx}_shape"] = np.array(blocks["shape"])
    saved[f"grid{geom.Nx}_slowest_spatial_mode"] = modes["slow"][:, 0]
    return geom.Nx, audit, rows, mode_rows, saved


def run_study(config, output_dir: Path):
    spec = problem_from_config(config)
    if not getattr(spec, "supports_physics_only", False):
        raise ValueError("Select a homogeneous physics-only benchmark with BENCHMARK")
    study = config["physics_test"]["conditioning"]
    mg_settings = config["physics_test"]["mg"]["solver"]
    prefix = config["physics_test"]["pino"]["prefix_steps"]
    lags = list(study["truncation_lags"])
    names = configurations(lags)
    horizons = (*study["horizons"], *study["benchmark_grid_horizons"])
    if prefix not in study["horizons"] or min(horizons) < 1:
        raise ValueError(f"horizons must be positive and include the benchmark prefix {prefix}")
    benchmark_names = [name for name in names if name in study["benchmark_grid_configurations"]]
    if (set(study["benchmark_grid_configurations"]) - set(names)
            or not set(SPATIAL_CONFIGURATIONS) <= set(benchmark_names)):
        raise ValueError(f"benchmark_grid_configurations must include {SPATIAL_CONFIGURATIONS} "
                         f"and come from {names}")
    if output_dir.exists():
        raise FileExistsError(f"Use a new isolated run directory: {output_dir}")
    output_dir.mkdir(parents=True)
    solver, _, geometry = build_benchmark(spec, None, study)
    frozen = dict(benchmark=config["benchmark"]["name"],
                  representation=config["benchmark"].get("representation", "temporal_encoder"),
                  problem_version=spec.problem_version, stage="space_time_conditioning",
                  protocol_role="development", conditioning=study, mg_solver=mg_settings,
                  benchmark_prefix_steps=prefix, benchmark_grid_size=int(solver.Nx),
                  dt=geometry["dt"], k=geometry["k"], rho=geometry["rho"], cp=geometry["cp"],
                  configurations=list(names),
                  loss="0.5*||P(Du-b)||^2 over free nodes; H=(PD)^T PD",
                  dtype="float64", device="cpu")
    frozen["protocol_sha256"] = hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()
    (output_dir / "config_used.yaml").write_text(yaml.safe_dump(frozen, sort_keys=False))
    for filename, args in (("git_commit.txt", ["git", "rev-parse", "HEAD"]),
                           ("git_status.txt", ["git", "status", "--short"])):
        (output_dir / filename).write_text(subprocess.check_output(args, cwd=PROJECT_ROOT, text=True))
    (output_dir / "source_hashes.json").write_text(json.dumps({
        str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (Path(__file__).resolve(), PROJECT_ROOT / "src/physics/fv_residual.py",
                     PROJECT_ROOT / "src/physics/fv_preconditioner.py",
                     PROJECT_ROOT / "src/physics/fv_solver_2d.py",
                     PROJECT_ROOT / "problems/forcing_homogeneous.py")}, indent=2))

    validation = dense_validation(spec, study, mg_settings, lags)
    audit = dict(dense_validation=validation)
    print(f"dense validation grid={validation['grid_size']} steps={validation['steps']} "
          f"passed={validation['passed']}", flush=True)
    cases = [(grid, study["horizons"], dense_blocks, names) for grid in study["grid_sizes"]]
    if study["benchmark_grid_horizons"]:
        cases.append((None, study["benchmark_grid_horizons"], operator_blocks, benchmark_names))
    rows, mode_rows, saved = [], [], {}
    for grid, grid_horizons, builder, grid_names in cases:
        size, grid_audit, grid_rows, grid_modes, grid_saved = run_grid(
            spec, grid, grid_horizons, builder, grid_names, study, mg_settings, prefix)
        audit[f"grid_{size}"] = grid_audit
        rows += grid_rows
        mode_rows += grid_modes
        saved.update(grid_saved)
        write_csv(output_dir / "conditioning.csv", rows)
        (output_dir / "audit.json").write_text(json.dumps(audit, indent=2))

    write_csv(output_dir / "modes.csv", mode_rows)
    np.savez_compressed(output_dir / "modes.npz", **saved)
    summary = summarize(rows, mode_rows, audit, prefix)
    (output_dir / "final_metrics.json").write_text(json.dumps(summary, indent=2))
    plot_conditioning(rows, lags, output_dir / "conditioning_vs_horizon.png")
    plot_grid_dependence(rows, lags, prefix, output_dir / f"conditioning_vs_grid_steps{prefix}.png")
    for grid in sorted({r["grid_size"] for r in rows}):
        if f"grid{grid}_A_none_vectors" in saved:
            plot_modes(saved, grid, prefix, geometry["dt"],
                       output_dir / f"softest_modes_grid{grid}_steps{prefix}.png")
    print(json.dumps(summary["prefix_condition_H"], indent=2), flush=True)
    return summary


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, mode_rows, audit, prefix):
    table = {(r["grid_size"], r["steps"], r["configuration"]): r["condition_H"] for r in rows}
    grids, prefix_condition = {}, {}
    for grid in sorted({r["grid_size"] for r in rows}):
        horizons = sorted({r["steps"] for r in rows if r["grid_size"] == grid})
        names = list(dict.fromkeys(r["configuration"] for r in rows if r["grid_size"] == grid))
        record = dict(condition_H={name: {str(steps): table[grid, steps, name] for steps in horizons}
                                   for name in names})
        if prefix in horizons:
            prefix_condition[str(grid)] = {name: table[grid, prefix, name] for name in names}
            record["prefix_ratios"] = dict(
                none_over_spatial_mg=table[grid, prefix, "A_none"] / table[grid, prefix, "B_spatial_mg"],
                none_over_spatial_exact=table[grid, prefix, "A_none"] / table[grid, prefix, "C_spatial_exact"],
                spatial_mg_over_spatial_exact=table[grid, prefix, "B_spatial_mg"]
                / table[grid, prefix, "C_spatial_exact"],
                **{f"spatial_exact_over_{name}": table[grid, prefix, "C_spatial_exact"]
                   / table[grid, prefix, name] for name in names if name.startswith("D_")})
            record["softest_modes"] = {r["configuration"]: {key: r[key] for key in (
                "singular_value", "slowest_mode_energy_fraction", "slowest_five_energy_fraction",
                "fastest_five_energy_fraction", "slowest_mode_time_sign_changes", "peak_time",
                "final_over_peak")} for r in mode_rows if r["grid_size"] == grid and r["rank"] == 1}
        if 1 in horizons:
            record["temporal_amplification"] = {
                name: {str(steps): table[grid, steps, name] / table[grid, 1, name] for steps in horizons}
                for name in names}
        grids[str(grid)] = record
    measured = [r for r in rows if "lambda_min_residual" in r]
    return dict(
        prefix_steps=prefix,
        prefix_condition_H=prefix_condition,
        grids=grids,
        estimator=dict(
            max_lambda_min_relative_residual=max(r["lambda_min_residual"] / r["lambda_min"]
                                                 for r in measured),
            max_lambda_max_relative_residual=max(r["lambda_max_residual"] / r["lambda_max"]
                                                 for r in measured),
            max_lambda_min_restart_rel_diff=max(r["lambda_min_restart_rel_diff"] for r in measured),
            max_lambda_max_restart_rel_diff=max(r["lambda_max_restart_rel_diff"] for r in measured)),
        validation_passed=audit["dense_validation"]["passed"])


TRUNCATION_COLORS = ("#eda100", "#e87ba4", "#008300", "#6250d6")


def _series(lags):
    # B is dashed because it lies on top of C wherever the V-cycle is accurate.
    return {"A_none": ("#2a78d6", "A  none", "-"), "B_spatial_mg": ("#eb6834", "B  spatial MG", "--"),
            "C_spatial_exact": ("#1baf7a", "C  exact spatial", "-"),
            **{f"D_truncated_L{lag}": (color, f"D  exact K, L={lag}", "-")
               for lag, color in zip(lags, TRUNCATION_COLORS)},
            "D_sequential_mg": ("#e34948", "D  sequential MG, all lags", "-")}


def _style(axis):
    axis.grid(True, which="major", color="#e4e3df", linewidth=0.8)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        axis.spines[spine].set_color("#8a8984")
    axis.tick_params(colors="#52514e", labelsize=9)


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_conditioning(rows, lags, path):
    plt = _pyplot()
    series = _series(lags)
    panels = (("A, B, C: spatial preconditioning", SPATIAL_CONFIGURATIONS),
              ("D: temporal coupling beyond C", ("C_spatial_exact",
                                                 *(f"D_truncated_L{lag}" for lag in lags),
                                                 "D_sequential_mg")))
    grids = sorted(grid for grid in {r["grid_size"] for r in rows}
                   if len({r["steps"] for r in rows if r["grid_size"] == grid}) > 1)
    figure, axes = plt.subplots(len(panels), len(grids), figsize=(5.4 * len(grids), 8.4),
                                squeeze=False)
    for column, grid in enumerate(grids):
        horizons = sorted({r["steps"] for r in rows if r["grid_size"] == grid})
        for row, (title, names) in enumerate(panels):
            axis = axes[row][column]
            for name in names:
                points = sorted((r["steps"], r["condition_H"]) for r in rows
                                if r["grid_size"] == grid and r["configuration"] == name)
                color, label, style = series[name]
                axis.plot(*zip(*points), color=color, linestyle=style, linewidth=2, marker="o",
                          markersize=6, markeredgecolor="white", markeredgewidth=1.2, label=label)
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.set_xticks(horizons)
            axis.set_xticklabels([str(s) for s in horizons])
            axis.minorticks_off()
            _style(axis)
            axis.set_title(f"{title}, {grid}x{grid} nodes", fontsize=10, color="#0b0b0b", loc="left")
            if column == 0:
                axis.set_ylabel("condition number of H = (PD)^T PD", color="#52514e", fontsize=9)
            if row == len(panels) - 1:
                axis.set_xlabel("CN intervals N at fixed dt", color="#52514e", fontsize=9)
            axis.legend(frameon=False, fontsize=8, labelcolor="#52514e", loc="upper left")
    figure.tight_layout()
    figure.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(figure)


def plot_grid_dependence(rows, lags, prefix, path):
    plt = _pyplot()
    series = _series(lags)
    grids = sorted({r["grid_size"] for r in rows if r["steps"] == prefix})
    figure, axis = plt.subplots(figsize=(6.4, 4.6))
    for name, (color, label, style) in series.items():
        points = sorted((r["grid_size"], r["condition_H"]) for r in rows
                        if r["steps"] == prefix and r["configuration"] == name)
        if points:
            axis.plot(*zip(*points), color=color, linestyle=style, linewidth=2, marker="o",
                      markersize=6, markeredgecolor="white", markeredgewidth=1.2, label=label)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xticks(grids)
    axis.set_xticklabels([f"{g}x{g}" for g in grids])
    axis.minorticks_off()
    _style(axis)
    axis.set_title(f"Grid dependence at the benchmark prefix, N={prefix}", fontsize=10,
                   color="#0b0b0b", loc="left")
    axis.set_xlabel("spatial grid (nodes)", color="#52514e", fontsize=9)
    axis.set_ylabel("condition number of H = (PD)^T PD", color="#52514e", fontsize=9)
    axis.legend(frameon=False, fontsize=8, labelcolor="#52514e", loc="upper left",
                bbox_to_anchor=(1.0, 1.0))
    figure.tight_layout()
    figure.savefig(path, dpi=150, facecolor="#fcfcfb", bbox_inches="tight")
    plt.close(figure)


def plot_modes(saved, grid, steps, dt, path):
    plt = _pyplot()
    from matplotlib.colors import LinearSegmentedColormap

    diverging = LinearSegmentedColormap.from_list(
        "blue_gray_red", ["#104281", "#3987e5", "#f0efec", "#e66767", "#a8302f"])
    nx, ny = saved[f"grid{grid}_shape"]
    snapshots = sorted({max(1, steps // 4), max(1, steps // 2), max(1, 3 * steps // 4), steps})
    figure, axes = plt.subplots(len(SPATIAL_CONFIGURATIONS), len(snapshots) + 1,
                                figsize=(2.3 * len(snapshots) + 5.2, 2.5 * len(SPATIAL_CONFIGURATIONS)),
                                squeeze=False,
                                gridspec_kw=dict(width_ratios=[1] * len(snapshots) + [1.7]))
    rank_colors = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
    for row, name in enumerate(SPATIAL_CONFIGURATIONS):
        field = saved[f"grid{grid}_{name}_vectors"][:, 0].reshape(steps, nx, ny)
        field = field * (1.0 if field[-1].sum() >= 0 else -1.0)
        limit = np.abs(field).max()
        for column, step in enumerate(snapshots):
            axis = axes[row][column]
            image = axis.imshow(field[step - 1].T, origin="lower", cmap=diverging, vmin=-limit,
                                vmax=limit, extent=(0, 1 - 1 / nx, 0, 1), aspect="equal")
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(f"t = {step * dt:.3g} s (step {step})", fontsize=9, color="#0b0b0b")
            if column == 0:
                sigma = np.sqrt(saved[f"grid{grid}_{name}_lambda"][0])
                axis.set_ylabel(f"{name}\nsigma_min = {sigma:.3g}", fontsize=9, color="#0b0b0b")
        axis = axes[row][-1]
        times = np.arange(1, steps + 1) * dt
        for rank, values in enumerate(saved[f"grid{grid}_{name}_step_norms"]):
            axis.plot(times, values, color=rank_colors[rank % len(rank_colors)], linewidth=2,
                      label=f"mode {rank + 1}")
        _style(axis)
        if row == 0:
            axis.set_title("per-step norm, 4 softest modes", fontsize=9, color="#0b0b0b")
            axis.legend(frameon=False, fontsize=7, labelcolor="#52514e", loc="upper left",
                        bbox_to_anchor=(1.0, 1.0))
        if row == len(SPATIAL_CONFIGURATIONS) - 1:
            axis.set_xlabel("time (s)", fontsize=9, color="#52514e")
    figure.colorbar(image, ax=axes[:, :-1], shrink=0.6, label="softest right singular vector")
    figure.suptitle(f"Softest error directions of H, {grid}x{grid} nodes, N={steps}",
                    fontsize=10, color="#0b0b0b", x=0.02, ha="left")
    figure.savefig(path, dpi=150, facecolor="#fcfcfb", bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "conf/config.yaml")
    args = parser.parse_args()
    os.environ.setdefault("BENCHMARK", "forcing_homogeneous")
    # Accelerate BLAS raises spurious floating-point flags inside matmul on macOS;
    # estimate_spectrum checks finiteness explicitly instead.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        summary = run_study(load_config(str(args.config)), args.output_dir)
    raise SystemExit(0 if summary["validation_passed"] else 2)


if __name__ == "__main__":
    main()
