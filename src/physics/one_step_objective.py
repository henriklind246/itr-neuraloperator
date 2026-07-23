"""Pure, differentiable one-step Crank-Nicolson physics for InterfaceCViT.

This module is the single source of truth for the state-conditioned one-step
Markov objective (plan Section 1). It collects the proven Crank-Nicolson (CN)
tensor operators and interface-flux closures that previously lived in the
`scripts/diagnose_fv_*` diagnostics, and adds the differentiable objective
functions the production runner needs:

  - `variational_objective` / `variational_energy` -- the CN energy
    ``J(u) = 1/2 u^T A u - b^T u`` whose free-DOF gradient is ``A u - b``.
  - `defect_terms` -- a differentiable Jacobi-preconditioned discrete defect
    ``||D^{-1/2}(A u - b)||^2`` (single sweep) replacing the non-autograd SciPy
    reference used for the direct-state experiment.
  - `one_step_objective` -- a ``kind in {variational, defect}`` dispatcher.

Everything here is torch/autograd only (no SciPy, no solver stepping). The CN
operator is represented by a ``cn`` dict of coupling coefficients
``{r_w, r_e, r_s, r_n, capacity, forcing}`` built either from a numpy FV solver
(`build_cn_tensors`) or from a batched per-sample `FVGeom`
(`build_cn_tensors_from_geom`) so per-sample ``interface_x`` / ``R_c`` flow
through. The CN action functions are batch-agnostic: ``r_*`` may be ``(Nx, Ny)``
(single problem) or ``(B, Nx, Ny)`` (per-sample geometry); the temperature
fields are always ``(B, Nx, Ny)``.

Conservation is split into hard (structurally enforced) and diagnostic (logged):
`conservative_energy_storage_projection` applies a right-wall-vanishing storage
correction (hard global storage + right Dirichlet), while
`one_step_interface_trace_terms` reports the diagnostic interface-flux and
contact-law residuals.

`build_fixed_solver` and the SciPy `rollout_residual_rms` reference are NOT here:
they are solver-construction / non-differentiable fixtures and stay in the
diagnostic scripts.
"""

from __future__ import annotations

import numpy as np
import torch

from src.physics.fv_residual import (
    FVGeom,
    interface_trace_constraint_residuals,
)

_CN_KEYS = ("r_w", "r_e", "r_s", "r_n")


# --------------------------------------------------------------------------- #
# CN coefficient builders
# --------------------------------------------------------------------------- #
def build_cn_tensors(
    solver,
    *,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """CN coupling coefficients + normalized forcing table from a numpy solver.

    ``r_*`` are ``(Nx, Ny)`` (single geometry); ``capacity`` is ``(Nx-1, Ny)``
    (``rho*cp*dx*dy`` over the active DOFs); ``forcing`` is a
    ``(len(t)-1, Ny)`` table of the CN-averaged normalized left-flux increment
    for every interval, indexed per step by the objective.
    """
    tensors = {
        name: torch.as_tensor(getattr(solver, name), device=device, dtype=dtype)
        for name in _CN_KEYS
    }
    capacity = (
        solver.rho_nodes[: solver.Nx - 1]
        * solver.cp_nodes[: solver.Nx - 1]
        * solver.dx[: solver.Nx - 1, None]
        * solver.dy[None, :]
    )
    tensors["capacity"] = torch.as_tensor(capacity, device=device, dtype=dtype)

    forcing = []
    rho_cp_left = solver.rho_nodes[0] * solver.cp_nodes[0]
    for n in range(len(solver.t) - 1):
        t_n = float(solver.t[n])
        if solver.q_left_integral is not None:
            q_integral = np.asarray(
                solver.q_left_integral(t_n, t_n + solver.dt), dtype=np.float64
            )
            if q_integral.ndim == 0:
                q_integral = np.full(solver.Ny, float(q_integral))
            increment = 2.0 * q_integral / (
                rho_cp_left * solver.hx * float(sigma)
            )
        else:
            q_n = np.asarray(solver.q_left(t_n), dtype=np.float64)
            q_np1 = np.asarray(solver.q_left(t_n + solver.dt), dtype=np.float64)
            if q_n.ndim == 0:
                q_n = np.full(solver.Ny, float(q_n))
            if q_np1.ndim == 0:
                q_np1 = np.full(solver.Ny, float(q_np1))
            increment = solver.dt * (q_n + q_np1) / (
                rho_cp_left * solver.hx * float(sigma)
            )
        forcing.append(increment)
    tensors["forcing"] = torch.as_tensor(
        np.stack(forcing), device=device, dtype=dtype
    )
    return tensors


def build_cn_tensors_from_geom(
    geom: FVGeom,
    forcing_increment: torch.Tensor,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> dict[str, torch.Tensor]:
    """CN coefficient dict from a batched per-sample `FVGeom` (plan Section 1).

    Unlike `build_cn_tensors` (single numpy solver), the per-interface geometry
    carries batched ``r_*`` of shape ``(B, Nx, Ny)`` so distinct ``interface_x``
    / ``R_c`` per sample flow into the CN action. ``forcing_increment`` is the
    already-normalized CN-averaged left-flux increment for the single one-step
    interval, shape ``(B, Ny)`` (or ``(Ny,)`` broadcast); it is stored under
    ``forcing`` and added at the left column of the explicit RHS.

    ``capacity`` is ``rho_cp * dx * dy`` over the active DOFs ``[:Nx-1]``,
    broadcast to ``(B, Nx-1, Ny)`` so the variational energy nondimensionalizes
    per sample.
    """
    dev = device
    dt_type = dtype
    r_tensors = {}
    for name in _CN_KEYS:
        t = getattr(geom, name)
        if dev is not None or dt_type is not None:
            t = t.to(device=dev, dtype=dt_type)
        r_tensors[name] = t
    ref = r_tensors["r_w"]
    dev = ref.device
    dt_type = ref.dtype

    Nx, Ny = int(geom.Nx), int(geom.Ny)
    dx = geom.dx.to(device=dev, dtype=dt_type)
    if dx.dim() == 1:
        dx = dx.unsqueeze(0)
    B = dx.shape[0]
    dy = geom.dy.to(device=dev, dtype=dt_type)
    rho_cp = geom.rho_cp.to(device=dev, dtype=dt_type)
    if rho_cp.dim() == 2:
        rho_cp = rho_cp.unsqueeze(0)
    capacity_full = rho_cp * dx[:, :, None] * dy[None, None, :]      # (B, Nx, Ny)
    capacity = capacity_full[:, : Nx - 1, :].expand(B, Nx - 1, Ny).contiguous()

    forcing = torch.as_tensor(forcing_increment, device=dev, dtype=dt_type)
    if forcing.dim() == 1:
        forcing = forcing.unsqueeze(0)
    r_tensors["capacity"] = capacity
    r_tensors["forcing"] = forcing
    return r_tensors


# --------------------------------------------------------------------------- #
# CN operator action (batch-agnostic in the coefficient tensors)
# --------------------------------------------------------------------------- #
def implicit_cn_action(
    deviation_np1: torch.Tensor,
    cn: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Apply the implicit CN operator ``A`` to the active DOFs ``[:Nx-1]``.

    ``deviation_np1`` is ``(B, Nx, Ny)`` (relative to the right-wall value);
    the returned action is ``(B, Nx-1, Ny)``. ``r_*`` may be ``(Nx, Ny)`` or
    ``(B, Nx, Ny)`` -- the x-axis is sliced with ellipsis indexing so a single
    geometry broadcasts over the batch and a per-sample geometry lines up.
    """
    nx = deviation_np1.shape[-2]
    active = deviation_np1[..., : nx - 1, :]
    rw, re, rs, rn = (cn[name] for name in _CN_KEYS)
    out = (
        1.0
        + rw[..., : nx - 1, :]
        + re[..., : nx - 1, :]
        + rs[..., : nx - 1, :]
        + rn[..., : nx - 1, :]
    ) * active
    out[..., 1 : nx - 1, :] = out[..., 1 : nx - 1, :] - (
        rw[..., 1 : nx - 1, :] * deviation_np1[..., : nx - 2, :]
    )
    out = out - re[..., : nx - 1, :] * deviation_np1[..., 1:nx, :]
    out[..., 1:] = out[..., 1:] - rs[..., : nx - 1, 1:] * active[..., :-1]
    out[..., :-1] = out[..., :-1] - rn[..., : nx - 1, :-1] * active[..., 1:]
    return out


def _explicit_cn_base(deviation_n: torch.Tensor, cn: dict[str, torch.Tensor]) -> torch.Tensor:
    """Explicit CN action (RHS without the forcing injection), ``(B, Nx-1, Ny)``.

    The source state is detached: ``T_n`` never receives gradient through the
    target-step objective (plan Section 4 on-manifold conditioning).
    """
    state = deviation_n.detach()
    nx = state.shape[-2]
    active = state[..., : nx - 1, :]
    rw, re, rs, rn = (cn[name] for name in _CN_KEYS)
    rhs = (
        1.0
        - rw[..., : nx - 1, :]
        - re[..., : nx - 1, :]
        - rs[..., : nx - 1, :]
        - rn[..., : nx - 1, :]
    ) * active
    rhs[..., 1 : nx - 1, :] = rhs[..., 1 : nx - 1, :] + (
        rw[..., 1 : nx - 1, :] * state[..., : nx - 2, :]
    )
    rhs = rhs + re[..., : nx - 1, :] * state[..., 1:nx, :]
    rhs[..., 1:] = rhs[..., 1:] + rs[..., : nx - 1, 1:] * active[..., :-1]
    rhs[..., :-1] = rhs[..., :-1] + rn[..., : nx - 1, :-1] * active[..., 1:]
    return rhs


def explicit_cn_rhs(
    deviation_n: torch.Tensor,
    step_indices: torch.Tensor,
    cn: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Explicit CN RHS ``b`` with the per-step forcing table injection.

    ``cn['forcing']`` is a ``(n_steps, Ny)`` table; ``step_indices`` selects the
    interval per batch element (the diagnostic multi-interval contract).
    """
    rhs = _explicit_cn_base(deviation_n, cn)
    rhs[..., 0, :] = rhs[..., 0, :] + cn["forcing"].index_select(0, step_indices)
    return rhs


def explicit_cn_rhs_one_step(
    deviation_n: torch.Tensor,
    cn: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Explicit CN RHS ``b`` for a single one-step interval.

    ``cn['forcing']`` is ``(B, Ny)`` (this interval's normalized increment); it
    is added directly at the left column, no per-step table indexing.
    """
    rhs = _explicit_cn_base(deviation_n, cn)
    forcing = cn["forcing"]
    rhs[..., 0, :] = rhs[..., 0, :] + forcing
    return rhs


def cn_free_diagonal(cn: dict[str, torch.Tensor], nx: int) -> torch.Tensor:
    """Diagonal of the implicit CN operator on the active DOFs ``[:Nx-1]``."""
    rw, re, rs, rn = (cn[name] for name in _CN_KEYS)
    return (
        1.0
        + rw[..., : nx - 1, :]
        + re[..., : nx - 1, :]
        + rs[..., : nx - 1, :]
        + rn[..., : nx - 1, :]
    )


# --------------------------------------------------------------------------- #
# Objectives
# --------------------------------------------------------------------------- #
def variational_energy(
    u: torch.Tensor,
    apply_A,
    b: torch.Tensor,
) -> torch.Tensor:
    """Generic CN variational energy ``J(u) = 1/2 u^T A u - b^T u``.

    ``apply_A`` is any callable returning ``A u``. For a symmetric positive
    definite ``A`` the gradient ``dJ/du = A u - b`` (used by the minimizer-parity
    tests). Contraction is over all but the leading batch axis.
    """
    Au = apply_A(u)
    dims = tuple(range(1, u.dim()))
    return 0.5 * (u * Au).sum(dim=dims) - (b * u).sum(dim=dims)


def causal_variational_terms(
    prediction_np1: torch.Tensor,
    prediction_n: torch.Tensor,
    step_indices: torch.Tensor,
    cn: dict[str, torch.Tensor],
    *,
    right_value: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CN variational energy + residual RMS for the multi-interval table path.

    Preserves the diagnostic contract: mean capacity-weighted energy over the
    batch and the residual RMS. ``prediction_*`` are ``(B, Nx, Ny)`` in
    normalized temperature; deviations are taken relative to ``right_value``.
    """
    deviation_np1 = prediction_np1 - float(right_value)
    deviation_n = prediction_n - float(right_value)
    implicit = implicit_cn_action(deviation_np1, cn)
    rhs = explicit_cn_rhs(deviation_n, step_indices, cn)
    capacity = cn["capacity"]
    energy = (
        0.5 * (capacity * deviation_np1[..., :-1, :] * implicit).sum(dim=(-2, -1))
        - (capacity * deviation_np1[..., :-1, :] * rhs).sum(dim=(-2, -1))
    ) / capacity.sum()
    residual = implicit - rhs
    return energy.mean(), residual.square().mean().sqrt()


def variational_objective(
    prediction_np1: torch.Tensor,
    prediction_n: torch.Tensor,
    cn: dict[str, torch.Tensor],
    *,
    right_value: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One-step CN variational energy per sample + the CN residual ``A u - b``.

    ``cn['forcing']`` is the ``(B, Ny)`` single-interval increment. The energy is
    nondimensionalized per sample by the active-DOF capacity so cases of
    different forcing amplitude / material / time state are comparable. Returns
    ``(energy_per_sample, residual)`` with ``residual`` shape ``(B, Nx-1, Ny)``.
    """
    deviation_np1 = prediction_np1 - float(right_value)
    deviation_n = prediction_n - float(right_value)
    implicit = implicit_cn_action(deviation_np1, cn)
    rhs = explicit_cn_rhs_one_step(deviation_n, cn)
    capacity = cn["capacity"]
    active = deviation_np1[..., :-1, :]
    norm = capacity.sum(dim=(-2, -1)).clamp_min(1e-12)
    quad = 0.5 * (capacity * active * implicit).sum(dim=(-2, -1))
    lin = (capacity * active * rhs).sum(dim=(-2, -1))
    energy = (quad - lin) / norm
    residual = implicit - rhs
    return energy, residual


def defect_terms(
    prediction_np1: torch.Tensor,
    prediction_n: torch.Tensor,
    cn: dict[str, torch.Tensor],
    *,
    right_value: float,
    sweeps: int = 1,
    omega: float = 2.0 / 3.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable Jacobi-preconditioned discrete defect per sample.

    Single sweep (``sweeps == 1``) returns the exact preconditioned residual
    energy ``mean_free ||D^{-1/2}(A u - b)||^2`` -- the objective that succeeded
    in the direct-state experiment. Multi-sweep runs the documented Jacobi error
    recurrence ``z_0 = 0; z_{k+1} = z_k + omega D^{-1}(r - A z_k)`` with the
    D-weighted error-energy objective ``mean_free (z^T D z)`` (proportional to the
    single-sweep form for one sweep up to ``omega^2``); it is intended only after
    single-sweep minimizer parity is demonstrated.

    Returns ``(energy_per_sample, residual)`` with ``residual = A u - b``.
    """
    deviation_np1 = prediction_np1 - float(right_value)
    deviation_n = prediction_n - float(right_value)
    implicit = implicit_cn_action(deviation_np1, cn)
    rhs = explicit_cn_rhs_one_step(deviation_n, cn)
    residual = implicit - rhs
    nx = deviation_np1.shape[-2]
    diag = cn_free_diagonal(cn, nx).clamp_min(1e-12)
    if int(sweeps) <= 1:
        precond = residual / diag.sqrt()
        energy = precond.square().mean(dim=(-2, -1))
        return energy, residual

    z = torch.zeros_like(residual)
    for _ in range(int(sweeps)):
        az = _apply_free_operator(z, cn)
        z = z + float(omega) * (residual - az) / diag
    energy = (z * diag * z).mean(dim=(-2, -1))
    return energy, residual


def _apply_free_operator(
    z_free: torch.Tensor, cn: dict[str, torch.Tensor]
) -> torch.Tensor:
    """Apply ``A`` to a free-DOF field ``(B, Nx-1, Ny)`` with a pinned wall.

    The right-wall DOF is Dirichlet (deviation 0), so ``z_free`` is padded with a
    zero column before the implicit action and the active output is returned.
    """
    pad = z_free.new_zeros(z_free.shape[:-2] + (1, z_free.shape[-1]))
    z_full = torch.cat((z_free, pad), dim=-2)
    return implicit_cn_action(z_full, cn)


def one_step_objective(
    kind: str,
    prediction_np1: torch.Tensor,
    prediction_n: torch.Tensor,
    cn: dict[str, torch.Tensor],
    *,
    right_value: float,
    defect_sweeps: int = 1,
    defect_omega: float = 2.0 / 3.0,
    case_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Dispatch to the ``variational`` or ``defect`` one-step objective.

    Returns ``(loss_scalar, metrics)`` where ``loss_scalar`` is the batch-mean
    objective and ``metrics`` carries the detached per-sample energy, the
    physical CN defect norm (RMS of ``A u - b``), and the per-case energy vector
    for gradient-domination diagnostics. Optional detached positive
    ``case_weights`` recondition a shared multi-state optimization without
    changing any individual state's physical minimizer.
    """
    if kind == "variational":
        energy, residual = variational_objective(
            prediction_np1, prediction_n, cn, right_value=right_value
        )
    elif kind == "defect":
        energy, residual = defect_terms(
            prediction_np1,
            prediction_n,
            cn,
            right_value=right_value,
            sweeps=defect_sweeps,
            omega=defect_omega,
        )
    else:
        raise ValueError(
            f"one_step_objective kind must be 'variational' or 'defect'; got {kind!r}"
        )
    if case_weights is None:
        loss = energy.mean()
    else:
        weights = case_weights.detach().to(device=energy.device, dtype=energy.dtype)
        if weights.shape != energy.shape:
            raise ValueError(
                f"case_weights must have shape {tuple(energy.shape)}; got "
                f"{tuple(weights.shape)}."
            )
        if not bool(torch.isfinite(weights).all()) or bool((weights <= 0.0).any()):
            raise ValueError("case_weights must be finite and strictly positive")
        loss = (weights * energy).mean()
    metrics = {
        "energy_per_case": energy.detach(),
        "energy": energy.detach().mean(),
        "defect_rms": residual.detach().square().mean().sqrt(),
    }
    return loss, metrics


# --------------------------------------------------------------------------- #
# State conditioning
# --------------------------------------------------------------------------- #
def state_conditioned_spatial(
    state: torch.Tensor,
    fixed_material_channels: torch.Tensor,
) -> torch.Tensor:
    """Stack the current normalized state with the fixed material channels."""
    return torch.cat((state[:, None], fixed_material_channels), dim=1)


# --------------------------------------------------------------------------- #
# Interface-flux closures (hard storage projection + diagnostics)
# --------------------------------------------------------------------------- #
def left_energy_interface_flux_closure(
    state_n: torch.Tensor,
    smooth_np1: torch.Tensor,
    geom,
    q_left_integral: torch.Tensor,
    *,
    resistance: float,
    sigma: float,
) -> torch.Tensor:
    """Solve the scalar uniform-in-y interface flux from left-layer energy."""
    batch_size = state_n.shape[0]
    face = int(geom.face_idx.reshape(-1)[0].item())
    dx = geom.dx
    if dx.dim() == 1:
        dx = dx.unsqueeze(0).expand(batch_size, -1)
    rho_cp = geom.rho_cp
    if rho_cp.dim() == 2:
        rho_cp = rho_cp.unsqueeze(0).expand(batch_size, -1, -1)
    dy = geom.dy.to(state_n)
    capacity = rho_cp.to(state_n) * dx.to(state_n)[:, :, None] * dy[None, None]
    left_capacity = capacity[:, : face + 1].sum(dim=(1, 2))
    smooth_storage = (
        float(sigma)
        * capacity[:, : face + 1]
        * (smooth_np1[:, : face + 1] - state_n[:, : face + 1])
    ).sum(dim=(1, 2))
    conductance = geom.G_x[:, face].to(state_n)
    q_series_n = conductance * float(sigma) * (
        state_n[:, face] - state_n[:, face + 1]
    )
    previous_interface_energy = 0.5 * float(geom.dt) * (
        q_series_n * dy[None]
    ).sum(dim=1)
    injected_energy = (q_left_integral.to(state_n) * dy[None]).sum(dim=1)
    interface_height = dy.sum()
    denominator = (
        float(resistance) * left_capacity
        + 0.5 * float(geom.dt) * interface_height
    )
    return (
        injected_energy - smooth_storage - previous_interface_energy
    ) / denominator


def two_sided_energy_interface_flux_closure(
    state_n: torch.Tensor,
    smooth_np1: torch.Tensor,
    geom,
    q_left_integral: torch.Tensor,
    *,
    resistance: float,
    right_value: float,
    q_ref: float,
    sigma: float,
) -> torch.Tensor:
    """Least-squares flux from normalized left and right layer balances."""
    batch_size = state_n.shape[0]
    face = int(geom.face_idx.reshape(-1)[0].item())
    dx = geom.dx
    if dx.dim() == 1:
        dx = dx.unsqueeze(0).expand(batch_size, -1)
    rho_cp = geom.rho_cp
    if rho_cp.dim() == 2:
        rho_cp = rho_cp.unsqueeze(0).expand(batch_size, -1, -1)
    dy = geom.dy.to(state_n)
    capacity = rho_cp.to(state_n) * dx.to(state_n)[:, :, None] * dy[None, None]
    delta = float(sigma) * capacity * (smooth_np1 - state_n)
    delta_left = delta[:, : face + 1].sum(dim=(1, 2))
    delta_right = delta[:, face + 1 : geom.Nx - 1].sum(dim=(1, 2))

    conductance = geom.G_x[:, face].to(state_n)
    q_series_n = conductance * float(sigma) * (
        state_n[:, face] - state_n[:, face + 1]
    )
    previous_interface_energy = 0.5 * float(geom.dt) * (
        q_series_n * dy[None]
    ).sum(dim=1)
    injected_energy = (q_left_integral.to(state_n) * dy[None]).sum(dim=1)
    height = dy.sum()
    half_step_area = 0.5 * float(geom.dt) * height
    left_capacity = capacity[:, : face + 1].sum(dim=(1, 2))
    coefficient_left = float(resistance) * left_capacity + half_step_area
    target_left = injected_energy - delta_left - previous_interface_energy

    right_face = geom.G_x[:, geom.Nx - 2].to(state_n)
    q_right_n = right_face * float(sigma) * (
        state_n[:, geom.Nx - 2] - float(right_value)
    )
    q_right_np1 = right_face * float(sigma) * (
        smooth_np1[:, geom.Nx - 2] - float(right_value)
    )
    right_outflow_energy = 0.5 * float(geom.dt) * (
        (q_right_n + q_right_np1) * dy[None]
    ).sum(dim=1)
    coefficient_right = half_step_area
    target_right = delta_right + right_outflow_energy - previous_interface_energy

    characteristic = float(q_ref) * float(geom.dt) * height
    scale_left = (injected_energy.abs() + characteristic).clamp_min(1.0e-12)
    scale_right = characteristic.expand_as(target_right).clamp_min(1.0e-12)
    weight_left = coefficient_left / scale_left
    weight_right = coefficient_right / scale_right
    numerator = (
        weight_left * target_left / scale_left
        + weight_right * target_right / scale_right
    )
    denominator = weight_left.square() + weight_right.square()
    return numerator / denominator.clamp_min(1.0e-12)


def conservative_energy_storage_projection(
    state_n: torch.Tensor,
    smooth_np1: torch.Tensor,
    geom,
    q_left_integral: torch.Tensor,
    *,
    resistance: float,
    right_value: float,
    sigma: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Satisfy both layer energies using flux plus a trace-neutral storage mode.

    Hard conservation (plan Section 1): the correction is carried by a bubble
    mode that vanishes at the right Dirichlet wall, so the fixed-temperature BC
    is never moved while global storage balance is enforced.
    """
    flux = left_energy_interface_flux_closure(
        state_n,
        smooth_np1,
        geom,
        q_left_integral,
        resistance=float(resistance),
        sigma=float(sigma),
    )
    batch_size = state_n.shape[0]
    face = int(geom.face_idx.reshape(-1)[0].item())
    dx = geom.dx
    if dx.dim() == 1:
        dx = dx.unsqueeze(0).expand(batch_size, -1)
    rho_cp = geom.rho_cp
    if rho_cp.dim() == 2:
        rho_cp = rho_cp.unsqueeze(0).expand(batch_size, -1, -1)
    dy = geom.dy.to(state_n)
    capacity = rho_cp.to(state_n) * dx.to(state_n)[:, :, None] * dy[None, None]

    right_start = face + 1
    right_stop = int(geom.Nx) - 2
    if right_stop - right_start < 2:
        raise ValueError("right layer is too narrow for a trace-neutral storage mode")
    phase = torch.linspace(
        0.0,
        1.0,
        right_stop - right_start + 1,
        device=state_n.device,
        dtype=state_n.dtype,
    )
    bubble = state_n.new_zeros((1, int(geom.Nx), 1))
    bubble[:, right_start : right_stop + 1, 0] = torch.sin(
        torch.pi * phase
    ).square()
    bubble[:, right_start, 0] = 0.0
    bubble[:, right_stop, 0] = 0.0

    delta = float(sigma) * capacity * (smooth_np1 - state_n)
    delta_right = delta[:, right_start : int(geom.Nx) - 1].sum(dim=(1, 2))
    conductance = geom.G_x[:, face].to(state_n)
    q_series_n = conductance * float(sigma) * (
        state_n[:, face] - state_n[:, face + 1]
    )
    previous_interface_energy = 0.5 * float(geom.dt) * (
        q_series_n * dy[None]
    ).sum(dim=1)
    right_face = geom.G_x[:, int(geom.Nx) - 2].to(state_n)
    q_right_n = right_face * float(sigma) * (
        state_n[:, int(geom.Nx) - 2] - float(right_value)
    )
    q_right_np1 = right_face * float(sigma) * (
        smooth_np1[:, int(geom.Nx) - 2] - float(right_value)
    )
    right_outflow_energy = 0.5 * float(geom.dt) * (
        (q_right_n + q_right_np1) * dy[None]
    ).sum(dim=1)
    half_step_area = 0.5 * float(geom.dt) * dy.sum()
    target_right = delta_right + right_outflow_energy - previous_interface_energy
    storage_sensitivity = (
        float(sigma) * capacity * bubble
    )[:, right_start : int(geom.Nx) - 1].sum(dim=(1, 2))
    amplitude = (
        half_step_area * flux - target_right
    ) / storage_sensitivity.clamp_min(1.0e-12)
    correction = amplitude[:, None, None] * bubble
    corrected_target_right = target_right + storage_sensitivity * amplitude
    storage_residual = flux - corrected_target_right / half_step_area.clamp_min(1.0e-12)
    return flux, smooth_np1 + correction, correction, storage_residual


# --------------------------------------------------------------------------- #
# Interface-physics diagnostics
# --------------------------------------------------------------------------- #
def one_step_interface_trace_terms(
    prediction_np1: torch.Tensor,
    geom,
    x_grid: np.ndarray,
    resistance: float,
    *,
    k_left: float,
    k_right: float,
    sigma: float,
    q_ref: float,
) -> dict[str, torch.Tensor]:
    """Diagnostic one-sided interface flux and contact-law residuals."""
    traces = interface_trace_constraint_residuals(
        prediction_np1,
        prediction_np1,
        geom,
        x_grid,
        [float(resistance)],
        k_left=float(k_left),
        k_right=float(k_right),
        sigma_global=float(sigma),
        q_ref=float(q_ref),
    )
    flux = traces["flux"][:, -1]
    contact = traces["contact"][:, -1]
    return {
        "flux_loss": flux.square().mean(),
        "contact_loss": contact.square().mean(),
        "flux_rms": flux.square().mean().sqrt(),
        "contact_rms": contact.square().mean().sqrt(),
        "q_series_rms": traces["q_series"][:, -1].square().mean().sqrt(),
        "jump_rms_K": traces["jump_K"][:, -1].square().mean().sqrt(),
    }


def interface_flux_head_physics_terms(
    state_n: torch.Tensor,
    prediction_np1: torch.Tensor,
    interface_flux_np1_normalized: torch.Tensor,
    geom,
    q_left_integral: torch.Tensor,
    *,
    right_value: float,
    q_ref: float,
    sigma: float,
) -> dict[str, torch.Tensor]:
    """Constrain the explicit flux head by FV flux and both layer balances."""
    batch_size = state_n.shape[0]
    face = int(geom.face_idx.reshape(-1)[0].item())
    dx = geom.dx
    if dx.dim() == 1:
        dx = dx.unsqueeze(0).expand(batch_size, -1)
    rho_cp = geom.rho_cp
    if rho_cp.dim() == 2:
        rho_cp = rho_cp.unsqueeze(0).expand(batch_size, -1, -1)
    dy = geom.dy.to(state_n)
    capacity = rho_cp.to(state_n) * dx.to(state_n)[:, :, None] * dy[None, None]
    conductance = geom.G_x[:, face].to(state_n)
    q_series_n = conductance * float(sigma) * (
        state_n[:, face] - state_n[:, face + 1]
    )
    q_series_np1 = conductance * float(sigma) * (
        prediction_np1[:, face] - prediction_np1[:, face + 1]
    )
    q_head_np1 = float(q_ref) * interface_flux_np1_normalized
    series_residual = (q_head_np1 - q_series_np1) / float(q_ref)

    q_left_energy = (q_left_integral.to(state_n) * dy[None]).sum(dim=1)
    q_interface_energy = 0.5 * float(geom.dt) * (
        (q_series_n + q_head_np1) * dy[None]
    ).sum(dim=1)
    delta = float(sigma) * capacity * (prediction_np1 - state_n)
    delta_left = delta[:, : face + 1].sum(dim=(1, 2))
    delta_right = delta[:, face + 1 : geom.Nx - 1].sum(dim=(1, 2))

    right_face = geom.G_x[:, geom.Nx - 2].to(state_n)
    q_right_n = right_face * float(sigma) * (
        state_n[:, geom.Nx - 2] - float(right_value)
    )
    q_right_np1 = right_face * float(sigma) * (
        prediction_np1[:, geom.Nx - 2] - float(right_value)
    )
    q_right_energy = 0.5 * float(geom.dt) * (
        (q_right_n + q_right_np1) * dy[None]
    ).sum(dim=1)

    height = dy.sum()
    characteristic = float(q_ref) * float(geom.dt) * height
    left_scale = (q_left_energy.abs() + characteristic).clamp_min(1.0e-12)
    right_scale = characteristic.expand_as(delta_right).clamp_min(1.0e-12)
    left_residual = (
        delta_left - q_left_energy + q_interface_energy
    ) / left_scale
    right_residual = (
        delta_right - q_interface_energy + q_right_energy
    ) / right_scale
    return {
        "series_loss": series_residual.square().mean(),
        "left_energy_loss": left_residual.square().mean(),
        "right_energy_loss": right_residual.square().mean(),
        "series_rms": series_residual.square().mean().sqrt(),
        "left_energy_rms": left_residual.square().mean().sqrt(),
        "right_energy_rms": right_residual.square().mean().sqrt(),
        "q_head_rms": q_head_np1.square().mean().sqrt(),
        "q_series_rms": q_series_np1.square().mean().sqrt(),
    }


__all__ = [
    "build_cn_tensors",
    "build_cn_tensors_from_geom",
    "implicit_cn_action",
    "explicit_cn_rhs",
    "explicit_cn_rhs_one_step",
    "cn_free_diagonal",
    "variational_energy",
    "causal_variational_terms",
    "variational_objective",
    "defect_terms",
    "one_step_objective",
    "state_conditioned_spatial",
    "left_energy_interface_flux_closure",
    "two_sided_energy_interface_flux_closure",
    "conservative_energy_storage_projection",
    "one_step_interface_trace_terms",
    "interface_flux_head_physics_terms",
]
