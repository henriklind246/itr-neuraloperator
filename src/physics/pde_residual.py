from __future__ import annotations

from typing import Callable

import torch

# ---- Continuous autodiff residuals for the diffusion benchmark ---------------
#
# The diffusion benchmark is a single homogeneous slab with k = rho = cp = 1, so
# alpha = 1 and the PDE is  dT/dt = laplacian(T)  (a.k.a. T_t = T_xx + T_yy).
#
# Queries are in PHYSICAL units (x, y in [0, 1], t in [0, t_final]) so no
# chain-rule scale factors enter the derivatives. Temperature is globally
# normalized T_tilde = (T - mu) / sigma; the heat operator is linear, so the
# residual of T_tilde equals the residual of T divided by sigma — a query on
# T_tilde is a valid (rescaled) residual and no de-normalization is needed.
#
# ``model_xyt`` below is the thin adapter that concatenates the separate (x, y)
# leaves into the coords tensor the CViT expects, so per-variable gradients are
# unambiguous.

CoordModel = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def model_xyt(
    model,
    u: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    t: torch.Tensor,
    q_left: torch.Tensor | None = None,
) -> torch.Tensor:
    """Query the CViT with x, y as separate leaves; returns (B, Nq, out_dim).

    x, y, t are each (B, Nq, 1). They are concatenated to coords=(B, Nq, 2) and
    passed through unchanged so autograd sees x and y as distinct inputs.
    ``q_left`` is forwarded to the model's hard-left-flux lifting when set; it is
    a detached, coords-aligned (B, Nq, 1) tensor and is ignored unless the model
    was built with ``hard_left_flux=True``.
    """
    coords = torch.cat([x, y], dim=-1)
    return model(u, coords, t, q_left=q_left)


def _grad(outputs: torch.Tensor, inputs: torch.Tensor, create_graph: bool) -> torch.Tensor:
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=create_graph,
        retain_graph=True,
    )[0]


def _as_batch_leaf(coord: torch.Tensor, batch: int) -> torch.Tensor:
    """Return a ``(batch, Nq, 1)`` leaf so per-variable gradients stay per-sim.

    ``sample_collocation`` shares one ``(1, Nq, 1)`` collocation leaf across the
    whole sim minibatch. Differentiating a shared leaf makes autograd accumulate
    ``d/dcoord`` over the batch (``grad_outputs=ones`` reduces the broadcast rows
    to their sum), so every derivative residual collapses to ``sum_b r[b]`` and
    the sims' physics couple. Expanding to an independent per-sim leaf here keeps
    each sim's derivative separate; an already per-sim leaf (``batch`` rows) and
    the ``batch == 1`` case pass through untouched.
    """
    if coord.shape[0] == batch:
        return coord
    if coord.shape[0] != 1:
        raise ValueError(
            f"coord batch {coord.shape[0]} is neither 1 nor u batch {batch}"
        )
    leaf = coord.detach().expand(batch, -1, -1).contiguous()
    return leaf.requires_grad_(bool(coord.requires_grad))


def diffusion_residual(
    model,
    u: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    t: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Return r = T_t - alpha (T_xx + T_yy) at the given points; shape (B, Nq, 1).

    x, y, t must be leaf tensors with requires_grad=True. The second-order graph
    is built with create_graph=True so the residual loss is itself
    differentiable w.r.t. the model parameters. Shared ``(1, Nq, 1)`` collocation
    leaves are expanded to per-sim leaves so the returned residual is per-sim
    ``(B, Nq, 1)`` rather than summed over the batch.
    """
    B = u.shape[0]
    x = _as_batch_leaf(x, B)
    y = _as_batch_leaf(y, B)
    t = _as_batch_leaf(t, B)
    T = model_xyt(model, u, x, y, t)
    T_t = _grad(T, t, create_graph=True)
    T_x = _grad(T, x, create_graph=True)
    T_xx = _grad(T_x, x, create_graph=True)
    T_y = _grad(T, y, create_graph=True)
    T_yy = _grad(T_y, y, create_graph=True)
    return T_t - alpha * (T_xx + T_yy)


def neumann_residual(
    model,
    u: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    t: torch.Tensor,
    wall: str,
) -> torch.Tensor:
    """Zero-flux wall residual; shape (B, Nq, 1).

    ``wall`` selects the derivative that must vanish:
      - "left"   (x=0): dT/dx
      - "top"    (y=1): dT/dy
      - "bottom" (y=0): dT/dy
    The right wall (x=1) is handled by the hard-Dirichlet ansatz, not here.
    Shared collocation leaves are expanded per-sim so the residual is per-sim.
    """
    B = u.shape[0]
    x = _as_batch_leaf(x, B)
    y = _as_batch_leaf(y, B)
    t = _as_batch_leaf(t, B)
    T = model_xyt(model, u, x, y, t)
    if wall == "left":
        return _grad(T, x, create_graph=True)
    if wall in ("top", "bottom"):
        return _grad(T, y, create_graph=True)
    raise ValueError(f"Unknown wall {wall!r}; expected 'left', 'top', or 'bottom'.")


def forcing_neumann_residual(
    model,
    u: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    t: torch.Tensor,
    q_L: torch.Tensor,
    sigma: float,
    k: float = 1.0,
) -> torch.Tensor:
    """Inhomogeneous left-wall (x=0) Neumann residual; shape (B, Nq, 1).

    The FV solver injects ``q_L`` as the INWARD heat flux at x=0, i.e. the
    physical wall condition is ``-k dT/dx|_{x=0} = q_L`` (see fv_solver_2d). With
    the global normalization ``T = mu + sigma * T_tilde`` (the model outputs
    ``T_tilde``), that becomes

        dT_tilde/dx|_{x=0} + q_L(y, t) / (k * sigma) = 0,

    so this returns ``T_tilde_x + q_L / (k * sigma)``. ``k = 1`` for the single
    homogeneous slab (K_SLAB), which keeps the interior operator alpha = 1.

    ``q_L`` is a PRECOMPUTED constant tensor (B, Nq, 1) aligned to the left-wall
    collocation points ``(y, t)``; autograd only needs dT/dx, never a graph
    through ``q_L``. Queries are in physical units (x in [0, 1]) so no coordinate
    rescale enters ``T_tilde_x``.

    ``q_L`` is also forwarded to the model as ``q_left`` so that, when the model
    uses the hard-left-flux lifting, the ``-q_L/(k*sigma) * (x - 1)`` particular
    term is present and this residual collapses to the forcing-independent
    ``raw_x(0)``. It is inert when the model has no lifting.

    Shared collocation leaves are expanded per-sim so ``dT/dx`` is not summed
    over the batch; ``q_L`` is already ``(B, Nq, 1)`` and aligns row-for-row.
    """
    B = u.shape[0]
    x = _as_batch_leaf(x, B)
    y = _as_batch_leaf(y, B)
    t = _as_batch_leaf(t, B)
    T = model_xyt(model, u, x, y, t, q_left=q_L)
    T_x = _grad(T, x, create_graph=True)
    return T_x + q_L / (float(k) * float(sigma))


def ic_residual(
    model,
    u: torch.Tensor,
    coords_ic: torch.Tensor,
    t_ic: torch.Tensor,
    ic_target_tilde: torch.Tensor,
) -> torch.Tensor:
    """Initial-condition residual T(x, y, 0) - T0_tilde; shape (B, Nq, out_dim).

    coords_ic:(B or 1, Nq, 2), t_ic:(B or 1, Nq, 1) all zeros, ic_target_tilde:
    (B, Nq, out_dim) the normalized IC sampled at those points.
    """
    T = model(u, coords_ic, t_ic)
    return T - ic_target_tilde
