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
    model, u: torch.Tensor, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor
) -> torch.Tensor:
    """Query the CViT with x, y as separate leaves; returns (B, Nq, out_dim).

    x, y, t are each (B, Nq, 1). They are concatenated to coords=(B, Nq, 2) and
    passed through unchanged so autograd sees x and y as distinct inputs.
    """
    coords = torch.cat([x, y], dim=-1)
    return model(u, coords, t)


def _grad(outputs: torch.Tensor, inputs: torch.Tensor, create_graph: bool) -> torch.Tensor:
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=create_graph,
        retain_graph=True,
    )[0]


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
    differentiable w.r.t. the model parameters.
    """
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
    """
    T = model_xyt(model, u, x, y, t)
    if wall == "left":
        return _grad(T, x, create_graph=True)
    if wall in ("top", "bottom"):
        return _grad(T, y, create_graph=True)
    raise ValueError(f"Unknown wall {wall!r}; expected 'left', 'top', or 'bottom'.")


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
