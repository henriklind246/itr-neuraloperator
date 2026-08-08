"""The physical interface quantities the paper plots, in exactly one place.

Training and monitoring use an adjacent-node temperature difference
(``per_sample_node_jump_errors``, ``src/operators/losses.py``). That difference
mixes three effects: the left half-cell bulk drop, the contact discontinuity,
and the right half-cell bulk drop. The paper plots the *contact* jump,

    dT_contact(y, t) = R_c(y) * q_n(y, t),
    q_n(y, t)        = G(y) * (T_left - T_right),
    G(y)             = 1 / (h_L/k_L + R_c(y) + h_R/k_R),

which is what the finite-volume solver actually imposes at the interface face
(``src/physics/fv_solver_2d.py``).

That expression is already implemented correctly in ``visual/dataset_plots.py``.
This module **imports** it rather than restating it, so the repo has a single
contact-jump implementation, and additionally exposes the through-interface flux
``q_n`` on its own so flux can be plotted directly. A test asserts that nothing
under ``visual/pub/`` writes ``R_c * G * dT`` a second time.
"""

from __future__ import annotations

import numpy as np

from visual.dataset_plots import (
    _interface_conductance_G as interface_conductance,
    _interface_contact_jump_map as contact_jump_map,
    _interface_flanking_nodes_from_grid as flanking_nodes,
)

JUMP_DEFINITION = {
    "quantity": "contact jump at the imperfect interface",
    "formula": "dT_contact(y, t) = R_c(y) * G(y) * (T_left - T_right)",
    "conductance": "G(y) = 1 / (h_L/k_L + R_c(y) + h_R/k_R)",
    "sign": "positive means a temperature drop left -> right",
    "unit": "K",
    "source": "visual/dataset_plots.py:_interface_contact_jump_map",
    "not": "the adjacent-node difference used by the training loss",
}


SPATIAL_RC_BENCHMARKS = frozenset({"source_itr", "source_itr_sin"})


def rc_symbol(benchmark: str) -> str:
    """LaTeX symbol for ``R_c`` on axis labels, without the enclosing ``$``.

    Only the ITR benchmarks (``source_itr``, ``source_itr_sin``) sample a per-row
    resistance profile; on every other benchmark ``R_c`` is a scalar, so writing
    ``R_c(y)`` claims a dependence the case does not have.
    """
    return "R_c(y)" if benchmark in SPATIAL_RC_BENCHMARKS else "R_c"


def contact_flux_map(fields: np.ndarray, x_grid: np.ndarray, interface_x: float,
                     R_c, k_left: float, k_right: float) -> np.ndarray:
    """Through-interface normal flux ``q_n(y, t) = G(y) (T_left - T_right)``.

    ``fields`` has shape ``(Nt, Nx, Ny)``; the result has shape ``(Nt, Ny)`` and
    the same sign convention as :func:`contact_jump_map`, of which this is the
    ``R_c``-free factor.
    """
    left_node, right_node = flanking_nodes(x_grid, interface_x)
    G = interface_conductance(x_grid, interface_x, R_c, k_left, k_right)
    flux = np.asarray(G) * (fields[:, left_node, :] - fields[:, right_node, :])
    return np.asarray(flux, dtype=np.float64)


def contact_jump_profile(field: np.ndarray, x_grid: np.ndarray, interface_x: float,
                         R_c, k_left: float, k_right: float) -> np.ndarray:
    """Contact jump ``dT_contact(y)`` for a single ``(Nx, Ny)`` snapshot."""
    return contact_jump_map(np.asarray(field)[None, ...], x_grid, interface_x,
                            R_c, k_left, k_right)[0]


__all__ = [
    "JUMP_DEFINITION",
    "SPATIAL_RC_BENCHMARKS",
    "contact_flux_map",
    "contact_jump_map",
    "contact_jump_profile",
    "flanking_nodes",
    "interface_conductance",
    "rc_symbol",
]
