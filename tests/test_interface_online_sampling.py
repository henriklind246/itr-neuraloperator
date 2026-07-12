import numpy as np
import pytest

from problems.base import ProblemSpec
from problems.interfaces import (
    INTERFACE_X_RANGE,
    RC_RANGE,
    InterfacesProblem,
)

"""
Stage 7 gates for `online` interface collocation sampling
(`InterfacesProblem.sample_online_params`). These check that fresh IID draws
carry the SAME schema the saved LHS path emits (so the same channel/forcing
builders and `validate_schema` consume them), draw from the correct marginal
target ranges, keep `interface_x` off grid nodes, and are reproducible per RNG.
The base `ProblemSpec` default must refuse online sampling loudly.
"""

REQUIRED_KEYS = (
    "R_c", "interface_x", "T0", "ic_family", "ic_params",
    "temporal_family", "temporal_params", "spatial_family", "spatial_params",
)


def _grids(Nx=24, Ny=20, a=0.0, b=1.0, c=0.0, d=1.0):
    x_grid = np.linspace(a, b, Nx)
    y_grid = np.linspace(c, d, Ny)
    X, Y = np.meshgrid(x_grid, y_grid, indexing="ij")
    return {"X": X, "Y": Y, "x_grid": x_grid, "y_grid": y_grid}


def _time_cfg():
    return {"dt": 0.005, "t_final": 0.3, "b": 1.0, "T_right": 300.0,
            "t_on": 0.0, "t_off": 0.2, "phase": 0.0, "tukey_alpha": 0.5}


def test_online_schema_matches_saved_rows():
    """Online rows carry every key the saved LHS rows do (same downstream
    channel/forcing builders and validate_schema consume them)."""
    spec = InterfacesProblem()
    grids, tcfg = _grids(), _time_cfg()
    rng = np.random.default_rng(0)
    params = spec.sample_online_params(rng, 8, grids, tcfg)

    assert len(params) == 8
    for p in params:
        for k in REQUIRED_KEYS:
            assert k in p, f"online row missing key {k!r}"
        assert p["temporal_family"] == "sin"
        assert p["spatial_family"] == "uniform"
        Nx, Ny = grids["X"].shape
        assert np.asarray(p["T0"]).shape == (Nx, Ny)

    # validate_schema (used by the training/eval loops) accepts the rows.
    arr = np.array(params, dtype=object)
    spec.validate_schema(arr, np.arange(len(params)))


def test_online_draws_within_target_ranges():
    """R_c and interface_x fall inside their marginal target ranges; the small
    off-node jitter keeps interface_x within a fraction of a cell of the range."""
    spec = InterfacesProblem()
    grids, tcfg = _grids(), _time_cfg()
    rng = np.random.default_rng(1)
    params = spec.sample_online_params(rng, 512, grids, tcfg)

    R_c = np.array([p["R_c"] for p in params])
    ix = np.array([p["interface_x"] for p in params])
    hx = float(grids["x_grid"][1] - grids["x_grid"][0])

    assert R_c.min() >= RC_RANGE[0] and R_c.max() <= RC_RANGE[1]
    # jitter can nudge interface_x by up to ~0.25*hx past a range endpoint.
    assert ix.min() >= INTERFACE_X_RANGE[0] - 0.5 * hx
    assert ix.max() <= INTERFACE_X_RANGE[1] + 0.5 * hx


def test_online_interface_x_off_grid_nodes():
    """No sampled interface_x sits on a grid node (the FV interface face must be
    unambiguous)."""
    spec = InterfacesProblem()
    grids, tcfg = _grids(), _time_cfg()
    rng = np.random.default_rng(2)
    params = spec.sample_online_params(rng, 512, grids, tcfg)

    x_grid = grids["x_grid"]
    hx = float(x_grid[1] - x_grid[0])
    node_tol = 1e-3 * hx
    for p in params:
        dist = np.min(np.abs(x_grid - float(p["interface_x"])))
        assert dist > node_tol


def test_online_reproducible_and_iid():
    """Same seed -> identical draws; different seeds -> different draws (fresh IID
    each call, not the fixed saved design)."""
    spec = InterfacesProblem()
    grids, tcfg = _grids(), _time_cfg()

    a = spec.sample_online_params(np.random.default_rng(7), 16, grids, tcfg)
    b = spec.sample_online_params(np.random.default_rng(7), 16, grids, tcfg)
    c = spec.sample_online_params(np.random.default_rng(8), 16, grids, tcfg)

    ax = np.array([p["interface_x"] for p in a])
    bx = np.array([p["interface_x"] for p in b])
    cx = np.array([p["interface_x"] for p in c])
    ar = np.array([p["R_c"] for p in a])
    cr = np.array([p["R_c"] for p in c])

    np.testing.assert_array_equal(ax, bx)
    assert not np.array_equal(ax, cx)
    assert not np.array_equal(ar, cr)


def test_online_matches_saved_sim_params_schema():
    """The online rows expose exactly the keys the saved sample_sim_params rows
    do, so both feed the same builders."""
    spec = InterfacesProblem()
    grids = _grids()
    tcfg = _time_cfg()

    saved = spec.sample_sim_params(
        np.random.default_rng(3), np.random.default_rng(4), grids,
        {**tcfg, "num_sims": 8, "lhs_seed": 0},
    )
    online = spec.sample_online_params(np.random.default_rng(5), 8, grids, tcfg)

    assert set(online[0].keys()) == set(saved[0].keys())


def test_base_spec_refuses_online_sampling():
    """A ProblemSpec that does not override sample_online_params must raise
    rather than silently reuse saved params."""

    class _Bare(ProblemSpec):
        name = "bare"

        def sample_sim_params(self, *a, **k):  # pragma: no cover - unused
            return []

        def configure_solver(self, *a, **k):  # pragma: no cover - unused
            return None

        def build_item(self, *a, **k):  # pragma: no cover - unused
            return {}

        def validate_schema(self, *a, **k):  # pragma: no cover - unused
            return None

    with pytest.raises(NotImplementedError):
        _Bare().sample_online_params(
            np.random.default_rng(0), 4, _grids(), _time_cfg()
        )
