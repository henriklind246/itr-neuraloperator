import pytest
import torch

from src.operators.train_pino import (
    WALLS,
    _collocation_bias_counts,
    _resolve_collocation_bias,
    _validate_collocation_bias,
    sample_collocation,
)


def _sample(n_r=100, n_ic=50, n_bc=40, t_final=0.3, Nx=10, Ny=12, bias=None):
    x_grid = torch.linspace(0.0, 1.0, Nx)
    y_grid = torch.linspace(0.0, 1.0, Ny)
    gen = torch.Generator().manual_seed(0)
    coll = sample_collocation(
        n_r, n_ic, n_bc, t_final, x_grid, y_grid, torch.device("cpu"), gen,
        bias=bias,
    )
    return coll, t_final, Nx, Ny


def _interior_with_bias(bias, n_r=4000, t_final=0.3, sampler="uniform", seed=0):
    x_grid = torch.linspace(0.0, 1.0, 10)
    y_grid = torch.linspace(0.0, 1.0, 12)
    gen = torch.Generator().manual_seed(seed)
    coll = sample_collocation(
        n_r, 50, 40, t_final, x_grid, y_grid, torch.device("cpu"), gen,
        sampler=sampler, bias=bias,
    )
    return coll["interior"]


def test_interior_shapes_and_ranges():
    coll, t_final, _, _ = _sample()
    x, y, t = coll["interior"]
    for leaf in (x, y, t):
        assert leaf.shape == (1, 100, 1)
        assert leaf.requires_grad
    assert x.min() >= 0.0 and x.max() <= 1.0
    assert y.min() >= 0.0 and y.max() <= 1.0
    assert t.min() >= 0.0 and t.max() <= t_final


def test_ic_is_on_grid_nodes_at_t0():
    coll, _, Nx, Ny = _sample()
    ic = coll["ic"]
    assert ic["coords"].shape == (1, 50, 2)
    assert torch.all(ic["t"] == 0.0)
    assert ic["ix"].min() >= 0 and ic["ix"].max() < Nx
    assert ic["iy"].min() >= 0 and ic["iy"].max() < Ny
    # coords must equal the grid nodes at the sampled indices
    x_grid = torch.linspace(0.0, 1.0, Nx)
    y_grid = torch.linspace(0.0, 1.0, Ny)
    assert torch.allclose(ic["coords"][0, :, 0], x_grid[ic["ix"]])
    assert torch.allclose(ic["coords"][0, :, 1], y_grid[ic["iy"]])


def test_walls_pinned():
    coll, t_final, _, _ = _sample()
    walls = coll["walls"]
    assert set(walls.keys()) == set(WALLS)
    xl, yl, tl = walls["left"]
    assert torch.all(xl == 0.0)              # left wall x = 0
    xt, yt, tt = walls["top"]
    assert torch.all(yt == 1.0)              # top wall y = 1
    xb, yb, tb = walls["bottom"]
    assert torch.all(yb == 0.0)              # bottom wall y = 0
    for _, _, tw in (walls["left"], walls["top"], walls["bottom"]):
        assert tw.min() >= 0.0 and tw.max() <= t_final
        assert tw.requires_grad
    # right wall (x=1) is enforced by the ansatz, not sampled
    assert "right" not in walls


def test_dense_ic_covers_every_grid_node():
    Nx, Ny = 10, 12
    x_grid = torch.linspace(0.0, 1.0, Nx)
    y_grid = torch.linspace(0.0, 1.0, Ny)
    gen = torch.Generator().manual_seed(0)
    coll = sample_collocation(
        100, 50, 40, 0.3, x_grid, y_grid, torch.device("cpu"), gen, dense_ic=True,
    )
    ic = coll["ic"]
    # n_ic is ignored: the IC anchor is the full Nx*Ny grid, at t=0.
    assert ic["coords"].shape == (1, Nx * Ny, 2)
    assert torch.all(ic["t"] == 0.0)
    # every (ix, iy) node appears exactly once
    pairs = set(zip(ic["ix"].tolist(), ic["iy"].tolist()))
    assert len(pairs) == Nx * Ny
    assert pairs == {(i, j) for i in range(Nx) for j in range(Ny)}
    # coords equal the grid nodes at those indices
    assert torch.allclose(ic["coords"][0, :, 0], x_grid[ic["ix"]])
    assert torch.allclose(ic["coords"][0, :, 1], y_grid[ic["iy"]])


# --------- collocation biasing (#1 near-wall, #2 long-lead) ---------

def test_bias_none_is_rng_identical_to_default():
    # Passing bias=None must reproduce the legacy interior draw byte-for-byte.
    a = _interior_with_bias(None, seed=3)
    b = _interior_with_bias(None, seed=3)
    for u, v in zip(a, b):
        assert torch.equal(u, v)
    # And a disabled explicit spec (both fracs zero) hits the same legacy path.
    zero = {"wall_frac": 0.0, "lead_frac": 0.0, "wall_x_cut": 0.333, "lead_t_lo": 0.5}
    c = _interior_with_bias(zero, seed=3)
    for u, v in zip(a, c):
        assert torch.equal(u, v)


def test_bias_counts_sum_to_n_r_across_rounding():
    # n_full absorbs the rounding remainder so the total is always exactly n_r.
    for n_r, wf, lf in [
        (100, 0.4, 0.3),
        (127, 0.4, 0.3),   # non-divisible -> int() truncates both fracs
        (127, 0.0, 0.0),
        (1, 0.45, 0.45),   # both truncate to 0 -> n_full = 1
        (1000, 0.45, 0.45),
    ]:
        n_wall, n_lead, n_full = _collocation_bias_counts(n_r, wf, lf)
        assert n_wall == int(wf * n_r)
        assert n_lead == int(lf * n_r)
        assert n_wall + n_lead + n_full == n_r
        assert n_full >= 0
    # explicit truncation check
    assert _collocation_bias_counts(127, 0.4, 0.3) == (50, 38, 39)


def test_bias_mixture_marginals_use_effective_fractions():
    n_r = 8000
    wall_x_cut, lead_t_lo = 0.333, 0.5
    t_final = 0.3
    bias = {
        "wall_frac": 0.4, "wall_x_cut": wall_x_cut,
        "lead_frac": 0.3, "lead_t_lo": lead_t_lo,
    }
    x, y, t = _interior_with_bias(bias, n_r=n_r, t_final=t_final, seed=1)
    assert x.shape == (1, n_r, 1)
    n_wall, n_lead, n_full = _collocation_bias_counts(n_r, 0.4, 0.3)
    f_w = n_wall / n_r
    f_l = n_lead / n_r
    c = wall_x_cut
    r = 1.0 - lead_t_lo
    late_lo = lead_t_lo * t_final

    xf = x.reshape(-1)
    tf = t.reshape(-1)
    p_x = float((xf <= c).float().mean())
    p_late = float((tf >= late_lo).float().mean())
    p_corner = float(((xf <= c) & (tf >= late_lo)).float().mean())

    exp_x = f_w + (1.0 - f_w) * c
    exp_late = f_l + (1.0 - f_l) * r
    exp_corner = f_w * r + f_l * c + (1.0 - f_w - f_l) * c * r

    tol = 0.03  # finite-n sampling tolerance (~3.5 sigma at n=8000)
    assert abs(p_x - exp_x) < tol
    assert abs(p_late - exp_late) < tol
    assert abs(p_corner - exp_corner) < tol

    # per-subgroup ranges on the known wall->lead->full concatenation order
    assert float(xf[:n_wall].max()) <= c + 1e-6           # wall x-band
    assert float(tf[n_wall:n_wall + n_lead].min()) >= late_lo - 1e-6  # lead t-band
    # global ranges stay inside the unit/t_final cube
    assert float(xf.min()) >= 0.0 and float(xf.max()) <= 1.0
    assert float(tf.min()) >= 0.0 and float(tf.max()) <= t_final + 1e-6


def test_bias_interior_are_leaves_with_grad():
    bias = {"wall_frac": 0.4, "wall_x_cut": 0.333, "lead_frac": 0.3, "lead_t_lo": 0.5}
    ref, *_ = _sample()  # legacy leaves, for dtype/device parity
    rx, _, _ = ref["interior"]
    x, y, t = _interior_with_bias(bias, n_r=200, seed=2)
    for leaf in (x, y, t):
        assert leaf.shape == (1, 200, 1)
        assert leaf.requires_grad and leaf.is_leaf
        assert leaf.dtype == rx.dtype
        assert leaf.device == rx.device


def test_bias_lhs_path_in_range():
    bias = {"wall_frac": 0.4, "wall_x_cut": 0.333, "lead_frac": 0.3, "lead_t_lo": 0.5}
    x, y, t = _interior_with_bias(bias, n_r=500, t_final=0.3, sampler="lhs", seed=4)
    for leaf in (x, y):
        assert leaf.requires_grad and leaf.is_leaf
        assert float(leaf.min()) >= 0.0 and float(leaf.max()) <= 1.0
    assert float(t.min()) >= 0.0 and float(t.max()) <= 0.3 + 1e-6


@pytest.mark.parametrize(
    "bias",
    [
        {"wall_frac": 0.6, "lead_frac": 0.5},        # sum > 1
        {"wall_frac": -0.1, "lead_frac": 0.2},       # negative frac
        {"wall_frac": 0.2, "lead_frac": -0.1},
        {"wall_frac": 0.2, "wall_x_cut": 1.5},       # x_cut > 1
        {"wall_frac": 0.2, "wall_x_cut": 0.0},       # x_cut = 0 (not allowed)
        {"lead_frac": 0.2, "lead_t_lo": 1.0},        # t_lo >= 1
        {"wall_frac": float("nan"), "lead_frac": 0.2},   # non-finite
        {"lead_frac": float("inf")},                 # non-finite
    ],
)
def test_bias_invalid_specs_raise(bias):
    with pytest.raises(ValueError):
        _validate_collocation_bias(bias)


def test_bias_invalid_spec_raises_through_sampler():
    # The bad spec must surface at the sample_collocation call, not silently pass.
    bad = {"wall_frac": 0.6, "lead_frac": 0.5, "wall_x_cut": 0.333, "lead_t_lo": 0.5}
    with pytest.raises(ValueError):
        _interior_with_bias(bad, n_r=100)


def test_resolve_collocation_bias_config_wiring():
    # disabled -> None
    assert _resolve_collocation_bias({}) is None
    assert _resolve_collocation_bias({"collocation_bias": {"enabled": False}}) is None
    # enabled but both fracs zero -> None (legacy RNG-identical path)
    assert _resolve_collocation_bias(
        {"collocation_bias": {"enabled": True, "wall_frac": 0.0, "lead_frac": 0.0}}
    ) is None
    # enabled with a non-zero fraction -> validated dict
    out = _resolve_collocation_bias(
        {"collocation_bias": {
            "enabled": True, "wall_frac": 0.4, "lead_frac": 0.3,
            "wall_x_cut": 0.333, "lead_t_lo": 0.5,
        }}
    )
    assert out == {
        "wall_frac": 0.4, "wall_x_cut": 0.333, "lead_frac": 0.3, "lead_t_lo": 0.5,
    }
    # enabled with an invalid fraction -> raises (eager validation)
    with pytest.raises(ValueError):
        _resolve_collocation_bias(
            {"collocation_bias": {"enabled": True, "wall_frac": 0.7, "lead_frac": 0.5}}
        )
