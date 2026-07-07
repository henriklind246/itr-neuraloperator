import torch

from src.operators.train_pino import WALLS, sample_collocation


def _sample(n_r=100, n_ic=50, n_bc=40, t_final=0.3, Nx=10, Ny=12):
    x_grid = torch.linspace(0.0, 1.0, Nx)
    y_grid = torch.linspace(0.0, 1.0, Ny)
    gen = torch.Generator().manual_seed(0)
    coll = sample_collocation(
        n_r, n_ic, n_bc, t_final, x_grid, y_grid, torch.device("cpu"), gen
    )
    return coll, t_final, Nx, Ny


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
