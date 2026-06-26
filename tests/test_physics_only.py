import numpy as np
import torch
import pytest

from problems.diffusion import DiffusionProblem, K_SLAB, T_RIGHT
from problems.forcing import FORCING_TEMPORAL_SAMPLES
from src.physics.fv_solver_2d import FVSolver2D, Layer2D
from src.physics.fv_residual import (
    build_face_conductances,
    build_cn_geom,
    full_bc_cn_residual,
    FullBCData,
)
from src.operators.rollout import build_rollout_item_from_base
from src.operators.train import _collocation_batch_loss, train_one_epoch

"""
Physics-only / W2-collocation correctness gates for the `diffusion` benchmark.

These exercise the same `full_bc` residual + collocation machinery that `forcing`
uses, but on the homogeneous force-free geometry, plus the diffusion-specific
additions: the explicit IC anchor (required to break the constant-300 degeneracy)
and the physics-only training loop (`lambda_data == 0`) that must NEVER iterate the
data loader. The capability dispatch (the three ProblemSpec collocation hooks) is
asserted directly so nothing branches on the benchmark name.
"""


# --- 1. consecutive FV truth -> ~0 full_bc residual (homogeneous geometry) ---


def _build_homogeneous_solver(dt=0.005, t_final=0.1, Nx=40, Ny=40):
    """Single homogeneous slab (k = K_SLAB), zero left flux (flux_A=0), adiabatic
    top/bottom, right Dirichlet — the `diffusion` geometry built directly so the
    residual gate is independent of the spec's solver wiring."""
    layers = [Layer2D(x_left=0.0, x_right=1.0, rho=1, cp=1, k=K_SLAB)]
    return FVSolver2D(
        a=0.0, b=1.0, c=0.0, d=1.0, Nx=Nx, Ny=Ny,
        layers=layers, interface_R=None, lam_target=0.5, dt=dt,
        flux_f=2.0, flux_A=0.0, t_on=0.0, t_off=0.2,
        t_final=t_final, phase=0.0,
    )


def test_full_bc_residual_zero_on_homogeneous_truth():
    """`full_bc_cn_residual` on consecutive solver snapshots of the homogeneous
    force-free sim is at the direct-solve floor in EVERY region, using the
    homogeneous geom (k_left==k_right==K_SLAB, R_c=0) and zero left flux. A
    non-uniform IC makes the interior evolution non-trivial so the gate is not
    vacuous."""
    sim = _build_homogeneous_solver()
    X, Y = np.meshgrid(sim.grid_x, sim.grid_y, indexing="ij")
    T0 = (300.0 + 40.0 * np.exp(-((X - 0.4) ** 2 + (Y - 0.5) ** 2) / 0.02)).astype(np.float64)
    _, _, _, T_hist = sim.solve(T0=T0, store_trajectory=True)

    geom = build_cn_geom(
        sim.grid_x, sim.grid_y,
        k_left=K_SLAB, k_right=K_SLAB, interface_x=0.5, R_c=0.0,
        dt=sim.dt, sigma_global=1.0, dtype=torch.float64,
    )

    region_max = {}
    for n in (1, 3, 5, 10):
        T_n = torch.as_tensor(T_hist[n], dtype=torch.float64)
        T_np1 = torch.as_tensor(T_hist[n + 1], dtype=torch.float64)
        tnp1 = float(sim.t[n + 1])
        bc = FullBCData(
            T_right_tilde=torch.as_tensor(float(sim.T_right(tnp1)), dtype=torch.float64),
            qL_n=torch.as_tensor(sim.q_left(float(sim.t[n])), dtype=torch.float64),
            qL_np1=torch.as_tensor(sim.q_left(tnp1), dtype=torch.float64),
        )
        parts = full_bc_cn_residual(T_n, T_np1, geom, bc)
        for name, vec in parts.items():
            region_max[name] = max(region_max.get(name, 0.0), float(vec.abs().max()))

    for name, m in region_max.items():
        assert m < 1e-7, f"homogeneous full_bc region {name} residual floor too high: {m:.3e}"


# --- 2. homogeneous equivalence: k_left==k_right, R_c=0 -> uniform k/hx ------


def test_homogeneous_face_conductances_uniform():
    """With k_left==k_right==K_SLAB and R_c=0 every x-face conductance (including
    the would-be interface face) collapses to the ordinary uniform k/hx, so no
    homogeneous-specific residual path is needed."""
    Nx, Ny = 20, 12
    x_grid = np.linspace(0.0, 1.0, Nx)
    y_grid = np.linspace(0.0, 1.0, Ny)
    G_x, G_y, dx, dy = build_face_conductances(
        x_grid, y_grid,
        k_left=K_SLAB, k_right=K_SLAB, interface_x=0.5, R_c=0.0,
        dtype=torch.float64,
    )
    hx = float(x_grid[1] - x_grid[0])
    np.testing.assert_allclose(G_x.numpy(), K_SLAB / hx, rtol=0, atol=1e-12)


# --- 3. IC anchor loss is zero iff the prediction equals T_0 -----------------


class _IdentityModel(torch.nn.Module):
    """Returns the source temperature channel (channel 0). The `(1 + w)` factor
    keeps a trainable parameter in the graph (so backward/optimizer.step have a
    gradient to act on) without changing the value at w == 0."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))

    def forward(self, spatial, cond_static, forcing_seq=None):
        return spatial[..., 0:1] * (1.0 + self.w)


class _StubColloc:
    """Minimal collocation sampler exposing the surface `_collocation_batch_loss`
    and the physics-only loop touch: `geom_cfg`, `sample_batch` (6-tuple), and
    `sample_anchor_batch` ((batch, target)). The homogeneous geom + zero flux keep
    the residual finite; the anchor target is the base field plus `anchor_offset`
    so `ic_loss == anchor_offset**2`."""

    def __init__(self, B=2, Nx=5, Ny=5, dt=0.01, seed=0, anchor_offset=0.0):
        self.B, self.Nx, self.Ny = B, Nx, Ny
        xg = np.linspace(0.0, 1.0, Nx)
        yg = np.linspace(0.0, 1.0, Ny)
        self.geom_cfg = {
            "x_grid": xg, "y_grid": yg,
            "k_left": K_SLAB, "k_right": K_SLAB,
            "interface_x": float(xg[Nx // 2]),
            "dt": dt, "sigma_global": 1.0, "T_right_tilde": 0.0,
        }
        self.rng = np.random.default_rng(seed)
        self.anchor_offset = float(anchor_offset)

    def _mk_batch(self, field):
        spatial = np.zeros((self.B, self.Nx, self.Ny, 4), np.float32)
        spatial[..., 0] = field
        return {
            "spatial": torch.from_numpy(spatial),
            "cond_static": torch.zeros(self.B, 2),
            "forcing_seq": torch.zeros(self.B, 128, 2),
        }

    def sample_batch(self, max_lead):
        f = (self.rng.standard_normal((self.B, self.Nx, self.Ny)) * 0.01).astype(np.float32)
        z = torch.zeros(self.B, self.Ny)
        return (self._mk_batch(f), self._mk_batch(f), torch.zeros(self.B),
                z, z.clone(), z.clone())

    def sample_anchor_batch(self):
        f = self.rng.standard_normal((self.B, self.Nx, self.Ny)).astype(np.float32)
        batch = self._mk_batch(f)
        target = torch.from_numpy((f + self.anchor_offset)[..., None].astype(np.float32))
        return batch, target


def test_ic_loss_zero_when_prediction_matches_T0():
    """`ic_loss == 0` exactly when the anchor prediction reproduces the base field
    (identity model, zero offset); a unit offset gives `ic_loss == 1`, confirming
    the term is a genuine MSE against T_0 and not a constant."""
    model = _IdentityModel()
    device = torch.device("cpu")

    sampler0 = _StubColloc(anchor_offset=0.0)
    _, _, ic0 = _collocation_batch_loss(model, sampler0, 0.05, None, device, lambda_ic=1.0)
    assert ic0 is not None
    assert float(ic0) == 0.0

    sampler1 = _StubColloc(anchor_offset=1.0)
    _, _, ic1 = _collocation_batch_loss(model, sampler1, 0.05, None, device, lambda_ic=1.0)
    assert abs(float(ic1) - 1.0) < 1e-5

    # lambda_ic == 0 must not build/forward an anchor batch (ic_loss is None).
    _, _, ic_none = _collocation_batch_loss(model, sampler0, 0.05, None, device, lambda_ic=0.0)
    assert ic_none is None


# --- 4/6. physics-only loop: skip-proof + optimization mechanics -------------


class _ExplodingLoader:
    """A `train_loader` that reports a length but raises if iterated. Proves the
    physics-only branch reads only `__len__` and never `iter()`s the data path."""

    def __init__(self, n):
        self.n = int(n)

    def __len__(self):
        return self.n

    def __iter__(self):
        raise AssertionError("data loader must not be iterated in physics-only mode")


def test_physics_only_skips_data_loader():
    """`train_one_epoch(lambda_data=0, lambda_physics>0, lambda_ic>0)` with a
    collocation sampler completes against a loader whose `__iter__` raises, and
    reports finite physics + IC metrics (data metrics fall through to 0.0)."""
    model = _IdentityModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    sampler = _StubColloc()
    loader = _ExplodingLoader(4)

    metrics = train_one_epoch(
        model=model, train_loader=loader, optimizer=optimizer,
        loss_fn=lambda *a, **k: None, device=torch.device("cpu"),
        physics_collocation=sampler, collocation_max_lead=0.05,
        lambda_data=0.0, lambda_physics=1.0, lambda_ic=1.0,
    )

    assert np.isfinite(metrics["physics_loss"])
    assert np.isfinite(metrics["ic_loss"])
    assert metrics["physics_loss"] > 0.0  # the small random fields give a real residual
    assert metrics["rel_l2"] == 0.0       # no data pass happened


def test_physics_only_optimization_mechanics(monkeypatch):
    """The physics-only loop runs exactly `len(train_loader)` optimizer steps and
    honors `grad_clip` (one clip per step), reusing the data path's
    zero_grad -> backward -> clip -> step mechanics. No AMP/scaler is involved."""
    model = _IdentityModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    sampler = _StubColloc()
    n_steps = 5
    loader = _ExplodingLoader(n_steps)

    counts = {"step": 0, "clip": 0}
    real_step = optimizer.step

    def counting_step(*a, **k):
        counts["step"] += 1
        return real_step(*a, **k)

    optimizer.step = counting_step
    monkeypatch.setattr(
        torch.nn.utils, "clip_grad_norm_",
        lambda *a, **k: counts.__setitem__("clip", counts["clip"] + 1),
    )

    train_one_epoch(
        model=model, train_loader=loader, optimizer=optimizer,
        loss_fn=lambda *a, **k: None, device=torch.device("cpu"),
        grad_clip=1.0,
        physics_collocation=sampler, collocation_max_lead=0.05,
        lambda_data=0.0, lambda_physics=1.0, lambda_ic=1.0,
    )

    assert counts["step"] == n_steps
    assert counts["clip"] == n_steps


# --- 5. fail-fast when no loss term is active --------------------------------


def test_fail_fast_no_active_loss_term():
    """With physics enabled but every weight zero, the collocation gate is False
    (mode needs lambda_physics>0 or lambda_ic>0) and `train_one_epoch` raises
    rather than silently optimizing nothing."""
    model = _IdentityModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    sampler = _StubColloc()
    loader = _ExplodingLoader(2)

    with pytest.raises(ValueError, match="no active loss term"):
        train_one_epoch(
            model=model, train_loader=loader, optimizer=optimizer,
            loss_fn=lambda *a, **k: None, device=torch.device("cpu"),
            physics_collocation=sampler, collocation_max_lead=0.05,
            lambda_data=0.0, lambda_physics=0.0, lambda_ic=0.0,
        )


# --- 7. collocation hooks return the homogeneous / on-grid contract ----------


class _HookDS:
    def __init__(self, Nx=10, Ny=8):
        self.x_grid = np.linspace(0.0, 1.0, Nx)
        self.y_grid = np.linspace(0.0, 1.0, Ny)
        self.Ny = Ny


def test_collocation_hooks_contract():
    spec = DiffusionProblem()
    ds = _HookDS()

    geom = spec.collocation_geom_cfg(ds, {}, mu_global=300.0, sigma_global=10.0, dt=0.01)
    assert geom["k_left"] == geom["k_right"] == K_SLAB
    # interface_x is pinned to an interior x-grid node (irrelevant once k_left==k_right).
    assert ds.x_grid[0] < geom["interface_x"] < ds.x_grid[-1]
    assert geom["T_right_tilde"] == (T_RIGHT - 300.0) / 10.0

    R_c, qLn, qLnp1, qLint = spec.collocation_closure(ds, 0, {}, 0.0, 0.01)
    assert R_c == 0.0
    for v in (qLn, qLnp1, qLint):
        assert v.shape == (ds.Ny,)
        assert float(np.abs(v).max()) == 0.0

    bp = spec.collocation_base_plan(ds, 0.01)
    assert bp["base_snapshot_index"] == 0
    assert bp["on_grid_pairs"] is True


# --- 8. collocation_source=train confines draws to the train sim ids ---------


def test_collocation_draws_confined_to_train_ids(monkeypatch):
    """A sampler built over the training split draws only training sim ids (the
    mechanism that makes `collocation_source=train` keep val_rel_l2 honest); the
    drawn set is disjoint from the validation ids. Heavy item builders are stubbed
    so only the RNG-driven sim selection is exercised."""
    import src.operators.train as train_mod
    from src.operators.train import CollocationSampler

    spec = DiffusionProblem()
    n_t = 11
    dt = 0.01
    t_grid = np.arange(n_t) * dt
    train_ids = np.array([0, 1, 2, 3, 4, 5, 6])
    val_ids = np.array([7, 8, 9])

    class _DS:
        def __init__(self, ids):
            self.x_grid = np.linspace(0.0, 1.0, 4)
            self.y_grid = np.linspace(0.0, 1.0, 5)
            self.Ny = 5
            self.t_grid = t_grid
            self.t_final = float(t_grid[-1])
            self.sim_ids = ids
            self.sim_params = {int(i): {} for i in range(10)}
            self.problem = self

        def build_item(self, ds, sid, s, j):
            return {"spatial": np.zeros((4, 5, 1), dtype=np.float32)}

    draws = []
    monkeypatch.setattr(
        train_mod, "build_rollout_item_from_base",
        lambda base_item, ds, spc, sid, current, t_lo, t_hi: (
            draws.append(int(sid)) or {"x": np.zeros(1, np.float32)}
        ),
    )
    monkeypatch.setattr(train_mod, "collate_fn", lambda items: items)

    ds = _DS(train_ids)
    geom_cfg = spec.collocation_geom_cfg(ds, {}, 300.0, 10.0, dt)
    base_plan = spec.collocation_base_plan(ds, dt)
    sampler = CollocationSampler(
        ds, spec, geom_cfg, batch_size=8, dt=dt, rng_seed=0, base_plan=base_plan,
    )
    for _ in range(20):
        sampler.sample_batch(max_lead=0.05)

    drawn = set(draws)
    assert drawn, "sampler drew no sim ids"
    assert drawn.issubset(set(train_ids.tolist()))
    assert drawn.isdisjoint(set(val_ids.tolist()))


# --- 9. zero-duration anchor item (t_s == t_j) is finite ---------------------


class _StubDiffDS:
    """In-memory dataset stub exposing exactly the attributes `build_item` reads,
    so the IC-anchor (zero-duration, t_s == t_j) build path can be exercised
    without a real on-disk dataset."""

    def __init__(self, Nx=6, Ny=5, n_t=8, dt=0.01):
        self.Nx, self.Ny = Nx, Ny
        self.t_grid = np.arange(n_t, dtype=float) * dt
        self.t_final = float(self.t_grid[-1])
        self.noise_std = 0.0
        self.mu_global = 300.0
        self.sigma_global = 10.0
        self.temporal_samples = FORCING_TEMPORAL_SAMPLES
        x = np.linspace(0.0, 1.0, Nx)
        y = np.linspace(0.0, 1.0, Ny)
        X, Y = np.meshgrid(x, y, indexing="ij")
        self.X_norm = X.astype(np.float32)
        self.Y_norm = Y.astype(np.float32)
        rng = np.random.default_rng(0)
        self.trajectories = (
            300.0 + 20.0 * rng.standard_normal((3, n_t, Nx, Ny))
        ).astype(np.float64)


def test_zero_duration_anchor_item_is_finite():
    """Building the IC anchor at t_s == t_j (both via `build_item` with s == j and
    via `build_rollout_item_from_base(..., t_s, t_s)`) yields all-finite spatial /
    cond_static / forcing_seq — no NaN/inf from a t_bar == 0 division."""
    spec = DiffusionProblem()
    ds = _StubDiffDS()

    base_item = spec.build_item(ds, sid=0, s=0, j=0)
    for key in ("spatial", "cond_static", "forcing_seq", "Y"):
        assert np.isfinite(base_item[key]).all(), f"{key} not finite at t_s == t_j"
    # t_bar_norm must be exactly 0 for the zero-duration anchor.
    assert base_item["cond_static"][0] == 0.0

    t_s = float(ds.t_grid[0])
    current = base_item["spatial"][..., 0]
    rolled = build_rollout_item_from_base(base_item, ds, spec, 0, current, t_s, t_s)
    for key in ("spatial", "cond_static", "forcing_seq"):
        assert np.isfinite(rolled[key]).all(), f"rolled {key} not finite at t_s == t_j"
    assert rolled["cond_static"][0] == 0.0


# --- 10. on_grid dt pinning tolerates float32 jitter, rejects real non-uniform -


class _GridDS:
    """Minimal dataset stub exposing the attributes `_build_collocation_sampler`
    and `CollocationSampler.__init__` read, parameterized by an arbitrary t_grid
    so the on-grid dt-pinning check can be exercised directly."""

    def __init__(self, t_grid):
        self.t_grid = np.asarray(t_grid)
        self.t_final = float(self.t_grid[-1])
        self.x_grid = np.linspace(0.0, 1.0, 4)
        self.y_grid = np.linspace(0.0, 1.0, 5)
        self.Ny = 5
        self.sim_ids = np.array([0, 1, 2])
        self.sim_params = {i: {} for i in range(3)}
        self.problem = None


def test_on_grid_dt_pinning_tolerates_float32_jitter():
    """A genuinely uniform grid saved as float32 jitters by the float32 quantum
    near t_final; the on-grid dt-pinning must accept it and pin dt to the mean
    spacing rather than raising 'non-uniform'."""
    import src.operators.train as train_mod

    spec = DiffusionProblem()
    # 31 snapshots at 0.01 spacing up to t_final = 0.30, round-tripped through
    # float32 so consecutive diffs jitter by ~1e-8 (the storage quantum).
    t_grid = (np.arange(31, dtype=np.float64) * 0.01).astype(np.float32)
    diffs = np.diff(t_grid.astype(np.float64))
    assert diffs.max() - diffs.min() > 0.0, "float32 round-trip did not jitter"
    assert diffs.max() - diffs.min() < 1e-6, "jitter unexpectedly large"

    ds = _GridDS(t_grid)
    sampler = train_mod._build_collocation_sampler(
        {}, {"dt": 0.005}, ds, spec, 300.0, 10.0, batch_size=4, rng_seed=0,
    )
    assert abs(sampler.dt - 0.01) < 1e-6


def test_on_grid_dt_pinning_rejects_real_nonuniform():
    """A grid with a real save_stride doubling partway through is rejected so a
    genuinely non-uniform t_grid cannot be silently pinned to a wrong dt."""
    import src.operators.train as train_mod

    spec = DiffusionProblem()
    t_grid = np.concatenate([
        np.arange(0, 0.10, 0.01),          # spacing 0.01
        np.arange(0.10, 0.30 + 0.02, 0.02),  # spacing 0.02
    ]).astype(np.float32)
    ds = _GridDS(t_grid)
    with pytest.raises(ValueError, match="non-uniform"):
        train_mod._build_collocation_sampler(
            {}, {"dt": 0.005}, ds, spec, 300.0, 10.0, batch_size=4, rng_seed=0,
        )
