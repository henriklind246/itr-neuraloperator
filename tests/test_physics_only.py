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


# --- 3b. hard right-Dirichlet output constraint zeros the right_dirichlet MSE -


def _hard_bc_fno(t_right_norm, *, seed=0):
    """A real FNO2d matching the `_StubColloc` batch layout (4 spatial channels,
    2 cond-static dims, 2-dim forcing tokens) with the hard right-Dirichlet
    constraint enabled and its wall value set to `t_right_norm`."""
    from src.operators.fno2d import FNO2d

    torch.manual_seed(seed)
    return FNO2d(
        modes1=2, modes2=2, width=8,
        in_channels=4, out_channels=1, n_layers=2,
        cond_static_dim=2, temporal_token_dim=2,
        temporal_hidden=16, forcing_embed_dim=16, forcing_spatial_dim=4,
        hard_right_dirichlet=True, t_right_norm=t_right_norm,
    )


def test_hard_right_dirichlet_zeros_phys_right_dirichlet_mse():
    """With the hard right-Dirichlet constraint set to the collocation geom's
    `T_right_tilde`, both collocation forwards pin the right wall to the exact BC
    value, so the `right_dirichlet` region residual is at the numeric floor
    (< 1e-12). A model whose wall value is mismatched leaves a real residual, so
    the gate is not vacuous."""
    device = torch.device("cpu")
    sampler = _StubColloc(Nx=11, Ny=11)
    t_right = float(sampler.geom_cfg["T_right_tilde"])  # 0.0

    matched = _hard_bc_fno(t_right)
    out, _b, _ic = _collocation_batch_loss(
        matched, sampler, 0.05, None, device, lambda_ic=0.0
    )
    assert float(out["phys_right_dirichlet_mse"]) < 1e-12

    # A mismatched wall value (t_right_tilde + 1) leaves a non-trivial residual.
    sampler2 = _StubColloc(Nx=11, Ny=11)
    mismatched = _hard_bc_fno(t_right + 1.0)
    out2, _b2, _ic2 = _collocation_batch_loss(
        mismatched, sampler2, 0.05, None, device, lambda_ic=0.0
    )
    assert float(out2["phys_right_dirichlet_mse"]) > 1e-6


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


# --- 4c. IC-anchoring warm-up (Step 5): freeze cadence + IC up-weight --------


from src.operators.train import CollocationSampler, _compose_physics_total


class _WarmupColloc(_StubColloc):
    """`_StubColloc` wearing the REAL `CollocationSampler` freeze-cache methods so
    the warm-up cadence exercises the production caching wrapper (not a re-impl).
    `_draw_collocation_batch` counts genuine fresh draws; frozen steps reuse the
    cache and never bump the counter."""

    set_frozen = CollocationSampler.set_frozen
    enable_freeze_cache = CollocationSampler.enable_freeze_cache
    reset_freeze_cache = CollocationSampler.reset_freeze_cache
    sample_batch = CollocationSampler.sample_batch

    def __init__(self, **kw):
        super().__init__(**kw)
        self._freeze_enabled = False
        self._frozen = False
        self._batch_cache = None
        self._cache_leads = None
        self._cache_bin_ids = None
        self._cache_anchor = None
        self._last_leads = None
        self._last_lead_bin_ids = None
        self._last_anchor = None
        self.draws = 0

    def _draw_collocation_batch(self, max_lead):
        self.draws += 1
        return _StubColloc.sample_batch(self, max_lead)


def _run_warmup_epoch(sampler, warmup, n_steps, global_step_start=0):
    model = _IdentityModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    loader = _ExplodingLoader(n_steps)
    return train_one_epoch(
        model=model, train_loader=loader, optimizer=optimizer,
        loss_fn=lambda *a, **k: None, device=torch.device("cpu"),
        physics_collocation=sampler, collocation_max_lead=0.05,
        lambda_data=0.0, lambda_physics=1.0, lambda_ic=1.0,
        warmup=warmup, global_step_start=global_step_start,
    )


def test_warmup_freezes_resample_cadence():
    """During warm-up the collocation batch is redrawn once per `resample_every`
    optimizer steps (steps 0, R, 2R, ...); the intervening frozen steps reuse the
    cache. For 6 steps at resample_every=3 -> fresh draws at gstep 0 and 3 = 2."""
    sampler = _WarmupColloc()
    warmup = {"enabled": True, "steps": 100, "resample_every": 3, "ic_multiplier": 5.0}
    _run_warmup_epoch(sampler, warmup, n_steps=6)
    assert sampler.draws == 2


def test_warmup_reverts_to_per_step_after_steps():
    """After `warmup.steps` optimizer steps the cadence reverts to a fresh draw
    every step. With steps=2, resample_every=3 over 6 steps: gstep0 fresh, gstep1
    frozen (within warm-up), gsteps 2-5 fresh (past warm-up) -> 1 + 4 = 5 draws."""
    sampler = _WarmupColloc()
    warmup = {"enabled": True, "steps": 2, "resample_every": 3, "ic_multiplier": 5.0}
    _run_warmup_epoch(sampler, warmup, n_steps=6)
    assert sampler.draws == 5


def test_warmup_disabled_draws_every_step():
    """Warm-up off (default) -> the sampler is drawn fresh every optimizer step and
    the freeze cache is never armed (bit-identical cadence to the pre-Step-5 loop)."""
    sampler = _WarmupColloc()
    _run_warmup_epoch(sampler, {"enabled": False}, n_steps=5)
    assert sampler.draws == 5
    assert sampler._freeze_enabled is False
    assert sampler._batch_cache is None


def test_warmup_global_step_counter_advances():
    """`global_optimizer_step` starts at `global_step_start` and advances by exactly
    one per optimizer step, so run_one_seed can checkpoint/resume the warm-up
    boundary without reconstructing epoch * steps_per_epoch."""
    sampler = _WarmupColloc()
    warmup = {"enabled": True, "steps": 100, "resample_every": 3, "ic_multiplier": 5.0}
    metrics = _run_warmup_epoch(sampler, warmup, n_steps=4, global_step_start=10)
    assert int(metrics["global_optimizer_step"]) == 14


def test_warmup_ic_multiplier_scales_only_ic_term():
    """The IC up-weight multiplies the IC coefficient AFTER any GradNorm multiplier
    (legacy branch: gradnorm=None). Raising `ic_multiplier` from 1 to m scales the
    IC contribution by m and leaves the physics contribution untouched, so the
    totals differ by exactly `(m - 1) * lambda_ic * ic`."""
    device = torch.device("cpu")
    phys = torch.tensor(0.7)
    ic = torch.tensor(0.3)
    out = {"physics_loss_weighted": phys}
    common = dict(
        out=out, data_loss=None, ic=ic, causal_weighter=None, gradnorm=None,
        model_unwrapped=None, dist_info=None,
        lambda_data=0.0, lambda_physics=1.0, lambda_ic=2.0, region_weights=None,
    )
    total1, _, _ = _compose_physics_total(ic_multiplier=1.0, **common)
    total5, _, _ = _compose_physics_total(ic_multiplier=5.0, **common)
    # physics part identical; IC part scales 1 -> 5 at lambda_ic=2, ic=0.3.
    assert abs(float(total5) - float(total1) - (5.0 - 1.0) * 2.0 * 0.3) < 1e-6


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


# --- 8b. on-grid path records _last_leads (interval midpoints) + bin ids ------


def _on_grid_sampler_with_capture(monkeypatch, batch_size=8, dt=0.01, n_t=11,
                                  bin_spec=None):
    """Build a real DiffusionProblem on-grid CollocationSampler with the heavy
    item builders stubbed, capturing the per-row `t` (first of each on-grid pair)
    so a test can reconstruct the expected interval midpoints independently of
    the sampler's RNG."""
    import src.operators.train as train_mod
    from src.operators.train import CollocationSampler

    spec = DiffusionProblem()
    t_grid = np.arange(n_t) * dt

    class _DS:
        def __init__(self):
            self.x_grid = np.linspace(0.0, 1.0, 4)
            self.y_grid = np.linspace(0.0, 1.0, 5)
            self.Ny = 5
            self.t_grid = t_grid
            self.t_final = float(t_grid[-1])
            self.sim_ids = np.array([0, 1, 2, 3, 4])
            self.sim_params = {int(i): {} for i in range(5)}
            self.problem = self

        def build_item(self, ds, sid, s, j):
            return {"spatial": np.zeros((4, 5, 1), dtype=np.float32)}

    ts_calls = []

    def _fake_rollout(base_item, ds, spc, sid, current, t_lo, t_hi):
        ts_calls.append(round(float(t_hi), 9))
        return {"x": np.zeros(1, np.float32)}

    monkeypatch.setattr(train_mod, "build_rollout_item_from_base", _fake_rollout)
    monkeypatch.setattr(train_mod, "collate_fn", lambda items: items)

    ds = _DS()
    geom_cfg = spec.collocation_geom_cfg(ds, {}, 300.0, 10.0, dt)
    base_plan = spec.collocation_base_plan(ds, dt)
    sampler = CollocationSampler(
        ds, spec, geom_cfg, batch_size=batch_size, dt=dt, rng_seed=0,
        base_plan=base_plan,
    )
    sampler.bin_spec = bin_spec
    return sampler, ts_calls, dt


def test_on_grid_records_last_leads_as_midpoints(monkeypatch):
    """The on-grid path (diffusion) populates `_last_leads` with the CN interval
    midpoint `(n + 0.5) * dt` per row in draw order — previously it set neither
    field (the bug that left `phys_lead_hist` empty for diffusion)."""
    sampler, ts_calls, dt = _on_grid_sampler_with_capture(monkeypatch)
    sampler.sample_batch(max_lead=0.05)

    leads = sampler._last_leads
    assert leads is not None
    assert leads.shape == (8,)
    # Each lead is an interval midpoint: leads/dt - 0.5 is a non-negative integer.
    steps = leads / dt - 0.5
    np.testing.assert_allclose(steps, np.round(steps), atol=1e-9)
    assert (steps >= 0).all()

    # Draw order matches the captured `t` (every other rollout call is the pair's
    # first time `t = n*dt`; base snapshot index 0 -> t_s = 0).
    t_firsts = np.asarray(ts_calls[0::2], dtype=float)
    np.testing.assert_allclose(leads, t_firsts + 0.5 * dt, atol=1e-9)

    # With no bin spec attached, bin ids stay None (bit-identical legacy path).
    assert sampler._last_lead_bin_ids is None


def test_on_grid_bin_ids_match_spec(monkeypatch):
    """With a shared `TemporalBinSpec` attached, `_last_lead_bin_ids` equals
    `spec.interval_to_bin(_last_leads)` row-for-row."""
    from src.operators.train import TemporalBinSpec

    dt = 0.01
    n_t = 11
    # On-grid midpoint domain: [0.5*dt, t_final - 0.5*dt].
    spec = TemporalBinSpec(lead_lo=0.5 * dt, lead_hi=(n_t - 1) * dt - 0.5 * dt,
                           n_bins=5)
    sampler, _, _ = _on_grid_sampler_with_capture(
        monkeypatch, dt=dt, n_t=n_t, bin_spec=spec
    )
    sampler.sample_batch(max_lead=0.10)

    ids = sampler._last_lead_bin_ids
    assert ids is not None
    expected = spec.interval_to_bin(sampler._last_leads)
    np.testing.assert_array_equal(ids, expected)
    assert (ids >= 0).all() and (ids < spec.n_bins).all()


def test_temporal_bin_spec_edges_and_active():
    """`TemporalBinSpec` edges/clamp/active-mask behave as specified."""
    from src.operators.train import TemporalBinSpec

    spec = TemporalBinSpec(lead_lo=0.0, lead_hi=1.0, n_bins=4)
    np.testing.assert_allclose(spec.bin_edges, [0.0, 0.25, 0.5, 0.75, 1.0])
    # Clamp at/beyond the edges.
    assert int(spec.interval_to_bin(-5.0)) == 0
    assert int(spec.interval_to_bin(5.0)) == 3
    assert int(spec.interval_to_bin(0.0)) == 0
    assert int(spec.interval_to_bin(1.0)) == 3
    # Interior mapping.
    assert int(spec.interval_to_bin(0.3)) == 1
    # Active mask: window [0.6, 0.9] overlaps bins 2 (0.5-0.75) and 3 (0.75-1.0).
    active = spec.active_bins(0.6, 0.9)
    np.testing.assert_array_equal(active, [False, False, True, True])


# --- 8c. Step 1: block-structured on-grid lead stratification ----------------


def _stratified_on_grid_sampler(monkeypatch, *, batch_size, dt, n_t, n_bins,
                                anchored=False):
    """Build a DiffusionProblem on-grid CollocationSampler with `stratify_leads`
    on and the shared on-grid `TemporalBinSpec` attached, capturing the per-row
    `sid` in draw order (via the stubbed `build_item`) so a test can assert the
    per-bin sim-id multiset. Returns `(sampler, sids)` where `sids` is filled on
    each `sample_batch` call."""
    import src.operators.train as train_mod
    from src.operators.train import CollocationSampler, TemporalBinSpec

    spec = DiffusionProblem()
    t_grid = np.arange(n_t) * dt
    sids: list[int] = []

    class _DS:
        def __init__(self):
            self.x_grid = np.linspace(0.0, 1.0, 4)
            self.y_grid = np.linspace(0.0, 1.0, 5)
            self.Ny = 5
            self.t_grid = t_grid
            self.t_final = float(t_grid[-1])
            self.sim_ids = np.array([0, 1, 2, 3, 4])
            self.sim_params = {int(i): {} for i in range(5)}
            self.problem = self

        def build_item(self, ds, sid, s, j):
            sids.append(int(sid))
            return {"spatial": np.zeros((4, 5, 1), dtype=np.float32)}

    monkeypatch.setattr(
        train_mod, "build_rollout_item_from_base",
        lambda base_item, ds, spc, sid, current, t_lo, t_hi: {
            "x": np.zeros(1, np.float32)
        },
    )
    monkeypatch.setattr(train_mod, "collate_fn", lambda items: items)

    ds = _DS()
    geom_cfg = spec.collocation_geom_cfg(ds, {}, 300.0, 10.0, dt)
    base_plan = spec.collocation_base_plan(ds, dt)
    sampler = CollocationSampler(
        ds, spec, geom_cfg, batch_size=batch_size, dt=dt, rng_seed=0,
        base_plan=base_plan, anchored_first_step=anchored, stratify_leads=True,
    )
    # On-grid midpoint domain, exactly as _make_temporal_bin_spec builds it.
    sampler.bin_spec = TemporalBinSpec(
        lead_lo=0.5 * dt, lead_hi=float(t_grid[-1]) - 0.5 * dt, n_bins=n_bins,
    )
    return sampler, sids


def test_stratified_per_chunk_allocation_and_in_chunk(monkeypatch):
    """B=48, 6 bins, n_max=11 -> 12 CN intervals split into 6 contiguous pairs;
    every bin is populated by exactly B/6=8 rows whose `n` lies in that bin's
    2-element chunk."""
    dt = 0.01
    # n_t=13 -> t_grid[0..12]; max_lead=0.12 -> n_max=min(12-1, 13-2)=11.
    sampler, _ = _stratified_on_grid_sampler(
        monkeypatch, batch_size=48, dt=dt, n_t=13, n_bins=6,
    )
    sampler.sample_batch(max_lead=0.12)

    ids = sampler._last_lead_bin_ids
    leads = sampler._last_leads
    assert ids is not None and leads.shape == (48,)
    # Every bin is populated equally (8 each) and no bin is empty.
    counts = np.bincount(ids, minlength=6)
    np.testing.assert_array_equal(counts, np.full(6, 8))
    # In-chunk: bin b holds exactly the two intervals {2b, 2b+1}.
    n_vals = np.round(leads / dt - 0.5).astype(int)
    for b in range(6):
        drawn = set(n_vals[ids == b].tolist())
        assert drawn.issubset({2 * b, 2 * b + 1}), (b, drawn)


def test_stratified_per_bin_sim_id_multiset_identical(monkeypatch):
    """Block-structured balance: for a fixed seed every populated bin sees the
    SAME multiset of sim ids (so a later bin's higher residual reflects longer
    lead, not a harder set of input functions)."""
    dt = 0.01
    sampler, sids = _stratified_on_grid_sampler(
        monkeypatch, batch_size=48, dt=dt, n_t=13, n_bins=6,
    )
    sampler.sample_batch(max_lead=0.12)

    ids = np.asarray(sampler._last_lead_bin_ids)
    sid_arr = np.asarray(sids)
    assert sid_arr.shape == ids.shape
    active = [b for b in range(6) if (ids == b).any()]
    ref = sorted(sid_arr[ids == active[0]].tolist())
    for b in active[1:]:
        assert sorted(sid_arr[ids == b].tolist()) == ref


def test_stratified_bin_ids_match_spec_rowwise(monkeypatch):
    """With stratification on, `_last_lead_bin_ids` still equals
    `spec.interval_to_bin(_last_leads)` row-for-row (chunk id == recorded bin)."""
    sampler, _ = _stratified_on_grid_sampler(
        monkeypatch, batch_size=48, dt=0.01, n_t=13, n_bins=6,
    )
    sampler.sample_batch(max_lead=0.12)
    expected = sampler.bin_spec.interval_to_bin(sampler._last_leads)
    np.testing.assert_array_equal(sampler._last_lead_bin_ids, expected)


def test_stratified_same_seed_lockstep(monkeypatch):
    """Two same-seed stratified samplers draw byte-identical batches (DDP ranks
    stay in lockstep given the single rng stream)."""
    a, _ = _stratified_on_grid_sampler(
        monkeypatch, batch_size=48, dt=0.01, n_t=13, n_bins=6,
    )
    b, _ = _stratified_on_grid_sampler(
        monkeypatch, batch_size=48, dt=0.01, n_t=13, n_bins=6,
    )
    a.sample_batch(max_lead=0.12)
    b.sample_batch(max_lead=0.12)
    np.testing.assert_array_equal(a._last_leads, b._last_leads)
    np.testing.assert_array_equal(a._last_lead_bin_ids, b._last_lead_bin_ids)


def test_stratified_degenerate_n_max_below_n_bins(monkeypatch):
    """When fewer valid CN intervals than bins exist (short lead window), empty
    bins are skipped: every drawn row still lands in a populated bin and the
    batch stays exactly B rows (no crash on the degenerate split)."""
    dt = 0.01
    # max_lead=0.03 -> n_max=min(3-1, 13-2)=2 -> intervals n=0,1,2 only.
    sampler, _ = _stratified_on_grid_sampler(
        monkeypatch, batch_size=48, dt=dt, n_t=13, n_bins=6,
    )
    sampler.sample_batch(max_lead=0.03)

    ids = np.asarray(sampler._last_lead_bin_ids)
    assert ids.shape == (48,)
    # Only the bins covering midpoints {0.005, 0.015, 0.025} are populated.
    populated = set(ids.tolist())
    expected_active = set(
        int(sampler.bin_spec.interval_to_bin((np.arange(3) + 0.5) * dt)[i])
        for i in range(3)
    )
    assert populated == expected_active
    assert len(populated) < 6  # genuinely degenerate


def test_stratified_anchored_marks_exactly_n0(monkeypatch):
    """`anchored_first_step` composes with stratification: the anchor mask marks
    exactly the `n==0` rows (bin 0 contains n=0)."""
    dt = 0.01
    sampler, _ = _stratified_on_grid_sampler(
        monkeypatch, batch_size=48, dt=dt, n_t=13, n_bins=6, anchored=True,
    )
    sampler.sample_batch(max_lead=0.12)
    mask, _T0 = sampler._last_anchor
    n_vals = np.round(sampler._last_leads / dt - 0.5).astype(int)
    np.testing.assert_array_equal(mask.numpy(), n_vals == 0)


# --- 8d. Step 1: _build_collocation_sampler stratify guards ------------------


class _GenericSpecStub:
    """Minimal off-grid ProblemSpec: no base plan (generic/forcing path) and a
    None geom_cfg so `_build_collocation_sampler` falls back to its two-slab
    default -- enough to reach the stratify_leads on-grid-only guard."""

    def collocation_base_plan(self, ds, dt):
        return None

    def collocation_geom_cfg(self, ds, phys_cfg, mu, sigma, dt):
        return None


def _guard_ds(dt=0.01, n_t=13):
    t_grid = np.arange(n_t) * dt

    class _DS:
        def __init__(self):
            self.x_grid = np.linspace(0.0, 1.0, 4)
            self.y_grid = np.linspace(0.0, 1.0, 5)
            self.Ny = 5
            self.t_grid = t_grid
            self.t_final = float(t_grid[-1])
            self.sim_ids = np.array([0, 1, 2])
            self.sim_params = {int(i): {} for i in range(3)}
            self.problem = self

    return _DS()


def test_stratify_on_generic_path_raises(monkeypatch):
    """`stratify_leads=true` on a benchmark whose base plan is off-grid raises a
    ValueError pointing at the generic-path coverage knobs."""
    from src.operators.train import _build_collocation_sampler

    with pytest.raises(ValueError, match="on-grid-only"):
        _build_collocation_sampler(
            {}, {"stratify_leads": True, "dt": 0.01}, _guard_ds(),
            _GenericSpecStub(), 300.0, 10.0, batch_size=8, rng_seed=0,
        )


def test_stratify_early_oversample_mutually_exclusive_raises(monkeypatch):
    """`stratify_leads` and `early_time_oversample` are mutually exclusive on the
    on-grid path; enabling both raises ValueError."""
    from src.operators.train import _build_collocation_sampler

    with pytest.raises(ValueError, match="mutually exclusive"):
        _build_collocation_sampler(
            {}, {"stratify_leads": True, "early_time_oversample": True},
            _guard_ds(), DiffusionProblem(), 300.0, 10.0,
            batch_size=8, rng_seed=0,
        )


# --- 9. zero-duration anchor item (t_s == t_j) is finite ---------------------


class _StubDiffDS:
    """In-memory dataset stub exposing exactly the attributes `build_item` reads,
    so the IC-anchor (zero-duration, t_s == t_j) build path can be exercised
    without a real on-disk dataset."""

    def __init__(self, Nx=6, Ny=5, n_t=8, dt=0.01):
        self.Nx, self.Ny = Nx, Ny
        self.t_grid = np.arange(n_t, dtype=float) * dt
        self.t_final = float(self.t_grid[-1])
        self.time_norm_horizon = self.t_final
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


# --- 11. causal anchored physics-only training (constant-300 collapse fix) ---
#
# These gate the structural fix for the diffusion collapse: (a) the anchored
# first step (feed the EXACT IC as T_n so only genuine one-step diffusion zeros
# the residual), (b) the collapse the anchor penalizes, and (c) the early-time
# oversampling mixture / anchor side-channel. All residual math is float64 with
# the direct-solve floor convention from `test_full_bc_residual_zero_on_...`.


def _homogeneous_ic_hist_geom(dt=0.005, t_final=0.1, Nx=40, Ny=40):
    """Homogeneous force-free sim seeded with a non-uniform Gaussian IC, its FV
    trajectory, and the matching float64 CN geom. Reused by the anchored-first-
    step floor test and the collapse test so the collapse residual is compared
    against the SAME solver-state floor (not an absolute number)."""
    sim = _build_homogeneous_solver(dt=dt, t_final=t_final, Nx=Nx, Ny=Ny)
    X, Y = np.meshgrid(sim.grid_x, sim.grid_y, indexing="ij")
    T0 = (300.0 + 40.0 * np.exp(-((X - 0.4) ** 2 + (Y - 0.5) ** 2) / 0.02)).astype(np.float64)
    _, _, _, T_hist = sim.solve(T0=T0, store_trajectory=True)
    geom = build_cn_geom(
        sim.grid_x, sim.grid_y,
        k_left=K_SLAB, k_right=K_SLAB, interface_x=0.5, R_c=0.0,
        dt=sim.dt, sigma_global=1.0, dtype=torch.float64,
    )
    return sim, T0, T_hist, geom


def _solver_state_bc(sim, n):
    """FullBCData for the (n -> n+1) solver step (raw K, sigma_global == 1)."""
    tnp1 = float(sim.t[n + 1])
    return FullBCData(
        T_right_tilde=torch.as_tensor(float(sim.T_right(tnp1)), dtype=torch.float64),
        qL_n=torch.as_tensor(sim.q_left(float(sim.t[n])), dtype=torch.float64),
        qL_np1=torch.as_tensor(sim.q_left(tnp1), dtype=torch.float64),
    )


def test_anchored_first_step_solver_state_residual_zero():
    """The anchored first step target: feeding the EXACT IC (T_hist[0]) as T_n and
    the true one-step FV evolution (T_hist[1]) as T_np1 gives a full_bc residual at
    the direct-solve floor in EVERY region. This is the n==0 solver-state residual
    the hard IC injection reproduces — the only residual-zeroing T_np1 is genuine
    one-step diffusion of T_0, not the constant-300 collapse."""
    sim, _, T_hist, geom = _homogeneous_ic_hist_geom()
    T_n = torch.as_tensor(T_hist[0], dtype=torch.float64)
    T_np1 = torch.as_tensor(T_hist[1], dtype=torch.float64)
    parts = full_bc_cn_residual(T_n, T_np1, geom, _solver_state_bc(sim, 0))
    for name, vec in parts.items():
        assert float(vec.abs().max()) < 1e-6, (
            f"anchored first-step region {name} residual floor too high: "
            f"{float(vec.abs().max()):.3e}"
        )


def test_steady_state_residual_zero():
    """The uniform-300 K field (normalized to 0 with mu==300) for BOTH T_n and
    T_np1, with T_right_tilde == 0 and zero left flux, gives ~0 residual in every
    region: constant 300 K satisfies the homogeneous PDE and every BC. This is the
    degeneracy the anchored first step exists to escape (a residual-only objective
    is minimized here)."""
    Nx, Ny = 30, 24
    x_grid = np.linspace(0.0, 1.0, Nx)
    y_grid = np.linspace(0.0, 1.0, Ny)
    mu, sigma = 300.0, 10.0
    geom = build_cn_geom(
        x_grid, y_grid, k_left=K_SLAB, k_right=K_SLAB, interface_x=0.5, R_c=0.0,
        dt=0.005, sigma_global=sigma, dtype=torch.float64,
    )
    T_flat = torch.full((Nx, Ny), (300.0 - mu) / sigma, dtype=torch.float64)
    bc = FullBCData(
        T_right_tilde=torch.as_tensor((300.0 - mu) / sigma, dtype=torch.float64),
        qL_n=torch.zeros(Ny, dtype=torch.float64),
        qL_np1=torch.zeros(Ny, dtype=torch.float64),
    )
    parts = full_bc_cn_residual(T_flat, T_flat.clone(), geom, bc)
    for name, vec in parts.items():
        assert float(vec.abs().max()) < 1e-6, (
            f"steady-state region {name} residual not ~0: {float(vec.abs().max()):.3e}"
        )


def test_artificial_collapse_residual_large():
    """The collapse the anchored first step penalizes: T_n == exact non-uniform IC,
    T_np1 == uniform 300 K. The interior/transient (storage) residual dominates and
    is orders of magnitude above the solver-state floor, so anchoring T_n to the IC
    makes the constant-300 jump costly. The right-Dirichlet residual is NOT required
    to be large: T_np1 == 300 K legitimately satisfies that wall (~0)."""
    sim, _, T_hist, geom = _homogeneous_ic_hist_geom()

    # Solver-state interior floor (the reference the collapse is compared to).
    T0 = torch.as_tensor(T_hist[0], dtype=torch.float64)
    T1 = torch.as_tensor(T_hist[1], dtype=torch.float64)
    R_solver_interior = float(
        full_bc_cn_residual(T0, T1, geom, _solver_state_bc(sim, 0))["interior"].abs().max()
    )

    # Collapse pair: exact IC -> uniform 300 K (sigma_global == 1 => raw K).
    T_flat300 = torch.full_like(T0, 300.0)
    bc = FullBCData(
        T_right_tilde=torch.as_tensor(300.0, dtype=torch.float64),
        qL_n=torch.zeros(sim.Ny if hasattr(sim, "Ny") else T0.shape[1], dtype=torch.float64),
        qL_np1=torch.zeros(T0.shape[1], dtype=torch.float64),
    )
    parts = full_bc_cn_residual(T0, T_flat300, geom, bc)
    R_collapse_interior = float(parts["interior"].abs().max())

    # The right Dirichlet end is satisfied by T_np1 == 300 K (both edges ~0).
    assert float(parts["right_dirichlet"].abs().max()) < 1e-6
    # The interior storage imbalance dominates and dwarfs the solver-state floor.
    assert R_collapse_interior > 1e3 * R_solver_interior, (
        f"collapse interior residual {R_collapse_interior:.3e} not >> solver floor "
        f"{R_solver_interior:.3e}"
    )


class _MixDS:
    """On-grid collocation dataset stub. `build_item` returns a per-sim constant
    IC field (300 + sid) in channel 0 so `T0_exact` is identifiable, and only the
    RNG-driven start-index draw + anchor side-channel are exercised."""

    def __init__(self, t_grid, Nx=4, Ny=5):
        self.x_grid = np.linspace(0.0, 1.0, Nx)
        self.y_grid = np.linspace(0.0, 1.0, Ny)
        self.Nx, self.Ny = Nx, Ny
        self.t_grid = np.asarray(t_grid, dtype=float)
        self.t_final = float(self.t_grid[-1])
        self.sim_ids = np.array([0, 1, 2])
        self.sim_params = {i: {} for i in range(3)}
        self.problem = self

    def build_item(self, ds, sid, s, j):
        field = np.full((self.Nx, self.Ny), 300.0 + float(sid), dtype=np.float32)
        return {"spatial": field[..., None]}


def test_early_time_mixture_and_anchor_mask(monkeypatch):
    """The 3-bucket early-time oversampling mixture draws the anchor (n==0) bucket
    at its RENORMALIZED weight, `_last_anchor` marks exactly the n==0 rows, and
    `T0_exact` equals the base IC field. Three windows exercise the renormalization:
    all buckets non-empty (0.4), `rest` empty (0.5), only `anchor` reachable (1.0)."""
    import src.operators.train as train_mod
    from src.operators.train import CollocationSampler

    spec = DiffusionProblem()
    dt = 0.01
    t_grid = np.arange(12, dtype=float) * dt  # len-2 == 10, enough for n_max up to 9

    calls = []

    def _fake_roll(base_item, ds, spc, sid, current, t_lo, t_hi):
        calls.append((float(t_hi), int(sid)))
        return {"x": np.zeros(1, np.float32)}

    monkeypatch.setattr(train_mod, "build_rollout_item_from_base", _fake_roll)
    monkeypatch.setattr(train_mod, "collate_fn", lambda items: items)

    ds = _MixDS(t_grid)
    geom_cfg = spec.collocation_geom_cfg(ds, {}, 300.0, 10.0, dt)
    base_plan = spec.collocation_base_plan(ds, dt)
    t_s = float(t_grid[0])

    def _run_window(max_lead, n_batches=400, B=16):
        sampler = CollocationSampler(
            ds, spec, geom_cfg, batch_size=B, dt=dt, rng_seed=0,
            base_plan=base_plan, anchored_first_step=True,
            early_oversample=True, early_band_steps=5,
            early_mix={"anchor": 0.4, "early": 0.4, "rest": 0.2},
        )
        n_anchor_rows, n_total_rows = 0, 0
        for _ in range(n_batches):
            calls.clear()
            sampler.sample_batch(max_lead=max_lead)
            mask, T0_exact = sampler._last_anchor
            # item_t (the `t` query) is built before item_tdt, so even-indexed
            # calls carry t = t_s + n*dt; derive n and the drawn sid per row.
            even = calls[0::2]
            t_queries = np.array([c[0] for c in even])
            sid_queries = [c[1] for c in even]
            n_vals = np.rint((t_queries - t_s) / dt).astype(int)
            # mask marks EXACTLY the n==0 rows.
            assert np.array_equal(mask.numpy(), (n_vals == 0))
            # T0_exact is the base IC field (300 + sid) for each drawn sim.
            for i, sid in enumerate(sid_queries):
                assert torch.allclose(
                    T0_exact[i],
                    torch.full((ds.Nx, ds.Ny), 300.0 + float(sid)),
                )
            n_anchor_rows += int(mask.sum().item())
            n_total_rows += int(mask.numel())
        return n_anchor_rows / n_total_rows

    # Window 1: n_max == 9, all three buckets non-empty -> anchor freq ~ 0.4.
    assert abs(_run_window(0.105) - 0.4) < 0.04
    # Window 2: n_max == 4, `rest` empty -> renormalized anchor freq ~ 0.5.
    assert abs(_run_window(0.055) - 0.5) < 0.04
    # Window 3: n_max == 0, only the anchor bucket is reachable -> freq == 1.0.
    assert _run_window(0.015, n_batches=20) == 1.0


class _ConstModel(torch.nn.Module):
    """Emits a constant normalized field (the collapse target). The `0 * w` term
    keeps a trainable parameter in the graph so the loss is differentiable."""

    def __init__(self, val=0.0):
        super().__init__()
        self.val = float(val)
        self.w = torch.nn.Parameter(torch.zeros(1))

    def forward(self, spatial, cond_static, forcing_seq=None):
        base = spatial[..., 0:1]
        return torch.full_like(base, self.val) + 0.0 * self.w


class _AnchorStubColloc:
    """Fully-anchored on-grid collocation stub: every row is an n==0 anchor whose
    T_n is the exact non-uniform IC. With a constant-300 K model (val==0 in raw K,
    sigma_global==1) the residual is the constant-300 collapse when anchoring is on
    and the trivial steady residual (~0) when off."""

    def __init__(self, B=4, Nx=12, Ny=12, dt=0.01, anchored=True):
        self.B, self.Nx, self.Ny = B, Nx, Ny
        xg = np.linspace(0.0, 1.0, Nx)
        yg = np.linspace(0.0, 1.0, Ny)
        self.geom_cfg = {
            "x_grid": xg, "y_grid": yg,
            "k_left": K_SLAB, "k_right": K_SLAB,
            "interface_x": float(xg[Nx // 2]),
            "dt": dt, "sigma_global": 1.0, "T_right_tilde": 0.0,
        }
        X, Y = np.meshgrid(xg, yg, indexing="ij")
        # Non-uniform IC in raw K minus the 300 K wall (sigma==1 => normalized==dev);
        # the Gaussian bump is centered away from the right edge so its right column
        # (and thus the right-Dirichlet residual) stays ~0.
        self.ic = (40.0 * np.exp(-((X - 0.4) ** 2 + (Y - 0.5) ** 2) / 0.02)).astype(np.float32)
        self.anchored = bool(anchored)
        self._last_anchor = None

    def _mk_batch(self, field):
        spatial = np.zeros((self.B, self.Nx, self.Ny, 4), np.float32)
        spatial[..., 0] = field
        return {
            "spatial": torch.from_numpy(spatial),
            "cond_static": torch.zeros(self.B, 2),
            "forcing_seq": torch.zeros(self.B, 128, 2),
        }

    def sample_batch(self, max_lead):
        self._last_anchor = None
        ic = np.broadcast_to(self.ic, (self.B, self.Nx, self.Ny)).copy()
        z = torch.zeros(self.B, self.Ny)
        batch = (self._mk_batch(ic), self._mk_batch(ic), torch.zeros(self.B),
                 z, z.clone(), z.clone())
        if self.anchored:
            mask = torch.ones(self.B, dtype=torch.bool)
            self._last_anchor = (mask, torch.from_numpy(ic))
        return batch


def test_anchored_batch_loss_penalizes_collapse():
    """`_collocation_batch_loss` with a constant-300 K model and a non-uniform IC:
    anchoring ON hard-injects the exact IC as T_n, so the (IC -> 300 K) collapse
    fires and `physics_loss_weighted` is large; anchoring OFF leaves both fields at
    the constant, giving the ~0 steady residual. This is the loss-level confirmation
    the anchor directly penalizes the observed collapse."""
    device = torch.device("cpu")
    model = _ConstModel(val=0.0)

    sampler_on = _AnchorStubColloc(anchored=True)
    out_on, _, _ = _collocation_batch_loss(model, sampler_on, 0.05, None, device)
    loss_on = float(out_on["physics_loss_weighted"])
    # The split diagnostic sees an all-anchor batch.
    assert out_on["_anchor_count"] == sampler_on.B
    assert out_on["_nonanchor_count"] == 0

    sampler_off = _AnchorStubColloc(anchored=False)
    out_off, _, _ = _collocation_batch_loss(model, sampler_off, 0.05, None, device)
    loss_off = float(out_off["physics_loss_weighted"])
    assert out_off["_anchor_count"] == 0

    assert loss_off < 1e-8, f"steady (non-anchored) residual not ~0: {loss_off:.3e}"
    assert loss_on > 1.0, f"anchored collapse residual not large: {loss_on:.3e}"
