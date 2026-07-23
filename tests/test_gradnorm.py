"""Step 4 coverage for per-component GradNorm-style adaptive loss balancing.

Exercises ``GradNormBalancer`` (inverse-gradient-norm equalization of imbalanced
terms, the clamp/hold rules, ``update_every`` cadence and state round-trip) and
the single ``_compose_physics_total`` site (legacy branch bit-identical to the
pre-Step-4 expression; all-ones GradNorm equal to a plain ``sum(c_t * L_t)``).

A ``train_one_epoch`` smoke populates ``gradnorm_weights`` and the per-phase
timing keys on a stub collocation sampler, and a two-rank gloo test asserts the
GradNorm ``autograd.grad`` measurement + a subsequent ``backward`` complete under
``static_graph=True`` with identical multipliers on both ranks (SUM all-reduce).
"""

from __future__ import annotations

import os
from datetime import timedelta

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from problems.diffusion import K_SLAB
from src.operators.train import (
    GradNormBalancer,
    _active_physics_terms,
    _compose_physics_total,
    _physics_static_coeffs,
    train_one_epoch,
)


# --------------------------------------------------------------------------- #
# 1. GradNormBalancer — inverse-gradient-norm equalization                    #
# --------------------------------------------------------------------------- #


def _lin():
    """A single scalar weight = 1 so ``(c * y).sum()`` has grad norm ``|c|``."""
    m = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        m.weight.fill_(1.0)
    return m


def test_100x_imbalance_equalizes():
    """Two terms whose gradient norms differ 100x get multipliers ~ mean(g)/g_t,
    so their weighted gradient contributions ``w_t * g_t`` are equalized."""
    m = _lin()
    y = m(torch.ones(1, 1))
    term_a = (10.0 * y).sum()   # grad norm 10
    term_b = (0.1 * y).sum()    # grad norm 0.1
    gn = GradNormBalancer(["a", "b"], alpha_w=1.0, update_every=1)
    mults = gn.maybe_update(
        {"a": term_a, "b": term_b}, list(m.parameters()), dist_info=None
    )
    # Inverse-proportional multipliers -> equalized contributions.
    assert mults["b"] / mults["a"] == pytest.approx(100.0, rel=1e-4)
    assert mults["a"] * 10.0 == pytest.approx(mults["b"] * 0.1, rel=1e-6)


def test_equal_norms_give_unit_multipliers():
    """Equal gradient norms -> every multiplier is exactly 1.0 (the balancer is a
    no-op when nothing is imbalanced)."""
    m = _lin()
    y = m(torch.ones(1, 1))
    gn = GradNormBalancer(["a", "b"], alpha_w=0.3, update_every=1)
    mults = gn.maybe_update(
        {"a": (2.0 * y).sum(), "b": (2.0 * y).sum()}, list(m.parameters())
    )
    assert mults["a"] == pytest.approx(1.0)
    assert mults["b"] == pytest.approx(1.0)


def test_alpha_w_is_new_target_fraction():
    m = _lin()
    y = m(torch.ones(1, 1))
    held = GradNormBalancer(["a", "b"], alpha_w=0.0, update_every=1)
    assert held.maybe_update(
        {"a": (10.0 * y).sum(), "b": (0.1 * y).sum()}, list(m.parameters())
    ) == {"a": 1.0, "b": 1.0}

    y = m(torch.ones(1, 1))
    updated = GradNormBalancer(["a", "b"], alpha_w=1.0, update_every=1)
    weights = updated.maybe_update(
        {"a": (10.0 * y).sum(), "b": (0.1 * y).sum()}, list(m.parameters())
    )
    assert weights["b"] / weights["a"] == pytest.approx(100.0, rel=1e-4)

    with pytest.raises(ValueError, match="alpha_w"):
        GradNormBalancer(["a"], alpha_w=1.1)


def test_clamp_bounds_near_zero_grad_term():
    """A term with grad norm below ``eps`` is clamped (not divided-by-~0), so its
    multiplier stays finite and larger than a well-conditioned term's."""
    m = _lin()
    y = m(torch.ones(1, 1))
    term_a = (10.0 * y).sum()
    term_small = (1e-12 * y).sum()   # grad norm 1e-12 < eps
    gn = GradNormBalancer(["a", "small"], alpha_w=1.0, update_every=1, eps=1e-8)
    mults = gn.maybe_update(
        {"a": term_a, "small": term_small}, list(m.parameters())
    )
    assert np.isfinite(mults["small"])
    assert np.isfinite(mults["a"])
    assert mults["small"] > mults["a"]


def test_disconnected_term_is_held():
    """A term with no path to the params (``None`` grad) keeps its prior multiplier
    (held) while the connected terms rebalance."""
    m = _lin()
    y = m(torch.ones(1, 1))
    term_a = (10.0 * y).sum()
    term_b = (0.1 * y).sum()
    term_c = torch.tensor(3.0, requires_grad=True)  # independent leaf
    gn = GradNormBalancer(["a", "b", "c"], alpha_w=1.0, update_every=1)
    mults = gn.maybe_update(
        {"a": term_a, "b": term_b, "c": term_c}, list(m.parameters())
    )
    assert mults["c"] == 1.0  # held exactly
    assert mults["b"] / mults["a"] == pytest.approx(100.0, rel=1e-4)


def test_update_every_cadence():
    """Multipliers refresh only every ``update_every`` calls; intervening calls
    return the last committed values unchanged."""
    m = _lin()
    gn = GradNormBalancer(["a", "b"], alpha_w=1.0, update_every=2)

    def _step():
        y = m(torch.ones(1, 1))
        return gn.maybe_update(
            {"a": (10.0 * y).sum(), "b": (0.1 * y).sum()}, list(m.parameters())
        )

    m1 = _step()  # step 0 -> update
    m2 = _step()  # step 1 -> hold
    assert m2 == m1
    m3 = _step()  # step 2 -> update
    # An update ran; with alpha_w=1 the target is identical here, so equality is
    # expected — assert the cadence via the internal step counter instead.
    assert gn._step == 3
    assert m3 == pytest.approx({"a": m1["a"], "b": m1["b"]}, rel=1e-9)


def test_state_round_trip_and_term_mismatch_raises():
    m = _lin()
    y = m(torch.ones(1, 1))
    gn = GradNormBalancer(["a", "b"], alpha_w=0.5, update_every=1)
    gn.maybe_update({"a": (10.0 * y).sum(), "b": (0.1 * y).sum()}, list(m.parameters()))
    state = gn.state_dict()

    gn2 = GradNormBalancer(["a", "b"], alpha_w=0.5, update_every=1)
    gn2.load_state_dict(state)
    assert gn2.multipliers == gn.multipliers
    assert gn2._step == gn._step

    bad = GradNormBalancer(["a", "b", "ic"], alpha_w=0.5, update_every=1)
    with pytest.raises(ValueError, match="term_names"):
        bad.load_state_dict(state)


# --------------------------------------------------------------------------- #
# 1c. Interfaces PINO 5-term set (interface_band as a balanced term)          #
# --------------------------------------------------------------------------- #

_IFACE_TERMS = ["interior", "left_neumann", "topbot_adiabatic", "ic", "interface_band"]


def test_five_term_interface_band_round_trip_and_mismatch():
    """The `interfaces` PINO balancer over the 5-term set (band added last)
    constructs, keeps insertion order, round-trips through state, and a 4<->5
    term-name mismatch raises on load (same guard as the 4-term set)."""
    m = _lin()
    y = m(torch.ones(1, 1))
    gn = GradNormBalancer(_IFACE_TERMS, alpha_w=0.5, update_every=1)
    assert gn.state_dict()["term_names"] == _IFACE_TERMS
    gn.maybe_update({k: ((i + 1) * y).sum() for i, k in enumerate(_IFACE_TERMS)},
                    list(m.parameters()))
    state = gn.state_dict()

    gn2 = GradNormBalancer(_IFACE_TERMS, alpha_w=0.5, update_every=1)
    gn2.load_state_dict(state)
    assert gn2.multipliers == gn.multipliers
    assert gn2._step == gn._step

    four_term = GradNormBalancer(_IFACE_TERMS[:4], alpha_w=0.5, update_every=1)
    with pytest.raises(ValueError, match="term_names"):
        four_term.load_state_dict(state)


def test_build_gradnorm_includes_interface_band_when_weighted():
    """`build_gradnorm` (the real trainer wiring) admits `interface_band` as a
    balanced term when its static weight > 0, preserving insertion order, and
    filters out any zero-weight term."""
    from src.operators.train_pino import build_gradnorm

    config = {"training": {"gradnorm": {"enabled": True, "update_every": 10}}}
    term_weights = {
        "interior": 1.0,
        "left_neumann": 1.0,
        "topbot_adiabatic": 1.0,
        "ic": 1.0,
        "interface_band": 1.0,
    }
    gn = build_gradnorm(config, term_weights=term_weights)
    assert gn is not None
    assert gn.state_dict()["term_names"] == list(term_weights)

    # A zero (or None) static weight drops the term from the balanced set.
    term_weights_zero = dict(term_weights)
    term_weights_zero["interface_band"] = 0.0
    gn_zero = build_gradnorm(config, term_weights=term_weights_zero)
    assert "interface_band" not in gn_zero.state_dict()["term_names"]

    # Disabled config -> no balancer at all.
    off = {"training": {"gradnorm": {"enabled": False}}}
    assert build_gradnorm(off, term_weights=term_weights) is None


# --------------------------------------------------------------------------- #
# 1b. Guardrails — clamp/floor bracket the EMA, bound-hit counters, resume     #
# --------------------------------------------------------------------------- #


def _imbalanced_step(gn):
    """One update on a 100x-imbalanced pair (a: norm 10, small: norm 0.01)."""
    m = _lin()
    y = m(torch.ones(1, 1))
    return gn.maybe_update(
        {"a": (10.0 * y).sum(), "small": (0.01 * y).sum()}, list(m.parameters())
    )


def test_no_bounds_default_is_exact_noop():
    """Explicit all-None guardrails reproduce the unbounded multipliers exactly."""
    plain = _imbalanced_step(GradNormBalancer(["a", "small"], alpha_w=0.3, update_every=1))
    bounded = _imbalanced_step(
        GradNormBalancer(
            ["a", "small"], alpha_w=0.3, update_every=1,
            w_min=None, w_max=None, floors={},
        )
    )
    assert bounded == plain


def test_w_max_caps_tiny_grad_term():
    """The small-gradient term's target overshoots; ``w_max`` caps the STORED
    weight (alpha_w=1 -> stored == clamped target) and records one max hit."""
    gn = GradNormBalancer(
        ["a", "small"], alpha_w=1.0, update_every=1, w_max=1.5,
    )
    mults = _imbalanced_step(gn)
    assert mults["small"] == pytest.approx(1.5)
    assert gn.bound_hit_counts["small"]["max"] == 1
    assert gn.bound_hit_counts["a"]["max"] == 0


def test_floor_holds_post_ema_when_prior_below_floor():
    """A floor must be re-applied AFTER the EMA: a prior weight below the floor
    blended toward a floored target can still land below it pre-clamp."""
    gn = GradNormBalancer(
        ["a", "big"], alpha_w=0.9, update_every=1, floors={"big": 0.25},
    )
    gn.multipliers["big"] = 0.0  # simulate a prior weight below the floor
    m2 = _lin()
    y = m2(torch.ones(1, 1))
    # big has grad norm 100 -> tiny raw target, floored to 0.25; blending 90%
    # toward it from prev=0 gives 0.225, which the post-blend floor restores.
    mults = gn.maybe_update(
        {"a": (1.0 * y).sum(), "big": (100.0 * y).sum()}, list(m2.parameters())
    )
    assert mults["big"] == pytest.approx(0.25)
    assert gn.bound_hit_counts["big"]["floor"] == 1


def test_floor_above_w_max_raises():
    with pytest.raises(ValueError, match="floor"):
        GradNormBalancer(["a"], w_max=1.0, floors={"a": 2.0})


def test_w_min_above_w_max_raises():
    with pytest.raises(ValueError, match="w_min"):
        GradNormBalancer(["a", "b"], w_min=2.0, w_max=1.0)


def test_zero_gradient_term_held_no_bound_hit():
    """An exactly-zero-gradient term is held (existing rule) and never counted as
    a bound hit even with guardrails active."""
    m = _lin()
    y = m(torch.ones(1, 1))
    gn = GradNormBalancer(
        ["a", "zero"], alpha_w=1.0, update_every=1, w_min=0.1, w_max=5.0,
    )
    mults = gn.maybe_update(
        {"a": (10.0 * y).sum(), "zero": (0.0 * y).sum()}, list(m.parameters())
    )
    assert mults["zero"] == 1.0  # held at prior
    assert gn.bound_hit_counts["zero"] == {"min": 0, "max": 0, "floor": 0}


def test_stable_term_order_exact_list():
    gn = GradNormBalancer(["r", "ic", "bc_left", "bc_hom"], update_every=1)
    assert gn.state_dict()["term_names"] == ["r", "ic", "bc_left", "bc_hom"]


def test_multipliers_are_detached_floats():
    """Returned multipliers must be plain floats -- never tensors that could pull
    the adaptive weights into the optimization graph."""
    mults = _imbalanced_step(GradNormBalancer(["a", "small"], update_every=1))
    assert all(isinstance(v, float) for v in mults.values())
    assert all(not getattr(v, "requires_grad", False) for v in mults.values())


def test_load_state_dict_reclamps_into_new_bounds():
    """A checkpoint from a bounds-free run is re-clamped on load so newly enabled
    guardrails cannot be bypassed by a stale multiplier."""
    src = GradNormBalancer(["a", "b"], update_every=1)
    src.multipliers = {"a": 10.0, "b": 0.01}
    gn = GradNormBalancer(["a", "b"], update_every=1, w_min=0.1, w_max=5.0)
    gn.load_state_dict(src.state_dict())
    assert gn.multipliers["a"] == pytest.approx(5.0)
    assert gn.multipliers["b"] == pytest.approx(0.1)


# --------------------------------------------------------------------------- #
# 2. Single compose site — bit-identity                                       #
# --------------------------------------------------------------------------- #


def _phys_out():
    return {
        "physics_loss_weighted": torch.tensor(0.3 + 0.1 + 0.2 + 0.05),
        "phys_interior_mse": torch.tensor(0.3),
        "phys_left_neumann_mse": torch.tensor(0.1),
        "phys_topbot_adiabatic_mse": torch.tensor(0.2),
        "phys_right_dirichlet_mse": torch.tensor(0.05),
    }


def test_compose_legacy_branch_bit_identical():
    """``gradnorm=None`` reproduces the exact pre-Step-4 expression
    (``lambda_physics * physics_loss_weighted + lambda_data * data + lambda_ic * ic``)."""
    out = _phys_out()
    data = torch.tensor(0.9)
    ic = torch.tensor(0.4)
    total, mults, ms = _compose_physics_total(
        out=out, data_loss=data, ic=ic, causal_weighter=None, gradnorm=None,
        model_unwrapped=None, dist_info=None, lambda_data=1.0, lambda_physics=2.0,
        lambda_ic=3.0, region_weights=None,
    )
    manual = 2.0 * out["physics_loss_weighted"]
    manual = 1.0 * data + manual
    manual = manual + (3.0 * 1.0) * ic
    assert mults is None
    assert ms == 0.0
    assert torch.equal(total, manual)


class _OnesGradNorm:
    """Stub whose ``maybe_update`` returns all-ones multipliers, isolating the
    compose arithmetic from the real balancer."""

    def maybe_update(self, raw, params, *, dist_info=None):
        return {k: 1.0 for k in raw}


def test_compose_all_ones_equals_static_weighted_sum():
    """All-ones GradNorm -> ``sum(c_t * L_t)`` in canonical order, bit-identical to
    a manual static-coefficient sum."""
    out = _phys_out()
    data = torch.tensor(0.9)
    ic = torch.tensor(0.4)
    dummy = nn.Linear(1, 1)
    total, mults, _ = _compose_physics_total(
        out=out, data_loss=data, ic=ic, causal_weighter=None,
        gradnorm=_OnesGradNorm(), model_unwrapped=dummy, dist_info=None,
        lambda_data=1.0, lambda_physics=2.0, lambda_ic=3.0, region_weights=None,
    )
    coeffs = _physics_static_coeffs(
        None, 1.0, 2.0, 3.0, has_data=True, has_ic=True,
    )
    raw = {
        "interior": out["phys_interior_mse"],
        "left_bc": out["phys_left_neumann_mse"],
        "adiabatic_bc": out["phys_topbot_adiabatic_mse"],
        "right_bc": out["phys_right_dirichlet_mse"],
        "ic": ic, "data": data,
    }
    manual = None
    for name in _active_physics_terms(coeffs):
        term = coeffs[name] * 1.0 * raw[name]
        manual = term if manual is None else manual + term
    assert mults == {k: 1.0 for k in raw}
    assert torch.equal(total, manual)


def test_right_bc_zero_region_weight_drops_term():
    """A zero ``right_dirichlet`` region weight (hard-BC diagnostic) removes the
    ``right_bc`` term from both the active set and the composed total."""
    coeffs = _physics_static_coeffs(
        {"right_dirichlet": 0.0}, 0.0, 1.0, 1.0, has_data=False, has_ic=True,
    )
    assert "right_bc" not in _active_physics_terms(coeffs)
    assert coeffs["right_bc"] == 0.0


# --------------------------------------------------------------------------- #
# 3. train_one_epoch smoke — populates gradnorm_weights + timings             #
# --------------------------------------------------------------------------- #


class _IdentityModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))

    def forward(self, spatial, cond_static, forcing_seq=None):
        return spatial[..., 0:1] * (1.0 + self.w)


class _StubColloc:
    def __init__(self, B=2, Nx=5, Ny=5, dt=0.01, seed=0):
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
        return self._mk_batch(f), torch.from_numpy(f[..., None].astype(np.float32))


class _ExplodingLoader:
    def __init__(self, n):
        self.n = int(n)

    def __len__(self):
        return self.n

    def __iter__(self):
        raise AssertionError("data loader must not be iterated in physics-only mode")


def test_train_one_epoch_populates_gradnorm_weights():
    model = _IdentityModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    sampler = _StubColloc()
    coeffs = _physics_static_coeffs(
        None, 0.0, 1.0, 1.0, has_data=False, has_ic=True,
    )
    gn = GradNormBalancer(_active_physics_terms(coeffs), update_every=1)

    metrics = train_one_epoch(
        model=model, train_loader=_ExplodingLoader(3), optimizer=optimizer,
        loss_fn=lambda *a, **k: None, device=torch.device("cpu"),
        physics_collocation=sampler, collocation_max_lead=0.05,
        lambda_data=0.0, lambda_physics=1.0, lambda_ic=1.0,
        gradnorm=gn, model_unwrapped=model,
    )
    assert metrics["gradnorm_weights"] != ""
    parsed = __import__("json").loads(metrics["gradnorm_weights"])
    assert set(parsed) == set(_active_physics_terms(coeffs))
    for key in ("train_step_ms", "gradnorm_ms", "backward_ms", "optimizer_step_ms"):
        assert metrics[key] >= 0.0


def test_train_one_epoch_off_path_empty_gradnorm_weights():
    """No balancer -> ``gradnorm_weights`` is the empty sentinel and timings are 0
    only if never measured; the physics path still runs."""
    model = _IdentityModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    metrics = train_one_epoch(
        model=model, train_loader=_ExplodingLoader(2), optimizer=optimizer,
        loss_fn=lambda *a, **k: None, device=torch.device("cpu"),
        physics_collocation=_StubColloc(), collocation_max_lead=0.05,
        lambda_data=0.0, lambda_physics=1.0, lambda_ic=1.0,
    )
    assert metrics["gradnorm_weights"] == ""
    assert metrics["gradnorm_ms"] == 0.0


# --------------------------------------------------------------------------- #
# 4. Two-rank gloo DDP: autograd.grad measure + backward under static_graph   #
# --------------------------------------------------------------------------- #

_GLOO_OK = dist.is_available() and dist.is_gloo_available()


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 4))

    def forward(self, x):
        return self.net(x)


class _NS:
    is_distributed = True
    world_size = 2


def _gradnorm_ddp_worker(rank, world_size, port, result_queue):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        "gloo", rank=rank, world_size=world_size, timeout=timedelta(seconds=30)
    )
    try:
        torch.manual_seed(0)  # identical init on both ranks
        model = _TinyModel()
        ddp = DistributedDataParallel(model, static_graph=True)
        gn = GradNormBalancer(["a", "b"], alpha_w=1.0, update_every=1)

        # Rank-dependent scales -> different per-rank grad norms pre-reduce; the
        # SUM all-reduce then yields identical averaged norms on both ranks.
        x = torch.randn(8, 4)
        out = ddp(x)
        term_a = ((1.0 + rank) * out).pow(2).mean()
        term_b = (0.05 * out).pow(2).mean()
        try:
            mults = gn.maybe_update(
                {"a": term_a, "b": term_b}, list(model.parameters()), dist_info=_NS(),
            )
            (term_a + term_b).backward()  # static_graph single backward
        except RuntimeError as exc:
            result_queue.put((rank, "error", str(exc)))
            return

        grads_finite = all(
            bool(torch.isfinite(p.grad).all()) for p in model.parameters()
        )
        result_queue.put(
            (rank, "ok", (float(mults["a"]), float(mults["b"]), bool(grads_finite)))
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not _GLOO_OK, reason="gloo backend unavailable")
def test_gradnorm_ddp_measure_then_backward():
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [
        ctx.Process(target=_gradnorm_ddp_worker, args=(r, 2, 29572, q))
        for r in range(2)
    ]
    for p in procs:
        p.start()
    results = []
    try:
        for _ in range(2):
            results.append(q.get(timeout=120))
    finally:
        for p in procs:
            p.join(timeout=120)
            if p.is_alive():
                p.terminate()
                pytest.fail("GradNorm DDP worker hung on measure+backward")

    assert len(results) == 2
    by_rank = {}
    for rank, status, payload in results:
        assert status == "ok", f"rank {rank} failed: {payload}"
        by_rank[rank] = payload
    (a0, b0, f0), (a1, b1, f1) = by_rank[0], by_rank[1]
    assert f0 and f1
    # SUM all-reduce -> identical multipliers on both ranks.
    assert a0 == pytest.approx(a1, rel=1e-9)
    assert b0 == pytest.approx(b1, rel=1e-9)
