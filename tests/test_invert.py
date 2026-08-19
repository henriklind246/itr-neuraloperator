"""Tests for the source_itr inverse solver (Stage 1 MVP).

The load-bearing checks are checkpoint-free: the torch reparameterization must
reproduce the two theta injection points *bit-for-bit* (within float tolerance)
against the real forward pipeline:

  1. cond_static[1:5]  <- build_cond_vector_itr  (linear over RC_VOID_RANGES)
  2. spatial[..., 4]   <- _rc_channel            (rc_log_norm o make_rc_void_profile)

A mismatch in either silently queries the FNO off-distribution and degrades
inversion, so these are the first thing to nail. The operator-wiring and
gradient-flow tests use a tiny untrained FNO (recovery quality needs a trained
model and is exercised by the end-to-end Stage 1 run, not unit tests).
"""

import numpy as np
import pytest
import torch

from problems.forcing import FORCING_TEMPORAL_TOKEN_DIM
from problems.source import _patch_center_ranges
from problems.source_itr import (
    RC_Y_CHANNEL,
    build_cond_vector_itr,
    rc_log_norm,
)
from src.physics.internal_source import (
    RC_MIN,
    RC_SIN_RANGES,
    RC_VOID_RANGES,
    R_PEAK_MAX,
    make_rc_sin_profile,
    make_rc_void_profile,
)
from scripts import invert as inv
from scripts.inverse_adapters import (
    ForcingAdapter,
    ForcingItrSinAdapter,
    InverseAdapter,
    SourceItrAdapter,
    SourceItrSinAdapter,
)


def test_legacy_forcing_checkpoint_adaptation_is_exact_at_zero_source_time():
    from problems.registry import get_problem

    dims = get_problem("forcing_itr", "temporal_encoder").dims
    embed_dim = 4
    config = {
        "benchmark": {"name": "forcing_itr"},
        "model": {
            "parameters": {
                "cond_static_dim": dims.cond_static_dim + 1,
                "forcing_embed_dim": embed_dim,
            }
        },
    }
    generator = torch.Generator().manual_seed(0)
    cond_weight = torch.randn(
        3, dims.cond_static_dim + 1 + embed_dim, generator=generator
    )
    aug_weight = torch.randn(3, embed_dim + 2, generator=generator)
    state = {
        "cond_mlp.net.0.weight": cond_weight,
        "forcing_aug_mlp.0.weight": aug_weight,
    }

    adapted = inv._adapt_legacy_zero_source_time_state(state, config, dims)
    current_cond = torch.randn(2, dims.cond_static_dim, generator=generator)
    forcing_embed = torch.randn(2, embed_dim, generator=generator)
    legacy_cond = torch.cat(
        [current_cond[:, :1], torch.zeros(2, 1), current_cond[:, 1:]], dim=1
    )
    legacy_full = torch.cat([legacy_cond, forcing_embed], dim=1)
    current_full = torch.cat([current_cond, forcing_embed], dim=1)
    torch.testing.assert_close(
        legacy_full @ cond_weight.T,
        current_full @ adapted["cond_mlp.net.0.weight"].T,
    )

    legacy_aug = torch.cat(
        [forcing_embed, current_cond[:, :1], torch.zeros(2, 1)], dim=1
    )
    current_aug = torch.cat([forcing_embed, current_cond[:, :1]], dim=1)
    torch.testing.assert_close(
        legacy_aug @ aug_weight.T,
        current_aug @ adapted["forcing_aug_mlp.0.weight"].T,
    )
    assert state["cond_mlp.net.0.weight"] is cond_weight
    assert state["forcing_aug_mlp.0.weight"] is aug_weight


# A handful of physical thetas spanning the box, including the dependent-ceiling
# edge (R_amp near R_PEAK_MAX - R_base) and the no-void floor (R_amp ~ 0).
THETAS = [
    (0.20, 0.50, 0.50, 0.10),
    (0.05, 2.90, 0.30, 0.05),   # near amp ceiling at the R_base floor
    (0.90, 0.10, 0.80, 0.20),
    (0.50, 0.00, 0.10, 0.08),   # no-void floor
    (0.75, 1.20, 0.65, 0.15),
]


@pytest.mark.parametrize("theta", THETAS)
def test_cond_slice_matches_build_cond_vector_itr(theta):
    R_base, R_amp, y0, sigma = theta
    # Reference: the real cond builder. Patch slots are irrelevant here; we only
    # compare the void slice cond[1:5], so any valid patch is fine.
    x_lo, x_hi, y_lo, y_hi = 0.0, 1.0, 0.0, 1.0
    w_h = h_h = 0.1
    xcr, ycr = _patch_center_ranges(x_lo, x_hi, y_lo, y_hi, w_h, h_h)
    cond = build_cond_vector_itr(
        t_bar_norm=0.3,
        R_base=R_base, R_amp=R_amp, y0=y0, sigma=sigma,
        x_h=0.5, y_h=0.5, w_h=w_h, h_h=h_h,
        x_center_range=xcr, y_center_range=ycr,
        x_length_scale=(x_hi - x_lo), y_length_scale=(y_hi - y_lo),
    )
    ref_slice = cond[1:5]

    theta_t = torch.tensor(theta, dtype=torch.float64)
    got = inv.theta_to_cond_slice(theta_t).numpy()
    np.testing.assert_allclose(got, ref_slice, rtol=0, atol=1e-6)


@pytest.mark.parametrize("theta", THETAS)
def test_rc_channel_matches_reference(theta):
    R_base, R_amp, y0, sigma = theta
    Ny, Nx = 100, 100
    y_grid = np.linspace(0.0, 1.0, Ny).astype(np.float32)

    Rc_y = make_rc_void_profile(y_grid, R_base=R_base, R_amp=R_amp, y0=y0, sigma=sigma)
    ref_channel = np.broadcast_to(rc_log_norm(Rc_y)[None, :], (Nx, Ny))

    theta_t = torch.tensor(theta, dtype=torch.float64)
    y_t = torch.from_numpy(y_grid).to(torch.float64)
    got = inv.theta_to_rc_channel(theta_t, y_t, Nx).numpy()

    assert got.shape == (Nx, Ny)
    np.testing.assert_allclose(got, ref_channel, rtol=0, atol=1e-5)


@pytest.mark.parametrize("theta", THETAS)
def test_unconstrained_theta_roundtrip(theta):
    theta_t = torch.tensor(theta, dtype=torch.float64)
    u = inv.unconstrained_from_theta(theta_t)
    back = inv.theta_from_unconstrained(u)
    np.testing.assert_allclose(back.numpy(), theta_t.numpy(), rtol=0, atol=1e-5)


def test_theta_from_unconstrained_respects_box_and_ceiling():
    rng = np.random.default_rng(0)
    u = torch.from_numpy(rng.normal(0, 3.0, size=(500, 4)))
    theta = inv.theta_from_unconstrained(u)
    R_base, R_amp, y0, sigma = (theta[:, i] for i in range(4))

    base_lo, base_hi = RC_VOID_RANGES["R_base"]
    y0_lo, y0_hi = RC_VOID_RANGES["y0"]
    sig_lo, sig_hi = RC_VOID_RANGES["sigma"]

    assert torch.all(R_base >= base_lo - 1e-9) and torch.all(R_base <= base_hi + 1e-9)
    assert torch.all(y0 >= y0_lo - 1e-9) and torch.all(y0 <= y0_hi + 1e-9)
    assert torch.all(sigma >= sig_lo - 1e-9) and torch.all(sigma <= sig_hi + 1e-9)
    # Dependent amplitude ceiling: R_amp <= R_PEAK_MAX - R_base, i.e. peak <= R_PEAK_MAX.
    assert torch.all(R_amp >= -1e-9)
    assert torch.all(R_base + R_amp <= R_PEAK_MAX + 1e-6)


def test_integrated_excess_resistance_matches_numpy():
    theta = torch.tensor([0.3, 1.0, 0.5, 0.12], dtype=torch.float64)
    y_grid = torch.linspace(0.0, 1.0, 200, dtype=torch.float64)
    got = inv.integrated_excess_resistance(theta, y_grid)

    y = y_grid.numpy()
    excess = 1.0 * np.exp(-(((y - 0.5) / 0.12) ** 2))
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    ref = float(trapz(excess, y))
    assert abs(got - ref) < 1e-9


# ---------------------------------------------------------------------------
# Operator wiring + gradient flow, using a tiny untrained source_itr FNO.
# ---------------------------------------------------------------------------

def _tiny_source_itr_model():
    from src.operators.fno2d import FNO2d
    from problems.source_itr import COND_STATIC_DIM, SPATIAL_CHANNELS_TEMPORAL
    from problems.forcing import FORCING_TEMPORAL_TOKEN_DIM

    torch.manual_seed(1)
    return FNO2d(
        modes1=2, modes2=2, width=8,
        in_channels=SPATIAL_CHANNELS_TEMPORAL, out_channels=1, n_layers=2,
        cond_static_dim=COND_STATIC_DIM,
        temporal_token_dim=FORCING_TEMPORAL_TOKEN_DIM,
        temporal_hidden=16, forcing_embed_dim=16,
        use_temporal_encoder=True, use_forcing_time_aug=False,
        s_y_channel=3,
    )


def _fake_observation_set(Nx=16, Ny=16, N=3, M=128):
    """Minimal ObservationSet with random scaffolding for wiring/grad tests."""
    spatial = torch.zeros(N, Nx, Ny, 5)
    spatial[..., 1] = torch.linspace(0, 1, Nx)[None, :, None]
    spatial[..., 2] = torch.linspace(0, 1, Ny)[None, None, :]
    cond = torch.zeros(N, 9)
    cond[:, 0] = torch.linspace(0.3, 1.0, N)  # t_bar_norm early/mid/late
    forcing_seq = torch.zeros(N, M, FORCING_TEMPORAL_TOKEN_DIM)
    forcing_seq[..., 0] = torch.linspace(0, 1, M)
    targets = torch.zeros(N, Nx, Ny, 1)
    y_grid = torch.linspace(0, 1, Ny)
    return inv.ObservationSet(
        sid=0, time_indices=list(range(N)),
        spatial=spatial, cond=cond, forcing_seq=forcing_seq,
        targets=targets, y_grid=y_grid, Nx=Nx,
        theta_true=torch.tensor([0.3, 1.0, 0.5, 0.12]),
    )


def test_predict_fullfield_injects_both_points_and_is_differentiable():
    model = _tiny_source_itr_model().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    obs = _fake_observation_set()

    u = torch.zeros(4, requires_grad=True)
    theta = inv.theta_from_unconstrained(u)
    pred = inv.predict_fullfield(model, obs, theta)
    assert pred.shape == (obs.spatial.shape[0], obs.Nx, obs.spatial.shape[2], 1)

    loss = (pred ** 2).mean()
    loss.backward()
    # Gradient must flow back to all four unconstrained params through the two
    # injection points (channel 4 + cond[1:5]).
    assert u.grad is not None
    assert torch.all(torch.isfinite(u.grad))
    assert torch.any(u.grad != 0)


def test_predict_fullfield_changes_with_theta():
    model = _tiny_source_itr_model().eval()
    obs = _fake_observation_set()
    theta_a = torch.tensor([0.2, 0.1, 0.3, 0.08])
    theta_b = torch.tensor([0.8, 2.0, 0.7, 0.18])
    with torch.no_grad():
        pa = inv.predict_fullfield(model, obs, theta_a)
        pb = inv.predict_fullfield(model, obs, theta_b)
    assert not torch.allclose(pa, pb)


def test_invert_sim_recovers_theta_against_self_generated_target():
    """End-to-end loop sanity: with the tiny model as its *own* ground truth,
    the optimizer drives the data loss down and lands a finite theta in-box.

    This is a wiring/convergence check on the optimizer plumbing, not a
    capability claim — the model is untrained, so "truth" here is whatever the
    model maps a planted theta to.
    """
    model = _tiny_source_itr_model().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    obs = _fake_observation_set()

    theta_star = torch.tensor([0.4, 1.5, 0.55, 0.13])
    with torch.no_grad():
        obs.targets = inv.predict_fullfield(model, obs, theta_star).detach()

    cfg = inv.InversionConfig(n_starts=4, adam_steps=150, adam_lr=0.1,
                              lbfgs_steps=30, seed=1)
    result = inv.invert_sim(model, obs, cfg)
    assert result.loss < 1e-4
    # Landed theta is inside the physical box.
    th = result.theta_hat
    assert 0.05 - 1e-6 <= th[0] <= 1.0 + 1e-6
    assert th[1] >= -1e-6 and (th[0] + th[1]) <= R_PEAK_MAX + 1e-4


# ---------------------------------------------------------------------------
# Stage 2: interface-proximal sensor operator M (masked observations).
# ---------------------------------------------------------------------------

def test_sensor_mask_selects_interface_band_and_rows():
    Nx, Ny = 21, 17
    x_grid = np.linspace(0.0, 1.0, Nx)
    y_grid = np.linspace(0.0, 1.0, Ny)
    mask = inv.build_interface_sensor_mask(
        x_grid, y_grid, interface_x=0.5, x_halfwidth=0.05, n_y=5
    ).numpy()

    assert mask.shape == (Nx, Ny)
    # Selected columns are exactly those within the band of the interface.
    sel_cols = np.unique(np.where(mask.any(axis=1))[0])
    expected_cols = np.where(np.abs(x_grid - 0.5) <= 0.05)[0]
    np.testing.assert_array_equal(sel_cols, expected_cols)
    # Each selected column carries the same set of (n_y) rows.
    sel_rows = np.where(mask[sel_cols[0]])[0]
    assert len(sel_rows) == 5
    for c in sel_cols:
        np.testing.assert_array_equal(np.where(mask[c])[0], sel_rows)


def test_sensor_mask_falls_back_to_nearest_column_when_band_empty():
    Nx, Ny = 21, 9
    x_grid = np.linspace(0.0, 1.0, Nx)  # spacing 0.05
    y_grid = np.linspace(0.0, 1.0, Ny)
    # A sub-spacing half-width that straddles no grid node off-center: only the
    # exact interface column (x=0.5 is a node here) should survive.
    mask = inv.build_interface_sensor_mask(
        x_grid, y_grid, interface_x=0.52, x_halfwidth=1e-4, n_y=Ny
    ).numpy()
    sel_cols = np.unique(np.where(mask.any(axis=1))[0])
    assert sel_cols.tolist() == [int(np.argmin(np.abs(x_grid - 0.52)))]


def test_apply_sensor_mask_gather_and_fullfield_flatten():
    N, Nx, Ny, C = 2, 4, 3, 1
    field = torch.arange(N * Nx * Ny * C, dtype=torch.float32).reshape(N, Nx, Ny, C)

    # Full-field: None mask flattens space, preserving every value.
    flat = inv.apply_sensor_mask(field, None)
    assert flat.shape == (N, Nx * Ny, C)
    assert torch.equal(flat, field.reshape(N, Nx * Ny, C))

    # Masked: pick two specific pixels and confirm the gather matches manual indexing.
    mask = torch.zeros(Nx, Ny, dtype=torch.bool)
    mask[1, 2] = True
    mask[3, 0] = True
    got = inv.apply_sensor_mask(field, mask)
    assert got.shape == (N, 2, C)
    expected = torch.stack([field[:, 1, 2, :], field[:, 3, 0, :]], dim=1)
    assert torch.equal(got, expected)


def test_data_loss_fullfield_matches_none_mask():
    model = _tiny_source_itr_model().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    obs = _fake_observation_set()
    obs.targets = torch.randn_like(obs.targets)
    u = torch.zeros(4)

    # mask=None is the full-field path; an all-True mask must give the same loss.
    obs.mask = None
    loss_none = inv._data_loss(model, obs, u, 0.0)
    obs.mask = torch.ones(obs.Nx, obs.spatial.shape[2], dtype=torch.bool)
    loss_all = inv._data_loss(model, obs, u, 0.0)
    assert torch.allclose(loss_none, loss_all, atol=1e-6)


def test_invert_sim_recovers_theta_with_sparse_sensors():
    """Stage 2 wiring: with a sparse interface-proximal mask and the tiny model
    as its own ground truth, the optimizer still drives the masked data loss
    down and lands an in-box theta. Wiring/plumbing check, not a capability
    claim (untrained model)."""
    model = _tiny_source_itr_model().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    obs = _fake_observation_set(Nx=16, Ny=16)

    Nx, Ny = obs.Nx, obs.spatial.shape[2]
    x_grid = np.linspace(0.0, 1.0, Nx)
    y_grid = np.linspace(0.0, 1.0, Ny)
    obs.mask = inv.build_interface_sensor_mask(
        x_grid, y_grid, interface_x=0.5, x_halfwidth=0.1, n_y=6
    )
    assert int(obs.mask.sum()) < Nx * Ny  # genuinely sparse

    theta_star = torch.tensor([0.4, 1.5, 0.55, 0.13])
    with torch.no_grad():
        obs.targets = inv.predict_fullfield(model, obs, theta_star).detach()

    cfg = inv.InversionConfig(n_starts=4, adam_steps=200, adam_lr=0.1,
                              lbfgs_steps=40, seed=2)
    result = inv.invert_sim(model, obs, cfg)
    assert result.loss < 1e-4
    th = result.theta_hat
    assert 0.05 - 1e-6 <= th[0] <= 1.0 + 1e-6
    assert th[1] >= -1e-6 and (th[0] + th[1]) <= R_PEAK_MAX + 1e-4


# ---------------------------------------------------------------------------
# Stage 3: LHS multistart + dG/dtheta SVD identifiability diagnostics.
# ---------------------------------------------------------------------------

def test_lhs_starts_are_stratified():
    """Each void scalar's box fraction has exactly one start per 1/n stratum."""
    rng = np.random.default_rng(0)
    n = 8
    u = inv.lhs_starts_u(n, rng)
    assert u.shape == (n, 4)
    frac = 1.0 / (1.0 + np.exp(-u))  # sigmoid: u -> box fraction
    for j in range(4):
        s = np.sort(frac[:, j])
        for k in range(n):
            assert k / n - 1e-9 <= s[k] <= (k + 1) / n + 1e-9


def test_lhs_starts_map_into_theta_box():
    rng = np.random.default_rng(1)
    u = torch.from_numpy(inv.lhs_starts_u(16, rng))
    theta = inv.theta_from_unconstrained(u)
    R_base, R_amp, y0, sigma = (theta[:, i] for i in range(4))
    assert torch.all(R_base >= 0.05 - 1e-6) and torch.all(R_base <= 1.0 + 1e-6)
    assert torch.all(R_amp >= -1e-6)
    assert torch.all(R_base + R_amp <= R_PEAK_MAX + 1e-4)  # dependent ceiling
    assert torch.all(y0 >= 0.1 - 1e-6) and torch.all(y0 <= 0.9 + 1e-6)
    assert torch.all(sigma >= 0.05 - 1e-6) and torch.all(sigma <= 0.2 + 1e-6)


def test_invert_sim_lhs_and_normal_both_recover():
    """The new default LHS starts and the legacy normal starts both converge."""
    model = _tiny_source_itr_model().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    obs = _fake_observation_set()
    theta_star = torch.tensor([0.4, 1.5, 0.55, 0.13])
    with torch.no_grad():
        obs.targets = inv.predict_fullfield(model, obs, theta_star).detach()
    for sampling in ("lhs", "normal"):
        cfg = inv.InversionConfig(n_starts=4, adam_steps=150, adam_lr=0.1,
                                  lbfgs_steps=30, seed=1, start_sampling=sampling)
        result = inv.invert_sim(model, obs, cfg)
        assert result.loss < 1e-4, sampling


def test_observation_jacobian_shape_finite_nonzero():
    model = _tiny_source_itr_model().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    obs = _fake_observation_set(Nx=8, Ny=8, N=2)
    theta = torch.tensor([0.4, 1.2, 0.5, 0.12])

    # Full-field: m = N * Nx * Ny * C.
    J = inv.observation_jacobian(model, obs, theta)
    m_full = obs.spatial.shape[0] * obs.Nx * obs.spatial.shape[2] * 1
    assert J.shape == (m_full, 4)
    assert torch.all(torch.isfinite(J))
    assert torch.any(J != 0)

    # Sparse: m shrinks to N * n_sensors * C.
    obs.mask = inv.build_interface_sensor_mask(
        np.linspace(0.0, 1.0, 8), np.linspace(0.0, 1.0, 8),
        interface_x=0.5, x_halfwidth=0.2, n_y=4,
    )
    Js = inv.observation_jacobian(model, obs, theta)
    assert Js.shape == (obs.spatial.shape[0] * int(obs.mask.sum()) * 1, 4)


def test_svd_identifiability_flags_ramp_sigma_ridge():
    """A Jacobian whose R_amp and sigma columns are collinear must surface a
    near-null direction lying in the (R_amp, sigma) plane — the void-mass ridge.
    """
    rng = np.random.default_rng(0)
    m = 50
    a = rng.normal(size=m)   # R_base response
    c = rng.normal(size=m)   # y0 response
    d = rng.normal(size=m)   # shared amp/sigma response
    J = np.stack([a, 2.0 * d, c, 1.0 * d], axis=1)  # cols 1,3 collinear (2:1)
    rep = inv.svd_identifiability(J)
    S = rep["singular_values"]
    assert S[-1] < 1e-8 * S[0]                      # genuinely rank-deficient
    assert rep["cond_number"] > 1e7
    assert rep["ramp_sigma_alignment"] > 0.99       # null dir is the ridge
    least = rep["least_identified_dir"]
    assert abs(least[0]) < 1e-6 and abs(least[2]) < 1e-6


def test_svd_identifiability_flags_weak_parameter():
    """When R_base barely moves the observations, it is the least-identified
    axis and the ridge-alignment metric stays low (not an amp/sigma issue)."""
    rng = np.random.default_rng(1)
    cols = rng.normal(size=(60, 4))
    cols[:, 0] *= 1e-6  # R_base nearly invisible
    rep = inv.svd_identifiability(cols)
    least = rep["least_identified_dir"]
    assert abs(least[0]) > 0.99
    assert rep["ramp_sigma_alignment"] < 0.05
    assert rep["param_sensitivity"][0] < rep["param_sensitivity"][1:].min()


def test_sensitivity_report_orthonormal_and_summary_columns():
    model = _tiny_source_itr_model().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    obs = _fake_observation_set(Nx=8, Ny=8, N=2)
    theta = torch.tensor([0.4, 1.2, 0.5, 0.12])
    rep = inv.sensitivity_report(model, obs, theta)

    S = rep["singular_values"]
    assert np.all(np.diff(S) <= 1e-9)                       # descending
    V = rep["right_vectors"]
    np.testing.assert_allclose(V.T @ V, np.eye(4), atol=1e-6)  # orthonormal
    assert rep["cond_number"] >= 1.0 - 1e-9
    assert 0.0 <= rep["ramp_sigma_alignment"] <= 1.0 + 1e-9

    flat = inv.sensitivity_summary(rep)
    for k in ("cond_number", "ramp_sigma_alignment",
              "sv_0", "sens_R_amp", "least_dir_sigma"):
        assert k in flat and np.isfinite(flat[k])


# ---------------------------------------------------------------------------
# Stage 4: FV refinement + theta-local surrogate error.
#
# The load-bearing check is checkpoint-free and needs no trained model: a tiny
# *real* source_itr FV dataset is generated with the exact data-generation
# operator, and ``fv_predict_masked`` at the stored theta_true must reproduce
# the stored trajectory snapshots. This proves the FV operator is rebuilt
# correctly (k=3/35 layers, interface_R=[Rc(y)], the patch source, the IC, the
# generation constants) so the FNO MAP can be honestly re-checked against the
# real solver.
# ---------------------------------------------------------------------------

def _tiny_source_itr_fv_dataset(Nx=12, Ny=12, num_sims=2, t_final=0.06, dt=0.01):
    """Build a tiny *real* source_itr FV dataset (save_stride=1) in memory.

    Mirrors ``data/generate_dataset.py`` with the same generation constants the
    inverse solver's ``build_fv_base_kwargs`` reconstructs, so the FV solve
    inside ``fv_predict_masked`` reproduces these trajectories bit-for-bit (up to
    the float32 storage cast). Returns ``(ds, mu_global, sigma_global)``.
    """
    from problems.registry import get_problem
    from src.physics.fv_solver_1d import build_time_grid
    from data.dataset import SnapshotPairDataset, compute_global_stats

    spec = get_problem("source_itr", "temporal_encoder")
    a, b, c, d = 0.0, 1.0, 0.0, 1.0
    x_grid = np.linspace(a, b, Nx)
    y_grid = np.linspace(c, d, Ny)
    X, Y = np.meshgrid(x_grid, y_grid, indexing="ij")
    dt, t_grid_template = build_time_grid(t_final, dt, explicit_dt=True)

    grids = {"X": X, "Y": Y, "x_grid": x_grid, "y_grid": y_grid}
    time_cfg = dict(num_sims=num_sims, dt=dt, t_final=t_final, lhs_seed=0,
                    t_on=0.0, t_off=0.2, phase=0.0, tukey_alpha=0.5)
    sim_params = spec.sample_sim_params(
        rng=np.random.default_rng(0), rng_profile=np.random.default_rng(1),
        grids=grids, time_cfg=time_cfg,
    )

    base_kwargs = dict(
        a=a, b=b, c=c, d=d, Nx=Nx, Ny=Ny, lam_target=0.8, t_final=t_final,
        flux_f=0.0, flux_A=0.0, t_on=0.0, phase=0.0, dt=dt, tukey_alpha=0.5,
        y_grid=y_grid,
    )
    Nt = len(t_grid_template)
    trajectories = np.zeros((num_sims, Nt, Nx, Ny), dtype=np.float32)
    t_grid = None
    for i, params in enumerate(sim_params):
        solver = spec.configure_solver(params, base_kwargs)
        t, _x, _y, T_hist = solver.solve(T0=params["T0"], store_trajectory=True)
        trajectories[i] = np.asarray(T_hist, dtype=np.float32)
        t_grid = t
    t_grid = np.asarray(t_grid, dtype=np.float32)

    mu_global, sigma_global = compute_global_stats(trajectories, np.arange(num_sims))
    ds = SnapshotPairDataset(
        trajectories=trajectories,
        sim_params=np.array(sim_params, dtype=object),
        t_grid=t_grid,
        x_grid=x_grid.astype(np.float32),
        y_grid=y_grid.astype(np.float32),
        sim_ids=np.arange(num_sims),
        mu_global=mu_global,
        sigma_global=sigma_global,
        problem=spec,
        dt=float(dt),
    )
    ds._split_ids = {"train": np.array([0]), "val": np.array([], dtype=int),
                     "test": np.arange(num_sims)}
    return ds, float(mu_global), float(sigma_global)


def _write_inverse_dataset_dir(ds, tmp_path):
    np.save(tmp_path / "trajectories.npy", np.asarray(ds.trajectories))
    np.save(tmp_path / "x_grid.npy", np.asarray(ds.x_grid))
    np.save(tmp_path / "y_grid.npy", np.asarray(ds.y_grid))
    np.save(tmp_path / "t_grid.npy", np.asarray(ds.t_grid))
    np.save(tmp_path / "sim_params.npy", np.asarray(ds.sim_params, dtype=object))
    np.save(tmp_path / "dt.npy", np.asarray(float(ds.dt)))


def test_build_dataset_from_dir_disjoint_partition(tmp_path):
    """The held-out dataset splits into two *disjoint* slices: the first
    ``n_invert`` sims are inverted (``test``) and the next ``n_calibration`` are
    reserved for surrogate calibration (``val``), never overlapping, with no
    training reservation. Keeping them disjoint is what stops calibration from
    measuring surrogate error on the very sims being inverted."""
    ds, mu_global, sigma_global = _tiny_source_itr_fv_dataset(num_sims=5)
    _write_inverse_dataset_dir(ds, tmp_path)

    config = {"benchmark": {"name": "source_itr", "representation": "temporal_encoder"}}
    built = inv.build_dataset_from_dir(
        str(tmp_path), config, mu_global=mu_global, sigma_global=sigma_global,
        n_invert=2, n_calibration=3,
    )

    assert np.asarray(built._split_ids["train"]).size == 0
    np.testing.assert_array_equal(built._split_ids["test"], np.array([0, 1]))
    np.testing.assert_array_equal(built._split_ids["val"], np.array([2, 3, 4]))
    # Disjoint: no sim id appears in both slices.
    assert not (set(built._split_ids["test"].tolist())
                & set(built._split_ids["val"].tolist()))
    assert inv._split_name_for(built, 0) == "test"
    assert inv._split_name_for(built, 3) == "val"


def test_build_dataset_from_dir_rejects_oversized_partition(tmp_path):
    """The partition cannot request more sims than the dataset holds."""
    ds, mu_global, sigma_global = _tiny_source_itr_fv_dataset(num_sims=4)
    _write_inverse_dataset_dir(ds, tmp_path)
    config = {"benchmark": {"name": "source_itr", "representation": "temporal_encoder"}}
    with pytest.raises(ValueError):
        inv.build_dataset_from_dir(
            str(tmp_path), config, mu_global=mu_global, sigma_global=sigma_global,
            n_invert=3, n_calibration=3,
        )


def test_prepare_inversion_dataset_generates_disjoint_slices(tmp_path):
    """The shared helper generates one dataset disjoint from training (non-zero
    seed) and returns contiguous, disjoint inversion/calibration slices."""
    calls = {}

    def fake_generate(*, num_sims, save_dir, benchmark, nx, ny, save_stride, rng_seed):
        calls.update(
            num_sims=num_sims, save_dir=save_dir, benchmark=benchmark,
            nx=nx, ny=ny, save_stride=save_stride, rng_seed=rng_seed,
        )

    out_dir = tmp_path / "inversion_data"
    partition = inv.prepare_inversion_dataset(
        benchmark="forcing", n_invert=3, n_calibration=5,
        out_dir=str(out_dir), generate_fn=fake_generate,
    )

    assert calls["num_sims"] == 8  # n_invert + n_calibration
    assert calls["benchmark"] == "forcing"
    assert calls["rng_seed"] == inv.INVERSION_DATASET_SEED != 0
    assert partition["data_dir"] == str(out_dir)
    assert partition["invert_ids"] == [0, 1, 2]
    assert partition["calibration_ids"] == [3, 4, 5, 6, 7]
    # Disjoint and exhaustive over the generated corpus.
    assert set(partition["invert_ids"]) & set(partition["calibration_ids"]) == set()


@pytest.fixture(scope="module")
def tiny_fv_dataset():
    return _tiny_source_itr_fv_dataset()


def test_build_fv_base_kwargs_raises_without_dt(tiny_fv_dataset):
    ds, _, _ = tiny_fv_dataset
    ds.dt = None
    try:
        with pytest.raises(ValueError):
            inv.build_fv_base_kwargs(ds)
    finally:
        ds.dt = float(ds.t_grid[1] - ds.t_grid[0])


def test_build_fv_base_kwargs_mirrors_generation_constants(tiny_fv_dataset):
    ds, _, _ = tiny_fv_dataset
    bk = inv.build_fv_base_kwargs(ds)
    # Generation constants the dataset does not persist.
    assert bk["a"] == 0.0 and bk["b"] == 1.0 and bk["c"] == 0.0 and bk["d"] == 1.0
    assert bk["lam_target"] == 0.8
    assert bk["flux_f"] == 0.0 and bk["flux_A"] == 0.0
    assert bk["t_on"] == 0.0 and bk["phase"] == 0.0 and bk["tukey_alpha"] == 0.5
    # Resolution / horizon / grid taken from the dataset.
    assert bk["Nx"] == ds.Nx and bk["Ny"] == ds.Ny
    assert bk["dt"] == float(ds.dt)
    assert bk["t_final"] == float(ds.t_final)
    np.testing.assert_allclose(bk["y_grid"], np.asarray(ds.y_grid, dtype=np.float64))


def test_build_fv_base_kwargs_snaps_float32_t_final_to_dt_grid(tiny_fv_dataset):
    """t_grid read back from a float32 .npy carries a ~1e-7 cast error, so a raw
    ds.t_final fails build_time_grid's explicit-dt divisibility check. The base
    kwargs must snap t_final back onto the solver-dt grid so the FV operator can
    be rebuilt from a disk-loaded dataset."""
    ds, _, _ = tiny_fv_dataset
    dt = float(ds.dt)
    n_steps = int(round(float(ds.t_final) / dt))
    true_t_final = float(ds.t_final)
    try:
        # Emulate the float32 round-trip that np.save/np.load(t_grid) introduces.
        ds.t_final = float(np.float32(n_steps * dt)) + 1.2e-7
        bk = inv.build_fv_base_kwargs(ds)
        # Snapped back to an exact whole number of dt steps.
        ratio = bk["t_final"] / dt
        assert abs(ratio - round(ratio)) < 1e-9
        assert round(ratio) == n_steps
        # And it actually rebuilds the solver (the original failure mode).
        params = dict(ds.sim_params[0])
        solver = ds.problem.configure_solver(params, bk)
        assert solver is not None
    finally:
        ds.t_final = true_t_final


def test_fv_predict_masked_reproduces_stored_trajectory(tiny_fv_dataset):
    """The core Stage 4 check: rebuilding the FV operator at the stored theta_true
    reproduces the dataset's own (normalized) snapshots full-field."""
    ds, mu, sigma = tiny_fv_dataset
    bk = inv.build_fv_base_kwargs(ds)
    sid = 0
    time_indices = inv._default_time_indices(ds.Nt)
    obs = inv.build_observation_set(ds, sid, time_indices)

    pred = inv.fv_predict_masked(
        ds, bk, sid, obs.theta_true, time_indices,
        mu_global=mu, sigma_global=sigma, mask=None,
    )
    target = inv.apply_sensor_mask(obs.targets.detach().cpu(), None)
    assert pred.shape == target.shape
    # Same solver, same params, same dt -> reproduction up to the float32 cast
    # of the stored trajectory.
    np.testing.assert_allclose(pred.numpy(), target.numpy(), atol=1e-3)


def test_fv_predict_masked_respects_sensor_mask(tiny_fv_dataset):
    ds, mu, sigma = tiny_fv_dataset
    bk = inv.build_fv_base_kwargs(ds)
    sid = 0
    time_indices = inv._default_time_indices(ds.Nt)
    mask = inv.build_interface_sensor_mask(
        np.asarray(ds.x_grid), np.asarray(ds.y_grid),
        interface_x=0.5, x_halfwidth=0.1, n_y=4,
    )
    obs = inv.build_observation_set(
        ds, sid, time_indices, interface_x=0.5,
        sensor_x_halfwidth=0.1, sensor_n_y=4,
    )
    full = inv.fv_predict_masked(
        ds, bk, sid, obs.theta_true, time_indices,
        mu_global=mu, sigma_global=sigma, mask=None,
    )
    masked = inv.fv_predict_masked(
        ds, bk, sid, obs.theta_true, time_indices,
        mu_global=mu, sigma_global=sigma, mask=mask,
    )
    n_sensors = int(mask.sum())
    assert masked.shape == (len(time_indices), n_sensors, 1)
    # The masked gather is exactly the full-field values at the sensor pixels.
    full_field = full.reshape(len(time_indices), ds.Nx, ds.Ny, 1)
    expected = inv.apply_sensor_mask(full_field, mask)
    np.testing.assert_allclose(masked.numpy(), expected.numpy(), atol=0.0)


def test_fv_refine_report_columns_and_zero_residual_at_truth(tiny_fv_dataset):
    """At theta_true the FV residual against the dataset's own observations is
    ~0 (the operator is reproduced); the report exposes finite fno/C_FNO and the
    summary carries the expected columns."""
    ds, mu, sigma = tiny_fv_dataset
    bk = inv.build_fv_base_kwargs(ds)
    model = _tiny_source_itr_model().eval()
    for p in model.parameters():
        p.requires_grad_(False)

    sid = 0
    time_indices = inv._default_time_indices(ds.Nt)
    obs = inv.build_observation_set(
        ds, sid, time_indices, interface_x=0.5,
        sensor_x_halfwidth=0.1, sensor_n_y=4,
    )
    res = inv.fv_refine_report(
        model, ds, bk, obs, obs.theta_true,
        mu_global=mu, sigma_global=sigma, polish=False,
    )
    assert res.fv_resid < 1e-6                       # operator reproduces obs
    assert np.isfinite(res.fno_resid)
    assert np.isfinite(res.fno_vs_fv_resid) and res.fno_vs_fv_resid >= 0.0

    flat = inv.fv_refine_summary(res, obs, obs.y_grid)
    for k in ("fno_resid", "fv_resid", "fno_vs_fv_resid"):
        assert k in flat and np.isfinite(flat[k])
    # No polish requested -> no polished columns.
    assert "fv_polish_resid" not in flat


def test_fv_polish_holds_or_improves_from_perturbed_start(tiny_fv_dataset):
    """Nelder-Mead FV polish, started from a perturbed theta against the real FV
    observations at theta_true, does not increase the FV residual and lands an
    in-box theta. (Cheap: few iters, tiny grid.)"""
    ds, mu, sigma = tiny_fv_dataset
    bk = inv.build_fv_base_kwargs(ds)
    model = _tiny_source_itr_model().eval()
    for p in model.parameters():
        p.requires_grad_(False)

    sid = 0
    time_indices = inv._default_time_indices(ds.Nt)
    obs = inv.build_observation_set(
        ds, sid, time_indices, interface_x=0.5,
        sensor_x_halfwidth=0.15, sensor_n_y=4,
    )
    theta_true = obs.theta_true
    # Perturb within the box (keep the dependent amp ceiling satisfied).
    theta_start = torch.tensor([
        float(theta_true[0]) + 0.05,
        max(0.0, float(theta_true[1]) - 0.1),
        min(0.9, float(theta_true[2]) + 0.03),
        float(theta_true[3]),
    ], dtype=torch.float32)

    mask_cpu = obs.mask.cpu()
    target_obs = inv.apply_sensor_mask(obs.targets.detach().cpu(), mask_cpu)
    start_pred = inv.fv_predict_masked(
        ds, bk, sid, theta_start, time_indices,
        mu_global=mu, sigma_global=sigma, mask=mask_cpu,
    )
    start_resid = inv._masked_half_mse(start_pred, target_obs)

    res = inv.fv_refine_report(
        model, ds, bk, obs, theta_start,
        mu_global=mu, sigma_global=sigma, polish=True, polish_maxiter=40,
    )
    assert res.fv_polish_resid is not None
    assert res.fv_polish_resid <= start_resid + 1e-9    # never worse than start
    tp = res.theta_fv_polish
    assert 0.05 - 1e-6 <= float(tp[0]) <= 1.0 + 1e-6
    assert float(tp[1]) >= -1e-6 and (float(tp[0]) + float(tp[1])) <= R_PEAK_MAX + 1e-4

    flat = inv.fv_refine_summary(res, obs, obs.y_grid)
    for name in ("R_base", "R_amp", "y0", "sigma"):
        assert f"{name}_fvpolish" in flat and np.isfinite(flat[f"{name}_fvpolish"])
        assert f"{name}_fvpolish_abserr" in flat
    assert np.isfinite(flat["excess_int_fvpolish"])
    assert flat["fv_polish_evals"] >= 1


# ---------------------------------------------------------------------------
# Stage 5: measurement noise + uncertainty quantification.
#
# The checkpoint-free checks pin the load-bearing math: the effective-variance
# C_total = C_meas + C_FNO formula, the iid-noise reproducibility, the Jacobian
# log-det that pushes a uniform-theta prior through the reparameterization, and
# the Wilks threshold-crossing interpolation. The tiny-model checks confirm the
# profile/Laplace/MCMC summaries are wired and finite (recovery quality needs a
# trained model and is exercised by the end-to-end run, not unit tests).
# ---------------------------------------------------------------------------

def test_effective_sigma2_is_cmeas_plus_2cfno():
    # C_meas = noise_std**2; FNO error variance = 2*c_fno (c_fno is a half-MSE).
    assert inv.effective_sigma2(0.1, 0.0) == pytest.approx(0.01)
    assert inv.effective_sigma2(0.2, 0.05) == pytest.approx(0.04 + 0.10)
    assert inv.effective_sigma2(0.0, 0.0) == 0.0


def test_effective_sigma2_scales_by_design_effect():
    base = inv.effective_sigma2(0.2, sigma_fno_cal=0.1)
    assert base == pytest.approx(0.04 + 0.01)
    # A design effect > 1 inflates the variance multiplicatively.
    assert inv.effective_sigma2(
        0.2, sigma_fno_cal=0.1, design_effect=3.0
    ) == pytest.approx(base * 3.0)
    # It defaults to a no-op and never shrinks below the independent baseline.
    assert inv.effective_sigma2(0.2, sigma_fno_cal=0.1) == pytest.approx(base)
    assert inv.effective_sigma2(
        0.2, sigma_fno_cal=0.1, design_effect=0.4
    ) == pytest.approx(base)


def test_residual_design_effect_independent_is_unity():
    rng = np.random.default_rng(0)
    indep = rng.normal(size=(400, 12))
    d = inv._residual_design_effect(indep, 0.0)
    assert d["dim"] == 12
    assert d["design_effect"] == pytest.approx(1.0, abs=0.1)
    assert d["n_eff"] == pytest.approx(12.0 / d["design_effect"])


def test_residual_design_effect_correlated_deflates_to_one_over_m():
    rng = np.random.default_rng(1)
    # Rank-1 residual field: every component is the same per-sim draw, so the
    # m=12 sensors carry the information of a single independent observation.
    shared = rng.normal(size=(400, 1))
    corr = shared * np.ones((1, 12))
    d = inv._residual_design_effect(corr, 0.0)
    assert d["design_effect"] == pytest.approx(12.0, rel=1e-6)
    assert d["n_eff"] == pytest.approx(1.0, rel=1e-6)


def test_residual_design_effect_measurement_noise_pulls_to_one():
    rng = np.random.default_rng(2)
    shared = rng.normal(size=(400, 1))
    corr = shared * np.ones((1, 12))
    # Correlated FNO error of unit scale is swamped by huge iid measurement
    # noise, so the independent measurement term drives D back toward 1.
    swamped = inv._residual_design_effect(corr, sigma_meas2=1.0e6)
    assert swamped["design_effect"] == pytest.approx(1.0, abs=1e-3)
    # With only a modest measurement floor the correlation still inflates D.
    modest = inv._residual_design_effect(corr, sigma_meas2=100.0)
    assert modest["design_effect"] > 1.05


def test_residual_design_effect_clamps_and_handles_degenerate():
    rng = np.random.default_rng(3)
    shared = rng.normal(size=(200, 1))
    # Anti-correlated columns: raw design effect < 1, applied value clamped to 1.
    anti = np.hstack([shared, -shared])
    d = inv._residual_design_effect(anti, 0.0)
    assert d["design_effect_raw"] < 1.0
    assert d["design_effect"] == pytest.approx(1.0)
    # Fewer than two simulations cannot estimate a covariance: no-op.
    degen = inv._residual_design_effect(np.zeros((1, 5)), 0.0)
    assert degen["design_effect"] == pytest.approx(1.0)
    assert degen["dim"] == 5


def test_add_measurement_noise_reproducible_and_noop():
    base = _fake_observation_set()
    base.targets = base.targets + 1.0  # nonzero field so a no-op is detectable

    a = inv.ObservationSet(**{**base.__dict__, "targets": base.targets.clone()})
    b = inv.ObservationSet(**{**base.__dict__, "targets": base.targets.clone()})
    c = inv.ObservationSet(**{**base.__dict__, "targets": base.targets.clone()})

    inv.add_measurement_noise(a, 0.1, seed=7)
    inv.add_measurement_noise(b, 0.1, seed=7)
    inv.add_measurement_noise(c, 0.1, seed=8)
    # Same seed -> identical draws; different seed -> different draws.
    assert torch.allclose(a.targets, b.targets)
    assert not torch.allclose(a.targets, c.targets)
    # Noise actually perturbed the field, with std near noise_std.
    assert (a.targets - base.targets).abs().mean() > 0

    z = inv.ObservationSet(**{**base.__dict__, "targets": base.targets.clone()})
    inv.add_measurement_noise(z, 0.0, seed=7)        # non-positive -> no-op
    assert torch.allclose(z.targets, base.targets)


def test_neg_log_likelihood_zero_at_fit_and_scales_inverse_variance():
    model = _tiny_source_itr_model().eval()
    obs = _fake_observation_set()
    theta = obs.theta_true.to(torch.float32)
    # Make the data exactly the model's own prediction -> residual is ~zero.
    with torch.no_grad():
        obs.targets = inv.predict_fullfield(model, obs, theta).detach()
    nll0 = float(inv.neg_log_likelihood(model, obs, theta, 0.01))
    assert nll0 == pytest.approx(0.0, abs=1e-6)

    # Now perturb theta so residuals are nonzero, and check 1/sigma_eff2 scaling.
    theta2 = theta.clone()
    theta2[1] = theta2[1] + 0.3
    nll_a = float(inv.neg_log_likelihood(model, obs, theta2, 0.02))
    nll_b = float(inv.neg_log_likelihood(model, obs, theta2, 0.08))
    assert nll_a > 0 and nll_b > 0
    assert nll_a / nll_b == pytest.approx(0.08 / 0.02, rel=1e-5)


def test_threshold_crossings_interpolates_ushape():
    # delta(x) = (x - 2)^2 crosses thresh=1 at x = 1 and x = 3.
    grid = np.linspace(-1.0, 5.0, 25)
    delta = (grid - 2.0) ** 2
    lo, hi = inv._threshold_crossings(grid, delta, 1.0)
    assert lo == pytest.approx(1.0, abs=2e-2)
    assert hi == pytest.approx(3.0, abs=2e-2)

    # All-inside -> clamps to the grid edges.
    lo2, hi2 = inv._threshold_crossings(grid, delta, 1e6)
    assert lo2 == pytest.approx(grid[0]) and hi2 == pytest.approx(grid[-1])

    # None-inside -> degenerate interval at the argmin.
    lo3, hi3 = inv._threshold_crossings(grid, delta, -1.0)
    assert lo3 == hi3 == pytest.approx(grid[int(np.argmin(delta))])


@pytest.mark.parametrize("fixed_index", [0, 1, 2, 3])
def test_theta_profile_pins_coord_and_respects_amp_ceiling(fixed_index):
    torch.manual_seed(0)
    u = torch.randn(4, dtype=torch.float64)
    pinned = {0: 0.4, 1: 0.7, 2: 0.6, 3: 0.12}[fixed_index]
    theta = inv._theta_profile(u, fixed_index, pinned)
    assert float(theta[fixed_index]) == pytest.approx(pinned)
    # Free coords stay in the physical box; the dependent amp ceiling holds.
    R_base, R_amp = float(theta[0]), float(theta[1])
    assert 0.05 - 1e-6 <= R_base <= 1.0 + 1e-6
    assert R_amp >= -1e-6 and (R_base + R_amp) <= R_PEAK_MAX + 1e-4


def test_theta_profile_high_pinned_amp_caps_free_base():
    # Pinning R_amp near its ceiling must shrink the free R_base box so the peak
    # R_base + R_amp never exceeds R_PEAK_MAX. A large positive u0 (sigmoid -> 1)
    # would map R_base to base_hi=1.0 without the cap, giving 1.0 + 2.8 = 3.8.
    pinned_amp = 2.8
    u = torch.tensor([20.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    theta = inv._theta_profile(u, fixed_index=1, fixed_value=pinned_amp)
    R_base, R_amp = float(theta[0]), float(theta[1])
    assert R_amp == pytest.approx(pinned_amp)
    base_ceiling = R_PEAK_MAX - pinned_amp  # 0.2
    assert R_base == pytest.approx(base_ceiling, abs=1e-6)
    assert (R_base + R_amp) <= R_PEAK_MAX + 1e-9
    # Lower edge of the shrunk box still reaches base_lo at u0 -> -inf.
    u_lo = torch.tensor([-20.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    theta_lo = inv._theta_profile(u_lo, fixed_index=1, fixed_value=pinned_amp)
    assert float(theta_lo[0]) == pytest.approx(RC_VOID_RANGES["R_base"][0], abs=1e-6)


def test_profile_likelihood_summary_columns_present():
    model = _tiny_source_itr_model().eval()
    obs = _fake_observation_set()
    obs.targets = obs.targets + 0.5  # nonzero residuals against the untrained net
    theta_hat = obs.theta_true.to(torch.float32)
    res = inv.profile_likelihood(
        model, obs, theta_hat, sigma_eff2=0.05,
        param_index=1, n_grid=5, span=0.3, level=0.95,
        adam_steps=8, lbfgs_steps=3,
    )
    assert res.param_name == "R_amp"
    assert res.excess_ci_low <= res.excess_ci_high
    assert np.isfinite(res.excess_ci_high - res.excess_ci_low)
    assert res.ci_low <= res.ci_high
    # Profiled NLL never beats the unconstrained min by construction.
    assert res.nll.min() >= res.nll_min - 1e-6


@pytest.mark.parametrize("adapter_cls", [SourceItrAdapter, SourceItrSinAdapter])
def test_theta_profile_severity_realizes_pinned_severity(adapter_cls):
    adapter = adapter_cls()
    y_grid = torch.linspace(0.0, 1.0, 64, dtype=torch.float64)
    assert adapter.max_feasible_severity(y_grid) > 0.0
    torch.manual_seed(0)
    u = torch.randn(adapter.theta_dim, dtype=torch.float64)
    # For a *fixed* shape u the realizable severity is capped by that shape's
    # amplitude ceiling; read it back by pinning far above the cap (the solve
    # clamps to the shape's max). Feasible pins are then realized exactly.
    s_cap = float(adapter.uq_quantity(
        adapter.theta_profile_severity(u, 1e9, y_grid), y_grid
    ))
    assert s_cap > 0.0
    for frac in (0.1, 0.5, 0.9):
        s_pin = frac * s_cap
        theta = adapter.theta_profile_severity(u, s_pin, y_grid)
        got = float(adapter.uq_quantity(theta, y_grid))
        assert got == pytest.approx(s_pin, rel=1e-4, abs=1e-6)
        # Peak resistance stays physical: R_base + amp <= R_PEAK_MAX.
        assert float(theta[0]) + float(theta[1]) <= R_PEAK_MAX + 1e-4


def test_theta_profile_severity_caps_at_amp_ceiling():
    # Pinning above the feasible ceiling clamps the amplitude (severity is then
    # the max attainable rather than the requested value) and keeps R_base+amp
    # inside the physical box.
    adapter = SourceItrAdapter()
    y_grid = torch.linspace(0.0, 1.0, 64, dtype=torch.float64)
    s_max = adapter.max_feasible_severity(y_grid)
    u = torch.tensor([20.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    theta = adapter.theta_profile_severity(u, 10.0 * s_max, y_grid)
    R_base, R_amp = float(theta[0]), float(theta[1])
    assert R_amp <= (R_PEAK_MAX - RC_VOID_RANGES["R_base"][0]) + 1e-6
    assert R_base + R_amp <= R_PEAK_MAX + 1e-6


def test_theta_profile_severity_is_differentiable():
    adapter = SourceItrAdapter()
    y_grid = torch.linspace(0.0, 1.0, 32, dtype=torch.float64)
    s_max = adapter.max_feasible_severity(y_grid)
    u = torch.zeros(4, dtype=torch.float64, requires_grad=True)
    theta = adapter.theta_profile_severity(u, 0.4 * s_max, y_grid)
    theta.sum().backward()
    assert u.grad is not None and torch.isfinite(u.grad).all()


def test_severity_profile_likelihood_is_valid_profile_interval():
    model = _tiny_source_itr_model().eval()
    obs = _fake_observation_set()
    obs.targets = obs.targets + 0.5  # nonzero residuals against the untrained net
    theta_hat = obs.theta_true.to(torch.float32)
    res = inv.severity_profile_likelihood(
        model, obs, theta_hat, sigma_eff2=0.05,
        n_grid=5, span=0.5, level=0.95, adam_steps=8, lbfgs_steps=3,
        adapter=inv._SOURCE_ITR_ADAPTER,
    )
    assert res.param_name == "S_R"
    # Direct severity profile: the reported CI *is* the severity CI.
    assert res.excess_ci_low == pytest.approx(res.ci_low)
    assert res.excess_ci_high == pytest.approx(res.ci_high)
    assert res.ci_low <= res.ci_high
    # Grid stays inside the feasible severity range.
    s_max = inv._SOURCE_ITR_ADAPTER.max_feasible_severity(obs.y_grid)
    assert res.grid.min() > 0.0 and res.grid.max() <= s_max + 1e-6
    # Profiled NLL never beats the unconstrained min by construction.
    assert res.nll.min() >= res.nll_min - 1e-6
    summ = inv.profile_interval_summary(res, inv._SOURCE_ITR_ADAPTER)
    assert summ["profile_param"] == "S_R"
    assert summ["profile_excess_ci_low"] <= summ["profile_excess_ci_high"]
    assert summ["profile_excess_ci_width"] == pytest.approx(
        summ["profile_excess_ci_high"] - summ["profile_excess_ci_low"]
    )
    assert isinstance(summ["profile_bound_limited"], bool)


def test_profile_refit_keeps_best_of_several_starts():
    # The nuisance objective is the same nonconvex surface the MAP fit needs
    # multistart for, so a refit must return the *best* start, not the last.
    model = _tiny_source_itr_model().eval()
    obs = _fake_observation_set()
    obs.targets = obs.targets + 0.5
    seeds = [
        torch.zeros(4, dtype=torch.float32),
        torch.full((4,), -3.0, dtype=torch.float32),
        torch.full((4,), 3.0, dtype=torch.float32),
    ]
    singles = [
        inv._profile_refit(
            model, obs, 0.05, 1, 0.4, [s], adam_steps=8, lbfgs_steps=3
        )[0]
        for s in seeds
    ]
    multi_nll, multi_theta, multi_u = inv._profile_refit(
        model, obs, 0.05, 1, 0.4, seeds, adam_steps=8, lbfgs_steps=3
    )
    assert multi_nll <= min(singles) + 1e-9
    assert multi_theta.shape == (4,) and multi_u.shape == (4,)
    assert float(multi_theta[1]) == pytest.approx(0.4, abs=1e-6)


def test_profile_sweep_reports_nonnegative_gap_and_is_deterministic():
    model = _tiny_source_itr_model().eval()
    obs = _fake_observation_set()
    obs.targets = obs.targets + 0.5
    theta_hat = obs.theta_true.to(torch.float32)
    kwargs = dict(
        sigma_eff2=0.05, n_grid=4, span=0.4, level=0.95,
        adam_steps=6, lbfgs_steps=2, n_starts=3, seed=1,
        adapter=inv._SOURCE_ITR_ADAPTER,
    )
    a = inv.severity_profile_likelihood(model, obs, theta_hat, **kwargs)
    b = inv.severity_profile_likelihood(model, obs, theta_hat, **kwargs)
    # sweep_gap is how much the reverse sweep clawed back: never negative.
    assert a.sweep_gap >= 0.0 and np.isfinite(a.sweep_gap)
    assert a.n_starts == 3
    np.testing.assert_allclose(a.nll, b.nll)
    np.testing.assert_allclose(a.grid, b.grid)
    assert a.ci_low == pytest.approx(b.ci_low)
    summ = inv.profile_interval_summary(a, inv._SOURCE_ITR_ADAPTER)
    assert summ["profile_n_starts"] == 3
    assert summ["profile_sweep_gap"] >= 0.0


def test_profile_multistart_never_worse_than_single_start():
    # Adding starts can only enlarge the candidate set at each pin, so the
    # profiled NLL curve must not rise anywhere. A rising profile is exactly the
    # failure mode that shrinks a Wilks interval below its nominal level.
    model = _tiny_source_itr_model().eval()
    obs = _fake_observation_set()
    obs.targets = obs.targets + 0.5
    theta_hat = obs.theta_true.to(torch.float32)
    common = dict(
        sigma_eff2=0.05, param_index=1, n_grid=4, span=0.3, level=0.95,
        adam_steps=6, lbfgs_steps=2, seed=0,
    )
    one = inv.profile_likelihood(model, obs, theta_hat, n_starts=1, **common)
    many = inv.profile_likelihood(model, obs, theta_hat, n_starts=4, **common)
    np.testing.assert_allclose(one.grid, many.grid)
    assert np.all(many.nll <= one.nll + 1e-6)


def test_profile_refit_scalar_adapter_ignores_extra_starts():
    # With a 1-D theta the pin leaves nothing free, so every seed gives the same
    # theta: multistart must not be paid for there.
    adapter = ForcingAdapter()
    model = _tiny_forcing_model()
    obs = _fake_forcing_observation_set(Nx=6, Ny=6, N=1)
    seeds = [torch.zeros(1), torch.full((1,), 2.0), torch.full((1,), -2.0)]
    nll, theta, _ = inv._profile_refit(
        model, obs, 0.05, 0, 0.6, seeds, adam_steps=4, lbfgs_steps=1,
        adapter=adapter,
    )
    assert float(theta[0]) == pytest.approx(0.6, abs=1e-6)
    assert np.isfinite(nll)


def test_laplace_spectrum_eigs_are_singular_values_squared_over_variance():
    model = _tiny_source_itr_model().eval()
    obs = _fake_observation_set()
    obs.targets = obs.targets + 0.5
    theta_hat = obs.theta_true.to(torch.float32)
    sigma_eff2 = 0.05
    spec = inv.laplace_spectrum(model, obs, theta_hat, sigma_eff2)

    J = inv.observation_jacobian(model, obs, theta_hat).cpu().numpy()
    S = np.linalg.svd(J, compute_uv=False)
    np.testing.assert_allclose(
        spec["eigenvalues"], (S ** 2) / sigma_eff2, rtol=1e-4, atol=1e-8
    )
    flat = inv.laplace_summary(spec)
    assert np.isfinite(flat["laplace_cond"]) or flat["laplace_cond"] == float("inf")
    for i in range(4):
        assert f"laplace_eig_{i}" in flat
    for nm in ("R_base", "R_amp", "y0", "sigma"):
        assert f"laplace_least_dir_{nm}" in flat
    # Eigenvalues are sorted descending (cond uses [0]/[-1]).
    eig = spec["eigenvalues"]
    assert np.all(np.diff(eig) <= 1e-8)


# ---------------------------------------------------------------------------
# ForcingAdapter: scalar R_c inverse path.
# ---------------------------------------------------------------------------

def _tiny_forcing_model():
    from src.operators.fno2d import FNO2d
    from problems.forcing import COND_STATIC_DIM, FORCING_TEMPORAL_TOKEN_DIM
    from problems.forcing import SPATIAL_CHANNELS_TEMPORAL

    class CondToBeta(torch.nn.Module):
        def forward(self, cond_full):
            out = torch.zeros(
                cond_full.shape[0], 1, 2, 1,
                dtype=cond_full.dtype, device=cond_full.device,
            )
            out[:, 0, 0, 0] = 1.0
            out[:, 0, 1, 0] = 5.0 * cond_full[:, 1]
            return out

    model = FNO2d(
        modes1=1, modes2=1, width=1,
        in_channels=SPATIAL_CHANNELS_TEMPORAL, out_channels=1, n_layers=1,
        cond_static_dim=COND_STATIC_DIM,
        temporal_token_dim=FORCING_TEMPORAL_TOKEN_DIM,
        temporal_hidden=8, forcing_embed_dim=4, forcing_spatial_dim=1,
        use_temporal_encoder=True, use_forcing_time_aug=True,
        s_y_channel=3,
    )
    model.padding = 0
    model.cond_mlp = CondToBeta()
    with torch.no_grad():
        model.linear_p.weight.zero_()
        model.linear_p.bias.zero_()
        for layer in model.spectral_layers:
            layer.weights1.zero_()
            layer.weights2.zero_()
        for layer in model.conv_layers:
            layer.weight.zero_()
            layer.bias.zero_()
        model.linear_q.weight.zero_()
        model.linear_q.bias.zero_()
        model.linear_q.weight[0, 0] = 1.0
        model.output_layer.weight.zero_()
        model.output_layer.bias.zero_()
        model.output_layer.weight[0, 0] = 1.0
    for p in model.parameters():
        p.requires_grad_(False)
    return model.eval()


def _fake_forcing_observation_set(Nx=8, Ny=8, N=2, M=128):
    spatial = torch.zeros(N, Nx, Ny, 4)
    spatial[..., 1] = torch.linspace(0, 1, Nx)[None, :, None]
    spatial[..., 2] = torch.linspace(0, 1, Ny)[None, None, :]
    spatial[..., 3] = 1.0
    cond = torch.zeros(N, 10)
    cond[:, 0] = torch.linspace(0.25, 0.75, N)
    forcing_seq = torch.zeros(N, M, FORCING_TEMPORAL_TOKEN_DIM)
    forcing_seq[..., 0] = torch.linspace(0, 1, M)
    targets = torch.zeros(N, Nx, Ny, 1)
    y_grid = torch.linspace(0, 1, Ny)
    return inv.ObservationSet(
        sid=0, time_indices=list(range(N)),
        spatial=spatial, cond=cond, forcing_seq=forcing_seq,
        targets=targets, y_grid=y_grid, Nx=Nx,
        theta_true=torch.tensor([0.65]),
    )


def _tiny_forcing_dataset(Nx=10, Ny=10):
    from data.dataset import SnapshotPairDataset
    from problems.registry import get_problem
    from src.physics.fv_solver_1d import build_time_grid

    spec = get_problem("forcing", "temporal_encoder")
    a, b, c, d = 0.0, 1.0, 0.0, 1.0
    x_grid = np.linspace(a, b, Nx)
    y_grid = np.linspace(c, d, Ny)
    X, Y = np.meshgrid(x_grid, y_grid, indexing="ij")
    dt, t_grid = build_time_grid(0.03, 0.01, explicit_dt=True)
    grids = {"X": X, "Y": Y, "x_grid": x_grid, "y_grid": y_grid}
    time_cfg = dict(
        num_sims=1, dt=dt, t_final=float(t_grid[-1]), lhs_seed=0,
        t_on=0.0, t_off=0.2, phase=0.0, tukey_alpha=0.5,
        T_right=300.0, b=b, ic_families=None,
    )
    sim_params = spec.sample_sim_params(
        rng=np.random.default_rng(0),
        rng_profile=np.random.default_rng(1),
        grids=grids,
        time_cfg=time_cfg,
    )
    trajectories = np.broadcast_to(
        np.asarray(sim_params[0]["T0"], dtype=np.float32)[None, None, :, :],
        (1, len(t_grid), Nx, Ny),
    ).copy()
    ds = SnapshotPairDataset(
        trajectories=trajectories,
        sim_params=np.array(sim_params, dtype=object),
        t_grid=np.asarray(t_grid, dtype=np.float32),
        x_grid=x_grid.astype(np.float32),
        y_grid=y_grid.astype(np.float32),
        sim_ids=np.array([0]),
        mu_global=300.0,
        sigma_global=1.0,
        problem=spec,
        dt=float(dt),
        ramp_seconds=2.0 * float(dt),
    )
    ds._split_ids = {"train": np.array([0]), "val": np.array([], dtype=int),
                     "test": np.array([0])}
    return ds


def test_forcing_adapter_reparameterization_roundtrip_and_shape():
    adapter = ForcingAdapter()
    u = torch.tensor([[-2.0], [0.0], [1.5]], dtype=torch.float64)
    theta = adapter.theta_from_unconstrained(u)
    back = adapter.unconstrained_from_theta(theta)
    np.testing.assert_allclose(back.numpy(), u.numpy(), rtol=0, atol=1e-8)

    theta1 = adapter.theta_from_unconstrained(torch.tensor([0.0], dtype=torch.float32))
    assert theta1.shape == (1,)
    assert theta1.dtype == torch.float32


def test_forcing_adapter_unconstrained_from_theta_clamps_near_bounds():
    from problems.forcing import RC_RANGE

    adapter = ForcingAdapter()
    bounds = torch.tensor([[RC_RANGE[0]], [RC_RANGE[1]]], dtype=torch.float32)
    u = adapter.unconstrained_from_theta(bounds)
    assert u.shape == (2, 1)
    assert torch.all(torch.isfinite(u))
    theta = adapter.theta_from_unconstrained(u)
    assert torch.all(theta[:, 0] >= RC_RANGE[0])
    assert torch.all(theta[:, 0] <= RC_RANGE[1])


@pytest.mark.parametrize("R_c", [0.05, 0.42, 1.0])
def test_forcing_adapter_cond_injection_matches_build_cond_vector(R_c):
    from problems.forcing import build_cond_vector

    adapter = ForcingAdapter()
    cond = build_cond_vector(
        t_bar_norm=0.3,
        R_c=R_c,
        spatial_family="uniform",
        spatial_params={},
    )
    got = adapter.cond_slice_from_theta(
        torch.tensor([R_c], dtype=torch.float64)
    ).numpy()
    np.testing.assert_allclose(got, cond[1:2], rtol=0, atol=1e-6)


def test_forcing_adapter_no_spatial_channel_and_sim_param_copy():
    adapter = ForcingAdapter()
    params = {
        "R_c": 0.3,
        "temporal_family": "sin",
        "temporal_params": {"A": 100.0, "f": 2.0},
        "spatial_family": "uniform",
        "spatial_params": {},
    }
    updated = adapter.inject_theta_into_sim_params(params, torch.tensor([0.7]))
    assert updated is not params
    assert updated["R_c"] == pytest.approx(0.7)
    assert params["R_c"] == pytest.approx(0.3)
    assert adapter.spatial_channel_index() is None
    assert adapter.spatial_channel_from_theta(torch.tensor([0.7]), torch.linspace(0, 1, 4), 4) is None


def test_forcing_adapter_build_fv_solver_roundtrip():
    adapter = ForcingAdapter()
    ds = _tiny_forcing_dataset()
    params = dict(ds.sim_params[0])
    solver = adapter.build_fv_solver(ds, params, torch.tensor([0.7]))
    t, _x, _y, T_hist = solver.solve(T0=params["T0"], store_trajectory=True)
    assert len(t) == len(ds.t_grid)
    assert np.asarray(T_hist).shape == (len(ds.t_grid), ds.Nx, ds.Ny)
    assert params["R_c"] != pytest.approx(0.7)


def test_forcing_adapter_scalar_recovery_through_cin_conditioning():
    adapter = ForcingAdapter()
    model = _tiny_forcing_model()
    obs = _fake_forcing_observation_set()
    theta_star = torch.tensor([0.72])
    with torch.no_grad():
        obs.targets = inv.predict_fullfield(model, obs, theta_star, adapter).detach()

    cfg = inv.InversionConfig(
        n_starts=3, adam_steps=120, adam_lr=0.08, lbfgs_steps=25,
        seed=4, start_sampling="lhs",
    )
    result = inv.invert_sim(model, obs, cfg, adapter)
    assert result.loss < 1e-7
    assert abs(float(result.theta_hat[0]) - float(theta_star[0])) < 2e-2


def test_forcing_adapter_one_dimensional_uq_smoke():
    adapter = ForcingAdapter()
    model = _tiny_forcing_model()
    obs = _fake_forcing_observation_set(Nx=6, Ny=6, N=1)
    theta_hat = torch.tensor([0.6])
    with torch.no_grad():
        obs.targets = inv.predict_fullfield(model, obs, theta_hat, adapter).detach()

    J = inv.observation_jacobian(model, obs, theta_hat, adapter)
    assert J.shape == (obs.Nx * obs.spatial.shape[2], 1)

    prof = inv.profile_likelihood(
        model, obs, theta_hat, sigma_eff2=0.05,
        param_index=0, n_grid=5, span=0.2, level=0.9,
        adam_steps=2, lbfgs_steps=1, adapter=adapter,
    )
    assert prof.param_name == "R_c"
    assert prof.ci_low <= prof.ci_high

    spec = inv.laplace_spectrum(model, obs, theta_hat, 0.05, adapter)
    flat_lap = inv.laplace_summary(spec, adapter)
    assert "laplace_least_dir_R_c" in flat_lap

    cfg = inv.InversionConfig(n_starts=1, adam_steps=5, lbfgs_steps=1, seed=0)
    result = inv.invert_sim(model, obs, cfg, adapter)
    assert result.theta_hat.shape == (1,)


def test_inverse_adapter_dispatch_and_validation(tiny_fv_dataset):
    from types import SimpleNamespace
    from problems.registry import get_problem

    assert isinstance(
        InverseAdapter.from_config({"benchmark": {"name": "source_itr"}}),
        SourceItrAdapter,
    )
    assert isinstance(
        InverseAdapter.from_config({"benchmark": {"name": "forcing"}}),
        ForcingAdapter,
    )
    with pytest.raises(ValueError, match="Unsupported inverse benchmark"):
        InverseAdapter.from_config({"benchmark": {"name": "source"}})

    forcing_adapter = ForcingAdapter()
    forcing_ds = _tiny_forcing_dataset()
    forcing_adapter.validate_dataset(forcing_ds)
    source_ds, _, _ = tiny_fv_dataset
    with pytest.raises(ValueError, match="expected dataset benchmark 'forcing'"):
        forcing_adapter.validate_dataset(source_ds)

    model = _tiny_forcing_model()
    dims = get_problem("forcing", "temporal_encoder").dims
    loaded = SimpleNamespace(
        model=model,
        dims=dims,
        config={"benchmark": {"name": "forcing", "representation": "temporal_encoder"}},
    )
    forcing_adapter.validate_model(loaded)
    bad_loaded = SimpleNamespace(
        model=model,
        dims=SimpleNamespace(**{**dims.__dict__, "cond_static_dim": 99}),
        config={"benchmark": {"name": "forcing", "representation": "temporal_encoder"}},
    )
    with pytest.raises(ValueError, match="cond_static_dim"):
        forcing_adapter.validate_model(bad_loaded)


# ---------------------------------------------------------------------------
# Per-sim NPZ artifact dump (--artifact-dir): keys/shapes for both benchmarks.
# ---------------------------------------------------------------------------

def test_artifact_dir_writes_expected_npz_keys_source_itr(tmp_path):
    from types import SimpleNamespace

    adapter = SourceItrAdapter()
    model = _tiny_source_itr_model().eval()
    obs = _fake_observation_set()
    obs.targets = obs.targets + 0.5
    theta_hat = obs.theta_true.to(torch.float32)

    report = inv.sensitivity_report(model, obs, theta_hat, adapter)
    J = inv.observation_jacobian(model, obs, theta_hat, adapter)
    result = SimpleNamespace(theta_hat=theta_hat)

    path = inv._write_sim_artifact(
        str(tmp_path), 7, adapter,
        obs=obs, result=result, report=report, J=J,
        fv_res=None, prof=None, sigma_eff2=0.1, c_fno=0.01,
        noise_std=0.01, ci_level=0.95,
        dataset_path="data/sourceitr_smoke", dataset_fingerprint="abc123",
        split_seed=0, split_name="test",
    )
    from pathlib import Path
    assert Path(path).name == "sim_00007.npz"

    d = np.load(path, allow_pickle=True)
    # Schema metadata (always present).
    for key in (
        "artifact_schema_version", "benchmark", "sim_id", "param_names",
        "theta_hat", "theta_true", "theta_bounds", "param_scales",
        "noise_std", "ci_level", "profile_threshold",
        "dataset_path", "dataset_fingerprint", "split_seed", "split_name",
    ):
        assert key in d, f"missing schema key {key}"
    assert str(d["benchmark"]) == "source_itr"
    assert int(d["sim_id"]) == 7
    assert d["theta_hat"].shape == (4,)
    assert d["theta_true"].shape == (4,)
    assert d["theta_bounds"].shape == (4, 2)
    assert d["param_scales"].shape == (4,)
    assert str(d["dataset_path"]) == "data/sourceitr_smoke"
    assert str(d["split_name"]) == "test"
    # Sensitivity block: source_itr persists the (4,4) right vectors + alignment.
    assert d["right_vectors"].shape == (4, 4)
    assert d["singular_values"].shape == (4,)
    assert "ramp_sigma_alignment" in d
    assert d["observation_jacobian"].shape[1] == 4
    # Inflation scalars mirror the UQ columns.
    assert float(d["sigma_eff2"]) == pytest.approx(0.1)
    assert float(d["c_fno"]) == pytest.approx(0.01)


def test_artifact_dir_writes_expected_npz_keys_forcing(tmp_path):
    from types import SimpleNamespace

    adapter = ForcingAdapter()
    model = _tiny_forcing_model()
    obs = _fake_forcing_observation_set()
    theta_hat = obs.theta_true.to(torch.float32)
    with torch.no_grad():
        obs.targets = inv.predict_fullfield(model, obs, theta_hat, adapter).detach()

    report = inv.sensitivity_report(model, obs, theta_hat, adapter)
    J = inv.observation_jacobian(model, obs, theta_hat, adapter)
    result = SimpleNamespace(theta_hat=theta_hat)

    path = inv._write_sim_artifact(
        str(tmp_path), 3, adapter,
        obs=obs, result=result, report=report, J=J,
        fv_res=None, prof=None, sigma_eff2=None, c_fno=None,
        noise_std=0.0, ci_level=0.95,
        dataset_path="data/forcing_inv", dataset_fingerprint="def456",
        split_seed=0, split_name="test",
    )
    from pathlib import Path
    assert Path(path).name == "sim_00003.npz"

    d = np.load(path, allow_pickle=True)
    assert str(d["benchmark"]) == "forcing"
    # Scalar R_c: width-1 theta and a degenerate (1,1) SVD.
    assert d["theta_hat"].shape == (1,)
    assert d["theta_bounds"].shape == (1, 2)
    assert d["param_scales"].shape == (1,)
    assert d["right_vectors"].shape == (1, 1)
    assert d["observation_jacobian"].shape[1] == 1
    # One parameter: no (R_amp, sigma) ridge alignment.
    assert "ramp_sigma_alignment" not in d
    assert "sigma_eff2" not in d
    assert "c_fno" not in d


# ---------------------------------------------------------------------------
# Sinusoid interface-resistance adapters (source_itr_sin / forcing_itr_sin):
# the 2-param R_c(y) = R_base + A*sin(pi*y) reparameterization. The load-bearing
# checks mirror the void's: the torch reparameterization must reproduce the two
# theta injection points bit-for-bit against the forward pipeline, and the
# lower-triangular Jacobian's log-det must match autograd.
# ---------------------------------------------------------------------------

# (R_base, A) pairs spanning the box; each satisfies the dependent ceiling
# A <= R_PEAK_MAX - R_base, including the near-ceiling edge and the no-hump floor.
THETAS_SIN = [
    (0.20, 0.50),
    (0.05, 2.90),   # near the amplitude ceiling at the R_base floor
    (0.90, 0.10),
    (0.50, 0.00),   # no-hump floor (A = 0)
    (0.75, 1.20),
]


@pytest.mark.parametrize("theta", THETAS_SIN)
def test_sin_adapter_cond_slice_matches_build_cond_vector_sin(theta):
    from problems.source_itr_sin import build_cond_vector_sin

    R_base, A = theta
    x_lo, x_hi, y_lo, y_hi = 0.0, 1.0, 0.0, 1.0
    w_h = h_h = 0.1
    xcr, ycr = _patch_center_ranges(x_lo, x_hi, y_lo, y_hi, w_h, h_h)
    cond = build_cond_vector_sin(
        t_bar_norm=0.3,
        R_base=R_base, A=A,
        x_h=0.5, y_h=0.5, w_h=w_h, h_h=h_h,
        x_center_range=xcr, y_center_range=ycr,
        x_length_scale=(x_hi - x_lo), y_length_scale=(y_hi - y_lo),
    )
    ref_slice = cond[1:3]  # [R_base_norm, A_norm] (global A norm over RC_SIN_RANGES)

    adapter = SourceItrSinAdapter()
    assert adapter.cond_slice_indices() == (1, 3)
    got = adapter.cond_slice_from_theta(
        torch.tensor(theta, dtype=torch.float64)
    ).numpy()
    np.testing.assert_allclose(got, ref_slice, rtol=0, atol=1e-6)


@pytest.mark.parametrize("theta", THETAS_SIN)
def test_sin_adapter_rc_channel_matches_numpy_profile(theta):
    # Mandatory NumPy<->Torch equivalence: the torch spatial channel must equal
    # the forward log-normalized make_rc_sin_profile broadcast across x. This is
    # the worst inverse bug class (data generated under one profile
    # parameterization, fit under another).
    R_base, A = theta
    Ny, Nx = 100, 100
    y_grid = np.linspace(0.0, 1.0, Ny).astype(np.float64)

    Rc_y = make_rc_sin_profile(y_grid, R_base=R_base, A=A)
    ref_channel = np.broadcast_to(rc_log_norm(Rc_y)[None, :], (Nx, Ny))

    adapter = SourceItrSinAdapter()
    y_t = torch.from_numpy(y_grid)
    got = adapter.spatial_channel_from_theta(
        torch.tensor(theta, dtype=torch.float64), y_t, Nx
    ).numpy()

    assert got.shape == (Nx, Ny)
    np.testing.assert_allclose(got, ref_channel, rtol=0, atol=1e-5)


@pytest.mark.parametrize("theta", THETAS_SIN)
def test_sin_adapter_unconstrained_roundtrip(theta):
    adapter = SourceItrSinAdapter()
    theta_t = torch.tensor(theta, dtype=torch.float64)
    u = adapter.unconstrained_from_theta(theta_t)
    back = adapter.theta_from_unconstrained(u)
    np.testing.assert_allclose(back.numpy(), theta_t.numpy(), rtol=0, atol=1e-5)


def test_sin_adapter_theta_from_unconstrained_respects_box_and_ceiling():
    adapter = SourceItrSinAdapter()
    rng = np.random.default_rng(0)
    u = torch.from_numpy(rng.normal(0, 3.0, size=(500, 2)))
    theta = adapter.theta_from_unconstrained(u)
    R_base, A = theta[:, 0], theta[:, 1]

    base_lo, base_hi = RC_SIN_RANGES["R_base"]
    assert torch.all(R_base >= base_lo - 1e-9) and torch.all(R_base <= base_hi + 1e-9)
    # Dependent ceiling: A >= 0 and the peak R_base + A <= R_PEAK_MAX.
    assert torch.all(A >= -1e-9)
    assert torch.all(R_base + A <= R_PEAK_MAX + 1e-6)


def test_sin_adapter_reparameterization_is_lower_triangular():
    adapter = SourceItrSinAdapter()
    u = torch.tensor([0.4, -0.7], dtype=torch.float64)
    J = torch.autograd.functional.jacobian(
        lambda v: adapter.theta_from_unconstrained(v), u
    )
    # Lower-triangular map: dR_base/du1 == 0 (the (0,1) entry).
    assert float(J[0, 1]) == pytest.approx(0.0, abs=1e-12)


def test_sin_adapter_theta_from_sim_params_and_injection():
    adapter = SourceItrSinAdapter()
    params = {"R_c_base": 0.4, "R_c_A": 0.9, "R_c": 0.4}
    theta = adapter.theta_from_sim_params(params)
    assert theta.shape == (2,)
    np.testing.assert_allclose(theta.numpy(), [0.4, 0.9], rtol=0, atol=1e-6)

    updated = adapter.inject_theta_into_sim_params(params, torch.tensor([0.7, 1.1]))
    assert updated is not params
    assert updated["R_c_base"] == pytest.approx(0.7)
    assert updated["R_c_A"] == pytest.approx(1.1)
    assert updated["R_c"] == pytest.approx(0.7)  # universal column mirrors R_base
    assert params["R_c_base"] == pytest.approx(0.4)  # source dict untouched


def test_sin_adapter_uq_quantity_matches_analytic_severity():
    adapter = SourceItrSinAdapter()
    A = 1.3
    theta = torch.tensor([0.4, A], dtype=torch.float64)
    y_grid = torch.linspace(0.0, 1.0, 400, dtype=torch.float64)
    got = adapter.uq_quantity(theta, y_grid)
    # Analytic integrated excess int_0^1 (R_c - R_base) dy = A * (2/pi).
    assert got == pytest.approx(A * 2.0 / np.pi, rel=1e-3)


def test_sin_adapter_dispatch_from_config():
    assert isinstance(
        InverseAdapter.from_config({"benchmark": {"name": "source_itr_sin"}}),
        SourceItrSinAdapter,
    )
    forcing_sin = InverseAdapter.from_config(
        {"benchmark": {"name": "forcing_itr_sin"}}
    )
    assert isinstance(forcing_sin, ForcingItrSinAdapter)
    assert isinstance(forcing_sin, SourceItrSinAdapter)  # subclass relationship


def test_forcing_sin_adapter_equivalent_scalar_on_sin_profile():
    from src.physics.internal_source import equivalent_scalar_resistance

    adapter = ForcingItrSinAdapter()
    assert adapter.supports_equivalent_scalar is True

    R_base, A = 0.5, 1.4
    theta = torch.tensor([R_base, A], dtype=torch.float64)
    y_grid = torch.linspace(0.0, 1.0, 256, dtype=torch.float64)
    r_base_out, r_eq = adapter.equivalent_scalar_values(theta, y_grid)

    y = y_grid.numpy()
    profile = make_rc_sin_profile(y, R_base, A)
    ref = equivalent_scalar_resistance(y, profile, bounds=(0.0, 1.0))
    assert r_base_out == pytest.approx(R_base)
    assert r_eq == pytest.approx(ref, rel=1e-9)
    # The single-hump profile raises the effective resistance above R_base.
    assert r_eq > R_base
