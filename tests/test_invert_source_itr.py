"""Tests for the source_itr inverse solver (Stage 1 MVP).

The load-bearing checks are checkpoint-free: the torch reparameterization must
reproduce the two theta injection points *bit-for-bit* (within float tolerance)
against the real forward pipeline:

  1. cond_static[2:6]  <- build_cond_vector_itr  (linear over RC_VOID_RANGES)
  2. spatial[..., 4]   <- _rc_channel            (rc_log_norm o make_rc_void_profile)

A mismatch in either silently queries the FNO off-distribution and degrades
inversion, so these are the first thing to nail. The operator-wiring and
gradient-flow tests use a tiny untrained FNO (recovery quality needs a trained
model and is exercised by the end-to-end Stage 1 run, not unit tests).
"""

import numpy as np
import pytest
import torch

from problems.source import _patch_center_ranges
from problems.source_itr import (
    RC_Y_CHANNEL,
    build_cond_vector_itr,
    rc_log_norm,
)
from src.physics.internal_source import (
    RC_VOID_RANGES,
    R_PEAK_MAX,
    make_rc_void_profile,
)
from scripts import invert_source_itr as inv


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
    # compare the void slice cond[2:6], so any valid patch is fine.
    x_lo, x_hi, y_lo, y_hi = 0.0, 1.0, 0.0, 1.0
    w_h = h_h = 0.1
    xcr, ycr = _patch_center_ranges(x_lo, x_hi, y_lo, y_hi, w_h, h_h)
    cond = build_cond_vector_itr(
        t_bar_norm=0.3, t_s_norm=0.0,
        R_base=R_base, R_amp=R_amp, y0=y0, sigma=sigma,
        x_h=0.5, y_h=0.5, w_h=w_h, h_h=h_h,
        x_center_range=xcr, y_center_range=ycr,
        x_length_scale=(x_hi - x_lo), y_length_scale=(y_hi - y_lo),
    )
    ref_slice = cond[2:6]

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

    torch.manual_seed(0)
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
    cond = torch.zeros(N, 10)
    cond[:, 0] = torch.linspace(0.3, 1.0, N)  # t_bar_norm early/mid/late
    forcing_seq = torch.zeros(N, M, 2)
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
    # injection points (channel 4 + cond[2:6]).
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


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_theta_logabsdet_du_matches_autograd(seed):
    torch.manual_seed(seed)
    u = (torch.randn(4, dtype=torch.float64) * 1.5).requires_grad_(False)
    # Reference: |det| of the full reparameterization Jacobian.
    J = torch.autograd.functional.jacobian(
        lambda v: inv.theta_from_unconstrained(v), u
    )
    ref = torch.linalg.slogdet(J)[1]  # log|det J|
    got = inv.theta_logabsdet_du(u)
    assert float(got) == pytest.approx(float(ref), rel=1e-5, abs=1e-6)


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
    flat = inv.profile_summary(res, obs)
    assert flat["profile_param"] == "R_amp"
    assert flat["profile_excess_ci_low"] <= flat["profile_excess_ci_high"]
    assert np.isfinite(flat["profile_excess_ci_width"])
    assert "profile_R_amp_ci_low" in flat and "profile_R_amp_ci_high" in flat
    # theta_true known -> coverage booleans emitted.
    assert isinstance(flat["profile_excess_covered"], bool)
    assert isinstance(flat["profile_R_amp_covered"], bool)
    # Profiled NLL never beats the unconstrained min by construction.
    assert res.nll.min() >= res.nll_min - 1e-6


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


def test_run_mcmc_accept_rate_and_summary_columns():
    model = _tiny_source_itr_model().eval()
    obs = _fake_observation_set()
    obs.targets = obs.targets + 0.5
    theta_hat = obs.theta_true.to(torch.float32)
    res = inv.run_mcmc(
        model, obs, theta_hat, sigma_eff2=0.1,
        n_samples=40, burn=10, step_size=0.1, level=0.9, seed=3,
    )
    assert res.theta_samples.shape == (40, 4)
    assert 0.0 <= res.accept_rate <= 1.0
    # Samples respect the physical box (uniform-theta prior via the Jacobian).
    assert np.all(res.theta_samples[:, 0] >= 0.05 - 1e-4)
    assert np.all(res.theta_samples[:, 0] <= 1.0 + 1e-4)
    assert np.all(res.theta_samples[:, 1] >= -1e-4)
    assert np.all(
        res.theta_samples[:, 0] + res.theta_samples[:, 1] <= R_PEAK_MAX + 1e-3
    )

    flat = inv.mcmc_summary(res, obs)
    assert flat["mcmc_excess_ci_low"] <= flat["mcmc_excess_ci_high"]
    assert np.isfinite(flat["mcmc_excess_mean"])
    for nm in ("R_base", "R_amp", "y0", "sigma"):
        assert f"mcmc_{nm}_mean" in flat
        assert flat[f"mcmc_{nm}_ci_low"] <= flat[f"mcmc_{nm}_ci_high"]
    assert isinstance(flat["mcmc_excess_covered"], bool)
