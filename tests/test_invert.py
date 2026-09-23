"""Tests for the source_itr_sin inverse solver (Stage 1 MVP).

The load-bearing checks are checkpoint-free: the torch reparameterization must
reproduce the two theta injection points *bit-for-bit* (within float tolerance)
against the real forward pipeline:

  1. cond_static[1:5]  <- build_cond_vector_sin  (linear over RC_SIN_RANGES)
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
from problems.source_itr_sin import (
    RC_MIN,
    RC_SIN_RANGES,
    R_PEAK_MAX,
    RC_Y_CHANNEL,
    build_cond_vector_sin,
    rc_log_norm,
)
from src.physics.internal_source import (
    make_rc_sin_profile,
)
from scripts import invert as inv
from scripts.inverse_adapters import (
    ForcingAdapter,
    ForcingItrSinAdapter,
    InverseAdapter,
    SourceItrSinAdapter,
)


def test_legacy_forcing_checkpoint_adaptation_is_exact_at_zero_source_time():
    from problems.registry import get_problem

    dims = get_problem("forcing_itr_sin", "temporal_encoder").dims
    embed_dim = 4
    config = {
        "benchmark": {"name": "forcing_itr_sin"},
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
    (0.20, 0.50),
    (0.05, 2.90),   # near amp ceiling at the R_base floor
    (0.90, 0.10),
    (0.50, 0.00),   # no-void floor
    (0.75, 1.20),
]
THETAS = [tuple(350.0 * value for value in theta) for theta in THETAS]


@pytest.mark.parametrize("theta", THETAS)
def test_cond_slice_matches_build_cond_vector_sin(theta):
    R_base, R_amp = theta
    # Reference: the real cond builder. Patch slots are irrelevant here; we only
    # compare the void slice cond[1:3], so any valid patch is fine.
    x_lo, x_hi, y_lo, y_hi = 0.0, 1.0, 0.0, 1.0
    w_h = h_h = 0.1
    xcr, ycr = _patch_center_ranges(x_lo, x_hi, y_lo, y_hi, w_h, h_h)
    cond = build_cond_vector_sin(
        t_bar_norm=0.3,
        R_base=R_base, A=R_amp,
        x_h=0.5, y_h=0.5, w_h=w_h, h_h=h_h,
        x_center_range=xcr, y_center_range=ycr,
        x_length_scale=(x_hi - x_lo), y_length_scale=(y_hi - y_lo),
    )
    ref_slice = cond[1:3]

    theta_t = torch.tensor(theta, dtype=torch.float64)
    got = inv.theta_to_cond_slice(theta_t).numpy()
    np.testing.assert_allclose(got, ref_slice, rtol=0, atol=1e-6)


@pytest.mark.parametrize("theta", THETAS)
def test_rc_channel_matches_reference(theta):
    R_base, R_amp = theta
    Ny, Nx = 100, 100
    y_grid = np.linspace(0.0, 1.0, Ny).astype(np.float32)

    Rc_y = make_rc_sin_profile(y_grid, R_base=R_base, A=R_amp)
    ref_channel = np.broadcast_to(rc_log_norm(Rc_y, rc_min=RC_MIN, rc_max=R_PEAK_MAX)[None, :], (Nx, Ny))

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
    u = torch.from_numpy(rng.normal(0, 3.0, size=(500, 2)))
    theta = inv.theta_from_unconstrained(u)
    R_base, R_amp = (theta[:, i] for i in range(2))

    base_lo, base_hi = RC_SIN_RANGES["R_base"]

    assert torch.all(R_base >= base_lo - 1e-9) and torch.all(R_base <= base_hi + 1e-9)
    # Dependent amplitude ceiling: R_amp <= R_PEAK_MAX - R_base, i.e. peak <= R_PEAK_MAX.
    assert torch.all(R_amp >= -1e-9)
    assert torch.all(R_base + R_amp <= R_PEAK_MAX + 1e-6)


# ---------------------------------------------------------------------------
# Operator wiring + gradient flow, using a tiny untrained source_itr_sin FNO.
# ---------------------------------------------------------------------------

def _tiny_source_itr_sin_model():
    from src.operators.fno2d import FNO2d
    from problems.source_itr_sin import COND_STATIC_DIM, SPATIAL_CHANNELS_TEMPORAL
    from problems.forcing import FORCING_TEMPORAL_TOKEN_DIM

    torch.manual_seed(1)
    return FNO2d(
        modes1=2, modes2=2, width=8,
        in_channels=SPATIAL_CHANNELS_TEMPORAL, out_channels=1, n_layers=2,
        cond_static_dim=COND_STATIC_DIM,
        temporal_token_dim=FORCING_TEMPORAL_TOKEN_DIM,
        temporal_hidden=16, forcing_embed_dim=16,
        use_forcing_time_aug=False,
        s_y_channel=3,
    )


def _fake_observation_set(Nx=16, Ny=16, N=3, M=128):
    """Minimal ObservationSet with random scaffolding for wiring/grad tests."""
    spatial = torch.zeros(N, Nx, Ny, 5)
    spatial[..., 1] = torch.linspace(0, 1, Nx)[None, :, None]
    spatial[..., 2] = torch.linspace(0, 1, Ny)[None, None, :]
    cond = torch.zeros(N, 7)
    cond[:, 0] = torch.linspace(0.3, 1.0, N)  # t_bar_norm early/mid/late
    forcing_seq = torch.zeros(N, M, FORCING_TEMPORAL_TOKEN_DIM)
    forcing_seq[..., 0] = torch.linspace(0, 1, M)
    targets = torch.zeros(N, Nx, Ny, 1)
    y_grid = torch.linspace(0, 1, Ny)
    return inv.ObservationSet(
        sid=0, time_indices=list(range(N)),
        spatial=spatial, cond=cond, forcing_seq=forcing_seq,
        targets=targets, y_grid=y_grid, Nx=Nx,
        theta_true=torch.tensor([105.0, 350.0]),
    )


def test_predict_fullfield_injects_both_points_and_is_differentiable():
    model = _tiny_source_itr_sin_model().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    obs = _fake_observation_set()

    u = torch.zeros(2, requires_grad=True)
    theta = inv.theta_from_unconstrained(u)
    pred = inv.predict_fullfield(model, obs, theta)
    assert pred.shape == (obs.spatial.shape[0], obs.Nx, obs.spatial.shape[2], 1)

    loss = (pred ** 2).mean()
    loss.backward()
    # Gradient must flow back to all four unconstrained params through the two
    # injection points (channel 4 + cond[1:3]).
    assert u.grad is not None
    assert torch.all(torch.isfinite(u.grad))
    assert torch.any(u.grad != 0)


def test_predict_fullfield_changes_with_theta():
    model = _tiny_source_itr_sin_model().eval()
    obs = _fake_observation_set()
    theta_a = torch.tensor([70.0, 35.0])
    theta_b = torch.tensor([280.0, 700.0])
    with torch.no_grad():
        pa = inv.predict_fullfield(model, obs, theta_a)
        pb = inv.predict_fullfield(model, obs, theta_b)
    assert not torch.allclose(pa, pb)


@pytest.mark.parametrize("optimizer", ["adam_lbfgs", "nelder_mead"])
def test_invert_sim_recovers_theta_against_self_generated_target(optimizer):
    """End-to-end loop sanity: with the tiny model as its *own* ground truth,
    the optimizer drives the data loss down and lands a finite theta in-box.

    This is a wiring/convergence check on the optimizer plumbing, not a
    capability claim — the model is untrained, so "truth" here is whatever the
    model maps a planted theta to.
    """
    model = _tiny_source_itr_sin_model().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    obs = _fake_observation_set()

    theta_star = torch.tensor([140.0, 525.0])
    with torch.no_grad():
        obs.targets = inv.predict_fullfield(model, obs, theta_star).detach()

    cfg = inv.InversionConfig(n_starts=4, adam_steps=150, adam_lr=0.1,
                              lbfgs_steps=30, seed=1, optimizer=optimizer)
    result = inv.invert_sim(model, obs, cfg)
    assert result.loss < 1e-4
    # Landed theta is inside the physical box.
    th = result.theta_hat
    assert 17.5 - 1e-6 <= th[0] <= 350.0 + 1e-6
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
    model = _tiny_source_itr_sin_model().eval()
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
    model = _tiny_source_itr_sin_model().eval()
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

    theta_star = torch.tensor([140.0, 525.0])
    with torch.no_grad():
        obs.targets = inv.predict_fullfield(model, obs, theta_star).detach()

    cfg = inv.InversionConfig(n_starts=4, adam_steps=200, adam_lr=0.1,
                              lbfgs_steps=40, seed=2)
    result = inv.invert_sim(model, obs, cfg)
    assert result.loss < 1e-4
    th = result.theta_hat
    assert 17.5 - 1e-6 <= th[0] <= 350.0 + 1e-6
    assert th[1] >= -1e-6 and (th[0] + th[1]) <= R_PEAK_MAX + 1e-4


# ---------------------------------------------------------------------------
# Stage 3: LHS multistart + dG/dtheta SVD identifiability diagnostics.
# ---------------------------------------------------------------------------

def test_lhs_starts_are_stratified():
    """Each void scalar's box fraction has exactly one start per 1/n stratum."""
    rng = np.random.default_rng(0)
    n = 8
    u = inv.lhs_starts_u(n, rng)
    assert u.shape == (n, 2)
    frac = 1.0 / (1.0 + np.exp(-u))  # sigmoid: u -> box fraction
    for j in range(2):
        s = np.sort(frac[:, j])
        for k in range(n):
            assert k / n - 1e-9 <= s[k] <= (k + 1) / n + 1e-9


def test_lhs_starts_map_into_theta_box():
    rng = np.random.default_rng(1)
    u = torch.from_numpy(inv.lhs_starts_u(16, rng))
    theta = inv.theta_from_unconstrained(u)
    R_base, R_amp = (theta[:, i] for i in range(2))
    assert torch.all(R_base >= 17.5 - 1e-6) and torch.all(R_base <= 350.0 + 1e-6)
    assert torch.all(R_amp >= -1e-6)
    assert torch.all(R_base + R_amp <= R_PEAK_MAX + 1e-4)  # dependent ceiling


def test_invert_sim_lhs_and_normal_both_recover():
    """The new default LHS starts and the legacy normal starts both converge."""
    model = _tiny_source_itr_sin_model().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    obs = _fake_observation_set()
    theta_star = torch.tensor([140.0, 525.0])
    with torch.no_grad():
        obs.targets = inv.predict_fullfield(model, obs, theta_star).detach()
    for sampling in ("lhs", "normal"):
        cfg = inv.InversionConfig(n_starts=4, adam_steps=150, adam_lr=0.1,
                                  lbfgs_steps=30, seed=1, start_sampling=sampling)
        result = inv.invert_sim(model, obs, cfg)
        assert result.loss < 1e-4, sampling


def test_observation_jacobian_shape_finite_nonzero():
    model = _tiny_source_itr_sin_model().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    obs = _fake_observation_set(Nx=8, Ny=8, N=2)
    theta = torch.tensor([140.0, 420.0])

    # Full-field: m = N * Nx * Ny * C.
    J = inv.observation_jacobian(model, obs, theta)
    m_full = obs.spatial.shape[0] * obs.Nx * obs.spatial.shape[2] * 1
    assert J.shape == (m_full, 2)
    assert torch.all(torch.isfinite(J))
    assert torch.any(J != 0)

    # Sparse: m shrinks to N * n_sensors * C.
    obs.mask = inv.build_interface_sensor_mask(
        np.linspace(0.0, 1.0, 8), np.linspace(0.0, 1.0, 8),
        interface_x=0.5, x_halfwidth=0.2, n_y=4,
    )
    Js = inv.observation_jacobian(model, obs, theta)
    assert Js.shape == (obs.spatial.shape[0] * int(obs.mask.sum()) * 1, 2)


def test_svd_identifiability_flags_weak_parameter():
    """When R_base barely moves the observations, it is the least-identified
    axis and the ridge-alignment metric stays low (not an amp/sigma issue)."""
    rng = np.random.default_rng(1)
    cols = rng.normal(size=(60, 4))
    cols[:, 0] *= 1e-6  # R_base nearly invisible
    rep = inv.svd_identifiability(cols)
    least = rep["least_identified_dir"]
    assert abs(least[0]) > 0.99
    assert rep["param_sensitivity"][0] < rep["param_sensitivity"][1:].min()


def test_sensitivity_report_orthonormal_and_summary_columns():
    model = _tiny_source_itr_sin_model().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    obs = _fake_observation_set(Nx=8, Ny=8, N=2)
    theta = torch.tensor([140.0, 420.0])
    rep = inv.sensitivity_report(model, obs, theta)

    S = rep["singular_values"]
    assert np.all(np.diff(S) <= 1e-9)                       # descending
    V = rep["right_vectors"]
    np.testing.assert_allclose(V.T @ V, np.eye(2), atol=1e-6)  # orthonormal
    assert rep["cond_number"] >= 1.0 - 1e-9

    flat = inv.sensitivity_summary(rep)
    for k in ("cond_number",
              "sv_0", "sens_A", "least_dir_A"):
        assert k in flat and np.isfinite(flat[k])


# ---------------------------------------------------------------------------
# Stage 4: FV refinement + theta-local surrogate error.
#
# The load-bearing check is checkpoint-free and needs no trained model: a tiny
# *real* source_itr_sin FV dataset is generated with the exact data-generation
# operator, and ``fv_predict_masked`` at the stored theta_true must reproduce
# the stored trajectory snapshots. This proves the FV operator is rebuilt
# correctly (k=3/35 layers, interface_R=[Rc(y)], the patch source, the IC, the
# generation constants) so the FNO MAP can be honestly re-checked against the
# real solver.
# ---------------------------------------------------------------------------

def _tiny_source_itr_sin_fv_dataset(Nx=12, Ny=12, num_sims=2, t_final=0.06, dt=0.01):
    """Build a tiny *real* source_itr_sin FV dataset (save_stride=1) in memory.

    Mirrors ``data/generate_dataset.py`` with the same generation constants the
    inverse solver's ``build_fv_base_kwargs`` reconstructs, so the FV solve
    inside ``fv_predict_masked`` reproduces these trajectories bit-for-bit (up to
    the float32 storage cast). Returns ``(ds, mu_global, sigma_global)``.
    """
    from problems.registry import get_problem
    from src.physics.fv_solver_1d import build_time_grid
    from data.dataset import SnapshotPairDataset, compute_global_stats

    spec = get_problem("source_itr_sin", "temporal_encoder")
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
    ds, mu_global, sigma_global = _tiny_source_itr_sin_fv_dataset(num_sims=5)
    _write_inverse_dataset_dir(ds, tmp_path)

    config = {"benchmark": {"name": "source_itr_sin", "representation": "temporal_encoder"}}
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
    ds, mu_global, sigma_global = _tiny_source_itr_sin_fv_dataset(num_sims=4)
    _write_inverse_dataset_dir(ds, tmp_path)
    config = {"benchmark": {"name": "source_itr_sin", "representation": "temporal_encoder"}}
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
    return _tiny_source_itr_sin_fv_dataset()


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
    model = _tiny_source_itr_sin_model().eval()
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
    model = _tiny_source_itr_sin_model().eval()
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
        float(theta_true[0]) + 17.5,
        max(0.0, float(theta_true[1]) - 35.0),
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
    assert 17.5 - 1e-6 <= float(tp[0]) <= 350.0 + 1e-6
    assert float(tp[1]) >= -1e-6 and (float(tp[0]) + float(tp[1])) <= R_PEAK_MAX + 1e-4

    flat = inv.fv_refine_summary(res, obs, obs.y_grid)
    for name in ("R_base", "A"):
        assert f"{name}_fvpolish" in flat and np.isfinite(flat[f"{name}_fvpolish"])
        assert f"{name}_fvpolish_abserr" in flat
    assert np.isfinite(flat["A_fvpolish"])
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
    model = _tiny_source_itr_sin_model().eval()
    obs = _fake_observation_set()
    theta = obs.theta_true.to(torch.float32)
    # Make the data exactly the model's own prediction -> residual is ~zero.
    with torch.no_grad():
        obs.targets = inv.predict_fullfield(model, obs, theta).detach()
    nll0 = float(inv.neg_log_likelihood(model, obs, theta, 0.01))
    assert nll0 == pytest.approx(0.0, abs=1e-6)

    # Now perturb theta so residuals are nonzero, and check 1/sigma_eff2 scaling.
    theta2 = theta.clone()
    theta2[1] = theta2[1] + 105.0
    nll_a = float(inv.neg_log_likelihood(model, obs, theta2, 0.02))
    nll_b = float(inv.neg_log_likelihood(model, obs, theta2, 0.08))
    assert nll_a > 0 and nll_b > 0
    assert nll_a / nll_b == pytest.approx(0.08 / 0.02, rel=1e-5)


@pytest.mark.parametrize("fixed_index", [0, 1])
def test_theta_profile_pins_coord_and_respects_amp_ceiling(fixed_index):
    torch.manual_seed(0)
    u = torch.randn(4, dtype=torch.float64)
    pinned = {0: 140.0, 1: 245.0}[fixed_index]
    theta = inv._theta_profile(u, fixed_index, pinned)
    assert float(theta[fixed_index]) == pytest.approx(pinned)
    # Free coords stay in the physical box; the dependent amp ceiling holds.
    R_base, R_amp = float(theta[0]), float(theta[1])
    assert 17.5 - 1e-6 <= R_base <= 350.0 + 1e-6
    assert R_amp >= -1e-6 and (R_base + R_amp) <= R_PEAK_MAX + 1e-4


def test_theta_profile_high_pinned_amp_caps_free_base():
    # Pinning R_amp near its ceiling must shrink the free R_base box so the peak
    # R_base + R_amp never exceeds R_PEAK_MAX. A large positive u0 (sigmoid -> 1)
    # would map R_base to base_hi=350 without the cap, giving 350 + 980 = 1330.
    pinned_amp = 980.0
    u = torch.tensor([20.0, 0.0], dtype=torch.float64)
    theta = inv._theta_profile(u, fixed_index=1, fixed_value=pinned_amp)
    R_base, R_amp = float(theta[0]), float(theta[1])
    assert R_amp == pytest.approx(pinned_amp)
    base_ceiling = R_PEAK_MAX - pinned_amp  # 70.0
    assert R_base == pytest.approx(base_ceiling, abs=1e-6)
    assert (R_base + R_amp) <= R_PEAK_MAX + 1e-9
    # Lower edge of the shrunk box still reaches base_lo at u0 -> -inf.
    u_lo = torch.tensor([-20.0, 0.0], dtype=torch.float64)
    theta_lo = inv._theta_profile(u_lo, fixed_index=1, fixed_value=pinned_amp)
    assert float(theta_lo[0]) == pytest.approx(RC_SIN_RANGES["R_base"][0], abs=1e-6)


def test_profile_likelihood_summary_columns_present():
    model = _tiny_source_itr_sin_model().eval()
    obs = _fake_observation_set()
    obs.targets = obs.targets + 0.5  # nonzero residuals against the untrained net
    theta_hat = obs.theta_true.to(torch.float32)
    res = inv.profile_likelihood(
        model, obs, theta_hat, sigma_eff2=0.05, param_index=1, level=0.95,
    )
    assert res.param_name == "A"
    assert np.isfinite(res.ci_high - res.ci_low)
    assert res.ci_low <= res.ci_high
    # A single call reports the reference it was given. Where the walk beat that
    # reference it hands back the better theta instead of silently keeping it;
    # reconciling the two is profile_parameters' job, not this function's.
    assert res.nll.min() >= res.nll_min - 1e-6 or res.theta_mle is not None
    summary = inv.profile_interval_summary(res)
    assert summary["profile_A_ci_width"] == pytest.approx(res.ci_high - res.ci_low)
    assert summary["profile_A_n_evals"] == res.n_evals == res.grid.size
    assert summary["profile_A_bound_limited"] is not (
        res.lower_closed and res.upper_closed
    )


def test_polish_unrestricted_lowers_the_likelihood_reference():
    # The MAP fit minimizes a regularized mean-of-squares, so its optimum is not
    # the likelihood's. The polish is what puts theta_hat on the NLL the Wilks
    # threshold is measured against.
    model = _tiny_source_itr_sin_model().eval()
    obs = _fake_observation_set()
    obs.targets = obs.targets + 0.5
    seed = torch.zeros(2, dtype=torch.float32)
    start = float(inv.neg_log_likelihood(
        model, obs, inv._SOURCE_ITR_SIN_ADAPTER.theta_from_unconstrained(seed),
        0.05, inv._SOURCE_ITR_SIN_ADAPTER,
    ))
    nll, theta = inv._polish_unrestricted(
        model, obs, 0.05, seed, adam_steps=40, lbfgs_steps=10
    )
    assert nll <= start + 1e-9
    assert theta.shape == (2,)


def test_profile_is_deterministic_and_evaluates_only_endpoint_decisions():
    model = _tiny_source_itr_sin_model().eval()
    obs = _fake_observation_set()
    obs.targets = obs.targets + 0.5
    theta_hat = obs.theta_true.to(torch.float32)
    kwargs = dict(sigma_eff2=0.05, level=0.95,
                  adapter=inv._SOURCE_ITR_SIN_ADAPTER)
    a = inv.profile_likelihood(model, obs, theta_hat, **kwargs)
    b = inv.profile_likelihood(model, obs, theta_hat, **kwargs)
    np.testing.assert_allclose(a.nll, b.nll)
    np.testing.assert_allclose(a.grid, b.grid)
    assert a.ci_low == pytest.approx(b.ci_low)
    assert a.ci_high == pytest.approx(b.ci_high)
    # Bracket-and-bisect is what makes a dense profile mesh unnecessary: the walk
    # is capped at two bound-to-bound marches plus two bisections.
    assert a.n_evals <= 1 + 2 * (inv.PROFILE_WALK_MAX_STEPS + inv.PROFILE_ROOT_MAX_STEPS)
    assert np.all(np.diff(a.grid) > 0)


def test_profile_refit_scalar_pins_and_finds_the_global_nuisance_optimum():
    # Pinning one of two coordinates leaves a single free sigmoid coordinate, so
    # a brute-force sweep of that coordinate is the ground truth. The scan in
    # front of Brent is what keeps this true: the objective is bimodal here and
    # a bare local solver settles into the interior basin.
    model = _tiny_source_itr_sin_model().eval()
    obs = _fake_observation_set()
    obs.targets = obs.targets + 0.5
    scalar_nll, scalar_theta, _ = inv._profile_refit_scalar(model, obs, 0.05, 1, 0.4)

    adapter = inv._SOURCE_ITR_SIN_ADAPTER
    best = float("inf")
    for s in np.linspace(1e-6, 1 - 1e-6, 201):
        u = torch.zeros(2, dtype=torch.float32)
        u[0] = float(np.log(s) - np.log1p(-s))
        with torch.no_grad():
            theta = adapter.theta_profile(u, 1, 0.4)
            best = min(best, float(
                inv.neg_log_likelihood(model, obs, theta, 0.05, adapter)
            ))

    assert float(scalar_theta[1]) == pytest.approx(0.4, abs=1e-6)
    assert scalar_nll <= best + 1e-3


def test_profile_refit_scalar_handles_a_one_parameter_adapter():
    # With a 1-D theta the pin leaves nothing free, so the refit is a single
    # likelihood evaluation at the pinned value.
    adapter = ForcingAdapter()
    model = _tiny_forcing_model()
    obs = _fake_forcing_observation_set(Nx=6, Ny=6, N=1)
    nll, theta, _ = inv._profile_refit_scalar(
        model, obs, 0.05, 0, 0.6, adapter=adapter
    )
    assert float(theta[0]) == pytest.approx(0.6, abs=1e-6)
    assert np.isfinite(nll)


def test_laplace_spectrum_eigs_are_singular_values_squared_over_variance():
    model = _tiny_source_itr_sin_model().eval()
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
    for i in range(2):
        assert f"laplace_eig_{i}" in flat
    for nm in ("R_base", "A"):
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
        use_forcing_time_aug=True,
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
        spatial_profile_bins=np.ones(8, dtype=np.float32),
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


@pytest.mark.parametrize("optimizer", ["adam_lbfgs", "nelder_mead"])
def test_forcing_adapter_scalar_recovery_through_cin_conditioning(optimizer):
    adapter = ForcingAdapter()
    model = _tiny_forcing_model()
    obs = _fake_forcing_observation_set()
    theta_star = torch.tensor([0.72])
    with torch.no_grad():
        obs.targets = inv.predict_fullfield(model, obs, theta_star, adapter).detach()

    if optimizer == "nelder_mead":
        def require_forward_only(module, inputs):
            assert not torch.is_grad_enabled()

        model.register_forward_pre_hook(require_forward_only)

    cfg = inv.InversionConfig(
        n_starts=3, adam_steps=120, adam_lr=0.08, lbfgs_steps=25,
        seed=4, start_sampling="lhs", optimizer=optimizer,
    )
    result = inv.invert_sim(model, obs, cfg, adapter)
    assert result.loss < 1e-7
    assert abs(float(result.theta_hat[0]) - float(theta_star[0])) < 2e-2
    assert len(result.per_start_losses) == cfg.n_starts
    assert result.loss == min(result.per_start_losses)
    best_start = int(np.argmin(result.per_start_losses))
    torch.testing.assert_close(result.theta_hat, result.per_start_theta[best_start])


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
        param_index=0, level=0.9, adapter=adapter,
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
        InverseAdapter.from_config({"benchmark": {"name": "source_itr_sin"}}),
        SourceItrSinAdapter,
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

def test_artifact_dir_writes_expected_npz_keys_source_itr_sin(tmp_path):
    from types import SimpleNamespace

    adapter = SourceItrSinAdapter()
    model = _tiny_source_itr_sin_model().eval()
    obs = _fake_observation_set()
    obs.targets = obs.targets + 0.5
    theta_hat = obs.theta_true.to(torch.float32)

    report = inv.sensitivity_report(model, obs, theta_hat, adapter)
    J = inv.observation_jacobian(model, obs, theta_hat, adapter)
    result = SimpleNamespace(theta_hat=theta_hat)

    path = inv._write_sim_artifact(
        str(tmp_path), 7, adapter,
        obs=obs, result=result, report=report, J=J,
        fv_res=None, profiles=None, sigma_eff2=0.1, c_fno=0.01,
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
    assert str(d["benchmark"]) == "source_itr_sin"
    assert int(d["sim_id"]) == 7
    assert d["theta_hat"].shape == (2,)
    assert d["theta_true"].shape == (2,)
    assert d["theta_bounds"].shape == (2, 2)
    assert d["param_scales"].shape == (2,)
    assert str(d["dataset_path"]) == "data/sourceitr_smoke"
    assert str(d["split_name"]) == "test"
    assert d["right_vectors"].shape == (2, 2)
    assert d["singular_values"].shape == (2,)
    assert d["observation_jacobian"].shape[1] == 2
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
        fv_res=None, profiles=None, sigma_eff2=None, c_fno=None,
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
THETAS_SIN = [tuple(350.0 * value for value in theta) for theta in THETAS_SIN]


def test_source_and_forcing_sin_adapters_use_distinct_physical_ranges():
    source, forcing = SourceItrSinAdapter(), ForcingItrSinAdapter()
    u = torch.tensor([[-2.0, -1.0], [0.0, 0.0], [2.0, 1.0]], dtype=torch.float64)
    source_theta = source.theta_from_unconstrained(u)
    forcing_theta = forcing.theta_from_unconstrained(u)
    assert source.profile_bounds(0) == (17.5, 350.0)
    assert source.profile_bounds(1) == (0.0, 1032.5)
    assert forcing.profile_bounds(0) == (0.05, 1.0)
    assert forcing.profile_bounds(1) == (0.0, 2.95)
    torch.testing.assert_close(source_theta, 350.0 * forcing_theta)
    torch.testing.assert_close(
        source.cond_slice_from_theta(source_theta),
        forcing.cond_slice_from_theta(forcing_theta),
    )
    y = torch.linspace(0.0, 1.0, 21, dtype=torch.float64)
    torch.testing.assert_close(
        source.spatial_channel_from_theta(source_theta, y, 4),
        forcing.spatial_channel_from_theta(forcing_theta, y, 4),
    )


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
    ref_channel = np.broadcast_to(rc_log_norm(Rc_y, rc_min=RC_MIN, rc_max=R_PEAK_MAX)[None, :], (Nx, Ny))

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


@pytest.mark.parametrize("param_index", [0, 1])
def test_parameter_profile_matches_correlated_gaussian(monkeypatch, param_index):
    from types import SimpleNamespace
    from scipy.stats import chi2

    mean = torch.tensor([0.4, 0.8], dtype=torch.float64)
    scales = torch.tensor([0.06, 0.16], dtype=torch.float64)
    rho = 0.8

    def nll(model, obs, theta, sigma_eff2, adapter):
        z = (theta - mean) / scales
        return (z[0] ** 2 - 2 * rho * z[0] * z[1] + z[1] ** 2) / (2 * (1 - rho**2))

    monkeypatch.setattr(inv, "neg_log_likelihood", nll)
    obs = SimpleNamespace(spatial=torch.empty(0))
    res = inv.profile_likelihood(
        None, obs, mean, 1.0, param_index=param_index, adapter=ForcingItrSinAdapter(),
    )
    half = np.sqrt(chi2.ppf(0.95, 1)) * float(scales[param_index])
    assert res.ci_low == pytest.approx(float(mean[param_index]) - half, abs=4e-4)
    assert res.ci_high == pytest.approx(float(mean[param_index]) + half, abs=4e-4)
    assert res.lower_closed and res.upper_closed
    assert res.nll_min == pytest.approx(0.0, abs=1e-8)
    flat = inv.profile_interval_summary(res)
    assert flat[f"profile_{res.param_name}_ci_width"] == pytest.approx(2 * half, abs=8e-4)


def test_parameter_profile_marks_physical_bounds(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(inv, "neg_log_likelihood", lambda model, obs, theta, sigma, adapter: (theta ** 2).sum() * 0)
    res = inv.profile_likelihood(
        None, SimpleNamespace(spatial=torch.empty(0)), torch.tensor([0.4, 0.8]),
        1.0, param_index=0, adapter=ForcingItrSinAdapter(),
    )
    assert (res.ci_low, res.ci_high) == (0.05, 1.0)
    assert not res.lower_closed and not res.upper_closed
    assert inv.profile_interval_summary(res)["profile_R_base_bound_limited"]


def test_parameter_errors_are_separate_and_zero_amplitude_is_undefined():
    from types import SimpleNamespace
    adapter = ForcingItrSinAdapter()
    result = SimpleNamespace(
        theta_hat=torch.tensor([0.55, 0.3], dtype=torch.float64),
        theta_true=torch.tensor([0.5, 0.2], dtype=torch.float64), loss=0.0,
    )
    flat = adapter.summarize_result(result, torch.linspace(0, 1, 8))
    assert flat["R_base_abserr"] == pytest.approx(0.05)
    assert flat["R_base_rel_error_pct"] == pytest.approx(10.0)
    assert flat["A_abserr"] == pytest.approx(0.1)
    assert flat["A_rel_error_pct"] == pytest.approx(50.0)
    for near_zero in (0.0, 1e-7):
        result.theta_true[1] = near_zero
        flat = adapter.summarize_result(result, torch.linspace(0, 1, 8))
        assert np.isnan(flat["A_rel_error_pct"])
        assert np.isfinite(flat["A_abserr"])


def test_profiles_share_checked_optimum_and_persist_both_parameters(monkeypatch, tmp_path):
    from types import SimpleNamespace

    mean = torch.tensor([0.4, 0.8], dtype=torch.float64)
    def nll(model, obs, theta, sigma, adapter):
        z = (theta - mean) / torch.tensor([0.08, 0.15], dtype=torch.float64)
        return 0.5 * (z**2).sum()
    monkeypatch.setattr(inv, "neg_log_likelihood", nll)
    obs = _fake_observation_set()
    adapter = ForcingItrSinAdapter()
    theta, profiles = inv.profile_parameters(
        None, obs, torch.tensor([0.2, 1.2], dtype=torch.float64), 1.0,
        adapter=adapter, adam_steps=200, lbfgs_steps=40,
    )
    np.testing.assert_allclose(theta, mean, atol=1e-5)
    assert [p.param_name for p in profiles] == ["R_base", "A"]
    for profile in profiles:
        assert profile.nll_min == pytest.approx(0.0, abs=1e-5)
        assert profile.ci_low < float(theta[profile.param_index]) < profile.ci_high
        # The driver re-profiles until no walk beats the shared reference, so a
        # returned profile never dips under the NLL its threshold is measured
        # against. visual/pub/records.py rejects an artifact that does.
        assert profile.nll.min() >= profile.nll_min - inv.PROFILE_OPTIMUM_ATOL
    path = inv._write_sim_artifact(
        str(tmp_path), 0, adapter, obs=obs, result=SimpleNamespace(theta_hat=theta),
        report=None, J=None, fv_res=None, profiles=profiles, sigma_eff2=1.0,
        c_fno=None, noise_std=0.01, ci_level=0.95,
        dataset_path="test", dataset_fingerprint="abc", split_seed=0, split_name="test",
    )
    with np.load(path, allow_pickle=False) as stored:
        for profile in profiles:
            stem = f"profile_{profile.param_name}"
            np.testing.assert_array_equal(stored[f"{stem}_grid"], profile.grid)
            np.testing.assert_array_equal(stored[f"{stem}_nll"], profile.nll)
            assert float(stored[f"{stem}_ci_width"]) == pytest.approx(profile.ci_high - profile.ci_low)
        assert "profile_excess_int" not in stored.files
        assert not [k for k in stored.files if k.startswith("joint_")]


# ---------------------------------------------------------------------------
# Joint (R_base, A) NLL grid (--joint-nll-grid): the measured joint region that
# the identifiability figure contours. Both coordinates are pinned, so this is
# not a quadratic approximation of the profile pair.
# ---------------------------------------------------------------------------

def _quadratic_nll(mean=(0.5, 1.0), scale=(0.08, 0.15)):
    center = torch.tensor(mean, dtype=torch.float64)
    width = torch.tensor(scale, dtype=torch.float64)

    def nll(model, obs, theta, sigma, adapter):
        return 0.5 * (((theta.to(torch.float64) - center) / width) ** 2).sum()

    return nll


def _profile_pair(adapter, grids):
    return [inv.ProfileResult(
        param_index=i, param_name=adapter.param_names[i], level=0.95,
        grid=np.asarray(grid, dtype=np.float64),
        nll=np.zeros(len(grid)), nll_min=0.0,
        ci_low=float(min(grid)), ci_high=float(max(grid)),
    ) for i, grid in enumerate(grids)]


@pytest.mark.parametrize("adapter_type,scale", [(SourceItrSinAdapter, 350.0), (ForcingItrSinAdapter, 1.0)])
def test_joint_nll_grid_spans_profiles_clips_bounds_and_masks_the_ceiling(monkeypatch, adapter_type, scale):
    from scipy.stats import chi2

    monkeypatch.setattr(inv, "neg_log_likelihood", _quadratic_nll(mean=(0.5 * scale, scale), scale=(0.08 * scale, 0.15 * scale)))
    adapter = adapter_type()
    obs = _fake_observation_set()
    theta_hat = torch.tensor([0.5, 1.0], dtype=torch.float64) * scale
    # R_base narrower than its bounds (span follows the profile); A wider
    # (span is clipped back to the adapter bounds).
    profiles = _profile_pair(adapter, (np.array([0.3, 0.5, 0.7]) * scale, np.array([-1.0, 1.0, 5.0]) * scale))

    joint = inv.joint_nll_grid(
        None, obs, 1.0, adapter=adapter, profiles=profiles,
        theta_hat=theta_hat, n_grid=5,
    )

    base_axis = joint["joint_nll_grid_R_base"]
    amp_axis = joint["joint_nll_grid_A"]
    np.testing.assert_allclose(base_axis, np.linspace(0.3, 0.7, 5) * scale)
    np.testing.assert_allclose(amp_axis, np.linspace(*adapter.profile_bounds(1), 5))

    nll = joint["joint_nll"]
    assert nll.shape == (5, 5)
    inadmissible = amp_axis[None, :] > (adapter.rc_peak_max - base_axis[:, None])
    assert inadmissible.any(), "test grid must straddle the dependent ceiling"
    np.testing.assert_array_equal(np.isnan(nll), inadmissible)

    # The reference is the fit the marginal profiles used, so delta-ell shares
    # one scale with them and is never negative even though no grid point sits
    # exactly on the optimum.
    assert float(joint["joint_nll_min"]) == pytest.approx(0.0, abs=1e-12)
    assert np.nanmin(nll) > 0.0
    assert np.nanmin(nll - joint["joint_nll_min"]) >= 0.0
    assert float(joint["joint_threshold"]) == pytest.approx(chi2.ppf(0.95, 2) / 2.0)


def test_joint_nll_grid_rejects_unusable_requests(monkeypatch):
    monkeypatch.setattr(inv, "neg_log_likelihood", _quadratic_nll())
    adapter = ForcingItrSinAdapter()
    obs = _fake_observation_set()
    theta_hat = torch.tensor([0.5, 1.0], dtype=torch.float64)
    grids = ([0.3, 0.5, 0.7], [0.5, 1.0, 1.5])
    kwargs = dict(adapter=adapter, profiles=_profile_pair(adapter, grids),
                  theta_hat=theta_hat, n_grid=5)

    with pytest.raises(ValueError, match="n_grid >= 3"):
        inv.joint_nll_grid(None, obs, 1.0, **{**kwargs, "n_grid": 2})
    with pytest.raises(ValueError, match="both parameters"):
        inv.joint_nll_grid(None, obs, 1.0,
                           **{**kwargs, "profiles": kwargs["profiles"][:1]})
    with pytest.raises(ValueError, match="finite positive variance"):
        inv.joint_nll_grid(None, obs, 0.0, **kwargs)

    scalar = ForcingAdapter()
    with pytest.raises(ValueError, match="2-parameter benchmark"):
        inv.joint_nll_grid(None, obs, 1.0, adapter=scalar,
                           profiles=kwargs["profiles"],
                           theta_hat=theta_hat[:1], n_grid=5)


def test_joint_grid_round_trips_through_the_artifact_at_schema_4(monkeypatch, tmp_path):
    from types import SimpleNamespace

    monkeypatch.setattr(inv, "neg_log_likelihood", _quadratic_nll())
    adapter = ForcingItrSinAdapter()
    obs = _fake_observation_set()
    theta_hat = torch.tensor([0.5, 1.0], dtype=torch.float64)
    profiles = _profile_pair(adapter, ([0.3, 0.5, 0.7], [0.5, 1.0, 1.5]))
    joint = inv.joint_nll_grid(
        None, obs, 1.0, adapter=adapter, profiles=profiles,
        theta_hat=theta_hat, n_grid=4,
    )

    path = inv._write_sim_artifact(
        str(tmp_path), 11, adapter, obs=obs,
        result=SimpleNamespace(theta_hat=theta_hat), report=None, J=None,
        fv_res=None, profiles=profiles, joint=joint, sigma_eff2=1.0, c_fno=None,
        noise_std=0.01, ci_level=0.95, dataset_path="test",
        dataset_fingerprint="abc", split_seed=0, split_name="test",
    )
    with np.load(path, allow_pickle=False) as stored:
        assert int(stored["artifact_schema_version"]) == 4
        for key, value in joint.items():
            np.testing.assert_array_equal(stored[key], value)
        # The joint threshold is the 2-d chi-square level; the marginal
        # profiles keep the 1-d one. Conflating them would inflate the region.
        assert float(stored["joint_threshold"]) > float(stored["profile_threshold"])


@pytest.mark.parametrize("profile_index", [None, 0])
def test_cli_reports_parameter_errors_and_requested_profiles(monkeypatch, tmp_path, capsys, profile_index):
    import csv
    from types import SimpleNamespace

    adapter = ForcingItrSinAdapter()
    obs = _fake_observation_set()
    ds = SimpleNamespace(
        t_grid=np.array([0.0, 0.07, 0.15, 0.30]), Nt=4, Nx=obs.Nx,
        y_grid=obs.y_grid.numpy(), _split_ids={"test": [0]},
    )
    loaded = SimpleNamespace(
        config={"benchmark": {"name": "forcing_itr_sin", "representation": "temporal_encoder"}},
        mu_global=0.0, sigma_global=1.0, model=None,
    )
    monkeypatch.setattr(inv, "load_checkpoint", lambda *a, **k: loaded)
    monkeypatch.setattr(ForcingItrSinAdapter, "validate_model", lambda *a: None)
    monkeypatch.setattr(ForcingItrSinAdapter, "validate_dataset", lambda *a: None)
    monkeypatch.setattr(ForcingItrSinAdapter, "fv_base_kwargs", lambda *a, **k: {"y_grid": obs.y_grid.numpy()})
    monkeypatch.setattr(inv, "build_dataset_from_dir", lambda *a, **k: ds)
    monkeypatch.setattr(inv, "build_observation_set", lambda *a, **k: obs)
    monkeypatch.setattr(inv, "_checkpoint_fingerprint", lambda *a: "test")
    monkeypatch.setattr(inv, "_dataset_fingerprint", lambda *a: "test")
    monkeypatch.setattr(inv, "load_surrogate_calibration", lambda *a, **k: {"local_gate_pass": True, "sigma_fno_norm": 0.1})
    monkeypatch.setattr(inv, "invert_sim", lambda *a: inv.InversionResult(obs.theta_true.clone(), 0.0, obs.theta_true))
    fitted = torch.tensor([0.45, 1.0])

    def profiles(model, observed, theta, sigma, *, adapter, indices, **kwargs):
        assert indices == (None if profile_index is None else [profile_index])
        selected = range(2) if indices is None else indices
        return fitted, [inv.ProfileResult(
            param_index=i, param_name=adapter.param_names[i], level=0.95,
            grid=np.array([0.1, 0.5, 1.1]), nll=np.array([2.0, 0.0, 2.0]),
            nll_min=0.0, ci_low=0.1, ci_high=1.1,
            theta_mle=fitted, lower_closed=True, upper_closed=True,
        ) for i in selected]
    monkeypatch.setattr(inv, "profile_parameters", profiles)
    monkeypatch.setattr(inv, "_masked_residual", lambda *a: torch.tensor([0.1]))

    def verify(model, dataset, base, observed, theta, **kwargs):
        torch.testing.assert_close(theta, fitted)
        return inv.FVRefineResult(0.005, 0.005, 0.0)
    monkeypatch.setattr(inv, "fv_refine_report", verify)
    output = tmp_path / "results.csv"
    args = ["--checkpoint", "test", "--data-dir", "test", "--calibration-artifact", "test",
            "--noise-std", "0.01", "--out-csv", str(output), "--sensor-layout", "full_field"]
    if profile_index is not None:
        args += ["--profile-index", str(profile_index)]
    assert inv.main(args) == 0
    with output.open() as handle:
        row = next(csv.DictReader(handle))
    for i, name in enumerate(adapter.param_names):
        assert float(row[f"{name}_hat"]) == pytest.approx(float(fitted[i]))
        assert float(row[f"{name}_abserr"]) == pytest.approx(abs(float(fitted[i]) - float(obs.theta_true[i])))
        assert f"{name}_rel_error_pct" in row
    assert "profile_R_base_ci_width" in row
    assert ("profile_A_ci_width" in row) == (profile_index is None)
    assert not any("excess" in key or "S_R" in key for key in row)
    stdout = capsys.readouterr().out
    assert "R_base: true=" in stdout and "A: true=" in stdout
    assert "relative error=" in stdout and "width=" in stdout


def test_cli_joint_nll_grid_writes_the_block_and_guards_bad_invocations(monkeypatch, tmp_path):
    from types import SimpleNamespace

    adapter = ForcingItrSinAdapter()
    obs = _fake_observation_set()
    ds = SimpleNamespace(
        t_grid=np.array([0.0, 0.07, 0.15, 0.30]), Nt=4, Nx=obs.Nx,
        y_grid=obs.y_grid.numpy(), _split_ids={"test": [0]},
    )
    loaded = SimpleNamespace(
        config={"benchmark": {"name": "forcing_itr_sin", "representation": "temporal_encoder"}},
        mu_global=0.0, sigma_global=1.0, model=None,
    )
    monkeypatch.setattr(inv, "load_checkpoint", lambda *a, **k: loaded)
    monkeypatch.setattr(ForcingItrSinAdapter, "validate_model", lambda *a: None)
    monkeypatch.setattr(ForcingItrSinAdapter, "validate_dataset", lambda *a: None)
    monkeypatch.setattr(ForcingItrSinAdapter, "fv_base_kwargs", lambda *a, **k: {"y_grid": obs.y_grid.numpy()})
    monkeypatch.setattr(inv, "build_dataset_from_dir", lambda *a, **k: ds)
    monkeypatch.setattr(inv, "build_observation_set", lambda *a, **k: obs)
    monkeypatch.setattr(inv, "_checkpoint_fingerprint", lambda *a: "test")
    monkeypatch.setattr(inv, "_dataset_fingerprint", lambda *a: "test")
    monkeypatch.setattr(inv, "load_surrogate_calibration",
                        lambda *a, **k: {"local_gate_pass": True, "sigma_fno_norm": 0.1})
    monkeypatch.setattr(inv, "invert_sim",
                        lambda *a: inv.InversionResult(obs.theta_true.clone(), 0.0, obs.theta_true))
    monkeypatch.setattr(inv, "neg_log_likelihood", _quadratic_nll())
    monkeypatch.setattr(inv, "_masked_residual", lambda *a: torch.tensor([0.1]))
    monkeypatch.setattr(inv, "fv_refine_report", lambda *a, **k: inv.FVRefineResult(0.005, 0.005, 0.0))
    fitted = torch.tensor([0.5, 1.0], dtype=torch.float64)
    monkeypatch.setattr(inv, "profile_parameters", lambda *a, **k: (
        fitted, _profile_pair(adapter, ([0.3, 0.5, 0.7], [0.5, 1.0, 1.5]))
    ))

    artifacts = tmp_path / "artifacts"
    base = ["--checkpoint", "test", "--data-dir", "test", "--calibration-artifact", "test",
            "--noise-std", "0.01", "--out-csv", str(tmp_path / "results.csv"),
            "--sensor-layout", "full_field"]
    assert inv.main(base + ["--joint-nll-grid", "4", "--artifact-dir", str(artifacts)]) == 0
    with np.load(artifacts / "sim_00000.npz", allow_pickle=False) as stored:
        assert stored["joint_nll"].shape == (4, 4)
        assert stored["joint_nll_grid_R_base"].shape == (4,)
        assert stored["joint_nll_grid_A"].shape == (4,)

    # Off by default: the block must not appear without the flag.
    plain = tmp_path / "plain"
    assert inv.main(base + ["--artifact-dir", str(plain)]) == 0
    with np.load(plain / "sim_00000.npz", allow_pickle=False) as stored:
        assert not [k for k in stored.files if k.startswith("joint_")]

    with pytest.raises(ValueError, match="at least 3"):
        inv.main(base + ["--joint-nll-grid", "2", "--artifact-dir", str(artifacts)])
    with pytest.raises(ValueError, match="nowhere to write"):
        inv.main(base + ["--joint-nll-grid", "4"])
    with pytest.raises(ValueError, match="drop --profile-index"):
        inv.main(base + ["--joint-nll-grid", "4", "--artifact-dir", str(artifacts),
                         "--profile-index", "0"])
    loaded.config["benchmark"]["name"] = "forcing"
    monkeypatch.setattr(ForcingAdapter, "validate_model", lambda *a: None)
    with pytest.raises(ValueError, match="2-parameter benchmarks"):
        inv.main(base + ["--joint-nll-grid", "4", "--artifact-dir", str(artifacts)])


def test_profile_returns_a_better_optimum_found_while_bracketing(monkeypatch):
    # The walk pins values theta_hat never visited, so it can land below the fit
    # it started from. Reporting an interval against a reference that is not the
    # minimum would make delta-NLL negative, so the better theta goes back out.
    from types import SimpleNamespace

    mean = torch.tensor([0.7, 0.9], dtype=torch.float64)

    def nll(model, obs, theta, sigma, adapter):
        z = (theta.to(torch.float64) - mean) / torch.tensor([0.05, 0.12], dtype=torch.float64)
        return 0.5 * (z**2).sum()

    monkeypatch.setattr(inv, "neg_log_likelihood", nll)
    res = inv.profile_likelihood(
        None, SimpleNamespace(spatial=torch.empty(0)),
        torch.tensor([0.2, 0.9], dtype=torch.float64), 1.0, param_index=0, adapter=ForcingItrSinAdapter(),
    )
    assert res.theta_mle is not None
    assert float(res.theta_mle[0]) == pytest.approx(0.7, abs=0.05)
