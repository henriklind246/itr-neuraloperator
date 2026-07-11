"""Focused tests for the forcing-PINO diagnostic suite (Phase 1).

Small, fast units plus one tiny integration:

- ``severity_from_grid`` on analytic ``q_L`` fields (closed-form q_rms/Q_in).
- image-fidelity metrics on a synthetic temporal pulse (peak_ratio <= 1;
  integrated-impulse error shrinks as nt_img grows).
- ``recover_config`` two-tier behaviour (raises on missing prediction-critical
  fields; warns loudly when the authoritative ``forcing_image`` block is absent).
- CSV/JSON/PNG schema for both scripts' public entry points.
- one tiny integration: build a minimal ForcingCViT, save a checkpoint with a
  ``forcing_image`` block + ``(mu, sigma)``, then ``load_model_and_data`` +
  ``predict_grid`` + ``run_error_surface`` produce finite gnrmse and the right
  shapes.

Everything runs on CPU against synthetic data; no cluster checkpoint needed.
"""

import csv
import json
import warnings

import numpy as np
import pytest
import torch

from scripts._forcing_diag_common import (
    CkptConfig,
    load_model_and_data,
    predict_grid,
    recover_config,
    severity_from_grid,
)
from scripts.diagnose_forcing_error_surface import (
    CSV_FIELDS as ES_FIELDS,
    PATHWAY_FIELDS,
    run_error_surface,
)
from scripts.diagnose_forcing_image_fidelity import (
    CSV_FIELDS as IF_FIELDS,
    _bilinear_to_dense,
    fidelity_metrics,
    run_image_fidelity,
)
from scripts.diagnose_forcing_linearity import (
    CSV_FIELDS as LIN_FIELDS,
    _delta_metrics,
    _fv_dev,
    _linfit,
    _match_indices,
    _rms,
)
from scripts.diagnose_forcing_residual_adequacy import (
    STATS_FIELDS as RES_FIELDS,
    _hom_wall_res_profile,
    _interior_res_field,
    _right_dirichlet_res,
    _snapshot_indices,
    run_residual_adequacy,
)
from src.operators.train_pino import build_cvit, sample_forcing_params
from problems.forcing import A_AMP_REF

# Reuse the synthetic-forcing scaffolding rather than duplicating it.
from tests.test_train_pino_e2e import _forcing_config, _write_synthetic_forcing


# --------------------------------------------------------------------------- #
# Unit: forcing-severity math on analytic fields.                             #
# --------------------------------------------------------------------------- #

def test_severity_constant_field():
    # q(y,t) = 2 on [0,1] x [0,1]: q_rms = 2, Q_in = 2*1*1 = 2, amp = 2.
    y = np.linspace(0.0, 1.0, 33)
    t = np.linspace(0.0, 1.0, 41)
    q = np.full((y.size, t.size), 2.0)
    s = severity_from_grid(q, y, t)
    assert s["q_rms"] == pytest.approx(2.0, rel=1e-12)
    assert s["Q_in"] == pytest.approx(2.0, rel=1e-10)
    assert s["amp"] == pytest.approx(2.0, rel=1e-12)


def test_severity_linear_in_y():
    # q(y,t) = y (independent of t) on [0,1]^2:
    #   q_rms = sqrt(mean(y^2)) = sqrt(1/3); Q_in = ∫∫ y dy dt = 1/2.
    y = np.linspace(0.0, 1.0, 2049)
    t = np.linspace(0.0, 1.0, 5)
    q = np.repeat(y[:, None], t.size, axis=1)
    s = severity_from_grid(q, y, t)
    # q_rms is the arithmetic RMS over samples (sqrt(mean(q^2))); it approaches
    # the continuous sqrt(1/3) with an O(1/N) endpoint bias, so allow a loose
    # tolerance. Q_in is trapezoidal and matches the exact integral tightly.
    assert s["q_rms"] == pytest.approx(np.sqrt(1.0 / 3.0), rel=2e-3)
    assert s["Q_in"] == pytest.approx(0.5, rel=1e-6)
    assert s["amp"] == pytest.approx(1.0, rel=1e-12)


# --------------------------------------------------------------------------- #
# Unit: image-fidelity metrics on a synthetic separable pulse.                #
# --------------------------------------------------------------------------- #

def _pulse_metrics(nt_img: int) -> dict[str, float]:
    """Fidelity metrics for a narrow temporal Gaussian pulse at one nt_img."""
    ny = 8
    y_dense = np.linspace(0.0, 1.0, ny)
    t_dense = np.linspace(0.0, 1.0, 4096)
    a = np.exp(-(((t_dense - 0.5) / 0.03) ** 2))
    q_ref = np.outer(np.ones(ny), a)  # s(y)=1, a(t)=pulse

    y_img = y_dense.copy()  # isolate the temporal axis
    t_img = np.linspace(0.0, 1.0, nt_img)
    a_c = np.exp(-(((t_img - 0.5) / 0.03) ** 2))
    q_coarse = np.outer(np.ones(ny), a_c)
    q_rec = _bilinear_to_dense(q_coarse, y_img, t_img, y_dense, t_dense)
    return fidelity_metrics(
        q_ref, q_rec, q_coarse, y_dense, t_dense, y_img, t_img
    )


def test_image_fidelity_peak_ratio_not_above_one():
    # Linear interpolation of an off-sample-centred smooth peak cannot exceed
    # the true peak amplitude.
    m = _pulse_metrics(nt_img=64)
    assert m["peak_ratio"] <= 1.0 + 1e-6


def test_image_fidelity_impulse_error_shrinks_with_resolution():
    coarse = _pulse_metrics(nt_img=16)
    fine = _pulse_metrics(nt_img=1024)
    # A severely undersampled pulse mis-integrates far more than a resolved one.
    assert fine["integrated_impulse_err"] < coarse["integrated_impulse_err"]
    assert fine["integrated_impulse_err"] < 5e-2


# --------------------------------------------------------------------------- #
# Unit: recover_config two-tier behaviour.                                    #
# --------------------------------------------------------------------------- #

def _min_forcing_config(tmp_path):
    cfg = _forcing_config(tmp_path)
    return cfg


def test_recover_config_requires_model_state(tmp_path):
    cfg = _min_forcing_config(tmp_path)
    ckpt = {"config": cfg, "mu_global": 300.0, "sigma_global": 5.0}
    with pytest.raises(KeyError):
        recover_config(ckpt, None)


def test_recover_config_uses_forcing_image_block(tmp_path):
    cfg = _min_forcing_config(tmp_path)
    ckpt = {
        "model_state": {},
        "config": cfg,
        "mu_global": 300.0,
        "sigma_global": 5.0,
        "forcing_image": {
            "ny_img": 16, "nt_img": 20, "a_ref": A_AMP_REF,
            "t_ramp": 0.003, "c_dom": 0.0, "d_dom": 1.0, "t_final": 0.3,
        },
    }
    rc = recover_config(ckpt, None)
    assert isinstance(rc, CkptConfig)
    assert rc.grid_size == (16, 20)
    assert rc.a_ref == pytest.approx(A_AMP_REF)
    assert rc.hard_left_flux is True
    assert rc.provenance["ny_img"] == "checkpoint"
    assert rc.warnings == ()


def test_recover_config_warns_when_forcing_block_missing(tmp_path):
    cfg = _min_forcing_config(tmp_path)
    ckpt = {
        "model_state": {}, "config": cfg,
        "mu_global": 300.0, "sigma_global": 5.0,
    }
    # An old checkpoint that predates the forcing_image block is loaded WITH a
    # dataset; the grid supplies the ny_img fallback while nt_img still comes
    # from the stored config. The missing block must warn loudly (not silently).
    data = {
        "y_grid": np.linspace(0.0, 1.0, 16),
        "t_grid": np.linspace(0.0, 0.3, 6),
        "mu_global": 300.0, "sigma_global": 5.0,
    }
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        rc = recover_config(ckpt, data)
    assert any("forcing_image" in str(x.message) for x in w)
    assert any("forcing_image" in msg for msg in rc.warnings)
    # config-tier resolves nt_img; the dataset grid resolves ny_img.
    assert rc.nt_img == 20  # from training.pino.forcing.nt_img
    assert rc.provenance["nt_img"] == "config"
    assert rc.ny_img == 16  # from data["y_grid"]
    assert rc.provenance["ny_img"] == "default"


# --------------------------------------------------------------------------- #
# Integration fixture: a real (random-init) forcing checkpoint on disk.       #
# --------------------------------------------------------------------------- #

@pytest.fixture
def forcing_checkpoint(tmp_path):
    """Synthetic dataset + a minimal saved ForcingCViT checkpoint.

    Mirrors the trainer's checkpoint contract (``model_state``, ``mu_global``,
    ``sigma_global``, ``config``, ``forcing_image`` block) so the diagnostic
    loader exercises the real recovery + build path without a training run.
    """
    _write_synthetic_forcing(tmp_path, num_sims=20, Nt=6, Nx=10, Ny=16)
    cfg = _forcing_config(tmp_path)
    mu, sigma = 300.0, 5.0
    model = build_cvit(
        cfg, mu, sigma, grid_size=(16, 20), t_final=0.3, variant="forcing"
    )
    ckpt = {
        "model_state": model.state_dict(),
        "mu_global": mu,
        "sigma_global": sigma,
        "config": cfg,
        "epoch": 1,
        "best_val": 1.0,
        "forcing_image": {
            "ny_img": 16, "nt_img": 20, "a_ref": A_AMP_REF,
            "t_ramp": 0.003, "c_dom": 0.0, "d_dom": 1.0, "t_final": 0.3,
        },
    }
    ckpt_path = tmp_path / "cvit_best.pt"
    torch.save(ckpt, ckpt_path)
    return ckpt_path, tmp_path


def test_load_and_predict_grid_finite(forcing_checkpoint):
    ckpt_path, data_dir = forcing_checkpoint
    model, data, rc = load_model_and_data(str(ckpt_path), str(data_dir), "cpu")
    from scripts._forcing_diag_common import load_sim_params, select_ids

    sim_params = load_sim_params(rc.config)
    ids = select_ids(data, split="train", num_sims=4)
    params = [dict(sim_params[int(i)]) for i in ids]

    pred = predict_grid(model, params, data, rc, data["t_grid"], "cpu")
    nx = int(np.asarray(data["x_grid"]).shape[0])
    ny = int(np.asarray(data["y_grid"]).shape[0])
    nt = int(np.asarray(data["t_grid"]).shape[0])
    assert pred.shape == (len(ids), nt, nx, ny)
    assert np.isfinite(pred).all()
    # Denormalized predictions sit in a physical Kelvin range, not normalized.
    assert pred.mean() > 100.0


def test_error_surface_schema_and_outputs(forcing_checkpoint, tmp_path):
    ckpt_path, data_dir = forcing_checkpoint
    out_dir = tmp_path / "es_out"
    summary, out = run_error_surface(
        str(ckpt_path), str(data_dir), split="train", num_sims=4,
        out_dir=str(out_dir), device="cpu",
    )
    assert out == out_dir

    with open(out_dir / "error_surface.csv") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ES_FIELDS
        rows = list(reader)
    assert rows, "error_surface.csv has no rows"
    for r in rows:
        assert np.isfinite(float(r["gnrmse"]))

    with open(out_dir / "pathway_probe.csv") as f:
        preader = csv.DictReader(f)
        assert preader.fieldnames == PATHWAY_FIELDS
        prows = list(preader)
    cases = {r["case"] for r in prows}
    assert "baseline" in cases and "boundary_zero" in cases

    with open(out_dir / "error_surface_summary.json") as f:
        js = json.load(f)
    for key in (
        "mean_gnrmse", "by_temporal_family", "by_spatial_family",
        "by_lead_bin", "by_qrms_tercile", "by_region_rmse_K", "metric",
    ):
        assert key in js
    assert np.isfinite(js["mean_gnrmse"])
    # plot files exist (contents not asserted).
    for p in js["plots"]:
        from pathlib import Path
        assert Path(p).exists()


def test_image_fidelity_schema_and_outputs(tmp_path):
    out_dir = tmp_path / "if_out"
    rows, out = run_image_fidelity(
        data_dir=None,
        nt_img_list=(16, 64),
        ny_img_list=(8,),
        out_dir=str(out_dir),
        dense_ny=64,
        dense_nt=256,
        t_final=0.3,
    )
    assert out == out_dir
    assert rows

    with open(out_dir / "image_fidelity.csv") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == IF_FIELDS
        csv_rows = list(reader)
    # 4 temporal x 4 spatial x 1 ny x 2 nt = 32 rows.
    assert len(csv_rows) == 4 * 4 * 1 * 2

    with open(out_dir / "image_fidelity_summary.json") as f:
        js = json.load(f)
    assert js["n_rows"] == len(csv_rows)
    from pathlib import Path
    for p in js["plots"]:
        assert Path(p).exists()
    for name in (
        "overlay_pulse_temporal.png",
        "overlay_gaussian_spatial.png",
        "joint_error_heatmap.png",
    ):
        assert (out_dir / name).exists()


# --------------------------------------------------------------------------- #
# Phase 2: linearity helper math on toy fields.                               #
# --------------------------------------------------------------------------- #

def test_delta_metrics_exact_linear_match():
    # When the model's response to a variant EXACTLY equals FV's, the two delta
    # fields coincide: delta_ratio == 1 and cosine == 1.
    rng = np.random.default_rng(0)
    base = rng.standard_normal((3, 5, 5))
    resp = rng.standard_normal((3, 5, 5))  # identical model/FV response
    dm_var, df_var = base + resp, base + resp
    m = _delta_metrics(dm_var, df_var, base, base, eps=1e-2)
    assert m["delta_ratio_valid"] is True
    assert m["delta_ratio"] == pytest.approx(1.0, rel=1e-12)
    assert m["cosine"] == pytest.approx(1.0, rel=1e-9)
    assert m["model_rmse_K"] == pytest.approx(0.0, abs=1e-12)


def test_delta_metrics_underresponsive_model():
    # Model moves half as far as FV in the same direction: ratio 0.5, cosine 1.
    rng = np.random.default_rng(1)
    base = rng.standard_normal((2, 4, 4))
    resp = rng.standard_normal((2, 4, 4))
    dm_var = base + 0.5 * resp
    df_var = base + resp
    m = _delta_metrics(dm_var, df_var, base, base, eps=1e-2)
    assert m["delta_ratio"] == pytest.approx(0.5, rel=1e-12)
    assert m["cosine"] == pytest.approx(1.0, rel=1e-9)


def test_delta_metrics_eps_guard_triggers():
    # FV barely moves (delta_fv ~ 0): the ratio is flagged invalid and NaN-ed so
    # a divide-by-tiny cannot masquerade as a huge model over-response.
    base = np.ones((2, 3, 3))
    df_var = base.copy()  # FV response is exactly zero
    dm_var = base + 1.0   # model still moves
    m = _delta_metrics(dm_var, df_var, base, base, eps=1e-2)
    assert m["delta_fv_K"] == pytest.approx(0.0, abs=1e-12)
    assert m["delta_ratio_valid"] is False
    assert np.isnan(m["delta_ratio"])
    assert np.isnan(m["cosine"])


def test_linfit_exact_line():
    x = np.array([0.25, 0.5, 1.0, 1.5, 2.0])
    y = 3.0 * x + 1.0
    slope, r2 = _linfit(x, y)
    assert slope == pytest.approx(3.0, rel=1e-12)
    assert r2 == pytest.approx(1.0, rel=1e-12)


def test_linfit_degenerate_x_returns_nan():
    x = np.array([0.5, 0.5, 0.5])
    y = np.array([1.0, 2.0, 3.0])
    slope, r2 = _linfit(x, y)
    assert np.isnan(slope) and np.isnan(r2)


def test_match_indices_nearest():
    t_fv = np.linspace(0.0, 0.3, 61)      # dt = 0.005
    t_model = np.array([0.0, 0.01, 0.155, 0.3])
    idx = _match_indices(t_fv, t_model)
    assert np.allclose(t_fv[idx], [0.0, 0.01, 0.155, 0.3], atol=2.5e-3)
    assert idx[0] == 0 and idx[-1] == 60


def test_linearity_csv_fields_documented():
    # The response-match columns the summary/plots rely on must stay present.
    for col in (
        "test", "regime", "sim_id", "partner_id", "variant",
        "dev_model_K", "dev_fv_K", "model_rmse_K",
        "delta_model_K", "delta_fv_K", "delta_ratio", "delta_ratio_valid",
        "cosine", "q_rms", "Q_in",
    ):
        assert col in LIN_FIELDS


# --------------------------------------------------------------------------- #
# Phase 2: tiny fresh-FV perturbation smoke on a small grid.                   #
# --------------------------------------------------------------------------- #

def _tiny_fv_setup(num_sims: int = 2):
    from data.generate_dataset import build_base_setup
    from problems.diffusion_forcing import DiffusionForcingProblem

    setup = build_base_setup(
        num_sims=num_sims, save_stride=1, nx=16, ny=16, t_final=0.05, dt=0.005
    )
    time_cfg = dict(setup["time_cfg"])
    time_cfg["num_sims"] = num_sims
    spec = DiffusionForcingProblem()
    params = spec.sample_sim_params(
        np.random.default_rng(0), np.random.default_rng(1),
        setup["grids"], time_cfg,
    )
    return spec, setup["base_kwargs"], params


def test_fv_amplitude_perturbation_scales_field():
    spec, base_kwargs, params = _tiny_fv_setup(num_sims=1)
    t_model = np.array([0.025, 0.05])
    d1 = _fv_dev(spec, base_kwargs, params[0], {"t_model": t_model, "idx": None})
    d2 = _fv_dev(
        spec, base_kwargs, params[0], {"t_model": t_model, "idx": None}, alpha=2.0
    )
    assert d1.shape == (2, 16, 16)
    # Heat equation is linear at fixed coefficients: doubling q doubles T-T_RIGHT.
    assert _rms(d2) == pytest.approx(2.0 * _rms(d1), rel=1e-6)
    assert _rms(d2 - d1) > 1e-6  # the variant genuinely changes the FV field


def test_fv_superposition_perturbation_changes_field():
    spec, base_kwargs, params = _tiny_fv_setup(num_sims=2)
    t_model = np.array([0.025, 0.05])
    d_single = _fv_dev(spec, base_kwargs, params[0], {"t_model": t_model, "idx": None})
    d_sum = _fv_dev(
        spec, base_kwargs, params[0], {"t_model": t_model, "idx": None},
        add_params=params[1],
    )
    d_other = _fv_dev(spec, base_kwargs, params[1], {"t_model": t_model, "idx": None})
    assert _rms(d_sum - d_single) > 1e-6
    # FV is exactly additive: G(q_i+q_j)-T0 == (G(q_i)-T0)+(G(q_j)-T0).
    assert _rms(d_sum - (d_single + d_other)) == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# Phase 3: residual-operator math on an analytic field with a known residual. #
# --------------------------------------------------------------------------- #

class _Quadratic(torch.nn.Module):
    """Analytic surrogate T(x,y,t) = a x^2 + b y^2 + c t with exact residuals.

    Interior:  T_t - (T_xx + T_yy) = c - 2a - 2b   (constant everywhere).
    Top (y=1): dT/dy = 2b y = 2b.   Bottom (y=0): dT/dy = 0.
    Right (x=1): T = a + b y^2 + c t.
    The ``u`` / ``q_left`` inputs are ignored; the module only reads coords/t so
    the diagnostic's autodiff evaluators can be checked against closed form.
    """

    def __init__(self, a: float, b: float, c: float):
        super().__init__()
        self.a, self.b, self.c = float(a), float(b), float(c)

    def forward(self, u, coords, t, q_left=None):
        x = coords[..., 0:1]
        y = coords[..., 1:2]
        return self.a * x**2 + self.b * y**2 + self.c * t


def test_interior_residual_matches_closed_form():
    model = _Quadratic(a=0.5, b=-0.3, c=0.7)
    u = torch.zeros(1, 1, 1)
    dev = torch.device("cpu")
    xs = np.linspace(0.0, 1.0, 5)
    ys = np.linspace(0.0, 1.0, 4)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    xy = np.stack([gx.reshape(-1), gy.reshape(-1)], axis=-1)
    r = _interior_res_field(model, u, xy, t_val=0.2, device=dev, chunk=7)
    expected = 0.7 - 2 * 0.5 - 2 * (-0.3)  # c - 2a - 2b = 0.3
    assert r.shape == (xy.shape[0],)
    assert np.allclose(r, expected, atol=1e-4)


def test_adiabatic_wall_residual_matches_closed_form():
    model = _Quadratic(a=0.4, b=0.25, c=-0.1)
    u = torch.zeros(1, 1, 1)
    dev = torch.device("cpu")
    x_pts = np.linspace(0.0, 1.0, 9)
    r_top = _hom_wall_res_profile(model, u, x_pts, 0.15, "top", dev, chunk=5)
    r_bot = _hom_wall_res_profile(model, u, x_pts, 0.15, "bottom", dev, chunk=5)
    assert np.allclose(r_top, 2 * 0.25, atol=1e-4)   # dT/dy at y=1 = 2b
    assert np.allclose(r_bot, 0.0, atol=1e-5)         # dT/dy at y=0 = 0


def test_right_dirichlet_residual_matches_closed_form():
    model = _Quadratic(a=0.5, b=-0.3, c=0.7)
    u = torch.zeros(1, 1, 1)
    dev = torch.device("cpu")
    y_pts = np.linspace(0.0, 1.0, 7)
    t_val, t_right_tilde = 0.15, 3.3
    r = _right_dirichlet_res(model, u, y_pts, t_val, t_right_tilde, dev)
    expected = 0.5 * 1.0 + (-0.3) * y_pts**2 + 0.7 * t_val - t_right_tilde
    assert np.allclose(r, expected, atol=1e-5)


def test_snapshot_indices_excludes_ic_and_honors_times():
    t_grid = np.linspace(0.0, 0.3, 7)  # [0, .05, .1, .15, .2, .25, .3]
    # explicit times -> nearest grid index.
    assert _snapshot_indices(7, 2, [0.1, 0.28], t_grid) == [2, 6]
    # evenly-spread default never picks the IC snapshot (index 0).
    spread = _snapshot_indices(7, 2, None, t_grid)
    assert spread == [1, 6]
    assert 0 not in spread
    # requesting more snapshots than usable clamps to the available interior.
    assert _snapshot_indices(3, 10, None, np.linspace(0.0, 0.3, 3)) == [1, 2]


# --------------------------------------------------------------------------- #
# Phase 3: schema + tiny integration on the synthetic forcing checkpoint.      #
# --------------------------------------------------------------------------- #

def test_residual_adequacy_schema_and_outputs(forcing_checkpoint, tmp_path):
    ckpt_path, data_dir = forcing_checkpoint
    out_dir = tmp_path / "res_out"
    summary, out = run_residual_adequacy(
        str(ckpt_path), str(data_dir), split="train", num_sims=4,
        worst_k=2, snapshots=2, eval_stride=2, query_chunk=64,
        calib_sims=2, calib_n_r=64, calib_n_bc=32, query_batch=4,
        out_dir=str(out_dir), device="cpu",
    )
    assert out == out_dir

    with open(out_dir / "residual_adequacy_stats.csv") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == RES_FIELDS
        rows = list(reader)
    assert rows, "residual_adequacy_stats.csv has no rows"
    # worst_k=2 sims x snapshots=2 = 4 rows.
    assert len(rows) == 2 * 2
    for r in rows:
        assert np.isfinite(float(r["gnrmse"]))
        assert np.isfinite(float(r["interior_res_rms"]))
        assert np.isfinite(float(r["left_bc_res_rms"]))
        # this checkpoint hard-enforces the right Dirichlet wall.
        assert r["right_hard_enforced"] == "True"

    with open(out_dir / "residual_adequacy_summary.json") as f:
        js = json.load(f)
    for key in (
        "worst_k_sim_ids", "calibration", "aggregate", "by_time",
        "interpretation_guardrails", "csv", "plots", "timing_s",
    ):
        assert key in js
    assert len(js["worst_k_sim_ids"]) == 2
    assert np.isfinite(js["aggregate"]["mean_gnrmse"])
    for term in ("interior_res_rms", "left_bc_res_rms", "ic_res_rms"):
        assert term in js["calibration"]
    from pathlib import Path
    for p in js["plots"]:
        assert Path(p).exists()
    # one panel per worst-K sim.
    assert len(js["plots"]) == 2
