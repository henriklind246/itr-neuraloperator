"""Smoke + guard tests for the inverse-problem plot layer (group ``inverse``).

These exercise the generic template in ``visual/inverse_plots.py`` against tiny
synthetic CSV + NPZ fixtures: every one of the 8 figures must render to a PNG
for forcing (with and without artifacts) and source_itr (with artifacts), the
graceful-degradation paths must not crash, malformed CSVs must raise a clear
error, and ``merge_uq_csv`` must merge by ``sim_id`` with the documented
validation. The plot layer only *reads* artifacts, so the NPZ fixtures are
written directly here rather than through the heavyweight ``invert.py`` pipeline.
"""

import csv as _csv
import warnings

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from visual import inverse_plots as ip
from visual.cli import _parse_benchmark_path_tokens


def _capture_fig(monkeypatch):
    """Patch ``ip._save_figure`` to record the rendered figure for inspection.

    The real save still runs (validating the figure renders); the captured
    Matplotlib ``Figure`` is returned in a dict under ``"fig"``.
    """
    captured: dict = {}
    real = ip._save_figure

    def fake(fig, save_path, group, name, layout="tight"):
        captured["fig"] = fig
        return real(fig, save_path, group, name, layout=layout)

    monkeypatch.setattr(ip, "_save_figure", fake)
    return captured


# ---------------------------------------------------------------------------
# Synthetic CSV / NPZ fixtures.
# ---------------------------------------------------------------------------

def _write_csv(path, columns: dict) -> None:
    n = len(next(iter(columns.values())))
    fieldnames = list(columns)
    with open(path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i in range(n):
            w.writerow({c: columns[c][i] for c in fieldnames})


def _forcing_csv_columns(n=6):
    rng = np.random.default_rng(0)
    true = np.linspace(0.1, 0.9, n)
    hat = true + rng.normal(0, 0.02, n)
    return {
        "sim_id": list(range(n)),
        "R_c_true": true.tolist(),
        "R_c_map": hat.tolist(),
        "R_c_abs_error": np.abs(hat - true).tolist(),
        "sens_R_c": (1.0 + 0.5 * true).tolist(),
        "sv_0": (1.0 + 0.5 * true).tolist(),
        "uq_sigma_eff2": np.full(n, 0.01).tolist(),
        "fno_resid": np.full(n, 5e-4).tolist(),
        "fv_resid": np.full(n, 6e-4).tolist(),
        "fno_vs_fv_resid": np.full(n, 2e-4).tolist(),
        "profile_R_c_ci_low": (hat - 0.05).tolist(),
        "profile_R_c_ci_high": (hat + 0.05).tolist(),
        "profile_R_c_covered": ["True"] * n,
        "mcmc_R_c_mean": hat.tolist(),
        "mcmc_R_c_ci_low": (hat - 0.06).tolist(),
        "mcmc_R_c_ci_high": (hat + 0.06).tolist(),
        "mcmc_R_c_covered": ["True", "False"] * (n // 2) + ["True"] * (n % 2),
    }


def _source_itr_csv_columns(n=6):
    rng = np.random.default_rng(1)
    names = ("R_base", "R_amp", "y0", "sigma")
    base = {
        "R_base": np.linspace(0.1, 0.9, n),
        "R_amp": np.linspace(0.2, 2.5, n),
        "y0": np.linspace(0.2, 0.8, n),
        "sigma": np.linspace(0.06, 0.18, n),
    }
    cols = {"sim_id": list(range(n))}
    for p in names:
        t = base[p]
        h = t + rng.normal(0, 0.02, n)
        cols[f"{p}_true"] = t.tolist()
        cols[f"{p}_hat"] = h.tolist()
        cols[f"{p}_abserr"] = np.abs(h - t).tolist()
    excess_t = base["R_amp"] * base["sigma"] * np.sqrt(np.pi)
    excess_h = excess_t + rng.normal(0, 0.01, n)
    cols["excess_int_true"] = excess_t.tolist()
    cols["excess_int_hat"] = excess_h.tolist()
    cols["excess_int_abserr"] = np.abs(excess_h - excess_t).tolist()
    for k in range(4):
        cols[f"sv_{k}"] = np.linspace(10.0 / (k + 1), 1.0 / (k + 1), n).tolist()
    cols["cond_number"] = np.linspace(5.0, 40.0, n).tolist()
    cols["ramp_sigma_alignment"] = np.linspace(0.6, 0.95, n).tolist()
    cols["uq_sigma_eff2"] = np.full(n, 0.01).tolist()
    cols["fno_resid"] = np.full(n, 5e-4).tolist()
    cols["fv_resid"] = np.full(n, 6e-4).tolist()
    cols["fno_vs_fv_resid"] = np.full(n, 2e-4).tolist()
    cols["profile_R_amp_ci_low"] = (base["R_amp"] - 0.3).tolist()
    cols["profile_R_amp_ci_high"] = (base["R_amp"] + 0.3).tolist()
    cols["profile_R_amp_covered"] = ["True"] * n
    cols["profile_excess_ci_low"] = (excess_h - 0.1).tolist()
    cols["profile_excess_ci_high"] = (excess_h + 0.1).tolist()
    cols["mcmc_excess_mean"] = excess_h.tolist()
    cols["mcmc_excess_ci_low"] = (excess_h - 0.12).tolist()
    cols["mcmc_excess_ci_high"] = (excess_h + 0.12).tolist()
    cols["mcmc_excess_covered"] = ["True", "False"] * (n // 2) + ["True"] * (n % 2)
    return cols


def _write_forcing_artifact(art_dir, sid):
    rng = np.random.default_rng(100 + sid)
    grid = np.linspace(0.2, 0.8, 9)
    nll = 0.5 * ((grid - 0.5) / 0.1) ** 2
    np.savez_compressed(
        art_dir / f"sim_{sid:05d}.npz",
        sim_id=np.int64(sid),
        benchmark=np.str_("forcing"),
        param_scales=np.array([0.95]),
        observation_jacobian=rng.normal(size=(12, 1)),
        profile_grid=grid,
        profile_nll=nll,
        profile_nll_min=np.float64(nll.min()),
        profile_threshold=np.float64(1.92),
        profile_param_name=np.str_("R_c"),
        mcmc_theta_samples=rng.normal(0.5, 0.05, size=(60, 1)),
    )


def _write_source_itr_artifact(art_dir, sid):
    rng = np.random.default_rng(200 + sid)
    grid = np.linspace(0.5, 2.0, 9)
    nll = 0.5 * ((grid - 1.2) / 0.3) ** 2
    samples = rng.normal([0.5, 1.2, 0.5, 0.12], [0.05, 0.3, 0.05, 0.02],
                         size=(80, 4))
    excess = samples[:, 1] * samples[:, 3] * np.sqrt(np.pi)
    np.savez_compressed(
        art_dir / f"sim_{sid:05d}.npz",
        sim_id=np.int64(sid),
        benchmark=np.str_("source_itr"),
        param_scales=np.array([0.95, 2.95, 0.8, 0.15]),
        observation_jacobian=rng.normal(size=(20, 4)),
        profile_grid=grid,
        profile_nll=nll,
        profile_nll_min=np.float64(nll.min()),
        profile_threshold=np.float64(1.92),
        profile_param_name=np.str_("R_amp"),
        profile_excess_int=excess[:9],
        mcmc_theta_samples=samples,
        mcmc_excess_int=excess,
    )


def _write_forcing_itr_artifact(art_dir, sid):
    _write_source_itr_artifact(art_dir, sid)
    path = art_dir / f"sim_{sid:05d}.npz"
    with np.load(path, allow_pickle=True) as stored:
        payload = {key: stored[key] for key in stored.files}
    payload["benchmark"] = np.str_("forcing_itr")
    np.savez_compressed(path, **payload)


@pytest.fixture
def forcing_csv(tmp_path):
    path = tmp_path / "inverse_forcing.csv"
    _write_csv(path, _forcing_csv_columns())
    return path


@pytest.fixture
def source_itr_csv(tmp_path):
    path = tmp_path / "inverse_source_itr.csv"
    _write_csv(path, _source_itr_csv_columns())
    return path


@pytest.fixture
def forcing_artifacts(tmp_path):
    d = tmp_path / "art_forcing"
    d.mkdir()
    for sid in range(6):
        _write_forcing_artifact(d, sid)
    return d


@pytest.fixture
def source_itr_artifacts(tmp_path):
    d = tmp_path / "art_source_itr"
    d.mkdir()
    for sid in range(6):
        _write_source_itr_artifact(d, sid)
    return d


@pytest.fixture
def forcing_itr_csv(tmp_path):
    path = tmp_path / "inverse_forcing_itr.csv"
    _write_csv(path, _source_itr_csv_columns())
    return path


@pytest.fixture
def forcing_itr_artifacts(tmp_path):
    d = tmp_path / "art_forcing_itr"
    d.mkdir()
    for sid in range(6):
        _write_forcing_itr_artifact(d, sid)
    return d


# ---------------------------------------------------------------------------
# Render smoke tests: every figure produces a PNG.
# ---------------------------------------------------------------------------

def test_forcing_four_plots_render_with_artifacts(tmp_path, forcing_csv,
                                                  forcing_artifacts):
    spec = ip.spec_for("forcing")
    out = tmp_path / "out"
    p1 = ip.plot_parameter_recovery(forcing_csv, spec,
                                    save_path=out / "f_pr.png")
    p2 = ip.plot_identifiability(forcing_csv, spec,
                                 artifact_dir=forcing_artifacts,
                                 save_path=out / "f_id.png")
    p3 = ip.plot_surrogate_fidelity(forcing_csv, spec,
                                    save_path=out / "f_sf.png")
    p4 = ip.plot_uncertainty(forcing_csv, spec,
                             artifact_dir=forcing_artifacts,
                             save_path=out / "f_uq.png")
    for p in (p1, p2, p3, p4):
        assert p.exists() and p.stat().st_size > 0


def test_forcing_plots_render_without_artifacts(tmp_path, forcing_csv):
    spec = ip.spec_for("forcing")
    out = tmp_path / "out"
    # forcing identifiability is CSV-only; uncertainty degrades to the
    # inclusion-rate strip with "no artifact" panels.
    p2 = ip.plot_identifiability(forcing_csv, spec, artifact_dir=None,
                                 save_path=out / "f_id.png")
    p4 = ip.plot_uncertainty(forcing_csv, spec, artifact_dir=None,
                             save_path=out / "f_uq.png")
    assert p2.exists() and p4.exists()


def test_source_itr_four_plots_render_with_artifacts(tmp_path, source_itr_csv,
                                                     source_itr_artifacts):
    spec = ip.spec_for("source_itr")
    out = tmp_path / "out"
    p1 = ip.plot_parameter_recovery(source_itr_csv, spec,
                                    save_path=out / "s_pr.png")
    p2 = ip.plot_identifiability(source_itr_csv, spec,
                                 artifact_dir=source_itr_artifacts,
                                 save_path=out / "s_id.png")
    p3 = ip.plot_surrogate_fidelity(source_itr_csv, spec,
                                    save_path=out / "s_sf.png")
    p4 = ip.plot_uncertainty(source_itr_csv, spec,
                             artifact_dir=source_itr_artifacts,
                             save_path=out / "s_uq.png")
    for p in (p1, p2, p3, p4):
        assert p.exists() and p.stat().st_size > 0


def test_forcing_itr_four_plots_render_with_artifacts(
    tmp_path, forcing_itr_csv, forcing_itr_artifacts
):
    spec = ip.spec_for("forcing_itr")
    out = tmp_path / "forcing_itr_out"
    paths = (
        ip.plot_parameter_recovery(
            forcing_itr_csv, spec, save_path=out / "parameter_recovery.png"
        ),
        ip.plot_identifiability(
            forcing_itr_csv, spec, artifact_dir=forcing_itr_artifacts,
            save_path=out / "identifiability.png",
        ),
        ip.plot_surrogate_fidelity(
            forcing_itr_csv, spec, save_path=out / "surrogate_fidelity.png"
        ),
        ip.plot_uncertainty(
            forcing_itr_csv, spec, artifact_dir=forcing_itr_artifacts,
            save_path=out / "uncertainty.png",
        ),
    )
    for path in paths:
        assert path.exists() and path.stat().st_size > 0


def test_source_itr_identifiability_falls_back_without_artifacts(tmp_path,
                                                                 source_itr_csv):
    spec = ip.spec_for("source_itr")
    out = tmp_path / "out"
    # No Jacobian artifacts → raw-CSV singular-value spectrum, no ridge.
    p = ip.plot_identifiability(source_itr_csv, spec, artifact_dir=None,
                                save_path=out / "s_id.png")
    assert p.exists() and p.stat().st_size > 0


# ---------------------------------------------------------------------------
# Guard cases: malformed CSV raises a clear error.
# ---------------------------------------------------------------------------

def test_missing_required_column_raises(tmp_path):
    spec = ip.spec_for("source_itr")
    cols = _source_itr_csv_columns()
    del cols["sv_2"]  # drop a structural identifiability column
    path = tmp_path / "bad.csv"
    _write_csv(path, cols)
    with pytest.raises(ValueError, match="missing required column"):
        ip.plot_identifiability(path, spec, artifact_dir=None,
                                save_path=tmp_path / "x.png")


def test_parameter_recovery_missing_column_raises(tmp_path):
    spec = ip.spec_for("forcing")
    cols = _forcing_csv_columns()
    del cols["R_c_map"]
    path = tmp_path / "bad.csv"
    _write_csv(path, cols)
    with pytest.raises(ValueError, match="missing required column"):
        ip.plot_parameter_recovery(path, spec, save_path=tmp_path / "x.png")


def test_missing_csv_file_raises(tmp_path):
    spec = ip.spec_for("forcing")
    with pytest.raises(FileNotFoundError, match="inverse CSV not found"):
        ip.plot_surrogate_fidelity(tmp_path / "nope.csv", spec,
                                   save_path=tmp_path / "x.png")


# ---------------------------------------------------------------------------
# Guard: benchmark=path token parsing rejects mismatches.
# ---------------------------------------------------------------------------

def test_benchmark_path_tokens_parse_and_reject():
    parsed = _parse_benchmark_path_tokens(
        ["forcing=a.csv", "source_itr=b.csv"], "--inverse-csv")
    assert parsed == {"forcing": "a.csv", "source_itr": "b.csv"}

    assert _parse_benchmark_path_tokens(None, "--inverse-csv") == {}

    with pytest.raises(ValueError, match="benchmark=path"):
        _parse_benchmark_path_tokens(["no_equals_sign"], "--inverse-csv")

    with pytest.raises(ValueError, match="unknown benchmark"):
        _parse_benchmark_path_tokens(["bogus=path.csv"], "--inverse-csv")

    with pytest.raises(ValueError, match="benchmark=path"):
        _parse_benchmark_path_tokens(["=path.csv"], "--inverse-csv")


# ---------------------------------------------------------------------------
# merge_uq_csv: merge by sim_id, validation, n_total / n_UQ.
# ---------------------------------------------------------------------------

def _write_master_uq(tmp_path, n_master=6, uq_ids=(1, 3, 5)):
    master_cols = {
        "sim_id": list(range(n_master)),
        "benchmark": ["forcing"] * n_master,
        "R_c_true": np.linspace(0.1, 0.9, n_master).tolist(),
        "R_c_map": np.linspace(0.1, 0.9, n_master).tolist(),
    }
    master = tmp_path / "master.csv"
    _write_csv(master, master_cols)

    true_lookup = {i: master_cols["R_c_true"][i] for i in range(n_master)}
    uq_cols = {
        "sim_id": list(uq_ids),
        "benchmark": ["forcing"] * len(uq_ids),
        "R_c_true": [true_lookup[i] for i in uq_ids],
        "mcmc_R_c_mean": [0.5] * len(uq_ids),
        "mcmc_R_c_covered": ["True"] * len(uq_ids),
        "profile_R_c_ci_low": [0.4] * len(uq_ids),
    }
    uq = tmp_path / "uq.csv"
    _write_csv(uq, uq_cols)
    return master, uq


def test_merge_uq_csv_merges_by_sim_id(tmp_path):
    master, uq = _write_master_uq(tmp_path)
    merged = ip.merge_uq_csv(master, uq)
    assert len(merged) == 6
    # UQ columns present, filled only on the 3 subset rows.
    assert "mcmc_R_c_mean" in merged.columns
    filled = merged.set_index("sim_id")["mcmc_R_c_mean"]
    assert filled.notna().sum() == 3
    assert {1, 3, 5} == set(filled[filled.notna()].index.tolist())
    assert filled.isna().sum() == 3


def test_merge_uq_csv_duplicate_sim_id_raises(tmp_path):
    master, uq = _write_master_uq(tmp_path)
    # Append a duplicate sim_id row to the master.
    dup = tmp_path / "master_dup.csv"
    cols = {
        "sim_id": [0, 1, 2, 2],
        "benchmark": ["forcing"] * 4,
        "R_c_true": [0.1, 0.2, 0.3, 0.3],
        "R_c_map": [0.1, 0.2, 0.3, 0.3],
    }
    _write_csv(dup, cols)
    with pytest.raises(ValueError, match="duplicate sim_id"):
        ip.merge_uq_csv(dup, uq)


def test_merge_uq_csv_true_mismatch_raises(tmp_path):
    master, _ = _write_master_uq(tmp_path)
    bad_uq = tmp_path / "uq_bad.csv"
    cols = {
        "sim_id": [1, 3],
        "benchmark": ["forcing", "forcing"],
        "R_c_true": [0.999, 0.888],   # disagree with master's ground truth
        "mcmc_R_c_mean": [0.5, 0.5],
    }
    _write_csv(bad_uq, cols)
    with pytest.raises(ValueError, match="ground-truth column"):
        ip.merge_uq_csv(master, bad_uq)


# ---------------------------------------------------------------------------
# Wilson CI numerics (no SciPy dependency).
# ---------------------------------------------------------------------------

def test_wilson_ci_known_value_and_edges():
    # 8/10 → Wilson 95% ~ (0.4902, 0.9433).
    lo, hi = ip._wilson_ci(8, 10)
    assert abs(lo - 0.4902) < 1e-3
    assert abs(hi - 0.9433) < 1e-3
    # n == 0 → (nan, nan).
    lo0, hi0 = ip._wilson_ci(0, 0)
    assert np.isnan(lo0) and np.isnan(hi0)
    # k clipped to [0, n]: k > n behaves like k == n; k < 0 like k == 0.
    assert ip._wilson_ci(15, 10) == ip._wilson_ci(10, 10)
    assert ip._wilson_ci(-3, 10) == ip._wilson_ci(0, 10)


def test_inclusion_yerr_skipped_when_no_data():
    assert ip._inclusion_yerr(0.0, 0) is None
    yerr = ip._inclusion_yerr(0.75, 4)
    assert yerr is not None and yerr.shape == (2, 1)
    assert np.all(yerr >= 0.0)


# ---------------------------------------------------------------------------
# Fig 1: median-|error| label + trade-off tag (spec-driven).
# ---------------------------------------------------------------------------

def test_scatter_parity_labels_median_abs_error_not_mae():
    fig, ax = plt.subplots()
    true = np.linspace(0.0, 1.0, 5)
    ip._scatter_parity(ax, true, true + 0.05, (0.0, 1.0), "R_c", "#000000")
    joined = "\n".join(t.get_text() for t in ax.texts)
    assert "med|err|" in joined
    assert "MAE" not in joined
    plt.close(fig)


def test_recovery_trade_off_tag_only_on_flagged_panels(monkeypatch, tmp_path,
                                                       source_itr_csv):
    cap = _capture_fig(monkeypatch)
    ip.plot_parameter_recovery(source_itr_csv, ip.spec_for("source_itr"),
                               save_path=tmp_path / "s_pr.png")
    fig = cap["fig"]
    tag = "expected parameter trade-off"
    tagged = {ax.get_title(): any(tag in t.get_text() for t in ax.texts)
              for ax in fig.axes}
    # R_amp and sigma trade off (flagged); R_base and y0 do not.
    assert tagged.get("R_amp") is True
    assert tagged.get("sigma") is True
    assert tagged.get("R_base") is False
    assert tagged.get("y0") is False
    plt.close(fig)


def test_recovery_subtitle_present_for_source_itr(monkeypatch, tmp_path,
                                                  source_itr_csv):
    cap = _capture_fig(monkeypatch)
    ip.plot_parameter_recovery(source_itr_csv, ip.spec_for("source_itr"),
                               save_path=tmp_path / "s_pr.png")
    fig = cap["fig"]
    assert "trade off" in fig._suptitle.get_text()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 2: ridge axis limits + centered direction segment.
# ---------------------------------------------------------------------------

def test_ridge_limits_robust_to_collapsed_coordinate():
    flat = np.column_stack([np.full(50, 0.3), np.full(50, 0.3)])
    (xlo, xhi), (ylo, yhi) = ip._ridge_axis_limits(flat)
    assert xhi - xlo >= 0.1 and yhi - ylo >= 0.1
    assert 0.0 <= xlo and xhi <= 1.0 and 0.0 <= ylo and yhi <= 1.0


def test_centered_direction_segment_within_limits():
    rng = np.random.default_rng(0)
    pts = np.clip(rng.normal([0.5, 0.5], [0.08, 0.02], size=(200, 2)), 0.0, 1.0)
    xlim, ylim = ip._ridge_axis_limits(pts)
    center = np.median(pts, axis=0)
    seg = ip._centered_direction_segment(center, np.array([0.7, -0.7]), pts,
                                         xlim, ylim)
    assert seg is not None
    for (x, y) in seg:
        assert xlim[0] - 1e-9 <= x <= xlim[1] + 1e-9
        assert ylim[0] - 1e-9 <= y <= ylim[1] + 1e-9
    # Segment is centered on the centroid.
    mid = np.mean(np.array(seg), axis=0)
    assert np.allclose(mid, center, atol=1e-9)


def test_centered_direction_segment_degenerate_direction_returns_none():
    pts = np.array([[0.4, 0.4], [0.6, 0.6]])
    xlim, ylim = ip._ridge_axis_limits(pts)
    assert ip._centered_direction_segment(np.array([0.5, 0.5]),
                                          np.array([0.0, 0.0]), pts,
                                          xlim, ylim) is None


def test_source_itr_ridge_title_is_spec_derived(monkeypatch, tmp_path,
                                                source_itr_csv,
                                                source_itr_artifacts):
    cap = _capture_fig(monkeypatch)
    ip.plot_identifiability(source_itr_csv, ip.spec_for("source_itr"),
                            artifact_dir=source_itr_artifacts,
                            save_path=tmp_path / "s_id.png")
    fig = cap["fig"]
    titles = " ".join(ax.get_title() for ax in fig.axes)
    assert "(R_amp, sigma) posterior ridge" in titles
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 3: conditional log scale + guarded Spearman annotation.
# ---------------------------------------------------------------------------

def test_surrogate_fidelity_log_scale_when_all_positive(monkeypatch, tmp_path,
                                                        forcing_csv):
    cap = _capture_fig(monkeypatch)
    ip.plot_surrogate_fidelity(forcing_csv, ip.spec_for("forcing"),
                               save_path=tmp_path / "f_sf.png")
    fig = cap["fig"]
    assert fig.axes[0].get_yscale() == "log"
    plt.close(fig)


def test_surrogate_fidelity_linear_fallback_on_zero_rms(monkeypatch, tmp_path):
    cols = _forcing_csv_columns()
    fno = np.full(len(cols["sim_id"]), 5e-4)
    fno[0] = 0.0  # mixed zero + positive forces linear fallback
    cols["fno_resid"] = fno.tolist()
    path = tmp_path / "zero_rms.csv"
    _write_csv(path, cols)
    cap = _capture_fig(monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ip.plot_surrogate_fidelity(path, ip.spec_for("forcing"),
                                   save_path=tmp_path / "f_sf.png")
    fig = cap["fig"]
    assert fig.axes[0].get_yscale() == "linear"
    plt.close(fig)


def test_spearman_omitted_on_constant_input_no_warning():
    fig, ax = plt.subplots()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ip._annotate_spearman(ax, np.array([1.0, 1.0, 1.0, 1.0]),
                              np.array([1.0, 2.0, 3.0, 4.0]))
    assert not any("Spearman" in t.get_text() for t in ax.texts)
    plt.close(fig)


def test_spearman_present_when_well_defined():
    fig, ax = plt.subplots()
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([2.0, 1.0, 4.0, 3.0, 6.0])
    ip._annotate_spearman(ax, x, y)
    assert any("Spearman" in t.get_text() for t in ax.texts)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 4: grouped inclusion vs scalar single-series outcome.
# ---------------------------------------------------------------------------

def test_inclusion_grouped_renders_both_estimands_with_n():
    fig, ax = plt.subplots()
    a = np.array([1.0, 1.0, 0.0, 1.0])   # n = 4
    b = np.array([1.0, 0.0, 1.0])        # n = 3
    ip._inclusion_grouped(ax, [("profile", a), ("MCMC", b)], "#000000")
    assert len(ax.patches) == 2
    xlabels = [t.get_text() for t in ax.get_xticklabels()]
    assert "profile" in xlabels and "MCMC" in xlabels
    txt = "\n".join(t.get_text() for t in ax.texts)
    assert "n=4" in txt and "n=3" in txt
    plt.close(fig)


def test_forcing_uncertainty_single_inclusion_category(monkeypatch, tmp_path,
                                                       forcing_csv,
                                                       forcing_artifacts):
    cap = _capture_fig(monkeypatch)
    ip.plot_uncertainty(forcing_csv, ip.spec_for("forcing"),
                        artifact_dir=forcing_artifacts,
                        save_path=tmp_path / "f_uq.png")
    fig = cap["fig"]
    xlabels = []
    for ax in fig.axes:
        xlabels += [t.get_text() for t in ax.get_xticklabels()]
    # No grouped "profile — …" / "MCMC — …" estimand labels for the scalar path.
    grouped = [s for s in xlabels
               if s.startswith("profile \u2014") or s.startswith("MCMC \u2014")]
    assert grouped == []
    assert any("MCMC credible-interval inclusion" in s for s in xlabels)
    plt.close(fig)


def test_forcing_identifiability_has_no_ridge_panel(monkeypatch, tmp_path,
                                                    forcing_csv,
                                                    forcing_artifacts):
    cap = _capture_fig(monkeypatch)
    ip.plot_identifiability(forcing_csv, ip.spec_for("forcing"),
                            artifact_dir=forcing_artifacts,
                            save_path=tmp_path / "f_id.png")
    fig = cap["fig"]
    titles = " ".join(ax.get_title() for ax in fig.axes)
    assert "posterior ridge" not in titles
    plt.close(fig)
