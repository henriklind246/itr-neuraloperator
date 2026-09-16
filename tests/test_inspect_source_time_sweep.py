"""Contracts for the F32 source-time sweep diagnostic."""

import numpy as np
import pytest
import yaml

from scripts import inspect_source_time_sweep as sweep
from tests.test_pub_global_field import shape_records


def write_runs(root, **kwargs):
    for benchmark, frame in shape_records(**kwargs).items():
        run = root / benchmark / "seed0"
        run.mkdir(parents=True)
        frame.to_csv(run / "test_records.csv", index=False)
        (run / "config_used.yaml").write_text(yaml.safe_dump({
            "benchmark": {"name": benchmark, "representation": "temporal_encoder"},
            "evaluation": {"rollout": {"enabled": False}}}))
    return root


def run(root, *extra):
    return sweep.main(["--runs-root", str(root), "--n-boot", "200", *extra])


def test_identical_shapes_report_agreement_and_exit_clean(tmp_path, capsys):
    # Levels differ by a large constant factor, which is not a shape difference.
    root = write_runs(tmp_path, n_sims=25, log_level=lambda s: .3 * (s == 2))
    assert run(root) == 0
    out = capsys.readouterr().out
    assert "published source index   2" in out
    assert out.count("shapes agree") == 4
    assert "Nothing contradicts" in out
    # The model predicts each pair directly from its source snapshot.
    assert "accumulat" not in out.lower()


def test_bend_at_the_published_source_time_is_flagged(tmp_path, capsys):
    root = write_runs(tmp_path, n_sims=25,
                      log_shape=lambda s, k: .5 * np.log10(k) + .5 * np.log10(k) * (s == 2))
    assert run(root) == 2
    out = capsys.readouterr().out
    assert "ATYPICAL" in out
    assert "ranks among the least typical curves" in out
    assert "'forcing'" in out


def test_bend_elsewhere_leaves_the_published_curve_alone(tmp_path, capsys):
    root = write_runs(tmp_path, n_sims=25,
                      log_shape=lambda s, k: .5 * np.log10(k) + .5 * np.log10(k) * (s == 8))
    assert run(root) == 0
    out = capsys.readouterr().out
    assert "ATYPICAL" not in out
    # s=8 only spans the two shorter windows, so the widest one cannot see it.
    assert "not spanned" not in out


def test_incomplete_cohort_is_disclosed(tmp_path, capsys):
    root = tmp_path / "runs"
    write_runs(root, n_sims=25)
    path = root / "forcing" / "seed0" / "test_records.csv"
    import pandas as pd
    frame = pd.read_csv(path)
    frame.drop(frame.index[(frame.sim_id == 4) & (frame.s == 1) & (frame.j == 5)]).to_csv(
        path, index=False)
    run(root)
    out = capsys.readouterr().out
    assert "DEGRADED incomplete_pair_cohort" in out
    assert "24 simulations" in out
    assert "excluded for missing pairs: [4]" in out


def test_window_and_fraction_flags_reach_the_reduction(tmp_path, capsys):
    root = write_runs(tmp_path, n_sims=25)
    assert run(root, "--window-fraction", ".5", "--source-fraction", ".4") == 0
    out = capsys.readouterr().out
    assert "lead windows compared    [6]" in out
    assert "published source index   4" in out


def test_missing_and_malformed_inputs_fail_without_a_traceback(tmp_path, capsys):
    assert sweep.main(["--runs-root", str(tmp_path / "absent")]) == 1
    assert "FAILED" in capsys.readouterr().err
    with pytest.raises(SystemExit) as exc:
        sweep.main(["--run", "forcing=/tmp/x"])
    assert exc.value.code == 2


def test_short_grid_is_refused_rather_than_reported(tmp_path, capsys):
    root = write_runs(tmp_path, n_sims=25, n_snap=4)
    assert run(root) == 1
    assert "too short" in capsys.readouterr().err
