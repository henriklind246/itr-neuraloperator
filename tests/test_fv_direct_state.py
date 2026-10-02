import copy
import json
from pathlib import Path

import pytest
import torch
import yaml

import scripts.diagnose_fv_direct_state as diagnostic


def config():
    return yaml.safe_load((Path(__file__).parents[1] / "conf/config.yaml").read_text())[
        "physics_test"]["phase0"]


def test_gate_does_not_pass_when_both_controls_learn_comparably():
    cfg = config()
    result = dict(final_rmse_normalized=1e-5, steps_to_tolerance=100)
    case = dict(grid_size=max(cfg["grid_sizes"]), solver_floor=1e-14,
                raw=copy.deepcopy(result), exact=copy.deepcopy(result))
    assert not diagnostic.gate_result([case], cfg)["passed"]
    case["raw"] = dict(final_rmse_normalized=0.1, steps_to_tolerance=None)
    assert diagnostic.gate_result([case], cfg)["passed"]
    case["exact"]["steps_to_tolerance"] = None
    assert not diagnostic.gate_result([case], cfg)["passed"]


def test_probe_suite_is_frozen_and_has_matching_perturbation_scale():
    fields = diagnostic.probe_fields(10, seed=1701, rms=0.2)
    repeat = diagnostic.probe_fields(10, seed=1701, rms=0.2)
    assert tuple(fields) == diagnostic.PROBES
    for name, field in fields.items():
        assert field.shape == (9, 10)
        torch.testing.assert_close(field, repeat[name], rtol=0, atol=0)
        torch.testing.assert_close(field.square().mean().sqrt(),
                                   torch.tensor(0.2, dtype=torch.float64))


def test_mg_gate_uses_mg_results_even_when_exact_passes():
    cfg = config()
    case = dict(grid_size=100, solver_floor=1e-14,
                raw=dict(final_rmse_normalized=.1, steps_to_tolerance=None),
                exact=dict(final_rmse_normalized=1e-5, steps_to_tolerance=100),
                mg=dict(final_rmse_normalized=.1, steps_to_tolerance=None))
    assert diagnostic.gate_result([case], cfg)['passed']
    assert not diagnostic.gate_result([case], cfg, 'mg')['passed']
    case['mg'] = dict(final_rmse_normalized=1e-4, steps_to_tolerance=120)
    assert diagnostic.gate_result([case], cfg, 'mg')['passed']


def test_first_interval_gate_stops_later_reference_cases(tmp_path, monkeypatch):
    cfg = config()
    cfg.update(grid_sizes=[4], updates=2)
    called = []

    def run_cases(problems, config, writer, stream, *, time_n):
        called.append(time_n)
        return [dict(grid_size=4, solver_floor=1e-14,
                     raw=dict(final_rmse_normalized=0.1, steps_to_tolerance=None),
                     exact=dict(final_rmse_normalized=0.1, steps_to_tolerance=None))]

    monkeypatch.setattr(diagnostic, "run_cases", run_cases)
    output = tmp_path / "screen"
    summary = diagnostic.run_gate(cfg, output)
    assert called == [0]
    assert summary["algebra_passed"]
    assert not summary["passed"] and summary["later_time_gate"]["skipped"]
    frozen = yaml.safe_load((output / "config_used.yaml").read_text())
    assert frozen["config"]["updates"] == 2
    assert frozen["protocol_sha256"] == summary["protocol_sha256"]
    assert json.loads((output / "source_hashes.json").read_text())
    with pytest.raises(FileExistsError):
        diagnostic.run_gate(cfg, output)
