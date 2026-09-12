import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import run_inverse_runtime_comparison as runtime
from scripts.inverse_adapters import ForcingItrSinAdapter


def _config(benchmark=runtime.BENCHMARK, representation=runtime.REPRESENTATION):
    return {
        "benchmark": {"name": benchmark, "representation": representation},
        "data": {"nx": 24, "ny": 20, "save_stride": 2},
    }


def _outcome(theta=(0.4, 0.8), loss=0.01):
    return runtime.MethodOutcome(
        theta_hat=torch.tensor(theta, dtype=torch.float32),
        loss=loss,
        nfev=12,
        nit=7,
        converged=True,
        termination_reason="done",
    )


def _case_rows():
    rows = []
    runtimes = {
        "fno_adam_lbfgs": [1.0, 2.0, 3.0],
        "fno_nelder_mead": [2.0, 4.0, 6.0],
        "fv_nelder_mead": [10.0, 20.0, 30.0],
    }
    for method in runtime.METHODS:
        for sim_id in range(runtime.N_CASES):
            rows.append(
                {
                    "method": method,
                    "forward_model": "FNO" if method.startswith("fno") else "finite_volume",
                    "optimizer": "test",
                    "sim_id": sim_id,
                    "runtime_s": runtimes[method][sim_id],
                    "objective_half_mse": 0.01,
                    "nfev": 10,
                    "nit": 5,
                    "converged": True,
                    "termination_reason": "done",
                    "R_base_hat": 0.4,
                    "R_base_true": 0.5,
                    "R_base_abs_error": [0.1, 0.2, 0.3][sim_id],
                    "A_hat": 0.6, "A_true": 0.5,
                    "A_abs_error": [0.4, 0.2, 0.3][sim_id],
                    "R_base_rel_error_pct": [20.0, 40.0, 60.0][sim_id],
                    "A_rel_error_pct": [80.0, 40.0, 60.0][sim_id],
                }
            )
    return rows


@pytest.mark.parametrize(
    ("benchmark", "representation"),
    [
        ("forcing", "temporal_encoder"),
        ("forcing_itr", "temporal_encoder"),
        ("source_itr_sin", "temporal_encoder"),
        ("forcing_itr_sin", "bins"),
    ],
)
def test_checkpoint_validation_is_exact(benchmark, representation):
    with pytest.raises(ValueError, match="requires forcing_itr_sin/temporal_encoder"):
        runtime.validate_checkpoint_config(_config(benchmark, representation))

    runtime.validate_checkpoint_config(_config())


def test_parser_defaults_to_normalized_noise_std_point_zero_one():
    args = runtime._build_parser().parse_args(["--checkpoint", "model.pt"])
    assert args.noise_std == pytest.approx(0.01)
    assert runtime.N_CASES == 3
    assert runtime.N_STARTS == 3
    assert runtime.SENSOR_COUNT == 8


def test_ground_truth_generation_is_three_fv_forcing_itr_sin_cases(tmp_path):
    calls = {}

    def fake_generate(**kwargs):
        calls.update(kwargs)

    runtime.generate_ground_truth_dataset(
        _config(), tmp_path / "inversion_data", generate_fn=fake_generate
    )

    assert calls == {
        "num_sims": 3,
        "save_dir": tmp_path / "inversion_data",
        "benchmark": "forcing_itr_sin",
        "nx": 24,
        "ny": 20,
        "save_stride": 2,
        "rng_seed": 1,
    }


def test_lhs_starts_are_deterministic_shared_and_physically_bounded():
    adapter = ForcingItrSinAdapter()
    starts_a = runtime.lhs_starts(adapter)
    starts_b = runtime.lhs_starts(adapter)

    np.testing.assert_array_equal(starts_a, starts_b)
    assert starts_a.shape == (3, 2)
    theta = adapter.theta_from_unconstrained(torch.from_numpy(starts_a))
    assert torch.all((theta[:, 0] >= 0.05) & (theta[:, 0] <= 1.0))
    assert torch.all(theta[:, 1] >= 0.0)
    assert torch.all(theta[:, 1] <= 3.0 - theta[:, 0])


def test_nelder_mead_multistart_selects_best_and_reports_diagnostics(monkeypatch):
    adapter = ForcingItrSinAdapter()
    starts = runtime.lhs_starts(adapter)
    target = np.array([0.15, -0.25])
    monkeypatch.setattr(runtime, "NM_MAXITER", 200)

    result = runtime.run_multistart_nelder_mead(
        adapter,
        starts,
        lambda u: float(np.sum((u - target) ** 2)),
    )

    assert result.loss < 1e-6
    assert result.nfev >= result.nit > 0
    assert result.converged
    assert result.termination_reason
    assert 0.05 <= float(result.theta_hat[0]) <= 1.0
    assert 0.0 <= float(result.theta_hat[1]) <= 3.0 - float(result.theta_hat[0])

    monkeypatch.setattr(runtime, "NM_MAXITER", 1)
    capped = runtime.run_multistart_nelder_mead(
        adapter,
        starts,
        lambda u: float(np.sum((u - target) ** 2)),
    )
    assert not capped.converged
    assert "maximum" in capped.termination_reason.lower()


def test_adam_lbfgs_counts_calls_and_labels_derived_status(monkeypatch):
    adapter = ForcingItrSinAdapter()
    starts = runtime.lhs_starts(adapter)
    monkeypatch.setattr(runtime, "ADAM_STEPS", 2)
    monkeypatch.setattr(runtime, "LBFGS_STEPS", 2)

    def quadratic_loss(_model, _obs, u, _reg_weight, _adapter):
        return 0.5 * torch.sum((u - 0.1) ** 2)

    monkeypatch.setattr(runtime, "_data_loss", quadratic_loss)
    result = runtime.run_fno_adam_lbfgs(
        None, None, adapter, starts, device="cpu"
    )

    assert np.isfinite(result.loss)
    assert result.nfev >= runtime.N_STARTS * (runtime.ADAM_STEPS + 1)
    assert result.nit >= runtime.N_STARTS * runtime.ADAM_STEPS
    assert result.termination_reason.startswith("derived: torch LBFGS")


def test_compare_case_shares_observation_and_starts_across_methods(monkeypatch):
    adapter = ForcingItrSinAdapter()
    starts = runtime.lhs_starts(adapter)
    obs = SimpleNamespace(
        sid=2,
        theta_true=torch.tensor([0.5, 0.7]),
        y_grid=torch.linspace(0.0, 1.0, 8),
        targets=torch.randn(3, 4, 8, 1),
    )
    seen = []

    def fake_fno(_model, actual_obs, _adapter, actual_starts, **_kwargs):
        seen.append((id(actual_obs), id(actual_obs.targets), id(actual_starts)))
        return _outcome()

    def fake_fv(_ds, _base, actual_obs, _adapter, actual_starts, **_kwargs):
        seen.append((id(actual_obs), id(actual_obs.targets), id(actual_starts)))
        return _outcome()

    monkeypatch.setattr(runtime, "run_fno_adam_lbfgs", fake_fno)
    monkeypatch.setattr(runtime, "run_fno_nelder_mead", fake_fno)
    monkeypatch.setattr(runtime, "run_fv_nelder_mead", fake_fv)

    rows = runtime.compare_case(
        case_index=0,
        obs=obs,
        model=None,
        adapter=adapter,
        starts_u=starts,
        ds=None,
        fv_base_kwargs={},
        device="cpu",
        mu_global=300.0,
        sigma_global=2.0,
    )

    assert len(rows) == 3
    assert len(set(seen)) == 1
    assert tuple(row["method"] for row in rows) == runtime.method_order(0)
    assert runtime.method_order(1)[0] == "fno_nelder_mead"
    assert runtime.method_order(2)[0] == "fv_nelder_mead"


def test_summary_uses_paired_medians_and_fv_speedup_baseline():
    summary = runtime.summarize_rows(_case_rows())
    by_method = {row["method"]: row for row in summary}

    assert len(summary) == 3
    assert by_method["fno_adam_lbfgs"]["median_runtime_s"] == pytest.approx(2.0)
    assert by_method["fno_adam_lbfgs"][
        "speedup_vs_fv_nelder_mead"
    ] == pytest.approx(10.0)
    assert by_method["fno_nelder_mead"][
        "speedup_vs_fv_nelder_mead"
    ] == pytest.approx(5.0)
    assert by_method["fv_nelder_mead"][
        "speedup_vs_fv_nelder_mead"
    ] == pytest.approx(1.0)
    assert by_method["fv_nelder_mead"]["median_A_abs_error"] == pytest.approx(0.3)
    assert by_method["fv_nelder_mead"][
        "median_R_base_abs_error"
    ] == pytest.approx(0.2)


def test_outputs_have_compact_schemas_and_precise_noise_protocol(tmp_path):
    case_rows = _case_rows()
    summary_rows = runtime.summarize_rows(case_rows)
    case_path = tmp_path / "inverse_runtime_cases.csv"
    summary_path = tmp_path / "inverse_runtime_summary.csv"
    protocol_path = tmp_path / "inverse_runtime_protocol.json"

    runtime._write_csv(case_path, runtime.CASE_COLUMNS, case_rows)
    runtime._write_csv(summary_path, runtime.SUMMARY_COLUMNS, summary_rows)
    ds = SimpleNamespace(
        Nx=24,
        Ny=20,
        t_grid=np.array([0.0, 0.01, 0.02]),
        dt=0.005,
    )
    protocol = runtime._protocol(
        checkpoint=tmp_path / "model.pt",
        checkpoint_fingerprint="abc123",
        data_dir=tmp_path / "inversion_data",
        device="mps",
        noise_std=0.01,
        sigma_global=4.0,
        ds=ds,
        requested_y=np.linspace(0.0, 1.0, 8),
        observation_times=np.array(runtime.PAPER_OBSERVATION_TIMES),
        starts_u=runtime.lhs_starts(ForcingItrSinAdapter()),
    )
    protocol_path.write_text(json.dumps(protocol))

    with case_path.open(newline="") as handle:
        stored_cases = list(csv.DictReader(handle))
    with summary_path.open(newline="") as handle:
        stored_summary = list(csv.DictReader(handle))
    stored_protocol = json.loads(protocol_path.read_text())

    assert len(stored_cases) == 9
    assert tuple(stored_cases[0]) == runtime.CASE_COLUMNS
    assert all(np.isfinite(float(row["runtime_s"])) for row in stored_cases)
    assert all(row["termination_reason"] for row in stored_cases)
    assert len(stored_summary) == 3
    assert tuple(stored_summary[0]) == runtime.SUMMARY_COLUMNS
    assert stored_protocol["cases"] == {"count": 3, "dataset_seed": 1}
    assert stored_protocol["observations"]["noise_std_normalized"] == pytest.approx(0.01)
    assert stored_protocol["observations"]["noise_std_K"] == pytest.approx(0.04)
    assert stored_protocol["observations"]["temperature_scale"] == "checkpoint sigma_global"
    assert stored_protocol["observations"]["sensor_count"] == 8
