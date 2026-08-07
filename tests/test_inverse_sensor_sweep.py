import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import run_inverse_sensor_sweep as sweep


def _checkpoint(path: Path, benchmark: str = "forcing") -> Path:
    torch.save(
        {
            "conf": {
                "benchmark": {
                    "name": benchmark,
                    "representation": "temporal_encoder",
                },
                "data": {"train_split": 0.7, "val_split": 0.15},
            }
        },
        path,
    )
    return path


def _data_dir(path: Path, num_sims: int = 10) -> Path:
    path.mkdir()
    np.save(path / "trajectories.npy", np.zeros((num_sims, 1, 2, 2), np.float32))
    np.save(path / "x_grid.npy", np.linspace(0.0, 1.0, 100, dtype=np.float32))
    np.save(path / "y_grid.npy", np.linspace(0.0, 1.0, 100, dtype=np.float32))
    np.save(path / "t_grid.npy", np.array([0.0, 0.3], dtype=np.float32))
    np.save(path / "dt.npy", np.array(0.005))
    np.save(
        path / "sim_params.npy",
        np.asarray([{} for _ in range(num_sims)], dtype=object),
        allow_pickle=True,
    )
    return path


def _option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def _write_mock_forcing_results(command: list[str]) -> None:
    count = int(_option(command, "--sensor-n-y"))
    out_csv = Path(_option(command, "--out-csv"))
    artifact_dir = Path(_option(command, "--artifact-dir"))
    sim_start = command.index("--sim-ids") + 1
    sim_stop = command.index("--sensor-layout")
    sim_ids = [int(value) for value in command[sim_start:sim_stop]]
    rows = []
    for sid in sim_ids:
        true_value = 0.5
        signed_error = ((-1) ** sid) * sid * 0.005 * (8.0 / count)
        map_value = true_value + signed_error
        rows.append(
            {
                "benchmark": "forcing",
                "sim_id": sid,
                "n_sensors": count,
                "noise_seed": sid,
                "noise_std_K": 0.25,
                "R_c_true": true_value,
                "R_c_map": map_value,
                "R_c_abs_error": abs(signed_error),
                "fv_resid_rms_K": 0.18 + 0.001 * sid,
                "fv_resid_over_noise": (0.18 + 0.001 * sid) / 0.25,
                "profile_R_c_ci_low": 0.2,
                "profile_R_c_ci_high": 0.8,
                "profile_bound_limited": False,
            }
        )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for sid in sim_ids:
        np.savez(artifact_dir / f"sim_{sid:05d}.npz", sim_id=sid)


def _write_mock_calibration(path: Path, *, gate_pass: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        calibration_schema_version=1,
        benchmark="forcing",
        split="val",
        checkpoint_fingerprint="mock-checkpoint",
        dataset_fingerprint="mock-dataset",
        observation_fingerprint="mock-observation",
        local_gate_pass=gate_pass,
        simulation_count=1,
        sigma_fno_norm=0.01 if gate_pass else 0.2,
        sigma_fno_K=0.1 if gate_pass else 2.0,
        sensor_rms_median_K=0.05 if gate_pass else 2.0,
        sensor_rms_p90_K=0.08 if gate_pass else 3.0,
        sigma_meas_K=0.2 if gate_pass else 0.5,
    )


def _paper_summary_rows(benchmark: str) -> list[dict[str, object]]:
    spec = sweep.spec_for(benchmark)
    bound_limited_cases = {8: 3, 16: 1, 32: 0}
    noise_std_K = 0.25
    rows = []
    for count in sweep.SENSOR_COUNTS:
        scale = 8.0 / count
        for sim_id in range(1, 9):
            truth = 0.5
            absolute_error = 0.01 * sim_id * scale
            signed_error = absolute_error if sim_id % 2 else -absolute_error
            profile_width = (0.10 + 0.01 * sim_id) * scale
            profile_low = truth - 0.5 * profile_width
            fv_resid_K = (0.15 + 0.01 * sim_id) * scale
            rows.append(
                {
                    "benchmark": benchmark,
                    "sim_id": sim_id,
                    "n_sensors": count,
                    "noise_seed": sim_id,
                    "init_seed": 0,
                    "noise_std_K": noise_std_K,
                    spec.lead_true_col: truth,
                    spec.lead_hat_col: truth + signed_error,
                    spec.lead_abserr_col: absolute_error,
                    "fv_resid_rms_K": fv_resid_K,
                    "fv_resid_over_noise": fv_resid_K / noise_std_K,
                    spec.lead_profile_ci_low_col: profile_low,
                    spec.lead_profile_ci_high_col: profile_low + profile_width,
                    "profile_bound_limited": sim_id <= bound_limited_cases[count],
                }
            )
    return rows


@pytest.mark.parametrize(
    ("benchmark", "estimand", "unit"),
    [
        ("forcing", "R_c", "m² K/W"),
        ("forcing_itr", "S_R", "m³ K/W"),
    ],
)
def test_paper_summary_contains_all_table_statistics(
    tmp_path, benchmark, estimand, unit
):
    combined = tmp_path / "inverse_sensor_sweep.csv"
    sweep._write_csv(combined, _paper_summary_rows(benchmark))

    result = sweep.generate_paper_summary(
        combined,
        benchmark=benchmark,
        out_dir=tmp_path,
    )

    assert result["status"] == "complete"
    assert result["estimand"] == estimand
    assert result["unit"] == unit
    assert result["reported_statistics"] == [
        "recovery_error", "fv_verification", "profile_interval"
    ]
    assert "no mean/SD" in result["statistics_conventions"]["spread"]
    assert Path(result["json"]).is_file()
    assert Path(result["csv"]).is_file()

    arm8 = result["by_sensor_count"][0]
    expected_errors = np.arange(1, 9, dtype=float) * 0.01

    # Statistic 1: recovery error, robust spread only.
    absolute = arm8["recovery"]["absolute_error"]
    assert set(absolute) == {"n", "median", "q1", "q3", "iqr"}
    assert absolute["median"] == pytest.approx(0.045)
    assert absolute["q1"] == pytest.approx(np.percentile(expected_errors, 25.0))
    assert absolute["q3"] == pytest.approx(np.percentile(expected_errors, 75.0))
    assert arm8["recovery"]["rmse"] == pytest.approx(
        np.sqrt(np.mean(expected_errors**2))
    )

    # Statistic 2: FV verification against the noise floor.
    expected_fv = (0.15 + 0.01 * np.arange(1, 9, dtype=float))
    fv = arm8["fv_verification"]
    assert fv["residual_K"]["median"] == pytest.approx(np.median(expected_fv))
    assert fv["over_noise_median"] == pytest.approx(np.median(expected_fv) / 0.25)

    # Statistic 3: interval width plus the censoring count.
    profile = arm8["profile_interval"]
    expected_width = 0.10 + 0.01 * np.arange(1, 9, dtype=float)
    assert profile["width"]["median"] == pytest.approx(np.median(expected_width))
    assert profile["bound_limited_cases"] == 3
    assert result["by_sensor_count"][2]["profile_interval"][
        "bound_limited_cases"
    ] == 0

    # Retired blocks are absent from the summary entirely.
    assert "mcmc" not in arm8
    assert "profile_likelihood" not in arm8
    assert "paired_absolute_error_changes" not in result

    with Path(result["csv"]).open(newline="") as handle:
        flat_rows = list(csv.DictReader(handle))
    assert len(flat_rows) == 3
    assert set(flat_rows[0]) == set(sweep._PAPER_SUMMARY_COLUMNS)
    assert float(flat_rows[0]["absolute_error_median"]) == pytest.approx(0.045)
    assert float(flat_rows[0]["fv_resid_K_median"]) == pytest.approx(
        np.median(expected_fv)
    )
    assert int(flat_rows[0]["profile_bound_limited_cases"]) == 3


def test_paper_summary_rejects_inconsistent_derived_columns(tmp_path):
    rows = _paper_summary_rows("forcing")
    rows[0]["R_c_abs_error"] = 99.0
    combined = tmp_path / "inverse_sensor_sweep.csv"
    sweep._write_csv(combined, rows)
    with pytest.raises(ValueError, match="absolute-error column is inconsistent"):
        sweep.generate_paper_summary(
            combined,
            benchmark="forcing",
            out_dir=tmp_path,
        )


def test_default_data_dirs_and_cli_overrides(tmp_path):
    assert sweep.DEFAULT_DATA_DIRS["forcing"].name == "forcinginverse"
    assert sweep.DEFAULT_DATA_DIRS["forcing_itr"].name == "forcing_itr_inverse_70"
    args = sweep._build_parser().parse_args(
        [
            "--benchmark",
            "forcing",
            "--checkpoint",
            "model.pt",
            "--data-dir",
            str(tmp_path / "data"),
            "--out-dir",
            str(tmp_path / "out"),
            "--device",
            "cpu",
        ]
    )
    assert args.data_dir == tmp_path / "data"
    assert args.out_dir == tmp_path / "out"
    assert args.device == "cpu"


def test_default_device_prefers_available_accelerator(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert sweep.default_device() == "mps"
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert sweep.default_device() == "cpu"


def test_select_case_ids_excludes_calibration_and_requires_eight(tmp_path):
    config = {"data": {"train_split": 0.7, "val_split": 0.15}}
    data_dir = _data_dir(tmp_path / "data")
    sim_ids, calibration_ids = sweep.select_case_ids(data_dir, config)
    assert calibration_ids == [0]
    assert sim_ids == list(range(1, 9))
    assert not set(sim_ids).intersection(calibration_ids)

    too_small = _data_dir(tmp_path / "small", num_sims=8)
    with pytest.raises(ValueError, match="non-calibration cases"):
        sweep.select_case_ids(too_small, config)


def test_command_builders_fix_the_reviewer_protocol(tmp_path):
    common = {
        "checkpoint": tmp_path / "model.pt",
        "data_dir": tmp_path / "data",
        "device": "cpu",
        "sensor_count": 16,
        "calibration_path": tmp_path / "calibration.npz",
    }
    calibration = sweep.build_calibration_command(**common)
    inversion = sweep.build_inversion_command(
        **common,
        result_path=tmp_path / "results.csv",
        artifact_dir=tmp_path / "artifacts",
        sim_ids=list(range(8)),
    )
    for command in (calibration, inversion):
        assert _option(command, "--sensor-layout") == "interface_band"
        assert float(_option(command, "--sensor-x-halfwidth")) == 0.005
        assert int(_option(command, "--sensor-n-y")) == 16
        assert float(_option(command, "--noise-std")) == 0.05
        assert int(_option(command, "--seed")) == 0
        assert int(_option(command, "--noise-seed")) == 0
    assert int(_option(inversion, "--n-starts")) == 8
    # The three reported statistics are unconditional in the entry point, so
    # the protocol carries no per-statistic flags. The retired ones no longer
    # exist as flags at all.
    for flag in (
        "--fv-refine",
        "--sensitivity",
        "--profile",
        "--laplace",
        "--mcmc",
        "--fv-polish",
        "--fv-equivalent-scalar",
    ):
        assert flag not in inversion
    assert "--calibration-artifact" in inversion
    assert "--allow-surrogate-limited" not in inversion
    assert float(_option(inversion, "--uq-level")) == 0.95


def test_run_sweep_rejects_checkpoint_benchmark_mismatch(tmp_path):
    checkpoint = _checkpoint(tmp_path / "model.pt", benchmark="forcing")
    with pytest.raises(ValueError, match="does not match checkpoint benchmark"):
        sweep.run_sweep(
            benchmark="forcing_itr",
            checkpoint=checkpoint,
            data_dir=tmp_path / "unused",
            out_dir=tmp_path / "out",
            device="cpu",
        )


def test_run_sweep_calibrates_three_arms_and_writes_paired_csv(
    tmp_path, monkeypatch
):
    checkpoint = _checkpoint(tmp_path / "model.pt")
    data_dir = _data_dir(tmp_path / "data")
    out_dir = tmp_path / "out"
    commands = []
    plot_calls = []

    def fake_run(command):
        commands.append(command)
        if "--calibration-out" in command:
            path = Path(_option(command, "--calibration-out"))
            _write_mock_calibration(path)
        else:
            _write_mock_forcing_results(command)

    def fake_generate_plots(combined_path, *, benchmark, out_dir):
        plot_calls.append((combined_path, benchmark, out_dir))
        return {
            "status": "complete",
            "recovery": {
                "files": {
                    "png": str(out_dir / "plots" / "sensor_recovery_vs_sensors.png"),
                    "pdf": str(out_dir / "plots" / "sensor_recovery_vs_sensors.pdf"),
                }
            },
            "uncertainty": {
                "files": {
                    "png": str(out_dir / "plots" / "sensor_uq_vs_sensors.png"),
                    "pdf": str(out_dir / "plots" / "sensor_uq_vs_sensors.pdf"),
                }
            },
        }

    monkeypatch.setattr(sweep, "_run_command", fake_run)
    monkeypatch.setattr(
        sweep, "_checkpoint_fingerprint", lambda _: "mock-checkpoint"
    )
    monkeypatch.setattr(sweep, "generate_sweep_plots", fake_generate_plots)
    combined = sweep.run_sweep(
        benchmark="forcing",
        checkpoint=checkpoint,
        data_dir=data_dir,
        out_dir=out_dir,
        device="cpu",
    )

    assert len(commands) == 6
    assert [int(_option(command, "--sensor-n-y")) for command in commands] == [
        8,
        8,
        16,
        16,
        32,
        32,
    ]
    assert sum("--calibration-out" in command for command in commands) == 3
    assert sum("--out-csv" in command for command in commands) == 3
    assert plot_calls == [(combined, "forcing", out_dir.resolve())]

    with combined.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 24
    assert {int(row["n_sensors"]) for row in rows} == {8, 16, 32}
    assert {int(row["init_seed"]) for row in rows} == {0}
    assert {int(row["noise_seed"]) for row in rows} == set(range(1, 9))
    for count in sweep.SENSOR_COUNTS:
        assert sorted(
            int(row["sim_id"])
            for row in rows
            if int(row["n_sensors"]) == count
        ) == list(range(1, 9))

    with (out_dir / "sweep_manifest.json").open() as handle:
        manifest = json.load(handle)
    assert manifest["status"] == "complete"
    assert manifest["combined_row_count"] == 24
    assert manifest["calibration_sim_ids"] == [0]
    assert manifest["evaluation_sim_ids"] == list(range(1, 9))
    assert manifest["plots"]["status"] == "complete"
    assert manifest["schema_version"] == 2
    assert manifest["paper_summary"]["status"] == "complete"
    assert Path(manifest["paper_summary"]["json"]).is_file()
    assert Path(manifest["paper_summary"]["csv"]).is_file()
    assert len(manifest["paper_summary"]["by_sensor_count"]) == 3
    arm8 = manifest["paper_summary"]["by_sensor_count"][0]
    assert arm8["recovery"]["absolute_error"]["median"] > 0.0
    assert arm8["fv_verification"]["residual_K"]["median"] > 0.0
    assert arm8["profile_interval"]["width"]["median"] > 0.0
    assert all(
        manifest["arms"][str(count)]["status"] == "complete"
        for count in sweep.SENSOR_COUNTS
    )

    commands.clear()
    resumed = sweep.run_sweep(
        benchmark="forcing",
        checkpoint=checkpoint,
        data_dir=data_dir,
        out_dir=out_dir,
        device="cpu",
    )
    assert resumed == combined
    assert commands == []
    assert len(plot_calls) == 2
    with (out_dir / "sweep_manifest.json").open() as handle:
        resumed_manifest = json.load(handle)
    assert all(
        resumed_manifest["arms"][str(count)]["resumed"] is True
        for count in sweep.SENSOR_COUNTS
    )
    assert resumed_manifest["dataset_fingerprint"] == "mock-dataset"
    assert resumed_manifest["paper_summary"]["status"] == "complete"

    commands.clear()
    other_data_dir = _data_dir(tmp_path / "other_data")
    sweep.run_sweep(
        benchmark="forcing",
        checkpoint=checkpoint,
        data_dir=other_data_dir,
        out_dir=out_dir,
        device="cpu",
    )
    assert len(commands) == 6
    assert len(plot_calls) == 3


def test_run_sweep_preserves_results_when_plotting_fails(tmp_path, monkeypatch):
    checkpoint = _checkpoint(tmp_path / "model.pt")
    data_dir = _data_dir(tmp_path / "data")
    out_dir = tmp_path / "out"

    def fake_run(command):
        if "--calibration-out" in command:
            _write_mock_calibration(Path(_option(command, "--calibration-out")))
        else:
            _write_mock_forcing_results(command)

    monkeypatch.setattr(sweep, "_run_command", fake_run)
    monkeypatch.setattr(
        sweep, "_checkpoint_fingerprint", lambda _: "mock-checkpoint"
    )
    monkeypatch.setattr(
        sweep,
        "generate_sweep_plots",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("render failed")),
    )

    with pytest.raises(sweep.PlotGenerationError, match="render failed"):
        sweep.run_sweep(
            benchmark="forcing",
            checkpoint=checkpoint,
            data_dir=data_dir,
            out_dir=out_dir,
            device="cpu",
        )

    assert (out_dir / "inverse_sensor_sweep.csv").is_file()
    assert (out_dir / "inverse_sensor_sweep_summary.json").is_file()
    assert (out_dir / "inverse_sensor_sweep_summary.csv").is_file()
    for count in sweep.SENSOR_COUNTS:
        assert (out_dir / f"sensors_{count:02d}" / "inverse_results.csv").is_file()
        assert len(list((out_dir / f"sensors_{count:02d}" / "artifacts").glob("*.npz"))) == 8
    with (out_dir / "sweep_manifest.json").open() as handle:
        manifest = json.load(handle)
    assert manifest["status"] == "plotting_failed"
    assert "render failed" in manifest["plot_error"]
    assert manifest["paper_summary"]["status"] == "complete"


def test_run_sweep_preserves_results_when_summary_generation_fails(
    tmp_path, monkeypatch
):
    checkpoint = _checkpoint(tmp_path / "model.pt")
    data_dir = _data_dir(tmp_path / "data")
    out_dir = tmp_path / "out"

    def fake_run(command):
        if "--calibration-out" in command:
            _write_mock_calibration(Path(_option(command, "--calibration-out")))
        else:
            _write_mock_forcing_results(command)

    monkeypatch.setattr(sweep, "_run_command", fake_run)
    monkeypatch.setattr(
        sweep, "_checkpoint_fingerprint", lambda _: "mock-checkpoint"
    )
    monkeypatch.setattr(
        sweep,
        "generate_paper_summary",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("summary failed")
        ),
    )

    with pytest.raises(sweep.SummaryGenerationError, match="summary failed"):
        sweep.run_sweep(
            benchmark="forcing",
            checkpoint=checkpoint,
            data_dir=data_dir,
            out_dir=out_dir,
            device="cpu",
        )

    assert (out_dir / "inverse_sensor_sweep.csv").is_file()
    with (out_dir / "sweep_manifest.json").open() as handle:
        manifest = json.load(handle)
    assert manifest["status"] == "summary_failed"
    assert "summary failed" in manifest["summary_error"]


def test_run_sweep_preserves_failed_calibration_artifact(tmp_path, monkeypatch):
    checkpoint = _checkpoint(tmp_path / "model.pt")
    data_dir = _data_dir(tmp_path / "data")
    out_dir = tmp_path / "out"

    def fake_run(command):
        path = Path(_option(command, "--calibration-out"))
        _write_mock_calibration(path, gate_pass=False)

    monkeypatch.setattr(sweep, "_run_command", fake_run)
    monkeypatch.setattr(
        sweep, "_checkpoint_fingerprint", lambda _: "mock-checkpoint"
    )
    with pytest.raises(sweep.CalibrationGateError, match="median=2"):
        sweep.run_sweep(
            benchmark="forcing",
            checkpoint=checkpoint,
            data_dir=data_dir,
            out_dir=out_dir,
            device="cpu",
        )
    assert (out_dir / "sensors_08" / "surrogate_calibration.npz").is_file()
    with (out_dir / "sweep_manifest.json").open() as handle:
        manifest = json.load(handle)
    assert manifest["status"] == "calibration_gate_failed"
    assert manifest["failed_sensor_count"] == 8
