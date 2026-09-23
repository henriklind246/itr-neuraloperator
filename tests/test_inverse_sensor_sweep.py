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


def _data_dir(path: Path, num_sims: int = sweep.N_CASES + sweep.CALIBRATION_FLOOR) -> Path:
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
    noise_std_K = float(_option(command, "--noise-std")) * 25.0
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
                "noise_std_K": noise_std_K,
                "R_c_true": true_value,
                "R_c_map": map_value,
                "R_c_abs_error": abs(signed_error),
                "fv_resid_rms_K": 0.18 + 0.001 * sid,
                "fv_resid_over_noise": (0.18 + 0.001 * sid) / noise_std_K if noise_std_K else float("nan"),
                "profile_R_c_ci_low": 0.2,
                "profile_R_c_ci_high": 0.8,
                "profile_R_c_bound_limited": False,
                "profile_R_c_ci_width": 0.6,
                "R_c_rel_error_pct": 100 * abs(signed_error) / true_value,
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
        calibration_schema_version=2,
        benchmark="forcing",
        split="val",
        checkpoint_fingerprint="mock-checkpoint",
        dataset_fingerprint="mock-dataset",
        observation_fingerprint="mock-observation",
        local_gate_pass=gate_pass,
        simulation_count=sweep.CALIBRATION_FLOOR,
        sigma_fno_norm=0.01 if gate_pass else 0.2,
        sigma_fno_K=0.1 if gate_pass else 2.0,
        sensor_rms_median_K=0.05 if gate_pass else 2.0,
        sensor_rms_p90_K=0.08 if gate_pass else 3.0,
        sigma_meas_K=0.2 if gate_pass else 0.5,
    )


def _paper_summary_rows(benchmark: str) -> list[dict[str, object]]:
    rows = []
    for count in sweep.SENSOR_COUNTS:
        scale = 8.0 / count
        for sim_id in range(1, 9):
            row = {
                "benchmark": benchmark, "sim_id": sim_id, "n_sensors": count,
                "noise_seed": sim_id, "init_seed": 0, "noise_std_K": 0.25,
                "fv_resid_rms_K": (0.15 + 0.01 * sim_id) * scale,
                "fv_resid_over_noise": (0.15 + 0.01 * sim_id) * scale / 0.25,
            }
            for i, spec in enumerate(sweep.specs_for(benchmark)):
                truth = 0.5
                error = (i + 1) * 0.01 * sim_id * scale
                width = (i + 1) * (0.10 + 0.01 * sim_id) * scale
                stem = f"profile_{spec.parameter}"
                row.update({
                    spec.truth_col: truth, spec.estimate_col: truth + error,
                    spec.absolute_error_col: error, spec.relative_error_col: 100 * error / truth,
                    f"{stem}_ci_low": truth - width / 2,
                    f"{stem}_ci_high": truth + width / 2,
                    f"{stem}_ci_width": width,
                    f"{stem}_bound_limited": sim_id <= {8: 3, 16: 1, 32: 0}[count],
                })
            rows.append(row)
    return rows


@pytest.mark.parametrize("benchmark", sweep.BENCHMARKS)
def test_paper_summary_contains_all_table_statistics(tmp_path, benchmark):
    combined = tmp_path / "inverse_sensor_sweep.csv"
    sweep._write_csv(combined, _paper_summary_rows(benchmark))
    result = sweep.generate_paper_summary(combined, benchmark=benchmark, out_dir=tmp_path)
    names = [spec.parameter for spec in sweep.specs_for(benchmark)]
    assert result["status"] == "complete"
    assert result["parameters"] == names
    assert result["reported_statistics"] == ["parameter_recovery", "fv_verification", "parameter_profile_intervals"]
    assert Path(result["json"]).is_file()
    arm8 = result["by_sensor_count"][0]
    for i, name in enumerate(names):
        metrics = arm8["parameters"][name]
        expected_errors = (i + 1) * np.arange(1, 9) * 0.01
        assert metrics["recovery"]["absolute_error"]["median"] == pytest.approx(np.median(expected_errors))
        assert metrics["recovery"]["relative_error_pct"]["median"] == pytest.approx(np.median(expected_errors) * 200)
        assert metrics["recovery"]["rmse"] == pytest.approx(np.sqrt(np.mean(expected_errors**2)))
        profile = metrics["profile_interval"]
        assert profile["width"]["n"] == 5
        assert profile["width"]["median"] == pytest.approx((i + 1) * np.median(0.10 + 0.01 * np.arange(4, 9)))
        assert profile["bound_limited_cases"] == 3
    assert arm8["fv_verification"]["residual_K"]["median"] == pytest.approx(0.195)
    with Path(result["csv"]).open(newline="") as handle:
        flat = list(csv.DictReader(handle))
    assert len(flat) == 3 * len(names)
    assert set(flat[0]) == set(sweep._PAPER_SUMMARY_COLUMNS)
    assert {row["parameter"] for row in flat} == set(names)


def test_summary_handles_no_closed_intervals_and_near_zero_amplitude(tmp_path):
    rows = _paper_summary_rows("forcing_itr_sin")
    for row in rows:
        row["profile_A_bound_limited"] = True
        row["A_true"] = 0.0
        row["A_abserr"] = abs(row["A_hat"])
        row["A_rel_error_pct"] = float("nan")
    combined = tmp_path / "inverse_sensor_sweep.csv"
    sweep._write_csv(combined, rows)
    result = sweep.generate_paper_summary(combined, benchmark="forcing_itr_sin", out_dir=tmp_path)
    for arm in result["by_sensor_count"]:
        amp = arm["parameters"]["A"]
        assert amp["recovery"]["relative_error_undefined_cases"] == 8
        assert amp["recovery"]["relative_error_pct"]["n"] == 0
        assert amp["profile_interval"]["width"]["median"] is None
        assert amp["profile_interval"]["width"]["n"] == 0
        assert arm["parameters"]["R_base"]["recovery"]["relative_error_pct"]["n"] == 8
    assert "NaN" not in Path(result["json"]).read_text()


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


def test_cli_data_dir_optional_and_overrides(tmp_path):
    # The sweep generates its own disjoint dataset by default, so --data-dir is
    # optional and defaults to None (generate).
    minimal = sweep._build_parser().parse_args(
        ["--benchmark", "forcing", "--checkpoint", "model.pt"]
    )
    assert minimal.data_dir is None
    assert set(sweep.BENCHMARKS) == {
        "forcing",
        "forcing_itr_sin",
        "source_itr_sin",
    }

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


def test_prepare_sweep_dataset_partitions_existing_dir_disjointly(tmp_path):
    config = {"data": {}}
    data_dir = _data_dir(tmp_path / "data")
    resolved, sim_ids, calibration_ids = sweep.prepare_sweep_dataset(
        data_dir=data_dir, benchmark="forcing", config=config,
        out_dir=tmp_path / "out",
    )
    assert resolved == data_dir.resolve()
    assert sim_ids == list(range(sweep.N_CASES))
    assert calibration_ids == list(
        range(sweep.N_CASES, sweep.N_CASES + sweep.CALIBRATION_FLOOR)
    )
    assert not set(sim_ids).intersection(calibration_ids)

    too_small = _data_dir(tmp_path / "small", num_sims=sweep.N_CASES + 1)
    with pytest.raises(ValueError, match="fixed sweep protocol needs"):
        sweep.prepare_sweep_dataset(
            data_dir=too_small, benchmark="forcing", config=config,
            out_dir=tmp_path / "out2",
        )


def test_prepare_sweep_dataset_generates_when_no_dir(tmp_path, monkeypatch):
    config = {"data": {"nx": 100, "ny": 100, "save_stride": 2}}
    captured = {}

    def fake_prepare(**kwargs):
        captured.update(kwargs)
        gen_dir = Path(kwargs["out_dir"])
        gen_dir.mkdir(parents=True, exist_ok=True)
        return {
            "data_dir": str(gen_dir),
            "invert_ids": list(range(kwargs["n_invert"])),
            "calibration_ids": list(
                range(kwargs["n_invert"], kwargs["n_invert"] + kwargs["n_calibration"])
            ),
        }

    monkeypatch.setattr(sweep, "prepare_inversion_dataset", fake_prepare)
    resolved, sim_ids, calibration_ids = sweep.prepare_sweep_dataset(
        data_dir=None, benchmark="forcing", config=config, out_dir=tmp_path / "out",
    )
    assert captured["benchmark"] == "forcing"
    assert captured["n_invert"] == sweep.N_CASES
    assert captured["n_calibration"] == sweep.CALIBRATION_FLOOR
    assert captured["dataset_seed"] == sweep.INVERSION_DATASET_SEED
    assert resolved == (tmp_path / "out" / "inversion_data").resolve()
    assert sim_ids == list(range(sweep.N_CASES))
    assert calibration_ids == list(
        range(sweep.N_CASES, sweep.N_CASES + sweep.CALIBRATION_FLOOR)
    )


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
        benchmark="forcing",
        result_path=tmp_path / "results.csv",
        artifact_dir=tmp_path / "artifacts",
        sim_ids=list(range(8)),
    )
    for command in (calibration, inversion):
        assert _option(command, "--sensor-layout") == "interface_band"
        assert float(_option(command, "--sensor-x-halfwidth")) == 0.005
        assert int(_option(command, "--sensor-n-y")) == 16
        assert float(_option(command, "--noise-std")) == 0.01
        assert int(_option(command, "--seed")) == 0
        assert int(_option(command, "--noise-seed")) == 0
        # Both subprocesses rebuild the identical disjoint partition so their
        # dataset fingerprints match.
        assert int(_option(command, "--n-invert")) == sweep.N_CASES
        assert int(_option(command, "--n-calibration")) == sweep.CALIBRATION_FLOOR
    assert int(_option(inversion, "--n-starts")) == 8
    assert _option(inversion, "--optimizer") == "nelder_mead"
    assert int(_option(inversion, "--nm-maxiter")) == 60
    assert float(_option(inversion, "--nm-xatol")) == 1e-4
    assert float(_option(inversion, "--nm-fatol")) == 1e-14
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


def test_joint_nll_grid_is_requested_only_where_two_parameters_are_recovered(tmp_path):
    common = {
        "checkpoint": tmp_path / "model.pt",
        "data_dir": tmp_path / "data",
        "device": "cpu",
        "sensor_count": 16,
        "calibration_path": tmp_path / "calibration.npz",
        "result_path": tmp_path / "results.csv",
        "artifact_dir": tmp_path / "artifacts",
        "sim_ids": list(range(8)),
    }
    # The joint surface is only defined over a 2-d parameter space, and
    # scripts/invert.py rejects the flag outright for any other theta_dim, so
    # requesting it for scalar R_c would abort the whole arm.
    assert "--joint-nll-grid" not in sweep.build_inversion_command(
        **common, benchmark="forcing"
    )
    for benchmark in ("forcing_itr_sin", "source_itr_sin"):
        command = sweep.build_inversion_command(**common, benchmark=benchmark)
        assert int(_option(command, "--joint-nll-grid")) == sweep.JOINT_NLL_GRID


def test_run_sweep_rejects_checkpoint_benchmark_mismatch(tmp_path):
    checkpoint = _checkpoint(tmp_path / "model.pt", benchmark="forcing")
    with pytest.raises(ValueError, match="does not match checkpoint benchmark"):
        sweep.run_sweep(
            benchmark="forcing_itr_sin",
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

    def fake_run(command):
        commands.append(command)
        if "--calibration-out" in command:
            path = Path(_option(command, "--calibration-out"))
            _write_mock_calibration(path)
        else:
            _write_mock_forcing_results(command)

    monkeypatch.setattr(sweep, "_run_command", fake_run)
    monkeypatch.setattr(
        sweep, "_checkpoint_fingerprint", lambda _: "mock-checkpoint"
    )
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
    with combined.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 24
    assert {int(row["n_sensors"]) for row in rows} == {8, 16, 32}
    assert {int(row["init_seed"]) for row in rows} == {0}
    assert {int(row["noise_seed"]) for row in rows} == set(range(sweep.N_CASES))
    for count in sweep.SENSOR_COUNTS:
        assert sorted(
            int(row["sim_id"])
            for row in rows
            if int(row["n_sensors"]) == count
        ) == list(range(sweep.N_CASES))

    with (out_dir / "sweep_manifest.json").open() as handle:
        manifest = json.load(handle)
    assert manifest["status"] == "complete"
    assert manifest["fixed_protocol"]["optimizer"] == "nelder_mead"
    assert manifest["fixed_protocol"]["noise_std_norm"] == 0.01
    assert manifest["fixed_protocol"]["nm_maxiter"] == 60
    assert manifest["fixed_protocol"]["nm_xatol"] == 1e-4
    assert manifest["fixed_protocol"]["nm_fatol"] == 1e-14
    # Scalar R_c: the joint surface does not exist, and the manifest says so
    # rather than staying silent, so a figure can tell "not requested" from
    # "requested and missing".
    assert manifest["fixed_protocol"]["joint_nll_grid"] == 0
    assert manifest["combined_row_count"] == 24
    assert Path(manifest["canonical_csv"]) == combined
    assert manifest["calibration_sim_ids"] == list(
        range(sweep.N_CASES, sweep.N_CASES + sweep.CALIBRATION_FLOOR)
    )
    assert manifest["evaluation_sim_ids"] == list(range(sweep.N_CASES))
    assert "plots" not in manifest
    assert manifest["schema_version"] == 4
    assert manifest["paper_summary"]["status"] == "complete"
    assert Path(manifest["paper_summary"]["json"]).is_file()
    assert Path(manifest["paper_summary"]["csv"]).is_file()
    assert len(manifest["paper_summary"]["by_sensor_count"]) == 3
    arm8 = manifest["paper_summary"]["by_sensor_count"][0]
    assert arm8["parameters"]["R_c"]["recovery"]["absolute_error"]["median"] > 0.0
    assert arm8["fv_verification"]["residual_K"]["median"] > 0.0
    assert arm8["parameters"]["R_c"]["profile_interval"]["width"]["median"] > 0.0
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
    with (out_dir / "sweep_manifest.json").open() as handle:
        resumed_manifest = json.load(handle)
    assert all(
        resumed_manifest["arms"][str(count)]["resumed"] is True
        for count in sweep.SENSOR_COUNTS
    )
    assert resumed_manifest["dataset_fingerprint"] == "mock-dataset"
    assert resumed_manifest["paper_summary"]["status"] == "complete"

    del resumed_manifest["fixed_protocol"]["optimizer"]
    sweep._write_json(out_dir / "sweep_manifest.json", resumed_manifest)
    sweep.run_sweep(
        benchmark="forcing",
        checkpoint=checkpoint,
        data_dir=data_dir,
        out_dir=out_dir,
        device="cpu",
    )
    assert len(commands) == 6

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


def test_sweep_runner_has_no_legacy_plot_dependency():
    source = Path(sweep.__file__).read_text()
    assert "generate_sweep_plots" not in source
    assert "PlotGenerationError" not in source


@pytest.mark.parametrize("existing_data", [False, True])
def test_main_runs_noisy_then_noise_free_on_the_same_dataset(
    tmp_path, monkeypatch, capsys, existing_data
):
    checkpoint = _checkpoint(tmp_path / "model.pt")
    data_dir = _data_dir(tmp_path / "data")
    out_dir = tmp_path / "out"
    commands = []
    generations = []

    def fake_prepare(**kwargs):
        generations.append(kwargs)
        return {
            "data_dir": str(data_dir),
            "invert_ids": list(range(sweep.N_CASES)),
            "calibration_ids": list(range(sweep.N_CASES, sweep.N_CASES + sweep.CALIBRATION_FLOOR)),
        }

    def fake_run(command):
        commands.append(command)
        if "--calibration-out" in command:
            _write_mock_calibration(
                Path(_option(command, "--calibration-out")),
                gate_pass=float(_option(command, "--noise-std")) > 0.0,
            )
        else:
            _write_mock_forcing_results(command)

    monkeypatch.setattr(sweep, "prepare_inversion_dataset", fake_prepare)
    monkeypatch.setattr(sweep, "_run_command", fake_run)
    monkeypatch.setattr(sweep, "_checkpoint_fingerprint", lambda _: "mock-checkpoint")
    argv = [
        "--benchmark", "forcing", "--checkpoint", str(checkpoint),
        "--out-dir", str(out_dir), "--device", "cpu", "--no-figures",
    ]
    if existing_data:
        argv += ["--data-dir", str(data_dir)]
    assert sweep.main(argv) == 0
    assert len(generations) == (0 if existing_data else 1)
    assert len(commands) == 12
    assert [float(_option(cmd, "--noise-std")) for cmd in commands] == [0.01] * 6 + [0.0] * 6
    for noisy, clean in zip(commands[:6], commands[6:]):
        for option in ("--checkpoint", "--data-dir", "--seed", "--noise-seed", "--sensor-n-y", "--device"):
            assert _option(noisy, option) == _option(clean, option)
        if "--out-csv" in clean:
            assert "--allow-surrogate-limited" in clean
            assert "--allow-surrogate-limited" not in noisy
            assert Path(_option(clean, "--out-csv")).is_relative_to(out_dir / "no_noise")

    manifest = json.loads((out_dir / "no_noise" / "sweep_manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["fixed_protocol"]["noise_std_norm"] == 0.0
    assert manifest["data_dir"] == str(data_dir)
    assert all(not arm["calibration"]["local_gate_pass"] for arm in manifest["arms"].values())
    assert all(arm["fv_verification"]["over_noise_median"] is None
               for arm in manifest["paper_summary"]["by_sensor_count"])
    final_output = capsys.readouterr().out.split("Results for both sensor-noise settings:")[1]
    assert "With sensor noise:" in final_output
    assert "No sensor noise:" in final_output
    assert "noise ratio unavailable (no sensor noise)" in final_output

    commands.clear()
    assert sweep.main(argv) == 0
    assert commands == []


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


# --------------------------------------------------------------- paper figures


def _sweep_with_mocked_subprocesses(tmp_path, monkeypatch, **kwargs):
    """A complete ``forcing`` sweep with the inversion subprocess stubbed out."""
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
    sweep.run_sweep(
        benchmark="forcing",
        checkpoint=checkpoint,
        data_dir=data_dir,
        out_dir=out_dir,
        device="cpu",
        **kwargs,
    )
    with (out_dir / "sweep_manifest.json").open() as handle:
        return out_dir, json.load(handle)


def test_a_benchmark_with_no_registered_figures_is_skipped_not_failed(
    tmp_path, monkeypatch
):
    # F29/F30 read the sinusoidal R_c(y) profile, which the scalar-R_c forcing
    # benchmark does not have. Having nothing to draw is a property of the
    # benchmark, so it must not look like an error in the manifest.
    out_dir, manifest = _sweep_with_mocked_subprocesses(tmp_path, monkeypatch)

    assert manifest["status"] == "complete"
    assert manifest["paper_figures"]["status"] == "skipped"
    assert "forcing" in manifest["paper_figures"]["reason"]
    assert not (out_dir / "figures").exists()


def test_a_figure_failure_does_not_invalidate_the_sweep(tmp_path, monkeypatch):
    """The whole reason the figure hook is wrapped separately from the summary.

    The summary *is* the sweep's numeric result. The figures are a redrawing of
    a CSV already on disk, so a backend or font-cache problem must not mark
    hours of GPU time as failed.
    """
    monkeypatch.setattr(
        sweep,
        "generate_paper_figures",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("no display available")
        ),
    )
    out_dir, manifest = _sweep_with_mocked_subprocesses(tmp_path, monkeypatch)

    assert manifest["status"] == "complete"
    assert manifest["paper_figures"]["status"] == "failed"
    assert "no display available" in manifest["paper_figures"]["error"]
    assert (out_dir / "inverse_sensor_sweep.csv").is_file()
    assert manifest["paper_summary"]["status"] == "complete"


def test_figures_can_be_turned_off_without_touching_anything_else(
    tmp_path, monkeypatch
):
    calls = []
    monkeypatch.setattr(
        sweep,
        "generate_paper_figures",
        lambda *args, **kwargs: calls.append(kwargs) or {},
    )
    out_dir, manifest = _sweep_with_mocked_subprocesses(
        tmp_path, monkeypatch, figures=False
    )

    assert calls == []
    assert manifest["paper_figures"] == {"status": "disabled", "figures": []}
    assert manifest["status"] == "complete"
    assert (out_dir / "inverse_sensor_sweep.csv").is_file()


def test_cli_draws_figures_unless_asked_not_to():
    base = ["--benchmark", "forcing_itr_sin", "--checkpoint", "model.pt"]
    assert sweep._build_parser().parse_args(base).figures is True
    assert sweep._build_parser().parse_args(base + ["--no-figures"]).figures is False


def test_generate_paper_figures_renders_the_registered_keys_from_its_own_csv(
    tmp_path,
):
    """Renders for real, deliberately: a mocked ``registry.render`` proves only
    that arguments were passed, and the requirement set this builds a manifest
    for has gates (min_seeds, required columns, benchmark coverage) that a mock
    would sail straight past.

    Provenance is the point. The strict in-memory manifest built from
    ``combined_path`` is what stops the sweep from redrawing whatever
    ``visual/pub/manifest.yaml`` happens to point at, which would produce a
    figure whose numbers came from a different run.
    """
    from tests.test_pub_figures import (
        REPRESENTATIVE_SIM,
        write_itr_sin_sweep_with_artifacts,
    )

    combined = write_itr_sin_sweep_with_artifacts(tmp_path / "sweep")
    out_dir = tmp_path / "out"
    status = sweep.generate_paper_figures(
        combined, benchmark="forcing_itr_sin", out_dir=out_dir,
        seed="mock-checkpoint",
    )

    assert status["status"] == "complete"
    assert status["dir"] == str(out_dir / "figures")
    assert [figure["key"] for figure in status["figures"]] == list(
        sweep.PAPER_FIGURES["forcing_itr_sin"]
    )
    for figure in status["figures"]:
        paths = [Path(path) for path in figure["paths"]]
        assert {path.suffix for path in paths} == {".png", ".pdf"}
        assert all(path.is_file() for path in paths)
        # Strict, so nothing may land in degraded/.
        assert all("degraded" not in path.parts for path in paths)
        # The chosen case is the one thing about these figures that is not
        # already recoverable from the sweep's numeric outputs, so it is lifted
        # out of the sidecar and into the sweep manifest.
        assert figure["selection"]["sim_id"] == REPRESENTATIVE_SIM


def test_a_sidecar_without_a_selection_is_not_an_error(tmp_path):
    assert sweep.read_sidecar_selection(tmp_path / "absent.json") == {}

    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"figure": "F29_inverse_rc_profile_recovery"}))
    assert sweep.read_sidecar_selection(empty) == {}


@pytest.mark.parametrize(
    "status, expected",
    [
        ({"status": "skipped", "reason": "no paper figures for forcing"},
         "no paper figures for forcing"),
        ({"status": "disabled", "figures": []}, "--no-figures"),
        ({"status": "complete", "dir": "/out/figures",
          "figures": [{"key": "F29_inverse_rc_profile_recovery",
                       "paths": ["/out/figures/F29.png"],
                       "selection": {"sim_id": 3}}]},
         "sim_id=3"),
    ],
)
def test_every_figure_state_prints_something_legible(status, expected, capsys):
    sweep._print_paper_figures(status)
    assert expected in capsys.readouterr().out
