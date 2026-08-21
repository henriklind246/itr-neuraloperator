import subprocess
from pathlib import Path

import pytest
import yaml

from scripts.run_train_fixed import (
    _apply_override,
    _parse_override_value,
    _resolve_run_dir,
    _sync_data_paths_from_data_dir,
    _validate_fixed_run_config,
    main as fixed_main,
)


class TestRunTrainFixedHelpers:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("10", 10),
            ("0.001", 0.001),
            ("true", True),
            ("false", False),
            ("[42]", [42]),
            ("snapshots_ablation_300ep", "snapshots_ablation_300ep"),
        ],
    )
    def test_parse_override_value(self, raw, expected):
        assert _parse_override_value(raw) == expected

    def test_apply_override_updates_nested_value(self):
        config = {"training": {"n_snapshots": 15, "loss": {"interface_weight": 10.0}}}
        _apply_override(config, "training.n_snapshots", 20)
        _apply_override(config, "training.loss.interface_weight", 5.0)

        assert config["training"]["n_snapshots"] == 20
        assert config["training"]["loss"]["interface_weight"] == pytest.approx(5.0)

    def test_apply_override_rejects_unknown_path(self):
        config = {"training": {"n_snapshots": 15}}
        with pytest.raises(KeyError, match="Unknown config path"):
            _apply_override(config, "training.missing.value", 1)

    def test_boundary_extender_depth_is_cli_overrideable(self):
        config_path = Path(__file__).resolve().parents[1] / "conf" / "config.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        _apply_override(config, "model.parameters.forcing_extender_depth", 3)

        params = config["model"]["parameters"]
        assert params["forcing_extender_depth"] == 3
        assert params["forcing_embed_dim"] == 64
        assert params["forcing_extender_heads"] == 4
        assert params["forcing_spatial_dim"] == 16
        assert params["forcing_extender_grid_size"] == 16

    def test_boundary_extender_rc_conditioning_is_cli_overrideable(self):
        config_path = Path(__file__).resolve().parents[1] / "conf" / "config.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        assert config["model"]["parameters"][
            "forcing_extender_condition_on_rc"
        ] is False
        _apply_override(
            config, "model.parameters.forcing_extender_condition_on_rc", True
        )

        assert config["model"]["parameters"][
            "forcing_extender_condition_on_rc"
        ] is True

    def test_forcing_scalar_spatial_inputs_are_cli_overrideable(self):
        config_path = Path(__file__).resolve().parents[1] / "conf" / "config.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        _apply_override(config, "benchmark.spatial_input.itr", True)
        _apply_override(config, "benchmark.spatial_input.lead_time", True)

        assert config["benchmark"]["spatial_input"] == {
            "itr": True,
            "lead_time": True,
        }

    def test_physics_extender_options_are_cli_overrideable(self):
        config_path = Path(__file__).resolve().parents[1] / "conf" / "config.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        _apply_override(
            config, "model.parameters.forcing_spatial_mode", "physics_extender"
        )
        _apply_override(
            config, "model.parameters.forcing_extender_physics_hidden", 24
        )
        _apply_override(
            config, "model.parameters.forcing_extender_interface_x_norm", 0.45
        )

        params = config["model"]["parameters"]
        assert params["forcing_spatial_mode"] == "physics_extender"
        assert params["forcing_extender_physics_hidden"] == 24
        assert params["forcing_extender_interface_x_norm"] == pytest.approx(0.45)

    def test_validate_fixed_run_requires_experiment_name(self):
        config = {
            "experiment": {"name": ""},
            "config_id": 10,
            "training": {"seeds": [42], "run": {"run_dir": "/tmp/run"}},
        }
        with pytest.raises(ValueError, match="experiment.name"):
            _validate_fixed_run_config(config)

    def test_validate_fixed_run_requires_config_id(self):
        config = {
            "experiment": {"name": "test"},
            "config_id": None,
            "training": {"seeds": [42], "run": {"run_dir": "/tmp/run"}},
        }
        with pytest.raises(ValueError, match="config_id"):
            _validate_fixed_run_config(config)

    def test_validate_fixed_run_requires_nonempty_seeds(self):
        config = {
            "experiment": {"name": "test"},
            "config_id": 10,
            "training": {"seeds": [], "run": {"run_dir": "/tmp/run"}},
        }
        with pytest.raises(ValueError, match="training.seeds"):
            _validate_fixed_run_config(config)

    def test_resolve_run_dir_uses_experiment_and_config_id(self, tmp_path):
        config = {
            "paths": {"runs_root": str(tmp_path / "runs")},
            "experiment": {"name": "snapshots_ablation_300ep"},
            "config_id": 20,
        }
        run_dir = _resolve_run_dir(config)
        assert run_dir == tmp_path / "runs" / "snapshots_ablation_300ep" / "config20"

    def test_two_config_ids_resolve_to_distinct_run_dirs(self, tmp_path):
        base = {
            "paths": {"runs_root": str(tmp_path / "runs")},
            "experiment": {"name": "snapshots_ablation_300ep"},
        }
        cfg10 = {**base, "config_id": 10}
        cfg15 = {**base, "config_id": 15}

        assert _resolve_run_dir(cfg10) != _resolve_run_dir(cfg15)

    def test_sync_data_paths_from_data_dir_updates_standard_files(self, tmp_path):
        data_dir = tmp_path / "custom_data"
        config = {
            "paths": {"data_dir": str(data_dir)},
            "data": {
                "t_grid_path": "old/t_grid.npy",
                "x_grid_path": "old/x_grid.npy",
                "y_grid_path": "old/y_grid.npy",
                "trajectories.npy": "old/trajectories.npy",
                "sim_params_path": "old/sim_params.npy",
            },
        }

        _sync_data_paths_from_data_dir(config, {"paths.data_dir"})

        assert config["data"]["t_grid_path"] == str(data_dir / "t_grid.npy")
        assert config["data"]["x_grid_path"] == str(data_dir / "x_grid.npy")
        assert config["data"]["y_grid_path"] == str(data_dir / "y_grid.npy")
        assert config["data"]["trajectories.npy"] == str(data_dir / "trajectories.npy")
        assert config["data"]["sim_params_path"] == str(data_dir / "sim_params.npy")

    def test_sync_data_paths_preserves_explicit_file_override(self, tmp_path):
        data_dir = tmp_path / "custom_data"
        explicit_t_grid = tmp_path / "elsewhere" / "t_grid.npy"
        config = {
            "paths": {"data_dir": str(data_dir)},
            "data": {
                "t_grid_path": str(explicit_t_grid),
                "x_grid_path": "old/x_grid.npy",
                "y_grid_path": "old/y_grid.npy",
                "trajectories.npy": "old/trajectories.npy",
                "sim_params_path": "old/sim_params.npy",
            },
        }

        _sync_data_paths_from_data_dir(config, {"paths.data_dir", "data.t_grid_path"})

        assert config["data"]["t_grid_path"] == str(explicit_t_grid)
        assert config["data"]["x_grid_path"] == str(data_dir / "x_grid.npy")

    def test_msi_slurm_scripts_accept_data_dir_env_override(self):
        root = Path(__file__).resolve().parents[1]
        scripts = [
            root / "slurm" / "train_fno_msi_fixed.sbatch",
            root / "slurm" / "train_fno_msi_fixed_ddp.sbatch",
            root / "slurm" / "eval_fno_msi.sbatch",
        ]

        for script in scripts:
            subprocess.run(["bash", "-n", str(script)], check=True)
            text = script.read_text(encoding="utf-8")
            assert 'export DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"' in text

    def test_generate_data_slurm_accepts_output_overrides(self):
        script = Path(__file__).resolve().parents[1] / "slurm" / "generate_data_msi.sbatch"

        subprocess.run(["bash", "-n", str(script)], check=True)
        text = script.read_text(encoding="utf-8")

        assert 'DATA_OUT_DIR="${DATA_OUT_DIR:-$PROJECT_DIR/data}"' in text
        assert 'NUM_SIMS="${NUM_SIMS:-8000}"' in text
        assert '--save-dir "$DATA_OUT_DIR"' in text
        assert '"$@"' in text


class TestRunTrainFixedMain:
    def test_main_applies_overrides_and_calls_run_config_seeds(self, tmp_path, monkeypatch, capsys):
        base_config = {
            "paths": {"runs_root": str(tmp_path / "runs")},
            "experiment": {"name": "placeholder"},
            "config_id": 0,
            "training": {
                "device": "cpu",
                "seeds": [0],
                "run": {"run_dir": "placeholder"},
                "n_snapshots": 15,
                "epochs": 1500,
            },
            "model": {"parameters": {"width": 64}},
        }

        monkeypatch.setattr("scripts.run_train_fixed.load_config", lambda: base_config)
        monkeypatch.setattr(
            "scripts.run_train_fixed.run_config_seeds",
            lambda config, base_run_dir, seeds: {
                "num_seeds": len(seeds),
                "mean_best_val": 1.23,
                "per_seed": [],
            },
        )
        monkeypatch.setattr(
            "sys.argv",
            [
                "run_train_fixed.py",
                "experiment.name=test_ablation",
                "config_id=10",
                "training.n_snapshots=10",
                "training.seeds=[42]",
            ],
        )

        exit_code = fixed_main()
        captured = capsys.readouterr()

        assert exit_code == 0
        assert "Using fixed experiment namespace: test_ablation" in captured.out
        assert "config10" in captured.out
        assert '"mean_best_val": 1.23' in captured.out

    def test_main_rejects_malformed_override(self, tmp_path, monkeypatch):
        base_config = {
            "paths": {"runs_root": str(tmp_path / "runs")},
            "experiment": {"name": "placeholder"},
            "config_id": 0,
            "training": {"device": "cpu", "seeds": [0], "run": {"run_dir": "placeholder"}},
        }
        monkeypatch.setattr("scripts.run_train_fixed.load_config", lambda: base_config)
        monkeypatch.setattr("sys.argv", ["run_train_fixed.py", "not_an_override"])

        with pytest.raises(ValueError, match="Expected key=value"):
            fixed_main()
