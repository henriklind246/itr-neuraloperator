import csv
import json
import os
import subprocess

import pytest
import yaml
from pathlib import Path
from unittest.mock import patch

from scripts.run_train import (
    _next_experiment_name,
    _job_number,
    _next_config_index,
    _write_yaml,
    _write_json,
    _read_index_rows,
    _update_index_and_best,
    _ensure_experiment_name,
    _validate_optuna_db,
    _pre_init_optuna_storage,
    _config_sort_key,
    EXPERIMENT_ENV_VAR,
)
from scripts.run_train_fixed import (
    _parse_override_value,
    _apply_override,
    _sync_data_paths_from_data_dir,
    _validate_fixed_run_config,
    _resolve_run_dir,
    main as fixed_main,
)


class TestNextExperimentName:
    def test_empty_dir(self, tmp_path):
        assert _next_experiment_name(tmp_path) == "experiment0"

    def test_increments(self, tmp_path):
        (tmp_path / "experiment0").mkdir()
        (tmp_path / "experiment1").mkdir()
        assert _next_experiment_name(tmp_path) == "experiment2"

    def test_skips_non_matching(self, tmp_path):
        (tmp_path / "experiment0").mkdir()
        (tmp_path / "other_stuff").mkdir()
        assert _next_experiment_name(tmp_path) == "experiment1"

    def test_handles_gaps(self, tmp_path):
        (tmp_path / "experiment0").mkdir()
        (tmp_path / "experiment5").mkdir()
        assert _next_experiment_name(tmp_path) == "experiment6"

    def test_creates_dir_if_missing(self, tmp_path):
        runs_root = tmp_path / "nonexistent" / "runs"
        name = _next_experiment_name(runs_root)
        assert name == "experiment0"
        assert runs_root.exists()


class TestNextConfigIndex:
    def test_empty_dir(self, tmp_path):
        assert _next_config_index(tmp_path) == 0

    def test_nonexistent_dir(self, tmp_path):
        assert _next_config_index(tmp_path / "does_not_exist") == 0

    def test_increments(self, tmp_path):
        (tmp_path / "conf0").mkdir()
        (tmp_path / "conf1").mkdir()
        (tmp_path / "conf2").mkdir()
        assert _next_config_index(tmp_path) == 3

    def test_handles_gaps(self, tmp_path):
        (tmp_path / "conf0").mkdir()
        (tmp_path / "conf5").mkdir()
        assert _next_config_index(tmp_path) == 6

    def test_skips_non_matching(self, tmp_path):
        (tmp_path / "conf0").mkdir()
        (tmp_path / "seed0").mkdir()
        (tmp_path / "other_stuff").mkdir()
        assert _next_config_index(tmp_path) == 1

    def test_ignores_files(self, tmp_path):
        (tmp_path / "conf0").mkdir()
        (tmp_path / "conf1.yaml").touch()  # file, not dir
        assert _next_config_index(tmp_path) == 1


class TestJobNumber:
    def test_returns_zero_when_hydra_not_initialized(self):
        assert _job_number() == 0


# ===================== _config_sort_key =====================

class TestConfigSortKey:
    def test_extracts_number(self):
        assert _config_sort_key("conf0") == 0
        assert _config_sort_key("conf12") == 12

    def test_returns_zero_for_non_matching(self):
        assert _config_sort_key("other") == 0


# ===================== _write_yaml / _write_json =====================

class TestWriteYaml:
    def test_creates_file(self, tmp_path):
        path = tmp_path / "sub" / "out.yaml"
        _write_yaml(path, {"key": "value", "num": 42})
        assert path.exists()
        loaded = yaml.safe_load(path.read_text())
        assert loaded == {"key": "value", "num": 42}

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "a" / "b" / "c.yaml"
        _write_yaml(path, {"x": 1})
        assert path.exists()


class TestWriteJson:
    def test_creates_file(self, tmp_path):
        path = tmp_path / "sub" / "out.json"
        _write_json(path, {"key": "value", "num": 42})
        assert path.exists()
        loaded = json.loads(path.read_text())
        assert loaded == {"key": "value", "num": 42}

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "a" / "b" / "c.json"
        _write_json(path, {"x": 1})
        assert path.exists()


# ===================== _read_index_rows =====================

class TestReadIndexRows:
    def test_returns_empty_for_missing_file(self, tmp_path):
        assert _read_index_rows(tmp_path / "nonexistent.csv") == []

    def test_reads_csv_rows(self, tmp_path):
        index_path = tmp_path / "index.csv"
        with index_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["config_name", "objective_mean_best_val", "num_seeds"])
            writer.writeheader()
            writer.writerow({"config_name": "conf0", "objective_mean_best_val": "0.05", "num_seeds": "2"})
            writer.writerow({"config_name": "conf1", "objective_mean_best_val": "0.03", "num_seeds": "2"})
        rows = _read_index_rows(index_path)
        assert len(rows) == 2
        assert rows[0]["config_name"] == "conf0"
        assert rows[1]["config_name"] == "conf1"


# ===================== _update_index_and_best =====================

class TestUpdateIndexAndBest:
    def test_creates_index_csv(self, tmp_path):
        _update_index_and_best(tmp_path, "conf0", objective=0.05, num_seeds=2)
        index_path = tmp_path / "index.csv"
        assert index_path.exists()
        rows = _read_index_rows(index_path)
        assert len(rows) == 1
        assert rows[0]["config_name"] == "conf0"
        assert float(rows[0]["objective_mean_best_val"]) == pytest.approx(0.05)

    def test_appends_to_existing_index(self, tmp_path):
        _update_index_and_best(tmp_path, "conf0", objective=0.05, num_seeds=2)
        _update_index_and_best(tmp_path, "conf1", objective=0.03, num_seeds=2)
        rows = _read_index_rows(tmp_path / "index.csv")
        assert len(rows) == 2
        assert rows[0]["config_name"] == "conf0"
        assert rows[1]["config_name"] == "conf1"

    def test_updates_existing_config_entry(self, tmp_path):
        _update_index_and_best(tmp_path, "conf0", objective=0.05, num_seeds=2)
        _update_index_and_best(tmp_path, "conf0", objective=0.02, num_seeds=2)
        rows = _read_index_rows(tmp_path / "index.csv")
        assert len(rows) == 1
        assert float(rows[0]["objective_mean_best_val"]) == pytest.approx(0.02)

    def test_copies_best_config_yaml(self, tmp_path):
        _write_yaml(tmp_path / "conf0.yaml", {"lr": 0.01})
        _write_yaml(tmp_path / "conf1.yaml", {"lr": 0.001})
        _update_index_and_best(tmp_path, "conf0", objective=0.05, num_seeds=2)
        _update_index_and_best(tmp_path, "conf1", objective=0.02, num_seeds=2)
        best = yaml.safe_load((tmp_path / "best_config.yaml").read_text())
        # conf1 has lower objective, so it should be copied as best
        assert best == {"lr": 0.001}

    def test_rows_sorted_by_config_index(self, tmp_path):
        _update_index_and_best(tmp_path, "conf3", objective=0.05, num_seeds=1)
        _update_index_and_best(tmp_path, "conf1", objective=0.03, num_seeds=1)
        _update_index_and_best(tmp_path, "conf2", objective=0.04, num_seeds=1)
        rows = _read_index_rows(tmp_path / "index.csv")
        assert [r["config_name"] for r in rows] == ["conf1", "conf2", "conf3"]


# ===================== _ensure_experiment_name =====================

class TestEnsureExperimentName:
    def test_uses_existing_env_var(self, tmp_path, monkeypatch):
        monkeypatch.setenv(EXPERIMENT_ENV_VAR, "my_experiment")
        name = _ensure_experiment_name(tmp_path)
        assert name == "my_experiment"

    def test_generates_name_when_env_not_set(self, tmp_path, monkeypatch):
        monkeypatch.delenv(EXPERIMENT_ENV_VAR, raising=False)
        name = _ensure_experiment_name(tmp_path)
        assert name == "experiment0"
        # Should also set the env var
        assert os.environ[EXPERIMENT_ENV_VAR] == "experiment0"

    def test_increments_when_experiments_exist(self, tmp_path, monkeypatch):
        monkeypatch.delenv(EXPERIMENT_ENV_VAR, raising=False)
        (tmp_path / "runs" / "experiment0").mkdir(parents=True)
        (tmp_path / "runs" / "experiment1").mkdir()
        name = _ensure_experiment_name(tmp_path)
        assert name == "experiment2"

    def test_creates_experiment_directory(self, tmp_path, monkeypatch):
        monkeypatch.delenv(EXPERIMENT_ENV_VAR, raising=False)
        name = _ensure_experiment_name(tmp_path)
        assert (tmp_path / "runs" / name).is_dir()

    def test_removes_corrupt_optuna_db_with_env_var(self, tmp_path, monkeypatch):
        """Corrupt DB is removed even when experiment name comes from env var."""
        monkeypatch.setenv(EXPERIMENT_ENV_VAR, "experiment2")
        exp_dir = tmp_path / "runs" / "experiment2"
        exp_dir.mkdir(parents=True)
        db_path = exp_dir / "optuna_study.db"
        db_path.touch()  # 0-byte corrupt file

        name = _ensure_experiment_name(tmp_path)
        assert name == "experiment2"
        assert exp_dir.is_dir()
        assert not db_path.exists()  # corrupt DB removed

    def test_keeps_valid_optuna_db(self, tmp_path, monkeypatch):
        """A valid Optuna DB is not removed."""
        import sqlite3
        monkeypatch.setenv(EXPERIMENT_ENV_VAR, "experiment0")
        exp_dir = tmp_path / "runs" / "experiment0"
        exp_dir.mkdir(parents=True)
        db_path = exp_dir / "optuna_study.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)")
        conn.execute("INSERT INTO alembic_version VALUES ('v3.1.0')")
        conn.commit()
        conn.close()

        _ensure_experiment_name(tmp_path)
        assert db_path.exists()  # valid DB preserved


# ===================== _validate_optuna_db =====================

class TestValidateOptunaDb:
    def test_missing_file_is_valid(self, tmp_path):
        assert _validate_optuna_db(tmp_path / "nonexistent.db") is True

    def test_empty_file_is_invalid(self, tmp_path):
        db = tmp_path / "optuna_study.db"
        db.touch()
        assert _validate_optuna_db(db) is False

    def test_corrupt_file_is_invalid(self, tmp_path):
        db = tmp_path / "optuna_study.db"
        db.write_bytes(b"not a sqlite database")
        assert _validate_optuna_db(db) is False

    def test_valid_sqlite_without_alembic_is_invalid(self, tmp_path):
        """A valid SQLite DB that lacks Optuna's alembic_version table."""
        import sqlite3
        db = tmp_path / "optuna_study.db"
        conn = sqlite3.connect(str(db))
        conn.execute("CREATE TABLE dummy (id INTEGER)")
        conn.commit()
        conn.close()
        assert _validate_optuna_db(db) is False

    def test_valid_optuna_db(self, tmp_path):
        """A SQLite DB with the alembic_version table and a row is valid."""
        import sqlite3
        db = tmp_path / "optuna_study.db"
        conn = sqlite3.connect(str(db))
        conn.execute("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)")
        conn.execute("INSERT INTO alembic_version VALUES ('v3.1.0')")
        conn.commit()
        conn.close()
        assert _validate_optuna_db(db) is True

    def test_empty_alembic_version_is_invalid(self, tmp_path):
        """A SQLite DB with alembic_version table but no rows is invalid."""
        import sqlite3
        db = tmp_path / "optuna_study.db"
        conn = sqlite3.connect(str(db))
        conn.execute("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)")
        conn.commit()
        conn.close()
        assert _validate_optuna_db(db) is False


# ===================== _pre_init_optuna_storage =====================

class TestPreInitOptunaStorage:
    def test_noop_when_no_db(self, tmp_path):
        """No DB file → pre-init does nothing (sweeper creates it)."""
        exp_name = "experiment_test"
        exp_dir = tmp_path / "runs" / exp_name
        exp_dir.mkdir(parents=True)

        _pre_init_optuna_storage(tmp_path, exp_name)
        # No DB file created — the sweeper handles creation
        assert not (exp_dir / "optuna_study.db").exists()

    def test_preserves_existing_valid_db(self, tmp_path):
        """Calling pre-init on an already-valid DB does not corrupt it."""
        import optuna
        exp_name = "experiment_existing"
        exp_dir = tmp_path / "runs" / exp_name
        exp_dir.mkdir(parents=True)
        db_path = exp_dir / "optuna_study.db"
        storage_url = f"sqlite:///{db_path.as_posix()}"

        # Create a study with one trial
        study = optuna.create_study(storage=storage_url, study_name=exp_name)
        study.add_trial(
            optuna.trial.create_trial(
                params={"x": 1.0},
                distributions={"x": optuna.distributions.UniformDistribution(0, 10)},
                values=[0.5],
            )
        )
        assert _validate_optuna_db(db_path) is True

        # Pre-init again — should not corrupt the DB or lose the trial
        _pre_init_optuna_storage(tmp_path, exp_name)
        assert _validate_optuna_db(db_path) is True

        loaded_study = optuna.load_study(storage=storage_url, study_name=exp_name)
        assert len(loaded_study.trials) == 1

    def test_removes_corrupt_db(self, tmp_path):
        """A 0-byte corrupt DB is cleaned up."""
        exp_name = "experiment_corrupt"
        exp_dir = tmp_path / "runs" / exp_name
        exp_dir.mkdir(parents=True)
        db_path = exp_dir / "optuna_study.db"
        db_path.touch()  # 0-byte corrupt file

        assert _validate_optuna_db(db_path) is False

        _pre_init_optuna_storage(tmp_path, exp_name)

        # Corrupt file should be removed
        assert not db_path.exists()


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
            root / "slurm" / "train_fno_msi.sbatch",
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
