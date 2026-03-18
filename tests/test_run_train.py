import csv
import json
import os

import pytest
import yaml
from pathlib import Path

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
    _stamp_optuna_alembic,
    _pre_init_optuna_storage,
    _config_sort_key,
    EXPERIMENT_ENV_VAR,
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


# ===================== _stamp_optuna_alembic =====================

class TestStampOptunaAlembic:
    def test_stamps_fresh_db_with_tables(self, tmp_path):
        """After creating tables with SQLAlchemy, alembic stamp fills alembic_version."""
        import sqlalchemy as sa
        from optuna.storages._rdb import models

        db = tmp_path / "optuna_study.db"
        engine = sa.create_engine(f"sqlite:///{db.as_posix()}")
        models.BaseModel.metadata.create_all(engine)
        engine.dispose()

        # Before stamp: DB exists but has no alembic_version table
        assert _validate_optuna_db(db) is False

        # Stamp it
        assert _stamp_optuna_alembic(db) is True
        assert _validate_optuna_db(db) is True

    def test_idempotent_on_already_stamped_db(self, tmp_path):
        """Stamping an already-stamped DB is a no-op."""
        import optuna
        db = tmp_path / "optuna_study.db"
        url = f"sqlite:///{db.as_posix()}"
        optuna.create_study(storage=url, study_name="test")
        assert _validate_optuna_db(db) is True

        # Stamp again — should succeed and remain valid
        assert _stamp_optuna_alembic(db) is True
        assert _validate_optuna_db(db) is True


# ===================== _pre_init_optuna_storage =====================

class TestPreInitOptunaStorage:
    def test_creates_valid_db_from_scratch(self, tmp_path):
        """Fresh experiment dir → DB created and valid."""
        exp_name = "experiment_test"
        exp_dir = tmp_path / "runs" / exp_name
        exp_dir.mkdir(parents=True)

        _pre_init_optuna_storage(tmp_path, exp_name)

        db_path = exp_dir / "optuna_study.db"
        assert db_path.exists()
        assert _validate_optuna_db(db_path) is True

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
                distributions={"x": optuna.distributions.FloatDistribution(0, 10)},
                values=[0.5],
            )
        )
        assert _validate_optuna_db(db_path) is True

        # Pre-init again — should not corrupt the DB or lose the trial
        _pre_init_optuna_storage(tmp_path, exp_name)
        assert _validate_optuna_db(db_path) is True

        loaded_study = optuna.load_study(storage=storage_url, study_name=exp_name)
        assert len(loaded_study.trials) == 1

    def test_handles_corrupt_db(self, tmp_path):
        """A 0-byte corrupt DB is either replaced with a valid one or cleaned up."""
        exp_name = "experiment_corrupt"
        exp_dir = tmp_path / "runs" / exp_name
        exp_dir.mkdir(parents=True)
        db_path = exp_dir / "optuna_study.db"
        db_path.touch()  # 0-byte corrupt file

        assert _validate_optuna_db(db_path) is False

        _pre_init_optuna_storage(tmp_path, exp_name)

        # Newer Optuna versions overwrite the corrupt file successfully;
        # older versions fail, print diagnostics, and attempt cleanup.
        # Either outcome is acceptable.
        if db_path.exists():
            assert _validate_optuna_db(db_path) is True  # overwritten with valid DB
        # else: file was cleaned up after failure — also fine
