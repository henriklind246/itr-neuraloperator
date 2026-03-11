import pytest
from pathlib import Path

from scripts.run_train import _next_experiment_name, _job_number


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


class TestJobNumber:
    def test_returns_zero_when_hydra_not_initialized(self):
        assert _job_number() == 0
