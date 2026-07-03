"""Tests for the benchmark-aware val_pairs.csv inspection script.

The script consumes the single superset CSV written by training and routes
to per-benchmark reports/plots based on the ``benchmark`` column (with config
and column-inference fallbacks). These tests exercise benchmark detection and
the end-to-end dispatch that produces the expected PNG set per benchmark.
"""

import csv

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from scripts import inspect_val_pairs as ivp
from src.operators.train import VAL_PAIR_FIELDNAMES

# Universal plots emitted for every benchmark, plus the benchmark-specific 5th.
UNIVERSAL_PNGS = {
    "01_training_dynamics.png",
    "02_regime_stratification.png",
    "03_error_vs_lead_time.png",
    "04_interface_vs_bulk.png",
    "06_lead_binned_rmse_k.png",
}
STRUCTURE_PNG = {
    "forcing": "05_ic_family_heatmap.png",
    "interfaces": "05_error_vs_interface_x.png",
    "source": "05_source_structure.png",
}


def _write_csv(path, benchmark, *, include_benchmark_col=True):
    """Write a synthetic val_pairs.csv with the production superset header.

    Only the columns the given benchmark actually populates are filled; every
    other benchmark's columns are left empty, mirroring train.py's writer.
    """
    rng = np.random.default_rng(0)
    n_sims, n_pairs, epochs = 6, 5, (1, 2)
    rows = []
    for epoch in epochs:
        for sim_id in range(n_sims):
            for _ in range(n_pairs):
                row = {k: "" for k in VAL_PAIR_FIELDNAMES}
                if include_benchmark_col:
                    row["benchmark"] = benchmark
                row["epoch"] = epoch
                row["sim_id"] = sim_id
                row["t_s"] = float(rng.uniform(0.0, 0.5))
                row["t_bar"] = float(rng.uniform(0.0, 1.0))
                row["R_c"] = float(rng.uniform(0.05, 1.0))
                row["rel_l2"] = float(rng.uniform(0.01, 0.5))
                row["iface_rel_l2"] = float(rng.uniform(0.01, 0.5))
                if benchmark == "forcing":
                    row["temporal_family"] = rng.choice(["sin", "exp", "step"])
                    row["spatial_family"] = rng.choice(["uniform", "gauss"])
                elif benchmark == "interfaces":
                    row["interface_x"] = float(rng.uniform(0.2, 0.8))
                elif benchmark == "source":
                    row["x_h"] = float(rng.uniform(0.1, 0.9))
                    row["y_h"] = float(rng.uniform(0.1, 0.9))
                    row["A"] = float(rng.uniform(1000.0, 5000.0))
                    row["regime"] = rng.choice(["left", "near", "right"])
                rows.append(row)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=VAL_PAIR_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return path


# ===================== benchmark detection =====================

class TestDetectBenchmark:
    @pytest.mark.parametrize("benchmark", ["forcing", "interfaces", "source"])
    def test_from_benchmark_column(self, tmp_path, benchmark):
        import pandas as pd

        path = _write_csv(tmp_path / "val_pairs.csv", benchmark)
        df = pd.read_csv(path)
        assert ivp._detect_benchmark(df, path) == benchmark

    def test_from_config_when_column_missing(self, tmp_path):
        import pandas as pd

        path = _write_csv(tmp_path / "val_pairs.csv", "source", include_benchmark_col=False)
        (tmp_path / "config_used.yaml").write_text(
            "benchmark:\n  name: interfaces\ntraining:\n  epochs: 1\n"
        )
        df = pd.read_csv(path)
        # Source-style columns are populated, but the config wins over inference.
        assert ivp._detect_benchmark(df, path) == "interfaces"

    def test_column_inference_fallback(self, tmp_path):
        import pandas as pd

        path = _write_csv(tmp_path / "val_pairs.csv", "source", include_benchmark_col=False)
        df = pd.read_csv(path)
        assert ivp._detect_benchmark(df, path) == "source"


# ===================== conditioning columns =====================

class TestCondCols:
    def test_forcing_uses_base_only(self, tmp_path):
        import pandas as pd

        df = pd.read_csv(_write_csv(tmp_path / "v.csv", "forcing"))
        assert ivp._cond_cols(df, "forcing") == ["t_s", "t_bar", "R_c"]

    def test_interfaces_appends_interface_x(self, tmp_path):
        import pandas as pd

        df = pd.read_csv(_write_csv(tmp_path / "v.csv", "interfaces"))
        assert ivp._cond_cols(df, "interfaces") == ["t_s", "t_bar", "R_c", "interface_x"]

    def test_source_appends_patch_cols(self, tmp_path):
        import pandas as pd

        df = pd.read_csv(_write_csv(tmp_path / "v.csv", "source"))
        assert ivp._cond_cols(df, "source") == ["t_s", "t_bar", "R_c", "x_h", "y_h", "A"]


# ===================== end-to-end dispatch =====================

class TestMainDispatch:
    @pytest.mark.parametrize("benchmark", ["forcing", "interfaces", "source"])
    def test_produces_expected_pngs(self, tmp_path, monkeypatch, capsys, benchmark):
        csv_path = _write_csv(tmp_path / "val_pairs.csv", benchmark)
        out = tmp_path / "plots"
        monkeypatch.setattr(
            "sys.argv",
            ["inspect_val_pairs.py", str(csv_path), "--out", str(out)],
        )
        ivp.main()

        produced = {p.name for p in out.glob("*.png")}
        expected = UNIVERSAL_PNGS | {STRUCTURE_PNG[benchmark]}
        assert produced == expected
        # The report header must name the detected benchmark.
        assert f"Benchmark: {benchmark}" in capsys.readouterr().out
