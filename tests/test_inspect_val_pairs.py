"""Tests for the benchmark-aware val_pairs.csv inspection script.

The script consumes the single superset CSV written by training and routes
to per-benchmark reports/plots based on the ``benchmark`` column (with config
and column-inference fallbacks). These tests exercise benchmark detection and
the end-to-end dispatch that produces the expected PNG set per benchmark.
"""

import csv

import matplotlib
import numpy as np
import pandas as pd
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


def test_regime_plot_handles_a_sparse_five_step_rollout(tmp_path):
    latest = pd.DataFrame({"t_bar": np.arange(6) * .005,
                           "rel_l2": [0, .8, 1.1, 1.5, 1.2, 1.0]})
    ivp.plot_regime_stratification(latest, ["t_bar"], tmp_path)
    assert (tmp_path / "02_regime_stratification.png").is_file()


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


class TestSimulationTailSummary:
    def test_reduces_pairs_within_simulation_before_tail_statistics(self):
        df = pd.DataFrame({
            "sim_id": [0, 0, 0, 1],
            "rel_l2": [1.0, 1.0, 10.0, 5.0],
        })
        rows = ivp.simulation_tail_rows(df, "interfaces", [])
        row = next(r for r in rows if r["metric"] == "rel_l2")

        assert row["n_sims"] == 2
        assert row["n_pairs"] == 4
        assert row["mean"] == pytest.approx(((1.0 + 1.0 + 10.0) / 3.0 + 5.0) / 2.0)
        assert np.isnan(row["p99"])
        assert row["p99_determined"] is False

    def test_categorical_and_numeric_strata_are_exported(self):
        df = pd.DataFrame({
            "sim_id": np.repeat(np.arange(10), 2),
            "rel_l2": np.linspace(0.1, 2.0, 20),
            "temporal_family": np.tile(["sin", "exp"], 10),
            "t_bar": np.linspace(0.01, 0.3, 20),
        })
        rows = ivp.simulation_tail_rows(df, "forcing", ["t_bar"])
        assert {r["stratum"] for r in rows} == {
            "overall", "temporal_family", "t_bar_quintile",
        }


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
        tail = pd.read_csv(out / "tail_summary_sim.csv")
        assert list(tail.columns) == list(ivp.TAIL_SUMMARY_FIELDS)
        assert set(tail["benchmark"]) == {benchmark}
        assert int(tail.loc[tail["stratum"] == "overall", "n_sims"].iloc[0]) == 6
        # The report header must name the detected benchmark.
        assert f"Benchmark: {benchmark}" in capsys.readouterr().out


def test_homogeneous_cvit_stratification_without_interface_columns(tmp_path,capsys):
    rows=[]
    for sid,family in enumerate(('uniform_2d','grf_2d')):
        for time in (0.005,0.05,0.1,0.15,0.2,0.3):
            rows.append(dict(epoch=100,sim_id=sid,benchmark='diffusion_forcing_single',
                             t_s=0,t_bar=time,absolute_time=time,ic_family=family,
                             amplitude=100+sid*100,frequency=2+sid*5,
                             rel_l2=10+time,rmse_K=1+time,sigma_nrmse_pct=5+time))
    df=pd.DataFrame(rows)
    columns=ivp._cond_cols(df,'diffusion_forcing_single')
    assert {'amplitude','frequency','absolute_time'} <= set(columns)
    ivp.report_structure(df,'diffusion_forcing_single')
    assert 'grf_2d' in capsys.readouterr().out
    ivp.plot_structure(df,'diffusion_forcing_single',tmp_path)
    ivp.plot_interface_ratio(df,tmp_path)
    assert (tmp_path/'05_ic_family_errors.png').exists()
    assert not (tmp_path/'04_interface_vs_bulk.png').exists()
