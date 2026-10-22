import csv

import numpy as np
import pytest
import yaml

from data.dataset import problem_from_config
import scripts.diagnose_space_time_conditioning as conditioning
from src.operators.train import load_config


MG = dict(pre_smooth=2, post_smooth=2, omega=2 / 3, coarse_max_unknowns=16)
# Accelerate BLAS raises spurious floating-point flags inside matmul on macOS.
pytestmark = pytest.mark.filterwarnings("ignore:.*encountered in matmul:RuntimeWarning")


@pytest.fixture
def config(monkeypatch):
    monkeypatch.setenv("BENCHMARK", "forcing_homogeneous")
    cfg = load_config()
    cfg["physics_test"]["conditioning"].update(
        grid_sizes=[8], horizons=[1, 3, 5], benchmark_grid_horizons=[5],
        truncation_lags=[1, 2], validation_grid_size=8, validation_horizon=4, modes=2,
        max_tolerance=1e-10, min_tolerance=1e-10,
        benchmark_grid_configurations=["A_none", "B_spatial_mg", "C_spatial_exact",
                                       "D_truncated_L1", "D_sequential_mg", "exact_space_time"])
    cfg["physics_test"]["mg"]["solver"] = MG
    cfg["physics_test"]["pino"]["prefix_steps"] = 5
    return cfg


def benchmark(config, grid):
    spec = problem_from_config(config)
    return conditioning.build_benchmark(spec, grid, config["physics_test"]["conditioning"])


def dense_operator(name, blocks, steps):
    operator = conditioning.space_time_operator(name, blocks, steps)
    size = operator.size
    return operator.apply(np.eye(size).reshape(steps, operator.n, size)).reshape(size, size)


@pytest.mark.parametrize("builder", [conditioning.dense_blocks, conditioning.operator_blocks])
def test_blocks_follow_the_benchmark_solver_and_training_modules(config, builder):
    solver, geom, cfg = benchmark(config, 8)
    assert (cfg["k"], cfg["rho"], cfg["cp"], cfg["dt"]) == (1.0, 1.0, 1.0, 0.005)
    audit = conditioning.audit_blocks(solver, geom, builder(geom, MG), MG, 0)
    assert max(value for key, value in audit.items() if key.endswith("_error")) < 1e-10
    assert audit["mg_adjoint_dot_test"] < 1e-12
    assert len(audit["mg_level_shapes"]) == 3


def test_space_time_operators_match_dense_products_of_training_modules(config):
    study = config["physics_test"]["conditioning"]
    result = conditioning.dense_validation(problem_from_config(config), study, MG, [1, 2])
    assert result["passed"], result
    assert set(result["configurations"]) == set(conditioning.configurations([1, 2]))
    # D_sequential_mg is measured on the training module and its training-loss Hessian.
    assert "D_sequential_mg" in result["loss_hessian_max_error"]
    assert max(result["loss_hessian_max_error"].values()) < 1e-10
    assert result["configurations"]["exact_space_time"]["dense_condition_H"] == pytest.approx(1)


def test_truncated_exact_preconditioner_leaves_only_the_next_lag(config):
    _, geom, _ = benchmark(config, 7)
    blocks = conditioning.dense_blocks(geom, MG)
    steps, n = 6, blocks["A"].shape[0]
    for lag in (1, 2, 5):
        expected = np.eye(steps * n) - np.kron(np.eye(steps, k=-(lag + 1)),
                                               np.linalg.matrix_power(blocks["K"], lag + 1))
        np.testing.assert_allclose(dense_operator(f"D_truncated_L{lag}", blocks, steps), expected,
                                   rtol=0, atol=1e-12)
    np.testing.assert_allclose(dense_operator("exact_space_time", blocks, steps),
                               np.eye(steps * n), rtol=0, atol=1e-12)


def test_energy_model_is_exact_in_the_capacity_weighted_norm(config):
    _, geom, _ = benchmark(config, 8)
    blocks = conditioning.dense_blocks(geom, MG)
    rates = conditioning.spatial_modes(geom)["rates"]
    steps = 7
    root = np.kron(np.ones(steps), np.sqrt(blocks["capacity"]))
    for name in ("A_none", "C_spatial_exact", "D_truncated_L1", "D_truncated_L2"):
        weighted = root[:, None] * dense_operator(name, blocks, steps) / root[None, :]
        singular = np.linalg.svd(weighted, compute_uv=False)
        low, high = conditioning.energy_model(name, rates, geom.dt, steps)
        assert low == pytest.approx(singular[-1], rel=1e-9)
        assert high == pytest.approx(singular[0], rel=1e-9)
    assert conditioning.energy_model("B_spatial_mg", rates, geom.dt, steps) is None


def test_bidiagonal_singular_values_match_closed_forms():
    length = 9
    unit = 2 * np.sin((2 * np.arange(1, length + 1) - 1) * np.pi / (2 * (2 * length + 1)))
    np.testing.assert_allclose(conditioning.bidiagonal_singular_values(1.0, length), unit, rtol=1e-12)
    matrix = np.eye(length) + 0.7 * np.eye(length, k=-1)
    np.testing.assert_allclose(conditioning.bidiagonal_singular_values(-0.7, length),
                               np.sort(np.linalg.svd(matrix, compute_uv=False)), rtol=1e-12)


def test_study_writes_artifacts_on_both_backends(config, tmp_path, monkeypatch):
    original = conditioning.build_benchmark
    monkeypatch.setattr(conditioning, "build_benchmark",
                        lambda spec, grid, study: original(spec, 9 if grid is None else grid, study))
    output = tmp_path / "run"
    summary = conditioning.run_study(config, output)
    assert summary["validation_passed"]
    frozen = yaml.safe_load((output / "config_used.yaml").read_text())
    assert (frozen["benchmark"], frozen["benchmark_prefix_steps"], frozen["benchmark_grid_size"]) == (
        "forcing_homogeneous", 5, 9)
    with (output / "conditioning.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert {(row["grid_size"], row["backend"]) for row in rows} == {
        ("8", "dense_blocks"), ("9", "operator_blocks")}
    assert len(rows) == 3 * len(conditioning.configurations([1, 2])) + 6
    for row in rows:
        if row["lambda_min_residual"]:
            assert float(row["condition_H"]) > 1.0
            assert float(row["lambda_min_residual"]) < 1e-8 * float(row["lambda_min"])
        else:
            assert float(row["identity_max_deviation"]) <= conditioning.IDENTITY_TOLERANCE
            assert float(row["condition_H"]) == 1.0
    assert {row["steps"] for row in rows if row["configuration"] == "C_spatial_exact"
            and not row["lambda_min_residual"]} == {"1"}
    assert all(not row["lambda_min_residual"] for row in rows
               if row["configuration"] == "exact_space_time")
    for name in ("audit.json", "final_metrics.json", "modes.csv", "modes.npz",
                 "conditioning_vs_horizon.png", "conditioning_vs_grid_steps5.png",
                 "softest_modes_grid8_steps5.png", "softest_modes_grid9_steps5.png"):
        assert (output / name).exists()
    assert set(summary["prefix_condition_H"]) == {"8", "9"}


def test_study_rejects_layered_benchmarks_and_missing_prefix(config, tmp_path):
    config["physics_test"]["pino"]["prefix_steps"] = 4
    with pytest.raises(ValueError, match="prefix"):
        conditioning.run_study(config, tmp_path / "prefix")
    config["benchmark"]["name"] = "forcing"
    with pytest.raises(ValueError, match="homogeneous"):
        conditioning.run_study(config, tmp_path / "layered")
    assert not (tmp_path / "prefix").exists() and not (tmp_path / "layered").exists()
