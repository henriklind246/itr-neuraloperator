"""Warm-start fine-tune plumbing for long-lead OOD adaptation (Capabilities 2, 6, 7).

Exercises the real ``run_one_seed`` on synthetic forcing data:
- weights-only warm-start (``init_from_checkpoint``) with a fresh optimizer
  (new peak LR, epoch 0) and a normalization-stats guard,
- the step-0 reference pass logged as epoch -1 before any optimizer step,
- post-cutoff checkpoint selection on ``val_rmse_K_lead_gt_tc`` plus the
  no-forgetting guard.
"""

import copy
import csv

import numpy as np
import pytest
import torch

from src.operators.train import _selection_metric_value, run_one_seed

_SPATIAL_IN_CHANNELS = 4
_COND_STATIC_DIM = 10
_TEMPORAL_TOKEN_DIM = 2
_TEMPORAL_SAMPLES = 128


@pytest.fixture
def ll_config(tmp_path, synthetic_trajectories, synthetic_sim_params):
    """Minimal forcing config pointing at synthetic data files (see test_train)."""
    trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
    np.save(tmp_path / "trajectories.npy", trajectories)
    np.save(tmp_path / "x_grid.npy", x_grid)
    np.save(tmp_path / "y_grid.npy", y_grid)
    np.save(tmp_path / "t_grid.npy", t_grid)
    np.save(tmp_path / "sim_params.npy", synthetic_sim_params)

    return {
        "data": {
            "trajectories.npy": str(tmp_path / "trajectories.npy"),
            "x_grid_path": str(tmp_path / "x_grid.npy"),
            "y_grid_path": str(tmp_path / "y_grid.npy"),
            "t_grid_path": str(tmp_path / "t_grid.npy"),
            "sim_params_path": str(tmp_path / "sim_params.npy"),
        },
        "model": {
            "parameters": {
                "modes1": 2,
                "modes2": 2,
                "width": 8,
                "in_channels": _SPATIAL_IN_CHANNELS,
                "out_channels": 1,
                "n_layers": 2,
                "cond_static_dim": _COND_STATIC_DIM,
                "cond_hidden": 32,
                "temporal_token_dim": _TEMPORAL_TOKEN_DIM,
                "temporal_samples": _TEMPORAL_SAMPLES,
                "temporal_hidden": 16,
                "forcing_embed_dim": 16,
            }
        },
        "training": {
            "batch_size": 4,
            "n_snapshots": 5,
            "device": "cpu",
            "epochs": 4,
            "learning_rate": 0.001,
            "weight_decay": 1e-5,
            "scheduler": {"step_size": 2, "gamma": 0.5},
            "validate_every": 1,
            "patience": 50,
            "seeds": [0],
        },
    }


def _baseline_best(cfg, run_dir):
    """Run a short baseline and return the path to its fno2d_best.pt."""
    run_one_seed(cfg, seed=0, run_dir=run_dir)
    best = run_dir / "fno2d_best.pt"
    assert best.exists(), "baseline must produce a best checkpoint"
    return best


def _load_state(ckpt_path):
    d = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return d["model_state"]


def _states_equal(sa, sb):
    if set(sa.keys()) != set(sb.keys()):
        return False
    return all(torch.equal(sa[k], sb[k]) for k in sa)


def _val_rows(run_dir, metric_key):
    """CSV rows on real (epoch>=0) validation epochs where `metric_key` is set."""
    with (run_dir / "train_metrics.csv").open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    return [
        r for r in rows
        if int(r["epoch"]) >= 0 and r.get(metric_key, "") not in ("", None)
    ]


class TestSelectionMetricKey:
    def test_post_cutoff_key_maps_to_rmse_K_lead_gt_tc(self):
        vm = {
            "rel_l2": 1.0, "nrmse": 2.0, "node_jump_nrmse": 3.0,
            "rmse_K_lead_gt_tc": 0.5,
        }
        assert _selection_metric_value(vm, "val_rmse_K_lead_gt_tc") == 0.5


class TestWarmStart:
    def test_zero_lr_freezes_weights(self, tmp_path, ll_config):
        """lr=0 warm-start leaves the loaded weights byte-identical to model A."""
        base_best = _baseline_best(ll_config, tmp_path / "baseline")
        base_state = _load_state(base_best)

        warm = copy.deepcopy(ll_config)
        warm["training"]["init_from_checkpoint"] = str(base_best)
        warm["training"]["learning_rate"] = 0.0
        warm_dir = tmp_path / "warm_frozen"
        run_one_seed(warm, seed=0, run_dir=warm_dir)

        warm_state = _load_state(warm_dir / "fno2d_best.pt")
        assert _states_equal(base_state, warm_state), (
            "AdamW with lr=0 must not move warm-started weights"
        )

    def test_fresh_optimizer_uses_new_peak_lr(self, tmp_path, ll_config):
        """The warm optimizer starts at the NEW learning_rate, not A's decayed LR."""
        base_best = _baseline_best(ll_config, tmp_path / "baseline")

        warm = copy.deepcopy(ll_config)
        warm["training"]["init_from_checkpoint"] = str(base_best)
        warm["training"]["learning_rate"] = 0.005
        # Constant LR isolates the peak value from any step-decay timing.
        warm["training"]["scheduler"] = {"type": "StepLR", "step_size": 1, "gamma": 1.0}
        warm_dir = tmp_path / "warm_lr"
        run_one_seed(warm, seed=0, run_dir=warm_dir)

        rows = _val_rows(warm_dir, "lr")
        # Every epoch's logged LR is the fresh peak (gamma=1.0 keeps it flat),
        # never the baseline's decayed 0.001*0.5*... value.
        for r in rows:
            assert float(r["lr"]) == pytest.approx(0.005, rel=1e-6)

    def test_normalization_mismatch_raises(self, tmp_path, ll_config):
        """A checkpoint whose baked stats differ from the recomputed ones fails loud."""
        base_best = _baseline_best(ll_config, tmp_path / "baseline")
        tampered = torch.load(base_best, map_location="cpu", weights_only=False)
        tampered["mu_global"] = float(tampered["mu_global"]) + 999.0
        tampered_path = tmp_path / "tampered.pt"
        torch.save(tampered, tampered_path)

        warm = copy.deepcopy(ll_config)
        warm["training"]["init_from_checkpoint"] = str(tampered_path)
        warm_dir = tmp_path / "warm_mismatch"
        with pytest.raises(ValueError, match="normalization mismatch"):
            run_one_seed(warm, seed=0, run_dir=warm_dir)
        assert not (warm_dir / "fno2d_best.pt").exists()


class TestStepZeroReference:
    def test_reference_row_logged_as_epoch_minus_one(self, tmp_path, ll_config):
        """A pre-finetune validation pass writes an epoch=-1 row with both cutoff aggregates."""
        base_best = _baseline_best(ll_config, tmp_path / "baseline")

        warm = copy.deepcopy(ll_config)
        warm["training"]["init_from_checkpoint"] = str(base_best)
        warm["training"]["lead_cutoff_time"] = 0.5
        warm_dir = tmp_path / "warm_step0"
        run_one_seed(warm, seed=0, run_dir=warm_dir)

        with (warm_dir / "train_metrics.csv").open("r", newline="") as f:
            rows = list(csv.DictReader(f))
        ref = [r for r in rows if int(r["epoch"]) == -1]
        assert len(ref) == 1, "exactly one step-0 reference row (epoch -1) expected"
        r = ref[0]
        for col in (
            "val_rmse_K_lead_le_tc", "val_rmse_K_lead_gt_tc",
            "val_rel_l2_lead_le_tc", "val_rel_l2_lead_gt_tc",
        ):
            assert r[col] not in ("", None), col
            assert np.isfinite(float(r[col])), col

    def test_reference_row_logged_without_a_lead_cutoff(self, tmp_path, ll_config):
        """Every warm start gets the E_0 row; the cutoff only adds the lead split."""
        base_best = _baseline_best(ll_config, tmp_path / "baseline")

        warm = copy.deepcopy(ll_config)
        warm["training"]["init_from_checkpoint"] = str(base_best)
        warm_dir = tmp_path / "warm_step0_nocutoff"
        run_one_seed(warm, seed=0, run_dir=warm_dir)

        with (warm_dir / "train_metrics.csv").open("r", newline="") as f:
            rows = list(csv.DictReader(f))
        ref = [r for r in rows if int(r["epoch"]) == -1]
        assert len(ref) == 1, "exactly one step-0 reference row (epoch -1) expected"
        r = ref[0]
        lead_cols = {
            "val_rmse_K_lead_le_tc", "val_rmse_K_lead_gt_tc",
            "val_rel_l2_lead_le_tc", "val_rel_l2_lead_gt_tc",
        }
        # Every val_* column, not just the headline pair: any of them may be the
        # run's checkpoint_metric, and a reference row missing the selected metric
        # cannot answer "did the fine-tune beat model A" in the units being read.
        val_cols = [c for c in r if c.startswith("val_") and c not in lead_cols]
        assert len(val_cols) > 2, "expected the full val_* block in the header"
        for col in val_cols:
            assert r[col] not in ("", None), col
            assert np.isfinite(float(r[col])), col
        # No cutoff means no lead split to report; the columns stay in the header
        # so the CSV schema does not depend on the run's diagnostics settings.
        for col in sorted(lead_cols):
            assert np.isnan(float(r[col])), col

    def test_no_reference_row_from_a_cold_start(self, tmp_path, ll_config):
        """A random init has no "before" worth measuring, so --scratch has no E_0."""
        cold_dir = tmp_path / "cold"
        run_one_seed(copy.deepcopy(ll_config), seed=0, run_dir=cold_dir)

        with (cold_dir / "train_metrics.csv").open("r", newline="") as f:
            rows = list(csv.DictReader(f))
        assert not [r for r in rows if int(r["epoch"]) == -1]


class TestPostCutoffSelection:
    def test_selects_min_post_cutoff_epoch(self, tmp_path, ll_config):
        """checkpoint_metric=val_rmse_K_lead_gt_tc picks the min post-cutoff RMSE_K epoch."""
        base_best = _baseline_best(ll_config, tmp_path / "baseline")

        warm = copy.deepcopy(ll_config)
        warm["training"]["init_from_checkpoint"] = str(base_best)
        warm["training"]["lead_cutoff_time"] = 0.5
        warm["training"]["checkpoint_metric"] = "val_rmse_K_lead_gt_tc"
        # Loose guard so selection is driven purely by the post-cutoff metric.
        warm["training"]["checkpoint_forgetting_ratio"] = 100.0
        warm_dir = tmp_path / "warm_select"
        result = run_one_seed(warm, seed=0, run_dir=warm_dir)

        ckpt = torch.load(
            warm_dir / "fno2d_best.pt", map_location="cpu", weights_only=False
        )
        best_epoch = int(ckpt["epoch"])
        rows = _val_rows(warm_dir, "val_rmse_K_lead_gt_tc")
        best_metric = min(float(r["val_rmse_K_lead_gt_tc"]) for r in rows)
        chosen = next(r for r in rows if int(r["epoch"]) == best_epoch)
        assert float(chosen["val_rmse_K_lead_gt_tc"]) == pytest.approx(
            best_metric, rel=1e-6
        )
        assert result["best_val"] == pytest.approx(best_metric, rel=1e-6)

    def test_forgetting_guard_admits_frozen_equality(self, tmp_path, ll_config):
        """With frozen weights (lr=0) and ratio=1.0 the guard admits the epoch (<=)."""
        base_best = _baseline_best(ll_config, tmp_path / "baseline")
        base_state = _load_state(base_best)

        warm = copy.deepcopy(ll_config)
        warm["training"]["init_from_checkpoint"] = str(base_best)
        warm["training"]["learning_rate"] = 0.0
        warm["training"]["lead_cutoff_time"] = 0.5
        warm["training"]["checkpoint_metric"] = "val_rmse_K_lead_gt_tc"
        warm["training"]["checkpoint_forgetting_ratio"] = 1.0
        warm_dir = tmp_path / "warm_guard"
        run_one_seed(warm, seed=0, run_dir=warm_dir)

        best = warm_dir / "fno2d_best.pt"
        assert best.exists(), "equality (pre_now == ref) must satisfy ratio=1.0 guard"
        assert _states_equal(base_state, _load_state(best))
