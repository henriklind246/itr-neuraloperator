"""End-to-end wiring for the five PINO-training knobs (Step 7).

Exercises the real ``run_one_seed`` on synthetic forcing data through the
generic collocation physics-only path with every technique switched on at once
(causal weighting + GradNorm + hard right-Dirichlet BC + SOAP + IC-anchoring
warm-up; ``stratify_leads`` is on-grid-only so forcing uses
``collocation_lead_bins`` for lead coverage instead). Asserts:

- the new per-epoch CSV columns are populated (causal interior residual, causal
  weights/eps JSON, GradNorm weights JSON, per-phase timings),
- both checkpoints carry ``causal_state`` / ``gradnorm_state`` /
  ``global_optimizer_step``,
- an auto-resume restores the causal weighter's committed EMA,
- and a no-op guard: with every knob at its OFF default the first-epoch
  ``model_state`` is bit-identical to a run that omits the knobs entirely, so the
  disabled path stays byte-for-byte the baseline.
"""

import copy
import csv
import json

import torch

from src.operators.train import run_one_seed
import src.operators.train as train_mod

_SPATIAL_IN_CHANNELS = 4
_COND_STATIC_DIM = 11
_TEMPORAL_TOKEN_DIM = 2
_TEMPORAL_SAMPLES = 128


def _base_config(tmp_path, synthetic_trajectories, synthetic_sim_params):
    """Minimal forcing config pointing at synthetic data files (see test_train)."""
    import numpy as np

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
            "epochs": 2,
            "learning_rate": 0.001,
            "weight_decay": 1e-5,
            "scheduler": {"type": "StepLR", "step_size": 2, "gamma": 0.5},
            "validate_every": 1,
            "patience": 50,
            "seeds": [0],
        },
    }


def _all_knobs_on(cfg):
    """Turn on all five techniques on the generic (forcing) collocation path."""
    cfg = copy.deepcopy(cfg)
    cfg["training"]["optimizer"] = "SOAP"
    cfg["training"]["soap"] = {
        "betas": [0.95, 0.95],
        "shampoo_beta": 0.95,
        "eps": 1.0e-8,
        # Small so a 2-epoch run exercises the preconditioner-update path.
        "precondition_frequency": 3,
        "max_precond_dim": 10000,
        "merge_dims": False,
        "precondition_1d": False,
    }
    cfg["training"]["gradnorm"] = {
        "enabled": True,
        "alpha_w": 0.9,
        "update_every": 1,
        "eps": 1.0e-8,
    }
    cfg["model"]["parameters"]["hard_right_dirichlet"] = True
    cfg["model"]["parameters"]["hard_right_dirichlet_t_right"] = 300.0
    cfg["training"]["physics"] = {
        "lambda_data": 0.0,
        "lambda_physics": 1.0,
        "lambda_ic": 1.0,
        "mode": "collocation",
        "collocation_source": "train",
        "dt": 0.005,
        # Open the full lead window (no staged curriculum); forcing gets lead
        # coverage from collocation_lead_bins (stratify_leads is on-grid-only).
        "collocation_lead_start": 0.1,
        "collocation_lead_max": 0.1,
        "collocation_lead_bins": 4,
        "causal_curriculum": {"enabled": False},
        "causal_weighting": {"enabled": True, "n_bins": 4, "update_every": 1},
        # Tiny warm-up so global_optimizer_step / the freeze cache are exercised.
        "warmup": {"enabled": True, "steps": 2, "resample_every": 2, "ic_multiplier": 5.0},
        "full_bc_region_weights": {
            "interior": 1.0,
            "left_neumann": 1.0,
            "right_dirichlet": 1.0,
            "topbot_adiabatic": 1.0,
        },
    }
    return cfg


def _rows(run_dir):
    with (run_dir / "train_metrics.csv").open("r", newline="") as f:
        return list(csv.DictReader(f))


def _real_rows(run_dir):
    """Real training rows (epoch >= 0)."""
    return [r for r in _rows(run_dir) if int(r["epoch"]) >= 0]


def _load_state(ckpt_path):
    d = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return d["model_state"]


def _states_equal(sa, sb):
    if set(sa.keys()) != set(sb.keys()):
        return False
    return all(torch.equal(sa[k], sb[k]) for k in sa)


class TestAllKnobsCsvColumns:
    def test_new_columns_populated(
        self, tmp_path, synthetic_trajectories, synthetic_sim_params
    ):
        cfg = _all_knobs_on(
            _base_config(tmp_path, synthetic_trajectories, synthetic_sim_params)
        )
        run_dir = tmp_path / "all_on"
        run_one_seed(cfg, seed=0, run_dir=run_dir)

        rows = _real_rows(run_dir)
        assert rows, "expected at least one real (epoch >= 0) training row"
        for r in rows:
            # Causal weighting diagnostics.
            assert r["train_physics_causal_interior"] not in ("", None)
            assert float(r["train_physics_causal_interior"]) >= 0.0
            weights = json.loads(r["causal_weights"])
            assert len(weights) == 4  # n_bins
            assert weights[0] == 1.0  # bin 0 weight is exp(0) == 1
            assert r["causal_eps"] not in ("", None)
            assert float(r["causal_eps"]) > 0.0
            # GradNorm multipliers (JSON dict over the active raw terms).
            gn = json.loads(r["gradnorm_weights"])
            assert isinstance(gn, dict) and gn
            # Per-phase timings (perf_counter based -> populated on CPU).
            for col in ("train_step_ms", "backward_ms", "optimizer_step_ms", "gradnorm_ms"):
                assert r[col] not in ("", None), col
                assert float(r[col]) >= 0.0
            assert float(r["gradnorm_ms"]) > 0.0  # GradNorm ran this step


class TestCheckpointState:
    def test_best_checkpoint_carries_causal_and_gradnorm_state(
        self, tmp_path, synthetic_trajectories, synthetic_sim_params
    ):
        # A completed run deletes the fno2d_latest.pt resume sentinel, so only
        # fno2d_best.pt survives; it carries the new state keys (train.py:3945).
        cfg = _all_knobs_on(
            _base_config(tmp_path, synthetic_trajectories, synthetic_sim_params)
        )
        run_dir = tmp_path / "ckpt_state"
        run_one_seed(cfg, seed=0, run_dir=run_dir)

        ckpt = torch.load(
            run_dir / "fno2d_best.pt", map_location="cpu", weights_only=False
        )
        assert ckpt["causal_state"] is not None
        assert "ema" in ckpt["causal_state"]
        assert ckpt["causal_state"]["n_bins"] == 4
        assert ckpt["gradnorm_state"] is not None
        assert "multipliers" in ckpt["gradnorm_state"]
        assert isinstance(ckpt["global_optimizer_step"], int)


class TestResumeRestoresWeighter:
    def test_resume_loads_committed_ema(
        self, tmp_path, synthetic_trajectories, synthetic_sim_params, monkeypatch
    ):
        base = _base_config(tmp_path, synthetic_trajectories, synthetic_sim_params)
        cfg = _all_knobs_on(base)
        run_dir = tmp_path / "resume"
        run_dir.mkdir()

        # First stint: 2 epochs -> writes fno2d_best.pt with a committed EMA.
        run_one_seed(cfg, seed=0, run_dir=run_dir)
        best = torch.load(
            run_dir / "fno2d_best.pt", map_location="cpu", weights_only=False
        )
        saved_ema = torch.as_tensor(best["causal_state"]["ema"], dtype=torch.float64)

        # A completed run removes the sentinel; simulate an interruption by
        # recreating fno2d_latest.pt from the best checkpoint (see test_train.py).
        torch.save(best, run_dir / "fno2d_latest.pt")

        # Spy on the weighter's load to capture exactly what the resume restores.
        captured = {}
        orig_load = train_mod.CausalLeadWeighter.load_state_dict

        def spy_load(self, state):
            captured["ema"] = torch.as_tensor(state["ema"], dtype=torch.float64)
            return orig_load(self, state)

        monkeypatch.setattr(
            train_mod.CausalLeadWeighter, "load_state_dict", spy_load
        )

        # Second stint on the same run dir auto-resumes (fno2d_latest.pt exists)
        # and must load_state_dict the saved causal state.
        resume_cfg = copy.deepcopy(cfg)
        resume_cfg["training"]["epochs"] = 4
        run_one_seed(resume_cfg, seed=0, run_dir=run_dir)

        assert "ema" in captured, "resume must call CausalLeadWeighter.load_state_dict"
        assert torch.equal(captured["ema"], saved_ema), (
            "resume must restore the committed causal EMA byte-for-byte"
        )


class TestDisabledPathBitIdentical:
    def test_all_off_matches_omitted_knobs(
        self, tmp_path, synthetic_trajectories, synthetic_sim_params
    ):
        """The five new knobs at their OFF defaults produce the same first-epoch
        weights as a config that never mentions them, so Steps 0-6 leave the
        disabled path byte-for-byte the baseline.

        Both runs share an identical (data-only) physics block so the variable
        under test is ONLY the new knobs -- not the presence of a physics block,
        which is a pre-existing config axis that perturbs training at ~1e-5.
        """
        base = _base_config(tmp_path, synthetic_trajectories, synthetic_sim_params)
        base["training"]["epochs"] = 1
        shared_physics = {
            "lambda_data": 1.0,
            "lambda_physics": 0.0,
            "lambda_ic": 0.0,
            "mode": "one_step",
        }

        # Run A: constant physics block, none of the new knobs mentioned.
        run_a_cfg = copy.deepcopy(base)
        run_a_cfg["training"]["physics"] = copy.deepcopy(shared_physics)
        run_a = tmp_path / "off_omitted"
        run_one_seed(run_a_cfg, seed=0, run_dir=run_a)

        # Run B: same physics block + every new knob present at its OFF default.
        # (optimizer is left at the shared base default -- SOAP is the new
        # optimizer knob, but AdamW/Adam selection is a pre-existing axis, so
        # forcing it here would itself perturb the baseline.)
        off = copy.deepcopy(base)
        off["training"]["gradnorm"] = {"enabled": False}
        off["model"]["parameters"]["hard_right_dirichlet"] = False
        off["training"]["physics"] = {
            **copy.deepcopy(shared_physics),
            "causal_weighting": {"enabled": False},
            "warmup": {"enabled": False},
            "stratify_leads": False,
        }
        run_b = tmp_path / "off_explicit"
        run_one_seed(off, seed=0, run_dir=run_b)

        assert _states_equal(
            _load_state(run_a / "fno2d_best.pt"),
            _load_state(run_b / "fno2d_best.pt"),
        ), "disabled knobs must not perturb the baseline first-epoch weights"
