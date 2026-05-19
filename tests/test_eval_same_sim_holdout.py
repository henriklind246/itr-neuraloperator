import subprocess
from pathlib import Path

from scripts import eval_same_sim_holdout

SPATIAL_IN_CHANNELS = 20
COND_STATIC_DIM = 23
TEMPORAL_TOKEN_DIM = 5
TEMPORAL_SAMPLES = 64


def _base_config(tmp_path):
    return {
        "paths": {"runs_root": str(tmp_path / "runs"), "data_dir": "unused"},
        "experiment": {"name": "placeholder"},
        "config_id": 99,
        "data": {
            "trajectories.npy": "unused",
            "x_grid_path": "unused",
            "y_grid_path": "unused",
            "t_grid_path": "unused",
            "sim_params_path": "unused",
        },
        "model": {
            "parameters": {
                "modes1": 2,
                "modes2": 2,
                "width": 8,
                "in_channels": SPATIAL_IN_CHANNELS,
                "out_channels": 1,
                "n_layers": 2,
                "cond_static_dim": COND_STATIC_DIM,
                "cond_hidden": 32,
                "temporal_token_dim": TEMPORAL_TOKEN_DIM,
                "temporal_samples": TEMPORAL_SAMPLES,
                "temporal_hidden": 16,
                "forcing_embed_dim": 16,
            }
        },
        "training": {
            "device": "cpu",
            "batch_size": 32,
            "epochs": 3,
            "seeds": [0],
            "run": {"run_dir": "placeholder"},
            "learning_rate": 0.001,
            "weight_decay": 1e-5,
            "scheduler": {"step_size": 2, "gamma": 0.5},
            "validate_every": 1,
            "patience": 10,
        },
    }


def test_same_sim_main_accepts_fixed_training_style_overrides(tmp_path, monkeypatch, capsys):
    calls = {}

    def fake_run_one_seed(config, seed, run_dir, *, train_loader_override, val_loader_override):
        calls["config"] = config
        calls["seed"] = seed
        calls["run_dir"] = Path(run_dir)
        calls["train_loader_override"] = train_loader_override
        calls["val_loader_override"] = val_loader_override
        return {"best_val": 0.123, "best_path": str(Path(run_dir) / "fno2d_best.pt"), "seed": seed}

    monkeypatch.setattr(
        "sys.argv",
        [
            "eval_same_sim_holdout.py",
            "experiment.name=2d_fno_same_sim",
            "config_id=0",
            "training.epochs=51",
            "training.seeds=[42]",
            "training.batch_size=512",
        ],
    )

    eval_same_sim_holdout.main(
        load_config_fn=lambda _path=None: _base_config(tmp_path),
        build_loaders_fn=lambda config, val_pair_frac, split_seed: ("train", "val"),
        run_one_seed_fn=fake_run_one_seed,
    )
    captured = capsys.readouterr()

    assert calls["seed"] == 42
    assert calls["run_dir"] == tmp_path / "runs" / "2d_fno_same_sim" / "config0" / "seed42"
    assert calls["config"]["training"]["epochs"] == 51
    assert calls["config"]["training"]["batch_size"] == 512
    assert calls["config"]["training"]["run"]["run_dir"] == str(tmp_path / "runs" / "2d_fno_same_sim" / "config0")
    assert calls["train_loader_override"] == "train"
    assert calls["val_loader_override"] == "val"
    assert "Using same-sim experiment namespace: 2d_fno_same_sim" in captured.out
    assert "same-sim held-out best_val_rel_l2: 0.123000" in captured.out


def test_same_sim_main_seed_flag_overrides_config_seed(tmp_path, monkeypatch):
    calls = {}

    def fake_run_one_seed(config, seed, run_dir, *, train_loader_override, val_loader_override):
        calls["payload"] = (seed, Path(run_dir))
        return {"best_val": 0.0, "best_path": str(Path(run_dir) / "fno2d_best.pt"), "seed": seed}

    monkeypatch.setattr(
        "sys.argv",
        [
            "eval_same_sim_holdout.py",
            "--seed",
            "7",
            "experiment.name=2d_fno_same_sim",
            "config_id=3",
            "training.seeds=[42]",
        ],
    )

    eval_same_sim_holdout.main(
        load_config_fn=lambda _path=None: _base_config(tmp_path),
        build_loaders_fn=lambda config, val_pair_frac, split_seed: ("train", "val"),
        run_one_seed_fn=fake_run_one_seed,
    )

    seed, run_dir = calls["payload"]
    assert seed == 7
    assert run_dir == tmp_path / "runs" / "2d_fno_same_sim" / "config3" / "seed7"


def test_same_sim_paths_data_dir_override_updates_dataset_paths(tmp_path, monkeypatch):
    calls = {}
    data_dir = tmp_path / "custom_data"

    def fake_run_one_seed(config, seed, run_dir, *, train_loader_override, val_loader_override):
        calls["config"] = config
        return {"best_val": 0.0, "best_path": str(Path(run_dir) / "fno2d_best.pt"), "seed": seed}

    monkeypatch.setattr(
        "sys.argv",
        [
            "eval_same_sim_holdout.py",
            "experiment.name=2d_fno_same_sim",
            "config_id=0",
            f"paths.data_dir={data_dir}",
        ],
    )

    eval_same_sim_holdout.main(
        load_config_fn=lambda _path=None: _base_config(tmp_path),
        build_loaders_fn=lambda config, val_pair_frac, split_seed: ("train", "val"),
        run_one_seed_fn=fake_run_one_seed,
    )

    assert calls["config"]["data"]["t_grid_path"] == str(data_dir / "t_grid.npy")
    assert calls["config"]["data"]["trajectories.npy"] == str(data_dir / "trajectories.npy")
    assert calls["config"]["data"]["sim_params_path"] == str(data_dir / "sim_params.npy")


def test_same_sim_slurm_script_has_valid_syntax_and_forwards_overrides():
    script = Path(__file__).resolve().parents[1] / "slurm" / "eval_same_sim_holdout_msi.sbatch"

    subprocess.run(["bash", "-n", str(script)], check=True)
    text = script.read_text(encoding="utf-8")

    assert "python scripts/eval_same_sim_holdout.py" in text
    assert 'cmd+=("$@")' in text
    assert 'cmd+=(--seed "$SAME_SIM_SEED")' in text
    assert 'SAME_SIM_SEED="${SAME_SIM_SEED:-}"' in text
    assert 'export DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"' in text
