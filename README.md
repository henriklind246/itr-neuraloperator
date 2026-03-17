# itr-neuraloperator

Repository for modeling interfacial thermal resistance (ITR) and multilayer heat conduction with neural operators. Simulation data is generated with a finite-difference solver, the learning model is a Fourier Neural Operator (FNO) trained on windowed trajectory data, and the repository includes training, evaluation, and plotting utilities. The long-term goal is to provide clean entry points for the full pipeline, but the current interface is still uneven and is documented here as it exists today.

## Project Status

### Current state

The repository already contains the main components for data generation, model training, model evaluation, and visualization. At present, those components are not all exposed through consistent top-level scripts, and some files under `scripts/` are placeholders rather than working entry points.

### Target state

The goal is to provide meaningful, stable entry points for each major stage of the workflow: generate data, train models, evaluate checkpoints, and generate plots. Until that surface is cleaned up, this README documents the real working entry points instead of the placeholder wrappers.

## Setup

Run commands from the repository root.

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

Currently, `requirements.txt` is the safest install source. `pyproject.toml` does not yet reflect the full runtime dependency set used by the codebase.

## Workflow Overview

1. Generate simulation data with [data/generate_dataset.py](/Users/henriklind/Desktop/no-tps-ihcp/data/generate_dataset.py).
2. Train a model or run a sweep with [scripts/run_train.py](/Users/henriklind/Desktop/no-tps-ihcp/scripts/run_train.py) or [src/operators/train.py](/Users/henriklind/Desktop/no-tps-ihcp/src/operators/train.py).
3. Evaluate saved checkpoints with [src/operators/eval.py](/Users/henriklind/Desktop/no-tps-ihcp/src/operators/eval.py).
4. Generate plots with [visual/plots.py](/Users/henriklind/Desktop/no-tps-ihcp/visual/plots.py).

## Current Entry Points

These are the current entry points that correspond to runnable code:

- `python data/generate_dataset.py`
- `python scripts/run_train.py -m`
- `python -m src.operators.train`
- `python -m src.operators.eval`
- `python -m visual.plots ...`
- `python scripts/collect_best_config.py`

These files are present but currently empty and should not be treated as working entry points:

- [scripts/run_gen_data.py](/Users/henriklind/Desktop/no-tps-ihcp/scripts/run_gen_data.py)
- [scripts/run_dataset.py](/Users/henriklind/Desktop/no-tps-ihcp/scripts/run_dataset.py)
- [scripts/run_eval.py](/Users/henriklind/Desktop/no-tps-ihcp/scripts/run_eval.py)

## Data Generation

[data/generate_dataset.py](/Users/henriklind/Desktop/no-tps-ihcp/data/generate_dataset.py) is the current entry point for synthetic data generation. It builds two-layer finite-difference solver configurations with sampled material properties and forcing parameters, runs the simulations, and saves the resulting trajectories for later training.

The current default is `1024` simulations. The script writes:

- `x_grid.npy`
- `t_grid.npy`
- `trajectories.npy`
- `sim_params.npy`

These files are later consumed by [data/dataset.py](/Users/henriklind/Desktop/no-tps-ihcp/data/dataset.py).

```bash
python data/generate_dataset.py
```

Currently, the files are written to the working directory, so running from the repository root is the safe default. There is not yet a dedicated wrapper script or a configurable output path for this direct entry point.

## Training

### 7a. Hydra Sweep Workflow

[scripts/run_train.py](/Users/henriklind/Desktop/no-tps-ihcp/scripts/run_train.py) is the current entry point for Hydra-based sweeps. Hydra handles multirun orchestration and output management, while Optuna is the hyperparameter search engine used by the Hydra sweeper. Optuna proposes trial parameter combinations from the configured search space in [conf/search_space/medium.yaml](/Users/henriklind/Desktop/no-tps-ihcp/conf/search_space/medium.yaml), and each Hydra job corresponds to one Optuna trial. The sweeper configuration lives in [conf/hydra/sweeper/optuna_local.yaml](/Users/henriklind/Desktop/no-tps-ihcp/conf/hydra/sweeper/optuna_local.yaml), and the optimization objective is the configured mean best validation score across seeds from [conf/config.yaml](/Users/henriklind/Desktop/no-tps-ihcp/conf/config.yaml).

Each invocation gets an experiment namespace such as `experiment0`. Each job is written under `runs/<experiment>/conf<job_num>/`, and each seed for that configuration is written under `seed<seed>/`. Resolved configs, JSON summaries, rankings, and `best_config.yaml` are written under `conf/generated/<experiment>/`.

```bash
python scripts/run_train.py -m
python scripts/run_train.py -m tuning.n_trials=10
```

### 7b. Direct Training Module

[src/operators/train.py](/Users/henriklind/Desktop/no-tps-ihcp/src/operators/train.py) is the lower-level training entry point. It loads [conf/config.yaml](/Users/henriklind/Desktop/no-tps-ihcp/conf/config.yaml), expects the dataset files to already exist, trains all configured seeds, and writes per-seed checkpoints together with `train_metrics.csv`.

```bash
python -m src.operators.train
```

## Evaluation

[src/operators/eval.py](/Users/henriklind/Desktop/no-tps-ihcp/src/operators/eval.py) is the current evaluation entry point. It loads `fno2d_best.pt` checkpoints from a run directory, computes per-seed test metrics, and writes `seed_report.json`.

```bash
python -m src.operators.eval
```

Currently, the module `__main__` assumes `runs/experiment0/config0`, which is a limitation of the current interface rather than a polished evaluation CLI. [scripts/run_eval.py](/Users/henriklind/Desktop/no-tps-ihcp/scripts/run_eval.py) is currently empty.

## Plot Generation

[visual/plots.py](/Users/henriklind/Desktop/no-tps-ihcp/visual/plots.py) provides the current plot-generation CLI. It supports grouped outputs for `physics`, `mms`, `training`, and `data`.

Important flags:

- `--data`
- `--params`
- `--csv`
- `--report`
- `--out`
- `--group`
- `--plots`

Examples:

```bash
python -m visual.plots --out visual/
python -m visual.plots --group physics --out visual/
python -m visual.plots --group training --csv runs/experiment0/conf0/seed0/train_metrics.csv --report runs/experiment0/conf0/seed_report.json --out visual/
python -m visual.plots --group data --data trajectories.npy --params sim_params.npy --out visual/
```

Some plots require extra inputs and will be skipped if the relevant arguments are not provided. For example, training plots need `--csv` or `--report`, and data plots need `--data` and `--params`.

## Configuration and Outputs

- [conf/config.yaml](/Users/henriklind/Desktop/no-tps-ihcp/conf/config.yaml) is the main training and sweep configuration.
- [conf/paths/default.yaml](/Users/henriklind/Desktop/no-tps-ihcp/conf/paths/default.yaml) defines the main path settings.
- `runs/` stores training outputs and checkpoints.
- `conf/generated/` stores resolved configs, trial summaries, rankings, and `best_config.yaml`.
- `data/` is the intended home for generated datasets, even though the current direct generation script writes to the working directory.

`PROJECT_ROOT` is used by the config system for path resolution and is set by the training wrapper when applicable.

## Project Layout

- [data/generate_dataset.py](/Users/henriklind/Desktop/no-tps-ihcp/data/generate_dataset.py): synthetic trajectory generation
- [data/dataset.py](/Users/henriklind/Desktop/no-tps-ihcp/data/dataset.py): windowed dataset construction and dataloaders
- [src/physics/fd_solver_1d.py](/Users/henriklind/Desktop/no-tps-ihcp/src/physics/fd_solver_1d.py): multilayer finite-difference solver
- [src/physics/mms_1d.py](/Users/henriklind/Desktop/no-tps-ihcp/src/physics/mms_1d.py): manufactured-solution verification
- [src/operators/fno2d.py](/Users/henriklind/Desktop/no-tps-ihcp/src/operators/fno2d.py): Fourier Neural Operator model
- [src/operators/train.py](/Users/henriklind/Desktop/no-tps-ihcp/src/operators/train.py): direct training module
- [src/operators/eval.py](/Users/henriklind/Desktop/no-tps-ihcp/src/operators/eval.py): checkpoint evaluation and reporting
- [scripts/run_train.py](/Users/henriklind/Desktop/no-tps-ihcp/scripts/run_train.py): Hydra + Optuna sweep entry point
- [scripts/collect_best_config.py](/Users/henriklind/Desktop/no-tps-ihcp/scripts/collect_best_config.py): rebuild generated config rankings
- [visual/plots.py](/Users/henriklind/Desktop/no-tps-ihcp/visual/plots.py): plot-generation CLI
- [conf/config.yaml](/Users/henriklind/Desktop/no-tps-ihcp/conf/config.yaml): main configuration
- [PROJECT_OVERVIEW.md](/Users/henriklind/Desktop/no-tps-ihcp/PROJECT_OVERVIEW.md): deeper technical overview

## Known Gaps

- Empty wrapper scripts remain in `scripts/`.
- The top-level CLI surface is inconsistent.
- Evaluation still defaults to a hard-coded run path in the module `__main__`.
- Dataset generation currently writes to the working directory instead of a configurable output path.
- Dependency metadata is split between [requirements.txt](/Users/henriklind/Desktop/no-tps-ihcp/requirements.txt) and [pyproject.toml](/Users/henriklind/Desktop/no-tps-ihcp/pyproject.toml).
