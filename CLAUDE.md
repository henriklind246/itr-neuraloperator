# Claude Guide - no-tps-ihcp

Durable context for Claude Code sessions in this repo. Read `AGENTS.md` first;
read `PROJECT_OVERVIEW.md` for broader architecture, but verify active tensor
contracts against the code and the run artifacts because this project has moved
quickly.

## What this project is

A **forward** heat-conduction surrogate: a time-conditioned 2D Fourier Neural
Operator trained on trajectories from a conservative Crank-Nicolson
finite-volume solver. The geometry is two fixed vertical slabs on `[0, 1] x
[0, 1]` with an interface at `x = 0.5`; material coefficients are fixed; `R_c`,
initial conditions, temporal forcing, and spatial forcing are sampled per
simulation.

The active research problem is the forcing-representation experiment. The model
needs to learn across four temporal families (`sin`, `exp`, `pulse_train`,
`exp_train`) and four spatial profiles (`uniform`, `patch`, `gaussian`,
`triangle`) for separable left flux `q_L(y, t) = a(t) * s(y)`. Current symptoms:
training error can get much lower than cross-sim validation error, and the model
appears to struggle to use the forcing representation robustly, especially for
pulse-like forcing, long leads, and unseen simulation combinations.

Historical anchor from the user: an older branch/run reportedly reached about
2.7% validation with training below 1% in fewer than 80 epochs on the same
forcing-family idea. Treat that as evidence the target is reachable, not as a
copy-paste recipe.

## Current Contracts

- 2D is the active target. Do not edit `src/physics/fv_solver_1d.py`,
  `src/operators/fno1d.py`, or `src/physics/mms_1d.py` unless the task is
  explicitly about the 1D baseline.
- Dataset item shape:
  `(spatial, cond_static, forcing_seq, Y, T_stats)`.
- `spatial`: `(B, Nx, Ny, 20)` with
  `[T_tilde_source, x_norm, y_norm, s_y, Q_y_bin_0..Q_y_bin_15]`.
- `cond_static`: `(B, 23)` with base time/`R_c`, spatial onehot+params,
  temporal onehot, and 8 scalar forcing-summary features.
- `forcing_seq`: `(B, 64, 5)` sampled from `a(t)` over `[t_s, t_j]`:
  `[r, a/A_ref, cumulative/A_cum_ref, time_to_target/t_final, t/t_final]`.
- `FNO2d.forward`: `model(spatial, cond_static, forcing_seq) -> y_pred`.
- The temporal branch encodes `forcing_seq` to `h_a`; `h_a` feeds both learned
  spatial forcing channels `s_y * z_a` and the CIN conditioning MLP.
- Temperature normalization is global training-set `(mu_global, sigma_global)`.
  Do not switch to per-sample normalization.
- Spatial tensors stay channels-last outside FNO internals: `(B, Nx, Ny, C)`.
- Use distinct `modes1` and `modes2`; never reintroduce a bare `modes` key.

## Layout

- Physics: `src/physics/fv_solver_2d.py`, `mms_2d.py`,
  `boundary_forcing.py`, `init_conditions.py`
- Model: `src/operators/fno2d.py`
- Data: `data/dataset.py`, `data/generate_dataset.py`
- Training/eval: `src/operators/train.py`, `src/operators/eval.py`
- Config: `conf/config.yaml`, `conf/search_space/*.yaml`
- Entrypoints: `scripts/run_train_fixed.py`, `scripts/run_train.py`
- Diagnostics: `scripts/inspect_val_pairs.py`,
  `scripts/eval_same_sim_holdout.py`, `scripts/run_eval.py`, `visual/*.py`
- Cluster: `slurm/*.sbatch`
- Tests: `tests/test_*.py`

## Experiment Discipline

- Start from artifacts, not memory: `config_used.yaml`, `train_metrics.csv`,
  `diagnostics.csv`, `val_pairs.csv`, `final_metrics.json`, and same-sim
  held-out results when available.
- Use `scripts/inspect_val_pairs.py <run>/val_pairs.csv` before drawing
  conclusions about forcing families, lead time, `R_c`, or worst sims.
- Run same-sim held-out validation when separating simulation-level
  generalization from a broken forcing/time representation.
- Prefer short screens for architecture or encoding ideas: 30-50 epochs should
  show whether the forcing path is learning; 60 epochs is a stronger screen;
  100+ epochs are confirmation runs.
- If curves are not moving toward a useful regime by about epoch 30-40, do not
  assume late epochs will rescue the run without diagnostics showing otherwise.
- Targets: aspirational below 1% train and val rel-L2; near-term useful screen
  is validation near 1-2% with train near or below 1%.
- Keep one hypothesis per run and use explicit `experiment.name` values. Do not
  mix branch changes, dataset changes, and architecture changes in one run
  unless that is the explicit experiment.

## Running Things

- Single fixed run:
  `python scripts/run_train_fixed.py experiment.name=forcing_screen_40ep training.epochs=41 training.validate_every=5`
- Optuna/local sweep:
  `python scripts/run_train.py --multirun`
- Data generation:
  `python data/generate_dataset.py --num-sims 8000`
- Val-pair inspection:
  `python scripts/inspect_val_pairs.py <run_dir>/seed42/val_pairs.csv`
- Same-sim held-out diagnostic:
  `python scripts/eval_same_sim_holdout.py experiment.name=<name> training.epochs=41`
- Test eval:
  `python scripts/run_eval.py <run_dir>`
- Tests:
  `pytest tests/ -q`

## Edit Rules

- Never commit or push.
- Never edit `.git/`, git config, hooks, or CI configuration.
- Prefer editing existing modules. New abstractions need a real second caller or
  prior discussion.
- New model/training config knobs need defaults in `conf/config.yaml`; sweepable
  knobs also need a relevant `conf/search_space/*.yaml` entry.
- Solver, loss, dataset, or model behavior changes need tests under `tests/`.
- Numerical-scheme changes require MMS verification in `src/physics/mms_2d.py`
  and 2nd-order convergence before claiming correctness.
- Do not add comments that merely describe code. Only explain non-obvious
  constraints or why a choice exists.

## Known Gotchas

- `PROJECT_OVERVIEW.md` and some inline docstrings may lag the code. The active
  code path is currently 20 spatial channels, 23 static conditioning dims, and a
  separate `(64, 5)` `forcing_seq`.
- `FNO2d.__init__` still has a legacy default `cond_static_dim=15`; current
  config passes `cond_static_dim: 23`.
- Simulation count is inconsistent across entry points: `generate_sim_data`
  defaults to 2000, the generation CLI defaults to 8000, and
  `conf/config.yaml` records `data.num_sims: 10000`. Actual
  `trajectories.npy` shape wins.
- `config["data"]["trajectories.npy"]` has a period in the key. Use bracket
  access.
- `validate_every=10` and `patience=20` means 20 validation checks, not 20
  epochs.
- Training and validation `rel_l2` are normalized-space percentages. Eval also
  reports physical-space Kelvin metrics; do not compare those directly.
- `val_pairs.csv` can be huge because validation writes one row per pair on
  validation epochs.
- The worktree may already contain user changes. Do not revert unrelated files.

## Style

Be concise, concrete, and artifact-based. When making an experiment claim, cite
the run path and metric convention. No emojis in files.
