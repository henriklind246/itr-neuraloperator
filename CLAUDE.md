# Claude Guide - no-tps-ihcp

Durable context for Claude Code sessions in this repo. Read `AGENTS.md` first;
read `PROJECT_OVERVIEW.md` for broader architecture, but verify active tensor
contracts against the code and the run artifacts because this project has moved
quickly. When the guidance files, docstrings, code, and run artifacts disagree,
the `problems/` ProblemSpec for the active `(benchmark, representation)` pair and
the run's `config_used.yaml` win.

## What this project is

A **forward** heat-conduction surrogate: a time-conditioned 2D Fourier Neural
Operator trained on trajectories from a conservative Crank-Nicolson
finite-volume solver. The geometry is two stacked vertical slabs on
`[0, 1] x [0, 1]` with a thin-resistance interface; material coefficients are
fixed (`k=2` left, `k=1` right; `rho=cp=1`).

The repo is now organized as a **benchmark suite**, not a single experiment.
Each benchmark is a `ProblemSpec` adapter (`problems/`) and is crossed with one
of two input **representations**. The shared model, dataset, training/eval
loops, and the data generator all route benchmark-specific behavior through the
spec so they never branch on the benchmark name themselves.

Benchmarks (`problems/registry.py`):

- `forcing` — separable left flux `q_L(y, t) = a(t) * s(y)`; four temporal
  families (`sin`, `exp`, `pulse_train`, `exp_train`) x four spatial profiles
  (`uniform`, `patch`, `gaussian`, `triangle`); scalar `R_c`; interface fixed at
  `x = 0.5`; uniform 300 K initial condition. This is the original
  forcing-representation experiment and the config default.
- `interfaces` — fixed `sin`/`uniform` forcing; the interface location
  `interface_x` is sampled per sim in `[0.2, 0.8]`; scalar `R_c`; varying
  initial conditions. Uses `per_sample_interface_x` in the loss.
- `source` — internal volumetric chip-heating patch (`x_h, y_h, A, w_h, h_h`,
  stratified `regime`); interface fixed at `x = 0.5`; scalar `R_c`; uniform
  300 K IC.
- `source_itr` — `source` plus a **spatially-varying interface resistance**
  `R_c(y)` following a Gaussian void profile (delamination / air-gap model).
  Reuses `source`'s patch/`A`/IC stream unchanged.

Representations (`problems/registry.py: REPRESENTATIONS`):

- `temporal_encoder` (default) — lean spatial channels plus a `(128, 2)`
  `forcing_seq` token stream consumed by the temporal branch.
- `bins` — integral Q-bin spatial channels and no temporal encoder; the
  `forcing_seq` is emitted as an explicit empty `(0, 0)` tensor.

The active research problem is still the forcing-representation gap: training
error can get much lower than cross-sim validation error, and the model
struggles to use the forcing representation robustly, especially for pulse-like
forcing, long leads, and unseen simulation combinations. Historical anchor from
the user: an older run reportedly reached about 2.7% validation with training
below 1% in fewer than 80 epochs on the same forcing-family idea. Treat that as
evidence the target is reachable, not as a copy-paste recipe.

## Current Contracts

- 2D is the active target. Do not edit `src/physics/fv_solver_1d.py`,
  `src/operators/fno1d.py`, or `src/physics/mms_1d.py` unless the task is
  explicitly about the 1D baseline.
- **There is no single fixed tensor contract anymore.** Dims are owned by the
  `ProblemDims` of the active `(benchmark, representation)` pair
  (`problems/base.py`), resolved via `get_problem(name, representation)` and, at
  inference, `data.dataset.problem_from_config(config)`.
- **Dataset item is a dict**, not a tuple. `build_item` returns
  `{"spatial", "cond_static", "forcing_seq", "Y", "T_stats"}`; `__getitem__`
  wraps each as a tensor. `forcing_seq` is always a key — empty `(0, 0)` in
  `bins` mode.
- `FNO2d.forward(spatial, cond_static, forcing_seq=None) -> y_pred`. The
  temporal encoder is skipped when `use_temporal_encoder=False` (bins mode); the
  lift then sees `in_channels` alone.
- The temporal branch encodes `forcing_seq` to `h_a`. `h_a` projects to the `K`
  learned spatial forcing channels (`s_y[..., s_y_channel] * z_a`) and feeds the
  CIN conditioning MLP. With `use_forcing_time_aug=True` (forcing, interfaces),
  `h_a` is augmented with `cond_static[:, 0:2] = [t_bar_norm, t_s_norm]` before
  the spatial projection.
- `s_y_channel` is the spatial channel the learned forcing multiplies against:
  index 3 for forcing/source/source_itr, index 5 for interfaces.
- Spatial input layout (channels-last `(B, Nx, Ny, C)`):
  - forcing `temporal_encoder`: `[T_tilde, x_norm, y_norm, s_y]` (4).
  - forcing `bins`: above + `Q_y_bin_0..15` (20).
  - source / source_itr add patch/`R_c(y)` channels; interfaces adds interface
    channels. See the contract table below for `in_channels`.
- `temporal_encoder` `forcing_seq`: `(128, 2)` tokens `[r_m, a_m / A_ref]` over
  `[t_s, t_j]`, with `A_ref = 300.0`. `FORCING_TEMPORAL_SAMPLES = 128` is forced
  in `setup_dataset`, overriding `model.parameters.temporal_samples`.
- `T_stats`: `[mu_global, sigma_global]` (dim 2) for forcing; dim 3 for the
  other benchmarks, where index 2 carries a per-benchmark scalar
  (`interface_x`; `0.5` for source/source_itr).
- Temperature normalization is global training-set `(mu_global, sigma_global)`,
  baked into every checkpoint. Do not switch to per-sample normalization.
- Solver `interface_R` is a **list with one entry per interface**; each entry is
  either a scalar (uniform `R_c`) or a `(Ny,)` array (per-row `R_c(y)`). Only
  `source_itr` uses the vector form.
- Use distinct `modes1` and `modes2`; never reintroduce a bare `modes` key.

### (benchmark, representation) contract table

Single source of truth: `tests/test_problems.py: CONTRACTS` and each spec's
`ProblemDims`. `in_ch` = spatial channels; `cond` = `cond_static_dim`;
`token` = `temporal_token_dim`; `enc` = `use_temporal_encoder`;
`aug` = `use_forcing_time_aug`.

| benchmark   | repr             | in_ch | cond | fseq    | token | t_stats | enc | s_y_ch | aug |
|-------------|------------------|-------|------|---------|-------|---------|-----|--------|-----|
| forcing     | temporal_encoder | 4     | 11   | (128,2) | 2     | 2       | yes | 3      | yes |
| forcing     | bins             | 20    | 11   | empty   | 2     | 2       | no  | 3      | no  |
| source      | temporal_encoder | 4     | 7    | (128,2) | 2     | 3       | yes | 3      | no  |
| source      | bins             | 20    | 7    | empty   | 2     | 3       | no  | 3      | no  |
| source_itr  | temporal_encoder | 5     | 10   | (128,2) | 2     | 3       | yes | 3      | no  |
| source_itr  | bins             | 21    | 10   | empty   | 2     | 3       | no  | 3      | no  |
| interfaces  | temporal_encoder | 6     | 4    | (128,2) | 2     | 3       | yes | 5      | yes |
| interfaces  | bins             | 22    | 4    | empty   | 2     | 3       | no  | 5      | no  |

`bins` always equals `temporal_encoder` `in_ch + 16` (the integral Q-bins).
`source_itr` adds one `R_c(y)` channel (index 4) on top of `source`.

## Layout

- Benchmarks: `problems/base.py` (`ProblemSpec`, `ProblemDims`,
  `empty_forcing_seq`), `problems/registry.py` (`REGISTRY`, `REPRESENTATIONS`,
  `get_problem`), `problems/{forcing,interfaces,source,source_itr}.py`
- Physics: `src/physics/fv_solver_2d.py`, `mms_2d.py`,
  `boundary_forcing.py`, `init_conditions.py`, `internal_source.py`
  (volumetric patch + `make_rc_void_profile` / `RC_VOID_RANGES`)
- Model: `src/operators/fno2d.py`
- Data: `data/dataset.py` (`SnapshotPairDataset`, `problem_from_config`,
  `create_dataloaders`), `data/generate_dataset.py`
- Training/eval: `src/operators/train.py`, `src/operators/eval.py`,
  `src/operators/losses.py`, `src/operators/rollout.py`,
  `src/operators/distributed.py`
- Config: `conf/config.yaml`, `conf/benchmark/*.yaml`,
  `conf/representation/*.yaml`
- Entrypoint: `scripts/run_train_fixed.py`
- Diagnostics: `scripts/inspect_val_pairs.py` (benchmark-aware),
  `scripts/run_eval.py`, `scripts/write_test_records.py`,
  `scripts/fv_convergence_baseline.py`, `visual/*.py`
- Cluster: `slurm/*.sbatch`
- Tests: `tests/test_*.py` (notably `test_problems.py` for the contract table)

## Experiment Discipline

- Start from artifacts, not memory: `config_used.yaml` (check `benchmark` and
  `representation`), `train_metrics.csv`, `diagnostics.csv`, `val_pairs.csv`,
  and `final_metrics.json`.
- Use `scripts/inspect_val_pairs.py <run>/seed42/val_pairs.csv` before drawing
  conclusions. It is benchmark-aware: it emits universal stratifications plus the
  benchmark's `val_pair_fields`.
- Prefer short screens for architecture or encoding ideas: 30-50 epochs should
  show whether the forcing path is learning; 60 epochs is a stronger screen;
  100+ epochs are confirmation runs.
- If curves are not moving toward a useful regime by about epoch 30-40, do not
  assume late epochs will rescue the run without diagnostics showing otherwise.
- Targets: aspirational below 1% train and val rel-L2; near-term useful screen
  is validation near 1-2% with train near or below 1%.
- Keep one hypothesis per run and use explicit `experiment.name` values. Do not
  mix benchmark changes, dataset changes, and architecture changes in one run
  unless that is the explicit experiment.

## Running Things

- Single fixed run (pick benchmark + representation):
  `BENCHMARK=forcing REPRESENTATION=temporal_encoder python scripts/run_train_fixed.py experiment.name=forcing_screen_40ep training.epochs=41 training.validate_every=5`
- Data generation (per benchmark):
  `python data/generate_dataset.py --num-sims 8000 --benchmark forcing`
  (benchmark also reads the `BENCHMARK` env var; default `forcing`)
- Val-pair inspection:
  `python scripts/inspect_val_pairs.py <run_dir>/seed42/val_pairs.csv`
- Test eval:
  `python scripts/run_eval.py <run_dir>`
- Tests:
  `pytest tests/ -q`

## Edit Rules

- Never commit or push.
- Never edit `.git/`, git config, hooks, or CI configuration.
- Prefer editing existing modules. New abstractions need a real second caller or
  prior discussion. New benchmarks are added as a `ProblemSpec` plus a
  `conf/benchmark/*.yaml` and a `REGISTRY` entry — not by branching on the
  benchmark name in the dataset/model/training code.
- New model/training config knobs need defaults in `conf/config.yaml`.
- Solver, loss, dataset, or model behavior changes need tests under `tests/`.
  Changes to the per-benchmark contract must update `tests/test_problems.py`.
- Numerical-scheme changes require MMS verification in `src/physics/mms_2d.py`
  and 2nd-order convergence before claiming correctness.
- Do not add comments that merely describe code. Only explain non-obvious
  constraints or why a choice exists.

## Known Gotchas

- `PROJECT_OVERVIEW.md` and some inline docstrings lag the code. In particular,
  `data/dataset.py` still defines a legacy `COND_STATIC_DIM = 23` block, a
  `TEMPORAL_SAMPLES = 64` / `TEMPORAL_TOKEN_DIM = 5` default, and a `(23,)`
  cond docstring. Those are stale: the live forcing contract is `cond=11`,
  `token=2`, `forcing_seq=(128,2)`, owned by `problems/forcing.py`. Trust
  `ProblemDims`.
- `FNO2d.__init__` still has legacy defaults (`in_channels=20`,
  `cond_static_dim=15`, `TemporalForcingEncoder(token_dim=5)`); the real values
  are passed from the resolved `ProblemDims`/config at construction.
- `model.parameters.temporal_samples: 64` in config is overridden to 128 by the
  forcing `setup_dataset` (`FORCING_TEMPORAL_SAMPLES`). The token grid is 128.
- Simulation count is inconsistent across entry points: `generate_sim_data`
  defaults to 2000, the generation CLI defaults to 8000, and
  `conf/config.yaml` records `data.num_sims: 10000`. Actual `trajectories.npy`
  shape wins.
- `config["data"]["trajectories.npy"]` has a period in the key. Use bracket
  access.
- `validate_every=10` and `patience=20` means 20 validation checks, not 20
  epochs.
- Training and validation `rel_l2` are normalized-space percentages. Eval also
  reports physical-space Kelvin metrics; do not compare those directly.
- `val_pairs.csv` is wide (universal columns plus the union of every benchmark's
  `val_pair_fields`) and can be huge; it writes one row per pair on validation
  epochs.
- `source_itr` sets `R_c` to mirror `R_c_base` so the universal `R_c` column
  stays valid even though the real resistance is the `(Ny,)` Gaussian-void
  profile (`R_c_base, R_c_amp, R_c_y0, R_c_sigma`). The solver receives the full
  profile via `interface_R=[profile]`.
- The worktree may already contain user changes. Do not revert unrelated files.

## Style

Be concise, concrete, and artifact-based. When making an experiment claim, cite
the run path, the `(benchmark, representation)` pair, and the metric convention.
No emojis in files.
