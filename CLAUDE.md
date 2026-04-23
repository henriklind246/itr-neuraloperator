# Claude Guide — no-tps-ihcp

Durable context for Claude Code sessions in this repo. For the full architecture, read `PROJECT_OVERVIEW.md`.

## What this project is

A **forward** heat-conduction surrogate: a time-conditioned **2D Fourier Neural Operator** trained on trajectories from a conservative 2D Crank-Nicolson finite-volume solver. Two vertical material layers with interfacial thermal resistance. 4000 sims, three varying parameters (amplitude, frequency, R_c), 70/15/15 split by sim ID.

Active branch: `feature/2d-multi-layer-fv-solver`. 2D is the primary target; the 1D files (`fno1d.py`, `fv_solver_1d.py`, `mms_1d.py`) are a baseline — **do not modify them unless the task is explicitly about the 1D baseline**.

## Layout (entry points)

- Physics: `src/physics/fv_solver_2d.py`, `mms_2d.py`
- Model: `src/operators/fno2d.py`
- Data: `data/dataset.py` (`SnapshotPairDataset`), `data/generate_dataset.py`
- Training: `src/operators/train.py`, eval in `src/operators/eval.py`
- Config: `conf/config.yaml` + `conf/search_space/*.yaml`
- Entrypoints: `scripts/run_train.py` (Optuna sweep), `scripts/run_train_fixed.py` (single config)
- MSI cluster: `slurm/*.sbatch`
- Tests: `tests/test_{fv_solver_2d,mms_2d,fno1d,fv_solver_1d,mms_1d}.py` (pytest)

## Conventions

- **Tensor layout**: spatial is `(B, Nx, Ny, C)` throughout — never transpose to channels-first outside FNO internals.
- **Conditioning vector**: `(B, 5) = [t_bar_norm, t_s_norm, A_norm, f_norm, R_c_norm]`.
- **Normalization**: global (training-set `mu_global, sigma_global`), never per-sample. Saved in every checkpoint.
- **Fourier modes**: always separate `modes1` (x) and `modes2` (y). Never use a bare `modes` key — it was 1D-only and has been removed from search spaces.
- **Checkpoints**: `fno2d_best.pt` (best val) + `fno2d_latest.pt` (resume sentinel, deleted on clean completion).
- **Data splits**: split seed is hardcoded to 0 in `split_sim_ids` — do not change. Model-init seed varies.

## Running things

- **Single training run**: `python scripts/run_train_fixed.py` (Hydra overrides work, e.g. `training.epochs=50 training.seeds=[42]`).
- **Optuna sweep (local)**: `python scripts/run_train.py --multirun` (search space set by `conf/config.yaml` `defaults:`).
- **MSI sweep**: `sbatch slurm/train_fno_msi.sbatch` — uses node-local `/tmp/fno_runs_${SLURM_JOB_ID}` for the Optuna SQLite DB (NFS locking would otherwise corrupt it), copies results back to `$HOME/fno_runs` at job end.
- **Tests**: `pytest tests/` from repo root.
- **Data regen** (expensive, ~hours): `python data/generate_dataset.py`.

## Rules for edits

- **Don't add error handling / validation** for conditions that can't happen in this codebase (internal code, trusted inputs). Validate only at boundaries.
- **Don't add comments that describe what the code does** — only non-obvious *why*.
- **Prefer editing existing files.** No new abstractions without a concrete second caller.
- **Numerical-scheme changes require MMS verification** — add or update a case in `mms_2d.py` (or `mms_1d.py` if touching 1D) and confirm 2nd-order convergence before claiming correctness.
- **Solver / loss changes require a test** in `tests/`.
- **New model or training params** must have a default in `conf/config.yaml`. If it's sweepable, add it to at least one file under `conf/search_space/`.
- **Don't commit** `runs/`, `data/*.npy`, `.claude/settings.local.json`, or anything in `/tmp/`.

## Known gotchas

- **Optuna 2.10.0 + SQLite**: `scripts/run_train.py` monkey-patches `RDBStorage` to skip the compatibility check and force `StaticPool` — don't remove it unless upgrading to Optuna 3.x.
- **`config["data"]["trajectories.npy"]`**: the key literally contains a period. Access via bracket, never dot.
- **`validate_every=10 × patience=20` = 200-epoch tolerance**, not 20 epochs. Doubled-counted patience has bitten before.
- **`SpectralConv2d` mode guard**: raises if `2*modes1 > Nx_freq` because positive/negative x-mode slices would overlap silently. The y-dimension is one-sided so no overlap there.
- **Interface geometry**: vertical slab only (x = const). Interfaces on nodes or y-faces are rejected by the solver.
- **Training `iface_rel_l2`** is in normalized space; `eval.py`'s `test_iface_rel_l2` is in physical (Kelvin) space. Not directly comparable.

## Style

Terse responses; no emojis in files or chat; final summaries ≤2 sentences. When citing code, use `file:line` format.
