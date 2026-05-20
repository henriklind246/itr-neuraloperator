# AGENTS.md — guidance for Codex (and other coding agents)

This file is the entry point for AI coding agents working in this repo. Read it
in full before making any changes. For more detail on conventions and gotchas,
also read `CLAUDE.md` and `PROJECT_OVERVIEW.md`. If the guidance files,
inline docstrings, code, and run artifacts disagree, verify against the current
code path and the exact run artifacts before acting.

---

## Repository rules

- **Never commit or push.** Do not run `git commit`, `git push`, `git rebase`,
  `git reset --hard`, or any history-rewriting command. Stage and modify files
  freely; let the user review and commit.
- **Never edit `.git/`, git config, hooks, or CI configuration.**
- **Prefer existing classes, modules, and file structures.** Do not introduce
  new abstractions, factories, registries, or wrapper layers when an existing
  one fits. If you think a new abstraction is needed, propose it in chat first.
- **Edit existing files in place rather than creating new ones.** New files
  require a concrete reason (e.g. a new test module, a new physics module that
  has no natural home).
- **Don't write documentation files (`*.md`, READMEs) unless the user asks.**
- **Don't add comments that describe what code does** — only non-obvious *why*
  (constraints, invariants, references to specific bugs).
- **Don't add error handling for impossible conditions.** Validate only at true
  system boundaries (user input, file I/O, external APIs). Trust internal code
  and framework guarantees.
- **Numerical-scheme changes require MMS verification.** Add or update a case
  in `src/physics/mms_2d.py` and confirm 2nd-order convergence before claiming
  correctness. Solver / loss changes require a test under `tests/`.
- **Do not modify the 1D baseline** (`src/physics/fv_solver_1d.py`,
  `src/operators/fno1d.py`, `src/physics/mms_1d.py`) unless the task is
  explicitly about the 1D baseline. The 2D pipeline is the active target.

---

## Current project state and experiment posture

The active work is the 2D forcing-representation experiment: fixed rectangular
two-layer geometry with interface at `x = 0.5`, fixed material coefficients,
separable left-boundary forcing `q_L(y, t) = a(t) * s(y)`, four temporal
families, four spatial families, sampled initial-condition families, and sampled
`R_c`.

The current empirical problem is not the FV solver. The model is struggling to
use the forcing representation robustly: training error can drop much lower than
cross-simulation validation error, and the hard regimes are usually pulse-like
temporal forcing, long lead times, and unseen simulation-level combinations.
Treat proposed model/training changes as hypotheses about this representation
gap, not as generic capacity tweaks.

Before changing architecture, losses, or data encoding for this experiment:

- Inspect the run artifacts that match the exact code/data/config being
  discussed: `config_used.yaml`, `train_metrics.csv`, `diagnostics.csv`,
  `val_pairs.csv`, `final_metrics.json`, and any same-sim held-out result.
- Use `scripts/inspect_val_pairs.py` to stratify validation error by temporal
  family, spatial family, lead time, `t_s`, `R_c`, and simulation ID.
- Use `scripts/eval_same_sim_holdout.py` when deciding whether the gap is mostly
  simulation-level generalization or a failure to learn the forcing/time
  representation even within seen simulations.
- Do not treat old `conf/generated/*` or old run folders as authoritative
  unless their `config_used.yaml`, git status, dataset shape, and tensor
  contracts match the current code path.

Experiment turnaround is a core constraint. A 100-epoch run on 5 A100s has been
taking a little over six hours, so use shorter screening runs when comparing
ideas. As a default heuristic, 30-50 epochs should reveal whether the model is
learning the forcing representation at all; 60 epochs is often enough for a
stronger screen; 100+ epochs should be reserved for confirmation runs. If train
or validation error is clearly not moving toward a useful regime by roughly
epoch 30-40, do not assume late epochs will rescue it without evidence from the
learning curves or diagnostics.

Targets for interpreting runs:

- Aspirational target: below 1% relative L2 on both train and validation.
- Pragmatic near-term target: validation near 1-2%, with train also near or
  below 1%.
- User-reported historical anchor: a previous branch/run reached about 2.7%
  validation with training below 1% in fewer than 80 epochs using the same
  temporal and spatial forcing-family idea. Treat this as evidence that the
  problem is reachable, but verify exact code/data differences before copying
  conclusions.

Keep experiments named and isolated. Prefer one clear hypothesis per run
(encoding, temporal branch, scheduler, snapshots, loss weighting, etc.) so the
result can be interpreted later.

---

## Governing physics

The PDE is the **transient 2D heat equation** in two stacked vertical material
layers with interfacial thermal contact resistance:

```
ρ(x) c_p(x) ∂T/∂t  =  ∇·( k(x) ∇T )           on  [a,b] × [c,d],  t ∈ [0, t_final]
```

with a thin-resistance interface condition at each layer boundary (continuous
heat flux, jump in T proportional to that flux, gap conductance = 1 / R_c).

**Boundary conditions** (fixed for this project):
- Left  (x = a):  Neumann, prescribed flux `q_L(y, t) = a(t) · s(y)`.
- Right (x = b):  Dirichlet `T = T_right(t)` (default 300 K, constant).
- Top and bottom (y = c, y = d):  adiabatic (zero flux).

**Initial condition** is sampled per-sim in `data/generate_dataset.py` from
`src/physics/init_conditions.py`, added around `T_right`, tapered near the right
edge, and pinned at the right Dirichlet boundary so it is consistent with
`T = T_right`.

This is a **forward** problem: given an initial field T(x, y, 0) and the
conditioning inputs, predict T(x, y, t) for any future time.

---

## Scope assumptions — 2D FV solver (`src/physics/fv_solver_2d.py`)

- **Grid is uniform and isotropic**: `hx == hy == h`. The solver raises if
  `hx != hy` (deliberate simplification, not a theoretical requirement).
- **Geometry is rectangular**: `[a, b] × [c, d]`.
- **Layers are vertical slabs only**: each `Layer1D(x_left, x_right, rho, cp, k)`
  spans the full y-extent; interfaces are at `x = const`. Interfaces on
  y-faces or arbitrary off-face positions are rejected.
- **Interfaces must lie on x-direction cell faces** (midpoints between x-nodes).
  On-node interfaces are rejected.
- **Time integration is Crank–Nicolson** (2nd order in time, unconditionally
  stable). Spatial discretization is conservative finite-volume on cell-centered
  nodes.
- **Banded structure**: tridiagonal block per row, assembled as a sparse matrix
  and solved with `scipy.sparse.linalg.splu`.
- **Left BC**: half-cell energy balance at i=0 with CN time-averaged flux. The
  callable `q_left_fn(t)` may return a scalar (uniform-in-y) or shape `(Ny,)`
  (separable `q_L(y, t) = a(t) s(y)`).
- **Tensor layout** for trajectories: `(Nt, Nx, Ny)` per sim,
  stored as `(num_sims, Nt, Nx, Ny)` on disk.
- The 1D dataclass `Layer1D` is reused as `Layer2D = Layer1D` since layers are
  x-slabs. **Do not add a separate `Layer2D` class.**

---

## Scope assumptions — data generation (`data/generate_dataset.py`)

- **Simulation count is experiment metadata, not a physics invariant.**
  `generate_sim_data(...)` defaults to 2000 sims, the CLI default is currently
  8000, and `conf/config.yaml` currently records `data.num_sims: 10000`.
  Before planning or interpreting an experiment, inspect the actual
  `trajectories.npy` shape and the run's `config_used.yaml`.
- **Split is fixed**: 70/15/15 train/val/test by sim ID with seed 0
  (split seed is hardcoded; do not change).
- **Material parameters are fixed per layer**: layer 1 (`x ∈ [0, 0.5]`) has
  `k=2`, layer 2 (`x ∈ [0.5, 1]`) has `k=1`; both have `ρ=cp=1`.
- **Single sweepable material parameter**: `R_c ∈ [0.05, 1.0]`, sampled by
  Latin Hypercube (`d=1`) with seed 0.
- **Forcing-family parameters are sampled per-sim** by the per-family samplers
  in `src/physics/boundary_forcing.py` (`TEMPORAL_SAMPLERS`,
  `SPATIAL_SAMPLERS`), not by LHS. Don't move them back into LHS.
- **Boundary flux is separable**: `q_L(y, t) = a(t) · s(y)`.
  - Temporal families: `sin`, `exp`, `pulse_train`, `exp_train`. Sampled
    uniformly across the four.
  - Spatial families: `uniform`, `patch`, `gaussian`, `triangle`. Sampled
    uniformly.
- **Initial-condition families** are sampled per-sim by
  `src/physics/init_conditions.py` (`uniform_2d`, `random_sinusoid_2d`,
  `grf_2d`, `hot_spot_2d`).
- **Time grid**: `t_final = 0.3`, `dt = 0.005`, `save_stride = 2` →
  `Nt_saved = 31`. Flux-active window for `sin` is `[t_on, t_off] = [0, 0.2]`.
- **`dt.npy` is mandatory run metadata**: the solver `dt` is saved alongside
  `t_grid.npy` because `t_grid[1] - t_grid[0] = solver_dt × save_stride` is the
  snapshot cadence, not the solver dt. `data.dataset.load_solver_dt(t_grid_path)`
  reads it; training/eval pass it through `create_dataloaders(dt=...)`.
- **Per-sim metadata schema** (saved to `sim_params.npy`):
  ```
  {
    R_c, T0, ic_family, ic_params,
    temporal_family, temporal_params,
    spatial_family, spatial_params
  }
  ```
  `temporal_params` and `spatial_params` are family-dependent and match the
  corresponding builder kwargs verbatim. `T0` is `(Nx, Ny) float32` centered
  around `T_right`.

---

## Scope assumptions — dataset / dataloader (`data/dataset.py`)

- `SnapshotPairDataset` enumerates **all-to-all snapshot pairs** within each
  sim from a uniformly subsampled set of `n_snapshots` time indices.
- Pairs are **sorted by lead time** (`t_j - t_s`) to support curriculum slicing
  via `set_curriculum_fraction(frac)`.
- **Global normalization**: `T̃ = (T − μ_global) / σ_global`, with
  `μ_global, σ_global` computed once over the *training* sims only and
  baked into every checkpoint. **Do not switch to per-sample normalization.**
- **Dataset item format**:
  ```
  (spatial, cond_static, forcing_seq, Y, T_stats)
  ```
- **Spatial input**: `(Nx, Ny, 20)`:
  ```
  [T̃_source, x_norm, y_norm, s_y, Q_y_bin_0, ..., Q_y_bin_15]
  ```
  `s_y` is the spatial forcing profile broadcast across x. Each `Q_y_bin_k`
  is the signed temporal integral of `a(t)` over one sub-interval of
  `[t_s, t_j]`, multiplied by `s_y` and normalized by `q_ref`.
- **Static conditioning vector**: `(23,)` — single source of truth is
  `COND_STATIC_DIM` in `data/dataset.py`. Layout:
  ```
  [0:3]    base:             t_bar_norm, t_s_norm, R_c_norm
  [3:7]    spatial onehot:   uniform, patch, gaussian, triangle
  [7:11]   spatial params:   y_c_norm, w_norm, sigma_y_norm, ell_norm
  [11:15]  temporal onehot:  sin, exp, pulse_train, exp_train
  [15:23]  forcing summary:  signed/abs/pos/neg impulse, mean, RMS, peak, final
  ```
- **Temporal forcing sequence**: `(64, 5)` by default, built from samples of
  `a(t)` over `[t_s, t_j]`:
  ```
  [r, a(t)/A_ref, cumulative_integral/A_cum_ref, (t_j - t)/t_final, t/t_final]
  ```
  `forcing_seq` and the 8-dim forcing-summary block must be built from the same
  sampled `a(t)` values so the two conditioning pathways cannot drift apart.
- Use `build_cond_vector`, `build_forcing_seq`, and `build_forcing_summary` in
  `data/dataset.py` whenever constructing inputs outside `__getitem__`
  (e.g. inference plotting). Do not duplicate the slot logic.
- **Tensor layout convention** project-wide: spatial is
  `(B, Nx, Ny, C)` (channels-last). Never transpose to channels-first outside
  internal FNO blocks.

---

## Scope assumptions — FNO model (`src/operators/fno2d.py`)

- **Forward signature**: `model(spatial, cond_static, forcing_seq) → y_pred`.
- **Architecture**: temporal forcing encoder + learned forcing injection +
  Lift → pad → N Fourier blocks (SpectralConv2d + 1×1 Conv + Conditional
  Instance Norm + GELU + dropout) → unpad → projection MLP.
- **Base spatial channels**: `in_channels=20`. The model then appends
  `forcing_spatial_dim` learned channels internally via
  `forcing_field_k(x, y) = s_y(y) * z_a[k]`, where `z_a` is projected from the
  temporal embedding `h_a`.
- **Conditioning**: `TemporalForcingEncoder(forcing_seq)` emits `h_a`
  (`forcing_embed_dim=64` by default). `ConditioningMLP` consumes
  `[cond_static, h_a]` (`23 + 64` dims by default) and emits per-layer
  `(γ, β)` for Conditional Instance Norm. The head is soft identity-initialized
  so the model starts near an unconditioned FNO while gradients still flow
  through the forcing path.
- **Spectral mode invariant**: `2 * modes1 ≤ Nx_freq`. Always use distinct
  `modes1` and `modes2`. No bare `modes` key — that was 1D-only and has been
  removed from search spaces.
- The model has **no family-specific branches** beyond what flows through
  `cond_static`, `forcing_seq`, and the spatial forcing channels. New forcing
  or spatial families are added by extending the boundary-forcing registry and
  dataset encoders, not by adding family logic to the model.

---

## File structure (entry points)

- Physics: `src/physics/fv_solver_2d.py`, `src/physics/mms_2d.py`,
  `src/physics/boundary_forcing.py`, `src/physics/init_conditions.py`
- Model: `src/operators/fno2d.py`, training in `src/operators/train.py`,
  eval in `src/operators/eval.py`, losses in `src/operators/losses.py`
- Data: `data/generate_dataset.py`, `data/dataset.py`
- Config: `conf/config.yaml` (+ `conf/search_space/*.yaml` for Optuna sweeps)
- Entry scripts: `scripts/run_train_fixed.py`, `scripts/run_train.py`
- Experiment diagnostics: `scripts/inspect_val_pairs.py`,
  `scripts/eval_same_sim_holdout.py`, `scripts/run_eval.py`,
  `visual/training_plots.py`, `visual/sweep_plots.py`
- Cluster: `slurm/*.sbatch`
- Tests: `tests/test_*.py` (`pytest tests/` from repo root)
- Visualization: `visual/cli.py` (dispatcher) + `visual/dataset_plots.py`,
  `visual/physics_plots.py`, etc.

When adding functionality, place it in the existing module that already owns
the concern. Do not split a single concern across new modules.

---

## When making changes

1. Read `CLAUDE.md` for the full conventions list and known gotchas.
2. Locate the existing function/class that owns the concern; edit there.
3. If a config knob changes, add a default in `conf/config.yaml`. If the knob
   should be sweepable, add it to a relevant `conf/search_space/*.yaml`.
4. Update or add a test under `tests/` for any solver, loss, dataset, or
   model change.
5. Run `pytest tests/ -q` before claiming the change works.
6. For experiment conclusions, cite the exact artifact path and metric
   convention (`train/val_rel_l2` normalized-space percent vs eval physical
   percent).
7. Stop. Do not commit. Do not push.
