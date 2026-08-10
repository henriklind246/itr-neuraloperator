# AGENTS.md — guidance for Codex (and other coding agents)

This file is the entry point for AI coding agents working in this repo. Read it
in full before making any changes. For more detail on conventions and gotchas,
also read `CLAUDE.md` and `PROJECT_OVERVIEW.md`. If the guidance files,
inline docstrings, code, and run artifacts disagree, the `problems/` ProblemSpec
for the active `(benchmark, representation)` pair and the run's
`config_used.yaml` win — verify against those before acting.

---

## Repository rules

- **Never commit or push.** Do not run `git commit`, `git push`, `git rebase`,
  `git reset --hard`, or any history-rewriting command. Stage and modify files
  freely; let the user review and commit.
- **Never edit `.git/`, git config, hooks, or CI configuration.**
- **Prefer existing classes, modules, and file structures.** Do not introduce
  new abstractions, factories, or wrapper layers when an existing one fits. New
  benchmarks are added as a `ProblemSpec` plus a `conf/benchmark/*.yaml` and a
  `REGISTRY` entry — not by branching on the benchmark name in shared
  dataset/model/training code. If you think another new abstraction is needed,
  propose it in chat first.
- **Edit existing files in place rather than creating new ones.** New files
  require a concrete reason (e.g. a new test module, a new physics module, a new
  `ProblemSpec`).
- **Don't write documentation files (`*.md`, READMEs) unless the user asks.**
- **Don't add comments that describe what code does** — only non-obvious *why*
  (constraints, invariants, references to specific bugs).
- **Don't add error handling for impossible conditions.** Validate only at true
  system boundaries (user input, file I/O, external APIs). Trust internal code
  and framework guarantees.
- **Numerical-scheme changes require MMS verification.** Add or update a case
  in `src/physics/mms_2d.py` and confirm 2nd-order convergence before claiming
  correctness. Solver / loss changes require a test under `tests/`; changes to a
  per-benchmark tensor contract require updating `tests/test_problems.py`.
- **Do not modify the 1D baseline** (`src/physics/fv_solver_1d.py`,
  `src/operators/fno1d.py`, `src/physics/mms_1d.py`) unless the task is
  explicitly about the 1D baseline. The 2D pipeline is the active target.

---

## Current project state and experiment posture

The repo is now a **benchmark suite**, not a single experiment. Each benchmark
is a `ProblemSpec` adapter under `problems/`, crossed with one of two input
**representations**. The shared solver, data generator, dataset, model, and
training/eval loops route all benchmark-specific behavior through the spec, so
they never branch on the benchmark name themselves.

Benchmarks (`problems/registry.py: REGISTRY`):

- `forcing` — separable left flux `q_L(y, t) = a(t) · s(y)`; four temporal
  families (`sin`, `exp`, `pulse_train`, `exp_train`) × four spatial profiles
  (`uniform`, `patch`, `gaussian`, `triangle`); scalar `R_c`; interface fixed at
  `x = 0.5`; uniform 300 K IC. This is the original forcing-representation
  experiment and the config default.
- `interfaces` — fixed `sin`/`uniform` forcing; the interface location
  `interface_x` is sampled per sim in `[0.2, 0.8]`; scalar `R_c`; varying ICs;
  `per_sample_interface_x` loss band.
- `source` — internal volumetric chip-heating patch (`x_h, y_h, A, w_h, h_h`,
  stratified `regime`); interface fixed at `x = 0.5`; scalar `R_c`; uniform
  300 K IC.
- `source_itr` — `source` plus a **spatially-varying interface resistance**
  `R_c(y)` following a Gaussian void profile (delamination / air-gap model).
  Reuses `source`'s patch/`A`/IC stream unchanged.

Representations (`problems/registry.py: REPRESENTATIONS`):

- `temporal_encoder` (default) — lean spatial channels plus a `(128, 2)`
  `forcing_seq` token stream consumed by the temporal branch.
- `bins` — integral Q-bin spatial channels and no temporal encoder; `forcing_seq`
  is an explicit empty `(0, 0)` tensor.

The active research problem is still the forcing-representation gap: training
error can drop much lower than cross-simulation validation error, and the hard
regimes are usually pulse-like temporal forcing, long lead times, and unseen
simulation-level combinations. Treat proposed model/training changes as
hypotheses about this representation gap, not as generic capacity tweaks.

Before changing architecture, losses, or data encoding:

- Inspect the run artifacts that match the exact code/data/config being
  discussed: `config_used.yaml` (check the `benchmark` and `representation`
  values), `train_metrics.csv`, `diagnostics.csv`, `val_pairs.csv`,
  and `final_metrics.json`.
- Use `scripts/inspect_val_pairs.py` to stratify validation error. It is
  benchmark-aware: it emits universal stratifications (lead time, `t_s`, `R_c`,
  sim ID) plus the active benchmark's `val_pair_fields`.
- Do not treat old run folders as authoritative unless their
  `config_used.yaml`, git status, dataset shape, and `(benchmark,
  representation)` contracts match the current code path.

Experiment turnaround is a core constraint. A 100-epoch run on 5 A100s has been
taking a little over six hours, so use shorter screening runs when comparing
ideas. As a default heuristic, 30-50 epochs should reveal whether the model is
learning the representation at all; 60 epochs is often enough for a stronger
screen; 100+ epochs should be reserved for confirmation runs. If train or
validation error is clearly not moving toward a useful regime by roughly epoch
30-40, do not assume late epochs will rescue it without evidence from the
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
(encoding, temporal branch, scheduler, snapshots, loss weighting, etc.) and do
not mix a benchmark change, a dataset change, and an architecture change in one
run unless that is the explicit experiment.

---

## Governing physics

The PDE is the **transient 2D heat equation** in two stacked vertical material
layers with interfacial thermal contact resistance:

```
ρ(x) c_p(x) ∂T/∂t  =  ∇·( k(x) ∇T )           on  [a,b] × [c,d],  t ∈ [0, t_final]
```

with a thin-resistance interface condition at each layer boundary (continuous
heat flux, jump in T proportional to that flux, gap conductance = 1 / R_c). The
resistance may be a scalar or, in `source_itr`, vary along the interface as
`R_c(y)`.

**Boundary conditions** (fixed for this project):
- Left  (x = a):  Neumann, prescribed flux `q_L(y, t)` (the `forcing` benchmark
  drives this as `a(t) · s(y)`; other benchmarks use fixed/simple left flux).
- Right (x = b):  Dirichlet `T = T_right(t)` (default 300 K, constant).
- Top and bottom (y = c, y = d):  adiabatic (zero flux).
- Internal volumetric source term in `source` / `source_itr`
  (`src/physics/internal_source.py`).

**Initial condition** is benchmark-dependent: `forcing`, `source`, and
`source_itr` use a uniform 300 K field; `interfaces` samples IC families from
`src/physics/init_conditions.py`. Sampled ICs are added around `T_right`,
tapered near the right edge, and pinned at the right Dirichlet boundary so they
are consistent with `T = T_right`.

This is a **forward** problem: given an initial field T(x, y, 0) and the
conditioning inputs, predict T(x, y, t) for any future time.

---

## Scope assumptions — 2D FV solver (`src/physics/fv_solver_2d.py`)

- **Grid is uniform and isotropic**: `hx == hy == h`. The solver raises if
  `hx != hy` (deliberate simplification, not a theoretical requirement).
- **Geometry is rectangular**: `[a, b] × [c, d]`.
- **Layers are vertical slabs only**: each `Layer2D(x_left, x_right, rho, cp, k)`
  spans the full y-extent; interfaces are at `x = const`. Interfaces on
  y-faces or arbitrary off-face positions are rejected.
- **Interfaces must lie on x-direction cell faces** (midpoints between x-nodes).
  On-node interfaces are rejected.
- **Interface resistance is per-interface and may be a vector.** `interface_R`
  is a `list` with one entry per interface; each entry is either a scalar
  (uniform `R_c`) or a `(Ny,)` array (per-row `R_c(y)`). A `(Ny, 1)` shape is
  rejected — pass a flat `(Ny,)`. The face conductance is
  `G_x = 1 / (h_L/k_L + R_c + h_R/k_R)`, broadcast per row when `R_c` is a
  vector. Only `source_itr` uses the vector form today.
- **Time integration is Crank–Nicolson** (2nd order in time, unconditionally
  stable). Spatial discretization is conservative finite-volume on cell-centered
  nodes.
- **Banded structure**: tridiagonal block per row, assembled as a sparse matrix
  and solved with `scipy.sparse.linalg.splu`.
- **Left BC**: half-cell energy balance at i=0 with CN time-averaged flux. The
  callable `q_left_fn(t)` may return a scalar (uniform-in-y) or shape `(Ny,)`
  (separable `q_L(y, t) = a(t) s(y)`).
- **Tensor layout** for trajectories: `(Nt, Nx, Ny)` per sim, stored as
  `(num_sims, Nt, Nx, Ny)` on disk.

---

## Scope assumptions — data generation (`data/generate_dataset.py`)

- **Benchmark routing**: `generate_sim_data` resolves the active spec via
  `get_problem(benchmark)` and delegates `sample_sim_params` and
  `configure_solver` to it. Select with `--benchmark` (or the `BENCHMARK` env
  var; default `forcing`). The spec decides forcing, source, IC, and `R_c`
  sampling; the generator loop is benchmark-agnostic.
- **Simulation count is experiment metadata, not a physics invariant.**
  `generate_sim_data(...)` defaults to 2000 sims, the CLI default is currently
  8000, and `conf/config.yaml` currently records `data.num_sims: 10000`.
  Before planning or interpreting an experiment, inspect the actual
  `trajectories.npy` shape and the run's `config_used.yaml`.
- **Split is fixed**: 70/15/15 train/val/test by sim ID with seed 0
  (split seed is hardcoded; do not change).
- **Material parameters are fixed per layer**: layer 1 (`x ∈ [0, 0.5]`) has
  `k=2`, layer 2 (`x ∈ [0.5, 1]`) has `k=1`; both have `ρ=cp=1`.
- **`R_c` sampling is by benchmark**: scalar `R_c ∈ [0.05, 1.0]` via Latin
  Hypercube (`d=1`, seed 0) for `forcing`/`interfaces`/`source`; `source_itr`
  instead samples a Gaussian-void profile (`make_rc_void_profile`,
  `RC_VOID_RANGES`, `RC_MIN=0.05`, `R_PEAK_MAX=3.0`) and mirrors `R_c` to its
  base level for bookkeeping.
- **Forcing-family parameters are sampled per-sim** by the per-family samplers
  in `src/physics/boundary_forcing.py` (`TEMPORAL_SAMPLERS`,
  `SPATIAL_SAMPLERS`), not by LHS. Don't move them back into LHS. Only the
  `forcing` benchmark varies these; the others use fixed/simple left flux.
- **Time grid**: `t_final = 0.3`, `dt = 0.005`, `save_stride = 2` →
  `Nt_saved = 31`. `ramp_seconds` (default `2·dt`) is persisted with the
  dataset so the solver flux and the model's forcing conditioning use the same
  startup ramp independent of `dt`.
- **`dt.npy` is mandatory run metadata**: the solver `dt` is saved alongside
  `t_grid.npy` because `t_grid[1] - t_grid[0] = solver_dt × save_stride` is the
  snapshot cadence, not the solver dt. Training/eval pass it through
  `create_dataloaders(dt=...)`.
- **Per-sim metadata** is saved to `sim_params.npy` as an object array of dicts.
  Keys are benchmark-dependent (e.g. `forcing` carries
  `temporal_family/temporal_params/spatial_family/spatial_params`; `source*`
  carry patch params; `source_itr` adds `R_c_base/R_c_amp/R_c_y0/R_c_sigma`).
  All carry `R_c`, `T0`, `ic_family`, `ic_params`. `T0` is `(Nx, Ny) float32`.

---

## Scope assumptions — dataset / dataloader (`data/dataset.py`)

- `SnapshotPairDataset` enumerates **all-to-all snapshot pairs** within each
  sim from a uniformly subsampled set of `n_snapshots` time indices.
- **The active `ProblemSpec` owns the per-item encoding.**
  `problem_from_config(config)` resolves the spec from `benchmark.name` and
  `benchmark.representation`. `__init__` calls `self.problem.setup_dataset(self)`
  (e.g. forcing forces `temporal_samples = 128`), and `__getitem__` calls
  `self.problem.build_item(...)`.
- Pairs are **sorted by lead time** (`t_j - t_s`) to support curriculum slicing
  via `set_curriculum_fraction(frac)`.
- **Global normalization**: `T̃ = (T − μ_global) / σ_global`, with
  `μ_global, σ_global` computed once over the *training* sims only and
  baked into every checkpoint. **Do not switch to per-sample normalization.**
- **Dataset item is a dict**, not a tuple:
  ```
  {"spatial", "cond_static", "forcing_seq", "Y", "T_stats"}
  ```
  `forcing_seq` is always present — an empty `(0, 0)` tensor in `bins` mode.
- **Tensor shapes are not fixed across benchmarks.** They are owned by the
  `ProblemDims` of the active `(benchmark, representation)` pair. See the
  contract table below (and `tests/test_problems.py: CONTRACTS`, the single
  source of truth). Do not hard-code the old `(…, 20)` / `(23,)` / `(64, 5)`
  shapes.
- **Spatial input** (channels-last `(Nx, Ny, C)`): always starts
  `[T̃_source, x_norm, y_norm, s_y, …]`. `temporal_encoder` keeps just the lean
  channels; `bins` appends `Q_y_bin_0..15` (the signed temporal integral of the
  forcing over sub-intervals of `[t_s, t_j]`). `source`/`source_itr`/`interfaces`
  add patch / `R_c(y)` / interface channels. `s_y_channel` (index 3, or 5 for
  `interfaces`) is the channel the learned forcing multiplies against.
- **Static conditioning vector** is `cond_static_dim`-wide per benchmark (11
  forcing, 7 source, 10 source_itr, 4 interfaces — see table). Built by the
  spec, not by a single global `COND_STATIC_DIM`. The `COND_STATIC_DIM = 23`,
  `TEMPORAL_SAMPLES = 64`, `TEMPORAL_TOKEN_DIM = 5` constants and the `(23,)`
  docstring still in `data/dataset.py` are **legacy/stale** — trust
  `ProblemDims`.
- **Temporal forcing sequence** (`temporal_encoder`): `(128, 2)` tokens
  `[r_m, a_m / A_ref]` over `[t_s, t_j]`, with `A_ref = 300.0`.
- **Tensor layout convention** project-wide: spatial is `(B, Nx, Ny, C)`
  (channels-last). Never transpose to channels-first outside internal FNO blocks.

### (benchmark, representation) contract table

`in_ch` = spatial channels; `cond` = `cond_static_dim`;
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

`bins` `in_ch` always equals `temporal_encoder` `in_ch + 16`. `source_itr` adds
one `R_c(y)` channel (spatial index 4) on top of `source`.

---

## Scope assumptions — FNO model (`src/operators/fno2d.py`)

- **Forward signature**: `model(spatial, cond_static, forcing_seq=None) →
  y_pred`. The temporal encoder is skipped when `use_temporal_encoder=False`
  (bins mode), and the lift then sees `in_channels` alone.
- **Architecture**: optional temporal forcing encoder + learned forcing
  injection + Lift → pad → N Fourier blocks (SpectralConv2d + 1×1 Conv +
  Conditional Instance Norm + GELU + dropout) → unpad → projection MLP.
- **Base spatial channels** are `in_channels` from the resolved `ProblemDims`.
  When the temporal encoder is on, the model appends `forcing_spatial_dim`
  learned channels internally via `forcing_field_k = s_y[..., s_y_channel] · z_a[k]`,
  where `z_a` is projected from the temporal embedding `h_a`. With
  `use_forcing_time_aug=True` (forcing, interfaces), `h_a` is augmented with
  `cond_static[:, 0:2] = [t_bar_norm, t_s_norm]` before the spatial projection.
- **Conditioning**: `TemporalForcingEncoder(forcing_seq)` emits `h_a`
  (`forcing_embed_dim=64` by default). `ConditioningMLP` consumes
  `[cond_static, h_a]` and emits per-layer `(γ, β)` for Conditional Instance
  Norm. The head is soft identity-initialized so the model starts near an
  unconditioned FNO while gradients still flow through the forcing path.
- **Legacy `__init__` defaults** (`in_channels=20`, `cond_static_dim=15`,
  `TemporalForcingEncoder(token_dim=5)`) are stale placeholders; the real values
  are passed from the resolved `ProblemDims`/config at construction. Do not rely
  on the defaults.
- **Spectral mode invariant**: `2 · modes1 ≤ Nx_freq`. Always use distinct
  `modes1` and `modes2`. No bare `modes` key — that was 1D-only and has been
  removed from the 2D configuration.
- The model has **no benchmark-specific branches** beyond what flows through
  `cond_static`, `forcing_seq`, and the spatial channels. New families or
  benchmarks are added by extending the boundary-forcing / source registries and
  the relevant `ProblemSpec`, not by adding logic to the model.

---

## File structure (entry points)

- Benchmarks: `problems/base.py` (`ProblemSpec`, `ProblemDims`,
  `empty_forcing_seq`), `problems/registry.py` (`REGISTRY`, `REPRESENTATIONS`,
  `get_problem`), `problems/{forcing,interfaces,source,source_itr}.py`
- Physics: `src/physics/fv_solver_2d.py`, `src/physics/mms_2d.py`,
  `src/physics/boundary_forcing.py`, `src/physics/init_conditions.py`,
  `src/physics/internal_source.py`
- Model: `src/operators/fno2d.py`, training in `src/operators/train.py`,
  eval in `src/operators/eval.py`, losses in `src/operators/losses.py`
- Data: `data/generate_dataset.py`, `data/dataset.py`
- Config: `conf/config.yaml`, `conf/benchmark/*.yaml`,
  `conf/representation/*.yaml`
- Entry script: `scripts/run_train_fixed.py`
- Experiment diagnostics: `scripts/inspect_val_pairs.py` (benchmark-aware),
  `scripts/run_eval.py`, `scripts/write_test_records.py`, `visual/*.py`
- Cluster: `slurm/*.sbatch`
- Tests: `tests/test_*.py` (`pytest tests/` from repo root; `test_problems.py`
  is the contract table's source of truth)
- Visualization: `visual/cli.py` (dispatcher) + `visual/dataset_plots.py`,
  `visual/physics_plots.py`, etc.

When adding functionality, place it in the existing module that already owns
the concern. Do not split a single concern across new modules.

---

## When making changes

1. Read `CLAUDE.md` for the full conventions list and known gotchas.
2. Confirm the active `(benchmark, representation)` pair and resolve dims via
   `get_problem(...)` / `problem_from_config(...)`; do not assume a fixed shape.
3. Locate the existing function/class that owns the concern; edit there. For
   benchmark-specific behavior, edit the relevant `ProblemSpec`.
4. If a config knob changes, add a default in `conf/config.yaml`.
5. Update or add a test under `tests/` for any solver, loss, dataset, or
   model change. Contract changes must update `tests/test_problems.py`.
6. Run `pytest tests/ -q` before claiming the change works.
7. For experiment conclusions, cite the exact artifact path, the
   `(benchmark, representation)` pair, and the metric convention
   (`train/val_rel_l2` normalized-space percent vs eval physical percent).
8. Stop. Do not commit. Do not push.
