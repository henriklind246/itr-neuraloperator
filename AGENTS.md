# AGENTS.md — guidance for Codex (and other coding agents)

This file is the entry point for AI coding agents working in this repo. Read it
in full before making any changes. For more detail on conventions and gotchas,
also read `CLAUDE.md` and `PROJECT_OVERVIEW.md`.

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

**Initial condition** is set per-sim in `data/generate_dataset.py` via
`random_ic(...) + T_right` so it is consistent with the right Dirichlet.

This is a **forward** problem: given an initial field T(x, y, 0) and the
temporal-conditioning vector, predict T(x, y, t) for any future time.

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

- **Currently 2000 sims**, 70/15/15 train/val/test split by sim ID with seed 0
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
- **Time grid**: `t_final = 0.3`, `dt = 0.005`, `save_stride = 2` →
  `Nt_saved = 31`. Flux-active window for `sin` is `[t_on, t_off] = [0, 0.2]`.
- **`dt.npy` is mandatory**: the solver `dt` is saved alongside `t_grid.npy`
  because `t_grid[1] - t_grid[0] = solver_dt × save_stride` is the snapshot
  cadence, not the solver dt. The cond-vec encoder uses solver dt to
  normalize `tau` / `dt_n` against the same bounds the samplers used.
  `data.dataset.load_solver_dt(t_grid_path)` reads it; `train.py` and
  `eval.py` pass it through `create_dataloaders(dt=...)`.
- **Per-sim metadata schema** (saved to `sim_params.npy`):
  ```
  {R_c, T0, temporal_family, temporal_params, spatial_family, spatial_params}
  ```
  `temporal_params` is family-dependent and matches the corresponding
  `TEMPORAL_BUILDERS[family]` kwargs verbatim. `T0` is `(Nx, Ny) float32`
  centered around `T_right`.

---

## Scope assumptions — dataset / dataloader (`data/dataset.py`)

- `SnapshotPairDataset` enumerates **all-to-all snapshot pairs** within each
  sim from a uniformly subsampled set of `n_snapshots` time indices.
- Pairs are **sorted by lead time** (`t_j - t_s`) to support curriculum slicing
  via `set_curriculum_fraction(frac)`.
- **Global normalization**: `T̃ = (T − μ_global) / σ_global`, with
  `μ_global, σ_global` computed once over the *training* sims only and
  baked into every checkpoint. **Do not switch to per-sample normalization.**
- **Spatial input**: `(Nx, Ny, 3) = [T̃_source, x_norm, y_norm]`.
- **Conditioning vector**: `(28,)` — single source of truth is `COND_DIM` in
  `data/dataset.py`. Layout (do not reorder; the FNO conditioning MLP depends
  on it):
  ```
  [0:3]    base:           t_bar_norm, t_s_norm, R_c_norm
  [3:7]    spatial onehot: uniform, patch, gaussian, triangle
  [7:11]   spatial params: y_c, w, sigma_y, ell  (zero-padded per family)
  [11:15]  temporal onehot: sin, exp, pulse_train, exp_train
  [15:28]  temporal params (1 + 4 pulse slots × 3): see encode_temporal_params
  ```
- Use the `build_cond_vector` helper in `data/dataset.py` to build cond
  vectors anywhere outside `__getitem__` (e.g. inference plotting). Do not
  duplicate the slot logic.
- **Tensor layout convention** project-wide: spatial is
  `(B, Nx, Ny, C)` (channels-last). Never transpose to channels-first outside
  internal FNO blocks.

---

## Scope assumptions — FNO model (`src/operators/fno2d.py`)

- **Forward signature**: `model(spatial, cond) → y_pred` with shapes above.
- **Architecture**: Lift → pad → N Fourier blocks (SpectralConv2d + 1×1 Conv +
  Conditional Instance Norm + GELU + dropout) → unpad → projection MLP.
- **Conditioning**: a small `ConditioningMLP` consumes the 28-dim cond vector
  and emits per-layer `(γ, β)` for Conditional Instance Norm. Identity-init
  (γ=1, β=0) so the model starts as an unconditioned FNO.
- **Spectral mode invariant**: `2 * modes1 ≤ Nx_freq`. Always use distinct
  `modes1` and `modes2`. No bare `modes` key — that was 1D-only and has been
  removed from search spaces.
- The model has **no awareness of family semantics** beyond what flows in
  through the cond vector. New forcing/spatial families are added by
  extending the boundary-forcing registry and the cond-vec encoder, not by
  changing the model.

---

## File structure (entry points)

- Physics: `src/physics/fv_solver_2d.py`, `src/physics/mms_2d.py`,
  `src/physics/boundary_forcing.py`
- Model: `src/operators/fno2d.py`, training in `src/operators/train.py`,
  eval in `src/operators/eval.py`, losses in `src/operators/losses.py`
- Data: `data/generate_dataset.py`, `data/dataset.py`
- Config: `conf/config.yaml` (+ `conf/search_space/*.yaml` for Optuna sweeps)
- Entry scripts: `scripts/run_train_fixed.py`, `scripts/run_train.py`
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
6. Stop. Do not commit. Do not push.
