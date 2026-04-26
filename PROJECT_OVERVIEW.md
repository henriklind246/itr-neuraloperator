# Project Overview: Time-Conditioned 2D FNO for Transient Multilayer Heat Conduction

## 1. Problem Statement

This project builds a **forward** surrogate for transient multilayer heat conduction with interfacial thermal resistance using a time-conditioned **2D Fourier Neural Operator (FNO)**. A conservative 2D finite volume solver generates training data by simulating unsteady heat conduction on a rectangular domain with two vertical material layers separated by a contact-resistance interface. The FNO learns the operator mapping:

    G_theta(T(x, y, t_s), x, y, t_bar, t_s, A, f, R_c) -> T(x, y, t_j)

given a source-time temperature field and a conditioning vector (lead time, absolute source time, flux amplitude, flux frequency, contact resistance).

The model must generalize across a **family of PDEs**, because each simulation has different forcing parameters and contact resistance. There are 4000 simulations with 3 varying parameters, split 70/15/15 by simulation ID.

A 1D baseline (`fno1d.py`, `fv_solver_1d.py`, `mms_1d.py`) remains in the repo for reference but is no longer the primary target.

---

## 2. The PDE

The governing equation is the 2D unsteady heat equation with spatially-varying coefficients:

    rho(x) * cp(x) * dT/dt = div [k(x) * grad T] + s(x, y, t)

- **T(x, y, t)**: temperature
- **k(x)**: thermal conductivity (piecewise constant per layer; x-dependent only)
- **rho(x), cp(x)**: density, specific heat (same x-piecewise structure)
- **s(x, y, t)**: volumetric source (zero in data generation; nonzero only for MMS)

**Domain**: `(x, y) in [0, 1] x [0, 1]`. Two vertical material slabs with interface at **x = 0.5**; material properties are invariant along y.

**Fixed material properties** (same for all data-generation simulations):
- Layer 1 (x in [0, 0.5]): k1 = 2, rho1 = 1, cp1 = 1
- Layer 2 (x in [0.5, 1]): k2 = 1, rho2 = 1, cp2 = 1

**Boundary conditions**:
- **Left** (x = 0, all y): Neumann flux q(t) = W(t) * A * sin(2*pi*f*t), where W(t) is a Tukey window active on `[t_on, t_off] = [0, 0.2]`. The flux is uniform along the full left edge.
- **Right** (x = 1, all y): Dirichlet, T = 300 K (constant).
- **Top/Bottom** (y = 0 and y = 1, all x): **Adiabatic** (zero flux). This is the key new 2D BC.

**Interface condition at x = 0.5 (all y)**: Temperature jump `dT = R_c * q_interface`, where R_c is the interfacial thermal contact resistance. The interface flux is continuous across the face.

---

## 3. Finite Volume Solver

**File**: `src/physics/fv_solver_2d.py` (primary)
**Baseline**: `src/physics/fv_solver_1d.py` (still available; shares the same layer and flux utilities)

### 3.1 Overview

The 2D solver is a **conservative** Crank-Nicolson finite volume scheme on a uniform isotropic grid. It is **second-order accurate in both space and time**, verified via MMS. CN is unconditionally stable.

### 3.2 Grid

- **Uniform isotropic grid**: `hx = hy = h`, enforced in the constructor (deliberate simplification; the FV method generalizes to non-uniform grids).
- **Nodes**: `(Nx, Ny)` on `[a, b] x [c, d]`, uniformly spaced.
- **Faces**: Faces are the midpoints between adjacent nodes in each direction. x-direction faces carry the material interface(s); y-direction faces are always intra-layer.
- **Interface constraint**: Material interfaces must be **vertical slabs** (x = const). Interfaces must lie on x-direction cell faces (not on nodes, not in the two face slots adjacent to the domain boundary). Off-center placements are supported: each interface face stores per-side offsets (h_L, h_R) so the flanking control volumes widen/narrow correctly.

### 3.3 Material Fields (built during `__init__`)

- **Per-node material arrays**: `rho_nodes, cp_nodes, k_nodes` of shape `(Nx, Ny)`, invariant along y since interfaces are vertical.
- **Control volume dimensions**: `dx[i]` and `dy[j]` of length Nx / Ny, with `h/2` at the domain boundaries and `h` for interior nodes. Off-center interfaces patch the two flanking `dx` entries with `(h/2 + h_L)` and `(h_R + h/2)`.
- **Face conductance**:
  - **x-faces** (shape `(Nx-1, Ny)`): harmonic mean plus contact resistance when a face carries an interface — `G = 1 / (h/(2*k_L) + R_c + h/(2*k_R))`. Non-interface x-faces reduce to `G = k/h`.
  - **y-faces** (shape `(Nx, Ny-1)`): always `G = k/h` (no interfaces in y).

### 3.4 Crank-Nicolson Coefficients

For each interior cell (i, j), the CN stencil couples to its four neighbors (W, E, S, N) via per-cell ratios `r_w, r_e, r_s, r_n` that depend on the local face conductances and the cell's thermal capacity (C = rho * cp * dx_i * dy_j). The `1/2` factor from CN time-centering is folded into these coefficients.

### 3.5 Sparse Matrix A (assembled once, factored once)

The CN system `A * T^{n+1} = rhs` is assembled as a **sparse CSR matrix of size `(Nx*Ny, Nx*Ny)`** with a 5-point stencil (center + W, E, S, N).

| Cell type | Purpose |
|-----------|---------|
| **Interior** (i = 1..Nx-2, j = 1..Ny-1 not on a Dirichlet column) | Implicit side of CN: `(1 + r_w + r_e + r_s + r_n) * T_ij - r_w T_{i-1,j} - r_e T_{i+1,j} - r_s T_{i,j-1} - r_n T_{i,j+1}` |
| **Left Neumann boundary** (i = 0) | **Half-cell FV energy balance**: CN time-averaged flux on a control volume of width `h/2`. The flux `q(t)` enters through the RHS as a time-averaged `0.5 * (q^n + q^{n+1})` term. This replaces the old 3-point one-sided `[3, -4, 1]` stencil used before the April 2026 refactor. The same half-cell approach was applied to the 1D solver. |
| **Right Dirichlet boundary** (i = Nx-1) | Identity row: `T = T_right(t^{n+1})` set directly. |
| **Top/Bottom adiabatic boundaries** (j = 0 or j = Ny-1) | Standard half-cell control volumes with zero-flux face contributions; reflected into the stencil by omitting the corresponding neighbor coupling. |

**Why A is constant**: A depends only on geometry, materials, and `dt` — all fixed over the run. Only the RHS changes per step. The matrix is LU-factored once via `scipy.sparse.linalg.splu` and the factor is reused every step.

### 3.6 Time Step Selection

Same rule as 1D: `dt = min(dt_a, dt_b)` where `dt_a = lam_target * h^2 / alpha_max` (Fourier-number stability target) and `dt_b = 1/(20*f)` (Nyquist on the forcing frequency). For the dataset `dt` is user-overridden to **0.005** with `N = 100`, `t_final = 1.0`.

### 3.7 Time-Stepping (`cn_step`)

Each step builds the RHS as `rhs = B * T^n + boundary_and_source_contributions`, where `B` is the explicit-side CN companion matrix. The Dirichlet column becomes the identity row at assembly; the Neumann contribution is injected as the time-averaged flux at i = 0; MMS sources enter as `dt * 0.5 * (s^n + s^{n+1}) / C`. The solve is `A_factor.solve(rhs_flat)`.

### 3.8 Boundary Flux Function

Unchanged: `windowed_sin_flux(f, A, t_on, t_off, phase, tukey_alpha)` returns q(t) that is zero outside `[t_on, t_off]` and `W(tau) * A * sin(2*pi*f*t + phase)` inside, with a Tukey taper.

### 3.9 1D Baseline

`fv_solver_1d.py` (`FVSolver1D`) is still in the repo and shares utilities (`Layer1D`, `windowed_sin_flux`, `compute_dt`) with the 2D solver. Its matrix is tridiagonal `(3, N)` banded, solved via `scipy.linalg.solve_banded`. The left Neumann BC was refactored to the same half-cell FV energy balance used in 2D; the old `[3, -4, 1]` one-sided stencil is gone.

---

## 4. Method of Manufactured Solutions (MMS)

**File**: `src/physics/mms_2d.py` (primary) | baseline `src/physics/mms_1d.py`

### 4.1 Procedure (unchanged)

Pick an exact T*(x, y, t), derive the forcings (source, left flux, right BC) by substituting into the PDE, feed those forcings into the unmodified solver, and compare numerical output to T* at the final time. Convergence in h and dt should be O(h^2) and O(dt^2).

### 4.2 2D Test Cases

| Case | Purpose | Manufactured T* structure |
|------|---------|---------------------------|
| **y-independent** (`run_mms_y_independent`) | Verify 2D infrastructure reproduces 1D physics | `T*(x, y, t) = 300 + A sin(omega t) (b - x)^4` — no y-dependence |
| **Full 2D** (`run_mms_2d`) | Exercise y-Laplacian coupling | 1D base + `B sin(omega t) phi(x) cos(pi y / (d - c))` correction satisfying adiabatic y-BCs |
| **Interface + R_c** (`run_mms_2d_interface`) | Vertical interface with contact resistance | Piecewise 2D with flux continuity and `dT = R_c * q_I` at x = 0.5, y-correction term in each layer |
| **Off-center interface** (`space_order_test_2d_off_center_interface`) | Off-midpoint interface placement | Same piecewise structure with x_I ~= 0.4734 (not on a face midpoint) |

All four cases verify **second order in space and in time**. Convergence tests follow the same principle as 1D: fix one discretization parameter very fine while varying the other, then compute pairwise orders `p = log(e_i / e_{i+1}) / log(h_i / h_{i+1})` and report the mean.

### 4.3 1D Baseline

`mms_1d.py` is still present with its single-layer and two-layer (interface + R_c) cases. The `t_final = 0.37` choice (rather than 1.0) is preserved so `sin(omega t)` stays nonzero at the measurement time.

---

## 5. Data Generation

**File**: `data/generate_dataset.py`

### 5.1 Parameter Sampling

4000 simulations. Three parameters are sampled via 3D Latin Hypercube Sampling (LHS) for space-filling coverage:

| Parameter | Symbol | Range | Sampling |
|-----------|--------|-------|----------|
| Flux amplitude | A | (50, 300) | Uniform |
| Flux frequency | f | (1, 20) | **Log-uniform**: `x = lo * (hi/lo)^u, u in [0, 1]` |
| Contact resistance | R_c | (0.05, 1.0) | Uniform |

Frequency is log-uniform because penetration depth `delta ~ 1/sqrt(f)`; log spacing gives more uniform coverage of the **physics** space. Amplitude and R_c have linear effects, so uniform sampling suffices. Materials (k1 = 2, k2 = 1, rho = cp = 1) are fixed across all sims.

### 5.2 Initial Conditions (now truly 2D)

Each simulation gets a unique, separable, y-varying IC:

    T0(x, y) = fx(x) * fy(y)

where

    fx(x) = cx0*cos(pi x/(2 Lx)) + cx1*sin(pi x / Lx) + cx2*cos(2 pi x / Lx)
    fy(y) = cy0*cos(pi y/(2 Ly)) + cy1*sin(pi y / Ly) + cy2*cos(2 pi y / Ly)

with `cx0, cy0 ~ U(0.5, 1.5)` and `cx{1,2}, cy{1,2} ~ U(-0.3, 0.3)`. This gives smooth, physically plausible 2D fields that vary across sims in both directions. The 300 K Dirichlet baseline is added at runtime by the solver.

### 5.3 Simulation Execution

For each sampled `(A, f, R_c)`, a fresh `FVSolver2D` is constructed with the fixed layer geometry and solved. Fixed simulation parameters: `a = 0, b = 1, c = 0, d = 1, Nx = Ny = 100, dt = 0.005, t_final = 1.0, t_on = 0, t_off = 0.2, tukey_alpha = 0.5`.

Solve runs at full dt internally; trajectories are **subsampled by `save_stride = 5`** before saving to cap file size.

### 5.4 Output Files

| File | Shape | Description |
|------|-------|-------------|
| `trajectories.npy` | `(4000, Nt_saved, 100, 100)` | All simulation trajectories: `(num_sims, Nt_saved, Nx, Ny)`. With `dt = 0.005, t_final = 1.0, save_stride = 5`, `Nt_saved = 41`. |
| `x_grid.npy` | `(100,)` | x-node coordinates |
| `y_grid.npy` | `(100,)` | y-node coordinates (new) |
| `t_grid.npy` | `(Nt_saved,)` | Saved time steps |
| `sim_params.npy` | `(4000,)` object array | Each entry: `(amp, freq, T0_array, R_c)` with `T0_array` of shape `(Nx, Ny)` |

---

## 6. Dataset and Normalization

**File**: `data/dataset.py`

### 6.1 SnapshotPairDataset

`SnapshotPairDataset` converts raw trajectories into (input, target) pairs:

1. For each sim, uniformly subsample `n_snapshots` time indices from the saved trajectory.
2. Enumerate all forward-in-time pairs `(s, j)` with `j > s`.
3. Globally sort all pairs by lead time `t_j - t_s` to support curriculum learning.

With `n_snapshots = 20` (config default), each sim contributes `C(20, 2) = 190` pairs; 2800 training sims → 532K training pairs. Effective data diversity is **2800 independent simulations** (pairs from the same sim share the trajectory).

### 6.2 Per-Sample Output (4-tuple)

| Tensor | Shape | Description |
|--------|-------|-------------|
| `x_spatial` | `(Nx, Ny, 3)` | Channel 0: `T_source` globally normalized; Channel 1: `x_norm` in [0, 1]; Channel 2: `y_norm` in [0, 1] |
| `cond` | `(5,)` | `[t_bar_norm, t_s_norm, A_norm, f_norm, R_c_norm]` |
| `Y` | `(Nx, Ny, 1)` | Target temperature, globally normalized |
| `T_stats` | `(2,)` | `[mu_global, sigma_global]` for denormalization at eval time (global constants, identical across samples) |

### 6.3 Normalization

**Global temperature normalization**: `T_tilde = (T - mu_global) / (sigma_global + eps)`, with `mu_global, sigma_global` computed once from the **training set only** (avoids leakage). `eps = 1e-6`. Not per-sample z-score — preserves absolute temperature scale so BC- and IC-dependent dynamics remain learnable.

**Conditioning normalization** (min-max to [0, 1]):
- `t_bar_norm = (t_j - t_s) / t_final`
- `t_s_norm = t_s / t_final`
- `A_norm = (A - 50) / (300 - 50)`
- `f_norm = (f - 1) / (20 - 1)`
- `R_c_norm = (R_c - 0.05) / (1.0 - 0.05)`

### 6.4 Why `t_s` Is in the Conditioning Vector

The boundary flux is active only on `t in [0, 0.2]`. Without `t_s`, the model cannot distinguish pairs whose source is in the active-flux window from pairs entirely in the diffusion-only regime — two pairs with the same lead time can have very different physics. This was diagnosed from a severe overfitting episode on the 1D baseline (0.19% train / 17% val) that disappeared once `t_s` was added to the conditioning. The same design is kept in 2D.

### 6.5 Curriculum Learning

Pairs are sorted by lead time. During training a curriculum warmup (default: 35 epochs) exposes longer lead times gradually: at epoch e, the fraction of pairs exposed is `frac = min(1.0, (e + 1) / warmup_epochs)`; `set_curriculum_fraction(frac)` uses `np.searchsorted` on the sorted lead times to pick the cutoff index. The model learns short-horizon predictions first.

### 6.6 Input Noise Augmentation

When `noise_std > 0`, Gaussian noise is added to the normalized source temperature (training set only). Acts as a regularizer. Default `noise_std = 0.0`.

### 6.7 Data Splits

Simulation IDs are split 70/15/15 (train/val/test) using a **fixed seed = 0** so that different model-init seeds always see the same partition. Splitting by sim ID (not by pair) is critical — pairs from the same sim are correlated and would leak across splits.

---

## 7. Neural Operator: FNO2d

**File**: `src/operators/fno2d.py`

### 7.1 Operator Learning Task

    G_theta(T_tilde(x, y, t_s), x, y, t_bar, t_s, A, f, R_c) -> T_tilde(x, y, t_j)

Given a globally-normalized 2D source field and a 5D conditioning vector, predict the globally-normalized 2D target field.

### 7.2 Architecture

Approximate parameter count depends on `modes1, modes2, width, n_layers, cond_hidden`. Default config: `modes1 = modes2 = 16, width = 64, n_layers = 4, cond_hidden = 256`.

Forward pass (input shapes for a batch of size B):

| Stage | Operation | Shape |
|-------|-----------|-------|
| **Lift** | `Linear(3, width)` on spatial input | `(B, Nx, Ny, 3) -> (B, Nx, Ny, width)` |
| **Permute** | Channels-first | `(B, Nx, Ny, width) -> (B, width, Nx, Ny)` |
| **Pad** | Zero-pad both x and y by 8 | `(B, width, Nx, Ny) -> (B, width, Nx+8, Ny+8)` |
| **`n_layers` Fourier Blocks** | `SpectralConv2d + Conv2d bypass + CIN + GELU + Dropout` | unchanged |
| **Unpad** | Remove padding | `(B, width, Nx+8, Ny+8) -> (B, width, Nx, Ny)` |
| **Permute** | Channels-last | `(B, width, Nx, Ny) -> (B, Nx, Ny, width)` |
| **Project** | `Linear(width, 128) + GELU + Dropout + Linear(128, 1)` | `(B, Nx, Ny, width) -> (B, Nx, Ny, 1)` |

### 7.3 SpectralConv2d

1. **FFT**: `torch.fft.rfft2(x, dim=(-2, -1))` → complex tensor of shape `(B, C, Nx, Ny//2 + 1)`.
2. **Truncate**: keep `modes1` along x (positive and negative halves) and `modes2` along y (one-sided). A guard raises if `2 * modes1 > Nx_freq` to prevent the positive/negative x-slices from silently overlapping.
3. **Channel mixing**: two learned complex weight tensors of shape `(C_in, C_out, modes1, modes2)` (one for positive-x modes, one for negative-x modes), applied via `torch.einsum("bixy,ioxy->boxy", ...)`.
4. **Zero high frequencies**: modes outside the kept block are zero (implicit low-pass). This is what makes the FNO resolution-invariant.
5. **IFFT**: `torch.fft.irfft2(out, s=(Nx, Ny), dim=(-2, -1))`.

**Spectral dropout**: when `spectral_dropout > 0`, random Fourier modes are zeroed during training (independently sampled per forward pass). The same mask is applied to the positive-x and negative-x weight blocks.

### 7.4 Conv2d Bypass

Each Fourier block includes a `nn.Conv2d(width, width, kernel_size=1)` that operates in physical space; its output is summed with the spectral output before normalization. Captures local/high-frequency features the truncated spectral path discards.

### 7.5 Padding

FFT assumes periodic signals. Temperature fields are non-periodic (flux left, Dirichlet right, adiabatic top/bottom). Zero-padding by 8 on both x and y reduces spectral leakage at boundaries.

### 7.6 Conditional Instance Normalization

`InstanceNorm2d` (no learned affine) followed by an externally-supplied `gamma, beta` of shape `(B, C)`, broadcast as `(B, C, 1, 1)`:

    CIN(x) = gamma[:, :, None, None] * InstanceNorm2d(x) + beta[:, :, None, None]

This is how the 5D conditioning vector modulates internal representations.

### 7.7 ConditioningMLP

Maps `cond` (B, 5) → per-layer `(gamma, beta)`:

    Linear(5, cond_hidden) -> SiLU -> Linear(cond_hidden, cond_hidden) -> SiLU -> Linear(cond_hidden, n_layers * 2 * width)

Reshaped to `(B, n_layers, 2, width)`.

**Identity init**: the head layer has `weight = 0` and bias set so `gamma = 1, beta = 0` for every layer at start. The model begins as a plain, unconditioned 2D FNO; the conditioning pathway influence grows during training.

Note: the conditioning MLP uses **SiLU** internally, while the main trunk uses **GELU**. They are different on purpose.

---

## 8. Loss Functions

**File**: `src/operators/losses.py`

### 8.1 SpatiallyWeightedMSE (training loss)

    L = mean(w(x, y) * (y_pred - y_true)^2)

where `w(x, y)` is a weight profile:
- Baseline: `w = 1.0` everywhere.
- Interface boost: `w = interface_weight` (default 10.0) on full **columns** within `|x - interface_x| <= interface_half_width` — i.e., the boosted region is a vertical slab spanning all y, matching the physical interface geometry.
- Weights are normalized so `mean(w) = 1.0`, keeping the loss magnitude comparable to plain MSE regardless of `interface_weight`.

When `interface_weight = 1.0`, this reduces to standard MSE. The buffer has shape `(1, Nx, Ny, 1)` to broadcast against `(B, Nx, Ny, 1)` predictions.

### 8.2 Interface-Specific Relative L2 (monitoring metric)

`build_interface_mask` returns a boolean tensor of shape `(Nx, Ny)`, True on the interface slab.

    iface_rel_l2 = sqrt(mean((pred - true)^2 over interface) / mean(true^2 over interface)) * 100%

`compute_interface_rel_l2` indexes as `y_pred[:, iface_mask, :]`, collapsing to `(B, N_iface, 1)` via PyTorch advanced indexing on contiguous dims. Logged during training but not part of the loss gradient — diagnostic for the discontinuity at the interface.

### 8.3 Training vs. Evaluation Metrics

- **Training loss**: `SpatiallyWeightedMSE` in normalized (z-score) space.
- **Training / validation metrics**: relative L2 in normalized space (used for early stopping and checkpoint selection).
- **Test metrics** (`eval.py`): relative L2 in **physical (denormalized) temperature space**. Denormalization: `T_phys = T_norm * (sigma_global + eps) + mu_global`. Scientifically meaningful error, in Kelvin. Note the `val_iface_rel_l2` (normalized) and `test_iface_rel_l2` (physical) magnitudes are not directly comparable.

---

## 9. Training Pipeline

**File**: `src/operators/train.py`

### 9.1 `run_one_seed`

Core per-seed training function:

1. **Data loading**: load trajectories, grids, sim_params. Split by sim ID (seed = 0, hardcoded). Compute global stats from training set only.
2. **Model construction**: `FNO2d` built from `model.parameters` in config.
3. **Resume logic**: if `fno2d_latest.pt` (sentinel) exists, resume from it. Validates optimizer/scheduler compatibility.
4. **Training loop** (up to 1500 epochs by default):
   - Update curriculum fraction if `curriculum_warmup > 0`.
   - Train one epoch: forward, loss, backward, gradient clipping, optimizer step.
   - Step the LR scheduler.
   - Every `validate_every` epochs (default 10): validate, check for improvement, update early-stopping counter.
   - Save `fno2d_latest.pt` every epoch (crash recovery).
   - Save `fno2d_best.pt` when validation improves.
5. **Completion**: delete `fno2d_latest.pt` sentinel. A run is "complete" when `fno2d_best.pt` exists and `fno2d_latest.pt` does not.

### 9.2 Optimizer

AdamW with `lr = 0.002`, `weight_decay = 1e-5`. Gradient clipping at `max_norm = 1.0`.

### 9.3 Learning Rate Scheduler: RIGNOThreePhase

Three phases (defaults over 1500 epochs):

| Phase | Fraction | LR behavior |
|-------|----------|-------------|
| Warmup | 0.02 (30 epochs) | Linear ramp from `0.05 * peak` to `peak` (0.0001 → 0.002) |
| Cosine decay | 0.88 (1320 epochs) | Cosine anneal from `peak` to `0.05 * peak` (0.002 → 0.0001) |
| Exponential decay | 0.10 (150 epochs) | Exponential decay from cosine floor to `0.005 * peak` (0.0001 → 0.00001) |

LR values are specified as ratios of `training.learning_rate`; the peak LR is the source of truth so Optuna sweeps scale the whole schedule cleanly.

### 9.4 Early Stopping

- `patience = 20` (counted in validation evaluations)
- `validate_every = 10`
- Effective tolerance: **200 epochs** of no validation improvement.

### 9.5 Checkpoints

`fno2d_best.pt` and `fno2d_latest.pt` both contain: epoch, config, seed, model_state, optimizer_state, scheduler_state, best_val, bad_epochs, mu_global, sigma_global. The global stats are saved so evaluation can denormalize without recomputing from data.

### 9.6 CSV Logging

`train_metrics.csv` logs per-epoch: `epoch, train_loss, train_rel_l2, train_iface_rel_l2, val_rel_l2, val_iface_rel_l2, lr, is_best`. Validation columns are empty on non-validation epochs.

### 9.7 Multi-Seed Training (`run_config_seeds`)

Trains the same config with multiple seeds (default `[42]`). Reports mean best validation loss across seeds — used as the Optuna objective.

---

## 10. Evaluation

**File**: `src/operators/eval.py`

1. Scans `runs/<experiment>/<config>/seed*/fno2d_best.pt` for all seed checkpoints.
2. Per seed: loads checkpoint, reconstructs `FNO2d`, builds test DataLoader with `n_snapshots_test = 40` (denser temporal coverage than training).
3. Evaluates in **physical space** (denormalized). `sigma_s` and `mu_s` broadcast as `[:, None, None, None]` against `(B, Nx, Ny, 1)` predictions.
4. Reports per-seed: `best_val` (from training, normalized-space), `test_rel_l2`, `test_iface_rel_l2` (physical-space).
5. Computes mean and sample std across seeds; saves `seed_report.json`.

---

## 11. Hyperparameter Optimization

**Files**: `scripts/run_train.py` (Optuna sweep via Hydra multirun), `scripts/run_train_fixed.py` (fixed config with CLI overrides).

The project uses Hydra for configuration and Optuna's TPE sampler for hyperparameter search. The objective is the mean best validation loss across seeds (minimized).

**Available search spaces** (in `conf/search_space/`):
- `medium.yaml` — full sweep. Sweeps **`modes1` and `modes2` independently**.
- `width_only.yaml`, `snapshots_only.yaml` — single-parameter ablations.
- `width_regularization.yaml` — combined width + regularization.
- `regularization.yaml` — regularization-only (dropout, spectral_dropout, weight_decay, noise_std).
- `dropout_only.yaml`, `spectral_dropout_only.yaml`, `noise_std_only.yaml`, `weight_decay_only.yaml` — isolated ablations.

MSI cluster support (`slurm/`): `train_fno_msi.sbatch` (Optuna sweep) and `train_fno_msi_fixed.sbatch` (single config). SQLite Optuna DB lives on node-local scratch (`/tmp/fno_runs_$SLURM_JOB_ID`) to avoid NFS locking, then results are copied back to persistent storage at job end.

---

## 12. Visualization

**Files**: `visual/cli.py` (CLI entry), `visual/_common.py` (registry, style, shared helpers), and per-group modules `visual/{physics,mms,training,dataset,sweep}_plots.py`.

Plot functions are grouped with a CLI dispatch system (physics, mms, training, data, sweep). Functions operate on 2D fields where applicable (`trajectory_heatmap`, `initial_conditions`, `prediction_vs_truth`, etc.). Sweep plots handle both 1D and 2D hyperparameter keys (e.g., `modes` vs. `modes1`) where needed for backward compatibility.

---

## 13. File Map

| File | Role |
|------|------|
| `src/physics/fv_solver_2d.py` | **2D conservative Crank-Nicolson FV solver** (FVSolver2D, 5-point sparse stencil, splu factor reuse, off-center interface support) |
| `src/physics/fv_solver_1d.py` | 1D baseline solver (Layer1D, FVSolver1D, windowed_sin_flux, compute_dt) — tridiagonal banded, same half-cell Neumann BC |
| `src/physics/mms_2d.py` | **2D MMS verification**: y-independent, full 2D, interface + R_c, off-center interface; spatial and temporal order tests |
| `src/physics/mms_1d.py` | 1D baseline MMS (single-layer + two-layer with interface resistance) |
| `src/physics/dt_convergence_study.py` | Temporal resolution analysis |
| `data/generate_dataset.py` | 3D LHS sampling, per-sim 2D solver construction, trajectory generation (4000 sims, 2D fields) |
| `data/dataset.py` | `SnapshotPairDataset` (all-to-all pairing, global normalization, curriculum), data loading, splitting |
| `src/operators/fno2d.py` | **Time-conditioned 2D FNO** (SpectralConv2d, ConditionalInstanceNorm2d, ConditioningMLP) |
| `src/operators/fno1d.py` | 1D baseline FNO (still referenced by 1D tests) |
| `src/operators/losses.py` | SpatiallyWeightedMSE (2D), interface mask builder, interface-specific rel L2 |
| `src/operators/train.py` | Training loop, RIGNOThreePhase scheduler, early stopping, multi-seed wrapper, resume logic |
| `src/operators/eval.py` | Test evaluation in physical space, seed report generation |
| `src/operators/utils.py` | Device resolution utility |
| `scripts/run_train.py` | Hydra + Optuna sweep entrypoint (includes Optuna 2.10.0 SQLite monkey-patch) |
| `scripts/run_train_fixed.py` | Fixed-config training with CLI overrides |
| `conf/config.yaml` | Main configuration (model, training, data, scheduler, loss) |
| `conf/paths/default.yaml` | Path resolution via PROJECT_ROOT env var |
| `conf/search_space/*.yaml` | Optuna search space definitions for ablations |
| `slurm/train_fno_msi.sbatch`, `train_fno_msi_fixed.sbatch`, `setup_env_msi.sh` | MSI GPU cluster scripts |
| `tests/test_fv_solver_2d.py`, `tests/test_mms_2d.py` | 2D physics and MMS tests |
| `tests/test_fv_solver_1d.py`, `tests/test_mms_1d.py`, `tests/test_fno1d.py` | 1D baseline tests |
| `visual/cli.py`, `visual/_common.py`, `visual/{physics,mms,training,dataset,sweep}_plots.py` | Visualization suite split by group with CLI dispatch |

---

## 14. Key Design Decisions Summary

| Decision | Rationale |
|----------|-----------|
| **2D FV surrogate with vertical slab layers (x = const)** | Real 2D physics with lateral (y) diffusion while keeping material interfaces simple; y-Laplacian is required for a genuine 2D FNO mapping (otherwise y is trivial) |
| **Uniform isotropic grid (hx = hy)** | Simplifies stencil derivation and matrix assembly; FV method generalizes to non-uniform grids if needed later |
| **Interfaces on x-direction cell faces only** | Conservative FV requires unambiguous material assignment per face; nodes or y-faces on interfaces would complicate the stencil or break the vertical-slab geometry |
| **Off-center interface support via per-face `(h_L, h_R)`** | Enables MMS verification at arbitrary x_I without requiring grids aligned to the interface |
| **Harmonic mean + R_c for x-face conductance** | Physically correct series resistance: half-cell in layer L + contact resistance + half-cell in layer R |
| **Adiabatic top/bottom BCs** | Matches a y-translation-invariant physical setup (infinite slab in y approximated by symmetric adiabatic boundaries); keeps the 1D baseline recoverable as a special case |
| **Half-cell FV energy balance at the left Neumann BC** | Standard FV (Patankar/Versteeg), energy-conservative, second-order accurate, and keeps A tridiagonal in 1D / 5-point in 2D. Replaces the earlier 3-point `[3, -4, 1]` one-sided FD stencil |
| **CN source term time-centered `0.5 * (s^n + s^{n+1})`** | Preserves 2nd-order temporal accuracy; one-sided evaluation drops to 1st-order |
| **Sparse CSR matrix + one-time `splu` factor** | A depends only on fixed geometry/materials/dt; factoring once and reusing every step is O(nnz) per solve instead of O(Nx^2 * Ny^2) naive |
| **Global (not per-sample) temperature normalization** | Preserves absolute temperature scale required for physics; per-sample z-score would erase hot/cold differences and destroy BC/IC-dependent dynamics |
| **`y_norm` added as a third spatial channel** | Lets the FNO condition on absolute y-position (adiabatic edges vs. interior) without ambiguity |
| **Two independent Fourier mode counts `modes1, modes2`** | x and y have very different dynamics (interface vs. smooth diffusion); decoupling mode counts lets Optuna search them separately |
| **`t_s` in conditioning vector** | Distinguishes flux-active (`t < 0.2`) from diffusion-only (`t > 0.2`) regimes; without it the model cannot generalize and memorizes training trajectories (carried over from a 1D overfitting diagnosis) |
| **Data split by simulation ID, not by pair** | Prevents leakage — pairs from the same trajectory are correlated and would inflate test metrics |
| **Curriculum warmup on lead time** | Start with easy short-horizon predictions; gradually expose harder long-horizon pairs as the model stabilizes |
| **Identity init for CIN (`gamma = 1, beta = 0`)** | Model starts as plain unconditioned FNO; conditioning pathway influence grows organically instead of destabilizing early learning |
| **Spatial loss weighting on the interface slab** | Concentrates gradient signal on the physically important temperature jump; normalized to `mean(w) = 1` so no LR retuning is needed when the boost factor changes |
| **Log-uniform sampling for frequency** | Penetration depth `delta ~ 1/sqrt(f)`; log-uniform gives uniform physics-space coverage |
| **Fixed data-split seed (0); variable model seed** | Isolates model-initialization variance from data-partition variance |
| **Fixed materials (k1, k2, rho, cp) not in conditioning** | They are constant across all 4000 sims; conditioning on constants wastes capacity. Adding material variability would require extending the conditioning vector and data generation |
