# Project Overview: Time-Conditioned FNO for Inverse Heat Conduction

## 1. Problem Statement

This project solves the Inverse Heat Conduction Problem (IHCP) using a time-conditioned 1D Fourier Neural Operator (FNO). A conservative finite volume solver generates training data by simulating unsteady heat conduction through a two-layer rod with interfacial thermal resistance. The FNO learns the operator mapping: given a temperature snapshot T(x, t_s) at a source time, predict T(x, t_j) at a future target time, conditioned on the lead time, source time, and simulation parameters (amplitude, frequency, contact resistance).

The model must generalize across a family of PDEs, not just one, because each simulation has different forcing parameters and interface resistance. There are 4000 simulations with 3 varying parameters, split 70/15/15 by simulation ID.

---

## 2. The PDE

The governing equation is the 1D unsteady heat equation with spatially-varying coefficients:

    rho(x) * cp(x) * dT/dt = d/dx [k(x) * dT/dx] + s(x,t)

- **T(x,t)**: temperature
- **k(x)**: thermal conductivity (piecewise constant per layer)
- **rho(x)**: density
- **cp(x)**: specific heat capacity
- **s(x,t)**: volumetric source term (zero in data generation; nonzero only for MMS verification)

**Domain**: x in [0, 1], two material layers with interface at x = 0.5.

**Fixed material properties** (same for all simulations):
- Layer 1 (x in [0, 0.5]): k1 = 2, rho1 = 1, cp1 = 1
- Layer 2 (x in [0.5, 1]): k2 = 1, rho2 = 1, cp2 = 1

**Boundary conditions**:
- Left (x = 0): Neumann BC with prescribed heat flux q(t) = W(t) * A * sin(2*pi*f*t), where W(t) is a Tukey window active during [t_on, t_off] = [0, 0.2]
- Right (x = 1): Dirichlet BC, T = 300 K (constant)

**Interface condition at x = 0.5**: Temperature jump dT = R_c * q_interface, where R_c is the interfacial thermal contact resistance. The interface flux q_interface is continuous across the face.

---

## 3. Finite Volume Solver

**File**: `src/physics/fv_solver_1d.py`

### 3.1 Overview

The solver is a **conservative** Crank-Nicolson finite volume scheme on a uniform grid. It is **second-order accurate in both space and time**, verified via MMS. The implicit CN scheme is unconditionally stable.

### 3.2 Grid

- **Nodes**: x_i = a + i*h, for i = 0, ..., N-1, with uniform spacing h = (b-a)/(N-1).
- **Faces**: x_{i+1/2} = a + (i + 0.5)*h, for i = 0, ..., N-2. There are N-1 faces, each at the midpoint between adjacent nodes.
- **Interface constraint**: Internal material interfaces must lie exactly on cell faces, not on nodes, and not at arbitrary off-face locations. This ensures the conservative discretization handles the conductivity jump correctly without ambiguous material assignment.

### 3.3 Material Fields (built during __init__)

Each node is assigned to a layer, yielding per-node arrays:
- **Node materials**: rho_nodes, cp_nodes, k_nodes -- shape (N,), looked up from the layer each node belongs to.
- **Storage capacity**: C_i = rho_i * cp_i * h -- thermal energy that node i can store per unit area. Shape (N,).
- **Face conductance**: G_{i+1/2} -- thermal coupling between adjacent nodes. Shape (N-1,).
  - Interior face (same material): G = k / h
  - Interface face (between layers with contact resistance R_c): G = 1 / (h/(2*k_L) + R_c + h/(2*k_R)). This is the harmonic mean of the two half-cell resistances plus the contact resistance. When R_c = 0 it reduces to the harmonic mean (perfect thermal contact).

### 3.4 Crank-Nicolson Coefficients

For each interior node i = 1, ..., N-2:
- r_minus[i] = dt * G_{i-1/2} / (2 * C_i) -- coupling to left neighbor
- r_plus[i]  = dt * G_{i+1/2} / (2 * C_i) -- coupling to right neighbor

The factor of 2 in the denominator comes from the CN time-centering (average of explicit and implicit sides).

### 3.5 Banded Matrix A (assembled once, reused every step)

The CN system A * T^{n+1} = rhs is tridiagonal, stored in banded format (4, N) for scipy.linalg.solve_banded with (l=1, u=2):

| Row | Stencil | Purpose |
|-----|---------|---------|
| i = 0 | [3, -4, 1] | Left Neumann BC via 2nd-order one-sided finite difference: (3T_0 - 4T_1 + T_2)/(2h) = -q/k. This 3-point stencil preserves 2nd-order spatial accuracy at the boundary (a 2-point stencil would be only 1st-order). |
| i = 1..N-2 | [-r_minus, 1 + r_minus + r_plus, -r_plus] | Implicit side of CN for interior nodes. The (1 + r_minus + r_plus) diagonal ensures diagonal dominance. |
| i = N-1 | [0, 1] | Right Dirichlet BC: T_{N-1} = T_right(t) directly. |

**Why A is constant**: The matrix depends only on material properties, grid spacing, and time step, all of which are fixed throughout the simulation. Only the RHS changes each step.

### 3.6 Time Step Selection

dt = min(dt_a, dt_b), where:
- **Stability rule**: dt_a = lam_target * h^2 / alpha_max, where alpha = k/(rho*cp) is the thermal diffusivity and lam_target is the target Fourier number. This controls the spatial Fourier number Fo = alpha*dt/h^2.
- **Nyquist rule**: dt_b = 1 / (20*f), ensuring at least 20 time steps per period of the forcing frequency. This prevents temporal aliasing of the boundary flux.

For the dataset: dt = 0.005 (user-overridden), N = 100 (h = 1/99), t_final = 1.0, giving Nt = 201 time steps. The code uses `np.arange(0, t_final + 1e-12, dt)` with the epsilon because np.arange excludes the stop value.

### 3.7 Time-Stepping (cn_step_banded)

Each step solves A * T^{n+1} = rhs. The RHS is assembled as:

| Component | Nodes | Formula |
|-----------|-------|---------|
| Neumann BC | i = 0 | rhs_0 = 2*h*q(t_{n+1}) / k_left |
| Explicit CN | i = 1..N-2 | rhs_i = r_minus * T^n_{i-1} + (1 - r_minus - r_plus) * T^n_i + r_plus * T^n_{i+1} |
| Source (MMS only) | i = 1..N-2 | rhs_i += dt*h * 0.5*(s^n_i + s^{n+1}_i) / C_i (time-centered for 2nd-order accuracy; using only s^n or s^{n+1} would drop to 1st-order) |
| Dirichlet BC | i = N-1 | rhs_{N-1} = T_right(t_{n+1}) |

The solve is O(N) per step via LAPACK's banded solver.

### 3.8 Boundary Flux Function

`windowed_sin_flux(f, A, t_on, t_off, phase, tukey_alpha)` returns a callable q(t):
- Returns 0 outside [t_on, t_off]
- Inside the window: q(t) = W(tau) * A * sin(2*pi*f*t + phase), where tau = (t - t_on)/(t_off - t_on) and W is a Tukey (tapered cosine) envelope. tukey_alpha controls the taper: 0 = rectangular, 0.5 = half-cosine taper (default), 1 = Hann window.

---

## 4. Method of Manufactured Solutions (MMS)

**File**: `src/physics/mms_1d.py`

### 4.1 MMS Procedure

1. **Choose** an exact analytical solution T*(x,t).
2. **Derive** the source term s*(x,t), boundary flux q*(t), and initial condition T*(x,0) that T* would require by substituting it into the PDE.
3. **Feed** these derived forcings into the unmodified solver. The solver does not know T* -- it just sees BCs and a source term.
4. **Compare** the numerical output to T* at the final time. The error should shrink as O(h^2) and O(dt^2).

### 4.2 Test Case 1: Single-Layer (run_mms_once)

**Manufactured solution**: T*(x,t) = 300 + A * sin(omega*t) * (b - x)^4

- The (b-x)^4 form was chosen because it vanishes at x = b (satisfying the Dirichlet BC T = 300 automatically) and its 4th power provides nontrivial spatial structure for the source term.
- The amplitude A is calibrated as A = flux_A / (4*k*L^3), derived from the boundary flux: q* = -k * dT*/dx|_{x=0} = 4*k*A*L^3*sin(omega*t), so |q*|_max = 4*k*A*L^3 = flux_A.

**Derived quantities** (from T* via calculus):
- Left flux: q*(t) = 4*k*A*L^3 * sin(omega*t)
- Source: s*(x,t) = rho*cp*A*omega*cos(omega*t)*(b-x)^4 - 12*k*A*sin(omega*t)*(b-x)^2
- Initial condition: T*(x, 0) = 300 (when phase = 0)

### 4.3 Test Case 2: Two-Layer with Interface Resistance (run_mms_interface)

**Parameters**: k1=2, k2=1, rho1=1.5, rho2=0.8, cp=1, R_c=0.1, interface at x_I=0.5.

**Piecewise manufactured solution**:
- Left: T_L*(x,t) = 300 + A_L * sin(omega*t) * [(x_I - x)^4 + D_L*(x_I - x)] + C_L * sin(omega*t)
- Right: T_R*(x,t) = 300 + A_R * sin(omega*t) * (b - x)^4

**Two interface constraints** (these determine A_R and C_L):
1. **Flux continuity**: -k1 * dT_L/dx|_{x_I} = -k2 * dT_R/dx|_{x_I}, giving A_R = 2*k1*A_L*D_L / k2 = 40
2. **Temperature jump**: T_L(x_I) - T_R(x_I) = R_c * q_interface, giving C_L = k1*A_L*D_L*(1/(8*k2) + R_c) = 4.5

The left solution is more complex than the right because it needs the extra linear term D_L*(x_I - x) to provide a free parameter for matching the interface flux, and the constant C_L to absorb the temperature jump imposed by R_c.

With A_L=10, D_L=1: interface flux q_I(t) = 20*sin(omega*t), temperature jump = 2*sin(omega*t) = 0.1 * 20*sin(omega*t).

**Derived source terms** (piecewise, one per layer):
- s_L(x,t) = rho1*cp1*dT_L*/dt - k1*d^2T_L*/dx^2
- s_R(x,t) = rho2*cp2*dT_R*/dt - k2*d^2T_R*/dx^2

### 4.4 Convergence Order Tests

**Spatial order** (`space_order_test`, `space_order_test_interface`):
- Fix dt very small (0.0005) so temporal error is negligible.
- Vary N across multiple levels (e.g., [21, 41, 81, 161] or [50, 100, 200]).
- Compute pairwise order: p = log(e_i / e_{i+1}) / log(h_i / h_{i+1}).
- Report the mean of all pairwise orders. Expected: p ~= 2.

**Temporal order** (`time_order_test`, `time_order_test_interface`):
- Fix N very large (801 or 800) so spatial error is negligible.
- Vary dt across multiple levels (e.g., [0.02, 0.01, 0.005]).
- Same pairwise order formula with dt replacing h. Expected: p ~= 2.

**Key principle**: One discretization parameter must be fixed very fine while the other is varied. Otherwise, the non-varying error contaminates the convergence rate, and you observe a plateau or degraded order.

If the spatial test returned p ~= 1.5 instead of 2.0, it would indicate a bug in the spatial discretization (e.g., a boundary stencil dropping to first order, or an interface discretization error).

---

## 5. Data Generation

**File**: `data/generate_dataset.py`

### 5.1 Parameter Sampling

4000 simulations. Three parameters are sampled via 3D Latin Hypercube Sampling (LHS) for space-filling coverage:

| Parameter | Symbol | Range | Sampling |
|-----------|--------|-------|----------|
| Flux amplitude | A | (50, 300) | Uniform |
| Flux frequency | f | (1, 20) | **Log-uniform**: x = lo * (hi/lo)^u, u in [0,1] |
| Contact resistance | R_c | (0.05, 1.0) | Uniform |

Frequency is sampled log-uniformly because the thermal penetration depth scales as delta ~ 1/sqrt(f). Log-uniform gives more uniform coverage of the *physics* space (deep vs. shallow penetration). Amplitude and R_c have linear effects, so uniform sampling suffices.

Material properties are **fixed** across all simulations (k1=2, k2=1, rho=cp=1). Only A, f, and R_c vary.

### 5.2 Initial Conditions

Each simulation gets a unique random IC:

    T0(x) = c0*cos(pi*x/(2L)) + c1*sin(pi*x/L) + c2*cos(2*pi*x/L)

where c0 ~ U(0.5, 1.5), c1 ~ U(-0.3, 0.3), c2 ~ U(-0.3, 0.3). This produces smooth, physically plausible initial temperature profiles that vary across simulations. The Dirichlet BC temperature (300 K) is added by the solver at runtime.

### 5.3 Simulation Execution

For each of the 4000 parameter combinations, a fresh `FVSolver1D` is constructed with the fixed layer geometry and sampled (A, f, R_c), then the full trajectory is solved and stored.

**Fixed simulation parameters**: a=0, b=1, N=100, dt=0.005, t_final=1.0, t_on=0, t_off=0.2, tukey_alpha=0.5.

### 5.4 Output Files

| File | Shape | Description |
|------|-------|-------------|
| `trajectories.npy` | (4000, 201, 100) | All simulation trajectories: (num_sims, Nt, Nx) |
| `x_grid.npy` | (100,) | Spatial node coordinates |
| `t_grid.npy` | (201,) | Time step coordinates |
| `sim_params.npy` | (4000,) object array | Each entry: (amp, freq, T0_array, R_c) |

---

## 6. Dataset and Normalization

**File**: `data/dataset.py`

### 6.1 SnapshotPairDataset

The `SnapshotPairDataset` converts raw trajectories into (input, target) pairs for the FNO. Instead of a sliding window, it uses an **all-to-all snapshot pairing** strategy:

1. For each simulation, uniformly subsample `n_snapshots` time indices from the full 201 time steps.
2. Enumerate all valid pairs (s, j) where j > s (forward-in-time only).
3. Sort all pairs globally by lead time t_j - t_s to support curriculum learning.

With n_snapshots = 20 (config default for training), each sim contributes C(20, 2) = 20*19/2 = **190 pairs**. With 2800 training sims (70% of 4000), that is 2800 * 190 = 532,000 training pairs. However, the effective data diversity is only **2800 independent simulations**, not 532K pairs, because pairs from the same simulation share the same trajectory.

### 6.2 Per-Sample Output (4-tuple)

| Tensor | Shape | Description |
|--------|-------|-------------|
| x_spatial | (Nx, 2) | Channel 0: T_source globally normalized; Channel 1: x_norm in [0, 1] |
| cond | (5,) | Conditioning vector: [t_bar_norm, t_s_norm, A_norm, f_norm, R_c_norm] |
| Y | (Nx, 1) | Target temperature, globally normalized |
| T_stats | (2,) | [mu_global, sigma_global] for denormalization at eval time |

### 6.3 Normalization

**Global normalization**: T_tilde = (T - mu_global) / (sigma_global + epsilon), where mu_global and sigma_global are computed once from the training set only (avoids data leakage). epsilon = 1e-6.

This is deliberately **not** per-sample z-score. Per-sample normalization would destroy the absolute temperature scale, making it impossible for the model to distinguish between physically different temperature fields that happen to have similar shapes. Global normalization preserves scale so the model can learn BC- and IC-dependent dynamics.

**Conditioning normalization**: Each conditioning parameter is min-max normalized to [0, 1] using the known parameter ranges:
- t_bar_norm = (t_j - t_s) / t_final -- lead time as fraction of total sim time
- t_s_norm = t_s / t_final -- absolute source time as fraction of total sim time
- A_norm = (A - 50) / (300 - 50)
- f_norm = (f - 1) / (20 - 1)
- R_c_norm = (R_c - 0.05) / (1.0 - 0.05)

### 6.4 Why t_s (Absolute Source Time) Is in the Conditioning Vector

The boundary flux is only active during t in [0, 0.2]. Without t_s, the model cannot distinguish between:
- A pair where t_s = 0.05 (source is during active flux, target may span active-to-diffusion transition)
- A pair where t_s = 0.5 (source is during pure diffusion, no flux influence)

Both might have the same lead time t_bar, but the underlying physics is completely different. Without t_s, the model was forced to memorize individual training trajectories to resolve this ambiguity, leading to severe overfitting (0.19% train error, 17% val error). Adding t_s fixed this by giving the model the information it needed to generalize.

### 6.5 Curriculum Learning

Pairs are sorted by lead time. During training, a curriculum warmup (default: 35 epochs) progressively exposes longer lead times:
- At epoch e, the fraction of pairs exposed is frac = min(1.0, (e+1) / warmup_epochs).
- `set_curriculum_fraction(frac)` uses binary search (`np.searchsorted`) on the sorted lead times to find the cutoff index.
- This means the model first learns short-horizon (easy) predictions before tackling long-horizon (hard) ones.

### 6.6 Input Noise Augmentation

When noise_std > 0 (configurable, default 0.0), Gaussian noise is added to the normalized source temperature: T_source_norm += N(0, noise_std). This applies only to the training set -- val and test sets never get noise. It acts as a regularizer, simulating measurement noise that the model should be robust to.

### 6.7 Data Splits

Simulation IDs are split 70/15/15 (train/val/test) using a **fixed seed = 0**. This seed is hardcoded so that different training seeds (for model initialization) always see the same train/val/test partition, isolating model variance from data variance.

Splitting by simulation ID (not by pair) is critical: if pairs from the same simulation appeared in both train and test, the model could exploit trajectory similarity, inflating test metrics without true generalization.

---

## 7. Neural Operator: FNO1d

**File**: `src/operators/fno1d.py`

### 7.1 Operator Learning Task

    G_theta(T_tilde(x, t_s), x, t_bar, t_s, A, f, R_c) -> T_tilde(x, t_j)

Given a globally-normalized source temperature field and a 5D conditioning vector, predict the globally-normalized target temperature field.

### 7.2 Architecture (~365K parameters by default)

**Forward pass** (input shapes for a batch of size B):

| Stage | Operation | Shape |
|-------|-----------|-------|
| **Lift** | Linear(2, 64) on spatial input | (B, Nx, 2) -> (B, Nx, 64) |
| **Permute** | Channels-first for convolutions | (B, Nx, 64) -> (B, 64, Nx) |
| **Pad** | Zero-pad spatial dim by 8 for non-periodic signals | (B, 64, Nx) -> (B, 64, Nx+8) |
| **4 Fourier Blocks** | SpectralConv1d + Conv1d bypass + CIN + GELU + Dropout | (B, 64, Nx+8) -> (B, 64, Nx+8) |
| **Unpad** | Remove padding | (B, 64, Nx+8) -> (B, 64, Nx) |
| **Permute** | Channels-last | (B, 64, Nx) -> (B, Nx, 64) |
| **Project** | Linear(64, 128) + GELU + Dropout + Linear(128, 1) | (B, Nx, 64) -> (B, Nx, 1) |

### 7.3 SpectralConv1d

Each spectral convolution layer:
1. **FFT**: `torch.fft.rfft(x, dim=-1)` along the spatial axis -> (B, C, Nx//2+1) complex coefficients.
2. **Truncate**: Keep only the first K = `modes` (default 16) low-frequency coefficients.
3. **Channel mixing**: Learned complex weight tensor of shape (C_in, C_out, K). Applied via einsum `"bik,iok->bok"` -- for each of the K retained modes, multiply the C_in input channels by a (C_in x C_out) complex matrix to produce C_out output channels.
4. **Zero high frequencies**: Modes above K are zeroed (implicit low-pass filter). This is what makes the FNO resolution-invariant.
5. **IFFT**: `torch.fft.irfft(out, n=Nx, dim=-1)` back to physical space.

**Spectral dropout**: When spectral_dropout > 0, random Fourier modes are zeroed during training (independently sampled per forward pass), preventing the model from relying on specific frequencies.

### 7.4 Conv1d Bypass

Each Fourier block also has a `nn.Conv1d(width, width, kernel_size=1)` that operates in physical space. Its output is summed with the spectral convolution output before normalization. This bypass captures local/high-frequency features that the mode-truncated spectral path discards, and ensures the model can represent the identity mapping even if the spectral path contributes nothing.

### 7.5 Padding

The FFT assumes periodic signals. Temperature fields in this problem are non-periodic (Neumann left, Dirichlet right). Zero-padding by 8 on the right extends the signal, reducing spectral leakage artifacts at the boundaries.

### 7.6 Conditional Instance Normalization (CIN)

Standard InstanceNorm1d normalizes each channel independently across the spatial dimension (zero mean, unit variance per sample per channel). CIN adds learned per-sample affine parameters:

    CIN(x) = gamma * InstanceNorm(x) + beta

where gamma and beta are (B, C) tensors produced by the conditioning MLP (not learned layer parameters). This is how the 5D conditioning vector (lead time, source time, A, f, R_c) modulates the FNO's internal representations.

### 7.7 ConditioningMLP

Maps the 5D conditioning vector to per-layer CIN parameters:

    cond (B, 5) -> Linear(5, 256) -> SiLU -> Linear(256, 256) -> SiLU -> head Linear(256, n_layers*2*width)

The output is reshaped to (B, n_layers, 2, width), where index 0 along dim 2 is gamma and index 1 is beta.

**Identity initialization**: The head layer has weight = 0 and bias arranged so gamma = 1, beta = 0 for all layers at initialization. This means the model starts as a plain (unconditioned) FNO with standard InstanceNorm. The conditioning influence grows gradually during training, preventing the randomly-initialized conditioning pathway from destabilizing early learning.

Note: The conditioning MLP uses **SiLU** activation internally, while the main FNO trunk uses **GELU**. They are different.

---

## 8. Loss Functions

**File**: `src/operators/losses.py`

### 8.1 SpatiallyWeightedMSE (training loss)

    L = mean(w(x) * (y_pred - y_true)^2)

where w(x) is a spatial weight profile:
- Baseline: w = 1.0 everywhere.
- Interface boost: w = interface_weight (default 10.0) for nodes within [interface_x - half_width, interface_x + half_width] = [0.45, 0.55].
- Weights are normalized so mean(w) = 1.0. This normalization is important because it keeps the loss magnitude comparable to plain MSE regardless of the boost factor, preventing the learning rate from needing adjustment when the weight is changed.

When interface_weight = 1.0, this reduces to standard MSE.

The weight tensor has shape (1, Nx, 1) and is stored as a buffer (moves with model device, not a parameter).

### 8.2 Interface-Specific Relative L2 (monitoring metric)

    iface_rel_l2 = sqrt(mean((pred - true)^2 over interface nodes) / mean(true^2 over interface nodes)) * 100%

Computed using a boolean mask for nodes with |x - 0.5| <= 0.05 (default). This metric is logged during training but does NOT contribute to the loss gradient. It serves as a diagnostic for how well the model captures the temperature discontinuity at the interface.

### 8.3 Training vs. Evaluation Metrics

- **Training loss**: SpatiallyWeightedMSE in normalized (z-score) space.
- **Training/validation metrics**: Relative L2 in normalized space (logged for monitoring and early stopping).
- **Test metrics** (eval.py): Relative L2 in **physical (denormalized) temperature space**. Denormalization uses T_stats: T_phys = T_norm * (sigma_global + eps) + mu_global. This gives the scientifically meaningful error in Kelvin.

---

## 9. Training Pipeline

**File**: `src/operators/train.py`

### 9.1 run_one_seed

The core training function for a single seed:

1. **Data loading**: Load trajectories, grids, sim_params. Split by sim ID (seed=0, hardcoded). Compute global stats from training set only.
2. **Model construction**: FNO1d with parameters from config.
3. **Resume logic**: If `fno1d_latest.pt` (sentinel) exists, resume from that checkpoint. Validates optimizer/scheduler compatibility.
4. **Training loop** (up to 1500 epochs by default):
   - Update curriculum fraction if warmup_epochs > 0.
   - Train one epoch (forward, loss, backward, gradient clipping, optimizer step).
   - Step the LR scheduler.
   - Every `validate_every` epochs (default 10): run validation, check for improvement, update early stopping counter.
   - Save `fno1d_latest.pt` every epoch (for crash recovery).
   - Save `fno1d_best.pt` when validation improves.
5. **Completion**: Delete `fno1d_latest.pt` sentinel. A run is "complete" when best exists and latest does not.

### 9.2 Optimizer

AdamW with lr=0.002, weight_decay=1e-5. Gradient clipping at max_norm=1.0.

### 9.3 Learning Rate Scheduler: RIGNOThreePhase

Three consecutive phases over 1500 epochs:

| Phase | Epochs | Fraction | LR Behavior |
|-------|--------|----------|-------------|
| **Warmup** | 30 (2%) | 0.02 | Linear ramp from 0.05*peak to peak (0.0001 -> 0.002) |
| **Cosine decay** | 1320 (88%) | 0.88 | Cosine anneal from peak down to 0.05*peak (0.002 -> 0.0001) |
| **Exponential decay** | 150 (10%) | 0.10 | Exponential decay from cosine floor to 0.005*peak (0.0001 -> 0.00001) |

LR values are specified via ratios relative to peak_lr (= training.learning_rate):
- init_lr_ratio = 0.05 -> init_lr = 0.0001
- cosine_floor_lr_ratio = 0.05 -> cosine_floor_lr = 0.0001
- final_lr_ratio = 0.005 -> final_lr = 0.00001

### 9.4 Early Stopping

- patience = 20 (counted in validation evaluations, not epochs).
- validate_every = 10 epochs.
- So early stopping triggers after 20 * 10 = **200 epochs** of no improvement.

### 9.5 Checkpoints

Both `fno1d_best.pt` and `fno1d_latest.pt` contain: epoch, config, seed, model_state, optimizer_state, scheduler_state, best_val, bad_epochs, mu_global, sigma_global. The global normalization stats are saved so that evaluation can denormalize without recomputing from data.

### 9.6 CSV Logging

`train_metrics.csv` logs per-epoch: epoch, train_loss, train_rel_l2, train_iface_rel_l2, val_rel_l2, val_iface_rel_l2, lr, is_best. Validation columns are empty on non-validation epochs.

### 9.7 Multi-Seed Training (run_config_seeds)

Trains the same config with multiple seeds (default: [42]). Reports mean best validation loss across seeds, used as the Optuna objective.

---

## 10. Evaluation

**File**: `src/operators/eval.py`

1. Scans `runs/<experiment>/<config>/seed*/fno1d_best.pt` for all seed checkpoints.
2. For each seed: loads checkpoint, reconstructs FNO1d, builds test DataLoader (n_snapshots_test=40 for denser temporal coverage), evaluates in **physical space** (denormalized).
3. Reports per-seed: best_val (from training), test_rel_l2, test_iface_rel_l2.
4. Computes mean and sample std across seeds.
5. Saves `seed_report.json` with per-seed results and summary statistics.

Note: test evaluation uses n_snapshots_test=40 (vs. 20 for training) to get denser temporal coverage and more pairs for thorough assessment.

---

## 11. Hyperparameter Optimization

**Files**: `scripts/run_train.py` (Optuna sweep), `scripts/run_train_fixed.py` (fixed params with CLI overrides)

The project uses Hydra for configuration and Optuna's TPE sampler for hyperparameter search. The objective is the mean best validation loss across seeds (minimized).

**Available search spaces** (in `conf/search_space/`):
- `medium.yaml` -- full 11-parameter sweep
- `width_only.yaml`, `snapshots_only.yaml` -- single-parameter ablations
- `width_regularization.yaml` -- combined width + regularization
- `regularization.yaml` -- regularization-only (dropout, spectral_dropout, weight_decay, noise_std)
- `dropout_only.yaml`, `spectral_dropout_only.yaml`, `noise_std_only.yaml`, `weight_decay_only.yaml` -- isolated ablations

---

## 12. Visualization

**File**: `visual/plots.py`

25 plot functions organized into 5 groups with a CLI dispatch system:

| Group | Plots | Requirements |
|-------|-------|-------------|
| **physics** | final_temperature, layer_geometry, face_conductance, multilayer_evolution, heat_flux_profile | Solver only (no data/model needed) |
| **mms** | mms_convergence, mms_order_estimation | Runs MMS internally |
| **training** | training_curves, seed_comparison | CSV logs / seed_report.json |
| **data** | trajectory_heatmap, initial_conditions, lhs_scatter, flux_profiles, snapshot_pair_samples, prediction_vs_truth, interface_error, lead_time_coverage, lead_time_error, parameter_error_slices | Data files + checkpoint for model plots |
| **sweep** | sweep_ranking, sweep_convergence, sweep_hyperparams | Experiment directory with sweep results |

**Key model-dependent plots** (require --checkpoint, --data, --x-grid, --t-grid, --params):
- `prediction_vs_truth`: 4-panel comparison (truth heatmap, prediction heatmap, error heatmap, T(x) cross-section)
- `interface_error`: per-sim diagnostics (T(x) overlay at interface, residual structure, jump magnitude vs time)
- `lead_time_error`: binned mean error vs lead time (reveals temporal generalization)
- `parameter_error_slices`: error vs A, f, R_c (reveals conditioning-space weaknesses)

---

## 13. File Map

| File | Role |
|------|------|
| `src/physics/fv_solver_1d.py` | Conservative multilayer Crank-Nicolson FV solver (Layer1D, FVSolver1D, windowed_sin_flux) |
| `src/physics/mms_1d.py` | MMS verification: single-layer + two-layer with interface resistance, spatial and temporal order tests |
| `src/physics/dt_convergence_study.py` | Temporal resolution analysis |
| `data/generate_dataset.py` | 3D LHS sampling, per-sim solver construction, trajectory generation (4000 sims) |
| `data/dataset.py` | SnapshotPairDataset (all-to-all pairing, global normalization, curriculum support), data loading, splitting |
| `src/operators/fno1d.py` | FNO1d (SpectralConv1d, ConditionalInstanceNorm1d, ConditioningMLP) |
| `src/operators/losses.py` | SpatiallyWeightedMSE, interface mask builder, interface-specific rel L2 |
| `src/operators/train.py` | Training loop, RIGNOThreePhase scheduler, early stopping, multi-seed wrapper, resume logic |
| `src/operators/eval.py` | Test evaluation in physical space, seed report generation |
| `src/operators/utils.py` | Device resolution utility |
| `scripts/run_train.py` | Hydra + Optuna sweep entrypoint |
| `scripts/run_train_fixed.py` | Fixed-config training with CLI overrides |
| `conf/config.yaml` | Main configuration (model, training, data, scheduler, loss) |
| `conf/paths/default.yaml` | Path resolution via PROJECT_ROOT env var |
| `conf/search_space/*.yaml` | Optuna search space definitions for various ablation studies |
| `visual/plots.py` | 25-function visualization suite with CLI group dispatch |

---

## 14. Key Design Decisions Summary

| Decision | Rationale |
|----------|-----------|
| Interfaces on cell faces only | Conservative FV requires unambiguous material assignment per face; nodes on interfaces would create dual-material nodes |
| Harmonic mean + R_c for face conductance | Physically correct series resistance model: half-cell in layer L + contact resistance + half-cell in layer R |
| 3-point one-sided stencil for Neumann BC | Maintains 2nd-order spatial accuracy at the boundary; a 2-point stencil would drop to 1st-order |
| CN source term time-centered: 0.5*(s^n + s^{n+1}) | Preserves 2nd-order temporal accuracy; using only one time level would reduce to 1st-order |
| Banded matrix assembled once | A depends only on geometry and material properties (constant); only RHS changes per step |
| Global (not per-sample) temperature normalization | Preserves absolute temperature scale needed for physics; per-sample z-score would erase differences between hot and cold fields |
| t_s in conditioning vector | Distinguishes flux-active (t < 0.2) from diffusion-only (t > 0.2) regimes; without it, model cannot generalize and memorizes training trajectories |
| Data split by simulation ID, not by pair | Prevents data leakage: pairs from the same trajectory are correlated; mixing them across splits inflates metrics |
| Curriculum warmup on lead time | Starts with easy short-horizon predictions; gradually exposes harder long-horizon pairs as the model stabilizes |
| Identity init for CIN (gamma=1, beta=0) | Model starts as plain unconditioned FNO; conditioning influence grows organically during training |
| Spatial loss weighting at interface | Focuses gradient signal on the interface region where the physically important temperature jump occurs |
| Weights normalized to mean=1 | Keeps loss magnitude comparable to plain MSE; no LR adjustment needed when changing weight factor |
| Log-uniform sampling for frequency | Penetration depth ~ 1/sqrt(f); log-uniform gives uniform physics-space coverage |
| Fixed data-split seed (0) vs variable model seed | Isolates model initialization variance from data partition variance |
| Model parameters not in conditioning (k1, k2, rho, cp, t_on, t_off, ...) | These are fixed across all 4000 sims; no variation means no need to condition on them. To generalize to variable materials, they would need to be added to the conditioning vector and sampled during data generation. |
