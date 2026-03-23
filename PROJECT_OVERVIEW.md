# Project Overview: Neural Operator for Heat Conduction

## 1. Problem Statement

This project trains a Fourier Neural Operator (FNO) to learn the forward operator of the 1D unsteady heat equation with spatially-varying material properties. The operator takes as input a window of recent temperature history, the upcoming boundary heat flux, spatial and temporal coordinates, and piecewise-constant material fields (thermal conductivity and volumetric heat capacity), and predicts the temperature field over a fixed future time horizon.

The training data comes from finite-difference simulations of a two-layer rod with a conservative Crank-Nicolson scheme. Material properties, boundary forcing amplitude, and forcing frequency all vary across simulations via Latin Hypercube Sampling, so the FNO must generalize across a family of PDEs, not just one.

## 2. The PDE

The governing equation is the 1D unsteady heat equation with spatially-varying coefficients:

    rho(x) * cp(x) * dT/dt = d/dx [k(x) * dT/dx] + s(x,t)

where T(x,t) is temperature, k(x) is thermal conductivity, rho(x) is density, cp(x) is specific heat capacity, and s(x,t) is an optional volumetric source term (zero in data generation, nonzero only for MMS verification).

**Domain**: x in [0, 1] with two material layers separated by an interface at x = 0.5.

**Boundary conditions**:
- Left (x = 0): Neumann BC with prescribed heat flux q(t) — a windowed sinusoid with LHS-sampled amplitude and frequency
- Right (x = 1): Dirichlet BC, T = 300 K (constant)

**Material structure**: Each layer has constant rho, cp, and k. The interface at x = 0.5 sits on a cell face (between nodes), ensuring the conservative discretization handles the conductivity jump correctly.

## 3. Finite-Difference Solver

**File**: `src/physics/fd_solver_1d.py`

The solver uses the Crank-Nicolson (CN) method, which is implicit, unconditionally stable, and second-order accurate in both space and time. This was verified via the Method of Manufactured Solutions (MMS) in `src/physics/mms_1d.py`.

**Key components**:
- `Layer1D` dataclass: defines one material layer with (x_left, x_right, rho, cp, k)
- `FDSolver1D` class: takes a list of layers, validates geometry, builds per-node material arrays and face conductances, assembles the banded CN system, and time-steps via LAPACK's banded solver
- Interface handling: at a face between two materials, the face conductance uses the harmonic mean of the two conductivities rather than a simple average, which preserves conservation

**Discretization parameters for the training dataset**:
- N = 100 nodes (dx = 0.01)
- dt = 0.005, t_final = 1.0 (Nt = 201 time steps)
- The resulting relative L2 error vs. the MMS exact solution is approximately 0.05%, adequate for training data

The left-boundary flux is a windowed sinusoid: q(t) = W(t) * A * sin(2*pi*f*t), where W(t) is a Tukey window active during [t_on, t_off] = [0, 0.2]. The amplitude A and frequency f are sampled per simulation.

## 4. Data Generation Pipeline

**File**: `data/generate_dataset.py`

Each simulation has its own material properties and forcing parameters, sampled jointly via a 6-dimensional Latin Hypercube Sample (LHS) for space-filling coverage:

| Parameter   | Symbol  | Range         |
|-------------|---------|---------------|
| Flux amplitude | A    | (50, 300)     |
| Flux frequency | f    | (1, 20)       |
| Conductivity layer 1 | k1 | (0.5, 5.0) |
| Conductivity layer 2 | k2 | (0.5, 5.0) |
| Vol. heat capacity layer 1 | rho_cp_1 | (0.5, 5.0) |
| Vol. heat capacity layer 2 | rho_cp_2 | (0.5, 5.0) |

The volumetric heat capacity rho*cp is sampled as a single product. In the solver, this is implemented by setting cp = 1 and rho = rho_cp, so that rho * cp = rho_cp directly.

Initial conditions are random superpositions of cosine and sine modes, different per simulation.

For each of the 2500 simulations, a fresh two-layer `FDSolver1D` is constructed with the sampled material properties, and the full trajectory is solved and stored.

**Output files** (saved to `data/`):
- `trajectories.npy`: shape (2500, 201, 100) — all simulation trajectories
- `x_grid.npy`: shape (100,) — spatial node coordinates
- `t_grid.npy`: shape (201,) — time step coordinates
- `sim_params.npy`: array of 2500 seven-tuples (amp, freq, k1, k2, rcp1, rcp2, T0_array)

## 5. Dataset and Windowing

**File**: `data/dataset.py`

The `WindowedForecastDataset` class converts raw simulation trajectories into input-output pairs for the FNO by sliding a window along the time axis.

**Window geometry**: Given a starting index s, the input uses k = 10 history time steps [s, s+k) and the output (target) covers a horizon of H = 40 future time steps [s+k, s+k+H). The constraint s + k + H <= Nt must hold, giving max_s = 201 - 10 - 40 = 151, so there are 152 valid starting positions per simulation.

**Input tensor construction** — shape (Nx, H, 15) with 15 channels:

| Channels | Count | Description |
|----------|-------|-------------|
| History  | 10    | Temperature at the k = 10 most recent time steps, broadcast over the horizon dimension |
| x-coordinate | 1 | Spatial position normalized to [0, 1] |
| t-coordinate | 1 | Future time steps normalized to [0, 1] over the full simulation duration |
| Boundary flux | 1 | q(t) evaluated at each future time step |
| k-field | 1 | Thermal conductivity as a step function over x, normalized to [0, 1] using the LHS bounds (0.5, 5.0) |
| rho_cp-field | 1 | Volumetric heat capacity as a step function over x, normalized to [0, 1] using the LHS bounds (0.5, 5.0) |

**Target tensor** — shape (Nx, H, 1): the true temperature field over the prediction horizon.

**Data splits**: Simulation IDs (not windows) are split 70/15/15 into train/val/test using a fixed seed = 0. This seed is intentionally hardcoded so that different training seeds only affect model initialization and optimization randomness, isolating model variance from data variance.

**Windowing modes**:
- Training: random windows, 1 per sim per epoch (stochastic)
- Validation: random windows, 8 per sim per epoch (faster than exhaustive but more signal than 1)
- Test: deterministic, all 152 valid windows per sim (exhaustive)

## 6. Neural Operator: FNO1d

**File**: `src/operators/fno1d.py`

The Fourier Neural Operator learns the mapping:

    G: (history, coordinates, forcing, material fields) -> future temperature

operating on 2D spatial-temporal grids. The two "spatial" dimensions of the FNO are the physical space axis (Nx = 100) and the time horizon axis (H = 40).

**Architecture** (input shape: batch x 100 x 40 x 15, output shape: batch x 100 x 40 x 1):

1. **Lift**: Linear(15, 64) projects the 15 input channels to the hidden width
2. **Pad**: The spatial dimension is zero-padded by 8 (100 -> 108) to reduce spectral aliasing
3. **4 Spectral Blocks**, each containing:
   - A `SpectralConv2d` layer that applies a learned linear transform in the Fourier domain, keeping modes1 = 16 modes in x and modes2 = 16 modes in t
   - A pointwise Conv2d(64, 64, kernel_size=1) skip connection
   - Tanh activation on the sum
4. **Un-pad**: Remove the padding (108 -> 100)
5. **Project**: Linear(64, 32) then Linear(32, 1)

The `SpectralConv2d` layer works by applying a 2D real FFT to the input, multiplying the retained low-frequency coefficients by learned complex weight matrices, zeroing the high-frequency modes, and applying the inverse FFT. This gives the FNO its global receptive field — every spatial and temporal point can influence every other point through the spectral domain.

The default configuration has approximately 8.4 million parameters.

## 7. Training Pipeline

**File**: `src/operators/train.py`

**Training loop** (`run_one_seed`):
- Optimizer: Adam with configurable learning rate and weight decay
- LR scheduler: StepLR (halves LR every 50 epochs by default)
- Loss function: MSELoss between predicted and true temperature fields
- Validation metric: relative L2 error (%) = sqrt(mean((pred - true)^2) / mean(true^2)) * 100
- Early stopping: stops after 15 consecutive validation checks (every 10 epochs) with no improvement
- Checkpointing: saves the best model (by validation rel_l2) to `fno1d_best.pt`, including the full config for reproducibility

**Multi-seed wrapper** (`run_config_seeds`):
- Trains the same configuration with multiple seeds (e.g., seed 0 and seed 1)
- Reports the mean best validation loss across seeds, which serves as the objective for hyperparameter optimization

**Metrics logged per epoch** (to `train_metrics.csv`):
- epoch, train_loss (MSE), train_rel_l2 (%), val_rel_l2 (%), learning rate, is_best flag

**Device handling**: A `resolve_device("auto")` utility detects CUDA and all model and batch placements flow through it. DataLoaders use `pin_memory=True` and `num_workers=2` when CUDA is available.

## 8. Evaluation

**File**: `src/operators/eval.py`

Evaluation loads the best checkpoint for each seed, reconstructs the model, and computes test_rel_l2 on the full test set (all valid windows, deterministic). Results are aggregated into a `seed_report.json` containing per-seed metrics and summary statistics (mean and standard deviation of validation and test errors across seeds).

Checkpoints are loaded with `map_location="cpu"` first and then moved to the resolved device, avoiding device mismatch errors when evaluating a GPU-trained model on CPU or vice versa.

## 9. Hyperparameter Optimization

**Files**: `scripts/run_train.py`, `conf/search_space/medium.yaml`, `conf/hydra/sweeper/optuna_local.yaml`

The project uses Hydra for configuration management and Optuna for hyperparameter search. Running `python scripts/run_train.py -m` launches a multirun sweep where Optuna's Tree-structured Parzen Estimator (TPE) proposes hyperparameter bundles and uses the mean validation loss across seeds as the optimization objective.

**Search space** (from `conf/search_space/medium.yaml`):
- learning_rate: log-uniform in [1e-4, 5e-3]
- weight_decay: log-uniform in [1e-7, 1e-3]
- batch_size: choice from {8, 10, 16, 20}
- modes1, modes2: choice from {8, 12, 16, 20, 24}
- width: choice from {32, 48, 64, 96}

Each trial trains all seeds, and the sweep output includes resolved configs for every trial, an index.csv ranking all trials, and a copy of the best configuration.

## 10. File Map

| File | Role |
|------|------|
| `src/physics/fd_solver_1d.py` | Conservative multilayer Crank-Nicolson FD solver (Layer1D, FDSolver1D) |
| `src/physics/mms_1d.py` | Method of Manufactured Solutions verification (spatial + temporal order tests) |
| `src/physics/dt_convergence_study.py` | Temporal resolution analysis script |
| `data/generate_dataset.py` | 6D LHS sampling, per-sim solver construction, trajectory generation |
| `data/dataset.py` | WindowedForecastDataset, data loading, splitting, DataLoader creation |
| `src/operators/fno1d.py` | FNO1d model with SpectralConv1d layers |
| `src/operators/train.py` | Training loop, validation, early stopping, multi-seed wrapper |
| `src/operators/eval.py` | Test evaluation, seed report generation |
| `src/operators/utils.py` | Device resolution utility |
| `scripts/run_train.py` | Hydra entrypoint for Optuna hyperparameter sweeps |
| `conf/config.yaml` | Main configuration (model, training, data paths) |
| `conf/paths/default.yaml` | Path resolution via PROJECT_ROOT env var |
| `conf/search_space/medium.yaml` | Optuna search space definition |
| `conf/hydra/sweeper/optuna_local.yaml` | Optuna sweeper settings (TPE, sequential) |
| `visual/plots.py` | Visualization suite (physics, training, data, and MMS plots) |
| `tests/` | Test suite (141 tests covering solver, model, dataset, training, and evaluation) |
