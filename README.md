# Time-conditioned FNO for Transient Heat Conduction with Imperfect Interfaces

A benchmark suite and time-conditioned 2D Fourier Neural Operator (FNO) for
transient heat conduction across imperfect interfaces, plus a
surrogate-based inverse solver that recovers the ITR from sparse, noisy sensors.

## Overview

Two material slabs on `[0, 1] x [0, 1]` meet at an interface with a scalar or
spatially varying resistance `R_c(y)`. Reference data come from a conservative
Crank-Nicolson finite-volume solver, verified with manufactured solutions. The
surrogate maps a source snapshot to any later target snapshot:

```
G_theta : ( T(x, y, t_s),  t_bar = t_j - t_s,  forcing on [t_s, t_j],  R_c )  ->  T(x, y, t_j)
```

- **Any lead time in one pass.** Training uses all snapshot pairs `(t_s, t_j)`,
  not fixed-step rollouts.
- **Temporal forcing encoder.** The forcing over `[t_s, t_j]` enters as `(128, 3)`
  tokens. Their embedding drives both learned spatial forcing channels and
  conditional instance normalization in the Fourier layers.
- **Benchmark-agnostic pipeline.** Each benchmark is a `ProblemSpec` in
  `problems/` that owns sampling, solver wiring, tensor dims and diagnostics.
  The generator, model, training and evaluation code never branch on the
  benchmark name.

## Benchmarks

| Benchmark         | Heat input                                                   | Interface           | `R_c`                  | Initial condition                |
|-------------------|--------------------------------------------------------------|---------------------|------------------------|----------------------------------|
| `forcing`         | left-wall flux `q_L = a(t) s(y)`, 4 temporal x 4 spatial families | `x = 0.5`      | scalar                 | uniform 300 K                    |
| `forcing_itr_sin` | as `forcing`                                                 | `x = 0.5`           | `R_base + A sin(pi y)` | uniform 300 K                    |
| `interfaces`      | fixed `sin` x `uniform` left-wall flux                       | `x_i in [0.2, 0.8]` | scalar                 | uniform, sinusoid, GRF, hot spot |
| `source`          | internal heating patch `(x_h, y_h, w_h, h_h, A)`, sin^2 pulse | `x = 0.5`          | scalar                 | uniform 300 K                    |
| `source_itr_sin`  | as `source`                                                  | `x = 0.5`           | `R_base + A sin(pi y)` | uniform 300 K                    |

The temporal families are `sin`, `exp`, `pulse_train` and `exp_train`. The
spatial families are `uniform`, `patch`, `gaussian` and `triangle`. The
`forcing` and `interfaces` benchmarks are nondimensional (`k = 2 | 1`,
`rho = cp = 1`). The `source` benchmarks use Ti-6Al-4V and brass in mm-s-K
units. In every benchmark the right wall is held at 300 K and the other walls
are insulated, apart from the left-wall flux. Each spec's `ProblemDims` owns
its tensor dims, and `tests/test_problems.py` pins them.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # add matplotlib for visual/
pytest tests/ -q -m "not slow"    # "slow" marks the MMS convergence studies
```

Requires Python >= 3.11. Run all commands from the repository root.

## Quick start

```bash
# 1. Generate trajectories (100 x 100 grid, written to data/)
python data/generate_dataset.py --benchmark forcing --num-sims 4000

# 2. Train (writes runs/forcing_baseline/config0/seed42/)
BENCHMARK=forcing python scripts/run_train_fixed.py \
  experiment.name=forcing_baseline training.epochs=41

# 3. Evaluate on the held-out test split
python scripts/run_eval.py runs/forcing_baseline/config0 --write-test-records
````

## Usage

**Configuration.** `conf/config.yaml` is the base config. The `BENCHMARK`
environment variable merges in `conf/benchmark/<name>.yaml`, and any key can be
overridden with a dotted `key=value` argument, for example
`model.parameters.width=96 training.seeds=[42,43,44]`. Data and runs go to
`data/` and `runs/` unless you set `DATA_DIR` and `RUNS_ROOT`. Each seed
directory holds `fno2d_best.pt`, `config_used.yaml`, `train_metrics.csv`,
`val_pairs.csv` and `final_metrics.json`. Training `rel_l2` is a percentage in
normalized space. `run_eval.py` also reports errors in Kelvin.

**Studies.**

| Study                         | Entry point                                                                 |
|-------------------------------|-----------------------------------------------------------------------------|
| Out-of-distribution sweeps    | `scripts/run_ood_suite.py conf/ood/<benchmark>.yaml <run_root>`             |
| Unseen spatial family         | `scripts/run_ood_spatial_family.py` (zero-shot), `run_ood_finetune.py` (few-shot) |
| Autoregressive rollout        | `scripts/run_rollout_study.py <run_root> --substeps 2,4,8`                  |
| Zero-shot resolution transfer | `scripts/run_resolution_study.py <run_root> --resolutions 100,150,200,256`  |

**Inverse estimation of `R_c`.** All of the paper's inverse results come from a
single run on a trained `forcing_itr_sin` checkpoint with default settings:

```bash
python scripts/run_inverse_sensor_sweep.py --benchmark forcing_itr_sin \
  --checkpoint runs/<experiment>/config0/seed42/fno2d_best.pt
```

The sweep generates a held-out dataset that is disjoint from training: 16
inversion cases and 32 calibration simulations. It calibrates the surrogate
error on the calibration set. Then it recovers `(R_base, A)` for each case
with 8, 16 and 32 interface sensors, once with sensor noise and once without.
Results go to `runs/inverse_sensor_sweeps/forcing_itr_sin/<checkpoint fingerprint>/`,
and the noise-free run goes to its `no_noise/` subdirectory. Each run writes
`inverse_sensor_sweep.csv`, `inverse_sensor_sweep_summary.csv` and figures F29
and F30 in `figures/`. The sweep exits with code 2 if the surrogate fails the
calibration gate. It calls `scripts/invert.py` for each sensor count, and you
can also run `invert.py` directly to invert a single configuration.

**Paper figures.** Each figure declares the run artifacts it needs, and a
figure whose artifacts are missing does not render. See
[`visual/pub/README.md`](visual/pub/README.md).

```bash
python -m visual.pub --verify                  # which figures have their artifacts
python -m visual.pub --all --out visual/pub_out
```

## Repository layout

```
problems/        ProblemSpec adapters and registry, one file per benchmark
src/physics/     FV solver, MMS verification, boundary forcing, sources, ICs
src/operators/   FNO2d, training, evaluation, losses, rollout, DDP helpers
data/            Trajectory generation and the all-to-all snapshot-pair dataset
conf/            Base config, benchmark/ groups, ood/ suite descriptors
scripts/         Train, eval, inverse, OOD, rollout and resolution entry points
visual/pub/      Publication figures with provenance tracking
slurm/           SLURM jobs for generation, single-GPU/DDP training, evaluation
tests/           Solver, model, dataset and contract tests
```

## Citation

```bibtex
@article{lind_time_conditioned_fno,
  title   = {Time-conditioned Fourier Neural Operator for Transient Heat Conduction
             with Imperfect Interfaces and Heterogeneous Functional Inputs},
  author  = {Lind, Henrik and Davis, Richard and Sakhalkar, Siddhesh},
  journal = {TODO},
  year    = {TODO}
}
```

## License

TODO
