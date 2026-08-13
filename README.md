# Neural operators for interfacial thermal resistance

A benchmark suite and time-conditioned 2D Fourier Neural Operator for transient
heat conduction across an imperfect material interface.

The geometry is two stacked slabs on `[0, 1] x [0, 1]` separated by a thin
contact resistance `R_c`. Reference trajectories come from a conservative
Crank-Nicolson finite-volume solver, verified against manufactured solutions.
The surrogate maps a source snapshot to any later target snapshot,

```
G_theta : ( T(x, y, t_s), lead time, forcing, R_c ) -> T(x, y, t_j)
```

and is trained on all-to-all snapshot pairs `(t_s, t_j)` rather than fixed
single-step rollouts, so one model covers every lead time in `(0, t_final]`.

Seven benchmarks are crossed with two input representations. The dataset,
model, and training loop never branch on the benchmark name: everything routes
through a `ProblemSpec` adapter in `problems/`.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python >= 3.11. Run all commands from the repository root.

## Quick start

```bash
# 1. Generate trajectories (100x100 grid)
python data/generate_dataset.py --benchmark forcing --num-sims 8000

# 2. Train
BENCHMARK=forcing REPRESENTATION=temporal_encoder \
python scripts/run_train_fixed.py experiment.name=forcing_baseline training.epochs=101

# 3. Evaluate on the test split
python scripts/run_eval.py runs/forcing_baseline/config0

# 4. Inspect per-pair validation error
python scripts/inspect_val_pairs.py runs/forcing_baseline/config0/seed42/val_pairs.csv
```

## Benchmarks

| Benchmark         | Varies                                                                      | `R_c`    |
|-------------------|-----------------------------------------------------------------------------|----------|
| `forcing`         | Separable left flux `q_L(y,t) = a(t) s(y)`; 4 temporal x 4 spatial families | scalar   |
| `forcing_itr`     | `forcing` with a Gaussian void profile `R_c(y)`                             | `R_c(y)` |
| `forcing_itr_sin` | `forcing` with `R_c(y) = R_base + A sin(pi y)`                              | `R_c(y)` |
| `source`          | Internal volumetric heating patch `(x_h, y_h, A, w_h, h_h)`                 | scalar   |
| `source_itr`      | `source` with a Gaussian void profile `R_c(y)`                              | `R_c(y)` |
| `source_itr_sin`  | `source` with `R_c(y) = R_base + A sin(pi y)`                               | `R_c(y)` |
| `interfaces`      | Interface location `x_i in [0.2, 0.8]` and initial condition                | scalar   |

Representations (set via `REPRESENTATION` or `benchmark.representation`):

- **`temporal_encoder`** (default) — lean spatial channels plus a `(128, 2)`
  forcing token sequence consumed by a temporal branch.
- **`bins`** — 16 additional spatial channels holding integral forcing bins over
  `[t_s, t_j]`; no temporal branch.

Channel counts and conditioning dims are owned by each spec's `ProblemDims` and
pinned by `tests/test_problems.py`.

## Configuration

`conf/config.yaml` is the base config, composed with `conf/benchmark/*.yaml` and
`conf/representation/*.yaml`. Any key is overridable with dotted `key=value`
arguments:

```bash
python scripts/run_train_fixed.py \
  experiment.name=ablation model.parameters.width=96 training.seeds=[42,43,44]
```

Each run writes `runs/<experiment>/config<id>/seed<seed>/` containing
`fno2d_best.pt`, `config_used.yaml`, `train_metrics.csv`, `val_pairs.csv`, and
`final_metrics.json`.

## Repository layout

```
problems/      ProblemSpec adapters + registry (one file per benchmark)
src/physics/   FV solver, MMS verification, boundary forcing, internal source
src/operators/ FNO2d model, training, evaluation, losses, rollout
data/          Dataset generation and all-to-all snapshot-pair dataset
conf/          Base config plus benchmark/ and representation/ groups
scripts/       Training, evaluation, inverse, and OOD entry points
visual/pub/    Publication figures with provenance tracking
tests/         Solver, model, and contract tests
```

## Inverse problems

`scripts/invert.py` recovers boundary forcing and interface resistance from
sparse noisy sensors, using a trained checkpoint as the forward map. The
benchmark is read from the checkpoint config.

```bash
python scripts/invert.py \
  --checkpoint runs/forcing_baseline/config0/seed42/fno2d_best.pt \
  --sensor-n-y 16 --noise-std 0.1
```

## Reproducing paper figures

```bash
python -m visual.pub --verify                  # check artifact availability
python -m visual.pub --all --out visual/pub_out
```

Figures declare their required run artifacts in `visual/pub/figures.yaml` and
refuse to render when those are missing. Point `visual/pub/manifest.yaml` at
your own run directories.

## Tests

```bash
pytest tests/ -q          # add -m "not slow" to skip MMS convergence studies
```

## Citation

```bibtex
@article{TODO,
  title  = {TODO},
  author = {TODO},
  year   = {TODO}
}
```

## License

TODO
