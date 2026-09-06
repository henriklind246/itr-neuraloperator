# `visual/pub/` -- publication figures

This is the repository's only manuscript/publication pipeline. The smaller
18-entry `PLOT_REGISTRY` in `visual/cli.py` is reserved for recurring experiment
diagnostics and numerical-correctness checks; the superseded `paper`, `inverse`,
`rollout`, and `resinv` groups have been retired.

Three things make this package different from the rest of `visual/`:

1. **A figure cannot render without provenance.** Every quantitative figure declares
   which artifacts it needs; the CLI refuses to draw when they are absent instead of
   skipping silently. `visual/_common.py:_print_skip` is the policy this package
   rejects.
2. **The unit of replication is a simulation, not a snapshot pair.** A single
   simulation contributes `S x J` strongly correlated pairs. Treating those pairs
   as independent inflates every `n` and narrows every interval. All reduction
   happens in `stats.py`, at simulation level; an AST test forbids figure modules
   from calling `np.mean`/`median`/`quantile` at all.
3. **The plotted interface jump is the physical contact jump**
   `dT_contact(y,t) = R_c(y) q_n(y,t)`, not the adjacent-node difference the training
   loss uses. It is implemented once, in `visual/dataset_plots.py`, and imported by
   `visual/pub/jump.py`; a test asserts nothing here restates it.

## Running it

Run from the repo root (`visual` is imported from the root, not installed).

```
python -m visual.pub --list                 # keys, tiers, requirements
python -m visual.pub --verify               # satisfiability table + regeneration commands
python -m visual.pub --figure F01_problem_schematic --out visual/pub_out
python -m visual.pub --all --tier 2
python -m visual.pub --audit visual/pub_out # re-hash sources, report drift
```

Exit codes: `0` clean, `1` a provenance requirement failed under `--strict` (the
default) or a figure is registered but not yet built, `2` at least one figure
rendered degraded under `--allow-missing`.

`--list` and `--verify` never import a figure module, so neither imports torch.

Every rendered figure writes `<key>.png`, `<key>.pdf` and `<key>.provenance.json`
(schema `visual.pub.provenance/1`) carrying the source sha256s, the counts
`{n_simulations, n_pairs, n_seeds}`, the case selection, and the degradation list.
`render()` refuses to overwrite a PNG whose sidecar records a different git commit
unless `--force`.

## The manifest

`figures.yaml` (version controlled) says what each figure *requires*.
The repository's default `visual/pub/manifest.yaml` says which concrete runs or
tables currently *satisfy* those requirements. Pass another manifest when working
with a different artifact collection:

```yaml
sources:
  source_itr_records:
    - run: runs/pub_source_itr/config0/seed42
    - run: runs/pub_source_itr/config0/seed43
    - run: runs/pub_source_itr/config0/seed44
  inverse_forcing:
    - table: results_inverse.csv
```

An entry has either a `run` key (a run directory; the artifact name is inferred from
the requirement's `kind`, or given explicitly as `artifact`) or a `table` key (a path
to a CSV). Paths are relative to the repo root.

## What is currently blocked, and why

As of the final cleanup check, `--verify` reports **5 of 26 figures satisfiable**:
F01, F02, F23, F24, and F25. That is not a defect in this package; it is an
accurate statement about the available artifacts:

| fact | value |
|---|---|
| canonical forward runs | absent from the paths named in `manifest.yaml` |
| `results_inverse.csv` | 10 scalar-forcing cases; not a sensor-count sweep |
| F18 paired sweep | not yet run for both `forcing` and `forcing_itr` |
| OOD per-sim CSVs | present, but only one seed per protocol |
| rollout and resolution studies | present and provenance-audited for F23/F24 |

No forward artifact survives for `forcing`, `source` or `interfaces`. The claim that
these benchmarks sit below 1% validation error is not currently supported by anything
in the repo.

Record schema v1 remains acceptable only as a development fixture. Publication
aggregation requires the schema-v2 pooled sufficient statistics (`sse_K2`,
`num_error_cells`, `target_sse_K2`, and the interface equivalents) so that
`rmse_K = sqrt(sum sse / sum n)` is computed before simulations are summarized.

## Regenerating the artifacts

`--verify` prints these next to each unsatisfied requirement. They are **not** run by
this package.

```bash
# Tier 1: canonical forward runs, per benchmark, >= 3 seeds
#   bench in {forcing, source, source_itr, interfaces}
python scripts/run_train_fixed.py experiment.name=pub_<bench> \
    benchmark=<bench> representation=temporal_encoder

# Tier 1: test records at schema v2, per run per seed
python scripts/write_test_records.py runs/pub_<bench>/config0 --seed <s>

# Tier 3 (F19, F20): OOD suite, once per model seed, >= 3 seeds
python scripts/run_ood_suite.py ...

# Tier 3 (F18): paired 8/16/32-sensor sweeps for both inverse benchmarks.
# The driver holds (sim_id, noise_seed, init_seed) fixed across counts and writes
# one canonical inverse_sensor_sweep.csv per benchmark.
python scripts/run_inverse_sensor_sweep.py --benchmark forcing \
    --checkpoint <matching-forcing-checkpoint>
python scripts/run_inverse_sensor_sweep.py --benchmark forcing_itr \
    --checkpoint <matching-forcing-itr-checkpoint>

# After both complete, add their exact CSV paths under inverse_sensor_sweep in
# visual/pub/manifest.yaml, then render F18 strictly.
python -m visual.pub --figure F18_sensor_count --strict
```

## Module boundaries

Enforced by AST tests in `tests/test_pub_style.py`:

- `manifest.py` imports no matplotlib.
- `records.py` loads and validates; it does not aggregate.
- `stats.py` reads no files and imports no matplotlib.
- `select.py` takes a frame and returns a `CaseSelection`.
- `panels.py` neither loads nor aggregates.
- `fig_*.py` compose only, and perform no reduction. The whitelist is empty; if it
  ever grows, the pseudo-replication defect has come back.
- Nothing here imports the retired `visual.paper_plots`, `visual.inverse_plots`,
  `visual.rollout_plots`, or `visual.resinv_plots` modules.

## Excluded from the figure pool

Any bins / `Q_0..Q_15` channel plot; legacy uncertainty figures; and the retired
`source_itr_*` inverse figures (the second inverse benchmark is `forcing_itr`). The
live forcing-token diagnostic remains in `visual.cli` and derives its dimensions
from the current production constants; F02 reads the live problem dimensions when
it renders.
