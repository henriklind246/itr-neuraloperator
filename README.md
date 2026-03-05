# itr-neuraloperator

Repository for modeling interfacial thermal resistance (ITR) in composite materials using neural operators.

## Setup

1. Create and activate a virtual environment (optional):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Hydra + Optuna training sweeps

`scripts/run_train.py` is a Hydra entrypoint that performs Optuna sweeps and trains each config across the shared seed list:

`seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`

Each **new invocation** of `run_train.py` gets a new namespace:

- `experiment0`
- `experiment1`
- `experiment2`
- ...

### Run a sweep

```bash
python scripts/run_train.py -m
```

Override trial count (for example 10 or 50):

```bash
python scripts/run_train.py -m tuning.n_trials=10
python scripts/run_train.py -m tuning.n_trials=50
```

### Output layout per invocation

Training artifacts:

- `runs/experimentN/config0/seed0..seed9/`
- `runs/experimentN/config1/seed0..seed9/`
- ...

Generated resolved configs:

- `config/generated/experimentN/config0.yaml`
- `config/generated/experimentN/config1.yaml`
- ...
- `config/generated/experimentN/index.csv`
- `config/generated/experimentN/best_config.yaml`

### Rebuild index/best config

If needed, rebuild `index.csv` and `best_config.yaml`:

```bash
python scripts/collect_best_config.py --experiment experimentN
```

Or use latest generated experiment automatically:

```bash
python scripts/collect_best_config.py
```
