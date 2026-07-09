#!/bin/bash
#SBATCH --job-name=cvit_pino
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --tmp=50G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lind1775@umn.edu

# Physics-only (PINO) training of a CViT on the diffusion benchmark.
#
# Single-GPU by design: run_config_seeds_pino has no DDP path (v1). The objective
# is purely physical -- interior heat residual + soft zero-Neumann walls + an IC
# anchor -- so NO paired snapshots are used; validation grid-queries the model
# against the saved trajectories. SOAP is the intended optimizer.
#
# Usage (submit from the repo root):
#   sbatch slurm/train_cvit_pino_msi.sbatch \
#       training.epochs=300 training.pino.n_r=4096 training.pino.sim_batch=16
#
# Any trailing key=value pairs are forwarded to run_train_pino.py and OVERRIDE
# the baked-in defaults below (last write wins). Select the benchmark and the
# experiment/data/output roots via the env vars just below, NOT dotted overrides:
#   BENCHMARK        conf/benchmark group (MUST be diffusion for this path)
#   DATA_DIR         diffusion dataset dir (trajectories.npy, *_grid.npy, ...)
#   EXPERIMENT_NAME  run label -> $RUNS_ROOT/$EXPERIMENT_NAME/config<id>/seed<N>
#   PROJECT_DIR      repo checkout (default $HOME/repos/itr-neuraloperator)

set -euo pipefail

# ──── Paths ────
PROJECT_DIR="${PROJECT_DIR:-$HOME/repos/itr-neuraloperator}"
export PROJECT_ROOT="$PROJECT_DIR"
export DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"

# load_config reads $BENCHMARK to pick conf/benchmark/<name>.yaml (model dims +
# dataset adapter). The CViT physics-only path is diffusion-only; a dotted
# benchmark.name= override is NOT sufficient, so always select via this env var.
export BENCHMARK="${BENCHMARK:-diffusion}"
export REPRESENTATION="${REPRESENTATION:-temporal_encoder}"

# Descriptive run label (config resolves experiment.name from $EXPERIMENT_NAME).
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-cvit_pino_diffusion}"

# Node-local scratch for run outputs, copied back to $HOME/cvit_runs on exit.
LOCAL_RUNS="/tmp/cvit_runs_${SLURM_JOB_ID}"
export RUNS_ROOT="$LOCAL_RUNS"
PERSISTENT_RUNS="${PERSISTENT_RUNS:-$HOME/cvit_runs}"
mkdir -p "$LOCAL_RUNS" "$PERSISTENT_RUNS"

# Always copy partial results back, even if training aborts.
copy_back() {
    echo "Copying results from $LOCAL_RUNS to $PERSISTENT_RUNS ..."
    cp -r "$LOCAL_RUNS"/* "$PERSISTENT_RUNS"/ 2>/dev/null || true
}
trap copy_back EXIT

# ──── Environment ────
module load miniforge
eval "$(conda shell.bash hook)"
conda activate itr

# ──── Preflight: the diffusion dataset must exist in $DATA_DIR ────
if [[ ! -f "$DATA_DIR/trajectories.npy" ]]; then
    echo "ERROR: $DATA_DIR/trajectories.npy not found." >&2
    echo "Generate it first, e.g.:" >&2
    echo "  BENCHMARK=diffusion python data/generate_dataset.py --num-sims 8000 --benchmark diffusion" >&2
    exit 1
fi

# ──── Logging ────
echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "DATA_DIR=$DATA_DIR"
echo "RUNS_ROOT=$RUNS_ROOT"
echo "BENCHMARK=$BENCHMARK / REPRESENTATION=$REPRESENTATION"
echo "EXPERIMENT_NAME=$EXPERIMENT_NAME"
nvidia-smi

# ──── Run ────
# Baked-in defaults for the physics-only recipe; forwarded "$@" overrides win.
# lambda_ic > 0 is REQUIRED on diffusion (T == 300 satisfies the PDE + every BC,
# so a residual-only objective is degenerate -- the IC anchor breaks it).
cd "$PROJECT_DIR"
# Stream stdout live: SLURM redirects to a file (not a TTY), so Python
# block-buffers stdout and the pre-training setup phase (imports, data load,
# model build) would otherwise show nothing until the buffer fills.
export PYTHONUNBUFFERED=1
python -u scripts/run_train_pino.py \
    training.optimizer=SOAP \
    training.pino.lambda_ic=1.0 \
    "$@"

echo "Job $SLURM_JOB_ID finished at $(date)"
