#!/bin/bash
#SBATCH --job-name=fno_train_fixed
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --tmp=50G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lind1775@umn.edu

set -euo pipefail

# ──── Paths ────
PROJECT_DIR="${PROJECT_DIR:-$HOME/repos/itr-neuraloperator}"
export DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
export PROJECT_ROOT="$PROJECT_DIR"

# Active benchmark adapter (forcing|interfaces|source). load_config reads
# $BENCHMARK to pick conf/benchmark/<name>.yaml, which sets BOTH the model dims
# and the dataset adapter. This must match the benchmark used to generate the
# data in $DATA_DIR. A dotted `benchmark.name=` override is NOT sufficient
# (it would not update the model dims), so always select via this env var.
export BENCHMARK="${BENCHMARK:-forcing}"

# Use node-local scratch for run outputs, then copy results back to $HOME/fno_runs.
LOCAL_RUNS="/tmp/fno_runs_${SLURM_JOB_ID}"
export RUNS_ROOT="$LOCAL_RUNS"
PERSISTENT_RUNS="$HOME/fno_runs"
mkdir -p "$LOCAL_RUNS" "$PERSISTENT_RUNS"

# ──── Environment ────
module load miniforge
eval "$(conda shell.bash hook)"
conda activate itr

# ──── Logging ────
echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "DATA_DIR=$DATA_DIR"
echo "RUNS_ROOT=$RUNS_ROOT"
echo "BENCHMARK=$BENCHMARK"
nvidia-smi

# ──── Run ────
cd "$PROJECT_DIR"
python scripts/run_train_fixed.py "$@"

# ──── Copy results back to persistent storage ────
echo "Copying results from $LOCAL_RUNS to $PERSISTENT_RUNS ..."
cp -r "$LOCAL_RUNS"/* "$PERSISTENT_RUNS"/ 2>/dev/null || true

echo "Job $SLURM_JOB_ID finished at $(date)"
