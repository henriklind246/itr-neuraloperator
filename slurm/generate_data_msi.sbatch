#!/bin/bash
#SBATCH --job-name=fno_datagen
#SBATCH --partition=msismall
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lind1775@umn.edu

set -euo pipefail

# ──── Paths ────
PROJECT_DIR="${PROJECT_DIR:-$HOME/repos/itr-neuraloperator}"
export PROJECT_ROOT="$PROJECT_DIR"
DATA_OUT_DIR="${DATA_OUT_DIR:-$PROJECT_DIR/data}"
NUM_SIMS="${NUM_SIMS:-8000}"

# Active benchmark adapter (forcing|interfaces|source). Selects sampling +
# physics + item representation. generate_dataset.py reads $BENCHMARK as the
# default for --benchmark, so exporting it here is sufficient.
export BENCHMARK="${BENCHMARK:-forcing}"

if [[ $# -gt 0 && "$1" != -* ]]; then
    DATA_OUT_DIR="$1"
    shift
fi

if [[ $# -gt 0 && "$1" != -* ]]; then
    NUM_SIMS="$1"
    shift
fi

# ──── Environment ────
module load miniforge
eval "$(conda shell.bash hook)"
conda activate itr

# Flush Python stdout line-by-line so `tail -f` shows progress in real time
export PYTHONUNBUFFERED=1

# ──── Logging ────
echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "DATA_OUT_DIR=$DATA_OUT_DIR"
echo "NUM_SIMS=$NUM_SIMS"
echo "BENCHMARK=$BENCHMARK"
echo "Extra generator args: $*"

# ──── Run ────
cd "$PROJECT_DIR"
mkdir -p "$DATA_OUT_DIR"
python "$PROJECT_DIR/data/generate_dataset.py" \
    --num-sims "$NUM_SIMS" \
    --save-dir "$DATA_OUT_DIR" \
    --benchmark "$BENCHMARK" \
    "$@"

echo "Job $SLURM_JOB_ID finished at $(date)"
ls -lh "$DATA_OUT_DIR"/*.npy
