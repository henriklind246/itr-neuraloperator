#!/bin/bash
#SBATCH --job-name=fno_eval
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lind1775@umn.edu

set -euo pipefail

# ──── Paths ────
PROJECT_DIR="$HOME/repos/itr-neuraloperator"
export DATA_DIR="$PROJECT_DIR/data"
export PROJECT_ROOT="$PROJECT_DIR"

# ──── Environment ────
module load miniforge
eval "$(conda shell.bash hook)"
conda activate itr

# ──── Logging ────
echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "DATA_DIR=$DATA_DIR"
echo "Run root arg(s): $*"
nvidia-smi

# ──── Run ────
cd "$PROJECT_DIR"
python scripts/run_eval.py "$@"

echo "Job $SLURM_JOB_ID finished at $(date)"
