#!/bin/bash
#SBATCH --job-name=fno_train
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --tmp=50G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=$USER@umn.edu

# ──── Paths ────
PROJECT_DIR="$HOME/no-tps-ihcp"
export DATA_DIR="$PROJECT_DIR/data"
export RUNS_ROOT="$HOME/fno_runs"
export PROJECT_ROOT="$PROJECT_DIR"

# ──── Environment ────
module load miniforge
conda activate itr

# ──── Optional: copy data to fast local scratch ────
# Uncomment these 2 lines if I/O is a bottleneck:
# mkdir -p /tmp/data && cp "$DATA_DIR"/*.npy /tmp/data/
# export DATA_DIR="/tmp/data"

# ──── Logging ────
echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "DATA_DIR=$DATA_DIR"
echo "RUNS_ROOT=$RUNS_ROOT"
nvidia-smi

# ──── Run ────
cd "$PROJECT_DIR"
python scripts/run_train.py

echo "Job $SLURM_JOB_ID finished at $(date)"
