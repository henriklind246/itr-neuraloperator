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

# ──── Paths ────
PROJECT_DIR="$HOME/repos/itr-neuraloperator"
export PROJECT_ROOT="$PROJECT_DIR"

# ──── Environment ────
module load miniforge
eval "$(conda shell.bash hook)"
conda activate itr

# Flush Python stdout line-by-line so `tail -f` shows progress in real time
export PYTHONUNBUFFERED=1

# ──── Logging ────
echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"
echo "PROJECT_ROOT=$PROJECT_ROOT"

# ──── Run ────
# Run from data/ so the relative np.save() calls land inside $PROJECT_DIR/data/
cd "$PROJECT_DIR/data"
python "$PROJECT_DIR/data/generate_dataset.py"

echo "Job $SLURM_JOB_ID finished at $(date)"
ls -lh "$PROJECT_DIR/data"/*.npy
