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
#SBATCH --mail-user=lind1775@umn.edu

# ──── Paths ────
PROJECT_DIR="$HOME/repos/itr-neuraloperator"
export DATA_DIR="$PROJECT_DIR/data"
export PROJECT_ROOT="$PROJECT_DIR"

# Use node-local scratch for Optuna SQLite DB (avoids NFS locking),
# then copy results back to $HOME/fno_runs when the job finishes.
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
nvidia-smi

# ──── Run ────
# Any extra args passed to sbatch are forwarded to the training script.
# Example: sbatch slurm/train_fno_msi.sbatch training.epochs=200 tuning.n_trials=1
cd "$PROJECT_DIR"
python scripts/run_train.py "$@"

# ──── Copy results back to persistent storage ────
echo "Copying results from $LOCAL_RUNS to $PERSISTENT_RUNS ..."
cp -r "$LOCAL_RUNS"/* "$PERSISTENT_RUNS"/ 2>/dev/null || true

echo "Job $SLURM_JOB_ID finished at $(date)"
