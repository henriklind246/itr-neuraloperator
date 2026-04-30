#!/bin/bash
#SBATCH --job-name=fno_same_sim
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --tmp=50G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lind1775@umn.edu

set -euo pipefail

PROJECT_DIR="$HOME/repos/itr-neuraloperator"
export DATA_DIR="$PROJECT_DIR/data"
export PROJECT_ROOT="$PROJECT_DIR"

LOCAL_RUNS="/tmp/fno_runs_${SLURM_JOB_ID}"
export RUNS_ROOT="$LOCAL_RUNS"
PERSISTENT_RUNS="$HOME/fno_runs"
mkdir -p "$LOCAL_RUNS" "$PERSISTENT_RUNS"

SAME_SIM_SEED="${SAME_SIM_SEED:-42}"
SAME_SIM_VAL_PAIR_FRAC="${SAME_SIM_VAL_PAIR_FRAC:-0.1}"
SAME_SIM_SPLIT_SEED="${SAME_SIM_SPLIT_SEED:-0}"
SAME_SIM_RUN_DIR="${SAME_SIM_RUN_DIR:-$LOCAL_RUNS/same_sim_holdout/seed${SAME_SIM_SEED}}"
CROSS_SIM_METRICS="${CROSS_SIM_METRICS:-}"

module load miniforge
eval "$(conda shell.bash hook)"
conda activate itr

echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "DATA_DIR=$DATA_DIR"
echo "RUNS_ROOT=$RUNS_ROOT"
echo "SAME_SIM_SEED=$SAME_SIM_SEED"
echo "SAME_SIM_VAL_PAIR_FRAC=$SAME_SIM_VAL_PAIR_FRAC"
echo "SAME_SIM_SPLIT_SEED=$SAME_SIM_SPLIT_SEED"
echo "SAME_SIM_RUN_DIR=$SAME_SIM_RUN_DIR"
echo "CROSS_SIM_METRICS=$CROSS_SIM_METRICS"
nvidia-smi

cd "$PROJECT_DIR"

cmd=(
  python scripts/eval_same_sim_holdout.py
  --seed "$SAME_SIM_SEED"
  --run-dir "$SAME_SIM_RUN_DIR"
  --val-pair-frac "$SAME_SIM_VAL_PAIR_FRAC"
  --split-seed "$SAME_SIM_SPLIT_SEED"
)

if [[ -n "$CROSS_SIM_METRICS" ]]; then
  cmd+=(--cross-sim-metrics "$CROSS_SIM_METRICS")
fi

cmd+=("$@")

echo "Running: ${cmd[*]}"
"${cmd[@]}"

echo "Copying results from $LOCAL_RUNS to $PERSISTENT_RUNS ..."
cp -r "$LOCAL_RUNS"/* "$PERSISTENT_RUNS"/ 2>/dev/null || true

echo "Job $SLURM_JOB_ID finished at $(date)"
