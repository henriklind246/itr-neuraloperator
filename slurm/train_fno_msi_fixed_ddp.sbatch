#!/bin/bash
#SBATCH --job-name=fno_train_fixed_ddp
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --tmp=50G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lind1775@umn.edu

# Multi-GPU (DDP) training launcher. Each seed consumes all GPUs sequentially;
# multi-seed loops iterate seeds one at a time. Effective batch size matches
# the single-GPU config: per-GPU batch = training.batch_size / world_size.
# Evaluation stays single-GPU — invoke slurm/eval_fno_msi.sbatch after this job.

set -euo pipefail

# ──── Paths ────
PROJECT_DIR="${PROJECT_DIR:-$HOME/repos/itr-neuraloperator}"
export DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
export PROJECT_ROOT="$PROJECT_DIR"

# Active benchmark adapter (forcing|interfaces|source). load_config reads
# $BENCHMARK to pick conf/benchmark/<name>.yaml, which sets BOTH the model dims
# and the dataset adapter. Must match the benchmark used to generate $DATA_DIR.
# torchrun propagates this env var to every rank.
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

# ──── DDP / NCCL env ────
export OMP_NUM_THREADS=1
export NCCL_ASYNC_ERROR_HANDLING=1
# Collective timeout (minutes) for the process group; raised above NCCL's 10-min
# default so rank-0-only validation doesn't trip the watchdog. See distributed.py.
export DDP_TIMEOUT_MIN="${DDP_TIMEOUT_MIN:-30}"
# Uncomment on first runs to diagnose NCCL hangs:
# export NCCL_DEBUG=INFO
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1

# Number of GPUs visible to this job. `SLURM_GPUS_ON_NODE` is the canonical
# value; fall back to counting via nvidia-smi.
if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
    NPROC_PER_NODE="${SLURM_GPUS_ON_NODE}"
else
    NPROC_PER_NODE="$(nvidia-smi -L | wc -l)"
fi

# ──── Logging ────
echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "DATA_DIR=$DATA_DIR"
echo "RUNS_ROOT=$RUNS_ROOT"
echo "NPROC_PER_NODE=$NPROC_PER_NODE"
echo "BENCHMARK=$BENCHMARK"
nvidia-smi

# ──── Run ────
cd "$PROJECT_DIR"
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="$NPROC_PER_NODE" \
    scripts/run_train_fixed.py "$@"

# ──── Copy results back to persistent storage ────
echo "Copying results from $LOCAL_RUNS to $PERSISTENT_RUNS ..."
cp -r "$LOCAL_RUNS"/* "$PERSISTENT_RUNS"/ 2>/dev/null || true

echo "Job $SLURM_JOB_ID finished at $(date)"
