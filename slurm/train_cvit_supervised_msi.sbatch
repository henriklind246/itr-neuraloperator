#!/bin/bash
#SBATCH --job-name=cvit_supervised
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --tmp=50G
#SBATCH --signal=TERM@180
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lind1775@umn.edu

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$PROJECT_DIR" || ! -f "$PROJECT_DIR/scripts/run_train_pino.py" ]]; then
    echo "ERROR: PROJECT_DIR must point to the repository checkout." >&2
    exit 1
fi
if [[ -z "${DATA_DIR:-}" ]]; then
    echo "ERROR: DATA_DIR must point to the existing diffusion_forcing_single dataset." >&2
    exit 1
fi

SOURCE_DATA_DIR="$DATA_DIR"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-dfs_single_cvit_supervised}"
PERSISTENT_RUNS="${PERSISTENT_RUNS:-$HOME/cvit_runs}"
RESUME="${RESUME:-0}"
STAGE_DATA="${STAGE_DATA:-1}"
SCRATCH_ROOT="${SLURM_TMPDIR:-/tmp}"
LOCAL_RUNS="$SCRATCH_ROOT/cvit_runs_${SLURM_JOB_ID}"
LOCAL_DATA="$SCRATCH_ROOT/cvit_data_${SLURM_JOB_ID}"
LOCAL_EXPERIMENT="$LOCAL_RUNS/$EXPERIMENT_NAME"
PERSISTENT_EXPERIMENT="$PERSISTENT_RUNS/$EXPERIMENT_NAME"

required_files=(
    trajectories.npy
    x_grid.npy
    y_grid.npy
    t_grid.npy
    sim_params.npy
    meta.npy
    ramp_seconds.npy
)
for file_name in "${required_files[@]}"; do
    if [[ ! -f "$SOURCE_DATA_DIR/$file_name" ]]; then
        echo "ERROR: required saved-data file is missing: $SOURCE_DATA_DIR/$file_name" >&2
        exit 1
    fi
done

mkdir -p "$LOCAL_RUNS" "$PERSISTENT_RUNS"
if [[ "$RESUME" == "1" ]]; then
    if [[ ! -d "$PERSISTENT_EXPERIMENT" ]]; then
        echo "ERROR: RESUME=1 but $PERSISTENT_EXPERIMENT does not exist." >&2
        exit 1
    fi
    cp -R "$PERSISTENT_EXPERIMENT" "$LOCAL_RUNS/"
elif [[ -e "$PERSISTENT_EXPERIMENT" ]]; then
    echo "ERROR: $PERSISTENT_EXPERIMENT already exists; use a fresh name or RESUME=1." >&2
    exit 1
fi

copy_back() {
    if [[ -d "$LOCAL_EXPERIMENT" ]]; then
        mkdir -p "$PERSISTENT_EXPERIMENT"
        cp -R "$LOCAL_EXPERIMENT/." "$PERSISTENT_EXPERIMENT/"
    fi
}
on_term() {
    echo "Termination signal received; preserving the latest run artifacts." >&2
    exit 143
}
trap copy_back EXIT
trap on_term TERM INT

if [[ "$STAGE_DATA" == "1" ]]; then
    mkdir -p "$LOCAL_DATA"
    for file_name in "${required_files[@]}"; do
        cp "$SOURCE_DATA_DIR/$file_name" "$LOCAL_DATA/$file_name"
    done
    if [[ -f "$SOURCE_DATA_DIR/dt.npy" ]]; then
        cp "$SOURCE_DATA_DIR/dt.npy" "$LOCAL_DATA/dt.npy"
    fi
    export DATA_DIR="$LOCAL_DATA"
else
    export DATA_DIR="$SOURCE_DATA_DIR"
fi

export PROJECT_ROOT="$PROJECT_DIR"
export RUNS_ROOT="$LOCAL_RUNS"
export BENCHMARK="diffusion_forcing_single"
export REPRESENTATION="temporal_encoder"
export EXPERIMENT_NAME
export PYTHONUNBUFFERED=1

module load miniforge
eval "$(conda shell.bash hook)"
conda activate itr

echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"
echo "PROJECT_DIR=$PROJECT_DIR"
echo "SOURCE_DATA_DIR=$SOURCE_DATA_DIR"
echo "DATA_DIR=$DATA_DIR"
echo "RUNS_ROOT=$RUNS_ROOT"
echo "PERSISTENT_EXPERIMENT=$PERSISTENT_EXPERIMENT"
echo "RESUME=$RESUME STAGE_DATA=$STAGE_DATA"
nvidia-smi

cd "$PROJECT_DIR"
python -u scripts/run_train_pino.py \
    training.optimizer=AdamW \
    training.learning_rate=5.0e-4 \
    training.epochs=10001 \
    training.validate_every=1000 \
    training.grad_clip=1.0 \
    training.pino.lambda_data=1.0 \
    training.pino.lambda_r=0.0 \
    training.pino.lambda_ic=0.0 \
    training.pino.lambda_bc=0.0 \
    training.pino.lambda_bc_left=0.0 \
    training.pino.n_data_sims=8 \
    training.pino.n_data_pts=1024 \
    training.pino.data_query_chunk=0 \
    training.pino.forcing.ny_img=96 \
    training.pino.forcing.nt_img=128 \
    training.pino.forcing.val_sim_batch=8 \
    training.pino.forcing.val_query_chunk=2048 \
    training.pino.forcing.save_latest_every=25 \
    training.gradnorm.enabled=false \
    training.pino.causal.enabled=false \
    "$@"

echo "Job $SLURM_JOB_ID finished at $(date)"
