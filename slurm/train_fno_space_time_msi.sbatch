#!/bin/bash
#SBATCH --job-name=fno_space_time
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --signal=USR1@120
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-user=lind1775@umn.edu
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80

# Physics-only online training of the time-conditioned FNO on forcing_homogeneous.
# The FNO predicts every CN interval from T0, the forcing history and the query
# time; one space_time control preconditions the residual; every update draws
# fresh problems. Jobs with the same SEED share initial weights and problem
# streams, so one job per CONTROL/OPTIMIZER gives matched comparisons.
#
#   sbatch --export=ALL,CONTROL=spatial_mg,OPTIMIZER=Adam,LEARNING_RATE=3e-4 slurm/train_fno_space_time_msi.sbatch
#   sbatch slurm/train_fno_space_time_msi.sbatch model.parameters.width=32 training.soap.precondition_frequency=5
#   sbatch --export=ALL,RESUME_RUN_DIR=$HOME/fno_runs/<run> slurm/train_fno_space_time_msi.sbatch
#
# Trailing arguments are dotted config overrides applied after the defaults below.
# DATA_DIR only supplies prescribed forcing_homogeneous inputs, e.g. from
#   sbatch --export=ALL,BENCHMARK=forcing_homogeneous slurm/generate_data_msi.sbatch data/forcing_homogeneous 512
# Without NORMALIZATION_PATH, (mu, sigma) are frozen from DATA_DIR's training split.
# runs/ is gitignored: rsync the direct-state gate directories to MSI.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:?Submit from the project directory or set PROJECT_DIR}}"
export PROJECT_ROOT="$PROJECT_DIR"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data/forcing_homogeneous}"
NORMALIZATION_PATH="${NORMALIZATION_PATH:-}"
PHASE0_DIR="${PHASE0_DIR:-$PROJECT_DIR/runs/physics_test_phase0_raw_exact_20261001}"
PHASE4_DIR="${PHASE4_DIR:-$PROJECT_DIR/runs/physics_test_phase4_direct_state_mg_20261002}"
PERSISTENT_RUNS="${PERSISTENT_RUNS:-$HOME/fno_runs}"

CONTROL="${CONTROL:-mg}"                    # raw | exact | spatial_exact | spatial_mg | spatial_mg_detached | mg
OPTIMIZER="${OPTIMIZER:-SOAP}"              # SOAP | Adam | AdamW
LEARNING_RATE="${LEARNING_RATE:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
SCHEDULER="${SCHEDULER:-PICViTExponential}" # PICViTExponential | RIGNOThreePhase | CosineWarmRestarts | StepLR
DECAY_EVERY="${DECAY_EVERY:-250}"           # PICViTExponential; schedule lengths count optimizer updates
DECAY_RATE="${DECAY_RATE:-0.9}"
MIN_LR="${MIN_LR:-0.00001}"
BATCH_SIZE="${BATCH_SIZE:-8}"
UPDATES="${UPDATES:-5000}"
PREFIX_STEPS="${PREFIX_STEPS:-60}"
TIME_CHUNK_SIZE="${TIME_CHUNK_SIZE:-4}"
VALIDATE_EVERY="${VALIDATE_EVERY:-500}"
FULL_VALIDATE_EVERY="${FULL_VALIDATE_EVERY:-1000}"
VALIDATION_CASES="${VALIDATION_CASES:-128}"
SEED="${SEED:-42}"
# Physics warm-up: during the first PHYSICS_WARMUP_UPDATES updates each problem
# batch is reused for PHYSICS_WARMUP_RESAMPLE_EVERY updates (archive used 500/50).
PHYSICS_WARMUP_UPDATES="${PHYSICS_WARMUP_UPDATES:-0}"
PHYSICS_WARMUP_RESAMPLE_EVERY="${PHYSICS_WARMUP_RESAMPLE_EVERY:-1}"
ALLOCATION_SECONDS="${ALLOCATION_SECONDS:-28800}"  # keep equal to --time for the startup fit estimate

case "$CONTROL" in
    mg|spatial_mg|spatial_mg_detached) GATE_DIR="$PHASE4_DIR" ;;
    *) GATE_DIR="$PHASE0_DIR" ;;
esac
WARMUP_TAG=""
if [[ "$PHYSICS_WARMUP_UPDATES" != 0 ]]; then
    WARMUP_TAG="_pw${PHYSICS_WARMUP_UPDATES}x${PHYSICS_WARMUP_RESAMPLE_EVERY}"
fi
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-fno_st_${CONTROL}_${OPTIMIZER}_lr${LEARNING_RATE}_bs${BATCH_SIZE}_n${PREFIX_STEPS}_u${UPDATES}_s${SEED}${WARMUP_TAG}}"
RUN_DIR="${RUN_DIR:-$PERSISTENT_RUNS/${EXPERIMENT_NAME}_${SLURM_JOB_ID:?}}"
export BENCHMARK=forcing_homogeneous
export REPRESENTATION=temporal_encoder
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PCVIT_JOB_START_EPOCH="$(date +%s)"

module load miniforge
eval "$(conda shell.bash hook)"
conda activate itr

cd "$PROJECT_DIR"
mkdir -p "$PERSISTENT_RUNS"
echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"
echo "PROJECT_DIR=$PROJECT_DIR DATA_DIR=$DATA_DIR GATE_DIR=$GATE_DIR NORMALIZATION_PATH=${NORMALIZATION_PATH:-<frozen from DATA_DIR>}"
echo "CONTROL=$CONTROL OPTIMIZER=$OPTIMIZER LEARNING_RATE=$LEARNING_RATE SCHEDULER=$SCHEDULER BATCH_SIZE=$BATCH_SIZE UPDATES=$UPDATES PREFIX_STEPS=$PREFIX_STEPS"
echo "PHYSICS_WARMUP_UPDATES=$PHYSICS_WARMUP_UPDATES PHYSICS_WARMUP_RESAMPLE_EVERY=$PHYSICS_WARMUP_RESAMPLE_EVERY"
echo "Trailing overrides (applied last): $*"
nvidia-smi

if [[ -n "${RESUME_RUN_DIR:-}" ]]; then
    echo "Resuming $RESUME_RUN_DIR using its frozen configuration"
    srun python -u scripts/run_train_pino.py --resume --output-dir "$RESUME_RUN_DIR"
else
    overrides=(
        training.device=cuda "training.optimizer=$OPTIMIZER" "training.learning_rate=$LEARNING_RATE"
        "training.weight_decay=$WEIGHT_DECAY" "training.grad_clip=$GRAD_CLIP"
        'training.soap.betas=[0.95,0.95]' training.soap.shampoo_beta=0.95
        training.soap.eps=1e-8 training.soap.precondition_frequency=10
        training.soap.max_precond_dim=2048 training.soap.merge_dims=false
        training.soap.precondition_1d=false
        "training.scheduler.type=$SCHEDULER" "training.scheduler.decay_every=$DECAY_EVERY"
        "training.scheduler.decay_rate=$DECAY_RATE" "training.scheduler.min_lr=$MIN_LR"
        physics_test.model_type=fno model.parameters.hard_right_dirichlet=true model.parameters.dropout=0.0
        physics_test.pino.residual_mode=space_time "physics_test.pino.controls=[\"$CONTROL\"]"
        "physics_test.pino.prefix_steps=$PREFIX_STEPS" "physics_test.pino.time_chunk_size=$TIME_CHUNK_SIZE"
        "physics_test.pino.batch_size=$BATCH_SIZE" "physics_test.pino.online_updates=$UPDATES"
        physics_test.pino.online_sampling=per_update
        "physics_test.pino.online_validate_every=$VALIDATE_EVERY"
        "physics_test.pino.full_validate_every=$FULL_VALIDATE_EVERY"
        "physics_test.pino.validation_cases=$VALIDATION_CASES" physics_test.pino.validation_seed=1042
        physics_test.pino.validation_rollout_steps=60 physics_test.pino.validation_batch_size=8
        physics_test.pino.compact_metrics=true physics_test.pino.log_every=50
        "physics_test.pino.seed=$SEED" physics_test.pino.ic_weight=0.0
        "physics_test.pino.physics_warmup_updates=$PHYSICS_WARMUP_UPDATES"
        "physics_test.pino.physics_warmup_resample_every=$PHYSICS_WARMUP_RESAMPLE_EVERY"
        physics_test.pino.checkpoint_keep_all=false
        'physics_test.pino.startup_profile_updates=[50,100]'
        "physics_test.pino.allocation_seconds=$ALLOCATION_SECONDS"
    )
    normalization=()
    if [[ -n "$NORMALIZATION_PATH" ]]; then
        normalization=(--normalization "$NORMALIZATION_PATH")
    fi
    echo "Writing directly to $RUN_DIR"
    srun python -u scripts/run_train_pino.py \
        --stage online --data-dir "$DATA_DIR" --output-dir "$RUN_DIR" --phase0-dir "$GATE_DIR" \
        ${normalization[@]+"${normalization[@]}"} "${overrides[@]}" "$@"
fi
