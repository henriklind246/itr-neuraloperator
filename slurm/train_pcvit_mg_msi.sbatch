#!/bin/bash
#SBATCH --job-name=fno_boundary_mg_soap
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=09:00:00
#SBATCH --signal=USR1@120
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-user=lind1775@umn.edu
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_50,TIME_LIMIT_80,TIME_LIMIT_90

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:?Submit from the project directory or set PROJECT_DIR}}"
export PROJECT_ROOT="$PROJECT_DIR"
export DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data/physics_test_single_development_20261001}"
PHASE4_DIR="${PHASE4_DIR:-$PROJECT_DIR/runs/physics_test_phase4_direct_state_mg_20261002}"
NORMALIZATION_PATH="${NORMALIZATION_PATH:-$PROJECT_DIR/runs/physics_test_phase4_cvit_autoregressive_mg_prefix5_adam_seed42_20261002/normalization.json}"
PERSISTENT_RUNS="${PERSISTENT_RUNS:-$HOME/fno_runs}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-fno_boundary_mg_soap_prefix20_10k}"
RUN_DIR="${RUN_DIR:-$PERSISTENT_RUNS/${EXPERIMENT_NAME}_${SLURM_JOB_ID:?}}"
FIXED_MG_DIR="${FIXED_MG_DIR:-${RUN_DIR}_fixed}"
export BENCHMARK=diffusion_forcing_single
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
echo "PROJECT_DIR=$PROJECT_DIR DATA_DIR=$DATA_DIR"
echo "PHASE4_DIR=$PHASE4_DIR FIXED_MG_DIR=$FIXED_MG_DIR NORMALIZATION_PATH=$NORMALIZATION_PATH"
nvidia-smi

if [[ -n "${RESUME_RUN_DIR:-}" ]]; then
    echo "Resuming $RESUME_RUN_DIR using its frozen configuration"
    srun python -u scripts/run_train_pino.py --resume --output-dir "$RESUME_RUN_DIR"
else
    overrides=(
        training.device=cuda training.optimizer=SOAP training.learning_rate=0.001
        training.weight_decay=0.0 training.grad_clip=1.0 training.batch_size=8
        'training.soap.betas=[0.95,0.95]' training.soap.shampoo_beta=0.95
        training.soap.eps=1e-8 training.soap.precondition_frequency=10
        training.soap.max_precond_dim=2048 training.soap.merge_dims=false
        training.soap.precondition_1d=false
        training.scheduler.type=PICViTExponential training.scheduler.decay_every=250
        training.scheduler.decay_rate=0.90 training.scheduler.min_lr=0.00001
        physics_test.model_type=fno
        model.parameters.modes1=16 model.parameters.modes2=16
        model.parameters.width=64 model.parameters.n_layers=4
        model.parameters.forcing_spatial_mode=boundary_extender
        model.parameters.forcing_cond_mode=spatial_only model.parameters.hard_right_dirichlet=true
        physics_test.pino.residual_mode=autoregressive 'physics_test.pino.controls=["mg"]'
        physics_test.pino.prefix_steps=20 physics_test.pino.batch_size=8
        physics_test.pino.online_updates=10000 physics_test.pino.online_sampling=per_update
        physics_test.pino.online_validate_every=500 physics_test.pino.full_validate_every=1000
        physics_test.pino.validation_cases=128 physics_test.pino.validation_seed=1042
        physics_test.pino.validation_rollout_steps=60 physics_test.pino.validation_batch_size=8
        physics_test.pino.compact_metrics=true physics_test.pino.log_every=50
        physics_test.pino.seed=42 physics_test.pino.ic_weight=0.0
        physics_test.pino.checkpoint_keep_all=false physics_test.pino.query_chunk_size=2048
        'physics_test.pino.startup_profile_updates=[50,100]' physics_test.pino.allocation_seconds=32400
    )
    if [[ ! -f "$FIXED_MG_DIR/final_metrics.json" ]]; then
        echo "Running the 1000-update FNO fixed-case gate in $FIXED_MG_DIR"
        srun python -u scripts/run_train_pino.py \
            --stage fixed --data-dir "$DATA_DIR" --output-dir "$FIXED_MG_DIR" \
            --phase0-dir "$PHASE4_DIR" --normalization "$NORMALIZATION_PATH" \
            "${overrides[@]}" physics_test.pino.online_sampling=precomputed \
            physics_test.pino.fixed_updates=1000 physics_test.pino.fixed_validate_every=100
    fi
    echo "Writing directly to $RUN_DIR"
    srun python -u scripts/run_train_pino.py \
        --stage online --data-dir "$DATA_DIR" --output-dir "$RUN_DIR" \
        --phase0-dir "$PHASE4_DIR" --fixed-screen-dir "$FIXED_MG_DIR" \
        --normalization "$NORMALIZATION_PATH" "${overrides[@]}"
fi
