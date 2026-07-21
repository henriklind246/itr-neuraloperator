#!/bin/bash
#SBATCH --job-name=cvit_pino_1step
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --tmp=50G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lind1775@umn.edu

# One-step Markov InterfaceCViT, physics-only (PINO) training.
#
# Trains the 2D forcing-image InterfaceCViT as a state-conditioned operator
# T_n -> T_{n+1} under a variational / preconditioned-defect CN objective with
# hard global-storage + right-Dirichlet conservation. The FV manifold is solved
# ONLINE from freshly sampled descriptors (a continuously refreshed state pool),
# so no paired training snapshots are consumed. The interfaces dataset in
# $DATA_DIR is still required: it supplies the frozen normalization constants
# (mu_global, sigma_global), the solver grids, and the jump scale baked into the
# run. Validation/test descriptor sets are seed-frozen inside the runner and the
# concrete encoder/architecture metadata is written to <run>/architecture.json
# at startup.
#
# Single-GPU by design (run_config_seeds_pino has no DDP path). SOAP optimizer.
#
# Usage (submit from the repo root):
#   sbatch slurm/train_interfaces_one_step_pino_msi.sbatch \
#       training.pino.one_step.objective=variational \
#       training.pino.one_step.updates=8000 training.pino.one_step.n_cases=8
#
# Any trailing key=value pairs are forwarded to run_train_pino.py and OVERRIDE
# the baked-in defaults below (last write wins). Select the experiment/data/
# output roots via the env vars just below, NOT dotted overrides:
#   DATA_DIR         interfaces dataset dir (trajectories.npy, *_grid.npy, ...)
#   EXPERIMENT_NAME  run label -> $RUNS_ROOT/$EXPERIMENT_NAME/config<id>/seed<N>
#   PROJECT_DIR      repo checkout (default $HOME/repos/itr-neuraloperator)
#   RESUME           set to 1 to exact-resume a prior run of the SAME
#                    EXPERIMENT_NAME (stages it back into node-local scratch,
#                    bypasses the fresh-name guard, forwards resume=true).

set -euo pipefail

# ──── Paths ────
PROJECT_DIR="${PROJECT_DIR:-$HOME/repos/itr-neuraloperator}"
export PROJECT_ROOT="$PROJECT_DIR"
export DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"

# The one-step image-CViT path is interfaces-only; a dotted benchmark.name=
# override is NOT sufficient, so always select via these env vars.
export BENCHMARK="${BENCHMARK:-interfaces}"
export REPRESENTATION="${REPRESENTATION:-temporal_encoder}"

# Descriptive run label (config resolves experiment.name from $EXPERIMENT_NAME).
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-interfaces_one_step_pino}"

RESUME="${RESUME:-0}"

# Node-local scratch for run outputs, copied back to $HOME/cvit_runs on exit.
LOCAL_RUNS="/tmp/cvit_runs_${SLURM_JOB_ID}"
export RUNS_ROOT="$LOCAL_RUNS"
PERSISTENT_RUNS="${PERSISTENT_RUNS:-$HOME/cvit_runs}"
mkdir -p "$LOCAL_RUNS" "$PERSISTENT_RUNS"

if [[ "$RESUME" == "1" ]]; then
    # Exact-resume: stage the prior run back into node-local scratch so the
    # runner finds its resume_state.pt, and forward the resume flag. The
    # fresh-name guard is intentionally skipped for this path.
    if [[ ! -d "$PERSISTENT_RUNS/$EXPERIMENT_NAME" ]]; then
        echo "ERROR: RESUME=1 but $PERSISTENT_RUNS/$EXPERIMENT_NAME not found." >&2
        exit 1
    fi
    echo "RESUME=1: staging $PERSISTENT_RUNS/$EXPERIMENT_NAME -> $LOCAL_RUNS ..."
    cp -r "$PERSISTENT_RUNS/$EXPERIMENT_NAME" "$LOCAL_RUNS"/
elif [[ "$BENCHMARK" == "interfaces" && -e "$PERSISTENT_RUNS/$EXPERIMENT_NAME" ]]; then
    echo "ERROR: interface image-CViT runs require a fresh EXPERIMENT_NAME;" >&2
    echo "       $PERSISTENT_RUNS/$EXPERIMENT_NAME already exists." >&2
    echo "       (set RESUME=1 to exact-resume this run instead.)" >&2
    exit 1
fi

# Always copy partial results back, even if training aborts.
copy_back() {
    echo "Copying results from $LOCAL_RUNS to $PERSISTENT_RUNS ..."
    cp -r "$LOCAL_RUNS"/* "$PERSISTENT_RUNS"/ 2>/dev/null || true
}
trap copy_back EXIT

# ──── Environment ────
module load miniforge
eval "$(conda shell.bash hook)"
conda activate itr

# ──── Preflight: the interfaces dataset must exist in $DATA_DIR ────
# Online FV-solve needs no training snapshots, but the run DOES require the
# frozen normalization constants + grids that load_diffusion_data reads here.
if [[ ! -f "$DATA_DIR/trajectories.npy" ]]; then
    echo "ERROR: $DATA_DIR/trajectories.npy not found." >&2
    echo "The one-step run reads mu/sigma/grids + the jump scale from it." >&2
    echo "Generate it first, e.g.:" >&2
    echo "  BENCHMARK=interfaces python data/generate_dataset.py --num-sims 8000 --benchmark interfaces" >&2
    exit 1
fi

# ──── Logging ────
echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "DATA_DIR=$DATA_DIR"
echo "RUNS_ROOT=$RUNS_ROOT"
echo "BENCHMARK=$BENCHMARK / REPRESENTATION=$REPRESENTATION"
echo "EXPERIMENT_NAME=$EXPERIMENT_NAME / RESUME=$RESUME"
nvidia-smi

# ──── Run ────
# Baked-in defaults for the one-step physics-only recipe; forwarded "$@"
# overrides win. mode=one_step selects run_one_seed_interfaces_one_step_pino in
# the interfaces dispatch branch.
cd "$PROJECT_DIR"
# Stream stdout live: SLURM redirects to a file (not a TTY), so Python
# block-buffers stdout and the pre-training setup phase (imports, data load,
# model build) would otherwise show nothing until the buffer fills.
export PYTHONUNBUFFERED=1

RESUME_OVERRIDE=()
if [[ "$RESUME" == "1" ]]; then
    RESUME_OVERRIDE=(training.pino.one_step.resume=true)
fi

python -u scripts/run_train_pino.py \
    training.pino.mode=one_step \
    training.optimizer=SOAP \
    "${RESUME_OVERRIDE[@]}" \
    "$@"

echo "Job $SLURM_JOB_ID finished at $(date)"
