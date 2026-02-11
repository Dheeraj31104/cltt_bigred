#!/bin/bash
# Example Slurm job array to sweep several SimCLR runs.
# Edit the SBATCH lines to match your cluster defaults before submitting.

#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -A r00117
#SBATCH --job-name=simclr_ego
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=logs/simclr_%A_%a.out
#SBATCH --mail-user=dhkara@iu.edu
#SBATCH --mail-type=BEGIN,END,REQUEUE
#SBATCH --requeue
#SBATCH --signal=B:USR1@120

set -euo pipefail

# ---- User settings (edit these) ---------------------------------------------
ROOT_DIR="/N/project/cogai/dhkara/cropped_clips"          # folder with extracted frame folders
CHECKPOINT_ROOT="$PWD/checkpoints"                        # base output dir for checkpoints

# Core training settings
EPOCHS=100
IMAGE_SIZE=224
BATCH_SIZE_DEFAULT=256
LR_DEFAULT=1e-3
SEED_DEFAULT=0
TEMPERATURE=0.5
PROJ_DIM=128
WEIGHT_DECAY=1e-4
MAX_GRAD_NORM=1.0

# Runtime tuning
SUB_BATCH_SIZE=""                                         # set to "" to disable grad accumulation
MAX_BATCHES_PER_EPOCH=""                                  # set to e.g. 500 to cap per-epoch work
NUM_WORKERS_DEFAULT="${SLURM_CPUS_PER_TASK:-4}"

# Dataset options
TWO_VIEW_OFFSET=1                                           # set to integer for paired-window mode
USE_OBJECT_FOCUS=1                                          # set to 1 to enable background blur

# Linear eval on CIFAR (set LINEAR_EVAL_EVERY>0 to enable)
LINEAR_EVAL_EVERY=0
LINEAR_EVAL_EPOCHS=5
LINEAR_EVAL_BATCH_SIZE=256
LINEAR_EVAL_LR=0.1
LINEAR_EVAL_WEIGHT_DECAY=0.0
LINEAR_EVAL_MAX_BATCHES=""
LINEAR_EVAL_TRAIN_FRACTION=0.7
LINEAR_EVAL_SPLIT_SEED=42
CIFAR_DATASET="cifar10"
CIFAR_DATA_DIR="$PWD/data/cifar"
CIFAR_DOWNLOAD=0

# CSV logging
ENABLE_CSV_LOG=1                                            # set to 1 to enable CSV logging

# Optional env bootstrap (module/conda/etc). Uncomment and edit for your cluster.
# module load cuda/12.1
# Avoid "PS1: unbound variable" from Lmod when nounset is on.
: "${PS1:=}"
set +u
module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate torch-env
set -u
# Interactive helper:
# srun -p gpu -A r00117 --gpus-per-node 1 --time=6:00:00 --pty bash

requeue_job() {
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "Requeuing job ${SLURM_JOB_ID}..."
    scontrol requeue "${SLURM_JOB_ID}"
  fi
}

should_requeue() {
  local latest_ckpt
  latest_ckpt=$(ls -1 "${CKPT_DIR}"/simclr_epoch_*.pth 2>/dev/null | sort | tail -n 1 || true)
  if [[ -z "${latest_ckpt}" ]]; then
    return 0
  fi
  local latest_epoch
  latest_epoch=$(basename "${latest_ckpt}" | sed -E 's/simclr_epoch_([0-9]+)\.pth/\1/')
  if [[ "${latest_epoch}" -ge "${EPOCHS}" ]]; then
    echo "Latest checkpoint is epoch ${latest_epoch}; target ${EPOCHS} reached, not requeuing."
    return 1
  fi
  return 0
}

trap 'echo "Caught SIGUSR1, requesting requeue."; should_requeue && requeue_job; exit 0' USR1
trap 'echo "Caught SIGTERM, requesting requeue."; should_requeue && requeue_job; exit 0' TERM

# Each entry becomes one array task. Tweak as needed.
# You can override any defaults above per-config (e.g., TEMPERATURE=0.2).
CONFIGS=(
  "WIN=5 STEP=10 BATCH=256 LR=1e-3 SEED=0"
)
# ----------------------------------------------------------------------------

IDX="${SLURM_ARRAY_TASK_ID:-0}"
if [[ "$IDX" -ge "${#CONFIGS[@]}" ]]; then
  echo "Array index $IDX exceeds CONFIGS size ${#CONFIGS[@]}" >&2
  exit 1
fi

# shellcheck disable=SC2086
eval "${CONFIGS[$IDX]}"

# Required per-config variables
: "${WIN:?WIN must be set in CONFIGS}"
: "${STEP:?STEP must be set in CONFIGS}"

# Provide safe fallbacks for optional per-config overrides
BATCH="${BATCH:-$BATCH_SIZE_DEFAULT}"
LR="${LR:-$LR_DEFAULT}"
SEED="${SEED:-$SEED_DEFAULT}"
NUM_WORKERS="${NUM_WORKERS:-$NUM_WORKERS_DEFAULT}"

RUN_NAME="ws${WIN}_step${STEP}_lr${LR}_seed${SEED}"
CKPT_DIR="${CHECKPOINT_ROOT}/${RUN_NAME}"
mkdir -p "$CKPT_DIR"

# Build optional args cleanly
EXTRA_ARGS=()

# Core args that now exist explicitly in train_simclr.py
EXTRA_ARGS+=(--image-size "$IMAGE_SIZE")
EXTRA_ARGS+=(--temperature "$TEMPERATURE")
EXTRA_ARGS+=(--proj-dim "$PROJ_DIM")
EXTRA_ARGS+=(--weight-decay "$WEIGHT_DECAY")
EXTRA_ARGS+=(--max-grad-norm "$MAX_GRAD_NORM")

# Runtime tuning
[[ -n "${SUB_BATCH_SIZE}" ]] && EXTRA_ARGS+=(--sub-batch-size "$SUB_BATCH_SIZE")
[[ -n "${MAX_BATCHES_PER_EPOCH}" ]] && EXTRA_ARGS+=(--max-batches-per-epoch "$MAX_BATCHES_PER_EPOCH")

# Dataset options
[[ -n "${TWO_VIEW_OFFSET}" ]] && EXTRA_ARGS+=(--two-view-offset "$TWO_VIEW_OFFSET")
[[ "${USE_OBJECT_FOCUS}" == "1" ]] && EXTRA_ARGS+=(--use-object-focus)

# Linear eval configuration
if [[ "${LINEAR_EVAL_EVERY}" -gt 0 ]]; then
  EXTRA_ARGS+=(--linear-eval-every "$LINEAR_EVAL_EVERY")
  EXTRA_ARGS+=(--linear-eval-epochs "$LINEAR_EVAL_EPOCHS")
  EXTRA_ARGS+=(--linear-eval-batch-size "$LINEAR_EVAL_BATCH_SIZE")
  EXTRA_ARGS+=(--linear-eval-lr "$LINEAR_EVAL_LR")
  EXTRA_ARGS+=(--linear-eval-weight-decay "$LINEAR_EVAL_WEIGHT_DECAY")
  EXTRA_ARGS+=(--linear-eval-train-fraction "$LINEAR_EVAL_TRAIN_FRACTION")
  EXTRA_ARGS+=(--linear-eval-split-seed "$LINEAR_EVAL_SPLIT_SEED")
  [[ -n "${LINEAR_EVAL_MAX_BATCHES}" ]] && EXTRA_ARGS+=(--linear-eval-max-batches "$LINEAR_EVAL_MAX_BATCHES")
  EXTRA_ARGS+=(--cifar-dataset "$CIFAR_DATASET")
  EXTRA_ARGS+=(--cifar-data-dir "$CIFAR_DATA_DIR")
  [[ "${CIFAR_DOWNLOAD}" == "1" ]] && EXTRA_ARGS+=(--cifar-download)
fi

# CSV logging configuration
if [[ "${ENABLE_CSV_LOG}" == "1" ]]; then
  EXTRA_ARGS+=(--csv-log --csv-log-path "${CKPT_DIR}/metrics.csv")
fi

echo "Starting run ${RUN_NAME}"
echo "Saving checkpoints to ${CKPT_DIR}"
echo "Using NUM_WORKERS=${NUM_WORKERS}, BATCH=${BATCH}, LR=${LR}, TEMP=${TEMPERATURE}"

srun python train_simclr.py \
  --root-dir "$ROOT_DIR" \
  --window-size "$WIN" \
  --frame-step "$STEP" \
  --batch-size "$BATCH" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --seed "$SEED" \
  --checkpoint-dir "$CKPT_DIR" \
  --resume \
  --num-workers "$NUM_WORKERS" \
  "${EXTRA_ARGS[@]}"
