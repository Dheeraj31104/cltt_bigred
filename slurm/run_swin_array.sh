#!/bin/bash
# Slurm job array to sweep SimCLR runs with Swin Transformer backbone.
# Mirrors run_simclr_array.sh but uses --backbone swin_t (or swin_s / swin_b).

#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -A r00117
#SBATCH --job-name=swin_ego
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/swin_%A_%a.out
#SBATCH --mail-user=dhkara@iu.edu
#SBATCH --mail-type=BEGIN,END,REQUEUE
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --array=0-2

set -euo pipefail

# ---- User settings (override any of these from the command line) ------------
# Usage examples:
#   BACKBONE=swin_s sbatch slurm/run_swin_array.sh
#   BACKBONE=swin_b EPOCHS=50 TEMPERATURE=0.2 sbatch slurm/run_swin_array.sh
ROOT_DIR="${ROOT_DIR:-/N/project/cogai/dhkara/cropped_clips}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$PWD/checkpoints}"

BACKBONE="${BACKBONE:-swin_t}"          # swin_t | swin_s | swin_b

# Core training settings
EPOCHS="${EPOCHS:-100}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
BATCH_SIZE_DEFAULT="${BATCH_SIZE_DEFAULT:-64}"   # Swin needs more memory; keep batch smaller
LR_DEFAULT="${LR_DEFAULT:-1e-3}"
SEED_DEFAULT="${SEED_DEFAULT:-0}"
TEMPERATURE="${TEMPERATURE:-0.5}"
PROJ_DIM="${PROJ_DIM:-128}"
WEIGHT_DECAY=1e-4
MAX_GRAD_NORM=1.0

# Runtime tuning
SUB_BATCH_SIZE=""
MAX_BATCHES_PER_EPOCH=""
NUM_WORKERS_DEFAULT="${SLURM_CPUS_PER_TASK:-4}"

# Dataset options
WINDOW_STRIDE=10
MAX_WINDOWS_PER_CLIP=0
SINGLE_WINDOW_SHORT_CLIPS=1
SHORT_CLIP_WINDOW_THRESHOLD=2
USE_OBJECT_FOCUS=1

# Linear eval on CIFAR
LINEAR_EVAL_EVERY=5
LINEAR_EVAL_EPOCHS=5
LINEAR_EVAL_BATCH_SIZE=256
LINEAR_EVAL_LR=0.1
LINEAR_EVAL_WEIGHT_DECAY=0.0
LINEAR_EVAL_MAX_BATCHES=""
LINEAR_EVAL_TRAIN_FRACTION=0.7
LINEAR_EVAL_SPLIT_SEED=42
CIFAR_DATASET="cifar10"
CIFAR_DATA_DIR="$PWD/data/cifar"
CIFAR_DOWNLOAD=1

ENABLE_CSV_LOG=1
# ----------------------------------------------------------------------------

: "${PS1:=}"
set +u
module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate torch-env
set -u

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

CONFIGS=(
  "WIN=5 STEP=2 BATCH=64 LR=1e-2 SEED=0"
  "WIN=5 STEP=2 BATCH=64 LR=1e-3 SEED=0"
  "WIN=5 STEP=2 BATCH=64 LR=1e-4 SEED=0"
)

IDX="${SLURM_ARRAY_TASK_ID:-0}"
if [[ "$IDX" -ge "${#CONFIGS[@]}" ]]; then
  echo "Array index $IDX exceeds CONFIGS size ${#CONFIGS[@]}" >&2
  exit 1
fi

# shellcheck disable=SC2086
eval "${CONFIGS[$IDX]}"

: "${WIN:?WIN must be set in CONFIGS}"
: "${STEP:?STEP must be set in CONFIGS}"

BATCH="${BATCH:-$BATCH_SIZE_DEFAULT}"
LR="${LR:-$LR_DEFAULT}"
SEED="${SEED:-$SEED_DEFAULT}"
NUM_WORKERS="${NUM_WORKERS:-$NUM_WORKERS_DEFAULT}"

RUN_NAME="${BACKBONE}_ws${WIN}_fs${STEP}_wstr${WINDOW_STRIDE}_lr${LR}_seed${SEED}"
CKPT_DIR="${CHECKPOINT_ROOT}/${RUN_NAME}"
mkdir -p "$CKPT_DIR"

EXTRA_ARGS=()
EXTRA_ARGS+=(--backbone "$BACKBONE")
EXTRA_ARGS+=(--image-size "$IMAGE_SIZE")
EXTRA_ARGS+=(--temperature "$TEMPERATURE")
EXTRA_ARGS+=(--proj-dim "$PROJ_DIM")
EXTRA_ARGS+=(--weight-decay "$WEIGHT_DECAY")
EXTRA_ARGS+=(--max-grad-norm "$MAX_GRAD_NORM")

[[ -n "${SUB_BATCH_SIZE}" ]] && EXTRA_ARGS+=(--sub-batch-size "$SUB_BATCH_SIZE")
[[ -n "${MAX_BATCHES_PER_EPOCH}" ]] && EXTRA_ARGS+=(--max-batches-per-epoch "$MAX_BATCHES_PER_EPOCH")

[[ -n "${WINDOW_STRIDE}" ]] && EXTRA_ARGS+=(--window-stride "$WINDOW_STRIDE")
[[ -n "${MAX_WINDOWS_PER_CLIP}" ]] && EXTRA_ARGS+=(--max-windows-per-clip "$MAX_WINDOWS_PER_CLIP")
if [[ "${SINGLE_WINDOW_SHORT_CLIPS}" == "1" ]]; then
  EXTRA_ARGS+=(--single-window-short-clips --short-clip-window-threshold "$SHORT_CLIP_WINDOW_THRESHOLD")
else
  EXTRA_ARGS+=(--allow-multiple-short-windows)
fi
[[ "${USE_OBJECT_FOCUS}" == "1" ]] && EXTRA_ARGS+=(--use-object-focus)

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

if [[ "${ENABLE_CSV_LOG}" == "1" ]]; then
  EXTRA_ARGS+=(--csv-log --csv-log-path "${CKPT_DIR}/metrics.csv")
fi

echo "Starting run ${RUN_NAME}"
echo "Saving checkpoints to ${CKPT_DIR}"
echo "Backbone: ${BACKBONE} | NUM_WORKERS=${NUM_WORKERS}, BATCH=${BATCH}, LR=${LR}, TEMP=${TEMPERATURE}"

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
