#!/bin/bash
# Example Slurm job array to sweep several SimCLR runs.
# Edit the SBATCH lines to match your cluster defaults before submitting.

#SBATCH -p gpu-debug
#SBATCH --gpus=1
#SBATCH -A r00117
#SBATCH --job-name=simclr_ego
#SBATCH --output=logs/simclr_%A_%a.out
#SBATCH --mail-user=dhkara@iu.edu
#SBATCH --mail-type=BEGIN,END,REQUEUE
#SBATCH --requeue
#SBATCH --signal=B:USR1@120

set -euo pipefail

# ---- User settings (edit these) ---------------------------------------------
ROOT_DIR="/N/project/cogai/dhkara/cropped_clips"          # folder with extracted frame folders
CHECKPOINT_ROOT="$PWD/checkpoints"      # base output dir for checkpoints
EPOCHS=100
SUB_BATCH_SIZE=""                      # set to "" to disable grad accumulation
MAX_BATCHES_PER_EPOCH=""               # set to e.g. 500 to cap per-epoch work
TWO_VIEW_OFFSET=1                     # set to integer for paired-window mode
USE_OBJECT_FOCUS=1                     # set to 1 to enable background blur
ENABLE_WANDB=1                         # set to 1 to enable wandb logging
WANDB_PROJECT="simclr-ego-sruns"
WANDB_ENTITY=""                        # optional
WANDB_MODE="online"                    # online|offline|disabled

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

RUN_NAME="ws${WIN}_step${STEP}_lr${LR}_seed${SEED}"
CKPT_DIR="${CHECKPOINT_ROOT}/${RUN_NAME}"
mkdir -p "$CKPT_DIR"

# Build optional args cleanly
EXTRA_ARGS=()
[[ -n "${SUB_BATCH_SIZE}" ]] && EXTRA_ARGS+=(--sub-batch-size "$SUB_BATCH_SIZE")
[[ -n "${MAX_BATCHES_PER_EPOCH}" ]] && EXTRA_ARGS+=(--max-batches-per-epoch "$MAX_BATCHES_PER_EPOCH")
[[ -n "${TWO_VIEW_OFFSET}" ]] && EXTRA_ARGS+=(--two-view-offset "$TWO_VIEW_OFFSET")
[[ "${USE_OBJECT_FOCUS}" == "1" ]] && EXTRA_ARGS+=(--use-object-focus)
if [[ "${ENABLE_WANDB}" == "1" ]]; then
  EXTRA_ARGS+=(--wandb --wandb-project "$WANDB_PROJECT" --wandb-mode "$WANDB_MODE")
  [[ -n "${WANDB_ENTITY}" ]] && EXTRA_ARGS+=(--wandb-entity "$WANDB_ENTITY")
  EXTRA_ARGS+=(--wandb-name "${RUN_NAME}")
fi

echo "Starting run ${RUN_NAME}"
echo "Saving checkpoints to ${CKPT_DIR}"

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
  --num-workers "${SLURM_CPUS_PER_TASK:-4}" \
  "${EXTRA_ARGS[@]}"
