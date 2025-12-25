#!/bin/bash
# Example Slurm job array to sweep several SimCLR runs.
# Edit the SBATCH lines to match your cluster defaults before submitting.

#SBATCH --job-name=simclr_ego
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/simclr_%A_%a.out

set -euo pipefail

# ---- User settings (edit these) ---------------------------------------------
ROOT_DIR="/path/to/Images_ego"          # folder with extracted frame folders
CHECKPOINT_ROOT="$PWD/checkpoints"      # base output dir for checkpoints
EPOCHS=25
SUB_BATCH_SIZE=64                      # set to "" to disable grad accumulation
MAX_BATCHES_PER_EPOCH=""               # set to e.g. 500 to cap per-epoch work
TWO_VIEW_OFFSET=""                     # set to integer for paired-window mode
USE_OBJECT_FOCUS=0                     # set to 1 to enable background blur
ENABLE_WANDB=0                         # set to 1 to enable wandb logging
WANDB_PROJECT="simclr-ego"
WANDB_ENTITY=""                        # optional
WANDB_MODE="online"                    # online|offline|disabled

# Optional env bootstrap (module/conda/etc). Uncomment and edit for your cluster.
# module load cuda/12.1
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate your-env

# Each entry becomes one array task. Tweak as needed.
CONFIGS=(
  "WIN=5 STEP=10 BATCH=256 LR=1e-3 SEED=0"
  "WIN=5 STEP=10 BATCH=256 LR=5e-4 SEED=1"
  "WIN=7 STEP=15 BATCH=128 LR=1e-3 SEED=2"
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
  --num-workers "${SLURM_CPUS_PER_TASK:-4}" \
  "${EXTRA_ARGS[@]}"
