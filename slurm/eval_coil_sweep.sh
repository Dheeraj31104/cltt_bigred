#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -A r00117
#SBATCH --job-name=coil_eval_sweep
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/coil_eval_%A_%a.out
#SBATCH --mail-user=dhkara@iu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --array=0-2

set -euo pipefail

# ---- Checkpoint dirs to evaluate (must match run_simclr_array.sh configs) ---
CHECKPOINT_DIRS=(
  "ws5_fs2_wstr10_lr1e-2_seed0"
  "ws5_fs2_wstr10_lr1e-3_seed0"
  "ws5_fs2_wstr10_lr1e-4_seed0"
)
# -----------------------------------------------------------------------------

IDX="${SLURM_ARRAY_TASK_ID:-0}"
if [[ "$IDX" -ge "${#CHECKPOINT_DIRS[@]}" ]]; then
  echo "Array index $IDX exceeds CHECKPOINT_DIRS size ${#CHECKPOINT_DIRS[@]}" >&2
  exit 1
fi

RUN_NAME="${CHECKPOINT_DIRS[$IDX]}"
CHECKPOINT_DIR="$PWD/checkpoints/${RUN_NAME}"
COIL_DIR="$PWD/data/coil-20/coil-20-proc"
OUTPUT_CSV="$PWD/results/coil_${RUN_NAME}.csv"

IMAGE_SIZE=224
BATCH_SIZE=64
EVAL_EPOCHS=20
LR=0.1
WEIGHT_DECAY=0.0
TRAIN_FRACTION=0.70
VAL_FRACTION=0.15
SPLIT_SEED=42
NUM_WORKERS=2
PROJ_DIM=128

mkdir -p logs results

: "${PS1:=}"
set +u
module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate torch-env
set -u

echo "[eval_coil_sweep] Job ${SLURM_JOB_ID} array task ${IDX} started on $(hostname)"
echo "[eval_coil_sweep] Checkpoint dir : ${CHECKPOINT_DIR}"
echo "[eval_coil_sweep] COIL dir        : ${COIL_DIR}"
echo "[eval_coil_sweep] Output CSV      : ${OUTPUT_CSV}"

srun python eval_coil_sweep.py \
    --checkpoint-dir  "$CHECKPOINT_DIR" \
    --coil-dir        "$COIL_DIR" \
    --output-csv      "$OUTPUT_CSV" \
    --image-size      "$IMAGE_SIZE" \
    --batch-size      "$BATCH_SIZE" \
    --eval-epochs     "$EVAL_EPOCHS" \
    --lr              "$LR" \
    --weight-decay    "$WEIGHT_DECAY" \
    --train-fraction  "$TRAIN_FRACTION" \
    --val-fraction    "$VAL_FRACTION" \
    --split-seed      "$SPLIT_SEED" \
    --num-workers     "$NUM_WORKERS" \
    --proj-dim        "$PROJ_DIM" \
    --random-init-baseline

echo "[eval_coil_sweep] Done."
