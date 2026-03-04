#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -A r00117
#SBATCH --job-name=coil_eval_sweep
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/coil_eval_%j.out
#SBATCH --mail-user=dhkara@iu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

# ---- Settings (edit these) --------------------------------------------------
CHECKPOINT_DIR="$PWD/checkpoints/ws5_fs2_wstr10"
COIL_DIR="$PWD/data/coil-20/coil-20-proc"
OUTPUT_CSV="$PWD/results/coil_sweep.csv"

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
# -----------------------------------------------------------------------------

mkdir -p logs results

: "${PS1:=}"
set +u
module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate torch-env
set -u

echo "[eval_coil_sweep] Job ${SLURM_JOB_ID} started on $(hostname)"
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
    --proj-dim        "$PROJ_DIM"

echo "[eval_coil_sweep] Done."
