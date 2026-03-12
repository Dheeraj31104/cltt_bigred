#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -A r00117
#SBATCH --job-name=imagenet_eval
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/imagenet_eval_%j.out
#SBATCH --mail-user=dhkara@iu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

CKPT_DIR="${CKPT_DIR:-checkpoints/ws5_fs2_wstr10_lr1e-3_seed0}"
DATA_DIR="${DATA_DIR:-$PWD/data/tiny-imagenet-200}"
IMAGE_SIZE="${IMAGE_SIZE:-64}"
EVAL_EPOCHS="${EVAL_EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-0.1}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
TRAIN_FRACTION="${TRAIN_FRACTION:-0.9}"
NUM_WORKERS="${SLURM_CPUS_PER_TASK:-8}"

: "${PS1:=}"
set +u
module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate torch-env
set -u

echo "Checkpoint dir : $CKPT_DIR"
echo "Data dir       : $DATA_DIR"

srun python eval_imagenet.py \
  --checkpoint-dir "$CKPT_DIR" \
  --data-dir "$DATA_DIR" \
  --image-size "$IMAGE_SIZE" \
  --eval-epochs "$EVAL_EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --train-fraction "$TRAIN_FRACTION" \
  --num-workers "$NUM_WORKERS"
