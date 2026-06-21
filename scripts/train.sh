#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/train.sh <SHOT> <TRANSFORM> <RESOLUTION> [EPOCHS]
# Example: bash scripts/train.sh 1 SP 64 10
#          bash scripts/train.sh 5 GS 128 10

SHOT=${1:?"SHOT is required: 1 or 5"}
TRANSFORM=${2:?"TRANSFORM is required: SP/SC/GS/RP/MTF/GAF"}
RESOLUTION=${3:?"RESOLUTION is required: 64 or 128"}
EPOCHS=${4:-10}
DATA_PATH=${DATA_PATH:-"./CWRU"}

if [ "$SHOT" -eq 1 ]; then
  SAMPLES=30
else
  SAMPLES=150
fi

python train.py \
  --dataset CWRU \
  --data_path "$DATA_PATH" \
  --shot_num "$SHOT" \
  --training_samples_CWRU "$SAMPLES" \
  --transform "$TRANSFORM" \
  --resolution "$RESOLUTION" \
  --num_epochs "$EPOCHS" \
  --episode_num_train 100 \
  --episode_num_test 75 \
  --output_dir outputs \
  --checkpoint_dir checkpoints
