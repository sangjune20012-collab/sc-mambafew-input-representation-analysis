#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/test.sh <SHOT> <TRANSFORM> <RESOLUTION> <WEIGHT_PATH> [NOISE_DB]
# Clean example: bash scripts/test.sh 1 SP 64 checkpoints/cwru_1shot_64_SP/best.pth
# Noise example: bash scripts/test.sh 1 SC 64 checkpoints/cwru_1shot_64_SC/best.pth 10

SHOT=${1:?"SHOT is required: 1 or 5"}
TRANSFORM=${2:?"TRANSFORM is required: SP/SC/GS/RP/MTF/GAF"}
RESOLUTION=${3:?"RESOLUTION is required: 64 or 128"}
WEIGHT=${4:?"WEIGHT_PATH is required"}
NOISE_DB=${5:-""}
DATA_PATH=${DATA_PATH:-"./CWRU"}

if [ "$SHOT" -eq 1 ]; then
  SAMPLES=30
else
  SAMPLES=150
fi

CMD=(python test.py
  --dataset CWRU
  --data_path "$DATA_PATH"
  --shot_num "$SHOT"
  --training_samples_CWRU "$SAMPLES"
  --transform "$TRANSFORM"
  --resolution "$RESOLUTION"
  --weight "$WEIGHT"
  --episode_num_test 75
  --test_iter 5
  --output_dir outputs
)

if [ -n "$NOISE_DB" ]; then
  CMD+=(--noise_db "$NOISE_DB")
fi

"${CMD[@]}"
