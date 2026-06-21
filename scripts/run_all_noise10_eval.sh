#!/usr/bin/env bash
set -euo pipefail

# Evaluate top transforms under Gaussian noise 10 dB.
# This assumes clean checkpoints have already been trained.

for SHOT in 1 5; do
  for RES in 64 128; do
    for TRANSFORM in SP SC GS; do
      WEIGHT="checkpoints/cwru_${SHOT}shot_${RES}_${TRANSFORM}/best.pth"
      bash scripts/test.sh "$SHOT" "$TRANSFORM" "$RES" "$WEIGHT" 10
    done
  done
done
