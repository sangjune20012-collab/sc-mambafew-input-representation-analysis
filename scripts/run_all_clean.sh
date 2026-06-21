#!/usr/bin/env bash
set -euo pipefail

# Train all clean-data conditions used in the paper-style comparison.
# Usage: bash scripts/run_all_clean.sh

for SHOT in 1 5; do
  for RES in 64 128; do
    for TRANSFORM in SP SC GS RP MTF GAF; do
      bash scripts/train.sh "$SHOT" "$TRANSFORM" "$RES" 10
    done
  done
done
