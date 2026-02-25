#!/usr/bin/env bash
set -euo pipefail

# Small local matrix for quick functionality checks.
METHODS=("lhs_static" "qbc_deep_ensemble" "marker_directed" "qbc_marker_hybrid")
BUDGET="b256"
SEEDS=("s01" "s02")

for method in "${METHODS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo "[RUN] method=${method} budget=${BUDGET} seed=${seed}"
    python run_experiment.py \
      --method "${method}" \
      --budget "${BUDGET}" \
      --seed "${seed}" \
      --phase main \
      --experiment-id thesis_sm4_local_smoke \
      --baseline-epochs 60
  done
done
