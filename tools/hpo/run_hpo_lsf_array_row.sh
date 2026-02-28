#!/usr/bin/env bash
set -euo pipefail

: "${LSB_JOBINDEX:?LSB_JOBINDEX is required}"
: "${MATRIX_PATH:?MATRIX_PATH is required}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
ROW_INDEX=$((LSB_JOBINDEX - 1))

echo "[hpo] LSB_JOBINDEX=${LSB_JOBINDEX} -> ROW_INDEX=${ROW_INDEX}"
echo "[hpo] MATRIX_PATH=${MATRIX_PATH}"

"${PYTHON_BIN}" tools/hpo/run_hpo_matrix_row.py \
  --matrix "${MATRIX_PATH}" \
  --row-index "${ROW_INDEX}" \
  --python-bin "${PYTHON_BIN}"

