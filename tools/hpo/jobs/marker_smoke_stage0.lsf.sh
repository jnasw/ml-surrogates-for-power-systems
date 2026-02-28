#!/usr/bin/env bash
#BSUB -J hpo_marker_smoke
#BSUB -q gpua100
#BSUB -n 1
#BSUB -W 01:00
#BSUB -R "rusage[mem=12GB]"
#BSUB -oo outputs/lsf_logs/hpo_marker_smoke.%J.out
#BSUB -eo outputs/lsf_logs/hpo_marker_smoke.%J.err

set -euo pipefail

REPO_ROOT="${LSB_SUBCWD:-$PWD}"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
HPO_CONFIG="${HPO_CONFIG:-src/config/hpo/marker_directed/smoke_stage0.yaml}"

mkdir -p outputs/lsf_logs

TMP_ENV="$(mktemp)"
trap 'rm -f "${TMP_ENV}"' EXIT

"${PYTHON_BIN}" tools/hpo/build_hpo_matrix.py \
  --config "${HPO_CONFIG}" \
  --env-out "${TMP_ENV}" \
  --no-timestamp

# shellcheck source=/dev/null
source "${TMP_ENV}"

if [[ "${TOTAL_ROWS}" -lt 1 ]]; then
  echo "[ERROR] smoke matrix has no rows"
  exit 1
fi

"${PYTHON_BIN}" tools/hpo/run_hpo_matrix_row.py \
  --matrix "${MATRIX_PATH}" \
  --row-index 0 \
  --python-bin "${PYTHON_BIN}"

