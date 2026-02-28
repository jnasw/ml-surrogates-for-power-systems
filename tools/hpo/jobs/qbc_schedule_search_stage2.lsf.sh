#!/usr/bin/env bash
#BSUB -J hpo_qbc_stage2_schedule
#BSUB -q gpua100
#BSUB -n 1
#BSUB -W 10:00
#BSUB -R "rusage[mem=24GB]"
#BSUB -oo outputs/lsf_logs/hpo_qbc_stage2_schedule.%J.out
#BSUB -eo outputs/lsf_logs/hpo_qbc_stage2_schedule.%J.err

set -euo pipefail

REPO_ROOT="${LSB_SUBCWD:-$PWD}"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
HPO_CONFIG="${HPO_CONFIG:-src/config/hpo/qbc_deep_ensemble/schedule_search_stage2.yaml}"

mkdir -p outputs/lsf_logs

TMP_ENV="$(mktemp)"
trap 'rm -f "${TMP_ENV}"' EXIT

"${PYTHON_BIN}" tools/hpo/build_hpo_matrix.py \
  --config "${HPO_CONFIG}" \
  --env-out "${TMP_ENV}"

# shellcheck source=/dev/null
source "${TMP_ENV}"

if [[ "${TOTAL_ROWS}" -lt 1 ]]; then
  echo "[ERROR] stage2 matrix has no rows"
  exit 1
fi

echo "[hpo] running ${TOTAL_ROWS} rows sequentially"

for ((ROW=0; ROW< TOTAL_ROWS; ROW++)); do
  echo "[hpo] row ${ROW}/${TOTAL_ROWS}"
  "${PYTHON_BIN}" tools/hpo/run_hpo_matrix_row.py \
    --matrix "${MATRIX_PATH}" \
    --row-index "${ROW}" \
    --python-bin "${PYTHON_BIN}"
done

