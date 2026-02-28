#!/usr/bin/env bash
set -euo pipefail

HPO_CONFIG="${1:-src/config/hpo/qbc_deep_ensemble/smoke_stage0.yaml}"
QUEUE="${QUEUE:-gpua100}"
N_CORES="${N_CORES:-1}"
MEM_GB="${MEM_GB:-12}"
WALL_HOURS="${WALL_HOURS:-01}"
JOB_NAME="${JOB_NAME:-hpo-smoke}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"

TMP_ENV="$(mktemp)"
trap 'rm -f "${TMP_ENV}"' EXIT

"${PYTHON_BIN}" tools/hpo/build_hpo_matrix.py --config "${HPO_CONFIG}" --env-out "${TMP_ENV}" --no-timestamp
# shellcheck source=/dev/null
source "${TMP_ENV}"

if [[ "${TOTAL_ROWS}" -lt 1 ]]; then
  echo "[ERROR] Smoke matrix has no rows."
  exit 1
fi

mkdir -p outputs/lsf_logs

CMD="cd '${REPO_ROOT}' && '${PYTHON_BIN}' tools/hpo/run_hpo_matrix_row.py --matrix '${MATRIX_PATH}' --row-index 0 --python-bin '${PYTHON_BIN}'"

echo "[hpo-smoke] submitting one smoke row from ${MATRIX_PATH}"
echo "[hpo-smoke] command: ${CMD}"

bsub \
  -J "${JOB_NAME}" \
  -q "${QUEUE}" \
  -n "${N_CORES}" \
  -M "$((MEM_GB * 1024))" \
  -W "${WALL_HOURS}:00" \
  -oo "outputs/lsf_logs/${JOB_NAME}.%J.out" \
  -eo "outputs/lsf_logs/${JOB_NAME}.%J.err" \
  "${CMD}"
