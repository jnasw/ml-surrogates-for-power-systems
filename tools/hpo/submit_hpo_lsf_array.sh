#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash tools/hpo/submit_hpo_lsf_array.sh <hpo-config-yaml>"
  exit 1
fi

HPO_CONFIG="$1"
QUEUE="${QUEUE:-gpua100}"
N_CORES="${N_CORES:-1}"
MEM_GB="${MEM_GB:-24}"
WALL_HOURS="${WALL_HOURS:-08}"
JOB_NAME="${JOB_NAME:-hpo-qbc}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"

TMP_ENV="$(mktemp)"
trap 'rm -f "${TMP_ENV}"' EXIT

"${PYTHON_BIN}" tools/hpo/build_hpo_matrix.py --config "${HPO_CONFIG}" --env-out "${TMP_ENV}"
# shellcheck source=/dev/null
source "${TMP_ENV}"

if [[ -z "${TOTAL_ROWS:-}" || "${TOTAL_ROWS}" -le 0 ]]; then
  echo "[ERROR] TOTAL_ROWS invalid after matrix build."
  exit 1
fi

if [[ ! -f "${MATRIX_PATH}" ]]; then
  echo "[ERROR] Matrix file not found: ${MATRIX_PATH}"
  exit 1
fi

echo "[hpo] HPO_ROOT=${HPO_ROOT}"
echo "[hpo] MATRIX_PATH=${MATRIX_PATH}"
echo "[hpo] TOTAL_ROWS=${TOTAL_ROWS}"

SUBMIT_CMD=(
  bsub
  -J "${JOB_NAME}[1-${TOTAL_ROWS}]"
  -q "${QUEUE}"
  -n "${N_CORES}"
  -M "$((MEM_GB * 1024))"
  -W "${WALL_HOURS}:00"
  -oo "outputs/lsf_logs/${JOB_NAME}.%J.%I.out"
  -eo "outputs/lsf_logs/${JOB_NAME}.%J.%I.err"
  "env MATRIX_PATH='${MATRIX_PATH}' PYTHON_BIN='${PYTHON_BIN}' bash tools/hpo/run_hpo_lsf_array_row.sh"
)

mkdir -p outputs/lsf_logs

echo "[hpo] submitting:"
printf ' %q' "${SUBMIT_CMD[@]}"
echo

"${SUBMIT_CMD[@]}"
