#!/usr/bin/env bash
#BSUB -J hpo_marker_stage1_policy
#BSUB -q gpua100
#BSUB -n 1
#BSUB -W 12:00
#BSUB -R "rusage[mem=24GB]"
#BSUB -oo outputs/lsf_logs/hpo_marker_stage1_policy.%J.out
#BSUB -eo outputs/lsf_logs/hpo_marker_stage1_policy.%J.err

set -euo pipefail

REPO_ROOT="${LSB_SUBCWD:-$PWD}"
cd "${REPO_ROOT}"

if [[ -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${REPO_ROOT}/.venv/bin/activate"
fi

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi
HPO_CONFIG="${HPO_CONFIG:-src/config/hpo/marker_directed/policy_search_stage1.yaml}"
PRUNE_ROW_RAW_DATA="${PRUNE_ROW_RAW_DATA:-true}"

if ! "${PYTHON_BIN}" -c "import omegaconf" >/dev/null 2>&1; then
  echo "[ERROR] omegaconf is missing for interpreter: ${PYTHON_BIN}"
  echo "[ERROR] submit with PYTHON_BIN=/path/to/venv/bin/python bsub < ...lsf.sh"
  exit 1
fi

mkdir -p outputs/lsf_logs

TMP_ENV="$(mktemp)"
trap 'rm -f "${TMP_ENV}"' EXIT

"${PYTHON_BIN}" tools/hpo/build_hpo_matrix.py \
  --config "${HPO_CONFIG}" \
  --env-out "${TMP_ENV}"

# shellcheck source=/dev/null
source "${TMP_ENV}"

if [[ "${TOTAL_ROWS}" -lt 1 ]]; then
  echo "[ERROR] stage1 matrix has no rows"
  exit 1
fi

echo "[hpo] running ${TOTAL_ROWS} rows sequentially"

for ((ROW=0; ROW< TOTAL_ROWS; ROW++)); do
  echo "[hpo] row ${ROW}/${TOTAL_ROWS}"
  "${PYTHON_BIN}" tools/hpo/run_hpo_matrix_row.py \
    --matrix "${MATRIX_PATH}" \
    --row-index "${ROW}" \
    --python-bin "${PYTHON_BIN}"
  if [[ "${PRUNE_ROW_RAW_DATA}" == "true" ]]; then
    RUN_ROOT="$("${PYTHON_BIN}" - "${MATRIX_PATH}" "${ROW}" <<'PY'
import csv
import sys

matrix_path = sys.argv[1]
row_idx = int(sys.argv[2])
with open(matrix_path, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))
print(rows[row_idx]["run_root"])
PY
)"
    if [[ -n "${RUN_ROOT}" ]]; then
      rm -rf "${RUN_ROOT}/data"
      echo "[hpo] pruned raw data for row ${ROW}: ${RUN_ROOT}/data"
    fi
  fi
done
