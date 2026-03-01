#!/usr/bin/env bash
#BSUB -J hpo_qbc_smoke
#BSUB -q gpua100
#BSUB -n 1
#BSUB -W 01:00
#BSUB -R "rusage[mem=4GB]"
#BSUB -oo outputs/lsf_logs/hpo_qbc_smoke.%J.out
#BSUB -eo outputs/lsf_logs/hpo_qbc_smoke.%J.err

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
HPO_CONFIG="${HPO_CONFIG:-src/config/hpo/qbc_deep_ensemble/smoke_stage0.yaml}"

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
