#!/usr/bin/env bash
#BSUB -J hpo_workflow
#BSUB -q gpua100
#BSUB -n 1
#BSUB -W 24:00
#BSUB -R "rusage[mem=12GB]"
#BSUB -oo outputs/lsf_logs/hpo_workflow.%J.out
#BSUB -eo outputs/lsf_logs/hpo_workflow.%J.err

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

if [[ -z "${HPO_WORKFLOW_CONFIG:-}" ]]; then
  echo "[ERROR] HPO_WORKFLOW_CONFIG is required."
  echo "[ERROR] Example:"
  echo "  export HPO_WORKFLOW_CONFIG=src/config/hpo_workflow/qbc_deep_ensemble/sm4.yaml"
  echo "  bsub < tools/hpo_workflow/jobs/run_hpo_workflow.lsf.sh"
  exit 1
fi

if [[ ! -f "${HPO_WORKFLOW_CONFIG}" ]]; then
  echo "[ERROR] workflow config not found: ${HPO_WORKFLOW_CONFIG}"
  exit 1
fi

if ! "${PYTHON_BIN}" -c "import omegaconf" >/dev/null 2>&1; then
  echo "[ERROR] omegaconf is missing for interpreter: ${PYTHON_BIN}"
  echo "[ERROR] submit with PYTHON_BIN=/path/to/venv/bin/python bsub < tools/hpo_workflow/jobs/run_hpo_workflow.lsf.sh"
  exit 1
fi

mkdir -p outputs/lsf_logs

CMD=(
  "${PYTHON_BIN}"
  tools/hpo_workflow/run_hpo_workflow.py
  --config "${HPO_WORKFLOW_CONFIG}"
)

if [[ -n "${HPO_OUTPUT_ROOT:-}" ]]; then
  CMD+=(--output-root "${HPO_OUTPUT_ROOT}")
fi
if [[ "${HPO_NO_TIMESTAMP:-false}" == "true" ]]; then
  CMD+=(--no-timestamp)
fi
if [[ "${HPO_SKIP_CONFIRM:-false}" == "true" && -z "${HPO_TO_STAGE:-}" ]]; then
  HPO_TO_STAGE="refine"
fi
if [[ -n "${HPO_FROM_STAGE:-}" ]]; then
  CMD+=(--from-stage "${HPO_FROM_STAGE}")
fi
if [[ -n "${HPO_TO_STAGE:-}" ]]; then
  CMD+=(--to-stage "${HPO_TO_STAGE}")
fi
if [[ "${HPO_RESUME:-false}" == "true" ]]; then
  CMD+=(--resume)
fi
if [[ -n "${HPO_MAX_ROWS:-}" ]]; then
  CMD+=(--max-rows "${HPO_MAX_ROWS}")
fi
if [[ "${HPO_PLAN_ONLY:-false}" == "true" ]]; then
  CMD+=(--plan-only)
fi
if [[ "${HPO_PRUNE_ROW_RAW_DATA:-true}" != "true" ]]; then
  CMD+=(--no-prune-row-data)
fi
if [[ "${HPO_PRUNE_ROW_QBC_ARTIFACTS:-true}" != "true" ]]; then
  CMD+=(--no-prune-row-qbc-artifacts)
fi

echo "[hpo-workflow] config=${HPO_WORKFLOW_CONFIG}"
echo "[hpo-workflow] python=${PYTHON_BIN}"
echo "[hpo-workflow] command=${CMD[*]}"

"${CMD[@]}"
