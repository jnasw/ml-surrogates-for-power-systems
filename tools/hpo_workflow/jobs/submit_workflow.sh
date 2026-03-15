#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <workflow-config> [bsub args...]"
  echo "Example: $0 src/config/hpo_workflow/qbc_deep_ensemble/sm4.yaml"
  exit 1
fi

CONFIG_PATH="$1"
shift || true

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[ERROR] workflow config not found: ${CONFIG_PATH}"
  exit 1
fi

METHOD_NAME="$(basename "$(dirname "${CONFIG_PATH}")")"
MODEL_NAME="$(basename "${CONFIG_PATH}" .yaml)"
JOB_NAME="hpo_${METHOD_NAME}_${MODEL_NAME}"

if [[ -n "${HPO_FROM_STAGE:-}" ]]; then
  JOB_NAME="${JOB_NAME}_${HPO_FROM_STAGE}"
fi

echo "[hpo-submit] config=${CONFIG_PATH}"
echo "[hpo-submit] job_name=${JOB_NAME}"

bsub \
  -J "${JOB_NAME}" \
  -env "all, HPO_WORKFLOW_CONFIG=${CONFIG_PATH}${HPO_FROM_STAGE:+, HPO_FROM_STAGE=${HPO_FROM_STAGE}}${HPO_TO_STAGE:+, HPO_TO_STAGE=${HPO_TO_STAGE}}${HPO_RESUME:+, HPO_RESUME=${HPO_RESUME}}${HPO_NO_TIMESTAMP:+, HPO_NO_TIMESTAMP=${HPO_NO_TIMESTAMP}}${HPO_PLAN_ONLY:+, HPO_PLAN_ONLY=${HPO_PLAN_ONLY}}${HPO_MAX_ROWS:+, HPO_MAX_ROWS=${HPO_MAX_ROWS}}${HPO_OUTPUT_ROOT:+, HPO_OUTPUT_ROOT=${HPO_OUTPUT_ROOT}}${PYTHON_BIN:+, PYTHON_BIN=${PYTHON_BIN}}${HPO_PRUNE_ROW_RAW_DATA:+, HPO_PRUNE_ROW_RAW_DATA=${HPO_PRUNE_ROW_RAW_DATA}}${HPO_PRUNE_ROW_QBC_ARTIFACTS:+, HPO_PRUNE_ROW_QBC_ARTIFACTS=${HPO_PRUNE_ROW_QBC_ARTIFACTS}}" \
  "$@" \
  < tools/hpo_workflow/jobs/run_hpo_workflow.lsf.sh
