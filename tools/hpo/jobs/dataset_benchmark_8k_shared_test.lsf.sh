#!/usr/bin/env bash
#BSUB -J dataset_benchmark_8k_shared_test
#BSUB -q gpua100
#BSUB -n 1
#BSUB -W 24:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -oo outputs/lsf_logs/dataset_benchmark_8k_shared_test.%J.out
#BSUB -eo outputs/lsf_logs/dataset_benchmark_8k_shared_test.%J.err

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

CAMPAIGN_CONFIG="${CAMPAIGN_CONFIG:-src/config/campaign/benchmark_sm4_8k.yaml}"
METRICS_OUT="${METRICS_OUT:-}"
CAMPAIGN_DRY_RUN="${CAMPAIGN_DRY_RUN:-false}"

if ! "${PYTHON_BIN}" -c "import omegaconf" >/dev/null 2>&1; then
  echo "[ERROR] omegaconf is missing for interpreter: ${PYTHON_BIN}"
  echo "[ERROR] submit with PYTHON_BIN=/path/to/venv/bin/python bsub < ...lsf.sh"
  exit 1
fi

mkdir -p outputs/lsf_logs

TMP_LOG="$(mktemp)"
trap 'rm -f "${TMP_LOG}"' EXIT

CAMPAIGN_CMD=(
  "${PYTHON_BIN}" tools/pipeline/run_campaign.py
  --config "${CAMPAIGN_CONFIG}"
)
if [[ "${CAMPAIGN_DRY_RUN}" == "true" ]]; then
  CAMPAIGN_CMD+=(--dry-run)
fi

"${CAMPAIGN_CMD[@]}" | tee "${TMP_LOG}"

CAMPAIGN_MANIFEST="$(grep -E "^\[campaign\] Manifest:" "${TMP_LOG}" | tail -n1 | awk '{print $3}')"
if [[ -z "${CAMPAIGN_MANIFEST}" ]]; then
  echo "[ERROR] Could not determine campaign manifest path from campaign output."
  exit 1
fi

if [[ -z "${METRICS_OUT}" ]]; then
  METRICS_OUT="$(dirname "${CAMPAIGN_MANIFEST}")/dataset_benchmark_metrics.csv"
fi

"${PYTHON_BIN}" tools/analysis/export_dataset_benchmark_metrics.py \
  --campaign-manifest "${CAMPAIGN_MANIFEST}" \
  --out "${METRICS_OUT}"

echo "[dataset_benchmark] campaign_manifest=${CAMPAIGN_MANIFEST}"
echo "[dataset_benchmark] metrics_csv=${METRICS_OUT}"
