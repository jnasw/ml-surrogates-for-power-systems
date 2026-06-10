#!/usr/bin/env bash
#BSUB -J eval_data
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 04:00
#BSUB -oo hpc/logs/evaluation/evaluation_datasets_%J.out
#BSUB -eo hpc/logs/evaluation/evaluation_datasets_%J.err

set -euo pipefail

# Example submissions:
#   Dry-run SM4 OOD evaluation dataset:
#     EVALUATION_IDS=ood_SM4_wide_ic_b512_ds01 DRY_RUN=true bsub < hpc/generate_evaluation_datasets.lsf.sh
#   Generate SM4 OOD evaluation dataset:
#     EVALUATION_IDS=ood_SM4_wide_ic_b512_ds01 bsub < hpc/generate_evaluation_datasets.lsf.sh
#   Generate both SM4 ID and OOD evaluation datasets:
#     EVALUATION_IDS=id_SM4_lhs_b512_ds01,ood_SM4_wide_ic_b512_ds01 bsub < hpc/generate_evaluation_datasets.lsf.sh
#   Generate all OOD evaluation datasets for all models:
#     KIND=ood MODEL_FLAG=all bsub < hpc/generate_evaluation_datasets.lsf.sh
#   Force rebuild one evaluation dataset:
#     EVALUATION_IDS=ood_SM4_wide_ic_b512_ds01 FORCE_REBUILD=true bsub < hpc/generate_evaluation_datasets.lsf.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "${SCRIPT_DIR}/../src" ]]; then
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
else
  # When submitted via `bsub < script`, BASH_SOURCE may not point to this file.
  # LSF sets LSB_SUBCWD to the directory where `bsub` was called.
  REPO_ROOT="${REPO_ROOT:-${LSB_SUBCWD:-$(pwd)}}"
fi

source "${REPO_ROOT}/hpc/common/lsf_defaults.sh"

QUEUE="${QUEUE:-${QUEUE_CPU}}"
WALLTIME="${WALLTIME:-04:00}"
MEM_GB="${MEM_GB:-${DEFAULT_MEM_GB}}"
N_CORES="${N_CORES:-${DEFAULT_N_CORES}}"
EVALUATION_IDS="${EVALUATION_IDS:-${EVALUATION_ID:-}}"
KIND="${KIND:-}"
MODEL_FLAG="${MODEL_FLAG:-all}"
DRY_RUN="${DRY_RUN:-false}"
FORCE_REBUILD="${FORCE_REBUILD:-false}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/evaluation}"

LSF_LOG_DIR="${REPO_ROOT}/hpc/logs/evaluation"
mkdir -p "${LSF_LOG_DIR}"

cd "${REPO_ROOT}"
activate_repo_venv "${REPO_ROOT}"

echo "[hpc] repo_root=${REPO_ROOT}"
echo "[hpc] lsf_log_root=${LSF_LOG_DIR}"
echo "[hpc] queue=${QUEUE} (static #BSUB default: gpua100)"
echo "[hpc] walltime=${WALLTIME} (static #BSUB default: 04:00)"
echo "[hpc] mem_gb=${MEM_GB} (static #BSUB default: 8GB)"
echo "[hpc] n_cores=${N_CORES} (static #BSUB default: 4)"
echo "[hpc] evaluation_ids=${EVALUATION_IDS:-<none>}"
echo "[hpc] kind=${KIND:-<all>}"
echo "[hpc] model_flag=${MODEL_FLAG}"
echo "[hpc] dry_run=${DRY_RUN}"
echo "[hpc] force_rebuild=${FORCE_REBUILD}"
echo "[hpc] output_root=${OUTPUT_ROOT}"

cmd=(
  python3
  -m
  src.experiments.pipeline.run_evaluation_datasets
  --output-root
  "${OUTPUT_ROOT}"
)

if [[ -n "${EVALUATION_IDS}" ]]; then
  IFS=',' read -r -a eval_ids <<< "${EVALUATION_IDS}"
  for eval_id in "${eval_ids[@]}"; do
    eval_id="$(printf '%s' "${eval_id}" | xargs)"
    if [[ -n "${eval_id}" ]]; then
      cmd+=(--evaluation-id "${eval_id}")
    fi
  done
else
  cmd+=(--model-flag "${MODEL_FLAG}")
  if [[ -n "${KIND}" ]]; then
    cmd+=(--kind "${KIND}")
  fi
fi

if [[ "${DRY_RUN}" == "true" ]]; then
  cmd+=(--dry-run)
fi

if [[ "${FORCE_REBUILD}" == "true" ]]; then
  cmd+=(--force-rebuild)
fi

echo "[hpc] command:"
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
