#!/usr/bin/env bash
#BSUB -J ref_data
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 04:00
#BSUB -oo hpc/logs/reference/reference_datasets_%J.out
#BSUB -eo hpc/logs/reference/reference_datasets_%J.err

set -euo pipefail

# Example submissions:
#   Dry-run all SM4 references:
#     MODEL_FLAG=SM4 SUITE=all DRY_RUN=true bsub < hpc/reference_datasets/generate_reference_datasets.lsf.sh
#   Generate smoke only:
#     MODEL_FLAG=SM4 SUITE=smoke DRY_RUN=false bsub < hpc/reference_datasets/generate_reference_datasets.lsf.sh
#   Generate main only:
#     MODEL_FLAG=SM4 SUITE=main DRY_RUN=false bsub < hpc/reference_datasets/generate_reference_datasets.lsf.sh
#   Generate one specific reference ID:
#     REFERENCE_IDS=main_SM4_qbc_b512_ds01 bsub < hpc/reference_datasets/generate_reference_datasets.lsf.sh
#   Force rebuild one reference ID:
#     REFERENCE_IDS=dev_SM4_lhs_b512_ds01 FORCE_REBUILD=true bsub < hpc/reference_datasets/generate_reference_datasets.lsf.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "${SCRIPT_DIR}/../../src" ]]; then
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
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
MODEL_FLAG="${MODEL_FLAG:-SM4}"
SUITE="${SUITE:-all}"
REFERENCE_IDS="${REFERENCE_IDS:-}"
DRY_RUN="${DRY_RUN:-false}"
FORCE_REBUILD="${FORCE_REBUILD:-false}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/reference}"

LSF_LOG_DIR="${REPO_ROOT}/hpc/logs/reference"
mkdir -p "${LSF_LOG_DIR}"

cd "${REPO_ROOT}"
activate_repo_venv "${REPO_ROOT}"

echo "[hpc] repo_root=${REPO_ROOT}"
echo "[hpc] lsf_log_root=${LSF_LOG_DIR}"
echo "[hpc] queue=${QUEUE} (static #BSUB default: gpua100)"
echo "[hpc] walltime=${WALLTIME} (static #BSUB default: 04:00)"
echo "[hpc] mem_gb=${MEM_GB} (static #BSUB default: 8GB)"
echo "[hpc] n_cores=${N_CORES} (static #BSUB default: 4)"
echo "[hpc] model_flag=${MODEL_FLAG}"
echo "[hpc] suite=${SUITE}"
echo "[hpc] reference_ids=${REFERENCE_IDS:-<none>}"
echo "[hpc] dry_run=${DRY_RUN}"
echo "[hpc] force_rebuild=${FORCE_REBUILD}"
echo "[hpc] output_root=${OUTPUT_ROOT}"

cmd=(
  python3
  -m
  src.experiments.pipeline.run_reference_datasets
  --output-root
  "${OUTPUT_ROOT}"
)

if [[ -n "${REFERENCE_IDS}" ]]; then
  IFS=',' read -r -a ref_ids <<< "${REFERENCE_IDS}"
  for ref_id in "${ref_ids[@]}"; do
    ref_id="$(printf '%s' "${ref_id}" | xargs)"
    if [[ -n "${ref_id}" ]]; then
      cmd+=(--reference-id "${ref_id}")
    fi
  done
else
  cmd+=(--suite "${SUITE}" --model-flag "${MODEL_FLAG}")
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
