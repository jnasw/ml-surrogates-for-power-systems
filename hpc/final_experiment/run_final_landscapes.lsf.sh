#!/usr/bin/env bash
#BSUB -J final_land
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 04:00
#BSUB -oo hpc/logs/final/final_landscapes_%J.out
#BSUB -eo hpc/logs/final/final_landscapes_%J.err

set -euo pipefail

# Example:
#   bsub -env "RUN_MANIFEST=outputs/pinn/final_experiment/<tag>/run_manifest.json,MODELS=SM4,SEED_LABELS=s01,DRY_RUN=true" \
#     < hpc/final_experiment/run_final_landscapes.lsf.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "${SCRIPT_DIR}/../../src" ]]; then
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
else
  REPO_ROOT="${REPO_ROOT:-${LSB_SUBCWD:-$(pwd)}}"
fi

source "${REPO_ROOT}/hpc/common/lsf_defaults.sh"

RUN_MANIFEST="${RUN_MANIFEST:-}"
RUN_MANIFESTS="${RUN_MANIFESTS:-}"
MODELS="${MODELS:-}"
STRATEGIES="${STRATEGIES:-}"
SEED_LABELS="${SEED_LABELS:-}"
CHECKPOINT_TAGS="${CHECKPOINT_TAGS:-}"
INCLUDE_DN_BOUNDARY="${INCLUDE_DN_BOUNDARY:-true}"
GRID="${GRID:-}"
RESOLUTION="${RESOLUTION:-}"
SUPERVISED_ROWS="${SUPERVISED_ROWS:-}"
COLLOCATION_ROWS="${COLLOCATION_ROWS:-}"
INIT_ROWS="${INIT_ROWS:-}"
ANALYSIS_SEED="${ANALYSIS_SEED:-}"
DIRECTION_SEED="${DIRECTION_SEED:-}"
WEIGHT_SOURCE="${WEIGHT_SOURCE:-}"
DEVICE="${DEVICE:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-}"
DRY_RUN="${DRY_RUN:-false}"

if [[ -z "${RUN_MANIFEST}" && -z "${RUN_MANIFESTS}" ]]; then
  echo "[hpc] ERROR: RUN_MANIFEST or RUN_MANIFESTS is required." >&2
  exit 2
fi

LSF_LOG_DIR="${REPO_ROOT}/hpc/logs/final"
mkdir -p "${LSF_LOG_DIR}"

cd "${REPO_ROOT}"
activate_repo_venv "${REPO_ROOT}"

echo "[hpc] run_manifest=${RUN_MANIFEST:-<unset>}"
echo "[hpc] run_manifests=${RUN_MANIFESTS:-<unset>}"
echo "[hpc] models=${MODELS:-<launcher default>}"
echo "[hpc] strategies=${STRATEGIES:-<launcher default>}"
echo "[hpc] seed_labels=${SEED_LABELS:-<launcher default>}"
echo "[hpc] checkpoint_tags=${CHECKPOINT_TAGS:-<launcher default>}"
echo "[hpc] include_dn_boundary=${INCLUDE_DN_BOUNDARY}"
echo "[hpc] output_root=${OUTPUT_ROOT:-<launcher default>}"
echo "[hpc] experiment_tag=${EXPERIMENT_TAG:-<timestamp>}"
echo "[hpc] dry_run=${DRY_RUN}"

cmd=(
  python3
  -m
  src.experiments.pipeline.run_final_landscapes
)

if [[ -n "${RUN_MANIFESTS}" ]]; then
  IFS=',' read -r -a manifest_items <<< "${RUN_MANIFESTS}"
  for manifest_path in "${manifest_items[@]}"; do
    if [[ -n "${manifest_path}" ]]; then
      cmd+=(--run-manifest "${manifest_path}")
    fi
  done
else
  cmd+=(--run-manifest "${RUN_MANIFEST}")
fi

if [[ -n "${MODELS}" ]]; then cmd+=(--models "${MODELS}"); fi
if [[ -n "${STRATEGIES}" ]]; then cmd+=(--strategies "${STRATEGIES}"); fi
if [[ -n "${SEED_LABELS}" ]]; then cmd+=(--seed-labels "${SEED_LABELS}"); fi
if [[ -n "${CHECKPOINT_TAGS}" ]]; then cmd+=(--checkpoint-tags "${CHECKPOINT_TAGS}"); fi
if [[ -n "${GRID}" ]]; then cmd+=(--grid "${GRID}"); fi
if [[ -n "${RESOLUTION}" ]]; then cmd+=(--resolution "${RESOLUTION}"); fi
if [[ -n "${SUPERVISED_ROWS}" ]]; then cmd+=(--supervised-rows "${SUPERVISED_ROWS}"); fi
if [[ -n "${COLLOCATION_ROWS}" ]]; then cmd+=(--collocation-rows "${COLLOCATION_ROWS}"); fi
if [[ -n "${INIT_ROWS}" ]]; then cmd+=(--init-rows "${INIT_ROWS}"); fi
if [[ -n "${ANALYSIS_SEED}" ]]; then cmd+=(--analysis-seed "${ANALYSIS_SEED}"); fi
if [[ -n "${DIRECTION_SEED}" ]]; then cmd+=(--direction-seed "${DIRECTION_SEED}"); fi
if [[ -n "${WEIGHT_SOURCE}" ]]; then cmd+=(--weight-source "${WEIGHT_SOURCE}"); fi
if [[ -n "${DEVICE}" ]]; then cmd+=(--device "${DEVICE}"); fi
if [[ -n "${OUTPUT_ROOT}" ]]; then cmd+=(--output-root "${OUTPUT_ROOT}"); fi
if [[ -n "${EXPERIMENT_TAG}" ]]; then cmd+=(--experiment-tag "${EXPERIMENT_TAG}"); fi

if [[ "${INCLUDE_DN_BOUNDARY}" == "true" ]]; then
  cmd+=(--include-dn-boundary)
else
  cmd+=(--no-include-dn-boundary)
fi

if [[ "${DRY_RUN}" == "true" ]]; then
  cmd+=(--dry-run)
fi

echo "[hpc] command:"
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
