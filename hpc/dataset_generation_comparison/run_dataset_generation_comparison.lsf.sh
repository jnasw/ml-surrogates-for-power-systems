#!/usr/bin/env bash
#BSUB -J data_cmp
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 24:00
#BSUB -oo hpc/logs/dataset_generation/dataset_generation_comparison_%J.out
#BSUB -eo hpc/logs/dataset_generation/dataset_generation_comparison_%J.err

set -euo pipefail

# Example submissions (use bsub -env to pass variables; inline VAR=val does not
# propagate to the batch environment on this cluster):
#
#   Smoke dry-run (pipeline check, no execution):
#     bsub -env "MODE=smoke,DRY_RUN=true" \
#       < hpc/dataset_generation_comparison/run_dataset_generation_comparison.lsf.sh
#
#   Smoke run (lhs_static b256 ds01 bs01, 1 epoch):
#     bsub -env "MODE=smoke,MODEL_FLAG=SM4" \
#       < hpc/dataset_generation_comparison/run_dataset_generation_comparison.lsf.sh
#
#   Final run, lhs_static subset (values with commas require export + bsub -env all):
#     (export CAMPAIGN_TAG=dataset_final_sm4 MODE=final MODEL_FLAG=SM4 METHODS=lhs_static \
#       BUDGETS=b256,b512 DATASET_SEEDS=ds01 BASELINE_SEEDS=bs01,bs02 && \
#       bsub -env "all" < hpc/dataset_generation_comparison/run_dataset_generation_comparison.lsf.sh)
#
#   Final run, one method per job (no commas in values — short form works):
#     bsub -env "CAMPAIGN_TAG=dataset_pinn_sm4,MODE=final,MODEL_FLAG=SM4,DOWNSTREAM_MODEL=pinn_data_only,METHODS=lhs_static" \
#       -J data_lhs_pinn < hpc/dataset_generation_comparison/run_dataset_generation_comparison.lsf.sh
#
#   Final run with storage cleanup:
#     bsub -env "MODE=final,MODEL_FLAG=SM4,METHODS=lhs_static,BUDGETS=b256,CLEANUP_DATA=true" \
#       < hpc/dataset_generation_comparison/run_dataset_generation_comparison.lsf.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "${SCRIPT_DIR}/../../src" ]]; then
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
else
  # When submitted via `bsub < script`, BASH_SOURCE may not resolve correctly.
  # LSF sets LSB_SUBCWD to the directory where `bsub` was called.
  REPO_ROOT="${REPO_ROOT:-${LSB_SUBCWD:-$(pwd)}}"
fi

source "${REPO_ROOT}/hpc/common/lsf_defaults.sh"

QUEUE="${QUEUE:-${QUEUE_CPU}}"
WALLTIME="${WALLTIME:-24:00}"
MEM_GB="${MEM_GB:-${DEFAULT_MEM_GB}}"
N_CORES="${N_CORES:-${DEFAULT_N_CORES}}"
MODE="${MODE:-smoke}"
CAMPAIGN_TAG="${CAMPAIGN_TAG:-}"
SHARD_LABEL="${SHARD_LABEL:-}"
METHODS="${METHODS:-}"
BUDGETS="${BUDGETS:-}"
DATASET_SEEDS="${DATASET_SEEDS:-}"
BASELINE_SEEDS="${BASELINE_SEEDS:-}"
MODEL_FLAG="${MODEL_FLAG:-SM4}"
DOWNSTREAM_MODEL="${DOWNSTREAM_MODEL:-}"
DEVICE="${DEVICE:-}"
DTYPE="${DTYPE:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
ADAM_LR="${ADAM_LR:-}"
ID_EVAL_ID="${ID_EVAL_ID:-}"
OOD_EVAL_ID="${OOD_EVAL_ID:-}"
NO_ID_EVAL="${NO_ID_EVAL:-false}"
NO_OOD_EVAL="${NO_OOD_EVAL:-false}"
BASELINE_EPOCHS="${BASELINE_EPOCHS:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
CLEANUP_DATA="${CLEANUP_DATA:-false}"
DRY_RUN="${DRY_RUN:-false}"

LSF_LOG_DIR="${REPO_ROOT}/hpc/logs/dataset_generation"
mkdir -p "${LSF_LOG_DIR}"

cd "${REPO_ROOT}"
activate_repo_venv "${REPO_ROOT}"

echo "[hpc] repo_root=${REPO_ROOT}"
echo "[hpc] lsf_log_root=${LSF_LOG_DIR}"
echo "[hpc] queue=${QUEUE} (static #BSUB default: gpua100)"
echo "[hpc] walltime=${WALLTIME} (static #BSUB default: 24:00)"
echo "[hpc] mem_gb=${MEM_GB} (static #BSUB default: 8GB)"
echo "[hpc] n_cores=${N_CORES} (static #BSUB default: 4)"
echo "[hpc] mode=${MODE}"
echo "[hpc] campaign_tag=${CAMPAIGN_TAG:-<none>}"
echo "[hpc] shard_label=${SHARD_LABEL:-<auto>}"
echo "[hpc] methods=${METHODS:-<mode default>}"
echo "[hpc] budgets=${BUDGETS:-<mode default>}"
echo "[hpc] dataset_seeds=${DATASET_SEEDS:-<mode default>}"
echo "[hpc] baseline_seeds=${BASELINE_SEEDS:-<mode default>}"
echo "[hpc] model_flag=${MODEL_FLAG}"
echo "[hpc] downstream_model=${DOWNSTREAM_MODEL:-<launcher default>}"
echo "[hpc] device=${DEVICE:-<launcher default>}"
echo "[hpc] dtype=${DTYPE:-<launcher default>}"
echo "[hpc] batch_size=${BATCH_SIZE:-<launcher default>}"
echo "[hpc] adam_lr=${ADAM_LR:-<launcher default>}"
echo "[hpc] id_eval_id=${ID_EVAL_ID:-<model-aware default>}"
echo "[hpc] ood_eval_id=${OOD_EVAL_ID:-<model-aware default>}"
echo "[hpc] no_id_eval=${NO_ID_EVAL}"
echo "[hpc] no_ood_eval=${NO_OOD_EVAL}"
echo "[hpc] baseline_epochs=${BASELINE_EPOCHS:-<mode default>}"
echo "[hpc] output_root=${OUTPUT_ROOT:-<auto>}"
echo "[hpc] cleanup_data=${CLEANUP_DATA}"
echo "[hpc] dry_run=${DRY_RUN}"

cmd=(
  python3
  -m
  src.experiments.pipeline.run_dataset_generation_comparison
  --mode
  "${MODE}"
  --model-flag
  "${MODEL_FLAG}"
)

if [[ -n "${CAMPAIGN_TAG}" ]]; then
  cmd+=(--campaign-tag "${CAMPAIGN_TAG}")
fi

if [[ -n "${SHARD_LABEL}" ]]; then
  cmd+=(--shard-label "${SHARD_LABEL}")
fi

if [[ -n "${METHODS}" ]]; then
  cmd+=(--methods "${METHODS}")
fi

if [[ -n "${BUDGETS}" ]]; then
  cmd+=(--budgets "${BUDGETS}")
fi

if [[ -n "${DATASET_SEEDS}" ]]; then
  cmd+=(--dataset-seeds "${DATASET_SEEDS}")
fi

if [[ -n "${BASELINE_SEEDS}" ]]; then
  cmd+=(--baseline-seeds "${BASELINE_SEEDS}")
fi

if [[ -n "${DOWNSTREAM_MODEL}" ]]; then
  cmd+=(--downstream-model "${DOWNSTREAM_MODEL}")
fi

if [[ -n "${DEVICE}" ]]; then
  cmd+=(--device "${DEVICE}")
fi

if [[ -n "${DTYPE}" ]]; then
  cmd+=(--dtype "${DTYPE}")
fi

if [[ -n "${BATCH_SIZE}" ]]; then
  cmd+=(--batch-size "${BATCH_SIZE}")
fi

if [[ -n "${ADAM_LR}" ]]; then
  cmd+=(--adam-lr "${ADAM_LR}")
fi

if [[ -n "${ID_EVAL_ID}" ]]; then
  cmd+=(--id-eval-id "${ID_EVAL_ID}")
fi

if [[ -n "${OOD_EVAL_ID}" ]]; then
  cmd+=(--ood-eval-id "${OOD_EVAL_ID}")
fi

if [[ "${NO_ID_EVAL}" == "true" ]]; then
  cmd+=(--no-id-eval)
fi

if [[ "${NO_OOD_EVAL}" == "true" ]]; then
  cmd+=(--no-ood-eval)
fi

if [[ -n "${BASELINE_EPOCHS}" ]]; then
  cmd+=(--baseline-epochs "${BASELINE_EPOCHS}")
fi

if [[ -n "${OUTPUT_ROOT}" ]]; then
  cmd+=(--output-root "${OUTPUT_ROOT}")
fi

if [[ "${CLEANUP_DATA}" == "true" ]]; then
  cmd+=(--cleanup-data)
fi

if [[ "${DRY_RUN}" == "true" ]]; then
  cmd+=(--dry-run)
fi

echo "[hpc] command:"
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
