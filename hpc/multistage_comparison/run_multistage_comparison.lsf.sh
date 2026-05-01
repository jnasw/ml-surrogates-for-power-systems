#!/usr/bin/env bash
#BSUB -J multi_cmp
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 16:00
#BSUB -oo hpc/logs/multistage/multistage_comparison_%J.out
#BSUB -eo hpc/logs/multistage/multistage_comparison_%J.err

set -euo pipefail

# Example submissions:
#
#   Dry-run screening on smoke reference dataset:
#     MODE=screening REFERENCE_ID=smoke_SM4_lhs_b256_ds01 DEVICE=cpu DRY_RUN=true \
#       bsub < hpc/multistage_comparison/run_multistage_comparison.lsf.sh
#
#   Screening run with a strategy subset:
#     MODE=screening REFERENCE_ID=smoke_SM4_lhs_b256_ds01 \
#       STRATEGIES=adam_30000,adam_ssbroyden_2stage SEED_LABELS=s01 DEVICE=cpu \
#       bsub < hpc/multistage_comparison/run_multistage_comparison.lsf.sh
#
#   Final run with all strategies and 5 seeds:
#     MODE=final REFERENCE_ID=main_SM4_qbc_b512_ds01 \
#       STRATEGIES=adam_30000,adam_ssbroyden_2stage,adam_ssbroyden_3stage,adam_ssbroyden_4stage,adam_ssbroyden_5stage \
#       SEED_LABELS=s01,s02,s03,s04,s05 \
#       bsub < hpc/multistage_comparison/run_multistage_comparison.lsf.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "${SCRIPT_DIR}/../../src" ]]; then
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
else
  # When submitted via `bsub < script`, BASH_SOURCE may not resolve correctly.
  # LSF sets LSB_SUBCWD to the directory where `bsub` was called.
  REPO_ROOT="${REPO_ROOT:-${LSB_SUBCWD:-$(pwd)}}"
fi

source "${REPO_ROOT}/hpc/common/lsf_defaults.sh"

QUEUE="${QUEUE:-${QUEUE_GPU}}"
WALLTIME="${WALLTIME:-16:00}"
MEM_GB="${MEM_GB:-${DEFAULT_MEM_GB}}"
N_CORES="${N_CORES:-${DEFAULT_N_CORES}}"
MODE="${MODE:-screening}"
REFERENCE_ID="${REFERENCE_ID:-}"
DATASET_ROOT="${DATASET_ROOT:-}"
MODEL_FLAG="${MODEL_FLAG:-SM4}"
SEED_LABELS="${SEED_LABELS:-}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-}"
ID_EVAL_ID="${ID_EVAL_ID:-}"
OOD_EVAL_ID="${OOD_EVAL_ID:-}"
NO_OOD_EVAL="${NO_OOD_EVAL:-false}"
STRATEGIES="${STRATEGIES:-}"
EPOCHS="${EPOCHS:-}"
WANDB_PROJECT="${WANDB_PROJECT:-}"
DRY_RUN="${DRY_RUN:-false}"

LSF_LOG_DIR="${REPO_ROOT}/hpc/logs/multistage"
mkdir -p "${LSF_LOG_DIR}"

cd "${REPO_ROOT}"
activate_repo_venv "${REPO_ROOT}"

echo "[hpc] repo_root=${REPO_ROOT}"
echo "[hpc] lsf_log_root=${LSF_LOG_DIR}"
echo "[hpc] queue=${QUEUE} (static #BSUB default: gpua100)"
echo "[hpc] walltime=${WALLTIME} (static #BSUB default: 16:00)"
echo "[hpc] mem_gb=${MEM_GB} (static #BSUB default: 8GB)"
echo "[hpc] n_cores=${N_CORES} (static #BSUB default: 4)"
echo "[hpc] mode=${MODE}"
echo "[hpc] reference_id=${REFERENCE_ID:-<launcher default>}"
echo "[hpc] dataset_root=${DATASET_ROOT:-<not set>}"
echo "[hpc] model_flag=${MODEL_FLAG}"
echo "[hpc] seed_labels=${SEED_LABELS:-<mode default>}"
echo "[hpc] device=${DEVICE}"
echo "[hpc] dtype=${DTYPE:-<launcher default>}"
echo "[hpc] id_eval_id=${ID_EVAL_ID:-<none>}"
echo "[hpc] ood_eval_id=${OOD_EVAL_ID:-<launcher default>}"
echo "[hpc] no_ood_eval=${NO_OOD_EVAL}"
echo "[hpc] strategies=${STRATEGIES:-<all strategies>}"
echo "[hpc] epochs=${EPOCHS:-<mode default>}"
echo "[hpc] wandb_project=${WANDB_PROJECT:-<mode default>}"
echo "[hpc] dry_run=${DRY_RUN}"

cmd=(
  python3
  -m
  src.experiments.pipeline.run_multistage_comparison
  --mode
  "${MODE}"
  --model-flag
  "${MODEL_FLAG}"
  --device
  "${DEVICE}"
)

if [[ -n "${REFERENCE_ID}" ]]; then
  cmd+=(--reference-id "${REFERENCE_ID}")
fi

if [[ -n "${DATASET_ROOT}" ]]; then
  cmd+=(--dataset-root "${DATASET_ROOT}")
fi

if [[ -n "${SEED_LABELS}" ]]; then
  cmd+=(--seed-labels "${SEED_LABELS}")
fi

if [[ -n "${DTYPE}" ]]; then
  cmd+=(--dtype "${DTYPE}")
fi

if [[ -n "${ID_EVAL_ID}" ]]; then
  cmd+=(--id-eval-id "${ID_EVAL_ID}")
fi

if [[ -n "${OOD_EVAL_ID}" ]]; then
  cmd+=(--ood-eval-id "${OOD_EVAL_ID}")
fi

if [[ "${NO_OOD_EVAL}" == "true" ]]; then
  cmd+=(--no-ood-eval)
fi

if [[ -n "${STRATEGIES}" ]]; then
  cmd+=(--strategies "${STRATEGIES}")
fi

if [[ -n "${EPOCHS}" ]]; then
  cmd+=(--epochs "${EPOCHS}")
fi

if [[ -n "${WANDB_PROJECT}" ]]; then
  cmd+=(--wandb-project "${WANDB_PROJECT}")
fi

if [[ "${DRY_RUN}" == "true" ]]; then
  cmd+=(--dry-run)
fi

echo "[hpc] command:"
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
