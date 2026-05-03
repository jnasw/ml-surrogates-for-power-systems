#!/usr/bin/env bash
#BSUB -J opt_cmp
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 04:00
#BSUB -oo hpc/logs/optimizer/optimizer_comparison_%J.out
#BSUB -eo hpc/logs/optimizer/optimizer_comparison_%J.err

set -euo pipefail

# Example submissions (use bsub -env to pass variables; inline VAR=val does not
# propagate to the batch environment on this cluster):
#
#   Screening dry-run:
#     bsub -env "MODE=screening,DRY_RUN=true,STRATEGIES=adam,SEED_LABELS=s01" \
#       < hpc/optimizer_comparison/run_optimizer_comparison.lsf.sh
#
#   Custom strategies (commas in values — use export + bsub -env all):
#     (export MODE=screening STRATEGIES=adam,lbfgs,adam_lbfgs SEED_LABELS=s01 && \
#       bsub -env "all" < hpc/optimizer_comparison/run_optimizer_comparison.lsf.sh)
#
#   Final run:
#     bsub -env "MODE=final,REFERENCE_ID=main_SM4_qbc_b512_ds01" \
#       < hpc/optimizer_comparison/run_optimizer_comparison.lsf.sh
#
#   Final run with explicit epoch budget:
#     bsub -env "MODE=final,TOTAL_EPOCHS=5000,ADAM_WARMUP_EPOCHS=500,REFERENCE_ID=main_SM4_qbc_b512_ds01" \
#       < hpc/optimizer_comparison/run_optimizer_comparison.lsf.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "${SCRIPT_DIR}/../../src" ]]; then
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
else
  # When submitted via `bsub < script`, BASH_SOURCE may not point to this file.
  # LSF sets LSB_SUBCWD to the directory where `bsub` was called.
  REPO_ROOT="${REPO_ROOT:-${LSB_SUBCWD:-$(pwd)}}"
fi

source "${REPO_ROOT}/hpc/common/lsf_defaults.sh"

QUEUE="${QUEUE:-${QUEUE_GPU}}"
WALLTIME="${WALLTIME:-04:00}"
MEM_GB="${MEM_GB:-${DEFAULT_MEM_GB}}"
N_CORES="${N_CORES:-${DEFAULT_N_CORES}}"
MODE="${MODE:-screening}"
REFERENCE_ID="${REFERENCE_ID:-}"
MODEL_FLAG="${MODEL_FLAG:-SM4}"
STRATEGIES="${STRATEGIES:-}"
SEED_LABELS="${SEED_LABELS:-}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-}"
MAIN_EPOCHS="${MAIN_EPOCHS:-}"
ADAM_WARMUP_EPOCHS="${ADAM_WARMUP_EPOCHS:-}"
ADAM_LR="${ADAM_LR:-}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-}"
ID_EVAL_ID="${ID_EVAL_ID:-}"
OOD_EVAL_ID="${OOD_EVAL_ID:-}"
NO_OOD_EVAL="${NO_OOD_EVAL:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-}"
DRY_RUN="${DRY_RUN:-false}"

LSF_LOG_DIR="${REPO_ROOT}/hpc/logs/optimizer"
mkdir -p "${LSF_LOG_DIR}"

cd "${REPO_ROOT}"
activate_repo_venv "${REPO_ROOT}"

echo "[hpc] repo_root=${REPO_ROOT}"
echo "[hpc] lsf_log_root=${LSF_LOG_DIR}"
echo "[hpc] queue=${QUEUE} (static #BSUB default: gpua100)"
echo "[hpc] walltime=${WALLTIME} (static #BSUB default: 04:00)"
echo "[hpc] mem_gb=${MEM_GB} (static #BSUB default: 8GB)"
echo "[hpc] n_cores=${N_CORES} (static #BSUB default: 4)"
echo "[hpc] mode=${MODE}"
echo "[hpc] reference_id=${REFERENCE_ID:-<launcher default>}"
echo "[hpc] model_flag=${MODEL_FLAG}"
echo "[hpc] strategies=${STRATEGIES:-<mode default>}"
echo "[hpc] seed_labels=${SEED_LABELS:-<mode default>}"
echo "[hpc] total_epochs=${TOTAL_EPOCHS:-<mode default>}"
echo "[hpc] main_epochs=${MAIN_EPOCHS:-<launcher computed from total/warmup>}"
echo "[hpc] adam_warmup_epochs=${ADAM_WARMUP_EPOCHS:-<mode default>}"
echo "[hpc] adam_lr=${ADAM_LR:-<launcher default>}"
echo "[hpc] device=${DEVICE}"
echo "[hpc] dtype=${DTYPE:-<launcher default>}"
echo "[hpc] id_eval_id=${ID_EVAL_ID:-<none>}"
echo "[hpc] ood_eval_id=${OOD_EVAL_ID:-<launcher default>}"
echo "[hpc] no_ood_eval=${NO_OOD_EVAL}"
echo "[hpc] wandb_project=${WANDB_PROJECT:-<mode default>}"
echo "[hpc] dry_run=${DRY_RUN}"

cmd=(
  python3
  -m
  src.experiments.pipeline.run_optimizer_comparison
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

if [[ -n "${STRATEGIES}" ]]; then
  cmd+=(--strategies "${STRATEGIES}")
fi

if [[ -n "${SEED_LABELS}" ]]; then
  cmd+=(--seed-labels "${SEED_LABELS}")
fi

if [[ -n "${TOTAL_EPOCHS}" ]]; then
  cmd+=(--total-epochs "${TOTAL_EPOCHS}")
fi

if [[ -n "${MAIN_EPOCHS}" ]]; then
  cmd+=(--main-epochs "${MAIN_EPOCHS}")
fi

if [[ -n "${ADAM_WARMUP_EPOCHS}" ]]; then
  cmd+=(--adam-warmup-epochs "${ADAM_WARMUP_EPOCHS}")
fi

if [[ -n "${ADAM_LR}" ]]; then
  cmd+=(--adam-lr "${ADAM_LR}")
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
