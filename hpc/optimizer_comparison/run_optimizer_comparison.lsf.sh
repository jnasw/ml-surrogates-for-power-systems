#!/usr/bin/env bash
#BSUB -J optimizer_cmp
#BSUB -q gpu
#BSUB -n 4
#BSUB -R "rusage[mem=8192]"
#BSUB -W 04:00
#BSUB -oo hpc/logs/optimizer_comparison_%J.out
#BSUB -eo hpc/logs/optimizer_comparison_%J.err

set -euo pipefail

# Example submissions:
#   Screening smoke run with default reference dataset:
#     export MODE=screening
#     bsub < hpc/optimizer_comparison/run_optimizer_comparison.lsf.sh
#   Custom strategies:
#     export MODE=screening STRATEGIES=adam,lbfgs,adam_lbfgs SEED_LABELS=s01
#     bsub < hpc/optimizer_comparison/run_optimizer_comparison.lsf.sh
#   Dry-run:
#     export MODE=screening DRY_RUN=true STRATEGIES=adam SEED_LABELS=s01
#     bsub < hpc/optimizer_comparison/run_optimizer_comparison.lsf.sh
#   Final run:
#     export MODE=final REFERENCE_ID=main_SM4_qbc_b512_ds01
#     bsub < hpc/optimizer_comparison/run_optimizer_comparison.lsf.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODE="${MODE:-screening}"
REFERENCE_ID="${REFERENCE_ID:-}"
STRATEGIES="${STRATEGIES:-}"
SEED_LABELS="${SEED_LABELS:-}"
DEVICE="${DEVICE:-cuda}"
WANDB_PROJECT="${WANDB_PROJECT:-}"
DRY_RUN="${DRY_RUN:-false}"

mkdir -p "${REPO_ROOT}/hpc/logs"

cd "${REPO_ROOT}"

echo "[hpc] repo_root=${REPO_ROOT}"
echo "[hpc] mode=${MODE}"
echo "[hpc] reference_id=${REFERENCE_ID:-<launcher default>}"
echo "[hpc] strategies=${STRATEGIES:-<mode default>}"
echo "[hpc] seed_labels=${SEED_LABELS:-<mode default>}"
echo "[hpc] device=${DEVICE}"
echo "[hpc] wandb_project=${WANDB_PROJECT:-<mode default>}"
echo "[hpc] dry_run=${DRY_RUN}"

cmd=(
  python3
  -m
  src.experiments.pipeline.run_optimizer_comparison
  --mode
  "${MODE}"
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
