#!/usr/bin/env bash
#BSUB -J hpo_calibration
#BSUB -q gpu
#BSUB -n 4
#BSUB -R "rusage[mem=8192]"
#BSUB -W 04:00
#BSUB -oo hpc/logs/hpo_calibration_%J.out
#BSUB -eo hpc/logs/hpo_calibration_%J.err

set -euo pipefail

# Example submissions:
#   PINN architecture dry-run:
#     export STUDY=pinn_architecture DRY_RUN=true
#     bsub < hpc/hpo/run_hpo_calibration.lsf.sh
#   Adam LR with selected architecture:
#     export STUDY=adam_lr PINN_HIDDEN_DIM=64 PINN_HIDDEN_LAYERS=4
#     bsub < hpc/hpo/run_hpo_calibration.lsf.sh
#   Second-order LR subset:
#     export STUDY=second_order_lr OPTIMIZERS=LBFGS,SSBFGS LRS=0.1,0.3,1.0
#     bsub < hpc/hpo/run_hpo_calibration.lsf.sh
#   Baseline architecture:
#     export STUDY=baseline_architecture DEVICE=cuda
#     bsub < hpc/hpo/run_hpo_calibration.lsf.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

STUDY="${STUDY:-pinn_architecture}"
REFERENCE_ID="${REFERENCE_ID:-}"
DATASET_ROOT="${DATASET_ROOT:-}"
MODEL_FLAG="${MODEL_FLAG:-SM4}"
SEED_LABELS="${SEED_LABELS:-s01}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-}"
WANDB_PROJECT="${WANDB_PROJECT:-}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_TAGS="${WANDB_TAGS:-}"
DRY_RUN="${DRY_RUN:-false}"
HIDDEN_DIMS="${HIDDEN_DIMS:-}"
HIDDEN_LAYERS="${HIDDEN_LAYERS:-}"
LRS="${LRS:-}"
OPTIMIZERS="${OPTIMIZERS:-}"
PINN_HIDDEN_DIM="${PINN_HIDDEN_DIM:-}"
PINN_HIDDEN_LAYERS="${PINN_HIDDEN_LAYERS:-}"

mkdir -p "${REPO_ROOT}/hpc/logs"

cd "${REPO_ROOT}"

echo "[hpc] repo_root=${REPO_ROOT}"
echo "[hpc] study=${STUDY}"
echo "[hpc] reference_id=${REFERENCE_ID:-<launcher default>}"
echo "[hpc] dataset_root=${DATASET_ROOT:-<none>}"
echo "[hpc] model_flag=${MODEL_FLAG}"
echo "[hpc] seed_labels=${SEED_LABELS}"
echo "[hpc] device=${DEVICE}"
echo "[hpc] epochs=${EPOCHS:-<study default>}"
echo "[hpc] dry_run=${DRY_RUN}"

cmd=(
  python3
  -m
  src.experiments.pipeline.run_hpo_calibration
  --study
  "${STUDY}"
  --model-flag
  "${MODEL_FLAG}"
  --seed-labels
  "${SEED_LABELS}"
  --device
  "${DEVICE}"
)

if [[ -n "${REFERENCE_ID}" ]]; then
  cmd+=(--reference-id "${REFERENCE_ID}")
fi

if [[ -n "${DATASET_ROOT}" ]]; then
  cmd+=(--dataset-root "${DATASET_ROOT}")
fi

if [[ -n "${EPOCHS}" ]]; then
  cmd+=(--epochs "${EPOCHS}")
fi

if [[ -n "${WANDB_PROJECT}" ]]; then
  cmd+=(--wandb-project "${WANDB_PROJECT}")
fi

if [[ -n "${WANDB_ENTITY}" ]]; then
  cmd+=(--wandb-entity "${WANDB_ENTITY}")
fi

if [[ -n "${WANDB_TAGS}" ]]; then
  cmd+=(--wandb-tags "${WANDB_TAGS}")
fi

if [[ -n "${HIDDEN_DIMS}" ]]; then
  cmd+=(--hidden-dims "${HIDDEN_DIMS}")
fi

if [[ -n "${HIDDEN_LAYERS}" ]]; then
  cmd+=(--hidden-layers "${HIDDEN_LAYERS}")
fi

if [[ -n "${LRS}" ]]; then
  cmd+=(--lrs "${LRS}")
fi

if [[ -n "${OPTIMIZERS}" ]]; then
  cmd+=(--optimizers "${OPTIMIZERS}")
fi

if [[ -n "${PINN_HIDDEN_DIM}" ]]; then
  cmd+=(--pinn-hidden-dim "${PINN_HIDDEN_DIM}")
fi

if [[ -n "${PINN_HIDDEN_LAYERS}" ]]; then
  cmd+=(--pinn-hidden-layers "${PINN_HIDDEN_LAYERS}")
fi

if [[ "${DRY_RUN}" == "true" ]]; then
  cmd+=(--dry-run)
fi

echo "[hpc] command:"
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
