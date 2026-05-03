#!/usr/bin/env bash
#BSUB -J hpo_calib
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 04:00
#BSUB -oo hpc/logs/hpo/hpo_calibration_%J.out
#BSUB -eo hpc/logs/hpo/hpo_calibration_%J.err

set -euo pipefail

# Example submissions (use bsub -env to pass variables; inline VAR=val does not
# propagate to the batch environment on this cluster):
#
#   PINN architecture dry-run:
#     bsub -env "STUDY=pinn_architecture,DRY_RUN=true" < hpc/hpo/run_hpo_calibration.lsf.sh
#
#   Adam LR with selected architecture:
#     bsub -env "STUDY=adam_lr,PINN_HIDDEN_DIM=64,PINN_HIDDEN_LAYERS=3" \
#       < hpc/hpo/run_hpo_calibration.lsf.sh
#
#   Second-order LR subset (commas in values — use export + bsub -env all):
#     (export STUDY=second_order_lr OPTIMIZERS=LBFGS,SSBFGS LRS=0.1,0.3,1.0 && \
#       bsub -env "all" < hpc/hpo/run_hpo_calibration.lsf.sh)
#
#   Baseline architecture:
#     bsub -env "STUDY=baseline_architecture,DEVICE=cuda" < hpc/hpo/run_hpo_calibration.lsf.sh

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
STUDY="${STUDY:-pinn_architecture}"
REFERENCE_ID="${REFERENCE_ID:-}"
DATASET_ROOT="${DATASET_ROOT:-}"
MODEL_FLAG="${MODEL_FLAG:-SM4}"
SEED_LABELS="${SEED_LABELS:-s01}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float64}"
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

LSF_LOG_DIR="${REPO_ROOT}/hpc/logs/hpo"
mkdir -p "${LSF_LOG_DIR}"

cd "${REPO_ROOT}"
activate_repo_venv "${REPO_ROOT}"

echo "[hpc] repo_root=${REPO_ROOT}"
echo "[hpc] lsf_log_root=${LSF_LOG_DIR}"
echo "[hpc] queue=${QUEUE} (static #BSUB default: gpua100)"
echo "[hpc] walltime=${WALLTIME} (static #BSUB default: 04:00)"
echo "[hpc] mem_gb=${MEM_GB} (static #BSUB default: 8GB)"
echo "[hpc] n_cores=${N_CORES} (static #BSUB default: 4)"
echo "[hpc] study=${STUDY}"
echo "[hpc] reference_id=${REFERENCE_ID:-<launcher default>}"
echo "[hpc] dataset_root=${DATASET_ROOT:-<none>}"
echo "[hpc] model_flag=${MODEL_FLAG}"
echo "[hpc] seed_labels=${SEED_LABELS}"
echo "[hpc] device=${DEVICE}"
echo "[hpc] dtype=${DTYPE}"
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
  --dtype
  "${DTYPE}"
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
