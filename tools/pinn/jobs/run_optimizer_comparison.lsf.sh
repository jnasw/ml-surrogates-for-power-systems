#!/usr/bin/env bash
#BSUB -J optimizer_comparison
#BSUB -q gpua100
#BSUB -n 1
#BSUB -W 24:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -oo outputs/lsf_logs/optimizer_comparison.%J.out
#BSUB -eo outputs/lsf_logs/optimizer_comparison.%J.err

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

if ! "${PYTHON_BIN}" -c "import omegaconf" >/dev/null 2>&1; then
  echo "[ERROR] omegaconf is missing for interpreter: ${PYTHON_BIN}"
  echo "[ERROR] submit with PYTHON_BIN=/path/to/venv/bin/python bsub < tools/pinn/jobs/run_optimizer_comparison.lsf.sh"
  exit 1
fi

mkdir -p outputs/lsf_logs

STAMP="$(date +%Y%m%d_%H%M%S)"
PROFILE="${PROFILE:-benchmark}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-optimizer_comparison_${PROFILE}_${STAMP}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/pinn_hpc_experiments/${EXPERIMENT_TAG}}"
MODEL_FLAG="${MODEL_FLAG:-SM4}"
PRESET="${PRESET:-default}"
PINN_BUDGET="${PINN_BUDGET:-b1024}"
DATASET_SEED="${DATASET_SEED:-s01}"
PINN_DEVICE="${PINN_DEVICE:-cuda}"
PINN_HIDDEN_DIM="${PINN_HIDDEN_DIM:-64}"
PINN_HIDDEN_LAYERS="${PINN_HIDDEN_LAYERS:-4}"
PINN_ACTIVATION="${PINN_ACTIVATION:-tanh}"
PINN_DTYPE="${PINN_DTYPE:-float64}"
PINN_BATCH_SIZE="${PINN_BATCH_SIZE:-1024}"
ADAM_LR="${ADAM_LR:-1e-3}"
QUASI_NEWTON_LR="${QUASI_NEWTON_LR:-1.0}"
LINE_SEARCH="${LINE_SEARCH:-strong_wolfe}"
OPTIMIZERS="${OPTIMIZERS:-LBFGS,BFGS,SSBFGS,SSBroyden}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-}"
QUASI_NEWTON_EPOCHS="${QUASI_NEWTON_EPOCHS:-}"
WANDB_PROJECT="${WANDB_PROJECT:-sm-surrogates-pinn-optimizer-comparison}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
LOG_EVERY_EPOCH="${LOG_EVERY_EPOCH:-1}"
LOSS_WEIGHT_DATA="${LOSS_WEIGHT_DATA:-1.0}"
LOSS_WEIGHT_DT="${LOSS_WEIGHT_DT:-1.0e-4}"
LOSS_WEIGHT_PHYSICS="${LOSS_WEIGHT_PHYSICS:-1.0e-4}"
LOSS_WEIGHT_IC="${LOSS_WEIGHT_IC:-1.0e-3}"
DATASET_ROOT="${DATASET_ROOT:-}"
DRY_RUN="${DRY_RUN:-false}"

CMD=(
  "${PYTHON_BIN}"
  tools/pinn/run_optimizer_comparison.py
  --profile "${PROFILE}"
  --experiment-tag "${EXPERIMENT_TAG}"
  --output-root "${OUTPUT_ROOT}"
  --model-flag "${MODEL_FLAG}"
  --preset "${PRESET}"
  --budget "${PINN_BUDGET}"
  --dataset-seed "${DATASET_SEED}"
  --device "${PINN_DEVICE}"
  --hidden-dim "${PINN_HIDDEN_DIM}"
  --hidden-layers "${PINN_HIDDEN_LAYERS}"
  --activation "${PINN_ACTIVATION}"
  --dtype "${PINN_DTYPE}"
  --batch-size "${PINN_BATCH_SIZE}"
  --adam-lr "${ADAM_LR}"
  --quasi-newton-lr "${QUASI_NEWTON_LR}"
  --line-search "${LINE_SEARCH}"
  --optimizers "${OPTIMIZERS}"
  --wandb-project "${WANDB_PROJECT}"
  --log-every-epoch "${LOG_EVERY_EPOCH}"
  --loss-weight-data "${LOSS_WEIGHT_DATA}"
  --loss-weight-dt "${LOSS_WEIGHT_DT}"
  --loss-weight-physics "${LOSS_WEIGHT_PHYSICS}"
  --loss-weight-ic "${LOSS_WEIGHT_IC}"
)

if [[ -n "${WARMUP_EPOCHS}" ]]; then
  CMD+=(--warmup-epochs "${WARMUP_EPOCHS}")
fi
if [[ -n "${QUASI_NEWTON_EPOCHS}" ]]; then
  CMD+=(--quasi-newton-epochs "${QUASI_NEWTON_EPOCHS}")
fi
if [[ -n "${WANDB_ENTITY}" ]]; then
  CMD+=(--wandb-entity "${WANDB_ENTITY}")
fi
if [[ -n "${DATASET_ROOT}" ]]; then
  CMD+=(--dataset-root "${DATASET_ROOT}")
fi
if [[ "${DRY_RUN}" == "true" ]]; then
  CMD+=(--dry-run)
fi

echo "[optimizer-comparison] repo_root=${REPO_ROOT}"
echo "[optimizer-comparison] output_root=${OUTPUT_ROOT}"
echo "[optimizer-comparison] profile=${PROFILE}"
echo "[optimizer-comparison] model_flag=${MODEL_FLAG}"
echo "[optimizer-comparison] budget=${PINN_BUDGET}"
echo "[optimizer-comparison] dataset_seed=${DATASET_SEED}"
echo "[optimizer-comparison] optimizers=${OPTIMIZERS}"
if [[ -n "${WARMUP_EPOCHS}" ]]; then
  echo "[optimizer-comparison] warmup_epochs=${WARMUP_EPOCHS}"
fi
if [[ -n "${QUASI_NEWTON_EPOCHS}" ]]; then
  echo "[optimizer-comparison] quasi_newton_epochs=${QUASI_NEWTON_EPOCHS}"
fi
if [[ -n "${DATASET_ROOT}" ]]; then
  echo "[optimizer-comparison] dataset_root=${DATASET_ROOT}"
fi
echo "[optimizer-comparison] command=${CMD[*]}"

"${CMD[@]}"
