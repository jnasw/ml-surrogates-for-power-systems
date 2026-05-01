#!/usr/bin/env bash
#BSUB -J pinn_qbc_small
#BSUB -q gpua100
#BSUB -n 1
#BSUB -W 24:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -oo outputs/lsf_logs/pinn_qbc_small.%J.out
#BSUB -eo outputs/lsf_logs/pinn_qbc_small.%J.err

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
  echo "[ERROR] submit with PYTHON_BIN=/path/to/venv/bin/python bsub < tools/pinn/jobs/run_qbc_small_experiment.lsf.sh"
  exit 1
fi

mkdir -p outputs/lsf_logs

STAMP="$(date +%Y%m%d_%H%M%S)"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-qbc_b1024_adam300_${STAMP}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/pinn_hpc_experiments/${EXPERIMENT_TAG}}"
MODELS="${MODELS:-SM4,SM6,SM_AVR_GOV}"
DATASET_SEED="${DATASET_SEED:-s01}"
PINN_BUDGET="${PINN_BUDGET:-b1024}"
PINN_EPOCHS="${PINN_EPOCHS:-300}"
PINN_DEVICE="${PINN_DEVICE:-cuda}"
PINN_BATCH_SIZE="${PINN_BATCH_SIZE:-1024}"
PINN_HIDDEN_DIM="${PINN_HIDDEN_DIM:-64}"
PINN_HIDDEN_LAYERS="${PINN_HIDDEN_LAYERS:-4}"
PINN_ACTIVATION="${PINN_ACTIVATION:-tanh}"
PINN_DTYPE="${PINN_DTYPE:-float64}"
WANDB_PROJECT="${WANDB_PROJECT:-sm-surrogates-pinn}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
LOG_EVERY_EPOCH="${LOG_EVERY_EPOCH:-1}"
DRY_RUN="${DRY_RUN:-false}"

CMD=(
  "${PYTHON_BIN}"
  tools/pinn/run_qbc_small_experiment.py
  --experiment-tag "${EXPERIMENT_TAG}"
  --output-root "${OUTPUT_ROOT}"
  --models "${MODELS}"
  --budget "${PINN_BUDGET}"
  --dataset-seed "${DATASET_SEED}"
  --epochs "${PINN_EPOCHS}"
  --device "${PINN_DEVICE}"
  --batch-size "${PINN_BATCH_SIZE}"
  --hidden-dim "${PINN_HIDDEN_DIM}"
  --hidden-layers "${PINN_HIDDEN_LAYERS}"
  --activation "${PINN_ACTIVATION}"
  --dtype "${PINN_DTYPE}"
  --wandb-project "${WANDB_PROJECT}"
  --log-every-epoch "${LOG_EVERY_EPOCH}"
)

if [[ -n "${WANDB_ENTITY}" ]]; then
  CMD+=(--wandb-entity "${WANDB_ENTITY}")
fi
if [[ "${DRY_RUN}" == "true" ]]; then
  CMD+=(--dry-run)
fi

echo "[pinn-qbc-small] repo_root=${REPO_ROOT}"
echo "[pinn-qbc-small] output_root=${OUTPUT_ROOT}"
echo "[pinn-qbc-small] command=${CMD[*]}"

"${CMD[@]}"
