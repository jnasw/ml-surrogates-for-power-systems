#!/usr/bin/env bash
#BSUB -J weighting_comparison
#BSUB -q gpua100
#BSUB -n 1
#BSUB -W 24:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -oo outputs/lsf_logs/weighting_comparison.%J.out
#BSUB -eo outputs/lsf_logs/weighting_comparison.%J.err

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
  echo "[ERROR] submit with PYTHON_BIN=/path/to/venv/bin/python bsub < tools/pinn/jobs/run_weighting_comparison.lsf.sh"
  exit 1
fi

mkdir -p outputs/lsf_logs

STAMP="$(date +%Y%m%d_%H%M%S)"
PROFILE="${PROFILE:-smoke}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-weighting_comparison_${PROFILE}_${STAMP}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/pinn/weighting_comparison/${EXPERIMENT_TAG}}"
MODEL_FLAG="${MODEL_FLAG:-SM4}"
METHOD="${METHOD:-}"
PRESET="${PRESET:-}"
PINN_BUDGET="${PINN_BUDGET:-}"
DATASET_SEED="${DATASET_SEED:-ds01}"
DATASET_ROOT="${DATASET_ROOT:-}"
PINN_SEED="${PINN_SEED:-37}"
PINN_DEVICE="${PINN_DEVICE:-cuda}"
PINN_HIDDEN_DIM="${PINN_HIDDEN_DIM:-64}"
PINN_HIDDEN_LAYERS="${PINN_HIDDEN_LAYERS:-4}"
PINN_ACTIVATION="${PINN_ACTIVATION:-tanh}"
PINN_DTYPE="${PINN_DTYPE:-float64}"
PINN_BATCH_SIZE="${PINN_BATCH_SIZE:-}"
PINN_EPOCHS="${PINN_EPOCHS:-}"
ADAM_LR="${ADAM_LR:-1e-3}"
ALLOW_SAMPLING="${ALLOW_SAMPLING:-false}"
SCHEMES="${SCHEMES:-static,ma,id,dn}"
EMA_BETA="${EMA_BETA:-0.99}"
UPDATE_INTERVAL_EPOCHS="${UPDATE_INTERVAL_EPOCHS:-10}"
PROBE_DATA_ROWS="${PROBE_DATA_ROWS:-}"
PROBE_PHYSICS_ROWS="${PROBE_PHYSICS_ROWS:-}"
PROBE_INIT_ROWS="${PROBE_INIT_ROWS:-}"
PROBE_SEED="${PROBE_SEED:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-sm-surrogates-pinn}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
LOG_EVERY_EPOCH="${LOG_EVERY_EPOCH:-10}"
LOSS_WEIGHT_DATA="${LOSS_WEIGHT_DATA:-}"
LOSS_WEIGHT_DT="${LOSS_WEIGHT_DT:-}"
LOSS_WEIGHT_PHYSICS="${LOSS_WEIGHT_PHYSICS:-}"
LOSS_WEIGHT_IC="${LOSS_WEIGHT_IC:-}"
GRADIENT_TELEMETRY="${GRADIENT_TELEMETRY:-true}"
STAGE1_OVERRIDES="${STAGE1_OVERRIDES:-}"
STAGE2_OVERRIDES="${STAGE2_OVERRIDES:-}"
DRY_RUN="${DRY_RUN:-false}"

CMD=(
  "${PYTHON_BIN}"
  tools/pinn/run_weighting_comparison.py
  --profile "${PROFILE}"
  --experiment-tag "${EXPERIMENT_TAG}"
  --output-root "${OUTPUT_ROOT}"
  --model-flag "${MODEL_FLAG}"
  --dataset-seed "${DATASET_SEED}"
  --seed "${PINN_SEED}"
  --device "${PINN_DEVICE}"
  --hidden-dim "${PINN_HIDDEN_DIM}"
  --hidden-layers "${PINN_HIDDEN_LAYERS}"
  --activation "${PINN_ACTIVATION}"
  --dtype "${PINN_DTYPE}"
  --adam-lr "${ADAM_LR}"
  --schemes "${SCHEMES}"
  --ema-beta "${EMA_BETA}"
  --update-interval-epochs "${UPDATE_INTERVAL_EPOCHS}"
  --probe-seed "${PROBE_SEED}"
  --wandb-project "${WANDB_PROJECT}"
  --log-every-epoch "${LOG_EVERY_EPOCH}"
)

if [[ "${ALLOW_SAMPLING}" == "true" ]]; then
  CMD+=(--allow-sampling)
else
  CMD+=(--no-allow-sampling)
fi

if [[ "${GRADIENT_TELEMETRY}" == "true" ]]; then
  CMD+=(--gradient-telemetry)
else
  CMD+=(--no-gradient-telemetry)
fi

if [[ -n "${METHOD}" ]]; then
  CMD+=(--method "${METHOD}")
fi
if [[ -n "${PRESET}" ]]; then
  CMD+=(--preset "${PRESET}")
fi
if [[ -n "${PINN_BUDGET}" ]]; then
  CMD+=(--budget "${PINN_BUDGET}")
fi
if [[ -n "${DATASET_ROOT}" ]]; then
  CMD+=(--dataset-root "${DATASET_ROOT}")
fi
if [[ -n "${PINN_BATCH_SIZE}" ]]; then
  CMD+=(--batch-size "${PINN_BATCH_SIZE}")
fi
if [[ -n "${PINN_EPOCHS}" ]]; then
  CMD+=(--epochs "${PINN_EPOCHS}")
fi
if [[ -n "${PROBE_DATA_ROWS}" ]]; then
  CMD+=(--probe-data-rows "${PROBE_DATA_ROWS}")
fi
if [[ -n "${PROBE_PHYSICS_ROWS}" ]]; then
  CMD+=(--probe-physics-rows "${PROBE_PHYSICS_ROWS}")
fi
if [[ -n "${PROBE_INIT_ROWS}" ]]; then
  CMD+=(--probe-init-rows "${PROBE_INIT_ROWS}")
fi
if [[ -n "${WANDB_ENTITY}" ]]; then
  CMD+=(--wandb-entity "${WANDB_ENTITY}")
fi
if [[ -n "${LOSS_WEIGHT_DATA}" ]]; then
  CMD+=(--loss-weight-data "${LOSS_WEIGHT_DATA}")
fi
if [[ -n "${LOSS_WEIGHT_DT}" ]]; then
  CMD+=(--loss-weight-dt "${LOSS_WEIGHT_DT}")
fi
if [[ -n "${LOSS_WEIGHT_PHYSICS}" ]]; then
  CMD+=(--loss-weight-physics "${LOSS_WEIGHT_PHYSICS}")
fi
if [[ -n "${LOSS_WEIGHT_IC}" ]]; then
  CMD+=(--loss-weight-ic "${LOSS_WEIGHT_IC}")
fi
if [[ -n "${STAGE1_OVERRIDES}" ]]; then
  IFS='|' read -r -a _stage1_override_array <<< "${STAGE1_OVERRIDES}"
  for override in "${_stage1_override_array[@]}"; do
    if [[ -n "${override}" ]]; then
      CMD+=(--stage1-override "${override}")
    fi
  done
fi
if [[ -n "${STAGE2_OVERRIDES}" ]]; then
  IFS='|' read -r -a _stage2_override_array <<< "${STAGE2_OVERRIDES}"
  for override in "${_stage2_override_array[@]}"; do
    if [[ -n "${override}" ]]; then
      CMD+=(--stage2-override "${override}")
    fi
  done
fi
if [[ "${DRY_RUN}" == "true" ]]; then
  CMD+=(--dry-run)
fi

echo "[weighting-comparison] repo_root=${REPO_ROOT}"
echo "[weighting-comparison] output_root=${OUTPUT_ROOT}"
echo "[weighting-comparison] profile=${PROFILE}"
echo "[weighting-comparison] model_flag=${MODEL_FLAG}"
echo "[weighting-comparison] dataset_seed=${DATASET_SEED}"
echo "[weighting-comparison] schemes=${SCHEMES}"
if [[ -n "${METHOD}" ]]; then
  echo "[weighting-comparison] method=${METHOD}"
fi
if [[ -n "${PRESET}" ]]; then
  echo "[weighting-comparison] preset=${PRESET}"
fi
if [[ -n "${PINN_BUDGET}" ]]; then
  echo "[weighting-comparison] budget=${PINN_BUDGET}"
fi
if [[ -n "${DATASET_ROOT}" ]]; then
  echo "[weighting-comparison] dataset_root=${DATASET_ROOT}"
fi
if [[ -n "${PINN_EPOCHS}" ]]; then
  echo "[weighting-comparison] epochs=${PINN_EPOCHS}"
fi
if [[ -n "${PINN_BATCH_SIZE}" ]]; then
  echo "[weighting-comparison] batch_size=${PINN_BATCH_SIZE}"
fi
if [[ -n "${STAGE1_OVERRIDES}" ]]; then
  echo "[weighting-comparison] stage1_overrides=${STAGE1_OVERRIDES}"
fi
if [[ -n "${STAGE2_OVERRIDES}" ]]; then
  echo "[weighting-comparison] stage2_overrides=${STAGE2_OVERRIDES}"
fi
echo "[weighting-comparison] command=${CMD[*]}"

"${CMD[@]}"
