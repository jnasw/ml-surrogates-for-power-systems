#!/usr/bin/env bash
#BSUB -J collocation_comparison
#BSUB -q gpua100
#BSUB -n 1
#BSUB -W 24:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -oo outputs/lsf_logs/collocation_comparison.%J.out
#BSUB -eo outputs/lsf_logs/collocation_comparison.%J.err

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
  echo "[ERROR] submit with PYTHON_BIN=/path/to/venv/bin/python bsub < tools/pinn/jobs/run_collocation_comparison.lsf.sh"
  exit 1
fi

mkdir -p outputs/lsf_logs

STAMP="$(date +%Y%m%d_%H%M%S)"
PROFILE="${PROFILE:-multipool_benchmark}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-collocation_comparison_${PROFILE}_${STAMP}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/pinn_hpc_experiments/${EXPERIMENT_TAG}}"
MODEL_FLAG="${MODEL_FLAG:-SM4}"
PRESET="${PRESET:-}"
PINN_BUDGET="${PINN_BUDGET:-}"
DATASET_SEED="${DATASET_SEED:-s01}"
DATASET_ROOT="${DATASET_ROOT:-}"
VARIANTS="${VARIANTS:-}"
PINN_SEED="${PINN_SEED:-37}"
PINN_DEVICE="${PINN_DEVICE:-cuda}"
PINN_HIDDEN_DIM="${PINN_HIDDEN_DIM:-64}"
PINN_HIDDEN_LAYERS="${PINN_HIDDEN_LAYERS:-4}"
PINN_ACTIVATION="${PINN_ACTIVATION:-tanh}"
PINN_DTYPE="${PINN_DTYPE:-float64}"
PINN_BATCH_SIZE="${PINN_BATCH_SIZE:-}"
PINN_EPOCHS="${PINN_EPOCHS:-}"
ADAM_LR="${ADAM_LR:-1e-3}"
ACTIVE_POINTS="${ACTIVE_POINTS:-}"
INITIAL_POINTS="${INITIAL_POINTS:-}"
CANDIDATE_POINTS="${CANDIDATE_POINTS:-}"
APPEND_POINTS="${APPEND_POINTS:-}"
REFRESH_PERIOD_EPOCHS="${REFRESH_PERIOD_EPOCHS:-}"
SAMPLER="${SAMPLER:-}"
SCORE_NORM="${SCORE_NORM:-}"
RAD_K="${RAD_K:-}"
RAD_C="${RAD_C:-}"
RAR_D_K="${RAR_D_K:-}"
RAR_D_C="${RAR_D_C:-}"
MULTIPOOL="${MULTIPOOL:-}"
MULTIPOOL_ALLOCATION="${MULTIPOOL_ALLOCATION:-}"
MULTIPOOL_TOTAL_ACTIVE_POINTS="${MULTIPOOL_TOTAL_ACTIVE_POINTS:-}"
MULTIPOOL_ALLOCATION_METHOD="${MULTIPOOL_ALLOCATION_METHOD:-}"
MULTIPOOL_UPDATE_INTERVAL_EPOCHS="${MULTIPOOL_UPDATE_INTERVAL_EPOCHS:-}"
MULTIPOOL_SMOOTHING="${MULTIPOOL_SMOOTHING:-}"
RESIDUAL_INITIAL_FRACTION="${RESIDUAL_INITIAL_FRACTION:-}"
RESIDUAL_MIN_FRACTION="${RESIDUAL_MIN_FRACTION:-}"
RESIDUAL_MAX_FRACTION="${RESIDUAL_MAX_FRACTION:-}"
IC_INITIAL_FRACTION="${IC_INITIAL_FRACTION:-}"
IC_MIN_FRACTION="${IC_MIN_FRACTION:-}"
IC_MAX_FRACTION="${IC_MAX_FRACTION:-}"
WANDB_USE="${WANDB_USE:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-sm-surrogates-pinn-collocation-comparison}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
LOG_EVERY_EPOCH="${LOG_EVERY_EPOCH:-1}"
GRADIENT_TELEMETRY="${GRADIENT_TELEMETRY:-false}"
EXTRA_TAGS="${EXTRA_TAGS:-}"
STAGE1_OVERRIDES="${STAGE1_OVERRIDES:-}"
STAGE2_OVERRIDES="${STAGE2_OVERRIDES:-}"
DRY_RUN="${DRY_RUN:-false}"

CMD=(
  "${PYTHON_BIN}"
  tools/pinn/run_collocation_comparison.py
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
  --wandb-project "${WANDB_PROJECT}"
  --log-every-epoch "${LOG_EVERY_EPOCH}"
)

if [[ "${WANDB_USE}" == "true" ]]; then
  CMD+=(--wandb-use)
else
  CMD+=(--no-wandb-use)
fi

if [[ "${GRADIENT_TELEMETRY}" == "true" ]]; then
  CMD+=(--gradient-telemetry)
else
  CMD+=(--no-gradient-telemetry)
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
if [[ -n "${VARIANTS}" ]]; then
  CMD+=(--variants "${VARIANTS}")
fi
if [[ -n "${PINN_BATCH_SIZE}" ]]; then
  CMD+=(--batch-size "${PINN_BATCH_SIZE}")
fi
if [[ -n "${PINN_EPOCHS}" ]]; then
  CMD+=(--epochs "${PINN_EPOCHS}")
fi
if [[ -n "${ACTIVE_POINTS}" ]]; then
  CMD+=(--active-points "${ACTIVE_POINTS}")
fi
if [[ -n "${INITIAL_POINTS}" ]]; then
  CMD+=(--initial-points "${INITIAL_POINTS}")
fi
if [[ -n "${CANDIDATE_POINTS}" ]]; then
  CMD+=(--candidate-points "${CANDIDATE_POINTS}")
fi
if [[ -n "${APPEND_POINTS}" ]]; then
  CMD+=(--append-points "${APPEND_POINTS}")
fi
if [[ -n "${REFRESH_PERIOD_EPOCHS}" ]]; then
  CMD+=(--refresh-period-epochs "${REFRESH_PERIOD_EPOCHS}")
fi
if [[ -n "${SAMPLER}" ]]; then
  CMD+=(--sampler "${SAMPLER}")
fi
if [[ -n "${SCORE_NORM}" ]]; then
  CMD+=(--score-norm "${SCORE_NORM}")
fi
if [[ -n "${RAD_K}" ]]; then
  CMD+=(--rad-k "${RAD_K}")
fi
if [[ -n "${RAD_C}" ]]; then
  CMD+=(--rad-c "${RAD_C}")
fi
if [[ -n "${RAR_D_K}" ]]; then
  CMD+=(--rar-d-k "${RAR_D_K}")
fi
if [[ -n "${RAR_D_C}" ]]; then
  CMD+=(--rar-d-c "${RAR_D_C}")
fi
if [[ -n "${MULTIPOOL}" ]]; then
  if [[ "${MULTIPOOL}" == "true" ]]; then
    CMD+=(--multipool)
  else
    CMD+=(--no-multipool)
  fi
fi
if [[ -n "${MULTIPOOL_ALLOCATION}" ]]; then
  if [[ "${MULTIPOOL_ALLOCATION}" == "true" ]]; then
    CMD+=(--multipool-allocation)
  else
    CMD+=(--no-multipool-allocation)
  fi
fi
if [[ -n "${MULTIPOOL_TOTAL_ACTIVE_POINTS}" ]]; then
  CMD+=(--multipool-total-active-points "${MULTIPOOL_TOTAL_ACTIVE_POINTS}")
fi
if [[ -n "${MULTIPOOL_ALLOCATION_METHOD}" ]]; then
  CMD+=(--multipool-allocation-method "${MULTIPOOL_ALLOCATION_METHOD}")
fi
if [[ -n "${MULTIPOOL_UPDATE_INTERVAL_EPOCHS}" ]]; then
  CMD+=(--multipool-update-interval-epochs "${MULTIPOOL_UPDATE_INTERVAL_EPOCHS}")
fi
if [[ -n "${MULTIPOOL_SMOOTHING}" ]]; then
  CMD+=(--multipool-smoothing "${MULTIPOOL_SMOOTHING}")
fi
if [[ -n "${RESIDUAL_INITIAL_FRACTION}" ]]; then
  CMD+=(--residual-initial-fraction "${RESIDUAL_INITIAL_FRACTION}")
fi
if [[ -n "${RESIDUAL_MIN_FRACTION}" ]]; then
  CMD+=(--residual-min-fraction "${RESIDUAL_MIN_FRACTION}")
fi
if [[ -n "${RESIDUAL_MAX_FRACTION}" ]]; then
  CMD+=(--residual-max-fraction "${RESIDUAL_MAX_FRACTION}")
fi
if [[ -n "${IC_INITIAL_FRACTION}" ]]; then
  CMD+=(--ic-initial-fraction "${IC_INITIAL_FRACTION}")
fi
if [[ -n "${IC_MIN_FRACTION}" ]]; then
  CMD+=(--ic-min-fraction "${IC_MIN_FRACTION}")
fi
if [[ -n "${IC_MAX_FRACTION}" ]]; then
  CMD+=(--ic-max-fraction "${IC_MAX_FRACTION}")
fi
if [[ -n "${WANDB_ENTITY}" ]]; then
  CMD+=(--wandb-entity "${WANDB_ENTITY}")
fi
if [[ -n "${EXTRA_TAGS}" ]]; then
  IFS=',' read -r -a _extra_tag_array <<< "${EXTRA_TAGS}"
  for tag in "${_extra_tag_array[@]}"; do
    if [[ -n "${tag}" ]]; then
      CMD+=(--tag "${tag}")
    fi
  done
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

echo "[collocation-comparison] repo_root=${REPO_ROOT}"
echo "[collocation-comparison] output_root=${OUTPUT_ROOT}"
echo "[collocation-comparison] profile=${PROFILE}"
echo "[collocation-comparison] model_flag=${MODEL_FLAG}"
echo "[collocation-comparison] dataset_seed=${DATASET_SEED}"
if [[ -n "${PINN_BUDGET}" ]]; then
  echo "[collocation-comparison] budget=${PINN_BUDGET}"
fi
if [[ -n "${PRESET}" ]]; then
  echo "[collocation-comparison] preset=${PRESET}"
fi
if [[ -n "${VARIANTS}" ]]; then
  echo "[collocation-comparison] variants=${VARIANTS}"
fi
if [[ -n "${DATASET_ROOT}" ]]; then
  echo "[collocation-comparison] dataset_root=${DATASET_ROOT}"
fi
if [[ -n "${PINN_EPOCHS}" ]]; then
  echo "[collocation-comparison] epochs=${PINN_EPOCHS}"
fi
if [[ -n "${PINN_BATCH_SIZE}" ]]; then
  echo "[collocation-comparison] batch_size=${PINN_BATCH_SIZE}"
fi
if [[ -n "${MULTIPOOL}" ]]; then
  echo "[collocation-comparison] multipool=${MULTIPOOL}"
fi
if [[ -n "${MULTIPOOL_ALLOCATION}" ]]; then
  echo "[collocation-comparison] multipool_allocation=${MULTIPOOL_ALLOCATION}"
fi
if [[ -n "${STAGE1_OVERRIDES}" ]]; then
  echo "[collocation-comparison] stage1_overrides=${STAGE1_OVERRIDES}"
fi
if [[ -n "${STAGE2_OVERRIDES}" ]]; then
  echo "[collocation-comparison] stage2_overrides=${STAGE2_OVERRIDES}"
fi
echo "[collocation-comparison] command=${CMD[*]}"

"${CMD[@]}"
