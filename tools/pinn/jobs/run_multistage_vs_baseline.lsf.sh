#!/usr/bin/env bash
#BSUB -J multistage_vs_baseline
#BSUB -q gpua100
#BSUB -n 1
#BSUB -W 24:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -oo outputs/lsf_logs/multistage_vs_baseline.%J.out
#BSUB -eo outputs/lsf_logs/multistage_vs_baseline.%J.err

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
  echo "[ERROR] submit with PYTHON_BIN=/path/to/venv/bin/python bsub < tools/pinn/jobs/run_multistage_vs_baseline.lsf.sh"
  exit 1
fi

mkdir -p outputs/lsf_logs

STAMP="$(date +%Y%m%d_%H%M%S)"
PROFILE="${PROFILE:-benchmark}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-multistage_vs_baseline_${PROFILE}_${STAMP}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/pinn_hpc_experiments/${EXPERIMENT_TAG}}"
MODEL_FLAG="${MODEL_FLAG:-SM4}"
PRESET="${PRESET:-}"
PINN_BUDGET="${PINN_BUDGET:-b256}"
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
PHASE_SEQUENCES="${PHASE_SEQUENCES:-LBFGS:300|Adam:300;LBFGS:100|Adam:300;LBFGS:100;Adam:300|Adam:300;LBFGS:100;Adam:300;SSBroyden:300}"
MODES="${MODES:-single_stage,multistage}"
MULTISTAGE_MAX_STAGES="${MULTISTAGE_MAX_STAGES:-2}"
ANALYSIS_COLLOCATION_ROWS="${ANALYSIS_COLLOCATION_ROWS:-2048}"
WANDB_PROJECT="${WANDB_PROJECT:-sm-surrogates-pinn-multistage-vs-baseline}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
LOG_EVERY_EPOCH="${LOG_EVERY_EPOCH:-1}"
LOSS_WEIGHT_DATA="${LOSS_WEIGHT_DATA:-1.0}"
LOSS_WEIGHT_DT="${LOSS_WEIGHT_DT:-1.0e-4}"
LOSS_WEIGHT_PHYSICS="${LOSS_WEIGHT_PHYSICS:-1.0e-4}"
LOSS_WEIGHT_IC="${LOSS_WEIGHT_IC:-1.0e-3}"
GRADIENT_TELEMETRY="${GRADIENT_TELEMETRY:-true}"
DATASET_ROOT="${DATASET_ROOT:-}"
STAGE1_OVERRIDES="${STAGE1_OVERRIDES:-}"
STAGE2_OVERRIDES="${STAGE2_OVERRIDES:-}"
DRY_RUN="${DRY_RUN:-false}"

CMD=(
  "${PYTHON_BIN}"
  tools/pinn/run_multistage_vs_baseline.py
  --profile "${PROFILE}"
  --experiment-tag "${EXPERIMENT_TAG}"
  --output-root "${OUTPUT_ROOT}"
  --model-flag "${MODEL_FLAG}"
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
  --modes "${MODES}"
  --multistage-max-stages "${MULTISTAGE_MAX_STAGES}"
  --analysis-collocation-rows "${ANALYSIS_COLLOCATION_ROWS}"
  --wandb-project "${WANDB_PROJECT}"
  --log-every-epoch "${LOG_EVERY_EPOCH}"
  --loss-weight-data "${LOSS_WEIGHT_DATA}"
  --loss-weight-dt "${LOSS_WEIGHT_DT}"
  --loss-weight-physics "${LOSS_WEIGHT_PHYSICS}"
  --loss-weight-ic "${LOSS_WEIGHT_IC}"
)

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
if [[ -n "${PHASE_SEQUENCES}" ]]; then
  IFS='|' read -r -a _phase_sequence_array <<< "${PHASE_SEQUENCES}"
  for sequence in "${_phase_sequence_array[@]}"; do
    if [[ -n "${sequence}" ]]; then
      CMD+=(--phase-sequence "${sequence}")
    fi
  done
fi
if [[ -n "${WANDB_ENTITY}" ]]; then
  CMD+=(--wandb-entity "${WANDB_ENTITY}")
fi
if [[ -n "${DATASET_ROOT}" ]]; then
  CMD+=(--dataset-root "${DATASET_ROOT}")
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

echo "[multistage-vs-baseline] repo_root=${REPO_ROOT}"
echo "[multistage-vs-baseline] output_root=${OUTPUT_ROOT}"
echo "[multistage-vs-baseline] profile=${PROFILE}"
echo "[multistage-vs-baseline] model_flag=${MODEL_FLAG}"
echo "[multistage-vs-baseline] dataset_seed=${DATASET_SEED}"
echo "[multistage-vs-baseline] modes=${MODES}"
echo "[multistage-vs-baseline] phase_sequences=${PHASE_SEQUENCES}"
if [[ -n "${PRESET}" ]]; then
  echo "[multistage-vs-baseline] preset=${PRESET}"
fi
if [[ -n "${PINN_BUDGET}" ]]; then
  echo "[multistage-vs-baseline] budget=${PINN_BUDGET}"
fi
if [[ -n "${DATASET_ROOT}" ]]; then
  echo "[multistage-vs-baseline] dataset_root=${DATASET_ROOT}"
fi
if [[ -n "${STAGE1_OVERRIDES}" ]]; then
  echo "[multistage-vs-baseline] stage1_overrides=${STAGE1_OVERRIDES}"
fi
if [[ -n "${STAGE2_OVERRIDES}" ]]; then
  echo "[multistage-vs-baseline] stage2_overrides=${STAGE2_OVERRIDES}"
fi
echo "[multistage-vs-baseline] command=${CMD[*]}"

"${CMD[@]}"
