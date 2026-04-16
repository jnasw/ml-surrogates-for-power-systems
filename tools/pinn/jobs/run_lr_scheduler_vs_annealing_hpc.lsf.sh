#!/usr/bin/env bash
#BSUB -J lr_sched_vs_anneal
#BSUB -q gpua100
#BSUB -n 1
#BSUB -W 36:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -oo outputs/lsf_logs/lr_sched_vs_anneal.%J.out
#BSUB -eo outputs/lsf_logs/lr_sched_vs_anneal.%J.err

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
  echo "[ERROR] submit with PYTHON_BIN=/path/to/venv/bin/python bsub < tools/pinn/jobs/run_lr_scheduler_vs_annealing_hpc.lsf.sh"
  exit 1
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
mkdir -p outputs/lsf_logs

STAMP="$(date +%Y%m%d_%H%M%S)"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-lr_scheduler_vs_annealing_hpc_${STAMP}}"
EXPERIMENT_ID="${EXPERIMENT_ID:-${EXPERIMENT_TAG}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/pinn_hpc_experiments/${EXPERIMENT_TAG}}"
DATASET_PIPELINE_ROOT="${OUTPUT_ROOT}/dataset_pipeline"
RUNS_ROOT="${OUTPUT_ROOT}/runs"
LOG_ROOT="${OUTPUT_ROOT}/logs"
mkdir -p "${RUNS_ROOT}" "${LOG_ROOT}"

MODEL_FLAG="${MODEL_FLAG:-SM4}"
METHOD="${METHOD:-qbc_deep_ensemble}"
PRESET="${PRESET:-default}"
PINN_BUDGET="${PINN_BUDGET:-b256}"
DATASET_SEED="${DATASET_SEED:-ds01}"
DATASET_ROOT="${DATASET_ROOT:-}"

PINN_SEED="${PINN_SEED:-37}"
PINN_DEVICE="${PINN_DEVICE:-cuda}"
PINN_DTYPE="${PINN_DTYPE:-float64}"
PINN_HIDDEN_DIM="${PINN_HIDDEN_DIM:-64}"
PINN_HIDDEN_LAYERS="${PINN_HIDDEN_LAYERS:-4}"
PINN_ACTIVATION="${PINN_ACTIVATION:-tanh}"
PINN_BATCH_SIZE="${PINN_BATCH_SIZE:-1024}"
PINN_EPOCHS="${PINN_EPOCHS:-1200}"
ADAM_LR="${ADAM_LR:-1e-3}"
ALLOW_SAMPLING="${ALLOW_SAMPLING:-false}"

LOSS_WEIGHT_DATA="${LOSS_WEIGHT_DATA:-1.0}"
LOSS_WEIGHT_DT="${LOSS_WEIGHT_DT:-1.0e-4}"
LOSS_WEIGHT_PHYSICS="${LOSS_WEIGHT_PHYSICS:-1.0e-4}"
LOSS_WEIGHT_IC="${LOSS_WEIGHT_IC:-1.0e-3}"

SCHEDULER_METRIC="${SCHEDULER_METRIC:-val_total_loss}"
SCHEDULER_FACTOR="${SCHEDULER_FACTOR:-0.5}"
SCHEDULER_PATIENCE="${SCHEDULER_PATIENCE:-15}"
SCHEDULER_THRESHOLD="${SCHEDULER_THRESHOLD:-1.0e-4}"
SCHEDULER_THRESHOLD_MODE="${SCHEDULER_THRESHOLD_MODE:-rel}"
SCHEDULER_COOLDOWN="${SCHEDULER_COOLDOWN:-0}"
SCHEDULER_MIN_LR="${SCHEDULER_MIN_LR:-1.0e-6}"
SCHEDULER_EPS="${SCHEDULER_EPS:-1.0e-8}"

WANDB_PROJECT="${WANDB_PROJECT:-sm-surrogates-pinn-hpc-lr-scheduler-vs-annealing}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_GROUP:-${EXPERIMENT_TAG}}"
LOG_EVERY_EPOCH="${LOG_EVERY_EPOCH:-1}"
GRADIENT_TELEMETRY="${GRADIENT_TELEMETRY:-false}"

STAGE1_OVERRIDES="${STAGE1_OVERRIDES:-}"
STAGE2_OVERRIDES="${STAGE2_OVERRIDES:-}"
DRY_RUN="${DRY_RUN:-false}"

echo "[lr-sched-vs-anneal] repo_root=${REPO_ROOT}"
echo "[lr-sched-vs-anneal] output_root=${OUTPUT_ROOT}"
echo "[lr-sched-vs-anneal] model_flag=${MODEL_FLAG}"
echo "[lr-sched-vs-anneal] method=${METHOD}"
echo "[lr-sched-vs-anneal] preset=${PRESET}"
echo "[lr-sched-vs-anneal] budget=${PINN_BUDGET}"
echo "[lr-sched-vs-anneal] dataset_seed=${DATASET_SEED}"
echo "[lr-sched-vs-anneal] epochs=${PINN_EPOCHS}"
echo "[lr-sched-vs-anneal] batch_size=${PINN_BATCH_SIZE}"
echo "[lr-sched-vs-anneal] dtype=${PINN_DTYPE}"
echo "[lr-sched-vs-anneal] wandb_project=${WANDB_PROJECT}"
echo "[lr-sched-vs-anneal] wandb_group=${WANDB_GROUP}"

if [[ "${DRY_RUN}" == "true" ]]; then
  echo "[lr-sched-vs-anneal] DRY_RUN=true"
fi

build_optimizer_override() {
  local with_scheduler="$1"
  if [[ "${with_scheduler}" == "true" ]]; then
    printf '%s' \
      "pinn.optimizer_phases=[{name:adam,optimizer:Adam,lr:${ADAM_LR},epochs:${PINN_EPOCHS},batch_size:${PINN_BATCH_SIZE},shuffle:true,full_batch:false,allow_sampling:${ALLOW_SAMPLING},optimizer_kwargs:{},scheduler:{name:reduce_on_plateau,metric:${SCHEDULER_METRIC},mode:min,factor:${SCHEDULER_FACTOR},patience:${SCHEDULER_PATIENCE},threshold:${SCHEDULER_THRESHOLD},threshold_mode:${SCHEDULER_THRESHOLD_MODE},cooldown:${SCHEDULER_COOLDOWN},min_lr:${SCHEDULER_MIN_LR},eps:${SCHEDULER_EPS}},line_search:null,convergence:null}]"
  else
    printf '%s' \
      "pinn.optimizer_phases=[{name:adam,optimizer:Adam,lr:${ADAM_LR},epochs:${PINN_EPOCHS},batch_size:${PINN_BATCH_SIZE},shuffle:true,full_batch:false,allow_sampling:${ALLOW_SAMPLING},optimizer_kwargs:{},scheduler:null,line_search:null,convergence:null}]"
  fi
}

dataset_root_from_manifest() {
  local manifest_path="$1"
  "${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

manifest_path = Path(r"${manifest_path}")
with manifest_path.open("r", encoding="utf-8") as f:
    manifest = json.load(f)
artifacts = dict(manifest.get("artifacts", {}))
dataset_root = artifacts.get("preprocessed_root") or artifacts.get("dataset_root")
if not dataset_root:
    raise SystemExit("dataset manifest missing preprocessed_root/dataset_root")
print(dataset_root)
PY
}

run_dataset_pipeline() {
  local dataset_cmd=(
    "${PYTHON_BIN}"
    -m
    src.pipeline.run_experiment
    --method "${METHOD}"
    --budget "${PINN_BUDGET}"
    --dataset-seed "${DATASET_SEED}"
    --preset "${PRESET}"
    --experiment-id "${EXPERIMENT_ID}"
    --model-flag "${MODEL_FLAG}"
    --run-root "${DATASET_PIPELINE_ROOT}"
    --skip-baseline
  )

  if [[ -n "${STAGE1_OVERRIDES}" ]]; then
    IFS='|' read -r -a _stage1_override_array <<< "${STAGE1_OVERRIDES}"
    for override in "${_stage1_override_array[@]}"; do
      if [[ -n "${override}" ]]; then
        dataset_cmd+=(--stage1-override "${override}")
      fi
    done
  fi
  if [[ -n "${STAGE2_OVERRIDES}" ]]; then
    IFS='|' read -r -a _stage2_override_array <<< "${STAGE2_OVERRIDES}"
    for override in "${_stage2_override_array[@]}"; do
      if [[ -n "${override}" ]]; then
        dataset_cmd+=(--stage2-override "${override}")
      fi
    done
  fi

  echo "[lr-sched-vs-anneal] dataset command=${dataset_cmd[*]}"
  if [[ "${DRY_RUN}" == "true" ]]; then
    return 0
  fi
  "${dataset_cmd[@]}" 2>&1 | tee "${LOG_ROOT}/dataset_pipeline.log"
}

run_case() {
  local run_name="$1"
  local weighting_scheme="$2"
  local with_scheduler="$3"

  local run_dir="${RUNS_ROOT}/${run_name}"
  local log_file="${LOG_ROOT}/${run_name}.log"
  local optimizer_override
  optimizer_override="$(build_optimizer_override "${with_scheduler}")"

  local cmd=(
    "${PYTHON_BIN}"
    20_run_pinn.py
    "model.model_flag=${MODEL_FLAG}"
    "model.seed=${PINN_SEED}"
    "dataset.root=${DATASET_ROOT}"
    "pinn.run_dir=${run_dir}"
    "pinn.device=${PINN_DEVICE}"
    "pinn.dtype=${PINN_DTYPE}"
    "pinn.hidden_dim=${PINN_HIDDEN_DIM}"
    "pinn.hidden_layers=${PINN_HIDDEN_LAYERS}"
    "pinn.activation=${PINN_ACTIVATION}"
    "pinn.gradient_telemetry.enabled=${GRADIENT_TELEMETRY}"
    "${optimizer_override}"
    "pinn.loss_weights.data=${LOSS_WEIGHT_DATA}"
    "pinn.loss_weights.dt=${LOSS_WEIGHT_DT}"
    "pinn.loss_weights.physics=${LOSS_WEIGHT_PHYSICS}"
    "pinn.loss_weights.ic=${LOSS_WEIGHT_IC}"
    "pinn.weighting.scheme=${weighting_scheme}"
    "pinn.weighting.dynamic_components=[data,dt,ic]"
    "logging.log_every_epoch=${LOG_EVERY_EPOCH}"
    "wandb.use=true"
    "wandb.project=${WANDB_PROJECT}"
    "wandb.group=${WANDB_GROUP}"
    "wandb.name=${run_name}"
  )

  if [[ "${weighting_scheme}" == "paper_lr_annealing" ]]; then
    cmd+=(
      "pinn.weighting.ema_beta=0.9"
      "pinn.weighting.update_mode=step"
      "pinn.weighting.use_live_batch=true"
      "wandb.tags=[hpc_compare,paper_lr_annealing,$([[ "${with_scheduler}" == "true" ]] && printf 'reduce_on_plateau' || printf 'no_scheduler'),${MODEL_FLAG,,},${PINN_EPOCHS}epochs]"
    )
  else
    cmd+=(
      "wandb.tags=[hpc_compare,static,$([[ "${with_scheduler}" == "true" ]] && printf 'reduce_on_plateau' || printf 'no_scheduler'),${MODEL_FLAG,,},${PINN_EPOCHS}epochs]"
    )
  fi

  if [[ -n "${WANDB_ENTITY}" ]]; then
    cmd+=("wandb.entity=${WANDB_ENTITY}")
  fi

  echo "[lr-sched-vs-anneal] run_name=${run_name}"
  echo "[lr-sched-vs-anneal] command=${cmd[*]}"
  if [[ "${DRY_RUN}" == "true" ]]; then
    return 0
  fi
  "${cmd[@]}" 2>&1 | tee "${log_file}"
}

if [[ -z "${DATASET_ROOT}" ]]; then
  run_dataset_pipeline
  if [[ "${DRY_RUN}" == "true" ]]; then
    DATASET_ROOT="${DATASET_PIPELINE_ROOT}/data/${MODEL_FLAG}/dataset_v1"
  else
    MANIFEST_PATH="${DATASET_PIPELINE_ROOT}/dataset_manifest.json"
    DATASET_ROOT="$(dataset_root_from_manifest "${MANIFEST_PATH}")"
  fi
fi

echo "[lr-sched-vs-anneal] dataset_root=${DATASET_ROOT}"

run_case "static" "static" "false"
run_case "static_reduce_on_plateau" "static" "true"
run_case "paper_lr_annealing" "paper_lr_annealing" "false"
run_case "paper_lr_annealing_reduce_on_plateau" "paper_lr_annealing" "true"

echo "[lr-sched-vs-anneal] completed all runs"
