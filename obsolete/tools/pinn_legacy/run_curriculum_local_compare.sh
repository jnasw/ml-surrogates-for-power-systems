#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${REPO_ROOT}/.venv/bin/activate"
fi

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
  elif [[ -x "${REPO_ROOT}/venv/bin/python" ]]; then
    PYTHON_BIN="${REPO_ROOT}/venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
MODEL_FLAG_LOWER="$(printf '%s' "${MODEL_FLAG:-SM4}" | tr '[:upper:]' '[:lower:]')"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-curriculum_local_${STAMP}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/${EXPERIMENT_TAG}}"
DATA_DIR="${DATA_DIR:-${OUTPUT_ROOT}/datasets}"
DATASET_ROOT="${DATASET_ROOT:-${DATA_DIR}/SM4/dataset_v1}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/logs}"

MODEL_FLAG="${MODEL_FLAG:-SM4}"
PRESET="${PRESET:-smoke}"
METHOD="${METHOD:-lhs_static}"
TRAJECTORIES="${TRAJECTORIES:-256}"
TIME_HORIZON="${TIME_HORIZON:-0.05}"
NUM_POINTS="${NUM_POINTS:-20}"
SPLIT_RATIO="${SPLIT_RATIO:-0.8}"
VALIDATION_FLAG="${VALIDATION_FLAG:-true}"
NEW_COL_POINTS="${NEW_COL_POINTS:-true}"

PINN_DEVICE="${PINN_DEVICE:-cpu}"
PINN_EPOCHS="${PINN_EPOCHS:-300}"
PINN_BATCH_SIZE="${PINN_BATCH_SIZE:-128}"
PINN_LR="${PINN_LR:-0.001}"
COLLOCATION_ACTIVE_POINTS="${COLLOCATION_ACTIVE_POINTS:-512}"
CURRICULUM_NUM_BINS="${CURRICULUM_NUM_BINS:-3}"
CURRICULUM_UNLOCK_EPOCHS="${CURRICULUM_UNLOCK_EPOCHS:-[1,100,200]}"
CURRICULUM_BALANCE_ACTIVE_BINS="${CURRICULUM_BALANCE_ACTIVE_BINS:-true}"

RUN_PARALLEL="${RUN_PARALLEL:-true}"
FORCE_CLEAN="${FORCE_CLEAN:-false}"
SKIP_STAGE1="${SKIP_STAGE1:-false}"
SKIP_STAGE2="${SKIP_STAGE2:-false}"
SKIP_BASELINE="${SKIP_BASELINE:-false}"
SKIP_CURRICULUM="${SKIP_CURRICULUM:-false}"
DRY_RUN="${DRY_RUN:-false}"

WANDB_USE="${WANDB_USE:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-sm-surrogates-curriculum-local}"
WANDB_GROUP="${WANDB_GROUP:-${MODEL_FLAG_LOWER}_curriculum_compare_${TRAJECTORIES}traj_${PINN_EPOCHS}ep_${STAMP}}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
BASELINE_RUN_NAME="${BASELINE_RUN_NAME:-baseline_${TRAJECTORIES}traj_${PINN_EPOCHS}ep}"
CURRICULUM_RUN_NAME="${CURRICULUM_RUN_NAME:-curriculum_${TRAJECTORIES}traj_${PINN_EPOCHS}ep}"
BASELINE_WANDB_TAGS="${BASELINE_WANDB_TAGS:-[local,baseline,curriculum_compare,${MODEL_FLAG_LOWER}]}"
CURRICULUM_WANDB_TAGS="${CURRICULUM_WANDB_TAGS:-[local,curriculum,curriculum_compare,${MODEL_FLAG_LOWER}]}"

BASELINE_RUN_DIR="${BASELINE_RUN_DIR:-${OUTPUT_ROOT}/pinn_baseline}"
CURRICULUM_RUN_DIR="${CURRICULUM_RUN_DIR:-${OUTPUT_ROOT}/pinn_curriculum}"

mkdir -p "${LOG_DIR}"

if [[ "${FORCE_CLEAN}" == "true" ]]; then
  echo "[curriculum-local] Removing output root: ${OUTPUT_ROOT}"
  rm -rf "${OUTPUT_ROOT}"
  mkdir -p "${LOG_DIR}"
fi

run_cmd() {
  echo "[curriculum-local] command:"
  printf ' %q' "$@"
  echo
  if [[ "${DRY_RUN}" == "true" ]]; then
    return 0
  fi
  "$@"
}

baseline_cmd=(
  "${PYTHON_BIN}" 20_run_pinn.py
  "dataset.root=${DATASET_ROOT}"
  "model.model_flag=${MODEL_FLAG}"
  "time=${TIME_HORIZON}"
  "num_of_points=${NUM_POINTS}"
  "pinn.device=${PINN_DEVICE}"
  "pinn.run_dir=${BASELINE_RUN_DIR}"
  "pinn.curriculum.enabled=false"
  "pinn.collocation.active_points=${COLLOCATION_ACTIVE_POINTS}"
  "pinn.collocation.multi_pool.total_active_points=${COLLOCATION_ACTIVE_POINTS}"
  "wandb.use=${WANDB_USE}"
  "wandb.project=${WANDB_PROJECT}"
  "wandb.group=${WANDB_GROUP}"
  "wandb.name=${BASELINE_RUN_NAME}"
  "wandb.tags=${BASELINE_WANDB_TAGS}"
  "pinn.optimizer_phases=[{name:adam_local,optimizer:Adam,lr:${PINN_LR},epochs:${PINN_EPOCHS},batch_size:${PINN_BATCH_SIZE},shuffle:true,full_batch:false,allow_sampling:true,optimizer_kwargs:{},scheduler:null,line_search:null,convergence:null}]"
)

curriculum_cmd=(
  "${PYTHON_BIN}" 20_run_pinn.py
  "dataset.root=${DATASET_ROOT}"
  "model.model_flag=${MODEL_FLAG}"
  "time=${TIME_HORIZON}"
  "num_of_points=${NUM_POINTS}"
  "pinn.device=${PINN_DEVICE}"
  "pinn.run_dir=${CURRICULUM_RUN_DIR}"
  "pinn.curriculum.enabled=true"
  "pinn.curriculum.num_bins=${CURRICULUM_NUM_BINS}"
  "pinn.curriculum.unlock_epochs=${CURRICULUM_UNLOCK_EPOCHS}"
  "pinn.curriculum.balance_active_bins=${CURRICULUM_BALANCE_ACTIVE_BINS}"
  "pinn.collocation.active_points=${COLLOCATION_ACTIVE_POINTS}"
  "pinn.collocation.multi_pool.total_active_points=${COLLOCATION_ACTIVE_POINTS}"
  "wandb.use=${WANDB_USE}"
  "wandb.project=${WANDB_PROJECT}"
  "wandb.group=${WANDB_GROUP}"
  "wandb.name=${CURRICULUM_RUN_NAME}"
  "wandb.tags=${CURRICULUM_WANDB_TAGS}"
  "pinn.optimizer_phases=[{name:adam_local,optimizer:Adam,lr:${PINN_LR},epochs:${PINN_EPOCHS},batch_size:${PINN_BATCH_SIZE},shuffle:true,full_batch:false,allow_sampling:true,optimizer_kwargs:{},scheduler:null,line_search:null,convergence:null}]"
)

if [[ -n "${WANDB_ENTITY}" ]]; then
  baseline_cmd+=("wandb.entity=${WANDB_ENTITY}")
  curriculum_cmd+=("wandb.entity=${WANDB_ENTITY}")
fi

echo "[curriculum-local] repo_root=${REPO_ROOT}"
echo "[curriculum-local] output_root=${OUTPUT_ROOT}"
echo "[curriculum-local] dataset_root=${DATASET_ROOT}"
echo "[curriculum-local] trajectories=${TRAJECTORIES}"
echo "[curriculum-local] collocation_active_points=${COLLOCATION_ACTIVE_POINTS}"
echo "[curriculum-local] pinn_epochs=${PINN_EPOCHS}"
echo "[curriculum-local] run_parallel=${RUN_PARALLEL}"
echo "[curriculum-local] wandb_use=${WANDB_USE}"
if [[ "${WANDB_USE}" == "true" ]]; then
  echo "[curriculum-local] wandb_project=${WANDB_PROJECT}"
  echo "[curriculum-local] wandb_group=${WANDB_GROUP}"
fi

if [[ "${SKIP_STAGE1}" != "true" ]]; then
  run_cmd \
    "${PYTHON_BIN}" 00_create_dataset.py \
    "preset=${PRESET}" \
    "+method=${METHOD}" \
    "dirs.dataset_dir=${DATA_DIR}" \
    "model.model_flag=${MODEL_FLAG}" \
    "model.ic_num_samples=${TRAJECTORIES}" \
    "time=${TIME_HORIZON}" \
    "num_of_points=${NUM_POINTS}"
fi

if [[ "${SKIP_STAGE2}" != "true" ]]; then
  run_cmd \
    "${PYTHON_BIN}" 01_preprocess_dataset.py \
    "dataset.root=${DATASET_ROOT}" \
    "dirs.dataset_dir=${DATA_DIR}" \
    "model.model_flag=${MODEL_FLAG}" \
    "time=${TIME_HORIZON}" \
    "num_of_points=${NUM_POINTS}" \
    "dataset.validation_flag=${VALIDATION_FLAG}" \
    "dataset.split_ratio=${SPLIT_RATIO}" \
    "dataset.new_coll_points_flag=${NEW_COL_POINTS}" \
    "+pinn.curriculum.enabled=true" \
    "+pinn.curriculum.num_bins=${CURRICULUM_NUM_BINS}"
fi

if [[ "${SKIP_BASELINE}" == "true" && "${SKIP_CURRICULUM}" == "true" ]]; then
  echo "[curriculum-local] Both training runs skipped."
  exit 0
fi

if [[ "${RUN_PARALLEL}" == "true" ]]; then
  pids=()
  if [[ "${SKIP_BASELINE}" != "true" ]]; then
    (
      exec "${baseline_cmd[@]}"
    ) > "${LOG_DIR}/baseline.log" 2>&1 &
    baseline_pid=$!
    pids+=("${baseline_pid}")
    echo "[curriculum-local] Baseline PID=${baseline_pid} log=${LOG_DIR}/baseline.log"
  fi
  if [[ "${SKIP_CURRICULUM}" != "true" ]]; then
    (
      exec "${curriculum_cmd[@]}"
    ) > "${LOG_DIR}/curriculum.log" 2>&1 &
    curriculum_pid=$!
    pids+=("${curriculum_pid}")
    echo "[curriculum-local] Curriculum PID=${curriculum_pid} log=${LOG_DIR}/curriculum.log"
  fi
  if [[ "${DRY_RUN}" == "true" ]]; then
    run_cmd "${baseline_cmd[@]}"
    run_cmd "${curriculum_cmd[@]}"
    exit 0
  fi
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done
else
  if [[ "${SKIP_BASELINE}" != "true" ]]; then
    run_cmd "${baseline_cmd[@]}"
  fi
  if [[ "${SKIP_CURRICULUM}" != "true" ]]; then
    run_cmd "${curriculum_cmd[@]}"
  fi
fi

echo "[curriculum-local] Complete."
echo "[curriculum-local] baseline_run_dir=${BASELINE_RUN_DIR}"
echo "[curriculum-local] curriculum_run_dir=${CURRICULUM_RUN_DIR}"
echo "[curriculum-local] logs=${LOG_DIR}"
