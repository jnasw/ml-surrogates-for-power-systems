#!/usr/bin/env bash
#BSUB -J curriculum_vs_baseline
#BSUB -q gpua100
#BSUB -n 1
#BSUB -W 36:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -oo outputs/lsf_logs/curriculum_vs_baseline.%J.out
#BSUB -eo outputs/lsf_logs/curriculum_vs_baseline.%J.err

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
  echo "[ERROR] submit with PYTHON_BIN=/path/to/venv/bin/python bsub < tools/pinn/jobs/run_curriculum_vs_baseline_hpc.lsf.sh"
  exit 1
fi

mkdir -p outputs/lsf_logs

STAMP="$(date +%Y%m%d_%H%M%S)"
MODEL_FLAG_LOWER="$(printf '%s' "${MODEL_FLAG:-SM4}" | tr '[:upper:]' '[:lower:]')"

EXPERIMENT_TAG="${EXPERIMENT_TAG:-curriculum_vs_baseline_${STAMP}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/pinn_hpc_experiments/${EXPERIMENT_TAG}}"
DATA_DIR="${DATA_DIR:-${OUTPUT_ROOT}/datasets}"
DATASET_ROOT="${DATASET_ROOT:-${DATA_DIR}/SM4/dataset_v1}"
RUNS_ROOT="${RUNS_ROOT:-${OUTPUT_ROOT}/runs}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/logs}"
mkdir -p "${RUNS_ROOT}" "${LOG_ROOT}"

MODEL_FLAG="${MODEL_FLAG:-SM4}"
PRESET="${PRESET:-default}"
METHOD="${METHOD:-lhs_static}"
TRAJECTORIES="${TRAJECTORIES:-1024}"
TIME_HORIZON="${TIME_HORIZON:-0.2}"
NUM_POINTS="${NUM_POINTS:-80}"
SPLIT_RATIO="${SPLIT_RATIO:-0.8}"
VALIDATION_FLAG="${VALIDATION_FLAG:-true}"
NEW_COL_POINTS="${NEW_COL_POINTS:-true}"
PINN_SEED="${PINN_SEED:-37}"

PINN_DEVICE="${PINN_DEVICE:-cuda}"
PINN_DTYPE="${PINN_DTYPE:-float64}"
PINN_HIDDEN_DIM="${PINN_HIDDEN_DIM:-64}"
PINN_HIDDEN_LAYERS="${PINN_HIDDEN_LAYERS:-4}"
PINN_ACTIVATION="${PINN_ACTIVATION:-tanh}"
PINN_BATCH_SIZE="${PINN_BATCH_SIZE:-256}"
PINN_EPOCHS="${PINN_EPOCHS:-3000}"
ADAM_LR="${ADAM_LR:-1e-3}"
ALLOW_SAMPLING="${ALLOW_SAMPLING:-true}"

COLLOCATION_ACTIVE_POINTS="${COLLOCATION_ACTIVE_POINTS:-2048}"
CURRICULUM_NUM_BINS="${CURRICULUM_NUM_BINS:-3}"
CURRICULUM_UNLOCK_EPOCHS="${CURRICULUM_UNLOCK_EPOCHS:-[1,750,1500]}"
CURRICULUM_BALANCE_ACTIVE_BINS="${CURRICULUM_BALANCE_ACTIVE_BINS:-true}"

SCHEDULER_METRIC="${SCHEDULER_METRIC:-val_total_loss}"
SCHEDULER_FACTOR="${SCHEDULER_FACTOR:-0.5}"
SCHEDULER_PATIENCE="${SCHEDULER_PATIENCE:-20}"
SCHEDULER_THRESHOLD="${SCHEDULER_THRESHOLD:-1.0e-4}"
SCHEDULER_THRESHOLD_MODE="${SCHEDULER_THRESHOLD_MODE:-rel}"
SCHEDULER_COOLDOWN="${SCHEDULER_COOLDOWN:-5}"
SCHEDULER_MIN_LR="${SCHEDULER_MIN_LR:-1.0e-6}"
SCHEDULER_EPS="${SCHEDULER_EPS:-1.0e-8}"

LOSS_WEIGHT_DATA="${LOSS_WEIGHT_DATA:-1.0}"
LOSS_WEIGHT_DT="${LOSS_WEIGHT_DT:-1.0e-4}"
LOSS_WEIGHT_PHYSICS="${LOSS_WEIGHT_PHYSICS:-1.0e-4}"
LOSS_WEIGHT_IC="${LOSS_WEIGHT_IC:-1.0e-3}"
LOG_EVERY_EPOCH="${LOG_EVERY_EPOCH:-1}"
GRADIENT_TELEMETRY="${GRADIENT_TELEMETRY:-false}"

WANDB_PROJECT="${WANDB_PROJECT:-sm-surrogates-curriculum-hpc}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_GROUP:-${MODEL_FLAG_LOWER}_curriculum_vs_baseline_${TRAJECTORIES}traj_${PINN_EPOCHS}ep_${STAMP}}"
BASELINE_WANDB_NAME="${BASELINE_WANDB_NAME:-baseline_${TRAJECTORIES}traj_${PINN_EPOCHS}ep_s${PINN_SEED}}"
CURRICULUM_WANDB_NAME="${CURRICULUM_WANDB_NAME:-curriculum_${TRAJECTORIES}traj_${PINN_EPOCHS}ep_s${PINN_SEED}}"

SKIP_STAGE1="${SKIP_STAGE1:-false}"
SKIP_STAGE2="${SKIP_STAGE2:-false}"
SKIP_BASELINE="${SKIP_BASELINE:-false}"
SKIP_CURRICULUM="${SKIP_CURRICULUM:-false}"
DRY_RUN="${DRY_RUN:-false}"

BASELINE_RUN_DIR="${BASELINE_RUN_DIR:-${RUNS_ROOT}/baseline}"
CURRICULUM_RUN_DIR="${CURRICULUM_RUN_DIR:-${RUNS_ROOT}/curriculum}"

run_cmd() {
  echo "[curriculum-vs-baseline-hpc] command:"
  printf ' %q' "$@"
  echo
  if [[ "${DRY_RUN}" == "true" ]]; then
    return 0
  fi
  "$@"
}

OPTIMIZER_PHASES_OVERRIDE="pinn.optimizer_phases=[{name:adam_long,optimizer:Adam,lr:${ADAM_LR},epochs:${PINN_EPOCHS},batch_size:${PINN_BATCH_SIZE},shuffle:true,full_batch:false,allow_sampling:${ALLOW_SAMPLING},optimizer_kwargs:{},scheduler:{name:reduce_on_plateau,metric:${SCHEDULER_METRIC},mode:min,factor:${SCHEDULER_FACTOR},patience:${SCHEDULER_PATIENCE},threshold:${SCHEDULER_THRESHOLD},threshold_mode:${SCHEDULER_THRESHOLD_MODE},cooldown:${SCHEDULER_COOLDOWN},min_lr:${SCHEDULER_MIN_LR},eps:${SCHEDULER_EPS}},line_search:null,convergence:null}]"

COMMON_PINN_OVERRIDES=(
  "model.model_flag=${MODEL_FLAG}"
  "model.seed=${PINN_SEED}"
  "dataset.root=${DATASET_ROOT}"
  "pinn.device=${PINN_DEVICE}"
  "pinn.dtype=${PINN_DTYPE}"
  "pinn.hidden_dim=${PINN_HIDDEN_DIM}"
  "pinn.hidden_layers=${PINN_HIDDEN_LAYERS}"
  "pinn.activation=${PINN_ACTIVATION}"
  "pinn.default_batch_size=${PINN_BATCH_SIZE}"
  "pinn.collocation.active_points=${COLLOCATION_ACTIVE_POINTS}"
  "pinn.collocation.multi_pool.total_active_points=${COLLOCATION_ACTIVE_POINTS}"
  "pinn.gradient_telemetry.enabled=${GRADIENT_TELEMETRY}"
  "pinn.loss_weights.data=${LOSS_WEIGHT_DATA}"
  "pinn.loss_weights.dt=${LOSS_WEIGHT_DT}"
  "pinn.loss_weights.physics=${LOSS_WEIGHT_PHYSICS}"
  "pinn.loss_weights.ic=${LOSS_WEIGHT_IC}"
  "logging.log_every_epoch=${LOG_EVERY_EPOCH}"
  "wandb.use=true"
  "wandb.project=${WANDB_PROJECT}"
  "wandb.group=${WANDB_GROUP}"
  "${OPTIMIZER_PHASES_OVERRIDE}"
)

if [[ -n "${WANDB_ENTITY}" ]]; then
  COMMON_PINN_OVERRIDES+=("wandb.entity=${WANDB_ENTITY}")
fi

echo "[curriculum-vs-baseline-hpc] repo_root=${REPO_ROOT}"
echo "[curriculum-vs-baseline-hpc] output_root=${OUTPUT_ROOT}"
echo "[curriculum-vs-baseline-hpc] dataset_root=${DATASET_ROOT}"
echo "[curriculum-vs-baseline-hpc] trajectories=${TRAJECTORIES}"
echo "[curriculum-vs-baseline-hpc] time_horizon=${TIME_HORIZON}"
echo "[curriculum-vs-baseline-hpc] num_points=${NUM_POINTS}"
echo "[curriculum-vs-baseline-hpc] collocation_active_points=${COLLOCATION_ACTIVE_POINTS}"
echo "[curriculum-vs-baseline-hpc] pinn_epochs=${PINN_EPOCHS}"
echo "[curriculum-vs-baseline-hpc] scheduler_patience=${SCHEDULER_PATIENCE}"
echo "[curriculum-vs-baseline-hpc] curriculum_unlock_epochs=${CURRICULUM_UNLOCK_EPOCHS}"
echo "[curriculum-vs-baseline-hpc] wandb_project=${WANDB_PROJECT}"
echo "[curriculum-vs-baseline-hpc] wandb_group=${WANDB_GROUP}"

if [[ "${SKIP_STAGE1}" != "true" ]]; then
  run_cmd \
    "${PYTHON_BIN}" 00_create_dataset.py \
    "preset=${PRESET}" \
    "+method=${METHOD}" \
    "dirs.dataset_dir=${DATA_DIR}" \
    "model.model_flag=${MODEL_FLAG}" \
    "model.ic_num_samples=${TRAJECTORIES}" \
    "time=${TIME_HORIZON}" \
    "num_of_points=${NUM_POINTS}" \
    2>&1 | tee "${LOG_ROOT}/stage1_dataset.log"
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
    "+pinn.curriculum.num_bins=${CURRICULUM_NUM_BINS}" \
    2>&1 | tee "${LOG_ROOT}/stage2_preprocess.log"
fi

if [[ "${SKIP_BASELINE}" != "true" ]]; then
  run_cmd \
    "${PYTHON_BIN}" 20_run_pinn.py \
    "${COMMON_PINN_OVERRIDES[@]}" \
    "pinn.run_dir=${BASELINE_RUN_DIR}" \
    "pinn.curriculum.enabled=false" \
    "wandb.name=${BASELINE_WANDB_NAME}" \
    "wandb.tags=[hpc,baseline,curriculum_compare,${MODEL_FLAG_LOWER},adam${PINN_EPOCHS},plateau]" \
    2>&1 | tee "${LOG_ROOT}/baseline.log"
fi

if [[ "${SKIP_CURRICULUM}" != "true" ]]; then
  run_cmd \
    "${PYTHON_BIN}" 20_run_pinn.py \
    "${COMMON_PINN_OVERRIDES[@]}" \
    "pinn.run_dir=${CURRICULUM_RUN_DIR}" \
    "pinn.curriculum.enabled=true" \
    "pinn.curriculum.num_bins=${CURRICULUM_NUM_BINS}" \
    "pinn.curriculum.unlock_epochs=${CURRICULUM_UNLOCK_EPOCHS}" \
    "pinn.curriculum.balance_active_bins=${CURRICULUM_BALANCE_ACTIVE_BINS}" \
    "wandb.name=${CURRICULUM_WANDB_NAME}" \
    "wandb.tags=[hpc,curriculum,curriculum_compare,${MODEL_FLAG_LOWER},adam${PINN_EPOCHS},plateau]" \
    2>&1 | tee "${LOG_ROOT}/curriculum.log"
fi

echo "[curriculum-vs-baseline-hpc] Complete."
echo "[curriculum-vs-baseline-hpc] baseline_run_dir=${BASELINE_RUN_DIR}"
echo "[curriculum-vs-baseline-hpc] curriculum_run_dir=${CURRICULUM_RUN_DIR}"
echo "[curriculum-vs-baseline-hpc] logs=${LOG_ROOT}"
