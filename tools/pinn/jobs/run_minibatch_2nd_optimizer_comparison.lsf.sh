#!/usr/bin/env bash
#BSUB -J pinn_minibatch_2nd_opt
#BSUB -q gpua100
#BSUB -n 1
#BSUB -W 36:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -oo outputs/lsf_logs/pinn_minibatch_2nd_optimizer.%J.out
#BSUB -eo outputs/lsf_logs/pinn_minibatch_2nd_optimizer.%J.err

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
  echo "[ERROR] submit with PYTHON_BIN=/path/to/venv/bin/python bsub < tools/pinn/jobs/run_minibatch_2nd_optimizer_comparison.lsf.sh"
  exit 1
fi

mkdir -p outputs/lsf_logs

STAMP="$(date +%Y%m%d_%H%M%S)"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-pinn_minibatch_2nd_optimizer_${STAMP}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/pinn_hpc_experiments/${EXPERIMENT_TAG}}"
DATASET_ROOT="${DATASET_ROOT:-}"
MODEL_FLAG="${MODEL_FLAG:-SM4}"
DATASET_SEED="${DATASET_SEED:-s01}"
PRESET="${PRESET:-default}"
PINN_BUDGET="${PINN_BUDGET:-b256}"

PINN_SEED="${PINN_SEED:-37}"
PINN_DEVICE="${PINN_DEVICE:-cuda}"
PINN_DTYPE="${PINN_DTYPE:-float64}"
PINN_HIDDEN_DIM="${PINN_HIDDEN_DIM:-64}"
PINN_HIDDEN_LAYERS="${PINN_HIDDEN_LAYERS:-4}"
PINN_ACTIVATION="${PINN_ACTIVATION:-tanh}"
PINN_BATCH_SIZE="${PINN_BATCH_SIZE:-1024}"
PINN_EPOCHS="${PINN_EPOCHS:-3000}"

ADAM_LR="${ADAM_LR:-1e-3}"
SSSBFGS_LR="${SSSBFGS_LR:-5e-2}"
SSSBROYDEN_LR="${SSSBROYDEN_LR:-5e-2}"
SSBROYDEN_LR="${SSBROYDEN_LR:-1.0}"

SCHEDULER_PATIENCE="${SCHEDULER_PATIENCE:-20}"
SCHEDULER_FACTOR="${SCHEDULER_FACTOR:-0.5}"
SCHEDULER_THRESHOLD="${SCHEDULER_THRESHOLD:-0.0001}"
SCHEDULER_THRESHOLD_MODE="${SCHEDULER_THRESHOLD_MODE:-rel}"
SCHEDULER_COOLDOWN="${SCHEDULER_COOLDOWN:-0}"
SCHEDULER_MIN_LR="${SCHEDULER_MIN_LR:-1.0e-6}"
SCHEDULER_EPS="${SCHEDULER_EPS:-1.0e-8}"
SCHEDULER_METRIC="${SCHEDULER_METRIC:-val_total_loss}"

STOCHASTIC_CURVATURE_THRESHOLD="${STOCHASTIC_CURVATURE_THRESHOLD:-1.0e-6}"
STOCHASTIC_INIT_HESSIAN_SCALE="${STOCHASTIC_INIT_HESSIAN_SCALE:-1.0e-1}"

WANDB_PROJECT="${WANDB_PROJECT:-pinn-minibatch-2nd-optimizer}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_GROUP:-${EXPERIMENT_TAG}}"
LOG_EVERY_EPOCH="${LOG_EVERY_EPOCH:-1}"
GRADIENT_TELEMETRY="${GRADIENT_TELEMETRY:-false}"
LOSS_WEIGHT_DATA="${LOSS_WEIGHT_DATA:-1.0}"
LOSS_WEIGHT_DT="${LOSS_WEIGHT_DT:-1.0e-4}"
LOSS_WEIGHT_PHYSICS="${LOSS_WEIGHT_PHYSICS:-1.0e-4}"
LOSS_WEIGHT_IC="${LOSS_WEIGHT_IC:-1.0e-3}"
STAGE1_OVERRIDES="${STAGE1_OVERRIDES:-}"
STAGE2_OVERRIDES="${STAGE2_OVERRIDES:-time=0.05|num_of_points=20}"

if [[ -z "${DATASET_ROOT}" ]]; then
  DATASET_PIPELINE_ROOT="${OUTPUT_ROOT}/dataset_pipeline"
  DATASET_CMD=(
    "${PYTHON_BIN}"
    -m src.pipeline.run_experiment
    --method qbc_deep_ensemble
    --budget "${PINN_BUDGET}"
    --dataset-seed "${DATASET_SEED}"
    --preset "${PRESET}"
    --experiment-id "${EXPERIMENT_TAG}"
    --model-flag "${MODEL_FLAG}"
    --run-root "${DATASET_PIPELINE_ROOT}"
    --skip-baseline
  )
  if [[ -n "${STAGE1_OVERRIDES}" ]]; then
    IFS='|' read -r -a _stage1_override_array <<< "${STAGE1_OVERRIDES}"
    for override in "${_stage1_override_array[@]}"; do
      if [[ -n "${override}" ]]; then
        DATASET_CMD+=(--stage1-override "${override}")
      fi
    done
  fi
  if [[ -n "${STAGE2_OVERRIDES}" ]]; then
    IFS='|' read -r -a _stage2_override_array <<< "${STAGE2_OVERRIDES}"
    for override in "${_stage2_override_array[@]}"; do
      if [[ -n "${override}" ]]; then
        DATASET_CMD+=(--stage2-override "${override}")
      fi
    done
  fi

  echo "[pinn-minibatch-2nd-optimizer] dataset command=${DATASET_CMD[*]}"
  "${DATASET_CMD[@]}"
  DATASET_ROOT="${DATASET_PIPELINE_ROOT}/data/${MODEL_FLAG}/dataset_v1"
fi

COMMON_ARGS=(
  "model.model_flag=${MODEL_FLAG}"
  "model.seed=${PINN_SEED}"
  "dataset.root=${DATASET_ROOT}"
  "pinn.device=${PINN_DEVICE}"
  "pinn.dtype=${PINN_DTYPE}"
  "pinn.hidden_dim=${PINN_HIDDEN_DIM}"
  "pinn.hidden_layers=${PINN_HIDDEN_LAYERS}"
  "pinn.activation=${PINN_ACTIVATION}"
  "pinn.default_batch_size=${PINN_BATCH_SIZE}"
  "pinn.supervised_sampling.enabled=false"
  "pinn.collocation_sampling.enabled=false"
  "pinn.gradient_telemetry.enabled=${GRADIENT_TELEMETRY}"
  "pinn.loss_weights.data=${LOSS_WEIGHT_DATA}"
  "pinn.loss_weights.dt=${LOSS_WEIGHT_DT}"
  "pinn.loss_weights.physics=${LOSS_WEIGHT_PHYSICS}"
  "pinn.loss_weights.ic=${LOSS_WEIGHT_IC}"
  "logging.log_every_epoch=${LOG_EVERY_EPOCH}"
  "wandb.use=true"
  "wandb.project=${WANDB_PROJECT}"
  "wandb.group=${WANDB_GROUP}"
)

if [[ -n "${WANDB_ENTITY}" ]]; then
  COMMON_ARGS+=("wandb.entity=${WANDB_ENTITY}")
fi

run_variant() {
  local run_name="$1"
  local optimizer_override="$2"
  local tags="$3"
  local run_dir="${OUTPUT_ROOT}/runs/${run_name}"

  local cmd=(
    "${PYTHON_BIN}"
    20_run_pinn.py
    "${COMMON_ARGS[@]}"
    "pinn.run_dir=${run_dir}"
    "wandb.name=${run_name}"
    "wandb.tags=${tags}"
    "${optimizer_override}"
  )

  echo "[pinn-minibatch-2nd-optimizer] run_name=${run_name}"
  echo "[pinn-minibatch-2nd-optimizer] command=${cmd[*]}"
  "${cmd[@]}"
}

ADAM_SCHED_OVERRIDE="pinn.optimizer_phases=[{name:adam_sched,optimizer:Adam,lr:${ADAM_LR},epochs:${PINN_EPOCHS},batch_size:${PINN_BATCH_SIZE},shuffle:true,full_batch:false,allow_sampling:false,optimizer_kwargs:{},scheduler:{name:reduce_on_plateau,metric:${SCHEDULER_METRIC},mode:min,factor:${SCHEDULER_FACTOR},patience:${SCHEDULER_PATIENCE},threshold:${SCHEDULER_THRESHOLD},threshold_mode:${SCHEDULER_THRESHOLD_MODE},cooldown:${SCHEDULER_COOLDOWN},min_lr:${SCHEDULER_MIN_LR},eps:${SCHEDULER_EPS}},line_search:null,convergence:null}]"
SSSBFGS_OVERRIDE="pinn.optimizer_phases=[{name:sssbfgs,optimizer:sSSBFGS,lr:${SSSBFGS_LR},epochs:${PINN_EPOCHS},batch_size:${PINN_BATCH_SIZE},shuffle:true,full_batch:false,allow_sampling:false,optimizer_kwargs:{curvature_threshold:${STOCHASTIC_CURVATURE_THRESHOLD},init_hessian_scale:${STOCHASTIC_INIT_HESSIAN_SCALE},tau_strategy:al_baali},line_search:null,convergence:null}]"
SSSBROYDEN_OVERRIDE="pinn.optimizer_phases=[{name:sssbroyden,optimizer:sSSBroyden,lr:${SSSBROYDEN_LR},epochs:${PINN_EPOCHS},batch_size:${PINN_BATCH_SIZE},shuffle:true,full_batch:false,allow_sampling:false,optimizer_kwargs:{curvature_threshold:${STOCHASTIC_CURVATURE_THRESHOLD},init_hessian_scale:${STOCHASTIC_INIT_HESSIAN_SCALE},tau_strategy:paper_default,phi_strategy:paper_default},line_search:null,convergence:null}]"
SSBROYDEN_OVERRIDE="pinn.optimizer_phases=[{name:ssbroyden,optimizer:SSBroyden,lr:${SSBROYDEN_LR},epochs:${PINN_EPOCHS},batch_size:null,shuffle:false,full_batch:true,allow_sampling:false,optimizer_kwargs:{tau_strategy:paper_default,phi_strategy:paper_default},line_search:{name:strong_wolfe},convergence:null}]"

echo "[pinn-minibatch-2nd-optimizer] repo_root=${REPO_ROOT}"
echo "[pinn-minibatch-2nd-optimizer] output_root=${OUTPUT_ROOT}"
echo "[pinn-minibatch-2nd-optimizer] dataset_root=${DATASET_ROOT}"
echo "[pinn-minibatch-2nd-optimizer] wandb_project=${WANDB_PROJECT}"
echo "[pinn-minibatch-2nd-optimizer] wandb_group=${WANDB_GROUP}"

run_variant "sssbroyden_3000" "${SSSBROYDEN_OVERRIDE}" "[minibatch_2nd_optimizer,sssbroyden,3000epochs,${MODEL_FLAG,,}]"
run_variant "sssbfgs_3000" "${SSSBFGS_OVERRIDE}" "[minibatch_2nd_optimizer,sssbfgs,3000epochs,${MODEL_FLAG,,}]"
run_variant "adam_scheduler_3000" "${ADAM_SCHED_OVERRIDE}" "[minibatch_2nd_optimizer,adam,reduce_on_plateau,patience${SCHEDULER_PATIENCE},3000epochs,${MODEL_FLAG,,}]"
run_variant "ssbroyden_3000" "${SSBROYDEN_OVERRIDE}" "[minibatch_2nd_optimizer,ssbroyden,3000epochs,${MODEL_FLAG,,}]"
