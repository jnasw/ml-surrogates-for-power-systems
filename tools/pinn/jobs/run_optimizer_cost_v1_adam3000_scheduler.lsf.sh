#!/usr/bin/env bash
#BSUB -J optimizer_cost_v1_adam3000_sched
#BSUB -q gpua100
#BSUB -n 1
#BSUB -W 24:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -oo outputs/lsf_logs/optimizer_cost_v1_adam3000_scheduler.%J.out
#BSUB -eo outputs/lsf_logs/optimizer_cost_v1_adam3000_scheduler.%J.err

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
  echo "[ERROR] submit with PYTHON_BIN=/path/to/venv/bin/python bsub < tools/pinn/jobs/run_optimizer_cost_v1_adam3000_scheduler.lsf.sh"
  exit 1
fi

mkdir -p outputs/lsf_logs

OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/pinn/optimizer_phase_experiment/optimizer_cost_v1_adam3000_scheduler}"
RUN_DIR="${RUN_DIR:-${OUTPUT_ROOT}/runs/optimizer_phase_adam3000_scheduler}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/outputs/pinn/optimizer_phase_experiment/optimizer_cost_v1_warmup300/dataset_pipeline/data/SM4/dataset_v1}"
MODEL_FLAG="${MODEL_FLAG:-SM4}"
PINN_SEED="${PINN_SEED:-37}"
PINN_DEVICE="${PINN_DEVICE:-cuda}"
PINN_DTYPE="${PINN_DTYPE:-float64}"
PINN_HIDDEN_DIM="${PINN_HIDDEN_DIM:-64}"
PINN_HIDDEN_LAYERS="${PINN_HIDDEN_LAYERS:-4}"
PINN_ACTIVATION="${PINN_ACTIVATION:-tanh}"
PINN_BATCH_SIZE="${PINN_BATCH_SIZE:-1024}"
ADAM_LR="${ADAM_LR:-1e-3}"
WANDB_PROJECT="${WANDB_PROJECT:-sm-surrogates-pinn-optimizer-cost-v1}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_GROUP:-optimizer_cost_v1_adam3000_scheduler}"
WANDB_NAME="${WANDB_NAME:-optimizer_phase_adam3000_scheduler}"
LOG_EVERY_EPOCH="${LOG_EVERY_EPOCH:-1}"
LOSS_WEIGHT_DATA="${LOSS_WEIGHT_DATA:-1.0}"
LOSS_WEIGHT_DT="${LOSS_WEIGHT_DT:-1.0e-4}"
LOSS_WEIGHT_PHYSICS="${LOSS_WEIGHT_PHYSICS:-1.0e-4}"
LOSS_WEIGHT_IC="${LOSS_WEIGHT_IC:-1.0e-3}"
GRADIENT_TELEMETRY="${GRADIENT_TELEMETRY:-false}"

OPTIMIZER_PHASES_OVERRIDE="pinn.optimizer_phases=[{name:adam_sched,optimizer:Adam,lr:${ADAM_LR},epochs:3000,batch_size:${PINN_BATCH_SIZE},shuffle:true,full_batch:false,allow_sampling:false,optimizer_kwargs:{},scheduler:{name:reduce_on_plateau,metric:val_total_loss,mode:min,factor:0.5,patience:200,threshold:0.0001,threshold_mode:rel,cooldown:0,min_lr:1.0e-6,eps:1.0e-8},line_search:null,convergence:null}]"

CMD=(
  "${PYTHON_BIN}"
  20_run_pinn.py
  "model.model_flag=${MODEL_FLAG}"
  "model.seed=${PINN_SEED}"
  "dataset.root=${DATASET_ROOT}"
  "pinn.run_dir=${RUN_DIR}"
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
  "wandb.name=${WANDB_NAME}"
  "wandb.tags=[optimizer_phase_experiment,adam3000,reduce_on_plateau,sm4]"
  "${OPTIMIZER_PHASES_OVERRIDE}"
)

if [[ -n "${WANDB_ENTITY}" ]]; then
  CMD+=("wandb.entity=${WANDB_ENTITY}")
fi

echo "[optimizer-cost-v1-adam3000-scheduler] repo_root=${REPO_ROOT}"
echo "[optimizer-cost-v1-adam3000-scheduler] output_root=${OUTPUT_ROOT}"
echo "[optimizer-cost-v1-adam3000-scheduler] run_dir=${RUN_DIR}"
echo "[optimizer-cost-v1-adam3000-scheduler] dataset_root=${DATASET_ROOT}"
echo "[optimizer-cost-v1-adam3000-scheduler] wandb_project=${WANDB_PROJECT}"
echo "[optimizer-cost-v1-adam3000-scheduler] command=${CMD[*]}"

"${CMD[@]}"
