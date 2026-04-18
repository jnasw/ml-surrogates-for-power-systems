#!/usr/bin/env bash
#BSUB -J phase_boundary_sched_loops_s01
#BSUB -q gpua100
#BSUB -n 1
#BSUB -W 36:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -oo outputs/lsf_logs/phase_boundary_scheduler_loops_s01.%J.out
#BSUB -eo outputs/lsf_logs/phase_boundary_scheduler_loops_s01.%J.err

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
  echo "[ERROR] submit with PYTHON_BIN=/path/to/venv/bin/python bsub < tools/pinn/jobs/run_phase_boundary_scheduler_loops_s01.lsf.sh"
  exit 1
fi

mkdir -p outputs/lsf_logs

STAMP="$(date +%Y%m%d_%H%M%S)"
PROFILE="${PROFILE:-phase_boundary_reference}"
DATASET_SEED="${DATASET_SEED:-s01}"
VARIANTS="${VARIANTS:-rar_g,rar_d,rad,random_r}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-phase_boundary_scheduler_loops_${DATASET_SEED}_${STAMP}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/pinn_hpc_experiments/${EXPERIMENT_TAG}}"
WANDB_PROJECT="${WANDB_PROJECT:-sm-surrogates-pinn-collocation-phase-boundary}"
WANDB_USE="${WANDB_USE:-true}"
DATASET_ROOT="${DATASET_ROOT:-}"

PINN_BATCH_SIZE="${PINN_BATCH_SIZE:-1024}"
ADAM_LR="${ADAM_LR:-0.001}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-1000}"
LOOP_COUNT="${LOOP_COUNT:-5}"
LOOP_ADAM_EPOCHS="${LOOP_ADAM_EPOCHS:-950}"
LOOP_LBFGS_EPOCHS="${LOOP_LBFGS_EPOCHS:-50}"
LBFGS_LR="${LBFGS_LR:-1.0}"

SCHEDULER_METRIC="${SCHEDULER_METRIC:-val_total_loss}"
SCHEDULER_FACTOR="${SCHEDULER_FACTOR:-0.5}"
SCHEDULER_PATIENCE="${SCHEDULER_PATIENCE:-120}"
SCHEDULER_THRESHOLD="${SCHEDULER_THRESHOLD:-0.0001}"
SCHEDULER_THRESHOLD_MODE="${SCHEDULER_THRESHOLD_MODE:-rel}"
SCHEDULER_COOLDOWN="${SCHEDULER_COOLDOWN:-0}"
SCHEDULER_MIN_LR="${SCHEDULER_MIN_LR:-1.0e-6}"
SCHEDULER_EPS="${SCHEDULER_EPS:-1.0e-8}"

refresh_phases=()
optimizer_phases="{name:adam_warmup,optimizer:Adam,lr:${ADAM_LR},epochs:${WARMUP_EPOCHS},batch_size:${PINN_BATCH_SIZE},shuffle:true,full_batch:false,allow_sampling:true,optimizer_kwargs:{},scheduler:{name:reduce_on_plateau,metric:${SCHEDULER_METRIC},mode:min,factor:${SCHEDULER_FACTOR},patience:${SCHEDULER_PATIENCE},threshold:${SCHEDULER_THRESHOLD},threshold_mode:${SCHEDULER_THRESHOLD_MODE},cooldown:${SCHEDULER_COOLDOWN},min_lr:${SCHEDULER_MIN_LR},eps:${SCHEDULER_EPS}},line_search:null,convergence:null}"

for ((idx=1; idx<=LOOP_COUNT; idx++)); do
  phase_id="$(printf "%02d" "${idx}")"
  adam_name="adam_${phase_id}"
  lbfgs_name="lbfgs_${phase_id}"
  refresh_phases+=("${adam_name}")
  optimizer_phases+=",{name:${adam_name},optimizer:Adam,lr:${ADAM_LR},epochs:${LOOP_ADAM_EPOCHS},batch_size:${PINN_BATCH_SIZE},shuffle:true,full_batch:false,allow_sampling:true,optimizer_kwargs:{},scheduler:{name:reduce_on_plateau,metric:${SCHEDULER_METRIC},mode:min,factor:${SCHEDULER_FACTOR},patience:${SCHEDULER_PATIENCE},threshold:${SCHEDULER_THRESHOLD},threshold_mode:${SCHEDULER_THRESHOLD_MODE},cooldown:${SCHEDULER_COOLDOWN},min_lr:${SCHEDULER_MIN_LR},eps:${SCHEDULER_EPS}},line_search:null,convergence:null}"
  optimizer_phases+=",{name:${lbfgs_name},optimizer:LBFGS,lr:${LBFGS_LR},epochs:${LOOP_LBFGS_EPOCHS},batch_size:null,shuffle:false,full_batch:true,allow_sampling:false,optimizer_kwargs:{},scheduler:null,line_search:{name:strong_wolfe},convergence:null}"
done

OPTIMIZER_PHASES_OVERRIDE="pinn.optimizer_phases=[${optimizer_phases}]"
REFRESH_ON_PHASE_START="$(IFS=,; echo "${refresh_phases[*]}")"

CMD=(
  "${PYTHON_BIN}"
  tools/pinn/run_collocation_comparison.py
  --profile "${PROFILE}"
  --variants "${VARIANTS}"
  --experiment-tag "${EXPERIMENT_TAG}"
  --output-root "${OUTPUT_ROOT}"
  --model-flag "SM4"
  --dataset-seed "${DATASET_SEED}"
  --optimizer-phases-override "${OPTIMIZER_PHASES_OVERRIDE}"
  --refresh-mode "phase_boundary"
  --refresh-on-phase-start "${REFRESH_ON_PHASE_START}"
  --wandb-project "${WANDB_PROJECT}"
)

if [[ -n "${DATASET_ROOT}" ]]; then
  CMD+=(--dataset-root "${DATASET_ROOT}")
fi

if [[ "${WANDB_USE}" == "true" ]]; then
  CMD+=(--wandb-use)
else
  CMD+=(--no-wandb-use)
fi

echo "[phase-boundary-scheduler-loops] repo_root=${REPO_ROOT}"
echo "[phase-boundary-scheduler-loops] output_root=${OUTPUT_ROOT}"
echo "[phase-boundary-scheduler-loops] profile=${PROFILE}"
echo "[phase-boundary-scheduler-loops] dataset_seed=${DATASET_SEED}"
echo "[phase-boundary-scheduler-loops] dataset_root=${DATASET_ROOT:-<generated from profile>}"
echo "[phase-boundary-scheduler-loops] variants=${VARIANTS}"
echo "[phase-boundary-scheduler-loops] refresh_on_phase_start=${REFRESH_ON_PHASE_START}"
echo "[phase-boundary-scheduler-loops] optimizer_phases_override=${OPTIMIZER_PHASES_OVERRIDE}"
echo "[phase-boundary-scheduler-loops] wandb_project=${WANDB_PROJECT}"
echo "[phase-boundary-scheduler-loops] command=${CMD[*]}"

"${CMD[@]}"
