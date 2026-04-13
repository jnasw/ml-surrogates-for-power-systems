#!/usr/bin/env bash
#BSUB -J optimizer_phase_warmup_resume
#BSUB -q gpua100
#BSUB -n 1
#BSUB -W 24:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -oo outputs/lsf_logs/optimizer_phase_warmup_resume.%J.out
#BSUB -eo outputs/lsf_logs/optimizer_phase_warmup_resume.%J.err

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
  echo "[ERROR] submit with PYTHON_BIN=/path/to/venv/bin/python bsub < tools/pinn/jobs/run_multiphase_optimizer_cost_v1.lsf.sh"
  exit 1
fi

mkdir -p outputs/lsf_logs

OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/pinn/optimizer_phase_experiment/optimizer_cost_v1_warmup300_resume}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/outputs/pinn/optimizer_phase_experiment/optimizer_cost_v1_warmup300/dataset_pipeline/data/SM4/dataset_v1}"
MODEL_FLAG="${MODEL_FLAG:-SM4}"
PINN_DEVICE="${PINN_DEVICE:-cuda}"
PINN_DTYPE="${PINN_DTYPE:-float64}"
WANDB_PROJECT="${WANDB_PROJECT:-sm-surrogates-pinn-optimizer-cost-v1}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-optimizer_cost_v1_warmup300_resume}"
LOG_EVERY_EPOCH="${LOG_EVERY_EPOCH:-1}"
GRADIENT_TELEMETRY="${GRADIENT_TELEMETRY:-false}"
PHASE_SEQUENCES="${PHASE_SEQUENCES:-Adam:300;SOAP:2700|Adam:300;LBFGS:2700|Adam:300;BFGS:2700|Adam:300;SSBFGS:2700|Adam:300;SSBroyden:2700}"

export OUTPUT_ROOT
export DATASET_ROOT
export MODEL_FLAG
export PINN_DEVICE
export PINN_DTYPE
export WANDB_PROJECT
export EXPERIMENT_TAG
export LOG_EVERY_EPOCH
export GRADIENT_TELEMETRY
export PHASE_SEQUENCES

source "${REPO_ROOT}/tools/pinn/jobs/run_multiphase_experiment.lsf.sh"
