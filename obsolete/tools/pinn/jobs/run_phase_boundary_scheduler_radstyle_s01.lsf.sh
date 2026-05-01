#!/usr/bin/env bash
#BSUB -J phase_boundary_sched_rad_s01
#BSUB -q gpua100
#BSUB -n 1
#BSUB -W 36:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -oo outputs/lsf_logs/phase_boundary_scheduler_radstyle_s01.%J.out
#BSUB -eo outputs/lsf_logs/phase_boundary_scheduler_radstyle_s01.%J.err

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
  echo "[ERROR] submit with PYTHON_BIN=/path/to/venv/bin/python bsub < tools/pinn/jobs/run_phase_boundary_scheduler_radstyle_s01.lsf.sh"
  exit 1
fi

mkdir -p outputs/lsf_logs

STAMP="$(date +%Y%m%d_%H%M%S)"
PROFILE="phase_boundary_reference"
DATASET_SEED="s01"
VARIANTS="static_generated,random_r,rad,rar_d,rar_g,vrba_sample_quad,vrba_full_quad_ic_dt"
EXPERIMENT_TAG="phase_boundary_scheduler_radstyle_${DATASET_SEED}_${STAMP}"
OUTPUT_ROOT="${REPO_ROOT}/outputs/pinn_hpc_experiments/${EXPERIMENT_TAG}"
WANDB_PROJECT="sm-surrogates-pinn-collocation-phase-boundary"
OPTIMIZER_PHASES_OVERRIDE="pinn.optimizer_phases=[{name:adam_01,optimizer:Adam,lr:0.0007,epochs:950,batch_size:1024,shuffle:true,full_batch:false,allow_sampling:true,optimizer_kwargs:{},scheduler:{name:reduce_on_plateau,metric:val_total_loss,mode:min,factor:0.5,patience:120,threshold:0.0001,threshold_mode:rel,cooldown:0,min_lr:1.0e-6,eps:1.0e-8},line_search:null,convergence:null},{name:lbfgs_01,optimizer:LBFGS,lr:1.0,epochs:50,batch_size:null,shuffle:false,full_batch:true,allow_sampling:false,optimizer_kwargs:{},scheduler:null,line_search:{name:strong_wolfe},convergence:null},{name:adam_02,optimizer:Adam,lr:0.0007,epochs:950,batch_size:1024,shuffle:true,full_batch:false,allow_sampling:true,optimizer_kwargs:{},scheduler:{name:reduce_on_plateau,metric:val_total_loss,mode:min,factor:0.5,patience:120,threshold:0.0001,threshold_mode:rel,cooldown:0,min_lr:1.0e-6,eps:1.0e-8},line_search:null,convergence:null},{name:lbfgs_02,optimizer:LBFGS,lr:1.0,epochs:50,batch_size:null,shuffle:false,full_batch:true,allow_sampling:false,optimizer_kwargs:{},scheduler:null,line_search:{name:strong_wolfe},convergence:null},{name:adam_03,optimizer:Adam,lr:0.0007,epochs:950,batch_size:1024,shuffle:true,full_batch:false,allow_sampling:true,optimizer_kwargs:{},scheduler:{name:reduce_on_plateau,metric:val_total_loss,mode:min,factor:0.5,patience:120,threshold:0.0001,threshold_mode:rel,cooldown:0,min_lr:1.0e-6,eps:1.0e-8},line_search:null,convergence:null},{name:lbfgs_03,optimizer:LBFGS,lr:1.0,epochs:50,batch_size:null,shuffle:false,full_batch:true,allow_sampling:false,optimizer_kwargs:{},scheduler:null,line_search:{name:strong_wolfe},convergence:null},{name:adam_04,optimizer:Adam,lr:0.0007,epochs:950,batch_size:1024,shuffle:true,full_batch:false,allow_sampling:true,optimizer_kwargs:{},scheduler:{name:reduce_on_plateau,metric:val_total_loss,mode:min,factor:0.5,patience:120,threshold:0.0001,threshold_mode:rel,cooldown:0,min_lr:1.0e-6,eps:1.0e-8},line_search:null,convergence:null},{name:lbfgs_04,optimizer:LBFGS,lr:1.0,epochs:50,batch_size:null,shuffle:false,full_batch:true,allow_sampling:false,optimizer_kwargs:{},scheduler:null,line_search:{name:strong_wolfe},convergence:null},{name:adam_05,optimizer:Adam,lr:0.0007,epochs:950,batch_size:1024,shuffle:true,full_batch:false,allow_sampling:true,optimizer_kwargs:{},scheduler:{name:reduce_on_plateau,metric:val_total_loss,mode:min,factor:0.5,patience:120,threshold:0.0001,threshold_mode:rel,cooldown:0,min_lr:1.0e-6,eps:1.0e-8},line_search:null,convergence:null},{name:lbfgs_05,optimizer:LBFGS,lr:1.0,epochs:50,batch_size:null,shuffle:false,full_batch:true,allow_sampling:false,optimizer_kwargs:{},scheduler:null,line_search:{name:strong_wolfe},convergence:null},{name:ssbroyden_tail,optimizer:SSBroyden,lr:1.0,epochs:1000,batch_size:null,shuffle:false,full_batch:true,allow_sampling:false,optimizer_kwargs:{},scheduler:null,line_search:{name:strong_wolfe},convergence:null}]"
REFRESH_ON_PHASE_START="adam_02,adam_03,adam_04,adam_05"

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
  --wandb-use
)

echo "[phase-boundary-scheduler-radstyle] repo_root=${REPO_ROOT}"
echo "[phase-boundary-scheduler-radstyle] output_root=${OUTPUT_ROOT}"
echo "[phase-boundary-scheduler-radstyle] profile=${PROFILE}"
echo "[phase-boundary-scheduler-radstyle] dataset_seed=${DATASET_SEED}"
echo "[phase-boundary-scheduler-radstyle] variants=${VARIANTS}"
echo "[phase-boundary-scheduler-radstyle] refresh_on_phase_start=${REFRESH_ON_PHASE_START}"
echo "[phase-boundary-scheduler-radstyle] wandb_project=${WANDB_PROJECT}"
echo "[phase-boundary-scheduler-radstyle] command=${CMD[*]}"

"${CMD[@]}"
