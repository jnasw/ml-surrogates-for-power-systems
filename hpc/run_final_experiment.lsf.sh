#!/usr/bin/env bash
#BSUB -J final_pinn
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 08:00
#BSUB -oo hpc/logs/final/final_experiment_%J.out
#BSUB -eo hpc/logs/final/final_experiment_%J.err

set -euo pipefail

# Example submissions (use bsub -env to pass variables; inline VAR=val does not
# propagate to the batch environment on this cluster):
#
#   Dry-run one seed:
#     bsub -env "SEED_LABELS=s01,DRY_RUN=true" \
#       < hpc/run_final_experiment.lsf.sh
#
#   Submit one seed for all models and all final strategies:
#     bsub -env "SEED_LABELS=s01" \
#       < hpc/run_final_experiment.lsf.sh
#
#   Submit a model subset (commas in values — use export + bsub -env all):
#     (export MODELS=SM4,SM6 SEED_LABELS=s01 && \
#       bsub -env "all" < hpc/run_final_experiment.lsf.sh)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "${SCRIPT_DIR}/../src" ]]; then
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
else
  REPO_ROOT="${REPO_ROOT:-${LSB_SUBCWD:-$(pwd)}}"
fi

source "${REPO_ROOT}/hpc/common/lsf_defaults.sh"

QUEUE="${QUEUE:-${QUEUE_GPU}}"
WALLTIME="${WALLTIME:-08:00}"
MEM_GB="${MEM_GB:-${DEFAULT_MEM_GB}}"
N_CORES="${N_CORES:-${DEFAULT_N_CORES}}"
MODELS="${MODELS:-}"
STRATEGIES="${STRATEGIES:-}"
SEED_LABELS="${SEED_LABELS:-}"
REFERENCE_ID_PATTERN="${REFERENCE_ID_PATTERN:-}"
ID_EVAL_ID_PATTERN="${ID_EVAL_ID_PATTERN:-}"
OOD_EVAL_ID_PATTERN="${OOD_EVAL_ID_PATTERN:-}"
NO_ID_EVAL="${NO_ID_EVAL:-false}"
NO_OOD_EVAL="${NO_OOD_EVAL:-false}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-}"
EPOCH_SCALE="${EPOCH_SCALE:-}"
ADAM_LR="${ADAM_LR:-}"
SSBROYDEN_LR="${SSBROYDEN_LR:-}"
ADAM_SCHEDULER="${ADAM_SCHEDULER:-}"
GRADIENT_TELEMETRY="${GRADIENT_TELEMETRY:-}"
CHECKPOINT_FRACTIONS="${CHECKPOINT_FRACTIONS:-}"
SAVE_INIT="${SAVE_INIT:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
DRY_RUN="${DRY_RUN:-false}"

LSF_LOG_DIR="${REPO_ROOT}/hpc/logs/final"
mkdir -p "${LSF_LOG_DIR}"

cd "${REPO_ROOT}"
activate_repo_venv "${REPO_ROOT}"

echo "[hpc] repo_root=${REPO_ROOT}"
echo "[hpc] lsf_log_root=${LSF_LOG_DIR}"
echo "[hpc] queue=${QUEUE} (static #BSUB default: gpua100)"
echo "[hpc] walltime=${WALLTIME} (static #BSUB default: 08:00)"
echo "[hpc] mem_gb=${MEM_GB} (static #BSUB default: 8GB)"
echo "[hpc] n_cores=${N_CORES} (static #BSUB default: 4)"
echo "[hpc] models=${MODELS:-<launcher default>}"
echo "[hpc] strategies=${STRATEGIES:-<launcher default>}"
echo "[hpc] seed_labels=${SEED_LABELS:-<launcher default>}"
echo "[hpc] reference_id_pattern=${REFERENCE_ID_PATTERN:-<launcher default>}"
echo "[hpc] id_eval_id_pattern=${ID_EVAL_ID_PATTERN:-<launcher default>}"
echo "[hpc] ood_eval_id_pattern=${OOD_EVAL_ID_PATTERN:-<launcher default>}"
echo "[hpc] no_id_eval=${NO_ID_EVAL}"
echo "[hpc] no_ood_eval=${NO_OOD_EVAL}"
echo "[hpc] device=${DEVICE}"
echo "[hpc] dtype=${DTYPE:-<launcher default>}"
echo "[hpc] epoch_scale=${EPOCH_SCALE:-<launcher default>}"
echo "[hpc] adam_lr=${ADAM_LR:-<launcher default>}"
echo "[hpc] ssbroyden_lr=${SSBROYDEN_LR:-<launcher default>}"
echo "[hpc] adam_scheduler=${ADAM_SCHEDULER:-<launcher default>}"
echo "[hpc] gradient_telemetry=${GRADIENT_TELEMETRY:-<launcher default>}"
echo "[hpc] checkpoint_fractions=${CHECKPOINT_FRACTIONS:-<launcher default>}"
echo "[hpc] save_init=${SAVE_INIT}"
echo "[hpc] wandb_project=${WANDB_PROJECT:-<launcher default>}"
echo "[hpc] experiment_tag=${EXPERIMENT_TAG:-<timestamp>}"
echo "[hpc] output_root=${OUTPUT_ROOT:-<launcher default>}"
echo "[hpc] dry_run=${DRY_RUN}"

cmd=(
  python3
  -m
  src.experiments.pipeline.run_final_experiment
  --device
  "${DEVICE}"
)

if [[ -n "${MODELS}" ]]; then
  cmd+=(--models "${MODELS}")
fi

if [[ -n "${STRATEGIES}" ]]; then
  cmd+=(--strategies "${STRATEGIES}")
fi

if [[ -n "${SEED_LABELS}" ]]; then
  cmd+=(--seed-labels "${SEED_LABELS}")
fi

if [[ -n "${REFERENCE_ID_PATTERN}" ]]; then
  cmd+=(--reference-id-pattern "${REFERENCE_ID_PATTERN}")
fi

if [[ -n "${ID_EVAL_ID_PATTERN}" ]]; then
  cmd+=(--id-eval-id-pattern "${ID_EVAL_ID_PATTERN}")
fi

if [[ -n "${OOD_EVAL_ID_PATTERN}" ]]; then
  cmd+=(--ood-eval-id-pattern "${OOD_EVAL_ID_PATTERN}")
fi

if [[ "${NO_ID_EVAL}" == "true" ]]; then
  cmd+=(--no-id-eval)
fi

if [[ "${NO_OOD_EVAL}" == "true" ]]; then
  cmd+=(--no-ood-eval)
fi

if [[ -n "${DTYPE}" ]]; then
  cmd+=(--dtype "${DTYPE}")
fi

if [[ -n "${EPOCH_SCALE}" ]]; then
  cmd+=(--epoch-scale "${EPOCH_SCALE}")
fi

if [[ -n "${ADAM_LR}" ]]; then
  cmd+=(--adam-lr "${ADAM_LR}")
fi

if [[ -n "${SSBROYDEN_LR}" ]]; then
  cmd+=(--ssbroyden-lr "${SSBROYDEN_LR}")
fi

if [[ -n "${CHECKPOINT_FRACTIONS}" ]]; then
  cmd+=(--checkpoint-fractions "${CHECKPOINT_FRACTIONS}")
fi

case "${SAVE_INIT}" in
  true|TRUE|1|yes|YES)
    cmd+=(--save-init)
    ;;
  false|FALSE|0|no|NO)
    cmd+=(--no-save-init)
    ;;
  *)
    echo "[hpc] ERROR: SAVE_INIT must be true or false, got '${SAVE_INIT}'." >&2
    exit 2
    ;;
esac

case "${ADAM_SCHEDULER}" in
  true|TRUE|1|yes|YES)
    cmd+=(--adam-scheduler)
    ;;
  false|FALSE|0|no|NO)
    cmd+=(--no-adam-scheduler)
    ;;
  "")
    ;;
  *)
    echo "[hpc] ERROR: ADAM_SCHEDULER must be true or false, got '${ADAM_SCHEDULER}'." >&2
    exit 2
    ;;
esac

case "${GRADIENT_TELEMETRY}" in
  true|TRUE|1|yes|YES)
    cmd+=(--gradient-telemetry)
    ;;
  false|FALSE|0|no|NO)
    cmd+=(--no-gradient-telemetry)
    ;;
  "")
    ;;
  *)
    echo "[hpc] ERROR: GRADIENT_TELEMETRY must be true or false, got '${GRADIENT_TELEMETRY}'." >&2
    exit 2
    ;;
esac

if [[ -n "${WANDB_PROJECT}" ]]; then
  cmd+=(--wandb-project "${WANDB_PROJECT}")
fi

if [[ -n "${WANDB_ENTITY}" ]]; then
  cmd+=(--wandb-entity "${WANDB_ENTITY}")
fi

if [[ -n "${EXPERIMENT_TAG}" ]]; then
  cmd+=(--experiment-tag "${EXPERIMENT_TAG}")
fi

if [[ -n "${OUTPUT_ROOT}" ]]; then
  cmd+=(--output-root "${OUTPUT_ROOT}")
fi

if [[ "${DRY_RUN}" == "true" ]]; then
  cmd+=(--dry-run)
fi

echo "[hpc] command:"
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
