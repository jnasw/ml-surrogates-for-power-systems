#!/usr/bin/env bash
#BSUB -J adapt_aug
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 08:00
#BSUB -oo hpc/logs/adaptive_augmentation/adaptive_augmentation_%J.out
#BSUB -eo hpc/logs/adaptive_augmentation/adaptive_augmentation_%J.err

set -euo pipefail

# Example submissions (use bsub -env to pass variables; inline VAR=val does not
# propagate to the batch environment on this cluster):
#
#   Screening dry-run:
#     (export MODE=screening REFERENCE_ID=main_SM4_lhs_b4096_ds01 \
#       SUPERVISED_STRATEGIES=fixed_low,mae_nearest_growth \
#       COLLOCATION_STRATEGIES=static_low,rar_d_growth \
#       SEED_LABELS=s01 DRY_RUN=true && \
#       bsub -env "all" < hpc/adaptive_augmentation/run_adaptive_augmentation.lsf.sh)
#
#   Final run:
#     (export MODE=final REFERENCE_ID=main_SM4_lhs_b4096_ds01 \
#       SUPERVISED_STRATEGIES=fixed_low,random_growth,mae_nearest_growth,fixed_full \
#       COLLOCATION_STRATEGIES=static_low,rar_d_growth \
#       SEED_LABELS=s01,s02,s03,s04,s05 REFRESH_PERIOD_EPOCHS=500 && \
#       bsub -env "all" < hpc/adaptive_augmentation/run_adaptive_augmentation.lsf.sh)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "${SCRIPT_DIR}/../../src" ]]; then
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
else
  # When submitted via `bsub < script`, BASH_SOURCE may not resolve correctly.
  # LSF sets LSB_SUBCWD to the directory where `bsub` was called.
  REPO_ROOT="${REPO_ROOT:-${LSB_SUBCWD:-$(pwd)}}"
fi

source "${REPO_ROOT}/hpc/common/lsf_defaults.sh"

QUEUE="${QUEUE:-${QUEUE_GPU}}"
WALLTIME="${WALLTIME:-08:00}"
MEM_GB="${MEM_GB:-${DEFAULT_MEM_GB}}"
N_CORES="${N_CORES:-${DEFAULT_N_CORES}}"
MODE="${MODE:-screening}"
REFERENCE_ID="${REFERENCE_ID:-}"
DATASET_ROOT="${DATASET_ROOT:-}"
MODEL_FLAG="${MODEL_FLAG:-SM4}"
SEED_LABELS="${SEED_LABELS:-}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-}"
ID_EVAL_ID="${ID_EVAL_ID:-}"
OOD_EVAL_ID="${OOD_EVAL_ID:-}"
NO_OOD_EVAL="${NO_OOD_EVAL:-false}"
EPOCHS="${EPOCHS:-}"
ADAM_LR="${ADAM_LR:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
SUPERVISED_STRATEGIES="${SUPERVISED_STRATEGIES:-}"
COLLOCATION_STRATEGIES="${COLLOCATION_STRATEGIES:-}"
INITIAL_TRAJECTORIES="${INITIAL_TRAJECTORIES:-}"
ADD_TRAJECTORIES="${ADD_TRAJECTORIES:-}"
FINAL_TRAJECTORIES="${FINAL_TRAJECTORIES:-}"
INITIAL_COLLOCATION_POINTS="${INITIAL_COLLOCATION_POINTS:-}"
ADD_COLLOCATION_POINTS="${ADD_COLLOCATION_POINTS:-}"
FINAL_COLLOCATION_POINTS="${FINAL_COLLOCATION_POINTS:-}"
CANDIDATE_COLLOCATION_POINTS="${CANDIDATE_COLLOCATION_POINTS:-}"
REFRESH_PERIOD_EPOCHS="${REFRESH_PERIOD_EPOCHS:-}"
CANDIDATE_BATCH_SIZE="${CANDIDATE_BATCH_SIZE:-}"
COLLOCATION_SAMPLER="${COLLOCATION_SAMPLER:-}"
SCORE_NORM="${SCORE_NORM:-}"
RAR_D_K="${RAR_D_K:-}"
RAR_D_C="${RAR_D_C:-}"
WANDB_PROJECT="${WANDB_PROJECT:-}"
DRY_RUN="${DRY_RUN:-false}"

LSF_LOG_DIR="${REPO_ROOT}/hpc/logs/adaptive_augmentation"
mkdir -p "${LSF_LOG_DIR}"

cd "${REPO_ROOT}"
activate_repo_venv "${REPO_ROOT}"

echo "[hpc] repo_root=${REPO_ROOT}"
echo "[hpc] lsf_log_root=${LSF_LOG_DIR}"
echo "[hpc] queue=${QUEUE} (static #BSUB default: gpua100)"
echo "[hpc] walltime=${WALLTIME} (static #BSUB default: 08:00)"
echo "[hpc] mem_gb=${MEM_GB} (static #BSUB default: 8GB)"
echo "[hpc] n_cores=${N_CORES} (static #BSUB default: 4)"
echo "[hpc] mode=${MODE}"
echo "[hpc] reference_id=${REFERENCE_ID:-<launcher default>}"
echo "[hpc] dataset_root=${DATASET_ROOT:-<none>}"
echo "[hpc] model_flag=${MODEL_FLAG}"
echo "[hpc] seed_labels=${SEED_LABELS:-<mode default>}"
echo "[hpc] device=${DEVICE}"
echo "[hpc] dtype=${DTYPE:-<launcher default>}"
echo "[hpc] id_eval_id=${ID_EVAL_ID:-<none>}"
echo "[hpc] ood_eval_id=${OOD_EVAL_ID:-<launcher default>}"
echo "[hpc] no_ood_eval=${NO_OOD_EVAL}"
echo "[hpc] epochs=${EPOCHS:-<mode default>}"
echo "[hpc] adam_lr=${ADAM_LR:-<launcher default>}"
echo "[hpc] batch_size=${BATCH_SIZE:-<launcher default>}"
echo "[hpc] supervised_strategies=${SUPERVISED_STRATEGIES:-<launcher default>}"
echo "[hpc] collocation_strategies=${COLLOCATION_STRATEGIES:-<launcher default>}"
echo "[hpc] initial_trajectories=${INITIAL_TRAJECTORIES:-<launcher default>}"
echo "[hpc] add_trajectories=${ADD_TRAJECTORIES:-<launcher default>}"
echo "[hpc] final_trajectories=${FINAL_TRAJECTORIES:-<launcher default>}"
echo "[hpc] initial_collocation_points=${INITIAL_COLLOCATION_POINTS:-<launcher default>}"
echo "[hpc] add_collocation_points=${ADD_COLLOCATION_POINTS:-<launcher default>}"
echo "[hpc] final_collocation_points=${FINAL_COLLOCATION_POINTS:-<launcher default>}"
echo "[hpc] candidate_collocation_points=${CANDIDATE_COLLOCATION_POINTS:-<launcher default>}"
echo "[hpc] refresh_period_epochs=${REFRESH_PERIOD_EPOCHS:-<mode default>}"
echo "[hpc] candidate_batch_size=${CANDIDATE_BATCH_SIZE:-<launcher default>}"
echo "[hpc] collocation_sampler=${COLLOCATION_SAMPLER:-<launcher default>}"
echo "[hpc] score_norm=${SCORE_NORM:-<launcher default>}"
echo "[hpc] rar_d_k=${RAR_D_K:-<launcher default>}"
echo "[hpc] rar_d_c=${RAR_D_C:-<launcher default>}"
echo "[hpc] wandb_project=${WANDB_PROJECT:-<mode default>}"
echo "[hpc] dry_run=${DRY_RUN}"

cmd=(
  python3
  -m
  src.experiments.pipeline.run_adaptive_augmentation
  --mode
  "${MODE}"
  --model-flag
  "${MODEL_FLAG}"
  --device
  "${DEVICE}"
)

if [[ -n "${REFERENCE_ID}" ]]; then
  cmd+=(--reference-id "${REFERENCE_ID}")
fi

if [[ -n "${DATASET_ROOT}" ]]; then
  cmd+=(--dataset-root "${DATASET_ROOT}")
fi

if [[ -n "${SEED_LABELS}" ]]; then
  cmd+=(--seed-labels "${SEED_LABELS}")
fi

if [[ -n "${DTYPE}" ]]; then
  cmd+=(--dtype "${DTYPE}")
fi

if [[ -n "${ID_EVAL_ID}" ]]; then
  cmd+=(--id-eval-id "${ID_EVAL_ID}")
fi

if [[ -n "${OOD_EVAL_ID}" ]]; then
  cmd+=(--ood-eval-id "${OOD_EVAL_ID}")
fi

if [[ "${NO_OOD_EVAL}" == "true" ]]; then
  cmd+=(--no-ood-eval)
fi

if [[ -n "${EPOCHS}" ]]; then
  cmd+=(--epochs "${EPOCHS}")
fi

if [[ -n "${ADAM_LR}" ]]; then
  cmd+=(--adam-lr "${ADAM_LR}")
fi

if [[ -n "${BATCH_SIZE}" ]]; then
  cmd+=(--batch-size "${BATCH_SIZE}")
fi

if [[ -n "${SUPERVISED_STRATEGIES}" ]]; then
  cmd+=(--supervised-strategies "${SUPERVISED_STRATEGIES}")
fi

if [[ -n "${COLLOCATION_STRATEGIES}" ]]; then
  cmd+=(--collocation-strategies "${COLLOCATION_STRATEGIES}")
fi

if [[ -n "${INITIAL_TRAJECTORIES}" ]]; then
  cmd+=(--initial-trajectories "${INITIAL_TRAJECTORIES}")
fi

if [[ -n "${ADD_TRAJECTORIES}" ]]; then
  cmd+=(--add-trajectories "${ADD_TRAJECTORIES}")
fi

if [[ -n "${FINAL_TRAJECTORIES}" ]]; then
  cmd+=(--final-trajectories "${FINAL_TRAJECTORIES}")
fi

if [[ -n "${INITIAL_COLLOCATION_POINTS}" ]]; then
  cmd+=(--initial-collocation-points "${INITIAL_COLLOCATION_POINTS}")
fi

if [[ -n "${ADD_COLLOCATION_POINTS}" ]]; then
  cmd+=(--add-collocation-points "${ADD_COLLOCATION_POINTS}")
fi

if [[ -n "${FINAL_COLLOCATION_POINTS}" ]]; then
  cmd+=(--final-collocation-points "${FINAL_COLLOCATION_POINTS}")
fi

if [[ -n "${CANDIDATE_COLLOCATION_POINTS}" ]]; then
  cmd+=(--candidate-collocation-points "${CANDIDATE_COLLOCATION_POINTS}")
fi

if [[ -n "${REFRESH_PERIOD_EPOCHS}" ]]; then
  cmd+=(--refresh-period-epochs "${REFRESH_PERIOD_EPOCHS}")
fi

if [[ -n "${CANDIDATE_BATCH_SIZE}" ]]; then
  cmd+=(--candidate-batch-size "${CANDIDATE_BATCH_SIZE}")
fi

if [[ -n "${COLLOCATION_SAMPLER}" ]]; then
  cmd+=(--collocation-sampler "${COLLOCATION_SAMPLER}")
fi

if [[ -n "${SCORE_NORM}" ]]; then
  cmd+=(--score-norm "${SCORE_NORM}")
fi

if [[ -n "${RAR_D_K}" ]]; then
  cmd+=(--rar-d-k "${RAR_D_K}")
fi

if [[ -n "${RAR_D_C}" ]]; then
  cmd+=(--rar-d-c "${RAR_D_C}")
fi

if [[ -n "${WANDB_PROJECT}" ]]; then
  cmd+=(--wandb-project "${WANDB_PROJECT}")
fi

if [[ "${DRY_RUN}" == "true" ]]; then
  cmd+=(--dry-run)
fi

echo "[hpc] command:"
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
