#!/usr/bin/env bash
#SBATCH --job-name=sm4-qbc-phase2
#SBATCH --output=outputs/slurm_logs/%x_%A_%a.out
#SBATCH --error=outputs/slurm_logs/%x_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1

set -euo pipefail

: "${MATRIX_PATH:?MATRIX_PATH is required}"
: "${PHASE2_RUN_BASE:?PHASE2_RUN_BASE is required}"
: "${SHARED_TEST_ROOT:?SHARED_TEST_ROOT is required}"

PHASE2_EXPERIMENT_ID="${PHASE2_EXPERIMENT_ID:-thesis_sm4_qbc_phase2}"
PHASE2_PHASE="${PHASE2_PHASE:-qbc_sensitivity}"
MODEL_FLAG="${MODEL_FLAG:-SM4}"
BASELINE_EPOCHS="${BASELINE_EPOCHS:-20}"
SURROGATE_DEVICE="${SURROGATE_DEVICE:-cuda}"
QBC_N_TEST="${QBC_N_TEST:-256}"
TIME_H="${TIME_H:-0.5}"
NUM_POINTS="${NUM_POINTS:-200}"
SHARED_TEST_MAX="${SHARED_TEST_MAX:-2048}"
BASELINE_BATCH_SIZE="${BASELINE_BATCH_SIZE:-128}"

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
LINE_NO=$((TASK_ID + 2))

if [[ ! -f "${MATRIX_PATH}" ]]; then
  echo "[ERROR] Matrix file not found: ${MATRIX_PATH}"
  exit 1
fi

line=$(awk -F'\t' -v n="${LINE_NO}" 'NR==n{print; exit}' "${MATRIX_PATH}")
if [[ -z "${line}" ]]; then
  echo "[ERROR] No matrix row for task ${TASK_ID} (line ${LINE_NO})"
  exit 1
fi

IFS=$'\t' read -r track cfg budget seed geometry_variant policy_overrides geometry_overrides <<< "${line}"

if [[ -z "${track}" || -z "${cfg}" || -z "${budget}" || -z "${seed}" ]]; then
  echo "[ERROR] Malformed matrix row: ${line}"
  exit 1
fi

run_tag="${cfg}__${budget}__${seed}"
if [[ "${track}" == "geometry" ]]; then
  run_tag="${cfg}__${geometry_variant}__${budget}__${seed}"
fi
run_root="${PHASE2_RUN_BASE}/${track}/${run_tag}"

mkdir -p outputs/slurm_logs

echo "[RUN] task=${TASK_ID} track=${track} cfg=${cfg} geometry=${geometry_variant} budget=${budget} seed=${seed}"
echo "[RUN] run_root=${run_root}"

cmd=(
  python run_experiment.py
  --method qbc_deep_ensemble
  --budget "${budget}"
  --seed "${seed}"
  --phase "${PHASE2_PHASE}"
  --experiment-id "${PHASE2_EXPERIMENT_ID}"
  --model-flag "${MODEL_FLAG}"
  --run-root "${run_root}"
  --baseline-epochs "${BASELINE_EPOCHS}"
  --stage1-override "qbc_n_test=${QBC_N_TEST}"
  --stage1-override "surrogate.deterministic=true"
  --stage1-override "surrogate.device=${SURROGATE_DEVICE}"
  --stage1-override "time=${TIME_H}"
  --stage1-override "num_of_points=${NUM_POINTS}"
  --stage2-override "dataset.test_split_mode=shared_dataset"
  --stage2-override "dataset.shared_test_dataset_root=${SHARED_TEST_ROOT}"
  --stage2-override "dataset.shared_test_max_trajectories=${SHARED_TEST_MAX}"
  --stage2-override "dataset.validation_flag=false"
  --stage2-override "dataset.new_coll_points_flag=false"
  --stage2-override "time=${TIME_H}"
  --stage2-override "num_of_points=${NUM_POINTS}"
  --stage3-override "baseline.batch_size=${BASELINE_BATCH_SIZE}"
)

append_override_tokens() {
  local text="$1"
  local toks=()
  if [[ -z "${text}" ]]; then
    return 0
  fi
  IFS=' ' read -r -a toks <<< "${text}"
  for tok in "${toks[@]}"; do
    [[ -z "${tok}" ]] && continue
    cmd+=(--stage1-override "${tok}")
  done
}

append_override_tokens "${policy_overrides}"
append_override_tokens "${geometry_overrides}"

"${cmd[@]}"

echo "[DONE] ${run_tag}"
