#!/usr/bin/env bash
#SBATCH --job-name=sm4-scaling
#SBATCH --output=outputs/slurm_logs/%x_%A_%a.out
#SBATCH --error=outputs/slurm_logs/%x_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --array=0-11

set -euo pipefail

# Run matrix (12 jobs):
# methods: lhs_static, qbc_deep_ensemble
# budgets: b8192, b16384
# seeds: s01,s02,s03

METHODS=("lhs_static" "qbc_deep_ensemble")
BUDGETS=("b8192" "b16384")
SEEDS=("s01" "s02" "s03")

N_METHODS=${#METHODS[@]}
N_BUDGETS=${#BUDGETS[@]}
N_SEEDS=${#SEEDS[@]}
TOTAL=$((N_METHODS * N_BUDGETS * N_SEEDS))

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if (( TASK_ID < 0 || TASK_ID >= TOTAL )); then
  echo "Invalid task id ${TASK_ID}; expected 0..$((TOTAL - 1))"
  exit 1
fi

method_idx=$(( TASK_ID / (N_BUDGETS * N_SEEDS) ))
rem=$(( TASK_ID % (N_BUDGETS * N_SEEDS) ))
budget_idx=$(( rem / N_SEEDS ))
seed_idx=$(( rem % N_SEEDS ))

METHOD=${METHODS[$method_idx]}
BUDGET=${BUDGETS[$budget_idx]}
SEED=${SEEDS[$seed_idx]}

RUN_ID="${METHOD}_${BUDGET}_${SEED}"
echo "[RUN] ${RUN_ID} (task=${TASK_ID}/${TOTAL})"

mkdir -p outputs/slurm_logs

python run_experiment.py \
  --method "${METHOD}" \
  --budget "${BUDGET}" \
  --seed "${SEED}" \
  --phase scaling \
  --experiment-id thesis_sm4_v1
