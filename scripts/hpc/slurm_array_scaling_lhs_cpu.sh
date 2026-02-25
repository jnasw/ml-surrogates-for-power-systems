#!/usr/bin/env bash
#SBATCH --job-name=sm4-scaling-lhs
#SBATCH --output=outputs/slurm_logs/%x_%A_%a.out
#SBATCH --error=outputs/slurm_logs/%x_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --array=0-5%3

set -euo pipefail

# Run matrix (6 jobs):
# method: lhs_static
# budgets: b8192, b16384
# seeds: s01,s02,s03

BUDGETS=("b8192" "b16384")
SEEDS=("s01" "s02" "s03")

N_BUDGETS=${#BUDGETS[@]}
N_SEEDS=${#SEEDS[@]}
TOTAL=$((N_BUDGETS * N_SEEDS))

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  TASK_ID=${SLURM_ARRAY_TASK_ID}
elif [[ -n "${LSB_JOBINDEX:-}" ]]; then
  TASK_ID=$((LSB_JOBINDEX - 1))
else
  TASK_ID=0
fi
if (( TASK_ID < 0 || TASK_ID >= TOTAL )); then
  echo "Invalid task id ${TASK_ID}; expected 0..$((TOTAL - 1))"
  exit 1
fi

budget_idx=$(( TASK_ID / N_SEEDS ))
seed_idx=$(( TASK_ID % N_SEEDS ))

BUDGET=${BUDGETS[$budget_idx]}
SEED=${SEEDS[$seed_idx]}

RUN_ID="lhs_static_${BUDGET}_${SEED}"
echo "[RUN] ${RUN_ID} (task=${TASK_ID}/${TOTAL})"

mkdir -p outputs/slurm_logs

python create_dataset.py \
  +exp=base \
  +exp/phase=scaling \
  +exp/method=lhs_static \
  +exp/budget/lhs="${BUDGET}" \
  +exp/seed="${SEED}" \
  hydra.run.dir="${PWD}/outputs/experiments/hydra/${RUN_ID}" \
  hydra.job.chdir=false
