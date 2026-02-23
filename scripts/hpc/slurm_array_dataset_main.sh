#!/usr/bin/env bash
#SBATCH --job-name=sm4-main
#SBATCH --output=outputs/slurm_logs/%x_%A_%a.out
#SBATCH --error=outputs/slurm_logs/%x_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --array=0-29

set -euo pipefail

# Run matrix:
# methods: lhs_static, qbc
# budgets: b256, b1024, b4096
# seeds:   s01..s05

METHODS=("lhs_static" "qbc")
BUDGETS=("b256" "b1024" "b4096")
SEEDS=("s01" "s02" "s03" "s04" "s05")

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

if [[ "${METHOD}" == "lhs_static" ]]; then
  BUDGET_GROUP="lhs"
else
  BUDGET_GROUP="qbc"
fi

RUN_ID="${METHOD}_${BUDGET}_${SEED}"
echo "[RUN] ${RUN_ID} (task=${TASK_ID}/${TOTAL})"

mkdir -p outputs/slurm_logs

# Optional: activate your env here
# source ~/.bashrc
# conda activate <env-name>

python create_dataset.py \
  +exp=base \
  +exp/phase=main \
  +exp/method="${METHOD}" \
  +exp/budget/"${BUDGET_GROUP}"="${BUDGET}" \
  +exp/seed="${SEED}" \
  hydra.run.dir="${PWD}/outputs/experiments/hydra/${RUN_ID}" \
  hydra.job.chdir=false
