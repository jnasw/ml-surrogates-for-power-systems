#!/usr/bin/env bash
#SBATCH --job-name=sm4-main-qbc
#SBATCH --output=outputs/slurm_logs/%x_%A_%a.out
#SBATCH --error=outputs/slurm_logs/%x_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --array=0-14%3

set -euo pipefail

<<<<<<< HEAD
# Run matrix (15 jobs):
# method: qbc
=======
# Run matrix:
# methods: lhs_static, qbc_deep_ensemble
>>>>>>> c209f76588b46aa058418ada6d48a49bf5c00f6d
# budgets: b256, b1024, b4096
# seeds: s01..s05

<<<<<<< HEAD
=======
METHODS=("lhs_static" "qbc_deep_ensemble")
>>>>>>> c209f76588b46aa058418ada6d48a49bf5c00f6d
BUDGETS=("b256" "b1024" "b4096")
SEEDS=("s01" "s02" "s03" "s04" "s05")

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

<<<<<<< HEAD
RUN_ID="qbc_deep_ensemble_${BUDGET}_${SEED}"
=======
RUN_ID="${METHOD}_${BUDGET}_${SEED}"
>>>>>>> c209f76588b46aa058418ada6d48a49bf5c00f6d
echo "[RUN] ${RUN_ID} (task=${TASK_ID}/${TOTAL})"

mkdir -p outputs/slurm_logs

# Optional: activate your env here
# source ~/.bashrc
# conda activate <env-name>

<<<<<<< HEAD
python create_dataset.py \
  +exp=base \
  +exp/phase=main \
  +exp/method=qbc \
  +exp/budget/qbc="${BUDGET}" \
  +exp/seed="${SEED}" \
  hydra.run.dir="${PWD}/outputs/experiments/hydra/${RUN_ID}" \
  hydra.job.chdir=false
=======
python run_experiment.py \
  --method "${METHOD}" \
  --budget "${BUDGET}" \
  --seed "${SEED}" \
  --phase main \
  --experiment-id thesis_sm4_v1
>>>>>>> c209f76588b46aa058418ada6d48a49bf5c00f6d
