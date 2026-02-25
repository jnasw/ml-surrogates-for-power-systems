#!/usr/bin/env bash
#SBATCH --job-name=sm4-qbc-sens
#SBATCH --output=outputs/slurm_logs/%x_%A_%a.out
#SBATCH --error=outputs/slurm_logs/%x_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --array=0-17%3

set -euo pipefail

# Run matrix (18 jobs):
# budget fixed: b1024
# qbc_M: 3,5,8
# qbc_P: 256,1024
# seeds: s01,s02,s03

M_VALUES=(3 5 8)
P_VALUES=(256 1024)
SEEDS=("s01" "s02" "s03")

N_M=${#M_VALUES[@]}
N_P=${#P_VALUES[@]}
N_SEEDS=${#SEEDS[@]}
TOTAL=$((N_M * N_P * N_SEEDS))

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

m_idx=$(( TASK_ID / (N_P * N_SEEDS) ))
rem=$(( TASK_ID % (N_P * N_SEEDS) ))
p_idx=$(( rem / N_SEEDS ))
seed_idx=$(( rem % N_SEEDS ))

M=${M_VALUES[$m_idx]}
P=${P_VALUES[$p_idx]}
SEED=${SEEDS[$seed_idx]}

# Keep final budget at 1024: n0 + T*K = 64 + 15*64
N0=64
K=64
T=15

RUN_ID="qbc_deep_ensemble_b1024_${SEED}_M${M}_P${P}"
echo "[RUN] ${RUN_ID} (task=${TASK_ID}/${TOTAL})"

mkdir -p outputs/slurm_logs

python run_experiment.py \
  --method qbc_deep_ensemble \
  --budget b1024 \
  --seed "${SEED}" \
  --phase qbc_sensitivity \
  --experiment-id thesis_sm4_v1 \
  --stage1-override "qbc_M=${M}" \
  --stage1-override "qbc_P=${P}" \
  --stage1-override "qbc_n0=${N0}" \
  --stage1-override "qbc_K=${K}" \
  --stage1-override "qbc_T=${T}"
