#!/usr/bin/env bash
#BSUB -J vrba_family_benchmark_s02
#BSUB -q gpua100
#BSUB -n 1
#BSUB -W 24:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -oo outputs/lsf_logs/vrba_family_benchmark_s02.%J.out
#BSUB -eo outputs/lsf_logs/vrba_family_benchmark_s02.%J.err

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
  echo "[ERROR] submit with PYTHON_BIN=/path/to/venv/bin/python bsub < tools/pinn/jobs/run_vrba_family_benchmark_s02.lsf.sh"
  exit 1
fi

mkdir -p outputs/lsf_logs

STAMP="$(date +%Y%m%d_%H%M%S)"
PROFILE="vrba_family_benchmark"
DATASET_SEED="s02"
EXPERIMENT_TAG="vrba_family_benchmark_${DATASET_SEED}_${STAMP}"
OUTPUT_ROOT="${REPO_ROOT}/outputs/pinn_hpc_experiments/${EXPERIMENT_TAG}"

CMD=(
  "${PYTHON_BIN}"
  tools/pinn/run_collocation_comparison.py
  --profile "${PROFILE}"
  --experiment-tag "${EXPERIMENT_TAG}"
  --output-root "${OUTPUT_ROOT}"
  --model-flag "SM4"
  --dataset-seed "${DATASET_SEED}"
  --wandb-use
)

echo "[vrba-family-benchmark] repo_root=${REPO_ROOT}"
echo "[vrba-family-benchmark] output_root=${OUTPUT_ROOT}"
echo "[vrba-family-benchmark] profile=${PROFILE}"
echo "[vrba-family-benchmark] dataset_seed=${DATASET_SEED}"
echo "[vrba-family-benchmark] command=${CMD[*]}"

"${CMD[@]}"
