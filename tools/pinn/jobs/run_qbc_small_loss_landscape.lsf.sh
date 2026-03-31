#!/usr/bin/env bash
#BSUB -J pinn_qbc_landscape
#BSUB -q gpua100
#BSUB -n 1
#BSUB -W 24:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -oo outputs/lsf_logs/pinn_qbc_landscape.%J.out
#BSUB -eo outputs/lsf_logs/pinn_qbc_landscape.%J.err

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
  echo "[ERROR] submit with PYTHON_BIN=/path/to/venv/bin/python bsub < tools/pinn/jobs/run_qbc_small_loss_landscape.lsf.sh"
  exit 1
fi

if [[ -z "${EXPERIMENT_ROOT:-}" ]]; then
  echo "[ERROR] EXPERIMENT_ROOT is required."
  echo "[ERROR] Example:"
  echo "  export EXPERIMENT_ROOT=/zhome/.../outputs/pinn_hpc_experiments/<experiment_tag>"
  echo "  bsub < tools/pinn/jobs/run_qbc_small_loss_landscape.lsf.sh"
  exit 1
fi

mkdir -p outputs/lsf_logs

CHECKPOINT_TAG="${CHECKPOINT_TAG:-best}"
GRID_MODE="${GRID_MODE:-both}"
RESOLUTION_1D="${RESOLUTION_1D:-41}"
RESOLUTION_2D="${RESOLUTION_2D:-21}"
ANALYSIS_SPLIT="${ANALYSIS_SPLIT:-train}"
SUPERVISED_ROWS="${SUPERVISED_ROWS:-1024}"
COLLOCATION_ROWS="${COLLOCATION_ROWS:-1024}"
INIT_ROWS="${INIT_ROWS:-128}"
ANALYSIS_SEED="${ANALYSIS_SEED:-0}"
DIRECTION_SEED="${DIRECTION_SEED:-0}"
LANDSCAPE_DEVICE="${LANDSCAPE_DEVICE:-cuda}"
NORMALIZATION="${NORMALIZATION:-filter}"
MODELS="${MODELS:-}"
EXPORT_ROOT="${EXPORT_ROOT:-${EXPERIMENT_ROOT%/}_landscape_export}"
SKIP_EXPORT="${SKIP_EXPORT:-false}"
EXPORT_METRICS="${EXPORT_METRICS:-true}"
DRY_RUN="${DRY_RUN:-false}"

CMD=(
  "${PYTHON_BIN}"
  tools/pinn/run_qbc_small_loss_landscape.py
  --experiment-root "${EXPERIMENT_ROOT}"
  --checkpoint-tag "${CHECKPOINT_TAG}"
  --grid "${GRID_MODE}"
  --resolution-1d "${RESOLUTION_1D}"
  --resolution-2d "${RESOLUTION_2D}"
  --split "${ANALYSIS_SPLIT}"
  --supervised-rows "${SUPERVISED_ROWS}"
  --collocation-rows "${COLLOCATION_ROWS}"
  --init-rows "${INIT_ROWS}"
  --analysis-seed "${ANALYSIS_SEED}"
  --direction-seed "${DIRECTION_SEED}"
  --normalization "${NORMALIZATION}"
  --device "${LANDSCAPE_DEVICE}"
  --export-root "${EXPORT_ROOT}"
)

if [[ -n "${MODELS}" ]]; then
  CMD+=(--models "${MODELS}")
fi
if [[ "${SKIP_EXPORT}" == "true" ]]; then
  CMD+=(--skip-export)
fi
if [[ "${EXPORT_METRICS}" != "true" ]]; then
  CMD+=(--no-export-metrics)
fi
if [[ "${DRY_RUN}" == "true" ]]; then
  CMD+=(--dry-run)
fi

echo "[pinn-qbc-landscape] experiment_root=${EXPERIMENT_ROOT}"
echo "[pinn-qbc-landscape] export_root=${EXPORT_ROOT}"
echo "[pinn-qbc-landscape] command=${CMD[*]}"

"${CMD[@]}"
