#!/usr/bin/env bash
# Shared LSF defaults for repository HPC wrappers.
#
# Note: LSF reads #BSUB directives before this file is sourced. The wrappers keep
# static #BSUB defaults for reliable submission; these variables document the
# intended defaults and are echoed by each job for auditability.

QUEUE_GPU="${QUEUE_GPU:-gpua100}"
QUEUE_CPU="${QUEUE_CPU:-gpua100}"
DEFAULT_MEM_GB="${DEFAULT_MEM_GB:-8}"
DEFAULT_N_CORES="${DEFAULT_N_CORES:-4}"
DEFAULT_WALLTIME="${DEFAULT_WALLTIME:-04:00}"

activate_repo_venv() {
  local repo_root="$1"

  if [[ -d "${repo_root}/.venv" ]]; then
    # shellcheck disable=SC1091
    source "${repo_root}/.venv/bin/activate"
    echo "[hpc] activated_venv=${repo_root}/.venv"
  elif [[ -d "${repo_root}/venv" ]]; then
    # shellcheck disable=SC1091
    source "${repo_root}/venv/bin/activate"
    echo "[hpc] activated_venv=${repo_root}/venv"
  else
    echo "[hpc] activated_venv=<none found; using python3 from PATH>"
  fi
}
