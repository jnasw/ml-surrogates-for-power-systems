#!/usr/bin/env bash
set -euo pipefail

# LSF array index is 1-based; reuse Slurm runner by mapping to 0-based.
: "${LSB_JOBINDEX:?LSB_JOBINDEX is required}"
export SLURM_ARRAY_TASK_ID=$((LSB_JOBINDEX - 1))

exec bash scripts/hpc/slurm_array_qbc_phase2.sh
