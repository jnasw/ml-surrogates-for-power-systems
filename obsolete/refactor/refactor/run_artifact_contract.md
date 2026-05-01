# Historical Refactor Note

This document records the run-artifact contract established during the refactor. Treat it as historical context. For current runnable experiment instructions, use `docs/setup/*.md`; for current Python entrypoints, use `src/experiments/pipeline/run_*.py`; for current HPC wrappers, use `hpc/`.

# Run Artifact Contract

## Purpose
This document defines the standard artifact structure for experiment runs during and after the refactor.

The goals are:
- make per-run outputs reproducible and machine-readable
- separate full run artifacts from summarized experiment results
- keep local files as the source of truth
- treat W&B as an optional reporting backend

---

## Artifact Policy

### outputs/
`outputs/` stores full per-run artifacts.

These are the source of truth for:
- reproducibility
- debugging
- resume/restart support
- later aggregation into results

### results/
`results/` stores aggregated and summarized outputs only.

These are intended for:
- comparison across runs
- plot-ready summaries
- benchmark tables
- thesis-facing exports

---

## Required Per-Run Artifacts in outputs/

Each normal run should write:

- `config.yaml`
  - fully resolved config used for the run

- `run_manifest.json`
  - run metadata, status, entrypoint, timestamps, dataset root, and artifact index

- `metrics.json`
  - canonical per-run metrics artifact

- `timings.json`
  - run walltimes and tracked sub-timings

Optional:
- `checkpoints/`
- `telemetry/`
- backend-specific metadata such as W&B local state

Plain-text logs may be stored under:
- `logs/`

---

## JSON vs CSV Policy

### outputs/
Use JSON for per-run artifacts:
- `run_manifest.json`
- `metrics.json`
- `timings.json`
- optional telemetry JSON files

### results/
Use CSV for aggregated or comparison-facing outputs:
- benchmark summaries
- cross-seed summaries
- comparison tables
- plot-ready exports

---

## Checkpoint Policy
- Checkpoints are opt-in.
- Runs should not save checkpoints unless explicitly configured.
- When enabled, checkpoints live under:
  - `checkpoints/`

---

## W&B Policy
- W&B is optional.
- W&B is a reporting backend, not the source of truth.
- All required run artifacts must exist locally even when W&B is enabled.

---

## Migration Note
During refactor, legacy artifacts such as per-run CSV metrics may temporarily coexist.
The target state is:
- JSON in `outputs/`
- CSV in `results/`

Current direct-run status:
- `10_run_baseline.py` writes canonical per-run JSON artifacts
- `20_run_pinn.py` writes canonical per-run JSON artifacts
- per-run PINN `metrics.csv` has been retired
- comparison-facing CSVs belong in `results/` or explicit summary artifacts, not as canonical per-run outputs
