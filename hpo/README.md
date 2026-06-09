# HPO Provenance

This folder preserves the HPO material used during thesis method selection.

The files here are historical/provenance artifacts, not active Hydra configuration
groups for `00_create_dataset.py`.

Current runnable HPO calibration lives in:

- `src/experiments/pipeline/run_hpo_calibration.py`
- `hpc/hpo/run_hpo_calibration.lsf.sh`
- `docs/setup/hpo_calibration.md`

Historical workflow material:

- `workflow_configs/` contains the workflow YAMLs used for dataset-generation
  method HPO.
- `workflow_design.md` documents the older multi-stage workflow design.
- `results/` contains compact summary and winner tables from HPO.
- `notebooks/` contains HPO analysis notebooks.

Keeping these files at the repository top level preserves thesis provenance while
keeping `src/config/` limited to currently active Hydra configs.
