# HPO Provenance

This folder preserves the HPO material used during thesis method selection.

The files here are historical/provenance artifacts, not active Hydra configuration
groups for `00_create_dataset.py`.

Current runnable HPO calibration lives in:

- `src/experiments/pipeline/run_hpo_calibration.py`
- `hpc/run_hpo_calibration.lsf.sh`
- `docs/setup/hpo_calibration.md`

Historical workflow material:

- `workflow_configs/` contains the workflow YAMLs used for dataset-generation
  method HPO.
- `results/` contains compact summary and winner tables from HPO.
- `hpo_analysis.ipynb` contains the analysis notebook used to inspect the HPO
  outcomes.

Keeping these files at the repository top level preserves thesis provenance while
keeping `src/config/` limited to currently active Hydra configs.
