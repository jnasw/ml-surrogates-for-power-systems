# Configuration Layout

This directory contains the Hydra configuration files used by the main thesis
entrypoints.

## Entrypoint Configs

The top-level `setup_*.yaml` files are the primary configs selected directly by
Python entrypoints:

- `setup_dataset.yaml`: raw ODE dataset generation via `00_create_dataset.py`
- `setup_dataset_nn.yaml`: neural-network dataset preprocessing via `01_preprocess_dataset.py`
- `setup_pinn.yaml`: PINN training via `20_run_pinn.py`

`setup_baseline.yaml` is retained for historical baseline configuration
provenance. The final thesis workflows use `20_run_pinn.py` for downstream
PINN and data-only PINN training.

## Supporting Configs

- `method/`: Hydra-selectable dataset-generation strategies for
  `00_create_dataset.py`, selected with commands such as
  `+method=qbc_deep_ensemble`
- `registry/`: budget and seed labels loaded by experiment launchers
- `params/`: physical model parameters for the simulated power-system models
- `ic/`: initial-condition bounds and model output guide files
