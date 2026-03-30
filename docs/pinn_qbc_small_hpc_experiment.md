# Small QBC-to-PINN HPC Experiment

This experiment runs the following pipeline for each model order:

1. Generate a fresh `qbc_deep_ensemble` dataset with budget `b1024`.
2. Preprocess the dataset into the PINN HDF5 layout.
3. Train the simple PINN with Adam-only for 300 epochs.
4. Log training metrics to Weights & Biases.

The default model set is:

- `SM4`
- `SM6`
- `SM_AVR_GOV`

## Files

- Orchestrator: [tools/pinn/run_qbc_small_experiment.py](/Users/jonaswiendl/local/Repos/ml-surrogates-for-power-systems/tools/pinn/run_qbc_small_experiment.py)
- HPC job script: [tools/pinn/jobs/run_qbc_small_experiment.lsf.sh](/Users/jonaswiendl/local/Repos/ml-surrogates-for-power-systems/tools/pinn/jobs/run_qbc_small_experiment.lsf.sh)

## Default setup

The LSF job script uses these defaults:

- dataset method: `qbc_deep_ensemble`
- budget: `b1024`
- dataset seed: `s01`
- PINN optimizer: Adam only
- epochs: `300`
- batch size: `1024`
- hidden dim: `64`
- hidden layers: `4`
- activation: `tanh`
- dtype: `float64`
- device: `cuda`
- loss weights:
  - `data=1.0`
  - `dt=1.0e-4`
  - `physics=1.0e-4`
  - `ic=1.0e-3`

## Basic submission

Make sure your environment can authenticate to W&B:

```bash
export WANDB_API_KEY=...
export WANDB_ENTITY=...
```

Then submit the job:

```bash
bsub < tools/pinn/jobs/run_qbc_small_experiment.lsf.sh
```

## Common overrides

You can override the main settings through environment variables at submit time.

Example:

```bash
EXPERIMENT_TAG=pinn_qbc_b1024_adam300_s01 \
OUTPUT_ROOT=/zhome/14/b/214266/Repos/ml-surrogates-for-power-systems/outputs/pinn_hpc_experiments/pinn_qbc_b1024_adam300_s01 \
MODELS=SM4,SM6,SM_AVR_GOV \
DATASET_SEED=s01 \
PINN_BUDGET=b1024 \
PINN_EPOCHS=300 \
PINN_DEVICE=cuda \
WANDB_PROJECT=sm-surrogates-pinn \
WANDB_ENTITY=your_entity \
bsub < tools/pinn/jobs/run_qbc_small_experiment.lsf.sh
```

Useful variables:

- `EXPERIMENT_TAG`: run suffix used in output paths and W&B group naming
- `OUTPUT_ROOT`: explicit output directory for the experiment
- `MODELS`: comma-separated model list, for example `SM4,SM6`
- `DATASET_SEED`: dataset seed label from [src/config/registry/seeds.yaml](/Users/jonaswiendl/local/Repos/ml-surrogates-for-power-systems/src/config/registry/seeds.yaml)
- `PINN_BUDGET`: budget label from [src/config/registry/budgets.yaml](/Users/jonaswiendl/local/Repos/ml-surrogates-for-power-systems/src/config/registry/budgets.yaml)
- `PINN_EPOCHS`: Adam epochs
- `PINN_BATCH_SIZE`: PINN batch size
- `PINN_DEVICE`: `cuda`, `cpu`, `mps`, or `auto`
- `PINN_HIDDEN_DIM`: network width
- `PINN_HIDDEN_LAYERS`: network depth
- `PINN_ACTIVATION`: activation name
- `PINN_DTYPE`: PINN dtype
- `WANDB_PROJECT`: W&B project name
- `WANDB_ENTITY`: optional W&B entity
- `DRY_RUN=true`: print commands without running them

## Outputs

By default, outputs are written under:

```text
outputs/pinn_hpc_experiments/<experiment_tag>/
```

For each model, the layout is:

```text
<output_root>/<model_lower>/
  dataset_pipeline/
    dataset_manifest.json
    data/<MODEL>/dataset_v1/
    logs/
  pinn_adam300/
    config.yaml
    metrics.csv
    checkpoints/
```

The orchestrator also writes:

```text
<output_root>/experiment_manifest.json
```

This top-level manifest records the dataset pipeline root, final dataset root, and PINN run directory for each model.

## W&B naming

The job creates one W&B group for the whole experiment:

```text
pinn_qbc_<budget>_adam<epochs>_<experiment_tag>
```

Default run names are:

- `pinn_sm4_qbc_b1024_adam300`
- `pinn_sm6_qbc_b1024_adam300`
- `pinn_sm_avr_gov_qbc_b1024_adam300`

## Dry-run check

To inspect the exact commands before submitting:

```bash
DRY_RUN=true bsub < tools/pinn/jobs/run_qbc_small_experiment.lsf.sh
```

Or locally:

```bash
python tools/pinn/run_qbc_small_experiment.py \
  --experiment-tag smoke_qbc_small \
  --output-root /tmp/qbc_small_pinn_smoke \
  --models SM4 \
  --budget b1024 \
  --dataset-seed s01 \
  --epochs 300 \
  --device cpu \
  --dry-run
```
