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

## Bundle Analysis Artifacts

If you want one portable folder for later loss-landscape analysis, use:

- [tools/pinn/bundle_qbc_small_analysis.py](/Users/jonaswiendl/local/Repos/ml-surrogates-for-power-systems/tools/pinn/bundle_qbc_small_analysis.py)

Example:

```bash
python tools/pinn/bundle_qbc_small_analysis.py \
  --experiment-root outputs/pinn_hpc_experiments/<experiment_tag> \
  --dst results/pinn_analysis/<experiment_tag>
```

This copies, for each model:

- `dataset_pipeline/dataset_manifest.json`
- `dataset_pipeline/data/<MODEL>/dataset_v1/`
- `pinn_run/config.yaml`
- `pinn_run/metrics.csv`
- `pinn_run/checkpoints/`

If you already computed landscapes and want to preserve them too:

```bash
python tools/pinn/bundle_qbc_small_analysis.py \
  --experiment-root outputs/pinn_hpc_experiments/<experiment_tag> \
  --dst results/pinn_analysis/<experiment_tag> \
  --include-loss-landscape
```

## Run Loss Landscapes On HPC

For the real study, it is usually better to compute the landscapes on HPC too.

Files:

- Orchestrator: [tools/pinn/run_qbc_small_loss_landscape.py](/Users/jonaswiendl/local/Repos/ml-surrogates-for-power-systems/tools/pinn/run_qbc_small_loss_landscape.py)
- HPC job script: [tools/pinn/jobs/run_qbc_small_loss_landscape.lsf.sh](/Users/jonaswiendl/local/Repos/ml-surrogates-for-power-systems/tools/pinn/jobs/run_qbc_small_loss_landscape.lsf.sh)

The HPC script:

1. reads the finished PINN experiment manifest
2. finds the per-model PINN run directories
3. computes 1D and/or 2D loss landscapes from the selected checkpoint
4. exports only the compact analysis artifacts into a clean folder

### Basic submission

```bash
export EXPERIMENT_ROOT=/zhome/14/b/214266/Repos/ml-surrogates-for-power-systems/outputs/pinn_hpc_experiments/<experiment_tag>
bsub < tools/pinn/jobs/run_qbc_small_loss_landscape.lsf.sh
```

### Common overrides

```bash
EXPERIMENT_ROOT=/zhome/.../outputs/pinn_hpc_experiments/<experiment_tag> \
EXPORT_ROOT=/zhome/.../results/pinn_landscape/<experiment_tag> \
CHECKPOINT_TAG=best \
GRID_MODE=both \
RESOLUTION_1D=41 \
RESOLUTION_2D=21 \
ANALYSIS_SPLIT=train \
SUPERVISED_ROWS=1024 \
COLLOCATION_ROWS=1024 \
INIT_ROWS=128 \
LANDSCAPE_DEVICE=cuda \
bsub < tools/pinn/jobs/run_qbc_small_loss_landscape.lsf.sh
```

Useful variables:

- `EXPERIMENT_ROOT`: required, root of the finished PINN experiment
- `EXPORT_ROOT`: destination for the compact export bundle
- `CHECKPOINT_TAG`: `best`, `last`, `init`, or another saved checkpoint tag
- `GRID_MODE`: `1d`, `2d`, or `both`
- `RESOLUTION_1D`: number of points for 1D landscapes
- `RESOLUTION_2D`: number of points per axis for 2D landscapes
- `ANALYSIS_SPLIT`: usually `train`
- `SUPERVISED_ROWS`, `COLLOCATION_ROWS`, `INIT_ROWS`: fixed analysis subset sizes
- `MODELS`: optional comma-separated subset, for example `SM4,SM6`
- `SKIP_EXPORT=true`: compute landscapes only
- `EXPORT_METRICS=false`: do not copy `metrics.csv` into the export bundle
- `DRY_RUN=true`: print commands without running them

## What To Commit

For git, the best practice is to commit only the exported landscape bundle, not the full training outputs.

Recommended commit payload:

- `experiment_manifest.json`
- per-model `config.yaml`
- per-model `metrics.csv` if useful
- per-model `loss_landscape/`

Do not commit:

- raw stage-1 dataset outputs
- full preprocessed datasets
- checkpoints unless you explicitly want model recovery in git

This keeps the tracked artifacts compact while preserving the actual landscape arrays and metadata needed for downstream analysis.

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
