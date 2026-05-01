# ml-surrogates-for-power-systems

Master's thesis repository for surrogate modeling of power-system dynamics using data-driven and physics-informed neural networks.

The repository is organized around reproducible experiment execution rather than a general-purpose framework. Canonical workflows live under `src/experiments/`, cluster wrappers live under `hpc/`, and full run artifacts are written to `outputs/`.

## Canonical Structure

- `src/config/` contains Hydra configuration, registries, model setup, budgets, seeds, and IC bounds.
- `src/data/` contains dataset loading, preprocessing helpers, active-learning and sampling utilities.
- `src/training/` contains supervised baseline and shared training components.
- `src/pinn/` contains PINN runtime data loading, losses, logging, checkpointing, collocation, and weighting logic.
- `src/experiments/` contains canonical Python experiment workflows.
- `src/experiments/pipeline/run_*.py` are the runnable experiment entrypoints.
- `src/experiments/pipeline/helpers/` contains shared pipeline helpers for references, evaluation sets, seeds, summaries, manifests, W&B metadata, and launch utilities.
- `src/sim/` contains simulator/model code.
- `hpc/` contains canonical LSF job wrappers.
- `tools/` is legacy/ad hoc/analysis utility space, not the main workflow surface.
- `obsolete/` contains archived legacy wrappers.
- `data/reference/` stores persistent reusable training reference datasets.
- `data/evaluation/` stores persistent reusable ID/OOD evaluation datasets.

## Core Entry Points

The low-level data and training entrypoints are:

```text
00_create_dataset.py
01_preprocess_dataset.py
10_run_baseline.py
20_run_pinn.py
```

The thesis experiment entrypoints are:

```text
python3 -m src.experiments.pipeline.run_reference_datasets
python3 -m src.experiments.pipeline.run_evaluation_datasets
python3 -m src.experiments.pipeline.run_optimizer_comparison
python3 -m src.experiments.pipeline.run_weighting_comparison
python3 -m src.experiments.pipeline.run_collocation_comparison
python3 -m src.experiments.pipeline.run_multistage_comparison
python3 -m src.experiments.pipeline.run_dataset_generation_comparison
python3 -m src.experiments.pipeline.run_hpo_calibration
```

Use the Python entrypoints locally. Use the matching `hpc/**/*.lsf.sh` wrappers on the cluster.

## Reference And Evaluation Datasets

Persistent training datasets are generated once and reused by PINN comparison experiments:

```bash
python3 -m src.experiments.pipeline.run_reference_datasets \
  --reference-id main_SM4_qbc_b512_ds01
```

External evaluation datasets are generated independently of training datasets:

```bash
python3 -m src.experiments.pipeline.run_evaluation_datasets \
  --evaluation-id id_SM4_lhs_b512_ds01

python3 -m src.experiments.pipeline.run_evaluation_datasets \
  --evaluation-id ood_SM4_wide_ic_b512_ds01
```

Indexes:

```text
data/reference/index.json
data/evaluation/index.json
```

PINN comparison launchers enable OOD evaluation by default using the model-aware OOD evaluation dataset. ID evaluation is opt-in.

## Running Experiments

Example local dry-run:

```bash
python3 -m src.experiments.pipeline.run_optimizer_comparison \
  --mode screening \
  --reference-id smoke_SM4_lhs_b256_ds01 \
  --strategies adam \
  --seed-labels s01 \
  --device cpu \
  --no-ood-eval \
  --dry-run
```

Example HPC submission:

```bash
MODE=screening REFERENCE_ID=smoke_SM4_lhs_b256_ds01 \
STRATEGIES=adam SEED_LABELS=s01 DEVICE=cpu NO_OOD_EVAL=true \
bsub < hpc/optimizer_comparison/run_optimizer_comparison.lsf.sh
```

HPC wrappers:

- source shared defaults from `hpc/common/lsf_defaults.sh`
- activate `.venv` or `venv` if present
- write cluster logs under `hpc/logs/<workflow>/`
- use inline environment variables in examples
- do not rely on `bsub -env`

The static `#BSUB` directives in each wrapper are the actual submitted resources on clusters that do not expand shell variables in directives.

## Outputs And Results

`outputs/` contains full run artifacts for reproducibility and debugging:

```text
run_manifest.json
summary.csv
summary.json
failures.json
logs/
runs/<run_name>/metrics.json
runs/<run_name>/timings.json
runs/<run_name>/epoch_metrics.csv
runs/<run_name>/checkpoints/best.pt
```

`results/` is reserved for lightweight, thesis-ready exports and plot-ready summaries. Do not treat `results/` as the default dump location for normal run artifacts.

## Documentation

- `docs/setup/` contains current runbooks for thesis experiments.
- `docs/experiments/` contains higher-level experiment design notes.
- `docs/refactor/` contains refactor and architecture notes, including historical context.

When behavior changes, update the relevant `docs/setup/*.md` file first.
