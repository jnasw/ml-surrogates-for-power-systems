# ml-surrogates-for-power-systems

Master's thesis repository for surrogate modeling of power-system dynamics using data-driven and physics-informed neural networks.

The repository is organized around reproducible thesis experiments. Canonical Python workflows live under `src/experiments/`, optional LSF wrappers live under `hpc/`, and full run artifacts are written to `outputs/`.

## Repository Layout

- `src/config/`: Hydra configs, model parameters, IC bounds, and run registries.
- `src/data/`: dataset generation, preprocessing, active learning, and sampling utilities.
- `src/pinn/`: PINN data loading, losses, logging, checkpointing, collocation, and weighting logic.
- `src/training/`: shared training, metrics, and runtime helpers.
- `src/experiments/pipeline/`: runnable thesis experiment entrypoints.
- `src/sim/`: synchronous-machine simulator and ODE model definitions.
- `hpc/`: LSF wrappers for cluster execution.
- `hpo/`: HPO provenance, compact HPO result tables, and `hpo_analysis.ipynb`.
- `data/reference/`: persistent reusable training reference datasets.
- `data/evaluation/`: persistent reusable ID/OOD evaluation datasets.
- `outputs/`: generated run artifacts.
- `results/`: thesis-facing notebooks and curated CSV table exports.

## Environment

Use Python 3.11. For local development, install the lightweight requirements:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

On the A100 cluster, use `requirements-lock-a100.txt` instead. 

## Core Entry Points

Low-level data and training entrypoints:

```text
00_create_dataset.py
01_preprocess_dataset.py
20_run_pinn.py
```

Thesis experiment entrypoints:

```text
python3 -m src.experiments.pipeline.run_reference_datasets
python3 -m src.experiments.pipeline.run_evaluation_datasets
python3 -m src.experiments.pipeline.run_dataset_generation_comparison
python3 -m src.experiments.pipeline.run_optimizer_comparison
python3 -m src.experiments.pipeline.run_weighting_comparison
python3 -m src.experiments.pipeline.run_collocation_comparison
python3 -m src.experiments.pipeline.run_multistage_comparison
python3 -m src.experiments.pipeline.run_adaptive_augmentation
python3 -m src.experiments.pipeline.run_hpo_calibration
python3 -m src.experiments.pipeline.run_final_experiment
python3 -m src.experiments.pipeline.run_final_landscapes
```

`python3 -m src.experiments.pipeline.run_dataset_pipeline` is the shared dataset-generation pipeline used internally by reference, evaluation, and dataset-generation comparison launchers. It is not the main PINN comparison entrypoint.

`python3 -m src.experiments.pipeline.run_loss_landscape` and `python3 -m src.experiments.pipeline.run_posthoc_checkpoint_evaluation` are post-processing helpers.

Use Python entrypoints locally. Use the matching `hpc/*.lsf.sh` wrappers on the cluster.

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

PINN comparison launchers enable OOD evaluation by default using the model-aware OOD evaluation dataset. ID evaluation is opt-in unless a launcher says otherwise.

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
bsub -env "MODE=screening,REFERENCE_ID=smoke_SM4_lhs_b256_ds01,STRATEGIES=adam,SEED_LABELS=s01,DEVICE=cpu,NO_OOD_EVAL=true" \
  < hpc/run_optimizer_comparison.lsf.sh
```

HPC wrappers source shared defaults from `hpc/common/lsf_defaults.sh`, activate `.venv`, and write cluster logs under `hpc/logs/`. Use `bsub -env` for wrapper variables.

## Reproducing Thesis Results

The high-level reproduction path is:

1. Generate or verify reference datasets in `data/reference/`.
2. Generate or verify external ID/OOD evaluation datasets in `data/evaluation/`.
3. Run the component experiments listed in `src/experiments/README.md`.
4. Run the final experiment and, if needed, final loss-landscape post-processing.
5. Execute the notebooks in `results/` from the repository root.
6. Use the curated CSV exports under `results/tables/` for thesis tables.

The experiment catalogue in `src/experiments/README.md` maps each thesis study
to its Python launcher, HPC wrapper, and results notebook.

## Outputs And Results

`outputs/` is generated and contains full run artifacts for reproducibility and
debugging:

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

`results/` contains thesis-facing notebooks. `results/tables/` is the curated
CSV export area for final thesis table data.

## Documentation

- `src/experiments/README.md` lists the experiment pipelines.
- `results/README.md` documents the analysis notebooks.
- `results/tables/README.md` documents curated table exports.
- `hpc/README.md`, `hpo/README.md`, and `src/config/README.md` provide folder-specific context.
