# Experiment Pipelines

This folder contains the Python launchers used for the thesis experiment
campaign. The launchers are the canonical local entrypoints; `hpc/*.lsf.sh`
contains optional LSF wrappers for cluster submission.

Run commands from the repository root.

## Thesis Campaign

| Study | Pipeline module | Purpose | HPC wrapper | Results notebook |
|---|---|---|---|---|
| Exp. 1: dataset generation | `src.experiments.pipeline.run_dataset_generation_comparison` | Compare LHS, QBC, and marker-based dataset generation under fixed trajectory budgets. | `hpc/run_dataset_generation_comparison.lsf.sh` | `results/01_dataset_generation.ipynb` |
| Exp. 2: optimizer comparison | `src.experiments.pipeline.run_optimizer_comparison` | Compare Adam, quasi-Newton, and Adam-to-quasi-Newton schedules. | `hpc/run_optimizer_comparison.lsf.sh` | `results/02_optimizer_comparison.ipynb` |
| Exp. 3: loss balancing | `src.experiments.pipeline.run_weighting_comparison` | Compare data-only, static, warmup, and adaptive PINN loss weighting. | `hpc/run_weighting_comparison.lsf.sh` | `results/03_loss_balancing.ipynb` |
| Exp. 4: collocation sampling | `src.experiments.pipeline.run_collocation_comparison` | Compare static LHS, random refresh, and residual-adaptive collocation sampling. | `hpc/run_collocation_comparison.lsf.sh` | `results/04_collocation_comparison.ipynb` |
| Exp. 5: multistage training | `src.experiments.pipeline.run_multistage_comparison` | Compare residual-correction stages and correction-stage optimizers. | `hpc/run_multistage_comparison.lsf.sh` | `results/05_multistage.ipynb` |
| Exp. 6: adaptive augmentation | `src.experiments.pipeline.run_adaptive_augmentation` | Compare supervised-data growth and collocation-growth strategies. | `hpc/run_adaptive_augmentation.lsf.sh` | `results/06_data_augmentation.ipynb` |
| Exp. 7: final experiment | `src.experiments.pipeline.run_final_experiment` | Evaluate the selected combined strategy against data-only and static PINN references on SM4, SM6, and SM_AVR_GOV. | `hpc/run_final_experiment.lsf.sh` | `results/07_final_experiment.ipynb` |

The common protocol uses fixed reference and evaluation datasets, repeated
training seeds, and external ID/OOD evaluation. The final experiment combines
the strongest components identified in the earlier studies.

## Protocol Summary

Unless a study states otherwise, the controlled experiments use the SM4 model,
the time-conditioned surrogate formulation `(t, x0, u) -> x(t)`, fixed
trajectory-wise data splits, fixed external ID/OOD evaluation datasets, and
repeated training seeds. The baseline PINN architecture is three hidden layers
with 64 neurons and `tanh` activation. Preliminary calibration runs fixed the
architecture, Adam learning rate, and acquisition-policy settings before the
main experiment campaign.

## Runbook Summary

These commands are intentionally minimal, as full runs should be submitted to the cluster as described below. Use `--help` on any launcher for the
full option list.

### Reference Datasets

Purpose: generate persistent training datasets reused across PINN experiments.

```bash
python3 -m src.experiments.pipeline.run_reference_datasets \
  --reference-id main_SM4_qbc_b512_ds01 \
  --dry-run
```

HPC wrapper: `hpc/generate_reference_datasets.lsf.sh`

Expected outputs: `data/reference/index.json` and generated datasets under
`data/reference/`.

### Evaluation Datasets

Purpose: generate fixed external ID/OOD datasets used for evaluation.

```bash
python3 -m src.experiments.pipeline.run_evaluation_datasets \
  --evaluation-id id_SM4_lhs_b512_ds01 \
  --dry-run
```

HPC wrapper: `hpc/generate_evaluation_datasets.lsf.sh`

Expected outputs: `data/evaluation/index.json` and generated datasets under
`data/evaluation/`.

### Dataset Generation Comparison

Purpose: compare dataset-generation methods with a fixed downstream data-only
PINN evaluator.

```bash
python3 -m src.experiments.pipeline.run_dataset_generation_comparison \
  --mode smoke \
  --model-flag SM4 \
  --dry-run
```

HPC wrapper: `hpc/run_dataset_generation_comparison.lsf.sh`

Expected outputs: `outputs/experiments/dataset_generation_comparison/<tag>/`.

### Optimizer Comparison

Purpose: compare Adam, quasi-Newton, and hybrid optimizer schedules.

```bash
python3 -m src.experiments.pipeline.run_optimizer_comparison \
  --mode screening \
  --reference-id smoke_SM4_lhs_b256_ds01 \
  --strategies adam \
  --seed-labels s01 \
  --device cpu \
  --dry-run
```

HPC wrapper: `hpc/run_optimizer_comparison.lsf.sh`

Expected outputs: `outputs/pinn/optimizer_comparison/<timestamp>/`.

### Loss Balancing

Purpose: compare static, warmup, data-only, and adaptive PINN loss weighting.

```bash
python3 -m src.experiments.pipeline.run_weighting_comparison \
  --mode screening \
  --reference-id smoke_SM4_lhs_b256_ds01 \
  --strategies data_only,static_tuned \
  --seed-labels s01 \
  --device cpu \
  --dry-run
```

HPC wrapper: `hpc/run_weighting_comparison.lsf.sh`

Expected outputs: `outputs/pinn/weighting_comparison/<timestamp>/`.

### Collocation Sampling

Purpose: compare fixed, random-refresh, and residual-adaptive collocation
strategies.

```bash
python3 -m src.experiments.pipeline.run_collocation_comparison \
  --mode cadence \
  --reference-id smoke_SM4_lhs_b256_ds01 \
  --device cpu \
  --dry-run
```

HPC wrapper: `hpc/run_collocation_comparison.lsf.sh`

Expected outputs: `outputs/pinn/collocation_comparison/<timestamp>/`.

### Multistage Training

Purpose: compare residual-correction stages and correction-stage optimizers.

```bash
python3 -m src.experiments.pipeline.run_multistage_comparison \
  --mode screening \
  --reference-id smoke_SM4_lhs_b256_ds01 \
  --device cpu \
  --dry-run
```

HPC wrapper: `hpc/run_multistage_comparison.lsf.sh`

Expected outputs: `outputs/pinn/multistage_comparison/<timestamp>/`.

### Adaptive Augmentation

Purpose: compare supervised-data growth and collocation-growth strategies.

```bash
python3 -m src.experiments.pipeline.run_adaptive_augmentation \
  --mode screening \
  --reference-id smoke_SM4_lhs_b256_ds01 \
  --device cpu \
  --dry-run
```

HPC wrapper: `hpc/run_adaptive_augmentation.lsf.sh`

Expected outputs: `outputs/pinn/adaptive_augmentation/<timestamp>/`.

### HPO Calibration

Purpose: record preliminary calibration runs for architecture, optimizer, and
acquisition-policy choices.

```bash
python3 -m src.experiments.pipeline.run_hpo_calibration \
  --study pinn_architecture \
  --dry-run
```

HPC wrapper: `hpc/run_hpo_calibration.lsf.sh`

Expected outputs: `outputs/hpo/<study>/<timestamp>/`.

### Final Experiment

Purpose: run the selected final comparison across SM4, SM6, and SM_AVR_GOV.

```bash
python3 -m src.experiments.pipeline.run_final_experiment \
  --models SM4 \
  --seed-labels s01 \
  --dry-run
```

HPC wrapper: `hpc/run_final_experiment.lsf.sh`

Expected outputs: `outputs/pinn/final_experiment/<timestamp>/`.

### Final Loss Landscapes

Purpose: post-process final experiment checkpoints with loss-landscape
evaluations.

```bash
python3 -m src.experiments.pipeline.run_final_landscapes \
  --run-manifest outputs/pinn/final_experiment/<tag>/run_manifest.json \
  --models SM4 \
  --seed-labels s01 \
  --dry-run
```

HPC wrapper: `hpc/run_final_landscapes.lsf.sh`

Expected outputs: `outputs/pinn/final_experiment/<tag>/loss_landscapes/<timestamp>/`.

## Supporting Pipelines

| Pipeline module | Purpose | HPC wrapper | Results notebook | Status |
|---|---|---|---|---|
| `src.experiments.pipeline.run_reference_datasets` | Generate persistent training reference datasets used by comparison experiments. | `hpc/generate_reference_datasets.lsf.sh` | none | supporting |
| `src.experiments.pipeline.run_evaluation_datasets` | Generate persistent external ID/OOD evaluation datasets. | `hpc/generate_evaluation_datasets.lsf.sh` | none | supporting |
| `src.experiments.pipeline.run_hpo_calibration` | Run preliminary calibration studies for architecture, optimizer, and acquisition-policy choices. | `hpc/run_hpo_calibration.lsf.sh` | `hpo/hpo_analysis.ipynb` | supporting |
| `src.experiments.pipeline.run_final_landscapes` | Launch loss-landscape post-processing for final experiment checkpoints. | `hpc/run_final_landscapes.lsf.sh` | `results/07_final_experiment.ipynb` | optional |
| `src.experiments.pipeline.run_loss_landscape` | Compute one loss-landscape job. Called by `run_final_landscapes`. | none | `results/07_final_experiment.ipynb` | optional |
| `src.experiments.pipeline.run_posthoc_checkpoint_evaluation` | Re-evaluate checkpoints after training, mainly for consistency checks. | none | none | optional |
| `src.experiments.pipeline.run_dataset_pipeline` | Shared dataset-generation/preprocessing pipeline used internally by reference, evaluation, and dataset-generation launchers. | none | none | supporting |

## Cluster Submission Prompts

Submit wrappers from the repository root. Use `bsub -env` for simple values.
For comma-separated values, export variables first and submit with
`bsub -env "all"`.

### Reference Datasets

Generates all registered reference datasets: smoke, dev, and main suites for
all supported model flags.

```bash
bsub -env "MODEL_FLAG=all,SUITE=all" \
  < hpc/generate_reference_datasets.lsf.sh
```

This currently covers SM4, SM6, and SM_AVR_GOV reference datasets, including
the thesis-default QBC references used by downstream training experiments.

### Evaluation Datasets

Generates all registered ID and OOD evaluation datasets for all supported model
flags.

```bash
bsub -env "MODEL_FLAG=all" \
  < hpc/generate_evaluation_datasets.lsf.sh
```

This currently covers SM4, SM6, and SM_AVR_GOV with both regular evaluation
sets and the larger evaluation sets used by the dataset-generation comparison.

### Exp. 1: Dataset Generation

```bash
bsub -env "MODE=smoke,MODEL_FLAG=SM4" \
  < hpc/run_dataset_generation_comparison.lsf.sh
```

```bash
(export CAMPAIGN_TAG=dataset_final_sm4 MODE=final MODEL_FLAG=SM4 \
  METHODS=lhs_static BUDGETS=b256,b512 DATASET_SEEDS=ds01 \
  BASELINE_SEEDS=bs01,bs02 && \
  bsub -env "all" < hpc/run_dataset_generation_comparison.lsf.sh)
```

```bash
bsub -env "CAMPAIGN_TAG=dataset_pinn_sm4,MODE=final,MODEL_FLAG=SM4,METHODS=lhs_static" \
  -J data_lhs_pinn < hpc/run_dataset_generation_comparison.lsf.sh
```

### Exp. 2: Optimizer Comparison

```bash
(export MODE=screening STRATEGIES=adam,lbfgs,adam_lbfgs SEED_LABELS=s01 && \
  bsub -env "all" < hpc/run_optimizer_comparison.lsf.sh)
```

```bash
bsub -env "MODE=final,REFERENCE_ID=main_SM4_qbc_b512_ds01" \
  < hpc/run_optimizer_comparison.lsf.sh
```

### Exp. 3: Loss Balancing

```bash
(export MODE=screening STRATEGIES=static_tuned,ma,id SEED_LABELS=s01 && \
  bsub -env "all" < hpc/run_weighting_comparison.lsf.sh)
```

```bash
bsub -env "MODE=final,REFERENCE_ID=main_SM4_qbc_b512_ds01" \
  < hpc/run_weighting_comparison.lsf.sh
```

### Exp. 4: Collocation Sampling

```bash
(export MODE=screening REFERENCE_ID=smoke_SM4_lhs_b256_ds01 \
  STRATEGIES=uniform_lhs,rad DENSITIES=p4k,p32k DEVICE=cpu && \
  bsub -env "all" < hpc/run_collocation_comparison.lsf.sh)
```

```bash
(export MODE=final REFERENCE_ID=main_SM4_qbc_b512_ds01 \
  SEED_LABELS=s01,s02,s03,s04,s05 && \
  bsub -env "all" < hpc/run_collocation_comparison.lsf.sh)
```

### Exp. 5: Multistage Training

```bash
(export MODE=screening REFERENCE_ID=smoke_SM4_lhs_b256_ds01 \
  STRATEGIES=adam_30000,adam_ssbroyden_2stage SEED_LABELS=s01 DEVICE=cpu && \
  bsub -env "all" < hpc/run_multistage_comparison.lsf.sh)
```

```bash
(export MODE=final REFERENCE_ID=main_SM4_qbc_b512_ds01 \
  STRATEGIES=adam_30000,adam_ssbroyden_2stage,adam_ssbroyden_3stage,adam_ssbroyden_4stage,adam_ssbroyden_5stage \
  SEED_LABELS=s01,s02,s03,s04,s05 && \
  bsub -env "all" < hpc/run_multistage_comparison.lsf.sh)
```

### Exp. 6: Adaptive Augmentation

```bash
(export MODE=screening REFERENCE_ID=main_SM4_lhs_b4096_ds01 \
  SUPERVISED_STRATEGIES=fixed_low,mae_nearest_growth \
  COLLOCATION_STRATEGIES=static_low,rar_d_growth \
  SEED_LABELS=s01 && \
  bsub -env "all" < hpc/run_adaptive_augmentation.lsf.sh)
```

```bash
(export MODE=final REFERENCE_ID=main_SM4_lhs_b4096_ds01 \
  SUPERVISED_STRATEGIES=fixed_low,random_growth,mae_nearest_growth,fixed_full \
  COLLOCATION_STRATEGIES=static_low,rar_d_growth \
  SEED_LABELS=s01,s02,s03,s04,s05 REFRESH_PERIOD_EPOCHS=500 && \
  bsub -env "all" < hpc/run_adaptive_augmentation.lsf.sh)
```

### HPO Calibration

```bash
bsub -env "STUDY=adam_lr,PINN_HIDDEN_DIM=64,PINN_HIDDEN_LAYERS=3" \
  < hpc/run_hpo_calibration.lsf.sh
```

```bash
(export STUDY=second_order_lr OPTIMIZERS=LBFGS,SSBFGS LRS=0.1,0.3,1.0 && \
  bsub -env "all" < hpc/run_hpo_calibration.lsf.sh)
```

### Exp. 7: Final Experiment

```bash
bsub -env "SEED_LABELS=s01" \
  < hpc/run_final_experiment.lsf.sh
```

```bash
(export MODELS=SM4,SM6 SEED_LABELS=s01 && \
  bsub -env "all" < hpc/run_final_experiment.lsf.sh)
```

### Final Loss Landscapes

```bash
bsub -env "RUN_MANIFEST=outputs/pinn/final_experiment/<tag>/run_manifest.json,MODELS=SM4,SEED_LABELS=s01" \
  < hpc/run_final_landscapes.lsf.sh
```

Wrapper details, environment-variable conventions, and log locations are
documented in `hpc/README.md`.

## Outputs

Full run artifacts are written under `outputs/`. Thesis-facing notebooks and
curated CSV tables live under `results/`.
