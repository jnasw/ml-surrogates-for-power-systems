# Final Thesis Experiment

## Objective

Run the final controlled thesis matrix selected from the preceding dataset, loss-balancing, optimizer, and multistage experiments.

The final comparison asks whether the selected physics-informed training recipe improves over a supervised data-only baseline, and whether it is competitive with direct SSBroyden optimization.

## Design

Fixed variables:
- Dataset method: QBC deep ensemble
- Dataset budget: `b1024`
- Dataset seed: `ds01`
- Model orders: `SM4`, `SM6`, `SM_AVR_GOV`
- Training seeds: `s01`, `s02`, `s03`
- Runtime dtype: `float64`
- Architecture defaults: hidden width `64`, hidden layers `4`, activation `tanh`
- Evaluation: model-specific ID and OOD evaluation datasets

Strategies:

| Strategy | Loss / weighting | Optimizer schedule | Purpose |
|---|---|---|---|
| `data_only_adam_5000` | data only | Adam 5000 | supervised reference baseline |
| `dn_adam3000_ssbroyden2000` | tuned base weights + DN | Adam 3000 -> SSBroyden 2000 | selected adaptive-weighting plus refinement recipe |
| `ssbroyden_5000` | tuned static PINN weights | SSBroyden 5000 | direct second-order PINN reference |

Default matrix:

```text
3 strategies x 3 seeds x 3 model orders = 27 runs
```

## Reference Datasets

The final launcher expects these reference IDs:

```text
main_SM4_qbc_b1024_ds01
main_SM6_qbc_b1024_ds01
main_SM_AVR_GOV_qbc_b1024_ds01
```

Dry-run generation:

```bash
python3 -m src.experiments.pipeline.run_reference_datasets \
  --reference-id main_SM4_qbc_b1024_ds01 \
  --reference-id main_SM6_qbc_b1024_ds01 \
  --reference-id main_SM_AVR_GOV_qbc_b1024_ds01 \
  --dry-run
```

Generate missing references:

```bash
python3 -m src.experiments.pipeline.run_reference_datasets \
  --reference-id main_SM4_qbc_b1024_ds01 \
  --reference-id main_SM6_qbc_b1024_ds01 \
  --reference-id main_SM_AVR_GOV_qbc_b1024_ds01
```

On the cluster:

```bash
(export REFERENCE_IDS=main_SM4_qbc_b1024_ds01,main_SM6_qbc_b1024_ds01,main_SM_AVR_GOV_qbc_b1024_ds01 \
  DRY_RUN=true && \
  bsub -env "all" < hpc/reference_datasets/generate_reference_datasets.lsf.sh)
```

Then submit with `DRY_RUN=false`.

## Evaluation Datasets

The final launcher uses:

```text
id_<MODEL>_lhs_b512_ds01
ood_<MODEL>_wide_ic_b512_ds01
```

Generate them if needed:

```bash
python3 -m src.experiments.pipeline.run_evaluation_datasets \
  --evaluation-id id_SM4_lhs_b512_ds01 \
  --evaluation-id ood_SM4_wide_ic_b512_ds01 \
  --evaluation-id id_SM6_lhs_b512_ds01 \
  --evaluation-id ood_SM6_wide_ic_b512_ds01 \
  --evaluation-id id_SM_AVR_GOV_lhs_b512_ds01 \
  --evaluation-id ood_SM_AVR_GOV_wide_ic_b512_ds01
```

## Local Dry Run

```bash
python3 -m src.experiments.pipeline.run_final_experiment \
  --models SM4 \
  --seed-labels s01 \
  --dry-run
```

This prints the three SM4 commands and writes a dry-run manifest under:

```text
outputs/pinn/final_experiment/<timestamp>/
```

## HPC Submission

The intended final submission is one job per training seed:

```bash
bsub -env "SEED_LABELS=s01" < hpc/final_experiment/run_final_experiment.lsf.sh
bsub -env "SEED_LABELS=s02" < hpc/final_experiment/run_final_experiment.lsf.sh
bsub -env "SEED_LABELS=s03" < hpc/final_experiment/run_final_experiment.lsf.sh
```

Let each seed job use its own timestamped output root, or set a distinct `OUTPUT_ROOT` / `EXPERIMENT_TAG` per seed. Do not point concurrent seed jobs at the same output root because they would write the same `run_manifest.json`.

Dry-run one seed first:

```bash
bsub -env "SEED_LABELS=s01,DRY_RUN=true" < hpc/final_experiment/run_final_experiment.lsf.sh
```

## Outputs

Outputs are written under:

```text
outputs/pinn/final_experiment/<timestamp>/
```

Each run writes normal PINN artifacts, including:
- `config.yaml`
- `metrics.json`
- `timings.json`
- `epoch_metrics.csv`
- `checkpoints/best.pt`
- `checkpoints/epoch_060pct.pt`
- `checkpoints/epoch_100pct.pt`

The launcher writes:
- `run_manifest.json`
- `summary.csv`
- `summary.json`
- `failures.json`

## Notes

- `dn_adam3000_ssbroyden2000` is implemented as one single-stage PINN run with two optimizer phases. DN weighting is active for the full run, so the SSBroyden phase refines the same adaptively weighted objective rather than switching to a separate static objective.
- The default checkpoint fractions are `0.6,1.0`. For the DN strategy, `epoch_060pct.pt` is the Adam-to-SSBroyden boundary after 3000 of 5000 epochs.
- Use `--epoch-scale` only for smoke checks. The final thesis matrix should use the default scale `1.0`.

## Loss Landscapes

Loss landscapes are post-processing jobs. Training writes the required checkpoints, but does not evaluate landscapes inline.

Dry-run a small landscape subset:

```bash
python3 -m src.experiments.pipeline.run_final_landscapes \
  --run-manifest outputs/pinn/final_experiment/<tag>/run_manifest.json \
  --models SM4 \
  --seed-labels s01 \
  --strategies data_only_adam_5000,dn_adam3000_ssbroyden2000,ssbroyden_5000 \
  --checkpoint-tags best,epoch_100pct \
  --dry-run
```

By default, the landscape launcher uses `--weight-source auto`:

```text
DN strategy -> checkpoint active weights
other strategies -> configured static weights
```

This means DN landscapes evaluate the adaptive objective stored in the checkpoint metrics, while static/data-only landscapes use their configured objective. For a common-objective comparison across all methods, pass:

```bash
--weight-source config
```

For the DN phase-boundary visualization, keep `--include-dn-boundary` enabled. This adds `epoch_060pct` for DN runs.

HPC example:

```bash
bsub -env "RUN_MANIFEST=outputs/pinn/final_experiment/<tag>/run_manifest.json,MODELS=SM4,SEED_LABELS=s01,DRY_RUN=true" \
  < hpc/final_experiment/run_final_landscapes.lsf.sh
```
