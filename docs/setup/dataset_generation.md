# Experiment 1: Dataset Generation

## Objective

Evaluate how different trajectory sampling strategies affect downstream surrogate accuracy, OOD generalization, and sample efficiency under a fixed simulator-call budget. The goal is to compare dataset quality, not model design.

## Research Questions

- Which sampling strategy achieves the lowest prediction error for a fixed simulator budget?
- Do sampling strategies saturate as dataset size increases?
- Are adaptive methods mainly beneficial in low-data regimes?
- Do adaptive methods improve OOD generalization or mostly ID performance?
- Do QBC, marker-based, and hybrid methods select fundamentally different trajectories?
- Is adaptive sampling more efficient than simply increasing LHS dataset size?

## Hypotheses

- Adaptive methods outperform LHS at small-to-medium budgets by selecting informative trajectories.
- The performance gap decreases at large dataset sizes.
- QBC focuses on uncertainty, while marker methods focus on dynamical difficulty.
- Adaptive methods introduce overhead, so they must be evaluated relative to total cost.
- Adaptive sampling may improve OOD performance if it covers more informative regions.

## Experimental Setup

- Compare dataset generation strategies across fixed trajectory budgets.
- Use identical preprocessing for all datasets.
- Train one fixed downstream surrogate per generated dataset.
- Downstream model: `pinn_data_only`, which uses the PINN architecture with only supervised data loss active.
- Fixed downstream setup: architecture, optimizer, and training protocol.
- Evaluation uses fixed external ID and OOD datasets.
- Generated datasets are ephemeral and may be deleted after use when cleanup is enabled.

## Sampling Strategies

- `lhs_static`
- `qbc_deep_ensemble`
- `marker_directed`
- `qbc_marker_hybrid`

## Budget Ladder

| Label | Trajectories |
|-------|-------------|
| `b256` | 256 |
| `b512` | 512 |
| `b1024` | 1024 |
| `b2048` | 2048 |
| `b4096` | 4096 |

## Run Matrix

Each run is defined by:

```text
sampling_strategy × trajectory_budget × dataset_seed × model_seed
```

Planned matrix:

```text
4 strategies × 5 budgets × 5 dataset seeds × 3 model seeds = 300 downstream training runs
```

- Dataset generation runs once per `strategy × budget × dataset_seed`.
- Each generated dataset is reused for 3 downstream model seeds (`bs01`, `bs02`, `bs03`).
- Dataset seeds: `ds01`–`ds05`.

## Evaluation

- External ID evaluation dataset is the primary in-distribution metric.
- External OOD evaluation dataset is the primary generalization metric.
- Internal dataset test split is secondary / debug only.
- Dataset-generation comparisons use larger independent default evaluation sets: `id_SM4_lhs_b4096_eval01` and `ood_SM4_wide_ic_b4096_eval01`. Other experiment pipelines may still use the smaller `b512` evaluation defaults.

## Metrics

### Performance
- ID RMSE / MSE / MAE
- OOD RMSE / MSE / MAE
- ID-OOD gap

### Sample Efficiency
- ID/OOD error vs trajectory budget
- Improvement over LHS at equal budget
- Budget needed to reach a target error threshold

### Robustness
- Mean and standard deviation across dataset seeds
- Mean and standard deviation across baseline seeds

### Cost
- Number of simulator calls
- Dataset generation walltime
- Preprocessing time
- Downstream training time
- Total pipeline time
- Adaptive-method overhead (if available)

## Storage Strategy

Generated datasets can become large on HPC. The pipeline supports a cleanup mode (`--cleanup-data`) that deletes large generated data artifacts — specifically `data/`, `qbc/rounds/`, and `qbc/checkpoints/` — after downstream PINN data-only runs complete. Metrics, manifests, logs, configs, checkpoints, and summary files are always retained. `data/reference` and `data/evaluation` are never touched. Cleanup is disabled by default.

QBC artifact storage can also be controlled before the run writes large intermediates. When QBC logging is enabled, `qbc/history.jsonl` is always retained. Set `QBC_SAVE_ROUND_ARRAYS=false` to skip `qbc/rounds/`, `QBC_SAVE_DATASET_CHECKPOINTS=false` to skip dataset checkpoints, and `QBC_SAVE_ENSEMBLE_CHECKPOINTS=false` to skip ensemble checkpoints. Keep dataset checkpoints enabled for QBC resume workflows.

## Downstream Model Choices

The launcher supports one downstream model:

| Value | Meaning | Use |
|-------|---------|-----|
| `pinn_data_only` | `20_run_pinn.py` with data loss active and physics/dt/IC weights set to zero | Default dataset-generation downstream evaluator |

`pinn_data_only` makes the dataset-generation study use the same input/output formulation as the PINN experiments while removing physics supervision. It does not change simulator data or preprocessing.

The `pinn_data_only` path exposes `DEVICE`, `DTYPE`, `BATCH_SIZE`, and `ADAM_LR`. The current calibrated Adam LR is `0.003`. It saves the best validation checkpoint by default and reports both final metrics and best-checkpoint metrics in the summary artifacts.

## How to Run

Run commands from the repository root.

HPC wrappers source shared defaults from `hpc/common/lsf_defaults.sh`; the static `#BSUB` directives remain the submitted resources. Use `bsub -env` to pass environment variables — inline `VAR=val bsub` does not propagate to the batch environment on this cluster. When a variable value contains commas (e.g. `DATASET_SEEDS=ds01,ds02`), use the subshell form `(export VAR=val ... && bsub -env "all" < script)` instead.

The dataset-generation wrapper exposes `ID_EVAL_ID`, `OOD_EVAL_ID`, `NO_ID_EVAL`, `NO_OOD_EVAL`, and `CLEANUP_DATA`. External ID/OOD evaluation is the primary metric for this experiment, but smoke/compressed datasets may require `NO_ID_EVAL=true NO_OOD_EVAL=true` if their time grid is incompatible with fixed evaluation datasets.

Generate the default SM4 evaluation datasets before final dataset-generation jobs:

```bash
(export EVALUATION_IDS=id_SM4_lhs_b4096_eval01,ood_SM4_wide_ic_b4096_eval01 \
  DRY_RUN=false \
  FORCE_REBUILD=false && \
  bsub -env "all" < hpc/evaluation_datasets/generate_evaluation_datasets.lsf.sh)
```

### Smoke test (local)

```bash
python3 -m src.experiments.pipeline.run_dataset_generation_comparison \
  --mode smoke \
  --model-flag SM4
```

### Controlled subset (local)

```bash
python3 -m src.experiments.pipeline.run_dataset_generation_comparison \
  --mode final \
  --model-flag SM4 \
  --methods lhs_static \
  --budgets b256 \
  --dataset-seeds ds01 \
  --baseline-seeds bs01,bs02
```

### Larger subset (local)

```bash
python3 -m src.experiments.pipeline.run_dataset_generation_comparison \
  --mode final \
  --model-flag SM4 \
  --methods lhs_static,qbc_deep_ensemble \
  --budgets b256,b512 \
  --dataset-seeds ds01,ds02 \
  --baseline-seeds bs01,bs02,bs03
```

### Data-only PINN downstream model (local)

```bash
python3 -m src.experiments.pipeline.run_dataset_generation_comparison \
  --mode final \
  --model-flag SM4 \
  --methods lhs_static \
  --budgets b256 \
  --dataset-seeds ds01 \
  --baseline-seeds bs01,bs02,bs03 \
  --device cuda \
  --dtype float64 \
  --adam-lr 0.003
```

### With cleanup (local)

```bash
python3 -m src.experiments.pipeline.run_dataset_generation_comparison \
  --mode final \
  --model-flag SM4 \
  --cleanup-data
```

### Full run (HPC)

```bash
bsub -env "MODE=final,MODEL_FLAG=SM4" \
  < hpc/dataset_generation_comparison/run_dataset_generation_comparison.lsf.sh
```

### Full run with cleanup (HPC)

```bash
bsub -env "MODE=final,MODEL_FLAG=SM4,CLEANUP_DATA=true" \
  < hpc/dataset_generation_comparison/run_dataset_generation_comparison.lsf.sh
```

### Split Final Matrix Into Cluster Jobs

For the full final matrix, prefer splitting by dataset-generation cell groups instead of submitting one very large sequential job. Use a shared `CAMPAIGN_TAG` so all split outputs are grouped under:

```text
outputs/experiments/dataset_generation_comparison/<campaign-tag>/<shard-label>/
```

If `SHARD_LABEL` is omitted, the launcher infers one from the selected subset, e.g. `lhs_static`, `lhs_static_b256`, or `qbc_deep_ensemble_b2048_ds01`.

One job per method (values contain commas — use subshell form):

```bash
(export CAMPAIGN_TAG=dataset_final_sm4 MODE=final MODEL_FLAG=SM4 \
  METHODS=lhs_static \
  DATASET_SEEDS=ds01,ds02,ds03,ds04,ds05 \
  BASELINE_SEEDS=bs01,bs02,bs03 \
  CLEANUP_DATA=true && \
  bsub -env "all" -J data_lhs \
    < hpc/dataset_generation_comparison/run_dataset_generation_comparison.lsf.sh)
```

One job per method and budget:

```bash
(export CAMPAIGN_TAG=dataset_final_sm4 MODE=final MODEL_FLAG=SM4 \
  METHODS=qbc_deep_ensemble \
  BUDGETS=b512 \
  DATASET_SEEDS=ds01,ds02,ds03,ds04,ds05 \
  BASELINE_SEEDS=bs01,bs02,bs03 \
  CLEANUP_DATA=true && \
  bsub -env "all" -J data_qbc_b512 \
    < hpc/dataset_generation_comparison/run_dataset_generation_comparison.lsf.sh)
```

One job per expensive adaptive dataset seed:

```bash
(export CAMPAIGN_TAG=dataset_final_sm4 MODE=final MODEL_FLAG=SM4 \
  METHODS=qbc_marker_hybrid \
  BUDGETS=b2048 \
  DATASET_SEEDS=ds01 \
  BASELINE_SEEDS=bs01,bs02,bs03 \
  CLEANUP_DATA=true && \
  bsub -env "all" -J data_hybrid_b2048_ds01 \
    < hpc/dataset_generation_comparison/run_dataset_generation_comparison.lsf.sh)
```

Keep all model seeds for a generated dataset in the same shard when possible, because the launcher generates and preprocesses one dataset per `method × budget × dataset_seed` and then trains all `BASELINE_SEEDS` against it. Splitting by `BASELINE_SEEDS` causes repeated dataset generation.

### Dry-run check (HPC)

```bash
bsub -env "MODE=smoke,DRY_RUN=true" \
  < hpc/dataset_generation_comparison/run_dataset_generation_comparison.lsf.sh
```

### Smoke check without external eval (HPC)

```bash
bsub -env "MODE=smoke,NO_ID_EVAL=true,NO_OOD_EVAL=true,DRY_RUN=true" \
  < hpc/dataset_generation_comparison/run_dataset_generation_comparison.lsf.sh
```

## Outputs

Experiment outputs are written under:

```text
outputs/experiments/dataset_generation_comparison/<timestamp>/
```

Each comparison run produces:

- `run_manifest.json` — full structured manifest of all cell runs
- `summary.csv` — main analysis table, one row per `method × budget × dataset_seed × baseline_seed`
- `summary.json` — same data as `summary.csv` in JSON format
- `failures.json` — details of any failed cells

`summary.csv` columns include: `method`, `budget`, `dataset_seed`, `baseline_seed`, downstream model, final internal test metrics, final `id_eval` metrics, final `ood_eval` metrics, best-checkpoint train/val/test/ID/OOD metrics, best checkpoint path, epoch metrics path, and timing fields where available.

Cluster logs are written under:

```text
hpc/logs/dataset_generation/
```

## Notes

- Smoke/reference smoke datasets disable validation. Normal/final datasets keep validation by default.
- Cleanup is opt-in via `CLEANUP_DATA=true` / `--cleanup-data`; metrics, manifests, logs, configs, and summaries are retained.
- `data/reference` and `data/evaluation` are never pruned by the dataset-generation comparison cleanup path.

## Expected Outcome

This experiment identifies which dataset generation strategy produces the most useful supervised training data for power-system ODE surrogate modeling. It quantifies sample efficiency, evaluates OOD generalization, and separates dataset-generation variability from baseline training variability across a full `strategy × budget × seed` matrix.
