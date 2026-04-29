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
- Train a baseline data-driven surrogate, not a PINN.
- Fixed training setup: architecture, optimizer, and training protocol.
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
sampling_strategy × trajectory_budget × dataset_seed × baseline_seed
```

Planned matrix:

```text
4 strategies × 5 budgets × 5 dataset seeds × 3 baseline seeds = 300 downstream training runs
```

- Dataset generation runs once per `strategy × budget × dataset_seed`.
- Each generated dataset is reused for 3 baseline seeds (`bs01`, `bs02`, `bs03`).
- Dataset seeds: `ds01`–`ds05`.

## Evaluation

- External ID evaluation dataset is the primary in-distribution metric.
- External OOD evaluation dataset is the primary generalization metric.
- Internal dataset test split is secondary / debug only.

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

Generated datasets can become large on HPC. The pipeline supports a cleanup mode (`--cleanup-data`) that deletes large generated data artifacts — specifically `data/`, `qbc/rounds/`, and `qbc/checkpoints/` — after downstream baseline runs complete. Metrics, manifests, logs, configs, and summary files are always retained. `data/reference` and `data/evaluation` are never touched. Cleanup is disabled by default.

## How to Run

Run commands from the repository root.

HPC wrappers source shared defaults from `hpc/common/lsf_defaults.sh`; the static `#BSUB` directives remain the submitted resources. Use inline environment variables only. Do not use `bsub -env` or `export` blocks in runbook commands.

The dataset-generation wrapper exposes `ID_EVAL_ID`, `OOD_EVAL_ID`, `NO_ID_EVAL`, `NO_OOD_EVAL`, and `CLEANUP_DATA`. External ID/OOD evaluation is the primary metric for this experiment, but smoke/compressed datasets may require `NO_ID_EVAL=true NO_OOD_EVAL=true` if their time grid is incompatible with fixed evaluation datasets.

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

### With cleanup (local)

```bash
python3 -m src.experiments.pipeline.run_dataset_generation_comparison \
  --mode final \
  --model-flag SM4 \
  --cleanup-data
```

### Full run (HPC)

```bash
MODE=final MODEL_FLAG=SM4 \
  bsub < hpc/dataset_generation_comparison/run_dataset_generation_comparison.lsf.sh
```

### Full run with cleanup (HPC)

```bash
MODE=final MODEL_FLAG=SM4 CLEANUP_DATA=true \
  bsub < hpc/dataset_generation_comparison/run_dataset_generation_comparison.lsf.sh
```

### Dry-run check (HPC)

```bash
MODE=smoke DRY_RUN=true \
  bsub < hpc/dataset_generation_comparison/run_dataset_generation_comparison.lsf.sh
```

### Smoke check without external eval (HPC)

```bash
MODE=smoke NO_ID_EVAL=true NO_OOD_EVAL=true DRY_RUN=true \
  bsub < hpc/dataset_generation_comparison/run_dataset_generation_comparison.lsf.sh
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

`summary.csv` columns include: `method`, `budget`, `dataset_seed`, `baseline_seed`, internal test metrics, `id_eval` metrics, `ood_eval` metrics, `id_ood_rmse_gap`, and timing fields where available.

Cluster logs are written under:

```text
hpc/logs/
```

## Notes

- Smoke/reference smoke datasets disable validation. Normal/final datasets keep validation by default.
- Cleanup is opt-in via `CLEANUP_DATA=true` / `--cleanup-data`; metrics, manifests, logs, configs, and summaries are retained.
- `data/reference` and `data/evaluation` are never pruned by the dataset-generation comparison cleanup path.

## Expected Outcome

This experiment identifies which dataset generation strategy produces the most useful supervised training data for power-system ODE surrogate modeling. It quantifies sample efficiency, evaluates OOD generalization, and separates dataset-generation variability from baseline training variability across a full `strategy × budget × seed` matrix.
