# Loss Balancing Experiment

## Objective
Evaluate how loss formulation and loss weighting affect PINN training stability, convergence behavior, and final predictive accuracy under a controlled setup.

The goal is to isolate the effect of loss design while keeping the dataset, model, optimizer, and collocation setup fixed.

## Research Questions
- Does physics-informed training improve performance compared to data-only training?
- Which static or adaptive loss weighting scheme stabilizes PINN training most effectively?
- How do data, physics, derivative, and initial-condition loss terms interact during training?
- Are adaptive weighting schemes necessary, or are simpler static or warmup strategies sufficient?
- Do weighting strategies improve convergence stability and final test accuracy?

## Hypotheses
- Poorly scaled static weights lead to unstable or imbalanced training due to differing loss magnitudes.
- Tuned static weights provide a stable baseline but may underweight physics constraints.
- Adaptive weighting improves balance and reduces sensitivity to manual tuning.
- Data warmup followed by full PINN training improves stability.
- The best method depends on whether performance is measured by final accuracy or training stability.

## Experimental Setup
- Task: single-stage PINN training
- Dataset: one fixed preprocessed reference dataset
- Model: fixed architecture, activation, initialization
- Dtype: float64
- Optimizer: Adam with ReduceLROnPlateau scheduler
- Epochs: 20,000
- Collocation: fixed strategy and budget
- Evaluation: fixed train/validation/test split
- Only loss formulation and weighting scheme vary

## Strategies

### Static Baselines
- `data_only` — supervised loss only; no physics terms
- `static_uniform` — equal weights on all loss terms
- `static_tuned` — hand-tuned static weights
- `data_warmup_static` — data-only warmup phase, then switch to tuned static weights

### Adaptive Schemes
- `id` — inverse Dirichlet
- `dn` — dynamic normalization
- `ma` — moving average
- `ntk` — NTK-based random batch weighting
- `relobralo` — relative loss balancing with random lookback

### Extended / Legacy
- `lra` — learning-rate annealing (paper variant); not included in the default core matrix

## Run Matrix
Each run is defined by:

`loss_weighting_strategy × seed`

Execution plan:
- Screening phase: 1 seed per strategy, 100 epochs
- Final comparison: 5 seeds per strategy, 20,000 epochs

All runs share:
- same dataset
- same architecture
- same optimizer and scheduler configuration
- same collocation setup

## Controlled Variables
The following are fixed across all runs:
- dataset and preprocessing
- model architecture and activation
- dtype
- optimizer type and hyperparameters
- collocation strategy and budget
- evaluation setup
- logging and artifact structure

## Metrics

### Performance
- Test RMSE / MSE / MAE

### Convergence
- Total loss vs epoch
- Total loss vs walltime

### Loss Behaviour
- Per-component losses: data, physics, dt, IC
- Relative scaling of loss terms over time
- Weight evolution for adaptive schemes
- Gradient norms per component

### Cost
- Total walltime
- Epoch walltime
- Number of training steps

### Stability
- Successful completion
- NaN / divergence detection
- Variance across seeds (mean and standard deviation)
- Oscillations and loss plateaus
- Boundedness of adaptive weights

## Expected Outcome
Identify which weighting strategies stabilize PINN training for stiff power-system ODEs, and whether adaptive schemes provide a meaningful advantage over static or warmup approaches.

## Extended Comparison: Loss Design vs Optimizer Choice
After identifying the best weighting strategy, a targeted follow-up evaluates:
- Adam + best weighting strategy
- Second-order optimizer + fixed static weights
- Adam + best weighting → short LBFGS refinement phase

Research questions:
- Can first-order training with dynamic weighting match second-order optimization?
- Does LBFGS refinement improve final accuracy?

Hypothesis: dynamic weighting improves Adam significantly, but LBFGS refinement is expected to give the best final convergence.

## How to Run

Run from the repository root. HPC wrappers source shared defaults from `hpc/common/lsf_defaults.sh`; the static `#BSUB` directives remain the submitted resources. Use inline environment variables only. Do not use `bsub -env` or `export` blocks in runbook commands.

PINN wrappers expose `MODEL_FLAG`, `DTYPE`, `ID_EVAL_ID`, `OOD_EVAL_ID`, and `NO_OOD_EVAL`. OOD evaluation is enabled by default using the launcher default unless `NO_OOD_EVAL=true` / `--no-ood-eval` is used. ID evaluation is optional and must be requested explicitly.

### 1. Prepare The Environment

```bash
cd /path/to/ml-surrogates-for-power-systems
python3 --version
python3 -m src.experiments.pipeline.run_weighting_comparison --help
```

Make sure `python3` resolves to the intended project environment before submitting jobs.

### 2. Generate Or Verify The Reference Dataset

Dry-run first:

```bash
REFERENCE_IDS=main_SM4_qbc_b512_ds01 DRY_RUN=true \
  bsub < hpc/reference_datasets/generate_reference_datasets.lsf.sh
```

Then submit the real job:

```bash
REFERENCE_IDS=main_SM4_qbc_b512_ds01 DRY_RUN=false FORCE_REBUILD=false \
  bsub < hpc/reference_datasets/generate_reference_datasets.lsf.sh
```

Expected reference outputs:

```text
data/reference/main/SM4/qbc_deep_ensemble/b512/ds01/
data/reference/index.json
```

The `data/reference/index.json` entry for `main_SM4_qbc_b512_ds01` must contain the `preprocessed_root` consumed by PINN training.

### 3. Local Screening Run

For a quick local sanity check with a small dataset:

```bash
python3 -m src.experiments.pipeline.run_weighting_comparison \
  --mode screening \
  --reference-id smoke_SM4_lhs_b256_ds01 \
  --strategies static_tuned,ma,ntk \
  --seed-labels s01
```

### 4. Dry-Run The Final Matrix

```bash
MODE=final REFERENCE_ID=main_SM4_qbc_b512_ds01 DRY_RUN=true \
  bsub < hpc/weighting_comparison/run_weighting_comparison.lsf.sh
```

The default final matrix uses 9 core strategies and 5 seeds:

```text
9 strategies × 5 seeds = 45 runs
```

### 5. Submit The Final Comparison

```bash
MODE=final REFERENCE_ID=main_SM4_qbc_b512_ds01 DEVICE=cuda DRY_RUN=false \
  bsub < hpc/weighting_comparison/run_weighting_comparison.lsf.sh
```

Final mode uses:

```text
seed labels:   s01,s02,s03,s04,s05
epochs:        20,000
scheduler:     ReduceLROnPlateau (patience=500, factor=0.5, min_lr=1e-6)
W&B project:   thesis-weighting-experiment
```

Equivalent Python command for reference:

```bash
python3 -m src.experiments.pipeline.run_weighting_comparison \
  --mode final \
  --reference-id main_SM4_qbc_b512_ds01 \
  --strategies data_only,static_uniform,static_tuned,data_warmup_static,id,dn,ma,ntk,relobralo \
  --seed-labels s01,s02,s03,s04,s05
```

### 6. Optional Screening Run On The Cluster

For a smaller pre-final check:

```bash
MODE=screening REFERENCE_ID=smoke_SM4_lhs_b256_ds01 \
  STRATEGIES=static_tuned,ma,ntk SEED_LABELS=s01 DEVICE=cuda DRY_RUN=false \
  bsub < hpc/weighting_comparison/run_weighting_comparison.lsf.sh
```

Screening mode uses 100 epochs and the W&B project `thesis-weighting-experiment-TEST`.

### 7. Inspect Outputs

Cluster logs are written under:

```text
hpc/logs/
```

Experiment outputs are written under:

```text
outputs/pinn/weighting_comparison/<timestamp>/
```

Key result files:

```text
run_manifest.json
summary.csv
summary.json
failures.json
logs/runs/*.log
runs/<strategy>_<seed>/metrics.json
runs/<strategy>_<seed>/timings.json
```

Use `summary.csv` and `summary.json` as the comparison-level result artifacts. If a job fails, inspect `failures.json` first, then the corresponding file under `logs/runs/`.

## Outputs

| File | Contents |
|---|---|
| `summary.csv` | One row per run: accuracy, loss components, active weights, timing |
| `summary.json` | Same data with per-strategy aggregates (mean, std) |
| `failures.json` | Subset of failed runs with return codes and log paths |
| `runs/<name>/metrics.json` | Full epoch metrics for one run |
| `runs/<name>/timings.json` | Walltime breakdown for one run |

W&B project is `thesis-weighting-experiment` in final mode and `thesis-weighting-experiment-TEST` in screening mode.

## Notes
- Console logging is reduced (`log_every_epoch=10`). Full per-epoch resolution is available in W&B.
- Gradient telemetry is enabled for all runs (`pinn.gradient_telemetry.enabled=true`). Component grad norms and weighted grad norms are logged to W&B under `train/grad_norm/{name}` and `train/grad_weighted_norm/{name}`.
- Active weights at the final epoch are recorded in `summary.csv` as `train_weight_{data,physics,dt,ic}`.
- Smoke/compressed datasets may have a time grid that differs from external evaluation datasets. Use `NO_OOD_EVAL=true` or `--no-ood-eval` for those checks.
- Smoke/reference smoke datasets disable validation. Normal/final datasets keep validation by default.
