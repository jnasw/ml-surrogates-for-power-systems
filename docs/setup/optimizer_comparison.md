# Optimizer Comparison Experiment

## Objective
Evaluate how optimizer choice affects PINN training performance, convergence behavior, computational cost, and robustness under a controlled setup.

The goal is to isolate optimizer effects while keeping all other components fixed.

## Research Questions
- Do quasi-Newton optimizers outperform Adam in convergence speed and/or final accuracy?
- Which optimizer provides the best accuracy-cost trade-off?
- Do multi-phase strategies improve reliability or final performance?
- Are optimizer rankings consistent when evaluated by accuracy versus walltime?
- Which optimizer strategies are most robust across random seeds?

## Hypotheses
- Quasi-Newton methods will reduce the training objective faster than Adam, but may incur higher per-epoch cost.
- Multi-phase strategies (e.g. Adam -> quasi-Newton) will be more reliable than starting with second-order methods directly.
- The optimizer that performs best in terms of final accuracy may differ from the one that is most efficient in terms of walltime.
- Optimizer performance will vary across seeds, requiring robustness evaluation.

## Experimental Setup
- Task: single-stage PINN training
- Dataset: one fixed preprocessed dataset
- Model: fixed architecture, activation, initialization
- Dtype: float64
- Loss weighting: fixed static weights
- Collocation: fixed strategy and budget
- Evaluation: fixed internal train/validation/test split plus optional external ID/OOD evaluation
- Execution: controlled experiment matrix (no sweeps)

## Optimizer Groups

### Core Single-Optimizer Strategies
- Adam
- BFGS
- LBFGS
- SSBFGS
- SSBroyden
- SOAP

### Experimental Stochastic Strategies
- sSSBFGS
- sSSBroyden

These strategies are currently unstable and are excluded from the default screening and final matrices. They may be run separately with an explicit opt-in, but they should not be included in the main thesis comparison unless their stability has been verified.

### Multi-Phase Strategies
- Adam -> SOAP
- Adam -> BFGS
- Adam -> LBFGS
- Adam -> SSBFGS
- Adam -> SSBroyden

Experimental optional strategies:
- Adam -> sSSBFGS
- Adam -> sSSBroyden

## Run Matrix
Each run is defined by:

`optimizer_strategy x seed`

Execution plan:
- Screening phase: 1 seed per optimizer
- Final comparison: 5 seeds for core/default strategies

All runs share:
- same dataset
- same architecture
- same collocation setup
- same loss configuration
- same calibrated optimizer hyperparameters

Default final matrix:

```text
11 core strategies x 5 seeds = 55 runs
```

Experimental stochastic self-scaled strategies are excluded by default and must be requested explicitly.

## Calibrated Optimizer Hyperparameters

Optimizer learning rates are fixed from the calibration stage before running the final controlled comparison.

| Optimizer | Default LR |
|-----------|------------|
| Adam | 0.003 |
| SOAP | 0.05 |
| LBFGS | 0.1 |
| BFGS | 1.0 |
| SSBFGS | 0.3 |
| SSBroyden | 1.0 |

The launcher exposes per-optimizer overrides:

```text
--adam-lr
--soap-lr
--lbfgs-lr
--bfgs-lr
--ssbfgs-lr
--ssbroyden-lr
```

`--quasi-newton-lr` is retained as a compatibility override that applies one shared learning rate to all full-batch quasi-Newton phases. It should not be used for the final thesis matrix unless intentionally testing a shared-LR variant.

Adam phases use a `ReduceLROnPlateau` scheduler by default. In final mode, the metric is `val_total_loss` when validation exists. The scheduler reduces the Adam learning rate by a fixed factor after plateauing, which makes long Adam phases less sensitive to the initial calibrated LR.

## Epoch Budget

The final comparison uses the same total epoch budget for single-optimizer and multi-phase strategies:

```text
total epochs per run: 5000
Adam warmup for Adam -> optimizer strategies: 500
second optimizer phase: 4500
```

Single-optimizer strategies use all 5000 epochs. Multi-phase strategies use 500 Adam warmup epochs plus 4500 epochs of the second optimizer, not 500 + 5000.

Screening mode uses a compressed budget:

```text
total epochs per run: 10
Adam warmup: 5
second optimizer phase: 5
```

## Controlled Variables
The following are fixed across all runs:
- dataset and preprocessing
- model architecture and activation
- dtype
- collocation strategy and budget
- loss formulation and weights
- evaluation setup
- logging and artifact structure

## Metrics

- Test RMSE/MSE and MAE
- Convergence vs epoch
- Convergence vs walltime
- Secondary: Final loss components (data, physics, dt, IC)

## Cost and Stability

### Cost
- Total walltime
- Epoch walltime
- Number of epochs
- Number of training steps

### Stability
- Successful completion (yes/no)
- Failure type (NaN, optimizer failure, timeout, etc.)
- Variance across seeds (mean and standard deviation)
- Multi-phase transition success

## Evaluation Protocol
- All optimizers are trained on the same dataset and model
- Each configuration is repeated across multiple seeds
- Results are aggregated using mean and standard deviation
- Optimizers are compared based on:
  - final accuracy
  - convergence behavior
  - computational cost
- OOD evaluation is enabled by default using `ood_SM4_wide_ic_b512_ds01`
- ID evaluation is optional and must be requested explicitly
- Best checkpoint metrics are selected by the lowest validation loss when validation exists

## Potential Outputs
- Loss vs epoch
- Loss vs walltime
- Test error vs walltime
- Final accuracy per optimizer
- Accuracy vs runtime trade-off
- Error bars across seeds
- Success/failure summary

## Expected Outcome
This experiment provides a controlled comparison of first-order, quasi-Newton, and multi-phase optimization strategies for PINN training. It identifies which approaches offer the best balance between accuracy, efficiency, and robustness under fixed conditions.

## Reproducing Results On The Cluster

The final optimizer comparison uses the persistent reference dataset:

```text
main_SM4_qbc_b512_ds01
```

Run from the repository root. HPC wrappers source shared defaults from `hpc/common/lsf_defaults.sh`; the static `#BSUB` directives remain the submitted resources. Use inline environment variables only. Do not use `bsub -env` or `export` blocks in runbook commands.

PINN wrappers expose `MODEL_FLAG`, `DTYPE`, `ID_EVAL_ID`, `OOD_EVAL_ID`, and `NO_OOD_EVAL`. OOD evaluation is enabled by default using the launcher default unless `NO_OOD_EVAL=true` / `--no-ood-eval` is used. ID evaluation is optional and must be requested explicitly.

### 1. Prepare The Environment

```bash
cd /path/to/ml-surrogates-for-power-systems
python3 --version
python3 -m src.experiments.pipeline.run_optimizer_comparison --help
```

Make sure `python3` resolves to the intended project environment before submitting jobs.

### 2. Generate Or Verify The Reference Dataset

First dry-run the reference dataset job:

```bash
REFERENCE_IDS=main_SM4_qbc_b512_ds01 DRY_RUN=true \
  bsub < hpc/reference_datasets/generate_reference_datasets.lsf.sh
```

Then submit the real reference dataset job:

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

### 3. Dry-Run The Final Optimizer Matrix

```bash
MODE=final REFERENCE_ID=main_SM4_qbc_b512_ds01 DRY_RUN=true \
  bsub < hpc/optimizer_comparison/run_optimizer_comparison.lsf.sh
```

The default final matrix should include 11 core strategies and 5 seeds:

```text
11 strategies x 5 seeds = 55 runs
```

The default final matrix excludes:

```text
sssbfgs
sssbroyden
adam_sssbfgs
adam_sssbroyden
```

Run these only as explicitly labeled experimental follow-up runs.

### 4. Submit The Final Optimizer Comparison

```bash
MODE=final REFERENCE_ID=main_SM4_qbc_b512_ds01 DEVICE=cuda DRY_RUN=false \
  bsub < hpc/optimizer_comparison/run_optimizer_comparison.lsf.sh
```

Final mode uses:

```text
seed labels: s01,s02,s03,s04,s05
total epochs per run: 5000
Adam warmup epochs: 500
second optimizer epochs for Adam -> optimizer strategies: 4500
W&B project: thesis-optimizer-experiment
```

### 5. Optional Screening Run

For a smaller pre-final check:

```bash
MODE=screening REFERENCE_ID=main_SM4_qbc_b512_ds01 \
  STRATEGIES=adam,lbfgs,adam_lbfgs SEED_LABELS=s01 DEVICE=cuda DRY_RUN=false \
  bsub < hpc/optimizer_comparison/run_optimizer_comparison.lsf.sh
```

Screening mode uses 10 total epochs, 5 Adam warmup epochs for multi-phase strategies, and the W&B project `thesis-optimizer-experiment-TEST`.

### 6. Inspect Outputs

Cluster logs are written under:

```text
hpc/logs/optimizer/
```

Experiment outputs are written under:

```text
outputs/pinn/optimizer_comparison/<timestamp>/
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
runs/<strategy>_<seed>/epoch_metrics.csv
runs/<strategy>_<seed>/checkpoints/best.pt
```

Use `summary.csv` and `summary.json` as the comparison-level result artifacts. If a job fails, inspect `failures.json` first, then the corresponding file under `logs/runs/`.

Notes:
- Smoke/compressed datasets may have a time grid that differs from external evaluation datasets. Use `NO_OOD_EVAL=true` or `--no-ood-eval` for those checks.
- Smoke/reference smoke datasets disable validation. Normal/final datasets keep validation by default.
- Gradient telemetry is disabled by default for optimizer comparison.
