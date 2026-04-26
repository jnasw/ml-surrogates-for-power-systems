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
- Evaluation: fixed in-distribution train/test split
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

The experiment should be run from the repository root on the cluster. The cluster does not support `bsub -env`, so set environment variables with `export` before calling `bsub`.

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
export REFERENCE_IDS=main_SM4_qbc_b512_ds01
export DRY_RUN=true
bsub < hpc/reference_datasets/generate_reference_datasets.lsf.sh
```

Then submit the real reference dataset job:

```bash
export REFERENCE_IDS=main_SM4_qbc_b512_ds01
export DRY_RUN=false
export FORCE_REBUILD=false
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
export MODE=final
export REFERENCE_ID=main_SM4_qbc_b512_ds01
export DRY_RUN=true
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
export MODE=final
export REFERENCE_ID=main_SM4_qbc_b512_ds01
export DEVICE=cuda
export DRY_RUN=false
bsub < hpc/optimizer_comparison/run_optimizer_comparison.lsf.sh
```

Final mode uses:

```text
seed labels: s01,s02,s03,s04,s05
main optimizer epochs: 100
Adam warmup epochs: 100
W&B project: thesis-optimizer-experiment
```

### 5. Optional Screening Run

For a smaller pre-final check:

```bash
export MODE=screening
export REFERENCE_ID=main_SM4_qbc_b512_ds01
export STRATEGIES=adam,lbfgs,adam_lbfgs
export SEED_LABELS=s01
export DEVICE=cuda
export DRY_RUN=false
bsub < hpc/optimizer_comparison/run_optimizer_comparison.lsf.sh
```

Screening mode uses 5 main epochs, 5 Adam warmup epochs, and the W&B project `thesis-optimizer-experiment-TEST`.

### 6. Inspect Outputs

Cluster logs are written under:

```text
hpc/logs/
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
```

Use `summary.csv` and `summary.json` as the comparison-level result artifacts. If a job fails, inspect `failures.json` first, then the corresponding file under `logs/runs/`.
