# Collocation Point Sampling Experiment

## Objective
Evaluate how different collocation point sampling strategies affect PINN training accuracy, convergence behavior, and computational efficiency under a controlled setup.

The goal is to isolate how physics enforcement through collocation affects model performance while keeping dataset, optimizer, and loss weighting fixed.

## Research Questions
- Do adaptive collocation strategies improve accuracy compared to static sampling?
- Which collocation method provides the best performance for a fixed collocation budget?
- Can adaptive methods achieve the same accuracy with fewer collocation points?
- How does performance scale with active collocation budget?
- At what collocation budget do performance gains begin to saturate?
- Do adaptive methods improve OOD generalization or mainly reduce local residual errors?
- Can residuals on unseen collocation points serve as a proxy for generalization error?

## Hypotheses
- Static collocation may require higher active budgets to enforce physics effectively.
- Adaptive methods focus sampling on high-residual regions and improve efficiency.
- Adaptive strategies can achieve comparable accuracy at lower collocation density.
- Increasing the active collocation budget yields diminishing returns beyond a certain threshold.

## Experimental Setup
- Single-stage PINN training
- Fixed reference dataset
- Fixed model architecture, activation, initialization
- Fixed optimizer: Adam
- Fixed static loss weighting
- Fixed train/test and optional OOD evaluation setup
- Collocation strategy and active collocation budget are the only variables

## Collocation Strategies
- Uniform LHS baseline
- Random resampling dynamic baseline
- Adaptive methods:
  - RAD
  - RAR-D
  - optional: RAR-G

## Collocation Budget
Collocation budget is defined as an absolute active point count. This keeps the experiment aligned with the default PINN collocation setting and avoids tying the budget to the size of a specific preprocessed pool.

Budget levels:
- Default: 4,096 (`p4k`)
- Medium: 32,768 (`p32k`)
- High: 65,536 (`p64k`)

## Experimental Procedure

### Stage 1: Cadence Calibration
Adaptive methods require a refresh frequency for updating collocation points.

Evaluate:
- Strategies: random resampling, RAD, RAR-D
- Fixed budget: `p32k`
- Single seed: `s01`
- Cadence: 100, 500, 1000, 2000 epochs

Selection is based on:
- final error
- stability
- computational cost

Outcome: one fixed cadence for all adaptive methods.

### Stage 2: Main Comparison
Using the selected cadence:
- Strategies: uniform LHS, random resampling, RAD, RAR-D
- Budgets: `p4k`, `p32k`, `p64k`
- Seeds: `s01`-`s05`

Run matrix:

```text
strategy x budget x seed = 4 x 3 x 5 = 60 runs
```

## Metrics

### Performance
- Test RMSE / MSE / MAE
- Final data loss

### Generalization
- ID test error
- OOD test error
- ID-OOD gap

### Physics Behaviour
- Training residual loss over time
- Residual on unseen collocation points
- Residual distribution statistics

### Convergence
- Loss vs epoch
- Loss vs walltime

### Cost
- Total training walltime
- Collocation update overhead
- Number of residual evaluations

## Potential Outputs
- Test error vs collocation budget
- Accuracy vs computational cost
- Static vs adaptive efficiency comparison
- Residual vs test error correlation
- Collocation point distribution plots
- Sample-efficiency curves: accuracy vs active collocation budget

## Expected Outcome
This experiment determines how collocation point placement affects physics-informed learning efficiency. It identifies whether adaptive sampling provides meaningful gains over static methods, whether similar accuracy can be achieved at lower active collocation budgets, and at which point increasing the budget yields diminishing returns. It also evaluates whether residual-based metrics can serve as reliable indicators of model generalization.

## Implementation Details

The canonical launcher is:

```bash
python3 -m src.experiments.pipeline.run_collocation_comparison
```

Collocation strategy mapping:
- `uniform_lhs` -> preprocessed static collocation
- `random_resampling` -> `random_r`, the dynamic random-resampling baseline
- `rad` -> generated residual adaptive distribution sampling
- `rar_d` -> generated residual adaptive refinement with distribution sampling
- `rar_g` -> generated residual adaptive refinement with greedy top-residual sampling

Budget is configured as an absolute active point count:

```text
p4k  = 4096 active points
p32k = 32768 active points
p64k = 65536 active points
```

For `uniform_lhs`, this is a fixed subset of the preprocessed collocation pool. For generated adaptive methods, the same active point budget is sampled from the continuous generated domain.

Cadence is controlled via:

```text
pinn.collocation.refresh_period_epochs
```

Important: the first refresh happens after the cadence boundary. For example, cadence `100` first refreshes at epoch `101`, so calibration runs must be long enough for refreshes to occur.

RAR-D and RAR-G behavior:
- start with a partial collocation set, approximately 50% of the target active points
- append points over time until the target active budget is reached
- use `pinn.collocation.append_points`, derived by the launcher unless explicitly changed in code later

## How To Run

Run commands from the repository root.

### 1. Generate Reference Dataset If Needed

```bash
python3 -m src.experiments.pipeline.run_reference_datasets \
  --reference-id smoke_SM4_lhs_b256_ds01
```

For the main thesis comparison, the intended default reference dataset is:

```bash
python3 -m src.experiments.pipeline.run_reference_datasets \
  --reference-id main_SM4_qbc_b512_ds01
```

### 2. Cadence Calibration Locally

```bash
python3 -m src.experiments.pipeline.run_collocation_comparison \
  --mode cadence \
  --reference-id smoke_SM4_lhs_b256_ds01 \
  --device cpu
```

Optional quick test:

```bash
python3 -m src.experiments.pipeline.run_collocation_comparison \
  --mode cadence \
  --reference-id smoke_SM4_lhs_b256_ds01 \
  --strategies random_resampling,rad,rar_d \
  --cadences 100,500 \
  --device cpu \
  --epochs 600
```

### 3. Main Experiment Locally

```bash
python3 -m src.experiments.pipeline.run_collocation_comparison \
  --mode final \
  --reference-id main_SM4_qbc_b512_ds01 \
  --strategies uniform_lhs,random_resampling,rad,rar_d \
  --budgets p4k,p32k,p64k \
  --seed-labels s01,s02,s03,s04,s05 \
  --refresh-period-epochs 1000
```

### 4. HPC Runs

HPC wrappers source shared defaults from `hpc/common/lsf_defaults.sh`; the static `#BSUB` directives remain the submitted resources. Use `bsub -env` to pass environment variables — inline `VAR=val bsub` does not propagate to the batch environment on this cluster. When a variable value contains commas (e.g. `SEED_LABELS=s01,s02`), use the subshell form `(export VAR=val ... && bsub -env "all" < script)` instead.

PINN wrappers expose `MODEL_FLAG`, `DTYPE`, `ID_EVAL_ID`, `OOD_EVAL_ID`, and `NO_OOD_EVAL`. OOD evaluation is enabled by default using the launcher default unless `NO_OOD_EVAL=true` / `--no-ood-eval` is used. ID evaluation is optional and must be requested explicitly.

Cadence:

```bash
bsub -env "MODE=cadence,REFERENCE_ID=smoke_SM4_lhs_b256_ds01" \
  < hpc/collocation_comparison/run_collocation_comparison.lsf.sh
```

Main (values contain commas — use subshell form):

```bash
(export MODE=final REFERENCE_ID=main_SM4_qbc_b512_ds01 \
  STRATEGIES=uniform_lhs,random_resampling,rad,rar_d \
  DENSITIES=p4k,p32k,p64k \
  SEED_LABELS=s01,s02,s03,s04,s05 \
  REFRESH_PERIOD_EPOCHS=1000 && \
  bsub -env "all" < hpc/collocation_comparison/run_collocation_comparison.lsf.sh)
```

Optional dry-run:

```bash
bsub -env "MODE=cadence,REFERENCE_ID=smoke_SM4_lhs_b256_ds01,DEVICE=cpu,DRY_RUN=true" \
  < hpc/collocation_comparison/run_collocation_comparison.lsf.sh
```

## Outputs

Experiment outputs are written under:

```text
outputs/pinn/collocation_comparison/<timestamp>/
```

Each comparison run produces:
- `run_manifest.json`
- `summary.csv`
- `summary.json`
- `failures.json`
- per-run `metrics.json`
- per-run `timings.json`
- per-run logs
- W&B logs when enabled

Cluster logs are written under:

```text
hpc/logs/collocation/
```

## Notes / Important Details
- Cadence must be smaller than total epochs to trigger refresh.
- `random_resampling` is the key baseline for isolating adaptivity.
- `rar_g` is optional and not part of the core comparison.
- Adaptive methods may use fewer points early in training, especially RAR-D and RAR-G.
- Density is a budget, not an exact sampling fraction for adaptive generated methods.
- Smoke/compressed datasets may have a time grid that differs from external evaluation datasets. Use `NO_OOD_EVAL=true` or `--no-ood-eval` for those checks.
- Smoke/reference smoke datasets disable validation. Normal/final datasets keep validation by default.
- Gradient telemetry is disabled by default for collocation comparison.
- Unseen residual metrics and OOD metrics are part of the experiment concept, but require dedicated evaluation support if not already available in the run artifacts.
