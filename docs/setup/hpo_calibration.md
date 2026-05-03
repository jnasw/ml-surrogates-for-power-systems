# HPO Calibration

## Objective

Run small calibration sweeps before the final controlled thesis experiments.

This is not a broad HPO framework. The purpose is to select reasonable fixed defaults for the final experiment matrices while keeping the final comparisons controlled and reproducible.

## Calibration Order

Run calibration in this order:

1. `pinn_architecture`
2. `adam_lr`
3. `second_order_lr`
4. `baseline_architecture`

PINN architecture is selected before tuning Adam. Adam and second-order LR calibration should then use the selected PINN architecture. Baseline architecture is independent and is mainly relevant for the dataset-generation experiment when using the legacy trajectory baseline.

## Entry Points

Local entrypoint:

```bash
python3 -m src.experiments.pipeline.run_hpo_calibration
```

HPC wrapper:

```bash
hpc/hpo/run_hpo_calibration.lsf.sh
```

Default reference dataset:

```text
dev_SM4_lhs_b512_ds01
```

If the reference dataset is missing, generate it first:

```bash
python3 -m src.experiments.pipeline.run_reference_datasets \
  --reference-id dev_SM4_lhs_b512_ds01
```

## Studies

### `pinn_architecture`

Purpose: select one reasonable PINN architecture before LR tuning.

Default grid:

```text
hidden_dim: 32,64,128
hidden_layers: 3,4,5
activation: tanh
```

Default W&B project:

```text
thesis-hpo-pinn-architecture
```

### `adam_lr`

Purpose: calibrate Adam LR for the selected PINN architecture.

Default grid:

```text
lr: 1e-4,3e-4,1e-3,3e-3
```

Use `--pinn-hidden-dim` and `--pinn-hidden-layers` to test the selected architecture from the previous study.

Default W&B project:

```text
thesis-hpo-adam-lr
```

Current selected value used by the optimizer comparison launcher:

```text
Adam LR: 0.003
```

### `second_order_lr`

Purpose: calibrate LR values for curvature-aware optimizers enough to avoid broken defaults.

Default grid:

```text
optimizers: LBFGS,BFGS,SSBFGS,SSBroyden,SOAP
lr: 0.05,0.1,0.3,0.5,1.0
```

Unstable stochastic self-scaled methods are excluded from this default study.

Default W&B project:

```text
thesis-hpo-second-order-lr
```

Current selected values used by the optimizer comparison launcher:

| Optimizer | LR |
|-----------|----|
| SOAP | 0.05 |
| LBFGS | 0.1 |
| BFGS | 1.0 |
| SSBFGS | 0.3 |
| SSBroyden | 1.0 |

### `baseline_architecture`

Purpose: select the supervised baseline architecture for the dataset-generation experiment when using the legacy `trajectory_baseline` downstream model.

Default grid:

```text
hidden_dim: 64,128,256
hidden_layers: 3,4,5
```

Default W&B project:

```text
thesis-hpo-baseline-architecture
```

## Local Commands

PINN architecture:

```bash
python3 -m src.experiments.pipeline.run_hpo_calibration \
  --study pinn_architecture \
  --reference-id dev_SM4_lhs_b512_ds01 \
  --seed-labels s01 \
  --device cuda
```

Adam LR with selected architecture:

```bash
python3 -m src.experiments.pipeline.run_hpo_calibration \
  --study adam_lr \
  --reference-id dev_SM4_lhs_b512_ds01 \
  --seed-labels s01 \
  --device cuda \
  --pinn-hidden-dim 64 \
  --pinn-hidden-layers 4
```

Second-order LR with selected architecture:

```bash
python3 -m src.experiments.pipeline.run_hpo_calibration \
  --study second_order_lr \
  --reference-id dev_SM4_lhs_b512_ds01 \
  --seed-labels s01 \
  --device cuda \
  --pinn-hidden-dim 64 \
  --pinn-hidden-layers 4
```

Baseline architecture:

```bash
python3 -m src.experiments.pipeline.run_hpo_calibration \
  --study baseline_architecture \
  --reference-id dev_SM4_lhs_b512_ds01 \
  --seed-labels s01 \
  --device cuda
```

## HPC Commands

Run from the repository root. Use `bsub -env` to pass environment variables — inline `VAR=val bsub` does not propagate to the batch environment on this cluster. When a variable value contains commas (e.g. `OPTIMIZERS=LBFGS,SSBFGS`), use the subshell form `(export VAR=val ... && bsub -env "all" < script)` instead.

PINN architecture dry-run:

```bash
bsub -env "STUDY=pinn_architecture,DRY_RUN=true" \
  < hpc/hpo/run_hpo_calibration.lsf.sh
```

Adam LR:

```bash
bsub -env "STUDY=adam_lr,PINN_HIDDEN_DIM=64,PINN_HIDDEN_LAYERS=3" \
  < hpc/hpo/run_hpo_calibration.lsf.sh
```

Second-order LR subset (values contain commas — use subshell form):

```bash
(export STUDY=second_order_lr OPTIMIZERS=LBFGS,SSBFGS,SSBroyden,SOAP LRS=0.05,0.1,0.3,1.0 && \
  bsub -env "all" < hpc/hpo/run_hpo_calibration.lsf.sh)
```

Baseline architecture:

```bash
bsub -env "STUDY=baseline_architecture,DEVICE=cuda" \
  < hpc/hpo/run_hpo_calibration.lsf.sh
```

HPC wrappers source shared defaults from `hpc/common/lsf_defaults.sh`, activate `.venv` or `venv` if present, and write cluster logs under:

```text
hpc/logs/hpo/
```

The static `#BSUB` directives remain the submitted resources on clusters that do not expand shell variables in directives.

## Outputs

Outputs are written under:

```text
outputs/hpo/<study>/<timestamp>/
```

Key files:

```text
run_manifest.json
summary.csv
summary.json
failures.json
logs/runs/*.log
runs/<run_name>/metrics.json
runs/<run_name>/timings.json
```

Use `summary.csv` to select the calibrated value. After selecting a value, explicitly patch the relevant final experiment launcher or config so the final matrix is fixed and reproducible.

## Notes

- HPO runs are calibration runs, not final thesis evidence by themselves.
- Final comparisons should use fixed calibrated values, not sweeps.
- Keep calibration small: one seed is usually enough unless a result is ambiguous.
- If smoke/compressed datasets are used for dry-runs, disable external evaluation when time grids differ.
