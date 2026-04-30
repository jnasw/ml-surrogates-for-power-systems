# Experiment 5: Multi-Stage Training

## Objective

Evaluate whether multi-stage training improves PINN performance, convergence behavior, and computational efficiency compared to a standard single-stage baseline. The goal is to assess whether sequential training stages provide meaningful improvements beyond well-tuned single-stage training.

## Research Questions

- Does multi-stage training improve final surrogate accuracy compared to single-stage training?
- How does approximation error evolve across stages?
- Do later stages provide meaningful residual correction, or do improvements saturate?
- Does multi-stage training improve accuracy in stiff regions?
- Does it enable better learning of multi-scale dynamics?
- Does switching optimizers (Adam → SSBroyden) improve convergence stability or speed?
- Does multi-stage training improve OOD performance?
- Is the additional computational cost justified?

## Hypotheses

- Multi-stage training improves convergence reliability compared to directly relying on second-order methods.
- Later stages reduce residual errors, but improvements diminish after a small number of stages.
- Gains mainly come from better optimization dynamics rather than fundamentally different model representations.
- The performance gain may not always justify the added runtime and implementation complexity.

## Experimental Setup

- PINN training using one fixed reference dataset.
- Fixed model architecture, activation, and initialization.
- Fixed collocation strategy and density.
- Fixed loss formulation and static loss weighting.
- Fixed evaluation setup: internal test split and external OOD evaluation by default.
- Training strategy is the only experimental variable.

## Training Strategies

All strategies share a fixed total budget of 30,000 epochs. Multi-stage strategies split this budget across stages. Each SSBroyden stage uses a decreasing learning rate, targeting finer residual correction in later stages.

| Strategy | Stages | Epoch split | Optimizers |
|----------|--------|-------------|------------|
| `adam_30000` | 1 | 30,000 | Adam |
| `adam_ssbroyden_2stage` | 2 | 15,000 + 15,000 | Adam → SSBroyden (lr=1.0) |
| `adam_ssbroyden_3stage` | 3 | 10,000 + 10,000 + 10,000 | Adam → SSBroyden (1.0 → 0.5) |
| `adam_ssbroyden_4stage` | 4 | 8,000 + 8,000 + 8,000 + 6,000 | Adam → SSBroyden (1.0 → 0.5 → 0.25) |
| `adam_ssbroyden_5stage` | 5 | 6,000 × 5 | Adam → SSBroyden (1.0 → 0.5 → 0.25 → 0.1) |

Screening mode scales all epoch budgets proportionally to 100 total epochs.

## Run Matrix

Each run is defined by:

```text
training_strategy × seed
```

Default final setup:

```text
5 strategies × 5 seeds (s01–s05) = 25 runs
```

## Metrics

### Performance
- Internal test RMSE / MSE / MAE
- OOD RMSE / MSE / MAE (enabled by default)
- ID RMSE / MSE / MAE (optional, disabled by default)
- ID-OOD RMSE gap

### Stage-wise behavior
- Stage-wise train loss
- Stage-wise test RMSE / MAE
- Residual RMS / max after each stage boundary
- Improvement delta between stages

### Convergence
- Loss vs epoch
- Loss vs walltime
- Stage-wise walltime

### Cost
- Total walltime
- Walltime per stage
- Total epochs
- Number of training steps

## Stage Artifacts

Each run produces two stage-specific JSON files:

- `stage_summary.json` — stage-wise train losses, test metrics, walltime, and improvement delta per completed stage.
- `stage_residual_probe.json` — residual diagnostics (RMS and max) computed at each stage boundary.

Both files index entries by `stage_idx`. Runs may complete fewer stages than planned if the early stopping criterion (`residual_rms_threshold=1e-9`) is met.

## How to Run

Run commands from the repository root.

HPC wrappers source shared defaults from `hpc/common/lsf_defaults.sh`; the static `#BSUB` directives remain the submitted resources. Use inline environment variables only. Do not use `bsub -env` or `export` blocks in runbook commands.

PINN wrappers expose `MODEL_FLAG`, `DTYPE`, `ID_EVAL_ID`, `OOD_EVAL_ID`, and `NO_OOD_EVAL`. OOD evaluation is enabled by default using the launcher default unless `NO_OOD_EVAL=true` / `--no-ood-eval` is used. ID evaluation is optional and must be requested explicitly.

### Screening (local)

```bash
python3 -m src.experiments.pipeline.run_multistage_comparison \
  --mode screening \
  --reference-id smoke_SM4_lhs_b256_ds01 \
  --strategies adam_30000,adam_ssbroyden_2stage \
  --seed-labels s01 \
  --device cpu \
  --no-ood-eval
```

### Final subset (local)

```bash
python3 -m src.experiments.pipeline.run_multistage_comparison \
  --mode final \
  --reference-id main_SM4_qbc_b512_ds01 \
  --strategies adam_30000,adam_ssbroyden_2stage \
  --seed-labels s01,s02
```

### Full final experiment (local)

```bash
python3 -m src.experiments.pipeline.run_multistage_comparison \
  --mode final \
  --reference-id main_SM4_qbc_b512_ds01 \
  --strategies adam_30000,adam_ssbroyden_2stage,adam_ssbroyden_3stage,adam_ssbroyden_4stage,adam_ssbroyden_5stage \
  --seed-labels s01,s02,s03,s04,s05
```

### Screening (HPC)

```bash
MODE=screening REFERENCE_ID=smoke_SM4_lhs_b256_ds01 \
  STRATEGIES=adam_30000,adam_ssbroyden_2stage SEED_LABELS=s01 DEVICE=cpu \
  bsub < hpc/multistage_comparison/run_multistage_comparison.lsf.sh
```

### Full final experiment (HPC)

```bash
MODE=final REFERENCE_ID=main_SM4_qbc_b512_ds01 \
  STRATEGIES=adam_30000,adam_ssbroyden_2stage,adam_ssbroyden_3stage,adam_ssbroyden_4stage,adam_ssbroyden_5stage \
  SEED_LABELS=s01,s02,s03,s04,s05 \
  bsub < hpc/multistage_comparison/run_multistage_comparison.lsf.sh
```

### Dry-run check (HPC)

```bash
MODE=screening REFERENCE_ID=smoke_SM4_lhs_b256_ds01 DEVICE=cpu DRY_RUN=true \
  bsub < hpc/multistage_comparison/run_multistage_comparison.lsf.sh
```

## Outputs

Experiment outputs are written under:

```text
outputs/pinn/multistage_comparison/<timestamp>/
```

Each comparison run produces:

- `run_manifest.json` — full structured manifest of all runs
- `summary.csv` — main analysis table, one row per `strategy × seed`
- `summary.json` — same data as `summary.csv` in JSON format, plus per-strategy aggregates
- `failures.json` — details of any failed runs
- per-run `metrics.json`
- per-run `timings.json`
- per-run `stage_summary.json`
- per-run `stage_residual_probe.json`

`summary.csv` columns include: `strategy`, `pinn_mode`, `planned_num_stages`, `completed_num_stages`, `seed_label`, internal test metrics, `id_eval` and `ood_eval` metrics, `id_ood_rmse_gap`, stage-wise RMSE / MAE / loss / walltime / improvement delta (for stages 0–4), residual RMS and max per stage boundary, and timing fields.

Cluster logs are written under:

```text
hpc/logs/multistage/
```

## Notes

- OOD evaluation is enabled by default for all modes. Use `--no-ood-eval` to disable it when the evaluation time grid is incompatible (e.g. smoke datasets).
- ID evaluation is optional and must be requested explicitly via `--id-eval-id` or `--id-eval-root`.
- Gradient telemetry is disabled by default. Enable with `--gradient-telemetry` if needed.
- Smoke/reference smoke datasets disable validation. Normal/final datasets keep validation by default.
- `completed_num_stages` may be less than `planned_num_stages` if the residual early-stopping threshold is reached.
- Stage-wise metrics are essential for interpreting whether later stages provide meaningful improvement over earlier ones.
- Default reference dataset for final runs: `main_SM4_qbc_b512_ds01`. Default OOD eval: `ood_SM4_wide_ic_b512_ds01`.

## Expected Outcome

This experiment determines whether multi-stage training provides a meaningful advantage over single-stage PINN training. It identifies whether sequential optimization improves convergence and final accuracy, whether residual correction saturates after a few stages, and whether the added computational cost is justified relative to a well-tuned Adam-only baseline.
