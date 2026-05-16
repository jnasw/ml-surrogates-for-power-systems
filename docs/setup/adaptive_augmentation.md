# Adaptive Augmentation Experiment

## Objective

Evaluate whether a PINN benefits from revealing more labelled supervised trajectories and adding more collocation points during training, instead of training from epoch 1 with one fixed visible budget.

This is a pool-based active reveal experiment:

- the full labelled supervised training pool already exists in the preprocessed dataset
- only an initial set of training trajectory IDs is active at the start
- acquisition events reveal additional whole trajectories from the hidden training pool
- collocation growth uses generated RAR-D append behaviour

The experiment does not generate new simulator trajectories during PINN training.

## Experiment Design

The launcher varies two axes:

```text
supervised strategy x collocation strategy x seed
```

Supervised strategies:

- `fixed_low`: train on the initial supervised trajectory budget only
- `random_growth`: reveal new supervised trajectories uniformly at random
- `mae_nearest_growth`: score active trajectories by current model MAE, then reveal hidden trajectories nearest to the hardest active trajectories in initial-condition space
- `fixed_full`: train with the full supervised training pool visible from the start

Collocation strategies:

- `static_low`: use a fixed low preprocessed collocation budget
- `rar_d_growth`: start with a smaller generated collocation set and append residual-weighted RAR-D points over time

The intended main comparison uses equal final budgets for the growth variants. This lets the experiment separate the effect of *when and where* data is added from the effect of simply using more final data.

## Budget Semantics

Supervised budget is counted in whole training trajectories.

```text
--initial-trajectories
--add-trajectories
--final-trajectories
```

These are active **train-split** trajectories, not total generated trajectories
before train/validation/test splitting. With the standard `0.8` split ratio, a
`b4096` total reference dataset gives approximately:

```text
total generated trajectories: 4096
train acquisition pool:       3276 trajectories
validation split:              410 trajectories
test split:                    410 trajectories
```

The agreed main supervised schedule is:

```text
initial active train trajectories: 256
add per refresh:                   32
final active train trajectories:   512
candidate pool at start:          3020
candidate pool at final budget:   2764
```

Collocation budget is counted in active collocation rows.

```text
--initial-collocation-points
--add-collocation-points
--final-collocation-points
--candidate-collocation-points
```

`--refresh-period-epochs` controls both supervised acquisition and RAR-D collocation append cadence. As in the collocation experiment, the first refresh happens after the cadence boundary. For example, cadence `500` first refreshes at epoch `501`.

## Constraints

The current implementation intentionally keeps the first experiment narrow:

- single-stage PINN only
- static loss weighting only
- curriculum disabled
- labelled candidate trajectories must be in the training split
- supervised acquisition uses training trajectory IDs only
- newly preprocessed supervised files must contain `trajectory_id_*` metadata

These constraints avoid changing baseline, multistage, weighting, and curriculum behaviour while the adaptive augmentation path is evaluated.

## Canonical Launcher

Run from the repository root:

```bash
python3 -m src.experiments.pipeline.run_adaptive_augmentation
```

Default outputs are written to:

```text
outputs/pinn/adaptive_augmentation/<timestamp>/
```

Each run writes normal PINN artifacts under:

```text
outputs/pinn/adaptive_augmentation/<timestamp>/runs/<run_name>/
```

The launcher writes:

- `run_manifest.json`
- `summary.csv`
- `summary.json`
- `failures.json`
- per-run logs under `logs/runs/`

## Local Smoke / Dry Run

Use dry-run first to check the run matrix and generated `20_run_pinn.py` overrides:

```bash
python3 -m src.experiments.pipeline.run_adaptive_augmentation \
  --mode screening \
  --reference-id main_SM4_lhs_b4096_ds01 \
  --supervised-strategies fixed_low,mae_nearest_growth \
  --collocation-strategies static_low,rar_d_growth \
  --seed-labels s01 \
  --epochs 100 \
  --dry-run
```

For a CPU smoke run, use a small dataset/reference and small budgets:

```bash
python3 -m src.experiments.pipeline.run_adaptive_augmentation \
  --mode screening \
  --reference-id smoke_SM4_lhs_b256_ds01 \
  --supervised-strategies fixed_low,random_growth,mae_nearest_growth \
  --collocation-strategies static_low,rar_d_growth \
  --initial-trajectories 8 \
  --add-trajectories 4 \
  --final-trajectories 16 \
  --initial-collocation-points 512 \
  --add-collocation-points 256 \
  --final-collocation-points 2048 \
  --refresh-period-epochs 10 \
  --seed-labels s01 \
  --epochs 50 \
  --device cpu
```

## Main Run Shape

A thesis-scale final run should use the main reference dataset, fixed evaluation datasets, and several seeds:

```bash
python3 -m src.experiments.pipeline.run_adaptive_augmentation \
  --mode final \
  --reference-id main_SM4_lhs_b4096_ds01 \
  --supervised-strategies fixed_low,random_growth,mae_nearest_growth,fixed_full \
  --collocation-strategies static_low,rar_d_growth \
  --initial-trajectories 256 \
  --add-trajectories 32 \
  --final-trajectories 512 \
  --initial-collocation-points 4096 \
  --add-collocation-points 2048 \
  --final-collocation-points 32768 \
  --refresh-period-epochs 500 \
  --seed-labels s01,s02,s03,s04,s05
```

This run matrix is:

```text
4 supervised strategies x 2 collocation strategies x 5 seeds = 40 runs
```

With cadence `500`, the supervised budget reaches 512 active train trajectories
after eight acquisition events:

```text
epochs: 501, 1001, 1501, 2001, 2501, 3001, 3501, 4001
```

The RAR-D collocation budget reaches 32,768 active points after fourteen
refreshes, at epoch `7001`.

## Training Metrics

Each run writes per-epoch telemetry to `epoch_metrics.csv`. On the configured
PINN evaluation cadence, the file includes supervised regression metrics:

- `eval_active_train_mae`, `eval_active_train_rmse`
- `eval_full_train_mae`, `eval_full_train_rmse`
- `eval_val_mae`, `eval_val_rmse`
- `eval_test_mae`, `eval_test_rmse`

Use these curves alongside `train_total_loss` and acquisition telemetry. In
growth runs, losses can increase when harder supervised trajectories or
collocation points are appended, while fixed-split MAE/RMSE can still improve.

## HPC

Submit the wrapper with `bsub -env`. Use the subshell export form when values contain commas:

```bash
(export MODE=screening REFERENCE_ID=main_SM4_lhs_b4096_ds01 \
  SUPERVISED_STRATEGIES=fixed_low,mae_nearest_growth \
  COLLOCATION_STRATEGIES=static_low,rar_d_growth \
  SEED_LABELS=s01 DRY_RUN=true && \
  bsub -env "all" < hpc/adaptive_augmentation/run_adaptive_augmentation.lsf.sh)
```

Final example:

```bash
(export MODE=final REFERENCE_ID=main_SM4_lhs_b4096_ds01 \
  SUPERVISED_STRATEGIES=fixed_low,random_growth,mae_nearest_growth,fixed_full \
  COLLOCATION_STRATEGIES=static_low,rar_d_growth \
  SEED_LABELS=s01,s02,s03,s04,s05 \
  INITIAL_TRAJECTORIES=256 ADD_TRAJECTORIES=32 FINAL_TRAJECTORIES=512 \
  REFRESH_PERIOD_EPOCHS=500 && \
  bsub -env "all" < hpc/adaptive_augmentation/run_adaptive_augmentation.lsf.sh)
```

## Interpretation

Key comparisons:

- `fixed_low/static_low` measures the low-budget baseline
- `random_growth/static_low` isolates supervised growth without residual collocation growth
- `mae_nearest_growth/static_low` tests whether local refinement around high-error active trajectories helps beyond random growth
- `fixed_low/rar_d_growth` isolates collocation growth
- `mae_nearest_growth/rar_d_growth` tests the combined adaptive strategy
- `fixed_full/static_low` or `fixed_full/rar_d_growth` gives a high-supervised-budget reference

The most important fairness rule is that growth methods should share the same final supervised and collocation budgets unless the budget itself is being studied.
