# Canonical Regression Workflow

This is the lightweight regression workflow to use after refactor or trainer/runtime changes.

It intentionally covers one trusted baseline path and one trusted PINN path:
- small enough to run locally
- stable enough to catch real breakage
- close enough to normal execution paths to protect future refactors

## Baseline Regression Target

Purpose:
- verify the dataset -> preprocess -> baseline pipeline end to end
- verify canonical dataset and baseline artifacts

Command:

```bash
cd /Users/jonaswiendl/local/Repos/ml-surrogates-for-power-systems

RUN_ROOT="$PWD/outputs/experiments/regression_baseline_smoke/smoke/lhs_static/b256/ds01"

python3 -m src.experiments.pipeline.run_experiment \
  --method lhs_static \
  --budget b256 \
  --dataset-seed ds01 \
  --baseline-seed bs01 \
  --preset smoke \
  --experiment-id regression_baseline_smoke \
  --run-root "$RUN_ROOT" \
  --baseline-epochs 1 \
  --stage1-override model.ic_num_samples=32 \
  --stage2-override dataset.validation_flag=false \
  --stage2-override time=0.05 \
  --stage2-override num_of_points=20 \
  --stage3-override baseline.device=cpu
```

Expected output locations:
- dataset manifest:
  - `outputs/experiments/regression_baseline_smoke/smoke/lhs_static/b256/ds01/dataset_manifest.json`
- baseline run:
  - `outputs/experiments/regression_baseline_smoke/smoke/lhs_static/b256/ds01/baseline/bs01/`
- baseline summary:
  - `outputs/experiments/regression_baseline_smoke/smoke/lhs_static/b256/ds01/baseline/summary.json`

Required artifacts:
- `dataset_manifest.json`
- `baseline/bs01/run_manifest.json`
- `baseline/bs01/metrics.json`
- `baseline/bs01/timings.json`

Required metrics fields in `baseline/bs01/metrics.json`:
- `schema_version`
- `run_summary`
- `final_train_metrics`
- `final_test_metrics`
- `final_train_metrics.mse`
- `final_train_metrics.rmse`
- `final_train_metrics.mae`
- `final_test_metrics.mse`
- `final_test_metrics.rmse`
- `final_test_metrics.mae`

Simple sanity checks:
- `dataset_manifest.json` shows completed stage 1 and stage 2
- `baseline/bs01/run_manifest.json` status is `completed`
- train/test metrics are finite numbers
- timings are present and positive
- no missing canonical JSON artifacts

Likely regression signals:
- stage 1/2/3 fails
- `dataset_manifest.json` missing expected artifacts or completed stages
- `metrics.json` missing `final_*_metrics`
- NaN/inf values in metrics
- unexpected artifact layout changes

## PINN Regression Target

Purpose:
- verify the intended normal direct PINN runtime path against a tiny preprocessed dataset
- verify canonical direct PINN artifacts and evaluation fields

Command:

```bash
cd /Users/jonaswiendl/local/Repos/ml-surrogates-for-power-systems

DATASET_ROOT="$PWD/outputs/experiments/dev_smoke_v1/smoke/lhs_static/b256/ds01/data/SM4/dataset_v4"
PINN_RUN_DIR="$PWD/outputs/pinn/regression_pinn_smoke_cpu"

python3 20_run_pinn.py \
  model.model_flag=SM4 \
  model.seed=37 \
  dataset.root="$DATASET_ROOT" \
  pinn.run_dir="$PINN_RUN_DIR" \
  pinn.device=cpu \
  pinn.dtype=float64 \
  pinn.default_batch_size=128 \
  wandb.use=false \
  logging.log_every_epoch=1 \
  pinn.collocation.mode=preprocessed \
  pinn.collocation.strategy=static \
  pinn.collocation.active_points=128 \
  'pinn.optimizer_phases=[{name:adam,optimizer:Adam,lr:0.001,epochs:2,batch_size:128,shuffle:true,full_batch:false,allow_sampling:false,optimizer_kwargs:{},scheduler:null,line_search:null,convergence:null}]'
```

Expected output locations:
- `outputs/pinn/regression_pinn_smoke_cpu/run_manifest.json`
- `outputs/pinn/regression_pinn_smoke_cpu/metrics.json`
- `outputs/pinn/regression_pinn_smoke_cpu/timings.json`

Required artifacts:
- `run_manifest.json`
- `metrics.json`
- `timings.json`

Required metrics fields in `metrics.json`:
- `schema_version`
- `run_summary`
- `final_train_metrics`
- `final_test_metrics`
- `final_train_losses`
- `epochs_recorded`
- `final_epoch`

Recommended additional fields to inspect when present:
- `final_val_metrics`
- `final_val_losses`

Simple sanity checks:
- `run_manifest.json` status is `completed`
- `run_summary.run_type` is `pinn`
- `final_train_metrics` and `final_test_metrics` contain finite `mse`, `rmse`, `mae`
- `final_train_losses.total_loss` is present
- `final_test_losses` is `null`
- `final_epoch.phase_name` matches the short Adam phase
- no checkpoint directory exists unless checkpointing was explicitly enabled

Likely regression signals:
- trainer exits before writing canonical JSON artifacts
- `metrics.json` missing `final_epoch` or canonical `final_*` fields
- `final_test_losses` becomes non-null again without a real test-loss summary
- NaN/inf losses or supervised metrics
- output layout drifts away from `config.yaml` / `run_manifest.json` / `metrics.json` / `timings.json`

## Notes

Dataset-root fragility:
- the PINN target uses the current smoke dataset path:
  - `outputs/experiments/dev_smoke_v1/smoke/lhs_static/b256/ds01/data/SM4/dataset_v4`
- if the smoke dataset is regenerated and the version changes, read:
  - `outputs/experiments/dev_smoke_v1/smoke/lhs_static/b256/ds01/dataset_manifest.json`
  - then use `artifacts.preprocessed_root` as the source of truth

What this workflow validates:
- canonical direct baseline and direct PINN run artifacts
- core dataset/preprocess/baseline pipeline behavior
- core direct PINN training/evaluation artifact behavior

What this workflow does not validate:
- HPC wrappers
- large campaign runs
- expensive multistage runs
- optimizer/weighting/collocation benchmark matrices
- full numerical equivalence beyond smoke-scale sanity
