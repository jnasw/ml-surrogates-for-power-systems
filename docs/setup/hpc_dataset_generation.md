# HPC Dataset Generation Commands

## 1. Clone And Environment

```bash
git clone https://github.com/jnasw/ml-surrogates-for-power-systems.git
cd ml-surrogates-for-power-systems

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install --extra-index-url https://download.pytorch.org/whl/cu124 \
  -r requirements-lock-a100.txt
```

## 2. Dependency Smoke Check

```bash
python3 - <<'PY'
import h5py
import hydra
import numpy
import omegaconf
import pytorch_optimizer
import scipy
import torch
import wandb

print("dependency imports ok")
print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
PY
```

## 3. W&B Login

```bash
wandb login
```

Paste the API key.

Quick check:

```bash
python3 - <<'PY'
import wandb

print("wandb import ok")
print("configured api key:", bool(wandb.api.api_key))
PY
```

## 4. Evaluation Datasets

```bash
(export EVALUATION_IDS=id_SM4_lhs_b4096_eval01,ood_SM4_wide_ic_b4096_eval01 \
  DRY_RUN=false \
  FORCE_REBUILD=false && \
  bsub -env "all" -J eval_data_b4096 \
    < hpc/evaluation_datasets/generate_evaluation_datasets.lsf.sh)
```

Expected outputs:

```text
data/evaluation/id/SM4/id_SM4_lhs_b4096_eval01/
data/evaluation/ood/SM4/ood_SM4_wide_ic_b4096_eval01/
data/evaluation/index.json
```

## 5. Dataset Generation

### b256 lhs_static

```bash
(export CAMPAIGN_TAG=sm4_b256 \
  SHARD_LABEL=lhs_static \
  MODE=final \
  MODEL_FLAG=SM4 \
  METHODS=lhs_static \
  BUDGETS=b256 \
  DATASET_SEEDS=ds01,ds02,ds03 \
  BASELINE_SEEDS=bs01,bs02 \
  BASELINE_EPOCHS=2000 \
  DEVICE=cuda \
  WANDB_USE=true \
  WANDB_PROJECT=thesis-dataset-generation \
  ID_EVAL_ID=id_SM4_lhs_b4096_eval01 \
  OOD_EVAL_ID=ood_SM4_wide_ic_b4096_eval01 \
  CLEANUP_DATA=false \
  DRY_RUN=false && \
  bsub -gpu "num=1:mode=exclusive_process" -W 48:00 -env "all" -J data_lhs_b256 \
    < hpc/dataset_generation_comparison/run_dataset_generation_comparison.lsf.sh)
```

### b256 qbc_deep_ensemble

```bash
(export CAMPAIGN_TAG=sm4_b256 \
  SHARD_LABEL=qbc_deep_ensemble \
  MODE=final \
  MODEL_FLAG=SM4 \
  METHODS=qbc_deep_ensemble \
  BUDGETS=b256 \
  DATASET_SEEDS=ds01,ds02,ds03 \
  BASELINE_SEEDS=bs01,bs02 \
  BASELINE_EPOCHS=2000 \
  DEVICE=cuda \
  WANDB_USE=true \
  WANDB_PROJECT=thesis-dataset-generation \
  ID_EVAL_ID=id_SM4_lhs_b4096_eval01 \
  OOD_EVAL_ID=ood_SM4_wide_ic_b4096_eval01 \
  QBC_SAVE_ROUND_ARRAYS=false \
  QBC_SAVE_DATASET_CHECKPOINTS=false \
  QBC_SAVE_ENSEMBLE_CHECKPOINTS=false \
  CLEANUP_DATA=false \
  DRY_RUN=false && \
  bsub -gpu "num=1:mode=exclusive_process" -W 48:00 -env "all" -J data_qbc_b256 \
    < hpc/dataset_generation_comparison/run_dataset_generation_comparison.lsf.sh)
```

### b256 marker_directed

```bash
(export CAMPAIGN_TAG=sm4_b256 \
  SHARD_LABEL=marker_directed \
  MODE=final \
  MODEL_FLAG=SM4 \
  METHODS=marker_directed \
  BUDGETS=b256 \
  DATASET_SEEDS=ds01,ds02,ds03 \
  BASELINE_SEEDS=bs01,bs02 \
  BASELINE_EPOCHS=2000 \
  DEVICE=cuda \
  WANDB_USE=true \
  WANDB_PROJECT=thesis-dataset-generation \
  ID_EVAL_ID=id_SM4_lhs_b4096_eval01 \
  OOD_EVAL_ID=ood_SM4_wide_ic_b4096_eval01 \
  QBC_SAVE_ROUND_ARRAYS=false \
  QBC_SAVE_DATASET_CHECKPOINTS=false \
  QBC_SAVE_ENSEMBLE_CHECKPOINTS=false \
  CLEANUP_DATA=false \
  DRY_RUN=false && \
  bsub -gpu "num=1:mode=exclusive_process" -W 48:00 -env "all" -J data_marker_b256 \
    < hpc/dataset_generation_comparison/run_dataset_generation_comparison.lsf.sh)
```

### b512 lhs_static

```bash
(export CAMPAIGN_TAG=sm4_b512 \
  SHARD_LABEL=lhs_static \
  MODE=final \
  MODEL_FLAG=SM4 \
  METHODS=lhs_static \
  BUDGETS=b512 \
  DATASET_SEEDS=ds01,ds02,ds03 \
  BASELINE_SEEDS=bs01,bs02 \
  BASELINE_EPOCHS=2000 \
  DEVICE=cuda \
  WANDB_USE=true \
  WANDB_PROJECT=thesis-dataset-generation \
  ID_EVAL_ID=id_SM4_lhs_b4096_eval01 \
  OOD_EVAL_ID=ood_SM4_wide_ic_b4096_eval01 \
  CLEANUP_DATA=false \
  DRY_RUN=false && \
  bsub -gpu "num=1:mode=exclusive_process" -W 48:00 -env "all" -J data_lhs_b512 \
    < hpc/dataset_generation_comparison/run_dataset_generation_comparison.lsf.sh)
```

### b512 qbc_deep_ensemble

```bash
(export CAMPAIGN_TAG=sm4_b512 \
  SHARD_LABEL=qbc_deep_ensemble \
  MODE=final \
  MODEL_FLAG=SM4 \
  METHODS=qbc_deep_ensemble \
  BUDGETS=b512 \
  DATASET_SEEDS=ds01,ds02,ds03 \
  BASELINE_SEEDS=bs01,bs02 \
  BASELINE_EPOCHS=2000 \
  DEVICE=cuda \
  WANDB_USE=true \
  WANDB_PROJECT=thesis-dataset-generation \
  ID_EVAL_ID=id_SM4_lhs_b4096_eval01 \
  OOD_EVAL_ID=ood_SM4_wide_ic_b4096_eval01 \
  QBC_SAVE_ROUND_ARRAYS=false \
  QBC_SAVE_DATASET_CHECKPOINTS=false \
  QBC_SAVE_ENSEMBLE_CHECKPOINTS=false \
  CLEANUP_DATA=false \
  DRY_RUN=false && \
  bsub -gpu "num=1:mode=exclusive_process" -W 48:00 -env "all" -J data_qbc_b512 \
    < hpc/dataset_generation_comparison/run_dataset_generation_comparison.lsf.sh)
```

### b512 marker_directed

```bash
(export CAMPAIGN_TAG=sm4_b512 \
  SHARD_LABEL=marker_directed \
  MODE=final \
  MODEL_FLAG=SM4 \
  METHODS=marker_directed \
  BUDGETS=b512 \
  DATASET_SEEDS=ds01,ds02,ds03 \
  BASELINE_SEEDS=bs01,bs02 \
  BASELINE_EPOCHS=2000 \
  DEVICE=cuda \
  WANDB_USE=true \
  WANDB_PROJECT=thesis-dataset-generation \
  ID_EVAL_ID=id_SM4_lhs_b4096_eval01 \
  OOD_EVAL_ID=ood_SM4_wide_ic_b4096_eval01 \
  QBC_SAVE_ROUND_ARRAYS=false \
  QBC_SAVE_DATASET_CHECKPOINTS=false \
  QBC_SAVE_ENSEMBLE_CHECKPOINTS=false \
  CLEANUP_DATA=false \
  DRY_RUN=false && \
  bsub -gpu "num=1:mode=exclusive_process" -W 48:00 -env "all" -J data_marker_b512 \
    < hpc/dataset_generation_comparison/run_dataset_generation_comparison.lsf.sh)
```

### b1024 lhs_static

```bash
(export CAMPAIGN_TAG=sm4_b1024 \
  SHARD_LABEL=lhs_static \
  MODE=final \
  MODEL_FLAG=SM4 \
  METHODS=lhs_static \
  BUDGETS=b1024 \
  DATASET_SEEDS=ds01,ds02,ds03 \
  BASELINE_SEEDS=bs01,bs02 \
  BASELINE_EPOCHS=2000 \
  DEVICE=cuda \
  WANDB_USE=true \
  WANDB_PROJECT=thesis-dataset-generation \
  ID_EVAL_ID=id_SM4_lhs_b4096_eval01 \
  OOD_EVAL_ID=ood_SM4_wide_ic_b4096_eval01 \
  CLEANUP_DATA=false \
  DRY_RUN=false && \
  bsub -gpu "num=1:mode=exclusive_process" -W 48:00 -env "all" -J data_lhs_b1024 \
    < hpc/dataset_generation_comparison/run_dataset_generation_comparison.lsf.sh)
```

### b1024 qbc_deep_ensemble

```bash
(export CAMPAIGN_TAG=sm4_b1024 \
  SHARD_LABEL=qbc_deep_ensemble \
  MODE=final \
  MODEL_FLAG=SM4 \
  METHODS=qbc_deep_ensemble \
  BUDGETS=b1024 \
  DATASET_SEEDS=ds01,ds02,ds03 \
  BASELINE_SEEDS=bs01,bs02 \
  BASELINE_EPOCHS=2000 \
  DEVICE=cuda \
  WANDB_USE=true \
  WANDB_PROJECT=thesis-dataset-generation \
  ID_EVAL_ID=id_SM4_lhs_b4096_eval01 \
  OOD_EVAL_ID=ood_SM4_wide_ic_b4096_eval01 \
  QBC_SAVE_ROUND_ARRAYS=false \
  QBC_SAVE_DATASET_CHECKPOINTS=false \
  QBC_SAVE_ENSEMBLE_CHECKPOINTS=false \
  CLEANUP_DATA=false \
  DRY_RUN=false && \
  bsub -gpu "num=1:mode=exclusive_process" -W 48:00 -env "all" -J data_qbc_b1024 \
    < hpc/dataset_generation_comparison/run_dataset_generation_comparison.lsf.sh)
```

### b1024 marker_directed

```bash
(export CAMPAIGN_TAG=sm4_b1024 \
  SHARD_LABEL=marker_directed \
  MODE=final \
  MODEL_FLAG=SM4 \
  METHODS=marker_directed \
  BUDGETS=b1024 \
  DATASET_SEEDS=ds01,ds02,ds03 \
  BASELINE_SEEDS=bs01,bs02 \
  BASELINE_EPOCHS=2000 \
  DEVICE=cuda \
  WANDB_USE=true \
  WANDB_PROJECT=thesis-dataset-generation \
  ID_EVAL_ID=id_SM4_lhs_b4096_eval01 \
  OOD_EVAL_ID=ood_SM4_wide_ic_b4096_eval01 \
  QBC_SAVE_ROUND_ARRAYS=false \
  QBC_SAVE_DATASET_CHECKPOINTS=false \
  QBC_SAVE_ENSEMBLE_CHECKPOINTS=false \
  CLEANUP_DATA=false \
  DRY_RUN=false && \
  bsub -gpu "num=1:mode=exclusive_process" -W 48:00 -env "all" -J data_marker_b1024 \
    < hpc/dataset_generation_comparison/run_dataset_generation_comparison.lsf.sh)
```

## 6. Relevant Outputs

```text
outputs/experiments/dataset_generation_comparison/sm4_<budget>/<method-shard>/summary.csv
outputs/experiments/dataset_generation_comparison/sm4_<budget>/<method-shard>/summary.json
outputs/experiments/dataset_generation_comparison/sm4_<budget>/<method-shard>/run_manifest.json
outputs/experiments/dataset_generation_comparison/sm4_<budget>/<method-shard>/failures.json
outputs/experiments/dataset_generation_comparison/sm4_<budget>/<method-shard>/logs/
outputs/experiments/dataset_generation_comparison/sm4_<budget>/<method-shard>/<method>/<budget>/<dataset_seed>/data/
```
