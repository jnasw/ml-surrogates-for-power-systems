# HPC Commands

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

## 2. W&B Login


```bash
wandb login
```

Paste the API-key

Quick check:

```bash
python3 - <<'PY'
import os
import wandb

print("wandb import ok")
print("WANDB_API_KEY set:", bool(os.environ.get("WANDB_API_KEY")))
print("configured api key:", bool(wandb.api.api_key))
PY
```


## 3. Dependency Smoke Check


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

python3 -m src.experiments.pipeline.run_weighting_comparison \
  --mode screening \
  --allow-dataset-generation \
  --strategies static_tuned \
  --seed-labels s01 \
  --epochs 1 \
  --device cpu \
  --no-ood-eval \
  --dry-run \
  --output-root outputs/smoke/dependency_check
```

## 4. Reference Dataset

```bash
(export REFERENCE_IDS=main_SM4_qbc_b512_ds01 \
  DRY_RUN=false \
  FORCE_REBUILD=false && \
  bsub -env "all" < hpc/reference_datasets/generate_reference_datasets.lsf.sh)
```

Expected outputs:

```text
data/reference/main/SM4/qbc_deep_ensemble/b512/ds01/
data/reference/index.json
```

## 5. Evaluation Datasets

```bash
(export EVALUATION_IDS=id_SM4_lhs_b512_ds01,ood_SM4_wide_ic_b512_ds01 \
  DRY_RUN=false \
  FORCE_REBUILD=false && \
  bsub -env "all" < hpc/evaluation_datasets/generate_evaluation_datasets.lsf.sh)
```

Expected outputs:

```text
data/evaluation/id/SM4/id_SM4_lhs_b512_ds01/
data/evaluation/ood/SM4/ood_SM4_wide_ic_b512_ds01/
data/evaluation/index.json
```

## 6. Weighting Comparison

### s02 Static

```bash
(export MODE=final \
  REFERENCE_ID=main_SM4_qbc_b512_ds01 \
  ID_EVAL_ID=id_SM4_lhs_b512_ds01 \
  OOD_EVAL_ID=ood_SM4_wide_ic_b512_ds01 \
  STRATEGIES=data_only,static_tuned,static_uniform,data_warmup_static \
  SEED_LABELS=s02 \
  EPOCHS=10000 \
  DEVICE=cuda \
  GRADIENT_TELEMETRY=false \
  DRY_RUN=false && \
  bsub -W 24:00 -env "all" -J weight_s02_static \
    < hpc/weighting_comparison/run_weighting_comparison.lsf.sh)
```

### s02 Dynamic

```bash
(export MODE=final \
  REFERENCE_ID=main_SM4_qbc_b512_ds01 \
  ID_EVAL_ID=id_SM4_lhs_b512_ds01 \
  OOD_EVAL_ID=ood_SM4_wide_ic_b512_ds01 \
  STRATEGIES=ma,id,dn,ntk \
  SEED_LABELS=s02 \
  EPOCHS=10000 \
  DEVICE=cuda \
  GRADIENT_TELEMETRY=false \
  DRY_RUN=false && \
  bsub -W 48:00 -env "all" -J weight_s02_dyn \
    < hpc/weighting_comparison/run_weighting_comparison.lsf.sh)
```

### s03 Static

```bash
(export MODE=final \
  REFERENCE_ID=main_SM4_qbc_b512_ds01 \
  ID_EVAL_ID=id_SM4_lhs_b512_ds01 \
  OOD_EVAL_ID=ood_SM4_wide_ic_b512_ds01 \
  STRATEGIES=data_only,static_tuned,static_uniform,data_warmup_static \
  SEED_LABELS=s03 \
  EPOCHS=10000 \
  DEVICE=cuda \
  GRADIENT_TELEMETRY=false \
  DRY_RUN=false && \
  bsub -W 24:00 -env "all" -J weight_s03_static \
    < hpc/weighting_comparison/run_weighting_comparison.lsf.sh)
```

### s03 Dynamic

```bash
(export MODE=final \
  REFERENCE_ID=main_SM4_qbc_b512_ds01 \
  ID_EVAL_ID=id_SM4_lhs_b512_ds01 \
  OOD_EVAL_ID=ood_SM4_wide_ic_b512_ds01 \
  STRATEGIES=ma,id,dn,ntk \
  SEED_LABELS=s03 \
  EPOCHS=10000 \
  DEVICE=cuda \
  GRADIENT_TELEMETRY=false \
  DRY_RUN=false && \
  bsub -W 48:00 -env "all" -J weight_s03_dyn \
    < hpc/weighting_comparison/run_weighting_comparison.lsf.sh)
```

## 7. Optimizer Comparison

Target seeds for this handoff: `s02` and `s03`.

This runs all single-optimizer methods plus the two selected warm-start/two-phase methods:

- single optimizer: `adam`, `soap`, `bfgs`, `lbfgs`, `ssbfgs`, `ssbroyden`
- two phase: `adam_lbfgs`, `adam_ssbroyden`

### s02

```bash
(export MODE=final \
  REFERENCE_ID=main_SM4_qbc_b512_ds01 \
  ID_EVAL_ID=id_SM4_lhs_b512_ds01 \
  OOD_EVAL_ID=ood_SM4_wide_ic_b512_ds01 \
  STRATEGIES=soap,bfgs,lbfgs,ssbfgs,ssbroyden,adam_lbfgs,adam_ssbroyden \
  SEED_LABELS=s02 \
  DEVICE=cuda \
  DRY_RUN=false && \
  bsub -W 48:00 -env "all" -J opt_s02 \
    < hpc/optimizer_comparison/run_optimizer_comparison.lsf.sh)
```

### s03

```bash
(export MODE=final \
  REFERENCE_ID=main_SM4_qbc_b512_ds01 \
  ID_EVAL_ID=id_SM4_lhs_b512_ds01 \
  OOD_EVAL_ID=ood_SM4_wide_ic_b512_ds01 \
  STRATEGIES=soap,bfgs,lbfgs,ssbfgs,ssbroyden,adam_lbfgs,adam_ssbroyden \
  SEED_LABELS=s03 \
  DEVICE=cuda \
  DRY_RUN=false && \
  bsub -W 48:00 -env "all" -J opt_s03 \
    < hpc/optimizer_comparison/run_optimizer_comparison.lsf.sh)
```

## Relevant Outputs

```text
outputs/pinn/weighting_comparison/<timestamp>/summary.csv
outputs/pinn/weighting_comparison/<timestamp>/summary.json
outputs/pinn/weighting_comparison/<timestamp>/run_manifest.json
outputs/pinn/weighting_comparison/<timestamp>/logs/

outputs/pinn/optimizer_comparison/<timestamp>/summary.csv
outputs/pinn/optimizer_comparison/<timestamp>/summary.json
outputs/pinn/optimizer_comparison/<timestamp>/run_manifest.json
outputs/pinn/optimizer_comparison/<timestamp>/logs/
```
