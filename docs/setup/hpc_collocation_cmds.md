# HPC Collocation Commands

## 1. Clone And Environment

```bash
git clone https://github.com/jnasw/ml-surrogates-for-power-systems.git
cd ml-surrogates-for-power-systems
git pull

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

Paste the API key.

## 3. Reference And Evaluation Datasets

```bash
(export REFERENCE_IDS=main_SM4_qbc_b512_ds01 \
  DRY_RUN=false \
  FORCE_REBUILD=false && \
  bsub -env "all" -J ref_main_sm4 \
    < hpc/reference_datasets/generate_reference_datasets.lsf.sh)
```

```bash
(export EVALUATION_IDS=id_SM4_lhs_b512_ds01,ood_SM4_wide_ic_b512_ds01 \
  DRY_RUN=false \
  FORCE_REBUILD=false && \
  bsub -env "all" -J eval_sm4_b512 \
    < hpc/evaluation_datasets/generate_evaluation_datasets.lsf.sh)
```

## 4. Primary Collocation Runs

Protocol:

```text
4850 Adam epochs + 150 LBFGS epochs = 5000 total
refresh_period_epochs=500
terminal_refresh=true
seeds=s01,s02,s03
```

### p4k uniform_lhs

```bash
(export MODE=final \
  REFERENCE_ID=main_SM4_qbc_b512_ds01 \
  ID_EVAL_ID=id_SM4_lhs_b512_ds01 \
  OOD_EVAL_ID=ood_SM4_wide_ic_b512_ds01 \
  STRATEGIES=uniform_lhs \
  DENSITIES=p4k \
  SEED_LABELS=s01,s02,s03 \
  EPOCHS=4850 \
  REFRESH_PERIOD_EPOCHS=500 \
  TERMINAL_OPTIMIZER=lbfgs \
  TERMINAL_EPOCHS=150 \
  TERMINAL_REFRESH=true \
  DEVICE=cuda \
  WANDB_PROJECT=thesis-collocation-experiment \
  DRY_RUN=false && \
  bsub -W 48:00 -env "all" -J colloc_p4k_uniform \
    < hpc/collocation_comparison/run_collocation_comparison.lsf.sh)
```

### p4k random_resampling

```bash
(export MODE=final \
  REFERENCE_ID=main_SM4_qbc_b512_ds01 \
  ID_EVAL_ID=id_SM4_lhs_b512_ds01 \
  OOD_EVAL_ID=ood_SM4_wide_ic_b512_ds01 \
  STRATEGIES=random_resampling \
  DENSITIES=p4k \
  SEED_LABELS=s01,s02,s03 \
  EPOCHS=4850 \
  REFRESH_PERIOD_EPOCHS=500 \
  TERMINAL_OPTIMIZER=lbfgs \
  TERMINAL_EPOCHS=150 \
  TERMINAL_REFRESH=true \
  DEVICE=cuda \
  WANDB_PROJECT=thesis-collocation-experiment \
  DRY_RUN=false && \
  bsub -W 48:00 -env "all" -J colloc_p4k_random \
    < hpc/collocation_comparison/run_collocation_comparison.lsf.sh)
```

### p4k rad

```bash
(export MODE=final \
  REFERENCE_ID=main_SM4_qbc_b512_ds01 \
  ID_EVAL_ID=id_SM4_lhs_b512_ds01 \
  OOD_EVAL_ID=ood_SM4_wide_ic_b512_ds01 \
  STRATEGIES=rad \
  DENSITIES=p4k \
  SEED_LABELS=s01,s02,s03 \
  EPOCHS=4850 \
  REFRESH_PERIOD_EPOCHS=500 \
  TERMINAL_OPTIMIZER=lbfgs \
  TERMINAL_EPOCHS=150 \
  TERMINAL_REFRESH=true \
  DEVICE=cuda \
  WANDB_PROJECT=thesis-collocation-experiment \
  DRY_RUN=false && \
  bsub -W 48:00 -env "all" -J colloc_p4k_rad \
    < hpc/collocation_comparison/run_collocation_comparison.lsf.sh)
```

### p32k uniform_lhs

```bash
(export MODE=final \
  REFERENCE_ID=main_SM4_qbc_b512_ds01 \
  ID_EVAL_ID=id_SM4_lhs_b512_ds01 \
  OOD_EVAL_ID=ood_SM4_wide_ic_b512_ds01 \
  STRATEGIES=uniform_lhs \
  DENSITIES=p32k \
  SEED_LABELS=s01,s02,s03 \
  EPOCHS=4850 \
  REFRESH_PERIOD_EPOCHS=500 \
  TERMINAL_OPTIMIZER=lbfgs \
  TERMINAL_EPOCHS=150 \
  TERMINAL_REFRESH=true \
  DEVICE=cuda \
  WANDB_PROJECT=thesis-collocation-experiment \
  DRY_RUN=false && \
  bsub -W 48:00 -env "all" -J colloc_p32k_uniform \
    < hpc/collocation_comparison/run_collocation_comparison.lsf.sh)
```

### p32k random_resampling

```bash
(export MODE=final \
  REFERENCE_ID=main_SM4_qbc_b512_ds01 \
  ID_EVAL_ID=id_SM4_lhs_b512_ds01 \
  OOD_EVAL_ID=ood_SM4_wide_ic_b512_ds01 \
  STRATEGIES=random_resampling \
  DENSITIES=p32k \
  SEED_LABELS=s01,s02,s03 \
  EPOCHS=4850 \
  REFRESH_PERIOD_EPOCHS=500 \
  TERMINAL_OPTIMIZER=lbfgs \
  TERMINAL_EPOCHS=150 \
  TERMINAL_REFRESH=true \
  DEVICE=cuda \
  WANDB_PROJECT=thesis-collocation-experiment \
  DRY_RUN=false && \
  bsub -W 48:00 -env "all" -J colloc_p32k_random \
    < hpc/collocation_comparison/run_collocation_comparison.lsf.sh)
```

### p32k rad

```bash
(export MODE=final \
  REFERENCE_ID=main_SM4_qbc_b512_ds01 \
  ID_EVAL_ID=id_SM4_lhs_b512_ds01 \
  OOD_EVAL_ID=ood_SM4_wide_ic_b512_ds01 \
  STRATEGIES=rad \
  DENSITIES=p32k \
  SEED_LABELS=s01,s02,s03 \
  EPOCHS=4850 \
  REFRESH_PERIOD_EPOCHS=500 \
  TERMINAL_OPTIMIZER=lbfgs \
  TERMINAL_EPOCHS=150 \
  TERMINAL_REFRESH=true \
  DEVICE=cuda \
  WANDB_PROJECT=thesis-collocation-experiment \
  DRY_RUN=false && \
  bsub -W 48:00 -env "all" -J colloc_p32k_rad \
    < hpc/collocation_comparison/run_collocation_comparison.lsf.sh)
```

## 5. Optional p64k Extension

```bash
(export MODE=final REFERENCE_ID=main_SM4_qbc_b512_ds01 \
  ID_EVAL_ID=id_SM4_lhs_b512_ds01 OOD_EVAL_ID=ood_SM4_wide_ic_b512_ds01 \
  STRATEGIES=uniform_lhs DENSITIES=p64k SEED_LABELS=s01,s02,s03 \
  EPOCHS=4850 REFRESH_PERIOD_EPOCHS=500 TERMINAL_OPTIMIZER=lbfgs \
  TERMINAL_EPOCHS=150 TERMINAL_REFRESH=true DEVICE=cuda \
  WANDB_PROJECT=thesis-collocation-experiment DRY_RUN=false && \
  bsub -W 48:00 -env "all" -J colloc_p64k_uniform \
    < hpc/collocation_comparison/run_collocation_comparison.lsf.sh)
```

```bash
(export MODE=final REFERENCE_ID=main_SM4_qbc_b512_ds01 \
  ID_EVAL_ID=id_SM4_lhs_b512_ds01 OOD_EVAL_ID=ood_SM4_wide_ic_b512_ds01 \
  STRATEGIES=random_resampling DENSITIES=p64k SEED_LABELS=s01,s02,s03 \
  EPOCHS=4850 REFRESH_PERIOD_EPOCHS=500 TERMINAL_OPTIMIZER=lbfgs \
  TERMINAL_EPOCHS=150 TERMINAL_REFRESH=true DEVICE=cuda \
  WANDB_PROJECT=thesis-collocation-experiment DRY_RUN=false && \
  bsub -W 48:00 -env "all" -J colloc_p64k_random \
    < hpc/collocation_comparison/run_collocation_comparison.lsf.sh)
```

```bash
(export MODE=final REFERENCE_ID=main_SM4_qbc_b512_ds01 \
  ID_EVAL_ID=id_SM4_lhs_b512_ds01 OOD_EVAL_ID=ood_SM4_wide_ic_b512_ds01 \
  STRATEGIES=rad DENSITIES=p64k SEED_LABELS=s01,s02,s03 \
  EPOCHS=4850 REFRESH_PERIOD_EPOCHS=500 TERMINAL_OPTIMIZER=lbfgs \
  TERMINAL_EPOCHS=150 TERMINAL_REFRESH=true DEVICE=cuda \
  WANDB_PROJECT=thesis-collocation-experiment DRY_RUN=false && \
  bsub -W 48:00 -env "all" -J colloc_p64k_rad \
    < hpc/collocation_comparison/run_collocation_comparison.lsf.sh)
```

## 6. Relevant Outputs

```text
outputs/pinn/collocation_comparison/<timestamp>/summary.csv
outputs/pinn/collocation_comparison/<timestamp>/summary.json
outputs/pinn/collocation_comparison/<timestamp>/run_manifest.json
outputs/pinn/collocation_comparison/<timestamp>/failures.json
outputs/pinn/collocation_comparison/<timestamp>/logs/
outputs/pinn/collocation_comparison/<timestamp>/runs/<strategy>_<budget>_c500_<seed>_adam_lbfgs150/
```
