# HPC Experiment Layout (Stage-1 Dataset Generation)

This setup is for **LHS vs QBC** now, with naming/structure ready for later methods.

## Config Tree

```text
src/conf/exp/
  base.yaml
  method/
    lhs_static.yaml
    qbc.yaml
  phase/
    main.yaml
    qbc_sensitivity.yaml
    scaling.yaml
  budget/
    lhs/
      b256.yaml
      b1024.yaml
      b4096.yaml
      b8192.yaml
      b16384.yaml
    qbc/
      b256.yaml
      b1024.yaml
      b4096.yaml
      b8192.yaml
      b16384.yaml
  seed/
    s01.yaml
    s02.yaml
    s03.yaml
    s04.yaml
    s05.yaml
```

## Naming Convention

Run ID:

```text
{method}_{budget}_{seed}
```

Examples:
- `lhs_static_b1024_s03`
- `qbc_deep_ensemble_b4096_s01`

Per-run root:

```text
outputs/experiments/{experiment.id}/{experiment.phase}/{experiment.method}/{experiment.budget}/{experiment.seed}/
```

Contents:
- `data/<MODEL>/dataset_v1/...` (raw trajectories + `info.txt`)
- `qbc/` (for adaptive runs, round checkpoints/history)

Hydra logs are separated at:

```text
outputs/experiments/hydra/{RUN_ID}
```

## Budget Definitions

LHS budgets use `model.ic_num_samples = N`.

QBC budgets use `N = qbc_n0 + qbc_T * qbc_K`:
- `b256`: `32 + 7*32`
- `b1024`: `64 + 15*64`
- `b4096`: `128 + 31*128`
- `b8192`: `256 + 31*256`
- `b16384`: `512 + 31*512`

## Slurm Array Scripts

- Main comparison split by resource profile:
  - QBC GPU jobs (15): `scripts/hpc/slurm_array_dataset_main.sh`
  - LHS CPU jobs (15): `scripts/hpc/slurm_array_dataset_main_lhs_cpu.sh`
- QBC sensitivity (18 jobs):
  - `scripts/hpc/slurm_array_qbc_sensitivity.sh`
  - Matrix: `M in {3,5,8} x P in {256,1024} x 3 seeds`, fixed budget `b1024`
- Scaling split by resource profile:
  - QBC GPU jobs (6): `scripts/hpc/slurm_array_scaling.sh`
  - LHS CPU jobs (6): `scripts/hpc/slurm_array_scaling_lhs_cpu.sh`

## Launch Commands

```bash
mkdir -p outputs/slurm_logs
sbatch scripts/hpc/slurm_array_dataset_main.sh
sbatch scripts/hpc/slurm_array_dataset_main_lhs_cpu.sh
sbatch scripts/hpc/slurm_array_qbc_sensitivity.sh
sbatch scripts/hpc/slurm_array_scaling.sh
sbatch scripts/hpc/slurm_array_scaling_lhs_cpu.sh
```

## Single-Run Example (no Slurm)

```bash
python create_dataset.py \
  +exp=base \
  +exp/phase=main \
  +exp/method=qbc \
  +exp/budget/qbc=b1024 \
  +exp/seed=s01 \
  hydra.run.dir=${PWD}/outputs/experiments/hydra/qbc_deep_ensemble_b1024_s01 \
  hydra.job.chdir=false
```

## Notes

- Keep `experiment.id` in `src/conf/exp/base.yaml` fixed for a campaign (e.g. `thesis_sm4_v1`).
- Change `#SBATCH` resources to match your cluster partitions and walltime constraints.
- Optional factor D can later be added as a separate config group (e.g. `src/conf/exp/scenario/...`) without changing the naming scheme.
