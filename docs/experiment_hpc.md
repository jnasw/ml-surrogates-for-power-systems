# HPC Experiment Layout (End-to-End Pipeline)

This setup runs full experiments (`create_dataset.py` -> `preprocess_dataset.py` -> `run_baseline.py`)
via a single entrypoint: `run_experiment.py`.

## Config Tree

```text
src/conf/exp/
  base.yaml
  method/
    lhs_static.yaml
    qbc.yaml
    marker_directed.yaml
    qbc_marker_hybrid.yaml
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
- `marker_directed_b4096_s01`
- `qbc_marker_hybrid_b4096_s01`

Per-run root:

```text
outputs/experiments/{experiment.id}/{experiment.phase}/{experiment.method}/{experiment.budget}/{experiment.seed}/
```

Contents:
- `data/<MODEL>/dataset_v1/...` (raw trajectories + `info.txt`)
- `qbc/` (for adaptive runs, round checkpoints/history)
- `baseline/metrics.json`
- `run_manifest.json`

Hydra stage logs are separated at:

```text
.../<run-root>/hydra/stage*
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

- Main comparison (30 jobs):
  - `scripts/hpc/slurm_array_dataset_main.sh`
  - Matrix: `2 methods x 3 budgets x 5 seeds`
- QBC sensitivity (18 jobs):
  - `scripts/hpc/slurm_array_qbc_sensitivity.sh`
  - Matrix: `M in {3,5,8} x P in {256,1024} x 3 seeds`, fixed budget `b1024`
- Scaling (12 jobs):
  - `scripts/hpc/slurm_array_scaling.sh`
  - Matrix: `2 methods x 2 budgets x 3 seeds`

## Launch Commands

```bash
mkdir -p outputs/slurm_logs
sbatch scripts/hpc/slurm_array_dataset_main.sh
sbatch scripts/hpc/slurm_array_qbc_sensitivity.sh
sbatch scripts/hpc/slurm_array_scaling.sh
```

## Phase-2 From Phase-1 (Scale + Geometry)

Use this when you already finished local Phase-1 sensitivity and want to run Phase-2 on HPC directly from:

- `selected_top_configs.txt`
- `shared_test_source/run_manifest.json`

### Scripts

- Submit/build matrix:
  - `scripts/hpc/submit_qbc_phase2_from_phase1.sh`
- Array runner:
  - `scripts/hpc/slurm_array_qbc_phase2.sh`

### Required

```bash
PHASE1_RUN_BASE=outputs/qbc_sens/<phase1_run_name> \
bash scripts/hpc/submit_qbc_phase2_from_phase1.sh
```

### Common customizations

```bash
PHASE1_RUN_BASE=outputs/qbc_sens/thesis_sm4_qbc_sens_local_evalrmse_20260225_093141 \
PHASE2_EXPERIMENT_ID=thesis_sm4_qbc_phase2 \
SCALE_BUDGETS=b4096,b8192,b16384 \
SEEDS=s01,s02,s03,s04,s05,s06,s07,s08,s09,s10 \
GEOMETRY_CFGS=cfg04_l2 \
GEOMETRY_VARIANTS=geo_base,geo_fine,geo_coarse,geo_low_n0,geo_high_n0,geo_high_p \
GEOMETRY_BUDGETS=b4096 \
QBC_N_TEST=256 \
SURROGATE_DEVICE=cuda \
bash scripts/hpc/submit_qbc_phase2_from_phase1.sh
```

### What it runs

- **Phase 2A scale transfer:** selected Phase-1 policy configs x `SCALE_BUDGETS` x `SEEDS`
- **Phase 2B geometry sensitivity:** `GEOMETRY_CFGS` x `GEOMETRY_VARIANTS` x `GEOMETRY_BUDGETS` x `SEEDS`

The submit script writes:

- phase-2 run root: `outputs/qbc_phase2/<experiment_id>_<timestamp>/`
- matrix TSV: `phase2_matrix.tsv`
- metadata: `phase2_meta.json`

### LSF / bsub submission (if your cluster is not Slurm)

If your HPC requires `bsub`, use:

- `scripts/hpc/submit_qbc_phase2_from_phase1_lsf.sh`
- `scripts/hpc/lsf_array_qbc_phase2.sh`

Example:

```bash
bsub < scripts/hpc/submit_qbc_phase2_from_phase1_lsf.sh
```

The wrapper first builds `phase2_matrix.tsv` from Phase-1 artifacts, then submits one LSF array job that executes one matrix row per task.

## Single-Run Example (no Slurm)

```bash
python run_experiment.py \
  --method qbc_deep_ensemble \
  --budget b1024 \
  --seed s01 \
  --phase main \
  --experiment-id thesis_sm4_v1
```

## Notes

- Keep `--experiment-id` fixed for a campaign (e.g. `thesis_sm4_v1`).
- Change `#SBATCH` resources to match your cluster partitions and walltime constraints.
- Optional factor D can later be added as a separate config group (e.g. `src/conf/exp/scenario/...`) without changing the naming scheme.
