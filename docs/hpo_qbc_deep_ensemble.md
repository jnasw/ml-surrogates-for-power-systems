# QBC Deep Ensemble HPO Pipeline

This is the current HPC-ready HPO workflow for `qbc_deep_ensemble`.

For exact submit commands and detailed output paths, see:
`docs/hpo_submit_and_results.md`.

## Files

- Matrix builder: `tools/hpo/build_hpo_matrix.py`
- Row runner: `tools/hpo/run_hpo_matrix_row.py`
- LSF job scripts (submit with `bsub < ...`):
  - `tools/hpo/jobs/qbc_smoke_stage0.lsf.sh`
  - `tools/hpo/jobs/qbc_policy_search_stage1.lsf.sh`
  - `tools/hpo/jobs/qbc_schedule_search_stage2.lsf.sh`
- LSF array submit: `tools/hpo/submit_hpo_lsf_array.sh`
- LSF array row runner: `tools/hpo/run_hpo_lsf_array_row.sh`
- LSF smoke submit: `tools/hpo/smoke_hpo_lsf.sh`
- QBC configs:
  - `src/config/hpo/qbc_deep_ensemble/smoke_stage0.yaml`
  - `src/config/hpo/qbc_deep_ensemble/policy_search_stage1.yaml`
  - `src/config/hpo/qbc_deep_ensemble/schedule_search_stage2.yaml`

## Folder Layout

Each matrix build writes to:

`outputs/hpo/qbc_deep_ensemble/<stage>/<hpo_id_timestamp>/`

Inside:

- `matrix.tsv`
- `matrix_meta.json`
- `runs/<cfg_id>/...` (per-row outputs)

Each row run writes:

- `hpo_row.json`
- `hpo_command.sh`
- `hpo_status.json`
- plus the full pipeline artifacts from `tools/pipeline/run_experiment.py`

## Local Sanity Checks

Build matrix only:

```bash
python3 tools/hpo/build_hpo_matrix.py \
  --config src/config/hpo/qbc_deep_ensemble/smoke_stage0.yaml \
  --no-timestamp
```

Dry-run one row:

```bash
python3 tools/hpo/run_hpo_matrix_row.py \
  --matrix outputs/hpo/qbc_deep_ensemble/stage0_smoke/qbc_deep_ensemble_smoke/matrix.tsv \
  --row-index 0 \
  --dry-run
```

## LSF Usage

Preferred: submit the job scripts directly.

Smoke (single row):

```bash
bsub < tools/hpo/jobs/qbc_smoke_stage0.lsf.sh
```

Stage-1 policy search:

```bash
bsub < tools/hpo/jobs/qbc_policy_search_stage1.lsf.sh
```

Stage-2 schedule search:

```bash
bsub < tools/hpo/jobs/qbc_schedule_search_stage2.lsf.sh
```

Optional resource overrides:

```bash
QUEUE=gpua100 N_CORES=1 MEM_GB=24 WALL_HOURS=08 \
bash tools/hpo/submit_hpo_lsf_array.sh <config.yaml>
```

## Notes

- `active.diversity.distance_weight` is auto-derived as `1 - active.diversity.uncertainty_weight`.
- `qbc_n0 + qbc_K * qbc_T == budget.final_n` is validated when enabled in config constraints.
- Scripts are generic and reusable for marker/hybrid by adding method-scoped configs under `src/config/hpo/<method>/`.
