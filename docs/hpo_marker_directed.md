# Marker-Directed HPO Pipeline

This is the HPC-ready HPO workflow for `marker_directed`.

For shared result-path conventions, see:
`docs/hpo_submit_and_results.md`.

## Files

- Matrix builder: `tools/hpo/build_hpo_matrix.py`
- Row runner: `tools/hpo/run_hpo_matrix_row.py`
- LSF job scripts (submit with `bsub < ...`):
  - `tools/hpo/jobs/marker_smoke_stage0.lsf.sh`
  - `tools/hpo/jobs/marker_policy_search_stage1.lsf.sh`
  - `tools/hpo/jobs/marker_schedule_search_stage2.lsf.sh`
- Marker HPO configs:
  - `src/config/hpo/marker_directed/smoke_stage0.yaml`
  - `src/config/hpo/marker_directed/policy_search_stage1.yaml`
  - `src/config/hpo/marker_directed/schedule_search_stage2.yaml`

## LSF Usage

Smoke:

```bash
bsub < tools/hpo/jobs/marker_smoke_stage0.lsf.sh
```

Stage-1 policy search:

```bash
bsub < tools/hpo/jobs/marker_policy_search_stage1.lsf.sh
```

Stage-2 schedule search:

```bash
bsub < tools/hpo/jobs/marker_schedule_search_stage2.lsf.sh
```

Resource overrides are controlled by editing `#BSUB` lines in each job file.

## Stage Intent

- Stage 0 (`b256`, 1 seed): smoke/sanity only.
- Stage 1 (`b1024`, 3 seeds): marker-policy search
  (`pca_explained_variance`, `k_density`, `preselect_factor`, `greedy_score_weight`, marker weights).
- Stage 2 (`b4096`, 3 seeds): schedule search with fixed marker policy and varying `qbc_P`.

