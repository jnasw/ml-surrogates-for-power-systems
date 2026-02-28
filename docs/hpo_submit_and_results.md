# HPO Submit Commands And Result Paths

This document gives exact commands to submit HPO jobs and where outputs are written.

## Scope

- `qbc_deep_ensemble`
- `marker_directed`

Config files:

- `src/config/hpo/qbc_deep_ensemble/smoke_stage0.yaml`
- `src/config/hpo/qbc_deep_ensemble/policy_search_stage1.yaml`
- `src/config/hpo/qbc_deep_ensemble/schedule_search_stage2.yaml`
- `src/config/hpo/marker_directed/smoke_stage0.yaml`
- `src/config/hpo/marker_directed/policy_search_stage1.yaml`
- `src/config/hpo/marker_directed/schedule_search_stage2.yaml`

## 1) Submit Commands (`bsub < file.sh`)

Run from repo root:

```bash
cd /Users/jonaswiendl/local/Repos/ml-surrogates-for-power-systems
```

Smoke check (1 job):

```bash
bsub < tools/hpo/jobs/qbc_smoke_stage0.lsf.sh
```

Stage 1 policy search:

```bash
bsub < tools/hpo/jobs/qbc_policy_search_stage1.lsf.sh
```

Stage 2 schedule search:

```bash
bsub < tools/hpo/jobs/qbc_schedule_search_stage2.lsf.sh
```

Marker smoke:

```bash
bsub < tools/hpo/jobs/marker_smoke_stage0.lsf.sh
```

Marker stage 1 policy search:

```bash
bsub < tools/hpo/jobs/marker_policy_search_stage1.lsf.sh
```

Marker stage 2 schedule search:

```bash
bsub < tools/hpo/jobs/marker_schedule_search_stage2.lsf.sh
```

Override config or python at submit time:

```bash
HPO_CONFIG=src/config/hpo/qbc_deep_ensemble/policy_search_stage1.yaml \
PYTHON_BIN=python3 \
bsub < tools/hpo/jobs/qbc_policy_search_stage1.lsf.sh
```

These scripts are fully self-contained:
- build matrix
- execute rows
- write all logs/artifacts

No wrapper call is required.

## Job Scripts

- `tools/hpo/jobs/qbc_smoke_stage0.lsf.sh`
- `tools/hpo/jobs/qbc_policy_search_stage1.lsf.sh`
- `tools/hpo/jobs/qbc_schedule_search_stage2.lsf.sh`
- `tools/hpo/jobs/marker_smoke_stage0.lsf.sh`
- `tools/hpo/jobs/marker_policy_search_stage1.lsf.sh`
- `tools/hpo/jobs/marker_schedule_search_stage2.lsf.sh`

You can edit `#BSUB` lines directly in each file (`queue`, `walltime`, `mem`, job name, stdout/stderr).

## Notes

- Job scripts are the single submit interface.
- Keep resource settings in `#BSUB` headers for reproducibility.

## 2) What Gets Created

When a matrix is built, this root is created:

`outputs/hpo/<method>/<stage>/<hpo_id_or_hpo_id_timestamp>/`

Inside that root:

- `matrix.tsv`: full parameter matrix (one row = one run)
- `matrix_meta.json`: summary (`total_rows`, stage, method, paths)
- `runs/<cfg_id>/...`: per-row outputs

LSF scheduler logs:

- `outputs/lsf_logs/<job_name>.<job_id>.out`
- `outputs/lsf_logs/<job_name>.<job_id>.err`

## 3) Per-Run Result Structure

For each row (`runs/<cfg_id>/`), HPO writes:

- `hpo_row.json`
- `hpo_command.sh`
- `hpo_status.json`

Then `tools/pipeline/run_experiment.py` writes:

- `run_manifest.json`
- `logs/stage1_create_dataset.log`
- `data/` (stage-1 dataset outputs)
- `qbc/` (adaptive loop history and round checkpoints, including `history.jsonl` when available)
- `hydra/` (Hydra job directories for stage scripts)

If preprocess/baseline are enabled in HPO config:

- `logs/stage2_preprocess.log`
- `logs/stage3_baseline.log`
- `baseline/metrics.json` (if stage-3 ran successfully)
- `telemetry/round_telemetry.csv` (+ parquet if dependencies available)

## 4) Useful Checks

Show latest HPO roots:

```bash
find outputs/hpo -maxdepth 3 -type d | sort
```

Check matrix size:

```bash
python3 - <<'PY'
import csv
p="outputs/hpo/qbc_deep_ensemble/stage1_policy_search/qbc_deep_ensemble_policy_search/matrix.tsv"
with open(p) as f:
    n=sum(1 for _ in csv.DictReader(f, delimiter="\t"))
print(n)
PY
```

Inspect one run status:

```bash
cat outputs/hpo/qbc_deep_ensemble/stage1_policy_search/qbc_deep_ensemble_policy_search/runs/<cfg_id>/hpo_status.json
```
