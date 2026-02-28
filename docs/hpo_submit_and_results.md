# HPO Submit Commands And Result Paths

This document gives exact commands to submit QBC deep ensemble HPO jobs and where outputs are written.

## Scope

- Method: `qbc_deep_ensemble`
- Configs:
  - `src/config/hpo/qbc_deep_ensemble/smoke_stage0.yaml`
  - `src/config/hpo/qbc_deep_ensemble/policy_search_stage1.yaml`
  - `src/config/hpo/qbc_deep_ensemble/schedule_search_stage2.yaml`

## 1) Submit Commands

Run from repo root:

```bash
cd /Users/jonaswiendl/local/Repos/ml-surrogates-for-power-systems
```

Smoke check (1 job):

```bash
bash tools/hpo/smoke_hpo_lsf.sh \
  src/config/hpo/qbc_deep_ensemble/smoke_stage0.yaml
```

Stage 1 policy search (array):

```bash
bash tools/hpo/submit_hpo_lsf_array.sh \
  src/config/hpo/qbc_deep_ensemble/policy_search_stage1.yaml
```

Stage 2 schedule search (array):

```bash
bash tools/hpo/submit_hpo_lsf_array.sh \
  src/config/hpo/qbc_deep_ensemble/schedule_search_stage2.yaml
```

Submit with explicit resources:

```bash
QUEUE=gpua100 N_CORES=1 MEM_GB=24 WALL_HOURS=08 PYTHON_BIN=python3 \
bash tools/hpo/submit_hpo_lsf_array.sh \
  src/config/hpo/qbc_deep_ensemble/policy_search_stage1.yaml
```

## 2) What Gets Created

When a matrix is built, this root is created:

`outputs/hpo/qbc_deep_ensemble/<stage>/<hpo_id_or_hpo_id_timestamp>/`

Inside that root:

- `matrix.tsv`: full parameter matrix (one row = one run)
- `matrix_meta.json`: summary (`total_rows`, stage, method, paths)
- `runs/<cfg_id>/...`: per-row outputs

LSF scheduler logs:

- `outputs/lsf_logs/<job_name>.<job_id>.<array_idx>.out`
- `outputs/lsf_logs/<job_name>.<job_id>.<array_idx>.err`
- Smoke uses:
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
find outputs/hpo/qbc_deep_ensemble -maxdepth 3 -type d | sort
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
