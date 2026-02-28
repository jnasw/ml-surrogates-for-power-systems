# HPO Submit Commands And Result Paths

This document gives exact commands to submit QBC deep ensemble HPO jobs and where outputs are written.

## Scope

- Method: `qbc_deep_ensemble`
- Configs:
  - `src/config/hpo/qbc_deep_ensemble/smoke_stage0.yaml`
  - `src/config/hpo/qbc_deep_ensemble/policy_search_stage1.yaml`
  - `src/config/hpo/qbc_deep_ensemble/schedule_search_stage2.yaml`

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

You can edit `#BSUB` lines directly in each file (`queue`, `walltime`, `mem`, job name, stdout/stderr).

## Direct `bsub` Custom Command (Optional)

If you need a one-off custom submit:

```bash
python3 tools/hpo/build_hpo_matrix.py \
  --config src/config/hpo/qbc_deep_ensemble/policy_search_stage1.yaml \
  --env-out /tmp/hpo_env.sh
source /tmp/hpo_env.sh

bsub \
  -J "hpo-qbc[1-${TOTAL_ROWS}]" \
  -q gpua100 \
  -n 1 \
  -M $((24 * 1024)) \
  -W 08:00 \
  -oo "outputs/lsf_logs/hpo-qbc.%J.%I.out" \
  -eo "outputs/lsf_logs/hpo-qbc.%J.%I.err" \
  "env MATRIX_PATH='${MATRIX_PATH}' PYTHON_BIN='python3' bash tools/hpo/run_hpo_lsf_array_row.sh"
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
