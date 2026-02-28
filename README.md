# ml-surrogates-for-power-systems
MSc thesis on surrogate modelling of synchronous machine dynamics using machine learning @DTU Wind and Energy Systems.

## Dataset pipeline

The project uses a two-stage data pipeline:

1. Stage-1 raw simulation dataset creation:
   - Entry point: `create_dataset.py`
   - Output contract: `data/<MODEL>/dataset_vN/raw/file*.pkl` + `data/<MODEL>/dataset_vN/info.txt`
2. Stage-2 preprocessing:
   - Entry point: `preprocess_dataset.py`
   - Consumes the stage-1 contract and writes train/val/test HDF5 files.
   - Supports fair-comparison mode with a shared external test set.
3. Baseline training/evaluation:
   - Entry point: `run_baseline.py`
   - Consumes stage-2 HDF5 output (not raw `pkl` files).

For end-to-end execution, use:
- `tools/pipeline/run_experiment.py` (stage-1 -> stage-2 -> baseline in one command, with run manifest).

### Sampling modes in stage-1

Configure `model.ic_generation_method` in `src/config/setup_dataset.yaml`:

- `full_factorial`: per-variable sampling with cartesian product.
- `joint_lhs`: D-dimensional LHS over the full IC vector.
- `adaptive_iterative`: QBC-based adaptive sampling loop.

For `adaptive_iterative`, stage-1 uses QBC controls from the same config:
`qbc_n0`, `qbc_n_test`, `qbc_M`, `qbc_P`, `qbc_K`, `qbc_T`, plus `active.*`.

Optional logging/checkpointing during stage-1 adaptive generation:
- `qbc_enable_logging: true` enables per-round history/checkpoints (disabled by default).
- `qbc_run_dir` sets output directory (defaults to `outputs/qbc/run_<timestamp>` when logging is enabled).
- `qbc_resume_from_round` and `qbc_resume_stage` support resuming interrupted adaptive runs.

### QBC logging and resume

`create_dataset.py` is the single stage-1 entrypoint for both static and adaptive generation.
For adaptive runs, optional per-round logging/checkpointing/resume is available via the
`qbc_enable_logging`, `qbc_run_dir`, `qbc_resume_from_round`, and `qbc_resume_stage` settings.

### Metadata in `info.txt`

All stage-1 datasets include:

- `IC generation method`
- `IC per-variable sampling method`
- `IC joint sample count override`

Adaptive (QBC) datasets additionally include QBC-specific metadata (candidate method, committee size, rounds, selected-per-round, final train size, etc.).

### Fair comparison in preprocess (paired common test set)

To compare LHS vs adaptive fairly without creating an extra dataset, preprocess both datasets in mirrored
`paired_common_from_datasets` mode:

```yaml
dataset:
  test_split_mode: paired_common_from_datasets
  paired_other_dataset_number: <other dataset number>
  ic_key_decimals: 8
```

Behavior:
- Builds a shared/common test set from the union of both datasets' internal test portions.
- Removes those ICs from train/val of the current dataset to prevent leakage.
- Running preprocess for both datasets with mirrored `paired_other_dataset_number` yields an identical common test split.

Then run `run_baseline.py` once per dataset version and compare the saved metrics in
`outputs/baseline/.../metrics.json`.

## One-command experiments

Run a full pipeline experiment:

```bash
python tools/pipeline/run_experiment.py \
  --method qbc_marker_hybrid \
  --budget b256 \
  --seed s01 \
  --experiment-id thesis_sm4_local
```

Artifacts are stored under:

```text
outputs/experiments/<experiment-id>/<preset>/<method>/<budget>/<seed>/
```

Key files:
- `run_manifest.json` (single source of truth for config, stage status, artifacts, metrics)
- `logs/stage*.log` (stage logs)
- `data/<MODEL>/dataset_vN` (raw + preprocessed)
- `qbc/` (adaptive round history/checkpoints when applicable)
- `baseline/metrics.json` (baseline results when enabled)

Aggregate all run manifests:

```bash
python tools/analysis/summarize_experiments.py --root outputs/experiments --out outputs/experiments_summary.csv
```

Run a declarative campaign matrix from YAML:

```bash
python tools/pipeline/run_campaign.py --config src/config/campaign/local_smoke.yaml
```

Campaigns can optionally bootstrap a shared/common test source dataset automatically
via a `shared_test` block (see `src/config/campaign/local_value_4k_common_test.yaml`).

Dry-run the campaign (print commands only):

```bash
python tools/pipeline/run_campaign.py --config src/config/campaign/local_smoke.yaml --dry-run
```

Export dashboard-ready tables (run-level + round-level):

```bash
python tools/analysis/export_dashboard_data.py --root outputs/experiments --out-dir outputs/dashboard
```
