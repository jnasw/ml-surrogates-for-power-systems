# Historical HPO Workflow Note

This document describes the older multi-stage HPO workflow used during thesis exploration. It is kept for historical context and provenance only. The workflow configs are archived under `hpo/workflow_configs/`, and compact result tables are archived under `hpo/results/`. For the current calibration launcher, use `docs/setup/hpo_calibration.md`, `src/experiments/pipeline/run_hpo_calibration.py`, and `hpc/hpo/run_hpo_calibration.lsf.sh`.

# HPO Workflow Design

This document defines the HPO workflow contract used in this repository.

## Goals

- One workflow per `method x model_flag`
- Multi-stage progressive narrowing
- Automatic stage-to-stage handoff
- One central summary layer for all evaluated configs
- One central winner table for downstream benchmark handoff
- Clean, config-driven orchestration

## Workflow Unit

A workflow run corresponds to exactly one:

- method
- model flag
- preset / dataset context

Examples:

- `qbc_deep_ensemble x SM4`
- `marker_directed x SM4`
- `qbc_marker_hybrid x SM4`

Suggested workflow id format:

`<method>_<model_flag_lower>_v<version>`

Examples:

- `qbc_deep_ensemble_sm4_v1`
- `marker_directed_sm6_v1`

## Stage Roles

The workflow uses four named stages.

### `smoke`

Purpose:
- config validation
- dependency validation
- method sanity check

Characteristics:
- tiny budget
- 1 seed
- no scientific interpretation

### `policy`

Purpose:
- broad policy search
- identify promising acquisition mechanisms

Characteristics:
- low or medium budget
- 3 dataset seeds
- broad search over policy parameters

### `refine`

Purpose:
- refine around the best policy-search region
- optionally add operating-point search such as `qbc_P`

Characteristics:
- medium budget
- 3 dataset seeds
- narrowed search space
- inherits a shortlist from `policy`

### `confirm`

Purpose:
- benchmark-ready confirmation
- validate finalists at the target budget

Characteristics:
- target budget
- small candidate set only
- 3 to 5 dataset seeds
- objective should be as close as possible to the benchmark objective

## Workflow Output Structure

Suggested root:

```text
outputs/hpo_workflows/<workflow_id>_<timestamp>/
  workflow_manifest.json
  smoke/
    matrix.tsv
    stage_results.csv
    stage_ranked.csv
    stage_winner.json
    stage_shortlist.json
  policy/
    ...
  refine/
    ...
  confirm/
    ...
```

Each stage writes the same artifact contract.

## Stage Artifact Contract

Each stage should write:

- `matrix.tsv`
  - raw expanded search matrix
- `stage_results.csv`
  - one normalized row per evaluated config row
- `stage_status.json`
  - completion status for the whole stage; ranking only happens when it is complete
- `stage_ranked.csv`
  - aggregated ranking across dataset seeds
- `stage_winner.json`
  - selected winner and supporting metadata
- `stage_shortlist.json`
  - top-k configs passed to the next stage

Each stage should also define a ranking contract in the workflow config:

- `ranking.group_by`
  - explicit flattened columns that define one config across dataset seeds
- `objective.metric`
  - metric column used for ranking
- `objective.direction`
  - `minimize` or `maximize`
- `objective.aggregation`
  - currently `mean_over_dataset_seeds`
- `objective.tie_breakers`
  - ordered list of secondary sort keys
- `shortlist.top_k`
  - number of configs propagated to the next stage

Optional:

- `stage_notes.md`
- `stage_plots/`

## Central Summary Artifacts

These are the canonical outputs for later analysis and benchmark handoff.

### `hpo/hpo/results_summary.csv`

One row per evaluated config row.

Required columns:

- `workflow_id`
- `workflow_version`
- `method`
- `model_flag`
- `preset`
- `stage`
- `stage_role`
- `stage_root`
- `cfg_id`
- `parent_cfg_id`
- `dataset_seed`
- `budget`
- `status`
- `objective_metric`
- `objective_direction`
- `objective_value`
- `rank_within_stage`
- `is_shortlisted`
- `is_stage_winner`
- `run_root`

Recommended metric columns:

- `final_eval_rmse`
- `final_eval_mse`
- `final_round_seconds`
- `final_train_seconds`
- `final_candidate_simulation_seconds`
- `final_acquisition_seconds`
- `final_candidate_generation_seconds`

Recommended flattened hyperparameter columns:

QBC:

- `qbc_n0`
- `qbc_M`
- `qbc_P`
- `qbc_K`
- `qbc_T`
- `active.disagreement.metric`
- `active.diversity.preselect_factor`
- `active.diversity.uncertainty_weight`
- `active.diversity.distance_weight`

Marker:

- `qbc_n0`
- `qbc_P`
- `qbc_K`
- `qbc_T`
- `active.marker.pca_explained_variance`
- `active.marker.k_density`
- `active.marker.preselect_factor`
- `active.marker.greedy_score_weight`
- `active.marker.weights.diversity`
- `active.marker.weights.sparsity`

Hybrid:

- `qbc_n0`
- `qbc_M`
- `qbc_P`
- `qbc_K`
- `qbc_T`
- `active.disagreement.metric`
- `active.hybrid.pca_explained_variance`
- `active.hybrid.k_density`
- `active.hybrid.preselect_factor`
- `active.hybrid.greedy_score_weight`
- `active.hybrid.weights.uncertainty`
- `active.hybrid.weights.diversity`
- `active.hybrid.weights.sparsity`

### `hpo/hpo/results_winners.csv`

One row per workflow-stage winner.

Required columns:

- `workflow_id`
- `workflow_version`
- `method`
- `model_flag`
- `preset`
- `stage`
- `stage_role`
- `winner_cfg_id`
- `source_stage_root`
- `objective_metric`
- `objective_direction`
- `objective_mean`
- `objective_std`
- `n_dataset_seeds`
- `shortlist_size`
- `parent_stage`

Plus flattened winner hyperparameter columns.

## Objective Contract

Each stage must define:

- `metric`
- `direction`
- `aggregation`
- `tie_breakers`

Example:

```yaml
objective:
  metric: final_eval_rmse
  direction: minimize
  aggregation: mean_over_dataset_seeds
  tie_breakers:
    - std_over_dataset_seeds
    - mean_final_round_seconds
```

## Ranking Contract

Each stage must also define which flattened columns identify one config across
dataset seeds.

Example:

```yaml
ranking:
  group_by:
    - qbc_M
    - active.disagreement.metric
    - active.diversity.preselect_factor
    - active.diversity.uncertainty_weight
    - active.diversity.distance_weight
```

The ranker aggregates all rows that share the same `group_by` values and then
computes:

- `objective_mean`
- `objective_std`
- `objective_min`
- `objective_max`
- `n_dataset_seeds`
- mean runtime metrics for tie-breaking and later reporting

Required stage artifacts produced by the ranker:

- `stage_ranked.csv`
- `stage_winner.json`
- `stage_shortlist.json`

For coupled policy weights, prefer explicit named profiles over independent
Cartesian axes. Example: hybrid acquisition should search a small set of
interpretable weight profiles such as `qbc_heavy`, `balanced`, and
`marker_heavy`, and derive the actual
`active.hybrid.weights.{uncertainty,diversity,sparsity}` overrides from that
profile before writing the stage matrix.

The same rule applies to other coupled policy weights. Example:
marker-directed should prefer explicit diversity/sparsity policy profiles such
as `diversity_heavy`, `balanced`, and `sparsity_heavy` over independent
Cartesian scans of `active.marker.weights.diversity` and
`active.marker.weights.sparsity`.

## Dependency Contract

Stages may depend on previous stages.

Required dependency fields:

- `depends_on`
- `inherit.mode`
- `inherit.top_k`
- `inherit.from_artifact`

Supported inherit modes:

- `winner_only`
- `shortlist`

The dependency should be file-based, not notebook-based. `refine` should
consume `policy/stage_shortlist.json`, and `confirm` should consume
`refine/stage_shortlist.json`.

## Narrowing Contract

Dependent stages must define how narrowing works.

Required fields:

- `lock_params`
- `refine`
- `new_axes`

Meaning:

- `lock_params`
  - parameters inherited and kept fixed
- `refine`
  - inherited parameters refined around the best region with an explicit rule
- `new_axes`
  - additional parameters introduced in later stages, such as `qbc_P`

Initial supported refinement mode:

```yaml
narrowing:
  refine:
    active.diversity.uncertainty_weight:
      mode: neighbors_from_grid
      values: [0.4, 0.6, 0.8]
      radius: 1
```

Meaning:

- take the shortlisted config value
- find it in the ordered `values` grid
- include its `radius` neighbors on both sides
- union and deduplicate refined configs across the shortlist

## Configuration Contract

Suggested config layout:

```yaml
workflow:
  id: qbc_deep_ensemble_sm4_v1
  version: 1
  method: qbc_deep_ensemble
  model_flag: SM4
  preset: main
  experiment_id_prefix: thesis_qbc_sm4_hpo
  output_root: outputs/hpo_workflows

stages:
  smoke:
    ...
  policy:
    ...
  refine:
    ...
  confirm:
    ...
```

Each stage should include:

- `enabled`
- `budget`
- `seeds`
- `objective`
- `ranking`
- `search`
- `fixed_overrides`
- optional `inherit`
- optional `shortlist`
- optional `narrowing`

## Rollout Order

Recommended execution order:

1. `qbc_deep_ensemble x SM4`
2. `marker_directed x SM4`
3. `qbc_marker_hybrid x SM4`
4. higher-order models

## Historical Submission Model

The archived configs in `hpo/workflow_configs/` were originally consumed by a
legacy workflow runner. That runner is not part of the current canonical
submission path. The preserved configs and compact result tables are kept here
to document the HPO search spaces and selected winners used during thesis
method selection.

For current runnable HPO calibration commands, use:

- `docs/setup/hpo_calibration.md`
- `src/experiments/pipeline/run_hpo_calibration.py`
- `hpc/hpo/run_hpo_calibration.lsf.sh`

Example archived workflow config:

- `hpo/workflow_configs/qbc_deep_ensemble/sm4.yaml`

Default retention policy for workflow HPO runs:

- prune `run_root/data`
- prune `run_root/qbc/rounds`
- prune `run_root/qbc/checkpoints`
- keep:
  - `dataset_manifest.json`
  - `run_manifest.json`
  - `qbc/history.jsonl`
  - `telemetry/round_telemetry.csv`
  - `hpo_status.json`
  - stage/local/global summary artifacts

Production execution rule:

- a stage only writes winner and shortlist artifacts after all rows completed successfully
- incomplete stages stop the workflow and do not hand off to downstream stages
- `--max-rows` is debug-only and therefore intentionally produces an incomplete stage
