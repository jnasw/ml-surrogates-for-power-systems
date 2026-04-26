"""Campaign planning helpers for declarative experiment matrices."""

from __future__ import annotations

import csv
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from omegaconf import OmegaConf


def to_container_if_config(value: Any, default: Any) -> Any:
    if OmegaConf.is_config(value):
        out = OmegaConf.to_container(value, resolve=True)
        return default if out is None else out
    if value is None:
        return default
    return value


def matches_rule(combo: dict[str, str], rule: dict[str, Any]) -> bool:
    for k, v in rule.items():
        if str(combo.get(k)) != str(v):
            return False
    return True


def build_matrix(axes: dict[str, list[Any]]) -> list[dict[str, str]]:
    import itertools

    keys = list(axes.keys())
    vals = [axes[k] for k in keys]
    out: list[dict[str, str]] = []
    for prod in itertools.product(*vals):
        out.append({k: str(v) for k, v in zip(keys, prod)})
    return out


def normalize_dataset_axes(raw_axes: dict[str, list[Any]]) -> dict[str, list[Any]]:
    axes = dict(raw_axes)
    if "baseline_seed" in axes:
        raise ValueError("Use top-level 'baseline_seeds', not 'axes.baseline_seed'.")
    if "dataset_seed" not in axes:
        raise ValueError("campaign config must define 'axes.dataset_seed'.")
    return axes


def apply_exclusions(
    combos: list[dict[str, str]],
    excludes: list[dict[str, Any]],
) -> list[dict[str, str]]:
    if not excludes:
        return combos
    keep: list[dict[str, str]] = []
    for combo in combos:
        if any(matches_rule(combo, rule) for rule in excludes):
            continue
        keep.append(combo)
    return keep


def _read_csv_rows(path: str) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def resolve_hpo_run_overrides(cfg: Any, repo_root: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    hpo_cfg = dict(to_container_if_config(getattr(cfg, "hpo_integration", {}), {}))
    if not bool(hpo_cfg.get("enabled", False)):
        return [], {"enabled": False, "resolved": []}

    winners_csv = str(hpo_cfg.get("winners_csv", "results/hpo/hpo_winners.csv"))
    winners_path = winners_csv if os.path.isabs(winners_csv) else os.path.join(repo_root, winners_csv)
    if not os.path.exists(winners_path):
        raise FileNotFoundError(f"hpo_integration winners_csv not found: {winners_path}")

    winner_rows = _read_csv_rows(winners_path)
    policies = list(to_container_if_config(hpo_cfg.get("policies", []), []))
    resolved_rules: list[dict[str, Any]] = []
    resolved_meta: list[dict[str, Any]] = []

    for policy in policies:
        method = str(policy.get("method", "")).strip()
        workflow_id = str(policy.get("workflow_id", "")).strip()
        preferred_stages = [str(x) for x in policy.get("preferred_stages", [])]
        stage1_keys = [str(x) for x in policy.get("stage1_keys", [])]
        allow_missing = bool(policy.get("allow_missing", False))
        if not method or not workflow_id or not preferred_stages or not stage1_keys:
            raise ValueError("Each hpo_integration policy requires method, workflow_id, preferred_stages, and stage1_keys.")

        selected_row: dict[str, str] | None = None
        selected_stage = ""
        for stage_name in preferred_stages:
            stage_matches = [
                row for row in winner_rows if row.get("workflow_id") == workflow_id and row.get("stage") == stage_name
            ]
            if stage_matches:
                selected_row = stage_matches[-1]
                selected_stage = stage_name
                break

        if selected_row is None:
            if allow_missing:
                resolved_meta.append(
                    {
                        "method": method,
                        "workflow_id": workflow_id,
                        "status": "missing_allowed",
                        "preferred_stages": preferred_stages,
                    }
                )
                continue
            raise ValueError(
                f"No HPO winner found for workflow_id='{workflow_id}' in stages {preferred_stages}. "
                "Either run HPO first or set allow_missing=true for this policy."
            )

        stage1 = []
        for key in stage1_keys:
            value = str(selected_row.get(key, "")).strip()
            if value:
                stage1.append(f"{key}={value}")
        if not stage1:
            if allow_missing:
                resolved_meta.append(
                    {
                        "method": method,
                        "workflow_id": workflow_id,
                        "status": "empty_allowed",
                        "selected_stage": selected_stage,
                    }
                )
                continue
            raise ValueError(
                f"HPO winner for workflow_id='{workflow_id}' stage='{selected_stage}' did not provide any configured stage1 keys."
            )

        resolved_rules.append(
            {
                "match": {"method": method},
                "stage1": stage1,
            }
        )
        resolved_meta.append(
            {
                "method": method,
                "workflow_id": workflow_id,
                "status": "resolved",
                "selected_stage": selected_stage,
                "stage1": stage1,
            }
        )

    return resolved_rules, {"enabled": True, "winners_path": winners_path, "resolved": resolved_meta}


def expected_run_root(
    *,
    repo_root: str,
    experiment_id: str,
    preset: str,
    method: str,
    budget: str,
    dataset_seed: str,
) -> str:
    return os.path.join(
        repo_root,
        "outputs",
        "experiments",
        experiment_id,
        preset,
        method,
        budget,
        dataset_seed,
    )


@dataclass(frozen=True)
class CampaignRunPlan:
    index: int
    combo: dict[str, str]
    method: str
    budget: str
    dataset_seed: str
    command: list[str]
    stage1_overrides: list[str]
    stage2_overrides: list[str]
    stage3_overrides: list[str]


@dataclass(frozen=True)
class CampaignPlan:
    axes: dict[str, list[Any]]
    combos: list[dict[str, str]]
    baseline_seeds: list[str]
    total_dataset_runs: int
    total_baseline_subruns: int
    run_override_rules: list[dict[str, Any]]
    hpo_integration_meta: dict[str, Any]
    stage1_overrides: list[str]
    stage2_overrides: list[str]
    stage3_overrides: list[str]
    campaign_root: str
    run_plans: list[CampaignRunPlan]


def resolve_campaign_plan(
    *,
    cfg: Any,
    config_path: str,
    repo_root: str,
    name: str,
    experiment_id: str,
    preset: str,
    model_flag: str,
    skip_preprocess: bool,
    skip_baseline: bool,
    dry_run: bool,
    force: bool,
) -> CampaignPlan:
    axes = normalize_dataset_axes(dict(to_container_if_config(cfg.axes, {})))
    combos = build_matrix(axes)
    excludes = list(to_container_if_config(getattr(cfg, "exclude", []), []))
    combos = apply_exclusions(combos, excludes)
    if not combos:
        raise ValueError("No campaign combinations remain after exclusions.")

    baseline_seeds = [str(x) for x in to_container_if_config(getattr(cfg, "baseline_seeds", []), [])]
    if not skip_baseline and not baseline_seeds:
        raise ValueError("Campaign requires 'baseline_seeds' unless skip_baseline=true.")

    stage1_overrides = [str(x) for x in to_container_if_config(getattr(cfg, "stage1_overrides", []), [])]
    stage2_overrides = [str(x) for x in to_container_if_config(getattr(cfg, "stage2_overrides", []), [])]
    stage3_overrides = [str(x) for x in to_container_if_config(getattr(cfg, "stage3_overrides", []), [])]

    hpo_run_overrides, hpo_integration_meta = resolve_hpo_run_overrides(cfg, repo_root)
    config_run_overrides = list(to_container_if_config(getattr(cfg, "run_overrides", []), []))
    run_override_rules = [*config_run_overrides, *hpo_run_overrides]

    campaign_root = os.path.dirname(os.path.abspath(config_path))
    run_plans: list[CampaignRunPlan] = []

    for idx, combo in enumerate(combos, start=1):
        method = str(combo.get("method", "")).strip()
        budget = str(combo.get("budget", "")).strip()
        dataset_seed = str(combo.get("dataset_seed", "")).strip()
        if not method or not budget or not dataset_seed:
            raise ValueError(f"Each combo must define method, budget, and dataset_seed. Got: {combo}")

        combo_stage1 = list(stage1_overrides)
        combo_stage2 = list(stage2_overrides)
        combo_stage3 = list(stage3_overrides)
        for rule in run_override_rules:
            match = dict(rule.get("match", {}))
            if matches_rule(combo, match):
                combo_stage1.extend(str(x) for x in rule.get("stage1", []))
                combo_stage2.extend(str(x) for x in rule.get("stage2", []))
                combo_stage3.extend(str(x) for x in rule.get("stage3", []))

        run_root = expected_run_root(
            repo_root=repo_root,
            experiment_id=experiment_id,
            preset=preset,
            method=method,
            budget=budget,
            dataset_seed=dataset_seed,
        )

        command = [
            sys.executable,
            "-m",
            "src.experiments.pipeline.run_experiment",
            "--method",
            method,
            "--budget",
            budget,
            "--dataset-seed",
            dataset_seed,
            "--preset",
            preset,
            "--experiment-id",
            experiment_id,
            "--model-flag",
            model_flag,
            "--run-root",
            run_root,
        ]
        if skip_preprocess:
            command.append("--skip-preprocess")
        if skip_baseline:
            command.append("--skip-baseline")
        else:
            for baseline_seed in baseline_seeds:
                command.extend(["--baseline-seed", baseline_seed])
        if dry_run:
            command.append("--dry-run")
        if force:
            command.append("--force")
        for override in combo_stage1:
            command.extend(["--stage1-override", override])
        for override in combo_stage2:
            command.extend(["--stage2-override", override])
        for override in combo_stage3:
            command.extend(["--stage3-override", override])

        run_plans.append(
            CampaignRunPlan(
                index=idx,
                combo=combo,
                method=method,
                budget=budget,
                dataset_seed=dataset_seed,
                command=command,
                stage1_overrides=combo_stage1,
                stage2_overrides=combo_stage2,
                stage3_overrides=combo_stage3,
            )
        )

    return CampaignPlan(
        axes=axes,
        combos=combos,
        baseline_seeds=baseline_seeds,
        total_dataset_runs=len(run_plans),
        total_baseline_subruns=len(run_plans) * len(baseline_seeds) if not skip_baseline else 0,
        run_override_rules=run_override_rules,
        hpo_integration_meta=hpo_integration_meta,
        stage1_overrides=stage1_overrides,
        stage2_overrides=stage2_overrides,
        stage3_overrides=stage3_overrides,
        campaign_root=campaign_root,
        run_plans=run_plans,
    )

