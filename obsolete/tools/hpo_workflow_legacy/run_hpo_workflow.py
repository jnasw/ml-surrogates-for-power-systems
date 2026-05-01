#!/usr/bin/env python3
"""Run one multi-stage HPO workflow end-to-end."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omegaconf import OmegaConf

from tools.hpo_workflow.matrix_utils import (
    cfg_hash,
    derive_stage1_complement,
    enforce_budget_constraint,
    merge_stage_overrides,
    serialize_overrides,
)
from tools.hpo_workflow.collect_stage_results import collect_stage_results, write_stage_results_csv
from tools.hpo_workflow.rank_stage_results import (
    _resolve_objective,
    aggregate_stage_results,
    rank_aggregated_results,
    write_ranked_csv,
    write_shortlist_json,
    write_winner_json,
)
from tools.hpo_workflow.update_hpo_summaries import (
    annotate_stage_results,
    flatten_winner_payload,
    update_hpo_summary,
    update_hpo_winners,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_dict(value: Any) -> dict[str, Any]:
    out = OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
    if out is None:
        return {}
    if not isinstance(out, dict):
        raise ValueError(f"Expected mapping, got {type(out)}")
    return out


def _to_list(value: Any) -> list[Any]:
    out = OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
    if out is None:
        return []
    if not isinstance(out, list):
        raise ValueError(f"Expected list, got {type(out)}")
    return out


def _load_workflow_config(path: Path) -> dict[str, Any]:
    cfg = OmegaConf.to_container(OmegaConf.load(str(path)), resolve=True)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid workflow config: {path}")
    return cfg


def _normalize_stage_name(name: str) -> str:
    text = str(name or "").strip()
    if not text:
        raise ValueError("Stage name must be non-empty.")
    return text


def _workflow_root(base_root: str, workflow_id: str, with_timestamp: bool) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{workflow_id}_{stamp}" if with_timestamp else workflow_id
    return Path(base_root) / name


def _execution_defaults(workflow: dict[str, Any]) -> dict[str, Any]:
    execution = workflow.get("execution", {})
    if not isinstance(execution, dict):
        execution = {}
    return {
        "skip_preprocess": bool(execution.get("skip_preprocess", True)),
        "skip_baseline": bool(execution.get("skip_baseline", True)),
        "baseline_epochs": execution.get("baseline_epochs"),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _read_shortlist(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid shortlist payload: {path}")
    return payload


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid JSON payload: {path}")
    return payload


def _matrix_fieldnames() -> list[str]:
    return [
        "row_idx",
        "cfg_id",
        "method",
        "preset",
        "budget",
        "seed",
        "experiment_id",
        "model_flag",
        "skip_preprocess",
        "skip_baseline",
        "baseline_epochs",
        "run_root",
        "stage1_overrides",
        "stage2_overrides",
        "stage3_overrides",
    ]


def _write_matrix(rows: list[dict[str, Any]], matrix_path: Path) -> None:
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    with matrix_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_matrix_fieldnames(), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _empty_stage_override_maps() -> dict[str, dict[str, str]]:
    return {"stage1": {}, "stage2": {}, "stage3": {}}


def _parse_fixed_overrides(stage_cfg: dict[str, Any]) -> dict[str, list[str]]:
    fixed = _to_dict(stage_cfg.get("fixed_overrides", {}))
    return {
        "stage1": [str(x) for x in _to_list(fixed.get("stage1"))],
        "stage2": [str(x) for x in _to_list(fixed.get("stage2"))],
        "stage3": [str(x) for x in _to_list(fixed.get("stage3"))],
    }


def _coerce_float(text: Any) -> float | None:
    try:
        return float(str(text).strip())
    except (TypeError, ValueError):
        return None


def _find_grid_index(value: str, values: list[Any]) -> int:
    for idx, candidate in enumerate(values):
        if str(candidate) == str(value):
            return idx
    value_num = _coerce_float(value)
    if value_num is None:
        raise ValueError(f"Value '{value}' not found in refinement grid {values}")
    for idx, candidate in enumerate(values):
        cand_num = _coerce_float(candidate)
        if cand_num is not None and abs(cand_num - value_num) < 1e-12:
            return idx
    raise ValueError(f"Value '{value}' not found in refinement grid {values}")


def _refine_neighbors_from_grid(current_value: str, spec: dict[str, Any]) -> list[str]:
    values = _to_list(spec.get("values"))
    if not values:
        raise ValueError("neighbors_from_grid requires a non-empty 'values' list.")
    radius = int(spec.get("radius", 1))
    if radius < 0:
        raise ValueError("neighbors_from_grid requires radius >= 0.")
    center = _find_grid_index(current_value, values)
    lo = max(0, center - radius)
    hi = min(len(values), center + radius + 1)
    return [str(value) for value in values[lo:hi]]


def _refinement_values(current_value: str, spec: dict[str, Any]) -> list[str]:
    mode = str(spec.get("mode", "")).strip()
    if mode == "neighbors_from_grid":
        return _refine_neighbors_from_grid(current_value, spec)
    raise ValueError(f"Unsupported refinement mode '{mode}'.")


def _refine_spec_map(narrowing_cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    refine_cfg = narrowing_cfg.get("refine", {})
    if isinstance(refine_cfg, dict):
        return {str(key): _to_dict(value) for key, value in refine_cfg.items()}
    return {}


def _inherit_stage_override_maps(
    *,
    stage_cfg: dict[str, Any],
    previous_stage_root: Path,
) -> list[dict[str, dict[str, str]]]:
    inherit_cfg = _to_dict(stage_cfg.get("inherit", {}))
    narrowing_cfg = _to_dict(stage_cfg.get("narrowing", {}))
    artifact_name = str(inherit_cfg.get("from_artifact", "stage_shortlist.json"))
    shortlist_path = previous_stage_root / artifact_name
    if not shortlist_path.exists():
        raise FileNotFoundError(
            f"Dependent stage requires inherited artifact '{artifact_name}' in {previous_stage_root}. "
            "Run the upstream stage first so its shortlist exists."
        )
    shortlist = _read_shortlist(shortlist_path)
    items = shortlist.get("items", [])
    if not isinstance(items, list):
        raise ValueError(f"Invalid shortlist items in {shortlist_path}")

    top_k = int(inherit_cfg.get("top_k", len(items)))
    keep_params = {str(x) for x in _to_list(narrowing_cfg.get("lock_params"))}
    keep_params.update(str(x) for x in _to_list(narrowing_cfg.get("refine_params")))
    keep_params.update(_refine_spec_map(narrowing_cfg).keys())

    inherited_rows: list[dict[str, dict[str, str]]] = []
    for item in items[:top_k]:
        if not isinstance(item, dict):
            continue
        hyper = item.get("hyperparameters", {})
        if not isinstance(hyper, dict):
            continue
        stage_maps = _empty_stage_override_maps()
        for key, value in hyper.items():
            if keep_params and str(key) not in keep_params:
                continue
            stage_maps["stage1"][str(key)] = str(value)
        inherited_rows.append(stage_maps)
    if not inherited_rows:
        raise ValueError(f"No inherited configs resolved from {shortlist_path}")
    return inherited_rows


def _independent_stage_override_maps(stage_cfg: dict[str, Any]) -> list[dict[str, dict[str, str]]]:
    search_cfg = _to_dict(stage_cfg.get("search", {}))
    stage_axes = {name: _to_dict(search_cfg.get(name, {})) for name in ("stage1", "stage2", "stage3")}
    fixed = _parse_fixed_overrides(stage_cfg)

    axes_keys = {name: list(stage_axes[name].keys()) for name in stage_axes}
    axes_values = {
        name: [list(stage_axes[name][key]) for key in axes_keys[name]] if axes_keys[name] else []
        for name in stage_axes
    }

    out: list[dict[str, dict[str, str]]] = []
    for combo1 in (product(*axes_values["stage1"]) if axes_keys["stage1"] else [tuple()]):
        for combo2 in (product(*axes_values["stage2"]) if axes_keys["stage2"] else [tuple()]):
            for combo3 in (product(*axes_values["stage3"]) if axes_keys["stage3"] else [tuple()]):
                out.append(
                    {
                        "stage1": merge_stage_overrides(fixed["stage1"], axes_keys["stage1"], combo1),
                        "stage2": merge_stage_overrides(fixed["stage2"], axes_keys["stage2"], combo2),
                        "stage3": merge_stage_overrides(fixed["stage3"], axes_keys["stage3"], combo3),
                    }
                )
    return out


def _expand_dependent_stage(
    *,
    stage_cfg: dict[str, Any],
    previous_stage_root: Path,
) -> list[dict[str, dict[str, str]]]:
    base_rows = _inherit_stage_override_maps(stage_cfg=stage_cfg, previous_stage_root=previous_stage_root)
    fixed = _parse_fixed_overrides(stage_cfg)
    narrowing_cfg = _to_dict(stage_cfg.get("narrowing", {}))
    refine_specs = _refine_spec_map(narrowing_cfg)
    new_axes_cfg = _to_dict(narrowing_cfg.get("new_axes", {}))
    new_stage_axes = {name: _to_dict(new_axes_cfg.get(name, {})) for name in ("stage1", "stage2", "stage3")}

    out: list[dict[str, dict[str, str]]] = []
    seen: set[tuple[str, str, str]] = set()
    for base in base_rows:
        stage_maps = {
            "stage1": dict(base["stage1"]),
            "stage2": dict(base["stage2"]),
            "stage3": dict(base["stage3"]),
        }
        for stage_name, overrides in fixed.items():
            merged = merge_stage_overrides(overrides, [], tuple())
            stage_maps[stage_name].update(merged)

        refine_keys = list(refine_specs.keys())
        refine_values = []
        for key in refine_keys:
            if key not in stage_maps["stage1"]:
                raise ValueError(f"Refine parameter '{key}' is not present in inherited stage1 overrides.")
            refine_values.append(_refinement_values(stage_maps["stage1"][key], refine_specs[key]))

        stage_keys = {name: list(new_stage_axes[name].keys()) for name in new_stage_axes}
        stage_values = {
            name: [list(new_stage_axes[name][key]) for key in stage_keys[name]] if stage_keys[name] else []
            for name in new_stage_axes
        }

        for refine_combo in (product(*refine_values) if refine_keys else [tuple()]):
            refined_stage_maps = {
                "stage1": dict(stage_maps["stage1"]),
                "stage2": dict(stage_maps["stage2"]),
                "stage3": dict(stage_maps["stage3"]),
            }
            for key, value in zip(refine_keys, refine_combo):
                refined_stage_maps["stage1"][key] = str(value)

            for combo1 in (product(*stage_values["stage1"]) if stage_keys["stage1"] else [tuple()]):
                for combo2 in (product(*stage_values["stage2"]) if stage_keys["stage2"] else [tuple()]):
                    for combo3 in (product(*stage_values["stage3"]) if stage_keys["stage3"] else [tuple()]):
                        row = {
                            "stage1": dict(refined_stage_maps["stage1"]),
                            "stage2": dict(refined_stage_maps["stage2"]),
                            "stage3": dict(refined_stage_maps["stage3"]),
                        }
                        for key, value in zip(stage_keys["stage1"], combo1):
                            row["stage1"][str(key)] = str(value)
                        for key, value in zip(stage_keys["stage2"], combo2):
                            row["stage2"][str(key)] = str(value)
                        for key, value in zip(stage_keys["stage3"], combo3):
                            row["stage3"][str(key)] = str(value)
                        signature = (
                            serialize_overrides(row["stage1"]),
                            serialize_overrides(row["stage2"]),
                            serialize_overrides(row["stage3"]),
                        )
                        if signature in seen:
                            continue
                        seen.add(signature)
                        out.append(row)
    return out


def _apply_derive_and_constraints(
    *,
    workflow_stage_cfg: dict[str, Any],
    workflow_wrapper_cfg: dict[str, Any],
    budget: str,
    stage_maps: dict[str, dict[str, str]],
    repo_root: Path,
) -> None:
    derive_cfg = _to_dict(workflow_stage_cfg.get("derive", {}))
    stage1_derive = _to_dict(derive_cfg.get("stage1", {}))
    profile_specs: list[dict[str, Any]] = []
    override_profiles = _to_dict(stage1_derive.get("override_profiles", {}))
    if override_profiles:
        profile_specs.append(override_profiles)
    hybrid_profiles = _to_dict(stage1_derive.get("hybrid_weight_profiles", {}))
    if hybrid_profiles:
        profile_specs.append(
            {
                "profile_key": hybrid_profiles.get("profile_key", "hybrid_weight_profile"),
                "profiles": {
                    name: {
                        "active.hybrid.weights.uncertainty": values.get("uncertainty"),
                        "active.hybrid.weights.diversity": values.get("diversity"),
                        "active.hybrid.weights.sparsity": values.get("sparsity"),
                    }
                    for name, values in _to_dict(hybrid_profiles.get("profiles", {})).items()
                },
            }
        )
    for spec in profile_specs:
        profile_key = str(spec.get("profile_key", "")).strip()
        profiles = _to_dict(spec.get("profiles", {}))
        if not profile_key or profile_key not in stage_maps["stage1"]:
            continue
        profile_name = str(stage_maps["stage1"].pop(profile_key))
        if profile_name not in profiles:
            raise ValueError(f"Unknown override profile '{profile_name}'. Available: {sorted(profiles.keys())}")
        overrides = _to_dict(profiles[profile_name])
        for target_key, value in overrides.items():
            if value is None:
                raise ValueError(f"Override profile '{profile_name}' is missing a value for '{target_key}'.")
            stage_maps["stage1"][str(target_key)] = str(value)

    shim_cfg = {
        "matrix": {
            "derive": derive_cfg,
            "constraints": _to_dict(workflow_stage_cfg.get("constraints", {})),
        }
    }
    derive_stage1_complement(shim_cfg, stage_maps["stage1"])
    enforce_budget_constraint(shim_cfg, repo_root, budget, stage_maps["stage1"])


def _build_stage_rows(
    *,
    workflow_cfg: dict[str, Any],
    workflow_root: Path,
    stage_name: str,
    stage_cfg: dict[str, Any],
    previous_stage_root: Path | None,
) -> list[dict[str, Any]]:
    repo_root = REPO_ROOT
    workflow = _to_dict(workflow_cfg.get("workflow", {}))
    execution = _execution_defaults(workflow)
    method = str(workflow.get("method", "")).strip()
    preset = str(workflow.get("preset", "default"))
    model_flag = str(workflow.get("model_flag", ""))
    experiment_prefix = str(workflow.get("experiment_id_prefix", f"{method}_hpo"))
    stage_root = workflow_root / stage_name
    runs_root = stage_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    inherit_cfg = _to_dict(stage_cfg.get("inherit", {}))
    use_inherit = bool(inherit_cfg)
    if previous_stage_root is None or not use_inherit:
        stage_rows = _independent_stage_override_maps(stage_cfg)
    else:
        stage_rows = _expand_dependent_stage(stage_cfg=stage_cfg, previous_stage_root=previous_stage_root)

    out: list[dict[str, Any]] = []
    row_idx = 0
    budget = str(stage_cfg.get("budget", "")).strip()
    seeds = [str(seed) for seed in _to_list(stage_cfg.get("seeds"))]
    if not budget or not seeds:
        raise ValueError(f"Stage '{stage_name}' requires non-empty budget and seeds.")

    for seed in seeds:
        for stage_maps in stage_rows:
            _apply_derive_and_constraints(
                workflow_stage_cfg=stage_cfg,
                workflow_wrapper_cfg=workflow_cfg,
                budget=budget,
                stage_maps=stage_maps,
                repo_root=repo_root,
            )
            payload = {
                "method": method,
                "preset": preset,
                "budget": budget,
                "seed": seed,
                "stage1": stage_maps["stage1"],
                "stage2": stage_maps["stage2"],
                "stage3": stage_maps["stage3"],
            }
            cfg_id = f"cfg_{row_idx:05d}_{cfg_hash(payload)}"
            out.append(
                {
                    "row_idx": row_idx,
                    "cfg_id": cfg_id,
                    "method": method,
                    "preset": preset,
                    "budget": budget,
                    "seed": seed,
                    "experiment_id": f"{experiment_prefix}_{stage_name}",
                    "model_flag": model_flag,
                    "skip_preprocess": str(bool(execution["skip_preprocess"])).lower(),
                    "skip_baseline": str(bool(execution["skip_baseline"])).lower(),
                    "baseline_epochs": ""
                    if execution["baseline_epochs"] is None
                    else str(int(float(execution["baseline_epochs"]))),
                    "run_root": str((runs_root / cfg_id).resolve()),
                    "stage1_overrides": serialize_overrides(stage_maps["stage1"]),
                    "stage2_overrides": serialize_overrides(stage_maps["stage2"]),
                    "stage3_overrides": serialize_overrides(stage_maps["stage3"]),
                }
            )
            row_idx += 1
    return out


def _write_stage_inputs(
    *,
    stage_root: Path,
    stage_name: str,
    stage_cfg: dict[str, Any],
    rows: list[dict[str, Any]],
    previous_stage: str | None,
) -> None:
    _write_json(
        stage_root / "stage_input.json",
        {
            "stage": stage_name,
            "previous_stage": previous_stage or "",
            "resolved_at_utc": _utc_now(),
            "stage_config": stage_cfg,
            "total_rows": len(rows),
        },
    )


def _prune_run_root(
    *,
    run_root: Path,
    prune_row_data: bool,
    prune_row_qbc_artifacts: bool,
) -> None:
    if prune_row_data:
        data_root = run_root / "data"
        if data_root.exists():
            subprocess.run(["rm", "-rf", str(data_root)], check=False)
            print(f"[hpo-workflow] pruned raw data: {data_root}")
    if prune_row_qbc_artifacts:
        for path in (run_root / "qbc" / "rounds", run_root / "qbc" / "checkpoints"):
            if path.exists():
                subprocess.run(["rm", "-rf", str(path)], check=False)
                print(f"[hpo-workflow] pruned qbc artifact: {path}")


def _run_matrix_rows(
    *,
    matrix_path: Path,
    python_bin: str,
    resume: bool,
    max_rows: int | None,
    prune_row_data: bool,
    prune_row_qbc_artifacts: bool,
) -> None:
    with matrix_path.open("r", encoding="utf-8") as f:
        matrix_rows = list(csv.DictReader(f, delimiter="\t"))

    for row in matrix_rows:
        row_index = int(row["row_idx"])
        if max_rows is not None and row_index >= max_rows:
            break
        run_root = Path(row["run_root"])
        status_path = run_root / "hpo_status.json"
        if resume and status_path.exists():
            try:
                with status_path.open("r", encoding="utf-8") as f:
                    status = json.load(f)
                if int(status.get("return_code", 1)) == 0:
                    print(f"[hpo-workflow] skip completed row_idx={row_index} cfg_id={row['cfg_id']}")
                    _prune_run_root(
                        run_root=run_root,
                        prune_row_data=prune_row_data,
                        prune_row_qbc_artifacts=prune_row_qbc_artifacts,
                    )
                    continue
            except Exception:
                pass

        cmd = [
            python_bin,
            "tools/hpo_workflow/run_matrix_row.py",
            "--matrix",
            str(matrix_path),
            "--row-index",
            str(row_index),
            "--python-bin",
            python_bin,
        ]
        print(f"[hpo-workflow] run stage row row_idx={row_index} cfg_id={row['cfg_id']}")
        proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"Row execution failed for row_idx={row_index} cfg_id={row['cfg_id']}")
        _prune_run_root(
            run_root=run_root,
            prune_row_data=prune_row_data,
            prune_row_qbc_artifacts=prune_row_qbc_artifacts,
        )


def _stage_status_from_rows(
    *,
    stage_name: str,
    stage_root: Path,
    rows: list[dict[str, Any]],
    expected_rows: int,
    truncated_by_max_rows: bool,
) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", "")).strip() or "unknown"
        status_counts[status] = status_counts.get(status, 0) + 1
    completed_rows = sum(1 for row in rows if str(row.get("status", "")) == "completed")
    failed_rows = expected_rows - completed_rows
    is_complete = completed_rows == expected_rows and expected_rows > 0
    payload = {
        "stage": stage_name,
        "stage_root": str(stage_root),
        "expected_rows": expected_rows,
        "observed_rows": len(rows),
        "completed_rows": completed_rows,
        "failed_or_incomplete_rows": failed_rows,
        "status_counts": status_counts,
        "truncated_by_max_rows": truncated_by_max_rows,
        "is_complete": is_complete,
    }
    return payload


def _collect_and_rank_stage(
    *,
    workflow_cfg: dict[str, Any],
    stage_name: str,
    stage_cfg: dict[str, Any],
    stage_root: Path,
) -> dict[str, Any]:
    workflow = _to_dict(workflow_cfg.get("workflow", {}))
    objective_metric, objective_direction, tie_breakers, shortlist_size = _resolve_objective(
        stage_cfg,
        "",
        "",
    )
    rows = collect_stage_results(
        stage_root=stage_root,
        workflow_id=str(workflow.get("id", "")),
        workflow_version=str(workflow.get("version", "")),
        method=str(workflow.get("method", "")),
        model_flag=str(workflow.get("model_flag", "")),
        preset=str(workflow.get("preset", "default")),
        stage=stage_name,
        stage_role=str(stage_cfg.get("role", "")),
        objective_metric=objective_metric,
        objective_direction=objective_direction,
    )
    write_stage_results_csv(rows, stage_root / "stage_results.csv")
    stage_status = _stage_status_from_rows(
        stage_name=stage_name,
        stage_root=stage_root,
        rows=rows,
        expected_rows=sum(1 for _ in rows),
        truncated_by_max_rows=False,
    )
    _write_json(stage_root / "stage_status.json", stage_status)

    ranking_cfg = _to_dict(stage_cfg.get("ranking", {}))
    group_by = [str(x) for x in _to_list(ranking_cfg.get("group_by"))]
    if not group_by:
        raise ValueError(f"Stage '{stage_name}' is missing ranking.group_by.")

    aggregated = aggregate_stage_results(
        rows=rows,
        group_by=group_by,
        objective_metric=objective_metric,
        objective_direction=objective_direction,
    )
    ranked = rank_aggregated_results(
        rows=aggregated,
        objective_direction=objective_direction,
        tie_breakers=tie_breakers,
        shortlist_size=shortlist_size,
    )
    write_ranked_csv(ranked, stage_root / "stage_ranked.csv", group_by)
    write_winner_json(
        ranked_rows=ranked,
        out_path=stage_root / "stage_winner.json",
        objective_direction=objective_direction,
        tie_breakers=tie_breakers,
        shortlist_size=shortlist_size,
        group_by=group_by,
    )
    write_shortlist_json(
        ranked_rows=ranked,
        out_path=stage_root / "stage_shortlist.json",
        shortlist_size=shortlist_size,
        group_by=group_by,
    )
    with (stage_root / "stage_winner.json").open("r", encoding="utf-8") as f:
        winner_payload = json.load(f)
    return {
        "stage_status": stage_status,
        "objective_metric": objective_metric,
        "objective_direction": objective_direction,
        "grouped_configs": len(ranked),
        "winner": ranked[0] if ranked else {},
        "group_by": group_by,
        "stage_results_rows": rows,
        "ranked_rows": ranked,
        "winner_payload": winner_payload,
    }


def _stage_enabled(stage_cfg: dict[str, Any]) -> bool:
    return bool(stage_cfg.get("enabled", True))


def _stage_sequence(stages: dict[str, Any], from_stage: str | None, to_stage: str | None) -> list[str]:
    names = [_normalize_stage_name(name) for name in stages.keys()]
    if from_stage:
        start_idx = names.index(from_stage)
    else:
        start_idx = 0
    if to_stage:
        end_idx = names.index(to_stage)
    else:
        end_idx = len(names) - 1
    return names[start_idx : end_idx + 1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to one workflow YAML config.")
    parser.add_argument("--python-bin", default=sys.executable, help="Python executable for row execution.")
    parser.add_argument("--output-root", default=None, help="Optional override of workflow.output_root.")
    parser.add_argument("--no-timestamp", action="store_true", help="Do not append a timestamp to the workflow root.")
    parser.add_argument("--from-stage", default=None, help="Optional first stage to execute.")
    parser.add_argument("--to-stage", default=None, help="Optional last stage to execute.")
    parser.add_argument("--resume", action="store_true", help="Skip rows with successful hpo_status.json.")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional per-stage row limit for local debugging. Incomplete stages do not rank or hand off.",
    )
    parser.add_argument("--plan-only", action="store_true", help="Materialize workflow inputs and matrix files without executing rows.")
    parser.add_argument(
        "--no-prune-row-data",
        action="store_true",
        help="Do not prune run_root/data after successful row execution.",
    )
    parser.add_argument(
        "--no-prune-row-qbc-artifacts",
        action="store_true",
        help="Do not prune run_root/qbc/{rounds,checkpoints} after successful row execution.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_path = Path(args.config).resolve()
    workflow_cfg = _load_workflow_config(config_path)
    workflow = _to_dict(workflow_cfg.get("workflow", {}))
    stages = _to_dict(workflow_cfg.get("stages", {}))
    workflow_id = str(workflow.get("id", config_path.stem))
    output_root = str(args.output_root or workflow.get("output_root", "outputs/hpo_workflows"))
    workflow_root = _workflow_root(output_root, workflow_id, with_timestamp=not args.no_timestamp).resolve()
    workflow_root.mkdir(parents=True, exist_ok=True)
    summary_root = REPO_ROOT / "results" / "hpo"
    summary_root.mkdir(parents=True, exist_ok=True)
    hpo_summary_path = summary_root / "hpo_summary.csv"
    hpo_winners_path = summary_root / "hpo_winners.csv"

    manifest = {
        "workflow_id": workflow_id,
        "workflow_version": workflow.get("version", ""),
        "config_path": str(config_path),
        "created_at_utc": _utc_now(),
        "workflow_root": str(workflow_root),
        "stages": {},
    }
    _write_json(workflow_root / "workflow_manifest.json", manifest)

    previous_stage_root: Path | None = None
    stage_names = _stage_sequence(stages, args.from_stage, args.to_stage)
    for stage_name in stage_names:
        stage_cfg = _to_dict(stages[stage_name])
        if not _stage_enabled(stage_cfg):
            continue
        depends_on = str(stage_cfg.get("depends_on", "")).strip() or None
        if depends_on:
            previous_stage_root = workflow_root / depends_on
            if not previous_stage_root.exists():
                raise FileNotFoundError(
                    f"Stage '{stage_name}' depends on '{depends_on}', but {previous_stage_root} does not exist."
                )
        stage_root = workflow_root / stage_name
        stage_root.mkdir(parents=True, exist_ok=True)

        print(f"[hpo-workflow] materialize stage={stage_name} root={stage_root}")
        rows = _build_stage_rows(
            workflow_cfg=workflow_cfg,
            workflow_root=workflow_root,
            stage_name=stage_name,
            stage_cfg=stage_cfg,
            previous_stage_root=previous_stage_root,
        )
        _write_matrix(rows, stage_root / "matrix.tsv")
        _write_stage_inputs(
            stage_root=stage_root,
            stage_name=stage_name,
            stage_cfg=stage_cfg,
            rows=rows,
            previous_stage=depends_on,
        )

        manifest["stages"][stage_name] = {
            "stage_root": str(stage_root),
            "depends_on": depends_on or "",
            "total_rows": len(rows),
            "status": "materialized" if args.plan_only else "pending",
        }
        _write_json(workflow_root / "workflow_manifest.json", manifest)

        if args.plan_only:
            previous_stage_root = stage_root
            continue

        _run_matrix_rows(
            matrix_path=stage_root / "matrix.tsv",
            python_bin=args.python_bin,
            resume=args.resume,
            max_rows=args.max_rows,
            prune_row_data=not args.no_prune_row_data,
            prune_row_qbc_artifacts=not args.no_prune_row_qbc_artifacts,
        )

        objective_metric, objective_direction, _, _ = _resolve_objective(stage_cfg, "", "")
        stage_rows = collect_stage_results(
            stage_root=stage_root,
            workflow_id=str(workflow.get("id", "")),
            workflow_version=str(workflow.get("version", "")),
            method=str(workflow.get("method", "")),
            model_flag=str(workflow.get("model_flag", "")),
            preset=str(workflow.get("preset", "default")),
            stage=stage_name,
            stage_role=str(stage_cfg.get("role", "")),
            objective_metric=objective_metric,
            objective_direction=objective_direction,
        )
        write_stage_results_csv(stage_rows, stage_root / "stage_results.csv")
        stage_status = _stage_status_from_rows(
            stage_name=stage_name,
            stage_root=stage_root,
            rows=stage_rows,
            expected_rows=len(rows),
            truncated_by_max_rows=args.max_rows is not None and args.max_rows < len(rows),
        )
        _write_json(stage_root / "stage_status.json", stage_status)
        manifest["stages"][stage_name]["stage_status"] = stage_status

        if not stage_status["is_complete"]:
            manifest["stages"][stage_name]["status"] = "incomplete"
            _write_json(workflow_root / "workflow_manifest.json", manifest)
            raise RuntimeError(
                f"Stage '{stage_name}' did not complete successfully "
                f"({stage_status['completed_rows']}/{stage_status['expected_rows']} rows completed). "
                "No ranking or downstream stage handoff was produced."
            )

        summary = _collect_and_rank_stage(
            workflow_cfg=workflow_cfg,
            stage_name=stage_name,
            stage_cfg=stage_cfg,
            stage_root=stage_root,
        )
        annotated_rows = annotate_stage_results(
            stage_results_rows=summary["stage_results_rows"],
            ranked_rows=summary["ranked_rows"],
            group_by=summary["group_by"],
        )
        update_hpo_summary(
            summary_path=hpo_summary_path,
            annotated_rows=annotated_rows,
        )
        update_hpo_winners(
            winners_path=hpo_winners_path,
            winner_row=flatten_winner_payload(summary["winner_payload"]),
        )
        manifest_summary = {
            "objective_metric": summary["objective_metric"],
            "objective_direction": summary["objective_direction"],
            "grouped_configs": summary["grouped_configs"],
            "winner": summary["winner"],
        }
        manifest["stages"][stage_name]["status"] = "completed"
        manifest["stages"][stage_name]["stage_status"] = summary["stage_status"]
        manifest["stages"][stage_name]["summary"] = manifest_summary
        _write_json(workflow_root / "workflow_manifest.json", manifest)
        previous_stage_root = stage_root

    print(f"[hpo-workflow] workflow_root={workflow_root}")


if __name__ == "__main__":
    main()
