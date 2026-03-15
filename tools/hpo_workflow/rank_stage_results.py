#!/usr/bin/env python3
"""Rank HPO stage results, select a winner, and emit shortlist artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from omegaconf import OmegaConf


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if math.isnan(number):
        return None
    return number


def _to_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _hash_group(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(data).hexdigest()[:10]


def _mean(values: list[float | None]) -> float | None:
    nums = [v for v in values if v is not None]
    return mean(nums) if nums else None


def _std(values: list[float | None]) -> float | None:
    nums = [v for v in values if v is not None]
    if not nums:
        return None
    if len(nums) == 1:
        return 0.0
    return pstdev(nums)


def _min(values: list[float | None]) -> float | None:
    nums = [v for v in values if v is not None]
    return min(nums) if nums else None


def _max(values: list[float | None]) -> float | None:
    nums = [v for v in values if v is not None]
    return max(nums) if nums else None


def _stage_cfg_from_workflow(workflow_config: Path, stage_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = OmegaConf.to_container(OmegaConf.load(str(workflow_config)), resolve=True)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid workflow config: {workflow_config}")
    workflow_meta = cfg.get("workflow", {})
    stages = cfg.get("stages", {})
    if stage_name not in stages:
        raise KeyError(f"Stage '{stage_name}' not found in workflow config: {workflow_config}")
    stage_cfg = stages[stage_name]
    if not isinstance(stage_cfg, dict):
        raise ValueError(f"Invalid stage config for '{stage_name}' in {workflow_config}")
    return workflow_meta if isinstance(workflow_meta, dict) else {}, stage_cfg


def _resolve_group_by(
    *,
    workflow_config: Path | None,
    stage_name: str,
    cli_group_by: list[str],
) -> tuple[list[str], dict[str, Any], dict[str, Any]]:
    workflow_meta: dict[str, Any] = {}
    stage_cfg: dict[str, Any] = {}
    if workflow_config:
        workflow_meta, stage_cfg = _stage_cfg_from_workflow(workflow_config, stage_name)
    ranking_cfg = stage_cfg.get("ranking", {}) if isinstance(stage_cfg, dict) else {}
    cfg_group_by = ranking_cfg.get("group_by", []) if isinstance(ranking_cfg, dict) else []
    group_by = [str(x) for x in (cli_group_by or cfg_group_by)]
    if not group_by:
        raise ValueError("No grouping columns provided. Use --group-by or define stages.<stage>.ranking.group_by.")
    return group_by, workflow_meta, stage_cfg


def _resolve_objective(stage_cfg: dict[str, Any], objective_metric: str, objective_direction: str) -> tuple[str, str, list[str], int]:
    objective_cfg = stage_cfg.get("objective", {}) if isinstance(stage_cfg, dict) else {}
    shortlist_cfg = stage_cfg.get("shortlist", {}) if isinstance(stage_cfg, dict) else {}
    metric = objective_metric or str(objective_cfg.get("metric", "")).strip()
    direction = objective_direction or str(objective_cfg.get("direction", "")).strip()
    tie_breakers = [str(x) for x in objective_cfg.get("tie_breakers", [])] if isinstance(objective_cfg, dict) else []
    shortlist_size = int(shortlist_cfg.get("top_k", 1)) if isinstance(shortlist_cfg, dict) else 1
    if not metric:
        raise ValueError("Objective metric is required. Use --objective-metric or define it in the workflow config.")
    if direction not in {"minimize", "maximize"}:
        raise ValueError("Objective direction must be 'minimize' or 'maximize'.")
    return metric, direction, tie_breakers, shortlist_size


def _objective_from_row(row: dict[str, str], metric: str) -> float | None:
    if metric == "success":
        completed = _to_bool(row.get("completed"))
        if completed is None:
            return None
        return 1.0 if completed else 0.0
    value = _to_float(row.get("objective_value"))
    if value is not None:
        return value
    value = _to_float(row.get(metric))
    if value is not None:
        return value
    final_metric = f"final_{metric}"
    return _to_float(row.get(final_metric))


def _runtime_means(rows: list[dict[str, str]]) -> dict[str, float | None]:
    runtime_keys = [
        "final_round_seconds",
        "final_train_seconds",
        "final_candidate_generation_seconds",
        "final_candidate_simulation_seconds",
        "final_acquisition_seconds",
        "final_selected_simulation_seconds",
        "final_eval_seconds",
    ]
    out: dict[str, float | None] = {}
    for key in runtime_keys:
        out[f"mean_{key}"] = _mean([_to_float(row.get(key)) for row in rows])
    return out


def _stable_context(rows: list[dict[str, str]], key: str) -> str:
    values = [str(row.get(key, "")).strip() for row in rows]
    values = [v for v in values if v]
    if not values:
        return ""
    return values[0]


def aggregate_stage_results(
    *,
    rows: list[dict[str, str]],
    group_by: list[str],
    objective_metric: str,
    objective_direction: str,
) -> list[dict[str, Any]]:
    missing_cols = [col for col in group_by if col not in rows[0]]
    if missing_cols:
        raise KeyError(f"Grouping columns missing from stage results: {missing_cols}")

    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = tuple(str(row.get(col, "")).strip() for col in group_by)
        grouped[key].append(row)

    out: list[dict[str, Any]] = []
    for key, members in grouped.items():
        objective_values = [_objective_from_row(row, objective_metric) for row in members]
        group_payload = {col: value for col, value in zip(group_by, key)}
        config_group_id = f"grp_{_hash_group(group_payload)}"
        dataset_seeds = sorted({str(row.get("dataset_seed", "")).strip() for row in members if str(row.get("dataset_seed", "")).strip()})
        member_cfg_ids = [str(row.get("cfg_id", "")).strip() for row in members if str(row.get("cfg_id", "")).strip()]
        member_statuses = sorted({str(row.get("status", "")).strip() for row in members if str(row.get("status", "")).strip()})
        completed_flags = [_to_bool(row.get("completed")) for row in members]

        rec: dict[str, Any] = {
            "workflow_id": _stable_context(members, "workflow_id"),
            "workflow_version": _stable_context(members, "workflow_version"),
            "method": _stable_context(members, "method"),
            "model_flag": _stable_context(members, "model_flag"),
            "preset": _stable_context(members, "preset"),
            "stage": _stable_context(members, "stage"),
            "stage_role": _stable_context(members, "stage_role"),
            "stage_root": _stable_context(members, "stage_root"),
            "budget": _stable_context(members, "budget"),
            "objective_metric": objective_metric,
            "objective_direction": objective_direction,
            "config_group_id": config_group_id,
            "n_dataset_seeds": len(dataset_seeds),
            "dataset_seeds": ";;".join(dataset_seeds),
            "member_cfg_ids": ";;".join(member_cfg_ids),
            "member_statuses": ";;".join(member_statuses),
            "all_completed": len([flag for flag in completed_flags if flag is not None]) == len(members)
            and all(flag is True for flag in completed_flags),
            "objective_mean": _mean(objective_values),
            "objective_std": _std(objective_values),
            "objective_min": _min(objective_values),
            "objective_max": _max(objective_values),
        }
        rec.update(_runtime_means(members))
        rec.update(group_payload)
        out.append(rec)

    return out


def _tie_break_value(row: dict[str, Any], name: str) -> float:
    if name == "std_over_dataset_seeds":
        value = row.get("objective_std")
    elif name.startswith("mean_"):
        value = row.get(name)
    else:
        value = row.get(name)
    num = _to_float(value)
    return num if num is not None else float("inf")


def rank_aggregated_results(
    *,
    rows: list[dict[str, Any]],
    objective_direction: str,
    tie_breakers: list[str],
    shortlist_size: int,
) -> list[dict[str, Any]]:
    def sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
        objective = _to_float(row.get("objective_mean"))
        if objective is None:
            primary = float("inf")
        elif objective_direction == "minimize":
            primary = objective
        else:
            primary = -objective
        return (primary, *[_tie_break_value(row, name) for name in tie_breakers], str(row.get("config_group_id", "")))

    ranked = sorted(rows, key=sort_key)
    for idx, row in enumerate(ranked, start=1):
        row["rank"] = idx
        row["is_winner"] = idx == 1
        row["is_shortlisted"] = idx <= shortlist_size
    return ranked


def _csv_fieldnames(rows: list[dict[str, Any]], group_by: list[str]) -> list[str]:
    priority = [
        "workflow_id",
        "workflow_version",
        "method",
        "model_flag",
        "preset",
        "stage",
        "stage_role",
        "stage_root",
        "budget",
        "objective_metric",
        "objective_direction",
        "config_group_id",
        "rank",
        "is_winner",
        "is_shortlisted",
        "n_dataset_seeds",
        "dataset_seeds",
        "member_cfg_ids",
        "member_statuses",
        "all_completed",
        "objective_mean",
        "objective_std",
        "objective_min",
        "objective_max",
        "mean_final_round_seconds",
        "mean_final_train_seconds",
        "mean_final_candidate_generation_seconds",
        "mean_final_candidate_simulation_seconds",
        "mean_final_acquisition_seconds",
        "mean_final_selected_simulation_seconds",
        "mean_final_eval_seconds",
    ]
    keys = set()
    for row in rows:
        keys.update(row.keys())
    ordered = [key for key in priority if key in keys]
    ordered.extend([key for key in group_by if key in keys and key not in ordered])
    remaining = sorted(keys - set(ordered))
    return ordered + remaining


def write_ranked_csv(rows: list[dict[str, Any]], out_path: Path, group_by: list[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _csv_fieldnames(rows, group_by)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _winner_hparams(row: dict[str, Any], group_by: list[str]) -> dict[str, Any]:
    return {key: row.get(key, "") for key in group_by}


def write_winner_json(
    *,
    ranked_rows: list[dict[str, Any]],
    out_path: Path,
    objective_direction: str,
    tie_breakers: list[str],
    shortlist_size: int,
    group_by: list[str],
) -> None:
    winner = ranked_rows[0]
    payload = {
        "workflow_id": winner.get("workflow_id", ""),
        "workflow_version": winner.get("workflow_version", ""),
        "method": winner.get("method", ""),
        "model_flag": winner.get("model_flag", ""),
        "preset": winner.get("preset", ""),
        "stage": winner.get("stage", ""),
        "stage_role": winner.get("stage_role", ""),
        "source_stage_root": winner.get("stage_root", ""),
        "winner_cfg_id": winner.get("config_group_id", ""),
        "winner_rank": winner.get("rank", 1),
        "objective_metric": winner.get("objective_metric", ""),
        "objective_direction": objective_direction,
        "objective_mean": winner.get("objective_mean"),
        "objective_std": winner.get("objective_std"),
        "n_dataset_seeds": winner.get("n_dataset_seeds"),
        "shortlist_size": shortlist_size,
        "tie_breakers": tie_breakers,
        "group_by": group_by,
        "member_cfg_ids": str(winner.get("member_cfg_ids", "")).split(";;") if winner.get("member_cfg_ids") else [],
        "dataset_seeds": str(winner.get("dataset_seeds", "")).split(";;") if winner.get("dataset_seeds") else [],
        "hyperparameters": _winner_hparams(winner, group_by),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def write_shortlist_json(
    *,
    ranked_rows: list[dict[str, Any]],
    out_path: Path,
    shortlist_size: int,
    group_by: list[str],
) -> None:
    shortlist = ranked_rows[:shortlist_size]
    first = shortlist[0] if shortlist else {}
    items = []
    for row in shortlist:
        items.append(
            {
                "rank": row.get("rank"),
                "cfg_id": row.get("config_group_id", ""),
                "objective_mean": row.get("objective_mean"),
                "objective_std": row.get("objective_std"),
                "member_cfg_ids": str(row.get("member_cfg_ids", "")).split(";;") if row.get("member_cfg_ids") else [],
                "dataset_seeds": str(row.get("dataset_seeds", "")).split(";;") if row.get("dataset_seeds") else [],
                "hyperparameters": _winner_hparams(row, group_by),
            }
        )

    payload = {
        "workflow_id": first.get("workflow_id", ""),
        "workflow_version": first.get("workflow_version", ""),
        "method": first.get("method", ""),
        "model_flag": first.get("model_flag", ""),
        "preset": first.get("preset", ""),
        "stage": first.get("stage", ""),
        "stage_role": first.get("stage_role", ""),
        "source_stage_root": first.get("stage_root", ""),
        "shortlist_size": shortlist_size,
        "group_by": group_by,
        "items": items,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-results", default=None, help="Path to stage_results.csv.")
    parser.add_argument("--stage-root", default=None, help="Path to stage root. Defaults stage-results to <stage-root>/stage_results.csv.")
    parser.add_argument("--workflow-config", default=None, help="Optional workflow config path.")
    parser.add_argument("--stage-name", default="", help="Stage name in the workflow config.")
    parser.add_argument("--group-by", action="append", default=[], help="Explicit config identity column. Repeatable.")
    parser.add_argument("--objective-metric", default="", help="Override objective metric.")
    parser.add_argument("--objective-direction", default="", help="Override objective direction.")
    parser.add_argument("--shortlist-size", type=int, default=None, help="Override shortlist size.")
    parser.add_argument("--ranked-out", default=None, help="CSV output path. Defaults to <stage-root>/stage_ranked.csv")
    parser.add_argument("--winner-out", default=None, help="JSON output path. Defaults to <stage-root>/stage_winner.json")
    parser.add_argument("--shortlist-out", default=None, help="JSON output path. Defaults to <stage-root>/stage_shortlist.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    stage_root = Path(args.stage_root).resolve() if args.stage_root else None
    if args.stage_results:
        stage_results_path = Path(args.stage_results).resolve()
    elif stage_root:
        stage_results_path = stage_root / "stage_results.csv"
    else:
        raise ValueError("Either --stage-results or --stage-root is required.")

    if not stage_results_path.exists():
        raise FileNotFoundError(f"stage_results.csv not found: {stage_results_path}")

    if stage_root is None:
        stage_root = stage_results_path.parent

    workflow_config = Path(args.workflow_config).resolve() if args.workflow_config else None
    stage_name = args.stage_name or stage_root.name
    group_by, _, stage_cfg = _resolve_group_by(
        workflow_config=workflow_config,
        stage_name=stage_name,
        cli_group_by=args.group_by,
    )
    objective_metric, objective_direction, tie_breakers, shortlist_size = _resolve_objective(
        stage_cfg,
        args.objective_metric,
        args.objective_direction,
    )
    if args.shortlist_size is not None:
        shortlist_size = int(args.shortlist_size)

    rows = _read_csv(stage_results_path)
    if not rows:
        raise ValueError(f"No rows found in {stage_results_path}")

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

    ranked_out = Path(args.ranked_out).resolve() if args.ranked_out else stage_root / "stage_ranked.csv"
    winner_out = Path(args.winner_out).resolve() if args.winner_out else stage_root / "stage_winner.json"
    shortlist_out = Path(args.shortlist_out).resolve() if args.shortlist_out else stage_root / "stage_shortlist.json"

    write_ranked_csv(ranked, ranked_out, group_by)
    write_winner_json(
        ranked_rows=ranked,
        out_path=winner_out,
        objective_direction=objective_direction,
        tie_breakers=tie_breakers,
        shortlist_size=shortlist_size,
        group_by=group_by,
    )
    write_shortlist_json(
        ranked_rows=ranked,
        out_path=shortlist_out,
        shortlist_size=shortlist_size,
        group_by=group_by,
    )

    print(f"[hpo-workflow] stage_results={stage_results_path}")
    print(f"[hpo-workflow] grouped_configs={len(ranked)}")
    print(f"[hpo-workflow] ranked_out={ranked_out}")
    print(f"[hpo-workflow] winner_out={winner_out}")
    print(f"[hpo-workflow] shortlist_out={shortlist_out}")


if __name__ == "__main__":
    main()
