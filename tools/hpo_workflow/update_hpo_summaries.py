#!/usr/bin/env python3
"""Update central HPO summary CSVs from workflow stage artifacts."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]], priority: list[str] | None = None) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = set()
    for row in rows:
        keys.update(row.keys())
    ordered = []
    if priority:
        ordered.extend([key for key in priority if key in keys])
    ordered.extend(sorted(keys - set(ordered)))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def _key_tuple(row: dict[str, Any], key_fields: list[str]) -> tuple[str, ...]:
    return tuple(str(row.get(field, "")).strip() for field in key_fields)


def _upsert_rows(
    *,
    existing_rows: list[dict[str, str]],
    new_rows: list[dict[str, Any]],
    key_fields: list[str],
) -> list[dict[str, Any]]:
    if not new_rows:
        return [dict(row) for row in existing_rows]
    replace_keys = {_key_tuple(row, key_fields) for row in new_rows}
    kept = [dict(row) for row in existing_rows if _key_tuple(row, key_fields) not in replace_keys]
    kept.extend(new_rows)
    return kept


def annotate_stage_results(
    *,
    stage_results_rows: list[dict[str, Any]],
    ranked_rows: list[dict[str, Any]],
    group_by: list[str],
) -> list[dict[str, Any]]:
    rank_lookup: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in ranked_rows:
        key = tuple(str(row.get(col, "")).strip() for col in group_by)
        rank_lookup[key] = row

    annotated: list[dict[str, Any]] = []
    for row in stage_results_rows:
        key = tuple(str(row.get(col, "")).strip() for col in group_by)
        ranked = rank_lookup.get(key, {})
        rec = dict(row)
        rec["config_group_id"] = ranked.get("config_group_id", "")
        rec["rank_within_stage"] = ranked.get("rank", "")
        rec["is_shortlisted"] = ranked.get("is_shortlisted", False)
        rec["is_stage_winner"] = ranked.get("is_winner", False)
        annotated.append(rec)
    return annotated


def flatten_winner_payload(payload: dict[str, Any]) -> dict[str, Any]:
    row = {
        "workflow_id": payload.get("workflow_id", ""),
        "workflow_version": payload.get("workflow_version", ""),
        "method": payload.get("method", ""),
        "model_flag": payload.get("model_flag", ""),
        "preset": payload.get("preset", ""),
        "stage": payload.get("stage", ""),
        "stage_role": payload.get("stage_role", ""),
        "winner_cfg_id": payload.get("winner_cfg_id", ""),
        "winner_rank": payload.get("winner_rank", ""),
        "source_stage_root": payload.get("source_stage_root", ""),
        "objective_metric": payload.get("objective_metric", ""),
        "objective_direction": payload.get("objective_direction", ""),
        "objective_mean": payload.get("objective_mean", ""),
        "objective_std": payload.get("objective_std", ""),
        "n_dataset_seeds": payload.get("n_dataset_seeds", ""),
        "shortlist_size": payload.get("shortlist_size", ""),
        "dataset_seeds": ";;".join(payload.get("dataset_seeds", []) or []),
        "member_cfg_ids": ";;".join(payload.get("member_cfg_ids", []) or []),
    }
    hyperparameters = payload.get("hyperparameters", {})
    if isinstance(hyperparameters, dict):
        for key, value in hyperparameters.items():
            row[str(key)] = value
    return row


def update_hpo_summary(
    *,
    summary_path: Path,
    annotated_rows: list[dict[str, Any]],
) -> None:
    existing = _read_csv(summary_path)
    merged = _upsert_rows(
        existing_rows=existing,
        new_rows=annotated_rows,
        key_fields=["workflow_id", "stage", "cfg_id"],
    )
    priority = [
        "workflow_id",
        "workflow_version",
        "method",
        "model_flag",
        "preset",
        "stage",
        "stage_role",
        "stage_root",
        "cfg_id",
        "config_group_id",
        "rank_within_stage",
        "is_shortlisted",
        "is_stage_winner",
        "dataset_seed",
        "budget",
        "status",
        "return_code",
        "completed",
        "objective_metric",
        "objective_direction",
        "objective_value",
        "final_eval_rmse",
        "final_eval_mse",
        "final_round_seconds",
        "final_train_seconds",
        "final_candidate_generation_seconds",
        "final_candidate_simulation_seconds",
        "final_acquisition_seconds",
        "run_root",
    ]
    _write_csv(summary_path, merged, priority=priority)


def update_hpo_winners(
    *,
    winners_path: Path,
    winner_row: dict[str, Any],
) -> None:
    existing = _read_csv(winners_path)
    merged = _upsert_rows(
        existing_rows=existing,
        new_rows=[winner_row],
        key_fields=["workflow_id", "stage"],
    )
    priority = [
        "workflow_id",
        "workflow_version",
        "method",
        "model_flag",
        "preset",
        "stage",
        "stage_role",
        "winner_cfg_id",
        "winner_rank",
        "source_stage_root",
        "objective_metric",
        "objective_direction",
        "objective_mean",
        "objective_std",
        "n_dataset_seeds",
        "shortlist_size",
        "dataset_seeds",
        "member_cfg_ids",
    ]
    _write_csv(winners_path, merged, priority=priority)
