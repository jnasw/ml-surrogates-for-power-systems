"""Manifest utilities for end-to-end experiment pipeline runs."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_manifest(
    *,
    run_id: str,
    run_root: str,
    method: str,
    budget: str,
    seed_label: str,
    seed_value: int,
    preset: str,
    experiment_id: str,
    model_flag: str,
    git_commit: str | None,
    stages_enabled: dict[str, bool],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "created_at_utc": utc_now_iso(),
        "updated_at_utc": utc_now_iso(),
        "run_root": run_root,
        "experiment": {
            "id": experiment_id,
            "preset": preset,
            "method": method,
            "budget": budget,
            "seed_label": seed_label,
            "seed_value": seed_value,
            "model_flag": model_flag,
        },
        "git": {"commit": git_commit},
        "stages_enabled": stages_enabled,
        "stages": {},
        "artifacts": {},
    }


def save_manifest(path: str, manifest: dict[str, Any]) -> None:
    manifest["updated_at_utc"] = utc_now_iso()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def set_stage_status(
    manifest: dict[str, Any],
    *,
    stage: str,
    status: str,
    command: list[str] | None = None,
    log_file: str | None = None,
    started_at_utc: str | None = None,
    completed_at_utc: str | None = None,
    return_code: int | None = None,
    error: str | None = None,
) -> None:
    entry = manifest["stages"].get(stage, {})
    entry["status"] = status
    if command is not None:
        entry["command"] = command
    if log_file is not None:
        entry["log_file"] = log_file
    if started_at_utc is not None:
        entry["started_at_utc"] = started_at_utc
    if completed_at_utc is not None:
        entry["completed_at_utc"] = completed_at_utc
    if return_code is not None:
        entry["return_code"] = int(return_code)
    if error is not None:
        entry["error"] = error
    manifest["stages"][stage] = entry

