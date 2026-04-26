"""Shared helpers for experiment launcher orchestration."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from time import monotonic
from typing import Any

from src.experiments.pipeline.helpers.manifest import save_manifest, set_stage_status, utc_now_iso


def init_experiment_manifest(*, run_root: str, experiment: dict[str, Any]) -> dict[str, Any]:
    """Create the common manifest scaffold used by launcher scripts."""
    return {
        "created_at_utc": utc_now_iso(),
        "updated_at_utc": utc_now_iso(),
        "run_root": run_root,
        "experiment": dict(experiment),
        "stages": {},
        "artifacts": {},
        "runs": [],
    }


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def tag_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def format_hydra_list(items: list[str]) -> str:
    return "[" + ",".join(items) + "]"


def parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def build_dataset_pipeline_command(
    *,
    python_bin: str,
    method: str,
    experiment_id: str,
    preset: str,
    budget: str,
    dataset_seed: str,
    model_flag: str,
    run_root: Path,
    stage1_overrides: list[str],
    stage2_overrides: list[str],
) -> tuple[list[str], Path]:
    dataset_run_root = run_root / "dataset_pipeline"
    command = [
        python_bin,
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
        str(dataset_run_root),
        "--skip-baseline",
    ]
    for override in stage1_overrides:
        command.extend(["--stage1-override", override])
    for override in stage2_overrides:
        command.extend(["--stage2-override", override])
    return command, dataset_run_root


def dataset_root_from_manifest(dataset_run_root: Path, *, dry_run: bool, model_flag: str) -> Path:
    if dry_run:
        return dataset_run_root / "data" / model_flag / "dataset_v1"
    manifest_path = dataset_run_root / "dataset_manifest.json"
    manifest = read_json(manifest_path)
    artifacts = dict(manifest.get("artifacts", {}))
    dataset_root = artifacts.get("preprocessed_root") or artifacts.get("dataset_root")
    if not dataset_root:
        raise RuntimeError(f"Dataset manifest does not contain a dataset root: {manifest_path}")
    return Path(str(dataset_root))


def run_logged_stage(
    *,
    label: str,
    stage_name: str,
    command: list[str],
    log_path: Path,
    manifest: dict[str, Any],
    manifest_path: Path,
    dry_run: bool,
    cwd: Path,
    extra_env: dict[str, str] | None = None,
) -> int:
    print(f"[{label}] command:")
    print(" ".join(command))
    set_stage_status(
        manifest,
        stage=stage_name,
        status="running" if not dry_run else "dry_run",
        command=command,
        log_file=str(log_path),
        started_at_utc=utc_now_iso(),
    )
    save_manifest(str(manifest_path), manifest)

    if dry_run:
        set_stage_status(
            manifest,
            stage=stage_name,
            status="dry_run",
            completed_at_utc=utc_now_iso(),
            return_code=0,
        )
        save_manifest(str(manifest_path), manifest)
        return 0

    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    stage_started = monotonic()
    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.run(command, cwd=cwd, text=True, check=False, env=env, stdout=logf, stderr=subprocess.STDOUT)

    status = "completed" if proc.returncode == 0 else "failed"
    extra = {"elapsed_seconds": monotonic() - stage_started}
    set_stage_status(
        manifest,
        stage=stage_name,
        status=status,
        completed_at_utc=utc_now_iso(),
        return_code=proc.returncode,
        error=None if proc.returncode == 0 else f"Stage '{stage_name}' failed. See {log_path}",
        extra=extra,
    )
    save_manifest(str(manifest_path), manifest)
    return int(proc.returncode)


def upsert_run_status(
    manifest: dict[str, Any],
    *,
    run_name: str,
    run_dir: str,
    run_metadata: dict[str, Any],
    status: str,
    command: list[str] | None = None,
    log_file: str | None = None,
    started_at_utc: str | None = None,
    completed_at_utc: str | None = None,
    return_code: int | None = None,
    error: str | None = None,
    metrics_summary: dict[str, Any] | None = None,
    extra_fields: dict[str, Any] | None = None,
) -> None:
    existing = None
    for item in manifest["runs"]:
        if item.get("run_name") == run_name:
            existing = item
            break
    if existing is None:
        existing = {
            "run_name": run_name,
            "run_dir": run_dir,
            **dict(run_metadata),
        }
        manifest["runs"].append(existing)
    existing["status"] = status
    if command is not None:
        existing["command"] = command
    if log_file is not None:
        existing["log_file"] = log_file
    if started_at_utc is not None:
        existing["started_at_utc"] = started_at_utc
    if completed_at_utc is not None:
        existing["completed_at_utc"] = completed_at_utc
    if return_code is not None:
        existing["return_code"] = int(return_code)
    if error is not None:
        existing["error"] = error
    if metrics_summary is not None:
        existing["metrics_summary"] = metrics_summary
    if extra_fields:
        existing.update(dict(extra_fields))


def run_logged_command(
    *,
    label: str,
    command: list[str],
    log_path: Path,
    dry_run: bool,
    cwd: Path,
    extra_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    started_at = utc_now_iso()
    if dry_run:
        print(f"[{label}] command:")
        print(" ".join(command))
        completed_at = utc_now_iso()
        return {
            "status": "dry_run",
            "return_code": 0,
            "started_at_utc": started_at,
            "completed_at_utc": completed_at,
            "elapsed_seconds": 0.0,
        }

    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    started = monotonic()
    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.run(command, cwd=cwd, text=True, check=False, env=env, stdout=logf, stderr=subprocess.STDOUT)
    elapsed = monotonic() - started
    return {
        "status": "completed" if proc.returncode == 0 else "failed",
        "return_code": int(proc.returncode),
        "started_at_utc": started_at,
        "completed_at_utc": utc_now_iso(),
        "elapsed_seconds": elapsed,
        "error": None if proc.returncode == 0 else f"{label} failed. See {log_path}",
    }

