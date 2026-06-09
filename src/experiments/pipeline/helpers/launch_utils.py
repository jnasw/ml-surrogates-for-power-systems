"""Shared helpers for experiment launcher orchestration."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from time import monotonic
from typing import Any

from src.experiments.pipeline.helpers.manifest import (
    init_experiment_manifest,
    save_manifest,
    set_stage_status,
    upsert_run_status,
    utc_now_iso,
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def tag_stamp() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d_%H%M%S")


def format_hydra_list(items: list[str]) -> str:
    return "[" + ",".join(items) + "]"


def parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _print_log_tail(log_path: Path, *, lines: int = 80) -> None:
    if not log_path.exists():
        print(f"[pipeline] log_tail_unavailable={log_path}", file=sys.stderr, flush=True)
        return
    try:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
    except OSError as exc:
        print(f"[pipeline] log_tail_error={log_path}: {exc}", file=sys.stderr, flush=True)
        return
    print(f"[pipeline] last_{lines}_log_lines={log_path}", file=sys.stderr, flush=True)
    for line in tail:
        print(line, file=sys.stderr, flush=True)


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
        "src.experiments.pipeline.run_dataset_pipeline",
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
    print(f"[{label}] starting {stage_name}; log_file={log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.run(command, cwd=cwd, text=True, check=False, env=env, stdout=logf, stderr=subprocess.STDOUT)

    status = "completed" if proc.returncode == 0 else "failed"
    if proc.returncode == 0:
        print(f"[{label}] completed {stage_name}; log_file={log_path}", flush=True)
    else:
        print(
            f"[{label}] failed {stage_name}; return_code={proc.returncode}; log_file={log_path}",
            file=sys.stderr,
            flush=True,
        )
        _print_log_tail(log_path)
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
    print(f"[{label}] starting; log_file={log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.run(command, cwd=cwd, text=True, check=False, env=env, stdout=logf, stderr=subprocess.STDOUT)
    elapsed = monotonic() - started
    if proc.returncode == 0:
        print(f"[{label}] completed; elapsed_seconds={elapsed:.3f}; log_file={log_path}", flush=True)
    else:
        print(
            f"[{label}] failed; return_code={proc.returncode}; elapsed_seconds={elapsed:.3f}; log_file={log_path}",
            file=sys.stderr,
            flush=True,
        )
        _print_log_tail(log_path)
    return {
        "status": "completed" if proc.returncode == 0 else "failed",
        "return_code": int(proc.returncode),
        "started_at_utc": started_at,
        "completed_at_utc": utc_now_iso(),
        "elapsed_seconds": elapsed,
        "error": None if proc.returncode == 0 else f"{label} failed. See {log_path}",
    }
