"""Recover benchmark runtime metadata from archived experiment manifests.

This script combines:

- campaign-level benchmark manifests under results/benchmark/campaigns
- archived dataset manifests under results/archive/experiments_2

and writes one CSV with per-dataset-run timing information. When campaign
runtime is missing because the run was reused via `skipped_existing`, the
script reconstructs runtime from the archived experiment manifest timestamps.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


def _duration_seconds(started_at: str | None, completed_at: str | None) -> float | None:
    start_dt = _parse_iso(started_at)
    end_dt = _parse_iso(completed_at)
    if start_dt is None or end_dt is None:
        return None
    return (end_dt - start_dt).total_seconds()


def _archive_manifest_path(
    archive_root: Path,
    *,
    experiment_id: str,
    preset: str,
    method: str,
    budget: str,
    dataset_seed: str,
) -> Path:
    return archive_root / experiment_id / preset / method / budget / dataset_seed / "dataset_manifest.json"


def _baseline_duration_stats(manifest: dict[str, Any]) -> dict[str, Any]:
    durations: list[float] = []
    for baseline_run in dict(manifest.get("baseline_runs", {})).values():
        dur = _duration_seconds(
            baseline_run.get("started_at_utc"),
            baseline_run.get("completed_at_utc"),
        )
        if dur is not None:
            durations.append(dur)

    if not durations:
        return {
            "baseline_subruns_with_runtime": 0,
            "baseline_total_runtime_seconds": None,
            "baseline_mean_runtime_seconds": None,
            "baseline_min_runtime_seconds": None,
            "baseline_max_runtime_seconds": None,
        }

    return {
        "baseline_subruns_with_runtime": len(durations),
        "baseline_total_runtime_seconds": sum(durations),
        "baseline_mean_runtime_seconds": sum(durations) / len(durations),
        "baseline_min_runtime_seconds": min(durations),
        "baseline_max_runtime_seconds": max(durations),
    }


def _stage_runtime_columns(manifest: dict[str, Any]) -> dict[str, Any]:
    stages = dict(manifest.get("stages", {}))
    out: dict[str, Any] = {}
    ordered_stage_names = (
        "stage1_create_dataset",
        "stage2_preprocess",
        "stage3_baseline",
    )
    for stage_name in ordered_stage_names:
        stage = dict(stages.get(stage_name, {}))
        out[f"{stage_name}_status"] = stage.get("status")
        out[f"{stage_name}_runtime_seconds"] = _duration_seconds(
            stage.get("started_at_utc"),
            stage.get("completed_at_utc"),
        )
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaigns-root",
        default="results/benchmark/campaigns",
        help="Campaign manifests root.",
    )
    parser.add_argument(
        "--archive-root",
        default="results/archive/experiments_2",
        help="Archived experiment manifests root.",
    )
    parser.add_argument(
        "--out",
        default="results/benchmark/tables/benchmark_runtime_recovered.csv",
        help="Output CSV path.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    campaigns_root = Path(args.campaigns_root).resolve()
    archive_root = Path(args.archive_root).resolve()
    out_path = Path(args.out).resolve()

    if not campaigns_root.exists():
        raise FileNotFoundError(f"Campaigns root does not exist: {campaigns_root}")
    if not archive_root.exists():
        raise FileNotFoundError(f"Archive root does not exist: {archive_root}")

    rows: list[dict[str, Any]] = []

    for manifest_path in sorted(campaigns_root.glob("benchmark_*/campaign_manifest.json")):
        campaign = _read_json(manifest_path)
        campaign_meta = dict(campaign.get("campaign", {}))
        campaign_dir = manifest_path.parent.name
        campaign_name = str(campaign_meta.get("name", campaign_dir))
        experiment_id = str(campaign_meta.get("experiment_id", "")).strip()
        preset = str(campaign_meta.get("preset", "")).strip()
        model_flag = str(campaign_meta.get("model_flag", "")).strip()

        for run in campaign.get("dataset_runs", []):
            method = str(run.get("method", "")).strip()
            budget = str(run.get("budget", "")).strip()
            dataset_seed = str(run.get("dataset_seed", "")).strip()
            status = str(run.get("status", "")).strip()
            if not method or not budget or not dataset_seed:
                continue

            archive_path = _archive_manifest_path(
                archive_root,
                experiment_id=experiment_id,
                preset=preset,
                method=method,
                budget=budget,
                dataset_seed=dataset_seed,
            )
            archive_manifest = _read_json(archive_path) if archive_path.exists() else {}

            campaign_runtime = run.get("duration_seconds")
            archive_total_runtime = None
            if archive_manifest:
                archive_total_runtime = _duration_seconds(
                    archive_manifest.get("created_at_utc"),
                    archive_manifest.get("updated_at_utc"),
                )
                if archive_total_runtime is None:
                    stage_runtimes = _stage_runtime_columns(archive_manifest)
                    stage_values = [
                        stage_runtimes["stage1_create_dataset_runtime_seconds"],
                        stage_runtimes["stage2_preprocess_runtime_seconds"],
                        stage_runtimes["stage3_baseline_runtime_seconds"],
                    ]
                    valid_stage_values = [float(v) for v in stage_values if v is not None]
                    if valid_stage_values:
                        archive_total_runtime = sum(valid_stage_values)
            else:
                stage_runtimes = {}

            runtime_source = ""
            total_runtime_seconds = None
            if isinstance(campaign_runtime, (int, float)):
                runtime_source = "campaign_manifest"
                total_runtime_seconds = float(campaign_runtime)
            elif archive_total_runtime is not None:
                runtime_source = "archive_manifest"
                total_runtime_seconds = float(archive_total_runtime)

            if archive_manifest:
                stage_runtimes = _stage_runtime_columns(archive_manifest)
                baseline_stats = _baseline_duration_stats(archive_manifest)
            else:
                stage_runtimes = {
                    "stage1_create_dataset_status": None,
                    "stage1_create_dataset_runtime_seconds": None,
                    "stage2_preprocess_status": None,
                    "stage2_preprocess_runtime_seconds": None,
                    "stage3_baseline_status": None,
                    "stage3_baseline_runtime_seconds": None,
                }
                baseline_stats = {
                    "baseline_subruns_with_runtime": 0,
                    "baseline_total_runtime_seconds": None,
                    "baseline_mean_runtime_seconds": None,
                    "baseline_min_runtime_seconds": None,
                    "baseline_max_runtime_seconds": None,
                }

            rows.append(
                {
                    "campaign_dir": campaign_dir,
                    "campaign_name": campaign_name,
                    "experiment_id": experiment_id,
                    "preset": preset,
                    "model_flag": model_flag,
                    "method": method,
                    "budget": budget,
                    "dataset_seed": dataset_seed,
                    "campaign_status": status,
                    "runtime_source": runtime_source,
                    "total_runtime_seconds": total_runtime_seconds,
                    "campaign_runtime_seconds": float(campaign_runtime)
                    if isinstance(campaign_runtime, (int, float))
                    else None,
                    "archive_runtime_seconds": archive_total_runtime,
                    "archive_manifest_path": str(archive_path) if archive_path.exists() else "",
                    **stage_runtimes,
                    **baseline_stats,
                }
            )

    rows.sort(key=lambda r: (r["campaign_name"], r["method"], r["budget"], r["dataset_seed"]))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    recovered = sum(1 for row in rows if row["runtime_source"] == "archive_manifest")
    missing = sum(1 for row in rows if row["total_runtime_seconds"] is None)
    print(f"Wrote {len(rows)} rows to {out_path}")
    print(f"Recovered runtime from archive for {recovered} runs")
    print(f"Runs still missing runtime: {missing}")


if __name__ == "__main__":
    main()
