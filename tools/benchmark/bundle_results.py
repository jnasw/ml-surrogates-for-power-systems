"""Copy compact benchmark analysis artifacts from outputs/ into a tracked results/ tree."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaigns-root",
        default="outputs/campaigns",
        help="Source benchmark campaigns root.",
    )
    parser.add_argument(
        "--experiments-root",
        default="outputs/experiments",
        help="Source benchmark experiments root.",
    )
    parser.add_argument(
        "--dst",
        default="results/benchmark",
        help="Destination tracked results root.",
    )
    parser.add_argument(
        "--campaign-glob",
        default="benchmark_*",
        help="Glob used to select campaign directories under --campaigns-root.",
    )
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _copy_file(src_path: Path, src_root: Path, dst_root: Path) -> int:
    rel_path = src_path.relative_to(src_root)
    dst_path = dst_root / rel_path
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_path)
    return src_path.stat().st_size


def _copy_if_exists(
    *,
    src_path: Path,
    src_root: Path,
    dst_root: Path,
    copied: set[Path],
) -> int:
    if not src_path.is_file() or src_path in copied:
        return 0
    copied.add(src_path)
    return _copy_file(src_path, src_root, dst_root)


def _copy_campaign_artifacts(
    *,
    campaign_dir: Path,
    campaigns_root: Path,
    dst_root: Path,
    copied: set[Path],
) -> tuple[int, int]:
    count = 0
    total_bytes = 0
    for name in ("campaign_manifest.json", "dataset_benchmark_metrics.csv"):
        src_path = campaign_dir / name
        copied_bytes = _copy_if_exists(
            src_path=src_path,
            src_root=campaigns_root,
            dst_root=dst_root / "campaigns",
            copied=copied,
        )
        if copied_bytes:
            count += 1
            total_bytes += copied_bytes
    return count, total_bytes


def _copy_run_artifacts(
    *,
    run_root: Path,
    experiments_root: Path,
    dst_root: Path,
    copied: set[Path],
) -> tuple[int, int]:
    count = 0
    total_bytes = 0
    direct_files = (
        run_root / "dataset_manifest.json",
        run_root / "baseline" / "summary.json",
        run_root / "telemetry" / "round_telemetry.csv",
        run_root / "qbc" / "history.jsonl",
        run_root / "qbc" / "config.yaml",
    )
    for src_path in direct_files:
        copied_bytes = _copy_if_exists(
            src_path=src_path,
            src_root=experiments_root,
            dst_root=dst_root / "experiments",
            copied=copied,
        )
        if copied_bytes:
            count += 1
            total_bytes += copied_bytes

    for metrics_path in sorted((run_root / "baseline").glob("bs*/metrics.json")):
        copied_bytes = _copy_if_exists(
            src_path=metrics_path,
            src_root=experiments_root,
            dst_root=dst_root / "experiments",
            copied=copied,
        )
        if copied_bytes:
            count += 1
            total_bytes += copied_bytes
    return count, total_bytes


def main() -> None:
    args = build_parser().parse_args()
    campaigns_root = Path(args.campaigns_root).resolve()
    experiments_root = Path(args.experiments_root).resolve()
    dst_root = Path(args.dst).resolve()

    if not campaigns_root.exists():
        raise FileNotFoundError(f"Source campaigns root does not exist: {campaigns_root}")
    if not experiments_root.exists():
        raise FileNotFoundError(f"Source experiments root does not exist: {experiments_root}")

    copied: set[Path] = set()
    copied_files = 0
    copied_bytes = 0
    matched_campaigns = 0

    for campaign_dir in sorted(campaigns_root.glob(args.campaign_glob)):
        if not campaign_dir.is_dir():
            continue
        campaign_manifest_path = campaign_dir / "campaign_manifest.json"
        if not campaign_manifest_path.is_file():
            continue
        matched_campaigns += 1

        count, total_bytes = _copy_campaign_artifacts(
            campaign_dir=campaign_dir,
            campaigns_root=campaigns_root,
            dst_root=dst_root,
            copied=copied,
        )
        copied_files += count
        copied_bytes += total_bytes

        campaign = _read_json(campaign_manifest_path)
        campaign_meta = campaign.get("campaign", {})
        experiment_id = str(campaign_meta.get("experiment_id", "")).strip()
        preset = str(campaign_meta.get("preset", "")).strip()
        if not experiment_id or not preset:
            continue

        run_items = campaign.get("dataset_runs", campaign.get("runs", []))
        for run in run_items:
            method = str(run.get("method", "")).strip()
            budget = str(run.get("budget", "")).strip()
            dataset_seed = str(run.get("dataset_seed", "")).strip()
            if not method or not budget or not dataset_seed:
                continue
            run_root = experiments_root / experiment_id / preset / method / budget / dataset_seed
            count, total_bytes = _copy_run_artifacts(
                run_root=run_root,
                experiments_root=experiments_root,
                dst_root=dst_root,
                copied=copied,
            )
            copied_files += count
            copied_bytes += total_bytes

    print(f"[benchmark-bundle] campaigns source: {campaigns_root}")
    print(f"[benchmark-bundle] experiments source: {experiments_root}")
    print(f"[benchmark-bundle] destination: {dst_root}")
    print(f"[benchmark-bundle] matched campaigns: {matched_campaigns}")
    print(f"[benchmark-bundle] copied files: {copied_files}")
    print(f"[benchmark-bundle] copied size: {copied_bytes / (1024 * 1024):.1f} MiB")


if __name__ == "__main__":
    main()
