"""Aggregate run manifests into a single CSV-friendly table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _flatten_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    exp = manifest.get("experiment", {})
    stages = manifest.get("stages", {})
    artifacts = manifest.get("artifacts", {})
    baseline = manifest.get("baseline_summary", {})
    return {
        "dataset_id": manifest.get("dataset_id"),
        "run_root": manifest.get("run_root"),
        "method": exp.get("method"),
        "budget": exp.get("budget"),
        "dataset_seed_label": exp.get("dataset_seed_label"),
        "dataset_seed_value": exp.get("dataset_seed_value"),
        "model_flag": exp.get("model_flag"),
        "stage1_status": stages.get("stage1_create_dataset", {}).get("status"),
        "stage2_status": stages.get("stage2_preprocess", {}).get("status"),
        "stage3_status": stages.get("stage3_baseline", {}).get("status"),
        "dataset_root": artifacts.get("dataset_root"),
        "qbc_history": artifacts.get("qbc_history"),
        "baseline_summary": artifacts.get("baseline_summary"),
        "baseline_runs_count": baseline.get("n_runs"),
        "n_train": baseline.get("n_train"),
        "n_test": baseline.get("n_test"),
        "mse_mean": baseline.get("mse_mean"),
        "mse_std": baseline.get("mse_std"),
        "rmse_mean": baseline.get("rmse_mean"),
        "rmse_std": baseline.get("rmse_std"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize pipeline run manifests.")
    parser.add_argument(
        "--root",
        default="outputs/experiments",
        help="Root folder to scan for dataset_manifest.json files.",
    )
    parser.add_argument(
        "--out",
        default="outputs/experiments_summary.csv",
        help="CSV path for aggregated summary.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    manifests = sorted(root.glob("**/dataset_manifest.json"))
    rows: list[dict[str, Any]] = []
    for path in manifests:
        with path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        rows.append(_flatten_manifest(manifest))

    if not rows:
        print(f"No manifests found under: {root}")
        return

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} runs to {out_path}")


if __name__ == "__main__":
    main()
