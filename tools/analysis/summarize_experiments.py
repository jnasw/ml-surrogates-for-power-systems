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
    baseline = artifacts.get("baseline_metrics_payload", {})
    return {
        "run_id": manifest.get("run_id"),
        "run_root": manifest.get("run_root"),
        "method": exp.get("method"),
        "budget": exp.get("budget"),
        "seed_label": exp.get("seed_label"),
        "seed_value": exp.get("seed_value"),
        "model_flag": exp.get("model_flag"),
        "stage1_status": stages.get("stage1_create_dataset", {}).get("status"),
        "stage2_status": stages.get("stage2_preprocess", {}).get("status"),
        "stage3_status": stages.get("stage3_baseline", {}).get("status"),
        "dataset_root": artifacts.get("dataset_root"),
        "qbc_history": artifacts.get("qbc_history"),
        "baseline_metrics": artifacts.get("baseline_metrics"),
        "n_train": baseline.get("n_train"),
        "n_test": baseline.get("n_test"),
        "mse": baseline.get("mse"),
        "rmse": baseline.get("rmse"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize pipeline run manifests.")
    parser.add_argument(
        "--root",
        default="outputs/experiments",
        help="Root folder to scan for run_manifest.json files.",
    )
    parser.add_argument(
        "--out",
        default="outputs/experiments_summary.csv",
        help="CSV path for aggregated summary.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    manifests = sorted(root.glob("**/run_manifest.json"))
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
