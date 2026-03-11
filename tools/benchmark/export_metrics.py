"""Export compact dataset-benchmark metrics (MSE/RMSE) from a campaign manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_dataset_manifest(
    *,
    repo_root: Path,
    experiment_id: str,
    preset: str,
    method: str,
    budget: str,
    dataset_seed: str,
) -> Path:
    return (
        repo_root
        / "outputs"
        / "experiments"
        / experiment_id
        / preset
        / method
        / budget
        / dataset_seed
        / "dataset_manifest.json"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export per-run MSE/RMSE for dataset benchmark campaigns.")
    parser.add_argument("--campaign-manifest", required=True, help="Path to campaign_manifest.json")
    parser.add_argument(
        "--out",
        default=None,
        help="Output CSV path (default: alongside campaign manifest as dataset_benchmark_metrics.csv).",
    )
    args = parser.parse_args()

    campaign_manifest_path = Path(args.campaign_manifest).resolve()
    campaign = _read_json(campaign_manifest_path)
    campaign_meta = campaign.get("campaign", {})
    experiment_id = str(campaign_meta.get("experiment_id", "")).strip()
    preset = str(campaign_meta.get("preset", "")).strip()
    if not experiment_id or not preset:
        raise ValueError("campaign_manifest.json missing campaign.experiment_id or campaign.preset")

    repo_root = Path(__file__).resolve().parents[2]
    out_path = (
        Path(args.out).resolve()
        if args.out
        else campaign_manifest_path.parent / "dataset_benchmark_metrics.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    run_items = campaign.get("dataset_runs", campaign.get("runs", []))
    for run in run_items:
        method = str(run.get("method", "")).strip()
        budget = str(run.get("budget", "")).strip()
        dataset_seed = str(run.get("dataset_seed", "")).strip()
        status = str(run.get("status", ""))
        if not method or not budget or not dataset_seed:
            continue

        row: dict[str, Any] = {
            "method": method,
            "budget": budget,
            "dataset_seed": dataset_seed,
            "mse_mean": "",
            "mse_std": "",
            "rmse_mean": "",
            "rmse_std": "",
        }

        if status != "completed":
            rows.append(row)
            continue

        run_manifest_path = _resolve_dataset_manifest(
            repo_root=repo_root,
            experiment_id=experiment_id,
            preset=preset,
            method=method,
            budget=budget,
            dataset_seed=dataset_seed,
        )
        if not run_manifest_path.exists():
            rows.append(row)
            continue

        run_manifest = _read_json(run_manifest_path)
        metrics = run_manifest.get("baseline_summary", {})
        row["mse_mean"] = metrics.get("mse_mean", "")
        row["mse_std"] = metrics.get("mse_std", "")
        row["rmse_mean"] = metrics.get("rmse_mean", "")
        row["rmse_std"] = metrics.get("rmse_std", "")
        rows.append(row)

    rows.sort(key=lambda r: (r["method"], r["budget"], r["dataset_seed"]))
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method", "budget", "dataset_seed", "mse_mean", "mse_std", "rmse_mean", "rmse_std"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
