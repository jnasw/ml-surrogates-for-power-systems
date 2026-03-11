"""Load dataset-, baseline-, and round-level tables from experiment manifests."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def discover_manifests(root: str) -> list[str]:
    return sorted(str(p) for p in Path(root).resolve().glob("**/dataset_manifest.json"))


def _read_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_csv(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _dataset_root_status(artifacts: dict[str, Any]) -> str | None:
    if artifacts.get("dataset_root_note"):
        return "pruned_after_run"
    return None


def load_dataset_table(root: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for manifest_path in discover_manifests(root):
        m = _read_json(manifest_path)
        exp = m.get("experiment", {})
        art = m.get("artifacts", {})
        base = m.get("baseline_summary", {})
        rows.append(
            {
                "dataset_id": m.get("dataset_id"),
                "run_root": m.get("run_root"),
                "method": exp.get("method"),
                "budget": exp.get("budget"),
                "dataset_seed_label": exp.get("dataset_seed_label"),
                "dataset_seed_value": exp.get("dataset_seed_value"),
                "preset": exp.get("preset", exp.get("phase")),
                "model_flag": exp.get("model_flag"),
                "dataset_root": art.get("dataset_root"),
                "dataset_root_status": _dataset_root_status(art),
                "qbc_history": art.get("qbc_history"),
                "round_telemetry_csv": art.get("round_telemetry_csv"),
                "baseline_summary_path": art.get("baseline_summary"),
                "baseline_runs_count": base.get("n_runs"),
                "n_train": base.get("n_train"),
                "n_test": base.get("n_test"),
                "mse_mean": base.get("mse_mean"),
                "mse_std": base.get("mse_std"),
                "rmse_mean": base.get("rmse_mean"),
                "rmse_std": base.get("rmse_std"),
                "manifest_path": manifest_path,
            }
        )
    return rows


def load_run_table(root: str) -> list[dict[str, Any]]:
    """Backward-compatible alias for dataset-level rows."""
    return load_dataset_table(root)


def load_baseline_table(root: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for manifest_path in discover_manifests(root):
        m = _read_json(manifest_path)
        exp = m.get("experiment", {})
        art = m.get("artifacts", {})
        for baseline_seed_label, baseline_run in sorted(dict(m.get("baseline_runs", {})).items()):
            metrics = dict(baseline_run.get("metrics_payload", {}))
            rows.append(
                {
                    "dataset_id": m.get("dataset_id"),
                    "run_root": m.get("run_root"),
                    "method": exp.get("method"),
                    "budget": exp.get("budget"),
                    "dataset_seed_label": exp.get("dataset_seed_label"),
                    "dataset_seed_value": exp.get("dataset_seed_value"),
                    "baseline_seed_label": baseline_run.get("baseline_seed_label", baseline_seed_label),
                    "baseline_seed_value": baseline_run.get("baseline_seed_value"),
                    "preset": exp.get("preset", exp.get("phase")),
                    "model_flag": exp.get("model_flag"),
                    "dataset_root": art.get("dataset_root"),
                    "dataset_root_status": _dataset_root_status(art),
                    "baseline_root": baseline_run.get("baseline_root"),
                    "baseline_status": baseline_run.get("status"),
                    "metrics_path": baseline_run.get("metrics_path"),
                    "n_train": metrics.get("n_train"),
                    "n_test": metrics.get("n_test"),
                    "mse": metrics.get("mse"),
                    "rmse": metrics.get("rmse"),
                    "manifest_path": manifest_path,
                }
            )
    return rows


def load_round_table(root: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for manifest_path in discover_manifests(root):
        m = _read_json(manifest_path)
        exp = m.get("experiment", {})
        art = m.get("artifacts", {})
        telemetry_csv = art.get("round_telemetry_csv")
        if not telemetry_csv:
            continue
        p = Path(str(telemetry_csv))
        if not p.exists():
            continue
        for row in _read_csv(str(p)):
            if "run_id" in row and "dataset_id" not in row:
                row["dataset_id"] = row.pop("run_id")
            if "seed_label" in row and "dataset_seed_label" not in row:
                row["dataset_seed_label"] = row["seed_label"]
            row.pop("seed_label", None)
            row["preset"] = exp.get("preset", exp.get("phase"))
            row["model_flag"] = exp.get("model_flag")
            row["dataset_seed_label"] = exp.get("dataset_seed_label")
            row["dataset_root_status"] = _dataset_root_status(art)
            row["manifest_path"] = manifest_path
            out.append(row)
    return out
