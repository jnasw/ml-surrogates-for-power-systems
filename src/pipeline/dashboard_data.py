"""Load run-level and round-level tables for dashboards."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def discover_manifests(root: str) -> list[str]:
    return sorted(str(p) for p in Path(root).resolve().glob("**/run_manifest.json"))


def _read_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_csv(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_run_table(root: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for manifest_path in discover_manifests(root):
        m = _read_json(manifest_path)
        exp = m.get("experiment", {})
        art = m.get("artifacts", {})
        base = art.get("baseline_metrics_payload", {})
        rows.append(
            {
                "run_id": m.get("run_id"),
                "run_root": m.get("run_root"),
                "method": exp.get("method"),
                "budget": exp.get("budget"),
                "seed_label": exp.get("seed_label"),
                "seed_value": exp.get("seed_value"),
                "phase": exp.get("phase"),
                "model_flag": exp.get("model_flag"),
                "dataset_root": art.get("dataset_root"),
                "qbc_history": art.get("qbc_history"),
                "round_telemetry_csv": art.get("round_telemetry_csv"),
                "n_train": base.get("n_train"),
                "n_test": base.get("n_test"),
                "mse": base.get("mse"),
                "rmse": base.get("rmse"),
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
            row["phase"] = exp.get("phase")
            row["model_flag"] = exp.get("model_flag")
            row["manifest_path"] = manifest_path
            out.append(row)
    return out

