"""Shared summary helpers for experiment launchers."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

from src.experiments.pipeline.helpers.manifest import utc_now_iso


def read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else None


def float_or_none(value: Any) -> float | None:
    if value in (None, "", "None"):
        return None
    return float(value)


def int_or_none(value: Any) -> int | None:
    if value in (None, "", "None"):
        return None
    return int(float(value))


def json_or_none(value: Any) -> Any:
    if value in (None, "", "None"):
        return None
    return value


def component_loss(payload: dict[str, Any] | None, component_name: str) -> float | None:
    if not payload:
        return None
    components = payload.get("component_losses")
    if not isinstance(components, dict):
        return None
    return float_or_none(components.get(component_name))


def eval_metrics(metrics: dict[str, Any] | None, kind: str) -> dict[str, Any]:
    if not metrics:
        return {}
    evaluation_sets = metrics.get("evaluation_sets")
    if not isinstance(evaluation_sets, dict):
        return {}
    entry = evaluation_sets.get(kind)
    if not isinstance(entry, dict):
        return {}
    values = entry.get("metrics")
    return dict(values) if isinstance(values, dict) else {}


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def mean_std(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None}
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def write_summary_csv(path: Path, *, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fieldnames})


def write_summary_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_failures_json(path: Path, failures: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump({"generated_at_utc": utc_now_iso(), "failures": failures}, f, indent=2)
