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


STANDARD_PINN_SUMMARY_FIELDNAMES = [
    "final_train_mse",
    "final_train_rmse",
    "final_train_mae",
    "final_val_mse",
    "final_val_rmse",
    "final_val_mae",
    "final_test_mse",
    "final_test_rmse",
    "final_test_mae",
    "id_eval_mse",
    "id_eval_rmse",
    "id_eval_mae",
    "ood_eval_mse",
    "ood_eval_rmse",
    "ood_eval_mae",
    "id_ood_rmse_gap",
    "test_ood_rmse_gap",
    "best_checkpoint_epoch",
    "best_checkpoint_phase_name",
    "best_checkpoint_wall_seconds",
    "best_checkpoint_selection_metric",
    "best_checkpoint_selection_metric_name",
    "best_train_mse",
    "best_train_rmse",
    "best_train_mae",
    "best_val_mse",
    "best_val_rmse",
    "best_val_mae",
    "best_test_mse",
    "best_test_rmse",
    "best_test_mae",
    "best_id_eval_mse",
    "best_id_eval_rmse",
    "best_id_eval_mae",
    "best_ood_eval_mse",
    "best_ood_eval_rmse",
    "best_ood_eval_mae",
    "best_test_ood_rmse_gap",
    "epoch_metrics_path",
    "best_checkpoint_path",
]


STANDARD_PINN_AGGREGATE_KEYS = [
    "final_test_rmse",
    "ood_eval_rmse",
    "test_ood_rmse_gap",
    "best_test_rmse",
    "best_ood_eval_rmse",
    "best_test_ood_rmse_gap",
]


def _metric_value(values: dict[str, Any], key: str) -> float | None:
    try:
        return float_or_none(values.get(key))
    except (TypeError, ValueError):
        return None


def _best_checkpoint_payload(metrics: dict[str, Any] | None) -> dict[str, Any]:
    if not metrics:
        return {}
    best = metrics.get("best_checkpoint_metrics")
    return dict(best) if isinstance(best, dict) else {}


def _best_checkpoint_split_metrics(metrics: dict[str, Any] | None, split: str) -> dict[str, Any]:
    best = _best_checkpoint_payload(metrics)
    entry = best.get(split)
    if not isinstance(entry, dict):
        return {}
    values = entry.get("metrics")
    return dict(values) if isinstance(values, dict) else {}


def _best_checkpoint_eval_metrics(metrics: dict[str, Any] | None, kind: str) -> dict[str, Any]:
    best = _best_checkpoint_payload(metrics)
    evaluation_sets = best.get("evaluation_sets")
    if not isinstance(evaluation_sets, dict):
        return {}
    entry = evaluation_sets.get(kind)
    if not isinstance(entry, dict):
        return {}
    values = entry.get("metrics")
    return dict(values) if isinstance(values, dict) else {}


def _best_checkpoint_selection(metrics: dict[str, Any] | None) -> dict[str, Any]:
    best = _best_checkpoint_payload(metrics)
    selection = best.get("selection")
    return dict(selection) if isinstance(selection, dict) else {}


def _selection_metric(selection: dict[str, Any]) -> tuple[str | None, float | None]:
    val_loss = _metric_value(selection, "val_total_loss")
    if val_loss is not None:
        return "val_total_loss", val_loss
    train_loss = _metric_value(selection, "train_total_loss")
    if train_loss is not None:
        return "train_total_loss", train_loss
    return None, None


def extract_pinn_epoch_artifact_paths(run_dir: Path | None) -> dict[str, str | None]:
    if run_dir is None:
        return {"epoch_metrics_path": None, "best_checkpoint_path": None}
    epoch_metrics_path = run_dir / "epoch_metrics.csv"
    best_checkpoint_path = run_dir / "checkpoints" / "best.pt"
    return {
        "epoch_metrics_path": str(epoch_metrics_path) if epoch_metrics_path.exists() else None,
        "best_checkpoint_path": str(best_checkpoint_path) if best_checkpoint_path.exists() else None,
    }


def extract_standard_pinn_summary_fields(
    *,
    metrics: dict[str, Any] | None,
    run_dir: Path | None,
) -> dict[str, Any]:
    final_train = dict(metrics.get("final_train_metrics", {}) or {}) if metrics else {}
    final_val = dict(metrics.get("final_val_metrics", {}) or {}) if metrics else {}
    final_test = dict(metrics.get("final_test_metrics", {}) or {}) if metrics else {}
    id_eval = eval_metrics(metrics, "id")
    ood_eval = eval_metrics(metrics, "ood")
    best_selection = _best_checkpoint_selection(metrics)
    best_selection_name, best_selection_value = _selection_metric(best_selection)
    best_train = _best_checkpoint_split_metrics(metrics, "train")
    best_val = _best_checkpoint_split_metrics(metrics, "val")
    best_test = _best_checkpoint_split_metrics(metrics, "test")
    best_id = _best_checkpoint_eval_metrics(metrics, "id")
    best_ood = _best_checkpoint_eval_metrics(metrics, "ood")

    final_test_rmse = _metric_value(final_test, "rmse")
    id_eval_rmse = _metric_value(id_eval, "rmse")
    ood_eval_rmse = _metric_value(ood_eval, "rmse")
    best_test_rmse = _metric_value(best_test, "rmse")
    best_ood_rmse = _metric_value(best_ood, "rmse")

    values = {
        "final_train_mse": _metric_value(final_train, "mse"),
        "final_train_rmse": _metric_value(final_train, "rmse"),
        "final_train_mae": _metric_value(final_train, "mae"),
        "final_val_mse": _metric_value(final_val, "mse"),
        "final_val_rmse": _metric_value(final_val, "rmse"),
        "final_val_mae": _metric_value(final_val, "mae"),
        "final_test_mse": _metric_value(final_test, "mse"),
        "final_test_rmse": final_test_rmse,
        "final_test_mae": _metric_value(final_test, "mae"),
        "id_eval_mse": _metric_value(id_eval, "mse"),
        "id_eval_rmse": id_eval_rmse,
        "id_eval_mae": _metric_value(id_eval, "mae"),
        "ood_eval_mse": _metric_value(ood_eval, "mse"),
        "ood_eval_rmse": ood_eval_rmse,
        "ood_eval_mae": _metric_value(ood_eval, "mae"),
        "id_ood_rmse_gap": None if id_eval_rmse is None or ood_eval_rmse is None else ood_eval_rmse - id_eval_rmse,
        "test_ood_rmse_gap": None if final_test_rmse is None or ood_eval_rmse is None else ood_eval_rmse - final_test_rmse,
        "best_checkpoint_epoch": int_or_none(best_selection.get("global_epoch")),
        "best_checkpoint_phase_name": best_selection.get("phase_name"),
        "best_checkpoint_wall_seconds": _metric_value(best_selection, "cumulative_wall_seconds"),
        "best_checkpoint_selection_metric": best_selection_value,
        "best_checkpoint_selection_metric_name": best_selection_name,
        "best_train_mse": _metric_value(best_train, "mse"),
        "best_train_rmse": _metric_value(best_train, "rmse"),
        "best_train_mae": _metric_value(best_train, "mae"),
        "best_val_mse": _metric_value(best_val, "mse"),
        "best_val_rmse": _metric_value(best_val, "rmse"),
        "best_val_mae": _metric_value(best_val, "mae"),
        "best_test_mse": _metric_value(best_test, "mse"),
        "best_test_rmse": best_test_rmse,
        "best_test_mae": _metric_value(best_test, "mae"),
        "best_id_eval_mse": _metric_value(best_id, "mse"),
        "best_id_eval_rmse": _metric_value(best_id, "rmse"),
        "best_id_eval_mae": _metric_value(best_id, "mae"),
        "best_ood_eval_mse": _metric_value(best_ood, "mse"),
        "best_ood_eval_rmse": best_ood_rmse,
        "best_ood_eval_mae": _metric_value(best_ood, "mae"),
        "best_test_ood_rmse_gap": None if best_test_rmse is None or best_ood_rmse is None else best_ood_rmse - best_test_rmse,
    }
    values.update(extract_pinn_epoch_artifact_paths(run_dir))
    return values


def aggregate_standard_pinn_metrics(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    aggregate: dict[str, float | None] = {}
    for key in STANDARD_PINN_AGGREGATE_KEYS:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        stats = mean_std(values)
        aggregate[f"{key}_mean"] = stats["mean"]
        aggregate[f"{key}_std"] = stats["std"]
    return aggregate


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
