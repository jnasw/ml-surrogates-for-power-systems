#!/usr/bin/env python3
"""Collect one normalized row per HPO config from a stage root."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _parse_override_text(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for chunk in str(text or "").split(";;"):
        chunk = chunk.strip()
        if not chunk or "=" not in chunk:
            continue
        key, value = chunk.split("=", 1)
        out[key] = value
    return out


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_matrix(matrix_path: Path) -> list[dict[str, str]]:
    with matrix_path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _infer_stage_from_root(stage_root: Path) -> str:
    return stage_root.name


def _metric_to_row_value(metric: str | None, row: dict[str, Any]) -> Any:
    if not metric:
        return None
    if metric in row:
        return row[metric]
    final_key = f"final_{metric}"
    if final_key in row:
        return row[final_key]
    return None


def _coerce_status(return_code: int | None, history_rows: list[dict[str, Any]]) -> str:
    if return_code is None:
        return "missing_status"
    if return_code != 0:
        return "failed"
    if not history_rows:
        return "completed_no_history"
    return "completed"


def collect_stage_results(
    *,
    stage_root: Path,
    workflow_id: str = "",
    workflow_version: str = "",
    method: str = "",
    model_flag: str = "",
    preset: str = "",
    stage: str = "",
    stage_role: str = "",
    objective_metric: str = "",
    objective_direction: str = "",
) -> list[dict[str, Any]]:
    matrix_path = stage_root / "matrix.tsv"
    if not matrix_path.exists():
        raise FileNotFoundError(f"matrix.tsv not found in stage root: {stage_root}")

    rows = _load_matrix(matrix_path)
    out: list[dict[str, Any]] = []
    inferred_stage = stage or _infer_stage_from_root(stage_root)

    for matrix_row in rows:
        cfg_id = str(matrix_row.get("cfg_id", "")).strip()
        run_root = stage_root / "runs" / cfg_id

        status_payload = _read_json(run_root / "hpo_status.json") or {}
        run_manifest = _read_json(run_root / "run_manifest.json") or {}
        dataset_manifest = _read_json(run_root / "dataset_manifest.json") or {}
        history_rows = _read_jsonl(run_root / "qbc" / "history.jsonl")
        final_history = history_rows[-1] if history_rows else {}

        rec: dict[str, Any] = {
            "workflow_id": workflow_id,
            "workflow_version": workflow_version,
            "method": method or matrix_row.get("method", ""),
            "model_flag": model_flag or matrix_row.get("model_flag", ""),
            "preset": preset or matrix_row.get("preset", ""),
            "stage": inferred_stage,
            "stage_role": stage_role,
            "stage_root": str(stage_root),
            "cfg_id": cfg_id,
            "row_idx": matrix_row.get("row_idx"),
            "run_root": str(run_root),
            "parent_cfg_id": "",
            "dataset_seed": matrix_row.get("seed", ""),
            "budget": matrix_row.get("budget", ""),
            "experiment_id": matrix_row.get("experiment_id", ""),
            "status": _coerce_status(status_payload.get("return_code"), history_rows),
            "return_code": status_payload.get("return_code"),
            "completed": bool(status_payload.get("return_code") == 0),
            "objective_metric": objective_metric,
            "objective_direction": objective_direction,
            "run_manifest_path": str(run_root / "run_manifest.json"),
            "dataset_manifest_path": str(run_root / "dataset_manifest.json") if dataset_manifest else "",
            "history_path": str(run_root / "qbc" / "history.jsonl") if history_rows else "",
            "stage1_overrides_raw": matrix_row.get("stage1_overrides", ""),
            "stage2_overrides_raw": matrix_row.get("stage2_overrides", ""),
            "stage3_overrides_raw": matrix_row.get("stage3_overrides", ""),
        }

        for key, value in final_history.items():
            rec[f"final_{key}"] = value

        rec["final_round_idx"] = final_history.get("round_idx")
        rec["final_train_size"] = final_history.get("train_size")
        rec["final_eval_rmse"] = final_history.get("eval_rmse")
        rec["final_eval_mse"] = final_history.get("eval_mse")
        rec["final_mean_score"] = final_history.get("mean_score")
        rec["final_selected_mean_score"] = final_history.get("selected_mean_score")
        rec["final_round_seconds"] = final_history.get("round_seconds")
        rec["final_train_seconds"] = final_history.get("train_seconds")
        rec["final_candidate_generation_seconds"] = final_history.get("candidate_generation_seconds")
        rec["final_candidate_simulation_seconds"] = final_history.get("candidate_simulation_seconds")
        rec["final_acquisition_seconds"] = final_history.get("acquisition_seconds")
        rec["final_selected_simulation_seconds"] = final_history.get("selected_simulation_seconds")
        rec["final_eval_seconds"] = final_history.get("eval_seconds")

        if run_manifest:
            rec["manifest_dataset_manifest_path"] = run_manifest.get("dataset_manifest_path", "")
            rec["manifest_telemetry_path"] = run_manifest.get("telemetry_path", "")

        if dataset_manifest:
            baseline_summary = dataset_manifest.get("baseline_summary", {})
            rec["baseline_summary_rmse_mean"] = baseline_summary.get("rmse_mean")
            rec["baseline_summary_rmse_std"] = baseline_summary.get("rmse_std")
            rec["baseline_summary_mse_mean"] = baseline_summary.get("mse_mean")
            rec["baseline_summary_mse_std"] = baseline_summary.get("mse_std")

        override_maps = [
            _parse_override_text(matrix_row.get("stage1_overrides", "")),
            _parse_override_text(matrix_row.get("stage2_overrides", "")),
            _parse_override_text(matrix_row.get("stage3_overrides", "")),
        ]
        for override_map in override_maps:
            for key, value in override_map.items():
                rec[key] = value

        rec["objective_value"] = _metric_to_row_value(objective_metric, rec)
        out.append(rec)

    return out


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    priority = [
        "workflow_id",
        "workflow_version",
        "method",
        "model_flag",
        "preset",
        "stage",
        "stage_role",
        "stage_root",
        "cfg_id",
        "row_idx",
        "run_root",
        "parent_cfg_id",
        "dataset_seed",
        "budget",
        "experiment_id",
        "status",
        "return_code",
        "completed",
        "objective_metric",
        "objective_direction",
        "objective_value",
        "final_eval_rmse",
        "final_eval_mse",
        "final_round_seconds",
        "final_train_seconds",
        "final_candidate_generation_seconds",
        "final_candidate_simulation_seconds",
        "final_acquisition_seconds",
        "final_selected_simulation_seconds",
        "final_eval_seconds",
    ]
    keys = set()
    for row in rows:
        keys.update(row.keys())
    ordered = [key for key in priority if key in keys]
    remaining = sorted(keys - set(ordered))
    return ordered + remaining


def write_stage_results_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _fieldnames(rows)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", required=True, help="Path to one HPO stage root.")
    parser.add_argument("--out", default=None, help="CSV output path. Defaults to <stage-root>/stage_results.csv")
    parser.add_argument("--workflow-id", default="", help="Optional workflow id metadata.")
    parser.add_argument("--workflow-version", default="", help="Optional workflow version metadata.")
    parser.add_argument("--method", default="", help="Optional method override.")
    parser.add_argument("--model-flag", default="", help="Optional model flag override.")
    parser.add_argument("--preset", default="", help="Optional preset override.")
    parser.add_argument("--stage", default="", help="Optional stage name override.")
    parser.add_argument("--stage-role", default="", help="Optional stage role metadata.")
    parser.add_argument("--objective-metric", default="", help="Optional objective metric name.")
    parser.add_argument("--objective-direction", default="", help="Optional objective direction.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    stage_root = Path(args.stage_root).resolve()
    out_path = Path(args.out).resolve() if args.out else stage_root / "stage_results.csv"

    rows = collect_stage_results(
        stage_root=stage_root,
        workflow_id=args.workflow_id,
        workflow_version=args.workflow_version,
        method=args.method,
        model_flag=args.model_flag,
        preset=args.preset,
        stage=args.stage,
        stage_role=args.stage_role,
        objective_metric=args.objective_metric,
        objective_direction=args.objective_direction,
    )
    write_stage_results_csv(rows, out_path)

    print(f"[hpo-workflow] stage_root={stage_root}")
    print(f"[hpo-workflow] rows={len(rows)}")
    print(f"[hpo-workflow] out={out_path}")


if __name__ == "__main__":
    main()
