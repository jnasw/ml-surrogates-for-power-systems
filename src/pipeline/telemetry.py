"""Round-level telemetry export helpers for adaptive runs."""

from __future__ import annotations

import csv
import json
import os
from typing import Any

import numpy as np


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _safe_np_load(path: str) -> np.ndarray | None:
    if not os.path.exists(path):
        return None
    return np.load(path)


def _selected_mean(values: np.ndarray | None, selected_idx: np.ndarray | None) -> float | None:
    if values is None or selected_idx is None:
        return None
    if values.size == 0 or selected_idx.size == 0:
        return None
    return float(np.mean(values[selected_idx]))


def _arr_stat(values: np.ndarray | None, stat: str) -> float | None:
    if values is None or values.size == 0:
        return None
    if stat == "mean":
        return float(np.mean(values))
    if stat == "p90":
        return float(np.percentile(values, 90))
    if stat == "max":
        return float(np.max(values))
    return None


def build_round_telemetry_rows(
    *,
    history_jsonl_path: str,
    rounds_dir: str,
    run_id: str,
    method: str,
    budget: str,
    seed_label: str,
) -> list[dict[str, Any]]:
    """Build flat per-round telemetry rows from logger artifacts."""
    history_rows = _read_jsonl(history_jsonl_path)
    out: list[dict[str, Any]] = []
    for row in history_rows:
        ridx = int(row["round_idx"])
        round_dir = os.path.join(rounds_dir, f"round_{ridx:03d}")
        selected_idx = _safe_np_load(os.path.join(round_dir, "selected_indices.npy"))
        if selected_idx is None:
            selected_idx = np.array([], dtype=int)
        else:
            selected_idx = selected_idx.astype(int)

        marker_div = _safe_np_load(os.path.join(round_dir, "marker_diversity.npy"))
        marker_sparse = _safe_np_load(os.path.join(round_dir, "marker_sparsity.npy"))
        hybrid_div = _safe_np_load(os.path.join(round_dir, "hybrid_diversity.npy"))
        hybrid_sparse = _safe_np_load(os.path.join(round_dir, "hybrid_sparsity.npy"))
        hybrid_unc = _safe_np_load(os.path.join(round_dir, "hybrid_uncertainty.npy"))
        hybrid_score = _safe_np_load(os.path.join(round_dir, "hybrid_score.npy"))

        out.append(
            {
                "run_id": run_id,
                "method": method,
                "budget": budget,
                "seed_label": seed_label,
                "round_idx": ridx,
                "train_size_before": int(row.get("train_size", -1)),
                "selected_count": int(len(row.get("selected_indices", []))),
                "disagreement_metric": row.get("disagreement_metric"),
                "mean_disagreement": row.get("mean_disagreement"),
                "max_disagreement": row.get("max_disagreement"),
                "mean_score": row.get("mean_score", row.get("mean_disagreement")),
                "max_score": row.get("max_score", row.get("max_disagreement")),
                "p90_score": row.get("p90_score"),
                "selected_mean_score": row.get("selected_mean_score"),
                "selected_min_score": row.get("selected_min_score"),
                "selected_max_score": row.get("selected_max_score"),
                "preselect_size": row.get("preselect_size"),
                "eval_mse": row.get("eval_mse"),
                "eval_rmse": row.get("eval_rmse"),
                "marker_pca_components": row.get("marker_pca_components"),
                "hybrid_marker_pca_components": row.get("hybrid_marker_pca_components"),
                "marker_diversity_weight": row.get("marker_diversity_weight"),
                "marker_sparsity_weight": row.get("marker_sparsity_weight"),
                "mean_selected_to_train_distance": row.get("mean_selected_to_train_distance"),
                "mean_marker_diversity": row.get("mean_marker_diversity", _arr_stat(marker_div, "mean")),
                "p90_marker_diversity": row.get("p90_marker_diversity", _arr_stat(marker_div, "p90")),
                "max_marker_diversity": _arr_stat(marker_div, "max"),
                "mean_marker_sparsity": row.get("mean_marker_sparsity", _arr_stat(marker_sparse, "mean")),
                "p90_marker_sparsity": row.get("p90_marker_sparsity", _arr_stat(marker_sparse, "p90")),
                "max_marker_sparsity": _arr_stat(marker_sparse, "max"),
                "mean_selected_marker_diversity": _selected_mean(marker_div, selected_idx),
                "mean_selected_marker_sparsity": _selected_mean(marker_sparse, selected_idx),
                "mean_selected_hybrid_diversity": _selected_mean(hybrid_div, selected_idx),
                "mean_selected_hybrid_sparsity": _selected_mean(hybrid_sparse, selected_idx),
                "mean_selected_hybrid_uncertainty": _selected_mean(hybrid_unc, selected_idx),
                "mean_selected_hybrid_score": _selected_mean(hybrid_score, selected_idx),
                "train_seconds": row.get("train_seconds"),
                "candidate_generation_seconds": row.get("candidate_generation_seconds"),
                "candidate_simulation_seconds": row.get("candidate_simulation_seconds"),
                "acquisition_seconds": row.get("acquisition_seconds"),
                "selected_simulation_seconds": row.get("selected_simulation_seconds"),
                "eval_seconds": row.get("eval_seconds"),
                "round_seconds": row.get("round_seconds"),
            }
        )
    return out


def write_rows_csv(rows: list[dict[str, Any]], out_path: str) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_rows_parquet_best_effort(rows: list[dict[str, Any]], out_path: str) -> bool:
    """Try writing parquet. Returns False when pyarrow is unavailable."""
    if not rows:
        return False
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception:
        return False

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, out_path)
    return True
