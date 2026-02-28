"""Disagreement scoring for QBC candidate selection."""

from __future__ import annotations

import numpy as np

from src.methods.ensemble import DeepEnsemble


def score_disagreement(
    ensemble: DeepEnsemble | None = None,
    candidate_ics: np.ndarray | None = None,
    batch_size: int = 2048,
    metric: str = "variance_mean",
    predictions: np.ndarray | None = None,
) -> np.ndarray:
    """Return one uncertainty score per candidate IC.

    Supported metrics:
    - variance_mean: mean predictive variance across time/state
    - variance_max: max predictive variance across time/state
    - variance_p90: 90th percentile predictive variance across time/state
    - member_l2_mean: mean L2 spread of members around ensemble mean
    """
    if predictions is None:
        if ensemble is None or candidate_ics is None:
            raise ValueError("Either predictions or (ensemble and candidate_ics) must be provided.")
        preds = ensemble.predict_all(candidate_ics, batch_size=batch_size)  # (M, N, T, S)
    else:
        preds = np.asarray(predictions)
    if preds.ndim != 4:
        raise ValueError(f"predictions must have shape (M,N,T,S), got {preds.shape}")

    var = np.var(preds, axis=0)  # (N, T, S)
    metric_key = metric.strip().lower()
    if metric_key == "variance_mean":
        return var.mean(axis=(1, 2))
    if metric_key == "variance_max":
        return var.max(axis=(1, 2))
    if metric_key == "variance_p90":
        return np.percentile(var.reshape(var.shape[0], -1), 90, axis=1)
    if metric_key == "member_l2_mean":
        pred_mean = np.mean(preds, axis=0, keepdims=True)
        member_l2 = np.sqrt(np.sum((preds - pred_mean) ** 2, axis=(2, 3)))  # (M, N)
        return np.mean(member_l2, axis=0)
    raise ValueError(
        "Unknown disagreement metric. Use one of: "
        "['variance_mean', 'variance_max', 'variance_p90', 'member_l2_mean']"
    )
