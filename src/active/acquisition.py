"""Acquisition and diversity selection in IC space."""

from __future__ import annotations

import numpy as np


def _minmax_01(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return np.asarray(v, dtype=np.float32)
    v_min = float(np.min(v))
    v_max = float(np.max(v))
    denom = v_max - v_min
    if denom <= 1e-12:
        return np.zeros_like(v, dtype=np.float32)
    return ((v - v_min) / denom).astype(np.float32)


def acquire_topk(scores: np.ndarray, K: int) -> np.ndarray:
    if K <= 0:
        raise ValueError("K must be positive.")
    if K > len(scores):
        raise ValueError("K cannot exceed number of scores.")
    return np.argsort(scores)[-K:][::-1]


def acquire_diverse(
    candidate_ics: np.ndarray,
    scores: np.ndarray,
    K: int,
    preselect_factor: int = 5,
    uncertainty_weight: float = 0.7,
    distance_weight: float = 0.3,
    normalize_uncertainty: bool = True,
    normalize_distance: bool = True,
) -> np.ndarray:
    """Greedy diverse top-K on a high-score preselection pool."""
    if K <= 0:
        raise ValueError("K must be positive.")
    n = len(scores)
    if K > n:
        raise ValueError("K cannot exceed number of candidates.")

    pre_n = min(n, max(K, K * preselect_factor))
    pool_idx = np.argsort(scores)[-pre_n:][::-1]
    pool = candidate_ics[pool_idx]
    pool_scores = scores[pool_idx]
    pool_scores_norm = _minmax_01(pool_scores) if normalize_uncertainty else pool_scores

    selected_local = [int(np.argmax(pool_scores))]
    while len(selected_local) < K:
        remaining = [i for i in range(pre_n) if i not in selected_local]
        selected_points = pool[selected_local]
        dist_vals = []
        for i in remaining:
            d = np.linalg.norm(pool[i][None, :] - selected_points, axis=1)
            dist_vals.append(float(np.min(d)))
        dist_vals_arr = np.asarray(dist_vals, dtype=np.float32)
        dist_vals_norm = _minmax_01(dist_vals_arr) if normalize_distance else dist_vals_arr

        best_i = None
        best_val = -np.inf
        for j, i in enumerate(remaining):
            min_d = float(dist_vals_norm[j])
            score_i = float(pool_scores_norm[i])
            value = float(uncertainty_weight) * score_i + float(distance_weight) * min_d
            if value > best_val:
                best_val = value
                best_i = i
        selected_local.append(int(best_i))

    return pool_idx[np.array(selected_local, dtype=int)]
