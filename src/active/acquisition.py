"""Acquisition and diversity selection in IC space."""

from __future__ import annotations

import numpy as np


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

    selected_local = [int(np.argmax(pool_scores))]
    while len(selected_local) < K:
        remaining = [i for i in range(pre_n) if i not in selected_local]
        selected_points = pool[selected_local]
        best_i = None
        best_val = -np.inf
        for i in remaining:
            d = np.linalg.norm(pool[i][None, :] - selected_points, axis=1)
            min_d = float(np.min(d))
            value = 0.7 * float(pool_scores[i]) + 0.3 * min_d
            if value > best_val:
                best_val = value
                best_i = i
        selected_local.append(int(best_i))

    return pool_idx[np.array(selected_local, dtype=int)]
