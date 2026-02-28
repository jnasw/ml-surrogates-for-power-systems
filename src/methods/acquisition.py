"""Acquisition and diversity selection in IC space."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.distance import cdist

from src.methods.marker_utils import (
    MarkerFeatureConfig,
    compute_marker_matrix,
    fit_pca,
    greedy_select,
    normalize01,
    pca_transform,
    standardize_apply,
    standardize_fit,
)


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
    pool_scores_norm = normalize01(pool_scores) if normalize_uncertainty else pool_scores

    selected_local = [int(np.argmax(pool_scores))]
    while len(selected_local) < K:
        remaining = [i for i in range(pre_n) if i not in selected_local]
        selected_points = pool[selected_local]
        dist_vals = []
        for i in remaining:
            d = np.linalg.norm(pool[i][None, :] - selected_points, axis=1)
            dist_vals.append(float(np.min(d)))
        dist_vals_arr = np.asarray(dist_vals, dtype=np.float32)
        dist_vals_norm = normalize01(dist_vals_arr) if normalize_distance else dist_vals_arr

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


def select_qbc_marker_hybrid(
    *,
    train_trajs: np.ndarray,
    candidate_pred_mean_trajs: np.ndarray,
    uncertainty_scores: np.ndarray,
    time_grid: np.ndarray,
    state_names: list[str] | None,
    k_select: int,
    config: Any,
) -> tuple[np.ndarray, dict[str, np.ndarray | float | int]]:
    """Select candidate indices with a hybrid uncertainty+marker objective."""
    if train_trajs.shape[0] < 1:
        raise ValueError("Hybrid acquisition requires at least one training trajectory.")
    if candidate_pred_mean_trajs.shape[0] != uncertainty_scores.shape[0]:
        raise ValueError("candidate_pred_mean_trajs and uncertainty_scores must have matching first dimension.")

    explained = float(getattr(getattr(config.active, "hybrid", {}), "pca_explained_variance", 0.90))
    preselect_factor = int(getattr(getattr(config.active, "hybrid", {}), "preselect_factor", 5))
    alpha_score = float(getattr(getattr(config.active, "hybrid", {}), "greedy_score_weight", 0.7))
    k_density = int(getattr(getattr(config.active, "hybrid", {}), "k_density", 15))

    weights = getattr(getattr(config.active, "hybrid", {}), "weights", {})
    w_u = float(getattr(weights, "uncertainty", 0.4))
    w_d = float(getattr(weights, "diversity", 0.4))
    w_s = float(getattr(weights, "sparsity", 0.2))
    if not (0.0 < explained <= 1.0):
        raise ValueError("active.hybrid.pca_explained_variance must be in (0, 1].")
    if preselect_factor < 1:
        raise ValueError("active.hybrid.preselect_factor must be >= 1.")
    if not (0.0 <= alpha_score <= 1.0):
        raise ValueError("active.hybrid.greedy_score_weight must be in [0, 1].")
    if k_density < 1:
        raise ValueError("active.hybrid.k_density must be >= 1.")
    w_sum = max(1e-12, w_u + w_d + w_s)
    if not np.isfinite(w_u) or not np.isfinite(w_d) or not np.isfinite(w_s):
        raise ValueError("active.hybrid.weights must be finite.")
    if w_u < 0.0 or w_d < 0.0 or w_s < 0.0:
        raise ValueError("active.hybrid.weights must be >= 0.")
    if (w_u + w_d + w_s) <= 0.0:
        raise ValueError("active.hybrid.weights sum must be > 0.")
    w_u, w_d, w_s = w_u / w_sum, w_d / w_sum, w_s / w_sum

    settling_fraction = float(getattr(getattr(config.active, "hybrid", {}), "settling_fraction", 0.05))
    if not (0.0 < settling_fraction <= 1.0):
        raise ValueError("active.hybrid.settling_fraction must be in (0, 1].")
    include_anchor = bool(getattr(getattr(config.active, "hybrid", {}), "include_anchor_state_markers", True))
    marker_cfg = MarkerFeatureConfig(
        settling_fraction=settling_fraction,
        include_anchor_state_markers=include_anchor,
    )

    m_train, _ = compute_marker_matrix(
        trajs=np.asarray(train_trajs, dtype=np.float32),
        time_grid=np.asarray(time_grid, dtype=np.float32),
        state_names=state_names,
        cfg=marker_cfg,
    )
    m_cand, _ = compute_marker_matrix(
        trajs=np.asarray(candidate_pred_mean_trajs, dtype=np.float32),
        time_grid=np.asarray(time_grid, dtype=np.float32),
        state_names=state_names,
        cfg=marker_cfg,
    )

    mean, std = standardize_fit(m_train)
    m_train_std = standardize_apply(m_train, mean, std)
    m_cand_std = standardize_apply(m_cand, mean, std)

    pca_center, pca_components, n_comp = fit_pca(m_train_std, explained_var_ratio=explained)
    z_train = pca_transform(m_train_std, pca_center, pca_components)
    z_cand = pca_transform(m_cand_std, pca_center, pca_components)

    d_ct = cdist(z_cand, z_train, metric="euclidean")
    diversity = d_ct.min(axis=1)
    kk = min(max(1, k_density), z_train.shape[0])
    sparsity = np.partition(d_ct, kk - 1, axis=1)[:, :kk].mean(axis=1)

    u_n = normalize01(np.asarray(uncertainty_scores, dtype=np.float32))
    d_n = normalize01(diversity)
    s_n = normalize01(sparsity)
    hybrid_scores = w_u * u_n + w_d * d_n + w_s * s_n

    selected_idx = greedy_select(
        embedding=z_cand,
        scores=hybrid_scores,
        k=k_select,
        preselect_factor=preselect_factor,
        alpha_score=alpha_score,
    )
    diagnostics: dict[str, np.ndarray | float | int] = {
        "hybrid_uncertainty": uncertainty_scores.astype(np.float32),
        "hybrid_diversity": diversity.astype(np.float32),
        "hybrid_sparsity": sparsity.astype(np.float32),
        "hybrid_score": hybrid_scores.astype(np.float32),
        "hybrid_embedding": z_cand.astype(np.float32),
        "hybrid_train_embedding": z_train.astype(np.float32),
        "hybrid_marker_pca_components": int(n_comp),
    }
    return selected_idx, diagnostics
