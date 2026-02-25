"""Hybrid QBC + marker-space acquisition utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.distance import cdist

from src.active.marker_features import MarkerFeatureConfig, compute_marker_matrix


def _standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def _standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def _fit_pca(x_std: np.ndarray, explained_var_ratio: float) -> tuple[np.ndarray, np.ndarray, int]:
    x_centered = x_std - x_std.mean(axis=0, keepdims=True)
    _, svals, vt = np.linalg.svd(x_centered, full_matrices=False)
    var = (svals**2) / max(1, x_centered.shape[0] - 1)
    total = float(np.sum(var))
    if total <= 0.0:
        n_comp = 1
    else:
        cum = np.cumsum(var / total)
        n_comp = int(np.searchsorted(cum, float(explained_var_ratio)) + 1)
        n_comp = max(1, min(n_comp, vt.shape[0]))
    components = vt[:n_comp].T
    return x_centered.mean(axis=0), components, n_comp


def _pca_transform(x_std: np.ndarray, center: np.ndarray, components: np.ndarray) -> np.ndarray:
    return (x_std - center) @ components


def _normalize01(x: np.ndarray) -> np.ndarray:
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if x_max - x_min < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - x_min) / (x_max - x_min)).astype(np.float32)


def _greedy_select(
    embedding: np.ndarray,
    scores: np.ndarray,
    k: int,
    preselect_factor: int,
    alpha_score: float,
) -> np.ndarray:
    n = embedding.shape[0]
    if k <= 0 or k > n:
        raise ValueError("k must be in [1, n_candidates].")
    pre_n = min(n, max(k, int(k * preselect_factor)))
    pool_idx = np.argsort(scores)[-pre_n:][::-1]
    pool = embedding[pool_idx]
    pool_scores = scores[pool_idx]

    selected_local = [int(np.argmax(pool_scores))]
    while len(selected_local) < k:
        remain = [i for i in range(pre_n) if i not in selected_local]
        sel_pts = pool[selected_local]
        best_i = None
        best_val = -np.inf
        for i in remain:
            d = np.linalg.norm(pool[i][None, :] - sel_pts, axis=1)
            min_d = float(np.min(d))
            value = float(alpha_score) * float(pool_scores[i]) + (1.0 - float(alpha_score)) * min_d
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
    explained = float(getattr(getattr(config.active, "hybrid", {}), "pca_explained_variance", 0.90))
    preselect_factor = int(getattr(getattr(config.active, "hybrid", {}), "preselect_factor", 5))
    alpha_score = float(getattr(getattr(config.active, "hybrid", {}), "greedy_score_weight", 0.7))
    k_density = int(getattr(getattr(config.active, "hybrid", {}), "k_density", 15))

    weights = getattr(getattr(config.active, "hybrid", {}), "weights", {})
    w_u = float(getattr(weights, "uncertainty", 0.4))
    w_d = float(getattr(weights, "diversity", 0.4))
    w_s = float(getattr(weights, "sparsity", 0.2))
    w_sum = max(1e-12, w_u + w_d + w_s)
    w_u, w_d, w_s = w_u / w_sum, w_d / w_sum, w_s / w_sum

    settling_fraction = float(getattr(getattr(config.active, "hybrid", {}), "settling_fraction", 0.05))
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

    mean, std = _standardize_fit(m_train)
    m_train_std = _standardize_apply(m_train, mean, std)
    m_cand_std = _standardize_apply(m_cand, mean, std)

    pca_center, pca_components, n_comp = _fit_pca(m_train_std, explained_var_ratio=explained)
    z_train = _pca_transform(m_train_std, pca_center, pca_components)
    z_cand = _pca_transform(m_cand_std, pca_center, pca_components)

    d_ct = cdist(z_cand, z_train, metric="euclidean")
    diversity = d_ct.min(axis=1)
    kk = min(max(1, k_density), z_train.shape[0])
    sparsity = np.partition(d_ct, kk - 1, axis=1)[:, :kk].mean(axis=1)

    u_n = _normalize01(np.asarray(uncertainty_scores, dtype=np.float32))
    d_n = _normalize01(diversity)
    s_n = _normalize01(sparsity)
    hybrid_scores = w_u * u_n + w_d * d_n + w_s * s_n

    selected_idx = _greedy_select(
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
