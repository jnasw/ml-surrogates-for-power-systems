"""Standalone marker-directed adaptive sampling loop."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.distance import cdist

from src.active.marker_features import MarkerFeatureConfig, compute_marker_matrix
from src.data.trajectory_dataset import TrajectoryDataset
from src.sampling.ic_sampler import sample_initial_ics
from src.simulation.simulator import TrajectorySimulator


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    cur = cfg
    for part in key.split("."):
        if isinstance(cur, dict):
            if part not in cur:
                return default
            cur = cur[part]
        else:
            if not hasattr(cur, part):
                return default
            cur = getattr(cur, part)
    return cur


def _standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def _standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def _fit_pca(x_std: np.ndarray, explained_var_ratio: float) -> tuple[np.ndarray, np.ndarray, int]:
    # x_std expected centered-ish; compute PCA with SVD.
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
    base_scores: np.ndarray,
    k: int,
    preselect_factor: int,
    alpha_score: float,
) -> np.ndarray:
    n = embedding.shape[0]
    if k <= 0 or k > n:
        raise ValueError("k must be in [1, n_candidates].")
    pre_n = min(n, max(k, int(k * preselect_factor)))
    # Stable sort keeps tie handling deterministic across runs/platforms.
    pool_idx = np.argsort(base_scores, kind="mergesort")[-pre_n:][::-1]
    pool = embedding[pool_idx]
    pool_scores = base_scores[pool_idx]

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


@dataclass
class MarkerRoundSummary:
    round_idx: int
    train_size: int
    selected_indices: np.ndarray
    mean_score: float
    max_score: float
    p90_score: float
    selected_mean_score: float
    selected_min_score: float
    selected_max_score: float
    marker_pca_components: int
    mean_marker_diversity: float
    p90_marker_diversity: float
    mean_marker_sparsity: float
    p90_marker_sparsity: float
    selected_mean_marker_diversity: float
    selected_mean_marker_sparsity: float
    preselect_size: int
    marker_diversity_weight: float
    marker_sparsity_weight: float
    mean_selected_to_train_distance: float
    train_seconds: float | None = None
    candidate_generation_seconds: float | None = None
    candidate_simulation_seconds: float | None = None
    acquisition_seconds: float | None = None
    selected_simulation_seconds: float | None = None
    eval_seconds: float | None = None
    round_seconds: float | None = None


def run_marker_loop(
    dataset: TrajectoryDataset,
    bounds: np.ndarray,
    simulator: TrajectorySimulator,
    P: int,
    K: int,
    T: int,
    base_seed: int,
    config: Any,
    round_callback: Any | None = None,
) -> tuple[TrajectoryDataset, list[MarkerRoundSummary]]:
    """Run iterative marker-directed sampling without uncertainty models."""
    history: list[MarkerRoundSummary] = []
    candidate_method = str(_cfg_get(config, "active.candidate_method", "lhs"))

    explained = float(_cfg_get(config, "active.marker.pca_explained_variance", 0.90))
    w_div = float(_cfg_get(config, "active.marker.weights.diversity", 0.6))
    w_sparse = float(_cfg_get(config, "active.marker.weights.sparsity", 0.4))
    preselect_factor = int(_cfg_get(config, "active.marker.preselect_factor", 5))
    alpha_score = float(_cfg_get(config, "active.marker.greedy_score_weight", 0.7))
    k_density = int(_cfg_get(config, "active.marker.k_density", 15))
    settling_fraction = float(_cfg_get(config, "active.marker.settling_fraction", 0.05))
    include_anchor = bool(_cfg_get(config, "active.marker.include_anchor_state_markers", True))

    if not (0.0 < explained <= 1.0):
        raise ValueError("active.marker.pca_explained_variance must be in (0, 1].")
    if not np.isfinite(w_div) or not np.isfinite(w_sparse):
        raise ValueError("active.marker.weights.diversity/sparsity must be finite.")
    if w_div < 0.0 or w_sparse < 0.0:
        raise ValueError("active.marker.weights.diversity/sparsity must be >= 0.")
    if (w_div + w_sparse) <= 0.0:
        raise ValueError("active.marker.weights.diversity + active.marker.weights.sparsity must be > 0.")
    if preselect_factor < 1:
        raise ValueError("active.marker.preselect_factor must be >= 1.")
    if not (0.0 <= alpha_score <= 1.0):
        raise ValueError("active.marker.greedy_score_weight must be in [0, 1].")
    if k_density < 1:
        raise ValueError("active.marker.k_density must be >= 1.")
    if not (0.0 < settling_fraction <= 1.0):
        raise ValueError("active.marker.settling_fraction must be in (0, 1].")

    # Normalize weights so configs are comparable even if users do not sum to 1.
    w_sum = w_div + w_sparse
    w_div = float(w_div / w_sum)
    w_sparse = float(w_sparse / w_sum)

    marker_cfg = MarkerFeatureConfig(
        settling_fraction=settling_fraction,
        include_anchor_state_markers=include_anchor,
    )
    state_names = list(getattr(simulator, "state_names", []))
    print(
        "[marker] Start loop | "
        f"rounds={T} P={P} K={K} candidate_method={candidate_method} "
        f"w_div={w_div:.3f} w_sparse={w_sparse:.3f}"
    )

    for round_idx in range(T):
        t_round_start = time.perf_counter()
        print(f"[marker][round {round_idx:03d}] train_size={dataset.n_train} -> generating candidates")
        t_cand_start = time.perf_counter()
        cand_seed = base_seed + 3000 * round_idx + 23
        x_cand = sample_initial_ics(method=candidate_method, n=P, bounds=bounds, seed=cand_seed)
        candidate_generation_seconds = float(time.perf_counter() - t_cand_start)
        t_cand_sim_start = time.perf_counter()
        _, y_cand = simulator.simulate_trajectory(x_cand)
        candidate_simulation_seconds = float(time.perf_counter() - t_cand_sim_start)
        print(f"[marker][round {round_idx:03d}] simulated candidates: {x_cand.shape[0]}")

        x_train, y_train = dataset.training_view()

        t_acq_start = time.perf_counter()
        m_train, _ = compute_marker_matrix(
            trajs=y_train,
            time_grid=dataset.time_grid if dataset.time_grid is not None else simulator.t_eval.astype(np.float32),
            state_names=state_names if state_names else None,
            cfg=marker_cfg,
        )
        m_cand, _ = compute_marker_matrix(
            trajs=y_cand,
            time_grid=dataset.time_grid if dataset.time_grid is not None else simulator.t_eval.astype(np.float32),
            state_names=state_names if state_names else None,
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
        sparse = np.partition(d_ct, kk - 1, axis=1)[:, :kk].mean(axis=1)

        base_scores = w_div * _normalize01(diversity) + w_sparse * _normalize01(sparse)
        preselect_size = min(int(base_scores.shape[0]), max(K, int(K * preselect_factor)))
        selected_idx = _greedy_select(
            embedding=z_cand,
            base_scores=base_scores,
            k=K,
            preselect_factor=preselect_factor,
            alpha_score=alpha_score,
        )
        acquisition_seconds = float(time.perf_counter() - t_acq_start)

        selected_ics = x_cand[selected_idx]
        selected_trajs = y_cand[selected_idx]
        dataset.append(selected_ics, selected_trajs)
        print(
            f"[marker][round {round_idx:03d}] selected={selected_ics.shape[0]} "
            f"mean_score={float(np.mean(base_scores)):.6f} max_score={float(np.max(base_scores)):.6f} "
            f"new_train_size={dataset.n_train} "
            f"(cand={candidate_generation_seconds:.3f}s, cand_sim={candidate_simulation_seconds:.3f}s, "
            f"acq={acquisition_seconds:.3f}s, total={float(time.perf_counter() - t_round_start):.3f}s)"
        )

        summary = MarkerRoundSummary(
            round_idx=round_idx,
            train_size=int(x_train.shape[0]),
            selected_indices=selected_idx.copy(),
            mean_score=float(np.mean(base_scores)),
            max_score=float(np.max(base_scores)),
            p90_score=float(np.percentile(base_scores, 90)),
            selected_mean_score=float(np.mean(base_scores[selected_idx])),
            selected_min_score=float(np.min(base_scores[selected_idx])),
            selected_max_score=float(np.max(base_scores[selected_idx])),
            marker_pca_components=int(n_comp),
            mean_marker_diversity=float(np.mean(diversity)),
            p90_marker_diversity=float(np.percentile(diversity, 90)),
            mean_marker_sparsity=float(np.mean(sparse)),
            p90_marker_sparsity=float(np.percentile(sparse, 90)),
            selected_mean_marker_diversity=float(np.mean(diversity[selected_idx])),
            selected_mean_marker_sparsity=float(np.mean(sparse[selected_idx])),
            preselect_size=int(preselect_size),
            marker_diversity_weight=float(w_div),
            marker_sparsity_weight=float(w_sparse),
            mean_selected_to_train_distance=float(np.mean(diversity[selected_idx])),
            train_seconds=0.0,
            candidate_generation_seconds=candidate_generation_seconds,
            candidate_simulation_seconds=candidate_simulation_seconds,
            acquisition_seconds=acquisition_seconds,
            selected_simulation_seconds=0.0,
            eval_seconds=0.0,
            round_seconds=float(time.perf_counter() - t_round_start),
        )
        history.append(summary)

        if round_callback is not None:
            round_callback(
                summary=summary,
                x_cand=x_cand,
                scores=base_scores,
                marker_diversity=diversity,
                marker_sparsity=sparse,
                marker_embedding=z_cand,
                marker_train_embedding=z_train,
                selected_idx=selected_idx,
                selected_ics=selected_ics,
                selected_trajs=selected_trajs,
                dataset=dataset,
            )

    print(f"[marker] Loop finished | final_train_size={dataset.n_train} rounds_executed={len(history)}")
    return dataset, history
