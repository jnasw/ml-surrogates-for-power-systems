"""Helpers for adaptive sampling metadata written to dataset info.txt."""

from __future__ import annotations

from typing import Any


def build_qbc_metadata(
    *,
    config: Any,
    n0: int,
    n_test: int,
    M: int,
    P: int,
    K: int,
    T: int,
    final_train_size: int,
    source_run_dir: str | None = None,
) -> dict[str, Any]:
    """Build standardized info.txt metadata for QBC-generated datasets."""
    acquisition_strategy = str(getattr(config.active, "acquisition_strategy", "qbc_only"))
    disagreement = getattr(config.active, "disagreement", {})
    diversity = getattr(config.active, "diversity", {})
    hybrid = getattr(config.active, "hybrid", {})
    hybrid_weights = getattr(hybrid, "weights", {})
    meta: dict[str, Any] = {
        "Adaptive sampling method": "qbc",
        "Adaptive candidate sampling method": str(getattr(config.active, "candidate_method", "sobol")),
        "Adaptive diversify": bool(getattr(config.active, "diversify", True)),
        "Adaptive acquisition strategy": acquisition_strategy,
        "Adaptive disagreement metric": str(getattr(disagreement, "metric", "variance_mean")),
        "Adaptive diversity preselect factor": int(getattr(diversity, "preselect_factor", 5)),
        "Adaptive diversity uncertainty weight": float(getattr(diversity, "uncertainty_weight", 0.7)),
        "Adaptive diversity distance weight": float(getattr(diversity, "distance_weight", 0.3)),
        "Adaptive diversity normalize uncertainty": bool(getattr(diversity, "normalize_uncertainty", True)),
        "Adaptive diversity normalize distance": bool(getattr(diversity, "normalize_distance", True)),
        "Adaptive surrogate deterministic": bool(getattr(getattr(config, "surrogate", {}), "deterministic", False)),
        "Adaptive initial sample size (qbc_n0)": int(n0),
        "Adaptive held-out test size (qbc_n_test)": int(n_test),
        "Adaptive committee size (qbc_M)": int(M),
        "Adaptive candidate pool size (qbc_P)": int(P),
        "Adaptive selected per round (qbc_K)": int(K),
        "Adaptive rounds (qbc_T)": int(T),
        "Adaptive final training trajectories": int(final_train_size),
    }
    if acquisition_strategy == "qbc_marker_hybrid":
        meta["Adaptive hybrid pca explained variance"] = float(getattr(hybrid, "pca_explained_variance", 0.90))
        meta["Adaptive hybrid k density"] = int(getattr(hybrid, "k_density", 15))
        meta["Adaptive hybrid preselect factor"] = int(getattr(hybrid, "preselect_factor", 5))
        meta["Adaptive hybrid uncertainty weight"] = float(getattr(hybrid_weights, "uncertainty", 0.4))
        meta["Adaptive hybrid diversity weight"] = float(getattr(hybrid_weights, "diversity", 0.4))
        meta["Adaptive hybrid sparsity weight"] = float(getattr(hybrid_weights, "sparsity", 0.2))
    if source_run_dir is not None:
        meta["Adaptive source run directory"] = str(source_run_dir)
    return meta


def build_marker_metadata(
    *,
    config: Any,
    n0: int,
    P: int,
    K: int,
    T: int,
    final_train_size: int,
    source_run_dir: str | None = None,
) -> dict[str, Any]:
    """Build standardized info.txt metadata for marker-directed adaptive datasets."""
    meta: dict[str, Any] = {
        "Adaptive sampling method": "marker_directed",
        "Adaptive candidate sampling method": str(getattr(config.active, "candidate_method", "lhs")),
        "Adaptive initial sample size (qbc_n0)": int(n0),
        "Adaptive candidate pool size (qbc_P)": int(P),
        "Adaptive selected per round (qbc_K)": int(K),
        "Adaptive rounds (qbc_T)": int(T),
        "Adaptive final training trajectories": int(final_train_size),
        "Adaptive marker pca explained variance": float(
            getattr(getattr(config.active, "marker", {}), "pca_explained_variance", 0.90)
        ),
        "Adaptive marker diversity weight": float(
            getattr(getattr(getattr(config.active, "marker", {}), "weights", {}), "diversity", 0.6)
        ),
        "Adaptive marker sparsity weight": float(
            getattr(getattr(getattr(config.active, "marker", {}), "weights", {}), "sparsity", 0.4)
        ),
        "Adaptive marker k density": int(getattr(getattr(config.active, "marker", {}), "k_density", 15)),
        "Adaptive marker preselect factor": int(getattr(getattr(config.active, "marker", {}), "preselect_factor", 5)),
        "Adaptive marker greedy score weight": float(
            getattr(getattr(config.active, "marker", {}), "greedy_score_weight", 0.7)
        ),
        "Adaptive marker settling fraction": float(
            getattr(getattr(config.active, "marker", {}), "settling_fraction", 0.05)
        ),
        "Adaptive marker include anchor state markers": bool(
            getattr(getattr(config.active, "marker", {}), "include_anchor_state_markers", True)
        ),
    }
    if source_run_dir is not None:
        meta["Adaptive source run directory"] = str(source_run_dir)
    return meta
