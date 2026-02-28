"""Adaptive sampling loops for QBC and marker-directed methods."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.distance import cdist

from src.data.generate.ic_sampler import sample_initial_ics
from src.data.loaders.trajectory_dataset import TrajectoryDataset
from src.methods.acquisition import acquire_diverse, acquire_topk, select_qbc_marker_hybrid
from src.methods.disagreement import score_disagreement
from src.methods.ensemble import DeepEnsemble, train_ensemble
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
from src.sim.simulator import TrajectorySimulator


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


def _should_log_round(round_idx: int, total_rounds: int, log_every: int) -> bool:
    return (round_idx % log_every == 0) or (round_idx == total_rounds - 1)


@dataclass
class QBCRoundSummary:
    round_idx: int
    train_size: int
    selected_indices: np.ndarray
    disagreement_metric: str
    mean_score: float
    max_score: float
    p90_score: float
    selected_mean_score: float
    selected_min_score: float
    selected_max_score: float
    mean_disagreement: float
    max_disagreement: float
    mean_hybrid_score: float | None = None
    mean_selected_diversity: float | None = None
    mean_selected_sparsity: float | None = None
    hybrid_marker_pca_components: int | None = None
    eval_mse: float | None = None
    eval_rmse: float | None = None
    train_seconds: float | None = None
    candidate_generation_seconds: float | None = None
    candidate_simulation_seconds: float | None = None
    acquisition_seconds: float | None = None
    selected_simulation_seconds: float | None = None
    eval_seconds: float | None = None
    round_seconds: float | None = None


@dataclass(frozen=True)
class QBCConfig:
    M: int
    P: int
    K: int
    T: int
    base_seed: int
    start_round: int
    candidate_method: str
    diversify: bool
    batch_size: int
    disagreement_metric: str
    diversity_preselect_factor: int
    diversity_uncertainty_weight: float
    diversity_distance_weight: float
    diversity_norm_uncertainty: bool
    diversity_norm_distance: bool
    log_every: int
    acquisition_strategy: str

    @classmethod
    def from_runtime(
        cls,
        *,
        config: Any,
        M: int,
        P: int,
        K: int,
        T: int,
        base_seed: int,
        start_round: int = 0,
    ) -> "QBCConfig":
        candidate_method = str(_cfg_get(config, "active.candidate_method", "sobol"))
        diversify = bool(_cfg_get(config, "active.diversify", True))
        batch_size = int(_cfg_get(config, "active.predict_batch_size", 2048))
        disagreement_metric = str(_cfg_get(config, "active.disagreement.metric", "variance_mean"))
        diversity_preselect_factor = int(_cfg_get(config, "active.diversity.preselect_factor", 5))
        diversity_uncertainty_weight = float(_cfg_get(config, "active.diversity.uncertainty_weight", 0.7))
        diversity_distance_weight = float(_cfg_get(config, "active.diversity.distance_weight", 0.3))
        diversity_norm_uncertainty = bool(_cfg_get(config, "active.diversity.normalize_uncertainty", True))
        diversity_norm_distance = bool(_cfg_get(config, "active.diversity.normalize_distance", True))
        log_every = int(_cfg_get(config, "active.log_every", 5))
        acquisition_strategy = str(_cfg_get(config, "active.acquisition_strategy", "qbc_only")).lower()

        if M < 1 or P < 1 or K < 1 or T < 1:
            raise ValueError("QBCConfig requires M, P, K, and T to be >= 1.")
        if K > P:
            raise ValueError("QBCConfig requires K <= P.")
        if log_every < 1:
            raise ValueError("active.log_every must be >= 1.")
        if acquisition_strategy not in {"qbc_only", "qbc_marker_hybrid"}:
            raise ValueError("active.acquisition_strategy must be one of: ['qbc_only', 'qbc_marker_hybrid']")

        return cls(
            M=M,
            P=P,
            K=K,
            T=T,
            base_seed=base_seed,
            start_round=start_round,
            candidate_method=candidate_method,
            diversify=diversify,
            batch_size=batch_size,
            disagreement_metric=disagreement_metric,
            diversity_preselect_factor=diversity_preselect_factor,
            diversity_uncertainty_weight=diversity_uncertainty_weight,
            diversity_distance_weight=diversity_distance_weight,
            diversity_norm_uncertainty=diversity_norm_uncertainty,
            diversity_norm_distance=diversity_norm_distance,
            log_every=log_every,
            acquisition_strategy=acquisition_strategy,
        )


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


def _evaluate_ensemble_mean(ensemble: DeepEnsemble, dataset: TrajectoryDataset, batch_size: int) -> dict[str, float]:
    x_test, y_test = dataset.test_view()
    if x_test is None:
        return {}
    pred = ensemble.predict_mean(x_test, batch_size=batch_size)
    mse = float(np.mean((pred - y_test) ** 2))
    rmse = float(np.sqrt(mse))
    return {"mse": mse, "rmse": rmse}


def run_qbc_loop(
    dataset: TrajectoryDataset,
    bounds: np.ndarray,
    simulator: TrajectorySimulator,
    qbc_config: QBCConfig,
    config: Any,
    history: list[QBCRoundSummary] | None = None,
    initial_ensemble: DeepEnsemble | None = None,
    post_train_callback: Any | None = None,
    round_callback: Any | None = None,
) -> tuple[TrajectoryDataset, DeepEnsemble, list[QBCRoundSummary]]:
    """Run iterative QBC with strict no-solver candidate scoring."""
    history = [] if history is None else history
    ensemble: DeepEnsemble | None = None

    print(
        "[qbc] Start loop | "
        f"start_round={qbc_config.start_round} total_rounds={qbc_config.T} "
        f"M={qbc_config.M} P={qbc_config.P} K={qbc_config.K} "
        f"candidate_method={qbc_config.candidate_method} acquisition={qbc_config.acquisition_strategy} "
        f"diversify={qbc_config.diversify} disagreement_metric={qbc_config.disagreement_metric} "
        f"log_every={qbc_config.log_every}"
    )

    for round_idx in range(qbc_config.start_round, qbc_config.T):
        t_round_start = time.perf_counter()
        log_this_round = _should_log_round(
            round_idx=round_idx, total_rounds=qbc_config.T, log_every=qbc_config.log_every
        )
        if log_this_round:
            print(f"[qbc][round {round_idx:03d}] train_size={dataset.n_train} -> training ensemble")
        t_train_start = time.perf_counter()
        if round_idx == qbc_config.start_round and initial_ensemble is not None:
            ensemble = initial_ensemble
            if log_this_round:
                print(f"[qbc][round {round_idx:03d}] using provided ensemble checkpoint")
        else:
            ensemble = train_ensemble(
                dataset=dataset,
                M=qbc_config.M,
                base_seed=qbc_config.base_seed + 1000 * round_idx,
                config=config,
            )
            if log_this_round:
                print(f"[qbc][round {round_idx:03d}] ensemble trained")
        train_seconds = float(time.perf_counter() - t_train_start)

        if post_train_callback is not None:
            post_train_callback(
                round_idx=round_idx,
                ensemble=ensemble,
                dataset=dataset,
            )

        t_cand_start = time.perf_counter()
        cand_seed = qbc_config.base_seed + 2000 * round_idx + 17
        x_cand = sample_initial_ics(method=qbc_config.candidate_method, n=qbc_config.P, bounds=bounds, seed=cand_seed)
        candidate_generation_seconds = float(time.perf_counter() - t_cand_start)
        if log_this_round:
            print(f"[qbc][round {round_idx:03d}] generated candidates: {x_cand.shape[0]}")

        t_acq_start = time.perf_counter()
        hybrid_diag: dict[str, np.ndarray | float | int] = {}
        if qbc_config.acquisition_strategy == "qbc_marker_hybrid":
            preds = ensemble.predict_all(x_cand, batch_size=qbc_config.batch_size)
            U = score_disagreement(metric=qbc_config.disagreement_metric, predictions=preds)
            pred_mean = np.mean(preds, axis=0)
            selected_idx, hybrid_diag = select_qbc_marker_hybrid(
                train_trajs=dataset.train_trajs,
                candidate_pred_mean_trajs=pred_mean,
                uncertainty_scores=U,
                time_grid=dataset.time_grid if dataset.time_grid is not None else simulator.t_eval.astype(np.float32),
                state_names=list(getattr(simulator, "state_names", [])) or None,
                k_select=qbc_config.K,
                config=config,
            )
        else:
            U = score_disagreement(
                ensemble=ensemble,
                candidate_ics=x_cand,
                batch_size=qbc_config.batch_size,
                metric=qbc_config.disagreement_metric,
            )
            if qbc_config.diversify:
                selected_idx = acquire_diverse(
                    candidate_ics=x_cand,
                    scores=U,
                    K=qbc_config.K,
                    preselect_factor=qbc_config.diversity_preselect_factor,
                    uncertainty_weight=qbc_config.diversity_uncertainty_weight,
                    distance_weight=qbc_config.diversity_distance_weight,
                    normalize_uncertainty=qbc_config.diversity_norm_uncertainty,
                    normalize_distance=qbc_config.diversity_norm_distance,
                )
            else:
                selected_idx = acquire_topk(scores=U, K=qbc_config.K)
        selected_scores = np.asarray(U)[selected_idx]
        acquisition_seconds = float(time.perf_counter() - t_acq_start)
        if log_this_round:
            print(
                f"[qbc][round {round_idx:03d}] selection complete | "
                f"mean_score={float(np.mean(U)):.6f} max_score={float(np.max(U)):.6f}"
            )

        model_train_size = dataset.n_train
        eval_metrics = {}
        t_eval_start = time.perf_counter()
        if dataset.n_test > 0:
            eval_metrics = _evaluate_ensemble_mean(ensemble=ensemble, dataset=dataset, batch_size=qbc_config.batch_size)
        eval_seconds = float(time.perf_counter() - t_eval_start)

        summary = QBCRoundSummary(
            round_idx=round_idx,
            train_size=model_train_size,
            selected_indices=selected_idx.copy(),
            disagreement_metric=qbc_config.disagreement_metric,
            mean_score=float(np.mean(U)),
            max_score=float(np.max(U)),
            p90_score=float(np.percentile(U, 90)),
            selected_mean_score=float(np.mean(selected_scores)),
            selected_min_score=float(np.min(selected_scores)),
            selected_max_score=float(np.max(selected_scores)),
            mean_disagreement=float(np.mean(U)),
            max_disagreement=float(np.max(U)),
            mean_hybrid_score=(
                float(np.mean(hybrid_diag["hybrid_score"])) if "hybrid_score" in hybrid_diag else None
            ),
            mean_selected_diversity=(
                float(np.mean(np.asarray(hybrid_diag["hybrid_diversity"])[selected_idx]))
                if "hybrid_diversity" in hybrid_diag
                else None
            ),
            mean_selected_sparsity=(
                float(np.mean(np.asarray(hybrid_diag["hybrid_sparsity"])[selected_idx]))
                if "hybrid_sparsity" in hybrid_diag
                else None
            ),
            hybrid_marker_pca_components=(
                int(hybrid_diag["hybrid_marker_pca_components"])
                if "hybrid_marker_pca_components" in hybrid_diag
                else None
            ),
            eval_mse=eval_metrics.get("mse"),
            eval_rmse=eval_metrics.get("rmse"),
            train_seconds=train_seconds,
            candidate_generation_seconds=candidate_generation_seconds,
            candidate_simulation_seconds=0.0,
            acquisition_seconds=acquisition_seconds,
            selected_simulation_seconds=None,
            eval_seconds=eval_seconds,
            round_seconds=None,
        )
        history.append(summary)

        selected_ics = x_cand[selected_idx]
        t_sel_sim_start = time.perf_counter()
        _, selected_trajs = simulator.simulate_trajectory(selected_ics)
        selected_simulation_seconds = float(time.perf_counter() - t_sel_sim_start)
        dataset.append(selected_ics, selected_trajs)
        summary.selected_simulation_seconds = selected_simulation_seconds
        summary.round_seconds = float(time.perf_counter() - t_round_start)
        if log_this_round:
            print(
                f"[qbc][round {round_idx:03d}] appended {selected_ics.shape[0]} trajectories | "
                f"new_train_size={dataset.n_train} "
                f"(train={train_seconds:.3f}s, acq={acquisition_seconds:.3f}s, "
                f"sel_sim={selected_simulation_seconds:.3f}s, total={summary.round_seconds:.3f}s)"
            )
        if round_callback is not None:
            round_callback(
                summary=summary,
                x_cand=x_cand,
                scores=U,
                **hybrid_diag,
                selected_idx=selected_idx,
                selected_ics=selected_ics,
                selected_trajs=selected_trajs,
                dataset=dataset,
            )

    if ensemble is None:
        ensemble = train_ensemble(dataset=dataset, M=qbc_config.M, base_seed=qbc_config.base_seed, config=config)
    print(f"[qbc] Loop finished | final_train_size={dataset.n_train} rounds_executed={len(history)}")
    return dataset, ensemble, history


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
    log_every = int(_cfg_get(config, "active.log_every", 5))

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
    if log_every < 1:
        raise ValueError("active.log_every must be >= 1.")

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
        f"w_div={w_div:.3f} w_sparse={w_sparse:.3f} log_every={log_every}"
    )

    for round_idx in range(T):
        t_round_start = time.perf_counter()
        log_this_round = _should_log_round(round_idx=round_idx, total_rounds=T, log_every=log_every)
        if log_this_round:
            print(f"[marker][round {round_idx:03d}] train_size={dataset.n_train} -> generating candidates")
        t_cand_start = time.perf_counter()
        cand_seed = base_seed + 3000 * round_idx + 23
        x_cand = sample_initial_ics(method=candidate_method, n=P, bounds=bounds, seed=cand_seed)
        candidate_generation_seconds = float(time.perf_counter() - t_cand_start)
        t_cand_sim_start = time.perf_counter()
        _, y_cand = simulator.simulate_trajectory(x_cand)
        candidate_simulation_seconds = float(time.perf_counter() - t_cand_sim_start)
        if log_this_round:
            print(f"[marker][round {round_idx:03d}] simulated candidates: {x_cand.shape[0]}")

        y_train = dataset.train_trajs
        model_train_size = dataset.n_train

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

        mean, std = standardize_fit(m_train)
        m_train_std = standardize_apply(m_train, mean, std)
        m_cand_std = standardize_apply(m_cand, mean, std)

        pca_center, pca_components, n_comp = fit_pca(m_train_std, explained_var_ratio=explained)
        z_train = pca_transform(m_train_std, pca_center, pca_components)
        z_cand = pca_transform(m_cand_std, pca_center, pca_components)

        d_ct = cdist(z_cand, z_train, metric="euclidean")
        diversity = d_ct.min(axis=1)

        kk = min(max(1, k_density), z_train.shape[0])
        sparse = np.partition(d_ct, kk - 1, axis=1)[:, :kk].mean(axis=1)

        base_scores = w_div * normalize01(diversity) + w_sparse * normalize01(sparse)
        preselect_size = min(int(base_scores.shape[0]), max(K, int(K * preselect_factor)))
        selected_idx = greedy_select(
            embedding=z_cand,
            scores=base_scores,
            k=K,
            preselect_factor=preselect_factor,
            alpha_score=alpha_score,
        )
        acquisition_seconds = float(time.perf_counter() - t_acq_start)

        selected_ics = x_cand[selected_idx]
        selected_trajs = y_cand[selected_idx]
        dataset.append(selected_ics, selected_trajs)
        if log_this_round:
            print(
                f"[marker][round {round_idx:03d}] selected={selected_ics.shape[0]} "
                f"mean_score={float(np.mean(base_scores)):.6f} max_score={float(np.max(base_scores)):.6f} "
                f"new_train_size={dataset.n_train} "
                f"(cand={candidate_generation_seconds:.3f}s, cand_sim={candidate_simulation_seconds:.3f}s, "
                f"acq={acquisition_seconds:.3f}s, total={float(time.perf_counter() - t_round_start):.3f}s)"
            )

        summary = MarkerRoundSummary(
            round_idx=round_idx,
            train_size=model_train_size,
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
