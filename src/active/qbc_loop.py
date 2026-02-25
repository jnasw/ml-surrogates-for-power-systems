"""Iterative Deep Ensemble QBC loop for trajectory surrogate modeling."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.active.acquisition import acquire_diverse, acquire_topk
from src.active.disagreement import score_disagreement
from src.active.ensemble import DeepEnsemble, train_ensemble
from src.active.hybrid_acquisition import select_qbc_marker_hybrid
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
    M: int,
    P: int,
    K: int,
    T: int,
    base_seed: int,
    config: Any,
    start_round: int = 0,
    history: list[QBCRoundSummary] | None = None,
    initial_ensemble: DeepEnsemble | None = None,
    post_train_callback: Any | None = None,
    round_callback: Any | None = None,
) -> tuple[TrajectoryDataset, DeepEnsemble, list[QBCRoundSummary]]:
    """Run iterative QBC with strict no-solver candidate scoring."""
    history = [] if history is None else history
    ensemble: DeepEnsemble | None = None

    candidate_method = str(_cfg_get(config, "active.candidate_method", "sobol"))
    diversify = bool(_cfg_get(config, "active.diversify", True))
    batch_size = int(_cfg_get(config, "active.predict_batch_size", 2048))
    disagreement_metric = str(_cfg_get(config, "active.disagreement.metric", "variance_mean"))
    diversity_preselect_factor = int(_cfg_get(config, "active.diversity.preselect_factor", 5))
    diversity_uncertainty_weight = float(_cfg_get(config, "active.diversity.uncertainty_weight", 0.7))
    diversity_distance_weight = float(_cfg_get(config, "active.diversity.distance_weight", 0.3))
    diversity_norm_uncertainty = bool(_cfg_get(config, "active.diversity.normalize_uncertainty", True))
    diversity_norm_distance = bool(_cfg_get(config, "active.diversity.normalize_distance", True))
    acquisition_strategy = str(_cfg_get(config, "active.acquisition_strategy", "qbc_only")).lower()
    if acquisition_strategy not in {"qbc_only", "qbc_marker_hybrid"}:
        raise ValueError("active.acquisition_strategy must be one of: ['qbc_only', 'qbc_marker_hybrid']")
    print(
        "[qbc] Start loop | "
        f"start_round={start_round} total_rounds={T} M={M} P={P} K={K} "
        f"candidate_method={candidate_method} acquisition={acquisition_strategy} diversify={diversify} "
        f"disagreement_metric={disagreement_metric}"
    )

    for round_idx in range(start_round, T):
        t_round_start = time.perf_counter()
        print(f"[qbc][round {round_idx:03d}] train_size={dataset.n_train} -> training ensemble")
        t_train_start = time.perf_counter()
        if round_idx == start_round and initial_ensemble is not None:
            ensemble = initial_ensemble
            print(f"[qbc][round {round_idx:03d}] using provided ensemble checkpoint")
        else:
            ensemble = train_ensemble(dataset=dataset, M=M, base_seed=base_seed + 1000 * round_idx, config=config)
            print(f"[qbc][round {round_idx:03d}] ensemble trained")
        train_seconds = float(time.perf_counter() - t_train_start)

        if post_train_callback is not None:
            post_train_callback(
                round_idx=round_idx,
                ensemble=ensemble,
                dataset=dataset,
            )

        # Candidate generation: IC-only, no solver calls.
        t_cand_start = time.perf_counter()
        cand_seed = base_seed + 2000 * round_idx + 17
        x_cand = sample_initial_ics(method=candidate_method, n=P, bounds=bounds, seed=cand_seed)
        candidate_generation_seconds = float(time.perf_counter() - t_cand_start)
        print(f"[qbc][round {round_idx:03d}] generated candidates: {x_cand.shape[0]}")

        # Disagreement scoring: model predictions only.
        t_acq_start = time.perf_counter()
        hybrid_diag: dict[str, np.ndarray | float | int] = {}
        if acquisition_strategy == "qbc_marker_hybrid":
            preds = ensemble.predict_all(x_cand, batch_size=batch_size)  # (M, N, T, S)
            U = score_disagreement(metric=disagreement_metric, predictions=preds)
            pred_mean = np.mean(preds, axis=0)
            selected_idx, hybrid_diag = select_qbc_marker_hybrid(
                train_trajs=dataset.train_trajs,
                candidate_pred_mean_trajs=pred_mean,
                uncertainty_scores=U,
                time_grid=dataset.time_grid if dataset.time_grid is not None else simulator.t_eval.astype(np.float32),
                state_names=list(getattr(simulator, "state_names", [])) or None,
                k_select=K,
                config=config,
            )
        else:
            U = score_disagreement(
                ensemble=ensemble,
                candidate_ics=x_cand,
                batch_size=batch_size,
                metric=disagreement_metric,
            )
            if diversify:
                selected_idx = acquire_diverse(
                    candidate_ics=x_cand,
                    scores=U,
                    K=K,
                    preselect_factor=diversity_preselect_factor,
                    uncertainty_weight=diversity_uncertainty_weight,
                    distance_weight=diversity_distance_weight,
                    normalize_uncertainty=diversity_norm_uncertainty,
                    normalize_distance=diversity_norm_distance,
                )
            else:
                selected_idx = acquire_topk(scores=U, K=K)
        selected_scores = np.asarray(U)[selected_idx]
        acquisition_seconds = float(time.perf_counter() - t_acq_start)
        print(
            f"[qbc][round {round_idx:03d}] selection complete | "
            f"mean_score={float(np.mean(U)):.6f} max_score={float(np.max(U)):.6f}"
        )

        model_train_size = dataset.n_train
        eval_metrics = {}
        t_eval_start = time.perf_counter()
        if dataset.n_test > 0:
            eval_metrics = _evaluate_ensemble_mean(ensemble=ensemble, dataset=dataset, batch_size=batch_size)
        eval_seconds = float(time.perf_counter() - t_eval_start)

        summary = QBCRoundSummary(
            round_idx=round_idx,
            train_size=model_train_size,
            selected_indices=selected_idx.copy(),
            disagreement_metric=disagreement_metric,
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
        ensemble = train_ensemble(dataset=dataset, M=M, base_seed=base_seed, config=config)
    print(f"[qbc] Loop finished | final_train_size={dataset.n_train} rounds_executed={len(history)}")
    return dataset, ensemble, history
