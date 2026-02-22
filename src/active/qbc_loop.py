"""Iterative Deep Ensemble QBC loop for trajectory surrogate modeling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.active.acquisition import acquire_diverse, acquire_topk
from src.active.disagreement import score_disagreement
from src.active.ensemble import DeepEnsemble, train_ensemble
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
    mean_disagreement: float
    max_disagreement: float
    eval_mse: float | None = None
    eval_rmse: float | None = None


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

    for round_idx in range(start_round, T):
        if round_idx == start_round and initial_ensemble is not None:
            ensemble = initial_ensemble
        else:
            ensemble = train_ensemble(dataset=dataset, M=M, base_seed=base_seed + 1000 * round_idx, config=config)

        if post_train_callback is not None:
            post_train_callback(
                round_idx=round_idx,
                ensemble=ensemble,
                dataset=dataset,
            )

        # Candidate generation: IC-only, no solver calls.
        cand_seed = base_seed + 2000 * round_idx + 17
        x_cand = sample_initial_ics(method=candidate_method, n=P, bounds=bounds, seed=cand_seed)

        # Disagreement scoring: model predictions only.
        U = score_disagreement(ensemble=ensemble, candidate_ics=x_cand, batch_size=batch_size)

        if diversify:
            selected_idx = acquire_diverse(candidate_ics=x_cand, scores=U, K=K)
        else:
            selected_idx = acquire_topk(scores=U, K=K)

        model_train_size = dataset.n_train
        eval_metrics = {}
        if dataset.n_test > 0:
            eval_metrics = _evaluate_ensemble_mean(ensemble=ensemble, dataset=dataset, batch_size=batch_size)

        summary = QBCRoundSummary(
            round_idx=round_idx,
            train_size=model_train_size,
            selected_indices=selected_idx.copy(),
            mean_disagreement=float(np.mean(U)),
            max_disagreement=float(np.max(U)),
            eval_mse=eval_metrics.get("mse"),
            eval_rmse=eval_metrics.get("rmse"),
        )
        history.append(summary)

        selected_ics = x_cand[selected_idx]
        _, selected_trajs = simulator.simulate_trajectory(selected_ics)
        dataset.append(selected_ics, selected_trajs)
        if round_callback is not None:
            round_callback(
                summary=summary,
                x_cand=x_cand,
                scores=U,
                selected_idx=selected_idx,
                selected_ics=selected_ics,
                selected_trajs=selected_trajs,
                dataset=dataset,
            )

    if ensemble is None:
        ensemble = train_ensemble(dataset=dataset, M=M, base_seed=base_seed, config=config)
    return dataset, ensemble, history
