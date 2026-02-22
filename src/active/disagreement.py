"""Disagreement scoring for QBC candidate selection."""

from __future__ import annotations

import numpy as np

from src.active.ensemble import DeepEnsemble


def score_disagreement(
    ensemble: DeepEnsemble,
    candidate_ics: np.ndarray,
    batch_size: int = 2048,
) -> np.ndarray:
    """Return one uncertainty score per candidate IC.

    Score is mean predictive variance across ensemble/time/state.
    """
    preds = ensemble.predict_all(candidate_ics, batch_size=batch_size)  # (M, N, T, S)
    var = np.var(preds, axis=0)  # (N, T, S)
    return var.mean(axis=(1, 2))
