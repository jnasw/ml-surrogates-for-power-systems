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
    meta: dict[str, Any] = {
        "Adaptive sampling method": "qbc",
        "Adaptive candidate sampling method": str(getattr(config.active, "candidate_method", "sobol")),
        "Adaptive diversify": bool(getattr(config.active, "diversify", True)),
        "Adaptive initial sample size (qbc_n0)": int(n0),
        "Adaptive held-out test size (qbc_n_test)": int(n_test),
        "Adaptive committee size (qbc_M)": int(M),
        "Adaptive candidate pool size (qbc_P)": int(P),
        "Adaptive selected per round (qbc_K)": int(K),
        "Adaptive rounds (qbc_T)": int(T),
        "Adaptive final training trajectories": int(final_train_size),
    }
    if source_run_dir is not None:
        meta["Adaptive source run directory"] = str(source_run_dir)
    return meta

