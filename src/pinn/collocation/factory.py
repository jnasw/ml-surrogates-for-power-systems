"""Factory for collocation strategies."""

from __future__ import annotations

from typing import Any

import torch

from src.pinn.collocation.strategies import CollocationStrategy, StaticCollocationStrategy
from src.train.runtime import cfg_get


def build_collocation_strategy(*, initial_points: torch.Tensor, config: Any) -> CollocationStrategy:
    mode = str(cfg_get(config, "pinn.collocation.mode", "preprocessed")).strip().lower()
    strategy_name = str(cfg_get(config, "pinn.collocation.strategy", "static")).strip().lower()

    if mode != "preprocessed":
        raise NotImplementedError(
            "Train-time generated collocation is not implemented yet. "
            "Use pinn.collocation.mode='preprocessed' for Phase 1."
        )
    if strategy_name != "static":
        raise NotImplementedError(
            "Only pinn.collocation.strategy='static' is implemented in Phase 1."
        )
    return StaticCollocationStrategy(initial_points=initial_points, config=config)
