"""Train-time collocation abstractions for PINN sampling strategies."""

from src.pinn.collocation.factory import build_collocation_strategy
from src.pinn.collocation.scoring import score_collocation_points
from src.pinn.collocation.state import CollocationState
from src.pinn.collocation.strategies import (
    CollocationStrategy,
    CollocationStrategyContext,
    ResidualAdaptiveDistributionStrategy,
    StaticCollocationStrategy,
    UniformResampleCollocationStrategy,
)

__all__ = [
    "CollocationState",
    "CollocationStrategy",
    "CollocationStrategyContext",
    "ResidualAdaptiveDistributionStrategy",
    "StaticCollocationStrategy",
    "UniformResampleCollocationStrategy",
    "build_collocation_strategy",
    "score_collocation_points",
]
