"""Train-time collocation abstractions for PINN sampling strategies."""

from src.pinn.collocation.factory import build_collocation_manager
from src.pinn.collocation.manager import EpochPoolBatch, MultiPoolCollocationManager
from src.pinn.collocation.scoring import score_collocation_points
from src.pinn.collocation.state import CollocationPoolState, CollocationState, MultiPoolCollocationState
from src.pinn.collocation.strategies import (
    CollocationStrategy,
    CollocationStrategyContext,
    ResidualAdaptiveRefinementDistributionStrategy,
    ResidualAdaptiveRefinementGreedyStrategy,
    ResidualAdaptiveDistributionStrategy,
    StaticCollocationStrategy,
    UniformResampleCollocationStrategy,
)

__all__ = [
    "CollocationPoolState",
    "CollocationState",
    "CollocationStrategy",
    "CollocationStrategyContext",
    "EpochPoolBatch",
    "MultiPoolCollocationManager",
    "MultiPoolCollocationState",
    "ResidualAdaptiveRefinementDistributionStrategy",
    "ResidualAdaptiveRefinementGreedyStrategy",
    "ResidualAdaptiveDistributionStrategy",
    "StaticCollocationStrategy",
    "UniformResampleCollocationStrategy",
    "build_collocation_manager",
    "score_collocation_points",
]
