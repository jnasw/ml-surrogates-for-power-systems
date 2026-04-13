"""Domain abstractions for train-time collocation point generation."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CollocationDomain:
    """Continuous sampling domain for collocation points.

    Phase 1 only introduces the abstraction. Concrete generators for this domain
    will be added in later phases once adaptive train-time sampling is enabled.
    """

    time_min: float
    time_max: float
    input_dim: int
    feature_bounds: torch.Tensor | None = None
