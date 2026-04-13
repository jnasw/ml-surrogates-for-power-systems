"""Mutable state containers for train-time collocation management."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class CollocationState:
    """Current collocation state owned by a strategy."""

    active_points: torch.Tensor
    refresh_count: int = 0
    last_refresh_epoch: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
