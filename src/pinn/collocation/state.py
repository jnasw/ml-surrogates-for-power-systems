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


@dataclass
class CollocationPoolState:
    """Current state for one named train-time point pool."""

    name: str
    role: str
    points_x: torch.Tensor
    points_y: torch.Tensor | None = None
    target_rows: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiPoolCollocationState:
    """Combined residual/constraint pool state and allocation history."""

    pools: dict[str, CollocationPoolState]
    total_target_rows: int
    allocation_step: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
