"""Collocation strategy abstractions used by PINN training."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from src.pinn.collocation.state import CollocationState
from src.train.runtime import cfg_get


@dataclass(frozen=True)
class CollocationStrategyContext:
    """Epoch/phase context passed to collocation strategies."""

    global_epoch: int
    phase_name: str
    phase_allows_sampling: bool
    model: nn.Module | None
    ode_model: Any | None
    formulation: str


class CollocationStrategy(ABC):
    """Base class for train-time collocation point ownership."""

    def __init__(self, *, initial_points: torch.Tensor, config: Any) -> None:
        self._config = config
        self._state = CollocationState(active_points=initial_points)

    @property
    def state(self) -> CollocationState:
        return self._state

    def current_points(self) -> torch.Tensor:
        return self._state.active_points

    def prepare_epoch_points(self, *, context: CollocationStrategyContext) -> torch.Tensor:
        self.maybe_refresh(context=context)
        return self._points_for_epoch(context=context)

    def maybe_refresh(self, *, context: CollocationStrategyContext) -> None:
        """Refresh strategy state before an epoch if needed."""
        return None

    @abstractmethod
    def _points_for_epoch(self, *, context: CollocationStrategyContext) -> torch.Tensor:
        """Return the collocation tensor that should be used for this epoch."""


class StaticCollocationStrategy(CollocationStrategy):
    """Behavior-preserving static collocation strategy.

    This keeps the active collocation tensor fixed and optionally performs the
    existing epoch-level row subsampling during phases that allow sampling.
    """

    def _points_for_epoch(self, *, context: CollocationStrategyContext) -> torch.Tensor:
        x_col = self.current_points()
        if not context.phase_allows_sampling:
            return x_col
        if not bool(_cfg_get_collocation(config=self._config, key="sampling.enabled", legacy_key="pinn.collocation_sampling.enabled", default=False)):
            return x_col
        return _sample_tensor_rows(
            x=x_col,
            rows_per_epoch_cfg=_cfg_get_collocation(
                config=self._config,
                key="sampling.rows_per_epoch",
                legacy_key="pinn.collocation_sampling.rows_per_epoch",
                default=None,
            ),
            fraction_per_epoch_cfg=_cfg_get_collocation(
                config=self._config,
                key="sampling.fraction_per_epoch",
                legacy_key="pinn.collocation_sampling.fraction_per_epoch",
                default=None,
            ),
            cfg_prefix="pinn.collocation.sampling",
        )


def _cfg_get_collocation(*, config: Any, key: str, legacy_key: str, default: Any) -> Any:
    value = cfg_get(config, f"pinn.collocation.{key}", None)
    if value in (None, "null"):
        return cfg_get(config, legacy_key, default)
    return value


def _sample_tensor_rows(
    *,
    x: torch.Tensor,
    rows_per_epoch_cfg: Any,
    fraction_per_epoch_cfg: Any,
    cfg_prefix: str,
) -> torch.Tensor:
    total_rows = int(x.shape[0])

    target_rows: int | None = None
    if rows_per_epoch_cfg not in (None, "null"):
        target_rows = int(rows_per_epoch_cfg)
    elif fraction_per_epoch_cfg not in (None, "null"):
        fraction = float(fraction_per_epoch_cfg)
        if not (0.0 < fraction <= 1.0):
            raise ValueError(f"{cfg_prefix}.fraction_per_epoch must be in (0, 1].")
        target_rows = max(1, int(round(total_rows * fraction)))

    if target_rows is None or target_rows >= total_rows:
        return x
    if target_rows <= 0:
        raise ValueError(f"{cfg_prefix}.rows_per_epoch must be > 0.")

    indices_np = np.random.choice(total_rows, size=target_rows, replace=False)
    indices = torch.as_tensor(indices_np, device=x.device, dtype=torch.long)
    return x.index_select(0, indices)
