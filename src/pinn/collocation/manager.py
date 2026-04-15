"""Multi-pool collocation manager for residual and IC point ownership."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from src.pinn.collocation.allocation import DynamicAllocationController
from src.pinn.collocation.state import CollocationPoolState, MultiPoolCollocationState
from src.pinn.collocation.strategies import CollocationStrategy, CollocationStrategyContext
from src.pinn.losses import PinnLossBreakdown


@dataclass(frozen=True)
class EpochPoolBatch:
    x_col: torch.Tensor
    x_col_weights: torch.Tensor | None
    x_init: torch.Tensor
    y_init: torch.Tensor


class MultiPoolCollocationManager:
    """Owns train-time residual and IC pools with optional dynamic reallocation."""

    def __init__(
        self,
        *,
        residual_strategy: CollocationStrategy,
        init_x: torch.Tensor,
        init_y: torch.Tensor,
        config: Any,
        seed: int,
        total_target_rows: int,
        enabled: bool,
    ) -> None:
        self._residual_strategy = residual_strategy
        self._init_x = init_x
        self._init_y = init_y
        self._config = config
        self._seed = int(seed)
        self._enabled = bool(enabled)
        available_total_rows = int(residual_strategy.current_points().shape[0]) + int(init_x.shape[0])
        total_target_rows = min(int(total_target_rows), available_total_rows)
        self._controller = DynamicAllocationController(config=config, total_rows=total_target_rows)
        budgets = self._controller.initial_budgets() if self._enabled else {
            "residual": int(residual_strategy.current_points().shape[0]),
            "ic_constraint": int(init_x.shape[0]),
        }

        self._state = MultiPoolCollocationState(
            pools={
                "residual": CollocationPoolState(
                    name="residual",
                    role="residual",
                    points_x=residual_strategy.current_points(),
                    points_weight=residual_strategy.current_weights(),
                    target_rows=int(budgets["residual"]),
                ),
                "ic_constraint": CollocationPoolState(
                    name="ic_constraint",
                    role="ic_constraint",
                    points_x=init_x,
                    points_y=init_y,
                    target_rows=int(budgets["ic_constraint"]),
                ),
            },
            total_target_rows=int(total_target_rows),
        )
        self._last_observed_epoch = 0

    @property
    def state(self) -> MultiPoolCollocationState:
        return self._state

    def residual_points(self) -> torch.Tensor:
        return self._residual_strategy.current_points()

    def init_points(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._init_x, self._init_y

    def initial_epoch_batch(self) -> EpochPoolBatch:
        return self._build_epoch_batch(epoch=0)

    def prepare_epoch_batch(self, *, context: CollocationStrategyContext) -> EpochPoolBatch:
        residual_points = self._residual_strategy.prepare_epoch_points(context=context)
        self._state.pools["residual"].points_x = residual_points
        self._state.pools["residual"].points_weight = self._residual_strategy.current_weights()
        self._sync_residual_metadata()
        return self._build_epoch_batch(epoch=context.global_epoch)

    def handle_phase_boundary(self, *, context: CollocationStrategyContext) -> None:
        residual_points = self._residual_strategy.prepare_epoch_points(context=context)
        self._state.pools["residual"].points_x = residual_points
        self._state.pools["residual"].points_weight = self._residual_strategy.current_weights()
        self._sync_residual_metadata()

    def observe_epoch_losses(self, *, global_epoch: int, losses: PinnLossBreakdown | None) -> None:
        self._last_observed_epoch = int(global_epoch)
        if not self._enabled or losses is None:
            return
        if (int(global_epoch) % self._controller.update_interval) != 0:
            return
        previous = {name: pool.target_rows for name, pool in self._state.pools.items()}
        budgets = self._controller.updated_budgets(previous_budgets=previous, losses=losses)
        for pool_name, rows in budgets.items():
            self._state.pools[pool_name].target_rows = int(rows)
        self._state.allocation_step += 1
        self._state.metadata["last_allocation_epoch"] = int(global_epoch)
        self._state.metadata["last_budgets"] = {k: int(v) for k, v in budgets.items()}
        self._state.metadata["last_physics_loss"] = float(losses.physics.detach().cpu().item())
        self._state.metadata["last_ic_loss"] = float(losses.ic.detach().cpu().item())

    def _build_epoch_batch(self, *, epoch: int) -> EpochPoolBatch:
        residual_pool = self._state.pools["residual"]
        ic_pool = self._state.pools["ic_constraint"]
        x_col = _sample_tensor_rows(
            x=residual_pool.points_x,
            target_rows=residual_pool.target_rows,
            seed=self._seed + int(epoch) * 17 + 1,
        )
        x_col_weights = _sample_optional_tensor_rows(
            x=residual_pool.points_weight,
            reference_rows=residual_pool.points_x,
            target_rows=residual_pool.target_rows,
            seed=self._seed + int(epoch) * 17 + 1,
        )
        x_init, y_init = _sample_xy_rows(
            x=ic_pool.points_x,
            y=ic_pool.points_y,
            target_rows=ic_pool.target_rows,
            seed=self._seed + int(epoch) * 17 + 2,
        )
        residual_pool.metadata["epoch_rows"] = int(x_col.shape[0])
        ic_pool.metadata["epoch_rows"] = int(x_init.shape[0])
        return EpochPoolBatch(x_col=x_col, x_col_weights=x_col_weights, x_init=x_init, y_init=y_init)

    def _sync_residual_metadata(self) -> None:
        residual_state = getattr(self._residual_strategy, "state", None)
        if residual_state is None:
            return
        residual_pool = self._state.pools["residual"]
        residual_pool.metadata.update(dict(residual_state.metadata or {}))
        for key, value in (residual_state.metadata or {}).items():
            if isinstance(value, (bool, int, float, str)) or value is None:
                self._state.metadata[f"residual_{key}"] = value


def _sample_tensor_rows(*, x: torch.Tensor, target_rows: int, seed: int) -> torch.Tensor:
    total_rows = int(x.shape[0])
    if target_rows >= total_rows:
        return x
    rng = np.random.default_rng(int(seed))
    indices = rng.choice(total_rows, size=int(target_rows), replace=False)
    idx = torch.as_tensor(indices, dtype=torch.long, device=x.device)
    return x.index_select(0, idx)


def _sample_optional_tensor_rows(
    *,
    x: torch.Tensor | None,
    reference_rows: torch.Tensor,
    target_rows: int,
    seed: int,
) -> torch.Tensor | None:
    if x is None:
        return None
    total_rows = int(reference_rows.shape[0])
    if int(x.shape[0]) != total_rows:
        raise ValueError("Optional sampled tensor must align with reference_rows on axis 0.")
    if target_rows >= total_rows:
        return x
    rng = np.random.default_rng(int(seed))
    indices = rng.choice(total_rows, size=int(target_rows), replace=False)
    idx = torch.as_tensor(indices, dtype=torch.long, device=x.device)
    return x.index_select(0, idx)


def _sample_xy_rows(*, x: torch.Tensor, y: torch.Tensor, target_rows: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    total_rows = int(x.shape[0])
    if target_rows >= total_rows:
        return x, y
    rng = np.random.default_rng(int(seed))
    indices = rng.choice(total_rows, size=int(target_rows), replace=False)
    idx = torch.as_tensor(indices, dtype=torch.long, device=x.device)
    return x.index_select(0, idx), y.index_select(0, idx)
