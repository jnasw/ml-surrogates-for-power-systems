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
from src.pinn.vrba import vrba_config_from_config


@dataclass(frozen=True)
class EpochPoolBatch:
    x_col: torch.Tensor
    x_col_weights: torch.Tensor | None
    x_init: torch.Tensor
    y_init: torch.Tensor
    x_init_weights: torch.Tensor | None


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
        self._vrba_config = vrba_config_from_config(config)
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
                    points_weight=self._initial_ic_weights(init_x=init_x),
                    target_rows=int(budgets["ic_constraint"]),
                ),
            },
            total_target_rows=int(total_target_rows),
        )
        self._last_observed_epoch = 0
        self._sync_residual_metadata()

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
        self._update_ic_weights(context=context)
        return self._build_epoch_batch(epoch=context.global_epoch)

    def handle_phase_boundary(self, *, context: CollocationStrategyContext) -> None:
        residual_points = self._residual_strategy.prepare_epoch_points(context=context)
        self._state.pools["residual"].points_x = residual_points
        self._state.pools["residual"].points_weight = self._residual_strategy.current_weights()
        self._sync_residual_metadata()
        self._update_ic_weights(context=context)

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
        x_init_weights = _sample_optional_tensor_rows(
            x=ic_pool.points_weight,
            reference_rows=ic_pool.points_x,
            target_rows=ic_pool.target_rows,
            seed=self._seed + int(epoch) * 17 + 2,
        )
        residual_pool.metadata["epoch_rows"] = int(x_col.shape[0])
        ic_pool.metadata["epoch_rows"] = int(x_init.shape[0])
        return EpochPoolBatch(
            x_col=x_col,
            x_col_weights=x_col_weights,
            x_init=x_init,
            y_init=y_init,
            x_init_weights=x_init_weights,
        )

    def _sync_residual_metadata(self) -> None:
        residual_state = getattr(self._residual_strategy, "state", None)
        if residual_state is None:
            return
        residual_pool = self._state.pools["residual"]
        residual_pool.metadata.update(dict(residual_state.metadata or {}))
        for key, value in (residual_state.metadata or {}).items():
            if isinstance(value, (bool, int, float, str)) or value is None:
                self._state.metadata[f"residual_{key}"] = value

    def _initial_ic_weights(self, *, init_x: torch.Tensor) -> torch.Tensor | None:
        if not self._ic_weighting_configured:
            return None
        lambda_init = 0.1 * float(self._vrba_config.lambda_max0)
        return torch.full(
            (int(init_x.shape[0]),),
            fill_value=lambda_init,
            dtype=torch.float64,
            device=init_x.device,
        )

    @property
    def _ic_weighting_configured(self) -> bool:
        return bool(
            self._vrba_config.enabled
            and self._vrba_config.adaptive_weighting
            and "ic" in set(self._vrba_config.target_sets)
        )

    def _update_ic_weights(self, *, context: CollocationStrategyContext) -> None:
        ic_pool = self._state.pools["ic_constraint"]
        weighting_frozen = bool(
            context.phase_is_full_batch and self._vrba_config.freeze_weighting_during_full_batch
        )
        weighting_enabled = bool(self._ic_weighting_configured and not weighting_frozen)
        ic_pool.metadata["vrba_weighting_configured"] = bool(self._ic_weighting_configured)
        ic_pool.metadata["vrba_weighting_enabled"] = bool(weighting_enabled)
        ic_pool.metadata["vrba_weighting_frozen"] = bool(weighting_frozen)
        ic_pool.metadata["vrba_phase_is_full_batch"] = bool(context.phase_is_full_batch)
        ic_pool.metadata["vrba_potential"] = str(self._vrba_config.potential)
        ic_pool.metadata["vrba_update_count"] = int(ic_pool.metadata.get("vrba_update_count", 0))
        for key in ("vrba_weighting_configured", "vrba_weighting_enabled", "vrba_weighting_frozen", "vrba_phase_is_full_batch", "vrba_potential", "vrba_update_count"):
            self._state.metadata[f"ic_constraint_{key}"] = ic_pool.metadata[key]
        if not weighting_enabled:
            ic_pool.points_weight = None
            return
        if not _should_update_local_weights(config=self._config, context=context):
            if ic_pool.points_weight is not None:
                self._state.metadata["ic_constraint_vrba_lambda_mean"] = float(ic_pool.points_weight.mean().item())
                self._state.metadata["ic_constraint_vrba_lambda_max"] = float(ic_pool.points_weight.max().item())
                self._state.metadata["ic_constraint_vrba_lambda_min"] = float(ic_pool.points_weight.min().item())
            return
        if context.model is None:
            raise ValueError("IC vRBA weighting requires model in the collocation strategy context.")
        with torch.enable_grad():
            init_pred = context.model(ic_pool.points_x)
        init_error = (init_pred - ic_pool.points_y).detach()
        scores = init_error.to(dtype=torch.float64).pow(2).sum(dim=1).sqrt()
        target = _vrba_target_distribution_local(
            scores=scores,
            potential=self._vrba_config.potential,
            temperature_scale=self._vrba_config.temperature_scale,
            temperature_floor=self._vrba_config.temperature_floor,
            step=int(ic_pool.metadata.get("vrba_update_count", 0)) + 1,
        )
        pool_rows = int(ic_pool.points_x.shape[0])
        lambda_current = ic_pool.points_weight
        if lambda_current is None:
            lambda_current = self._initial_ic_weights(init_x=ic_pool.points_x)
        lambda_max = min(
            float(self._vrba_config.lambda_cap),
            float(self._vrba_config.lambda_max0)
            + float(int(ic_pool.metadata.get("vrba_update_count", 0)) + 1) / float(self._vrba_config.update_interval_epochs),
        )
        gamma = max(0.0, min(1.0 - (float(self._vrba_config.eta) / max(lambda_max, 1.0e-12)), 0.999999))
        uniform = torch.full_like(target, fill_value=1.0 / float(pool_rows))
        mixed_target = float(self._vrba_config.phi) * target + (1.0 - float(self._vrba_config.phi)) * uniform
        eta_star = float(self._vrba_config.eta) / max(float(target.max().item()), 1.0e-12)
        lambda_next = gamma * lambda_current + eta_star * mixed_target
        ic_pool.points_weight = lambda_next
        update_count = int(ic_pool.metadata.get("vrba_update_count", 0)) + 1
        ic_pool.metadata["vrba_update_count"] = update_count
        ic_pool.metadata["vrba_lambda_mean"] = float(lambda_next.mean().item())
        ic_pool.metadata["vrba_lambda_max"] = float(lambda_next.max().item())
        ic_pool.metadata["vrba_lambda_min"] = float(lambda_next.min().item())
        ic_pool.metadata["vrba_gamma"] = float(gamma)
        ic_pool.metadata["vrba_eta_star"] = float(eta_star)
        for key in ("vrba_update_count", "vrba_lambda_mean", "vrba_lambda_max", "vrba_lambda_min", "vrba_gamma", "vrba_eta_star"):
            self._state.metadata[f"ic_constraint_{key}"] = ic_pool.metadata[key]


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


def _should_update_local_weights(*, config: Any, context: CollocationStrategyContext) -> bool:
    from src.pinn.collocation.strategies import _should_refresh  # local import to avoid circular import at module load

    return _should_refresh(config=config, context=context)


def _normalize_probabilities_local(weights: torch.Tensor) -> torch.Tensor:
    safe = weights.to(dtype=torch.float64).clamp_min(0.0)
    total = safe.sum()
    if float(total.item()) <= 0.0:
        safe = torch.ones_like(safe)
        total = safe.sum()
    return safe / total


def _vrba_target_distribution_local(
    *,
    scores: torch.Tensor,
    potential: str,
    temperature_scale: float,
    temperature_floor: float,
    step: int,
) -> torch.Tensor:
    safe_scores = scores.to(dtype=torch.float64).abs()
    if int(safe_scores.numel()) == 0:
        raise ValueError("vRBA target distribution requires at least one score.")
    if str(potential).strip().lower() == "quadratic":
        weighted = safe_scores
        if float(weighted.sum().item()) <= 0.0:
            weighted = torch.ones_like(weighted)
        return _normalize_probabilities_local(weighted)
    if str(potential).strip().lower() == "exponential":
        score_scale = float(safe_scores.mean().item())
        epsilon = max(float(temperature_floor), float(temperature_scale) * score_scale / max(int(step), 1))
        shifted = safe_scores - safe_scores.max()
        return _normalize_probabilities_local(torch.exp(shifted / max(epsilon, 1.0e-12)))
    raise ValueError("vRBA potential must be one of: quadratic, exponential")
