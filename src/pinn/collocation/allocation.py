"""Dynamic collocation budget allocation across multiple train-time pools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.pinn.losses import PinnLossBreakdown
from src.training.runtime import cfg_get


@dataclass(frozen=True)
class PoolAllocationSpec:
    name: str
    initial_fraction: float
    min_fraction: float
    max_fraction: float


class DynamicAllocationController:
    """Budget controller that shifts rows between residual and IC pools."""

    def __init__(self, *, config: Any, total_rows: int) -> None:
        self._config = config
        self._total_rows = int(total_rows)
        self._enabled = bool(cfg_get(config, "pinn.collocation.multi_pool.allocation.enabled", False))
        self._method = str(cfg_get(config, "pinn.collocation.multi_pool.allocation.method", "loss_ratio")).strip().lower()
        self._update_interval = int(cfg_get(config, "pinn.collocation.multi_pool.allocation.update_interval_epochs", 1))
        self._smoothing = float(cfg_get(config, "pinn.collocation.multi_pool.allocation.smoothing", 0.8))
        if not (0.0 <= self._smoothing < 1.0):
            raise ValueError("pinn.collocation.multi_pool.allocation.smoothing must be in [0, 1).")

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def update_interval(self) -> int:
        return max(1, self._update_interval)

    def initial_budgets(self) -> dict[str, int]:
        fractions = {
            "residual": _pool_spec(self._config, "residual").initial_fraction,
            "ic_constraint": _pool_spec(self._config, "ic_constraint").initial_fraction,
        }
        return _fractions_to_budgets(total_rows=self._total_rows, fractions=fractions)

    def updated_budgets(
        self,
        *,
        previous_budgets: dict[str, int],
        losses: PinnLossBreakdown | None,
    ) -> dict[str, int]:
        if not self._enabled or losses is None:
            return dict(previous_budgets)
        if self._method != "loss_ratio":
            raise NotImplementedError("Supported multi-pool allocation methods: loss_ratio")

        residual_spec = _pool_spec(self._config, "residual")
        ic_spec = _pool_spec(self._config, "ic_constraint")
        residual_score = max(float(losses.physics.detach().cpu().item()), 1e-12)
        ic_score = max(float(losses.ic.detach().cpu().item()), 1e-12)
        total_score = residual_score + ic_score

        target_residual = residual_score / total_score
        target_ic = ic_score / total_score
        target_residual = _clamp_fraction(target_residual, residual_spec.min_fraction, residual_spec.max_fraction)
        target_ic = _clamp_fraction(target_ic, ic_spec.min_fraction, ic_spec.max_fraction)
        normalized = _normalize_two_pool_fractions(
            residual=target_residual,
            ic_constraint=target_ic,
            residual_bounds=(residual_spec.min_fraction, residual_spec.max_fraction),
            ic_bounds=(ic_spec.min_fraction, ic_spec.max_fraction),
        )

        prev_total = max(1, sum(int(v) for v in previous_budgets.values()))
        prev_residual_fraction = float(previous_budgets.get("residual", 0)) / float(prev_total)
        smoothed_residual = self._smoothing * prev_residual_fraction + (1.0 - self._smoothing) * normalized["residual"]
        smoothed_residual = _clamp_fraction(smoothed_residual, residual_spec.min_fraction, residual_spec.max_fraction)
        smoothed_ic = 1.0 - smoothed_residual
        smoothed = _normalize_two_pool_fractions(
            residual=smoothed_residual,
            ic_constraint=smoothed_ic,
            residual_bounds=(residual_spec.min_fraction, residual_spec.max_fraction),
            ic_bounds=(ic_spec.min_fraction, ic_spec.max_fraction),
        )
        return _fractions_to_budgets(total_rows=self._total_rows, fractions=smoothed)


def _pool_spec(config: Any, pool_name: str) -> PoolAllocationSpec:
    prefix = f"pinn.collocation.multi_pool.pools.{pool_name}"
    initial_fraction = float(cfg_get(config, f"{prefix}.initial_fraction", 0.5))
    min_fraction = float(cfg_get(config, f"{prefix}.min_fraction", 0.1))
    max_fraction = float(cfg_get(config, f"{prefix}.max_fraction", 0.9))
    if not (0.0 <= min_fraction <= initial_fraction <= max_fraction <= 1.0):
        raise ValueError(
            f"{prefix}.min_fraction <= initial_fraction <= max_fraction must hold and all values must be in [0, 1]."
        )
    return PoolAllocationSpec(
        name=pool_name,
        initial_fraction=initial_fraction,
        min_fraction=min_fraction,
        max_fraction=max_fraction,
    )


def _clamp_fraction(value: float, min_fraction: float, max_fraction: float) -> float:
    return float(max(min_fraction, min(max_fraction, value)))


def _normalize_two_pool_fractions(
    *,
    residual: float,
    ic_constraint: float,
    residual_bounds: tuple[float, float],
    ic_bounds: tuple[float, float],
) -> dict[str, float]:
    residual = float(residual)
    ic_constraint = float(ic_constraint)
    total = residual + ic_constraint
    if total <= 0.0:
        residual = 0.5
        ic_constraint = 0.5
        total = 1.0
    residual /= total
    ic_constraint /= total
    residual = _clamp_fraction(residual, *residual_bounds)
    ic_constraint = _clamp_fraction(ic_constraint, *ic_bounds)
    total = residual + ic_constraint
    residual /= total
    ic_constraint /= total
    return {"residual": residual, "ic_constraint": ic_constraint}


def _fractions_to_budgets(*, total_rows: int, fractions: dict[str, float]) -> dict[str, int]:
    total_rows = int(total_rows)
    residual_rows = max(1, int(round(total_rows * float(fractions["residual"]))))
    ic_rows = max(1, total_rows - residual_rows)
    if residual_rows + ic_rows != total_rows:
        ic_rows = total_rows - residual_rows
    if ic_rows <= 0:
        ic_rows = 1
        residual_rows = max(1, total_rows - 1)
    return {"residual": residual_rows, "ic_constraint": ic_rows}
