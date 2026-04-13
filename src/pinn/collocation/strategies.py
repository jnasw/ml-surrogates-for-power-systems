"""Collocation strategy abstractions used by PINN training."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from src.pinn.collocation.domain import CollocationDomain, sample_collocation_points
from src.pinn.collocation.scoring import score_collocation_points
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
        return _maybe_subsample_epoch_points(x_col=x_col, config=self._config, phase_allows_sampling=context.phase_allows_sampling)


class UniformResampleCollocationStrategy(StaticCollocationStrategy):
    """Random-R: periodically replace the full active set with uniform samples."""

    def __init__(
        self,
        *,
        initial_points: torch.Tensor,
        config: Any,
        domain: CollocationDomain,
        sampler: str,
        seed: int,
        active_points: int,
    ) -> None:
        super().__init__(initial_points=initial_points, config=config)
        self._domain = domain
        self._sampler = str(sampler)
        self._seed = int(seed)
        self._active_points = int(active_points)

    def maybe_refresh(self, *, context: CollocationStrategyContext) -> None:
        if not _should_refresh(config=self._config, context=context):
            return
        self._state.active_points = sample_collocation_points(
            domain=self._domain,
            n=self._active_points,
            method=self._sampler,
            seed=self._seed + self._state.refresh_count + int(context.global_epoch),
            dtype=self.current_points().dtype,
            device=self.current_points().device,
        )
        self._state.refresh_count += 1
        self._state.last_refresh_epoch = int(context.global_epoch)
        self._state.metadata["last_strategy"] = "random_r"


class ResidualAdaptiveDistributionStrategy(StaticCollocationStrategy):
    """RAD: periodically replace the active set using residual-weighted sampling."""

    def __init__(
        self,
        *,
        initial_points: torch.Tensor,
        config: Any,
        domain: CollocationDomain,
        sampler: str,
        seed: int,
        active_points: int,
        candidate_points: int,
        rad_k: float,
        rad_c: float,
        score_norm: str,
    ) -> None:
        super().__init__(initial_points=initial_points, config=config)
        self._domain = domain
        self._sampler = str(sampler)
        self._seed = int(seed)
        self._active_points = int(active_points)
        self._candidate_points = int(candidate_points)
        self._rad_k = float(rad_k)
        self._rad_c = float(rad_c)
        self._score_norm = str(score_norm)

    def maybe_refresh(self, *, context: CollocationStrategyContext) -> None:
        if not _should_refresh(config=self._config, context=context):
            return
        if context.model is None or context.ode_model is None:
            raise ValueError("RAD requires model and ode_model in the collocation strategy context.")
        if self._candidate_points < self._active_points:
            raise ValueError("pinn.collocation.candidate_points must be >= pinn.collocation.active_points for RAD.")

        candidates = sample_collocation_points(
            domain=self._domain,
            n=self._candidate_points,
            method=self._sampler,
            seed=self._seed + self._state.refresh_count + int(context.global_epoch),
            dtype=self.current_points().dtype,
            device=self.current_points().device,
        )
        scores = score_collocation_points(
            model=context.model,
            x=candidates,
            ode_model=context.ode_model,
            formulation=context.formulation,
            norm=self._score_norm,
        ).detach()

        probabilities = _rad_probabilities(
            scores=scores,
            k=self._rad_k,
            c=self._rad_c,
        )
        rng = np.random.default_rng(self._seed + self._state.refresh_count + int(context.global_epoch))
        selected_ids = rng.choice(
            self._candidate_points,
            size=self._active_points,
            replace=False,
            p=probabilities,
        )
        indices = torch.as_tensor(selected_ids, dtype=torch.long, device=candidates.device)
        self._state.active_points = candidates.index_select(0, indices)
        self._state.refresh_count += 1
        self._state.last_refresh_epoch = int(context.global_epoch)
        self._state.metadata["last_strategy"] = "rad"
        self._state.metadata["last_score_mean"] = float(scores.mean().item())
        self._state.metadata["last_score_max"] = float(scores.max().item())


def _maybe_subsample_epoch_points(*, x_col: torch.Tensor, config: Any, phase_allows_sampling: bool) -> torch.Tensor:
    if not phase_allows_sampling:
        return x_col
    if not bool(
        _cfg_get_collocation(
            config=config,
            key="sampling.enabled",
            legacy_key="pinn.collocation_sampling.enabled",
            default=False,
        )
    ):
        return x_col
    return _sample_tensor_rows(
        x=x_col,
        rows_per_epoch_cfg=_cfg_get_collocation(
            config=config,
            key="sampling.rows_per_epoch",
            legacy_key="pinn.collocation_sampling.rows_per_epoch",
            default=None,
        ),
        fraction_per_epoch_cfg=_cfg_get_collocation(
            config=config,
            key="sampling.fraction_per_epoch",
            legacy_key="pinn.collocation_sampling.fraction_per_epoch",
            default=None,
        ),
        cfg_prefix="pinn.collocation.sampling",
    )


def _should_refresh(*, config: Any, context: CollocationStrategyContext) -> bool:
    if not context.phase_allows_sampling:
        return False
    period = int(_cfg_get_collocation(config=config, key="refresh_period_epochs", legacy_key="pinn.collocation.refresh_period_epochs", default=0))
    if period <= 0:
        return False
    epoch = int(context.global_epoch)
    return epoch > 1 and ((epoch - 1) % period == 0)


def _rad_probabilities(*, scores: torch.Tensor, k: float, c: float) -> np.ndarray:
    safe_scores = scores.to(dtype=torch.float64).clamp_min(0.0)
    if k == 0.0:
        weighted = torch.ones_like(safe_scores)
    else:
        weighted = torch.pow(safe_scores, k)
    mean_value = weighted.mean()
    if float(mean_value.item()) <= 0.0:
        weighted = torch.ones_like(weighted)
        mean_value = weighted.mean()
    weighted = (weighted / mean_value) + float(c)
    total = weighted.sum()
    if float(total.item()) <= 0.0:
        weighted = torch.ones_like(weighted)
        total = weighted.sum()
    probabilities = (weighted / total).cpu().numpy()
    return probabilities


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
