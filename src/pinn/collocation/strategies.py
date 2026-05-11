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
from src.pinn.vrba import VrbAConfig
from src.training.runtime import cfg_get


@dataclass(frozen=True)
class CollocationStrategyContext:
    """Epoch/phase context passed to collocation strategies."""

    global_epoch: int
    phase_name: str
    phase_allows_sampling: bool
    phase_is_full_batch: bool
    model: nn.Module | None
    ode_model: Any | None
    formulation: str
    phase_epoch: int | None = None
    refresh_event: str | None = None


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

    def current_weights(self) -> torch.Tensor | None:
        """Optional local weights aligned with ``current_points``."""
        return None

    def export_adaptive_state(self) -> dict[str, Any] | None:
        """Optional serializable adaptive state for checkpoint payloads."""
        return None

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


class VariationalResidualAdaptiveSamplingStrategy(StaticCollocationStrategy):
    """vRBA-inspired adaptive collocation sampling with persistent point weights."""

    def __init__(
        self,
        *,
        initial_points: torch.Tensor,
        config: Any,
        pool_points: torch.Tensor,
        seed: int,
        active_points: int,
        score_norm: str,
        vrba_config: VrbAConfig,
        initial_indices: torch.Tensor | None = None,
    ) -> None:
        pool_rows = int(pool_points.shape[0])
        if pool_rows < int(active_points):
            raise ValueError("vRBA sampling requires pool_points >= active_points.")
        if initial_indices is None:
            initial_indices = torch.arange(int(active_points), dtype=torch.long, device=pool_points.device)
        if initial_indices.ndim != 1:
            raise ValueError("initial_indices must be a rank-1 tensor.")
        if int(initial_indices.shape[0]) != int(active_points):
            raise ValueError("initial_indices must have length active_points.")
        if int(torch.unique(initial_indices).shape[0]) != int(active_points):
            raise ValueError("initial_indices must contain unique pool indices.")
        if int(initial_indices.min().item()) < 0 or int(initial_indices.max().item()) >= pool_rows:
            raise ValueError("initial_indices must lie within the persistent pool.")
        normalized_initial_points = pool_points.index_select(0, initial_indices.to(device=pool_points.device, dtype=torch.long))
        super().__init__(initial_points=normalized_initial_points, config=config)
        self._pool_points = pool_points
        self._seed = int(seed)
        self._active_points = int(active_points)
        self._score_norm = str(score_norm)
        self._vrba_config = vrba_config
        self._active_indices = initial_indices.to(device=pool_points.device, dtype=torch.long)
        lambda_init = 0.1 * float(vrba_config.lambda_max0)
        self._lambda = torch.full(
            (pool_rows,),
            fill_value=lambda_init,
            dtype=torch.float64,
            device=pool_points.device,
        )
        self._state.metadata["last_strategy"] = "vrba_sample"
        self._state.metadata["vrba_pool_rows"] = int(pool_points.shape[0])
        self._state.metadata["vrba_active_rows"] = int(active_points)
        self._state.metadata["vrba_potential"] = str(vrba_config.potential)
        self._state.metadata["vrba_update_count"] = 0
        self._state.metadata["vrba_sampling_configured"] = bool(vrba_config.adaptive_sampling)
        self._state.metadata["vrba_weighting_configured"] = bool(vrba_config.adaptive_weighting)
        self._state.metadata["vrba_sampling_enabled"] = bool(vrba_config.adaptive_sampling)
        self._state.metadata["vrba_weighting_enabled"] = bool(vrba_config.adaptive_weighting)
        self._state.metadata["vrba_sampling_frozen"] = False
        self._state.metadata["vrba_weighting_frozen"] = False
        self._state.metadata["vrba_phase_is_full_batch"] = False
        self._active_lambda = self._lambda.index_select(0, self._active_indices)
        self._effective_sampling_enabled = bool(vrba_config.adaptive_sampling)
        self._effective_weighting_enabled = bool(vrba_config.adaptive_weighting)

    def current_weights(self) -> torch.Tensor | None:
        if not bool(self._effective_weighting_enabled):
            return None
        return self._active_lambda.to(device=self.current_points().device, dtype=self.current_points().dtype)

    def maybe_refresh(self, *, context: CollocationStrategyContext) -> None:
        sampling_enabled, weighting_enabled = self._update_effective_mode_metadata(context=context)
        if not _should_refresh(config=self._config, context=context):
            return
        if not (sampling_enabled or weighting_enabled):
            return
        if context.model is None or context.ode_model is None:
            raise ValueError("vRBA sampling requires model and ode_model in the collocation strategy context.")
        pool_rows = int(self._pool_points.shape[0])

        scores = score_collocation_points(
            model=context.model,
            x=self._pool_points,
            ode_model=context.ode_model,
            formulation=context.formulation,
            norm=self._score_norm,
        ).detach()

        target = _vrba_target_distribution(
            scores=scores,
            potential=self._vrba_config.potential,
            temperature_scale=self._vrba_config.temperature_scale,
            temperature_floor=self._vrba_config.temperature_floor,
            step=self._state.refresh_count + 1,
        )
        lambda_max = min(
            float(self._vrba_config.lambda_cap),
            float(self._vrba_config.lambda_max0)
            + float(self._state.refresh_count + 1) / float(self._vrba_config.update_interval_epochs),
        )
        gamma = max(0.0, min(1.0 - (float(self._vrba_config.eta) / max(lambda_max, 1.0e-12)), 0.999999))
        uniform = torch.full_like(target, fill_value=1.0 / float(pool_rows))
        mixed_target = float(self._vrba_config.phi) * target + (1.0 - float(self._vrba_config.phi)) * uniform
        eta_star = float(self._vrba_config.eta) / max(float(target.max().item()), 1.0e-12)
        self._lambda = gamma * self._lambda + eta_star * mixed_target
        normalized = _normalize_probabilities(self._lambda)

        if sampling_enabled:
            rng = np.random.default_rng(self._seed + self._state.refresh_count + int(context.global_epoch))
            selected_ids = rng.choice(
                pool_rows,
                size=self._active_points,
                replace=False,
                p=normalized,
            )
            self._active_indices = torch.as_tensor(selected_ids, dtype=torch.long, device=self._pool_points.device)

        self._state.active_points = self._pool_points.index_select(0, self._active_indices)
        self._active_lambda = self._lambda.index_select(0, self._active_indices)
        self._state.refresh_count += 1
        self._state.last_refresh_epoch = int(context.global_epoch)
        self._state.metadata["last_strategy"] = "vrba_sample"
        self._state.metadata["last_score_mean"] = float(scores.mean().item())
        self._state.metadata["last_score_max"] = float(scores.max().item())
        self._state.metadata["vrba_lambda_mean"] = float(self._lambda.mean().item())
        self._state.metadata["vrba_lambda_max"] = float(self._lambda.max().item())
        self._state.metadata["vrba_lambda_min"] = float(self._lambda.min().item())
        self._state.metadata["vrba_lambda_sum"] = float(self._lambda.sum().item())
        self._state.metadata["vrba_gamma"] = float(gamma)
        self._state.metadata["vrba_eta_star"] = float(eta_star)
        self._state.metadata["vrba_update_count"] = int(self._state.refresh_count)

    def _update_effective_mode_metadata(self, *, context: CollocationStrategyContext) -> tuple[bool, bool]:
        phase_is_full_batch = bool(context.phase_is_full_batch)
        sampling_frozen = bool(
            phase_is_full_batch
            and self._vrba_config.adaptive_sampling
            and self._vrba_config.freeze_sampling_during_full_batch
        )
        weighting_frozen = bool(
            phase_is_full_batch
            and self._vrba_config.adaptive_weighting
            and self._vrba_config.freeze_weighting_during_full_batch
        )
        self._effective_sampling_enabled = bool(self._vrba_config.adaptive_sampling and not sampling_frozen)
        self._effective_weighting_enabled = bool(self._vrba_config.adaptive_weighting and not weighting_frozen)
        self._state.metadata["vrba_sampling_enabled"] = bool(self._effective_sampling_enabled)
        self._state.metadata["vrba_weighting_enabled"] = bool(self._effective_weighting_enabled)
        self._state.metadata["vrba_sampling_frozen"] = bool(sampling_frozen)
        self._state.metadata["vrba_weighting_frozen"] = bool(weighting_frozen)
        self._state.metadata["vrba_phase_is_full_batch"] = bool(phase_is_full_batch)
        return self._effective_sampling_enabled, self._effective_weighting_enabled

    def export_adaptive_state(self) -> dict[str, Any] | None:
        return {
            "strategy": "vrba_sample",
            "pool_rows": int(self._pool_points.shape[0]),
            "active_points": int(self._active_points),
            "active_indices": self._active_indices.detach().cpu().clone(),
            "lambda": self._lambda.detach().cpu().clone(),
            "active_lambda": self._active_lambda.detach().cpu().clone(),
            "refresh_count": int(self._state.refresh_count),
            "last_refresh_epoch": self._state.last_refresh_epoch,
            "metadata": dict(self._state.metadata or {}),
        }


class _AppendCollocationStrategy(StaticCollocationStrategy):
    """Base class for append-based collocation refinement strategies."""

    def __init__(
        self,
        *,
        initial_points: torch.Tensor,
        config: Any,
        domain: CollocationDomain,
        sampler: str,
        seed: int,
        target_points: int,
        candidate_points: int,
        append_points: int,
        score_norm: str,
    ) -> None:
        super().__init__(initial_points=initial_points, config=config)
        self._domain = domain
        self._sampler = str(sampler)
        self._seed = int(seed)
        self._target_points = int(target_points)
        self._candidate_points = int(candidate_points)
        self._append_points = int(append_points)
        self._score_norm = str(score_norm)

    def maybe_refresh(self, *, context: CollocationStrategyContext) -> None:
        if not _should_refresh(config=self._config, context=context):
            return
        if context.model is None or context.ode_model is None:
            raise ValueError("Append-based adaptive sampling requires model and ode_model in the context.")

        current_rows = int(self.current_points().shape[0])
        if current_rows >= self._target_points:
            return
        add_rows = min(self._append_points, self._target_points - current_rows)
        if add_rows <= 0:
            return
        if self._candidate_points < add_rows:
            raise ValueError("pinn.collocation.candidate_points must be >= points appended per refresh.")

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

        new_points = self._select_points(
            candidates=candidates,
            scores=scores,
            add_rows=add_rows,
            context=context,
        )
        self._state.active_points = torch.cat((self.current_points(), new_points), dim=0)
        self._state.refresh_count += 1
        self._state.last_refresh_epoch = int(context.global_epoch)
        self._state.metadata["last_score_mean"] = float(scores.mean().item())
        self._state.metadata["last_score_max"] = float(scores.max().item())
        self._state.metadata["active_point_count"] = int(self._state.active_points.shape[0])

    @abstractmethod
    def _select_points(
        self,
        *,
        candidates: torch.Tensor,
        scores: torch.Tensor,
        add_rows: int,
        context: CollocationStrategyContext,
    ) -> torch.Tensor:
        """Select new points from a scored candidate pool."""


class ResidualAdaptiveRefinementDistributionStrategy(_AppendCollocationStrategy):
    """RAR-D: append points sampled from the residual-based distribution."""

    def __init__(
        self,
        *,
        initial_points: torch.Tensor,
        config: Any,
        domain: CollocationDomain,
        sampler: str,
        seed: int,
        target_points: int,
        candidate_points: int,
        append_points: int,
        rad_k: float,
        rad_c: float,
        score_norm: str,
    ) -> None:
        super().__init__(
            initial_points=initial_points,
            config=config,
            domain=domain,
            sampler=sampler,
            seed=seed,
            target_points=target_points,
            candidate_points=candidate_points,
            append_points=append_points,
            score_norm=score_norm,
        )
        self._rad_k = float(rad_k)
        self._rad_c = float(rad_c)

    def _select_points(
        self,
        *,
        candidates: torch.Tensor,
        scores: torch.Tensor,
        add_rows: int,
        context: CollocationStrategyContext,
    ) -> torch.Tensor:
        probabilities = _rad_probabilities(scores=scores, k=self._rad_k, c=self._rad_c)
        rng = np.random.default_rng(self._seed + self._state.refresh_count + int(context.global_epoch))
        selected_ids = rng.choice(
            self._candidate_points,
            size=add_rows,
            replace=False,
            p=probabilities,
        )
        indices = torch.as_tensor(selected_ids, dtype=torch.long, device=candidates.device)
        self._state.metadata["last_strategy"] = "rar_d"
        return candidates.index_select(0, indices)


class ResidualAdaptiveRefinementGreedyStrategy(_AppendCollocationStrategy):
    """RAR-G: append the top-k residual points from the candidate pool."""

    def _select_points(
        self,
        *,
        candidates: torch.Tensor,
        scores: torch.Tensor,
        add_rows: int,
        context: CollocationStrategyContext,
    ) -> torch.Tensor:
        indices = torch.topk(scores, k=add_rows, largest=True).indices.to(device=candidates.device)
        self._state.metadata["last_strategy"] = "rar_g"
        return candidates.index_select(0, indices)


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
    mode = str(
        _cfg_get_collocation(
            config=config,
            key="refresh.mode",
            legacy_key="pinn.collocation.refresh.mode",
            default="epoch_periodic",
        )
    ).strip().lower()
    if mode in {"epoch_periodic", "epoch_periodic_plus_phase_boundary"}:
        event = None if context.refresh_event in (None, "", "null") else str(context.refresh_event).strip().lower()
        if mode == "epoch_periodic_plus_phase_boundary" and event in {"phase_start", "phase_end"}:
            phase_list_key = "refresh.on_phase_start" if event == "phase_start" else "refresh.on_phase_end"
            phase_names = _cfg_get_collocation(
                config=config,
                key=phase_list_key,
                legacy_key=f"pinn.collocation.{phase_list_key}",
                default=[],
            )
            if phase_names not in (None, "null"):
                normalized_names = {str(name).strip().lower() for name in phase_names}
                if str(context.phase_name).strip().lower() in normalized_names:
                    return True
        if not context.phase_allows_sampling:
            return False
        period = int(
            _cfg_get_collocation(
                config=config,
                key="refresh_period_epochs",
                legacy_key="pinn.collocation.refresh_period_epochs",
                default=0,
            )
        )
        if period <= 0:
            return False
        epoch = int(context.global_epoch)
        return epoch > 1 and ((epoch - 1) % period == 0)
    if mode == "phase_boundary":
        event = None if context.refresh_event in (None, "", "null") else str(context.refresh_event).strip().lower()
        if event not in {"phase_start", "phase_end"}:
            return False
        phase_list_key = "refresh.on_phase_start" if event == "phase_start" else "refresh.on_phase_end"
        phase_names = _cfg_get_collocation(
            config=config,
            key=phase_list_key,
            legacy_key=f"pinn.collocation.{phase_list_key}",
            default=[],
        )
        if phase_names in (None, "null"):
            return False
        normalized_names = {str(name).strip().lower() for name in phase_names}
        return str(context.phase_name).strip().lower() in normalized_names
    raise ValueError(
        "pinn.collocation.refresh.mode must be one of: "
        "epoch_periodic, phase_boundary, epoch_periodic_plus_phase_boundary"
    )


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


def _normalize_probabilities(values: torch.Tensor) -> np.ndarray:
    weighted = values.to(dtype=torch.float64).clamp_min(0.0)
    total = weighted.sum()
    if float(total.item()) <= 0.0:
        weighted = torch.ones_like(weighted)
        total = weighted.sum()
    return (weighted / total).cpu().numpy()


def _vrba_target_distribution(
    *,
    scores: torch.Tensor,
    potential: str,
    temperature_scale: float,
    temperature_floor: float,
    step: int,
) -> torch.Tensor:
    safe_scores = scores.to(dtype=torch.float64).clamp_min(0.0)
    potential_name = str(potential).strip().lower()
    if potential_name == "quadratic":
        weighted = safe_scores
    elif potential_name == "exponential":
        max_score = float(safe_scores.max().item())
        if max_score <= 0.0:
            weighted = torch.ones_like(safe_scores)
        else:
            log_term = np.log(float(step) + 2.0)
            epsilon = max(float(temperature_floor), float(temperature_scale) * max_score / max(log_term, 1.0))
            weighted = torch.exp((safe_scores / epsilon).clamp(max=80.0))
    else:
        raise ValueError("vRBA potential must be one of: quadratic, exponential")
    total = weighted.sum()
    if float(total.item()) <= 0.0:
        weighted = torch.ones_like(weighted)
        total = weighted.sum()
    return weighted / total


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
