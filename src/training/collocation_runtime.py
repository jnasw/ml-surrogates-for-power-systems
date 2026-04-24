"""Trainer-side collocation runtime helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from src.pinn.collocation import CollocationStrategyContext
from src.pinn.runtime import ResolvedCollocationSettings
from src.pinn.residuals import compute_supervised_dt_terms


def collocation_refresh_mode(collocation: ResolvedCollocationSettings) -> str:
    return str(collocation.refresh_mode).strip().lower()


def collocation_phase_boundary_enabled(collocation: ResolvedCollocationSettings) -> bool:
    return collocation_refresh_mode(collocation) == "phase_boundary"


def phase_boundary_refresh_planned(
    collocation: ResolvedCollocationSettings,
    *,
    phase_name: str,
    event: str,
) -> bool:
    if collocation_refresh_mode(collocation) != "phase_boundary":
        return False
    if event == "start":
        names = set(collocation.refresh_on_phase_start)
    elif event == "end":
        names = set(collocation.refresh_on_phase_end)
    else:
        raise ValueError("Phase-boundary refresh event must be one of: start, end.")
    return str(phase_name).strip().lower() in names


def dt_vrba_configured(vrba_config: Any) -> bool:
    target_sets = set(tuple(getattr(vrba_config, "target_sets", ()) or ()))
    return bool(
        getattr(vrba_config, "enabled", False)
        and getattr(vrba_config, "adaptive_weighting", False)
        and "dt" in target_sets
    )


def initialize_dt_vrba_weights(*, vrba_config: Any, train_x: torch.Tensor) -> torch.Tensor | None:
    if not dt_vrba_configured(vrba_config):
        return None
    lambda_init = 0.1 * float(getattr(vrba_config, "lambda_max0", 1.0))
    return torch.full(
        (int(train_x.shape[0]),),
        fill_value=lambda_init,
        dtype=torch.float64,
        device=train_x.device,
    )


def dt_vrba_weighting_enabled_for_phase(*, vrba_config: Any, phase_is_full_batch: bool) -> bool:
    if not dt_vrba_configured(vrba_config):
        return False
    if bool(phase_is_full_batch) and bool(getattr(vrba_config, "freeze_weighting_during_full_batch", False)):
        return False
    return True


def _should_update_local_vrba_weights(*, config: Any, context: CollocationStrategyContext) -> bool:
    from src.pinn.collocation.strategies import _should_refresh

    return _should_refresh(config=config, context=context)


def update_dt_vrba_weights(
    *,
    weights: torch.Tensor | None,
    metadata: dict[str, Any],
    model: nn.Module,
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    row_indices: torch.Tensor,
    ode_model: Any,
    formulation: str,
    vrba_config: Any,
    context: CollocationStrategyContext,
    config: Any,
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    configured = dt_vrba_configured(vrba_config)
    weighting_enabled = dt_vrba_weighting_enabled_for_phase(
        vrba_config=vrba_config,
        phase_is_full_batch=context.phase_is_full_batch,
    )
    updated_metadata = dict(metadata or {})
    updated_metadata["dt_vrba_weighting_configured"] = bool(configured)
    updated_metadata["dt_vrba_weighting_enabled"] = bool(weighting_enabled)
    updated_metadata["dt_vrba_weighting_frozen"] = bool(configured and not weighting_enabled)
    updated_metadata["dt_vrba_phase_is_full_batch"] = bool(context.phase_is_full_batch)
    updated_metadata["dt_vrba_potential"] = str(getattr(vrba_config, "potential", "quadratic"))
    if not configured:
        return None, updated_metadata
    if weights is None:
        weights = initialize_dt_vrba_weights(vrba_config=vrba_config, train_x=x_data)
    if not weighting_enabled:
        return weights, updated_metadata
    if not _should_update_local_vrba_weights(config=config, context=context):
        if weights is not None:
            updated_metadata["dt_vrba_lambda_mean"] = float(weights.mean().item())
            updated_metadata["dt_vrba_lambda_max"] = float(weights.max().item())
            updated_metadata["dt_vrba_lambda_min"] = float(weights.min().item())
        return weights, updated_metadata
    x_req = x_data.detach().clone().requires_grad_(True)
    pred = model(x_req)
    dt_terms = compute_supervised_dt_terms(
        model=model,
        x=x_req,
        y_true=y_data,
        ode_model=ode_model,
        formulation=formulation,
        create_graph=False,
        prediction=pred,
    )
    scores = dt_terms.residual.detach().to(dtype=torch.float64).pow(2).sum(dim=1).sqrt()
    safe_scores = scores.abs()
    if str(getattr(vrba_config, "potential", "quadratic")).strip().lower() == "quadratic":
        target = safe_scores
        if float(target.sum().item()) <= 0.0:
            target = torch.ones_like(target)
        target = target / target.sum()
    else:
        score_scale = float(safe_scores.mean().item())
        step = int(updated_metadata.get("dt_vrba_update_count", 0)) + 1
        epsilon = max(
            float(getattr(vrba_config, "temperature_floor", 1.0e-12)),
            float(getattr(vrba_config, "temperature_scale", 1.0)) * score_scale / max(step, 1),
        )
        shifted = safe_scores - safe_scores.max()
        exp_target = torch.exp(shifted / max(epsilon, 1.0e-12))
        target = exp_target / exp_target.sum()
    lambda_max = min(
        float(getattr(vrba_config, "lambda_cap", getattr(vrba_config, "lambda_max0", 1.0))),
        float(getattr(vrba_config, "lambda_max0", 1.0))
        + float(int(updated_metadata.get("dt_vrba_update_count", 0)) + 1)
        / float(getattr(vrba_config, "update_interval_epochs", 1)),
    )
    gamma = max(0.0, min(1.0 - (float(getattr(vrba_config, "eta", 0.1)) / max(lambda_max, 1.0e-12)), 0.999999))
    uniform = torch.full_like(target, fill_value=1.0 / float(target.shape[0]))
    mixed_target = float(getattr(vrba_config, "phi", 1.0)) * target + (1.0 - float(getattr(vrba_config, "phi", 1.0))) * uniform
    eta_star = float(getattr(vrba_config, "eta", 0.1)) / max(float(target.max().item()), 1.0e-12)
    next_weights = weights.clone()
    next_weights.index_copy_(0, row_indices, gamma * weights.index_select(0, row_indices) + eta_star * mixed_target)
    updated_metadata["dt_vrba_update_count"] = int(updated_metadata.get("dt_vrba_update_count", 0)) + 1
    updated_metadata["dt_vrba_lambda_mean"] = float(next_weights.mean().item())
    updated_metadata["dt_vrba_lambda_max"] = float(next_weights.max().item())
    updated_metadata["dt_vrba_lambda_min"] = float(next_weights.min().item())
    updated_metadata["dt_vrba_gamma"] = float(gamma)
    updated_metadata["dt_vrba_eta_star"] = float(eta_star)
    return next_weights, updated_metadata


def active_collocation_weights_or_none(
    *,
    vrba_config: Any,
    x_col_weights: torch.Tensor | None,
    weighting_enabled: bool | None = None,
) -> torch.Tensor | None:
    if weighting_enabled is False:
        return None
    if weighting_enabled is None and not bool(getattr(vrba_config, "adaptive_weighting", False)):
        return None
    if x_col_weights is None:
        raise ValueError(
            "pinn.vrba.adaptive_weighting=true requires a collocation strategy that supplies local weights. "
            "Use pinn.collocation.strategy=vrba_sample with pinn.vrba.enabled=true."
        )
    return x_col_weights


def active_init_weights_or_none(
    *,
    vrba_config: Any,
    x_init_weights: torch.Tensor | None,
    weighting_enabled: bool | None = None,
) -> torch.Tensor | None:
    if weighting_enabled is False:
        return None
    target_sets = tuple(getattr(vrba_config, "target_sets", ()) or ())
    if weighting_enabled is None and not (
        bool(getattr(vrba_config, "adaptive_weighting", False)) and "ic" in set(target_sets)
    ):
        return None
    if x_init_weights is None:
        raise ValueError(
            "IC vRBA weighting requires a collocation manager that supplies local init weights. "
            "Use pinn.collocation.strategy=vrba_sample with pinn.vrba.enabled=true and include ic in pinn.vrba.target_sets."
        )
    return x_init_weights


def active_dt_weights_or_none(
    *,
    vrba_config: Any,
    x_data_dt_weights: torch.Tensor | None,
    weighting_enabled: bool | None = None,
) -> torch.Tensor | None:
    if weighting_enabled is False:
        return None
    target_sets = tuple(getattr(vrba_config, "target_sets", ()) or ())
    if weighting_enabled is None and not (
        bool(getattr(vrba_config, "adaptive_weighting", False)) and "dt" in set(target_sets)
    ):
        return None
    if x_data_dt_weights is None:
        raise ValueError(
            "DT vRBA weighting requires local supervised-row weights. "
            "Enable pinn.vrba.enabled=true, pinn.vrba.adaptive_weighting=true, and include dt in pinn.vrba.target_sets."
        )
    return x_data_dt_weights


@dataclass(frozen=True)
class EpochConstraintBatch:
    pool_batch: Any
    train_dt_weights: torch.Tensor | None
    epoch_train_dt_weights: torch.Tensor | None
    active_collocation_weights: torch.Tensor | None
    active_dt_weights: torch.Tensor | None
    active_init_weights: torch.Tensor | None


def prepare_epoch_constraint_batch(
    *,
    collocation_manager: Any,
    vrba_config: Any,
    train_dt_weights: torch.Tensor | None,
    model: nn.Module,
    ode_model: Any,
    formulation: str,
    config: Any,
    global_epoch: int,
    phase_name: str,
    phase_allows_sampling: bool,
    phase_is_full_batch: bool,
    phase_epoch: int,
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    row_indices: torch.Tensor,
) -> EpochConstraintBatch:
    """Prepare optional collocation/vRBA epoch inputs for the trainer loop."""
    from src.training.pinn_runtime import sample_rows_with_indices

    context = CollocationStrategyContext(
        global_epoch=global_epoch,
        phase_name=phase_name,
        phase_allows_sampling=phase_allows_sampling,
        phase_is_full_batch=phase_is_full_batch,
        phase_epoch=phase_epoch,
        refresh_event=None,
        model=model,
        ode_model=ode_model,
        formulation=formulation,
    )
    pool_batch = collocation_manager.prepare_epoch_batch(context=context)
    active_collocation_weights = active_collocation_weights_or_none(
        vrba_config=vrba_config,
        x_col_weights=pool_batch.x_col_weights,
        weighting_enabled=bool(collocation_manager.state.metadata.get("residual_vrba_weighting_enabled", False)),
    )
    next_train_dt_weights, dt_vrba_metadata = update_dt_vrba_weights(
        weights=train_dt_weights,
        metadata=dict(collocation_manager.state.metadata or {}),
        model=model,
        x_data=x_data,
        y_data=y_data,
        row_indices=row_indices,
        ode_model=ode_model,
        formulation=formulation,
        vrba_config=vrba_config,
        context=context,
        config=config,
    )
    collocation_manager.state.metadata.update(dt_vrba_metadata)
    epoch_train_dt_weights = (
        None
        if next_train_dt_weights is None
        else sample_rows_with_indices(next_train_dt_weights, row_indices)
    )
    active_dt_weights = active_dt_weights_or_none(
        vrba_config=vrba_config,
        x_data_dt_weights=epoch_train_dt_weights,
        weighting_enabled=bool(collocation_manager.state.metadata.get("dt_vrba_weighting_enabled", False)),
    )
    active_init_weights = active_init_weights_or_none(
        vrba_config=vrba_config,
        x_init_weights=pool_batch.x_init_weights,
        weighting_enabled=bool(collocation_manager.state.metadata.get("ic_constraint_vrba_weighting_enabled", False)),
    )
    return EpochConstraintBatch(
        pool_batch=pool_batch,
        train_dt_weights=next_train_dt_weights,
        epoch_train_dt_weights=epoch_train_dt_weights,
        active_collocation_weights=active_collocation_weights,
        active_dt_weights=active_dt_weights,
        active_init_weights=active_init_weights,
    )
