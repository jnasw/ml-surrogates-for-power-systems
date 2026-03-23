"""Loss helpers for phase-1 PINN training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.pinn.residuals import ResidualTerms


@dataclass(frozen=True)
class LossWeights:
    data: float
    physics: float
    ic: float


@dataclass(frozen=True)
class PinnLossBreakdown:
    total: torch.Tensor
    data: torch.Tensor
    physics: torch.Tensor
    ic: torch.Tensor


def compute_pinn_losses(
    *,
    criterion: nn.Module,
    supervised_prediction: torch.Tensor,
    supervised_target: torch.Tensor,
    collocation_terms: ResidualTerms,
    init_prediction: torch.Tensor,
    init_target: torch.Tensor,
    weights: LossWeights,
) -> PinnLossBreakdown:
    loss_data = criterion(supervised_prediction, supervised_target)
    loss_physics = criterion(collocation_terms.residual, torch.zeros_like(collocation_terms.residual))
    loss_ic = criterion(init_prediction, init_target)
    total = (
        (weights.data * loss_data)
        + (weights.physics * loss_physics)
        + (weights.ic * loss_ic)
    )
    return PinnLossBreakdown(total=total, data=loss_data, physics=loss_physics, ic=loss_ic)

