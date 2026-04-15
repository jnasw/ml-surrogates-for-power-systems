"""Loss helpers for PINN training objectives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import torch
from torch import nn

from src.pinn.residuals import ResidualTerms

LOSS_COMPONENTS: tuple[str, ...] = ("data", "dt", "physics", "ic")


@dataclass(frozen=True)
class LossWeights:
    data: float
    dt: float
    physics: float
    ic: float

    def as_dict(self) -> dict[str, float]:
        return {
            "data": float(self.data),
            "dt": float(self.dt),
            "physics": float(self.physics),
            "ic": float(self.ic),
        }

    def items(self) -> Iterable[tuple[str, float]]:
        return self.as_dict().items()

    def get(self, name: str) -> float:
        return float(self.as_dict()[name])


@dataclass(frozen=True)
class PinnLossBreakdown:
    total: torch.Tensor
    components: dict[str, torch.Tensor] = field(default_factory=dict)

    def component(self, name: str) -> torch.Tensor:
        return self.components[name]

    @property
    def data(self) -> torch.Tensor:
        return self.component("data")

    @property
    def dt(self) -> torch.Tensor:
        return self.component("dt")

    @property
    def physics(self) -> torch.Tensor:
        return self.component("physics")

    @property
    def ic(self) -> torch.Tensor:
        return self.component("ic")


def compute_pinn_losses(
    *,
    criterion: nn.Module,
    supervised_prediction: torch.Tensor,
    supervised_target: torch.Tensor,
    supervised_dt_terms: ResidualTerms,
    collocation_terms: ResidualTerms,
    collocation_weights: torch.Tensor | None,
    init_prediction: torch.Tensor,
    init_target: torch.Tensor,
    init_weights: torch.Tensor | None,
    weights: LossWeights,
) -> PinnLossBreakdown:
    physics_residual = collocation_terms.residual
    if collocation_weights is not None:
        physics_weights = collocation_weights
        if physics_weights.ndim == 1:
            physics_weights = physics_weights.unsqueeze(1)
        physics_residual = physics_weights * physics_residual
    init_residual = init_prediction - init_target
    if init_weights is not None:
        ic_weights = init_weights
        if ic_weights.ndim == 1:
            ic_weights = ic_weights.unsqueeze(1)
        init_residual = ic_weights * init_residual
    components = {
        "data": criterion(supervised_prediction, supervised_target),
        "dt": criterion(supervised_dt_terms.residual, torch.zeros_like(supervised_dt_terms.residual)),
        "physics": criterion(physics_residual, torch.zeros_like(physics_residual)),
        "ic": criterion(init_residual, torch.zeros_like(init_residual)),
    }
    total = sum(float(weight) * components[name] for name, weight in weights.items())
    return PinnLossBreakdown(total=total, components=components)
