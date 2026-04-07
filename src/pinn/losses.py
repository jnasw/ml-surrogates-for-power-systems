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
    init_prediction: torch.Tensor,
    init_target: torch.Tensor,
    weights: LossWeights,
) -> PinnLossBreakdown:
    components = {
        "data": criterion(supervised_prediction, supervised_target),
        "dt": criterion(supervised_dt_terms.residual, torch.zeros_like(supervised_dt_terms.residual)),
        "physics": criterion(collocation_terms.residual, torch.zeros_like(collocation_terms.residual)),
        "ic": criterion(init_prediction, init_target),
    }
    total = sum(float(weight) * components[name] for name, weight in weights.items())
    return PinnLossBreakdown(total=total, components=components)
