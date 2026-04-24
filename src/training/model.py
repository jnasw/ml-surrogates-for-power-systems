"""Neural surrogate model definitions."""

from __future__ import annotations

import torch
from torch import nn


class TrajectoryMLP(nn.Module):
    """MLP mapping IC -> flattened trajectory."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, hidden_layers: int):
        super().__init__()
        layers: list[nn.Module] = []
        d = input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.Tanh())
            d = hidden_dim
        layers.append(nn.Linear(d, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BaselineTrajectoryMLP(nn.Module):
    """Robust baseline MLP for IC -> flattened trajectory."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, hidden_layers: int, dropout: float):
        super().__init__()
        self.inp = nn.Linear(input_dim, hidden_dim)
        blocks: list[nn.Module] = []
        for _ in range(hidden_layers):
            blocks.append(nn.Linear(hidden_dim, hidden_dim))
            blocks.append(nn.LayerNorm(hidden_dim))
            blocks.append(nn.SiLU())
            if dropout > 0:
                blocks.append(nn.Dropout(dropout))
        self.blocks = nn.Sequential(*blocks)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.inp(x)
        return self.out(self.blocks(h))


def _activation(name: str) -> nn.Module:
    normalized = str(name).strip().lower()
    if normalized == "tanh":
        return nn.Tanh()
    if normalized == "silu":
        return nn.SiLU()
    if normalized == "relu":
        return nn.ReLU()
    raise ValueError("Unsupported activation. Use one of: tanh, silu, relu.")


class TimeConditionedMLP(nn.Module):
    """Simple PINN baseline MLP with input [time, IC/features]."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(_activation(activation))
            d = hidden_dim
        layers.append(nn.Linear(d, output_dim))
        self.net = nn.Sequential(*layers)
        self.activation_name = str(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
