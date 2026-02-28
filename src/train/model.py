"""Trajectory surrogate model definitions."""

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
