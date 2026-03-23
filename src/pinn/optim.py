"""Optimizer construction for PINN training."""

from __future__ import annotations

import torch
from torch import nn


def build_optimizer(model: nn.Module, optimizer_name: str, lr: float) -> torch.optim.Optimizer:
    normalized = str(optimizer_name).strip().lower()
    if normalized == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    if normalized == "lbfgs":
        return torch.optim.LBFGS(model.parameters(), lr=lr, line_search_fn="strong_wolfe")
    raise ValueError("Unsupported optimizer. Use one of: Adam, LBFGS.")

