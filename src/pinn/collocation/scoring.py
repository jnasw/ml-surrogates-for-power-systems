"""Residual-based scoring helpers for adaptive collocation strategies."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.pinn.residuals import compute_residual_terms


def score_collocation_points(
    *,
    model: nn.Module,
    x: torch.Tensor,
    ode_model: Any,
    formulation: str,
    norm: str = "l2",
) -> torch.Tensor:
    """Return per-point residual scores for a collocation candidate set."""
    terms = compute_residual_terms(
        model=model,
        x=x,
        ode_model=ode_model,
        formulation=formulation,
        create_graph=False,
    )
    residual = terms.residual

    norm_name = str(norm).strip().lower()
    if norm_name == "l1":
        return residual.abs().sum(dim=1)
    if norm_name == "linf":
        return residual.abs().amax(dim=1)
    if norm_name == "l2":
        return torch.linalg.vector_norm(residual, dim=1)
    raise ValueError("norm must be one of: l1, l2, linf")
