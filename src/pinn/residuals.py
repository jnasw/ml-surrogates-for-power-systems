"""Autograd derivative and residual utilities for PINN training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class ResidualTerms:
    prediction: torch.Tensor
    dy_dt: torch.Tensor
    ode_rhs: torch.Tensor
    residual: torch.Tensor


def compute_time_derivative(prediction: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Compute d(prediction)/dt using autograd on the first input column."""
    grads = []
    for idx in range(prediction.shape[1]):
        grad = torch.autograd.grad(
            prediction[:, idx].sum(),
            x,
            create_graph=True,
            retain_graph=True,
        )[0]
        grads.append(grad[:, :1])
    return torch.cat(grads, dim=1)


def evaluate_ode_rhs(
    ode_model: Any,
    prediction: torch.Tensor,
    x: torch.Tensor,
    formulation: str = "odequations",
) -> torch.Tensor:
    formulation_name = str(formulation).strip().lower()
    if formulation_name not in {"odequations", "odequations_v2"}:
        raise ValueError("formulation must be one of: odequations, odequations_v2")

    if formulation_name == "odequations":
        append_dim = int(x.shape[1] - prediction.shape[1] - 1)
        if append_dim < 0:
            raise ValueError("Prediction width exceeds available input state/features.")
        state = prediction
        if append_dim > 0:
            state = torch.cat((prediction, x[:, -append_dim:]), dim=1)
        rhs_list = ode_model.odequations(0.0, state.split(split_size=1, dim=1))
    else:
        rhs_list = ode_model.odequations_v2(0.0, prediction.split(split_size=1, dim=1))

    rhs_parts = []
    for part in rhs_list:
        if isinstance(part, torch.Tensor):
            rhs_parts.append(part if part.ndim == 2 else part.unsqueeze(1))
        else:
            rhs_parts.append(torch.full((prediction.shape[0], 1), float(part), dtype=prediction.dtype, device=prediction.device))
    return torch.cat(rhs_parts, dim=1)


def compute_residual_terms(
    model: nn.Module,
    x: torch.Tensor,
    ode_model: Any,
    formulation: str = "odequations",
) -> ResidualTerms:
    x_req = x.detach().clone().requires_grad_(True)
    prediction = model(x_req)
    dy_dt = compute_time_derivative(prediction=prediction, x=x_req)
    ode_rhs = evaluate_ode_rhs(
        ode_model=ode_model,
        prediction=prediction,
        x=x_req,
        formulation=formulation,
    )
    return ResidualTerms(
        prediction=prediction,
        dy_dt=dy_dt,
        ode_rhs=ode_rhs,
        residual=dy_dt - ode_rhs,
    )

