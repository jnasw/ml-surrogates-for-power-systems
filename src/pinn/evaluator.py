"""Reusable evaluation helpers for PINN training and analysis."""

from __future__ import annotations

from dataclasses import dataclass
from math import nan
from typing import Any

import torch
from torch import nn

from src.pinn.analysis_data import PinnAnalysisBundle
from src.pinn.data import PinnDatasetBundle
from src.pinn.losses import LossWeights, PinnLossBreakdown, compute_pinn_losses
from src.pinn.residuals import compute_residual_terms, compute_supervised_dt_terms


@dataclass(frozen=True)
class PinnLossScalars:
    total: float
    data: float
    dt: float
    physics: float
    ic: float


def move_pinn_dataset_to_device(
    dataset: PinnDatasetBundle,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> PinnDatasetBundle:
    """Move a PINN dataset bundle to a device/dtype."""
    return PinnDatasetBundle(
        train_x=dataset.train_x.to(device=device, dtype=dtype),
        train_y=dataset.train_y.to(device=device, dtype=dtype),
        train_col_x=dataset.train_col_x.to(device=device, dtype=dtype),
        train_init_x=dataset.train_init_x.to(device=device, dtype=dtype),
        train_init_y=dataset.train_init_y.to(device=device, dtype=dtype),
        val_x=None if dataset.val_x is None else dataset.val_x.to(device=device, dtype=dtype),
        val_y=None if dataset.val_y is None else dataset.val_y.to(device=device, dtype=dtype),
        val_col_x=None if dataset.val_col_x is None else dataset.val_col_x.to(device=device, dtype=dtype),
        test_x=None if dataset.test_x is None else dataset.test_x.to(device=device, dtype=dtype),
        test_y=None if dataset.test_y is None else dataset.test_y.to(device=device, dtype=dtype),
    )


def evaluate_pinn_loss_breakdown(
    *,
    model: nn.Module,
    criterion: nn.Module,
    ode_model: Any,
    formulation: str,
    weights: LossWeights,
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    x_col: torch.Tensor,
    x_init: torch.Tensor,
    y_init: torch.Tensor,
    create_graph: bool = True,
) -> PinnLossBreakdown:
    """Evaluate the supervised, dt, physics, IC, and total PINN losses."""
    pred = model(x_data)
    supervised_dt_terms = compute_supervised_dt_terms(
        model=model,
        x=x_data,
        y_true=y_data,
        ode_model=ode_model,
        formulation=formulation,
        create_graph=create_graph,
    )
    collocation_terms = compute_residual_terms(
        model=model,
        x=x_col,
        ode_model=ode_model,
        formulation=formulation,
        create_graph=create_graph,
    )
    init_pred = model(x_init)
    return compute_pinn_losses(
        criterion=criterion,
        supervised_prediction=pred,
        supervised_target=y_data,
        supervised_dt_terms=supervised_dt_terms,
        collocation_terms=collocation_terms,
        init_prediction=init_pred,
        init_target=y_init,
        weights=weights,
    )


def evaluate_analysis_loss_scalars(
    *,
    model: nn.Module,
    criterion: nn.Module,
    ode_model: Any,
    formulation: str,
    weights: LossWeights,
    bundle: PinnAnalysisBundle,
) -> PinnLossScalars:
    """Evaluate scalar PINN losses for a fixed analysis bundle.

    Missing components are reported as ``nan`` and are excluded from the
    weighted total.
    """

    total = 0.0
    loss_data = nan
    loss_dt = nan
    loss_physics = nan
    loss_ic = nan

    if bundle.has_supervised:
        pred = model(bundle.x_data)
        loss_data = float(criterion(pred, bundle.y_data).item())
        total += float(weights.data) * loss_data
        supervised_dt_terms = compute_supervised_dt_terms(
            model=model,
            x=bundle.x_data,
            y_true=bundle.y_data,
            ode_model=ode_model,
            formulation=formulation,
            create_graph=False,
        )
        loss_dt = float(
            criterion(
                supervised_dt_terms.residual,
                torch.zeros_like(supervised_dt_terms.residual),
            ).item()
        )
        total += float(weights.dt) * loss_dt

    if bundle.has_collocation:
        collocation_terms = compute_residual_terms(
            model=model,
            x=bundle.x_col,
            ode_model=ode_model,
            formulation=formulation,
            create_graph=False,
        )
        loss_physics = float(
            criterion(
                collocation_terms.residual,
                torch.zeros_like(collocation_terms.residual),
            ).item()
        )
        total += float(weights.physics) * loss_physics

    if bundle.has_init:
        init_pred = model(bundle.x_init)
        loss_ic = float(criterion(init_pred, bundle.y_init).item())
        total += float(weights.ic) * loss_ic

    return PinnLossScalars(
        total=float(total),
        data=float(loss_data),
        dt=float(loss_dt),
        physics=float(loss_physics),
        ic=float(loss_ic),
    )
