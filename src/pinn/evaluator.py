"""Reusable evaluation helpers for PINN training and analysis."""

from __future__ import annotations

from dataclasses import dataclass
from math import nan
from typing import Any

import torch
from torch import nn

from src.pinn.analysis_data import PinnAnalysisBundle
from src.pinn.data import PinnDatasetBundle
from src.pinn.losses import LOSS_COMPONENTS, LossWeights, PinnLossBreakdown, compute_pinn_losses
from src.pinn.residuals import compute_residual_terms, compute_supervised_dt_terms, prepare_input_for_autograd


@dataclass(frozen=True)
class PinnLossScalars:
    total: float
    components: dict[str, float]

    def component(self, name: str) -> float:
        return float(self.components[name])

    @property
    def data(self) -> float:
        return self.component("data")

    @property
    def dt(self) -> float:
        return self.component("dt")

    @property
    def physics(self) -> float:
        return self.component("physics")

    @property
    def ic(self) -> float:
        return self.component("ic")


@dataclass(frozen=True)
class PinnWeightingTerms:
    components: dict[str, torch.Tensor]

    def component(self, name: str) -> torch.Tensor:
        return self.components[name]


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
        train_dt_weights=None if dataset.train_dt_weights is None else dataset.train_dt_weights.to(device=device, dtype=dtype),
        train_col_weights=None if dataset.train_col_weights is None else dataset.train_col_weights.to(device=device, dtype=dtype),
        train_init_weights=None if dataset.train_init_weights is None else dataset.train_init_weights.to(device=device, dtype=dtype),
        val_x=None if dataset.val_x is None else dataset.val_x.to(device=device, dtype=dtype),
        val_y=None if dataset.val_y is None else dataset.val_y.to(device=device, dtype=dtype),
        val_col_x=None if dataset.val_col_x is None else dataset.val_col_x.to(device=device, dtype=dtype),
        test_x=None if dataset.test_x is None else dataset.test_x.to(device=device, dtype=dtype),
        test_y=None if dataset.test_y is None else dataset.test_y.to(device=device, dtype=dtype),
    )


def move_pinn_training_data_to_device(
    dataset: PinnDatasetBundle,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> PinnDatasetBundle:
    """Move only training tensors onto the training device."""
    return PinnDatasetBundle(
        train_x=dataset.train_x.to(device=device, dtype=dtype),
        train_y=dataset.train_y.to(device=device, dtype=dtype),
        train_col_x=dataset.train_col_x.to(device=device, dtype=dtype),
        train_init_x=dataset.train_init_x.to(device=device, dtype=dtype),
        train_init_y=dataset.train_init_y.to(device=device, dtype=dtype),
        train_dt_weights=None if dataset.train_dt_weights is None else dataset.train_dt_weights.to(device=device, dtype=dtype),
        train_col_weights=None if dataset.train_col_weights is None else dataset.train_col_weights.to(device=device, dtype=dtype),
        train_init_weights=None if dataset.train_init_weights is None else dataset.train_init_weights.to(device=device, dtype=dtype),
        val_x=None if dataset.val_x is None else dataset.val_x.to(dtype=dtype),
        val_y=None if dataset.val_y is None else dataset.val_y.to(dtype=dtype),
        val_col_x=None if dataset.val_col_x is None else dataset.val_col_x.to(dtype=dtype),
        test_x=None if dataset.test_x is None else dataset.test_x.to(dtype=dtype),
        test_y=None if dataset.test_y is None else dataset.test_y.to(dtype=dtype),
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
    x_data_dt_weights: torch.Tensor | None,
    x_col: torch.Tensor,
    x_col_weights: torch.Tensor | None,
    x_init: torch.Tensor,
    y_init: torch.Tensor,
    x_init_weights: torch.Tensor | None,
    create_graph: bool = True,
) -> PinnLossBreakdown:
    """Evaluate the supervised, dt, physics, IC, and total PINN losses."""
    x_data_req = prepare_input_for_autograd(x_data)
    pred = model(x_data_req)
    supervised_dt_terms = compute_supervised_dt_terms(
        model=model,
        x=x_data_req,
        y_true=y_data,
        ode_model=ode_model,
        formulation=formulation,
        create_graph=create_graph,
        prediction=pred,
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
        supervised_dt_weights=x_data_dt_weights,
        collocation_terms=collocation_terms,
        collocation_weights=x_col_weights,
        init_prediction=init_pred,
        init_target=y_init,
        init_weights=x_init_weights,
        weights=weights,
    )


def evaluate_pinn_weighting_terms(
    *,
    model: nn.Module,
    ode_model: Any,
    formulation: str,
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    x_col: torch.Tensor,
    x_init: torch.Tensor,
    y_init: torch.Tensor,
    create_graph: bool = True,
) -> PinnWeightingTerms:
    """Return unreduced per-sample terms used for adaptive weighting updates."""

    x_data_req = prepare_input_for_autograd(x_data)
    pred = model(x_data_req)
    supervised_dt_terms = compute_supervised_dt_terms(
        model=model,
        x=x_data_req,
        y_true=y_data,
        ode_model=ode_model,
        formulation=formulation,
        create_graph=create_graph,
        prediction=pred,
    )
    collocation_terms = compute_residual_terms(
        model=model,
        x=x_col,
        ode_model=ode_model,
        formulation=formulation,
        create_graph=create_graph,
    )
    init_pred = model(x_init)
    return PinnWeightingTerms(
        components={
            "data": pred - y_data,
            "dt": supervised_dt_terms.residual,
            "physics": collocation_terms.residual,
            "ic": init_pred - y_init,
        }
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
    components = {name: nan for name in LOSS_COMPONENTS}

    if bundle.has_supervised:
        pred = model(bundle.x_data)
        loss_data = float(criterion(pred, bundle.y_data).item())
        components["data"] = loss_data
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
        components["dt"] = loss_dt
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
        components["physics"] = loss_physics
        total += float(weights.physics) * loss_physics

    if bundle.has_init:
        init_pred = model(bundle.x_init)
        loss_ic = float(criterion(init_pred, bundle.y_init).item())
        components["ic"] = loss_ic
        total += float(weights.ic) * loss_ic

    return PinnLossScalars(
        total=float(total),
        components=components,
    )
