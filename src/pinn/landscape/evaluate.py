"""Raw 1D/2D loss-landscape evaluation for PINNs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from src.pinn.analysis_data import PinnAnalysisBundle
from src.pinn.evaluator import PinnLossScalars, evaluate_analysis_loss_scalars
from src.pinn.landscape.directions import ParameterDirection, sample_random_directions
from src.pinn.landscape.perturb import ParameterState, capture_parameter_state, perturbed_model
from src.pinn.losses import LossWeights


@dataclass(frozen=True)
class LandscapeAxes1D:
    alpha: np.ndarray


@dataclass(frozen=True)
class LandscapeAxes2D:
    alpha: np.ndarray
    beta: np.ndarray


@dataclass(frozen=True)
class RawLandscape1D:
    axes: LandscapeAxes1D
    total: np.ndarray
    data: np.ndarray
    dt: np.ndarray
    physics: np.ndarray
    ic: np.ndarray
    direction: ParameterDirection


@dataclass(frozen=True)
class RawLandscape2D:
    axes: LandscapeAxes2D
    total: np.ndarray
    data: np.ndarray
    dt: np.ndarray
    physics: np.ndarray
    ic: np.ndarray
    direction_a: ParameterDirection
    direction_b: ParameterDirection


def _default_criterion() -> nn.Module:
    return nn.MSELoss()


def _scalar_to_tuple(losses: PinnLossScalars) -> tuple[float, float, float, float, float]:
    return losses.total, losses.data, losses.dt, losses.physics, losses.ic


def evaluate_landscape_1d(
    *,
    model: nn.Module,
    bundle: PinnAnalysisBundle,
    ode_model: Any,
    weights: LossWeights,
    formulation: str,
    direction: ParameterDirection | None = None,
    base_state: ParameterState | None = None,
    alpha_values: np.ndarray | None = None,
    seed: int = 0,
    normalization: str = "filter",
    criterion: nn.Module | None = None,
) -> RawLandscape1D:
    """Evaluate a 1D loss slice around a base checkpoint."""
    criterion = _default_criterion() if criterion is None else criterion
    direction_obj = (
        sample_random_directions(model, seed=seed, normalization=normalization, count=1)[0]
        if direction is None
        else direction
    )
    base_state_obj = capture_parameter_state(model) if base_state is None else base_state
    alpha = np.asarray(
        np.linspace(-1.0, 1.0, 21, dtype=np.float64) if alpha_values is None else alpha_values,
        dtype=np.float64,
    )

    total = np.empty(alpha.shape, dtype=np.float64)
    data = np.empty(alpha.shape, dtype=np.float64)
    dt = np.empty(alpha.shape, dtype=np.float64)
    physics = np.empty(alpha.shape, dtype=np.float64)
    ic = np.empty(alpha.shape, dtype=np.float64)

    for idx, alpha_value in enumerate(alpha):
        with perturbed_model(
            model,
            base_state=base_state_obj,
            direction_a=direction_obj,
            alpha=float(alpha_value),
        ):
            losses = evaluate_analysis_loss_scalars(
                model=model,
                criterion=criterion,
                ode_model=ode_model,
                formulation=formulation,
                weights=weights,
                bundle=bundle,
            )
        total[idx], data[idx], dt[idx], physics[idx], ic[idx] = _scalar_to_tuple(losses)

    return RawLandscape1D(
        axes=LandscapeAxes1D(alpha=alpha),
        total=total,
        data=data,
        dt=dt,
        physics=physics,
        ic=ic,
        direction=direction_obj,
    )


def evaluate_landscape_2d(
    *,
    model: nn.Module,
    bundle: PinnAnalysisBundle,
    ode_model: Any,
    weights: LossWeights,
    formulation: str,
    direction_a: ParameterDirection | None = None,
    direction_b: ParameterDirection | None = None,
    base_state: ParameterState | None = None,
    alpha_values: np.ndarray | None = None,
    beta_values: np.ndarray | None = None,
    seed: int = 0,
    normalization: str = "filter",
    criterion: nn.Module | None = None,
) -> RawLandscape2D:
    """Evaluate a 2D loss surface around a base checkpoint."""
    criterion = _default_criterion() if criterion is None else criterion
    if direction_a is None or direction_b is None:
        sampled = sample_random_directions(model, seed=seed, normalization=normalization, count=2)
        if direction_a is None:
            direction_a = sampled[0]
        if direction_b is None:
            direction_b = sampled[1]
    base_state_obj = capture_parameter_state(model) if base_state is None else base_state
    alpha = np.asarray(
        np.linspace(-1.0, 1.0, 21, dtype=np.float64) if alpha_values is None else alpha_values,
        dtype=np.float64,
    )
    beta = np.asarray(
        np.linspace(-1.0, 1.0, 21, dtype=np.float64) if beta_values is None else beta_values,
        dtype=np.float64,
    )

    shape = (alpha.shape[0], beta.shape[0])
    total = np.empty(shape, dtype=np.float64)
    data = np.empty(shape, dtype=np.float64)
    dt = np.empty(shape, dtype=np.float64)
    physics = np.empty(shape, dtype=np.float64)
    ic = np.empty(shape, dtype=np.float64)

    for alpha_idx, alpha_value in enumerate(alpha):
        for beta_idx, beta_value in enumerate(beta):
            with perturbed_model(
                model,
                base_state=base_state_obj,
                direction_a=direction_a,
                alpha=float(alpha_value),
                direction_b=direction_b,
                beta=float(beta_value),
            ):
                losses = evaluate_analysis_loss_scalars(
                    model=model,
                    criterion=criterion,
                    ode_model=ode_model,
                    formulation=formulation,
                    weights=weights,
                    bundle=bundle,
                )
            total[alpha_idx, beta_idx], data[alpha_idx, beta_idx], dt[alpha_idx, beta_idx], physics[alpha_idx, beta_idx], ic[alpha_idx, beta_idx] = _scalar_to_tuple(losses)

    return RawLandscape2D(
        axes=LandscapeAxes2D(alpha=alpha, beta=beta),
        total=total,
        data=data,
        dt=dt,
        physics=physics,
        ic=ic,
        direction_a=direction_a,
        direction_b=direction_b,
    )
