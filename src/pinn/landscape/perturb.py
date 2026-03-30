"""Safe parameter perturbation helpers for loss-landscape analysis."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
from torch import nn

from src.pinn.landscape.directions import ParameterDirection


ParameterState = dict[str, torch.Tensor]


def capture_parameter_state(model: nn.Module) -> ParameterState:
    """Capture a detached clone of the model parameters."""
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }


def restore_parameter_state(model: nn.Module, state: ParameterState) -> None:
    """Restore model parameters from a captured base state."""
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name not in state:
                raise KeyError(f"Missing parameter '{name}' in restore state.")
            parameter.copy_(state[name].to(device=parameter.device, dtype=parameter.dtype))


def apply_parameter_perturbation(
    model: nn.Module,
    *,
    base_state: ParameterState,
    direction_a: ParameterDirection | None = None,
    alpha: float = 0.0,
    direction_b: ParameterDirection | None = None,
    beta: float = 0.0,
) -> None:
    """Apply a linear 1D or 2D perturbation around a base parameter state."""
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name not in base_state:
                raise KeyError(f"Missing parameter '{name}' in base state.")
            updated = base_state[name].to(device=parameter.device, dtype=parameter.dtype)
            if direction_a is not None:
                if name not in direction_a.tensors:
                    raise KeyError(f"Missing parameter '{name}' in direction_a.")
                updated = updated + (float(alpha) * direction_a.tensors[name].to(device=parameter.device, dtype=parameter.dtype))
            if direction_b is not None:
                if name not in direction_b.tensors:
                    raise KeyError(f"Missing parameter '{name}' in direction_b.")
                updated = updated + (float(beta) * direction_b.tensors[name].to(device=parameter.device, dtype=parameter.dtype))
            parameter.copy_(updated)


@contextmanager
def perturbed_model(
    model: nn.Module,
    *,
    base_state: ParameterState,
    direction_a: ParameterDirection | None = None,
    alpha: float = 0.0,
    direction_b: ParameterDirection | None = None,
    beta: float = 0.0,
) -> Iterator[nn.Module]:
    """Temporarily perturb a model and restore its base state afterward."""
    apply_parameter_perturbation(
        model,
        base_state=base_state,
        direction_a=direction_a,
        alpha=alpha,
        direction_b=direction_b,
        beta=beta,
    )
    try:
        yield model
    finally:
        restore_parameter_state(model, base_state)
