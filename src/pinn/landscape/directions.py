"""Direction sampling and normalization utilities for loss-landscape analysis."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ParameterDirection:
    """A named collection of parameter-space directions."""

    tensors: dict[str, torch.Tensor]
    seed: int
    normalization: str

    def clone(self) -> "ParameterDirection":
        return ParameterDirection(
            tensors={name: tensor.clone() for name, tensor in self.tensors.items()},
            seed=int(self.seed),
            normalization=str(self.normalization),
        )


def _parameter_rows(tensor: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    if tensor.ndim == 0:
        return tensor.reshape(1, 1), tuple(tensor.shape)
    if tensor.ndim == 1:
        return tensor.reshape(1, -1), tuple(tensor.shape)
    return tensor.reshape(tensor.shape[0], -1), tuple(tensor.shape)


def _normalize_like_parameter(direction: torch.Tensor, parameter: torch.Tensor) -> torch.Tensor:
    if direction.shape != parameter.shape:
        raise ValueError("Direction and parameter shapes must match for normalization.")
    param_norm = torch.linalg.vector_norm(parameter)
    dir_norm = torch.linalg.vector_norm(direction)
    if float(param_norm.item()) == 0.0 or float(dir_norm.item()) == 0.0:
        return torch.zeros_like(direction)
    return direction * (param_norm / dir_norm)


def _normalize_like_filter(direction: torch.Tensor, parameter: torch.Tensor) -> torch.Tensor:
    if direction.shape != parameter.shape:
        raise ValueError("Direction and parameter shapes must match for normalization.")
    if parameter.ndim <= 1:
        return _normalize_like_parameter(direction, parameter)

    direction_rows, original_shape = _parameter_rows(direction)
    parameter_rows, _ = _parameter_rows(parameter)
    dir_norms = torch.linalg.vector_norm(direction_rows, dim=1, keepdim=True)
    param_norms = torch.linalg.vector_norm(parameter_rows, dim=1, keepdim=True)
    safe_scale = torch.zeros_like(param_norms)
    nonzero = (dir_norms.squeeze(1) > 0) & (param_norms.squeeze(1) > 0)
    safe_scale[nonzero] = param_norms[nonzero] / dir_norms[nonzero]
    normalized = direction_rows * safe_scale
    return normalized.reshape(original_shape)


def sample_random_direction(
    model: nn.Module,
    *,
    seed: int = 0,
    normalization: str = "filter",
) -> ParameterDirection:
    """Sample a reproducible random direction aligned with a model's parameters."""
    normalized = str(normalization).strip().lower()
    if normalized not in {"filter", "parameter"}:
        raise ValueError("normalization must be one of: filter, parameter")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    tensors: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        raw = torch.randn(
            parameter.shape,
            generator=generator,
            device=parameter.device,
            dtype=parameter.dtype,
        )
        if normalized == "filter":
            raw = _normalize_like_filter(raw, parameter.detach())
        elif normalized == "parameter":
            raw = _normalize_like_parameter(raw, parameter.detach())
        tensors[name] = raw

    return ParameterDirection(tensors=tensors, seed=int(seed), normalization=normalized)


def sample_random_directions(
    model: nn.Module,
    *,
    seed: int = 0,
    normalization: str = "filter",
    count: int = 2,
) -> list[ParameterDirection]:
    """Sample multiple reproducible directions for one model."""
    if count <= 0:
        raise ValueError("count must be > 0.")
    return [
        sample_random_direction(
            model,
            seed=int(seed) + idx,
            normalization=normalization,
        )
        for idx in range(count)
    ]
