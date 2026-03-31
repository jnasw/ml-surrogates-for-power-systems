"""Low-level parameter-vector helpers for PINN optimizers."""

from __future__ import annotations

import torch
from torch import nn


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [param for param in model.parameters() if param.requires_grad]


def count_trainable_parameters(model: nn.Module) -> int:
    return int(sum(param.numel() for param in trainable_parameters(model)))


def parameters_to_flat_vector(model: nn.Module) -> torch.Tensor:
    params = trainable_parameters(model)
    if not params:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat([param.detach().reshape(-1) for param in params], dim=0)


def clone_parameters_to_flat_vector(model: nn.Module) -> torch.Tensor:
    params = trainable_parameters(model)
    if not params:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat([param.detach().clone().reshape(-1) for param in params], dim=0)


def gradients_to_flat_vector(model: nn.Module) -> torch.Tensor:
    params = trainable_parameters(model)
    if not params:
        return torch.empty(0, dtype=torch.float32)
    vectors = []
    for param in params:
        grad = param.grad
        if grad is None:
            vectors.append(torch.zeros_like(param, memory_format=torch.contiguous_format).reshape(-1))
        else:
            vectors.append(grad.detach().reshape(-1))
    return torch.cat(vectors, dim=0)


@torch.no_grad()
def set_parameters_from_flat_vector(model: nn.Module, vec: torch.Tensor) -> None:
    params = trainable_parameters(model)
    expected = sum(param.numel() for param in params)
    if int(vec.numel()) != int(expected):
        raise ValueError(f"Flat parameter vector has {vec.numel()} elements but model expects {expected}.")

    offset = 0
    for param in params:
        numel = param.numel()
        reshaped = vec[offset : offset + numel].view_as(param).to(device=param.device, dtype=param.dtype)
        param.copy_(reshaped)
        offset += numel


def flat_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.dot(a.reshape(-1), b.reshape(-1))


def flat_norm(a: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(a.reshape(-1), ord=2)
