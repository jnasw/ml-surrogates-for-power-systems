"""Shared helpers for direct-run regression metrics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import nn


def regression_metrics_from_arrays(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute basic regression metrics from aligned arrays."""
    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    if true.shape != pred.shape:
        raise ValueError(f"Regression metric arrays must match, got {true.shape} vs {pred.shape}.")

    diff = pred - true
    mse = float(np.mean(diff**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
    }


def regression_metrics_for_torch_model(
    *,
    model: nn.Module,
    x: torch.Tensor | None,
    y: torch.Tensor | None,
    batch_size: int = 4096,
) -> dict[str, float] | None:
    """Compute basic regression metrics for a torch model on one split."""
    if x is None or y is None:
        return None
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same leading dimension.")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be > 0.")

    param = next(model.parameters())
    device = param.device
    dtype = param.dtype

    total_squared = 0.0
    total_abs = 0.0
    total_count = 0

    model.eval()
    with torch.no_grad():
        for start in range(0, int(x.shape[0]), int(batch_size)):
            stop = min(start + int(batch_size), int(x.shape[0]))
            xb = x[start:stop].to(device=device, dtype=dtype)
            yb = y[start:stop].to(device=device, dtype=dtype)
            pred = model(xb)
            diff = pred - yb
            total_squared += float(diff.pow(2).sum().item())
            total_abs += float(diff.abs().sum().item())
            total_count += int(diff.numel())

    if total_count <= 0:
        raise ValueError("Cannot compute regression metrics on an empty split.")

    mse = total_squared / float(total_count)
    rmse = math.sqrt(mse)
    mae = total_abs / float(total_count)
    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
    }


def build_run_summary(
    *,
    run_type: str,
    model_flag: str,
    dataset_root: str,
    seed: int | None,
    dataset_number: int | None = None,
    split_sizes: dict[str, int | None] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "run_type": str(run_type),
        "model_flag": str(model_flag),
        "dataset_root": str(dataset_root),
        "seed": None if seed is None else int(seed),
        "dataset_number": None if dataset_number is None else int(dataset_number),
        "split_sizes": {} if split_sizes is None else dict(split_sizes),
    }
    if extra:
        payload.update(dict(extra))
    return payload
