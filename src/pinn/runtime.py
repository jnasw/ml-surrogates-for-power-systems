"""Runtime helpers for PINN training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


def resolve_torch_dtype(name: str) -> torch.dtype:
    """Resolve a config dtype string to a torch dtype."""
    normalized = str(name).strip().lower()
    if normalized in {"float32", "fp32", "single"}:
        return torch.float32
    if normalized in {"float64", "fp64", "double"}:
        return torch.float64
    raise ValueError("Unsupported dtype. Use one of: float32, float64.")


def torch_dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.float32:
        return "float32"
    if dtype == torch.float64:
        return "float64"
    raise ValueError(f"Unsupported torch dtype: {dtype}")


def torch_dtype_to_numpy(dtype: torch.dtype) -> np.dtype:
    if dtype == torch.float32:
        return np.float32
    if dtype == torch.float64:
        return np.float64
    raise ValueError(f"Unsupported torch dtype: {dtype}")


@dataclass(frozen=True)
class OptimizerStage:
    name: str
    optimizer: str
    lr: float
    epochs: int
    batch_size: int | None
    shuffle: bool


def load_optimizer_stages(config: Any) -> list[OptimizerStage]:
    raw_stages = getattr(config.pinn, "stages", None)
    if raw_stages is None:
        raise ValueError("config.pinn.stages must be configured.")

    stages: list[OptimizerStage] = []
    for idx, item in enumerate(raw_stages):
        optimizer = str(item.optimizer)
        epochs = int(item.epochs)
        batch_size = None if getattr(item, "batch_size", None) in (None, "null") else int(item.batch_size)
        stage = OptimizerStage(
            name=str(getattr(item, "name", f"stage_{idx:02d}")),
            optimizer=optimizer,
            lr=float(item.lr),
            epochs=epochs,
            batch_size=batch_size,
            shuffle=bool(getattr(item, "shuffle", True)),
        )
        if stage.epochs <= 0:
            raise ValueError("Each optimizer stage must have epochs > 0.")
        stages.append(stage)

    return stages

