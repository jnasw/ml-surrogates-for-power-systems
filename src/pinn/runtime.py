"""Runtime helpers for PINN training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from src.pinn.line_search import normalize_line_search_config


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
class OptimizerPhase:
    name: str
    optimizer: str
    lr: float
    epochs: int
    batch_size: int | None
    shuffle: bool
    full_batch: bool | None = None
    allow_sampling: bool | None = None
    optimizer_kwargs: dict[str, Any] = field(default_factory=dict)
    line_search: dict[str, Any] | None = None
    convergence: dict[str, Any] | None = None


def _normalize_optional_mapping(value: Any, field_name: str) -> dict[str, Any] | None:
    if value in (None, "null"):
        return None
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "items"):
        return {str(key): nested for key, nested in value.items()}
    raise ValueError(f"{field_name} must be a mapping or null.")


def _validate_optimizer_phase(phase: OptimizerPhase) -> None:
    if phase.epochs <= 0:
        raise ValueError("Each optimizer phase must have epochs > 0.")
    if phase.lr <= 0.0:
        raise ValueError("Each optimizer phase must have lr > 0.")
    if phase.batch_size is not None and phase.batch_size <= 0:
        raise ValueError("Each optimizer phase batch_size must be > 0 when provided.")

    optimizer_name = phase.optimizer.strip().lower()
    if optimizer_name in {"lbfgs", "bfgs", "ssbfgs", "ssbroyden"}:
        effective_full_batch = True if phase.full_batch is None else bool(phase.full_batch)
        if not effective_full_batch:
            raise ValueError(f"{phase.optimizer} phases must use full_batch=true.")
        if phase.batch_size is not None:
            raise ValueError(f"{phase.optimizer} phases must set batch_size=null because they are full-batch.")
        if phase.allow_sampling is True:
            raise ValueError(f"{phase.optimizer} phases must not enable sampling because curvature updates require a stable objective.")


def load_optimizer_phases_from_raw(
    raw_phases: Any,
    *,
    config_label: str = "config.pinn.optimizer_phases",
) -> list[OptimizerPhase]:
    if raw_phases is None:
        raise ValueError(f"{config_label} must be configured.")
    phases: list[OptimizerPhase] = []
    for idx, item in enumerate(raw_phases):
        optimizer = str(item.optimizer)
        epochs = int(item.epochs)
        batch_size = None if getattr(item, "batch_size", None) in (None, "null") else int(item.batch_size)
        optimizer_kwargs = _normalize_optional_mapping(
            getattr(item, "optimizer_kwargs", None),
            field_name="optimizer_kwargs",
        )
        line_search = normalize_line_search_config(
            _normalize_optional_mapping(
                getattr(item, "line_search", None),
                field_name="line_search",
            )
        )
        convergence = _normalize_optional_mapping(
            getattr(item, "convergence", None),
            field_name="convergence",
        )
        phase = OptimizerPhase(
            name=str(getattr(item, "name", f"phase_{idx:02d}")),
            optimizer=optimizer,
            lr=float(item.lr),
            epochs=epochs,
            batch_size=batch_size,
            shuffle=bool(getattr(item, "shuffle", True)),
            full_batch=None if getattr(item, "full_batch", None) in (None, "null") else bool(getattr(item, "full_batch")),
            allow_sampling=None if getattr(item, "allow_sampling", None) in (None, "null") else bool(getattr(item, "allow_sampling")),
            optimizer_kwargs={} if optimizer_kwargs is None else optimizer_kwargs,
            line_search=line_search,
            convergence=convergence,
        )
        _validate_optimizer_phase(phase)
        phases.append(phase)

    return phases


def load_optimizer_phases(config: Any) -> list[OptimizerPhase]:
    raw_phases = getattr(config.pinn, "optimizer_phases", None)
    if raw_phases is None:
        raw_phases = getattr(config.pinn, "stages", None)
    return load_optimizer_phases_from_raw(raw_phases, config_label="config.pinn.optimizer_phases")


def load_optimizer_stages(config: Any) -> list[OptimizerPhase]:
    """Backward-compatible alias for older code paths."""
    return load_optimizer_phases(config)
