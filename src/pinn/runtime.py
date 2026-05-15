"""Stable front door for PINN runtime helpers and resolved-stack imports."""

from __future__ import annotations

import numpy as np
import torch

from src.pinn.runtime_stack import (
    OptimizerPhase,
    ResolvedCollocationSettings,
    ResolvedCurriculumSettings,
    ResolvedLossWeightScheduleStage,
    ResolvedMultistageOptimizerSchedule,
    ResolvedPinnStack,
    ResolvedSupervisedAcquisitionSettings,
    SchedulerConfig,
    load_optimizer_phases,
    load_optimizer_phases_from_raw,
    resolve_pinn_stack,
    resolve_pinn_training_mode,
)


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


__all__ = [
    "OptimizerPhase",
    "ResolvedCollocationSettings",
    "ResolvedCurriculumSettings",
    "ResolvedLossWeightScheduleStage",
    "ResolvedMultistageOptimizerSchedule",
    "ResolvedPinnStack",
    "ResolvedSupervisedAcquisitionSettings",
    "SchedulerConfig",
    "load_optimizer_phases",
    "load_optimizer_phases_from_raw",
    "resolve_pinn_stack",
    "resolve_pinn_training_mode",
    "resolve_torch_dtype",
    "torch_dtype_name",
    "torch_dtype_to_numpy",
]
