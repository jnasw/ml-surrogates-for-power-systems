"""Backward-compatible PINN trainer exports."""

from __future__ import annotations

from src.train.trainer import PinnModel, train_pinn

__all__ = ["PinnModel", "train_pinn"]
