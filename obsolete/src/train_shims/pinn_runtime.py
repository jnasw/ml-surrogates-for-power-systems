"""Compatibility shim for src.training.pinn_runtime."""

from src.training import pinn_runtime as _impl

globals().update({k: v for k, v in _impl.__dict__.items() if not k.startswith("__")})
