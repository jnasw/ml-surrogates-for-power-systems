"""Compatibility shim for src.training.pinn_single_stage."""

from src.training import pinn_single_stage as _impl

globals().update({k: v for k, v in _impl.__dict__.items() if not k.startswith("__")})
