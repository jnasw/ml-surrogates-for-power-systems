"""Compatibility shim for src.training.pinn_multistage."""

from src.training import pinn_multistage as _impl

globals().update({k: v for k, v in _impl.__dict__.items() if not k.startswith("__")})
