"""Compatibility shim for src.training.device."""

from src.training import device as _impl

globals().update({k: v for k, v in _impl.__dict__.items() if not k.startswith("__")})
