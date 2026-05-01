"""Compatibility shim for src.training.collocation_runtime."""

from src.training import collocation_runtime as _impl

globals().update({k: v for k, v in _impl.__dict__.items() if not k.startswith("__")})
