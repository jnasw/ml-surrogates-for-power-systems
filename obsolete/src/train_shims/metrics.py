"""Compatibility shim for src.training.metrics."""

from src.training import metrics as _impl

globals().update({k: v for k, v in _impl.__dict__.items() if not k.startswith("__")})
