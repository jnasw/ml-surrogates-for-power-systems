"""Compatibility shim for src.training.model."""

from src.training import model as _impl

globals().update({k: v for k, v in _impl.__dict__.items() if not k.startswith("__")})
