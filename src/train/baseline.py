"""Compatibility shim for src.training.baseline."""

from src.training import baseline as _impl

globals().update({k: v for k, v in _impl.__dict__.items() if not k.startswith("__")})
