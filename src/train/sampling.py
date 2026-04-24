"""Compatibility shim for src.training.sampling."""

from src.training import sampling as _impl

globals().update({k: v for k, v in _impl.__dict__.items() if not k.startswith("__")})
