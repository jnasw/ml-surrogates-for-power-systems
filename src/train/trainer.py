"""Compatibility shim for src.training.trainer."""

from src.training import trainer as _impl

globals().update({k: v for k, v in _impl.__dict__.items() if not k.startswith("__")})
