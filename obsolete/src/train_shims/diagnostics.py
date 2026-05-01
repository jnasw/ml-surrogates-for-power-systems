"""Compatibility shim for src.training.diagnostics."""

from src.training import diagnostics as _impl

globals().update({k: v for k, v in _impl.__dict__.items() if not k.startswith("__")})
