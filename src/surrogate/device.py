"""Device selection utilities for surrogate training/inference."""

from __future__ import annotations

import torch


def select_torch_device(preference: str = "auto") -> torch.device:
    """Select torch device from preference.

    Supported values:
    - ``auto``: prefer CUDA, then MPS, else CPU
    - ``cuda``: require CUDA
    - ``mps``: require Apple Metal (MPS)
    - ``cpu``: force CPU
    """
    pref = preference.lower().strip()

    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if pref == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Device preference is 'cuda' but CUDA is not available.")
        return torch.device("cuda")

    if pref == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Device preference is 'mps' but MPS is not available.")
        return torch.device("mps")

    if pref == "cpu":
        return torch.device("cpu")

    raise ValueError("Invalid device preference. Use one of: auto, cuda, mps, cpu.")
