"""Utilities to load IC bounds from config YAML files."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from omegaconf import OmegaConf


def load_ic_bounds(config: Any, use_nn_file: bool = False) -> np.ndarray:
    bounds_dir = str(config.dirs.init_conditions_dir)
    model_flag = str(config.model.model_flag)
    suffix = int(config.model.init_condition_bounds)
    prefix = "nn_init_cond" if use_nn_file else "init_cond"
    path = os.path.join(bounds_dir, model_flag, f"{prefix}{suffix}.yaml")
    if not os.path.exists(path):
        raise FileNotFoundError(f"IC bounds file not found: {path}")

    entries = OmegaConf.load(path)
    bounds = []
    for e in entries:
        r = list(e["range"])
        if len(r) == 1:
            v = float(r[0])
            bounds.append([v, v])
        else:
            bounds.append([float(r[0]), float(r[1])])
    return np.asarray(bounds, dtype=np.float32)
