"""Shared runtime helpers for neural surrogate training entrypoints."""

from __future__ import annotations

import json
import os
import random
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf


def cfg_get(cfg: Any, key: str, default: Any) -> Any:
    cur = cfg
    for part in key.split("."):
        if isinstance(cur, dict):
            if part not in cur:
                return default
            cur = cur[part]
        else:
            if not hasattr(cur, part):
                return default
            cur = getattr(cur, part)
    return cur


def resolve_path_from_cwd(path_value: str, original_cwd: str) -> str:
    path = str(path_value)
    return path if os.path.isabs(path) else os.path.join(original_cwd, path)


def resolve_dataset_root(config: Any, original_cwd: str) -> str:
    dataset_root_cfg = cfg_get(config, "dataset.root", None)
    if dataset_root_cfg not in (None, ""):
        return resolve_path_from_cwd(str(dataset_root_cfg), original_cwd)

    dataset_dir = resolve_path_from_cwd(str(cfg_get(config, "dirs.dataset_dir", "data")), original_cwd)
    model_flag = str(cfg_get(config, "model.model_flag", ""))
    dataset_number = int(cfg_get(config, "dataset.number", 1))
    return os.path.join(dataset_dir, model_flag, f"dataset_v{dataset_number}")


def resolve_run_dir(
    config: Any,
    section: str,
    original_cwd: str,
    legacy_key: str | None = None,
) -> str:
    run_dir_cfg = cfg_get(config, f"{section}.run_dir", None)
    if run_dir_cfg in (None, "") and legacy_key is not None:
        run_dir_cfg = cfg_get(config, legacy_key, None)
    if run_dir_cfg in (None, ""):
        raise ValueError(f"Missing run directory config for '{section}'.")
    return resolve_path_from_cwd(str(run_dir_cfg), original_cwd)


def save_resolved_config(config: Any, run_dir: str) -> None:
    os.makedirs(run_dir, exist_ok=True)
    OmegaConf.save(
        config=OmegaConf.create(OmegaConf.to_container(config, resolve=True)),
        f=os.path.join(run_dir, "config.yaml"),
    )


def write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def configure_reproducibility(
    seed: int,
    deterministic: bool = False,
    strict: bool = False,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if not deterministic:
        return

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=not strict)


def configure_reproducibility_from_config(seed: int, config: Any, prefix: str) -> None:
    configure_reproducibility(
        seed=seed,
        deterministic=bool(cfg_get(config, f"{prefix}.deterministic", False)),
        strict=bool(cfg_get(config, f"{prefix}.deterministic_strict", False)),
    )
