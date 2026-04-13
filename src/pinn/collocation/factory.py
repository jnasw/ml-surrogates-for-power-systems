"""Factory for collocation strategies."""

from __future__ import annotations

from typing import Any

import torch

from src.data.generate.bounds import load_ic_bounds
from src.pinn.collocation.domain import CollocationDomain, sample_collocation_points
from src.pinn.collocation.strategies import CollocationStrategy, StaticCollocationStrategy
from src.train.runtime import cfg_get


def build_collocation_strategy(*, initial_points: torch.Tensor, config: Any) -> CollocationStrategy:
    mode = str(cfg_get(config, "pinn.collocation.mode", "preprocessed")).strip().lower()
    strategy_name = str(cfg_get(config, "pinn.collocation.strategy", "static")).strip().lower()

    if mode == "generated":
        initial_points = _build_generated_initial_points(config=config, fallback_points=initial_points)
    elif mode != "preprocessed":
        raise ValueError("pinn.collocation.mode must be one of: preprocessed, generated")
    if strategy_name != "static":
        raise NotImplementedError(
            "Only pinn.collocation.strategy='static' is implemented in Phase 2."
        )
    return StaticCollocationStrategy(initial_points=initial_points, config=config)


def _build_generated_initial_points(*, config: Any, fallback_points: torch.Tensor) -> torch.Tensor:
    if fallback_points.ndim != 2:
        raise ValueError("fallback_points must be a rank-2 tensor.")

    active_points_cfg = cfg_get(config, "pinn.collocation.active_points", None)
    if active_points_cfg in (None, "null"):
        fallback_rows = int(fallback_points.shape[0])
        if fallback_rows <= 0:
            raise ValueError(
                "pinn.collocation.active_points must be set when generated mode is used "
                "without preprocessed collocation rows."
            )
        active_points = fallback_rows
    else:
        active_points = int(active_points_cfg)
    if active_points <= 0:
        raise ValueError("pinn.collocation.active_points must be > 0.")

    feature_bounds = torch.as_tensor(load_ic_bounds(config, use_nn_file=True), dtype=torch.float32)
    input_dim = int(fallback_points.shape[1]) if int(fallback_points.shape[1]) > 0 else int(feature_bounds.shape[0]) + 1
    domain = CollocationDomain(
        time_min=0.0,
        time_max=float(cfg_get(config, "time", 1.0)),
        input_dim=input_dim,
        feature_bounds=feature_bounds,
    )
    return sample_collocation_points(
        domain=domain,
        n=active_points,
        method=str(cfg_get(config, "pinn.collocation.sampler", "lhs")),
        seed=int(cfg_get(config, "pinn.collocation.seed", cfg_get(config, "model.seed", 0))),
        dtype=fallback_points.dtype,
        device=fallback_points.device,
    )
