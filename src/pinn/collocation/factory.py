"""Factory for collocation strategies."""

from __future__ import annotations

from typing import Any

import torch

from src.data.generate.bounds import load_ic_bounds
from src.pinn.collocation.domain import CollocationDomain, sample_collocation_points
from src.pinn.collocation.strategies import (
    CollocationStrategy,
    ResidualAdaptiveRefinementDistributionStrategy,
    ResidualAdaptiveRefinementGreedyStrategy,
    ResidualAdaptiveDistributionStrategy,
    StaticCollocationStrategy,
    UniformResampleCollocationStrategy,
)
from src.train.runtime import cfg_get


def build_collocation_strategy(*, initial_points: torch.Tensor, config: Any) -> CollocationStrategy:
    mode = str(cfg_get(config, "pinn.collocation.mode", "preprocessed")).strip().lower()
    strategy_name = str(cfg_get(config, "pinn.collocation.strategy", "static")).strip().lower()
    seed = int(cfg_get(config, "pinn.collocation.seed", cfg_get(config, "model.seed", 0)))
    sampler = str(cfg_get(config, "pinn.collocation.sampler", "lhs"))
    domain = _build_collocation_domain(config=config, fallback_points=initial_points)
    active_points = _resolve_active_points(config=config, fallback_points=initial_points)
    initial_strategy_points = _resolve_initial_strategy_points(
        config=config,
        fallback_points=initial_points,
        target_points=active_points,
        strategy_name=strategy_name,
    )

    if mode == "generated":
        initial_points = sample_collocation_points(
            domain=domain,
            n=initial_strategy_points,
            method=sampler,
            seed=seed,
            dtype=initial_points.dtype,
            device=initial_points.device,
        )
    elif mode != "preprocessed":
        raise ValueError("pinn.collocation.mode must be one of: preprocessed, generated")

    if strategy_name == "static":
        return StaticCollocationStrategy(initial_points=initial_points, config=config)
    initial_points = _normalize_initial_points(
        initial_points=initial_points,
        active_points=initial_strategy_points,
        seed=seed,
    )
    if strategy_name == "random_r":
        return UniformResampleCollocationStrategy(
            initial_points=initial_points,
            config=config,
            domain=domain,
            sampler=sampler,
            seed=seed,
            active_points=active_points,
        )
    if strategy_name == "rad":
        candidate_points = int(
            cfg_get(config, "pinn.collocation.candidate_points", max(active_points * 4, active_points))
        )
        return ResidualAdaptiveDistributionStrategy(
            initial_points=initial_points,
            config=config,
            domain=domain,
            sampler=sampler,
            seed=seed,
            active_points=active_points,
            candidate_points=candidate_points,
            rad_k=float(cfg_get(config, "pinn.collocation.rad.k", 1.0)),
            rad_c=float(cfg_get(config, "pinn.collocation.rad.c", 1.0)),
            score_norm=str(cfg_get(config, "pinn.collocation.score_norm", "l2")),
        )
    if strategy_name == "rar_d":
        candidate_points = int(
            cfg_get(config, "pinn.collocation.candidate_points", max(active_points * 4, active_points))
        )
        return ResidualAdaptiveRefinementDistributionStrategy(
            initial_points=initial_points,
            config=config,
            domain=domain,
            sampler=sampler,
            seed=seed,
            target_points=active_points,
            candidate_points=candidate_points,
            append_points=int(cfg_get(config, "pinn.collocation.append_points", 64)),
            rad_k=float(cfg_get(config, "pinn.collocation.rad.k", 2.0)),
            rad_c=float(cfg_get(config, "pinn.collocation.rad.c", 0.0)),
            score_norm=str(cfg_get(config, "pinn.collocation.score_norm", "l2")),
        )
    if strategy_name == "rar_g":
        candidate_points = int(
            cfg_get(config, "pinn.collocation.candidate_points", max(active_points * 4, active_points))
        )
        return ResidualAdaptiveRefinementGreedyStrategy(
            initial_points=initial_points,
            config=config,
            domain=domain,
            sampler=sampler,
            seed=seed,
            target_points=active_points,
            candidate_points=candidate_points,
            append_points=int(cfg_get(config, "pinn.collocation.append_points", 64)),
            score_norm=str(cfg_get(config, "pinn.collocation.score_norm", "l2")),
        )
    raise NotImplementedError(
        "Supported pinn.collocation.strategy values are: static, random_r, rad, rar_d, rar_g."
    )


def _resolve_active_points(*, config: Any, fallback_points: torch.Tensor) -> int:
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
    return active_points


def _build_collocation_domain(*, config: Any, fallback_points: torch.Tensor) -> CollocationDomain:
    feature_bounds = torch.as_tensor(load_ic_bounds(config, use_nn_file=True), dtype=torch.float32)
    input_dim = int(fallback_points.shape[1]) if int(fallback_points.shape[1]) > 0 else int(feature_bounds.shape[0]) + 1
    return CollocationDomain(
        time_min=0.0,
        time_max=float(cfg_get(config, "time", 1.0)),
        input_dim=input_dim,
        feature_bounds=feature_bounds,
    )


def _resolve_initial_strategy_points(
    *,
    config: Any,
    fallback_points: torch.Tensor,
    target_points: int,
    strategy_name: str,
) -> int:
    cfg_value = cfg_get(config, "pinn.collocation.initial_points", None)
    if cfg_value not in (None, "null"):
        initial_points = int(cfg_value)
    elif strategy_name in {"rar_d", "rar_g"}:
        initial_points = max(1, target_points // 2)
    else:
        fallback_rows = int(fallback_points.shape[0])
        initial_points = target_points if fallback_rows <= 0 else min(target_points, fallback_rows)

    if initial_points <= 0:
        raise ValueError("pinn.collocation.initial_points must be > 0 when provided.")
    if initial_points > target_points:
        raise ValueError("pinn.collocation.initial_points must be <= pinn.collocation.active_points.")
    return initial_points


def _normalize_initial_points(*, initial_points: torch.Tensor, active_points: int, seed: int) -> torch.Tensor:
    rows = int(initial_points.shape[0])
    if rows == active_points:
        return initial_points
    if rows < active_points:
        raise ValueError(
            "Initial collocation set is smaller than pinn.collocation.active_points. "
            "Use generated mode or lower active_points."
        )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    indices = torch.randperm(rows, generator=generator, device="cpu")[:active_points].to(initial_points.device)
    return initial_points.index_select(0, indices)
