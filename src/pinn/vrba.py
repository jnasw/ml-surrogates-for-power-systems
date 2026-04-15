"""Configuration and dormant state helpers for PINN vRBA integration.

Stage 0/1 only:
- define the configuration surface
- define serializable state containers
- provide initialization and serialization helpers

This module is intentionally behavior-neutral. No existing training path uses
the returned state to alter collocation sampling or the loss function yet.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from src.pinn.losses import LOSS_COMPONENTS
from src.train.runtime import cfg_get


SUPPORTED_VRBA_TARGET_SETS: tuple[str, ...] = LOSS_COMPONENTS
SUPPORTED_VRBA_POTENTIALS: tuple[str, ...] = ("quadratic", "exponential")


@dataclass(frozen=True)
class VrbAConfig:
    enabled: bool
    adaptive_sampling: bool
    adaptive_weighting: bool
    target_sets: tuple[str, ...]
    potential: str
    eta: float
    lambda_max0: float
    lambda_cap: float
    phi: float
    update_interval_epochs: int
    freeze_sampling_during_full_batch: bool
    freeze_weighting_during_full_batch: bool
    track_in_checkpoints: bool

    @property
    def is_active(self) -> bool:
        return bool(self.enabled and (self.adaptive_sampling or self.adaptive_weighting))


@dataclass
class VrbASetState:
    point_count: int | None = None
    update_count: int = 0
    last_update_epoch: int | None = None
    last_refresh_epoch: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VrbAState:
    sets: dict[str, VrbASetState]
    global_update_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


def vrba_config_from_config(config: Any) -> VrbAConfig:
    raw_target_sets = cfg_get(config, "pinn.vrba.target_sets", ["physics"])
    target_sets = tuple(str(name).strip().lower() for name in raw_target_sets)
    if not target_sets:
        raise ValueError("pinn.vrba.target_sets must not be empty.")
    if len(set(target_sets)) != len(target_sets):
        raise ValueError("pinn.vrba.target_sets must not contain duplicates.")
    for name in target_sets:
        if name not in SUPPORTED_VRBA_TARGET_SETS:
            supported = ", ".join(SUPPORTED_VRBA_TARGET_SETS)
            raise ValueError(f"Unsupported pinn.vrba.target_sets entry '{name}'. Use one of: {supported}.")

    potential = str(cfg_get(config, "pinn.vrba.potential", "quadratic")).strip().lower()
    if potential not in SUPPORTED_VRBA_POTENTIALS:
        supported = ", ".join(SUPPORTED_VRBA_POTENTIALS)
        raise ValueError(f"pinn.vrba.potential must be one of: {supported}.")

    eta = float(cfg_get(config, "pinn.vrba.eta", 0.1))
    if eta <= 0.0:
        raise ValueError("pinn.vrba.eta must be > 0.")
    lambda_max0 = float(cfg_get(config, "pinn.vrba.lambda_max0", 15.0))
    if lambda_max0 <= 0.0:
        raise ValueError("pinn.vrba.lambda_max0 must be > 0.")
    lambda_cap = float(cfg_get(config, "pinn.vrba.lambda_cap", 25.0))
    if lambda_cap < lambda_max0:
        raise ValueError("pinn.vrba.lambda_cap must be >= pinn.vrba.lambda_max0.")
    phi = float(cfg_get(config, "pinn.vrba.phi", 1.0))
    if not (0.0 <= phi <= 1.0):
        raise ValueError("pinn.vrba.phi must be in [0, 1].")
    update_interval_epochs = int(cfg_get(config, "pinn.vrba.update_interval_epochs", 1))
    if update_interval_epochs <= 0:
        raise ValueError("pinn.vrba.update_interval_epochs must be > 0.")

    return VrbAConfig(
        enabled=bool(cfg_get(config, "pinn.vrba.enabled", False)),
        adaptive_sampling=bool(cfg_get(config, "pinn.vrba.adaptive_sampling", False)),
        adaptive_weighting=bool(cfg_get(config, "pinn.vrba.adaptive_weighting", False)),
        target_sets=target_sets,
        potential=potential,
        eta=eta,
        lambda_max0=lambda_max0,
        lambda_cap=lambda_cap,
        phi=phi,
        update_interval_epochs=update_interval_epochs,
        freeze_sampling_during_full_batch=bool(cfg_get(config, "pinn.vrba.freeze_sampling_during_full_batch", True)),
        freeze_weighting_during_full_batch=bool(cfg_get(config, "pinn.vrba.freeze_weighting_during_full_batch", True)),
        track_in_checkpoints=bool(cfg_get(config, "pinn.vrba.track_in_checkpoints", True)),
    )


def initialize_vrba_state(
    vrba_config: VrbAConfig,
    *,
    initial_point_counts: Mapping[str, int] | None = None,
) -> VrbAState:
    point_counts = {} if initial_point_counts is None else {str(k): int(v) for k, v in initial_point_counts.items()}
    sets = {
        name: VrbASetState(point_count=point_counts.get(name))
        for name in vrba_config.target_sets
    }
    return VrbAState(
        sets=sets,
        metadata={
            "enabled": bool(vrba_config.enabled),
            "adaptive_sampling": bool(vrba_config.adaptive_sampling),
            "adaptive_weighting": bool(vrba_config.adaptive_weighting),
            "potential": str(vrba_config.potential),
        },
    )


def serialize_vrba_config(vrba_config: VrbAConfig) -> dict[str, Any]:
    payload = asdict(vrba_config)
    payload["target_sets"] = list(vrba_config.target_sets)
    payload["is_active"] = bool(vrba_config.is_active)
    return payload


def serialize_vrba_state(vrba_state: VrbAState | None) -> dict[str, Any] | None:
    if vrba_state is None:
        return None
    payload = asdict(vrba_state)
    payload["sets"] = {
        name: asdict(state)
        for name, state in vrba_state.sets.items()
    }
    return payload
