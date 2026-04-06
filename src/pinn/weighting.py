"""Dynamic loss-weighting policies for PINN training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from src.pinn.losses import LOSS_COMPONENTS, LossWeights
from src.train.runtime import cfg_get


SUPPORTED_WEIGHTING_SCHEMES: tuple[str, ...] = ("static", "ma", "id", "dn")
DEFAULT_DYNAMIC_COMPONENTS: tuple[str, ...] = ("data", "dt", "ic")
DEFAULT_WEIGHTING_ANCHOR = "physics"


@dataclass(frozen=True)
class WeightUpdateStats:
    grad_l2_norms: dict[str, float]
    grad_mean_abs: dict[str, float]
    grad_max_abs: dict[str, float]
    grad_std: dict[str, float]
    anchor_component: str
    epoch: int
    global_epoch: int


@dataclass(frozen=True)
class WeightingState:
    active_weights: LossWeights
    raw_candidate_weights: dict[str, float]
    ema_weights: dict[str, float]
    last_update_epoch: int = 0
    last_update_global_epoch: int = 0


@dataclass(frozen=True)
class ProbeSubsetConfig:
    data_rows: int
    physics_rows: int
    init_rows: int
    seed: int = 0


@dataclass(frozen=True)
class WeightingConfig:
    scheme: str
    anchor: str
    ema_beta: float
    update_interval_epochs: int
    dynamic_components: tuple[str, ...]
    probe: ProbeSubsetConfig

    @property
    def is_dynamic(self) -> bool:
        return self.scheme != "static"


def _weights_from_mapping(values: dict[str, float]) -> LossWeights:
    return LossWeights(
        data=float(values["data"]),
        dt=float(values["dt"]),
        physics=float(values["physics"]),
        ic=float(values["ic"]),
    )


def weighting_config_from_config(config: Any) -> WeightingConfig:
    scheme = str(cfg_get(config, "pinn.weighting.scheme", "static")).strip().lower()
    if scheme not in SUPPORTED_WEIGHTING_SCHEMES:
        supported = ", ".join(SUPPORTED_WEIGHTING_SCHEMES)
        raise ValueError(f"pinn.weighting.scheme must be one of: {supported}.")

    anchor = str(cfg_get(config, "pinn.weighting.anchor", DEFAULT_WEIGHTING_ANCHOR)).strip().lower()
    if anchor != DEFAULT_WEIGHTING_ANCHOR:
        raise ValueError("pinn.weighting.anchor must be 'physics' in v1.")

    ema_beta = float(cfg_get(config, "pinn.weighting.ema_beta", 0.99))
    if not (0.0 <= ema_beta < 1.0):
        raise ValueError("pinn.weighting.ema_beta must be in [0, 1).")

    update_interval_epochs = int(cfg_get(config, "pinn.weighting.update_interval_epochs", 10))
    if update_interval_epochs <= 0:
        raise ValueError("pinn.weighting.update_interval_epochs must be > 0.")

    raw_dynamic_components = cfg_get(config, "pinn.weighting.dynamic_components", list(DEFAULT_DYNAMIC_COMPONENTS))
    dynamic_components = tuple(str(name).strip().lower() for name in raw_dynamic_components)
    if not dynamic_components:
        raise ValueError("pinn.weighting.dynamic_components must not be empty.")
    if len(set(dynamic_components)) != len(dynamic_components):
        raise ValueError("pinn.weighting.dynamic_components must not contain duplicates.")
    for name in dynamic_components:
        if name not in LOSS_COMPONENTS:
            raise ValueError(f"Unsupported dynamic component '{name}'.")
        if name == anchor:
            raise ValueError("pinn.weighting.dynamic_components must not include the anchor component.")

    probe_seed = int(cfg_get(config, "pinn.weighting.probe.seed", 0))
    probe_data_rows = int(cfg_get(config, "pinn.weighting.probe.data_rows", 256))
    probe_physics_rows = int(cfg_get(config, "pinn.weighting.probe.physics_rows", 256))
    probe_init_rows = int(cfg_get(config, "pinn.weighting.probe.init_rows", 256))
    for value, field_name in (
        (probe_data_rows, "pinn.weighting.probe.data_rows"),
        (probe_physics_rows, "pinn.weighting.probe.physics_rows"),
        (probe_init_rows, "pinn.weighting.probe.init_rows"),
    ):
        if value <= 0:
            raise ValueError(f"{field_name} must be > 0.")

    return WeightingConfig(
        scheme=scheme,
        anchor=anchor,
        ema_beta=ema_beta,
        update_interval_epochs=update_interval_epochs,
        dynamic_components=dynamic_components,
        probe=ProbeSubsetConfig(
            data_rows=probe_data_rows,
            physics_rows=probe_physics_rows,
            init_rows=probe_init_rows,
            seed=probe_seed,
        ),
    )


class LossWeightingPolicy(Protocol):
    config: WeightingConfig

    def initial_state(self, base_weights: LossWeights) -> WeightingState: ...

    def current_weights(self, state: WeightingState) -> LossWeights: ...

    def should_update(self, *, epoch: int, global_epoch: int) -> bool: ...

    def update(self, state: WeightingState, stats: WeightUpdateStats) -> WeightingState: ...


@dataclass(frozen=True)
class BaseWeightingPolicy:
    config: WeightingConfig

    def initial_state(self, base_weights: LossWeights) -> WeightingState:
        base_map = base_weights.as_dict()
        if self.config.is_dynamic:
            # Dynamic weighting treats pinn.loss_weights as base initialization values.
            base_map[self.config.anchor] = 1.0
        return WeightingState(
            active_weights=_weights_from_mapping(base_map),
            raw_candidate_weights={name: float(base_map[name]) for name in self.config.dynamic_components},
            ema_weights={name: float(base_map[name]) for name in self.config.dynamic_components},
        )

    def current_weights(self, state: WeightingState) -> LossWeights:
        return state.active_weights

    def should_update(self, *, epoch: int, global_epoch: int) -> bool:
        if not self.config.is_dynamic:
            return False
        return int(epoch) % int(self.config.update_interval_epochs) == 0

    def update(self, state: WeightingState, stats: WeightUpdateStats) -> WeightingState:
        if not self.config.is_dynamic:
            return state
        raw_candidate_weights = self._raw_candidate_weights(state=state, stats=stats)
        ema_weights = {}
        for name in self.config.dynamic_components:
            prev = float(state.ema_weights[name])
            raw = float(raw_candidate_weights[name])
            ema_weights[name] = float(self.config.ema_beta) * prev + (1.0 - float(self.config.ema_beta)) * raw

        next_map = state.active_weights.as_dict()
        next_map[self.config.anchor] = 1.0
        for name in self.config.dynamic_components:
            next_map[name] = float(ema_weights[name])

        return WeightingState(
            active_weights=_weights_from_mapping(next_map),
            raw_candidate_weights={name: float(raw_candidate_weights[name]) for name in self.config.dynamic_components},
            ema_weights={name: float(ema_weights[name]) for name in self.config.dynamic_components},
            last_update_epoch=int(stats.epoch),
            last_update_global_epoch=int(stats.global_epoch),
        )

    def _raw_candidate_weights(self, state: WeightingState, stats: WeightUpdateStats) -> dict[str, float]:
        raise NotImplementedError


@dataclass(frozen=True)
class StaticWeightingPolicy(BaseWeightingPolicy):
    def _raw_candidate_weights(self, state: WeightingState, stats: WeightUpdateStats) -> dict[str, float]:
        return dict(state.raw_candidate_weights)


@dataclass(frozen=True)
class MAPINNWeightingPolicy(BaseWeightingPolicy):
    def _raw_candidate_weights(self, state: WeightingState, stats: WeightUpdateStats) -> dict[str, float]:
        anchor_value = float(stats.grad_max_abs[self.config.anchor])
        result: dict[str, float] = {}
        eps = 1.0e-12
        for name in self.config.dynamic_components:
            prev_weight = max(float(state.active_weights.get(name)), eps)
            mean_abs = max(float(stats.grad_mean_abs[name]), eps)
            result[name] = anchor_value / (prev_weight * mean_abs)
        return result


@dataclass(frozen=True)
class IDPINNWeightingPolicy(BaseWeightingPolicy):
    def _raw_candidate_weights(self, state: WeightingState, stats: WeightUpdateStats) -> dict[str, float]:
        anchor_value = float(stats.grad_std[self.config.anchor])
        eps = 1.0e-12
        return {
            name: anchor_value / max(float(stats.grad_std[name]), eps)
            for name in self.config.dynamic_components
        }


@dataclass(frozen=True)
class DNPINNWeightingPolicy(BaseWeightingPolicy):
    def _raw_candidate_weights(self, state: WeightingState, stats: WeightUpdateStats) -> dict[str, float]:
        anchor_value = float(stats.grad_l2_norms[self.config.anchor])
        eps = 1.0e-12
        return {
            name: anchor_value / max(float(stats.grad_l2_norms[name]), eps)
            for name in self.config.dynamic_components
        }


def build_weighting_policy(weighting_config: WeightingConfig) -> LossWeightingPolicy:
    if weighting_config.scheme == "static":
        return StaticWeightingPolicy(config=weighting_config)
    if weighting_config.scheme == "ma":
        return MAPINNWeightingPolicy(config=weighting_config)
    if weighting_config.scheme == "id":
        return IDPINNWeightingPolicy(config=weighting_config)
    if weighting_config.scheme == "dn":
        return DNPINNWeightingPolicy(config=weighting_config)
    raise ValueError(f"Unsupported weighting scheme: {weighting_config.scheme}")
