"""Dynamic loss-weighting policies for PINN training."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Protocol

from src.pinn.losses import LOSS_COMPONENTS, LossWeights
from src.train.runtime import cfg_get


SUPPORTED_WEIGHTING_SCHEMES: tuple[str, ...] = ("static", "ma", "id", "dn", "relobralo")
DEFAULT_DYNAMIC_COMPONENTS: tuple[str, ...] = ("data", "dt", "ic")
DEFAULT_WEIGHTING_ANCHOR = "physics"
DEFAULT_RELOBRALO_COMPONENTS: tuple[str, ...] = LOSS_COMPONENTS


@dataclass(frozen=True)
class WeightUpdateStats:
    grad_l2_norms: dict[str, float]
    grad_mean_abs: dict[str, float]
    grad_max_abs: dict[str, float]
    grad_std: dict[str, float]
    component_losses: dict[str, float]
    anchor_component: str | None
    epoch: int
    global_epoch: int


@dataclass(frozen=True)
class WeightingState:
    active_weights: LossWeights
    raw_candidate_weights: dict[str, float]
    ema_weights: dict[str, float]
    previous_losses: dict[str, float] | None = None
    baseline_losses: dict[str, float] | None = None
    last_update_epoch: int = 0
    last_update_global_epoch: int = 0
    update_count: int = 0


@dataclass(frozen=True)
class ProbeSubsetConfig:
    data_rows: int
    physics_rows: int
    init_rows: int
    seed: int = 0


@dataclass(frozen=True)
class WeightingConfig:
    scheme: str
    anchor: str | None
    ema_beta: float
    update_interval_epochs: int
    dynamic_components: tuple[str, ...]
    probe: ProbeSubsetConfig
    relobralo_temperature: float
    relobralo_alpha: float
    relobralo_rho: float
    random_seed: int = 0

    @property
    def is_dynamic(self) -> bool:
        return self.scheme != "static"

    @property
    def uses_anchor(self) -> bool:
        return self.scheme in {"ma", "id", "dn"}

    @property
    def uses_uniform_initialization(self) -> bool:
        return self.scheme == "relobralo"


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

    anchor_raw = str(cfg_get(config, "pinn.weighting.anchor", DEFAULT_WEIGHTING_ANCHOR)).strip().lower()
    anchor = anchor_raw
    if scheme in {"ma", "id", "dn", "static"}:
        if anchor != DEFAULT_WEIGHTING_ANCHOR:
            raise ValueError("pinn.weighting.anchor must be 'physics' in v1.")
    elif scheme == "relobralo":
        # ReLoBRaLo reweights all objectives, so the paper-style physics anchor is
        # not used even if it stays present in config for backward compatibility.
        anchor = None

    ema_beta = float(cfg_get(config, "pinn.weighting.ema_beta", 0.99))
    if not (0.0 <= ema_beta < 1.0):
        raise ValueError("pinn.weighting.ema_beta must be in [0, 1).")

    update_interval_epochs = int(cfg_get(config, "pinn.weighting.update_interval_epochs", 10))
    if update_interval_epochs <= 0:
        raise ValueError("pinn.weighting.update_interval_epochs must be > 0.")

    raw_dynamic_components = cfg_get(config, "pinn.weighting.dynamic_components", list(DEFAULT_DYNAMIC_COMPONENTS))
    dynamic_components = tuple(str(name).strip().lower() for name in raw_dynamic_components)
    if scheme == "relobralo" and set(dynamic_components) == set(DEFAULT_DYNAMIC_COMPONENTS):
        dynamic_components = DEFAULT_RELOBRALO_COMPONENTS
    if not dynamic_components:
        raise ValueError("pinn.weighting.dynamic_components must not be empty.")
    if len(set(dynamic_components)) != len(dynamic_components):
        raise ValueError("pinn.weighting.dynamic_components must not contain duplicates.")
    for name in dynamic_components:
        if name not in LOSS_COMPONENTS:
            raise ValueError(f"Unsupported dynamic component '{name}'.")
        if anchor is not None and name == anchor:
            raise ValueError("pinn.weighting.dynamic_components must not include the anchor component.")
    if scheme == "relobralo" and set(dynamic_components) != set(LOSS_COMPONENTS):
        raise ValueError("pinn.weighting.dynamic_components must include all loss components for relobralo.")

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

    relobralo_temperature = float(cfg_get(config, "pinn.weighting.relobralo.temperature", 1.0))
    if relobralo_temperature <= 0.0:
        raise ValueError("pinn.weighting.relobralo.temperature must be > 0.")
    relobralo_alpha = float(cfg_get(config, "pinn.weighting.relobralo.alpha", 0.999))
    if not (0.0 <= relobralo_alpha <= 1.0):
        raise ValueError("pinn.weighting.relobralo.alpha must be in [0, 1].")
    relobralo_rho = float(cfg_get(config, "pinn.weighting.relobralo.rho", 0.95))
    if not (0.0 <= relobralo_rho <= 1.0):
        raise ValueError("pinn.weighting.relobralo.rho must be in [0, 1].")
    random_seed = int(cfg_get(config, "pinn.weighting.random_seed", 0))

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
        relobralo_temperature=relobralo_temperature,
        relobralo_alpha=relobralo_alpha,
        relobralo_rho=relobralo_rho,
        random_seed=random_seed,
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
        if self.config.uses_uniform_initialization:
            for name in self.config.dynamic_components:
                base_map[name] = 1.0
        elif self.config.is_dynamic and self.config.anchor is not None:
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
        if self.config.scheme == "relobralo":
            return True
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
        if self.config.anchor is not None:
            next_map[self.config.anchor] = 1.0
        for name in self.config.dynamic_components:
            next_map[name] = float(ema_weights[name])

        return WeightingState(
            active_weights=_weights_from_mapping(next_map),
            raw_candidate_weights={name: float(raw_candidate_weights[name]) for name in self.config.dynamic_components},
            ema_weights={name: float(ema_weights[name]) for name in self.config.dynamic_components},
            previous_losses=dict(stats.component_losses),
            baseline_losses=dict(stats.component_losses) if state.baseline_losses is None else dict(state.baseline_losses),
            last_update_epoch=int(stats.epoch),
            last_update_global_epoch=int(stats.global_epoch),
            update_count=int(state.update_count) + 1,
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
        if self.config.anchor is None:
            raise ValueError("DN-PINN requires an anchor component.")
        anchor_value = float(stats.grad_l2_norms[self.config.anchor])
        eps = 1.0e-12
        return {
            name: anchor_value / max(float(stats.grad_l2_norms[name]), eps)
            for name in self.config.dynamic_components
        }


@dataclass(frozen=True)
class ReLoBRaLoWeightingPolicy(BaseWeightingPolicy):
    def update(self, state: WeightingState, stats: WeightUpdateStats) -> WeightingState:
        if not self.config.is_dynamic:
            return state
        raw_candidate_weights = self._raw_candidate_weights(state=state, stats=stats)
        next_map = state.active_weights.as_dict()
        for name in self.config.dynamic_components:
            next_map[name] = float(raw_candidate_weights[name])
        return WeightingState(
            active_weights=_weights_from_mapping(next_map),
            raw_candidate_weights={name: float(raw_candidate_weights[name]) for name in self.config.dynamic_components},
            ema_weights={name: float(raw_candidate_weights[name]) for name in self.config.dynamic_components},
            previous_losses=dict(stats.component_losses),
            baseline_losses=_relobralo_next_baseline_losses(state=state, stats=stats),
            last_update_epoch=int(stats.epoch),
            last_update_global_epoch=int(stats.global_epoch),
            update_count=int(state.update_count) + 1,
        )

    def _raw_candidate_weights(self, state: WeightingState, stats: WeightUpdateStats) -> dict[str, float]:
        if state.previous_losses is None or state.baseline_losses is None:
            return {name: float(state.active_weights.get(name)) for name in self.config.dynamic_components}

        losses = {name: float(stats.component_losses[name]) for name in self.config.dynamic_components}
        previous_losses = {name: float(state.previous_losses[name]) for name in self.config.dynamic_components}
        baseline_losses = {name: float(state.baseline_losses[name]) for name in self.config.dynamic_components}

        temperature = max(float(self.config.relobralo_temperature), 1.0e-12)
        progress_logits = {
            name: losses[name] / max(previous_losses[name] * temperature, 1.0e-12)
            for name in self.config.dynamic_components
        }
        lookback_logits = {
            name: losses[name] / max(baseline_losses[name] * temperature, 1.0e-12)
            for name in self.config.dynamic_components
        }
        lambs_hat = _softmax_weights(progress_logits)
        lambs0_hat = _softmax_weights(lookback_logits)

        alpha = _relobralo_effective_alpha(self.config, state)
        rho_keep_history = 1.0 if _sample_relobralo_rho(self.config, stats.global_epoch) else 0.0
        return {
            name: (
                rho_keep_history * alpha * float(state.active_weights.get(name))
                + (1.0 - rho_keep_history) * alpha * float(lambs0_hat[name])
                + (1.0 - alpha) * float(lambs_hat[name])
            )
            for name in self.config.dynamic_components
        }


def _softmax_weights(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    names = list(values.keys())
    logits = [float(values[name]) for name in names]
    max_logit = max(logits)
    exp_values = [math.exp(logit - max_logit) for logit in logits]
    total = sum(exp_values)
    if total <= 0.0:
        equal = float(len(names))
        return {name: equal / float(len(names)) for name in names}
    scale = float(len(names)) / total
    return {name: scale * exp_val for name, exp_val in zip(names, exp_values)}


def _sample_relobralo_rho(config: WeightingConfig, global_epoch: int) -> bool:
    if float(config.relobralo_rho) <= 0.0:
        return False
    if float(config.relobralo_rho) >= 1.0:
        return True
    rng = random.Random(int(config.random_seed) + int(global_epoch))
    return rng.random() < float(config.relobralo_rho)


def _relobralo_effective_alpha(config: WeightingConfig, state: WeightingState) -> float:
    if int(state.update_count) == 0:
        return 1.0
    if int(state.update_count) == 1:
        return 0.0
    return float(config.relobralo_alpha)


def _relobralo_next_baseline_losses(
    *,
    state: WeightingState,
    stats: WeightUpdateStats,
) -> dict[str, float]:
    current_losses = dict(stats.component_losses)
    if state.baseline_losses is None:
        return current_losses
    if int(state.update_count) == 1:
        # Reference-style warmup: after the second update, anchor the random
        # lookback baseline to the current losses.
        return current_losses
    return dict(state.baseline_losses)


def build_weighting_policy(weighting_config: WeightingConfig) -> LossWeightingPolicy:
    if weighting_config.scheme == "static":
        return StaticWeightingPolicy(config=weighting_config)
    if weighting_config.scheme == "ma":
        return MAPINNWeightingPolicy(config=weighting_config)
    if weighting_config.scheme == "id":
        return IDPINNWeightingPolicy(config=weighting_config)
    if weighting_config.scheme == "dn":
        return DNPINNWeightingPolicy(config=weighting_config)
    if weighting_config.scheme == "relobralo":
        return ReLoBRaLoWeightingPolicy(config=weighting_config)
    raise ValueError(f"Unsupported weighting scheme: {weighting_config.scheme}")
