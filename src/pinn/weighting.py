"""Dynamic loss-weighting policies for PINN training."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Protocol

from src.pinn.losses import LOSS_COMPONENTS, LossWeights
from src.training.runtime import cfg_get


SUPPORTED_WEIGHTING_SCHEMES: tuple[str, ...] = (
    "static",
    "ma",
    "paper_lr_annealing",
    "id",
    "dn",
    "relobralo",
    "ntk_random_batch",
)
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
    ntk_mean_trace: dict[str, float] | None = None
    ntk_batch_sizes: dict[str, int] | None = None


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
class NTKBatchConfig:
    data: int
    dt: int
    physics: int
    ic: int

    def as_dict(self) -> dict[str, int]:
        return {
            "data": int(self.data),
            "dt": int(self.dt),
            "physics": int(self.physics),
            "ic": int(self.ic),
        }


@dataclass(frozen=True)
class WeightingConfig:
    scheme: str
    anchor: str | None
    ema_beta: float
    update_interval_epochs: int
    update_mode: str
    use_live_batch: bool
    dynamic_components: tuple[str, ...]
    probe: ProbeSubsetConfig
    ntk_batch_sizes: NTKBatchConfig
    ntk_eps: float
    ntk_seed: int
    ntk_refresh_each_update: bool
    ntk_use_ema: bool
    relobralo_temperature: float
    relobralo_alpha: float
    relobralo_rho: float
    gradient_eps: float
    candidate_weight_min: float
    candidate_weight_max: float
    random_seed: int = 0

    @property
    def is_dynamic(self) -> bool:
        return self.scheme != "static"

    @property
    def uses_anchor(self) -> bool:
        return self.scheme in {"ma", "paper_lr_annealing", "id", "dn"}

    @property
    def uses_uniform_initialization(self) -> bool:
        return self.scheme in {"relobralo", "ntk_random_batch"}

    @property
    def uses_step_updates(self) -> bool:
        return self.is_dynamic and self.update_mode == "step"


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
    if scheme in {"ma", "paper_lr_annealing", "id", "dn", "static"}:
        if anchor != DEFAULT_WEIGHTING_ANCHOR:
            raise ValueError("pinn.weighting.anchor must be 'physics' in v1.")
    elif scheme in {"relobralo", "ntk_random_batch"}:
        # ReLoBRaLo reweights all objectives, so the paper-style physics anchor is
        # not used even if it stays present in config for backward compatibility.
        anchor = None

    ema_beta_default = 0.9 if scheme == "paper_lr_annealing" else 0.99
    ema_beta_raw = cfg_get(config, "pinn.weighting.ema_beta", None)
    ema_beta = ema_beta_default if ema_beta_raw in (None, "null") else float(ema_beta_raw)
    if not (0.0 <= ema_beta < 1.0):
        raise ValueError("pinn.weighting.ema_beta must be in [0, 1).")

    update_interval_epochs = int(cfg_get(config, "pinn.weighting.update_interval_epochs", 10))
    if update_interval_epochs <= 0:
        raise ValueError("pinn.weighting.update_interval_epochs must be > 0.")
    update_mode_default = "step" if scheme == "paper_lr_annealing" else "epoch"
    update_mode_raw = cfg_get(config, "pinn.weighting.update_mode", None)
    if update_mode_raw in (None, "null"):
        update_mode = update_mode_default
    else:
        update_mode = str(update_mode_raw).strip().lower()
    if update_mode not in {"epoch", "step"}:
        raise ValueError("pinn.weighting.update_mode must be one of: epoch, step.")
    use_live_batch_default = scheme == "paper_lr_annealing"
    use_live_batch_raw = cfg_get(config, "pinn.weighting.use_live_batch", None)
    use_live_batch = use_live_batch_default if use_live_batch_raw in (None, "null") else bool(use_live_batch_raw)
    if update_mode == "step" and not use_live_batch:
        raise ValueError("pinn.weighting.use_live_batch must be true when pinn.weighting.update_mode=step.")

    raw_dynamic_components = cfg_get(config, "pinn.weighting.dynamic_components", list(DEFAULT_DYNAMIC_COMPONENTS))
    dynamic_components = tuple(str(name).strip().lower() for name in raw_dynamic_components)
    if scheme in {"relobralo", "ntk_random_batch"} and set(dynamic_components) == set(DEFAULT_DYNAMIC_COMPONENTS):
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
    if scheme in {"relobralo", "ntk_random_batch"} and set(dynamic_components) != set(LOSS_COMPONENTS):
        raise ValueError("pinn.weighting.dynamic_components must include all loss components for relobralo and ntk_random_batch.")

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

    ntk_batch_sizes = NTKBatchConfig(
        data=int(cfg_get(config, "pinn.weighting.ntk.batch_size.data", probe_data_rows)),
        dt=int(cfg_get(config, "pinn.weighting.ntk.batch_size.dt", probe_data_rows)),
        physics=int(cfg_get(config, "pinn.weighting.ntk.batch_size.physics", probe_physics_rows)),
        ic=int(cfg_get(config, "pinn.weighting.ntk.batch_size.ic", probe_init_rows)),
    )
    for name, value in ntk_batch_sizes.as_dict().items():
        if value <= 0:
            raise ValueError(f"pinn.weighting.ntk.batch_size.{name} must be > 0.")
    ntk_eps = float(cfg_get(config, "pinn.weighting.ntk.eps", 1.0e-12))
    if ntk_eps <= 0.0:
        raise ValueError("pinn.weighting.ntk.eps must be > 0.")
    ntk_seed = int(cfg_get(config, "pinn.weighting.ntk.seed", probe_seed))
    ntk_refresh_each_update = bool(cfg_get(config, "pinn.weighting.probe.refresh_each_update", False))
    ntk_use_ema = bool(cfg_get(config, "pinn.weighting.ntk.use_ema", False))

    relobralo_temperature = float(cfg_get(config, "pinn.weighting.relobralo.temperature", 1.0))
    if relobralo_temperature <= 0.0:
        raise ValueError("pinn.weighting.relobralo.temperature must be > 0.")
    relobralo_alpha = float(cfg_get(config, "pinn.weighting.relobralo.alpha", 0.999))
    if not (0.0 <= relobralo_alpha <= 1.0):
        raise ValueError("pinn.weighting.relobralo.alpha must be in [0, 1].")
    relobralo_rho = float(cfg_get(config, "pinn.weighting.relobralo.rho", 0.95))
    if not (0.0 <= relobralo_rho <= 1.0):
        raise ValueError("pinn.weighting.relobralo.rho must be in [0, 1].")
    gradient_eps = float(cfg_get(config, "pinn.weighting.gradient_eps", 1.0e-12))
    if gradient_eps <= 0.0:
        raise ValueError("pinn.weighting.gradient_eps must be > 0.")
    candidate_weight_min = float(cfg_get(config, "pinn.weighting.candidate_weight_min", 1.0e-8))
    candidate_weight_max = float(cfg_get(config, "pinn.weighting.candidate_weight_max", 1.0e8))
    if candidate_weight_min <= 0.0:
        raise ValueError("pinn.weighting.candidate_weight_min must be > 0.")
    if candidate_weight_max < candidate_weight_min:
        raise ValueError("pinn.weighting.candidate_weight_max must be >= pinn.weighting.candidate_weight_min.")
    random_seed = int(cfg_get(config, "pinn.weighting.random_seed", 0))

    return WeightingConfig(
        scheme=scheme,
        anchor=anchor,
        ema_beta=ema_beta,
        update_interval_epochs=update_interval_epochs,
        update_mode=update_mode,
        use_live_batch=use_live_batch,
        dynamic_components=dynamic_components,
        probe=ProbeSubsetConfig(
            data_rows=probe_data_rows,
            physics_rows=probe_physics_rows,
            init_rows=probe_init_rows,
            seed=probe_seed,
        ),
        ntk_batch_sizes=ntk_batch_sizes,
        ntk_eps=ntk_eps,
        ntk_seed=ntk_seed,
        ntk_refresh_each_update=ntk_refresh_each_update,
        ntk_use_ema=ntk_use_ema,
        relobralo_temperature=relobralo_temperature,
        relobralo_alpha=relobralo_alpha,
        relobralo_rho=relobralo_rho,
        gradient_eps=gradient_eps,
        candidate_weight_min=candidate_weight_min,
        candidate_weight_max=candidate_weight_max,
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
        if self.config.uses_step_updates:
            return False
        if self.config.scheme == "relobralo":
            return True
        return int(epoch) % int(self.config.update_interval_epochs) == 0

    def update(self, state: WeightingState, stats: WeightUpdateStats) -> WeightingState:
        if not self.config.is_dynamic:
            return state
        raw_candidate_weights = self._raw_candidate_weights(state=state, stats=stats)
        if self.config.scheme == "ntk_random_batch" and not bool(self.config.ntk_use_ema):
            ema_weights = {name: float(raw_candidate_weights[name]) for name in self.config.dynamic_components}
        else:
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
class PaperLearningRateAnnealingPolicy(BaseWeightingPolicy):
    def _raw_candidate_weights(self, state: WeightingState, stats: WeightUpdateStats) -> dict[str, float]:
        anchor_value = float(stats.grad_max_abs[self.config.anchor])
        eps = max(float(self.config.gradient_eps), 1.0e-18)
        result: dict[str, float] = {}
        for name in self.config.dynamic_components:
            mean_abs = max(float(stats.grad_mean_abs[name]), eps)
            candidate = anchor_value / mean_abs
            candidate = min(max(candidate, float(self.config.candidate_weight_min)), float(self.config.candidate_weight_max))
            result[name] = candidate
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


@dataclass(frozen=True)
class NTKRandomBatchWeightingPolicy(BaseWeightingPolicy):
    def _raw_candidate_weights(self, state: WeightingState, stats: WeightUpdateStats) -> dict[str, float]:
        if stats.ntk_mean_trace is None:
            raise ValueError("NTK random-batch weighting requires ntk_mean_trace statistics.")
        eps = max(float(self.config.ntk_eps), 1.0e-18)
        active_traces = {
            name: max(float(stats.ntk_mean_trace[name]), eps)
            for name in self.config.dynamic_components
        }
        trace_sum = sum(active_traces.values())
        return {
            name: trace_sum / value
            for name, value in active_traces.items()
        }


def build_weighting_policy(weighting_config: WeightingConfig) -> LossWeightingPolicy:
    if weighting_config.scheme == "static":
        return StaticWeightingPolicy(config=weighting_config)
    if weighting_config.scheme == "ma":
        return MAPINNWeightingPolicy(config=weighting_config)
    if weighting_config.scheme == "paper_lr_annealing":
        return PaperLearningRateAnnealingPolicy(config=weighting_config)
    if weighting_config.scheme == "id":
        return IDPINNWeightingPolicy(config=weighting_config)
    if weighting_config.scheme == "dn":
        return DNPINNWeightingPolicy(config=weighting_config)
    if weighting_config.scheme == "relobralo":
        return ReLoBRaLoWeightingPolicy(config=weighting_config)
    if weighting_config.scheme == "ntk_random_batch":
        return NTKRandomBatchWeightingPolicy(config=weighting_config)
    raise ValueError(f"Unsupported weighting scheme: {weighting_config.scheme}")
