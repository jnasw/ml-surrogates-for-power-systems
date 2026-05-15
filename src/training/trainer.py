"""Surrogate training and evaluation utilities."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import time
from typing import Any

import torch
from torch import nn

from src.pinn.collocation import (
    CollocationStrategyContext,
    build_collocation_manager,
)
from src.pinn.data import PinnDatasetBundle
from src.pinn.evaluator import move_pinn_training_data_to_device
from src.pinn.logging import EpochMetrics, PinnLogger
from src.pinn.losses import LossWeights
from src.pinn.multistage import (
    MultistagePinnEnsemble,
    MultistageStageMLP,
    StageDiagnostics,
    estimate_stage_diagnostics,
)
from src.pinn.optim import OptimizerSpec, build_optimizer
from src.pinn.runtime import (
    OptimizerPhase,
    ResolvedPinnStack,
    resolve_pinn_stack,
    resolve_torch_dtype,
    torch_dtype_name,
)
from src.pinn.vrba import (
    initialize_vrba_state,
)
from src.pinn.weighting import WeightingConfig, build_weighting_policy
from src.training.device import select_torch_device
from src.training.model import TimeConditionedMLP
from src.training.baseline import (
    BaselineConfig,
    SurrogateModel,
    evaluate,
    train_baseline_surrogate,
    train_surrogate,
)
from src.training.collocation_runtime import (
    active_collocation_weights_or_none as _active_collocation_weights_or_none,
    active_dt_weights_or_none as _active_dt_weights_or_none,
    active_init_weights_or_none as _active_init_weights_or_none,
    collocation_phase_boundary_enabled as _collocation_phase_boundary_enabled,
    initialize_dt_vrba_weights as _initialize_dt_vrba_weights,
    phase_boundary_refresh_planned as _phase_boundary_refresh_planned,
    prepare_epoch_constraint_batch as _prepare_epoch_constraint_batch,
)
from src.training.diagnostics import (
    build_checkpoint_payload as _build_checkpoint_payload,
    build_epoch_metrics_row as _build_epoch_metrics_row,
    build_single_stage_optimizer_diagnostics as _build_single_stage_optimizer_diagnostics,
    checkpointing_enabled as _checkpointing_enabled,
    collocation_pool_diagnostics as _collocation_pool_diagnostics,
    collocation_vrba_details as _collocation_vrba_details,
    collocation_vrba_summary as _collocation_vrba_summary,
    mean_epoch_breakdown as _mean_epoch_breakdown,
    record_epoch_row_and_checkpoints as _record_epoch_row_and_checkpoints,
    resolve_checkpoint_milestones as _resolve_checkpoint_milestones,
    scheduler_diagnostics as _scheduler_diagnostics,
    scheduler_metric_value as _scheduler_metric_value,
)
from src.training.pinn_runtime import (
    GradientTelemetry,
    PinnBatch,
    apply_single_stage_live_weight_update as _apply_single_stage_live_weight_update,
    build_phase_scheduler as _build_phase_scheduler,
    evaluate_epoch_validation_and_test as _evaluate_epoch_validation_and_test,
    iter_minibatch_batches as _iter_minibatch_batches,
    maybe_apply_single_stage_epoch_weight_update as _maybe_apply_single_stage_epoch_weight_update,
    measure_pinn_state as _measure_pinn_state,
    phase_allows_sampling as _phase_allows_sampling,
    phase_effective_full_batch as _phase_effective_full_batch,
    phase_supports_dynamic_weight_updates as _phase_supports_dynamic_weight_updates,
    sample_rows_with_indices as _sample_rows_with_indices,
    scheduled_loss_weights as _scheduled_loss_weights,
    train_pinn_step as _train_pinn_step,
)
from src.training.runtime import cfg_get, configure_reproducibility, configure_reproducibility_from_config
from src.training.sampling import (
    curriculum_active_max_bin as _curriculum_active_max_bin,
    sample_curriculum_indices_balanced as _sample_curriculum_indices_balanced,
    sample_supervised_rows_with_indices as _sample_supervised_rows_with_indices,
)
from src.training.supervised_acquisition import SupervisedAcquisitionManager as _SupervisedAcquisitionManager


@dataclass
class PinnModel:
    model: nn.Module
    device: torch.device
    dtype: torch.dtype
    input_dim: int
    output_dim: int
    hidden_dim: int
    hidden_layers: int
    activation: str
    architecture: str = "single_stage"
    stage_first_activations: list[str] | None = None
    stage_hidden_activations: list[str] | None = None
    stage_time_scales: list[float] | None = None
    stage_epsilons: list[float] | None = None

    def save_checkpoint(self, path: str) -> None:
        payload = {
            "state_dict": self.model.state_dict(),
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_dim": self.hidden_dim,
            "hidden_layers": self.hidden_layers,
            "activation": self.activation,
            "dtype": torch_dtype_name(self.dtype),
            "architecture": self.architecture,
            "stage_first_activations": [] if self.stage_first_activations is None else list(self.stage_first_activations),
            "stage_hidden_activations": [] if self.stage_hidden_activations is None else list(self.stage_hidden_activations),
            "stage_time_scales": [] if self.stage_time_scales is None else [float(x) for x in self.stage_time_scales],
            "stage_epsilons": [] if self.stage_epsilons is None else [float(x) for x in self.stage_epsilons],
        }
        torch.save(payload, path)

    @staticmethod
    def load_checkpoint(path: str, device_preference: str = "auto") -> "PinnModel":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        dtype = resolve_torch_dtype(str(payload["dtype"]))
        device = select_torch_device(device_preference)
        if device.type == "mps" and dtype == torch.float64:
            device = torch.device("cpu")
        architecture = str(payload.get("architecture", "single_stage"))
        if architecture == "multistage":
            stage_first_activations = [str(x) for x in payload.get("stage_first_activations", [])]
            stage_hidden_activations = [str(x) for x in payload.get("stage_hidden_activations", [])]
            stage_time_scales = [float(x) for x in payload.get("stage_time_scales", [])]
            stage_epsilons = [float(x) for x in payload.get("stage_epsilons", [])]
            stage_count = len(stage_first_activations)
            stages = []
            for idx in range(stage_count):
                stages.append(
                    MultistageStageMLP(
                        input_dim=int(payload["input_dim"]),
                        output_dim=int(payload["output_dim"]),
                        hidden_dim=int(payload["hidden_dim"]),
                        hidden_layers=int(payload["hidden_layers"]),
                        first_activation=stage_first_activations[idx],
                        hidden_activation=stage_hidden_activations[idx],
                        time_scale=stage_time_scales[idx],
                    )
                )
            model = MultistagePinnEnsemble(stages=stages, epsilons=stage_epsilons).to(device=device, dtype=dtype)
        else:
            model = TimeConditionedMLP(
                input_dim=int(payload["input_dim"]),
                output_dim=int(payload["output_dim"]),
                hidden_dim=int(payload["hidden_dim"]),
                hidden_layers=int(payload["hidden_layers"]),
                activation=str(payload["activation"]),
            ).to(device=device, dtype=dtype)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return PinnModel(
            model=model,
            device=device,
            dtype=dtype,
            input_dim=int(payload["input_dim"]),
            output_dim=int(payload["output_dim"]),
            hidden_dim=int(payload["hidden_dim"]),
            hidden_layers=int(payload["hidden_layers"]),
            activation=str(payload["activation"]),
            architecture=architecture,
            stage_first_activations=None if architecture != "multistage" else [str(x) for x in payload.get("stage_first_activations", [])],
            stage_hidden_activations=None if architecture != "multistage" else [str(x) for x in payload.get("stage_hidden_activations", [])],
            stage_time_scales=None if architecture != "multistage" else [float(x) for x in payload.get("stage_time_scales", [])],
            stage_epsilons=None if architecture != "multistage" else [float(x) for x in payload.get("stage_epsilons", [])],
        )


@dataclass(frozen=True)
class OptimizerTransitionSettings:
    enabled: bool
    preserve_state_for: tuple[str, ...]
    family_mode: str
    adam_damping_after_refresh: float
    adam_damping_after_switch: float


@dataclass
class OptimizerTransitionState:
    cache: dict[str, dict[str, Any]]
    previous_optimizer_name: str | None = None


def _cost_tracking_enabled(config: Any) -> bool:
    return bool(cfg_get(config, "pinn.cost_tracking.enabled", True))


def _reset_peak_memory_if_cuda(device: torch.device, *, enabled: bool) -> None:
    if not enabled or device.type != "cuda":
        return
    torch.cuda.reset_peak_memory_stats(device)


def _peak_memory_if_cuda(device: torch.device, *, enabled: bool) -> tuple[int | None, int | None]:
    if not enabled or device.type != "cuda":
        return None, None
    torch.cuda.synchronize(device)
    return int(torch.cuda.max_memory_allocated(device)), int(torch.cuda.max_memory_reserved(device))


def _sample_tensor_rows(
    *,
    x: torch.Tensor,
    rows_per_epoch_cfg: Any,
    fraction_per_epoch_cfg: Any,
    cfg_prefix: str,
) -> torch.Tensor:
    total_rows = int(x.shape[0])

    target_rows: int | None = None
    if rows_per_epoch_cfg not in (None, "null"):
        target_rows = int(rows_per_epoch_cfg)
    elif fraction_per_epoch_cfg not in (None, "null"):
        fraction = float(fraction_per_epoch_cfg)
        if not (0.0 < fraction <= 1.0):
            raise ValueError(f"{cfg_prefix}.fraction_per_epoch must be in (0, 1].")
        target_rows = max(1, int(round(total_rows * fraction)))

    if target_rows is None or target_rows >= total_rows:
        return x
    if target_rows <= 0:
        raise ValueError(f"{cfg_prefix}.rows_per_epoch must be > 0.")

    indices_np = np.random.choice(total_rows, size=target_rows, replace=False)
    indices = torch.as_tensor(indices_np, device=x.device, dtype=torch.long)
    return x.index_select(0, indices)


def _take_fixed_tensor_rows(
    x: torch.Tensor,
    *,
    target_rows: int,
    seed: int,
    y: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
    total_rows = int(x.shape[0])
    if target_rows >= total_rows:
        return x if y is None else (x, y)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    indices_cpu = torch.randperm(total_rows, generator=generator, device="cpu")[:target_rows]
    indices = indices_cpu.to(device=x.device)
    return _sample_rows_with_indices(x, indices, y)


def _build_weight_update_probe_batch(
    *,
    dataset: PinnDatasetBundle,
    weighting_config: WeightingConfig,
    seed_offset: int = 0,
) -> PinnBatch:
    data_total_rows = int(dataset.train_x.shape[0])
    data_target_rows = int(weighting_config.probe.data_rows)
    if data_target_rows >= data_total_rows:
        data_indices = torch.arange(data_total_rows, device=dataset.train_x.device, dtype=torch.long)
    else:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(weighting_config.probe.seed) + int(seed_offset))
        indices_cpu = torch.randperm(data_total_rows, generator=generator, device="cpu")[:data_target_rows]
        data_indices = indices_cpu.to(device=dataset.train_x.device)
    x_data, y_data = _sample_rows_with_indices(dataset.train_x, data_indices, dataset.train_y)
    x_data_dt_weights = None if dataset.train_dt_weights is None else _sample_rows_with_indices(dataset.train_dt_weights, data_indices)
    x_col = _take_fixed_tensor_rows(
        dataset.train_col_x,
        target_rows=weighting_config.probe.physics_rows,
        seed=weighting_config.probe.seed + seed_offset + 1,
    )
    x_init, y_init = _take_fixed_tensor_rows(
        dataset.train_init_x,
        target_rows=weighting_config.probe.init_rows,
        seed=weighting_config.probe.seed + seed_offset + 2,
        y=dataset.train_init_y,
    )
    return PinnBatch(
        x_data=x_data,
        y_data=y_data,
        x_data_dt_weights=x_data_dt_weights,
        x_col=x_col,
        x_col_weights=None,
        x_init=x_init,
        y_init=y_init,
        x_init_weights=None,
    )
def _optimizer_transition_settings(config: Any) -> OptimizerTransitionSettings:
    preserve_raw = cfg_get(config, "pinn.optimizer_transition.preserve_state_for", ["Adam"])
    preserve = tuple(str(name).strip().lower() for name in preserve_raw)
    family_mode = str(cfg_get(config, "pinn.optimizer_transition.family_mode", "optimizer")).strip().lower()
    if family_mode not in {"optimizer", "phase_name"}:
        raise ValueError("pinn.optimizer_transition.family_mode must be one of: optimizer, phase_name.")
    return OptimizerTransitionSettings(
        enabled=bool(cfg_get(config, "pinn.optimizer_transition.enabled", True)),
        preserve_state_for=preserve,
        family_mode=family_mode,
        adam_damping_after_refresh=float(cfg_get(config, "pinn.optimizer_transition.adam.damping_after_refresh", 0.3)),
        adam_damping_after_switch=float(cfg_get(config, "pinn.optimizer_transition.adam.damping_after_switch", 0.6)),
    )


def _optimizer_transition_family(phase: OptimizerPhase, settings: OptimizerTransitionSettings) -> str:
    optimizer_name = str(phase.optimizer).strip().lower()
    if settings.family_mode == "phase_name":
        return str(phase.name).strip().lower()
    return optimizer_name


def _scale_optimizer_state_tensors(state_dict: dict[str, Any], factor: float) -> dict[str, Any]:
    scaled = {
        "state": {},
        "param_groups": [dict(group) for group in state_dict.get("param_groups", [])],
    }
    for param_id, param_state in state_dict.get("state", {}).items():
        new_state: dict[str, Any] = {}
        for key, value in param_state.items():
            if key in {"exp_avg", "exp_avg_sq", "max_exp_avg_sq"} and torch.is_tensor(value):
                new_state[key] = value.clone().mul_(float(factor))
            elif torch.is_tensor(value):
                new_state[key] = value.clone()
            else:
                new_state[key] = value
        scaled["state"][param_id] = new_state
    return scaled


def _copy_optimizer_state_to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _copy_optimizer_state_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_optimizer_state_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_optimizer_state_to_cpu(item) for item in value)
    return copy.deepcopy(value)


def _maybe_restore_optimizer_state(
    *,
    optimizer_spec: OptimizerSpec,
    phase: OptimizerPhase,
    config: Any,
    transition_settings: OptimizerTransitionSettings,
    transition_state: OptimizerTransitionState,
    refresh_on_phase_start: bool,
) -> dict[str, Any]:
    optimizer_name = str(phase.optimizer).strip().lower()
    diagnostics = {
        "state_transition_enabled": bool(transition_settings.enabled),
        "state_restored": False,
        "state_damping_factor": 1.0,
        "state_cache_family": None,
    }
    if not transition_settings.enabled or optimizer_name not in transition_settings.preserve_state_for:
        return diagnostics

    family = _optimizer_transition_family(phase, transition_settings)
    diagnostics["state_cache_family"] = family
    cached = transition_state.cache.get(family)
    if cached is None:
        return diagnostics

    optimizer_spec.optimizer.load_state_dict(cached["state_dict"])
    damping = 1.0
    previous_optimizer = transition_state.previous_optimizer_name
    if optimizer_name == "adam":
        if refresh_on_phase_start:
            damping *= float(transition_settings.adam_damping_after_refresh)
        if previous_optimizer is not None and previous_optimizer != optimizer_name:
            damping *= float(transition_settings.adam_damping_after_switch)
    if damping < 1.0:
        optimizer_spec.optimizer.load_state_dict(
            _scale_optimizer_state_tensors(optimizer_spec.optimizer.state_dict(), factor=damping)
        )
    diagnostics["state_restored"] = True
    diagnostics["state_damping_factor"] = float(damping)
    return diagnostics


def _cache_optimizer_state(
    *,
    optimizer_spec: OptimizerSpec,
    phase: OptimizerPhase,
    transition_settings: OptimizerTransitionSettings,
    transition_state: OptimizerTransitionState,
) -> None:
    optimizer_name = str(phase.optimizer).strip().lower()
    if not transition_settings.enabled or optimizer_name not in transition_settings.preserve_state_for:
        transition_state.previous_optimizer_name = optimizer_name
        return
    family = _optimizer_transition_family(phase, transition_settings)
    transition_state.cache[family] = {
        "state_dict": _copy_optimizer_state_to_cpu(optimizer_spec.optimizer.state_dict()),
        "optimizer_name": optimizer_name,
    }
    transition_state.previous_optimizer_name = optimizer_name


def _multistage_max_stages(config: Any) -> int:
    return int(cfg_get(config, "pinn.multistage.max_stages", 2))


def _multistage_stop_threshold(config: Any) -> float:
    return float(cfg_get(config, "pinn.multistage.stop.residual_rms_threshold", 1.0e-6))


def _multistage_stage_epsilon(config: Any, stage_idx: int) -> float:
    base = float(cfg_get(config, "pinn.multistage.residual_stage.epsilon", 1.0e-2))
    return 1.0 if stage_idx == 0 else base


def _multistage_stage_output_init_scale(config: Any, stage_idx: int) -> float:
    if stage_idx == 0:
        return 1.0
    return float(cfg_get(config, "pinn.multistage.residual_stage.output_init_scale", 1.0e-3))


def _multistage_loss_ref_enabled(config: Any) -> bool:
    return bool(cfg_get(config, "pinn.multistage.loss_ref.enabled", True))


def _multistage_loss_ref_min_value(config: Any) -> float:
    return float(cfg_get(config, "pinn.multistage.loss_ref.min_value", 1.0e-12))


def _multistage_stage_epsilon_warmup_epochs(config: Any) -> int:
    return int(cfg_get(config, "pinn.multistage.residual_stage.epsilon_warmup_epochs", 50))


def _multistage_stage_epsilon_start_fraction(config: Any) -> float:
    return float(cfg_get(config, "pinn.multistage.residual_stage.epsilon_start_fraction", 0.0))


def _multistage_analysis_collocation(dataset: PinnDatasetBundle, config: Any) -> torch.Tensor:
    target_rows = cfg_get(config, "pinn.multistage.analysis.collocation_rows", None)
    if target_rows in (None, "null"):
        return dataset.train_col_x
    return _take_fixed_tensor_rows(
        dataset.train_col_x,
        target_rows=int(target_rows),
        seed=int(cfg_get(config, "pinn.multistage.analysis.seed", 0)),
    )


def _build_multistage_stage_model_from_config(
    *,
    config: Any,
    stage_idx: int,
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    hidden_layers: int,
    base_activation: str,
    time_scale: float,
) -> MultistageStageMLP:
    if stage_idx == 0:
        first_activation = str(cfg_get(config, "pinn.multistage.base_stage.first_activation", base_activation))
        hidden_activation = str(cfg_get(config, "pinn.multistage.base_stage.hidden_activation", base_activation))
        return MultistageStageMLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            first_activation=first_activation,
            hidden_activation=hidden_activation,
            time_scale=1.0,
            output_init_scale=1.0,
        )
    first_activation = str(cfg_get(config, "pinn.multistage.residual_stage.first_activation", "sin"))
    hidden_activation = str(cfg_get(config, "pinn.multistage.residual_stage.hidden_activation", "tanh"))
    return MultistageStageMLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
        first_activation=first_activation,
        hidden_activation=hidden_activation,
        time_scale=time_scale,
        output_init_scale=_multistage_stage_output_init_scale(config, stage_idx),
    )


def _assemble_multistage_pinn_model(
    *,
    ensemble: MultistagePinnEnsemble,
    device: torch.device,
    dtype: torch.dtype,
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    hidden_layers: int,
    activation: str,
) -> PinnModel:
    stage_first_activations = [stage.first_activation_name for stage in ensemble.stages]
    stage_hidden_activations = [stage.hidden_activation_name for stage in ensemble.stages]
    stage_time_scales = [float(stage.time_scale) for stage in ensemble.stages]
    return PinnModel(
        model=ensemble,
        device=device,
        dtype=dtype,
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
        activation=activation,
        architecture="multistage",
        stage_first_activations=stage_first_activations,
        stage_hidden_activations=stage_hidden_activations,
        stage_time_scales=stage_time_scales,
        stage_epsilons=ensemble.epsilon_values(),
    )


def _multistage_effective_epsilon(
    *,
    target_epsilon: float,
    stage_idx: int,
    stage_epoch: int,
    stage_total_epochs: int,
    config: Any,
) -> float:
    if stage_idx == 0:
        return 1.0
    warmup_epochs = min(_multistage_stage_epsilon_warmup_epochs(config), max(0, int(stage_total_epochs)))
    if warmup_epochs <= 0:
        return float(target_epsilon)
    start_fraction = max(0.0, min(1.0, _multistage_stage_epsilon_start_fraction(config)))
    progress = min(1.0, max(0.0, float(stage_epoch) / float(warmup_epochs)))
    fraction = start_fraction + (1.0 - start_fraction) * progress
    return float(target_epsilon) * fraction


def _train_multistage_pinn(
    *,
    dataset: PinnDatasetBundle,
    ode_model: Any,
    config: Any,
    resolved_stack: ResolvedPinnStack,
    logger: PinnLogger | None = None,
) -> tuple[PinnModel, list[EpochMetrics]]:
    from src.training.pinn_multistage import train_multistage_pinn_loop

    return train_multistage_pinn_loop(
        dataset=dataset,
        ode_model=ode_model,
        config=config,
        resolved_stack=resolved_stack,
        logger=logger,
    )


def train_pinn(
    *,
    dataset: PinnDatasetBundle,
    ode_model: Any,
    config: Any,
    logger: PinnLogger | None = None,
) -> tuple[PinnModel, list[EpochMetrics]]:
    resolved_stack = resolve_pinn_stack(config)
    if bool(resolved_stack.supervised_acquisition.enabled) and resolved_stack.mode == "multistage_pinn":
        raise ValueError("pinn.supervised_acquisition.enabled=true currently supports single-stage PINN mode only.")
    if resolved_stack.mode == "multistage_pinn":
        return _train_multistage_pinn(
            dataset=dataset,
            ode_model=ode_model,
            config=config,
            resolved_stack=resolved_stack,
            logger=logger,
        )
    from src.training.pinn_single_stage import train_single_stage_pinn_loop

    return train_single_stage_pinn_loop(
        dataset=dataset,
        ode_model=ode_model,
        config=config,
        resolved_stack=resolved_stack,
        logger=logger,
    )
