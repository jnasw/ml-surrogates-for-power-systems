"""Resolved PINN stack and config-resolution helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.pinn.line_search import normalize_line_search_config
from src.pinn.losses import LossWeights
from src.pinn.vrba import VrbAConfig, vrba_config_from_config
from src.pinn.weighting import WeightingConfig, weighting_config_from_config
from src.training.runtime import cfg_get


@dataclass(frozen=True)
class SchedulerConfig:
    name: str
    mode: str
    factor: float
    patience: int
    threshold: float
    threshold_mode: str
    cooldown: int
    min_lr: float
    eps: float
    metric: str


@dataclass(frozen=True)
class OptimizerPhase:
    name: str
    optimizer: str
    lr: float
    epochs: int
    batch_size: int | None
    shuffle: bool
    full_batch: bool | None = None
    allow_sampling: bool | None = None
    optimizer_kwargs: dict[str, Any] = field(default_factory=dict)
    scheduler: SchedulerConfig | None = None
    line_search: dict[str, Any] | None = None
    convergence: dict[str, Any] | None = None


@dataclass(frozen=True)
class ResolvedCollocationSettings:
    mode: str
    strategy: str
    refresh_mode: str
    refresh_on_phase_start: tuple[str, ...]
    refresh_on_phase_end: tuple[str, ...]


@dataclass(frozen=True)
class ResolvedCurriculumSettings:
    enabled: bool
    balance_active_bins: bool
    unlock_epochs: tuple[int, ...] | None
    stage_length_epochs: int


@dataclass(frozen=True)
class ResolvedSupervisedAcquisitionSettings:
    enabled: bool
    strategy: str
    initial_trajectories: int | None
    add_trajectories: int
    max_trajectories: int | None
    refresh_period_epochs: int
    candidate_batch_size: int
    seed: int


@dataclass(frozen=True)
class ResolvedMultistageOptimizerSchedule:
    explicit_stage_optimizer_phases: list[list[OptimizerPhase]] | None
    base_optimizer_phases: list[OptimizerPhase]
    residual_optimizer_phases: list[OptimizerPhase]


@dataclass(frozen=True)
class ResolvedLossWeightScheduleStage:
    epochs: int | None
    weights: LossWeights


@dataclass(frozen=True)
class ResolvedPinnStack:
    mode: str
    optimizer_phases: list[OptimizerPhase] | None
    multistage_optimizer_schedule: ResolvedMultistageOptimizerSchedule | None
    base_loss_weights: LossWeights
    loss_weight_schedule: tuple[ResolvedLossWeightScheduleStage, ...] | None
    weighting_config: WeightingConfig
    vrba_config: VrbAConfig
    collocation: ResolvedCollocationSettings
    curriculum: ResolvedCurriculumSettings
    supervised_acquisition: ResolvedSupervisedAcquisitionSettings


def _normalize_optional_mapping(value: Any, field_name: str) -> dict[str, Any] | None:
    if value in (None, "null"):
        return None
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "items"):
        return {str(key): nested for key, nested in value.items()}
    raise ValueError(f"{field_name} must be a mapping or null.")


def _validate_optimizer_phase(phase: OptimizerPhase) -> None:
    if phase.epochs <= 0:
        raise ValueError("Each optimizer phase must have epochs > 0.")
    if phase.lr <= 0.0:
        raise ValueError("Each optimizer phase must have lr > 0.")
    if phase.batch_size is not None and phase.batch_size <= 0:
        raise ValueError("Each optimizer phase batch_size must be > 0 when provided.")

    optimizer_name = phase.optimizer.strip().lower()
    if phase.scheduler is not None:
        if optimizer_name != "adam":
            raise ValueError("Optimizer schedulers currently support Adam phases only.")
        if phase.scheduler.name != "reduce_on_plateau":
            raise ValueError("Unsupported scheduler. Use scheduler.name='reduce_on_plateau' or null.")
    if optimizer_name in {"lbfgs", "bfgs", "ssbfgs", "ssbroyden"}:
        effective_full_batch = True if phase.full_batch is None else bool(phase.full_batch)
        if not effective_full_batch:
            raise ValueError(f"{phase.optimizer} phases must use full_batch=true.")
        if phase.batch_size is not None:
            raise ValueError(f"{phase.optimizer} phases must set batch_size=null because they are full-batch.")
        if phase.allow_sampling is True:
            raise ValueError(
                f"{phase.optimizer} phases must not enable sampling because curvature updates require a stable objective."
            )
    if optimizer_name in {"sssbfgs", "sssbroyden"} and phase.scheduler is not None:
        raise ValueError(f"{phase.optimizer} phases do not currently support schedulers.")


def load_optimizer_phases_from_raw(
    raw_phases: Any,
    *,
    config_label: str = "config.pinn.optimizer_phases",
) -> list[OptimizerPhase]:
    if raw_phases is None:
        raise ValueError(f"{config_label} must be configured.")
    phases: list[OptimizerPhase] = []
    for idx, item in enumerate(raw_phases):
        optimizer = str(item.optimizer)
        epochs = int(item.epochs)
        batch_size = None if getattr(item, "batch_size", None) in (None, "null") else int(item.batch_size)
        optimizer_kwargs = _normalize_optional_mapping(
            getattr(item, "optimizer_kwargs", None),
            field_name="optimizer_kwargs",
        )
        raw_scheduler = _normalize_optional_mapping(
            getattr(item, "scheduler", None),
            field_name="scheduler",
        )
        scheduler = None
        if raw_scheduler is not None:
            scheduler = SchedulerConfig(
                name=str(raw_scheduler.get("name", "reduce_on_plateau")).strip().lower(),
                mode=str(raw_scheduler.get("mode", "min")).strip().lower(),
                factor=float(raw_scheduler.get("factor", 0.5)),
                patience=int(raw_scheduler.get("patience", 10)),
                threshold=float(raw_scheduler.get("threshold", 1.0e-4)),
                threshold_mode=str(raw_scheduler.get("threshold_mode", "rel")).strip().lower(),
                cooldown=int(raw_scheduler.get("cooldown", 0)),
                min_lr=float(raw_scheduler.get("min_lr", 0.0)),
                eps=float(raw_scheduler.get("eps", 1.0e-8)),
                metric=str(raw_scheduler.get("metric", "train_total_loss")).strip().lower(),
            )
        line_search = normalize_line_search_config(
            _normalize_optional_mapping(
                getattr(item, "line_search", None),
                field_name="line_search",
            )
        )
        convergence = _normalize_optional_mapping(
            getattr(item, "convergence", None),
            field_name="convergence",
        )
        phase = OptimizerPhase(
            name=str(getattr(item, "name", f"phase_{idx:02d}")),
            optimizer=optimizer,
            lr=float(item.lr),
            epochs=epochs,
            batch_size=batch_size,
            shuffle=bool(getattr(item, "shuffle", True)),
            full_batch=None
            if getattr(item, "full_batch", None) in (None, "null")
            else bool(getattr(item, "full_batch")),
            allow_sampling=None
            if getattr(item, "allow_sampling", None) in (None, "null")
            else bool(getattr(item, "allow_sampling")),
            optimizer_kwargs={} if optimizer_kwargs is None else optimizer_kwargs,
            scheduler=scheduler,
            line_search=line_search,
            convergence=convergence,
        )
        _validate_optimizer_phase(phase)
        phases.append(phase)

    return phases


def load_optimizer_phases(config: Any) -> list[OptimizerPhase]:
    raw_phases = getattr(config.pinn, "optimizer_phases", None)
    return load_optimizer_phases_from_raw(raw_phases, config_label="config.pinn.optimizer_phases")


def resolve_pinn_training_mode(config: Any) -> str:
    raw_mode = str(cfg_get(config, "pinn.mode", "single_stage")).strip().lower()
    if raw_mode in {"single_stage", "pinn"}:
        return "pinn"
    if raw_mode == "multistage":
        return "multistage_pinn"
    raise ValueError("pinn.mode must be one of: single_stage, multistage.")


def _resolve_multistage_optimizer_schedule(config: Any) -> ResolvedMultistageOptimizerSchedule:
    raw_explicit = cfg_get(config, "pinn.multistage.stage_optimizer_phases", None)
    explicit_stage_optimizer_phases = None
    if raw_explicit not in (None, "null"):
        explicit_stage_optimizer_phases = [
            load_optimizer_phases_from_raw(
                raw_schedule,
                config_label=f"config.pinn.multistage.stage_optimizer_phases[{idx}]",
            )
            for idx, raw_schedule in enumerate(raw_explicit)
        ]

    raw_base = cfg_get(config, "pinn.multistage.base_stage.optimizer_phases", None)
    if raw_base in (None, "null"):
        base_optimizer_phases = load_optimizer_phases(config)
    else:
        base_optimizer_phases = load_optimizer_phases_from_raw(
            raw_base,
            config_label="config.pinn.multistage.base_stage.optimizer_phases",
        )

    raw_residual = cfg_get(config, "pinn.multistage.residual_stage.optimizer_phases", None)
    if raw_residual in (None, "null"):
        residual_optimizer_phases = load_optimizer_phases(config)
    else:
        residual_optimizer_phases = load_optimizer_phases_from_raw(
            raw_residual,
            config_label="config.pinn.multistage.residual_stage.optimizer_phases",
        )

    return ResolvedMultistageOptimizerSchedule(
        explicit_stage_optimizer_phases=explicit_stage_optimizer_phases,
        base_optimizer_phases=base_optimizer_phases,
        residual_optimizer_phases=residual_optimizer_phases,
    )


def _resolve_loss_weights(config: Any) -> LossWeights:
    return LossWeights(
        data=float(cfg_get(config, "pinn.loss_weights.data", 1.0)),
        dt=float(cfg_get(config, "pinn.loss_weights.dt", 1.0e-4)),
        physics=float(cfg_get(config, "pinn.loss_weights.physics", 1.0)),
        ic=float(cfg_get(config, "pinn.loss_weights.ic", 1.0)),
    )


def _resolve_loss_weight_schedule(config: Any) -> tuple[ResolvedLossWeightScheduleStage, ...] | None:
    raw_schedule = cfg_get(config, "pinn.loss_weight_schedule", None)
    if raw_schedule in (None, "null"):
        return None
    schedule: list[ResolvedLossWeightScheduleStage] = []
    for idx, item in enumerate(raw_schedule):
        raw_epochs = getattr(item, "epochs", None)
        epochs = None if raw_epochs in (None, "null") else int(raw_epochs)
        if epochs is not None and epochs <= 0:
            raise ValueError(f"config.pinn.loss_weight_schedule[{idx}].epochs must be > 0 when provided.")
        raw_weights = getattr(item, "weights", None)
        if raw_weights in (None, "null"):
            raise ValueError(f"config.pinn.loss_weight_schedule[{idx}].weights must be provided.")
        schedule.append(
            ResolvedLossWeightScheduleStage(
                epochs=epochs,
                weights=LossWeights(
                    data=float(getattr(raw_weights, "data")),
                    dt=float(getattr(raw_weights, "dt")),
                    physics=float(getattr(raw_weights, "physics")),
                    ic=float(getattr(raw_weights, "ic")),
                ),
            )
        )
    if not schedule:
        raise ValueError("config.pinn.loss_weight_schedule must not be empty when provided.")
    return tuple(schedule)


def _optional_int(config: Any, key: str) -> int | None:
    value = cfg_get(config, key, None)
    if value in (None, "null"):
        return None
    return int(value)


def _resolve_supervised_acquisition(config: Any) -> ResolvedSupervisedAcquisitionSettings:
    enabled = bool(cfg_get(config, "pinn.supervised_acquisition.enabled", False))
    strategy = str(cfg_get(config, "pinn.supervised_acquisition.strategy", "random")).strip().lower()
    supported_strategies = {"random", "mae_topk", "mae_weighted"}
    if strategy not in supported_strategies:
        raise ValueError(
            "pinn.supervised_acquisition.strategy must be one of: "
            f"{', '.join(sorted(supported_strategies))}."
        )

    initial_trajectories = _optional_int(config, "pinn.supervised_acquisition.initial_trajectories")
    max_trajectories = _optional_int(config, "pinn.supervised_acquisition.max_trajectories")
    add_trajectories = int(cfg_get(config, "pinn.supervised_acquisition.add_trajectories", 1))
    refresh_period_epochs = int(cfg_get(config, "pinn.supervised_acquisition.refresh_period_epochs", 500))
    candidate_batch_size = int(cfg_get(config, "pinn.supervised_acquisition.candidate_batch_size", 4096))
    seed = int(cfg_get(config, "pinn.supervised_acquisition.seed", cfg_get(config, "model.seed", 0)))

    if enabled:
        if initial_trajectories is not None and initial_trajectories <= 0:
            raise ValueError("pinn.supervised_acquisition.initial_trajectories must be > 0 when provided.")
        if max_trajectories is not None and max_trajectories <= 0:
            raise ValueError("pinn.supervised_acquisition.max_trajectories must be > 0 when provided.")
        if add_trajectories <= 0:
            raise ValueError("pinn.supervised_acquisition.add_trajectories must be > 0.")
        if refresh_period_epochs <= 0:
            raise ValueError("pinn.supervised_acquisition.refresh_period_epochs must be > 0.")
        if candidate_batch_size <= 0:
            raise ValueError("pinn.supervised_acquisition.candidate_batch_size must be > 0.")
        if initial_trajectories is not None and max_trajectories is not None:
            if initial_trajectories > max_trajectories:
                raise ValueError(
                    "pinn.supervised_acquisition.initial_trajectories must be <= "
                    "pinn.supervised_acquisition.max_trajectories."
                )

    return ResolvedSupervisedAcquisitionSettings(
        enabled=enabled,
        strategy=strategy,
        initial_trajectories=initial_trajectories,
        add_trajectories=add_trajectories,
        max_trajectories=max_trajectories,
        refresh_period_epochs=refresh_period_epochs,
        candidate_batch_size=candidate_batch_size,
        seed=seed,
    )


def resolve_pinn_stack(config: Any) -> ResolvedPinnStack:
    mode = resolve_pinn_training_mode(config)
    optimizer_phases = load_optimizer_phases(config) if mode == "pinn" else None
    multistage_optimizer_schedule = _resolve_multistage_optimizer_schedule(config) if mode == "multistage_pinn" else None
    refresh_on_phase_start = tuple(
        str(name).strip().lower()
        for name in cfg_get(config, "pinn.collocation.refresh.on_phase_start", [])
    )
    refresh_on_phase_end = tuple(
        str(name).strip().lower()
        for name in cfg_get(config, "pinn.collocation.refresh.on_phase_end", [])
    )
    return ResolvedPinnStack(
        mode=mode,
        optimizer_phases=optimizer_phases,
        multistage_optimizer_schedule=multistage_optimizer_schedule,
        base_loss_weights=_resolve_loss_weights(config),
        loss_weight_schedule=_resolve_loss_weight_schedule(config),
        weighting_config=weighting_config_from_config(config),
        vrba_config=vrba_config_from_config(config),
        collocation=ResolvedCollocationSettings(
            mode=str(cfg_get(config, "pinn.collocation.mode", "preprocessed")).strip().lower(),
            strategy=str(cfg_get(config, "pinn.collocation.strategy", "static")).strip().lower(),
            refresh_mode=str(cfg_get(config, "pinn.collocation.refresh.mode", "epoch_periodic")).strip().lower(),
            refresh_on_phase_start=refresh_on_phase_start,
            refresh_on_phase_end=refresh_on_phase_end,
        ),
        curriculum=ResolvedCurriculumSettings(
            enabled=bool(cfg_get(config, "pinn.curriculum.enabled", False)),
            balance_active_bins=bool(cfg_get(config, "pinn.curriculum.balance_active_bins", True)),
            unlock_epochs=None
            if cfg_get(config, "pinn.curriculum.unlock_epochs", None) in (None, "null")
            else tuple(int(value) for value in cfg_get(config, "pinn.curriculum.unlock_epochs", None)),
            stage_length_epochs=int(cfg_get(config, "pinn.curriculum.stage_length_epochs", 100)),
        ),
        supervised_acquisition=_resolve_supervised_acquisition(config),
    )
