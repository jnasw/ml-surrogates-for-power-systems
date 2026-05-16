"""Trainer diagnostics, epoch-row assembly, and checkpoint bookkeeping helpers."""

from __future__ import annotations

from typing import Any

from omegaconf import OmegaConf
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.pinn.logging import EpochMetrics, PinnLogger
from src.pinn.losses import PinnLossBreakdown
from src.pinn.optim import OptimizerSpec
from src.pinn.runtime import SchedulerConfig, torch_dtype_name
from src.pinn.vrba import serialize_vrba_config, serialize_vrba_state
from src.training.runtime import cfg_get


def collocation_pool_diagnostics(manager: Any) -> dict[str, float | int | str | bool | None]:
    state = getattr(manager, "state", None)
    if state is None:
        return {}
    residual_pool = state.pools.get("residual")
    ic_pool = state.pools.get("ic_constraint")
    diagnostics: dict[str, float | int | str | bool | None] = {
        "multipool_enabled": True,
        "multipool_total_target_rows": int(state.total_target_rows),
        "multipool_allocation_step": int(state.allocation_step),
    }
    if residual_pool is not None:
        diagnostics["multipool_residual_target_rows"] = int(residual_pool.target_rows)
        diagnostics["multipool_residual_epoch_rows"] = int(
            residual_pool.metadata.get("epoch_rows", residual_pool.points_x.shape[0])
        )
        diagnostics["multipool_residual_pool_rows"] = int(residual_pool.points_x.shape[0])
        for key, value in (residual_pool.metadata or {}).items():
            if key == "epoch_rows":
                continue
            if isinstance(value, (bool, int, float, str)) or value is None:
                diagnostics[f"multipool_residual_{key}"] = value
    if ic_pool is not None:
        diagnostics["multipool_ic_target_rows"] = int(ic_pool.target_rows)
        diagnostics["multipool_ic_epoch_rows"] = int(
            ic_pool.metadata.get("epoch_rows", ic_pool.points_x.shape[0])
        )
        diagnostics["multipool_ic_pool_rows"] = int(ic_pool.points_x.shape[0])
        for key, value in (ic_pool.metadata or {}).items():
            if key == "epoch_rows":
                continue
            if isinstance(value, (bool, int, float, str)) or value is None:
                diagnostics[f"multipool_ic_{key}"] = value
    for key, value in (state.metadata or {}).items():
        if isinstance(value, (bool, int, float, str)) or value is None:
            diagnostics[f"multipool_{key}"] = value
    return diagnostics


def collocation_vrba_summary(manager: Any) -> dict[str, Any]:
    state = getattr(manager, "state", None)
    if state is None:
        return {
            "enabled": False,
            "sampling_enabled": False,
            "weighting_enabled": False,
            "potential": None,
            "target_sets": None,
            "update_count": None,
        }
    metadata = dict(getattr(state, "metadata", {}) or {})
    strategy_name = metadata.get("residual_last_strategy")
    enabled = bool(
        str(strategy_name).strip().lower() == "vrba_sample"
        or bool(metadata.get("ic_constraint_vrba_weighting_configured", False))
        or bool(metadata.get("dt_vrba_weighting_configured", False))
    )
    target_sets: list[str] = []
    if enabled:
        target_sets.append("physics")
    if enabled and (
        bool(metadata.get("ic_constraint_vrba_weighting_enabled", False))
        or bool(metadata.get("ic_constraint_vrba_weighting_configured", False))
    ):
        target_sets.append("ic")
    if enabled and (
        bool(metadata.get("dt_vrba_weighting_enabled", False))
        or bool(metadata.get("dt_vrba_weighting_configured", False))
    ):
        target_sets.append("dt")
    return {
        "enabled": bool(enabled),
        "sampling_enabled": bool(enabled and bool(metadata.get("residual_vrba_sampling_enabled", False))),
        "weighting_enabled": bool(
            enabled
            and (
                bool(metadata.get("residual_vrba_weighting_enabled", False))
                or bool(metadata.get("ic_constraint_vrba_weighting_enabled", False))
                or bool(metadata.get("dt_vrba_weighting_enabled", False))
            )
        ),
        "potential": None
        if not enabled
        else (
            metadata.get("residual_vrba_potential")
            or metadata.get("ic_constraint_vrba_potential")
            or metadata.get("dt_vrba_potential")
        ),
        "target_sets": None if not target_sets else tuple(target_sets),
        "update_count": None if not enabled else metadata.get("residual_vrba_update_count"),
    }


def collocation_vrba_details(manager: Any) -> dict[str, Any]:
    state = getattr(manager, "state", None)
    metadata = dict(getattr(state, "metadata", {}) or {}) if state is not None else {}
    return {
        "sampling_frozen": bool(metadata.get("residual_vrba_sampling_frozen", False)),
        "weighting_frozen": bool(
            bool(metadata.get("residual_vrba_weighting_frozen", False))
            or bool(metadata.get("ic_constraint_vrba_weighting_frozen", False))
            or bool(metadata.get("dt_vrba_weighting_frozen", False))
        ),
        "physics_lambda_mean": metadata.get("residual_vrba_lambda_mean"),
        "physics_lambda_max": metadata.get("residual_vrba_lambda_max"),
        "ic_lambda_mean": metadata.get("ic_constraint_vrba_lambda_mean"),
        "ic_lambda_max": metadata.get("ic_constraint_vrba_lambda_max"),
        "dt_lambda_mean": metadata.get("dt_vrba_lambda_mean"),
        "dt_lambda_max": metadata.get("dt_vrba_lambda_max"),
    }


def serialize_vrba_runtime_state(
    *,
    collocation_manager: Any | None,
    train_dt_weights: torch.Tensor | None,
) -> dict[str, Any] | None:
    if collocation_manager is None and train_dt_weights is None:
        return None
    runtime_state: dict[str, Any] = {}
    if collocation_manager is not None and hasattr(collocation_manager, "export_adaptive_state"):
        runtime_state["collocation"] = collocation_manager.export_adaptive_state()
    if train_dt_weights is not None:
        runtime_state["dt_weights"] = train_dt_weights.detach().cpu().clone()
    return runtime_state


def build_checkpoint_payload(
    *,
    pinn_model: Any,
    metrics: EpochMetrics | None,
    config: Any,
    tag: str,
    vrba_config: Any | None = None,
    vrba_state: Any | None = None,
    collocation_manager: Any | None = None,
    train_dt_weights: torch.Tensor | None = None,
) -> dict[str, Any]:
    track_vrba_runtime = bool(vrba_config is not None and getattr(vrba_config, "track_in_checkpoints", True))
    vrba_runtime_state = None
    if track_vrba_runtime:
        vrba_runtime_state = serialize_vrba_runtime_state(
            collocation_manager=collocation_manager,
            train_dt_weights=train_dt_weights,
        )
    return {
        "state_dict": pinn_model.model.state_dict(),
        "input_dim": pinn_model.input_dim,
        "output_dim": pinn_model.output_dim,
        "hidden_dim": pinn_model.hidden_dim,
        "hidden_layers": pinn_model.hidden_layers,
        "activation": pinn_model.activation,
        "dtype": torch_dtype_name(pinn_model.dtype),
        "architecture": pinn_model.architecture,
        "stage_first_activations": []
        if pinn_model.stage_first_activations is None
        else list(pinn_model.stage_first_activations),
        "stage_hidden_activations": []
        if pinn_model.stage_hidden_activations is None
        else list(pinn_model.stage_hidden_activations),
        "stage_time_scales": []
        if pinn_model.stage_time_scales is None
        else [float(x) for x in pinn_model.stage_time_scales],
        "stage_epsilons": []
        if pinn_model.stage_epsilons is None
        else [float(x) for x in pinn_model.stage_epsilons],
        "checkpoint_tag": tag,
        "metrics": metrics,
        "config": OmegaConf.to_container(config, resolve=True),
        "vrba": {
            "config": None if vrba_config is None else serialize_vrba_config(vrba_config),
            "state": None if vrba_state is None else serialize_vrba_state(vrba_state),
            "runtime_state": vrba_runtime_state,
        },
    }


def checkpointing_enabled(config: Any, key: str, default: bool) -> bool:
    if not bool(cfg_get(config, "pinn.checkpointing.enabled", True)):
        return False
    return bool(cfg_get(config, f"pinn.checkpointing.{key}", default))


def resolve_checkpoint_milestones(config: Any, total_epochs: int) -> dict[int, str]:
    raw = cfg_get(config, "pinn.checkpointing.epoch_fractions", [])
    if raw in (None, "null"):
        return {}

    milestones: dict[int, str] = {}
    for item in raw:
        fraction = float(item)
        if not (0.0 < fraction <= 1.0):
            raise ValueError("pinn.checkpointing.epoch_fractions values must be in (0, 1].")
        epoch = max(1, min(total_epochs, int(round(total_epochs * fraction))))
        percent = int(round(fraction * 100.0))
        tag = f"epoch_{percent:03d}pct"
        milestones.setdefault(epoch, tag)
    return dict(sorted(milestones.items()))


def build_epoch_metrics_row(
    *,
    epoch: int,
    global_epoch: int,
    phase_name: str,
    optimizer_name: str,
    train_total_loss: float,
    train_component_losses: dict[str, float],
    train_total_grad_norm: float | None,
    train_component_grad_norms: dict[str, float] | None,
    train_weighted_component_grad_norms: dict[str, float] | None,
    val_total_loss: float | None,
    val_component_losses: dict[str, float] | None,
    test_metrics: dict[str, float] | None,
    regression_metrics: dict[str, dict[str, float]] | None,
    weighting_scheme: str,
    weighting_updated: bool,
    train_loss_weights: dict[str, float],
    weighting_raw_candidate_weights: dict[str, float],
    weighting_probe_grad_l2_norms: dict[str, float] | None,
    weighting_probe_grad_mean_abs: dict[str, float] | None,
    weighting_probe_grad_max_abs: dict[str, float] | None,
    weighting_probe_grad_std: dict[str, float] | None,
    weighting_probe_ntk_mean_trace: dict[str, float] | None,
    weighting_probe_ntk_batch_sizes: dict[str, int] | None,
    weighting_anchor: str | None,
    vrba_summary: dict[str, Any],
    vrba_details: dict[str, Any],
    epoch_wall_seconds: float | None,
    cumulative_wall_seconds: float | None,
    num_batches: int | None,
    num_train_steps: int | None,
    num_supervised_rows: int | None,
    num_collocation_rows: int | None,
    num_init_rows: int | None,
    peak_gpu_memory_allocated_bytes: int | None,
    peak_gpu_memory_reserved_bytes: int | None,
    optimizer_diagnostics: dict[str, Any],
    supervised_acquisition: Any | None = None,
) -> EpochMetrics:
    acquisition = None if supervised_acquisition is None else supervised_acquisition.as_dict()
    return EpochMetrics(
        epoch=epoch,
        global_epoch=global_epoch,
        phase_name=phase_name,
        optimizer=optimizer_name,
        train_total_loss=train_total_loss,
        train_component_losses=train_component_losses,
        train_total_grad_norm=train_total_grad_norm,
        train_component_grad_norms=train_component_grad_norms,
        train_weighted_component_grad_norms=train_weighted_component_grad_norms,
        val_total_loss=val_total_loss,
        val_component_losses=val_component_losses,
        test_metrics=test_metrics,
        regression_metrics=regression_metrics,
        weighting_scheme=weighting_scheme,
        weighting_updated=weighting_updated,
        train_loss_weights=train_loss_weights,
        weighting_raw_candidate_weights=weighting_raw_candidate_weights,
        weighting_probe_grad_l2_norms=weighting_probe_grad_l2_norms,
        weighting_probe_grad_mean_abs=weighting_probe_grad_mean_abs,
        weighting_probe_grad_max_abs=weighting_probe_grad_max_abs,
        weighting_probe_grad_std=weighting_probe_grad_std,
        weighting_probe_ntk_mean_trace=weighting_probe_ntk_mean_trace,
        weighting_probe_ntk_batch_sizes=weighting_probe_ntk_batch_sizes,
        weighting_anchor=weighting_anchor,
        vrba_enabled=bool(vrba_summary["enabled"]),
        vrba_sampling_enabled=bool(vrba_summary["sampling_enabled"]),
        vrba_weighting_enabled=bool(vrba_summary["weighting_enabled"]),
        vrba_potential=vrba_summary["potential"],
        vrba_target_sets=vrba_summary["target_sets"],
        vrba_update_count=None if vrba_summary["update_count"] is None else int(vrba_summary["update_count"]),
        vrba_sampling_frozen=bool(vrba_details["sampling_frozen"]),
        vrba_weighting_frozen=bool(vrba_details["weighting_frozen"]),
        vrba_physics_lambda_mean=vrba_details["physics_lambda_mean"],
        vrba_physics_lambda_max=vrba_details["physics_lambda_max"],
        vrba_ic_lambda_mean=vrba_details["ic_lambda_mean"],
        vrba_ic_lambda_max=vrba_details["ic_lambda_max"],
        vrba_dt_lambda_mean=vrba_details["dt_lambda_mean"],
        vrba_dt_lambda_max=vrba_details["dt_lambda_max"],
        epoch_wall_seconds=epoch_wall_seconds,
        cumulative_wall_seconds=cumulative_wall_seconds,
        num_batches=num_batches,
        num_train_steps=num_train_steps,
        num_supervised_rows=num_supervised_rows,
        num_collocation_rows=num_collocation_rows,
        num_init_rows=num_init_rows,
        supervised_acquisition_enabled=bool(acquisition and acquisition.get("enabled")),
        supervised_acquisition_strategy=None if acquisition is None else acquisition.get("strategy"),
        supervised_active_trajectories=None if acquisition is None else acquisition.get("active_trajectories"),
        supervised_candidate_trajectories=None if acquisition is None else acquisition.get("candidate_trajectories"),
        supervised_active_rows=None if acquisition is None else acquisition.get("active_rows"),
        supervised_acquired_trajectories=None if acquisition is None else acquisition.get("acquired_trajectories"),
        supervised_acquisition_count=None if acquisition is None else acquisition.get("acquisition_count"),
        supervised_last_acquisition_epoch=None if acquisition is None else acquisition.get("last_acquisition_epoch"),
        supervised_last_acquired_trajectory_ids=None
        if acquisition is None or acquisition.get("last_acquired_trajectory_ids") is None
        else tuple(int(value) for value in acquisition.get("last_acquired_trajectory_ids")),
        supervised_last_anchor_score_mean=None if acquisition is None else acquisition.get("last_anchor_score_mean"),
        supervised_last_anchor_score_max=None if acquisition is None else acquisition.get("last_anchor_score_max"),
        supervised_last_selected_distance_mean=None if acquisition is None else acquisition.get("last_selected_distance_mean"),
        supervised_last_selected_distance_max=None if acquisition is None else acquisition.get("last_selected_distance_max"),
        peak_gpu_memory_allocated_bytes=peak_gpu_memory_allocated_bytes,
        peak_gpu_memory_reserved_bytes=peak_gpu_memory_reserved_bytes,
        optimizer_diagnostics=optimizer_diagnostics,
    )


def mean_epoch_breakdown(
    epoch_losses: list[PinnLossBreakdown],
    train_total: float,
    train_component_losses: dict[str, float],
) -> PinnLossBreakdown:
    return PinnLossBreakdown(
        total=epoch_losses[0].total.detach().new_tensor(train_total),
        components={
            name: epoch_losses[0].components[name].detach().new_tensor(train_component_losses[name])
            for name in epoch_losses[0].components
        },
    )


def record_epoch_row_and_checkpoints(
    *,
    row: EpochMetrics,
    rows: list[EpochMetrics],
    mean_epoch_breakdown: PinnLossBreakdown,
    logger: PinnLogger | None,
    checkpoint_milestones: dict[int, str],
    pinn_model: Any,
    config: Any,
    vrba_config: Any,
    vrba_state: Any,
    collocation_manager: Any,
    train_dt_weights: torch.Tensor | None,
    best_metric: float | None,
    validation_available: bool = False,
) -> float | None:
    rows.append(row)
    collocation_manager.observe_epoch_losses(
        global_epoch=row.global_epoch,
        losses=mean_epoch_breakdown,
    )

    if logger is not None:
        logger.print_epoch_metrics(row)
        logger.log_epoch_metrics(row)
        milestone_tag = checkpoint_milestones.get(row.global_epoch)
        if milestone_tag is not None:
            logger.save_checkpoint(
                build_checkpoint_payload(
                    pinn_model=pinn_model,
                    metrics=row,
                    config=config,
                    tag=milestone_tag,
                    vrba_config=vrba_config,
                    vrba_state=vrba_state,
                    collocation_manager=collocation_manager,
                    train_dt_weights=train_dt_weights,
                ),
                tag=milestone_tag,
            )
        if checkpointing_enabled(config, "save_last", True):
            logger.save_checkpoint(
                build_checkpoint_payload(
                    pinn_model=pinn_model,
                    metrics=row,
                    config=config,
                    tag="last",
                    vrba_config=vrba_config,
                    vrba_state=vrba_state,
                    collocation_manager=collocation_manager,
                    train_dt_weights=train_dt_weights,
                ),
                tag="last",
            )

    if validation_available and row.val_total_loss is None:
        return best_metric
    selection_metric = row.val_total_loss if row.val_total_loss is not None else row.train_total_loss
    if best_metric is None or selection_metric < best_metric:
        best_metric = selection_metric
        if logger is not None and checkpointing_enabled(config, "save_best", True):
            logger.save_checkpoint(
                build_checkpoint_payload(
                    pinn_model=pinn_model,
                    metrics=row,
                    config=config,
                    tag="best",
                    vrba_config=vrba_config,
                    vrba_state=vrba_state,
                    collocation_manager=collocation_manager,
                    train_dt_weights=train_dt_weights,
                ),
                tag="best",
            )
    return best_metric


def scheduler_metric_value(
    *,
    scheduler_config: SchedulerConfig | None,
    train_total_loss: float,
    val_total_loss: float | None,
) -> tuple[float | None, str | None]:
    if scheduler_config is None:
        return None, None
    metric_name = str(scheduler_config.metric)
    if metric_name == "train_total_loss":
        return float(train_total_loss), metric_name
    if metric_name == "val_total_loss":
        if val_total_loss is not None:
            return float(val_total_loss), metric_name
        return None, "val_total_loss_unavailable"
    raise ValueError("Unsupported scheduler metric. Use scheduler.metric='train_total_loss' or 'val_total_loss'.")


def current_optimizer_lr(optimizer: torch.optim.Optimizer) -> float | None:
    if not optimizer.param_groups:
        return None
    lr = optimizer.param_groups[0].get("lr")
    return None if lr is None else float(lr)


def scheduler_diagnostics(
    *,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau | None,
    scheduler_metric_name: str | None,
    scheduler_metric_value: float | None,
) -> dict[str, float | int | str | bool | None]:
    diagnostics: dict[str, float | int | str | bool | None] = {
        "current_lr": current_optimizer_lr(optimizer),
        "scheduler_enabled": bool(scheduler is not None),
    }
    if scheduler is None:
        diagnostics["scheduler_name"] = None
        return diagnostics
    diagnostics["scheduler_name"] = "reduce_on_plateau"
    diagnostics["scheduler_metric"] = scheduler_metric_name
    diagnostics["scheduler_metric_value"] = scheduler_metric_value
    diagnostics["scheduler_best"] = None if getattr(scheduler, "best", None) is None else float(scheduler.best)
    diagnostics["scheduler_num_bad_epochs"] = int(getattr(scheduler, "num_bad_epochs", 0))
    diagnostics["scheduler_cooldown_counter"] = int(getattr(scheduler, "cooldown_counter", 0))
    return diagnostics


def build_single_stage_optimizer_diagnostics(
    *,
    optimizer_spec: OptimizerSpec,
    phase_full_batch: bool,
    phase_allows_sampling: bool,
    phase_supports_dynamic_weight_updates: bool,
    step_weight_updates_enabled: bool,
    weighting_policy: Any,
    weighting_state: Any,
    scheduler: ReduceLROnPlateau | None,
    scheduler_metric_name: str | None,
    scheduler_metric_value: float | None,
    optimizer: torch.optim.Optimizer,
    collocation_manager: Any,
) -> dict[str, Any]:
    return {
        "requires_closure": optimizer_spec.requires_closure,
        "full_batch": phase_full_batch,
        "sampling_enabled": phase_allows_sampling,
        "line_search": optimizer_spec.line_search_name,
        "dynamic_weight_updates_enabled": phase_supports_dynamic_weight_updates,
        "dynamic_weight_update_mode": str(getattr(weighting_policy, "update_mode", "epoch")),
        "dynamic_weight_update_step_mode": bool(step_weight_updates_enabled),
        "weighting_probe_refresh_each_update": bool(getattr(weighting_policy.config, "probe_refresh_each_update", False)),
        "weighting_policy_anchor": getattr(weighting_policy.config, "anchor", None),
        "weighting_policy_scheme": getattr(weighting_policy.config, "scheme", None),
        **scheduler_diagnostics(
            optimizer=optimizer,
            scheduler=scheduler,
            scheduler_metric_name=scheduler_metric_name,
            scheduler_metric_value=scheduler_metric_value,
        ),
        **collocation_pool_diagnostics(collocation_manager),
    }
