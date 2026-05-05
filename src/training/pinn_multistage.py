"""Focused multistage PINN loop implementation."""

from __future__ import annotations

import gc
import json
import os
from typing import Any

import numpy as np
import torch

from src.pinn.residuals import compute_residual_terms as _compute_residual_terms
from src.training import trainer as trainer_impl
from src.training.pinn_runtime import measure_pinn_state as _measure_pinn_state


def _cleanup_cuda_stage_boundary() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_multistage_pinn_loop(
    *,
    dataset,
    ode_model: Any,
    config: Any,
    resolved_stack,
    logger=None,
):
    seed = int(trainer_impl.cfg_get(config, "model.seed", 0))
    trainer_impl.configure_reproducibility_from_config(seed=seed, config=config, prefix="pinn")

    dtype = trainer_impl.resolve_torch_dtype(str(trainer_impl.cfg_get(config, "pinn.dtype", "float64")))
    device = trainer_impl.select_torch_device(str(trainer_impl.cfg_get(config, "pinn.device", "auto")))
    hidden_dim = int(trainer_impl.cfg_get(config, "pinn.hidden_dim", 64))
    hidden_layers = int(trainer_impl.cfg_get(config, "pinn.hidden_layers", 4))
    activation = str(trainer_impl.cfg_get(config, "pinn.activation", "tanh"))
    formulation = str(trainer_impl.cfg_get(config, "pinn.formulation", "odequations"))
    if resolved_stack.multistage_optimizer_schedule is None:
        raise ValueError("Resolved multistage optimizer schedule is required for multistage PINN mode.")
    explicit_stage_optimizer_phases = resolved_stack.multistage_optimizer_schedule.explicit_stage_optimizer_phases
    base_optimizer_phases = resolved_stack.multistage_optimizer_schedule.base_optimizer_phases
    residual_optimizer_phases = resolved_stack.multistage_optimizer_schedule.residual_optimizer_phases
    base_weights = resolved_stack.base_loss_weights
    vrba_config = resolved_stack.vrba_config
    gradient_telemetry_enabled = bool(trainer_impl.cfg_get(config, "pinn.gradient_telemetry.enabled", False))
    cost_tracking_enabled = trainer_impl._cost_tracking_enabled(config)
    criterion = trainer_impl.nn.MSELoss()
    transition_settings = trainer_impl._optimizer_transition_settings(config)
    transition_state = trainer_impl.OptimizerTransitionState(cache={})

    dataset = trainer_impl.move_pinn_training_data_to_device(dataset, device=device, dtype=dtype)
    collocation_manager = trainer_impl.build_collocation_manager(
        initial_points=dataset.train_col_x,
        init_x=dataset.train_init_x,
        init_y=dataset.train_init_y,
        config=config,
    )
    initial_pool_batch = collocation_manager.initial_epoch_batch()
    dataset = trainer_impl.PinnDatasetBundle(
        train_x=dataset.train_x,
        train_y=dataset.train_y,
        train_col_x=initial_pool_batch.x_col,
        train_init_x=initial_pool_batch.x_init,
        train_init_y=initial_pool_batch.y_init,
        train_traj_ids=dataset.train_traj_ids,
        train_difficulty_scores=dataset.train_difficulty_scores,
        train_difficulty_bins=dataset.train_difficulty_bins,
        train_dt_weights=trainer_impl._initialize_dt_vrba_weights(vrba_config=vrba_config, train_x=dataset.train_x),
        train_col_weights=initial_pool_batch.x_col_weights,
        train_init_weights=initial_pool_batch.x_init_weights,
        val_x=dataset.val_x,
        val_y=dataset.val_y,
        val_col_x=dataset.val_col_x,
        test_x=dataset.test_x,
        test_y=dataset.test_y,
    )
    vrba_state = trainer_impl.initialize_vrba_state(
        vrba_config,
        initial_point_counts={
            "data": int(dataset.train_x.shape[0]),
            "dt": int(dataset.train_x.shape[0]),
            "physics": int(dataset.train_col_x.shape[0]),
            "ic": int(dataset.train_init_x.shape[0]),
        },
    )
    analysis_x_col = trainer_impl._multistage_analysis_collocation(dataset, config)
    max_stages = (
        len(explicit_stage_optimizer_phases)
        if explicit_stage_optimizer_phases is not None
        else trainer_impl._multistage_max_stages(config)
    )
    stop_threshold = trainer_impl._multistage_stop_threshold(config)
    if explicit_stage_optimizer_phases is not None:
        total_epochs = int(sum(sum(phase.epochs for phase in schedule) for schedule in explicit_stage_optimizer_phases))
    else:
        base_total_epochs = int(sum(phase.epochs for phase in base_optimizer_phases))
        residual_total_epochs = int(sum(phase.epochs for phase in residual_optimizer_phases))
        total_epochs = int(base_total_epochs + max(0, max_stages - 1) * residual_total_epochs)
    checkpoint_milestones = trainer_impl._resolve_checkpoint_milestones(config, total_epochs)

    stages: list[Any] = []
    epsilons: list[float] = []
    rows: list[Any] = []
    best_metric: float | None = None
    global_epoch = 0
    train_started = trainer_impl.time.perf_counter()
    ensemble = None
    stage_start_diagnostics = None
    train_dt_weights = dataset.train_dt_weights
    prev_stage_total_loss: float | None = None

    for stage_idx in range(max_stages):
        if stage_idx > 0:
            _cleanup_cuda_stage_boundary()
        stage_wall_start = trainer_impl.time.perf_counter()
        if stage_idx == 0:
            time_scale = 1.0
        else:
            if ensemble is None:
                raise RuntimeError("Expected an existing ensemble before training a residual stage.")
            residual_terms = _compute_residual_terms(
                model=ensemble,
                x=analysis_x_col,
                ode_model=ode_model,
                formulation=formulation,
                create_graph=False,
            )
            stage_start_diagnostics = trainer_impl.estimate_stage_diagnostics(
                x_col=analysis_x_col,
                residual=residual_terms.residual,
                min_kappa=float(trainer_impl.cfg_get(config, "pinn.multistage.residual_stage.kappa_min", 1.0)),
                max_kappa=float(trainer_impl.cfg_get(config, "pinn.multistage.residual_stage.kappa_max", 100.0)),
            )
            if logger is not None:
                _probe_path = os.path.join(logger.run_dir, "stage_residual_probe.json")
                _probe_entries: list = []
                if os.path.isfile(_probe_path):
                    with open(_probe_path) as _f:
                        _probe_entries = json.load(_f)
                _probe_entries.append({
                    "stage_idx": stage_idx,
                    "residual_rms": float(stage_start_diagnostics.residual_rms),
                    "residual_max": float(residual_terms.residual.abs().max().item()),
                    "kappa_suggested": float(stage_start_diagnostics.kappa_suggested),
                    "zero_crossings": int(stage_start_diagnostics.residual_zero_crossings),
                    "dominant_residual_channel": int(stage_start_diagnostics.dominant_residual_channel),
                    "probe_collocation_rows": int(analysis_x_col.shape[0]),
                })
                with open(_probe_path, "w") as _f:
                    json.dump(_probe_entries, _f, indent=2)
            if stage_start_diagnostics.residual_rms <= stop_threshold:
                break
            time_scale = stage_start_diagnostics.kappa_suggested

        stage_model = trainer_impl._build_multistage_stage_model_from_config(
            config=config,
            stage_idx=stage_idx,
            input_dim=dataset.input_dim,
            output_dim=dataset.output_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            base_activation=activation,
            time_scale=time_scale,
        ).to(device=device, dtype=dtype)
        stages.append(stage_model)
        epsilons.append(trainer_impl._multistage_stage_epsilon(config, stage_idx))
        ensemble = trainer_impl.MultistagePinnEnsemble(stages=list(stages), epsilons=list(epsilons)).to(device=device, dtype=dtype)
        ensemble.freeze_stages_up_to(stage_idx)
        pinn_model = trainer_impl._assemble_multistage_pinn_model(
            ensemble=ensemble,
            device=device,
            dtype=dtype,
            input_dim=dataset.input_dim,
            output_dim=dataset.output_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation=activation,
        )

        if logger is not None and stage_idx == 0 and trainer_impl._checkpointing_enabled(config, "save_init", True):
            logger.save_checkpoint(
                trainer_impl._build_checkpoint_payload(
                    pinn_model=pinn_model,
                    metrics=None,
                    config=config,
                    tag="init",
                    vrba_config=vrba_config,
                    vrba_state=vrba_state,
                    collocation_manager=collocation_manager,
                    train_dt_weights=train_dt_weights,
                ),
                tag="init",
            )

        current_stage = ensemble.stages[stage_idx]
        if explicit_stage_optimizer_phases is not None:
            stage_optimizer_phases = explicit_stage_optimizer_phases[stage_idx]
        else:
            stage_optimizer_phases = base_optimizer_phases if stage_idx == 0 else residual_optimizer_phases
        stage_total_epochs = int(sum(phase.epochs for phase in stage_optimizer_phases))
        target_stage_epsilon = float(epsilons[stage_idx])
        initial_stage_epsilon = trainer_impl._multistage_effective_epsilon(
            target_epsilon=target_stage_epsilon,
            stage_idx=stage_idx,
            stage_epoch=0,
            stage_total_epochs=stage_total_epochs,
            config=config,
        )
        ensemble.set_stage_epsilon(stage_idx, initial_stage_epsilon)
        stage_loss_ref = 1.0
        if trainer_impl._multistage_loss_ref_enabled(config):
            stage_ref_losses, _ = _measure_pinn_state(
                model=ensemble,
                criterion=criterion,
                ode_model=ode_model,
                formulation=formulation,
                weights=base_weights,
                x_data=dataset.train_x,
                y_data=dataset.train_y,
                x_data_dt_weights=trainer_impl._active_dt_weights_or_none(vrba_config=vrba_config, x_data_dt_weights=train_dt_weights),
                x_col=dataset.train_col_x,
                x_col_weights=trainer_impl._active_collocation_weights_or_none(vrba_config=vrba_config, x_col_weights=dataset.train_col_weights),
                x_init=dataset.train_init_x,
                y_init=dataset.train_init_y,
                x_init_weights=trainer_impl._active_init_weights_or_none(vrba_config=vrba_config, x_init_weights=dataset.train_init_weights),
                capture_gradient_telemetry=False,
            )
            stage_loss_ref = max(
                float(stage_ref_losses.total.detach().cpu().item()),
                trainer_impl._multistage_loss_ref_min_value(config),
            )
        stage_epoch_cursor = 0
        for phase in stage_optimizer_phases:
            optimizer_spec = trainer_impl.build_optimizer(
                model=current_stage,
                optimizer_name=phase.optimizer,
                lr=phase.lr,
                optimizer_kwargs=phase.optimizer_kwargs,
                line_search=phase.line_search,
            )
            scheduler = trainer_impl._build_phase_scheduler(
                optimizer=optimizer_spec.optimizer,
                scheduler_config=phase.scheduler,
            )
            phase_full_batch = trainer_impl._phase_effective_full_batch(phase, optimizer_spec)
            if phase_full_batch and phase.batch_size is not None:
                raise ValueError(f"Optimizer phase '{phase.name}' is full-batch and must set batch_size=null.")
            phase_allows_sampling = trainer_impl._phase_allows_sampling(phase, optimizer_spec)
            if phase_full_batch and phase_allows_sampling:
                raise ValueError(f"Optimizer phase '{phase.name}' is full-batch and must disable sampling for a stable objective.")
            if phase_full_batch:
                phase_batch_size = None
            else:
                if not optimizer_spec.supports_minibatch:
                    raise ValueError(f"Optimizer '{phase.optimizer}' does not support minibatch execution.")
                phase_batch_size = phase.batch_size if phase.batch_size is not None else int(trainer_impl.cfg_get(config, "pinn.default_batch_size", 1024))
            refresh_on_phase_start = trainer_impl._phase_boundary_refresh_planned(
                resolved_stack.collocation,
                phase_name=phase.name,
                event="start",
            )
            transition_diagnostics = trainer_impl._maybe_restore_optimizer_state(
                optimizer_spec=optimizer_spec,
                phase=phase,
                config=config,
                transition_settings=transition_settings,
                transition_state=transition_state,
                refresh_on_phase_start=refresh_on_phase_start,
            )

            if trainer_impl._collocation_phase_boundary_enabled(resolved_stack.collocation):
                collocation_manager.handle_phase_boundary(
                    context=trainer_impl.CollocationStrategyContext(
                        global_epoch=global_epoch,
                        phase_name=phase.name,
                        phase_allows_sampling=phase_allows_sampling,
                        phase_is_full_batch=phase_full_batch,
                        phase_epoch=0,
                        refresh_event="phase_start",
                        model=ensemble,
                        ode_model=ode_model,
                        formulation=formulation,
                    )
                )

            for epoch in range(1, phase.epochs + 1):
                stage_epoch_cursor += 1
                epoch_started = trainer_impl.time.perf_counter()
                trainer_impl._reset_peak_memory_if_cuda(device, enabled=cost_tracking_enabled)
                effective_stage_epsilon = trainer_impl._multistage_effective_epsilon(
                    target_epsilon=target_stage_epsilon,
                    stage_idx=stage_idx,
                    stage_epoch=stage_epoch_cursor,
                    stage_total_epochs=stage_total_epochs,
                    config=config,
                )
                ensemble.set_stage_epsilon(stage_idx, effective_stage_epsilon)
                ensemble.train()
                epoch_losses: list[Any] = []
                epoch_gradients: list[Any] = []
                active_weights = base_weights
                epoch_train_x, epoch_train_y, epoch_train_data_idx = trainer_impl._sample_supervised_rows_with_indices(
                    dataset.train_x,
                    dataset.train_y,
                    config,
                    resolved_stack.curriculum,
                    global_epoch=global_epoch + 1,
                    difficulty_bins=dataset.train_difficulty_bins,
                    allow_row_subsampling=phase_allows_sampling,
                )
                epoch_constraints = trainer_impl._prepare_epoch_constraint_batch(
                    collocation_manager=collocation_manager,
                    vrba_config=vrba_config,
                    train_dt_weights=train_dt_weights,
                    model=ensemble,
                    ode_model=ode_model,
                    formulation=formulation,
                    config=config,
                    global_epoch=global_epoch + 1,
                    phase_name=phase.name,
                    phase_allows_sampling=phase_allows_sampling,
                    phase_is_full_batch=phase_full_batch,
                    phase_epoch=epoch,
                    x_data=epoch_train_x,
                    y_data=epoch_train_y,
                    row_indices=epoch_train_data_idx,
                )
                train_dt_weights = epoch_constraints.train_dt_weights
                epoch_pool_batch = epoch_constraints.pool_batch
                epoch_train_col_x = epoch_pool_batch.x_col
                active_collocation_weights = epoch_constraints.active_collocation_weights
                active_dt_weights = epoch_constraints.active_dt_weights
                epoch_train_init_x = epoch_pool_batch.x_init
                epoch_train_init_y = epoch_pool_batch.y_init
                active_init_weights = epoch_constraints.active_init_weights
                num_batches = 0
                num_train_steps = 0
                num_supervised_rows = 0
                num_collocation_rows = 0
                num_init_rows = 0

                if phase_batch_size is None:
                    breakdown, gradient_telemetry = trainer_impl._train_pinn_step(
                        model=ensemble,
                        optimizer_spec=optimizer_spec,
                        criterion=criterion,
                        ode_model=ode_model,
                        formulation=formulation,
                        weights=active_weights,
                        x_data=epoch_train_x,
                        y_data=epoch_train_y,
                        x_data_dt_weights=active_dt_weights,
                        x_col=epoch_train_col_x,
                        x_col_weights=active_collocation_weights,
                        x_init=epoch_train_init_x,
                        y_init=epoch_train_init_y,
                        x_init_weights=active_init_weights,
                        capture_gradient_telemetry=gradient_telemetry_enabled,
                        objective_scale=stage_loss_ref,
                    )
                    epoch_losses.append(breakdown)
                    num_batches = 1
                    num_train_steps = 1
                    num_supervised_rows = int(epoch_train_x.shape[0])
                    num_collocation_rows = int(epoch_train_col_x.shape[0])
                    num_init_rows = int(epoch_train_init_x.shape[0])
                    if gradient_telemetry is not None:
                        epoch_gradients.append(gradient_telemetry)
                else:
                    _steps, epoch_batches = trainer_impl._iter_minibatch_batches(
                        dataset=trainer_impl.PinnDatasetBundle(
                            train_x=epoch_train_x,
                            train_y=epoch_train_y,
                            train_dt_weights=active_dt_weights,
                            train_col_x=epoch_train_col_x,
                            train_col_weights=active_collocation_weights,
                            train_init_x=epoch_train_init_x,
                            train_init_y=epoch_train_init_y,
                            train_init_weights=active_init_weights,
                        ),
                        batch_size=phase_batch_size,
                        shuffle=phase.shuffle,
                        seed=seed + stage_idx * 1000 + global_epoch,
                    )
                    for batch in epoch_batches:
                        num_batches += 1
                        num_train_steps += 1
                        num_supervised_rows += int(batch.x_data.shape[0])
                        num_collocation_rows += int(batch.x_col.shape[0])
                        num_init_rows += int(batch.x_init.shape[0])
                        breakdown, gradient_telemetry = trainer_impl._train_pinn_step(
                            model=ensemble,
                            optimizer_spec=optimizer_spec,
                            criterion=criterion,
                            ode_model=ode_model,
                            formulation=formulation,
                            weights=active_weights,
                            x_data=batch.x_data,
                            y_data=batch.y_data,
                            x_data_dt_weights=batch.x_data_dt_weights,
                            x_col=batch.x_col,
                            x_col_weights=batch.x_col_weights,
                            x_init=batch.x_init,
                            y_init=batch.y_init,
                            x_init_weights=batch.x_init_weights,
                            capture_gradient_telemetry=gradient_telemetry_enabled,
                            objective_scale=stage_loss_ref,
                        )
                        epoch_losses.append(breakdown)
                        if gradient_telemetry is not None:
                            epoch_gradients.append(gradient_telemetry)

                epoch_wall_seconds = float(trainer_impl.time.perf_counter() - epoch_started) if cost_tracking_enabled else None
                cumulative_wall_seconds = float(trainer_impl.time.perf_counter() - train_started) if cost_tracking_enabled else None
                peak_gpu_memory_allocated_bytes, peak_gpu_memory_reserved_bytes = trainer_impl._peak_memory_if_cuda(
                    device,
                    enabled=cost_tracking_enabled,
                )
                train_total = float(trainer_impl.torch.stack([x.total.detach() for x in epoch_losses]).mean().item())
                train_component_losses = {
                    name: float(trainer_impl.torch.stack([loss.components[name].detach() for loss in epoch_losses]).mean().item())
                    for name in epoch_losses[0].components
                }
                train_total_grad_norm = None
                train_component_grad_norms = None
                train_weighted_component_grad_norms = None
                if epoch_gradients:
                    train_total_grad_norm = float(np.mean([item.total_grad_norm for item in epoch_gradients]))
                    component_names = epoch_gradients[0].component_grad_norms.keys()
                    train_component_grad_norms = {
                        name: float(np.mean([item.component_grad_norms[name] for item in epoch_gradients]))
                        for name in component_names
                    }
                    train_weighted_component_grad_norms = {
                        name: float(np.mean([item.weighted_component_grad_norms[name] for item in epoch_gradients]))
                        for name in component_names
                    }
                global_epoch += 1
                val_total_loss, val_component_losses, test_metrics = trainer_impl._evaluate_epoch_validation_and_test(
                    model=ensemble,
                    dataset=dataset,
                    criterion=criterion,
                    ode_model=ode_model,
                    formulation=formulation,
                    active_weights=active_weights,
                    config=config,
                    global_epoch=global_epoch,
                    eval_init_x=dataset.val_init_x if dataset.val_init_x is not None else epoch_train_init_x,
                    eval_init_y=dataset.val_init_y if dataset.val_init_y is not None else epoch_train_init_y,
                )
                scheduler_metric_value, scheduler_metric_name = trainer_impl._scheduler_metric_value(
                    scheduler_config=phase.scheduler,
                    train_total_loss=train_total,
                    val_total_loss=val_total_loss,
                )
                if scheduler is not None and scheduler_metric_value is not None:
                    scheduler.step(float(scheduler_metric_value))

                optimizer_diagnostics = {
                    "requires_closure": optimizer_spec.requires_closure,
                    "full_batch": phase_full_batch,
                    "sampling_enabled": phase_allows_sampling,
                    "line_search": optimizer_spec.line_search_name,
                    "dynamic_weight_updates_enabled": False,
                    "multistage_stage_idx": stage_idx,
                    "multistage_num_stages": ensemble.num_stages,
                    "multistage_stage_time_scale": float(time_scale),
                    "multistage_stage_epsilon_target": float(target_stage_epsilon),
                    "multistage_stage_epsilon": float(effective_stage_epsilon),
                    "multistage_stage_loss_ref": float(stage_loss_ref),
                }
                optimizer_diagnostics.update(
                    trainer_impl._scheduler_diagnostics(
                        optimizer=optimizer_spec.optimizer,
                        scheduler=scheduler,
                        scheduler_metric_name=scheduler_metric_name,
                        scheduler_metric_value=scheduler_metric_value,
                    )
                )
                optimizer_diagnostics.update(transition_diagnostics)
                optimizer_diagnostics.update(trainer_impl._collocation_pool_diagnostics(collocation_manager))
                if stage_start_diagnostics is not None and stage_idx > 0:
                    optimizer_diagnostics["multistage_stage_start_residual_rms"] = float(stage_start_diagnostics.residual_rms)
                    optimizer_diagnostics["multistage_stage_zero_crossings"] = int(stage_start_diagnostics.residual_zero_crossings)
                    optimizer_diagnostics["multistage_stage_dominant_residual_channel"] = int(stage_start_diagnostics.dominant_residual_channel)
                if hasattr(optimizer_spec.optimizer, "get_last_diagnostics"):
                    optimizer_diagnostics.update(optimizer_spec.optimizer.get_last_diagnostics())

                vrba_summary = trainer_impl._collocation_vrba_summary(collocation_manager)
                vrba_details = trainer_impl._collocation_vrba_details(collocation_manager)
                row = trainer_impl._build_epoch_metrics_row(
                    epoch=epoch,
                    global_epoch=global_epoch,
                    phase_name=f"stage{stage_idx:02d}_{phase.name}",
                    optimizer_name=phase.optimizer,
                    train_total_loss=train_total,
                    train_component_losses=train_component_losses,
                    train_total_grad_norm=train_total_grad_norm,
                    train_component_grad_norms=train_component_grad_norms,
                    train_weighted_component_grad_norms=train_weighted_component_grad_norms,
                    val_total_loss=val_total_loss,
                    val_component_losses=val_component_losses,
                    test_metrics=test_metrics,
                    weighting_scheme="static",
                    weighting_updated=False,
                    train_loss_weights=active_weights.as_dict(),
                    weighting_raw_candidate_weights={},
                    weighting_probe_grad_l2_norms=None,
                    weighting_probe_grad_mean_abs=None,
                    weighting_probe_grad_max_abs=None,
                    weighting_probe_grad_std=None,
                    weighting_probe_ntk_mean_trace=None,
                    weighting_probe_ntk_batch_sizes=None,
                    weighting_anchor="physics",
                    vrba_summary=vrba_summary,
                    vrba_details=vrba_details,
                    epoch_wall_seconds=epoch_wall_seconds,
                    cumulative_wall_seconds=cumulative_wall_seconds,
                    num_batches=num_batches if cost_tracking_enabled else None,
                    num_train_steps=num_train_steps if cost_tracking_enabled else None,
                    num_supervised_rows=num_supervised_rows if cost_tracking_enabled else None,
                    num_collocation_rows=num_collocation_rows if cost_tracking_enabled else None,
                    num_init_rows=num_init_rows if cost_tracking_enabled else None,
                    peak_gpu_memory_allocated_bytes=peak_gpu_memory_allocated_bytes,
                    peak_gpu_memory_reserved_bytes=peak_gpu_memory_reserved_bytes,
                    optimizer_diagnostics=optimizer_diagnostics,
                )
                mean_epoch_breakdown = trainer_impl._mean_epoch_breakdown(epoch_losses, train_total, train_component_losses)
                best_metric = trainer_impl._record_epoch_row_and_checkpoints(
                    row=row,
                    rows=rows,
                    mean_epoch_breakdown=mean_epoch_breakdown,
                    logger=logger,
                    checkpoint_milestones=checkpoint_milestones,
                    pinn_model=pinn_model,
                    config=config,
                    vrba_config=vrba_config,
                    vrba_state=vrba_state,
                    collocation_manager=collocation_manager,
                    train_dt_weights=train_dt_weights,
                    best_metric=best_metric,
                )
            if trainer_impl._collocation_phase_boundary_enabled(resolved_stack.collocation):
                collocation_manager.handle_phase_boundary(
                    context=trainer_impl.CollocationStrategyContext(
                        global_epoch=global_epoch,
                        phase_name=phase.name,
                        phase_allows_sampling=phase_allows_sampling,
                        phase_is_full_batch=phase_full_batch,
                        phase_epoch=phase.epochs,
                        refresh_event="phase_end",
                        model=ensemble,
                        ode_model=ode_model,
                        formulation=formulation,
                    )
                )
            trainer_impl._cache_optimizer_state(
                optimizer_spec=optimizer_spec,
                phase=phase,
                transition_settings=transition_settings,
                transition_state=transition_state,
            )
            del optimizer_spec
            del scheduler

        ensemble.eval()

        if logger is not None and rows:
            _last_row = rows[-1]
            _stage_total_loss = float(_last_row.train_total_loss)
            _stage_improvement = (prev_stage_total_loss - _stage_total_loss) if prev_stage_total_loss is not None else 0.0
            _test_metrics = _last_row.test_metrics or {}
            _summary_path = os.path.join(logger.run_dir, "stage_summary.json")
            _stage_entries: list = []
            if os.path.isfile(_summary_path):
                with open(_summary_path) as _f:
                    _stage_entries = json.load(_f)
            _stage_entries.append({
                "stage_idx": stage_idx,
                "phase_name": f"stage{stage_idx:02d}_{stage_optimizer_phases[-1].name}",
                "optimizer": stage_optimizer_phases[-1].optimizer,
                "epochs": stage_total_epochs,
                "cumulative_epoch": global_epoch,
                "walltime_s": float(trainer_impl.time.perf_counter() - stage_wall_start),
                "final_train_total_loss": _stage_total_loss,
                "final_train_component_losses": dict(_last_row.train_component_losses) if _last_row.train_component_losses else {},
                "test_rmse": float(_test_metrics["rmse"]) if "rmse" in _test_metrics else None,
                "test_mae": float(_test_metrics["mae"]) if "mae" in _test_metrics else None,
                "stage_improvement_delta": float(_stage_improvement),
            })
            with open(_summary_path, "w") as _f:
                json.dump(_stage_entries, _f, indent=2)
            prev_stage_total_loss = _stage_total_loss

    if ensemble is None:
        raise RuntimeError("Multistage PINN training did not instantiate any stages.")
    final_model = trainer_impl._assemble_multistage_pinn_model(
        ensemble=ensemble,
        device=device,
        dtype=dtype,
        input_dim=dataset.input_dim,
        output_dim=dataset.output_dim,
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
        activation=activation,
    )
    return final_model, rows
