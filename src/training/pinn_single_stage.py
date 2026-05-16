"""Focused single-stage PINN loop implementation."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.training import trainer as trainer_impl


def _sample_epoch_supervised_rows(
    *,
    dataset,
    config,
    curriculum,
    global_epoch: int,
    phase_allows_sampling: bool,
    acquisition_manager,
    model=None,
):
    if acquisition_manager is None:
        epoch_train_x, epoch_train_y, epoch_train_data_idx = trainer_impl._sample_supervised_rows_with_indices(
            dataset.train_x,
            dataset.train_y,
            config,
            curriculum,
            global_epoch=global_epoch,
            difficulty_bins=dataset.train_difficulty_bins,
            allow_row_subsampling=phase_allows_sampling,
        )
        return epoch_train_x, epoch_train_y, epoch_train_data_idx, None

    diagnostics = acquisition_manager.maybe_acquire(
        global_epoch=global_epoch,
        model=model,
        x=dataset.train_x,
        y=dataset.train_y,
    )
    active_row_indices = acquisition_manager.active_row_indices(device=dataset.train_x.device)
    active_train_x = dataset.train_x.index_select(0, active_row_indices)
    active_train_y = dataset.train_y.index_select(0, active_row_indices)
    epoch_train_x, epoch_train_y, local_idx = trainer_impl._sample_supervised_rows_with_indices(
        active_train_x,
        active_train_y,
        config,
        curriculum,
        global_epoch=global_epoch,
        difficulty_bins=None,
        allow_row_subsampling=phase_allows_sampling,
    )
    epoch_train_data_idx = active_row_indices.index_select(0, local_idx)
    return epoch_train_x, epoch_train_y, epoch_train_data_idx, diagnostics


def train_single_stage_pinn_loop(
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
    if resolved_stack.optimizer_phases is None:
        raise ValueError("Resolved optimizer phases are required for single-stage PINN mode.")
    optimizer_phases = resolved_stack.optimizer_phases
    base_weights = resolved_stack.base_loss_weights
    loss_weight_schedule = resolved_stack.loss_weight_schedule
    weighting_config = resolved_stack.weighting_config
    vrba_config = resolved_stack.vrba_config
    if loss_weight_schedule is not None and weighting_config.scheme != "static":
        raise ValueError("pinn.loss_weight_schedule currently supports static weighting only. Set pinn.weighting.scheme=static.")
    acquisition_settings = resolved_stack.supervised_acquisition
    if bool(acquisition_settings.enabled):
        if bool(resolved_stack.curriculum.enabled):
            raise ValueError(
                "pinn.supervised_acquisition.enabled=true cannot currently be combined with "
                "pinn.curriculum.enabled=true. Curriculum is still partially pre-refactor and both features "
                "control supervised row visibility."
            )
        if weighting_config.scheme != "static":
            raise ValueError(
                "pinn.supervised_acquisition.enabled=true currently supports pinn.weighting.scheme=static only "
                "to avoid scoring hidden candidate-pool labels through dynamic weighting probes."
            )
    weighting_policy = trainer_impl.build_weighting_policy(weighting_config)
    weighting_state = weighting_policy.initial_state(base_weights)
    gradient_telemetry_enabled = bool(trainer_impl.cfg_get(config, "pinn.gradient_telemetry.enabled", False))
    cost_tracking_enabled = trainer_impl._cost_tracking_enabled(config)
    criterion = trainer_impl.nn.MSELoss()
    transition_settings = trainer_impl._optimizer_transition_settings(config)
    transition_state = trainer_impl.OptimizerTransitionState(cache={})

    model = trainer_impl.TimeConditionedMLP(
        input_dim=dataset.input_dim,
        output_dim=dataset.output_dim,
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
        activation=activation,
    ).to(device=device, dtype=dtype)
    if bool(trainer_impl.cfg_get(config, "debug.print_architecture", False)):
        n_params = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("[pinn] Architecture:")
        print(model)
        print(
            "[pinn] Architecture summary | "
            f"input_dim={dataset.input_dim} output_dim={dataset.output_dim} "
            f"hidden_dim={hidden_dim} hidden_layers={hidden_layers} activation={activation} "
            f"dtype={dtype} device={device} parameters={n_params} trainable_parameters={n_trainable}"
        )

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
        val_init_x=dataset.val_init_x,
        val_init_y=dataset.val_init_y,
        test_x=dataset.test_x,
        test_y=dataset.test_y,
    )
    supervised_acquisition_manager = None
    if bool(acquisition_settings.enabled):
        supervised_acquisition_manager = trainer_impl._SupervisedAcquisitionManager(
            train_traj_ids=dataset.train_traj_ids,
            strategy=acquisition_settings.strategy,
            initial_trajectories=acquisition_settings.initial_trajectories,
            add_trajectories=acquisition_settings.add_trajectories,
            max_trajectories=acquisition_settings.max_trajectories,
            refresh_period_epochs=acquisition_settings.refresh_period_epochs,
            candidate_batch_size=acquisition_settings.candidate_batch_size,
            anchor_trajectories=acquisition_settings.anchor_trajectories,
            similarity_space=acquisition_settings.similarity_space,
            seed=acquisition_settings.seed,
        )
    initial_train_x, initial_train_y, initial_train_data_idx, initial_acquisition_diagnostics = _sample_epoch_supervised_rows(
        dataset=dataset,
        config=config,
        curriculum=resolved_stack.curriculum,
        global_epoch=1,
        phase_allows_sampling=False,
        acquisition_manager=supervised_acquisition_manager,
        model=model,
    )
    vrba_state = trainer_impl.initialize_vrba_state(
        vrba_config,
        initial_point_counts={
            "data": int(initial_train_x.shape[0]),
            "dt": int(initial_train_x.shape[0]),
            "physics": int(dataset.train_col_x.shape[0]),
            "ic": int(dataset.train_init_x.shape[0]),
        },
    )
    weight_probe_batch = trainer_impl._build_weight_update_probe_batch(
        dataset=dataset,
        weighting_config=weighting_config,
    )

    pinn_model = trainer_impl.PinnModel(
        model=model,
        device=device,
        dtype=dtype,
        input_dim=dataset.input_dim,
        output_dim=dataset.output_dim,
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
        activation=activation,
    )

    rows: list[Any] = []
    best_metric: float | None = None
    global_epoch = 0
    train_started = trainer_impl.time.perf_counter()
    total_epochs = int(sum(phase.epochs for phase in optimizer_phases))
    checkpoint_milestones = trainer_impl._resolve_checkpoint_milestones(config, total_epochs)
    train_dt_weights = dataset.train_dt_weights
    initial_weights = (
        trainer_impl._scheduled_loss_weights(
            base_weights=base_weights,
            schedule=loss_weight_schedule,
            next_global_epoch=1,
        )
        if weighting_config.scheme == "static"
        else weighting_policy.current_weights(weighting_state)
    )
    model.eval()
    initial_losses, _ = trainer_impl._measure_pinn_state(
        model=model,
        criterion=criterion,
        ode_model=ode_model,
        formulation=formulation,
        weights=initial_weights,
        x_data=initial_train_x,
        y_data=initial_train_y,
        x_data_dt_weights=trainer_impl._active_dt_weights_or_none(
            vrba_config=vrba_config,
            x_data_dt_weights=(
                None
                if train_dt_weights is None
                else trainer_impl._sample_rows_with_indices(train_dt_weights, initial_train_data_idx)
            ),
        ),
        x_col=dataset.train_col_x,
        x_col_weights=trainer_impl._active_collocation_weights_or_none(vrba_config=vrba_config, x_col_weights=dataset.train_col_weights),
        x_init=dataset.train_init_x,
        y_init=dataset.train_init_y,
        x_init_weights=trainer_impl._active_init_weights_or_none(vrba_config=vrba_config, x_init_weights=dataset.train_init_weights),
        capture_gradient_telemetry=False,
    )
    val_total_loss, val_component_losses, test_metrics = trainer_impl._evaluate_epoch_validation_and_test(
        model=model,
        dataset=dataset,
        criterion=criterion,
        ode_model=ode_model,
        formulation=formulation,
        active_weights=initial_weights,
        config=config,
        global_epoch=0,
        eval_init_x=dataset.val_init_x,
        eval_init_y=dataset.val_init_y,
    )
    regression_metrics = trainer_impl._evaluate_epoch_regression_metrics(
        model=model,
        active_train_x=initial_train_x,
        active_train_y=initial_train_y,
        dataset=dataset,
        config=config,
        global_epoch=0,
    )
    initial_train_component_losses = {
        name: float(value.detach().item())
        for name, value in initial_losses.components.items()
    }
    initial_row = trainer_impl._build_epoch_metrics_row(
        epoch=0,
        global_epoch=0,
        phase_name="init",
        optimizer_name="none",
        train_total_loss=float(initial_losses.total.detach().item()),
        train_component_losses=initial_train_component_losses,
        train_total_grad_norm=None,
        train_component_grad_norms=None,
        train_weighted_component_grad_norms=None,
        val_total_loss=val_total_loss,
        val_component_losses=val_component_losses,
        test_metrics=test_metrics,
        regression_metrics=regression_metrics,
        weighting_scheme=weighting_config.scheme,
        weighting_updated=False,
        train_loss_weights=initial_weights.as_dict(),
        weighting_raw_candidate_weights=dict(weighting_state.raw_candidate_weights),
        weighting_probe_grad_l2_norms=None,
        weighting_probe_grad_mean_abs=None,
        weighting_probe_grad_max_abs=None,
        weighting_probe_grad_std=None,
        weighting_probe_ntk_mean_trace=None,
        weighting_probe_ntk_batch_sizes=None,
        weighting_anchor=weighting_config.anchor,
        vrba_summary=trainer_impl._collocation_vrba_summary(collocation_manager),
        vrba_details=trainer_impl._collocation_vrba_details(collocation_manager),
        epoch_wall_seconds=None,
        cumulative_wall_seconds=None,
        num_batches=0 if cost_tracking_enabled else None,
        num_train_steps=0 if cost_tracking_enabled else None,
        num_supervised_rows=int(initial_train_x.shape[0]) if cost_tracking_enabled else None,
        num_collocation_rows=int(dataset.train_col_x.shape[0]) if cost_tracking_enabled else None,
        num_init_rows=int(dataset.train_init_x.shape[0]) if cost_tracking_enabled else None,
        peak_gpu_memory_allocated_bytes=None,
        peak_gpu_memory_reserved_bytes=None,
        optimizer_diagnostics={"initial_evaluation": True},
        supervised_acquisition=initial_acquisition_diagnostics,
    )
    rows.append(initial_row)
    if logger is not None:
        logger.print_epoch_metrics(initial_row)
        logger.log_epoch_metrics(initial_row)

    if logger is not None and trainer_impl._checkpointing_enabled(config, "save_init", True):
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

    for phase_idx, phase in enumerate(optimizer_phases):
        optimizer_spec = trainer_impl.build_optimizer(
            model=model,
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
        phase_supports_dynamic_weight_updates = trainer_impl._phase_supports_dynamic_weight_updates(phase)
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
                    model=model,
                    ode_model=ode_model,
                    formulation=formulation,
                )
            )

        for epoch in range(1, phase.epochs + 1):
            epoch_started = trainer_impl.time.perf_counter()
            trainer_impl._reset_peak_memory_if_cuda(device, enabled=cost_tracking_enabled)
            model.train()
            epoch_losses: list[Any] = []
            epoch_gradients: list[Any] = []
            capture_gradient_telemetry = gradient_telemetry_enabled
            step_weight_updates_enabled = bool(
                phase_supports_dynamic_weight_updates
                and weighting_config.uses_step_updates
                and weighting_config.use_live_batch
            )
            weighting_updated = False
            weight_update_stats = None
            scheduled_base_weights = trainer_impl._scheduled_loss_weights(
                base_weights=base_weights,
                schedule=loss_weight_schedule,
                next_global_epoch=global_epoch + 1,
            )
            active_weights = scheduled_base_weights if weighting_config.scheme == "static" else weighting_policy.current_weights(weighting_state)
            epoch_train_x, epoch_train_y, epoch_train_data_idx, supervised_acquisition_diagnostics = _sample_epoch_supervised_rows(
                dataset=dataset,
                config=config,
                curriculum=resolved_stack.curriculum,
                global_epoch=global_epoch + 1,
                phase_allows_sampling=phase_allows_sampling,
                acquisition_manager=supervised_acquisition_manager,
                model=model,
            )
            epoch_constraints = trainer_impl._prepare_epoch_constraint_batch(
                collocation_manager=collocation_manager,
                vrba_config=vrba_config,
                train_dt_weights=train_dt_weights,
                model=model,
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
                    model=model,
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
                    capture_gradient_telemetry=capture_gradient_telemetry,
                )
                epoch_losses.append(breakdown)
                num_batches = 1
                num_train_steps = 1
                num_supervised_rows = int(epoch_train_x.shape[0])
                num_collocation_rows = int(epoch_train_col_x.shape[0])
                num_init_rows = int(epoch_train_init_x.shape[0])
                if gradient_telemetry is not None:
                    epoch_gradients.append(gradient_telemetry)
                if step_weight_updates_enabled:
                    weighting_state, weight_update_stats = trainer_impl._apply_single_stage_live_weight_update(
                        weighting_policy=weighting_policy,
                        weighting_state=weighting_state,
                        model=model,
                        criterion=criterion,
                        ode_model=ode_model,
                        formulation=formulation,
                        active_weights=active_weights,
                        batch=trainer_impl.PinnBatch(
                            x_data=epoch_train_x,
                            y_data=epoch_train_y,
                            x_data_dt_weights=active_dt_weights,
                            x_col=epoch_train_col_x,
                            x_col_weights=active_collocation_weights,
                            x_init=epoch_train_init_x,
                            y_init=epoch_train_init_y,
                            x_init_weights=active_init_weights,
                        ),
                        weighting_config=weighting_config,
                        epoch=epoch,
                        global_epoch=global_epoch + 1,
                    )
                    weighting_updated = True
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
                    seed=seed + phase_idx + global_epoch,
                )
                for batch in epoch_batches:
                    active_weights = scheduled_base_weights if weighting_config.scheme == "static" else weighting_policy.current_weights(weighting_state)
                    num_batches += 1
                    num_train_steps += 1
                    num_supervised_rows += int(batch.x_data.shape[0])
                    num_collocation_rows += int(batch.x_col.shape[0])
                    num_init_rows += int(batch.x_init.shape[0])
                    breakdown, gradient_telemetry = trainer_impl._train_pinn_step(
                        model=model,
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
                        capture_gradient_telemetry=capture_gradient_telemetry,
                    )
                    epoch_losses.append(breakdown)
                    if gradient_telemetry is not None:
                        epoch_gradients.append(gradient_telemetry)
                    if step_weight_updates_enabled:
                        weighting_state, weight_update_stats = trainer_impl._apply_single_stage_live_weight_update(
                            weighting_policy=weighting_policy,
                            weighting_state=weighting_state,
                            model=model,
                            criterion=criterion,
                            ode_model=ode_model,
                            formulation=formulation,
                            active_weights=active_weights,
                            batch=batch,
                            weighting_config=weighting_config,
                            epoch=epoch,
                            global_epoch=global_epoch + 1,
                        )
                        weighting_updated = True

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
            weighting_state, maybe_epoch_weight_update_stats, epoch_weighting_updated = trainer_impl._maybe_apply_single_stage_epoch_weight_update(
                weighting_policy=weighting_policy,
                weighting_state=weighting_state,
                weighting_config=weighting_config,
                dataset=dataset,
                weight_probe_batch=weight_probe_batch,
                model=model,
                criterion=criterion,
                ode_model=ode_model,
                formulation=formulation,
                active_weights=active_weights,
                train_component_losses=train_component_losses,
                epoch=epoch,
                global_epoch=global_epoch,
                phase_supports_dynamic_weight_updates=phase_supports_dynamic_weight_updates,
                step_weight_updates_enabled=step_weight_updates_enabled,
            )
            if maybe_epoch_weight_update_stats is not None:
                weight_update_stats = maybe_epoch_weight_update_stats
            if epoch_weighting_updated:
                weighting_updated = True
            active_weights = scheduled_base_weights if weighting_config.scheme == "static" else weighting_policy.current_weights(weighting_state)
            val_total_loss, val_component_losses, test_metrics = trainer_impl._evaluate_epoch_validation_and_test(
                model=model,
                dataset=dataset,
                criterion=criterion,
                ode_model=ode_model,
                formulation=formulation,
                active_weights=active_weights,
                config=config,
                global_epoch=global_epoch,
                eval_init_x=dataset.val_init_x,
                eval_init_y=dataset.val_init_y,
            )
            regression_metrics = trainer_impl._evaluate_epoch_regression_metrics(
                model=model,
                active_train_x=epoch_train_x,
                active_train_y=epoch_train_y,
                dataset=dataset,
                config=config,
                global_epoch=global_epoch,
            )
            scheduler_metric_value, scheduler_metric_name = trainer_impl._scheduler_metric_value(
                scheduler_config=phase.scheduler,
                train_total_loss=train_total,
                val_total_loss=val_total_loss,
            )
            if scheduler is not None and scheduler_metric_value is not None:
                scheduler.step(float(scheduler_metric_value))
            vrba_summary = trainer_impl._collocation_vrba_summary(collocation_manager)
            vrba_details = trainer_impl._collocation_vrba_details(collocation_manager)
            row = trainer_impl._build_epoch_metrics_row(
                epoch=epoch,
                global_epoch=global_epoch,
                phase_name=phase.name,
                optimizer_name=phase.optimizer,
                train_total_loss=train_total,
                train_component_losses=train_component_losses,
                train_total_grad_norm=train_total_grad_norm,
                train_component_grad_norms=train_component_grad_norms,
                train_weighted_component_grad_norms=train_weighted_component_grad_norms,
                val_total_loss=val_total_loss,
                val_component_losses=val_component_losses,
                test_metrics=test_metrics,
                regression_metrics=regression_metrics,
                weighting_scheme=weighting_config.scheme,
                weighting_updated=weighting_updated,
                train_loss_weights=active_weights.as_dict(),
                weighting_raw_candidate_weights=dict(weighting_state.raw_candidate_weights),
                weighting_probe_grad_l2_norms=None if weight_update_stats is None else dict(weight_update_stats.grad_l2_norms),
                weighting_probe_grad_mean_abs=None if weight_update_stats is None else dict(weight_update_stats.grad_mean_abs),
                weighting_probe_grad_max_abs=None if weight_update_stats is None else dict(weight_update_stats.grad_max_abs),
                weighting_probe_grad_std=None if weight_update_stats is None else dict(weight_update_stats.grad_std),
                weighting_probe_ntk_mean_trace=None if weight_update_stats is None else None if weight_update_stats.ntk_mean_trace is None else dict(weight_update_stats.ntk_mean_trace),
                weighting_probe_ntk_batch_sizes=None if weight_update_stats is None else None if weight_update_stats.ntk_batch_sizes is None else dict(weight_update_stats.ntk_batch_sizes),
                weighting_anchor=weighting_config.anchor,
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
                optimizer_diagnostics={
                    **trainer_impl._build_single_stage_optimizer_diagnostics(
                        optimizer_spec=optimizer_spec,
                        phase_full_batch=phase_full_batch,
                        phase_allows_sampling=phase_allows_sampling,
                        phase_supports_dynamic_weight_updates=phase_supports_dynamic_weight_updates,
                        step_weight_updates_enabled=step_weight_updates_enabled,
                        weighting_policy=weighting_policy,
                        weighting_state=weighting_state,
                        scheduler=scheduler,
                        scheduler_metric_name=scheduler_metric_name,
                        scheduler_metric_value=scheduler_metric_value,
                        optimizer=optimizer_spec.optimizer,
                        collocation_manager=collocation_manager,
                    ),
                    **transition_diagnostics,
                    **(
                        optimizer_spec.optimizer.get_last_diagnostics()
                        if hasattr(optimizer_spec.optimizer, "get_last_diagnostics")
                        else {}
                    ),
                },
                supervised_acquisition=supervised_acquisition_diagnostics,
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
                validation_available=dataset.val_x is not None and dataset.val_y is not None,
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
                    model=model,
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

    model.eval()
    return pinn_model, rows
