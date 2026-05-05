"""Numerical PINN runtime helpers used by trainer loop modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.pinn.data import PinnDatasetBundle
from src.pinn.evaluator import evaluate_pinn_loss_breakdown, evaluate_pinn_weighting_terms
from src.pinn.losses import LOSS_COMPONENTS, LossWeights, PinnLossBreakdown
from src.pinn.optim import OptimizerSpec
from src.pinn.residuals import compute_residual_terms, compute_supervised_dt_terms, prepare_input_for_autograd
from src.pinn.runtime import OptimizerPhase, ResolvedLossWeightScheduleStage, SchedulerConfig
from src.pinn.weighting import WeightUpdateStats, WeightingConfig
from src.training.runtime import cfg_get


@dataclass(frozen=True)
class PinnBatch:
    x_data: torch.Tensor
    y_data: torch.Tensor
    x_data_dt_weights: torch.Tensor | None
    x_col: torch.Tensor
    x_col_weights: torch.Tensor | None
    x_init: torch.Tensor
    y_init: torch.Tensor
    x_init_weights: torch.Tensor | None


@dataclass(frozen=True)
class GradientTelemetry:
    total_grad_norm: float
    component_grad_norms: dict[str, float]
    weighted_component_grad_norms: dict[str, float]

    def component(self, name: str) -> float:
        return float(self.component_grad_norms[name])

    def weighted_component(self, name: str) -> float:
        return float(self.weighted_component_grad_norms[name])

    @property
    def data_grad_norm(self) -> float:
        return self.component("data")

    @property
    def dt_grad_norm(self) -> float:
        return self.component("dt")

    @property
    def physics_grad_norm(self) -> float:
        return self.component("physics")

    @property
    def ic_grad_norm(self) -> float:
        return self.component("ic")


def scheduled_loss_weights(
    *,
    base_weights: LossWeights,
    schedule: tuple[ResolvedLossWeightScheduleStage, ...] | None,
    next_global_epoch: int,
) -> LossWeights:
    if not schedule:
        return base_weights
    cursor = 0
    for stage in schedule:
        if stage.epochs is None:
            return stage.weights
        cursor += int(stage.epochs)
        if int(next_global_epoch) <= cursor:
            return stage.weights
    return schedule[-1].weights


def should_run_evaluation(global_epoch: int, config: Any) -> bool:
    if int(global_epoch) in (0, 1):
        return True
    frequency = int(cfg_get(config, "pinn.evaluation.frequency", 1))
    if frequency < 1:
        frequency = 1
    return (int(global_epoch) % frequency) == 0


def model_device_and_dtype(model: nn.Module) -> tuple[torch.device, torch.dtype]:
    param = next(model.parameters())
    return param.device, param.dtype


def to_model_tensor(tensor: torch.Tensor | None, model: nn.Module) -> torch.Tensor | None:
    if tensor is None:
        return None
    device, dtype = model_device_and_dtype(model)
    return tensor.to(device=device, dtype=dtype)


def evaluate_data_loss(
    model: nn.Module,
    x: torch.Tensor | None,
    y: torch.Tensor | None,
    criterion: nn.Module,
) -> float | None:
    if x is None or y is None:
        return None
    x_eval = to_model_tensor(x, model)
    y_eval = to_model_tensor(y, model)
    model.eval()
    with torch.no_grad():
        pred = model(x_eval)
        return float(criterion(pred, y_eval).item())


def evaluate_physics_loss(
    model: nn.Module,
    x_rows: torch.Tensor | None,
    ode_model: Any,
    criterion: nn.Module,
    formulation: str,
) -> float | None:
    if x_rows is None:
        return None
    x_eval = to_model_tensor(x_rows, model)
    model.eval()
    terms = compute_residual_terms(
        model=model,
        x=x_eval,
        ode_model=ode_model,
        formulation=formulation,
        create_graph=False,
    )
    return float(criterion(terms.residual, torch.zeros_like(terms.residual)).item())


def evaluate_dt_loss(
    model: nn.Module,
    x: torch.Tensor | None,
    y: torch.Tensor | None,
    ode_model: Any,
    criterion: nn.Module,
    formulation: str,
) -> float | None:
    if x is None or y is None:
        return None
    x_eval = to_model_tensor(x, model)
    y_eval = to_model_tensor(y, model)
    model.eval()
    terms = compute_supervised_dt_terms(
        model=model,
        x=x_eval,
        y_true=y_eval,
        ode_model=ode_model,
        formulation=formulation,
        create_graph=False,
    )
    return float(criterion(terms.residual, torch.zeros_like(terms.residual)).item())


def evaluate_data_and_dt_loss(
    model: nn.Module,
    x: torch.Tensor | None,
    y: torch.Tensor | None,
    ode_model: Any,
    criterion: nn.Module,
    formulation: str,
) -> tuple[float | None, float | None]:
    """Compute data loss and dt loss with a single shared forward pass on the same inputs."""
    if x is None or y is None:
        return None, None
    x_eval = to_model_tensor(x, model)
    y_eval = to_model_tensor(y, model)
    model.eval()
    x_req = prepare_input_for_autograd(x_eval)
    pred = model(x_req)
    data_loss = float(criterion(pred, y_eval).item())
    terms = compute_supervised_dt_terms(
        model=model,
        x=x_req,
        y_true=y_eval,
        ode_model=ode_model,
        formulation=formulation,
        create_graph=False,
        prediction=pred,
    )
    dt_loss = float(criterion(terms.residual, torch.zeros_like(terms.residual)).item())
    return data_loss, dt_loss


def compute_weighted_total_loss(
    component_losses: dict[str, float | None] | None,
    weights: LossWeights,
) -> float | None:
    if component_losses is None:
        return None
    total = 0.0
    used_any = False
    for name, weight in weights.items():
        value = component_losses.get(name)
        if value is None:
            continue
        total += float(weight) * float(value)
        used_any = True
    if not used_any:
        return None
    return float(total)


def _subsample_val_col(val_col_x: torch.Tensor, *, config: Any) -> torch.Tensor:
    """Subsample val_col_x to match the training active_points budget.

    Uses the same deterministic randperm technique as the collocation factory so
    that the validation physics loss is computed on the same number of points as
    training, making the two losses directly comparable.
    """
    active_points = int(cfg_get(config, "pinn.collocation.active_points", val_col_x.shape[0]))
    n_rows = int(val_col_x.shape[0])
    if n_rows <= active_points:
        return val_col_x
    seed = int(cfg_get(config, "pinn.collocation.seed", cfg_get(config, "model.seed", 0)))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    indices = torch.randperm(n_rows, generator=generator, device="cpu")[:active_points].to(val_col_x.device)
    return val_col_x.index_select(0, indices)


def evaluate_epoch_validation_and_test(
    *,
    model: nn.Module,
    dataset: PinnDatasetBundle,
    criterion: nn.Module,
    ode_model: Any,
    formulation: str,
    active_weights: LossWeights,
    config: Any,
    global_epoch: int,
    eval_init_x: torch.Tensor | None = None,
    eval_init_y: torch.Tensor | None = None,
) -> tuple[float | None, dict[str, float] | None, dict[str, float] | None]:
    val_total_loss = None
    val_component_losses = None
    test_metrics = None
    if should_run_evaluation(global_epoch, config):
        val_component_losses = {}
        if dataset.val_x is not None and dataset.val_y is not None:
            val_component_losses["data"], val_component_losses["dt"] = evaluate_data_and_dt_loss(
                model=model,
                x=dataset.val_x,
                y=dataset.val_y,
                ode_model=ode_model,
                criterion=criterion,
                formulation=formulation,
            )
            val_col_x = dataset.val_col_x if dataset.val_col_x is not None else dataset.val_x
            val_col_x = _subsample_val_col(val_col_x, config=config)
            val_component_losses["physics"] = evaluate_physics_loss(
                model=model,
                x_rows=val_col_x,
                ode_model=ode_model,
                criterion=criterion,
                formulation=formulation,
            )
            if eval_init_x is not None and eval_init_y is not None:
                val_component_losses["ic"] = evaluate_data_loss(
                    model=model,
                    x=eval_init_x,
                    y=eval_init_y,
                    criterion=criterion,
                )
            val_total_loss = compute_weighted_total_loss(val_component_losses, active_weights)
        if dataset.test_x is not None and dataset.test_y is not None:
            test_metrics = {
                "data_loss": evaluate_data_loss(
                    model=model, x=dataset.test_x, y=dataset.test_y, criterion=criterion
                )
            }
    return val_total_loss, val_component_losses, test_metrics


def ceil_div(num: int, den: int) -> int:
    return (int(num) + int(den) - 1) // int(den)


def epoch_component_index_chunks(
    *,
    size: int,
    steps_per_epoch: int,
    shuffle: bool,
    seed: int,
) -> list[torch.Tensor]:
    if size <= 0:
        raise ValueError("Component size must be > 0.")
    if steps_per_epoch <= 0:
        raise ValueError("steps_per_epoch must be > 0.")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    if shuffle:
        base_indices = torch.randperm(int(size), generator=generator, device="cpu")
    else:
        base_indices = torch.arange(int(size), dtype=torch.long)

    if int(size) >= int(steps_per_epoch):
        chunks: list[torch.Tensor] = []
        for step_idx in range(int(steps_per_epoch)):
            start = (step_idx * int(size)) // int(steps_per_epoch)
            end = ((step_idx + 1) * int(size)) // int(steps_per_epoch)
            chunk = base_indices[start:end]
            if int(chunk.numel()) <= 0:
                raise RuntimeError("Balanced minibatch planning produced an empty chunk unexpectedly.")
            chunks.append(chunk)
        return chunks

    repeats = ceil_div(int(steps_per_epoch), int(size))
    tiled = base_indices.repeat(repeats)[: int(steps_per_epoch)]
    return [tiled[idx : idx + 1] for idx in range(int(steps_per_epoch))]


def sample_rows_with_indices(
    x: torch.Tensor,
    indices: torch.Tensor,
    y: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
    if y is None:
        return x.index_select(0, indices)
    return x.index_select(0, indices), y.index_select(0, indices)


def iter_minibatch_batches(
    *,
    dataset: PinnDatasetBundle,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> tuple[int, Iterator[PinnBatch]]:
    steps_per_epoch = max(
        ceil_div(dataset.train_x.shape[0], batch_size),
        ceil_div(dataset.train_col_x.shape[0], batch_size),
        ceil_div(dataset.train_init_x.shape[0], batch_size),
    )
    data_chunks = epoch_component_index_chunks(
        size=int(dataset.train_x.shape[0]),
        steps_per_epoch=steps_per_epoch,
        shuffle=shuffle,
        seed=seed,
    )
    col_chunks = epoch_component_index_chunks(
        size=int(dataset.train_col_x.shape[0]),
        steps_per_epoch=steps_per_epoch,
        shuffle=shuffle,
        seed=seed + 1,
    )
    init_chunks = epoch_component_index_chunks(
        size=int(dataset.train_init_x.shape[0]),
        steps_per_epoch=steps_per_epoch,
        shuffle=shuffle,
        seed=seed + 2,
    )

    def batch_iterator():
        for step_idx in range(steps_per_epoch):
            data_idx = data_chunks[step_idx].to(device=dataset.train_x.device)
            col_idx = col_chunks[step_idx].to(device=dataset.train_col_x.device)
            init_idx = init_chunks[step_idx].to(device=dataset.train_init_x.device)
            x_data, y_data = sample_rows_with_indices(dataset.train_x, data_idx, dataset.train_y)
            x_data_dt_weights = (
                None if dataset.train_dt_weights is None else sample_rows_with_indices(dataset.train_dt_weights, data_idx)
            )
            x_init, y_init = sample_rows_with_indices(dataset.train_init_x, init_idx, dataset.train_init_y)
            x_col = sample_rows_with_indices(dataset.train_col_x, col_idx)
            x_col_weights = (
                None if dataset.train_col_weights is None else sample_rows_with_indices(dataset.train_col_weights, col_idx)
            )
            x_init_weights = (
                None if dataset.train_init_weights is None else sample_rows_with_indices(dataset.train_init_weights, init_idx)
            )
            yield PinnBatch(
                x_data=x_data,
                y_data=y_data,
                x_data_dt_weights=x_data_dt_weights,
                x_col=x_col,
                x_col_weights=x_col_weights,
                x_init=x_init,
                y_init=y_init,
                x_init_weights=x_init_weights,
            )

    return steps_per_epoch, batch_iterator()


def grad_norm(loss: torch.Tensor, model: nn.Module, retain_graph: bool) -> float:
    params = [param for param in model.parameters() if param.requires_grad]
    grads = torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=True)
    sq_norm = torch.zeros((), dtype=loss.dtype, device=loss.device)
    for grad in grads:
        if grad is None:
            continue
        sq_norm = sq_norm + grad.pow(2).sum()
    return float(torch.sqrt(sq_norm).detach().cpu().item())


def flattened_grads(loss: torch.Tensor, model: nn.Module, retain_graph: bool) -> torch.Tensor:
    params = [param for param in model.parameters() if param.requires_grad]
    grads = torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=True)
    flat_parts = []
    for param, grad in zip(params, grads):
        if grad is None:
            flat_parts.append(torch.zeros(param.numel(), dtype=loss.dtype, device=loss.device))
        else:
            flat_parts.append(grad.reshape(-1))
    if not flat_parts:
        return torch.zeros(0, dtype=loss.dtype, device=loss.device)
    return torch.cat(flat_parts, dim=0)


def compute_output_jacobian(output: torch.Tensor, model: nn.Module) -> torch.Tensor:
    output_flat = output.reshape(-1)
    if int(output_flat.numel()) == 0:
        return torch.zeros((0, 0), dtype=output.dtype, device=output.device)
    params = [param for param in model.parameters() if param.requires_grad]
    eye = torch.eye(int(output_flat.numel()), dtype=output.dtype, device=output.device)
    grads = torch.autograd.grad(
        output_flat,
        params,
        grad_outputs=eye,
        is_grads_batched=True,
        retain_graph=True,
        allow_unused=True,
    )
    jacobian_parts = []
    for param, grad in zip(params, grads):
        if grad is None:
            jacobian_parts.append(
                torch.zeros((int(output_flat.numel()), param.numel()), dtype=output.dtype, device=output.device)
            )
            continue
        jacobian_parts.append(grad.reshape(int(output_flat.numel()), -1))
    if not jacobian_parts:
        return torch.zeros((int(output_flat.numel()), 0), dtype=output.dtype, device=output.device)
    return torch.cat(jacobian_parts, dim=1)


def sample_ntk_term_rows(term: torch.Tensor, *, target_rows: int, seed: int) -> torch.Tensor:
    total_rows = int(term.shape[0])
    if target_rows >= total_rows:
        return term
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    indices = torch.randperm(total_rows, generator=generator, device="cpu")[:target_rows].to(device=term.device)
    return term.index_select(0, indices)


def compute_ntk_mean_trace(output: torch.Tensor, model: nn.Module) -> float:
    num_points = max(int(output.shape[0]), 1)
    jacobian = compute_output_jacobian(output, model)
    trace = torch.einsum("na,na->", jacobian, jacobian)
    return float((trace / float(num_points)).detach().cpu().item())


def compute_gradient_telemetry(
    *,
    model: nn.Module,
    losses: PinnLossBreakdown,
    weights: LossWeights,
) -> GradientTelemetry:
    component_grad_norms = {
        name: grad_norm(losses.component(name), model, retain_graph=True)
        for name in losses.components
    }
    weighted_component_grad_norms = {
        name: grad_norm(float(weights.get(name)) * losses.component(name), model, retain_graph=True)
        for name in losses.components
    }
    return GradientTelemetry(
        total_grad_norm=grad_norm(losses.total, model, retain_graph=True),
        component_grad_norms=component_grad_norms,
        weighted_component_grad_norms=weighted_component_grad_norms,
    )


def compute_weight_update_stats(
    *,
    model: nn.Module,
    losses: PinnLossBreakdown,
    anchor_component: str | None,
    epoch: int,
    global_epoch: int,
) -> WeightUpdateStats:
    grad_vectors = {
        name: flattened_grads(losses.component(name), model, retain_graph=True)
        for name in losses.components
    }
    grad_l2_norms = {
        name: float(torch.linalg.vector_norm(grad_vector, ord=2).detach().cpu().item())
        for name, grad_vector in grad_vectors.items()
    }
    grad_mean_abs = {
        name: float(grad_vector.abs().mean().detach().cpu().item())
        for name, grad_vector in grad_vectors.items()
    }
    grad_max_abs = {
        name: float(grad_vector.abs().max().detach().cpu().item())
        for name, grad_vector in grad_vectors.items()
    }
    grad_std = {}
    for name, grad_vector in grad_vectors.items():
        if int(grad_vector.numel()) <= 1:
            grad_std[name] = 0.0
            continue
        grad_std[name] = float(grad_vector.std(unbiased=True).detach().cpu().item())
    return WeightUpdateStats(
        grad_l2_norms=grad_l2_norms,
        grad_mean_abs=grad_mean_abs,
        grad_max_abs=grad_max_abs,
        grad_std=grad_std,
        component_losses={
            name: float(losses.component(name).detach().cpu().item())
            for name in losses.components
        },
        anchor_component=anchor_component,
        epoch=int(epoch),
        global_epoch=int(global_epoch),
    )


def compute_ntk_weight_update_stats(
    *,
    model: nn.Module,
    losses: PinnLossBreakdown,
    weighting_terms: dict[str, torch.Tensor],
    weighting_config: WeightingConfig,
    epoch: int,
    global_epoch: int,
) -> WeightUpdateStats:
    zero_stats = {name: 0.0 for name in LOSS_COMPONENTS}
    ntk_batch_sizes = weighting_config.ntk_batch_sizes.as_dict()
    ntk_mean_trace: dict[str, float] = {}
    for offset, name in enumerate(weighting_config.dynamic_components):
        sampled_term = sample_ntk_term_rows(
            weighting_terms[name],
            target_rows=int(ntk_batch_sizes[name]),
            seed=int(weighting_config.ntk_seed) + int(global_epoch) * 97 + offset,
        )
        ntk_mean_trace[name] = compute_ntk_mean_trace(sampled_term, model)
    return WeightUpdateStats(
        grad_l2_norms=dict(zero_stats),
        grad_mean_abs=dict(zero_stats),
        grad_max_abs=dict(zero_stats),
        grad_std=dict(zero_stats),
        component_losses={
            name: float(losses.component(name).detach().cpu().item())
            for name in losses.components
        },
        anchor_component=None,
        epoch=int(epoch),
        global_epoch=int(global_epoch),
        ntk_mean_trace=ntk_mean_trace,
        ntk_batch_sizes={name: int(ntk_batch_sizes[name]) for name in weighting_config.dynamic_components},
    )


def compute_live_batch_weight_update_stats(
    *,
    model: nn.Module,
    criterion: nn.Module,
    ode_model: Any,
    formulation: str,
    weights: LossWeights,
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    x_data_dt_weights: torch.Tensor | None,
    x_col: torch.Tensor,
    x_col_weights: torch.Tensor | None,
    x_init: torch.Tensor,
    y_init: torch.Tensor,
    x_init_weights: torch.Tensor | None,
    anchor_component: str | None,
    epoch: int,
    global_epoch: int,
) -> WeightUpdateStats:
    probe_losses = evaluate_pinn_loss_breakdown(
        model=model,
        criterion=criterion,
        ode_model=ode_model,
        formulation=formulation,
        weights=weights,
        x_data=x_data,
        y_data=y_data,
        x_data_dt_weights=x_data_dt_weights,
        x_col=x_col,
        x_col_weights=x_col_weights,
        x_init=x_init,
        y_init=y_init,
        x_init_weights=x_init_weights,
        create_graph=True,
    )
    return compute_weight_update_stats(
        model=model,
        losses=probe_losses,
        anchor_component=anchor_component,
        epoch=epoch,
        global_epoch=global_epoch,
    )


def apply_single_stage_live_weight_update(
    *,
    weighting_policy: Any,
    weighting_state: Any,
    model: nn.Module,
    criterion: nn.Module,
    ode_model: Any,
    formulation: str,
    active_weights: LossWeights,
    batch: PinnBatch,
    weighting_config: WeightingConfig,
    epoch: int,
    global_epoch: int,
) -> tuple[Any, WeightUpdateStats]:
    weight_update_stats = compute_live_batch_weight_update_stats(
        model=model,
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
        anchor_component=weighting_config.anchor,
        epoch=epoch,
        global_epoch=global_epoch,
    )
    next_weighting_state = weighting_policy.update(weighting_state, weight_update_stats)
    return next_weighting_state, weight_update_stats


def maybe_apply_single_stage_epoch_weight_update(
    *,
    weighting_policy: Any,
    weighting_state: Any,
    weighting_config: WeightingConfig,
    dataset: PinnDatasetBundle,
    weight_probe_batch: PinnBatch,
    model: nn.Module,
    criterion: nn.Module,
    ode_model: Any,
    formulation: str,
    active_weights: LossWeights,
    train_component_losses: dict[str, float],
    epoch: int,
    global_epoch: int,
    phase_supports_dynamic_weight_updates: bool,
    step_weight_updates_enabled: bool,
) -> tuple[Any, WeightUpdateStats | None, bool]:
    from src.training import trainer as trainer_impl

    if (
        step_weight_updates_enabled
        or not phase_supports_dynamic_weight_updates
        or not weighting_policy.should_update(epoch=epoch, global_epoch=global_epoch)
    ):
        return weighting_state, None, False

    if weighting_config.scheme == "relobralo":
        zero_stats = {name: 0.0 for name in LOSS_COMPONENTS}
        weight_update_stats = WeightUpdateStats(
            grad_l2_norms=dict(zero_stats),
            grad_mean_abs=dict(zero_stats),
            grad_max_abs=dict(zero_stats),
            grad_std=dict(zero_stats),
            component_losses=dict(train_component_losses),
            anchor_component=weighting_config.anchor,
            epoch=epoch,
            global_epoch=global_epoch,
        )
    elif weighting_config.scheme == "ntk_random_batch":
        probe_batch = (
            trainer_impl._build_weight_update_probe_batch(
                dataset=dataset,
                weighting_config=weighting_config,
                seed_offset=int(global_epoch) * 17,
            )
            if weighting_config.ntk_refresh_each_update
            else weight_probe_batch
        )
        probe_losses = evaluate_pinn_loss_breakdown(
            model=model,
            criterion=criterion,
            ode_model=ode_model,
            formulation=formulation,
            weights=active_weights,
            x_data=probe_batch.x_data,
            y_data=probe_batch.y_data,
            x_data_dt_weights=probe_batch.x_data_dt_weights,
            x_col=probe_batch.x_col,
            x_col_weights=None,
            x_init=probe_batch.x_init,
            y_init=probe_batch.y_init,
            x_init_weights=None,
            create_graph=True,
        )
        weighting_terms = evaluate_pinn_weighting_terms(
            model=model,
            ode_model=ode_model,
            formulation=formulation,
            x_data=probe_batch.x_data,
            y_data=probe_batch.y_data,
            x_col=probe_batch.x_col,
            x_init=probe_batch.x_init,
            y_init=probe_batch.y_init,
            create_graph=True,
        )
        weight_update_stats = compute_ntk_weight_update_stats(
            model=model,
            losses=probe_losses,
            weighting_terms=weighting_terms.components,
            weighting_config=weighting_config,
            epoch=epoch,
            global_epoch=global_epoch,
        )
    else:
        probe_losses = evaluate_pinn_loss_breakdown(
            model=model,
            criterion=criterion,
            ode_model=ode_model,
            formulation=formulation,
            weights=active_weights,
            x_data=weight_probe_batch.x_data,
            y_data=weight_probe_batch.y_data,
            x_data_dt_weights=weight_probe_batch.x_data_dt_weights,
            x_col=weight_probe_batch.x_col,
            x_col_weights=None,
            x_init=weight_probe_batch.x_init,
            y_init=weight_probe_batch.y_init,
            x_init_weights=None,
            create_graph=True,
        )
        weight_update_stats = compute_weight_update_stats(
            model=model,
            losses=probe_losses,
            anchor_component=weighting_config.anchor,
            epoch=epoch,
            global_epoch=global_epoch,
        )

    next_weighting_state = weighting_policy.update(weighting_state, weight_update_stats)
    return next_weighting_state, weight_update_stats, True


def measure_pinn_state(
    *,
    model: nn.Module,
    criterion: nn.Module,
    ode_model: Any,
    formulation: str,
    weights: LossWeights,
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    x_data_dt_weights: torch.Tensor | None,
    x_col: torch.Tensor,
    x_col_weights: torch.Tensor | None,
    x_init: torch.Tensor,
    y_init: torch.Tensor,
    x_init_weights: torch.Tensor | None,
    capture_gradient_telemetry: bool = False,
) -> tuple[PinnLossBreakdown, GradientTelemetry | None]:
    losses = evaluate_pinn_loss_breakdown(
        model=model,
        criterion=criterion,
        ode_model=ode_model,
        formulation=formulation,
        weights=weights,
        x_data=x_data,
        y_data=y_data,
        x_data_dt_weights=x_data_dt_weights,
        x_col=x_col,
        x_col_weights=x_col_weights,
        x_init=x_init,
        y_init=y_init,
        x_init_weights=x_init_weights,
        create_graph=capture_gradient_telemetry,
    )
    telemetry = None
    if capture_gradient_telemetry:
        telemetry = compute_gradient_telemetry(
            model=model,
            losses=losses,
            weights=weights,
        )
    return losses, telemetry


def train_pinn_step(
    *,
    model: nn.Module,
    optimizer_spec: OptimizerSpec,
    criterion: nn.Module,
    ode_model: Any,
    formulation: str,
    weights: LossWeights,
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    x_data_dt_weights: torch.Tensor | None,
    x_col: torch.Tensor,
    x_col_weights: torch.Tensor | None,
    x_init: torch.Tensor,
    y_init: torch.Tensor,
    x_init_weights: torch.Tensor | None,
    capture_gradient_telemetry: bool = False,
    objective_scale: float = 1.0,
) -> tuple[PinnLossBreakdown, GradientTelemetry | None]:
    breakdown_box: dict[str, PinnLossBreakdown] = {}
    telemetry_box: dict[str, GradientTelemetry | None] = {"telemetry": None}
    optimizer = optimizer_spec.optimizer
    scale = max(float(objective_scale), 1.0e-12)
    retain_graph_for_telemetry = bool(capture_gradient_telemetry and not optimizer_spec.requires_closure)

    def raise_on_nonfinite_losses(losses: PinnLossBreakdown, *, stage: str) -> None:
        if not bool(torch.isfinite(losses.total.detach()).item()):
            raise FloatingPointError(f"Encountered non-finite total loss during {stage}.")
        for name, value in losses.components.items():
            if not bool(torch.isfinite(value.detach()).item()):
                raise FloatingPointError(f"Encountered non-finite loss component '{name}' during {stage}.")

    def raise_on_nonfinite_params(*, stage: str) -> None:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if not bool(torch.isfinite(param.detach()).all().item()):
                raise FloatingPointError(f"Encountered non-finite parameter '{name}' during {stage}.")

    def detach_loss_breakdown(losses: PinnLossBreakdown) -> PinnLossBreakdown:
        return PinnLossBreakdown(
            total=losses.total.detach(),
            components={name: value.detach() for name, value in losses.components.items()},
        )

    def closure_step_needs_post_measure() -> bool:
        if not hasattr(optimizer, "get_last_diagnostics"):
            return True
        diagnostics = optimizer.get_last_diagnostics()
        reason = diagnostics.get("reason")
        if reason in {"strong_wolfe_failed", "backtracking_failed", "step_below_min_threshold"}:
            return True
        if diagnostics.get("line_search_success") is False and int(diagnostics.get("line_search_evals") or 0) > 0:
            return True
        return False

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        losses = evaluate_pinn_loss_breakdown(
            model=model,
            criterion=criterion,
            ode_model=ode_model,
            formulation=formulation,
            weights=weights,
            x_data=x_data,
            y_data=y_data,
            x_data_dt_weights=x_data_dt_weights,
            x_col=x_col,
            x_col_weights=x_col_weights,
            x_init=x_init,
            y_init=y_init,
            x_init_weights=x_init_weights,
            create_graph=True,
        )
        raise_on_nonfinite_losses(losses, stage="forward pass")
        scaled_total = losses.total / scale
        scaled_total.backward(retain_graph=retain_graph_for_telemetry)
        breakdown_box["losses"] = losses
        return scaled_total

    if optimizer_spec.requires_closure:
        optimizer.step(closure)
        raise_on_nonfinite_params(stage="optimizer step")
        optimizer.zero_grad(set_to_none=True)
        # The closure writes the losses from its final (accepted) evaluation into
        # breakdown_box["losses"] on successful custom line-search steps. If a
        # line search failed or the optimizer cannot expose that guarantee,
        # re-measure on the restored/accepted parameters.
        if capture_gradient_telemetry or closure_step_needs_post_measure():
            measured_losses, measured_telemetry = measure_pinn_state(
                model=model,
                criterion=criterion,
                ode_model=ode_model,
                formulation=formulation,
                weights=weights,
                x_data=x_data,
                y_data=y_data,
                x_data_dt_weights=x_data_dt_weights,
                x_col=x_col,
                x_col_weights=x_col_weights,
                x_init=x_init,
                y_init=y_init,
                x_init_weights=x_init_weights,
                capture_gradient_telemetry=capture_gradient_telemetry,
            )
            breakdown_box["losses"] = measured_losses
            telemetry_box["telemetry"] = measured_telemetry
        raise_on_nonfinite_losses(breakdown_box["losses"], stage="post-step evaluation")
    else:
        closure()
        if capture_gradient_telemetry:
            telemetry_box["telemetry"] = compute_gradient_telemetry(
                model=model,
                losses=breakdown_box["losses"],
                weights=weights,
            )
        optimizer.step()
        raise_on_nonfinite_params(stage="optimizer step")
    return detach_loss_breakdown(breakdown_box["losses"]), telemetry_box["telemetry"]


def phase_effective_full_batch(phase: OptimizerPhase, optimizer_spec: OptimizerSpec) -> bool:
    return optimizer_spec.default_full_batch if phase.full_batch is None else bool(phase.full_batch)


def phase_allows_sampling(phase: OptimizerPhase, optimizer_spec: OptimizerSpec) -> bool:
    if phase.allow_sampling is None:
        return not phase_effective_full_batch(phase, optimizer_spec)
    return bool(phase.allow_sampling)


def phase_supports_dynamic_weight_updates(phase: OptimizerPhase) -> bool:
    return str(phase.optimizer).strip().lower() == "adam"


def build_phase_scheduler(
    *,
    optimizer: torch.optim.Optimizer,
    scheduler_config: SchedulerConfig | None,
) -> ReduceLROnPlateau | None:
    if scheduler_config is None:
        return None
    if scheduler_config.name != "reduce_on_plateau":
        raise ValueError("Unsupported scheduler. Use scheduler.name='reduce_on_plateau' or null.")
    return ReduceLROnPlateau(
        optimizer,
        mode=str(scheduler_config.mode),
        factor=float(scheduler_config.factor),
        patience=int(scheduler_config.patience),
        threshold=float(scheduler_config.threshold),
        threshold_mode=str(scheduler_config.threshold_mode),
        cooldown=int(scheduler_config.cooldown),
        min_lr=float(scheduler_config.min_lr),
        eps=float(scheduler_config.eps),
    )
