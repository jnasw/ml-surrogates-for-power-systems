"""Surrogate training and evaluation utilities."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import time
from typing import Any, Iterator

import numpy as np
import os
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from omegaconf import OmegaConf

from src.data.loaders.trajectory_dataset import TrajectoryDataset
from src.pinn.collocation import (
    CollocationStrategyContext,
    build_collocation_manager,
)
from src.pinn.data import PinnDatasetBundle
from src.pinn.evaluator import (
    evaluate_pinn_loss_breakdown,
    evaluate_pinn_weighting_terms,
    move_pinn_training_data_to_device,
)
from src.pinn.logging import EpochMetrics, PinnLogger
from src.pinn.losses import LOSS_COMPONENTS, LossWeights, PinnLossBreakdown
from src.pinn.multistage import (
    MultistagePinnEnsemble,
    MultistageStageMLP,
    StageDiagnostics,
    estimate_stage_diagnostics,
)
from src.pinn.optim import OptimizerSpec, build_optimizer
from src.pinn.residuals import compute_residual_terms, compute_supervised_dt_terms
from src.pinn.runtime import (
    OptimizerPhase,
    load_optimizer_phases,
    load_optimizer_phases_from_raw,
    resolve_torch_dtype,
    torch_dtype_name,
)
from src.pinn.vrba import (
    initialize_vrba_state,
    serialize_vrba_config,
    serialize_vrba_state,
    vrba_config_from_config,
)
from src.pinn.weighting import (
    WeightUpdateStats,
    WeightingConfig,
    build_weighting_policy,
    weighting_config_from_config,
)
from src.train.device import select_torch_device
from src.train.model import BaselineTrajectoryMLP, TimeConditionedMLP, TrajectoryMLP
from src.train.runtime import cfg_get, configure_reproducibility, configure_reproducibility_from_config


@dataclass
class SurrogateModel:
    model: TrajectoryMLP
    traj_steps: int
    traj_dim: int
    device: torch.device
    input_dim: int
    output_dim: int
    hidden_dim: int
    hidden_layers: int

    def predict(self, x_np: np.ndarray, batch_size: int = 2048) -> np.ndarray:
        x = torch.from_numpy(np.asarray(x_np, dtype=np.float32))
        loader = DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=False)
        self.model.eval()
        out = []
        with torch.no_grad():
            for (xb,) in loader:
                pred = self.model(xb.to(self.device)).cpu().numpy()
                out.append(pred)
        flat = np.concatenate(out, axis=0)
        return flat.reshape(-1, self.traj_steps, self.traj_dim)

    def save_checkpoint(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "state_dict": self.model.state_dict(),
            "traj_steps": self.traj_steps,
            "traj_dim": self.traj_dim,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_dim": self.hidden_dim,
            "hidden_layers": self.hidden_layers,
        }
        torch.save(payload, path)

    @staticmethod
    def load_checkpoint(path: str, device_preference: str = "auto") -> "SurrogateModel":
        device = select_torch_device(device_preference)
        payload = torch.load(path, map_location=device)
        model = TrajectoryMLP(
            input_dim=int(payload["input_dim"]),
            output_dim=int(payload["output_dim"]),
            hidden_dim=int(payload["hidden_dim"]),
            hidden_layers=int(payload["hidden_layers"]),
        ).to(device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return SurrogateModel(
            model=model,
            traj_steps=int(payload["traj_steps"]),
            traj_dim=int(payload["traj_dim"]),
            device=device,
            input_dim=int(payload["input_dim"]),
            output_dim=int(payload["output_dim"]),
            hidden_dim=int(payload["hidden_dim"]),
            hidden_layers=int(payload["hidden_layers"]),
        )


@dataclass
class BaselineConfig:
    hidden_dim: int = 256
    hidden_layers: int = 4
    dropout: float = 0.05
    lr: float = 1e-3
    batch_size: int = 64
    epochs: int = 300
    device: str = "auto"  # auto | cuda | mps | cpu


@dataclass(frozen=True)
class LossWeightScheduleStage:
    epochs: int | None
    weights: LossWeights


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


def _collocation_pool_diagnostics(manager: Any) -> dict[str, float | int | str | bool | None]:
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
        diagnostics["multipool_residual_epoch_rows"] = int(residual_pool.metadata.get("epoch_rows", residual_pool.points_x.shape[0]))
        diagnostics["multipool_residual_pool_rows"] = int(residual_pool.points_x.shape[0])
        for key, value in (residual_pool.metadata or {}).items():
            if key == "epoch_rows":
                continue
            if isinstance(value, (bool, int, float, str)) or value is None:
                diagnostics[f"multipool_residual_{key}"] = value
    if ic_pool is not None:
        diagnostics["multipool_ic_target_rows"] = int(ic_pool.target_rows)
        diagnostics["multipool_ic_epoch_rows"] = int(ic_pool.metadata.get("epoch_rows", ic_pool.points_x.shape[0]))
        diagnostics["multipool_ic_pool_rows"] = int(ic_pool.points_x.shape[0])
    for key, value in (state.metadata or {}).items():
        if isinstance(value, (bool, int, float, str)) or value is None:
            diagnostics[f"multipool_{key}"] = value
    return diagnostics


def _collocation_vrba_summary(manager: Any) -> dict[str, Any]:
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
    enabled = str(strategy_name).strip().lower() == "vrba_sample"
    return {
        "enabled": bool(enabled),
        "sampling_enabled": bool(enabled and bool(metadata.get("residual_vrba_sampling_enabled", False))),
        "weighting_enabled": bool(enabled and bool(metadata.get("residual_vrba_weighting_enabled", False))),
        "potential": None if not enabled else metadata.get("residual_vrba_potential"),
        "target_sets": None if not enabled else ("physics",),
        "update_count": None if not enabled else metadata.get("residual_vrba_update_count"),
    }


def _active_collocation_weights_or_none(
    *,
    vrba_config: Any,
    x_col_weights: torch.Tensor | None,
    weighting_enabled: bool | None = None,
) -> torch.Tensor | None:
    if weighting_enabled is False:
        return None
    if weighting_enabled is None and not bool(getattr(vrba_config, "adaptive_weighting", False)):
        return None
    if x_col_weights is None:
        raise ValueError(
            "pinn.vrba.adaptive_weighting=true requires a collocation strategy that supplies local weights. "
            "Use pinn.collocation.strategy=vrba_sample with pinn.vrba.enabled=true."
        )
    return x_col_weights


@dataclass(frozen=True)
class PinnBatch:
    x_data: torch.Tensor
    y_data: torch.Tensor
    x_col: torch.Tensor
    x_col_weights: torch.Tensor | None
    x_init: torch.Tensor
    y_init: torch.Tensor


def train_surrogate(dataset: TrajectoryDataset, seed: int, config: Any) -> SurrogateModel:
    configure_reproducibility_from_config(seed=seed, config=config, prefix="surrogate")

    x_np, y_np = dataset.training_view()
    n, t_steps, t_dim = y_np.shape
    y_flat = y_np.reshape(n, -1)

    hidden_dim = int(cfg_get(config, "surrogate.hidden_dim", 128))
    hidden_layers = int(cfg_get(config, "surrogate.hidden_layers", 3))
    batch_size = int(cfg_get(config, "surrogate.batch_size", 64))
    epochs = int(cfg_get(config, "surrogate.epochs", 200))
    lr = float(cfg_get(config, "surrogate.lr", 1e-3))

    device_pref = cfg_get(config, "surrogate.device", None)
    if device_pref is None:
        # Backward compatibility with old configs.
        use_cuda = bool(cfg_get(config, "surrogate.use_cuda", True))
        device_pref = "auto" if use_cuda else "cpu"
    device_pref = str(device_pref)
    if bool(cfg_get(config, "surrogate.require_explicit_device", False)) and device_pref.lower() == "auto":
        raise ValueError("surrogate.require_explicit_device=true requires surrogate.device to be one of: cpu/cuda/mps.")
    #if bool(cfg_get(config, "surrogate.warn_on_auto_device", True)) and device_pref.lower() == "auto":
    #    print("[surrogate] device=auto may reduce run-to-run reproducibility across heterogeneous machines.")
    device = select_torch_device(device_pref)
    model = TrajectoryMLP(input_dim=x_np.shape[1], output_dim=y_flat.shape[1], hidden_dim=hidden_dim, hidden_layers=hidden_layers).to(device)

    x = torch.from_numpy(x_np)
    y = torch.from_numpy(y_flat)
    train_shuffle = bool(cfg_get(config, "surrogate.train_data_shuffle", True))
    data_gen = torch.Generator()
    data_gen.manual_seed(seed)
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=train_shuffle, generator=data_gen)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

    return SurrogateModel(
        model=model,
        traj_steps=t_steps,
        traj_dim=t_dim,
        device=device,
        input_dim=x_np.shape[1],
        output_dim=y_flat.shape[1],
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
    )


def train_baseline_surrogate(dataset: TrajectoryDataset, seed: int, cfg: BaselineConfig) -> SurrogateModel:
    configure_reproducibility(seed=seed)

    x_np, y_np = dataset.training_view()
    n, t_steps, t_dim = y_np.shape
    y_flat = y_np.reshape(n, -1)

    device = select_torch_device(cfg.device)
    model = BaselineTrajectoryMLP(
        input_dim=x_np.shape[1],
        output_dim=y_flat.shape[1],
        hidden_dim=cfg.hidden_dim,
        hidden_layers=cfg.hidden_layers,
        dropout=cfg.dropout,
    ).to(device)

    x = torch.from_numpy(x_np)
    y = torch.from_numpy(y_flat)
    data_gen = torch.Generator()
    data_gen.manual_seed(seed)
    loader = DataLoader(TensorDataset(x, y), batch_size=cfg.batch_size, shuffle=True, generator=data_gen)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(cfg.epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

    return SurrogateModel(
        model=model,
        traj_steps=t_steps,
        traj_dim=t_dim,
        device=device,
        input_dim=x_np.shape[1],
        output_dim=y_flat.shape[1],
        hidden_dim=cfg.hidden_dim,
        hidden_layers=cfg.hidden_layers,
    )


def _loss_weights_from_config(config: Any) -> LossWeights:
    # Backward compatibility: static configs still read pinn.loss_weights directly.
    return LossWeights(
        data=float(cfg_get(config, "pinn.loss_weights.data", 1.0)),
        dt=float(cfg_get(config, "pinn.loss_weights.dt", 1.0e-4)),
        physics=float(cfg_get(config, "pinn.loss_weights.physics", 1.0)),
        ic=float(cfg_get(config, "pinn.loss_weights.ic", 1.0)),
    )


def _loss_weight_schedule_from_config(config: Any) -> list[LossWeightScheduleStage] | None:
    raw_schedule = cfg_get(config, "pinn.loss_weight_schedule", None)
    if raw_schedule in (None, "null"):
        return None
    schedule: list[LossWeightScheduleStage] = []
    for idx, item in enumerate(raw_schedule):
        raw_epochs = getattr(item, "epochs", None)
        epochs = None if raw_epochs in (None, "null") else int(raw_epochs)
        if epochs is not None and epochs <= 0:
            raise ValueError(f"config.pinn.loss_weight_schedule[{idx}].epochs must be > 0 when provided.")
        raw_weights = getattr(item, "weights", None)
        if raw_weights in (None, "null"):
            raise ValueError(f"config.pinn.loss_weight_schedule[{idx}].weights must be provided.")
        schedule.append(
            LossWeightScheduleStage(
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
    return schedule


def _scheduled_loss_weights(
    *,
    base_weights: LossWeights,
    schedule: list[LossWeightScheduleStage] | None,
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


def _sample_supervised_rows(x: torch.Tensor, y: torch.Tensor, config: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if not bool(cfg_get(config, "pinn.supervised_sampling.enabled", False)):
        return x, y
    total_rows = int(x.shape[0])
    rows_per_epoch_cfg = cfg_get(config, "pinn.supervised_sampling.rows_per_epoch", None)
    fraction_per_epoch_cfg = cfg_get(config, "pinn.supervised_sampling.fraction_per_epoch", None)

    target_rows: int | None = None
    if rows_per_epoch_cfg not in (None, "null"):
        target_rows = int(rows_per_epoch_cfg)
    elif fraction_per_epoch_cfg not in (None, "null"):
        fraction = float(fraction_per_epoch_cfg)
        if not (0.0 < fraction <= 1.0):
            raise ValueError("pinn.supervised_sampling.fraction_per_epoch must be in (0, 1].")
        target_rows = max(1, int(round(total_rows * fraction)))

    if target_rows is None or target_rows >= total_rows:
        return x, y
    if target_rows <= 0:
        raise ValueError("pinn.supervised_sampling.rows_per_epoch must be > 0.")

    indices_np = np.random.choice(total_rows, size=target_rows, replace=False)
    indices = torch.as_tensor(indices_np, device=x.device, dtype=torch.long)
    return x.index_select(0, indices), y.index_select(0, indices)


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
    x_data, y_data = _take_fixed_tensor_rows(
        dataset.train_x,
        target_rows=weighting_config.probe.data_rows,
        seed=weighting_config.probe.seed + seed_offset,
        y=dataset.train_y,
    )
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
        x_col=x_col,
        x_init=x_init,
        y_init=y_init,
    )


def _should_run_evaluation(global_epoch: int, config: Any) -> bool:
    frequency = int(cfg_get(config, "pinn.evaluation.frequency", 1))
    if frequency < 1:
        frequency = 1
    return (int(global_epoch) % frequency) == 0


def _model_device_and_dtype(model: nn.Module) -> tuple[torch.device, torch.dtype]:
    param = next(model.parameters())
    return param.device, param.dtype


def _to_model_tensor(tensor: torch.Tensor | None, model: nn.Module) -> torch.Tensor | None:
    if tensor is None:
        return None
    device, dtype = _model_device_and_dtype(model)
    return tensor.to(device=device, dtype=dtype)


def _evaluate_data_loss(
    model: nn.Module,
    x: torch.Tensor | None,
    y: torch.Tensor | None,
    criterion: nn.Module,
) -> float | None:
    if x is None or y is None:
        return None
    x_eval = _to_model_tensor(x, model)
    y_eval = _to_model_tensor(y, model)
    model.eval()
    with torch.no_grad():
        pred = model(x_eval)
        return float(criterion(pred, y_eval).item())


def _evaluate_physics_loss(
    model: nn.Module,
    x_rows: torch.Tensor | None,
    ode_model: Any,
    criterion: nn.Module,
    formulation: str,
) -> float | None:
    if x_rows is None:
        return None
    x_eval = _to_model_tensor(x_rows, model)
    model.eval()
    terms = compute_residual_terms(
        model=model,
        x=x_eval,
        ode_model=ode_model,
        formulation=formulation,
        create_graph=False,
    )
    return float(criterion(terms.residual, torch.zeros_like(terms.residual)).item())


def _evaluate_dt_loss(
    model: nn.Module,
    x: torch.Tensor | None,
    y: torch.Tensor | None,
    ode_model: Any,
    criterion: nn.Module,
    formulation: str,
) -> float | None:
    if x is None or y is None:
        return None
    x_eval = _to_model_tensor(x, model)
    y_eval = _to_model_tensor(y, model)
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


def _compute_weighted_total_loss(
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


def _ceil_div(num: int, den: int) -> int:
    return (int(num) + int(den) - 1) // int(den)


class _ComponentBatchSampler:
    def __init__(self, size: int, batch_size: int, *, shuffle: bool, seed: int) -> None:
        if size <= 0:
            raise ValueError("Sampler size must be > 0.")
        if batch_size <= 0:
            raise ValueError("Sampler batch_size must be > 0.")
        self.size = int(size)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(int(seed))
        self._indices = torch.empty(0, dtype=torch.long)
        self._offset = 0

    def _reset(self) -> None:
        if self.shuffle:
            self._indices = torch.randperm(self.size, generator=self._generator, device="cpu")
        else:
            self._indices = torch.arange(self.size, dtype=torch.long)
        self._offset = 0

    def next_indices(self, *, target_device: torch.device) -> torch.Tensor:
        if self._offset >= int(self._indices.numel()):
            self._reset()

        end = min(self._offset + self.batch_size, int(self._indices.numel()))
        chunk = self._indices[self._offset:end]
        self._offset = end

        if int(chunk.numel()) < self.batch_size:
            remainder = self.batch_size - int(chunk.numel())
            self._reset()
            refill = self._indices[:remainder]
            self._offset = remainder
            chunk = torch.cat((chunk, refill), dim=0)

        return chunk.to(device=target_device)


def _sample_rows_with_indices(
    x: torch.Tensor,
    indices: torch.Tensor,
    y: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
    if y is None:
        return x.index_select(0, indices)
    return x.index_select(0, indices), y.index_select(0, indices)


def _iter_minibatch_batches(
    *,
    dataset: PinnDatasetBundle,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> tuple[int, Iterator[PinnBatch]]:
    steps_per_epoch = max(
        _ceil_div(dataset.train_x.shape[0], batch_size),
        _ceil_div(dataset.train_col_x.shape[0], batch_size),
        _ceil_div(dataset.train_init_x.shape[0], batch_size),
    )
    data_sampler = _ComponentBatchSampler(
        int(dataset.train_x.shape[0]),
        batch_size,
        shuffle=shuffle,
        seed=seed,
    )
    col_sampler = _ComponentBatchSampler(
        int(dataset.train_col_x.shape[0]),
        batch_size,
        shuffle=shuffle,
        seed=seed + 1,
    )
    init_sampler = _ComponentBatchSampler(
        int(dataset.train_init_x.shape[0]),
        batch_size,
        shuffle=shuffle,
        seed=seed + 2,
    )

    def batch_iterator():
        for _ in range(steps_per_epoch):
            data_idx = data_sampler.next_indices(target_device=dataset.train_x.device)
            col_idx = col_sampler.next_indices(target_device=dataset.train_col_x.device)
            init_idx = init_sampler.next_indices(target_device=dataset.train_init_x.device)
            x_data, y_data = _sample_rows_with_indices(dataset.train_x, data_idx, dataset.train_y)
            x_init, y_init = _sample_rows_with_indices(dataset.train_init_x, init_idx, dataset.train_init_y)
            x_col = _sample_rows_with_indices(dataset.train_col_x, col_idx)
            x_col_weights = None if dataset.train_col_weights is None else _sample_rows_with_indices(dataset.train_col_weights, col_idx)
            yield PinnBatch(
                x_data=x_data,
                y_data=y_data,
                x_col=x_col,
                x_col_weights=x_col_weights,
                x_init=x_init,
                y_init=y_init,
            )

    return steps_per_epoch, batch_iterator()


def _build_checkpoint_payload(
    *,
    pinn_model: PinnModel,
    metrics: EpochMetrics | None,
    config: Any,
    tag: str,
    vrba_config: Any | None = None,
    vrba_state: Any | None = None,
) -> dict[str, Any]:
    return {
        "state_dict": pinn_model.model.state_dict(),
        "input_dim": pinn_model.input_dim,
        "output_dim": pinn_model.output_dim,
        "hidden_dim": pinn_model.hidden_dim,
        "hidden_layers": pinn_model.hidden_layers,
        "activation": pinn_model.activation,
        "dtype": torch_dtype_name(pinn_model.dtype),
        "architecture": pinn_model.architecture,
        "stage_first_activations": [] if pinn_model.stage_first_activations is None else list(pinn_model.stage_first_activations),
        "stage_hidden_activations": [] if pinn_model.stage_hidden_activations is None else list(pinn_model.stage_hidden_activations),
        "stage_time_scales": [] if pinn_model.stage_time_scales is None else [float(x) for x in pinn_model.stage_time_scales],
        "stage_epsilons": [] if pinn_model.stage_epsilons is None else [float(x) for x in pinn_model.stage_epsilons],
        "checkpoint_tag": tag,
        "metrics": metrics,
        "config": OmegaConf.to_container(config, resolve=True),
        "vrba": {
            "config": None if vrba_config is None else serialize_vrba_config(vrba_config),
            "state": None if vrba_state is None else serialize_vrba_state(vrba_state),
        },
    }


def _checkpointing_enabled(config: Any, key: str, default: bool) -> bool:
    return bool(cfg_get(config, f"pinn.checkpointing.{key}", default))


def _resolve_checkpoint_milestones(config: Any, total_epochs: int) -> dict[int, str]:
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


def _grad_norm(loss: torch.Tensor, model: nn.Module, retain_graph: bool) -> float:
    params = [param for param in model.parameters() if param.requires_grad]
    grads = torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=True)
    sq_norm = torch.zeros((), dtype=loss.dtype, device=loss.device)
    for grad in grads:
        if grad is None:
            continue
        sq_norm = sq_norm + grad.pow(2).sum()
    return float(torch.sqrt(sq_norm).detach().cpu().item())


def _flattened_grads(loss: torch.Tensor, model: nn.Module, retain_graph: bool) -> torch.Tensor:
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


def _compute_output_jacobian(output: torch.Tensor, model: nn.Module) -> torch.Tensor:
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
            jacobian_parts.append(torch.zeros((int(output_flat.numel()), param.numel()), dtype=output.dtype, device=output.device))
            continue
        jacobian_parts.append(grad.reshape(int(output_flat.numel()), -1))
    if not jacobian_parts:
        return torch.zeros((int(output_flat.numel()), 0), dtype=output.dtype, device=output.device)
    return torch.cat(jacobian_parts, dim=1)


def _sample_ntk_term_rows(term: torch.Tensor, *, target_rows: int, seed: int) -> torch.Tensor:
    total_rows = int(term.shape[0])
    if target_rows >= total_rows:
        return term
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    indices = torch.randperm(total_rows, generator=generator, device="cpu")[:target_rows].to(device=term.device)
    return term.index_select(0, indices)


def _compute_ntk_mean_trace(output: torch.Tensor, model: nn.Module) -> float:
    num_points = max(int(output.shape[0]), 1)
    jacobian = _compute_output_jacobian(output, model)
    trace = torch.einsum("na,na->", jacobian, jacobian)
    return float((trace / float(num_points)).detach().cpu().item())


def _compute_gradient_telemetry(
    *,
    model: nn.Module,
    losses: PinnLossBreakdown,
    weights: LossWeights,
) -> GradientTelemetry:
    component_grad_norms = {
        name: _grad_norm(losses.component(name), model, retain_graph=True)
        for name in losses.components
    }
    weighted_component_grad_norms = {
        name: _grad_norm(float(weights.get(name)) * losses.component(name), model, retain_graph=True)
        for name in losses.components
    }
    return GradientTelemetry(
        total_grad_norm=_grad_norm(losses.total, model, retain_graph=True),
        component_grad_norms=component_grad_norms,
        weighted_component_grad_norms=weighted_component_grad_norms,
    )


def _compute_weight_update_stats(
    *,
    model: nn.Module,
    losses: PinnLossBreakdown,
    anchor_component: str | None,
    epoch: int,
    global_epoch: int,
) -> WeightUpdateStats:
    grad_vectors = {
        name: _flattened_grads(losses.component(name), model, retain_graph=True)
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


def _compute_ntk_weight_update_stats(
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
        sampled_term = _sample_ntk_term_rows(
            weighting_terms[name],
            target_rows=int(ntk_batch_sizes[name]),
            seed=int(weighting_config.ntk_seed) + int(global_epoch) * 97 + offset,
        )
        ntk_mean_trace[name] = _compute_ntk_mean_trace(sampled_term, model)
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


def _measure_pinn_state(
    *,
    model: nn.Module,
    criterion: nn.Module,
    ode_model: Any,
    formulation: str,
    weights: LossWeights,
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    x_col: torch.Tensor,
    x_col_weights: torch.Tensor | None,
    x_init: torch.Tensor,
    y_init: torch.Tensor,
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
        x_col=x_col,
        x_col_weights=x_col_weights,
        x_init=x_init,
        y_init=y_init,
        create_graph=capture_gradient_telemetry,
    )
    telemetry = None
    if capture_gradient_telemetry:
        telemetry = _compute_gradient_telemetry(
            model=model,
            losses=losses,
            weights=weights,
        )
    return losses, telemetry


def _train_pinn_step(
    *,
    model: nn.Module,
    optimizer_spec: OptimizerSpec,
    criterion: nn.Module,
    ode_model: Any,
    formulation: str,
    weights: LossWeights,
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    x_col: torch.Tensor,
    x_col_weights: torch.Tensor | None,
    x_init: torch.Tensor,
    y_init: torch.Tensor,
    capture_gradient_telemetry: bool = False,
    objective_scale: float = 1.0,
) -> tuple[PinnLossBreakdown, GradientTelemetry | None]:
    breakdown_box: dict[str, PinnLossBreakdown] = {}
    telemetry_box: dict[str, GradientTelemetry | None] = {"telemetry": None}
    optimizer = optimizer_spec.optimizer
    scale = max(float(objective_scale), 1.0e-12)
    retain_graph_for_telemetry = bool(capture_gradient_telemetry and not optimizer_spec.requires_closure)

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
            x_col=x_col,
            x_col_weights=x_col_weights,
            x_init=x_init,
            y_init=y_init,
            create_graph=True,
        )
        scaled_total = losses.total / scale
        scaled_total.backward(retain_graph=retain_graph_for_telemetry)
        breakdown_box["losses"] = losses
        return scaled_total

    if optimizer_spec.requires_closure:
        optimizer.step(closure)
        optimizer.zero_grad(set_to_none=True)
        measured_losses, measured_telemetry = _measure_pinn_state(
            model=model,
            criterion=criterion,
            ode_model=ode_model,
            formulation=formulation,
            weights=weights,
            x_data=x_data,
            y_data=y_data,
            x_col=x_col,
            x_col_weights=x_col_weights,
            x_init=x_init,
            y_init=y_init,
            capture_gradient_telemetry=capture_gradient_telemetry,
        )
        breakdown_box["losses"] = measured_losses
        telemetry_box["telemetry"] = measured_telemetry
    else:
        closure()
        if capture_gradient_telemetry:
            telemetry_box["telemetry"] = _compute_gradient_telemetry(
                model=model,
                losses=breakdown_box["losses"],
                weights=weights,
            )
        optimizer.step()
    return breakdown_box["losses"], telemetry_box["telemetry"]


def _phase_effective_full_batch(phase: OptimizerPhase, optimizer_spec: OptimizerSpec) -> bool:
    return optimizer_spec.default_full_batch if phase.full_batch is None else bool(phase.full_batch)


def _phase_allows_sampling(phase: OptimizerPhase, optimizer_spec: OptimizerSpec) -> bool:
    if phase.allow_sampling is None:
        return not _phase_effective_full_batch(phase, optimizer_spec)
    return bool(phase.allow_sampling)


def _phase_supports_dynamic_weight_updates(phase: OptimizerPhase) -> bool:
    return str(phase.optimizer).strip().lower() == "adam"


def _collocation_refresh_mode(config: Any) -> str:
    return str(cfg_get(config, "pinn.collocation.refresh.mode", "epoch_periodic")).strip().lower()


def _collocation_phase_boundary_enabled(config: Any) -> bool:
    return _collocation_refresh_mode(config) == "phase_boundary"


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


def _phase_boundary_refresh_planned(config: Any, *, phase_name: str, event: str) -> bool:
    if _collocation_refresh_mode(config) != "phase_boundary":
        return False
    raw_names = cfg_get(config, f"pinn.collocation.refresh.on_phase_{event}", [])
    names = {str(name).strip().lower() for name in raw_names}
    return str(phase_name).strip().lower() in names


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
        "state_dict": copy.deepcopy(optimizer_spec.optimizer.state_dict()),
        "optimizer_name": optimizer_name,
    }
    transition_state.previous_optimizer_name = optimizer_name


def _pinn_mode(config: Any) -> str:
    return str(cfg_get(config, "pinn.mode", "single_stage")).strip().lower()


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


def _multistage_base_optimizer_phases(config: Any) -> list[OptimizerPhase]:
    raw = cfg_get(config, "pinn.multistage.base_stage.optimizer_phases", None)
    if raw in (None, "null"):
        return load_optimizer_phases(config)
    return load_optimizer_phases_from_raw(raw, config_label="config.pinn.multistage.base_stage.optimizer_phases")


def _multistage_residual_optimizer_phases(config: Any) -> list[OptimizerPhase]:
    raw = cfg_get(config, "pinn.multistage.residual_stage.optimizer_phases", None)
    if raw in (None, "null"):
        return load_optimizer_phases(config)
    return load_optimizer_phases_from_raw(raw, config_label="config.pinn.multistage.residual_stage.optimizer_phases")


def _multistage_stage_optimizer_phases(config: Any) -> list[list[OptimizerPhase]] | None:
    raw = cfg_get(config, "pinn.multistage.stage_optimizer_phases", None)
    if raw in (None, "null"):
        return None
    schedules: list[list[OptimizerPhase]] = []
    for idx, raw_schedule in enumerate(raw):
        schedules.append(
            load_optimizer_phases_from_raw(
                raw_schedule,
                config_label=f"config.pinn.multistage.stage_optimizer_phases[{idx}]",
            )
        )
    return schedules


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
    logger: PinnLogger | None = None,
) -> tuple[PinnModel, list[EpochMetrics]]:
    seed = int(cfg_get(config, "model.seed", 0))
    configure_reproducibility_from_config(seed=seed, config=config, prefix="pinn")

    dtype = resolve_torch_dtype(str(cfg_get(config, "pinn.dtype", "float64")))
    device = select_torch_device(str(cfg_get(config, "pinn.device", "auto")))
    hidden_dim = int(cfg_get(config, "pinn.hidden_dim", 64))
    hidden_layers = int(cfg_get(config, "pinn.hidden_layers", 4))
    activation = str(cfg_get(config, "pinn.activation", "tanh"))
    formulation = str(cfg_get(config, "pinn.formulation", "odequations"))
    explicit_stage_optimizer_phases = _multistage_stage_optimizer_phases(config)
    base_optimizer_phases = _multistage_base_optimizer_phases(config)
    residual_optimizer_phases = _multistage_residual_optimizer_phases(config)
    base_weights = _loss_weights_from_config(config)
    vrba_config = vrba_config_from_config(config)
    gradient_telemetry_enabled = bool(cfg_get(config, "pinn.gradient_telemetry.enabled", False))
    cost_tracking_enabled = _cost_tracking_enabled(config)
    criterion = nn.MSELoss()
    transition_settings = _optimizer_transition_settings(config)
    transition_state = OptimizerTransitionState(cache={})

    dataset = move_pinn_training_data_to_device(dataset, device=device, dtype=dtype)
    collocation_manager = build_collocation_manager(
        initial_points=dataset.train_col_x,
        init_x=dataset.train_init_x,
        init_y=dataset.train_init_y,
        config=config,
    )
    initial_pool_batch = collocation_manager.initial_epoch_batch()
    dataset = PinnDatasetBundle(
        train_x=dataset.train_x,
        train_y=dataset.train_y,
        train_col_x=initial_pool_batch.x_col,
        train_init_x=initial_pool_batch.x_init,
        train_init_y=initial_pool_batch.y_init,
        train_col_weights=initial_pool_batch.x_col_weights,
        val_x=dataset.val_x,
        val_y=dataset.val_y,
        val_col_x=dataset.val_col_x,
        test_x=dataset.test_x,
        test_y=dataset.test_y,
    )
    vrba_state = initialize_vrba_state(
        vrba_config,
        initial_point_counts={
            "data": int(dataset.train_x.shape[0]),
            "dt": int(dataset.train_x.shape[0]),
            "physics": int(dataset.train_col_x.shape[0]),
            "ic": int(dataset.train_init_x.shape[0]),
        },
    )
    analysis_x_col = _multistage_analysis_collocation(dataset, config)
    max_stages = len(explicit_stage_optimizer_phases) if explicit_stage_optimizer_phases is not None else _multistage_max_stages(config)
    stop_threshold = _multistage_stop_threshold(config)
    if explicit_stage_optimizer_phases is not None:
        total_epochs = int(sum(sum(phase.epochs for phase in schedule) for schedule in explicit_stage_optimizer_phases))
    else:
        base_total_epochs = int(sum(phase.epochs for phase in base_optimizer_phases))
        residual_total_epochs = int(sum(phase.epochs for phase in residual_optimizer_phases))
        total_epochs = int(base_total_epochs + max(0, max_stages - 1) * residual_total_epochs)
    checkpoint_milestones = _resolve_checkpoint_milestones(config, total_epochs)

    stages: list[MultistageStageMLP] = []
    epsilons: list[float] = []
    rows: list[EpochMetrics] = []
    best_metric: float | None = None
    global_epoch = 0
    train_started = time.perf_counter()
    ensemble: MultistagePinnEnsemble | None = None
    stage_start_diagnostics: StageDiagnostics | None = None

    for stage_idx in range(max_stages):
        if stage_idx == 0:
            time_scale = 1.0
        else:
            if ensemble is None:
                raise RuntimeError("Expected an existing ensemble before training a residual stage.")
            residual_terms = compute_residual_terms(
                model=ensemble,
                x=analysis_x_col,
                ode_model=ode_model,
                formulation=formulation,
                create_graph=False,
            )
            stage_start_diagnostics = estimate_stage_diagnostics(
                x_col=analysis_x_col,
                residual=residual_terms.residual,
                min_kappa=float(cfg_get(config, "pinn.multistage.residual_stage.kappa_min", 1.0)),
                max_kappa=float(cfg_get(config, "pinn.multistage.residual_stage.kappa_max", 100.0)),
            )
            if stage_start_diagnostics.residual_rms <= stop_threshold:
                break
            time_scale = stage_start_diagnostics.kappa_suggested

        stage_model = _build_multistage_stage_model_from_config(
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
        epsilons.append(_multistage_stage_epsilon(config, stage_idx))
        ensemble = MultistagePinnEnsemble(stages=list(stages), epsilons=list(epsilons)).to(device=device, dtype=dtype)
        ensemble.freeze_stages_up_to(stage_idx)
        pinn_model = _assemble_multistage_pinn_model(
            ensemble=ensemble,
            device=device,
            dtype=dtype,
            input_dim=dataset.input_dim,
            output_dim=dataset.output_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation=activation,
        )

        if logger is not None and stage_idx == 0 and _checkpointing_enabled(config, "save_init", True):
            logger.save_checkpoint(
                _build_checkpoint_payload(
                    pinn_model=pinn_model,
                    metrics=None,
                    config=config,
                    tag="init",
                    vrba_config=vrba_config,
                    vrba_state=vrba_state,
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
        initial_stage_epsilon = _multistage_effective_epsilon(
            target_epsilon=target_stage_epsilon,
            stage_idx=stage_idx,
            stage_epoch=0,
            stage_total_epochs=stage_total_epochs,
            config=config,
        )
        ensemble.set_stage_epsilon(stage_idx, initial_stage_epsilon)
        stage_loss_ref = 1.0
        if _multistage_loss_ref_enabled(config):
            stage_ref_losses, _ = _measure_pinn_state(
                model=ensemble,
                criterion=criterion,
                ode_model=ode_model,
                formulation=formulation,
                weights=base_weights,
                x_data=dataset.train_x,
                y_data=dataset.train_y,
                x_col=dataset.train_col_x,
                x_col_weights=_active_collocation_weights_or_none(vrba_config=vrba_config, x_col_weights=dataset.train_col_weights),
                x_init=dataset.train_init_x,
                y_init=dataset.train_init_y,
                capture_gradient_telemetry=False,
            )
            stage_loss_ref = max(
                float(stage_ref_losses.total.detach().cpu().item()),
                _multistage_loss_ref_min_value(config),
            )
        stage_epoch_cursor = 0
        for phase in stage_optimizer_phases:
            optimizer_spec = build_optimizer(
                model=current_stage,
                optimizer_name=phase.optimizer,
                lr=phase.lr,
                optimizer_kwargs=phase.optimizer_kwargs,
                line_search=phase.line_search,
            )
            phase_full_batch = _phase_effective_full_batch(phase, optimizer_spec)
            if phase_full_batch and phase.batch_size is not None:
                raise ValueError(f"Optimizer phase '{phase.name}' is full-batch and must set batch_size=null.")
            phase_allows_sampling = _phase_allows_sampling(phase, optimizer_spec)
            if phase_full_batch and phase_allows_sampling:
                raise ValueError(f"Optimizer phase '{phase.name}' is full-batch and must disable sampling for a stable objective.")
            if phase_full_batch:
                phase_batch_size = None
            else:
                if not optimizer_spec.supports_minibatch:
                    raise ValueError(f"Optimizer '{phase.optimizer}' does not support minibatch execution.")
                phase_batch_size = phase.batch_size if phase.batch_size is not None else int(cfg_get(config, "pinn.default_batch_size", 1024))
            refresh_on_phase_start = _phase_boundary_refresh_planned(config, phase_name=phase.name, event="start")
            transition_diagnostics = _maybe_restore_optimizer_state(
                optimizer_spec=optimizer_spec,
                phase=phase,
                config=config,
                transition_settings=transition_settings,
                transition_state=transition_state,
                refresh_on_phase_start=refresh_on_phase_start,
            )

            if _collocation_phase_boundary_enabled(config):
                collocation_manager.handle_phase_boundary(
                    context=CollocationStrategyContext(
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
                epoch_started = time.perf_counter()
                _reset_peak_memory_if_cuda(device, enabled=cost_tracking_enabled)
                effective_stage_epsilon = _multistage_effective_epsilon(
                    target_epsilon=target_stage_epsilon,
                    stage_idx=stage_idx,
                    stage_epoch=stage_epoch_cursor,
                    stage_total_epochs=stage_total_epochs,
                    config=config,
                )
                ensemble.set_stage_epsilon(stage_idx, effective_stage_epsilon)
                ensemble.train()
                epoch_losses: list[PinnLossBreakdown] = []
                epoch_gradients: list[GradientTelemetry] = []
                active_weights = base_weights
                if phase_allows_sampling:
                    epoch_train_x, epoch_train_y = _sample_supervised_rows(dataset.train_x, dataset.train_y, config)
                else:
                    epoch_train_x, epoch_train_y = dataset.train_x, dataset.train_y
                epoch_pool_batch = collocation_manager.prepare_epoch_batch(
                    context=CollocationStrategyContext(
                        global_epoch=global_epoch + 1,
                        phase_name=phase.name,
                        phase_allows_sampling=phase_allows_sampling,
                        phase_is_full_batch=phase_full_batch,
                        phase_epoch=epoch,
                        refresh_event=None,
                        model=ensemble,
                        ode_model=ode_model,
                        formulation=formulation,
                    )
                )
                epoch_train_col_x = epoch_pool_batch.x_col
                epoch_train_col_weights = epoch_pool_batch.x_col_weights
                active_collocation_weights = _active_collocation_weights_or_none(
                    vrba_config=vrba_config,
                    x_col_weights=epoch_train_col_weights,
                    weighting_enabled=bool(collocation_manager.state.metadata.get("residual_vrba_weighting_enabled", False)),
                )
                epoch_train_init_x = epoch_pool_batch.x_init
                epoch_train_init_y = epoch_pool_batch.y_init
                num_batches = 0
                num_train_steps = 0
                num_supervised_rows = 0
                num_collocation_rows = 0
                num_init_rows = 0

                if phase_batch_size is None:
                    breakdown, gradient_telemetry = _train_pinn_step(
                        model=ensemble,
                        optimizer_spec=optimizer_spec,
                        criterion=criterion,
                        ode_model=ode_model,
                        formulation=formulation,
                        weights=active_weights,
                        x_data=epoch_train_x,
                        y_data=epoch_train_y,
                        x_col=epoch_train_col_x,
                        x_col_weights=active_collocation_weights,
                        x_init=epoch_train_init_x,
                        y_init=epoch_train_init_y,
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
                    _steps, epoch_batches = _iter_minibatch_batches(
                        dataset=PinnDatasetBundle(
                            train_x=epoch_train_x,
                            train_y=epoch_train_y,
                            train_col_x=epoch_train_col_x,
                            train_col_weights=active_collocation_weights,
                            train_init_x=epoch_train_init_x,
                            train_init_y=epoch_train_init_y,
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
                        breakdown, gradient_telemetry = _train_pinn_step(
                            model=ensemble,
                            optimizer_spec=optimizer_spec,
                            criterion=criterion,
                            ode_model=ode_model,
                            formulation=formulation,
                            weights=active_weights,
                            x_data=batch.x_data,
                            y_data=batch.y_data,
                            x_col=batch.x_col,
                            x_col_weights=batch.x_col_weights,
                            x_init=batch.x_init,
                            y_init=batch.y_init,
                            capture_gradient_telemetry=gradient_telemetry_enabled,
                            objective_scale=stage_loss_ref,
                        )
                        epoch_losses.append(breakdown)
                        if gradient_telemetry is not None:
                            epoch_gradients.append(gradient_telemetry)

                epoch_wall_seconds = float(time.perf_counter() - epoch_started) if cost_tracking_enabled else None
                cumulative_wall_seconds = float(time.perf_counter() - train_started) if cost_tracking_enabled else None
                peak_gpu_memory_allocated_bytes, peak_gpu_memory_reserved_bytes = _peak_memory_if_cuda(
                    device,
                    enabled=cost_tracking_enabled,
                )
                train_total = float(torch.stack([x.total.detach() for x in epoch_losses]).mean().item())
                train_component_losses = {
                    name: float(torch.stack([loss.components[name].detach() for loss in epoch_losses]).mean().item())
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
                val_total_loss = None
                val_component_losses = None
                test_metrics = None
                if _should_run_evaluation(global_epoch, config):
                    val_component_losses = {}
                    val_component_losses["data"] = _evaluate_data_loss(model=ensemble, x=dataset.val_x, y=dataset.val_y, criterion=criterion)
                    val_component_losses["dt"] = _evaluate_dt_loss(
                        model=ensemble,
                        x=dataset.val_x,
                        y=dataset.val_y,
                        ode_model=ode_model,
                        criterion=criterion,
                        formulation=formulation,
                    )
                    val_component_losses["physics"] = _evaluate_physics_loss(
                        model=ensemble,
                        x_rows=dataset.val_col_x if dataset.val_col_x is not None else dataset.val_x,
                        ode_model=ode_model,
                        criterion=criterion,
                        formulation=formulation,
                    )
                    init_val = _evaluate_data_loss(model=ensemble, x=epoch_train_init_x, y=epoch_train_init_y, criterion=criterion)
                    val_component_losses["ic"] = init_val
                    val_total_loss = _compute_weighted_total_loss(val_component_losses, active_weights)
                    test_metrics = {"data_loss": _evaluate_data_loss(model=ensemble, x=dataset.test_x, y=dataset.test_y, criterion=criterion)}

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
                optimizer_diagnostics.update(transition_diagnostics)
                optimizer_diagnostics.update(_collocation_pool_diagnostics(collocation_manager))
                if stage_start_diagnostics is not None and stage_idx > 0:
                    optimizer_diagnostics["multistage_stage_start_residual_rms"] = float(stage_start_diagnostics.residual_rms)
                    optimizer_diagnostics["multistage_stage_zero_crossings"] = int(stage_start_diagnostics.residual_zero_crossings)
                    optimizer_diagnostics["multistage_stage_dominant_residual_channel"] = int(stage_start_diagnostics.dominant_residual_channel)
                if hasattr(optimizer_spec.optimizer, "get_last_diagnostics"):
                    optimizer_diagnostics.update(optimizer_spec.optimizer.get_last_diagnostics())

                vrba_summary = _collocation_vrba_summary(collocation_manager)
                row = EpochMetrics(
                    epoch=epoch,
                    global_epoch=global_epoch,
                    phase_name=f"stage{stage_idx:02d}_{phase.name}",
                    optimizer=phase.optimizer,
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
                    weighting_anchor="physics",
                    vrba_enabled=bool(vrba_summary["enabled"]),
                    vrba_sampling_enabled=bool(vrba_summary["sampling_enabled"]),
                    vrba_weighting_enabled=bool(vrba_summary["weighting_enabled"]),
                    vrba_potential=vrba_summary["potential"],
                    vrba_target_sets=vrba_summary["target_sets"],
                    vrba_update_count=None if vrba_summary["update_count"] is None else int(vrba_summary["update_count"]),
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
                mean_epoch_breakdown = PinnLossBreakdown(
                    total=epoch_losses[0].total.detach().new_tensor(train_total),
                    components={
                        name: epoch_losses[0].components[name].detach().new_tensor(train_component_losses[name])
                        for name in epoch_losses[0].components
                    },
                )
                rows.append(row)
                collocation_manager.observe_epoch_losses(
                    global_epoch=global_epoch,
                    losses=mean_epoch_breakdown,
                )

                if logger is not None:
                    logger.write_metrics(rows)
                    logger.print_epoch_metrics(row)
                    logger.log_epoch_metrics(row)
                    milestone_tag = checkpoint_milestones.get(global_epoch)
                    if milestone_tag is not None:
                        logger.save_checkpoint(
                            _build_checkpoint_payload(
                                pinn_model=pinn_model,
                                metrics=row,
                                config=config,
                                tag=milestone_tag,
                                vrba_config=vrba_config,
                                vrba_state=vrba_state,
                            ),
                            tag=milestone_tag,
                        )
                    if _checkpointing_enabled(config, "save_last", True):
                        logger.save_checkpoint(
                            _build_checkpoint_payload(
                                pinn_model=pinn_model,
                                metrics=row,
                                config=config,
                                tag="last",
                                vrba_config=vrba_config,
                                vrba_state=vrba_state,
                            ),
                            tag="last",
                        )
                selection_metric = row.val_total_loss if row.val_total_loss is not None else row.train_total_loss
                if best_metric is None or selection_metric < best_metric:
                    best_metric = selection_metric
                    if logger is not None and _checkpointing_enabled(config, "save_best", True):
                        logger.save_checkpoint(
                            _build_checkpoint_payload(
                                pinn_model=pinn_model,
                                metrics=row,
                                config=config,
                                tag="best",
                                vrba_config=vrba_config,
                                vrba_state=vrba_state,
                            ),
                            tag="best",
                        )
            if _collocation_phase_boundary_enabled(config):
                collocation_manager.handle_phase_boundary(
                context=CollocationStrategyContext(
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
            _cache_optimizer_state(
                optimizer_spec=optimizer_spec,
                phase=phase,
                transition_settings=transition_settings,
                transition_state=transition_state,
            )

        ensemble.eval()

    if ensemble is None:
        raise RuntimeError("Multistage PINN training did not instantiate any stages.")
    final_model = _assemble_multistage_pinn_model(
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


def train_pinn(
    *,
    dataset: PinnDatasetBundle,
    ode_model: Any,
    config: Any,
    logger: PinnLogger | None = None,
) -> tuple[PinnModel, list[EpochMetrics]]:
    if _pinn_mode(config) == "multistage":
        return _train_multistage_pinn(dataset=dataset, ode_model=ode_model, config=config, logger=logger)

    seed = int(cfg_get(config, "model.seed", 0))
    configure_reproducibility_from_config(seed=seed, config=config, prefix="pinn")

    dtype = resolve_torch_dtype(str(cfg_get(config, "pinn.dtype", "float64")))
    device = select_torch_device(str(cfg_get(config, "pinn.device", "auto")))
    hidden_dim = int(cfg_get(config, "pinn.hidden_dim", 64))
    hidden_layers = int(cfg_get(config, "pinn.hidden_layers", 4))
    activation = str(cfg_get(config, "pinn.activation", "tanh"))
    formulation = str(cfg_get(config, "pinn.formulation", "odequations"))
    optimizer_phases = load_optimizer_phases(config)
    base_weights = _loss_weights_from_config(config)
    loss_weight_schedule = _loss_weight_schedule_from_config(config)
    weighting_config = weighting_config_from_config(config)
    vrba_config = vrba_config_from_config(config)
    if loss_weight_schedule is not None and weighting_config.scheme != "static":
        raise ValueError("pinn.loss_weight_schedule currently supports static weighting only. Set pinn.weighting.scheme=static.")
    weighting_policy = build_weighting_policy(weighting_config)
    weighting_state = weighting_policy.initial_state(base_weights)
    gradient_telemetry_enabled = bool(cfg_get(config, "pinn.gradient_telemetry.enabled", False))
    cost_tracking_enabled = _cost_tracking_enabled(config)
    criterion = nn.MSELoss()
    transition_settings = _optimizer_transition_settings(config)
    transition_state = OptimizerTransitionState(cache={})

    model = TimeConditionedMLP(
        input_dim=dataset.input_dim,
        output_dim=dataset.output_dim,
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
        activation=activation,
    ).to(device=device, dtype=dtype)

    dataset = move_pinn_training_data_to_device(dataset, device=device, dtype=dtype)
    collocation_manager = build_collocation_manager(
        initial_points=dataset.train_col_x,
        init_x=dataset.train_init_x,
        init_y=dataset.train_init_y,
        config=config,
    )
    initial_pool_batch = collocation_manager.initial_epoch_batch()
    dataset = PinnDatasetBundle(
        train_x=dataset.train_x,
        train_y=dataset.train_y,
        train_col_x=initial_pool_batch.x_col,
        train_init_x=initial_pool_batch.x_init,
        train_init_y=initial_pool_batch.y_init,
        train_col_weights=initial_pool_batch.x_col_weights,
        val_x=dataset.val_x,
        val_y=dataset.val_y,
        val_col_x=dataset.val_col_x,
        test_x=dataset.test_x,
        test_y=dataset.test_y,
    )
    vrba_state = initialize_vrba_state(
        vrba_config,
        initial_point_counts={
            "data": int(dataset.train_x.shape[0]),
            "dt": int(dataset.train_x.shape[0]),
            "physics": int(dataset.train_col_x.shape[0]),
            "ic": int(dataset.train_init_x.shape[0]),
        },
    )
    weight_probe_batch = _build_weight_update_probe_batch(
        dataset=dataset,
        weighting_config=weighting_config,
    )

    pinn_model = PinnModel(
        model=model,
        device=device,
        dtype=dtype,
        input_dim=dataset.input_dim,
        output_dim=dataset.output_dim,
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
        activation=activation,
    )

    rows: list[EpochMetrics] = []
    best_metric: float | None = None
    global_epoch = 0
    train_started = time.perf_counter()
    total_epochs = int(sum(phase.epochs for phase in optimizer_phases))
    checkpoint_milestones = _resolve_checkpoint_milestones(config, total_epochs)

    if logger is not None and _checkpointing_enabled(config, "save_init", True):
        logger.save_checkpoint(
            _build_checkpoint_payload(
                pinn_model=pinn_model,
                metrics=None,
                config=config,
                tag="init",
                vrba_config=vrba_config,
                vrba_state=vrba_state,
            ),
            tag="init",
        )

    for phase_idx, phase in enumerate(optimizer_phases):
        optimizer_spec = build_optimizer(
            model=model,
            optimizer_name=phase.optimizer,
            lr=phase.lr,
            optimizer_kwargs=phase.optimizer_kwargs,
            line_search=phase.line_search,
        )
        phase_full_batch = _phase_effective_full_batch(phase, optimizer_spec)
        if phase_full_batch and phase.batch_size is not None:
            raise ValueError(f"Optimizer phase '{phase.name}' is full-batch and must set batch_size=null.")
        phase_allows_sampling = _phase_allows_sampling(phase, optimizer_spec)
        if phase_full_batch and phase_allows_sampling:
            raise ValueError(f"Optimizer phase '{phase.name}' is full-batch and must disable sampling for a stable objective.")

        if phase_full_batch:
            phase_batch_size = None
        else:
            if not optimizer_spec.supports_minibatch:
                raise ValueError(f"Optimizer '{phase.optimizer}' does not support minibatch execution.")
            phase_batch_size = phase.batch_size if phase.batch_size is not None else int(cfg_get(config, "pinn.default_batch_size", 1024))
        phase_supports_dynamic_weight_updates = _phase_supports_dynamic_weight_updates(phase)
        refresh_on_phase_start = _phase_boundary_refresh_planned(config, phase_name=phase.name, event="start")
        transition_diagnostics = _maybe_restore_optimizer_state(
            optimizer_spec=optimizer_spec,
            phase=phase,
            config=config,
            transition_settings=transition_settings,
            transition_state=transition_state,
            refresh_on_phase_start=refresh_on_phase_start,
        )

        if _collocation_phase_boundary_enabled(config):
            collocation_manager.handle_phase_boundary(
                context=CollocationStrategyContext(
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
            epoch_started = time.perf_counter()
            _reset_peak_memory_if_cuda(device, enabled=cost_tracking_enabled)
            model.train()
            epoch_losses: list[PinnLossBreakdown] = []
            epoch_gradients: list[GradientTelemetry] = []
            capture_gradient_telemetry = gradient_telemetry_enabled
            scheduled_base_weights = _scheduled_loss_weights(
                base_weights=base_weights,
                schedule=loss_weight_schedule,
                next_global_epoch=global_epoch + 1,
            )
            active_weights = scheduled_base_weights if weighting_config.scheme == "static" else weighting_policy.current_weights(weighting_state)
            if phase_allows_sampling:
                epoch_train_x, epoch_train_y = _sample_supervised_rows(dataset.train_x, dataset.train_y, config)
            else:
                epoch_train_x, epoch_train_y = dataset.train_x, dataset.train_y
            epoch_pool_batch = collocation_manager.prepare_epoch_batch(
                context=CollocationStrategyContext(
                    global_epoch=global_epoch + 1,
                    phase_name=phase.name,
                    phase_allows_sampling=phase_allows_sampling,
                    phase_is_full_batch=phase_full_batch,
                    phase_epoch=epoch,
                    refresh_event=None,
                    model=model,
                    ode_model=ode_model,
                    formulation=formulation,
                )
            )
            epoch_train_col_x = epoch_pool_batch.x_col
            epoch_train_col_weights = epoch_pool_batch.x_col_weights
            active_collocation_weights = _active_collocation_weights_or_none(
                vrba_config=vrba_config,
                x_col_weights=epoch_train_col_weights,
                weighting_enabled=bool(collocation_manager.state.metadata.get("residual_vrba_weighting_enabled", False)),
            )
            epoch_train_init_x = epoch_pool_batch.x_init
            epoch_train_init_y = epoch_pool_batch.y_init
            num_batches = 0
            num_train_steps = 0
            num_supervised_rows = 0
            num_collocation_rows = 0
            num_init_rows = 0

            if phase_batch_size is None:
                breakdown, gradient_telemetry = _train_pinn_step(
                    model=model,
                    optimizer_spec=optimizer_spec,
                    criterion=criterion,
                    ode_model=ode_model,
                    formulation=formulation,
                    weights=active_weights,
                    x_data=epoch_train_x,
                    y_data=epoch_train_y,
                    x_col=epoch_train_col_x,
                    x_col_weights=active_collocation_weights,
                    x_init=epoch_train_init_x,
                    y_init=epoch_train_init_y,
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
            else:
                _steps, epoch_batches = _iter_minibatch_batches(
                    dataset=PinnDatasetBundle(
                        train_x=epoch_train_x,
                        train_y=epoch_train_y,
                        train_col_x=epoch_train_col_x,
                        train_col_weights=active_collocation_weights,
                        train_init_x=epoch_train_init_x,
                        train_init_y=epoch_train_init_y,
                    ),
                    batch_size=phase_batch_size,
                    shuffle=phase.shuffle,
                    seed=seed + phase_idx + global_epoch,
                )
                for batch in epoch_batches:
                    num_batches += 1
                    num_train_steps += 1
                    num_supervised_rows += int(batch.x_data.shape[0])
                    num_collocation_rows += int(batch.x_col.shape[0])
                    num_init_rows += int(batch.x_init.shape[0])
                    breakdown, gradient_telemetry = _train_pinn_step(
                        model=model,
                        optimizer_spec=optimizer_spec,
                        criterion=criterion,
                        ode_model=ode_model,
                        formulation=formulation,
                        weights=active_weights,
                        x_data=batch.x_data,
                        y_data=batch.y_data,
                        x_col=batch.x_col,
                        x_col_weights=batch.x_col_weights,
                        x_init=batch.x_init,
                        y_init=batch.y_init,
                        capture_gradient_telemetry=capture_gradient_telemetry,
                    )
                    epoch_losses.append(breakdown)
                    if gradient_telemetry is not None:
                        epoch_gradients.append(gradient_telemetry)

            epoch_wall_seconds = float(time.perf_counter() - epoch_started) if cost_tracking_enabled else None
            cumulative_wall_seconds = float(time.perf_counter() - train_started) if cost_tracking_enabled else None
            peak_gpu_memory_allocated_bytes, peak_gpu_memory_reserved_bytes = _peak_memory_if_cuda(
                device,
                enabled=cost_tracking_enabled,
            )
            train_total = float(torch.stack([x.total.detach() for x in epoch_losses]).mean().item())
            train_component_losses = {
                name: float(torch.stack([loss.components[name].detach() for loss in epoch_losses]).mean().item())
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
            weighting_updated = False
            weight_update_stats = None
            if phase_supports_dynamic_weight_updates and weighting_policy.should_update(epoch=epoch, global_epoch=global_epoch):
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
                        _build_weight_update_probe_batch(
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
                        x_col=probe_batch.x_col,
                        x_col_weights=None,
                        x_init=probe_batch.x_init,
                        y_init=probe_batch.y_init,
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
                    weight_update_stats = _compute_ntk_weight_update_stats(
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
                        x_col=weight_probe_batch.x_col,
                        x_col_weights=None,
                        x_init=weight_probe_batch.x_init,
                        y_init=weight_probe_batch.y_init,
                        create_graph=True,
                    )
                    weight_update_stats = _compute_weight_update_stats(
                        model=model,
                        losses=probe_losses,
                        anchor_component=weighting_config.anchor,
                        epoch=epoch,
                        global_epoch=global_epoch,
                    )
                weighting_state = weighting_policy.update(weighting_state, weight_update_stats)
                weighting_updated = True
            val_total_loss = None
            val_component_losses = None
            test_metrics = None
            if _should_run_evaluation(global_epoch, config):
                val_component_losses = {}
                val_component_losses["data"] = _evaluate_data_loss(model=model, x=dataset.val_x, y=dataset.val_y, criterion=criterion)
                val_component_losses["dt"] = _evaluate_dt_loss(
                    model=model,
                    x=dataset.val_x,
                    y=dataset.val_y,
                    ode_model=ode_model,
                    criterion=criterion,
                    formulation=formulation,
                )
                val_component_losses["physics"] = _evaluate_physics_loss(
                    model=model,
                    x_rows=dataset.val_col_x if dataset.val_col_x is not None else dataset.val_x,
                    ode_model=ode_model,
                    criterion=criterion,
                    formulation=formulation,
                )
                val_total_loss = _compute_weighted_total_loss(val_component_losses, active_weights)
                test_metrics = {
                    "data_loss": _evaluate_data_loss(model=model, x=dataset.test_x, y=dataset.test_y, criterion=criterion)
                }
            vrba_summary = _collocation_vrba_summary(collocation_manager)
            row = EpochMetrics(
                epoch=epoch,
                global_epoch=global_epoch,
                phase_name=phase.name,
                optimizer=phase.optimizer,
                train_total_loss=train_total,
                train_component_losses=train_component_losses,
                train_total_grad_norm=train_total_grad_norm,
                train_component_grad_norms=train_component_grad_norms,
                train_weighted_component_grad_norms=train_weighted_component_grad_norms,
                val_total_loss=val_total_loss,
                val_component_losses=val_component_losses,
                test_metrics=test_metrics,
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
                vrba_enabled=bool(vrba_summary["enabled"]),
                vrba_sampling_enabled=bool(vrba_summary["sampling_enabled"]),
                vrba_weighting_enabled=bool(vrba_summary["weighting_enabled"]),
                vrba_potential=vrba_summary["potential"],
                vrba_target_sets=vrba_summary["target_sets"],
                vrba_update_count=None if vrba_summary["update_count"] is None else int(vrba_summary["update_count"]),
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
                    "requires_closure": optimizer_spec.requires_closure,
                    "full_batch": phase_full_batch,
                    "sampling_enabled": phase_allows_sampling,
                    "line_search": optimizer_spec.line_search_name,
                    "dynamic_weight_updates_enabled": phase_supports_dynamic_weight_updates,
                    **transition_diagnostics,
                    **_collocation_pool_diagnostics(collocation_manager),
                    **(
                        optimizer_spec.optimizer.get_last_diagnostics()
                        if hasattr(optimizer_spec.optimizer, "get_last_diagnostics")
                        else {}
                    ),
                },
            )
            mean_epoch_breakdown = PinnLossBreakdown(
                total=epoch_losses[0].total.detach().new_tensor(train_total),
                components={
                    name: epoch_losses[0].components[name].detach().new_tensor(train_component_losses[name])
                    for name in epoch_losses[0].components
                },
            )
            rows.append(row)
            collocation_manager.observe_epoch_losses(
                global_epoch=global_epoch,
                losses=mean_epoch_breakdown,
            )

            if logger is not None:
                logger.write_metrics(rows)
                logger.print_epoch_metrics(row)
                logger.log_epoch_metrics(row)
                milestone_tag = checkpoint_milestones.get(global_epoch)
                if milestone_tag is not None:
                    logger.save_checkpoint(
                            _build_checkpoint_payload(
                                pinn_model=pinn_model,
                                metrics=row,
                                config=config,
                                tag=milestone_tag,
                                vrba_config=vrba_config,
                                vrba_state=vrba_state,
                            ),
                        tag=milestone_tag,
                    )
                if _checkpointing_enabled(config, "save_last", True):
                    logger.save_checkpoint(
                            _build_checkpoint_payload(
                                pinn_model=pinn_model,
                                metrics=row,
                                config=config,
                                tag="last",
                                vrba_config=vrba_config,
                                vrba_state=vrba_state,
                            ),
                        tag="last",
                    )
            selection_metric = row.val_total_loss if row.val_total_loss is not None else row.train_total_loss
            if best_metric is None or selection_metric < best_metric:
                best_metric = selection_metric
                if logger is not None and _checkpointing_enabled(config, "save_best", True):
                    logger.save_checkpoint(
                            _build_checkpoint_payload(
                                pinn_model=pinn_model,
                                metrics=row,
                                config=config,
                                tag="best",
                                vrba_config=vrba_config,
                                vrba_state=vrba_state,
                            ),
                        tag="best",
                    )
        if _collocation_phase_boundary_enabled(config):
            collocation_manager.handle_phase_boundary(
                context=CollocationStrategyContext(
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
        _cache_optimizer_state(
            optimizer_spec=optimizer_spec,
            phase=phase,
            transition_settings=transition_settings,
            transition_state=transition_state,
        )

    model.eval()
    return pinn_model, rows


def evaluate(model: SurrogateModel, dataset: TrajectoryDataset) -> dict[str, float]:
    x_test, y_test = dataset.test_view()
    if x_test is None:
        return {}
    pred = model.predict(x_test)
    mse = float(np.mean((pred - y_test) ** 2))
    rmse = float(np.sqrt(mse))
    return {"mse": mse, "rmse": rmse}
