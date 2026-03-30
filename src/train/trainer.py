"""Surrogate training and evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import cycle
from typing import Any

import numpy as np
import os
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from omegaconf import OmegaConf

from src.data.loaders.trajectory_dataset import TrajectoryDataset
from src.pinn.data import PinnDatasetBundle
from src.pinn.evaluator import evaluate_pinn_loss_breakdown, move_pinn_dataset_to_device
from src.pinn.logging import EpochMetrics, PinnLogger
from src.pinn.losses import LossWeights, PinnLossBreakdown
from src.pinn.optim import build_optimizer
from src.pinn.residuals import compute_residual_terms, compute_supervised_dt_terms
from src.pinn.runtime import load_optimizer_stages, resolve_torch_dtype, torch_dtype_name
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


@dataclass
class PinnModel:
    model: TimeConditionedMLP
    device: torch.device
    dtype: torch.dtype
    input_dim: int
    output_dim: int
    hidden_dim: int
    hidden_layers: int
    activation: str

    def save_checkpoint(self, path: str) -> None:
        payload = {
            "state_dict": self.model.state_dict(),
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_dim": self.hidden_dim,
            "hidden_layers": self.hidden_layers,
            "activation": self.activation,
            "dtype": torch_dtype_name(self.dtype),
        }
        torch.save(payload, path)

    @staticmethod
    def load_checkpoint(path: str, device_preference: str = "auto") -> "PinnModel":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        dtype = resolve_torch_dtype(str(payload["dtype"]))
        device = select_torch_device(device_preference)
        if device.type == "mps" and dtype == torch.float64:
            device = torch.device("cpu")
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
        )


@dataclass(frozen=True)
class GradientTelemetry:
    total_grad_norm: float
    data_grad_norm: float
    dt_grad_norm: float
    physics_grad_norm: float
    ic_grad_norm: float


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
    return LossWeights(
        data=float(cfg_get(config, "pinn.loss_weights.data", 1.0)),
        dt=float(cfg_get(config, "pinn.loss_weights.dt", 1.0e-4)),
        physics=float(cfg_get(config, "pinn.loss_weights.physics", 1.0)),
        ic=float(cfg_get(config, "pinn.loss_weights.ic", 1.0)),
    )


def _sample_collocation_rows(x_col: torch.Tensor, config: Any) -> torch.Tensor:
    if not bool(cfg_get(config, "pinn.collocation_sampling.enabled", False)):
        return x_col
    return _sample_tensor_rows(
        x=x_col,
        rows_per_epoch_cfg=cfg_get(config, "pinn.collocation_sampling.rows_per_epoch", None),
        fraction_per_epoch_cfg=cfg_get(config, "pinn.collocation_sampling.fraction_per_epoch", None),
        cfg_prefix="pinn.collocation_sampling",
    )


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


def _should_run_evaluation(global_epoch: int, config: Any) -> bool:
    frequency = int(cfg_get(config, "pinn.evaluation.frequency", 1))
    if frequency < 1:
        frequency = 1
    return (int(global_epoch) % frequency) == 0


def _evaluate_data_loss(
    model: nn.Module,
    x: torch.Tensor | None,
    y: torch.Tensor | None,
    criterion: nn.Module,
) -> float | None:
    if x is None or y is None:
        return None
    model.eval()
    with torch.no_grad():
        pred = model(x)
        return float(criterion(pred, y).item())


def _evaluate_physics_loss(
    model: nn.Module,
    x_rows: torch.Tensor | None,
    ode_model: Any,
    criterion: nn.Module,
    formulation: str,
) -> float | None:
    if x_rows is None:
        return None
    model.eval()
    terms = compute_residual_terms(
        model=model,
        x=x_rows,
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
    model.eval()
    terms = compute_supervised_dt_terms(
        model=model,
        x=x,
        y_true=y,
        ode_model=ode_model,
        formulation=formulation,
        create_graph=False,
    )
    return float(criterion(terms.residual, torch.zeros_like(terms.residual)).item())


def _build_checkpoint_payload(
    *,
    pinn_model: PinnModel,
    metrics: EpochMetrics | None,
    config: Any,
    tag: str,
) -> dict[str, Any]:
    return {
        "state_dict": pinn_model.model.state_dict(),
        "input_dim": pinn_model.input_dim,
        "output_dim": pinn_model.output_dim,
        "hidden_dim": pinn_model.hidden_dim,
        "hidden_layers": pinn_model.hidden_layers,
        "activation": pinn_model.activation,
        "dtype": torch_dtype_name(pinn_model.dtype),
        "checkpoint_tag": tag,
        "metrics": metrics,
        "config": OmegaConf.to_container(config, resolve=True),
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


def _compute_gradient_telemetry(
    *,
    model: nn.Module,
    loss_total: torch.Tensor,
    loss_data: torch.Tensor,
    loss_dt: torch.Tensor,
    loss_physics: torch.Tensor,
    loss_ic: torch.Tensor,
) -> GradientTelemetry:
    return GradientTelemetry(
        total_grad_norm=_grad_norm(loss_total, model, retain_graph=True),
        data_grad_norm=_grad_norm(loss_data, model, retain_graph=True),
        dt_grad_norm=_grad_norm(loss_dt, model, retain_graph=True),
        physics_grad_norm=_grad_norm(loss_physics, model, retain_graph=True),
        ic_grad_norm=_grad_norm(loss_ic, model, retain_graph=True),
    )


def _train_pinn_step(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    ode_model: Any,
    formulation: str,
    weights: LossWeights,
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    x_col: torch.Tensor,
    x_init: torch.Tensor,
    y_init: torch.Tensor,
    capture_gradient_telemetry: bool = False,
) -> tuple[PinnLossBreakdown, GradientTelemetry | None]:
    breakdown_box: dict[str, PinnLossBreakdown] = {}
    telemetry_box: dict[str, GradientTelemetry | None] = {"telemetry": None}

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
            x_init=x_init,
            y_init=y_init,
            create_graph=True,
        )
        if capture_gradient_telemetry:
            telemetry_box["telemetry"] = _compute_gradient_telemetry(
                model=model,
                loss_total=losses.total,
                loss_data=losses.data,
                loss_dt=losses.dt,
                loss_physics=losses.physics,
                loss_ic=losses.ic,
            )
        losses.total.backward()
        breakdown_box["losses"] = losses
        return losses.total

    if isinstance(optimizer, torch.optim.LBFGS):
        optimizer.step(closure)
    else:
        closure()
        optimizer.step()
    return breakdown_box["losses"], telemetry_box["telemetry"]


def train_pinn(
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
    stages = load_optimizer_stages(config)
    weights = _loss_weights_from_config(config)
    gradient_telemetry_enabled = bool(cfg_get(config, "pinn.gradient_telemetry.enabled", False))
    criterion = nn.MSELoss()

    model = TimeConditionedMLP(
        input_dim=dataset.input_dim,
        output_dim=dataset.output_dim,
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
        activation=activation,
    ).to(device=device, dtype=dtype)

    dataset = move_pinn_dataset_to_device(dataset, device=device, dtype=dtype)

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
    total_epochs = int(sum(stage.epochs for stage in stages))
    checkpoint_milestones = _resolve_checkpoint_milestones(config, total_epochs)

    if logger is not None and _checkpointing_enabled(config, "save_init", True):
        logger.save_checkpoint(
            _build_checkpoint_payload(
                pinn_model=pinn_model,
                metrics=None,
                config=config,
                tag="init",
            ),
            tag="init",
        )

    for stage_idx, stage in enumerate(stages):
        optimizer = build_optimizer(model=model, optimizer_name=stage.optimizer, lr=stage.lr)
        if stage.optimizer.lower() == "lbfgs":
            stage_batch_size = None
        else:
            stage_batch_size = stage.batch_size if stage.batch_size is not None else int(cfg_get(config, "pinn.default_batch_size", 1024))

        if stage_batch_size is None:
            train_batches = [(dataset.train_x, dataset.train_y)]
            col_batches = [dataset.train_col_x]
            init_batches = [(dataset.train_init_x, dataset.train_init_y)]
        else:
            data_gen = torch.Generator()
            data_gen.manual_seed(seed + stage_idx)
            train_loader = DataLoader(dataset.train_dataset, batch_size=stage_batch_size, shuffle=stage.shuffle, generator=data_gen)
            col_loader = DataLoader(dataset.collocation_dataset, batch_size=stage_batch_size, shuffle=stage.shuffle, generator=data_gen)
            init_loader = DataLoader(dataset.init_dataset, batch_size=stage_batch_size, shuffle=stage.shuffle, generator=data_gen)
            train_batches = train_loader
            col_batches = col_loader
            init_batches = init_loader

        for epoch in range(1, stage.epochs + 1):
            model.train()
            epoch_losses: list[PinnLossBreakdown] = []
            epoch_gradients: list[GradientTelemetry] = []
            capture_gradient_telemetry = gradient_telemetry_enabled
            epoch_train_x, epoch_train_y = _sample_supervised_rows(dataset.train_x, dataset.train_y, config)
            epoch_train_col_x = _sample_collocation_rows(dataset.train_col_x, config)

            if stage_batch_size is None:
                breakdown, gradient_telemetry = _train_pinn_step(
                    model=model,
                    optimizer=optimizer,
                    criterion=criterion,
                    ode_model=ode_model,
                    formulation=formulation,
                    weights=weights,
                    x_data=epoch_train_x,
                    y_data=epoch_train_y,
                    x_col=epoch_train_col_x,
                    x_init=dataset.train_init_x,
                    y_init=dataset.train_init_y,
                    capture_gradient_telemetry=capture_gradient_telemetry,
                )
                epoch_losses.append(breakdown)
                if gradient_telemetry is not None:
                    epoch_gradients.append(gradient_telemetry)
            else:
                data_gen = torch.Generator()
                data_gen.manual_seed(seed + stage_idx + global_epoch)
                train_loader = DataLoader(
                    TensorDataset(epoch_train_x, epoch_train_y),
                    batch_size=stage_batch_size,
                    shuffle=stage.shuffle,
                    generator=data_gen,
                )
                col_loader = DataLoader(
                    TensorDataset(epoch_train_col_x),
                    batch_size=stage_batch_size,
                    shuffle=stage.shuffle,
                    generator=data_gen,
                )
                init_loader = DataLoader(
                    dataset.init_dataset,
                    batch_size=stage_batch_size,
                    shuffle=stage.shuffle,
                    generator=data_gen,
                )
                train_batches = train_loader
                col_batches = col_loader
                init_batches = init_loader
                for (x_data, y_data), x_col, (x_init, y_init) in zip(
                    train_batches,
                    cycle(col_batches),
                    cycle(init_batches),
                ):
                    breakdown, gradient_telemetry = _train_pinn_step(
                        model=model,
                        optimizer=optimizer,
                        criterion=criterion,
                        ode_model=ode_model,
                        formulation=formulation,
                        weights=weights,
                        x_data=x_data,
                        y_data=y_data,
                        x_col=x_col[0],
                        x_init=x_init,
                        y_init=y_init,
                        capture_gradient_telemetry=capture_gradient_telemetry,
                    )
                    epoch_losses.append(breakdown)
                    if gradient_telemetry is not None:
                        epoch_gradients.append(gradient_telemetry)

            train_total = float(torch.stack([x.total.detach() for x in epoch_losses]).mean().item())
            train_data = float(torch.stack([x.data.detach() for x in epoch_losses]).mean().item())
            train_dt = float(torch.stack([x.dt.detach() for x in epoch_losses]).mean().item())
            train_physics = float(torch.stack([x.physics.detach() for x in epoch_losses]).mean().item())
            train_ic = float(torch.stack([x.ic.detach() for x in epoch_losses]).mean().item())
            train_total_grad_norm = None
            train_data_grad_norm = None
            train_dt_grad_norm = None
            train_physics_grad_norm = None
            train_ic_grad_norm = None
            if epoch_gradients:
                train_total_grad_norm = float(np.mean([item.total_grad_norm for item in epoch_gradients]))
                train_data_grad_norm = float(np.mean([item.data_grad_norm for item in epoch_gradients]))
                train_dt_grad_norm = float(np.mean([item.dt_grad_norm for item in epoch_gradients]))
                train_physics_grad_norm = float(np.mean([item.physics_grad_norm for item in epoch_gradients]))
                train_ic_grad_norm = float(np.mean([item.ic_grad_norm for item in epoch_gradients]))
            global_epoch += 1
            val_data = None
            val_dt = None
            val_physics = None
            test_data = None
            if _should_run_evaluation(global_epoch, config):
                val_data = _evaluate_data_loss(model=model, x=dataset.val_x, y=dataset.val_y, criterion=criterion)
                val_dt = _evaluate_dt_loss(
                    model=model,
                    x=dataset.val_x,
                    y=dataset.val_y,
                    ode_model=ode_model,
                    criterion=criterion,
                    formulation=formulation,
                )
                val_physics = _evaluate_physics_loss(
                    model=model,
                    x_rows=dataset.val_col_x if dataset.val_col_x is not None else dataset.val_x,
                    ode_model=ode_model,
                    criterion=criterion,
                    formulation=formulation,
                )
                test_data = _evaluate_data_loss(model=model, x=dataset.test_x, y=dataset.test_y, criterion=criterion)
            row = EpochMetrics(
                epoch=epoch,
                global_epoch=global_epoch,
                stage_name=stage.name,
                optimizer=stage.optimizer,
                train_total_loss=train_total,
                train_data_loss=train_data,
                train_dt_loss=train_dt,
                train_physics_loss=train_physics,
                train_ic_loss=train_ic,
                train_total_grad_norm=train_total_grad_norm,
                train_data_grad_norm=train_data_grad_norm,
                train_dt_grad_norm=train_dt_grad_norm,
                train_physics_grad_norm=train_physics_grad_norm,
                train_ic_grad_norm=train_ic_grad_norm,
                val_data_loss=val_data,
                val_dt_loss=val_dt,
                val_physics_loss=val_physics,
                test_data_loss=test_data,
            )
            rows.append(row)

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
                        ),
                        tag="last",
                    )

            selection_metric = row.val_data_loss if row.val_data_loss is not None else row.train_total_loss
            if best_metric is None or selection_metric < best_metric:
                best_metric = selection_metric
                if logger is not None and _checkpointing_enabled(config, "save_best", True):
                    logger.save_checkpoint(
                        _build_checkpoint_payload(
                            pinn_model=pinn_model,
                            metrics=row,
                            config=config,
                            tag="best",
                        ),
                        tag="best",
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
