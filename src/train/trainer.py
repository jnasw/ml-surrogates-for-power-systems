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
from src.pinn.logging import EpochMetrics, PinnLogger
from src.pinn.losses import LossWeights, PinnLossBreakdown, compute_pinn_losses
from src.pinn.optim import build_optimizer
from src.pinn.residuals import compute_residual_terms
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
        device = select_torch_device(device_preference)
        payload = torch.load(path, map_location=device)
        dtype = resolve_torch_dtype(str(payload["dtype"]))
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
        physics=float(cfg_get(config, "pinn.loss_weights.physics", 1.0)),
        ic=float(cfg_get(config, "pinn.loss_weights.ic", 1.0)),
    )


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
    terms = compute_residual_terms(model=model, x=x_rows, ode_model=ode_model, formulation=formulation)
    return float(criterion(terms.residual, torch.zeros_like(terms.residual)).item())


def _build_checkpoint_payload(
    *,
    pinn_model: PinnModel,
    metrics: EpochMetrics,
    config: Any,
) -> dict[str, Any]:
    return {
        "state_dict": pinn_model.model.state_dict(),
        "input_dim": pinn_model.input_dim,
        "output_dim": pinn_model.output_dim,
        "hidden_dim": pinn_model.hidden_dim,
        "hidden_layers": pinn_model.hidden_layers,
        "activation": pinn_model.activation,
        "dtype": torch_dtype_name(pinn_model.dtype),
        "metrics": metrics,
        "config": OmegaConf.to_container(config, resolve=True),
    }


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
) -> PinnLossBreakdown:
    breakdown_box: dict[str, PinnLossBreakdown] = {}

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        pred = model(x_data)
        collocation_terms = compute_residual_terms(
            model=model,
            x=x_col,
            ode_model=ode_model,
            formulation=formulation,
        )
        init_pred = model(x_init)
        losses = compute_pinn_losses(
            criterion=criterion,
            supervised_prediction=pred,
            supervised_target=y_data,
            collocation_terms=collocation_terms,
            init_prediction=init_pred,
            init_target=y_init,
            weights=weights,
        )
        losses.total.backward()
        breakdown_box["losses"] = losses
        return losses.total

    if isinstance(optimizer, torch.optim.LBFGS):
        optimizer.step(closure)
    else:
        closure()
        optimizer.step()
    return breakdown_box["losses"]


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
    criterion = nn.MSELoss()

    model = TimeConditionedMLP(
        input_dim=dataset.input_dim,
        output_dim=dataset.output_dim,
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
        activation=activation,
    ).to(device=device, dtype=dtype)

    dataset = PinnDatasetBundle(
        train_x=dataset.train_x.to(device=device, dtype=dtype),
        train_y=dataset.train_y.to(device=device, dtype=dtype),
        train_col_x=dataset.train_col_x.to(device=device, dtype=dtype),
        train_init_x=dataset.train_init_x.to(device=device, dtype=dtype),
        train_init_y=dataset.train_init_y.to(device=device, dtype=dtype),
        val_x=None if dataset.val_x is None else dataset.val_x.to(device=device, dtype=dtype),
        val_y=None if dataset.val_y is None else dataset.val_y.to(device=device, dtype=dtype),
        val_col_x=None if dataset.val_col_x is None else dataset.val_col_x.to(device=device, dtype=dtype),
        test_x=None if dataset.test_x is None else dataset.test_x.to(device=device, dtype=dtype),
        test_y=None if dataset.test_y is None else dataset.test_y.to(device=device, dtype=dtype),
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

            if stage_batch_size is None:
                breakdown = _train_pinn_step(
                    model=model,
                    optimizer=optimizer,
                    criterion=criterion,
                    ode_model=ode_model,
                    formulation=formulation,
                    weights=weights,
                    x_data=dataset.train_x,
                    y_data=dataset.train_y,
                    x_col=dataset.train_col_x,
                    x_init=dataset.train_init_x,
                    y_init=dataset.train_init_y,
                )
                epoch_losses.append(breakdown)
            else:
                for (x_data, y_data), x_col, (x_init, y_init) in zip(
                    train_batches,
                    cycle(col_batches),
                    cycle(init_batches),
                ):
                    breakdown = _train_pinn_step(
                        model=model,
                        optimizer=optimizer,
                        criterion=criterion,
                        ode_model=ode_model,
                        formulation=formulation,
                        weights=weights,
                        x_data=x_data,
                        y_data=y_data,
                        x_col=x_col,
                        x_init=x_init,
                        y_init=y_init,
                    )
                    epoch_losses.append(breakdown)

            train_total = float(torch.stack([x.total.detach() for x in epoch_losses]).mean().item())
            train_data = float(torch.stack([x.data.detach() for x in epoch_losses]).mean().item())
            train_physics = float(torch.stack([x.physics.detach() for x in epoch_losses]).mean().item())
            train_ic = float(torch.stack([x.ic.detach() for x in epoch_losses]).mean().item())
            val_data = _evaluate_data_loss(model=model, x=dataset.val_x, y=dataset.val_y, criterion=criterion)
            val_physics = _evaluate_physics_loss(
                model=model,
                x_rows=dataset.val_col_x if dataset.val_col_x is not None else dataset.val_x,
                ode_model=ode_model,
                criterion=criterion,
                formulation=formulation,
            )
            test_data = _evaluate_data_loss(model=model, x=dataset.test_x, y=dataset.test_y, criterion=criterion)

            global_epoch += 1
            row = EpochMetrics(
                epoch=epoch,
                global_epoch=global_epoch,
                stage_name=stage.name,
                optimizer=stage.optimizer,
                train_total_loss=train_total,
                train_data_loss=train_data,
                train_physics_loss=train_physics,
                train_ic_loss=train_ic,
                val_data_loss=val_data,
                val_physics_loss=val_physics,
                test_data_loss=test_data,
            )
            rows.append(row)

            if logger is not None:
                logger.write_metrics(rows)
                logger.log_epoch_metrics(row)
                logger.save_checkpoint(_build_checkpoint_payload(pinn_model=pinn_model, metrics=row, config=config), tag="last")

            selection_metric = row.val_data_loss if row.val_data_loss is not None else row.train_total_loss
            if best_metric is None or selection_metric < best_metric:
                best_metric = selection_metric
                if logger is not None:
                    logger.save_checkpoint(_build_checkpoint_payload(pinn_model=pinn_model, metrics=row, config=config), tag="best")

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
