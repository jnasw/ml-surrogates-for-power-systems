"""Baseline model wrappers, training, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import os
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.loaders.trajectory_dataset import TrajectoryDataset
from src.training.device import select_torch_device
from src.training.model import BaselineTrajectoryMLP, TrajectoryMLP
from src.training.runtime import cfg_get, configure_reproducibility, configure_reproducibility_from_config


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
    device: str = "auto"


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
        use_cuda = bool(cfg_get(config, "surrogate.use_cuda", True))
        device_pref = "auto" if use_cuda else "cpu"
    device_pref = str(device_pref)
    if bool(cfg_get(config, "surrogate.require_explicit_device", False)) and device_pref.lower() == "auto":
        raise ValueError("surrogate.require_explicit_device=true requires surrogate.device to be one of: cpu/cuda/mps.")
    device = select_torch_device(device_pref)
    model = TrajectoryMLP(
        input_dim=x_np.shape[1],
        output_dim=y_flat.shape[1],
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
    ).to(device)

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


def evaluate(model: SurrogateModel, dataset: TrajectoryDataset) -> dict[str, float]:
    x_test, y_test = dataset.test_view()
    if x_test is None:
        return {}
    pred = model.predict(x_test)
    mse = float(np.mean((pred - y_test) ** 2))
    rmse = float(np.sqrt(mse))
    return {"mse": mse, "rmse": rmse}


def evaluate_baseline(model: SurrogateModel, dataset: TrajectoryDataset) -> dict[str, float]:
    return evaluate(model=model, dataset=dataset)
