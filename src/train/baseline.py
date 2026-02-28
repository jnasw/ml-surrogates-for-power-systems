"""Fixed baseline surrogate architecture and training routine."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.loaders.trajectory_dataset import TrajectoryDataset
from src.train.device import select_torch_device
from src.train.trainer import SurrogateModel, evaluate


class BaselineTrajectoryMLP(nn.Module):
    """Robust baseline MLP for IC -> flattened trajectory."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, hidden_layers: int, dropout: float):
        super().__init__()
        self.inp = nn.Linear(input_dim, hidden_dim)
        blocks = []
        for _ in range(hidden_layers):
            blocks.append(nn.Linear(hidden_dim, hidden_dim))
            blocks.append(nn.LayerNorm(hidden_dim))
            blocks.append(nn.SiLU())
            if dropout > 0:
                blocks.append(nn.Dropout(dropout))
        self.blocks = nn.Sequential(*blocks)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.inp(x)
        return self.out(self.blocks(h))


@dataclass
class BaselineConfig:
    hidden_dim: int = 256
    hidden_layers: int = 4
    dropout: float = 0.05
    lr: float = 1e-3
    batch_size: int = 64
    epochs: int = 300
    device: str = "auto"  # auto | cuda | mps | cpu


def train_baseline_surrogate(dataset: TrajectoryDataset, seed: int, cfg: BaselineConfig) -> SurrogateModel:
    torch.manual_seed(seed)
    np.random.seed(seed)

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
    loader = DataLoader(TensorDataset(x, y), batch_size=cfg.batch_size, shuffle=True)

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


def evaluate_baseline(model: SurrogateModel, dataset: TrajectoryDataset) -> dict[str, float]:
    return evaluate(model=model, dataset=dataset)
