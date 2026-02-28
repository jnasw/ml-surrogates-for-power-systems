"""Deep ensemble utilities for Query-by-Committee."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import numpy as np

from src.data.loaders.trajectory_dataset import TrajectoryDataset
from src.train.trainer import SurrogateModel, train_surrogate


@dataclass
class DeepEnsemble:
    members: list[SurrogateModel]

    def predict_all(self, x: np.ndarray, batch_size: int = 2048) -> np.ndarray:
        preds = [m.predict(x, batch_size=batch_size) for m in self.members]
        return np.stack(preds, axis=0)

    def predict_mean(self, x: np.ndarray, batch_size: int = 2048) -> np.ndarray:
        return self.predict_all(x, batch_size=batch_size).mean(axis=0)


def train_ensemble(dataset: TrajectoryDataset, M: int, base_seed: int, config: Any) -> DeepEnsemble:
    if M <= 0:
        raise ValueError("M must be positive.")
    members = []
    for i in range(M):
        members.append(train_surrogate(dataset, seed=base_seed + i, config=config))
    return DeepEnsemble(members=members)


def save_ensemble_checkpoint(ensemble: DeepEnsemble, folder: str, tag: str) -> str:
    ckpt_dir = os.path.join(folder, f"ensemble_{tag}")
    os.makedirs(ckpt_dir, exist_ok=True)
    for i, member in enumerate(ensemble.members):
        member.save_checkpoint(os.path.join(ckpt_dir, f"member_{i:03d}.pt"))
    return ckpt_dir


def load_ensemble_checkpoint(folder: str, tag: str, device_preference: str = "auto") -> DeepEnsemble:
    ckpt_dir = os.path.join(folder, f"ensemble_{tag}")
    if not os.path.exists(ckpt_dir):
        raise FileNotFoundError(f"Ensemble checkpoint not found: {ckpt_dir}")
    member_files = sorted([f for f in os.listdir(ckpt_dir) if f.startswith("member_") and f.endswith(".pt")])
    if not member_files:
        raise FileNotFoundError(f"No ensemble member checkpoints in: {ckpt_dir}")
    members = [
        SurrogateModel.load_checkpoint(os.path.join(ckpt_dir, mf), device_preference=device_preference)
        for mf in member_files
    ]
    return DeepEnsemble(members=members)
