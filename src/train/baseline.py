"""Backward-compatible baseline training exports."""

from __future__ import annotations

from src.data.loaders.trajectory_dataset import TrajectoryDataset
from src.train.trainer import BaselineConfig, SurrogateModel, evaluate, train_baseline_surrogate


def evaluate_baseline(model: SurrogateModel, dataset: TrajectoryDataset) -> dict[str, float]:
    return evaluate(model=model, dataset=dataset)
