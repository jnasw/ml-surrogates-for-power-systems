"""In-memory trajectory dataset for iterative active learning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def _ensure_2d(a: np.ndarray, name: str) -> np.ndarray:
    if a.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {a.shape}")
    return a


def _ensure_3d(a: np.ndarray, name: str) -> np.ndarray:
    if a.ndim != 3:
        raise ValueError(f"{name} must be 3D, got shape {a.shape}")
    return a


@dataclass
class TrajectoryDataset:
    """Container with mutable train set and immutable test set."""

    train_ics: np.ndarray
    train_trajs: np.ndarray
    test_ics: Optional[np.ndarray] = None
    test_trajs: Optional[np.ndarray] = None
    time_grid: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.train_ics = _ensure_2d(np.asarray(self.train_ics, dtype=np.float32), "train_ics")
        self.train_trajs = _ensure_3d(np.asarray(self.train_trajs, dtype=np.float32), "train_trajs")
        if self.train_ics.shape[0] != self.train_trajs.shape[0]:
            raise ValueError("train_ics and train_trajs must have the same first dimension.")

        if self.test_ics is not None:
            self.test_ics = _ensure_2d(np.asarray(self.test_ics, dtype=np.float32), "test_ics")
        if self.test_trajs is not None:
            self.test_trajs = _ensure_3d(np.asarray(self.test_trajs, dtype=np.float32), "test_trajs")
        if (self.test_ics is None) != (self.test_trajs is None):
            raise ValueError("test_ics and test_trajs must both be set or both be None.")
        if self.test_ics is not None and self.test_ics.shape[0] != self.test_trajs.shape[0]:
            raise ValueError("test_ics and test_trajs must have the same first dimension.")

        if self.time_grid is not None:
            self.time_grid = np.asarray(self.time_grid, dtype=np.float32)
            if self.time_grid.ndim != 1:
                raise ValueError("time_grid must be 1D.")
            if self.time_grid.shape[0] != self.train_trajs.shape[1]:
                raise ValueError("time_grid length must match trajectory time dimension.")

    @property
    def n_train(self) -> int:
        return int(self.train_ics.shape[0])

    @property
    def n_test(self) -> int:
        return 0 if self.test_ics is None else int(self.test_ics.shape[0])

    @property
    def ic_dim(self) -> int:
        return int(self.train_ics.shape[1])

    @property
    def traj_shape(self) -> tuple[int, int]:
        return int(self.train_trajs.shape[1]), int(self.train_trajs.shape[2])

    def append(self, new_ics: np.ndarray, new_trajs: np.ndarray) -> None:
        new_ics = _ensure_2d(np.asarray(new_ics, dtype=np.float32), "new_ics")
        new_trajs = _ensure_3d(np.asarray(new_trajs, dtype=np.float32), "new_trajs")
        if new_ics.shape[0] != new_trajs.shape[0]:
            raise ValueError("new_ics and new_trajs must have the same first dimension.")
        if new_ics.shape[1] != self.ic_dim:
            raise ValueError("new_ics ic dimension mismatch.")
        if new_trajs.shape[1:] != self.train_trajs.shape[1:]:
            raise ValueError("new_trajs trajectory shape mismatch.")

        self.train_ics = np.concatenate((self.train_ics, new_ics), axis=0)
        self.train_trajs = np.concatenate((self.train_trajs, new_trajs), axis=0)

    def training_view(self) -> tuple[np.ndarray, np.ndarray]:
        return self.train_ics.copy(), self.train_trajs.copy()

    def test_view(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self.test_ics is None:
            return None, None
        return self.test_ics.copy(), self.test_trajs.copy()
