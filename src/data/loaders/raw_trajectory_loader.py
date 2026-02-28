"""Load trajectory-level datasets from raw simulation files."""

from __future__ import annotations

import os
import pickle
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from src.data.loaders.trajectory_dataset import TrajectoryDataset


def _load_output_dim(init_conditions_dir: str, model_flag: str) -> int:
    guide_path = os.path.join(init_conditions_dir, "modellings_guide.yaml")
    guide = OmegaConf.load(guide_path)
    for entry in guide:
        if entry.get("name") == model_flag:
            return len(entry.get("keys"))
    raise ValueError(f"Model '{model_flag}' not found in modeling guide.")


def load_trajectory_dataset_from_raw(
    dataset_dir: str,
    init_conditions_dir: str,
    model_flag: str,
    dataset_number: int,
    split_ratio: float = 0.8,
    shuffle: bool = False,
    seed: int | None = None,
) -> TrajectoryDataset:
    """Load IC->trajectory samples from dataset_vN/raw.

    IC is extracted as the first time-step values of all channels (excluding time row).
    Target trajectory contains only dynamic model states (keys) over time.
    """
    root = os.path.join(dataset_dir, model_flag, f"dataset_v{dataset_number}")
    raw_dir = os.path.join(root, "raw")
    if not os.path.exists(raw_dir):
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

    output_dim = _load_output_dim(init_conditions_dir=init_conditions_dir, model_flag=model_flag)

    trajectories: list[np.ndarray] = []
    for fn in sorted([f for f in os.listdir(raw_dir) if f.endswith(".pkl")]):
        with open(os.path.join(raw_dir, fn), "rb") as f:
            payload = pickle.load(f)
        for traj in payload:
            trajectories.append(np.asarray(traj, dtype=np.float32))

    if not trajectories:
        raise ValueError(f"No raw trajectories found in {raw_dir}")

    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(trajectories)

    ics = []
    ys = []
    for arr in trajectories:
        # arr: [time; channels...] with shape (1 + F, T)
        ic = arr[1:, 0].astype(np.float32)
        y = arr[1 : 1 + output_dim, :].T.astype(np.float32)  # (T, S)
        ics.append(ic)
        ys.append(y)

    x = np.stack(ics, axis=0)
    y = np.stack(ys, axis=0)
    n = x.shape[0]
    n_train = int(n * split_ratio)
    if n_train <= 0 or n_train >= n:
        raise ValueError("split_ratio must produce non-empty train and test sets.")

    return TrajectoryDataset(
        train_ics=x[:n_train],
        train_trajs=y[:n_train],
        test_ics=x[n_train:],
        test_trajs=y[n_train:],
        time_grid=trajectories[0][0].astype(np.float32),
    )


def load_train_arrays_from_raw(
    dataset_dir: str,
    init_conditions_dir: str,
    model_flag: str,
    dataset_number: int,
    shuffle: bool = False,
    seed: int | None = None,
    max_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load all raw trajectories as train arrays.

    Returns:
        ics: (N, D)
        trajs: (N, T, S)
        time_grid: (T,)
    """
    root = os.path.join(dataset_dir, model_flag, f"dataset_v{dataset_number}")
    raw_dir = os.path.join(root, "raw")
    if not os.path.exists(raw_dir):
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

    output_dim = _load_output_dim(init_conditions_dir=init_conditions_dir, model_flag=model_flag)

    trajectories: list[np.ndarray] = []
    for fn in sorted([f for f in os.listdir(raw_dir) if f.endswith(".pkl")]):
        with open(os.path.join(raw_dir, fn), "rb") as f:
            payload = pickle.load(f)
        for traj in payload:
            trajectories.append(np.asarray(traj, dtype=np.float32))

    if not trajectories:
        raise ValueError(f"No raw trajectories found in {raw_dir}")

    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(trajectories)

    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive.")
        trajectories = trajectories[:max_samples]

    ics = []
    ys = []
    for arr in trajectories:
        ics.append(arr[1:, 0].astype(np.float32))
        ys.append(arr[1 : 1 + output_dim, :].T.astype(np.float32))

    x = np.stack(ics, axis=0)
    y = np.stack(ys, axis=0)
    t = trajectories[0][0].astype(np.float32)
    return x, y, t
