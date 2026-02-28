"""Load trajectory datasets from stage-2 preprocessed HDF5 files."""

from __future__ import annotations

import os
from typing import Iterable

import h5py
import numpy as np

from src.data.loaders.trajectory_dataset import TrajectoryDataset
from src.data.contracts.data_contract import (
    H5_COLLOCATION_SUFFIX,
    H5_DATA_SUFFIX,
    H5_FILE_SUFFIX,
    H5_INIT_SUFFIX,
    H5_X_KEYS,
    H5_Y_KEYS,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VAL_SPLIT,
)


def _split_xy_rows_to_trajectories(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert flattened preprocessed rows back into per-trajectory tensors.

    The preprocessed format stores each trajectory as contiguous rows where time is
    strictly increasing. A new trajectory starts when time resets.
    """
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("Expected 2D arrays for x/y rows.")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y row counts must match.")
    if x.shape[0] == 0:
        raise ValueError("Cannot decode empty split rows.")
    if x.shape[1] < 2:
        raise ValueError("x must contain at least time + one IC column.")

    t = x[:, 0]
    resets = np.where(t[1:] <= t[:-1])[0] + 1
    boundaries = np.concatenate(([0], resets, [len(t)]))

    ics: list[np.ndarray] = []
    trajs: list[np.ndarray] = []
    time_grid: np.ndarray | None = None
    expected_len: int | None = None
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end <= start:
            continue
        seg_x = x[start:end]
        seg_y = y[start:end]
        if expected_len is None:
            expected_len = int(seg_x.shape[0])
            time_grid = seg_x[:, 0].astype(np.float32).copy()
        elif seg_x.shape[0] != expected_len:
            raise ValueError(
                "Inconsistent trajectory lengths in preprocessed data; "
                "baseline loader expects fixed-length trajectories."
            )
        ics.append(seg_x[0, 1:].astype(np.float32))
        trajs.append(seg_y.astype(np.float32))

    if not ics:
        raise ValueError("No trajectories decoded from preprocessed rows.")
    return np.stack(ics, axis=0), np.stack(trajs, axis=0), time_grid


def _iter_split_files(split_dir: str) -> Iterable[str]:
    for fn in sorted(os.listdir(split_dir)):
        if (
            fn.endswith(H5_FILE_SUFFIX)
            and H5_DATA_SUFFIX in fn
            and H5_COLLOCATION_SUFFIX not in fn
            and H5_INIT_SUFFIX not in fn
        ):
            yield os.path.join(split_dir, fn)


def _load_split(dataset_root: str, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split_dir = os.path.join(dataset_root, split)
    if not os.path.exists(split_dir):
        raise FileNotFoundError(f"Preprocessed split directory not found: {split_dir}")

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for path in _iter_split_files(split_dir):
        with h5py.File(path, "r") as h5f:
            x_parts.append(np.asarray(h5f[H5_X_KEYS[split]], dtype=np.float32))
            y_parts.append(np.asarray(h5f[H5_Y_KEYS[split]], dtype=np.float32))

    if not x_parts:
        raise ValueError(f"No preprocessed supervised files found in {split_dir}")
    x_rows = np.concatenate(x_parts, axis=0)
    y_rows = np.concatenate(y_parts, axis=0)
    return _split_xy_rows_to_trajectories(x_rows, y_rows)


def load_trajectory_dataset_from_preprocessed(
    dataset_dir: str,
    model_flag: str,
    dataset_number: int,
    include_val_in_train: bool = False,
) -> TrajectoryDataset:
    """Load TrajectoryDataset from `dataset_vN/{train,val,test}` HDF5 outputs."""
    dataset_root = os.path.join(dataset_dir, model_flag, f"dataset_v{dataset_number}")
    return load_trajectory_dataset_from_preprocessed_root(
        dataset_root=dataset_root,
        include_val_in_train=include_val_in_train,
    )


def load_trajectory_dataset_from_preprocessed_root(
    dataset_root: str,
    include_val_in_train: bool = False,
) -> TrajectoryDataset:
    """Load TrajectoryDataset from an explicit dataset root path."""
    if not os.path.exists(dataset_root):
        raise FileNotFoundError(f"Dataset folder not found: {dataset_root}")

    x_train, y_train, t_train = _load_split(dataset_root=dataset_root, split=TRAIN_SPLIT)
    if include_val_in_train:
        val_dir = os.path.join(dataset_root, VAL_SPLIT)
        if os.path.exists(val_dir):
            x_val, y_val, t_val = _load_split(dataset_root=dataset_root, split=VAL_SPLIT)
            if t_val.shape != t_train.shape or not np.allclose(t_val, t_train):
                raise ValueError("Time grids differ between train and val splits.")
            x_train = np.concatenate((x_train, x_val), axis=0)
            y_train = np.concatenate((y_train, y_val), axis=0)

    x_test, y_test, t_test = _load_split(dataset_root=dataset_root, split=TEST_SPLIT)
    if t_test.shape != t_train.shape or not np.allclose(t_test, t_train):
        raise ValueError("Time grids differ between train and test splits.")
    return TrajectoryDataset(
        train_ics=x_train,
        train_trajs=y_train,
        test_ics=x_test,
        test_trajs=y_test,
        time_grid=t_train,
    )
