"""PINN dataset loading utilities for stage-2 HDF5 outputs."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Iterable

import h5py
import torch
from torch.utils.data import Dataset

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
from src.pinn.runtime import resolve_torch_dtype


def _iter_split_files(split_dir: str, suffix: str) -> Iterable[str]:
    if not os.path.exists(split_dir):
        return []
    files = []
    for fn in sorted(os.listdir(split_dir)):
        if not fn.endswith(H5_FILE_SUFFIX):
            continue
        if suffix == H5_DATA_SUFFIX:
            if (
                H5_DATA_SUFFIX not in fn
                or H5_COLLOCATION_SUFFIX in fn
                or H5_INIT_SUFFIX in fn
            ):
                continue
        elif suffix not in fn:
            continue
        files.append(os.path.join(split_dir, fn))
    return files


def _load_h5_tensor_rows(paths: Iterable[str], dataset_key: str, dtype: torch.dtype) -> torch.Tensor | None:
    parts = []
    for path in paths:
        with h5py.File(path, "r") as h5f:
            parts.append(torch.tensor(h5f[dataset_key][:], dtype=dtype))
    if not parts:
        return None
    return torch.cat(parts, dim=0)


class TensorRowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same first dimension.")
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class TensorInputDataset(Dataset[torch.Tensor]):
    def __init__(self, x: torch.Tensor):
        self.x = x

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.x[idx]


@dataclass(frozen=True)
class PinnDatasetBundle:
    train_x: torch.Tensor
    train_y: torch.Tensor
    train_col_x: torch.Tensor
    train_init_x: torch.Tensor
    train_init_y: torch.Tensor
    train_col_weights: torch.Tensor | None = None
    train_init_weights: torch.Tensor | None = None
    val_x: torch.Tensor | None = None
    val_y: torch.Tensor | None = None
    val_col_x: torch.Tensor | None = None
    test_x: torch.Tensor | None = None
    test_y: torch.Tensor | None = None

    @property
    def input_dim(self) -> int:
        return int(self.train_x.shape[1])

    @property
    def output_dim(self) -> int:
        return int(self.train_y.shape[1])

    @property
    def train_dataset(self) -> TensorRowDataset:
        return TensorRowDataset(self.train_x, self.train_y)

    @property
    def collocation_dataset(self) -> TensorInputDataset:
        return TensorInputDataset(self.train_col_x)

    @property
    def init_dataset(self) -> TensorRowDataset:
        return TensorRowDataset(self.train_init_x, self.train_init_y)

    @property
    def val_dataset(self) -> TensorRowDataset | None:
        if self.val_x is None or self.val_y is None:
            return None
        return TensorRowDataset(self.val_x, self.val_y)

    @property
    def test_dataset(self) -> TensorRowDataset | None:
        if self.test_x is None or self.test_y is None:
            return None
        return TensorRowDataset(self.test_x, self.test_y)


def load_pinn_dataset_from_preprocessed_root(
    dataset_root: str,
    dtype: str = "float32",
    allow_missing_collocation: bool = False,
) -> PinnDatasetBundle:
    if not os.path.exists(dataset_root):
        raise FileNotFoundError(f"Dataset folder not found: {dataset_root}")

    torch_dtype = resolve_torch_dtype(dtype)

    train_dir = os.path.join(dataset_root, TRAIN_SPLIT)
    val_dir = os.path.join(dataset_root, VAL_SPLIT)
    test_dir = os.path.join(dataset_root, TEST_SPLIT)

    train_x = _load_h5_tensor_rows(_iter_split_files(train_dir, H5_DATA_SUFFIX), H5_X_KEYS[TRAIN_SPLIT], torch_dtype)
    train_y = _load_h5_tensor_rows(_iter_split_files(train_dir, H5_DATA_SUFFIX), H5_Y_KEYS[TRAIN_SPLIT], torch_dtype)
    train_col_x = _load_h5_tensor_rows(_iter_split_files(train_dir, H5_COLLOCATION_SUFFIX), H5_X_KEYS[TRAIN_SPLIT], torch_dtype)
    train_init_x = _load_h5_tensor_rows(_iter_split_files(train_dir, H5_INIT_SUFFIX), H5_X_KEYS[TRAIN_SPLIT], torch_dtype)
    train_init_y = _load_h5_tensor_rows(_iter_split_files(train_dir, H5_INIT_SUFFIX), H5_Y_KEYS[TRAIN_SPLIT], torch_dtype)

    if train_x is None or train_y is None:
        raise ValueError(f"No supervised training HDF5 files found in {train_dir}")
    if train_col_x is None and not allow_missing_collocation:
        raise ValueError(f"No collocation HDF5 files found in {train_dir}")
    if train_init_x is None or train_init_y is None:
        raise ValueError(f"No initial-condition HDF5 files found in {train_dir}")
    if train_col_x is None:
        train_col_x = torch.empty((0, train_x.shape[1]), dtype=torch_dtype)

    val_x = _load_h5_tensor_rows(_iter_split_files(val_dir, H5_DATA_SUFFIX), H5_X_KEYS[VAL_SPLIT], torch_dtype)
    val_y = _load_h5_tensor_rows(_iter_split_files(val_dir, H5_DATA_SUFFIX), H5_Y_KEYS[VAL_SPLIT], torch_dtype)
    val_col_x = _load_h5_tensor_rows(_iter_split_files(val_dir, H5_COLLOCATION_SUFFIX), H5_X_KEYS[VAL_SPLIT], torch_dtype)
    test_x = _load_h5_tensor_rows(_iter_split_files(test_dir, H5_DATA_SUFFIX), H5_X_KEYS[TEST_SPLIT], torch_dtype)
    test_y = _load_h5_tensor_rows(_iter_split_files(test_dir, H5_DATA_SUFFIX), H5_Y_KEYS[TEST_SPLIT], torch_dtype)

    return PinnDatasetBundle(
        train_x=train_x,
        train_y=train_y,
        train_col_x=train_col_x,
        train_init_x=train_init_x,
        train_init_y=train_init_y,
        val_x=val_x,
        val_y=val_y,
        val_col_x=val_col_x,
        test_x=test_x,
        test_y=test_y,
    )


def load_pinn_dataset_from_preprocessed(
    dataset_dir: str,
    model_flag: str,
    dataset_number: int,
    dtype: str = "float32",
    allow_missing_collocation: bool = False,
) -> PinnDatasetBundle:
    dataset_root = os.path.join(dataset_dir, model_flag, f"dataset_v{dataset_number}")
    return load_pinn_dataset_from_preprocessed_root(
        dataset_root=dataset_root,
        dtype=dtype,
        allow_missing_collocation=allow_missing_collocation,
    )
