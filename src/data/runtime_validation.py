"""Early validation for runtime dependencies on preprocessed datasets."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import h5py

from src.data.contracts.data_contract import (
    H5_COLLOCATION_SUFFIX,
    H5_DATA_SUFFIX,
    H5_DIFFICULTY_BIN_KEYS,
    H5_DIFFICULTY_SCORE_KEYS,
    H5_FILE_SUFFIX,
    H5_INIT_SUFFIX,
    H5_TRAJECTORY_ID_KEYS,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VAL_SPLIT,
)


def _split_files(split_dir: str, suffix: str) -> list[str]:
    if not os.path.exists(split_dir):
        return []
    files: list[str] = []
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


def _dataset_root_hint(dataset_root: str) -> str:
    hints: list[str] = []
    info_path = os.path.join(dataset_root, "info.txt")
    raw_dir = os.path.join(dataset_root, "raw")
    has_preprocessed_split = any(
        os.path.exists(os.path.join(dataset_root, split))
        for split in (TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT)
    )
    if (os.path.exists(info_path) or os.path.exists(raw_dir)) and not has_preprocessed_split:
        hints.append(
            "The dataset root looks like a raw stage-1 dataset. "
            "Run 01_preprocess_dataset.py first or use artifacts.preprocessed_root from dataset_manifest.json."
        )

    parent = Path(dataset_root).parent
    if parent.exists():
        candidates = sorted(
            p.name
            for p in parent.iterdir()
            if p.is_dir() and p.name.startswith("dataset_v")
        )
        if candidates:
            hints.append(f"Available dataset versions under '{parent}': {', '.join(candidates)}.")

    return " ".join(hints)


def _raise_missing(dataset_root: str, requirement: str, details: Iterable[str]) -> None:
    detail_text = " ".join(str(item) for item in details if item)
    hint = _dataset_root_hint(dataset_root)
    message = f"{requirement} Dataset root: {dataset_root}."
    if detail_text:
        message += f" {detail_text}"
    if hint:
        message += f" {hint}"
    raise ValueError(message)


def _require_metadata_keys(dataset_root: str, paths: list[str], keys: list[str], context: str) -> None:
    missing: list[str] = []
    for key in keys:
        for path in paths:
            with h5py.File(path, "r") as h5f:
                if key not in h5f:
                    missing.append(f"{os.path.basename(path)} missing '{key}'")
    if missing:
        _raise_missing(
            dataset_root,
            requirement=f"{context} requires preprocessing metadata keys {', '.join(keys)}.",
            details=missing,
        )


def validate_baseline_runtime_dataset(dataset_root: str) -> None:
    """Validate the minimal preprocessed artifacts required by the baseline runtime."""
    if not os.path.exists(dataset_root):
        raise FileNotFoundError(f"Dataset folder not found: {dataset_root}")

    train_dir = os.path.join(dataset_root, TRAIN_SPLIT)
    test_dir = os.path.join(dataset_root, TEST_SPLIT)
    train_files = _split_files(train_dir, H5_DATA_SUFFIX)
    test_files = _split_files(test_dir, H5_DATA_SUFFIX)
    if not train_files:
        _raise_missing(
            dataset_root,
            requirement="Baseline runtime requires supervised train HDF5 files under 'train/'.",
            details=[f"Expected files containing '{H5_DATA_SUFFIX}' in {train_dir}."],
        )
    if not test_files:
        _raise_missing(
            dataset_root,
            requirement="Baseline runtime requires supervised test HDF5 files under 'test/'.",
            details=[
                f"Expected files containing '{H5_DATA_SUFFIX}' in {test_dir}.",
                "Smoke runs should still retain an internal test split.",
            ],
        )


def validate_pinn_runtime_dataset(
    dataset_root: str,
    *,
    collocation_mode: str,
    curriculum_enabled: bool,
    supervised_acquisition_enabled: bool = False,
) -> list[str]:
    """Validate the minimal preprocessed artifacts required by the PINN runtime."""
    if not os.path.exists(dataset_root):
        raise FileNotFoundError(f"Dataset folder not found: {dataset_root}")

    notes: list[str] = []
    train_dir = os.path.join(dataset_root, TRAIN_SPLIT)
    val_dir = os.path.join(dataset_root, VAL_SPLIT)
    test_dir = os.path.join(dataset_root, TEST_SPLIT)
    train_supervised = _split_files(train_dir, H5_DATA_SUFFIX)
    train_init = _split_files(train_dir, H5_INIT_SUFFIX)
    train_collocation = _split_files(train_dir, H5_COLLOCATION_SUFFIX)
    val_supervised = _split_files(val_dir, H5_DATA_SUFFIX)
    test_supervised = _split_files(test_dir, H5_DATA_SUFFIX)

    if not train_supervised:
        _raise_missing(
            dataset_root,
            requirement="PINN runtime requires supervised train HDF5 files under 'train/'.",
            details=[f"Expected files containing '{H5_DATA_SUFFIX}' in {train_dir}."],
        )
    if not train_init:
        _raise_missing(
            dataset_root,
            requirement="PINN runtime requires initial-condition HDF5 files under 'train/'.",
            details=[
                f"Expected files containing '{H5_INIT_SUFFIX}' in {train_dir}.",
                "Generated-collocation mode still depends on preprocessing-time init exports.",
            ],
        )
    normalized_mode = str(collocation_mode).strip().lower()
    if normalized_mode == "preprocessed" and not train_collocation:
        _raise_missing(
            dataset_root,
            requirement="PINN collocation mode 'preprocessed' requires collocation HDF5 files under 'train/'.",
            details=[f"Expected files containing '{H5_COLLOCATION_SUFFIX}' in {train_dir}."],
        )
    if normalized_mode == "generated" and not train_collocation:
        notes.append(
            "No preprocessed collocation files found under 'train/'. This is allowed because "
            "pinn.collocation.mode=generated, but supervised train and init artifacts are still required."
        )

    if curriculum_enabled:
        _require_metadata_keys(
            dataset_root,
            train_supervised,
            keys=[
                H5_DIFFICULTY_SCORE_KEYS[TRAIN_SPLIT],
                H5_DIFFICULTY_BIN_KEYS[TRAIN_SPLIT],
            ],
            context="pinn.curriculum.enabled=true",
        )
    if supervised_acquisition_enabled:
        _require_metadata_keys(
            dataset_root,
            train_supervised,
            keys=[H5_TRAJECTORY_ID_KEYS[TRAIN_SPLIT]],
            context="pinn.supervised_acquisition.enabled=true",
        )

    if not val_supervised:
        notes.append(
            "No supervised validation split found under 'val/'. This is allowed; validation losses and "
            "validation-based scheduler signals will be unavailable for this run "
            "(common in smoke runs with dataset.validation_flag=false)."
        )
    if not test_supervised:
        notes.append(
            "No supervised test split found under 'test/'. This is allowed for startup, but final test metrics "
            "will be unavailable for this run."
        )
    return notes
