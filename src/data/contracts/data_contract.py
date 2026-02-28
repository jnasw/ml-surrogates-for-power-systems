"""Canonical data contract for the dataset generation pipeline.

This module defines file naming, required metadata keys, and expected tensor
layouts for both raw simulation output and preprocessed training datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


RAW_FILE_PREFIX = "file"
RAW_FILE_SUFFIX = ".pkl"
INFO_FILE_NAME = "info.txt"

TRAIN_SPLIT = "train"
VAL_SPLIT = "val"
TEST_SPLIT = "test"
SPLITS = (TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT)

H5_DATA_SUFFIX = "_data"
H5_COLLOCATION_SUFFIX = "_data_col"
H5_INIT_SUFFIX = "_data_init"
H5_FILE_SUFFIX = ".h5"

H5_X_KEYS = {
    TRAIN_SPLIT: "x_train",
    VAL_SPLIT: "x_val",
    TEST_SPLIT: "x_test",
}
H5_Y_KEYS = {
    TRAIN_SPLIT: "y_train",
    VAL_SPLIT: "y_val",
    TEST_SPLIT: "y_test",
}

INFO_REQUIRED_KEYS = (
    "Number of files",
    "Initial conditions file",
    "Number of different simulated trajectories",
    "Time horizon of the simulations",
    "Number of points in the each simulation",
)


@dataclass(frozen=True)
class RawTrajectoryLayout:
    """Layout for one raw trajectory record."""

    row0: str = "time vector t"
    rows1plus: str = "simulated channels in model-defined order"


@dataclass(frozen=True)
class SupervisedLayout:
    """Layout for one supervised sample row in preprocessed datasets."""

    x: str = "[time, initial-condition/features...]"
    y: str = "[dynamic target states...]"


RAW_TRAJECTORY_LAYOUT = RawTrajectoryLayout()
SUPERVISED_LAYOUT = SupervisedLayout()


def validate_info_lines(lines: Sequence[str]) -> None:
    """Validate minimal info.txt structure.

    Raises:
        ValueError: If one of the required keys is missing in the file.
    """

    present_keys = {line.split(":", 1)[0].strip() for line in lines if ":" in line}
    missing = [key for key in INFO_REQUIRED_KEYS if key not in present_keys]
    if missing:
        raise ValueError(
            "info.txt is missing required keys: " + ", ".join(missing)
        )


def validate_split_name(split: str) -> None:
    """Validate split name for h5 serialization."""

    if split not in SPLITS:
        raise ValueError(f"Invalid split '{split}'. Expected one of {SPLITS}.")


def validate_raw_trajectory(trajectory: Iterable[Sequence[float]]) -> None:
    """Validate raw trajectory record shape assumptions.

    Expected structure for each trajectory:
    - First row is a time vector.
    - All rows have equal length.
    - At least two rows exist (time + one channel).
    """

    rows = list(trajectory)
    if len(rows) < 2:
        raise ValueError("Raw trajectory must contain at least time + one channel.")
    expected_len = len(rows[0])
    if expected_len == 0:
        raise ValueError("Time vector in raw trajectory cannot be empty.")
    for idx, row in enumerate(rows):
        if len(row) != expected_len:
            raise ValueError(
                f"Raw trajectory row {idx} has length {len(row)}; "
                f"expected {expected_len}."
            )
