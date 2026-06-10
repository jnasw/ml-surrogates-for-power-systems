"""Shared helpers for thesis analysis notebooks.

The notebooks in this directory should export compact thesis table data under
``results/tables``. Plots can be displayed inline in notebooks, but generated
figure files should stay out of the committed results tree.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


RESULTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = RESULTS_DIR.parent
TABLES_DIR = RESULTS_DIR / "tables"

EXPERIMENT_TABLE_DIRS: Mapping[str, str] = {
    "dataset_generation": "01_dataset_generation",
    "optimizer_comparison": "02_optimizer_comparison",
    "loss_balancing": "03_loss_balancing",
    "collocation_comparison": "04_collocation_comparison",
    "multistage": "05_multistage",
    "data_augmentation": "06_data_augmentation",
    "final_experiment": "07_final_experiment",
    "across_experiments": "08_across_experiments",
}


def table_dir(experiment: str) -> Path:
    """Return the canonical table export directory for an experiment.

    Args:
        experiment: Either a short experiment key, such as
            ``"dataset_generation"``, or a canonical directory name, such as
            ``"01_dataset_generation"``.
    """

    dirname = EXPERIMENT_TABLE_DIRS.get(experiment, experiment)
    path = TABLES_DIR / dirname
    path.mkdir(parents=True, exist_ok=True)
    return path


def table_path(experiment: str, name: str = "main", suffix: str = ".csv") -> Path:
    """Return the canonical path for a table export."""

    if not suffix.startswith("."):
        suffix = "." + suffix
    clean_name = name[:-len(suffix)] if name.endswith(suffix) else name
    return table_dir(experiment) / f"{clean_name}{suffix}"


def write_table(frame: Any, experiment: str, name: str = "main", **to_csv_kwargs: Any) -> Path:
    """Write a dataframe-like object to the canonical table export path.

    ``frame`` must provide a ``to_csv`` method. The helper defaults to
    ``index=False`` to keep notebook exports compact and thesis-friendly.
    """

    path = table_path(experiment, name=name)
    kwargs = {"index": False}
    kwargs.update(to_csv_kwargs)
    frame.to_csv(path, **kwargs)
    return path


def write_numeric_table(frame: Any, experiment: str, name: str = "main_numeric", **to_csv_kwargs: Any) -> Path:
    """Write the numeric companion table for an experiment."""

    return write_table(frame, experiment=experiment, name=name, **to_csv_kwargs)

