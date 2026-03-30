"""Deterministic analysis subsets for PINN evaluation and loss landscapes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.pinn.data import PinnDatasetBundle


ANALYSIS_SPLITS = {"train", "val", "test"}


@dataclass(frozen=True)
class PinnAnalysisBundle:
    """A fixed evaluation bundle for one analysis split."""

    split: str
    x_data: torch.Tensor | None
    y_data: torch.Tensor | None
    x_col: torch.Tensor | None
    x_init: torch.Tensor | None
    y_init: torch.Tensor | None
    supervised_rows: int
    collocation_rows: int
    init_rows: int
    seed: int

    @property
    def has_supervised(self) -> bool:
        return self.x_data is not None and self.y_data is not None

    @property
    def has_collocation(self) -> bool:
        return self.x_col is not None

    @property
    def has_init(self) -> bool:
        return self.x_init is not None and self.y_init is not None

    def as_manifest(self) -> dict[str, int | str | bool]:
        return {
            "split": self.split,
            "seed": int(self.seed),
            "supervised_rows": int(self.supervised_rows),
            "collocation_rows": int(self.collocation_rows),
            "init_rows": int(self.init_rows),
            "has_supervised": bool(self.has_supervised),
            "has_collocation": bool(self.has_collocation),
            "has_init": bool(self.has_init),
        }


def move_analysis_bundle_to_device(
    bundle: PinnAnalysisBundle,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> PinnAnalysisBundle:
    """Move an analysis bundle to a device/dtype."""
    return PinnAnalysisBundle(
        split=bundle.split,
        x_data=None if bundle.x_data is None else bundle.x_data.to(device=device, dtype=dtype),
        y_data=None if bundle.y_data is None else bundle.y_data.to(device=device, dtype=dtype),
        x_col=None if bundle.x_col is None else bundle.x_col.to(device=device, dtype=dtype),
        x_init=None if bundle.x_init is None else bundle.x_init.to(device=device, dtype=dtype),
        y_init=None if bundle.y_init is None else bundle.y_init.to(device=device, dtype=dtype),
        supervised_rows=int(bundle.supervised_rows),
        collocation_rows=int(bundle.collocation_rows),
        init_rows=int(bundle.init_rows),
        seed=int(bundle.seed),
    )


def _sample_rows(x: torch.Tensor, rows: int | None, seed: int) -> torch.Tensor:
    total_rows = int(x.shape[0])
    if rows in (None, "null"):
        return x

    target_rows = int(rows)
    if target_rows <= 0:
        raise ValueError("Requested analysis rows must be > 0.")
    if target_rows >= total_rows:
        return x

    indices_np = np.random.default_rng(seed).choice(total_rows, size=target_rows, replace=False)
    indices = torch.as_tensor(indices_np, device=x.device, dtype=torch.long)
    return x.index_select(0, indices)


def _sample_xy_rows(
    x: torch.Tensor,
    y: torch.Tensor,
    rows: int | None,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    total_rows = int(x.shape[0])
    if total_rows != int(y.shape[0]):
        raise ValueError("x and y must have the same number of rows.")
    if rows in (None, "null"):
        return x, y

    target_rows = int(rows)
    if target_rows <= 0:
        raise ValueError("Requested analysis rows must be > 0.")
    if target_rows >= total_rows:
        return x, y

    indices_np = np.random.default_rng(seed).choice(total_rows, size=target_rows, replace=False)
    indices = torch.as_tensor(indices_np, device=x.device, dtype=torch.long)
    return x.index_select(0, indices), y.index_select(0, indices)


def _resolve_split_views(
    dataset: PinnDatasetBundle,
    split: str,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    normalized = str(split).strip().lower()
    if normalized not in ANALYSIS_SPLITS:
        raise ValueError(f"Unsupported analysis split '{split}'. Expected one of {sorted(ANALYSIS_SPLITS)}.")

    if normalized == "train":
        return (
            dataset.train_x,
            dataset.train_y,
            dataset.train_col_x,
            dataset.train_init_x,
            dataset.train_init_y,
        )
    if normalized == "val":
        return (
            dataset.val_x,
            dataset.val_y,
            dataset.val_col_x,
            None,
            None,
        )
    return (
        dataset.test_x,
        dataset.test_y,
        None,
        None,
        None,
    )


def build_analysis_bundle(
    dataset: PinnDatasetBundle,
    *,
    split: str = "train",
    supervised_rows: int | None = None,
    collocation_rows: int | None = None,
    init_rows: int | None = None,
    seed: int = 0,
    require_all_components: bool = False,
) -> PinnAnalysisBundle:
    """Build a deterministic analysis subset for one split.

    Availability rules:
    - `train` exposes supervised, collocation, and init tensors.
    - `val` exposes supervised and collocation tensors.
    - `test` exposes supervised tensors only.
    """

    x_data, y_data, x_col, x_init, y_init = _resolve_split_views(dataset, split)

    if x_data is None or y_data is None:
        raise ValueError(f"Requested analysis split '{split}' has no supervised tensors available.")

    sampled_x_data, sampled_y_data = _sample_xy_rows(x_data, y_data, supervised_rows, seed + 11)

    sampled_x_col = None
    if x_col is not None:
        sampled_x_col = _sample_rows(x_col, collocation_rows, seed + 23)

    sampled_x_init = None
    sampled_y_init = None
    if x_init is not None and y_init is not None:
        sampled_x_init, sampled_y_init = _sample_xy_rows(x_init, y_init, init_rows, seed + 37)

    bundle = PinnAnalysisBundle(
        split=str(split).strip().lower(),
        x_data=sampled_x_data,
        y_data=sampled_y_data,
        x_col=sampled_x_col,
        x_init=sampled_x_init,
        y_init=sampled_y_init,
        supervised_rows=int(sampled_x_data.shape[0]),
        collocation_rows=0 if sampled_x_col is None else int(sampled_x_col.shape[0]),
        init_rows=0 if sampled_x_init is None else int(sampled_x_init.shape[0]),
        seed=int(seed),
    )

    if require_all_components and not (bundle.has_supervised and bundle.has_collocation and bundle.has_init):
        raise ValueError(
            "The requested analysis split does not provide all PINN components "
            "(supervised, collocation, init). Use split='train' or disable require_all_components."
        )

    return bundle
