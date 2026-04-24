"""Supervised sampling and curriculum helpers for trainer loops."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.pinn.runtime import ResolvedCurriculumSettings
from src.training.runtime import cfg_get


def sample_supervised_rows_with_indices(
    x: torch.Tensor,
    y: torch.Tensor,
    config: Any,
    curriculum: ResolvedCurriculumSettings,
    *,
    global_epoch: int = 1,
    difficulty_bins: torch.Tensor | None = None,
    allow_row_subsampling: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    total_rows = int(x.shape[0])
    eligible_indices = torch.arange(total_rows, device=x.device, dtype=torch.long)
    curriculum_enabled = bool(curriculum.enabled)
    if curriculum_enabled:
        if difficulty_bins is None:
            raise ValueError(
                "pinn.curriculum.enabled=true requires supervised HDF5 files with difficulty_bin metadata. "
                "Re-run preprocessing for the dataset."
            )
        active_max_bin = curriculum_active_max_bin(
            curriculum=curriculum,
            global_epoch=global_epoch,
            num_bins=int(torch.max(difficulty_bins).item()) + 1,
        )
        eligible_mask = difficulty_bins.to(device=x.device) <= int(active_max_bin)
        eligible_indices = torch.nonzero(eligible_mask, as_tuple=False).flatten()
        if int(eligible_indices.shape[0]) == 0:
            raise ValueError(
                f"Curriculum produced no supervised rows for epoch {int(global_epoch)} "
                f"and max active bin {int(active_max_bin)}."
            )

    if not allow_row_subsampling or not bool(cfg_get(config, "pinn.supervised_sampling.enabled", False)):
        return x.index_select(0, eligible_indices), y.index_select(0, eligible_indices), eligible_indices

    total_rows = int(eligible_indices.shape[0])
    rows_per_epoch_cfg = cfg_get(config, "pinn.supervised_sampling.rows_per_epoch", None)
    fraction_per_epoch_cfg = cfg_get(config, "pinn.supervised_sampling.fraction_per_epoch", None)

    target_rows: int | None = None
    if rows_per_epoch_cfg not in (None, "null"):
        target_rows = int(rows_per_epoch_cfg)
    elif fraction_per_epoch_cfg not in (None, "null"):
        fraction = float(fraction_per_epoch_cfg)
        if not (0.0 < fraction <= 1.0):
            raise ValueError("pinn.supervised_sampling.fraction_per_epoch must be in (0, 1].")
        target_rows = max(1, int(round(total_rows * fraction)))

    if target_rows is None or target_rows >= total_rows:
        return x.index_select(0, eligible_indices), y.index_select(0, eligible_indices), eligible_indices
    if target_rows <= 0:
        raise ValueError("pinn.supervised_sampling.rows_per_epoch must be > 0.")

    if curriculum_enabled and bool(curriculum.balance_active_bins):
        sampled_local = sample_curriculum_indices_balanced(
            eligible_indices=eligible_indices,
            difficulty_bins=difficulty_bins.to(device=x.device),
            target_rows=target_rows,
        )
        indices = sampled_local.to(device=x.device, dtype=torch.long)
    else:
        local_positions = np.random.choice(total_rows, size=target_rows, replace=False)
        local_idx = torch.as_tensor(local_positions, device=x.device, dtype=torch.long)
        indices = eligible_indices.index_select(0, local_idx)
    return x.index_select(0, indices), y.index_select(0, indices), indices


def curriculum_active_max_bin(
    *,
    curriculum: ResolvedCurriculumSettings,
    global_epoch: int,
    num_bins: int,
) -> int:
    unlock_epochs = curriculum.unlock_epochs
    if unlock_epochs is None:
        stage_length = int(curriculum.stage_length_epochs)
        if stage_length <= 0:
            raise ValueError("pinn.curriculum.stage_length_epochs must be > 0.")
        unlock_epochs = [1 + idx * stage_length for idx in range(num_bins)]
    unlock_list = [int(value) for value in unlock_epochs]
    if len(unlock_list) != num_bins:
        raise ValueError("pinn.curriculum.unlock_epochs must have length pinn.curriculum.num_bins.")
    if unlock_list[0] != 1:
        raise ValueError("pinn.curriculum.unlock_epochs must start at epoch 1.")
    active_bin = 0
    for idx, unlock_epoch in enumerate(unlock_list):
        if int(global_epoch) >= int(unlock_epoch):
            active_bin = idx
    return int(active_bin)


def sample_curriculum_indices_balanced(
    *,
    eligible_indices: torch.Tensor,
    difficulty_bins: torch.Tensor,
    target_rows: int,
) -> torch.Tensor:
    if target_rows >= int(eligible_indices.shape[0]):
        return eligible_indices
    active_bins = torch.unique(difficulty_bins.index_select(0, eligible_indices)).to(dtype=torch.long)
    active_bins = torch.sort(active_bins).values
    if int(active_bins.shape[0]) <= 1:
        local_positions = np.random.choice(int(eligible_indices.shape[0]), size=target_rows, replace=False)
        return eligible_indices.index_select(
            0,
            torch.as_tensor(local_positions, device=eligible_indices.device, dtype=torch.long),
        )

    base = target_rows // int(active_bins.shape[0])
    remainder = target_rows % int(active_bins.shape[0])
    selected_parts: list[torch.Tensor] = []
    leftovers: list[torch.Tensor] = []
    for idx, bin_id in enumerate(active_bins.tolist()):
        bin_mask = difficulty_bins.index_select(0, eligible_indices) == int(bin_id)
        bin_indices = eligible_indices.index_select(0, torch.nonzero(bin_mask, as_tuple=False).flatten())
        desired = base + (1 if idx < remainder else 0)
        if desired >= int(bin_indices.shape[0]):
            selected_parts.append(bin_indices)
        else:
            choice = np.random.choice(int(bin_indices.shape[0]), size=desired, replace=False)
            selected = bin_indices.index_select(
                0,
                torch.as_tensor(choice, device=bin_indices.device, dtype=torch.long),
            )
            selected_parts.append(selected)
            keep_mask = torch.ones(int(bin_indices.shape[0]), device=bin_indices.device, dtype=torch.bool)
            keep_mask[torch.as_tensor(choice, device=bin_indices.device, dtype=torch.long)] = False
            leftovers.append(bin_indices.index_select(0, torch.nonzero(keep_mask, as_tuple=False).flatten()))
    selected = torch.cat(selected_parts, dim=0)
    if int(selected.shape[0]) >= target_rows:
        if int(selected.shape[0]) == target_rows:
            return selected
        choice = np.random.choice(int(selected.shape[0]), size=target_rows, replace=False)
        return selected.index_select(0, torch.as_tensor(choice, device=selected.device, dtype=torch.long))
    if leftovers:
        pool = torch.cat(leftovers, dim=0)
        fill = target_rows - int(selected.shape[0])
        choice = np.random.choice(int(pool.shape[0]), size=fill, replace=False)
        fill_rows = pool.index_select(0, torch.as_tensor(choice, device=pool.device, dtype=torch.long))
        return torch.cat((selected, fill_rows), dim=0)
    return selected
