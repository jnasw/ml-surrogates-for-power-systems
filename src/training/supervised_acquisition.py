"""Trajectory-level helpers for pool-based supervised acquisition."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TrajectoryRowIndex:
    """Map trajectory IDs to supervised row indices."""

    trajectory_ids: torch.Tensor
    rows_by_trajectory_id: dict[int, torch.Tensor]
    total_rows: int


def build_trajectory_row_index(train_traj_ids: torch.Tensor | None) -> TrajectoryRowIndex:
    """Build a deterministic trajectory-ID to row-index map."""
    if train_traj_ids is None:
        raise ValueError(
            "Supervised acquisition requires trajectory_id metadata on supervised training rows. "
            "Re-run preprocessing with trajectory metadata enabled."
        )
    if train_traj_ids.ndim != 1:
        raise ValueError("train_traj_ids must be a rank-1 tensor.")
    if int(train_traj_ids.shape[0]) == 0:
        raise ValueError("train_traj_ids must not be empty.")

    ids_cpu = train_traj_ids.detach().to(device="cpu", dtype=torch.long)
    trajectory_ids = torch.unique(ids_cpu, sorted=True)
    rows_by_id: dict[int, torch.Tensor] = {}
    for trajectory_id in trajectory_ids.tolist():
        row_indices = torch.nonzero(ids_cpu == int(trajectory_id), as_tuple=False).flatten().to(dtype=torch.long)
        if int(row_indices.shape[0]) == 0:
            raise ValueError(f"No supervised rows found for trajectory_id={int(trajectory_id)}.")
        rows_by_id[int(trajectory_id)] = row_indices
    return TrajectoryRowIndex(
        trajectory_ids=trajectory_ids.to(dtype=torch.long),
        rows_by_trajectory_id=rows_by_id,
        total_rows=int(ids_cpu.shape[0]),
    )


def select_initial_trajectory_ids(
    trajectory_ids: torch.Tensor,
    *,
    initial_trajectories: int | None,
    seed: int,
) -> torch.Tensor:
    """Select the initial active trajectory IDs for a pool-based run."""
    all_ids = _normalize_trajectory_ids(trajectory_ids)
    if initial_trajectories in (None, "null"):
        return all_ids.clone()

    n_initial = int(initial_trajectories)
    if n_initial <= 0:
        raise ValueError("initial_trajectories must be > 0 when provided.")
    if n_initial > int(all_ids.shape[0]):
        raise ValueError(
            f"initial_trajectories={n_initial} exceeds available trajectories={int(all_ids.shape[0])}."
        )
    if n_initial == int(all_ids.shape[0]):
        return all_ids.clone()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    selected_positions = torch.randperm(int(all_ids.shape[0]), generator=generator)[:n_initial]
    selected_ids = all_ids.index_select(0, selected_positions)
    return torch.sort(selected_ids).values


def split_active_candidate_trajectory_ids(
    trajectory_ids: torch.Tensor,
    active_trajectory_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return validated active IDs and remaining candidate IDs."""
    all_ids = _normalize_trajectory_ids(trajectory_ids)
    active_ids = _normalize_trajectory_ids(active_trajectory_ids)
    all_set = set(int(value) for value in all_ids.tolist())
    active_set = set(int(value) for value in active_ids.tolist())
    missing = sorted(active_set - all_set)
    if missing:
        raise ValueError(f"Active trajectory IDs are not present in the pool: {missing}.")
    if len(active_set) != int(active_ids.shape[0]):
        raise ValueError("active_trajectory_ids must not contain duplicates.")

    active_ordered = torch.as_tensor(
        [int(value) for value in all_ids.tolist() if int(value) in active_set],
        dtype=torch.long,
    )
    candidate_ordered = torch.as_tensor(
        [int(value) for value in all_ids.tolist() if int(value) not in active_set],
        dtype=torch.long,
    )
    return active_ordered, candidate_ordered


def row_indices_for_trajectory_ids(
    index: TrajectoryRowIndex,
    trajectory_ids: torch.Tensor,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return supervised row indices for whole trajectories."""
    ids = _normalize_trajectory_ids(trajectory_ids)
    parts: list[torch.Tensor] = []
    missing: list[int] = []
    for trajectory_id in ids.tolist():
        rows = index.rows_by_trajectory_id.get(int(trajectory_id))
        if rows is None:
            missing.append(int(trajectory_id))
        else:
            parts.append(rows)
    if missing:
        raise ValueError(f"Trajectory IDs are not present in the row index: {missing}.")
    if not parts:
        return torch.empty((0,), dtype=torch.long, device=device)
    out = torch.cat(parts, dim=0).to(dtype=torch.long)
    if device is not None:
        out = out.to(device=device)
    return out


def _normalize_trajectory_ids(trajectory_ids: torch.Tensor) -> torch.Tensor:
    if trajectory_ids.ndim != 1:
        raise ValueError("trajectory_ids must be a rank-1 tensor.")
    if int(trajectory_ids.shape[0]) == 0:
        return trajectory_ids.detach().to(device="cpu", dtype=torch.long)
    ids = trajectory_ids.detach().to(device="cpu", dtype=torch.long)
    unique_ids = torch.unique(ids, sorted=True)
    if int(unique_ids.shape[0]) != int(ids.shape[0]):
        raise ValueError("trajectory_ids must not contain duplicates.")
    return unique_ids
