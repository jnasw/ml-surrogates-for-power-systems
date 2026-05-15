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


@dataclass(frozen=True)
class SupervisedAcquisitionDiagnostics:
    """Serializable state summary for supervised acquisition."""

    enabled: bool
    strategy: str
    active_trajectories: int
    candidate_trajectories: int
    active_rows: int
    acquired_trajectories: int
    acquisition_count: int
    last_acquisition_epoch: int | None
    last_acquired_trajectory_ids: tuple[int, ...]
    last_candidate_score_mean: float | None = None
    last_candidate_score_max: float | None = None
    last_selected_score_mean: float | None = None
    last_selected_score_min: float | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "enabled": bool(self.enabled),
            "strategy": self.strategy,
            "active_trajectories": int(self.active_trajectories),
            "candidate_trajectories": int(self.candidate_trajectories),
            "active_rows": int(self.active_rows),
            "acquired_trajectories": int(self.acquired_trajectories),
            "acquisition_count": int(self.acquisition_count),
            "last_acquisition_epoch": self.last_acquisition_epoch,
            "last_acquired_trajectory_ids": list(self.last_acquired_trajectory_ids),
            "last_candidate_score_mean": self.last_candidate_score_mean,
            "last_candidate_score_max": self.last_candidate_score_max,
            "last_selected_score_mean": self.last_selected_score_mean,
            "last_selected_score_min": self.last_selected_score_min,
        }


class SupervisedAcquisitionManager:
    """Owns active/candidate supervised trajectories for pool-based acquisition."""

    def __init__(
        self,
        *,
        train_traj_ids: torch.Tensor | None,
        strategy: str,
        initial_trajectories: int | None,
        add_trajectories: int,
        max_trajectories: int | None,
        refresh_period_epochs: int,
        seed: int,
    ) -> None:
        strategy_name = str(strategy).strip().lower()
        if strategy_name != "random":
            raise NotImplementedError(
                "Only supervised acquisition strategy 'random' is implemented in this slice. "
                "MAE-based strategies will be added in the scoring slice."
            )
        if int(add_trajectories) <= 0:
            raise ValueError("add_trajectories must be > 0.")
        if int(refresh_period_epochs) <= 0:
            raise ValueError("refresh_period_epochs must be > 0.")

        self._index = build_trajectory_row_index(train_traj_ids)
        self._strategy = strategy_name
        self._add_trajectories = int(add_trajectories)
        self._refresh_period_epochs = int(refresh_period_epochs)
        self._seed = int(seed)
        self._acquisition_count = 0
        self._last_acquisition_epoch: int | None = None
        self._last_acquired_trajectory_ids: tuple[int, ...] = ()

        total_trajectories = int(self._index.trajectory_ids.shape[0])
        if max_trajectories in (None, "null"):
            self._max_trajectories = total_trajectories
        else:
            self._max_trajectories = int(max_trajectories)
        if self._max_trajectories <= 0:
            raise ValueError("max_trajectories must be > 0 when provided.")
        if self._max_trajectories > total_trajectories:
            raise ValueError(
                f"max_trajectories={self._max_trajectories} exceeds available trajectories={total_trajectories}."
            )

        initial_ids = select_initial_trajectory_ids(
            self._index.trajectory_ids,
            initial_trajectories=initial_trajectories,
            seed=self._seed,
        )
        if int(initial_ids.shape[0]) > self._max_trajectories:
            raise ValueError("initial_trajectories must be <= max_trajectories.")
        self._active_trajectory_ids, self._candidate_trajectory_ids = split_active_candidate_trajectory_ids(
            self._index.trajectory_ids,
            initial_ids,
        )

    @property
    def active_trajectory_ids(self) -> torch.Tensor:
        return self._active_trajectory_ids.clone()

    @property
    def candidate_trajectory_ids(self) -> torch.Tensor:
        return self._candidate_trajectory_ids.clone()

    def active_row_indices(self, *, device: torch.device | None = None) -> torch.Tensor:
        return row_indices_for_trajectory_ids(
            self._index,
            self._active_trajectory_ids,
            device=device,
        )

    def maybe_acquire(self, *, global_epoch: int) -> SupervisedAcquisitionDiagnostics:
        """Append trajectories if the acquisition cadence fires for this epoch."""
        self._last_acquired_trajectory_ids = ()
        if not self._should_acquire(global_epoch=global_epoch):
            return self.diagnostics()

        remaining_capacity = self._max_trajectories - int(self._active_trajectory_ids.shape[0])
        add_count = min(self._add_trajectories, remaining_capacity, int(self._candidate_trajectory_ids.shape[0]))
        if add_count <= 0:
            return self.diagnostics()

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self._seed + self._acquisition_count + int(global_epoch))
        selected_positions = torch.randperm(
            int(self._candidate_trajectory_ids.shape[0]),
            generator=generator,
        )[:add_count]
        selected_ids = self._candidate_trajectory_ids.index_select(0, selected_positions)
        updated_active = torch.cat((self._active_trajectory_ids, selected_ids), dim=0)
        self._active_trajectory_ids, self._candidate_trajectory_ids = split_active_candidate_trajectory_ids(
            self._index.trajectory_ids,
            updated_active,
        )
        self._acquisition_count += 1
        self._last_acquisition_epoch = int(global_epoch)
        self._last_acquired_trajectory_ids = tuple(int(value) for value in torch.sort(selected_ids).values.tolist())
        return self.diagnostics()

    def diagnostics(self) -> SupervisedAcquisitionDiagnostics:
        return SupervisedAcquisitionDiagnostics(
            enabled=True,
            strategy=self._strategy,
            active_trajectories=int(self._active_trajectory_ids.shape[0]),
            candidate_trajectories=int(self._candidate_trajectory_ids.shape[0]),
            active_rows=int(self.active_row_indices().shape[0]),
            acquired_trajectories=len(self._last_acquired_trajectory_ids),
            acquisition_count=int(self._acquisition_count),
            last_acquisition_epoch=self._last_acquisition_epoch,
            last_acquired_trajectory_ids=self._last_acquired_trajectory_ids,
        )

    def _should_acquire(self, *, global_epoch: int) -> bool:
        epoch = int(global_epoch)
        if epoch <= 1:
            return False
        if int(self._active_trajectory_ids.shape[0]) >= self._max_trajectories:
            return False
        if int(self._candidate_trajectory_ids.shape[0]) <= 0:
            return False
        return ((epoch - 1) % self._refresh_period_epochs) == 0


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
