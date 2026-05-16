"""Trajectory-level helpers for pool-based supervised acquisition."""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn
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
    last_anchor_score_mean: float | None = None
    last_anchor_score_max: float | None = None
    last_selected_distance_mean: float | None = None
    last_selected_distance_max: float | None = None

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
            "last_anchor_score_mean": self.last_anchor_score_mean,
            "last_anchor_score_max": self.last_anchor_score_max,
            "last_selected_distance_mean": self.last_selected_distance_mean,
            "last_selected_distance_max": self.last_selected_distance_max,
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
        candidate_batch_size: int = 4096,
        anchor_trajectories: int | None = None,
        similarity_space: str = "initial_condition",
        seed: int,
    ) -> None:
        strategy_name = str(strategy).strip().lower()
        if strategy_name not in {"random", "mae_nearest"}:
            raise ValueError("strategy must be one of: random, mae_nearest.")
        similarity_space_name = str(similarity_space).strip().lower()
        if similarity_space_name != "initial_condition":
            raise ValueError("similarity_space must be 'initial_condition'.")
        if int(add_trajectories) <= 0:
            raise ValueError("add_trajectories must be > 0.")
        if int(refresh_period_epochs) <= 0:
            raise ValueError("refresh_period_epochs must be > 0.")
        if int(candidate_batch_size) <= 0:
            raise ValueError("candidate_batch_size must be > 0.")
        if anchor_trajectories not in (None, "null") and int(anchor_trajectories) <= 0:
            raise ValueError("anchor_trajectories must be > 0 when provided.")

        self._index = build_trajectory_row_index(train_traj_ids)
        self._strategy = strategy_name
        self._add_trajectories = int(add_trajectories)
        self._refresh_period_epochs = int(refresh_period_epochs)
        self._candidate_batch_size = int(candidate_batch_size)
        self._anchor_trajectories = None if anchor_trajectories in (None, "null") else int(anchor_trajectories)
        self._similarity_space = similarity_space_name
        self._seed = int(seed)
        self._acquisition_count = 0
        self._last_acquisition_epoch: int | None = None
        self._last_acquired_trajectory_ids: tuple[int, ...] = ()
        self._last_anchor_score_mean: float | None = None
        self._last_anchor_score_max: float | None = None
        self._last_selected_distance_mean: float | None = None
        self._last_selected_distance_max: float | None = None

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

    def maybe_acquire(
        self,
        *,
        global_epoch: int,
        model: nn.Module | None = None,
        x: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
    ) -> SupervisedAcquisitionDiagnostics:
        """Append trajectories if the acquisition cadence fires for this epoch."""
        self._last_acquired_trajectory_ids = ()
        self._last_anchor_score_mean = None
        self._last_anchor_score_max = None
        self._last_selected_distance_mean = None
        self._last_selected_distance_max = None
        if not self._should_acquire(global_epoch=global_epoch):
            return self.diagnostics()

        remaining_capacity = self._max_trajectories - int(self._active_trajectory_ids.shape[0])
        add_count = min(self._add_trajectories, remaining_capacity, int(self._candidate_trajectory_ids.shape[0]))
        if add_count <= 0:
            return self.diagnostics()

        if self._strategy == "random":
            selected_positions = self._select_random_positions(add_count=add_count, global_epoch=global_epoch)
        else:
            selected_positions = self._select_nearest_to_hard_active_positions(
                model=model,
                x=x,
                y=y,
                add_count=add_count,
            )
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
            last_anchor_score_mean=self._last_anchor_score_mean,
            last_anchor_score_max=self._last_anchor_score_max,
            last_selected_distance_mean=self._last_selected_distance_mean,
            last_selected_distance_max=self._last_selected_distance_max,
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

    def _select_random_positions(self, *, add_count: int, global_epoch: int) -> torch.Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self._seed + self._acquisition_count + int(global_epoch))
        return torch.randperm(
            int(self._candidate_trajectory_ids.shape[0]),
            generator=generator,
        )[:add_count]

    def _select_nearest_to_hard_active_positions(
        self,
        *,
        model: nn.Module | None,
        x: torch.Tensor | None,
        y: torch.Tensor | None,
        add_count: int,
    ) -> torch.Tensor:
        active_scores = self._score_trajectories(
            trajectory_ids=self._active_trajectory_ids,
            model=model,
            x=x,
            y=y,
            score_label="active",
        )
        anchor_count = min(
            self._anchor_trajectories if self._anchor_trajectories is not None else self._add_trajectories,
            int(self._active_trajectory_ids.shape[0]),
        )
        anchor_positions = torch.topk(active_scores, k=anchor_count, largest=True).indices.to(dtype=torch.long)
        anchor_ids = self._active_trajectory_ids.index_select(0, anchor_positions)
        anchor_scores = active_scores.index_select(0, anchor_positions)
        self._last_anchor_score_mean = float(anchor_scores.mean().item())
        self._last_anchor_score_max = float(anchor_scores.max().item())

        descriptors = self._initial_condition_descriptors(x=x)
        normalized_descriptors = _standardize_descriptors(descriptors)
        anchor_descriptors = _descriptors_for_trajectory_ids(
            normalized_descriptors,
            self._index.trajectory_ids,
            anchor_ids,
        )
        candidate_descriptors = _descriptors_for_trajectory_ids(
            normalized_descriptors,
            self._index.trajectory_ids,
            self._candidate_trajectory_ids,
        )
        distances = torch.cdist(
            candidate_descriptors.to(dtype=torch.float64),
            anchor_descriptors.to(dtype=torch.float64),
            p=2.0,
        ).min(dim=1).values
        selected_positions = torch.topk(distances, k=add_count, largest=False).indices.to(dtype=torch.long, device="cpu")
        selected_distances = distances.index_select(0, selected_positions)
        self._last_selected_distance_mean = float(selected_distances.mean().item())
        self._last_selected_distance_max = float(selected_distances.max().item())
        return selected_positions

    def _score_trajectories(
        self,
        *,
        trajectory_ids: torch.Tensor,
        model: nn.Module | None,
        x: torch.Tensor | None,
        y: torch.Tensor | None,
        score_label: str,
    ) -> torch.Tensor:
        if model is None or x is None or y is None:
            raise ValueError(f"strategy={self._strategy} requires model, x, and y for {score_label} scoring.")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same leading dimension for trajectory scoring.")

        scores: list[float] = []
        was_training = bool(model.training)
        model.eval()
        try:
            with torch.no_grad():
                for trajectory_id in trajectory_ids.tolist():
                    rows = self._index.rows_by_trajectory_id[int(trajectory_id)].to(device=x.device, dtype=torch.long)
                    total_abs = 0.0
                    total_count = 0
                    for start in range(0, int(rows.shape[0]), self._candidate_batch_size):
                        stop = min(start + self._candidate_batch_size, int(rows.shape[0]))
                        batch_rows = rows[start:stop]
                        xb = x.index_select(0, batch_rows)
                        yb = y.index_select(0, batch_rows)
                        pred = model(xb)
                        diff = pred - yb
                        total_abs += float(diff.abs().sum().item())
                        total_count += int(diff.numel())
                    if total_count <= 0:
                        raise ValueError(f"Cannot score empty trajectory_id={int(trajectory_id)}.")
                    scores.append(total_abs / float(total_count))
        finally:
            if was_training:
                model.train()
        return torch.as_tensor(scores, dtype=torch.float64, device="cpu")

    def _initial_condition_descriptors(self, *, x: torch.Tensor | None) -> torch.Tensor:
        if x is None:
            raise ValueError(f"strategy={self._strategy} requires x for initial-condition similarity.")
        if x.ndim != 2:
            raise ValueError("x must be rank-2 for initial-condition similarity.")
        if int(x.shape[1]) < 2:
            raise ValueError("initial-condition similarity expects x to contain time plus at least one feature.")

        descriptors: list[torch.Tensor] = []
        x_cpu = x.detach().to(device="cpu", dtype=torch.float64)
        for trajectory_id in self._index.trajectory_ids.tolist():
            rows = self._index.rows_by_trajectory_id[int(trajectory_id)]
            first_row = int(rows[0].item())
            descriptors.append(x_cpu[first_row, 1:])
        return torch.stack(descriptors, dim=0)


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


def _descriptors_for_trajectory_ids(
    descriptors: torch.Tensor,
    all_trajectory_ids: torch.Tensor,
    trajectory_ids: torch.Tensor,
) -> torch.Tensor:
    id_to_position = {int(value): idx for idx, value in enumerate(all_trajectory_ids.tolist())}
    positions = [id_to_position[int(value)] for value in trajectory_ids.tolist()]
    return descriptors.index_select(0, torch.as_tensor(positions, dtype=torch.long))


def _standardize_descriptors(descriptors: torch.Tensor) -> torch.Tensor:
    if descriptors.ndim != 2:
        raise ValueError("descriptors must be rank-2.")
    mean = descriptors.mean(dim=0, keepdim=True)
    std = descriptors.std(dim=0, unbiased=False, keepdim=True)
    std = torch.where(std > 1.0e-12, std, torch.ones_like(std))
    return (descriptors - mean) / std


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
