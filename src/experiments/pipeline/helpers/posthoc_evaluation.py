"""Shared utilities for post-training checkpoint evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

from src.data.contracts.data_contract import (
    H5_COLLOCATION_SUFFIX,
    H5_DATA_SUFFIX,
    H5_FILE_SUFFIX,
    H5_INIT_SUFFIX,
    H5_TRAJECTORY_ID_KEYS,
    H5_X_KEYS,
    H5_Y_KEYS,
    SPLITS,
)
from src.experiments.pipeline.helpers.evaluation import resolve_evaluation_dataset
from src.experiments.pipeline.helpers.manifest import utc_now_iso
from src.pinn.runtime import resolve_torch_dtype
from src.training.trainer import PinnModel


OUTPUT_PATH_ALIASES = {
    "adaptive_augmentation": "data_augmentation_comparison",
}


@dataclass(frozen=True)
class TrajectorySlice:
    """Contiguous row span belonging to one trajectory."""

    start: int
    stop: int
    trajectory_id: int | None = None


@dataclass(frozen=True)
class SupervisedEvaluationSplit:
    """Supervised rows and optional trajectory metadata for one split."""

    dataset_root: str
    split: str
    x: torch.Tensor
    y: torch.Tensor
    trajectory_ids: torch.Tensor | None = None


@dataclass(frozen=True)
class PosthocPrediction:
    """Loaded external split and predictions from one checkpoint."""

    split: SupervisedEvaluationSplit
    y_pred: torch.Tensor
    checkpoint_path: str
    batch_size: int


@dataclass(frozen=True)
class PosthocRunCandidate:
    """Resolved inputs for one checkpoint posthoc evaluation attempt."""

    experiment_name: str
    batch_id: str
    run_name: str
    batch_root: Path
    run_dir: Path
    checkpoint_path: Path
    config_path: Path
    id_eval_root: Path | None
    ood_eval_root: Path | None
    status: str
    missing: tuple[str, ...]
    metadata: dict[str, Any]

    @property
    def ready(self) -> bool:
        return self.status == "ready"


POSTHOC_SUMMARY_FIELDNAMES = [
    "experiment_name",
    "batch_id",
    "run_name",
    "status",
    "missing",
    "error",
    "checkpoint_tag",
    "eval_split",
    "ood_mode",
    "run_dir",
    "checkpoint_path",
    "config_path",
    "id_eval_root",
    "ood_eval_root",
    "id_n_trajectories",
    "id_n_rows",
    "id_mse",
    "id_rmse",
    "id_mae",
    "id_max_abs_error",
    "id_trajectory_mse_mean",
    "id_trajectory_mse_p90",
    "id_trajectory_mse_p95",
    "id_trajectory_mse_max",
    "id_trajectory_rmse_p90",
    "id_trajectory_rmse_p95",
    "id_trajectory_rmse_max",
    "ood_n_trajectories_before_filter",
    "ood_n_trajectories",
    "ood_n_rows_before_filter",
    "ood_n_rows",
    "ood_mse",
    "ood_rmse",
    "ood_mae",
    "ood_max_abs_error",
    "ood_trajectory_mse_mean",
    "ood_trajectory_mse_p90",
    "ood_trajectory_mse_p95",
    "ood_trajectory_mse_max",
    "ood_trajectory_rmse_p90",
    "ood_trajectory_rmse_p95",
    "ood_trajectory_rmse_max",
    "id_ood_mse_gap",
    "id_ood_rmse_gap",
    "run_result_path",
]


def _iter_supervised_h5_files(split_dir: Path) -> Iterable[Path]:
    if not split_dir.exists():
        return []
    paths: list[Path] = []
    for path in sorted(split_dir.iterdir()):
        name = path.name
        if not path.is_file() or path.suffix != H5_FILE_SUFFIX:
            continue
        if H5_DATA_SUFFIX not in name or H5_COLLOCATION_SUFFIX in name or H5_INIT_SUFFIX in name:
            continue
        paths.append(path)
    return paths


def resolve_posthoc_path(
    path: str | Path | None,
    *,
    repo_root: str | Path = ".",
    must_exist: bool = False,
) -> Path | None:
    """Resolve local paths from manifests copied from HPC.

    The experiment summaries often contain absolute paths from the cluster. This
    helper maps stable repository-relative suffixes back onto the local checkout.
    """

    if path in (None, ""):
        return None

    raw = Path(str(path)).expanduser()
    if raw.exists():
        return raw.resolve()

    repo = Path(repo_root).resolve()
    raw_text = str(path)
    candidates: list[Path] = []

    if "/outputs/pinn/" in raw_text:
        tail = raw_text.split("/outputs/pinn/", 1)[1]
        parts = Path(tail).parts
        if parts:
            alias = OUTPUT_PATH_ALIASES.get(parts[0], parts[0])
            tail_path = Path(alias, *parts[1:])
        else:
            tail_path = Path(tail)
        candidates.append(repo / "outputs" / tail_path)

    if "/outputs/" in raw_text:
        tail = raw_text.split("/outputs/", 1)[1]
        candidates.append(repo / "outputs" / tail)

    if "/data/" in raw_text:
        tail = raw_text.split("/data/", 1)[1]
        candidates.append(repo / "data" / tail)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    if must_exist:
        tried = ", ".join(str(candidate) for candidate in candidates) or str(raw)
        raise FileNotFoundError(f"Could not resolve posthoc path {path!r}. Tried: {tried}")

    return candidates[0].resolve() if candidates else raw.resolve()


def _read_json_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _read_summary_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _run_rows_by_name(batch_root: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    manifest = _read_json_dict(batch_root / "run_manifest.json")
    for run in manifest.get("runs", []) or []:
        if not isinstance(run, dict):
            continue
        run_name = run.get("run_name")
        if run_name in (None, ""):
            continue
        rows[str(run_name)] = dict(run)

    for row in _read_summary_rows(batch_root / "summary.csv"):
        run_name = row.get("run_name")
        if run_name in (None, ""):
            continue
        merged = dict(rows.get(str(run_name), {}))
        # Prefer non-empty summary values where available; summaries often carry
        # checkpoint paths not present in the manifest.
        for key, value in row.items():
            if value not in (None, ""):
                merged[key] = value
        rows[str(run_name)] = merged
    return rows


def _local_run_dir_from_row(
    *,
    row: dict[str, Any],
    batch_root: Path,
    run_name: str,
    repo_root: Path,
) -> Path:
    raw_run_dir = row.get("run_dir")
    if raw_run_dir not in (None, ""):
        resolved = resolve_posthoc_path(raw_run_dir, repo_root=repo_root, must_exist=False)
        if resolved is not None and resolved.exists():
            return resolved
    return (batch_root / "runs" / run_name).resolve()


def _checkpoint_path_from_row(
    *,
    row: dict[str, Any],
    run_dir: Path,
    checkpoint_tag: str,
    repo_root: Path,
) -> Path:
    if checkpoint_tag == "best":
        raw_checkpoint = row.get("best_checkpoint_path")
        if raw_checkpoint not in (None, ""):
            resolved = resolve_posthoc_path(raw_checkpoint, repo_root=repo_root, must_exist=False)
            if resolved is not None:
                return resolved
    return (run_dir / "checkpoints" / f"{checkpoint_tag}.pt").resolve()


def _resolve_eval_root_from_row(
    *,
    row: dict[str, Any],
    kind: str,
    repo_root: Path,
) -> Path | None:
    raw = row.get(f"{kind}_eval_root")
    if raw in (None, ""):
        return None
    return resolve_posthoc_path(raw, repo_root=repo_root, must_exist=False)


def resolve_posthoc_eval_root(
    *,
    kind: str,
    repo_root: str | Path = ".",
    eval_id: str | None = None,
    eval_root: str | Path | None = None,
) -> Path | None:
    """Resolve an explicit posthoc evaluation dataset override."""

    if kind not in {"id", "ood"}:
        raise ValueError("kind must be 'id' or 'ood'.")
    if eval_id not in (None, "") and eval_root not in (None, ""):
        raise ValueError(f"{kind} eval_id and eval_root are mutually exclusive.")
    if eval_root not in (None, ""):
        return resolve_posthoc_path(eval_root, repo_root=repo_root, must_exist=False)
    if eval_id in (None, ""):
        return None
    entry = resolve_evaluation_dataset(str(eval_id))
    return resolve_posthoc_path(entry["dataset_root"], repo_root=repo_root, must_exist=False)


def discover_posthoc_batch_runs(
    *,
    batch_root: str | Path,
    checkpoint_tag: str = "best",
    repo_root: str | Path = ".",
    require_id_eval: bool = True,
    require_ood_eval: bool = True,
    id_eval_id: str | None = None,
    ood_eval_id: str | None = None,
    id_eval_root: str | Path | None = None,
    ood_eval_root: str | Path | None = None,
    require_config: bool = True,
) -> list[PosthocRunCandidate]:
    """Discover checkpoint-evaluation candidates under one timestamp batch."""

    repo = Path(repo_root).resolve()
    batch = resolve_posthoc_path(batch_root, repo_root=repo, must_exist=True)
    if batch is None:
        raise FileNotFoundError(f"Batch root not found: {batch_root}")
    batch = batch.resolve()
    experiment_name = batch.parent.name
    batch_id = batch.name
    rows = _run_rows_by_name(batch)
    id_eval_override = resolve_posthoc_eval_root(
        kind="id",
        repo_root=repo,
        eval_id=id_eval_id,
        eval_root=id_eval_root,
    )
    ood_eval_override = resolve_posthoc_eval_root(
        kind="ood",
        repo_root=repo,
        eval_id=ood_eval_id,
        eval_root=ood_eval_root,
    )

    if not rows:
        runs_dir = batch / "runs"
        if runs_dir.exists():
            for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
                rows[run_dir.name] = {"run_name": run_dir.name, "run_dir": str(run_dir)}

    candidates: list[PosthocRunCandidate] = []
    for run_name in sorted(rows):
        row = rows[run_name]
        run_dir = _local_run_dir_from_row(row=row, batch_root=batch, run_name=run_name, repo_root=repo)
        checkpoint_path = _checkpoint_path_from_row(
            row=row,
            run_dir=run_dir,
            checkpoint_tag=checkpoint_tag,
            repo_root=repo,
        )
        config_path = (run_dir / "config.yaml").resolve()
        id_eval_root_resolved = id_eval_override or _resolve_eval_root_from_row(row=row, kind="id", repo_root=repo)
        ood_eval_root_resolved = ood_eval_override or _resolve_eval_root_from_row(row=row, kind="ood", repo_root=repo)

        missing: list[str] = []
        if not run_dir.exists():
            missing.append("run_dir")
        if not checkpoint_path.exists():
            missing.append("checkpoint")
        if require_config and not config_path.exists():
            missing.append("config")
        if require_id_eval and (id_eval_root_resolved is None or not id_eval_root_resolved.exists()):
            missing.append("id_eval_root")
        if require_ood_eval and (ood_eval_root_resolved is None or not ood_eval_root_resolved.exists()):
            missing.append("ood_eval_root")

        metadata = dict(row)
        if id_eval_override is not None:
            metadata["posthoc_id_eval_override"] = True
            metadata["posthoc_id_eval_id"] = id_eval_id
            metadata["posthoc_id_eval_root"] = str(id_eval_override)
        if ood_eval_override is not None:
            metadata["posthoc_ood_eval_override"] = True
            metadata["posthoc_ood_eval_id"] = ood_eval_id
            metadata["posthoc_ood_eval_root"] = str(ood_eval_override)

        candidates.append(
            PosthocRunCandidate(
                experiment_name=experiment_name,
                batch_id=batch_id,
                run_name=run_name,
                batch_root=batch,
                run_dir=run_dir,
                checkpoint_path=checkpoint_path,
                config_path=config_path,
                id_eval_root=id_eval_root_resolved,
                ood_eval_root=ood_eval_root_resolved,
                status="ready" if not missing else "missing_inputs",
                missing=tuple(missing),
                metadata=metadata,
            )
        )
    return candidates


def discover_posthoc_runs(
    *,
    root: str | Path,
    checkpoint_tag: str = "best",
    repo_root: str | Path = ".",
    require_id_eval: bool = True,
    require_ood_eval: bool = True,
    id_eval_id: str | None = None,
    ood_eval_id: str | None = None,
    id_eval_root: str | Path | None = None,
    ood_eval_root: str | Path | None = None,
    require_config: bool = True,
) -> list[PosthocRunCandidate]:
    """Discover candidates from a batch root, experiment root, or single run dir."""

    resolved_root = resolve_posthoc_path(root, repo_root=repo_root, must_exist=True)
    if resolved_root is None:
        raise FileNotFoundError(f"Root not found: {root}")
    resolved_root = resolved_root.resolve()

    if (resolved_root / "checkpoints").is_dir():
        batch_root = resolved_root.parents[1] if resolved_root.parent.name == "runs" else resolved_root.parent
        candidate = discover_posthoc_batch_runs(
            batch_root=batch_root,
            checkpoint_tag=checkpoint_tag,
            repo_root=repo_root,
            require_id_eval=require_id_eval,
            require_ood_eval=require_ood_eval,
            id_eval_id=id_eval_id,
            ood_eval_id=ood_eval_id,
            id_eval_root=id_eval_root,
            ood_eval_root=ood_eval_root,
            require_config=require_config,
        )
        return [item for item in candidate if item.run_dir == resolved_root]

    if (resolved_root / "runs").is_dir():
        return discover_posthoc_batch_runs(
            batch_root=resolved_root,
            checkpoint_tag=checkpoint_tag,
            repo_root=repo_root,
            require_id_eval=require_id_eval,
            require_ood_eval=require_ood_eval,
            id_eval_id=id_eval_id,
            ood_eval_id=ood_eval_id,
            id_eval_root=id_eval_root,
            ood_eval_root=ood_eval_root,
            require_config=require_config,
        )

    batches = [
        path for path in sorted(resolved_root.iterdir())
        if path.is_dir() and (path / "runs").is_dir()
    ]
    candidates: list[PosthocRunCandidate] = []
    for batch in batches:
        candidates.extend(
            discover_posthoc_batch_runs(
                batch_root=batch,
                checkpoint_tag=checkpoint_tag,
                repo_root=repo_root,
                require_id_eval=require_id_eval,
                require_ood_eval=require_ood_eval,
                id_eval_id=id_eval_id,
                ood_eval_id=ood_eval_id,
                id_eval_root=id_eval_root,
                ood_eval_root=ood_eval_root,
                require_config=require_config,
            )
        )
    return candidates


def _dataset_generation_run_dir_from_summary_row(
    *,
    method_root: Path,
    row: dict[str, Any],
) -> Path:
    method = str(row["method"])
    budget = str(row["budget"])
    dataset_seed = str(row["dataset_seed_label"])
    baseline_seed = str(row["baseline_seed_label"])
    return (method_root / method / budget / dataset_seed / "pinn_data_only" / baseline_seed).resolve()


def discover_dataset_generation_posthoc_runs(
    *,
    root: str | Path,
    checkpoint_tag: str = "best",
    repo_root: str | Path = ".",
    require_id_eval: bool = True,
    require_ood_eval: bool = True,
    id_eval_id: str | None = None,
    ood_eval_id: str | None = None,
    id_eval_root: str | Path | None = None,
    ood_eval_root: str | Path | None = None,
    require_config: bool = True,
) -> list[PosthocRunCandidate]:
    """Discover posthoc candidates from the dataset-generation analysis layout.

    Dataset-generation comparison artifacts are nested by generated dataset:
    ``sm4_b*/<method>/<method>/<budget>/<dataset_seed>/pinn_data_only/<baseline_seed>``.
    The method-level ``summary.csv`` is the source of the 72-run comparison matrix.
    """

    repo = Path(repo_root).resolve()
    resolved_root = resolve_posthoc_path(root, repo_root=repo, must_exist=True)
    if resolved_root is None:
        raise FileNotFoundError(f"Root not found: {root}")
    resolved_root = resolved_root.resolve()

    summary_paths = sorted(path for path in resolved_root.glob("**/summary.csv") if "posthoc_evaluation" not in path.parts)
    candidates: list[PosthocRunCandidate] = []
    for summary_path in summary_paths:
        method_root = summary_path.parent
        if not (method_root / "run_manifest.json").exists():
            continue
        budget_id = method_root.parent.name
        method_id = method_root.name
        batch_id = f"{budget_id}/{method_id}"
        id_eval_override = resolve_posthoc_eval_root(
            kind="id",
            repo_root=repo,
            eval_id=id_eval_id,
            eval_root=id_eval_root,
        )
        ood_eval_override = resolve_posthoc_eval_root(
            kind="ood",
            repo_root=repo,
            eval_id=ood_eval_id,
            eval_root=ood_eval_root,
        )

        for row in _read_summary_rows(summary_path):
            run_name = row.get("run_name")
            if run_name in (None, ""):
                continue
            try:
                run_dir = _dataset_generation_run_dir_from_summary_row(method_root=method_root, row=row)
            except KeyError as exc:
                metadata = dict(row)
                metadata["posthoc_discovery_error"] = f"missing summary column: {exc}"
                run_dir = (method_root / "missing_run_dir" / str(run_name)).resolve()
            checkpoint_path = (run_dir / "checkpoints" / f"{checkpoint_tag}.pt").resolve()
            config_path = (run_dir / "config.yaml").resolve()
            id_eval_root_resolved = id_eval_override or _resolve_eval_root_from_row(row=row, kind="id", repo_root=repo)
            ood_eval_root_resolved = ood_eval_override or _resolve_eval_root_from_row(row=row, kind="ood", repo_root=repo)

            missing: list[str] = []
            if not run_dir.exists():
                missing.append("run_dir")
            if not checkpoint_path.exists():
                missing.append("checkpoint")
            if require_config and not config_path.exists():
                missing.append("config")
            if require_id_eval and (id_eval_root_resolved is None or not id_eval_root_resolved.exists()):
                missing.append("id_eval_root")
            if require_ood_eval and (ood_eval_root_resolved is None or not ood_eval_root_resolved.exists()):
                missing.append("ood_eval_root")

            metadata = dict(row)
            metadata["posthoc_layout"] = "dataset_generation"
            metadata["posthoc_summary_path"] = str(summary_path)
            if id_eval_override is not None:
                metadata["posthoc_id_eval_override"] = True
                metadata["posthoc_id_eval_id"] = id_eval_id
                metadata["posthoc_id_eval_root"] = str(id_eval_override)
            if ood_eval_override is not None:
                metadata["posthoc_ood_eval_override"] = True
                metadata["posthoc_ood_eval_id"] = ood_eval_id
                metadata["posthoc_ood_eval_root"] = str(ood_eval_override)

            candidates.append(
                PosthocRunCandidate(
                    experiment_name="dataset_generation_analysis",
                    batch_id=batch_id,
                    run_name=str(run_name),
                    batch_root=method_root,
                    run_dir=run_dir,
                    checkpoint_path=checkpoint_path,
                    config_path=config_path,
                    id_eval_root=id_eval_root_resolved,
                    ood_eval_root=ood_eval_root_resolved,
                    status="ready" if not missing else "missing_inputs",
                    missing=tuple(missing),
                    metadata=metadata,
                )
            )
    return candidates


def posthoc_output_dir(
    *,
    batch_root: str | Path,
    checkpoint_tag: str,
    eval_split: str,
    ood_mode: str,
    repo_root: str | Path = ".",
) -> Path:
    """Return the settings-specific posthoc output directory for one batch."""

    batch = resolve_posthoc_path(batch_root, repo_root=repo_root, must_exist=True)
    if batch is None:
        raise FileNotFoundError(f"Batch root not found: {batch_root}")
    key = f"{checkpoint_tag}_{eval_split}_{ood_mode}_ood"
    return batch / "posthoc_evaluation" / key


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _safe_run_result_name(run_name: str, checkpoint_tag: str) -> str:
    safe_run = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in run_name)
    safe_tag = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in checkpoint_tag)
    return f"{safe_run}__{safe_tag}.json"


def posthoc_run_result_path(
    *,
    output_dir: str | Path,
    run_name: str,
    checkpoint_tag: str,
) -> Path:
    return Path(output_dir) / "runs" / _safe_run_result_name(run_name, checkpoint_tag)


def base_posthoc_run_result(
    *,
    candidate: PosthocRunCandidate,
    checkpoint_tag: str,
    eval_split: str,
    ood_mode: str,
    status: str,
    missing: Iterable[str] | None = None,
    error: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a serializable posthoc result scaffold for one run."""

    payload: dict[str, Any] = {
        "schema_version": "posthoc_checkpoint_eval_v1",
        "created_at_utc": utc_now_iso(),
        "status": status,
        "missing": list(missing or ()),
        "error": error,
        "experiment_name": candidate.experiment_name,
        "batch_id": candidate.batch_id,
        "run_name": candidate.run_name,
        "checkpoint_tag": checkpoint_tag,
        "eval_split": eval_split,
        "ood_mode": ood_mode,
        "run_dir": str(candidate.run_dir),
        "checkpoint_path": str(candidate.checkpoint_path),
        "config_path": str(candidate.config_path),
        "id_eval_root": None if candidate.id_eval_root is None else str(candidate.id_eval_root),
        "ood_eval_root": None if candidate.ood_eval_root is None else str(candidate.ood_eval_root),
        "source_metadata": dict(candidate.metadata),
    }
    if extra:
        payload.update(dict(extra))
    return _json_safe(payload)


def write_posthoc_run_result(
    *,
    output_dir: str | Path,
    result: dict[str, Any],
) -> Path:
    """Persist one per-run posthoc JSON result."""

    out = Path(output_dir)
    path = posthoc_run_result_path(
        output_dir=out,
        run_name=str(result["run_name"]),
        checkpoint_tag=str(result["checkpoint_tag"]),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(result), indent=2) + "\n", encoding="utf-8")
    return path


def read_posthoc_run_result(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def read_posthoc_run_results(output_dir: str | Path) -> list[dict[str, Any]]:
    """Load all per-run posthoc JSON results under one output directory."""

    runs_dir = Path(output_dir) / "runs"
    if not runs_dir.exists():
        return []
    results: list[dict[str, Any]] = []
    for path in sorted(runs_dir.glob("*.json")):
        result = read_posthoc_run_result(path)
        if result:
            results.append(result)
    return results


def _metric_value(section: dict[str, Any] | None, metric_name: str) -> Any:
    if not isinstance(section, dict):
        return None
    metrics = section.get("metrics")
    if not isinstance(metrics, dict):
        return None
    return metrics.get(metric_name)


def _section_value(section: dict[str, Any] | None, key: str) -> Any:
    return None if not isinstance(section, dict) else section.get(key)


def flatten_posthoc_run_result(
    *,
    result: dict[str, Any],
    run_result_path: str | Path | None = None,
) -> dict[str, Any]:
    """Flatten a nested posthoc run JSON into one summary row."""

    id_section = result.get("id") if isinstance(result.get("id"), dict) else None
    ood_section = result.get("ood") if isinstance(result.get("ood"), dict) else None
    id_mse = _metric_value(id_section, "mse")
    id_rmse = _metric_value(id_section, "rmse")
    ood_mse = _metric_value(ood_section, "mse")
    ood_rmse = _metric_value(ood_section, "rmse")
    return {
        "experiment_name": result.get("experiment_name"),
        "batch_id": result.get("batch_id"),
        "run_name": result.get("run_name"),
        "status": result.get("status"),
        "missing": ",".join(str(item) for item in result.get("missing", []) or []),
        "error": result.get("error"),
        "checkpoint_tag": result.get("checkpoint_tag"),
        "eval_split": result.get("eval_split"),
        "ood_mode": result.get("ood_mode"),
        "run_dir": result.get("run_dir"),
        "checkpoint_path": result.get("checkpoint_path"),
        "config_path": result.get("config_path"),
        "id_eval_root": result.get("id_eval_root"),
        "ood_eval_root": result.get("ood_eval_root"),
        "id_n_trajectories": _section_value(id_section, "n_trajectories"),
        "id_n_rows": _section_value(id_section, "n_rows"),
        "id_mse": id_mse,
        "id_rmse": id_rmse,
        "id_mae": _metric_value(id_section, "mae"),
        "id_max_abs_error": _metric_value(id_section, "max_abs_error"),
        "id_trajectory_mse_mean": _metric_value(id_section, "trajectory_mse_mean"),
        "id_trajectory_mse_p90": _metric_value(id_section, "trajectory_mse_p90"),
        "id_trajectory_mse_p95": _metric_value(id_section, "trajectory_mse_p95"),
        "id_trajectory_mse_max": _metric_value(id_section, "trajectory_mse_max"),
        "id_trajectory_rmse_p90": _metric_value(id_section, "trajectory_rmse_p90"),
        "id_trajectory_rmse_p95": _metric_value(id_section, "trajectory_rmse_p95"),
        "id_trajectory_rmse_max": _metric_value(id_section, "trajectory_rmse_max"),
        "ood_n_trajectories_before_filter": _section_value(ood_section, "n_trajectories_before_filter"),
        "ood_n_trajectories": _section_value(ood_section, "n_trajectories"),
        "ood_n_rows_before_filter": _section_value(ood_section, "n_rows_before_filter"),
        "ood_n_rows": _section_value(ood_section, "n_rows"),
        "ood_mse": ood_mse,
        "ood_rmse": ood_rmse,
        "ood_mae": _metric_value(ood_section, "mae"),
        "ood_max_abs_error": _metric_value(ood_section, "max_abs_error"),
        "ood_trajectory_mse_mean": _metric_value(ood_section, "trajectory_mse_mean"),
        "ood_trajectory_mse_p90": _metric_value(ood_section, "trajectory_mse_p90"),
        "ood_trajectory_mse_p95": _metric_value(ood_section, "trajectory_mse_p95"),
        "ood_trajectory_mse_max": _metric_value(ood_section, "trajectory_mse_max"),
        "ood_trajectory_rmse_p90": _metric_value(ood_section, "trajectory_rmse_p90"),
        "ood_trajectory_rmse_p95": _metric_value(ood_section, "trajectory_rmse_p95"),
        "ood_trajectory_rmse_max": _metric_value(ood_section, "trajectory_rmse_max"),
        "id_ood_mse_gap": None if id_mse is None or ood_mse is None else float(ood_mse) - float(id_mse),
        "id_ood_rmse_gap": None if id_rmse is None or ood_rmse is None else float(ood_rmse) - float(id_rmse),
        "run_result_path": None if run_result_path is None else str(run_result_path),
    }


def write_posthoc_batch_outputs(
    *,
    output_dir: str | Path,
    run_results: list[dict[str, Any]],
    settings: dict[str, Any],
) -> dict[str, Path]:
    """Write batch summary CSV/JSON and failures JSON from per-run results."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for result in run_results:
        run_path = posthoc_run_result_path(
            output_dir=out,
            run_name=str(result["run_name"]),
            checkpoint_tag=str(result["checkpoint_tag"]),
        )
        rows.append(flatten_posthoc_run_result(result=result, run_result_path=run_path))
        if result.get("status") not in {"completed", "pending_evaluation"}:
            failures.append(
                {
                    "experiment_name": result.get("experiment_name"),
                    "batch_id": result.get("batch_id"),
                    "run_name": result.get("run_name"),
                    "status": result.get("status"),
                    "missing": result.get("missing", []),
                    "error": result.get("error"),
                }
            )

    summary_csv = out / "summary.csv"
    summary_json = out / "summary.json"
    failures_json = out / "failures.json"

    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=POSTHOC_SUMMARY_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json_safe(row.get(key)) for key in POSTHOC_SUMMARY_FIELDNAMES})

    summary_payload = {
        "schema_version": "posthoc_checkpoint_eval_summary_v1",
        "created_at_utc": utc_now_iso(),
        "settings": _json_safe(settings),
        "summary_csv": str(summary_csv),
        "failures_json": str(failures_json),
        "runs": rows,
    }
    summary_json.write_text(json.dumps(_json_safe(summary_payload), indent=2) + "\n", encoding="utf-8")

    failures_payload = {
        "schema_version": "posthoc_checkpoint_eval_failures_v1",
        "created_at_utc": utc_now_iso(),
        "settings": _json_safe(settings),
        "failures": failures,
    }
    failures_json.write_text(json.dumps(_json_safe(failures_payload), indent=2) + "\n", encoding="utf-8")

    return {
        "summary_csv": summary_csv,
        "summary_json": summary_json,
        "failures_json": failures_json,
    }


def load_supervised_evaluation_split(
    *,
    dataset_root: str | Path,
    split: str,
    dtype: str = "float64",
) -> SupervisedEvaluationSplit:
    """Load supervised rows for a train/val/test split from a preprocessed root."""

    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}.")

    root = Path(dataset_root).resolve()
    split_dir = root / split
    torch_dtype = resolve_torch_dtype(dtype)
    x_parts: list[torch.Tensor] = []
    y_parts: list[torch.Tensor] = []
    trajectory_parts: list[torch.Tensor] = []
    trajectory_key_missing = False

    for path in _iter_supervised_h5_files(split_dir):
        with h5py.File(path, "r") as h5f:
            x_parts.append(torch.tensor(h5f[H5_X_KEYS[split]][:], dtype=torch_dtype))
            y_parts.append(torch.tensor(h5f[H5_Y_KEYS[split]][:], dtype=torch_dtype))
            trajectory_key = H5_TRAJECTORY_ID_KEYS[split]
            if trajectory_key in h5f and not trajectory_key_missing:
                trajectory_parts.append(torch.tensor(h5f[trajectory_key][:], dtype=torch.int64))
            else:
                trajectory_key_missing = True
                trajectory_parts = []

    if not x_parts or not y_parts:
        raise ValueError(f"No supervised {split} HDF5 files found in {split_dir}")

    x = torch.cat(x_parts, dim=0)
    y = torch.cat(y_parts, dim=0)
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"Evaluation x/y row count mismatch in {split_dir}")

    trajectory_ids = None
    if trajectory_parts and not trajectory_key_missing:
        trajectory_ids = torch.cat(trajectory_parts, dim=0)
        if trajectory_ids.shape[0] != x.shape[0]:
            raise ValueError(f"Evaluation trajectory_id row count mismatch in {split_dir}")

    return SupervisedEvaluationSplit(
        dataset_root=str(root),
        split=split,
        x=x,
        y=y,
        trajectory_ids=trajectory_ids,
    )


def predict_checkpoint_on_evaluation_split(
    *,
    candidate: PosthocRunCandidate,
    dataset_root: str | Path,
    split: str,
    batch_size: int = 4096,
) -> PosthocPrediction:
    """Load a PINN checkpoint on CPU and predict one external evaluation split."""

    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive.")
    if not candidate.checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {candidate.checkpoint_path}")

    pinn_model = PinnModel.load_checkpoint(str(candidate.checkpoint_path), device_preference="cpu")
    return predict_pinn_model_on_evaluation_split(
        pinn_model=pinn_model,
        checkpoint_path=candidate.checkpoint_path,
        dataset_root=dataset_root,
        split=split,
        batch_size=batch_size,
    )


def predict_pinn_model_on_evaluation_split(
    *,
    pinn_model: PinnModel,
    checkpoint_path: str | Path,
    dataset_root: str | Path,
    split: str,
    batch_size: int = 4096,
) -> PosthocPrediction:
    """Predict one external evaluation split with an already-loaded PINN."""

    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive.")
    eval_split = load_supervised_evaluation_split(
        dataset_root=dataset_root,
        split=split,
        dtype=str(pinn_model.dtype).replace("torch.", ""),
    )
    model = pinn_model.model
    model.eval()
    predictions: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, int(eval_split.x.shape[0]), int(batch_size)):
            stop = min(start + int(batch_size), int(eval_split.x.shape[0]))
            xb = eval_split.x[start:stop].to(device=pinn_model.device, dtype=pinn_model.dtype)
            pred = model(xb).detach().cpu()
            predictions.append(pred)
    if not predictions:
        raise ValueError("Cannot predict an empty evaluation split.")
    y_pred = torch.cat(predictions, dim=0)
    if y_pred.shape != eval_split.y.shape:
        raise ValueError(f"Prediction shape {tuple(y_pred.shape)} does not match y shape {tuple(eval_split.y.shape)}.")
    return PosthocPrediction(
        split=eval_split,
        y_pred=y_pred,
        checkpoint_path=str(checkpoint_path),
        batch_size=int(batch_size),
    )


def evaluate_posthoc_prediction(
    *,
    prediction: PosthocPrediction,
    kind: str,
    ood_mode: str = "exclusive",
    model_flag: str | None = None,
    init_condition_bounds: int = 1,
    init_conditions_dir: str | Path = "src/config/ic",
) -> dict[str, Any]:
    """Compute posthoc metrics for one ID or OOD prediction payload."""

    if kind not in {"id", "ood"}:
        raise ValueError("kind must be 'id' or 'ood'.")
    if ood_mode not in {"exclusive", "wide"}:
        raise ValueError("ood_mode must be one of: exclusive, wide.")

    slices_before = trajectory_slices_from_rows(
        x=prediction.split.x,
        trajectory_ids=prediction.split.trajectory_ids,
    )
    x_eval = prediction.split.x
    y_eval = prediction.split.y
    pred_eval = prediction.y_pred
    slices_eval = slices_before
    n_rows_before = int(x_eval.shape[0])
    n_trajectories_before = int(len(slices_before))
    filter_info: dict[str, Any] = {
        "mode": "none" if kind == "id" else ood_mode,
        "applied": False,
    }

    if kind == "ood" and ood_mode == "exclusive":
        if model_flag in (None, ""):
            raise ValueError("model_flag is required for exclusive OOD filtering.")
        bounds = load_ic_bounds_from_config(
            model_flag=str(model_flag),
            init_condition_bounds=int(init_condition_bounds),
            init_conditions_dir=init_conditions_dir,
        )
        keep_mask = trajectory_exclusive_ood_mask(
            x=x_eval,
            trajectory_slices=slices_before,
            training_bounds=bounds,
        )
        row_indices_np = row_indices_for_trajectory_mask(
            trajectory_slices=slices_before,
            keep_mask=keep_mask,
        )
        if row_indices_np.size == 0:
            raise ValueError("Exclusive OOD filtering removed all trajectories.")
        row_indices = torch.as_tensor(row_indices_np, dtype=torch.long)
        x_eval = x_eval.index_select(0, row_indices)
        y_eval = y_eval.index_select(0, row_indices)
        pred_eval = pred_eval.index_select(0, row_indices)
        # Row indices are compacted after filtering, so rebuild slices from time
        # resets instead of reusing original absolute spans.
        slices_eval = trajectory_slices_from_rows(x=x_eval)
        filter_info = {
            "mode": "exclusive",
            "applied": True,
            "model_flag": str(model_flag),
            "init_condition_bounds": int(init_condition_bounds),
            "n_trajectories_removed": int((~keep_mask).sum()),
            "n_trajectories_kept": int(keep_mask.sum()),
        }

    metrics = posthoc_regression_metrics_from_arrays(
        y_true=y_eval,
        y_pred=pred_eval,
        trajectory_slices=slices_eval,
    )
    return {
        "kind": kind,
        "dataset_root": prediction.split.dataset_root,
        "split": prediction.split.split,
        "checkpoint_path": prediction.checkpoint_path,
        "batch_size": int(prediction.batch_size),
        "n_rows_before_filter": n_rows_before,
        "n_trajectories_before_filter": n_trajectories_before,
        "n_rows": int(metrics["n_rows"]),
        "n_trajectories": int(metrics["n_trajectories"]),
        "filter": filter_info,
        "metrics": metrics,
    }


def trajectory_slices_from_rows(
    *,
    x: torch.Tensor | np.ndarray,
    trajectory_ids: torch.Tensor | np.ndarray | None = None,
) -> list[TrajectorySlice]:
    """Recover contiguous trajectory spans from IDs, falling back to time resets."""

    x_np = x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    if x_np.ndim != 2 or x_np.shape[0] == 0:
        raise ValueError("x must be a non-empty 2D row array.")
    n_rows = int(x_np.shape[0])

    if trajectory_ids is not None:
        ids_np = trajectory_ids.detach().cpu().numpy() if torch.is_tensor(trajectory_ids) else np.asarray(trajectory_ids)
        if ids_np.ndim != 1 or ids_np.shape[0] != n_rows:
            raise ValueError("trajectory_ids must be rank-1 and match x rows.")
        boundaries = np.flatnonzero(ids_np[1:] != ids_np[:-1]) + 1
        starts = np.concatenate(([0], boundaries))
        stops = np.concatenate((boundaries, [n_rows]))
        return [
            TrajectorySlice(start=int(start), stop=int(stop), trajectory_id=int(ids_np[start]))
            for start, stop in zip(starts, stops)
            if int(stop) > int(start)
        ]

    time = x_np[:, 0]
    boundaries = np.flatnonzero(time[1:] <= time[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    stops = np.concatenate((boundaries, [n_rows]))
    return [
        TrajectorySlice(start=int(start), stop=int(stop), trajectory_id=None)
        for start, stop in zip(starts, stops)
        if int(stop) > int(start)
    ]


def trajectory_initial_conditions(
    *,
    x: torch.Tensor | np.ndarray,
    trajectory_slices: list[TrajectorySlice],
) -> np.ndarray:
    """Return one IC vector per trajectory using each trajectory's first row."""

    x_np = x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    if x_np.ndim != 2 or x_np.shape[1] < 2:
        raise ValueError("x must contain [time, initial-condition/features...] rows.")
    return np.asarray([x_np[item.start, 1:] for item in trajectory_slices], dtype=np.float64)


def trajectory_exclusive_ood_mask(
    *,
    x: torch.Tensor | np.ndarray,
    trajectory_slices: list[TrajectorySlice],
    training_bounds: np.ndarray,
    atol: float = 1e-8,
) -> np.ndarray:
    """Mark trajectories whose IC is outside the training bounds in any dimension."""

    ics = trajectory_initial_conditions(x=x, trajectory_slices=trajectory_slices)
    bounds = np.asarray(training_bounds, dtype=np.float64)
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError("training_bounds must have shape (n_features, 2).")
    if bounds.shape[0] != ics.shape[1]:
        raise ValueError(f"training_bounds feature count {bounds.shape[0]} does not match IC count {ics.shape[1]}.")

    lower = bounds[:, 0] - float(atol)
    upper = bounds[:, 1] + float(atol)
    inside = np.all((ics >= lower) & (ics <= upper), axis=1)
    return ~inside


def row_indices_for_trajectory_mask(
    *,
    trajectory_slices: list[TrajectorySlice],
    keep_mask: np.ndarray,
) -> np.ndarray:
    """Expand a per-trajectory keep mask into row indices."""

    keep = np.asarray(keep_mask, dtype=bool)
    if keep.ndim != 1 or keep.shape[0] != len(trajectory_slices):
        raise ValueError("keep_mask must be rank-1 and match trajectory_slices.")
    parts = [
        np.arange(item.start, item.stop, dtype=np.int64)
        for item, should_keep in zip(trajectory_slices, keep)
        if bool(should_keep)
    ]
    if not parts:
        return np.empty((0,), dtype=np.int64)
    return np.concatenate(parts)


def posthoc_regression_metrics_from_arrays(
    *,
    y_true: torch.Tensor | np.ndarray,
    y_pred: torch.Tensor | np.ndarray,
    trajectory_slices: list[TrajectorySlice],
) -> dict[str, float | int | None]:
    """Compute global and trajectory-level checkpoint evaluation metrics."""

    true = y_true.detach().cpu().numpy() if torch.is_tensor(y_true) else np.asarray(y_true)
    pred = y_pred.detach().cpu().numpy() if torch.is_tensor(y_pred) else np.asarray(y_pred)
    true = np.asarray(true, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if true.shape != pred.shape:
        raise ValueError(f"Regression metric arrays must match, got {true.shape} vs {pred.shape}.")
    if true.ndim < 2 or true.shape[0] == 0:
        raise ValueError("Metric arrays must be non-empty with rows on axis 0.")

    diff = pred - true
    abs_diff = np.abs(diff)
    mse = float(np.mean(diff**2))
    trajectory_mse: list[float] = []
    trajectory_rmse: list[float] = []
    trajectory_max_abs: list[float] = []
    for item in trajectory_slices:
        if item.stop <= item.start:
            continue
        segment = diff[item.start : item.stop]
        segment_mse = float(np.mean(segment**2))
        trajectory_mse.append(segment_mse)
        trajectory_rmse.append(float(np.sqrt(segment_mse)))
        trajectory_max_abs.append(float(np.max(np.abs(segment))))
    if not trajectory_rmse:
        raise ValueError("At least one non-empty trajectory is required.")

    mse_values = np.asarray(trajectory_mse, dtype=np.float64)
    rmse_values = np.asarray(trajectory_rmse, dtype=np.float64)
    max_abs_values = np.asarray(trajectory_max_abs, dtype=np.float64)
    worst_idx = int(np.argmax(rmse_values))
    worst_slice = trajectory_slices[worst_idx]

    return {
        "n_rows": int(true.shape[0]),
        "n_values": int(true.size),
        "n_trajectories": int(rmse_values.shape[0]),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(abs_diff)),
        "max_abs_error": float(np.max(abs_diff)),
        "trajectory_mse_mean": float(np.mean(mse_values)),
        "trajectory_mse_p90": float(np.percentile(mse_values, 90)),
        "trajectory_mse_p95": float(np.percentile(mse_values, 95)),
        "trajectory_mse_max": float(mse_values[worst_idx]),
        "trajectory_rmse_mean": float(np.mean(rmse_values)),
        "trajectory_rmse_p90": float(np.percentile(rmse_values, 90)),
        "trajectory_rmse_p95": float(np.percentile(rmse_values, 95)),
        "trajectory_rmse_max": float(rmse_values[worst_idx]),
        "trajectory_max_abs_error_mean": float(np.mean(max_abs_values)),
        "trajectory_max_abs_error_p95": float(np.percentile(max_abs_values, 95)),
        "trajectory_max_abs_error_max": float(max_abs_values[worst_idx]),
        "worst_trajectory_index": worst_idx,
        "worst_trajectory_id": worst_slice.trajectory_id,
        "worst_trajectory_row_start": int(worst_slice.start),
        "worst_trajectory_row_stop": int(worst_slice.stop),
    }


def load_ic_bounds_from_config(
    *,
    model_flag: str,
    init_condition_bounds: int = 1,
    init_conditions_dir: str | Path = "src/config/ic",
) -> np.ndarray:
    """Load model IC bounds from the repository config files."""

    path = Path(init_conditions_dir) / str(model_flag) / f"init_cond{int(init_condition_bounds)}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"IC bounds file not found: {path}")
    entries = OmegaConf.load(path)
    bounds: list[list[float]] = []
    for entry in entries:
        values = list(entry["range"])
        if len(values) == 1:
            value = float(values[0])
            bounds.append([value, value])
        else:
            bounds.append([float(values[0]), float(values[1])])
    return np.asarray(bounds, dtype=np.float64)
