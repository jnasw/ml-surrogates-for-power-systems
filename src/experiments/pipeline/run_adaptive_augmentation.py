"""Thesis experiment: adaptive supervised-data and collocation augmentation.

Run single-stage PINN experiments where supervised trajectories are revealed from a
preprocessed labelled pool while collocation points are optionally grown with RAR-D.

Fixed:  reference/preprocessed dataset, architecture, Adam optimizer, static weights.
Vary:   supervised acquisition strategy x collocation growth strategy x seed.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.data.contracts.data_contract import (
    H5_COLLOCATION_SUFFIX,
    H5_DATA_SUFFIX,
    H5_FILE_SUFFIX,
    H5_INIT_SUFFIX,
    H5_TRAJECTORY_ID_KEYS,
    H5_X_KEYS,
    TRAIN_SPLIT,
)
from src.experiments.pipeline.helpers.evaluation import resolve_eval_inputs as _resolve_shared_eval_inputs
from src.experiments.pipeline.helpers.launch_utils import (
    format_hydra_list,
    init_experiment_manifest,
    parse_csv_list,
    run_logged_command,
    tag_stamp,
    upsert_run_status,
)
from src.experiments.pipeline.helpers.manifest import save_manifest, utc_now_iso
from src.experiments.pipeline.helpers.reference import resolve_reference_dataset as _resolve_reference_dataset
from src.experiments.pipeline.helpers.seeds import (
    parse_label_list as _parse_label_list,
    seed_pairs_from_labels as _seed_pairs_from_labels,
)
from src.experiments.pipeline.helpers.summary import (
    STANDARD_PINN_SUMMARY_FIELDNAMES,
    aggregate_standard_pinn_metrics as _aggregate_standard_pinn_metrics,
    component_loss as _component_loss,
    extract_standard_pinn_summary_fields as _extract_standard_pinn_summary_fields,
    float_or_none as _float_or_none,
    int_or_none as _int_or_none,
    mean_std as _mean_std,
    read_json_if_exists as _read_json_if_exists,
    write_failures_json,
    write_summary_csv,
    write_summary_json,
)
from src.experiments.pipeline.helpers.wandb import (
    default_project_for_mode,
    group_name,
    tags as wandb_tags_list,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REFERENCE_ID = "main_SM4_lhs_b4096_ds01"
DEFAULT_OOD_EVAL_ID = "ood_SM4_wide_ic_b512_ds01"
DEFAULT_ADAM_LR = 3.0e-3

SCREENING_SEED_LABELS = ("s01",)
FINAL_SEED_LABELS = ("s01", "s02", "s03", "s04", "s05")
SCREENING_DEFAULT_EPOCHS = 100
FINAL_DEFAULT_EPOCHS = 20_000
SCREENING_DEFAULT_CADENCE = 10
FINAL_DEFAULT_CADENCE = 500
SCREENING_DEFAULT_INITIAL_TRAJECTORIES = 64
SCREENING_DEFAULT_ADD_TRAJECTORIES = 16
SCREENING_DEFAULT_FINAL_TRAJECTORIES = 128
FINAL_DEFAULT_INITIAL_TRAJECTORIES = 256
FINAL_DEFAULT_ADD_TRAJECTORIES = 32
FINAL_DEFAULT_FINAL_TRAJECTORIES = 512

SUPERVISED_STRATEGIES = ("fixed_low", "random_growth", "mae_nearest_growth", "fixed_full")
SCREENING_DEFAULT_SUPERVISED_STRATEGIES = ("fixed_low", "random_growth", "mae_nearest_growth")
FINAL_DEFAULT_SUPERVISED_STRATEGIES = ("fixed_low", "random_growth", "mae_nearest_growth", "fixed_full")
COLLOCATION_STRATEGIES = ("static_low", "rar_d_growth")
DEFAULT_COLLOCATION_STRATEGIES = ("static_low", "rar_d_growth")

SUMMARY_FIELDNAMES = [
    "run_name",
    "supervised_strategy",
    "supervised_acquisition_strategy",
    "supervised_initial_trajectories",
    "supervised_add_trajectories",
    "supervised_max_trajectories",
    "collocation_strategy",
    "collocation_initial_points",
    "collocation_add_points",
    "collocation_final_points",
    "collocation_candidate_points",
    "refresh_period_epochs",
    "seed_label",
    "seed_value",
    "dataset_reference_id",
    "dataset_root",
    "id_eval_id",
    "id_eval_root",
    "ood_eval_id",
    "ood_eval_root",
    "run_dir",
    "status",
    "return_code",
    "failure_reason",
    "log_file",
    *STANDARD_PINN_SUMMARY_FIELDNAMES,
    "final_train_total_loss",
    "final_train_data_loss",
    "final_train_physics_loss",
    "final_train_dt_loss",
    "final_train_ic_loss",
    "test_data_loss",
    "global_epoch",
    "num_train_steps",
    "num_supervised_rows",
    "num_collocation_rows",
    "supervised_active_trajectories",
    "supervised_candidate_trajectories",
    "supervised_active_rows",
    "supervised_acquired_trajectories",
    "supervised_acquisition_count",
    "supervised_last_anchor_score_mean",
    "supervised_last_anchor_score_max",
    "supervised_last_selected_distance_mean",
    "supervised_last_selected_distance_max",
    "epoch_wall_seconds",
    "cumulative_wall_seconds",
    "total_seconds",
    "training_seconds",
]


@dataclass(frozen=True)
class AdaptiveAugmentationRunSpec:
    supervised_strategy: str
    supervised_acquisition_strategy: str | None
    supervised_initial_trajectories: int | None
    supervised_add_trajectories: int
    supervised_max_trajectories: int | None
    collocation_strategy: str
    collocation_initial_points: int
    collocation_add_points: int
    collocation_final_points: int
    collocation_candidate_points: int | None
    refresh_period_epochs: int
    seed_label: str
    seed_value: int

    @property
    def run_name(self) -> str:
        return f"{self.supervised_strategy}_{self.collocation_strategy}_c{self.refresh_period_epochs}_{self.seed_label}"

    def wandb_tags(self) -> list[str]:
        return [self.supervised_strategy, self.collocation_strategy, f"c{self.refresh_period_epochs}", self.seed_label]


def _parse_named_list(raw: str | None, *, defaults: tuple[str, ...], allowed: tuple[str, ...], flag: str) -> list[str]:
    values = list(defaults) if raw in (None, "") else [item.strip().lower() for item in parse_csv_list(str(raw))]
    if not values:
        raise ValueError(f"{flag} must contain at least one value.")
    unsupported = [value for value in values if value not in allowed]
    if unsupported:
        raise ValueError(f"Unsupported {flag}: {', '.join(unsupported)}. Use one of: {', '.join(allowed)}.")
    if len(set(values)) != len(values):
        raise ValueError(f"{flag} must not contain duplicates.")
    return values


def _default_supervised_strategies_for_mode(mode: str) -> tuple[str, ...]:
    return SCREENING_DEFAULT_SUPERVISED_STRATEGIES if mode == "screening" else FINAL_DEFAULT_SUPERVISED_STRATEGIES


def _default_supervised_budgets_for_mode(mode: str) -> tuple[int, int, int]:
    if mode == "screening":
        return (
            SCREENING_DEFAULT_INITIAL_TRAJECTORIES,
            SCREENING_DEFAULT_ADD_TRAJECTORIES,
            SCREENING_DEFAULT_FINAL_TRAJECTORIES,
        )
    return (
        FINAL_DEFAULT_INITIAL_TRAJECTORIES,
        FINAL_DEFAULT_ADD_TRAJECTORIES,
        FINAL_DEFAULT_FINAL_TRAJECTORIES,
    )


def _resolve_seed_pairs(args: argparse.Namespace, *, mode: str) -> list[tuple[str, int]]:
    if args.seed_labels:
        return _seed_pairs_from_labels(_parse_label_list(str(args.seed_labels)))
    if args.seed is not None:
        return [(f"raw{int(args.seed)}", int(args.seed))]
    return _seed_pairs_from_labels(SCREENING_SEED_LABELS if mode == "screening" else FINAL_SEED_LABELS)


def _resolve_eval_inputs(args: argparse.Namespace) -> dict[str, str | None]:
    return _resolve_shared_eval_inputs(args, default_ood_eval_id=DEFAULT_OOD_EVAL_ID)


def _resolve_dataset(args: argparse.Namespace, *, manifest: dict[str, Any], manifest_path: Path) -> tuple[Path, str | None]:
    if args.dataset_root:
        dataset_root = Path(args.dataset_root).resolve()
        manifest["artifacts"]["dataset_root"] = str(dataset_root)
        manifest["artifacts"]["dataset_source"] = "provided"
        save_manifest(str(manifest_path), manifest)
        return dataset_root, None

    reference_id = str(args.reference_id or DEFAULT_REFERENCE_ID)
    reference_entry = _resolve_reference_dataset(reference_id)
    dataset_root = Path(str(reference_entry["preprocessed_root"])).resolve()
    manifest["artifacts"]["dataset_root"] = str(dataset_root)
    manifest["artifacts"]["dataset_source"] = "reference"
    manifest["artifacts"]["dataset_reference_id"] = reference_id
    manifest["artifacts"]["dataset_reference"] = reference_entry
    save_manifest(str(manifest_path), manifest)
    return dataset_root, reference_id


def _iter_train_h5_files(dataset_root: Path, suffix: str) -> list[Path]:
    train_dir = dataset_root / TRAIN_SPLIT
    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    out: list[Path] = []
    for fn in sorted(os.listdir(str(train_dir))):
        if not fn.endswith(H5_FILE_SUFFIX):
            continue
        if suffix == H5_DATA_SUFFIX:
            if H5_DATA_SUFFIX not in fn or H5_COLLOCATION_SUFFIX in fn or H5_INIT_SUFFIX in fn:
                continue
        elif suffix not in fn:
            continue
        out.append(train_dir / fn)
    return out


def _count_training_trajectories(dataset_root: Path) -> int:
    import h5py

    trajectory_ids: set[int] = set()
    paths = _iter_train_h5_files(dataset_root, H5_DATA_SUFFIX)
    if not paths:
        raise FileNotFoundError(f"No supervised train HDF5 files found under {dataset_root / TRAIN_SPLIT}.")
    for path in paths:
        with h5py.File(str(path), "r") as h5f:
            key = H5_TRAJECTORY_ID_KEYS[TRAIN_SPLIT]
            if key not in h5f:
                raise ValueError(
                    f"{path} does not contain '{key}'. Adaptive supervised acquisition requires trajectory metadata."
                )
            trajectory_ids.update(int(value) for value in h5f[key][:].tolist())
    if not trajectory_ids:
        raise ValueError("No trajectory IDs found in supervised training files.")
    return len(trajectory_ids)


def _count_preprocessed_collocation_rows(dataset_root: Path) -> int:
    import h5py

    total = 0
    paths = _iter_train_h5_files(dataset_root, H5_COLLOCATION_SUFFIX)
    for path in paths:
        with h5py.File(str(path), "r") as h5f:
            if H5_X_KEYS[TRAIN_SPLIT] in h5f:
                total += int(h5f[H5_X_KEYS[TRAIN_SPLIT]].shape[0])
    return total


def _build_run_specs(
    *,
    supervised_strategies: list[str],
    collocation_strategies: list[str],
    seed_pairs: list[tuple[str, int]],
    refresh_period_epochs: int,
    supervised_initial_trajectories: int,
    supervised_add_trajectories: int,
    supervised_final_trajectories: int,
    collocation_initial_points: int,
    collocation_add_points: int,
    collocation_final_points: int,
    collocation_candidate_points: int | None,
) -> list[AdaptiveAugmentationRunSpec]:
    specs: list[AdaptiveAugmentationRunSpec] = []
    for supervised_strategy in supervised_strategies:
        if supervised_strategy == "fixed_full":
            acquisition_strategy = None
            initial_trajectories = None
            max_trajectories = None
            add_trajectories = supervised_add_trajectories
        elif supervised_strategy == "fixed_low":
            acquisition_strategy = "random"
            initial_trajectories = supervised_initial_trajectories
            max_trajectories = supervised_initial_trajectories
            add_trajectories = supervised_add_trajectories
        elif supervised_strategy == "random_growth":
            acquisition_strategy = "random"
            initial_trajectories = supervised_initial_trajectories
            max_trajectories = supervised_final_trajectories
            add_trajectories = supervised_add_trajectories
        elif supervised_strategy == "mae_nearest_growth":
            acquisition_strategy = "mae_nearest"
            initial_trajectories = supervised_initial_trajectories
            max_trajectories = supervised_final_trajectories
            add_trajectories = supervised_add_trajectories
        else:
            raise ValueError(f"Unsupported supervised strategy: {supervised_strategy}")

        for collocation_strategy in collocation_strategies:
            for seed_label, seed_value in seed_pairs:
                specs.append(
                    AdaptiveAugmentationRunSpec(
                        supervised_strategy=supervised_strategy,
                        supervised_acquisition_strategy=acquisition_strategy,
                        supervised_initial_trajectories=initial_trajectories,
                        supervised_add_trajectories=add_trajectories,
                        supervised_max_trajectories=max_trajectories,
                        collocation_strategy=collocation_strategy,
                        collocation_initial_points=collocation_initial_points,
                        collocation_add_points=collocation_add_points,
                        collocation_final_points=collocation_final_points,
                        collocation_candidate_points=collocation_candidate_points,
                        refresh_period_epochs=refresh_period_epochs,
                        seed_label=seed_label,
                        seed_value=int(seed_value),
                    )
                )
    return specs


def _optimizer_phase_override(*, epochs: int, lr: float, batch_size: int) -> str:
    return (
        "pinn.optimizer_phases=["
        "{"
        "name:adam,optimizer:Adam,"
        f"lr:{float(lr)},epochs:{int(epochs)},batch_size:{int(batch_size)},"
        "shuffle:true,full_batch:false,allow_sampling:true,"
        "optimizer_kwargs:{eps:1.0e-6},scheduler:null,line_search:null,convergence:null"
        "}]"
    )


def _supervised_overrides(run_spec: AdaptiveAugmentationRunSpec, *, candidate_batch_size: int) -> list[str]:
    if run_spec.supervised_acquisition_strategy is None:
        return ["pinn.supervised_acquisition.enabled=false"]
    return [
        "pinn.supervised_acquisition.enabled=true",
        f"pinn.supervised_acquisition.strategy={run_spec.supervised_acquisition_strategy}",
        f"pinn.supervised_acquisition.initial_trajectories={int(run_spec.supervised_initial_trajectories)}",
        f"pinn.supervised_acquisition.add_trajectories={int(run_spec.supervised_add_trajectories)}",
        f"pinn.supervised_acquisition.max_trajectories={int(run_spec.supervised_max_trajectories)}",
        f"pinn.supervised_acquisition.refresh_period_epochs={int(run_spec.refresh_period_epochs)}",
        f"pinn.supervised_acquisition.candidate_batch_size={int(candidate_batch_size)}",
        f"pinn.supervised_acquisition.anchor_trajectories={int(run_spec.supervised_add_trajectories)}",
        "pinn.supervised_acquisition.similarity_space=initial_condition",
    ]


def _collocation_overrides(
    run_spec: AdaptiveAugmentationRunSpec,
    *,
    sampler: str,
    score_norm: str,
    rar_d_k: float,
    rar_d_c: float,
) -> list[str]:
    if run_spec.collocation_strategy == "static_low":
        return [
            "pinn.collocation.mode=preprocessed",
            "pinn.collocation.strategy=static",
            f"pinn.collocation.active_points={int(run_spec.collocation_initial_points)}",
            "pinn.collocation.initial_points=null",
            "pinn.collocation.refresh_period_epochs=0",
            "pinn.collocation.sampling.enabled=false",
        ]
    if run_spec.collocation_strategy == "rar_d_growth":
        candidate_points = (
            int(run_spec.collocation_candidate_points)
            if run_spec.collocation_candidate_points is not None
            else max(int(run_spec.collocation_final_points) * 4, 256)
        )
        return [
            "pinn.collocation.mode=generated",
            "pinn.collocation.strategy=rar_d",
            f"pinn.collocation.active_points={int(run_spec.collocation_final_points)}",
            f"pinn.collocation.initial_points={int(run_spec.collocation_initial_points)}",
            f"pinn.collocation.append_points={int(run_spec.collocation_add_points)}",
            f"pinn.collocation.candidate_points={candidate_points}",
            f"pinn.collocation.refresh_period_epochs={int(run_spec.refresh_period_epochs)}",
            "pinn.collocation.refresh.mode=epoch_periodic",
            "pinn.collocation.sampling.enabled=false",
            f"pinn.collocation.sampler={sampler}",
            f"pinn.collocation.score_norm={score_norm}",
            f"pinn.collocation.rad.k={float(rar_d_k)}",
            f"pinn.collocation.rad.c={float(rar_d_c)}",
        ]
    raise ValueError(f"Unsupported collocation strategy: {run_spec.collocation_strategy}")


def _build_pinn_command(
    *,
    python_bin: str,
    model_flag: str,
    dataset_root: Path,
    run_dir: Path,
    run_name: str,
    run_spec: AdaptiveAugmentationRunSpec,
    wandb_project: str,
    wandb_group: str,
    wandb_entity: str | None,
    wandb_tags: list[str],
    device: str,
    hidden_dim: int,
    hidden_layers: int,
    activation: str,
    dtype_name: str,
    epochs: int,
    batch_size: int,
    adam_lr: float,
    candidate_batch_size: int,
    collocation_sampler: str,
    score_norm: str,
    rar_d_k: float,
    rar_d_c: float,
    log_every_epoch: int,
    id_eval_root: str | None,
    ood_eval_root: str | None,
) -> list[str]:
    command = [
        python_bin,
        "20_run_pinn.py",
        f"model.model_flag={model_flag}",
        f"model.seed={int(run_spec.seed_value)}",
        "pinn.mode=single_stage",
        f"dataset.root={dataset_root}",
        f"pinn.run_dir={run_dir}",
        f"pinn.device={device}",
        f"pinn.dtype={dtype_name}",
        f"pinn.hidden_dim={int(hidden_dim)}",
        f"pinn.hidden_layers={int(hidden_layers)}",
        f"pinn.activation={activation}",
        f"pinn.default_batch_size={int(batch_size)}",
        "pinn.weighting.scheme=static",
        "pinn.supervised_sampling.enabled=false",
        "pinn.curriculum.enabled=false",
        "pinn.gradient_telemetry.enabled=false",
        "pinn.checkpointing.enabled=true",
        "pinn.checkpointing.save_best=true",
        "pinn.checkpointing.save_last=false",
        "pinn.checkpointing.save_init=false",
        "pinn.checkpointing.epoch_fractions=[]",
        "pinn.evaluation.frequency=25",
        f"logging.log_every_epoch={int(log_every_epoch)}",
        "wandb.use=true",
        f"wandb.project={wandb_project}",
        f"wandb.group={wandb_group}",
        f"wandb.name={run_name}",
        f"wandb.tags={format_hydra_list(wandb_tags)}",
        *_supervised_overrides(run_spec, candidate_batch_size=candidate_batch_size),
        *_collocation_overrides(
            run_spec,
            sampler=collocation_sampler,
            score_norm=score_norm,
            rar_d_k=rar_d_k,
            rar_d_c=rar_d_c,
        ),
        _optimizer_phase_override(epochs=epochs, lr=adam_lr, batch_size=batch_size),
    ]
    if wandb_entity:
        command.append(f"wandb.entity={wandb_entity}")
    if id_eval_root:
        command.append(f"evaluation.id.root={id_eval_root}")
    if ood_eval_root:
        command.append(f"evaluation.ood.root={ood_eval_root}")
    return command


def _summary_row_for_run(run: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(str(run.get("run_dir", ""))) if run.get("run_dir") else None
    metrics = _read_json_if_exists(run_dir / "metrics.json") if run_dir else None
    timings = _read_json_if_exists(run_dir / "timings.json") if run_dir else None
    final_train_losses = dict(metrics.get("final_train_losses", {}) or {}) if metrics else {}
    final_epoch = dict(metrics.get("final_epoch", {}) or {}) if metrics else {}
    standard_metrics = _extract_standard_pinn_summary_fields(metrics=metrics, run_dir=run_dir)
    return {
        "run_name": run.get("run_name"),
        "supervised_strategy": run.get("supervised_strategy"),
        "supervised_acquisition_strategy": run.get("supervised_acquisition_strategy"),
        "supervised_initial_trajectories": run.get("supervised_initial_trajectories"),
        "supervised_add_trajectories": run.get("supervised_add_trajectories"),
        "supervised_max_trajectories": run.get("supervised_max_trajectories"),
        "collocation_strategy": run.get("collocation_strategy"),
        "collocation_initial_points": run.get("collocation_initial_points"),
        "collocation_add_points": run.get("collocation_add_points"),
        "collocation_final_points": run.get("collocation_final_points"),
        "collocation_candidate_points": run.get("collocation_candidate_points"),
        "refresh_period_epochs": run.get("refresh_period_epochs"),
        "seed_label": run.get("seed_label"),
        "seed_value": run.get("seed_value"),
        "dataset_reference_id": run.get("dataset_reference_id"),
        "dataset_root": run.get("dataset_root"),
        "id_eval_id": run.get("id_eval_id"),
        "id_eval_root": run.get("id_eval_root"),
        "ood_eval_id": run.get("ood_eval_id"),
        "ood_eval_root": run.get("ood_eval_root"),
        "run_dir": run.get("run_dir"),
        "status": run.get("status"),
        "return_code": run.get("return_code"),
        "failure_reason": run.get("error"),
        "log_file": run.get("log_file"),
        **standard_metrics,
        "final_train_total_loss": _float_or_none(final_train_losses.get("total_loss")),
        "final_train_data_loss": _component_loss(final_train_losses, "data"),
        "final_train_physics_loss": _component_loss(final_train_losses, "physics"),
        "final_train_dt_loss": _component_loss(final_train_losses, "dt"),
        "final_train_ic_loss": _component_loss(final_train_losses, "ic"),
        "test_data_loss": _float_or_none(final_epoch.get("test_data_loss")),
        "global_epoch": _int_or_none(final_epoch.get("global_epoch")),
        "num_train_steps": _int_or_none(final_epoch.get("num_train_steps")),
        "num_supervised_rows": _int_or_none(final_epoch.get("num_supervised_rows")),
        "num_collocation_rows": _int_or_none(final_epoch.get("num_collocation_rows")),
        "supervised_active_trajectories": _int_or_none(final_epoch.get("supervised_active_trajectories")),
        "supervised_candidate_trajectories": _int_or_none(final_epoch.get("supervised_candidate_trajectories")),
        "supervised_active_rows": _int_or_none(final_epoch.get("supervised_active_rows")),
        "supervised_acquired_trajectories": _int_or_none(final_epoch.get("supervised_acquired_trajectories")),
        "supervised_acquisition_count": _int_or_none(final_epoch.get("supervised_acquisition_count")),
        "supervised_last_anchor_score_mean": _float_or_none(final_epoch.get("supervised_last_anchor_score_mean")),
        "supervised_last_anchor_score_max": _float_or_none(final_epoch.get("supervised_last_anchor_score_max")),
        "supervised_last_selected_distance_mean": _float_or_none(
            final_epoch.get("supervised_last_selected_distance_mean")
        ),
        "supervised_last_selected_distance_max": _float_or_none(
            final_epoch.get("supervised_last_selected_distance_max")
        ),
        "epoch_wall_seconds": _float_or_none(final_epoch.get("epoch_wall_seconds")),
        "cumulative_wall_seconds": _float_or_none(final_epoch.get("cumulative_wall_seconds")),
        "total_seconds": _float_or_none(timings.get("total_seconds") if timings else None),
        "training_seconds": _float_or_none(timings.get("training_seconds") if timings else None),
    }


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = sorted(
        {
            (str(row.get("supervised_strategy")), str(row.get("collocation_strategy")))
            for row in rows
            if row.get("supervised_strategy") and row.get("collocation_strategy")
        }
    )
    out: dict[str, Any] = {}
    for supervised_strategy, collocation_strategy in keys:
        members = [
            row for row in rows
            if row.get("supervised_strategy") == supervised_strategy
            and row.get("collocation_strategy") == collocation_strategy
        ]
        successful = [row for row in members if row.get("status") == "completed"]
        failed = [row for row in members if row.get("status") != "completed"]
        rmse_values = [float(row["final_test_rmse"]) for row in successful if row.get("final_test_rmse") is not None]
        training_seconds = [float(row["training_seconds"]) for row in successful if row.get("training_seconds") is not None]
        group_key = f"{supervised_strategy}/{collocation_strategy}"
        out[group_key] = {
            "supervised_strategy": supervised_strategy,
            "collocation_strategy": collocation_strategy,
            "success_count": len(successful),
            "failure_count": len(failed),
            "final_test_rmse_mean": _mean_std(rmse_values)["mean"],
            "final_test_rmse_std": _mean_std(rmse_values)["std"],
            "training_seconds_mean": _mean_std(training_seconds)["mean"],
            "training_seconds_std": _mean_std(training_seconds)["std"],
        }
        out[group_key].update(_aggregate_standard_pinn_metrics(successful))
    return out


def _write_summary_artifacts(*, output_root: Path, manifest: dict[str, Any]) -> dict[str, str]:
    rows = [_summary_row_for_run(dict(run)) for run in manifest.get("runs", [])]
    summary_csv = output_root / "summary.csv"
    summary_json = output_root / "summary.json"
    failures_json = output_root / "failures.json"

    write_summary_csv(summary_csv, rows=rows, fieldnames=SUMMARY_FIELDNAMES)
    failures = [
        {
            "run_name": row.get("run_name"),
            "supervised_strategy": row.get("supervised_strategy"),
            "collocation_strategy": row.get("collocation_strategy"),
            "seed_label": row.get("seed_label"),
            "run_dir": row.get("run_dir"),
            "status": row.get("status"),
            "return_code": row.get("return_code"),
            "failure_reason": row.get("failure_reason"),
            "log_file": row.get("log_file"),
        }
        for row in rows
        if row.get("status") != "completed"
    ]
    summary_payload = {
        "generated_at_utc": utc_now_iso(),
        "output_root": str(output_root),
        "mode": manifest.get("experiment", {}).get("mode"),
        "dataset": {
            "source": manifest.get("artifacts", {}).get("dataset_source"),
            "reference_id": manifest.get("artifacts", {}).get("dataset_reference_id"),
            "root": manifest.get("artifacts", {}).get("dataset_root"),
            "reference": manifest.get("artifacts", {}).get("dataset_reference"),
        },
        "evaluation": {
            "id_eval_id": manifest.get("artifacts", {}).get("id_eval_id"),
            "id_eval_root": manifest.get("artifacts", {}).get("id_eval_root"),
            "ood_eval_id": manifest.get("artifacts", {}).get("ood_eval_id"),
            "ood_eval_root": manifest.get("artifacts", {}).get("ood_eval_root"),
        },
        "rows": rows,
        "aggregates_by_supervised_collocation": _aggregate_rows(rows),
    }
    write_summary_json(summary_json, summary_payload)
    write_failures_json(failures_json, failures)
    return {
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
        "failures_json": str(failures_json),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--python-bin", default=sys.executable, help="Python executable.")
    parser.add_argument("--mode", default="screening", choices=["screening", "final"], help="Experiment mode.")
    parser.add_argument("--experiment-tag", default=None, help="Output/W&B tag suffix. Default: timestamp.")
    parser.add_argument("--output-root", default=None, help="Explicit output root.")
    parser.add_argument("--model-flag", default="SM4", help="Model flag.")

    dataset_group = parser.add_mutually_exclusive_group()
    dataset_group.add_argument("--reference-id", default=None, help=f"Reference dataset ID. Default: {DEFAULT_REFERENCE_ID}.")
    dataset_group.add_argument("--dataset-root", default=None, help="Explicit preprocessed dataset root.")
    parser.add_argument("--id-eval-id", default=None, help="Optional ID evaluation dataset ID.")
    parser.add_argument("--ood-eval-id", default=None, help="Optional OOD evaluation dataset ID.")
    parser.add_argument("--id-eval-root", default=None, help="Optional explicit ID evaluation preprocessed dataset root.")
    parser.add_argument("--ood-eval-root", default=None, help="Optional explicit OOD evaluation preprocessed dataset root.")
    parser.add_argument("--no-ood-eval", action="store_true", help=f"Disable default OOD evaluation ({DEFAULT_OOD_EVAL_ID}).")

    parser.add_argument("--seed-labels", default=None, help="Comma-separated seed labels.")
    parser.add_argument("--seed", type=int, default=None, help="Deprecated single raw integer seed.")
    parser.add_argument(
        "--supervised-strategies",
        default=None,
        help=(
            f"Comma-separated supervised strategies: {', '.join(SUPERVISED_STRATEGIES)}. "
            "Defaults by mode."
        ),
    )
    parser.add_argument("--collocation-strategies", default=None, help=f"Comma-separated collocation strategies: {', '.join(COLLOCATION_STRATEGIES)}.")

    parser.add_argument("--initial-trajectories", type=int, default=None, help="Initial active supervised trajectory count. Defaults by mode.")
    parser.add_argument("--add-trajectories", type=int, default=None, help="Supervised trajectories added per refresh. Defaults by mode.")
    parser.add_argument("--final-trajectories", type=int, default=None, help="Final active supervised trajectory count. Defaults by mode.")
    parser.add_argument("--initial-collocation-points", type=int, default=4096, help="Initial/static collocation points.")
    parser.add_argument("--add-collocation-points", type=int, default=2048, help="Collocation points appended per refresh.")
    parser.add_argument("--final-collocation-points", type=int, default=32768, help="Final active collocation points.")
    parser.add_argument(
        "--candidate-collocation-points",
        type=int,
        default=None,
        help="RAR-D candidate pool size. Default: max(4 * final collocation points, 256).",
    )
    parser.add_argument("--refresh-period-epochs", type=int, default=None, help="Shared supervised/collocation refresh cadence.")

    parser.add_argument("--device", default="cuda", help="PINN device.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="PINN hidden width.")
    parser.add_argument("--hidden-layers", type=int, default=4, help="PINN hidden depth.")
    parser.add_argument("--activation", default="tanh", help="PINN activation.")
    parser.add_argument("--dtype", default="float64", help="PINN dtype.")
    parser.add_argument("--epochs", type=int, default=None, help="Adam epochs.")
    parser.add_argument("--batch-size", type=int, default=4096, help="Adam mini-batch size.")
    parser.add_argument("--adam-lr", type=float, default=DEFAULT_ADAM_LR, help="Adam learning rate.")
    parser.add_argument("--candidate-batch-size", type=int, default=4096, help="MAE candidate scoring batch size.")

    parser.add_argument("--collocation-sampler", default="random", choices=["random", "lhs", "sobol"], help="Generated collocation sampler.")
    parser.add_argument("--score-norm", default="l2", choices=["l1", "l2", "linf"], help="Residual score norm.")
    parser.add_argument("--rar-d-k", type=float, default=2.0, help="RAR-D exponent k.")
    parser.add_argument("--rar-d-c", type=float, default=0.0, help="RAR-D additive floor c.")

    parser.add_argument("--wandb-project", default=None, help="Override W&B project.")
    parser.add_argument("--wandb-entity", default=None, help="Optional W&B entity.")
    parser.add_argument("--log-every-epoch", type=int, default=10, help="Metric/W&B logging cadence.")
    parser.add_argument("--tag", action="append", default=[], help="Extra W&B tag. Repeatable.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    return parser


def _validate_budgets(*, args: argparse.Namespace, total_trajectories: int, total_preprocessed_col_rows: int) -> None:
    if args.initial_trajectories <= 0 or args.add_trajectories <= 0 or args.final_trajectories <= 0:
        raise ValueError("Supervised trajectory budgets must be positive.")
    if args.initial_trajectories > args.final_trajectories:
        raise ValueError("--initial-trajectories must be <= --final-trajectories.")
    if args.final_trajectories > total_trajectories:
        raise ValueError(
            f"--final-trajectories={args.final_trajectories} exceeds available training trajectories={total_trajectories}."
        )
    if args.initial_collocation_points <= 0 or args.add_collocation_points <= 0 or args.final_collocation_points <= 0:
        raise ValueError("Collocation budgets must be positive.")
    if args.initial_collocation_points > args.final_collocation_points:
        raise ValueError("--initial-collocation-points must be <= --final-collocation-points.")
    if args.candidate_collocation_points is not None and args.candidate_collocation_points <= 0:
        raise ValueError("--candidate-collocation-points must be positive when provided.")
    if total_preprocessed_col_rows > 0 and args.initial_collocation_points > total_preprocessed_col_rows:
        raise ValueError(
            f"--initial-collocation-points={args.initial_collocation_points} exceeds "
            f"preprocessed collocation rows={total_preprocessed_col_rows}."
        )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    mode = str(args.mode)

    try:
        seed_pairs = _resolve_seed_pairs(args, mode=mode)
        supervised_strategies = _parse_named_list(
            args.supervised_strategies,
            defaults=_default_supervised_strategies_for_mode(mode),
            allowed=SUPERVISED_STRATEGIES,
            flag="--supervised-strategies",
        )
        collocation_strategies = _parse_named_list(
            args.collocation_strategies,
            defaults=DEFAULT_COLLOCATION_STRATEGIES,
            allowed=COLLOCATION_STRATEGIES,
            flag="--collocation-strategies",
        )
        eval_inputs = _resolve_eval_inputs(args)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from None

    default_initial_trajectories, default_add_trajectories, default_final_trajectories = _default_supervised_budgets_for_mode(mode)
    initial_trajectories = (
        int(args.initial_trajectories)
        if args.initial_trajectories is not None
        else int(default_initial_trajectories)
    )
    add_trajectories = (
        int(args.add_trajectories)
        if args.add_trajectories is not None
        else int(default_add_trajectories)
    )
    final_trajectories = (
        int(args.final_trajectories)
        if args.final_trajectories is not None
        else int(default_final_trajectories)
    )
    args.initial_trajectories = initial_trajectories
    args.add_trajectories = add_trajectories
    args.final_trajectories = final_trajectories

    epochs = int(args.epochs) if args.epochs is not None else (SCREENING_DEFAULT_EPOCHS if mode == "screening" else FINAL_DEFAULT_EPOCHS)
    if epochs <= 0:
        raise SystemExit("--epochs must be positive.")
    refresh_period_epochs = int(
        args.refresh_period_epochs
        if args.refresh_period_epochs is not None
        else (SCREENING_DEFAULT_CADENCE if mode == "screening" else FINAL_DEFAULT_CADENCE)
    )
    if refresh_period_epochs <= 0:
        raise SystemExit("--refresh-period-epochs must be positive.")

    wandb_project = args.wandb_project or default_project_for_mode(
        mode=mode,
        screening_project="thesis-adaptive-augmentation-TEST",
        final_project="thesis-adaptive-augmentation",
    )
    stamp = args.experiment_tag or tag_stamp()
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else (REPO_ROOT / "outputs" / "pinn" / "adaptive_augmentation" / stamp).resolve()
    )
    output_root.mkdir(parents=True, exist_ok=True)

    experiment_id = f"adaptive_augmentation_{mode}_{stamp}"
    wandb_group = group_name("adaptive_augmentation", args.model_flag.lower(), mode, stamp)
    manifest_path = output_root / "run_manifest.json"
    manifest = init_experiment_manifest(
        run_root=str(output_root),
        experiment={
            "id": experiment_id,
            "tag": stamp,
            "mode": mode,
            "model_flag": args.model_flag,
            "supervised_strategies": supervised_strategies,
            "collocation_strategies": collocation_strategies,
            "refresh_period_epochs": refresh_period_epochs,
            "initial_trajectories": initial_trajectories,
            "add_trajectories": add_trajectories,
            "final_trajectories": final_trajectories,
        },
    )
    manifest["artifacts"]["wandb_project"] = wandb_project
    manifest["artifacts"]["wandb_group"] = wandb_group
    manifest["artifacts"]["id_eval_id"] = eval_inputs["id_eval_id"]
    manifest["artifacts"]["id_eval_root"] = eval_inputs["id_eval_root"]
    manifest["artifacts"]["ood_eval_id"] = eval_inputs["ood_eval_id"]
    manifest["artifacts"]["ood_eval_root"] = eval_inputs["ood_eval_root"]
    save_manifest(str(manifest_path), manifest)

    try:
        dataset_root, dataset_reference_id = _resolve_dataset(args, manifest=manifest, manifest_path=manifest_path)
        total_trajectories = _count_training_trajectories(dataset_root)
        total_preprocessed_col_rows = _count_preprocessed_collocation_rows(dataset_root)
        _validate_budgets(args=args, total_trajectories=total_trajectories, total_preprocessed_col_rows=total_preprocessed_col_rows)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from None

    manifest["artifacts"]["total_training_trajectories"] = total_trajectories
    manifest["artifacts"]["total_preprocessed_collocation_points"] = total_preprocessed_col_rows
    manifest["artifacts"]["epochs"] = epochs
    manifest["artifacts"]["batch_size"] = int(args.batch_size)
    manifest["artifacts"]["adam_lr"] = float(args.adam_lr)
    manifest["artifacts"]["candidate_collocation_points"] = (
        int(args.candidate_collocation_points) if args.candidate_collocation_points is not None else None
    )
    manifest["artifacts"]["seed_labels"] = [label for label, _ in seed_pairs]
    manifest["artifacts"]["seed_values"] = [value for _, value in seed_pairs]
    save_manifest(str(manifest_path), manifest)

    print(f"[adaptive-augmentation] repo_root={REPO_ROOT}")
    print(f"[adaptive-augmentation] mode={mode}")
    print(f"[adaptive-augmentation] dataset_root={dataset_root}")
    print(f"[adaptive-augmentation] total_training_trajectories={total_trajectories}")
    print(f"[adaptive-augmentation] total_preprocessed_collocation_points={total_preprocessed_col_rows}")
    print(f"[adaptive-augmentation] supervised_strategies={supervised_strategies}")
    print(f"[adaptive-augmentation] collocation_strategies={collocation_strategies}")
    print(
        "[adaptive-augmentation] supervised_budget="
        f"{initial_trajectories}->{final_trajectories} add={add_trajectories}"
    )
    print(f"[adaptive-augmentation] seed_labels={[label for label, _ in seed_pairs]}")

    tags_base = wandb_tags_list(
        "adaptive_augmentation",
        mode,
        args.model_flag.lower(),
        "reference" if dataset_reference_id else None,
        dataset_reference_id,
        args.tag,
    )
    run_specs = _build_run_specs(
        supervised_strategies=supervised_strategies,
        collocation_strategies=collocation_strategies,
        seed_pairs=seed_pairs,
        refresh_period_epochs=refresh_period_epochs,
        supervised_initial_trajectories=initial_trajectories,
        supervised_add_trajectories=add_trajectories,
        supervised_final_trajectories=final_trajectories,
        collocation_initial_points=int(args.initial_collocation_points),
        collocation_add_points=int(args.add_collocation_points),
        collocation_final_points=int(args.final_collocation_points),
        collocation_candidate_points=(
            int(args.candidate_collocation_points) if args.candidate_collocation_points is not None else None
        ),
    )

    for run_spec in run_specs:
        run_name = run_spec.run_name
        run_dir = output_root / "runs" / run_name
        wandb_tags = [*tags_base, *run_spec.wandb_tags()]
        command = _build_pinn_command(
            python_bin=args.python_bin,
            model_flag=args.model_flag,
            dataset_root=dataset_root,
            run_dir=run_dir,
            run_name=run_name,
            run_spec=run_spec,
            wandb_project=wandb_project,
            wandb_group=wandb_group,
            wandb_entity=args.wandb_entity,
            wandb_tags=wandb_tags,
            device=args.device,
            hidden_dim=args.hidden_dim,
            hidden_layers=args.hidden_layers,
            activation=args.activation,
            dtype_name=args.dtype,
            epochs=epochs,
            batch_size=args.batch_size,
            adam_lr=args.adam_lr,
            candidate_batch_size=args.candidate_batch_size,
            collocation_sampler=args.collocation_sampler,
            score_norm=args.score_norm,
            rar_d_k=args.rar_d_k,
            rar_d_c=args.rar_d_c,
            log_every_epoch=args.log_every_epoch,
            id_eval_root=eval_inputs["id_eval_root"],
            ood_eval_root=eval_inputs["ood_eval_root"],
        )
        log_path = output_root / "logs" / "runs" / f"{run_name}.log"
        run_metadata = {
            "supervised_strategy": run_spec.supervised_strategy,
            "supervised_acquisition_strategy": run_spec.supervised_acquisition_strategy,
            "supervised_initial_trajectories": run_spec.supervised_initial_trajectories,
            "supervised_add_trajectories": run_spec.supervised_add_trajectories,
            "supervised_max_trajectories": run_spec.supervised_max_trajectories,
            "collocation_strategy": run_spec.collocation_strategy,
            "collocation_initial_points": run_spec.collocation_initial_points,
            "collocation_add_points": run_spec.collocation_add_points,
            "collocation_final_points": run_spec.collocation_final_points,
            "collocation_candidate_points": run_spec.collocation_candidate_points,
            "refresh_period_epochs": run_spec.refresh_period_epochs,
            "seed_label": run_spec.seed_label,
            "seed_value": run_spec.seed_value,
            "dataset_reference_id": dataset_reference_id,
            "dataset_root": str(dataset_root),
            "id_eval_id": eval_inputs["id_eval_id"],
            "id_eval_root": eval_inputs["id_eval_root"],
            "ood_eval_id": eval_inputs["ood_eval_id"],
            "ood_eval_root": eval_inputs["ood_eval_root"],
        }
        upsert_run_status(
            manifest,
            run_name=run_name,
            status="running" if not args.dry_run else "dry_run",
            run_dir=str(run_dir),
            run_metadata=run_metadata,
            command=command,
            log_file=str(log_path),
            started_at_utc=utc_now_iso(),
        )
        save_manifest(str(manifest_path), manifest)
        result = run_logged_command(
            label="adaptive-augmentation",
            command=command,
            log_path=log_path,
            dry_run=bool(args.dry_run),
            cwd=REPO_ROOT,
        )
        upsert_run_status(
            manifest,
            run_name=run_name,
            status=str(result["status"]),
            run_dir=str(run_dir),
            run_metadata=run_metadata,
            completed_at_utc=str(result["completed_at_utc"]),
            return_code=int(result["return_code"]),
            error=None if int(result["return_code"]) == 0 else f"Run '{run_name}' failed. See {log_path}",
            extra_fields={"elapsed_seconds": float(result["elapsed_seconds"])},
        )
        save_manifest(str(manifest_path), manifest)
        if int(result["return_code"]) != 0:
            if not args.dry_run:
                summary_artifacts = _write_summary_artifacts(output_root=output_root, manifest=manifest)
                manifest["artifacts"].update(summary_artifacts)
                save_manifest(str(manifest_path), manifest)
            raise SystemExit(int(result["return_code"]))

    summary_artifacts = _write_summary_artifacts(output_root=output_root, manifest=manifest)
    manifest["artifacts"].update(summary_artifacts)
    save_manifest(str(manifest_path), manifest)
    print(f"[adaptive-augmentation] summary_csv={summary_artifacts['summary_csv']}")
    print(f"[adaptive-augmentation] summary_json={summary_artifacts['summary_json']}")
    print(f"[adaptive-augmentation] failures_json={summary_artifacts['failures_json']}")


if __name__ == "__main__":
    main()
