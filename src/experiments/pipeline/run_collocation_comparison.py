"""Thesis Experiment 4: Collocation Point Sampling comparison for single-stage PINN training.

Evaluate how collocation strategy and collocation density affect PINN accuracy, residual
behaviour, convergence, and computational cost.

Fixed:  single-stage PINN, reference dataset, architecture, Adam optimizer, static weights.
Vary:   collocation strategy × collocation density × seed.

Run matrix:  strategy × density × seed.
Run name:    <strategy>_<density_label>_<seed_label>  (e.g. rad_d25_s01).
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.experiments.pipeline.helpers.evaluation import resolve_eval_inputs as _resolve_shared_eval_inputs
from src.experiments.pipeline.helpers.launch_utils import (
    build_dataset_pipeline_command,
    dataset_root_from_manifest,
    format_hydra_list,
    init_experiment_manifest,
    parse_csv_list,
    run_logged_command,
    run_logged_stage,
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
DEFAULT_REFERENCE_ID = "main_SM4_qbc_b512_ds01"
DEFAULT_OOD_EVAL_ID = "ood_SM4_wide_ic_b512_ds01"
DEFAULT_ADAM_LR = 3.0e-3

SCREENING_SEED_LABELS = ("s01",)
FINAL_SEED_LABELS = ("s01", "s02", "s03", "s04", "s05")
CADENCE_SEED_LABELS = ("s01",)
SCREENING_DEFAULT_EPOCHS = 100
FINAL_DEFAULT_EPOCHS = 20_000
CADENCE_DEFAULT_EPOCHS = 5_000
SCREENING_DEFAULT_REFRESH_PERIOD = 10
FINAL_DEFAULT_REFRESH_PERIOD = 500
CADENCE_DEFAULT_REFRESH_PERIODS = (100, 500, 1000, 2000)

# Density labels and their fraction of total preprocessed collocation rows.
DENSITY_LABELS: dict[str, float] = {
    "d10": 0.10,
    "d25": 0.25,
    "d50": 0.50,
    "d100": 1.00,
}
SCREENING_DEFAULT_DENSITIES = ("d10", "d25")
FINAL_DEFAULT_DENSITIES = ("d10", "d25", "d50", "d100")
CADENCE_DEFAULT_DENSITIES = ("d25",)

# Maps user-facing strategy names to the variant names consumed by _variant_overrides().
STRATEGY_TO_VARIANT: dict[str, str] = {
    "uniform_lhs": "static_preprocessed",
    "random_resampling": "random_r",
    "random_r": "random_r",
    "rad": "rad",
    "rar_d": "rar_d",
    "rar_g": "rar_g",
}
CORE_STRATEGIES = ("uniform_lhs", "random_resampling", "rad", "rar_d")
CADENCE_DEFAULT_STRATEGIES = ("random_resampling", "rad", "rar_d")

SUMMARY_FIELDNAMES = [
    "run_name",
    "strategy",
    "density_label",
    "density_value",
    "refresh_period_epochs",
    "total_preprocessed_collocation_points",
    "active_collocation_points",
    "candidate_points",
    "append_points",
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
    "num_collocation_rows",
    "epoch_wall_seconds",
    "cumulative_wall_seconds",
    "total_seconds",
    "training_seconds",
]


@dataclass(frozen=True)
class CollocationRunSpec:
    strategy: str
    variant: str
    density_label: str
    density_value: float
    refresh_period_epochs: int | None
    active_points: int
    candidate_points: int | None
    append_points: int | None
    total_preprocessed_col_rows: int
    seed_label: str
    seed_value: int

    @property
    def run_name(self) -> str:
        cadence_part = "" if self.refresh_period_epochs is None else f"_c{self.refresh_period_epochs}"
        return f"{self.strategy}_{self.density_label}{cadence_part}_{self.seed_label}"

    def wandb_tags(self) -> list[str]:
        return [self.strategy, self.density_label, self.seed_label]


# ---------------------------------------------------------------------------
# Strategy / density helpers
# ---------------------------------------------------------------------------

def _default_strategies_for_mode(mode: str) -> list[str]:
    if mode == "cadence":
        return list(CADENCE_DEFAULT_STRATEGIES)
    return list(CORE_STRATEGIES)


def _parse_strategies(raw: str | None, *, mode: str) -> list[str]:
    if raw:
        strategies = [s.strip().lower().replace("-", "_") for s in parse_csv_list(raw) if s.strip()]
    else:
        strategies = _default_strategies_for_mode(mode)
    if not strategies:
        raise ValueError("Expected at least one strategy.")
    unsupported = [s for s in strategies if s not in STRATEGY_TO_VARIANT]
    if unsupported:
        raise ValueError(
            f"Unsupported strategy/strategies: {', '.join(unsupported)}. "
            f"Use one of: {', '.join(STRATEGY_TO_VARIANT)}."
        )
    if len(set(strategies)) != len(strategies):
        raise ValueError("--strategies must not contain duplicates.")
    return strategies


def _parse_densities(raw: str | None, *, mode: str) -> list[str]:
    if raw:
        items = [item.strip().lower() for item in parse_csv_list(raw) if item.strip()]
    else:
        if mode == "cadence":
            items = list(CADENCE_DEFAULT_DENSITIES)
        else:
            items = list(SCREENING_DEFAULT_DENSITIES if mode == "screening" else FINAL_DEFAULT_DENSITIES)
    if not items:
        raise ValueError("Expected at least one density.")
    resolved: list[str] = []
    for item in items:
        if item in DENSITY_LABELS:
            resolved.append(item)
        else:
            try:
                value = float(item)
            except ValueError:
                raise ValueError(
                    f"Unknown density '{item}'. Use a label ({', '.join(DENSITY_LABELS)}) "
                    "or a float fraction in (0, 1]."
                )
            if not (0.0 < value <= 1.0):
                raise ValueError(f"Density value {value} out of range (0, 1].")
            label = next((k for k, v in DENSITY_LABELS.items() if abs(v - value) < 1e-9), None)
            if label is None:
                label = f"d{int(round(value * 100)):03d}"
                DENSITY_LABELS[label] = value  # register ad-hoc label
            resolved.append(label)
    if len(set(resolved)) != len(resolved):
        raise ValueError("--densities must not contain duplicates.")
    return resolved


def _parse_cadences(raw: str | None, *, default: int | tuple[int, ...]) -> list[int]:
    if raw:
        items = [item.strip() for item in parse_csv_list(raw) if item.strip()]
        if not items:
            raise ValueError("Expected at least one cadence.")
        cadences = [int(item) for item in items]
    elif isinstance(default, tuple):
        cadences = list(default)
    else:
        cadences = [int(default)]
    if any(cadence <= 0 for cadence in cadences):
        raise ValueError("Cadence values must be positive integers.")
    if len(set(cadences)) != len(cadences):
        raise ValueError("--cadences must not contain duplicates.")
    return cadences


def _resolve_eval_inputs(args: argparse.Namespace) -> dict[str, str | None]:
    return _resolve_shared_eval_inputs(args, default_ood_eval_id=DEFAULT_OOD_EVAL_ID)


# ---------------------------------------------------------------------------
# Collocation pool counting
# ---------------------------------------------------------------------------

def _count_preprocessed_collocation_rows(dataset_root: Path) -> int:
    """Return total collocation rows in the train split by reading HDF5 headers."""
    import h5py

    train_dir = dataset_root / "train"
    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    total = 0
    found = False
    for fn in sorted(os.listdir(str(train_dir))):
        if not fn.endswith(".h5"):
            continue
        if "_data_col" not in fn:
            continue
        found = True
        with h5py.File(str(train_dir / fn), "r") as h5f:
            if "x_train" in h5f:
                total += int(h5f["x_train"].shape[0])
    if not found:
        raise FileNotFoundError(
            f"No collocation HDF5 files found in {train_dir}. "
            "Expected files matching *_data_col*.h5."
        )
    if total == 0:
        raise ValueError(f"Collocation pool is empty in {train_dir}.")
    return total


def _active_points_for_density(*, total: int, density_label: str) -> int:
    density_value = DENSITY_LABELS[density_label]
    if abs(density_value - 1.0) < 1e-9:
        return total
    return max(1, int(total * density_value))


# ---------------------------------------------------------------------------
# Run-spec builder
# ---------------------------------------------------------------------------

def _build_collocation_run_specs(
    *,
    strategies: list[str],
    density_labels: list[str],
    cadences: list[int],
    seed_pairs: list[tuple[str, int]],
    total_preprocessed_col_rows: int,
) -> list[CollocationRunSpec]:
    specs: list[CollocationRunSpec] = []
    for strategy in strategies:
        variant = STRATEGY_TO_VARIANT[strategy]
        strategy_cadences: list[int | None]
        if variant == "static_preprocessed" and len(cadences) > 1:
            # Static preprocessed collocation never refreshes; avoid duplicate
            # static baseline runs during cadence calibration.
            strategy_cadences = [None]
        else:
            strategy_cadences = list(cadences)
        for density_label in density_labels:
            active_points = _active_points_for_density(
                total=total_preprocessed_col_rows,
                density_label=density_label,
            )
            for refresh_period_epochs in strategy_cadences:
                candidate_points = None if variant == "static_preprocessed" else max(active_points * 4, 256)
                append_points = None if variant not in {"rar_d", "rar_g"} else max(16, active_points // 20)
                for seed_label, seed_value in seed_pairs:
                    specs.append(CollocationRunSpec(
                        strategy=strategy,
                        variant=variant,
                        density_label=density_label,
                        density_value=DENSITY_LABELS[density_label],
                        refresh_period_epochs=refresh_period_epochs,
                        active_points=active_points,
                        candidate_points=candidate_points,
                        append_points=append_points,
                        total_preprocessed_col_rows=total_preprocessed_col_rows,
                        seed_label=seed_label,
                        seed_value=seed_value,
                    ))
    return specs


# ---------------------------------------------------------------------------
# Collocation config builder (per run spec)
# ---------------------------------------------------------------------------

def _collocation_cfg_for_run(
    *,
    run_spec: CollocationRunSpec,
    sampler: str,
    score_norm: str,
    rad_k: float,
    rad_c: float,
    rar_d_k: float,
    rar_d_c: float,
) -> dict[str, Any]:
    active = run_spec.active_points
    candidate = run_spec.candidate_points if run_spec.candidate_points is not None else max(active * 4, 256)
    append = run_spec.append_points if run_spec.append_points is not None else max(16, active // 20)
    return {
        "active_points": active,
        "candidate_points": candidate,
        "initial_points": None,  # let factory derive default (half of active for rar_d/rar_g)
        "append_points": append,
        "refresh_period_epochs": 0 if run_spec.refresh_period_epochs is None else run_spec.refresh_period_epochs,
        "refresh_mode": "epoch_periodic",
        "refresh_on_phase_start": [],
        "refresh_on_phase_end": [],
        "sampler": sampler,
        "score_norm": score_norm,
        "rad_k": rad_k,
        "rad_c": rad_c,
        "rar_d_k": rar_d_k,
        "rar_d_c": rar_d_c,
    }


# ---------------------------------------------------------------------------
# Variant overrides (retained from original pipeline)
# ---------------------------------------------------------------------------

def _variant_overrides(*, variant: str, collocation_cfg: dict[str, Any]) -> list[str]:
    common = [
        "pinn.collocation.sampling.enabled=false",
        "pinn.evaluation.frequency=25",
        f"pinn.collocation.active_points={int(collocation_cfg['active_points'])}",
        f"pinn.collocation.candidate_points={int(collocation_cfg['candidate_points'])}",
        f"pinn.collocation.append_points={int(collocation_cfg['append_points'])}",
        f"pinn.collocation.refresh_period_epochs={int(collocation_cfg['refresh_period_epochs'])}",
        f"pinn.collocation.refresh.mode={str(collocation_cfg.get('refresh_mode', 'epoch_periodic'))}",
        f"pinn.collocation.refresh.on_phase_start={format_hydra_list([str(x) for x in collocation_cfg.get('refresh_on_phase_start', [])])}",
        f"pinn.collocation.refresh.on_phase_end={format_hydra_list([str(x) for x in collocation_cfg.get('refresh_on_phase_end', [])])}",
        f"pinn.collocation.sampler={str(collocation_cfg['sampler'])}",
        f"pinn.collocation.score_norm={str(collocation_cfg['score_norm'])}",
    ]
    if collocation_cfg.get("initial_points") not in (None, "null"):
        common.append(f"pinn.collocation.initial_points={int(collocation_cfg['initial_points'])}")

    if variant == "static_preprocessed":
        return [*common, "pinn.collocation.mode=preprocessed", "pinn.collocation.strategy=static"]
    if variant == "static_generated":
        return [*common, "pinn.collocation.mode=generated", "pinn.collocation.strategy=static"]
    if variant == "random_r":
        return [*common, "pinn.collocation.mode=generated", "pinn.collocation.strategy=random_r"]
    if variant == "rad":
        return [
            *common,
            "pinn.collocation.mode=generated",
            "pinn.collocation.strategy=rad",
            f"pinn.collocation.rad.k={float(collocation_cfg['rad_k'])}",
            f"pinn.collocation.rad.c={float(collocation_cfg['rad_c'])}",
        ]
    if variant == "rar_d":
        return [
            *common,
            "pinn.collocation.mode=generated",
            "pinn.collocation.strategy=rar_d",
            f"pinn.collocation.rad.k={float(collocation_cfg['rar_d_k'])}",
            f"pinn.collocation.rad.c={float(collocation_cfg['rar_d_c'])}",
        ]
    if variant == "rar_g":
        return [*common, "pinn.collocation.mode=generated", "pinn.collocation.strategy=rar_g"]
    raise ValueError(f"Unsupported variant: {variant}")


def _adam_phase_override(*, epochs: int, lr: float, batch_size: int) -> str:
    return (
        "pinn.optimizer_phases=["
        "{name:adam,optimizer:Adam,"
        f"lr:{float(lr)},epochs:{int(epochs)},batch_size:{int(batch_size)},"
        "shuffle:true,full_batch:false,allow_sampling:true,"
        "optimizer_kwargs:{eps:1.0e-6},scheduler:null,line_search:null,convergence:null}"
        "]"
    )


# ---------------------------------------------------------------------------
# PINN command builder
# ---------------------------------------------------------------------------

def _build_pinn_command(
    *,
    python_bin: str,
    model_flag: str,
    dataset_root: Path,
    run_dir: Path,
    wandb_project: str,
    wandb_group: str,
    wandb_entity: str | None,
    wandb_tags: list[str],
    run_spec: CollocationRunSpec,
    collocation_cfg: dict[str, Any],
    device: str,
    hidden_dim: int,
    hidden_layers: int,
    activation: str,
    dtype_name: str,
    epochs: int,
    batch_size: int,
    adam_lr: float,
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
        "pinn.gradient_telemetry.enabled=false",
        "pinn.checkpointing.enabled=true",
        "pinn.checkpointing.save_best=true",
        "pinn.checkpointing.save_last=false",
        "pinn.checkpointing.save_init=false",
        "pinn.checkpointing.epoch_fractions=[]",
        f"logging.log_every_epoch={int(log_every_epoch)}",
        "wandb.use=true",
        f"wandb.project={wandb_project}",
        f"wandb.group={wandb_group}",
        f"wandb.name={run_spec.run_name}",
        f"wandb.tags={format_hydra_list(wandb_tags)}",
        *_variant_overrides(variant=run_spec.variant, collocation_cfg=collocation_cfg),
        _adam_phase_override(epochs=epochs, lr=adam_lr, batch_size=batch_size),
    ]
    if wandb_entity:
        command.append(f"wandb.entity={wandb_entity}")
    if id_eval_root:
        command.append(f"evaluation.id.root={id_eval_root}")
    if ood_eval_root:
        command.append(f"evaluation.ood.root={ood_eval_root}")
    return command


# ---------------------------------------------------------------------------
def _summary_row_for_run(run: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(str(run.get("run_dir", ""))) if run.get("run_dir") else None
    metrics = _read_json_if_exists(run_dir / "metrics.json") if run_dir else None
    timings = _read_json_if_exists(run_dir / "timings.json") if run_dir else None
    final_train_losses = dict(metrics.get("final_train_losses", {}) or {}) if metrics else {}
    final_epoch = dict(metrics.get("final_epoch", {}) or {}) if metrics else {}
    standard_metrics = _extract_standard_pinn_summary_fields(metrics=metrics, run_dir=run_dir)
    return {
        "run_name": run.get("run_name"),
        "strategy": run.get("strategy"),
        "density_label": run.get("density_label"),
        "density_value": run.get("density_value"),
        "refresh_period_epochs": run.get("refresh_period_epochs"),
        "total_preprocessed_collocation_points": run.get("total_preprocessed_col_rows"),
        "active_collocation_points": run.get("active_points"),
        "candidate_points": run.get("candidate_points"),
        "append_points": run.get("append_points"),
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
        "num_collocation_rows": _int_or_none(final_epoch.get("num_collocation_rows")),
        "epoch_wall_seconds": _float_or_none(final_epoch.get("epoch_wall_seconds")),
        "cumulative_wall_seconds": _float_or_none(final_epoch.get("cumulative_wall_seconds")),
        "total_seconds": _float_or_none(timings.get("total_seconds") if timings else None),
        "training_seconds": _float_or_none(timings.get("training_seconds") if timings else None),
    }


def _aggregate_by_strategy_density_cadence(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    keys: set[tuple[str, str, str | None]] = set()
    for row in rows:
        strategy = row.get("strategy")
        density = row.get("density_label")
        if strategy and density:
            cadence = None if row.get("refresh_period_epochs") in (None, "", "None") else str(row.get("refresh_period_epochs"))
            keys.add((str(strategy), str(density), cadence))
    out: dict[str, dict[str, Any]] = {}
    for strategy, density, cadence in sorted(keys, key=lambda x: (x[0], x[1], "" if x[2] is None else x[2])):
        group_key = f"{strategy}/{density}" if cadence is None else f"{strategy}/{density}/c{cadence}"
        members = [
            row for row in rows
            if row.get("strategy") == strategy and row.get("density_label") == density
            and (None if row.get("refresh_period_epochs") in (None, "", "None") else str(row.get("refresh_period_epochs"))) == cadence
        ]
        successful = [r for r in members if r.get("status") == "completed"]
        failed = [r for r in members if r.get("status") != "completed"]
        rmse_values = [float(r["final_test_rmse"]) for r in successful if r.get("final_test_rmse") is not None]
        training_seconds = [float(r["training_seconds"]) for r in successful if r.get("training_seconds") is not None]
        rmse_stats = _mean_std(rmse_values)
        timing_stats = _mean_std(training_seconds)
        out[group_key] = {
            "strategy": strategy,
            "density_label": density,
            "refresh_period_epochs": None if cadence is None else int(cadence),
            "success_count": len(successful),
            "failure_count": len(failed),
            "final_test_rmse_mean": rmse_stats["mean"],
            "final_test_rmse_std": rmse_stats["std"],
            "training_seconds_mean": timing_stats["mean"],
            "training_seconds_std": timing_stats["std"],
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
            "strategy": row.get("strategy"),
            "density_label": row.get("density_label"),
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
        "strategies": manifest.get("artifacts", {}).get("strategies", []),
        "density_labels": manifest.get("artifacts", {}).get("density_labels", []),
        "cadences": manifest.get("artifacts", {}).get("cadences", []),
        "seed_labels": manifest.get("artifacts", {}).get("seed_labels", []),
        "seed_values": manifest.get("artifacts", {}).get("seed_values", []),
        "total_preprocessed_collocation_points": manifest.get("artifacts", {}).get("total_preprocessed_collocation_points"),
        "rows": rows,
        "aggregates_by_strategy_density_cadence": _aggregate_by_strategy_density_cadence(rows),
    }
    write_summary_json(summary_json, summary_payload)
    write_failures_json(failures_json, failures)

    return {
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
        "failures_json": str(failures_json),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--python-bin", default=sys.executable, help="Python executable.")
    parser.add_argument(
        "--mode", default="screening", choices=["screening", "final", "cadence"],
        help="Experiment mode. Controls default epochs, seeds, densities, and W&B project.",
    )
    parser.add_argument("--experiment-tag", default=None, help="Output/W&B tag suffix. Default: timestamp.")
    parser.add_argument("--output-root", default=None, help="Explicit output root. Default: outputs/pinn/collocation_comparison/<tag>.")
    parser.add_argument("--model-flag", default="SM4", help="Model flag.")

    # Dataset
    dataset_group = parser.add_mutually_exclusive_group()
    dataset_group.add_argument(
        "--reference-id", default=None,
        help=f"Reference dataset ID. Default: {DEFAULT_REFERENCE_ID}.",
    )
    dataset_group.add_argument(
        "--dataset-root", default=None,
        help="Explicit preprocessed dataset root. Mutually exclusive with --reference-id.",
    )
    parser.add_argument(
        "--allow-dataset-generation", action="store_true",
        help="Fallback: generate a new dataset. Mutually exclusive with --reference-id and --dataset-root.",
    )
    parser.add_argument("--preset", default=None, help="Dataset pipeline preset (--allow-dataset-generation only).")
    parser.add_argument("--budget", default="b512", help="Dataset budget (--allow-dataset-generation only).")
    parser.add_argument("--dataset-seed", default="ds01", help="Dataset seed label (--allow-dataset-generation only).")
    parser.add_argument("--id-eval-id", default=None, help="Optional ID evaluation dataset ID from data/evaluation/index.json.")
    parser.add_argument("--ood-eval-id", default=None, help="Optional OOD evaluation dataset ID from data/evaluation/index.json.")
    parser.add_argument("--id-eval-root", default=None, help="Optional explicit ID evaluation preprocessed dataset root.")
    parser.add_argument("--ood-eval-root", default=None, help="Optional explicit OOD evaluation preprocessed dataset root.")
    parser.add_argument("--no-ood-eval", action="store_true", help=f"Disable default OOD evaluation ({DEFAULT_OOD_EVAL_ID}).")

    # Seeds
    parser.add_argument(
        "--seed-labels", default=None,
        help="Comma-separated seed labels from src/config/registry/seeds.yaml. Defaults by --mode.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Deprecated single raw integer seed. Prefer --seed-labels.",
    )

    # Strategies and densities
    parser.add_argument(
        "--strategies", default=None,
        help=(
            "Comma-separated collocation strategies. Defaults by --mode. "
            f"Main choices: {', '.join(CORE_STRATEGIES)}. "
            "Optional: rar_g. Alias accepted: random_r."
        ),
    )
    parser.add_argument(
        "--densities", default=None,
        help=(
            f"Comma-separated density labels or fractions. "
            f"Labels: {', '.join(DENSITY_LABELS)}. "
            "Cadence default: d25. Screening default: d10,d25. Final default: d10,d25,d50,d100."
        ),
    )

    # Architecture / training
    parser.add_argument("--device", default="cuda", help="PINN device.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="PINN hidden width.")
    parser.add_argument("--hidden-layers", type=int, default=4, help="PINN hidden depth.")
    parser.add_argument("--activation", default="tanh", help="PINN activation.")
    parser.add_argument("--dtype", default="float64", help="PINN dtype.")
    parser.add_argument(
        "--epochs", type=int, default=None,
        help=f"Adam epochs. Defaults: cadence={CADENCE_DEFAULT_EPOCHS}, screening={SCREENING_DEFAULT_EPOCHS}, final={FINAL_DEFAULT_EPOCHS}.",
    )
    parser.add_argument("--batch-size", type=int, default=1024, help="Adam mini-batch size.")
    parser.add_argument("--adam-lr", type=float, default=DEFAULT_ADAM_LR, help="Adam learning rate.")

    # Collocation parameters
    parser.add_argument(
        "--refresh-period-epochs", type=int, default=None,
        help=f"Collocation refresh cadence. Defaults: screening={SCREENING_DEFAULT_REFRESH_PERIOD}, final={FINAL_DEFAULT_REFRESH_PERIOD}.",
    )
    parser.add_argument(
        "--cadences", default=None,
        help="Comma-separated refresh cadences for cadence calibration, e.g. 100,500,1000,2000.",
    )
    parser.add_argument("--sampler", default="random", choices=["random", "lhs", "sobol"], help="Collocation sampler for generated modes.")
    parser.add_argument("--score-norm", default="l2", choices=["l1", "l2", "linf"], help="Residual norm for adaptive scoring.")
    parser.add_argument("--rad-k", type=float, default=1.0, help="RAD exponent k.")
    parser.add_argument("--rad-c", type=float, default=1.0, help="RAD additive floor c.")
    parser.add_argument("--rar-d-k", type=float, default=2.0, help="RAR-D exponent k.")
    parser.add_argument("--rar-d-c", type=float, default=0.0, help="RAR-D additive floor c.")

    # W&B
    parser.add_argument("--wandb-project", default=None, help="Override W&B project. Defaults by --mode.")
    parser.add_argument("--wandb-entity", default=None, help="Optional W&B entity.")
    parser.add_argument("--log-every-epoch", type=int, default=10, help="Metric/W&B logging cadence.")
    parser.add_argument("--tag", action="append", default=[], help="Extra W&B tag. Repeatable.")

    # Dataset-generation overrides (only used with --allow-dataset-generation)
    parser.add_argument("--stage1-override", action="append", default=[], help="Extra stage-1 dataset-generation override.")
    parser.add_argument("--stage2-override", action="append", default=[], help="Extra stage-2 preprocess override.")

    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    return parser


def _resolve_seed_pairs(args: argparse.Namespace, *, mode: str) -> list[tuple[str, int]]:
    if args.seed_labels:
        return _seed_pairs_from_labels(_parse_label_list(str(args.seed_labels)))
    if args.seed is not None:
        return [(f"raw{int(args.seed)}", int(args.seed))]
    if mode == "cadence":
        default_labels = list(CADENCE_SEED_LABELS)
    else:
        default_labels = list(SCREENING_SEED_LABELS if mode == "screening" else FINAL_SEED_LABELS)
    return _seed_pairs_from_labels(default_labels)


def _resolve_dataset(
    args: argparse.Namespace,
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
    experiment_id: str,
) -> tuple[Path, str | None]:
    """Return (dataset_root, reference_id_or_None). Updates and saves manifest."""
    if args.allow_dataset_generation and (args.dataset_root or args.reference_id):
        raise ValueError("--allow-dataset-generation is mutually exclusive with --dataset-root and --reference-id.")

    if args.dataset_root:
        dataset_root = Path(args.dataset_root).resolve()
        manifest["artifacts"]["dataset_root"] = str(dataset_root)
        manifest["artifacts"]["dataset_source"] = "provided"
        save_manifest(str(manifest_path), manifest)
        return dataset_root, None

    if not args.allow_dataset_generation:
        reference_id = str(args.reference_id or DEFAULT_REFERENCE_ID)
        reference_entry = _resolve_reference_dataset(reference_id)
        dataset_root = Path(str(reference_entry["preprocessed_root"])).resolve()
        manifest["artifacts"]["dataset_root"] = str(dataset_root)
        manifest["artifacts"]["dataset_source"] = "reference"
        manifest["artifacts"]["dataset_reference_id"] = reference_id
        manifest["artifacts"]["dataset_reference"] = reference_entry
        save_manifest(str(manifest_path), manifest)
        return dataset_root, reference_id

    # --allow-dataset-generation fallback
    resolved_preset = args.preset or "default"
    dataset_command, dataset_pipeline_root = build_dataset_pipeline_command(
        python_bin=args.python_bin,
        method="qbc_deep_ensemble",
        experiment_id=experiment_id,
        preset=resolved_preset,
        budget=str(args.budget),
        dataset_seed=str(args.dataset_seed),
        model_flag=str(args.model_flag),
        run_root=Path(str(manifest["run_root"])),
        stage1_overrides=list(args.stage1_override),
        stage2_overrides=list(args.stage2_override),
    )
    rc = run_logged_stage(
        label="collocation-comparison",
        stage_name="dataset_pipeline",
        command=dataset_command,
        log_path=Path(str(manifest["run_root"])) / "logs" / "dataset_pipeline.log",
        manifest=manifest,
        manifest_path=manifest_path,
        dry_run=bool(args.dry_run),
        cwd=REPO_ROOT,
    )
    if rc != 0:
        raise SystemExit(rc)
    dataset_root = dataset_root_from_manifest(
        dataset_pipeline_root,
        dry_run=bool(args.dry_run),
        model_flag=str(args.model_flag),
    )
    manifest["artifacts"]["dataset_root"] = str(dataset_root)
    manifest["artifacts"]["dataset_source"] = "generated"
    manifest["artifacts"]["dataset_pipeline_root"] = str(dataset_pipeline_root)
    manifest["artifacts"]["dataset_manifest_path"] = str(dataset_pipeline_root / "dataset_manifest.json")
    save_manifest(str(manifest_path), manifest)
    return dataset_root, None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    mode = str(args.mode)

    try:
        seed_pairs = _resolve_seed_pairs(args, mode=mode)
        strategies = _parse_strategies(args.strategies, mode=mode)
        density_labels = _parse_densities(args.densities, mode=mode)
        if args.cadences and args.refresh_period_epochs is not None:
            raise ValueError("--cadences and --refresh-period-epochs are mutually exclusive.")
        eval_inputs = _resolve_eval_inputs(args)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from None

    if args.epochs is not None:
        epochs = int(args.epochs)
    elif mode == "cadence":
        epochs = CADENCE_DEFAULT_EPOCHS
    else:
        epochs = SCREENING_DEFAULT_EPOCHS if mode == "screening" else FINAL_DEFAULT_EPOCHS
    if epochs < 0:
        raise SystemExit("--epochs must be non-negative.")

    if args.refresh_period_epochs is not None:
        cadence_default: int | tuple[int, ...] = int(args.refresh_period_epochs)
    elif mode == "cadence":
        cadence_default = CADENCE_DEFAULT_REFRESH_PERIODS
    else:
        cadence_default = SCREENING_DEFAULT_REFRESH_PERIOD if mode == "screening" else FINAL_DEFAULT_REFRESH_PERIOD
    try:
        cadences = _parse_cadences(args.cadences, default=cadence_default)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None

    wandb_project = args.wandb_project or default_project_for_mode(
        mode=mode,
        screening_project="thesis-collocation-experiment-TEST",
        final_project="thesis-collocation-experiment",
        cadence_project="thesis-collocation-experiment-CADENCE",
    )

    stamp = args.experiment_tag or tag_stamp()
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else (REPO_ROOT / "outputs" / "pinn" / "collocation_comparison" / stamp).resolve()
    )
    output_root.mkdir(parents=True, exist_ok=True)

    experiment_id = f"collocation_comparison_{mode}_{stamp}"
    wandb_group = group_name("collocation_comparison", args.model_flag.lower(), mode, stamp)
    manifest_path = output_root / "run_manifest.json"
    manifest = init_experiment_manifest(
        run_root=str(output_root),
        experiment={
            "id": experiment_id,
            "tag": stamp,
            "mode": mode,
            "model_flag": args.model_flag,
            "strategies": strategies,
            "density_labels": density_labels,
            "cadences": cadences,
        },
    )
    manifest["artifacts"]["wandb_project"] = wandb_project
    manifest["artifacts"]["wandb_group"] = wandb_group
    manifest["artifacts"]["strategies"] = strategies
    manifest["artifacts"]["density_labels"] = density_labels
    manifest["artifacts"]["density_values"] = [DENSITY_LABELS[d] for d in density_labels]
    manifest["artifacts"]["cadences"] = cadences
    manifest["artifacts"]["seed_labels"] = [label for label, _ in seed_pairs]
    manifest["artifacts"]["seed_values"] = [value for _, value in seed_pairs]
    manifest["artifacts"]["epochs"] = epochs
    manifest["artifacts"]["batch_size"] = args.batch_size
    manifest["artifacts"]["adam_lr"] = args.adam_lr
    manifest["artifacts"]["id_eval_id"] = eval_inputs["id_eval_id"]
    manifest["artifacts"]["id_eval_root"] = eval_inputs["id_eval_root"]
    manifest["artifacts"]["ood_eval_id"] = eval_inputs["ood_eval_id"]
    manifest["artifacts"]["ood_eval_root"] = eval_inputs["ood_eval_root"]
    save_manifest(str(manifest_path), manifest)

    try:
        dataset_root, dataset_reference_id = _resolve_dataset(
            args, manifest=manifest, manifest_path=manifest_path, experiment_id=experiment_id
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from None

    try:
        total_preprocessed_col_rows = _count_preprocessed_collocation_rows(dataset_root)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from None

    manifest["artifacts"]["total_preprocessed_collocation_points"] = total_preprocessed_col_rows
    save_manifest(str(manifest_path), manifest)

    print(f"[collocation-comparison] repo_root={REPO_ROOT}")
    print(f"[collocation-comparison] mode={mode}")
    print(f"[collocation-comparison] reference_id={dataset_reference_id or '<not set>'}")
    print(f"[collocation-comparison] dataset_root={dataset_root}")
    print(f"[collocation-comparison] total_preprocessed_collocation_points={total_preprocessed_col_rows}")
    print(f"[collocation-comparison] strategies={strategies}")
    print(f"[collocation-comparison] density_labels={density_labels}")
    print(f"[collocation-comparison] cadences={cadences}")
    print(f"[collocation-comparison] seed_labels={[l for l, _ in seed_pairs]}")
    print(f"[collocation-comparison] epochs={epochs}")
    print(f"[collocation-comparison] wandb_project={wandb_project}")

    tags_base = wandb_tags_list(
        "collocation_comparison",
        mode,
        args.model_flag.lower(),
        "reference" if dataset_reference_id else None,
        dataset_reference_id,
        args.tag,
    )

    run_specs = _build_collocation_run_specs(
        strategies=strategies,
        density_labels=density_labels,
        cadences=cadences,
        seed_pairs=seed_pairs,
        total_preprocessed_col_rows=total_preprocessed_col_rows,
    )

    for run_spec in run_specs:
        run_name = run_spec.run_name
        run_dir = output_root / "runs" / run_name
        wandb_tags = [*tags_base, *run_spec.wandb_tags()]
        collocation_cfg = _collocation_cfg_for_run(
            run_spec=run_spec,
            sampler=args.sampler,
            score_norm=args.score_norm,
            rad_k=args.rad_k,
            rad_c=args.rad_c,
            rar_d_k=args.rar_d_k,
            rar_d_c=args.rar_d_c,
        )
        command = _build_pinn_command(
            python_bin=args.python_bin,
            model_flag=args.model_flag,
            dataset_root=dataset_root,
            run_dir=run_dir,
            wandb_project=wandb_project,
            wandb_group=wandb_group,
            wandb_entity=args.wandb_entity,
            wandb_tags=wandb_tags,
            run_spec=run_spec,
            collocation_cfg=collocation_cfg,
            device=args.device,
            hidden_dim=args.hidden_dim,
            hidden_layers=args.hidden_layers,
            activation=args.activation,
            dtype_name=args.dtype,
            epochs=epochs,
            batch_size=args.batch_size,
            adam_lr=args.adam_lr,
            log_every_epoch=args.log_every_epoch,
            id_eval_root=eval_inputs["id_eval_root"],
            ood_eval_root=eval_inputs["ood_eval_root"],
        )
        log_path = output_root / "logs" / "runs" / f"{run_name}.log"
        run_metadata = {
            "strategy": run_spec.strategy,
            "variant": run_spec.variant,
            "density_label": run_spec.density_label,
            "density_value": run_spec.density_value,
            "refresh_period_epochs": run_spec.refresh_period_epochs,
            "active_points": run_spec.active_points,
            "candidate_points": run_spec.candidate_points,
            "append_points": run_spec.append_points,
            "total_preprocessed_col_rows": run_spec.total_preprocessed_col_rows,
            "seed_label": run_spec.seed_label,
            "seed_value": run_spec.seed_value,
            "epochs": epochs,
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
            label="collocation-comparison",
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
                print(f"[collocation-comparison] summary_csv={summary_artifacts['summary_csv']}")
                print(f"[collocation-comparison] summary_json={summary_artifacts['summary_json']}")
                print(f"[collocation-comparison] failures_json={summary_artifacts['failures_json']}")
            raise SystemExit(int(result["return_code"]))

    if not args.dry_run:
        summary_artifacts = _write_summary_artifacts(output_root=output_root, manifest=manifest)
        manifest["artifacts"].update(summary_artifacts)
        save_manifest(str(manifest_path), manifest)
        print(f"[collocation-comparison] summary_csv={summary_artifacts['summary_csv']}")
        print(f"[collocation-comparison] summary_json={summary_artifacts['summary_json']}")
        print(f"[collocation-comparison] failures_json={summary_artifacts['failures_json']}")
    else:
        print("[collocation-comparison] dry-run: summary artifacts were not written")
    print(f"[collocation-comparison] run_manifest={manifest_path}")


if __name__ == "__main__":
    main()
