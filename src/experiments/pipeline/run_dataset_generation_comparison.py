"""Thesis dataset generation comparison: downstream surrogate data source evaluation.

Compares dataset generation methods (lhs_static, qbc_deep_ensemble, etc.) as training
data sources for a fixed downstream surrogate, evaluated on fixed external ID/OOD datasets.

Experiment matrix: method × budget × dataset_seed × baseline_seed.
Dataset generation and preprocessing run once per method × budget × dataset_seed.
Model seeds (historically named baseline seeds: bs01, bs02, bs03) are trained
on each generated dataset.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.experiments.pipeline.helpers.evaluation import resolve_evaluation_dataset as _resolve_evaluation_dataset
from src.experiments.pipeline.helpers.launch_utils import (
    format_hydra_list,
    init_experiment_manifest,
    parse_csv_list,
    run_logged_command,
    tag_stamp,
    upsert_run_status,
)
from src.experiments.pipeline.helpers.manifest import save_manifest, utc_now_iso
from src.experiments.pipeline.helpers.seeds import load_seed_registry
from src.experiments.pipeline.helpers.summary import (
    STANDARD_PINN_SUMMARY_FIELDNAMES,
    csv_value as _csv_value,
    eval_metrics as _eval_metrics_from_payload,
    extract_standard_pinn_summary_fields as _extract_standard_pinn_summary_fields,
    float_or_none as _float_or_none,
    mean_std as _mean_std_dict,
    read_json_if_exists as _read_json_if_exists,
    write_failures_json,
    write_summary_csv,
    write_summary_json,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
SEED_REGISTRY_PATH = REPO_ROOT / "src" / "config" / "registry" / "seeds.yaml"

# Directories that must never be touched by cleanup, regardless of arguments.
_PROTECTED_ROOTS: tuple[Path, ...] = (
    REPO_ROOT / "data" / "reference",
    REPO_ROOT / "data" / "evaluation",
)

ALL_METHODS = ("lhs_static", "qbc_deep_ensemble", "marker_directed", "qbc_marker_hybrid")
ALL_BUDGETS = ("b256", "b512", "b1024", "b2048", "b4096")
DEFAULT_OUTPUT_BASE = REPO_ROOT / "outputs" / "experiments" / "dataset_generation_comparison"

SMOKE_METHODS = ("lhs_static",)
SMOKE_BUDGETS = ("b256",)
SMOKE_DATASET_SEEDS = ("ds01",)
SMOKE_BASELINE_SEEDS = ("bs01",)
SMOKE_BASELINE_EPOCHS = 1
SMOKE_PRESET = "smoke"
SMOKE_STAGE1_OVERRIDES: tuple[str, ...] = ("model.ic_num_samples=32",)
SMOKE_STAGE2_OVERRIDES: tuple[str, ...] = ("dataset.validation_flag=false", "time=0.05", "num_of_points=20")

FINAL_METHODS = ALL_METHODS
FINAL_BUDGETS = ALL_BUDGETS
FINAL_DATASET_SEEDS = ("ds01", "ds02", "ds03", "ds04", "ds05")
FINAL_BASELINE_SEEDS = ("bs01", "bs02", "bs03")
FINAL_PRESET = "main"
DOWNSTREAM_MODELS = ("pinn_data_only",)
DEFAULT_PINN_DATA_ONLY_EPOCHS = 300
DEFAULT_ADAM_LR = 3.0e-3

DEFAULT_ID_EVAL_IDS: dict[str, str] = {
    "SM4": "id_SM4_lhs_b4096_eval01",
    "SM6": "id_SM6_lhs_b4096_eval01",
    "SM_AVR_GOV": "id_SM_AVR_GOV_lhs_b4096_eval01",
}
DEFAULT_OOD_EVAL_IDS: dict[str, str] = {
    "SM4": "ood_SM4_wide_ic_b4096_eval01",
    "SM6": "ood_SM6_wide_ic_b4096_eval01",
    "SM_AVR_GOV": "ood_SM_AVR_GOV_wide_ic_b4096_eval01",
}

SUMMARY_FIELDNAMES = [
    "run_name",
    "method",
    "budget",
    "dataset_seed_label",
    "dataset_seed_value",
    "downstream_model",
    "model_seed_label",
    "model_seed_value",
    "baseline_seed_label",
    "baseline_seed_value",
    "dataset_run_root",
    "dataset_root",
    "preprocessed_root",
    "baseline_run_dir",
    "downstream_run_dir",
    "id_eval_id",
    "id_eval_root",
    "ood_eval_id",
    "ood_eval_root",
    "status",
    "return_code",
    "failure_reason",
    "log_file",
    "internal_test_mse",
    "internal_test_rmse",
    "internal_test_mae",
    "id_eval_mse",
    "id_eval_rmse",
    "id_eval_mae",
    "ood_eval_mse",
    "ood_eval_rmse",
    "ood_eval_mae",
    "id_ood_rmse_gap",
    *STANDARD_PINN_SUMMARY_FIELDNAMES,
    "dataset_generation_seconds",
    "preprocessing_seconds",
    "baseline_training_seconds",
]


# ---------------------------------------------------------------------------
# Seed registry
# ---------------------------------------------------------------------------

def _validate_seed_labels(labels: list[str], registry: dict[str, int]) -> None:
    missing = [lb for lb in labels if lb not in registry]
    if missing:
        raise ValueError(
            f"Unknown seed label(s): {', '.join(missing)}. Check {SEED_REGISTRY_PATH}."
        )


def _safe_path_token(value: str) -> str:
    out: list[str] = []
    for char in value.strip().lower():
        if char.isalnum() or char in ("_", "-"):
            out.append(char)
        elif char in (",", "+", ".", " "):
            out.append("_")
    token = "".join(out).strip("_")
    return token or "run"


def _axis_label(
    *,
    values: list[str],
    default_values: tuple[str, ...],
    many_label: str,
) -> str | None:
    if len(values) == 1:
        return _safe_path_token(values[0])
    if tuple(values) == default_values:
        return None
    return f"{many_label}{len(values)}"


def _default_shard_label(
    *,
    mode: str,
    methods: list[str],
    budgets: list[str],
    dataset_seed_labels: list[str],
    baseline_seed_labels: list[str],
) -> str:
    default_methods = SMOKE_METHODS if mode == "smoke" else FINAL_METHODS
    default_budgets = SMOKE_BUDGETS if mode == "smoke" else FINAL_BUDGETS
    default_dataset_seeds = SMOKE_DATASET_SEEDS if mode == "smoke" else FINAL_DATASET_SEEDS
    default_baseline_seeds = SMOKE_BASELINE_SEEDS if mode == "smoke" else FINAL_BASELINE_SEEDS
    tokens = [
        _axis_label(values=methods, default_values=default_methods, many_label="methods"),
        _axis_label(values=budgets, default_values=default_budgets, many_label="budgets"),
        _axis_label(values=dataset_seed_labels, default_values=default_dataset_seeds, many_label="dataset_seeds"),
        _axis_label(values=baseline_seed_labels, default_values=default_baseline_seeds, many_label="model_seeds"),
    ]
    return "_".join(token for token in tokens if token) or "full"


# ---------------------------------------------------------------------------
# Evaluation dataset resolution
# ---------------------------------------------------------------------------

def _resolve_eval_inputs(
    args: argparse.Namespace,
    *,
    model_flag: str,
    mode: str,
) -> dict[str, str | None]:
    resolved: dict[str, str | None] = {
        "id_eval_id": None,
        "id_eval_root": None,
        "ood_eval_id": None,
        "ood_eval_root": None,
    }
    for kind in ("id", "ood"):
        disabled = bool(getattr(args, f"no_{kind}_eval", False))
        explicit_id = getattr(args, f"{kind}_eval_id", None)
        explicit_root = getattr(args, f"{kind}_eval_root", None)
        if disabled:
            if explicit_id or explicit_root:
                raise ValueError(
                    f"--no-{kind}-eval is mutually exclusive with "
                    f"--{kind}-eval-id and --{kind}-eval-root."
                )
            continue
        if explicit_id and explicit_root:
            raise ValueError(
                f"--{kind}-eval-id and --{kind}-eval-root are mutually exclusive."
            )
        eval_id = explicit_id
        eval_root = explicit_root
        if not eval_id and not eval_root:
            if mode == "smoke":
                # Smoke datasets use a compressed time grid (num_of_points=20)
                # incompatible with standard eval datasets. Skip by default.
                continue
            defaults = DEFAULT_ID_EVAL_IDS if kind == "id" else DEFAULT_OOD_EVAL_IDS
            eval_id = defaults.get(model_flag)
        if eval_id:
            entry = _resolve_evaluation_dataset(str(eval_id))
            resolved[f"{kind}_eval_id"] = str(eval_id)
            resolved[f"{kind}_eval_root"] = str(entry["dataset_root"])
        elif eval_root:
            resolved[f"{kind}_eval_root"] = str(Path(str(eval_root)).resolve())
    return resolved


# ---------------------------------------------------------------------------
# Command builder
# ---------------------------------------------------------------------------

def _build_run_experiment_command(
    *,
    python_bin: str,
    method: str,
    budget: str,
    dataset_seed: str,
    baseline_seeds: list[str],
    preset: str,
    model_flag: str,
    run_root: Path,
    baseline_epochs: int | None,
    id_eval_root: str | None,
    ood_eval_root: str | None,
    skip_baseline: bool = False,
    stage1_overrides: tuple[str, ...] = (),
    stage2_overrides: tuple[str, ...] = (),
    dataset_run_index: int | None = None,
    dataset_run_total: int | None = None,
) -> list[str]:
    cmd = [
        python_bin,
        "-m",
        "src.experiments.pipeline.run_experiment",
        "--method", method,
        "--budget", budget,
        "--dataset-seed", dataset_seed,
        "--preset", preset,
        "--experiment-id", "dataset_generation_comparison",
        "--model-flag", model_flag,
        "--run-root", str(run_root),
    ]
    for seed in baseline_seeds:
        cmd.extend(["--baseline-seed", seed])
    if skip_baseline:
        cmd.append("--skip-baseline")
    if baseline_epochs is not None:
        cmd.extend(["--baseline-epochs", str(int(baseline_epochs))])
    if dataset_run_index is not None and dataset_run_total is not None:
        cmd.extend(["--dataset-run-index", str(dataset_run_index)])
        cmd.extend(["--dataset-run-total", str(dataset_run_total)])
    for override in stage1_overrides:
        cmd.extend(["--stage1-override", override])
    for override in stage2_overrides:
        cmd.extend(["--stage2-override", override])
    if id_eval_root:
        cmd.extend(["--stage3-override", f"evaluation.id.root={id_eval_root}"])
    if ood_eval_root:
        cmd.extend(["--stage3-override", f"evaluation.ood.root={ood_eval_root}"])
    return cmd


def _qbc_storage_stage1_overrides(args: argparse.Namespace) -> tuple[str, ...]:
    overrides: list[str] = []
    if args.qbc_save_round_arrays is not None:
        overrides.append(f"qbc_save_round_arrays={str(bool(args.qbc_save_round_arrays)).lower()}")
    if args.qbc_save_dataset_checkpoints is not None:
        overrides.append(f"qbc_save_dataset_checkpoints={str(bool(args.qbc_save_dataset_checkpoints)).lower()}")
    if args.qbc_save_ensemble_checkpoints is not None:
        overrides.append(f"qbc_save_ensemble_checkpoints={str(bool(args.qbc_save_ensemble_checkpoints)).lower()}")
    return tuple(overrides)


def _build_pinn_data_only_command(
    *,
    python_bin: str,
    model_flag: str,
    seed_value: int,
    dataset_root: Path,
    run_dir: Path,
    epochs: int,
    adam_lr: float,
    device: str,
    dtype: str,
    batch_size: int,
    id_eval_root: str | None,
    ood_eval_root: str | None,
    wandb_use: bool,
    wandb_project: str,
    wandb_entity: str | None,
    wandb_group: str,
    wandb_name: str,
    wandb_tags: list[str],
) -> list[str]:
    command = [
        python_bin,
        "20_run_pinn.py",
        f"model.model_flag={model_flag}",
        f"model.seed={int(seed_value)}",
        f"dataset.root={dataset_root}",
        f"pinn.run_dir={run_dir}",
        f"pinn.device={device}",
        f"pinn.dtype={dtype}",
        f"pinn.default_batch_size={int(batch_size)}",
        "pinn.weighting.scheme=static",
        "pinn.loss_weights.data=1.0",
        "pinn.loss_weights.dt=0.0",
        "pinn.loss_weights.physics=0.0",
        "pinn.loss_weights.ic=0.0",
        "pinn.collocation.mode=preprocessed",
        "pinn.collocation.strategy=static",
        "pinn.supervised_sampling.enabled=false",
        "pinn.collocation.sampling.enabled=false",
        "pinn.gradient_telemetry.enabled=false",
        "pinn.checkpointing.enabled=true",
        "pinn.checkpointing.save_best=true",
        "pinn.checkpointing.save_last=false",
        "pinn.checkpointing.save_init=false",
        "pinn.checkpointing.epoch_fractions=[]",
        f"wandb.use={str(bool(wandb_use)).lower()}",
        "logging.log_every_epoch=1",
        (
            "pinn.optimizer_phases=["
            f"{{name:adam,optimizer:Adam,lr:{float(adam_lr)},"
            f"epochs:{int(epochs)},batch_size:{int(batch_size)},shuffle:true,"
            "full_batch:false,allow_sampling:false,optimizer_kwargs:{},"
            "scheduler:null,line_search:null,convergence:null}"
            "]"
        ),
    ]
    if id_eval_root:
        command.append(f"evaluation.id.root={id_eval_root}")
    if ood_eval_root:
        command.append(f"evaluation.ood.root={ood_eval_root}")
    if wandb_use:
        command.extend([
            f"wandb.project={wandb_project}",
            f"wandb.group={wandb_group}",
            f"wandb.name={wandb_name}",
            f"wandb.tags={format_hydra_list(wandb_tags)}",
        ])
        if wandb_entity:
            command.append(f"wandb.entity={wandb_entity}")
    return command


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def _iso_elapsed_seconds(started: str | None, completed: str | None) -> float | None:
    if not started or not completed:
        return None
    try:
        t0 = datetime.fromisoformat(str(started))
        t1 = datetime.fromisoformat(str(completed))
        return (t1 - t0).total_seconds()
    except (ValueError, TypeError):
        return None


def _summary_rows_for_dataset_run(
    dataset_run: dict[str, Any],
    *,
    baseline_seed_labels: list[str],
    seed_registry: dict[str, int],
    downstream_model: str,
    id_eval_id: str | None,
    id_eval_root: str | None,
    ood_eval_id: str | None,
    ood_eval_root: str | None,
) -> list[dict[str, Any]]:
    """Expand one dataset-level run entry into one row per baseline seed."""
    run_root_str = dataset_run.get("run_root") or dataset_run.get("run_dir")
    run_root = Path(str(run_root_str)) if run_root_str else None

    dataset_manifest: dict[str, Any] | None = None
    if run_root:
        dataset_manifest = _read_json_if_exists(run_root / "dataset_manifest.json")

    artifacts = dict(dataset_manifest.get("artifacts", {})) if dataset_manifest else {}
    stages = dict(dataset_manifest.get("stages", {})) if dataset_manifest else {}

    dataset_root = artifacts.get("dataset_root")
    preprocessed_root = artifacts.get("preprocessed_root") or dataset_root

    stage1 = dict(stages.get("stage1_create_dataset", {}))
    stage2 = dict(stages.get("stage2_preprocess", {}))
    dataset_generation_seconds = _iso_elapsed_seconds(
        stage1.get("started_at_utc"), stage1.get("completed_at_utc")
    )
    preprocessing_seconds = _iso_elapsed_seconds(
        stage2.get("started_at_utc"), stage2.get("completed_at_utc")
    )

    method = dataset_run.get("method")
    budget = dataset_run.get("budget")
    dataset_seed_label = dataset_run.get("dataset_seed_label")
    dataset_seed_value = dataset_run.get("dataset_seed_value")
    dataset_run_status = dataset_run.get("status")
    downstream_runs: dict[str, Any] = dict(dataset_run.get("downstream_runs", {}) or {})

    rows: list[dict[str, Any]] = []
    for seed_label in baseline_seed_labels:
        run_name = f"{method}_{budget}_{dataset_seed_label}_{seed_label}"
        baseline_run = dict(downstream_runs.get(seed_label, {}))

        if not baseline_run and dataset_run_status not in ("completed", "dry_run"):
            status = dataset_run_status or "failed"
            failure_reason = dataset_run.get("error")
            return_code = dataset_run.get("return_code")
            baseline_seed_value = seed_registry.get(seed_label)
            baseline_root = None
            metrics_payload = None
            baseline_training_seconds = None
        else:
            status = baseline_run.get("status") or "not_started"
            failure_reason = baseline_run.get("error")
            return_code = baseline_run.get("return_code")
            baseline_seed_value = (
                baseline_run.get("baseline_seed_value")
                or baseline_run.get("model_seed_value")
                or seed_registry.get(seed_label)
            )
            baseline_root = baseline_run.get("baseline_root") or baseline_run.get("run_dir")
            raw_payload = baseline_run.get("metrics_payload")
            metrics_payload = dict(raw_payload) if isinstance(raw_payload, dict) else None
            timings_payload = (
                _read_json_if_exists(Path(str(baseline_root)) / "timings.json")
                if baseline_root
                else None
            )
            baseline_training_seconds = (
                _float_or_none(timings_payload.get("training_seconds"))
                if timings_payload
                else _iso_elapsed_seconds(
                    baseline_run.get("started_at_utc"),
                    baseline_run.get("completed_at_utc"),
                )
            )

        final_test_metrics = dict(metrics_payload.get("final_test_metrics") or {}) if metrics_payload else {}
        id_eval_m = _eval_metrics_from_payload(metrics_payload, "id")
        ood_eval_m = _eval_metrics_from_payload(metrics_payload, "ood")
        id_rmse = _float_or_none(id_eval_m.get("rmse"))
        ood_rmse = _float_or_none(ood_eval_m.get("rmse"))
        standard_pinn_metrics = _extract_standard_pinn_summary_fields(
            metrics=metrics_payload,
            run_dir=Path(str(baseline_root)) if baseline_root else None,
        )

        rows.append({
            "run_name": run_name,
            "method": method,
            "budget": budget,
            "dataset_seed_label": dataset_seed_label,
            "dataset_seed_value": dataset_seed_value,
            "downstream_model": downstream_model,
            "model_seed_label": seed_label,
            "model_seed_value": baseline_seed_value,
            "baseline_seed_label": seed_label,
            "baseline_seed_value": baseline_seed_value,
            "dataset_run_root": str(run_root) if run_root else None,
            "dataset_root": dataset_root,
            "preprocessed_root": preprocessed_root,
            "baseline_run_dir": baseline_root,
            "downstream_run_dir": baseline_root,
            "id_eval_id": id_eval_id,
            "id_eval_root": id_eval_root,
            "ood_eval_id": ood_eval_id,
            "ood_eval_root": ood_eval_root,
            "status": status,
            "return_code": return_code,
            "failure_reason": failure_reason,
            "log_file": dataset_run.get("log_file"),
            "internal_test_mse": _float_or_none(final_test_metrics.get("mse")),
            "internal_test_rmse": _float_or_none(final_test_metrics.get("rmse")),
            "internal_test_mae": _float_or_none(final_test_metrics.get("mae")),
            "id_eval_mse": _float_or_none(id_eval_m.get("mse")),
            "id_eval_rmse": id_rmse,
            "id_eval_mae": _float_or_none(id_eval_m.get("mae")),
            "ood_eval_mse": _float_or_none(ood_eval_m.get("mse")),
            "ood_eval_rmse": ood_rmse,
            "ood_eval_mae": _float_or_none(ood_eval_m.get("mae")),
            "id_ood_rmse_gap": (
                None if id_rmse is None or ood_rmse is None else ood_rmse - id_rmse
            ),
            **standard_pinn_metrics,
            "dataset_generation_seconds": dataset_generation_seconds,
            "preprocessing_seconds": preprocessing_seconds,
            "baseline_training_seconds": baseline_training_seconds,
        })
    return rows


def _aggregate_by_method_budget(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    keys: set[tuple[str, str]] = set()
    for row in rows:
        m, b = row.get("method"), row.get("budget")
        if m and b:
            keys.add((str(m), str(b)))

    out: dict[str, dict[str, Any]] = {}
    for method, budget in sorted(keys):
        group_key = f"{method}/{budget}"
        members = [r for r in rows if r.get("method") == method and r.get("budget") == budget]
        successful = [r for r in members if r.get("status") == "completed"]
        failed = [r for r in members if r.get("status") not in ("completed", "dry_run")]

        def _floats(key: str) -> list[float]:
            return [float(r[key]) for r in successful if r.get(key) is not None]

        int_stats = _mean_std_dict(_floats("internal_test_rmse"))
        id_stats = _mean_std_dict(_floats("id_eval_rmse"))
        ood_stats = _mean_std_dict(_floats("ood_eval_rmse"))
        out[group_key] = {
            "method": method,
            "budget": budget,
            "success_count": len(successful),
            "failure_count": len(failed),
            "internal_test_rmse_mean": int_stats["mean"],
            "internal_test_rmse_std": int_stats["std"],
            "id_eval_rmse_mean": id_stats["mean"],
            "id_eval_rmse_std": id_stats["std"],
            "ood_eval_rmse_mean": ood_stats["mean"],
            "ood_eval_rmse_std": ood_stats["std"],
        }
    return out


def _write_summary_artifacts(
    *,
    output_root: Path,
    manifest: dict[str, Any],
    baseline_seed_labels: list[str],
    seed_registry: dict[str, int],
    downstream_model: str,
    id_eval_id: str | None,
    id_eval_root: str | None,
    ood_eval_id: str | None,
    ood_eval_root: str | None,
) -> dict[str, str]:
    all_rows: list[dict[str, Any]] = []
    for dataset_run in manifest.get("runs", []):
        rows = _summary_rows_for_dataset_run(
            dict(dataset_run),
            baseline_seed_labels=baseline_seed_labels,
            seed_registry=seed_registry,
            downstream_model=downstream_model,
            id_eval_id=id_eval_id,
            id_eval_root=id_eval_root,
            ood_eval_id=ood_eval_id,
            ood_eval_root=ood_eval_root,
        )
        all_rows.extend(rows)

    summary_csv = output_root / "summary.csv"
    summary_json = output_root / "summary.json"
    failures_json = output_root / "failures.json"

    write_summary_csv(summary_csv, rows=all_rows, fieldnames=SUMMARY_FIELDNAMES)

    failures = [
        {
            k: row.get(k)
            for k in (
                "method", "budget", "dataset_seed_label", "baseline_seed_label",
                "downstream_model",
                "status", "return_code", "failure_reason", "log_file", "dataset_run_root",
            )
        }
        for row in all_rows
        if row.get("status") not in ("completed", "dry_run")
    ]

    exp = manifest.get("experiment", {})
    summary_payload = {
        "generated_at_utc": utc_now_iso(),
        "output_root": str(output_root),
        "mode": exp.get("mode"),
        "model_flag": exp.get("model_flag"),
        "methods": exp.get("methods", []),
        "budgets": exp.get("budgets", []),
        "dataset_seeds": exp.get("dataset_seeds", []),
        "baseline_seeds": exp.get("baseline_seeds", []),
        "downstream_model": downstream_model,
        "id_eval_id": id_eval_id,
        "id_eval_root": id_eval_root,
        "ood_eval_id": ood_eval_id,
        "ood_eval_root": ood_eval_root,
        "rows": all_rows,
        "aggregates_by_method_budget": _aggregate_by_method_budget(all_rows),
    }
    write_summary_json(summary_json, summary_payload)
    write_failures_json(failures_json, failures)

    return {
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
        "failures_json": str(failures_json),
    }


# ---------------------------------------------------------------------------
# Cleanup helpers
# ---------------------------------------------------------------------------

def _is_under_protected_root(path: Path) -> bool:
    resolved = path.resolve()
    for protected in _PROTECTED_ROOTS:
        try:
            resolved.relative_to(protected.resolve())
            return True
        except ValueError:
            pass
    return False


def _cleanup_cell_data(
    *,
    cell_run_root: Path,
    output_root: Path,
    run_name: str,
    dry_run: bool,
) -> dict[str, Any]:
    """Delete large generated data under cell_run_root; preserve all result artifacts.

    Safe targets:
      <cell_run_root>/data/          — generated dataset HDF5 files
      <cell_run_root>/qbc/rounds/    — QBC round data (can be GB-scale)
      <cell_run_root>/qbc/checkpoints/ — QBC model checkpoints

    Preserved unconditionally:
      dataset_manifest.json, logs/, hydra/, baseline/, telemetry/, qbc/history.jsonl
    """
    result: dict[str, Any] = {
        "cleanup_performed": False,
        "cleanup_paths": [],
        "cleanup_error": None,
    }

    # In dry-run, directories may not exist yet — show planned targets.
    if dry_run:
        planned = [str(cell_run_root / "data")]
        for subdir in ("rounds", "checkpoints"):
            planned.append(str(cell_run_root / "qbc" / subdir))
        print(f"[dataset-gen-comparison] cleanup dry-run for {run_name}: would delete (if exists):")
        for p in planned:
            print(f"  {p}")
        result["cleanup_paths"] = planned
        return result

    if not cell_run_root.exists():
        return result

    # Guard: must be inside the experiment output root.
    try:
        cell_run_root.resolve().relative_to(output_root.resolve())
    except ValueError:
        result["cleanup_error"] = (
            f"Refusing cleanup: {cell_run_root} is not inside output root {output_root}."
        )
        return result

    # Guard: must not touch protected roots.
    if _is_under_protected_root(cell_run_root):
        result["cleanup_error"] = (
            f"Refusing cleanup: {cell_run_root} is inside a protected directory."
        )
        return result

    targets: list[Path] = []
    data_dir = cell_run_root / "data"
    if data_dir.exists():
        targets.append(data_dir)
    for subdir in ("rounds", "checkpoints"):
        p = cell_run_root / "qbc" / subdir
        if p.exists():
            targets.append(p)

    if not targets:
        result["cleanup_performed"] = True
        return result

    deleted: list[str] = []
    for target in targets:
        try:
            shutil.rmtree(str(target))
            deleted.append(str(target))
            print(f"[dataset-gen-comparison] cleanup: deleted {target}", flush=True)
        except Exception as exc:
            result["cleanup_error"] = f"Failed to delete {target}: {exc}"
            result["cleanup_paths"] = deleted
            result["cleanup_performed"] = len(deleted) > 0
            return result

    result["cleanup_performed"] = True
    result["cleanup_paths"] = deleted
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--python-bin", default=sys.executable, help="Python executable.")
    parser.add_argument(
        "--mode", default="smoke", choices=["smoke", "final"],
        help="Experiment mode. Controls default methods, budgets, seeds, epochs, and run label.",
    )
    parser.add_argument(
        "--experiment-tag", default=None,
        help="Output directory tag suffix. Default: timestamp.",
    )
    parser.add_argument(
        "--campaign-tag",
        default=None,
        help=(
            "Group split submissions under one campaign directory. "
            "When set without --output-root, outputs go to "
            "outputs/experiments/dataset_generation_comparison/<campaign-tag>/<shard-label>."
        ),
    )
    parser.add_argument(
        "--shard-label",
        default=None,
        help="Human-readable label for this split submission. Default: inferred from the selected matrix subset.",
    )
    parser.add_argument(
        "--output-root", default=None,
        help="Explicit output root. Default: outputs/experiments/dataset_generation_comparison/<tag>.",
    )
    parser.add_argument("--model-flag", default="SM4", help="Model flag (SM4, SM6, SM_AVR_GOV).")

    # Matrix
    parser.add_argument(
        "--methods", default=None,
        help=(
            f"Comma-separated dataset generation methods. "
            f"Choices: {', '.join(ALL_METHODS)}. Defaults by --mode."
        ),
    )
    parser.add_argument(
        "--budgets", default=None,
        help=f"Comma-separated budget labels. Choices: {', '.join(ALL_BUDGETS)}. Defaults by --mode.",
    )
    parser.add_argument(
        "--dataset-seeds", default=None,
        help="Comma-separated dataset seed labels from seeds.yaml. Defaults by --mode.",
    )
    parser.add_argument(
        "--baseline-seeds", default=None,
        help="Comma-separated baseline seed labels from seeds.yaml. Defaults by --mode.",
    )

    # Training
    parser.add_argument(
        "--downstream-model",
        default="pinn_data_only",
        choices=DOWNSTREAM_MODELS,
        help=(
            "Downstream model trained on each generated dataset. "
            "Only pinn_data_only is supported: 20_run_pinn.py with supervised data loss active "
            "and physics/dt/IC losses disabled."
        ),
    )
    parser.add_argument(
        "--baseline-epochs", type=int, default=None,
        help=(
            "Override downstream training epochs. Smoke default: 1. "
            f"For pinn_data_only final default is {DEFAULT_PINN_DATA_ONLY_EPOCHS}."
        ),
    )
    parser.add_argument("--device", default="auto", help="Device for pinn_data_only downstream runs.")
    parser.add_argument("--dtype", default="float64", help="Dtype for pinn_data_only downstream runs.")
    parser.add_argument("--batch-size", type=int, default=4096, help="Batch size for pinn_data_only downstream runs.")
    parser.add_argument("--adam-lr", type=float, default=DEFAULT_ADAM_LR, help="Adam learning rate for pinn_data_only downstream runs.")
    parser.add_argument(
        "--preset",
        default=None,
        help="Dataset-generation run label for manifests/output paths. Defaults by --mode.",
    )
    parser.add_argument(
        "--wandb-use",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable W&B for downstream pinn_data_only runs. Default: disabled.",
    )
    parser.add_argument("--wandb-project", default="thesis-dataset-generation", help="W&B project for downstream runs.")
    parser.add_argument("--wandb-entity", default=None, help="Optional W&B entity for downstream runs.")
    parser.add_argument("--wandb-tags", default="", help="Comma-separated extra W&B tags for downstream runs.")

    # Adaptive dataset-generation artifacts
    parser.add_argument(
        "--qbc-save-round-arrays",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Save per-round adaptive arrays under qbc/rounds. Default: stage-1 config default.",
    )
    parser.add_argument(
        "--qbc-save-dataset-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Save adaptive dataset resume checkpoints under qbc/checkpoints. Default: stage-1 config default.",
    )
    parser.add_argument(
        "--qbc-save-ensemble-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Save QBC ensemble model checkpoints under qbc/checkpoints. Default: stage-1 config default.",
    )

    # External evaluation
    parser.add_argument(
        "--id-eval-id", default=None,
        help="ID evaluation dataset ID from data/evaluation/index.json. Default: model-aware.",
    )
    parser.add_argument(
        "--id-eval-root", default=None,
        help="Explicit ID evaluation dataset root. Mutually exclusive with --id-eval-id.",
    )
    parser.add_argument(
        "--ood-eval-id", default=None,
        help="OOD evaluation dataset ID from data/evaluation/index.json. Default: model-aware.",
    )
    parser.add_argument(
        "--ood-eval-root", default=None,
        help="Explicit OOD evaluation dataset root. Mutually exclusive with --ood-eval-id.",
    )
    parser.add_argument(
        "--no-id-eval", action="store_true",
        help="Disable ID evaluation (no external ID eval root passed to baseline).",
    )
    parser.add_argument(
        "--no-ood-eval", action="store_true",
        help="Disable OOD evaluation.",
    )

    parser.add_argument(
        "--cleanup-data", action="store_true",
        help=(
            "After each dataset cell finishes, delete generated data artifacts "
            "(data/ and qbc/rounds/, qbc/checkpoints/) to save HPC storage. "
            "Preserves dataset_manifest.json, logs/, hydra/, pinn_data_only/, baseline/, and summary artifacts. "
            "Never deletes anything under data/reference or data/evaluation. "
            "Ignored during --dry-run (shows planned deletions instead)."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    mode = str(args.mode)
    model_flag = str(args.model_flag)
    downstream_model = str(args.downstream_model)
    cleanup_data = bool(args.cleanup_data)
    qbc_storage_overrides = _qbc_storage_stage1_overrides(args)
    wandb_use = bool(args.wandb_use)
    wandb_project = str(args.wandb_project)
    wandb_entity = str(args.wandb_entity) if args.wandb_entity else None
    wandb_extra_tags = parse_csv_list(str(args.wandb_tags)) if args.wandb_tags else []

    # Resolve matrix from mode defaults + CLI overrides
    methods: list[str] = (
        parse_csv_list(str(args.methods)) if args.methods else list(SMOKE_METHODS if mode == "smoke" else FINAL_METHODS)
    )
    budgets: list[str] = (
        parse_csv_list(str(args.budgets)) if args.budgets else list(SMOKE_BUDGETS if mode == "smoke" else FINAL_BUDGETS)
    )
    dataset_seed_labels: list[str] = (
        parse_csv_list(str(args.dataset_seeds))
        if args.dataset_seeds
        else list(SMOKE_DATASET_SEEDS if mode == "smoke" else FINAL_DATASET_SEEDS)
    )
    baseline_seed_labels: list[str] = (
        parse_csv_list(str(args.baseline_seeds))
        if args.baseline_seeds
        else list(SMOKE_BASELINE_SEEDS if mode == "smoke" else FINAL_BASELINE_SEEDS)
    )
    preset = str(args.preset) if args.preset else (SMOKE_PRESET if mode == "smoke" else FINAL_PRESET)
    baseline_epochs: int | None = (
        int(args.baseline_epochs)
        if args.baseline_epochs is not None
        else (
            SMOKE_BASELINE_EPOCHS
            if mode == "smoke"
            else DEFAULT_PINN_DATA_ONLY_EPOCHS
        )
    )

    # Validate methods
    unsupported = [m for m in methods if m not in ALL_METHODS]
    if unsupported:
        raise SystemExit(
            f"Unsupported method(s): {', '.join(unsupported)}. "
            f"Choose from: {', '.join(ALL_METHODS)}."
        )
    if not methods or not budgets or not dataset_seed_labels or not baseline_seed_labels:
        raise SystemExit("methods, budgets, dataset-seeds, and baseline-seeds must all be non-empty.")

    # Load and validate seeds
    try:
        seed_registry = load_seed_registry(SEED_REGISTRY_PATH)
        _validate_seed_labels(dataset_seed_labels + baseline_seed_labels, seed_registry)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from None

    # Resolve evaluation inputs
    try:
        eval_inputs = _resolve_eval_inputs(args, model_flag=model_flag, mode=mode)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from None

    id_eval_id = eval_inputs["id_eval_id"]
    id_eval_root = eval_inputs["id_eval_root"]
    ood_eval_id = eval_inputs["ood_eval_id"]
    ood_eval_root = eval_inputs["ood_eval_root"]

    # Output root. A campaign groups independently submitted shards under one
    # directory while keeping each shard manifest self-contained.
    stamp = args.experiment_tag or tag_stamp()
    campaign_tag = _safe_path_token(str(args.campaign_tag)) if args.campaign_tag else None
    shard_label = _safe_path_token(str(args.shard_label)) if args.shard_label else _default_shard_label(
        mode=mode,
        methods=methods,
        budgets=budgets,
        dataset_seed_labels=dataset_seed_labels,
        baseline_seed_labels=baseline_seed_labels,
    )
    if args.output_root:
        output_root = Path(args.output_root).resolve()
    elif campaign_tag:
        output_root = (DEFAULT_OUTPUT_BASE / campaign_tag / shard_label).resolve()
    else:
        output_root = (DEFAULT_OUTPUT_BASE / stamp).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    experiment_id = (
        f"dataset_generation_comparison_{mode}_{campaign_tag}_{shard_label}"
        if campaign_tag
        else f"dataset_generation_comparison_{mode}_{stamp}"
    )
    manifest_path = output_root / "run_manifest.json"
    manifest = init_experiment_manifest(
        run_root=str(output_root),
        experiment={
            "id": experiment_id,
            "tag": stamp,
            "campaign_tag": campaign_tag,
            "shard_label": shard_label,
            "mode": mode,
            "model_flag": model_flag,
            "methods": methods,
            "budgets": budgets,
            "dataset_seeds": dataset_seed_labels,
            "baseline_seeds": baseline_seed_labels,
            "downstream_model": downstream_model,
            "preset": preset,
            "baseline_epochs": baseline_epochs,
            "downstream_epochs": baseline_epochs,
            "adam_lr": float(args.adam_lr),
            "cleanup_data": cleanup_data,
            "qbc_save_round_arrays": args.qbc_save_round_arrays,
            "qbc_save_dataset_checkpoints": args.qbc_save_dataset_checkpoints,
            "qbc_save_ensemble_checkpoints": args.qbc_save_ensemble_checkpoints,
            "wandb_use": wandb_use,
            "wandb_project": wandb_project if wandb_use else None,
            "wandb_entity": wandb_entity if wandb_use else None,
        },
    )
    manifest["artifacts"]["id_eval_id"] = id_eval_id
    manifest["artifacts"]["id_eval_root"] = id_eval_root
    manifest["artifacts"]["ood_eval_id"] = ood_eval_id
    manifest["artifacts"]["ood_eval_root"] = ood_eval_root
    save_manifest(str(manifest_path), manifest)

    total_dataset_runs = len(methods) * len(budgets) * len(dataset_seed_labels)
    total_baseline_runs = total_dataset_runs * len(baseline_seed_labels)

    print(f"[dataset-gen-comparison] repo_root={REPO_ROOT}")
    print(f"[dataset-gen-comparison] mode={mode}")
    print(f"[dataset-gen-comparison] model_flag={model_flag}")
    print(f"[dataset-gen-comparison] campaign_tag={campaign_tag or '<none>'}")
    print(f"[dataset-gen-comparison] shard_label={shard_label}")
    print(f"[dataset-gen-comparison] methods={methods}")
    print(f"[dataset-gen-comparison] budgets={budgets}")
    print(f"[dataset-gen-comparison] dataset_seeds={dataset_seed_labels}")
    print(f"[dataset-gen-comparison] downstream_model={downstream_model}")
    print(f"[dataset-gen-comparison] baseline_seeds={baseline_seed_labels}")
    print(f"[dataset-gen-comparison] preset={preset}")
    print(f"[dataset-gen-comparison] downstream_epochs={baseline_epochs or '<config default>'}")
    print(f"[dataset-gen-comparison] adam_lr={float(args.adam_lr)}")
    if downstream_model == "pinn_data_only":
        print(f"[dataset-gen-comparison] pinn_data_only_device={args.device}")
        print(f"[dataset-gen-comparison] pinn_data_only_dtype={args.dtype}")
        print(f"[dataset-gen-comparison] pinn_data_only_batch_size={args.batch_size}")
    print(f"[dataset-gen-comparison] id_eval_id={id_eval_id or '<none>'}")
    print(f"[dataset-gen-comparison] id_eval_root={id_eval_root or '<none>'}")
    print(f"[dataset-gen-comparison] ood_eval_id={ood_eval_id or '<none>'}")
    print(f"[dataset-gen-comparison] ood_eval_root={ood_eval_root or '<none>'}")
    print(f"[dataset-gen-comparison] cleanup_data={cleanup_data}")
    print(f"[dataset-gen-comparison] wandb_use={wandb_use}")
    if wandb_use:
        print(f"[dataset-gen-comparison] wandb_project={wandb_project}")
        print(f"[dataset-gen-comparison] wandb_entity={wandb_entity or '<none>'}")
    if qbc_storage_overrides:
        print(f"[dataset-gen-comparison] qbc_storage_overrides={list(qbc_storage_overrides)}")
    print(f"[dataset-gen-comparison] total_dataset_runs={total_dataset_runs}")
    print(f"[dataset-gen-comparison] total_baseline_runs={total_baseline_runs}")
    print(f"[dataset-gen-comparison] output_root={output_root}")

    dataset_run_idx = 0
    for method in methods:
        for budget in budgets:
            for dataset_seed in dataset_seed_labels:
                dataset_run_idx += 1
                run_name = f"{method}_{budget}_{dataset_seed}"
                dataset_run_root = output_root / method / budget / dataset_seed
                log_path = output_root / "logs" / f"{run_name}.log"

                command = _build_run_experiment_command(
                    python_bin=args.python_bin,
                    method=method,
                    budget=budget,
                    dataset_seed=dataset_seed,
                    baseline_seeds=baseline_seed_labels,
                    preset=preset,
                    model_flag=model_flag,
                    run_root=dataset_run_root,
                    baseline_epochs=baseline_epochs,
                    id_eval_root=id_eval_root,
                    ood_eval_root=ood_eval_root,
                    skip_baseline=downstream_model == "pinn_data_only",
                    stage1_overrides=(
                        *(SMOKE_STAGE1_OVERRIDES if mode == "smoke" else ()),
                        *qbc_storage_overrides,
                    ),
                    stage2_overrides=SMOKE_STAGE2_OVERRIDES if mode == "smoke" else (),
                    dataset_run_index=dataset_run_idx,
                    dataset_run_total=total_dataset_runs,
                )

                run_metadata = {
                    "method": method,
                    "budget": budget,
                    "dataset_seed_label": dataset_seed,
                    "dataset_seed_value": seed_registry.get(dataset_seed),
                    "baseline_seeds": baseline_seed_labels,
                    "run_root": str(dataset_run_root),
                }
                upsert_run_status(
                    manifest,
                    run_name=run_name,
                    run_dir=str(dataset_run_root),
                    run_metadata=run_metadata,
                    status="running" if not args.dry_run else "dry_run",
                    command=command,
                    log_file=str(log_path),
                    started_at_utc=utc_now_iso(),
                )
                save_manifest(str(manifest_path), manifest)

                print(
                    f"[dataset-gen-comparison] dataset run {dataset_run_idx}/{total_dataset_runs}: "
                    f"{run_name}",
                    flush=True,
                )

                result = run_logged_command(
                    label="dataset-gen-comparison",
                    command=command,
                    log_path=log_path,
                    dry_run=bool(args.dry_run),
                    cwd=REPO_ROOT,
                )

                upsert_run_status(
                    manifest,
                    run_name=run_name,
                    run_dir=str(dataset_run_root),
                    run_metadata={},
                    status=str(result["status"]),
                    completed_at_utc=str(result["completed_at_utc"]),
                    return_code=int(result["return_code"]),
                    error=(
                        None
                        if int(result["return_code"]) == 0
                        else f"Dataset run '{run_name}' failed. See {log_path}"
                    ),
                    extra_fields={"elapsed_seconds": float(result["elapsed_seconds"])},
                )
                save_manifest(str(manifest_path), manifest)

                if int(result["return_code"]) == 0 and downstream_model == "pinn_data_only":
                    dataset_manifest_path = dataset_run_root / "dataset_manifest.json"
                    dataset_manifest = _read_json_if_exists(dataset_manifest_path)
                    artifacts = dict(dataset_manifest.get("artifacts", {}) or {}) if dataset_manifest else {}
                    preprocessed_root = artifacts.get("preprocessed_root") or artifacts.get("dataset_root")
                    if args.dry_run and not preprocessed_root:
                        preprocessed_root = str(dataset_run_root / "data" / model_flag / "dataset_v1")
                    if not preprocessed_root:
                        raise SystemExit(
                            f"Dataset run '{run_name}' completed but no preprocessed_root was found in "
                            f"{dataset_manifest_path}."
                        )
                    downstream_runs: dict[str, Any] = {}
                    for seed_idx, seed_label in enumerate(baseline_seed_labels, start=1):
                        seed_value = int(seed_registry[seed_label])
                        pinn_run_name = f"{run_name}_{seed_label}"
                        pinn_run_dir = dataset_run_root / "pinn_data_only" / seed_label
                        pinn_log_path = output_root / "logs" / f"{pinn_run_name}_pinn_data_only.log"
                        pinn_command = _build_pinn_data_only_command(
                            python_bin=args.python_bin,
                            model_flag=model_flag,
                            seed_value=seed_value,
                            dataset_root=Path(str(preprocessed_root)),
                            run_dir=pinn_run_dir,
                            epochs=int(baseline_epochs or DEFAULT_PINN_DATA_ONLY_EPOCHS),
                            adam_lr=float(args.adam_lr),
                            device=str(args.device),
                            dtype=str(args.dtype),
                            batch_size=int(args.batch_size),
                            id_eval_root=id_eval_root,
                            ood_eval_root=ood_eval_root,
                            wandb_use=wandb_use,
                            wandb_project=wandb_project,
                            wandb_entity=wandb_entity,
                            wandb_group=f"{campaign_tag or stamp}_{shard_label}",
                            wandb_name=pinn_run_name,
                            wandb_tags=[
                                "dataset_generation",
                                model_flag.lower(),
                                str(method),
                                str(budget),
                                str(dataset_seed),
                                str(seed_label),
                                *wandb_extra_tags,
                            ],
                        )
                        print(
                            f"[dataset-gen-comparison] pinn_data_only subrun "
                            f"{seed_idx}/{len(baseline_seed_labels)} seed={seed_label} starting",
                            flush=True,
                        )
                        pinn_result = run_logged_command(
                            label="dataset-gen-comparison:pinn_data_only",
                            command=pinn_command,
                            log_path=pinn_log_path,
                            dry_run=bool(args.dry_run),
                            cwd=REPO_ROOT,
                        )
                        metrics_path = pinn_run_dir / "metrics.json"
                        metrics_payload = (
                            _read_json_if_exists(metrics_path)
                            if metrics_path.exists()
                            else None
                        )
                        downstream_runs[seed_label] = {
                            "model_seed_label": seed_label,
                            "model_seed_value": seed_value,
                            "baseline_seed_label": seed_label,
                            "baseline_seed_value": seed_value,
                            "run_dir": str(pinn_run_dir),
                            "baseline_root": str(pinn_run_dir),
                            "status": str(pinn_result["status"]),
                            "return_code": int(pinn_result["return_code"]),
                            "error": (
                                None
                                if int(pinn_result["return_code"]) == 0
                                else f"pinn_data_only subrun failed. See {pinn_log_path}"
                            ),
                            "log_file": str(pinn_log_path),
                            "started_at_utc": str(pinn_result["started_at_utc"]),
                            "completed_at_utc": str(pinn_result["completed_at_utc"]),
                            "metrics_path": str(metrics_path) if metrics_path.exists() else None,
                            "metrics_payload": metrics_payload,
                        }
                        if int(pinn_result["return_code"]) != 0:
                            upsert_run_status(
                                manifest,
                                run_name=run_name,
                                run_dir=str(dataset_run_root),
                                run_metadata={},
                                status="failed",
                                error=f"pinn_data_only subrun '{seed_label}' failed. See {pinn_log_path}",
                                extra_fields={"downstream_runs": downstream_runs},
                            )
                            save_manifest(str(manifest_path), manifest)
                            raise SystemExit(int(pinn_result["return_code"]))
                    upsert_run_status(
                        manifest,
                        run_name=run_name,
                        run_dir=str(dataset_run_root),
                        run_metadata={},
                        status="completed" if not args.dry_run else "dry_run",
                        extra_fields={"downstream_runs": downstream_runs},
                    )
                    save_manifest(str(manifest_path), manifest)

                # Cleanup: run after every cell (success or failure path below), but
                # only when --cleanup-data is set. In dry-run, show planned deletions.
                if cleanup_data and (int(result["return_code"]) == 0 or args.dry_run):
                    cleanup_result = _cleanup_cell_data(
                        cell_run_root=dataset_run_root,
                        output_root=output_root,
                        run_name=run_name,
                        dry_run=bool(args.dry_run),
                    )
                    upsert_run_status(
                        manifest,
                        run_name=run_name,
                        run_dir=str(dataset_run_root),
                        run_metadata={},
                        status=str(result["status"]),
                        extra_fields={"cleanup": cleanup_result},
                    )
                    save_manifest(str(manifest_path), manifest)
                    if cleanup_result.get("cleanup_error"):
                        print(
                            f"[dataset-gen-comparison] WARNING: cleanup error for {run_name}: "
                            f"{cleanup_result['cleanup_error']}",
                            flush=True,
                        )

                if int(result["return_code"]) != 0:
                    if not args.dry_run:
                        summary_artifacts = _write_summary_artifacts(
                            output_root=output_root,
                            manifest=manifest,
                            baseline_seed_labels=baseline_seed_labels,
                            seed_registry=seed_registry,
                            downstream_model=downstream_model,
                            id_eval_id=id_eval_id,
                            id_eval_root=id_eval_root,
                            ood_eval_id=ood_eval_id,
                            ood_eval_root=ood_eval_root,
                        )
                        manifest["artifacts"].update(summary_artifacts)
                        save_manifest(str(manifest_path), manifest)
                        _print_summary_paths(summary_artifacts)
                    raise SystemExit(int(result["return_code"]))

    if not args.dry_run:
        summary_artifacts = _write_summary_artifacts(
            output_root=output_root,
            manifest=manifest,
            baseline_seed_labels=baseline_seed_labels,
            seed_registry=seed_registry,
            downstream_model=downstream_model,
            id_eval_id=id_eval_id,
            id_eval_root=id_eval_root,
            ood_eval_id=ood_eval_id,
            ood_eval_root=ood_eval_root,
        )
        manifest["artifacts"].update(summary_artifacts)
        save_manifest(str(manifest_path), manifest)
        _print_summary_paths(summary_artifacts)
    else:
        print("[dataset-gen-comparison] dry-run: summary artifacts not written")

    print(f"[dataset-gen-comparison] run_manifest={manifest_path}")


def _print_summary_paths(summary_artifacts: dict[str, str]) -> None:
    print(f"[dataset-gen-comparison] summary_csv={summary_artifacts['summary_csv']}")
    print(f"[dataset-gen-comparison] summary_json={summary_artifacts['summary_json']}")
    print(f"[dataset-gen-comparison] failures_json={summary_artifacts['failures_json']}")


if __name__ == "__main__":
    main()
