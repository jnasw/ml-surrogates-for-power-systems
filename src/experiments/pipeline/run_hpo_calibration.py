"""Run small thesis calibration sweeps before controlled final experiments."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    parse_label_list as _parse_seed_labels,
    seed_pairs_from_labels as _seed_pairs_from_labels,
)
from src.experiments.pipeline.helpers.summary import (
    component_loss as _component_loss,
    csv_value as _csv_value,
    float_or_none as _float_or_none,
    int_or_none as _int_or_none,
    mean_std as _mean_std,
    read_json_if_exists as _read_json_if_exists,
    write_failures_json,
    write_summary_csv,
    write_summary_json,
)
from src.experiments.pipeline.helpers.wandb import tags as wandb_tags_list


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REFERENCE_ID = "dev_SM4_lhs_b512_ds01"
STUDIES = ("pinn_architecture", "adam_lr", "second_order_lr", "baseline_architecture")
DEFAULT_WANDB_PROJECTS = {
    "pinn_architecture": "thesis-hpo-pinn-architecture",
    "adam_lr": "thesis-hpo-adam-lr",
    "second_order_lr": "thesis-hpo-second-order-lr",
    "baseline_architecture": "thesis-hpo-baseline-architecture",
}
SECOND_ORDER_OPTIMIZERS = ("LBFGS", "BFGS", "SSBFGS", "SSBroyden", "SOAP")
FULL_BATCH_OPTIMIZERS = frozenset({"LBFGS", "BFGS", "SSBFGS", "SSBroyden"})
SUMMARY_FIELDNAMES = [
    "study",
    "run_name",
    "seed_label",
    "seed_value",
    "dataset_reference_id",
    "dataset_root",
    "hyperparameters",
    "run_dir",
    "status",
    "return_code",
    "failure_reason",
    "log_file",
    "final_test_mse",
    "final_test_rmse",
    "final_test_mae",
    "final_train_mse",
    "final_train_rmse",
    "final_train_mae",
    "final_train_total_loss",
    "final_train_data_loss",
    "final_train_physics_loss",
    "final_train_dt_loss",
    "final_train_ic_loss",
    "test_data_loss",
    "global_epoch",
    "phase_name",
    "num_train_steps",
    "epoch_wall_seconds",
    "cumulative_wall_seconds",
    "total_seconds",
    "training_seconds",
]


@dataclass(frozen=True)
class CalibrationRunSpec:
    study: str
    run_name: str
    seed_label: str
    seed_value: int
    hyperparameters: dict[str, Any]
    command: list[str]
    run_dir: Path


def _parse_int_list(raw: str | None, *, default: tuple[int, ...]) -> list[int]:
    if raw is None or raw == "":
        return list(default)
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def _parse_float_list(raw: str | None, *, default: tuple[float, ...]) -> list[float]:
    if raw is None or raw == "":
        return list(default)
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one float value.")
    return values


def _resolve_dataset(args: argparse.Namespace) -> tuple[Path, str | None, dict[str, Any] | None]:
    if args.dataset_root and args.reference_id:
        raise ValueError("--dataset-root and --reference-id are mutually exclusive.")
    if args.dataset_root:
        return Path(args.dataset_root).resolve(), None, None
    reference_id = str(args.reference_id or DEFAULT_REFERENCE_ID)
    reference_entry = _resolve_reference_dataset(reference_id)
    return Path(str(reference_entry["preprocessed_root"])).resolve(), reference_id, reference_entry


def _float_token(value: float) -> str:
    text = f"{value:g}"
    return text.replace("-", "m").replace(".", "p")


def _safe_name(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def _pinn_batch_size(args: argparse.Namespace) -> int:
    return 1024 if args.batch_size is None else int(args.batch_size)


def _optimizer_stage(
    *,
    name: str,
    optimizer: str,
    lr: float,
    epochs: int,
    batch_size: int,
) -> str:
    optimizer_kwargs = "{}"
    line_search = "null"
    batch_size_value = str(int(batch_size))
    shuffle = "true"
    full_batch = "false"
    if optimizer in FULL_BATCH_OPTIMIZERS:
        batch_size_value = "null"
        shuffle = "false"
        full_batch = "true"
        line_search = "{name:strong_wolfe}"
        if optimizer == "BFGS":
            optimizer_kwargs = "{curvature_eps:1.0e-12,init_hessian_scale:1.0}"
        elif optimizer == "SSBFGS":
            optimizer_kwargs = "{tau_strategy:al_baali}"
        elif optimizer == "SSBroyden":
            optimizer_kwargs = "{tau_strategy:paper_default,phi_strategy:paper_default}"
    return (
        "{"
        f"name:{name},"
        f"optimizer:{optimizer},"
        f"lr:{lr},"
        f"epochs:{int(epochs)},"
        f"batch_size:{batch_size_value},"
        f"shuffle:{shuffle},"
        f"full_batch:{full_batch},"
        "allow_sampling:false,"
        f"optimizer_kwargs:{optimizer_kwargs},"
        f"line_search:{line_search},"
        "convergence:null"
        "}"
    )


def _pinn_command(
    *,
    python_bin: str,
    model_flag: str,
    seed_value: int,
    dataset_root: Path,
    run_dir: Path,
    device: str,
    dtype: str,
    hidden_dim: int,
    hidden_layers: int,
    activation: str,
    batch_size: int,
    optimizer_phase: str,
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
        f"pinn.hidden_dim={int(hidden_dim)}",
        f"pinn.hidden_layers={int(hidden_layers)}",
        f"pinn.activation={activation}",
        f"pinn.default_batch_size={int(batch_size)}",
        "pinn.weighting.scheme=static",
        "pinn.collocation.mode=preprocessed",
        "pinn.collocation.strategy=static",
        "pinn.supervised_sampling.enabled=false",
        "pinn.collocation.sampling.enabled=false",
        "pinn.gradient_telemetry.enabled=false",
        "wandb.use=true",
        f"wandb.project={wandb_project}",
        f"wandb.group={wandb_group}",
        f"wandb.name={wandb_name}",
        f"wandb.tags={format_hydra_list(wandb_tags)}",
        "logging.log_every_epoch=1",
        f"pinn.optimizer_phases=[{optimizer_phase}]",
    ]
    if wandb_entity:
        command.append(f"wandb.entity={wandb_entity}")
    return command


def _baseline_command(
    *,
    python_bin: str,
    model_flag: str,
    seed_value: int,
    dataset_root: Path,
    run_dir: Path,
    device: str,
    hidden_dim: int,
    hidden_layers: int,
    epochs: int | None,
    batch_size: int | None,
    wandb_project: str,
    wandb_entity: str | None,
    wandb_group: str,
    wandb_name: str,
    wandb_tags: list[str],
) -> list[str]:
    wandb_payload = (
        "{"
        "use:true,"
        f"project:{wandb_project},"
        f"group:{wandb_group},"
        f"name:{wandb_name},"
        f"tags:{format_hydra_list(wandb_tags)}"
        "}"
    )
    if wandb_entity:
        wandb_payload = (
            "{"
            "use:true,"
            f"project:{wandb_project},"
            f"entity:{wandb_entity},"
            f"group:{wandb_group},"
            f"name:{wandb_name},"
            f"tags:{format_hydra_list(wandb_tags)}"
            "}"
        )
    command = [
        python_bin,
        "10_run_baseline.py",
        f"model.model_flag={model_flag}",
        f"model.seed={int(seed_value)}",
        f"dataset.root={dataset_root}",
        f"baseline.run_dir={run_dir}",
        f"baseline.device={device}",
        f"baseline.hidden_dim={int(hidden_dim)}",
        f"baseline.hidden_layers={int(hidden_layers)}",
        f"+wandb={wandb_payload}",
    ]
    if epochs is not None:
        command.append(f"baseline.epochs={int(epochs)}")
    if batch_size is not None:
        command.append(f"baseline.batch_size={int(batch_size)}")
    return command


def _study_specs(
    *,
    args: argparse.Namespace,
    dataset_root: Path,
    dataset_reference_id: str | None,
    output_root: Path,
    seed_pairs: list[tuple[str, int]],
    wandb_project: str,
    wandb_tags: list[str],
) -> list[CalibrationRunSpec]:
    specs: list[CalibrationRunSpec] = []
    group = f"hpo_{args.study}_{args.model_flag.lower()}_{output_root.name}"

    def add_spec(
        *,
        run_name: str,
        seed_label: str,
        seed_value: int,
        hyperparameters: dict[str, Any],
        command: list[str],
        run_dir: Path,
    ) -> None:
        specs.append(
            CalibrationRunSpec(
                study=args.study,
                run_name=run_name,
                seed_label=seed_label,
                seed_value=seed_value,
                hyperparameters={
                    **hyperparameters,
                    "dataset_reference_id": dataset_reference_id,
                },
                command=command,
                run_dir=run_dir,
            )
        )

    if args.study == "pinn_architecture":
        hidden_dims = _parse_int_list(args.hidden_dims, default=(32, 64, 128))
        hidden_layers_values = _parse_int_list(args.hidden_layers, default=(3, 4, 5))
        epochs = int(args.epochs or 100)
        lr = _parse_float_list(args.lrs, default=(1.0e-3,))[0]
        batch_size = _pinn_batch_size(args)
        for hidden_dim in hidden_dims:
            for hidden_layers in hidden_layers_values:
                for seed_label, seed_value in seed_pairs:
                    run_name = f"hd{hidden_dim}_hl{hidden_layers}_{seed_label}"
                    run_dir = output_root / "runs" / run_name
                    phase = _optimizer_stage(
                        name="adam",
                        optimizer="Adam",
                        lr=lr,
                        epochs=epochs,
                        batch_size=batch_size,
                    )
                    hp = {
                        "hidden_dim": hidden_dim,
                        "hidden_layers": hidden_layers,
                        "activation": "tanh",
                        "optimizer": "Adam",
                        "lr": lr,
                        "epochs": epochs,
                        "batch_size": batch_size,
                    }
                    command = _pinn_command(
                        python_bin=args.python_bin,
                        model_flag=args.model_flag,
                        seed_value=seed_value,
                        dataset_root=dataset_root,
                        run_dir=run_dir,
                        device=args.device,
                        dtype=args.dtype,
                        hidden_dim=hidden_dim,
                        hidden_layers=hidden_layers,
                        activation="tanh",
                        batch_size=batch_size,
                        optimizer_phase=phase,
                        wandb_project=wandb_project,
                        wandb_entity=args.wandb_entity,
                        wandb_group=group,
                        wandb_name=run_name,
                        wandb_tags=[*wandb_tags, "pinn_architecture", f"hd{hidden_dim}", f"hl{hidden_layers}"],
                    )
                    add_spec(run_name=run_name, seed_label=seed_label, seed_value=seed_value, hyperparameters=hp, command=command, run_dir=run_dir)
    elif args.study == "adam_lr":
        lrs = _parse_float_list(args.lrs, default=(1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3))
        epochs = int(args.epochs or 100)
        hidden_dim = int(args.pinn_hidden_dim)
        hidden_layers = int(args.pinn_hidden_layers)
        batch_size = _pinn_batch_size(args)
        for lr in lrs:
            for seed_label, seed_value in seed_pairs:
                run_name = f"adam_lr{_float_token(lr)}_{seed_label}"
                run_dir = output_root / "runs" / run_name
                phase = _optimizer_stage(
                    name="adam",
                    optimizer="Adam",
                    lr=lr,
                    epochs=epochs,
                    batch_size=batch_size,
                )
                hp = {
                    "hidden_dim": hidden_dim,
                    "hidden_layers": hidden_layers,
                    "activation": "tanh",
                    "optimizer": "Adam",
                    "lr": lr,
                    "epochs": epochs,
                    "batch_size": batch_size,
                }
                command = _pinn_command(
                    python_bin=args.python_bin,
                    model_flag=args.model_flag,
                    seed_value=seed_value,
                    dataset_root=dataset_root,
                    run_dir=run_dir,
                    device=args.device,
                    dtype=args.dtype,
                    hidden_dim=hidden_dim,
                    hidden_layers=hidden_layers,
                    activation="tanh",
                    batch_size=batch_size,
                    optimizer_phase=phase,
                    wandb_project=wandb_project,
                    wandb_entity=args.wandb_entity,
                    wandb_group=group,
                    wandb_name=run_name,
                    wandb_tags=[*wandb_tags, "adam_lr", f"lr{_float_token(lr)}"],
                )
                add_spec(run_name=run_name, seed_label=seed_label, seed_value=seed_value, hyperparameters=hp, command=command, run_dir=run_dir)
    elif args.study == "second_order_lr":
        lrs = _parse_float_list(args.lrs, default=(0.05, 0.1, 0.3, 0.5, 1.0))
        optimizers = parse_csv_list(args.optimizers) if args.optimizers else list(SECOND_ORDER_OPTIMIZERS)
        unsupported = [name for name in optimizers if name not in SECOND_ORDER_OPTIMIZERS]
        if unsupported:
            raise ValueError(
                f"Unsupported second_order_lr optimizer(s): {', '.join(unsupported)}. "
                f"Use one of: {', '.join(SECOND_ORDER_OPTIMIZERS)}."
            )
        epochs = int(args.epochs or 100)
        hidden_dim = int(args.pinn_hidden_dim)
        hidden_layers = int(args.pinn_hidden_layers)
        batch_size = _pinn_batch_size(args)
        for optimizer in optimizers:
            for lr in lrs:
                for seed_label, seed_value in seed_pairs:
                    opt_token = _safe_name(optimizer)
                    run_name = f"{opt_token}_lr{_float_token(lr)}_{seed_label}"
                    run_dir = output_root / "runs" / run_name
                    phase = _optimizer_stage(
                        name=opt_token,
                        optimizer=optimizer,
                        lr=lr,
                        epochs=epochs,
                        batch_size=batch_size,
                    )
                    hp = {
                        "hidden_dim": hidden_dim,
                        "hidden_layers": hidden_layers,
                        "activation": "tanh",
                        "optimizer": optimizer,
                        "lr": lr,
                        "epochs": epochs,
                        "batch_size": batch_size,
                    }
                    command = _pinn_command(
                        python_bin=args.python_bin,
                        model_flag=args.model_flag,
                        seed_value=seed_value,
                        dataset_root=dataset_root,
                        run_dir=run_dir,
                        device=args.device,
                        dtype=args.dtype,
                        hidden_dim=hidden_dim,
                        hidden_layers=hidden_layers,
                        activation="tanh",
                        batch_size=batch_size,
                        optimizer_phase=phase,
                        wandb_project=wandb_project,
                        wandb_entity=args.wandb_entity,
                        wandb_group=group,
                        wandb_name=run_name,
                        wandb_tags=[*wandb_tags, "second_order_lr", opt_token, f"lr{_float_token(lr)}"],
                    )
                    add_spec(run_name=run_name, seed_label=seed_label, seed_value=seed_value, hyperparameters=hp, command=command, run_dir=run_dir)
    elif args.study == "baseline_architecture":
        hidden_dims = _parse_int_list(args.hidden_dims, default=(64, 128, 256))
        hidden_layers_values = _parse_int_list(args.hidden_layers, default=(3, 4, 5))
        for hidden_dim in hidden_dims:
            for hidden_layers in hidden_layers_values:
                for seed_label, seed_value in seed_pairs:
                    run_name = f"baseline_hd{hidden_dim}_hl{hidden_layers}_{seed_label}"
                    run_dir = output_root / "runs" / run_name
                    hp = {
                        "hidden_dim": hidden_dim,
                        "hidden_layers": hidden_layers,
                        "epochs": None if args.epochs is None else int(args.epochs),
                        "batch_size": None if args.batch_size is None else int(args.batch_size),
                    }
                    command = _baseline_command(
                        python_bin=args.python_bin,
                        model_flag=args.model_flag,
                        seed_value=seed_value,
                        dataset_root=dataset_root,
                        run_dir=run_dir,
                        device=args.device,
                        hidden_dim=hidden_dim,
                        hidden_layers=hidden_layers,
                        epochs=None if args.epochs is None else int(args.epochs),
                        batch_size=None if args.batch_size is None else int(args.batch_size),
                        wandb_project=wandb_project,
                        wandb_entity=args.wandb_entity,
                        wandb_group=group,
                        wandb_name=run_name,
                        wandb_tags=[*wandb_tags, "baseline_architecture", f"hd{hidden_dim}", f"hl{hidden_layers}"],
                    )
                    add_spec(run_name=run_name, seed_label=seed_label, seed_value=seed_value, hyperparameters=hp, command=command, run_dir=run_dir)
    else:
        raise ValueError(f"Unsupported study: {args.study}")
    return specs


def _summary_row_for_run(run: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(str(run.get("run_dir", "")))
    metrics = _read_json_if_exists(run_dir / "metrics.json") if run_dir else None
    timings = _read_json_if_exists(run_dir / "timings.json") if run_dir else None
    final_test_metrics = dict(metrics.get("final_test_metrics", {}) or {}) if metrics else {}
    final_train_metrics = dict(metrics.get("final_train_metrics", {}) or {}) if metrics else {}
    final_train_losses = dict(metrics.get("final_train_losses", {}) or {}) if metrics else {}
    final_epoch = dict(metrics.get("final_epoch", {}) or {}) if metrics else {}
    return {
        "study": run.get("study"),
        "run_name": run.get("run_name"),
        "seed_label": run.get("seed_label"),
        "seed_value": run.get("seed_value"),
        "dataset_reference_id": run.get("dataset_reference_id"),
        "dataset_root": run.get("dataset_root"),
        "hyperparameters": run.get("hyperparameters", {}),
        "run_dir": run.get("run_dir"),
        "status": run.get("status"),
        "return_code": run.get("return_code"),
        "failure_reason": run.get("error"),
        "log_file": run.get("log_file"),
        "final_test_mse": _float_or_none(final_test_metrics.get("mse")),
        "final_test_rmse": _float_or_none(final_test_metrics.get("rmse")),
        "final_test_mae": _float_or_none(final_test_metrics.get("mae")),
        "final_train_mse": _float_or_none(final_train_metrics.get("mse")),
        "final_train_rmse": _float_or_none(final_train_metrics.get("rmse")),
        "final_train_mae": _float_or_none(final_train_metrics.get("mae")),
        "final_train_total_loss": _float_or_none(final_train_losses.get("total_loss")),
        "final_train_data_loss": _component_loss(final_train_losses, "data"),
        "final_train_physics_loss": _component_loss(final_train_losses, "physics"),
        "final_train_dt_loss": _component_loss(final_train_losses, "dt"),
        "final_train_ic_loss": _component_loss(final_train_losses, "ic"),
        "test_data_loss": _float_or_none(final_epoch.get("test_data_loss")),
        "global_epoch": _int_or_none(final_epoch.get("global_epoch")),
        "phase_name": final_epoch.get("phase_name"),
        "num_train_steps": _int_or_none(final_epoch.get("num_train_steps")),
        "epoch_wall_seconds": _float_or_none(final_epoch.get("epoch_wall_seconds")),
        "cumulative_wall_seconds": _float_or_none(final_epoch.get("cumulative_wall_seconds")),
        "total_seconds": _float_or_none(timings.get("total_seconds") if timings else None),
        "training_seconds": _float_or_none(timings.get("training_seconds") if timings else None),
    }


def _aggregate_by_hyperparameters(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    keys = sorted({json.dumps(row.get("hyperparameters", {}), sort_keys=True) for row in rows})
    for key in keys:
        members = [row for row in rows if json.dumps(row.get("hyperparameters", {}), sort_keys=True) == key]
        successful = [row for row in members if row.get("status") == "completed"]
        failed = [row for row in members if row.get("status") != "completed"]
        rmse_values = [float(row["final_test_rmse"]) for row in successful if row.get("final_test_rmse") is not None]
        timing_values = [float(row["training_seconds"]) for row in successful if row.get("training_seconds") is not None]
        rmse_stats = _mean_std(rmse_values)
        timing_stats = _mean_std(timing_values)
        out[key] = {
            "hyperparameters": json.loads(key),
            "success_count": len(successful),
            "failure_count": len(failed),
            "final_test_rmse_mean": rmse_stats["mean"],
            "final_test_rmse_std": rmse_stats["std"],
            "training_seconds_mean": timing_stats["mean"],
            "training_seconds_std": timing_stats["std"],
        }
    return out


def _write_summary_artifacts(*, output_root: Path, manifest: dict[str, Any]) -> dict[str, str]:
    rows = [_summary_row_for_run(dict(run)) for run in manifest.get("runs", [])]
    summary_csv = output_root / "summary.csv"
    summary_json = output_root / "summary.json"
    failures_json = output_root / "failures.json"

    write_summary_csv(summary_csv, rows=rows, fieldnames=SUMMARY_FIELDNAMES)

    failures = [
        {
            "study": row.get("study"),
            "run_name": row.get("run_name"),
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
        "study": manifest.get("experiment", {}).get("study"),
        "dataset": {
            "source": manifest.get("artifacts", {}).get("dataset_source"),
            "reference_id": manifest.get("artifacts", {}).get("dataset_reference_id"),
            "root": manifest.get("artifacts", {}).get("dataset_root"),
            "reference": manifest.get("artifacts", {}).get("dataset_reference"),
        },
        "seed_labels": manifest.get("artifacts", {}).get("seed_labels", []),
        "seed_values": manifest.get("artifacts", {}).get("seed_values", []),
        "rows": rows,
        "aggregates_by_hyperparameters": _aggregate_by_hyperparameters(rows),
    }
    write_summary_json(summary_json, summary_payload)
    write_failures_json(failures_json, failures)
    return {
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
        "failures_json": str(failures_json),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--study", required=True, choices=STUDIES)
    parser.add_argument("--reference-id", default=None, help=f"Reference dataset ID. Default: {DEFAULT_REFERENCE_ID}.")
    parser.add_argument("--dataset-root", default=None, help="Explicit preprocessed dataset root. Mutually exclusive with --reference-id.")
    parser.add_argument("--model-flag", default="SM4")
    parser.add_argument("--seed-labels", default="s01", help="Comma-separated labels from src/config/registry/seeds.yaml.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float64")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-root", default=str(REPO_ROOT / "outputs" / "hpo"))
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-tags", default="", help="Comma-separated extra W&B tags.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lrs", default=None, help="Comma-separated LR override.")
    parser.add_argument("--hidden-dims", default=None, help="Comma-separated hidden_dim override.")
    parser.add_argument("--hidden-layers", default=None, help="Comma-separated hidden_layers override.")
    parser.add_argument("--optimizers", default=None, help="Comma-separated optimizer override for second_order_lr.")
    parser.add_argument("--pinn-hidden-dim", type=int, default=64)
    parser.add_argument("--pinn-hidden-layers", type=int, default=4)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        dataset_root, dataset_reference_id, dataset_reference = _resolve_dataset(args)
        seed_pairs = _seed_pairs_from_labels(_parse_seed_labels(str(args.seed_labels)))
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from None

    stamp = tag_stamp()
    output_root = (Path(args.output_root).resolve() / args.study / stamp).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    wandb_project = args.wandb_project or DEFAULT_WANDB_PROJECTS[args.study]
    extra_tags = parse_csv_list(args.wandb_tags) if args.wandb_tags else []
    base_tags = wandb_tags_list("hpo_calibration", args.study, args.model_flag.lower(), extra_tags)
    manifest_path = output_root / "run_manifest.json"
    manifest = init_experiment_manifest(
        run_root=str(output_root),
        experiment={
            "id": f"hpo_{args.study}_{stamp}",
            "study": args.study,
            "model_flag": args.model_flag,
        },
    )
    manifest["artifacts"]["dataset_root"] = str(dataset_root)
    manifest["artifacts"]["dataset_source"] = "provided" if dataset_reference_id is None else "reference"
    manifest["artifacts"]["dataset_reference_id"] = dataset_reference_id
    manifest["artifacts"]["dataset_reference"] = dataset_reference
    manifest["artifacts"]["seed_labels"] = [label for label, _ in seed_pairs]
    manifest["artifacts"]["seed_values"] = [value for _, value in seed_pairs]
    manifest["artifacts"]["wandb_project"] = wandb_project
    save_manifest(str(manifest_path), manifest)

    try:
        specs = _study_specs(
            args=args,
            dataset_root=dataset_root,
            dataset_reference_id=dataset_reference_id,
            output_root=output_root,
            seed_pairs=seed_pairs,
            wandb_project=wandb_project,
            wandb_tags=base_tags,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    manifest["artifacts"]["planned_run_count"] = len(specs)
    save_manifest(str(manifest_path), manifest)
    print(f"[hpo] study={args.study} planned_runs={len(specs)} output_root={output_root}")

    for spec in specs:
        log_path = output_root / "logs" / "runs" / f"{spec.run_name}.log"
        run_metadata = {
            "study": spec.study,
            "seed_label": spec.seed_label,
            "seed_value": spec.seed_value,
            "dataset_reference_id": dataset_reference_id,
            "dataset_root": str(dataset_root),
            "hyperparameters": spec.hyperparameters,
        }
        upsert_run_status(
            manifest,
            run_name=spec.run_name,
            status="running" if not args.dry_run else "dry_run",
            run_dir=str(spec.run_dir),
            run_metadata=run_metadata,
            command=spec.command,
            log_file=str(log_path),
            started_at_utc=utc_now_iso(),
        )
        save_manifest(str(manifest_path), manifest)
        result = run_logged_command(
            label="hpo",
            command=spec.command,
            log_path=log_path,
            dry_run=bool(args.dry_run),
            cwd=REPO_ROOT,
        )
        upsert_run_status(
            manifest,
            run_name=spec.run_name,
            status=str(result["status"]),
            run_dir=str(spec.run_dir),
            run_metadata=run_metadata,
            completed_at_utc=str(result["completed_at_utc"]),
            return_code=int(result["return_code"]),
            error=None if int(result["return_code"]) == 0 else f"Run '{spec.run_name}' failed. See {log_path}",
            extra_fields={"elapsed_seconds": float(result["elapsed_seconds"])},
        )
        save_manifest(str(manifest_path), manifest)
        if int(result["return_code"]) != 0:
            if not args.dry_run:
                summary_artifacts = _write_summary_artifacts(output_root=output_root, manifest=manifest)
                manifest["artifacts"].update(summary_artifacts)
                save_manifest(str(manifest_path), manifest)
            raise SystemExit(int(result["return_code"]))

    if not args.dry_run:
        summary_artifacts = _write_summary_artifacts(output_root=output_root, manifest=manifest)
        manifest["artifacts"].update(summary_artifacts)
        save_manifest(str(manifest_path), manifest)
        print(f"[hpo] summary_csv={summary_artifacts['summary_csv']}")
        print(f"[hpo] summary_json={summary_artifacts['summary_json']}")
        print(f"[hpo] failures_json={summary_artifacts['failures_json']}")
    else:
        print("[hpo] dry-run: summary artifacts were not written")
    print(f"[hpo] run_manifest={manifest_path}")


if __name__ == "__main__":
    main()
