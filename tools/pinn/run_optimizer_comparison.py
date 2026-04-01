#!/usr/bin/env python3
"""Run a staged PINN optimizer comparison on one shared QBC dataset.

This launcher creates one dataset through the normal experiment pipeline and then
reuses the preprocessed dataset root for a matrix of PINN runs. It is intended for:

1. A small smoke comparison that checks the whole optimizer matrix runs end-to-end.
2. A larger benchmark comparison with fixed optimizer budgets.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_OPTIMIZERS = ("LBFGS", "BFGS", "SSBFGS", "SSBroyden")


def _run(command: list[str], *, dry_run: bool, extra_env: dict[str, str] | None = None) -> None:
    print("[optimizer-comparison] command:")
    print(" ".join(command))
    if dry_run:
        return
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    proc = subprocess.run(command, cwd=REPO_ROOT, text=True, check=False, env=env)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def _tag_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _format_hydra_list(items: list[str]) -> str:
    return "[" + ",".join(items) + "]"


def _parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_int_list(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def _parse_optimizers(raw: str) -> list[str]:
    optimizers = _parse_csv_list(raw)
    if not optimizers:
        raise ValueError("--optimizers must include at least one optimizer.")
    unsupported = [item for item in optimizers if item not in SUPPORTED_OPTIMIZERS]
    if unsupported:
        raise ValueError(
            f"Unsupported optimizer(s): {', '.join(unsupported)}. "
            f"Use one of: {', '.join(SUPPORTED_OPTIMIZERS)}."
        )
    return optimizers


def _profile_defaults(profile: str) -> tuple[list[int], list[int]]:
    profile_name = profile.strip().lower()
    if profile_name == "smoke":
        return [0, 100], [20]
    if profile_name == "benchmark":
        return [0, 100, 300, 10000], [100]
    raise ValueError("profile must be one of: smoke, benchmark")


def _build_dataset_command(
    *,
    python_bin: str,
    experiment_id: str,
    preset: str,
    budget: str,
    dataset_seed: str,
    model_flag: str,
    run_root: Path,
    stage1_overrides: list[str],
    stage2_overrides: list[str],
    dry_run: bool,
) -> Path:
    dataset_run_root = run_root / "dataset_pipeline"
    command = [
        python_bin,
        "src/pipeline/run_experiment.py",
        "--method",
        "qbc_deep_ensemble",
        "--budget",
        budget,
        "--dataset-seed",
        dataset_seed,
        "--preset",
        preset,
        "--experiment-id",
        experiment_id,
        "--model-flag",
        model_flag,
        "--run-root",
        str(dataset_run_root),
        "--skip-baseline",
    ]
    for override in stage1_overrides:
        command.extend(["--stage1-override", override])
    for override in stage2_overrides:
        command.extend(["--stage2-override", override])
    _run(command, dry_run=dry_run, extra_env={"PYTHONPATH": str(REPO_ROOT)})
    return dataset_run_root


def _dataset_root_from_manifest(dataset_run_root: Path, *, dry_run: bool, model_flag: str) -> Path:
    if dry_run:
        return dataset_run_root / "data" / model_flag / "dataset_v1"
    manifest_path = dataset_run_root / "dataset_manifest.json"
    manifest = _read_json(manifest_path)
    artifacts = dict(manifest.get("artifacts", {}))
    dataset_root = artifacts.get("preprocessed_root") or artifacts.get("dataset_root")
    if not dataset_root:
        raise RuntimeError(f"Dataset manifest does not contain a dataset root: {manifest_path}")
    return Path(str(dataset_root))


def _optimizer_stage(
    *,
    optimizer: str,
    epochs: int,
    lr: float,
    line_search_name: str,
) -> str:
    lower = optimizer.lower()
    optimizer_kwargs = "{}"
    if optimizer == "SSBFGS":
        optimizer_kwargs = "{tau_strategy:al_baali,tau_min:1.0e-12,tau_max:1.0}"
    elif optimizer == "SSBroyden":
        optimizer_kwargs = (
            "{tau_strategy:paper_default,phi_strategy:paper_default,"
            "tau_min:1.0e-6,tau_max:1.0,phi_min:-1.0e6,phi_max:1.0e6}"
        )
    elif optimizer == "BFGS":
        optimizer_kwargs = "{curvature_eps:1.0e-12,init_hessian_scale:1.0}"
    elif optimizer == "LBFGS":
        optimizer_kwargs = "{}"
        line_search_name = "strong_wolfe"

    return (
        "{"
        f"name:{lower},"
        f"optimizer:{optimizer},"
        f"lr:{lr},"
        f"epochs:{int(epochs)},"
        "batch_size:null,"
        "shuffle:false,"
        "full_batch:true,"
        "allow_sampling:false,"
        f"optimizer_kwargs:{optimizer_kwargs},"
        f"line_search:{{name:{line_search_name}}},"
        "convergence:null"
        "}"
    )


def _stage_override(
    *,
    optimizer: str,
    adam_warmup_epochs: int,
    adam_lr: float,
    quasi_newton_epochs: int,
    quasi_newton_lr: float,
    batch_size: int,
    line_search_name: str,
) -> str:
    stages: list[str] = []
    if adam_warmup_epochs > 0:
        stages.append(
            "{"
            "name:adam,"
            "optimizer:Adam,"
            f"lr:{adam_lr},"
            f"epochs:{int(adam_warmup_epochs)},"
            f"batch_size:{int(batch_size)},"
            "shuffle:true,"
            "full_batch:false,"
            "allow_sampling:false,"
            "optimizer_kwargs:{},"
            "line_search:null,"
            "convergence:null"
            "}"
        )
    stages.append(
        _optimizer_stage(
            optimizer=optimizer,
            epochs=quasi_newton_epochs,
            lr=quasi_newton_lr,
            line_search_name=line_search_name,
        )
    )
    return "pinn.stages=[" + ",".join(stages) + "]"


def _build_pinn_command(
    *,
    python_bin: str,
    model_flag: str,
    dataset_root: Path,
    run_dir: Path,
    wandb_project: str,
    wandb_group: str,
    wandb_name: str,
    wandb_entity: str | None,
    wandb_tags: list[str],
    seed: int,
    device: str,
    hidden_dim: int,
    hidden_layers: int,
    activation: str,
    dtype_name: str,
    batch_size: int,
    adam_warmup_epochs: int,
    adam_lr: float,
    quasi_newton_epochs: int,
    quasi_newton_lr: float,
    optimizer: str,
    line_search_name: str,
    log_every_epoch: int,
    loss_weight_data: float,
    loss_weight_dt: float,
    loss_weight_physics: float,
    loss_weight_ic: float,
    gradient_telemetry: bool,
) -> list[str]:
    stage_override = _stage_override(
        optimizer=optimizer,
        adam_warmup_epochs=adam_warmup_epochs,
        adam_lr=adam_lr,
        quasi_newton_epochs=quasi_newton_epochs,
        quasi_newton_lr=quasi_newton_lr,
        batch_size=batch_size,
        line_search_name=line_search_name,
    )
    command = [
        python_bin,
        "20_run_pinn.py",
        f"model.model_flag={model_flag}",
        f"model.seed={int(seed)}",
        f"dataset.root={dataset_root}",
        f"pinn.run_dir={run_dir}",
        f"pinn.device={device}",
        f"pinn.dtype={dtype_name}",
        f"pinn.hidden_dim={int(hidden_dim)}",
        f"pinn.hidden_layers={int(hidden_layers)}",
        f"pinn.activation={activation}",
        f"pinn.default_batch_size={int(batch_size)}",
        "pinn.supervised_sampling.enabled=false",
        "pinn.collocation_sampling.enabled=false",
        f"pinn.gradient_telemetry.enabled={'true' if gradient_telemetry else 'false'}",
        f"pinn.loss_weights.data={loss_weight_data}",
        f"pinn.loss_weights.dt={loss_weight_dt}",
        f"pinn.loss_weights.physics={loss_weight_physics}",
        f"pinn.loss_weights.ic={loss_weight_ic}",
        "wandb.use=true",
        f"wandb.project={wandb_project}",
        f"wandb.group={wandb_group}",
        f"wandb.name={wandb_name}",
        f"wandb.tags={_format_hydra_list(wandb_tags)}",
        f"logging.log_every_epoch={int(log_every_epoch)}",
        stage_override,
    ]
    if wandb_entity:
        command.append(f"wandb.entity={wandb_entity}")
    return command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-bin", default=sys.executable, help="Python executable for pipeline and training.")
    parser.add_argument("--profile", default="benchmark", choices=["smoke", "benchmark"], help="Run a reduced smoke matrix or the larger benchmark matrix.")
    parser.add_argument("--experiment-tag", default=None, help="Output/W&B tag suffix. Default: timestamp.")
    parser.add_argument("--output-root", default=None, help="Explicit output root. Default: outputs/pinn/optimizer_comparison/<tag>.")
    parser.add_argument("--model-flag", default="SM4", help="Model flag.")
    parser.add_argument("--preset", default="default", help="Dataset pipeline preset.")
    parser.add_argument("--budget", default="b1024", help="Budget label for QBC dataset generation.")
    parser.add_argument("--dataset-seed", default="s01", help="Dataset seed label from the registry.")
    parser.add_argument("--dataset-root", default=None, help="Optional explicit preprocessed dataset root. Skips dataset generation when set.")
    parser.add_argument("--seed", type=int, default=37, help="Training seed.")
    parser.add_argument("--device", default="cuda", help="PINN device override.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="PINN hidden width.")
    parser.add_argument("--hidden-layers", type=int, default=4, help="PINN hidden depth.")
    parser.add_argument("--activation", default="tanh", help="PINN activation.")
    parser.add_argument("--dtype", default="float64", help="PINN dtype.")
    parser.add_argument("--batch-size", type=int, default=1024, help="Adam warm-up batch size.")
    parser.add_argument("--adam-lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--quasi-newton-lr", type=float, default=1.0, help="Learning rate passed to quasi-Newton stages.")
    parser.add_argument("--line-search", default="strong_wolfe", choices=["strong_wolfe", "backtracking"], help="Line-search method for BFGS-family stages.")
    parser.add_argument("--optimizers", default="LBFGS,BFGS,SSBFGS,SSBroyden", help="Comma-separated optimizer list.")
    parser.add_argument("--warmup-epochs", default=None, help="Comma-separated Adam warm-up lengths. Default depends on --profile.")
    parser.add_argument("--quasi-newton-epochs", default=None, help="Comma-separated quasi-Newton epoch counts. Default depends on --profile.")
    parser.add_argument("--wandb-project", default="sm-surrogates-pinn-optimizer-comparison", help="Dedicated W&B project for this benchmark.")
    parser.add_argument("--wandb-entity", default=None, help="Optional W&B entity.")
    parser.add_argument("--loss-weight-data", type=float, default=1.0, help="Static supervised loss weight.")
    parser.add_argument("--loss-weight-dt", type=float, default=1.0e-4, help="Static dt loss weight.")
    parser.add_argument("--loss-weight-physics", type=float, default=1.0e-4, help="Static physics loss weight.")
    parser.add_argument("--loss-weight-ic", type=float, default=1.0e-3, help="Static IC loss weight.")
    parser.add_argument(
        "--gradient-telemetry",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable expensive gradient telemetry. Defaults to false for HPC safety.",
    )
    parser.add_argument("--log-every-epoch", type=int, default=1, help="Metric/W&B logging cadence.")
    parser.add_argument("--tag", action="append", default=[], help="Optional extra W&B tag. Can be passed multiple times.")
    parser.add_argument("--stage1-override", action="append", default=[], help="Extra stage-1 dataset-generation override.")
    parser.add_argument("--stage2-override", action="append", default=[], help="Extra stage-2 preprocess override.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    warmup_default, qn_default = _profile_defaults(args.profile)
    warmup_epochs = _parse_int_list(args.warmup_epochs) if args.warmup_epochs else warmup_default
    quasi_newton_epochs = _parse_int_list(args.quasi_newton_epochs) if args.quasi_newton_epochs else qn_default
    optimizers = _parse_optimizers(args.optimizers)

    stamp = args.experiment_tag or _tag_stamp()
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else (REPO_ROOT / "outputs" / "pinn" / "optimizer_comparison" / stamp).resolve()
    )
    output_root.mkdir(parents=True, exist_ok=True)

    experiment_id = f"optimizer_comparison_{args.budget}_{stamp}"
    wandb_group = f"optimizer_comparison_{args.model_flag.lower()}_{stamp}"

    if args.dataset_root:
        dataset_root = Path(args.dataset_root).resolve()
        dataset_pipeline_root = None
    else:
        dataset_pipeline_root = output_root / "dataset_pipeline"
        _build_dataset_command(
            python_bin=args.python_bin,
            experiment_id=experiment_id,
            preset=args.preset,
            budget=args.budget,
            dataset_seed=args.dataset_seed,
            model_flag=args.model_flag,
            run_root=output_root,
            stage1_overrides=list(args.stage1_override),
            stage2_overrides=list(args.stage2_override),
            dry_run=args.dry_run,
        )
        dataset_root = _dataset_root_from_manifest(
            dataset_pipeline_root,
            dry_run=args.dry_run,
            model_flag=args.model_flag,
        )

    summary: dict[str, Any] = {
        "experiment_tag": stamp,
        "experiment_id": experiment_id,
        "wandb_project": args.wandb_project,
        "wandb_group": wandb_group,
        "profile": args.profile,
        "budget": args.budget,
        "dataset_seed": args.dataset_seed,
        "model_flag": args.model_flag,
        "dataset_root": str(dataset_root),
        "dataset_pipeline_root": None if dataset_pipeline_root is None else str(dataset_pipeline_root),
        "warmup_epochs": warmup_epochs,
        "quasi_newton_epochs": quasi_newton_epochs,
        "optimizers": optimizers,
        "output_root": str(output_root),
        "runs": [],
    }

    tags_base = [
        "optimizer_comparison",
        args.profile,
        "qbc_deep_ensemble",
        args.budget,
        args.model_flag.lower(),
        *args.tag,
    ]

    for warmup in warmup_epochs:
        for qn_epochs in quasi_newton_epochs:
            for optimizer in optimizers:
                run_name = f"{optimizer.lower()}_warm{int(warmup)}_qn{int(qn_epochs)}"
                run_dir = output_root / "runs" / run_name
                wandb_tags = [*tags_base, optimizer.lower(), f"warmup{int(warmup)}", f"qn{int(qn_epochs)}"]
                command = _build_pinn_command(
                    python_bin=args.python_bin,
                    model_flag=args.model_flag,
                    dataset_root=dataset_root,
                    run_dir=run_dir,
                    wandb_project=args.wandb_project,
                    wandb_group=wandb_group,
                    wandb_name=run_name,
                    wandb_entity=args.wandb_entity,
                    wandb_tags=wandb_tags,
                    seed=args.seed,
                    device=args.device,
                    hidden_dim=args.hidden_dim,
                    hidden_layers=args.hidden_layers,
                    activation=args.activation,
                    dtype_name=args.dtype,
                    batch_size=args.batch_size,
                    adam_warmup_epochs=warmup,
                    adam_lr=args.adam_lr,
                    quasi_newton_epochs=qn_epochs,
                    quasi_newton_lr=args.quasi_newton_lr,
                    optimizer=optimizer,
                    line_search_name=args.line_search,
                    log_every_epoch=args.log_every_epoch,
                    loss_weight_data=args.loss_weight_data,
                    loss_weight_dt=args.loss_weight_dt,
                    loss_weight_physics=args.loss_weight_physics,
                    loss_weight_ic=args.loss_weight_ic,
                    gradient_telemetry=args.gradient_telemetry,
                )
                _run(command, dry_run=args.dry_run)
                summary["runs"].append(
                    {
                        "optimizer": optimizer,
                        "adam_warmup_epochs": int(warmup),
                        "quasi_newton_epochs": int(qn_epochs),
                        "run_name": run_name,
                        "run_dir": str(run_dir),
                    }
                )

    summary_path = output_root / "experiment_manifest.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[optimizer-comparison] summary_manifest={summary_path}")


if __name__ == "__main__":
    main()
