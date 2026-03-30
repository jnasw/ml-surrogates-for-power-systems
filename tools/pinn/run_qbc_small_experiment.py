#!/usr/bin/env python3
"""Run a small end-to-end QBC -> PINN experiment for multiple model orders.

Pipeline per model:
1. Generate a fresh qbc_deep_ensemble dataset with the existing experiment pipeline.
2. Preprocess the dataset into the PINN HDF5 layout.
3. Train the simple PINN with Adam-only for a fixed number of epochs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(command: list[str], *, dry_run: bool) -> None:
    print("[qbc-pinn] command:")
    print(" ".join(command))
    if dry_run:
        return
    proc = subprocess.run(command, cwd=REPO_ROOT, text=True, check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def _parse_models(raw: str) -> list[str]:
    models = [item.strip() for item in raw.split(",") if item.strip()]
    if not models:
        raise ValueError("--models must include at least one model flag.")
    return models


def _format_hydra_list(items: list[str]) -> str:
    return "[" + ",".join(items) + "]"


def _tag_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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
    dry_run: bool,
) -> Path:
    dataset_run_root = run_root / model_flag.lower() / "dataset_pipeline"
    command = [
        python_bin,
        "-m",
        "src.pipeline.run_experiment",
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
    _run(command, dry_run=dry_run)
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


def _build_pinn_command(
    *,
    python_bin: str,
    model_flag: str,
    dataset_root: Path,
    pinn_run_dir: Path,
    wandb_project: str,
    wandb_group: str,
    wandb_entity: str | None,
    wandb_tags: list[str],
    epochs: int,
    lr: float,
    batch_size: int,
    device: str,
    hidden_dim: int,
    hidden_layers: int,
    activation: str,
    dtype_name: str,
    log_every_epoch: int,
    loss_weight_data: float,
    loss_weight_dt: float,
    loss_weight_physics: float,
    loss_weight_ic: float,
) -> list[str]:
    stage_override = (
        "pinn.stages="
        f"[{{name:adam,optimizer:Adam,lr:{lr},epochs:{int(epochs)},batch_size:{int(batch_size)},shuffle:true}}]"
    )
    command = [
        python_bin,
        "20_run_pinn.py",
        f"model.model_flag={model_flag}",
        f"dataset.root={dataset_root}",
        f"pinn.run_dir={pinn_run_dir}",
        f"pinn.device={device}",
        f"pinn.dtype={dtype_name}",
        f"pinn.hidden_dim={int(hidden_dim)}",
        f"pinn.hidden_layers={int(hidden_layers)}",
        f"pinn.activation={activation}",
        f"pinn.default_batch_size={int(batch_size)}",
        f"pinn.loss_weights.data={loss_weight_data}",
        f"pinn.loss_weights.dt={loss_weight_dt}",
        f"pinn.loss_weights.physics={loss_weight_physics}",
        f"pinn.loss_weights.ic={loss_weight_ic}",
        "wandb.use=true",
        f"wandb.project={wandb_project}",
        f"wandb.group={wandb_group}",
        f"wandb.name=pinn_{model_flag.lower()}_qbc_b1024_adam300",
        f"wandb.tags={_format_hydra_list(wandb_tags)}",
        f"logging.log_every_epoch={int(log_every_epoch)}",
        stage_override,
    ]
    if wandb_entity:
        command.append(f"wandb.entity={wandb_entity}")
    return command


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-bin", default=sys.executable, help="Python executable for pipeline and training.")
    parser.add_argument("--experiment-tag", default=None, help="Output/W&B tag suffix. Default: timestamp.")
    parser.add_argument("--output-root", default=None, help="Explicit output root. Default: outputs/pinn_hpc_experiments/<tag>.")
    parser.add_argument("--models", default="SM4,SM6,SM_AVR_GOV", help="Comma-separated model flags.")
    parser.add_argument("--preset", default="default", help="Dataset pipeline preset.")
    parser.add_argument("--budget", default="b1024", help="Budget label for QBC dataset generation.")
    parser.add_argument("--dataset-seed", default="s01", help="Dataset seed label from the seed registry.")
    parser.add_argument("--epochs", type=int, default=300, help="Adam epochs per PINN run.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--batch-size", type=int, default=1024, help="PINN batch size.")
    parser.add_argument("--device", default="cuda", help="PINN device override.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="PINN hidden width.")
    parser.add_argument("--hidden-layers", type=int, default=4, help="PINN hidden depth.")
    parser.add_argument("--activation", default="tanh", help="PINN activation.")
    parser.add_argument("--dtype", default="float64", help="PINN dtype.")
    parser.add_argument("--log-every-epoch", type=int, default=1, help="Metric/W&B logging cadence.")
    parser.add_argument("--wandb-project", default="sm-surrogates-pinn", help="W&B project.")
    parser.add_argument("--wandb-entity", default=None, help="Optional W&B entity.")
    parser.add_argument("--loss-weight-data", type=float, default=1.0, help="Static supervised loss weight.")
    parser.add_argument("--loss-weight-dt", type=float, default=1.0e-4, help="Static dt loss weight.")
    parser.add_argument("--loss-weight-physics", type=float, default=1.0e-4, help="Static physics loss weight.")
    parser.add_argument("--loss-weight-ic", type=float, default=1.0e-3, help="Static IC loss weight.")
    parser.add_argument(
        "--stage1-override",
        action="append",
        default=[],
        help="Extra stage-1 dataset-generation override passed through run_experiment.py.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    stamp = args.experiment_tag or _tag_stamp()
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else (REPO_ROOT / "outputs" / "pinn_hpc_experiments" / stamp).resolve()
    )
    models = _parse_models(args.models)
    experiment_id = f"pinn_qbc_small_{args.budget}_{stamp}"
    wandb_group = f"pinn_qbc_{args.budget}_adam{int(args.epochs)}_{stamp}"

    summary: dict[str, Any] = {
        "experiment_tag": stamp,
        "experiment_id": experiment_id,
        "wandb_group": wandb_group,
        "budget": args.budget,
        "dataset_seed": args.dataset_seed,
        "models": models,
        "output_root": str(output_root),
        "runs": {},
    }
    output_root.mkdir(parents=True, exist_ok=True)

    for model_flag in models:
        print(f"[qbc-pinn] starting model={model_flag}")
        dataset_run_root = _build_dataset_command(
            python_bin=args.python_bin,
            experiment_id=experiment_id,
            preset=args.preset,
            budget=args.budget,
            dataset_seed=args.dataset_seed,
            model_flag=model_flag,
            run_root=output_root,
            stage1_overrides=list(args.stage1_override),
            dry_run=args.dry_run,
        )
        dataset_root = _dataset_root_from_manifest(dataset_run_root, dry_run=args.dry_run, model_flag=model_flag)
        pinn_run_dir = output_root / model_flag.lower() / f"pinn_adam{int(args.epochs)}"
        tags = [
            "hpc",
            "qbc_deep_ensemble",
            args.budget,
            f"adam{int(args.epochs)}",
            model_flag.lower(),
        ]
        pinn_command = _build_pinn_command(
            python_bin=args.python_bin,
            model_flag=model_flag,
            dataset_root=dataset_root,
            pinn_run_dir=pinn_run_dir,
            wandb_project=args.wandb_project,
            wandb_group=wandb_group,
            wandb_entity=args.wandb_entity,
            wandb_tags=tags,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            device=args.device,
            hidden_dim=args.hidden_dim,
            hidden_layers=args.hidden_layers,
            activation=args.activation,
            dtype_name=args.dtype,
            log_every_epoch=args.log_every_epoch,
            loss_weight_data=args.loss_weight_data,
            loss_weight_dt=args.loss_weight_dt,
            loss_weight_physics=args.loss_weight_physics,
            loss_weight_ic=args.loss_weight_ic,
        )
        _run(pinn_command, dry_run=args.dry_run)
        summary["runs"][model_flag] = {
            "dataset_pipeline_root": str(dataset_run_root),
            "dataset_root": str(dataset_root),
            "pinn_run_dir": str(pinn_run_dir),
        }

    summary_path = output_root / "experiment_manifest.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[qbc-pinn] summary_manifest={summary_path}")


if __name__ == "__main__":
    main()
