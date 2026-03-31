#!/usr/bin/env python3
"""Run a local deterministic smoke benchmark for quasi-Newton PINN optimizers.

This script launches one PINN training run per optimizer using a comparable full-batch
stage setup. It is intended for Phase 5A sanity checks before longer HPC campaigns.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_OPTIMIZERS = ("LBFGS", "BFGS", "SSBFGS", "SSBroyden")


def _run_command(command: list[str], dry_run: bool) -> None:
    print("[pinn-smoke] command:")
    print(" ".join(command))
    if dry_run:
        return
    proc = subprocess.run(command, cwd=REPO_ROOT, text=True, check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def _parse_optimizers(raw: str) -> list[str]:
    optimizers = [item.strip() for item in raw.split(",") if item.strip()]
    if not optimizers:
        raise ValueError("--optimizers must include at least one optimizer name.")
    unsupported = [item for item in optimizers if item not in SUPPORTED_OPTIMIZERS]
    if unsupported:
        supported_text = ", ".join(SUPPORTED_OPTIMIZERS)
        raise ValueError(f"Unsupported optimizer(s): {', '.join(unsupported)}. Use one of: {supported_text}.")
    return optimizers


def _format_hydra_list(items: list[str]) -> str:
    return "[" + ",".join(items) + "]"


def _tag_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


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


def _base_overrides(args: argparse.Namespace, group: str) -> list[str]:
    overrides = [
        f"model.model_flag={args.model_flag}",
        f"model.seed={int(args.seed)}",
        f"pinn.device={args.device}",
        f"pinn.dtype={args.dtype}",
        f"pinn.hidden_dim={int(args.hidden_dim)}",
        f"pinn.hidden_layers={int(args.hidden_layers)}",
        f"pinn.activation={args.activation}",
        f"pinn.default_batch_size={int(args.batch_size)}",
        "pinn.supervised_sampling.enabled=false",
        "pinn.collocation_sampling.enabled=false",
        f"pinn.loss_weights.data={args.loss_weight_data}",
        f"pinn.loss_weights.dt={args.loss_weight_dt}",
        f"pinn.loss_weights.physics={args.loss_weight_physics}",
        f"pinn.loss_weights.ic={args.loss_weight_ic}",
        f"logging.log_every_epoch={int(args.log_every_epoch)}",
    ]
    if args.dataset_root:
        overrides.append(f"dataset.root={Path(args.dataset_root).resolve()}")
    else:
        overrides.append(f"dataset.number={int(args.dataset_number)}")

    if args.wandb_use:
        overrides.extend(
            [
                "wandb.use=true",
                f"wandb.project={args.project}",
                f"wandb.group={group}",
            ]
        )
        if args.wandb_entity:
            overrides.append(f"wandb.entity={args.wandb_entity}")
    else:
        overrides.append("wandb.use=false")

    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="sm-surrogates-pinn", help="W&B project name when --wandb-use is set.")
    parser.add_argument("--wandb-entity", default=None, help="Optional W&B entity.")
    parser.add_argument("--wandb-use", action="store_true", help="Enable W&B logging for the smoke runs.")
    parser.add_argument("--model-flag", default="SM4", help="Model flag for setup_pinn.")
    parser.add_argument("--dataset-number", type=int, default=1, help="Dataset version number when dataset.root is not provided.")
    parser.add_argument("--dataset-root", default=None, help="Optional explicit dataset root (.../<MODEL>/dataset_vN).")
    parser.add_argument("--seed", type=int, default=37, help="Training seed.")
    parser.add_argument("--device", default="cpu", help="PINN device override: auto | cpu | cuda | mps.")
    parser.add_argument("--dtype", default="float64", help="PINN dtype. Recommended: float64.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="PINN hidden width.")
    parser.add_argument("--hidden-layers", type=int, default=4, help="PINN hidden depth.")
    parser.add_argument("--activation", default="tanh", help="PINN activation.")
    parser.add_argument("--batch-size", type=int, default=1024, help="Adam warm-up batch size.")
    parser.add_argument("--adam-warmup-epochs", type=int, default=0, help="Optional Adam warm-up epochs prepended before the quasi-Newton stage.")
    parser.add_argument("--adam-lr", type=float, default=1e-3, help="Learning rate for optional Adam warm-up.")
    parser.add_argument("--quasi-newton-epochs", type=int, default=20, help="Iterations/epochs for each quasi-Newton optimizer stage.")
    parser.add_argument("--quasi-newton-lr", type=float, default=1.0, help="Learning rate passed to the quasi-Newton optimizer stage.")
    parser.add_argument("--line-search", default="strong_wolfe", choices=["strong_wolfe", "backtracking"], help="Line-search method for BFGS-family stages.")
    parser.add_argument("--optimizers", default="LBFGS,BFGS,SSBFGS,SSBroyden", help="Comma-separated list of optimizers to run.")
    parser.add_argument("--loss-weight-data", type=float, default=1.0, help="Supervised data loss weight.")
    parser.add_argument("--loss-weight-dt", type=float, default=1.0e-4, help="Supervised derivative-consistency loss weight.")
    parser.add_argument("--loss-weight-physics", type=float, default=1.0e-4, help="Physics residual loss weight.")
    parser.add_argument("--loss-weight-ic", type=float, default=1.0e-3, help="Initial-condition loss weight.")
    parser.add_argument("--log-every-epoch", type=int, default=1, help="Metric/W&B logging cadence.")
    parser.add_argument("--tag", action="append", default=[], help="Optional extra tag. Can be passed multiple times.")
    parser.add_argument("--output-root", default=None, help="Explicit output root. Default: outputs/pinn/quasi_newton_smoke/<timestamp>.")
    parser.add_argument("--python-bin", default=sys.executable, help="Python executable for the training script.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    args = parser.parse_args()

    stamp = _tag_stamp()
    optimizers = _parse_optimizers(args.optimizers)
    group = f"pinn_quasi_newton_smoke_{args.model_flag.lower()}_{stamp}"
    run_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else (REPO_ROOT / "outputs" / "pinn" / "quasi_newton_smoke" / stamp).resolve()
    )
    base_overrides = _base_overrides(args=args, group=group)

    tags = [
        "quasi_newton_smoke",
        "local",
        args.model_flag.lower(),
        args.line_search,
        f"adamwarmup{int(args.adam_warmup_epochs)}",
        *args.tag,
    ]
    common_cmd = [args.python_bin, "20_run_pinn.py", *base_overrides, f"wandb.tags={_format_hydra_list(tags)}"]

    print(f"[pinn-smoke] output_root={run_root}")
    if args.wandb_use:
        print(f"[pinn-smoke] wandb.group={group}")

    for optimizer in optimizers:
        run_dir = run_root / optimizer.lower()
        stage_override = _stage_override(
            optimizer=optimizer,
            adam_warmup_epochs=args.adam_warmup_epochs,
            adam_lr=args.adam_lr,
            quasi_newton_epochs=args.quasi_newton_epochs,
            quasi_newton_lr=args.quasi_newton_lr,
            batch_size=args.batch_size,
            line_search_name=args.line_search,
        )
        command = [
            *common_cmd,
            f"pinn.run_dir={run_dir}",
            f"wandb.name=pinn_{optimizer.lower()}_smoke" if args.wandb_use else "hydra.job.name=pinn_smoke",
            stage_override,
        ]
        _run_command(command, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
