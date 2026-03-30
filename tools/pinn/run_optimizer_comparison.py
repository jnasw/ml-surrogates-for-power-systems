#!/usr/bin/env python3
"""Run a simple local PINN optimizer comparison on W&B.

Launches two runs with identical settings except for the optimizer stage:
- Adam-only
- LBFGS-only

Both runs share the same W&B group so their curves can be compared directly.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _optimizer_stage_override(
    *,
    optimizer: str,
    epochs: int,
    lr: float,
    batch_size: int | None,
    shuffle: bool,
) -> str:
    batch_size_text = "null" if batch_size is None else str(int(batch_size))
    shuffle_text = "true" if shuffle else "false"
    name = optimizer.lower()
    return (
        "pinn.stages="
        f"[{{name:{name},optimizer:{optimizer},lr:{lr},epochs:{int(epochs)},batch_size:{batch_size_text},shuffle:{shuffle_text}}}]"
    )


def _base_overrides(args: argparse.Namespace, group: str) -> list[str]:
    overrides = [
        "wandb.use=true",
        f"wandb.project={args.project}",
        f"wandb.group={group}",
        f"logging.log_every_epoch={int(args.log_every_epoch)}",
        f"model.model_flag={args.model_flag}",
        f"model.seed={int(args.seed)}",
        f"dataset.number={int(args.dataset_number)}",
        f"pinn.device={args.device}",
        f"pinn.hidden_dim={int(args.hidden_dim)}",
        f"pinn.hidden_layers={int(args.hidden_layers)}",
        f"pinn.default_batch_size={int(args.batch_size)}",
        f"pinn.loss_weights.data={args.loss_weight_data}",
        f"pinn.loss_weights.dt={args.loss_weight_dt}",
        f"pinn.loss_weights.physics={args.loss_weight_physics}",
        f"pinn.loss_weights.ic={args.loss_weight_ic}",
    ]
    if args.dataset_root:
        overrides.append(f"dataset.root={Path(args.dataset_root).resolve()}")
    return overrides


def _run_command(command: list[str], dry_run: bool) -> None:
    print("[pinn-compare] command:")
    print(" ".join(command))
    if dry_run:
        return
    proc = subprocess.run(command, cwd=REPO_ROOT, text=True, check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="sm-surrogates-pinn", help="W&B project name.")
    parser.add_argument("--model-flag", default="SM4", help="Model flag for setup_pinn.")
    parser.add_argument("--dataset-number", type=int, default=1, help="Dataset version number when dataset.root is not provided.")
    parser.add_argument("--dataset-root", default=None, help="Optional explicit dataset root (.../<MODEL>/dataset_vN).")
    parser.add_argument("--seed", type=int, default=37, help="Training seed.")
    parser.add_argument("--device", default="auto", help="PINN device override: auto | cpu | cuda | mps.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="PINN hidden width.")
    parser.add_argument("--hidden-layers", type=int, default=4, help="PINN hidden depth.")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size for Adam run and default batch size.")
    parser.add_argument("--adam-epochs", type=int, default=300, help="Epochs for Adam-only run.")
    parser.add_argument("--adam-lr", type=float, default=1e-3, help="Learning rate for Adam-only run.")
    parser.add_argument("--lbfgs-epochs", type=int, default=300, help="Epochs for LBFGS-only run.")
    parser.add_argument("--lbfgs-lr", type=float, default=1.0, help="Learning rate for LBFGS-only run.")
    parser.add_argument("--loss-weight-data", type=float, default=1.0, help="Supervised data loss weight.")
    parser.add_argument("--loss-weight-dt", type=float, default=1.0e-4, help="Supervised derivative-consistency loss weight.")
    parser.add_argument("--loss-weight-physics", type=float, default=1.0e-4, help="Physics residual loss weight.")
    parser.add_argument("--loss-weight-ic", type=float, default=1.0e-3, help="Initial-condition loss weight.")
    parser.add_argument("--log-every-epoch", type=int, default=1, help="W&B logging cadence.")
    parser.add_argument("--tag", action="append", default=[], help="Optional extra W&B tag. Can be passed multiple times.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--python-bin", default=sys.executable, help="Python executable for the training script.")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    group = f"pinn_optimizer_compare_{args.model_flag.lower()}_{stamp}"
    run_root = REPO_ROOT / "outputs" / "pinn" / "optimizer_compare" / group
    base_overrides = _base_overrides(args=args, group=group)

    tags = ["optimizer_compare", "local", *args.tag]
    tag_override = f"wandb.tags=[{','.join(tags)}]"

    common_cmd = [args.python_bin, "20_run_pinn.py", *base_overrides, tag_override]

    adam_run_dir = run_root / "adam"
    adam_cmd = [
        *common_cmd,
        "wandb.name=pinn_adam",
        f"pinn.run_dir={adam_run_dir}",
        _optimizer_stage_override(
            optimizer="Adam",
            epochs=args.adam_epochs,
            lr=args.adam_lr,
            batch_size=args.batch_size,
            shuffle=True,
        ),
    ]

    lbfgs_run_dir = run_root / "lbfgs"
    lbfgs_cmd = [
        *common_cmd,
        "wandb.name=pinn_lbfgs",
        f"pinn.run_dir={lbfgs_run_dir}",
        _optimizer_stage_override(
            optimizer="LBFGS",
            epochs=args.lbfgs_epochs,
            lr=args.lbfgs_lr,
            batch_size=None,
            shuffle=False,
        ),
    ]

    print(f"[pinn-compare] wandb.group={group}")
    _run_command(adam_cmd, dry_run=args.dry_run)
    _run_command(lbfgs_cmd, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
