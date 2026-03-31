#!/usr/bin/env python3
"""Run the full small-QBC PINN landscape study in one pipeline.

Pipeline:
1. Generate and preprocess fresh qbc_deep_ensemble datasets.
2. Train simple Adam-only PINNs with gradient telemetry enabled.
3. Save init / midpoint / final checkpoints.
4. Compute loss landscapes for the selected checkpoints.
5. Export a compact analysis bundle to results/pinn_landscape/<experiment>.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(command: list[str], *, dry_run: bool) -> None:
    print("[qbc-landscape-study] command:")
    print(" ".join(command))
    if dry_run:
        return
    proc = subprocess.run(command, cwd=REPO_ROOT, text=True, check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-bin", default=sys.executable, help="Python executable for all sub-steps.")
    parser.add_argument("--experiment-tag", required=True, help="Experiment tag used for outputs, W&B grouping, and exports.")
    parser.add_argument(
        "--output-root",
        default=None,
        help="Root for raw HPC outputs. Default: outputs/pinn_hpc_experiments/<experiment-tag>.",
    )
    parser.add_argument(
        "--export-root",
        default=None,
        help="Root for compact repo-tracked analysis artifacts. Default: results/pinn_landscape/<experiment-tag>.",
    )
    parser.add_argument("--models", default="SM4,SM6,SM_AVR_GOV", help="Comma-separated model flags.")
    parser.add_argument("--preset", default="default", help="Dataset pipeline preset.")
    parser.add_argument("--budget", default="b1024", help="QBC budget label.")
    parser.add_argument("--dataset-seed", default="s01", help="Dataset seed label.")
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
        "--checkpoint-fractions",
        default="0.5",
        help="Comma-separated fractional checkpoint milestones. Default saves midpoint as epoch_050pct.",
    )
    parser.add_argument(
        "--landscape-checkpoint-tags",
        default="init,epoch_050pct,last",
        help="Comma-separated checkpoint tags to probe in the landscape stage.",
    )
    parser.add_argument("--grid", choices=["1d", "2d", "both"], default="both", help="Landscape grid(s) to evaluate.")
    parser.add_argument("--resolution-1d", type=int, default=41, help="Resolution for 1D landscapes.")
    parser.add_argument("--resolution-2d", type=int, default=21, help="Resolution for 2D landscapes.")
    parser.add_argument("--alpha-min", type=float, default=-1.0, help="Minimum alpha value.")
    parser.add_argument("--alpha-max", type=float, default=1.0, help="Maximum alpha value.")
    parser.add_argument("--beta-min", type=float, default=-1.0, help="Minimum beta value.")
    parser.add_argument("--beta-max", type=float, default=1.0, help="Maximum beta value.")
    parser.add_argument("--analysis-split", default="train", help="Analysis split for landscape evaluation.")
    parser.add_argument("--supervised-rows", type=int, default=1024, help="Analysis supervised row count.")
    parser.add_argument("--collocation-rows", type=int, default=1024, help="Analysis collocation row count.")
    parser.add_argument("--init-rows", type=int, default=128, help="Analysis init row count.")
    parser.add_argument("--analysis-seed", type=int, default=0, help="Analysis subset seed.")
    parser.add_argument("--direction-seed", type=int, default=0, help="Direction seed for landscape evaluation.")
    parser.add_argument("--normalization", default="filter", help="Landscape direction normalization mode.")
    parser.add_argument(
        "--stage1-override",
        action="append",
        default=[],
        help="Extra stage-1 dataset-generation override passed through run_experiment.py.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else (REPO_ROOT / "outputs" / "pinn_hpc_experiments" / args.experiment_tag).resolve()
    )
    export_root = (
        Path(args.export_root).resolve()
        if args.export_root
        else (REPO_ROOT / "results" / "pinn_landscape" / args.experiment_tag).resolve()
    )

    train_command = [
        args.python_bin,
        "tools/pinn/run_qbc_small_experiment.py",
        "--experiment-tag",
        args.experiment_tag,
        "--output-root",
        str(output_root),
        "--models",
        args.models,
        "--preset",
        args.preset,
        "--budget",
        args.budget,
        "--dataset-seed",
        args.dataset_seed,
        "--epochs",
        str(int(args.epochs)),
        "--lr",
        str(float(args.lr)),
        "--batch-size",
        str(int(args.batch_size)),
        "--device",
        args.device,
        "--hidden-dim",
        str(int(args.hidden_dim)),
        "--hidden-layers",
        str(int(args.hidden_layers)),
        "--activation",
        args.activation,
        "--dtype",
        args.dtype,
        "--log-every-epoch",
        str(int(args.log_every_epoch)),
        "--wandb-project",
        args.wandb_project,
        "--checkpoint-fractions",
        args.checkpoint_fractions,
        "--enable-gradient-telemetry",
        "--loss-weight-data",
        str(float(args.loss_weight_data)),
        "--loss-weight-dt",
        str(float(args.loss_weight_dt)),
        "--loss-weight-physics",
        str(float(args.loss_weight_physics)),
        "--loss-weight-ic",
        str(float(args.loss_weight_ic)),
    ]
    if args.wandb_entity:
        train_command.extend(["--wandb-entity", args.wandb_entity])
    for override in args.stage1_override:
        train_command.extend(["--stage1-override", override])
    if args.dry_run:
        train_command.append("--dry-run")
    _run(train_command, dry_run=args.dry_run)

    landscape_command = [
        args.python_bin,
        "tools/pinn/run_qbc_small_loss_landscape.py",
        "--experiment-root",
        str(output_root),
        "--models",
        args.models,
        "--checkpoint-tags",
        args.landscape_checkpoint_tags,
        "--grid",
        args.grid,
        "--resolution-1d",
        str(int(args.resolution_1d)),
        "--resolution-2d",
        str(int(args.resolution_2d)),
        "--alpha-min",
        str(float(args.alpha_min)),
        "--alpha-max",
        str(float(args.alpha_max)),
        "--beta-min",
        str(float(args.beta_min)),
        "--beta-max",
        str(float(args.beta_max)),
        "--split",
        args.analysis_split,
        "--supervised-rows",
        str(int(args.supervised_rows)),
        "--collocation-rows",
        str(int(args.collocation_rows)),
        "--init-rows",
        str(int(args.init_rows)),
        "--analysis-seed",
        str(int(args.analysis_seed)),
        "--direction-seed",
        str(int(args.direction_seed)),
        "--normalization",
        args.normalization,
        "--device",
        args.device,
        "--export-root",
        str(export_root),
    ]
    if args.dry_run:
        landscape_command.append("--dry-run")
    _run(landscape_command, dry_run=args.dry_run)

    summary = {
        "experiment_tag": args.experiment_tag,
        "output_root": str(output_root),
        "export_root": str(export_root),
        "landscape_checkpoint_tags": args.landscape_checkpoint_tags,
        "models": args.models,
    }
    print("[qbc-landscape-study] summary:")
    print(json.dumps(summary, indent=2))
    if not args.dry_run:
        experiment_manifest = _read_json(output_root / "experiment_manifest.json")
        study_manifest_path = export_root / "study_manifest.json"
        export_root.mkdir(parents=True, exist_ok=True)
        study_manifest = {
            **summary,
            "wandb_group": experiment_manifest.get("wandb_group"),
            "runs": experiment_manifest.get("runs", {}),
        }
        with study_manifest_path.open("w", encoding="utf-8") as f:
            json.dump(study_manifest, f, indent=2)
        print(f"[qbc-landscape-study] study_manifest={study_manifest_path}")


if __name__ == "__main__":
    main()
