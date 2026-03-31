#!/usr/bin/env python3
"""Compute and optionally export compact loss-landscape artifacts for a QBC PINN experiment."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _run(command: list[str], *, dry_run: bool) -> None:
    print("[qbc-landscape] command:")
    print(" ".join(command))
    if dry_run:
        return
    proc = subprocess.run(command, cwd=REPO_ROOT, text=True, check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _bundle_export(
    *,
    experiment_root: Path,
    export_root: Path,
    manifest: dict[str, Any],
    include_metrics: bool,
    dry_run: bool,
) -> None:
    print(f"[qbc-landscape] export_root={export_root}")
    if dry_run:
        return

    export_root.mkdir(parents=True, exist_ok=True)
    _copy_file(experiment_root / "experiment_manifest.json", export_root / "experiment_manifest.json")

    export_manifest: dict[str, Any] = {
        "source_experiment_root": str(experiment_root),
        "runs": {},
    }

    for model_flag, run_info in sorted(dict(manifest.get("runs", {})).items()):
        model_key = model_flag.lower()
        model_export_root = export_root / model_key
        pinn_run_dir = Path(str(run_info["pinn_run_dir"]))
        loss_landscape_dir = pinn_run_dir / "loss_landscape"
        if not loss_landscape_dir.is_dir():
            continue

        model_export_root.mkdir(parents=True, exist_ok=True)
        if (pinn_run_dir / "config.yaml").is_file():
            _copy_file(pinn_run_dir / "config.yaml", model_export_root / "config.yaml")
        if include_metrics and (pinn_run_dir / "metrics.csv").is_file():
            _copy_file(pinn_run_dir / "metrics.csv", model_export_root / "metrics.csv")
        _copy_tree(loss_landscape_dir, model_export_root / "loss_landscape")

        export_manifest["runs"][model_flag] = {
            "source_pinn_run_dir": str(pinn_run_dir),
            "exported_config": str(model_export_root / "config.yaml"),
            "exported_metrics": str(model_export_root / "metrics.csv") if include_metrics else None,
            "exported_loss_landscape": str(model_export_root / "loss_landscape"),
        }

    with (export_root / "export_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(export_manifest, f, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", required=True, help="Root of one completed qbc_small_pinn experiment.")
    parser.add_argument("--python-bin", default=sys.executable, help="Python executable for the landscape script.")
    parser.add_argument("--models", default=None, help="Optional comma-separated subset of models to process.")
    parser.add_argument("--checkpoint-tag", default="best", help="Checkpoint tag to analyze: best | last | init | ...")
    parser.add_argument("--grid", choices=["1d", "2d", "both"], default="both", help="Landscape grid(s) to evaluate.")
    parser.add_argument("--resolution-1d", type=int, default=41, help="Resolution for 1D landscapes.")
    parser.add_argument("--resolution-2d", type=int, default=21, help="Resolution for 2D landscapes.")
    parser.add_argument("--alpha-min", type=float, default=-1.0, help="Minimum alpha value.")
    parser.add_argument("--alpha-max", type=float, default=1.0, help="Maximum alpha value.")
    parser.add_argument("--beta-min", type=float, default=-1.0, help="Minimum beta value.")
    parser.add_argument("--beta-max", type=float, default=1.0, help="Maximum beta value.")
    parser.add_argument("--split", default="train", help="Analysis split passed to run_loss_landscape.py.")
    parser.add_argument("--supervised-rows", type=int, default=1024, help="Analysis supervised row count.")
    parser.add_argument("--collocation-rows", type=int, default=1024, help="Analysis collocation row count.")
    parser.add_argument("--init-rows", type=int, default=128, help="Analysis init row count.")
    parser.add_argument("--analysis-seed", type=int, default=0, help="Analysis subset seed.")
    parser.add_argument("--direction-seed", type=int, default=0, help="Landscape direction seed.")
    parser.add_argument("--normalization", default="filter", help="Direction normalization.")
    parser.add_argument("--device", default="cuda", help="Device for loss-landscape evaluation.")
    parser.add_argument("--export-root", default=None, help="Optional compact export folder for git-friendly artifacts.")
    parser.add_argument("--skip-export", action="store_true", help="Compute landscapes but do not export artifacts.")
    parser.add_argument("--no-export-metrics", action="store_true", help="Do not copy metrics.csv into the export bundle.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    experiment_root = Path(args.experiment_root).resolve()
    manifest_path = experiment_root / "experiment_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Experiment manifest not found: {manifest_path}")
    manifest = _read_json(manifest_path)

    selected_models = None
    if args.models:
        selected_models = {item.strip() for item in args.models.split(",") if item.strip()}

    runs = dict(manifest.get("runs", {}))
    if not runs:
        raise ValueError(f"No runs found in experiment manifest: {manifest_path}")

    export_root = (
        Path(args.export_root).resolve()
        if args.export_root
        else (REPO_ROOT / "results" / "pinn_landscape" / experiment_root.name).resolve()
    )

    grid_modes = ["1d", "2d"] if args.grid == "both" else [args.grid]

    for model_flag, run_info in sorted(runs.items()):
        if selected_models is not None and model_flag not in selected_models:
            continue
        pinn_run_dir = Path(str(run_info["pinn_run_dir"]))
        checkpoint_path = pinn_run_dir / "checkpoints" / f"{args.checkpoint_tag}.pt"
        if not args.dry_run and not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found for {model_flag}: {checkpoint_path}")

        for grid in grid_modes:
            resolution = args.resolution_1d if grid == "1d" else args.resolution_2d
            output_dir = pinn_run_dir / "loss_landscape" / f"{args.checkpoint_tag}_{grid}"
            command = [
                args.python_bin,
                "tools/pinn/run_loss_landscape.py",
                "--checkpoint",
                str(checkpoint_path),
                "--grid",
                grid,
                "--resolution",
                str(int(resolution)),
                "--alpha-min",
                str(float(args.alpha_min)),
                "--alpha-max",
                str(float(args.alpha_max)),
                "--split",
                str(args.split),
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
                str(args.normalization),
                "--device",
                str(args.device),
                "--output-dir",
                str(output_dir),
                "--require-all-components",
            ]
            if grid == "2d":
                command.extend(
                    [
                        "--beta-min",
                        str(float(args.beta_min)),
                        "--beta-max",
                        str(float(args.beta_max)),
                    ]
                )
            _run(command, dry_run=args.dry_run)

    if not args.skip_export:
        _bundle_export(
            experiment_root=experiment_root,
            export_root=export_root,
            manifest=manifest,
            include_metrics=not bool(args.no_export_metrics),
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
