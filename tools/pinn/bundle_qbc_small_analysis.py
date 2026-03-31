#!/usr/bin/env python3
"""Bundle analysis-ready artifacts from a small QBC-to-PINN experiment."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-root",
        required=True,
        help="Root directory of one qbc_small_pinn experiment.",
    )
    parser.add_argument(
        "--dst",
        required=True,
        help="Destination directory for the bundled analysis artifacts.",
    )
    parser.add_argument(
        "--include-loss-landscape",
        action="store_true",
        help="Also copy any existing loss_landscape folders from the PINN runs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be copied without writing files.",
    )
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _copy_file(src: Path, dst: Path, *, dry_run: bool) -> int:
    print(f"[pinn-bundle] file: {src} -> {dst}")
    if dry_run:
        return 0 if not src.exists() else src.stat().st_size
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return src.stat().st_size


def _copy_tree(src: Path, dst: Path, *, dry_run: bool) -> int:
    print(f"[pinn-bundle] tree: {src} -> {dst}")
    if dry_run:
        total = 0
        for path in src.rglob("*"):
            if path.is_file():
                total += path.stat().st_size
        return total
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    total = 0
    for path in src.rglob("*"):
        if path.is_file():
            total += path.stat().st_size
    return total


def _require_path(path: Path, description: str, *, allow_missing: bool = False) -> Path:
    if not allow_missing and not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def main() -> None:
    args = build_parser().parse_args()
    experiment_root = Path(args.experiment_root).resolve()
    dst_root = Path(args.dst).resolve()

    manifest_path = _require_path(experiment_root / "experiment_manifest.json", "experiment manifest")
    manifest = _read_json(manifest_path)
    runs = dict(manifest.get("runs", {}))
    if not runs:
        raise ValueError(f"No runs found in experiment manifest: {manifest_path}")

    copied_files = 0
    copied_bytes = 0

    copied_bytes += _copy_file(manifest_path, dst_root / "experiment_manifest.json", dry_run=args.dry_run)
    copied_files += 1

    for model_flag, run_info in sorted(runs.items()):
        model_dir = dst_root / model_flag.lower()
        allow_missing = bool(args.dry_run)
        dataset_pipeline_root = _require_path(
            Path(str(run_info["dataset_pipeline_root"])),
            f"{model_flag} dataset pipeline root",
            allow_missing=allow_missing,
        )
        dataset_root = _require_path(
            Path(str(run_info["dataset_root"])),
            f"{model_flag} dataset root",
            allow_missing=allow_missing,
        )
        pinn_run_dir = _require_path(
            Path(str(run_info["pinn_run_dir"])),
            f"{model_flag} PINN run dir",
            allow_missing=allow_missing,
        )

        dataset_manifest_path = _require_path(
            dataset_pipeline_root / "dataset_manifest.json",
            f"{model_flag} dataset manifest",
            allow_missing=allow_missing,
        )
        copied_bytes += _copy_file(
            dataset_manifest_path,
            model_dir / "dataset_pipeline" / "dataset_manifest.json",
            dry_run=args.dry_run,
        )
        copied_files += 1

        copied_bytes += _copy_tree(
            dataset_root,
            model_dir / "dataset_pipeline" / "data" / dataset_root.parent.name / dataset_root.name,
            dry_run=args.dry_run,
        )
        copied_files += sum(1 for path in dataset_root.rglob("*") if path.is_file())

        pinn_files = [
            pinn_run_dir / "config.yaml",
            pinn_run_dir / "metrics.csv",
        ]
        for src in pinn_files:
            if src.is_file():
                copied_bytes += _copy_file(src, model_dir / "pinn_run" / src.name, dry_run=args.dry_run)
                copied_files += 1

        checkpoints_dir = _require_path(
            pinn_run_dir / "checkpoints",
            f"{model_flag} checkpoints",
            allow_missing=allow_missing,
        )
        copied_bytes += _copy_tree(checkpoints_dir, model_dir / "pinn_run" / "checkpoints", dry_run=args.dry_run)
        copied_files += sum(1 for path in checkpoints_dir.rglob("*") if path.is_file())

        if args.include_loss_landscape:
            loss_landscape_dir = pinn_run_dir / "loss_landscape"
            if loss_landscape_dir.is_dir():
                copied_bytes += _copy_tree(
                    loss_landscape_dir,
                    model_dir / "pinn_run" / "loss_landscape",
                    dry_run=args.dry_run,
                )
                copied_files += sum(1 for path in loss_landscape_dir.rglob("*") if path.is_file())

    print(f"[pinn-bundle] experiment root: {experiment_root}")
    print(f"[pinn-bundle] destination: {dst_root}")
    print(f"[pinn-bundle] copied files: {copied_files}")
    print(f"[pinn-bundle] copied size: {copied_bytes / (1024 * 1024):.1f} MiB")


if __name__ == "__main__":
    main()
