#!/usr/bin/env python3
"""Run raw 1D/2D loss-landscape evaluation for a saved PINN checkpoint."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pinn.analysis_data import build_analysis_bundle, move_analysis_bundle_to_device
from src.pinn.data import load_pinn_dataset_from_preprocessed_root
from src.pinn.landscape import evaluate_landscape_1d, evaluate_landscape_2d
from src.pinn.landscape.io import write_landscape_artifacts_1d, write_landscape_artifacts_2d
from src.pinn.losses import LossWeights
from src.sim.ode.model_definitions import SynchronousMachineModels
from src.train.runtime import cfg_get, resolve_dataset_root
from src.train.trainer import PinnModel


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to a saved PINN checkpoint (*.pt).")
    parser.add_argument("--dataset-root", default=None, help="Optional dataset root override.")
    parser.add_argument("--output-dir", default=None, help="Optional output directory override.")
    parser.add_argument("--grid", choices=["1d", "2d"], default=None, help="Landscape grid type.")
    parser.add_argument("--resolution", type=int, default=None, help="Number of alpha/beta points.")
    parser.add_argument("--alpha-min", type=float, default=None, help="Minimum alpha value.")
    parser.add_argument("--alpha-max", type=float, default=None, help="Maximum alpha value.")
    parser.add_argument("--beta-min", type=float, default=None, help="Minimum beta value.")
    parser.add_argument("--beta-max", type=float, default=None, help="Maximum beta value.")
    parser.add_argument("--split", default=None, help="Analysis split override: train|val|test.")
    parser.add_argument("--supervised-rows", type=int, default=None, help="Override supervised analysis rows.")
    parser.add_argument("--collocation-rows", type=int, default=None, help="Override collocation analysis rows.")
    parser.add_argument("--init-rows", type=int, default=None, help="Override init analysis rows.")
    parser.add_argument("--analysis-seed", type=int, default=None, help="Override analysis subset seed.")
    parser.add_argument("--direction-seed", type=int, default=None, help="Seed for random directions.")
    parser.add_argument("--normalization", default=None, help="Direction normalization mode.")
    parser.add_argument("--device", default="auto", help="Device preference for checkpoint loading.")
    parser.add_argument("--require-all-components", action="store_true", help="Require supervised, collocation, and init data.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve paths/config and print summary without evaluating.")
    return parser.parse_args()


def _default_output_dir(checkpoint_path: Path, grid: str) -> str:
    run_dir = checkpoint_path.parent.parent if checkpoint_path.parent.name == "checkpoints" else checkpoint_path.parent
    return str(run_dir / "loss_landscape" / f"{checkpoint_path.stem}_{grid}")


def _alpha_values(args: argparse.Namespace) -> np.ndarray:
    if int(args.resolution) <= 0:
        raise ValueError("--resolution must be > 0.")
    return np.linspace(float(args.alpha_min), float(args.alpha_max), int(args.resolution), dtype=np.float64)


def _beta_values(args: argparse.Namespace) -> np.ndarray:
    if int(args.resolution) <= 0:
        raise ValueError("--resolution must be > 0.")
    return np.linspace(float(args.beta_min), float(args.beta_max), int(args.resolution), dtype=np.float64)


def main() -> None:
    args = _parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "config" not in payload:
        raise ValueError("Checkpoint payload does not include a resolved config.")
    config = OmegaConf.create(payload["config"])

    current_cwd = os.getcwd()
    dataset_root = (
        str(Path(args.dataset_root).resolve())
        if args.dataset_root
        else resolve_dataset_root(config=config, original_cwd=current_cwd)
    )

    grid_type = str(args.grid if args.grid is not None else cfg_get(config, "pinn.landscape.grid.type", "2d"))
    resolution = int(args.resolution if args.resolution is not None else cfg_get(config, "pinn.landscape.grid.resolution", 21))
    alpha_min = float(args.alpha_min if args.alpha_min is not None else cfg_get(config, "pinn.landscape.grid.alpha_min", -1.0))
    alpha_max = float(args.alpha_max if args.alpha_max is not None else cfg_get(config, "pinn.landscape.grid.alpha_max", 1.0))
    beta_min = float(args.beta_min if args.beta_min is not None else cfg_get(config, "pinn.landscape.grid.beta_min", -1.0))
    beta_max = float(args.beta_max if args.beta_max is not None else cfg_get(config, "pinn.landscape.grid.beta_max", 1.0))
    direction_seed = int(args.direction_seed if args.direction_seed is not None else cfg_get(config, "pinn.landscape.direction_seed", 0))
    normalization = str(args.normalization if args.normalization is not None else cfg_get(config, "pinn.landscape.normalization", "filter"))
    analysis_split = str(args.split if args.split is not None else cfg_get(config, "pinn.analysis_dataset.split", "train"))
    supervised_rows = args.supervised_rows if args.supervised_rows is not None else cfg_get(config, "pinn.analysis_dataset.supervised_rows", None)
    collocation_rows = args.collocation_rows if args.collocation_rows is not None else cfg_get(config, "pinn.analysis_dataset.collocation_rows", None)
    init_rows = args.init_rows if args.init_rows is not None else cfg_get(config, "pinn.analysis_dataset.init_rows", None)
    analysis_seed = int(args.analysis_seed if args.analysis_seed is not None else cfg_get(config, "pinn.analysis_dataset.seed", 0))
    require_all_components = bool(args.require_all_components or cfg_get(config, "pinn.analysis_dataset.require_all_components", False))
    config_output_dir = cfg_get(config, "pinn.landscape.output_dir", None)
    output_dir = (
        str(Path(args.output_dir).resolve())
        if args.output_dir
        else (
            str(Path(str(config_output_dir)).resolve())
            if config_output_dir not in (None, "")
            else _default_output_dir(checkpoint_path, grid_type)
        )
    )
    checkpoint_tag = payload.get("checkpoint_tag", checkpoint_path.stem)

    summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_tag": str(checkpoint_tag),
        "dataset_root": dataset_root,
        "output_dir": output_dir,
        "grid": grid_type,
        "resolution": resolution,
        "split": analysis_split,
        "supervised_rows": supervised_rows,
        "collocation_rows": collocation_rows,
        "init_rows": init_rows,
        "analysis_seed": int(analysis_seed),
        "direction_seed": direction_seed,
        "normalization": normalization,
    }

    if args.dry_run:
        print("[loss-landscape] dry run summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        return

    dtype_name = str(cfg_get(config, "pinn.dtype", "float64"))
    dataset = load_pinn_dataset_from_preprocessed_root(dataset_root=dataset_root, dtype=dtype_name)
    analysis_bundle = build_analysis_bundle(
        dataset,
        split=analysis_split,
        supervised_rows=supervised_rows,
        collocation_rows=collocation_rows,
        init_rows=init_rows,
        seed=analysis_seed,
        require_all_components=require_all_components,
    )

    pinn_model = PinnModel.load_checkpoint(str(checkpoint_path), device_preference=args.device)
    analysis_bundle = move_analysis_bundle_to_device(
        analysis_bundle,
        device=pinn_model.device,
        dtype=pinn_model.dtype,
    )

    ode_model = SynchronousMachineModels(config)
    weights = LossWeights(
        data=float(cfg_get(config, "pinn.loss_weights.data", 1.0)),
        dt=float(cfg_get(config, "pinn.loss_weights.dt", 1.0e-4)),
        physics=float(cfg_get(config, "pinn.loss_weights.physics", 1.0)),
        ic=float(cfg_get(config, "pinn.loss_weights.ic", 1.0)),
    )

    manifest = {
        **summary,
        "analysis_bundle": analysis_bundle.as_manifest(),
        "weights": {
            "data": float(weights.data),
            "dt": float(weights.dt),
            "physics": float(weights.physics),
            "ic": float(weights.ic),
        },
        "formulation": str(cfg_get(config, "pinn.formulation", "odequations")),
        "dtype": dtype_name,
        "device": str(pinn_model.device),
    }

    if grid_type == "1d":
        args.resolution = resolution
        args.alpha_min = alpha_min
        args.alpha_max = alpha_max
        result = evaluate_landscape_1d(
            model=pinn_model.model,
            bundle=analysis_bundle,
            ode_model=ode_model,
            weights=weights,
            formulation=str(cfg_get(config, "pinn.formulation", "odequations")),
            alpha_values=_alpha_values(args),
            seed=direction_seed,
            normalization=normalization,
        )
        write_landscape_artifacts_1d(output_dir=output_dir, result=result, manifest=manifest)
    else:
        args.resolution = resolution
        args.alpha_min = alpha_min
        args.alpha_max = alpha_max
        args.beta_min = beta_min
        args.beta_max = beta_max
        result = evaluate_landscape_2d(
            model=pinn_model.model,
            bundle=analysis_bundle,
            ode_model=ode_model,
            weights=weights,
            formulation=str(cfg_get(config, "pinn.formulation", "odequations")),
            alpha_values=_alpha_values(args),
            beta_values=_beta_values(args),
            seed=direction_seed,
            normalization=normalization,
        )
        write_landscape_artifacts_2d(output_dir=output_dir, result=result, manifest=manifest)

    print(f"[loss-landscape] wrote raw artifacts to: {output_dir}")


if __name__ == "__main__":
    main()
