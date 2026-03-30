"""Artifact I/O helpers for raw PINN loss-landscape runs."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch

from src.pinn.landscape.evaluate import RawLandscape1D, RawLandscape2D
from src.train.runtime import write_json


def ensure_output_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _save_direction(path: str, payload: dict[str, Any]) -> str:
    torch.save(payload, path)
    return path


def write_landscape_artifacts_1d(
    *,
    output_dir: str,
    result: RawLandscape1D,
    manifest: dict[str, Any],
) -> str:
    output_dir = ensure_output_dir(output_dir)
    np.save(os.path.join(output_dir, "loss_total.npy"), result.total)
    np.save(os.path.join(output_dir, "loss_data.npy"), result.data)
    np.save(os.path.join(output_dir, "loss_dt.npy"), result.dt)
    np.save(os.path.join(output_dir, "loss_physics.npy"), result.physics)
    np.save(os.path.join(output_dir, "loss_ic.npy"), result.ic)
    np.savez(os.path.join(output_dir, "coordinates.npz"), alpha=result.axes.alpha)
    _save_direction(
        os.path.join(output_dir, "direction.pt"),
        {
            "seed": int(result.direction.seed),
            "normalization": str(result.direction.normalization),
            "tensors": result.direction.tensors,
        },
    )
    write_json(os.path.join(output_dir, "manifest.json"), manifest)
    return output_dir


def write_landscape_artifacts_2d(
    *,
    output_dir: str,
    result: RawLandscape2D,
    manifest: dict[str, Any],
) -> str:
    output_dir = ensure_output_dir(output_dir)
    np.save(os.path.join(output_dir, "loss_total.npy"), result.total)
    np.save(os.path.join(output_dir, "loss_data.npy"), result.data)
    np.save(os.path.join(output_dir, "loss_dt.npy"), result.dt)
    np.save(os.path.join(output_dir, "loss_physics.npy"), result.physics)
    np.save(os.path.join(output_dir, "loss_ic.npy"), result.ic)
    np.savez(
        os.path.join(output_dir, "coordinates.npz"),
        alpha=result.axes.alpha,
        beta=result.axes.beta,
    )
    _save_direction(
        os.path.join(output_dir, "directions.pt"),
        {
            "direction_a": {
                "seed": int(result.direction_a.seed),
                "normalization": str(result.direction_a.normalization),
                "tensors": result.direction_a.tensors,
            },
            "direction_b": {
                "seed": int(result.direction_b.seed),
                "normalization": str(result.direction_b.normalization),
                "tensors": result.direction_b.tensors,
            },
        },
    )
    write_json(os.path.join(output_dir, "manifest.json"), manifest)
    return output_dir
