"""Pipeline-style collocation strategy comparison for small/local PINN sweeps.

This workflow:
1. Optionally generates one shared preprocessed dataset via the existing dataset
   experiment pipeline.
2. Launches a matrix of PINN runs against that dataset where the primary
   comparison axis is the collocation strategy.
3. Writes a run manifest plus per-run logs and a compact summary CSV suitable
   for local use today and HPC wrappers later.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from time import monotonic
from typing import Any

from src.pipeline.manifest import save_manifest, set_stage_status, utc_now_iso


REPO_ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_VARIANTS = (
    "static_preprocessed",
    "static_generated",
    "random_r",
    "rad",
    "rar_d",
    "rar_g",
)
PROFILE_CONFIGS: dict[str, dict[str, Any]] = {
    "smoke": {
        "budget": "b256",
        "preset": "smoke",
        "epochs": 30,
        "batch_size": 128,
        "device": "cpu",
        "gradient_telemetry": False,
        "variants": ["static_preprocessed", "static_generated", "random_r", "rad", "rar_d", "rar_g"],
        "stage1_overrides": [],
        "stage2_overrides": [
            "time=0.05",
            "num_of_points=20",
            "model.ic_generation_method=joint_lhs",
            "model.ic_num_samples=64",
        ],
        "collocation": {
            "active_points": 512,
            "candidate_points": 2048,
            "initial_points": 256,
            "append_points": 16,
            "refresh_period_epochs": 5,
            "refresh_mode": "epoch_periodic",
            "refresh_on_phase_start": [],
            "refresh_on_phase_end": [],
            "sampler": "random",
            "score_norm": "l2",
            "rad_k": 1.0,
            "rad_c": 1.0,
            "rar_d_k": 2.0,
            "rar_d_c": 0.0,
        },
    },
    "benchmark": {
        "budget": "b256",
        "preset": "default",
        "epochs": 100,
        "batch_size": 256,
        "device": "cuda",
        "gradient_telemetry": False,
        "variants": ["static_preprocessed", "static_generated", "random_r", "rad", "rar_d", "rar_g"],
        "stage1_overrides": [],
        "stage2_overrides": [
            "time=0.05",
            "num_of_points=20",
            "model.ic_generation_method=joint_lhs",
            "model.ic_num_samples=64",
        ],
        "collocation": {
            "active_points": 4096,
            "candidate_points": 16384,
            "initial_points": 2048,
            "append_points": 64,
            "refresh_period_epochs": 10,
            "refresh_mode": "epoch_periodic",
            "refresh_on_phase_start": [],
            "refresh_on_phase_end": [],
            "sampler": "random",
            "score_norm": "l2",
            "rad_k": 1.0,
            "rad_c": 1.0,
            "rar_d_k": 2.0,
            "rar_d_c": 0.0,
        },
    },
    "adam_periodic_benchmark": {
        "budget": "b256",
        "preset": "default",
        "epochs": 3000,
        "batch_size": 1024,
        "device": "cuda",
        "gradient_telemetry": False,
        "variants": ["static_generated", "random_r", "rad", "rar_d", "rar_g"],
        "stage1_overrides": [],
        "stage2_overrides": [
            "time=0.05",
            "num_of_points=80",
            "model.ic_generation_method=joint_lhs",
            "model.ic_num_samples=64",
        ],
        "collocation": {
            "active_points": 4096,
            "candidate_points": 16384,
            "initial_points": 2048,
            "append_points": 64,
            "refresh_period_epochs": 500,
            "refresh_mode": "epoch_periodic",
            "refresh_on_phase_start": [],
            "refresh_on_phase_end": [],
            "sampler": "random",
            "score_norm": "l2",
            "rad_k": 1.0,
            "rad_c": 1.0,
            "rar_d_k": 2.0,
            "rar_d_c": 0.0,
        },
        "optimizer_phases_override": (
            "pinn.optimizer_phases=["
            "{name:adam_periodic,optimizer:Adam,lr:0.001,epochs:3000,batch_size:1024,shuffle:true,full_batch:false,allow_sampling:true,optimizer_kwargs:{},line_search:null,convergence:null}"
            "]"
        ),
    },
    "adam_periodic_ssbroyden_benchmark": {
        "budget": "b256",
        "preset": "default",
        "epochs": 3000,
        "batch_size": 1024,
        "device": "cuda",
        "gradient_telemetry": False,
        "variants": ["static_generated", "random_r", "rad", "rar_d", "rar_g"],
        "stage1_overrides": [],
        "stage2_overrides": [
            "time=0.05",
            "num_of_points=80",
            "model.ic_generation_method=joint_lhs",
            "model.ic_num_samples=64",
        ],
        "collocation": {
            "active_points": 4096,
            "candidate_points": 16384,
            "initial_points": 2048,
            "append_points": 64,
            "refresh_period_epochs": 500,
            "refresh_mode": "epoch_periodic",
            "refresh_on_phase_start": [],
            "refresh_on_phase_end": [],
            "sampler": "random",
            "score_norm": "l2",
            "rad_k": 1.0,
            "rad_c": 1.0,
            "rar_d_k": 2.0,
            "rar_d_c": 0.0,
        },
        "optimizer_phases_override": (
            "pinn.optimizer_phases=["
            "{name:adam_periodic,optimizer:Adam,lr:0.001,epochs:3000,batch_size:1024,shuffle:true,full_batch:false,allow_sampling:true,optimizer_kwargs:{},line_search:null,convergence:null},"
            "{name:ssbroyden_tail,optimizer:SSBroyden,lr:1.0,epochs:2000,batch_size:null,shuffle:false,full_batch:true,allow_sampling:false,optimizer_kwargs:{},line_search:{name:strong_wolfe},convergence:null}"
            "]"
        ),
    },
    "multipool_benchmark": {
        "budget": "b256",
        "preset": "default",
        "epochs": 100,
        "batch_size": 256,
        "device": "cuda",
        "gradient_telemetry": False,
        "variants": ["static_generated", "rad", "rar_d"],
        "stage1_overrides": [],
        "stage2_overrides": [
            "time=0.05",
            "num_of_points=20",
            "model.ic_generation_method=joint_lhs",
            "model.ic_num_samples=64",
        ],
        "collocation": {
            "active_points": 4096,
            "candidate_points": 16384,
            "initial_points": 2048,
            "append_points": 64,
            "refresh_period_epochs": 10,
            "refresh_mode": "epoch_periodic",
            "refresh_on_phase_start": [],
            "refresh_on_phase_end": [],
            "sampler": "random",
            "score_norm": "l2",
            "rad_k": 1.0,
            "rad_c": 1.0,
            "rar_d_k": 2.0,
            "rar_d_c": 0.0,
            "multi_pool": {
                "enabled": True,
                "total_active_points": 4096,
                "allocation_enabled": True,
                "allocation_method": "loss_ratio",
                "allocation_update_interval_epochs": 1,
                "allocation_smoothing": 0.8,
                "residual_initial_fraction": 0.7,
                "residual_min_fraction": 0.4,
                "residual_max_fraction": 0.9,
                "ic_initial_fraction": 0.3,
                "ic_min_fraction": 0.1,
                "ic_max_fraction": 0.6,
            },
        },
    },
    "phase_boundary_reference": {
        "budget": "b256",
        "preset": "default",
        "epochs": 300,
        "batch_size": 1024,
        "device": "cuda",
        "gradient_telemetry": False,
        "variants": ["static_generated", "random_r", "rad", "rar_d", "rar_g"],
        "stage1_overrides": [],
        "stage2_overrides": [
            "time=0.05",
            "num_of_points=80",
            "model.ic_generation_method=joint_lhs",
            "model.ic_num_samples=64",
        ],
        "collocation": {
            "active_points": 4096,
            "candidate_points": 16384,
            "initial_points": 2048,
            "append_points": 64,
            "refresh_period_epochs": 350,
            "refresh_mode": "phase_boundary",
            "refresh_on_phase_start": ["adam_02", "adam_03", "adam_04", "adam_05", "ssbroyden_tail"],
            "refresh_on_phase_end": [],
            "sampler": "random",
            "score_norm": "l2",
            "rad_k": 1.0,
            "rad_c": 1.0,
            "rar_d_k": 2.0,
            "rar_d_c": 0.0,
        },
        "optimizer_phases_override": (
            "pinn.optimizer_phases=["
            "{name:adam_01,optimizer:Adam,lr:0.001,epochs:300,batch_size:1024,shuffle:true,full_batch:false,allow_sampling:true,optimizer_kwargs:{},line_search:null,convergence:null},"
            "{name:lbfgs_01,optimizer:LBFGS,lr:1.0,epochs:50,batch_size:null,shuffle:false,full_batch:true,allow_sampling:false,optimizer_kwargs:{},line_search:{name:strong_wolfe},convergence:null},"
            "{name:adam_02,optimizer:Adam,lr:0.0007,epochs:250,batch_size:1024,shuffle:true,full_batch:false,allow_sampling:true,optimizer_kwargs:{},line_search:null,convergence:null},"
            "{name:lbfgs_02,optimizer:LBFGS,lr:1.0,epochs:50,batch_size:null,shuffle:false,full_batch:true,allow_sampling:false,optimizer_kwargs:{},line_search:{name:strong_wolfe},convergence:null},"
            "{name:adam_03,optimizer:Adam,lr:0.0005,epochs:200,batch_size:1024,shuffle:true,full_batch:false,allow_sampling:true,optimizer_kwargs:{},line_search:null,convergence:null},"
            "{name:lbfgs_03,optimizer:LBFGS,lr:1.0,epochs:50,batch_size:null,shuffle:false,full_batch:true,allow_sampling:false,optimizer_kwargs:{},line_search:{name:strong_wolfe},convergence:null},"
            "{name:adam_04,optimizer:Adam,lr:0.0003,epochs:150,batch_size:1024,shuffle:true,full_batch:false,allow_sampling:true,optimizer_kwargs:{},line_search:null,convergence:null},"
            "{name:lbfgs_04,optimizer:LBFGS,lr:1.0,epochs:50,batch_size:null,shuffle:false,full_batch:true,allow_sampling:false,optimizer_kwargs:{},line_search:{name:strong_wolfe},convergence:null},"
            "{name:adam_05,optimizer:Adam,lr:0.0001,epochs:100,batch_size:1024,shuffle:true,full_batch:false,allow_sampling:true,optimizer_kwargs:{},line_search:null,convergence:null},"
            "{name:ssbroyden_tail,optimizer:SSBroyden,lr:1.0,epochs:500,batch_size:null,shuffle:false,full_batch:true,allow_sampling:false,optimizer_kwargs:{},line_search:{name:strong_wolfe},convergence:null}"
            "]"
        ),
    },
    "phase_boundary_benchmark": {
        "budget": "b256",
        "preset": "default",
        "epochs": 300,
        "batch_size": 1024,
        "device": "cuda",
        "gradient_telemetry": False,
        "variants": ["static_generated", "random_r", "rad", "rar_d", "rar_g"],
        "stage1_overrides": [],
        "stage2_overrides": [
            "time=0.05",
            "num_of_points=80",
            "model.ic_generation_method=joint_lhs",
            "model.ic_num_samples=64",
        ],
        "collocation": {
            "active_points": 4096,
            "candidate_points": 16384,
            "initial_points": 2048,
            "append_points": 64,
            "refresh_period_epochs": 350,
            "refresh_mode": "phase_boundary",
            "refresh_on_phase_start": ["warmup_02", "warmup_03", "warmup_04", "warmup_05", "warmup_tail"],
            "refresh_on_phase_end": [],
            "sampler": "random",
            "score_norm": "l2",
            "rad_k": 1.0,
            "rad_c": 1.0,
            "rar_d_k": 2.0,
            "rar_d_c": 0.0,
        },
        "optimizer_phases_override": (
            "pinn.optimizer_phases=["
            "{name:adam_01,optimizer:Adam,lr:0.001,epochs:300,batch_size:1024,shuffle:true,full_batch:false,allow_sampling:true,optimizer_kwargs:{},line_search:null,convergence:null},"
            "{name:lbfgs_01,optimizer:LBFGS,lr:1.0,epochs:50,batch_size:null,shuffle:false,full_batch:true,allow_sampling:false,optimizer_kwargs:{},line_search:{name:strong_wolfe},convergence:null},"
            "{name:warmup_02,optimizer:Adam,lr:0.0001,epochs:25,batch_size:1024,shuffle:true,full_batch:false,allow_sampling:true,optimizer_kwargs:{},line_search:null,convergence:null},"
            "{name:adam_02,optimizer:Adam,lr:0.0005,epochs:225,batch_size:1024,shuffle:true,full_batch:false,allow_sampling:true,optimizer_kwargs:{},line_search:null,convergence:null},"
            "{name:lbfgs_02,optimizer:LBFGS,lr:1.0,epochs:50,batch_size:null,shuffle:false,full_batch:true,allow_sampling:false,optimizer_kwargs:{},line_search:{name:strong_wolfe},convergence:null},"
            "{name:warmup_03,optimizer:Adam,lr:0.00008,epochs:20,batch_size:1024,shuffle:true,full_batch:false,allow_sampling:true,optimizer_kwargs:{},line_search:null,convergence:null},"
            "{name:adam_03,optimizer:Adam,lr:0.00035,epochs:180,batch_size:1024,shuffle:true,full_batch:false,allow_sampling:true,optimizer_kwargs:{},line_search:null,convergence:null},"
            "{name:lbfgs_03,optimizer:LBFGS,lr:1.0,epochs:50,batch_size:null,shuffle:false,full_batch:true,allow_sampling:false,optimizer_kwargs:{},line_search:{name:strong_wolfe},convergence:null},"
            "{name:warmup_04,optimizer:Adam,lr:0.00006,epochs:15,batch_size:1024,shuffle:true,full_batch:false,allow_sampling:true,optimizer_kwargs:{},line_search:null,convergence:null},"
            "{name:adam_04,optimizer:Adam,lr:0.0002,epochs:135,batch_size:1024,shuffle:true,full_batch:false,allow_sampling:true,optimizer_kwargs:{},line_search:null,convergence:null},"
            "{name:lbfgs_04,optimizer:LBFGS,lr:1.0,epochs:50,batch_size:null,shuffle:false,full_batch:true,allow_sampling:false,optimizer_kwargs:{},line_search:{name:strong_wolfe},convergence:null},"
            "{name:warmup_05,optimizer:Adam,lr:0.00004,epochs:10,batch_size:1024,shuffle:true,full_batch:false,allow_sampling:true,optimizer_kwargs:{},line_search:null,convergence:null},"
            "{name:adam_05,optimizer:Adam,lr:0.0001,epochs:90,batch_size:1024,shuffle:true,full_batch:false,allow_sampling:true,optimizer_kwargs:{},line_search:null,convergence:null},"
            "{name:warmup_tail,optimizer:Adam,lr:0.00003,epochs:10,batch_size:1024,shuffle:true,full_batch:false,allow_sampling:true,optimizer_kwargs:{},line_search:null,convergence:null},"
            "{name:ssbroyden_tail,optimizer:SSBroyden,lr:1.0,epochs:490,batch_size:null,shuffle:false,full_batch:true,allow_sampling:false,optimizer_kwargs:{},line_search:{name:strong_wolfe},convergence:null}"
            "]"
        ),
    },
}


def _init_manifest(
    *,
    run_root: str,
    experiment_id: str,
    experiment_tag: str,
    profile: str,
    budget: str,
    preset: str,
    dataset_seed: str,
    model_flag: str,
    variants: list[str],
) -> dict[str, Any]:
    return {
        "created_at_utc": utc_now_iso(),
        "updated_at_utc": utc_now_iso(),
        "run_root": run_root,
        "experiment": {
            "id": experiment_id,
            "tag": experiment_tag,
            "profile": profile,
            "budget": budget,
            "preset": preset,
            "dataset_seed": dataset_seed,
            "model_flag": model_flag,
            "variants": list(variants),
        },
        "stages": {},
        "artifacts": {},
        "runs": [],
    }


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _tag_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _format_hydra_list(items: list[str]) -> str:
    return "[" + ",".join(items) + "]"


def _parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_variants(raw: str | None, *, profile_variants: list[str]) -> list[str]:
    variants = profile_variants if raw in (None, "") else _parse_csv_list(raw)
    if not variants:
        raise ValueError("At least one collocation variant must be selected.")
    unsupported = [item for item in variants if item not in SUPPORTED_VARIANTS]
    if unsupported:
        raise ValueError(
            f"Unsupported variant(s): {', '.join(unsupported)}. "
            f"Use one of: {', '.join(SUPPORTED_VARIANTS)}."
        )
    return variants


def _profile_config(profile: str) -> dict[str, Any]:
    profile_name = profile.strip().lower()
    if profile_name not in PROFILE_CONFIGS:
        raise ValueError(f"profile must be one of: {', '.join(PROFILE_CONFIGS.keys())}")
    return dict(PROFILE_CONFIGS[profile_name])


def _run_logged_stage(
    *,
    stage_name: str,
    command: list[str],
    log_path: Path,
    manifest: dict[str, Any],
    manifest_path: Path,
    dry_run: bool,
    extra_env: dict[str, str] | None = None,
) -> int:
    print("[collocation-comparison] command:")
    print(" ".join(command))
    set_stage_status(
        manifest,
        stage=stage_name,
        status="running" if not dry_run else "dry_run",
        command=command,
        log_file=str(log_path),
        started_at_utc=utc_now_iso(),
    )
    save_manifest(str(manifest_path), manifest)

    if dry_run:
        set_stage_status(
            manifest,
            stage=stage_name,
            status="dry_run",
            completed_at_utc=utc_now_iso(),
            return_code=0,
        )
        save_manifest(str(manifest_path), manifest)
        return 0

    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    stage_started = monotonic()
    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.run(command, cwd=REPO_ROOT, text=True, check=False, env=env, stdout=logf, stderr=subprocess.STDOUT)

    status = "completed" if proc.returncode == 0 else "failed"
    extra = {"elapsed_seconds": monotonic() - stage_started}
    set_stage_status(
        manifest,
        stage=stage_name,
        status=status,
        completed_at_utc=utc_now_iso(),
        return_code=proc.returncode,
        error=None if proc.returncode == 0 else f"Stage '{stage_name}' failed. See {log_path}",
        extra=extra,
    )
    save_manifest(str(manifest_path), manifest)
    return int(proc.returncode)


def _set_run_status(
    manifest: dict[str, Any],
    *,
    run_name: str,
    status: str,
    run_dir: str,
    variant: str,
    command: list[str] | None = None,
    log_file: str | None = None,
    started_at_utc: str | None = None,
    completed_at_utc: str | None = None,
    return_code: int | None = None,
    error: str | None = None,
    metrics_summary: dict[str, Any] | None = None,
) -> None:
    existing = None
    for item in manifest["runs"]:
        if item.get("run_name") == run_name:
            existing = item
            break
    if existing is None:
        existing = {
            "run_name": run_name,
            "run_dir": run_dir,
            "variant": variant,
        }
        manifest["runs"].append(existing)
    existing["status"] = status
    if command is not None:
        existing["command"] = command
    if log_file is not None:
        existing["log_file"] = log_file
    if started_at_utc is not None:
        existing["started_at_utc"] = started_at_utc
    if completed_at_utc is not None:
        existing["completed_at_utc"] = completed_at_utc
    if return_code is not None:
        existing["return_code"] = int(return_code)
    if error is not None:
        existing["error"] = error
    if metrics_summary is not None:
        existing["metrics_summary"] = metrics_summary


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
) -> tuple[list[str], Path]:
    dataset_run_root = run_root / "dataset_pipeline"
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
    for override in stage2_overrides:
        command.extend(["--stage2-override", override])
    return command, dataset_run_root


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


def _variant_overrides(
    *,
    variant: str,
    collocation_cfg: dict[str, Any],
) -> list[str]:
    common = [
        "pinn.collocation.sampling.enabled=false",
        f"pinn.collocation.active_points={int(collocation_cfg['active_points'])}",
        f"pinn.collocation.candidate_points={int(collocation_cfg['candidate_points'])}",
        f"pinn.collocation.append_points={int(collocation_cfg['append_points'])}",
        f"pinn.collocation.refresh_period_epochs={int(collocation_cfg['refresh_period_epochs'])}",
        f"pinn.collocation.refresh.mode={str(collocation_cfg.get('refresh_mode', 'epoch_periodic'))}",
        f"pinn.collocation.refresh.on_phase_start={_format_hydra_list([str(x) for x in collocation_cfg.get('refresh_on_phase_start', [])])}",
        f"pinn.collocation.refresh.on_phase_end={_format_hydra_list([str(x) for x in collocation_cfg.get('refresh_on_phase_end', [])])}",
        f"pinn.collocation.sampler={str(collocation_cfg['sampler'])}",
        f"pinn.collocation.score_norm={str(collocation_cfg['score_norm'])}",
    ]
    if collocation_cfg.get("initial_points") not in (None, "null"):
        common.append(f"pinn.collocation.initial_points={int(collocation_cfg['initial_points'])}")
    common.extend(_multipool_overrides(collocation_cfg))

    if variant == "static_preprocessed":
        return [
            *common,
            "pinn.collocation.mode=preprocessed",
            "pinn.collocation.strategy=static",
        ]
    if variant == "static_generated":
        return [
            *common,
            "pinn.collocation.mode=generated",
            "pinn.collocation.strategy=static",
        ]
    if variant == "random_r":
        return [
            *common,
            "pinn.collocation.mode=generated",
            "pinn.collocation.strategy=random_r",
        ]
    if variant == "rad":
        return [
            *common,
            "pinn.collocation.mode=generated",
            "pinn.collocation.strategy=rad",
            f"pinn.collocation.rad.k={float(collocation_cfg['rad_k'])}",
            f"pinn.collocation.rad.c={float(collocation_cfg['rad_c'])}",
        ]
    if variant == "rar_d":
        return [
            *common,
            "pinn.collocation.mode=generated",
            "pinn.collocation.strategy=rar_d",
            f"pinn.collocation.rad.k={float(collocation_cfg['rar_d_k'])}",
            f"pinn.collocation.rad.c={float(collocation_cfg['rar_d_c'])}",
        ]
    if variant == "rar_g":
        return [
            *common,
            "pinn.collocation.mode=generated",
            "pinn.collocation.strategy=rar_g",
        ]
    raise ValueError(f"Unsupported variant: {variant}")


def _multipool_overrides(collocation_cfg: dict[str, Any]) -> list[str]:
    multipool_cfg = dict(collocation_cfg.get("multi_pool") or {})
    if not multipool_cfg:
        return []
    overrides = [
        f"pinn.collocation.multi_pool.enabled={'true' if bool(multipool_cfg.get('enabled', False)) else 'false'}",
        f"pinn.collocation.multi_pool.total_active_points={int(multipool_cfg.get('total_active_points', collocation_cfg['active_points']))}",
        f"pinn.collocation.multi_pool.allocation.enabled={'true' if bool(multipool_cfg.get('allocation_enabled', False)) else 'false'}",
        f"pinn.collocation.multi_pool.allocation.method={str(multipool_cfg.get('allocation_method', 'loss_ratio'))}",
        f"pinn.collocation.multi_pool.allocation.update_interval_epochs={int(multipool_cfg.get('allocation_update_interval_epochs', 1))}",
        f"pinn.collocation.multi_pool.allocation.smoothing={float(multipool_cfg.get('allocation_smoothing', 0.8))}",
        f"pinn.collocation.multi_pool.pools.residual.initial_fraction={float(multipool_cfg.get('residual_initial_fraction', 0.7))}",
        f"pinn.collocation.multi_pool.pools.residual.min_fraction={float(multipool_cfg.get('residual_min_fraction', 0.4))}",
        f"pinn.collocation.multi_pool.pools.residual.max_fraction={float(multipool_cfg.get('residual_max_fraction', 0.9))}",
        f"pinn.collocation.multi_pool.pools.ic_constraint.initial_fraction={float(multipool_cfg.get('ic_initial_fraction', 0.3))}",
        f"pinn.collocation.multi_pool.pools.ic_constraint.min_fraction={float(multipool_cfg.get('ic_min_fraction', 0.1))}",
        f"pinn.collocation.multi_pool.pools.ic_constraint.max_fraction={float(multipool_cfg.get('ic_max_fraction', 0.6))}",
    ]
    return overrides


def _build_pinn_command(
    *,
    python_bin: str,
    model_flag: str,
    dataset_root: Path,
    run_dir: Path,
    seed: int,
    device: str,
    hidden_dim: int,
    hidden_layers: int,
    activation: str,
    dtype_name: str,
    epochs: int,
    batch_size: int,
    adam_lr: float,
    optimizer_phases_override: str | None,
    log_every_epoch: int,
    gradient_telemetry: bool,
    variant: str,
    collocation_cfg: dict[str, Any],
    wandb_use: bool,
    wandb_project: str,
    wandb_group: str,
    wandb_name: str,
    wandb_entity: str | None,
    wandb_tags: list[str],
) -> list[str]:
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
        "pinn.weighting.scheme=static",
        "pinn.supervised_sampling.enabled=false",
        f"pinn.gradient_telemetry.enabled={'true' if gradient_telemetry else 'false'}",
        f"logging.log_every_epoch={int(log_every_epoch)}",
        f"wandb.use={'true' if wandb_use else 'false'}",
        *_variant_overrides(variant=variant, collocation_cfg=collocation_cfg),
    ]
    if optimizer_phases_override not in (None, ""):
        command.append(str(optimizer_phases_override))
    if wandb_use:
        command.extend(
            [
                f"wandb.project={wandb_project}",
                f"wandb.group={wandb_group}",
                f"wandb.name={wandb_name}",
                f"wandb.tags={_format_hydra_list(wandb_tags)}",
            ]
        )
        if wandb_entity:
            command.append(f"wandb.entity={wandb_entity}")
    return command


def _load_run_metrics_summary(run_dir: Path) -> dict[str, Any] | None:
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        return None
    with metrics_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    last = rows[-1]

    def _float_or_none(key: str) -> float | None:
        raw = last.get(key)
        if raw in (None, "", "None"):
            return None
        return float(raw)

    def _int_or_none(key: str) -> int | None:
        raw = last.get(key)
        if raw in (None, "", "None"):
            return None
        return int(float(raw))

    return {
        "global_epoch": _int_or_none("global_epoch"),
        "phase_name": last.get("phase_name"),
        "train_total_loss": _float_or_none("train_total_loss"),
        "train_physics_loss": _float_or_none("train_physics_loss"),
        "val_total_loss": _float_or_none("val_total_loss"),
        "val_physics_loss": _float_or_none("val_physics_loss"),
        "epoch_wall_seconds": _float_or_none("epoch_wall_seconds"),
        "cumulative_wall_seconds": _float_or_none("cumulative_wall_seconds"),
        "num_collocation_rows": _int_or_none("num_collocation_rows"),
        "optimizer_multipool_enabled": last.get("optimizer_multipool_enabled"),
        "optimizer_multipool_allocation_step": _int_or_none("optimizer_multipool_allocation_step"),
        "optimizer_multipool_residual_target_rows": _int_or_none("optimizer_multipool_residual_target_rows"),
        "optimizer_multipool_residual_epoch_rows": _int_or_none("optimizer_multipool_residual_epoch_rows"),
        "optimizer_multipool_ic_target_rows": _int_or_none("optimizer_multipool_ic_target_rows"),
        "optimizer_multipool_ic_epoch_rows": _int_or_none("optimizer_multipool_ic_epoch_rows"),
    }


def _write_summary_csv(*, output_root: Path, manifest: dict[str, Any]) -> Path:
    summary_path = output_root / "summary.csv"
    fieldnames = [
        "run_name",
        "variant",
        "status",
        "run_dir",
        "global_epoch",
        "train_total_loss",
        "train_physics_loss",
        "val_total_loss",
        "val_physics_loss",
        "cumulative_wall_seconds",
        "num_collocation_rows",
        "multipool_enabled",
        "multipool_allocation_step",
        "multipool_residual_target_rows",
        "multipool_residual_epoch_rows",
        "multipool_ic_target_rows",
        "multipool_ic_epoch_rows",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for run in manifest.get("runs", []):
            metrics = dict(run.get("metrics_summary", {}) or {})
            writer.writerow(
                {
                    "run_name": run.get("run_name"),
                    "variant": run.get("variant"),
                    "status": run.get("status"),
                    "run_dir": run.get("run_dir"),
                    "global_epoch": metrics.get("global_epoch"),
                    "train_total_loss": metrics.get("train_total_loss"),
                    "train_physics_loss": metrics.get("train_physics_loss"),
                    "val_total_loss": metrics.get("val_total_loss"),
                    "val_physics_loss": metrics.get("val_physics_loss"),
                    "cumulative_wall_seconds": metrics.get("cumulative_wall_seconds"),
                    "num_collocation_rows": metrics.get("num_collocation_rows"),
                    "multipool_enabled": metrics.get("optimizer_multipool_enabled"),
                    "multipool_allocation_step": metrics.get("optimizer_multipool_allocation_step"),
                    "multipool_residual_target_rows": metrics.get("optimizer_multipool_residual_target_rows"),
                    "multipool_residual_epoch_rows": metrics.get("optimizer_multipool_residual_epoch_rows"),
                    "multipool_ic_target_rows": metrics.get("optimizer_multipool_ic_target_rows"),
                    "multipool_ic_epoch_rows": metrics.get("optimizer_multipool_ic_epoch_rows"),
                }
            )
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-bin", default=sys.executable, help="Python executable for pipeline and training.")
    parser.add_argument("--profile", default="smoke", choices=sorted(PROFILE_CONFIGS.keys()), help="Run a predefined collocation comparison profile.")
    parser.add_argument("--experiment-tag", default=None, help="Output/W&B tag suffix. Default: timestamp.")
    parser.add_argument("--output-root", default=None, help="Explicit output root. Default: outputs/pinn/collocation_comparison/<tag>.")
    parser.add_argument("--model-flag", default="SM4", help="Model flag.")
    parser.add_argument("--preset", default=None, help="Dataset pipeline preset. Defaults from --profile.")
    parser.add_argument("--budget", default=None, help="Budget label for QBC dataset generation. Defaults from --profile.")
    parser.add_argument("--dataset-seed", default="s01", help="Dataset seed label from the registry.")
    parser.add_argument("--dataset-root", default=None, help="Optional explicit preprocessed dataset root. Skips dataset generation when set.")
    parser.add_argument("--variants", default=None, help=f"Comma-separated collocation variants. Default depends on --profile. Choices: {', '.join(SUPPORTED_VARIANTS)}.")
    parser.add_argument("--seed", type=int, default=37, help="Training seed.")
    parser.add_argument("--device", default=None, help="PINN device override. Defaults from --profile.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="PINN hidden width.")
    parser.add_argument("--hidden-layers", type=int, default=4, help="PINN hidden depth.")
    parser.add_argument("--activation", default="tanh", help="PINN activation.")
    parser.add_argument("--dtype", default="float64", help="PINN dtype.")
    parser.add_argument("--epochs", type=int, default=None, help="Adam epochs. Defaults from --profile.")
    parser.add_argument("--batch-size", type=int, default=None, help="Adam batch size. Defaults from --profile.")
    parser.add_argument("--adam-lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument(
        "--optimizer-phases-override",
        default=None,
        help=(
            "Raw Hydra override for pinn.optimizer_phases. "
            "When omitted, the pipeline uses the selected profile's configured optimizer phases or the base PINN config."
        ),
    )
    parser.add_argument("--active-points", type=int, default=None, help="Target/final collocation budget.")
    parser.add_argument("--initial-points", type=int, default=None, help="Initial collocation budget for append-based strategies.")
    parser.add_argument("--candidate-points", type=int, default=None, help="Candidate pool size for adaptive strategies.")
    parser.add_argument("--append-points", type=int, default=None, help="Points added at each append-based refresh.")
    parser.add_argument("--refresh-period-epochs", type=int, default=None, help="Refresh cadence for adaptive strategies.")
    parser.add_argument(
        "--refresh-mode",
        default=None,
        choices=["epoch_periodic", "phase_boundary"],
        help="Collocation refresh policy.",
    )
    parser.add_argument(
        "--refresh-on-phase-start",
        default=None,
        help="Comma-separated phase names that trigger a collocation refresh at phase start.",
    )
    parser.add_argument(
        "--refresh-on-phase-end",
        default=None,
        help="Comma-separated phase names that trigger a collocation refresh at phase end.",
    )
    parser.add_argument("--sampler", default=None, choices=["random", "lhs", "sobol"], help="Train-time collocation sampler.")
    parser.add_argument("--score-norm", default=None, choices=["l1", "l2", "linf"], help="Residual norm for adaptive scoring.")
    parser.add_argument("--rad-k", type=float, default=None, help="RAD exponent k.")
    parser.add_argument("--rad-c", type=float, default=None, help="RAD additive floor c.")
    parser.add_argument("--rar-d-k", type=float, default=None, help="RAR-D exponent k.")
    parser.add_argument("--rar-d-c", type=float, default=None, help="RAR-D additive floor c.")
    parser.add_argument("--multipool", action=argparse.BooleanOptionalAction, default=None, help="Enable residual+IC multi-pool collocation.")
    parser.add_argument("--multipool-allocation", action=argparse.BooleanOptionalAction, default=None, help="Enable dynamic multi-pool budget allocation.")
    parser.add_argument("--multipool-total-active-points", type=int, default=None, help="Total budget shared across the residual and IC pools.")
    parser.add_argument("--multipool-allocation-method", default=None, help="Dynamic allocation method. Currently supports: loss_ratio.")
    parser.add_argument("--multipool-update-interval-epochs", type=int, default=None, help="Epoch interval between multi-pool allocation updates.")
    parser.add_argument("--multipool-smoothing", type=float, default=None, help="EMA smoothing factor for dynamic multi-pool allocation.")
    parser.add_argument("--residual-initial-fraction", type=float, default=None, help="Initial residual-pool budget fraction.")
    parser.add_argument("--residual-min-fraction", type=float, default=None, help="Minimum residual-pool budget fraction.")
    parser.add_argument("--residual-max-fraction", type=float, default=None, help="Maximum residual-pool budget fraction.")
    parser.add_argument("--ic-initial-fraction", type=float, default=None, help="Initial IC-pool budget fraction.")
    parser.add_argument("--ic-min-fraction", type=float, default=None, help="Minimum IC-pool budget fraction.")
    parser.add_argument("--ic-max-fraction", type=float, default=None, help="Maximum IC-pool budget fraction.")
    parser.add_argument("--wandb-use", action=argparse.BooleanOptionalAction, default=False, help="Enable W&B logging for the sweep.")
    parser.add_argument("--wandb-project", default="sm-surrogates-pinn-collocation-comparison", help="Dedicated W&B project for this comparison.")
    parser.add_argument("--wandb-entity", default=None, help="Optional W&B entity.")
    parser.add_argument("--log-every-epoch", type=int, default=1, help="Metric/W&B logging cadence.")
    parser.add_argument(
        "--gradient-telemetry",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable expensive gradient telemetry. Defaults from --profile.",
    )
    parser.add_argument("--tag", action="append", default=[], help="Optional extra W&B tag. Can be passed multiple times.")
    parser.add_argument("--stage1-override", action="append", default=[], help="Extra stage-1 dataset-generation override.")
    parser.add_argument("--stage2-override", action="append", default=[], help="Extra stage-2 preprocess override.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    profile_config = _profile_config(args.profile)
    variants = _parse_variants(args.variants, profile_variants=list(profile_config["variants"]))
    resolved_budget = args.budget or str(profile_config["budget"])
    resolved_preset = args.preset or str(profile_config["preset"])
    resolved_epochs = int(profile_config["epochs"] if args.epochs is None else args.epochs)
    resolved_batch_size = int(profile_config["batch_size"] if args.batch_size is None else args.batch_size)
    resolved_device = str(profile_config["device"] if args.device is None else args.device)
    resolved_gradient_telemetry = (
        bool(profile_config["gradient_telemetry"]) if args.gradient_telemetry is None else bool(args.gradient_telemetry)
    )
    resolved_optimizer_phases_override = (
        args.optimizer_phases_override
        if args.optimizer_phases_override is not None
        else profile_config.get("optimizer_phases_override")
    )
    resolved_stage1_overrides = [*list(profile_config["stage1_overrides"]), *list(args.stage1_override)]
    resolved_stage2_overrides = [*list(profile_config["stage2_overrides"]), *list(args.stage2_override)]
    base_collocation = dict(profile_config["collocation"])
    if args.active_points is not None:
        base_collocation["active_points"] = int(args.active_points)
    if args.initial_points is not None:
        base_collocation["initial_points"] = int(args.initial_points)
    if args.candidate_points is not None:
        base_collocation["candidate_points"] = int(args.candidate_points)
    if args.append_points is not None:
        base_collocation["append_points"] = int(args.append_points)
    if args.refresh_period_epochs is not None:
        base_collocation["refresh_period_epochs"] = int(args.refresh_period_epochs)
    if args.refresh_mode is not None:
        base_collocation["refresh_mode"] = str(args.refresh_mode)
    if args.refresh_on_phase_start is not None:
        base_collocation["refresh_on_phase_start"] = _parse_csv_list(args.refresh_on_phase_start)
    if args.refresh_on_phase_end is not None:
        base_collocation["refresh_on_phase_end"] = _parse_csv_list(args.refresh_on_phase_end)
    if args.sampler is not None:
        base_collocation["sampler"] = str(args.sampler)
    if args.score_norm is not None:
        base_collocation["score_norm"] = str(args.score_norm)
    if args.rad_k is not None:
        base_collocation["rad_k"] = float(args.rad_k)
    if args.rad_c is not None:
        base_collocation["rad_c"] = float(args.rad_c)
    if args.rar_d_k is not None:
        base_collocation["rar_d_k"] = float(args.rar_d_k)
    if args.rar_d_c is not None:
        base_collocation["rar_d_c"] = float(args.rar_d_c)
    multipool_cfg = dict(base_collocation.get("multi_pool") or {})
    if args.multipool is not None:
        multipool_cfg["enabled"] = bool(args.multipool)
    if args.multipool_allocation is not None:
        multipool_cfg["allocation_enabled"] = bool(args.multipool_allocation)
    if args.multipool_total_active_points is not None:
        multipool_cfg["total_active_points"] = int(args.multipool_total_active_points)
    if args.multipool_allocation_method is not None:
        multipool_cfg["allocation_method"] = str(args.multipool_allocation_method)
    if args.multipool_update_interval_epochs is not None:
        multipool_cfg["allocation_update_interval_epochs"] = int(args.multipool_update_interval_epochs)
    if args.multipool_smoothing is not None:
        multipool_cfg["allocation_smoothing"] = float(args.multipool_smoothing)
    if args.residual_initial_fraction is not None:
        multipool_cfg["residual_initial_fraction"] = float(args.residual_initial_fraction)
    if args.residual_min_fraction is not None:
        multipool_cfg["residual_min_fraction"] = float(args.residual_min_fraction)
    if args.residual_max_fraction is not None:
        multipool_cfg["residual_max_fraction"] = float(args.residual_max_fraction)
    if args.ic_initial_fraction is not None:
        multipool_cfg["ic_initial_fraction"] = float(args.ic_initial_fraction)
    if args.ic_min_fraction is not None:
        multipool_cfg["ic_min_fraction"] = float(args.ic_min_fraction)
    if args.ic_max_fraction is not None:
        multipool_cfg["ic_max_fraction"] = float(args.ic_max_fraction)
    if multipool_cfg:
        base_collocation["multi_pool"] = multipool_cfg

    stamp = args.experiment_tag or _tag_stamp()
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else (REPO_ROOT / "outputs" / "pinn" / "collocation_comparison" / stamp).resolve()
    )
    output_root.mkdir(parents=True, exist_ok=True)

    experiment_id = f"collocation_comparison_{resolved_budget}_{stamp}"
    wandb_group = f"collocation_comparison_{args.model_flag.lower()}_{stamp}"
    manifest_path = output_root / "run_manifest.json"
    manifest = _init_manifest(
        run_root=str(output_root),
        experiment_id=experiment_id,
        experiment_tag=stamp,
        profile=args.profile,
        budget=resolved_budget,
        preset=resolved_preset,
        dataset_seed=args.dataset_seed,
        model_flag=args.model_flag,
        variants=variants,
    )
    manifest["artifacts"]["wandb_project"] = args.wandb_project
    manifest["artifacts"]["wandb_group"] = wandb_group
    manifest["artifacts"]["stage1_overrides"] = resolved_stage1_overrides
    manifest["artifacts"]["stage2_overrides"] = resolved_stage2_overrides
    manifest["artifacts"]["variants"] = variants
    manifest["artifacts"]["collocation_config"] = base_collocation
    save_manifest(str(manifest_path), manifest)

    if args.dataset_root:
        dataset_root = Path(args.dataset_root).resolve()
        dataset_pipeline_root = None
        manifest["artifacts"]["dataset_root"] = str(dataset_root)
        manifest["artifacts"]["dataset_source"] = "provided"
        save_manifest(str(manifest_path), manifest)
    else:
        dataset_command, dataset_pipeline_root = _build_dataset_command(
            python_bin=args.python_bin,
            experiment_id=experiment_id,
            preset=resolved_preset,
            budget=resolved_budget,
            dataset_seed=args.dataset_seed,
            model_flag=args.model_flag,
            run_root=output_root,
            stage1_overrides=resolved_stage1_overrides,
            stage2_overrides=resolved_stage2_overrides,
        )
        rc = _run_logged_stage(
            stage_name="dataset_pipeline",
            command=dataset_command,
            log_path=output_root / "logs" / "dataset_pipeline.log",
            manifest=manifest,
            manifest_path=manifest_path,
            dry_run=args.dry_run,
        )
        if rc != 0:
            raise SystemExit(rc)
        dataset_root = _dataset_root_from_manifest(dataset_pipeline_root, dry_run=args.dry_run, model_flag=args.model_flag)
        manifest["artifacts"]["dataset_root"] = str(dataset_root)
        manifest["artifacts"]["dataset_pipeline_root"] = str(dataset_pipeline_root)
        manifest["artifacts"]["dataset_manifest_path"] = str(dataset_pipeline_root / "dataset_manifest.json")
        save_manifest(str(manifest_path), manifest)

    tags_base = [
        "collocation_comparison",
        args.profile,
        resolved_budget,
        args.model_flag.lower(),
        *args.tag,
    ]

    for variant in variants:
        run_name = variant
        run_dir = output_root / "runs" / run_name
        wandb_tags = [*tags_base, variant]
        command = _build_pinn_command(
            python_bin=args.python_bin,
            model_flag=args.model_flag,
            dataset_root=dataset_root,
            run_dir=run_dir,
            seed=args.seed,
            device=resolved_device,
            hidden_dim=args.hidden_dim,
            hidden_layers=args.hidden_layers,
            activation=args.activation,
            dtype_name=args.dtype,
            epochs=resolved_epochs,
            batch_size=resolved_batch_size,
            adam_lr=args.adam_lr,
            optimizer_phases_override=resolved_optimizer_phases_override,
            log_every_epoch=args.log_every_epoch,
            gradient_telemetry=resolved_gradient_telemetry,
            variant=variant,
            collocation_cfg=base_collocation,
            wandb_use=bool(args.wandb_use),
            wandb_project=args.wandb_project,
            wandb_group=wandb_group,
            wandb_name=run_name,
            wandb_entity=args.wandb_entity,
            wandb_tags=wandb_tags,
        )
        log_path = output_root / "logs" / "runs" / f"{run_name}.log"
        _set_run_status(
            manifest,
            run_name=run_name,
            status="running" if not args.dry_run else "dry_run",
            run_dir=str(run_dir),
            variant=variant,
            command=command,
            log_file=str(log_path),
            started_at_utc=utc_now_iso(),
        )
        save_manifest(str(manifest_path), manifest)

        if args.dry_run:
            print("[collocation-comparison] command:")
            print(" ".join(command))
            _set_run_status(
                manifest,
                run_name=run_name,
                status="dry_run",
                run_dir=str(run_dir),
                variant=variant,
                completed_at_utc=utc_now_iso(),
                return_code=0,
            )
            save_manifest(str(manifest_path), manifest)
            continue

        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as logf:
            proc = subprocess.run(command, cwd=REPO_ROOT, text=True, check=False, stdout=logf, stderr=subprocess.STDOUT)
        metrics_summary = _load_run_metrics_summary(run_dir)
        _set_run_status(
            manifest,
            run_name=run_name,
            status="completed" if proc.returncode == 0 else "failed",
            run_dir=str(run_dir),
            variant=variant,
            completed_at_utc=utc_now_iso(),
            return_code=proc.returncode,
            error=None if proc.returncode == 0 else f"Run '{run_name}' failed. See {log_path}",
            metrics_summary=metrics_summary,
        )
        save_manifest(str(manifest_path), manifest)
        if proc.returncode != 0:
            raise SystemExit(proc.returncode)

    summary_path = _write_summary_csv(output_root=output_root, manifest=manifest)
    manifest["artifacts"]["summary_csv"] = str(summary_path)
    save_manifest(str(manifest_path), manifest)
    print(f"[collocation-comparison] run_manifest={manifest_path}")
    print(f"[collocation-comparison] summary_csv={summary_path}")


if __name__ == "__main__":
    main()
