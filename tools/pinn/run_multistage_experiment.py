#!/usr/bin/env python3
"""Run fixed multistage PINN experiments on one shared QBC dataset.

This launcher creates one dataset through the normal experiment pipeline and then
reuses the preprocessed dataset root for one or more explicit multistage PINN runs.
It is intended for experiments where the stage sequence itself is the subject of
the study, for example:

1. Adam -> LBFGS -> Adam -> BFGS
2. Adam -> SSBFGS -> Adam -> SSBroyden
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_OPTIMIZERS = ("Adam", "LBFGS", "BFGS", "SSBFGS", "SSBroyden")
PROFILE_CONFIGS: dict[str, dict[str, Any]] = {
    "smoke": {
        "budget": "b256",
        "preset": "default",
        "stage_sequences": [
            "Adam:20;LBFGS:10;Adam:20;BFGS:10",
        ],
        "stage1_overrides": [],
        "stage2_overrides": ["time=0.05", "num_of_points=20"],
        "gradient_telemetry": True,
        "log_every_epoch": 5,
    },
    "benchmark": {
        "budget": "b256",
        "preset": "default",
        "stage_sequences": [
            "Adam:300;LBFGS:100",
            "Adam:300;LBFGS:100;Adam:300;LBFGS:100",
            "Adam:3000;LBFGS:100",
            "Adam:3000;LBFGS:100;Adam:300;LBFGS:100",
            "LBFGS:500",
            "Adam:3000",
        ],
        "stage1_overrides": [],
        "stage2_overrides": ["time=0.05", "num_of_points=20"],
        "gradient_telemetry": True,
        "log_every_epoch": 5,
    },
}


@dataclass(frozen=True)
class LauncherStageSpec:
    optimizer: str
    epochs: int
    lr: float
    batch_size: int | None
    shuffle: bool
    full_batch: bool
    allow_sampling: bool
    optimizer_kwargs: str
    line_search_name: str | None

    @property
    def short_name(self) -> str:
        return self.optimizer.lower()


def _run(command: list[str], *, dry_run: bool, extra_env: dict[str, str] | None = None) -> None:
    print("[multistage-experiment] command:")
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


def _profile_config(profile: str) -> dict[str, Any]:
    profile_name = profile.strip().lower()
    if profile_name not in PROFILE_CONFIGS:
        raise ValueError("profile must be one of: smoke, benchmark")
    return dict(PROFILE_CONFIGS[profile_name])


def _parse_stage_sequences(raw_values: list[str]) -> list[str]:
    parsed = [item.strip() for item in raw_values if item.strip()]
    if not parsed:
        raise ValueError("At least one stage sequence must be configured.")
    return parsed


def _optimizer_defaults(
    *,
    optimizer: str,
    line_search_name: str,
) -> tuple[str, str | None]:
    optimizer_kwargs = "{}"
    resolved_line_search_name: str | None = None
    if optimizer == "Adam":
        return optimizer_kwargs, None
    resolved_line_search_name = line_search_name
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
        resolved_line_search_name = "strong_wolfe"
    return optimizer_kwargs, resolved_line_search_name


def _parse_stage_sequence(
    raw: str,
    *,
    adam_lr: float,
    quasi_newton_lr: float,
    batch_size: int,
    line_search_name: str,
) -> list[LauncherStageSpec]:
    parts = [item.strip() for item in raw.split(";") if item.strip()]
    if not parts:
        raise ValueError("Each --stage-sequence must include at least one stage.")

    stages: list[LauncherStageSpec] = []
    for part in parts:
        fields = [item.strip() for item in part.split(":")]
        if len(fields) not in {2, 3}:
            raise ValueError(
                "Stage sequence entries must use OPTIMIZER:EPOCHS or OPTIMIZER:EPOCHS:LR, "
                f"got '{part}'."
            )
        optimizer = fields[0]
        if optimizer not in SUPPORTED_OPTIMIZERS:
            raise ValueError(
                f"Unsupported stage optimizer '{optimizer}'. "
                f"Use one of: {', '.join(SUPPORTED_OPTIMIZERS)}."
            )
        epochs = int(fields[1])
        if epochs <= 0:
            raise ValueError(f"Stage '{part}' must set epochs > 0.")
        lr = float(fields[2]) if len(fields) == 3 else (adam_lr if optimizer == "Adam" else quasi_newton_lr)
        optimizer_kwargs, resolved_line_search_name = _optimizer_defaults(
            optimizer=optimizer,
            line_search_name=line_search_name,
        )
        is_adam = optimizer == "Adam"
        stages.append(
            LauncherStageSpec(
                optimizer=optimizer,
                epochs=epochs,
                lr=lr,
                batch_size=int(batch_size) if is_adam else None,
                shuffle=is_adam,
                full_batch=not is_adam,
                allow_sampling=False,
                optimizer_kwargs=optimizer_kwargs,
                line_search_name=resolved_line_search_name if not is_adam else None,
            )
        )
    return stages


def _serialize_stage(stage: LauncherStageSpec, *, index: int) -> str:
    lower = stage.optimizer.lower()
    return (
        "{"
        f"name:{lower}_{int(index):02d},"
        f"optimizer:{stage.optimizer},"
        f"lr:{stage.lr},"
        f"epochs:{int(stage.epochs)},"
        f"batch_size:{'null' if stage.batch_size is None else int(stage.batch_size)},"
        f"shuffle:{'true' if stage.shuffle else 'false'},"
        f"full_batch:{'true' if stage.full_batch else 'false'},"
        f"allow_sampling:{'true' if stage.allow_sampling else 'false'},"
        f"optimizer_kwargs:{stage.optimizer_kwargs},"
        f"line_search:{'null' if stage.line_search_name is None else '{name:' + stage.line_search_name + '}'},"  # noqa: E501
        "convergence:null"
        "}"
    )


def _stage_override(stages: list[LauncherStageSpec]) -> str:
    return "pinn.stages=[" + ",".join(_serialize_stage(stage, index=idx) for idx, stage in enumerate(stages)) + "]"


def _sequence_slug(stages: list[LauncherStageSpec]) -> str:
    return "_".join(f"{stage.short_name}{int(stage.epochs)}" for stage in stages)


def _sequence_tag(stages: list[LauncherStageSpec]) -> str:
    slug = _sequence_slug(stages)
    safe = re.sub(r"[^a-z0-9_]+", "_", slug.lower()).strip("_")
    return safe or "sequence"


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
    stages: list[LauncherStageSpec],
    log_every_epoch: int,
    loss_weight_data: float,
    loss_weight_dt: float,
    loss_weight_physics: float,
    loss_weight_ic: float,
    gradient_telemetry: bool,
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
        _stage_override(stages),
    ]
    if wandb_entity:
        command.append(f"wandb.entity={wandb_entity}")
    return command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-bin", default=sys.executable, help="Python executable for pipeline and training.")
    parser.add_argument("--profile", default="benchmark", choices=["smoke", "benchmark"], help="Run a reduced multistage smoke setup or the longer benchmark variant.")
    parser.add_argument("--experiment-tag", default=None, help="Output/W&B tag suffix. Default: timestamp.")
    parser.add_argument("--output-root", default=None, help="Explicit output root. Default: outputs/pinn/multistage_experiment/<tag>.")
    parser.add_argument("--model-flag", default="SM4", help="Model flag.")
    parser.add_argument("--preset", default=None, help="Dataset pipeline preset. Defaults from --profile.")
    parser.add_argument("--budget", default=None, help="Budget label for QBC dataset generation. Defaults from --profile.")
    parser.add_argument("--dataset-seed", default="s01", help="Dataset seed label from the registry.")
    parser.add_argument("--dataset-root", default=None, help="Optional explicit preprocessed dataset root. Skips dataset generation when set.")
    parser.add_argument("--seed", type=int, default=37, help="Training seed.")
    parser.add_argument("--device", default="cuda", help="PINN device override.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="PINN hidden width.")
    parser.add_argument("--hidden-layers", type=int, default=4, help="PINN hidden depth.")
    parser.add_argument("--activation", default="tanh", help="PINN activation.")
    parser.add_argument("--dtype", default="float64", help="PINN dtype.")
    parser.add_argument("--batch-size", type=int, default=1024, help="Adam stage batch size.")
    parser.add_argument("--adam-lr", type=float, default=1e-3, help="Default learning rate for Adam stages.")
    parser.add_argument("--quasi-newton-lr", type=float, default=1.0, help="Default learning rate for quasi-Newton stages.")
    parser.add_argument("--line-search", default="strong_wolfe", choices=["strong_wolfe", "backtracking"], help="Line-search method for BFGS-family stages.")
    parser.add_argument(
        "--stage-sequence",
        action="append",
        default=[],
        help=(
            "Explicit multistage sequence using 'OPTIMIZER:EPOCHS[:LR]' entries separated by ';'. "
            "Example: 'Adam:100;LBFGS:100;Adam:100;BFGS:100'. Can be passed multiple times."
        ),
    )
    parser.add_argument("--wandb-project", default="sm-surrogates-pinn-multistage", help="Dedicated W&B project for this experiment.")
    parser.add_argument("--wandb-entity", default=None, help="Optional W&B entity.")
    parser.add_argument("--loss-weight-data", type=float, default=1.0, help="Static supervised loss weight.")
    parser.add_argument("--loss-weight-dt", type=float, default=1.0e-4, help="Static dt loss weight.")
    parser.add_argument("--loss-weight-physics", type=float, default=1.0e-4, help="Static physics loss weight.")
    parser.add_argument("--loss-weight-ic", type=float, default=1.0e-3, help="Static IC loss weight.")
    parser.add_argument(
        "--gradient-telemetry",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable expensive gradient telemetry. Defaults from --profile.",
    )
    parser.add_argument("--log-every-epoch", type=int, default=None, help="Metric/W&B logging cadence. Defaults from --profile.")
    parser.add_argument("--tag", action="append", default=[], help="Optional extra W&B tag. Can be passed multiple times.")
    parser.add_argument("--stage1-override", action="append", default=[], help="Extra stage-1 dataset-generation override.")
    parser.add_argument("--stage2-override", action="append", default=[], help="Extra stage-2 preprocess override.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    profile_config = _profile_config(args.profile)
    resolved_stage_sequences = (
        _parse_stage_sequences(args.stage_sequence)
        if args.stage_sequence
        else _parse_stage_sequences(list(profile_config["stage_sequences"]))
    )
    resolved_budget = args.budget or str(profile_config["budget"])
    resolved_preset = args.preset or str(profile_config["preset"])
    resolved_stage1_overrides = [*list(profile_config["stage1_overrides"]), *list(args.stage1_override)]
    resolved_stage2_overrides = [*list(profile_config["stage2_overrides"]), *list(args.stage2_override)]
    resolved_gradient_telemetry = (
        bool(profile_config["gradient_telemetry"]) if args.gradient_telemetry is None else args.gradient_telemetry
    )
    resolved_log_every_epoch = (
        int(profile_config["log_every_epoch"]) if args.log_every_epoch is None else int(args.log_every_epoch)
    )

    stamp = args.experiment_tag or _tag_stamp()
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else (REPO_ROOT / "outputs" / "pinn" / "multistage_experiment" / stamp).resolve()
    )
    output_root.mkdir(parents=True, exist_ok=True)

    experiment_id = f"multistage_experiment_{resolved_budget}_{stamp}"
    wandb_group = f"multistage_experiment_{args.model_flag.lower()}_{stamp}"

    if args.dataset_root:
        dataset_root = Path(args.dataset_root).resolve()
        dataset_pipeline_root = None
    else:
        dataset_pipeline_root = output_root / "dataset_pipeline"
        _build_dataset_command(
            python_bin=args.python_bin,
            experiment_id=experiment_id,
            preset=resolved_preset,
            budget=resolved_budget,
            dataset_seed=args.dataset_seed,
            model_flag=args.model_flag,
            run_root=output_root,
            stage1_overrides=resolved_stage1_overrides,
            stage2_overrides=resolved_stage2_overrides,
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
        "budget": resolved_budget,
        "preset": resolved_preset,
        "dataset_seed": args.dataset_seed,
        "model_flag": args.model_flag,
        "stage1_overrides": resolved_stage1_overrides,
        "stage2_overrides": resolved_stage2_overrides,
        "gradient_telemetry": resolved_gradient_telemetry,
        "log_every_epoch": resolved_log_every_epoch,
        "dataset_root": str(dataset_root),
        "dataset_pipeline_root": None if dataset_pipeline_root is None else str(dataset_pipeline_root),
        "stage_sequences": resolved_stage_sequences,
        "output_root": str(output_root),
        "runs": [],
    }

    tags_base = [
        "multistage_experiment",
        args.profile,
        "qbc_deep_ensemble",
        resolved_budget,
        args.model_flag.lower(),
        *args.tag,
    ]

    for raw_sequence in resolved_stage_sequences:
        stages = _parse_stage_sequence(
            raw_sequence,
            adam_lr=args.adam_lr,
            quasi_newton_lr=args.quasi_newton_lr,
            batch_size=args.batch_size,
            line_search_name=args.line_search,
        )
        sequence_slug = _sequence_slug(stages)
        run_name = f"multistage_{sequence_slug}"
        run_dir = output_root / "runs" / run_name
        wandb_tags = [*tags_base, f"sequence_{_sequence_tag(stages)}"]
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
            stages=stages,
            log_every_epoch=resolved_log_every_epoch,
            loss_weight_data=args.loss_weight_data,
            loss_weight_dt=args.loss_weight_dt,
            loss_weight_physics=args.loss_weight_physics,
            loss_weight_ic=args.loss_weight_ic,
            gradient_telemetry=resolved_gradient_telemetry,
        )
        _run(command, dry_run=args.dry_run)
        summary["runs"].append(
            {
                "run_name": run_name,
                "run_dir": str(run_dir),
                "stage_sequence": raw_sequence,
                "stages": [
                    {"optimizer": stage.optimizer, "epochs": int(stage.epochs), "lr": float(stage.lr)}
                    for stage in stages
                ],
            }
        )

    summary_path = output_root / "experiment_manifest.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[multistage-experiment] summary_manifest={summary_path}")


if __name__ == "__main__":
    main()
