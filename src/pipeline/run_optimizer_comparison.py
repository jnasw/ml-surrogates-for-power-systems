"""Pipeline-style optimizer comparison for PINN quasi-Newton phases.

This workflow:
1. Optionally generates one shared preprocessed dataset via the existing dataset
   experiment pipeline.
2. Launches a matrix of PINN optimizer-comparison runs against that dataset.
3. Writes a run manifest plus per-stage/per-run log files suitable for local or
   HPC execution.
"""

from __future__ import annotations

import argparse
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
SUPPORTED_OPTIMIZERS = ("LBFGS", "BFGS", "SSBFGS", "SSBroyden")
PROFILE_CONFIGS: dict[str, dict[str, Any]] = {
    "smoke": {
        "budget": "b256",
        "preset": "default",
        "warmup_epochs": [0, 100],
        "quasi_newton_epochs": [20],
        "stage1_overrides": [],
        "stage2_overrides": ["time=0.05", "num_of_points=20"],
        "gradient_telemetry": False,
    },
    "benchmark": {
        "budget": "b256",
        "preset": "default",
        "warmup_epochs": [0, 100, 300, 10000],
        "quasi_newton_epochs": [100],
        "stage1_overrides": [],
        "stage2_overrides": ["time=0.05", "num_of_points=20"],
        "gradient_telemetry": False,
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


def _parse_int_list(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def _parse_optimizers(raw: str) -> list[str]:
    optimizers = _parse_csv_list(raw)
    if not optimizers:
        raise ValueError("--optimizers must include at least one optimizer.")
    unsupported = [item for item in optimizers if item not in SUPPORTED_OPTIMIZERS]
    if unsupported:
        raise ValueError(
            f"Unsupported optimizer(s): {', '.join(unsupported)}. "
            f"Use one of: {', '.join(SUPPORTED_OPTIMIZERS)}."
        )
    return optimizers


def _profile_config(profile: str) -> dict[str, Any]:
    profile_name = profile.strip().lower()
    if profile_name not in PROFILE_CONFIGS:
        raise ValueError("profile must be one of: smoke, benchmark")
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
    print("[optimizer-comparison] command:")
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
    optimizer: str,
    adam_warmup_epochs: int,
    quasi_newton_epochs: int,
    command: list[str] | None = None,
    log_file: str | None = None,
    started_at_utc: str | None = None,
    completed_at_utc: str | None = None,
    return_code: int | None = None,
    error: str | None = None,
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
            "optimizer": optimizer,
            "adam_warmup_epochs": int(adam_warmup_epochs),
            "quasi_newton_epochs": int(quasi_newton_epochs),
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
        optimizer_kwargs = "{tau_strategy:al_baali}"
    elif optimizer == "SSBroyden":
        optimizer_kwargs = "{tau_strategy:paper_default,phi_strategy:paper_default}"
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


def _optimizer_phase_override(
    *,
    optimizer: str,
    adam_warmup_epochs: int,
    adam_lr: float,
    quasi_newton_epochs: int,
    quasi_newton_lr: float,
    batch_size: int,
    line_search_name: str,
) -> str:
    phases: list[str] = []
    if adam_warmup_epochs > 0:
        phases.append(
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
    phases.append(
        _optimizer_stage(
            optimizer=optimizer,
            epochs=quasi_newton_epochs,
            lr=quasi_newton_lr,
            line_search_name=line_search_name,
        )
    )
    return "pinn.optimizer_phases=[" + ",".join(phases) + "]"


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
    adam_warmup_epochs: int,
    adam_lr: float,
    quasi_newton_epochs: int,
    quasi_newton_lr: float,
    optimizer: str,
    line_search_name: str,
    log_every_epoch: int,
    loss_weight_data: float,
    loss_weight_dt: float,
    loss_weight_physics: float,
    loss_weight_ic: float,
    gradient_telemetry: bool,
) -> list[str]:
    phase_override = _optimizer_phase_override(
        optimizer=optimizer,
        adam_warmup_epochs=adam_warmup_epochs,
        adam_lr=adam_lr,
        quasi_newton_epochs=quasi_newton_epochs,
        quasi_newton_lr=quasi_newton_lr,
        batch_size=batch_size,
        line_search_name=line_search_name,
    )
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
        phase_override,
    ]
    if wandb_entity:
        command.append(f"wandb.entity={wandb_entity}")
    return command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-bin", default=sys.executable, help="Python executable for pipeline and training.")
    parser.add_argument("--profile", default="benchmark", choices=["smoke", "benchmark"], help="Run a reduced smoke matrix or the larger benchmark matrix.")
    parser.add_argument("--experiment-tag", default=None, help="Output/W&B tag suffix. Default: timestamp.")
    parser.add_argument("--output-root", default=None, help="Explicit output root. Default: outputs/pinn/optimizer_comparison/<tag>.")
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
    parser.add_argument("--batch-size", type=int, default=1024, help="Adam warm-up batch size.")
    parser.add_argument("--adam-lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--quasi-newton-lr", type=float, default=1.0, help="Learning rate passed to quasi-Newton optimizer phases.")
    parser.add_argument("--line-search", default="strong_wolfe", choices=["strong_wolfe", "backtracking"], help="Line-search method for BFGS-family optimizer phases.")
    parser.add_argument("--optimizers", default="LBFGS,BFGS,SSBFGS,SSBroyden", help="Comma-separated optimizer list.")
    parser.add_argument("--warmup-epochs", default=None, help="Comma-separated Adam warm-up lengths. Default depends on --profile.")
    parser.add_argument("--quasi-newton-epochs", default=None, help="Comma-separated quasi-Newton epoch counts. Default depends on --profile.")
    parser.add_argument("--wandb-project", default="sm-surrogates-pinn-optimizer-comparison", help="Dedicated W&B project for this benchmark.")
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
    parser.add_argument("--log-every-epoch", type=int, default=1, help="Metric/W&B logging cadence.")
    parser.add_argument("--tag", action="append", default=[], help="Optional extra W&B tag. Can be passed multiple times.")
    parser.add_argument("--stage1-override", action="append", default=[], help="Extra stage-1 dataset-generation override.")
    parser.add_argument("--stage2-override", action="append", default=[], help="Extra stage-2 preprocess override.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    profile_config = _profile_config(args.profile)
    warmup_default = list(profile_config["warmup_epochs"])
    qn_default = list(profile_config["quasi_newton_epochs"])
    warmup_epochs = _parse_int_list(args.warmup_epochs) if args.warmup_epochs else warmup_default
    quasi_newton_epochs = _parse_int_list(args.quasi_newton_epochs) if args.quasi_newton_epochs else qn_default
    optimizers = _parse_optimizers(args.optimizers)
    resolved_budget = args.budget or str(profile_config["budget"])
    resolved_preset = args.preset or str(profile_config["preset"])
    resolved_stage1_overrides = [*list(profile_config["stage1_overrides"]), *list(args.stage1_override)]
    resolved_stage2_overrides = [*list(profile_config["stage2_overrides"]), *list(args.stage2_override)]
    resolved_gradient_telemetry = (
        bool(profile_config["gradient_telemetry"]) if args.gradient_telemetry is None else args.gradient_telemetry
    )

    stamp = args.experiment_tag or _tag_stamp()
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else (REPO_ROOT / "outputs" / "pinn" / "optimizer_comparison" / stamp).resolve()
    )
    output_root.mkdir(parents=True, exist_ok=True)

    experiment_id = f"optimizer_comparison_{resolved_budget}_{stamp}"
    wandb_group = f"optimizer_comparison_{args.model_flag.lower()}_{stamp}"
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
    )
    manifest["artifacts"]["wandb_project"] = args.wandb_project
    manifest["artifacts"]["wandb_group"] = wandb_group
    manifest["artifacts"]["stage1_overrides"] = resolved_stage1_overrides
    manifest["artifacts"]["stage2_overrides"] = resolved_stage2_overrides
    manifest["artifacts"]["warmup_epochs"] = warmup_epochs
    manifest["artifacts"]["quasi_newton_epochs"] = quasi_newton_epochs
    manifest["artifacts"]["optimizers"] = optimizers
    manifest["artifacts"]["gradient_telemetry"] = resolved_gradient_telemetry
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
        dataset_root = _dataset_root_from_manifest(
            dataset_pipeline_root,
            dry_run=args.dry_run,
            model_flag=args.model_flag,
        )
        manifest["artifacts"]["dataset_root"] = str(dataset_root)
        manifest["artifacts"]["dataset_pipeline_root"] = str(dataset_pipeline_root)
        manifest["artifacts"]["dataset_manifest_path"] = str(dataset_pipeline_root / "dataset_manifest.json")
        save_manifest(str(manifest_path), manifest)

    tags_base = [
        "optimizer_comparison",
        args.profile,
        "qbc_deep_ensemble",
        resolved_budget,
        args.model_flag.lower(),
        *args.tag,
    ]

    for warmup in warmup_epochs:
        for qn_epochs in quasi_newton_epochs:
            for optimizer in optimizers:
                run_name = f"{optimizer.lower()}_warm{int(warmup)}_qn{int(qn_epochs)}"
                run_dir = output_root / "runs" / run_name
                wandb_tags = [*tags_base, optimizer.lower(), f"warmup{int(warmup)}", f"qn{int(qn_epochs)}"]
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
                    adam_warmup_epochs=warmup,
                    adam_lr=args.adam_lr,
                    quasi_newton_epochs=qn_epochs,
                    quasi_newton_lr=args.quasi_newton_lr,
                    optimizer=optimizer,
                    line_search_name=args.line_search,
                    log_every_epoch=args.log_every_epoch,
                    loss_weight_data=args.loss_weight_data,
                    loss_weight_dt=args.loss_weight_dt,
                    loss_weight_physics=args.loss_weight_physics,
                    loss_weight_ic=args.loss_weight_ic,
                    gradient_telemetry=resolved_gradient_telemetry,
                )
                log_path = output_root / "logs" / "runs" / f"{run_name}.log"
                _set_run_status(
                    manifest,
                    run_name=run_name,
                    status="running" if not args.dry_run else "dry_run",
                    run_dir=str(run_dir),
                    optimizer=optimizer,
                    adam_warmup_epochs=int(warmup),
                    quasi_newton_epochs=int(qn_epochs),
                    command=command,
                    log_file=str(log_path),
                    started_at_utc=utc_now_iso(),
                )
                save_manifest(str(manifest_path), manifest)

                if args.dry_run:
                    print("[optimizer-comparison] command:")
                    print(" ".join(command))
                    _set_run_status(
                        manifest,
                        run_name=run_name,
                        status="dry_run",
                        run_dir=str(run_dir),
                        optimizer=optimizer,
                        adam_warmup_epochs=int(warmup),
                        quasi_newton_epochs=int(qn_epochs),
                        completed_at_utc=utc_now_iso(),
                        return_code=0,
                    )
                    save_manifest(str(manifest_path), manifest)
                    continue

                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("w", encoding="utf-8") as logf:
                    proc = subprocess.run(command, cwd=REPO_ROOT, text=True, check=False, stdout=logf, stderr=subprocess.STDOUT)
                _set_run_status(
                    manifest,
                    run_name=run_name,
                    status="completed" if proc.returncode == 0 else "failed",
                    run_dir=str(run_dir),
                    optimizer=optimizer,
                    adam_warmup_epochs=int(warmup),
                    quasi_newton_epochs=int(qn_epochs),
                    completed_at_utc=utc_now_iso(),
                    return_code=proc.returncode,
                    error=None if proc.returncode == 0 else f"Run '{run_name}' failed. See {log_path}",
                )
                save_manifest(str(manifest_path), manifest)
                if proc.returncode != 0:
                    raise SystemExit(proc.returncode)

    print(f"[optimizer-comparison] run_manifest={manifest_path}")


if __name__ == "__main__":
    main()
