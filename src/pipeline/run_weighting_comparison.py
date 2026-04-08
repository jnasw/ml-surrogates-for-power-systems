"""Pipeline-style weighting-scheme comparison for single-stage PINN runs.

This workflow:
1. Optionally generates one shared preprocessed dataset via the existing dataset
   experiment pipeline.
2. Launches a matrix of PINN runs against that dataset where the primary
   comparison axis is the weighting scheme.
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
SUPPORTED_WEIGHTING_SCHEMES = ("static", "ma", "id", "dn", "relobralo")
PROFILE_CONFIGS: dict[str, dict[str, Any]] = {
    "smoke": {
        "method": "qbc_deep_ensemble",
        "budget": "b256",
        "preset": "smoke",
        "epochs": 500,
        "batch_size": 128,
        "stage1_overrides": [],
        "stage2_overrides": [
            "time=0.05",
            "num_of_points=20",
            "model.ic_generation_method=joint_lhs",
            "model.ic_num_samples=64",
        ],
        "gradient_telemetry": True,
        "loss_weights": {"data": 1.0, "dt": 1.0e-4, "physics": 1.0e-4, "ic": 1.0e-3},
        "probe_rows": {"data": 64, "physics": 64, "init": 64},
    },
    "benchmark": {
        "method": "qbc_deep_ensemble",
        "budget": "b256",
        "preset": "default",
        "epochs": 500,
        "batch_size": 1024,
        "stage1_overrides": [],
        "stage2_overrides": [
            "time=0.05",
            "num_of_points=20",
            "model.ic_generation_method=joint_lhs",
            "model.ic_num_samples=64",
        ],
        "gradient_telemetry": True,
        "loss_weights": {"data": 1.0, "dt": 1.0e-4, "physics": 1.0e-4, "ic": 1.0e-3},
        "probe_rows": {"data": 256, "physics": 256, "init": 256},
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
    weighting_schemes: list[str],
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
            "weighting_schemes": list(weighting_schemes),
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


def _parse_weighting_schemes(raw: str) -> list[str]:
    schemes = [item.strip().lower() for item in raw.split(",") if item.strip()]
    if not schemes:
        raise ValueError("--schemes must include at least one weighting scheme.")
    unsupported = [item for item in schemes if item not in SUPPORTED_WEIGHTING_SCHEMES]
    if unsupported:
        raise ValueError(
            f"Unsupported weighting scheme(s): {', '.join(unsupported)}. "
            f"Use one of: {', '.join(SUPPORTED_WEIGHTING_SCHEMES)}."
        )
    if len(set(schemes)) != len(schemes):
        raise ValueError("--schemes must not contain duplicates.")
    return schemes


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
    print("[weighting-comparison] command:")
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
    weighting_scheme: str,
    epochs: int,
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
            "weighting_scheme": weighting_scheme,
            "epochs": int(epochs),
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
    method: str,
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
        method,
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


def _adam_stage_override(*, epochs: int, lr: float, batch_size: int, allow_sampling: bool) -> str:
    return (
        "pinn.optimizer_phases=["
        "{"
        "name:adam,"
        "optimizer:Adam,"
        f"lr:{lr},"
        f"epochs:{int(epochs)},"
        f"batch_size:{int(batch_size)},"
        "shuffle:true,"
        "full_batch:false,"
        f"allow_sampling:{'true' if allow_sampling else 'false'},"
        "optimizer_kwargs:{},"
        "line_search:null,"
        "convergence:null"
        "}"
        "]"
    )


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
    epochs: int,
    batch_size: int,
    adam_lr: float,
    allow_sampling: bool,
    log_every_epoch: int,
    loss_weight_data: float,
    loss_weight_dt: float,
    loss_weight_physics: float,
    loss_weight_ic: float,
    gradient_telemetry: bool,
    weighting_scheme: str,
    update_interval_epochs: int,
    ema_beta: float,
    probe_data_rows: int,
    probe_physics_rows: int,
    probe_init_rows: int,
    probe_seed: int,
) -> list[str]:
    command = [
        python_bin,
        "20_run_pinn.py",
        f"model.model_flag={model_flag}",
        f"model.seed={int(seed)}",
        "pinn.mode=single_stage",
        f"dataset.root={dataset_root}",
        f"pinn.run_dir={run_dir}",
        f"pinn.device={device}",
        f"pinn.dtype={dtype_name}",
        f"pinn.hidden_dim={int(hidden_dim)}",
        f"pinn.hidden_layers={int(hidden_layers)}",
        f"pinn.activation={activation}",
        f"pinn.default_batch_size={int(batch_size)}",
        "pinn.collocation_sampling.enabled=false",
        "pinn.supervised_sampling.enabled=false",
        f"pinn.gradient_telemetry.enabled={'true' if gradient_telemetry else 'false'}",
        f"pinn.loss_weights.data={loss_weight_data}",
        f"pinn.loss_weights.dt={loss_weight_dt}",
        f"pinn.loss_weights.physics={loss_weight_physics}",
        f"pinn.loss_weights.ic={loss_weight_ic}",
        f"pinn.weighting.scheme={weighting_scheme}",
        "pinn.weighting.anchor=physics",
        f"pinn.weighting.ema_beta={ema_beta}",
        f"pinn.weighting.update_interval_epochs={int(update_interval_epochs)}",
        "pinn.weighting.dynamic_components=[data,dt,ic]",
        f"pinn.weighting.probe.data_rows={int(probe_data_rows)}",
        f"pinn.weighting.probe.physics_rows={int(probe_physics_rows)}",
        f"pinn.weighting.probe.init_rows={int(probe_init_rows)}",
        f"pinn.weighting.probe.seed={int(probe_seed)}",
        "wandb.use=true",
        f"wandb.project={wandb_project}",
        f"wandb.group={wandb_group}",
        f"wandb.name={wandb_name}",
        f"wandb.tags={_format_hydra_list(wandb_tags)}",
        f"logging.log_every_epoch={int(log_every_epoch)}",
        _adam_stage_override(epochs=epochs, lr=adam_lr, batch_size=batch_size, allow_sampling=allow_sampling),
    ]
    if wandb_entity:
        command.append(f"wandb.entity={wandb_entity}")
    return command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-bin", default=sys.executable, help="Python executable for pipeline and training.")
    parser.add_argument("--profile", default="smoke", choices=["smoke", "benchmark"], help="Run a reduced local smoke comparison or a larger benchmark-style comparison.")
    parser.add_argument("--experiment-tag", default=None, help="Output/W&B tag suffix. Default: timestamp.")
    parser.add_argument("--output-root", default=None, help="Explicit output root. Default: outputs/pinn/weighting_comparison/<tag>.")
    parser.add_argument("--model-flag", default="SM4", help="Model flag.")
    parser.add_argument("--method", default=None, help="Dataset pipeline method. Defaults from --profile.")
    parser.add_argument("--preset", default=None, help="Dataset pipeline preset. Defaults from --profile.")
    parser.add_argument("--budget", default=None, help="Budget label for dataset generation. Defaults from --profile.")
    parser.add_argument("--dataset-seed", default="ds01", help="Dataset seed label from the registry.")
    parser.add_argument("--dataset-root", default=None, help="Optional explicit preprocessed dataset root. Skips dataset generation when set.")
    parser.add_argument("--seed", type=int, default=37, help="Training seed.")
    parser.add_argument("--device", default="cuda", help="PINN device override.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="PINN hidden width.")
    parser.add_argument("--hidden-layers", type=int, default=4, help="PINN hidden depth.")
    parser.add_argument("--activation", default="tanh", help="PINN activation.")
    parser.add_argument("--dtype", default="float64", help="PINN dtype.")
    parser.add_argument("--epochs", type=int, default=None, help="Adam epochs per run. Defaults from --profile.")
    parser.add_argument("--batch-size", type=int, default=None, help="Adam batch size. Defaults from --profile.")
    parser.add_argument("--adam-lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--allow-sampling", action=argparse.BooleanOptionalAction, default=False, help="Whether the Adam stage may use configured sampling.")
    parser.add_argument("--schemes", default="static,ma,id,dn", help="Comma-separated weighting schemes.")
    parser.add_argument("--ema-beta", type=float, default=0.99, help="EMA beta used by dynamic weighting.")
    parser.add_argument("--update-interval-epochs", type=int, default=10, help="Dynamic weighting update cadence.")
    parser.add_argument("--probe-data-rows", type=int, default=None, help="Fixed supervised probe rows. Defaults from --profile.")
    parser.add_argument("--probe-physics-rows", type=int, default=None, help="Fixed physics probe rows. Defaults from --profile.")
    parser.add_argument("--probe-init-rows", type=int, default=None, help="Fixed init probe rows. Defaults from --profile.")
    parser.add_argument("--probe-seed", type=int, default=0, help="Probe subset seed.")
    parser.add_argument(
        "--wandb-project",
        default="sm-surrogates-pinn-weighting-comparison",
        help="W&B project.",
    )
    parser.add_argument("--wandb-entity", default=None, help="Optional W&B entity.")
    parser.add_argument("--log-every-epoch", type=int, default=10, help="Metric/W&B logging cadence.")
    parser.add_argument("--loss-weight-data", type=float, default=None, help="Base supervised loss weight. Defaults from --profile.")
    parser.add_argument("--loss-weight-dt", type=float, default=None, help="Base dt loss weight. Defaults from --profile.")
    parser.add_argument("--loss-weight-physics", type=float, default=None, help="Base physics loss weight. Defaults from --profile.")
    parser.add_argument("--loss-weight-ic", type=float, default=None, help="Base IC loss weight. Defaults from --profile.")
    parser.add_argument(
        "--gradient-telemetry",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable gradient telemetry. Defaults from --profile.",
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
    resolved_method = args.method or str(profile_config["method"])
    resolved_budget = args.budget or str(profile_config["budget"])
    resolved_preset = args.preset or str(profile_config["preset"])
    resolved_epochs = int(profile_config["epochs"] if args.epochs is None else args.epochs)
    resolved_batch_size = int(profile_config["batch_size"] if args.batch_size is None else args.batch_size)
    resolved_stage1_overrides = [*list(profile_config["stage1_overrides"]), *list(args.stage1_override)]
    resolved_stage2_overrides = [*list(profile_config["stage2_overrides"]), *list(args.stage2_override)]
    resolved_gradient_telemetry = (
        bool(profile_config["gradient_telemetry"]) if args.gradient_telemetry is None else args.gradient_telemetry
    )
    schemes = _parse_weighting_schemes(args.schemes)
    profile_loss_weights = dict(profile_config["loss_weights"])
    loss_weight_data = float(profile_loss_weights["data"] if args.loss_weight_data is None else args.loss_weight_data)
    loss_weight_dt = float(profile_loss_weights["dt"] if args.loss_weight_dt is None else args.loss_weight_dt)
    loss_weight_physics = float(
        profile_loss_weights["physics"] if args.loss_weight_physics is None else args.loss_weight_physics
    )
    loss_weight_ic = float(profile_loss_weights["ic"] if args.loss_weight_ic is None else args.loss_weight_ic)
    profile_probe_rows = dict(profile_config["probe_rows"])
    probe_data_rows = int(profile_probe_rows["data"] if args.probe_data_rows is None else args.probe_data_rows)
    probe_physics_rows = int(
        profile_probe_rows["physics"] if args.probe_physics_rows is None else args.probe_physics_rows
    )
    probe_init_rows = int(profile_probe_rows["init"] if args.probe_init_rows is None else args.probe_init_rows)

    stamp = args.experiment_tag or _tag_stamp()
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else (REPO_ROOT / "outputs" / "pinn" / "weighting_comparison" / stamp).resolve()
    )
    output_root.mkdir(parents=True, exist_ok=True)

    experiment_id = f"weighting_comparison_{resolved_budget}_{stamp}"
    wandb_group = f"weighting_comparison_{args.model_flag.lower()}_{stamp}"
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
        weighting_schemes=schemes,
    )
    manifest["artifacts"]["wandb_project"] = args.wandb_project
    manifest["artifacts"]["wandb_group"] = wandb_group
    manifest["artifacts"]["dataset_method"] = resolved_method
    manifest["artifacts"]["stage1_overrides"] = resolved_stage1_overrides
    manifest["artifacts"]["stage2_overrides"] = resolved_stage2_overrides
    manifest["artifacts"]["epochs"] = resolved_epochs
    manifest["artifacts"]["batch_size"] = resolved_batch_size
    manifest["artifacts"]["adam_lr"] = args.adam_lr
    manifest["artifacts"]["gradient_telemetry"] = resolved_gradient_telemetry
    manifest["artifacts"]["weighting_schemes"] = schemes
    manifest["artifacts"]["weighting_probe_rows"] = {
        "data": probe_data_rows,
        "physics": probe_physics_rows,
        "init": probe_init_rows,
        "seed": int(args.probe_seed),
    }
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
            method=resolved_method,
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
        "weighting_comparison",
        args.profile,
        resolved_method,
        resolved_budget,
        args.model_flag.lower(),
        f"adam{int(resolved_epochs)}",
        *args.tag,
    ]

    for scheme in schemes:
        run_name = f"{scheme}_adam{int(resolved_epochs)}"
        run_dir = output_root / "runs" / run_name
        wandb_tags = [*tags_base, scheme]
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
            epochs=resolved_epochs,
            batch_size=resolved_batch_size,
            adam_lr=args.adam_lr,
            allow_sampling=bool(args.allow_sampling),
            log_every_epoch=args.log_every_epoch,
            loss_weight_data=loss_weight_data,
            loss_weight_dt=loss_weight_dt,
            loss_weight_physics=loss_weight_physics,
            loss_weight_ic=loss_weight_ic,
            gradient_telemetry=resolved_gradient_telemetry,
            weighting_scheme=scheme,
            update_interval_epochs=args.update_interval_epochs,
            ema_beta=args.ema_beta,
            probe_data_rows=probe_data_rows,
            probe_physics_rows=probe_physics_rows,
            probe_init_rows=probe_init_rows,
            probe_seed=args.probe_seed,
        )
        log_path = output_root / "logs" / "runs" / f"{run_name}.log"
        _set_run_status(
            manifest,
            run_name=run_name,
            status="running" if not args.dry_run else "dry_run",
            run_dir=str(run_dir),
            weighting_scheme=scheme,
            epochs=resolved_epochs,
            command=command,
            log_file=str(log_path),
            started_at_utc=utc_now_iso(),
        )
        save_manifest(str(manifest_path), manifest)

        if args.dry_run:
            print("[weighting-comparison] command:")
            print(" ".join(command))
            _set_run_status(
                manifest,
                run_name=run_name,
                status="dry_run",
                run_dir=str(run_dir),
                weighting_scheme=scheme,
                epochs=resolved_epochs,
                completed_at_utc=utc_now_iso(),
                return_code=0,
            )
            save_manifest(str(manifest_path), manifest)
            continue

        log_path.parent.mkdir(parents=True, exist_ok=True)
        run_started = monotonic()
        with log_path.open("w", encoding="utf-8") as logf:
            proc = subprocess.run(command, cwd=REPO_ROOT, text=True, check=False, stdout=logf, stderr=subprocess.STDOUT)

        status = "completed" if proc.returncode == 0 else "failed"
        _set_run_status(
            manifest,
            run_name=run_name,
            status=status,
            run_dir=str(run_dir),
            weighting_scheme=scheme,
            epochs=resolved_epochs,
            completed_at_utc=utc_now_iso(),
            return_code=proc.returncode,
            error=None if proc.returncode == 0 else f"Run '{run_name}' failed. See {log_path}",
        )
        manifest["stages"][f"run_{run_name}"] = {
            "status": status,
            "log_file": str(log_path),
            "elapsed_seconds": monotonic() - run_started,
            "completed_at_utc": utc_now_iso(),
            "return_code": int(proc.returncode),
        }
        save_manifest(str(manifest_path), manifest)
        if proc.returncode != 0:
            raise SystemExit(proc.returncode)

    print(f"[weighting-comparison] completed: {manifest_path}")


if __name__ == "__main__":
    main()
