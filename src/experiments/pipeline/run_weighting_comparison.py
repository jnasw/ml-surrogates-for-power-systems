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
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.experiments.pipeline.launch_utils import (
    build_dataset_pipeline_command,
    dataset_root_from_manifest,
    format_hydra_list,
    init_experiment_manifest,
    parse_csv_list,
    run_logged_command,
    run_logged_stage,
    tag_stamp,
    upsert_run_status,
)
from src.experiments.pipeline.manifest import save_manifest, utc_now_iso


REPO_ROOT = Path(__file__).resolve().parents[3]
SUPPORTED_WEIGHTING_SCHEMES = ("static", "ma", "paper_lr_annealing", "id", "dn", "relobralo", "ntk_random_batch")
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


@dataclass(frozen=True)
class WeightingRunSpec:
    run_name: str
    weighting_scheme: str
    loss_weights: dict[str, float]
    schedule_override: str | None
    extra_tags: tuple[str, ...]


def _build_weighting_run_specs(
    *,
    schemes: list[str],
    resolved_epochs: int,
    loss_weight_data: float,
    loss_weight_dt: float,
    loss_weight_physics: float,
    loss_weight_ic: float,
    include_static_uniform: bool,
    include_static_data_only: bool,
    static_data_warmup_epochs: int,
) -> list[WeightingRunSpec]:
    base_weights = {
        "data": float(loss_weight_data),
        "dt": float(loss_weight_dt),
        "physics": float(loss_weight_physics),
        "ic": float(loss_weight_ic),
    }
    run_specs: list[WeightingRunSpec] = []
    for scheme in schemes:
        run_specs.append(
            WeightingRunSpec(
                run_name=f"{scheme}_adam{int(resolved_epochs)}",
                weighting_scheme=scheme,
                loss_weights=dict(base_weights),
                schedule_override=None,
                extra_tags=(scheme,),
            )
        )
    if include_static_uniform:
        run_specs.append(
            WeightingRunSpec(
                run_name=f"static_uniform_adam{int(resolved_epochs)}",
                weighting_scheme="static",
                loss_weights={"data": 1.0, "dt": 1.0, "physics": 1.0, "ic": 1.0},
                schedule_override=None,
                extra_tags=("static", "uniform_weights"),
            )
        )
    if include_static_data_only:
        run_specs.append(
            WeightingRunSpec(
                run_name=f"static_data_only_adam{int(resolved_epochs)}",
                weighting_scheme="static",
                loss_weights={"data": 1.0, "dt": 0.0, "physics": 0.0, "ic": 0.0},
                schedule_override=None,
                extra_tags=("static", "data_only"),
            )
        )
    if int(static_data_warmup_epochs) > 0:
        run_specs.append(
            WeightingRunSpec(
                run_name=f"static_data_warmup{int(static_data_warmup_epochs)}_adam{int(resolved_epochs)}",
                weighting_scheme="static",
                loss_weights=dict(base_weights),
                schedule_override=_loss_weight_schedule_override(
                    warmup_epochs=int(static_data_warmup_epochs),
                    base_weights=dict(base_weights),
                ),
                extra_tags=("static", "data_warmup", f"warmup{int(static_data_warmup_epochs)}"),
            )
        )
    return run_specs


def _parse_weighting_schemes(raw: str) -> list[str]:
    schemes = [item.lower() for item in parse_csv_list(raw)]
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
        f"allow_sampling:{'true' if allow_sampling else 'false'}"
        "}"
        "]"
    )


def _loss_weight_schedule_override(*, warmup_epochs: int, base_weights: dict[str, float]) -> str:
    return (
        "pinn.loss_weight_schedule=["
        "{epochs:"
        f"{int(warmup_epochs)},"
        "weights:{data:1.0,dt:0.0,physics:0.0,ic:0.0}"
        "},"
        "{epochs:null,weights:{"
        f"data:{float(base_weights['data'])},"
        f"dt:{float(base_weights['dt'])},"
        f"physics:{float(base_weights['physics'])},"
        f"ic:{float(base_weights['ic'])}"
        "}}]"
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
    ntk_batch_size_data: int,
    ntk_batch_size_dt: int,
    ntk_batch_size_physics: int,
    ntk_batch_size_ic: int,
    ntk_seed: int,
    ntk_refresh_each_update: bool,
    ntk_use_ema: bool,
    loss_weight_schedule_override: str | None = None,
) -> list[str]:
    dynamic_components = "[data,dt,physics,ic]" if weighting_scheme == "ntk_random_batch" else "[data,dt,ic]"
    update_mode = "step" if weighting_scheme == "paper_lr_annealing" else "epoch"
    use_live_batch = "true" if weighting_scheme == "paper_lr_annealing" else "false"
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
        f"pinn.weighting.update_mode={update_mode}",
        f"pinn.weighting.use_live_batch={use_live_batch}",
        f"pinn.weighting.dynamic_components={dynamic_components}",
        "pinn.weighting.relobralo.rho=0.95",
        f"pinn.weighting.probe.data_rows={int(probe_data_rows)}",
        f"pinn.weighting.probe.physics_rows={int(probe_physics_rows)}",
        f"pinn.weighting.probe.init_rows={int(probe_init_rows)}",
        f"pinn.weighting.probe.seed={int(probe_seed)}",
        f"pinn.weighting.probe.refresh_each_update={'true' if ntk_refresh_each_update else 'false'}",
        f"pinn.weighting.ntk.batch_size.data={int(ntk_batch_size_data)}",
        f"pinn.weighting.ntk.batch_size.dt={int(ntk_batch_size_dt)}",
        f"pinn.weighting.ntk.batch_size.physics={int(ntk_batch_size_physics)}",
        f"pinn.weighting.ntk.batch_size.ic={int(ntk_batch_size_ic)}",
        f"pinn.weighting.ntk.seed={int(ntk_seed)}",
        f"pinn.weighting.ntk.use_ema={'true' if ntk_use_ema else 'false'}",
        "wandb.use=true",
        f"wandb.project={wandb_project}",
        f"wandb.group={wandb_group}",
        f"wandb.name={wandb_name}",
        f"wandb.tags={format_hydra_list(wandb_tags)}",
        f"logging.log_every_epoch={int(log_every_epoch)}",
        _adam_stage_override(epochs=epochs, lr=adam_lr, batch_size=batch_size, allow_sampling=allow_sampling),
    ]
    if loss_weight_schedule_override is not None:
        command.append(loss_weight_schedule_override)
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
    parser.add_argument("--include-static-uniform", action="store_true", help="Add a static baseline with uniform 1:1:1:1 loss weights.")
    parser.add_argument("--include-static-data-only", action="store_true", help="Add a static baseline with data-only loss weights (1,0,0,0).")
    parser.add_argument("--static-data-warmup-epochs", type=int, default=0, help="Add a static run that uses data-only weights for N epochs, then switches to the configured static loss weights.")
    parser.add_argument("--ema-beta", type=float, default=0.99, help="EMA beta used by dynamic weighting.")
    parser.add_argument("--update-interval-epochs", type=int, default=10, help="Dynamic weighting update cadence.")
    parser.add_argument("--probe-data-rows", type=int, default=None, help="Fixed supervised probe rows. Defaults from --profile.")
    parser.add_argument("--probe-physics-rows", type=int, default=None, help="Fixed physics probe rows. Defaults from --profile.")
    parser.add_argument("--probe-init-rows", type=int, default=None, help="Fixed init probe rows. Defaults from --profile.")
    parser.add_argument("--probe-seed", type=int, default=0, help="Probe subset seed.")
    parser.add_argument("--ntk-batch-size-data", type=int, default=None, help="NTK random-batch size for the supervised data term.")
    parser.add_argument("--ntk-batch-size-dt", type=int, default=None, help="NTK random-batch size for the supervised dt term.")
    parser.add_argument("--ntk-batch-size-physics", type=int, default=None, help="NTK random-batch size for the physics term.")
    parser.add_argument("--ntk-batch-size-ic", type=int, default=None, help="NTK random-batch size for the IC term.")
    parser.add_argument("--ntk-seed", type=int, default=0, help="NTK random-batch seed.")
    parser.add_argument(
        "--ntk-refresh-each-update",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Refresh the weighting probe subset on each NTK update.",
    )
    parser.add_argument(
        "--ntk-use-ema",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply EMA smoothing to NTK-derived weights.",
    )
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
    ntk_batch_size_data = int(probe_data_rows if args.ntk_batch_size_data is None else args.ntk_batch_size_data)
    ntk_batch_size_dt = int(probe_data_rows if args.ntk_batch_size_dt is None else args.ntk_batch_size_dt)
    ntk_batch_size_physics = int(
        probe_physics_rows if args.ntk_batch_size_physics is None else args.ntk_batch_size_physics
    )
    ntk_batch_size_ic = int(probe_init_rows if args.ntk_batch_size_ic is None else args.ntk_batch_size_ic)

    stamp = args.experiment_tag or tag_stamp()
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else (REPO_ROOT / "outputs" / "pinn" / "weighting_comparison" / stamp).resolve()
    )
    output_root.mkdir(parents=True, exist_ok=True)

    experiment_id = f"weighting_comparison_{resolved_budget}_{stamp}"
    wandb_group = f"weighting_comparison_{args.model_flag.lower()}_{stamp}"
    manifest_path = output_root / "run_manifest.json"
    manifest = init_experiment_manifest(
        run_root=str(output_root),
        experiment={
            "id": experiment_id,
            "tag": stamp,
            "profile": args.profile,
            "budget": resolved_budget,
            "preset": resolved_preset,
            "dataset_seed": args.dataset_seed,
            "model_flag": args.model_flag,
            "weighting_schemes": list(schemes),
        },
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
    manifest["artifacts"]["include_static_uniform"] = bool(args.include_static_uniform)
    manifest["artifacts"]["include_static_data_only"] = bool(args.include_static_data_only)
    manifest["artifacts"]["static_data_warmup_epochs"] = int(args.static_data_warmup_epochs)
    manifest["artifacts"]["weighting_probe_rows"] = {
        "data": probe_data_rows,
        "physics": probe_physics_rows,
        "init": probe_init_rows,
        "seed": int(args.probe_seed),
    }
    manifest["artifacts"]["weighting_ntk"] = {
        "batch_size": {
            "data": ntk_batch_size_data,
            "dt": ntk_batch_size_dt,
            "physics": ntk_batch_size_physics,
            "ic": ntk_batch_size_ic,
        },
        "seed": int(args.ntk_seed),
        "refresh_each_update": bool(args.ntk_refresh_each_update),
        "use_ema": bool(args.ntk_use_ema),
    }
    save_manifest(str(manifest_path), manifest)

    if args.dataset_root:
        dataset_root = Path(args.dataset_root).resolve()
        dataset_pipeline_root = None
        manifest["artifacts"]["dataset_root"] = str(dataset_root)
        manifest["artifacts"]["dataset_source"] = "provided"
        save_manifest(str(manifest_path), manifest)
    else:
        dataset_command, dataset_pipeline_root = build_dataset_pipeline_command(
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
        rc = run_logged_stage(
            label="weighting-comparison",
            stage_name="dataset_pipeline",
            command=dataset_command,
            log_path=output_root / "logs" / "dataset_pipeline.log",
            manifest=manifest,
            manifest_path=manifest_path,
            dry_run=args.dry_run,
            cwd=REPO_ROOT,
        )
        if rc != 0:
            raise SystemExit(rc)
        dataset_root = dataset_root_from_manifest(
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

    run_specs = _build_weighting_run_specs(
        schemes=schemes,
        resolved_epochs=resolved_epochs,
        loss_weight_data=loss_weight_data,
        loss_weight_dt=loss_weight_dt,
        loss_weight_physics=loss_weight_physics,
        loss_weight_ic=loss_weight_ic,
        include_static_uniform=bool(args.include_static_uniform),
        include_static_data_only=bool(args.include_static_data_only),
        static_data_warmup_epochs=int(args.static_data_warmup_epochs),
    )

    for spec in run_specs:
        run_name = str(spec.run_name)
        run_dir = output_root / "runs" / run_name
        wandb_tags = [*tags_base, *list(spec.extra_tags)]
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
            loss_weight_data=float(spec.loss_weights["data"]),
            loss_weight_dt=float(spec.loss_weights["dt"]),
            loss_weight_physics=float(spec.loss_weights["physics"]),
            loss_weight_ic=float(spec.loss_weights["ic"]),
            gradient_telemetry=resolved_gradient_telemetry,
            weighting_scheme=str(spec.weighting_scheme),
            update_interval_epochs=args.update_interval_epochs,
            ema_beta=args.ema_beta,
            probe_data_rows=probe_data_rows,
            probe_physics_rows=probe_physics_rows,
            probe_init_rows=probe_init_rows,
            probe_seed=args.probe_seed,
            ntk_batch_size_data=ntk_batch_size_data,
            ntk_batch_size_dt=ntk_batch_size_dt,
            ntk_batch_size_physics=ntk_batch_size_physics,
            ntk_batch_size_ic=ntk_batch_size_ic,
            ntk_seed=args.ntk_seed,
            ntk_refresh_each_update=bool(args.ntk_refresh_each_update),
            ntk_use_ema=bool(args.ntk_use_ema),
            loss_weight_schedule_override=spec.schedule_override,
        )
        log_path = output_root / "logs" / "runs" / f"{run_name}.log"
        run_metadata = {
            "weighting_scheme": str(spec.weighting_scheme),
            "epochs": resolved_epochs,
        }
        upsert_run_status(
            manifest,
            run_name=run_name,
            status="running" if not args.dry_run else "dry_run",
            run_dir=str(run_dir),
            run_metadata=run_metadata,
            command=command,
            log_file=str(log_path),
            started_at_utc=utc_now_iso(),
        )
        save_manifest(str(manifest_path), manifest)
        result = run_logged_command(
            label="weighting-comparison",
            command=command,
            log_path=log_path,
            dry_run=args.dry_run,
            cwd=REPO_ROOT,
        )
        upsert_run_status(
            manifest,
            run_name=run_name,
            status=str(result["status"]),
            run_dir=str(run_dir),
            run_metadata=run_metadata,
            completed_at_utc=str(result["completed_at_utc"]),
            return_code=int(result["return_code"]),
            error=None if int(result["return_code"]) == 0 else f"Run '{run_name}' failed. See {log_path}",
            extra_fields={"elapsed_seconds": float(result["elapsed_seconds"])},
        )
        manifest["stages"][f"run_{run_name}"] = {
            "status": str(result["status"]),
            "log_file": str(log_path),
            "elapsed_seconds": float(result["elapsed_seconds"]),
            "completed_at_utc": str(result["completed_at_utc"]),
            "return_code": int(result["return_code"]),
        }
        save_manifest(str(manifest_path), manifest)
        if int(result["return_code"]) != 0:
            raise SystemExit(int(result["return_code"]))

    print(f"[weighting-comparison] completed: {manifest_path}")


if __name__ == "__main__":
    main()
