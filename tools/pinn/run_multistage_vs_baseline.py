#!/usr/bin/env python3
"""Run paper-style multistage PINN vs single-stage baseline comparisons on one shared dataset."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SINGLE_STAGE_PHASE_SEQUENCES = [
    "LBFGS:300",
]
DEFAULT_MULTISTAGE_STAGE_PLANS = [
    "Adam:300||LBFGS:100",
    "Adam:300||LBFGS:100||Adam:1000||SSBroyden:300",
    "Adam:300||LBFGS:100||Adam:1000||LBFGS:100",
]
PROFILE_CONFIGS: dict[str, dict[str, Any]] = {
    "benchmark": {
        "budget": "b256",
        "preset": "default",
        "stage1_overrides": [],
        "stage2_overrides": ["time=0.05", "num_of_points=20"],
        "gradient_telemetry": True,
        "log_every_epoch": 1,
        "single_stage_phase_sequences": DEFAULT_SINGLE_STAGE_PHASE_SEQUENCES,
        "multistage_stage_plans": DEFAULT_MULTISTAGE_STAGE_PLANS,
    }
}


def _run(command: list[str], *, dry_run: bool, extra_env: dict[str, str] | None = None) -> None:
    print("[multistage-vs-baseline] command:")
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
        raise ValueError(f"profile must be one of: {', '.join(sorted(PROFILE_CONFIGS))}")
    return dict(PROFILE_CONFIGS[profile_name])


def _parse_sequences(raw_values: list[str], defaults: list[str]) -> list[str]:
    values = [item.strip() for item in raw_values if item.strip()]
    return values if values else list(defaults)


def _sequence_slug(raw: str) -> str:
    safe = re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")
    return safe or "sequence"


def _wandb_safe_sequence_tag(raw: str) -> str:
    slug = _sequence_slug(raw)
    prefix = "sequence_"
    max_len = 64
    available = max_len - len(prefix)
    if len(slug) <= available:
        return prefix + slug
    return prefix + slug[:available]


def _parse_phase_entry(
    entry: str,
    *,
    batch_size: int,
    adam_lr: float,
    quasi_newton_lr: float,
    line_search_name: str,
) -> str:
    fields = [item.strip() for item in entry.split(":")]
    if len(fields) not in {2, 3}:
        raise ValueError(
            "Phase sequence entries must use OPTIMIZER:EPOCHS or OPTIMIZER:EPOCHS:LR, "
            f"got '{entry}'."
        )
    optimizer = fields[0]
    epochs = int(fields[1])
    if epochs <= 0:
        raise ValueError(f"Phase '{entry}' must set epochs > 0.")
    lr = float(fields[2]) if len(fields) == 3 else (adam_lr if optimizer == "Adam" else quasi_newton_lr)
    lower = optimizer.lower()
    is_adam = optimizer == "Adam"
    optimizer_kwargs = "{}"
    resolved_line_search_name = "null"
    if not is_adam:
        resolved_line_search = line_search_name
        if optimizer == "LBFGS":
            resolved_line_search = "strong_wolfe"
        elif optimizer == "SSBroyden":
            optimizer_kwargs = "{tau_strategy:paper_default,phi_strategy:paper_default}"
        elif optimizer == "SSBFGS":
            optimizer_kwargs = "{tau_strategy:al_baali}"
        elif optimizer == "BFGS":
            optimizer_kwargs = "{curvature_eps:1.0e-12,init_hessian_scale:1.0}"
        resolved_line_search_name = "{name:" + resolved_line_search + "}"
    return (
        "{"
        f"name:{lower},"
        f"optimizer:{optimizer},"
        f"lr:{lr},"
        f"epochs:{epochs},"
        f"batch_size:{int(batch_size) if is_adam else 'null'},"
        f"shuffle:{'true' if is_adam else 'false'},"
        f"full_batch:{'false' if is_adam else 'true'},"
        "allow_sampling:false,"
        f"optimizer_kwargs:{optimizer_kwargs},"
        f"line_search:{'null' if is_adam else resolved_line_search_name},"
        "convergence:null"
        "}"
    )


def _optimizer_phases_override(
    key_path: str,
    raw_sequence: str,
    *,
    batch_size: int,
    adam_lr: float,
    quasi_newton_lr: float,
    line_search_name: str,
) -> str:
    entries = [item.strip() for item in raw_sequence.split(";") if item.strip()]
    if not entries:
        raise ValueError("Each sequence must include at least one optimizer phase.")
    phases = [
        _parse_phase_entry(
            entry,
            batch_size=batch_size,
            adam_lr=adam_lr,
            quasi_newton_lr=quasi_newton_lr,
            line_search_name=line_search_name,
        )
        for entry in entries
    ]
    return key_path + "=[" + ",".join(phases) + "]"


def _multistage_stage_plan_override(
    raw_plan: str,
    *,
    batch_size: int,
    adam_lr: float,
    quasi_newton_lr: float,
    line_search_name: str,
) -> tuple[str, int]:
    raw_stage_sequences = [item.strip() for item in raw_plan.split("||") if item.strip()]
    if not raw_stage_sequences:
        raise ValueError("Each multistage stage plan must include at least one stage schedule.")
    serialized_stages = [
        _optimizer_phases_override(
            "",
            raw_sequence,
            batch_size=batch_size,
            adam_lr=adam_lr,
            quasi_newton_lr=quasi_newton_lr,
            line_search_name=line_search_name,
        )[1:]
        for raw_sequence in raw_stage_sequences
    ]
    return "pinn.multistage.stage_optimizer_phases=[" + ",".join(serialized_stages) + "]", len(raw_stage_sequences)


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
    mode: str,
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
    raw_sequence: str,
    multistage_stage_plan: str | None,
    adam_lr: float,
    quasi_newton_lr: float,
    line_search_name: str,
    log_every_epoch: int,
    loss_weight_data: float,
    loss_weight_dt: float,
    loss_weight_physics: float,
    loss_weight_ic: float,
    gradient_telemetry: bool,
    multistage_max_stages: int,
    analysis_collocation_rows: int,
) -> list[str]:
    command = [
        python_bin,
        "20_run_pinn.py",
        f"model.model_flag={model_flag}",
        f"model.seed={int(seed)}",
        f"dataset.root={dataset_root}",
        f"pinn.run_dir={run_dir}",
        f"pinn.mode={mode}",
        f"pinn.device={device}",
        f"pinn.dtype={dtype_name}",
        f"pinn.hidden_dim={int(hidden_dim)}",
        f"pinn.hidden_layers={int(hidden_layers)}",
        f"pinn.activation={activation}",
        f"pinn.default_batch_size={int(batch_size)}",
        "pinn.supervised_sampling.enabled=false",
        "pinn.collocation_sampling.enabled=false",
        "pinn.weighting.scheme=static",
        "pinn.evaluation.frequency=1",
        "pinn.checkpointing.epoch_fractions=[0.25,0.5,0.75,1.0]",
        f"pinn.gradient_telemetry.enabled={'true' if gradient_telemetry else 'false'}",
        f"pinn.loss_weights.data={loss_weight_data}",
        f"pinn.loss_weights.dt={loss_weight_dt}",
        f"pinn.loss_weights.physics={loss_weight_physics}",
        f"pinn.loss_weights.ic={loss_weight_ic}",
        f"pinn.multistage.analysis.collocation_rows={int(analysis_collocation_rows)}",
        "wandb.use=true",
        f"wandb.project={wandb_project}",
        f"wandb.group={wandb_group}",
        f"wandb.name={wandb_name}",
        f"wandb.tags={_format_hydra_list(wandb_tags)}",
        f"logging.log_every_epoch={int(log_every_epoch)}",
    ]
    if mode == "single_stage":
        command.append(
            _optimizer_phases_override(
                "pinn.optimizer_phases",
                raw_sequence,
                batch_size=batch_size,
                adam_lr=adam_lr,
                quasi_newton_lr=quasi_newton_lr,
                line_search_name=line_search_name,
            )
        )
    if wandb_entity:
        command.append(f"wandb.entity={wandb_entity}")
    if mode == "multistage":
        if multistage_stage_plan is None:
            raise ValueError("multistage_stage_plan must be provided when mode=multistage.")
        stage_plan_override, stage_count = _multistage_stage_plan_override(
            multistage_stage_plan,
            batch_size=batch_size,
            adam_lr=adam_lr,
            quasi_newton_lr=quasi_newton_lr,
            line_search_name=line_search_name,
        )
        command.append(f"pinn.multistage.max_stages={int(stage_count)}")
        command.append(stage_plan_override)
    else:
        command.append(f"pinn.multistage.max_stages={int(multistage_max_stages)}")
    return command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-bin", default=sys.executable, help="Python executable for pipeline and training.")
    parser.add_argument("--profile", default="benchmark", choices=["benchmark"], help="Experiment profile.")
    parser.add_argument("--experiment-tag", default=None, help="Output/W&B tag suffix. Default: timestamp.")
    parser.add_argument("--output-root", default=None, help="Explicit output root. Default: outputs/pinn/multistage_vs_baseline/<tag>.")
    parser.add_argument("--model-flag", default="SM4", help="Model flag.")
    parser.add_argument("--preset", default=None, help="Dataset pipeline preset. Defaults from profile.")
    parser.add_argument("--budget", default=None, help="Budget label for dataset generation. Defaults from profile.")
    parser.add_argument("--dataset-seed", default="s01", help="Dataset seed label from the registry.")
    parser.add_argument("--dataset-root", default=None, help="Optional explicit preprocessed dataset root.")
    parser.add_argument("--seed", type=int, default=37, help="Training seed.")
    parser.add_argument("--device", default="cuda", help="PINN device override.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="PINN hidden width.")
    parser.add_argument("--hidden-layers", type=int, default=4, help="PINN hidden depth.")
    parser.add_argument("--activation", default="tanh", help="PINN activation.")
    parser.add_argument("--dtype", default="float64", help="PINN dtype.")
    parser.add_argument("--batch-size", type=int, default=1024, help="Adam batch size.")
    parser.add_argument("--adam-lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--quasi-newton-lr", type=float, default=1.0, help="Quasi-Newton learning rate.")
    parser.add_argument("--line-search", default="strong_wolfe", choices=["strong_wolfe", "backtracking"], help="Line-search method.")
    parser.add_argument("--phase-sequence", action="append", default=[], help="Custom optimizer phase sequence for single-stage mode. Can be repeated.")
    parser.add_argument("--multistage-stage-plan", action="append", default=[], help="Explicit multistage stage plan with stages separated by '||'. Example: 'Adam:300||LBFGS:100||Adam:300'. Can be repeated.")
    parser.add_argument("--modes", default="single_stage,multistage", help="Comma-separated PINN modes to run.")
    parser.add_argument("--multistage-max-stages", type=int, default=2, help="Maximum number of multistage residual stages.")
    parser.add_argument("--analysis-collocation-rows", type=int, default=2048, help="Probe collocation rows for multistage diagnostics.")
    parser.add_argument("--wandb-project", default="sm-surrogates-pinn-multistage-vs-baseline", help="Dedicated W&B project.")
    parser.add_argument("--wandb-entity", default=None, help="Optional W&B entity.")
    parser.add_argument("--loss-weight-data", type=float, default=1.0, help="Static supervised loss weight.")
    parser.add_argument("--loss-weight-dt", type=float, default=1.0e-4, help="Static dt loss weight.")
    parser.add_argument("--loss-weight-physics", type=float, default=1.0e-4, help="Static physics loss weight.")
    parser.add_argument("--loss-weight-ic", type=float, default=1.0e-3, help="Static IC loss weight.")
    parser.add_argument("--gradient-telemetry", action=argparse.BooleanOptionalAction, default=True, help="Enable gradient telemetry.")
    parser.add_argument("--log-every-epoch", type=int, default=None, help="Metric/W&B logging cadence. Defaults from profile.")
    parser.add_argument("--tag", action="append", default=[], help="Optional extra W&B tag. Can be passed multiple times.")
    parser.add_argument("--stage1-override", action="append", default=[], help="Extra stage-1 dataset-generation override.")
    parser.add_argument("--stage2-override", action="append", default=[], help="Extra stage-2 preprocess override.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    profile_config = _profile_config(args.profile)
    resolved_sequences = _parse_sequences(args.phase_sequence, list(profile_config["single_stage_phase_sequences"]))
    resolved_multistage_stage_plans = _parse_sequences(args.multistage_stage_plan, list(profile_config["multistage_stage_plans"]))
    resolved_budget = args.budget or str(profile_config["budget"])
    resolved_preset = args.preset or str(profile_config["preset"])
    resolved_stage1_overrides = [*list(profile_config["stage1_overrides"]), *list(args.stage1_override)]
    resolved_stage2_overrides = [*list(profile_config["stage2_overrides"]), *list(args.stage2_override)]
    resolved_gradient_telemetry = bool(profile_config["gradient_telemetry"]) if args.gradient_telemetry is None else bool(args.gradient_telemetry)
    resolved_log_every_epoch = int(profile_config["log_every_epoch"]) if args.log_every_epoch is None else int(args.log_every_epoch)
    modes = [item.strip() for item in str(args.modes).split(",") if item.strip()]
    supported_modes = {"single_stage", "multistage"}
    unknown_modes = [item for item in modes if item not in supported_modes]
    if unknown_modes:
        raise ValueError(f"Unsupported mode(s): {', '.join(unknown_modes)}. Use single_stage,multistage.")

    stamp = args.experiment_tag or _tag_stamp()
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else (REPO_ROOT / "outputs" / "pinn" / "multistage_vs_baseline" / stamp).resolve()
    )
    output_root.mkdir(parents=True, exist_ok=True)

    experiment_id = f"multistage_vs_baseline_{resolved_budget}_{stamp}"
    wandb_group = f"multistage_vs_baseline_{args.model_flag.lower()}_{stamp}"

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
        "dataset_root": str(dataset_root),
        "dataset_pipeline_root": None if dataset_pipeline_root is None else str(dataset_pipeline_root),
        "modes": modes,
        "phase_sequences": resolved_sequences,
        "multistage_stage_plans": resolved_multistage_stage_plans,
        "gradient_telemetry": resolved_gradient_telemetry,
        "analysis_collocation_rows": int(args.analysis_collocation_rows),
        "multistage_max_stages": int(args.multistage_max_stages),
        "output_root": str(output_root),
        "runs": [],
    }

    tags_base = [
        "multistage_vs_baseline",
        args.profile,
        "qbc_deep_ensemble",
        resolved_budget,
        args.model_flag.lower(),
        *args.tag,
    ]

    for mode in modes:
        run_sequences = resolved_multistage_stage_plans if mode == "multistage" else resolved_sequences
        for raw_sequence in run_sequences:
            sequence_slug = _sequence_slug(raw_sequence.replace("||", "_then_"))
            run_name = f"{mode}_{sequence_slug}"
            run_dir = output_root / "runs" / run_name
            wandb_tags = [
                *tags_base,
                mode,
                _wandb_safe_sequence_tag(raw_sequence.replace("||", "_then_")),
                "gradient_telemetry" if resolved_gradient_telemetry else "no_gradient_telemetry",
            ]
            command = _build_pinn_command(
                python_bin=args.python_bin,
                mode=mode,
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
                raw_sequence=raw_sequence,
                multistage_stage_plan=None if mode != "multistage" else raw_sequence,
                adam_lr=args.adam_lr,
                quasi_newton_lr=args.quasi_newton_lr,
                line_search_name=args.line_search,
                log_every_epoch=resolved_log_every_epoch,
                loss_weight_data=args.loss_weight_data,
                loss_weight_dt=args.loss_weight_dt,
                loss_weight_physics=args.loss_weight_physics,
                loss_weight_ic=args.loss_weight_ic,
                gradient_telemetry=resolved_gradient_telemetry,
                multistage_max_stages=args.multistage_max_stages,
                analysis_collocation_rows=args.analysis_collocation_rows,
            )
            _run(command, dry_run=args.dry_run)
            summary["runs"].append(
                {
                    "mode": mode,
                    "phase_sequence": None if mode == "multistage" else raw_sequence,
                    "multistage_stage_plan": None if mode != "multistage" else raw_sequence,
                    "run_name": run_name,
                    "run_dir": str(run_dir),
                }
            )

    summary_path = output_root / "experiment_manifest.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[multistage-vs-baseline] summary_manifest={summary_path}")


if __name__ == "__main__":
    main()
