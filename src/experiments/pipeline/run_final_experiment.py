"""Run the final thesis PINN experiment matrix.

This launcher encodes the selected final design after the dataset, weighting,
optimizer, and multistage comparison notebooks:

- data-only Adam baseline, 5000 epochs
- DN adaptive weighting, 3000 Adam epochs + 2000 SSBroyden refinement epochs
- static-weight SSBroyden, 5000 epochs

The default matrix runs these three strategies for three seeds and all three
model orders using QBC b1024 reference datasets.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.experiments.pipeline.helpers.evaluation import (
    resolve_evaluation_dataset as _resolve_evaluation_dataset,
)
from src.experiments.pipeline.helpers.launch_utils import (
    format_hydra_list,
    init_experiment_manifest,
    parse_csv_list,
    run_logged_command,
    tag_stamp,
    upsert_run_status,
)
from src.experiments.pipeline.helpers.manifest import save_manifest, utc_now_iso
from src.experiments.pipeline.helpers.reference import (
    resolve_reference_dataset as _resolve_reference_dataset,
)
from src.experiments.pipeline.helpers.seeds import (
    parse_label_list as _parse_label_list,
    seed_pairs_from_labels as _seed_pairs_from_labels,
)
from src.experiments.pipeline.helpers.summary import (
    STANDARD_PINN_SUMMARY_FIELDNAMES,
    aggregate_standard_pinn_metrics as _aggregate_standard_pinn_metrics,
    component_loss as _component_loss,
    extract_standard_pinn_summary_fields as _extract_standard_pinn_summary_fields,
    float_or_none as _float_or_none,
    int_or_none as _int_or_none,
    mean_std as _mean_std,
    read_json_if_exists as _read_json_if_exists,
    write_failures_json,
    write_summary_csv,
    write_summary_json,
)
from src.experiments.pipeline.helpers.wandb import (
    group_name,
    tags as wandb_tags_list,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_FLAGS = ("SM4", "SM6", "SM_AVR_GOV")
DEFAULT_MODELS = MODEL_FLAGS
DEFAULT_SEED_LABELS = ("s01", "s02", "s03")
DEFAULT_REFERENCE_PATTERN = "main_{model}_qbc_b1024_ds01"
DEFAULT_ID_EVAL_PATTERN = "id_{model}_lhs_b512_ds01"
DEFAULT_OOD_EVAL_PATTERN = "ood_{model}_wide_ic_b512_ds01"
DEFAULT_ADAM_LR = 3.0e-3
DEFAULT_SSBROYDEN_LR = 1.0
DEFAULT_CHECKPOINT_FRACTIONS = (0.6, 1.0)


STRATEGIES = (
    "data_only_adam_5000",
    "dn_adam3000_ssbroyden2000",
    "ssbroyden_5000",
)

SUMMARY_FIELDNAMES = [
    "run_name",
    "model_flag",
    "strategy",
    "weighting_scheme",
    "optimizer_schedule",
    "seed_label",
    "seed_value",
    "reference_id",
    "dataset_root",
    "id_eval_id",
    "id_eval_root",
    "ood_eval_id",
    "ood_eval_root",
    "run_dir",
    "status",
    "return_code",
    "failure_reason",
    "log_file",
    *STANDARD_PINN_SUMMARY_FIELDNAMES,
    "final_train_total_loss",
    "final_train_data_loss",
    "final_train_physics_loss",
    "final_train_dt_loss",
    "final_train_ic_loss",
    "global_epoch",
    "phase_name",
    "optimizer",
    "num_train_steps",
    "epoch_wall_seconds",
    "cumulative_wall_seconds",
    "total_seconds",
    "training_seconds",
    "train_weight_data",
    "train_weight_physics",
    "train_weight_dt",
    "train_weight_ic",
]


@dataclass(frozen=True)
class FinalRunSpec:
    model_flag: str
    strategy: str
    seed_label: str
    seed_value: int

    @property
    def run_name(self) -> str:
        return f"{self.model_flag.lower()}_{self.strategy}_{self.seed_label}"


def _parse_models(raw: str | None) -> list[str]:
    if raw:
        models = [item.strip() for item in parse_csv_list(raw) if item.strip()]
    else:
        models = list(DEFAULT_MODELS)
    unsupported = [item for item in models if item not in MODEL_FLAGS]
    if unsupported:
        raise ValueError(f"Unsupported model flag(s): {', '.join(unsupported)}. Use: {', '.join(MODEL_FLAGS)}.")
    if len(set(models)) != len(models):
        raise ValueError("--models must not contain duplicates.")
    return models


def _parse_strategies(raw: str | None) -> list[str]:
    if raw:
        strategies = [item.strip().lower().replace("-", "_") for item in parse_csv_list(raw) if item.strip()]
    else:
        strategies = list(STRATEGIES)
    unsupported = [item for item in strategies if item not in STRATEGIES]
    if unsupported:
        raise ValueError(f"Unsupported strategy/strategies: {', '.join(unsupported)}. Use: {', '.join(STRATEGIES)}.")
    if len(set(strategies)) != len(strategies):
        raise ValueError("--strategies must not contain duplicates.")
    return strategies


def _parse_checkpoint_fractions(raw: str | None) -> list[float]:
    if raw in (None, ""):
        values = list(DEFAULT_CHECKPOINT_FRACTIONS)
    else:
        values = [float(item) for item in parse_csv_list(str(raw)) if str(item).strip()]
    for value in values:
        if not (0.0 < float(value) <= 1.0):
            raise ValueError("--checkpoint-fractions values must be in (0, 1].")
    return values


def _format_model_pattern(pattern: str, model_flag: str) -> str:
    return str(pattern).format(model=model_flag, model_lower=model_flag.lower())


def _expected_reference_root(reference_id: str, model_flag: str) -> Path:
    parts = reference_id.split("_")
    budget = next((part for part in parts if part.startswith("b") and part[1:].isdigit()), "b1024")
    return (
        REPO_ROOT
        / "data"
        / "reference"
        / "main"
        / model_flag
        / "qbc_deep_ensemble"
        / budget
        / "ds01"
        / "data"
        / model_flag
        / "dataset_v1"
    ).resolve()


def _expected_eval_root(eval_id: str, model_flag: str, kind: str) -> Path:
    return (
        REPO_ROOT
        / "data"
        / "evaluation"
        / kind
        / model_flag
        / eval_id
        / "data"
        / model_flag
        / "dataset_v1"
    ).resolve()


def _resolve_reference_root(*, reference_id: str, model_flag: str, dry_run: bool) -> Path:
    try:
        entry = _resolve_reference_dataset(reference_id)
        return Path(str(entry["preprocessed_root"])).resolve()
    except (FileNotFoundError, ValueError):
        if dry_run:
            return _expected_reference_root(reference_id, model_flag)
        raise


def _resolve_eval_root(*, eval_id: str | None, model_flag: str, kind: str, dry_run: bool) -> str | None:
    if not eval_id:
        return None
    try:
        entry = _resolve_evaluation_dataset(eval_id)
        return str(Path(str(entry["dataset_root"])).resolve())
    except (FileNotFoundError, ValueError):
        if dry_run:
            return str(_expected_eval_root(eval_id, model_flag, kind))
        raise


def _adam_scheduler_dict(*, enabled: bool, metric: str, factor: float, patience: int) -> str:
    if not enabled:
        return "null"
    return (
        "{"
        "name:reduce_on_plateau,"
        f"metric:{metric},"
        "mode:min,"
        f"factor:{float(factor)},"
        f"patience:{int(patience)},"
        "threshold:0.0001,"
        "threshold_mode:rel,"
        "cooldown:0,"
        "min_lr:0.0,"
        "eps:1.0e-8"
        "}"
    )


def _adam_phase(*, name: str, epochs: int, lr: float, batch_size: int, scheduler: str) -> str:
    return (
        "{"
        f"name:{name},"
        "optimizer:Adam,"
        f"lr:{float(lr)},"
        f"epochs:{int(epochs)},"
        f"batch_size:{int(batch_size)},"
        "shuffle:true,"
        "full_batch:false,"
        "allow_sampling:false,"
        "optimizer_kwargs:{eps:1.0e-6},"
        f"scheduler:{scheduler},"
        "line_search:null,"
        "convergence:null"
        "}"
    )


def _ssbroyden_phase(*, name: str, epochs: int, lr: float) -> str:
    return (
        "{"
        f"name:{name},"
        "optimizer:SSBroyden,"
        f"lr:{float(lr)},"
        f"epochs:{int(epochs)},"
        "batch_size:null,"
        "shuffle:false,"
        "full_batch:true,"
        "allow_sampling:false,"
        "optimizer_kwargs:{tau_strategy:paper_default,phi_strategy:paper_default},"
        "scheduler:null,"
        "line_search:{name:strong_wolfe},"
        "convergence:null"
        "}"
    )


def _strategy_loss_weights(strategy: str, *, data: float, dt: float, physics: float, ic: float) -> dict[str, float]:
    if strategy == "data_only_adam_5000":
        return {"data": 1.0, "dt": 0.0, "physics": 0.0, "ic": 0.0}
    return {"data": float(data), "dt": float(dt), "physics": float(physics), "ic": float(ic)}


def _strategy_weighting_scheme(strategy: str) -> str:
    return "dn" if strategy == "dn_adam3000_ssbroyden2000" else "static"


def _strategy_phases(
    strategy: str,
    *,
    adam_lr: float,
    ssbroyden_lr: float,
    batch_size: int,
    adam_scheduler: str,
    epoch_scale: float,
) -> list[str]:
    def scaled(value: int) -> int:
        return max(1, round(int(value) * float(epoch_scale)))

    if strategy == "data_only_adam_5000":
        return [_adam_phase(name="adam_data_only", epochs=scaled(5000), lr=adam_lr, batch_size=batch_size, scheduler=adam_scheduler)]
    if strategy == "dn_adam3000_ssbroyden2000":
        return [
            _adam_phase(name="adam_dn", epochs=scaled(3000), lr=adam_lr, batch_size=batch_size, scheduler=adam_scheduler),
            _ssbroyden_phase(name="ssbroyden_refine", epochs=scaled(2000), lr=ssbroyden_lr),
        ]
    if strategy == "ssbroyden_5000":
        return [_ssbroyden_phase(name="ssbroyden", epochs=scaled(5000), lr=ssbroyden_lr)]
    raise ValueError(f"Unsupported strategy: {strategy}")


def _build_pinn_command(
    *,
    python_bin: str,
    spec: FinalRunSpec,
    dataset_root: Path,
    run_dir: Path,
    wandb_project: str,
    wandb_group: str,
    wandb_entity: str | None,
    wandb_tags: list[str],
    device: str,
    dtype: str,
    hidden_dim: int,
    hidden_layers: int,
    activation: str,
    batch_size: int,
    adam_lr: float,
    ssbroyden_lr: float,
    adam_scheduler: str,
    epoch_scale: float,
    loss_weights: dict[str, float],
    id_eval_root: str | None,
    ood_eval_root: str | None,
    log_every_epoch: int,
    gradient_telemetry: bool,
    update_interval_epochs: int,
    ema_beta: float,
    checkpoint_fractions: list[float],
    save_init: bool,
) -> list[str]:
    weighting_scheme = _strategy_weighting_scheme(spec.strategy)
    phases = _strategy_phases(
        spec.strategy,
        adam_lr=adam_lr,
        ssbroyden_lr=ssbroyden_lr,
        batch_size=batch_size,
        adam_scheduler=adam_scheduler,
        epoch_scale=epoch_scale,
    )
    command = [
        python_bin,
        "20_run_pinn.py",
        f"model.model_flag={spec.model_flag}",
        f"model.seed={int(spec.seed_value)}",
        "pinn.mode=single_stage",
        f"dataset.root={dataset_root}",
        f"pinn.run_dir={run_dir}",
        f"pinn.device={device}",
        f"pinn.dtype={dtype}",
        f"pinn.hidden_dim={int(hidden_dim)}",
        f"pinn.hidden_layers={int(hidden_layers)}",
        f"pinn.activation={activation}",
        f"pinn.default_batch_size={int(batch_size)}",
        "pinn.supervised_sampling.enabled=false",
        "pinn.collocation.sampling.enabled=false",
        "pinn.evaluation.frequency=25",
        f"pinn.gradient_telemetry.enabled={'true' if gradient_telemetry else 'false'}",
        "pinn.checkpointing.enabled=true",
        "pinn.checkpointing.save_best=true",
        "pinn.checkpointing.save_last=false",
        f"pinn.checkpointing.save_init={'true' if save_init else 'false'}",
        f"pinn.checkpointing.epoch_fractions={format_hydra_list([str(value) for value in checkpoint_fractions])}",
        f"pinn.loss_weights.data={loss_weights['data']}",
        f"pinn.loss_weights.dt={loss_weights['dt']}",
        f"pinn.loss_weights.physics={loss_weights['physics']}",
        f"pinn.loss_weights.ic={loss_weights['ic']}",
        f"pinn.weighting.scheme={weighting_scheme}",
        "pinn.weighting.anchor=physics",
        f"pinn.weighting.ema_beta={float(ema_beta)}",
        f"pinn.weighting.update_interval_epochs={int(update_interval_epochs)}",
        "pinn.weighting.update_mode=epoch",
        "pinn.weighting.use_live_batch=false",
        "pinn.weighting.dynamic_components=[data,dt,ic]",
        "wandb.use=true",
        f"wandb.project={wandb_project}",
        f"wandb.group={wandb_group}",
        f"wandb.name={spec.run_name}",
        f"wandb.tags={format_hydra_list(wandb_tags)}",
        f"logging.log_every_epoch={int(log_every_epoch)}",
        "pinn.optimizer_phases=[" + ",".join(phases) + "]",
    ]
    if wandb_entity:
        command.append(f"wandb.entity={wandb_entity}")
    if id_eval_root:
        command.append(f"evaluation.id.root={id_eval_root}")
    if ood_eval_root:
        command.append(f"evaluation.ood.root={ood_eval_root}")
    return command


def _summary_row_for_run(run: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(str(run.get("run_dir", ""))) if run.get("run_dir") else None
    metrics = _read_json_if_exists(run_dir / "metrics.json") if run_dir else None
    timings = _read_json_if_exists(run_dir / "timings.json") if run_dir else None
    final_train_losses = dict(metrics.get("final_train_losses", {}) or {}) if metrics else {}
    final_epoch = dict(metrics.get("final_epoch", {}) or {}) if metrics else {}
    standard_metrics = _extract_standard_pinn_summary_fields(metrics=metrics, run_dir=run_dir)
    return {
        "run_name": run.get("run_name"),
        "model_flag": run.get("model_flag"),
        "strategy": run.get("strategy"),
        "weighting_scheme": run.get("weighting_scheme"),
        "optimizer_schedule": run.get("optimizer_schedule"),
        "seed_label": run.get("seed_label"),
        "seed_value": run.get("seed_value"),
        "reference_id": run.get("reference_id"),
        "dataset_root": run.get("dataset_root"),
        "id_eval_id": run.get("id_eval_id"),
        "id_eval_root": run.get("id_eval_root"),
        "ood_eval_id": run.get("ood_eval_id"),
        "ood_eval_root": run.get("ood_eval_root"),
        "run_dir": run.get("run_dir"),
        "status": run.get("status"),
        "return_code": run.get("return_code"),
        "failure_reason": run.get("error"),
        "log_file": run.get("log_file"),
        **standard_metrics,
        "final_train_total_loss": _float_or_none(final_train_losses.get("total_loss")),
        "final_train_data_loss": _component_loss(final_train_losses, "data"),
        "final_train_physics_loss": _component_loss(final_train_losses, "physics"),
        "final_train_dt_loss": _component_loss(final_train_losses, "dt"),
        "final_train_ic_loss": _component_loss(final_train_losses, "ic"),
        "global_epoch": _int_or_none(final_epoch.get("global_epoch")),
        "phase_name": final_epoch.get("phase_name"),
        "optimizer": final_epoch.get("optimizer"),
        "num_train_steps": _int_or_none(final_epoch.get("num_train_steps")),
        "epoch_wall_seconds": _float_or_none(final_epoch.get("epoch_wall_seconds")),
        "cumulative_wall_seconds": _float_or_none(final_epoch.get("cumulative_wall_seconds")),
        "total_seconds": _float_or_none(timings.get("total_seconds") if timings else None),
        "training_seconds": _float_or_none(timings.get("training_seconds") if timings else None),
        "train_weight_data": _float_or_none(final_epoch.get("train_weight_data")),
        "train_weight_physics": _float_or_none(final_epoch.get("train_weight_physics")),
        "train_weight_dt": _float_or_none(final_epoch.get("train_weight_dt")),
        "train_weight_ic": _float_or_none(final_epoch.get("train_weight_ic")),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    keys = sorted({
        f"{row.get('model_flag')}::{row.get('strategy')}"
        for row in rows
        if row.get("model_flag") and row.get("strategy")
    })
    for key in keys:
        model_flag, strategy = key.split("::", 1)
        members = [row for row in rows if row.get("model_flag") == model_flag and row.get("strategy") == strategy]
        successful = [row for row in members if row.get("status") == "completed"]
        failed = [row for row in members if row.get("status") != "completed"]
        training_seconds = [
            float(row["training_seconds"])
            for row in successful
            if row.get("training_seconds") is not None
        ]
        timing_stats = _mean_std(training_seconds)
        out[key] = {
            "model_flag": model_flag,
            "strategy": strategy,
            "success_count": len(successful),
            "failure_count": len(failed),
            "training_seconds_mean": timing_stats["mean"],
            "training_seconds_std": timing_stats["std"],
        }
        out[key].update(_aggregate_standard_pinn_metrics(successful))
    return out


def _write_summary_artifacts(*, output_root: Path, manifest: dict[str, Any]) -> dict[str, str]:
    rows = [_summary_row_for_run(dict(run)) for run in manifest.get("runs", [])]
    summary_csv = output_root / "summary.csv"
    summary_json = output_root / "summary.json"
    failures_json = output_root / "failures.json"

    write_summary_csv(summary_csv, rows=rows, fieldnames=SUMMARY_FIELDNAMES)
    failures = [
        {
            "model_flag": row.get("model_flag"),
            "strategy": row.get("strategy"),
            "seed_label": row.get("seed_label"),
            "run_dir": row.get("run_dir"),
            "status": row.get("status"),
            "return_code": row.get("return_code"),
            "failure_reason": row.get("failure_reason"),
            "log_file": row.get("log_file"),
        }
        for row in rows
        if row.get("status") != "completed"
    ]
    payload = {
        "generated_at_utc": utc_now_iso(),
        "output_root": str(output_root),
        "experiment": manifest.get("experiment", {}),
        "artifacts": manifest.get("artifacts", {}),
        "rows": rows,
        "aggregates_by_model_strategy": _aggregate(rows),
    }
    write_summary_json(summary_json, payload)
    write_failures_json(failures_json, failures)
    return {
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
        "failures_json": str(failures_json),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--experiment-tag", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--models", default=None, help=f"Comma-separated model flags. Default: {','.join(DEFAULT_MODELS)}.")
    parser.add_argument("--strategies", default=None, help=f"Comma-separated strategies. Default: {','.join(STRATEGIES)}.")
    parser.add_argument("--seed-labels", default=None, help=f"Comma-separated seed labels. Default: {','.join(DEFAULT_SEED_LABELS)}.")
    parser.add_argument("--reference-id-pattern", default=DEFAULT_REFERENCE_PATTERN)
    parser.add_argument("--id-eval-id-pattern", default=DEFAULT_ID_EVAL_PATTERN)
    parser.add_argument("--ood-eval-id-pattern", default=DEFAULT_OOD_EVAL_PATTERN)
    parser.add_argument("--no-id-eval", action="store_true")
    parser.add_argument("--no-ood-eval", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float64")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--hidden-layers", type=int, default=4)
    parser.add_argument("--activation", default="tanh")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--adam-lr", type=float, default=DEFAULT_ADAM_LR)
    parser.add_argument("--ssbroyden-lr", type=float, default=DEFAULT_SSBROYDEN_LR)
    parser.add_argument("--epoch-scale", type=float, default=1.0, help="Scale all fixed strategy epoch counts. Use for smoke checks only.")
    parser.add_argument("--adam-scheduler", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--adam-scheduler-metric", default="val_total_loss", choices=["train_total_loss", "val_total_loss"])
    parser.add_argument("--adam-scheduler-factor", type=float, default=0.5)
    parser.add_argument("--adam-scheduler-patience", type=int, default=500)
    parser.add_argument("--loss-weight-data", type=float, default=1.0)
    parser.add_argument("--loss-weight-dt", type=float, default=1.0e-4)
    parser.add_argument("--loss-weight-physics", type=float, default=1.0e-4)
    parser.add_argument("--loss-weight-ic", type=float, default=1.0e-3)
    parser.add_argument("--ema-beta", type=float, default=0.99)
    parser.add_argument("--update-interval-epochs", type=int, default=10)
    parser.add_argument("--gradient-telemetry", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--log-every-epoch", type=int, default=10)
    parser.add_argument(
        "--checkpoint-fractions",
        default=",".join(str(value) for value in DEFAULT_CHECKPOINT_FRACTIONS),
        help=(
            "Comma-separated checkpoint fractions to save for post-training analysis. "
            "Default 0.6,1.0 saves the Adam->SSBroyden boundary and final checkpoint."
        ),
    )
    parser.add_argument(
        "--save-init",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save the initialization checkpoint. Useful for posthoc landscape analysis; default keeps final runs compact.",
    )
    parser.add_argument("--wandb-project", default="thesis-final-experiment")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _resolve_seed_pairs(args: argparse.Namespace) -> list[tuple[str, int]]:
    labels = _parse_label_list(str(args.seed_labels)) if args.seed_labels else list(DEFAULT_SEED_LABELS)
    return _seed_pairs_from_labels(labels)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        models = _parse_models(args.models)
        strategies = _parse_strategies(args.strategies)
        seed_pairs = _resolve_seed_pairs(args)
        checkpoint_fractions = _parse_checkpoint_fractions(args.checkpoint_fractions)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from None

    if float(args.epoch_scale) <= 0:
        raise SystemExit("--epoch-scale must be positive.")

    stamp = args.experiment_tag or tag_stamp()
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else (REPO_ROOT / "outputs" / "pinn" / "final_experiment" / stamp).resolve()
    )
    output_root.mkdir(parents=True, exist_ok=True)

    experiment_id = f"final_experiment_{stamp}"
    wandb_group = group_name("final_experiment", "all_models" if len(models) > 1 else models[0].lower(), "final", stamp)
    manifest_path = output_root / "run_manifest.json"
    manifest = init_experiment_manifest(
        run_root=str(output_root),
        experiment={
            "id": experiment_id,
            "tag": stamp,
            "models": models,
            "strategies": strategies,
            "seed_labels": [label for label, _ in seed_pairs],
        },
    )
    manifest["artifacts"].update({
        "wandb_project": args.wandb_project,
        "wandb_group": wandb_group,
        "reference_id_pattern": args.reference_id_pattern,
        "id_eval_id_pattern": None if args.no_id_eval else args.id_eval_id_pattern,
        "ood_eval_id_pattern": None if args.no_ood_eval else args.ood_eval_id_pattern,
        "epoch_scale": float(args.epoch_scale),
        "adam_lr": float(args.adam_lr),
        "ssbroyden_lr": float(args.ssbroyden_lr),
        "adam_scheduler": bool(args.adam_scheduler),
        "checkpoint_fractions": checkpoint_fractions,
        "save_init": bool(args.save_init),
        "loss_weights_base": {
            "data": float(args.loss_weight_data),
            "dt": float(args.loss_weight_dt),
            "physics": float(args.loss_weight_physics),
            "ic": float(args.loss_weight_ic),
        },
    })
    save_manifest(str(manifest_path), manifest)

    adam_scheduler = _adam_scheduler_dict(
        enabled=bool(args.adam_scheduler),
        metric=str(args.adam_scheduler_metric),
        factor=float(args.adam_scheduler_factor),
        patience=int(args.adam_scheduler_patience),
    )
    tags_base = wandb_tags_list("final_experiment", "final", "qbc_b1024", args.tag)

    for model_flag in models:
        reference_id = _format_model_pattern(args.reference_id_pattern, model_flag)
        id_eval_id = None if args.no_id_eval else _format_model_pattern(args.id_eval_id_pattern, model_flag)
        ood_eval_id = None if args.no_ood_eval else _format_model_pattern(args.ood_eval_id_pattern, model_flag)

        try:
            dataset_root = _resolve_reference_root(reference_id=reference_id, model_flag=model_flag, dry_run=bool(args.dry_run))
            id_eval_root = _resolve_eval_root(eval_id=id_eval_id, model_flag=model_flag, kind="id", dry_run=bool(args.dry_run))
            ood_eval_root = _resolve_eval_root(eval_id=ood_eval_id, model_flag=model_flag, kind="ood", dry_run=bool(args.dry_run))
        except (FileNotFoundError, ValueError) as exc:
            raise SystemExit(str(exc)) from None

        for strategy in strategies:
            loss_weights = _strategy_loss_weights(
                strategy,
                data=args.loss_weight_data,
                dt=args.loss_weight_dt,
                physics=args.loss_weight_physics,
                ic=args.loss_weight_ic,
            )
            weighting_scheme = _strategy_weighting_scheme(strategy)
            optimizer_schedule = "+".join(
                "Adam" if "optimizer:Adam" in phase else "SSBroyden"
                for phase in _strategy_phases(
                    strategy,
                    adam_lr=args.adam_lr,
                    ssbroyden_lr=args.ssbroyden_lr,
                    batch_size=args.batch_size,
                    adam_scheduler=adam_scheduler,
                    epoch_scale=args.epoch_scale,
                )
            )
            for seed_label, seed_value in seed_pairs:
                spec = FinalRunSpec(model_flag=model_flag, strategy=strategy, seed_label=seed_label, seed_value=int(seed_value))
                run_dir = output_root / "runs" / model_flag / strategy / seed_label
                log_path = output_root / "logs" / model_flag / f"{strategy}_{seed_label}.log"
                wandb_tags = [*tags_base, model_flag.lower(), strategy, weighting_scheme, seed_label]
                command = _build_pinn_command(
                    python_bin=args.python_bin,
                    spec=spec,
                    dataset_root=dataset_root,
                    run_dir=run_dir,
                    wandb_project=args.wandb_project,
                    wandb_group=wandb_group,
                    wandb_entity=args.wandb_entity,
                    wandb_tags=wandb_tags,
                    device=args.device,
                    dtype=args.dtype,
                    hidden_dim=args.hidden_dim,
                    hidden_layers=args.hidden_layers,
                    activation=args.activation,
                    batch_size=args.batch_size,
                    adam_lr=args.adam_lr,
                    ssbroyden_lr=args.ssbroyden_lr,
                    adam_scheduler=adam_scheduler,
                    epoch_scale=args.epoch_scale,
                    loss_weights=loss_weights,
                    id_eval_root=id_eval_root,
                    ood_eval_root=ood_eval_root,
                    log_every_epoch=args.log_every_epoch,
                    gradient_telemetry=bool(args.gradient_telemetry),
                    update_interval_epochs=args.update_interval_epochs,
                    ema_beta=args.ema_beta,
                    checkpoint_fractions=checkpoint_fractions,
                    save_init=bool(args.save_init),
                )
                run_metadata = {
                    "model_flag": model_flag,
                    "strategy": strategy,
                    "weighting_scheme": weighting_scheme,
                    "optimizer_schedule": optimizer_schedule,
                    "seed_label": seed_label,
                    "seed_value": int(seed_value),
                    "reference_id": reference_id,
                    "dataset_root": str(dataset_root),
                    "id_eval_id": id_eval_id,
                    "id_eval_root": id_eval_root,
                    "ood_eval_id": ood_eval_id,
                    "ood_eval_root": ood_eval_root,
                    "loss_weights": loss_weights,
                }
                upsert_run_status(
                    manifest,
                    run_name=spec.run_name,
                    status="dry_run" if args.dry_run else "running",
                    run_dir=str(run_dir),
                    run_metadata=run_metadata,
                    command=command,
                    log_file=str(log_path),
                    started_at_utc=utc_now_iso(),
                )
                save_manifest(str(manifest_path), manifest)
                result = run_logged_command(
                    label="final-experiment",
                    command=command,
                    log_path=log_path,
                    dry_run=bool(args.dry_run),
                    cwd=REPO_ROOT,
                )
                upsert_run_status(
                    manifest,
                    run_name=spec.run_name,
                    status=str(result["status"]),
                    run_dir=str(run_dir),
                    run_metadata=run_metadata,
                    completed_at_utc=str(result["completed_at_utc"]),
                    return_code=int(result["return_code"]),
                    error=None if int(result["return_code"]) == 0 else f"Run '{spec.run_name}' failed. See {log_path}",
                    extra_fields={"elapsed_seconds": float(result["elapsed_seconds"])},
                )
                save_manifest(str(manifest_path), manifest)
                if int(result["return_code"]) != 0:
                    if not args.dry_run:
                        summary_artifacts = _write_summary_artifacts(output_root=output_root, manifest=manifest)
                        manifest["artifacts"].update(summary_artifacts)
                        save_manifest(str(manifest_path), manifest)
                    raise SystemExit(int(result["return_code"]))

    if not args.dry_run:
        summary_artifacts = _write_summary_artifacts(output_root=output_root, manifest=manifest)
        manifest["artifacts"].update(summary_artifacts)
        save_manifest(str(manifest_path), manifest)
        print(f"[final-experiment] summary_csv={summary_artifacts['summary_csv']}")
        print(f"[final-experiment] summary_json={summary_artifacts['summary_json']}")
        print(f"[final-experiment] failures_json={summary_artifacts['failures_json']}")
    else:
        print("[final-experiment] dry-run: summary artifacts were not written")
    print(f"[final-experiment] run_manifest={manifest_path}")


if __name__ == "__main__":
    main()
