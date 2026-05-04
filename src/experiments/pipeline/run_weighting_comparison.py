"""Run the thesis loss-balancing / weighting-scheme comparison for single-stage PINN training.

Experiment 3: Loss Balancing
- Single-stage PINN, fixed reference dataset, fixed architecture, fixed collocation.
- Adam optimizer with optional reduce_on_plateau scheduler (enabled by default in final mode).
- Primary comparison axis: loss formulation / weighting scheme.
- Run matrix: strategy × seed_label.
- Screening mode: 1 seed, 100 epochs, W&B project thesis-weighting-experiment-TEST.
- Final mode:    5 seeds, 20 000 epochs, W&B project thesis-weighting-experiment.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.experiments.pipeline.helpers.evaluation import resolve_eval_inputs as _resolve_shared_eval_inputs
from src.experiments.pipeline.helpers.launch_utils import (
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
from src.experiments.pipeline.helpers.manifest import save_manifest, utc_now_iso
from src.experiments.pipeline.helpers.reference import resolve_reference_dataset as _resolve_reference_dataset
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
    default_project_for_mode,
    group_name,
    tags as wandb_tags_list,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REFERENCE_ID = "main_SM4_qbc_b512_ds01"
DEFAULT_OOD_EVAL_ID = "ood_SM4_wide_ic_b512_ds01"
DEFAULT_ADAM_LR = 3.0e-3

SCREENING_SEED_LABELS = ("s01",)
FINAL_SEED_LABELS = ("s01", "s02", "s03", "s04", "s05")
SCREENING_DEFAULT_EPOCHS = 100
FINAL_DEFAULT_EPOCHS = 20_000

# Maps user-facing strategy names → underlying PINN weighting scheme config value.
# Strategies expressed as static + distinct weight/schedule combinations are all
# mapped to "static" here; _build_weighting_run_specs handles the differences.
STRATEGY_TO_SCHEME: dict[str, str] = {
    "data_only": "static",
    "static_uniform": "static",
    "static_tuned": "static",
    "data_warmup_static": "static",
    "ma": "ma",
    "id": "id",
    "dn": "dn",
    "ntk": "ntk_random_batch",
    "relobralo": "relobralo",
    "lra": "paper_lr_annealing",   # legacy alias; not in core default set
}
CORE_STRATEGIES = (
    "data_only",
    "static_tuned",
    "static_uniform",
    "data_warmup_static",
    "ma",
    "id",
    "dn",
    "ntk",
    "relobralo",
)
ALL_STRATEGIES = (*CORE_STRATEGIES, "lra")

SUMMARY_FIELDNAMES = [
    "run_name",
    "strategy",
    "scheme",
    "seed_label",
    "seed_value",
    "dataset_reference_id",
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
    "test_data_loss",
    "global_epoch",
    "num_train_steps",
    "epoch_wall_seconds",
    "cumulative_wall_seconds",
    "total_seconds",
    "training_seconds",
    # Active weights at final epoch (stored in metrics.json final_epoch as train_weight_{name}).
    "train_weight_data",
    "train_weight_physics",
    "train_weight_dt",
    "train_weight_ic",
]


@dataclass(frozen=True)
class WeightingRunSpec:
    strategy: str
    scheme: str
    seed_label: str
    seed_value: int
    loss_weights: dict[str, float]
    schedule_override: str | None

    @property
    def run_name(self) -> str:
        return f"{self.strategy}_{self.seed_label}"

    def wandb_tags(self) -> list[str]:
        return [self.strategy, self.seed_label]


# ---------------------------------------------------------------------------
# Strategy helpers
# ---------------------------------------------------------------------------

def _parse_strategies(raw: str | None) -> list[str]:
    if raw:
        strategies = [item.strip().lower().replace("-", "_") for item in parse_csv_list(raw) if item.strip()]
    else:
        strategies = list(CORE_STRATEGIES)
    if not strategies:
        raise ValueError("Expected at least one strategy.")
    unsupported = [s for s in strategies if s not in STRATEGY_TO_SCHEME]
    if unsupported:
        raise ValueError(
            f"Unsupported strategy/strategies: {', '.join(unsupported)}. "
            f"Use one of: {', '.join(ALL_STRATEGIES)}."
        )
    if len(set(strategies)) != len(strategies):
        raise ValueError("--strategies must not contain duplicates.")
    return strategies


def _resolve_eval_inputs(args: argparse.Namespace) -> dict[str, str | None]:
    return _resolve_shared_eval_inputs(args, default_ood_eval_id=DEFAULT_OOD_EVAL_ID)


# ---------------------------------------------------------------------------
# Run-spec builders
# ---------------------------------------------------------------------------

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


def _build_weighting_run_specs(
    *,
    strategies: list[str],
    seed_pairs: list[tuple[str, int]],
    loss_weight_data: float,
    loss_weight_dt: float,
    loss_weight_physics: float,
    loss_weight_ic: float,
    data_warmup_epochs: int,
) -> list[WeightingRunSpec]:
    base_weights = {
        "data": float(loss_weight_data),
        "dt": float(loss_weight_dt),
        "physics": float(loss_weight_physics),
        "ic": float(loss_weight_ic),
    }
    specs: list[WeightingRunSpec] = []
    for strategy in strategies:
        scheme = STRATEGY_TO_SCHEME[strategy]
        if strategy == "data_only":
            weights: dict[str, float] = {"data": 1.0, "dt": 0.0, "physics": 0.0, "ic": 0.0}
            schedule_override: str | None = None
        elif strategy == "static_uniform":
            weights = {"data": 1.0, "dt": 1.0, "physics": 1.0, "ic": 1.0}
            schedule_override = None
        elif strategy == "static_tuned":
            weights = dict(base_weights)
            schedule_override = None
        elif strategy == "data_warmup_static":
            weights = dict(base_weights)
            schedule_override = (
                _loss_weight_schedule_override(warmup_epochs=int(data_warmup_epochs), base_weights=base_weights)
                if int(data_warmup_epochs) > 0
                else None
            )
        else:
            # Dynamic schemes (ma, id, dn, ntk, relobralo, lra): use base weights
            weights = dict(base_weights)
            schedule_override = None
        for seed_label, seed_value in seed_pairs:
            specs.append(
                WeightingRunSpec(
                    strategy=strategy,
                    scheme=scheme,
                    seed_label=seed_label,
                    seed_value=int(seed_value),
                    loss_weights=weights,
                    schedule_override=schedule_override,
                )
            )
    return specs


# ---------------------------------------------------------------------------
# Command builders
# ---------------------------------------------------------------------------

def _adam_stage_override(
    *,
    epochs: int,
    lr: float,
    batch_size: int,
    allow_sampling: bool,
    use_scheduler: bool,
    scheduler_patience: int,
    scheduler_factor: float,
    scheduler_min_lr: float,
) -> str:
    if use_scheduler:
        scheduler_str = (
            "{"
            "name:reduce_on_plateau,"
            "metric:val_total_loss,"
            "mode:min,"
            f"factor:{float(scheduler_factor)},"
            f"patience:{int(scheduler_patience)},"
            "threshold:0.0001,"
            "threshold_mode:rel,"
            "cooldown:0,"
            f"min_lr:{float(scheduler_min_lr)},"
            "eps:1.0e-8"
            "}"
        )
    else:
        scheduler_str = "null"
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
        "optimizer_kwargs:{eps:1.0e-6},"
        f"scheduler:{scheduler_str},"
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
    run_spec: WeightingRunSpec,
    device: str,
    hidden_dim: int,
    hidden_layers: int,
    activation: str,
    dtype_name: str,
    epochs: int,
    batch_size: int,
    adam_lr: float,
    allow_sampling: bool,
    use_scheduler: bool,
    scheduler_patience: int,
    scheduler_factor: float,
    scheduler_min_lr: float,
    log_every_epoch: int,
    gradient_telemetry: bool,
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
    id_eval_root: str | None,
    ood_eval_root: str | None,
) -> list[str]:
    scheme = run_spec.scheme
    dynamic_components = "[data,dt,physics,ic]" if scheme == "ntk_random_batch" else "[data,dt,ic]"
    update_mode = "step" if scheme == "paper_lr_annealing" else "epoch"
    use_live_batch = "true" if scheme == "paper_lr_annealing" else "false"
    command = [
        python_bin,
        "20_run_pinn.py",
        f"model.model_flag={model_flag}",
        f"model.seed={int(run_spec.seed_value)}",
        "pinn.mode=single_stage",
        f"dataset.root={dataset_root}",
        f"pinn.run_dir={run_dir}",
        f"pinn.device={device}",
        f"pinn.dtype={dtype_name}",
        f"pinn.hidden_dim={int(hidden_dim)}",
        f"pinn.hidden_layers={int(hidden_layers)}",
        f"pinn.activation={activation}",
        f"pinn.default_batch_size={int(batch_size)}",
        "pinn.collocation.sampling.enabled=false",
        "pinn.supervised_sampling.enabled=false",
        f"pinn.gradient_telemetry.enabled={'true' if gradient_telemetry else 'false'}",
        "pinn.checkpointing.enabled=true",
        "pinn.checkpointing.save_best=true",
        "pinn.checkpointing.save_last=false",
        "pinn.checkpointing.save_init=false",
        "pinn.checkpointing.epoch_fractions=[]",
        f"pinn.loss_weights.data={run_spec.loss_weights['data']}",
        f"pinn.loss_weights.dt={run_spec.loss_weights['dt']}",
        f"pinn.loss_weights.physics={run_spec.loss_weights['physics']}",
        f"pinn.loss_weights.ic={run_spec.loss_weights['ic']}",
        f"pinn.weighting.scheme={scheme}",
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
        _adam_stage_override(
            epochs=epochs,
            lr=adam_lr,
            batch_size=batch_size,
            allow_sampling=allow_sampling,
            use_scheduler=use_scheduler,
            scheduler_patience=scheduler_patience,
            scheduler_factor=scheduler_factor,
            scheduler_min_lr=scheduler_min_lr,
        ),
    ]
    if run_spec.schedule_override is not None:
        command.append(run_spec.schedule_override)
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
        "strategy": run.get("strategy"),
        "scheme": run.get("scheme"),
        "seed_label": run.get("seed_label"),
        "seed_value": run.get("seed_value"),
        "dataset_reference_id": run.get("dataset_reference_id"),
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
        "test_data_loss": _float_or_none(final_epoch.get("test_data_loss")),
        "global_epoch": _int_or_none(final_epoch.get("global_epoch")),
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


def _aggregate_by_strategy(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for strategy in sorted({str(row.get("strategy")) for row in rows if row.get("strategy")}):
        members = [row for row in rows if row.get("strategy") == strategy]
        successful = [row for row in members if row.get("status") == "completed"]
        failed = [row for row in members if row.get("status") != "completed"]
        rmse_values = [
            float(row["final_test_rmse"])
            for row in successful
            if row.get("final_test_rmse") is not None
        ]
        training_seconds = [
            float(row["training_seconds"])
            for row in successful
            if row.get("training_seconds") is not None
        ]
        rmse_stats = _mean_std(rmse_values)
        timing_stats = _mean_std(training_seconds)
        out[strategy] = {
            "success_count": len(successful),
            "failure_count": len(failed),
            "final_test_rmse_mean": rmse_stats["mean"],
            "final_test_rmse_std": rmse_stats["std"],
            "training_seconds_mean": timing_stats["mean"],
            "training_seconds_std": timing_stats["std"],
        }
        out[strategy].update(_aggregate_standard_pinn_metrics(successful))
    return out


def _write_summary_artifacts(*, output_root: Path, manifest: dict[str, Any]) -> dict[str, str]:
    rows = [_summary_row_for_run(dict(run)) for run in manifest.get("runs", [])]
    summary_csv = output_root / "summary.csv"
    summary_json = output_root / "summary.json"
    failures_json = output_root / "failures.json"

    write_summary_csv(summary_csv, rows=rows, fieldnames=SUMMARY_FIELDNAMES)

    failures = [
        {
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
    summary_payload = {
        "generated_at_utc": utc_now_iso(),
        "output_root": str(output_root),
        "mode": manifest.get("experiment", {}).get("mode"),
        "dataset": {
            "source": manifest.get("artifacts", {}).get("dataset_source"),
            "reference_id": manifest.get("artifacts", {}).get("dataset_reference_id"),
            "root": manifest.get("artifacts", {}).get("dataset_root"),
            "reference": manifest.get("artifacts", {}).get("dataset_reference"),
        },
        "evaluation": {
            "id_eval_id": manifest.get("artifacts", {}).get("id_eval_id"),
            "id_eval_root": manifest.get("artifacts", {}).get("id_eval_root"),
            "ood_eval_id": manifest.get("artifacts", {}).get("ood_eval_id"),
            "ood_eval_root": manifest.get("artifacts", {}).get("ood_eval_root"),
        },
        "strategies": manifest.get("artifacts", {}).get("strategies", []),
        "seed_labels": manifest.get("artifacts", {}).get("seed_labels", []),
        "seed_values": manifest.get("artifacts", {}).get("seed_values", []),
        "rows": rows,
        "aggregates_by_strategy": _aggregate_by_strategy(rows),
    }
    write_summary_json(summary_json, summary_payload)
    write_failures_json(failures_json, failures)

    return {
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
        "failures_json": str(failures_json),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--python-bin", default=sys.executable, help="Python executable.")
    parser.add_argument(
        "--mode", default="screening", choices=["screening", "final"],
        help="Thesis weighting experiment mode. Controls default epochs, seeds, and W&B project.",
    )
    parser.add_argument("--experiment-tag", default=None, help="Output/W&B tag suffix. Default: timestamp.")
    parser.add_argument("--output-root", default=None, help="Explicit output root. Default: outputs/pinn/weighting_comparison/<tag>.")
    parser.add_argument("--model-flag", default="SM4", help="Model flag.")

    # Dataset
    dataset_group = parser.add_mutually_exclusive_group()
    dataset_group.add_argument(
        "--reference-id", default=None,
        help=f"Reference dataset ID. Default: {DEFAULT_REFERENCE_ID}.",
    )
    dataset_group.add_argument(
        "--dataset-root", default=None,
        help="Explicit preprocessed dataset root. Mutually exclusive with --reference-id.",
    )
    parser.add_argument(
        "--allow-dataset-generation", action="store_true",
        help="Fallback: generate a new dataset instead of using a reference. Mutually exclusive with --reference-id and --dataset-root.",
    )
    parser.add_argument("--preset", default=None, help="Dataset pipeline preset (--allow-dataset-generation only).")
    parser.add_argument("--budget", default="b512", help="Dataset budget (--allow-dataset-generation only).")
    parser.add_argument("--dataset-seed", default="ds01", help="Dataset seed label (--allow-dataset-generation only).")
    parser.add_argument("--id-eval-id", default=None, help="Optional ID evaluation dataset ID from data/evaluation/index.json.")
    parser.add_argument("--ood-eval-id", default=None, help="Optional OOD evaluation dataset ID from data/evaluation/index.json.")
    parser.add_argument("--id-eval-root", default=None, help="Optional explicit ID evaluation preprocessed dataset root.")
    parser.add_argument("--ood-eval-root", default=None, help="Optional explicit OOD evaluation preprocessed dataset root.")
    parser.add_argument("--no-ood-eval", action="store_true", help=f"Disable default OOD evaluation ({DEFAULT_OOD_EVAL_ID}).")

    # Seeds
    parser.add_argument(
        "--seed-labels", default=None,
        help="Comma-separated seed labels from src/config/registry/seeds.yaml. Defaults by --mode.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Deprecated single raw integer seed. Prefer --seed-labels.",
    )

    # Strategies
    parser.add_argument(
        "--strategies", default=None,
        help=(
            f"Comma-separated weighting strategies. Default: core strategies. "
            f"Core: {', '.join(CORE_STRATEGIES)}. All: {', '.join(ALL_STRATEGIES)}."
        ),
    )

    # Architecture / training
    parser.add_argument("--device", default="cuda", help="PINN device.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="PINN hidden width.")
    parser.add_argument("--hidden-layers", type=int, default=4, help="PINN hidden depth.")
    parser.add_argument("--activation", default="tanh", help="PINN activation.")
    parser.add_argument("--dtype", default="float64", help="PINN dtype.")
    parser.add_argument(
        "--epochs", type=int, default=None,
        help=f"Adam epochs. Defaults: screening={SCREENING_DEFAULT_EPOCHS}, final={FINAL_DEFAULT_EPOCHS}.",
    )
    parser.add_argument("--batch-size", type=int, default=1024, help="Adam mini-batch size.")
    parser.add_argument("--adam-lr", type=float, default=DEFAULT_ADAM_LR, help="Adam learning rate.")
    parser.add_argument(
        "--allow-sampling", action=argparse.BooleanOptionalAction, default=False,
        help="Allow configured sampling inside the Adam phase.",
    )

    # Scheduler (reduce_on_plateau)
    parser.add_argument(
        "--use-scheduler", action=argparse.BooleanOptionalAction, default=None,
        help="Use reduce_on_plateau LR scheduler. Default: True in final mode, False in screening.",
    )
    parser.add_argument("--scheduler-patience", type=int, default=500, help="Scheduler patience (epochs).")
    parser.add_argument("--scheduler-factor", type=float, default=0.5, help="Scheduler LR reduction factor.")
    parser.add_argument("--scheduler-min-lr", type=float, default=1e-6, help="Scheduler minimum learning rate.")

    # Loss weights (base / static_tuned values)
    parser.add_argument("--loss-weight-data", type=float, default=1.0, help="Base supervised loss weight.")
    parser.add_argument("--loss-weight-dt", type=float, default=1e-4, help="Base dt loss weight.")
    parser.add_argument("--loss-weight-physics", type=float, default=1e-4, help="Base physics loss weight.")
    parser.add_argument("--loss-weight-ic", type=float, default=1e-3, help="Base IC loss weight.")
    parser.add_argument(
        "--data-warmup-epochs", type=int, default=500,
        help="Warmup epochs for the data_warmup_static strategy (data-only phase before switching to base weights).",
    )

    # Dynamic weighting params
    parser.add_argument("--ema-beta", type=float, default=0.99, help="EMA beta for dynamic weighting schemes.")
    parser.add_argument("--update-interval-epochs", type=int, default=10, help="Dynamic weighting update cadence.")

    # Probe rows (used by NTK and other probe-based schemes)
    parser.add_argument("--probe-data-rows", type=int, default=256, help="Supervised probe rows.")
    parser.add_argument("--probe-physics-rows", type=int, default=256, help="Physics probe rows.")
    parser.add_argument("--probe-init-rows", type=int, default=256, help="IC probe rows.")
    parser.add_argument("--probe-seed", type=int, default=0, help="Probe subset seed.")

    # NTK-specific
    parser.add_argument("--ntk-batch-size-data", type=int, default=None, help="NTK batch size for the data term. Default: probe-data-rows.")
    parser.add_argument("--ntk-batch-size-dt", type=int, default=None, help="NTK batch size for the dt term. Default: probe-data-rows.")
    parser.add_argument("--ntk-batch-size-physics", type=int, default=None, help="NTK batch size for physics. Default: probe-physics-rows.")
    parser.add_argument("--ntk-batch-size-ic", type=int, default=None, help="NTK batch size for IC. Default: probe-init-rows.")
    parser.add_argument("--ntk-seed", type=int, default=0, help="NTK random-batch seed.")
    parser.add_argument("--ntk-refresh-each-update", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ntk-use-ema", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument(
        "--gradient-telemetry", action=argparse.BooleanOptionalAction, default=True,
        help="Enable gradient telemetry (default: on). Pass --no-gradient-telemetry to disable.",
    )

    # W&B
    parser.add_argument("--wandb-project", default=None, help="Override W&B project. Defaults by --mode.")
    parser.add_argument("--wandb-entity", default=None, help="Optional W&B entity.")
    parser.add_argument("--log-every-epoch", type=int, default=10, help="Metric/W&B logging cadence.")
    parser.add_argument("--tag", action="append", default=[], help="Extra W&B tag. Repeatable.")

    # Dataset-generation overrides (only used with --allow-dataset-generation)
    parser.add_argument("--stage1-override", action="append", default=[], help="Extra stage-1 dataset-generation override.")
    parser.add_argument("--stage2-override", action="append", default=[], help="Extra stage-2 preprocess override.")

    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    return parser


def _resolve_seed_pairs(args: argparse.Namespace, *, mode: str) -> list[tuple[str, int]]:
    if args.seed_labels:
        return _seed_pairs_from_labels(_parse_label_list(str(args.seed_labels)))
    if args.seed is not None:
        return [(f"raw{int(args.seed)}", int(args.seed))]
    default_labels = list(SCREENING_SEED_LABELS if mode == "screening" else FINAL_SEED_LABELS)
    return _seed_pairs_from_labels(default_labels)


def _resolve_dataset(
    args: argparse.Namespace,
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
    experiment_id: str,
) -> tuple[Path, str | None]:
    """Return (dataset_root, reference_id_or_None). Updates and saves manifest."""
    if args.allow_dataset_generation and (args.dataset_root or args.reference_id):
        raise ValueError("--allow-dataset-generation is mutually exclusive with --dataset-root and --reference-id.")

    if args.dataset_root:
        dataset_root = Path(args.dataset_root).resolve()
        manifest["artifacts"]["dataset_root"] = str(dataset_root)
        manifest["artifacts"]["dataset_source"] = "provided"
        save_manifest(str(manifest_path), manifest)
        return dataset_root, None

    if not args.allow_dataset_generation:
        reference_id = str(args.reference_id or DEFAULT_REFERENCE_ID)
        reference_entry = _resolve_reference_dataset(reference_id)
        dataset_root = Path(str(reference_entry["preprocessed_root"])).resolve()
        manifest["artifacts"]["dataset_root"] = str(dataset_root)
        manifest["artifacts"]["dataset_source"] = "reference"
        manifest["artifacts"]["dataset_reference_id"] = reference_id
        manifest["artifacts"]["dataset_reference"] = reference_entry
        save_manifest(str(manifest_path), manifest)
        return dataset_root, reference_id

    # --allow-dataset-generation fallback
    resolved_preset = args.preset or "default"
    dataset_command, dataset_pipeline_root = build_dataset_pipeline_command(
        python_bin=args.python_bin,
        method="qbc_deep_ensemble",
        experiment_id=experiment_id,
        preset=resolved_preset,
        budget=str(args.budget),
        dataset_seed=str(args.dataset_seed),
        model_flag=str(args.model_flag),
        run_root=Path(str(manifest["run_root"])),
        stage1_overrides=list(args.stage1_override),
        stage2_overrides=list(args.stage2_override),
    )
    rc = run_logged_stage(
        label="weighting-comparison",
        stage_name="dataset_pipeline",
        command=dataset_command,
        log_path=Path(str(manifest["run_root"])) / "logs" / "dataset_pipeline.log",
        manifest=manifest,
        manifest_path=manifest_path,
        dry_run=bool(args.dry_run),
        cwd=REPO_ROOT,
    )
    if rc != 0:
        raise SystemExit(rc)
    dataset_root = dataset_root_from_manifest(
        dataset_pipeline_root,
        dry_run=bool(args.dry_run),
        model_flag=str(args.model_flag),
    )
    manifest["artifacts"]["dataset_root"] = str(dataset_root)
    manifest["artifacts"]["dataset_source"] = "generated"
    manifest["artifacts"]["dataset_pipeline_root"] = str(dataset_pipeline_root)
    manifest["artifacts"]["dataset_manifest_path"] = str(dataset_pipeline_root / "dataset_manifest.json")
    save_manifest(str(manifest_path), manifest)
    return dataset_root, None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    mode = str(args.mode)

    try:
        seed_pairs = _resolve_seed_pairs(args, mode=mode)
        strategies = _parse_strategies(args.strategies)
        eval_inputs = _resolve_eval_inputs(args)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from None

    epochs = (SCREENING_DEFAULT_EPOCHS if mode == "screening" else FINAL_DEFAULT_EPOCHS) if args.epochs is None else int(args.epochs)
    if epochs < 0:
        raise SystemExit("--epochs must be non-negative.")

    use_scheduler = (mode == "final") if args.use_scheduler is None else bool(args.use_scheduler)

    wandb_project = args.wandb_project or default_project_for_mode(
        mode=mode,
        screening_project="thesis-weighting-experiment-TEST",
        final_project="thesis-weighting-experiment",
    )

    ntk_batch_size_data = int(args.probe_data_rows if args.ntk_batch_size_data is None else args.ntk_batch_size_data)
    ntk_batch_size_dt = int(args.probe_data_rows if args.ntk_batch_size_dt is None else args.ntk_batch_size_dt)
    ntk_batch_size_physics = int(args.probe_physics_rows if args.ntk_batch_size_physics is None else args.ntk_batch_size_physics)
    ntk_batch_size_ic = int(args.probe_init_rows if args.ntk_batch_size_ic is None else args.ntk_batch_size_ic)

    stamp = args.experiment_tag or tag_stamp()
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else (REPO_ROOT / "outputs" / "pinn" / "weighting_comparison" / stamp).resolve()
    )
    output_root.mkdir(parents=True, exist_ok=True)

    experiment_id = f"weighting_comparison_{mode}_{stamp}"
    wandb_group = group_name("weighting_comparison", args.model_flag.lower(), mode, stamp)
    manifest_path = output_root / "run_manifest.json"
    manifest = init_experiment_manifest(
        run_root=str(output_root),
        experiment={
            "id": experiment_id,
            "tag": stamp,
            "mode": mode,
            "model_flag": args.model_flag,
            "strategies": strategies,
        },
    )
    manifest["artifacts"]["wandb_project"] = wandb_project
    manifest["artifacts"]["wandb_group"] = wandb_group
    manifest["artifacts"]["strategies"] = strategies
    manifest["artifacts"]["seed_labels"] = [label for label, _ in seed_pairs]
    manifest["artifacts"]["seed_values"] = [value for _, value in seed_pairs]
    manifest["artifacts"]["epochs"] = epochs
    manifest["artifacts"]["batch_size"] = args.batch_size
    manifest["artifacts"]["adam_lr"] = args.adam_lr
    manifest["artifacts"]["use_scheduler"] = use_scheduler
    manifest["artifacts"]["scheduler_patience"] = args.scheduler_patience
    manifest["artifacts"]["scheduler_factor"] = args.scheduler_factor
    manifest["artifacts"]["scheduler_min_lr"] = args.scheduler_min_lr
    manifest["artifacts"]["gradient_telemetry"] = bool(args.gradient_telemetry)
    manifest["artifacts"]["loss_weights_base"] = {
        "data": args.loss_weight_data,
        "dt": args.loss_weight_dt,
        "physics": args.loss_weight_physics,
        "ic": args.loss_weight_ic,
    }
    manifest["artifacts"]["id_eval_id"] = eval_inputs["id_eval_id"]
    manifest["artifacts"]["id_eval_root"] = eval_inputs["id_eval_root"]
    manifest["artifacts"]["ood_eval_id"] = eval_inputs["ood_eval_id"]
    manifest["artifacts"]["ood_eval_root"] = eval_inputs["ood_eval_root"]
    save_manifest(str(manifest_path), manifest)

    try:
        dataset_root, dataset_reference_id = _resolve_dataset(
            args, manifest=manifest, manifest_path=manifest_path, experiment_id=experiment_id
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from None

    tags_base = wandb_tags_list(
        "weighting_comparison",
        mode,
        args.model_flag.lower(),
        "reference" if dataset_reference_id else None,
        dataset_reference_id,
        args.tag,
    )

    run_specs = _build_weighting_run_specs(
        strategies=strategies,
        seed_pairs=seed_pairs,
        loss_weight_data=args.loss_weight_data,
        loss_weight_dt=args.loss_weight_dt,
        loss_weight_physics=args.loss_weight_physics,
        loss_weight_ic=args.loss_weight_ic,
        data_warmup_epochs=int(args.data_warmup_epochs),
    )

    for run_spec in run_specs:
        run_name = run_spec.run_name
        run_dir = output_root / "runs" / run_name
        wandb_tags = [*tags_base, *run_spec.wandb_tags()]
        command = _build_pinn_command(
            python_bin=args.python_bin,
            model_flag=args.model_flag,
            dataset_root=dataset_root,
            run_dir=run_dir,
            wandb_project=wandb_project,
            wandb_group=wandb_group,
            wandb_name=run_name,
            wandb_entity=args.wandb_entity,
            wandb_tags=wandb_tags,
            run_spec=run_spec,
            device=args.device,
            hidden_dim=args.hidden_dim,
            hidden_layers=args.hidden_layers,
            activation=args.activation,
            dtype_name=args.dtype,
            epochs=epochs,
            batch_size=args.batch_size,
            adam_lr=args.adam_lr,
            allow_sampling=bool(args.allow_sampling),
            use_scheduler=use_scheduler,
            scheduler_patience=args.scheduler_patience,
            scheduler_factor=args.scheduler_factor,
            scheduler_min_lr=args.scheduler_min_lr,
            log_every_epoch=args.log_every_epoch,
            gradient_telemetry=bool(args.gradient_telemetry),
            update_interval_epochs=args.update_interval_epochs,
            ema_beta=args.ema_beta,
            probe_data_rows=args.probe_data_rows,
            probe_physics_rows=args.probe_physics_rows,
            probe_init_rows=args.probe_init_rows,
            probe_seed=args.probe_seed,
            ntk_batch_size_data=ntk_batch_size_data,
            ntk_batch_size_dt=ntk_batch_size_dt,
            ntk_batch_size_physics=ntk_batch_size_physics,
            ntk_batch_size_ic=ntk_batch_size_ic,
            ntk_seed=args.ntk_seed,
            ntk_refresh_each_update=bool(args.ntk_refresh_each_update),
            ntk_use_ema=bool(args.ntk_use_ema),
            id_eval_root=eval_inputs["id_eval_root"],
            ood_eval_root=eval_inputs["ood_eval_root"],
        )
        log_path = output_root / "logs" / "runs" / f"{run_name}.log"
        run_metadata = {
            "strategy": run_spec.strategy,
            "scheme": run_spec.scheme,
            "seed_label": run_spec.seed_label,
            "seed_value": run_spec.seed_value,
            "loss_weights": run_spec.loss_weights,
            "epochs": epochs,
            "dataset_reference_id": dataset_reference_id,
            "dataset_root": str(dataset_root),
            "id_eval_id": eval_inputs["id_eval_id"],
            "id_eval_root": eval_inputs["id_eval_root"],
            "ood_eval_id": eval_inputs["ood_eval_id"],
            "ood_eval_root": eval_inputs["ood_eval_root"],
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
            dry_run=bool(args.dry_run),
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
        save_manifest(str(manifest_path), manifest)
        if int(result["return_code"]) != 0:
            if not args.dry_run:
                summary_artifacts = _write_summary_artifacts(output_root=output_root, manifest=manifest)
                manifest["artifacts"].update(summary_artifacts)
                save_manifest(str(manifest_path), manifest)
                print(f"[weighting-comparison] summary_csv={summary_artifacts['summary_csv']}")
                print(f"[weighting-comparison] summary_json={summary_artifacts['summary_json']}")
                print(f"[weighting-comparison] failures_json={summary_artifacts['failures_json']}")
            raise SystemExit(int(result["return_code"]))

    if not args.dry_run:
        summary_artifacts = _write_summary_artifacts(output_root=output_root, manifest=manifest)
        manifest["artifacts"].update(summary_artifacts)
        save_manifest(str(manifest_path), manifest)
        print(f"[weighting-comparison] summary_csv={summary_artifacts['summary_csv']}")
        print(f"[weighting-comparison] summary_json={summary_artifacts['summary_json']}")
        print(f"[weighting-comparison] failures_json={summary_artifacts['failures_json']}")
    else:
        print("[weighting-comparison] dry-run: summary artifacts were not written")
    print(f"[weighting-comparison] run_manifest={manifest_path}")


if __name__ == "__main__":
    main()
