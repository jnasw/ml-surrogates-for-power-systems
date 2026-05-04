"""Thesis Experiment: Multi-stage PINN comparison pipeline.

Compare multi-stage PINN strategies (Adam followed by SSBroyden residual stages)
against a single-stage Adam baseline.

Fixed:  reference dataset, architecture, Adam lr, static loss weights, fixed collocation.
Vary:   strategy (number of stages / optimizer schedule) × seed.

Run matrix:  strategy × seed.
Run name:    <strategy>_<seed_label>  (e.g. adam_ssbroyden_2stage_s01).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.experiments.pipeline.helpers.evaluation import resolve_eval_inputs as _resolve_shared_eval_inputs
from src.experiments.pipeline.helpers.launch_utils import (
    format_hydra_list,
    init_experiment_manifest,
    parse_csv_list,
    run_logged_command,
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
FINAL_TOTAL_EPOCHS = 30_000
SCREENING_TOTAL_EPOCHS = 100

# ---------------------------------------------------------------------------
# Strategy registry
# Each strategy: list of stages, each stage: list of phases.
# Phase tuple: (phase_name, optimizer, lr, final_epochs)
# final_epochs is the epoch count for FINAL_TOTAL_EPOCHS; scaled for other budgets.
# ---------------------------------------------------------------------------
STRATEGY_REGISTRY: dict[str, dict[str, Any]] = {
    "adam_30000": {
        "pinn_mode": "single_stage",
        "stages": [
            [("adam", "Adam", DEFAULT_ADAM_LR, 30_000)],
        ],
    },
    "adam_ssbroyden_2stage": {
        "pinn_mode": "multistage",
        "stages": [
            [("adam", "Adam", DEFAULT_ADAM_LR, 15_000)],
            [("ssbroyden", "SSBroyden", 1.0, 15_000)],
        ],
    },
    "adam_ssbroyden_3stage": {
        "pinn_mode": "multistage",
        "stages": [
            [("adam", "Adam", DEFAULT_ADAM_LR, 10_000)],
            [("ssbroyden", "SSBroyden", 1.0, 10_000)],
            [("ssbroyden", "SSBroyden", 0.5, 10_000)],
        ],
    },
    "adam_ssbroyden_4stage": {
        "pinn_mode": "multistage",
        "stages": [
            [("adam", "Adam", DEFAULT_ADAM_LR, 8_000)],
            [("ssbroyden", "SSBroyden", 1.0, 8_000)],
            [("ssbroyden", "SSBroyden", 0.5, 8_000)],
            [("ssbroyden", "SSBroyden", 0.25, 6_000)],
        ],
    },
    "adam_ssbroyden_5stage": {
        "pinn_mode": "multistage",
        "stages": [
            [("adam", "Adam", DEFAULT_ADAM_LR, 6_000)],
            [("ssbroyden", "SSBroyden", 1.0, 6_000)],
            [("ssbroyden", "SSBroyden", 0.5, 6_000)],
            [("ssbroyden", "SSBroyden", 0.25, 6_000)],
            [("ssbroyden", "SSBroyden", 0.1, 6_000)],
        ],
    },
}

ALL_STRATEGIES = tuple(STRATEGY_REGISTRY)

SUMMARY_FIELDNAMES = [
    "run_name",
    "strategy",
    "pinn_mode",
    "planned_num_stages",
    "completed_num_stages",
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
    "global_epoch",
    "num_train_steps",
    "epoch_wall_seconds",
    "cumulative_wall_seconds",
    "total_seconds",
    "training_seconds",
    "stage_summary_path",
    *[f"stage{i}_test_rmse" for i in range(5)],
    *[f"stage{i}_test_mae" for i in range(5)],
    *[f"stage{i}_train_total_loss" for i in range(5)],
    *[f"stage{i}_walltime_s" for i in range(5)],
    *[f"stage{i}_improvement_delta" for i in range(5)],
    "residual_probe_path",
    *[f"stage{i}_residual_rms" for i in range(5)],
    *[f"stage{i}_residual_max" for i in range(5)],
]


@dataclass(frozen=True)
class MultistageRunSpec:
    strategy: str
    pinn_mode: str
    planned_num_stages: int
    total_epochs: int
    seed_label: str
    seed_value: int

    @property
    def run_name(self) -> str:
        return f"{self.strategy}_{self.seed_label}"


# ---------------------------------------------------------------------------
# Strategy helpers
# ---------------------------------------------------------------------------

def _parse_strategies(raw: str | None) -> list[str]:
    if raw:
        strategies = [s.strip().lower().replace("-", "_") for s in parse_csv_list(raw) if s.strip()]
    else:
        strategies = list(ALL_STRATEGIES)
    if not strategies:
        raise ValueError("Expected at least one strategy.")
    unsupported = [s for s in strategies if s not in STRATEGY_REGISTRY]
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
# Phase dict builders (Hydra inline dict format)
# ---------------------------------------------------------------------------

def _adam_phase_dict(*, name: str, epochs: int, batch_size: int, lr: float = DEFAULT_ADAM_LR) -> str:
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
        "scheduler:null,"
        "line_search:null,"
        "convergence:null"
        "}"
    )


def _ssbroyden_phase_dict(*, name: str, epochs: int, lr: float) -> str:
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


def _build_phase_dict(*, name: str, optimizer: str, lr: float, epochs: int, batch_size: int) -> str:
    if optimizer == "Adam":
        return _adam_phase_dict(name=name, epochs=epochs, batch_size=batch_size, lr=lr)
    if optimizer == "SSBroyden":
        return _ssbroyden_phase_dict(name=name, epochs=epochs, lr=lr)
    raise ValueError(f"Unsupported optimizer in strategy registry: {optimizer}")


def _scale_epochs(final_epochs: int, *, total_epochs: int) -> int:
    return max(1, round(final_epochs * total_epochs / FINAL_TOTAL_EPOCHS))


def _strategy_overrides(*, run_spec: MultistageRunSpec, batch_size: int) -> list[str]:
    spec = STRATEGY_REGISTRY[run_spec.strategy]
    pinn_mode = spec["pinn_mode"]
    stages = spec["stages"]

    if pinn_mode == "single_stage":
        phase_dicts = [
            _build_phase_dict(
                name=phase_name,
                optimizer=optimizer,
                lr=lr,
                epochs=_scale_epochs(final_epochs, total_epochs=run_spec.total_epochs),
                batch_size=batch_size,
            )
            for (phase_name, optimizer, lr, final_epochs) in stages[0]
        ]
        return [
            "pinn.mode=single_stage",
            "pinn.optimizer_phases=[" + ",".join(phase_dicts) + "]",
        ]

    num_stages = len(stages)
    stage_strings: list[str] = []
    for stage_phases in stages:
        phase_dicts = [
            _build_phase_dict(
                name=phase_name,
                optimizer=optimizer,
                lr=lr,
                epochs=_scale_epochs(final_epochs, total_epochs=run_spec.total_epochs),
                batch_size=batch_size,
            )
            for (phase_name, optimizer, lr, final_epochs) in stage_phases
        ]
        stage_strings.append("[" + ",".join(phase_dicts) + "]")
    return [
        "pinn.mode=multistage",
        "pinn.multistage.stage_optimizer_phases=[" + ",".join(stage_strings) + "]",
        f"pinn.multistage.max_stages={num_stages}",
        "pinn.multistage.stop.residual_rms_threshold=1e-9",
    ]


# ---------------------------------------------------------------------------
# Run spec builder
# ---------------------------------------------------------------------------

def _build_run_specs(
    *,
    strategies: list[str],
    seed_pairs: list[tuple[str, int]],
    total_epochs: int,
) -> list[MultistageRunSpec]:
    specs: list[MultistageRunSpec] = []
    for strategy in strategies:
        spec = STRATEGY_REGISTRY[strategy]
        for seed_label, seed_value in seed_pairs:
            specs.append(MultistageRunSpec(
                strategy=strategy,
                pinn_mode=spec["pinn_mode"],
                planned_num_stages=len(spec["stages"]),
                total_epochs=total_epochs,
                seed_label=seed_label,
                seed_value=int(seed_value),
            ))
    return specs


# ---------------------------------------------------------------------------
# PINN command builder
# ---------------------------------------------------------------------------

def _build_pinn_command(
    *,
    python_bin: str,
    model_flag: str,
    dataset_root: Path,
    run_dir: Path,
    wandb_project: str,
    wandb_group: str,
    wandb_entity: str | None,
    wandb_tags: list[str],
    run_spec: MultistageRunSpec,
    device: str,
    hidden_dim: int,
    hidden_layers: int,
    activation: str,
    dtype_name: str,
    batch_size: int,
    log_every_epoch: int,
    loss_weight_data: float,
    loss_weight_dt: float,
    loss_weight_physics: float,
    loss_weight_ic: float,
    gradient_telemetry: bool,
    id_eval_root: str | None,
    ood_eval_root: str | None,
) -> list[str]:
    command = [
        python_bin,
        "20_run_pinn.py",
        f"model.model_flag={model_flag}",
        f"model.seed={int(run_spec.seed_value)}",
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
        "pinn.collocation.sampling.enabled=false",
        f"pinn.gradient_telemetry.enabled={'true' if gradient_telemetry else 'false'}",
        "pinn.checkpointing.enabled=true",
        "pinn.checkpointing.save_best=true",
        "pinn.checkpointing.save_last=false",
        "pinn.checkpointing.save_init=false",
        "pinn.checkpointing.epoch_fractions=[]",
        f"pinn.loss_weights.data={loss_weight_data}",
        f"pinn.loss_weights.dt={loss_weight_dt}",
        f"pinn.loss_weights.physics={loss_weight_physics}",
        f"pinn.loss_weights.ic={loss_weight_ic}",
        "wandb.use=true",
        f"wandb.project={wandb_project}",
        f"wandb.group={wandb_group}",
        f"wandb.name={run_spec.run_name}",
        f"wandb.tags={format_hydra_list(wandb_tags)}",
        f"logging.log_every_epoch={int(log_every_epoch)}",
        *_strategy_overrides(run_spec=run_spec, batch_size=batch_size),
    ]
    if wandb_entity:
        command.append(f"wandb.entity={wandb_entity}")
    if id_eval_root:
        command.append(f"evaluation.id.root={id_eval_root}")
    if ood_eval_root:
        command.append(f"evaluation.ood.root={ood_eval_root}")
    return command


# ---------------------------------------------------------------------------
# Summary artifact helpers
# ---------------------------------------------------------------------------

def _read_json_list_if_exists(path: Path) -> list[Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, list) else None


def _summary_row_for_run(run: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(str(run.get("run_dir", ""))) if run.get("run_dir") else None
    metrics = _read_json_if_exists(run_dir / "metrics.json") if run_dir else None
    timings = _read_json_if_exists(run_dir / "timings.json") if run_dir else None
    final_train_losses = dict(metrics.get("final_train_losses", {}) or {}) if metrics else {}
    final_epoch = dict(metrics.get("final_epoch", {}) or {}) if metrics else {}
    standard_metrics = _extract_standard_pinn_summary_fields(metrics=metrics, run_dir=run_dir)

    stage_summary_path = (run_dir / "stage_summary.json") if run_dir else None
    stage_summary = _read_json_list_if_exists(stage_summary_path) if stage_summary_path else None
    completed_num_stages = len(stage_summary) if stage_summary is not None else None

    probe_path = (run_dir / "stage_residual_probe.json") if run_dir else None
    probe_entries = _read_json_list_if_exists(probe_path) if probe_path else None

    row: dict[str, Any] = {
        "run_name": run.get("run_name"),
        "strategy": run.get("strategy"),
        "pinn_mode": run.get("pinn_mode"),
        "planned_num_stages": run.get("planned_num_stages"),
        "completed_num_stages": completed_num_stages,
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
        "global_epoch": _int_or_none(final_epoch.get("global_epoch")),
        "num_train_steps": _int_or_none(final_epoch.get("num_train_steps")),
        "epoch_wall_seconds": _float_or_none(final_epoch.get("epoch_wall_seconds")),
        "cumulative_wall_seconds": _float_or_none(final_epoch.get("cumulative_wall_seconds")),
        "total_seconds": _float_or_none(timings.get("total_seconds") if timings else None),
        "training_seconds": _float_or_none(timings.get("training_seconds") if timings else None),
        "stage_summary_path": str(stage_summary_path) if stage_summary is not None else None,
    }

    for i in range(5):
        entry = next((e for e in stage_summary if e.get("stage_idx") == i), None) if stage_summary else None
        row[f"stage{i}_test_rmse"] = _float_or_none(entry.get("test_rmse")) if entry else None
        row[f"stage{i}_test_mae"] = _float_or_none(entry.get("test_mae")) if entry else None
        row[f"stage{i}_train_total_loss"] = _float_or_none(entry.get("final_train_total_loss")) if entry else None
        row[f"stage{i}_walltime_s"] = _float_or_none(entry.get("walltime_s")) if entry else None
        row[f"stage{i}_improvement_delta"] = _float_or_none(entry.get("stage_improvement_delta")) if entry else None

    # probe entry stage_idx=N means "diagnostics computed before starting stage N" = "residuals after stage N-1"
    row["residual_probe_path"] = str(probe_path) if probe_entries is not None else None
    for i in range(5):
        probe_entry = (
            next((e for e in probe_entries if e.get("stage_idx") == i + 1), None)
            if probe_entries else None
        )
        row[f"stage{i}_residual_rms"] = _float_or_none(probe_entry.get("residual_rms")) if probe_entry else None
        row[f"stage{i}_residual_max"] = _float_or_none(probe_entry.get("residual_max")) if probe_entry else None

    return row


def _aggregate_by_strategy(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for strategy in sorted({str(row.get("strategy")) for row in rows if row.get("strategy")}):
        members = [row for row in rows if row.get("strategy") == strategy]
        successful = [row for row in members if row.get("status") == "completed"]
        failed = [row for row in members if row.get("status") != "completed"]
        rmse_values = [float(r["final_test_rmse"]) for r in successful if r.get("final_test_rmse") is not None]
        training_seconds = [float(r["training_seconds"]) for r in successful if r.get("training_seconds") is not None]
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
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--python-bin", default=sys.executable, help="Python executable.")
    parser.add_argument(
        "--mode", default="screening", choices=["screening", "final"],
        help="Experiment mode. Controls default epochs, seeds, and W&B project.",
    )
    parser.add_argument("--experiment-tag", default=None, help="Output/W&B tag suffix. Default: timestamp.")
    parser.add_argument(
        "--output-root", default=None,
        help="Explicit output root. Default: outputs/pinn/multistage_comparison/<tag>.",
    )
    parser.add_argument("--model-flag", default="SM4", help="Model flag.")

    dataset_group = parser.add_mutually_exclusive_group()
    dataset_group.add_argument(
        "--reference-id", default=None,
        help=f"Reference dataset ID. Default: {DEFAULT_REFERENCE_ID}.",
    )
    dataset_group.add_argument(
        "--dataset-root", default=None,
        help="Explicit preprocessed dataset root. Mutually exclusive with --reference-id.",
    )
    parser.add_argument("--id-eval-id", default=None, help="Optional ID evaluation dataset ID from data/evaluation/index.json.")
    parser.add_argument("--ood-eval-id", default=None, help="Optional OOD evaluation dataset ID from data/evaluation/index.json.")
    parser.add_argument("--id-eval-root", default=None, help="Optional explicit ID evaluation preprocessed dataset root.")
    parser.add_argument("--ood-eval-root", default=None, help="Optional explicit OOD evaluation preprocessed dataset root.")
    parser.add_argument("--no-ood-eval", action="store_true", help=f"Disable default OOD evaluation ({DEFAULT_OOD_EVAL_ID}).")

    parser.add_argument(
        "--seed-labels", default=None,
        help="Comma-separated seed labels from src/config/registry/seeds.yaml. Defaults by --mode.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Deprecated single raw integer seed. Prefer --seed-labels.",
    )

    parser.add_argument(
        "--strategies", default=None,
        help=(
            f"Comma-separated strategies. Default: all strategies. "
            f"Choices: {', '.join(ALL_STRATEGIES)}."
        ),
    )

    parser.add_argument("--device", default="cuda", help="PINN device.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="PINN hidden width.")
    parser.add_argument("--hidden-layers", type=int, default=4, help="PINN hidden depth.")
    parser.add_argument("--activation", default="tanh", help="PINN activation.")
    parser.add_argument("--dtype", default="float64", help="PINN dtype.")
    parser.add_argument(
        "--epochs", type=int, default=None,
        help=(
            f"Total epoch budget (all phases proportionally scaled). "
            f"Defaults: screening={SCREENING_TOTAL_EPOCHS}, final={FINAL_TOTAL_EPOCHS}."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1024, help="Mini-batch size for Adam phases.")

    parser.add_argument("--loss-weight-data", type=float, default=1.0, help="Static supervised loss weight.")
    parser.add_argument("--loss-weight-dt", type=float, default=1.0e-4, help="Static dt loss weight.")
    parser.add_argument("--loss-weight-physics", type=float, default=1.0e-4, help="Static physics loss weight.")
    parser.add_argument("--loss-weight-ic", type=float, default=1.0e-3, help="Static IC loss weight.")

    parser.add_argument("--wandb-project", default=None, help="Override W&B project. Defaults by --mode.")
    parser.add_argument("--wandb-entity", default=None, help="Optional W&B entity.")
    parser.add_argument(
        "--gradient-telemetry", action=argparse.BooleanOptionalAction, default=False,
        help="Log gradient norms to W&B (default: disabled).",
    )
    parser.add_argument("--log-every-epoch", type=int, default=10, help="Metric/W&B logging cadence.")
    parser.add_argument("--tag", action="append", default=[], help="Extra W&B tag. Repeatable.")

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
) -> tuple[Path, str | None]:
    if args.dataset_root:
        dataset_root = Path(args.dataset_root).resolve()
        manifest["artifacts"]["dataset_root"] = str(dataset_root)
        manifest["artifacts"]["dataset_source"] = "provided"
        save_manifest(str(manifest_path), manifest)
        return dataset_root, None

    reference_id = str(args.reference_id or DEFAULT_REFERENCE_ID)
    reference_entry = _resolve_reference_dataset(reference_id)
    dataset_root = Path(str(reference_entry["preprocessed_root"])).resolve()
    manifest["artifacts"]["dataset_root"] = str(dataset_root)
    manifest["artifacts"]["dataset_source"] = "reference"
    manifest["artifacts"]["dataset_reference_id"] = reference_id
    manifest["artifacts"]["dataset_reference"] = reference_entry
    save_manifest(str(manifest_path), manifest)
    return dataset_root, reference_id


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

    total_epochs = args.epochs if args.epochs is not None else (
        SCREENING_TOTAL_EPOCHS if mode == "screening" else FINAL_TOTAL_EPOCHS
    )
    if total_epochs <= 0:
        raise SystemExit("--epochs must be positive.")

    wandb_project = args.wandb_project or default_project_for_mode(
        mode=mode,
        screening_project="thesis-multistage-experiment-TEST",
        final_project="thesis-multistage-experiment",
    )

    stamp = args.experiment_tag or tag_stamp()
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else (REPO_ROOT / "outputs" / "pinn" / "multistage_comparison" / stamp).resolve()
    )
    output_root.mkdir(parents=True, exist_ok=True)

    experiment_id = f"multistage_comparison_{mode}_{stamp}"
    wandb_group = group_name("multistage_comparison", args.model_flag.lower(), mode, stamp)
    manifest_path = output_root / "run_manifest.json"
    manifest = init_experiment_manifest(
        run_root=str(output_root),
        experiment={
            "id": experiment_id,
            "tag": stamp,
            "mode": mode,
            "model_flag": args.model_flag,
        },
    )
    manifest["artifacts"]["wandb_project"] = wandb_project
    manifest["artifacts"]["wandb_group"] = wandb_group
    manifest["artifacts"]["strategies"] = strategies
    manifest["artifacts"]["seed_labels"] = [label for label, _ in seed_pairs]
    manifest["artifacts"]["seed_values"] = [value for _, value in seed_pairs]
    manifest["artifacts"]["total_epochs"] = total_epochs
    manifest["artifacts"]["gradient_telemetry"] = bool(args.gradient_telemetry)
    manifest["artifacts"]["id_eval_id"] = eval_inputs["id_eval_id"]
    manifest["artifacts"]["id_eval_root"] = eval_inputs["id_eval_root"]
    manifest["artifacts"]["ood_eval_id"] = eval_inputs["ood_eval_id"]
    manifest["artifacts"]["ood_eval_root"] = eval_inputs["ood_eval_root"]
    save_manifest(str(manifest_path), manifest)

    try:
        dataset_root, dataset_reference_id = _resolve_dataset(
            args, manifest=manifest, manifest_path=manifest_path
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from None

    print(f"[multistage-comparison] repo_root={REPO_ROOT}")
    print(f"[multistage-comparison] mode={mode}")
    print(f"[multistage-comparison] reference_id={dataset_reference_id or '<not set>'}")
    print(f"[multistage-comparison] dataset_root={dataset_root}")
    print(f"[multistage-comparison] strategies={strategies}")
    print(f"[multistage-comparison] seed_labels={[label for label, _ in seed_pairs]}")
    print(f"[multistage-comparison] total_epochs={total_epochs}")
    print(f"[multistage-comparison] wandb_project={wandb_project}")

    tags_base = wandb_tags_list(
        "multistage_comparison",
        mode,
        args.model_flag.lower(),
        "reference" if dataset_reference_id else None,
        dataset_reference_id,
        args.tag,
    )

    run_specs = _build_run_specs(
        strategies=strategies,
        seed_pairs=seed_pairs,
        total_epochs=total_epochs,
    )

    for run_spec in run_specs:
        run_name = run_spec.run_name
        run_dir = output_root / "runs" / run_name
        wandb_tags = [*tags_base, run_spec.strategy, run_spec.seed_label]
        command = _build_pinn_command(
            python_bin=args.python_bin,
            model_flag=args.model_flag,
            dataset_root=dataset_root,
            run_dir=run_dir,
            wandb_project=wandb_project,
            wandb_group=wandb_group,
            wandb_entity=args.wandb_entity,
            wandb_tags=wandb_tags,
            run_spec=run_spec,
            device=args.device,
            hidden_dim=args.hidden_dim,
            hidden_layers=args.hidden_layers,
            activation=args.activation,
            dtype_name=args.dtype,
            batch_size=args.batch_size,
            log_every_epoch=args.log_every_epoch,
            loss_weight_data=args.loss_weight_data,
            loss_weight_dt=args.loss_weight_dt,
            loss_weight_physics=args.loss_weight_physics,
            loss_weight_ic=args.loss_weight_ic,
            gradient_telemetry=bool(args.gradient_telemetry),
            id_eval_root=eval_inputs["id_eval_root"],
            ood_eval_root=eval_inputs["ood_eval_root"],
        )
        log_path = output_root / "logs" / "runs" / f"{run_name}.log"
        run_metadata = {
            "strategy": run_spec.strategy,
            "pinn_mode": run_spec.pinn_mode,
            "planned_num_stages": run_spec.planned_num_stages,
            "total_epochs": run_spec.total_epochs,
            "seed_label": run_spec.seed_label,
            "seed_value": run_spec.seed_value,
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
            label="multistage-comparison",
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
                print(f"[multistage-comparison] summary_csv={summary_artifacts['summary_csv']}")
                print(f"[multistage-comparison] summary_json={summary_artifacts['summary_json']}")
                print(f"[multistage-comparison] failures_json={summary_artifacts['failures_json']}")
            raise SystemExit(int(result["return_code"]))

    if not args.dry_run:
        summary_artifacts = _write_summary_artifacts(output_root=output_root, manifest=manifest)
        manifest["artifacts"].update(summary_artifacts)
        save_manifest(str(manifest_path), manifest)
        print(f"[multistage-comparison] summary_csv={summary_artifacts['summary_csv']}")
        print(f"[multistage-comparison] summary_json={summary_artifacts['summary_json']}")
        print(f"[multistage-comparison] failures_json={summary_artifacts['failures_json']}")
    else:
        print("[multistage-comparison] dry-run: summary artifacts were not written")
    print(f"[multistage-comparison] run_manifest={manifest_path}")


if __name__ == "__main__":
    main()
