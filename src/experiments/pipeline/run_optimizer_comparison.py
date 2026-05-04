"""Run the thesis optimizer comparison matrix for single-stage PINN training."""

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
    parse_int_list as _parse_int_list,
    parse_label_list as _parse_label_list,
    raw_seed_pairs,
    seed_pairs_from_labels as _seed_pairs_from_labels,
)
from src.experiments.pipeline.helpers.summary import (
    component_loss as _component_loss,
    eval_metrics as _eval_metrics,
    float_or_none as _float_or_none,
    int_or_none as _int_or_none,
    json_or_none as _json_or_none,
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
CALIBRATED_OPTIMIZER_LRS = {
    "Adam": DEFAULT_ADAM_LR,
    "SOAP": 5.0e-2,
    "LBFGS": 1.0e-1,
    "BFGS": 1.0,
    "SSBFGS": 3.0e-1,
    "SSBroyden": 1.0,
}
SCREENING_SEED_LABELS = ("s01",)
FINAL_SEED_LABELS = ("s01", "s02", "s03", "s04", "s05")

FULL_BATCH_OPTIMIZERS = frozenset({"BFGS", "LBFGS", "SSBFGS", "SSBroyden"})
STOCHASTIC_QN_OPTIMIZERS = frozenset({"sSSBFGS", "sSSBroyden"})
MINIBATCH_OPTIMIZERS = frozenset({"Adam", "SOAP", *STOCHASTIC_QN_OPTIMIZERS})

CORE_STRATEGIES = (
    "adam",
    "soap",
    "bfgs",
    "lbfgs",
    "ssbfgs",
    "ssbroyden",
    "adam_soap",
    "adam_bfgs",
    "adam_lbfgs",
    "adam_ssbfgs",
    "adam_ssbroyden",
)
EXPERIMENTAL_STRATEGIES = (
    "sssbfgs",
    "sssbroyden",
    "adam_sssbfgs",
    "adam_sssbroyden",
)
ALL_STRATEGIES = (*CORE_STRATEGIES, *EXPERIMENTAL_STRATEGIES)
STRATEGY_TO_OPTIMIZER = {
    "adam": "Adam",
    "soap": "SOAP",
    "bfgs": "BFGS",
    "lbfgs": "LBFGS",
    "ssbfgs": "SSBFGS",
    "ssbroyden": "SSBroyden",
    "sssbfgs": "sSSBFGS",
    "sssbroyden": "sSSBroyden",
    "adam_soap": "SOAP",
    "adam_bfgs": "BFGS",
    "adam_lbfgs": "LBFGS",
    "adam_ssbfgs": "SSBFGS",
    "adam_ssbroyden": "SSBroyden",
    "adam_sssbfgs": "sSSBFGS",
    "adam_sssbroyden": "sSSBroyden",
}
WARMUP_STRATEGIES = {name for name in STRATEGY_TO_OPTIMIZER if name.startswith("adam_")}
EXPERIMENTAL_STRATEGY_SET = frozenset(EXPERIMENTAL_STRATEGIES)
SUMMARY_FIELDNAMES = [
    "run_name",
    "strategy",
    "strategy_group",
    "is_experimental_strategy",
    "optimizer",
    "training_strategy",
    "seed_label",
    "seed_value",
    "total_epochs",
    "main_epochs",
    "adam_warmup_epochs",
    "adam_lr",
    "main_lr",
    "lr_source",
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
    "final_train_mse",
    "final_train_rmse",
    "final_train_mae",
    "final_val_mse",
    "final_val_rmse",
    "final_val_mae",
    "final_test_mse",
    "final_test_rmse",
    "final_test_mae",
    "id_eval_mse",
    "id_eval_rmse",
    "id_eval_mae",
    "ood_eval_mse",
    "ood_eval_rmse",
    "ood_eval_mae",
    "id_ood_rmse_gap",
    "test_ood_rmse_gap",
    "best_checkpoint_epoch",
    "best_checkpoint_wall_seconds",
    "best_checkpoint_phase_name",
    "best_checkpoint_train_mse",
    "best_checkpoint_train_rmse",
    "best_checkpoint_train_mae",
    "best_checkpoint_val_mse",
    "best_checkpoint_val_rmse",
    "best_checkpoint_val_mae",
    "best_checkpoint_test_mse",
    "best_checkpoint_test_rmse",
    "best_checkpoint_test_mae",
    "best_checkpoint_ood_eval_mse",
    "best_checkpoint_ood_eval_rmse",
    "best_checkpoint_ood_eval_mae",
    "best_checkpoint_test_ood_rmse_gap",
    "final_train_total_loss",
    "final_train_data_loss",
    "final_train_physics_loss",
    "final_train_dt_loss",
    "final_train_ic_loss",
    "final_test_losses",
    "test_data_loss",
    "global_epoch",
    "phase_name",
    "num_train_steps",
    "epoch_wall_seconds",
    "cumulative_wall_seconds",
    "total_seconds",
    "training_seconds",
]


@dataclass(frozen=True)
class OptimizerRunSpec:
    strategy: str
    optimizer: str
    seed_label: str
    seed_value: int
    total_epochs: int
    main_epochs: int
    adam_warmup_epochs: int

    @property
    def run_name(self) -> str:
        return f"{self.strategy}_{self.seed_label}"

    @property
    def training_strategy(self) -> str:
        return "multi_phase" if self.adam_warmup_epochs > 0 else "single_optimizer"

    @property
    def is_experimental_strategy(self) -> bool:
        return self.strategy in EXPERIMENTAL_STRATEGY_SET

    @property
    def strategy_group(self) -> str:
        return "experimental" if self.is_experimental_strategy else "core"

    def wandb_tags(self) -> list[str]:
        return [
            self.strategy,
            self.strategy_group,
            self.optimizer.lower(),
            self.training_strategy,
            self.seed_label,
        ]


def _parse_strategies(raw: str | None, *, include_experimental: bool) -> list[str]:
    if raw:
        strategies = parse_csv_list(raw)
    else:
        strategies = list(CORE_STRATEGIES)
        if include_experimental:
            strategies.extend(EXPERIMENTAL_STRATEGIES)
    normalized = [item.strip().lower().replace("-", "_") for item in strategies if item.strip()]
    if not normalized:
        raise ValueError("Expected at least one optimizer strategy.")
    unsupported = [item for item in normalized if item not in STRATEGY_TO_OPTIMIZER]
    if unsupported:
        raise ValueError(
            f"Unsupported optimizer strategy/strategies: {', '.join(unsupported)}. "
            f"Use one of: {', '.join(ALL_STRATEGIES)}."
        )
    return normalized


def _strategy_group(strategy: str) -> str:
    return "experimental" if strategy in EXPERIMENTAL_STRATEGY_SET else "core"


def _resolve_eval_inputs(args: argparse.Namespace) -> dict[str, str | None]:
    return _resolve_shared_eval_inputs(args, default_ood_eval_id=DEFAULT_OOD_EVAL_ID)


def _build_optimizer_run_specs(
    *,
    strategies: list[str],
    seed_pairs: list[tuple[str, int]],
    total_epochs: int,
    main_epochs_override: int | None,
    adam_warmup_epochs: int,
) -> list[OptimizerRunSpec]:
    specs: list[OptimizerRunSpec] = []
    for strategy in strategies:
        optimizer = STRATEGY_TO_OPTIMIZER[strategy]
        warmup_epochs = int(adam_warmup_epochs) if strategy in WARMUP_STRATEGIES else 0
        main_epochs = (
            int(main_epochs_override)
            if main_epochs_override is not None
            else int(total_epochs) - warmup_epochs
        )
        if main_epochs <= 0:
            raise ValueError(
                f"Strategy '{strategy}' has no main optimizer epochs. "
                "Increase --total-epochs or reduce --adam-warmup-epochs."
            )
        run_total_epochs = warmup_epochs + main_epochs
        for seed_label, seed_value in seed_pairs:
            specs.append(
                OptimizerRunSpec(
                    strategy=strategy,
                    optimizer=optimizer,
                    seed_label=seed_label,
                    seed_value=int(seed_value),
                    total_epochs=int(run_total_epochs),
                    main_epochs=int(main_epochs),
                    adam_warmup_epochs=warmup_epochs,
                )
            )
    return specs


def _optimizer_lr(
    *,
    optimizer: str,
    adam_lr: float,
    soap_lr: float,
    lbfgs_lr: float,
    bfgs_lr: float,
    ssbfgs_lr: float,
    ssbroyden_lr: float,
    quasi_newton_lr: float | None,
    stochastic_qn_lr: float,
) -> float:
    if optimizer == "Adam":
        return float(adam_lr)
    if optimizer == "SOAP":
        return float(soap_lr)
    if optimizer in FULL_BATCH_OPTIMIZERS and quasi_newton_lr is not None:
        return float(quasi_newton_lr)
    if optimizer == "LBFGS":
        return float(lbfgs_lr)
    if optimizer == "BFGS":
        return float(bfgs_lr)
    if optimizer == "SSBFGS":
        return float(ssbfgs_lr)
    if optimizer == "SSBroyden":
        return float(ssbroyden_lr)
    if optimizer in STOCHASTIC_QN_OPTIMIZERS:
        return float(stochastic_qn_lr)
    return float(adam_lr)


def _optimizer_lr_source(*, optimizer: str, quasi_newton_lr: float | None) -> str:
    if optimizer in FULL_BATCH_OPTIMIZERS and quasi_newton_lr is not None:
        return "quasi_newton_override"
    if optimizer in CALIBRATED_OPTIMIZER_LRS:
        return "calibrated_default"
    if optimizer in STOCHASTIC_QN_OPTIMIZERS:
        return "stochastic_qn_default"
    return "explicit_or_default"


def _optimizer_stage(
    *,
    optimizer: str,
    name: str,
    epochs: int,
    lr: float,
    batch_size: int,
    line_search_name: str,
    stochastic_curvature_threshold: float,
    stochastic_init_hessian_scale: float,
    adam_scheduler: bool,
    adam_scheduler_metric: str,
    adam_scheduler_factor: float,
    adam_scheduler_patience: int,
    adam_scheduler_threshold: float,
) -> str:
    optimizer_kwargs = "{}"
    line_search = "null"
    scheduler = "null"
    batch_size_value = str(int(batch_size))
    shuffle = "true"
    full_batch = "false"
    allow_sampling = "false"

    if optimizer == "Adam" and adam_scheduler:
        scheduler = (
            "{"
            "name:reduce_on_plateau,"
            f"metric:{adam_scheduler_metric},"
            "mode:min,"
            f"factor:{adam_scheduler_factor},"
            f"patience:{int(adam_scheduler_patience)},"
            f"threshold:{adam_scheduler_threshold},"
            "threshold_mode:rel,"
            "cooldown:0,"
            "min_lr:0.0,"
            "eps:1.0e-8"
            "}"
        )
    if optimizer == "Adam":
        optimizer_kwargs = "{eps:1.0e-6}"
    elif optimizer in FULL_BATCH_OPTIMIZERS:
        batch_size_value = "null"
        shuffle = "false"
        full_batch = "true"
        line_search = f"{{name:{line_search_name}}}"
        if optimizer == "LBFGS":
            line_search = "{name:strong_wolfe}"
        elif optimizer == "BFGS":
            optimizer_kwargs = "{curvature_eps:1.0e-12,init_hessian_scale:1.0}"
        elif optimizer == "SSBFGS":
            optimizer_kwargs = "{tau_strategy:al_baali}"
        elif optimizer == "SSBroyden":
            optimizer_kwargs = "{tau_strategy:paper_default,phi_strategy:paper_default}"
    elif optimizer in STOCHASTIC_QN_OPTIMIZERS:
        if optimizer == "sSSBFGS":
            optimizer_kwargs = (
                "{"
                f"curvature_threshold:{stochastic_curvature_threshold},"
                f"init_hessian_scale:{stochastic_init_hessian_scale},"
                "tau_strategy:al_baali"
                "}"
            )
        elif optimizer == "sSSBroyden":
            optimizer_kwargs = (
                "{"
                f"curvature_threshold:{stochastic_curvature_threshold},"
                f"init_hessian_scale:{stochastic_init_hessian_scale},"
                "tau_strategy:paper_default,phi_strategy:paper_default"
                "}"
            )

    return (
        "{"
        f"name:{name},"
        f"optimizer:{optimizer},"
        f"lr:{lr},"
        f"epochs:{int(epochs)},"
        f"batch_size:{batch_size_value},"
        f"shuffle:{shuffle},"
        f"full_batch:{full_batch},"
        f"allow_sampling:{allow_sampling},"
        f"optimizer_kwargs:{optimizer_kwargs},"
        f"scheduler:{scheduler},"
        f"line_search:{line_search},"
        "convergence:null"
        "}"
    )


def _optimizer_phase_override(
    *,
    run_spec: OptimizerRunSpec,
    adam_lr: float,
    soap_lr: float,
    lbfgs_lr: float,
    bfgs_lr: float,
    ssbfgs_lr: float,
    ssbroyden_lr: float,
    quasi_newton_lr: float | None,
    stochastic_qn_lr: float,
    batch_size: int,
    line_search_name: str,
    stochastic_curvature_threshold: float,
    stochastic_init_hessian_scale: float,
    adam_scheduler: bool,
    adam_scheduler_metric: str,
    adam_scheduler_factor: float,
    adam_scheduler_patience: int,
    adam_scheduler_threshold: float,
) -> str:
    phases: list[str] = []
    if run_spec.adam_warmup_epochs > 0:
        phases.append(
            _optimizer_stage(
                optimizer="Adam",
                name="adam_warmup",
                epochs=run_spec.adam_warmup_epochs,
                lr=_optimizer_lr(
                    optimizer="Adam",
                    adam_lr=adam_lr,
                    soap_lr=soap_lr,
                    lbfgs_lr=lbfgs_lr,
                    bfgs_lr=bfgs_lr,
                    ssbfgs_lr=ssbfgs_lr,
                    ssbroyden_lr=ssbroyden_lr,
                    quasi_newton_lr=quasi_newton_lr,
                    stochastic_qn_lr=stochastic_qn_lr,
                ),
                batch_size=batch_size,
                line_search_name=line_search_name,
                stochastic_curvature_threshold=stochastic_curvature_threshold,
                stochastic_init_hessian_scale=stochastic_init_hessian_scale,
                adam_scheduler=adam_scheduler,
                adam_scheduler_metric=adam_scheduler_metric,
                adam_scheduler_factor=adam_scheduler_factor,
                adam_scheduler_patience=adam_scheduler_patience,
                adam_scheduler_threshold=adam_scheduler_threshold,
            )
        )
    phases.append(
        _optimizer_stage(
            optimizer=run_spec.optimizer,
            name=run_spec.strategy,
            epochs=run_spec.main_epochs,
            lr=_optimizer_lr(
                optimizer=run_spec.optimizer,
                adam_lr=adam_lr,
                soap_lr=soap_lr,
                lbfgs_lr=lbfgs_lr,
                bfgs_lr=bfgs_lr,
                ssbfgs_lr=ssbfgs_lr,
                ssbroyden_lr=ssbroyden_lr,
                quasi_newton_lr=quasi_newton_lr,
                stochastic_qn_lr=stochastic_qn_lr,
            ),
            batch_size=batch_size,
            line_search_name=line_search_name,
            stochastic_curvature_threshold=stochastic_curvature_threshold,
            stochastic_init_hessian_scale=stochastic_init_hessian_scale,
            adam_scheduler=adam_scheduler,
            adam_scheduler_metric=adam_scheduler_metric,
            adam_scheduler_factor=adam_scheduler_factor,
            adam_scheduler_patience=adam_scheduler_patience,
            adam_scheduler_threshold=adam_scheduler_threshold,
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
    device: str,
    hidden_dim: int,
    hidden_layers: int,
    activation: str,
    dtype_name: str,
    batch_size: int,
    adam_lr: float,
    soap_lr: float,
    lbfgs_lr: float,
    bfgs_lr: float,
    ssbfgs_lr: float,
    ssbroyden_lr: float,
    quasi_newton_lr: float | None,
    stochastic_qn_lr: float,
    run_spec: OptimizerRunSpec,
    line_search_name: str,
    stochastic_curvature_threshold: float,
    stochastic_init_hessian_scale: float,
    adam_scheduler: bool,
    adam_scheduler_metric: str,
    adam_scheduler_factor: float,
    adam_scheduler_patience: int,
    adam_scheduler_threshold: float,
    log_every_epoch: int,
    loss_weight_data: float,
    loss_weight_dt: float,
    loss_weight_physics: float,
    loss_weight_ic: float,
    gradient_telemetry: bool,
    save_best_checkpoint: bool,
    save_last_checkpoint: bool,
    save_init_checkpoint: bool,
    id_eval_root: str | None,
    ood_eval_root: str | None,
) -> list[str]:
    phase_override = _optimizer_phase_override(
        run_spec=run_spec,
        adam_lr=adam_lr,
        soap_lr=soap_lr,
        lbfgs_lr=lbfgs_lr,
        bfgs_lr=bfgs_lr,
        ssbfgs_lr=ssbfgs_lr,
        ssbroyden_lr=ssbroyden_lr,
        quasi_newton_lr=quasi_newton_lr,
        stochastic_qn_lr=stochastic_qn_lr,
        batch_size=batch_size,
        line_search_name=line_search_name,
        stochastic_curvature_threshold=stochastic_curvature_threshold,
        stochastic_init_hessian_scale=stochastic_init_hessian_scale,
        adam_scheduler=adam_scheduler,
        adam_scheduler_metric=adam_scheduler_metric,
        adam_scheduler_factor=adam_scheduler_factor,
        adam_scheduler_patience=adam_scheduler_patience,
        adam_scheduler_threshold=adam_scheduler_threshold,
    )
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
        "pinn.supervised_sampling.enabled=false",
        "pinn.collocation.sampling.enabled=false",
        "pinn.evaluation.frequency=25",
        f"pinn.gradient_telemetry.enabled={'true' if gradient_telemetry else 'false'}",
        f"pinn.checkpointing.enabled={'true' if save_best_checkpoint or save_last_checkpoint or save_init_checkpoint else 'false'}",
        f"pinn.checkpointing.save_best={'true' if save_best_checkpoint else 'false'}",
        f"pinn.checkpointing.save_last={'true' if save_last_checkpoint else 'false'}",
        f"pinn.checkpointing.save_init={'true' if save_init_checkpoint else 'false'}",
        "pinn.checkpointing.epoch_fractions=[]",
        f"pinn.loss_weights.data={loss_weight_data}",
        f"pinn.loss_weights.dt={loss_weight_dt}",
        f"pinn.loss_weights.physics={loss_weight_physics}",
        f"pinn.loss_weights.ic={loss_weight_ic}",
        "wandb.use=true",
        f"wandb.project={wandb_project}",
        f"wandb.group={wandb_group}",
        f"wandb.name={wandb_name}",
        f"wandb.tags={format_hydra_list(wandb_tags)}",
        f"logging.log_every_epoch={int(log_every_epoch)}",
        phase_override,
    ]
    if wandb_entity:
        command.append(f"wandb.entity={wandb_entity}")
    if id_eval_root:
        command.append(f"evaluation.id.root={id_eval_root}")
    if ood_eval_root:
        command.append(f"evaluation.ood.root={ood_eval_root}")
    return command


def _best_checkpoint_split_metrics(metrics: dict[str, Any] | None, split: str) -> dict[str, Any]:
    if not metrics:
        return {}
    best = metrics.get("best_checkpoint_metrics")
    if not isinstance(best, dict):
        return {}
    entry = best.get(split)
    if not isinstance(entry, dict):
        return {}
    values = entry.get("metrics")
    return dict(values) if isinstance(values, dict) else {}


def _best_checkpoint_eval_metrics(metrics: dict[str, Any] | None, kind: str) -> dict[str, Any]:
    if not metrics:
        return {}
    best = metrics.get("best_checkpoint_metrics")
    if not isinstance(best, dict):
        return {}
    evaluation_sets = best.get("evaluation_sets")
    if not isinstance(evaluation_sets, dict):
        return {}
    entry = evaluation_sets.get(kind)
    if not isinstance(entry, dict):
        return {}
    values = entry.get("metrics")
    return dict(values) if isinstance(values, dict) else {}


def _best_checkpoint_selection(metrics: dict[str, Any] | None) -> dict[str, Any]:
    if not metrics:
        return {}
    best = metrics.get("best_checkpoint_metrics")
    if not isinstance(best, dict):
        return {}
    selection = best.get("selection")
    return dict(selection) if isinstance(selection, dict) else {}


def _summary_row_for_run(run: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(str(run.get("run_dir", "")))
    metrics = _read_json_if_exists(run_dir / "metrics.json") if run_dir else None
    timings = _read_json_if_exists(run_dir / "timings.json") if run_dir else None
    final_train_metrics = dict(metrics.get("final_train_metrics", {}) or {}) if metrics else {}
    final_val_metrics = dict(metrics.get("final_val_metrics", {}) or {}) if metrics else {}
    final_test_metrics = dict(metrics.get("final_test_metrics", {}) or {}) if metrics else {}
    final_train_losses = dict(metrics.get("final_train_losses", {}) or {}) if metrics else {}
    final_epoch = dict(metrics.get("final_epoch", {}) or {}) if metrics else {}
    id_eval_metrics = _eval_metrics(metrics, "id")
    ood_eval_metrics = _eval_metrics(metrics, "ood")
    best_selection = _best_checkpoint_selection(metrics)
    best_train_metrics = _best_checkpoint_split_metrics(metrics, "train")
    best_val_metrics = _best_checkpoint_split_metrics(metrics, "val")
    best_test_metrics = _best_checkpoint_split_metrics(metrics, "test")
    best_ood_eval_metrics = _best_checkpoint_eval_metrics(metrics, "ood")
    id_eval_rmse = _float_or_none(id_eval_metrics.get("rmse"))
    ood_eval_rmse = _float_or_none(ood_eval_metrics.get("rmse"))
    final_test_rmse = _float_or_none(final_test_metrics.get("rmse"))
    best_test_rmse = _float_or_none(best_test_metrics.get("rmse"))
    best_ood_eval_rmse = _float_or_none(best_ood_eval_metrics.get("rmse"))
    strategy = str(run.get("strategy", ""))
    is_experimental = run.get("is_experimental_strategy")
    if is_experimental is None:
        is_experimental = strategy in EXPERIMENTAL_STRATEGY_SET
    return {
        "run_name": run.get("run_name"),
        "strategy": run.get("strategy"),
        "strategy_group": run.get("strategy_group") or _strategy_group(strategy),
        "is_experimental_strategy": bool(is_experimental),
        "optimizer": run.get("optimizer"),
        "training_strategy": run.get("training_strategy"),
        "seed_label": run.get("seed_label"),
        "seed_value": run.get("seed_value"),
        "total_epochs": _int_or_none(run.get("total_epochs")),
        "main_epochs": _int_or_none(run.get("main_epochs")),
        "adam_warmup_epochs": _int_or_none(run.get("adam_warmup_epochs")),
        "adam_lr": _float_or_none(run.get("adam_lr")),
        "main_lr": _float_or_none(run.get("main_lr")),
        "lr_source": run.get("lr_source"),
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
        "final_train_mse": _float_or_none(final_train_metrics.get("mse")),
        "final_train_rmse": _float_or_none(final_train_metrics.get("rmse")),
        "final_train_mae": _float_or_none(final_train_metrics.get("mae")),
        "final_val_mse": _float_or_none(final_val_metrics.get("mse")),
        "final_val_rmse": _float_or_none(final_val_metrics.get("rmse")),
        "final_val_mae": _float_or_none(final_val_metrics.get("mae")),
        "final_test_mse": _float_or_none(final_test_metrics.get("mse")),
        "final_test_rmse": final_test_rmse,
        "final_test_mae": _float_or_none(final_test_metrics.get("mae")),
        "id_eval_mse": _float_or_none(id_eval_metrics.get("mse")),
        "id_eval_rmse": id_eval_rmse,
        "id_eval_mae": _float_or_none(id_eval_metrics.get("mae")),
        "ood_eval_mse": _float_or_none(ood_eval_metrics.get("mse")),
        "ood_eval_rmse": ood_eval_rmse,
        "ood_eval_mae": _float_or_none(ood_eval_metrics.get("mae")),
        "id_ood_rmse_gap": None if id_eval_rmse is None or ood_eval_rmse is None else ood_eval_rmse - id_eval_rmse,
        "test_ood_rmse_gap": None if final_test_rmse is None or ood_eval_rmse is None else ood_eval_rmse - final_test_rmse,
        "best_checkpoint_epoch": _int_or_none(best_selection.get("global_epoch")),
        "best_checkpoint_wall_seconds": _float_or_none(best_selection.get("cumulative_wall_seconds")),
        "best_checkpoint_phase_name": best_selection.get("phase_name"),
        "best_checkpoint_train_mse": _float_or_none(best_train_metrics.get("mse")),
        "best_checkpoint_train_rmse": _float_or_none(best_train_metrics.get("rmse")),
        "best_checkpoint_train_mae": _float_or_none(best_train_metrics.get("mae")),
        "best_checkpoint_val_mse": _float_or_none(best_val_metrics.get("mse")),
        "best_checkpoint_val_rmse": _float_or_none(best_val_metrics.get("rmse")),
        "best_checkpoint_val_mae": _float_or_none(best_val_metrics.get("mae")),
        "best_checkpoint_test_mse": _float_or_none(best_test_metrics.get("mse")),
        "best_checkpoint_test_rmse": best_test_rmse,
        "best_checkpoint_test_mae": _float_or_none(best_test_metrics.get("mae")),
        "best_checkpoint_ood_eval_mse": _float_or_none(best_ood_eval_metrics.get("mse")),
        "best_checkpoint_ood_eval_rmse": best_ood_eval_rmse,
        "best_checkpoint_ood_eval_mae": _float_or_none(best_ood_eval_metrics.get("mae")),
        "best_checkpoint_test_ood_rmse_gap": None
        if best_test_rmse is None or best_ood_eval_rmse is None
        else best_ood_eval_rmse - best_test_rmse,
        "final_train_total_loss": _float_or_none(final_train_losses.get("total_loss")),
        "final_train_data_loss": _component_loss(final_train_losses, "data"),
        "final_train_physics_loss": _component_loss(final_train_losses, "physics"),
        "final_train_dt_loss": _component_loss(final_train_losses, "dt"),
        "final_train_ic_loss": _component_loss(final_train_losses, "ic"),
        "final_test_losses": _json_or_none(metrics.get("final_test_losses") if metrics else None),
        "test_data_loss": _float_or_none(final_epoch.get("test_data_loss")),
        "global_epoch": _int_or_none(final_epoch.get("global_epoch")),
        "phase_name": final_epoch.get("phase_name"),
        "num_train_steps": _int_or_none(final_epoch.get("num_train_steps")),
        "epoch_wall_seconds": _float_or_none(final_epoch.get("epoch_wall_seconds")),
        "cumulative_wall_seconds": _float_or_none(final_epoch.get("cumulative_wall_seconds")),
        "total_seconds": _float_or_none(timings.get("total_seconds") if timings else None),
        "training_seconds": _float_or_none(timings.get("training_seconds") if timings else None),
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
        "mode": manifest.get("experiment", {}).get("mode"),
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-bin", default=sys.executable, help="Python executable for pipeline and training.")
    parser.add_argument("--mode", default="screening", choices=["screening", "final"], help="Thesis optimizer experiment mode.")
    parser.add_argument("--profile", choices=["smoke", "benchmark"], help="Deprecated alias: smoke->screening, benchmark->final.")
    parser.add_argument("--experiment-tag", default=None, help="Output/W&B tag suffix. Default: timestamp.")
    parser.add_argument("--output-root", default=None, help="Explicit output root. Default: outputs/pinn/optimizer_comparison/<tag>.")
    parser.add_argument("--model-flag", default="SM4", help="Model flag.")
    parser.add_argument("--reference-id", default=None, help=f"Reference dataset ID. Default: {DEFAULT_REFERENCE_ID}.")
    parser.add_argument("--dataset-root", default=None, help="Explicit preprocessed dataset root. Mutually exclusive with --reference-id.")
    parser.add_argument("--id-eval-id", default=None, help="Optional ID evaluation dataset ID from data/evaluation/index.json.")
    parser.add_argument("--ood-eval-id", default=None, help="Optional OOD evaluation dataset ID from data/evaluation/index.json.")
    parser.add_argument("--id-eval-root", default=None, help="Optional explicit ID evaluation preprocessed dataset root.")
    parser.add_argument("--ood-eval-root", default=None, help="Optional explicit OOD evaluation preprocessed dataset root.")
    parser.add_argument("--no-ood-eval", action="store_true", help=f"Disable default OOD evaluation ({DEFAULT_OOD_EVAL_ID}).")
    parser.add_argument("--allow-dataset-generation", action="store_true", help="Use the old shared dataset-generation fallback instead of reference lookup.")
    parser.add_argument("--preset", default="main", help="Dataset pipeline preset used only with --allow-dataset-generation.")
    parser.add_argument("--budget", default="b512", help="Dataset budget used only with --allow-dataset-generation.")
    parser.add_argument("--dataset-seed", default="ds01", help="Dataset seed used only with --allow-dataset-generation.")
    parser.add_argument(
        "--seed-labels",
        default=None,
        help="Preferred: comma-separated training seed labels from src/config/registry/seeds.yaml. Defaults by --mode.",
    )
    parser.add_argument("--seeds", default=None, help="Raw integer seed override. Prefer --seed-labels for registry-backed runs.")
    parser.add_argument("--seed", type=int, default=None, help="Deprecated single raw integer seed alias.")
    parser.add_argument(
        "--strategies",
        default=None,
        help=(
            "Comma-separated optimizer strategies. Defaults exclude unstable experimental strategies "
            f"({', '.join(EXPERIMENTAL_STRATEGIES)}). Explicit --strategies may still include them."
        ),
    )
    parser.add_argument(
        "--include-experimental-strategies",
        action="store_true",
        help=(
            "Include unstable stochastic self-scaled strategies in the default matrix. "
            f"Experimental strategies: {', '.join(EXPERIMENTAL_STRATEGIES)}."
        ),
    )
    parser.add_argument("--device", default="cuda", help="PINN device override.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="PINN hidden width.")
    parser.add_argument("--hidden-layers", type=int, default=4, help="PINN hidden depth.")
    parser.add_argument("--activation", default="tanh", help="PINN activation.")
    parser.add_argument("--dtype", default="float64", help="PINN dtype.")
    parser.add_argument("--batch-size", type=int, default=1024, help="Mini-batch size for Adam/SOAP/stochastic phases.")
    parser.add_argument("--adam-lr", type=float, default=DEFAULT_ADAM_LR, help="Adam learning rate.")
    parser.add_argument("--soap-lr", type=float, default=CALIBRATED_OPTIMIZER_LRS["SOAP"], help="SOAP learning rate.")
    parser.add_argument(
        "--total-epochs",
        type=int,
        default=None,
        help=(
            "Preferred epoch control: total epochs per run. Single-optimizer runs use all epochs; "
            "Adam -> optimizer runs use --adam-warmup-epochs plus the remaining epochs. "
            "Defaults: screening=10, final=5000."
        ),
    )
    parser.add_argument(
        "--main-epochs",
        type=int,
        default=None,
        help=(
            "Compatibility override for each strategy's main optimizer phase. "
            "If set, multi-phase total epochs become --adam-warmup-epochs + --main-epochs."
        ),
    )
    parser.add_argument("--adam-warmup-epochs", type=int, default=None, help="Adam warm-up epochs for Adam -> optimizer strategies. Defaults: screening=5, final=500.")
    parser.add_argument("--adam-scheduler", action=argparse.BooleanOptionalAction, default=True, help="Use ReduceLROnPlateau on Adam phases.")
    parser.add_argument("--adam-scheduler-metric", default="train_total_loss", choices=["train_total_loss", "val_total_loss"], help="Metric for Adam ReduceLROnPlateau.")
    parser.add_argument("--adam-scheduler-factor", type=float, default=0.5, help="Adam scheduler reduction factor.")
    parser.add_argument("--adam-scheduler-patience", type=int, default=100, help="Adam scheduler patience in epochs.")
    parser.add_argument("--adam-scheduler-threshold", type=float, default=1.0e-4, help="Adam scheduler relative threshold.")
    parser.add_argument(
        "--lbfgs-lr",
        type=float,
        default=CALIBRATED_OPTIMIZER_LRS["LBFGS"],
        help="LBFGS learning rate.",
    )
    parser.add_argument(
        "--bfgs-lr",
        type=float,
        default=CALIBRATED_OPTIMIZER_LRS["BFGS"],
        help="BFGS learning rate.",
    )
    parser.add_argument(
        "--ssbfgs-lr",
        type=float,
        default=CALIBRATED_OPTIMIZER_LRS["SSBFGS"],
        help="SSBFGS learning rate.",
    )
    parser.add_argument(
        "--ssbroyden-lr",
        type=float,
        default=CALIBRATED_OPTIMIZER_LRS["SSBroyden"],
        help="SSBroyden learning rate.",
    )
    parser.add_argument(
        "--quasi-newton-lr",
        type=float,
        default=None,
        help="Optional compatibility override applied to all full-batch quasi-Newton phases.",
    )
    parser.add_argument("--stochastic-qn-lr", type=float, default=5.0e-2, help="Learning rate for stochastic quasi-Newton phases.")
    parser.add_argument("--line-search", default="strong_wolfe", choices=["strong_wolfe", "backtracking"], help="Line-search method for BFGS-family phases.")
    parser.add_argument("--stochastic-curvature-threshold", type=float, default=1.0e-6)
    parser.add_argument("--stochastic-init-hessian-scale", type=float, default=1.0e-1)
    parser.add_argument("--wandb-project", default=None, help="Override W&B project. Defaults by --mode.")
    parser.add_argument("--wandb-entity", default=None, help="Optional W&B entity.")
    parser.add_argument("--loss-weight-data", type=float, default=1.0, help="Static supervised loss weight.")
    parser.add_argument("--loss-weight-dt", type=float, default=1.0e-4, help="Static dt loss weight.")
    parser.add_argument("--loss-weight-physics", type=float, default=1.0e-4, help="Static physics loss weight.")
    parser.add_argument("--loss-weight-ic", type=float, default=1.0e-3, help="Static IC loss weight.")
    parser.add_argument("--gradient-telemetry", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-best-checkpoint", action=argparse.BooleanOptionalAction, default=True, help="Write best.pt for each run.")
    parser.add_argument("--save-last-checkpoint", action=argparse.BooleanOptionalAction, default=False, help="Write/overwrite last.pt for each run.")
    parser.add_argument("--save-init-checkpoint", action=argparse.BooleanOptionalAction, default=False, help="Write init.pt for each run.")
    parser.add_argument("--log-every-epoch", type=int, default=1, help="Metric/W&B logging cadence.")
    parser.add_argument("--tag", action="append", default=[], help="Optional extra W&B tag. Can be passed multiple times.")
    parser.add_argument("--stage1-override", action="append", default=[], help="Extra stage-1 override for --allow-dataset-generation.")
    parser.add_argument("--stage2-override", action="append", default=[], help="Extra stage-2 override for --allow-dataset-generation.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    return parser


def _resolve_mode(args: argparse.Namespace) -> str:
    if args.profile == "smoke":
        return "screening"
    if args.profile == "benchmark":
        return "final"
    return str(args.mode)


def _resolve_seed_pairs(args: argparse.Namespace, *, mode: str) -> list[tuple[str, int]]:
    if args.seed_labels:
        return _seed_pairs_from_labels(_parse_label_list(str(args.seed_labels)))
    if args.seeds:
        return raw_seed_pairs(_parse_int_list(str(args.seeds)))
    if args.seed is not None:
        return [(f"raw{int(args.seed)}", int(args.seed))]
    default_labels = list(SCREENING_SEED_LABELS if mode == "screening" else FINAL_SEED_LABELS)
    return _seed_pairs_from_labels(default_labels)


def _resolve_epoch_controls(args: argparse.Namespace, *, mode: str) -> tuple[int, int | None, int]:
    default_total_epochs = 10 if mode == "screening" else 5000
    default_warmup_epochs = 5 if mode == "screening" else 500
    total_epochs = default_total_epochs if args.total_epochs is None else int(args.total_epochs)
    main_epochs_override = None if args.main_epochs is None else int(args.main_epochs)
    adam_warmup_epochs = default_warmup_epochs if args.adam_warmup_epochs is None else int(args.adam_warmup_epochs)
    if total_epochs <= 0:
        raise ValueError("--total-epochs must be positive.")
    if main_epochs_override is not None and main_epochs_override <= 0:
        raise ValueError("--main-epochs must be positive when provided.")
    if adam_warmup_epochs < 0:
        raise ValueError("--adam-warmup-epochs must be non-negative.")
    if main_epochs_override is None and adam_warmup_epochs >= total_epochs:
        raise ValueError("--adam-warmup-epochs must be smaller than --total-epochs.")
    return total_epochs, main_epochs_override, adam_warmup_epochs


def _resolve_dataset(args: argparse.Namespace, *, manifest: dict[str, Any], manifest_path: Path) -> tuple[Path, str | None]:
    if args.dataset_root and args.reference_id and not args.allow_dataset_generation:
        raise ValueError("--dataset-root and --reference-id are mutually exclusive.")
    if args.dataset_root and args.allow_dataset_generation:
        raise ValueError("--dataset-root and --allow-dataset-generation are mutually exclusive.")
    if args.reference_id and args.allow_dataset_generation:
        raise ValueError("--reference-id and --allow-dataset-generation are mutually exclusive.")

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

    dataset_command, dataset_pipeline_root = build_dataset_pipeline_command(
        python_bin=args.python_bin,
        method="qbc_deep_ensemble",
        experiment_id=str(manifest["experiment"]["id"]),
        preset=str(args.preset),
        budget=str(args.budget),
        dataset_seed=str(args.dataset_seed),
        model_flag=str(args.model_flag),
        run_root=Path(str(manifest["run_root"])),
        stage1_overrides=list(args.stage1_override),
        stage2_overrides=list(args.stage2_override),
    )
    rc = run_logged_stage(
        label="optimizer-comparison",
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


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    mode = _resolve_mode(args)
    try:
        seed_pairs = _resolve_seed_pairs(args, mode=mode)
        strategies = _parse_strategies(
            args.strategies,
            include_experimental=bool(args.include_experimental_strategies),
        )
        total_epochs, main_epochs_override, adam_warmup_epochs = _resolve_epoch_controls(args, mode=mode)
        eval_inputs = _resolve_eval_inputs(args)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from None
    wandb_project = args.wandb_project or default_project_for_mode(
        mode=mode,
        screening_project="thesis-optimizer-experiment-TEST",
        final_project="thesis-optimizer-experiment",
    )

    stamp = args.experiment_tag or tag_stamp()
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else (REPO_ROOT / "outputs" / "pinn" / "optimizer_comparison" / stamp).resolve()
    )
    output_root.mkdir(parents=True, exist_ok=True)

    experiment_id = f"optimizer_comparison_{mode}_{stamp}"
    wandb_group = group_name("optimizer_comparison", args.model_flag.lower(), mode, stamp)
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
    manifest["artifacts"]["core_strategies"] = list(CORE_STRATEGIES)
    manifest["artifacts"]["experimental_strategies"] = list(EXPERIMENTAL_STRATEGIES)
    manifest["artifacts"]["include_experimental_strategies"] = bool(args.include_experimental_strategies)
    manifest["artifacts"]["seed_labels"] = [label for label, _ in seed_pairs]
    manifest["artifacts"]["seed_values"] = [value for _, value in seed_pairs]
    manifest["artifacts"]["total_epochs"] = int(total_epochs)
    manifest["artifacts"]["main_epochs_override"] = main_epochs_override
    manifest["artifacts"]["adam_warmup_epochs"] = int(adam_warmup_epochs)
    manifest["artifacts"]["adam_scheduler"] = bool(args.adam_scheduler)
    manifest["artifacts"]["adam_scheduler_metric"] = args.adam_scheduler_metric
    manifest["artifacts"]["adam_scheduler_factor"] = float(args.adam_scheduler_factor)
    manifest["artifacts"]["adam_scheduler_patience"] = int(args.adam_scheduler_patience)
    manifest["artifacts"]["adam_scheduler_threshold"] = float(args.adam_scheduler_threshold)
    manifest["artifacts"]["optimizer_lr_defaults"] = dict(CALIBRATED_OPTIMIZER_LRS)
    manifest["artifacts"]["adam_lr"] = float(args.adam_lr)
    manifest["artifacts"]["soap_lr"] = float(args.soap_lr)
    manifest["artifacts"]["lbfgs_lr"] = float(args.lbfgs_lr)
    manifest["artifacts"]["bfgs_lr"] = float(args.bfgs_lr)
    manifest["artifacts"]["ssbfgs_lr"] = float(args.ssbfgs_lr)
    manifest["artifacts"]["ssbroyden_lr"] = float(args.ssbroyden_lr)
    manifest["artifacts"]["quasi_newton_lr_override"] = (
        None if args.quasi_newton_lr is None else float(args.quasi_newton_lr)
    )
    manifest["artifacts"]["stochastic_qn_lr"] = float(args.stochastic_qn_lr)
    manifest["artifacts"]["gradient_telemetry"] = bool(args.gradient_telemetry)
    manifest["artifacts"]["save_best_checkpoint"] = bool(args.save_best_checkpoint)
    manifest["artifacts"]["save_last_checkpoint"] = bool(args.save_last_checkpoint)
    manifest["artifacts"]["save_init_checkpoint"] = bool(args.save_init_checkpoint)
    manifest["artifacts"]["id_eval_id"] = eval_inputs["id_eval_id"]
    manifest["artifacts"]["id_eval_root"] = eval_inputs["id_eval_root"]
    manifest["artifacts"]["ood_eval_id"] = eval_inputs["ood_eval_id"]
    manifest["artifacts"]["ood_eval_root"] = eval_inputs["ood_eval_root"]
    manifest["artifacts"]["training_strategies"] = ["single_optimizer", "multi_phase"]
    save_manifest(str(manifest_path), manifest)

    try:
        dataset_root, dataset_reference_id = _resolve_dataset(args, manifest=manifest, manifest_path=manifest_path)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from None
    tags_base = wandb_tags_list(
        "optimizer_comparison",
        mode,
        args.model_flag.lower(),
        "reference" if dataset_reference_id else None,
        dataset_reference_id,
        args.tag,
    )
    run_specs = _build_optimizer_run_specs(
        strategies=strategies,
        seed_pairs=seed_pairs,
        total_epochs=int(total_epochs),
        main_epochs_override=main_epochs_override,
        adam_warmup_epochs=int(adam_warmup_epochs),
    )

    for run_spec in run_specs:
        run_name = run_spec.run_name
        run_dir = output_root / "runs" / run_name
        wandb_tags = [*tags_base, *run_spec.wandb_tags()]
        main_lr = _optimizer_lr(
            optimizer=run_spec.optimizer,
            adam_lr=float(args.adam_lr),
            soap_lr=float(args.soap_lr),
            lbfgs_lr=float(args.lbfgs_lr),
            bfgs_lr=float(args.bfgs_lr),
            ssbfgs_lr=float(args.ssbfgs_lr),
            ssbroyden_lr=float(args.ssbroyden_lr),
            quasi_newton_lr=args.quasi_newton_lr,
            stochastic_qn_lr=float(args.stochastic_qn_lr),
        )
        lr_source = _optimizer_lr_source(
            optimizer=run_spec.optimizer,
            quasi_newton_lr=args.quasi_newton_lr,
        )
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
            device=args.device,
            hidden_dim=args.hidden_dim,
            hidden_layers=args.hidden_layers,
            activation=args.activation,
            dtype_name=args.dtype,
            batch_size=args.batch_size,
            adam_lr=args.adam_lr,
            soap_lr=args.soap_lr,
            lbfgs_lr=args.lbfgs_lr,
            bfgs_lr=args.bfgs_lr,
            ssbfgs_lr=args.ssbfgs_lr,
            ssbroyden_lr=args.ssbroyden_lr,
            quasi_newton_lr=args.quasi_newton_lr,
            stochastic_qn_lr=args.stochastic_qn_lr,
            run_spec=run_spec,
            line_search_name=args.line_search,
            stochastic_curvature_threshold=args.stochastic_curvature_threshold,
            stochastic_init_hessian_scale=args.stochastic_init_hessian_scale,
            adam_scheduler=bool(args.adam_scheduler),
            adam_scheduler_metric=args.adam_scheduler_metric,
            adam_scheduler_factor=float(args.adam_scheduler_factor),
            adam_scheduler_patience=int(args.adam_scheduler_patience),
            adam_scheduler_threshold=float(args.adam_scheduler_threshold),
            log_every_epoch=args.log_every_epoch,
            loss_weight_data=args.loss_weight_data,
            loss_weight_dt=args.loss_weight_dt,
            loss_weight_physics=args.loss_weight_physics,
            loss_weight_ic=args.loss_weight_ic,
            gradient_telemetry=bool(args.gradient_telemetry),
            save_best_checkpoint=bool(args.save_best_checkpoint),
            save_last_checkpoint=bool(args.save_last_checkpoint),
            save_init_checkpoint=bool(args.save_init_checkpoint),
            id_eval_root=eval_inputs["id_eval_root"],
            ood_eval_root=eval_inputs["ood_eval_root"],
        )
        log_path = output_root / "logs" / "runs" / f"{run_name}.log"
        run_metadata = {
            "strategy": run_spec.strategy,
            "strategy_group": run_spec.strategy_group,
            "is_experimental_strategy": run_spec.is_experimental_strategy,
            "optimizer": run_spec.optimizer,
            "training_strategy": run_spec.training_strategy,
            "seed_label": run_spec.seed_label,
            "seed_value": run_spec.seed_value,
            "total_epochs": run_spec.total_epochs,
            "main_epochs": run_spec.main_epochs,
            "adam_warmup_epochs": run_spec.adam_warmup_epochs,
            "adam_lr": float(args.adam_lr),
            "main_lr": main_lr,
            "lr_source": lr_source,
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
            label="optimizer-comparison",
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
                print(f"[optimizer-comparison] summary_csv={summary_artifacts['summary_csv']}")
                print(f"[optimizer-comparison] summary_json={summary_artifacts['summary_json']}")
                print(f"[optimizer-comparison] failures_json={summary_artifacts['failures_json']}")
            raise SystemExit(int(result["return_code"]))

    if not args.dry_run:
        summary_artifacts = _write_summary_artifacts(output_root=output_root, manifest=manifest)
        manifest["artifacts"].update(summary_artifacts)
        save_manifest(str(manifest_path), manifest)
        print(f"[optimizer-comparison] summary_csv={summary_artifacts['summary_csv']}")
        print(f"[optimizer-comparison] summary_json={summary_artifacts['summary_json']}")
        print(f"[optimizer-comparison] failures_json={summary_artifacts['failures_json']}")
    else:
        print("[optimizer-comparison] dry-run: summary artifacts were not written")
    print(f"[optimizer-comparison] run_manifest={manifest_path}")


if __name__ == "__main__":
    main()
