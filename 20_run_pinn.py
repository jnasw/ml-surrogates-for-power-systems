"""Train a phase-1 PINN from preprocessed stage-2 outputs."""

from __future__ import annotations

import os
from time import monotonic

import hydra
import torch
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf

from src.data.runtime_validation import validate_pinn_runtime_dataset
from src.pinn.data import load_pinn_dataset_from_preprocessed_root, load_pinn_supervised_test_split_from_preprocessed_root
from src.pinn.logging import PinnLogger
from src.sim.ode.model_definitions import SynchronousMachineModels
from src.training.metrics import build_run_summary, regression_metrics_for_torch_model
from src.training.runtime import (
    build_direct_run_manifest,
    cfg_get,
    resolve_path_from_cwd,
    resolve_dataset_root,
    resolve_run_dir,
    save_resolved_config,
    utc_now_iso,
    write_json,
)
from src.training.trainer import PinnModel
from src.training.trainer import train_pinn


def _format_example_values(values, *, max_items: int = 8) -> list[float]:
    return [round(float(value), 6) for value in values[:max_items]]


def _model_io_names(config) -> tuple[list[str], list[str]]:
    guide_path = os.path.join(str(config.dirs.init_conditions_dir), "modellings_guide.yaml")
    guide = OmegaConf.load(guide_path)
    for model in guide:
        if str(model.get("name")) == str(config.model.model_flag):
            state_names = [str(value) for value in (model.get("keys") or [])]
            extra_names = [str(value) for value in (model.get("keys_ext") or [])]
            return ["time"] + state_names + extra_names, state_names
    return [], []


def _print_pinn_io_example(config, dataset) -> None:
    input_names, output_names = _model_io_names(config)
    train_x0 = dataset.train_x[0].detach().cpu()
    train_y0 = dataset.train_y[0].detach().cpu()
    print("[pinn] Input/output definition:")
    print(f"[pinn] input = row-wise [time, initial-condition/features...] {input_names}")
    print(f"[pinn] output = dynamic target state row {output_names}")
    print(
        "[pinn] example supervised row | "
        f"x_shape={tuple(train_x0.shape)} x={_format_example_values(train_x0)} "
        f"y_shape={tuple(train_y0.shape)} y={_format_example_values(train_y0)}"
    )
    if dataset.train_col_x is not None and int(dataset.train_col_x.shape[0]) > 0:
        col_x0 = dataset.train_col_x[0].detach().cpu()
        print(
            "[pinn] example collocation input | "
            f"x_shape={tuple(col_x0.shape)} x={_format_example_values(col_x0)} "
            "(no supervised y; physics residual target is implicit)"
        )
    if dataset.train_init_x is not None and int(dataset.train_init_x.shape[0]) > 0:
        init_x0 = dataset.train_init_x[0].detach().cpu()
        init_y0 = dataset.train_init_y[0].detach().cpu()
        print(
            "[pinn] example initial-condition row | "
            f"x_shape={tuple(init_x0.shape)} x={_format_example_values(init_x0)} "
            f"y_shape={tuple(init_y0.shape)} y={_format_example_values(init_y0)}"
        )


def _resolve_optional_eval_root(config, original_cwd: str, *, kind: str) -> str | None:
    if kind not in {"id", "ood"}:
        raise ValueError(f"Unsupported evaluation kind: {kind}")
    root_cfg = cfg_get(config, f"evaluation.{kind}.root", None)
    if root_cfg in (None, ""):
        root_cfg = cfg_get(config, f"{kind}_eval_root", None)
    if root_cfg in (None, ""):
        return None
    return resolve_path_from_cwd(str(root_cfg), original_cwd)


def _evaluate_external_sets(
    *,
    config,
    original_cwd: str,
    model,
    batch_size: int,
) -> dict[str, dict[str, object]] | None:
    evaluation_sets: dict[str, dict[str, object]] = {}
    for kind in ("id", "ood"):
        eval_root = _resolve_optional_eval_root(config, original_cwd, kind=kind)
        if eval_root is None:
            continue
        eval_x, eval_y = load_pinn_supervised_test_split_from_preprocessed_root(
            dataset_root=eval_root,
            dtype=str(config.pinn.dtype),
        )
        evaluation_sets[kind] = {
            "dataset_root": eval_root,
            "n_rows": int(eval_x.shape[0]),
            "metrics": regression_metrics_for_torch_model(
                model=model,
                x=eval_x,
                y=eval_y,
                batch_size=batch_size,
            ),
        }
    return evaluation_sets or None


def _split_row_count(x) -> int | None:
    return None if x is None else int(x.shape[0])


def _evaluate_best_checkpoint(
    *,
    checkpoint_path: str,
    config,
    original_cwd: str,
    dataset,
    batch_size: int,
) -> dict[str, object] | None:
    if not os.path.exists(checkpoint_path):
        return None
    best_model = PinnModel.load_checkpoint(
        checkpoint_path,
        device_preference=str(cfg_get(config, "pinn.device", "auto")),
    )
    payload = {
        "checkpoint_path": checkpoint_path,
        "checkpoint_tag": "best",
        "train": {
            "n_rows": _split_row_count(dataset.train_x),
            "metrics": regression_metrics_for_torch_model(
                model=best_model.model,
                x=dataset.train_x,
                y=dataset.train_y,
                batch_size=batch_size,
            ),
        },
        "val": {
            "n_rows": _split_row_count(dataset.val_x),
            "metrics": regression_metrics_for_torch_model(
                model=best_model.model,
                x=dataset.val_x,
                y=dataset.val_y,
                batch_size=batch_size,
            ),
        },
        "test": {
            "n_rows": _split_row_count(dataset.test_x),
            "metrics": regression_metrics_for_torch_model(
                model=best_model.model,
                x=dataset.test_x,
                y=dataset.test_y,
                batch_size=batch_size,
            ),
        },
        "evaluation_sets": _evaluate_external_sets(
            config=config,
            original_cwd=original_cwd,
            model=best_model.model,
            batch_size=batch_size,
        ),
    }
    try:
        checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception:
        checkpoint_payload = {}
    if isinstance(checkpoint_payload, dict):
        metrics = checkpoint_payload.get("metrics")
        if metrics is not None and hasattr(metrics, "as_flat_dict"):
            row = metrics.as_flat_dict()
            payload["selection"] = {
                "global_epoch": row.get("global_epoch"),
                "phase_name": row.get("phase_name"),
                "cumulative_wall_seconds": row.get("cumulative_wall_seconds"),
                "val_total_loss": row.get("val_total_loss"),
                "train_total_loss": row.get("train_total_loss"),
                "test_data_loss": row.get("test_data_loss"),
            }
    return payload


@hydra.main(config_path="src/config", config_name="setup_pinn", version_base=None)
def main(config) -> None:
    started_at_utc = utc_now_iso()
    total_started = monotonic()
    original_cwd = get_original_cwd()
    dataset_root = resolve_dataset_root(config=config, original_cwd=original_cwd)
    run_dir = resolve_run_dir(config=config, section="pinn", original_cwd=original_cwd)
    os.makedirs(run_dir, exist_ok=True)
    config_path = os.path.join(run_dir, "config.yaml")
    metrics_json_path = os.path.join(run_dir, "metrics.json")
    epoch_metrics_csv_path = os.path.join(run_dir, "epoch_metrics.csv")
    timings_path = os.path.join(run_dir, "timings.json")
    manifest_path = os.path.join(run_dir, "run_manifest.json")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    checkpointing_enabled = bool(cfg_get(config, "pinn.checkpointing.enabled", True))
    timings = {
        "dataset_load_seconds": None,
        "training_seconds": None,
        "artifact_write_seconds": None,
        "total_seconds": None,
    }
    status = "failed"
    error_message: str | None = None
    print(f"[pinn] Loading dataset from root: {dataset_root}")

    logger = PinnLogger(run_dir=run_dir, config=config)
    try:
        dataset_notes = validate_pinn_runtime_dataset(
            dataset_root,
            collocation_mode=str(getattr(config.pinn.collocation, "mode", "preprocessed")),
            curriculum_enabled=bool(cfg_get(config, "pinn.curriculum.enabled", False)),
            supervised_acquisition_enabled=bool(cfg_get(config, "pinn.supervised_acquisition.enabled", False)),
        )
        for note in dataset_notes:
            print(f"[pinn] Note: {note}")
        dataset_load_started = monotonic()
        dataset = load_pinn_dataset_from_preprocessed_root(
            dataset_root=dataset_root,
            dtype=str(config.pinn.dtype),
            allow_missing_collocation=str(getattr(config.pinn.collocation, "mode", "preprocessed")).strip().lower() == "generated",
            supervised_acquisition_enabled=bool(cfg_get(config, "pinn.supervised_acquisition.enabled", False)),
        )
        timings["dataset_load_seconds"] = monotonic() - dataset_load_started
        active_col_points = int(cfg_get(config, "pinn.collocation.active_points", dataset.train_col_x.shape[0]))
        val_col_rows = 0 if dataset.val_col_x is None else dataset.val_col_x.shape[0]
        val_col_active = min(val_col_rows, active_col_points) if val_col_rows > 0 else 0
        print(
            "[pinn] Dataset loaded | "
            f"train_rows={dataset.train_x.shape[0]} "
            f"col_rows={dataset.train_col_x.shape[0]} (active={active_col_points}) "
            f"init_rows={dataset.train_init_x.shape[0]} "
            f"val_rows={0 if dataset.val_x is None else dataset.val_x.shape[0]} "
            f"val_col_rows={val_col_rows} (active={val_col_active}) "
            f"val_init_rows={0 if dataset.val_init_x is None else dataset.val_init_x.shape[0]} "
            f"test_rows={0 if dataset.test_x is None else dataset.test_x.shape[0]}"
        )
        if bool(cfg_get(config, "debug.print_io_example", False)):
            _print_pinn_io_example(config, dataset)

        ode_model = SynchronousMachineModels(config)
        save_resolved_config(config=config, run_dir=run_dir)
        train_started = monotonic()
        pinn_model, rows = train_pinn(dataset=dataset, ode_model=ode_model, config=config, logger=logger)
        timings["training_seconds"] = monotonic() - train_started
        artifact_write_started = monotonic()
        metric_batch_size = int(cfg_get(config, "pinn.default_batch_size", 1024))
        run_summary = build_run_summary(
            run_type="pinn",
            model_flag=str(config.model.model_flag),
            dataset_root=dataset_root,
            seed=int(config.model.seed),
            dataset_number=(
                int(config.dataset.number) if cfg_get(config, "dataset.root", None) in (None, "") else None
            ),
            split_sizes={
                "train_supervised_rows": int(dataset.train_x.shape[0]),
                "val_supervised_rows": None if dataset.val_x is None else int(dataset.val_x.shape[0]),
                "test_supervised_rows": None if dataset.test_x is None else int(dataset.test_x.shape[0]),
                "train_collocation_rows": int(dataset.train_col_x.shape[0]),
                "train_init_rows": int(dataset.train_init_x.shape[0]),
            },
        )
        final_train_metrics = regression_metrics_for_torch_model(
            model=pinn_model.model,
            x=dataset.train_x,
            y=dataset.train_y,
            batch_size=metric_batch_size,
        )
        final_val_metrics = regression_metrics_for_torch_model(
            model=pinn_model.model,
            x=dataset.val_x,
            y=dataset.val_y,
            batch_size=metric_batch_size,
        )
        final_test_metrics = regression_metrics_for_torch_model(
            model=pinn_model.model,
            x=dataset.test_x,
            y=dataset.test_y,
            batch_size=metric_batch_size,
        )
        evaluation_sets = _evaluate_external_sets(
            config=config,
            original_cwd=original_cwd,
            model=pinn_model.model,
            batch_size=metric_batch_size,
        )
        best_checkpoint_metrics = _evaluate_best_checkpoint(
            checkpoint_path=os.path.join(logger.ckpt_dir, "best.pt"),
            config=config,
            original_cwd=original_cwd,
            dataset=dataset,
            batch_size=metric_batch_size,
        )
        logger.write_metrics_json(
            rows,
            run_summary=run_summary,
            final_train_metrics=final_train_metrics,
            final_val_metrics=final_val_metrics,
            final_test_metrics=final_test_metrics,
            evaluation_sets=evaluation_sets,
            best_checkpoint_metrics=best_checkpoint_metrics,
        )
        timings["artifact_write_seconds"] = monotonic() - artifact_write_started
        status = "completed"
        print(f"[pinn] Training complete. epochs={len(rows)} run_dir={run_dir}")
        if checkpointing_enabled and os.path.isdir(logger.ckpt_dir):
            print(f"[pinn] Checkpoints written to: {logger.ckpt_dir}")
        else:
            print("[pinn] Checkpointing disabled; no checkpoints written.")
    except Exception as exc:
        error_message = str(exc)
        raise
    finally:
        timings["total_seconds"] = monotonic() - total_started
        write_json(timings_path, timings)
        write_json(
            manifest_path,
            build_direct_run_manifest(
                run_type="pinn",
                entrypoint="20_run_pinn.py",
                run_dir=run_dir,
                dataset_root=dataset_root,
                model_flag=str(config.model.model_flag),
                seed=int(config.model.seed),
                started_at_utc=started_at_utc,
                completed_at_utc=utc_now_iso(),
                status=status,
                error=error_message,
                timings_path=timings_path,
                artifacts={
                    "config": config_path,
                    "metrics": metrics_json_path if os.path.exists(metrics_json_path) else None,
                    "epoch_metrics": epoch_metrics_csv_path if os.path.exists(epoch_metrics_csv_path) else None,
                    "timings": timings_path,
                    "checkpoints_dir": ckpt_dir if os.path.isdir(ckpt_dir) else None,
                },
                extra={
                    "wandb_enabled": bool(cfg_get(config, "wandb.use", False)),
                },
            ),
        )
        logger.finish()


if __name__ == "__main__":
    main()
