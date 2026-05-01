"""Train/evaluate a fixed baseline surrogate from preprocessed dataset outputs."""

from __future__ import annotations

import os
from time import monotonic

import hydra
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf

from src.data.loaders.preprocessed_trajectory_loader import load_trajectory_dataset_from_preprocessed_root
from src.data.runtime_validation import validate_baseline_runtime_dataset
from src.training.baseline import BaselineConfig, evaluate_baseline, train_baseline_surrogate
from src.training.metrics import build_run_summary, regression_metrics_from_arrays
from src.training.runtime import (
    build_direct_run_manifest,
    cfg_get,
    resolve_dataset_root,
    resolve_run_dir,
    save_resolved_config,
    utc_now_iso,
    write_json,
)


def _format_example_values(values, *, max_items: int = 8) -> list[float]:
    return [round(float(value), 6) for value in values[:max_items]]


def _model_io_names(config) -> tuple[list[str], list[str]]:
    guide_path = os.path.join(str(config.dirs.init_conditions_dir), "modellings_guide.yaml")
    guide = OmegaConf.load(guide_path)
    for model in guide:
        if str(model.get("name")) == str(config.model.model_flag):
            state_names = [str(value) for value in (model.get("keys") or [])]
            extra_names = [str(value) for value in (model.get("keys_ext") or [])]
            return state_names + extra_names, state_names
    return [], []


def _print_baseline_io_example(config, dataset) -> None:
    input_names, output_state_names = _model_io_names(config)
    train_x, train_y = dataset.training_view()
    print("[baseline] Input/output definition:")
    print(f"[baseline] input = initial-condition/features {input_names}")
    print(f"[baseline] output = full trajectory over time for states {output_state_names}")
    print(
        "[baseline] example train input | "
        f"shape={train_x[0].shape} values={_format_example_values(train_x[0])}"
    )
    print(
        "[baseline] example train output | "
        f"trajectory_shape={train_y[0].shape} flattened_dim={train_y[0].reshape(-1).shape[0]}"
    )
    if dataset.time_grid is not None and train_y.shape[1] > 0:
        print(
            "[baseline] example output first step | "
            f"t={round(float(dataset.time_grid[0]), 6)} values={_format_example_values(train_y[0, 0])}"
        )
        print(
            "[baseline] example output last step | "
            f"t={round(float(dataset.time_grid[-1]), 6)} values={_format_example_values(train_y[0, -1])}"
        )


@hydra.main(config_path="src/config", config_name="setup_baseline", version_base=None)
def main(config) -> None:
    started_at_utc = utc_now_iso()
    total_started = monotonic()
    original_cwd = get_original_cwd()
    dataset_root = resolve_dataset_root(config=config, original_cwd=original_cwd)
    run_dir = resolve_run_dir(config=config, section="baseline", original_cwd=original_cwd)
    os.makedirs(run_dir, exist_ok=True)
    config_path = os.path.join(run_dir, "config.yaml")
    metrics_path = os.path.join(run_dir, "metrics.json")
    timings_path = os.path.join(run_dir, "timings.json")
    manifest_path = os.path.join(run_dir, "run_manifest.json")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    ckpt_name = str(cfg_get(config, "baseline.checkpoint_name", "baseline_model.pt"))
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    checkpointing_enabled = bool(cfg_get(config, "baseline.checkpointing.enabled", True))
    timings = {
        "dataset_load_seconds": None,
        "training_seconds": None,
        "evaluation_seconds": None,
        "artifact_write_seconds": None,
        "total_seconds": None,
    }
    status = "failed"
    error_message: str | None = None

    print(f"[baseline] Loading preprocessed dataset from root: {dataset_root}")
    try:
        validate_baseline_runtime_dataset(dataset_root)
        dataset_load_started = monotonic()
        dataset = load_trajectory_dataset_from_preprocessed_root(
            dataset_root=dataset_root,
            include_val_in_train=bool(cfg_get(config, "dataset.include_val_in_train", False)),
        )
        timings["dataset_load_seconds"] = monotonic() - dataset_load_started
        print(f"[baseline] Dataset loaded | n_train={dataset.n_train} n_test={dataset.n_test}")
        if bool(cfg_get(config, "debug.print_io_example", False)):
            _print_baseline_io_example(config, dataset)

        bcfg = BaselineConfig(
            hidden_dim=int(config.baseline.hidden_dim),
            hidden_layers=int(config.baseline.hidden_layers),
            dropout=float(config.baseline.dropout),
            lr=float(config.baseline.lr),
            batch_size=int(config.baseline.batch_size),
            epochs=int(config.baseline.epochs),
            device=str(getattr(config.baseline, "device", "auto")),
        )
        print(
            "[baseline] Training baseline | "
            f"seed={int(config.model.seed)} hidden_dim={bcfg.hidden_dim} hidden_layers={bcfg.hidden_layers} "
            f"batch_size={bcfg.batch_size} epochs={bcfg.epochs} lr={bcfg.lr}"
        )

        save_resolved_config(config=config, run_dir=run_dir)
        train_started = monotonic()
        model = train_baseline_surrogate(dataset=dataset, seed=int(config.model.seed), cfg=bcfg)
        timings["training_seconds"] = monotonic() - train_started
        if bool(cfg_get(config, "debug.print_architecture", False)):
            n_params = sum(p.numel() for p in model.model.parameters())
            n_trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
            print("[baseline] Architecture:")
            print(model.model)
            print(
                "[baseline] Architecture summary | "
                f"input_dim={model.input_dim} output_dim={model.output_dim} "
                f"hidden_dim={model.hidden_dim} hidden_layers={model.hidden_layers} dropout={bcfg.dropout} "
                f"device={model.device} parameters={n_params} trainable_parameters={n_trainable}"
            )

        eval_started = monotonic()
        metrics = evaluate_baseline(model=model, dataset=dataset)
        timings["evaluation_seconds"] = monotonic() - eval_started
        print("[baseline] Baseline training and evaluation finished.")

        artifact_write_started = monotonic()
        if checkpointing_enabled:
            os.makedirs(ckpt_dir, exist_ok=True)
            model.save_checkpoint(ckpt_path)

        train_x, train_y = dataset.training_view()
        test_x, test_y = dataset.test_view()
        train_metrics = regression_metrics_from_arrays(
            y_true=train_y,
            y_pred=model.predict(train_x),
        )
        test_metrics = None
        if test_x is not None and test_y is not None:
            test_metrics = regression_metrics_from_arrays(
                y_true=test_y,
                y_pred=model.predict(test_x),
            )
        legacy_test_metrics = {} if test_metrics is None else dict(test_metrics)

        evaluation_sets: dict | None = None
        for eval_kind in ("id", "ood"):
            eval_root = cfg_get(config, f"evaluation.{eval_kind}.root", None)
            if not eval_root:
                continue
            eval_root = str(eval_root)
            print(f"[baseline] Evaluating external {eval_kind} dataset: {eval_root}")
            ext_dataset = load_trajectory_dataset_from_preprocessed_root(dataset_root=eval_root)
            ext_x, ext_y = ext_dataset.test_view()
            if ext_x is None or ext_y is None:
                print(f"[baseline] Warning: external {eval_kind} dataset has no test split; skipping.")
                continue
            ext_metrics = regression_metrics_from_arrays(y_true=ext_y, y_pred=model.predict(ext_x))
            if evaluation_sets is None:
                evaluation_sets = {}
            evaluation_sets[eval_kind] = {
                "dataset_root": eval_root,
                "n_rows": int(ext_x.shape[0]),
                "metrics": ext_metrics,
            }

        write_json(
            metrics_path,
            {
                "schema_version": "run_metrics_v1",
                "model_flag": str(config.model.model_flag),
                "dataset_number": int(config.dataset.number) if cfg_get(config, "dataset.root", None) in (None, "") else None,
                "dataset_root": dataset_root,
                "n_train": dataset.n_train,
                "n_test": dataset.n_test,
                "run_summary": build_run_summary(
                    run_type="baseline",
                    model_flag=str(config.model.model_flag),
                    dataset_root=dataset_root,
                    seed=int(config.model.seed),
                    dataset_number=(
                        int(config.dataset.number) if cfg_get(config, "dataset.root", None) in (None, "") else None
                    ),
                    split_sizes={
                        "train_trajectories": int(dataset.n_train),
                        "val_trajectories": None,
                        "test_trajectories": int(dataset.n_test),
                    },
                ),
                "final_train_metrics": train_metrics,
                "final_val_metrics": None,
                "final_test_metrics": test_metrics,
                "final_train_losses": None,
                "final_val_losses": None,
                "final_test_losses": None,
                "evaluation_sets": evaluation_sets,
            },
        )
        timings["artifact_write_seconds"] = monotonic() - artifact_write_started
        status = "completed"

        if checkpointing_enabled and os.path.exists(ckpt_path):
            print(f"Baseline checkpoint: {ckpt_path}")
        else:
            print("[baseline] Checkpointing disabled; no checkpoint written.")
        print(f"Baseline metrics: {legacy_test_metrics}")
    except Exception as exc:
        error_message = str(exc)
        raise
    finally:
        timings["total_seconds"] = monotonic() - total_started
        write_json(timings_path, timings)
        write_json(
            manifest_path,
            build_direct_run_manifest(
                run_type="baseline",
                entrypoint="10_run_baseline.py",
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
                    "metrics": metrics_path,
                    "timings": timings_path,
                    "checkpoints_dir": ckpt_dir if os.path.isdir(ckpt_dir) else None,
                    "checkpoint": ckpt_path if os.path.exists(ckpt_path) else None,
                },
                extra={
                    "wandb_enabled": bool(cfg_get(config, "wandb.use", False)),
                },
            ),
        )


if __name__ == "__main__":
    main()
