"""Train/evaluate a fixed baseline surrogate from preprocessed dataset outputs."""

from __future__ import annotations

import os

import hydra
from hydra.utils import get_original_cwd

from src.data.loaders.preprocessed_trajectory_loader import load_trajectory_dataset_from_preprocessed_root
from src.train.baseline import BaselineConfig, evaluate_baseline, train_baseline_surrogate
from src.train.runtime import cfg_get, resolve_dataset_root, resolve_run_dir, save_resolved_config, write_json


@hydra.main(config_path="src/config", config_name="setup_baseline", version_base=None)
def main(config) -> None:
    original_cwd = get_original_cwd()
    dataset_root = resolve_dataset_root(config=config, original_cwd=original_cwd)
    print(f"[baseline] Loading preprocessed dataset from root: {dataset_root}")
    dataset = load_trajectory_dataset_from_preprocessed_root(
        dataset_root=dataset_root,
        include_val_in_train=bool(cfg_get(config, "dataset.include_val_in_train", False)),
    )
    print(f"[baseline] Dataset loaded | n_train={dataset.n_train} n_test={dataset.n_test}")

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
    run_dir = resolve_run_dir(config=config, section="baseline", original_cwd=original_cwd, legacy_key="baseline.save_dir")
    save_resolved_config(config=config, run_dir=run_dir)

    model = train_baseline_surrogate(dataset=dataset, seed=int(config.model.seed), cfg=bcfg)
    metrics = evaluate_baseline(model=model, dataset=dataset)
    print("[baseline] Baseline training and evaluation finished.")

    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_name = str(cfg_get(config, "baseline.checkpoint_name", cfg_get(config, "baseline.save_name", "baseline_model.pt")))
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    model.save_checkpoint(ckpt_path)

    metrics_path = os.path.join(run_dir, "metrics.json")
    write_json(
        metrics_path,
        {
            "model_flag": str(config.model.model_flag),
            "dataset_number": int(config.dataset.number) if cfg_get(config, "dataset.root", None) in (None, "") else None,
            "dataset_root": dataset_root,
            "n_train": dataset.n_train,
            "n_test": dataset.n_test,
            **metrics,
        },
    )

    print(f"Baseline checkpoint: {ckpt_path}")
    print(f"Baseline metrics: {metrics}")


if __name__ == "__main__":
    main()
