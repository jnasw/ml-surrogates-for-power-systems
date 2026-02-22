"""Train/evaluate a fixed baseline surrogate on dataset_vN."""

from __future__ import annotations

import json
import os

import hydra
from hydra.utils import get_original_cwd

from src.data.preprocessed_trajectory_loader import load_trajectory_dataset_from_preprocessed
from src.surrogate.baseline import BaselineConfig, evaluate_baseline, train_baseline_surrogate


@hydra.main(config_path="src/conf", config_name="setup_baseline", version_base=None)
def main(config) -> None:
    dataset = load_trajectory_dataset_from_preprocessed(
        dataset_dir=str(config.dirs.dataset_dir),
        model_flag=str(config.model.model_flag),
        dataset_number=int(config.dataset.number),
        include_val_in_train=bool(getattr(config.dataset, "include_val_in_train", False)),
    )

    bcfg = BaselineConfig(
        hidden_dim=int(config.baseline.hidden_dim),
        hidden_layers=int(config.baseline.hidden_layers),
        dropout=float(config.baseline.dropout),
        lr=float(config.baseline.lr),
        batch_size=int(config.baseline.batch_size),
        epochs=int(config.baseline.epochs),
        device=str(getattr(config.baseline, "device", "auto")),
    )
    model = train_baseline_surrogate(dataset=dataset, seed=int(config.model.seed), cfg=bcfg)
    metrics = evaluate_baseline(model=model, dataset=dataset)

    save_dir_cfg = str(config.baseline.save_dir)
    save_dir = save_dir_cfg if os.path.isabs(save_dir_cfg) else os.path.join(get_original_cwd(), save_dir_cfg)
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, str(config.baseline.save_name))
    model.save_checkpoint(ckpt_path)

    metrics_path = os.path.join(save_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_flag": str(config.model.model_flag),
                "dataset_number": int(config.dataset.number),
                "n_train": dataset.n_train,
                "n_test": dataset.n_test,
                **metrics,
            },
            f,
            indent=2,
        )

    print(f"Baseline checkpoint: {ckpt_path}")
    print(f"Baseline metrics: {metrics}")


if __name__ == "__main__":
    main()
