"""Train a phase-1 PINN from preprocessed stage-2 outputs."""

from __future__ import annotations

import hydra
from hydra.utils import get_original_cwd

from src.pinn.data import load_pinn_dataset_from_preprocessed_root
from src.pinn.logging import PinnLogger
from src.sim.ode.model_definitions import SynchronousMachineModels
from src.train.runtime import resolve_dataset_root, resolve_run_dir, save_resolved_config
from src.train.trainer import train_pinn


@hydra.main(config_path="src/config", config_name="setup_pinn", version_base=None)
def main(config) -> None:
    original_cwd = get_original_cwd()
    dataset_root = resolve_dataset_root(config=config, original_cwd=original_cwd)
    print(f"[pinn] Loading dataset from root: {dataset_root}")

    dataset = load_pinn_dataset_from_preprocessed_root(
        dataset_root=dataset_root,
        dtype=str(config.pinn.dtype),
        allow_missing_collocation=str(getattr(config.pinn.collocation, "mode", "preprocessed")).strip().lower() == "generated",
    )
    print(
        "[pinn] Dataset loaded | "
        f"train_rows={dataset.train_x.shape[0]} "
        f"col_rows={dataset.train_col_x.shape[0]} "
        f"init_rows={dataset.train_init_x.shape[0]} "
        f"val_rows={0 if dataset.val_x is None else dataset.val_x.shape[0]} "
        f"test_rows={0 if dataset.test_x is None else dataset.test_x.shape[0]}"
    )

    ode_model = SynchronousMachineModels(config)
    run_dir = resolve_run_dir(config=config, section="pinn", original_cwd=original_cwd)
    logger = PinnLogger(run_dir=run_dir, config=config)
    try:
        save_resolved_config(config=config, run_dir=run_dir)
        _model, rows = train_pinn(dataset=dataset, ode_model=ode_model, config=config, logger=logger)
        print(f"[pinn] Training complete. epochs={len(rows)} run_dir={run_dir}")
        print(f"[pinn] Best/last checkpoints written to: {logger.ckpt_dir}")
    finally:
        logger.finish()


if __name__ == "__main__":
    main()
