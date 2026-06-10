"""Preprocess raw ODE datasets into PINN-ready training data.

This is the canonical stage-2 entrypoint after `00_create_dataset.py`. Hydra
composes `src/config/setup_dataset_nn.yaml` and writes the supervised,
collocation, and initial-condition arrays used by `20_run_pinn.py`.
"""

from __future__ import annotations

import hydra

from src.data.preprocess.dataset_pre import Datapreprocessor


@hydra.main(config_path="src/config", config_name="setup_dataset_nn", version_base=None)
def main(config) -> None:
    print(
        "[stage-2] Starting preprocess | "
        f"model={config.model.model_flag} dataset_root={getattr(config.dataset, 'root', None)} "
        f"dataset_number={getattr(config.dataset, 'number', None)}"
    )
    datapreprocessor = Datapreprocessor(config)
    # These calls write the supervised splits, collocation data, and metadata
    # expected by the PINN training entrypoint.
    datapreprocessor.get_preprocess_save_data()
    datapreprocessor.create_save_col_data()
    datapreprocessor.update_info_file()
    print("[stage-2] Preprocess completed.")


if __name__ == "__main__":
    main()
