"""Stage-2 entrypoint: preprocess raw datasets into HDF5 training data."""

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
    datapreprocessor.get_preprocess_save_data()
    datapreprocessor.create_save_col_data()
    datapreprocessor.update_info_file()
    print("[stage-2] Preprocess completed.")


if __name__ == "__main__":
    main()
