"""Stage-2 entrypoint: preprocess raw datasets into HDF5 training data."""

from __future__ import annotations

import hydra

from src.nn.dataset_pre import Datapreprocessor


@hydra.main(config_path="src/conf", config_name="setup_dataset_nn", version_base=None)
def main(config) -> None:
    datapreprocessor = Datapreprocessor(config)
    datapreprocessor.get_preprocess_save_data()
    datapreprocessor.create_save_col_data()
    datapreprocessor.update_info_file()


if __name__ == "__main__":
    main()
