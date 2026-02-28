"""Stage-1 entrypoint: generate raw ODE simulation datasets."""

from __future__ import annotations

import hydra

from src.data.generate.dataset_functions import ODETrajectoryBuilder
from src.data.generate.dataset_functions import resolve_method, run_method


@hydra.main(config_path="src/config", config_name="setup_dataset", version_base=None)
def main(config) -> None:
    dataset_builder = ODETrajectoryBuilder(config)
    dataset_builder.create_init_conditions_info()

    method_name = resolve_method(config)
    print(
        "[stage-1] Starting dataset generation | "
        f"method={method_name} model={config.model.model_flag} "
        f"ic_generation_method={dataset_builder.ic_generation_method} "
        f"seed={getattr(config.model, 'seed', None)}"
    )

    run_method(config, dataset_builder, method_name)


if __name__ == "__main__":
    main()
