"""Generate raw ODE trajectory datasets.

This is the canonical stage-1 entrypoint for dataset generation. Hydra composes
`src/config/setup_dataset.yaml`; method-specific dataset generation is selected
with overrides such as `+method=qbc_deep_ensemble`.
"""

from __future__ import annotations

import hydra

from src.data.active_learning.orchestration import resolve_method, run_method
from src.data.generate.dataset_functions import ODETrajectoryBuilder


@hydra.main(config_path="src/config", config_name="setup_dataset", version_base=None)
def main(config) -> None:
    dataset_builder = ODETrajectoryBuilder(config)
    dataset_builder.create_init_conditions_info()

    # Keep method-specific logic in src.data.active_learning; this file only
    # composes the Hydra config and dispatches the selected dataset generator.
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
