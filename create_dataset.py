"""Stage-1 entrypoint: generate raw ODE simulation datasets."""

from __future__ import annotations

import hydra

from src.dataset.create_dataset_functions import ODEModelling
from src.ode.model_definitions import SynchronousMachineModels


@hydra.main(config_path="src/conf", config_name="setup_dataset", version_base=None)
def main(config) -> None:
    modelling_full = SynchronousMachineModels(config)
    dataset_builder = ODEModelling(config)

    init_conditions = dataset_builder.build_initial_conditions()
    print(f"Generated {len(init_conditions)} initial conditions using '{dataset_builder.ic_generation_method}'.")

    dataset_builder.solve_sm_model(init_conditions, modelling_full, flag_time=False)


if __name__ == "__main__":
    main()
