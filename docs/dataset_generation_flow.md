# Dataset Generation Flow

This document describes the current stage-1 dataset generation process (`raw/file*.pkl` + `info.txt`).

## Entry Command

Run stage-1 with:

```bash
PYTHONPATH=. python 00_create_dataset.py preset=<preset> +method=<method>
```

Methods:
- `lhs_static`
- `qbc_deep_ensemble`
- `marker_directed`
- `qbc_marker_hybrid`

Presets live in `src/config/preset/` and provide runtime/output defaults.

## Main Modules

- Entry + dispatch: `00_create_dataset.py`
- Orchestration + builder + writer: `src/data/generate/dataset_functions.py`
- QBC + marker loops: `src/methods/loop.py`
- Acquisition utilities (regular + hybrid): `src/methods/acquisition.py`
- Marker features/selection utils: `src/methods/marker_utils.py`
- Adaptive logging/checkpoints: `src/methods/logger.py`
- ODE simulation utility: `src/sim/simulator.py`
- Adaptive metadata assembly: `src/data/generate/adaptive_metadata.py`
- Dataset state container: `src/data/loaders/trajectory_dataset.py`

## Functional Stages

1. Hydra config composition (`setup_dataset` + `preset` + `method`).
2. Method resolution + validation.
3. IC generation:
   - static: full-factorial or joint LHS table
   - adaptive: initial pool + candidate pools per round
4. ODE simulation to trajectories.
5. Acquisition (adaptive only): QBC/marker/hybrid selection of new ICs.
6. Dataset state updates (`append`, train/test views).
7. Optional adaptive logging/checkpointing.
8. Metadata creation for `info.txt`.
9. Raw contract writing:
   - `.../dataset_vN/raw/file*.pkl`
   - `.../dataset_vN/info.txt`

## Flowchart

```mermaid
flowchart TD
    A["CLI: 00_create_dataset.py preset=... +method=..."] --> B["Hydra compose config"]
    B --> C["main() in 00_create_dataset.py"]
    C --> D["ODETrajectoryBuilder(config)"]
    C --> E["resolve_method(config)"]

    E --> F{"method"}
    F -->|lhs_static| G["run_lhs_static(...)"]
    F -->|qbc_deep_ensemble| H["run_qbc_deep_ensemble(...)"]
    F -->|marker_directed| I["run_marker_directed(...)"]
    F -->|qbc_marker_hybrid| J["run_qbc_marker_hybrid(...)"]

    G --> G1["build_initial_conditions()"]
    G1 --> G2["solve_sm_model()"]
    G2 --> G3["save_dataset() -> raw/file*.pkl + info.txt"]

    H --> K["build adaptive config + logger"]
    I --> K
    J --> K

    K --> L["init/resume TrajectoryDataset"]
    L --> M{"acquisition loop"}
    M -->|QBC / Hybrid| N["run_qbc_loop() in loop.py"]
    M -->|Marker| O["run_marker_loop() in loop.py"]

    N --> P["selected ICs -> simulate_trajectory() -> dataset.append()"]
    O --> P

    P --> Q["build adaptive metadata"]
    Q --> R["save_dataset_from_arrays() -> raw/file*.pkl + info.txt"]
```

## Notes

- Adaptive round logs are throttled via `active.log_every` (default: `5`).
- The final output contract is identical across methods (`raw/` + `info.txt`).
