# Results Notebooks

This folder contains the thesis-facing analysis notebooks and curated table exports.

Run notebooks from the repository root so their project-root discovery and relative paths resolve consistently.

## Notebook Sequence

- `00_dynamical_system_analysis.ipynb`: Chapter 4 dynamical-system analysis and methodology checks using the canonical ODE implementation in `src/sim/ode/model_definitions.py` and canonical configuration files under `src/config/`.
- `01_dataset_generation.ipynb`: Dataset-generation comparison.
- `02_optimizer_comparison.ipynb`: Optimizer comparison.
- `03_loss_balancing.ipynb`: Loss-balancing comparison.
- `04_collocation_comparison.ipynb`: Collocation-sampling comparison.
- `05_multistage.ipynb`: Multistage training comparison.
- `06_data_augmentation.ipynb`: Adaptive data-augmentation comparison.
- `07_final_experiment.ipynb`: Final experiment analysis.
- `08_across_experiments.ipynb`: Cross-experiment synthesis.

Curated CSV outputs are written under `results/tables/`. Scratch exports to `results/data/` are disabled for repository submission.
